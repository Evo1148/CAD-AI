from __future__ import annotations

import argparse
import math
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from .benchmark.m3 import DEFAULT_M3B_MODEL_ID, m3b_http_config, m3b_llm_config
from .cad import CADStatus
from .cases import lab001_case
from .contracts import StrictModel
from .pipeline import run_pipeline
from .planning import (
    BoundaryStatus,
    ErrorPlannerResponse,
    HTTPInferenceBackend,
    LLMRepairPlanner,
    PlannerErrorCode,
    PlannerRun,
    RepairPlanner,
    RepairPlannerResponse,
    RepairPlanningBoundary,
    RepairPlanningContext,
    RepairPlanningContextBuilder,
    UnsupportedPlannerResponse,
)
from .revision import (
    RepairExecutor,
    RepairPlanValidation,
    RepairPlanValidator,
    RepairRule,
    RepairStatus,
    changed_semantic_plan_paths,
    preserved_passing_constraints,
)
from .validation import CheckStatus, ReportStatus, ValidationReport


LAB001_RULE = RepairRule(
    constraint_id="C_HOLE_DIAMETER",
    operation_id="OP02",
    parameter_name="diameter",
    related_feature_ids=["mount_hole"],
)
LAB001_EXPECTED_CHANGED_PATHS = {"OP02.params.diameter"}


class LabStatus(str, Enum):
    VALIDATED = "VALIDATED"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"


class LabResult(StrictModel):
    lab_id: Literal["LAB-001"] = "LAB-001"
    status: LabStatus
    stage: str
    reason_code: str | None = None
    message: str | None = None
    design_id: str = "LAB-001"
    spec_version: str = "1.0"
    r01_cad_status: str | None = None
    r01_validation_report: ValidationReport | None = None
    r01_blocking_failures: list[str] = Field(default_factory=list)
    planning_context: RepairPlanningContext | None = None
    planner_response: RepairPlannerResponse | None = None
    planner_run: PlannerRun | None = None
    boundary_status: BoundaryStatus | None = None
    repair_plan_validation: RepairPlanValidation | None = None
    repair_executor_status: RepairStatus | None = None
    r02_cad_status: str | None = None
    r02_revision_id: str | None = None
    r02_parent_revision_id: str | None = None
    r02_validation_report: ValidationReport | None = None
    changed_semantic_paths: list[str] = Field(default_factory=list)
    change_locality: bool = False
    preserved_constraint_ids: list[str] = Field(default_factory=list)
    constraint_preservation: bool = False
    feature_ids_r01: list[str] = Field(default_factory=list)
    feature_ids_r02: list[str] = Field(default_factory=list)
    feature_identity_preserved: bool = False
    artifacts: dict[str, dict[str, str]] = Field(default_factory=dict)


def run_lab001(
    planner: RepairPlanner,
    output: Path,
    result_json: Path | None = None,
) -> LabResult:
    """Execute LAB-001 through the M3 trust boundary without fallback."""

    spec, r01_plan = lab001_case()
    spec_snapshot = spec.model_dump(mode="json")
    plan_snapshot = r01_plan.model_dump(mode="json")
    state: dict[str, Any] = {
        "stage": "R01_CAD",
        "design_id": spec.design_id,
        "spec_version": spec.spec_version,
    }

    def finish(status: LabStatus, reason_code: str | None = None, message: str | None = None) -> LabResult:
        result = LabResult(status=status, reason_code=reason_code, message=message, **state)
        destination = result_json or output / "result.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return result

    try:
        r01 = run_pipeline(spec, r01_plan, output / "R01")
        state["r01_cad_status"] = r01.cad_result.status.value
        state["artifacts"] = {"R01": _artifact_paths(r01.artifacts)}
        if r01.cad_result.status != CADStatus.SUCCESS or r01.validation_report is None:
            return finish(LabStatus.FAIL, "R01_CAD_FAILED", _cad_error_message(r01.cad_result.error))

        r01_report = r01.validation_report
        state["stage"] = "R01_VALIDATION"
        state["r01_validation_report"] = r01_report
        blocking_failures = [
            check for check in r01_report.checks
            if check.blocking and check.status == CheckStatus.FAIL
        ]
        state["r01_blocking_failures"] = [check.constraint_id for check in blocking_failures if check.constraint_id]
        if not _is_expected_r01_failure(r01_report, blocking_failures):
            return finish(
                LabStatus.FAIL,
                "UNEXPECTED_R01_VALIDATION",
                "R01 must contain only C_HOLE_DIAMETER FAIL with actual 5.4 and target 6.0 ± 0.01",
            )

        state["feature_ids_r01"] = sorted(r01.cad_result.feature_registry.features)
        state["stage"] = "REPAIR_PLANNING"
        context = RepairPlanningContextBuilder().build(
            spec,
            r01_plan,
            r01_report,
            [LAB001_RULE],
        )
        state["planning_context"] = context
        invocation = planner.plan(context)
        state["planner_response"] = invocation.response
        state["planner_run"] = invocation.planner_run

        state["stage"] = "REPAIR_BOUNDARY"
        boundary = RepairPlanningBoundary(RepairPlanValidator([LAB001_RULE]), RepairExecutor())
        boundary_result = boundary.process(invocation, spec, r01_plan, r01_report)
        state["boundary_status"] = boundary_result.status
        state["repair_plan_validation"] = boundary_result.repair_plan_validation
        if boundary_result.repair_result is not None:
            state["repair_executor_status"] = boundary_result.repair_result.status

        response = invocation.response.root
        if isinstance(response, ErrorPlannerResponse):
            status = (
                LabStatus.BLOCKED
                if response.error.code in {PlannerErrorCode.MODEL_UNAVAILABLE, PlannerErrorCode.MODEL_TIMEOUT}
                else LabStatus.FAIL
            )
            return finish(status, response.error.code.value, response.error.message)
        if isinstance(response, UnsupportedPlannerResponse):
            return finish(LabStatus.FAIL, f"UNSUPPORTED:{response.reason_code.value}", response.reason)
        if boundary_result.status != BoundaryStatus.APPLIED:
            errors = boundary_result.repair_plan_validation.errors if boundary_result.repair_plan_validation else []
            return finish(
                LabStatus.FAIL,
                boundary_result.status.value,
                ", ".join(error.code.value for error in errors) or "repair plan did not reach the executor",
            )

        repair = boundary_result.repair_result
        if repair is None or repair.status != RepairStatus.REPAIRED or repair.revised_plan is None:
            return finish(LabStatus.FAIL, "REPAIR_NOT_APPLIED", repair.result_code if repair else None)
        if spec.model_dump(mode="json") != spec_snapshot or r01_plan.model_dump(mode="json") != plan_snapshot:
            return finish(LabStatus.FAIL, "SOURCE_MUTATED", "DesignSpec or CADPlan R01 was modified in place")

        state["stage"] = "R02_CAD"
        r02_plan = repair.revised_plan
        state["r02_revision_id"] = r02_plan.revision_id
        state["r02_parent_revision_id"] = r02_plan.parent_revision_id
        r02 = run_pipeline(spec, r02_plan, output / "R02")
        state["r02_cad_status"] = r02.cad_result.status.value
        state["artifacts"]["R02"] = _artifact_paths(r02.artifacts)
        if r02.cad_result.status != CADStatus.SUCCESS or r02.validation_report is None:
            return finish(LabStatus.FAIL, "R02_CAD_FAILED", _cad_error_message(r02.cad_result.error))

        state["stage"] = "R02_VALIDATION"
        state["r02_validation_report"] = r02.validation_report
        changed_paths = changed_semantic_plan_paths(r01_plan, r02_plan)
        state["changed_semantic_paths"] = sorted(changed_paths)
        state["change_locality"] = changed_paths == LAB001_EXPECTED_CHANGED_PATHS
        preservation, preserved_ids = preserved_passing_constraints(r01_report, r02.validation_report)
        state["constraint_preservation"] = preservation
        state["preserved_constraint_ids"] = preserved_ids
        state["feature_ids_r02"] = sorted(r02.cad_result.feature_registry.features)
        state["feature_identity_preserved"] = state["feature_ids_r01"] == state["feature_ids_r02"]

        artifacts_valid = _artifacts_exist(state["artifacts"])
        validated = all((
            r02.validation_report.status == ReportStatus.PASS,
            r02_plan.revision_id == "R02",
            r02_plan.parent_revision_id == "R01",
            state["change_locality"],
            state["constraint_preservation"],
            state["feature_identity_preserved"],
            artifacts_valid,
        ))
        if not validated:
            return finish(
                LabStatus.FAIL,
                "LAB_ACCEPTANCE_FAILED",
                "R02 did not satisfy every validation, locality, preservation, identity and artifact criterion",
            )
        state["stage"] = "COMPLETED"
        return finish(LabStatus.VALIDATED)
    except Exception as exc:
        return finish(LabStatus.FAIL, "LAB_RUNTIME_ERROR", str(exc))


def _is_expected_r01_failure(report: ValidationReport, failures: list[Any]) -> bool:
    if report.status != ReportStatus.FAIL or len(failures) != 1:
        return False
    failure = failures[0]
    expected = failure.expected or {}
    return (
        failure.constraint_id == "C_HOLE_DIAMETER"
        and isinstance(failure.actual, (int, float))
        and math.isclose(float(failure.actual), 5.4, abs_tol=1e-9)
        and math.isclose(float(expected.get("target")), 6.0, abs_tol=1e-9)
        and math.isclose(float(expected.get("tolerance")), 0.01, abs_tol=1e-12)
    )


def _artifact_paths(artifacts: dict[str, Path]) -> dict[str, str]:
    return {kind: str(path.resolve()) for kind, path in artifacts.items()}


def _artifacts_exist(artifacts: dict[str, dict[str, str]]) -> bool:
    return all(
        set(artifacts.get(revision, {})) == {"STEP", "STL"}
        and all(Path(path).is_file() for path in artifacts[revision].values())
        for revision in ("R01", "R02")
    )


def _cad_error_message(error: Any) -> str | None:
    return error.message if error is not None else None


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CAD AI reproducible laboratory experiments")
    parser.add_argument("lab_id", choices=["LAB-001"])
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--model-id", default=DEFAULT_M3B_MODEL_ID)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/lab/LAB-001"))
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_cli_parser().parse_args()
    planner = LLMRepairPlanner(
        HTTPInferenceBackend(
            m3b_http_config(
                base_url=args.base_url,
                endpoint=args.endpoint,
                model_id=args.model_id,
                timeout_seconds=args.timeout_seconds,
            )
        ),
        m3b_llm_config(),
    )
    result = run_lab001(planner, args.output, args.result_json)
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(f"{result.lab_id}: {result.status.value} ({result.stage})")
        if result.reason_code:
            print(f"reason={result.reason_code}")
        for revision, artifacts in result.artifacts.items():
            for kind, path in artifacts.items():
                print(f"{revision} {kind}: {path}")
    return 0 if result.status == LabStatus.VALIDATED else (2 if result.status == LabStatus.BLOCKED else 1)


if __name__ == "__main__":
    raise SystemExit(main())
