from __future__ import annotations

import argparse
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import Field

from .benchmark.m3 import DEFAULT_M3B_MODEL_ID, m3b_http_config, m3b_llm_config
from .cad import CADStatus
from .contracts import CADPlan, StrictModel
from .generation import (
    CADPlanGenerationResult,
    DesignSpecGenerationResult,
    DeterministicIntentToSpecCompiler,
    DeterministicSpecToPlanCompiler,
    GenerationError,
    GenerationErrorCode,
    GenerationStatus,
    LLMIntentGenerator,
    M4IntentGenerationResult,
    SpecToPlanContext,
    default_prompt_to_spec_context,
)
from .m4 import DEFAULT_MODEL_ID, M4_001_DESIGN_ID, M4_001_PROMPT, M4_001_SPEC_VERSION
from .pipeline import run_pipeline
from .planning import (
    BoundaryStatus,
    ErrorPlannerResponse,
    HTTPInferenceBackend,
    InferenceBackend,
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


M5_REPAIR_RULES = [
    RepairRule(
        constraint_id="C_WIDTH",
        operation_id="OP01",
        parameter_name="x",
        related_feature_ids=["main_body"],
    ),
    RepairRule(
        constraint_id="C_DEPTH",
        operation_id="OP01",
        parameter_name="y",
        related_feature_ids=["main_body"],
    ),
    RepairRule(
        constraint_id="C_HEIGHT",
        operation_id="OP01",
        parameter_name="z",
        related_feature_ids=["main_body"],
    ),
    RepairRule(
        constraint_id="C_HOLE_DIAMETER",
        operation_id="OP02",
        parameter_name="diameter",
        related_feature_ids=["mount_hole"],
    ),
]


class M5Status(str, Enum):
    VALIDATED = "VALIDATED"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"


class M5Result(StrictModel):
    case_id: Literal["M5-001"] = "M5-001"
    status: M5Status
    stage: str
    reason_code: str | None = None
    message: str | None = None
    user_prompt: str
    design_id: str
    spec_version: str
    intent_generation: M4IntentGenerationResult | None = None
    spec_generation: DesignSpecGenerationResult | None = None
    plan_generation: CADPlanGenerationResult | None = None
    r01_plan: CADPlan | None = None
    r01_cad_status: CADStatus | None = None
    r01_validation_report: ValidationReport | None = None
    repair_attempted: bool = False
    planning_context: RepairPlanningContext | None = None
    planner_response: RepairPlannerResponse | None = None
    planner_run: PlannerRun | None = None
    boundary_status: BoundaryStatus | None = None
    repair_plan_validation: RepairPlanValidation | None = None
    repair_executor_status: RepairStatus | None = None
    r02_plan: CADPlan | None = None
    r02_cad_status: CADStatus | None = None
    r02_validation_report: ValidationReport | None = None
    changed_semantic_paths: list[str] = Field(default_factory=list)
    change_locality: bool | None = None
    preserved_constraint_ids: list[str] = Field(default_factory=list)
    constraint_preservation: bool | None = None
    feature_ids_r01: list[str] = Field(default_factory=list)
    feature_ids_r02: list[str] = Field(default_factory=list)
    feature_identity_preserved: bool | None = None
    artifacts: dict[str, dict[str, str]] = Field(default_factory=dict)
    final_revision_id: str | None = None
    final_artifacts: dict[str, str] = Field(default_factory=dict)


TestPlanTransform = Callable[[CADPlan], CADPlan]


def run_m5_001(
    intent_backend: InferenceBackend,
    output: Path,
    *,
    repair_planner: RepairPlanner | None = None,
    user_prompt: str = M4_001_PROMPT,
    result_json: Path | None = None,
    test_plan_transform: TestPlanTransform | None = None,
) -> M5Result:
    """Run M4 generation plus at most one M2/M3-authorized repair.

    ``test_plan_transform`` is deliberately absent from the CLI. It exists only
    to create reproducible R01 failures in tests/benchmarks after a correct plan
    has been compiled, while keeping the DesignSpec unchanged.
    """

    state: dict[str, Any] = {
        "stage": "PROMPT_TO_INTENT",
        "user_prompt": user_prompt,
        "design_id": M4_001_DESIGN_ID,
        "spec_version": M4_001_SPEC_VERSION,
    }

    def finish(
        status: M5Status,
        reason_code: str | None = None,
        message: str | None = None,
    ) -> M5Result:
        result = M5Result(status=status, reason_code=reason_code, message=message, **state)
        destination = result_json or output / "result.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return result

    try:
        prompt_context = default_prompt_to_spec_context(
            user_prompt,
            M4_001_DESIGN_ID,
            M4_001_SPEC_VERSION,
        )
        intent_generation = LLMIntentGenerator(
            intent_backend,
            max_tokens=1024,
            seed=12345,
        ).generate(prompt_context)
        state["intent_generation"] = intent_generation
        if intent_generation.status != GenerationStatus.SUCCESS or intent_generation.intent is None:
            return finish(
                _generation_failure_status(intent_generation.error),
                _generation_error_code(intent_generation.error),
                intent_generation.error.message if intent_generation.error else None,
            )

        state["stage"] = "INTENT_TO_SPEC"
        spec_generation = DeterministicIntentToSpecCompiler().compile(
            intent_generation.intent,
            design_id=M4_001_DESIGN_ID,
            spec_version=M4_001_SPEC_VERSION,
        )
        state["spec_generation"] = spec_generation
        if spec_generation.status != GenerationStatus.SUCCESS or spec_generation.design_spec is None:
            return finish(
                M5Status.FAIL,
                _generation_error_code(spec_generation.error),
                spec_generation.error.message if spec_generation.error else None,
            )
        spec = spec_generation.design_spec

        state["stage"] = "SPEC_TO_PLAN"
        plan_generation = DeterministicSpecToPlanCompiler().compile(SpecToPlanContext(
            design_id=spec.design_id,
            spec_version=spec.spec_version,
            revision_id="R01",
            design_spec=spec,
        ))
        state["plan_generation"] = plan_generation
        if plan_generation.status != GenerationStatus.SUCCESS or plan_generation.cad_plan is None:
            return finish(
                M5Status.FAIL,
                _generation_error_code(plan_generation.error),
                plan_generation.error.message if plan_generation.error else None,
            )

        compiled_plan = plan_generation.cad_plan
        r01_plan = compiled_plan
        if test_plan_transform is not None:
            state["stage"] = "TEST_FAULT_INJECTION"
            r01_plan = test_plan_transform(compiled_plan.model_copy(deep=True))
            if not isinstance(r01_plan, CADPlan):
                return finish(M5Status.FAIL, "INVALID_TEST_PLAN_TRANSFORM", "transform must return CADPlan")
            if (
                r01_plan.design_id != compiled_plan.design_id
                or r01_plan.spec_version != compiled_plan.spec_version
                or r01_plan.revision_id != "R01"
                or r01_plan.parent_revision_id is not None
            ):
                return finish(
                    M5Status.FAIL,
                    "INVALID_TEST_PLAN_TRANSFORM",
                    "transform must preserve design/spec/R01 identity",
                )
        state["r01_plan"] = r01_plan
        source_snapshot = r01_plan.model_dump(mode="json")
        spec_snapshot = spec.model_dump(mode="json")

        state["stage"] = "R01_CAD"
        r01 = run_pipeline(spec, r01_plan, output / "R01")
        state["r01_cad_status"] = r01.cad_result.status
        state["artifacts"] = {"R01": _artifact_paths(r01.artifacts)}
        if r01.cad_result.status != CADStatus.SUCCESS or r01.validation_report is None:
            return finish(M5Status.FAIL, "R01_CAD_FAILED", _cad_error_message(r01.cad_result.error))

        state["stage"] = "R01_VALIDATION"
        state["r01_validation_report"] = r01.validation_report
        state["feature_ids_r01"] = sorted(r01.cad_result.feature_registry.features)
        if r01.validation_report.status == ReportStatus.PASS:
            state["final_revision_id"] = "R01"
            state["final_artifacts"] = state["artifacts"]["R01"]
            state["stage"] = "COMPLETE"
            return finish(M5Status.VALIDATED)

        failures = [
            check for check in r01.validation_report.checks
            if check.blocking and check.status == CheckStatus.FAIL
        ]
        if not failures:
            return finish(
                M5Status.FAIL,
                "R01_NOT_REPAIRABLE",
                "R01 did not PASS and has no demonstrated blocking FAIL",
            )
        if repair_planner is None:
            return finish(M5Status.FAIL, "REPAIR_PLANNER_REQUIRED", "R01 failed but no repair planner was supplied")

        state["repair_attempted"] = True
        state["stage"] = "REPAIR_PLANNING"
        context = RepairPlanningContextBuilder().build(
            spec,
            r01_plan,
            r01.validation_report,
            M5_REPAIR_RULES,
        )
        state["planning_context"] = context
        invocation = repair_planner.plan(context)
        state["planner_response"] = invocation.response
        state["planner_run"] = invocation.planner_run

        state["stage"] = "REPAIR_BOUNDARY"
        boundary_result = RepairPlanningBoundary(
            RepairPlanValidator(M5_REPAIR_RULES),
            RepairExecutor(),
        ).process(invocation, spec, r01_plan, r01.validation_report)
        state["boundary_status"] = boundary_result.status
        state["repair_plan_validation"] = boundary_result.repair_plan_validation
        if boundary_result.repair_result is not None:
            state["repair_executor_status"] = boundary_result.repair_result.status

        response = invocation.response.root
        if isinstance(response, UnsupportedPlannerResponse):
            return finish(
                M5Status.FAIL,
                f"UNSUPPORTED:{response.reason_code.value}",
                response.reason,
            )
        if isinstance(response, ErrorPlannerResponse):
            blocked = response.error.code in {
                PlannerErrorCode.MODEL_UNAVAILABLE,
                PlannerErrorCode.MODEL_TIMEOUT,
            }
            return finish(
                M5Status.BLOCKED if blocked else M5Status.FAIL,
                response.error.code.value,
                response.error.message,
            )
        if boundary_result.status != BoundaryStatus.APPLIED:
            errors = boundary_result.repair_plan_validation.errors if boundary_result.repair_plan_validation else []
            return finish(
                M5Status.FAIL,
                boundary_result.status.value,
                ", ".join(error.code.value for error in errors) or "repair plan was not applied",
            )

        repair = boundary_result.repair_result
        if repair is None or repair.status != RepairStatus.REPAIRED or repair.revised_plan is None:
            return finish(M5Status.FAIL, "REPAIR_NOT_APPLIED", repair.result_code if repair else None)
        if r01_plan.model_dump(mode="json") != source_snapshot or spec.model_dump(mode="json") != spec_snapshot:
            return finish(M5Status.FAIL, "SOURCE_MUTATED", "DesignSpec or CADPlan R01 was modified in place")

        r02_plan = repair.revised_plan
        state["r02_plan"] = r02_plan
        state["stage"] = "R02_CAD"
        r02 = run_pipeline(spec, r02_plan, output / "R02")
        state["r02_cad_status"] = r02.cad_result.status
        state["artifacts"]["R02"] = _artifact_paths(r02.artifacts)
        if r02.cad_result.status != CADStatus.SUCCESS or r02.validation_report is None:
            return finish(M5Status.FAIL, "R02_CAD_FAILED", _cad_error_message(r02.cad_result.error))

        state["stage"] = "R02_VALIDATION"
        state["r02_validation_report"] = r02.validation_report
        changed_paths = changed_semantic_plan_paths(r01_plan, r02_plan)
        expected_paths = {change.target for change in repair.changes}
        state["changed_semantic_paths"] = sorted(changed_paths)
        state["change_locality"] = changed_paths == expected_paths
        preservation, preserved_ids = preserved_passing_constraints(
            r01.validation_report,
            r02.validation_report,
        )
        state["constraint_preservation"] = preservation
        state["preserved_constraint_ids"] = preserved_ids
        state["feature_ids_r02"] = sorted(r02.cad_result.feature_registry.features)
        state["feature_identity_preserved"] = state["feature_ids_r01"] == state["feature_ids_r02"]

        validated = all((
            r02.validation_report.status == ReportStatus.PASS,
            r02_plan.revision_id == "R02",
            r02_plan.parent_revision_id == "R01",
            state["change_locality"],
            state["constraint_preservation"],
            state["feature_identity_preserved"],
            _artifacts_exist(state["artifacts"].get("R02", {})),
        ))
        if not validated:
            return finish(
                M5Status.FAIL,
                "R02_ACCEPTANCE_FAILED",
                "R02 failed validation, revision, locality, preservation, feature, or artifact checks",
            )
        state["final_revision_id"] = "R02"
        state["final_artifacts"] = state["artifacts"]["R02"]
        state["stage"] = "COMPLETE"
        return finish(M5Status.VALIDATED)
    except Exception as exc:
        return finish(M5Status.FAIL, "M5_RUNTIME_ERROR", str(exc))


def _generation_failure_status(error: GenerationError | None) -> M5Status:
    if error and error.code in {
        GenerationErrorCode.MODEL_TIMEOUT,
        GenerationErrorCode.MODEL_UNAVAILABLE,
    }:
        return M5Status.BLOCKED
    return M5Status.FAIL


def _generation_error_code(error: GenerationError | None) -> str | None:
    return error.code.value if error else None


def _artifact_paths(artifacts: dict[str, Path]) -> dict[str, str]:
    return {kind: str(path.resolve()) for kind, path in artifacts.items()}


def _artifacts_exist(artifacts: dict[str, str]) -> bool:
    return set(artifacts) == {"STEP", "STL"} and all(Path(path).is_file() for path in artifacts.values())


def _cad_error_message(error: Any) -> str | None:
    return error.message if error is not None else None


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CAD AI M5 V0.1 bounded generation and repair flow")
    parser.add_argument("--prompt", default=M4_001_PROMPT)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/m5/M5-001"))
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_cli_parser().parse_args()
    http_config = m3b_http_config(
        base_url=args.base_url,
        endpoint=args.endpoint,
        model_id=args.model_id or DEFAULT_M3B_MODEL_ID,
        timeout_seconds=args.timeout_seconds,
    )
    intent_backend = HTTPInferenceBackend(http_config)
    repair_planner = LLMRepairPlanner(HTTPInferenceBackend(http_config), m3b_llm_config())
    result = run_m5_001(
        intent_backend,
        args.output,
        repair_planner=repair_planner,
        user_prompt=args.prompt,
        result_json=args.result_json,
    )
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(f"M5-001 | status={result.status.value} | stage={result.stage}")
        if result.reason_code:
            print(f"reason={result.reason_code}")
        if result.final_revision_id:
            print(f"final_revision={result.final_revision_id}")
        for kind, path in result.final_artifacts.items():
            print(f"{kind}: {path}")
    return 0 if result.status == M5Status.VALIDATED else (2 if result.status == M5Status.BLOCKED else 1)


if __name__ == "__main__":
    raise SystemExit(main())
