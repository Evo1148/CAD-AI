from __future__ import annotations

import argparse
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import Field

from .benchmark.m3 import DEFAULT_M3B_MODEL_ID, m3b_http_config, m3b_llm_config
from .cad import CADStatus
from .capability_v02 import (
    CapabilityIntentToSpecCompiler,
    CapabilityPlanResult,
    CapabilityRun,
    CapabilitySpecResult,
    CapabilitySpecToPlanCompiler,
    DeterministicPlateIntentGate,
    ExtractedPlateFacts,
    LLMPlateFactExtractor,
    PlateFactExtractionResult,
    PlateIntentGenerationResult,
    PlateIntentResponse,
    PlateIntentUnsupported,
    PlatePromptContext,
    capability_repair_rules,
)
from .contracts import CADPlan, StrictModel
from .generation import GenerationErrorCode, GenerationStatus
from .pipeline import run_pipeline
from .planning import (
    BackendStatus,
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
from .prompt_grounding import (
    DeterministicExtractionCoverage,
    DeterministicFactAssembler,
    DeterministicPromptGrounder,
    ExtractionCoverage,
    ExtractionCoverageStatus,
    ExtractionMode,
    FactGroundingResult,
    FactGroundingValidator,
    GroundedPromptEvidence,
    ResidualExtractionResult,
    ResidualLLMExtractor,
)
from .revision import (
    RepairExecutor,
    RepairPlanValidation,
    RepairPlanValidator,
    RepairStatus,
    changed_semantic_plan_paths,
    preserved_passing_constraints,
)
from .validation import CheckStatus, ReportStatus, ValidationReport


CASE_PROMPTS = {
    "CASE-01": "Create a 60 x 40 x 4 mm rectangular plate with one centered 6 mm through hole.",
    "CASE-02": "Create a 100 x 60 x 4 mm rectangular plate with four 5 mm through holes at (-40,-20), (40,-20), (-40,20), and (40,20).",
    "CASE-03": "Create a 120 x 70 x 5 mm rectangular plate with a 5 mm through hole at (-35,0) and an 8 mm through hole at (35,0).",
    "CASE-04": "Create a 100 x 60 x 4 mm rectangular plate with 5 mm through holes at (-30,0) and (30,0), and a 3 mm fillet on the outer vertical plate edges.",
    "CASE-05": "Create a 100 x 60 x 4 mm rectangular plate with 5 mm through holes at (-30,0) and (30,0), and a 3 mm chamfer on the outer vertical plate edges.",
    "CASE-06": "Create a 100 x 60 x 4 mm rectangular plate with 5 mm through holes at (-30,-15), (30,-15), (-30,15), and (30,15).",
    "CP2-01": "Create a 100 x 60 x 4 mm rectangular plate with four 5 mm through holes at (-40,-20), (40,-20), (-40,20), and (40,20).",
    "CP2-02": "Create a 100 x 60 x 10 mm block with a centered rectangular pocket 40 x 20 mm and 4 mm deep.",
    "CP2-03": "Create an 80 x 50 x 6 mm plate with a centered 30 x 15 mm rectangular through cutout.",
    "CP2-04": "Create a 60 x 60 x 12 mm block with a centered 30 mm diameter circular pocket 5 mm deep.",
    "CP2-05": "Create an 80 x 40 x 6 mm plate with a centered horizontal through slot 30 x 8 mm.",
    "CP2-06": "Create a 100 x 40 x 6 mm plate with four 5 mm through holes in a horizontal linear pattern centered at (0,0), with 20 mm spacing.",
    "CP2-07": "Create a 120 x 70 x 8 mm plate with 5 mm through holes at (-40,20) and (40,20), a centered 36 x 18 mm rectangular pocket 3 mm deep, and a 3 mm fillet on the outer vertical edges.",
    "CP2-08": "Create a 100 x 60 x 10 mm block with a centered rectangular pocket 40 x 20 mm and 4 mm deep.",
    "CP2-09": "Create a 60 x 40 x 8 mm block with an M8 threaded hole.",
    "CP3-01": "Create a 100 x 60 x 10 mm block with a centered rectangular pocket 40 x 20 mm and 4 mm deep.",
    "CP3-02": "Create a 100 x 60 x 4 mm plate with a centered 30 x 20 mm rectangular boss 8 mm high.",
    "CP3-03": "Create an 80 x 60 x 5 mm plate with a 16 mm diameter cylindrical boss 10 mm high at (20,-10).",
    "CP3-04": "Create an 80 x 60 x 4 mm plate. Add an 8 mm diameter standoff 12 mm high at (20,15).",
    "CP3-05": "Create an 80 x 60 x 4 mm plate. Add a 10 mm diameter standoff 12 mm high at (-20,10) with a 3 mm through hole.",
    "CP3-06": "Create a 100 x 70 x 4 mm plate with four 8 mm diameter standoffs 12 mm high at (-35,-20), (35,-20), (-35,20), (35,20), each with a 3 mm through hole.",
    "CP3-07": "Create a 100 x 50 x 4 mm plate with four 8 mm diameter standoffs 10 mm high in a horizontal linear pattern centered at (0,0), spaced 20 mm apart.",
    "CP3-08": "Create a 140 x 90 x 8 mm plate with 5 mm through holes at (-55,-30) and (55,-30). Add four 10 mm diameter standoffs 12 mm high at (-50,-5), (50,-5), (-50,25), (50,25), each with a 3 mm through hole. Add a centered 30 x 18 mm rectangular pocket 3 mm deep and a 2 mm fillet on the outer vertical edges.",
    "CP3-09": "Create an 80 x 60 x 4 mm plate. Add a 10 mm diameter standoff 12 mm high at (-20,10) with a 3 mm through hole.",
    "CP3-10": "Create an 80 x 60 x 4 mm plate with a 16 mm diameter cylindrical boss 10 mm high on the left side face.",
}


class V02Status(str, Enum):
    VALIDATED = "VALIDATED"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    UNSUPPORTED = "UNSUPPORTED"


class CapabilityPackResult(StrictModel):
    case_id: str
    status: V02Status
    stage: str
    reason_code: str | None = None
    message: str | None = None
    user_prompt: str
    design_id: str
    spec_version: str = "2.0"
    fact_extraction: PlateFactExtractionResult | None = None
    prompt_evidence: GroundedPromptEvidence | None = None
    grounding_coverage: ExtractionCoverage | None = None
    extraction_mode: ExtractionMode | None = None
    fact_source: Literal["deterministic_grounding", "hybrid_llm"] | None = None
    llm_invoked: bool = False
    residual_extraction: ResidualExtractionResult | None = None
    canonical_facts: ExtractedPlateFacts | None = None
    fact_grounding: FactGroundingResult | None = None
    intent_response: PlateIntentResponse | None = None
    intent_generation: PlateIntentGenerationResult | None = None
    spec_generation: CapabilitySpecResult | None = None
    plan_generation: CapabilityPlanResult | None = None
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


def run_capability_case(
    case_id: str,
    intent_backend: InferenceBackend,
    output: Path,
    *,
    user_prompt: str | None = None,
    repair_planner: RepairPlanner | None = None,
    result_json: Path | None = None,
    test_plan_transform: TestPlanTransform | None = None,
) -> CapabilityPackResult:
    if case_id not in CASE_PROMPTS:
        raise ValueError(f"unknown capability case: {case_id}")
    prompt = user_prompt or CASE_PROMPTS[case_id]
    design_id = f"V02-{case_id}"
    state: dict[str, Any] = {
        "case_id": case_id,
        "stage": "PROMPT_GROUNDING",
        "user_prompt": prompt,
        "design_id": design_id,
    }

    def finish(
        status: V02Status,
        reason_code: str | None = None,
        message: str | None = None,
    ) -> CapabilityPackResult:
        result = CapabilityPackResult(status=status, reason_code=reason_code, message=message, **state)
        destination = result_json or output / "result.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return result

    try:
        prompt_evidence = DeterministicPromptGrounder().ground(prompt)
        state["prompt_evidence"] = prompt_evidence
        coverage = DeterministicExtractionCoverage().evaluate(prompt_evidence)
        state["grounding_coverage"] = coverage
        if coverage.status == ExtractionCoverageStatus.INSUFFICIENT:
            return finish(
                V02Status.FAIL,
                GenerationErrorCode.EXTRACTION_COVERAGE_INSUFFICIENT.value,
                coverage.reason,
            )
        residual_extraction = None
        if coverage.status == ExtractionCoverageStatus.COMPLETE:
            state["extraction_mode"] = ExtractionMode.DETERMINISTIC
            state["fact_source"] = "deterministic_grounding"
            state["llm_invoked"] = False
        else:
            state["extraction_mode"] = ExtractionMode.HYBRID_LLM
            state["fact_source"] = "hybrid_llm"
            state["llm_invoked"] = True
            state["stage"] = "RESIDUAL_FACT_EXTRACTION"
            residual_extraction = ResidualLLMExtractor(intent_backend).extract(
                prompt, prompt_evidence, coverage,
            )
            state["residual_extraction"] = residual_extraction
            if residual_extraction.status != GenerationStatus.SUCCESS or residual_extraction.residual_facts is None:
                error = residual_extraction.error
                return finish(
                    _generation_status(error.code if error else None),
                    error.code.value if error else "RESIDUAL_EXTRACTION_ERROR",
                    error.message if error else None,
                )

        state["stage"] = "FACT_ASSEMBLY"
        canonical_facts = DeterministicFactAssembler().assemble(
            prompt_evidence,
            residual_extraction.residual_facts if residual_extraction else None,
        )
        state["canonical_facts"] = canonical_facts
        if residual_extraction is not None:
            state["stage"] = "FACT_GROUNDING"
            fact_grounding = FactGroundingValidator().validate(prompt_evidence, canonical_facts)
            state["fact_grounding"] = fact_grounding
            if fact_grounding.status == "GROUNDING_ERROR" or fact_grounding.grounded_facts is None:
                return finish(
                    V02Status.FAIL,
                    GenerationErrorCode.GROUNDING_ERROR.value,
                    ", ".join(
                        f"{violation.code.value}:{violation.field}"
                        for violation in fact_grounding.violations
                    ),
                )
            canonical_facts = fact_grounding.grounded_facts
            state["canonical_facts"] = canonical_facts

        state["stage"] = "FACTS_TO_INTENT"
        intent_response = DeterministicPlateIntentGate().evaluate(canonical_facts)
        state["intent_response"] = intent_response
        legacy_run = (
            CapabilityRun(
                stage="PROMPT_TO_INTENT",
                implementation=intent_backend.backend_type,
                model_id=intent_backend.model_id,
                latency_ms=residual_extraction.run.latency_ms,
                backend_status=residual_extraction.run.backend_status,
                schema_valid=residual_extraction.run.schema_valid,
                backend_metadata=residual_extraction.run.backend_metadata,
            )
            if residual_extraction
            else CapabilityRun(
                stage="PROMPT_TO_INTENT",
                implementation="deterministic-evidence-first",
                latency_ms=0.0,
                backend_status=BackendStatus.SUCCESS,
                schema_valid=True,
            )
        )
        if isinstance(intent_response.root, PlateIntentUnsupported):
            state["intent_generation"] = PlateIntentGenerationResult(
                status="UNSUPPORTED",
                response=intent_response,
                run=legacy_run,
                raw_output=residual_extraction.raw_output if residual_extraction else None,
            )
            return finish(
                V02Status.UNSUPPORTED,
                intent_response.root.reason_code.value,
                intent_response.root.reason,
            )
        intent = intent_response.root.intent
        state["intent_generation"] = PlateIntentGenerationResult(
            status="SUCCESS",
            intent=intent,
            response=intent_response,
            run=legacy_run,
            raw_output=residual_extraction.raw_output if residual_extraction else None,
        )

        state["stage"] = "INTENT_TO_SPEC"
        spec_generation = CapabilityIntentToSpecCompiler().compile(
            intent,
            design_id=design_id,
        )
        state["spec_generation"] = spec_generation
        if spec_generation.status != GenerationStatus.SUCCESS or spec_generation.design_spec is None:
            error = spec_generation.error
            return finish(V02Status.FAIL, error.code.value if error else "SPEC_COMPILATION_ERROR", error.message if error else None)
        spec = spec_generation.design_spec

        state["stage"] = "SPEC_TO_PLAN"
        plan_generation = CapabilitySpecToPlanCompiler().compile(spec)
        state["plan_generation"] = plan_generation
        if plan_generation.status != GenerationStatus.SUCCESS or plan_generation.cad_plan is None:
            error = plan_generation.error
            return finish(V02Status.FAIL, error.code.value if error else "PLAN_COMPILATION_ERROR", error.message if error else None)
        compiled_plan = plan_generation.cad_plan
        r01_plan = compiled_plan
        if test_plan_transform is not None:
            state["stage"] = "TEST_FAULT_INJECTION"
            r01_plan = test_plan_transform(compiled_plan.model_copy(deep=True))
            if not isinstance(r01_plan, CADPlan):
                return finish(V02Status.FAIL, "INVALID_TEST_PLAN_TRANSFORM", "transform must return CADPlan")
            if (
                r01_plan.design_id != compiled_plan.design_id
                or r01_plan.spec_version != compiled_plan.spec_version
                or r01_plan.revision_id != "R01"
                or r01_plan.parent_revision_id is not None
            ):
                return finish(V02Status.FAIL, "INVALID_TEST_PLAN_TRANSFORM", "transform must preserve source identity")
        state["r01_plan"] = r01_plan
        plan_snapshot = r01_plan.model_dump(mode="json")
        spec_snapshot = spec.model_dump(mode="json")

        state["stage"] = "R01_CAD"
        r01 = run_pipeline(spec, r01_plan, output / "R01")
        state["r01_cad_status"] = r01.cad_result.status
        state["artifacts"] = {"R01": _artifact_paths(r01.artifacts)}
        if r01.cad_result.status != CADStatus.SUCCESS or r01.validation_report is None:
            return finish(V02Status.FAIL, "R01_CAD_FAILED", _cad_error_message(r01.cad_result.error))
        state["r01_validation_report"] = r01.validation_report
        state["feature_ids_r01"] = sorted(r01.cad_result.feature_registry.features)
        if r01.validation_report.status == ReportStatus.PASS:
            state["stage"] = "COMPLETE"
            state["final_revision_id"] = "R01"
            state["final_artifacts"] = state["artifacts"]["R01"]
            return finish(V02Status.VALIDATED)

        failures = [
            check for check in r01.validation_report.checks
            if check.blocking and check.status == CheckStatus.FAIL
        ]
        if not failures or repair_planner is None:
            return finish(
                V02Status.FAIL,
                "REPAIR_UNAVAILABLE",
                "R01 failed without a demonstrated repair path",
            )

        rules = capability_repair_rules(spec, r01_plan)
        state["repair_attempted"] = True
        state["stage"] = "REPAIR_PLANNING"
        context = RepairPlanningContextBuilder().build(spec, r01_plan, r01.validation_report, rules)
        state["planning_context"] = context
        invocation = repair_planner.plan(context)
        state["planner_response"] = invocation.response
        state["planner_run"] = invocation.planner_run

        state["stage"] = "REPAIR_BOUNDARY"
        boundary_result = RepairPlanningBoundary(
            RepairPlanValidator(rules),
            RepairExecutor(),
        ).process(invocation, spec, r01_plan, r01.validation_report)
        state["boundary_status"] = boundary_result.status
        state["repair_plan_validation"] = boundary_result.repair_plan_validation
        if boundary_result.repair_result:
            state["repair_executor_status"] = boundary_result.repair_result.status
        response = invocation.response.root
        if isinstance(response, UnsupportedPlannerResponse):
            return finish(V02Status.FAIL, f"UNSUPPORTED:{response.reason_code.value}", response.reason)
        if isinstance(response, ErrorPlannerResponse):
            blocked = response.error.code in {PlannerErrorCode.MODEL_UNAVAILABLE, PlannerErrorCode.MODEL_TIMEOUT}
            return finish(V02Status.BLOCKED if blocked else V02Status.FAIL, response.error.code.value, response.error.message)
        if boundary_result.status != BoundaryStatus.APPLIED:
            errors = boundary_result.repair_plan_validation.errors if boundary_result.repair_plan_validation else []
            return finish(
                V02Status.FAIL,
                boundary_result.status.value,
                ", ".join(error.code.value for error in errors) or "repair plan was not applied",
            )
        repair = boundary_result.repair_result
        if repair is None or repair.status != RepairStatus.REPAIRED or repair.revised_plan is None:
            return finish(V02Status.FAIL, "REPAIR_NOT_APPLIED", repair.result_code if repair else None)
        if r01_plan.model_dump(mode="json") != plan_snapshot or spec.model_dump(mode="json") != spec_snapshot:
            return finish(V02Status.FAIL, "SOURCE_MUTATED", "DesignSpec or R01 was modified in place")

        r02_plan = repair.revised_plan
        state["r02_plan"] = r02_plan
        state["stage"] = "R02_CAD"
        r02 = run_pipeline(spec, r02_plan, output / "R02")
        state["r02_cad_status"] = r02.cad_result.status
        state["artifacts"]["R02"] = _artifact_paths(r02.artifacts)
        if r02.cad_result.status != CADStatus.SUCCESS or r02.validation_report is None:
            return finish(V02Status.FAIL, "R02_CAD_FAILED", _cad_error_message(r02.cad_result.error))
        state["r02_validation_report"] = r02.validation_report
        changed = changed_semantic_plan_paths(r01_plan, r02_plan)
        expected = {change.target for change in repair.changes}
        state["changed_semantic_paths"] = sorted(changed)
        state["change_locality"] = changed == expected
        preservation, preserved_ids = preserved_passing_constraints(r01.validation_report, r02.validation_report)
        state["constraint_preservation"] = preservation
        state["preserved_constraint_ids"] = preserved_ids
        state["feature_ids_r02"] = sorted(r02.cad_result.feature_registry.features)
        state["feature_identity_preserved"] = state["feature_ids_r01"] == state["feature_ids_r02"]
        accepted = all((
            r02.validation_report.status == ReportStatus.PASS,
            r02_plan.revision_id == "R02",
            r02_plan.parent_revision_id == "R01",
            state["change_locality"],
            state["constraint_preservation"],
            state["feature_identity_preserved"],
            _artifacts_exist(state["artifacts"]["R02"]),
        ))
        if not accepted:
            return finish(V02Status.FAIL, "R02_ACCEPTANCE_FAILED", "R02 did not satisfy all acceptance checks")
        state["stage"] = "COMPLETE"
        state["final_revision_id"] = "R02"
        state["final_artifacts"] = state["artifacts"]["R02"]
        return finish(V02Status.VALIDATED)
    except Exception as exc:
        return finish(V02Status.FAIL, "CAPABILITY_RUNTIME_ERROR", str(exc))


def _generation_status(code: GenerationErrorCode | None) -> V02Status:
    if code in {GenerationErrorCode.MODEL_TIMEOUT, GenerationErrorCode.MODEL_UNAVAILABLE}:
        return V02Status.BLOCKED
    return V02Status.FAIL


def _artifact_paths(artifacts: dict[str, Path]) -> dict[str, str]:
    return {kind: str(path.resolve()) for kind, path in artifacts.items()}


def _artifacts_exist(artifacts: dict[str, str]) -> bool:
    return set(artifacts) == {"STEP", "STL"} and all(Path(path).is_file() for path in artifacts.values())


def _cad_error_message(error: Any) -> str | None:
    return error.message if error else None


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CAD AI V0.2 Capability Packs 1, 2 and 3")
    parser.add_argument("case_id", choices=sorted(CASE_PROMPTS))
    parser.add_argument("--prompt")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--model-id", default=DEFAULT_M3B_MODEL_ID)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_cli_parser().parse_args()
    output = args.output or Path("artifacts/v02") / args.case_id
    config = m3b_http_config(
        base_url=args.base_url,
        endpoint=args.endpoint,
        model_id=args.model_id,
        timeout_seconds=args.timeout_seconds,
    )
    result = run_capability_case(
        args.case_id,
        HTTPInferenceBackend(config),
        output,
        user_prompt=args.prompt,
        repair_planner=LLMRepairPlanner(HTTPInferenceBackend(config), m3b_llm_config()),
        result_json=args.result_json,
    )
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(f"{result.case_id} | status={result.status.value} | stage={result.stage}")
        if result.reason_code:
            print(f"reason={result.reason_code}")
        for kind, path in result.final_artifacts.items():
            print(f"{kind}: {path}")
    return 0 if result.status == V02Status.VALIDATED else (2 if result.status == V02Status.BLOCKED else 1)


if __name__ == "__main__":
    raise SystemExit(main())
