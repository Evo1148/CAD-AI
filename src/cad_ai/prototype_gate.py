from __future__ import annotations

import argparse
import time
from enum import Enum
from pathlib import Path
from typing import Iterable

from pydantic import Field, model_validator

from .benchmark.m3 import DEFAULT_M3B_MODEL_ID, m3b_http_config
from .cad import CADStatus
from .capability_v02 import (
    CapabilityIntentToSpecCompiler,
    CapabilitySpecToPlanCompiler,
    DeterministicPlateIntentGate,
    ExtractedCircularPocketFacts,
    ExtractedCoordinateFacts,
    ExtractedHoleGroupFacts,
    ExtractedLinearHolePatternFacts,
    ExtractedPlateFacts,
    ExtractedRectangularPocketFacts,
    ExtractedSlotFacts,
    PlateIntentPayload,
)
from .contracts import StrictModel
from .planning import (
    BackendFailureCode,
    BackendResponse,
    BackendRunMetadata,
    BackendStatus,
    HTTPInferenceBackend,
    InferenceRequest,
    ScriptedFakeBackend,
)
from .prompt_grounding import (
    ExtractionCoverageStatus,
    ExtractionMode,
    ResidualFacts,
)
from .validation import ReportStatus
from .v02 import CapabilityPackResult, V02Status, run_capability_case


PROTOTYPE_GATE_VERSION = "prototype-usability-gate-v0.1"


class PrototypeCaseClass(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"
    UNSUPPORTED = "UNSUPPORTED"


class PrototypeCase(StrictModel):
    case_id: str
    prompt: str
    expected_class: PrototypeCaseClass
    expected_coverage: ExtractionCoverageStatus
    expected_facts: ExtractedPlateFacts | None = None
    expected_llm_usage: bool
    expected_unresolved_fields: list[str] = Field(default_factory=list)
    controlled_residual_values: dict[str, float] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent_expectation(self) -> "PrototypeCase":
        supported = {PrototypeCaseClass.COMPLETE, PrototypeCaseClass.PARTIAL}
        if self.expected_class in supported and self.expected_facts is None:
            raise ValueError("supported cases require expected_facts")
        if self.expected_class == PrototypeCaseClass.PARTIAL:
            if self.expected_coverage != ExtractionCoverageStatus.PARTIAL:
                raise ValueError("PARTIAL cases require PARTIAL coverage")
            if set(self.controlled_residual_values) != set(self.expected_unresolved_fields):
                raise ValueError("controlled residual values must exactly match unresolved fields")
        elif self.controlled_residual_values:
            raise ValueError("only PARTIAL cases may define controlled residual values")
        if self.expected_llm_usage != (self.expected_class == PrototypeCaseClass.PARTIAL):
            raise ValueError("only PARTIAL cases invoke the residual LLM")
        return self


class PrototypeCaseMetrics(StrictModel):
    case_id: str
    classification: PrototypeCaseClass
    coverage: ExtractionCoverageStatus | None = None
    extraction_mode: ExtractionMode | None = None
    llm_invoked: bool
    unresolved_fields: list[str] = Field(default_factory=list)
    grounding_status: str
    intent_status: str
    cad_status: str
    validator_status: str
    final_revision: str | None = None
    step_generated: bool
    stl_generated: bool
    total_latency_ms: float = Field(ge=0)
    model_id: str | None = None
    prompt_version: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    llm_latency_ms: float | None = Field(default=None, ge=0)
    generation_tps: float | None = Field(default=None, ge=0)
    schema_valid: bool | None = None


class PrototypeCaseResult(StrictModel):
    benchmark_version: str = PROTOTYPE_GATE_VERSION
    case_id: str
    prompt: str
    expected_class: PrototypeCaseClass
    expected_llm_usage: bool
    actual_class: PrototypeCaseClass
    passed: bool
    false_validated: bool
    semantic_errors: list[str] = Field(default_factory=list)
    pipeline_status: V02Status
    pipeline_stage: str
    pipeline_reason_code: str | None = None
    actual_canonical_facts: ExtractedPlateFacts | None = None
    metrics: PrototypeCaseMetrics
    pipeline_result_path: str
    result_path: str


class PrototypeGateSummary(StrictModel):
    benchmark_version: str = PROTOTYPE_GATE_VERSION
    backend: str
    model_id: str | None = None
    total: int
    passed: int
    failed: int
    complete_count: int
    partial_count: int
    insufficient_count: int
    unsupported_count: int
    deterministic_success: int
    hybrid_success: int
    grounding_error: int
    intent_error: int
    cad_fail: int
    validation_fail: int
    false_validated: int
    total_latency_ms: float = Field(ge=0)
    results: list[PrototypeCaseResult]


class _FailIfCalledBackend:
    backend_type = "deterministic-no-llm"
    model_id = None

    def generate(self, request: InferenceRequest) -> BackendResponse:
        return BackendResponse(
            status=BackendStatus.ERROR,
            error_code=BackendFailureCode.MODEL_ERROR,
            message="LLM invocation was forbidden by the expected benchmark classification",
        )


def _coordinates(items: list[tuple[float, float]]) -> list[ExtractedCoordinateFacts]:
    return [ExtractedCoordinateFacts(x=float(x), y=float(y)) for x, y in items]


def prototype_cases() -> list[PrototypeCase]:
    empty = dict(
        holes=[], hole_groups=[], rectangular_pockets=[], circular_pockets=[], slots=[],
        linear_hole_patterns=[], fillet_radius=None, chamfer_distance=None,
        unsupported_features=[],
    )
    cases = [
        PrototypeCase(
            case_id="PUG-C01",
            prompt="Drawing note 2026. Build a 92 x 54 x 7 mm rectangular block; no cut features are requested.",
            expected_class="COMPLETE", expected_coverage="COMPLETE", expected_llm_usage=False,
            expected_facts=ExtractedPlateFacts(width=92.0, depth=54.0, height=7.0, **empty),
            tags=["base", "irrelevant-number"],
        ),
        PrototypeCase(
            case_id="PUG-C02",
            prompt="Machine a rectangular plate measuring 64 x 44 x 5 mm with one centered 7 mm through hole.",
            expected_class="COMPLETE", expected_coverage="COMPLETE", expected_llm_usage=False,
            expected_facts=ExtractedPlateFacts(
                width=64.0, depth=44.0, height=5.0,
                hole_groups=[ExtractedHoleGroupFacts(
                    diameter=7.0, positions=_coordinates([(0.0, 0.0)]),
                )],
            ),
            tags=["centered-hole", "regression"],
        ),
        PrototypeCase(
            case_id="PUG-C03",
            prompt="Start from a 110 x 70 x 6 mm plate and add 4 mm through holes at (-45,-25), (45,-25), (-45,25), and (45,25).",
            expected_class="COMPLETE", expected_coverage="COMPLETE", expected_llm_usage=False,
            expected_facts=ExtractedPlateFacts(
                width=110.0, depth=70.0, height=6.0,
                hole_groups=[ExtractedHoleGroupFacts(
                    diameter=4.0,
                    positions=_coordinates([(-45.0, -25.0), (45.0, -25.0), (-45.0, 25.0), (45.0, 25.0)]),
                )],
            ),
            tags=["hole-group", "shared-diameter", "adversarial"],
        ),
        PrototypeCase(
            case_id="PUG-C04",
            prompt="Make a 90 x 50 x 8 mm block. Add a centered rectangular pocket 32 x 18 mm and 3 mm deep.",
            expected_class="COMPLETE", expected_coverage="COMPLETE", expected_llm_usage=False,
            expected_facts=ExtractedPlateFacts(
                width=90.0, depth=50.0, height=8.0,
                rectangular_pockets=[ExtractedRectangularPocketFacts(
                    width=32.0, depth=18.0, centered=True, cut_depth=3.0,
                )],
            ),
            tags=["rectangular-pocket", "paraphrase"],
        ),
        PrototypeCase(
            case_id="PUG-C05",
            prompt="Use a 70 x 50 x 10 mm base with a centered 24 mm diameter circular pocket 4 mm deep.",
            expected_class="COMPLETE", expected_coverage="COMPLETE", expected_llm_usage=False,
            expected_facts=ExtractedPlateFacts(
                width=70.0, depth=50.0, height=10.0,
                circular_pockets=[ExtractedCircularPocketFacts(
                    diameter=24.0, centered=True, cut_depth=4.0,
                )],
            ),
            tags=["circular-pocket", "paraphrase"],
        ),
        PrototypeCase(
            case_id="PUG-C06",
            prompt="Produce a 96 x 58 x 6 mm plate, including a centered 28 x 14 mm rectangular through cutout.",
            expected_class="COMPLETE", expected_coverage="COMPLETE", expected_llm_usage=False,
            expected_facts=ExtractedPlateFacts(
                width=96.0, depth=58.0, height=6.0,
                rectangular_pockets=[ExtractedRectangularPocketFacts(
                    width=28.0, depth=14.0, centered=True, through=True,
                )],
            ),
            tags=["rectangular-cutout", "through"],
        ),
        PrototypeCase(
            case_id="PUG-C07",
            prompt="On an 85 x 45 x 6 mm plate, make a centered horizontal through slot 28 x 7 mm.",
            expected_class="COMPLETE", expected_coverage="COMPLETE", expected_llm_usage=False,
            expected_facts=ExtractedPlateFacts(
                width=85.0, depth=45.0, height=6.0,
                slots=[ExtractedSlotFacts(
                    length=28.0, width=7.0, centered=True, angle_deg=0.0, through=True,
                )],
            ),
            tags=["slot", "through"],
        ),
        PrototypeCase(
            case_id="PUG-C08",
            prompt="Prepare a 105 x 45 x 6 mm plate with five 4 mm through holes in a vertical linear pattern centered at (0,0), with 8 mm spacing.",
            expected_class="COMPLETE", expected_coverage="COMPLETE", expected_llm_usage=False,
            expected_facts=ExtractedPlateFacts(
                width=105.0, depth=45.0, height=6.0,
                linear_hole_patterns=[ExtractedLinearHolePatternFacts(
                    diameter=4.0, count=5, spacing=8.0, axis="Y",
                    anchor_x=0.0, anchor_y=0.0, anchor_mode="CENTER",
                )],
            ),
            tags=["linear-pattern", "spacing", "adversarial"],
        ),
        PrototypeCase(
            case_id="PUG-C09",
            prompt="Machine a 100 x 64 x 9 mm block with a centered rectangular pocket 30 x 16 mm and 3 mm deep, and a 2 mm fillet on the outer vertical edges.",
            expected_class="COMPLETE", expected_coverage="COMPLETE", expected_llm_usage=False,
            expected_facts=ExtractedPlateFacts(
                width=100.0, depth=64.0, height=9.0,
                rectangular_pockets=[ExtractedRectangularPocketFacts(
                    width=30.0, depth=16.0, centered=True, cut_depth=3.0,
                )], fillet_radius=2.0,
            ),
            tags=["pocket", "fillet-only", "adversarial"],
        ),
        PrototypeCase(
            case_id="PUG-C10",
            prompt="Make a 100 x 60 x 8 mm plate with 5 mm through holes at (-30,0) and (30,0), plus a 2 mm chamfer on the outer vertical edges.",
            expected_class="COMPLETE", expected_coverage="COMPLETE", expected_llm_usage=False,
            expected_facts=ExtractedPlateFacts(
                width=100.0, depth=60.0, height=8.0,
                hole_groups=[ExtractedHoleGroupFacts(
                    diameter=5.0, positions=_coordinates([(-30.0, 0.0), (30.0, 0.0)]),
                )], chamfer_distance=2.0,
            ),
            tags=["holes", "chamfer-only", "adversarial"],
        ),
        PrototypeCase(
            case_id="PUG-C11",
            prompt="Create a 75 x 45 x 5 mm thread-free plate with one centered 5 mm through hole.",
            expected_class="COMPLETE", expected_coverage="COMPLETE", expected_llm_usage=False,
            expected_facts=ExtractedPlateFacts(
                width=75.0, depth=45.0, height=5.0,
                hole_groups=[ExtractedHoleGroupFacts(
                    diameter=5.0, positions=_coordinates([(0.0, 0.0)]),
                )],
            ),
            tags=["supported-keyword-near-unsupported", "adversarial"],
        ),
        PrototypeCase(
            case_id="PUG-P01",
            prompt="Create an 80 x 40 x 6 mm plate with a horizontal through slot 30 x 8 mm. Put the slot centre at X=-10 mm and Y=0 mm.",
            expected_class="PARTIAL", expected_coverage="PARTIAL", expected_llm_usage=True,
            expected_unresolved_fields=["slots[0].x", "slots[0].y"],
            controlled_residual_values={"slots[0].x": -10.0, "slots[0].y": 0.0},
            expected_facts=ExtractedPlateFacts(
                width=80.0, depth=40.0, height=6.0,
                slots=[ExtractedSlotFacts(
                    length=30.0, width=8.0, x=-10.0, y=0.0,
                    angle_deg=0.0, through=True,
                )],
            ), tags=["hybrid", "slot-position"],
        ),
        PrototypeCase(
            case_id="PUG-P02",
            prompt="Make a 100 x 60 x 10 mm block with a rectangular pocket 24 x 12 mm and 3 mm deep. Its centre is at X=15 mm and Y=-5 mm.",
            expected_class="PARTIAL", expected_coverage="PARTIAL", expected_llm_usage=True,
            expected_unresolved_fields=["rectangular_pockets[0].x", "rectangular_pockets[0].y"],
            controlled_residual_values={
                "rectangular_pockets[0].x": 15.0, "rectangular_pockets[0].y": -5.0,
            },
            expected_facts=ExtractedPlateFacts(
                width=100.0, depth=60.0, height=10.0,
                rectangular_pockets=[ExtractedRectangularPocketFacts(
                    width=24.0, depth=12.0, x=15.0, y=-5.0, cut_depth=3.0,
                )],
            ), tags=["hybrid", "rectangular-pocket-position"],
        ),
        PrototypeCase(
            case_id="PUG-P03",
            prompt="Use a 70 x 60 x 12 mm block with a 20 mm diameter circular pocket 4 mm deep. Locate its centre at X=8 mm and Y=6 mm.",
            expected_class="PARTIAL", expected_coverage="PARTIAL", expected_llm_usage=True,
            expected_unresolved_fields=["circular_pockets[0].x", "circular_pockets[0].y"],
            controlled_residual_values={
                "circular_pockets[0].x": 8.0, "circular_pockets[0].y": 6.0,
            },
            expected_facts=ExtractedPlateFacts(
                width=70.0, depth=60.0, height=12.0,
                circular_pockets=[ExtractedCircularPocketFacts(
                    diameter=20.0, x=8.0, y=6.0, cut_depth=4.0,
                )],
            ), tags=["hybrid", "circular-pocket-position"],
        ),
        PrototypeCase(
            case_id="PUG-P04",
            prompt="Build an 88 x 52 x 7 mm plate with a vertical through slot 26 x 6 mm. Its centre coordinates are X=0 mm and Y=9 mm.",
            expected_class="PARTIAL", expected_coverage="PARTIAL", expected_llm_usage=True,
            expected_unresolved_fields=["slots[0].x", "slots[0].y"],
            controlled_residual_values={"slots[0].x": 0.0, "slots[0].y": 9.0},
            expected_facts=ExtractedPlateFacts(
                width=88.0, depth=52.0, height=7.0,
                slots=[ExtractedSlotFacts(
                    length=26.0, width=6.0, x=0.0, y=9.0,
                    angle_deg=90.0, through=True,
                )],
            ), tags=["hybrid", "vertical-slot-position"],
        ),
        PrototypeCase(
            case_id="PUG-I01", prompt="Create a plate with a hole.",
            expected_class="INSUFFICIENT", expected_coverage="INSUFFICIENT",
            expected_llm_usage=False, tags=["missing-base", "missing-hole-data"],
        ),
        PrototypeCase(
            case_id="PUG-I02", prompt="Create a 100 x 50 x 6 mm plate with a slot.",
            expected_class="INSUFFICIENT", expected_coverage="INSUFFICIENT",
            expected_llm_usage=False, expected_unresolved_fields=["slots"],
            tags=["missing-slot-data"],
        ),
        PrototypeCase(
            case_id="PUG-U01", prompt="Build a 70 x 40 x 8 mm block with an M8 threaded hole.",
            expected_class="UNSUPPORTED", expected_coverage="COMPLETE", expected_llm_usage=False,
            expected_facts=ExtractedPlateFacts(
                width=70.0, depth=40.0, height=8.0, unsupported_features=["thread"],
            ), tags=["thread", "unsupported"],
        ),
        PrototypeCase(
            case_id="PUG-U02", prompt="Make a 90 x 60 x 12 mm block and shell it to 2 mm walls.",
            expected_class="UNSUPPORTED", expected_coverage="COMPLETE", expected_llm_usage=False,
            expected_facts=ExtractedPlateFacts(
                width=90.0, depth=60.0, height=12.0, unsupported_features=["shell"],
            ), tags=["shell", "unsupported"],
        ),
        PrototypeCase(
            case_id="PUG-U03", prompt="Create a 100 x 60 x 6 mm plate with six 4 mm holes in a circular pattern.",
            expected_class="UNSUPPORTED", expected_coverage="COMPLETE", expected_llm_usage=False,
            expected_facts=ExtractedPlateFacts(
                width=100.0, depth=60.0, height=6.0,
                unsupported_features=["circular pattern"],
            ), tags=["circular-pattern", "unsupported"],
        ),
    ]
    return cases


def _controlled_backend(case: PrototypeCase):
    if case.expected_class != PrototypeCaseClass.PARTIAL:
        return _FailIfCalledBackend()
    residual = ResidualFacts(values=case.controlled_residual_values)
    return ScriptedFakeBackend([residual.model_dump_json()], model_id="controlled-residual")


def _actual_class(result: CapabilityPackResult) -> PrototypeCaseClass:
    coverage = result.grounding_coverage.status if result.grounding_coverage else None
    if coverage == ExtractionCoverageStatus.INSUFFICIENT:
        return PrototypeCaseClass.INSUFFICIENT
    if result.status == V02Status.UNSUPPORTED:
        return PrototypeCaseClass.UNSUPPORTED
    if coverage == ExtractionCoverageStatus.PARTIAL:
        return PrototypeCaseClass.PARTIAL
    return PrototypeCaseClass.COMPLETE


def _artifact_exists(result: CapabilityPackResult, kind: str) -> bool:
    value = result.final_artifacts.get(kind)
    return bool(value and Path(value).is_file())


def _telemetry(result: CapabilityPackResult) -> tuple[
    str | None, str | None, int | None, int | None, float | None, float | None, bool | None
]:
    extraction = result.residual_extraction
    if extraction is None:
        return None, None, None, None, None, None, None
    run = extraction.run
    metadata: BackendRunMetadata | None = run.backend_metadata
    return (
        run.model_id,
        run.prompt_version,
        metadata.prompt_tokens if metadata else None,
        metadata.completion_tokens if metadata else None,
        run.latency_ms,
        metadata.generation_tps if metadata else None,
        run.schema_valid,
    )


def evaluate_case_result(
    case: PrototypeCase,
    pipeline: CapabilityPackResult,
    *,
    total_latency_ms: float,
    pipeline_result_path: Path,
    result_path: Path,
) -> PrototypeCaseResult:
    errors: list[str] = []
    coverage = pipeline.grounding_coverage.status if pipeline.grounding_coverage else None
    unresolved = pipeline.grounding_coverage.unresolved_fields if pipeline.grounding_coverage else []
    actual_class = _actual_class(pipeline)
    if coverage != case.expected_coverage:
        errors.append("COVERAGE_MISMATCH")
    if unresolved != case.expected_unresolved_fields:
        errors.append("UNRESOLVED_FIELDS_MISMATCH")
    if pipeline.llm_invoked != case.expected_llm_usage:
        errors.append("LLM_USAGE_MISMATCH")
    if actual_class != case.expected_class:
        errors.append("CLASSIFICATION_MISMATCH")

    semantically_supported = case.expected_class in {
        PrototypeCaseClass.COMPLETE, PrototypeCaseClass.PARTIAL,
    }
    if case.expected_facts is not None and pipeline.canonical_facts != case.expected_facts:
        errors.append("CANONICAL_FACTS_MISMATCH")

    if semantically_supported and case.expected_facts is not None:
        expected_response = DeterministicPlateIntentGate().evaluate(case.expected_facts)
        if not isinstance(expected_response.root, PlateIntentPayload):
            raise ValueError(f"invalid supported fixture: {case.case_id}")
        if pipeline.intent_response != expected_response:
            errors.append("INTENT_MISMATCH")
        expected_spec = CapabilityIntentToSpecCompiler().compile(
            expected_response.root.intent, design_id=pipeline.design_id,
        ).design_spec
        actual_spec = pipeline.spec_generation.design_spec if pipeline.spec_generation else None
        if actual_spec != expected_spec:
            errors.append("DESIGN_SPEC_MISMATCH")
        expected_plan = CapabilitySpecToPlanCompiler().compile(expected_spec).cad_plan
        actual_plan = pipeline.plan_generation.cad_plan if pipeline.plan_generation else None
        if actual_plan != expected_plan:
            errors.append("CAD_PLAN_MISMATCH")
        if pipeline.status != V02Status.VALIDATED:
            errors.append("PIPELINE_NOT_VALIDATED")
        if pipeline.r01_cad_status != CADStatus.SUCCESS:
            errors.append("CAD_NOT_SUCCESS")
        if pipeline.r01_validation_report is None or pipeline.r01_validation_report.status != ReportStatus.PASS:
            errors.append("VALIDATOR_NOT_PASS")
        if pipeline.final_revision_id != "R01":
            errors.append("FINAL_REVISION_NOT_R01")
        if not _artifact_exists(pipeline, "STEP"):
            errors.append("STEP_MISSING")
        if not _artifact_exists(pipeline, "STL"):
            errors.append("STL_MISSING")
    elif case.expected_class == PrototypeCaseClass.INSUFFICIENT:
        if pipeline.status != V02Status.FAIL or pipeline.canonical_facts is not None:
            errors.append("INSUFFICIENT_DID_NOT_STOP")
        if pipeline.r01_plan is not None or pipeline.r01_cad_status is not None:
            errors.append("CAD_REACHED_FOR_INSUFFICIENT")
    elif case.expected_class == PrototypeCaseClass.UNSUPPORTED:
        if pipeline.status != V02Status.UNSUPPORTED or pipeline.reason_code != "UNSUPPORTED_GEOMETRY":
            errors.append("UNSUPPORTED_OUTCOME_MISMATCH")
        if pipeline.r01_plan is not None or pipeline.r01_cad_status is not None:
            errors.append("CAD_REACHED_FOR_UNSUPPORTED")

    semantic_codes = {
        "CANONICAL_FACTS_MISMATCH", "INTENT_MISMATCH", "DESIGN_SPEC_MISMATCH",
        "CAD_PLAN_MISMATCH", "CLASSIFICATION_MISMATCH",
    }
    false_validated = pipeline.status == V02Status.VALIDATED and bool(semantic_codes.intersection(errors))
    model_id, prompt_version, prompt_tokens, completion_tokens, llm_latency, tps, schema_valid = _telemetry(pipeline)
    grounding_status = (
        pipeline.fact_grounding.status if pipeline.fact_grounding
        else "NOT_REQUIRED" if coverage == ExtractionCoverageStatus.COMPLETE
        else "NOT_RUN"
    )
    intent_status = pipeline.intent_response.root.status if pipeline.intent_response else "NOT_RUN"
    cad_status = pipeline.r01_cad_status.value if pipeline.r01_cad_status else "NOT_RUN"
    validator_status = (
        pipeline.r01_validation_report.status.value if pipeline.r01_validation_report else "NOT_RUN"
    )
    metrics = PrototypeCaseMetrics(
        case_id=case.case_id,
        classification=actual_class,
        coverage=coverage,
        extraction_mode=pipeline.extraction_mode,
        llm_invoked=pipeline.llm_invoked,
        unresolved_fields=unresolved,
        grounding_status=grounding_status,
        intent_status=intent_status,
        cad_status=cad_status,
        validator_status=validator_status,
        final_revision=pipeline.final_revision_id,
        step_generated=_artifact_exists(pipeline, "STEP"),
        stl_generated=_artifact_exists(pipeline, "STL"),
        total_latency_ms=total_latency_ms,
        model_id=model_id,
        prompt_version=prompt_version,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        llm_latency_ms=llm_latency,
        generation_tps=tps,
        schema_valid=schema_valid,
    )
    return PrototypeCaseResult(
        case_id=case.case_id,
        prompt=case.prompt,
        expected_class=case.expected_class,
        expected_llm_usage=case.expected_llm_usage,
        actual_class=actual_class,
        passed=not errors,
        false_validated=false_validated,
        semantic_errors=errors,
        pipeline_status=pipeline.status,
        pipeline_stage=pipeline.stage,
        pipeline_reason_code=pipeline.reason_code,
        actual_canonical_facts=pipeline.canonical_facts,
        metrics=metrics,
        pipeline_result_path=str(pipeline_result_path.resolve()),
        result_path=str(result_path.resolve()),
    )


def run_prototype_case(
    case: PrototypeCase,
    output_root: Path,
    *,
    backend_mode: str = "controlled",
    http_backend: HTTPInferenceBackend | None = None,
) -> PrototypeCaseResult:
    case_output = output_root / case.case_id
    pipeline_path = case_output / "pipeline_result.json"
    result_path = case_output / "result.json"
    if backend_mode == "controlled":
        backend = _controlled_backend(case)
    elif backend_mode == "http-llm":
        backend = http_backend if case.expected_llm_usage else _FailIfCalledBackend()
        if backend is None:
            raise ValueError("http_backend is required for http-llm mode")
    else:
        raise ValueError(f"unknown backend mode: {backend_mode}")
    started = time.perf_counter()
    pipeline = run_capability_case(
        "CP2-01",
        backend,
        case_output,
        user_prompt=case.prompt,
        result_json=pipeline_path,
    )
    result = evaluate_case_result(
        case,
        pipeline,
        total_latency_ms=(time.perf_counter() - started) * 1000.0,
        pipeline_result_path=pipeline_path,
        result_path=result_path,
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return result


def run_prototype_gate(
    cases: Iterable[PrototypeCase],
    output_root: Path,
    *,
    backend_mode: str = "controlled",
    http_backend: HTTPInferenceBackend | None = None,
) -> PrototypeGateSummary:
    started = time.perf_counter()
    selected = list(cases)
    results = [
        run_prototype_case(
            case, output_root, backend_mode=backend_mode, http_backend=http_backend,
        )
        for case in selected
    ]
    counts = {classification: sum(case.expected_class == classification for case in selected)
              for classification in PrototypeCaseClass}
    summary = PrototypeGateSummary(
        backend=backend_mode,
        model_id=http_backend.model_id if http_backend else None,
        total=len(results),
        passed=sum(result.passed for result in results),
        failed=sum(not result.passed for result in results),
        complete_count=counts[PrototypeCaseClass.COMPLETE],
        partial_count=counts[PrototypeCaseClass.PARTIAL],
        insufficient_count=counts[PrototypeCaseClass.INSUFFICIENT],
        unsupported_count=counts[PrototypeCaseClass.UNSUPPORTED],
        deterministic_success=sum(result.passed and not result.metrics.llm_invoked for result in results),
        hybrid_success=sum(result.passed and result.metrics.llm_invoked for result in results),
        grounding_error=sum(
            result.pipeline_reason_code == "GROUNDING_ERROR"
            or result.metrics.grounding_status == "GROUNDING_ERROR"
            for result in results
        ),
        intent_error=sum(
            "INTENT_MISMATCH" in result.semantic_errors
            or "CLASSIFICATION_MISMATCH" in result.semantic_errors
            for result in results
        ),
        cad_fail=sum(
            result.metrics.cad_status not in {"NOT_RUN", CADStatus.SUCCESS.value}
            for result in results
        ),
        validation_fail=sum(
            result.metrics.validator_status not in {"NOT_RUN", ReportStatus.PASS.value}
            for result in results
        ),
        false_validated=sum(result.false_validated for result in results),
        total_latency_ms=(time.perf_counter() - started) * 1000.0,
        results=results,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        summary.model_dump_json(indent=2) + "\n", encoding="utf-8",
    )
    return summary


def select_cases(
    *, classification: str = "all", case_ids: list[str] | None = None,
) -> list[PrototypeCase]:
    cases = prototype_cases()
    if classification != "all":
        wanted = PrototypeCaseClass(classification.upper())
        cases = [case for case in cases if case.expected_class == wanted]
    if case_ids:
        requested = set(case_ids)
        known = {case.case_id for case in cases}
        missing = sorted(requested - known)
        if missing:
            raise ValueError(f"unknown or filtered case ids: {missing}")
        cases = [case for case in cases if case.case_id in requested]
    return cases


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CAD AI Prototype Usability Gate V0.1")
    parser.add_argument(
        "--classification", default="all",
        choices=["all", "complete", "partial", "insufficient", "unsupported"],
    )
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--backend", choices=["controlled", "http-llm"], default="controlled")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--model-id", default=DEFAULT_M3B_MODEL_ID)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/prototype_gate_v01"))
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_cli_parser().parse_args()
    cases = select_cases(classification=args.classification, case_ids=args.case_ids)
    http_backend = None
    if args.backend == "http-llm":
        http_backend = HTTPInferenceBackend(m3b_http_config(
            base_url=args.base_url,
            endpoint=args.endpoint,
            model_id=args.model_id,
            timeout_seconds=args.timeout_seconds,
        ))
    summary = run_prototype_gate(
        cases, args.output, backend_mode=args.backend, http_backend=http_backend,
    )
    if args.json:
        print(summary.model_dump_json(indent=2))
    else:
        print(
            f"Prototype Usability Gate V0.1 | backend={summary.backend} | "
            f"passed={summary.passed}/{summary.total} | false_validated={summary.false_validated}"
        )
        for result in summary.results:
            errors = ",".join(result.semantic_errors) if result.semantic_errors else "-"
            print(
                f"{result.case_id} | expected={result.expected_class.value} | "
                f"actual={result.actual_class.value} | {'PASS' if result.passed else 'FAIL'} | {errors}"
            )
        print(f"summary: {(args.output / 'summary.json').resolve()}")
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
