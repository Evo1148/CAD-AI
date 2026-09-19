from __future__ import annotations

import json
import math
import time
from enum import Enum
from typing import Any, Literal, cast

from pydantic import Field, model_validator

from .cad import CADStatus
from .contracts import (
    CADOperation,
    CADPlan,
    Comparison,
    Constraint,
    DesignSpec,
    ExecutionMode,
    FeatureKind,
    Measurement,
    MeasurementType,
    OperationOutput,
    Origin,
    Priority,
    StrictModel,
    ensure_spec_plannable,
)
from .planning import (
    BackendFailureCode,
    BackendResponse,
    BackendRunMetadata,
    BackendStatus,
    InferenceBackend,
    InferenceRequest,
)
from .validation import ReportStatus, ValidationReport

M4_CONTEXT_VERSION = "0.1"
M4_PROMPT_VERSION = "m4-v0.1.2"
M4_V01_MEASUREMENTS = frozenset({
    MeasurementType.MODEL_VALID,
    MeasurementType.SOLID_COUNT,
    MeasurementType.EXTENT_X,
    MeasurementType.EXTENT_Y,
    MeasurementType.EXTENT_Z,
    MeasurementType.DIAMETER,
})


class PromptToSpecContext(StrictModel):
    contract_version: Literal["0.1"] = M4_CONTEXT_VERSION
    user_prompt: str = Field(min_length=1)
    design_id: str
    spec_version: str
    design_subset: Literal["RECTANGULAR_PLATE_SINGLE_CENTERED_THROUGH_HOLE"] = (
        "RECTANGULAR_PLATE_SINGLE_CENTERED_THROUGH_HOLE"
    )
    canonical_length_unit: Literal["mm"] = "mm"
    supported_geometry: Literal["RECTANGULAR_PLATE_WITH_ONE_CENTERED_THROUGH_HOLE"] = (
        "RECTANGULAR_PLATE_WITH_ONE_CENTERED_THROUGH_HOLE"
    )


class M4Intent(StrictModel):
    """The complete and only semantic decision delegated to the M4 V0.1 LLM."""

    contract_version: Literal["0.1"] = M4_CONTEXT_VERSION
    width: float = Field(gt=0, allow_inf_nan=False, strict=True)
    depth: float = Field(gt=0, allow_inf_nan=False, strict=True)
    height: float = Field(gt=0, allow_inf_nan=False, strict=True)
    hole_diameter: float = Field(gt=0, allow_inf_nan=False, strict=True)
    centered: Literal[True]


class SpecToPlanContext(StrictModel):
    contract_version: Literal["0.1"] = M4_CONTEXT_VERSION
    design_id: str
    spec_version: str
    revision_id: Literal["R01"] = "R01"
    design_spec: DesignSpec
    operation_catalog: dict[str, dict[str, Any]] = Field(default_factory=lambda: {
        "box": {
            "parameters": ["x", "y", "z"],
            "inputs": 0,
            "outputs": ["BODY"],
        },
        "hole": {
            "parameters": ["x", "y", "diameter"],
            "inputs": 1,
            "outputs": ["FEATURE", "DATUM_AXIS"],
            "meaning": "through hole along +Z through the current body",
        },
    })

    @model_validator(mode="after")
    def coherent_identity(self) -> "SpecToPlanContext":
        if self.design_spec.design_id != self.design_id or self.design_spec.spec_version != self.spec_version:
            raise ValueError("DesignSpec identity does not match planning context")
        return self


class M4InferenceRequest(StrictModel):
    """M4 request with the structural fields consumed by HTTPInferenceBackend."""

    system_prompt: str
    context: PromptToSpecContext | SpecToPlanContext
    response_schema: dict[str, Any]
    temperature: float = Field(default=0.0, ge=0)
    max_tokens: int = Field(default=1024, ge=1)
    seed: int | None = None


class GenerationStatus(str, Enum):
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"


class GenerationErrorCode(str, Enum):
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_ERROR = "MODEL_ERROR"
    OUTPUT_PARSE_ERROR = "OUTPUT_PARSE_ERROR"
    SCHEMA_VALIDATION_ERROR = "SCHEMA_VALIDATION_ERROR"
    GROUNDING_ERROR = "GROUNDING_ERROR"
    EXTRACTION_COVERAGE_INSUFFICIENT = "EXTRACTION_COVERAGE_INSUFFICIENT"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    BLOCKING_UNKNOWNS = "BLOCKING_UNKNOWNS"
    HARD_MEASUREMENT_UNSUPPORTED = "HARD_MEASUREMENT_UNSUPPORTED"
    OPERATION_NOT_ALLOWED = "OPERATION_NOT_ALLOWED"
    PARAMETER_SET_NOT_ALLOWED = "PARAMETER_SET_NOT_ALLOWED"
    INVALID_PARAMETER_VALUE = "INVALID_PARAMETER_VALUE"
    INVALID_EXECUTION_MODE = "INVALID_EXECUTION_MODE"
    SPEC_FEATURE_NOT_PRODUCED = "SPEC_FEATURE_NOT_PRODUCED"
    REQUIRED_CONSTRAINT_MISSING = "REQUIRED_CONSTRAINT_MISSING"
    INCOMPATIBLE_CONSTRAINTS = "INCOMPATIBLE_CONSTRAINTS"
    UNSUPPORTED_SPEC_SUBSET = "UNSUPPORTED_SPEC_SUBSET"
    SEMANTIC_FEATURE_REQUIRED = "SEMANTIC_FEATURE_REQUIRED"
    CAD_EXECUTION_ERROR = "CAD_EXECUTION_ERROR"
    VALIDATION_NOT_PASS = "VALIDATION_NOT_PASS"


class GenerationError(StrictModel):
    code: GenerationErrorCode
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class GenerationRun(StrictModel):
    stage: Literal["PROMPT_TO_INTENT", "INTENT_TO_SPEC", "PROMPT_TO_SPEC", "SPEC_TO_PLAN"]
    backend_type: str
    model_id: str | None = None
    context_version: str = M4_CONTEXT_VERSION
    prompt_version: str = M4_PROMPT_VERSION
    latency_ms: float = Field(ge=0)
    backend_status: BackendStatus
    schema_valid: bool
    backend_metadata: BackendRunMetadata | None = None


class DesignSpecGenerationResult(StrictModel):
    status: GenerationStatus
    design_spec: DesignSpec | None = None
    run: GenerationRun
    raw_output: str | None = None
    error: GenerationError | None = None

    @model_validator(mode="after")
    def status_invariants(self) -> "DesignSpecGenerationResult":
        if self.status == GenerationStatus.SUCCESS and (self.design_spec is None or self.error is not None):
            raise ValueError("successful DesignSpec generation requires output and no error")
        if self.status == GenerationStatus.ERROR and self.error is None:
            raise ValueError("failed DesignSpec generation requires error")
        return self


class M4IntentGenerationResult(StrictModel):
    status: GenerationStatus
    intent: M4Intent | None = None
    run: GenerationRun
    raw_output: str | None = None
    error: GenerationError | None = None

    @model_validator(mode="after")
    def status_invariants(self) -> "M4IntentGenerationResult":
        if self.status == GenerationStatus.SUCCESS and (self.intent is None or self.error is not None):
            raise ValueError("successful M4Intent generation requires output and no error")
        if self.status == GenerationStatus.ERROR and self.error is None:
            raise ValueError("failed M4Intent generation requires error")
        return self


class CADPlanGenerationResult(StrictModel):
    status: GenerationStatus
    cad_plan: CADPlan | None = None
    run: GenerationRun
    raw_output: str | None = None
    error: GenerationError | None = None

    @model_validator(mode="after")
    def status_invariants(self) -> "CADPlanGenerationResult":
        if self.status == GenerationStatus.SUCCESS and (self.cad_plan is None or self.error is not None):
            raise ValueError("successful CADPlan generation requires output and no error")
        if self.status == GenerationStatus.ERROR and self.error is None:
            raise ValueError("failed CADPlan generation requires error")
        return self


class M4Status(str, Enum):
    VALIDATED = "VALIDATED"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"


class M4GenerationResult(StrictModel):
    case_id: str
    status: M4Status
    stage: str
    user_prompt: str
    design_id: str
    spec_version: str
    revision_id: Literal["R01"] = "R01"
    intent_generation: M4IntentGenerationResult | None = None
    spec_generation: DesignSpecGenerationResult | None = None
    plan_generation: CADPlanGenerationResult | None = None
    cad_status: CADStatus | None = None
    validation_report: ValidationReport | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    feature_ids: list[str] = Field(default_factory=list)
    error: GenerationError | None = None


M4_INTENT_SYSTEM_PROMPT = """Interpret the supplied user request for the single supported M4 V0.1 geometry.
Extract only width, depth, height, hole_diameter, and whether the through hole is centered.
All dimensions must be positive numbers in millimeters. Do not create CAD operations, identifiers,
constraints, measurements, priorities, comparisons, features, or other internal design metadata.
Return only JSON matching the supplied M4Intent schema. Do not emit code or commentary.
"""


CAD_PLAN_SYSTEM_PROMPT = """Convert the supplied validated DesignSpec into one executable CADPlan JSON object.
Use exactly the supplied design_id, spec_version, and revision_id R01.
Use only CONTROLLED operations and only operation types and exact parameter names present in operation_catalog.
All parameter values must be fully resolved numeric values in canonical units.
Create stable semantic feature IDs needed by the DesignSpec. Return only JSON matching the supplied CADPlan schema.
Do not emit code, commentary, generated operations, or unsupported operation types.
"""


class LLMIntentGenerator:
    def __init__(
        self,
        backend: InferenceBackend,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = 12345,
    ):
        self.backend = backend
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

    def generate(self, context: PromptToSpecContext) -> M4IntentGenerationResult:
        request = M4InferenceRequest(
            system_prompt=M4_INTENT_SYSTEM_PROMPT,
            context=context,
            response_schema=M4Intent.model_json_schema(),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        response, run = _invoke(self.backend, request, "PROMPT_TO_INTENT")
        if response.status == BackendStatus.ERROR:
            return M4IntentGenerationResult(
                status=GenerationStatus.ERROR,
                run=run,
                error=_backend_error(response),
            )
        assert response.raw_output is not None
        parsed, error = _parse_json(response.raw_output)
        if error:
            return M4IntentGenerationResult(
                status=GenerationStatus.ERROR,
                run=run,
                raw_output=response.raw_output,
                error=error,
            )
        try:
            intent = M4Intent.model_validate(parsed)
        except Exception as exc:
            return M4IntentGenerationResult(
                status=GenerationStatus.ERROR,
                run=run,
                raw_output=response.raw_output,
                error=GenerationError(code=GenerationErrorCode.SCHEMA_VALIDATION_ERROR, message=str(exc)),
            )
        return M4IntentGenerationResult(
            status=GenerationStatus.SUCCESS,
            intent=intent,
            run=run.model_copy(update={"schema_valid": True}),
            raw_output=response.raw_output,
        )


class DeterministicIntentToSpecCompiler:
    """Build the canonical M4 V0.1 DesignSpec without delegating metadata to an LLM."""

    def compile(
        self,
        intent: M4Intent,
        *,
        design_id: str,
        spec_version: str,
    ) -> DesignSpecGenerationResult:
        started = time.perf_counter()

        def run(schema_valid: bool) -> GenerationRun:
            return GenerationRun(
                stage="INTENT_TO_SPEC",
                backend_type="deterministic-compiler",
                model_id=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                backend_status=BackendStatus.SUCCESS,
                schema_valid=schema_valid,
            )

        if intent.hole_diameter >= min(intent.width, intent.depth):
            return DesignSpecGenerationResult(
                status=GenerationStatus.ERROR,
                run=run(True),
                error=GenerationError(
                    code=GenerationErrorCode.INVALID_PARAMETER_VALUE,
                    message="centered hole diameter must be smaller than both plate extents",
                    details={
                        "diameter": intent.hole_diameter,
                        "width": intent.width,
                        "depth": intent.depth,
                    },
                ),
            )

        def constraint(
            constraint_id: str,
            measurement_type: MeasurementType,
            target: float | bool,
            *,
            origin: Origin,
            feature_id: str | None = None,
            tolerance: float = 0.01,
        ) -> Constraint:
            return Constraint(
                constraint_id=constraint_id,
                priority=Priority.HARD,
                origin=origin,
                measurement=Measurement(type=measurement_type, feature_id=feature_id),
                comparison=Comparison.EQ,
                target=target,
                tolerance=tolerance,
            )

        spec = DesignSpec(
            design_id=design_id,
            spec_version=spec_version,
            parameters={
                "width": intent.width,
                "depth": intent.depth,
                "height": intent.height,
                "hole_diameter": intent.hole_diameter,
                "centered": intent.centered,
            },
            constraints=[
                constraint(
                    "C_VALID",
                    MeasurementType.MODEL_VALID,
                    True,
                    origin=Origin.SYSTEM,
                    tolerance=0.0,
                ),
                constraint(
                    "C_SOLID_COUNT",
                    MeasurementType.SOLID_COUNT,
                    1.0,
                    origin=Origin.DERIVED,
                    tolerance=0.0,
                ),
                constraint("C_WIDTH", MeasurementType.EXTENT_X, intent.width, origin=Origin.USER),
                constraint("C_DEPTH", MeasurementType.EXTENT_Y, intent.depth, origin=Origin.USER),
                constraint("C_HEIGHT", MeasurementType.EXTENT_Z, intent.height, origin=Origin.USER),
                constraint(
                    "C_HOLE_DIAMETER",
                    MeasurementType.DIAMETER,
                    intent.hole_diameter,
                    origin=Origin.USER,
                    feature_id="mount_hole",
                ),
            ],
        )
        try:
            ensure_spec_plannable(spec)
        except ValueError as exc:
            return DesignSpecGenerationResult(
                status=GenerationStatus.ERROR,
                run=run(True),
                error=GenerationError(code=GenerationErrorCode.BLOCKING_UNKNOWNS, message=str(exc)),
            )
        _, _, subset_error = _extract_m4_compiler_inputs(spec)
        if subset_error:
            return DesignSpecGenerationResult(
                status=GenerationStatus.ERROR,
                run=run(True),
                error=subset_error,
            )
        return DesignSpecGenerationResult(
            status=GenerationStatus.SUCCESS,
            design_spec=spec,
            run=run(True),
        )


class LLMDesignSpecGenerator:
    """Compatibility facade: LLM M4Intent followed by deterministic DesignSpec compilation."""

    def __init__(
        self,
        backend: InferenceBackend,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = 12345,
    ):
        self.intent_generator = LLMIntentGenerator(
            backend,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )

    def generate(self, context: PromptToSpecContext) -> DesignSpecGenerationResult:
        generated = self.intent_generator.generate(context)
        if generated.status != GenerationStatus.SUCCESS or generated.intent is None:
            return DesignSpecGenerationResult(
                status=GenerationStatus.ERROR,
                run=generated.run,
                raw_output=generated.raw_output,
                error=generated.error,
            )
        compiled = DeterministicIntentToSpecCompiler().compile(
            generated.intent,
            design_id=context.design_id,
            spec_version=context.spec_version,
        )
        return compiled.model_copy(update={
            "raw_output": generated.raw_output,
            "run": generated.run,
        })


class LLMCADPlanGenerator:
    def __init__(
        self,
        backend: InferenceBackend,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = 12345,
    ):
        self.backend = backend
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

    def generate(self, context: SpecToPlanContext) -> CADPlanGenerationResult:
        request = M4InferenceRequest(
            system_prompt=CAD_PLAN_SYSTEM_PROMPT,
            context=context,
            response_schema=CADPlan.model_json_schema(),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        response, run = _invoke(self.backend, request, "SPEC_TO_PLAN")
        if response.status == BackendStatus.ERROR:
            return CADPlanGenerationResult(
                status=GenerationStatus.ERROR,
                run=run,
                error=_backend_error(response),
            )
        assert response.raw_output is not None
        parsed, error = _parse_json(response.raw_output)
        if error:
            return CADPlanGenerationResult(
                status=GenerationStatus.ERROR,
                run=run,
                raw_output=response.raw_output,
                error=error,
            )
        try:
            plan = CADPlan.model_validate(parsed)
        except Exception as exc:
            return CADPlanGenerationResult(
                status=GenerationStatus.ERROR,
                run=run,
                raw_output=response.raw_output,
                error=GenerationError(code=GenerationErrorCode.SCHEMA_VALIDATION_ERROR, message=str(exc)),
            )
        semantic_error = validate_generated_plan(plan, context)
        if semantic_error:
            return CADPlanGenerationResult(
                status=GenerationStatus.ERROR,
                run=run.model_copy(update={"schema_valid": True}),
                raw_output=response.raw_output,
                error=semantic_error,
            )
        return CADPlanGenerationResult(
            status=GenerationStatus.SUCCESS,
            cad_plan=plan,
            run=run.model_copy(update={"schema_valid": True}),
            raw_output=response.raw_output,
        )


class DeterministicSpecToPlanCompiler:
    """Compile the single M4 V0.1 rectangular-plate subset without an LLM."""

    supported_hard_measurements = M4_V01_MEASUREMENTS

    def compile(self, context: SpecToPlanContext) -> CADPlanGenerationResult:
        started = time.perf_counter()

        def run(schema_valid: bool) -> GenerationRun:
            return GenerationRun(
                stage="SPEC_TO_PLAN",
                backend_type="deterministic-compiler",
                model_id=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                backend_status=BackendStatus.SUCCESS,
                schema_valid=schema_valid,
            )

        resolved, hole_feature_id, subset_error = _extract_m4_compiler_inputs(context.design_spec)
        if subset_error:
            return CADPlanGenerationResult(
                status=GenerationStatus.ERROR,
                run=run(True),
                error=subset_error,
            )
        assert resolved is not None and hole_feature_id is not None

        plan = CADPlan(
            design_id=context.design_id,
            spec_version=context.spec_version,
            revision_id="R01",
            operations=[
                CADOperation(
                    operation_id="OP01",
                    operation_type="box",
                    params={
                        "x": resolved[MeasurementType.EXTENT_X],
                        "y": resolved[MeasurementType.EXTENT_Y],
                        "z": resolved[MeasurementType.EXTENT_Z],
                    },
                    outputs=[OperationOutput(
                        feature_id="main_body",
                        kind=FeatureKind.BODY,
                        semantic_role="primary solid",
                    )],
                ),
                CADOperation(
                    operation_id="OP02",
                    operation_type="hole",
                    inputs=["main_body"],
                    params={
                        "x": 0.0,
                        "y": 0.0,
                        "diameter": resolved[MeasurementType.DIAMETER],
                    },
                    outputs=[
                        OperationOutput(feature_id=hole_feature_id, kind=FeatureKind.FEATURE),
                        OperationOutput(
                            feature_id=f"{hole_feature_id}_axis",
                            kind=FeatureKind.DATUM_AXIS,
                        ),
                    ],
                ),
            ],
        )
        semantic_error = validate_generated_plan(plan, context)
        if semantic_error:
            return CADPlanGenerationResult(
                status=GenerationStatus.ERROR,
                run=run(True),
                error=semantic_error,
            )
        return CADPlanGenerationResult(
            status=GenerationStatus.SUCCESS,
            cad_plan=plan,
            run=run(True),
        )


def _extract_m4_compiler_inputs(
    spec: DesignSpec,
) -> tuple[dict[MeasurementType, float] | None, str | None, GenerationError | None]:
    unsupported = sorted({
        constraint.measurement.type.value
        for constraint in spec.constraints
        if constraint.measurement.type not in M4_V01_MEASUREMENTS
    })
    if unsupported:
        return None, None, GenerationError(
            code=GenerationErrorCode.UNSUPPORTED_SPEC_SUBSET,
            message="DesignSpec contains measurements outside the M4 V0.1 subset",
            details={"measurements": unsupported},
        )

    non_hard = [
        constraint.constraint_id
        for constraint in spec.constraints
        if constraint.measurement.type in M4_V01_MEASUREMENTS
        and constraint.priority != Priority.HARD
    ]
    if non_hard:
        return None, None, GenerationError(
            code=GenerationErrorCode.UNSUPPORTED_SPEC_SUBSET,
            message="M4 V0.1 requires its exposed measurements to be HARD",
            details={"constraint_ids": non_hard},
        )

    reference_error = _validate_measurement_references(spec)
    if reference_error:
        return None, None, reference_error

    fixed_error = _validate_fixed_subset_requirements(spec)
    if fixed_error:
        return None, None, fixed_error

    resolved: dict[MeasurementType, float] = {}
    for measurement in (
        MeasurementType.EXTENT_X,
        MeasurementType.EXTENT_Y,
        MeasurementType.EXTENT_Z,
        MeasurementType.DIAMETER,
    ):
        value, error = _resolve_hard_eq_target(spec, measurement)
        if error:
            return None, None, error
        assert value is not None
        resolved[measurement] = value

    diameter_constraints = [
        constraint
        for constraint in spec.constraints
        if constraint.measurement.type == MeasurementType.DIAMETER
    ]
    feature_ids = {constraint.measurement.feature_id for constraint in diameter_constraints}
    if len(feature_ids) != 1 or None in feature_ids:
        return None, None, GenerationError(
            code=GenerationErrorCode.SEMANTIC_FEATURE_REQUIRED,
            message="the single HARD DIAMETER requirement must reference one semantic feature",
        )
    hole_feature_id = next(iter(feature_ids))
    if not hole_feature_id.strip() or hole_feature_id == "main_body":
        return None, None, GenerationError(
            code=GenerationErrorCode.SEMANTIC_FEATURE_REQUIRED,
            message="hole feature_id must be non-empty and distinct from main_body",
        )

    diameter = resolved[MeasurementType.DIAMETER]
    width = resolved[MeasurementType.EXTENT_X]
    depth = resolved[MeasurementType.EXTENT_Y]
    if diameter >= min(width, depth):
        return None, None, GenerationError(
            code=GenerationErrorCode.INVALID_PARAMETER_VALUE,
            message="centered hole diameter must be smaller than both plate extents",
            details={"diameter": diameter, "extent_x": width, "extent_y": depth},
        )
    return resolved, hole_feature_id, None


def _resolve_hard_eq_target(
    spec: DesignSpec,
    measurement: MeasurementType,
) -> tuple[float | None, GenerationError | None]:
    candidates = [
        constraint
        for constraint in spec.constraints
        if constraint.priority == Priority.HARD and constraint.measurement.type == measurement
    ]
    if not candidates:
        return None, GenerationError(
            code=GenerationErrorCode.REQUIRED_CONSTRAINT_MISSING,
            message=f"missing required HARD {measurement.value} constraint",
            details={"measurement": measurement.value},
        )
    if any(
        constraint.comparison != Comparison.EQ
        or isinstance(constraint.target, bool)
        or not isinstance(constraint.target, (int, float))
        or not math.isfinite(float(constraint.target))
        for constraint in candidates
    ):
        return None, GenerationError(
            code=GenerationErrorCode.UNSUPPORTED_SPEC_SUBSET,
            message=f"{measurement.value} must be expressed as a numeric HARD EQ constraint",
            details={"constraint_ids": [item.constraint_id for item in candidates]},
        )

    lower = max(float(item.target) - item.tolerance for item in candidates)
    upper = min(float(item.target) + item.tolerance for item in candidates)
    if lower > upper:
        return None, GenerationError(
            code=GenerationErrorCode.INCOMPATIBLE_CONSTRAINTS,
            message=f"multiple HARD {measurement.value} constraints are incompatible",
            details={"constraint_ids": [item.constraint_id for item in candidates]},
        )
    value = float(candidates[0].target) if len(candidates) == 1 else (lower + upper) / 2.0
    if value <= 0:
        return None, GenerationError(
            code=GenerationErrorCode.INVALID_PARAMETER_VALUE,
            message=f"{measurement.value} must resolve to a positive value",
            details={"value": value},
        )
    return value, None


def _validate_fixed_subset_requirements(spec: DesignSpec) -> GenerationError | None:
    for measurement in (MeasurementType.MODEL_VALID, MeasurementType.SOLID_COUNT):
        candidates = [
            constraint
            for constraint in spec.constraints
            if constraint.priority == Priority.HARD and constraint.measurement.type == measurement
        ]
        if not candidates:
            return GenerationError(
                code=GenerationErrorCode.REQUIRED_CONSTRAINT_MISSING,
                message=f"missing required HARD {measurement.value} constraint",
                details={"measurement": measurement.value},
            )
        valid_target = all(
            constraint.comparison == Comparison.EQ
            and (
                constraint.target is True
                if measurement == MeasurementType.MODEL_VALID
                else not isinstance(constraint.target, bool) and constraint.target == 1
            )
            for constraint in candidates
        )
        if not valid_target:
            return GenerationError(
                code=GenerationErrorCode.UNSUPPORTED_SPEC_SUBSET,
                message=f"{measurement.value} conflicts with the single-solid M4 V0.1 subset",
                details={"constraint_ids": [item.constraint_id for item in candidates]},
            )
    return None


def _validate_measurement_references(spec: DesignSpec) -> GenerationError | None:
    for constraint in spec.constraints:
        measurement = constraint.measurement
        if measurement.type == MeasurementType.DIAMETER:
            if measurement.feature_id is None or measurement.reference_feature_id is not None:
                return GenerationError(
                    code=GenerationErrorCode.SEMANTIC_FEATURE_REQUIRED,
                    message="DIAMETER requires exactly one feature_id and no reference_feature_id",
                    details={"constraint_id": constraint.constraint_id},
                )
        elif measurement.feature_id is not None or measurement.reference_feature_id is not None:
            return GenerationError(
                code=GenerationErrorCode.UNSUPPORTED_SPEC_SUBSET,
                message=f"{measurement.type.value} is global in M4 V0.1 and cannot reference features",
                details={"constraint_id": constraint.constraint_id},
            )
    return None


def validate_generated_plan(plan: CADPlan, context: SpecToPlanContext) -> GenerationError | None:
    if (
        plan.design_id != context.design_id
        or plan.spec_version != context.spec_version
        or plan.revision_id != "R01"
        or plan.parent_revision_id is not None
    ):
        return GenerationError(
            code=GenerationErrorCode.IDENTITY_MISMATCH,
            message="generated CADPlan does not preserve deterministic design/spec/revision identity",
        )

    allowed_parameters = {
        operation_type: set(definition["parameters"])
        for operation_type, definition in context.operation_catalog.items()
    }
    for operation in plan.operations:
        if operation.execution_mode != ExecutionMode.CONTROLLED:
            return GenerationError(
                code=GenerationErrorCode.INVALID_EXECUTION_MODE,
                message=f"operation {operation.operation_id} is not CONTROLLED",
            )
        if operation.operation_type not in allowed_parameters:
            return GenerationError(
                code=GenerationErrorCode.OPERATION_NOT_ALLOWED,
                message=f"operation type is outside the M4 V0.1 catalog: {operation.operation_type}",
                details={"operation_id": operation.operation_id},
            )
        actual_parameters = set(operation.params)
        if actual_parameters != allowed_parameters[operation.operation_type]:
            return GenerationError(
                code=GenerationErrorCode.PARAMETER_SET_NOT_ALLOWED,
                message=f"operation {operation.operation_id} has an invalid parameter set",
                details={
                    "expected": sorted(allowed_parameters[operation.operation_type]),
                    "actual": sorted(actual_parameters),
                },
            )
        for name, value in operation.params.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                return GenerationError(
                    code=GenerationErrorCode.INVALID_PARAMETER_VALUE,
                    message=f"{operation.operation_id}.params.{name} must be a finite number",
                )
        positive_names = {"x", "y", "z"} if operation.operation_type == "box" else {"diameter"}
        if any(float(operation.params[name]) <= 0 for name in positive_names):
            return GenerationError(
                code=GenerationErrorCode.INVALID_PARAMETER_VALUE,
                message=f"operation {operation.operation_id} contains a non-positive size",
            )

    produced_features = {
        output.feature_id
        for operation in plan.operations
        for output in operation.outputs
    }
    required_features = {
        feature_id
        for constraint in context.design_spec.constraints
        for feature_id in (
            constraint.measurement.feature_id,
            constraint.measurement.reference_feature_id,
        )
        if feature_id is not None
    }
    missing = sorted(required_features - produced_features)
    if missing:
        return GenerationError(
            code=GenerationErrorCode.SPEC_FEATURE_NOT_PRODUCED,
            message="CADPlan does not produce semantic features required by DesignSpec",
            details={"feature_ids": missing},
        )
    return None


def default_prompt_to_spec_context(user_prompt: str, design_id: str, spec_version: str) -> PromptToSpecContext:
    return PromptToSpecContext(
        user_prompt=user_prompt,
        design_id=design_id,
        spec_version=spec_version,
    )


def _invoke(
    backend: InferenceBackend,
    request: M4InferenceRequest,
    stage: Literal["PROMPT_TO_INTENT", "INTENT_TO_SPEC", "PROMPT_TO_SPEC", "SPEC_TO_PLAN"],
) -> tuple[BackendResponse, GenerationRun]:
    started = time.perf_counter()
    # HTTPInferenceBackend consumes these fields structurally; M3 contracts remain unchanged.
    response = backend.generate(cast(InferenceRequest, request))
    run = GenerationRun(
        stage=stage,
        backend_type=backend.backend_type,
        model_id=backend.model_id,
        latency_ms=(time.perf_counter() - started) * 1000,
        backend_status=response.status,
        schema_valid=False,
        backend_metadata=response.metadata,
    )
    return response, run


def _parse_json(raw_output: str) -> tuple[Any | None, GenerationError | None]:
    try:
        return json.loads(raw_output), None
    except json.JSONDecodeError as exc:
        return None, GenerationError(code=GenerationErrorCode.OUTPUT_PARSE_ERROR, message=str(exc))


def _backend_error(response: BackendResponse) -> GenerationError:
    code_map = {
        BackendFailureCode.MODEL_TIMEOUT: GenerationErrorCode.MODEL_TIMEOUT,
        BackendFailureCode.MODEL_UNAVAILABLE: GenerationErrorCode.MODEL_UNAVAILABLE,
        BackendFailureCode.MODEL_ERROR: GenerationErrorCode.MODEL_ERROR,
    }
    assert response.error_code is not None
    return GenerationError(code=code_map[response.error_code], message=response.message or response.error_code.value)


def final_result_error(code: GenerationErrorCode, message: str, **details: Any) -> GenerationError:
    return GenerationError(code=code, message=message, details=details)


def final_status_from_report(report: ValidationReport) -> M4Status:
    return M4Status.VALIDATED if report.status == ReportStatus.PASS else M4Status.FAIL
