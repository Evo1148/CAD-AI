import pytest

from cad_ai.cad import CADEngine
from cad_ai.contracts import (
    CADOperation,
    CADPlan,
    Comparison,
    Constraint,
    DesignSpec,
    FeatureKind,
    Measurement,
    MeasurementType,
    OperationOutput,
    Origin,
    Priority,
)
from cad_ai.generation import (
    DeterministicIntentToSpecCompiler,
    DeterministicSpecToPlanCompiler,
    GenerationErrorCode,
    GenerationStatus,
    LLMCADPlanGenerator,
    LLMIntentGenerator,
    M4Intent,
    M4Status,
    PromptToSpecContext,
    SpecToPlanContext,
    default_prompt_to_spec_context,
)
from cad_ai.m4 import M4_001_PROMPT, run_m4_001
from cad_ai.planning import ScriptedFakeBackend
from cad_ai.validation import CheckStatus, ReportStatus, Validator


def _constraint(
    constraint_id: str,
    measurement: MeasurementType,
    target: float | bool,
    *,
    feature_id: str | None = None,
    tolerance: float = 0.01,
) -> Constraint:
    return Constraint(
        constraint_id=constraint_id,
        priority=Priority.HARD,
        origin=Origin.USER,
        measurement=Measurement(type=measurement, feature_id=feature_id),
        comparison=Comparison.EQ,
        target=target,
        tolerance=tolerance,
    )


def _m4_spec(
    width: float = 60.0,
    depth: float = 40.0,
    height: float = 4.0,
    diameter: float = 6.0,
) -> DesignSpec:
    return DesignSpec(
        design_id="M4-001",
        spec_version="1.0",
        parameters={"width": width, "depth": depth, "height": height, "hole_diameter": diameter},
        constraints=[
            _constraint("C_VALID", MeasurementType.MODEL_VALID, True, tolerance=0),
            _constraint("C_SOLID_COUNT", MeasurementType.SOLID_COUNT, 1, tolerance=0),
            _constraint("C_WIDTH", MeasurementType.EXTENT_X, width),
            _constraint("C_DEPTH", MeasurementType.EXTENT_Y, depth),
            _constraint("C_HEIGHT", MeasurementType.EXTENT_Z, height),
            _constraint("C_HOLE_DIAMETER", MeasurementType.DIAMETER, diameter, feature_id="mount_hole"),
        ],
    )


def _m4_plan(operation_type: str = "hole") -> CADPlan:
    return CADPlan(
        design_id="M4-001",
        spec_version="1.0",
        revision_id="R01",
        operations=[
            CADOperation(
                operation_id="OP01",
                operation_type="box",
                params={"x": 60.0, "y": 40.0, "z": 4.0},
                outputs=[OperationOutput(
                    feature_id="main_body",
                    kind=FeatureKind.BODY,
                    semantic_role="primary solid",
                )],
            ),
            CADOperation(
                operation_id="OP02",
                operation_type=operation_type,
                inputs=["main_body"],
                params={"x": 0.0, "y": 0.0, "diameter": 6.0},
                outputs=[
                    OperationOutput(feature_id="mount_hole", kind=FeatureKind.FEATURE),
                    OperationOutput(feature_id="mount_hole_axis", kind=FeatureKind.DATUM_AXIS),
                ],
            ),
        ],
    )


def _intent(
    width: float = 60.0,
    depth: float = 40.0,
    height: float = 4.0,
    diameter: float = 6.0,
) -> M4Intent:
    return M4Intent(
        width=width,
        depth=depth,
        height=height,
        hole_diameter=diameter,
        centered=True,
    )


def test_m4_001_prompt_to_real_cad_end_to_end(tmp_path):
    plan = _m4_plan()
    intent = _intent()
    backend = ScriptedFakeBackend([intent.model_dump_json()])

    result = run_m4_001(backend, tmp_path)

    assert result.status == M4Status.VALIDATED
    assert result.stage == "COMPLETE"
    assert result.intent_generation.intent == intent
    assert result.spec_generation.design_spec.parameters == {
        "width": 60.0,
        "depth": 40.0,
        "height": 4.0,
        "hole_diameter": 6.0,
        "centered": True,
    }
    assert result.plan_generation.cad_plan == plan
    assert result.cad_status.value == "SUCCESS"
    assert result.validation_report.status == ReportStatus.PASS
    assert all(check.status == CheckStatus.PASS for check in result.validation_report.checks)
    assert result.feature_ids == ["main_body", "mount_hole", "mount_hole_axis"]
    assert set(result.artifacts) == {"STEP", "STL"}
    assert all(tmp_path.joinpath("R01", f"M4-001_R01.{extension}").exists() for extension in ("step", "stl"))
    assert (tmp_path / "result.json").exists()
    assert len(backend.requests) == 1
    assert isinstance(backend.requests[0].context, PromptToSpecContext)
    assert backend.requests[0].context.design_subset == "RECTANGULAR_PLATE_SINGLE_CENTERED_THROUGH_HOLE"
    assert backend.requests[0].context.supported_geometry == "RECTANGULAR_PLATE_WITH_ONE_CENTERED_THROUGH_HOLE"
    assert backend.requests[0].response_schema == M4Intent.model_json_schema()
    assert result.spec_generation.run.backend_type == "deterministic-compiler"
    assert result.plan_generation.run.backend_type == "deterministic-compiler"


def test_fake_prompt_generates_only_valid_m4_intent():
    intent = _intent(width=72.0, depth=31.0, height=5.0, diameter=8.0)
    backend = ScriptedFakeBackend([intent.model_dump_json()])
    context = default_prompt_to_spec_context("A plate with one centered hole", "D-ALT", "3.0")

    result = LLMIntentGenerator(backend).generate(context)

    assert result.status == GenerationStatus.SUCCESS
    assert result.intent == intent
    assert backend.requests[0].response_schema == M4Intent.model_json_schema()
    assert set(result.intent.model_dump()) == {
        "contract_version", "width", "depth", "height", "hole_diameter", "centered",
    }


def test_intent_compiles_to_canonical_design_spec():
    result = DeterministicIntentToSpecCompiler().compile(
        _intent(),
        design_id="M4-001",
        spec_version="1.0",
    )

    assert result.status == GenerationStatus.SUCCESS
    spec = result.design_spec
    assert spec.design_id == "M4-001"
    assert spec.spec_version == "1.0"
    assert spec.assumptions == []
    assert spec.unknowns == []
    assert [item.constraint_id for item in spec.constraints] == [
        "C_VALID",
        "C_SOLID_COUNT",
        "C_WIDTH",
        "C_DEPTH",
        "C_HEIGHT",
        "C_HOLE_DIAMETER",
    ]
    assert all(item.priority == Priority.HARD for item in spec.constraints)
    assert all(item.comparison == Comparison.EQ for item in spec.constraints)
    by_id = {item.constraint_id: item for item in spec.constraints}
    assert by_id["C_VALID"].origin == Origin.SYSTEM
    assert by_id["C_VALID"].measurement == Measurement(type=MeasurementType.MODEL_VALID)
    assert by_id["C_VALID"].target is True
    assert by_id["C_SOLID_COUNT"].origin == Origin.DERIVED
    assert by_id["C_SOLID_COUNT"].target == 1.0
    assert by_id["C_HOLE_DIAMETER"].measurement == Measurement(
        type=MeasurementType.DIAMETER,
        feature_id="mount_hole",
    )


def test_alternative_intent_values_reach_real_cad_and_validator(tmp_path):
    intent = _intent(width=72.0, depth=31.0, height=5.0, diameter=8.0)
    backend = ScriptedFakeBackend([intent.model_dump_json()])

    result = run_m4_001(
        backend,
        tmp_path,
        user_prompt="Create a 72 x 31 x 5 mm plate with one centered 8 mm through hole.",
    )

    assert result.status == M4Status.VALIDATED
    assert result.plan_generation.cad_plan.operations[0].params == {"x": 72.0, "y": 31.0, "z": 5.0}
    assert result.plan_generation.cad_plan.operations[1].params == {
        "x": 0.0,
        "y": 0.0,
        "diameter": 8.0,
    }
    assert result.validation_report.status == ReportStatus.PASS
    assert all(tmp_path.joinpath("R01", f"M4-001_R01.{extension}").exists() for extension in ("step", "stl"))


def test_deterministic_compiler_derives_all_dimensions_from_hard_constraints():
    spec = _m4_spec(width=72.0, depth=31.0, height=5.0, diameter=8.0)
    context = SpecToPlanContext(
        design_id="M4-001",
        spec_version="1.0",
        design_spec=spec,
    )

    result = DeterministicSpecToPlanCompiler().compile(context)

    assert result.status == GenerationStatus.SUCCESS
    assert result.cad_plan.operations[0].params == {"x": 72.0, "y": 31.0, "z": 5.0}
    assert result.cad_plan.operations[1].params == {"x": 0.0, "y": 0.0, "diameter": 8.0}
    assert [operation.operation_type for operation in result.cad_plan.operations] == ["box", "hole"]
    assert result.cad_plan.operations[1].inputs == ["main_body"]
    assert result.cad_plan.operations[1].outputs[0].feature_id == "mount_hole"
    assert result.cad_plan.operations[1].outputs[1].feature_id == "mount_hole_axis"


def test_m4_contract_is_closed_from_llm_context_through_engine_and_validator():
    prompt_context = default_prompt_to_spec_context(M4_001_PROMPT, "M4-001", "1.0")
    generated = DeterministicIntentToSpecCompiler().compile(
        _intent(), design_id="M4-001", spec_version="1.0"
    )
    spec = generated.design_spec
    plan_context = SpecToPlanContext(design_id="M4-001", spec_version="1.0", design_spec=spec)

    result = DeterministicSpecToPlanCompiler().compile(plan_context)

    hard_spec_measurements = {
        item.measurement.type for item in spec.constraints if item.priority == Priority.HARD
    }
    assert set(prompt_context.model_dump()) == {
        "contract_version",
        "user_prompt",
        "design_id",
        "spec_version",
        "design_subset",
        "canonical_length_unit",
        "supported_geometry",
    }
    assert hard_spec_measurements == DeterministicSpecToPlanCompiler.supported_hard_measurements
    assert hard_spec_measurements <= Validator.implemented
    assert result.status == GenerationStatus.SUCCESS
    for operation in result.cad_plan.operations:
        assert operation.operation_type in CADEngine.supported_operations
        assert set(operation.params) == set(
            plan_context.operation_catalog[operation.operation_type]["parameters"]
        )
    assert {operation.operation_type for operation in result.cad_plan.operations} == {"box", "hole"}


@pytest.mark.parametrize("missing_measurement", [
    MeasurementType.MODEL_VALID,
    MeasurementType.SOLID_COUNT,
    MeasurementType.EXTENT_X,
    MeasurementType.EXTENT_Y,
    MeasurementType.EXTENT_Z,
    MeasurementType.DIAMETER,
])
def test_deterministic_compiler_rejects_every_missing_required_constraint(missing_measurement):
    spec = _m4_spec()
    spec = spec.model_copy(update={
        "constraints": [item for item in spec.constraints if item.measurement.type != missing_measurement],
    })
    context = SpecToPlanContext(design_id="M4-001", spec_version="1.0", design_spec=spec)

    result = DeterministicSpecToPlanCompiler().compile(context)

    assert result.status == GenerationStatus.ERROR
    assert result.error.code == GenerationErrorCode.REQUIRED_CONSTRAINT_MISSING


def test_invalid_intent_output_stops_with_structured_error():
    backend = ScriptedFakeBackend(['{"width":60,"depth":40}'])
    context = default_prompt_to_spec_context(M4_001_PROMPT, "M4-001", "1.0")

    result = LLMIntentGenerator(backend).generate(context)

    assert result.status == GenerationStatus.ERROR
    assert result.intent is None
    assert result.error.code == GenerationErrorCode.SCHEMA_VALIDATION_ERROR
    assert result.run.schema_valid is False


@pytest.mark.parametrize("payload", [
    {"width": -60.0, "depth": 40.0, "height": 4.0, "hole_diameter": 6.0, "centered": True},
    {"width": 60.0, "depth": 40.0, "height": 4.0, "hole_diameter": 6.0, "centered": False},
])
def test_invalid_or_non_centered_intent_is_rejected_structurally(payload):
    import json

    backend = ScriptedFakeBackend([json.dumps(payload)])
    context = default_prompt_to_spec_context(M4_001_PROMPT, "M4-001", "1.0")

    result = LLMIntentGenerator(backend).generate(context)

    assert result.status == GenerationStatus.ERROR
    assert result.intent is None
    assert result.error.code == GenerationErrorCode.SCHEMA_VALIDATION_ERROR


def test_intent_compiler_rejects_hole_larger_than_plate():
    result = DeterministicIntentToSpecCompiler().compile(
        _intent(width=10.0, depth=8.0, height=2.0, diameter=8.0),
        design_id="M4-001",
        spec_version="1.0",
    )

    assert result.status == GenerationStatus.ERROR
    assert result.design_spec is None
    assert result.error.code == GenerationErrorCode.INVALID_PARAMETER_VALUE


def test_deterministic_compiler_rejects_incompatible_candidates():
    spec = _m4_spec()
    spec = spec.model_copy(update={
        "constraints": [
            *spec.constraints,
            _constraint("C_WIDTH_CONFLICT", MeasurementType.EXTENT_X, 61.0),
        ],
    })
    context = SpecToPlanContext(design_id="M4-001", spec_version="1.0", design_spec=spec)

    result = DeterministicSpecToPlanCompiler().compile(context)

    assert result.status == GenerationStatus.ERROR
    assert result.error.code == GenerationErrorCode.INCOMPATIBLE_CONSTRAINTS


@pytest.mark.parametrize(("measurement", "target"), [
    (MeasurementType.EXTENT_Z, 0.0),
    (MeasurementType.DIAMETER, -1.0),
])
def test_deterministic_compiler_rejects_non_positive_values(measurement, target):
    spec = _m4_spec()
    constraints = [
        item for item in spec.constraints if item.measurement.type != measurement
    ]
    constraints.append(_constraint("C_INVALID_SIZE", measurement, target, feature_id="mount_hole" if measurement == MeasurementType.DIAMETER else None))
    spec = spec.model_copy(update={"constraints": constraints})
    context = SpecToPlanContext(design_id="M4-001", spec_version="1.0", design_spec=spec)

    result = DeterministicSpecToPlanCompiler().compile(context)

    assert result.status == GenerationStatus.ERROR
    assert result.error.code == GenerationErrorCode.INVALID_PARAMETER_VALUE


def test_deterministic_compiler_rejects_spec_outside_supported_subset():
    spec = _m4_spec()
    spec = spec.model_copy(update={
        "constraints": [
            *spec.constraints,
            _constraint("C_ANGLE", MeasurementType.ANGLE, 90.0),
        ],
    })
    context = SpecToPlanContext(design_id="M4-001", spec_version="1.0", design_spec=spec)

    result = DeterministicSpecToPlanCompiler().compile(context)

    assert result.status == GenerationStatus.ERROR
    assert result.error.code == GenerationErrorCode.UNSUPPORTED_SPEC_SUBSET


def test_schema_valid_but_disallowed_cad_plan_stops_before_engine():
    spec = _m4_spec()
    disallowed_plan = _m4_plan(operation_type="sphere")
    backend = ScriptedFakeBackend([disallowed_plan.model_dump_json()])
    context = SpecToPlanContext(
        design_id="M4-001",
        spec_version="1.0",
        revision_id="R01",
        design_spec=spec,
    )

    result = LLMCADPlanGenerator(backend).generate(context)

    assert result.status == GenerationStatus.ERROR
    assert result.cad_plan is None
    assert result.error.code == GenerationErrorCode.OPERATION_NOT_ALLOWED
    assert result.run.schema_valid is True
