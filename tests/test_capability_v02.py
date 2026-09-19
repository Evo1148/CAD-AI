from pathlib import Path

from cad_ai.capability_v02 import (
    CAPABILITY_INTENT_SYSTEM_PROMPT,
    CAPABILITY_PROMPT_VERSION,
    CapabilityIntentGenerator,
    CapabilityIntentToSpecCompiler,
    CapabilitySpecToPlanCompiler,
    DeterministicPlateIntentGate,
    ExtractedHoleFacts,
    ExtractedPlateFacts,
    HoleIntent,
    IntentUnsupportedReason,
    LLMPlateFactExtractor,
    PLATE_FACT_EXTRACTION_SYSTEM_PROMPT,
    PlateIntent,
    PlateIntentResponse,
    PlatePromptContext,
    capability_intent_generation_schema,
    capability_repair_rules,
)
from cad_ai.contracts import CADPlan, MeasurementType
from cad_ai.planning import (
    HTTPInferenceBackend,
    HTTPInferenceConfig,
    LLMPlannerConfig,
    LLMRepairPlanner,
    RepairPlannerResponse,
    ScriptedFakeBackend,
)
from cad_ai.validation import CheckStatus, ReportStatus
from cad_ai.v02 import CASE_PROMPTS, V02Status, run_capability_case
from cad_ai.prompt_grounding import ExtractionMode


def _intent(
    width: float,
    depth: float,
    height: float,
    holes: list[tuple[float, float, float]],
    *,
    fillet: float | None = None,
    chamfer: float | None = None,
) -> PlateIntent:
    return PlateIntent(
        contract_version="0.2",
        width=width,
        depth=depth,
        height=height,
        holes=[HoleIntent(diameter=diameter, x=x, y=y) for diameter, x, y in holes],
        fillet_radius=fillet,
        chamfer_distance=chamfer,
    )


def _backend(intent: PlateIntent) -> ScriptedFakeBackend:
    holes = [
        ExtractedHoleFacts(
            diameter=hole.diameter,
            x=None if len(intent.holes) == 1 and hole.x == 0 and hole.y == 0 else hole.x,
            y=None if len(intent.holes) == 1 and hole.x == 0 and hole.y == 0 else hole.y,
            centered=len(intent.holes) == 1 and hole.x == 0 and hole.y == 0,
        )
        for hole in intent.holes
    ]
    facts = ExtractedPlateFacts(
        width=intent.width,
        depth=intent.depth,
        height=intent.height,
        holes=holes,
        fillet_radius=intent.fillet_radius,
        chamfer_distance=intent.chamfer_distance,
    )
    return ScriptedFakeBackend([facts.model_dump_json()])


def _facts(**updates) -> ExtractedPlateFacts:
    values = {
        "width": 90.0,
        "depth": 50.0,
        "height": 3.0,
        "holes": [ExtractedHoleFacts(diameter=4.0, x=0.0, y=0.0)],
    }
    values.update(updates)
    return ExtractedPlateFacts(**values)


class _RecordingTransport:
    def __init__(self, raw_output: str):
        self.raw_output = raw_output
        self.calls = []

    def post_json(self, url, payload, timeout_seconds):
        self.calls.append((url, payload, timeout_seconds))
        return {"choices": [{"message": {"content": self.raw_output}}]}


def _assert_validated(result, expected_revision: str = "R01"):
    assert result.status == V02Status.VALIDATED
    assert result.stage == "COMPLETE"
    report = result.r02_validation_report if expected_revision == "R02" else result.r01_validation_report
    assert report.status == ReportStatus.PASS
    assert all(check.status == CheckStatus.PASS for check in report.checks)
    assert result.final_revision_id == expected_revision
    assert set(result.final_artifacts) == {"STEP", "STL"}
    assert all(Path(path).is_file() for path in result.final_artifacts.values())
    assert all(expected_revision in Path(path).stem for path in result.final_artifacts.values())


def test_case01_v01_regression_passes_without_repair(tmp_path):
    intent = _intent(60.0, 40.0, 4.0, [(6.0, 0.0, 0.0)])
    backend = _backend(intent)

    result = run_capability_case("CASE-01", backend, tmp_path)

    _assert_validated(result)
    assert result.repair_attempted is False
    assert result.canonical_facts.hole_groups[0].positions[0].model_dump() == {"x": 0.0, "y": 0.0}
    assert result.extraction_mode == ExtractionMode.DETERMINISTIC
    assert result.fact_source == "deterministic_grounding"
    assert result.llm_invoked is False
    assert backend.requests == []
    assert result.intent_response.root.status == "INTENT"
    assert (result.intent_response.root.intent.holes[0].x, result.intent_response.root.intent.holes[0].y) == (0.0, 0.0)
    assert [operation.operation_type for operation in result.r01_plan.operations] == ["box", "hole"]
    assert result.feature_ids_r01 == ["hole_h01", "hole_h01_axis", "main_body"]
    generator_backend = _backend(intent)
    parsed = CapabilityIntentGenerator(generator_backend).generate(PlatePromptContext(
        user_prompt=CASE_PROMPTS["CASE-01"],
    ))
    assert parsed.intent == intent
    assert generator_backend.requests == []


def test_case02_multiple_holes_have_deterministic_ids_positions_and_geometry(tmp_path):
    holes = [
        (5.0, -40.0, -20.0),
        (5.0, 40.0, -20.0),
        (5.0, -40.0, 20.0),
        (5.0, 40.0, 20.0),
    ]
    intent = _intent(100.0, 60.0, 4.0, holes)

    backend = _backend(intent)
    result = run_capability_case("CASE-02", backend, tmp_path)

    _assert_validated(result)
    assert result.intent_response.root.status == "INTENT"
    assert len(result.canonical_facts.hole_groups) == 1
    assert len(result.canonical_facts.hole_groups[0].positions) == 4
    assert result.canonical_facts.fillet_radius is None
    assert result.canonical_facts.chamfer_distance is None
    assert result.llm_invoked is False
    assert backend.requests == []
    hole_operations = [operation for operation in result.r01_plan.operations if operation.operation_type == "hole"]
    assert len(hole_operations) == 4
    assert [operation.operation_id for operation in hole_operations] == ["OP02", "OP03", "OP04", "OP05"]
    assert [operation.outputs[0].feature_id for operation in hole_operations] == [
        "hole_h01", "hole_h02", "hole_h03", "hole_h04",
    ]
    actual = [(op.params["diameter"], op.params["x"], op.params["y"]) for op in hole_operations]
    assert actual == [
        (5.0, -40.0, -20.0),
        (5.0, -40.0, 20.0),
        (5.0, 40.0, -20.0),
        (5.0, 40.0, 20.0),
    ]
    checks = {check.constraint_id: check.actual for check in result.r01_validation_report.checks}
    assert checks["C_HOLE_COUNT"] == 4
    assert checks["C_H03_X"] == 40.0
    assert checks["C_H03_Y"] == -20.0

    reversed_spec = CapabilityIntentToSpecCompiler().compile(
        intent.model_copy(update={"holes": list(reversed(intent.holes))}),
        design_id="V02-CASE-02",
    ).design_spec
    reversed_plan = CapabilitySpecToPlanCompiler().compile(reversed_spec).cad_plan
    assert reversed_plan == result.plan_generation.cad_plan


def test_case03_independent_diameters_are_feature_scoped(tmp_path):
    intent = _intent(120.0, 70.0, 5.0, [(8.0, 35.0, 0.0), (5.0, -35.0, 0.0)])

    result = run_capability_case("CASE-03", _backend(intent), tmp_path)

    _assert_validated(result)
    constraints = {item.constraint_id: item for item in result.spec_generation.design_spec.constraints}
    assert constraints["C_H01_DIAMETER"].target == 5.0
    assert constraints["C_H01_DIAMETER"].measurement.feature_id == "hole_h01"
    assert constraints["C_H02_DIAMETER"].target == 8.0
    assert constraints["C_H02_DIAMETER"].measurement.feature_id == "hole_h02"
    actual = {check.constraint_id: check.actual for check in result.r01_validation_report.checks}
    assert actual["C_H01_DIAMETER"] == 5.0
    assert actual["C_H02_DIAMETER"] == 8.0


def test_case04_multiple_holes_and_outer_vertical_fillet_pass(tmp_path):
    intent = _intent(100.0, 60.0, 4.0, [(5.0, -30.0, 0.0), (5.0, 30.0, 0.0)], fillet=3.0)

    backend = _backend(intent)
    result = run_capability_case("CASE-04", backend, tmp_path)

    _assert_validated(result)
    assert result.intent_response.root.status == "INTENT"
    assert result.canonical_facts.fillet_radius == 3.0
    assert result.canonical_facts.chamfer_distance is None
    assert result.llm_invoked is False
    assert backend.requests == []
    assert [operation.operation_type for operation in result.r01_plan.operations] == [
        "box", "hole", "hole", "fillet",
    ]
    assert result.r01_plan.operations[-1].params == {"radius": 3.0}
    checks = {check.constraint_id: check.actual for check in result.r01_validation_report.checks}
    assert checks["C_FILLET_EXISTS"] is True
    assert checks["C_FILLET_RADIUS"] == 3.0
    assert "plate_fillet" in result.feature_ids_r01


def test_case05_multiple_holes_and_outer_vertical_chamfer_pass(tmp_path):
    intent = _intent(100.0, 60.0, 4.0, [(5.0, -30.0, 0.0), (5.0, 30.0, 0.0)], chamfer=3.0)

    backend = _backend(intent)
    result = run_capability_case("CASE-05", backend, tmp_path)

    _assert_validated(result)
    assert result.intent_response.root.status == "INTENT"
    assert result.canonical_facts.fillet_radius is None
    assert result.canonical_facts.chamfer_distance == 3.0
    assert result.llm_invoked is False
    assert backend.requests == []
    assert [operation.operation_type for operation in result.r01_plan.operations] == [
        "box", "hole", "hole", "chamfer",
    ]
    assert result.r01_plan.operations[-1].params == {"distance": 3.0}
    checks = {check.constraint_id: check.actual for check in result.r01_validation_report.checks}
    assert checks["C_CHAMFER_EXISTS"] is True
    assert checks["C_CHAMFER_DISTANCE"] == 3.0
    assert "plate_chamfer" in result.feature_ids_r01


def test_case06_localized_h03_repair_preserves_every_other_hole(tmp_path):
    intent = _intent(100.0, 60.0, 4.0, [
        (5.0, -30.0, -15.0),
        (5.0, -30.0, 15.0),
        (5.0, 30.0, -15.0),
        (5.0, 30.0, 15.0),
    ])

    def fault(plan: CADPlan) -> CADPlan:
        data = plan.model_dump(mode="python")
        operations = {item["operation_id"]: item for item in data["operations"]}
        operations["OP04"]["params"]["diameter"] = 4.4
        return CADPlan.model_validate(data)

    response = RepairPlannerResponse.model_validate({
        "status": "PLAN",
        "repair_plan": {
            "design_id": "V02-CASE-06",
            "spec_version": "2.0",
            "source_revision_id": "R01",
            "target_revision_id": "R02",
            "actions": [{
                "action_id": "RA001",
                "type": "SET_PARAMETER",
                "target": {"operation_id": "OP04", "parameter": "diameter"},
                "old_value": 4.4,
                "new_value": 5.0,
                "constraint_ids": ["C_H03_DIAMETER"],
                "related_feature_ids": ["hole_h03"],
                "reason": "C_H03_DIAMETER failed dimensional validation",
            }],
        },
    })
    planner = LLMRepairPlanner(
        ScriptedFakeBackend([response.model_dump_json()]),
        LLMPlannerConfig(max_attempts=1, temperature=0.0, max_tokens=1024, seed=12345),
    )

    result = run_capability_case(
        "CASE-06",
        _backend(intent),
        tmp_path,
        repair_planner=planner,
        test_plan_transform=fault,
    )

    _assert_validated(result, "R02")
    assert result.plan_generation.cad_plan.operations[3].params["diameter"] == 5.0
    assert result.r01_plan.operations[3].params["diameter"] == 4.4
    failures = [
        check for check in result.r01_validation_report.checks
        if check.blocking and check.status == CheckStatus.FAIL
    ]
    assert [check.constraint_id for check in failures] == ["C_H03_DIAMETER"]
    assert [item.constraint_id for item in result.planning_context.failed_constraints] == ["C_H03_DIAMETER"]
    h03_operation = next(
        item for item in result.planning_context.repairable_operations
        if item.operation_id == "OP04"
    )
    assert {item.name for item in h03_operation.repairable_parameters} == {"diameter", "x", "y"}
    diameter = next(item for item in h03_operation.repairable_parameters if item.name == "diameter")
    assert diameter.current_value == 4.4
    assert diameter.constraint_ids == ["C_H03_DIAMETER"]
    assert result.repair_plan_validation.status.value == "VALID"
    assert result.repair_executor_status.value == "REPAIRED"
    assert result.r02_plan.revision_id == "R02"
    assert result.r02_plan.parent_revision_id == "R01"
    assert result.changed_semantic_paths == ["OP04.params.diameter"]
    assert result.change_locality is True
    assert result.constraint_preservation is True
    assert result.feature_identity_preserved is True
    for operation_id in ("OP02", "OP03", "OP05"):
        r01 = next(item for item in result.r01_plan.operations if item.operation_id == operation_id)
        r02 = next(item for item in result.r02_plan.operations if item.operation_id == operation_id)
        assert r01 == r02


def test_v02_intent_rejects_conflicting_edge_treatments():
    try:
        _intent(100.0, 60.0, 4.0, [(5.0, 0.0, 0.0)], fillet=2.0, chamfer=2.0)
    except ValueError as exc:
        assert "does not combine fillet and chamfer" in str(exc)
    else:
        raise AssertionError("fillet plus chamfer must be rejected")


def test_fact_extractor_prompt_is_extraction_only():
    prompt = " ".join(PLATE_FACT_EXTRACTION_SYSTEM_PROMPT.split())

    assert CAPABILITY_PROMPT_VERSION == "capability-facts-3.0"
    assert CAPABILITY_INTENT_SYSTEM_PROMPT == PLATE_FACT_EXTRACTION_SYSTEM_PROMPT
    assert '"A x B x C mm" means width=A, depth=B, height=C' in prompt
    assert "Do not judge validity, ambiguity, collisions, conflicts, bounds, or support" in prompt
    assert "Do not return status, reason_code" in prompt


def test_extracted_facts_contract_contains_no_authorization_or_cad_fields():
    properties = set(ExtractedPlateFacts.model_json_schema()["properties"])

    assert properties == {
        "contract_version",
        "width",
        "depth",
        "height",
        "holes",
        "hole_groups",
        "rectangular_pockets",
        "circular_pockets",
        "slots",
        "linear_hole_patterns",
        "rectangular_bosses",
        "cylindrical_bosses",
        "standoffs",
        "standoff_groups",
        "additive_linear_patterns",
        "fillet_radius",
        "chamfer_distance",
        "unsupported_features",
    }
    assert properties.isdisjoint({
        "status", "reason_code", "operations", "constraints", "repair_plan",
    })


def test_extractor_returns_standard_dimensions_multiple_holes_and_no_invented_features():
    facts = ExtractedPlateFacts(
        width=83.0,
        depth=47.0,
        height=3.5,
        holes=[
            ExtractedHoleFacts(diameter=4.0, x=-20.0, y=0.0),
            ExtractedHoleFacts(diameter=6.0, x=20.0, y=0.0),
        ],
    )
    backend = ScriptedFakeBackend([facts.model_dump_json()])

    result = LLMPlateFactExtractor(backend).extract(PlatePromptContext(
        user_prompt="Create an 83 x 47 x 3.5 mm plate with holes at (-20,0) and (20,0).",
    ))

    assert result.status.value == "SUCCESS"
    assert result.facts == facts
    assert result.facts.unsupported_features == []
    assert backend.requests[0].response_schema == ExtractedPlateFacts.model_json_schema()


def test_extractor_returns_centered_hole_fillet_and_chamfer_as_literal_facts():
    centered = _facts(holes=[ExtractedHoleFacts(diameter=6.0, centered=True)])
    fillet = _facts(fillet_radius=2.0)
    chamfer = _facts(chamfer_distance=2.5)

    centered_result = LLMPlateFactExtractor(ScriptedFakeBackend([centered.model_dump_json()])).extract(
        PlatePromptContext(user_prompt="Create a plate with one centered hole.")
    )
    fillet_result = LLMPlateFactExtractor(ScriptedFakeBackend([fillet.model_dump_json()])).extract(
        PlatePromptContext(user_prompt="Create a plate with a 2 mm fillet.")
    )
    chamfer_result = LLMPlateFactExtractor(ScriptedFakeBackend([chamfer.model_dump_json()])).extract(
        PlatePromptContext(user_prompt="Create a plate with a 2.5 mm chamfer.")
    )

    assert centered_result.facts.holes[0].centered is True
    assert centered_result.facts.holes[0].x is None and centered_result.facts.holes[0].y is None
    assert fillet_result.facts.fillet_radius == 2.0 and fillet_result.facts.chamfer_distance is None
    assert chamfer_result.facts.chamfer_distance == 2.5 and chamfer_result.facts.fillet_radius is None


def test_llm_user_message_and_http_payload_contain_only_authoritative_request():
    facts = _facts(fillet_radius=2.0)
    transport = _RecordingTransport(facts.model_dump_json())
    backend = HTTPInferenceBackend(HTTPInferenceConfig(
        base_url="http://127.0.0.1:8080",
        model_id="test-model",
        response_format_dialect="llama.cpp",
        chat_template_kwargs={"enable_thinking": False},
    ), transport)
    prompt = "Create a plate with a fillet on the outer vertical plate edges."

    result = LLMPlateFactExtractor(backend).extract(PlatePromptContext(user_prompt=prompt))

    payload = transport.calls[0][1]
    assert result.status.value == "SUCCESS"
    assert payload["messages"][1] == {
        "role": "user",
        "content": f'{{"user_request":"{prompt}"}}',
    }
    assert "supported_edge_treatments" not in payload["messages"][1]["content"]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["temperature"] == 0.0
    assert payload["seed"] == 12345
    assert payload["json_schema"] == ExtractedPlateFacts.model_json_schema()


def _assert_gate_unsupported(facts: ExtractedPlateFacts, reason: IntentUnsupportedReason):
    response = DeterministicPlateIntentGate().evaluate(facts)
    assert response.root.status == "UNSUPPORTED"
    assert response.root.reason_code == reason


def test_gate_complete_dimensions_produce_intent():
    response = DeterministicPlateIntentGate().evaluate(_facts())
    assert response.root.status == "INTENT"
    assert (response.root.intent.width, response.root.intent.depth, response.root.intent.height) == (
        90.0, 50.0, 3.0,
    )


def test_gate_missing_width_is_insufficient_dimensions():
    _assert_gate_unsupported(
        _facts(width=None), IntentUnsupportedReason.INSUFFICIENT_DIMENSIONS,
    )


def test_gate_missing_depth_is_insufficient_dimensions():
    _assert_gate_unsupported(
        _facts(depth=None), IntentUnsupportedReason.INSUFFICIENT_DIMENSIONS,
    )


def test_gate_missing_height_is_insufficient_dimensions():
    _assert_gate_unsupported(
        _facts(height=None), IntentUnsupportedReason.INSUFFICIENT_DIMENSIONS,
    )


def test_gate_multiple_explicit_holes_produce_intent_without_centered():
    facts = _facts(holes=[
        ExtractedHoleFacts(diameter=4.0, x=-20.0, y=-10.0),
        ExtractedHoleFacts(diameter=6.0, x=20.0, y=10.0),
    ])
    response = DeterministicPlateIntentGate().evaluate(facts)
    assert response.root.status == "INTENT"
    assert [(hole.x, hole.y) for hole in response.root.intent.holes] == [(-20.0, -10.0), (20.0, 10.0)]


def test_gate_missing_hole_coordinates_are_ambiguous():
    _assert_gate_unsupported(
        _facts(holes=[ExtractedHoleFacts(diameter=4.0)]),
        IntentUnsupportedReason.AMBIGUOUS_POSITION,
    )


def test_gate_single_centered_hole_normalizes_to_origin():
    response = DeterministicPlateIntentGate().evaluate(
        _facts(holes=[ExtractedHoleFacts(diameter=4.0, centered=True)])
    )
    assert response.root.status == "INTENT"
    assert (response.root.intent.holes[0].x, response.root.intent.holes[0].y) == (0.0, 0.0)


def test_gate_only_fillet_produces_intent():
    response = DeterministicPlateIntentGate().evaluate(_facts(fillet_radius=2.0))
    assert response.root.status == "INTENT"
    assert response.root.intent.fillet_radius == 2.0
    assert response.root.intent.chamfer_distance is None


def test_gate_only_chamfer_produces_intent():
    response = DeterministicPlateIntentGate().evaluate(_facts(chamfer_distance=2.0))
    assert response.root.status == "INTENT"
    assert response.root.intent.chamfer_distance == 2.0
    assert response.root.intent.fillet_radius is None


def test_gate_explicit_fillet_and_chamfer_are_conflicting():
    _assert_gate_unsupported(
        _facts(fillet_radius=2.0, chamfer_distance=2.0),
        IntentUnsupportedReason.CONFLICTING_EDGE_TREATMENTS,
    )


def test_gate_without_edge_treatment_produces_intent():
    response = DeterministicPlateIntentGate().evaluate(_facts())
    assert response.root.status == "INTENT"
    assert response.root.intent.fillet_radius is None
    assert response.root.intent.chamfer_distance is None


def test_gate_explicit_unsupported_feature_is_unsupported_geometry():
    _assert_gate_unsupported(
        _facts(unsupported_features=["slot"]),
        IntentUnsupportedReason.UNSUPPORTED_GEOMETRY,
    )


def test_v02_compiler_emits_all_validator_measurements():
    intent = _intent(100.0, 60.0, 4.0, [(5.0, -20.0, 0.0), (8.0, 20.0, 0.0)], fillet=2.0)
    spec = CapabilityIntentToSpecCompiler().compile(intent, design_id="D").design_spec

    measurements = {constraint.measurement.type for constraint in spec.constraints}

    assert {
        MeasurementType.MODEL_VALID,
        MeasurementType.SOLID_COUNT,
        MeasurementType.EXTENT_X,
        MeasurementType.EXTENT_Y,
        MeasurementType.EXTENT_Z,
        MeasurementType.FEATURE_EXISTS,
        MeasurementType.HOLE_COUNT,
        MeasurementType.DIAMETER,
        MeasurementType.POSITION_X,
        MeasurementType.POSITION_Y,
        MeasurementType.FILLET_RADIUS,
    } <= measurements


def test_v02_repair_rules_expose_local_hole_positions_and_edge_treatment():
    intent = _intent(100.0, 60.0, 4.0, [(5.0, -20.0, 0.0), (8.0, 20.0, 0.0)], fillet=2.0)
    spec = CapabilityIntentToSpecCompiler().compile(intent, design_id="D").design_spec
    plan = CapabilitySpecToPlanCompiler().compile(spec).cad_plan

    rules = {rule.constraint_id: rule.target_path for rule in capability_repair_rules(spec, plan)}

    assert rules["C_H02_DIAMETER"] == "OP03.params.diameter"
    assert rules["C_H02_X"] == "OP03.params.x"
    assert rules["C_H02_Y"] == "OP03.params.y"
    assert rules["C_FILLET_RADIUS"] == "OP04.params.radius"
