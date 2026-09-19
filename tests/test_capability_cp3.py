from __future__ import annotations

from pathlib import Path

import pytest

from cad_ai.capability_v02 import (
    CapabilityIntentToSpecCompiler,
    CapabilitySpecToPlanCompiler,
    DeterministicPlateIntentGate,
    ExtractedPlateFacts,
    PlateIntentPayload,
    capability_repair_rules,
)
from cad_ai.contracts import CADPlan
from cad_ai.planning import LLMPlannerConfig, LLMRepairPlanner, RepairPlannerResponse, ScriptedFakeBackend
from cad_ai.prompt_grounding import (
    DeterministicExtractionCoverage,
    DeterministicFactAssembler,
    DeterministicPromptGrounder,
    ExtractionCoverageStatus,
)
from cad_ai.validation import CheckStatus, ReportStatus
from cad_ai.v02 import CASE_PROMPTS, V02Status, run_capability_case


class _FailIfCalledBackend:
    backend_type = "must-not-be-called"
    model_id = None

    def generate(self, request):  # pragma: no cover - failure is the assertion
        raise AssertionError("COMPLETE CP3 prompts must bypass the LLM")


def _canonical(prompt: str):
    evidence = DeterministicPromptGrounder().ground(prompt)
    coverage = DeterministicExtractionCoverage().evaluate(evidence)
    facts = DeterministicFactAssembler().assemble(evidence)
    return evidence, coverage, facts


@pytest.mark.parametrize("case_id", [f"CP3-{index:02d}" for index in range(1, 9)])
def test_cp3_complete_goldens_rebuild_real_cad(case_id: str, tmp_path: Path):
    result = run_capability_case(case_id, _FailIfCalledBackend(), tmp_path / case_id)

    assert result.status == V02Status.VALIDATED
    assert result.grounding_coverage.status == ExtractionCoverageStatus.COMPLETE
    assert result.llm_invoked is False
    assert result.r01_validation_report.status == ReportStatus.PASS
    assert result.r01_cad_status.value == "SUCCESS"
    assert result.final_revision_id == "R01"
    assert set(result.final_artifacts) == {"STEP", "STL"}
    assert all(Path(path).is_file() for path in result.final_artifacts.values())


def test_cp3_grounding_and_gate_preserve_additive_semantics_and_pattern_math():
    _, coverage, group = _canonical(CASE_PROMPTS["CP3-06"])
    assert coverage.status == ExtractionCoverageStatus.COMPLETE
    assert len(group.standoff_groups) == 1
    assert group.standoff_groups[0].model_dump() == {
        "outer_diameter": 8.0,
        "height": 12.0,
        "hole_diameter": 3.0,
        "positions": [
            {"x": -35.0, "y": -20.0}, {"x": 35.0, "y": -20.0},
            {"x": -35.0, "y": 20.0}, {"x": 35.0, "y": 20.0},
        ],
    }
    response = DeterministicPlateIntentGate().evaluate(group)
    assert isinstance(response.root, PlateIntentPayload)
    assert len(response.root.intent.standoffs) == 4
    assert all(item.hole_diameter == 3.0 for item in response.root.intent.standoffs)

    _, coverage, pattern = _canonical(CASE_PROMPTS["CP3-07"])
    assert coverage.status == ExtractionCoverageStatus.COMPLETE
    assert pattern.standoffs == [] and pattern.standoff_groups == []
    assert pattern.additive_linear_patterns[0].model_dump() == {
        "feature_type": "STANDOFF", "diameter": 8.0, "height": 10.0,
        "hole_diameter": None, "count": 4, "spacing": 20.0, "axis": "X",
        "anchor_x": 0.0, "anchor_y": 0.0, "anchor_mode": "CENTER",
    }
    response = DeterministicPlateIntentGate().evaluate(pattern)
    assert isinstance(response.root, PlateIntentPayload)
    assert [(item.x, item.y) for item in response.root.intent.standoffs] == [
        (-30.0, 0.0), (-10.0, 0.0), (10.0, 0.0), (30.0, 0.0),
    ]

    _, coverage, cyl_pattern = _canonical(
        "Create a 100 x 50 x 4 mm plate with three 8 mm diameter cylindrical bosses "
        "10 mm high in a vertical linear pattern centered at (0,0), spaced 15 mm apart."
    )
    assert coverage.status == ExtractionCoverageStatus.COMPLETE
    response = DeterministicPlateIntentGate().evaluate(cyl_pattern)
    assert isinstance(response.root, PlateIntentPayload)
    assert [(item.x, item.y) for item in response.root.intent.cylindrical_bosses] == [
        (0.0, -15.0), (0.0, 0.0), (0.0, 15.0),
    ]


@pytest.mark.parametrize(
    "facts_update",
    [
        {"standoffs": [{"outer_diameter": 8.0, "height": 10.0, "hole_diameter": 8.0, "x": 0.0, "y": 0.0}]},
        {"standoffs": [{"outer_diameter": 12.0, "height": 10.0, "x": 47.0, "y": 0.0}]},
        {"standoffs": [
            {"outer_diameter": 12.0, "height": 10.0, "x": 0.0, "y": 0.0},
            {"outer_diameter": 12.0, "height": 10.0, "x": 5.0, "y": 0.0},
        ]},
    ],
)
def test_cp3_gate_rejects_degenerate_out_of_bounds_or_overlapping_additives(facts_update):
    _, _, facts = _canonical("Create a 100 x 60 x 4 mm plate.")
    facts = ExtractedPlateFacts.model_validate({**facts.model_dump(mode="python"), **facts_update})
    response = DeterministicPlateIntentGate().evaluate(facts)
    assert response.root.status == "UNSUPPORTED"


def test_cp3_semantic_ids_counts_and_operation_order_are_deterministic(tmp_path: Path):
    result = run_capability_case("CP3-08", _FailIfCalledBackend(), tmp_path)
    assert result.status == V02Status.VALIDATED
    operations = [item.operation_type for item in result.r01_plan.operations]
    assert operations == [
        "box", "hole", "hole", "rect_pocket",
        "standoff", "standoff", "standoff", "standoff", "fillet",
    ]
    assert {f"standoff_st{index:02d}" for index in range(1, 5)} <= set(result.feature_ids_r01)
    assert {f"standoff_st{index:02d}_hole" for index in range(1, 5)} <= set(result.feature_ids_r01)
    checks = {item.constraint_id: item.actual for item in result.r01_validation_report.checks}
    assert checks["C_STANDOFF_COUNT"] == 4
    assert checks["C_HOLE_COUNT"] == 2
    assert checks["C_RECT_POCKET_COUNT"] == 1


def test_cp3_partial_residual_only_fills_position_and_cannot_overwrite_dimensions(tmp_path: Path):
    prompt = (
        "Create an 80 x 60 x 5 mm plate with a 16 mm diameter cylindrical boss "
        "10 mm high whose centre is X 20 mm and Y -10 mm."
    )
    backend = ScriptedFakeBackend([
        '{"contract_version":"0.2","values":{'
        '"cylindrical_bosses[0].x":20.0,"cylindrical_bosses[0].y":-10.0}}'
    ])
    result = run_capability_case("CP3-03", backend, tmp_path, user_prompt=prompt)

    assert result.status == V02Status.VALIDATED
    assert result.grounding_coverage.status == ExtractionCoverageStatus.PARTIAL
    assert result.grounding_coverage.unresolved_fields == [
        "cylindrical_bosses[0].x", "cylindrical_bosses[0].y",
    ]
    assert result.llm_invoked is True
    assert len(backend.requests) == 1
    assert set(backend.requests[0].response_schema["properties"]["values"]["properties"]) == set(
        result.grounding_coverage.unresolved_fields
    )
    boss = result.canonical_facts.cylindrical_bosses[0]
    assert (boss.diameter, boss.height, boss.x, boss.y) == (16.0, 10.0, 20.0, -10.0)


@pytest.mark.parametrize(
    "prompt",
    [
        "Create an 80 x 60 x 4 mm plate with a centered 18 mm boss 8 mm high.",
        "Create an 80 x 60 x 4 mm plate with a 16 mm diameter cylindrical boss 8 mm high on the left side face.",
    ],
)
def test_cp3_ambiguous_or_side_face_additive_never_reaches_cad(prompt: str, tmp_path: Path):
    result = run_capability_case("CP3-10", _FailIfCalledBackend(), tmp_path, user_prompt=prompt)
    assert result.status in {V02Status.FAIL, V02Status.UNSUPPORTED}
    assert result.r01_plan is None
    assert result.r01_cad_status is None


def test_cp3_standoff_hole_is_semantic_and_feature_aware(tmp_path: Path):
    result = run_capability_case("CP3-05", _FailIfCalledBackend(), tmp_path)
    assert result.status == V02Status.VALIDATED
    assert "standoff_st01" in result.feature_ids_r01
    assert "standoff_st01_axis" in result.feature_ids_r01
    assert "standoff_st01_hole" in result.feature_ids_r01
    assert "standoff_st01_hole_axis" in result.feature_ids_r01
    checks = {item.constraint_id: item for item in result.r01_validation_report.checks}
    assert checks["C_ST01_OUTER_DIAMETER"].actual == 10.0
    assert checks["C_ST01_HEIGHT"].actual == 12.0
    assert checks["C_ST01_HOLE_DIAMETER"].actual == 3.0


def test_cp3_09_localized_additive_repair_preserves_history_and_identity(tmp_path: Path):
    def fault(plan: CADPlan) -> CADPlan:
        data = plan.model_dump(mode="python")
        operation = next(item for item in data["operations"] if item["operation_type"] == "standoff")
        operation["params"]["height"] = 10.5
        return CADPlan.model_validate(data)

    response = RepairPlannerResponse.model_validate({
        "status": "PLAN",
        "repair_plan": {
            "design_id": "V02-CP3-09", "spec_version": "2.0",
            "source_revision_id": "R01", "target_revision_id": "R02",
            "actions": [{
                "action_id": "RA001", "type": "SET_PARAMETER",
                "target": {"operation_id": "OP02", "parameter": "height"},
                "old_value": 10.5, "new_value": 12.0,
                "constraint_ids": ["C_ST01_HEIGHT"],
                "related_feature_ids": ["standoff_st01"],
                "reason": "Restore the required standoff height.",
            }],
        },
    })
    planner = LLMRepairPlanner(
        ScriptedFakeBackend([response.model_dump_json()]),
        LLMPlannerConfig(max_attempts=1, temperature=0.0, max_tokens=1024, seed=12345),
    )
    result = run_capability_case(
        "CP3-09", _FailIfCalledBackend(), tmp_path,
        repair_planner=planner, test_plan_transform=fault,
    )

    assert result.status == V02Status.VALIDATED
    assert result.final_revision_id == "R02"
    failures = [item.constraint_id for item in result.r01_validation_report.checks if item.blocking and item.status == CheckStatus.FAIL]
    assert failures == ["C_ST01_HEIGHT"]
    assert result.r02_validation_report.status == ReportStatus.PASS
    assert result.changed_semantic_paths == ["OP02.params.height"]
    assert result.change_locality is True
    assert result.constraint_preservation is True
    assert result.feature_identity_preserved is True
    assert result.r01_plan.revision_id == "R01" and result.r01_plan.parent_revision_id is None
    assert result.r02_plan.revision_id == "R02" and result.r02_plan.parent_revision_id == "R01"
    assert all(Path(path).is_file() for path in result.final_artifacts.values())


def test_cp3_repair_rules_authorize_only_local_additive_parameters():
    _, _, facts = _canonical(CASE_PROMPTS["CP3-05"])
    response = DeterministicPlateIntentGate().evaluate(facts)
    assert isinstance(response.root, PlateIntentPayload)
    spec = CapabilityIntentToSpecCompiler().compile(response.root.intent, design_id="D-CP3").design_spec
    plan = CapabilitySpecToPlanCompiler().compile(spec).cad_plan
    rules = capability_repair_rules(spec, plan)
    standoff_rules = {rule.constraint_id: (rule.operation_id, rule.parameter_name) for rule in rules if rule.constraint_id.startswith("C_ST01_")}
    assert standoff_rules == {
        "C_ST01_OUTER_DIAMETER": ("OP02", "outer_diameter"),
        "C_ST01_HEIGHT": ("OP02", "height"),
        "C_ST01_X": ("OP02", "x"),
        "C_ST01_Y": ("OP02", "y"),
        "C_ST01_HOLE_DIAMETER": ("OP02", "hole_diameter"),
    }
