from pathlib import Path

from cad_ai.capability_v02 import (
    CAPABILITY_PROMPT_VERSION,
    PLATE_FACT_EXTRACTION_SYSTEM_PROMPT,
    DeterministicPlateIntentGate,
    ExtractedCircularPocketFacts,
    ExtractedHoleFacts,
    ExtractedLinearHolePatternFacts,
    ExtractedPlateFacts,
    ExtractedRectangularPocketFacts,
    ExtractedSlotFacts,
    IntentUnsupportedReason,
)
from cad_ai.contracts import CADPlan
from cad_ai.planning import LLMPlannerConfig, LLMRepairPlanner, RepairPlannerResponse, ScriptedFakeBackend
from cad_ai.validation import CheckStatus, ReportStatus
from cad_ai.v02 import V02Status, run_capability_case


def _backend(facts: ExtractedPlateFacts) -> ScriptedFakeBackend:
    return ScriptedFakeBackend([facts.model_dump_json()])


def _validated(result, revision="R01"):
    assert result.status == V02Status.VALIDATED
    report = result.r01_validation_report if revision == "R01" else result.r02_validation_report
    assert report.status == ReportStatus.PASS
    assert all(check.status == CheckStatus.PASS for check in report.checks)
    assert result.final_revision_id == revision
    assert set(result.final_artifacts) == {"STEP", "STL"}
    assert all(Path(path).is_file() for path in result.final_artifacts.values())


def test_cp2_fact_contract_and_prompt_are_extraction_only():
    schema = ExtractedPlateFacts.model_json_schema()
    properties = set(schema["properties"])

    assert CAPABILITY_PROMPT_VERSION == "capability-facts-3.0"
    assert {
        "holes", "hole_groups", "rectangular_pockets", "circular_pockets", "slots",
        "linear_hole_patterns", "fillet_radius", "chamfer_distance",
    } <= properties
    assert properties.isdisjoint({"status", "reason_code", "operations", "constraints", "repair_plan"})
    assert "never calculate pattern instance coordinates" in PLATE_FACT_EXTRACTION_SYSTEM_PROMPT
    assert "Do not judge validity" in PLATE_FACT_EXTRACTION_SYSTEM_PROMPT


def test_cp2_gate_normalizes_patterns_and_rejects_invalid_geometry():
    gate = DeterministicPlateIntentGate()
    accepted = gate.evaluate(ExtractedPlateFacts(
        width=100.0,
        depth=40.0,
        height=6.0,
        linear_hole_patterns=[ExtractedLinearHolePatternFacts(
            diameter=5.0,
            count=4,
            spacing=20.0,
            axis="X",
            anchor_x=0.0,
            anchor_y=0.0,
            anchor_mode="CENTER",
        )],
    ))
    assert accepted.root.status == "INTENT"
    assert [(hole.x, hole.y) for hole in accepted.root.intent.holes] == [
        (-30.0, 0.0), (-10.0, 0.0), (10.0, 0.0), (30.0, 0.0),
    ]

    cases = [
        ExtractedPlateFacts(width=100.0, depth=60.0, height=10.0, rectangular_pockets=[
            ExtractedRectangularPocketFacts(width=20.0, depth=10.0, centered=True, cut_depth=10.0),
        ]),
        ExtractedPlateFacts(width=40.0, depth=30.0, height=8.0, circular_pockets=[
            ExtractedCircularPocketFacts(diameter=20.0, x=18.0, y=0.0, cut_depth=2.0),
        ]),
        ExtractedPlateFacts(width=80.0, depth=40.0, height=6.0, slots=[
            ExtractedSlotFacts(length=8.0, width=30.0, centered=True, angle_deg=0.0, through=True),
        ]),
        ExtractedPlateFacts(
            width=100.0, depth=40.0, height=6.0,
            holes=[ExtractedHoleFacts(diameter=5.0, x=-30.0, y=0.0)],
            linear_hole_patterns=[ExtractedLinearHolePatternFacts(
                diameter=5.0, count=2, spacing=20.0, axis="X",
                anchor_x=-30.0, anchor_y=0.0, anchor_mode="START",
            )],
        ),
    ]
    for facts in cases:
        rejected = gate.evaluate(facts)
        assert rejected.root.status == "UNSUPPORTED"
        assert rejected.root.reason_code == IntentUnsupportedReason.UNSUPPORTED_GEOMETRY


def test_cp2_01_regression_explicit_holes(tmp_path):
    facts = ExtractedPlateFacts(
        width=100.0,
        depth=60.0,
        height=4.0,
        holes=[
            ExtractedHoleFacts(diameter=5.0, x=-40.0, y=-20.0),
            ExtractedHoleFacts(diameter=5.0, x=40.0, y=-20.0),
            ExtractedHoleFacts(diameter=5.0, x=-40.0, y=20.0),
            ExtractedHoleFacts(diameter=5.0, x=40.0, y=20.0),
        ],
    )
    result = run_capability_case("CP2-01", _backend(facts), tmp_path)
    _validated(result)
    assert [op.operation_type for op in result.r01_plan.operations] == ["box", "hole", "hole", "hole", "hole"]


def test_cp2_02_rectangular_pocket(tmp_path):
    facts = ExtractedPlateFacts(
        width=100.0,
        depth=60.0,
        height=10.0,
        rectangular_pockets=[ExtractedRectangularPocketFacts(
            width=40.0, depth=20.0, centered=True, cut_depth=4.0,
        )],
    )
    result = run_capability_case("CP2-02", _backend(facts), tmp_path)
    _validated(result)
    assert result.r01_plan.operations[1].operation_type == "rect_pocket"
    actual = {check.constraint_id: check.actual for check in result.r01_validation_report.checks}
    assert actual["C_RP01_WIDTH"] == 40.0
    assert actual["C_RP01_DEPTH"] == 20.0
    assert actual["C_RP01_CUT_DEPTH"] == 4.0
    assert actual["C_RP01_THROUGH"] is False
    assert result.feature_ids_r01 == ["main_body", "rect_pocket_rp01", "rect_pocket_rp01_center"]


def test_cp2_03_rectangular_through_cutout(tmp_path):
    facts = ExtractedPlateFacts(
        width=80.0,
        depth=50.0,
        height=6.0,
        rectangular_pockets=[ExtractedRectangularPocketFacts(
            width=30.0, depth=15.0, centered=True, through=True,
        )],
    )
    result = run_capability_case("CP2-03", _backend(facts), tmp_path)
    _validated(result)
    operation = result.r01_plan.operations[1]
    assert operation.params["through"] is True
    assert operation.params["cut_depth"] is None
    assert "C_RP01_CUT_DEPTH" not in {item.constraint_id for item in result.spec_generation.design_spec.constraints}


def test_cp2_04_circular_pocket(tmp_path):
    facts = ExtractedPlateFacts(
        width=60.0,
        depth=60.0,
        height=12.0,
        circular_pockets=[ExtractedCircularPocketFacts(
            diameter=30.0, centered=True, cut_depth=5.0,
        )],
    )
    result = run_capability_case("CP2-04", _backend(facts), tmp_path)
    _validated(result)
    actual = {check.constraint_id: check.actual for check in result.r01_validation_report.checks}
    assert actual["C_CP01_DIAMETER"] == 30.0
    assert actual["C_CP01_CUT_DEPTH"] == 5.0
    assert "circle_pocket_cp01" in result.feature_ids_r01


def test_cp2_05_horizontal_through_slot(tmp_path):
    facts = ExtractedPlateFacts(
        width=80.0,
        depth=40.0,
        height=6.0,
        slots=[ExtractedSlotFacts(
            length=30.0, width=8.0, centered=True, angle_deg=0.0, through=True,
        )],
    )
    result = run_capability_case("CP2-05", _backend(facts), tmp_path)
    _validated(result)
    actual = {check.constraint_id: check.actual for check in result.r01_validation_report.checks}
    assert actual["C_S01_LENGTH"] == 30.0
    assert actual["C_S01_WIDTH"] == 8.0
    assert actual["C_S01_ANGLE"] == 0.0
    assert actual["C_S01_THROUGH"] is True


def test_cp2_06_linear_pattern_expands_to_explicit_holes(tmp_path):
    facts = ExtractedPlateFacts(
        width=100.0,
        depth=40.0,
        height=6.0,
        linear_hole_patterns=[ExtractedLinearHolePatternFacts(
            diameter=5.0, count=4, spacing=20.0, axis="X",
            anchor_x=0.0, anchor_y=0.0, anchor_mode="CENTER",
        )],
    )
    result = run_capability_case("CP2-06", _backend(facts), tmp_path)
    _validated(result)
    holes = [op for op in result.r01_plan.operations if op.operation_type == "hole"]
    assert [op.params["x"] for op in holes] == [-30.0, -10.0, 10.0, 30.0]
    assert [op.outputs[0].feature_id for op in holes] == ["hole_h01", "hole_h02", "hole_h03", "hole_h04"]


def test_cp2_07_combines_holes_pocket_and_outer_fillet(tmp_path):
    facts = ExtractedPlateFacts(
        width=120.0,
        depth=70.0,
        height=8.0,
        holes=[
            ExtractedHoleFacts(diameter=5.0, x=-40.0, y=20.0),
            ExtractedHoleFacts(diameter=5.0, x=40.0, y=20.0),
        ],
        rectangular_pockets=[ExtractedRectangularPocketFacts(
            width=36.0, depth=18.0, centered=True, cut_depth=3.0,
        )],
        fillet_radius=3.0,
    )
    result = run_capability_case("CP2-07", _backend(facts), tmp_path)
    _validated(result)
    assert [op.operation_type for op in result.r01_plan.operations] == [
        "box", "hole", "hole", "rect_pocket", "fillet",
    ]


def test_cp2_08_localized_pocket_depth_repair(tmp_path):
    facts = ExtractedPlateFacts(
        width=100.0,
        depth=60.0,
        height=10.0,
        rectangular_pockets=[ExtractedRectangularPocketFacts(
            width=40.0, depth=20.0, centered=True, cut_depth=4.0,
        )],
    )

    def fault(plan: CADPlan) -> CADPlan:
        data = plan.model_dump(mode="python")
        data["operations"][1]["params"]["cut_depth"] = 3.2
        return CADPlan.model_validate(data)

    response = RepairPlannerResponse.model_validate({
        "status": "PLAN",
        "repair_plan": {
            "design_id": "V02-CP2-08",
            "spec_version": "2.0",
            "source_revision_id": "R01",
            "target_revision_id": "R02",
            "actions": [{
                "action_id": "RA001",
                "type": "SET_PARAMETER",
                "target": {"operation_id": "OP02", "parameter": "cut_depth"},
                "old_value": 3.2,
                "new_value": 4.0,
                "constraint_ids": ["C_RP01_CUT_DEPTH"],
                "related_feature_ids": ["rect_pocket_rp01"],
                "reason": "Restore the required pocket depth.",
            }],
        },
    })
    planner = LLMRepairPlanner(
        ScriptedFakeBackend([response.model_dump_json()]),
        LLMPlannerConfig(max_attempts=1, temperature=0.0, max_tokens=1024, seed=12345),
    )
    result = run_capability_case(
        "CP2-08", _backend(facts), tmp_path,
        repair_planner=planner, test_plan_transform=fault,
    )

    _validated(result, "R02")
    failures = [
        check.constraint_id for check in result.r01_validation_report.checks
        if check.blocking and check.status == CheckStatus.FAIL
    ]
    assert failures == ["C_RP01_CUT_DEPTH"]
    assert result.changed_semantic_paths == ["OP02.params.cut_depth"]
    assert result.change_locality is True
    assert result.constraint_preservation is True
    assert result.feature_identity_preserved is True
    assert result.r01_plan.operations[1].params["cut_depth"] == 3.2
    assert result.r02_plan.operations[1].params["cut_depth"] == 4.0
    assert result.r01_plan is not result.r02_plan


def test_cp2_09_unsupported_geometry_stops_before_cad(tmp_path):
    facts = ExtractedPlateFacts(
        width=60.0, depth=40.0, height=8.0,
        unsupported_features=["thread"],
    )
    result = run_capability_case("CP2-09", _backend(facts), tmp_path)

    assert result.status == V02Status.UNSUPPORTED
    assert result.reason_code == IntentUnsupportedReason.UNSUPPORTED_GEOMETRY.value
    assert result.r01_plan is None
    assert result.r01_cad_status is None
