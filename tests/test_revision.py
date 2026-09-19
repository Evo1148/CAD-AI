from copy import deepcopy

import pytest

from cad_ai.cad import CADStatus
from cad_ai.cases import block_case, repair_case
from cad_ai.pipeline import run_pipeline, run_revision_cycle
from cad_ai.revision import (
    DeterministicRepairController,
    RepairRule,
    RepairStatus,
    changed_semantic_plan_paths,
    preserved_passing_constraints,
)
from cad_ai.validation import CheckStatus, ReportStatus


@pytest.fixture(scope="module")
def cycle(tmp_path_factory):
    spec, plan = repair_case()
    controller = DeterministicRepairController([
        RepairRule(constraint_id="C_WIDTH", operation_id="OP01", parameter_name="x", related_feature_ids=["main_body"])
    ])
    result = run_revision_cycle(spec, plan, controller, tmp_path_factory.mktemp("revision-cycle"))
    return spec, plan, result


def _checks(report):
    return {check.constraint_id: check for check in report.checks}


def test_rev001_r01_hard_dimensional_failure_is_detected(cycle):
    _, _, result = cycle
    check = _checks(result.r01.validation_report)["C_WIDTH"]
    assert check.status == CheckStatus.FAIL and check.blocking
    assert check.actual == 99.5 and check.expected["target"] == 100.0


def test_rev002_supported_failure_produces_deterministic_patch(cycle):
    _, _, result = cycle
    assert result.repair.status == RepairStatus.REPAIRED
    assert result.repair.changes[0].model_dump(mode="json") == {
        "change_id": "CHG-001", "reason": "HARD constraint C_WIDTH failed", "constraint_id": "C_WIDTH",
        "target": "OP01.params.x", "old_value": 99.5, "new_value": 100.0,
        "related_operation_id": "OP01", "related_feature_ids": ["main_body"],
    }


def test_rev003_r02_revision_id(cycle):
    assert cycle[2].r02_plan.revision_id == "R02"


def test_rev004_r02_parent_is_r01(cycle):
    assert cycle[2].r02_plan.parent_revision_id == "R01"


def test_rev005_design_id_is_preserved(cycle):
    spec, r01_plan, result = cycle
    assert spec.design_id == r01_plan.design_id == result.r02_plan.design_id == "D001"


def test_rev006_spec_version_is_preserved(cycle):
    spec, r01_plan, result = cycle
    assert spec.spec_version == r01_plan.spec_version == result.r02_plan.spec_version == "1.0"


def test_rev007_design_spec_is_not_modified(tmp_path):
    spec, plan = repair_case()
    before = spec.model_dump(mode="json")
    r01 = run_pipeline(spec, plan, tmp_path)
    controller = DeterministicRepairController([RepairRule(constraint_id="C_WIDTH", operation_id="OP01", parameter_name="x")])
    controller.propose(spec, plan, r01.validation_report)
    assert spec.model_dump(mode="json") == before


def test_rev008_semantic_feature_ids_survive_rebuild(cycle):
    _, _, result = cycle
    before = set(result.r01.cad_result.feature_registry.features)
    after = set(result.r02.cad_result.feature_registry.features)
    assert before == after == {"main_body", "mount_hole_left", "mount_hole_left_axis", "mount_hole_right", "mount_hole_right_axis"}


def test_rev009_change_locality_only_intended_value(cycle):
    _, plan, result = cycle
    assert changed_semantic_plan_paths(plan, result.r02_plan) == {"OP01.params.x"}
    assert result.change_locality


def test_rev010_unrelated_parameters_are_unchanged(cycle):
    _, plan, result = cycle
    before = {op.operation_id: deepcopy(op.params) for op in plan.operations}
    after = {op.operation_id: deepcopy(op.params) for op in result.r02_plan.operations}
    assert before["OP01"]["y"] == after["OP01"]["y"] == 60.0
    assert before["OP01"]["z"] == after["OP01"]["z"] == 4.0
    assert before["OP02"] == after["OP02"] and before["OP03"] == after["OP03"]


def test_rev011_r02_executes_real_cad_engine(cycle):
    result = cycle[2]
    assert result.r02.cad_result.status == CADStatus.SUCCESS
    assert result.r02.cad_result.model_handle is not result.r01.cad_result.model_handle
    assert result.r02.cad_result.model_handle.isValid()


def test_rev012_r02_has_new_cad_result(cycle):
    result = cycle[2]
    assert result.r01.cad_result.revision_id == "R01"
    assert result.r02.cad_result.revision_id == "R02"
    assert result.r01.cad_result.model_reference != result.r02.cad_result.model_reference


def test_rev013_r02_has_new_validation_report(cycle):
    result = cycle[2]
    assert result.r01.validation_report.revision_id == "R01"
    assert result.r02.validation_report.revision_id == "R02"
    assert result.r01.validation_report is not result.r02.validation_report


def test_rev014_target_constraint_transitions_fail_to_pass(cycle):
    result = cycle[2]
    assert _checks(result.r01.validation_report)["C_WIDTH"].status == CheckStatus.FAIL
    assert _checks(result.r02.validation_report)["C_WIDTH"].status == CheckStatus.PASS


def test_rev015_previously_passing_constraints_remain_pass(cycle):
    result = cycle[2]
    preserved, ids = preserved_passing_constraints(result.r01.validation_report, result.r02.validation_report)
    assert preserved and result.constraint_preservation
    assert set(ids) == {"C_VALID", "C_SOLID_COUNT", "C_DEPTH", "C_HEIGHT", "C_HOLE_DIAMETER", "C_HOLE_SPACING"}
    after = _checks(result.r02.validation_report)
    assert all(after[constraint_id].status == CheckStatus.PASS for constraint_id in ids)


def test_rev016_final_report_is_pass(cycle):
    assert cycle[2].r01.validation_report.status == ReportStatus.FAIL
    assert cycle[2].r02.validation_report.status == ReportStatus.PASS


def test_rev017_unsupported_failure_does_not_repair(tmp_path):
    spec, plan = block_case("fail")
    report = run_pipeline(spec, plan, tmp_path).validation_report
    controller = DeterministicRepairController([
        RepairRule(constraint_id="width", operation_id="op-main-body", parameter_name="x")
    ])
    result = controller.propose(spec, plan, report)
    assert result.status == RepairStatus.UNSUPPORTED
    assert result.result_code == "NO_REPAIR_AVAILABLE"
    assert result.changes == [] and result.revised_plan is None


def test_rev018_repair_result_serializes_round_trip(cycle):
    result = cycle[2].repair
    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored == result
