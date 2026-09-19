import json
from pathlib import Path

import pytest

from cad_ai.cases import lab001_case
from cad_ai.lab import LAB001_RULE, LabResult, LabStatus, run_lab001
from cad_ai.pipeline import run_pipeline
from cad_ai.planning import (
    BackendFailureCode,
    BackendResponse,
    BackendStatus,
    DeterministicPlannerAdapter,
    LLMPlannerConfig,
    LLMRepairPlanner,
    RepairPlanningContextBuilder,
    ScriptedFakeBackend,
)
from cad_ai.validation import CheckStatus, ReportStatus


def _golden_raw(tmp_path) -> str:
    spec, plan = lab001_case()
    r01 = run_pipeline(spec, plan, tmp_path / "golden-source")
    context = RepairPlanningContextBuilder().build(
        spec, plan, r01.validation_report, [LAB001_RULE]
    )
    return DeterministicPlannerAdapter().plan(context).response.model_dump_json()


@pytest.fixture()
def validated_lab(tmp_path):
    backend = ScriptedFakeBackend([_golden_raw(tmp_path)])
    planner = LLMRepairPlanner(backend, LLMPlannerConfig(max_attempts=1))
    result_path = tmp_path / "LAB-001" / "result.json"
    result = run_lab001(planner, tmp_path / "LAB-001", result_path)
    return result, result_path, backend


def test_lab001_fixture_separates_requirement_from_deliberate_r01_defect():
    spec, plan = lab001_case()
    constraints = {constraint.constraint_id: constraint for constraint in spec.constraints}
    operations = {operation.operation_id: operation for operation in plan.operations}

    assert spec.parameters == {
        "width": 60.0, "depth": 40.0, "height": 4.0, "hole_diameter": 6.0
    }
    assert set(constraints) == {
        "C_VALID", "C_SOLID_COUNT", "C_WIDTH", "C_DEPTH", "C_HEIGHT", "C_HOLE_DIAMETER"
    }
    assert constraints["C_HOLE_DIAMETER"].target == 6.0
    assert constraints["C_HOLE_DIAMETER"].tolerance == 0.01
    assert operations["OP01"].params == {"x": 60.0, "y": 40.0, "z": 4.0}
    assert operations["OP02"].params == {"x": 0.0, "y": 0.0, "diameter": 5.4}
    assert [output.feature_id for output in operations["OP02"].outputs] == [
        "mount_hole", "mount_hole_axis"
    ]


def test_lab001_real_r01_has_exactly_one_expected_blocking_failure(tmp_path):
    spec, plan = lab001_case()
    r01 = run_pipeline(spec, plan, tmp_path)
    failures = [
        check for check in r01.validation_report.checks
        if check.blocking and check.status == CheckStatus.FAIL
    ]

    assert r01.validation_report.status == ReportStatus.FAIL
    assert len(failures) == 1
    assert failures[0].constraint_id == "C_HOLE_DIAMETER"
    assert failures[0].actual == pytest.approx(5.4)
    assert failures[0].expected["target"] == 6.0
    assert failures[0].expected["tolerance"] == 0.01


def test_lab001_full_m3_boundary_reaches_validated_r02(validated_lab):
    result, result_path, backend = validated_lab

    assert result.status == LabStatus.VALIDATED
    assert result.stage == "COMPLETED"
    assert result.r01_blocking_failures == ["C_HOLE_DIAMETER"]
    assert result.planner_run.attempt_count == 1
    assert len(backend.requests) == 1
    action = result.planner_response.root.repair_plan.actions[0]
    assert action.target.semantic_path == "OP02.params.diameter"
    assert (action.old_value, action.new_value) == (5.4, 6.0)
    assert action.constraint_ids == ["C_HOLE_DIAMETER"]
    assert action.related_feature_ids == ["mount_hole"]
    assert result.repair_plan_validation.status.value == "VALID"
    assert result.repair_executor_status.value == "REPAIRED"
    assert result.r02_revision_id == "R02"
    assert result.r02_parent_revision_id == "R01"
    assert result.r02_validation_report.status == ReportStatus.PASS
    assert result.changed_semantic_paths == ["OP02.params.diameter"]
    assert result.change_locality and result.constraint_preservation
    assert set(result.preserved_constraint_ids) == {
        "C_VALID", "C_SOLID_COUNT", "C_WIDTH", "C_DEPTH", "C_HEIGHT"
    }
    assert result.feature_identity_preserved
    assert result.feature_ids_r01 == result.feature_ids_r02 == [
        "main_body", "mount_hole", "mount_hole_axis"
    ]
    assert set(result.artifacts) == {"R01", "R02"}
    assert all(set(items) == {"STEP", "STL"} for items in result.artifacts.values())
    assert all(Path(path).is_file() for items in result.artifacts.values() for path in items.values())
    assert result_path.is_file()
    assert LabResult.model_validate_json(result_path.read_text(encoding="utf-8")) == result


def test_lab001_invalid_json_is_fail_without_fallback_or_r02(tmp_path):
    backend = ScriptedFakeBackend(["not-json", _golden_raw(tmp_path)])
    planner = LLMRepairPlanner(backend, LLMPlannerConfig(max_attempts=1))
    result = run_lab001(planner, tmp_path / "invalid-json")

    assert result.status == LabStatus.FAIL
    assert result.reason_code == "OUTPUT_PARSE_ERROR"
    assert len(backend.requests) == 1
    assert "R02" not in result.artifacts
    assert result.repair_executor_status is None


def test_lab001_unsupported_is_fail_and_never_executes(tmp_path):
    raw = json.dumps({
        "status": "UNSUPPORTED",
        "reason_code": "NO_ALLOWED_REPAIR",
    })
    planner = LLMRepairPlanner(ScriptedFakeBackend([raw]), LLMPlannerConfig(max_attempts=1))
    result = run_lab001(planner, tmp_path / "unsupported")

    assert result.status == LabStatus.FAIL
    assert result.reason_code == "UNSUPPORTED:NO_ALLOWED_REPAIR"
    assert result.boundary_status.value == "UNSUPPORTED"
    assert result.repair_executor_status is None
    assert "R02" not in result.artifacts


def test_lab001_unavailable_model_is_blocked(tmp_path):
    unavailable = BackendResponse(
        status=BackendStatus.ERROR,
        error_code=BackendFailureCode.MODEL_UNAVAILABLE,
        message="llama-server unavailable",
    )
    planner = LLMRepairPlanner(
        ScriptedFakeBackend([unavailable]),
        LLMPlannerConfig(max_attempts=1),
    )
    result = run_lab001(planner, tmp_path / "blocked")

    assert result.status == LabStatus.BLOCKED
    assert result.reason_code == "MODEL_UNAVAILABLE"
    assert result.boundary_status.value == "PLANNER_ERROR"
    assert result.repair_executor_status is None
    assert "R02" not in result.artifacts
