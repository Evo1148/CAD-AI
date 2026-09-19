import json
from pathlib import Path

import pytest

from cad_ai.contracts import CADPlan
from cad_ai.generation import M4Intent
from cad_ai.m4 import M4_001_PROMPT
from cad_ai.m5 import M5Result, M5Status, run_m5_001
from cad_ai.planning import (
    LLMPlannerConfig,
    LLMRepairPlanner,
    RepairPlannerResponse,
    ScriptedFakeBackend,
)
from cad_ai.revision import (
    RepairAction,
    RepairActionType,
    RepairPlan,
    RepairTarget,
)
from cad_ai.validation import CheckStatus, ReportStatus


def _intent_backend() -> ScriptedFakeBackend:
    return ScriptedFakeBackend([M4Intent(
        width=60.0,
        depth=40.0,
        height=4.0,
        hole_diameter=6.0,
        centered=True,
    ).model_dump_json()])


def _diameter_fault(plan: CADPlan) -> CADPlan:
    data = plan.model_dump(mode="python")
    operations = {operation["operation_id"]: operation for operation in data["operations"]}
    operations["OP02"]["params"]["diameter"] = 5.4
    return CADPlan.model_validate(data)


def _valid_repair_response() -> str:
    return RepairPlannerResponse.model_validate({
        "status": "PLAN",
        "repair_plan": {
            "design_id": "M4-001",
            "spec_version": "1.0",
            "source_revision_id": "R01",
            "target_revision_id": "R02",
            "actions": [{
                "action_id": "RA001",
                "type": "SET_PARAMETER",
                "target": {"operation_id": "OP02", "parameter": "diameter"},
                "old_value": 5.4,
                "new_value": 6.0,
                "constraint_ids": ["C_HOLE_DIAMETER"],
                "related_feature_ids": ["mount_hole"],
                "reason": "C_HOLE_DIAMETER failed dimensional validation",
            }],
        },
    }).model_dump_json()


def _llm_planner(raw_response: str) -> LLMRepairPlanner:
    return LLMRepairPlanner(
        ScriptedFakeBackend([raw_response]),
        LLMPlannerConfig(max_attempts=1, temperature=0.0, max_tokens=1024, seed=12345),
    )


class _MustNotRunPlanner:
    def plan(self, context):
        raise AssertionError("repair planner must not be called when R01 passes")


class _CapturingPlanner:
    def __init__(self, delegate):
        self.delegate = delegate
        self.context = None
        self.calls = 0

    def plan(self, context):
        self.context = context
        self.calls += 1
        return self.delegate.plan(context)


def test_m5_r01_pass_exports_without_invoking_repair(tmp_path):
    result_path = tmp_path / "result.json"

    result = run_m5_001(
        _intent_backend(),
        tmp_path,
        repair_planner=_MustNotRunPlanner(),
        result_json=result_path,
    )

    assert result.status == M5Status.VALIDATED
    assert result.stage == "COMPLETE"
    assert result.r01_validation_report.status == ReportStatus.PASS
    assert result.repair_attempted is False
    assert result.planning_context is None
    assert result.r02_plan is None
    assert result.final_revision_id == "R01"
    assert result.final_artifacts == result.artifacts["R01"]
    assert set(result.final_artifacts) == {"STEP", "STL"}
    assert all(Path(path).is_file() for path in result.final_artifacts.values())
    assert M5Result.model_validate_json(result_path.read_text(encoding="utf-8")) == result


def test_m5_real_r01_failure_builds_correct_repair_context(tmp_path):
    planner = _CapturingPlanner(_llm_planner(_valid_repair_response()))

    result = run_m5_001(
        _intent_backend(),
        tmp_path,
        repair_planner=planner,
        test_plan_transform=_diameter_fault,
    )

    assert planner.calls == 1
    context = planner.context
    assert [failure.constraint_id for failure in context.failed_constraints] == ["C_HOLE_DIAMETER"]
    failure = context.failed_constraints[0]
    assert failure.actual == pytest.approx(5.4)
    assert failure.expected["target"] == 6.0
    assert failure.related_feature_ids == ["mount_hole"]
    assert {item.constraint_id for item in context.protected_constraints} == {
        "C_VALID", "C_SOLID_COUNT", "C_WIDTH", "C_DEPTH", "C_HEIGHT",
    }
    operation = next(item for item in context.repairable_operations if item.operation_id == "OP02")
    parameter = operation.repairable_parameters[0]
    assert operation.operation_type == "hole"
    assert parameter.name == "diameter"
    assert parameter.current_value == 5.4
    assert parameter.constraint_ids == ["C_HOLE_DIAMETER"]
    assert result.status == M5Status.VALIDATED


def test_m5_valid_plan_creates_preserved_r02_and_final_artifacts(tmp_path):
    result = run_m5_001(
        _intent_backend(),
        tmp_path,
        repair_planner=_llm_planner(_valid_repair_response()),
        test_plan_transform=_diameter_fault,
    )

    assert result.status == M5Status.VALIDATED
    assert result.plan_generation.cad_plan.operations[1].params["diameter"] == 6.0
    assert result.r01_plan.revision_id == "R01"
    assert result.r01_plan.parent_revision_id is None
    assert result.r01_plan.operations[1].params["diameter"] == 5.4
    assert result.r01_validation_report.status == ReportStatus.FAIL
    failures = [
        check for check in result.r01_validation_report.checks
        if check.blocking and check.status == CheckStatus.FAIL
    ]
    assert [check.constraint_id for check in failures] == ["C_HOLE_DIAMETER"]
    assert result.boundary_status.value == "APPLIED"
    assert result.repair_plan_validation.status.value == "VALID"
    assert result.repair_executor_status.value == "REPAIRED"
    assert result.r02_plan.revision_id == "R02"
    assert result.r02_plan.parent_revision_id == "R01"
    assert result.r02_plan.operations[1].params["diameter"] == 6.0
    assert result.r02_validation_report.status == ReportStatus.PASS
    assert result.changed_semantic_paths == ["OP02.params.diameter"]
    assert result.change_locality is True
    assert result.constraint_preservation is True
    assert result.feature_identity_preserved is True
    assert result.feature_ids_r01 == result.feature_ids_r02 == [
        "main_body", "mount_hole", "mount_hole_axis",
    ]
    assert result.final_revision_id == "R02"
    assert result.final_artifacts == result.artifacts["R02"]
    assert result.final_artifacts != result.artifacts["R01"]
    assert all(Path(path).is_file() for path in result.final_artifacts.values())
    assert all("M4-001_R02" in Path(path).name for path in result.final_artifacts.values())


def test_m5_unsupported_finishes_structurally_without_fake_r02(tmp_path):
    planner = _llm_planner(json.dumps({
        "status": "UNSUPPORTED",
        "reason_code": "NO_ALLOWED_REPAIR",
    }))

    result = run_m5_001(
        _intent_backend(),
        tmp_path,
        repair_planner=planner,
        test_plan_transform=_diameter_fault,
    )

    assert result.status == M5Status.FAIL
    assert result.reason_code == "UNSUPPORTED:NO_ALLOWED_REPAIR"
    assert result.boundary_status.value == "UNSUPPORTED"
    assert result.r02_plan is None
    assert "R02" not in result.artifacts
    assert result.final_revision_id is None
    assert result.final_artifacts == {}


def test_m5_planner_error_finishes_structurally(tmp_path):
    planner = _llm_planner(json.dumps({
        "status": "ERROR",
        "error": {"code": "MODEL_ERROR", "message": "controlled planner failure"},
    }))

    result = run_m5_001(
        _intent_backend(),
        tmp_path,
        repair_planner=planner,
        test_plan_transform=_diameter_fault,
    )

    assert result.status == M5Status.FAIL
    assert result.reason_code == "MODEL_ERROR"
    assert result.boundary_status.value == "PLANNER_ERROR"
    assert result.r02_plan is None
    assert result.repair_executor_status is None


def test_m5_unauthorized_repair_is_rejected_before_executor(tmp_path):
    unauthorized = RepairPlannerResponse.model_validate({
        "status": "PLAN",
        "repair_plan": RepairPlan(
            design_id="M4-001",
            spec_version="1.0",
            source_revision_id="R01",
            target_revision_id="R02",
            actions=[RepairAction(
                action_id="RA001",
                type=RepairActionType.SET_PARAMETER,
                target=RepairTarget(operation_id="OP01", parameter="x"),
                old_value=60.0,
                new_value=6.0,
                constraint_ids=["C_HOLE_DIAMETER"],
                related_feature_ids=["mount_hole"],
                reason="attempt an unauthorized target",
            )],
        ).model_dump(mode="json"),
    }).model_dump_json()

    result = run_m5_001(
        _intent_backend(),
        tmp_path,
        repair_planner=_llm_planner(unauthorized),
        test_plan_transform=_diameter_fault,
    )

    assert result.status == M5Status.FAIL
    assert result.reason_code == "REPAIR_PLAN_REJECTED"
    assert result.repair_plan_validation.status.value == "INVALID"
    assert "ACTION_NOT_ALLOWED" in {
        error.code.value for error in result.repair_plan_validation.errors
    }
    assert result.repair_executor_status is None
    assert result.r02_plan is None
    assert result.r01_plan.operations[1].params["diameter"] == 5.4
    assert "R02" not in result.artifacts
