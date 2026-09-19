from copy import deepcopy

import pytest
from pydantic import ValidationError

from cad_ai.cad import CADStatus
from cad_ai.cases import block_case, repair_case
from cad_ai.pipeline import run_pipeline, run_revision_cycle
from cad_ai.revision import (
    DeterministicRepairController,
    DeterministicRepairPlanner,
    RepairAction,
    RepairActionType,
    RepairExecutor,
    RepairPlan,
    RepairPlanErrorCode,
    RepairPlanningStatus,
    RepairPlanValidation,
    RepairPlanValidationError,
    RepairPlanValidationStatus,
    RepairPlanValidator,
    RepairRule,
    RepairStatus,
    changed_semantic_plan_paths,
)
from cad_ai.validation import CheckStatus, ReportStatus


RULES = [
    RepairRule(
        constraint_id="C_WIDTH",
        operation_id="OP01",
        parameter_name="x",
        related_feature_ids=["main_body"],
    )
]


@pytest.fixture(scope="module")
def boundary(tmp_path_factory):
    spec, source_plan = repair_case()
    source = run_pipeline(spec, source_plan, tmp_path_factory.mktemp("m2-source"))
    planner = DeterministicRepairPlanner(RULES)
    validator = RepairPlanValidator(RULES)
    executor = RepairExecutor()
    planning = planner.plan(spec, source_plan, source.validation_report)
    repair_plan = planning.repair_plan
    validation = validator.validate(repair_plan, spec, source_plan, source.validation_report)
    execution = executor.execute(repair_plan, validation, source_plan)
    return {
        "spec": spec,
        "source_plan": source_plan,
        "source": source,
        "planner": planner,
        "validator": validator,
        "executor": executor,
        "planning": planning,
        "repair_plan": repair_plan,
        "validation": validation,
        "execution": execution,
    }


@pytest.fixture(scope="module")
def cycle(tmp_path_factory):
    spec, source_plan = repair_case()
    controller = DeterministicRepairController(RULES)
    return run_revision_cycle(spec, source_plan, controller, tmp_path_factory.mktemp("m2-cycle"))


def _modified(plan: RepairPlan, mutate) -> RepairPlan:
    data = plan.model_dump(mode="json")
    mutate(data)
    return RepairPlan.model_validate(data)


def _error_codes(validation: RepairPlanValidation) -> set[RepairPlanErrorCode]:
    return {error.code for error in validation.errors}


def _checks(report):
    return {check.constraint_id: check for check in report.checks}


def test_rp001_repair_plan_serializes_and_deserializes(boundary):
    repair_plan = boundary["repair_plan"]
    assert RepairPlan.model_validate_json(repair_plan.model_dump_json()) == repair_plan


def test_rp002_valid_deterministic_failure_produces_repair_plan(boundary):
    planning = boundary["planning"]
    assert planning.status == RepairPlanningStatus.PLANNED
    assert planning.repair_plan.actions[0].constraint_ids == ["C_WIDTH"]
    assert planning.repair_plan.actions[0].old_value == 99.5
    assert planning.repair_plan.actions[0].new_value == 100.0


def test_rp003_planner_does_not_mutate_source_cadplan(boundary):
    source_plan = boundary["source_plan"]
    before = source_plan.model_dump(mode="json")
    boundary["planner"].plan(boundary["spec"], source_plan, boundary["source"].validation_report)
    assert source_plan.model_dump(mode="json") == before


def test_rp004_planner_does_not_mutate_designspec(boundary):
    spec = boundary["spec"]
    before = spec.model_dump(mode="json")
    boundary["planner"].plan(spec, boundary["source_plan"], boundary["source"].validation_report)
    assert spec.model_dump(mode="json") == before


def test_rp005_valid_repair_plan_passes_schema_and_semantic_validation(boundary):
    assert boundary["repair_plan"].actions[0].type == RepairActionType.SET_PARAMETER
    assert boundary["validation"].status == RepairPlanValidationStatus.VALID
    assert boundary["validation"].errors == []


def test_rp006_unknown_operation_is_rejected(boundary):
    repair_plan = _modified(boundary["repair_plan"], lambda data: data["actions"][0]["target"].update(operation_id="missing"))
    validation = boundary["validator"].validate(repair_plan, boundary["spec"], boundary["source_plan"], boundary["source"].validation_report)
    assert RepairPlanErrorCode.UNKNOWN_OPERATION in _error_codes(validation)


def test_rp007_unknown_parameter_is_rejected(boundary):
    repair_plan = _modified(boundary["repair_plan"], lambda data: data["actions"][0]["target"].update(parameter="missing"))
    validation = boundary["validator"].validate(repair_plan, boundary["spec"], boundary["source_plan"], boundary["source"].validation_report)
    assert RepairPlanErrorCode.UNKNOWN_PARAMETER in _error_codes(validation)


def test_rp008_stale_old_value_is_rejected(boundary):
    repair_plan = _modified(boundary["repair_plan"], lambda data: data["actions"][0].update(old_value=98.0))
    validation = boundary["validator"].validate(repair_plan, boundary["spec"], boundary["source_plan"], boundary["source"].validation_report)
    assert RepairPlanErrorCode.STALE_OLD_VALUE in _error_codes(validation)


def test_rp009_unknown_constraint_is_rejected(boundary):
    repair_plan = _modified(boundary["repair_plan"], lambda data: data["actions"][0].update(constraint_ids=["C_UNKNOWN"]))
    validation = boundary["validator"].validate(repair_plan, boundary["spec"], boundary["source_plan"], boundary["source"].validation_report)
    assert RepairPlanErrorCode.UNKNOWN_CONSTRAINT in _error_codes(validation)


def test_rp010_design_id_mismatch_is_rejected(boundary):
    repair_plan = _modified(boundary["repair_plan"], lambda data: data.update(design_id="OTHER"))
    validation = boundary["validator"].validate(repair_plan, boundary["spec"], boundary["source_plan"], boundary["source"].validation_report)
    assert RepairPlanErrorCode.DESIGN_MISMATCH in _error_codes(validation)


def test_rp011_spec_version_mismatch_is_rejected(boundary):
    repair_plan = _modified(boundary["repair_plan"], lambda data: data.update(spec_version="2.0"))
    validation = boundary["validator"].validate(repair_plan, boundary["spec"], boundary["source_plan"], boundary["source"].validation_report)
    assert RepairPlanErrorCode.SPEC_VERSION_MISMATCH in _error_codes(validation)


def test_rp012_source_revision_mismatch_is_rejected(boundary):
    repair_plan = _modified(boundary["repair_plan"], lambda data: data.update(source_revision_id="R00"))
    validation = boundary["validator"].validate(repair_plan, boundary["spec"], boundary["source_plan"], boundary["source"].validation_report)
    assert RepairPlanErrorCode.REVISION_MISMATCH in _error_codes(validation)


def test_rp013_source_and_target_revision_cannot_match(boundary):
    data = boundary["repair_plan"].model_dump(mode="json")
    data["target_revision_id"] = data["source_revision_id"]
    with pytest.raises(ValidationError, match="must differ"):
        RepairPlan.model_validate(data)


def test_rp014_forbidden_action_type_is_rejected_by_schema(boundary):
    data = boundary["repair_plan"].model_dump(mode="json")
    data["actions"][0]["type"] = "CHANGE_DESIGN_SPEC"
    with pytest.raises(ValidationError):
        RepairPlan.model_validate(data)


def test_rp015_executor_refuses_unvalidated_plan_without_mutation(boundary):
    source_plan = boundary["source_plan"]
    before = source_plan.model_dump(mode="json")
    invalid = RepairPlanValidation(
        status=RepairPlanValidationStatus.INVALID,
        errors=[RepairPlanValidationError(code=RepairPlanErrorCode.ACTION_NOT_ALLOWED, message="not authorized")],
    )
    result = boundary["executor"].execute(boundary["repair_plan"], invalid, source_plan)
    assert result.status == RepairStatus.ERROR
    assert result.result_code == "REPAIR_PLAN_NOT_VALIDATED"
    assert source_plan.model_dump(mode="json") == before


def test_rp016_executor_applies_exactly_set_parameter(boundary):
    result = boundary["execution"]
    assert result.status == RepairStatus.REPAIRED
    assert changed_semantic_plan_paths(boundary["source_plan"], result.revised_plan) == {"OP01.params.x"}
    assert result.revised_plan.operations[0].params["x"] == 100.0


def test_rp017_executor_creates_r02(boundary):
    assert boundary["execution"].revised_plan.revision_id == "R02"


def test_rp018_r02_parent_is_r01(boundary):
    assert boundary["execution"].revised_plan.parent_revision_id == "R01"


def test_rp019_design_id_is_preserved(boundary):
    assert boundary["execution"].revised_plan.design_id == boundary["source_plan"].design_id == "D001"


def test_rp020_spec_version_is_preserved(boundary):
    assert boundary["execution"].revised_plan.spec_version == boundary["source_plan"].spec_version == "1.0"


def test_rp021_semantic_feature_ids_are_preserved(cycle):
    before = set(cycle.r01.cad_result.feature_registry.features)
    after = set(cycle.r02.cad_result.feature_registry.features)
    assert before == after


def test_rp022_repair_action_matches_actual_plan_diff(boundary):
    planned_paths = {action.target.semantic_path for action in boundary["repair_plan"].actions}
    actual_paths = changed_semantic_plan_paths(boundary["source_plan"], boundary["execution"].revised_plan)
    assert planned_paths == actual_paths == {"OP01.params.x"}


def test_rp023_change_locality_remains_pass(cycle):
    assert cycle.change_locality
    assert cycle.changed_paths == {"OP01.params.x"}


def test_rp024_constraint_preservation_remains_pass(cycle):
    assert cycle.constraint_preservation
    assert set(cycle.preserved_constraint_ids) == {
        "C_VALID", "C_SOLID_COUNT", "C_DEPTH", "C_HEIGHT", "C_HOLE_DIAMETER", "C_HOLE_SPACING"
    }


def test_rp025_r02_executes_through_real_cad_engine(cycle):
    assert cycle.r02.cad_result.status == CADStatus.SUCCESS
    assert cycle.r02.cad_result.model_handle.isValid()
    assert cycle.r02.cad_result.model_handle is not cycle.r01.cad_result.model_handle


def test_rp026_target_constraint_transitions_fail_to_pass(cycle):
    assert _checks(cycle.r01.validation_report)["C_WIDTH"].status == CheckStatus.FAIL
    assert _checks(cycle.r02.validation_report)["C_WIDTH"].status == CheckStatus.PASS


def test_rp027_final_validation_report_is_pass(cycle):
    assert cycle.r01.validation_report.status == ReportStatus.FAIL
    assert cycle.r02.validation_report.status == ReportStatus.PASS


def test_rp028_unsupported_report_does_not_produce_executable_plan(tmp_path):
    spec, source_plan = block_case("fail")
    source = run_pipeline(spec, source_plan, tmp_path)
    planner = DeterministicRepairPlanner([
        RepairRule(constraint_id="width", operation_id="op-main-body", parameter_name="x")
    ])
    result = planner.plan(spec, source_plan, source.validation_report)
    assert result.status == RepairPlanningStatus.UNSUPPORTED
    assert result.repair_plan is None


def test_schema_rejects_empty_actions(boundary):
    data = boundary["repair_plan"].model_dump(mode="json")
    data["actions"] = []
    with pytest.raises(ValidationError):
        RepairPlan.model_validate(data)


def test_schema_rejects_duplicate_action_ids(boundary):
    data = boundary["repair_plan"].model_dump(mode="json")
    data["actions"].append(deepcopy(data["actions"][0]))
    with pytest.raises(ValidationError, match="unique"):
        RepairPlan.model_validate(data)


def test_schema_rejects_incompatible_repair_values(boundary):
    data = boundary["repair_plan"].model_dump(mode="json")
    data["actions"][0]["new_value"] = "100.0"
    with pytest.raises(ValidationError, match="numeric JSON"):
        RepairPlan.model_validate(data)


def test_semantic_validation_rejects_unknown_related_feature(boundary):
    repair_plan = _modified(boundary["repair_plan"], lambda data: data["actions"][0].update(related_feature_ids=["missing"]))
    validation = boundary["validator"].validate(repair_plan, boundary["spec"], boundary["source_plan"], boundary["source"].validation_report)
    assert RepairPlanErrorCode.UNKNOWN_FEATURE in _error_codes(validation)


def test_executor_rejects_plan_changed_after_validation(boundary):
    changed = _modified(boundary["repair_plan"], lambda data: data.update(rationale="tampered after validation"))
    result = boundary["executor"].execute(changed, boundary["validation"], boundary["source_plan"])
    assert result.status == RepairStatus.ERROR
    assert result.result_code == "REPAIR_PLAN_NOT_VALIDATED"


def test_semantic_validation_rejects_non_next_target_revision(boundary):
    repair_plan = _modified(boundary["repair_plan"], lambda data: data.update(target_revision_id="R03"))
    validation = boundary["validator"].validate(repair_plan, boundary["spec"], boundary["source_plan"], boundary["source"].validation_report)
    assert RepairPlanErrorCode.TARGET_REVISION_INVALID in _error_codes(validation)


def test_semantic_validation_rejects_mismatched_source_report(boundary):
    mismatched_report = boundary["source"].validation_report.model_copy(update={"revision_id": "R99"})
    validation = boundary["validator"].validate(boundary["repair_plan"], boundary["spec"], boundary["source_plan"], mismatched_report)
    assert RepairPlanErrorCode.SOURCE_REPORT_MISMATCH in _error_codes(validation)


def test_schema_cannot_express_design_spec_mutation(boundary):
    data = boundary["repair_plan"].model_dump(mode="json")
    data["actions"][0]["design_spec_patch"] = {"constraints": []}
    with pytest.raises(ValidationError):
        RepairPlan.model_validate(data)
