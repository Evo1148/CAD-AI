from copy import deepcopy
import json

import pytest

from cad_ai.cases import repair_case
from cad_ai.pipeline import run_pipeline
from cad_ai.planning import (
    BackendFailureCode,
    BackendResponse,
    BackendStatus,
    BoundaryStatus,
    DeterministicPlannerAdapter,
    ErrorPlannerResponse,
    LLMPlannerConfig,
    LLMRepairPlanner,
    PlanPlannerResponse,
    PlannerError,
    PlannerErrorCode,
    PROMPT_VERSION,
    RepairPlannerResponse,
    RepairPlanningBoundary,
    RepairPlanningContext,
    RepairPlanningContextBuilder,
    ScriptedFakeBackend,
    UnsupportedPlannerResponse,
    UnsupportedReasonCode,
)
from cad_ai.revision import (
    RepairExecutor,
    RepairPlan,
    RepairPlanErrorCode,
    RepairPlanValidationStatus,
    RepairPlanValidator,
    RepairRule,
    RepairStatus,
)


RULES = [RepairRule(constraint_id="C_WIDTH", operation_id="OP01", parameter_name="x", related_feature_ids=["main_body"])]


@pytest.fixture()
def planning_state(tmp_path):
    spec, plan = repair_case()
    source = run_pipeline(spec, plan, tmp_path)
    context = RepairPlanningContextBuilder().build(spec, plan, source.validation_report, RULES)
    deterministic = DeterministicPlannerAdapter().plan(context)
    return spec, plan, source.validation_report, context, deterministic


def _valid_raw(invocation) -> str:
    return invocation.response.model_dump_json()


def _unsafe_raw(invocation, operation_id: str = "OP99") -> str:
    data = invocation.response.model_dump(mode="json")
    data["repair_plan"]["actions"][0]["target"]["operation_id"] = operation_id
    return json.dumps(data)


def _semantic_defect_raw(invocation, defect: str) -> str:
    data = invocation.response.model_dump(mode="json")
    plan = data["repair_plan"]
    action = plan["actions"][0]
    if defect == "unknown_operation":
        action["target"]["operation_id"] = "OP99"
    elif defect == "unknown_parameter":
        action["target"]["parameter"] = "nonexistent"
    elif defect == "stale_old_value":
        action["old_value"] = 98.0
    elif defect == "design_mismatch":
        plan["design_id"] = "OTHER"
    elif defect == "spec_mismatch":
        plan["spec_version"] = "99.0"
    elif defect == "revision_mismatch":
        plan["source_revision_id"] = "R99"
    elif defect == "unknown_constraint":
        action["constraint_ids"] = ["C_UNKNOWN"]
    elif defect == "conflicting_actions":
        duplicate = deepcopy(action)
        duplicate["action_id"] = "RA002"
        duplicate["new_value"] = 101.0
        plan["actions"].append(duplicate)
    else:
        raise AssertionError(f"unknown test defect: {defect}")
    return json.dumps(data)


def test_m3001_context_round_trip(planning_state):
    context = planning_state[3]
    assert RepairPlanningContext.model_validate_json(context.model_dump_json()) == context


def test_m3002_context_contains_only_permitted_targets(planning_state):
    context = planning_state[3]
    targets = {(op.operation_id, param.name) for op in context.repairable_operations for param in op.repairable_parameters}
    assert targets == {("OP01", "x")}


def test_m3003_failed_constraints_are_structurally_derived(planning_state):
    failure = planning_state[3].failed_constraints
    assert len(failure) == 1
    assert (failure[0].constraint_id, failure[0].measurement, failure[0].actual) == ("C_WIDTH", "EXTENT_X", 99.5)
    assert failure[0].expected["target"] == 100.0


def test_m3004_protected_constraints_are_derived(planning_state):
    protected = {item.constraint_id for item in planning_state[3].protected_constraints}
    assert protected == {"C_VALID", "C_SOLID_COUNT", "C_DEPTH", "C_HEIGHT", "C_HOLE_DIAMETER", "C_HOLE_SPACING"}


def test_m3005_context_omits_unrelated_internal_data(planning_state):
    context = planning_state[3]
    assert set(context.model_dump()) == {
        "contract_version", "design_id", "spec_version", "revision_id", "failed_constraints",
        "protected_constraints", "repairable_operations", "allowed_action_types",
        "previous_attempt_error", "non_authoritative_hint",
    }
    serialized = context.model_dump_json().lower()
    assert all(forbidden not in serialized for forbidden in ("model_handle", "step", "stl", "filesystem", "cadquery", "locator"))


def test_prompt_m3a3_defines_context_bounded_repair_semantics(planning_state):
    context = planning_state[3]
    generic_operation = context.repairable_operations[0].model_copy(
        update={"operation_type": "generic_cad_operation"},
    )
    generic_context = context.model_copy(update={"repairable_operations": [generic_operation]})
    raw = RepairPlannerResponse(
        root=UnsupportedPlannerResponse(reason_code=UnsupportedReasonCode.NO_ALLOWED_REPAIR),
    ).model_dump_json()
    backend = ScriptedFakeBackend([raw])

    invocation = LLMRepairPlanner(backend, LLMPlannerConfig(max_attempts=1)).plan(generic_context)
    prompt = backend.requests[0].system_prompt

    assert generic_operation.operation_type == "generic_cad_operation"
    assert [action.value for action in generic_context.allowed_action_types] == ["SET_PARAMETER"]
    assert [parameter.name for parameter in generic_operation.repairable_parameters] == ["x"]
    assert "operation_type describes the CAD operation type; it is not a repair action type." in prompt
    assert "allowed_action_types describes the permitted repair action types." in prompt
    assert "Use only information explicitly represented in RepairPlanningContext." in prompt
    assert (
        "Do not speculate about hidden CAD effects, unrepresented constraints, kernel behavior, "
        "or risks outside the context."
    ) in prompt
    assert (
        "Together, allowed_action_types and repairable_parameters define the actions and targets "
        "you may propose."
    ) in prompt
    assert (
        "When SET_PARAMETER is allowed, it may target any parameter explicitly listed in "
        "repairable_parameters, regardless of operation_type."
    ) in prompt
    assert (
        "preserve protected_constraints according to the information available in the context."
    ) in prompt
    assert (
        "Final authorization and execution safety belong to RepairPlanValidator and RepairExecutor; "
        "you only propose."
    ) in prompt
    assert (
        "Return UNSUPPORTED only when the context does not permit an unambiguous repair using the "
        "exposed actions and parameters, not for hypothetical risks outside the context."
    ) in prompt
    assert all(term not in prompt.casefold() for term in ("lab-001", "hole", "diameter", "5.4", "6.0"))
    assert PROMPT_VERSION == "m3-a.3"
    assert invocation.planner_run.prompt_version == "m3-a.3"


def test_m3006_plan_response_parses(planning_state):
    restored = RepairPlannerResponse.model_validate_json(_valid_raw(planning_state[4]))
    assert isinstance(restored.root, PlanPlannerResponse)


def test_m3007_unsupported_response_parses():
    raw = '{"status":"UNSUPPORTED","reason_code":"NO_ALLOWED_REPAIR"}'
    response = RepairPlannerResponse.model_validate_json(raw)
    assert isinstance(response.root, UnsupportedPlannerResponse)


def test_m3008_error_response_parses():
    raw = '{"status":"ERROR","error":{"code":"MODEL_ERROR"}}'
    response = RepairPlannerResponse.model_validate_json(raw)
    assert isinstance(response.root, ErrorPlannerResponse)


def test_m3009_malformed_output_becomes_parse_error(planning_state):
    planner = LLMRepairPlanner(ScriptedFakeBackend(["not-json"]), LLMPlannerConfig(max_attempts=1))
    invocation = planner.plan(planning_state[3])
    assert invocation.response.root.error.code == PlannerErrorCode.OUTPUT_PARSE_ERROR
    assert not invocation.schema_valid


def test_m3010_schema_invalid_output_becomes_schema_error(planning_state):
    planner = LLMRepairPlanner(ScriptedFakeBackend(['{"status":"PLAN"}']), LLMPlannerConfig(max_attempts=1))
    invocation = planner.plan(planning_state[3])
    assert invocation.response.root.error.code == PlannerErrorCode.SCHEMA_VALIDATION_ERROR


def test_m3011_schema_valid_unsafe_plan_reaches_semantic_validator(planning_state):
    spec, plan, report, context, deterministic = planning_state
    invocation = LLMRepairPlanner(ScriptedFakeBackend([_unsafe_raw(deterministic)]), LLMPlannerConfig(max_attempts=1)).plan(context)
    assert invocation.schema_valid
    result = RepairPlanningBoundary(RepairPlanValidator(RULES)).process(invocation, spec, plan, report)
    assert result.status == BoundaryStatus.REPAIR_PLAN_REJECTED
    assert RepairPlanErrorCode.UNKNOWN_OPERATION in {error.code for error in result.repair_plan_validation.errors}


class CountingExecutor(RepairExecutor):
    def __init__(self):
        self.calls = 0

    def execute(self, repair_plan, validation, source_plan):
        self.calls += 1
        return super().execute(repair_plan, validation, source_plan)


def test_m3012_rejected_plan_never_reaches_executor(planning_state):
    spec, plan, report, context, deterministic = planning_state
    invocation = LLMRepairPlanner(ScriptedFakeBackend([_unsafe_raw(deterministic)]), LLMPlannerConfig(max_attempts=1)).plan(context)
    executor = CountingExecutor()
    result = RepairPlanningBoundary(RepairPlanValidator(RULES), executor).process(invocation, spec, plan, report)
    assert result.status == BoundaryStatus.REPAIR_PLAN_REJECTED
    assert executor.calls == 0


def test_m3013_valid_plan_reaches_executor(planning_state):
    spec, plan, report, _, invocation = planning_state
    executor = CountingExecutor()
    result = RepairPlanningBoundary(RepairPlanValidator(RULES), executor).process(invocation, spec, plan, report)
    assert result.status == BoundaryStatus.APPLIED
    assert executor.calls == 1 and result.repair_result.status == RepairStatus.REPAIRED


def test_m3014_llm_cannot_bypass_validator_with_schema_valid_plan(planning_state):
    spec, plan, report, context, deterministic = planning_state
    before = plan.model_dump(mode="json")
    invocation = LLMRepairPlanner(ScriptedFakeBackend([_unsafe_raw(deterministic)]), LLMPlannerConfig(max_attempts=1)).plan(context)
    boundary = RepairPlanningBoundary(RepairPlanValidator(RULES))
    assert boundary.process(invocation, spec, plan, report).status == BoundaryStatus.REPAIR_PLAN_REJECTED
    assert plan.model_dump(mode="json") == before


@pytest.mark.parametrize(("defect", "expected_code"), [
    ("unknown_operation", RepairPlanErrorCode.UNKNOWN_OPERATION),
    ("unknown_parameter", RepairPlanErrorCode.UNKNOWN_PARAMETER),
    ("stale_old_value", RepairPlanErrorCode.STALE_OLD_VALUE),
    ("design_mismatch", RepairPlanErrorCode.DESIGN_MISMATCH),
    ("spec_mismatch", RepairPlanErrorCode.SPEC_VERSION_MISMATCH),
    ("revision_mismatch", RepairPlanErrorCode.REVISION_MISMATCH),
    ("unknown_constraint", RepairPlanErrorCode.UNKNOWN_CONSTRAINT),
    ("conflicting_actions", RepairPlanErrorCode.DUPLICATE_TARGET),
])
def test_schema_valid_semantic_defects_never_execute_or_mutate_source(planning_state, defect, expected_code):
    spec, plan, report, context, deterministic = planning_state
    before = plan.model_dump(mode="json")
    raw = _semantic_defect_raw(deterministic, defect)
    invocation = LLMRepairPlanner(ScriptedFakeBackend([raw]), LLMPlannerConfig(max_attempts=1)).plan(context)
    executor = CountingExecutor()

    result = RepairPlanningBoundary(RepairPlanValidator(RULES), executor).process(invocation, spec, plan, report)

    assert invocation.schema_valid
    assert result.status == BoundaryStatus.REPAIR_PLAN_REJECTED
    assert expected_code in {error.code for error in result.repair_plan_validation.errors}
    assert executor.calls == 0
    assert plan.model_dump(mode="json") == before


def test_m3015_downstream_is_deterministic_for_same_validated_plan(planning_state):
    spec, plan, report, _, invocation = planning_state
    boundary = RepairPlanningBoundary(RepairPlanValidator(RULES))
    first = boundary.process(invocation, spec, plan, report).repair_result.revised_plan
    second = boundary.process(invocation, spec, plan, report).repair_result.revised_plan
    assert first == second


def test_m3016_retry_occurs_on_recoverable_output_failure(planning_state):
    backend = ScriptedFakeBackend(["not-json", _valid_raw(planning_state[4])])
    invocation = LLMRepairPlanner(backend, LLMPlannerConfig(max_attempts=2)).plan(planning_state[3])
    assert invocation.response.status == "PLAN"
    assert invocation.planner_run.attempt_count == 2
    assert backend.requests[1].context.previous_attempt_error.code == "OUTPUT_PARSE_ERROR"


def test_m3017_retry_stops_at_max_attempts(planning_state):
    backend = ScriptedFakeBackend(["bad", "still-bad", _valid_raw(planning_state[4])])
    invocation = LLMRepairPlanner(backend, LLMPlannerConfig(max_attempts=2)).plan(planning_state[3])
    assert invocation.response.root.error.code == PlannerErrorCode.RETRY_EXHAUSTED
    assert invocation.response.root.error.cause_code == PlannerErrorCode.OUTPUT_PARSE_ERROR
    assert len(backend.requests) == 2


def test_m3018_unsupported_does_not_retry(planning_state):
    raw = RepairPlannerResponse(root=UnsupportedPlannerResponse(reason_code=UnsupportedReasonCode.NO_ALLOWED_REPAIR)).model_dump_json()
    backend = ScriptedFakeBackend([raw, _valid_raw(planning_state[4])])
    invocation = LLMRepairPlanner(backend, LLMPlannerConfig(max_attempts=2)).plan(planning_state[3])
    assert invocation.response.status == "UNSUPPORTED"
    assert len(backend.requests) == 1


def test_m3019_timeout_becomes_structured_error(planning_state):
    backend = ScriptedFakeBackend([BackendResponse(status=BackendStatus.ERROR, error_code=BackendFailureCode.MODEL_TIMEOUT)])
    invocation = LLMRepairPlanner(backend, LLMPlannerConfig(max_attempts=1)).plan(planning_state[3])
    assert invocation.response.root.error.code == PlannerErrorCode.MODEL_TIMEOUT


def test_m3020_unavailable_is_permanent_and_does_not_retry(planning_state):
    backend = ScriptedFakeBackend([
        BackendResponse(status=BackendStatus.ERROR, error_code=BackendFailureCode.MODEL_UNAVAILABLE),
        _valid_raw(planning_state[4]),
    ])
    invocation = LLMRepairPlanner(backend, LLMPlannerConfig(max_attempts=2)).plan(planning_state[3])
    assert invocation.response.root.error.code == PlannerErrorCode.MODEL_UNAVAILABLE
    assert len(backend.requests) == 1


def test_m3021_fake_backend_valid_plan_end_to_end(planning_state):
    spec, plan, report, context, deterministic = planning_state
    invocation = LLMRepairPlanner(ScriptedFakeBackend([_valid_raw(deterministic)])).plan(context)
    result = RepairPlanningBoundary(RepairPlanValidator(RULES)).process(invocation, spec, plan, report)
    assert result.status == BoundaryStatus.APPLIED
    assert result.repair_result.revised_plan.revision_id == "R02"


def test_m3022_fake_backend_unsupported(planning_state):
    raw = RepairPlannerResponse(root=UnsupportedPlannerResponse(reason_code=UnsupportedReasonCode.REQUIRES_REPLAN)).model_dump_json()
    invocation = LLMRepairPlanner(ScriptedFakeBackend([raw])).plan(planning_state[3])
    result = RepairPlanningBoundary(RepairPlanValidator(RULES)).process(invocation, *planning_state[:3])
    assert result.status == BoundaryStatus.UNSUPPORTED


def test_m3023_fake_backend_semantic_rejection(planning_state):
    spec, plan, report, context, deterministic = planning_state
    invocation = LLMRepairPlanner(ScriptedFakeBackend([_unsafe_raw(deterministic)]), LLMPlannerConfig(max_attempts=1)).plan(context)
    result = RepairPlanningBoundary(RepairPlanValidator(RULES)).process(invocation, spec, plan, report)
    assert result.status == BoundaryStatus.REPAIR_PLAN_REJECTED
    assert result.repair_result is None


def test_m3024_fake_backend_malformed_output(planning_state):
    invocation = LLMRepairPlanner(ScriptedFakeBackend(["{"]), LLMPlannerConfig(max_attempts=1)).plan(planning_state[3])
    assert invocation.response.status == "ERROR" and not invocation.schema_valid


def test_m3025_fake_backend_timeout_simulation(planning_state):
    timeout = BackendResponse(status=BackendStatus.ERROR, error_code=BackendFailureCode.MODEL_TIMEOUT)
    invocation = LLMRepairPlanner(ScriptedFakeBackend([timeout]), LLMPlannerConfig(max_attempts=1)).plan(planning_state[3])
    assert invocation.planner_run.error_code == PlannerErrorCode.MODEL_TIMEOUT


def test_planner_response_rejects_arbitrary_action_type(planning_state):
    data = planning_state[4].response.model_dump(mode="json")
    data["repair_plan"]["actions"][0]["type"] = "EXEC_PYTHON"
    with pytest.raises(Exception):
        RepairPlannerResponse.model_validate(data)
