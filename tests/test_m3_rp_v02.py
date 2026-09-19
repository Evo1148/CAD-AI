import json

import pytest

from cad_ai.benchmark.comparison import METRIC_FIELDS, compare_benchmark_summaries
from cad_ai.benchmark.m3 import (
    SUITE_V01,
    SUITE_V02,
    benchmark_cases,
    benchmark_cases_v02,
    build_cli_parser,
    run_benchmark,
    run_case,
)
from cad_ai.planning import (
    LLMPlannerConfig,
    LLMRepairPlanner,
    RepairPlannerResponse,
    RepairPlanningContextBuilder,
    ScriptedFakeBackend,
    UnsupportedPlannerResponse,
    UnsupportedReasonCode,
    authorized_set_parameter_targets,
)
from cad_ai.pipeline import run_pipeline


@pytest.fixture(scope="module")
def v02_deterministic(tmp_path_factory):
    return run_benchmark(
        "deterministic",
        artifact_root=tmp_path_factory.mktemp("m3-rp-v02-det"),
        suite=SUITE_V02,
    )


@pytest.fixture(scope="module")
def v02_fake(tmp_path_factory):
    return run_benchmark(
        "fake-llm",
        artifact_root=tmp_path_factory.mktemp("m3-rp-v02-fake"),
        suite=SUITE_V02,
    )


def test_v01_case_set_and_objects_remain_historical():
    assert list(benchmark_cases(SUITE_V01)) == ["RP01", "RP02", "RP03", "RP04", "RP05", "RP06", "RP07"]
    assert benchmark_cases() == benchmark_cases(SUITE_V01)


def test_v02_case_set_replaces_only_rp06_and_rp07():
    assert list(benchmark_cases_v02()) == [
        "RP01", "RP02", "RP03", "RP04", "RP05", "RP06A", "RP06B", "RP07A", "RP07B"
    ]


@pytest.mark.parametrize("case_id", ["RP01", "RP02", "RP03", "RP04"])
def test_rp01_to_rp04_are_unchanged_between_suites(case_id):
    assert benchmark_cases(SUITE_V01)[case_id] == benchmark_cases(SUITE_V02)[case_id]


def test_v02_deterministic_and_fake_baselines_pass_all_cases(v02_deterministic, v02_fake):
    assert (v02_deterministic.successful_cases, v02_deterministic.case_count) == (9, 9)
    assert (v02_fake.successful_cases, v02_fake.case_count) == (9, 9)


@pytest.mark.parametrize(
    "reason_code",
    [UnsupportedReasonCode.NO_ALLOWED_REPAIR, UnsupportedReasonCode.REQUIRES_REPLAN],
)
def test_rp05_accepts_each_safe_abstention(reason_code, tmp_path):
    case = benchmark_cases_v02()["RP05"]
    raw = RepairPlannerResponse(root=UnsupportedPlannerResponse(reason_code=reason_code)).model_dump_json()
    result = run_case(
        case,
        LLMRepairPlanner(ScriptedFakeBackend([raw]), LLMPlannerConfig(max_attempts=1)),
        tmp_path,
    )
    assert result.metrics.planner_case_success
    assert result.metrics.reason_code_correct
    assert result.response_status == "UNSUPPORTED"


def test_authorization_is_actions_plus_enumerated_targets(tmp_path):
    case = benchmark_cases_v02()["RP06A"]
    r01 = run_pipeline(case.spec, case.source_plan, tmp_path / "R01")
    context = RepairPlanningContextBuilder().build(
        case.spec, case.source_plan, r01.validation_report, case.context_rules
    )
    assert {item.value for item in context.allowed_action_types} == {"SET_PARAMETER"}
    assert authorized_set_parameter_targets(context) == {("OP01", "x"), ("OP01", "y")}
    assert set(context.repairable_operations[0].parameters) == {"x", "y", "z"}
    assert ("OP01", "z") not in authorized_set_parameter_targets(context)


def test_constraint_relationship_does_not_make_every_related_target_correct(tmp_path):
    case = benchmark_cases_v02()["RP06A"]
    data = {"status": "PLAN", "repair_plan": case.golden_plan.model_dump(mode="json")}
    action = data["repair_plan"]["actions"][0]
    action["target"]["parameter"] = "y"
    action["old_value"] = 60.0
    result = run_case(
        case,
        LLMRepairPlanner(ScriptedFakeBackend([json.dumps(data)]), LLMPlannerConfig(max_attempts=1)),
        tmp_path,
    )
    assert result.metrics.schema_valid
    assert not result.metrics.invented_action  # y is authorized, merely incorrect for EXTENT_X
    assert not result.metrics.target_correct
    assert not result.metrics.planner_case_success


def test_rp06a_semantically_disambiguates_extent_x(v02_deterministic):
    result = next(item for item in v02_deterministic.results if item.case_id == "RP06A")
    action = result.planner_response.root.repair_plan.actions[0]
    assert action.target.parameter == "x"
    assert result.metrics.planner_case_success


def test_rp06b_is_genuinely_ambiguous(v02_deterministic):
    case = benchmark_cases_v02()["RP06B"]
    assert case.spec.constraints[-1].measurement.type.value == "VOLUME"
    assert {rule.parameter_name for rule in case.context_rules} == {"x", "y"}
    result = next(item for item in v02_deterministic.results if item.case_id == "RP06B")
    assert result.response_status == "UNSUPPORTED"
    assert result.planner_response.root.reason_code == UnsupportedReasonCode.AMBIGUOUS_TARGET


def test_rp07b_differs_from_rp07a_only_by_identity_and_adversarial_hint():
    a, b = benchmark_cases_v02()["RP07A"], benchmark_cases_v02()["RP07B"]
    action_a, action_b = a.golden_plan.actions[0], b.golden_plan.actions[0]
    assert (action_a.target, action_a.old_value, action_a.new_value, action_a.constraint_ids) == (
        action_b.target, action_b.old_value, action_b.new_value, action_b.constraint_ids
    )
    assert a.non_authoritative_hint is None
    assert "change OP01.params.x" in b.non_authoritative_hint


def test_v02_cli_selection_is_explicit_and_v01_remains_default():
    parser = build_cli_parser()
    assert parser.parse_args([]).suite == SUITE_V01
    args = parser.parse_args(["--suite", SUITE_V02, "--case", "RP06A"])
    assert args.suite == SUITE_V02 and args.case == "RP06A"


def test_comparison_reports_cases_and_every_metric(v02_deterministic, tmp_path):
    v01 = run_benchmark("deterministic", artifact_root=tmp_path / "v01", suite=SUITE_V01)
    comparison = compare_benchmark_summaries([v01, v02_deterministic])
    assert [run.benchmark_version for run in comparison.runs] == ["m3-a.1", "M3-RP-v0.2"]
    assert len(comparison.cases) == 16
    assert set(comparison.runs[1].metrics) == set(METRIC_FIELDS)
    assert {item.lineage for item in comparison.cases if item.case_id.startswith("RP06")} == {"RP06"}
