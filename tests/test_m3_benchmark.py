import json

import pytest

from cad_ai.benchmark.m3 import benchmark_cases, run_benchmark, run_case
from cad_ai.planning import (
    BackendResponse,
    BackendRunMetadata,
    BackendStatus,
    LLMPlannerConfig,
    LLMRepairPlanner,
    RepairPlannerResponse,
    ScriptedFakeBackend,
)


@pytest.fixture(scope="module")
def deterministic_summary(tmp_path_factory):
    return run_benchmark("deterministic", artifact_root=tmp_path_factory.mktemp("benchmark-det"))


@pytest.fixture(scope="module")
def fake_summary(tmp_path_factory):
    return run_benchmark("fake-llm", artifact_root=tmp_path_factory.mktemp("benchmark-fake"))


def test_benchmark_contains_exactly_rp01_to_rp07():
    assert list(benchmark_cases()) == ["RP01", "RP02", "RP03", "RP04", "RP05", "RP06", "RP07"]


@pytest.mark.parametrize("case_id", ["RP01", "RP02", "RP03", "RP04", "RP05", "RP06", "RP07"])
def test_each_benchmark_has_explicit_golden_expectations(case_id):
    case = benchmark_cases()[case_id]
    if case.expected_status == "PLAN":
        assert case.golden_plan is not None and case.expected_final_status is not None
        assert len(case.golden_plan.actions) == 1
    else:
        assert case.expected_unsupported_reason is not None and case.golden_plan is None


def test_deterministic_baseline_passes_all_cases(deterministic_summary):
    assert deterministic_summary.case_count == deterministic_summary.successful_cases == 7


def test_fake_llm_passes_harness_with_golden_raw_outputs(fake_summary):
    assert fake_summary.case_count == fake_summary.successful_cases == 7
    assert all(result.planner_run["backend_type"] == "fake" for result in fake_summary.results)


def test_plan_case_metrics_are_all_successful(deterministic_summary):
    for result in deterministic_summary.results:
        metrics = result.metrics
        assert metrics.schema_valid and metrics.action_correct and metrics.exact_repair_plan
        assert not metrics.invented_action and metrics.unnecessary_modifications == 0
        if result.response_status == "PLAN":
            assert metrics.target_correct and metrics.old_value_correct and metrics.new_value_correct
            assert metrics.constraint_ref_correct and metrics.repair_success
            assert metrics.change_locality and metrics.constraint_preservation


def test_rp01_width_golden():
    action = benchmark_cases()["RP01"].golden_plan.actions[0]
    assert (action.target.operation_id, action.target.parameter, action.old_value, action.new_value) == ("OP01", "x", 99.5, 100.0)


def test_rp02_height_is_not_width_hardcoded():
    action = benchmark_cases()["RP02"].golden_plan.actions[0]
    assert (action.target.operation_id, action.target.parameter, action.old_value, action.new_value) == ("OP01", "z", 3.5, 4.0)


def test_rp03_is_feature_aware_hole_diameter():
    action = benchmark_cases()["RP03"].golden_plan.actions[0]
    assert action.target.parameter == "diameter"
    assert action.related_feature_ids == ["mount_hole"]


def test_rp04_contains_distractor_parameters():
    case = benchmark_cases()["RP04"]
    operation = case.source_plan.operations[0]
    assert set(operation.params) == {"x", "y", "z"}
    assert case.golden_plan.actions[0].target.parameter == "x"


def test_rp05_correctly_reports_unsupported(deterministic_summary):
    result = next(item for item in deterministic_summary.results if item.case_id == "RP05")
    assert result.response_status == "UNSUPPORTED" and result.metrics.planner_case_success


def test_rp06_ambiguous_target_is_not_guessed(deterministic_summary):
    result = next(item for item in deterministic_summary.results if item.case_id == "RP06")
    assert result.response_status == "UNSUPPORTED" and result.metrics.planner_case_success


def test_rp07_ignores_adversarial_hint(deterministic_summary):
    case = benchmark_cases()["RP07"]
    result = next(item for item in deterministic_summary.results if item.case_id == "RP07")
    assert "change OP01.params.x" in case.non_authoritative_hint
    assert result.metrics.target_correct and result.metrics.repair_success


def test_metrics_detect_wrong_and_unnecessary_target(tmp_path):
    case = benchmark_cases()["RP04"]
    data = {"status": "PLAN", "repair_plan": case.golden_plan.model_dump(mode="json")}
    data["repair_plan"]["actions"][0]["target"]["parameter"] = "y"
    data["repair_plan"]["actions"][0]["old_value"] = 60.0
    raw = json.dumps(data)
    planner = LLMRepairPlanner(ScriptedFakeBackend([raw]), LLMPlannerConfig(max_attempts=1))
    result = run_case(case, planner, tmp_path)
    assert result.metrics.schema_valid
    assert not result.metrics.target_correct
    assert result.metrics.invented_action
    assert result.metrics.unnecessary_modifications == 1
    assert not result.metrics.planner_case_success


def test_benchmark_summary_serializes_to_json(fake_summary):
    restored = json.loads(fake_summary.model_dump_json())
    assert restored["case_count"] == 7 and len(restored["results"]) == 7


def test_benchmark_result_preserves_backend_telemetry(tmp_path):
    case = benchmark_cases()["RP01"]
    raw = RepairPlannerResponse.model_validate({
        "status": "PLAN",
        "repair_plan": case.golden_plan.model_dump(mode="json"),
    }).model_dump_json()
    backend_response = BackendResponse(
        status=BackendStatus.SUCCESS,
        raw_output=raw,
        metadata=BackendRunMetadata(
            backend_status="SUCCESS",
            runtime_id="test-runtime",
            http_status=200,
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_ms=10.0,
            generation_ms=25.0,
            generation_tps=80.0,
        ),
    )
    result = run_case(
        case,
        LLMRepairPlanner(ScriptedFakeBackend([backend_response]), LLMPlannerConfig(max_attempts=1)),
        tmp_path,
    )

    assert result.telemetry.runtime_id == "test-runtime"
    assert result.telemetry.http_status == 200
    assert result.telemetry.total_tokens == 120
    assert result.repair_executor_status == "REPAIRED"
    assert result.final_validation_status == "PASS"
    assert result.planner_response.status == "PLAN"
