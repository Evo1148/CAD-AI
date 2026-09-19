from pathlib import Path

from cad_ai.capability_v02 import ExtractedPlateFacts
from cad_ai.planning import BackendResponse, BackendRunMetadata, BackendStatus, ScriptedFakeBackend
from cad_ai.prompt_grounding import residual_facts_schema
from cad_ai.prototype_gate import (
    PROTOTYPE_GATE_VERSION,
    PrototypeCaseClass,
    evaluate_case_result,
    prototype_cases,
    run_prototype_case,
    run_prototype_gate,
    select_cases,
)
from cad_ai.v02 import CASE_PROMPTS, CapabilityPackResult, V02Status, run_capability_case


def _case(case_id):
    return next(item for item in prototype_cases() if item.case_id == case_id)


def test_catalog_has_twenty_new_prompts_and_all_required_classes():
    cases = prototype_cases()

    assert len(cases) == 20
    assert len({case.case_id for case in cases}) == 20
    assert len({case.prompt for case in cases}) == 20
    assert not ({case.prompt for case in cases} & set(CASE_PROMPTS.values()))
    assert sum(case.expected_class == PrototypeCaseClass.COMPLETE for case in cases) == 11
    assert sum(case.expected_class == PrototypeCaseClass.PARTIAL for case in cases) == 4
    assert sum(case.expected_class == PrototypeCaseClass.INSUFFICIENT for case in cases) == 2
    assert sum(case.expected_class == PrototypeCaseClass.UNSUPPORTED for case in cases) == 3
    tags = {tag for case in cases for tag in case.tags}
    assert {
        "irrelevant-number", "fillet-only", "chamfer-only", "linear-pattern",
        "shared-diameter", "supported-keyword-near-unsupported",
    } <= tags


def test_case_selection_supports_class_and_specific_case():
    partial = select_cases(classification="partial")
    assert [case.case_id for case in partial] == ["PUG-P01", "PUG-P02", "PUG-P03", "PUG-P04"]
    selected = select_cases(classification="complete", case_ids=["PUG-C02", "PUG-C09"])
    assert [case.case_id for case in selected] == ["PUG-C02", "PUG-C09"]


def test_partial_runner_uses_only_exact_residual_fields(tmp_path):
    case = _case("PUG-P01")
    backend = ScriptedFakeBackend([
        '{"contract_version":"0.2","values":{"slots[0].x":-10,"slots[0].y":0}}',
    ], model_id="fake-qwen")

    result = run_prototype_case(
        case, tmp_path, backend_mode="http-llm", http_backend=backend,
    )

    assert result.passed is True
    assert result.metrics.llm_invoked is True
    assert result.metrics.model_id == "fake-qwen"
    assert result.metrics.prompt_version == "residual-facts-1.0"
    request = backend.requests[0]
    assert request.context.unresolved_fields == ["slots[0].x", "slots[0].y"]
    assert request.response_schema == residual_facts_schema(request.context.unresolved_fields)
    assert set(request.response_schema["properties"]["values"]["properties"]) == {
        "slots[0].x", "slots[0].y",
    }
    assert request.context.grounded_evidence.base_dimensions == (80.0, 40.0, 6.0)
    assert request.context.grounded_evidence.slots[0].length == 30.0
    assert request.context.grounded_evidence.slots[0].through is True


def test_partial_runner_preserves_real_backend_telemetry(tmp_path):
    case = _case("PUG-P03")
    raw = (
        '{"contract_version":"0.2","values":{'
        '"circular_pockets[0].x":8,"circular_pockets[0].y":6}}'
    )
    backend = ScriptedFakeBackend([
        BackendResponse(
            status=BackendStatus.SUCCESS,
            raw_output=raw,
            metadata=BackendRunMetadata(
                backend_status="SUCCESS", runtime_id="llama.cpp test",
                http_status=200, prompt_tokens=321, completion_tokens=27,
                total_tokens=348, generation_ms=400.0, generation_tps=67.5,
            ),
        ),
    ], model_id="cad-ai-test-model")

    result = run_prototype_case(
        case, tmp_path, backend_mode="http-llm", http_backend=backend,
    )

    assert result.passed is True
    assert result.metrics.model_id == "cad-ai-test-model"
    assert result.metrics.prompt_tokens == 321
    assert result.metrics.completion_tokens == 27
    assert result.metrics.generation_tps == 67.5
    assert result.metrics.llm_latency_ms is not None
    assert result.metrics.schema_valid is True


def test_controlled_gate_passes_all_product_level_assertions(tmp_path):
    summary = run_prototype_gate(prototype_cases(), tmp_path)

    assert summary.benchmark_version == PROTOTYPE_GATE_VERSION
    assert summary.total == 20
    assert (summary.complete_count, summary.partial_count) == (11, 4)
    assert (summary.insufficient_count, summary.unsupported_count) == (2, 3)
    assert summary.passed == 20
    assert summary.failed == 0
    assert summary.deterministic_success == 16
    assert summary.hybrid_success == 4
    assert summary.false_validated == 0
    assert summary.intent_error == 0
    assert summary.cad_fail == 0
    assert summary.validation_fail == 0

    c11 = next(result for result in summary.results if result.case_id == "PUG-C11")
    assert c11.passed is True
    assert c11.pipeline_status == V02Status.VALIDATED
    assert c11.actual_class == PrototypeCaseClass.COMPLETE
    assert c11.metrics.llm_invoked is False
    assert c11.metrics.validator_status == "PASS"
    assert c11.metrics.final_revision == "R01"
    assert c11.metrics.step_generated is True
    assert c11.metrics.stl_generated is True
    assert c11.actual_canonical_facts.unsupported_features == []

    assert (tmp_path / "summary.json").is_file()
    for result in summary.results:
        assert Path(result.result_path).is_file()
        assert Path(result.pipeline_result_path).is_file()
        if result.passed and result.expected_class in {
            PrototypeCaseClass.COMPLETE, PrototypeCaseClass.PARTIAL,
        }:
            assert result.metrics.cad_status == "SUCCESS"
            assert result.metrics.validator_status == "PASS"
            assert result.metrics.final_revision == "R01"
            assert result.metrics.step_generated is True
            assert result.metrics.stl_generated is True


def test_false_validated_is_detected_from_prompt_to_facts_chain(tmp_path):
    case = _case("PUG-C01")
    pipeline_path = tmp_path / "pipeline.json"
    pipeline = run_capability_case(
        "CP2-01",
        ScriptedFakeBackend([]),
        tmp_path / "cad",
        user_prompt=case.prompt,
        result_json=pipeline_path,
    )
    assert pipeline.status == V02Status.VALIDATED
    wrong_facts = ExtractedPlateFacts(width=93.0, depth=54.0, height=7.0)
    wrong_case = case.model_copy(update={"expected_facts": wrong_facts})

    result = evaluate_case_result(
        wrong_case,
        CapabilityPackResult.model_validate_json(pipeline_path.read_text(encoding="utf-8")),
        total_latency_ms=1.0,
        pipeline_result_path=pipeline_path,
        result_path=tmp_path / "result.json",
    )

    assert result.pipeline_status == V02Status.VALIDATED
    assert result.passed is False
    assert result.false_validated is True
    assert {
        "CANONICAL_FACTS_MISMATCH", "INTENT_MISMATCH",
        "DESIGN_SPEC_MISMATCH", "CAD_PLAN_MISMATCH",
    } <= set(result.semantic_errors)
