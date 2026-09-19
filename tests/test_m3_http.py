from copy import deepcopy
import urllib.error

import pytest

from cad_ai.benchmark.m3 import (
    DEFAULT_M3B_MODEL_ID,
    build_cli_parser,
    m3b_http_config,
    m3b_llm_config,
)
from cad_ai.cases import repair_case
from cad_ai.pipeline import run_pipeline
from cad_ai.planning import (
    BackendFailureCode,
    BackendStatus,
    DeterministicPlannerAdapter,
    HTTPInferenceBackend,
    HTTPInferenceConfig,
    InferenceRequest,
    LLMPlannerConfig,
    LLMRepairPlanner,
    PlannerErrorCode,
    RepairPlannerResponse,
    RepairPlanningContextBuilder,
    STRUCTURED_OUTPUT_CANARY_SCHEMA,
    StructuredOutputCanaryStatus,
    llama_cpp_generation_schema,
    run_structured_output_canary,
)
from cad_ai.revision import RepairRule


class RecordingTransport:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def post_json(self, url, payload, timeout_seconds):
        self.calls.append((url, payload, timeout_seconds))
        if self.error:
            raise self.error
        return self.result


@pytest.fixture()
def request_data(tmp_path):
    spec, plan = repair_case()
    source = run_pipeline(spec, plan, tmp_path)
    rules = [RepairRule(constraint_id="C_WIDTH", operation_id="OP01", parameter_name="x")]
    context = RepairPlanningContextBuilder().build(spec, plan, source.validation_report, rules)
    response = DeterministicPlannerAdapter().plan(context).response
    request = InferenceRequest(
        system_prompt="system",
        context=context,
        response_schema=RepairPlannerResponse.model_json_schema(),
        temperature=0.1,
        max_tokens=256,
        seed=7,
    )
    return request, response


def test_http_endpoint_construction():
    backend = HTTPInferenceBackend(HTTPInferenceConfig(base_url="http://127.0.0.1:8080/", endpoint="v1/chat/completions"))
    assert backend.endpoint_url == "http://127.0.0.1:8080/v1/chat/completions"


def test_http_request_serialization(request_data):
    request, response = request_data
    transport = RecordingTransport({"choices": [{"message": {"content": response.model_dump_json()}}]})
    config = HTTPInferenceConfig(
        base_url="http://localhost:8080",
        model_id="neutral-model",
        timeout_seconds=12,
        chat_template_kwargs={"enable_thinking": False},
    )
    backend = HTTPInferenceBackend(config, transport)
    result = backend.generate(request)
    url, payload, timeout = transport.calls[0]
    assert result.status == BackendStatus.SUCCESS
    assert url.endswith("/v1/chat/completions") and timeout == 12
    assert payload["model"] == "neutral-model"
    assert payload["temperature"] == 0.1 and payload["max_tokens"] == 256 and payload["seed"] == 7
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema"] == RepairPlannerResponse.model_json_schema()
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "failed_constraints" in payload["messages"][1]["content"]
    assert result.metadata.request_payload == payload
    assert result.metadata.response_schema == RepairPlannerResponse.model_json_schema()
    assert result.metadata.raw_message_content == response.model_dump_json()


def test_http_valid_structured_response(request_data):
    request, response = request_data
    transport = RecordingTransport({"choices": [{"message": {"content": response.model_dump_json()}}]})
    result = HTTPInferenceBackend(HTTPInferenceConfig(base_url="http://localhost"), transport).generate(request)
    assert RepairPlannerResponse.model_validate_json(result.raw_output) == response


def test_llama_cpp_request_uses_direct_response_schema(request_data):
    request, response = request_data
    transport = RecordingTransport({"choices": [{"message": {"content": response.model_dump_json()}}]})
    config = HTTPInferenceConfig(
        base_url="http://127.0.0.1:8080",
        model_id="cad-ai-m3-qwen35-9b-q6",
        response_format_dialect="llama.cpp",
        chat_template_kwargs={"enable_thinking": False},
    )

    result = HTTPInferenceBackend(config, transport).generate(request)
    payload = transport.calls[0][1]

    expected_schema = llama_cpp_generation_schema(RepairPlannerResponse.model_json_schema())
    assert payload["json_schema"] == expected_schema
    assert not ({"response_format", "grammar"} & payload.keys())
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert result.metadata.request_payload == payload
    assert result.metadata.response_schema == expected_schema


def test_llama_cpp_generation_schema_requires_discriminator_without_mutating_contract():
    contract_schema = RepairPlannerResponse.model_json_schema()
    original = deepcopy(contract_schema)

    generation_schema = llama_cpp_generation_schema(contract_schema)

    assert contract_schema == original
    for branch_name in (
        "PlanPlannerResponse",
        "UnsupportedPlannerResponse",
        "ErrorPlannerResponse",
    ):
        required = generation_schema["$defs"][branch_name]["required"]
        assert "status" in required
        assert "contract_version" in required


def test_structured_output_canary_passes_only_for_schema_constrained_output():
    transport = RecordingTransport({
        "choices": [{"message": {"content": '{"canary":"CAD_AI_SCHEMA_ENFORCED"}'}}],
    })
    backend = HTTPInferenceBackend(HTTPInferenceConfig(
        base_url="http://localhost",
        model_id="cad-ai-m3-ministral3-14b-q5km",
        response_format_dialect="llama.cpp",
    ), transport)

    result = run_structured_output_canary(backend)

    assert result.status == StructuredOutputCanaryStatus.PASS
    assert result.schema_enforced
    payload = transport.calls[0][1]
    assert payload["model"] == "cad-ai-m3-ministral3-14b-q5km"
    assert payload["json_schema"] == STRUCTURED_OUTPUT_CANARY_SCHEMA
    assert "response_format" not in payload


def test_structured_output_canary_detects_unconstrained_output():
    raw = '{"schema_not_enforced":true}'
    transport = RecordingTransport({"choices": [{"message": {"content": raw}}]})
    backend = HTTPInferenceBackend(HTTPInferenceConfig(
        base_url="http://localhost",
        response_format_dialect="llama.cpp",
    ), transport)

    result = run_structured_output_canary(backend)

    assert result.status == StructuredOutputCanaryStatus.FAIL
    assert not result.schema_enforced
    assert result.raw_output == raw


def test_http_timeout_handling(request_data):
    transport = RecordingTransport(error=TimeoutError("slow"))
    result = HTTPInferenceBackend(HTTPInferenceConfig(base_url="http://localhost"), transport).generate(request_data[0])
    assert result.error_code == BackendFailureCode.MODEL_TIMEOUT


def test_http_error_handling(request_data):
    error = urllib.error.HTTPError("http://localhost", 500, "boom", {}, None)
    transport = RecordingTransport(error=error)
    result = HTTPInferenceBackend(HTTPInferenceConfig(base_url="http://localhost"), transport).generate(request_data[0])
    assert result.error_code == BackendFailureCode.MODEL_ERROR


def test_http_server_unavailable(request_data):
    transport = RecordingTransport(error=urllib.error.URLError("refused"))
    result = HTTPInferenceBackend(HTTPInferenceConfig(base_url="http://localhost"), transport).generate(request_data[0])
    assert result.error_code == BackendFailureCode.MODEL_UNAVAILABLE


def test_http_malformed_envelope(request_data):
    transport = RecordingTransport({"unexpected": []})
    result = HTTPInferenceBackend(HTTPInferenceConfig(base_url="http://localhost"), transport).generate(request_data[0])
    assert result.error_code == BackendFailureCode.MODEL_ERROR


def test_http_malformed_model_output_is_left_for_planner_parser(request_data):
    transport = RecordingTransport({"choices": [{"message": {"content": "not-json"}}]})
    result = HTTPInferenceBackend(HTTPInferenceConfig(base_url="http://localhost"), transport).generate(request_data[0])
    assert result.status == BackendStatus.SUCCESS and result.raw_output == "not-json"
    assert result.metadata.raw_message_content == "not-json"


def test_http_preserves_schema_invalid_raw_output_for_diagnosis(request_data):
    raw = '{"repair_plan":{"design_id":"D001"}}'
    transport = RecordingTransport({"choices": [{"message": {"content": raw}}]})
    result = HTTPInferenceBackend(HTTPInferenceConfig(base_url="http://localhost"), transport).generate(request_data[0])

    assert result.status == BackendStatus.SUCCESS
    assert result.metadata.raw_message_content == raw
    assert result.metadata.request_payload["response_format"]["json_schema"]["schema"] == request_data[0].response_schema


def test_schema_failure_invocation_retains_exact_http_debug_data(request_data):
    request, _ = request_data
    raw = '{"repair_plan":{"design_id":"D001"}}'
    transport = RecordingTransport({"choices": [{"message": {"content": raw}}]})
    backend = HTTPInferenceBackend(HTTPInferenceConfig(base_url="http://localhost"), transport)

    invocation = LLMRepairPlanner(backend, LLMPlannerConfig(max_attempts=1)).plan(request.context)

    assert invocation.response.root.error.code == PlannerErrorCode.SCHEMA_VALIDATION_ERROR
    assert invocation.planner_run.backend_metadata.raw_message_content == raw
    assert invocation.planner_run.backend_metadata.request_payload == transport.calls[0][1]
    assert invocation.planner_run.backend_metadata.response_schema == RepairPlannerResponse.model_json_schema()


def test_http_configuration_is_model_neutral():
    config = HTTPInferenceConfig(base_url="http://localhost", model_id=None, structured_output=False)
    assert config.model_id is None and not config.structured_output


def test_http_preserves_usage_and_llama_timings(request_data):
    request, response = request_data
    transport = RecordingTransport({
        "choices": [{"message": {"content": response.model_dump_json()}}],
        "usage": {"prompt_tokens": 321, "completion_tokens": 45, "total_tokens": 366},
        "timings": {"prompt_ms": 120.5, "predicted_ms": 590.0, "predicted_per_second": 76.3},
    })

    result = HTTPInferenceBackend(HTTPInferenceConfig(base_url="http://localhost"), transport).generate(request)

    assert result.metadata.http_status == 200
    assert result.metadata.backend_status == "SUCCESS"
    assert result.metadata.prompt_tokens == 321
    assert result.metadata.completion_tokens == 45
    assert result.metadata.total_tokens == 366
    assert result.metadata.prompt_ms == 120.5
    assert result.metadata.generation_ms == 590.0
    assert result.metadata.generation_tps == 76.3


def test_m3b_http_and_planner_defaults_are_exact():
    http = m3b_http_config()
    llm = m3b_llm_config()
    assert http.base_url == "http://127.0.0.1:8080"
    assert http.endpoint == "/v1/chat/completions"
    assert http.model_id == "cad-ai-m3-qwen35-9b-q6"
    assert http.runtime_id == "llama.cpp b10985"
    assert http.chat_template_kwargs == {"enable_thinking": False}
    assert http.structured_output
    assert http.response_format_dialect == "llama.cpp"
    assert llm.max_attempts == 1
    assert llm.temperature == 0
    assert llm.seed == 12345
    assert llm.max_tokens == 1024


def test_cli_model_id_default_and_custom_value():
    parser = build_cli_parser()

    defaults = parser.parse_args(["--planner", "http-llm"])
    custom = parser.parse_args([
        "--planner", "http-llm",
        "--model-id", "cad-ai-m3-ministral3-14b-q5km",
    ])
    legacy = parser.parse_args(["--planner", "http-llm", "--model", "legacy-model-id"])

    assert defaults.model_id == DEFAULT_M3B_MODEL_ID
    assert custom.model_id == "cad-ai-m3-ministral3-14b-q5km"
    assert legacy.model_id == "legacy-model-id"


def test_custom_model_id_is_the_only_request_change_and_reaches_planner_run(request_data):
    request, response = request_data

    def invoke(model_id):
        transport = RecordingTransport({
            "choices": [{"message": {"content": response.model_dump_json()}}],
        })
        backend = HTTPInferenceBackend(m3b_http_config(model_id=model_id), transport)
        invocation = LLMRepairPlanner(backend, m3b_llm_config()).plan(request.context)
        return invocation, transport.calls[0][1]

    default_invocation, default_payload = invoke(DEFAULT_M3B_MODEL_ID)
    custom_id = "cad-ai-m3-ministral3-14b-q5km"
    custom_invocation, custom_payload = invoke(custom_id)

    assert custom_payload["model"] == custom_id
    assert custom_invocation.planner_run.model_id == custom_id
    assert default_invocation.planner_run.model_id == DEFAULT_M3B_MODEL_ID
    assert custom_payload["temperature"] == 0
    assert custom_payload["seed"] == 12345
    assert custom_payload["max_tokens"] == 1024
    assert custom_payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert custom_payload["json_schema"] == default_payload["json_schema"]
    assert {key: value for key, value in custom_payload.items() if key != "model"} == {
        key: value for key, value in default_payload.items() if key != "model"
    }
