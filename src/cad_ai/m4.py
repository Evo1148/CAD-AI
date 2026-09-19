from __future__ import annotations

import argparse
from pathlib import Path

from .cad import CADStatus
from .generation import (
    DeterministicIntentToSpecCompiler,
    DeterministicSpecToPlanCompiler,
    GenerationError,
    GenerationErrorCode,
    GenerationStatus,
    LLMIntentGenerator,
    M4GenerationResult,
    M4Status,
    SpecToPlanContext,
    default_prompt_to_spec_context,
    final_result_error,
)
from .pipeline import run_pipeline
from .planning import HTTPInferenceBackend, HTTPInferenceConfig, InferenceBackend
from .validation import ReportStatus

M4_001_PROMPT = "Create a 60 x 40 x 4 mm rectangular plate with one centered 6 mm through hole."
M4_001_DESIGN_ID = "M4-001"
M4_001_SPEC_VERSION = "1.0"
DEFAULT_MODEL_ID = "cad-ai-m3-qwen35-9b-q6"


def run_m4_001(
    backend: InferenceBackend,
    output: Path,
    *,
    user_prompt: str = M4_001_PROMPT,
    result_json: Path | None = None,
) -> M4GenerationResult:
    state: dict = {
        "case_id": "M4-001",
        "stage": "PROMPT_TO_INTENT",
        "user_prompt": user_prompt,
        "design_id": M4_001_DESIGN_ID,
        "spec_version": M4_001_SPEC_VERSION,
    }

    def finish(status: M4Status, error: GenerationError | None = None) -> M4GenerationResult:
        result = M4GenerationResult(status=status, error=error, **state)
        destination = result_json or output / "result.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return result

    spec_context = default_prompt_to_spec_context(
        user_prompt,
        M4_001_DESIGN_ID,
        M4_001_SPEC_VERSION,
    )
    intent_generation = LLMIntentGenerator(backend, max_tokens=1024, seed=12345).generate(spec_context)
    state["intent_generation"] = intent_generation
    if intent_generation.status != GenerationStatus.SUCCESS or intent_generation.intent is None:
        return finish(_failure_status(intent_generation.error), intent_generation.error)

    state["stage"] = "DETERMINISTIC_INTENT_TO_SPEC"
    spec_generation = DeterministicIntentToSpecCompiler().compile(
        intent_generation.intent,
        design_id=M4_001_DESIGN_ID,
        spec_version=M4_001_SPEC_VERSION,
    )
    state["spec_generation"] = spec_generation
    if spec_generation.status != GenerationStatus.SUCCESS or spec_generation.design_spec is None:
        return finish(_failure_status(spec_generation.error), spec_generation.error)

    spec = spec_generation.design_spec
    state["stage"] = "DETERMINISTIC_SPEC_TO_PLAN"
    plan_context = SpecToPlanContext(
        design_id=M4_001_DESIGN_ID,
        spec_version=M4_001_SPEC_VERSION,
        revision_id="R01",
        design_spec=spec,
    )
    plan_generation = DeterministicSpecToPlanCompiler().compile(plan_context)
    state["plan_generation"] = plan_generation
    if plan_generation.status != GenerationStatus.SUCCESS or plan_generation.cad_plan is None:
        return finish(_failure_status(plan_generation.error), plan_generation.error)

    state["stage"] = "CAD_PIPELINE"
    try:
        pipeline = run_pipeline(spec, plan_generation.cad_plan, output / "R01")
    except Exception as exc:
        return finish(
            M4Status.FAIL,
            final_result_error(GenerationErrorCode.CAD_EXECUTION_ERROR, str(exc)),
        )

    state["cad_status"] = pipeline.cad_result.status
    state["artifacts"] = {name: str(path) for name, path in pipeline.artifacts.items()}
    if pipeline.cad_result.feature_registry is not None:
        state["feature_ids"] = sorted(pipeline.cad_result.feature_registry.features)
    if pipeline.cad_result.status != CADStatus.SUCCESS or pipeline.validation_report is None:
        cad_error = pipeline.cad_result.error
        return finish(
            M4Status.FAIL,
            final_result_error(
                GenerationErrorCode.CAD_EXECUTION_ERROR,
                cad_error.message if cad_error else "CAD Engine did not produce a valid result",
                cad_error_code=cad_error.code if cad_error else None,
            ),
        )

    state["stage"] = "VALIDATION"
    state["validation_report"] = pipeline.validation_report
    if pipeline.validation_report.status != ReportStatus.PASS:
        return finish(
            M4Status.FAIL,
            final_result_error(
                GenerationErrorCode.VALIDATION_NOT_PASS,
                "generated geometry does not satisfy the generated DesignSpec",
                validation_status=pipeline.validation_report.status.value,
            ),
        )

    state["stage"] = "COMPLETE"
    return finish(M4Status.VALIDATED)


def _failure_status(error: GenerationError | None) -> M4Status:
    if error and error.code in {
        GenerationErrorCode.MODEL_TIMEOUT,
        GenerationErrorCode.MODEL_UNAVAILABLE,
    }:
        return M4Status.BLOCKED
    return M4Status.FAIL


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CAD AI M4 V0.1 initial-generation vertical slice")
    parser.add_argument("--prompt", default=M4_001_PROMPT)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", type=Path, default=Path("artifacts/m4/M4-001"))
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_cli_parser().parse_args()
    backend = HTTPInferenceBackend(HTTPInferenceConfig(
        base_url=args.base_url,
        endpoint="/v1/chat/completions",
        model_id=args.model_id,
        runtime_id="llama.cpp",
        timeout_seconds=args.timeout,
        temperature=0.0,
        max_tokens=1024,
        seed=12345,
        structured_output=True,
        response_format_dialect="llama.cpp",
        chat_template_kwargs={"enable_thinking": False},
    ))
    result = run_m4_001(
        backend,
        args.output,
        user_prompt=args.prompt,
        result_json=args.result_json,
    )
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(f"M4-001 | status={result.status.value} | stage={result.stage}")
        if result.validation_report:
            print(f"Validation: {result.validation_report.status.value}")
        for name, path in result.artifacts.items():
            print(f"{name}: {path}")
        if result.error:
            print(f"Error: {result.error.code.value} - {result.error.message}")
    return 0 if result.status == M4Status.VALIDATED else 1


if __name__ == "__main__":
    raise SystemExit(main())
