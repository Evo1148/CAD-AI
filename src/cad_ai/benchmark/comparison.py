from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import Field

from ..contracts import StrictModel
from .m3 import BenchmarkSummary


METRIC_FIELDS = (
    "schema_valid",
    "action_correct",
    "target_correct",
    "old_value_correct",
    "new_value_correct",
    "constraint_ref_correct",
    "invented_action",
    "unnecessary_modifications",
    "repair_success",
    "change_locality",
    "constraint_preservation",
    "exact_repair_plan",
    "reason_code_correct",
    "planner_case_success",
)


class MetricScore(StrictModel):
    applicable: int = Field(ge=0)
    successful: int = Field(ge=0)
    failed: int = Field(ge=0)
    not_applicable: int = Field(ge=0)
    success_rate: float | None = Field(default=None, ge=0, le=1)


class ComparedRun(StrictModel):
    run_id: str
    benchmark_version: str
    planner: str
    model_id: str | None = None
    successful_cases: int
    case_count: int
    metrics: dict[str, MetricScore]


class CaseComparison(StrictModel):
    run_id: str
    benchmark_version: str
    model_id: str | None = None
    lineage: str
    case_id: str
    response_status: str
    planner_case_success: bool
    metrics: dict[str, Any]


class BenchmarkComparison(StrictModel):
    comparison_version: str = "M3-RP-comparison-v0.1"
    runs: list[ComparedRun]
    cases: list[CaseComparison]


def compare_benchmark_summaries(summaries: list[BenchmarkSummary]) -> BenchmarkComparison:
    """Compare any v0.1/v0.2 model runs without rewriting historical results."""

    compared_runs: list[ComparedRun] = []
    case_rows: list[CaseComparison] = []
    used_ids: dict[str, int] = {}
    for summary in summaries:
        model_id = _model_id(summary)
        base_id = f"{summary.benchmark_version}:{model_id or summary.planner}"
        used_ids[base_id] = used_ids.get(base_id, 0) + 1
        run_id = base_id if used_ids[base_id] == 1 else f"{base_id}#{used_ids[base_id]}"
        compared_runs.append(ComparedRun(
            run_id=run_id,
            benchmark_version=summary.benchmark_version,
            planner=summary.planner,
            model_id=model_id,
            successful_cases=summary.successful_cases,
            case_count=summary.case_count,
            metrics={name: _score(summary, name) for name in METRIC_FIELDS},
        ))
        for result in summary.results:
            case_rows.append(CaseComparison(
                run_id=run_id,
                benchmark_version=summary.benchmark_version,
                model_id=model_id,
                lineage=_case_lineage(result.case_id),
                case_id=result.case_id,
                response_status=result.response_status,
                planner_case_success=result.metrics.planner_case_success,
                metrics=result.metrics.model_dump(mode="json"),
            ))
    return BenchmarkComparison(runs=compared_runs, cases=case_rows)


def _model_id(summary: BenchmarkSummary) -> str | None:
    for result in summary.results:
        model_id = result.planner_run.get("model_id")
        if model_id:
            return str(model_id)
    return None


def _score(summary: BenchmarkSummary, metric_name: str) -> MetricScore:
    values = [getattr(result.metrics, metric_name) for result in summary.results]
    applicable = [value for value in values if value is not None]
    successful = sum(_is_success(metric_name, value) for value in applicable)
    failed = len(applicable) - successful
    return MetricScore(
        applicable=len(applicable),
        successful=successful,
        failed=failed,
        not_applicable=len(values) - len(applicable),
        success_rate=(successful / len(applicable)) if applicable else None,
    )


def _is_success(metric_name: str, value: Any) -> bool:
    if metric_name == "invented_action":
        return value is False
    if metric_name == "unnecessary_modifications":
        return value == 0
    return value is True


def _case_lineage(case_id: str) -> str:
    if case_id in {"RP06", "RP06A", "RP06B"}:
        return "RP06"
    if case_id in {"RP07", "RP07A", "RP07B"}:
        return "RP07"
    return case_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare M3-RP v0.1/v0.2 benchmark JSON results")
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    summaries = [BenchmarkSummary.model_validate_json(path.read_text(encoding="utf-8")) for path in args.results]
    comparison = compare_benchmark_summaries(summaries)
    if args.json:
        print(comparison.model_dump_json(indent=2))
    else:
        print("run                                      cases    success")
        for run in comparison.runs:
            print(f"{run.run_id:<40} {run.case_count:>5}    {run.successful_cases}/{run.case_count}")
        print("\nmetric aggregate")
        for run in comparison.runs:
            print(run.run_id)
            for name, score in run.metrics.items():
                value = "n/a" if score.success_rate is None else f"{score.successful}/{score.applicable}"
                print(f"  {name:<28} {value}")
        print("\ncase-by-case")
        for item in comparison.cases:
            status = "PASS" if item.planner_case_success else "FAIL"
            print(f"{item.run_id:<40} {item.case_id:<6} {item.response_status:<12} {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
