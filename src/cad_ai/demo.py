from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cases import CASES, repair_case
from .pipeline import run_pipeline, run_revision_cycle
from .revision import DeterministicRepairController, RepairRule


def main() -> int:
    parser = argparse.ArgumentParser(description="CAD AI deterministic V0.1 demo")
    parser.add_argument("case", nargs="?", choices=sorted([*CASES, "repair"]), default="pass")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    if args.case == "repair":
        return _run_repair_demo(args.output)

    spec, plan = CASES[args.case]()
    result = run_pipeline(spec, plan, args.output / args.case)
    print(f"Case: {args.case}")
    print(f"CADResult: {result.cad_result.status.value}")
    if result.cad_result.error:
        print(json.dumps(result.cad_result.error.model_dump(mode="json"), indent=2))
        return 0
    assert result.validation_report is not None
    report = result.validation_report
    print(f"ValidationReport: {report.status.value} (complete={report.complete})")
    for check in report.checks:
        print(f"  {check.constraint_id}: {check.status.value} expected={check.expected} actual={check.actual}")
    for artifact_type, path in result.artifacts.items():
        print(f"{artifact_type}: {path.resolve()}")
    return 0


def _run_repair_demo(output: Path) -> int:
    spec, r01_plan = repair_case()
    controller = DeterministicRepairController([
        RepairRule(constraint_id="C_WIDTH", operation_id="OP01", parameter_name="x", related_feature_ids=["main_body"])
    ])
    cycle = run_revision_cycle(spec, r01_plan, controller, output / "repair")
    r01_report = cycle.r01.validation_report
    assert r01_report is not None
    failed = next(check for check in r01_report.checks if check.constraint_id == "C_WIDTH")
    print(f"Design: {spec.design_id}")
    print(f"Spec version: {spec.spec_version}")
    print(f"R01 Validation: {r01_report.status.value}")
    print(f"  C_WIDTH expected={failed.expected['target']} ± {failed.expected['tolerance']} actual={failed.actual}")
    assert cycle.repair_plan is not None and cycle.repair_plan_validation is not None
    print("RepairPlan:")
    for action in cycle.repair_plan.actions:
        print(f"  {action.action_id} {action.type.value} {action.target.semantic_path}")
        print(f"  {action.old_value} -> {action.new_value} because {','.join(action.constraint_ids)}")
    print(f"RepairPlan validation: {cycle.repair_plan_validation.status.value}")
    print(f"Repair execution: {cycle.repair.status.value}")
    for change in cycle.repair.changes:
        print(f"  {change.target}: {change.old_value} -> {change.new_value} ({change.constraint_id})")
    assert cycle.r02_plan is not None and cycle.r02 is not None and cycle.r02.validation_report is not None
    print(f"R02 parent: {cycle.r02_plan.parent_revision_id}")
    print(f"R02 Validation: {cycle.r02.validation_report.status.value}")
    print(f"Feature IDs preserved: {set(cycle.r01.cad_result.feature_registry.features) == set(cycle.r02.cad_result.feature_registry.features)}")
    print(f"Changed semantic fields: {sorted(cycle.changed_paths)}")
    print(f"Change locality: {'PASS' if cycle.change_locality else 'FAIL'}")
    print(f"Constraint preservation: {'PASS' if cycle.constraint_preservation else 'FAIL'}")
    for label, pipeline_result in (("R01", cycle.r01), ("R02", cycle.r02)):
        for artifact_type, path in pipeline_result.artifacts.items():
            print(f"{label} {artifact_type}: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
