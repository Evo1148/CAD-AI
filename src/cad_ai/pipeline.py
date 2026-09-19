from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cad import CADEngine, CADResult, CADStatus
from .contracts import CADPlan, DesignSpec
from .validation import ValidationReport, Validator
from .revision import (
    DeterministicRepairController,
    RepairPlan,
    RepairPlanningResult,
    RepairPlanningStatus,
    RepairPlanValidation,
    RepairPlanValidationStatus,
    RepairResult,
    RepairStatus,
    changed_semantic_plan_paths,
    preserved_passing_constraints,
)


@dataclass(frozen=True)
class PipelineResult:
    cad_result: CADResult
    validation_report: ValidationReport | None
    artifacts: dict[str, Path]


def run_pipeline(spec: DesignSpec, plan: CADPlan, artifact_directory: Path) -> PipelineResult:
    if spec.design_id != plan.design_id or spec.spec_version != plan.spec_version:
        raise ValueError("DesignSpec and CADPlan identifiers are incoherent")
    engine = CADEngine()
    cad_result = engine.execute(plan)
    if cad_result.status == CADStatus.ERROR:
        return PipelineResult(cad_result=cad_result, validation_report=None, artifacts={})
    report = Validator().validate(spec, cad_result)
    artifacts = engine.export(cad_result, artifact_directory)
    return PipelineResult(cad_result=cad_result, validation_report=report, artifacts=artifacts)


@dataclass(frozen=True)
class RevisionCycleResult:
    r01_plan: CADPlan
    r01: PipelineResult
    planning: RepairPlanningResult
    repair_plan: RepairPlan | None
    repair_plan_validation: RepairPlanValidation | None
    repair: RepairResult
    r02_plan: CADPlan | None
    r02: PipelineResult | None
    changed_paths: set[str]
    change_locality: bool
    preserved_constraint_ids: list[str]
    constraint_preservation: bool


def run_revision_cycle(
    spec: DesignSpec,
    r01_plan: CADPlan,
    controller: DeterministicRepairController,
    artifact_directory: Path,
) -> RevisionCycleResult:
    r01 = run_pipeline(spec, r01_plan, artifact_directory / r01_plan.revision_id)
    if r01.validation_report is None:
        raise ValueError("revision repair requires a ValidationReport")
    planning = controller.planner.plan(spec, r01_plan, r01.validation_report)
    repair_plan = planning.repair_plan
    plan_validation = None
    if planning.status == RepairPlanningStatus.PLANNED and repair_plan is not None:
        plan_validation = controller.validator.validate(repair_plan, spec, r01_plan, r01.validation_report)
        if plan_validation.status == RepairPlanValidationStatus.VALID:
            repair = controller.executor.execute(repair_plan, plan_validation, r01_plan)
        else:
            repair = RepairResult(
                design_id=spec.design_id, spec_version=spec.spec_version, source_revision_id=r01_plan.revision_id,
                status=RepairStatus.UNSUPPORTED, result_code="REPAIR_PLAN_INVALID",
            )
    else:
        status_map = {
            RepairPlanningStatus.NO_CHANGE: RepairStatus.NO_CHANGE,
            RepairPlanningStatus.UNSUPPORTED: RepairStatus.UNSUPPORTED,
            RepairPlanningStatus.ERROR: RepairStatus.ERROR,
        }
        repair = RepairResult(
            design_id=spec.design_id, spec_version=spec.spec_version, source_revision_id=r01_plan.revision_id,
            status=status_map[planning.status], result_code=planning.result_code,
        )
    if repair.status != RepairStatus.REPAIRED or repair.revised_plan is None:
        return RevisionCycleResult(
            r01_plan, r01, planning, repair_plan, plan_validation, repair,
            None, None, set(), False, [], False,
        )

    r02_plan = repair.revised_plan
    r02 = run_pipeline(spec, r02_plan, artifact_directory / r02_plan.revision_id)
    if r02.validation_report is None:
        raise ValueError("revised CAD execution did not produce a ValidationReport")
    changed_paths = changed_semantic_plan_paths(r01_plan, r02_plan)
    expected_paths = {change.target for change in repair.changes}
    preservation, preserved_ids = preserved_passing_constraints(r01.validation_report, r02.validation_report)
    return RevisionCycleResult(
        r01_plan=r01_plan,
        r01=r01,
        planning=planning,
        repair_plan=repair_plan,
        repair_plan_validation=plan_validation,
        repair=repair,
        r02_plan=r02_plan,
        r02=r02,
        changed_paths=changed_paths,
        change_locality=changed_paths == expected_paths,
        preserved_constraint_ids=preserved_ids,
        constraint_preservation=preservation,
    )
