from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .contracts import (
    CADPlan,
    Comparison,
    CONTRACT_VERSION,
    DesignSpec,
    MeasurementType,
    Priority,
    StrictModel,
)
from .validation import CheckStatus, ReportStatus, ValidationReport


class RepairStatus(str, Enum):
    REPAIRED = "REPAIRED"
    UNSUPPORTED = "UNSUPPORTED"
    NO_CHANGE = "NO_CHANGE"
    ERROR = "ERROR"


class RepairRule(StrictModel):
    """Trusted M2 authorization rule plus a repair-relevance relationship.

    The projected M3 ``constraint_ids`` relationship is contextual evidence,
    not target equivalence.  Final permission remains here, in the trusted
    RepairPlanValidator.
    """

    constraint_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    parameter_name: str = Field(min_length=1)
    related_feature_ids: list[str] = Field(default_factory=list)

    @property
    def target_path(self) -> str:
        return f"{self.operation_id}.params.{self.parameter_name}"


class RepairActionType(str, Enum):
    SET_PARAMETER = "SET_PARAMETER"


class RepairTarget(StrictModel):
    operation_id: str = Field(min_length=1)
    parameter: str = Field(min_length=1)

    @property
    def semantic_path(self) -> str:
        return f"{self.operation_id}.params.{self.parameter}"


class RepairAction(StrictModel):
    action_id: str = Field(min_length=1)
    type: RepairActionType
    target: RepairTarget
    old_value: float
    new_value: float
    constraint_ids: list[str] = Field(min_length=1)
    related_feature_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)

    @field_validator("old_value", "new_value", mode="before")
    @classmethod
    def numeric_values_only(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("repair values must be numeric JSON values")
        return value

    @model_validator(mode="after")
    def unique_references(self) -> "RepairAction":
        if len(self.constraint_ids) != len(set(self.constraint_ids)):
            raise ValueError("constraint_ids must be unique within an action")
        if len(self.related_feature_ids) != len(set(self.related_feature_ids)):
            raise ValueError("related_feature_ids must be unique within an action")
        return self


class RepairPlan(StrictModel):
    contract_version: Literal["0.1"] = CONTRACT_VERSION
    design_id: str = Field(min_length=1)
    spec_version: str = Field(min_length=1)
    source_revision_id: str
    target_revision_id: str
    actions: list[RepairAction] = Field(min_length=1)
    rationale: str | None = None

    @model_validator(mode="after")
    def validate_plan_shape(self) -> "RepairPlan":
        if self.source_revision_id == self.target_revision_id:
            raise ValueError("source_revision_id and target_revision_id must differ")
        action_ids = [action.action_id for action in self.actions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("action_id values must be unique")
        return self


class RepairPlanningStatus(str, Enum):
    PLANNED = "PLANNED"
    UNSUPPORTED = "UNSUPPORTED"
    NO_CHANGE = "NO_CHANGE"
    ERROR = "ERROR"


class RepairPlanningResult(StrictModel):
    status: RepairPlanningStatus
    result_code: str
    repair_plan: RepairPlan | None = None

    @model_validator(mode="after")
    def validate_result(self) -> "RepairPlanningResult":
        if self.status == RepairPlanningStatus.PLANNED and self.repair_plan is None:
            raise ValueError("PLANNED requires RepairPlan")
        if self.status != RepairPlanningStatus.PLANNED and self.repair_plan is not None:
            raise ValueError("non-planned result cannot contain RepairPlan")
        return self


class RepairPlanErrorCode(str, Enum):
    DESIGN_MISMATCH = "DESIGN_MISMATCH"
    SPEC_VERSION_MISMATCH = "SPEC_VERSION_MISMATCH"
    REVISION_MISMATCH = "REVISION_MISMATCH"
    TARGET_REVISION_INVALID = "TARGET_REVISION_INVALID"
    SOURCE_REPORT_MISMATCH = "SOURCE_REPORT_MISMATCH"
    UNKNOWN_OPERATION = "UNKNOWN_OPERATION"
    UNKNOWN_PARAMETER = "UNKNOWN_PARAMETER"
    STALE_OLD_VALUE = "STALE_OLD_VALUE"
    UNKNOWN_CONSTRAINT = "UNKNOWN_CONSTRAINT"
    CONSTRAINT_NOT_FAILED = "CONSTRAINT_NOT_FAILED"
    UNKNOWN_FEATURE = "UNKNOWN_FEATURE"
    ACTION_NOT_ALLOWED = "ACTION_NOT_ALLOWED"
    NEW_VALUE_NOT_AUTHORIZED = "NEW_VALUE_NOT_AUTHORIZED"
    DUPLICATE_TARGET = "DUPLICATE_TARGET"


class RepairPlanValidationError(StrictModel):
    code: RepairPlanErrorCode
    action_id: str | None = None
    message: str


class RepairPlanValidationStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"


class RepairPlanValidation(StrictModel):
    contract_version: Literal["0.1"] = CONTRACT_VERSION
    status: RepairPlanValidationStatus
    errors: list[RepairPlanValidationError] = Field(default_factory=list)
    repair_plan_digest: str | None = None
    source_plan_digest: str | None = None

    @model_validator(mode="after")
    def validate_summary(self) -> "RepairPlanValidation":
        if self.status == RepairPlanValidationStatus.VALID:
            if self.errors or not self.repair_plan_digest or not self.source_plan_digest:
                raise ValueError("VALID requires both digests and no errors")
        elif not self.errors:
            raise ValueError("INVALID requires structured errors")
        return self


class RevisionChange(StrictModel):
    """Record of an action actually applied by RepairExecutor."""

    change_id: str
    reason: str
    constraint_id: str
    target: str
    old_value: float
    new_value: float
    related_operation_id: str
    related_feature_ids: list[str] = Field(default_factory=list)


class RepairResult(StrictModel):
    contract_version: Literal["0.1"] = CONTRACT_VERSION
    design_id: str
    spec_version: str
    source_revision_id: str
    status: RepairStatus
    changes: list[RevisionChange] = Field(default_factory=list)
    revised_plan: CADPlan | None = None
    result_code: str

    @model_validator(mode="after")
    def validate_status(self) -> "RepairResult":
        if self.status == RepairStatus.REPAIRED:
            if not self.changes or self.revised_plan is None:
                raise ValueError("REPAIRED requires changes and revised_plan")
            if self.revised_plan.parent_revision_id != self.source_revision_id:
                raise ValueError("revised plan must reference the source revision")
        elif self.changes or self.revised_plan is not None:
            raise ValueError("non-repaired outcomes cannot contain changes or a revised plan")
        return self


class DeterministicRepairPlanner:
    """Produces RepairPlan only; it never mutates or creates a CADPlan revision."""

    supported_measurements = {
        MeasurementType.EXTENT_X,
        MeasurementType.EXTENT_Y,
        MeasurementType.EXTENT_Z,
    }

    def __init__(self, rules: list[RepairRule]):
        self.rules = _index_rules(rules)

    def plan(self, spec: DesignSpec, source_plan: CADPlan, report: ValidationReport) -> RepairPlanningResult:
        try:
            _assert_source_coherent(spec, source_plan, report)
            if report.status == ReportStatus.PASS:
                return RepairPlanningResult(status=RepairPlanningStatus.NO_CHANGE, result_code="NO_CHANGE_REQUIRED")
            failures = [check for check in report.checks if check.blocking and check.status == CheckStatus.FAIL]
            if not failures:
                return RepairPlanningResult(status=RepairPlanningStatus.UNSUPPORTED, result_code="NO_REPAIR_AVAILABLE")

            constraints = {constraint.constraint_id: constraint for constraint in spec.constraints}
            operations = {operation.operation_id: operation for operation in source_plan.operations}
            actions: list[RepairAction] = []
            for index, check in enumerate(failures, start=1):
                constraint = constraints.get(check.constraint_id)
                rule = self.rules.get(check.constraint_id)
                if constraint is None or rule is None or not self._constraint_supported(constraint):
                    return RepairPlanningResult(status=RepairPlanningStatus.UNSUPPORTED, result_code="NO_REPAIR_AVAILABLE")
                operation = operations.get(rule.operation_id)
                if operation is None or rule.parameter_name not in operation.params:
                    return RepairPlanningResult(status=RepairPlanningStatus.UNSUPPORTED, result_code="REPAIR_TARGET_NOT_FOUND")
                old_value = operation.params[rule.parameter_name]
                if isinstance(old_value, bool) or not isinstance(old_value, (int, float)):
                    return RepairPlanningResult(status=RepairPlanningStatus.UNSUPPORTED, result_code="REPAIR_TARGET_NOT_NUMERIC")
                actions.append(RepairAction(
                    action_id=f"RA{index:03d}",
                    type=RepairActionType.SET_PARAMETER,
                    target=RepairTarget(operation_id=rule.operation_id, parameter=rule.parameter_name),
                    old_value=float(old_value),
                    new_value=float(constraint.target),
                    constraint_ids=[constraint.constraint_id],
                    related_feature_ids=rule.related_feature_ids,
                    reason=f"{constraint.constraint_id} failed dimensional validation",
                ))
            return RepairPlanningResult(
                status=RepairPlanningStatus.PLANNED,
                result_code="SUPPORTED_REPAIR_PLAN_CREATED",
                repair_plan=RepairPlan(
                    design_id=spec.design_id,
                    spec_version=spec.spec_version,
                    source_revision_id=source_plan.revision_id,
                    target_revision_id=_next_revision_id(source_plan.revision_id),
                    actions=actions,
                    rationale="Apply only explicitly mapped HARD dimensional corrections.",
                ),
            )
        except Exception:
            return RepairPlanningResult(status=RepairPlanningStatus.ERROR, result_code="REPAIR_PLANNER_ERROR")

    @classmethod
    def _constraint_supported(cls, constraint: Any) -> bool:
        return (
            constraint.priority == Priority.HARD
            and constraint.comparison == Comparison.EQ
            and constraint.measurement.type in cls.supported_measurements
            and isinstance(constraint.target, (int, float))
            and not isinstance(constraint.target, bool)
        )


class RepairPlanValidator:
    """Trusted semantic authorization boundary for schema-valid RepairPlans."""

    def __init__(self, rules: list[RepairRule]):
        self.rules = _index_rules(rules)

    def validate(
        self,
        repair_plan: RepairPlan,
        spec: DesignSpec,
        source_plan: CADPlan,
        source_report: ValidationReport,
    ) -> RepairPlanValidation:
        errors: list[RepairPlanValidationError] = []

        def add(code: RepairPlanErrorCode, message: str, action_id: str | None = None) -> None:
            errors.append(RepairPlanValidationError(code=code, action_id=action_id, message=message))

        if repair_plan.design_id != source_plan.design_id or repair_plan.design_id != spec.design_id:
            add(RepairPlanErrorCode.DESIGN_MISMATCH, "repair plan design_id does not match source")
        if repair_plan.spec_version != source_plan.spec_version or repair_plan.spec_version != spec.spec_version:
            add(RepairPlanErrorCode.SPEC_VERSION_MISMATCH, "repair plan spec_version does not match source")
        if repair_plan.source_revision_id != source_plan.revision_id:
            add(RepairPlanErrorCode.REVISION_MISMATCH, "repair plan source revision does not match CADPlan")
        if repair_plan.target_revision_id != _next_revision_id(source_plan.revision_id):
            add(RepairPlanErrorCode.TARGET_REVISION_INVALID, "target revision is not the next revision")
        if (source_report.design_id, source_report.spec_version, source_report.revision_id) != (
            source_plan.design_id, source_plan.spec_version, source_plan.revision_id
        ):
            add(RepairPlanErrorCode.SOURCE_REPORT_MISMATCH, "ValidationReport does not belong to source CADPlan")

        operations = {operation.operation_id: operation for operation in source_plan.operations}
        constraints = {constraint.constraint_id: constraint for constraint in spec.constraints}
        failed_constraints = {
            check.constraint_id for check in source_report.checks
            if check.constraint_id and check.blocking and check.status == CheckStatus.FAIL
        }
        known_features = {output.feature_id for operation in source_plan.operations for output in operation.outputs}
        seen_targets: set[str] = set()

        for action in repair_plan.actions:
            action_id = action.action_id
            target_path = action.target.semantic_path
            if target_path in seen_targets:
                add(RepairPlanErrorCode.DUPLICATE_TARGET, "multiple actions target the same parameter", action_id)
            seen_targets.add(target_path)

            operation = operations.get(action.target.operation_id)
            if operation is None:
                add(RepairPlanErrorCode.UNKNOWN_OPERATION, "target operation does not exist", action_id)
            elif action.target.parameter not in operation.params:
                add(RepairPlanErrorCode.UNKNOWN_PARAMETER, "target parameter does not exist", action_id)
            elif operation.params[action.target.parameter] != action.old_value:
                add(RepairPlanErrorCode.STALE_OLD_VALUE, "old_value does not match source CADPlan", action_id)

            for constraint_id in action.constraint_ids:
                constraint = constraints.get(constraint_id)
                if constraint is None:
                    add(RepairPlanErrorCode.UNKNOWN_CONSTRAINT, "constraint does not exist", action_id)
                    continue
                if constraint_id not in failed_constraints:
                    add(RepairPlanErrorCode.CONSTRAINT_NOT_FAILED, "constraint is not a blocking FAIL in source report", action_id)
                rule = self.rules.get(constraint_id)
                if rule is None or rule.operation_id != action.target.operation_id or rule.parameter_name != action.target.parameter:
                    add(RepairPlanErrorCode.ACTION_NOT_ALLOWED, "SET_PARAMETER target is not permitted for constraint", action_id)
                if constraint.comparison != Comparison.EQ or constraint.target != action.new_value:
                    add(RepairPlanErrorCode.NEW_VALUE_NOT_AUTHORIZED, "new_value is not the constraint target", action_id)

            for feature_id in action.related_feature_ids:
                if feature_id not in known_features:
                    add(RepairPlanErrorCode.UNKNOWN_FEATURE, "related semantic feature does not exist", action_id)

        if errors:
            return RepairPlanValidation(status=RepairPlanValidationStatus.INVALID, errors=errors)
        return RepairPlanValidation(
            status=RepairPlanValidationStatus.VALID,
            repair_plan_digest=_digest(repair_plan),
            source_plan_digest=_digest(source_plan),
        )


class RepairExecutor:
    """Applies only an authorized SET_PARAMETER plan; performs no planning or CAD."""

    def execute(
        self,
        repair_plan: RepairPlan,
        validation: RepairPlanValidation,
        source_plan: CADPlan,
    ) -> RepairResult:
        if (
            validation.status != RepairPlanValidationStatus.VALID
            or validation.repair_plan_digest != _digest(repair_plan)
            or validation.source_plan_digest != _digest(source_plan)
        ):
            return _repair_result(source_plan, RepairStatus.ERROR, "REPAIR_PLAN_NOT_VALIDATED")
        try:
            plan_data = source_plan.model_dump(mode="python")
            plan_data["revision_id"] = repair_plan.target_revision_id
            plan_data["parent_revision_id"] = repair_plan.source_revision_id
            operations = {item["operation_id"]: item for item in plan_data["operations"]}
            changes: list[RevisionChange] = []
            for index, action in enumerate(repair_plan.actions, start=1):
                operation = operations[action.target.operation_id]
                if operation["params"][action.target.parameter] != action.old_value:
                    return _repair_result(source_plan, RepairStatus.ERROR, "STALE_SOURCE_PLAN")
                operation["params"][action.target.parameter] = action.new_value
                changes.append(RevisionChange(
                    change_id=f"CHG-{index:03d}",
                    reason=f"HARD constraint {action.constraint_ids[0]} failed",
                    constraint_id=action.constraint_ids[0],
                    target=action.target.semantic_path,
                    old_value=action.old_value,
                    new_value=action.new_value,
                    related_operation_id=action.target.operation_id,
                    related_feature_ids=action.related_feature_ids,
                ))
            revised_plan = CADPlan.model_validate(plan_data)
            return RepairResult(
                design_id=source_plan.design_id,
                spec_version=source_plan.spec_version,
                source_revision_id=source_plan.revision_id,
                status=RepairStatus.REPAIRED,
                changes=changes,
                revised_plan=revised_plan,
                result_code="VALIDATED_REPAIR_PLAN_APPLIED",
            )
        except Exception:
            return _repair_result(source_plan, RepairStatus.ERROR, "REPAIR_EXECUTION_ERROR")


class DeterministicRepairController:
    """Backward-compatible M1 facade assembled from the explicit M2 stages."""

    def __init__(self, rules: list[RepairRule]):
        self.planner = DeterministicRepairPlanner(rules)
        self.validator = RepairPlanValidator(rules)
        self.executor = RepairExecutor()

    def propose(self, spec: DesignSpec, plan: CADPlan, report: ValidationReport) -> RepairResult:
        planning = self.planner.plan(spec, plan, report)
        if planning.status == RepairPlanningStatus.NO_CHANGE:
            return _repair_result(plan, RepairStatus.NO_CHANGE, planning.result_code)
        if planning.status == RepairPlanningStatus.UNSUPPORTED:
            return _repair_result(plan, RepairStatus.UNSUPPORTED, planning.result_code)
        if planning.status != RepairPlanningStatus.PLANNED or planning.repair_plan is None:
            return _repair_result(plan, RepairStatus.ERROR, planning.result_code)
        validation = self.validator.validate(planning.repair_plan, spec, plan, report)
        if validation.status != RepairPlanValidationStatus.VALID:
            return _repair_result(plan, RepairStatus.UNSUPPORTED, "REPAIR_PLAN_INVALID")
        return self.executor.execute(planning.repair_plan, validation, plan)


def _index_rules(rules: list[RepairRule]) -> dict[str, RepairRule]:
    indexed = {rule.constraint_id: rule for rule in rules}
    if len(indexed) != len(rules):
        raise ValueError("repair rule constraint_id values must be unique")
    return indexed


def _assert_source_coherent(spec: DesignSpec, plan: CADPlan, report: ValidationReport) -> None:
    expected = (spec.design_id, spec.spec_version, plan.revision_id)
    actual = (report.design_id, report.spec_version, report.revision_id)
    if spec.design_id != plan.design_id or spec.spec_version != plan.spec_version or actual != expected:
        raise ValueError("spec, plan and report identifiers are incoherent")


def _next_revision_id(revision_id: str) -> str:
    match = re.fullmatch(r"R(\d{2,})", revision_id)
    if not match:
        raise ValueError("invalid revision_id")
    width = len(match.group(1))
    return f"R{int(match.group(1)) + 1:0{width}d}"


def _digest(model: StrictModel) -> str:
    canonical = json.dumps(model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _repair_result(plan: CADPlan, status: RepairStatus, code: str) -> RepairResult:
    return RepairResult(
        design_id=plan.design_id,
        spec_version=plan.spec_version,
        source_revision_id=plan.revision_id,
        status=status,
        result_code=code,
    )


def changed_semantic_plan_paths(before: CADPlan, after: CADPlan) -> set[str]:
    """Compare plan semantics while deliberately ignoring revision metadata."""

    changed: set[str] = set()
    for field in ("contract_version", "design_id", "spec_version", "units_length", "units_angle"):
        if getattr(before, field) != getattr(after, field):
            changed.add(field)

    before_ids = [operation.operation_id for operation in before.operations]
    after_ids = [operation.operation_id for operation in after.operations]
    if before_ids != after_ids:
        changed.add("operations.order")
    before_ops = {operation.operation_id: operation for operation in before.operations}
    after_ops = {operation.operation_id: operation for operation in after.operations}
    for operation_id in sorted(set(before_ops) | set(after_ops)):
        left, right = before_ops.get(operation_id), after_ops.get(operation_id)
        if left is None or right is None:
            changed.add(f"{operation_id}.presence")
            continue
        for field in ("operation_type", "operation_version", "execution_mode", "inputs", "outputs"):
            if getattr(left, field) != getattr(right, field):
                changed.add(f"{operation_id}.{field}")
        for name in sorted(set(left.params) | set(right.params)):
            if left.params.get(name) != right.params.get(name):
                changed.add(f"{operation_id}.params.{name}")
    return changed


def preserved_passing_constraints(before: ValidationReport, after: ValidationReport) -> tuple[bool, list[str]]:
    """Return whether every R01 PASS constraint remains PASS in R02."""

    passed_before = [check.constraint_id for check in before.checks if check.constraint_id and check.status == CheckStatus.PASS]
    after_by_id = {check.constraint_id: check.status for check in after.checks}
    return all(after_by_id.get(constraint_id) == CheckStatus.PASS for constraint_id in passed_before), passed_before
