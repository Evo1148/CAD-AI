from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import Field, model_validator

from .cad import CADResult, CADStatus, GeometryInspector
from .contracts import Comparison, Constraint, CONTRACT_VERSION, DesignSpec, MeasurementType, Priority, StrictModel


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"
    UNSUPPORTED = "UNSUPPORTED"
    ERROR = "ERROR"


class CheckResult(StrictModel):
    check_id: str
    constraint_id: str | None = None
    status: CheckStatus
    blocking: bool
    measurement: MeasurementType
    expected: dict[str, Any] | None = None
    actual: Any = None
    error_code: str | None = None
    repair_hint: str | None = None


class ReportStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCOMPLETE = "INCOMPLETE"


class ValidationReport(StrictModel):
    contract_version: Literal["0.1"] = CONTRACT_VERSION
    design_id: str
    spec_version: str
    revision_id: str
    status: ReportStatus
    complete: bool
    checks: list[CheckResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_summary(self) -> "ValidationReport":
        hard_fail = any(c.blocking and c.status == CheckStatus.FAIL for c in self.checks)
        uncertain = any(c.blocking and c.status in {CheckStatus.ERROR, CheckStatus.UNSUPPORTED, CheckStatus.SKIPPED} for c in self.checks)
        if self.status == ReportStatus.PASS and (not self.complete or hard_fail or uncertain):
            raise ValueError("PASS requires complete=true and all blocking checks to pass")
        if self.status == ReportStatus.FAIL and not hard_fail:
            raise ValueError("FAIL requires a blocking FAIL check")
        if self.status == ReportStatus.INCOMPLETE and (hard_fail or not uncertain):
            raise ValueError("INCOMPLETE requires unresolved blocking checks and no blocking FAIL")
        return self


class Validator:
    implemented = {
        MeasurementType.MODEL_VALID,
        MeasurementType.SOLID_COUNT,
        MeasurementType.EXTENT_X,
        MeasurementType.EXTENT_Y,
        MeasurementType.EXTENT_Z,
        MeasurementType.DISTANCE,
        MeasurementType.DIAMETER,
        MeasurementType.VOLUME,
        MeasurementType.FEATURE_EXISTS,
        MeasurementType.HOLE_COUNT,
        MeasurementType.POSITION_X,
        MeasurementType.POSITION_Y,
        MeasurementType.FILLET_RADIUS,
        MeasurementType.CHAMFER_DISTANCE,
        MeasurementType.RECTANGULAR_POCKET_COUNT,
        MeasurementType.CIRCULAR_POCKET_COUNT,
        MeasurementType.SLOT_COUNT,
        MeasurementType.FEATURE_WIDTH,
        MeasurementType.FEATURE_DEPTH,
        MeasurementType.CUT_DEPTH,
        MeasurementType.THROUGH,
        MeasurementType.SLOT_LENGTH,
        MeasurementType.SLOT_WIDTH,
        MeasurementType.ORIENTATION_ANGLE,
        MeasurementType.RECTANGULAR_BOSS_COUNT,
        MeasurementType.CYLINDRICAL_BOSS_COUNT,
        MeasurementType.STANDOFF_COUNT,
        MeasurementType.FEATURE_HEIGHT,
        MeasurementType.BASE_HEIGHT,
        MeasurementType.OUTER_DIAMETER,
        MeasurementType.INNER_DIAMETER,
    }

    def validate(self, spec: DesignSpec, result: CADResult) -> ValidationReport:
        if result.design_id != spec.design_id or result.spec_version != spec.spec_version:
            raise ValueError("DesignSpec and CADResult identifiers are incoherent")
        if result.status != CADStatus.SUCCESS:
            raise ValueError("CAD generation errors are not validation failures")
        inspector = GeometryInspector(result)
        checks = [self._check(constraint, inspector) for constraint in spec.constraints]
        hard_fail = any(c.blocking and c.status == CheckStatus.FAIL for c in checks)
        uncertain = any(c.blocking and c.status in {CheckStatus.ERROR, CheckStatus.UNSUPPORTED, CheckStatus.SKIPPED} for c in checks)
        if hard_fail:
            status, complete = ReportStatus.FAIL, not uncertain
        elif uncertain:
            status, complete = ReportStatus.INCOMPLETE, False
        else:
            status, complete = ReportStatus.PASS, True
        return ValidationReport(
            design_id=spec.design_id,
            spec_version=spec.spec_version,
            revision_id=result.revision_id,
            status=status,
            complete=complete,
            checks=checks,
        )

    def _check(self, constraint: Constraint, inspector: GeometryInspector) -> CheckResult:
        blocking = constraint.priority == Priority.HARD
        expected = self._expected(constraint)
        if constraint.measurement.type not in self.implemented:
            return CheckResult(
                check_id=f"check:{constraint.constraint_id}", constraint_id=constraint.constraint_id,
                status=CheckStatus.UNSUPPORTED, blocking=blocking, measurement=constraint.measurement.type,
                expected=expected, error_code="MEASUREMENT_UNSUPPORTED",
            )
        try:
            actual = self._measure(constraint, inspector)
            passes = self._compare(actual, constraint)
            status = CheckStatus.PASS if passes else (CheckStatus.FAIL if blocking else CheckStatus.WARNING)
            return CheckResult(
                check_id=f"check:{constraint.constraint_id}", constraint_id=constraint.constraint_id,
                status=status, blocking=blocking, measurement=constraint.measurement.type,
                expected=expected, actual=actual,
            )
        except Exception:
            return CheckResult(
                check_id=f"check:{constraint.constraint_id}", constraint_id=constraint.constraint_id,
                status=CheckStatus.ERROR, blocking=blocking, measurement=constraint.measurement.type,
                expected=expected, error_code="MEASUREMENT_FAILED",
            )

    @staticmethod
    def _measure(constraint: Constraint, inspector: GeometryInspector) -> float | int | bool:
        measurement = constraint.measurement
        if measurement.type == MeasurementType.MODEL_VALID:
            return inspector.model_valid()
        if measurement.type == MeasurementType.SOLID_COUNT:
            return inspector.solid_count()
        if measurement.type in {MeasurementType.EXTENT_X, MeasurementType.EXTENT_Y, MeasurementType.EXTENT_Z}:
            return inspector.extent(measurement.type.value[-1])
        if measurement.type == MeasurementType.VOLUME:
            return inspector.volume()
        if measurement.type == MeasurementType.DISTANCE:
            if not measurement.feature_id or not measurement.reference_feature_id:
                raise ValueError("DISTANCE requires two semantic feature references")
            return inspector.distance(measurement.feature_id, measurement.reference_feature_id)
        if measurement.type == MeasurementType.DIAMETER:
            if not measurement.feature_id:
                raise ValueError("DIAMETER requires feature_id")
            return inspector.diameter(measurement.feature_id)
        if measurement.type == MeasurementType.FEATURE_EXISTS:
            if not measurement.feature_id:
                raise ValueError("FEATURE_EXISTS requires feature_id")
            return inspector.feature_exists(measurement.feature_id)
        if measurement.type == MeasurementType.HOLE_COUNT:
            return inspector.hole_count()
        if measurement.type == MeasurementType.RECTANGULAR_POCKET_COUNT:
            return inspector.feature_count("rectangular_pocket")
        if measurement.type == MeasurementType.CIRCULAR_POCKET_COUNT:
            return inspector.feature_count("circular_pocket")
        if measurement.type == MeasurementType.SLOT_COUNT:
            return inspector.feature_count("slot")
        if measurement.type == MeasurementType.RECTANGULAR_BOSS_COUNT:
            return inspector.feature_count("rectangular_boss")
        if measurement.type == MeasurementType.CYLINDRICAL_BOSS_COUNT:
            return inspector.feature_count("cylindrical_boss")
        if measurement.type == MeasurementType.STANDOFF_COUNT:
            return inspector.feature_count("standoff")
        if measurement.type in {MeasurementType.POSITION_X, MeasurementType.POSITION_Y}:
            if not measurement.feature_id:
                raise ValueError("POSITION requires feature_id")
            return inspector.position(measurement.feature_id, measurement.type.value[-1])
        if measurement.type == MeasurementType.FILLET_RADIUS:
            if not measurement.feature_id:
                raise ValueError("FILLET_RADIUS requires feature_id")
            return inspector.fillet_radius(measurement.feature_id)
        if measurement.type == MeasurementType.CHAMFER_DISTANCE:
            if not measurement.feature_id:
                raise ValueError("CHAMFER_DISTANCE requires feature_id")
            return inspector.chamfer_distance(measurement.feature_id)
        if not measurement.feature_id:
            raise ValueError(f"{measurement.type.value} requires feature_id")
        if measurement.type == MeasurementType.FEATURE_WIDTH:
            return inspector.feature_width(measurement.feature_id)
        if measurement.type == MeasurementType.FEATURE_DEPTH:
            return inspector.feature_depth(measurement.feature_id)
        if measurement.type == MeasurementType.FEATURE_HEIGHT:
            return inspector.feature_height(measurement.feature_id)
        if measurement.type == MeasurementType.BASE_HEIGHT:
            return inspector.base_height(measurement.feature_id)
        if measurement.type == MeasurementType.OUTER_DIAMETER:
            return inspector.outer_diameter(measurement.feature_id)
        if measurement.type == MeasurementType.INNER_DIAMETER:
            return inspector.inner_diameter(measurement.feature_id)
        if measurement.type == MeasurementType.CUT_DEPTH:
            return inspector.cut_depth(measurement.feature_id)
        if measurement.type == MeasurementType.THROUGH:
            return inspector.through(measurement.feature_id)
        if measurement.type == MeasurementType.SLOT_LENGTH:
            return inspector.slot_length(measurement.feature_id)
        if measurement.type == MeasurementType.SLOT_WIDTH:
            return inspector.slot_width(measurement.feature_id)
        if measurement.type == MeasurementType.ORIENTATION_ANGLE:
            return inspector.orientation_angle(measurement.feature_id)
        raise NotImplementedError(measurement.type)

    @staticmethod
    def _compare(actual: Any, constraint: Constraint) -> bool:
        if constraint.comparison == Comparison.EQ:
            if isinstance(actual, bool):
                return actual is constraint.target
            return abs(float(actual) - float(constraint.target)) <= constraint.tolerance
        if constraint.comparison == Comparison.GTE:
            return float(actual) >= float(constraint.minimum)
        if constraint.comparison == Comparison.LTE:
            return float(actual) <= float(constraint.maximum)
        return float(constraint.minimum) <= float(actual) <= float(constraint.maximum)

    @staticmethod
    def _expected(constraint: Constraint) -> dict[str, Any]:
        return {
            "comparison": constraint.comparison.value,
            "target": constraint.target,
            "minimum": constraint.minimum,
            "maximum": constraint.maximum,
            "tolerance": constraint.tolerance,
            "units": "mm/mm2/mm3/deg as defined by measurement",
        }
