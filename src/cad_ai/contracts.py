from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_VERSION = "0.1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class Priority(str, Enum):
    HARD = "HARD"
    SOFT = "SOFT"


class Origin(str, Enum):
    USER = "USER"
    DERIVED = "DERIVED"
    SYSTEM = "SYSTEM"


class MeasurementType(str, Enum):
    MODEL_VALID = "MODEL_VALID"
    SOLID_COUNT = "SOLID_COUNT"
    EXTENT_X = "EXTENT_X"
    EXTENT_Y = "EXTENT_Y"
    EXTENT_Z = "EXTENT_Z"
    DISTANCE = "DISTANCE"
    ANGLE = "ANGLE"
    DIAMETER = "DIAMETER"
    RADIUS = "RADIUS"
    VOLUME = "VOLUME"
    FEATURE_EXISTS = "FEATURE_EXISTS"
    HOLE_COUNT = "HOLE_COUNT"
    POSITION_X = "POSITION_X"
    POSITION_Y = "POSITION_Y"
    FILLET_RADIUS = "FILLET_RADIUS"
    CHAMFER_DISTANCE = "CHAMFER_DISTANCE"
    RECTANGULAR_POCKET_COUNT = "RECTANGULAR_POCKET_COUNT"
    CIRCULAR_POCKET_COUNT = "CIRCULAR_POCKET_COUNT"
    SLOT_COUNT = "SLOT_COUNT"
    FEATURE_WIDTH = "FEATURE_WIDTH"
    FEATURE_DEPTH = "FEATURE_DEPTH"
    CUT_DEPTH = "CUT_DEPTH"
    THROUGH = "THROUGH"
    SLOT_LENGTH = "SLOT_LENGTH"
    SLOT_WIDTH = "SLOT_WIDTH"
    ORIENTATION_ANGLE = "ORIENTATION_ANGLE"
    RECTANGULAR_BOSS_COUNT = "RECTANGULAR_BOSS_COUNT"
    CYLINDRICAL_BOSS_COUNT = "CYLINDRICAL_BOSS_COUNT"
    STANDOFF_COUNT = "STANDOFF_COUNT"
    FEATURE_HEIGHT = "FEATURE_HEIGHT"
    BASE_HEIGHT = "BASE_HEIGHT"
    OUTER_DIAMETER = "OUTER_DIAMETER"
    INNER_DIAMETER = "INNER_DIAMETER"


class Comparison(str, Enum):
    EQ = "EQ"
    GTE = "GTE"
    LTE = "LTE"
    BETWEEN = "BETWEEN"


class Measurement(StrictModel):
    type: MeasurementType
    feature_id: str | None = None
    reference_feature_id: str | None = None


class Constraint(StrictModel):
    constraint_id: str
    priority: Priority
    origin: Origin
    measurement: Measurement
    comparison: Comparison
    target: float | bool | None = None
    minimum: float | None = None
    maximum: float | None = None
    tolerance: float = 0.0

    @model_validator(mode="after")
    def validate_comparison(self) -> "Constraint":
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        if self.comparison == Comparison.EQ:
            if self.target is None or self.minimum is not None or self.maximum is not None:
                raise ValueError("EQ requires target and forbids minimum/maximum")
        elif self.comparison == Comparison.GTE:
            if self.minimum is None or self.target is not None or self.maximum is not None:
                raise ValueError("GTE requires only minimum")
        elif self.comparison == Comparison.LTE:
            if self.maximum is None or self.target is not None or self.minimum is not None:
                raise ValueError("LTE requires only maximum")
        elif self.comparison == Comparison.BETWEEN:
            if self.minimum is None or self.maximum is None or self.target is not None:
                raise ValueError("BETWEEN requires minimum/maximum and forbids target")
            if self.minimum > self.maximum:
                raise ValueError("BETWEEN minimum cannot exceed maximum")
        return self


class Assumption(StrictModel):
    assumption_id: str
    description: str


class Unknown(StrictModel):
    unknown_id: str
    description: str
    blocking: bool


class DesignSpec(StrictModel):
    contract_version: Literal["0.1"] = CONTRACT_VERSION
    design_id: str
    spec_version: str
    parameters: dict[str, float | int | str | bool] = Field(default_factory=dict)
    constraints: list[Constraint]
    assumptions: list[Assumption] = Field(default_factory=list)
    unknowns: list[Unknown] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_ids(self) -> "DesignSpec":
        constraint_ids = [item.constraint_id for item in self.constraints]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("constraint_id values must be unique")
        return self


class PlanningBlockedError(ValueError):
    pass


def ensure_spec_plannable(spec: DesignSpec) -> None:
    blocking = [item.unknown_id for item in spec.unknowns if item.blocking]
    if blocking:
        raise PlanningBlockedError(f"blocking unknowns prevent executable planning: {blocking}")


class FeatureKind(str, Enum):
    BODY = "BODY"
    FEATURE = "FEATURE"
    DATUM_POINT = "DATUM_POINT"
    DATUM_AXIS = "DATUM_AXIS"
    DATUM_PLANE = "DATUM_PLANE"


class OperationOutput(StrictModel):
    feature_id: str
    kind: FeatureKind
    semantic_role: str | None = None


class ExecutionMode(str, Enum):
    CONTROLLED = "CONTROLLED"
    GENERATED = "GENERATED"


def _contains_unresolved(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith("$")
    if isinstance(value, dict):
        return any(_contains_unresolved(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_unresolved(v) for v in value)
    return False


class CADOperation(StrictModel):
    operation_id: str
    operation_type: str
    operation_version: Literal["0.1"] = "0.1"
    execution_mode: ExecutionMode = ExecutionMode.CONTROLLED
    inputs: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    outputs: list[OperationOutput]

    @model_validator(mode="after")
    def resolved_and_unique_outputs(self) -> "CADOperation":
        if _contains_unresolved(self.params):
            raise ValueError("CAD operation parameters must be fully resolved")
        ids = [output.feature_id for output in self.outputs]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("operation outputs must be non-empty and unique")
        return self


def _revision_number(value: str) -> int:
    match = re.fullmatch(r"R(\d{2,})", value)
    if not match:
        raise ValueError("revision_id must use R01/R02/... format")
    return int(match.group(1))


class CADPlan(StrictModel):
    contract_version: Literal["0.1"] = CONTRACT_VERSION
    design_id: str
    spec_version: str
    revision_id: str
    parent_revision_id: str | None = None
    units_length: Literal["mm"] = "mm"
    units_angle: Literal["deg"] = "deg"
    operations: list[CADOperation]

    @model_validator(mode="after")
    def validate_revision_and_ids(self) -> "CADPlan":
        current = _revision_number(self.revision_id)
        if current == 1 and self.parent_revision_id is not None:
            raise ValueError("R01 cannot have a parent revision")
        if current > 1:
            if self.parent_revision_id is None:
                raise ValueError("revisions after R01 require parent_revision_id")
            if _revision_number(self.parent_revision_id) != current - 1:
                raise ValueError("parent_revision_id must be the immediately preceding revision")
        operation_ids = [op.operation_id for op in self.operations]
        output_ids = [out.feature_id for op in self.operations for out in op.outputs]
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("operation_id values must be unique")
        if len(output_ids) != len(set(output_ids)):
            raise ValueError("feature_id values must be globally unique")
        return self
