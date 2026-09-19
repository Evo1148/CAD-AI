from __future__ import annotations

import math
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import cadquery as cq
from pydantic import Field, model_validator

from .contracts import CADPlan, CONTRACT_VERSION, ExecutionMode, FeatureKind, StrictModel


class LocatorType(str, Enum):
    BODY = "BODY"
    CYLINDRICAL_SURFACE = "CYLINDRICAL_SURFACE"
    POINT = "POINT"
    AXIS = "AXIS"
    PLANE = "PLANE"
    EDGE_TREATMENT = "EDGE_TREATMENT"
    SUBTRACTIVE_FEATURE = "SUBTRACTIVE_FEATURE"
    ADDITIVE_FEATURE = "ADDITIVE_FEATURE"


class GeometryLocator(StrictModel):
    locator_type: LocatorType
    data: dict[str, Any]

    @model_validator(mode="after")
    def forbid_public_topology_indices(self) -> "GeometryLocator":
        serialized = str(self.data).lower()
        forbidden = ("face_index", "edge_index", "vertex_index", "face[", "edge[", "vertex[")
        if any(item in serialized for item in forbidden):
            raise ValueError("fragile OCC topology indices are not valid semantic locators")
        return self


class FeatureRecord(StrictModel):
    feature_id: str
    kind: FeatureKind
    produced_by_operation_id: str
    semantic_role: str | None = None
    locator: GeometryLocator


class FeatureRegistry(StrictModel):
    features: dict[str, FeatureRecord] = Field(default_factory=dict)

    def register(self, feature: FeatureRecord) -> None:
        if feature.feature_id in self.features:
            raise ValueError(f"duplicate feature_id: {feature.feature_id}")
        self.features[feature.feature_id] = feature

    def require(self, feature_id: str) -> FeatureRecord:
        try:
            return self.features[feature_id]
        except KeyError as exc:
            raise KeyError(f"unknown feature_id: {feature_id}") from exc


class CADStatus(str, Enum):
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"


class CADExecutionErrorCategory(str, Enum):
    PLAN_REJECTED = "PLAN_REJECTED"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    KERNEL_ERROR = "KERNEL_ERROR"
    GENERATED_CODE_REJECTED = "GENERATED_CODE_REJECTED"
    TIMEOUT = "TIMEOUT"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"


class CADExecutionError(StrictModel):
    code: str
    category: CADExecutionErrorCategory
    operation_id: str | None = None
    message: str
    retryable: bool = False


class CADResult(StrictModel):
    contract_version: Literal["0.1"] = CONTRACT_VERSION
    design_id: str
    spec_version: str
    revision_id: str
    status: CADStatus
    model_reference: str | None = None
    feature_registry: FeatureRegistry | None = None
    error: CADExecutionError | None = None
    model_handle: Any = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def enforce_status_invariants(self) -> "CADResult":
        if self.status == CADStatus.SUCCESS:
            if self.model_reference is None or self.model_handle is None:
                raise ValueError("SUCCESS requires model_reference and model_handle")
            if self.feature_registry is None:
                raise ValueError("SUCCESS requires FeatureRegistry")
            if self.error is not None:
                raise ValueError("SUCCESS cannot contain an error")
        else:
            if self.error is None:
                raise ValueError("ERROR requires CADExecutionError")
        return self


class _PlanRejected(Exception):
    def __init__(self, code: str, message: str, operation_id: str | None = None):
        super().__init__(message)
        self.code = code
        self.operation_id = operation_id


class CADEngine:
    """Small controlled-operation CadQuery engine for the V0.1 slice."""

    supported_operations = frozenset({
        "box", "cylinder", "hole", "rect_pocket", "circular_pocket", "slot",
        "fillet", "chamfer", "rect_boss", "cyl_boss", "standoff",
    })

    def execute(self, plan: CADPlan) -> CADResult:
        registry = FeatureRegistry()
        model: cq.Shape | None = None
        current_operation: str | None = None
        try:
            declared: set[str] = set()
            for operation in plan.operations:
                current_operation = operation.operation_id
                if operation.execution_mode != ExecutionMode.CONTROLLED:
                    raise _PlanRejected("GENERATED_NOT_ALLOWED", "generated operations are disabled", current_operation)
                if operation.operation_type not in self.supported_operations:
                    raise _PlanRejected("UNKNOWN_CONTROLLED_OPERATION", f"unsupported controlled operation: {operation.operation_type}", current_operation)
                missing = [item for item in operation.inputs if item not in declared]
                if missing:
                    raise _PlanRejected("UNKNOWN_FEATURE_REFERENCE", f"inputs reference unknown features: {missing}", current_operation)
                model = self._execute_operation(operation, model, registry)
                for output in operation.outputs:
                    declared.add(output.feature_id)
                missing_outputs = [item.feature_id for item in operation.outputs if item.feature_id not in registry.features]
                if missing_outputs:
                    raise _PlanRejected("MISSING_DECLARED_OUTPUT", f"declared outputs not registered: {missing_outputs}", current_operation)
            if model is None:
                raise _PlanRejected("EMPTY_PLAN", "plan produced no model")
            return CADResult(
                design_id=plan.design_id,
                spec_version=plan.spec_version,
                revision_id=plan.revision_id,
                status=CADStatus.SUCCESS,
                model_reference=f"memory://{plan.design_id}/{plan.revision_id}",
                model_handle=model,
                feature_registry=registry,
            )
        except _PlanRejected as exc:
            return self._error_result(plan, exc.code, CADExecutionErrorCategory.PLAN_REJECTED, str(exc), exc.operation_id)
        except (ValueError, KeyError) as exc:
            return self._error_result(plan, "INVALID_OPERATION_PARAMETERS", CADExecutionErrorCategory.EXECUTION_ERROR, str(exc), current_operation)
        except Exception as exc:  # CadQuery/OCP exceptions are normalized at the public boundary.
            return self._error_result(plan, "KERNEL_OPERATION_FAILED", CADExecutionErrorCategory.KERNEL_ERROR, str(exc), current_operation)

    def _execute_operation(self, operation: Any, model: cq.Shape | None, registry: FeatureRegistry) -> cq.Shape:
        params = operation.params
        if operation.operation_type == "box":
            if model is not None:
                raise _PlanRejected("MULTIPLE_ROOT_BODIES", "box requires an empty model", operation.operation_id)
            x, y, z = (self._positive(params, name) for name in ("x", "y", "z"))
            shape = cq.Workplane("XY").box(x, y, z, centered=(True, True, False)).val()
            self._register_outputs(operation, registry, {FeatureKind.BODY: GeometryLocator(
                locator_type=LocatorType.BODY, data={"width": x, "depth": y, "height": z},
            )})
            return shape
        if operation.operation_type == "cylinder":
            if model is not None:
                raise _PlanRejected("MULTIPLE_ROOT_BODIES", "cylinder requires an empty model", operation.operation_id)
            radius, height = self._positive(params, "radius"), self._positive(params, "height")
            shape = cq.Workplane("XY").circle(radius).extrude(height).val()
            self._register_outputs(operation, registry, {FeatureKind.BODY: GeometryLocator(locator_type=LocatorType.BODY, data={})})
            return shape
        if model is None:
            raise _PlanRejected("MISSING_BODY", f"{operation.operation_type} requires an existing body", operation.operation_id)
        if operation.operation_type == "hole":
            diameter = self._positive(params, "diameter")
            x, y = float(params["x"]), float(params["y"])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("hole coordinates must be finite numbers")
            bbox = model.BoundingBox()
            cutter = cq.Workplane("XY").workplane(offset=bbox.zmin - 1.0).center(x, y).circle(diameter / 2.0).extrude(bbox.zlen + 2.0).val()
            shape = model.cut(cutter)
            locators = {
                FeatureKind.FEATURE: GeometryLocator(
                    locator_type=LocatorType.CYLINDRICAL_SURFACE,
                    data={"origin": [x, y, bbox.zmin], "direction": [0.0, 0.0, 1.0]},
                ),
                FeatureKind.DATUM_AXIS: GeometryLocator(
                    locator_type=LocatorType.AXIS,
                    data={"origin": [x, y, bbox.zmin], "direction": [0.0, 0.0, 1.0]},
                ),
            }
            self._register_outputs(operation, registry, locators)
            return shape
        if operation.operation_type in {"rect_pocket", "circular_pocket", "slot"}:
            x, y = self._finite_coordinates(params)
            through = params.get("through")
            if not isinstance(through, bool):
                raise ValueError("through must be boolean")
            bbox = model.BoundingBox()
            cut_depth = bbox.zlen if through else self._positive(params, "cut_depth")
            if not through and cut_depth >= bbox.zlen:
                raise ValueError("cut_depth must be smaller than the current body height")
            start_z = bbox.zmin - 1.0 if through else bbox.zmax - cut_depth
            extrusion = bbox.zlen + 2.0 if through else cut_depth + 0.001
            common = {
                "origin": [x, y, bbox.zmax],
                "direction": [0.0, 0.0, 1.0],
                "cut_depth": float(cut_depth),
                "through": through,
                "base_height": float(bbox.zlen),
            }
            if operation.operation_type == "rect_pocket":
                width = self._positive(params, "width")
                depth = self._positive(params, "depth")
                cutter = (
                    cq.Workplane("XY").workplane(offset=start_z).center(x, y)
                    .rect(width, depth).extrude(extrusion).val()
                )
                feature_locator = GeometryLocator(
                    locator_type=LocatorType.SUBTRACTIVE_FEATURE,
                    data={**common, "feature_type": "RECTANGULAR_POCKET", "width": width, "depth": depth},
                )
                datum_type = LocatorType.POINT
            elif operation.operation_type == "circular_pocket":
                diameter = self._positive(params, "diameter")
                cutter = (
                    cq.Workplane("XY").workplane(offset=start_z).center(x, y)
                    .circle(diameter / 2.0).extrude(extrusion).val()
                )
                feature_locator = GeometryLocator(
                    locator_type=LocatorType.CYLINDRICAL_SURFACE,
                    data={**common, "feature_type": "CIRCULAR_POCKET", "diameter": diameter},
                )
                datum_type = LocatorType.POINT
            else:
                length = self._positive(params, "length")
                width = self._positive(params, "width")
                if length < width:
                    raise ValueError("slot length must be greater than or equal to slot width")
                angle = float(params["angle_deg"])
                if angle not in {0.0, 90.0}:
                    raise ValueError("slot angle_deg must be 0 or 90")
                cutter = (
                    cq.Workplane("XY").workplane(offset=start_z).center(x, y)
                    .slot2D(length, width, angle).extrude(extrusion).val()
                )
                feature_locator = GeometryLocator(
                    locator_type=LocatorType.SUBTRACTIVE_FEATURE,
                    data={
                        **common, "feature_type": "SLOT", "length": length,
                        "width": width, "angle_deg": angle,
                    },
                )
                datum_type = LocatorType.AXIS
            shape = model.cut(cutter)
            self._register_outputs(operation, registry, {
                FeatureKind.FEATURE: feature_locator,
                FeatureKind.DATUM_POINT: GeometryLocator(
                    locator_type=datum_type,
                    data={"origin": [x, y, bbox.zmax], "direction": [0.0, 0.0, 1.0]},
                ),
                FeatureKind.DATUM_AXIS: GeometryLocator(
                    locator_type=datum_type,
                    data={"origin": [x, y, bbox.zmax], "direction": [0.0, 0.0, 1.0]},
                ),
            })
            return shape
        if operation.operation_type in {"rect_boss", "cyl_boss", "standoff"}:
            x, y = self._finite_coordinates(params)
            height = self._positive(params, "height")
            base = registry.require("main_body").locator.data
            base_height = float(base["height"])
            common = {
                "origin": [x, y, base_height], "direction": [0.0, 0.0, 1.0],
                "height": height, "base_height": base_height,
            }
            if operation.operation_type == "rect_boss":
                width = self._positive(params, "width")
                depth = self._positive(params, "depth")
                additive = (
                    cq.Workplane("XY").workplane(offset=base_height).center(x, y)
                    .rect(width, depth).extrude(height).val()
                )
                feature_data = {**common, "feature_type": "RECTANGULAR_BOSS", "width": width, "depth": depth}
            else:
                outer_name = "diameter" if operation.operation_type == "cyl_boss" else "outer_diameter"
                diameter = self._positive(params, outer_name)
                additive = (
                    cq.Workplane("XY").workplane(offset=base_height).center(x, y)
                    .circle(diameter / 2.0).extrude(height).val()
                )
                feature_data = {
                    **common,
                    "feature_type": "CYLINDRICAL_BOSS" if operation.operation_type == "cyl_boss" else "STANDOFF",
                    "diameter": diameter,
                    "outer_diameter": diameter,
                }
                if operation.operation_type == "standoff" and params.get("hole_diameter") is not None:
                    hole_diameter = self._positive(params, "hole_diameter")
                    if hole_diameter >= diameter:
                        raise ValueError("hole_diameter must be smaller than outer_diameter")
                    cutter = (
                        cq.Workplane("XY").workplane(offset=base_height - 0.001).center(x, y)
                        .circle(hole_diameter / 2.0).extrude(height + 0.002).val()
                    )
                    additive = additive.cut(cutter)
                    feature_data["hole_diameter"] = hole_diameter
            shape = model.fuse(additive)
            if len(shape.Solids()) != 1:
                raise ValueError("additive feature did not fuse into one solid")
            self._register_additive_outputs(operation, registry, feature_data)
            return shape
        if operation.operation_type == "fillet":
            radius = self._positive(params, "radius")
            shape = cq.Workplane(obj=model).newObject(self._outer_vertical_edges(model, registry)).fillet(radius).val()
            self._register_outputs(operation, registry, {
                FeatureKind.FEATURE: GeometryLocator(
                    locator_type=LocatorType.EDGE_TREATMENT,
                    data={
                        "treatment": "FILLET",
                        "selection": "OUTER_VERTICAL_EDGES",
                        "value": radius,
                        "expected_count": 4,
                    },
                ),
            })
            return shape
        distance = self._positive(params, "distance")
        shape = cq.Workplane(obj=model).newObject(self._outer_vertical_edges(model, registry)).chamfer(distance).val()
        self._register_outputs(operation, registry, {
            FeatureKind.FEATURE: GeometryLocator(
                locator_type=LocatorType.EDGE_TREATMENT,
                data={
                    "treatment": "CHAMFER",
                    "selection": "OUTER_VERTICAL_EDGES",
                    "value": distance,
                    "expected_count": 4,
                },
            ),
        })
        return shape

    @staticmethod
    def _positive(params: dict[str, Any], name: str) -> float:
        value = float(params[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")
        return value

    @staticmethod
    def _finite_coordinates(params: dict[str, Any]) -> tuple[float, float]:
        x, y = float(params["x"]), float(params["y"])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("feature coordinates must be finite numbers")
        return x, y

    @staticmethod
    def _outer_vertical_edges(model: cq.Shape, registry: FeatureRegistry) -> list[cq.Edge]:
        base = registry.require("main_body").locator.data
        base_width, base_depth, base_height = float(base["width"]), float(base["depth"]), float(base["height"])
        tolerance = 1e-6
        selected: list[cq.Edge] = []
        for edge in model.Edges():
            edge_box = edge.BoundingBox()
            if edge.geomType() != "LINE" or edge_box.zlen <= tolerance:
                continue
            center = edge.Center()
            if (
                math.isclose(abs(center.x), base_width / 2.0, abs_tol=tolerance)
                and math.isclose(abs(center.y), base_depth / 2.0, abs_tol=tolerance)
                and math.isclose(edge_box.zlen, base_height, abs_tol=tolerance)
            ):
                selected.append(edge)
        if len(selected) != 4:
            raise ValueError(f"expected four outer vertical edges, found {len(selected)}")
        return selected

    @staticmethod
    def _register_additive_outputs(operation: Any, registry: FeatureRegistry, feature_data: dict[str, Any]) -> None:
        origin = feature_data["origin"]
        for output in operation.outputs:
            if output.semantic_role in {"rectangular_boss", "cylindrical_boss", "standoff"}:
                locator = GeometryLocator(locator_type=LocatorType.ADDITIVE_FEATURE, data=feature_data)
            elif output.semantic_role in {"rectangular_boss_center", "cylindrical_boss_axis", "standoff_axis"}:
                locator = GeometryLocator(
                    locator_type=LocatorType.POINT if output.kind == FeatureKind.DATUM_POINT else LocatorType.AXIS,
                    data={"origin": origin, "direction": [0.0, 0.0, 1.0]},
                )
            elif output.semantic_role == "standoff_hole":
                locator = GeometryLocator(locator_type=LocatorType.ADDITIVE_FEATURE, data={
                    **feature_data, "feature_type": "STANDOFF_HOLE", "diameter": feature_data["hole_diameter"],
                })
            elif output.semantic_role == "standoff_hole_axis":
                locator = GeometryLocator(locator_type=LocatorType.AXIS, data={"origin": origin, "direction": [0.0, 0.0, 1.0]})
            else:
                raise _PlanRejected("INVALID_OUTPUT_KIND", f"unknown additive semantic role: {output.semantic_role}", operation.operation_id)
            registry.register(FeatureRecord(
                feature_id=output.feature_id, kind=output.kind,
                produced_by_operation_id=operation.operation_id,
                semantic_role=output.semantic_role, locator=locator,
            ))

    @staticmethod
    def _register_outputs(operation: Any, registry: FeatureRegistry, locators: dict[FeatureKind, GeometryLocator]) -> None:
        for output in operation.outputs:
            if output.kind not in locators:
                raise _PlanRejected("INVALID_OUTPUT_KIND", f"{operation.operation_type} cannot produce {output.kind.value}", operation.operation_id)
            registry.register(FeatureRecord(
                feature_id=output.feature_id,
                kind=output.kind,
                produced_by_operation_id=operation.operation_id,
                semantic_role=output.semantic_role,
                locator=locators[output.kind],
            ))

    @staticmethod
    def _error_result(plan: CADPlan, code: str, category: CADExecutionErrorCategory, message: str, operation_id: str | None) -> CADResult:
        return CADResult(
            design_id=plan.design_id,
            spec_version=plan.spec_version,
            revision_id=plan.revision_id,
            status=CADStatus.ERROR,
            error=CADExecutionError(code=code, category=category, operation_id=operation_id, message=message, retryable=False),
        )

    @staticmethod
    def export(result: CADResult, directory: Path) -> dict[str, Path]:
        if result.status != CADStatus.SUCCESS or result.model_handle is None:
            raise ValueError("only successful CAD results can be exported")
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"{result.design_id}_{result.revision_id}"
        step_path, stl_path = directory / f"{stem}.step", directory / f"{stem}.stl"
        cq.exporters.export(result.model_handle, str(step_path))
        cq.exporters.export(result.model_handle, str(stl_path), tolerance=0.01, angularTolerance=0.1)
        return {"STEP": step_path, "STL": stl_path}


class GeometryInspector:
    def __init__(self, result: CADResult):
        if result.status != CADStatus.SUCCESS or result.model_handle is None or result.feature_registry is None:
            raise ValueError("geometry inspection requires a successful CADResult")
        self.shape: cq.Shape = result.model_handle
        self.registry = result.feature_registry

    def model_valid(self) -> bool:
        return bool(self.shape.isValid())

    def solid_count(self) -> int:
        return len(self.shape.Solids())

    def extent(self, axis: str) -> float:
        bbox = self.shape.BoundingBox()
        return float({"X": bbox.xlen, "Y": bbox.ylen, "Z": bbox.zlen}[axis])

    def volume(self) -> float:
        return float(self.shape.Volume())

    def distance(self, first_feature_id: str, second_feature_id: str) -> float:
        first = self.registry.require(first_feature_id).locator
        second = self.registry.require(second_feature_id).locator
        if first.locator_type not in (LocatorType.AXIS, LocatorType.POINT) or second.locator_type not in (LocatorType.AXIS, LocatorType.POINT):
            raise ValueError("distance requires axis or point semantic locators")
        a, b = first.data["origin"], second.data["origin"]
        return math.dist(a, b)

    def diameter(self, feature_id: str) -> float:
        locator = self.registry.require(feature_id).locator
        if locator.locator_type not in {LocatorType.CYLINDRICAL_SURFACE, LocatorType.ADDITIVE_FEATURE}:
            raise ValueError("diameter requires a cylindrical semantic feature")
        expected = locator.data["origin"]
        candidates: list[tuple[float, float]] = []
        for face in self.shape.Faces():
            if face.geomType() != "CYLINDER":
                continue
            cylinder = face._geomAdaptor().Cylinder()
            location = cylinder.Axis().Location()
            distance_xy = math.hypot(location.X() - expected[0], location.Y() - expected[1])
            candidates.append((distance_xy, float(cylinder.Radius()) * 2.0))
        if not candidates or min(candidates)[0] > 1e-6:
            raise ValueError(f"no cylindrical geometry found for feature_id {feature_id}")
        colocated = [diameter for distance, diameter in candidates if distance <= 1e-6]
        expected_diameter = locator.data.get("diameter")
        return min(colocated, key=lambda value: abs(value - float(expected_diameter))) if expected_diameter else min(candidates)[1]

    def feature_exists(self, feature_id: str) -> bool:
        return feature_id in self.registry.features

    def hole_count(self) -> int:
        return sum(
            record.locator.locator_type == LocatorType.CYLINDRICAL_SURFACE
            and record.semantic_role == "through_hole"
            for record in self.registry.features.values()
        )

    def feature_count(self, semantic_role: str) -> int:
        return sum(record.semantic_role == semantic_role for record in self.registry.features.values())

    def position(self, feature_id: str, axis: str) -> float:
        locator = self.registry.require(feature_id).locator
        if locator.locator_type not in {
            LocatorType.AXIS, LocatorType.POINT, LocatorType.CYLINDRICAL_SURFACE,
            LocatorType.SUBTRACTIVE_FEATURE,
            LocatorType.ADDITIVE_FEATURE,
        }:
            raise ValueError("position requires a point, axis, or cylindrical semantic locator")
        coordinate = {"X": 0, "Y": 1}[axis]
        return float(locator.data["origin"][coordinate])

    def feature_width(self, feature_id: str) -> float:
        return self._feature_value(feature_id, "width", {"RECTANGULAR_POCKET", "RECTANGULAR_BOSS"})

    def feature_depth(self, feature_id: str) -> float:
        return self._feature_value(feature_id, "depth", {"RECTANGULAR_POCKET", "RECTANGULAR_BOSS"})

    def feature_height(self, feature_id: str) -> float:
        return self._additive_value(feature_id, "height", {"RECTANGULAR_BOSS", "CYLINDRICAL_BOSS", "STANDOFF"})

    def base_height(self, feature_id: str) -> float:
        locator = self.registry.require(feature_id).locator
        if locator.locator_type != LocatorType.BODY:
            raise ValueError("base height requires the main body")
        height = float(locator.data["height"])
        width, depth = float(locator.data["width"]), float(locator.data["depth"])
        probes = [
            cq.Vector(x * width, y * depth, height / 2.0)
            for x in (-0.4, 0.0, 0.4) for y in (-0.4, 0.0, 0.4)
        ]
        if not any(self.shape.isInside(point, 1e-7) for point in probes):
            raise ValueError("main body geometry was not found")
        candidates = []
        for face in self.shape.Faces():
            if face.geomType() != "PLANE":
                continue
            normal = face.normalAt()
            box = face.BoundingBox()
            if abs(normal.z) <= 1e-6 and abs(box.zmin) <= 1e-6 and box.zlen > 1e-6:
                candidates.append(float(box.zlen))
        if not candidates:
            raise ValueError("base side geometry was not found")
        return max(candidates)

    def outer_diameter(self, feature_id: str) -> float:
        return self._additive_value(feature_id, "outer_diameter", {"STANDOFF"}, cylindrical=True)

    def inner_diameter(self, feature_id: str) -> float:
        return self._additive_value(feature_id, "diameter", {"STANDOFF_HOLE"}, cylindrical=True, void=True)

    def cut_depth(self, feature_id: str) -> float:
        locator = self.registry.require(feature_id).locator
        if locator.locator_type not in {LocatorType.SUBTRACTIVE_FEATURE, LocatorType.CYLINDRICAL_SURFACE}:
            raise ValueError("cut depth requires a subtractive semantic feature")
        if locator.data.get("through"):
            raise ValueError("through features do not have a finite cut depth")
        self._verify_feature_void(locator)
        return float(locator.data["cut_depth"])

    def through(self, feature_id: str) -> bool:
        locator = self.registry.require(feature_id).locator
        if locator.locator_type not in {LocatorType.SUBTRACTIVE_FEATURE, LocatorType.CYLINDRICAL_SURFACE}:
            raise ValueError("through requires a subtractive semantic feature")
        self._verify_feature_void(locator)
        return bool(locator.data["through"])

    def slot_length(self, feature_id: str) -> float:
        return self._subtractive_value(feature_id, "length", {"SLOT"})

    def slot_width(self, feature_id: str) -> float:
        return self._subtractive_value(feature_id, "width", {"SLOT"})

    def orientation_angle(self, feature_id: str) -> float:
        return self._subtractive_value(feature_id, "angle_deg", {"SLOT"})

    def _subtractive_value(self, feature_id: str, key: str, feature_types: set[str]) -> float:
        locator = self.registry.require(feature_id).locator
        if locator.locator_type != LocatorType.SUBTRACTIVE_FEATURE or locator.data.get("feature_type") not in feature_types:
            raise ValueError(f"{key} requires a matching subtractive semantic feature")
        self._verify_feature_void(locator)
        return float(locator.data[key])

    def _feature_value(self, feature_id: str, key: str, feature_types: set[str]) -> float:
        locator = self.registry.require(feature_id).locator
        if locator.locator_type == LocatorType.SUBTRACTIVE_FEATURE:
            return self._subtractive_value(feature_id, key, feature_types)
        return self._additive_value(feature_id, key, feature_types)

    def _additive_value(
        self, feature_id: str, key: str, feature_types: set[str], *,
        cylindrical: bool = False, void: bool = False,
    ) -> float:
        locator = self.registry.require(feature_id).locator
        if locator.locator_type != LocatorType.ADDITIVE_FEATURE or locator.data.get("feature_type") not in feature_types:
            raise ValueError(f"{key} requires a matching additive semantic feature")
        origin = locator.data["origin"]
        height = float(locator.data["height"])
        if void:
            probe = cq.Vector(float(origin[0]), float(origin[1]), float(origin[2]) + height / 2.0)
            if self.shape.isInside(probe, 1e-7):
                raise ValueError("registered standoff hole is not void")
        else:
            if cylindrical or locator.data.get("feature_type") in {"CYLINDRICAL_BOSS", "STANDOFF"}:
                radius = float(locator.data["outer_diameter"]) / 2.0
                probe = cq.Vector(float(origin[0]) + radius * 0.75, float(origin[1]), float(origin[2]) + height / 2.0)
            else:
                probe = cq.Vector(float(origin[0]), float(origin[1]), float(origin[2]) + height / 2.0)
            if not self.shape.isInside(probe, 1e-7):
                raise ValueError("registered additive feature is not solid")
        if cylindrical:
            self.diameter(feature_id)
        return float(locator.data[key])

    def _verify_feature_void(self, locator: GeometryLocator) -> None:
        origin = locator.data["origin"]
        depth = float(locator.data["base_height"] if locator.data["through"] else locator.data["cut_depth"])
        probe = cq.Vector(float(origin[0]), float(origin[1]), float(origin[2]) - depth / 2.0)
        if self.shape.isInside(probe, 1e-7):
            raise ValueError("registered subtractive feature does not correspond to a void in the model")

    def fillet_radius(self, feature_id: str) -> float:
        locator = self._edge_treatment(feature_id, "FILLET")
        radius = float(locator.data["value"])
        expected_count = int(locator.data["expected_count"])
        matching = 0
        for face in self.shape.Faces():
            if face.geomType() != "CYLINDER":
                continue
            cylinder = face._geomAdaptor().Cylinder()
            direction = cylinder.Axis().Direction()
            if abs(abs(direction.Z()) - 1.0) <= 1e-6 and math.isclose(
                float(cylinder.Radius()), radius, abs_tol=1e-6
            ):
                matching += 1
        if matching < expected_count:
            raise ValueError("expected outer vertical fillet geometry was not found")
        return radius

    def chamfer_distance(self, feature_id: str) -> float:
        locator = self._edge_treatment(feature_id, "CHAMFER")
        expected = float(locator.data["value"])
        expected_count = int(locator.data["expected_count"])
        candidates: list[float] = []
        for face in self.shape.Faces():
            if face.geomType() != "PLANE":
                continue
            normal = face.normalAt()
            if abs(normal.z) > 1e-6 or abs(normal.x) < 1e-6 or abs(normal.y) < 1e-6:
                continue
            horizontal = [
                edge.Length() for edge in face.Edges()
                if abs(edge.BoundingBox().zlen) <= 1e-6
            ]
            if horizontal:
                candidates.append(float(max(horizontal)) / math.sqrt(2.0))
        matching = sum(math.isclose(value, expected, abs_tol=1e-6) for value in candidates)
        if matching < expected_count:
            raise ValueError("expected outer vertical chamfer geometry was not found")
        return expected

    def _edge_treatment(self, feature_id: str, treatment: str) -> GeometryLocator:
        locator = self.registry.require(feature_id).locator
        if (
            locator.locator_type != LocatorType.EDGE_TREATMENT
            or locator.data.get("treatment") != treatment
            or locator.data.get("selection") != "OUTER_VERTICAL_EDGES"
        ):
            raise ValueError(f"{feature_id} is not the expected {treatment.lower()} feature")
        return locator
