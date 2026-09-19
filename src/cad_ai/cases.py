from __future__ import annotations

from .contracts import (
    CADOperation, CADPlan, Comparison, Constraint, DesignSpec, FeatureKind,
    Measurement, MeasurementType, OperationOutput, Origin, Priority, ensure_spec_plannable,
)


def _constraint(cid: str, metric: MeasurementType, target: float | bool, *, tolerance: float = 1e-6,
                priority: Priority = Priority.HARD, feature: str | None = None,
                reference: str | None = None) -> Constraint:
    return Constraint(
        constraint_id=cid, priority=priority, origin=Origin.USER,
        measurement=Measurement(type=metric, feature_id=feature, reference_feature_id=reference),
        comparison=Comparison.EQ, target=target, tolerance=tolerance,
    )


def block_case(case: str = "pass") -> tuple[DesignSpec, CADPlan]:
    spec = DesignSpec(
        design_id=f"case-{case}", spec_version="1.0", parameters={"width": 100.0, "depth": 60.0, "height": 4.0},
        constraints=[
            _constraint("valid", MeasurementType.MODEL_VALID, True, tolerance=0),
            _constraint("one-solid", MeasurementType.SOLID_COUNT, 1, tolerance=0),
            _constraint("width", MeasurementType.EXTENT_X, 100.0, tolerance=0.1),
            _constraint("depth", MeasurementType.EXTENT_Y, 60.0, tolerance=0.1),
            _constraint("height", MeasurementType.EXTENT_Z, 4.0, tolerance=0.1),
            _constraint("volume", MeasurementType.VOLUME, 24000.0, tolerance=0.1,
                        priority=Priority.SOFT if case == "soft" else Priority.HARD),
        ],
    )
    width = 98.0 if case == "fail" else 100.0
    if case == "soft":
        spec.constraints[-1] = _constraint("preferred-volume", MeasurementType.VOLUME, 23000.0, tolerance=0.1, priority=Priority.SOFT)
    ensure_spec_plannable(spec)
    plan = CADPlan(
        design_id=spec.design_id, spec_version=spec.spec_version, revision_id="R01",
        operations=[CADOperation(
            operation_id="op-main-body", operation_type="box", params={"x": width, "y": 60.0, "z": 4.0},
            outputs=[OperationOutput(feature_id="main_body", kind=FeatureKind.BODY, semantic_role="primary solid")],
        )],
    )
    return spec, plan


def semantic_case() -> tuple[DesignSpec, CADPlan]:
    spec = DesignSpec(
        design_id="case-semantic", spec_version="1.0", parameters={"hole_diameter": 6.0, "axis_distance": 60.0},
        constraints=[
            _constraint("valid", MeasurementType.MODEL_VALID, True, tolerance=0),
            _constraint("left-hole-diameter", MeasurementType.DIAMETER, 6.0, tolerance=0.01, feature="mount_hole_left"),
            _constraint("hole-axis-distance", MeasurementType.DISTANCE, 60.0, tolerance=0.01,
                        feature="mount_hole_left_axis", reference="mount_hole_right_axis"),
        ],
    )
    ensure_spec_plannable(spec)
    plan = CADPlan(
        design_id=spec.design_id, spec_version=spec.spec_version, revision_id="R01",
        operations=[
            CADOperation(operation_id="op-body", operation_type="box", params={"x": 100.0, "y": 40.0, "z": 5.0},
                         outputs=[OperationOutput(feature_id="main_body", kind=FeatureKind.BODY)]),
            CADOperation(operation_id="op-hole-left", operation_type="hole", inputs=["main_body"],
                         params={"x": -30.0, "y": 0.0, "diameter": 6.0}, outputs=[
                             OperationOutput(feature_id="mount_hole_left", kind=FeatureKind.FEATURE),
                             OperationOutput(feature_id="mount_hole_left_axis", kind=FeatureKind.DATUM_AXIS),
                         ]),
            CADOperation(operation_id="op-hole-right", operation_type="hole", inputs=["main_body"],
                         params={"x": 30.0, "y": 0.0, "diameter": 6.0}, outputs=[
                             OperationOutput(feature_id="mount_hole_right", kind=FeatureKind.FEATURE),
                             OperationOutput(feature_id="mount_hole_right_axis", kind=FeatureKind.DATUM_AXIS),
                         ]),
        ],
    )
    return spec, plan


def error_case() -> tuple[DesignSpec, CADPlan]:
    spec, _ = block_case("error")
    plan = CADPlan(
        design_id=spec.design_id, spec_version=spec.spec_version, revision_id="R01",
        operations=[CADOperation(operation_id="op-invalid", operation_type="box", params={"x": -1.0, "y": 10.0, "z": 2.0},
                                 outputs=[OperationOutput(feature_id="main_body", kind=FeatureKind.BODY)])],
    )
    return spec, plan


def repair_case() -> tuple[DesignSpec, CADPlan]:
    """R01 has one deliberate HARD width failure; all other checks pass."""

    spec = DesignSpec(
        design_id="D001", spec_version="1.0",
        parameters={"width": 100.0, "depth": 60.0, "height": 4.0, "hole_diameter": 4.0},
        constraints=[
            _constraint("C_VALID", MeasurementType.MODEL_VALID, True, tolerance=0),
            _constraint("C_SOLID_COUNT", MeasurementType.SOLID_COUNT, 1, tolerance=0),
            _constraint("C_WIDTH", MeasurementType.EXTENT_X, 100.0, tolerance=0.1),
            _constraint("C_DEPTH", MeasurementType.EXTENT_Y, 60.0, tolerance=0.1),
            _constraint("C_HEIGHT", MeasurementType.EXTENT_Z, 4.0, tolerance=0.1),
            _constraint("C_HOLE_DIAMETER", MeasurementType.DIAMETER, 4.0, tolerance=0.01, feature="mount_hole_left"),
            _constraint("C_HOLE_SPACING", MeasurementType.DISTANCE, 40.0, tolerance=0.01,
                        feature="mount_hole_left_axis", reference="mount_hole_right_axis"),
        ],
    )
    ensure_spec_plannable(spec)
    plan = CADPlan(
        design_id=spec.design_id, spec_version=spec.spec_version, revision_id="R01",
        operations=[
            CADOperation(operation_id="OP01", operation_type="box", params={"x": 99.5, "y": 60.0, "z": 4.0},
                         outputs=[OperationOutput(feature_id="main_body", kind=FeatureKind.BODY)]),
            CADOperation(operation_id="OP02", operation_type="hole", inputs=["main_body"],
                         params={"x": -20.0, "y": 0.0, "diameter": 4.0}, outputs=[
                             OperationOutput(feature_id="mount_hole_left", kind=FeatureKind.FEATURE),
                             OperationOutput(feature_id="mount_hole_left_axis", kind=FeatureKind.DATUM_AXIS),
                         ]),
            CADOperation(operation_id="OP03", operation_type="hole", inputs=["main_body"],
                         params={"x": 20.0, "y": 0.0, "diameter": 4.0}, outputs=[
                             OperationOutput(feature_id="mount_hole_right", kind=FeatureKind.FEATURE),
                             OperationOutput(feature_id="mount_hole_right_axis", kind=FeatureKind.DATUM_AXIS),
                         ]),
        ],
    )
    return spec, plan


def lab001_case() -> tuple[DesignSpec, CADPlan]:
    """LAB-001: one central hole with a deliberate R01 diameter defect."""

    spec = DesignSpec(
        design_id="LAB-001",
        spec_version="1.0",
        parameters={"width": 60.0, "depth": 40.0, "height": 4.0, "hole_diameter": 6.0},
        constraints=[
            _constraint("C_VALID", MeasurementType.MODEL_VALID, True, tolerance=0),
            _constraint("C_SOLID_COUNT", MeasurementType.SOLID_COUNT, 1, tolerance=0),
            _constraint("C_WIDTH", MeasurementType.EXTENT_X, 60.0, tolerance=0.1),
            _constraint("C_DEPTH", MeasurementType.EXTENT_Y, 40.0, tolerance=0.1),
            _constraint("C_HEIGHT", MeasurementType.EXTENT_Z, 4.0, tolerance=0.1),
            _constraint(
                "C_HOLE_DIAMETER",
                MeasurementType.DIAMETER,
                6.0,
                tolerance=0.01,
                feature="mount_hole",
            ),
        ],
    )
    ensure_spec_plannable(spec)
    plan = CADPlan(
        design_id=spec.design_id,
        spec_version=spec.spec_version,
        revision_id="R01",
        operations=[
            CADOperation(
                operation_id="OP01",
                operation_type="box",
                params={"x": 60.0, "y": 40.0, "z": 4.0},
                outputs=[OperationOutput(feature_id="main_body", kind=FeatureKind.BODY)],
            ),
            CADOperation(
                operation_id="OP02",
                operation_type="hole",
                inputs=["main_body"],
                params={"x": 0.0, "y": 0.0, "diameter": 5.4},
                outputs=[
                    OperationOutput(feature_id="mount_hole", kind=FeatureKind.FEATURE),
                    OperationOutput(feature_id="mount_hole_axis", kind=FeatureKind.DATUM_AXIS),
                ],
            ),
        ],
    )
    return spec, plan


CASES = {"pass": lambda: block_case("pass"), "fail": lambda: block_case("fail"), "soft": lambda: block_case("soft"),
         "semantic": semantic_case, "error": error_case}
