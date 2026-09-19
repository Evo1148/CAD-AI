from pathlib import Path

from cad_ai.cad import CADEngine, CADStatus, GeometryInspector
from cad_ai.cases import block_case, error_case, semantic_case
from cad_ai.contracts import CADOperation, CADPlan, FeatureKind, OperationOutput


def test_box_geometry_and_registry():
    _, plan = block_case()
    result = CADEngine().execute(plan)
    assert result.status == CADStatus.SUCCESS
    assert result.feature_registry.require("main_body").produced_by_operation_id == "op-main-body"
    inspector = GeometryInspector(result)
    assert inspector.model_valid() and inspector.solid_count() == 1
    assert (inspector.extent("X"), inspector.extent("Y"), inspector.extent("Z")) == (100.0, 60.0, 4.0)
    assert abs(inspector.volume() - 24000.0) < 1e-6


def test_cylinder_operation_creates_real_solid():
    plan = CADPlan(design_id="cylinder", spec_version="1", revision_id="R01", operations=[
        CADOperation(operation_id="op-cylinder", operation_type="cylinder", params={"radius": 5, "height": 12},
                     outputs=[OperationOutput(feature_id="main_body", kind=FeatureKind.BODY)])
    ])
    result = CADEngine().execute(plan)
    inspector = GeometryInspector(result)
    assert result.status == CADStatus.SUCCESS
    assert inspector.solid_count() == 1
    assert abs(inspector.volume() - (3.141592653589793 * 25 * 12)) < 1e-6


def test_semantic_holes_measure_real_cylinders_and_axis_distance():
    _, plan = semantic_case()
    result = CADEngine().execute(plan)
    inspector = GeometryInspector(result)
    assert abs(inspector.diameter("mount_hole_left") - 6.0) < 1e-6
    assert abs(inspector.distance("mount_hole_left_axis", "mount_hole_right_axis") - 60.0) < 1e-6


def test_ct006_unknown_controlled_operation_is_structured_error():
    plan = CADPlan(design_id="d", spec_version="1", revision_id="R01", operations=[
        CADOperation(operation_id="bad", operation_type="magic", outputs=[OperationOutput(feature_id="x", kind=FeatureKind.BODY)])
    ])
    result = CADEngine().execute(plan)
    assert result.status == CADStatus.ERROR
    assert result.error.code == "UNKNOWN_CONTROLLED_OPERATION"


def test_ct008_nonexistent_feature_reference_is_rejected():
    plan = CADPlan(design_id="d", spec_version="1", revision_id="R01", operations=[
        CADOperation(operation_id="hole", operation_type="hole", inputs=["missing"], params={"diameter": 2, "x": 0, "y": 0},
                     outputs=[OperationOutput(feature_id="hole-1", kind=FeatureKind.FEATURE)])
    ])
    result = CADEngine().execute(plan)
    assert result.error.code == "UNKNOWN_FEATURE_REFERENCE"


def test_case04_cad_error_is_not_validation_fail():
    _, plan = error_case()
    result = CADEngine().execute(plan)
    assert result.status == CADStatus.ERROR
    assert result.error.code == "INVALID_OPERATION_PARAMETERS"


def test_exports_real_step_and_stl(tmp_path: Path):
    _, plan = block_case()
    result = CADEngine().execute(plan)
    artifacts = CADEngine.export(result, tmp_path)
    assert artifacts["STEP"].stat().st_size > 100
    assert artifacts["STL"].stat().st_size > 100
