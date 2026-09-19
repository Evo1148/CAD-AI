import pytest
from pydantic import ValidationError

from cad_ai.cad import CADExecutionError, CADExecutionErrorCategory, CADResult, CADStatus, FeatureRegistry, GeometryLocator, LocatorType
from cad_ai.cases import block_case
from cad_ai.contracts import (
    CADOperation, CADPlan, Comparison, Constraint, DesignSpec, FeatureKind,
    Measurement, MeasurementType, OperationOutput, Origin, Priority, Unknown, ensure_spec_plannable,
)
from cad_ai.validation import CheckResult, CheckStatus, ReportStatus, ValidationReport


def test_ct001_design_spec_round_trip():
    spec, _ = block_case()
    assert DesignSpec.model_validate_json(spec.model_dump_json()) == spec


def test_ct002_reject_contract_version():
    spec, _ = block_case()
    data = spec.model_dump(mode="json")
    data["contract_version"] = "0.2"
    with pytest.raises(ValidationError):
        DesignSpec.model_validate(data)


@pytest.mark.parametrize("kwargs", [
    {"comparison": Comparison.EQ, "target": 1, "minimum": 0},
    {"comparison": Comparison.BETWEEN, "minimum": 0},
    {"comparison": Comparison.EQ, "target": 1, "tolerance": -1},
])
def test_ct003_reject_invalid_comparison(kwargs):
    with pytest.raises(ValidationError):
        Constraint(constraint_id="c", priority=Priority.HARD, origin=Origin.USER,
                   measurement=Measurement(type=MeasurementType.EXTENT_X), **kwargs)


def test_ct004_blocking_unknown_stops_planning():
    spec, _ = block_case()
    spec.unknowns.append(Unknown(unknown_id="u1", description="missing dimension", blocking=True))
    with pytest.raises(ValueError, match="blocking unknowns"):
        ensure_spec_plannable(spec)


def test_ct005_reject_unresolved_plan_parameter():
    with pytest.raises(ValidationError, match="fully resolved"):
        CADOperation(operation_id="op", operation_type="box", params={"x": "$width"},
                     outputs=[OperationOutput(feature_id="body", kind=FeatureKind.BODY)])


def test_ct007_reject_duplicate_feature_id():
    operations = [
        CADOperation(operation_id=f"op-{index}", operation_type="box", params={"x": 1, "y": 1, "z": 1},
                     outputs=[OperationOutput(feature_id="same-feature", kind=FeatureKind.BODY)])
        for index in range(2)
    ]
    with pytest.raises(ValidationError, match="globally unique"):
        CADPlan(design_id="d", spec_version="1", revision_id="R01", operations=operations)


def test_ct009_010_success_requires_model_and_registry():
    with pytest.raises(ValidationError):
        CADResult(design_id="d", spec_version="1", revision_id="R01", status=CADStatus.SUCCESS,
                  model_reference="memory://d/R01", feature_registry=FeatureRegistry())
    with pytest.raises(ValidationError):
        CADResult(design_id="d", spec_version="1", revision_id="R01", status=CADStatus.SUCCESS,
                  model_reference="memory://d/R01", model_handle=object())


def test_ct011_error_requires_structured_error():
    with pytest.raises(ValidationError):
        CADResult(design_id="d", spec_version="1", revision_id="R01", status=CADStatus.ERROR)
    result = CADResult(design_id="d", spec_version="1", revision_id="R01", status=CADStatus.ERROR,
                       error=CADExecutionError(code="X", category=CADExecutionErrorCategory.PLAN_REJECTED, message="x"))
    assert result.error.code == "X"


def test_ct018_report_pass_requires_complete():
    with pytest.raises(ValidationError):
        ValidationReport(design_id="d", spec_version="1", revision_id="R01", status=ReportStatus.PASS, complete=False, checks=[])


def test_ct019_parent_revision_semantics():
    op = CADOperation(operation_id="op", operation_type="box", params={"x": 1},
                      outputs=[OperationOutput(feature_id="body", kind=FeatureKind.BODY)])
    with pytest.raises(ValidationError):
        CADPlan(design_id="d", spec_version="1", revision_id="R02", operations=[op])
    plan = CADPlan(design_id="d", spec_version="1", revision_id="R02", parent_revision_id="R01", operations=[op])
    assert plan.parent_revision_id == "R01"


def test_ct020_locators_reject_topology_indices():
    with pytest.raises(ValidationError):
        GeometryLocator(locator_type=LocatorType.BODY, data={"face_index": 7})
