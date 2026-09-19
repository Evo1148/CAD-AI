from cad_ai.cad import CADEngine
from cad_ai.cases import block_case
from cad_ai.contracts import Comparison, Constraint, Measurement, MeasurementType, Origin, Priority
from cad_ai.validation import CheckStatus, ReportStatus, Validator


def test_case01_passes_real_geometry():
    spec, plan = block_case("pass")
    report = Validator().validate(spec, CADEngine().execute(plan))
    assert report.status == ReportStatus.PASS and report.complete


def test_ct014_case02_hard_violation_blocks_and_fails():
    spec, plan = block_case("fail")
    report = Validator().validate(spec, CADEngine().execute(plan))
    width = next(c for c in report.checks if c.constraint_id == "width")
    assert width.actual == 98.0
    assert width.status == CheckStatus.FAIL and width.blocking
    assert report.status == ReportStatus.FAIL


def test_ct015_case05_soft_violation_warns_but_report_passes():
    spec, plan = block_case("soft")
    report = Validator().validate(spec, CADEngine().execute(plan))
    warning = next(c for c in report.checks if c.constraint_id == "preferred-volume")
    assert warning.status == CheckStatus.WARNING and not warning.blocking
    assert report.status == ReportStatus.PASS


def test_ct016_blocking_unsupported_makes_report_incomplete():
    spec, plan = block_case()
    spec.constraints.append(Constraint(
        constraint_id="angle", priority=Priority.HARD, origin=Origin.USER,
        measurement=Measurement(type=MeasurementType.ANGLE), comparison=Comparison.EQ, target=90,
    ))
    report = Validator().validate(spec, CADEngine().execute(plan))
    assert report.status == ReportStatus.INCOMPLETE and not report.complete
    assert report.checks[-1].status == CheckStatus.UNSUPPORTED


def test_ct017_blocking_measurement_error_makes_report_incomplete():
    spec, plan = block_case()
    spec.constraints.append(Constraint(
        constraint_id="missing-hole", priority=Priority.HARD, origin=Origin.USER,
        measurement=Measurement(type=MeasurementType.DIAMETER, feature_id="no-such-hole"),
        comparison=Comparison.EQ, target=6,
    ))
    report = Validator().validate(spec, CADEngine().execute(plan))
    assert report.status == ReportStatus.INCOMPLETE
    assert report.checks[-1].status == CheckStatus.ERROR


def test_gte_lte_between_comparisons_on_real_geometry():
    spec, plan = block_case()
    spec.constraints.extend([
        Constraint(constraint_id="x-min", priority=Priority.HARD, origin=Origin.USER,
                   measurement=Measurement(type=MeasurementType.EXTENT_X), comparison=Comparison.GTE, minimum=99.9),
        Constraint(constraint_id="x-max", priority=Priority.HARD, origin=Origin.USER,
                   measurement=Measurement(type=MeasurementType.EXTENT_X), comparison=Comparison.LTE, maximum=100.1),
        Constraint(constraint_id="x-range", priority=Priority.HARD, origin=Origin.USER,
                   measurement=Measurement(type=MeasurementType.EXTENT_X), comparison=Comparison.BETWEEN,
                   minimum=99.9, maximum=100.1),
    ])
    report = Validator().validate(spec, CADEngine().execute(plan))
    assert report.status == ReportStatus.PASS
    assert all(check.status == CheckStatus.PASS for check in report.checks[-3:])
