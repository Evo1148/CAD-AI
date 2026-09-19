from cad_ai.cad import CADStatus
from cad_ai.cases import semantic_case
from cad_ai.pipeline import run_pipeline
from cad_ai.validation import ReportStatus


def test_end_to_end_semantic_case(tmp_path):
    spec, plan = semantic_case()
    result = run_pipeline(spec, plan, tmp_path)
    assert result.cad_result.status == CADStatus.SUCCESS
    assert result.validation_report.status == ReportStatus.PASS
    assert set(result.artifacts) == {"STEP", "STL"}
    assert all(path.exists() for path in result.artifacts.values())


def test_ct012_013_declared_outputs_and_identifiers_are_coherent(tmp_path):
    spec, plan = semantic_case()
    result = run_pipeline(spec, plan, tmp_path)
    declared = {out.feature_id for op in plan.operations for out in op.outputs}
    assert declared == set(result.cad_result.feature_registry.features)
    assert (result.cad_result.design_id, result.cad_result.spec_version, result.cad_result.revision_id) == (
        spec.design_id, spec.spec_version, plan.revision_id)
    assert result.validation_report.revision_id == plan.revision_id

