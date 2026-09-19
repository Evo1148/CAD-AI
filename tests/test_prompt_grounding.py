import pytest

from cad_ai.capability_v02 import (
    DeterministicPlateIntentGate,
    ExtractedCircularPocketFacts,
    ExtractedHoleFacts,
    ExtractedLinearHolePatternFacts,
    ExtractedPlateFacts,
    ExtractedRectangularPocketFacts,
    ExtractedSlotFacts,
)
from cad_ai.planning import ScriptedFakeBackend
from cad_ai.prompt_grounding import (
    DeterministicExtractionCoverage,
    DeterministicFactAssembler,
    DeterministicPromptGrounder,
    ExtractionCoverageStatus,
    ExtractionMode,
    FactGroundingValidator,
    GroundingViolationCode,
    PromptFeatureType,
    residual_facts_schema,
)
from cad_ai.v02 import CASE_PROMPTS, V02Status, run_capability_case


class _FailIfCalledBackend:
    backend_type = "fail-if-called"
    model_id = "must-not-run"

    def __init__(self):
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        raise AssertionError("LLM backend must not be invoked for COMPLETE grounding")


def _codes(result):
    return {item.code for item in result.violations}


@pytest.mark.parametrize(
    "phrase",
    ["thread-free", "without thread", "without threads", "no thread", "no threads", "non-threaded", "unthreaded"],
)
def test_explicit_thread_negations_are_not_requested_features(phrase):
    evidence = DeterministicPromptGrounder().ground(
        f"Create a 75 x 45 x 5 mm {phrase} plate with one centered 5 mm through hole."
    )
    assert "thread" not in evidence.unsupported_feature_terms


def test_positive_thread_request_remains_unsupported():
    evidence = DeterministicPromptGrounder().ground(
        "Create a 75 x 45 x 5 mm plate with a threaded hole."
    )
    assert evidence.unsupported_feature_terms == ["thread"]


def test_negated_thread_does_not_hide_positive_shell_request():
    evidence = DeterministicPromptGrounder().ground(
        "Create a 75 x 45 x 5 mm block without threads, but with a shell."
    )
    assert evidence.unsupported_feature_terms == ["shell"]


def test_no_fillet_is_not_a_positive_fillet_request():
    evidence = DeterministicPromptGrounder().ground(
        "Create a 75 x 45 x 5 mm plate with no fillet."
    )
    assert PromptFeatureType.FILLET not in evidence.requested_feature_types
    assert evidence.explicit_fillet_radius is None


def test_positive_fillet_request_is_unchanged():
    evidence = DeterministicPromptGrounder().ground(
        "Create a 75 x 45 x 5 mm plate with a 2 mm fillet."
    )
    assert PromptFeatureType.FILLET in evidence.requested_feature_types
    assert evidence.explicit_fillet_radius == 2.0


def test_prompt_grounder_extracts_the_five_real_smoke_evidence():
    grounder = DeterministicPromptGrounder()

    cp202 = grounder.ground(CASE_PROMPTS["CP2-02"])
    assert cp202.base_dimensions == (100.0, 60.0, 10.0)
    assert cp202.requested_feature_types == [PromptFeatureType.RECTANGULAR_POCKET]
    assert cp202.rectangular_features[0].model_dump() == {
        "feature_type": "RECTANGULAR_POCKET", "width": 40.0, "depth": 20.0,
        "centered": True, "x": None, "y": None, "cut_depth": 4.0, "through": False,
    }

    cp204 = grounder.ground(CASE_PROMPTS["CP2-04"])
    assert cp204.circular_pockets[0].model_dump() == {
        "diameter": 30.0, "centered": True, "x": None, "y": None, "cut_depth": 5.0,
    }

    cp205 = grounder.ground(CASE_PROMPTS["CP2-05"])
    assert cp205.slots[0].model_dump() == {
        "length": 30.0, "width": 8.0, "centered": True,
        "x": None, "y": None, "angle_deg": 0.0, "through": True, "cut_depth": None,
    }

    cp206 = grounder.ground(CASE_PROMPTS["CP2-06"])
    assert cp206.linear_hole_patterns[0].model_dump() == {
        "diameter": 5.0, "count": 4, "spacing": 20.0, "axis": "X",
        "anchor_x": 0.0, "anchor_y": 0.0, "anchor_mode": "CENTER",
    }

    cp207 = grounder.ground(CASE_PROMPTS["CP2-07"])
    assert cp207.hole_groups[0].model_dump() == {
        "diameter": 5.0,
        "positions": [{"x": -40.0, "y": 20.0}, {"x": 40.0, "y": 20.0}],
    }
    assert cp207.explicit_fillet_radius == 3.0


@pytest.mark.parametrize(
    ("case_id", "facts", "expected_code"),
    [
        (
            "CP2-02",
            ExtractedPlateFacts(
                width=100.0, depth=60.0, height=10.0,
                unsupported_features=["centered rectangular pocket 40 x 20 mm and 4 mm deep"],
            ),
            GroundingViolationCode.SUPPORTED_FEATURE_MISCLASSIFIED,
        ),
        (
            "CP2-04",
            ExtractedPlateFacts(
                width=60.0, depth=60.0, height=12.0,
                unsupported_features=["centered 30 mm diameter circular pocket 5 mm deep"],
            ),
            GroundingViolationCode.SUPPORTED_FEATURE_MISCLASSIFIED,
        ),
        (
            "CP2-05",
            ExtractedPlateFacts(
                width=80.0, depth=40.0, height=6.0,
                unsupported_features=["centered horizontal through slot 30 x 8 mm"],
            ),
            GroundingViolationCode.SUPPORTED_FEATURE_MISCLASSIFIED,
        ),
        (
            "CP2-06",
            ExtractedPlateFacts(
                width=100.0, depth=40.0, height=6.0,
                holes=[
                    ExtractedHoleFacts(diameter=5.0, x=-15.0, y=0.0),
                    ExtractedHoleFacts(diameter=5.0, x=0.0, y=0.0),
                    ExtractedHoleFacts(diameter=5.0, x=15.0, y=0.0),
                    ExtractedHoleFacts(diameter=5.0, x=30.0, y=0.0),
                ],
            ),
            GroundingViolationCode.PATTERN_EXPANDED_BY_MODEL,
        ),
        (
            "CP2-07",
            ExtractedPlateFacts(
                width=120.0, depth=70.0, height=8.0,
                holes=[
                    ExtractedHoleFacts(diameter=None, x=-40.0, y=20.0),
                    ExtractedHoleFacts(diameter=None, x=40.0, y=20.0),
                ],
                rectangular_pockets=[ExtractedRectangularPocketFacts(
                    width=36.0, depth=18.0, centered=True, cut_depth=3.0,
                )],
                fillet_radius=3.0,
                unsupported_features=["Fillet on outer vertical edges"],
            ),
            GroundingViolationCode.SUPPORTED_FEATURE_MISCLASSIFIED,
        ),
    ],
)
def test_real_qwen_smoke_failures_stop_as_grounding_errors(
    case_id: str,
    facts: ExtractedPlateFacts,
    expected_code: GroundingViolationCode,
):
    evidence = DeterministicPromptGrounder().ground(CASE_PROMPTS[case_id])
    grounding = FactGroundingValidator().validate(evidence, facts)

    assert grounding.status == "GROUNDING_ERROR"
    assert expected_code in _codes(grounding)
    assert grounding.grounded_facts is None


def test_shared_hole_diameter_is_safely_normalized_without_llm_repetition():
    prompt = CASE_PROMPTS["CP2-07"]
    evidence = DeterministicPromptGrounder().ground(prompt)
    facts = ExtractedPlateFacts(
        width=120.0, depth=70.0, height=8.0,
        holes=[
            ExtractedHoleFacts(diameter=None, x=-40.0, y=20.0),
            ExtractedHoleFacts(diameter=None, x=40.0, y=20.0),
        ],
        rectangular_pockets=[ExtractedRectangularPocketFacts(
            width=36.0, depth=18.0, centered=True, cut_depth=3.0,
        )],
        fillet_radius=3.0,
    )

    result = FactGroundingValidator().validate(evidence, facts)

    assert result.status == "PASS"
    assert [hole.diameter for hole in result.grounded_facts.holes] == [5.0, 5.0]
    assert [hole.diameter for hole in facts.holes] == [None, None]


def test_hole_group_contract_expands_only_after_grounding_and_gate():
    evidence = DeterministicPromptGrounder().ground(CASE_PROMPTS["CASE-02"])
    facts = ExtractedPlateFacts(
        width=100.0, depth=60.0, height=4.0,
        hole_groups=[],
        holes=[
            ExtractedHoleFacts(diameter=None, x=-40.0, y=-20.0),
            ExtractedHoleFacts(diameter=None, x=40.0, y=-20.0),
            ExtractedHoleFacts(diameter=None, x=-40.0, y=20.0),
            ExtractedHoleFacts(diameter=None, x=40.0, y=20.0),
        ],
    )
    grounding = FactGroundingValidator().validate(evidence, facts)
    assert grounding.status == "PASS"
    response = DeterministicPlateIntentGate().evaluate(grounding.grounded_facts)
    assert response.root.status == "INTENT"
    assert {(hole.diameter, hole.x, hole.y) for hole in response.root.intent.holes} == {
        (5.0, -40.0, -20.0), (5.0, 40.0, -20.0),
        (5.0, -40.0, 20.0), (5.0, 40.0, 20.0),
    }


def test_pattern_must_remain_pattern_then_expands_to_exact_coordinates():
    evidence = DeterministicPromptGrounder().ground(CASE_PROMPTS["CP2-06"])
    facts = ExtractedPlateFacts(
        width=100.0, depth=40.0, height=6.0,
        linear_hole_patterns=[ExtractedLinearHolePatternFacts(
            diameter=5.0, count=4, spacing=20.0, axis="X",
            anchor_x=0.0, anchor_y=0.0, anchor_mode="CENTER",
        )],
    )
    grounding = FactGroundingValidator().validate(evidence, facts)
    assert grounding.status == "PASS"
    response = DeterministicPlateIntentGate().evaluate(grounding.grounded_facts)
    assert response.root.status == "INTENT"
    assert [(hole.x, hole.y) for hole in response.root.intent.holes] == [
        (-30.0, 0.0), (-10.0, 0.0), (10.0, 0.0), (30.0, 0.0),
    ]
    assert (-15.0, 0.0) not in [(hole.x, hole.y) for hole in response.root.intent.holes]


@pytest.mark.parametrize(
    ("case_id", "facts"),
    [
        (
            "CP2-02",
            ExtractedPlateFacts(
                width=100.0, depth=60.0, height=10.0,
                rectangular_pockets=[ExtractedRectangularPocketFacts(
                    width=40.0, depth=20.0, centered=True, cut_depth=4.0,
                )],
            ),
        ),
        (
            "CP2-04",
            ExtractedPlateFacts(
                width=60.0, depth=60.0, height=12.0,
                circular_pockets=[ExtractedCircularPocketFacts(
                    diameter=30.0, centered=True, cut_depth=5.0,
                )],
            ),
        ),
        (
            "CP2-05",
            ExtractedPlateFacts(
                width=80.0, depth=40.0, height=6.0,
                slots=[ExtractedSlotFacts(
                    length=30.0, width=8.0, centered=True, angle_deg=0.0, through=True,
                )],
            ),
        ),
    ],
)
def test_supported_feature_facts_pass_grounding(case_id, facts):
    evidence = DeterministicPromptGrounder().ground(CASE_PROMPTS[case_id])
    result = FactGroundingValidator().validate(evidence, facts)
    assert result.status == "PASS"
    assert result.violations == []


def test_user_unsupported_and_llm_extraction_error_are_distinct(tmp_path):
    unsupported = ExtractedPlateFacts(
        width=60.0, depth=40.0, height=8.0,
        unsupported_features=["thread"],
    )
    user_result = run_capability_case(
        "CP2-09", ScriptedFakeBackend([unsupported.model_dump_json()]), tmp_path / "unsupported",
    )
    extraction_error = ExtractedPlateFacts(
        width=80.0, depth=40.0, height=6.0,
        unsupported_features=["centered horizontal through slot 30 x 8 mm"],
    )
    assert user_result.status == V02Status.UNSUPPORTED
    assert user_result.reason_code == "UNSUPPORTED_GEOMETRY"
    assert user_result.fact_grounding is None
    assert user_result.extraction_mode == ExtractionMode.DETERMINISTIC
    evidence = DeterministicPromptGrounder().ground(CASE_PROMPTS["CP2-05"])
    grounding = FactGroundingValidator().validate(evidence, extraction_error)
    assert grounding.status == "GROUNDING_ERROR"
    assert GroundingViolationCode.SUPPORTED_FEATURE_MISCLASSIFIED in _codes(grounding)


@pytest.mark.parametrize("case_id", ["CP2-02", "CP2-04", "CP2-05", "CP2-06", "CP2-07"])
def test_real_cp2_prompts_are_complete_and_do_not_invoke_llm(tmp_path, case_id):
    backend = _FailIfCalledBackend()
    evidence = DeterministicPromptGrounder().ground(CASE_PROMPTS[case_id])
    coverage = DeterministicExtractionCoverage().evaluate(evidence)

    assert coverage.status == ExtractionCoverageStatus.COMPLETE
    result = run_capability_case(case_id, backend, tmp_path / case_id)
    assert result.status == V02Status.VALIDATED
    assert result.extraction_mode == ExtractionMode.DETERMINISTIC
    assert result.fact_source == "deterministic_grounding"
    assert result.llm_invoked is False
    assert result.residual_extraction is None
    assert backend.calls == 0


def test_complete_assembler_produces_exact_cp2_canonical_facts():
    grounder = DeterministicPromptGrounder()
    assembler = DeterministicFactAssembler()

    cp202 = assembler.assemble(grounder.ground(CASE_PROMPTS["CP2-02"]))
    assert cp202.model_dump(mode="json") == {
        "contract_version": "0.2", "width": 100.0, "depth": 60.0, "height": 10.0,
        "holes": [], "hole_groups": [],
        "rectangular_pockets": [{
            "width": 40.0, "depth": 20.0, "x": None, "y": None, "centered": True,
            "cut_depth": 4.0, "through": False,
        }],
        "circular_pockets": [], "slots": [], "linear_hole_patterns": [],
        "rectangular_bosses": [], "cylindrical_bosses": [], "standoffs": [],
        "standoff_groups": [], "additive_linear_patterns": [],
        "fillet_radius": None, "chamfer_distance": None, "unsupported_features": [],
    }
    cp204 = assembler.assemble(grounder.ground(CASE_PROMPTS["CP2-04"]))
    assert cp204.circular_pockets[0].model_dump() == {
        "diameter": 30.0, "x": None, "y": None, "centered": True,
        "cut_depth": 5.0, "through": False,
    }
    cp205 = assembler.assemble(grounder.ground(CASE_PROMPTS["CP2-05"]))
    assert cp205.slots[0].model_dump() == {
        "length": 30.0, "width": 8.0, "x": None, "y": None, "centered": True,
        "angle_deg": 0.0, "cut_depth": None, "through": True,
    }
    cp206 = assembler.assemble(grounder.ground(CASE_PROMPTS["CP2-06"]))
    assert cp206.holes == [] and cp206.hole_groups == []
    assert cp206.linear_hole_patterns[0].model_dump() == {
        "diameter": 5.0, "count": 4, "spacing": 20.0, "axis": "X",
        "anchor_x": 0.0, "anchor_y": 0.0, "anchor_mode": "CENTER",
    }
    cp207 = assembler.assemble(grounder.ground(CASE_PROMPTS["CP2-07"]))
    assert cp207.holes == []
    assert cp207.hole_groups[0].model_dump() == {
        "diameter": 5.0,
        "positions": [{"x": -40.0, "y": 20.0}, {"x": 40.0, "y": 20.0}],
    }
    assert cp207.rectangular_pockets[0].width == 36.0
    assert cp207.rectangular_pockets[0].depth == 18.0
    assert cp207.rectangular_pockets[0].cut_depth == 3.0
    assert cp207.fillet_radius == 3.0
    assert cp207.unsupported_features == []


def test_partial_path_uses_reduced_schema_and_preserves_grounded_facts(tmp_path):
    prompt = "Create an 80 x 40 x 6 mm plate with a horizontal through slot 30 x 8 mm."
    backend = ScriptedFakeBackend([
        '{"contract_version":"0.2","values":{"slots[0].x":-10,"slots[0].y":0}}',
    ])

    result = run_capability_case(
        "CP2-05", backend, tmp_path / "partial", user_prompt=prompt,
    )

    assert result.status == V02Status.VALIDATED
    assert result.grounding_coverage.status == ExtractionCoverageStatus.PARTIAL
    assert result.grounding_coverage.unresolved_fields == ["slots[0].x", "slots[0].y"]
    assert result.extraction_mode == ExtractionMode.HYBRID_LLM
    assert result.fact_source == "hybrid_llm"
    assert result.llm_invoked is True
    assert result.residual_extraction.run.schema_valid is True
    assert result.residual_extraction.run.model_id == "fake-model"
    assert result.residual_extraction.run.prompt_version == "residual-facts-1.0"
    assert result.residual_extraction.run.latency_ms >= 0
    assert result.canonical_facts.slots[0].model_dump() == {
        "length": 30.0, "width": 8.0, "x": -10.0, "y": 0.0, "centered": False,
        "angle_deg": 0.0, "cut_depth": None, "through": True,
    }
    request = backend.requests[0]
    assert request.context.unresolved_fields == ["slots[0].x", "slots[0].y"]
    assert request.response_schema == residual_facts_schema(request.context.unresolved_fields)
    assert set(request.response_schema["properties"]["values"]["properties"]) == {
        "slots[0].x", "slots[0].y",
    }


def test_partial_llm_cannot_overwrite_grounded_fields(tmp_path):
    prompt = "Create an 80 x 40 x 6 mm plate with a horizontal through slot 30 x 8 mm."
    backend = ScriptedFakeBackend([
        '{"contract_version":"0.2","values":{'
        '"slots[0].x":-10,"slots[0].y":0,"width":999}}',
    ])
    result = run_capability_case(
        "CP2-05", backend, tmp_path / "override", user_prompt=prompt,
    )
    assert result.status == V02Status.FAIL
    assert result.reason_code == "SCHEMA_VALIDATION_ERROR"
    assert result.stage == "RESIDUAL_FACT_EXTRACTION"
    assert result.canonical_facts is None
    assert result.r01_plan is None


def test_insufficient_grounding_stops_without_llm(tmp_path):
    backend = _FailIfCalledBackend()
    result = run_capability_case(
        "CP2-05", backend, tmp_path / "insufficient", user_prompt="Create a plate with a slot.",
    )
    assert result.status == V02Status.FAIL
    assert result.grounding_coverage.status == ExtractionCoverageStatus.INSUFFICIENT
    assert result.llm_invoked is False
    assert backend.calls == 0


def test_explicit_unsupported_geometry_is_complete_without_llm(tmp_path):
    backend = _FailIfCalledBackend()
    result = run_capability_case("CP2-09", backend, tmp_path / "thread")
    assert result.status == V02Status.UNSUPPORTED
    assert result.reason_code == "UNSUPPORTED_GEOMETRY"
    assert result.grounding_coverage.status == ExtractionCoverageStatus.COMPLETE
    assert result.llm_invoked is False
    assert result.canonical_facts.unsupported_features == ["thread"]
    assert backend.calls == 0
