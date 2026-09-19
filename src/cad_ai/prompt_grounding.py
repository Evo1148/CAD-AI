from __future__ import annotations

import json
import math
import re
import time
from enum import Enum
from typing import Any, Literal, cast

from pydantic import Field

from .capability_v02 import (
    ExtractedCircularPocketFacts,
    ExtractedCoordinateFacts,
    ExtractedHoleFacts,
    ExtractedHoleGroupFacts,
    ExtractedLinearHolePatternFacts,
    ExtractedPlateFacts,
    ExtractedRectangularPocketFacts,
    ExtractedSlotFacts,
    ExtractedRectangularBossFacts,
    ExtractedCylindricalBossFacts,
    ExtractedStandoffFacts,
    ExtractedStandoffGroupFacts,
    ExtractedAdditiveLinearPatternFacts,
)
from .contracts import StrictModel
from .generation import GenerationError, GenerationErrorCode, GenerationStatus
from .planning import (
    BackendFailureCode,
    BackendRunMetadata,
    BackendStatus,
    InferenceBackend,
    InferenceRequest,
)


GROUNDING_CONTRACT_VERSION = "0.2"

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_COORDINATE_RE = re.compile(
    rf"\(\s*(?P<x>{_NUMBER})\s*,\s*(?P<y>{_NUMBER})\s*\)", re.IGNORECASE,
)
_BASE_RE = re.compile(
    rf"(?P<width>{_NUMBER})\s*[x×]\s*(?P<depth>{_NUMBER})\s*[x×]\s*"
    rf"(?P<height>{_NUMBER})\s*mm\b",
    re.IGNORECASE,
)
_COUNT_WORDS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_EXPLICIT_NEGATIONS = {
    "thread": (
        r"\bthread[- ]free\b",
        r"\bwithout\s+(?:any\s+)?threads?\b",
        r"\bno\s+threads?\b",
        r"\bnon[- ]threaded\b",
        r"\bunthreaded\b",
    ),
    "shell": (
        r"\bshell[- ]free\b",
        r"\bwithout\s+(?:a\s+)?shell\b",
        r"\bno\s+shell\b",
    ),
    "fillet": (
        r"\bfillet[- ]free\b",
        r"\bwithout\s+(?:a\s+)?fillets?\b",
        r"\bno\s+fillets?\b",
    ),
    "chamfer": (
        r"\bchamfer[- ]free\b",
        r"\bwithout\s+(?:a\s+)?chamfers?\b",
        r"\bno\s+chamfers?\b",
    ),
}


def _without_explicit_negations(text: str, feature: str) -> str:
    """Remove only explicit lexical negations for one known feature term."""
    cleaned = text
    for expression in _EXPLICIT_NEGATIONS.get(feature, ()):
        cleaned = re.sub(expression, " ", cleaned, flags=re.IGNORECASE)
    return cleaned


def _positive_feature_mention(text: str, feature: str, expression: str) -> bool:
    return re.search(expression, _without_explicit_negations(text, feature), re.IGNORECASE) is not None


class PromptFeatureType(str, Enum):
    THROUGH_HOLE = "THROUGH_HOLE"
    RECTANGULAR_POCKET = "RECTANGULAR_POCKET"
    RECTANGULAR_CUTOUT = "RECTANGULAR_CUTOUT"
    CIRCULAR_POCKET = "CIRCULAR_POCKET"
    SLOT = "SLOT"
    LINEAR_HOLE_PATTERN = "LINEAR_HOLE_PATTERN"
    FILLET = "FILLET"
    CHAMFER = "CHAMFER"
    RECTANGULAR_BOSS = "RECTANGULAR_BOSS"
    CYLINDRICAL_BOSS = "CYLINDRICAL_BOSS"
    STANDOFF = "STANDOFF"
    ADDITIVE_LINEAR_PATTERN = "ADDITIVE_LINEAR_PATTERN"
    ADDITIVE_BOSS = "ADDITIVE_BOSS"


class CoordinateEvidence(StrictModel):
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)


class HoleGroupEvidence(StrictModel):
    diameter: float = Field(gt=0, allow_inf_nan=False)
    positions: list[CoordinateEvidence] = Field(min_length=1)


class RectangularFeatureEvidence(StrictModel):
    feature_type: Literal["RECTANGULAR_POCKET", "RECTANGULAR_CUTOUT"]
    width: float = Field(gt=0, allow_inf_nan=False)
    depth: float = Field(gt=0, allow_inf_nan=False)
    centered: bool
    x: float | None = Field(default=None, allow_inf_nan=False)
    y: float | None = Field(default=None, allow_inf_nan=False)
    cut_depth: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    through: bool


class CircularPocketEvidence(StrictModel):
    diameter: float = Field(gt=0, allow_inf_nan=False)
    centered: bool
    x: float | None = Field(default=None, allow_inf_nan=False)
    y: float | None = Field(default=None, allow_inf_nan=False)
    cut_depth: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class SlotEvidence(StrictModel):
    length: float = Field(gt=0, allow_inf_nan=False)
    width: float = Field(gt=0, allow_inf_nan=False)
    centered: bool
    x: float | None = Field(default=None, allow_inf_nan=False)
    y: float | None = Field(default=None, allow_inf_nan=False)
    angle_deg: Literal[0.0, 90.0]
    through: bool
    cut_depth: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class LinearHolePatternEvidence(StrictModel):
    diameter: float = Field(gt=0, allow_inf_nan=False)
    count: int = Field(ge=2)
    spacing: float = Field(gt=0, allow_inf_nan=False)
    axis: Literal["X", "Y"]
    anchor_x: float = Field(allow_inf_nan=False)
    anchor_y: float = Field(allow_inf_nan=False)
    anchor_mode: Literal["START", "CENTER"]


class RectangularBossEvidence(StrictModel):
    width: float = Field(gt=0, allow_inf_nan=False)
    depth: float = Field(gt=0, allow_inf_nan=False)
    height: float = Field(gt=0, allow_inf_nan=False)
    centered: bool
    x: float | None = Field(default=None, allow_inf_nan=False)
    y: float | None = Field(default=None, allow_inf_nan=False)


class CylindricalBossEvidence(StrictModel):
    diameter: float = Field(gt=0, allow_inf_nan=False)
    height: float = Field(gt=0, allow_inf_nan=False)
    centered: bool
    x: float | None = Field(default=None, allow_inf_nan=False)
    y: float | None = Field(default=None, allow_inf_nan=False)


class StandoffEvidence(StrictModel):
    outer_diameter: float = Field(gt=0, allow_inf_nan=False)
    height: float = Field(gt=0, allow_inf_nan=False)
    hole_diameter: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    centered: bool
    x: float | None = Field(default=None, allow_inf_nan=False)
    y: float | None = Field(default=None, allow_inf_nan=False)


class StandoffGroupEvidence(StrictModel):
    outer_diameter: float = Field(gt=0, allow_inf_nan=False)
    height: float = Field(gt=0, allow_inf_nan=False)
    hole_diameter: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    positions: list[CoordinateEvidence] = Field(min_length=1)


class AdditiveLinearPatternEvidence(StrictModel):
    feature_type: Literal["CYLINDRICAL_BOSS", "STANDOFF"]
    diameter: float = Field(gt=0, allow_inf_nan=False)
    height: float = Field(gt=0, allow_inf_nan=False)
    hole_diameter: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    count: int = Field(ge=1)
    spacing: float = Field(ge=0, allow_inf_nan=False)
    axis: Literal["X", "Y"]
    anchor_x: float = Field(allow_inf_nan=False)
    anchor_y: float = Field(allow_inf_nan=False)
    anchor_mode: Literal["START", "CENTER"]


class GroundedPromptEvidence(StrictModel):
    contract_version: Literal["0.2"] = GROUNDING_CONTRACT_VERSION
    base_dimensions: tuple[float, float, float] | None = None
    requested_feature_types: list[PromptFeatureType] = Field(default_factory=list)
    explicit_coordinate_tuples: list[CoordinateEvidence] = Field(default_factory=list)
    hole_groups: list[HoleGroupEvidence] = Field(default_factory=list)
    rectangular_features: list[RectangularFeatureEvidence] = Field(default_factory=list)
    circular_pockets: list[CircularPocketEvidence] = Field(default_factory=list)
    slots: list[SlotEvidence] = Field(default_factory=list)
    linear_hole_patterns: list[LinearHolePatternEvidence] = Field(default_factory=list)
    rectangular_bosses: list[RectangularBossEvidence] = Field(default_factory=list)
    cylindrical_bosses: list[CylindricalBossEvidence] = Field(default_factory=list)
    standoffs: list[StandoffEvidence] = Field(default_factory=list)
    standoff_groups: list[StandoffGroupEvidence] = Field(default_factory=list)
    additive_linear_patterns: list[AdditiveLinearPatternEvidence] = Field(default_factory=list)
    explicit_fillet_radius: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    explicit_chamfer_distance: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    unsupported_feature_terms: list[str] = Field(default_factory=list)


class GroundingViolationCode(str, Enum):
    BASE_DIMENSION_MISMATCH = "BASE_DIMENSION_MISMATCH"
    MISSING_SUPPORTED_FEATURE = "MISSING_SUPPORTED_FEATURE"
    SUPPORTED_FEATURE_MISCLASSIFIED = "SUPPORTED_FEATURE_MISCLASSIFIED"
    FEATURE_VALUE_MISMATCH = "FEATURE_VALUE_MISMATCH"
    PATTERN_NOT_PRESERVED = "PATTERN_NOT_PRESERVED"
    PATTERN_VALUE_MISMATCH = "PATTERN_VALUE_MISMATCH"
    PATTERN_EXPANDED_BY_MODEL = "PATTERN_EXPANDED_BY_MODEL"
    SHARED_HOLE_GROUP_MISMATCH = "SHARED_HOLE_GROUP_MISMATCH"
    MISSING_UNSUPPORTED_FEATURE = "MISSING_UNSUPPORTED_FEATURE"


class GroundingViolation(StrictModel):
    code: GroundingViolationCode
    field: str
    expected: object | None = None
    actual: object | None = None


class FactGroundingResult(StrictModel):
    status: Literal["PASS", "GROUNDING_ERROR"]
    grounded_facts: ExtractedPlateFacts | None = None
    violations: list[GroundingViolation] = Field(default_factory=list)


class ExtractionCoverageStatus(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"


class ExtractionMode(str, Enum):
    DETERMINISTIC = "DETERMINISTIC"
    HYBRID_LLM = "HYBRID_LLM"


class ExtractionCoverage(StrictModel):
    status: ExtractionCoverageStatus
    unresolved_fields: list[str] = Field(default_factory=list)
    reason: str | None = None


class ResidualFacts(StrictModel):
    contract_version: Literal["0.2"] = GROUNDING_CONTRACT_VERSION
    values: dict[str, float | int | bool | str] = Field(default_factory=dict)


class ResidualExtractionContext(StrictModel):
    contract_version: Literal["0.2"] = GROUNDING_CONTRACT_VERSION
    user_request: str
    grounded_evidence: GroundedPromptEvidence
    unresolved_fields: list[str] = Field(min_length=1)


class ResidualInferenceRequest(StrictModel):
    system_prompt: str
    context: ResidualExtractionContext
    response_schema: dict[str, Any]
    temperature: float = 0.0
    max_tokens: int = Field(default=256, ge=1)
    seed: int | None = 12345


class ResidualExtractionRun(StrictModel):
    extraction_mode: Literal["HYBRID_LLM"] = "HYBRID_LLM"
    model_id: str | None = None
    prompt_version: str = "residual-facts-1.0"
    latency_ms: float = Field(ge=0)
    backend_status: BackendStatus
    schema_valid: bool
    backend_metadata: BackendRunMetadata | None = None


class ResidualExtractionResult(StrictModel):
    status: GenerationStatus
    residual_facts: ResidualFacts | None = None
    run: ResidualExtractionRun
    raw_output: str | None = None
    error: GenerationError | None = None


RESIDUAL_FACT_EXTRACTION_SYSTEM_PROMPT = """Extract only the explicitly listed unresolved_fields.
Grounded evidence is authoritative and must not be repeated, changed, or contradicted. Return exactly
the requested keys inside values and no others. Do not classify support, add features, create CAD
operations, or return unsupported_features. Return compact JSON matching the supplied schema.
"""


class DeterministicPromptGrounder:
    """Extract only explicit, high-confidence lexical evidence from Capability Pack prompts."""

    def ground(self, prompt: str) -> GroundedPromptEvidence:
        text = " ".join(prompt.strip().split())
        lowered = text.lower()
        base_match = _BASE_RE.search(text)
        base = None
        if base_match:
            base = tuple(float(base_match.group(name)) for name in ("width", "depth", "height"))

        coordinates = [
            CoordinateEvidence(x=float(match.group("x")), y=float(match.group("y")))
            for match in _COORDINATE_RE.finditer(text)
        ]
        requested: list[PromptFeatureType] = []
        non_standoff_holes = re.sub(
            rf"(?:each\s+)?with\s+(?:a\s+)?{_NUMBER}\s*mm\s+through hole\b",
            " ", lowered, flags=re.IGNORECASE,
        )
        if re.search(r"\bthrough holes?\b", non_standoff_holes):
            requested.append(PromptFeatureType.THROUGH_HOLE)
        if "rectangular pocket" in lowered:
            requested.append(PromptFeatureType.RECTANGULAR_POCKET)
        if "rectangular through cutout" in lowered or "rectangular cutout" in lowered:
            requested.append(PromptFeatureType.RECTANGULAR_CUTOUT)
        if "circular pocket" in lowered:
            requested.append(PromptFeatureType.CIRCULAR_POCKET)
        if re.search(r"\bslot\b", lowered):
            requested.append(PromptFeatureType.SLOT)
        if re.search(r"\bthrough holes\s+in\s+(?:a\s+)?(?:horizontal|vertical)\s+linear pattern\b", lowered):
            requested.append(PromptFeatureType.LINEAR_HOLE_PATTERN)
        if _positive_feature_mention(text, "fillet", r"\bfillet\b"):
            requested.append(PromptFeatureType.FILLET)
        if _positive_feature_mention(text, "chamfer", r"\bchamfer\b"):
            requested.append(PromptFeatureType.CHAMFER)
        if "rectangular boss" in lowered:
            requested.append(PromptFeatureType.RECTANGULAR_BOSS)
        if "cylindrical boss" in lowered:
            requested.append(PromptFeatureType.CYLINDRICAL_BOSS)
        if re.search(r"\bboss\b", lowered) and not re.search(r"\b(?:rectangular|cylindrical) boss\b", lowered):
            requested.append(PromptFeatureType.ADDITIVE_BOSS)
        if re.search(r"\b(?:standoffs?|mounting posts?)\b", lowered):
            requested.append(PromptFeatureType.STANDOFF)
        elif re.search(r"\bposts?\b", lowered):
            requested.append(PromptFeatureType.STANDOFF)
        if "linear pattern" in lowered and re.search(r"\b(?:standoffs?|bosses?)\b", lowered):
            requested.append(PromptFeatureType.ADDITIVE_LINEAR_PATTERN)

        pattern = self._linear_pattern(text)
        hole_groups = self._hole_groups(text)
        rectangular = self._rectangular_features(text)
        circular = self._circular_pockets(text)
        slots = self._slots(text)
        fillet = self._edge_value(text, "fillet")
        chamfer = self._edge_value(text, "chamfer")
        rectangular_bosses = self._rectangular_bosses(text)
        cylindrical_bosses = self._cylindrical_bosses(text)
        standoffs, standoff_groups = self._standoffs(text)
        additive_patterns = self._additive_patterns(text)
        unsupported = [
            term for term, expression in (
                ("thread", r"\bthreads?\b|\bthreaded\b"),
                ("shell", r"\bshell\b"),
                ("revolve", r"\brevolv(?:e|ed|ing)\b"),
                ("circular pattern", r"\bcircular pattern\b"),
                ("side-face feature", r"\b(?:side|lateral) face\b"),
            )
            if _positive_feature_mention(text, term, expression)
        ]
        return GroundedPromptEvidence(
            base_dimensions=base,
            requested_feature_types=list(dict.fromkeys(requested)),
            explicit_coordinate_tuples=coordinates,
            hole_groups=hole_groups,
            rectangular_features=rectangular,
            circular_pockets=circular,
            slots=slots,
            linear_hole_patterns=[pattern] if pattern else [],
            rectangular_bosses=rectangular_bosses,
            cylindrical_bosses=cylindrical_bosses,
            standoffs=standoffs,
            standoff_groups=standoff_groups,
            additive_linear_patterns=additive_patterns,
            explicit_fillet_radius=fillet,
            explicit_chamfer_distance=chamfer,
            unsupported_feature_terms=unsupported,
        )

    @staticmethod
    def _hole_groups(text: str) -> list[HoleGroupEvidence]:
        expression = re.compile(
            rf"(?P<diameter>{_NUMBER})\s*mm\s+through holes\s+at\s+(?P<positions>.+?)"
            r"(?=,\s+(?:a|an)\s+|\s+and\s+(?:a|an)\s+|\.|$)",
            re.IGNORECASE,
        )
        groups: list[HoleGroupEvidence] = []
        for match in expression.finditer(text):
            positions = [
                CoordinateEvidence(x=float(item.group("x")), y=float(item.group("y")))
                for item in _COORDINATE_RE.finditer(match.group("positions"))
            ]
            if positions:
                groups.append(HoleGroupEvidence(
                    diameter=float(match.group("diameter")), positions=positions,
                ))
        singular = re.compile(
            rf"(?:\ba\b|\ban\b|\bone\b)\s+(?P<centered>centered\s+)?"
            rf"(?P<diameter>{_NUMBER})\s*mm\s+through hole\b"
            rf"(?:\s+at\s*(?P<position>\(\s*{_NUMBER}\s*,\s*{_NUMBER}\s*\)))?",
            re.IGNORECASE,
        )
        for match in singular.finditer(text):
            coordinate = _COORDINATE_RE.search(match.group("position") or "")
            if coordinate:
                position = CoordinateEvidence(
                    x=float(coordinate.group("x")), y=float(coordinate.group("y")),
                )
            elif match.group("centered"):
                position = CoordinateEvidence(x=0.0, y=0.0)
            else:
                continue
            groups.append(HoleGroupEvidence(
                diameter=float(match.group("diameter")), positions=[position],
            ))
        return groups

    @staticmethod
    def _linear_pattern(text: str) -> LinearHolePatternEvidence | None:
        expression = re.compile(
            rf"(?P<count>\d+|{'|'.join(_COUNT_WORDS)})\s+(?P<diameter>{_NUMBER})\s*mm\s+"
            rf"through holes\s+in\s+(?:a\s+)?(?P<orientation>horizontal|vertical)\s+linear pattern\s+"
            rf"(?P<mode>centered|starting)\s+at\s*(?P<anchor>\(\s*{_NUMBER}\s*,\s*{_NUMBER}\s*\))"
            rf"\s*,?\s*with\s+(?P<spacing>{_NUMBER})\s*mm\s+spacing",
            re.IGNORECASE,
        )
        match = expression.search(text)
        if not match:
            return None
        coordinate = _COORDINATE_RE.search(match.group("anchor"))
        assert coordinate is not None
        raw_count = match.group("count").lower()
        count = int(raw_count) if raw_count.isdigit() else _COUNT_WORDS[raw_count]
        return LinearHolePatternEvidence(
            diameter=float(match.group("diameter")),
            count=count,
            spacing=float(match.group("spacing")),
            axis="X" if match.group("orientation").lower() == "horizontal" else "Y",
            anchor_x=float(coordinate.group("x")),
            anchor_y=float(coordinate.group("y")),
            anchor_mode="CENTER" if match.group("mode").lower() == "centered" else "START",
        )

    @staticmethod
    def _rectangular_features(text: str) -> list[RectangularFeatureEvidence]:
        after_type = re.compile(
            rf"(?P<centered>centered\s+)?rectangular\s+(?P<through>through\s+)?(?P<kind>pocket|cutout)\s+"
            rf"(?P<width>{_NUMBER})\s*[x×]\s*(?P<depth>{_NUMBER})\s*mm"
            rf"(?:\s+and\s+(?P<cut>{_NUMBER})\s*mm\s+deep)?"
            rf"(?:\s+at\s*(?P<position>\(\s*{_NUMBER}\s*,\s*{_NUMBER}\s*\)))?",
            re.IGNORECASE,
        )
        before_type = re.compile(
            rf"(?P<centered>centered\s+)?(?P<width>{_NUMBER})\s*[x×]\s*(?P<depth>{_NUMBER})\s*mm\s+"
            rf"rectangular\s+(?P<through>through\s+)?(?P<kind>pocket|cutout)"
            rf"(?:\s+(?P<cut>{_NUMBER})\s*mm\s+deep)?"
            rf"(?:\s+at\s*(?P<position>\(\s*{_NUMBER}\s*,\s*{_NUMBER}\s*\)))?",
            re.IGNORECASE,
        )
        matches = [*after_type.finditer(text), *before_type.finditer(text)]
        results = []
        for match in sorted(matches, key=lambda item: item.start()):
            kind = match.group("kind").lower()
            through = bool(match.group("through")) or kind == "cutout" and "through" in match.group(0).lower()
            position = _COORDINATE_RE.search(match.group("position") or "")
            results.append(RectangularFeatureEvidence(
                feature_type="RECTANGULAR_CUTOUT" if kind == "cutout" else "RECTANGULAR_POCKET",
                width=float(match.group("width")),
                depth=float(match.group("depth")),
                centered=bool(match.group("centered")),
                x=float(position.group("x")) if position else None,
                y=float(position.group("y")) if position else None,
                cut_depth=None if through or match.group("cut") is None else float(match.group("cut")),
                through=through,
            ))
        return results

    @staticmethod
    def _circular_pockets(text: str) -> list[CircularPocketEvidence]:
        expression = re.compile(
            rf"(?P<centered>centered\s+)?(?P<diameter>{_NUMBER})\s*mm\s+diameter\s+circular pocket"
            rf"(?:\s+(?P<cut>{_NUMBER})\s*mm\s+deep)?"
            rf"(?:\s+at\s*(?P<position>\(\s*{_NUMBER}\s*,\s*{_NUMBER}\s*\)))?",
            re.IGNORECASE,
        )
        results = []
        for match in expression.finditer(text):
            position = _COORDINATE_RE.search(match.group("position") or "")
            results.append(CircularPocketEvidence(
                diameter=float(match.group("diameter")),
                centered=bool(match.group("centered")),
                x=float(position.group("x")) if position else None,
                y=float(position.group("y")) if position else None,
                cut_depth=float(match.group("cut")) if match.group("cut") else None,
            ))
        return results

    @staticmethod
    def _slots(text: str) -> list[SlotEvidence]:
        expression = re.compile(
            rf"(?P<centered>centered\s+)?(?P<orientation>horizontal|vertical)\s+"
            rf"(?P<through>through\s+)?slot\s+(?P<length>{_NUMBER})\s*[x×]\s*(?P<width>{_NUMBER})\s*mm"
            rf"(?:\s+(?P<cut>{_NUMBER})\s*mm\s+deep)?"
            rf"(?:\s+at\s*(?P<position>\(\s*{_NUMBER}\s*,\s*{_NUMBER}\s*\)))?",
            re.IGNORECASE,
        )
        results = []
        for match in expression.finditer(text):
            position = _COORDINATE_RE.search(match.group("position") or "")
            results.append(SlotEvidence(
                length=float(match.group("length")),
                width=float(match.group("width")),
                centered=bool(match.group("centered")),
                x=float(position.group("x")) if position else None,
                y=float(position.group("y")) if position else None,
                angle_deg=0.0 if match.group("orientation").lower() == "horizontal" else 90.0,
                through=bool(match.group("through")),
                cut_depth=None if match.group("through") or not match.group("cut") else float(match.group("cut")),
            ))
        return results

    @staticmethod
    def _edge_value(text: str, feature: str) -> float | None:
        positive_text = _without_explicit_negations(text, feature)
        match = re.search(
            rf"(?P<value>{_NUMBER})\s*mm\s+{feature}\b", positive_text, re.IGNORECASE,
        )
        return float(match.group("value")) if match else None

    @staticmethod
    def _rectangular_bosses(text: str) -> list[RectangularBossEvidence]:
        expression = re.compile(
            rf"(?P<centered>centered\s+)?(?P<width>{_NUMBER})\s*[x×]\s*(?P<depth>{_NUMBER})\s*mm\s+"
            rf"rectangular boss\s+(?P<height>{_NUMBER})\s*mm\s+high"
            rf"(?:\s+at\s*(?P<position>\(\s*{_NUMBER}\s*,\s*{_NUMBER}\s*\)))?",
            re.IGNORECASE,
        )
        results = []
        for match in expression.finditer(text):
            position = _COORDINATE_RE.search(match.group("position") or "")
            results.append(RectangularBossEvidence(
                width=float(match.group("width")), depth=float(match.group("depth")),
                height=float(match.group("height")), centered=bool(match.group("centered")),
                x=float(position.group("x")) if position else None,
                y=float(position.group("y")) if position else None,
            ))
        return results

    @staticmethod
    def _cylindrical_bosses(text: str) -> list[CylindricalBossEvidence]:
        expression = re.compile(
            rf"(?P<centered>centered\s+)?(?P<diameter>{_NUMBER})\s*mm\s+diameter\s+cylindrical boss\s+"
            rf"(?P<height>{_NUMBER})\s*mm\s+high(?!\s+in\s+(?:a\s+)?(?:horizontal|vertical)\s+linear pattern)"
            rf"(?:\s+at\s*(?P<position>\(\s*{_NUMBER}\s*,\s*{_NUMBER}\s*\)))?",
            re.IGNORECASE,
        )
        results = []
        for match in expression.finditer(text):
            position = _COORDINATE_RE.search(match.group("position") or "")
            results.append(CylindricalBossEvidence(
                diameter=float(match.group("diameter")), height=float(match.group("height")),
                centered=bool(match.group("centered")),
                x=float(position.group("x")) if position else None,
                y=float(position.group("y")) if position else None,
            ))
        return results

    @staticmethod
    def _standoffs(text: str) -> tuple[list[StandoffEvidence], list[StandoffGroupEvidence]]:
        single_re = re.compile(
            rf"(?:add\s+)?(?:an?|one)\s+(?P<centered>centered\s+)?(?P<diameter>{_NUMBER})\s*mm\s+diameter\s+"
            rf"(?:standoff|mounting post)\s+(?P<height>{_NUMBER})\s*mm\s+high"
            rf"(?:\s+at\s*(?P<position>\(\s*{_NUMBER}\s*,\s*{_NUMBER}\s*\)))?"
            rf"(?:\s+with\s+(?:a\s+)?(?P<hole>{_NUMBER})\s*mm\s+through hole)?",
            re.IGNORECASE,
        )
        group_re = re.compile(
            rf"(?P<count>\d+|{'|'.join(_COUNT_WORDS)})\s+(?P<diameter>{_NUMBER})\s*mm\s+diameter\s+standoffs\s+"
            rf"(?P<height>{_NUMBER})\s*mm\s+high\s+at\s+(?P<positions>.+?)"
            rf"(?:,?\s*each\s+with\s+(?:a\s+)?(?P<hole>{_NUMBER})\s*mm\s+through hole)?(?=\.|$)",
            re.IGNORECASE,
        )
        singles: list[StandoffEvidence] = []
        groups: list[StandoffGroupEvidence] = []
        for match in single_re.finditer(text):
            position = _COORDINATE_RE.search(match.group("position") or "")
            singles.append(StandoffEvidence(
                outer_diameter=float(match.group("diameter")), height=float(match.group("height")),
                hole_diameter=float(match.group("hole")) if match.group("hole") else None,
                centered=bool(match.group("centered")),
                x=float(position.group("x")) if position else None,
                y=float(position.group("y")) if position else None,
            ))
        for match in group_re.finditer(text):
            positions = [CoordinateEvidence(x=float(item.group("x")), y=float(item.group("y"))) for item in _COORDINATE_RE.finditer(match.group("positions"))]
            raw_count = match.group("count").lower()
            count = int(raw_count) if raw_count.isdigit() else _COUNT_WORDS[raw_count]
            if len(positions) == count:
                groups.append(StandoffGroupEvidence(
                    outer_diameter=float(match.group("diameter")), height=float(match.group("height")),
                    hole_diameter=float(match.group("hole")) if match.group("hole") else None,
                    positions=positions,
                ))
        return singles, groups

    @staticmethod
    def _additive_patterns(text: str) -> list[AdditiveLinearPatternEvidence]:
        expression = re.compile(
            rf"(?P<count>\d+|{'|'.join(_COUNT_WORDS)})\s+(?P<diameter>{_NUMBER})\s*mm\s+diameter\s+"
            rf"(?P<feature>standoffs?|cylindrical bosses?)\s+(?P<height>{_NUMBER})\s*mm\s+high\s+in\s+(?:a\s+)?"
            rf"(?P<orientation>horizontal|vertical)\s+linear pattern\s+(?P<mode>centered|starting)\s+at\s*"
            rf"(?P<anchor>\(\s*{_NUMBER}\s*,\s*{_NUMBER}\s*\))\s*,?\s*"
            rf"(?:with\s+)?(?:spaced\s+)?(?P<spacing>{_NUMBER})\s*mm\s+(?:spacing|apart)"
            rf"(?:\s*,?\s*each\s+with\s+(?:a\s+)?(?P<hole>{_NUMBER})\s*mm\s+through hole)?",
            re.IGNORECASE,
        )
        results = []
        for match in expression.finditer(text):
            anchor = _COORDINATE_RE.search(match.group("anchor"))
            assert anchor is not None
            raw_count = match.group("count").lower()
            results.append(AdditiveLinearPatternEvidence(
                feature_type="STANDOFF" if match.group("feature").lower().startswith("standoff") else "CYLINDRICAL_BOSS",
                diameter=float(match.group("diameter")), height=float(match.group("height")),
                hole_diameter=float(match.group("hole")) if match.group("hole") else None,
                count=int(raw_count) if raw_count.isdigit() else _COUNT_WORDS[raw_count],
                spacing=float(match.group("spacing")),
                axis="X" if match.group("orientation").lower() == "horizontal" else "Y",
                anchor_x=float(anchor.group("x")), anchor_y=float(anchor.group("y")),
                anchor_mode="CENTER" if match.group("mode").lower() == "centered" else "START",
            ))
        return results


class DeterministicExtractionCoverage:
    def evaluate(self, evidence: GroundedPromptEvidence) -> ExtractionCoverage:
        if evidence.unsupported_feature_terms:
            return ExtractionCoverage(
                status=ExtractionCoverageStatus.COMPLETE,
                reason="Explicit unsupported geometry is grounded deterministically.",
            )
        if evidence.base_dimensions is None:
            return ExtractionCoverage(
                status=ExtractionCoverageStatus.INSUFFICIENT,
                reason="No explicit A x B x C mm base dimensions were grounded.",
            )

        requested = set(evidence.requested_feature_types)
        missing_structure = []
        if PromptFeatureType.THROUGH_HOLE in requested and not (
            evidence.hole_groups or evidence.linear_hole_patterns
        ):
            missing_structure.append("through_holes")
        if PromptFeatureType.RECTANGULAR_POCKET in requested and not any(
            item.feature_type == "RECTANGULAR_POCKET" for item in evidence.rectangular_features
        ):
            missing_structure.append("rectangular_pockets")
        if PromptFeatureType.RECTANGULAR_CUTOUT in requested and not any(
            item.feature_type == "RECTANGULAR_CUTOUT" for item in evidence.rectangular_features
        ):
            missing_structure.append("rectangular_cutouts")
        if PromptFeatureType.CIRCULAR_POCKET in requested and not evidence.circular_pockets:
            missing_structure.append("circular_pockets")
        if PromptFeatureType.SLOT in requested and not evidence.slots:
            missing_structure.append("slots")
        if PromptFeatureType.LINEAR_HOLE_PATTERN in requested and not evidence.linear_hole_patterns:
            missing_structure.append("linear_hole_patterns")
        if PromptFeatureType.FILLET in requested and evidence.explicit_fillet_radius is None:
            missing_structure.append("fillet_radius")
        if PromptFeatureType.CHAMFER in requested and evidence.explicit_chamfer_distance is None:
            missing_structure.append("chamfer_distance")
        if PromptFeatureType.RECTANGULAR_BOSS in requested and not evidence.rectangular_bosses:
            missing_structure.append("rectangular_bosses")
        if PromptFeatureType.CYLINDRICAL_BOSS in requested and not (
            evidence.cylindrical_bosses
            or any(item.feature_type == "CYLINDRICAL_BOSS" for item in evidence.additive_linear_patterns)
        ):
            missing_structure.append("cylindrical_bosses")
        if PromptFeatureType.STANDOFF in requested and not (
            evidence.standoffs or evidence.standoff_groups or evidence.additive_linear_patterns
        ):
            missing_structure.append("standoffs")
        if PromptFeatureType.ADDITIVE_LINEAR_PATTERN in requested and not evidence.additive_linear_patterns:
            missing_structure.append("additive_linear_patterns")
        if PromptFeatureType.ADDITIVE_BOSS in requested:
            missing_structure.append("additive_boss.type")
        if missing_structure:
            return ExtractionCoverage(
                status=ExtractionCoverageStatus.INSUFFICIENT,
                unresolved_fields=missing_structure,
                reason="Requested feature structure was not grounded with high confidence.",
            )

        unresolved: list[str] = []
        for name, features in (
            ("rectangular_pockets", evidence.rectangular_features),
            ("circular_pockets", evidence.circular_pockets),
            ("slots", evidence.slots),
            ("rectangular_bosses", evidence.rectangular_bosses),
            ("cylindrical_bosses", evidence.cylindrical_bosses),
            ("standoffs", evidence.standoffs),
        ):
            for index, feature in enumerate(features):
                if feature.centered:
                    continue
                if feature.x is None:
                    unresolved.append(f"{name}[{index}].x")
                if feature.y is None:
                    unresolved.append(f"{name}[{index}].y")
        if unresolved:
            return ExtractionCoverage(
                status=ExtractionCoverageStatus.PARTIAL,
                unresolved_fields=unresolved,
                reason="Feature structure is grounded but explicit position remains unresolved.",
            )
        return ExtractionCoverage(status=ExtractionCoverageStatus.COMPLETE)


class DeterministicFactAssembler:
    def assemble(
        self,
        evidence: GroundedPromptEvidence,
        residual: ResidualFacts | None = None,
    ) -> ExtractedPlateFacts:
        values = residual.values if residual else {}

        def resolve(path: str, grounded: float | None) -> float | None:
            if grounded is not None:
                return grounded
            value = values.get(path)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            return float(value)

        width = depth = height = None
        if evidence.base_dimensions is not None:
            width, depth, height = evidence.base_dimensions
        rectangular = [
            ExtractedRectangularPocketFacts(
                width=item.width,
                depth=item.depth,
                x=None if item.centered else resolve(f"rectangular_pockets[{index}].x", item.x),
                y=None if item.centered else resolve(f"rectangular_pockets[{index}].y", item.y),
                centered=item.centered,
                cut_depth=item.cut_depth,
                through=item.through,
            )
            for index, item in enumerate(evidence.rectangular_features)
        ]
        circular = [
            ExtractedCircularPocketFacts(
                diameter=item.diameter,
                x=None if item.centered else resolve(f"circular_pockets[{index}].x", item.x),
                y=None if item.centered else resolve(f"circular_pockets[{index}].y", item.y),
                centered=item.centered,
                cut_depth=item.cut_depth,
                through=False,
            )
            for index, item in enumerate(evidence.circular_pockets)
        ]
        slots = [
            ExtractedSlotFacts(
                length=item.length,
                width=item.width,
                x=None if item.centered else resolve(f"slots[{index}].x", item.x),
                y=None if item.centered else resolve(f"slots[{index}].y", item.y),
                centered=item.centered,
                angle_deg=item.angle_deg,
                cut_depth=item.cut_depth,
                through=item.through,
            )
            for index, item in enumerate(evidence.slots)
        ]
        rectangular_bosses = [
            ExtractedRectangularBossFacts(
                width=item.width, depth=item.depth, height=item.height,
                x=None if item.centered else resolve(f"rectangular_bosses[{index}].x", item.x),
                y=None if item.centered else resolve(f"rectangular_bosses[{index}].y", item.y),
                centered=item.centered,
            ) for index, item in enumerate(evidence.rectangular_bosses)
        ]
        cylindrical_bosses = [
            ExtractedCylindricalBossFacts(
                diameter=item.diameter, height=item.height,
                x=None if item.centered else resolve(f"cylindrical_bosses[{index}].x", item.x),
                y=None if item.centered else resolve(f"cylindrical_bosses[{index}].y", item.y),
                centered=item.centered,
            ) for index, item in enumerate(evidence.cylindrical_bosses)
        ]
        standoffs = [
            ExtractedStandoffFacts(
                outer_diameter=item.outer_diameter, height=item.height, hole_diameter=item.hole_diameter,
                x=None if item.centered else resolve(f"standoffs[{index}].x", item.x),
                y=None if item.centered else resolve(f"standoffs[{index}].y", item.y),
                centered=item.centered,
            ) for index, item in enumerate(evidence.standoffs)
        ]
        return ExtractedPlateFacts(
            width=width,
            depth=depth,
            height=height,
            holes=[],
            hole_groups=[
                ExtractedHoleGroupFacts(
                    diameter=group.diameter,
                    positions=[
                        ExtractedCoordinateFacts(x=position.x, y=position.y)
                        for position in group.positions
                    ],
                )
                for group in evidence.hole_groups
            ],
            rectangular_pockets=rectangular,
            circular_pockets=circular,
            slots=slots,
            linear_hole_patterns=[
                ExtractedLinearHolePatternFacts(
                    diameter=pattern.diameter,
                    count=pattern.count,
                    spacing=pattern.spacing,
                    axis=pattern.axis,
                    anchor_x=pattern.anchor_x,
                    anchor_y=pattern.anchor_y,
                    anchor_mode=pattern.anchor_mode,
                )
                for pattern in evidence.linear_hole_patterns
            ],
            rectangular_bosses=rectangular_bosses,
            cylindrical_bosses=cylindrical_bosses,
            standoffs=standoffs,
            standoff_groups=[ExtractedStandoffGroupFacts(
                outer_diameter=group.outer_diameter, height=group.height, hole_diameter=group.hole_diameter,
                positions=[ExtractedCoordinateFacts(x=item.x, y=item.y) for item in group.positions],
            ) for group in evidence.standoff_groups],
            additive_linear_patterns=[ExtractedAdditiveLinearPatternFacts(
                feature_type=pattern.feature_type, diameter=pattern.diameter, height=pattern.height,
                hole_diameter=pattern.hole_diameter, count=pattern.count, spacing=pattern.spacing,
                axis=pattern.axis, anchor_x=pattern.anchor_x, anchor_y=pattern.anchor_y,
                anchor_mode=pattern.anchor_mode,
            ) for pattern in evidence.additive_linear_patterns],
            fillet_radius=evidence.explicit_fillet_radius,
            chamfer_distance=evidence.explicit_chamfer_distance,
            unsupported_features=list(evidence.unsupported_feature_terms),
        )


class ResidualLLMExtractor:
    def __init__(
        self,
        backend: InferenceBackend,
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        seed: int | None = 12345,
    ):
        self.backend = backend
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

    def extract(
        self,
        user_prompt: str,
        evidence: GroundedPromptEvidence,
        coverage: ExtractionCoverage,
    ) -> ResidualExtractionResult:
        if coverage.status != ExtractionCoverageStatus.PARTIAL or not coverage.unresolved_fields:
            raise ValueError("ResidualLLMExtractor requires PARTIAL coverage with unresolved fields")
        started = time.perf_counter()
        schema = residual_facts_schema(coverage.unresolved_fields)
        request = ResidualInferenceRequest(
            system_prompt=RESIDUAL_FACT_EXTRACTION_SYSTEM_PROMPT,
            context=ResidualExtractionContext(
                user_request=user_prompt,
                grounded_evidence=evidence,
                unresolved_fields=coverage.unresolved_fields,
            ),
            response_schema=schema,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        response = self.backend.generate(cast(InferenceRequest, request))
        run = ResidualExtractionRun(
            model_id=self.backend.model_id,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            backend_status=response.status,
            schema_valid=False,
            backend_metadata=response.metadata,
        )
        if response.status == BackendStatus.ERROR:
            return ResidualExtractionResult(
                status=GenerationStatus.ERROR,
                run=run,
                error=_residual_backend_error(response.error_code, response.message),
            )
        assert response.raw_output is not None
        try:
            parsed = json.loads(response.raw_output)
        except json.JSONDecodeError as exc:
            return ResidualExtractionResult(
                status=GenerationStatus.ERROR,
                run=run,
                raw_output=response.raw_output,
                error=GenerationError(code=GenerationErrorCode.OUTPUT_PARSE_ERROR, message=str(exc)),
            )
        try:
            residual = ResidualFacts.model_validate(parsed)
        except Exception as exc:
            return ResidualExtractionResult(
                status=GenerationStatus.ERROR,
                run=run,
                raw_output=response.raw_output,
                error=GenerationError(code=GenerationErrorCode.SCHEMA_VALIDATION_ERROR, message=str(exc)),
            )
        if set(residual.values) != set(coverage.unresolved_fields) or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in residual.values.values()
        ):
            return ResidualExtractionResult(
                status=GenerationStatus.ERROR,
                run=run,
                raw_output=response.raw_output,
                error=GenerationError(
                    code=GenerationErrorCode.SCHEMA_VALIDATION_ERROR,
                    message="Residual facts must contain exactly the requested finite numeric fields.",
                ),
            )
        return ResidualExtractionResult(
            status=GenerationStatus.SUCCESS,
            residual_facts=residual,
            run=run.model_copy(update={"schema_valid": True}),
            raw_output=response.raw_output,
        )


def residual_facts_schema(unresolved_fields: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "contract_version": {"type": "string", "const": GROUNDING_CONTRACT_VERSION},
            "values": {
                "type": "object",
                "additionalProperties": False,
                "properties": {field: {"type": "number"} for field in unresolved_fields},
                "required": list(unresolved_fields),
            },
        },
        "required": ["contract_version", "values"],
    }


def _residual_backend_error(code: BackendFailureCode | None, message: str | None) -> GenerationError:
    mapping = {
        BackendFailureCode.MODEL_TIMEOUT: GenerationErrorCode.MODEL_TIMEOUT,
        BackendFailureCode.MODEL_UNAVAILABLE: GenerationErrorCode.MODEL_UNAVAILABLE,
        BackendFailureCode.MODEL_ERROR: GenerationErrorCode.MODEL_ERROR,
    }
    return GenerationError(
        code=mapping.get(code, GenerationErrorCode.MODEL_ERROR),
        message=message or "residual inference backend failed",
    )


class FactGroundingValidator:
    """Validate and safely normalize untrusted facts against deterministic prompt evidence."""

    def validate(
        self,
        evidence: GroundedPromptEvidence,
        facts: ExtractedPlateFacts,
    ) -> FactGroundingResult:
        normalized = facts.model_copy(deep=True)
        violations: list[GroundingViolation] = []

        if evidence.base_dimensions is not None:
            actual = (facts.width, facts.depth, facts.height)
            if not _sequence_equal(evidence.base_dimensions, actual):
                violations.append(_violation(
                    GroundingViolationCode.BASE_DIMENSION_MISMATCH,
                    "base_dimensions", evidence.base_dimensions, actual,
                ))

        self._validate_supported_features(evidence, facts, violations)
        self._validate_rectangular(evidence, facts, violations)
        self._validate_circular(evidence, facts, violations)
        self._validate_slots(evidence, facts, violations)
        self._validate_edges(evidence, facts, violations)
        self._validate_patterns(evidence, facts, violations)
        self._normalize_hole_groups(evidence, normalized, violations)
        self._validate_unsupported(evidence, facts, violations)

        if violations:
            return FactGroundingResult(status="GROUNDING_ERROR", violations=violations)
        return FactGroundingResult(status="PASS", grounded_facts=normalized)

    @staticmethod
    def _validate_supported_features(evidence, facts, violations) -> None:
        presence = {
            PromptFeatureType.THROUGH_HOLE: bool(facts.holes or facts.hole_groups or facts.linear_hole_patterns),
            PromptFeatureType.RECTANGULAR_POCKET: bool(facts.rectangular_pockets),
            PromptFeatureType.RECTANGULAR_CUTOUT: bool(facts.rectangular_pockets),
            PromptFeatureType.CIRCULAR_POCKET: bool(facts.circular_pockets),
            PromptFeatureType.SLOT: bool(facts.slots),
            PromptFeatureType.LINEAR_HOLE_PATTERN: bool(facts.linear_hole_patterns),
            PromptFeatureType.FILLET: facts.fillet_radius is not None,
            PromptFeatureType.CHAMFER: facts.chamfer_distance is not None,
            PromptFeatureType.RECTANGULAR_BOSS: bool(facts.rectangular_bosses),
            PromptFeatureType.CYLINDRICAL_BOSS: bool(facts.cylindrical_bosses),
            PromptFeatureType.STANDOFF: bool(facts.standoffs or facts.standoff_groups or facts.additive_linear_patterns),
            PromptFeatureType.ADDITIVE_LINEAR_PATTERN: bool(facts.additive_linear_patterns),
            PromptFeatureType.ADDITIVE_BOSS: False,
        }
        unsupported_text = " ".join(facts.unsupported_features).lower()
        aliases = {
            PromptFeatureType.THROUGH_HOLE: ("through hole",),
            PromptFeatureType.RECTANGULAR_POCKET: ("rectangular pocket",),
            PromptFeatureType.RECTANGULAR_CUTOUT: ("rectangular cutout", "rectangular through cutout"),
            PromptFeatureType.CIRCULAR_POCKET: ("circular pocket",),
            PromptFeatureType.SLOT: ("slot",),
            PromptFeatureType.LINEAR_HOLE_PATTERN: ("linear pattern",),
            PromptFeatureType.FILLET: ("fillet",),
            PromptFeatureType.CHAMFER: ("chamfer",),
            PromptFeatureType.RECTANGULAR_BOSS: ("rectangular boss",),
            PromptFeatureType.CYLINDRICAL_BOSS: ("cylindrical boss",),
            PromptFeatureType.STANDOFF: ("standoff", "mounting post"),
            PromptFeatureType.ADDITIVE_LINEAR_PATTERN: ("linear pattern",),
            PromptFeatureType.ADDITIVE_BOSS: ("boss",),
        }
        for feature in evidence.requested_feature_types:
            if any(alias in unsupported_text for alias in aliases[feature]):
                violations.append(_violation(
                    GroundingViolationCode.SUPPORTED_FEATURE_MISCLASSIFIED,
                    "unsupported_features", feature.value, facts.unsupported_features,
                ))
            if not presence[feature]:
                violations.append(_violation(
                    GroundingViolationCode.MISSING_SUPPORTED_FEATURE,
                    feature.value, True, False,
                ))

    @staticmethod
    def _validate_rectangular(evidence, facts, violations) -> None:
        for index, expected in enumerate(evidence.rectangular_features):
            if index >= len(facts.rectangular_pockets):
                continue
            actual = facts.rectangular_pockets[index]
            expected_values = (
                expected.width, expected.depth, expected.centered, expected.cut_depth, expected.through,
            )
            actual_values = (actual.width, actual.depth, actual.centered, actual.cut_depth, actual.through)
            if not _sequence_equal(expected_values, actual_values):
                violations.append(_violation(
                    GroundingViolationCode.FEATURE_VALUE_MISMATCH,
                    f"rectangular_pockets[{index}]", expected_values, actual_values,
                ))

    @staticmethod
    def _validate_circular(evidence, facts, violations) -> None:
        for index, expected in enumerate(evidence.circular_pockets):
            if index >= len(facts.circular_pockets):
                continue
            actual = facts.circular_pockets[index]
            expected_values = (expected.diameter, expected.centered, expected.cut_depth, False)
            actual_values = (actual.diameter, actual.centered, actual.cut_depth, actual.through)
            if not _sequence_equal(expected_values, actual_values):
                violations.append(_violation(
                    GroundingViolationCode.FEATURE_VALUE_MISMATCH,
                    f"circular_pockets[{index}]", expected_values, actual_values,
                ))

    @staticmethod
    def _validate_slots(evidence, facts, violations) -> None:
        for index, expected in enumerate(evidence.slots):
            if index >= len(facts.slots):
                continue
            actual = facts.slots[index]
            expected_values = (
                expected.length, expected.width, expected.centered, expected.angle_deg,
                expected.cut_depth, expected.through,
            )
            actual_values = (
                actual.length, actual.width, actual.centered, actual.angle_deg,
                actual.cut_depth, actual.through,
            )
            if not _sequence_equal(expected_values, actual_values):
                violations.append(_violation(
                    GroundingViolationCode.FEATURE_VALUE_MISMATCH,
                    f"slots[{index}]", expected_values, actual_values,
                ))

    @staticmethod
    def _validate_edges(evidence, facts, violations) -> None:
        for field, expected, actual in (
            ("fillet_radius", evidence.explicit_fillet_radius, facts.fillet_radius),
            ("chamfer_distance", evidence.explicit_chamfer_distance, facts.chamfer_distance),
        ):
            if expected is not None and not _value_equal(expected, actual):
                violations.append(_violation(
                    GroundingViolationCode.FEATURE_VALUE_MISMATCH, field, expected, actual,
                ))

    @staticmethod
    def _validate_patterns(evidence, facts, violations) -> None:
        if not evidence.linear_hole_patterns:
            return
        allowed_explicit_positions = {
            (position.x, position.y)
            for group in evidence.hole_groups
            for position in group.positions
        }
        unexpected_holes = [
            item for item in facts.holes
            if _fact_hole_position(item) not in allowed_explicit_positions
        ]
        unexpected_groups = [
            item for item in facts.hole_groups
            if any((position.x, position.y) not in allowed_explicit_positions for position in item.positions)
        ]
        if unexpected_holes or unexpected_groups:
            violations.append(_violation(
                GroundingViolationCode.PATTERN_EXPANDED_BY_MODEL,
                "holes", "pattern facts only", [
                    *[item.model_dump(mode="json") for item in unexpected_holes],
                    *[item.model_dump(mode="json") for item in unexpected_groups],
                ],
            ))
        if len(facts.linear_hole_patterns) != len(evidence.linear_hole_patterns):
            violations.append(_violation(
                GroundingViolationCode.PATTERN_NOT_PRESERVED,
                "linear_hole_patterns", len(evidence.linear_hole_patterns), len(facts.linear_hole_patterns),
            ))
            return
        for index, (expected, actual) in enumerate(zip(evidence.linear_hole_patterns, facts.linear_hole_patterns)):
            expected_values = (
                expected.diameter, expected.count, expected.spacing, expected.axis,
                expected.anchor_x, expected.anchor_y, expected.anchor_mode,
            )
            actual_values = (
                actual.diameter, actual.count, actual.spacing, actual.axis,
                actual.anchor_x, actual.anchor_y, actual.anchor_mode,
            )
            if not _sequence_equal(expected_values, actual_values):
                violations.append(_violation(
                    GroundingViolationCode.PATTERN_VALUE_MISMATCH,
                    f"linear_hole_patterns[{index}]", expected_values, actual_values,
                ))

    @staticmethod
    def _normalize_hole_groups(evidence, facts, violations) -> None:
        for index, expected in enumerate(evidence.hole_groups):
            expected_positions = [(item.x, item.y) for item in expected.positions]
            if index < len(facts.hole_groups):
                group = facts.hole_groups[index]
                actual_positions = [(item.x, item.y) for item in group.positions]
                if actual_positions != expected_positions or (
                    group.diameter is not None and not _value_equal(group.diameter, expected.diameter)
                ):
                    violations.append(_violation(
                        GroundingViolationCode.SHARED_HOLE_GROUP_MISMATCH,
                        f"hole_groups[{index}]", expected.model_dump(mode="json"), group.model_dump(mode="json"),
                    ))
                else:
                    group.diameter = expected.diameter
                continue
            matching = [hole for hole in facts.holes if _fact_hole_position(hole) in expected_positions]
            if len(matching) != len(expected_positions) or {_fact_hole_position(hole) for hole in matching} != set(expected_positions):
                violations.append(_violation(
                    GroundingViolationCode.SHARED_HOLE_GROUP_MISMATCH,
                    "holes", expected_positions, [_fact_hole_position(hole) for hole in facts.holes],
                ))
                continue
            for hole in matching:
                if hole.diameter is not None and not _value_equal(hole.diameter, expected.diameter):
                    violations.append(_violation(
                        GroundingViolationCode.SHARED_HOLE_GROUP_MISMATCH,
                        "holes.diameter", expected.diameter, hole.diameter,
                    ))
                else:
                    hole.diameter = expected.diameter

    @staticmethod
    def _validate_unsupported(evidence, facts, violations) -> None:
        actual = " ".join(facts.unsupported_features).lower()
        for term in evidence.unsupported_feature_terms:
            if term not in actual and not (term == "thread" and "threaded" in actual):
                violations.append(_violation(
                    GroundingViolationCode.MISSING_UNSUPPORTED_FEATURE,
                    "unsupported_features", term, facts.unsupported_features,
                ))


def _value_equal(expected: object, actual: object) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isclose(
            float(expected), float(actual), rel_tol=0.0, abs_tol=1e-9,
        )
    return expected == actual


def _fact_hole_position(hole: ExtractedHoleFacts) -> tuple[float, float] | None:
    if hole.x is not None and hole.y is not None:
        return hole.x, hole.y
    if hole.centered and hole.x is None and hole.y is None:
        return 0.0, 0.0
    return None


def _sequence_equal(expected: tuple | list, actual: tuple | list) -> bool:
    return len(expected) == len(actual) and all(_value_equal(left, right) for left, right in zip(expected, actual))


def _violation(code, field, expected, actual) -> GroundingViolation:
    return GroundingViolation(code=code, field=field, expected=expected, actual=actual)
