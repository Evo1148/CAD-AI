from __future__ import annotations

import json
import math
import time
from enum import Enum
from typing import Annotated, Any, Literal, cast

from pydantic import Field, RootModel, model_validator

from .contracts import (
    CADOperation,
    CADPlan,
    Comparison,
    Constraint,
    DesignSpec,
    FeatureKind,
    Measurement,
    MeasurementType,
    OperationOutput,
    Origin,
    Priority,
    StrictModel,
    ensure_spec_plannable,
)
from .generation import GenerationError, GenerationErrorCode, GenerationStatus
from .planning import (
    BackendFailureCode,
    BackendRunMetadata,
    BackendStatus,
    InferenceBackend,
    InferenceRequest,
)
from .revision import RepairRule


CAPABILITY_CONTRACT_VERSION = "0.2"
CAPABILITY_PROMPT_VERSION = "capability-facts-3.0"


class HoleIntent(StrictModel):
    diameter: float = Field(gt=0, allow_inf_nan=False, strict=True)
    x: float = Field(allow_inf_nan=False, strict=True)
    y: float = Field(allow_inf_nan=False, strict=True)


class ExtractedHoleFacts(StrictModel):
    diameter: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    x: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    y: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    centered: bool = False


class ExtractedCoordinateFacts(StrictModel):
    x: float = Field(allow_inf_nan=False, strict=True)
    y: float = Field(allow_inf_nan=False, strict=True)


class ExtractedHoleGroupFacts(StrictModel):
    """Compact factual representation for one shared diameter and explicit positions."""

    diameter: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    positions: list[ExtractedCoordinateFacts] = Field(min_length=1)


class ExtractedRectangularPocketFacts(StrictModel):
    width: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    depth: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    x: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    y: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    centered: bool = False
    cut_depth: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    through: bool = False


class ExtractedCircularPocketFacts(StrictModel):
    diameter: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    x: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    y: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    centered: bool = False
    cut_depth: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    through: bool = False


class ExtractedSlotFacts(StrictModel):
    length: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    width: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    x: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    y: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    centered: bool = False
    angle_deg: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    cut_depth: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    through: bool = False


class ExtractedLinearHolePatternFacts(StrictModel):
    diameter: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    count: int | None = Field(default=None, strict=True)
    spacing: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    axis: Literal["X", "Y"] | None = None
    anchor_x: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    anchor_y: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    anchor_mode: Literal["START", "CENTER"] | None = None


class ExtractedRectangularBossFacts(StrictModel):
    width: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    depth: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    height: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    x: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    y: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    centered: bool = False


class ExtractedCylindricalBossFacts(StrictModel):
    diameter: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    height: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    x: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    y: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    centered: bool = False


class ExtractedStandoffFacts(StrictModel):
    outer_diameter: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    height: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    hole_diameter: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    x: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    y: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    centered: bool = False


class ExtractedStandoffGroupFacts(StrictModel):
    outer_diameter: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    height: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    hole_diameter: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    positions: list[ExtractedCoordinateFacts] = Field(min_length=1)


class ExtractedAdditiveLinearPatternFacts(StrictModel):
    feature_type: Literal["CYLINDRICAL_BOSS", "STANDOFF"]
    diameter: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    height: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    hole_diameter: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    count: int | None = Field(default=None, strict=True)
    spacing: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    axis: Literal["X", "Y"] | None = None
    anchor_x: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    anchor_y: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    anchor_mode: Literal["START", "CENTER"] | None = None


class ExtractedPlateFacts(StrictModel):
    contract_version: Literal["0.2"] = CAPABILITY_CONTRACT_VERSION
    width: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    depth: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    height: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    holes: list[ExtractedHoleFacts] = Field(default_factory=list)
    hole_groups: list[ExtractedHoleGroupFacts] = Field(default_factory=list)
    rectangular_pockets: list[ExtractedRectangularPocketFacts] = Field(default_factory=list)
    circular_pockets: list[ExtractedCircularPocketFacts] = Field(default_factory=list)
    slots: list[ExtractedSlotFacts] = Field(default_factory=list)
    linear_hole_patterns: list[ExtractedLinearHolePatternFacts] = Field(default_factory=list)
    rectangular_bosses: list[ExtractedRectangularBossFacts] = Field(default_factory=list)
    cylindrical_bosses: list[ExtractedCylindricalBossFacts] = Field(default_factory=list)
    standoffs: list[ExtractedStandoffFacts] = Field(default_factory=list)
    standoff_groups: list[ExtractedStandoffGroupFacts] = Field(default_factory=list)
    additive_linear_patterns: list[ExtractedAdditiveLinearPatternFacts] = Field(default_factory=list)
    fillet_radius: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    chamfer_distance: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    unsupported_features: list[str] = Field(default_factory=list)


class RectangularPocketIntent(StrictModel):
    width: float = Field(gt=0, allow_inf_nan=False, strict=True)
    depth: float = Field(gt=0, allow_inf_nan=False, strict=True)
    x: float = Field(allow_inf_nan=False, strict=True)
    y: float = Field(allow_inf_nan=False, strict=True)
    cut_depth: float | None = Field(default=None, gt=0, allow_inf_nan=False, strict=True)
    through: bool = False


class CircularPocketIntent(StrictModel):
    diameter: float = Field(gt=0, allow_inf_nan=False, strict=True)
    x: float = Field(allow_inf_nan=False, strict=True)
    y: float = Field(allow_inf_nan=False, strict=True)
    cut_depth: float | None = Field(default=None, gt=0, allow_inf_nan=False, strict=True)
    through: bool = False


class SlotIntent(StrictModel):
    length: float = Field(gt=0, allow_inf_nan=False, strict=True)
    width: float = Field(gt=0, allow_inf_nan=False, strict=True)
    x: float = Field(allow_inf_nan=False, strict=True)
    y: float = Field(allow_inf_nan=False, strict=True)
    angle_deg: Literal[0.0, 90.0]
    cut_depth: float | None = Field(default=None, gt=0, allow_inf_nan=False, strict=True)
    through: bool = False


class RectangularBossIntent(StrictModel):
    width: float = Field(gt=0, allow_inf_nan=False, strict=True)
    depth: float = Field(gt=0, allow_inf_nan=False, strict=True)
    height: float = Field(gt=0, allow_inf_nan=False, strict=True)
    x: float = Field(allow_inf_nan=False, strict=True)
    y: float = Field(allow_inf_nan=False, strict=True)


class CylindricalBossIntent(StrictModel):
    diameter: float = Field(gt=0, allow_inf_nan=False, strict=True)
    height: float = Field(gt=0, allow_inf_nan=False, strict=True)
    x: float = Field(allow_inf_nan=False, strict=True)
    y: float = Field(allow_inf_nan=False, strict=True)


class StandoffIntent(StrictModel):
    outer_diameter: float = Field(gt=0, allow_inf_nan=False, strict=True)
    height: float = Field(gt=0, allow_inf_nan=False, strict=True)
    hole_diameter: float | None = Field(default=None, gt=0, allow_inf_nan=False, strict=True)
    x: float = Field(allow_inf_nan=False, strict=True)
    y: float = Field(allow_inf_nan=False, strict=True)

    @model_validator(mode="after")
    def hole_fits(self) -> "StandoffIntent":
        if self.hole_diameter is not None and self.hole_diameter >= self.outer_diameter:
            raise ValueError("standoff hole_diameter must be smaller than outer_diameter")
        return self


class PlateIntent(StrictModel):
    contract_version: Literal["0.2"]
    width: float = Field(gt=0, allow_inf_nan=False, strict=True)
    depth: float = Field(gt=0, allow_inf_nan=False, strict=True)
    height: float = Field(gt=0, allow_inf_nan=False, strict=True)
    holes: list[HoleIntent] = Field(default_factory=list)
    rectangular_pockets: list[RectangularPocketIntent] = Field(default_factory=list)
    circular_pockets: list[CircularPocketIntent] = Field(default_factory=list)
    slots: list[SlotIntent] = Field(default_factory=list)
    rectangular_bosses: list[RectangularBossIntent] = Field(default_factory=list)
    cylindrical_bosses: list[CylindricalBossIntent] = Field(default_factory=list)
    standoffs: list[StandoffIntent] = Field(default_factory=list)
    fillet_radius: float | None = Field(default=None, gt=0, allow_inf_nan=False, strict=True)
    chamfer_distance: float | None = Field(default=None, gt=0, allow_inf_nan=False, strict=True)

    @model_validator(mode="after")
    def supported_subset(self) -> "PlateIntent":
        if self.fillet_radius is not None and self.chamfer_distance is not None:
            raise ValueError("Capability Pack 2 does not combine fillet and chamfer")
        coordinates = [(hole.x, hole.y) for hole in self.holes]
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("holes must have unique X/Y coordinates")
        for feature in [*self.rectangular_pockets, *self.circular_pockets, *self.slots]:
            if feature.through and feature.cut_depth is not None:
                raise ValueError("through features cannot retain cut_depth")
            if not feature.through and feature.cut_depth is None:
                raise ValueError("non-through features require cut_depth")
        return self


class IntentUnsupportedReason(str, Enum):
    AMBIGUOUS_POSITION = "AMBIGUOUS_POSITION"
    UNSUPPORTED_GEOMETRY = "UNSUPPORTED_GEOMETRY"
    INSUFFICIENT_DIMENSIONS = "INSUFFICIENT_DIMENSIONS"
    CONFLICTING_EDGE_TREATMENTS = "CONFLICTING_EDGE_TREATMENTS"


class PlateIntentPayload(StrictModel):
    contract_version: Literal["0.2"]
    status: Literal["INTENT"]
    intent: PlateIntent


class PlateIntentUnsupported(StrictModel):
    contract_version: Literal["0.2"]
    status: Literal["UNSUPPORTED"]
    reason_code: IntentUnsupportedReason
    reason: str | None = None


PlateIntentResponseValue = Annotated[
    PlateIntentPayload | PlateIntentUnsupported,
    Field(discriminator="status"),
]


class PlateIntentResponse(RootModel[PlateIntentResponseValue]):
    pass


class PlatePromptContext(StrictModel):
    contract_version: Literal["0.2"] = CAPABILITY_CONTRACT_VERSION
    user_prompt: str = Field(min_length=1)
    canonical_length_unit: Literal["mm"] = "mm"
    supported_geometry: Literal[
        "RECTANGULAR_PRISM_CAPABILITY_PACK_2",
        "RECTANGULAR_PRISM_CAPABILITY_PACK_3",
    ] = (
        "RECTANGULAR_PRISM_CAPABILITY_PACK_3"
    )
    supported_edge_treatments: list[Literal["FILLET", "CHAMFER"]] = Field(
        default_factory=lambda: ["FILLET", "CHAMFER"]
    )
    coordinate_system: Literal["PLATE_CENTERED_XY"] = "PLATE_CENTERED_XY"


class CapabilityRun(StrictModel):
    stage: Literal[
        "PROMPT_TO_FACTS",
        "FACTS_TO_INTENT",
        "PROMPT_TO_INTENT",
        "INTENT_TO_SPEC",
        "SPEC_TO_PLAN",
    ]
    implementation: str
    model_id: str | None = None
    prompt_version: str = CAPABILITY_PROMPT_VERSION
    latency_ms: float = Field(ge=0)
    backend_status: BackendStatus
    schema_valid: bool
    backend_metadata: BackendRunMetadata | None = None


class PlateIntentGenerationResult(StrictModel):
    status: Literal["SUCCESS", "UNSUPPORTED", "ERROR"]
    intent: PlateIntent | None = None
    response: PlateIntentResponse | None = None
    run: CapabilityRun
    raw_output: str | None = None
    error: GenerationError | None = None


class PlateFactExtractionResult(StrictModel):
    status: GenerationStatus
    facts: ExtractedPlateFacts | None = None
    run: CapabilityRun
    raw_output: str | None = None
    error: GenerationError | None = None


class CapabilitySpecResult(StrictModel):
    status: GenerationStatus
    design_spec: DesignSpec | None = None
    run: CapabilityRun
    error: GenerationError | None = None


class CapabilityPlanResult(StrictModel):
    status: GenerationStatus
    cad_plan: CADPlan | None = None
    run: CapabilityRun
    error: GenerationError | None = None


PLATE_FACT_EXTRACTION_SYSTEM_PROMPT = """Extract only facts requested in user_request into ExtractedPlateFacts.
For a rectangular prism, "A x B x C mm" means width=A, depth=B, height=C. Extract explicit through
holes; rectangular or circular pockets/cutouts; straight slots; linear hole patterns; fillet; and
chamfer; rectangular/cylindrical bosses; and standoffs. Preserve stated dimensions, (x,y) coordinates, depth/through, slot orientation, and pattern
count, spacing, axis, anchor and START/CENTER mode. For an individual centered feature set
centered=true; its coordinates may remain null. Horizontal slot/pattern means axis or angle X/0;
vertical means Y/90. Use null for unstated facts and never calculate pattern instance coordinates.
Represent holes or standoffs sharing stated dimensions plus explicit coordinates as compact groups; do not copy
the shared diameter redundantly. A linear pattern must remain a pattern, never calculated holes.
Put explicitly requested geometry outside this list in unsupported_features. Do not judge validity,
ambiguity, collisions, conflicts, bounds, or support. Do not return status, reason_code, CAD
operations, IDs, constraints, or explanations. Return only compact JSON matching
ExtractedPlateFacts.
"""

# Compatibility alias for callers that imported the previous prompt constant.
CAPABILITY_INTENT_SYSTEM_PROMPT = PLATE_FACT_EXTRACTION_SYSTEM_PROMPT


class _CapabilityUserRequest(StrictModel):
    user_request: str = Field(min_length=1)


class _CapabilityInferenceRequest(StrictModel):
    system_prompt: str
    context: _CapabilityUserRequest
    response_schema: dict[str, Any]
    temperature: float = 0.0
    max_tokens: int = Field(default=1024, ge=1)
    seed: int | None = 12345


class LLMPlateFactExtractor:
    def __init__(
        self,
        backend: InferenceBackend,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = 12345,
    ):
        self.backend = backend
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

    def extract(self, context: PlatePromptContext) -> PlateFactExtractionResult:
        started = time.perf_counter()
        request = _CapabilityInferenceRequest(
            system_prompt=PLATE_FACT_EXTRACTION_SYSTEM_PROMPT,
            context=_CapabilityUserRequest(user_request=context.user_prompt),
            response_schema=ExtractedPlateFacts.model_json_schema(),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        response = self.backend.generate(cast(InferenceRequest, request))
        run = CapabilityRun(
            stage="PROMPT_TO_FACTS",
            implementation=self.backend.backend_type,
            model_id=self.backend.model_id,
            latency_ms=(time.perf_counter() - started) * 1000,
            backend_status=response.status,
            schema_valid=False,
            backend_metadata=response.metadata,
        )
        if response.status == BackendStatus.ERROR:
            return PlateFactExtractionResult(
                status=GenerationStatus.ERROR,
                run=run,
                error=_backend_error(response.error_code, response.message),
            )
        assert response.raw_output is not None
        try:
            parsed = json.loads(response.raw_output)
        except json.JSONDecodeError as exc:
            return PlateFactExtractionResult(
                status=GenerationStatus.ERROR,
                run=run,
                raw_output=response.raw_output,
                error=GenerationError(code=GenerationErrorCode.OUTPUT_PARSE_ERROR, message=str(exc)),
            )
        try:
            structured = ExtractedPlateFacts.model_validate(parsed)
        except Exception as exc:
            return PlateFactExtractionResult(
                status=GenerationStatus.ERROR,
                run=run,
                raw_output=response.raw_output,
                error=GenerationError(code=GenerationErrorCode.SCHEMA_VALIDATION_ERROR, message=str(exc)),
            )
        run = run.model_copy(update={"schema_valid": True})
        return PlateFactExtractionResult(
            status=GenerationStatus.SUCCESS,
            facts=structured,
            run=run,
            raw_output=response.raw_output,
        )


class DeterministicPlateIntentGate:
    def evaluate(self, facts: ExtractedPlateFacts) -> PlateIntentResponse:
        if facts.unsupported_features:
            return _unsupported_intent(
                IntentUnsupportedReason.UNSUPPORTED_GEOMETRY,
                "Unsupported geometry was explicitly requested.",
            )
        if any(value is None or value <= 0 for value in (facts.width, facts.depth, facts.height)):
            return _unsupported_intent(
                IntentUnsupportedReason.INSUFFICIENT_DIMENSIONS,
                "Positive width, depth, and height are required.",
            )
        if facts.fillet_radius is not None and facts.chamfer_distance is not None:
            return _unsupported_intent(
                IntentUnsupportedReason.CONFLICTING_EDGE_TREATMENTS,
                "Fillet and chamfer cannot be combined.",
            )
        if (
            facts.fillet_radius is not None and facts.fillet_radius <= 0
            or facts.chamfer_distance is not None and facts.chamfer_distance <= 0
        ):
            return _unsupported_intent(
                IntentUnsupportedReason.UNSUPPORTED_GEOMETRY,
                "Edge-treatment values must be positive.",
            )
        width, depth, height = float(facts.width), float(facts.depth), float(facts.height)

        holes: list[HoleIntent] = []
        for group in facts.hole_groups:
            if group.diameter is None or group.diameter <= 0:
                return _unsupported_intent(
                    IntentUnsupportedReason.UNSUPPORTED_GEOMETRY,
                    "Every explicit hole group requires a positive diameter.",
                )
            holes.extend(
                HoleIntent(diameter=group.diameter, x=position.x, y=position.y)
                for position in group.positions
            )
        for hole in facts.holes:
            if hole.diameter is None or hole.diameter <= 0:
                return _unsupported_intent(
                    IntentUnsupportedReason.UNSUPPORTED_GEOMETRY,
                    "Every through hole requires a positive diameter.",
                )
            position = _normalized_position(hole.x, hole.y, hole.centered)
            if position is None:
                return _unsupported_intent(
                    IntentUnsupportedReason.AMBIGUOUS_POSITION,
                    "Every through hole requires an explicit position.",
                )
            x, y = position
            holes.append(HoleIntent(diameter=hole.diameter, x=x, y=y))

        for pattern in facts.linear_hole_patterns:
            expanded = _expand_linear_pattern(pattern)
            if expanded is None:
                return _unsupported_intent(
                    IntentUnsupportedReason.UNSUPPORTED_GEOMETRY,
                    "Linear patterns require positive diameter, count and spacing plus axis and anchor.",
                )
            holes.extend(expanded)

        hole_positions = [(hole.x, hole.y) for hole in holes]
        if len(hole_positions) != len(set(hole_positions)):
            return _unsupported_intent(
                IntentUnsupportedReason.UNSUPPORTED_GEOMETRY,
                "Through-hole positions must be unique.",
            )
        if any(not _circle_fits(width, depth, hole.x, hole.y, hole.diameter) for hole in holes):
            return _unsupported_intent(
                IntentUnsupportedReason.UNSUPPORTED_GEOMETRY,
                "A through hole lies outside the base footprint.",
            )

        rectangular_pockets: list[RectangularPocketIntent] = []
        for feature in facts.rectangular_pockets:
            position = _normalized_position(feature.x, feature.y, feature.centered)
            if position is None:
                return _unsupported_intent(
                    IntentUnsupportedReason.AMBIGUOUS_POSITION,
                    "Every rectangular pocket requires an explicit position.",
                )
            if feature.width is None or feature.width <= 0 or feature.depth is None or feature.depth <= 0:
                return _unsupported_intent(
                    IntentUnsupportedReason.UNSUPPORTED_GEOMETRY,
                    "Rectangular pocket dimensions must be positive.",
                )
            cut_depth = _normalized_cut_depth(feature.through, feature.cut_depth, height)
            if cut_depth is False:
                return _unsupported_intent(
                    IntentUnsupportedReason.UNSUPPORTED_GEOMETRY,
                    "Pocket depth must be positive and smaller than base height.",
                )
            x, y = position
            if not _rectangle_fits(width, depth, x, y, feature.width, feature.depth):
                return _unsupported_intent(
                    IntentUnsupportedReason.UNSUPPORTED_GEOMETRY,
                    "A rectangular pocket lies outside the base footprint.",
                )
            rectangular_pockets.append(RectangularPocketIntent(
                width=feature.width,
                depth=feature.depth,
                x=x,
                y=y,
                cut_depth=cut_depth,
                through=feature.through,
            ))

        circular_pockets: list[CircularPocketIntent] = []
        for feature in facts.circular_pockets:
            position = _normalized_position(feature.x, feature.y, feature.centered)
            if position is None:
                return _unsupported_intent(
                    IntentUnsupportedReason.AMBIGUOUS_POSITION,
                    "Every circular pocket requires an explicit position.",
                )
            if feature.diameter is None or feature.diameter <= 0:
                return _unsupported_intent(
                    IntentUnsupportedReason.UNSUPPORTED_GEOMETRY,
                    "Circular pocket diameter must be positive.",
                )
            cut_depth = _normalized_cut_depth(feature.through, feature.cut_depth, height)
            if cut_depth is False:
                return _unsupported_intent(
                    IntentUnsupportedReason.UNSUPPORTED_GEOMETRY,
                    "Pocket depth must be positive and smaller than base height.",
                )
            x, y = position
            if not _circle_fits(width, depth, x, y, feature.diameter):
                return _unsupported_intent(
                    IntentUnsupportedReason.UNSUPPORTED_GEOMETRY,
                    "A circular pocket lies outside the base footprint.",
                )
            circular_pockets.append(CircularPocketIntent(
                diameter=feature.diameter,
                x=x,
                y=y,
                cut_depth=cut_depth,
                through=feature.through,
            ))

        slots: list[SlotIntent] = []
        for feature in facts.slots:
            position = _normalized_position(feature.x, feature.y, feature.centered)
            if position is None:
                return _unsupported_intent(
                    IntentUnsupportedReason.AMBIGUOUS_POSITION,
                    "Every slot requires an explicit position.",
                )
            if feature.length is None or feature.length <= 0 or feature.width is None or feature.width <= 0:
                return _unsupported_intent(
                    IntentUnsupportedReason.UNSUPPORTED_GEOMETRY,
                    "Slot length and width must be positive.",
                )
            if feature.length < feature.width:
                return _unsupported_intent(
                    IntentUnsupportedReason.UNSUPPORTED_GEOMETRY,
                    "Slot length must be greater than or equal to slot width.",
                )
            if feature.angle_deg not in {0.0, 90.0}:
                return _unsupported_intent(
                    IntentUnsupportedReason.UNSUPPORTED_GEOMETRY,
                    "Capability Pack 2 supports only horizontal or vertical slots.",
                )
            cut_depth = _normalized_cut_depth(feature.through, feature.cut_depth, height)
            if cut_depth is False:
                return _unsupported_intent(
                    IntentUnsupportedReason.UNSUPPORTED_GEOMETRY,
                    "Slot depth must be positive and smaller than base height.",
                )
            x, y = position
            footprint_x = feature.length if feature.angle_deg == 0.0 else feature.width
            footprint_y = feature.width if feature.angle_deg == 0.0 else feature.length
            if not _rectangle_fits(width, depth, x, y, footprint_x, footprint_y):
                return _unsupported_intent(
                    IntentUnsupportedReason.UNSUPPORTED_GEOMETRY,
                    "A slot lies outside the base footprint.",
                )
            slots.append(SlotIntent(
                length=feature.length,
                width=feature.width,
                x=x,
                y=y,
                angle_deg=feature.angle_deg,
                cut_depth=cut_depth,
                through=feature.through,
            ))

        rectangular_bosses: list[RectangularBossIntent] = []
        for feature in facts.rectangular_bosses:
            position = _normalized_position(feature.x, feature.y, feature.centered)
            if position is None:
                return _unsupported_intent(IntentUnsupportedReason.AMBIGUOUS_POSITION, "Every rectangular boss requires an explicit position.")
            if any(value is None or value <= 0 for value in (feature.width, feature.depth, feature.height)):
                return _unsupported_intent(IntentUnsupportedReason.UNSUPPORTED_GEOMETRY, "Rectangular boss dimensions must be positive.")
            x, y = position
            assert feature.width is not None and feature.depth is not None and feature.height is not None
            if not _rectangle_fits(width, depth, x, y, feature.width, feature.depth):
                return _unsupported_intent(IntentUnsupportedReason.UNSUPPORTED_GEOMETRY, "A rectangular boss lies outside the base footprint.")
            rectangular_bosses.append(RectangularBossIntent(width=feature.width, depth=feature.depth, height=feature.height, x=x, y=y))

        cylindrical_bosses: list[CylindricalBossIntent] = []
        for feature in facts.cylindrical_bosses:
            position = _normalized_position(feature.x, feature.y, feature.centered)
            if position is None:
                return _unsupported_intent(IntentUnsupportedReason.AMBIGUOUS_POSITION, "Every cylindrical boss requires an explicit position.")
            if feature.diameter is None or feature.diameter <= 0 or feature.height is None or feature.height <= 0:
                return _unsupported_intent(IntentUnsupportedReason.UNSUPPORTED_GEOMETRY, "Cylindrical boss dimensions must be positive.")
            x, y = position
            if not _circle_fits(width, depth, x, y, feature.diameter):
                return _unsupported_intent(IntentUnsupportedReason.UNSUPPORTED_GEOMETRY, "A cylindrical boss lies outside the base footprint.")
            cylindrical_bosses.append(CylindricalBossIntent(diameter=feature.diameter, height=feature.height, x=x, y=y))

        standoffs: list[StandoffIntent] = []
        expanded_standoffs = list(facts.standoffs)
        for group in facts.standoff_groups:
            expanded_standoffs.extend(ExtractedStandoffFacts(
                outer_diameter=group.outer_diameter, height=group.height,
                hole_diameter=group.hole_diameter, x=position.x, y=position.y,
            ) for position in group.positions)
        for pattern in facts.additive_linear_patterns:
            expanded = _expand_additive_pattern(pattern)
            if expanded is None:
                return _unsupported_intent(IntentUnsupportedReason.UNSUPPORTED_GEOMETRY, "Additive patterns require positive dimensions, count and spacing plus axis and anchor.")
            if pattern.feature_type == "STANDOFF":
                expanded_standoffs.extend(expanded)
            else:
                cylindrical_bosses.extend(CylindricalBossIntent(
                    diameter=item.outer_diameter or 0.0, height=item.height or 0.0,
                    x=item.x or 0.0, y=item.y or 0.0,
                ) for item in expanded)
        for feature in expanded_standoffs:
            position = _normalized_position(feature.x, feature.y, feature.centered)
            if position is None:
                return _unsupported_intent(IntentUnsupportedReason.AMBIGUOUS_POSITION, "Every standoff requires an explicit position.")
            if feature.outer_diameter is None or feature.outer_diameter <= 0 or feature.height is None or feature.height <= 0:
                return _unsupported_intent(IntentUnsupportedReason.UNSUPPORTED_GEOMETRY, "Standoff dimensions must be positive.")
            if feature.hole_diameter is not None and (feature.hole_diameter <= 0 or feature.hole_diameter >= feature.outer_diameter):
                return _unsupported_intent(IntentUnsupportedReason.UNSUPPORTED_GEOMETRY, "Standoff hole diameter must be positive and smaller than its outer diameter.")
            x, y = position
            if not _circle_fits(width, depth, x, y, feature.outer_diameter):
                return _unsupported_intent(IntentUnsupportedReason.UNSUPPORTED_GEOMETRY, "A standoff lies outside the base footprint.")
            standoffs.append(StandoffIntent(
                outer_diameter=feature.outer_diameter, height=feature.height,
                hole_diameter=feature.hole_diameter, x=x, y=y,
            ))

        additive_centers = [
            *((item.x, item.y) for item in rectangular_bosses),
            *((item.x, item.y) for item in cylindrical_bosses),
            *((item.x, item.y) for item in standoffs),
        ]
        if len(additive_centers) != len(set(additive_centers)):
            return _unsupported_intent(IntentUnsupportedReason.UNSUPPORTED_GEOMETRY, "Additive features cannot share the same center.")
        footprints: list[tuple[str, float, float, float, float]] = [
            *(("RECT", item.x, item.y, item.width, item.depth) for item in rectangular_bosses),
            *(("CIRCLE", item.x, item.y, item.diameter, item.diameter) for item in cylindrical_bosses),
            *(("CIRCLE", item.x, item.y, item.outer_diameter, item.outer_diameter) for item in standoffs),
        ]
        if any(_footprints_overlap(first, second) for index, first in enumerate(footprints) for second in footprints[index + 1:]):
            return _unsupported_intent(IntentUnsupportedReason.UNSUPPORTED_GEOMETRY, "Additive feature footprints cannot overlap.")

        try:
            intent = PlateIntent(
                contract_version=CAPABILITY_CONTRACT_VERSION,
                width=width,
                depth=depth,
                height=height,
                holes=holes,
                rectangular_pockets=rectangular_pockets,
                circular_pockets=circular_pockets,
                slots=slots,
                rectangular_bosses=rectangular_bosses,
                cylindrical_bosses=cylindrical_bosses,
                standoffs=standoffs,
                fillet_radius=facts.fillet_radius,
                chamfer_distance=facts.chamfer_distance,
            )
        except ValueError:
            return _unsupported_intent(
                IntentUnsupportedReason.UNSUPPORTED_GEOMETRY,
                "Extracted geometry is outside the supported capability packs.",
            )
        return PlateIntentResponse(root=PlateIntentPayload(
            contract_version=CAPABILITY_CONTRACT_VERSION,
            status="INTENT",
            intent=intent,
        ))


def _normalized_position(
    x: float | None,
    y: float | None,
    centered: bool,
) -> tuple[float, float] | None:
    if x is not None and y is not None:
        return float(x), float(y)
    if x is None and y is None and centered:
        return 0.0, 0.0
    return None


def _normalized_cut_depth(
    through: bool,
    cut_depth: float | None,
    base_height: float,
) -> float | None | Literal[False]:
    if through:
        return None
    if cut_depth is None or cut_depth <= 0 or cut_depth >= base_height:
        return False
    return float(cut_depth)


def _expand_linear_pattern(
    pattern: ExtractedLinearHolePatternFacts,
) -> list[HoleIntent] | None:
    if (
        pattern.diameter is None or pattern.diameter <= 0
        or pattern.count is None or pattern.count < 2
        or pattern.spacing is None or pattern.spacing <= 0
        or pattern.axis not in {"X", "Y"}
        or pattern.anchor_mode not in {"START", "CENTER"}
        or pattern.anchor_x is None or pattern.anchor_y is None
    ):
        return None
    offset = 0.0 if pattern.anchor_mode == "START" else -(pattern.count - 1) * pattern.spacing / 2.0
    holes: list[HoleIntent] = []
    for index in range(pattern.count):
        delta = offset + index * pattern.spacing
        x = pattern.anchor_x + (delta if pattern.axis == "X" else 0.0)
        y = pattern.anchor_y + (delta if pattern.axis == "Y" else 0.0)
        holes.append(HoleIntent(diameter=pattern.diameter, x=x, y=y))
    return holes


def _expand_additive_pattern(
    pattern: ExtractedAdditiveLinearPatternFacts,
) -> list[ExtractedStandoffFacts] | None:
    if (
        pattern.diameter is None or pattern.diameter <= 0
        or pattern.height is None or pattern.height <= 0
        or pattern.count is None or pattern.count < 1
        or pattern.axis not in {"X", "Y"}
        or pattern.anchor_mode not in {"START", "CENTER"}
        or pattern.anchor_x is None or pattern.anchor_y is None
        or (pattern.count > 1 and (pattern.spacing is None or pattern.spacing <= 0))
        or (pattern.hole_diameter is not None and (
            pattern.hole_diameter <= 0 or pattern.hole_diameter >= pattern.diameter
        ))
    ):
        return None
    spacing = float(pattern.spacing or 0.0)
    offset = 0.0 if pattern.anchor_mode == "START" else -(pattern.count - 1) * spacing / 2.0
    return [ExtractedStandoffFacts(
        outer_diameter=pattern.diameter,
        height=pattern.height,
        hole_diameter=pattern.hole_diameter,
        x=pattern.anchor_x + ((offset + index * spacing) if pattern.axis == "X" else 0.0),
        y=pattern.anchor_y + ((offset + index * spacing) if pattern.axis == "Y" else 0.0),
    ) for index in range(pattern.count)]


_FOOTPRINT_TOLERANCE = 1e-9


def _circle_fits(
    base_width: float,
    base_depth: float,
    x: float,
    y: float,
    diameter: float,
) -> bool:
    radius = diameter / 2.0
    return (
        abs(x) + radius < base_width / 2.0 - _FOOTPRINT_TOLERANCE
        and abs(y) + radius < base_depth / 2.0 - _FOOTPRINT_TOLERANCE
    )


def _rectangle_fits(
    base_width: float,
    base_depth: float,
    x: float,
    y: float,
    width: float,
    depth: float,
) -> bool:
    return (
        abs(x) + width / 2.0 < base_width / 2.0 - _FOOTPRINT_TOLERANCE
        and abs(y) + depth / 2.0 < base_depth / 2.0 - _FOOTPRINT_TOLERANCE
    )


def _footprints_overlap(
    first: tuple[str, float, float, float, float],
    second: tuple[str, float, float, float, float],
) -> bool:
    kind_a, ax, ay, aw, ad = first
    kind_b, bx, by, bw, bd = second
    if kind_a == kind_b == "CIRCLE":
        return math.hypot(ax - bx, ay - by) < (aw + bw) / 2.0 - _FOOTPRINT_TOLERANCE
    if kind_a == kind_b == "RECT":
        return abs(ax - bx) < (aw + bw) / 2.0 and abs(ay - by) < (ad + bd) / 2.0
    if kind_a == "CIRCLE":
        kind_a, ax, ay, aw, ad, kind_b, bx, by, bw, bd = kind_b, bx, by, bw, bd, kind_a, ax, ay, aw, ad
    closest_x = max(ax - aw / 2.0, min(bx, ax + aw / 2.0))
    closest_y = max(ay - ad / 2.0, min(by, ay + ad / 2.0))
    return math.hypot(bx - closest_x, by - closest_y) < bw / 2.0 - _FOOTPRINT_TOLERANCE


class CapabilityIntentGenerator:
    """Backward-compatible facade: extraction first, deterministic authorization second."""

    def __init__(
        self,
        backend: InferenceBackend,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = 12345,
    ):
        self.backend = backend
        self.extractor = LLMPlateFactExtractor(
            backend,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        self.gate = DeterministicPlateIntentGate()

    def generate(self, context: PlatePromptContext) -> PlateIntentGenerationResult:
        from .prompt_grounding import (
            DeterministicExtractionCoverage,
            DeterministicFactAssembler,
            DeterministicPromptGrounder,
            ExtractionCoverageStatus,
            FactGroundingValidator,
            ResidualLLMExtractor,
        )

        evidence = DeterministicPromptGrounder().ground(context.user_prompt)
        coverage = DeterministicExtractionCoverage().evaluate(evidence)
        if coverage.status == ExtractionCoverageStatus.INSUFFICIENT:
            return PlateIntentGenerationResult(
                status="ERROR",
                run=CapabilityRun(
                    stage="PROMPT_TO_INTENT",
                    implementation="deterministic-evidence-first",
                    latency_ms=0.0,
                    backend_status=BackendStatus.SUCCESS,
                    schema_valid=True,
                ),
                error=GenerationError(
                    code=GenerationErrorCode.EXTRACTION_COVERAGE_INSUFFICIENT,
                    message=coverage.reason or "Prompt grounding is insufficient.",
                ),
            )
        residual_result = None
        if coverage.status == ExtractionCoverageStatus.PARTIAL:
            residual_result = ResidualLLMExtractor(
                self.backend,
                temperature=self.extractor.temperature,
                max_tokens=min(self.extractor.max_tokens, 256),
                seed=self.extractor.seed,
            ).extract(context.user_prompt, evidence, coverage)
            if residual_result.status != GenerationStatus.SUCCESS or residual_result.residual_facts is None:
                return PlateIntentGenerationResult(
                    status="ERROR",
                    run=CapabilityRun(
                        stage="PROMPT_TO_INTENT",
                        implementation=self.backend.backend_type,
                        model_id=self.backend.model_id,
                        latency_ms=residual_result.run.latency_ms,
                        backend_status=residual_result.run.backend_status,
                        schema_valid=residual_result.run.schema_valid,
                        backend_metadata=residual_result.run.backend_metadata,
                    ),
                    raw_output=residual_result.raw_output,
                    error=residual_result.error,
                )
        assembled = DeterministicFactAssembler().assemble(
            evidence,
            residual_result.residual_facts if residual_result else None,
        )
        if residual_result:
            grounding = FactGroundingValidator().validate(evidence, assembled)
            if grounding.status == "GROUNDING_ERROR" or grounding.grounded_facts is None:
                return PlateIntentGenerationResult(
                    status="ERROR",
                    run=CapabilityRun(
                        stage="PROMPT_TO_INTENT",
                        implementation=self.backend.backend_type,
                        model_id=self.backend.model_id,
                        latency_ms=residual_result.run.latency_ms,
                        backend_status=residual_result.run.backend_status,
                        schema_valid=residual_result.run.schema_valid,
                        backend_metadata=residual_result.run.backend_metadata,
                    ),
                    raw_output=residual_result.raw_output,
                    error=GenerationError(
                        code=GenerationErrorCode.GROUNDING_ERROR,
                        message="; ".join(item.code.value for item in grounding.violations),
                    ),
                )
            assembled = grounding.grounded_facts
        run = CapabilityRun(
            stage="PROMPT_TO_INTENT",
            implementation=self.backend.backend_type if residual_result else "deterministic-evidence-first",
            model_id=self.backend.model_id if residual_result else None,
            latency_ms=residual_result.run.latency_ms if residual_result else 0.0,
            backend_status=residual_result.run.backend_status if residual_result else BackendStatus.SUCCESS,
            schema_valid=residual_result.run.schema_valid if residual_result else True,
            backend_metadata=residual_result.run.backend_metadata if residual_result else None,
        )
        response = self.gate.evaluate(assembled)
        if isinstance(response.root, PlateIntentUnsupported):
            return PlateIntentGenerationResult(
                status="UNSUPPORTED",
                response=response,
                run=run,
                raw_output=residual_result.raw_output if residual_result else None,
            )
        return PlateIntentGenerationResult(
            status="SUCCESS",
            intent=response.root.intent,
            response=response,
            run=run,
            raw_output=residual_result.raw_output if residual_result else None,
        )


def capability_intent_generation_schema() -> dict[str, Any]:
    """Compatibility name for the current untrusted fact-extraction schema."""
    return ExtractedPlateFacts.model_json_schema()


def _unsupported_intent(
    reason_code: IntentUnsupportedReason,
    reason: str,
) -> PlateIntentResponse:
    return PlateIntentResponse(root=PlateIntentUnsupported(
        contract_version=CAPABILITY_CONTRACT_VERSION,
        status="UNSUPPORTED",
        reason_code=reason_code,
        reason=reason,
    ))


class CapabilityIntentToSpecCompiler:
    def compile(
        self,
        intent: PlateIntent,
        *,
        design_id: str,
        spec_version: str = "2.0",
    ) -> CapabilitySpecResult:
        started = time.perf_counter()
        run = lambda: CapabilityRun(
            stage="INTENT_TO_SPEC",
            implementation="deterministic-compiler-v0.2",
            latency_ms=(time.perf_counter() - started) * 1000,
            backend_status=BackendStatus.SUCCESS,
            schema_valid=True,
        )
        canonical = canonical_holes(intent)
        rectangular = canonical_rectangular_pockets(intent)
        circular = canonical_circular_pockets(intent)
        slots = canonical_slots(intent)
        rectangular_bosses = canonical_rectangular_bosses(intent)
        cylindrical_bosses = canonical_cylindrical_bosses(intent)
        standoffs = canonical_standoffs(intent)
        error = _validate_geometry(
            intent, canonical, rectangular, circular, slots,
            rectangular_bosses, cylindrical_bosses, standoffs,
        )
        if error:
            return CapabilitySpecResult(status=GenerationStatus.ERROR, run=run(), error=error)

        constraints = [
            _constraint("C_VALID", MeasurementType.MODEL_VALID, True, Origin.SYSTEM, tolerance=0.0),
            _constraint("C_SOLID_COUNT", MeasurementType.SOLID_COUNT, 1.0, Origin.DERIVED, tolerance=0.0),
            _constraint("C_WIDTH", MeasurementType.EXTENT_X, intent.width, Origin.USER),
            _constraint("C_DEPTH", MeasurementType.EXTENT_Y, intent.depth, Origin.USER),
            _constraint(
                "C_HEIGHT",
                MeasurementType.BASE_HEIGHT if (rectangular_bosses or cylindrical_bosses or standoffs) else MeasurementType.EXTENT_Z,
                intent.height,
                Origin.USER,
                feature_id="main_body" if (rectangular_bosses or cylindrical_bosses or standoffs) else None,
            ),
            _constraint("C_HOLE_COUNT", MeasurementType.HOLE_COUNT, float(len(canonical)), Origin.DERIVED, tolerance=0.0),
            _constraint("C_RECT_POCKET_COUNT", MeasurementType.RECTANGULAR_POCKET_COUNT, float(len(rectangular)), Origin.DERIVED, tolerance=0.0),
            _constraint("C_CIRCLE_POCKET_COUNT", MeasurementType.CIRCULAR_POCKET_COUNT, float(len(circular)), Origin.DERIVED, tolerance=0.0),
            _constraint("C_SLOT_COUNT", MeasurementType.SLOT_COUNT, float(len(slots)), Origin.DERIVED, tolerance=0.0),
            _constraint("C_RECT_BOSS_COUNT", MeasurementType.RECTANGULAR_BOSS_COUNT, float(len(rectangular_bosses)), Origin.DERIVED, tolerance=0.0),
            _constraint("C_CYL_BOSS_COUNT", MeasurementType.CYLINDRICAL_BOSS_COUNT, float(len(cylindrical_bosses)), Origin.DERIVED, tolerance=0.0),
            _constraint("C_STANDOFF_COUNT", MeasurementType.STANDOFF_COUNT, float(len(standoffs)), Origin.DERIVED, tolerance=0.0),
        ]
        parameters: dict[str, float | int | str | bool] = {
            "width": intent.width,
            "depth": intent.depth,
            "height": intent.height,
            "hole_count": len(canonical),
            "rectangular_pocket_count": len(rectangular),
            "circular_pocket_count": len(circular),
            "slot_count": len(slots),
            "rectangular_boss_count": len(rectangular_bosses),
            "cylindrical_boss_count": len(cylindrical_bosses),
            "standoff_count": len(standoffs),
        }
        for hole_id, hole in canonical:
            feature_id = hole_feature_id(hole_id)
            axis_id = hole_axis_id(hole_id)
            parameters[f"{hole_id}_diameter"] = hole.diameter
            parameters[f"{hole_id}_x"] = hole.x
            parameters[f"{hole_id}_y"] = hole.y
            constraints.extend([
                _constraint(
                    f"C_{hole_id}_EXISTS",
                    MeasurementType.FEATURE_EXISTS,
                    True,
                    Origin.DERIVED,
                    feature_id=feature_id,
                    tolerance=0.0,
                ),
                _constraint(
                    f"C_{hole_id}_DIAMETER",
                    MeasurementType.DIAMETER,
                    hole.diameter,
                    Origin.USER,
                    feature_id=feature_id,
                ),
                _constraint(
                    f"C_{hole_id}_X",
                    MeasurementType.POSITION_X,
                    hole.x,
                    Origin.USER,
                    feature_id=axis_id,
                ),
                _constraint(
                    f"C_{hole_id}_Y",
                    MeasurementType.POSITION_Y,
                    hole.y,
                    Origin.USER,
                    feature_id=axis_id,
                ),
            ])
        for pocket_id, pocket in rectangular:
            feature_id = rectangular_pocket_feature_id(pocket_id)
            center_id = f"{feature_id}_center"
            parameters.update({
                f"{pocket_id}_width": pocket.width,
                f"{pocket_id}_depth": pocket.depth,
                f"{pocket_id}_x": pocket.x,
                f"{pocket_id}_y": pocket.y,
                f"{pocket_id}_through": pocket.through,
            })
            constraints.extend(_subtractive_constraints(
                pocket_id,
                feature_id,
                center_id,
                width=pocket.width,
                depth=pocket.depth,
                x=pocket.x,
                y=pocket.y,
                cut_depth=pocket.cut_depth,
                through=pocket.through,
            ))
            if pocket.cut_depth is not None:
                parameters[f"{pocket_id}_cut_depth"] = pocket.cut_depth
        for pocket_id, pocket in circular:
            feature_id = circular_pocket_feature_id(pocket_id)
            center_id = f"{feature_id}_center"
            parameters.update({
                f"{pocket_id}_diameter": pocket.diameter,
                f"{pocket_id}_x": pocket.x,
                f"{pocket_id}_y": pocket.y,
                f"{pocket_id}_through": pocket.through,
            })
            constraints.extend([
                _constraint(f"C_{pocket_id}_EXISTS", MeasurementType.FEATURE_EXISTS, True, Origin.DERIVED, feature_id=feature_id, tolerance=0.0),
                _constraint(f"C_{pocket_id}_DIAMETER", MeasurementType.DIAMETER, pocket.diameter, Origin.USER, feature_id=feature_id),
                _constraint(f"C_{pocket_id}_X", MeasurementType.POSITION_X, pocket.x, Origin.USER, feature_id=center_id),
                _constraint(f"C_{pocket_id}_Y", MeasurementType.POSITION_Y, pocket.y, Origin.USER, feature_id=center_id),
                _constraint(f"C_{pocket_id}_THROUGH", MeasurementType.THROUGH, pocket.through, Origin.USER, feature_id=feature_id, tolerance=0.0),
            ])
            if pocket.cut_depth is not None:
                parameters[f"{pocket_id}_cut_depth"] = pocket.cut_depth
                constraints.append(_constraint(
                    f"C_{pocket_id}_CUT_DEPTH", MeasurementType.CUT_DEPTH, pocket.cut_depth,
                    Origin.USER, feature_id=feature_id,
                ))
        for slot_id, slot in slots:
            feature_id = slot_feature_id(slot_id)
            axis_id = f"{feature_id}_axis"
            parameters.update({
                f"{slot_id}_length": slot.length,
                f"{slot_id}_width": slot.width,
                f"{slot_id}_x": slot.x,
                f"{slot_id}_y": slot.y,
                f"{slot_id}_angle_deg": slot.angle_deg,
                f"{slot_id}_through": slot.through,
            })
            constraints.extend([
                _constraint(f"C_{slot_id}_EXISTS", MeasurementType.FEATURE_EXISTS, True, Origin.DERIVED, feature_id=feature_id, tolerance=0.0),
                _constraint(f"C_{slot_id}_LENGTH", MeasurementType.SLOT_LENGTH, slot.length, Origin.USER, feature_id=feature_id),
                _constraint(f"C_{slot_id}_WIDTH", MeasurementType.SLOT_WIDTH, slot.width, Origin.USER, feature_id=feature_id),
                _constraint(f"C_{slot_id}_X", MeasurementType.POSITION_X, slot.x, Origin.USER, feature_id=axis_id),
                _constraint(f"C_{slot_id}_Y", MeasurementType.POSITION_Y, slot.y, Origin.USER, feature_id=axis_id),
                _constraint(f"C_{slot_id}_ANGLE", MeasurementType.ORIENTATION_ANGLE, slot.angle_deg, Origin.USER, feature_id=feature_id, tolerance=0.01),
                _constraint(f"C_{slot_id}_THROUGH", MeasurementType.THROUGH, slot.through, Origin.USER, feature_id=feature_id, tolerance=0.0),
            ])
            if slot.cut_depth is not None:
                parameters[f"{slot_id}_cut_depth"] = slot.cut_depth
                constraints.append(_constraint(
                    f"C_{slot_id}_CUT_DEPTH", MeasurementType.CUT_DEPTH, slot.cut_depth,
                    Origin.USER, feature_id=feature_id,
                ))
        for boss_id, boss in rectangular_bosses:
            feature_id = rectangular_boss_feature_id(boss_id)
            center_id = f"{feature_id}_center"
            parameters.update({
                f"{boss_id}_width": boss.width, f"{boss_id}_depth": boss.depth,
                f"{boss_id}_height": boss.height, f"{boss_id}_x": boss.x, f"{boss_id}_y": boss.y,
            })
            constraints.extend(_additive_constraints(
                boss_id, feature_id, center_id, width=boss.width, depth=boss.depth,
                height=boss.height, x=boss.x, y=boss.y,
            ))
        for boss_id, boss in cylindrical_bosses:
            feature_id = cylindrical_boss_feature_id(boss_id)
            axis_id = f"{feature_id}_axis"
            parameters.update({
                f"{boss_id}_diameter": boss.diameter, f"{boss_id}_height": boss.height,
                f"{boss_id}_x": boss.x, f"{boss_id}_y": boss.y,
            })
            constraints.extend([
                _constraint(f"C_{boss_id}_EXISTS", MeasurementType.FEATURE_EXISTS, True, Origin.DERIVED, feature_id=feature_id, tolerance=0.0),
                _constraint(f"C_{boss_id}_DIAMETER", MeasurementType.DIAMETER, boss.diameter, Origin.USER, feature_id=feature_id),
                _constraint(f"C_{boss_id}_HEIGHT", MeasurementType.FEATURE_HEIGHT, boss.height, Origin.USER, feature_id=feature_id),
                _constraint(f"C_{boss_id}_X", MeasurementType.POSITION_X, boss.x, Origin.USER, feature_id=axis_id),
                _constraint(f"C_{boss_id}_Y", MeasurementType.POSITION_Y, boss.y, Origin.USER, feature_id=axis_id),
            ])
        for standoff_id, standoff in standoffs:
            feature_id = standoff_feature_id(standoff_id)
            axis_id = f"{feature_id}_axis"
            parameters.update({
                f"{standoff_id}_outer_diameter": standoff.outer_diameter,
                f"{standoff_id}_height": standoff.height,
                f"{standoff_id}_x": standoff.x, f"{standoff_id}_y": standoff.y,
            })
            constraints.extend([
                _constraint(f"C_{standoff_id}_EXISTS", MeasurementType.FEATURE_EXISTS, True, Origin.DERIVED, feature_id=feature_id, tolerance=0.0),
                _constraint(f"C_{standoff_id}_OUTER_DIAMETER", MeasurementType.OUTER_DIAMETER, standoff.outer_diameter, Origin.USER, feature_id=feature_id),
                _constraint(f"C_{standoff_id}_HEIGHT", MeasurementType.FEATURE_HEIGHT, standoff.height, Origin.USER, feature_id=feature_id),
                _constraint(f"C_{standoff_id}_X", MeasurementType.POSITION_X, standoff.x, Origin.USER, feature_id=axis_id),
                _constraint(f"C_{standoff_id}_Y", MeasurementType.POSITION_Y, standoff.y, Origin.USER, feature_id=axis_id),
            ])
            if standoff.hole_diameter is not None:
                hole_id = f"{feature_id}_hole"
                parameters[f"{standoff_id}_hole_diameter"] = standoff.hole_diameter
                constraints.extend([
                    _constraint(f"C_{standoff_id}_HOLE_EXISTS", MeasurementType.FEATURE_EXISTS, True, Origin.DERIVED, feature_id=hole_id, tolerance=0.0),
                    _constraint(f"C_{standoff_id}_HOLE_DIAMETER", MeasurementType.INNER_DIAMETER, standoff.hole_diameter, Origin.USER, feature_id=hole_id),
                ])
        if intent.fillet_radius is not None:
            parameters["fillet_radius"] = intent.fillet_radius
            constraints.extend([
                _constraint(
                    "C_FILLET_EXISTS", MeasurementType.FEATURE_EXISTS, True, Origin.DERIVED,
                    feature_id="plate_fillet", tolerance=0.0,
                ),
                _constraint(
                    "C_FILLET_RADIUS", MeasurementType.FILLET_RADIUS, intent.fillet_radius,
                    Origin.USER, feature_id="plate_fillet",
                ),
            ])
        if intent.chamfer_distance is not None:
            parameters["chamfer_distance"] = intent.chamfer_distance
            constraints.extend([
                _constraint(
                    "C_CHAMFER_EXISTS", MeasurementType.FEATURE_EXISTS, True, Origin.DERIVED,
                    feature_id="plate_chamfer", tolerance=0.0,
                ),
                _constraint(
                    "C_CHAMFER_DISTANCE", MeasurementType.CHAMFER_DISTANCE, intent.chamfer_distance,
                    Origin.USER, feature_id="plate_chamfer",
                ),
            ])
        spec = DesignSpec(
            design_id=design_id,
            spec_version=spec_version,
            parameters=parameters,
            constraints=constraints,
        )
        ensure_spec_plannable(spec)
        return CapabilitySpecResult(status=GenerationStatus.SUCCESS, design_spec=spec, run=run())


class CapabilitySpecToPlanCompiler:
    def compile(self, spec: DesignSpec) -> CapabilityPlanResult:
        started = time.perf_counter()
        run = lambda: CapabilityRun(
            stage="SPEC_TO_PLAN",
            implementation="deterministic-compiler-v0.2",
            latency_ms=(time.perf_counter() - started) * 1000,
            backend_status=BackendStatus.SUCCESS,
            schema_valid=True,
        )
        try:
            _boolean_target(spec, "C_VALID", MeasurementType.MODEL_VALID, True)
            _numeric_target(spec, "C_SOLID_COUNT", MeasurementType.SOLID_COUNT, expected_value=1.0)
            width = _numeric_target(spec, "C_WIDTH", MeasurementType.EXTENT_X)
            depth = _numeric_target(spec, "C_DEPTH", MeasurementType.EXTENT_Y)
            height_constraint = next(item for item in spec.constraints if item.constraint_id == "C_HEIGHT")
            height = _numeric_target(spec, "C_HEIGHT", height_constraint.measurement.type, feature_id=height_constraint.measurement.feature_id)
            hole_count_value = _numeric_target(spec, "C_HOLE_COUNT", MeasurementType.HOLE_COUNT)
            rect_count_value = _numeric_target(spec, "C_RECT_POCKET_COUNT", MeasurementType.RECTANGULAR_POCKET_COUNT)
            circle_count_value = _numeric_target(spec, "C_CIRCLE_POCKET_COUNT", MeasurementType.CIRCULAR_POCKET_COUNT)
            slot_count_value = _numeric_target(spec, "C_SLOT_COUNT", MeasurementType.SLOT_COUNT)
            rect_boss_count_value = _numeric_target(spec, "C_RECT_BOSS_COUNT", MeasurementType.RECTANGULAR_BOSS_COUNT)
            cyl_boss_count_value = _numeric_target(spec, "C_CYL_BOSS_COUNT", MeasurementType.CYLINDRICAL_BOSS_COUNT)
            standoff_count_value = _numeric_target(spec, "C_STANDOFF_COUNT", MeasurementType.STANDOFF_COUNT)
            if hole_count_value < 0 or not hole_count_value.is_integer():
                raise ValueError("C_HOLE_COUNT must be a non-negative integer")
            for name, value in (
                ("C_RECT_POCKET_COUNT", rect_count_value),
                ("C_CIRCLE_POCKET_COUNT", circle_count_value),
                ("C_SLOT_COUNT", slot_count_value),
                ("C_RECT_BOSS_COUNT", rect_boss_count_value),
                ("C_CYL_BOSS_COUNT", cyl_boss_count_value),
                ("C_STANDOFF_COUNT", standoff_count_value),
            ):
                if value < 0 or not value.is_integer():
                    raise ValueError(f"{name} must be a non-negative integer")
            hole_count = int(hole_count_value)
            rect_count = int(rect_count_value)
            circle_count = int(circle_count_value)
            slot_count = int(slot_count_value)
            rect_boss_count = int(rect_boss_count_value)
            cyl_boss_count = int(cyl_boss_count_value)
            standoff_count = int(standoff_count_value)
            operations = [CADOperation(
                operation_id="OP01",
                operation_type="box",
                params={"x": width, "y": depth, "z": height},
                outputs=[OperationOutput(
                    feature_id="main_body",
                    kind=FeatureKind.BODY,
                    semantic_role="primary solid",
                )],
            )]
            for index in range(1, hole_count + 1):
                hole_id = f"H{index:02d}"
                feature_id = hole_feature_id(hole_id)
                axis_id = hole_axis_id(hole_id)
                _boolean_target(
                    spec,
                    f"C_{hole_id}_EXISTS",
                    MeasurementType.FEATURE_EXISTS,
                    True,
                    feature_id=feature_id,
                )
                operations.append(CADOperation(
                    operation_id=f"OP{index + 1:02d}",
                    operation_type="hole",
                    inputs=["main_body"],
                    params={
                        "x": _numeric_target(
                            spec, f"C_{hole_id}_X", MeasurementType.POSITION_X, feature_id=axis_id,
                        ),
                        "y": _numeric_target(
                            spec, f"C_{hole_id}_Y", MeasurementType.POSITION_Y, feature_id=axis_id,
                        ),
                        "diameter": _numeric_target(
                            spec, f"C_{hole_id}_DIAMETER", MeasurementType.DIAMETER, feature_id=feature_id,
                        ),
                    },
                    outputs=[
                        OperationOutput(
                            feature_id=feature_id,
                            kind=FeatureKind.FEATURE,
                            semantic_role="through_hole",
                        ),
                        OperationOutput(
                            feature_id=axis_id,
                            kind=FeatureKind.DATUM_AXIS,
                            semantic_role="through_hole_axis",
                        ),
                    ],
                ))
            next_index = hole_count + 2
            for index in range(1, rect_count + 1):
                pocket_id = f"RP{index:02d}"
                feature_id = rectangular_pocket_feature_id(pocket_id)
                center_id = f"{feature_id}_center"
                through = _boolean_value(
                    spec, f"C_{pocket_id}_THROUGH", MeasurementType.THROUGH, feature_id=feature_id,
                )
                params: dict[str, Any] = {
                    "width": _numeric_target(spec, f"C_{pocket_id}_WIDTH", MeasurementType.FEATURE_WIDTH, feature_id=feature_id),
                    "depth": _numeric_target(spec, f"C_{pocket_id}_DEPTH", MeasurementType.FEATURE_DEPTH, feature_id=feature_id),
                    "x": _numeric_target(spec, f"C_{pocket_id}_X", MeasurementType.POSITION_X, feature_id=center_id),
                    "y": _numeric_target(spec, f"C_{pocket_id}_Y", MeasurementType.POSITION_Y, feature_id=center_id),
                    "through": through,
                    "cut_depth": None if through else _numeric_target(
                        spec, f"C_{pocket_id}_CUT_DEPTH", MeasurementType.CUT_DEPTH, feature_id=feature_id,
                    ),
                }
                operations.append(CADOperation(
                    operation_id=f"OP{next_index:02d}",
                    operation_type="rect_pocket",
                    inputs=["main_body"],
                    params=params,
                    outputs=[
                        OperationOutput(feature_id=feature_id, kind=FeatureKind.FEATURE, semantic_role="rectangular_pocket"),
                        OperationOutput(feature_id=center_id, kind=FeatureKind.DATUM_POINT, semantic_role="rectangular_pocket_center"),
                    ],
                ))
                next_index += 1
            for index in range(1, circle_count + 1):
                pocket_id = f"CP{index:02d}"
                feature_id = circular_pocket_feature_id(pocket_id)
                center_id = f"{feature_id}_center"
                through = _boolean_value(
                    spec, f"C_{pocket_id}_THROUGH", MeasurementType.THROUGH, feature_id=feature_id,
                )
                operations.append(CADOperation(
                    operation_id=f"OP{next_index:02d}",
                    operation_type="circular_pocket",
                    inputs=["main_body"],
                    params={
                        "diameter": _numeric_target(spec, f"C_{pocket_id}_DIAMETER", MeasurementType.DIAMETER, feature_id=feature_id),
                        "x": _numeric_target(spec, f"C_{pocket_id}_X", MeasurementType.POSITION_X, feature_id=center_id),
                        "y": _numeric_target(spec, f"C_{pocket_id}_Y", MeasurementType.POSITION_Y, feature_id=center_id),
                        "through": through,
                        "cut_depth": None if through else _numeric_target(
                            spec, f"C_{pocket_id}_CUT_DEPTH", MeasurementType.CUT_DEPTH, feature_id=feature_id,
                        ),
                    },
                    outputs=[
                        OperationOutput(feature_id=feature_id, kind=FeatureKind.FEATURE, semantic_role="circular_pocket"),
                        OperationOutput(feature_id=center_id, kind=FeatureKind.DATUM_POINT, semantic_role="circular_pocket_center"),
                    ],
                ))
                next_index += 1
            for index in range(1, slot_count + 1):
                slot_id = f"S{index:02d}"
                feature_id = slot_feature_id(slot_id)
                axis_id = f"{feature_id}_axis"
                through = _boolean_value(
                    spec, f"C_{slot_id}_THROUGH", MeasurementType.THROUGH, feature_id=feature_id,
                )
                operations.append(CADOperation(
                    operation_id=f"OP{next_index:02d}",
                    operation_type="slot",
                    inputs=["main_body"],
                    params={
                        "length": _numeric_target(spec, f"C_{slot_id}_LENGTH", MeasurementType.SLOT_LENGTH, feature_id=feature_id),
                        "width": _numeric_target(spec, f"C_{slot_id}_WIDTH", MeasurementType.SLOT_WIDTH, feature_id=feature_id),
                        "x": _numeric_target(spec, f"C_{slot_id}_X", MeasurementType.POSITION_X, feature_id=axis_id),
                        "y": _numeric_target(spec, f"C_{slot_id}_Y", MeasurementType.POSITION_Y, feature_id=axis_id),
                        "angle_deg": _numeric_target(spec, f"C_{slot_id}_ANGLE", MeasurementType.ORIENTATION_ANGLE, feature_id=feature_id),
                        "through": through,
                        "cut_depth": None if through else _numeric_target(
                            spec, f"C_{slot_id}_CUT_DEPTH", MeasurementType.CUT_DEPTH, feature_id=feature_id,
                        ),
                    },
                    outputs=[
                        OperationOutput(feature_id=feature_id, kind=FeatureKind.FEATURE, semantic_role="slot"),
                        OperationOutput(feature_id=axis_id, kind=FeatureKind.DATUM_AXIS, semantic_role="slot_axis"),
                    ],
                ))
                next_index += 1
            for index in range(1, rect_boss_count + 1):
                boss_id = f"RB{index:02d}"
                feature_id = rectangular_boss_feature_id(boss_id)
                center_id = f"{feature_id}_center"
                operations.append(CADOperation(
                    operation_id=f"OP{next_index:02d}", operation_type="rect_boss", inputs=["main_body"],
                    params={
                        "width": _numeric_target(spec, f"C_{boss_id}_WIDTH", MeasurementType.FEATURE_WIDTH, feature_id=feature_id),
                        "depth": _numeric_target(spec, f"C_{boss_id}_DEPTH", MeasurementType.FEATURE_DEPTH, feature_id=feature_id),
                        "height": _numeric_target(spec, f"C_{boss_id}_HEIGHT", MeasurementType.FEATURE_HEIGHT, feature_id=feature_id),
                        "x": _numeric_target(spec, f"C_{boss_id}_X", MeasurementType.POSITION_X, feature_id=center_id),
                        "y": _numeric_target(spec, f"C_{boss_id}_Y", MeasurementType.POSITION_Y, feature_id=center_id),
                    },
                    outputs=[
                        OperationOutput(feature_id=feature_id, kind=FeatureKind.FEATURE, semantic_role="rectangular_boss"),
                        OperationOutput(feature_id=center_id, kind=FeatureKind.DATUM_POINT, semantic_role="rectangular_boss_center"),
                    ],
                ))
                next_index += 1
            for index in range(1, cyl_boss_count + 1):
                boss_id = f"CB{index:02d}"
                feature_id = cylindrical_boss_feature_id(boss_id)
                axis_id = f"{feature_id}_axis"
                operations.append(CADOperation(
                    operation_id=f"OP{next_index:02d}", operation_type="cyl_boss", inputs=["main_body"],
                    params={
                        "diameter": _numeric_target(spec, f"C_{boss_id}_DIAMETER", MeasurementType.DIAMETER, feature_id=feature_id),
                        "height": _numeric_target(spec, f"C_{boss_id}_HEIGHT", MeasurementType.FEATURE_HEIGHT, feature_id=feature_id),
                        "x": _numeric_target(spec, f"C_{boss_id}_X", MeasurementType.POSITION_X, feature_id=axis_id),
                        "y": _numeric_target(spec, f"C_{boss_id}_Y", MeasurementType.POSITION_Y, feature_id=axis_id),
                    },
                    outputs=[
                        OperationOutput(feature_id=feature_id, kind=FeatureKind.FEATURE, semantic_role="cylindrical_boss"),
                        OperationOutput(feature_id=axis_id, kind=FeatureKind.DATUM_AXIS, semantic_role="cylindrical_boss_axis"),
                    ],
                ))
                next_index += 1
            for index in range(1, standoff_count + 1):
                standoff_id = f"ST{index:02d}"
                feature_id = standoff_feature_id(standoff_id)
                axis_id = f"{feature_id}_axis"
                hole_id = f"{feature_id}_hole"
                constraint_ids = {item.constraint_id for item in spec.constraints}
                has_hole = f"C_{standoff_id}_HOLE_DIAMETER" in constraint_ids
                params = {
                    "outer_diameter": _numeric_target(spec, f"C_{standoff_id}_OUTER_DIAMETER", MeasurementType.OUTER_DIAMETER, feature_id=feature_id),
                    "height": _numeric_target(spec, f"C_{standoff_id}_HEIGHT", MeasurementType.FEATURE_HEIGHT, feature_id=feature_id),
                    "x": _numeric_target(spec, f"C_{standoff_id}_X", MeasurementType.POSITION_X, feature_id=axis_id),
                    "y": _numeric_target(spec, f"C_{standoff_id}_Y", MeasurementType.POSITION_Y, feature_id=axis_id),
                    "hole_diameter": _numeric_target(spec, f"C_{standoff_id}_HOLE_DIAMETER", MeasurementType.INNER_DIAMETER, feature_id=hole_id) if has_hole else None,
                }
                outputs = [
                    OperationOutput(feature_id=feature_id, kind=FeatureKind.FEATURE, semantic_role="standoff"),
                    OperationOutput(feature_id=axis_id, kind=FeatureKind.DATUM_AXIS, semantic_role="standoff_axis"),
                ]
                if has_hole:
                    outputs.extend([
                        OperationOutput(feature_id=hole_id, kind=FeatureKind.FEATURE, semantic_role="standoff_hole"),
                        OperationOutput(feature_id=f"{hole_id}_axis", kind=FeatureKind.DATUM_AXIS, semantic_role="standoff_hole_axis"),
                    ])
                operations.append(CADOperation(
                    operation_id=f"OP{next_index:02d}", operation_type="standoff", inputs=["main_body"],
                    params=params, outputs=outputs,
                ))
                next_index += 1
            constraint_ids = {item.constraint_id for item in spec.constraints}
            if "C_FILLET_RADIUS" in constraint_ids and "C_CHAMFER_DISTANCE" in constraint_ids:
                raise ValueError("DesignSpec cannot request fillet and chamfer together")
            if "C_FILLET_RADIUS" in constraint_ids:
                _boolean_target(
                    spec, "C_FILLET_EXISTS", MeasurementType.FEATURE_EXISTS, True,
                    feature_id="plate_fillet",
                )
                operations.append(CADOperation(
                    operation_id=f"OP{next_index:02d}",
                    operation_type="fillet",
                    inputs=["main_body"],
                    params={"radius": _numeric_target(
                        spec, "C_FILLET_RADIUS", MeasurementType.FILLET_RADIUS,
                        feature_id="plate_fillet",
                    )},
                    outputs=[OperationOutput(
                        feature_id="plate_fillet",
                        kind=FeatureKind.FEATURE,
                        semantic_role="outer_vertical_edge_fillet",
                    )],
                ))
            elif "C_CHAMFER_DISTANCE" in constraint_ids:
                _boolean_target(
                    spec, "C_CHAMFER_EXISTS", MeasurementType.FEATURE_EXISTS, True,
                    feature_id="plate_chamfer",
                )
                operations.append(CADOperation(
                    operation_id=f"OP{next_index:02d}",
                    operation_type="chamfer",
                    inputs=["main_body"],
                    params={"distance": _numeric_target(
                        spec, "C_CHAMFER_DISTANCE", MeasurementType.CHAMFER_DISTANCE,
                        feature_id="plate_chamfer",
                    )},
                    outputs=[OperationOutput(
                        feature_id="plate_chamfer",
                        kind=FeatureKind.FEATURE,
                        semantic_role="outer_vertical_edge_chamfer",
                    )],
                ))
            plan = CADPlan(
                design_id=spec.design_id,
                spec_version=spec.spec_version,
                revision_id="R01",
                operations=operations,
            )
            return CapabilityPlanResult(status=GenerationStatus.SUCCESS, cad_plan=plan, run=run())
        except (KeyError, TypeError, ValueError) as exc:
            return CapabilityPlanResult(
                status=GenerationStatus.ERROR,
                run=run(),
                error=GenerationError(
                    code=GenerationErrorCode.UNSUPPORTED_SPEC_SUBSET,
                    message=str(exc),
                ),
            )


def canonical_holes(intent: PlateIntent) -> list[tuple[str, HoleIntent]]:
    ordered = sorted(intent.holes, key=lambda item: (item.x, item.y, item.diameter))
    return [(f"H{index:02d}", hole) for index, hole in enumerate(ordered, start=1)]


def canonical_rectangular_pockets(intent: PlateIntent) -> list[tuple[str, RectangularPocketIntent]]:
    ordered = sorted(
        intent.rectangular_pockets,
        key=lambda item: (item.x, item.y, item.width, item.depth, item.through, item.cut_depth or 0.0),
    )
    return [(f"RP{index:02d}", pocket) for index, pocket in enumerate(ordered, start=1)]


def canonical_circular_pockets(intent: PlateIntent) -> list[tuple[str, CircularPocketIntent]]:
    ordered = sorted(
        intent.circular_pockets,
        key=lambda item: (item.x, item.y, item.diameter, item.through, item.cut_depth or 0.0),
    )
    return [(f"CP{index:02d}", pocket) for index, pocket in enumerate(ordered, start=1)]


def canonical_slots(intent: PlateIntent) -> list[tuple[str, SlotIntent]]:
    ordered = sorted(
        intent.slots,
        key=lambda item: (item.x, item.y, item.angle_deg, item.length, item.width, item.through, item.cut_depth or 0.0),
    )
    return [(f"S{index:02d}", slot) for index, slot in enumerate(ordered, start=1)]


def canonical_rectangular_bosses(intent: PlateIntent) -> list[tuple[str, RectangularBossIntent]]:
    ordered = sorted(intent.rectangular_bosses, key=lambda item: (item.x, item.y, item.width, item.depth, item.height))
    return [(f"RB{index:02d}", item) for index, item in enumerate(ordered, start=1)]


def canonical_cylindrical_bosses(intent: PlateIntent) -> list[tuple[str, CylindricalBossIntent]]:
    ordered = sorted(intent.cylindrical_bosses, key=lambda item: (item.x, item.y, item.diameter, item.height))
    return [(f"CB{index:02d}", item) for index, item in enumerate(ordered, start=1)]


def canonical_standoffs(intent: PlateIntent) -> list[tuple[str, StandoffIntent]]:
    ordered = sorted(intent.standoffs, key=lambda item: (item.x, item.y, item.outer_diameter, item.height, item.hole_diameter or 0.0))
    return [(f"ST{index:02d}", item) for index, item in enumerate(ordered, start=1)]


def hole_feature_id(hole_id: str) -> str:
    return f"hole_{hole_id.lower()}"


def hole_axis_id(hole_id: str) -> str:
    return f"{hole_feature_id(hole_id)}_axis"


def rectangular_pocket_feature_id(pocket_id: str) -> str:
    return f"rect_pocket_{pocket_id.lower()}"


def circular_pocket_feature_id(pocket_id: str) -> str:
    return f"circle_pocket_{pocket_id.lower()}"


def slot_feature_id(slot_id: str) -> str:
    return f"slot_{slot_id.lower()}"


def rectangular_boss_feature_id(boss_id: str) -> str:
    return f"rect_boss_{boss_id.lower()}"


def cylindrical_boss_feature_id(boss_id: str) -> str:
    return f"cyl_boss_{boss_id.lower()}"


def standoff_feature_id(standoff_id: str) -> str:
    return f"standoff_{standoff_id.lower()}"


def capability_repair_rules(spec: DesignSpec, plan: CADPlan) -> list[RepairRule]:
    rules = [
        RepairRule(constraint_id="C_WIDTH", operation_id="OP01", parameter_name="x", related_feature_ids=["main_body"]),
        RepairRule(constraint_id="C_DEPTH", operation_id="OP01", parameter_name="y", related_feature_ids=["main_body"]),
        RepairRule(constraint_id="C_HEIGHT", operation_id="OP01", parameter_name="z", related_feature_ids=["main_body"]),
    ]
    operations = {item.operation_id: item for item in plan.operations}
    for operation in plan.operations:
        if operation.operation_type == "hole":
            feature_id = next(output.feature_id for output in operation.outputs if output.kind == FeatureKind.FEATURE)
            feature_key = feature_id.removeprefix("hole_").upper()
            center_id = hole_axis_id(feature_key)
            mapping = (("DIAMETER", "diameter", feature_id), ("X", "x", center_id), ("Y", "y", center_id))
        elif operation.operation_type == "rect_pocket":
            feature_id = next(output.feature_id for output in operation.outputs if output.kind == FeatureKind.FEATURE)
            feature_key = feature_id.removeprefix("rect_pocket_").upper()
            center_id = f"{feature_id}_center"
            mapping = (("WIDTH", "width", feature_id), ("DEPTH", "depth", feature_id), ("X", "x", center_id), ("Y", "y", center_id))
            if not operation.params["through"]:
                mapping += (("CUT_DEPTH", "cut_depth", feature_id),)
        elif operation.operation_type == "circular_pocket":
            feature_id = next(output.feature_id for output in operation.outputs if output.kind == FeatureKind.FEATURE)
            feature_key = feature_id.removeprefix("circle_pocket_").upper()
            center_id = f"{feature_id}_center"
            mapping = (("DIAMETER", "diameter", feature_id), ("X", "x", center_id), ("Y", "y", center_id))
            if not operation.params["through"]:
                mapping += (("CUT_DEPTH", "cut_depth", feature_id),)
        elif operation.operation_type == "slot":
            feature_id = next(output.feature_id for output in operation.outputs if output.kind == FeatureKind.FEATURE)
            feature_key = feature_id.removeprefix("slot_").upper()
            center_id = f"{feature_id}_axis"
            mapping = (("LENGTH", "length", feature_id), ("WIDTH", "width", feature_id), ("X", "x", center_id), ("Y", "y", center_id))
            if not operation.params["through"]:
                mapping += (("CUT_DEPTH", "cut_depth", feature_id),)
        elif operation.operation_type == "rect_boss":
            feature_id = next(output.feature_id for output in operation.outputs if output.semantic_role == "rectangular_boss")
            feature_key = feature_id.removeprefix("rect_boss_").upper()
            center_id = f"{feature_id}_center"
            mapping = (("WIDTH", "width", feature_id), ("DEPTH", "depth", feature_id), ("HEIGHT", "height", feature_id), ("X", "x", center_id), ("Y", "y", center_id))
        elif operation.operation_type == "cyl_boss":
            feature_id = next(output.feature_id for output in operation.outputs if output.semantic_role == "cylindrical_boss")
            feature_key = feature_id.removeprefix("cyl_boss_").upper()
            axis_id = f"{feature_id}_axis"
            mapping = (("DIAMETER", "diameter", feature_id), ("HEIGHT", "height", feature_id), ("X", "x", axis_id), ("Y", "y", axis_id))
        elif operation.operation_type == "standoff":
            feature_id = next(output.feature_id for output in operation.outputs if output.semantic_role == "standoff")
            feature_key = feature_id.removeprefix("standoff_").upper()
            axis_id = f"{feature_id}_axis"
            mapping = (("OUTER_DIAMETER", "outer_diameter", feature_id), ("HEIGHT", "height", feature_id), ("X", "x", axis_id), ("Y", "y", axis_id))
            if operation.params.get("hole_diameter") is not None:
                mapping += (("HOLE_DIAMETER", "hole_diameter", f"{feature_id}_hole"),)
        else:
            continue
        rules.extend(
            RepairRule(
                constraint_id=f"C_{feature_key}_{suffix}",
                operation_id=operation.operation_id,
                parameter_name=parameter,
                related_feature_ids=[related_id],
            )
            for suffix, parameter, related_id in mapping
        )
    for operation in operations.values():
        if operation.operation_type == "fillet":
            rules.append(RepairRule(
                constraint_id="C_FILLET_RADIUS",
                operation_id=operation.operation_id,
                parameter_name="radius",
                related_feature_ids=["plate_fillet"],
            ))
        elif operation.operation_type == "chamfer":
            rules.append(RepairRule(
                constraint_id="C_CHAMFER_DISTANCE",
                operation_id=operation.operation_id,
                parameter_name="distance",
                related_feature_ids=["plate_chamfer"],
            ))
    return rules


def _subtractive_constraints(
    feature_key: str,
    feature_id: str,
    center_id: str,
    *,
    width: float,
    depth: float,
    x: float,
    y: float,
    cut_depth: float | None,
    through: bool,
) -> list[Constraint]:
    constraints = [
        _constraint(f"C_{feature_key}_EXISTS", MeasurementType.FEATURE_EXISTS, True, Origin.DERIVED, feature_id=feature_id, tolerance=0.0),
        _constraint(f"C_{feature_key}_WIDTH", MeasurementType.FEATURE_WIDTH, width, Origin.USER, feature_id=feature_id),
        _constraint(f"C_{feature_key}_DEPTH", MeasurementType.FEATURE_DEPTH, depth, Origin.USER, feature_id=feature_id),
        _constraint(f"C_{feature_key}_X", MeasurementType.POSITION_X, x, Origin.USER, feature_id=center_id),
        _constraint(f"C_{feature_key}_Y", MeasurementType.POSITION_Y, y, Origin.USER, feature_id=center_id),
        _constraint(f"C_{feature_key}_THROUGH", MeasurementType.THROUGH, through, Origin.USER, feature_id=feature_id, tolerance=0.0),
    ]
    if cut_depth is not None:
        constraints.append(_constraint(
            f"C_{feature_key}_CUT_DEPTH", MeasurementType.CUT_DEPTH, cut_depth,
            Origin.USER, feature_id=feature_id,
        ))
    return constraints


def _additive_constraints(
    feature_key: str,
    feature_id: str,
    center_id: str,
    *,
    width: float,
    depth: float,
    height: float,
    x: float,
    y: float,
) -> list[Constraint]:
    return [
        _constraint(f"C_{feature_key}_EXISTS", MeasurementType.FEATURE_EXISTS, True, Origin.DERIVED, feature_id=feature_id, tolerance=0.0),
        _constraint(f"C_{feature_key}_WIDTH", MeasurementType.FEATURE_WIDTH, width, Origin.USER, feature_id=feature_id),
        _constraint(f"C_{feature_key}_DEPTH", MeasurementType.FEATURE_DEPTH, depth, Origin.USER, feature_id=feature_id),
        _constraint(f"C_{feature_key}_HEIGHT", MeasurementType.FEATURE_HEIGHT, height, Origin.USER, feature_id=feature_id),
        _constraint(f"C_{feature_key}_X", MeasurementType.POSITION_X, x, Origin.USER, feature_id=center_id),
        _constraint(f"C_{feature_key}_Y", MeasurementType.POSITION_Y, y, Origin.USER, feature_id=center_id),
    ]


def _constraint(
    constraint_id: str,
    measurement_type: MeasurementType,
    target: float | bool,
    origin: Origin,
    *,
    feature_id: str | None = None,
    tolerance: float = 0.01,
) -> Constraint:
    return Constraint(
        constraint_id=constraint_id,
        priority=Priority.HARD,
        origin=origin,
        measurement=Measurement(type=measurement_type, feature_id=feature_id),
        comparison=Comparison.EQ,
        target=target,
        tolerance=tolerance,
    )


def _numeric_target(
    spec: DesignSpec,
    constraint_id: str,
    measurement_type: MeasurementType,
    *,
    feature_id: str | None = None,
    expected_value: float | None = None,
) -> float:
    matches = [item for item in spec.constraints if item.constraint_id == constraint_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {constraint_id} constraint")
    constraint = matches[0]
    if (
        constraint.priority != Priority.HARD
        or constraint.comparison != Comparison.EQ
        or constraint.measurement.type != measurement_type
        or constraint.measurement.feature_id != feature_id
        or constraint.measurement.reference_feature_id is not None
        or isinstance(constraint.target, bool)
        or not isinstance(constraint.target, (int, float))
        or not math.isfinite(float(constraint.target))
    ):
        raise ValueError(f"{constraint_id} must be a finite numeric HARD EQ constraint")
    value = float(constraint.target)
    if expected_value is not None and value != expected_value:
        raise ValueError(f"{constraint_id} must target {expected_value}")
    return value


def _boolean_target(
    spec: DesignSpec,
    constraint_id: str,
    measurement_type: MeasurementType,
    expected_value: bool,
    *,
    feature_id: str | None = None,
) -> None:
    matches = [item for item in spec.constraints if item.constraint_id == constraint_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {constraint_id} constraint")
    constraint = matches[0]
    if (
        constraint.priority != Priority.HARD
        or constraint.comparison != Comparison.EQ
        or constraint.measurement.type != measurement_type
        or constraint.measurement.feature_id != feature_id
        or constraint.measurement.reference_feature_id is not None
        or constraint.target is not expected_value
    ):
        raise ValueError(f"{constraint_id} must be the canonical boolean HARD EQ constraint")


def _boolean_value(
    spec: DesignSpec,
    constraint_id: str,
    measurement_type: MeasurementType,
    *,
    feature_id: str | None = None,
) -> bool:
    matches = [item for item in spec.constraints if item.constraint_id == constraint_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {constraint_id} constraint")
    constraint = matches[0]
    if (
        constraint.priority != Priority.HARD
        or constraint.comparison != Comparison.EQ
        or constraint.measurement.type != measurement_type
        or constraint.measurement.feature_id != feature_id
        or constraint.measurement.reference_feature_id is not None
        or not isinstance(constraint.target, bool)
    ):
        raise ValueError(f"{constraint_id} must be a boolean HARD EQ constraint")
    return constraint.target


def _validate_geometry(
    intent: PlateIntent,
    holes: list[tuple[str, HoleIntent]],
    rectangular: list[tuple[str, RectangularPocketIntent]],
    circular: list[tuple[str, CircularPocketIntent]],
    slots: list[tuple[str, SlotIntent]],
    rectangular_bosses: list[tuple[str, RectangularBossIntent]],
    cylindrical_bosses: list[tuple[str, CylindricalBossIntent]],
    standoffs: list[tuple[str, StandoffIntent]],
) -> GenerationError | None:
    max_treatment = min(intent.width, intent.depth) / 2.0
    treatment = intent.fillet_radius if intent.fillet_radius is not None else intent.chamfer_distance
    if treatment is not None and treatment >= max_treatment:
        return GenerationError(
            code=GenerationErrorCode.INVALID_PARAMETER_VALUE,
            message="edge treatment must be smaller than half both plate extents",
        )
    for hole_id, hole in holes:
        radius = hole.diameter / 2.0
        if abs(hole.x) + radius >= intent.width / 2.0 or abs(hole.y) + radius >= intent.depth / 2.0:
            return GenerationError(
                code=GenerationErrorCode.INVALID_PARAMETER_VALUE,
                message=f"{hole_id} does not fit inside the plate",
            )
    for index, (left_id, left) in enumerate(holes):
        for right_id, right in holes[index + 1:]:
            if math.hypot(left.x - right.x, left.y - right.y) <= (left.diameter + right.diameter) / 2.0:
                return GenerationError(
                    code=GenerationErrorCode.INVALID_PARAMETER_VALUE,
                    message=f"{left_id} and {right_id} overlap",
                )
    for pocket_id, pocket in rectangular:
        if not _rectangle_fits(intent.width, intent.depth, pocket.x, pocket.y, pocket.width, pocket.depth):
            return GenerationError(
                code=GenerationErrorCode.INVALID_PARAMETER_VALUE,
                message=f"{pocket_id} does not fit inside the base",
            )
        if not pocket.through and (pocket.cut_depth is None or pocket.cut_depth >= intent.height):
            return GenerationError(
                code=GenerationErrorCode.INVALID_PARAMETER_VALUE,
                message=f"{pocket_id} has invalid cut depth",
            )
    for pocket_id, pocket in circular:
        if not _circle_fits(intent.width, intent.depth, pocket.x, pocket.y, pocket.diameter):
            return GenerationError(
                code=GenerationErrorCode.INVALID_PARAMETER_VALUE,
                message=f"{pocket_id} does not fit inside the base",
            )
        if not pocket.through and (pocket.cut_depth is None or pocket.cut_depth >= intent.height):
            return GenerationError(
                code=GenerationErrorCode.INVALID_PARAMETER_VALUE,
                message=f"{pocket_id} has invalid cut depth",
            )
    for slot_id, slot in slots:
        footprint_x = slot.length if slot.angle_deg == 0.0 else slot.width
        footprint_y = slot.width if slot.angle_deg == 0.0 else slot.length
        if not _rectangle_fits(intent.width, intent.depth, slot.x, slot.y, footprint_x, footprint_y):
            return GenerationError(
                code=GenerationErrorCode.INVALID_PARAMETER_VALUE,
                message=f"{slot_id} does not fit inside the base",
            )
        if not slot.through and (slot.cut_depth is None or slot.cut_depth >= intent.height):
            return GenerationError(
                code=GenerationErrorCode.INVALID_PARAMETER_VALUE,
                message=f"{slot_id} has invalid cut depth",
            )
    additive_footprints: list[tuple[str, float, float, float, float]] = []
    for boss_id, boss in rectangular_bosses:
        if not _rectangle_fits(intent.width, intent.depth, boss.x, boss.y, boss.width, boss.depth):
            return GenerationError(code=GenerationErrorCode.INVALID_PARAMETER_VALUE, message=f"{boss_id} does not fit inside the base")
        additive_footprints.append(("RECT", boss.x, boss.y, boss.width, boss.depth))
    for boss_id, boss in cylindrical_bosses:
        if not _circle_fits(intent.width, intent.depth, boss.x, boss.y, boss.diameter):
            return GenerationError(code=GenerationErrorCode.INVALID_PARAMETER_VALUE, message=f"{boss_id} does not fit inside the base")
        additive_footprints.append(("CIRCLE", boss.x, boss.y, boss.diameter, boss.diameter))
    for standoff_id, standoff in standoffs:
        if not _circle_fits(intent.width, intent.depth, standoff.x, standoff.y, standoff.outer_diameter):
            return GenerationError(code=GenerationErrorCode.INVALID_PARAMETER_VALUE, message=f"{standoff_id} does not fit inside the base")
        additive_footprints.append(("CIRCLE", standoff.x, standoff.y, standoff.outer_diameter, standoff.outer_diameter))
    if any(_footprints_overlap(first, second) for index, first in enumerate(additive_footprints) for second in additive_footprints[index + 1:]):
        return GenerationError(code=GenerationErrorCode.INVALID_PARAMETER_VALUE, message="additive feature footprints overlap")
    return None


def _backend_error(code: BackendFailureCode | None, message: str | None) -> GenerationError:
    mapping = {
        BackendFailureCode.MODEL_TIMEOUT: GenerationErrorCode.MODEL_TIMEOUT,
        BackendFailureCode.MODEL_UNAVAILABLE: GenerationErrorCode.MODEL_UNAVAILABLE,
        BackendFailureCode.MODEL_ERROR: GenerationErrorCode.MODEL_ERROR,
    }
    return GenerationError(
        code=mapping.get(code, GenerationErrorCode.MODEL_ERROR),
        message=message or "inference backend failed",
    )
