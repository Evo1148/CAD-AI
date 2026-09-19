# Architecture

CAD AI separates **interpretation**, **authorization**, **execution** and **validation**.

That separation is the core safety and reliability property of the project.

## 1. Prompt grounding

Natural-language input first goes through deterministic grounding.

The goal is to extract facts that can be established without an LLM and preserve their provenance.

```text
User Prompt
    ↓
DeterministicPromptGrounder
    ↓
ExtractionCoverage
```

Coverage is classified as:

- `COMPLETE` — all required facts can be assembled deterministically;
- `PARTIAL` — some unresolved facts may be proposed by a residual LLM extractor;
- `INSUFFICIENT` — the request does not contain enough information to authorize CAD generation.

## 2. Partial extraction

For `PARTIAL` requests, the LLM only proposes unresolved facts.

```text
Grounded facts
      +
ResidualLLMExtractor
      ↓
ResidualFacts
      ↓
DeterministicFactAssembler
      ↓
FactGroundingValidator
```

Grounded facts have priority and cannot be overwritten by residual model output.

## 3. Intent authorization

Canonical facts are not yet CAD instructions.

They pass through a deterministic intent gate that decides whether the requested design belongs to the supported capability surface.

```text
Canonical Facts
      ↓
Deterministic Intent Gate
      ↓
Intent
```

A schema-valid model response is not sufficient to authorize execution.

## 4. Planning

Authorized intent is transformed deterministically:

```text
Intent
  ↓
DesignSpec
  ↓
CADPlan
```

`DesignSpec` represents what the object must be.

`CADPlan` represents how the supported CAD engine will build it.

This boundary makes it possible to validate design intent separately from implementation details.

## 5. CAD execution

The CAD engine executes a validated `CADPlan` with programmatic geometry operations.

Current technology:

- CadQuery
- OpenCascade
- STEP output
- STL output

The LLM never receives authority to invoke arbitrary CAD operations.

## 6. Validation

Generated geometry is checked independently.

Validation can include:

- dimensions;
- tolerances;
- feature presence;
- constraint preservation;
- geometric relationships;
- output integrity.

A generated model is not considered successful merely because the CAD engine produced a solid.

## 7. Repair

A failed validation produces structured repair context.

```text
Validator FAIL
      ↓
Repair context
      ↓
Repair planner
      ↓
RepairPlan
      ↓
RepairPlanValidator
      ↓
RepairExecutor
      ↓
New revision
      ↓
Validator
```

The repair planner may be deterministic or LLM-backed.

In both cases:

- the planner proposes;
- the validator authorizes;
- the executor performs deterministic operations.

Repair is expected to preserve unrelated constraints and minimize change locality.

## 8. Model boundary

The architecture is model-agnostic.

Local or remote models can be swapped behind an inference backend as long as they satisfy the same structured contract.

Model quality affects interpretation/planning quality, but does not change the authority boundaries of the system.
