# Engineering Principles

## LLMs are not execution engines

An LLM can interpret ambiguity or propose a repair, but it is not allowed to directly mutate geometry.

## Deterministic evidence wins

Facts established deterministically are authoritative and cannot be overwritten by residual model output.

## Schema-valid is not authorized

Structured output only proves that a response matches a schema. Separate deterministic validation decides whether an operation is supported and allowed.

## Intent is gated

The supported CAD capability surface is controlled by a deterministic intent gate.

Unsupported intent should fail explicitly rather than being improvised.

## Validation is independent

Generation and validation are different responsibilities.

A CAD engine producing a solid does not prove that the design satisfies the user's constraints.

## Repair is constrained

Repair proposals are checked before execution.

The executor performs only explicitly supported operations and should preserve unrelated constraints.

## Prefer local change

A repair should modify the smallest possible part of the design required to resolve the failure.

Useful repair metrics include:

- change locality;
- constraint preservation;
- exact target selection;
- revision count.

## Insufficient information means no CAD

The system should return a structured insufficiency error instead of inventing missing dimensions or features.

## Failures become tests

Important corrected failures should become regression tests so the same class of bug does not silently return.

## Reproducibility matters

Benchmarks should keep inputs, schemas, model settings and evaluation criteria stable when comparing changes.

Change one meaningful variable at a time.
