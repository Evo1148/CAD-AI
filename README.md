<p align="center">
  <strong>🇬🇧 English</strong> · <a href="./README.es.md">🇪🇸 Español</a>
</p>

<h1 align="center">CAD AI</h1>

<p align="center">
  A local-first system for turning natural-language requirements into
  <strong>validated parametric CAD</strong> through deterministic grounding,
  structured planning and controlled automated repair.
</p>

<p align="center">
  <strong>Natural language → grounded facts → DesignSpec → CADPlan → geometry → validation → repair → STEP/STL</strong>
</p>

---

## Overview

CAD AI explores a simple idea: an LLM can help interpret design intent, but it should **not** be the authority that directly creates or approves geometry.

Instead, the system separates interpretation from execution.

Natural-language requirements are converted into structured facts, passed through deterministic gates, transformed into a formal design specification and CAD plan, executed by a parametric CAD engine, and then checked by an independent validator. If validation fails, a constrained repair loop can propose a correction without giving the LLM unrestricted control over geometry.

> **Status:** active development. The current prototype architecture is functional and covered by automated tests and benchmark cases. The source repository is being prepared for a clean public import.

## Core pipeline

```text
User Prompt
    │
    ▼
DeterministicPromptGrounder
    │
    ▼
ExtractionCoverage
    │
    ├── COMPLETE ─────────► DeterministicFactAssembler
    │
    ├── PARTIAL ──────────► ResidualLLMExtractor
    │                           │
    │                           ▼
    │                      ResidualFacts
    │                           │
    │                           ▼
    │                  DeterministicFactAssembler
    │                           │
    │                           ▼
    │                  FactGroundingValidator
    │
    └── INSUFFICIENT ─────► Structured error / no CAD
                               
Canonical Extracted Facts
    │
    ▼
Deterministic Intent Gate
    │
    ▼
Intent
    │
    ▼
DesignSpec
    │
    ▼
CADPlan
    │
    ▼
CAD Engine
    │
    ▼
Validator
    │
    ├── PASS ─────────────► STEP / STL
    │
    └── FAIL
         │
         ▼
    Repair Planner
         │
         ▼
    RepairPlanValidator
         │
         ▼
    Repair Executor
         │
         └───────────────► re-validate
```

A more detailed description is available in [docs/architecture.md](./docs/architecture.md).

## Design principles

CAD AI is built around a few strict rules:

- **The LLM proposes and interprets; it never executes geometry directly.**
- **Deterministic evidence has priority over model output.**
- Grounded facts cannot be overwritten by residual LLM extraction.
- **Schema-valid does not mean authorized.**
- The deterministic Intent Gate remains the authority on supported design intent.
- `DesignSpec → CADPlan` is deterministic.
- Validation is independent from generation.
- Repair operations are constrained by a validator and deterministic executor.
- Insufficient information produces a structured error rather than speculative CAD.
- Important failures become regression tests.

See [docs/principles.md](./docs/principles.md).

## Current capabilities

The current prototype focuses on parametric part generation and controlled feature editing.

Implemented or validated areas include:

- deterministic prompt grounding;
- complete / partial / insufficient extraction coverage;
- residual LLM extraction for unresolved facts;
- canonical fact assembly and grounding validation;
- deterministic intent gating;
- formal `DesignSpec` and `CADPlan`;
- CAD generation with CadQuery / OpenCascade;
- independent validation;
- deterministic repair execution;
- constrained LLM-assisted repair planning;
- structured output;
- local-LLM inference experiments;
- regression tests and benchmark suites;
- STEP / STL generation.

## Example repair flow

A generated revision can fail a dimensional constraint without invalidating the entire design.

```text
R01
 └─ Validator
      └─ FAIL: hole diameter outside tolerance
             │
             ▼
       Repair context
             │
             ▼
       Repair planner
             │
             ▼
       SET_PARAMETER
       target: hole diameter
             │
             ▼
       RepairPlanValidator
             │
             ▼
       Deterministic execution
             │
             ▼
            R02
             │
             ▼
        Validator PASS
```

The repair path is deliberately local: unrelated parameters and constraints must remain unchanged.

## Technology

| Area | Technologies |
| --- | --- |
| Language | Python |
| CAD | CadQuery · OpenCascade |
| AI | Local LLMs · structured output · JSON Schema |
| Inference experiments | llama.cpp · Vulkan |
| Validation | Deterministic geometry / constraint checks |
| Testing | pytest · regression suites · benchmarks |
| Outputs | STEP · STL |

## Repository layout

The public repository is being prepared around this structure:

```text
CAD-AI/
├── src/             # Core package
├── tests/           # Unit and regression tests
├── benchmarks/      # Reproducible evaluation cases
├── examples/        # Small usage examples
├── docs/            # Architecture and design documentation
├── assets/          # Diagrams and screenshots
├── pyproject.toml
├── README.md
└── README.es.md
```

The source tree will be imported only after generated files, local model artifacts, test outputs and machine-specific configuration have been reviewed.

## Evaluation philosophy

The project distinguishes between:

- **unit tests** — correctness of individual components;
- **CAD benchmarks** — whether geometry matches expected requirements;
- **repair benchmarks** — whether failures are corrected safely;
- **regression tests** — whether fixed failures stay fixed;
- **model benchmarks** — whether different models/prompts produce valid structured proposals under identical conditions.

Useful metrics include validation pass rate, repair success rate, first-pass success, average revisions, constraint preservation, change locality, determinism, latency and resource use.

## Why local-first?

The architecture is intentionally model-agnostic.

Local models are useful for:

- privacy;
- reproducibility;
- offline experimentation;
- stable benchmark conditions;
- avoiding dependence on a single provider.

The system can still support remote models, but no model is treated as a source of geometric truth.

## License

A license has not been selected yet. Until one is added, the repository remains under default copyright rules.

---

Built in Spain 🇪🇸 as a personal engineering and AI/CAD research project.
