<p align="center">
  <strong>🇬🇧 English</strong> · <a href="./README.es.md">🇪🇸 Español</a>
</p>

<h1 align="center">CAD AI</h1>

<p align="center">
  <a href="https://github.com/Evo1148/CAD-AI/actions/workflows/ci.yml">
    <img src="https://github.com/Evo1148/CAD-AI/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
</p>

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

The system therefore separates interpretation, authorization, execution and validation.

Natural-language requirements are converted into structured facts, passed through deterministic gates, transformed into a formal design specification and CAD plan, executed by a parametric CAD engine, and checked by an independent validator. If validation fails, a constrained repair loop can propose a correction without giving the LLM unrestricted control over geometry.

> **Status:** active development. The public source currently includes **CAD AI V0.2**, Capability Packs 1–3, the Prototype Usability Gate, LAB-001, benchmark tooling and the automated test suite.

## Source

- 🧠 [Core package](./src/cad_ai/)
- 🧪 [Tests](./tests/)
- 📐 [V0.2 capability layer](./src/cad_ai/capability_v02.py)
- 🧭 [Prompt grounding](./src/cad_ai/prompt_grounding.py)
- 🚪 [Prototype usability gate](./src/cad_ai/prototype_gate.py)
- 🔬 [LAB runner](./src/cad_ai/lab.py)
- 📊 [Benchmark tooling](./src/cad_ai/benchmark/)
- 🏗️ [Architecture](./docs/architecture.md)
- 🧱 [Engineering principles](./docs/principles.md)
- 📜 [Development history](./docs/development-history.es.md)

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

## Current public capability surface

The current V0.2 source extends the original plate vertical slice through three capability packs.

**Capability Pack 1** adds explicit through holes, linear hole patterns, fillets and chamfers.

**Capability Pack 2** adds rectangular/circular pockets, through cutouts and straight slots while preserving the same deterministic authorization boundary.

**Capability Pack 3** adds controlled additive features on the top support face, including rectangular bosses, cylindrical bosses, standoffs and linear patterns.

The public repository also includes:

- deterministic and hybrid extraction paths;
- controlled local-LLM integration;
- formal `DesignSpec` / `CADPlan` contracts;
- CadQuery / OpenCascade execution;
- independent geometry validation;
- constrained repair planning and execution;
- Prototype Usability Gate evaluation;
- LAB-001 real repair case;
- regression tests and benchmark tooling.

## Example repair flow

```text
R01
 └─ Validator
      └─ FAIL: dimensional constraint outside tolerance
             │
             ▼
       Repair context
             │
             ▼
       Repair planner
             │
             ▼
       RepairPlan
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
| Language | Python 3.11–3.12 |
| CAD | CadQuery · OpenCascade |
| Contracts | Pydantic |
| AI | Local LLMs · structured output · JSON Schema |
| Inference experiments | llama.cpp · Vulkan |
| Validation | Deterministic geometry / constraint checks |
| Testing | pytest · regression suites · benchmarks |
| Outputs | STEP · STL |

## Repository layout

```text
CAD-AI/
├── src/
│   └── cad_ai/
│       ├── benchmark/
│       ├── capability_v02.py
│       ├── prompt_grounding.py
│       ├── prototype_gate.py
│       ├── lab.py
│       ├── planning.py
│       ├── generation.py
│       ├── validation.py
│       └── ...
├── tests/
├── docs/
├── assets/
├── tools/
├── pyproject.toml
├── uv.lock
├── README.md
└── README.es.md
```

Generated CAD outputs, local model weights, virtual environments and benchmark run artifacts are intentionally excluded from version control.

## Installation

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[test]"
```

Run the test suite:

```powershell
pytest
```

CadQuery brings OpenCascade through its OCP stack. On platforms where its wheels do not resolve cleanly, a compatible Conda environment may be preferable.

## Evaluation philosophy

The project distinguishes between unit tests, CAD benchmarks, repair benchmarks, regression tests and model benchmarks.

Useful metrics include validation pass rate, repair success rate, first-pass success, average revisions, constraint preservation, change locality, determinism, latency and resource use.

## Why local-first?

The architecture is deliberately model-agnostic. Local models provide privacy, reproducibility, offline experimentation and stable benchmark conditions. Remote models can also be supported, but no model is treated as a source of geometric truth.

## License

A license has not been selected yet. Until one is added, the repository remains under default copyright rules.

---

Built in Spain 🇪🇸 as a personal engineering and AI/CAD research project.
