# M3-RP-v0.2 — Repair Planner benchmark contract

M3-RP-v0.2 revises only the benchmark fixtures, goldens and evaluator. The M3
contracts, prompt, planning boundary, semantic validator and executor remain
unchanged. The historical `m3-a.1` / M3-RP-v0.1 suite remains executable.

## Authorization semantics

- `allowed_action_types` is the complete set of action kinds a planner may
  propose. In v0.2 the only supported kind is `SET_PARAMETER`.
- For `SET_PARAMETER`, the authorized targets are exactly the pairs
  `(repairable_operation.operation_id, repairable_parameter.name)` enumerated
  in `repairable_operations`. An operation's general `parameters` mapping is
  descriptive and does not authorize those parameters.
- `repairable_parameter.constraint_ids` records structured relevance between a
  target and constraints. It is not permission, target identity, or a claim
  that the constraint and target are equivalent.
- A proposal still requires authorization by the trusted
  `RepairPlanValidator`. Context-level authorization never bypasses it.

## Semantic selection and ambiguity

For a failed constraint, first retain only authorized `SET_PARAMETER` targets
related to that constraint. Stable direct metric mappings may then disambiguate:

| Measurement | Direct parameter semantic |
|---|---|
| `EXTENT_X` | `x` |
| `EXTENT_Y` | `y` |
| `EXTENT_Z` | `z` |
| `DIAMETER` | `diameter` |
| `RADIUS` | `radius` |

A target is **ambiguous** only when two or more authorized and semantically
compatible targets remain and no authoritative structured fact selects one.
Merely listing multiple related parameters is not ambiguity when the
measurement gives a unique mapping. Text in `non_authoritative_hint` never
breaks a tie and never grants authorization.

## `UNSUPPORTED.reason_code`

| Code | Meaning |
|---|---|
| `NO_ALLOWED_REPAIR` | The exposed action/target capability set contains no authorized repair for the failure. |
| `REQUIRES_REPLAN` | Satisfying the requirement needs a structural planning change outside the allowed action vocabulary. |
| `INSUFFICIENT_CONTEXT` | The repair class may be available, but required authoritative data such as a numeric target is absent. |
| `CONFLICTING_CONSTRAINTS` | Authorized repairs imply mutually incompatible values or outcomes. |
| `AMBIGUOUS_TARGET` | Multiple authorized, semantically compatible targets remain after all structured disambiguation. |

RP05 accepts `NO_ALLOWED_REPAIR` and `REQUIRES_REPLAN`: both are safe,
contractually accurate abstentions for a structural failure with no exposed
`SET_PARAMETER` target. No other reason is accepted for that case.

## Cases

- RP01–RP04: byte-for-byte construction logic and goldens retained from v0.1.
- RP05: same structural failure; evaluator accepts either safe abstention above.
- RP06A: `C_WIDTH` relates to `x` and `y`; `EXTENT_X → x` uniquely selects `x`.
- RP06B: a volume failure can be addressed through `x` or `y`; both remain
  equally compatible, so `AMBIGUOUS_TARGET` is required.
- RP07A: one target is explicitly authorized.
- RP07B: the same repair shape plus a contradictory non-authoritative hint;
  expected output is unchanged.

## Result comparison

Each run can be written without overwriting another run:

```bash
python -m cad_ai.benchmark.m3 --suite M3-RP-v0.2 --planner http-llm \
  --model-id MODEL --result-json artifacts/benchmark_m3/MODEL-v0.2.json --json
```

Compare any set of preserved v0.1/v0.2 results structurally:

```bash
python -m cad_ai.benchmark.comparison \
  artifacts/benchmark_m3/qwen-v0.1.json \
  artifacts/benchmark_m3/qwen-v0.2.json \
  artifacts/benchmark_m3/ministral-v0.1.json \
  artifacts/benchmark_m3/ministral-v0.2.json --json
```
