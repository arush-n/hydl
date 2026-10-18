# belief/ — the shared belief B_t

The one belief object every level reads (ADR-001 D6: B_t is shared, built from legal observation only, and
is no level's private latent). Written by the ADK `BeliefCore` here, published into the scratchpad's belief
section by the session, read by every decision instance (`jev/`) and by the skills.

| File | Lines | What it is |
|---|---:|---|
| `core.py` | 465 | `SharedBelief`, the ADK `BeliefCore`: carries the per-lane features, their `known` flags and a 64-wide latent through the hierarchy step. |
| `features.py` | 265 | The instantaneous features (`BELIEF_FEATURE_NAMES`, 10 per lane: target visibility, distance, bearing, health fractions, ticks since damage, ticks since the target was seen, ...) read from the legal structured observation. Tick counters are published as `counter / TICKS_SATURATION` (256) and multiplied back by whoever evaluates a predicate (`jev/programs.py` `TICK_COLUMNS`). |
| `__init__.py` | 39 | Exports. |

Gate: `tests/experimental/neural_agent/test_na_belief.py`. Formulas: `../docs/FORMULAS.md` section B; the
normalisation is this module's decision (2026-10-16, `SKILL.md`, gitignored notes).

Consumers that depend on the feature order: `contracts.feature_index`, `jev/system_one/data.py` (rows in
physical units), `jev/programs.py` and `jev/typed_rule.py` (predicates), `shared_state.BeliefSection`.
Changing a feature name or index is a contract change: refresh `contracts.py`'s pins and re-run the whole
`test_na_*` tree.

Limits: no memory beyond the counters and the latent; one target per lane (the request stream names it as
entity 0, `jev/programs.py` item 9).
