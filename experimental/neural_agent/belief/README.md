# belief/ — the shared belief B_t

The one belief object every level reads (ADR-001 D6: B_t is shared, built from legal observation only, and
is no level's private latent). Written by the ADK `BeliefCore` here, published into the scratchpad's belief
section by the session, read by every decision instance (`jev/`) and by the skills.

| File | Lines | What it is |
|---|---:|---|
| `core.py` | 481 | `SharedBelief`, the ADK `BeliefCore`: carries the per-lane features, their `known` flags, a 64-wide latent and the remembered target position (task #96) through the hierarchy step. |
| `features.py` | 277 | The instantaneous features (`BELIEF_FEATURE_NAMES`, 10 per lane: target visibility, distance, bearing, health fractions, ticks since damage, ticks since the target was seen, ...) read from the legal structured observation. Tick counters are published as `counter / TICKS_SATURATION` (256) and multiplied back by whoever evaluates a predicate (`jev/programs.py` `TICK_COLUMNS`). |
| `target_memory.py` | 141 | The remembered target position (task #96, 2026-10-19). Recorded from a sighting only while the gen-7 perception channel is present: the channel's self position plus the target row's body-frame offset, rotated by the channel's body yaw. Held while the target is unseen (`ticks_since_target_seen` gives its age). `target_arguments` is the request-side rule (position while unseen, for verbs whose schema takes entity-or-position). That schema is proposed, not in effect: `requests.ARGUMENT_SCHEMAS` keeps the entity flag for pursuit, evader and keep_range until the skill layer lowers `any_of` groups without losing the entity check (task #96 report). |
| `__init__.py` | 39 | Exports. |

Gate: `tests/experimental/neural_agent/test_na_belief.py`. Formulas: `../docs/FORMULAS.md` section B; the
normalisation is this module's decision (2026-10-16, `SKILL.md`, gitignored notes).

Consumers that depend on the feature order: `contracts.feature_index`, `jev/system_one/data.py` (rows in
physical units), `jev/programs.py` and `jev/typed_rule.py` (predicates), `shared_state.BeliefSection`.
Changing a feature name or index is a contract change: refresh `contracts.py`'s pins and re-run the whole
`test_na_*` tree.

Limits: one target per lane (the request stream names it as entity 0, `jev/programs.py` item 9). Beyond the counters and
the latent, the only memory is the remembered position
(`target_memory.py`, task #96): it is set only with the gen-7 perception channel, and it is held until the next sighting.
