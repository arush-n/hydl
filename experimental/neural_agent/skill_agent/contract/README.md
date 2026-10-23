# skill_agent/contract/ — what a skill is and what it speaks

The fixed interface of the lower layer, which every other subfolder builds against: the gen-7 surface (token sets
and action heads in six scopes), the skill catalogue read into verb contracts, the request protocol shared with the
middle layer, and the resolver from a request's target to what a body sees. Nothing here is learned.

| Module | Lines | What it holds |
|---|---:|---|
| `surface.py` | 951 | The gen-7 surface: the token types (entity, block candidate, inventory slot, geometry, self, kit) with their `FEATURES`, the 21 gen-7 action heads (the gym's 12 plus 9 planned) and the six scopes that partition them (A1). Two token schemas (2026-10-19): `skill_tokens_v1` (`TOKEN_TYPES`, unchanged, the A2 runs') and `skill_tokens_v2` (`TOKEN_TYPES_V2`, `../docs/PERCEPTION.md` section 4): entity, block candidate, inventory slot and self carry the content registry's property rows (pinned here), description vectors, relations and spare features. |
| `tokens_v3.py` | 277 | Token schema v3 (`skill_tokens_v3`, 2026-10-20 night, `../docs/PERCEPTION.md` section 10): thirteen types (self, act, kit, status, entity, projectile, hazard, geometry, block candidate, inventory slot, recipe, persona, affect) as feature lists; the shared `where` block on every spatial type (the body and the head frame); `remembered`, `age` and `tagged` on the remembered types; default capacities (`CAPACITIES_V3`: this tick's terrain is the gym's per-line `WORLD_GEOMETRY_TOKEN_CAPACITY`, 96 on 0.7.2 since 04c3b18 and 44 on 0.5.7, plus `REMEMBERED_GEOMETRY_CELLS`, 32 nearby remembered cells; 340 tokens an agent on 0.7.2, 288 on 0.5.7). The `persona` and `affect` columns are generated from the variable registry (`variables.py`). v1 and v2 are unchanged. |
| `variables.py` | 178 | The variable registry (2026-10-21, `../docs/VIRTUAL_HUMAN.md` sections 10.1 and 14): every variable the agent tracks on its own side, never a value the game gives (owner: "states that are not given in game but we track locally"), each with its section, group, source, range, default and whether the body reads it. Today: the 24 self variables (15.6's traits and the motor bias) and the 8 emotion variables. `transfer` and `remap` carry columns from one list of names to another (new ones default, merged ones by an explicit map, dropped ones by name, never silently); `fingerprint` names a list or a layout. Since step 2 (2026-10-22, `../docs/VIRTUAL_HUMAN.md` section 19): `WRITERS` (the decision layer and the LLM), each written field's one writer (`Variable.writer`), and the fields of the written entries about an entity and about a place (`ENTITY_ENTRIES`, `PLACE_ENTRIES`: `remember` in [0, 1], written by the decision layer). |
| `catalogue.py` | 845 | The catalogue parser: the private `SKILL_CATALOGUE.md` rows as typed records, verb contracts (argument schema, scope, progress, stop labels), descriptions and activity programs; a JSON dump (A2). Weapon verbs (melee, ranged) and the strike tactics also hold `abilities_magic`: the gym's weapons attack through their ability slots (2026-10-19). `keep_range` and `avoid` take the target entity or its remembered position (`REMEMBERED_TARGET_VERBS`, 2026-10-20). `shortbow` also holds `gaze`, as the decision layer's contract declares: the arrow leaves along the head's yaw and pitch (2026-10-20). |
| `protocol.py` | 150 | The contracts lowered to the middle layer's request records (`experimental/neural_agent/requests.py`) and the check a request must pass (A3). |
| `references.py` | 350 | The resolver: a request's world point, entity handle or cell and face as the agent-frame vector, the entity token index and the marked block candidate (A4). |

Run (reads the private catalogue at run time):

```sh
PYTHONPATH=/Users/arush/hydl JAX_PLATFORMS=cpu .venv/bin/python -m \
    experimental.neural_agent.skill_agent.contract.catalogue [--path FILE] [--json]
```

Gates (in `tests/experimental/neural_agent/`, gitignored by policy): `test_sa_surface.py`, `test_sa_catalogue.py`,
`test_sa_protocol.py`, `test_sa_references.py`, `test_sa_tokens_v2.py`, `test_sa_tokens_v3.py`, `test_sa_state_registry.py`. The middle layer's
`jev/system_one/catalogue_matching.py` reads its skill rows through `catalogue.parse_catalogue` (B5), so run
`test_na_catalogue_matching.py` too after changing the parser or the catalogue.

Rules: the catalogue (`agent_skills/training_runs/SKILL_CATALOGUE.md`) is private: read at run time, never copied or
staged. The frame is the gym's: forward = (-sin yaw, 0, -cos yaw), right = (cos yaw, 0, -sin yaw), up = y.
