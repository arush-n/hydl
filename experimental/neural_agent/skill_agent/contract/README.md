# skill_agent/contract/ — what a skill is and what it speaks

The fixed interface of the lower layer, which every other subfolder builds against: the gen-7 surface (token sets
and action heads in six scopes), the skill catalogue read into verb contracts, the request protocol shared with the
middle layer, and the resolver from a request's target to what a body sees. Nothing here is learned.

| Module | Lines | What it holds |
|---|---:|---|
| `surface.py` | 951 | The gen-7 surface: the token types (entity, block candidate, inventory slot, geometry, self, kit) with their `FEATURES`, the 21 gen-7 action heads (the gym's 12 plus 9 planned) and the six scopes that partition them (A1). Two token schemas (2026-10-19): `skill_tokens_v1` (`TOKEN_TYPES`, unchanged, the A2 runs') and `skill_tokens_v2` (`TOKEN_TYPES_V2`, `../docs/PERCEPTION.md` section 4): entity, block candidate, inventory slot and self carry the content registry's property rows (pinned here), description vectors, relations and spare features. |
| `catalogue.py` | 845 | The catalogue parser: the private `SKILL_CATALOGUE.md` rows as typed records, verb contracts (argument schema, scope, progress, stop labels), descriptions and activity programs; a JSON dump (A2). Weapon verbs (melee, ranged) and the strike tactics also hold `abilities_magic`: the gym's weapons attack through their ability slots (2026-10-19). `keep_range` and `avoid` take the target entity or its remembered position (`REMEMBERED_TARGET_VERBS`, 2026-10-20). `shortbow` also holds `gaze`, as the decision layer's contract declares: the arrow leaves along the head's yaw and pitch (2026-10-20). |
| `protocol.py` | 150 | The contracts lowered to the middle layer's request records (`experimental/neural_agent/requests.py`) and the check a request must pass (A3). |
| `references.py` | 350 | The resolver: a request's world point, entity handle or cell and face as the agent-frame vector, the entity token index and the marked block candidate (A4). |

Run (reads the private catalogue at run time):

```sh
PYTHONPATH=/Users/arush/hydl JAX_PLATFORMS=cpu .venv/bin/python -m \
    experimental.neural_agent.skill_agent.contract.catalogue [--path FILE] [--json]
```

Gates (in `tests/experimental/neural_agent/`, gitignored by policy): `test_sa_surface.py`, `test_sa_catalogue.py`,
`test_sa_protocol.py`, `test_sa_references.py`, `test_sa_tokens_v2.py`. The middle layer's
`jev/system_one/catalogue_matching.py` reads its skill rows through `catalogue.parse_catalogue` (B5), so run
`test_na_catalogue_matching.py` too after changing the parser or the catalogue.

Rules: the catalogue (`agent_skills/training_runs/SKILL_CATALOGUE.md`) is private: read at run time, never copied or
staged. The frame is the gym's: forward = (-sin yaw, 0, -cos yaw), right = (cos yaw, 0, -sin yaw), up = y.
