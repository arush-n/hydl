# skill_agent/perception/ — what the skill layer perceives

The skill layer's input contract (`../docs/PERCEPTION.md`, accepted by the owner on 2026-10-19): content described by
its properties and by relations to the agent, built from the game's own asset files, so the base game and mods are
read the same way. Legal perception only: the adapter reads the legal observation, never the simulator's state.

| Module | Lines | What it holds |
|---|---:|---|
| `registry.py` | 783 | The offline content registry (build step 1). **Items**: every item of a server version's `Assets.zip` resolved through its `Parent` chain, with `ITEM_FEATURES`; the blocks with `BLOCK_FEATURES` and their breaking rule; the open gather-type table with each tool's power and quality per gather type. **Creatures**: every NPC role a world can spawn, resolved through its references and parameters, with `CREATURE_FEATURES` (health, motion, model size, attack distance, taming, trade). **Attitudes**: each creature's attitude toward a player and toward every creature, from the NPC groups and attitude tables by the engine's rules. A 32-number description vector per item and per creature (a pinned local sentence encoder, projected by a basis fitted once per registry and kept for content added later); saved with a manifest and SHA-256s. |
| `adapter.py` | 271 | The v2 adapter (build step 3). `legal_tokens_v2`: the legal observation as `skill_tokens_v2` token sets, reading only the observation and the body yaw: the self token (v1's features), every published entity row (state, facing in the body frame, attack phase and progress), content left unknown until the simulator publishes ids (`../docs/PERCEPTION_ASKS.md`). `registry_tables`, `lookup` and `tool_relations`: the registry on the device keyed by the gym's semantic ids, the id-to-row gather, and the agent's tools against block candidates, ready for those ids. |

Run (offline; it never downloads):

```sh
PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym .venv/bin/python -m \
    experimental.neural_agent.skill_agent.perception.registry --assets ASSETS.zip --out PREFIX
```

The 0.7.2 registry builds in about 15 seconds: 3,960 items, 3,182 blocks, 19 gather types, 29 tools with powers,
894 creatures (523 of them gameplay roles; the rest are the game's test roles, marked `non_gameplay`), 114 NPC groups
and 29 attitude tables. Its tables are derived from the game's assets, so they stay private
(`agent_skills/training_runs/perception/`).

The creature rules come from the 0.7.2 server itself (read from its classes; each constant in `registry.py` names its
class): the attitude order and how overlapping groups resolve, the defaults (hostile to players and neutral to NPCs
when a role names neither), how groups include and exclude, the glob, the shop actions and the default speeds. Role
resolution follows the repo's extractor (`hytalegym/tools/content_extract/npc_roles.py`) and agrees with its gen-7 table
on all 894 roles.

Gates: `test_sa_perception_registry.py`, `test_sa_perception_adapter.py`. Next (`../docs/PERCEPTION.md` section 9): the
per-tick cost at 16, 32 and 64 entities (step 5); the adapter reads each id the simulator layer publishes as its ask
lands.
