# skill_agent/perception/ — what the skill layer perceives

The skill layer's input contract (`../docs/PERCEPTION.md`, accepted by the owner on 2026-10-19): content described by
its properties and by relations to the agent, built from the game's own asset files, so the base game and mods are
read the same way. Legal perception only: the adapter reads the legal observation, never the simulator's state.

| Module | Lines | What it holds |
|---|---:|---|
| `registry.py` | 783 | The offline content registry (build step 1). **Items**: every item of a server version's `Assets.zip` resolved through its `Parent` chain, with `ITEM_FEATURES`; the blocks with `BLOCK_FEATURES` and their breaking rule; the open gather-type table with each tool's power and quality per gather type. **Creatures**: every NPC role a world can spawn, resolved through its references and parameters, with `CREATURE_FEATURES` (health, motion, model size, attack distance, taming, trade). **Attitudes**: each creature's attitude toward a player and toward every creature, from the NPC groups and attitude tables by the engine's rules. A 64-number description vector per item and per creature (a pinned local sentence encoder, projected by a basis fitted once per registry and kept for content added later); saved with a manifest and SHA-256s. |
| `effects.py` | 254 | The offline status-effect registry (`../docs/VIRTUAL_HUMAN.md` 11.4, 2026-10-22). Every `EntityEffect` of `Server/Entity/Effects/`, resolved through its `Parent` chain, with `EFFECT_FEATURES` (debuff, infinite, duration, overlap behaviour, damage over time, stat, damage-taken and damage-dealt modifiers, speed multiplier, invulnerable, model change, conditional) at the simulator's own defaults, and a 64-number description vector (its own projection). `effect_tables` keys it by the gym's semantic id, the id the observation publishes in `status_i32` (`effect_id`), for `adapter.lookup`. Not yet read by the status token. |
| `adapter.py` | 372 | The v2 adapter (build step 3). `legal_tokens_v2`: the legal observation as `skill_tokens_v2` token sets, reading only the observation, the body yaw and the registry tables: the self token (v1's features; with the tables, the held weapon's registry row), every published entity row (state, facing in the body frame, attack phase and progress). With the gen-7 perception channel (`perception_v2`, 2026-10-19 night) it also reads each published row's kind, held item and role, the agent's inventory tokens and the body yaw (`../docs/PERCEPTION_ASKS.md`). `registry_tables`, `lookup` and `tool_relations`: the registry on the device keyed by the gym's semantic ids, the id-to-row gather, and the agent's tools against block candidates, ready for those ids. |

**Token schema v3** moved to `../state/` on 2026-10-21: the senses, frames, the action history, the stores, the persona,
the affect proposal and the tick are the state layer now (`../state/README.md`, `../docs/VIRTUAL_HUMAN.md`). They
read this folder's registry tables and v2's inventory rows.

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

The effects registry runs the same way (`perception.effects`). On the 0.7.2 archive: 260 effects, 22 debuffs, 44
infinite, 20 with damage over time. Only `Poison` has a name in the language file, so the other 259 are described by
their folders and id. On a gen-7 Region reset the one active effect, id 1315835084, resolves to
`Immunity_Environmental`.

Gates: `test_sa_perception_registry.py`, `test_sa_perception_adapter.py`, `test_sa_perception_v2.py` (the channel, and its
acceptance on a private gen-7 render), `test_sa_perception_effects.py`. Next: the role id, block candidates and fluid
ids, read as each seam lands; the status token reads the effects registry after the geometry capacity change (96).
