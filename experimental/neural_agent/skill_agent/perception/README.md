# skill_agent/perception/ — what the skill layer perceives

The skill layer's input contract (`../docs/PERCEPTION.md`, accepted by the owner on 2026-10-19): content described by
its properties and by relations to the agent, built from the game's own asset files, so the base game and mods are
read the same way. Legal perception only: the adapter reads the legal observation, never the simulator's state.

| Module | Lines | What it holds |
|---|---:|---|
| `registry.py` | 463 | The offline content registry (build step 1): every item of a server version's `Assets.zip` resolved through its `Parent` chain, with `ITEM_FEATURES`; the blocks with `BLOCK_FEATURES` and their breaking rule; the open gather-type table with each tool's power and quality per gather type; a 32-number description vector per item (a pinned local sentence encoder, projected by a basis fitted once per registry and kept for content added later); saved with a manifest and SHA-256s. |

Run (offline; it never downloads):

```sh
PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym .venv/bin/python -m \
    experimental.neural_agent.skill_agent.perception.registry --assets ASSETS.zip --out PREFIX
```

The 0.7.2 registry builds in about 10 seconds: 3,960 items, 3,182 blocks, 19 gather types, 29 tools with powers. Its
tables are derived from the game's assets, so they stay private (`agent_skills/training_runs/perception/`).

Gate: `test_sa_perception_registry.py`. Next (`../docs/PERCEPTION.md` section 9): creatures (NPC roles and attitudes)
in the registry, the v2 token types, and the v2 adapter.
