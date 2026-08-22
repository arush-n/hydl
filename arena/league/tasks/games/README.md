# Shipped games

This directory is the concrete task catalog. Importing it does not register games globally; call `register_all()` or register individual `Task` objects with `GameManager`.

## Catalog

| Subfolder/file | Games | What they exercise |
| --- | --- | --- |
| [`combat/`](combat/README.md) | `SURVIVE`, `DUEL`, `DEFEND` | Survival, target defeat, health protection, weapon/opponent ladders. |
| [`movement/`](movement/README.md) | `TRACK`, `TRAVERSE` | Target visibility and sustained movement/jumping. |
| [`world/`](world/README.md) | `REACH`, `BUILD`, generated WorldGen tasks | Interaction reach, terrain change, and bounded native worlds. |
| [`TEMPLATE.py`](TEMPLATE.py) | — | Authoring pattern for a new task module. |
| [`__init__.py`](__init__.py) | — | `BY_DOMAIN`, `SHIPPED`, `UNCOVERED_HEADS`, and `register_all()`. |

## Generated WorldGen tasks

`world/generated.py` defines:

- `PLAINS_DUEL`: defeat a target in a Plains pool.
- `DESERT_TRACK`: keep a target visible while also reducing its health below a threshold.
- `VOLCANIC_SURVIVE`: stay alive in a Volcanic pool.

Each uses a distinct two-seed bounded WorldGen V2 pool. The selected seeds and world semantic hashes become part of the scene identity.

## Coverage boundary

The catalog declares the currently uncovered inventory/hotbar-related action surface in `UNCOVERED_HEADS`. The exact label has changed across historical documentation; the source declaration is authoritative. Inventory quantity is not yet readable enough to support a sound item-collection game.

