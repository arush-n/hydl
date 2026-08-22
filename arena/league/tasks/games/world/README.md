# World-backed games

These tasks require geometry, traversal, interaction, or a WorldGen-backed reset. They are built on the same `Task` and `RewardConfig` contracts as combat tasks, but their `Task.world` binding supplies native region providers and world provenance.

| File | Task | Success condition |
| --- | --- | --- |
| [`reach.py`](reach.py) | `REACH` | Bring the interaction target into reach. |
| [`build.py`](build.py) | `BUILD` | Change traversability at the interaction location. |
| [`generated.py`](generated.py) | `PLAINS_DUEL`, `DESERT_TRACK`, `VOLCANIC_SURVIVE` | World-specific compositions of combat, visibility, health, and survival goals. |
| [`__init__.py`](__init__.py) | — | Re-exports world tasks. |

`BUILD` is intentionally described as a terrain-change task, not an exact block-placement task. The current observation contract does not identify the resulting block reliably enough to claim more.

