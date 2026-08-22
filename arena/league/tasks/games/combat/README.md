# Combat games

The combat games are small task definitions over the Arsenal combat surface. Their difficulty ladders change loadout, opponent profile, health, or sensor conditions; the task criterion remains explicit.

| File | Task | Success condition |
| --- | --- | --- |
| [`survive.py`](survive.py) | `SURVIVE` | Remain alive through the evaluation horizon. |
| [`duel.py`](duel.py) | `DUEL` | Observe the designated target defeated. |
| [`defend.py`](defend.py) | `DEFEND` | Keep the learner above the configured health fraction. |
| [`__init__.py`](__init__.py) | — | Re-exports the combat tasks. |

The tasks do not implement a new combat simulator. They choose HytaleGym profiles/loadouts and bind them to the common Arena scene builder. An “inert” opponent is useful for a controlled ladder rung, but it is not evidence that a defensive mechanic works; the armed rungs are the meaningful guard tests.

