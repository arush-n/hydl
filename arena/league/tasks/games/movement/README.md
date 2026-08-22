# Movement games

The movement games isolate navigation and tracking behaviour from full combat objectives.

| File | Task | Success condition |
| --- | --- | --- |
| [`track.py`](track.py) | `TRACK` | Keep the target visible for the required fraction of recorded ticks. |
| [`traverse.py`](traverse.py) | `TRAVERSE` | Hold the jumping/movement state for the required fraction. |
| [`__init__.py`](__init__.py) | — | Re-exports the movement tasks. |

`TRACK` has stationary and mobile opponent rungs. `TRAVERSE` deliberately exposes the jump head and scores the engine’s movement-state evidence; the current status notes that this column can be flat under a uniformly legal policy, so the task acts as a forcing function rather than assuming the feature is informative.

