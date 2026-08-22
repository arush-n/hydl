# ADK self-play

`adk.selfplay` provides versioned opponent populations, immutable snapshots,
seating, selection, drift checks, and multi-actor arena construction. It is a
reusable population layer rather than the Console's current one-side-per-run
pursuit loop.

It depends on the ADK runtime and Arena's multi-actor training primitives. The
population records who is frozen and who may learn; it does not silently make a
historical policy trainable.

## Entry points

- [`population.py`](population.py) defines immutable opponent records and
  selection.
- [`arena.py`](arena.py) binds a population to actor slots and a policy bank.
- [`__init__.py`](__init__.py) exposes the public self-play helpers.

For the current recursive pursuit lesson, continue to
[`arena/training/runs/`](../../arena/training/runs/README.md).
