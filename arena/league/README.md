# League and task ownership

The `league` package contains two related but separate concerns:

1. [`pool.py`](pool.py) provides host-side opponent records, ratings, expected scores, and sampling primitives.
2. [`tasks/`](tasks/README.md) is the physical home of Arena’s task framework and shipped games.

The task tree remains importable as `arena.tasks` for compatibility; see [`arena/tasks/README.md`](../tasks/README.md).

## `pool.py`

`Opponent` describes scripted, parameterized, or learned opponents. `Pool` stores them, while `Record` and `expected_score()` support rating-aware selection and Elo-style reports. Learned opponents are metadata until a caller supplies a playable multi-actor path; the pool does not silently turn a single-actor evaluator into self-play.

## What is not here

The production frozen-opponent league used by JAX training lives in [`training/selfplay/`](../training/selfplay/README.md). This package is the older/general task-and-opponent namespace; do not assume that registering an `Opponent` in `pool.py` automatically creates a trainable JAX policy seat.

