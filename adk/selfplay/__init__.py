"""Self-play — a versioned opponent population with drift detection.

Self-play fails in one characteristic way: the population converges on a shared
convention, everyone beats everyone, and the whole pool is weak against
anything outside it.  **That is undetectable after the fact from the
population's own numbers**, because internal win rates look healthy precisely
when everybody has learned the same blind spot.

The guards are therefore structural, not advisory:

* opponents are **immutably versioned** -- `add` refuses a duplicate name;
* `Population` is frozen, so a snapshot handed to an evaluation cannot move;
* **baselines are excluded from selection** -- the yardstick is not the
  curriculum;
* `drift()` compares internal win rate against the fixed baseline, which is the
  only signal that can reveal convergence.

    from adk.selfplay import Population, OpponentSpec, Selection, is_drifting

    pool = Population().add(OpponentSpec("uniform-legal", 0, params, baseline=True))
    pool = pool.add(OpponentSpec("clone-v1", 1, trained))
    foe  = pool.select(Selection.RECENT, key, window=5)

These are cheap to build **before** the first population run and near-worthless
after it -- you cannot retroactively learn what a run would have scored against
an anchor you never kept.

`arena` binds a population onto the Gym's multi-actor transport, where the same
discipline has to survive contact with an optimizer:

    arena   = make_arena(pool, weapons=("iron_sword", "iron_mace"))
    trainer = shared_policy_trainer(arena)
    state   = trainer.initialize(arena.bank, arena.state)
    state, metrics = trainer.train_step(state, key)

Exactly one member is the learner; every other row -- anchor or old snapshot --
stays byte-identical through the update.

For two independently learning agents, use the explicit duel mode instead:

    arena   = make_duel(weapons=("iron_sword", "iron_mace"), batch=8)
    trainer = duel_trainer(arena)

Both policy rows act in the same JAX environment and own separate optimizers.
Use ``make_region_duel`` for the same ownership contract on exact WorldGen V2
terrain; ``make_duel`` remains the fast open-flat control.
"""

from adk.selfplay.arena import (
    Arena,
    assignment_for,
    duel_trainer,
    make_arena,
    make_duel,
    make_region_duel,
    seating,
    shared_policy_trainer,
    trainable_count,
)
from adk.selfplay.population import (
    OpponentSpec,
    Population,
    Selection,
    drift,
    is_drifting,
)

__all__ = [
    "Arena",
    "OpponentSpec",
    "Population",
    "Selection",
    "assignment_for",
    "drift",
    "duel_trainer",
    "is_drifting",
    "make_arena",
    "make_duel",
    "make_region_duel",
    "seating",
    "shared_policy_trainer",
    "trainable_count",
]
