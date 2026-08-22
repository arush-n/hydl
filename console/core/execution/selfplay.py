"""Self-play as a runnable job: train, snapshot, seat the snapshot, repeat.

`adk.selfplay` already has every piece — a versioned :class:`Population`, an
:class:`Arena` bound to the Gym's multi-actor transport, and a shared-policy
PPO trainer. What it did not have is the loop that makes those pieces *do*
something: train a while, freeze what you learned as a numbered generation, put
it in the pool, and train again against a population that now contains it.

**The bank is not the population.** This is the trap this module exists to
avoid, and it is invisible from the outside. ``make_arena`` reads the
population only to build the *assignment* — how many actor slots there are,
which single row the optimizer owns. The policy bank it returns is built from
``initialize_policy``: one **freshly random** row per slot
(``adk/selfplay/arena.py:317-324``). A generation loop that adds snapshots to a
`Population` and never writes them into the bank trains against random noise
every generation, reports rising returns, and is not self-play at all. So
:func:`run` seats each drawn opponent's parameters into its bank row explicitly
and records *which member* sat in each slot.

**The failure mode the shape guards against.** Self-play converges on a shared
convention, everybody beats everybody, and the whole pool is weak against
anything outside it — invisible from the population's own numbers, because
internal win rates look *healthiest* exactly when the pool has learned one
collective blind spot. So a fixed baseline is seeded first and never removed,
and `Population.select` excludes it: the yardstick must not become part of the
curriculum.

**Generations are snapshots, not checkpoints.** After each block of updates the
learner's row is lifted out of the policy bank by value and frozen into an
immutable `OpponentSpec`. Nothing later can mutate it, so a generation-5 result
measured against generation-2 means the same thing next month as today.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

_DEFAULT_WEAPONS = ("iron_sword", "iron_mace")


@dataclass(frozen=True)
class SelfPlaySpec:
    """One self-play run. Defaults are small enough to finish while watched."""

    #: One authored profile per entity, so this also sets the entity count and
    #: — since every entity is policy-driven here — the actor count.
    weapons: tuple[str, ...] = _DEFAULT_WEAPONS
    #: How many freeze-and-add cycles. Generation 0 is the seeded snapshot.
    generations: int = 3
    #: PPO updates between snapshots.
    updates_per_generation: int = 10
    #: Long enough that a match can finish inside one rollout. Measured
    #: episode lengths are 62-419 ticks, so the previous default of 8 put
    #: every terminal outside the horizon.
    rollout_steps: int = 64
    batch: int = 4
    #: **Starting health is what makes this environment learnable at all.**
    #: The arena has no time limit -- it ends only on death or completion
    #: (`arsenal/runtime.py:1616-1621`), and `truncated = ~valid` fires only on
    #: an arsenal failure, never on a step limit. At the authored 105.0,
    #: measured 2026-08-08, damage arrives in an early burst, reward then reads
    #: exactly 0.0 for hundreds of steps and heal effects return the entity to
    #: full: **zero episodes completed in 1,280 steps per arena row**, so every
    #: PPO advantage was bootstrapped from a value head that had never seen a
    #: terminal. At 20.0 the same 1,280 steps produce 23 episodes and 17
    #: deaths, with returns spanning -40 to +90. This is a sufficient value,
    #: not a tuned one.
    health: float = 20.0
    seed: int = 0
    #: How an opponent is drawn: ``latest`` cycles fastest and is the most
    #: prone to A-beats-B-beats-C loops; ``uniform`` resists it; ``recent`` is
    #: the compromise. Recorded because it changes the dynamics as much as any
    #: learning rate.
    selection: str = "recent"
    window: int = 5
    ppo: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if len(self.weapons) < 2:
            raise ValueError(
                "self-play needs at least two entities; "
                f"got weapons={self.weapons}")
        if self.generations < 1:
            raise ValueError("generations must be >= 1")
        if self.updates_per_generation < 1:
            raise ValueError("updates_per_generation must be >= 1")
        if self.batch < 1:
            raise ValueError("batch must be >= 1")
        if self.health <= 0.0:
            raise ValueError("health must be positive")
        from adk.selfplay import Selection

        if self.selection not in {rule.value for rule in Selection}:
            raise ValueError(
                f"unknown selection {self.selection!r}; expected one of "
                + ", ".join(sorted(rule.value for rule in Selection)))


def _row(bank: Any, policy_id: int) -> Any:
    """Lift one policy out of a ``[K, ...]`` bank, by value.

    Indexing copies rather than views, which is what makes the snapshot
    immutable in practice and not merely by convention: a later update writes a
    new bank and this row is unaffected.
    """

    import jax

    return jax.tree_util.tree_map(lambda leaf: leaf[policy_id], bank)


def _seat_row(bank: Any, policy_id: int, row: Any) -> Any:
    """Write one policy *into* a bank row, returning a new bank."""

    import jax

    return jax.tree_util.tree_map(
        lambda leaf, value: leaf.at[policy_id].set(value), bank, row)


def _policy_ids(assignment: Any) -> tuple[int, list[int]]:
    """``(learner_policy_id, frozen_policy_ids)``.

    Derived exactly the way the Gym's own trainer derives it
    (``multi_actor/training.py:86-94``) rather than assumed to be slot 0, so a
    change to `assignment_for`'s seating cannot silently desync the row this
    module snapshots from the row the optimizer actually owns.
    """

    import jax
    import numpy as np

    ids = np.asarray(jax.device_get(assignment.policy_id), dtype=np.int32)
    trainable = np.asarray(jax.device_get(assignment.trainable), dtype=np.bool_)
    selected = np.unique(ids[trainable])
    if selected.shape != (1,):
        raise ValueError(
            "shared-policy self-play needs exactly one trainable policy_id, "
            f"got {selected.tolist()}")
    learner = int(selected[0])
    return learner, [int(v) for v in np.unique(ids) if int(v) != learner]


def _flat_scalars(metrics: Any, prefix: str = "") -> dict[str, float]:
    """Every 0-d array on a metrics NamedTuple, nested ones included.

    `MultiActorTrainingMetrics.loss` is itself a NamedTuple, so a flat
    ``_fields`` walk hands back something `float()` refuses. Anything without a
    0-d ``ndim`` is skipped rather than coerced.
    """

    found: dict[str, float] = {}
    for name in getattr(metrics, "_fields", ()):
        value = getattr(metrics, name)
        if getattr(value, "_fields", None) is not None:
            found.update(_flat_scalars(value, f"{prefix}{name}."))
        elif getattr(value, "ndim", None) == 0:
            found[f"{prefix}{name}"] = float(value)
    return found


def run(spec: SelfPlaySpec, *, on_update=None) -> dict[str, Any]:
    """Run the generation loop and return what each generation produced.

    `on_update(generation, update, metrics)` is called after every PPO update
    so a caller can stream progress; a self-play run is long enough that a
    result-only API is unusable from a UI.
    """

    import jax
    import jax.numpy as jnp

    from adk.selfplay import (OpponentSpec, Population, Selection, make_arena,
                              shared_policy_trainer)

    spec.validate()
    started = time.perf_counter()
    rule = Selection(spec.selection)

    # `make_arena` refuses an empty population and needs one member per actor
    # slot, but it reads only their *count* and which one is the learner — the
    # bank it builds is random regardless. So this bootstrap exists purely to
    # shape the assignment, and its `parameters=None` never reaches the bank.
    actors = len(spec.weapons)
    bootstrap = Population()
    for index in range(actors):
        bootstrap = bootstrap.add(
            OpponentSpec(name=f"bootstrap-{index}", generation=0,
                         parameters=None))

    arena = make_arena(
        bootstrap,
        weapons=tuple(spec.weapons),
        batch=spec.batch,
        rollout_steps=spec.rollout_steps,
        learner="bootstrap-0",
        seed=spec.seed,
        health=spec.health,
        **spec.ppo,
    )
    learner_id, frozen_ids = _policy_ids(arena.assignment)
    if not frozen_ids:
        raise ValueError(
            "every actor slot is trainable, so there is no opponent to seat; "
            "self-play needs at least one frozen slot")

    trainer = shared_policy_trainer(arena)
    state = trainer.initialize(arena.bank, arena.state)

    # Generation 0 twice, deliberately. The anchor is a baseline: excluded from
    # selection, never dropped, and the only fixed reference a later drift
    # measurement can use. `gen-0` is the same weights as an ordinary learned
    # member, because generation 1 needs something in the pool to draw from.
    seed_row = _row(arena.bank, learner_id)
    pool = Population().add(OpponentSpec(
        name="anchor-untrained", generation=0, parameters=seed_row,
        baseline=True))
    pool = pool.add(OpponentSpec(
        name="gen-0", generation=0, parameters=seed_row))

    key = jax.random.PRNGKey(spec.seed)
    generations: list[dict[str, Any]] = []

    for generation in range(1, spec.generations + 1):
        # Seat this generation's opponents. Only frozen rows are written, so
        # the optimizer state — which was initialised for the learner row alone
        # — stays valid across the boundary.
        bank = state.policy_bank
        seated: list[str] = []
        for policy_id in frozen_ids:
            key, sub = jax.random.split(key)
            member = pool.select(rule, sub, window=spec.window)
            bank = _seat_row(bank, policy_id, member.parameters)
            seated.append(member.name)

        # The opponent's identity just changed, so a carry computed by the
        # previous weights no longer means anything, and an episode spanning
        # the boundary would attribute its return to two different matchups.
        # Losing one partial episode buys every reported number a single owner.
        state = state._replace(
            policy_bank=bank,
            recurrent_state=jnp.zeros_like(state.recurrent_state),
            episode_start=arena.assignment.active,
            running_episode_return=jnp.zeros_like(state.running_episode_return),
            running_episode_length=jnp.zeros_like(state.running_episode_length),
        )

        returns: list[float] = []
        applied = 0
        episodes = 0.0
        last: dict[str, float] = {}
        for update in range(spec.updates_per_generation):
            key, sub = jax.random.split(key)
            state, metrics = trainer.train_step(state, sub)
            last = _flat_scalars(metrics)
            completed = last.get("episodes_completed", 0.0)
            episodes += completed
            if completed > 0.0:
                returns.append(last["mean_episode_return"])
            applied += int(last.get("update_applied", 0.0))
            if on_update is not None:
                on_update(generation, update, last)

        snapshot = _row(state.policy_bank, learner_id)
        pool = pool.add(OpponentSpec(
            name=f"gen-{generation}", generation=generation,
            parameters=snapshot))
        generations.append({
            "generation": generation,
            "updates": spec.updates_per_generation,
            # How many updates the optimizer actually applied. A generation
            # where this is 0 trained on nothing, however healthy the returns
            # look, so it is reported next to them rather than derived later.
            "updates_applied": applied,
            "opponents": seated,
            # Summed across the generation, not read off the last update. A
            # generation's worth of matches is the unit a reader compares
            # between generations, and the final update's count is neither
            # that nor obviously not that.
            "episodes_completed": episodes,
            # Averaged over the updates that actually completed an episode.
            # Including the zero-episode updates would drag this toward 0 and
            # make a generation look worse the sparser its terminals were.
            "mean_episode_return": (
                sum(returns) / len(returns) if returns else None),
            "final_loss": last.get("loss.total_loss"),
            "population_size": len(pool.members),
        })

    return {
        "spec": asdict(spec),
        "generations": generations,
        "baselines": [member.name for member in pool.baselines],
        "learned": [member.name for member in pool.learned],
        "selection": spec.selection,
        "learner_policy_id": learner_id,
        "frozen_policy_ids": frozen_ids,
        "seconds": round(time.perf_counter() - started, 2),
        # Stated rather than implied: the loop keeps a fixed anchor and excludes
        # it from selection, but it does not play generations against the anchor
        # to produce a win rate, so `adk.selfplay.drift` has nothing to consume.
        # Reporting a drift number here would be inventing one.
        "drift": None,
        "drift_note": (
            "not computed: this loop trains and snapshots but does not yet run "
            "generation-vs-anchor evaluations, which is what drift() needs"),
    }
