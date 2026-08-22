"""Training tasks: same contract, different objective.

The environment exposes exactly four reward terms (`combat/types.py:182`):

    target_damage * target_damage_reward_scale
  + agent_damage  * agent_damage_reward_scale
  + completion_reward   when the target dies
  + death_reward        when the agent dies

They are `CombatParams` fields, and `CombatParams` is a NamedTuple -- so a task
is a `._replace()` on those four numbers plus a scene choice. That is the whole
mechanism, and it needs no Gym change.

What a task is NOT: a new reward *signal*. There is no time penalty, no distance
term, no stamina term. A behaviour that no combination of these four terms can
distinguish cannot be trained here by reweighting alone; it needs a wrapper that
shapes on observed state, and `shaped()` below is the one example of that.

Every task declares `teaches` (the behaviour it should produce) and `tell` (the
metric that would show it worked). A task whose `tell` cannot move on its scene
is a broken task, not a failed agent -- see `open_flat` vs `fail_closed`, where
the ability head collapses to a single no-op and no offensive task is learnable
at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax.numpy as jnp


@dataclass(frozen=True)
class Task:
    name: str
    teaches: str
    tell: str
    #: Overrides applied to `CombatParams` via `._replace`.
    reward: dict[str, float] = field(default_factory=dict)
    loadout: str = "iron_sword"
    opponent: str = "iron_sword"
    #: `None` keeps the caller's provider. Tasks do not choose `fail_closed`:
    #: it denies every world query, the ability head collapses to `{0}`, and
    #: nothing offensive is reachable. Measured, not assumed.
    permissive: bool = True

    def params(self, base):
        """Apply this task's reward weights to a `CombatParams`."""

        if not self.reward:
            return base
        unknown = set(self.reward) - set(base._fields)
        if unknown:
            raise ValueError(
                f"task {self.name!r} sets unknown CombatParams field(s): "
                f"{sorted(unknown)}")
        return base._replace(**{
            key: jnp.asarray(value, dtype=jnp.float32)
            for key, value in self.reward.items()
        })


# Baseline first: every other task is read as a delta from this one, and
# without it a "the agent learned X" claim has nothing to be a difference from.
BASELINE = Task(
    name="baseline",
    teaches="the packaged objective, unmodified",
    tell="mean_episode_return and episode_successes",
)

TASKS: dict[str, Task] = {task.name: task for task in (
    BASELINE,
    Task(
        name="executioner",
        teaches="finish the kill instead of farming chip damage",
        tell="episode_successes should rise while mean_episode_length falls",
        # Damage becomes worth nothing on its own; only the kill pays. This is
        # the sparsest of these tasks and the one most likely to fail to train
        # at all -- which is itself the result worth having.
        reward={"target_damage_reward_scale": 0.0, "completion_reward": 200.0},
    ),
    Task(
        name="survivor",
        teaches="stay alive; disengagement beats a bad trade",
        tell="episode_deaths should fall toward zero, length should rise",
        reward={"target_damage_reward_scale": 0.0,
                "agent_damage_reward_scale": -4.0,
                "completion_reward": 0.0,
                "death_reward": -200.0},
    ),
    Task(
        name="duelist",
        teaches="win the damage trade, not just deal damage",
        tell="agent_damage per episode should fall at equal target_damage",
        reward={"agent_damage_reward_scale": -4.0},
    ),
    Task(
        name="berserker",
        teaches="ignore incoming damage entirely; pure aggression control",
        tell="a ceiling on what damage-only optimisation reaches",
        reward={"agent_damage_reward_scale": 0.0, "death_reward": 0.0},
    ),
    # Weapon-conditioned variants. Same objective, different reach and ability
    # bank, so a policy that has learned spacing should transfer unevenly --
    # daggers peak at a shorter range than the sword.
    Task(
        name="dagger_duel",
        teaches="close-range spacing; daggers deal zero at point-blank",
        tell="damage-per-episode against the sword baseline",
        loadout="iron_daggers",
    ),
    Task(
        name="ranged",
        teaches="keep distance and manage ammo",
        tell="ammo resource should deplete; distance should stay high",
        loadout="iron_crossbow",
    ),
    Task(
        name="outnumbered",
        teaches="fight a better-armed opponent",
        tell="win rate against a battleaxe opponent vs the sword baseline",
        opponent="iron_battleaxe",
    ),
)}


def shaped(step, shape: Callable, weight: float = 1.0):
    """Wrap `step_detailed` to add a shaping term computed from state.

    Kept here as the name every existing caller imports. The implementation
    moved to :func:`adk.scenarios.shaping.apply` when shaping became a generic
    seam rather than a minigame detail -- a task reweights the four native
    terms, a *shaping* term adds a fifth computed from state, and minigames are
    one implementation of the second rather than being the second.
    """

    from .shaping import apply

    return apply(step, shape, weight)
