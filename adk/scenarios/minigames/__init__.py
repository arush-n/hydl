"""Minigames: shaped objectives that train behaviours combat reward cannot.

The native reward has four terms -- target damage, agent damage, completion,
death (`combat/types.py:182`). None mention position, time, speed, accuracy or
resource discipline, so those are not reweightings. They need a term computed
from state, which `tasks.shaped()` adds *on top of* the native reward rather
than replacing, so a shaped run stays comparable to the baseline on the native
terms alone.

Prefer the generic seam, which resolves a name, a callable, a `Shaping` or a
list of them, and forwards `observe_done` through composition:

    from adk.scenarios import shaping
    from adk.scenarios.minigames import REGISTRY

    term = shaping.resolve("sprint", registry=REGISTRY)
    handle = handle.with_shaping(term)          # reaches PPO and rollouts alike

A minigame is one *implementation* of `adk.scenarios.shaping.Shaping`, not the
thing itself -- a curriculum term or a one-off diagnostic satisfies the same
contract and composes with these. The older step-wrapping form still works:

    from adk.scenarios import tasks
    step = tasks.shaped(jax.jit(env.step_detailed), REGISTRY["sprint"].build())

Each game declares `requires`. Check it before trusting a number. Note that
"no opponent can land a hit" (ISSUES.md #4) was **refuted by measurement on
2026-08-07** -- 70.0 damage over 7 hit events against an armed opponent and
0.0 against inert -- so the `armed_opponent` games are not void for that
reason. See DEV.md for the table.

See `DEV.md` for the framework, the blocked games, and what each `tell` means.
"""

from __future__ import annotations

from . import (accuracy, block, bulwark, checkpoint, dodge, efficiency, flee,
               footing, jump, reach, spacing, sprint, strike, tracking)
from .framework import (
    AGENT_ENTITY, TARGET_ENTITY, Minigame, accepted, band, damage, defended,
    displacement, distance, event, facing, goal_distance, grounded, health,
    landed, penalise, progress, resource, separation, threshold, waypoints,
)

#: name -> Minigame. Two need an argument and cannot be built bare -- `reach`
#: needs a `goal` and `checkpoint` a `route` -- so check `.requires` first.
REGISTRY: dict[str, Minigame] = {
    module.MINIGAME.name: module.MINIGAME
    for module in (reach, checkpoint, flee, sprint, spacing, footing,
                   efficiency, accuracy, strike, block, dodge, bulwark,
                   tracking, jump)
}

__all__ = [
    "REGISTRY", "Minigame", "framework",
    "reach", "checkpoint", "flee", "sprint", "spacing", "footing",
    "efficiency", "accuracy", "strike", "block", "dodge", "bulwark",
    "tracking", "jump",
    "distance", "separation", "goal_distance", "displacement", "health",
    "resource", "accepted", "penalise", "damage", "landed", "defended",
    "grounded", "facing", "waypoints", "event",
    "progress", "band", "threshold", "AGENT_ENTITY", "TARGET_ENTITY",
]
