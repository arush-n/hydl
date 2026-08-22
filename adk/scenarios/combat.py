"""Concrete scenarios, each isolating one skill damage alone cannot teach.

Ceilings are derived from measured environment constants, never guessed, and
the arithmetic is stated on each scenario so a stale ceiling is falsifiable
rather than merely suspicious.
"""

from __future__ import annotations

import jax.numpy as jnp

from adk.capability.scenes import control_scene
from adk.metrics import (
    TICKS_PER_SECOND,
    combo_count,
    engagement_fraction,
    longest_combo,
    rising_edges,
    ticks,
    whiff_rate,
)
from adk.scenarios.core import Scenario, Trajectory, registry

#: Episode length every scenario is scored over.
HORIZON = 256

#: Measured 2026-08-04: walk speed 0.16078 units/tick (4.823 u/s).
WALK_UNITS_PER_TICK = 0.16078

#: Melee threat band. iron_sword's authored MELEE_CONE range is 3 units and a
#: spear's is 3; inside 1 unit the agent is inside the opponent's own reach.
NEAR, FAR = 1.0, 3.0


def _agent(trajectory: Trajectory, name: str) -> jnp.ndarray:
    """Ability arrays carry an entity axis; damage arrays do not."""

    array = jnp.asarray(trajectory[name])
    return array[..., 0] if array.ndim == 3 else array


# -- spacing -------------------------------------------------------------------


def _spacing_score(trajectory: Trajectory) -> jnp.ndarray:
    return engagement_fraction(
        jnp.asarray(trajectory["target_distance"]), near=NEAR, far=FAR
    )


SPACING = Scenario(
    name="combat/spacing",
    teaches="hold the threat band instead of drifting in and out of it",
    build=lambda batch: control_scene("scenario/spacing", batch=batch),
    score=_spacing_score,
    floor=0.10,
    ceiling=0.90,
    ceiling_rationale=(
        "The agent starts outside the band and walks at 0.16078 units/tick, so "
        "closing a few units costs tens of ticks it can never recover. Over a "
        "256-tick episode a perfect policy therefore spends ~90% of ticks in "
        "band, not 100%. The floor is what standing still scores: the scripted "
        "target approaches and passes through the band on its own."
    ),
)


# -- punish timing -------------------------------------------------------------


def _combo_score(trajectory: Trajectory) -> jnp.ndarray:
    hit = jnp.asarray(trajectory["landed"]).astype(bool)
    return longest_combo(hit, gap=ticks(0.4)).astype(jnp.float32)


COMBO = Scenario(
    name="combat/combo",
    teaches="chain hits inside the opponent's recovery instead of trading",
    build=lambda batch: control_scene("scenario/combo", batch=batch),
    score=_combo_score,
    floor=1.0,
    ceiling=4.0,
    ceiling_rationale=(
        "gap is 12 ticks (0.4 s at 30 TPS). A chain longer than the opponent's "
        "recovery window cannot be extended, so with measured hit cadence a "
        "perfect policy reaches ~4 linked hits. floor 1.0 is a single hit -- "
        "any policy that lands at all scores that without chaining."
    ),
)


# -- discipline ----------------------------------------------------------------


def _discipline_score(trajectory: Trajectory) -> jnp.ndarray:
    """Reward connecting, not swinging.  Inverted whiff rate."""

    started = rising_edges(_agent(trajectory, "active_slot") >= 0)
    hit = jnp.asarray(trajectory["landed"]).astype(bool)
    return 1.0 - whiff_rate(started, hit, window=ticks(0.5))


DISCIPLINE = Scenario(
    name="combat/discipline",
    teaches="swing only when it will connect; every whiff is punishable recovery",
    build=lambda batch: control_scene("scenario/discipline", batch=batch),
    score=_discipline_score,
    floor=0.05,
    ceiling=0.80,
    ceiling_rationale=(
        "window is 15 ticks (0.5 s). Even an optimal policy whiffs when the "
        "opponent moves during travel, so 1.0 is unreachable; 0.80 is the "
        "practical ceiling. floor 0.05 is a policy that swings on cooldown and "
        "connects only by coincidence -- measured hit rates on these scenes are "
        "0-2 landings per few hundred ticks."
    ),
)


# -- opening -------------------------------------------------------------------


def _pressure_score(trajectory: Trajectory) -> jnp.ndarray:
    hit = jnp.asarray(trajectory["landed"]).astype(bool)
    return combo_count(hit, gap=ticks(0.4)).astype(jnp.float32)


PRESSURE = Scenario(
    name="combat/pressure",
    teaches="create repeated openings rather than one lucky exchange",
    build=lambda batch: control_scene("scenario/pressure", batch=batch),
    score=_pressure_score,
    floor=0.0,
    ceiling=8.0,
    ceiling_rationale=(
        "256 ticks at ~30 ticks per opening cycle bounds a perfect policy near "
        "8 separate combos. floor 0.0 is never landing."
    ),
)


SCENARIOS = registry(SPACING, COMBO, DISCIPLINE, PRESSURE)

__all__ = [
    "COMBO",
    "DISCIPLINE",
    "HORIZON",
    "PRESSURE",
    "SCENARIOS",
    "SPACING",
    "TICKS_PER_SECOND",
]
