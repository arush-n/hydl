"""Scripted Skeleton_Archer_Turret for entity 1 of the duel arena (opt-in; ``duel_arena`` defaults are unchanged).

Each tick it reads entity 1's own observation (the perception the body would have of entity 0) and returns that
entity's factors. Rules:

* Stationary: no locomotion, ever. The variant's ``BodyMotion`` is ``Nothing`` (Template_Skeleton_Turret.json).
* Face the target: the body turns toward the target's bearing, at most 45 degrees a tick (the duel syncs the head to the
  body, so the body yaw is the aim).
* Throw (ability slot 0, the bow shot) when the target is visible and within ``AttackDistance`` 18 blocks, and the bow is
  idle.

Cadence (the default, ``cadence="asset"``): the variant's ``AttackPauseRange`` [1.5, 2] is the minimum time before the
next attack, so after a throw the controller waits a seeded uniform draw in [1.5, 2] s and then throws again once the bow
is idle. The draw is the asset's range, not a fit to the live data. ``cadence="idle"`` throws whenever the bow is idle
and the target is in range.

Not modelled: the ``AlertedTime`` [1, 2] wait before the first attack (the first throw is at tick 0).

Conventions, measured on the duel engine (see ``shaman_controller.py``, 2026-10-23):

* One block is 1/24 of a perception unit (``planar_distance``); ``relative_right`` is positive on the body's right.
* Positive ``body_yaw_delta_bins`` turns the body right.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import numpy as np

from adk.policy import fields as schema
from hytalegym.jax.combat.observation.v3.policy.layout import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES,
    neutral_arsenal_policy_action_factors,
)
from hytalegym.jax.combat.observation.v3.policy.look_deltas import look_delta_values

#: Variant Skeleton_Archer_Turret ``AttackDistance`` 18 (blocks): the whole reach of the turret.
ATTACK_DISTANCE_BLOCKS = 18.0
#: One perception unit per 24 blocks (measured; see ``shaman_controller.py``).
BLOCKS_PER_PERCEPTION_UNIT = 24.0
#: Half the perception resolution: a reading within this of the range edge counts as in range.
EDGE_TOLERANCE_BLOCKS = 0.05
#: One duel step is 1/30 s (the game tick; the cadence converts seconds to steps with it).
TICK_SECONDS = 1.0 / 30.0
#: Variant ``AttackPauseRange`` [1.5, 2]: a uniform draw in seconds after each throw (the asset's range).
PAUSE_SECONDS_RANGE = (1.5, 2.0)
#: The ability_none_plus_slots choice that requests ability slot 0 (layout.py: choice k+1 is slot k).
THROW_CHOICE = 1
#: Nine-bin look codebook (+-45 degrees a tick at the ends; index 4 is zero).
YAW_BINS = np.asarray(look_delta_values(9), dtype=np.float64)
NO_TURN_BIN = 4
CADENCES = ("asset", "idle")

_I = {name: index for index, name in enumerate(ARSENAL_POLICY_ACTION_HEAD_NAMES)}
_VISIBLE = schema.field_index("target_f32", "visible")
_RIGHT = schema.field_index("target_f32", "relative_right")
_FORWARD = schema.field_index("target_f32", "relative_forward")
_PLANAR = schema.field_index("target_f32", "planar_distance")
_BUSY = schema.field_index("self_f32", "attack_executing")


class TurretState(NamedTuple):
    """Per-duel memory: the earliest tick of each row's next throw (int64[B]) and the seeded key for the pause draws."""

    ready_tick: np.ndarray
    key: jax.Array | None


def initial_state(batch: int, rng_key: jax.Array | None = None) -> TurretState:
    """Start a duel. The first throw is at tick 0. ``rng_key`` seeds the pause draws (``asset`` cadence)."""

    return TurretState(np.zeros((batch,), dtype=np.int64), rng_key)


def turret_factors(
    observation,
    state: TurretState,
    tick: int,
    *,
    cadence: str = "asset",
) -> tuple[np.ndarray, TurretState]:
    """Return ``(factors, state)``: int32[B, heads] for entity 1, and the state after this tick.

    ``observation`` is entity 1's observation from the previous step (``obs1`` of ``DuelArena.step``); ``tick`` is the
    count of steps taken so far. ``cadence`` is ``"asset"`` (a seeded 1.5 to 2 s pause after each throw; needs the key
    from :func:`initial_state`) or ``"idle"``. The factors are host arrays; pass ``jnp.asarray(factors, dtype=jnp.int32)``.
    """

    if cadence not in CADENCES:
        raise ValueError(f"cadence must be one of {CADENCES}, got {cadence!r}")
    self_f = np.asarray(observation.base.self_f32, dtype=np.float64)
    target = np.asarray(observation.base.target_f32, dtype=np.float64)
    batch = self_f.shape[0]
    visible = target[:, _VISIBLE] > 0.5
    forward = target[:, _FORWARD]
    right = target[:, _RIGHT]
    blocks = target[:, _PLANAR] * BLOCKS_PER_PERCEPTION_UNIT
    busy = self_f[:, _BUSY] > 0.5

    factors = np.array(neutral_arsenal_policy_action_factors(batch), dtype=np.int32)

    # Face the target: the nearest bin to the bearing, clipped to the codebook's +-45 degrees.
    bearing = np.degrees(np.arctan2(right, forward))
    wanted = np.clip(bearing, YAW_BINS[0], YAW_BINS[-1])
    nearest = np.argmin(np.abs(YAW_BINS[None, :] - wanted[:, None]), axis=1)
    factors[:, _I["body_yaw_delta_bins"]] = np.where(visible, nearest, NO_TURN_BIN)

    # Throw when in range, the bow is idle, and (asset cadence) the pause since the last throw has passed.
    in_range = visible & (blocks <= ATTACK_DISTANCE_BLOCKS + EDGE_TOLERANCE_BLOCKS)
    ready = tick >= state.ready_tick if cadence == "asset" else np.ones(batch, dtype=bool)
    throw = in_range & ~busy & ready
    factors[:, _I["ability_none_plus_slots"]] = np.where(throw, THROW_CHOICE, 0)

    ready_next, key = state.ready_tick, state.key
    if cadence == "asset" and bool(throw.any()):
        if key is None:
            raise ValueError("cadence 'asset' needs the rng_key that initial_state was given")
        key, draw_key = jax.random.split(key)
        low, high = PAUSE_SECONDS_RANGE
        pause = np.asarray(jax.random.uniform(draw_key, (batch,), minval=low, maxval=high), dtype=np.float64)
        gap = np.round(pause / TICK_SECONDS).astype(np.int64)
        ready_next = np.where(throw, tick + gap, state.ready_tick)
    return factors, TurretState(ready_next, key)
