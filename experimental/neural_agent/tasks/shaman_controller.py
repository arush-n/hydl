"""Scripted Trork Shaman for entity 1 of the duel arena (opt-in; ``duel_arena`` defaults are unchanged).

Each tick it reads entity 1's own observation (the perception the body would have of entity 0) and returns that
entity's factors. Rules:

* Hold the distance in the band [10, 15] blocks (Template_Trork_Mage ``DesiredRangedAttackDistanceRange``, asset
  read 2026-10-23): walk back when closer than 10, walk forward when beyond 15, stand still in the band.
* Face the target: the body turns toward the target's bearing, at most 45 degrees a tick.
* Throw (ability slot 0, the rubble throw) when the target is in the band and the ability is idle.

Cadence (the default, ``cadence="asset"``), from the 0.7.2 asset
``Server/NPC/Roles/_Core/Components/Instructions/Combat/Component_Instruction_Attack_Sequence_Temp_Ranged.json``:
``AimingTimeRange [3, 3]`` is the aim held before each throw, and ``AttackPauseRange [1, 2]`` is the pause after it.
The pause is a uniform draw from a seeded key (``initial_state(batch, rng_key)``), so a seed fixes every gap. The
draw is the asset's range, not a fit to the live data. The asset range gives 4 to 5 s per throw (3 s aim, 1 to 2 s
pause), inside the live Shaman's measured 2.7 to 10 s (FIDELITY-0.7.2.md, Trork_Shaman rows).
``cadence="idle"`` is the explicit alternative: a throw whenever the ability is idle and the target is in the band.

Conventions, measured on the duel engine (gen 5, 2026-10-23, probes in the lane scratch dir):

* Perception values are normalised: one block is 1/24 of a unit (``relative_forward``, ``relative_right``,
  ``planar_distance``). The target drops out of view (``visible`` 0) beyond about 20 blocks.
* ``relative_right`` is positive on the body's right. Positive ``body_yaw_delta_bins`` turns the body right
  (a target at +0.25 right fell to 0.02 in three ticks under bin 8).
* ``locomotion_gait_compass`` choice 1 walks in the body's forward direction and choice 5 walks backward
  (a target behind the body got closer under choice 5 and farther under choice 1).
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

#: Template_Trork_Mage DesiredRangedAttackDistanceRange [10, 15] (blocks).
BAND_MIN_BLOCKS = 10.0
BAND_MAX_BLOCKS = 15.0
#: One perception unit per 24 blocks (measured at 7, 16 and 12 blocks; see the module docstring).
BLOCKS_PER_PERCEPTION_UNIT = 24.0
#: Half the perception resolution (the duel's distance rounds to 0.1 block): a reading within this of a band edge is
#: inside the band, so float32 rounding at exactly 10 or 15 blocks does not flip the walk.
EDGE_TOLERANCE_BLOCKS = 0.05
#: One duel step is 1/30 s (the game tick; the cadence converts seconds to steps with it).
TICK_SECONDS = 1.0 / 30.0
#: Component_Instruction_Attack_Sequence_Temp_Ranged: AimingTimeRange [3, 3] (seconds of aim before each throw).
AIM_SECONDS = 3.0
#: The same asset's AttackPauseRange [1, 2]: a uniform draw in seconds after each throw (the asset's range).
PAUSE_SECONDS_RANGE = (1.0, 2.0)
#: Locomotion choices (layout.py ``split_locomotion_choice``): choice 1 = walk, direction 1 (forward);
#: choice 5 = walk, direction 5 (backward).
WALK_FORWARD = 1
WALK_BACK = 5
#: The ability_none_plus_slots choice that requests ability slot 0 (layout.py: choice k+1 is slot k).
THROW_CHOICE = 1
#: Nine-bin look codebook (+-45 degrees a tick at the ends; index 4 is zero).
YAW_BINS = np.asarray(look_delta_values(9), dtype=np.float64)
NO_TURN_BIN = 4
CADENCES = ("asset", "idle")

_I = {name: index for index, name in enumerate(ARSENAL_POLICY_ACTION_HEAD_NAMES)}
_VISIBLE = schema.field_index("target_f32", "visible")
_FORWARD = schema.field_index("target_f32", "relative_forward")
_RIGHT = schema.field_index("target_f32", "relative_right")
_PLANAR = schema.field_index("target_f32", "planar_distance")
_BUSY = schema.field_index("self_f32", "attack_executing")


class ShamanState(NamedTuple):
    """Per-duel memory: the earliest tick of each row's next throw (int64[B]) and the seeded key for the pause draws."""

    ready_tick: np.ndarray
    key: jax.Array | None


def initial_state(batch: int, rng_key: jax.Array | None = None) -> ShamanState:
    """Start a duel. The first throw waits out the asset's aim. ``rng_key`` seeds the pause draws (``asset`` cadence)."""

    return ShamanState(np.full((batch,), round(AIM_SECONDS / TICK_SECONDS), dtype=np.int64), rng_key)


def shaman_factors(
    observation,
    state: ShamanState,
    tick: int,
    *,
    cadence: str = "asset",
) -> tuple[np.ndarray, ShamanState]:
    """Return ``(factors, state)``: int32[B, heads] for entity 1, and the state after this tick.

    ``observation`` is entity 1's observation from the previous step (``obs1`` of ``DuelArena.step``); ``tick`` is the
    count of steps taken so far. ``cadence`` is ``"asset"`` (aim 3.0 s, then a seeded 1 to 2 s pause; needs the key
    from :func:`initial_state`) or ``"idle"`` (throw whenever idle). The factors are host arrays; pass
    ``jnp.asarray(factors, dtype=jnp.int32)`` to step.
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

    # Hold the band: back off inside 10 blocks, close in beyond 15, stand still in between.
    move = np.zeros(batch, dtype=np.int32)
    move = np.where(visible & (blocks < BAND_MIN_BLOCKS - EDGE_TOLERANCE_BLOCKS), WALK_BACK, move)
    move = np.where(visible & (blocks > BAND_MAX_BLOCKS + EDGE_TOLERANCE_BLOCKS), WALK_FORWARD, move)
    factors[:, _I["locomotion_gait_compass"]] = move

    # Throw when in the band, idle, and (asset cadence) past the aim-and-pause gap.
    in_band = (
        visible
        & (blocks >= BAND_MIN_BLOCKS - EDGE_TOLERANCE_BLOCKS)
        & (blocks <= BAND_MAX_BLOCKS + EDGE_TOLERANCE_BLOCKS)
    )
    ready = tick >= state.ready_tick if cadence == "asset" else np.ones(batch, dtype=bool)
    throw = in_band & ~busy & ready
    factors[:, _I["ability_none_plus_slots"]] = np.where(throw, THROW_CHOICE, 0)

    ready_next, key = state.ready_tick, state.key
    if cadence == "asset" and bool(throw.any()):
        if key is None:
            raise ValueError("cadence 'asset' needs the rng_key that initial_state was given")
        key, draw_key = jax.random.split(key)
        low, high = PAUSE_SECONDS_RANGE
        pause = np.asarray(jax.random.uniform(draw_key, (batch,), minval=low, maxval=high), dtype=np.float64)
        gap = np.round((AIM_SECONDS + pause) / TICK_SECONDS).astype(np.int64)
        ready_next = np.where(throw, tick + gap, state.ready_tick)
    return factors, ShamanState(ready_next, key)
