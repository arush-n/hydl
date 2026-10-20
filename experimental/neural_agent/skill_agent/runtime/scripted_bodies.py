"""Scripted bodies (2026-10-19; owner: scripted first, neural later): the skill layer's own bodies for the decision
layer's combat verbs and for ``evader`` and ``idle``, so no stand-in skill acts.

Each body is pure JAX over the batch: ``body(observation, mask, ticks, scalars) -> factors int32[B, 12]``, with its
request's scalar arguments ``scalars`` float32[B, PARAM_WIDTH] in its contract's ``param_names`` order. It reads only
the legal observation (the target's entity row, in the body frame, and its attack phase) and the action mask, sets the
heads of its verb's scopes, leaves every other head neutral, and replaces any choice the mask forbids with the neutral
one. The target is the entity in row 0, the decision layer's convention (``TARGET_ENTITY``). Wind-up and recovery are
the observation's attack phases ``windup`` and ``recovery`` (``contract/surface.py``, ``ATTACK_PHASES``), the signals
the decision layer's combat program reads too.

| Verb | Scopes | What it does |
|---|---|---|
| ``evader`` | locomotion, hands | turns away from the target and runs; sprints once facing away |
| ``sword`` | hands, abilities_magic | presses the weapon's first ability (its primary attack; in the gym a weapon attacks through its ability slots) while the target is within reach (3.5 blocks) and within 60 degrees of the body's front; ``hold_seconds`` is accepted, and charged holds are not built yet |
| ``guard`` | posture_defence | guards while the target winds up within 5 blocks |
| ``punish`` | hands, abilities_magic | the same strike, only while the target is in its recovery |
| ``keep_range`` | locomotion | holds the band ``band_min`` to ``band_max`` blocks (``band_max`` defaults to one block more): runs in when farther, walks out when nearer, faces the target |
| ``idle`` | none | every head neutral |

Conventions (``neural_skills/walk_to/teacher.py``): a target at bearing ``b`` degrees, positive to the right, is
faced by a body-yaw step of ``-b``; the compass direction of a body-frame heading is ``round(b / 45) mod 8 + 1``
(1 forward, then clockwise); a locomotion choice is ``(gait - 1) * 8 + direction``; sprint applies only forward.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.contract.surface import ATTACK_PHASES
from experimental.neural_agent.skill_agent.teaching.teachers import BODY_YAW_HEAD, COMPASS_DIRECTIONS, LOCOMOTION_HEAD
from hytalegym.jax.combat.observation.v1.schema.contract import (
    ENTITY_FLOAT_FEATURES,
    ENTITY_INTEGER_FEATURES,
    NEARBY_ENTITY_RADIUS_BLOCKS,
)
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES,
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
)
from hytalegym.jax.combat.observation.v3.policy.layout import neutral_arsenal_policy_action_factors
from hytalegym.jax.combat.observation.v3.policy.look_deltas import encode_look_delta
from hytalegym.jax.combat.types import GAIT_RUN, GAIT_SPRINT, GAIT_WALK

REACH_BLOCKS = 3.5
"""The strike's reach in blocks, centre to centre, less a 0.2 margin: the iron sword's first ability hit a still
target from 3.4 to 3.7 blocks and never from 3.8 (measured 2026-10-19 on gen 5 and gen 7, open flat, microticks 1).
The opponent's hits land within 3.0, so a strike from 3.0 to 3.5 lands out of its reach."""
FRONT_DEGREES = 60.0
GUARD_RANGE_BLOCKS = 5.0
MAXIMUM_TURN_DEGREES = 45.0
YAW_BINS = ARSENAL_POLICY_ACTION_HEAD_SIZES[ARSENAL_POLICY_ACTION_HEAD_NAMES.index("body_yaw_delta_bins")]
HEAD = {name: index for index, name in enumerate(ARSENAL_POLICY_ACTION_HEAD_NAMES)}
STARTS = tuple(int(value) for value in np.cumsum((0, *ARSENAL_POLICY_ACTION_HEAD_SIZES[:-1])))
PRIMARY_ABILITY = 1
"""The ability head's choice for the weapon's first ability (0 is none). The gym masks the base action's attacks for
the learner (``observation/v3/encoding/encoder.py:470-476``); in the open-flat combat arena this choice took the
opponent from 61 health to 1 (2026-10-19)."""
WINDUP, RECOVERY = ATTACK_PHASES.index("windup"), ATTACK_PHASES.index("recovery")
_FLOAT = {name: index for index, name in enumerate(ENTITY_FLOAT_FEATURES)}
_INTEGER = {name: index for index, name in enumerate(ENTITY_INTEGER_FEATURES)}


class Target(NamedTuple):
    """The target as the observation publishes it: body-frame blocks, bearing in degrees (positive right)."""

    forward: jax.Array
    right: jax.Array
    distance: jax.Array
    bearing: jax.Array
    seen: jax.Array
    phase: jax.Array


def target(observation: Any) -> Target:
    """The target's row 0 of the learner observation."""

    row = observation.base.entity_f32[:, 0]
    forward = row[:, _FLOAT["relative_forward"]] * NEARBY_ENTITY_RADIUS_BLOCKS
    right = row[:, _FLOAT["relative_right"]] * NEARBY_ENTITY_RADIUS_BLOCKS
    return Target(forward, right, jnp.hypot(forward, right), jnp.rad2deg(jnp.arctan2(right, forward)),
                  observation.base.entity_mask[:, 0].astype(jnp.bool_),
                  observation.base.entity_i32[:, 0, _INTEGER["attack_phase"]])


def legal(factors: jax.Array, mask: jax.Array) -> jax.Array:
    """Each head's choice where the flat action mask allows it, else the head's neutral choice."""

    neutral = neutral_arsenal_policy_action_factors(factors.shape[0])
    columns = []
    for head, start in enumerate(STARTS):
        allowed = jnp.take_along_axis(mask[:, start:start + ARSENAL_POLICY_ACTION_HEAD_SIZES[head]],
                                      factors[:, head:head + 1], axis=1)[:, 0]
        columns.append(jnp.where(allowed, factors[:, head], neutral[:, head]))
    return jnp.stack(columns, axis=1).astype(jnp.int32)


def _direction(bearing: jax.Array) -> jax.Array:
    return jnp.mod(jnp.round(bearing / 45.0).astype(jnp.int32), COMPASS_DIRECTIONS) + 1


def _turn(bearing: jax.Array) -> jax.Array:
    """The body-yaw bin that turns the body toward ``bearing``."""

    return encode_look_delta(-bearing, YAW_BINS, maximum_degrees=MAXIMUM_TURN_DEGREES)


def _set(neutral: jax.Array, head: str, value: jax.Array, where: jax.Array) -> jax.Array:
    return neutral.at[:, HEAD[head]].set(jnp.where(where, value, neutral[:, HEAD[head]]).astype(jnp.int32))


def idle(observation: Any, mask: jax.Array, ticks: jax.Array, scalars: jax.Array) -> jax.Array:
    return neutral_arsenal_policy_action_factors(observation.base.entity_mask.shape[0])


def evader(observation: Any, mask: jax.Array, ticks: jax.Array, scalars: jax.Array) -> jax.Array:
    seen = target(observation)
    away = seen.bearing + jnp.where(seen.bearing > 0.0, -180.0, 180.0)  # the bearing directly away
    facing = jnp.abs(away) <= 22.5
    gait = jnp.where(facing, GAIT_SPRINT, GAIT_RUN)
    direction = jnp.where(facing, 1, _direction(away))
    factors = idle(observation, mask, ticks, scalars)
    factors = _set(factors, "locomotion_gait_compass", (gait - 1) * COMPASS_DIRECTIONS + direction, seen.seen)
    factors = _set(factors, "body_yaw_delta_bins", _turn(away), seen.seen)
    return legal(factors, mask)


def _strike(observation: Any, mask: jax.Array, ticks: jax.Array, scalars: jax.Array, when: jax.Array) -> jax.Array:
    seen = target(observation)
    ready = seen.seen & (seen.distance <= REACH_BLOCKS) & (jnp.abs(seen.bearing) <= FRONT_DEGREES) & when
    return legal(_set(idle(observation, mask, ticks, scalars), "ability_none_plus_slots", jnp.int32(PRIMARY_ABILITY),
                      ready), mask)


def sword(observation: Any, mask: jax.Array, ticks: jax.Array, scalars: jax.Array) -> jax.Array:
    return _strike(observation, mask, ticks, scalars, jnp.ones(observation.base.entity_mask.shape[0], jnp.bool_))


def punish(observation: Any, mask: jax.Array, ticks: jax.Array, scalars: jax.Array) -> jax.Array:
    return _strike(observation, mask, ticks, scalars, target(observation).phase == RECOVERY)


def guard(observation: Any, mask: jax.Array, ticks: jax.Array, scalars: jax.Array) -> jax.Array:
    seen = target(observation)
    on = seen.seen & (seen.phase == WINDUP) & (seen.distance <= GUARD_RANGE_BLOCKS)
    return legal(_set(idle(observation, mask, ticks, scalars), "guard_off_on", jnp.int32(1), on), mask)


def keep_range(observation: Any, mask: jax.Array, ticks: jax.Array, scalars: jax.Array) -> jax.Array:
    seen = target(observation)
    near = scalars[:, 0]
    far = jnp.where(scalars[:, 1] > near, scalars[:, 1], near + 1.0)
    closing, opening = seen.seen & (seen.distance > far), seen.seen & (seen.distance < near)
    away = seen.bearing + jnp.where(seen.bearing > 0.0, -180.0, 180.0)
    locomotion = jnp.where(closing, (GAIT_RUN - 1) * COMPASS_DIRECTIONS + _direction(seen.bearing),
                           (GAIT_WALK - 1) * COMPASS_DIRECTIONS + _direction(away))
    factors = idle(observation, mask, ticks, scalars)
    factors = _set(factors, "locomotion_gait_compass", locomotion, closing | opening)
    factors = _set(factors, "body_yaw_delta_bins", _turn(seen.bearing), seen.seen)
    return legal(factors, mask)


BODIES = {"idle": idle, "evader": evader, "sword": sword, "guard": guard, "punish": punish,
          "keep_range": keep_range}
"""Verb -> scripted body."""

assert HEAD["locomotion_gait_compass"] == LOCOMOTION_HEAD and HEAD["body_yaw_delta_bins"] == BODY_YAW_HEAD

__all__ = ["BODIES", "Target", "evader", "guard", "idle", "keep_range", "legal", "punish", "sword", "target"]
