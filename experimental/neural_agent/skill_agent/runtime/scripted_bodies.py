"""Scripted bodies (2026-10-19; owner: scripted first, neural later): the skill layer's own bodies for the decision
layer's combat verbs and for ``evader`` and ``idle``, so no stand-in skill acts.

Each body is pure JAX over the batch: ``body(observation, mask, ticks, scalars, position=None) -> factors
int32[B, 12]``, with its request's scalar arguments ``scalars`` float32[B, PARAM_WIDTH] in its contract's
``param_names`` order and its request's world ``position`` (``Position``). It reads only the legal observation (the
target's entity row, in the body frame, and its attack phase) and the action mask, sets the heads of its verb's
scopes, leaves every other head neutral, and replaces any choice the mask forbids with the neutral one. The target is
the entity in row 0, the decision layer's convention (``TARGET_ENTITY``). Wind-up and recovery are the observation's
attack phases ``windup`` and ``recovery`` (``contract/surface.py``, ``ATTACK_PHASES``), the signals the decision
layer's combat program reads too.

A target out of sight is remembered: when the row is unseen and the request carries a position (the decision layer
passes the target's last-known world position, 2026-10-19 night), ``target`` turns it into the body frame with the
perception channel's own position and body yaw (``perception_v2``: ``self_position``, ``body_yaw_sin``,
``body_yaw_cos``; gen 7). It is then ``known`` but not ``seen``: a body can move toward it or away from it, and no
body strikes or guards what it cannot see.

| Verb | Scopes | What it does |
|---|---|---|
| ``evader`` | locomotion, hands | turns away from the target, seen or remembered, and runs; sprints once facing away; with the threat neither seen nor remembered (behind it, out of the view sector) it keeps sprinting the way it faces, which it turned away from the threat |
| ``sword`` | hands, abilities_magic | presses the weapon's first ability (its primary attack; in the gym a weapon attacks through its ability slots) while the target is within reach (3.5 blocks) and within 60 degrees of the body's front; ``hold_seconds`` is accepted, and charged holds are not built yet |
| ``guard`` | posture_defence | guards while the target winds up within 5 blocks |
| ``punish`` | hands, abilities_magic | the same strike, only while the target is in its recovery |
| ``keep_range`` | locomotion | holds the band ``band_min`` to ``band_max`` blocks (``band_max`` defaults to one block more) from the target, seen or remembered: runs in when farther, walks out when nearer, faces the target |
| ``avoid`` | locomotion | retreats from the target, seen or remembered, until beyond the band (``band_min``, ``band_max``; ``AVOID_BAND_BLOCKS`` when the request names none): sprints inside the near edge, runs inside the far edge, beyond it holds and faces the target; with the threat neither seen nor remembered it keeps running the way it faces |
| ``hide`` | locomotion | with a cover point in its request (``position``), runs to it, sneaks the last blocks and stops there; with none, retreats from the threat as ``avoid`` does (cover found from the geometry comes with the cover arena) |
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
from hytalegym.jax.combat.types import GAIT_RUN, GAIT_SNEAK, GAIT_SPRINT, GAIT_WALK

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


class Position(NamedTuple):
    """A request's world position (the game's frame) and whether the request carries one."""

    world: jax.Array  # float32[B, 3]
    present: jax.Array  # bool[B]


class Target(NamedTuple):
    """The target in the body frame: blocks, bearing in degrees (positive right). ``seen``: its row is published;
    ``known``: seen, or remembered from the request's position."""

    forward: jax.Array
    right: jax.Array
    distance: jax.Array
    bearing: jax.Array
    seen: jax.Array
    phase: jax.Array
    known: jax.Array


def body_frame(observation: Any, world: jax.Array) -> tuple[jax.Array, jax.Array, bool]:
    """A world point [B, 3] as (forward, right) blocks in the body frame, from the perception channel's own position
    and body yaw; the frame is the gym's (forward = (-sin yaw, 0, -cos yaw), right = (cos yaw, 0, -sin yaw)). The flag
    is False, and both zero, when the observation carries no ``self_position``."""

    channel = getattr(observation, "perception_v2", None)
    if channel is None or not hasattr(channel, "self_position"):
        zero = jnp.zeros(observation.base.entity_mask.shape[0], jnp.float32)
        return zero, zero, False
    yaw = jnp.arctan2(channel.body_yaw_sin, channel.body_yaw_cos)
    offset = jnp.asarray(world, jnp.float32) - jnp.asarray(channel.self_position, jnp.float32)
    return (-offset[:, 0] * jnp.sin(yaw) - offset[:, 2] * jnp.cos(yaw),
            offset[:, 0] * jnp.cos(yaw) - offset[:, 2] * jnp.sin(yaw), True)


def target(observation: Any, position: Position | None = None) -> Target:
    """The target's row 0 of the learner observation; while the row is unseen, the request's remembered position
    (module docstring)."""

    row = observation.base.entity_f32[:, 0]
    forward = row[:, _FLOAT["relative_forward"]] * NEARBY_ENTITY_RADIUS_BLOCKS
    right = row[:, _FLOAT["relative_right"]] * NEARBY_ENTITY_RADIUS_BLOCKS
    seen = observation.base.entity_mask[:, 0].astype(jnp.bool_)
    known = seen
    if position is not None:
        remembered_forward, remembered_right, available = body_frame(observation, position.world)
        if available:
            remembered = ~seen & jnp.asarray(position.present, jnp.bool_)
            forward = jnp.where(remembered, remembered_forward, forward)
            right = jnp.where(remembered, remembered_right, right)
            known = seen | remembered
    return Target(forward, right, jnp.hypot(forward, right), jnp.rad2deg(jnp.arctan2(right, forward)), seen,
                  observation.base.entity_i32[:, 0, _INTEGER["attack_phase"]], known)


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


def idle(observation: Any, mask: jax.Array, ticks: jax.Array, scalars: jax.Array,
         position: Position | None = None) -> jax.Array:
    return neutral_arsenal_policy_action_factors(observation.base.entity_mask.shape[0])


def _flee(observation: Any, mask: jax.Array, ticks: jax.Array, scalars: jax.Array, threat: Target, flee: jax.Array,
          fast: jax.Array) -> jax.Array:
    """Run directly away from ``threat`` where ``flee``: sprinting (where ``fast``) once facing away, else running in
    the away sector; the body turns away."""

    away = threat.bearing + jnp.where(threat.bearing > 0.0, -180.0, 180.0)  # the bearing directly away
    facing = jnp.abs(away) <= 22.5
    gait = jnp.where(facing & fast, GAIT_SPRINT, GAIT_RUN)
    direction = jnp.where(facing, 1, _direction(away))
    factors = idle(observation, mask, ticks, scalars)
    factors = _set(factors, "locomotion_gait_compass", (gait - 1) * COMPASS_DIRECTIONS + direction, flee)
    return _set(factors, "body_yaw_delta_bins", _turn(away), flee)


def _onward(factors: jax.Array, gait: int, where: jax.Array) -> jax.Array:
    """Keep moving the way the body faces where ``where``: a fleeing body that can no longer place its threat turned
    away from it while it could, so forward is away (measured 2026-10-20: standing still instead left the referee's
    evader granted for most of a fight, taking hits it never saw)."""

    return _set(factors, "locomotion_gait_compass", jnp.int32((gait - 1) * COMPASS_DIRECTIONS + 1), where)


def evader(observation: Any, mask: jax.Array, ticks: jax.Array, scalars: jax.Array,
           position: Position | None = None) -> jax.Array:
    threat = target(observation, position)
    factors = _flee(observation, mask, ticks, scalars, threat, threat.known, threat.known)
    return legal(_onward(factors, GAIT_SPRINT, ~threat.known), mask)


def _strike(observation: Any, mask: jax.Array, ticks: jax.Array, scalars: jax.Array, when: jax.Array) -> jax.Array:
    seen = target(observation)
    ready = seen.seen & (seen.distance <= REACH_BLOCKS) & (jnp.abs(seen.bearing) <= FRONT_DEGREES) & when
    return legal(_set(idle(observation, mask, ticks, scalars), "ability_none_plus_slots", jnp.int32(PRIMARY_ABILITY),
                      ready), mask)


def sword(observation: Any, mask: jax.Array, ticks: jax.Array, scalars: jax.Array,
          position: Position | None = None) -> jax.Array:
    return _strike(observation, mask, ticks, scalars, jnp.ones(observation.base.entity_mask.shape[0], jnp.bool_))


def punish(observation: Any, mask: jax.Array, ticks: jax.Array, scalars: jax.Array,
           position: Position | None = None) -> jax.Array:
    return _strike(observation, mask, ticks, scalars, target(observation).phase == RECOVERY)


def guard(observation: Any, mask: jax.Array, ticks: jax.Array, scalars: jax.Array,
          position: Position | None = None) -> jax.Array:
    seen = target(observation)
    on = seen.seen & (seen.phase == WINDUP) & (seen.distance <= GUARD_RANGE_BLOCKS)
    return legal(_set(idle(observation, mask, ticks, scalars), "guard_off_on", jnp.int32(1), on), mask)


def keep_range(observation: Any, mask: jax.Array, ticks: jax.Array, scalars: jax.Array,
               position: Position | None = None) -> jax.Array:
    seen = target(observation, position)
    near = scalars[:, 0]
    far = jnp.where(scalars[:, 1] > near, scalars[:, 1], near + 1.0)
    closing, opening = seen.known & (seen.distance > far), seen.known & (seen.distance < near)
    away = seen.bearing + jnp.where(seen.bearing > 0.0, -180.0, 180.0)
    locomotion = jnp.where(closing, (GAIT_RUN - 1) * COMPASS_DIRECTIONS + _direction(seen.bearing),
                           (GAIT_WALK - 1) * COMPASS_DIRECTIONS + _direction(away))
    factors = idle(observation, mask, ticks, scalars)
    factors = _set(factors, "locomotion_gait_compass", locomotion, closing | opening)
    factors = _set(factors, "body_yaw_delta_bins", _turn(seen.bearing), seen.known)
    return legal(factors, mask)


AVOID_BAND_BLOCKS = (6.0, 10.0)
"""``avoid``'s band (near, far) in blocks when its request names none."""
HIDE_ARRIVED_BLOCKS = 0.5
"""``hide`` stops within this distance of its cover point."""
HIDE_SNEAK_BLOCKS = 3.0
"""``hide`` sneaks the last blocks to its cover point, and runs before them."""


def avoid(observation: Any, mask: jax.Array, ticks: jax.Array, scalars: jax.Array,
          position: Position | None = None) -> jax.Array:
    threat = target(observation, position)
    near = jnp.where(scalars[:, 0] > 0.0, scalars[:, 0], AVOID_BAND_BLOCKS[0])
    far = jnp.where(scalars[:, 1] > near, scalars[:, 1], jnp.maximum(near, AVOID_BAND_BLOCKS[1]))
    retreat = threat.known & (threat.distance < far)
    factors = _flee(observation, mask, ticks, scalars, threat, retreat, threat.distance < near)
    factors = _set(factors, "body_yaw_delta_bins", _turn(threat.bearing), threat.known & ~retreat)
    return legal(_onward(factors, GAIT_RUN, ~threat.known), mask)


def hide(observation: Any, mask: jax.Array, ticks: jax.Array, scalars: jax.Array,
         position: Position | None = None) -> jax.Array:
    fallback = avoid(observation, mask, ticks, scalars, None)
    if position is None:
        return fallback
    forward, right, available = body_frame(observation, position.world)
    if not available:
        return fallback
    distance, bearing = jnp.hypot(forward, right), jnp.rad2deg(jnp.arctan2(right, forward))
    has_point = jnp.asarray(position.present, jnp.bool_)
    moving = has_point & (distance > HIDE_ARRIVED_BLOCKS)
    gait = jnp.where(distance > HIDE_SNEAK_BLOCKS, GAIT_RUN, GAIT_SNEAK)
    factors = idle(observation, mask, ticks, scalars)
    factors = _set(factors, "locomotion_gait_compass", (gait - 1) * COMPASS_DIRECTIONS + _direction(bearing), moving)
    factors = _set(factors, "body_yaw_delta_bins", _turn(bearing), moving)
    return legal(jnp.where(has_point[:, None], factors, fallback), mask)


BODIES = {"idle": idle, "evader": evader, "sword": sword, "guard": guard, "punish": punish,
          "keep_range": keep_range, "avoid": avoid, "hide": hide}
"""Verb -> scripted body."""

FACING_ME_DEGREES = 60.0
"""A target faces the agent when its head turns within this angle of the bearing to the agent: half the 120-degree
view sector of the gym's legal observation, so the agent stands in the sector the target looks through. The entity
row publishes the angle as ``head_facing_error``, degrees over 180."""

REPORTS: tuple[str, ...] = ("seen", "distance", "facing_me", "in_reach", "acting")
"""What a body reports each tick, by name; the list is append-only, as the verb ids are (IDEAS.md idea 8). ``seen``:
the target is perceived (0 or 1). ``distance``: its distance in blocks, 0 when unseen. ``facing_me``: it faces the
agent (``FACING_ME_DEGREES``). ``in_reach``: a strike would land (the strike's gate: within ``REACH_BLOCKS`` and
``FRONT_DEGREES``). ``acting``: the body moves some head away from neutral this tick; a granted body that never acts
is waiting, for reach, a wind-up or a sight of the target."""


def reports(observation: Any, factors: jax.Array, neutral: jax.Array) -> jax.Array:
    """``REPORTS`` of one body this tick, float32[B, len(REPORTS)], from the legal observation and the body's choices
    ``factors`` int32[B, 12] against the neutral choices ``neutral``."""

    seen = target(observation)
    facing = jnp.abs(observation.base.entity_f32[:, 0, _FLOAT["head_facing_error"]]) * 180.0 <= FACING_ME_DEGREES
    in_reach = seen.seen & (seen.distance <= REACH_BLOCKS) & (jnp.abs(seen.bearing) <= FRONT_DEGREES)
    columns = (seen.seen, jnp.where(seen.seen, seen.distance, 0.0), seen.seen & facing, in_reach,
               jnp.any(factors != neutral, axis=1))
    return jnp.stack([jnp.asarray(column, jnp.float32) for column in columns], axis=1)

assert HEAD["locomotion_gait_compass"] == LOCOMOTION_HEAD and HEAD["body_yaw_delta_bins"] == BODY_YAW_HEAD

__all__ = ["AVOID_BAND_BLOCKS", "BODIES", "FACING_ME_DEGREES", "HIDE_ARRIVED_BLOCKS", "HIDE_SNEAK_BLOCKS", "REPORTS",
           "Position", "Target", "avoid", "body_frame", "evader", "guard", "hide", "idle", "keep_range", "legal",
           "punish", "reports", "sword", "target"]
