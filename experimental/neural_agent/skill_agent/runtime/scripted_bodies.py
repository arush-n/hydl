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
| ``sword`` | hands, abilities_magic | presses the weapon's first ability (its primary attack; in the gym a weapon attacks through its ability slots) while the target is within the weapon's reach (``reach``: 3.5 blocks for the sword and the mace, 3.25 for the battleaxe, 2.2 for the daggers) and within 60 degrees of the body's front; ``hold_seconds`` is accepted, and charged holds are not built yet |
| ``guard`` | posture_defence | guards while the target winds up within 5 blocks |
| ``punish`` | hands, abilities_magic | the same strike, only while the target is in its recovery |
| ``keep_range`` | locomotion | holds the band ``band_min`` to ``band_max`` blocks (``band_max`` defaults to one block more) from the target, seen or remembered: runs in when farther, walks out when nearer, faces the target |
| ``avoid`` | locomotion | retreats from the target, seen or remembered, until beyond the band (``band_min``, ``band_max``; ``AVOID_BAND_BLOCKS`` when the request names none): sprints inside the near edge, runs inside the far edge, beyond it holds and faces the target; with the threat neither seen nor remembered it keeps running the way it faces |
| ``hide`` | locomotion | with a cover point in its request (``position``), runs to it, sneaks the last blocks and stops there; with none, retreats from the threat as ``avoid`` does (cover found from the geometry comes with the cover arena) |
| ``shortbow`` | hands, abilities_magic, gaze | aims the head where the seen target will be when the arrow arrives (its velocity times the flight), the pitch from the engine's own arrow flight for the draw tier ``hold_seconds`` selects, and taps the tier once the aim is within a degree; holds fire while the target is unseen or out of the tier's range |
| ``idle`` | none | every head neutral |

Conventions (``neural_skills/walk_to/teacher.py``): a target at bearing ``b`` degrees, positive to the right, is
faced by a body-yaw step of ``-b``; the compass direction of a body-frame heading is ``round(b / 45) mod 8 + 1``
(1 forward, then clockwise); a locomotion choice is ``(gait - 1) * 8 + direction``; sprint applies only forward.
"""

from __future__ import annotations

from functools import cache
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.contract.surface import ATTACK_PHASES
from experimental.neural_agent.skill_agent.teaching.teachers import BODY_YAW_HEAD, COMPASS_DIRECTIONS, LOCOMOTION_HEAD
from hytalegym.jax.combat import default_combat_params, hytale_0_5_7_loadouts
from hytalegym.jax.combat.arsenal.schema import contract as arsenal_contract
from hytalegym.jax.combat.observation.v1.schema.contract import (
    COMBAT_FLOAT_FEATURES,
    ENTITY_FLOAT_FEATURES,
    ENTITY_INTEGER_FEATURES,
    NEARBY_ENTITY_RADIUS_BLOCKS,
    SELF_FLOAT_FEATURES,
)
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES,
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
)
from hytalegym.jax.combat.observation.v3.policy.layout import neutral_arsenal_policy_action_factors
from hytalegym.jax.combat.observation.v3.policy.look_deltas import encode_look_delta
from hytalegym.jax.combat.observation.v3.schema.contract import ABILITY_INTEGER_FEATURES
from hytalegym.jax.combat.types import GAIT_RUN, GAIT_SNEAK, GAIT_SPRINT, GAIT_WALK

REACH_BLOCKS = 3.5
"""The strike's reach in blocks, centre to centre, less a 0.2 margin: the iron sword's first ability hit a still
target from 3.4 to 3.7 blocks and never from 3.8 (measured 2026-10-19 on gen 5 and gen 7, open flat, microticks 1).
The opponent's hits land within 3.0, so a strike from 3.0 to 3.5 lands out of its reach. It is also the reach of a
weapon ``MELEE_REACH_BLOCKS`` does not list."""
MELEE_REACH_BLOCKS: dict[str, float] = {"iron_sword": 3.5, "iron_daggers": 2.2, "iron_mace": 3.5,
                                        "iron_battleaxe": 3.25}
"""Each melee weapon's strike reach, centre to centre, less a 0.25 margin: its first ability's longest hit on a still
body straight ahead, one tap in 0.05-block steps (measured 2026-10-20, the same on gen 5 and the gen-7 render and
for either attacker role): 3.75, 2.45, 3.75 and 3.5 blocks against a Trork; 3.55, 2.3, 3.55 and 3.35 against the
smaller Kweebec, which every entry also clears. Not the selector's end distance (2.75, 2.0, 3.0 and 3.0): the reach
beyond it differs by weapon. ``reach`` picks the entry by the first ability's id."""
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


_ABILITY_ID = ABILITY_INTEGER_FEATURES.index("ability_id")


@cache
def _reach_table() -> tuple[np.ndarray, np.ndarray]:
    """(the first ability's id, its reach) for each weapon of ``MELEE_REACH_BLOCKS``, the ids read once from the
    running engine's compiled loadouts."""

    names = list(MELEE_REACH_BLOCKS)
    with jax.ensure_compile_time_eval():  # concrete even when first called inside a traced body
        loadout = hytale_0_5_7_loadouts(names)
    return (np.asarray(loadout.ability_id)[:, 0, 0],
            np.asarray([MELEE_REACH_BLOCKS[name] for name in names], np.float32))


def reach(observation: Any) -> jax.Array:
    """float32[B]: the strike reach of the weapon the body holds, found by its first ability's id in the body's own
    ability rows (``MELEE_REACH_BLOCKS``; ``REACH_BLOCKS`` for a weapon it does not list)."""

    ids, reaches = _reach_table()
    held = observation.ability_i32[:, 0, 0, _ABILITY_ID]  # own row, slot 0
    match = held[:, None] == jnp.asarray(ids)[None, :]
    return jnp.where(match.any(axis=1), jnp.asarray(reaches)[jnp.argmax(match, axis=1)], jnp.float32(REACH_BLOCKS))


def _strike(observation: Any, mask: jax.Array, ticks: jax.Array, scalars: jax.Array, when: jax.Array) -> jax.Array:
    seen = target(observation)
    ready = seen.seen & (seen.distance <= reach(observation)) & (jnp.abs(seen.bearing) <= FRONT_DEGREES) & when
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


SHORTBOW_THRESHOLD_SECONDS = (0.1, 0.3, 0.6, 0.9, 1.2)
"""The iron shortbow's authored draw thresholds, tiers 0 to 4, the same on 0.5.7 and 0.7.2 (the gym's content adapter
``shortbow_tiers``). ``hold_seconds`` takes the largest tier whose threshold is at most the hold, and a hold below the
first threshold is tier 0 (the contract, ``requests.py``). In the gym tier k is one tap of ability slot k, so the
ability head's choice is k + 1, and the draw time is the tier's own (its arrow spawns 9, 15, 24, 33 and 42 ticks after
the tap on gen 5, a tick later on gen 7). 0.7.2 authors a branch at 0.0 that fires nothing; the gym has no slot for
that release, so the body's shortest draw is tier 0."""
ARROW_LAUNCH_HEIGHTS = (0.75, 1.318)
"""Measured height of the arrow's launch point above the shooter's feet, for the Kweebec and the Trork (2026-10-20):
the eye (1.0 and 1.5675) less 0.25, and 0.15 to the right, along the head's yaw at its pitch plus the tier's pitch
offset (2 degrees)."""
ASSUMED_LAUNCH_HEIGHT = sum(ARROW_LAUNCH_HEIGHTS) / len(ARROW_LAUNCH_HEIGHTS)
"""The launch height the aim assumes (1.034): the observation publishes no eye height, so the mean of the two roles."""
AIM_HEIGHT_BLOCKS = 0.7
"""Where the aim meets the target, above its feet: the middle of the smaller box (the Kweebec's, 1.4 tall; the
Trork's is 1.805). With the assumed launch height an arrow arrives 0.42 above the feet from a Kweebec and 0.98 from a
Trork, inside both boxes."""
AIM_TOLERANCE_DEGREES = 1.0
"""The shortbow taps once the head is within this much of the aim in yaw and in pitch (the look heads' finest step)."""
ARROW_TICK_SECONDS = 1.0 / 30.0
"""The engine's projectile step (``params.nominal_dt``): an arrow moves a fixed step per engine tick."""
ARROW_TICKS = 64
"""How far the aim follows an arrow: 64 engine ticks, past 16 blocks (the legal sensor range) for every tier that
reaches them."""
ARROW_PITCHES = tuple(float(value) for value in np.arange(-45.0, 47.0001, 0.25))
"""The arrow pitches the aim searches, degrees: the head's pitch range plus the tiers' 2-degree offset."""
LEAD_LAG_TICKS = 1.0
"""The shortbow leads a moving target by its flight time plus this many ticks: a choice acts a tick after the
observation it answers. Measured 2026-10-20 against a body running across the line of fire (gen 5, tiers 1, 2 and 4
at 4.5 and 8 blocks): lags of 0, 1 and 2 ticks hit 86, 88 and 82% of 51 arrows; without lead, 2 of 20 for tiers 1
and 2 and 0 of 4 for tier 4 at 4.5 blocks."""
PITCH_BINS = ARSENAL_POLICY_ACTION_HEAD_SIZES[ARSENAL_POLICY_ACTION_HEAD_NAMES.index("pitch_delta_bins")]
_SELF = {name: index for index, name in enumerate(SELF_FLOAT_FEATURES)}
_COMBAT = {name: index for index, name in enumerate(COMBAT_FLOAT_FEATURES)}


class BowTiers(NamedTuple):
    """The equipped iron shortbow's tiers 0 to 4 as the running engine compiled them: launch speed (blocks a second),
    gravity, terminal velocity and the launch pitch offset (degrees), float32[5] each."""

    speed: np.ndarray
    gravity: np.ndarray
    terminal: np.ndarray
    pitch_offset: np.ndarray


@cache
def bow_tiers() -> BowTiers:
    """Read once from the iron shortbow's compiled loadout, so the aim follows the engine it runs on (speeds 15, 30,
    50, 70 and 85 on gen 5; 12, 22, 35, 50 and 85 on the gen-7 render). The agent knows its own bow; nothing here is
    about the world or the opponent."""

    with jax.ensure_compile_time_eval():  # concrete even when first called inside a traced body
        loadout = hytale_0_5_7_loadouts("iron_shortbow")
    rows = np.asarray(loadout.event_f32)[0, 0]
    live = np.asarray(loadout.event_mask)[0, 0]
    tiers = []
    for slot in range(len(SHORTBOW_THRESHOLD_SECONDS)):
        launches = [row for row, present in zip(rows[slot], live[slot])
                    if present and row[arsenal_contract.EF_PROJECTILE_SPEED] > 0.0]
        tiers.append(launches[0])
    table = np.asarray(tiers, np.float32)
    return BowTiers(table[:, arsenal_contract.EF_PROJECTILE_SPEED], table[:, arsenal_contract.EF_PROJECTILE_GRAVITY],
                    table[:, arsenal_contract.EF_PROJECTILE_TERMINAL_VELOCITY],
                    table[:, arsenal_contract.EF_PITCH_OFFSET_DEGREES])


@cache
def velocity_scales() -> tuple[float, float]:
    """The running engine's velocity normalisers: the entity row divides the target's relative velocity by
    ``target_chase_speed`` and the self row the agent's own by ``agent_maximum_gait_speed``
    (``observation/v1/runtime/encoder.py``)."""

    with jax.ensure_compile_time_eval():
        params = default_combat_params(microticks=1)
    return float(params.target_chase_speed), float(params.agent_maximum_gait_speed)


class ArrowAim(NamedTuple):
    """``arrow_pitch``'s answer per lane: the head pitch in degrees, whether the tier reaches, and the arrow's flight
    to the target in engine ticks (0 where it does not reach)."""

    pitch: jax.Array
    reachable: jax.Array
    flight_ticks: jax.Array


def arrow_pitch(tier: jax.Array, distance: jax.Array, rise: jax.Array, heading: jax.Array) -> ArrowAim:
    """The lowest head pitch (degrees) whose arrow of ``tier`` passes ``rise`` blocks above its launch point when it
    has flown ``distance`` horizontal blocks, whether any pitch does, and how many ticks that arrow flies. ``heading``
    is ``|sin yaw| + |cos yaw|`` of the head's world yaw: the engine drags the arrow's 0.2-block cube by the sum of its
    axis speeds.

    The flight is the engine's (``StandardPhysicsTickSystem``, ``jax/combat/arsenal/projectiles/runtime.py``,
    ``_standard_velocity``): each tick the arrow moves by its velocity, then loses ``gravity * (|vx| + |vy| + |vz|) /
    terminal**2`` of it per second and falls by ``gravity``. It reproduced the measured gen-5 and gen-7 tracks of
    every tier within 0.03 blocks (2026-10-20). The search runs the candidate pitches ``ARROW_PITCHES`` side by
    side and interpolates between the two that bracket ``rise`` on the rising branch."""

    tiers = bow_tiers()
    speed, gravity = jnp.asarray(tiers.speed)[tier], jnp.asarray(tiers.gravity)[tier]
    terminal, offset = jnp.asarray(tiers.terminal)[tier], jnp.asarray(tiers.pitch_offset)[tier]
    pitches = jnp.asarray(ARROW_PITCHES, jnp.float32)
    radians = jnp.deg2rad(pitches)[None, :]
    horizontal, vertical = speed[:, None] * jnp.cos(radians), speed[:, None] * jnp.sin(radians)
    drag = gravity[:, None] / (terminal[:, None] ** 2)
    reach = distance[:, None]

    def tick(carry, index):
        x, y, horizontal, vertical, height, when = carry
        x_next, y_next = x + horizontal * ARROW_TICK_SECONDS, y + vertical * ARROW_TICK_SECONDS
        crossing = jnp.isnan(height) & (x < reach) & (x_next >= reach)
        fraction = (reach - x) / jnp.maximum(x_next - x, 1e-9)
        height = jnp.where(crossing, y + fraction * (y_next - y), height)
        when = jnp.where(crossing, index + fraction, when)
        rate = drag * (horizontal * heading[:, None] + jnp.abs(vertical))
        horizontal = horizontal - horizontal * rate * ARROW_TICK_SECONDS
        vertical = vertical - vertical * rate * ARROW_TICK_SECONDS - gravity[:, None] * ARROW_TICK_SECONDS
        return (x_next, y_next, horizontal, vertical, height, when), None

    zero = jnp.zeros_like(horizontal)
    unset = jnp.full_like(zero, jnp.nan)
    (*_, height, when), _ = jax.lax.scan(tick, (zero, zero, horizontal, vertical, unset, unset),
                                         jnp.arange(ARROW_TICKS, dtype=jnp.float32))
    above = ~jnp.isnan(height) & (height >= rise[:, None])
    reachable = jnp.any(above, axis=1)
    first = jnp.argmax(above, axis=1)
    before = jnp.maximum(first - 1, 0)

    def at(values, index):
        return jnp.take_along_axis(values, index[:, None], axis=1)[:, 0]

    high, low = at(height, first), at(height, before)
    fraction = jnp.clip(jnp.where((first > 0) & ~jnp.isnan(low) & (high > low),
                                  (rise - low) / jnp.maximum(high - low, 1e-9), 1.0), 0.0, 1.0)
    arrow = pitches[before] + fraction * (pitches[first] - pitches[before])
    early, late = at(when, before), at(when, first)
    flight = jnp.where(jnp.isnan(early), late, early + fraction * (late - early))
    return ArrowAim(arrow - offset, reachable, jnp.where(reachable, flight, 0.0))


def draw_tier(hold_seconds: jax.Array) -> jax.Array:
    """The shortbow tier, 0 to 4, that ``hold_seconds`` selects (``SHORTBOW_THRESHOLD_SECONDS``)."""

    thresholds = jnp.asarray(SHORTBOW_THRESHOLD_SECONDS, jnp.float32)
    hold = jnp.maximum(jnp.asarray(hold_seconds, jnp.float32), thresholds[0])
    return (jnp.sum(hold[:, None] >= thresholds[None, :], axis=1) - 1).astype(jnp.int32)


def shortbow(observation: Any, mask: jax.Array, ticks: jax.Array, scalars: jax.Array,
             position: Position | None = None) -> jax.Array:
    """Aim where the seen target will be when the arrow arrives, and loose the tier ``hold_seconds`` selects.

    The target's place now is its entity row (body frame); the head's yaw off the body is the difference between the
    target's bearing in the head's frame (the combat prefix, which the engine encodes from the head yaw and blanks
    while the target is not perceptible) and in the body's. The target's own velocity is its relative velocity plus
    the shooter's (an arrow keeps none of the shooter's motion, measured 2026-10-20), and the aim point is its place
    ``flight + LEAD_LAG_TICKS`` ticks ahead, the flight from ``arrow_pitch`` (solved twice, the second time for the
    led point). The pitch is ``arrow_pitch`` for that point, from the assumed launch height to ``AIM_HEIGHT_BLOCKS``
    above the target's feet. The tap waits until both errors are within ``AIM_TOLERANCE_DEGREES``; the head keeps
    following while the arrow is drawn, and the engine admits the next tap once the shot is over. The arrow leaves 0.15
    to the right of the eye and converges by a quarter of a degree, so it passes 0.12 to 0.15 blocks right of the aim
    at 4 to 8 blocks, inside the target's half width."""

    seen = target(observation)
    tier = draw_tier(scalars[:, 0])
    prefix = observation.base.combat_f32
    perceived = seen.seen & (prefix[:, _COMBAT["target_visible"]] > 0.5)
    head_bearing = jnp.rad2deg(jnp.arctan2(prefix[:, _COMBAT["visible_target_right"]],
                                           prefix[:, _COMBAT["visible_target_forward"]]))
    head_offset = head_bearing - seen.bearing  # degrees the head is turned left of the body
    row = observation.base.entity_f32[:, 0]
    self_row = observation.base.self_f32
    target_scale, gait_scale = velocity_scales()
    own_forward, own_right = self_row[:, _SELF["forward_velocity"]], self_row[:, _SELF["right_velocity"]]
    own_bearing = jnp.arctan2(own_right, own_forward) - jnp.deg2rad(head_offset)  # the head frame to the body's
    own_speed = jnp.hypot(own_forward, own_right) * gait_scale
    velocity_forward = row[:, _FLOAT["relative_velocity_forward"]] * target_scale + own_speed * jnp.cos(own_bearing)
    velocity_right = row[:, _FLOAT["relative_velocity_right"]] * target_scale + own_speed * jnp.sin(own_bearing)
    heading = jnp.abs(self_row[:, _SELF["yaw_sin"]]) + jnp.abs(self_row[:, _SELF["yaw_cos"]])
    rise = row[:, _FLOAT["relative_up"]] * NEARBY_ENTITY_RADIUS_BLOCKS + AIM_HEIGHT_BLOCKS - ASSUMED_LAUNCH_HEIGHT
    forward, right = seen.forward, seen.right
    for _ in range(2):
        lead = (arrow_pitch(tier, jnp.hypot(forward, right), rise, heading).flight_ticks + LEAD_LAG_TICKS) \
            * ARROW_TICK_SECONDS
        forward, right = seen.forward + velocity_forward * lead, seen.right + velocity_right * lead
    aim = arrow_pitch(tier, jnp.hypot(forward, right), rise, heading)
    yaw_error = jnp.rad2deg(jnp.arctan2(right, forward)) + head_offset  # the led point's bearing in the head's frame
    pitch_error = aim.pitch - self_row[:, _SELF["pitch"]] * 90.0
    aiming = perceived & aim.reachable
    aimed = (aiming & (jnp.abs(yaw_error) <= AIM_TOLERANCE_DEGREES)
             & (jnp.abs(pitch_error) <= AIM_TOLERANCE_DEGREES))
    factors = idle(observation, mask, ticks, scalars)
    factors = _set(factors, "yaw_delta_bins", _turn(yaw_error), aiming)
    factors = _set(factors, "pitch_delta_bins",
                   encode_look_delta(pitch_error, PITCH_BINS, maximum_degrees=MAXIMUM_TURN_DEGREES), aiming)
    return legal(_set(factors, "ability_none_plus_slots", tier + 1, aimed), mask)


BODIES = {"idle": idle, "evader": evader, "sword": sword, "guard": guard, "punish": punish,
          "keep_range": keep_range, "avoid": avoid, "hide": hide, "shortbow": shortbow}
"""Verb -> scripted body."""

POSITION_BODIES: frozenset[str] = frozenset({"evader", "keep_range", "avoid", "hide"})
"""The bodies that read their request's position: the first three as the remembered target, ``hide`` as its cover
point. The others ignore it (the shortbow aims at the seen row only)."""

FACING_ME_DEGREES = 60.0
"""A target faces the agent when its head turns within this angle of the bearing to the agent: half the 120-degree
view sector of the gym's legal observation, so the agent stands in the sector the target looks through. The entity
row publishes the angle as ``head_facing_error``, degrees over 180."""

REPORTS: tuple[str, ...] = ("seen", "distance", "facing_me", "in_reach", "acting", "position_used")
"""What a body reports each tick, by name; the list is append-only, as the verb ids are (IDEAS.md idea 8). ``seen``:
the target is perceived (0 or 1). ``distance``: its distance in blocks, 0 when unseen. ``facing_me``: it faces the
agent (``FACING_ME_DEGREES``). ``in_reach``: a strike would land (the strike's gate: within the weapon's ``reach``
and ``FRONT_DEGREES``). ``acting``: the body moves some head away from neutral this tick; a granted body that never acts
is waiting, for reach, a wind-up or a sight of the target. ``position_used`` (2026-10-20): the body can place its
request's position, which needs a request that carries one, a body that reads it (``POSITION_BODIES``) and the
perception channel's own position in the observation (``body_frame``; the gen-7 ``perception_v2`` construct). A
request that sent a position and reads 0 has it ignored: a remembered target counts as unknown, and ``hide`` takes
``avoid``'s retreat."""


def reports(observation: Any, factors: jax.Array, neutral: jax.Array, position: Position | None = None) -> jax.Array:
    """``REPORTS`` of one body this tick, float32[B, len(REPORTS)], from the legal observation and the body's choices
    ``factors`` int32[B, 12] against the neutral choices ``neutral``. ``position``: the granted request's position for
    a body in ``POSITION_BODIES``, None for the others."""

    seen = target(observation)
    facing = jnp.abs(observation.base.entity_f32[:, 0, _FLOAT["head_facing_error"]]) * 180.0 <= FACING_ME_DEGREES
    in_reach = seen.seen & (seen.distance <= reach(observation)) & (jnp.abs(seen.bearing) <= FRONT_DEGREES)
    used = jnp.zeros_like(seen.seen)
    if position is not None:
        used = jnp.asarray(position.present, jnp.bool_) & body_frame(observation, position.world)[2]
    columns = (seen.seen, jnp.where(seen.seen, seen.distance, 0.0), seen.seen & facing, in_reach,
               jnp.any(factors != neutral, axis=1), used)
    return jnp.stack([jnp.asarray(column, jnp.float32) for column in columns], axis=1)

assert HEAD["locomotion_gait_compass"] == LOCOMOTION_HEAD and HEAD["body_yaw_delta_bins"] == BODY_YAW_HEAD

__all__ = ["AIM_HEIGHT_BLOCKS", "AIM_TOLERANCE_DEGREES", "ARROW_LAUNCH_HEIGHTS", "ASSUMED_LAUNCH_HEIGHT",
           "AVOID_BAND_BLOCKS", "BODIES", "FACING_ME_DEGREES", "HIDE_ARRIVED_BLOCKS", "HIDE_SNEAK_BLOCKS",
           "LEAD_LAG_TICKS", "MELEE_REACH_BLOCKS", "POSITION_BODIES", "REPORTS", "SHORTBOW_THRESHOLD_SECONDS",
           "ArrowAim", "BowTiers", "Position", "Target", "arrow_pitch", "avoid", "body_frame", "bow_tiers", "draw_tier",
           "evader", "guard", "hide", "idle", "keep_range", "legal", "punish", "reach", "reports", "shortbow", "sword",
           "target", "velocity_scales"]
