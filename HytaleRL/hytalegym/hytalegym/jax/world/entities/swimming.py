"""Fixed-shape, source-backed NPC Dive-controller primitives."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.geometry import movement_medium_result
from hytalegym.jax.world.entities.locomotion import (
    _batch_float,
    _normalize_turn_angle,
    _turn_angle,
)
from hytalegym.jax.world.region.geometry import region_movement_medium_result
from hytalegym.jax.world.region.types import RegionGeometryState
from hytalegym.jax.world.types import GeometryState


Array = jax.Array
GeometryProvider = GeometryState | RegionGeometryState

DIVE_MOTION_SCHEMA = "hytalerl_dive_motion_v2"
DIVE_MOTION_VERSION = 2

# Native MotionKind ordinals. Unknown is a JAX-only fail-closed sentinel.
DIVE_MOTION_KIND_UNKNOWN = -1
DIVE_MOTION_KIND_MOVING = 4
DIVE_MOTION_KIND_SWIMMING = 6
DIVE_MOTION_KIND_SWIMMING_TURNING = 7

_DEFAULT_MAX_HORIZONTAL_SPEED = 3.0
_DEFAULT_MAX_VERTICAL_SPEED = 8.0
_DEFAULT_ACCELERATION = 3.0
_DEFAULT_MAX_ROTATION_SPEED = 2.0 * jnp.pi
_DEFAULT_MAX_MOVE_TURN_ANGLE = jnp.pi / 2.0
_DEFAULT_EPSILON_ANGLE = jnp.pi / 60.0
_DEFAULT_EPSILON_SPEED = 1.0e-5
_DEFAULT_SWIM_DEPTH_FALLBACK = 0.5
_DEFAULT_MAX_DIVE_DEPTH_FLOAT32 = jnp.finfo(jnp.float32).max
_WATER_PROBE_EPSILON = jnp.float32(1.0e-6)
_MOVING_SPEED_SQUARED = jnp.float32(1.0e-12)
_HEADING_SPEED_SQUARED = jnp.float32(1.0000000000000002e-10)
_PI = jnp.float32(jnp.pi)
_RIVEN_TABLE_SIZE = 1 << 12
_RIVEN_TABLE_MASK = _RIVEN_TABLE_SIZE - 1
_RIVEN_RAD_FULL = np.float32(6.2831855)
_RIVEN_RAD_TO_INDEX = np.float32(_RIVEN_TABLE_SIZE) / _RIVEN_RAD_FULL
_ICECORE_TABLE_SCALE = 100_000


def _build_riven_tables() -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(_RIVEN_TABLE_SIZE, dtype=np.float32)
    angles = (
        (indices + np.float32(0.5))
        / np.float32(_RIVEN_TABLE_SIZE)
        * _RIVEN_RAD_FULL
    ).astype(np.float32)
    sine = np.sin(angles.astype(np.float64)).astype(np.float32)
    cosine = np.cos(angles.astype(np.float64)).astype(np.float32)
    degree_to_index = (
        np.float32(_RIVEN_TABLE_SIZE) / np.float32(360.0)
    )
    for degree in range(0, 360, 90):
        index = int(np.float32(degree) * degree_to_index) & _RIVEN_TABLE_MASK
        radians = float(degree) * np.pi / 180.0
        sine[index] = np.float32(np.sin(radians))
        cosine[index] = np.float32(np.cos(radians))
    return sine, cosine


_RIVEN_SINE, _RIVEN_COSINE = _build_riven_tables()
_ICECORE_ATAN2 = np.arctan2(
    np.arange(_ICECORE_TABLE_SCALE + 1, dtype=np.float64)
    / float(_ICECORE_TABLE_SCALE),
    1.0,
).astype(np.float32)


class DiveMotionResult(NamedTuple):
    """One active ``MotionControllerDive.computeMove`` projection."""

    available: Array
    in_water: Array
    can_steer: Array
    motion_kind: Array
    heading: Array
    pitch: Array
    move_speed: Array
    climb_speed: Array
    translation: Array
    passive_branch_required: Array
    geometry_exhausted: Array
    invalid: Array


class DiveVerticalRangeResult(NamedTuple):
    """Native desired-depth interval from a completed water probe."""

    available: Array
    current_y: Array
    minimum_y: Array
    maximum_y: Array
    collapsed_to_current: Array
    invalid: Array


def dive_motion_result(
    geometry: GeometryProvider,
    position: Array,
    heading: Array,
    pitch: Array,
    steering_translation: Array,
    previous_move_speed: Array,
    previous_climb_speed: Array,
    delta_seconds: Array | float,
    on_ground: Array | bool,
    swim_depth: Array | float,
    *,
    base_can_steer: Array | bool = True,
    collision_with_solid: Array | bool = False,
    has_translation: Array | bool = True,
    steering_heading: Array | float = 0.0,
    steering_pitch: Array | float = 0.0,
    has_steering_heading: Array | bool = False,
    has_steering_pitch: Array | bool = False,
    effect_horizontal_speed_multiplier: Array | float = 1.0,
    maximum_horizontal_speed: Array | float = _DEFAULT_MAX_HORIZONTAL_SPEED,
    maximum_vertical_speed: Array | float = _DEFAULT_MAX_VERTICAL_SPEED,
    acceleration: Array | float = _DEFAULT_ACCELERATION,
    maximum_rotation_speed: Array | float = _DEFAULT_MAX_ROTATION_SPEED,
    maximum_move_turn_angle: Array | float = _DEFAULT_MAX_MOVE_TURN_ANGLE,
    epsilon_angle: Array | float = _DEFAULT_EPSILON_ANGLE,
    epsilon_speed: Array | float = _DEFAULT_EPSILON_SPEED,
) -> DiveMotionResult:
    """Port the Dive controller's active in-water motion branch.

    ``swim_depth`` is the model-derived absolute height, not the authored
    relative ``SwimDepth`` value. The native water probe samples
    ``position.y + swim_depth + 1e-6`` and compares against the integer
    fluid-block top; fluid fill height is intentionally not substituted.

    Rows that need the controller's passive gravity, death, applied-force,
    or collision branch return zero translation and set
    ``passive_branch_required``. Missing geometry and invalid values also
    fail closed.
    """

    if not isinstance(geometry, (GeometryState, RegionGeometryState)):
        raise TypeError("geometry must be GeometryState or RegionGeometryState")
    points = jnp.asarray(position, dtype=jnp.float32)
    steering = jnp.asarray(steering_translation, dtype=jnp.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("position must have non-empty shape [batch, 3]")
    if steering.shape != points.shape:
        raise ValueError("steering_translation must match position")
    batch = points.shape[0]

    current_heading = _batch_float(heading, batch, "heading")
    current_pitch = _batch_float(pitch, batch, "pitch")
    prior_move = _batch_float(
        previous_move_speed,
        batch,
        "previous_move_speed",
    )
    prior_climb = _batch_float(
        previous_climb_speed,
        batch,
        "previous_climb_speed",
    )
    delta = _batch_float(delta_seconds, batch, "delta_seconds")
    depth = _batch_float(swim_depth, batch, "swim_depth")
    desired_heading_input = _batch_float(
        steering_heading,
        batch,
        "steering_heading",
    )
    desired_pitch_input = _batch_float(
        steering_pitch,
        batch,
        "steering_pitch",
    )
    effect = _batch_float(
        effect_horizontal_speed_multiplier,
        batch,
        "effect_horizontal_speed_multiplier",
    )
    max_horizontal = _batch_float(
        maximum_horizontal_speed,
        batch,
        "maximum_horizontal_speed",
    )
    max_vertical = _batch_float(
        maximum_vertical_speed,
        batch,
        "maximum_vertical_speed",
    )
    accel = _batch_float(acceleration, batch, "acceleration")
    max_rotation = _batch_float(
        maximum_rotation_speed,
        batch,
        "maximum_rotation_speed",
    )
    max_move_turn = _batch_float(
        maximum_move_turn_angle,
        batch,
        "maximum_move_turn_angle",
    )
    angle_epsilon = _batch_float(epsilon_angle, batch, "epsilon_angle")
    speed_epsilon = _batch_float(epsilon_speed, batch, "epsilon_speed")
    grounded = _batch_bool(on_ground, batch, "on_ground")
    base_steer = _batch_bool(base_can_steer, batch, "base_can_steer")
    collided = _batch_bool(
        collision_with_solid,
        batch,
        "collision_with_solid",
    )
    translation_present = _batch_bool(
        has_translation,
        batch,
        "has_translation",
    )
    heading_present = _batch_bool(
        has_steering_heading,
        batch,
        "has_steering_heading",
    )
    pitch_present = _batch_bool(
        has_steering_pitch,
        batch,
        "has_steering_pitch",
    )

    point_finite = jnp.all(jnp.isfinite(points), axis=1)
    depth_valid = jnp.isfinite(depth) & (depth >= 0.0)
    safe_points = jnp.where(point_finite[:, None], points, 0.0)
    safe_depth = jnp.where(depth_valid, depth, 0.0)
    probe_position = safe_points.at[:, 1].add(
        safe_depth + _WATER_PROBE_EPSILON
    )
    medium = _movement_medium(geometry, probe_position)

    valid = (
        point_finite
        & depth_valid
        & jnp.isfinite(current_heading)
        & jnp.isfinite(current_pitch)
        & jnp.isfinite(prior_move)
        & jnp.isfinite(prior_climb)
        & jnp.all(jnp.isfinite(steering), axis=1)
        & jnp.isfinite(desired_heading_input)
        & jnp.isfinite(desired_pitch_input)
        & jnp.isfinite(delta)
        & (delta > 0.0)
        & jnp.isfinite(effect)
        & (effect >= 0.0)
        & jnp.isfinite(max_horizontal)
        & (max_horizontal > 0.0)
        & jnp.isfinite(max_vertical)
        & (max_vertical > 0.0)
        & jnp.isfinite(accel)
        & (accel > 0.0)
        & jnp.isfinite(max_rotation)
        & (max_rotation > 0.0)
        & jnp.isfinite(max_move_turn)
        & (max_move_turn >= 0.0)
        & (max_move_turn <= _PI)
        & jnp.isfinite(angle_epsilon)
        & (angle_epsilon > 0.0)
        & (angle_epsilon <= _PI)
        & jnp.isfinite(speed_epsilon)
        & (speed_epsilon > 0.0)
    )
    exhausted = medium.geometry_exhausted
    available = valid & ~exhausted
    in_water = available & medium.in_fluid
    can_steer = available & base_steer & in_water

    safe_heading = jnp.where(valid, current_heading, 0.0)
    safe_pitch = jnp.where(valid, current_pitch, 0.0)
    safe_prior_move = jnp.where(valid & ~collided, prior_move, 0.0)
    safe_prior_climb = jnp.where(valid & ~collided, prior_climb, 0.0)
    safe_steering = jnp.where(valid[:, None], steering, 0.0)
    safe_effect = jnp.where(valid, effect, 1.0)
    effective_horizontal = (
        jnp.where(valid, max_horizontal, _DEFAULT_MAX_HORIZONTAL_SPEED)
        * safe_effect
    )
    effective_vertical = (
        jnp.where(valid, max_vertical, _DEFAULT_MAX_VERTICAL_SPEED)
        * safe_effect
    )
    safe_delta = jnp.where(valid, delta, 1.0)
    safe_accel = jnp.where(valid, accel, _DEFAULT_ACCELERATION)

    target_move = (
        jnp.linalg.norm(safe_steering[:, (0, 2)], axis=1)
        * effective_horizontal
    )
    target_climb = safe_steering[:, 1] * effective_vertical
    next_move = _accelerate_to_target_speed(
        safe_prior_move,
        target_move,
        safe_delta,
        safe_accel,
        safe_accel,
        jnp.zeros(batch, dtype=jnp.float32),
        effective_horizontal,
    )
    next_climb = _accelerate_to_target_speed(
        safe_prior_climb,
        target_climb,
        safe_delta,
        safe_accel,
        safe_accel,
        -effective_vertical,
        effective_vertical,
    )
    next_move = jnp.where(can_steer, next_move, 0.0)
    next_climb = jnp.where(can_steer, next_climb, 0.0)

    moving = (
        can_steer
        & translation_present
        & (
            next_move * next_move + next_climb * next_climb
            > _MOVING_SPEED_SQUARED
        )
    )
    direction_heading = _normalize_turn_angle(
        _native_icecore_atan2(
            -safe_steering[:, 0],
            -safe_steering[:, 2],
        )
    )
    direction_pitch = _normalize_turn_angle(
        _native_icecore_atan2(
            safe_steering[:, 1],
            jnp.sqrt(
                safe_steering[:, 0] * safe_steering[:, 0]
                + safe_steering[:, 2] * safe_steering[:, 2]
            ),
        )
    )
    derive_direction = (
        can_steer
        & translation_present
        & (next_move * next_move > _HEADING_SPEED_SQUARED)
    )
    desired_heading = jnp.where(
        derive_direction,
        direction_heading,
        jnp.where(
            heading_present,
            desired_heading_input,
            safe_heading,
        ),
    )
    desired_pitch = jnp.where(
        derive_direction,
        direction_pitch,
        jnp.where(
            pitch_present,
            desired_pitch_input,
            safe_pitch,
        ),
    )
    heading_turn = _turn_angle(safe_heading, desired_heading)
    pitch_turn = _turn_angle(safe_pitch, desired_pitch)
    heading_near = jnp.abs(heading_turn) <= angle_epsilon
    pitch_near = jnp.abs(pitch_turn) <= angle_epsilon
    maximum_turn = (
        safe_delta
        * jnp.where(valid, max_rotation, _DEFAULT_MAX_ROTATION_SPEED)
        * safe_effect
    )
    updated_heading = _normalize_turn_angle(
        jnp.where(
            heading_near,
            desired_heading,
            safe_heading
            + jnp.clip(heading_turn, -maximum_turn, maximum_turn),
        )
    )
    updated_pitch = _normalize_turn_angle(
        jnp.where(
            pitch_near,
            desired_pitch,
            safe_pitch + jnp.clip(pitch_turn, -maximum_turn, maximum_turn),
        )
    )
    stopped_move = jnp.where(
        jnp.abs(heading_turn) > max_move_turn,
        0.0,
        next_move,
    )
    motion_kind = jnp.where(
        in_water | ~grounded,
        DIVE_MOTION_KIND_SWIMMING,
        DIVE_MOTION_KIND_MOVING,
    )
    motion_kind = jnp.where(
        can_steer & ~moving & ~heading_near,
        DIVE_MOTION_KIND_SWIMMING_TURNING,
        motion_kind,
    )
    motion_kind = jnp.where(
        available,
        motion_kind,
        DIVE_MOTION_KIND_UNKNOWN,
    ).astype(jnp.int8)

    raw_translation = jnp.stack(
        (
            stopped_move
            * safe_delta
            * -_native_riven_lookup(_RIVEN_SINE, updated_heading),
            next_climb * safe_delta,
            stopped_move
            * safe_delta
            * -_native_riven_lookup(_RIVEN_COSINE, updated_heading),
        ),
        axis=1,
    )
    clipped_translation = jnp.where(
        jnp.abs(raw_translation) < speed_epsilon[:, None],
        0.0,
        raw_translation,
    )
    return DiveMotionResult(
        available=available,
        in_water=in_water,
        can_steer=can_steer,
        motion_kind=motion_kind,
        heading=jnp.where(
            available,
            jnp.where(can_steer, updated_heading, safe_heading),
            0.0,
        ),
        pitch=jnp.where(
            available,
            jnp.where(can_steer, updated_pitch, safe_pitch),
            0.0,
        ),
        move_speed=jnp.where(can_steer, stopped_move, 0.0),
        climb_speed=jnp.where(can_steer, next_climb, 0.0),
        translation=jnp.where(
            can_steer[:, None],
            clipped_translation,
            0.0,
        ),
        passive_branch_required=available & ~can_steer,
        geometry_exhausted=exhausted,
        invalid=~valid,
    )


def dive_vertical_range_result(
    current_y: Array,
    water_block_y: Array,
    ground_block_y: Array,
    swim_depth: Array,
    on_ground: Array | bool,
    touches_ceiling: Array | bool,
    *,
    probe_available: Array | bool = True,
    maximum_dive_depth: Array | float = _DEFAULT_MAX_DIVE_DEPTH_FLOAT32,
    minimum_depth_above_ground: Array | float = 1.0,
    minimum_depth_below_surface: Array | float = 1.0,
) -> DiveVerticalRangeResult:
    """Port ``MotionControllerDive.getDesiredVerticalRange`` exactly."""

    current = jnp.asarray(current_y, dtype=jnp.float32)
    if current.ndim != 1 or current.shape[0] == 0:
        raise ValueError("current_y must have non-empty shape [batch]")
    batch = current.shape[0]
    water = _batch_float(water_block_y, batch, "water_block_y")
    ground = _batch_float(ground_block_y, batch, "ground_block_y")
    depth = _batch_float(swim_depth, batch, "swim_depth")
    maximum_depth = _batch_float(
        maximum_dive_depth,
        batch,
        "maximum_dive_depth",
    )
    above_ground = _batch_float(
        minimum_depth_above_ground,
        batch,
        "minimum_depth_above_ground",
    )
    below_surface = _batch_float(
        minimum_depth_below_surface,
        batch,
        "minimum_depth_below_surface",
    )
    grounded = _batch_bool(on_ground, batch, "on_ground")
    ceiling = _batch_bool(touches_ceiling, batch, "touches_ceiling")
    probe = _batch_bool(probe_available, batch, "probe_available")

    valid = (
        jnp.isfinite(current)
        & jnp.isfinite(water)
        & (water == jnp.floor(water))
        & jnp.isfinite(ground)
        & (ground == jnp.floor(ground))
        & jnp.isfinite(depth)
        & (depth >= 0.0)
        & jnp.isfinite(maximum_depth)
        & (maximum_depth > 0.0)
        & jnp.isfinite(above_ground)
        & (above_ground >= 0.0)
        & jnp.isfinite(below_surface)
        & (below_surface >= 0.0)
    )
    safe_current = jnp.where(valid, current, 0.0)
    water_surface = jnp.where(valid, water, 0.0) + 1.0
    minimum = jnp.maximum(
        jnp.where(valid, ground + above_ground, 0.0),
        water_surface - jnp.where(valid, maximum_depth, 1.0),
    )
    maximum = (
        water_surface
        - jnp.where(valid, depth, 0.0)
        - jnp.where(valid, below_surface, 0.0)
    )
    minimum = jnp.where(grounded, safe_current, minimum)
    maximum = jnp.where(ceiling, safe_current, maximum)
    available = valid & probe
    collapsed = available & (minimum > maximum)
    minimum = jnp.where(collapsed, safe_current, minimum)
    maximum = jnp.where(collapsed, safe_current, maximum)
    return DiveVerticalRangeResult(
        available=available,
        current_y=jnp.where(available, safe_current, 0.0),
        minimum_y=jnp.where(available, minimum, 0.0),
        maximum_y=jnp.where(available, maximum, 0.0),
        collapsed_to_current=collapsed,
        invalid=~valid,
    )


def dive_motion_contract() -> dict[str, object]:
    """Return the pinnable implementation and unsupported-runtime boundary."""

    return {
        "schema": DIVE_MOTION_SCHEMA,
        "version": DIVE_MOTION_VERSION,
        "controller": "MotionControllerDive",
        "functions": {
            "motion": "dive_motion_result",
            "desired_vertical_range": "dive_vertical_range_result",
        },
        "source": {
            "motion": (
                "MotionControllerDive.computeMove_active_branch"
                "->NPCPhysicsMath.accelerateToTargetSpeed"
                "->PhysicsMath.headingFromDirection/pitchFromDirection"
                "->TrigMathUtil.Icecore"
                "->MotionControllerDive.computeTranslation"
                "->PhysicsMath.headingX/headingZ"
                "->TrigMathUtil.Riven"
            ),
            "water_probe": (
                "PositionProbeWater.probePosition"
                "(position_y+swim_depth+1e-6)"
            ),
            "vertical_range": (
                "MotionControllerDive.getDesiredVerticalRange"
            ),
            "motion_kind_ordinals": {
                "MOVING": DIVE_MOTION_KIND_MOVING,
                "SWIMMING": DIVE_MOTION_KIND_SWIMMING,
                "SWIMMING_TURNING": (
                    DIVE_MOTION_KIND_SWIMMING_TURNING
                ),
                "jax_unavailable": DIVE_MOTION_KIND_UNKNOWN,
            },
        },
        "builder_defaults": {
            "maximum_horizontal_speed": _DEFAULT_MAX_HORIZONTAL_SPEED,
            "maximum_vertical_speed": _DEFAULT_MAX_VERTICAL_SPEED,
            "acceleration": _DEFAULT_ACCELERATION,
            "maximum_rotation_speed_radians_per_second": float(
                _DEFAULT_MAX_ROTATION_SPEED
            ),
            "maximum_move_turn_angle_radians": float(
                _DEFAULT_MAX_MOVE_TURN_ANGLE
            ),
            "epsilon_angle_radians": float(_DEFAULT_EPSILON_ANGLE),
            "epsilon_speed": _DEFAULT_EPSILON_SPEED,
            "relative_swim_depth": 0.4,
            "missing_model_absolute_swim_depth_fallback": (
                _DEFAULT_SWIM_DEPTH_FALLBACK
            ),
            "maximum_dive_depth_native": "Double.MAX_VALUE",
            "maximum_dive_depth_jax_default": (
                "largest_finite_float32"
            ),
        },
        "water_semantics": {
            "provider": (
                "movement_medium_result_or_region_movement_medium_result"
            ),
            "probe_y": "position_y+absolute_swim_depth+1e-6",
            "surface_test": "integer_fluid_block_top_at_or_above_probe_y",
            "fluid_fill_height_used": False,
            "swim_up_down_speed_fields": (
                "FluidFX_plumbing_not_Dive_maximum_vertical_speed"
            ),
        },
        "implemented": [
            "active_in_water_speed_acceleration",
            "translation_derived_heading_and_pitch",
            "resolved_explicit_heading_and_pitch_inputs",
            "turn_clamp_and_horizontal_stop",
            "MOVING_SWIMMING_SWIMMING_TURNING_state",
            "epsilon_translation_clip",
            "Riven_4096_entry_float_heading_lookup",
            "Icecore_100001_entry_float_atan2_lookup",
            "desired_vertical_range_from_completed_probe_levels",
        ],
        "unsupported": [
            "passive_gravity_death_and_applied_velocity_branches",
            "collision_executeMove_bisection_and_no_slide_resolution",
            "full_AABB_PositionProbeWater_overlap_beyond_point_projection",
            "model_asset_to_absolute_swim_depth_adapter",
            "player_client_swim_transition",
        ],
        "fail_closed": {
            "invalid_or_uncovered": "unavailable_and_zero_translation",
            "passive_branch": (
                "zero_translation_with_passive_branch_required"
            ),
        },
        "native_equivalence": {
            "source_pin": "Hytale_0.5.7_class_and_role_assets",
            "runtime_differential": (
                "hytalerl_native_dive_motion_evidence_v1"
            ),
            "fixture": "dive_motion",
            "certified": True,
            "certified_scope": [
                "native_NPC_active_in_water_branch",
                "role_parameterized_horizontal_and_vertical_acceleration",
                "heading_and_pitch_derivation_and_turn_clamp",
                "motion_kind",
                "open_water_translation",
            ],
        },
        "provenance": "native_runtime_certified_active_branch_jax_float32",
        "action_surface": "owned_by_combat_lane_not_published_here",
    }


def dive_motion_contract_sha256() -> str:
    payload = json.dumps(
        dive_motion_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _movement_medium(
    geometry: GeometryProvider,
    position: Array,
):
    if isinstance(geometry, GeometryState):
        return movement_medium_result(geometry, position)
    return region_movement_medium_result(geometry, position)


def _native_riven_lookup(table: np.ndarray, radians: Array) -> Array:
    index = (
        jnp.asarray(radians, dtype=jnp.float32)
        * jnp.float32(_RIVEN_RAD_TO_INDEX)
    ).astype(jnp.int32)
    index = jnp.bitwise_and(index, jnp.int32(_RIVEN_TABLE_MASK))
    return jnp.asarray(table, dtype=jnp.float32)[index]


def _native_icecore_atan2(y: Array, x: Array) -> Array:
    y_value = jnp.asarray(y, dtype=jnp.float32)
    x_value = jnp.asarray(x, dtype=jnp.float32)
    abs_y = jnp.abs(y_value)
    abs_x = jnp.abs(x_value)
    y_dominant = abs_y > abs_x
    maximum = jnp.maximum(abs_y, abs_x)
    # Preserve Java's exact x/x quotient across GPU reciprocal lowering.
    ratio = jnp.where(
        maximum > 0.0,
        jnp.where(
            abs_y == abs_x,
            jnp.float32(1.0),
            jnp.minimum(abs_y, abs_x) / maximum,
        ),
        0.0,
    )
    index = (
        ratio * jnp.float32(_ICECORE_TABLE_SCALE)
    ).astype(jnp.int32)
    base = jnp.asarray(_ICECORE_ATAN2, dtype=jnp.float32)[index]
    angle = jnp.where(y_dominant, jnp.float32(jnp.pi / 2.0) - base, base)
    angle = jnp.where(x_value < 0.0, _PI - angle, angle)
    return jnp.where(y_value < 0.0, -angle, angle)


def _accelerate_to_target_speed(
    current: Array,
    target: Array,
    delta: Array,
    acceleration: Array,
    deceleration: Array,
    minimum_speed: Array,
    maximum_speed: Array,
) -> Array:
    target = jnp.clip(target, minimum_speed, maximum_speed)
    degenerate = maximum_speed <= minimum_speed
    positive_denominator = jnp.where(maximum_speed != 0.0, maximum_speed, 1.0)
    negative_denominator = jnp.where(minimum_speed != 0.0, minimum_speed, -1.0)
    drag = jnp.where(
        current == 0.0,
        0.0,
        jnp.where(
            current > 0.0,
            -acceleration
            * jnp.abs(current / positive_denominator) ** 3,
            jnp.where(
                minimum_speed < 0.0,
                deceleration
                * jnp.abs(current / negative_denominator) ** 3,
                acceleration
                * jnp.abs(current / positive_denominator) ** 3,
            ),
        ),
    )
    accelerated = jnp.minimum(
        current + delta * (drag + acceleration),
        target,
    )
    decelerated = jnp.maximum(
        current + delta * (drag - deceleration),
        target,
    )
    result = jnp.where(current < target, accelerated, decelerated)
    result = jnp.where(current == target, target, result)
    return jnp.where(degenerate, target, result)


def _batch_bool(
    value: Array | bool,
    batch: int,
    label: str,
) -> Array:
    result = jnp.asarray(value)
    if result.dtype != jnp.bool_:
        raise TypeError(f"{label} must be boolean")
    if result.ndim == 0:
        result = jnp.broadcast_to(result, (batch,))
    if result.shape != (batch,):
        raise ValueError(f"{label} must be scalar or shape [batch]")
    return result


__all__ = [
    "DIVE_MOTION_KIND_MOVING",
    "DIVE_MOTION_KIND_SWIMMING",
    "DIVE_MOTION_KIND_SWIMMING_TURNING",
    "DIVE_MOTION_KIND_UNKNOWN",
    "DIVE_MOTION_SCHEMA",
    "DIVE_MOTION_VERSION",
    "DiveMotionResult",
    "DiveVerticalRangeResult",
    "dive_motion_contract",
    "dive_motion_contract_sha256",
    "dive_motion_result",
    "dive_vertical_range_result",
]
