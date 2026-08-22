"""Source-backed neutral locomotion primitives."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp


Array = jax.Array
WALK_STEERING_SCHEMA = "hytalerl_walk_steering_v1"
WALK_STEERING_VERSION = 1
WALK_DEFAULT_MAX_ROTATION_SPEED_RADIANS = 2.0 * jnp.pi
WALK_DEFAULT_EPSILON_ANGLE_RADIANS = jnp.pi / 60.0
WALK_DEFAULT_MAX_MOVE_TURN_ANGLE_RADIANS = jnp.pi / 2.0
_PI = jnp.float32(jnp.pi)
_TAU = jnp.float32(2.0 * jnp.pi)


class WalkSteeringResult(NamedTuple):
    """One ``MotionControllerWalk.computeHeading`` update."""

    available: Array
    heading: Array
    applied_turn_angle: Array
    fully_rotated: Array
    translation_allowed: Array
    invalid: Array


def walk_steering_result(
    heading: Array,
    desired_heading: Array,
    delta_seconds: Array | float,
    *,
    maximum_rotation_speed: Array | float = (WALK_DEFAULT_MAX_ROTATION_SPEED_RADIANS),
    relative_turn_speed: Array | float = 1.0,
    horizontal_speed_multiplier: Array | float = 1.0,
    epsilon_angle: Array | float = WALK_DEFAULT_EPSILON_ANGLE_RADIANS,
    maximum_move_turn_angle: Array | float = (WALK_DEFAULT_MAX_MOVE_TURN_ANGLE_RADIANS),
    stop_if_turned_too_far: bool = True,
) -> WalkSteeringResult:
    """Apply Hytale's shortest-turn clamp and movement-stop predicate.

    Angles are radians. Configuration values remain inputs because NPC roles
    may override the builder defaults. Invalid rows return zero and deny
    translation.
    """

    if not isinstance(stop_if_turned_too_far, bool):
        raise TypeError("stop_if_turned_too_far must be bool")
    current = jnp.asarray(heading, dtype=jnp.float32)
    desired = jnp.asarray(desired_heading, dtype=jnp.float32)
    if current.ndim != 1 or current.shape[0] == 0:
        raise ValueError("heading must have non-empty shape [batch]")
    if desired.shape != current.shape:
        raise ValueError("desired_heading must match heading")
    batch = current.shape[0]
    delta = _batch_float(delta_seconds, batch, "delta_seconds")
    maximum_speed = _batch_float(
        maximum_rotation_speed,
        batch,
        "maximum_rotation_speed",
    )
    relative_speed = _batch_float(
        relative_turn_speed,
        batch,
        "relative_turn_speed",
    )
    effect_speed = _batch_float(
        horizontal_speed_multiplier,
        batch,
        "horizontal_speed_multiplier",
    )
    epsilon = _batch_float(epsilon_angle, batch, "epsilon_angle")
    maximum_move_turn = _batch_float(
        maximum_move_turn_angle,
        batch,
        "maximum_move_turn_angle",
    )
    valid = (
        jnp.isfinite(current)
        & jnp.isfinite(desired)
        & jnp.isfinite(delta)
        & (delta > 0.0)
        & jnp.isfinite(maximum_speed)
        & (maximum_speed > 0.0)
        & jnp.isfinite(relative_speed)
        & (relative_speed >= 0.0)
        & jnp.isfinite(effect_speed)
        & (effect_speed >= 0.0)
        & jnp.isfinite(epsilon)
        & (epsilon > 0.0)
        & (epsilon <= _PI)
        & jnp.isfinite(maximum_move_turn)
        & (maximum_move_turn >= 0.0)
        & (maximum_move_turn <= _PI)
    )
    safe_current = jnp.where(valid, current, jnp.float32(0.0))
    safe_desired = jnp.where(valid, desired, jnp.float32(0.0))
    turn = _turn_angle(safe_current, safe_desired)
    fully_rotated = valid & (jnp.abs(turn) <= epsilon)
    maximum_turn = delta * maximum_speed * effect_speed * relative_speed
    applied = jnp.where(
        fully_rotated,
        turn,
        jnp.clip(turn, -maximum_turn, maximum_turn),
    )
    updated = jnp.where(
        fully_rotated,
        safe_desired,
        _normalize_turn_angle(safe_current + applied),
    )
    translation_allowed = valid
    if stop_if_turned_too_far:
        translation_allowed &= jnp.abs(applied) <= maximum_move_turn
    return WalkSteeringResult(
        available=valid,
        heading=jnp.where(valid, updated, jnp.float32(0.0)),
        applied_turn_angle=jnp.where(valid, applied, jnp.float32(0.0)),
        fully_rotated=fully_rotated,
        translation_allowed=translation_allowed,
        invalid=~valid,
    )


def walk_steering_contract() -> dict[str, object]:
    """Return the pinnable source and unsupported-mechanics boundary."""

    return {
        "schema": WALK_STEERING_SCHEMA,
        "version": WALK_STEERING_VERSION,
        "function": "walk_steering_result",
        "source": (
            "MotionControllerWalk.computeHeading"
            "->NPCPhysicsMath.turnAngle"
            "->PhysicsMath.normalizeTurnAngle"
        ),
        "units": "radians_and_seconds",
        "configuration": {
            "maximum_rotation_speed": (
                "role_value_or_BuilderMotionControllerWalk_default"
            ),
            "relative_turn_speed": "live_Steering_value",
            "horizontal_speed_multiplier": (
                "MotionControllerWalk.effectHorizontalSpeedMultiplier"
            ),
            "epsilon_angle": ("role_value_or_BuilderMotionControllerBase_default"),
            "maximum_move_turn_angle": (
                "role_value_or_BuilderMotionControllerWalk_default"
            ),
        },
        "builder_defaults": {
            "maximum_rotation_speed_radians_per_second": float(
                WALK_DEFAULT_MAX_ROTATION_SPEED_RADIANS
            ),
            "epsilon_angle_radians": float(WALK_DEFAULT_EPSILON_ANGLE_RADIANS),
            "maximum_move_turn_angle_radians": float(
                WALK_DEFAULT_MAX_MOVE_TURN_ANGLE_RADIANS
            ),
        },
        "turn_order": [
            "shortest_signed_angle",
            "epsilon_snap_or_dt_scaled_clamp",
            "normalize_non_snap_result_to_half_open_minus_pi_pi",
            "compare_applied_turn_to_move_turn_limit",
        ],
        "fail_closed": ("non_finite_or_out_of_domain_rows_deny_translation"),
        "sprint_boundary": {
            "npc_walk_controller": "always_publishes_sprinting_false",
            "player_runtime_config": (
                "GameplayConfig.Player.MovementConfig_asset_selected"
            ),
            "installed_default_forward_multiplier": 1.273,
            "master_fallback_forward_multiplier": 1.65,
            "transition_physics": "client_side_not_native_certified",
            "jax_publication": None,
        },
        "action_surface": "owned_by_combat_lane",
        "provenance": "native_source_port_jax_float32",
    }


def walk_steering_contract_sha256() -> str:
    payload = json.dumps(
        walk_steering_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _normalize_angle(value: Array) -> Array:
    normalized = jnp.fmod(value, _TAU)
    return jnp.where(normalized < 0.0, normalized + _TAU, normalized)


def _normalize_turn_angle(value: Array) -> Array:
    normalized = _normalize_angle(value)
    return jnp.where(normalized >= _PI, normalized - _TAU, normalized)


def _turn_angle(current: Array, desired: Array) -> Array:
    delta = _normalize_angle(desired) - _normalize_angle(current)
    delta = jnp.where(delta < -_PI, delta + _TAU, delta)
    return jnp.where(delta > _PI, delta - _TAU, delta)


def _batch_float(
    value: Array | float,
    batch: int,
    label: str,
) -> Array:
    result = jnp.asarray(value, dtype=jnp.float32)
    if result.ndim == 0:
        result = jnp.broadcast_to(result, (batch,))
    if result.shape != (batch,):
        raise ValueError(f"{label} must be scalar or shape [batch]")
    return result


__all__ = [
    "WALK_DEFAULT_EPSILON_ANGLE_RADIANS",
    "WALK_DEFAULT_MAX_MOVE_TURN_ANGLE_RADIANS",
    "WALK_DEFAULT_MAX_ROTATION_SPEED_RADIANS",
    "WALK_STEERING_SCHEMA",
    "WALK_STEERING_VERSION",
    "WalkSteeringResult",
    "walk_steering_contract",
    "walk_steering_contract_sha256",
    "walk_steering_result",
]
