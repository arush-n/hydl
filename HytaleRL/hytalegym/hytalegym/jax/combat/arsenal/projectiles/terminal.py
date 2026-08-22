"""Typed encode/decode boundary for projectile-terminal deployable payloads."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.schema.contract import (
    AREA_SHAPE_CYLINDER,
    AREA_SHAPE_SPHERE,
    DEPLOYABLE_ATTACK_FLAG_MASK,
    EF_ANGLED_ANGLE_DEGREES,
    EF_ANGLED_DAMAGE,
    EF_ANGLED_DISTANCE_DEGREES,
    EF_ANGLED_FORCE_X,
    EF_ANGLED_FORCE_Y,
    EF_ANGLED_FORCE_Z,
    EF_AREA_DURATION_SECONDS,
    EF_AREA_END_RADIUS,
    EF_AREA_HEIGHT,
    EF_AREA_INTERVAL_SECONDS,
    EF_AREA_RADIUS_CHANGE_SECONDS,
    EF_DAMAGE,
    EF_FALLOFF,
    EF_FORCE_X,
    EF_FORCE_Y,
    EF_FORCE_Z,
    EF_PROJECTILE_DIRECT_DAMAGE,
    EF_PROJECTILE_HALF_EXTENT,
    EF_RADIUS,
    EF_STATUS_HEALING,
    EI_BLOCK_DAMAGE_RADIUS,
    EI_FORCE_DIRECTION_MODE,
    EI_FORCE_MODE,
    EI_PROJECTILE_DIRECT_DAMAGE_CAUSE,
    EI_RESISTANCE_STYLE,
    EI_TARGET_MODE,
    EVENT_FLAG_PROJECTILE_TERMINAL_DEPLOYABLE_AREA,
)


class TerminalDeployableEncoding(NamedTuple):
    """Sparse event-channel values plus the reserved typed flag."""

    f32: dict[int, float]
    i32: dict[int, int]
    flags: int


class TerminalDeployablePayload(NamedTuple):
    """Decoded terminal payload for arbitrary leading array axes."""

    requested: jax.Array
    collision_half_extent: jax.Array
    collision_center_offset: jax.Array
    deployable_collision_half_extent: jax.Array
    deployable_collision_center_offset: jax.Array
    deployable_id: jax.Array
    deployable_count_towards_global_limit: jax.Array
    deployable_max_live_count: jax.Array
    area_duration_seconds: jax.Array
    area_interval_seconds: jax.Array
    area_start_radius: jax.Array
    area_end_radius: jax.Array
    area_height: jax.Array
    area_radius_change_seconds: jax.Array
    area_damage: jax.Array
    area_shape: jax.Array
    area_attack_flags: jax.Array
    sticks_vertically: jax.Array
    status_healing_per_cycle: jax.Array


def encode_terminal_deployable_payload(
    *,
    collision_min: tuple[float, float, float],
    collision_max: tuple[float, float, float],
    deployable_collision_min: tuple[float, float, float],
    deployable_collision_max: tuple[float, float, float],
    deployable_id: int,
    deployable_count_towards_global_limit: bool,
    deployable_max_live_count: int,
    area_duration_seconds: float,
    area_interval_seconds: float,
    area_start_radius: float,
    area_end_radius: float,
    area_height: float,
    area_radius_change_seconds: float,
    area_damage: float,
    area_shape: int,
    area_attack_flags: int,
    sticks_vertically: bool,
    status_healing_per_cycle: float,
) -> TerminalDeployableEncoding:
    """Encode conditional aliases after strict host-side validation."""

    minimum = _vector(collision_min, "collision_min")
    maximum = _vector(collision_max, "collision_max")
    if any(high <= low for low, high in zip(minimum, maximum, strict=True)):
        raise ValueError("terminal projectile collision bounds must be positive")
    half_extent = tuple(
        (high - low) * 0.5 for low, high in zip(minimum, maximum, strict=True)
    )
    center = tuple(
        (low + high) * 0.5 for low, high in zip(minimum, maximum, strict=True)
    )
    deployable_minimum = _vector(
        deployable_collision_min,
        "deployable_collision_min",
    )
    deployable_maximum = _vector(
        deployable_collision_max,
        "deployable_collision_max",
    )
    if any(
        high <= low
        for low, high in zip(
            deployable_minimum,
            deployable_maximum,
            strict=True,
        )
    ):
        raise ValueError("deployable collision bounds must be positive")
    deployable_half_extent = tuple(
        (high - low) * 0.5
        for low, high in zip(
            deployable_minimum,
            deployable_maximum,
            strict=True,
        )
    )
    deployable_center = tuple(
        (low + high) * 0.5
        for low, high in zip(
            deployable_minimum,
            deployable_maximum,
            strict=True,
        )
    )
    if (
        isinstance(deployable_id, bool)
        or not isinstance(deployable_id, int)
        or deployable_id <= 0
        or deployable_id > 2_147_483_647
    ):
        raise ValueError("deployable_id must be a positive integer")
    if not isinstance(deployable_count_towards_global_limit, bool):
        raise ValueError("deployable_count_towards_global_limit must be a bool")
    if (
        isinstance(deployable_max_live_count, bool)
        or not isinstance(deployable_max_live_count, int)
        or deployable_max_live_count <= 0
        or deployable_max_live_count > 2_147_483_647
    ):
        raise ValueError("deployable_max_live_count must be positive")
    if (
        isinstance(area_attack_flags, bool)
        or not isinstance(area_attack_flags, int)
        or area_attack_flags < 0
        or area_attack_flags & ~DEPLOYABLE_ATTACK_FLAG_MASK
    ):
        raise ValueError("terminal deployable attack flags have unknown bits")
    if area_shape not in (AREA_SHAPE_SPHERE, AREA_SHAPE_CYLINDER):
        raise ValueError("terminal deployable area shape is unknown")
    if not isinstance(sticks_vertically, bool):
        raise ValueError("sticks_vertically must be a bool")
    duration = _finite_positive(area_duration_seconds, "area_duration_seconds")
    interval = _finite_positive(area_interval_seconds, "area_interval_seconds")
    start_radius = _finite_nonnegative(area_start_radius, "area_start_radius")
    end_radius = _finite_nonnegative(area_end_radius, "area_end_radius")
    height = _finite_positive(area_height, "area_height")
    radius_change = _finite_positive(
        area_radius_change_seconds,
        "area_radius_change_seconds",
    )
    damage = _finite_nonnegative(area_damage, "area_damage")
    healing = _finite_nonnegative(
        status_healing_per_cycle,
        "status_healing_per_cycle",
    )
    return TerminalDeployableEncoding(
        f32={
            # Typed aliases. Outside bit31, these retain their ordinary
            # explosion/direct-damage/local-force meanings unchanged.
            EF_PROJECTILE_HALF_EXTENT: half_extent[0],
            EF_FALLOFF: half_extent[1],
            EF_PROJECTILE_DIRECT_DAMAGE: half_extent[2],
            EF_FORCE_X: center[0],
            EF_FORCE_Y: center[1],
            EF_FORCE_Z: center[2],
            EF_AREA_DURATION_SECONDS: duration,
            EF_AREA_INTERVAL_SECONDS: interval,
            EF_RADIUS: start_radius,
            EF_AREA_END_RADIUS: end_radius,
            EF_AREA_HEIGHT: height,
            EF_AREA_RADIUS_CHANGE_SECONDS: radius_change,
            EF_DAMAGE: damage,
            EF_STATUS_HEALING: healing,
            # The angled flag is forbidden for bit31 rows; these six channels
            # therefore carry the spawned deployable entity's AABB only.
            EF_ANGLED_DAMAGE: deployable_half_extent[0],
            EF_ANGLED_ANGLE_DEGREES: deployable_half_extent[1],
            EF_ANGLED_DISTANCE_DEGREES: deployable_half_extent[2],
            EF_ANGLED_FORCE_X: deployable_center[0],
            EF_ANGLED_FORCE_Y: deployable_center[1],
            EF_ANGLED_FORCE_Z: deployable_center[2],
        },
        i32={
            # These channels retain force-mode/block-damage meanings for every
            # row without bit31. The typed terminal owns both aliases here.
            EI_FORCE_MODE: area_shape,
            EI_BLOCK_DAMAGE_RADIUS: int(sticks_vertically),
            EI_TARGET_MODE: area_attack_flags,
            EI_PROJECTILE_DIRECT_DAMAGE_CAUSE: deployable_id,
            EI_RESISTANCE_STYLE: int(deployable_count_towards_global_limit),
            EI_FORCE_DIRECTION_MODE: deployable_max_live_count,
        },
        flags=EVENT_FLAG_PROJECTILE_TERMINAL_DEPLOYABLE_AREA,
    )


def decode_terminal_deployable_payload(
    event_f32: jax.Array,
    event_i32: jax.Array,
    event_flags: jax.Array,
) -> TerminalDeployablePayload:
    """Decode the conditional aliases without inspecting any asset identity."""

    f32 = jnp.asarray(event_f32, dtype=jnp.float32)
    i32 = jnp.asarray(event_i32, dtype=jnp.int32)
    flags = jnp.asarray(event_flags, dtype=jnp.uint32)
    if f32.ndim < 1 or i32.ndim < 1:
        raise ValueError("event payloads must have a channel axis")
    if f32.shape[:-1] != i32.shape[:-1] or flags.shape != f32.shape[:-1]:
        raise ValueError("terminal event payload leading axes must match")
    return TerminalDeployablePayload(
        requested=(
            (flags & jnp.uint32(EVENT_FLAG_PROJECTILE_TERMINAL_DEPLOYABLE_AREA)) != 0
        ),
        collision_half_extent=jnp.stack(
            (
                f32[..., EF_PROJECTILE_HALF_EXTENT],
                f32[..., EF_FALLOFF],
                f32[..., EF_PROJECTILE_DIRECT_DAMAGE],
            ),
            axis=-1,
        ),
        collision_center_offset=jnp.stack(
            (
                f32[..., EF_FORCE_X],
                f32[..., EF_FORCE_Y],
                f32[..., EF_FORCE_Z],
            ),
            axis=-1,
        ),
        deployable_collision_half_extent=jnp.stack(
            (
                f32[..., EF_ANGLED_DAMAGE],
                f32[..., EF_ANGLED_ANGLE_DEGREES],
                f32[..., EF_ANGLED_DISTANCE_DEGREES],
            ),
            axis=-1,
        ),
        deployable_collision_center_offset=jnp.stack(
            (
                f32[..., EF_ANGLED_FORCE_X],
                f32[..., EF_ANGLED_FORCE_Y],
                f32[..., EF_ANGLED_FORCE_Z],
            ),
            axis=-1,
        ),
        deployable_id=i32[..., EI_PROJECTILE_DIRECT_DAMAGE_CAUSE],
        deployable_count_towards_global_limit=(
            i32[..., EI_RESISTANCE_STYLE] == jnp.int32(1)
        ),
        deployable_max_live_count=i32[..., EI_FORCE_DIRECTION_MODE],
        area_duration_seconds=f32[..., EF_AREA_DURATION_SECONDS],
        area_interval_seconds=f32[..., EF_AREA_INTERVAL_SECONDS],
        area_start_radius=f32[..., EF_RADIUS],
        area_end_radius=f32[..., EF_AREA_END_RADIUS],
        area_height=f32[..., EF_AREA_HEIGHT],
        area_radius_change_seconds=f32[..., EF_AREA_RADIUS_CHANGE_SECONDS],
        area_damage=f32[..., EF_DAMAGE],
        area_shape=i32[..., EI_FORCE_MODE],
        area_attack_flags=i32[..., EI_TARGET_MODE],
        sticks_vertically=i32[..., EI_BLOCK_DAMAGE_RADIUS] == jnp.int32(1),
        status_healing_per_cycle=f32[..., EF_STATUS_HEALING],
    )


def _vector(value, label: str) -> tuple[float, float, float]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{label} must be a three-float tuple")
    return tuple(_finite(number, label) for number in value)


def _finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _finite_positive(value, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _finite_nonnegative(value, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


__all__ = [
    "TerminalDeployableEncoding",
    "TerminalDeployablePayload",
    "decode_terminal_deployable_payload",
    "encode_terminal_deployable_payload",
]
