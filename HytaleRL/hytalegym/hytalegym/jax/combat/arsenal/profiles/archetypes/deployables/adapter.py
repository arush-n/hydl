"""JAX event adapter for host-compiled projectile-terminal deployables."""

from __future__ import annotations

from dataclasses import dataclass

from hytalegym.combat.assets.weapon_archetypes.deployables import (
    ProjectileTerminalDeployableSpec,
)
from hytalegym.jax.combat.arsenal.projectiles.terminal import (
    encode_terminal_deployable_payload,
)
from hytalegym.jax.combat.arsenal.schema.contract import *
from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.combat.mechanics import (
    DAMAGE_CLASS_UNKNOWN,
    DAMAGE_PHYSICAL,
    STATUS_FLAG_DEBUFF,
    STATUS_OVERLAP_EXTEND,
    STATUS_OVERLAP_IGNORE,
    STATUS_OVERLAP_OVERWRITE,
)


@dataclass(frozen=True)
class DeployableTerminalEvent:
    """One sparse fixed-shape event and its ability timing metadata."""

    event_time_seconds: float
    ability_duration_seconds: float
    cooldown_seconds: float
    f32: dict[int, float]
    i32: dict[int, int]
    flags: int


_DAMAGE_CAUSES = {"Physical": DAMAGE_PHYSICAL}
_OVERLAP_MODES = {
    "Ignore": STATUS_OVERLAP_IGNORE,
    "Extend": STATUS_OVERLAP_EXTEND,
    "Overwrite": STATUS_OVERLAP_OVERWRITE,
}
_AREA_SHAPES = {
    "Sphere": AREA_SHAPE_SPHERE,
    "Cylinder": AREA_SHAPE_CYLINDER,
}


def adapt_projectile_terminal_deployable(
    spec: ProjectileTerminalDeployableSpec,
) -> DeployableTerminalEvent:
    """Translate one typed host record without branching on asset identity."""

    if not isinstance(spec, ProjectileTerminalDeployableSpec):
        raise TypeError("spec must be ProjectileTerminalDeployableSpec")
    try:
        damage_cause = _DAMAGE_CAUSES[spec.area_damage_cause]
    except KeyError as error:
        raise ValueError(
            f"unsupported deployable damage cause {spec.area_damage_cause!r}"
        ) from error
    try:
        overlap = _OVERLAP_MODES[spec.status.overlap_behavior]
    except KeyError as error:
        raise ValueError(
            f"unsupported deployable status overlap {spec.status.overlap_behavior!r}"
        ) from error
    try:
        area_shape = _AREA_SHAPES[spec.area_shape]
    except KeyError as error:
        raise ValueError(
            f"unsupported deployable area shape {spec.area_shape!r}"
        ) from error
    terminal = encode_terminal_deployable_payload(
        collision_min=spec.collision_min,
        collision_max=spec.collision_max,
        deployable_collision_min=spec.deployable_collision_min,
        deployable_collision_max=spec.deployable_collision_max,
        deployable_id=semantic_id(spec.deployable_id),
        deployable_count_towards_global_limit=(
            spec.deployable_count_towards_global_limit
        ),
        deployable_max_live_count=spec.deployable_max_live_count,
        area_duration_seconds=spec.area_duration_seconds,
        area_interval_seconds=spec.area_interval_seconds,
        area_start_radius=spec.area_start_radius,
        area_end_radius=spec.area_end_radius,
        area_height=spec.area_height,
        area_radius_change_seconds=spec.area_radius_change_seconds,
        area_damage=spec.area_damage,
        area_shape=area_shape,
        area_attack_flags=spec.area_attack_flags,
        sticks_vertically=spec.sticks_vertically,
        status_healing_per_cycle=spec.status.healing_per_cycle,
    )
    f32 = {
        EF_PROJECTILE_SPEED: spec.launch_force,
        EF_PROJECTILE_GRAVITY: spec.gravity,
        EF_PROJECTILE_TERMINAL_VELOCITY: spec.terminal_velocity_air,
        EF_PROJECTILE_LIFETIME_SECONDS: spec.projectile_lifetime_seconds,
        EF_YAW_OFFSET_DEGREES: spec.spawn_yaw_degrees,
        EF_PITCH_OFFSET_DEGREES: spec.spawn_pitch_degrees,
        EF_PROJECTILE_FUSE_SECONDS: 0.0,
        EF_PROJECTILE_DEAD_TIME_SECONDS: -1.0,
        EF_FORCE_MAGNITUDE: 0.0,
        EF_AIR_RESISTANCE: 0.97,
        EF_AIR_RESISTANCE_MAX: 0.96,
        EF_GROUND_RESISTANCE: 0.94,
        EF_GROUND_RESISTANCE_MAX: 0.3,
        EF_RESISTANCE_THRESHOLD: 3.0,
        EF_ON_HIT_RESOURCE_DELTA: 0.0,
        EF_RANDOM_PERCENTAGE: 0.0,
        EF_PROJECTILE_SPAWN_OFFSET_X: spec.spawn_offset[0],
        EF_PROJECTILE_SPAWN_OFFSET_Y: spec.spawn_offset[1],
        EF_PROJECTILE_SPAWN_OFFSET_Z: spec.spawn_offset[2],
        EF_PROJECTILE_BOUNCINESS: spec.bounciness,
        EF_PROJECTILE_BOUNCE_LIMIT: spec.bounce_limit,
        EF_PROJECTILE_ROLLING_FRICTION_FACTOR: (spec.rolling_friction_factor),
        EF_STATUS_DURATION_SECONDS: spec.status.duration_seconds,
        EF_STATUS_COOLDOWN_SECONDS: (spec.status.calculator_cooldown_seconds),
        EF_STATUS_DAMAGE: 0.0,
        EF_STATUS_RESOURCE_DELTA: 0.0,
        EF_STATUS_SPEED_MULTIPLIER: (spec.status.horizontal_speed_multiplier),
        **terminal.f32,
    }
    i32 = {
        EI_DAMAGE_CAUSE: damage_cause,
        EI_DAMAGE_CLASS: DAMAGE_CLASS_UNKNOWN,
        EI_BLOCK_DAMAGE_RADIUS: 0,
        EI_PROJECTILE_KIND: PROJECTILE_DEPLOYABLE,
        EI_STATUS_ID: semantic_id(spec.status.effect_asset_id),
        EI_STATUS_DAMAGE_CAUSE: damage_cause,
        EI_STATUS_RESOURCE_ID: -1,
        EI_STATUS_OVERLAP_MODE: overlap,
        EI_FORCE_MODE: FORCE_SET,
        EI_FORCE_DIRECTION_MODE: FORCE_DIRECTION_LOCAL,
        EI_RESISTANCE_STYLE: 1,
        EI_ON_HIT_RESOURCE_ID: -1,
        EI_PROJECTILE_BOUNCE_COUNT: spec.bounce_count,
        EI_PROJECTILE_DIRECT_DAMAGE_CAUSE: damage_cause,
        **terminal.i32,
    }
    flags = (
        terminal.flags
        | EVENT_FLAG_ENTITY_ONLY_AREA
        | EVENT_FLAG_PROJECTILE_STANDARD_PHYSICS
    )
    if spec.allow_rolling:
        flags |= EVENT_FLAG_PROJECTILE_ALLOW_ROLLING
    if spec.status.debuff:
        flags |= STATUS_FLAG_DEBUFF
    return DeployableTerminalEvent(
        event_time_seconds=spec.launch_event_seconds,
        ability_duration_seconds=spec.ability_duration_seconds,
        cooldown_seconds=spec.cooldown_seconds,
        f32=f32,
        i32=i32,
        flags=flags,
    )


__all__ = ["DeployableTerminalEvent", "adapt_projectile_terminal_deployable"]
