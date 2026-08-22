"""Projection of arsenal projectiles and areas into frozen scene tokens."""

from __future__ import annotations

import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.schema.contract import (
    ARSENAL_FAILURE_AREA_OVERFLOW,
    ARSENAL_FAILURE_PROJECTILE_OVERFLOW,
)
from hytalegym.jax.combat.arsenal.schema.types import ArsenalState
from hytalegym.jax.combat.observation.v1.schema.contract import (
    HAZARD_RADIUS_BLOCKS,
    PROJECTILE_RADIUS_BLOCKS,
)
from hytalegym.jax.combat.observation.v1.runtime.factory import empty_combat_scene
from hytalegym.jax.combat.types import (
    AGENT_ENTITY,
    CombatParams,
    CombatState,
)


def arsenal_scene(
    combat: CombatState,
    arsenal: ArsenalState,
    params: CombatParams,
):
    """Encode active projectiles/areas without selecting world geometry."""

    scene = empty_combat_scene(combat.position.shape[0])
    valid = arsenal.failure_bits == jnp.uint32(0)
    origin = combat.position[:, None, AGENT_ENTITY]
    projectile = arsenal.projectiles
    relative = projectile.position - origin
    distance = jnp.linalg.norm(relative, axis=2)
    lifetime = jnp.maximum(projectile.lifetime_seconds, 1.0e-6)
    velocity_scale = jnp.maximum(projectile.terminal_velocity, 1.0e-6)
    projectile_mask = (
        projectile.active
        & (distance <= jnp.float32(PROJECTILE_RADIUS_BLOCKS))
        & valid[:, None]
    )
    projectile_f32 = jnp.concatenate(
        (
            relative / jnp.float32(PROJECTILE_RADIUS_BLOCKS),
            projectile.velocity / velocity_scale[..., None],
            (distance / PROJECTILE_RADIUS_BLOCKS)[..., None],
            (projectile.age_seconds / lifetime)[..., None],
            (
                jnp.maximum(lifetime - projectile.age_seconds, 0.0)
                / lifetime
            )[..., None],
            (
                projectile.damage
                / jnp.maximum(params.agent_max_health, 1.0e-6)
            )[..., None],
            (
                projectile.explosion_radius / PROJECTILE_RADIUS_BLOCKS
            )[..., None],
            (projectile.owner_entity_id != AGENT_ENTITY)
            .astype(jnp.float32)[..., None],
        ),
        axis=2,
    )
    projectile_flags = (
        projectile.physics_initialized.astype(jnp.int32)
        | ((projectile.explosion_radius > 0.0).astype(jnp.int32) << 1)
        | ((projectile.fuse_seconds > 0.0).astype(jnp.int32) << 2)
        | (projectile.impacted.astype(jnp.int32) << 3)
    )
    projectile_i32 = jnp.stack(
        (
            projectile.kind,
            projectile.owner_entity_id,
            projectile_flags,
        ),
        axis=2,
    )
    projectile_f32, projectile_i32, projectile_mask = _pack(
        projectile_f32,
        projectile_i32,
        projectile_mask,
    )

    area = arsenal.areas
    relative = area.center - origin
    distance = jnp.linalg.norm(relative, axis=2)
    duration = jnp.maximum(area.duration_seconds, 1.0e-6)
    fraction = jnp.clip(
        area.age_seconds / jnp.maximum(area.radius_change_seconds, 1.0e-6),
        0.0,
        1.0,
    )
    radius = area.start_radius + (area.end_radius - area.start_radius) * fraction
    half_extent = jnp.stack((radius, area.height * 0.5, radius), axis=2)
    area_mask = (
        area.active
        & (distance <= jnp.float32(HAZARD_RADIUS_BLOCKS))
        & valid[:, None]
    )
    area_f32 = jnp.concatenate(
        (
            relative / jnp.float32(HAZARD_RADIUS_BLOCKS),
            half_extent / jnp.float32(HAZARD_RADIUS_BLOCKS),
            (distance / HAZARD_RADIUS_BLOCKS)[..., None],
            jnp.ones_like(distance)[..., None],
            (
                jnp.maximum(duration - area.age_seconds, 0.0) / duration
            )[..., None],
            (
                area.damage
                / jnp.maximum(area.interval_seconds, 1.0e-6)
                / jnp.maximum(params.agent_max_health, 1.0e-6)
            )[..., None],
            jnp.ones_like(distance)[..., None],
            (area.owner_entity_id != AGENT_ENTITY)
            .astype(jnp.float32)[..., None],
        ),
        axis=2,
    )
    area_i32 = jnp.stack(
        (
            area.kind,
            area.owner_entity_id,
            jnp.zeros_like(area.kind),
        ),
        axis=2,
    )
    area_f32, area_i32, area_mask = _pack(
        area_f32,
        area_i32,
        area_mask,
    )
    return scene._replace(
        projectile_f32=jnp.clip(projectile_f32, -1.0, 1.0),
        projectile_i32=projectile_i32,
        projectile_mask=projectile_mask,
        projectile_overflow=(
            arsenal.failure_bits
            & jnp.uint32(ARSENAL_FAILURE_PROJECTILE_OVERFLOW)
        )
        != 0,
        hazard_f32=jnp.clip(area_f32, -1.0, 1.0),
        hazard_i32=area_i32,
        hazard_mask=area_mask,
        hazard_overflow=(
            arsenal.failure_bits & jnp.uint32(ARSENAL_FAILURE_AREA_OVERFLOW)
        )
        != 0,
    )


def merge_arsenal_scene(base, effects):
    """Keep base entities and replace only arsenal-owned token categories."""

    return base._replace(
        projectile_f32=effects.projectile_f32,
        projectile_i32=effects.projectile_i32,
        projectile_mask=effects.projectile_mask,
        projectile_overflow=effects.projectile_overflow,
        hazard_f32=effects.hazard_f32,
        hazard_i32=effects.hazard_i32,
        hazard_mask=effects.hazard_mask,
        hazard_overflow=effects.hazard_overflow,
    )


def _pack(float_values, integer_values, mask):
    capacity = mask.shape[1]
    source = jnp.broadcast_to(
        jnp.arange(capacity, dtype=jnp.int32)[None, :],
        mask.shape,
    )
    active_count = jnp.sum(mask.astype(jnp.int32), axis=1)
    active_rank = jnp.cumsum(mask.astype(jnp.int32), axis=1) - 1
    inactive_rank = active_count[:, None] + (
        jnp.cumsum((~mask).astype(jnp.int32), axis=1) - 1
    )
    destination = jnp.where(mask, active_rank, inactive_rank)
    order = jnp.zeros_like(source).at[
        jnp.arange(mask.shape[0], dtype=jnp.int32)[:, None],
        destination,
    ].set(source, unique_indices=True)
    packed_mask = jnp.take_along_axis(mask, order, axis=1)
    return (
        jnp.where(
            packed_mask[..., None],
            jnp.take_along_axis(float_values, order[..., None], axis=1),
            0.0,
        ),
        jnp.where(
            packed_mask[..., None],
            jnp.take_along_axis(integer_values, order[..., None], axis=1),
            0,
        ),
        packed_mask,
    )
