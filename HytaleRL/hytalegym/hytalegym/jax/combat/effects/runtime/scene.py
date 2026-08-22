"""Learner-scene projection for compiled combat effects."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.effects.schema.contract import (
    EFFECT_FAILURE_HAZARD_OVERFLOW,
    EFFECT_FAILURE_PROJECTILE_OVERFLOW,
    PROJECTILE_FLAG_IMPACTED,
)
from hytalegym.jax.combat.effects.schema.types import CombatEffectsState
from hytalegym.jax.combat.observation.v1.schema.contract import (
    HAZARD_RADIUS_BLOCKS,
    PROJECTILE_RADIUS_BLOCKS,
)
from hytalegym.jax.combat.observation.v1.runtime.factory import empty_combat_scene
from hytalegym.jax.combat.observation.v1.schema.types import CombatSceneFeatures
from hytalegym.jax.combat.types import (
    AGENT_ENTITY,
    CombatParams,
    CombatState,
)


def combat_effects_scene(
    combat: CombatState,
    effects: CombatEffectsState,
    params: CombatParams,
) -> CombatSceneFeatures:
    """Encode active effects into the frozen v1 projectile/hazard slots."""

    batch = combat.position.shape[0]
    scene = empty_combat_scene(batch)
    valid = effects.failure_bits == jnp.uint32(0)
    agent_position = combat.position[:, None, AGENT_ENTITY]

    projectile = effects.projectiles
    projectile_relative = projectile.position - agent_position
    projectile_distance = jnp.linalg.norm(projectile_relative, axis=2)
    projectile_mask = (
        projectile.active
        & (projectile_distance <= jnp.float32(PROJECTILE_RADIUS_BLOCKS))
        & valid[:, None]
    )
    authored_lifetime = jnp.maximum(
        projectile.authored_lifetime_seconds,
        jnp.float32(1.0e-6),
    )
    velocity_scale = jnp.maximum(
        projectile.velocity_scale,
        jnp.float32(1.0e-6),
    )
    projectile_f32 = jnp.concatenate(
        (
            projectile_relative / jnp.float32(PROJECTILE_RADIUS_BLOCKS),
            projectile.velocity / velocity_scale[:, :, None],
            (projectile_distance / PROJECTILE_RADIUS_BLOCKS)[:, :, None],
            (projectile.age_seconds / authored_lifetime)[:, :, None],
            (
                jnp.maximum(
                    authored_lifetime - projectile.age_seconds,
                    jnp.float32(0.0),
                )
                / authored_lifetime
            )[:, :, None],
            (
                projectile.damage
                / jnp.maximum(params.agent_max_health, jnp.float32(1.0e-6))
            )[:, :, None],
            (
                jnp.max(projectile.half_extent, axis=2)
                / jnp.float32(PROJECTILE_RADIUS_BLOCKS)
            )[:, :, None],
            projectile.hostile.astype(jnp.float32)[:, :, None],
        ),
        axis=2,
    )
    projectile_flags = jnp.bitwise_or(
        projectile.flags,
        jnp.where(
            projectile.impacted,
            jnp.uint32(PROJECTILE_FLAG_IMPACTED),
            jnp.uint32(0),
        ),
    )
    projectile_i32 = jnp.stack(
        (
            projectile.kind,
            projectile.owner_entity_id,
            projectile_flags.astype(jnp.int32),
        ),
        axis=2,
    )
    (
        projectile_f32,
        projectile_i32,
        projectile_mask,
    ) = _pack_category(
        jnp.clip(projectile_f32, -1.0, 1.0).astype(jnp.float32),
        projectile_i32,
        projectile_mask,
    )

    hazard = effects.hazards
    hazard_relative = hazard.center - agent_position
    hazard_distance = jnp.linalg.norm(hazard_relative, axis=2)
    hazard_mask = (
        hazard.active
        & (hazard_distance <= jnp.float32(HAZARD_RADIUS_BLOCKS))
        & valid[:, None]
    )
    duration = jnp.maximum(
        hazard.duration_seconds,
        jnp.float32(1.0e-6),
    )
    hazard_f32 = jnp.concatenate(
        (
            hazard_relative / jnp.float32(HAZARD_RADIUS_BLOCKS),
            hazard.half_extent / jnp.float32(HAZARD_RADIUS_BLOCKS),
            (hazard_distance / HAZARD_RADIUS_BLOCKS)[:, :, None],
            hazard.intensity[:, :, None],
            (
                jnp.maximum(
                    duration - hazard.age_seconds,
                    jnp.float32(0.0),
                )
                / duration
            )[:, :, None],
            (
                hazard.damage_per_second
                / jnp.maximum(params.agent_max_health, jnp.float32(1.0e-6))
            )[:, :, None],
            hazard.activation[:, :, None],
            hazard.hostile.astype(jnp.float32)[:, :, None],
        ),
        axis=2,
    )
    hazard_i32 = jnp.stack(
        (
            hazard.kind,
            hazard.owner_entity_id,
            hazard.flags.astype(jnp.int32),
        ),
        axis=2,
    )
    hazard_f32, hazard_i32, hazard_mask = _pack_category(
        jnp.clip(hazard_f32, -1.0, 1.0).astype(jnp.float32),
        hazard_i32,
        hazard_mask,
    )
    return scene._replace(
        projectile_f32=projectile_f32,
        projectile_i32=projectile_i32,
        projectile_mask=projectile_mask,
        projectile_overflow=(
            effects.failure_bits & jnp.uint32(EFFECT_FAILURE_PROJECTILE_OVERFLOW)
        )
        != 0,
        hazard_f32=hazard_f32,
        hazard_i32=hazard_i32,
        hazard_mask=hazard_mask,
        hazard_overflow=(
            effects.failure_bits & jnp.uint32(EFFECT_FAILURE_HAZARD_OVERFLOW)
        )
        != 0,
    )


def merge_combat_effects_scene(
    base: CombatSceneFeatures,
    effects: CombatSceneFeatures,
) -> CombatSceneFeatures:
    """Use calibrated entities plus effect-owned projectile/hazard fields."""

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


def _pack_category(
    float_values: jax.Array,
    integer_values: jax.Array,
    mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    capacity = mask.shape[1]
    slot = jnp.arange(capacity, dtype=jnp.int32)[None, :]
    order = jnp.argsort(
        jnp.where(mask, slot, slot + capacity),
        axis=1,
        stable=True,
    )
    packed_mask = jnp.take_along_axis(mask, order, axis=1)
    packed_f32 = jnp.take_along_axis(
        float_values,
        order[:, :, None],
        axis=1,
    )
    packed_i32 = jnp.take_along_axis(
        integer_values,
        order[:, :, None],
        axis=1,
    )
    return (
        jnp.where(packed_mask[:, :, None], packed_f32, 0.0),
        jnp.where(packed_mask[:, :, None], packed_i32, 0),
        packed_mask,
    )
