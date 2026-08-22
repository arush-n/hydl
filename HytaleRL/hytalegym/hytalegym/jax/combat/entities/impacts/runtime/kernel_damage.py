"""Projectile and area damage application for the impact kernel."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal import (
    AREA_GENERIC,
    AREA_NONE,
    FORCE_ADD,
    FORCE_DIRECTION_LOCAL,
    FORCE_DIRECTION_POINT,
    FORCE_SET,
    PROJECTILE_ARROW,
    PROJECTILE_BIG_ARROW,
    PROJECTILE_NONE,
    PROJECTILE_SPEAR,
    ArsenalState,
)
from hytalegym.jax.combat.arsenal.projectiles.runtime import (
    _standard_velocity,
)
from hytalegym.jax.combat.entities import (
    ENTITY_CAPACITY,
    SELECTOR_CAPACITY,
    TEAM_NONE,
    EntityStatusPayloads,
    EntityTargetSelection,
)
from hytalegym.jax.combat.entities.effects import (
    EntityEffectState,
    apply_selected_entity_effects,
    invalid_effect_rows,
    stale_effect_source_rows,
)
from hytalegym.jax.combat.entities.impacts.schema.contract import (
    IMPACT_CAPABILITIES,
    IMPACT_FAILURE_AMBIGUOUS_TRIGGER,
    IMPACT_FAILURE_CROSSBOW,
    IMPACT_FAILURE_DAMAGE_OVERFLOW,
    IMPACT_FAILURE_INVALID_BINDING,
    IMPACT_FAILURE_INVALID_DT,
    IMPACT_FAILURE_INVALID_STATE,
    IMPACT_FAILURE_MECHANICS,
    IMPACT_FAILURE_MIXED_INTERACTION_ORDER,
    IMPACT_FAILURE_QUERY,
    IMPACT_FAILURE_STALE_SOURCE,
    IMPACT_FAILURE_STALE_TARGET,
    IMPACT_FAILURE_UPSTREAM,
)
from hytalegym.jax.combat.entities.impacts.runtime.damage import (
    DenseImpactDamage,
    apply_dense_impact_damage,
)
from hytalegym.jax.combat.entities.impacts.schema.types import (
    EntityImpactBindings,
    EntityImpactInfo,
    EntityImpactQueries,
    EntityImpactState,
)
from hytalegym.jax.combat.entities.impacts.schema.validation import (
    validate_impact_layout,
)
from hytalegym.jax.combat.entities.interactions import (
    CROSSBOW_IMPACT_BIG_ARROW,
    CROSSBOW_IMPACT_KIND_COUNT,
    CROSSBOW_IMPACT_NONE,
    CROSSBOW_IMPACT_STANDARD,
    CrossbowImpactCommands,
    apply_crossbow_impacts,
    entity_interaction_state,
)
from hytalegym.jax.combat.mechanics import (
    CONTROL_IMMUNITY_MAXIMUM,
    DAMAGE_COUNT,
    RESOURCE_COUNT,
    STATUS_FLAG_DEBUFF,
    STATUS_FLAG_DISABLE_ABILITIES,
    STATUS_FLAG_DISABLE_MOVEMENT,
    STATUS_FLAG_DISABLE_SPRINT,
    STATUS_FLAG_CONTROL_IMMUNITY_GATED,
    STATUS_FLAG_IGNORE_KNOCKBACK,
    STATUS_FLAG_INVULNERABLE,
    STATUS_OVERLAP_IGNORE,
    STATUS_OVERLAP_OVERWRITE,
    CombatMechanicsRules,
)


_STATUS_FLAG_MASK = (
    STATUS_FLAG_INVULNERABLE
    | STATUS_FLAG_DISABLE_MOVEMENT
    | STATUS_FLAG_DISABLE_ABILITIES
    | STATUS_FLAG_IGNORE_KNOCKBACK
    | STATUS_FLAG_DEBUFF
    | STATUS_FLAG_DISABLE_SPRINT
    | STATUS_FLAG_CONTROL_IMMUNITY_GATED
)


def _apply_projectile_damage(
    effects,
    projectile,
    bindings,
    mask,
    impact,
    explosion_distance,
    row_enabled,
    rules,
    random_keys,
):
    offset = effects.combat.roster.position[:, None, :, :] - impact[:, :, None, :]
    direction = offset / jnp.maximum(
        jnp.linalg.norm(offset, axis=3, keepdims=True),
        jnp.finfo(jnp.float32).tiny,
    )
    point_force = direction * projectile.force_magnitude[..., None, None]
    point_force = point_force.at[..., 1].set(projectile.force_velocity_y[..., None])
    authored_force = (
        projectile.force_direction * projectile.force_magnitude[..., None]
    )[..., None, :]
    force = jnp.where(
        (projectile.force_direction_mode == FORCE_DIRECTION_POINT)[..., None, None],
        point_force,
        authored_force,
    )
    attenuation = jnp.where(
        (projectile.explosion_radius > 0.0)[..., None],
        jnp.clip(
            1.0
            - projectile.explosion_falloff[..., None]
            * explosion_distance
            / jnp.maximum(
                projectile.explosion_radius[..., None],
                jnp.finfo(jnp.float32).tiny,
            ),
            0.0,
            1.0,
        ),
        1.0,
    )
    dense = DenseImpactDamage(
        requested=mask,
        source_slot=projectile.owner_entity_id,
        amount=(projectile.damage * bindings.projectile_damage_multiplier)[..., None]
        * attenuation,
        random_percentage=projectile.random_percentage,
        damage_class=projectile.damage_class,
        cause=projectile.damage_cause,
        knockback_velocity=jnp.where(mask[..., None], force, jnp.float32(0.0)),
        force_mode=projectile.force_mode,
        air_resistance=projectile.air_resistance,
        air_resistance_max=projectile.air_resistance_max,
        ground_resistance=projectile.ground_resistance,
        ground_resistance_max=projectile.ground_resistance_max,
        resistance_threshold=projectile.resistance_threshold,
        resistance_style=projectile.resistance_style,
        on_hit_resource_id=projectile.on_hit_resource_id,
        on_hit_resource_delta=projectile.on_hit_resource_delta,
    )
    combat, resolution, count, overflow, failed = apply_dense_impact_damage(
        effects.combat,
        dense,
        rules,
        row_enabled,
        random_keys=random_keys,
    )
    return (
        effects._replace(combat=combat),
        resolution,
        count,
        overflow,
        failed,
    )


def _apply_area_damage(
    effects,
    area,
    mask,
    row_enabled,
    rules,
    random_keys,
):
    force = (area.force_direction * area.force_magnitude[..., None])[..., None, :]
    damage_mask = mask & (area.damage > 0.0)[..., None]
    dense = DenseImpactDamage(
        requested=damage_mask,
        source_slot=area.owner_entity_id,
        amount=jnp.broadcast_to(area.damage[..., None], mask.shape),
        random_percentage=area.random_percentage,
        damage_class=area.damage_class,
        cause=area.damage_cause,
        knockback_velocity=jnp.where(mask[..., None], force, jnp.float32(0.0)),
        force_mode=area.force_mode,
        air_resistance=area.air_resistance,
        air_resistance_max=area.air_resistance_max,
        ground_resistance=area.ground_resistance,
        ground_resistance_max=area.ground_resistance_max,
        resistance_threshold=area.resistance_threshold,
        resistance_style=area.resistance_style,
        on_hit_resource_id=jnp.full_like(area.kind, -1),
        on_hit_resource_delta=jnp.zeros_like(area.damage),
    )
    combat, resolution, count, overflow, failed = apply_dense_impact_damage(
        effects.combat,
        dense,
        rules,
        row_enabled,
        random_keys=random_keys,
    )
    return (
        effects._replace(combat=combat),
        resolution,
        count,
        overflow,
        failed,
    )
