"""Row/payload validity predicates for the impact kernel."""
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
    PROJECTILE_BLUNDERBUSS_BULLET,
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
from hytalegym.jax.combat.entities.impacts.runtime.kernel_support import (
    _gather,
)


def _invalid_projectile_rows(projectile):
    status_valid = _valid_status_payload(projectile)
    force_valid = _valid_force_payload(
        projectile,
        projectile.force_direction_mode == FORCE_DIRECTION_POINT,
    )
    valid = (
        (projectile.owner_entity_id >= 0)
        & (projectile.owner_entity_id < ENTITY_CAPACITY)
        & (projectile.kind > PROJECTILE_NONE)
        & (projectile.kind <= PROJECTILE_BLUNDERBUSS_BULLET)
        & jnp.all(jnp.isfinite(projectile.position), axis=2)
        & jnp.all(jnp.isfinite(projectile.velocity), axis=2)
        & jnp.all(jnp.isfinite(projectile.half_extent), axis=2)
        & jnp.all(projectile.half_extent > 0.0, axis=2)
        & jnp.isfinite(projectile.age_seconds)
        & (projectile.age_seconds >= 0.0)
        & jnp.isfinite(projectile.lifetime_seconds)
        & (projectile.lifetime_seconds > 0.0)
        & jnp.isfinite(projectile.damage)
        & (projectile.damage >= 0.0)
        & jnp.isfinite(projectile.direct_damage)
        & (projectile.direct_damage >= 0.0)
        & (
            (projectile.direct_damage == 0.0)
            | (
                (projectile.direct_damage_cause >= 0)
                & (projectile.direct_damage_cause < DAMAGE_COUNT)
            )
        )
        & jnp.isfinite(projectile.random_percentage)
        & (projectile.random_percentage >= 0.0)
        & (projectile.damage_cause >= 0)
        & (projectile.damage_cause < DAMAGE_COUNT)
        & jnp.isfinite(projectile.gravity)
        & (projectile.gravity >= 0.0)
        & jnp.isfinite(projectile.terminal_velocity)
        & (projectile.terminal_velocity > 0.0)
        & jnp.isfinite(projectile.fuse_seconds)
        & (projectile.fuse_seconds >= 0.0)
        & jnp.isfinite(projectile.dead_time_seconds)
        & (projectile.dead_time_seconds >= -1.0)
        & jnp.isfinite(projectile.dead_time_remaining)
        & (projectile.dead_time_remaining >= 0.0)
        & (~projectile.impacted | (projectile.dead_time_seconds >= 0.0))
        & jnp.isfinite(projectile.explosion_radius)
        & (projectile.explosion_radius >= 0.0)
        & jnp.isfinite(projectile.explosion_falloff)
        & (projectile.explosion_falloff >= 0.0)
        & jnp.isfinite(projectile.force_velocity_y)
        & (
            (projectile.force_direction_mode == FORCE_DIRECTION_LOCAL)
            | (projectile.force_direction_mode == FORCE_DIRECTION_POINT)
        )
        & (projectile.on_hit_resource_id >= -1)
        & (projectile.on_hit_resource_id < RESOURCE_COUNT)
        & jnp.isfinite(projectile.on_hit_resource_delta)
        & force_valid
        & status_valid
    )
    return jnp.any(projectile.active & ~valid, axis=1)


def _invalid_area_rows(area):
    valid = (
        (area.owner_entity_id >= 0)
        & (area.owner_entity_id < ENTITY_CAPACITY)
        & (area.kind > AREA_NONE)
        & (area.kind <= AREA_GENERIC)
        & jnp.all(jnp.isfinite(area.center), axis=2)
        & jnp.isfinite(area.age_seconds)
        & (area.age_seconds >= 0.0)
        & jnp.isfinite(area.duration_seconds)
        & (area.duration_seconds > 0.0)
        & jnp.isfinite(area.interval_seconds)
        & (area.interval_seconds > 0.0)
        & jnp.isfinite(area.radius_change_seconds)
        & (area.radius_change_seconds > 0.0)
        & jnp.isfinite(area.start_radius)
        & (area.start_radius >= 0.0)
        & jnp.isfinite(area.end_radius)
        & (area.end_radius >= 0.0)
        & jnp.isfinite(area.height)
        & (area.height > 0.0)
        & jnp.isfinite(area.damage)
        & (area.damage >= 0.0)
        & jnp.isfinite(area.random_percentage)
        & (area.random_percentage >= 0.0)
        & (area.damage_cause >= 0)
        & (area.damage_cause < DAMAGE_COUNT)
        & _valid_force_payload(area)
        & _valid_status_payload(area)
    )
    return jnp.any(area.active & ~valid, axis=1)


def _valid_force_payload(value, point_direction=None):
    direction_length = jnp.linalg.norm(value.force_direction, axis=2)
    if point_direction is None:
        point_direction = jnp.zeros_like(value.force_magnitude, dtype=jnp.bool_)
    return (
        jnp.all(jnp.isfinite(value.force_direction), axis=2)
        & jnp.isfinite(value.force_magnitude)
        & (value.force_magnitude >= 0.0)
        & ((value.force_magnitude == 0.0) | point_direction | (direction_length > 0.0))
        & ((value.force_mode == FORCE_SET) | (value.force_mode == FORCE_ADD))
        & jnp.isfinite(value.air_resistance)
        & (value.air_resistance >= 0.0)
        & (value.air_resistance <= 1.0)
        & jnp.isfinite(value.air_resistance_max)
        & (value.air_resistance_max >= 0.0)
        & (value.air_resistance_max <= 1.0)
        & jnp.isfinite(value.ground_resistance)
        & (value.ground_resistance >= 0.0)
        & (value.ground_resistance <= 1.0)
        & jnp.isfinite(value.ground_resistance_max)
        & (value.ground_resistance_max >= 0.0)
        & (value.ground_resistance_max <= 1.0)
        & jnp.isfinite(value.resistance_threshold)
        & (value.resistance_threshold > 0.0)
        & ((value.resistance_style == 0) | (value.resistance_style == 1))
    )


def _valid_status_payload(value):
    active = value.status_id > 0
    return (
        (value.status_id >= 0)
        & jnp.isfinite(value.status_duration_seconds)
        & (value.status_duration_seconds >= 0.0)
        & (~active | (value.status_duration_seconds > 0.0))
        & jnp.isfinite(value.status_cooldown_seconds)
        & (value.status_cooldown_seconds >= 0.0)
        & jnp.isfinite(value.status_damage)
        & (value.status_damage >= 0.0)
        & (value.status_damage_cause >= 0)
        & (value.status_damage_cause < DAMAGE_COUNT)
        & (value.status_resource_id >= -1)
        & (value.status_resource_id < RESOURCE_COUNT)
        & jnp.isfinite(value.status_resource_delta)
        & jnp.isfinite(value.status_speed_multiplier)
        & (value.status_speed_multiplier > 0.0)
        & ((value.status_flags & jnp.uint32(0xFFFFFFFF ^ _STATUS_FLAG_MASK)) == 0)
        & (value.status_overlap_mode >= STATUS_OVERLAP_IGNORE)
        & (value.status_overlap_mode <= STATUS_OVERLAP_OVERWRITE)
    )


def _relationship_filter(
    roster,
    mask,
    source,
    friendly_fire,
):
    source_slot = jnp.clip(source, 0, ENTITY_CAPACITY - 1)
    source_team = _gather(roster.team_id, source_slot)
    same_team = (source_team[..., None] != TEAM_NONE) & (
        roster.team_id[:, None, :] == source_team[..., None]
    )
    entity_id = jnp.arange(ENTITY_CAPACITY, dtype=jnp.int32)[None, None, :]
    targetable = roster.active & roster.damageable & ~roster.intangible & ~roster.dead
    return (
        mask
        & targetable[:, None, :]
        & (entity_id != source[..., None])
        & (friendly_fire[..., None] | ~same_team)
    )


def _invalid_bindings(projectile, area, bindings):
    projectile_active = projectile.active
    interaction = bindings.projectile_interaction_kind
    invalid_interaction = (
        (interaction < CROSSBOW_IMPACT_NONE)
        | (interaction >= CROSSBOW_IMPACT_KIND_COUNT)
        | (
            (interaction == CROSSBOW_IMPACT_STANDARD)
            & (projectile.kind != PROJECTILE_ARROW)
        )
        | (
            (interaction == CROSSBOW_IMPACT_BIG_ARROW)
            & (projectile.kind != PROJECTILE_BIG_ARROW)
        )
        | (
            (projectile.kind == PROJECTILE_BIG_ARROW)
            & (interaction != CROSSBOW_IMPACT_BIG_ARROW)
        )
        | ((interaction != CROSSBOW_IMPACT_NONE) & (projectile.explosion_radius > 0.0))
    )
    invalid_projectile = projectile_active & (
        invalid_interaction
        | ~jnp.isfinite(bindings.projectile_damage_multiplier)
        | (bindings.projectile_damage_multiplier <= 0.0)
        | ~jnp.isfinite(bindings.projectile_knockback_yaw_degrees)
    )
    return (
        bindings.overflow
        | jnp.any(invalid_projectile, axis=1)
        | jnp.any(
            area.active & (~jnp.isfinite(area.damage) | (area.damage < 0.0)),
            axis=1,
        )
    )


def _invalid_query_values(projectile, queries):
    projectile_hit = queries.projectile_entity_hit_mask
    bad_hit_fraction = projectile_hit & (
        ~jnp.isfinite(queries.projectile_entity_hit_fraction)
        | (queries.projectile_entity_hit_fraction < 0.0)
        | (queries.projectile_entity_hit_fraction > 1.0)
    )
    bad_world_fraction = queries.projectile_world_hit & (
        ~jnp.isfinite(queries.projectile_world_hit_fraction)
        | (queries.projectile_world_hit_fraction < 0.0)
        | (queries.projectile_world_hit_fraction > 1.0)
    )
    explosion = queries.projectile_explosion_candidate_mask
    bad_explosion = explosion & (
        ~jnp.isfinite(queries.projectile_explosion_distance)
        | (queries.projectile_explosion_distance < 0.0)
        | (
            queries.projectile_explosion_distance
            > projectile.explosion_radius[..., None]
        )
    )
    return jnp.any(
        projectile.active[..., None] & (bad_hit_fraction | bad_explosion),
        axis=(1, 2),
    ) | jnp.any(projectile.active & bad_world_fraction, axis=1)


def _unsupported_candidates(roster, mask, relevant):
    supported = roster.damageable & ~roster.intangible & ~roster.dead
    return jnp.any(
        relevant[..., None] & mask & ~supported[:, None, :],
        axis=(1, 2),
    )


def _source_handles_valid(roster, source, generation):
    safe = jnp.clip(source, 0, ENTITY_CAPACITY - 1)
    return (
        (source >= 0)
        & (source < ENTITY_CAPACITY)
        & _gather(roster.active, safe)
        & (_gather(roster.generation, safe) == generation)
    )


def _stale_targets(roster, mask, generation):
    current = roster.generation[:, None, :]
    active = roster.active[:, None, :]
    return mask & (~active | (generation != current))
