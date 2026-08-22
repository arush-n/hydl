"""Status/effect application and post-interaction effects for the impact kernel."""
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
from hytalegym.jax.combat.entities.impacts.runtime.kernel_support import (
    _gather,
    _pad_queries,
)


def _apply_projectile_statuses(
    effects,
    projectile,
    mask,
    row_enabled,
):
    return _apply_statuses(
        effects,
        mask,
        projectile.owner_entity_id,
        projectile.status_id,
        projectile.status_duration_seconds,
        projectile.status_cooldown_seconds,
        projectile.status_damage,
        projectile.status_damage_cause,
        projectile.status_resource_id,
        projectile.status_resource_delta,
        projectile.status_speed_multiplier,
        projectile.status_flags,
        projectile.status_overlap_mode,
        row_enabled,
    )


def _apply_area_statuses(
    effects,
    area,
    mask,
    row_enabled,
):
    return _apply_statuses(
        effects,
        mask,
        area.owner_entity_id,
        area.status_id,
        area.status_duration_seconds,
        area.status_cooldown_seconds,
        area.status_damage,
        area.status_damage_cause,
        area.status_resource_id,
        area.status_resource_delta,
        area.status_speed_multiplier,
        area.status_flags,
        area.status_overlap_mode,
        row_enabled,
    )


def _apply_statuses(
    effects,
    mask,
    source,
    effect_id,
    duration,
    cooldown,
    damage,
    damage_cause,
    resource_id,
    resource_delta,
    speed,
    flags,
    overlap_mode,
    row_enabled,
):
    mask = _pad_queries(mask, False)
    source = _pad_queries(source, -1)
    effect_id = _pad_queries(effect_id, 0)
    duration = _pad_queries(duration, 0.0)
    cooldown = _pad_queries(cooldown, 0.0)
    damage = _pad_queries(damage, 0.0)
    damage_cause = _pad_queries(damage_cause, 0)
    resource_id = _pad_queries(resource_id, -1)
    resource_delta = _pad_queries(resource_delta, 0.0)
    speed = _pad_queries(speed, 1.0)
    flags = _pad_queries(flags, 0)
    overlap_mode = _pad_queries(overlap_mode, 0)
    requested = jnp.any(mask, axis=2) & (effect_id > 0)
    valid = requested & row_enabled[:, None]
    selection = EntityTargetSelection(
        target_mask=mask & valid[..., None],
        target_count=jnp.sum(mask.astype(jnp.int32), axis=2),
        available_count=jnp.sum(mask.astype(jnp.int32), axis=2),
        source_slot=jnp.where(valid, source, jnp.int32(-1)),
        failure_bits=jnp.zeros_like(source, dtype=jnp.uint32),
        row_failure_bits=jnp.zeros((source.shape[0],), dtype=jnp.uint32),
        valid=valid,
    )
    payload = EntityStatusPayloads(
        requested=valid,
        effect_id=effect_id,
        duration_seconds=duration,
        cycle_cooldown_seconds=cooldown,
        damage_per_cycle=damage,
        damage_cause=damage_cause,
        healing_per_cycle=jnp.zeros_like(damage),
        resource_id=resource_id,
        resource_delta_per_cycle=resource_delta,
        speed_multiplier=speed,
        flags=flags,
        overlap_mode=overlap_mode,
    )
    result, info = apply_selected_entity_effects(effects, selection, payload)
    return result, info.application_count


def _effects_after_interactions(
    before: EntityEffectState,
    combat,
) -> EntityEffectState:
    old = before.combat.mechanics.statuses
    new = combat.mechanics.statuses
    changed = new.active & (
        ~old.active
        | (new.effect_id != old.effect_id)
        | (new.source_entity_id != old.source_entity_id)
    )
    source = jnp.clip(new.source_entity_id, 0, ENTITY_CAPACITY - 1)
    generation = combat.roster.generation[
        jnp.arange(source.shape[0])[:, None, None], source
    ]
    provenance = jnp.where(
        ~new.active,
        jnp.uint32(0),
        jnp.where(
            changed & (new.source_entity_id >= 0),
            generation,
            before.source_generation,
        ),
    )
    return before._replace(
        combat=combat,
        source_generation=provenance,
    )


def _crossbow_commands(
    effects,
    projectile,
    bindings,
    mask,
    row_enabled,
):
    requested = jnp.any(mask, axis=2) & row_enabled[:, None]
    target = jnp.argmax(mask, axis=2).astype(jnp.int32)
    target_generation = _gather(effects.combat.roster.generation, target)
    return CrossbowImpactCommands(
        requested=requested,
        kind=bindings.projectile_interaction_kind,
        source_slot=projectile.owner_entity_id,
        source_generation=bindings.projectile_source_generation,
        target_slot=jnp.where(requested, target, jnp.int32(-1)),
        target_generation=jnp.where(requested, target_generation, jnp.uint32(0)),
        knockback_yaw_degrees=(bindings.projectile_knockback_yaw_degrees),
        damage_multiplier=bindings.projectile_damage_multiplier,
        overflow=jnp.zeros((requested.shape[0],), dtype=jnp.bool_),
    )
