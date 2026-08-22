"""Compiled crowd projectile, explosion, and persistent-area integration."""

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
    _batch_dt,
    _fold_random_keys,
    _reduce_query_bits,
    _select_tree,
    _set_failure,
)
from hytalegym.jax.combat.entities.impacts.runtime.kernel_damage import (
    _apply_area_damage,
    _apply_projectile_damage,
)
from hytalegym.jax.combat.entities.impacts.runtime.kernel_statuses import (
    _apply_area_statuses,
    _apply_projectile_statuses,
    _crossbow_commands,
    _effects_after_interactions,
)
from hytalegym.jax.combat.entities.impacts.runtime.kernel_checks import (
    _invalid_area_rows,
    _invalid_bindings,
    _invalid_projectile_rows,
    _invalid_query_values,
    _relationship_filter,
    _source_handles_valid,
    _stale_targets,
    _unsupported_candidates,
)


def tick_entity_impacts(
    state: EntityImpactState,
    arsenal: ArsenalState,
    bindings: EntityImpactBindings,
    queries: EntityImpactQueries,
    dt_seconds: jax.Array,
    rules: CombatMechanicsRules,
    *,
    random_keys: jax.Array | None = None,
) -> tuple[EntityImpactState, ArsenalState, EntityImpactInfo]:
    """Advance Arsenal projectile/area slots against complete world queries.

    Physics candidates and overlap masks are injected. All relationship
    filtering, source/target generations, payload resolution, status order,
    death, and lifecycle commits remain in this compiled combat function.
    """

    validate_impact_layout(state, bindings, queries)
    original_state, original_arsenal = state, arsenal
    effects = state.effects
    combat = effects.combat
    roster = combat.roster
    projectile = arsenal.projectiles
    area = arsenal.areas
    batch = roster.active.shape[0]
    dt = _batch_dt(dt_seconds, batch)
    bits = state.failure_bits
    bits = _set_failure(
        bits,
        (state.capability_bits != jnp.uint32(IMPACT_CAPABILITIES))
        | invalid_effect_rows(effects)
        | _invalid_projectile_rows(projectile)
        | _invalid_area_rows(area),
        IMPACT_FAILURE_INVALID_STATE,
    )
    bits = _set_failure(
        bits,
        ~jnp.isfinite(dt) | (dt < 0.0),
        IMPACT_FAILURE_INVALID_DT,
    )
    upstream = (
        (effects.failure_bits != jnp.uint32(0))
        | (roster.failure_bits != jnp.uint32(0))
        | (combat.mechanics.failure_bits != jnp.uint32(0))
        | (arsenal.failure_bits != jnp.uint32(0))
    )
    bits = _set_failure(bits, upstream, IMPACT_FAILURE_UPSTREAM)
    bits = _set_failure(
        bits,
        stale_effect_source_rows(effects),
        IMPACT_FAILURE_STALE_SOURCE,
    )
    bits = _set_failure(
        bits,
        _invalid_bindings(projectile, area, bindings),
        IMPACT_FAILURE_INVALID_BINDING,
    )

    projectile_source_valid = _source_handles_valid(
        roster,
        projectile.owner_entity_id,
        bindings.projectile_source_generation,
    )
    area_source_valid = _source_handles_valid(
        roster,
        area.owner_entity_id,
        bindings.area_source_generation,
    )
    bits = _set_failure(
        bits,
        jnp.any(projectile.active & ~projectile_source_valid, axis=1)
        | jnp.any(area.active & ~area_source_valid, axis=1),
        IMPACT_FAILURE_STALE_SOURCE,
    )

    delta = dt[:, None]
    ticking_dead = projectile.active & projectile.impacted
    dead_remaining = jnp.where(
        ticking_dead,
        projectile.dead_time_remaining - delta,
        projectile.dead_time_remaining,
    )
    removed_dead = ticking_dead & (dead_remaining <= jnp.float32(0.0))
    active_before = projectile.active & ~removed_dead
    free = active_before & ~projectile.impacted
    velocity = _standard_velocity(projectile, delta)
    end = projectile.position + velocity * delta[..., None]
    after_age = projectile.age_seconds + jnp.where(
        active_before,
        delta,
        0.0,
    )
    fuse = (
        free
        & (projectile.fuse_seconds > 0.0)
        & (projectile.age_seconds < projectile.fuse_seconds)
        & (after_age >= projectile.fuse_seconds)
    )
    safe_dt = jnp.maximum(delta, jnp.finfo(jnp.float32).tiny)
    fuse_fraction = jnp.clip(
        (projectile.fuse_seconds - projectile.age_seconds) / safe_dt,
        0.0,
        1.0,
    )

    area_age = area.age_seconds + jnp.where(area.active, delta, 0.0)
    area_clock = area.interval_clock_seconds + jnp.where(area.active, delta, 0.0)
    area_fires = area.active & (area_clock > area.interval_seconds)

    relevant_projectile = free
    relevant_area = area_fires
    projectile_candidates = (
        queries.projectile_entity_hit_mask | queries.projectile_explosion_candidate_mask
    )
    world_bits = (
        state.world_failure_bits
        | _reduce_query_bits(
            queries.projectile_failure_bits,
            relevant_projectile,
        )
        | _reduce_query_bits(
            queries.area_failure_bits,
            relevant_area,
        )
    )
    bad_query = (
        jnp.any(
            relevant_projectile
            & (
                ~queries.projectile_query_valid
                | (queries.projectile_failure_bits != jnp.uint32(0))
            ),
            axis=1,
        )
        | jnp.any(
            relevant_area
            & (
                ~queries.area_query_valid | (queries.area_failure_bits != jnp.uint32(0))
            ),
            axis=1,
        )
        | _invalid_query_values(
            projectile,
            queries,
        )
        | _unsupported_candidates(
            roster,
            projectile_candidates,
            relevant_projectile,
        )
        | _unsupported_candidates(
            roster,
            queries.area_candidate_mask,
            relevant_area,
        )
    )
    bits = _set_failure(bits, bad_query, IMPACT_FAILURE_QUERY)

    stale_projectile_target = _stale_targets(
        roster,
        projectile_candidates,
        queries.projectile_candidate_generation,
    )
    stale_area_target = _stale_targets(
        roster,
        queries.area_candidate_mask,
        queries.area_candidate_generation,
    )
    bits = _set_failure(
        bits,
        jnp.any(
            relevant_projectile[..., None] & stale_projectile_target,
            axis=(1, 2),
        )
        | jnp.any(
            relevant_area[..., None] & stale_area_target,
            axis=(1, 2),
        ),
        IMPACT_FAILURE_STALE_TARGET,
    )

    projectile_hit = _relationship_filter(
        roster,
        queries.projectile_entity_hit_mask,
        projectile.owner_entity_id,
        bindings.projectile_friendly_fire,
    )
    hit_fraction = jnp.where(
        projectile_hit,
        queries.projectile_entity_hit_fraction,
        jnp.inf,
    )
    first_entity_fraction = jnp.min(hit_fraction, axis=2)
    has_entity_hit = jnp.any(projectile_hit, axis=2)
    first_entity_mask = projectile_hit & (
        queries.projectile_entity_hit_fraction == first_entity_fraction[..., None]
    )
    world_fraction = jnp.where(
        queries.projectile_world_hit,
        queries.projectile_world_hit_fraction,
        jnp.inf,
    )
    candidate_fraction = jnp.stack(
        (
            jnp.where(has_entity_hit, first_entity_fraction, jnp.inf),
            world_fraction,
            jnp.where(fuse, fuse_fraction, jnp.inf),
        ),
        axis=2,
    )
    trigger_fraction = jnp.min(candidate_fraction, axis=2)
    trigger_cause = jnp.isfinite(candidate_fraction) & (
        candidate_fraction == trigger_fraction[..., None]
    )
    triggered = free & jnp.any(trigger_cause, axis=2)
    entity_trigger = triggered & trigger_cause[..., 0]
    has_explosion = projectile.explosion_radius > 0.0
    crossbow_bound = bindings.projectile_interaction_kind != CROSSBOW_IMPACT_NONE
    ambiguous = (triggered & (jnp.sum(trigger_cause.astype(jnp.int32), axis=2) > 1)) | (
        entity_trigger
        & (~has_explosion | crossbow_bound)
        & (jnp.sum(first_entity_mask.astype(jnp.int32), axis=2) > 1)
    )
    bits = _set_failure(
        bits,
        jnp.any(ambiguous, axis=1),
        IMPACT_FAILURE_AMBIGUOUS_TRIGGER,
    )
    impact = (
        projectile.position
        + (end - projectile.position)
        * jnp.where(
            triggered,
            trigger_fraction,
            jnp.float32(1.0),
        )[..., None]
    )

    explosion_mask = _relationship_filter(
        roster,
        queries.projectile_explosion_candidate_mask,
        projectile.owner_entity_id,
        bindings.projectile_friendly_fire,
    )
    explosion_mask &= triggered[..., None] & has_explosion[..., None]
    direct_mask = (
        first_entity_mask & entity_trigger[..., None] & ~has_explosion[..., None]
    )
    crossbow_mask = direct_mask & crossbow_bound[..., None]
    generic_mask = explosion_mask | (direct_mask & ~crossbow_bound[..., None])
    projectile_control_immune = (
        generic_mask
        & (
            (projectile.status_flags & jnp.uint32(STATUS_FLAG_CONTROL_IMMUNITY_GATED))
            != 0
        )[..., None]
        & (
            effects.combat.mechanics.control_immunity[:, None, :]
            >= CONTROL_IMMUNITY_MAXIMUM
        )
    )
    generic_mask &= ~projectile_control_immune
    component_invulnerable = roster.invulnerable[:, None, :] & generic_mask
    generic_mask &= ~roster.invulnerable[:, None, :]
    mixed_order = jnp.any(crossbow_mask, axis=(1, 2)) & jnp.any(
        generic_mask, axis=(1, 2)
    )
    bits = _set_failure(
        bits,
        mixed_order,
        IMPACT_FAILURE_MIXED_INTERACTION_ORDER,
    )

    row_enabled = bits == jnp.uint32(0)
    effects, projectile_resolution, projectile_events, overflow, failed = (
        _apply_projectile_damage(
            effects,
            projectile,
            bindings,
            generic_mask,
            impact,
            queries.projectile_explosion_distance,
            row_enabled,
            rules,
            _fold_random_keys(random_keys, 0),
        )
    )
    bits = _set_failure(bits, overflow, IMPACT_FAILURE_DAMAGE_OVERFLOW)
    bits = _set_failure(bits, failed, IMPACT_FAILURE_MECHANICS)
    row_enabled = bits == jnp.uint32(0)
    effects, projectile_status_count = _apply_projectile_statuses(
        effects,
        projectile,
        generic_mask,
        row_enabled,
    )
    bits = _set_failure(
        bits,
        effects.failure_bits != jnp.uint32(0),
        IMPACT_FAILURE_MECHANICS,
    )

    crossbow_commands = _crossbow_commands(
        effects,
        projectile,
        bindings,
        crossbow_mask,
        bits == jnp.uint32(0),
    )
    before_crossbow = effects
    interaction, crossbow_info = apply_crossbow_impacts(
        entity_interaction_state(effects.combat),
        crossbow_commands,
        rules,
    )
    effects = _effects_after_interactions(
        effects,
        interaction.combat,
    )
    crossbow_failed = interaction.failure_bits != jnp.uint32(0)
    bits = _set_failure(bits, crossbow_failed, IMPACT_FAILURE_CROSSBOW)
    effects = _select_tree(
        ~crossbow_failed,
        effects,
        before_crossbow,
    )

    area_mask = _relationship_filter(
        effects.combat.roster,
        queries.area_candidate_mask,
        area.owner_entity_id,
        bindings.area_friendly_fire,
    )
    area_mask &= area_fires[..., None]
    area_overlap_mask = area_mask
    area_control_immune = (
        area_mask
        & ((area.status_flags & jnp.uint32(STATUS_FLAG_CONTROL_IMMUNITY_GATED)) != 0)[
            ..., None
        ]
        & (
            effects.combat.mechanics.control_immunity[:, None, :]
            >= CONTROL_IMMUNITY_MAXIMUM
        )
    )
    area_mask &= ~area_control_immune
    area_component_invulnerable = (
        effects.combat.roster.invulnerable[:, None, :] & area_mask
    )
    area_mask &= ~effects.combat.roster.invulnerable[:, None, :]
    row_enabled = bits == jnp.uint32(0)
    effects, area_resolution, area_events, overflow, failed = _apply_area_damage(
        effects,
        area,
        area_mask,
        row_enabled,
        rules,
        _fold_random_keys(random_keys, 1),
    )
    bits = _set_failure(bits, overflow, IMPACT_FAILURE_DAMAGE_OVERFLOW)
    bits = _set_failure(bits, failed, IMPACT_FAILURE_MECHANICS)
    row_enabled = bits == jnp.uint32(0)
    effects, area_status_count = _apply_area_statuses(
        effects,
        area,
        area_mask,
        row_enabled,
    )
    bits = _set_failure(
        bits,
        effects.failure_bits != jnp.uint32(0),
        IMPACT_FAILURE_MECHANICS,
    )

    expired = active_before & (after_age >= projectile.lifetime_seconds)
    retained_impact = triggered & (projectile.dead_time_seconds >= jnp.float32(0.0))
    projectile_active = active_before & ~expired & (~triggered | retained_impact)
    projectile_impacted = (projectile.impacted & projectile_active) | retained_impact
    dead_remaining = jnp.where(
        retained_impact,
        projectile.dead_time_seconds,
        jnp.where(removed_dead, jnp.float32(0.0), dead_remaining),
    )
    projectile_candidate = projectile._replace(
        position=jnp.where(
            free[..., None],
            jnp.where(triggered[..., None], impact, end),
            projectile.position,
        ),
        velocity=jnp.where(
            (free & ~triggered)[..., None],
            velocity,
            jnp.where(
                triggered[..., None],
                jnp.float32(0.0),
                projectile.velocity,
            ),
        ),
        age_seconds=jnp.where(
            active_before,
            after_age,
            projectile.age_seconds,
        ),
        dead_time_remaining=dead_remaining,
        active=projectile_active,
        impacted=projectile_impacted,
        physics_initialized=(projectile.physics_initialized | free),
    )
    area_candidate = area._replace(
        age_seconds=area_age,
        interval_clock_seconds=jnp.where(area_fires, jnp.float32(0.0), area_clock),
        active=area.active & (area_age < area.duration_seconds),
    )
    arsenal_candidate = arsenal._replace(
        projectiles=projectile_candidate,
        areas=area_candidate,
    )

    valid = (bits == jnp.uint32(0)) & (world_bits == jnp.uint32(0))
    final_effects = _select_tree(valid, effects, original_state.effects)
    final_arsenal = _select_tree(valid, arsenal_candidate, original_arsenal)
    result = EntityImpactState(
        effects=final_effects,
        capability_bits=state.capability_bits,
        failure_bits=bits,
        world_failure_bits=world_bits,
    )
    damage_applied = (
        jnp.sum(projectile_resolution.applied_damage, axis=1)
        + jnp.sum(area_resolution.applied_damage, axis=1)
        + jnp.sum(crossbow_info.damage_applied, axis=1)
    )
    blocked = (
        jnp.sum(projectile_resolution.blocked.astype(jnp.int32), axis=1)
        + jnp.sum(area_resolution.blocked.astype(jnp.int32), axis=1)
        + jnp.sum(crossbow_info.blocked.astype(jnp.int32), axis=1)
    )
    invulnerable = (
        jnp.sum(
            projectile_resolution.invulnerable.astype(jnp.int32),
            axis=1,
        )
        + jnp.sum(area_resolution.invulnerable.astype(jnp.int32), axis=1)
        + jnp.sum(crossbow_info.invulnerable.astype(jnp.int32), axis=1)
        + jnp.sum(component_invulnerable.astype(jnp.int32), axis=(1, 2))
        + jnp.sum(
            projectile_control_immune.astype(jnp.int32),
            axis=(1, 2),
        )
        + jnp.sum(
            area_component_invulnerable.astype(jnp.int32),
            axis=(1, 2),
        )
        + jnp.sum(
            area_control_immune.astype(jnp.int32),
            axis=(1, 2),
        )
    )
    newly_dead = (
        final_effects.combat.roster.dead & ~original_state.effects.combat.roster.dead
    )
    event_mask = valid
    return (
        result,
        final_arsenal,
        EntityImpactInfo(
            projectile_triggers=jnp.where(
                event_mask,
                jnp.sum(triggered.astype(jnp.int32), axis=1),
                jnp.int32(0),
            ),
            projectile_direct_hits=jnp.where(
                event_mask,
                jnp.sum(direct_mask.astype(jnp.int32), axis=(1, 2)),
                jnp.int32(0),
            ),
            projectile_explosion_hits=jnp.where(
                event_mask,
                jnp.sum(explosion_mask.astype(jnp.int32), axis=(1, 2)),
                jnp.int32(0),
            ),
            area_hits=jnp.where(
                event_mask,
                jnp.sum(
                    area_overlap_mask.astype(jnp.int32),
                    axis=(1, 2),
                ),
                jnp.int32(0),
            ),
            damage_events=jnp.where(
                event_mask,
                projectile_events
                + area_events
                + jnp.sum(
                    crossbow_commands.requested.astype(jnp.int32),
                    axis=1,
                ),
                jnp.int32(0),
            ),
            status_applications=jnp.where(
                event_mask,
                projectile_status_count
                + area_status_count
                + jnp.sum(
                    (
                        crossbow_info.combo_1_applied | crossbow_info.combo_2_applied
                    ).astype(jnp.int32),
                    axis=1,
                ),
                jnp.int32(0),
            ),
            crossbow_impacts=jnp.where(
                event_mask,
                jnp.sum(
                    crossbow_commands.requested.astype(jnp.int32),
                    axis=1,
                ),
                jnp.int32(0),
            ),
            damage_applied=jnp.where(event_mask, damage_applied, jnp.float32(0.0)),
            blocked_hits=jnp.where(event_mask, blocked, jnp.int32(0)),
            invulnerable_hits=jnp.where(event_mask, invulnerable, jnp.int32(0)),
            newly_dead=newly_dead,
            failure_bits=bits,
            world_failure_bits=world_bits,
            valid=valid,
        ),
    )
