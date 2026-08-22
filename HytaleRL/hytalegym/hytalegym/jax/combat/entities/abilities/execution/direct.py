"""Direct damage, potion, resource, status, and force ability events."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal import (
    EF_AIR_RESISTANCE,
    EF_AIR_RESISTANCE_MAX,
    EF_ANGLED_ANGLE_DEGREES,
    EF_ANGLED_DAMAGE,
    EF_ANGLED_DISTANCE_DEGREES,
    EF_ANGLED_FORCE_MAGNITUDE,
    EF_ANGLED_FORCE_X,
    EF_ANGLED_FORCE_Y,
    EF_ANGLED_FORCE_Z,
    EF_DAMAGE,
    EF_FORCE_MAGNITUDE,
    EF_FORCE_X,
    EF_FORCE_Y,
    EF_FORCE_Z,
    EF_GROUND_RESISTANCE,
    EF_GROUND_RESISTANCE_MAX,
    EF_ON_HIT_RESOURCE_DELTA,
    EF_RANDOM_PERCENTAGE,
    EF_RESISTANCE_THRESHOLD,
    EF_STATUS_COOLDOWN_SECONDS,
    EF_STATUS_DAMAGE,
    EF_STATUS_DURATION_SECONDS,
    EF_STATUS_HEALING,
    EF_STATUS_RESOURCE_DELTA,
    EF_STATUS_SPEED_MULTIPLIER,
    EI_DAMAGE_CAUSE,
    EI_DAMAGE_CLASS,
    EI_FORCE_DIRECTION_MODE,
    EI_FORCE_MODE,
    EI_ON_HIT_RESOURCE_ID,
    EI_RESISTANCE_STYLE,
    EI_STATUS_DAMAGE_CAUSE,
    EI_STATUS_ID,
    EI_STATUS_OVERLAP_MODE,
    EI_STATUS_RESOURCE_ID,
    EI_TARGET_MODE,
    EVENT_AREA,
    EVENT_CLEAR_STATUS,
    EVENT_FLAG_ANGLED_DAMAGE,
    EVENT_FLAG_STATUS_VALUE_PERCENT,
    EVENT_FLAG_VALUE_PERCENT,
    EVENT_FORCE,
    EVENT_HEAL,
    EVENT_MELEE_CONE,
    EVENT_PROJECTILE,
    EVENT_RADIAL_DAMAGE,
    EVENT_RESOURCE,
    EVENT_STATUS,
    EVENT_STATUS_FLAG_MASK,
    FORCE_SET,
    FORCE_DIRECTION_DIRECTIONAL,
    FORCE_DIRECTION_POINT,
    TARGET_BOTH,
    TARGET_OTHER,
    TARGET_OTHER_OR_SELF,
    TARGET_SELF,
)
from hytalegym.jax.combat.arsenal.effects.events import (
    adjust_vertical_force,
)
from hytalegym.jax.combat.entities import (
    ENTITY_CAPACITY,
    TEAM_NONE,
)
from hytalegym.jax.combat.entities.abilities.schema.contract import (
    ABILITY_DAMAGE_CAPACITY,
    ABILITY_FAILURE_AMBIGUOUS_TARGET,
    ABILITY_FAILURE_DAMAGE_OVERFLOW,
    ABILITY_FAILURE_INVALID_PROGRAM,
    ABILITY_FAILURE_INVALID_STATE,
    ABILITY_FAILURE_MECHANICS,
    ABILITY_FAILURE_QUERY,
    ABILITY_FAILURE_STALE_SOURCE,
    ABILITY_FAILURE_STALE_TARGET,
    ABILITY_FAILURE_STATUS_OVERFLOW,
    ABILITY_FAILURE_UPSTREAM,
)
from hytalegym.jax.combat.entities.abilities.execution.programs import (
    events_match_program,
)
from hytalegym.jax.combat.entities.abilities.schema.types import (
    EntityAbilityResolveInfo,
)
from hytalegym.jax.combat.entities.abilities.schema.validation import (
    validate_entity_ability_query_layout,
)
from hytalegym.jax.combat.entities.effects import (
    invalid_effect_rows,
    stale_effect_source_rows,
)
from hytalegym.jax.combat.mechanics import (
    RESOURCE_COUNT,
    RESOURCE_STAMINA,
    STATUS_APPLICATION_CAPACITY,
    apply_damage_forces,
    apply_statuses,
    clear_statuses,
    empty_damage_events,
    empty_status_applications,
    resolve_damage_events,
)


def resolve_entity_ability_events(
    state,
    effects,
    events,
    queries,
    programs,
    rules,
    *,
    random_keys=None,
):
    """Resolve fixed packed events against injected target selections."""

    validate_entity_ability_query_layout(events, queries)
    batch = events.requested.shape[0]
    roster = effects.combat.roster
    mechanics = effects.combat.mechanics
    batch_index = jnp.arange(batch, dtype=jnp.int32)[:, None]
    source = events.source_slot
    safe_source = jnp.clip(source, 0, ENTITY_CAPACITY - 1)
    source_valid = (
        (source >= 0)
        & (source < ENTITY_CAPACITY)
        & roster.active[batch_index, safe_source]
        & roster.damageable[batch_index, safe_source]
        & ~roster.dead[batch_index, safe_source]
        & (events.source_generation == roster.generation[batch_index, safe_source])
        & (
            events.source_generation
            == state.source_generation[batch_index, safe_source]
        )
        & (events.weapon_family == state.weapon_family[batch_index, safe_source])
    )
    requested = events.requested
    kind = events.kind
    target_mode = events.i32[..., EI_TARGET_MODE]
    damage_kind = (kind == EVENT_MELEE_CONE) | (kind == EVENT_RADIAL_DAMAGE)
    targeted_kind = (
        (kind == EVENT_STATUS)
        | (kind == EVENT_RESOURCE)
        | (kind == EVENT_HEAL)
        | (kind == EVENT_FORCE)
        | (kind == EVENT_CLEAR_STATUS)
    )
    direct_kind = damage_kind | targeted_kind
    query_required = requested & (
        damage_kind | (targeted_kind & (target_mode != TARGET_SELF))
    )

    bits = state.failure_bits
    world_bits = state.world_failure_bits | events.world_failure_bits
    bits = _set_failure(
        bits,
        invalid_effect_rows(effects) | stale_effect_source_rows(effects),
        ABILITY_FAILURE_INVALID_STATE,
    )
    bits = _set_failure(
        bits,
        ~events.valid
        | (events.failure_bits != jnp.uint32(0))
        | (effects.failure_bits != jnp.uint32(0))
        | (roster.failure_bits != jnp.uint32(0))
        | (mechanics.failure_bits != jnp.uint32(0)),
        ABILITY_FAILURE_UPSTREAM,
    )
    bits = _set_failure(
        bits,
        jnp.any(requested & ~source_valid, axis=1),
        ABILITY_FAILURE_STALE_SOURCE,
    )
    bits = _set_failure(
        bits,
        jnp.any(
            requested & ~events_match_program(events, programs),
            axis=1,
        ),
        ABILITY_FAILURE_INVALID_PROGRAM,
    )
    valid_kind = direct_kind | (kind == EVENT_PROJECTILE) | (kind == EVENT_AREA)
    bits = _set_failure(
        bits,
        jnp.any(requested & ~valid_kind, axis=1),
        ABILITY_FAILURE_INVALID_PROGRAM,
    )
    query_failed = query_required & (
        ~queries.query_valid | (queries.failure_bits != jnp.uint32(0))
    )
    bits = _set_failure(
        bits,
        jnp.any(query_failed, axis=1),
        ABILITY_FAILURE_QUERY,
    )
    world_bits |= jnp.bitwise_or.reduce(
        jnp.where(
            query_required,
            queries.failure_bits,
            jnp.uint32(0),
        ),
        axis=1,
    )

    entity_ids = jnp.arange(ENTITY_CAPACITY, dtype=jnp.int32)[None, None, :]
    raw_candidate = queries.candidate_mask & query_required[..., None]
    generation_matches = queries.candidate_generation == roster.generation[:, None, :]
    stale_target = raw_candidate & (~roster.active[:, None, :] | ~generation_matches)
    bits = _set_failure(
        bits,
        jnp.any(stale_target, axis=(1, 2)),
        ABILITY_FAILURE_STALE_TARGET,
    )
    targetable = (
        roster.active
        & roster.damageable
        & ~roster.intangible
        & ~roster.invulnerable
        & ~roster.dead
    )
    invalid_candidate = raw_candidate & (
        ~targetable[:, None, :] | (entity_ids == source[..., None])
    )
    bits = _set_failure(
        bits,
        jnp.any(invalid_candidate, axis=(1, 2)),
        ABILITY_FAILURE_QUERY,
    )
    source_team = roster.team_id[batch_index, safe_source]
    same_team = (source_team[..., None] != TEAM_NONE) & (
        roster.team_id[:, None, :] == source_team[..., None]
    )
    external = (
        raw_candidate
        & targetable[:, None, :]
        & generation_matches
        & (entity_ids != source[..., None])
        & (queries.friendly_fire[..., None] | ~same_team)
    )
    single_target_kind = targeted_kind & (target_mode != TARGET_SELF)
    external_count = jnp.sum(external.astype(jnp.int32), axis=2)
    bits = _set_failure(
        bits,
        jnp.any(
            requested & single_target_kind & (external_count > 1),
            axis=1,
        ),
        ABILITY_FAILURE_AMBIGUOUS_TARGET,
    )

    source_mask = jax.nn.one_hot(safe_source, ENTITY_CAPACITY, dtype=jnp.bool_)
    self_target = (
        requested
        & targeted_kind
        & (
            (target_mode == TARGET_SELF)
            | (target_mode == TARGET_BOTH)
            | ((target_mode == TARGET_OTHER_OR_SELF) & (external_count == 0))
        )
    )[..., None] & source_mask
    other_target = (
        requested
        & targeted_kind
        & (
            (target_mode == TARGET_OTHER)
            | (target_mode == TARGET_BOTH)
            | (target_mode == TARGET_OTHER_OR_SELF)
        )
    )[..., None] & external
    direct_targets = self_target | other_target
    damage_targets = (requested & damage_kind)[..., None] & external

    source_yaw = roster.yaw_degrees[batch_index, safe_source]
    damage_events, damage_count, damage_overflow = _pack_damage(
        events, damage_targets, roster
    )
    applications, status_count, status_overflow = _pack_statuses(
        events,
        direct_targets & (kind == EVENT_STATUS)[..., None],
        roster,
        rules,
    )
    bits = _set_failure(
        bits,
        damage_overflow,
        ABILITY_FAILURE_DAMAGE_OVERFLOW,
    )
    bits = _set_failure(
        bits,
        status_overflow,
        ABILITY_FAILURE_STATUS_OVERFLOW,
    )
    pre_valid = (bits == jnp.uint32(0)) & (world_bits == jnp.uint32(0))

    health = _apply_heals(
        roster.health,
        roster.max_health,
        events,
        direct_targets,
        pre_valid,
    )
    candidate_mechanics = _apply_resources(
        mechanics,
        rules,
        events,
        direct_targets,
        pre_valid,
    )
    candidate_mechanics = _apply_forces(
        candidate_mechanics,
        events,
        direct_targets,
        source_yaw,
        pre_valid,
    )
    clear_targets = direct_targets & (kind == EVENT_CLEAR_STATUS)[..., None]
    clear_ids = jnp.swapaxes(
        jnp.where(
            clear_targets,
            events.i32[..., EI_STATUS_ID, None],
            jnp.int32(0),
        ),
        1,
        2,
    )
    candidate_mechanics, cleared = clear_statuses(candidate_mechanics, clear_ids)
    source_generation = jnp.where(
        cleared,
        jnp.uint32(0),
        effects.source_generation,
    )
    applications = applications._replace(
        requested=applications.requested & pre_valid[:, None, None]
    )
    before_status = candidate_mechanics.statuses
    candidate_mechanics = apply_statuses(candidate_mechanics, applications)
    source_generation = _track_status_sources(
        source_generation,
        before_status,
        candidate_mechanics.statuses,
        roster,
    )
    damage_events = damage_events._replace(
        requested=damage_events.requested & pre_valid[:, None]
    )
    candidate_mechanics, health, resolution = resolve_damage_events(
        candidate_mechanics,
        health,
        roster.position,
        roster.yaw_degrees,
        damage_events,
        rules,
        random_keys=random_keys,
    )
    candidate_mechanics = apply_damage_forces(candidate_mechanics, resolution)
    bits = _set_failure(
        bits,
        candidate_mechanics.failure_bits != jnp.uint32(0),
        ABILITY_FAILURE_MECHANICS,
    )
    valid = (bits == jnp.uint32(0)) & (world_bits == jnp.uint32(0))
    newly_dead = (
        valid[:, None]
        & roster.active
        & roster.damageable
        & ~roster.dead
        & (health <= 0.0)
    )
    candidate_roster = roster._replace(
        health=health,
        dead=roster.dead | newly_dead,
    )
    candidate_effects = effects._replace(
        combat=effects.combat._replace(
            roster=candidate_roster,
            mechanics=candidate_mechanics,
        ),
        source_generation=source_generation,
    )
    result_effects = _select_tree(valid, candidate_effects, effects)
    result_state = state._replace(
        failure_bits=bits,
        world_failure_bits=world_bits,
    )
    healing_mask = direct_targets & (kind == EVENT_HEAL)[..., None]
    resource_mask = direct_targets & (kind == EVENT_RESOURCE)[..., None]
    force_mask = direct_targets & (kind == EVENT_FORCE)[..., None]
    return (
        result_state,
        result_effects,
        EntityAbilityResolveInfo(
            damage_events=jnp.where(valid, damage_count, 0),
            status_applications=jnp.where(valid, status_count, 0),
            status_clears=jnp.where(
                valid,
                jnp.sum(cleared.astype(jnp.int32), axis=(1, 2)),
                0,
            ),
            healing_targets=jnp.where(
                valid,
                jnp.sum(healing_mask.astype(jnp.int32), axis=(1, 2)),
                0,
            ),
            resource_targets=jnp.where(
                valid,
                jnp.sum(resource_mask.astype(jnp.int32), axis=(1, 2)),
                0,
            ),
            force_targets=jnp.where(
                valid,
                jnp.sum(force_mask.astype(jnp.int32), axis=(1, 2)),
                0,
            ),
            newly_dead=newly_dead,
            failure_bits=bits,
            world_failure_bits=world_bits,
            valid=valid,
        ),
    )


def _pack_damage(events, mask, roster):
    batch = mask.shape[0]
    flat_size = mask.shape[1] * ENTITY_CAPACITY
    flat = mask.reshape((batch, flat_size))
    count = jnp.sum(flat.astype(jnp.int32), axis=1)
    overflow = count > ABILITY_DAMAGE_CAPACITY
    score, index = jax.lax.top_k(
        jnp.where(
            flat,
            flat_size - jnp.arange(flat_size, dtype=jnp.int32)[None, :],
            jnp.int32(-1),
        ),
        ABILITY_DAMAGE_CAPACITY,
    )
    requested = (score >= 0) & ~overflow[:, None]
    source = jnp.broadcast_to(events.source_slot[..., None], mask.shape)
    target = jnp.broadcast_to(
        jnp.arange(ENTITY_CAPACITY, dtype=jnp.int32)[None, None, :],
        mask.shape,
    )
    batch_index = jnp.arange(batch, dtype=jnp.int32)[:, None]
    safe_source = jnp.clip(events.source_slot, 0, ENTITY_CAPACITY - 1)
    source_position = roster.position[batch_index, safe_source]
    source_yaw = roster.yaw_degrees[batch_index, safe_source]
    target_position = roster.position[:, None, :, :]
    angled = _angled_damage_mask(
        events,
        source_position,
        target_position,
        roster.yaw_degrees[:, None, :],
    )
    base_force = _damage_force_velocity(
        events.f32[..., [EF_FORCE_X, EF_FORCE_Y, EF_FORCE_Z]],
        events.f32[..., EF_FORCE_MAGNITUDE],
        events.i32[..., EI_FORCE_DIRECTION_MODE],
        source_position,
        target_position,
        source_yaw,
    )
    angled_force = _damage_force_velocity(
        events.f32[
            ...,
            [
                EF_ANGLED_FORCE_X,
                EF_ANGLED_FORCE_Y,
                EF_ANGLED_FORCE_Z,
            ],
        ],
        events.f32[..., EF_ANGLED_FORCE_MAGNITUDE],
        events.i32[..., EI_FORCE_DIRECTION_MODE],
        source_position,
        target_position,
        source_yaw,
    )
    force = jnp.where(angled[..., None], angled_force, base_force)
    amount = jnp.where(
        angled,
        events.f32[..., EF_ANGLED_DAMAGE, None],
        events.f32[..., EF_DAMAGE, None],
    )

    def scalar(value):
        return _gather_flat(
            jnp.broadcast_to(value[..., None], mask.shape),
            index,
        )

    result = empty_damage_events(batch, ABILITY_DAMAGE_CAPACITY)._replace(
        requested=requested,
        source_entity_id=_gather_flat(source, index),
        target_entity_id=_gather_flat(target, index),
        amount=_gather_flat(amount, index),
        random_percentage=scalar(
            events.f32[..., EF_RANDOM_PERCENTAGE],
        ),
        damage_class=scalar(events.i32[..., EI_DAMAGE_CLASS]),
        cause=scalar(events.i32[..., EI_DAMAGE_CAUSE]),
        knockback_velocity=_gather_flat(force, index),
        force_mode=scalar(events.i32[..., EI_FORCE_MODE]),
        air_resistance=scalar(events.f32[..., EF_AIR_RESISTANCE]),
        air_resistance_max=scalar(events.f32[..., EF_AIR_RESISTANCE_MAX]),
        ground_resistance=scalar(events.f32[..., EF_GROUND_RESISTANCE]),
        ground_resistance_max=scalar(events.f32[..., EF_GROUND_RESISTANCE_MAX]),
        resistance_threshold=scalar(events.f32[..., EF_RESISTANCE_THRESHOLD]),
        resistance_style=scalar(events.i32[..., EI_RESISTANCE_STYLE]),
        on_hit_resource_id=scalar(events.i32[..., EI_ON_HIT_RESOURCE_ID]),
        on_hit_resource_delta=scalar(events.f32[..., EF_ON_HIT_RESOURCE_DELTA]),
    )
    return result, count, overflow


def _angled_damage_mask(
    events,
    source_position,
    target_position,
    target_yaw_degrees,
):
    """Mirror DamageEntityInteraction's target-body angle selection."""

    offset = source_position[..., None, :] - target_position
    bearing = jnp.rad2deg(jnp.arctan2(offset[..., 0], offset[..., 2]))
    relative = _normalize_degrees(bearing + 180.0 - target_yaw_degrees)
    delta = jnp.abs(
        _normalize_degrees(relative - events.f32[..., EF_ANGLED_ANGLE_DEGREES, None])
    )
    enabled = (events.flags & jnp.uint32(EVENT_FLAG_ANGLED_DAMAGE)) != 0
    return enabled[..., None] & (
        delta < events.f32[..., EF_ANGLED_DISTANCE_DEGREES, None]
    )


def _damage_force_velocity(
    direction,
    magnitude,
    direction_mode,
    source_position,
    target_position,
    source_yaw_degrees,
):
    length = jnp.linalg.norm(direction, axis=2)
    local = direction / jnp.maximum(length[..., None], jnp.finfo(jnp.float32).tiny)
    radians = jnp.deg2rad(source_yaw_degrees)
    world_x = local[..., 0] * jnp.cos(radians) + local[..., 2] * jnp.sin(radians)
    world_z = -local[..., 0] * jnp.sin(radians) + local[..., 2] * jnp.cos(radians)
    local_force = (
        jnp.stack((world_x, local[..., 1], world_z), axis=2) * magnitude[..., None]
    )[..., None, :]
    local_force = jnp.broadcast_to(
        local_force,
        source_position.shape[:2] + (target_position.shape[2], 3),
    )
    offset = target_position - source_position[..., None, :]
    point_force = offset / jnp.maximum(
        jnp.linalg.norm(offset, axis=3, keepdims=True),
        jnp.finfo(jnp.float32).tiny,
    )
    point_force = point_force * magnitude[..., None, None]
    point_force = point_force.at[..., 1].set(direction[..., 1, None])
    target_to_source = source_position[..., None, :] - target_position
    target_to_source = target_to_source / jnp.maximum(
        jnp.linalg.norm(target_to_source, axis=3, keepdims=True),
        jnp.finfo(jnp.float32).tiny,
    )
    relative = jnp.stack(
        (
            direction[..., 0] * jnp.cos(radians) + direction[..., 2] * jnp.sin(radians),
            jnp.zeros_like(direction[..., 1]),
            -direction[..., 0] * jnp.sin(radians)
            + direction[..., 2] * jnp.cos(radians),
        ),
        axis=2,
    )
    combined = target_to_source + relative[..., None, :]
    directional_force = jnp.stack(
        (
            combined[..., 0] * magnitude[..., None],
            jnp.broadcast_to(
                direction[..., 1, None],
                combined[..., 1].shape,
            ),
            combined[..., 2] * magnitude[..., None],
        ),
        axis=3,
    )
    result = jnp.where(
        (direction_mode == FORCE_DIRECTION_POINT)[..., None, None],
        point_force,
        local_force,
    )
    return jnp.where(
        (direction_mode == FORCE_DIRECTION_DIRECTIONAL)[..., None, None],
        directional_force,
        result,
    )


def _pack_statuses(events, mask, roster, rules):
    batch = mask.shape[0]
    eligible = jnp.swapaxes(mask, 1, 2)
    count_by_target = jnp.sum(eligible.astype(jnp.int32), axis=2)
    overflow = jnp.any(count_by_target > STATUS_APPLICATION_CAPACITY, axis=1)
    score, query_index = jax.lax.top_k(
        jnp.where(
            eligible,
            mask.shape[1] - jnp.arange(mask.shape[1], dtype=jnp.int32)[None, None, :],
            jnp.int32(-1),
        ),
        STATUS_APPLICATION_CAPACITY,
    )
    requested = (score >= 0) & ~overflow[:, None, None]

    def gather(value):
        expanded = jnp.broadcast_to(value[:, None, :], eligible.shape)
        return jnp.take_along_axis(expanded, query_index, axis=2)

    percent = (events.flags & jnp.uint32(EVENT_FLAG_STATUS_VALUE_PERCENT)) != 0
    healing = jnp.where(
        percent[..., None],
        (
            events.f32[..., EF_STATUS_HEALING, None]
            * roster.max_health[:, None, :]
            / 100.0
        ),
        events.f32[..., EF_STATUS_HEALING, None],
    )
    resource_id = events.i32[..., EI_STATUS_RESOURCE_ID]
    resource_one_hot = jax.nn.one_hot(
        resource_id,
        RESOURCE_COUNT,
        dtype=jnp.float32,
    )
    resource_max = jnp.einsum(
        "ber,bqr->bqe",
        rules.resource_maximum,
        resource_one_hot,
    )
    resource = jnp.where(
        percent[..., None],
        (events.f32[..., EF_STATUS_RESOURCE_DELTA, None] * resource_max / 100.0),
        events.f32[..., EF_STATUS_RESOURCE_DELTA, None],
    )
    applications = empty_status_applications(
        batch, entity_count=ENTITY_CAPACITY
    )._replace(
        requested=requested,
        effect_id=gather(events.i32[..., EI_STATUS_ID]),
        source_entity_id=gather(events.source_slot),
        duration_seconds=gather(events.f32[..., EF_STATUS_DURATION_SECONDS]),
        cycle_cooldown_seconds=gather(events.f32[..., EF_STATUS_COOLDOWN_SECONDS]),
        damage_per_cycle=gather(events.f32[..., EF_STATUS_DAMAGE]),
        damage_cause=gather(events.i32[..., EI_STATUS_DAMAGE_CAUSE]),
        healing_per_cycle=jnp.take_along_axis(
            jnp.swapaxes(healing, 1, 2),
            query_index,
            axis=2,
        ),
        resource_id=gather(resource_id),
        resource_delta_per_cycle=jnp.take_along_axis(
            jnp.swapaxes(resource, 1, 2),
            query_index,
            axis=2,
        ),
        speed_multiplier=gather(events.f32[..., EF_STATUS_SPEED_MULTIPLIER]),
        flags=gather(events.flags & jnp.uint32(EVENT_STATUS_FLAG_MASK)),
        overlap_mode=gather(events.i32[..., EI_STATUS_OVERLAP_MODE]),
    )
    return (
        applications,
        jnp.sum(count_by_target, axis=1),
        overflow,
    )


def _apply_heals(health, maximum, events, targets, valid):
    mask = targets & (events.kind == EVENT_HEAL)[..., None] & valid[:, None, None]
    amount = events.f32[..., EF_STATUS_HEALING]
    percent = (events.flags & jnp.uint32(EVENT_FLAG_VALUE_PERCENT)) != 0
    value = jnp.where(
        percent[..., None],
        amount[..., None] * maximum[:, None, :] / 100.0,
        amount[..., None],
    )
    return jnp.minimum(
        maximum,
        health + jnp.sum(jnp.where(mask, value, 0.0), axis=1),
    )


def _apply_resources(mechanics, rules, events, targets, valid):
    mask = targets & (events.kind == EVENT_RESOURCE)[..., None] & valid[:, None, None]
    resource_id = events.i32[..., EI_STATUS_RESOURCE_ID]
    one_hot = jax.nn.one_hot(
        resource_id,
        RESOURCE_COUNT,
        dtype=jnp.float32,
    )
    amount = events.f32[..., EF_STATUS_RESOURCE_DELTA]
    percent = (events.flags & jnp.uint32(EVENT_FLAG_VALUE_PERCENT)) != 0
    value = jnp.where(
        percent[..., None, None],
        (amount[..., None, None] * rules.resource_maximum[:, None, :, :] / 100.0),
        amount[..., None, None],
    )
    delta = jnp.sum(
        mask[..., None] * one_hot[:, :, None, :] * value,
        axis=1,
    )
    resources = jnp.clip(
        mechanics.resources + delta,
        rules.resource_minimum,
        rules.resource_maximum,
    )
    stamina_full = (
        resources[..., RESOURCE_STAMINA]
        >= rules.resource_maximum[..., RESOURCE_STAMINA]
    )
    return mechanics._replace(
        resources=resources,
        stamina_broken=mechanics.stamina_broken & ~stamina_full,
    )


def _apply_forces(mechanics, events, targets, source_yaw, valid):
    mask = targets & (events.kind == EVENT_FORCE)[..., None] & valid[:, None, None]
    velocity = adjust_vertical_force(
        _local_force_velocity(events.f32, source_yaw),
        events.f32,
        events.i32,
    )
    inputs = (
        jnp.swapaxes(mask, 0, 1),
        jnp.swapaxes(velocity, 0, 1),
        jnp.swapaxes(events.f32, 0, 1),
        jnp.swapaxes(events.i32, 0, 1),
    )
    carry = (
        mechanics.applied_velocity,
        mechanics.applied_velocity_can_clear,
        mechanics.applied_air_resistance,
        mechanics.applied_air_resistance_max,
        mechanics.applied_ground_resistance,
        mechanics.applied_ground_resistance_max,
        mechanics.applied_resistance_threshold,
        mechanics.applied_resistance_style,
    )

    def apply_one(current, values):
        (
            applied,
            can_clear,
            air,
            air_max,
            ground,
            ground_max,
            threshold,
            style,
        ) = current
        enabled, event_velocity, f32, i32 = values
        next_velocity = jnp.where(
            i32[:, EI_FORCE_MODE, None, None] == FORCE_SET,
            event_velocity[:, None, :],
            applied + event_velocity[:, None, :],
        )

        def write(current_value, new_value):
            shaped = enabled.reshape(
                enabled.shape + (1,) * (current_value.ndim - enabled.ndim)
            )
            return jnp.where(shaped, new_value, current_value)

        return (
            write(applied, next_velocity),
            write(
                can_clear,
                jnp.zeros_like(can_clear, dtype=jnp.bool_),
            ),
            write(air, f32[:, EF_AIR_RESISTANCE, None]),
            write(
                air_max,
                f32[:, EF_AIR_RESISTANCE_MAX, None],
            ),
            write(ground, f32[:, EF_GROUND_RESISTANCE, None]),
            write(
                ground_max,
                f32[:, EF_GROUND_RESISTANCE_MAX, None],
            ),
            write(
                threshold,
                f32[:, EF_RESISTANCE_THRESHOLD, None],
            ),
            write(style, i32[:, EI_RESISTANCE_STYLE, None]),
        ), None

    result, _ = jax.lax.scan(apply_one, carry, inputs)
    return mechanics._replace(
        applied_velocity=result[0],
        applied_velocity_can_clear=result[1],
        applied_air_resistance=result[2],
        applied_air_resistance_max=result[3],
        applied_ground_resistance=result[4],
        applied_ground_resistance_max=result[5],
        applied_resistance_threshold=result[6],
        applied_resistance_style=result[7],
    )


def _track_status_sources(
    previous_generation,
    before,
    after,
    roster,
):
    changed = after.active & (
        ~before.active
        | (after.effect_id != before.effect_id)
        | (after.source_entity_id != before.source_entity_id)
    )
    source = jnp.clip(after.source_entity_id, 0, ENTITY_CAPACITY - 1)
    generation = roster.generation[jnp.arange(source.shape[0])[:, None, None], source]
    return jnp.where(
        ~after.active,
        jnp.uint32(0),
        jnp.where(
            changed & (after.source_entity_id >= 0),
            generation,
            previous_generation,
        ),
    )


def _local_force_velocity(f32, yaw_degrees):
    direction = f32[..., [EF_FORCE_X, EF_FORCE_Y, EF_FORCE_Z]]
    length = jnp.linalg.norm(direction, axis=2)
    normalized = direction / jnp.maximum(length[..., None], jnp.finfo(jnp.float32).tiny)
    radians = jnp.deg2rad(yaw_degrees)
    world_x = normalized[..., 0] * jnp.cos(radians) + normalized[..., 2] * jnp.sin(
        radians
    )
    world_z = -normalized[..., 0] * jnp.sin(radians) + normalized[..., 2] * jnp.cos(
        radians
    )
    return (
        jnp.stack((world_x, normalized[..., 1], world_z), axis=2)
        * f32[..., EF_FORCE_MAGNITUDE, None]
    )


def _gather_flat(value, index):
    flat = value.reshape((value.shape[0], -1) + value.shape[3:])
    shape = index.shape + (1,) * (flat.ndim - 2)
    gather = jnp.broadcast_to(index.reshape(shape), index.shape + flat.shape[2:])
    return jnp.take_along_axis(flat, gather, axis=1)


def _normalize_degrees(value):
    return jnp.mod(value + 180.0, 360.0) - 180.0


def _set_failure(bits, mask, code):
    return jnp.where(mask, bits | jnp.uint32(code), bits)


def _select_tree(mask, selected, fallback):
    return jax.tree_util.tree_map(
        lambda yes, no: jnp.where(
            mask.reshape((mask.shape[0],) + (1,) * (yes.ndim - 1)),
            yes,
            no,
        ),
        selected,
        fallback,
    )
