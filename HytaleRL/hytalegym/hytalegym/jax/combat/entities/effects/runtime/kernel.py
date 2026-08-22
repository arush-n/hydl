"""Compiled periodic effect transition for the 32-slot entity roster."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.entities import ENTITY_CAPACITY
from hytalegym.jax.combat.entities.effects.schema.contract import (
    ENTITY_EFFECT_DAMAGE_CAPACITY,
    ENTITY_EFFECT_FAILURE_DAMAGE_OVERFLOW,
    ENTITY_EFFECT_FAILURE_INVALID_DT,
    ENTITY_EFFECT_FAILURE_INVALID_STATE,
    ENTITY_EFFECT_FAILURE_MECHANICS,
    ENTITY_EFFECT_FAILURE_STALE_SOURCE,
    ENTITY_EFFECT_FAILURE_UPSTREAM,
    ENTITY_EFFECT_MAX_DT_SECONDS,
)
from hytalegym.jax.combat.entities.effects.schema.types import (
    EntityEffectState,
    EntityEffectTickInfo,
)
from hytalegym.jax.combat.entities.effects.schema.validation import (
    invalid_effect_rows,
    stale_effect_source_rows,
    validate_effect_layout,
)
from hytalegym.jax.combat.mechanics import (
    STATUS_CAPACITY,
    CombatMechanicsRules,
    apply_damage_forces,
    empty_damage_events,
    resolve_damage_events,
    tick_resources,
    tick_statuses,
)


def tick_entity_effects(
    state: EntityEffectState,
    dt_seconds,
    rules: CombatMechanicsRules,
) -> tuple[EntityEffectState, EntityEffectTickInfo]:
    """Advance statuses and commit heal/damage/resource outputs atomically."""

    validate_effect_layout(state)
    batch = state.combat.roster.active.shape[0]
    if rules.resource_maximum.shape[:2] != (batch, ENTITY_CAPACITY):
        raise ValueError(
            f"mechanics rules must have shape [{batch},{ENTITY_CAPACITY},...]"
        )
    dt = _batch_dt(dt_seconds, batch)
    bits = state.failure_bits
    bits = _set_failure(
        bits,
        invalid_effect_rows(state),
        ENTITY_EFFECT_FAILURE_INVALID_STATE,
    )
    bits = _set_failure(
        bits,
        ~jnp.isfinite(dt) | (dt <= 0.0) | (dt > ENTITY_EFFECT_MAX_DT_SECONDS),
        ENTITY_EFFECT_FAILURE_INVALID_DT,
    )
    upstream = (state.combat.roster.failure_bits != jnp.uint32(0)) | (
        state.combat.mechanics.failure_bits != jnp.uint32(0)
    )
    bits = _set_failure(
        bits,
        upstream,
        ENTITY_EFFECT_FAILURE_UPSTREAM,
    )
    bits = _set_failure(
        bits,
        stale_effect_source_rows(state),
        ENTITY_EFFECT_FAILURE_STALE_SOURCE,
    )
    pre_valid = bits == jnp.uint32(0)
    mechanics = tick_resources(
        state.combat.mechanics,
        rules,
        jnp.where(pre_valid, dt, 0.0),
        jnp.ones_like(
            state.combat.mechanics.resources,
            dtype=jnp.bool_,
        ),
    )
    mechanics, tick = tick_statuses(
        mechanics,
        jnp.where(pre_valid, dt, 0.0),
        rules,
    )
    roster = state.combat.roster
    living = roster.active & roster.damageable & ~roster.dead
    mechanics = _restore_nonliving_mechanics(
        mechanics,
        state.combat.mechanics,
        living,
    )
    fires = tick.requested & living[..., None] & pre_valid[:, None, None]
    healing = jnp.sum(
        jnp.where(fires, tick.healing, 0.0),
        axis=2,
    )
    healed_health = jnp.where(
        living,
        jnp.minimum(roster.max_health, roster.health + healing),
        roster.health,
    )
    healing_applied = healed_health - roster.health

    roster_invulnerable = roster.invulnerable[..., None]
    raw_damage = fires & (tick.damage > 0.0)
    damage_mask = raw_damage & ~roster_invulnerable
    events, event_count, overflow = _pack_status_damage(
        tick,
        damage_mask,
        pre_valid,
    )
    bits = _set_failure(
        bits,
        overflow,
        ENTITY_EFFECT_FAILURE_DAMAGE_OVERFLOW,
    )
    damage_valid = bits == jnp.uint32(0)
    events = events._replace(requested=events.requested & damage_valid[:, None])
    mechanics, health, resolution = resolve_damage_events(
        mechanics,
        healed_health,
        roster.position,
        roster.yaw_degrees,
        events,
        rules,
    )
    mechanics = apply_damage_forces(mechanics, resolution)
    bits = _set_failure(
        bits,
        mechanics.failure_bits != jnp.uint32(0),
        ENTITY_EFFECT_FAILURE_MECHANICS,
    )
    valid = bits == jnp.uint32(0)
    damage_applied = _sum_resolution_by_target(
        resolution.applied_damage,
        resolution.target_entity_id,
    )
    blocked = _sum_resolution_by_target(
        resolution.blocked.astype(jnp.int32),
        resolution.target_entity_id,
    )
    invulnerable = _sum_resolution_by_target(
        resolution.invulnerable.astype(jnp.int32),
        resolution.target_entity_id,
    )
    invulnerable += jnp.sum(
        (raw_damage & roster_invulnerable).astype(jnp.int32),
        axis=2,
    )
    newly_dead = roster.active & roster.damageable & ~roster.dead & (health <= 0.0)
    candidate_combat = state.combat._replace(
        roster=roster._replace(
            health=health,
            dead=roster.dead | newly_dead,
        ),
        mechanics=mechanics,
    )
    source_generation = jnp.where(
        mechanics.statuses.active,
        state.source_generation,
        jnp.uint32(0),
    )
    result = EntityEffectState(
        combat=_select_tree(valid, candidate_combat, state.combat),
        source_generation=jnp.where(
            valid[:, None, None],
            source_generation,
            state.source_generation,
        ),
        capability_bits=state.capability_bits,
        failure_bits=bits,
    )
    info_mask = valid[:, None]
    status_cycles = jnp.sum(fires.astype(jnp.int32), axis=2)
    flags = jnp.where(
        living,
        tick.aggregate_flags,
        jnp.uint32(0),
    )
    speed = jnp.where(living, tick.speed_multiplier, 1.0)
    active_count = jnp.sum(
        mechanics.statuses.active.astype(jnp.int32),
        axis=2,
    )
    return result, EntityEffectTickInfo(
        status_cycles=jnp.where(info_mask, status_cycles, 0),
        damage_event_count=jnp.where(valid, event_count, 0),
        damage_applied=jnp.where(info_mask, damage_applied, 0.0),
        healing_applied=jnp.where(info_mask, healing_applied, 0.0),
        blocked_hits=jnp.where(info_mask, blocked, 0),
        invulnerable_hits=jnp.where(info_mask, invulnerable, 0),
        newly_dead=newly_dead & info_mask,
        aggregate_flags=jnp.where(
            info_mask,
            flags,
            jnp.uint32(0),
        ),
        speed_multiplier=jnp.where(info_mask, speed, 1.0),
        active_status_count=jnp.where(info_mask, active_count, 0),
        failure_bits=bits,
        valid=valid,
    )


def _pack_status_damage(tick, mask, row_valid):
    batch = mask.shape[0]
    flat_size = ENTITY_CAPACITY * STATUS_CAPACITY
    flat_mask = mask.reshape((batch, flat_size))
    count = jnp.sum(flat_mask.astype(jnp.int32), axis=1)
    overflow = count > ENTITY_EFFECT_DAMAGE_CAPACITY
    scores = jnp.where(
        flat_mask,
        flat_size - jnp.arange(flat_size, dtype=jnp.int32)[None, :],
        jnp.int32(-1),
    )
    score, index = jax.lax.top_k(
        scores,
        ENTITY_EFFECT_DAMAGE_CAPACITY,
    )
    requested = (score >= 0) & row_valid[:, None] & ~overflow[:, None]
    events = empty_damage_events(
        batch,
        ENTITY_EFFECT_DAMAGE_CAPACITY,
    )._replace(
        requested=requested,
        source_entity_id=_gather_flat(tick.source_entity_id, index),
        target_entity_id=_gather_flat(tick.target_entity_id, index),
        amount=jnp.where(
            requested,
            _gather_flat(tick.damage, index),
            0.0,
        ),
        cause=_gather_flat(tick.damage_cause, index),
    )
    return events, jnp.where(~overflow, count, 0), overflow


def _restore_nonliving_mechanics(candidate, original, living):
    entity_mask = living[..., None]
    return candidate._replace(
        resources=jnp.where(
            entity_mask,
            candidate.resources,
            original.resources,
        ),
        resource_regen_clock=jnp.where(
            entity_mask,
            candidate.resource_regen_clock,
            original.resource_regen_clock,
        ),
        stamina_broken=jnp.where(
            living,
            candidate.stamina_broken,
            original.stamina_broken,
        ),
        guard_held=jnp.where(
            living,
            candidate.guard_held,
            original.guard_held,
        ),
        guard_active=jnp.where(
            living,
            candidate.guard_active,
            original.guard_active,
        ),
        guard_windup_elapsed_seconds=jnp.where(
            living,
            candidate.guard_windup_elapsed_seconds,
            original.guard_windup_elapsed_seconds,
        ),
        control_immunity=jnp.where(
            living,
            candidate.control_immunity,
            original.control_immunity,
        ),
        control_immunity_regen_clock=jnp.where(
            living,
            candidate.control_immunity_regen_clock,
            original.control_immunity_regen_clock,
        ),
        stamina_regen_delay_seconds=jnp.where(
            living,
            candidate.stamina_regen_delay_seconds,
            original.stamina_regen_delay_seconds,
        ),
    )


def _sum_resolution_by_target(value, target):
    selected = jax.nn.one_hot(
        jnp.clip(target, 0, ENTITY_CAPACITY - 1),
        ENTITY_CAPACITY,
        dtype=value.dtype,
    )
    return jnp.sum(selected * value[..., None], axis=1)


def _gather_flat(value, index):
    flat = value.reshape((value.shape[0], -1))
    return jnp.take_along_axis(flat, index, axis=1)


def _batch_dt(value, batch):
    result = jnp.asarray(value, dtype=jnp.float32)
    if result.ndim == 0:
        result = jnp.broadcast_to(result, (batch,))
    if result.shape != (batch,):
        raise ValueError(f"dt_seconds must be scalar or [{batch}]")
    return result


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
