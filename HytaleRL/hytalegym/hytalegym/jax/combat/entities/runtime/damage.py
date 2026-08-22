"""Bounded multi-target damage packing and shared mechanics resolution."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.entities.schema.contract import (
    ENTITY_CAPACITY,
    ENTITY_DAMAGE_CAPACITY,
    ENTITY_FAILURE_COMBAT_MECHANICS,
    ENTITY_FAILURE_DAMAGE_OVERFLOW,
    ENTITY_FAILURE_INVALID_COMMAND,
    SELECTOR_CAPACITY,
)
from hytalegym.jax.combat.entities.schema.types import (
    EntityCombatState,
    EntityDamageInfo,
    EntityDamagePayloads,
    EntityTargetSelection,
    PackedEntityDamage,
)
from hytalegym.jax.combat.entities.schema.validation import (
    validate_damage_payload_layout,
    validate_roster_layout,
    validate_selection_layout,
)
from hytalegym.jax.combat.mechanics import (
    DAMAGE_CLASS_COUNT,
    DAMAGE_COUNT,
    RESOURCE_COUNT,
    CombatMechanicsRules,
    apply_damage_forces,
    empty_damage_events,
    resolve_damage_events,
)


def pack_entity_damage(
    roster,
    selection: EntityTargetSelection,
    payloads: EntityDamagePayloads,
) -> PackedEntityDamage:
    """Pack query-major, slot-major targets into 64 ordered damage events."""

    validate_roster_layout(roster)
    validate_damage_payload_layout(roster, payloads)
    validate_selection_layout(roster.active.shape[0], selection)
    requested = payloads.requested
    force = payloads.knockback_velocity
    force_present = jnp.any(force != 0.0, axis=(2, 3))
    force_valid = (
        ((payloads.force_mode == 0) | (payloads.force_mode == 1))
        & jnp.isfinite(payloads.air_resistance)
        & jnp.isfinite(payloads.air_resistance_max)
        & jnp.isfinite(payloads.ground_resistance)
        & jnp.isfinite(payloads.ground_resistance_max)
        & jnp.isfinite(payloads.resistance_threshold)
        & (payloads.air_resistance >= 0.0)
        & (payloads.air_resistance <= 1.0)
        & (payloads.air_resistance_max >= 0.0)
        & (payloads.air_resistance_max <= 1.0)
        & (payloads.ground_resistance >= 0.0)
        & (payloads.ground_resistance <= 1.0)
        & (payloads.ground_resistance_max >= 0.0)
        & (payloads.ground_resistance_max <= 1.0)
        & (payloads.resistance_threshold > 0.0)
        & (
            (payloads.resistance_style == 0)
            | (payloads.resistance_style == 1)
        )
    )
    payload_valid = (
        jnp.isfinite(payloads.amount)
        & (payloads.amount >= 0.0)
        & jnp.isfinite(payloads.random_percentage)
        & (payloads.random_percentage >= 0.0)
        & (payloads.damage_class >= 0)
        & (payloads.damage_class < DAMAGE_CLASS_COUNT)
        & (payloads.cause >= 0)
        & (payloads.cause < DAMAGE_COUNT)
        & jnp.all(jnp.isfinite(force), axis=(2, 3))
        & (~force_present | force_valid)
        & (payloads.on_hit_resource_id >= -1)
        & (payloads.on_hit_resource_id < RESOURCE_COUNT)
        & jnp.isfinite(payloads.on_hit_resource_delta)
        & (
            (payloads.on_hit_resource_id < 0)
            | (selection.source_slot >= 0)
        )
    )
    invalid_payload = jnp.any(
        requested & (~payload_valid | ~selection.valid),
        axis=1,
    )
    mask = selection.target_mask & requested[..., None]
    batch = mask.shape[0]
    flat_mask = mask.reshape((batch, -1))
    count = jnp.sum(flat_mask.astype(jnp.int32), axis=1)
    overflow = count > ENTITY_DAMAGE_CAPACITY
    bits = selection.row_failure_bits
    bits = _set_failure(
        bits,
        invalid_payload,
        ENTITY_FAILURE_INVALID_COMMAND,
    )
    bits = _set_failure(
        bits,
        overflow,
        ENTITY_FAILURE_DAMAGE_OVERFLOW,
    )
    row_valid = bits == jnp.uint32(0)
    flat_size = SELECTOR_CAPACITY * ENTITY_CAPACITY
    scores = jnp.where(
        flat_mask,
        flat_size
        - jnp.arange(flat_size, dtype=jnp.int32)[None, :],
        jnp.int32(-1),
    )
    score, index = jax.lax.top_k(scores, ENTITY_DAMAGE_CAPACITY)
    packed_requested = (score >= 0) & row_valid[:, None]
    target = jnp.broadcast_to(
        jnp.arange(ENTITY_CAPACITY, dtype=jnp.int32)[None, None, :],
        mask.shape,
    )
    source = jnp.broadcast_to(
        selection.source_slot[..., None],
        mask.shape,
    )

    def scalar(value):
        expanded = jnp.broadcast_to(value[..., None], mask.shape)
        return _gather_flat(expanded, index)

    events = empty_damage_events(batch, ENTITY_DAMAGE_CAPACITY)._replace(
        requested=packed_requested,
        source_entity_id=_gather_flat(source, index),
        target_entity_id=_gather_flat(target, index),
        amount=scalar(payloads.amount),
        random_percentage=scalar(payloads.random_percentage),
        damage_class=scalar(payloads.damage_class),
        cause=scalar(payloads.cause),
        knockback_velocity=_gather_flat(force, index),
        force_mode=scalar(payloads.force_mode),
        air_resistance=scalar(payloads.air_resistance),
        air_resistance_max=scalar(payloads.air_resistance_max),
        ground_resistance=scalar(payloads.ground_resistance),
        ground_resistance_max=scalar(payloads.ground_resistance_max),
        resistance_threshold=scalar(payloads.resistance_threshold),
        resistance_style=scalar(payloads.resistance_style),
        dampen_y=scalar(payloads.dampen_y),
        on_hit_resource_id=scalar(payloads.on_hit_resource_id),
        on_hit_resource_delta=scalar(payloads.on_hit_resource_delta),
    )
    return PackedEntityDamage(
        events=events,
        event_count=jnp.where(
            row_valid,
            count,
            jnp.int32(0),
        ),
        failure_bits=bits,
        valid=row_valid,
    )


def apply_selected_entity_damage(
    state: EntityCombatState,
    selection: EntityTargetSelection,
    payloads: EntityDamagePayloads,
    rules: CombatMechanicsRules,
) -> tuple[EntityCombatState, EntityDamageInfo]:
    """Resolve selected targets through the shared guard/dodge/status kernel."""

    packed = pack_entity_damage(state.roster, selection, payloads)
    original = state
    mechanics_failed = state.mechanics.failure_bits != jnp.uint32(0)
    bits = _set_failure(
        state.roster.failure_bits | packed.failure_bits,
        mechanics_failed,
        ENTITY_FAILURE_COMBAT_MECHANICS,
    )
    pre_valid = bits == jnp.uint32(0)
    events = packed.events._replace(
        requested=packed.events.requested & pre_valid[:, None]
    )
    mechanics, health, resolution = resolve_damage_events(
        state.mechanics,
        state.roster.health,
        state.roster.position,
        state.roster.yaw_degrees,
        events,
        rules,
    )
    mechanics = apply_damage_forces(mechanics, resolution)
    bits = _set_failure(
        bits,
        mechanics.failure_bits != jnp.uint32(0),
        ENTITY_FAILURE_COMBAT_MECHANICS,
    )
    final_valid = bits == jnp.uint32(0)
    newly_dead = (
        final_valid[:, None]
        & state.roster.active
        & state.roster.damageable
        & ~state.roster.dead
        & (health <= 0.0)
    )
    candidate_roster = state.roster._replace(
        health=health,
        dead=state.roster.dead | newly_dead,
        failure_bits=bits,
    )
    roster = _select_tree(
        final_valid,
        candidate_roster,
        original.roster,
    )._replace(failure_bits=bits)
    selected_mechanics = _select_tree(
        final_valid,
        mechanics,
        original.mechanics,
    )
    selected_mechanics = selected_mechanics._replace(
        failure_bits=mechanics.failure_bits
    )
    return (
        EntityCombatState(roster=roster, mechanics=selected_mechanics),
        EntityDamageInfo(
            resolution=resolution,
            event_count=jnp.where(
                final_valid,
                packed.event_count,
                jnp.int32(0),
            ),
            newly_dead=newly_dead,
            failure_bits=bits,
            valid=final_valid,
        ),
    )


def _gather_flat(value: jax.Array, index: jax.Array) -> jax.Array:
    flat = value.reshape((value.shape[0], -1) + value.shape[3:])
    trailing = (1,) * (flat.ndim - 2)
    gather_index = index.reshape(index.shape + trailing)
    gather_index = jnp.broadcast_to(
        gather_index,
        index.shape + flat.shape[2:],
    )
    return jnp.take_along_axis(flat, gather_index, axis=1)


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
