"""Bounded multi-target status packing and application."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.entities.schema.contract import (
    ENTITY_CAPACITY,
    ENTITY_FAILURE_COMBAT_MECHANICS,
    ENTITY_FAILURE_INVALID_COMMAND,
    ENTITY_FAILURE_STATUS_OVERFLOW,
    SELECTOR_CAPACITY,
)
from hytalegym.jax.combat.entities.schema.types import (
    EntityCombatState,
    EntityStatusPayloads,
    EntityTargetSelection,
    PackedEntityStatuses,
)
from hytalegym.jax.combat.entities.schema.validation import (
    validate_roster_layout,
    validate_selection_layout,
    validate_status_payload_layout,
)
from hytalegym.jax.combat.mechanics import (
    DAMAGE_COUNT,
    RESOURCE_COUNT,
    STATUS_APPLICATION_CAPACITY,
    STATUS_OVERLAP_IGNORE,
    STATUS_OVERLAP_OVERWRITE,
    apply_statuses,
    empty_status_applications,
)
def pack_entity_statuses(
    roster,
    selection: EntityTargetSelection,
    payloads: EntityStatusPayloads,
) -> PackedEntityStatuses:
    """Pack at most four query-ordered applications into each entity."""

    validate_roster_layout(roster)
    validate_status_payload_layout(roster, payloads)
    batch = roster.active.shape[0]
    validate_selection_layout(batch, selection)
    requested = payloads.requested
    valid_payload = (
        (payloads.effect_id > 0)
        & jnp.isfinite(payloads.duration_seconds)
        & (payloads.duration_seconds > 0.0)
        & jnp.isfinite(payloads.cycle_cooldown_seconds)
        & (payloads.cycle_cooldown_seconds >= 0.0)
        & jnp.isfinite(payloads.damage_per_cycle)
        & (payloads.damage_per_cycle >= 0.0)
        & (payloads.damage_cause >= 0)
        & (payloads.damage_cause < DAMAGE_COUNT)
        & jnp.isfinite(payloads.healing_per_cycle)
        & (payloads.healing_per_cycle >= 0.0)
        & (payloads.resource_id >= -1)
        & (payloads.resource_id < RESOURCE_COUNT)
        & jnp.isfinite(payloads.resource_delta_per_cycle)
        & jnp.isfinite(payloads.speed_multiplier)
        & (payloads.speed_multiplier > 0.0)
        & (payloads.overlap_mode >= STATUS_OVERLAP_IGNORE)
        & (payloads.overlap_mode <= STATUS_OVERLAP_OVERWRITE)
        & selection.valid
    )
    invalid = jnp.any(requested & ~valid_payload, axis=1)
    eligible = (
        selection.target_mask & requested[..., None]
    ).transpose((0, 2, 1))
    count = jnp.sum(eligible.astype(jnp.int32), axis=2)
    overflow = jnp.any(count > STATUS_APPLICATION_CAPACITY, axis=1)
    bits = selection.row_failure_bits
    bits = _set_failure(
        bits,
        invalid,
        ENTITY_FAILURE_INVALID_COMMAND,
    )
    bits = _set_failure(
        bits,
        overflow,
        ENTITY_FAILURE_STATUS_OVERFLOW,
    )
    row_valid = bits == jnp.uint32(0)
    score = jnp.where(
        eligible,
        SELECTOR_CAPACITY
        - jnp.arange(SELECTOR_CAPACITY, dtype=jnp.int32)[None, None, :],
        jnp.int32(-1),
    )
    score, query_index = jax.lax.top_k(
        score,
        STATUS_APPLICATION_CAPACITY,
    )
    packed_requested = (score >= 0) & row_valid[:, None, None]

    def gather(value):
        expanded = jnp.broadcast_to(
            value[:, None, :],
            (batch, ENTITY_CAPACITY, SELECTOR_CAPACITY),
        )
        return jnp.take_along_axis(expanded, query_index, axis=2)

    applications = empty_status_applications(
        batch,
        entity_count=ENTITY_CAPACITY,
    )._replace(
        requested=packed_requested,
        effect_id=gather(payloads.effect_id),
        source_entity_id=gather(selection.source_slot),
        duration_seconds=gather(payloads.duration_seconds),
        cycle_cooldown_seconds=gather(
            payloads.cycle_cooldown_seconds
        ),
        damage_per_cycle=gather(payloads.damage_per_cycle),
        damage_cause=gather(payloads.damage_cause),
        healing_per_cycle=gather(payloads.healing_per_cycle),
        resource_id=gather(payloads.resource_id),
        resource_delta_per_cycle=gather(
            payloads.resource_delta_per_cycle
        ),
        speed_multiplier=gather(payloads.speed_multiplier),
        flags=gather(payloads.flags),
        overlap_mode=gather(payloads.overlap_mode),
    )
    return PackedEntityStatuses(
        applications=applications,
        application_count=jnp.where(
            row_valid,
            jnp.sum(count, axis=1),
            jnp.int32(0),
        ),
        failure_bits=bits,
        valid=row_valid,
    )


def apply_selected_entity_statuses(
    state: EntityCombatState,
    selection: EntityTargetSelection,
    payloads: EntityStatusPayloads,
) -> tuple[EntityCombatState, PackedEntityStatuses]:
    packed = pack_entity_statuses(state.roster, selection, payloads)
    mechanics = apply_statuses(
        state.mechanics,
        packed.applications,
    )
    mechanics_failed = mechanics.failure_bits != jnp.uint32(0)
    bits = _set_failure(
        state.roster.failure_bits | packed.failure_bits,
        mechanics_failed,
        ENTITY_FAILURE_COMBAT_MECHANICS,
    )
    valid = bits == jnp.uint32(0)
    roster = state.roster._replace(failure_bits=bits)
    mechanics = _select_tree(valid, mechanics, state.mechanics)
    mechanics = mechanics._replace(
        failure_bits=jnp.where(
            valid,
            mechanics.failure_bits,
            state.mechanics.failure_bits,
        )
    )
    return EntityCombatState(roster, mechanics), packed._replace(
        application_count=jnp.where(
            valid,
            packed.application_count,
            jnp.int32(0),
        ),
        failure_bits=bits,
        valid=valid,
    )


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
