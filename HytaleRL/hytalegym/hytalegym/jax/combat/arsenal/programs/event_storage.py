"""Storage-neutral authored-event views for Arsenal programs."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def ability_event_bank(loadout) -> tuple[jax.Array, ...]:
    """Return an ability-major view for packed or authored event storage."""

    if loadout.event_mask.ndim == 4:
        return (
            loadout.event_mask,
            loadout.event_time_seconds,
            loadout.event_kind,
            loadout.event_f32,
            loadout.event_i32,
            loadout.event_flags,
        )
    if loadout.event_mask.ndim != 3:
        raise ValueError("loadout event storage must be packed or ability-major")
    ability_capacity = loadout.ability_mask.shape[2]
    event_capacity = loadout.event_mask.shape[2]
    event_index = jnp.arange(event_capacity)[None, None, None, :]
    start = loadout.ability_event_start[..., None]
    end = start + loadout.ability_event_count[..., None]
    mask = (
        loadout.event_mask[:, :, None, :]
        & (event_index >= start)
        & (event_index < end)
    )

    def expand(value: jax.Array) -> jax.Array:
        return jnp.broadcast_to(
            value[:, :, None, ...],
            value.shape[:2] + (ability_capacity,) + value.shape[2:],
        )

    return (
        mask,
        expand(loadout.event_time_seconds),
        expand(loadout.event_kind),
        expand(loadout.event_f32),
        expand(loadout.event_i32),
        expand(loadout.event_flags),
    )


def selected_ability_events(
    loadout,
    entity_index: jax.Array,
    ability_slot: jax.Array,
) -> tuple[jax.Array, ...]:
    """Gather one entity/ability program per batch without rectangular expansion."""

    batch = loadout.ability_mask.shape[0]
    if entity_index.shape != (batch,) or ability_slot.shape != (batch,):
        raise ValueError("entity_index and ability_slot must have shape [B]")
    row = jnp.arange(batch, dtype=jnp.int32)
    if loadout.event_mask.ndim == 4:
        return (
            loadout.event_mask[row, entity_index, ability_slot],
            loadout.event_time_seconds[row, entity_index, ability_slot],
            loadout.event_f32[row, entity_index, ability_slot],
            loadout.event_flags[row, entity_index, ability_slot],
        )
    if loadout.event_mask.ndim != 3:
        raise ValueError("loadout event storage must be packed or ability-major")

    local_mask = loadout.ability_event_mask[row, entity_index, ability_slot]
    local_index = jnp.arange(local_mask.shape[1], dtype=jnp.int32)[None, :]
    start = loadout.ability_event_start[row, entity_index, ability_slot]
    count = loadout.ability_event_count[row, entity_index, ability_slot]
    packed_index = jnp.clip(
        start[:, None] + local_index,
        jnp.int32(0),
        jnp.int32(loadout.event_mask.shape[2] - 1),
    )
    packed_row = row[:, None]
    packed_entity = entity_index[:, None]
    mask = (
        loadout.event_mask[packed_row, packed_entity, packed_index]
        & local_mask
        & (local_index < count[:, None])
    )
    return (
        mask,
        loadout.event_time_seconds[packed_row, packed_entity, packed_index],
        loadout.event_f32[packed_row, packed_entity, packed_index],
        loadout.event_flags[packed_row, packed_entity, packed_index],
    )


__all__ = ["ability_event_bank", "selected_ability_events"]
