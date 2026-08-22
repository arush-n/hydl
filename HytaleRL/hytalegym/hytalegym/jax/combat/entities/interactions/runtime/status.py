"""Small fixed-shape status queries and clears used by interaction chains."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def target_has_effect(status, target_slot, effect_id):
    target = jnp.clip(target_slot, 0, status.active.shape[1] - 1)
    active = _gather_entity(status.active, target)
    identifiers = _gather_entity(status.effect_id, target)
    return jnp.any(
        active & (identifiers == jnp.asarray(effect_id, dtype=jnp.int32)),
        axis=1,
    )


def clear_target_effects(status, target_slot, effect_ids, requested):
    """Remove named gameplay effects from one target per batch row."""

    target = jnp.clip(target_slot, 0, status.active.shape[1] - 1)
    target_mask = jax.nn.one_hot(
        target,
        status.active.shape[1],
        dtype=jnp.bool_,
    )[..., None]
    identifiers = jnp.asarray(effect_ids, dtype=jnp.int32)
    matches = jnp.any(
        status.effect_id[..., None] == identifiers,
        axis=3,
    )
    clear = requested[:, None, None] & target_mask & status.active & matches
    any_cleared = jnp.any(clear, axis=(1, 2))
    return status._replace(
        effect_id=jnp.where(clear, 0, status.effect_id),
        source_entity_id=jnp.where(clear, -1, status.source_entity_id),
        remaining_seconds=jnp.where(clear, 0.0, status.remaining_seconds),
        cycle_elapsed_seconds=jnp.where(clear, 0.0, status.cycle_elapsed_seconds),
        cycle_cooldown_seconds=jnp.where(clear, 0.0, status.cycle_cooldown_seconds),
        damage_per_cycle=jnp.where(clear, 0.0, status.damage_per_cycle),
        damage_cause=jnp.where(clear, 0, status.damage_cause),
        healing_per_cycle=jnp.where(clear, 0.0, status.healing_per_cycle),
        resource_id=jnp.where(clear, -1, status.resource_id),
        resource_delta_per_cycle=jnp.where(clear, 0.0, status.resource_delta_per_cycle),
        speed_multiplier=jnp.where(clear, 1.0, status.speed_multiplier),
        damage_resistance_present=jnp.where(
            clear[..., None],
            False,
            status.damage_resistance_present,
        ),
        damage_resistance_flat=jnp.where(
            clear[..., None],
            0.0,
            status.damage_resistance_flat,
        ),
        damage_resistance_multiplier=jnp.where(
            clear[..., None],
            0.0,
            status.damage_resistance_multiplier,
        ),
        flags=jnp.where(clear, jnp.uint32(0), status.flags),
        overlap_mode=jnp.where(clear, 0, status.overlap_mode),
        active=status.active & ~clear,
        has_cycled=status.has_cycled & ~clear,
    ), any_cleared


def _gather_entity(array, entity_id):
    return array[jnp.arange(array.shape[0]), entity_id]
