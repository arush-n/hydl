"""Layout and compiled semantic checks for entity effects."""

from __future__ import annotations

import jax.numpy as jnp

from hytalegym.jax.combat.entities import (
    ENTITY_CAPACITY,
    invalid_roster_rows,
    validate_roster_layout,
)
from hytalegym.jax.combat.entities.effects.schema.contract import (
    ENTITY_EFFECT_CAPABILITIES,
)
from hytalegym.jax.combat.entities.effects.schema.types import EntityEffectState
from hytalegym.jax.combat.mechanics import STATUS_CAPACITY
from hytalegym.jax.combat.mechanics import (
    DAMAGE_COUNT,
    RESOURCE_COUNT,
    STATUS_FLAG_DEBUFF,
    STATUS_FLAG_CONTROL_IMMUNITY_GATED,
    STATUS_FLAG_DISABLE_ABILITIES,
    STATUS_FLAG_DISABLE_MOVEMENT,
    STATUS_FLAG_DISABLE_SPRINT,
    STATUS_FLAG_IGNORE_KNOCKBACK,
    STATUS_FLAG_INVULNERABLE,
    STATUS_OVERLAP_IGNORE,
    STATUS_OVERLAP_OVERWRITE,
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


def validate_effect_layout(state: EntityEffectState) -> None:
    validate_roster_layout(state.combat.roster)
    batch = state.combat.roster.active.shape[0]
    _field(
        state.source_generation,
        (batch, ENTITY_CAPACITY, STATUS_CAPACITY),
        jnp.uint32,
        "source_generation",
    )
    _field(
        state.capability_bits,
        (batch,),
        jnp.uint32,
        "capability_bits",
    )
    _field(state.failure_bits, (batch,), jnp.uint32, "failure_bits")
    shape = (batch, ENTITY_CAPACITY, STATUS_CAPACITY)
    status = state.combat.mechanics.statuses
    for name, dtype in (
        ("effect_id", jnp.int32),
        ("source_entity_id", jnp.int32),
        ("remaining_seconds", jnp.float32),
        ("cycle_elapsed_seconds", jnp.float32),
        ("cycle_cooldown_seconds", jnp.float32),
        ("damage_per_cycle", jnp.float32),
        ("damage_cause", jnp.int32),
        ("healing_per_cycle", jnp.float32),
        ("resource_id", jnp.int32),
        ("resource_delta_per_cycle", jnp.float32),
        ("speed_multiplier", jnp.float32),
        ("flags", jnp.uint32),
        ("overlap_mode", jnp.int32),
        ("active", jnp.bool_),
        ("has_cycled", jnp.bool_),
    ):
        _field(getattr(status, name), shape, dtype, f"statuses.{name}")
    resistance_shape = shape + (DAMAGE_COUNT,)
    for name, dtype in (
        ("damage_resistance_present", jnp.bool_),
        ("damage_resistance_flat", jnp.float32),
        ("damage_resistance_multiplier", jnp.float32),
    ):
        _field(
            getattr(status, name),
            resistance_shape,
            dtype,
            f"statuses.{name}",
        )


def invalid_effect_rows(state: EntityEffectState):
    status = state.combat.mechanics.statuses
    active = status.active
    invalid_resistance = ~jnp.all(
        jnp.isfinite(status.damage_resistance_flat), axis=3
    ) | ~jnp.all(
        jnp.isfinite(status.damage_resistance_multiplier),
        axis=3,
    )
    invalid_status = active & (
        (status.effect_id <= 0)
        | ~jnp.isfinite(status.remaining_seconds)
        | (status.remaining_seconds <= 0.0)
        | ~jnp.isfinite(status.cycle_elapsed_seconds)
        | (status.cycle_elapsed_seconds < 0.0)
        | ~jnp.isfinite(status.cycle_cooldown_seconds)
        | (status.cycle_cooldown_seconds < 0.0)
        | ~jnp.isfinite(status.damage_per_cycle)
        | (status.damage_per_cycle < 0.0)
        | (status.damage_cause < 0)
        | (status.damage_cause >= DAMAGE_COUNT)
        | ~jnp.isfinite(status.healing_per_cycle)
        | (status.healing_per_cycle < 0.0)
        | (status.resource_id < -1)
        | (status.resource_id >= RESOURCE_COUNT)
        | ~jnp.isfinite(status.resource_delta_per_cycle)
        | ~jnp.isfinite(status.speed_multiplier)
        | (status.speed_multiplier <= 0.0)
        | ((status.flags & jnp.uint32(0xFFFFFFFF ^ _STATUS_FLAG_MASK)) != 0)
        | (status.overlap_mode < STATUS_OVERLAP_IGNORE)
        | (status.overlap_mode > STATUS_OVERLAP_OVERWRITE)
        | invalid_resistance
    )
    inactive_provenance = ~active & (state.source_generation != jnp.uint32(0))
    return (
        invalid_roster_rows(state.combat.roster)
        | jnp.any(invalid_status | inactive_provenance, axis=(1, 2))
        | (state.capability_bits != jnp.uint32(ENTITY_EFFECT_CAPABILITIES))
    )


def stale_effect_source_rows(state: EntityEffectState):
    status = state.combat.mechanics.statuses
    source = status.source_entity_id
    entity_count = state.combat.roster.active.shape[1]
    safe_source = jnp.clip(source, 0, entity_count - 1)
    generation = state.combat.roster.generation[
        jnp.arange(source.shape[0])[:, None, None],
        safe_source,
    ]
    active = state.combat.roster.active[
        jnp.arange(source.shape[0])[:, None, None],
        safe_source,
    ]
    invalid = status.active & (
        (source < -1)
        | (source >= entity_count)
        | (
            (source >= 0)
            & (
                ~active
                | (state.source_generation == jnp.uint32(0))
                | (state.source_generation != generation)
            )
        )
        | ((source < 0) & (state.source_generation != jnp.uint32(0)))
    )
    return jnp.any(invalid, axis=(1, 2))


def _field(value, shape, dtype, name):
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}")
