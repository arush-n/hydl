"""Static layout and compiled command checks for entity interactions."""

from __future__ import annotations

import jax.numpy as jnp

from hytalegym.jax.combat.entities import (
    ENTITY_CAPACITY,
    invalid_roster_rows,
    validate_roster_layout,
)
from hytalegym.jax.combat.entities.interactions.schema.contract import (
    CROSSBOW_IMPACT_CAPACITY,
    CROSSBOW_IMPACT_KIND_COUNT,
    CROSSBOW_IMPACT_NONE,
    INTERACTION_CAPABILITIES,
)
from hytalegym.jax.combat.entities.interactions.schema.types import (
    CrossbowImpactCommands,
    EntityInteractionState,
)


def validate_interaction_layout(
    state: EntityInteractionState,
    commands: CrossbowImpactCommands | None = None,
) -> None:
    validate_roster_layout(state.combat.roster)
    batch = state.combat.roster.active.shape[0]
    _field(
        state.capability_bits,
        (batch,),
        jnp.uint32,
        "capability_bits",
    )
    _field(state.failure_bits, (batch,), jnp.uint32, "failure_bits")
    if commands is None:
        return
    shape = (batch, CROSSBOW_IMPACT_CAPACITY)
    for name, dtype in (
        ("requested", jnp.bool_),
        ("kind", jnp.int32),
        ("source_slot", jnp.int32),
        ("source_generation", jnp.uint32),
        ("target_slot", jnp.int32),
        ("target_generation", jnp.uint32),
        ("knockback_yaw_degrees", jnp.float32),
        ("damage_multiplier", jnp.float32),
    ):
        _field(getattr(commands, name), shape, dtype, name)
    _field(commands.overflow, (batch,), jnp.bool_, "overflow")


def invalid_interaction_rows(
    state: EntityInteractionState,
) -> jnp.ndarray:
    return invalid_roster_rows(state.combat.roster) | (
        state.capability_bits != jnp.uint32(INTERACTION_CAPABILITIES)
    )


def invalid_crossbow_commands(
    commands: CrossbowImpactCommands,
) -> jnp.ndarray:
    requested = commands.requested
    invalid = requested & (
        (commands.kind <= CROSSBOW_IMPACT_NONE)
        | (commands.kind >= CROSSBOW_IMPACT_KIND_COUNT)
        | (commands.source_slot < 0)
        | (commands.source_slot >= ENTITY_CAPACITY)
        | (commands.target_slot < 0)
        | (commands.target_slot >= ENTITY_CAPACITY)
        | ~jnp.isfinite(commands.knockback_yaw_degrees)
        | ~jnp.isfinite(commands.damage_multiplier)
        | (commands.damage_multiplier < 0.0)
    )
    return jnp.any(invalid, axis=1)


def _field(value, shape, dtype, name):
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}")
