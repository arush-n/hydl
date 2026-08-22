"""Host factories for ordered entity interaction arrays."""

from __future__ import annotations

import operator

import jax.numpy as jnp

from hytalegym.jax.combat.entities import (
    EntityCombatState,
    empty_entity_combat_state,
    validate_roster_layout,
)
from hytalegym.jax.combat.entities.interactions.schema.contract import (
    CROSSBOW_IMPACT_CAPACITY,
    CROSSBOW_IMPACT_STANDARD,
    INTERACTION_CAPABILITIES,
)
from hytalegym.jax.combat.entities.interactions.schema.types import (
    CrossbowImpactCommands,
    EntityInteractionState,
)
from hytalegym.jax.combat.mechanics import CombatMechanicsRules


def entity_interaction_state(
    combat: EntityCombatState,
) -> EntityInteractionState:
    """Attach a clean interaction failure domain to an existing roster."""

    validate_roster_layout(combat.roster)
    batch = combat.roster.active.shape[0]
    return EntityInteractionState(
        combat=combat,
        capability_bits=jnp.full(
            (batch,),
            INTERACTION_CAPABILITIES,
            dtype=jnp.uint32,
        ),
        failure_bits=jnp.zeros((batch,), dtype=jnp.uint32),
    )


def empty_entity_interaction_state(
    batch_size: int,
    rules: CombatMechanicsRules,
) -> EntityInteractionState:
    return entity_interaction_state(
        empty_entity_combat_state(batch_size, rules)
    )


def empty_crossbow_impact_commands(
    batch_size: int,
) -> CrossbowImpactCommands:
    batch = _positive_size(batch_size, "batch_size")
    shape = (batch, CROSSBOW_IMPACT_CAPACITY)
    return CrossbowImpactCommands(
        requested=jnp.zeros(shape, dtype=jnp.bool_),
        kind=jnp.full(
            shape,
            CROSSBOW_IMPACT_STANDARD,
            dtype=jnp.int32,
        ),
        source_slot=jnp.full(shape, -1, dtype=jnp.int32),
        source_generation=jnp.zeros(shape, dtype=jnp.uint32),
        target_slot=jnp.full(shape, -1, dtype=jnp.int32),
        target_generation=jnp.zeros(shape, dtype=jnp.uint32),
        knockback_yaw_degrees=jnp.zeros(shape, dtype=jnp.float32),
        damage_multiplier=jnp.ones(shape, dtype=jnp.float32),
        overflow=jnp.zeros((batch,), dtype=jnp.bool_),
    )


def _positive_size(value: int, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{label} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result
