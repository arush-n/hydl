"""Host factories for the logical-entity effect runtime."""

from __future__ import annotations

import jax.numpy as jnp

from hytalegym.jax.combat.entities import (
    EntityCombatState,
    empty_entity_combat_state,
    validate_roster_layout,
)
from hytalegym.jax.combat.entities.effects.schema.contract import (
    ENTITY_EFFECT_CAPABILITIES,
)
from hytalegym.jax.combat.entities.effects.schema.types import EntityEffectState
from hytalegym.jax.combat.mechanics import CombatMechanicsRules


def entity_effect_state(combat: EntityCombatState) -> EntityEffectState:
    """Attach source-generation provenance to all existing status slots."""

    validate_roster_layout(combat.roster)
    batch = combat.roster.active.shape[0]
    status = combat.mechanics.statuses
    source = jnp.clip(
        status.source_entity_id,
        0,
        combat.roster.active.shape[1] - 1,
    )
    batch_index = jnp.arange(batch)[:, None, None]
    generation = combat.roster.generation[batch_index, source]
    generation = jnp.where(
        status.active & (status.source_entity_id >= 0),
        generation,
        jnp.uint32(0),
    )
    return EntityEffectState(
        combat=combat,
        source_generation=generation,
        capability_bits=jnp.full(
            (batch,),
            ENTITY_EFFECT_CAPABILITIES,
            dtype=jnp.uint32,
        ),
        failure_bits=jnp.zeros((batch,), dtype=jnp.uint32),
    )


def empty_entity_effect_state(
    batch_size: int,
    rules: CombatMechanicsRules,
) -> EntityEffectState:
    return entity_effect_state(empty_entity_combat_state(batch_size, rules))
