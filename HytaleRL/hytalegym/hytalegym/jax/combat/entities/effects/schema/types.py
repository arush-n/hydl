"""Array-only state and output trees for logical-entity effects."""

from __future__ import annotations

from typing import NamedTuple

import jax

from hytalegym.jax.combat.entities import EntityCombatState


Array = jax.Array


class EntityEffectState(NamedTuple):
    combat: EntityCombatState
    source_generation: Array
    capability_bits: Array
    failure_bits: Array


class EntityEffectTickInfo(NamedTuple):
    status_cycles: Array
    damage_event_count: Array
    damage_applied: Array
    healing_applied: Array
    blocked_hits: Array
    invulnerable_hits: Array
    newly_dead: Array
    aggregate_flags: Array
    speed_multiplier: Array
    active_status_count: Array
    failure_bits: Array
    valid: Array
