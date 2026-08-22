"""Array-only PyTrees for ordered logical-entity interactions."""

from __future__ import annotations

from typing import NamedTuple

import jax

from hytalegym.jax.combat.entities import EntityCombatState


Array = jax.Array


class EntityInteractionState(NamedTuple):
    combat: EntityCombatState
    capability_bits: Array
    failure_bits: Array


class CrossbowImpactCommands(NamedTuple):
    requested: Array
    kind: Array
    source_slot: Array
    source_generation: Array
    target_slot: Array
    target_generation: Array
    knockback_yaw_degrees: Array
    damage_multiplier: Array
    overflow: Array


class CrossbowImpactEvents(NamedTuple):
    """Per-capacity-axis command leaves, excluding row-level overflow."""

    requested: Array
    kind: Array
    source_slot: Array
    source_generation: Array
    target_slot: Array
    target_generation: Array
    knockback_yaw_degrees: Array
    damage_multiplier: Array


class CrossbowImpactInfo(NamedTuple):
    combo_stage: Array
    damage_applied: Array
    blocked: Array
    invulnerable: Array
    combo_1_applied: Array
    combo_2_applied: Array
    combo_cleared: Array
    gameplay_regen_cleared: Array
    signature_energy_gained: Array
    knockback_velocity: Array
    failure_bits: Array
    valid: Array
