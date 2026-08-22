"""Status programs authored by entity-interaction graphs.

These rows are host-side catalog data.  They let native actor evidence
reconstruct an observed ``EntityEffect`` through the same scalar program that
the JAX interaction runtime applies.  Keeping a row per effect avoids a
weapon-name branch and permits future interaction families to contribute
different durations, periodic effects, resources, flags, and resistances.
"""

from __future__ import annotations

from dataclasses import dataclass

from hytalegym.jax.combat.entities.interactions.schema.contract import (
    CROSSBOW_COMBO_1_EFFECT_ID,
    CROSSBOW_COMBO_2_EFFECT_ID,
    CROSSBOW_COMBO_DURATION_SECONDS,
)
from hytalegym.jax.combat.mechanics.schema.contract import (
    DAMAGE_COUNT,
    DAMAGE_PROJECTILE,
    STATUS_OVERLAP_IGNORE,
)


@dataclass(frozen=True)
class EntityInteractionStatusProgram:
    """Complete scalar semantics for one interaction-authored effect."""

    effect_id: int
    duration_seconds: float
    cycle_cooldown_seconds: float
    damage_per_cycle: float
    damage_cause: int
    healing_per_cycle: float
    resource_id: int
    resource_delta_per_cycle: float
    speed_multiplier: float
    overlap_mode: int
    flags: int
    value_percent: bool
    infinite: bool
    source_entity_is_other: bool
    damage_resistance_present: tuple[bool, ...]
    damage_resistance_flat: tuple[int, ...]
    damage_resistance_multiplier: tuple[float, ...]


def entity_interaction_status_programs_by_semantic_id(
) -> dict[int, EntityInteractionStatusProgram]:
    """Return collision-checked interaction status programs by semantic ID."""

    neutral_resistance_present = (False,) * DAMAGE_COUNT
    neutral_resistance_flat = (0,) * DAMAGE_COUNT
    neutral_resistance_multiplier = (0.0,) * DAMAGE_COUNT
    rows = tuple(
        EntityInteractionStatusProgram(
            effect_id=effect_id,
            duration_seconds=CROSSBOW_COMBO_DURATION_SECONDS,
            cycle_cooldown_seconds=0.0,
            damage_per_cycle=0.0,
            damage_cause=DAMAGE_PROJECTILE,
            healing_per_cycle=0.0,
            resource_id=-1,
            resource_delta_per_cycle=0.0,
            speed_multiplier=1.0,
            overlap_mode=STATUS_OVERLAP_IGNORE,
            flags=0,
            value_percent=False,
            infinite=False,
            source_entity_is_other=True,
            damage_resistance_present=neutral_resistance_present,
            damage_resistance_flat=neutral_resistance_flat,
            damage_resistance_multiplier=neutral_resistance_multiplier,
        )
        for effect_id in (
            CROSSBOW_COMBO_1_EFFECT_ID,
            CROSSBOW_COMBO_2_EFFECT_ID,
        )
    )
    result = {row.effect_id: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate entity-interaction status semantic ID")
    return result


__all__ = [
    "EntityInteractionStatusProgram",
    "entity_interaction_status_programs_by_semantic_id",
]
