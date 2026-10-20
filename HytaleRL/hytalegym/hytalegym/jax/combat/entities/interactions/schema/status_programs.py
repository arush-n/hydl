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
from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.combat.mechanics.schema.contract import (
    DAMAGE_COUNT,
    DAMAGE_PHYSICAL,
    DAMAGE_PROJECTILE,
    STATUS_OVERLAP_IGNORE,
    STATUS_OVERLAP_OVERWRITE,
)
from hytalegym.server_versions import server_profile


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


#: Extra rows per server release line, keyed by ``server_profile().certified_version``. Gen 5 (0.5.7) adds none.
#: 0.7.2 adds the eight ``Hit_Interrupt_*`` effects that the flagged weapon chain applies (asset files
#: ``Server/Entity/Effects/Status/Hit_Interrupt_*.json`` in the 0.7.2 Assets.zip, read 2026-10-19). Each is finite (no
#: ``Infinite``), not a debuff, with no damage or resource effect; Flinch carries ``HorizontalSpeedMultiplier`` 0.
#: Without them the native learner decoder fails closed on an eligible actor (FIDELITY-0.7.2.md (an)). Columns: asset
#: id, Duration seconds, OverlapBehavior, speed multiplier.
_HIT_INTERRUPT_STATUSES_072 = (
    ("Hit_Interrupt_Eligible", 1.0, STATUS_OVERLAP_OVERWRITE, 1.0),
    ("Hit_Interrupt_Flinch", 0.5, STATUS_OVERLAP_IGNORE, 0.0),
    ("Hit_Interrupt_Stunned", 0.7, STATUS_OVERLAP_OVERWRITE, 1.0),
    ("Hit_Interrupt_Recent_Reaction_A", 4.0, STATUS_OVERLAP_IGNORE, 1.0),
    ("Hit_Interrupt_Recent_Reaction_B", 4.0, STATUS_OVERLAP_IGNORE, 1.0),
    ("Hit_Interrupt_Recovery_Pending", 2.0, STATUS_OVERLAP_IGNORE, 1.0),
    ("Hit_Interrupt_Enraged", 8.0, STATUS_OVERLAP_OVERWRITE, 1.0),
    ("Hit_Interrupt_Protected_Attack", 8.0, STATUS_OVERLAP_IGNORE, 1.0),
)
_EXTRA_ROWS_BY_CERTIFIED_VERSION = {
    "0.5.7": (),
    "0.7.2": _HIT_INTERRUPT_STATUSES_072,
}


def _extra_row(name, duration_seconds, overlap_mode, speed_multiplier):
    return EntityInteractionStatusProgram(
        effect_id=semantic_id(name),
        duration_seconds=duration_seconds,
        cycle_cooldown_seconds=0.0,
        damage_per_cycle=0.0,
        damage_cause=DAMAGE_PHYSICAL,
        healing_per_cycle=0.0,
        resource_id=-1,
        resource_delta_per_cycle=0.0,
        speed_multiplier=speed_multiplier,
        overlap_mode=overlap_mode,
        flags=0,
        value_percent=False,
        infinite=False,
        source_entity_is_other=False,
        damage_resistance_present=(False,) * DAMAGE_COUNT,
        damage_resistance_flat=(0,) * DAMAGE_COUNT,
        damage_resistance_multiplier=(0.0,) * DAMAGE_COUNT,
    )


def entity_interaction_status_programs_by_semantic_id(
) -> dict[int, EntityInteractionStatusProgram]:
    """Return collision-checked interaction status programs by semantic ID for the active server line."""

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
    certified = server_profile().certified_version
    if certified not in _EXTRA_ROWS_BY_CERTIFIED_VERSION:
        raise ValueError(
            f"no entity-interaction status table for server version {certified!r}"
        )
    for name, duration_seconds, overlap_mode, speed_multiplier in _EXTRA_ROWS_BY_CERTIFIED_VERSION[certified]:
        row = _extra_row(name, duration_seconds, overlap_mode, speed_multiplier)
        if row.effect_id in result:
            raise ValueError(f"entity-interaction status semantic ID collision: {name!r}")
        result[row.effect_id] = row
    return result


__all__ = [
    "EntityInteractionStatusProgram",
    "entity_interaction_status_programs_by_semantic_id",
]
