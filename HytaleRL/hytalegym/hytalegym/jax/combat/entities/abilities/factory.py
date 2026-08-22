"""Host factories for the pinned 0.5.7 crowd ability contract."""

from __future__ import annotations

from functools import lru_cache
import operator

import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat.arsenal import (
    ABILITY_CAPACITY,
    EVENT_SCHEDULER_CLOCK_COUNT,
    EVENT_AREA,
    EVENT_CAPACITY,
    EVENT_FLOAT_FEATURES,
    EVENT_INTEGER_FEATURES,
    EVENT_PROJECTILE,
    PROFILE_NAMES,
    effective_resource_cost,
    hytale_0_5_7_loadouts,
    validate_ability_loadout,
)
from hytalegym.jax.combat.entities import (
    ENTITY_CAPACITY,
    EntityRoster,
    validate_roster_layout,
)
from hytalegym.jax.combat.entities.abilities.schema.contract import (
    ABILITY_CAPABILITIES,
    ABILITY_EVENT_CAPACITY,
    ABILITY_PROGRAM_FAMILY_CAPACITY,
)
from hytalegym.jax.combat.entities.abilities.schema.types import (
    EntityAbilityAvailability,
    EntityAbilityCommands,
    EntityAbilityProgramBank,
    EntityAbilityQueries,
    EntityAbilityState,
)
from hytalegym.jax.combat.mechanics import (
    RESOURCE_COUNT,
    CombatMechanicsRules,
    default_mechanics_rules,
)


@lru_cache(maxsize=1)
def hytale_0_5_7_entity_ability_programs() -> EntityAbilityProgramBank:
    """Lift every pinned two-entity profile into one family-indexed bank."""

    loadout = hytale_0_5_7_loadouts(PROFILE_NAMES)
    if bool(jnp.any(validate_ability_loadout(loadout))):
        raise ValueError("pinned Arsenal profile validation failed")
    # The crowd runtime executes server-side NPC programs, so its compact
    # bank stores the already-resolved amount rather than duplicating Arsenal's
    # authored mechanism metadata.
    loadout = loadout._replace(
        ability_resource_cost=effective_resource_cost(
            loadout.ability_resource_cost,
            loadout.ability_resource_cost_kind,
        )
    )
    family = np.asarray(loadout.weapon_family[:, 0], dtype=np.int32)
    if np.any((family <= 0) | (family >= ABILITY_PROGRAM_FAMILY_CAPACITY)) or len(
        np.unique(family)
    ) != len(family):
        raise ValueError("pinned Arsenal family IDs are not unique/in range")
    family_mask = np.zeros((ABILITY_PROGRAM_FAMILY_CAPACITY,), dtype=np.bool_)

    def bank(field, tail, dtype, default=0):
        result = np.full(
            (ABILITY_PROGRAM_FAMILY_CAPACITY,) + tail,
            default,
            dtype=dtype,
        )
        result[family] = np.asarray(getattr(loadout, field)[:, 0])
        return result

    family_mask[family] = True
    ability = (ABILITY_CAPACITY,)
    event = ability + (EVENT_CAPACITY,)
    event_kind = bank("event_kind", event, np.int32)
    event_mask = bank("event_mask", event, np.bool_)
    projectile_compatible = _launch_compatible(
        event_mask,
        event_kind,
        bank("event_time_seconds", event, np.float32),
        EVENT_PROJECTILE,
    )
    area_compatible = _launch_compatible(
        event_mask,
        event_kind,
        bank("event_time_seconds", event, np.float32),
        EVENT_AREA,
    )
    return EntityAbilityProgramBank(
        family_mask=jnp.asarray(family_mask),
        weapon_id=jnp.asarray(bank("weapon_id", (), np.int32, -1)),
        guard_entry_cost=jnp.asarray(bank("guard_entry_cost", (), np.float32, 0.5)),
        guard_stamina_value=jnp.asarray(
            bank("guard_stamina_value", (), np.float32, 7.0)
        ),
        guard_half_angle_degrees=jnp.asarray(
            bank("guard_half_angle_degrees", (), np.float32, 90.0)
        ),
        guard_entry_delay_seconds=jnp.asarray(
            bank("guard_entry_delay_seconds", (), np.float32)
        ),
        guard_exit_regen_delay_seconds=jnp.asarray(
            bank("guard_exit_regen_delay_seconds", (), np.float32, -1.0)
        ),
        guard_required_resource_id=jnp.asarray(
            bank("guard_required_resource_id", (), np.int32, -1)
        ),
        guard_required_resource_minimum=jnp.asarray(
            bank("guard_required_resource_minimum", (), np.float32)
        ),
        resource_maximum=jnp.asarray(
            bank(
                "resource_maximum",
                (RESOURCE_COUNT,),
                np.float32,
            )
        ),
        resource_initial=jnp.asarray(
            bank(
                "resource_initial",
                (RESOURCE_COUNT,),
                np.float32,
            )
        ),
        ability_id=jnp.asarray(bank("ability_id", ability, np.int32)),
        ability_mask=jnp.asarray(bank("ability_mask", ability, np.bool_)),
        ability_evidence=jnp.asarray(bank("ability_evidence", ability, np.int32)),
        ability_duration_seconds=jnp.asarray(
            bank("ability_duration_seconds", ability, np.float32)
        ),
        ability_cooldown_seconds=jnp.asarray(
            bank("ability_cooldown_seconds", ability, np.float32)
        ),
        ability_stamina_regen_delay_seconds=jnp.asarray(
            bank(
                "ability_stamina_regen_delay_seconds",
                ability,
                np.float32,
            )
        ),
        ability_scheduler_prelude_ticks=jnp.asarray(
            bank("ability_scheduler_prelude_ticks", ability, np.int32)
        ),
        ability_resource_cost=jnp.asarray(
            bank(
                "ability_resource_cost",
                ability + (RESOURCE_COUNT,),
                np.float32,
            )
        ),
        ability_resource_commit_time_seconds=jnp.asarray(
            bank(
                "ability_resource_commit_time_seconds",
                ability + (RESOURCE_COUNT,),
                np.float32,
            )
        ),
        ability_resource_commit_flags=jnp.asarray(
            bank(
                "ability_resource_commit_flags",
                ability + (RESOURCE_COUNT,),
                np.uint32,
            )
        ),
        ability_resource_minimum=jnp.asarray(
            bank(
                "ability_resource_minimum",
                ability + (RESOURCE_COUNT,),
                np.float32,
            )
        ),
        ability_requirements=jnp.asarray(
            bank("ability_requirements", ability, np.uint32)
        ),
        event_mask=jnp.asarray(event_mask),
        event_time_seconds=jnp.asarray(bank("event_time_seconds", event, np.float32)),
        event_kind=jnp.asarray(event_kind),
        event_f32=jnp.asarray(
            bank(
                "event_f32",
                event + (EVENT_FLOAT_FEATURES,),
                np.float32,
            )
        ),
        event_i32=jnp.asarray(
            bank(
                "event_i32",
                event + (EVENT_INTEGER_FEATURES,),
                np.int32,
            )
        ),
        event_flags=jnp.asarray(bank("event_flags", event, np.uint32)),
        projectile_launch_compatible=jnp.asarray(projectile_compatible),
        area_launch_compatible=jnp.asarray(area_compatible),
        overflow=jnp.asarray(bank("overflow", (), np.bool_)),
    )


def mechanics_rules_for_entity_ability_families(
    weapon_family: jnp.ndarray,
    programs: EntityAbilityProgramBank | None = None,
) -> CombatMechanicsRules:
    """Project family-specific resources and guard drain into common rules."""

    family = _families(weapon_family)
    bank = programs or hytale_0_5_7_entity_ability_programs()
    rules = default_mechanics_rules(family.shape[0], entity_count=ENTITY_CAPACITY)
    index = jnp.clip(family, 0, ABILITY_PROGRAM_FAMILY_CAPACITY - 1)
    return rules._replace(
        resource_maximum=bank.resource_maximum[index],
        guard_entry_cost=bank.guard_entry_cost[index],
        guard_stamina_value=bank.guard_stamina_value[index],
        guard_half_angle_degrees=bank.guard_half_angle_degrees[index],
        guard_entry_delay_seconds=bank.guard_entry_delay_seconds[index],
        guard_exit_regen_delay_seconds=(bank.guard_exit_regen_delay_seconds[index]),
        guard_required_resource_id=bank.guard_required_resource_id[index],
        guard_required_resource_minimum=(bank.guard_required_resource_minimum[index]),
    )


def initial_resources_for_entity_ability_families(
    weapon_family: jnp.ndarray,
    programs: EntityAbilityProgramBank | None = None,
):
    family = _families(weapon_family)
    bank = programs or hytale_0_5_7_entity_ability_programs()
    return bank.resource_initial[
        jnp.clip(family, 0, ABILITY_PROGRAM_FAMILY_CAPACITY - 1)
    ]


def empty_entity_ability_state(
    roster: EntityRoster,
    weapon_family: jnp.ndarray,
) -> EntityAbilityState:
    validate_roster_layout(roster)
    family = _families(weapon_family, batch=roster.active.shape[0])
    equipped = family > 0
    if bool(jnp.any(equipped & ~roster.active)):
        raise ValueError("equipped ability families require active slots")
    if bool(jnp.any(equipped & (roster.generation == jnp.uint32(0)))):
        raise ValueError("equipped ability families require generations")
    entity = roster.active.shape
    batch = entity[0]
    return EntityAbilityState(
        weapon_family=family,
        source_generation=jnp.where(equipped, roster.generation, jnp.uint32(0)),
        active_ability_slot=jnp.full(entity, -1, dtype=jnp.int32),
        ability_elapsed_seconds=jnp.zeros(entity, dtype=jnp.float32),
        ability_scheduler_tick=jnp.zeros(entity, dtype=jnp.int32),
        ability_scheduler_clock_seconds=jnp.zeros(
            entity + (EVENT_SCHEDULER_CLOCK_COUNT,),
            dtype=jnp.float32,
        ),
        ability_cooldown_seconds=jnp.zeros(
            entity + (ABILITY_CAPACITY,), dtype=jnp.float32
        ),
        capability_bits=jnp.full((batch,), ABILITY_CAPABILITIES, dtype=jnp.uint32),
        failure_bits=jnp.zeros((batch,), dtype=jnp.uint32),
        world_failure_bits=jnp.zeros((batch,), dtype=jnp.uint32),
    )


def empty_entity_ability_commands(
    batch_size: int,
) -> EntityAbilityCommands:
    batch = _positive_size(batch_size, "batch_size")
    entity = (batch, ENTITY_CAPACITY)
    return EntityAbilityCommands(
        requested_slot=jnp.full(entity, -1, dtype=jnp.int32),
        interrupted=jnp.zeros(entity, dtype=jnp.bool_),
        failure_bits=jnp.zeros((batch,), dtype=jnp.uint32),
        valid=jnp.ones((batch,), dtype=jnp.bool_),
    )


def empty_entity_ability_availability(
    batch_size: int,
) -> EntityAbilityAvailability:
    batch = _positive_size(batch_size, "batch_size")
    entity = (batch, ENTITY_CAPACITY)
    return EntityAbilityAvailability(
        source_generation=jnp.zeros(entity, dtype=jnp.uint32),
        available_requirement_bits=jnp.zeros(entity, dtype=jnp.uint32),
        valid=jnp.zeros(entity, dtype=jnp.bool_),
        failure_bits=jnp.zeros(entity, dtype=jnp.uint32),
    )


def empty_entity_ability_queries(
    batch_size: int,
) -> EntityAbilityQueries:
    batch = _positive_size(batch_size, "batch_size")
    query = (batch, ABILITY_EVENT_CAPACITY)
    return EntityAbilityQueries(
        candidate_mask=jnp.zeros(query + (ENTITY_CAPACITY,), dtype=jnp.bool_),
        candidate_generation=jnp.zeros(query + (ENTITY_CAPACITY,), dtype=jnp.uint32),
        friendly_fire=jnp.zeros(query, dtype=jnp.bool_),
        query_valid=jnp.zeros(query, dtype=jnp.bool_),
        failure_bits=jnp.zeros(query, dtype=jnp.uint32),
    )


def _launch_compatible(mask, kind, time, event_kind):
    selected = mask & (kind == event_kind)
    count = np.sum(selected, axis=2)
    first = np.min(np.where(selected, time, np.inf), axis=2)
    last = np.max(np.where(selected, time, -np.inf), axis=2)
    return (count == 0) | (first == last)


def _families(value, *, batch=None):
    result = jnp.asarray(value)
    expected = (batch, ENTITY_CAPACITY) if batch is not None else result.shape
    if result.ndim != 2 or result.shape[1] != ENTITY_CAPACITY:
        raise ValueError(f"weapon_family must have shape [B,{ENTITY_CAPACITY}]")
    if result.shape != expected:
        raise ValueError(f"weapon_family must have shape {expected}")
    if result.dtype != jnp.int32:
        raise TypeError("weapon_family must have dtype int32")
    if bool(jnp.any((result < 0) | (result >= ABILITY_PROGRAM_FAMILY_CAPACITY))):
        raise ValueError("weapon_family contains an unsupported family")
    return result


def _positive_size(value: int, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result
