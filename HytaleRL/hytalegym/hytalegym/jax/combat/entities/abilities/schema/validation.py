"""Host layout gates for crowd ability state and handoffs."""

from __future__ import annotations

import jax.numpy as jnp

from hytalegym.jax.combat.arsenal import (
    ABILITY_CAPACITY,
    EVENT_CAPACITY,
    EVENT_FLOAT_FEATURES,
    EVENT_INTEGER_FEATURES,
    EVENT_SCHEDULER_CLOCK_COUNT,
)
from hytalegym.jax.combat.entities import ENTITY_CAPACITY
from hytalegym.jax.combat.entities.abilities.schema.contract import (
    ABILITY_EVENT_CAPACITY,
    ABILITY_PROGRAM_FAMILY_CAPACITY,
)
from hytalegym.jax.combat.entities.abilities.schema.types import (
    EntityAbilityAvailability,
    EntityAbilityCommands,
    EntityAbilityEvents,
    EntityAbilityProgramBank,
    EntityAbilityQueries,
    EntityAbilityState,
)
from hytalegym.jax.combat.entities.effects import (
    EntityEffectState,
    validate_effect_layout,
)
from hytalegym.jax.combat.mechanics import (
    RESOURCE_COUNT,
    CombatMechanicsRules,
)


def validate_entity_ability_step_layout(
    state: EntityAbilityState,
    effects: EntityEffectState,
    commands: EntityAbilityCommands,
    availability: EntityAbilityAvailability,
    programs: EntityAbilityProgramBank,
    rules: CombatMechanicsRules,
) -> None:
    validate_effect_layout(effects)
    batch = effects.combat.roster.active.shape[0]
    entity = (batch, ENTITY_CAPACITY)
    ability = entity + (ABILITY_CAPACITY,)
    for name, shape, dtype in (
        ("weapon_family", entity, jnp.int32),
        ("source_generation", entity, jnp.uint32),
        ("active_ability_slot", entity, jnp.int32),
        ("ability_elapsed_seconds", entity, jnp.float32),
        ("ability_scheduler_tick", entity, jnp.int32),
        (
            "ability_scheduler_clock_seconds",
            entity + (EVENT_SCHEDULER_CLOCK_COUNT,),
            jnp.float32,
        ),
        ("ability_cooldown_seconds", ability, jnp.float32),
        ("capability_bits", (batch,), jnp.uint32),
        ("failure_bits", (batch,), jnp.uint32),
        ("world_failure_bits", (batch,), jnp.uint32),
    ):
        _field(getattr(state, name), shape, dtype, f"state.{name}")
    for name, shape, dtype in (
        ("requested_slot", entity, jnp.int32),
        ("interrupted", entity, jnp.bool_),
        ("failure_bits", (batch,), jnp.uint32),
        ("valid", (batch,), jnp.bool_),
    ):
        _field(
            getattr(commands, name),
            shape,
            dtype,
            f"commands.{name}",
        )
    for name, dtype in (
        ("source_generation", jnp.uint32),
        ("available_requirement_bits", jnp.uint32),
        ("valid", jnp.bool_),
        ("failure_bits", jnp.uint32),
    ):
        _field(
            getattr(availability, name),
            entity,
            dtype,
            f"availability.{name}",
        )
    validate_entity_ability_program_layout(programs)
    _validate_rules(rules, batch)


def validate_entity_ability_program_layout(
    programs: EntityAbilityProgramBank,
) -> None:
    family = (ABILITY_PROGRAM_FAMILY_CAPACITY,)
    ability = family + (ABILITY_CAPACITY,)
    event = ability + (EVENT_CAPACITY,)
    expected = (
        ("family_mask", family, jnp.bool_),
        ("weapon_id", family, jnp.int32),
        ("guard_entry_cost", family, jnp.float32),
        ("guard_stamina_value", family, jnp.float32),
        ("guard_half_angle_degrees", family, jnp.float32),
        ("guard_entry_delay_seconds", family, jnp.float32),
        ("guard_exit_regen_delay_seconds", family, jnp.float32),
        ("guard_required_resource_id", family, jnp.int32),
        ("guard_required_resource_minimum", family, jnp.float32),
        (
            "resource_maximum",
            family + (RESOURCE_COUNT,),
            jnp.float32,
        ),
        (
            "resource_initial",
            family + (RESOURCE_COUNT,),
            jnp.float32,
        ),
        ("ability_id", ability, jnp.int32),
        ("ability_mask", ability, jnp.bool_),
        ("ability_evidence", ability, jnp.int32),
        ("ability_duration_seconds", ability, jnp.float32),
        ("ability_cooldown_seconds", ability, jnp.float32),
        (
            "ability_stamina_regen_delay_seconds",
            ability,
            jnp.float32,
        ),
        ("ability_scheduler_prelude_ticks", ability, jnp.int32),
        (
            "ability_resource_cost",
            ability + (RESOURCE_COUNT,),
            jnp.float32,
        ),
        (
            "ability_resource_commit_time_seconds",
            ability + (RESOURCE_COUNT,),
            jnp.float32,
        ),
        (
            "ability_resource_commit_flags",
            ability + (RESOURCE_COUNT,),
            jnp.uint32,
        ),
        (
            "ability_resource_minimum",
            ability + (RESOURCE_COUNT,),
            jnp.float32,
        ),
        ("ability_requirements", ability, jnp.uint32),
        ("event_mask", event, jnp.bool_),
        ("event_time_seconds", event, jnp.float32),
        ("event_kind", event, jnp.int32),
        (
            "event_f32",
            event + (EVENT_FLOAT_FEATURES,),
            jnp.float32,
        ),
        (
            "event_i32",
            event + (EVENT_INTEGER_FEATURES,),
            jnp.int32,
        ),
        ("event_flags", event, jnp.uint32),
        ("projectile_launch_compatible", ability, jnp.bool_),
        ("area_launch_compatible", ability, jnp.bool_),
        ("overflow", family, jnp.bool_),
    )
    for name, shape, dtype in expected:
        _field(
            getattr(programs, name),
            shape,
            dtype,
            f"programs.{name}",
        )


def validate_entity_ability_event_layout(
    events: EntityAbilityEvents,
) -> int:
    batch = events.requested.shape[0]
    query = (batch, ABILITY_EVENT_CAPACITY)
    for name, shape, dtype in (
        ("requested", query, jnp.bool_),
        ("source_slot", query, jnp.int32),
        ("source_generation", query, jnp.uint32),
        ("weapon_family", query, jnp.int32),
        ("ability_slot", query, jnp.int32),
        ("event_slot", query, jnp.int32),
        ("kind", query, jnp.int32),
        (
            "f32",
            query + (EVENT_FLOAT_FEATURES,),
            jnp.float32,
        ),
        (
            "i32",
            query + (EVENT_INTEGER_FEATURES,),
            jnp.int32,
        ),
        ("flags", query, jnp.uint32),
        ("overflow", (batch,), jnp.bool_),
        ("failure_bits", (batch,), jnp.uint32),
        ("world_failure_bits", (batch,), jnp.uint32),
        ("valid", (batch,), jnp.bool_),
    ):
        _field(getattr(events, name), shape, dtype, f"events.{name}")
    return batch


def validate_entity_ability_query_layout(
    events: EntityAbilityEvents,
    queries: EntityAbilityQueries,
) -> None:
    batch = validate_entity_ability_event_layout(events)
    query = (batch, ABILITY_EVENT_CAPACITY)
    for name, shape, dtype in (
        (
            "candidate_mask",
            query + (ENTITY_CAPACITY,),
            jnp.bool_,
        ),
        (
            "candidate_generation",
            query + (ENTITY_CAPACITY,),
            jnp.uint32,
        ),
        ("friendly_fire", query, jnp.bool_),
        ("query_valid", query, jnp.bool_),
        ("failure_bits", query, jnp.uint32),
    ):
        _field(
            getattr(queries, name),
            shape,
            dtype,
            f"queries.{name}",
        )


def _validate_rules(rules, batch):
    entity = (batch, ENTITY_CAPACITY)
    _field(
        rules.resource_minimum,
        entity + (RESOURCE_COUNT,),
        jnp.float32,
        "rules.resource_minimum",
    )
    _field(
        rules.resource_maximum,
        entity + (RESOURCE_COUNT,),
        jnp.float32,
        "rules.resource_maximum",
    )
    for name, dtype in (
        ("guard_entry_cost", jnp.float32),
        ("guard_stamina_value", jnp.float32),
        ("guard_half_angle_degrees", jnp.float32),
        ("guard_entry_delay_seconds", jnp.float32),
        ("guard_exit_regen_delay_seconds", jnp.float32),
        ("guard_required_resource_id", jnp.int32),
        ("guard_required_resource_minimum", jnp.float32),
    ):
        _field(getattr(rules, name), entity, dtype, f"rules.{name}")


def _field(value, shape, dtype, name):
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}")
