"""Static layout checks and compiled semantic validation for entity state."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.entities.schema.contract import (
    DESPAWN_CAPACITY,
    ENTITY_CAPABILITIES,
    ENTITY_CAPACITY,
    SELECTOR_CAPACITY,
    SPAWN_CAPACITY,
    TEAM_NONE,
)
from hytalegym.jax.combat.entities.schema.types import (
    EntityDamagePayloads,
    EntityLifecycleCommands,
    EntityRoster,
    EntitySelectorQueries,
    EntityStatusPayloads,
)
from hytalegym.jax.combat.mechanics import RESOURCE_COUNT


def validate_roster_layout(roster: EntityRoster) -> None:
    batch = roster.active.shape[0]
    entity = (batch, ENTITY_CAPACITY)
    _field(roster.semantic_id, entity, jnp.int32, "semantic_id")
    _field(roster.generation, entity, jnp.uint32, "generation")
    _field(roster.team_id, entity, jnp.int32, "team_id")
    _field(roster.position, entity + (3,), jnp.float32, "position")
    _field(roster.velocity, entity + (3,), jnp.float32, "velocity")
    _field(roster.yaw_degrees, entity, jnp.float32, "yaw_degrees")
    _field(roster.health, entity, jnp.float32, "health")
    _field(roster.max_health, entity, jnp.float32, "max_health")
    for name in (
        "damageable",
        "intangible",
        "invulnerable",
        "dead",
        "active",
    ):
        _field(getattr(roster, name), entity, jnp.bool_, name)
    _field(roster.capability_bits, (batch,), jnp.uint32, "capability_bits")
    _field(roster.failure_bits, (batch,), jnp.uint32, "failure_bits")


def validate_lifecycle_layout(
    roster: EntityRoster,
    commands: EntityLifecycleCommands,
) -> None:
    validate_roster_layout(roster)
    batch = roster.active.shape[0]
    spawn = (batch, SPAWN_CAPACITY)
    despawn = (batch, DESPAWN_CAPACITY)
    for name, dtype in (
        ("requested", jnp.bool_),
        ("semantic_id", jnp.int32),
        ("team_id", jnp.int32),
        ("yaw_degrees", jnp.float32),
        ("health", jnp.float32),
        ("max_health", jnp.float32),
        ("damageable", jnp.bool_),
        ("intangible", jnp.bool_),
        ("invulnerable", jnp.bool_),
    ):
        _field(getattr(commands.spawn, name), spawn, dtype, f"spawn.{name}")
    _field(
        commands.spawn.position,
        spawn + (3,),
        jnp.float32,
        "spawn.position",
    )
    _field(
        commands.spawn.velocity,
        spawn + (3,),
        jnp.float32,
        "spawn.velocity",
    )
    _field(
        commands.spawn.resource_initial,
        spawn + (RESOURCE_COUNT,),
        jnp.float32,
        "spawn.resource_initial",
    )
    _field(
        commands.despawn.requested,
        despawn,
        jnp.bool_,
        "despawn.requested",
    )
    _field(commands.despawn.slot, despawn, jnp.int32, "despawn.slot")
    _field(
        commands.despawn.generation,
        despawn,
        jnp.uint32,
        "despawn.generation",
    )


def validate_selector_layout(
    roster: EntityRoster,
    queries: EntitySelectorQueries,
) -> None:
    validate_roster_layout(roster)
    batch = roster.active.shape[0]
    query = (batch, SELECTOR_CAPACITY)
    for name, dtype in (
        ("requested", jnp.bool_),
        ("source_slot", jnp.int32),
        ("source_generation", jnp.uint32),
        ("ignore_owner", jnp.bool_),
        ("friendly_fire", jnp.bool_),
        ("max_targets", jnp.int32),
    ):
        _field(getattr(queries, name), query, dtype, name)
    _field(
        queries.candidate_mask,
        query + (ENTITY_CAPACITY,),
        jnp.bool_,
        "candidate_mask",
    )


def validate_damage_payload_layout(
    roster: EntityRoster,
    payloads: EntityDamagePayloads,
) -> None:
    batch = roster.active.shape[0]
    query = (batch, SELECTOR_CAPACITY)
    for name, dtype in (
        ("requested", jnp.bool_),
        ("amount", jnp.float32),
        ("random_percentage", jnp.float32),
        ("damage_class", jnp.int32),
        ("cause", jnp.int32),
        ("force_mode", jnp.int32),
        ("air_resistance", jnp.float32),
        ("air_resistance_max", jnp.float32),
        ("ground_resistance", jnp.float32),
        ("ground_resistance_max", jnp.float32),
        ("resistance_threshold", jnp.float32),
        ("resistance_style", jnp.int32),
        ("dampen_y", jnp.bool_),
        ("on_hit_resource_id", jnp.int32),
        ("on_hit_resource_delta", jnp.float32),
    ):
        _field(getattr(payloads, name), query, dtype, name)
    _field(
        payloads.knockback_velocity,
        query + (ENTITY_CAPACITY, 3),
        jnp.float32,
        "knockback_velocity",
    )


def validate_status_payload_layout(
    roster: EntityRoster,
    payloads: EntityStatusPayloads,
) -> None:
    batch = roster.active.shape[0]
    query = (batch, SELECTOR_CAPACITY)
    for name, dtype in (
        ("requested", jnp.bool_),
        ("effect_id", jnp.int32),
        ("duration_seconds", jnp.float32),
        ("cycle_cooldown_seconds", jnp.float32),
        ("damage_per_cycle", jnp.float32),
        ("damage_cause", jnp.int32),
        ("healing_per_cycle", jnp.float32),
        ("resource_id", jnp.int32),
        ("resource_delta_per_cycle", jnp.float32),
        ("speed_multiplier", jnp.float32),
        ("flags", jnp.uint32),
        ("overlap_mode", jnp.int32),
    ):
        _field(getattr(payloads, name), query, dtype, name)


def validate_selection_layout(batch: int, selection) -> None:
    query = (batch, SELECTOR_CAPACITY)
    expected = {
        "target_mask": (query + (ENTITY_CAPACITY,), jnp.bool_),
        "target_count": (query, jnp.int32),
        "available_count": (query, jnp.int32),
        "source_slot": (query, jnp.int32),
        "failure_bits": (query, jnp.uint32),
        "row_failure_bits": ((batch,), jnp.uint32),
        "valid": (query, jnp.bool_),
    }
    for name, (shape, dtype) in expected.items():
        _field(getattr(selection, name), shape, dtype, f"selection.{name}")


def invalid_roster_rows(roster: EntityRoster) -> jax.Array:
    active = roster.active
    damageable = active & roster.damageable
    nondamageable = active & ~roster.damageable
    invalid_entity = active & (
        (roster.semantic_id <= 0)
        | (roster.generation == jnp.uint32(0))
        | (roster.team_id < TEAM_NONE)
        | ~jnp.all(jnp.isfinite(roster.position), axis=2)
        | ~jnp.all(jnp.isfinite(roster.velocity), axis=2)
        | ~jnp.isfinite(roster.yaw_degrees)
        | (
            damageable
            & (
                ~jnp.isfinite(roster.health)
                | ~jnp.isfinite(roster.max_health)
                | (roster.max_health <= 0.0)
                | (roster.health < 0.0)
                | (roster.health > roster.max_health)
                | (roster.dead != (roster.health <= 0.0))
            )
        )
        | (
            nondamageable
            & (
                (roster.health != 0.0)
                | (roster.max_health != 0.0)
                | roster.dead
            )
        )
    )
    capability_invalid = (
        roster.capability_bits
        != jnp.uint32(ENTITY_CAPABILITIES)
    )
    return jnp.any(invalid_entity, axis=1) | capability_invalid


def _field(array, shape, dtype, name):
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}")
