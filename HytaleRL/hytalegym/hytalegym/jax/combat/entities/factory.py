"""Host factories for fixed-shape combat entity arrays."""

from __future__ import annotations

import operator

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
    EntityCombatState,
    EntityDamagePayloads,
    EntityDespawnCommands,
    EntityLifecycleCommands,
    EntityRoster,
    EntitySelectorQueries,
    EntitySpawnCommands,
    EntityStatusPayloads,
)
from hytalegym.jax.combat.mechanics import (
    CombatMechanicsRules,
    RESOURCE_COUNT,
    empty_mechanics_state,
)


def empty_entity_roster(batch_size: int) -> EntityRoster:
    batch = _positive_size(batch_size, "batch_size")
    entity = (batch, ENTITY_CAPACITY)
    f32 = jnp.zeros(entity, dtype=jnp.float32)
    return EntityRoster(
        semantic_id=jnp.zeros(entity, dtype=jnp.int32),
        generation=jnp.zeros(entity, dtype=jnp.uint32),
        team_id=jnp.full(entity, TEAM_NONE, dtype=jnp.int32),
        position=jnp.zeros(entity + (3,), dtype=jnp.float32),
        velocity=jnp.zeros(entity + (3,), dtype=jnp.float32),
        yaw_degrees=f32,
        health=f32,
        max_health=f32,
        damageable=jnp.zeros(entity, dtype=jnp.bool_),
        intangible=jnp.zeros(entity, dtype=jnp.bool_),
        invulnerable=jnp.zeros(entity, dtype=jnp.bool_),
        dead=jnp.zeros(entity, dtype=jnp.bool_),
        active=jnp.zeros(entity, dtype=jnp.bool_),
        capability_bits=jnp.full(
            (batch,),
            ENTITY_CAPABILITIES,
            dtype=jnp.uint32,
        ),
        failure_bits=jnp.zeros((batch,), dtype=jnp.uint32),
    )


def empty_entity_lifecycle_commands(
    batch_size: int,
) -> EntityLifecycleCommands:
    batch = _positive_size(batch_size, "batch_size")
    spawn = (batch, SPAWN_CAPACITY)
    despawn = (batch, DESPAWN_CAPACITY)
    return EntityLifecycleCommands(
        spawn=EntitySpawnCommands(
            requested=jnp.zeros(spawn, dtype=jnp.bool_),
            semantic_id=jnp.zeros(spawn, dtype=jnp.int32),
            team_id=jnp.full(spawn, TEAM_NONE, dtype=jnp.int32),
            position=jnp.zeros(spawn + (3,), dtype=jnp.float32),
            velocity=jnp.zeros(spawn + (3,), dtype=jnp.float32),
            yaw_degrees=jnp.zeros(spawn, dtype=jnp.float32),
            health=jnp.zeros(spawn, dtype=jnp.float32),
            max_health=jnp.zeros(spawn, dtype=jnp.float32),
            damageable=jnp.zeros(spawn, dtype=jnp.bool_),
            intangible=jnp.zeros(spawn, dtype=jnp.bool_),
            invulnerable=jnp.zeros(spawn, dtype=jnp.bool_),
            resource_initial=jnp.full(
                spawn + (RESOURCE_COUNT,),
                jnp.nan,
                dtype=jnp.float32,
            ),
        ),
        despawn=EntityDespawnCommands(
            requested=jnp.zeros(despawn, dtype=jnp.bool_),
            slot=jnp.full(despawn, -1, dtype=jnp.int32),
            generation=jnp.zeros(despawn, dtype=jnp.uint32),
        ),
    )


def empty_entity_selector_queries(
    batch_size: int,
) -> EntitySelectorQueries:
    batch = _positive_size(batch_size, "batch_size")
    query = (batch, SELECTOR_CAPACITY)
    return EntitySelectorQueries(
        requested=jnp.zeros(query, dtype=jnp.bool_),
        source_slot=jnp.full(query, -1, dtype=jnp.int32),
        source_generation=jnp.zeros(query, dtype=jnp.uint32),
        candidate_mask=jnp.zeros(
            query + (ENTITY_CAPACITY,),
            dtype=jnp.bool_,
        ),
        ignore_owner=jnp.ones(query, dtype=jnp.bool_),
        friendly_fire=jnp.ones(query, dtype=jnp.bool_),
        max_targets=jnp.zeros(query, dtype=jnp.int32),
    )


def empty_entity_damage_payloads(
    batch_size: int,
) -> EntityDamagePayloads:
    batch = _positive_size(batch_size, "batch_size")
    query = (batch, SELECTOR_CAPACITY)
    f32 = jnp.zeros(query, dtype=jnp.float32)
    return EntityDamagePayloads(
        requested=jnp.zeros(query, dtype=jnp.bool_),
        amount=f32,
        random_percentage=f32,
        damage_class=jnp.zeros(query, dtype=jnp.int32),
        cause=jnp.zeros(query, dtype=jnp.int32),
        knockback_velocity=jnp.zeros(
            query + (ENTITY_CAPACITY, 3),
            dtype=jnp.float32,
        ),
        force_mode=jnp.zeros(query, dtype=jnp.int32),
        air_resistance=jnp.ones(query, dtype=jnp.float32),
        air_resistance_max=jnp.ones(query, dtype=jnp.float32),
        ground_resistance=jnp.ones(query, dtype=jnp.float32),
        ground_resistance_max=jnp.ones(query, dtype=jnp.float32),
        resistance_threshold=jnp.ones(query, dtype=jnp.float32),
        resistance_style=jnp.zeros(query, dtype=jnp.int32),
        dampen_y=jnp.zeros(query, dtype=jnp.bool_),
        on_hit_resource_id=jnp.full(query, -1, dtype=jnp.int32),
        on_hit_resource_delta=f32,
    )


def empty_entity_status_payloads(
    batch_size: int,
) -> EntityStatusPayloads:
    batch = _positive_size(batch_size, "batch_size")
    query = (batch, SELECTOR_CAPACITY)
    f32 = jnp.zeros(query, dtype=jnp.float32)
    return EntityStatusPayloads(
        requested=jnp.zeros(query, dtype=jnp.bool_),
        effect_id=jnp.zeros(query, dtype=jnp.int32),
        duration_seconds=f32,
        cycle_cooldown_seconds=f32,
        damage_per_cycle=f32,
        damage_cause=jnp.zeros(query, dtype=jnp.int32),
        healing_per_cycle=f32,
        resource_id=jnp.full(query, -1, dtype=jnp.int32),
        resource_delta_per_cycle=f32,
        speed_multiplier=jnp.ones(query, dtype=jnp.float32),
        flags=jnp.zeros(query, dtype=jnp.uint32),
        overlap_mode=jnp.zeros(query, dtype=jnp.int32),
    )


def empty_entity_combat_state(
    batch_size: int,
    rules: CombatMechanicsRules,
) -> EntityCombatState:
    batch = _positive_size(batch_size, "batch_size")
    if rules.resource_maximum.shape[:2] != (batch, ENTITY_CAPACITY):
        raise ValueError(
            "mechanics rules must have shape "
            f"[{batch},{ENTITY_CAPACITY},...]"
        )
    return EntityCombatState(
        roster=empty_entity_roster(batch),
        mechanics=empty_mechanics_state(batch, rules),
    )


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
