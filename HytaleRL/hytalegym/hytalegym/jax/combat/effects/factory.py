"""Host factories for empty fixed-shape combat-effect trees."""

from __future__ import annotations

import operator

import jax.numpy as jnp

from hytalegym.jax.combat.effects.schema.contract import (
    HAZARD_CAPACITY,
    PROJECTILE_CAPACITY,
)
from hytalegym.jax.combat.effects.schema.types import (
    CombatEffectCommands,
    CombatEffectsState,
    HazardSpawn,
    HazardState,
    ProjectileSpawn,
    ProjectileState,
)


def empty_effects_state(batch_size: int) -> CombatEffectsState:
    """Return a masked effect state with no sticky failures."""

    batch = _batch_size(batch_size)
    return CombatEffectsState(
        projectiles=_empty_projectiles(batch),
        hazards=_empty_hazards(batch),
        failure_bits=jnp.zeros((batch,), dtype=jnp.uint32),
    )


def empty_effect_commands(batch_size: int) -> CombatEffectCommands:
    """Return one unrequested projectile and hazard command per row."""

    batch = _batch_size(batch_size)
    return CombatEffectCommands(
        projectile=ProjectileSpawn(
            requested=jnp.zeros((batch,), dtype=jnp.bool_),
            position=jnp.zeros((batch, 3), dtype=jnp.float32),
            velocity=jnp.zeros((batch, 3), dtype=jnp.float32),
            half_extent=jnp.zeros((batch, 3), dtype=jnp.float32),
            despawn_seconds=jnp.zeros((batch,), dtype=jnp.float32),
            authored_lifetime_seconds=jnp.zeros((batch,), dtype=jnp.float32),
            damage=jnp.zeros((batch,), dtype=jnp.float32),
            gravity=jnp.zeros((batch,), dtype=jnp.float32),
            terminal_velocity=jnp.zeros((batch,), dtype=jnp.float32),
            dead_time_seconds=jnp.zeros((batch,), dtype=jnp.float32),
            velocity_scale=jnp.zeros((batch,), dtype=jnp.float32),
            kind=jnp.zeros((batch,), dtype=jnp.int32),
            owner_entity_id=jnp.zeros((batch,), dtype=jnp.int32),
            flags=jnp.zeros((batch,), dtype=jnp.uint32),
            hostile=jnp.zeros((batch,), dtype=jnp.bool_),
            entity_collision_only=jnp.zeros((batch,), dtype=jnp.bool_),
        ),
        hazard=HazardSpawn(
            requested=jnp.zeros((batch,), dtype=jnp.bool_),
            center=jnp.zeros((batch, 3), dtype=jnp.float32),
            half_extent=jnp.zeros((batch, 3), dtype=jnp.float32),
            duration_seconds=jnp.zeros((batch,), dtype=jnp.float32),
            damage_per_second=jnp.zeros((batch,), dtype=jnp.float32),
            intensity=jnp.zeros((batch,), dtype=jnp.float32),
            activation=jnp.zeros((batch,), dtype=jnp.float32),
            kind=jnp.zeros((batch,), dtype=jnp.int32),
            owner_entity_id=jnp.zeros((batch,), dtype=jnp.int32),
            flags=jnp.zeros((batch,), dtype=jnp.uint32),
            hostile=jnp.zeros((batch,), dtype=jnp.bool_),
            entity_overlap_only=jnp.zeros((batch,), dtype=jnp.bool_),
        ),
    )


def _empty_projectiles(batch: int) -> ProjectileState:
    vectors = jnp.zeros((batch, PROJECTILE_CAPACITY, 3), dtype=jnp.float32)
    floats = jnp.zeros((batch, PROJECTILE_CAPACITY), dtype=jnp.float32)
    integers = jnp.zeros((batch, PROJECTILE_CAPACITY), dtype=jnp.int32)
    unsigned = jnp.zeros((batch, PROJECTILE_CAPACITY), dtype=jnp.uint32)
    booleans = jnp.zeros((batch, PROJECTILE_CAPACITY), dtype=jnp.bool_)
    return ProjectileState(
        position=vectors,
        velocity=vectors,
        half_extent=vectors,
        age_seconds=floats,
        despawn_seconds=floats,
        authored_lifetime_seconds=floats,
        damage=floats,
        gravity=floats,
        terminal_velocity=floats,
        dead_time_seconds=floats,
        dead_time_remaining=floats,
        velocity_scale=floats,
        kind=integers,
        owner_entity_id=integers,
        flags=unsigned,
        hostile=booleans,
        active=booleans,
        impacted=booleans,
        physics_initialized=booleans,
        entity_collision_only=booleans,
    )


def _empty_hazards(batch: int) -> HazardState:
    vectors = jnp.zeros((batch, HAZARD_CAPACITY, 3), dtype=jnp.float32)
    floats = jnp.zeros((batch, HAZARD_CAPACITY), dtype=jnp.float32)
    integers = jnp.zeros((batch, HAZARD_CAPACITY), dtype=jnp.int32)
    unsigned = jnp.zeros((batch, HAZARD_CAPACITY), dtype=jnp.uint32)
    booleans = jnp.zeros((batch, HAZARD_CAPACITY), dtype=jnp.bool_)
    return HazardState(
        center=vectors,
        half_extent=vectors,
        age_seconds=floats,
        duration_seconds=floats,
        damage_per_second=floats,
        intensity=floats,
        activation=floats,
        kind=integers,
        owner_entity_id=integers,
        flags=unsigned,
        hostile=booleans,
        active=booleans,
        entity_overlap_only=booleans,
    )


def _batch_size(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("batch_size must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError("batch_size must be a positive integer") from error
    if result <= 0:
        raise ValueError("batch_size must be a positive integer")
    return result
