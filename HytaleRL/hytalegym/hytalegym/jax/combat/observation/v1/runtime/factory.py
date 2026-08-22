"""Host-side factories for fixed-shape observation inputs."""

from __future__ import annotations

import operator

import jax.numpy as jnp

from hytalegym.jax.combat.observation.v1.schema.contract import (
    DOOR_INTENT_COUNT,
    ENTITY_FLOAT_SIZE,
    ENTITY_INTEGER_SIZE,
    HAZARD_CAPACITY,
    HAZARD_FLOAT_SIZE,
    HAZARD_INTEGER_SIZE,
    INTERACTION_CAPACITY,
    INTERACTION_FLOAT_SIZE,
    NEARBY_ENTITY_CAPACITY,
    PROJECTILE_CAPACITY,
    PROJECTILE_FLOAT_SIZE,
    PROJECTILE_INTEGER_SIZE,
    TERRAIN_FLOAT_SIZE,
    TERRAIN_TOKEN_CAPACITY,
    TRAVERSAL_FLOAT_SIZE,
    TRAVERSAL_TOKEN_CAPACITY,
)
from hytalegym.jax.combat.observation.v1.schema.types import (
    CombatSceneFeatures,
    InjectedWorldFeatures,
)


def empty_combat_scene(batch_size: int) -> CombatSceneFeatures:
    """Return a completely masked combat scene."""

    batch = _batch_size(batch_size)
    return CombatSceneFeatures(
        entity_f32=jnp.zeros(
            (batch, NEARBY_ENTITY_CAPACITY, ENTITY_FLOAT_SIZE),
            dtype=jnp.float32,
        ),
        entity_i32=jnp.zeros(
            (batch, NEARBY_ENTITY_CAPACITY, ENTITY_INTEGER_SIZE),
            dtype=jnp.int32,
        ),
        entity_mask=jnp.zeros(
            (batch, NEARBY_ENTITY_CAPACITY),
            dtype=jnp.bool_,
        ),
        entity_overflow=jnp.zeros((batch,), dtype=jnp.bool_),
        projectile_f32=jnp.zeros(
            (batch, PROJECTILE_CAPACITY, PROJECTILE_FLOAT_SIZE),
            dtype=jnp.float32,
        ),
        projectile_i32=jnp.zeros(
            (batch, PROJECTILE_CAPACITY, PROJECTILE_INTEGER_SIZE),
            dtype=jnp.int32,
        ),
        projectile_mask=jnp.zeros(
            (batch, PROJECTILE_CAPACITY),
            dtype=jnp.bool_,
        ),
        projectile_overflow=jnp.zeros((batch,), dtype=jnp.bool_),
        hazard_f32=jnp.zeros(
            (batch, HAZARD_CAPACITY, HAZARD_FLOAT_SIZE),
            dtype=jnp.float32,
        ),
        hazard_i32=jnp.zeros(
            (batch, HAZARD_CAPACITY, HAZARD_INTEGER_SIZE),
            dtype=jnp.int32,
        ),
        hazard_mask=jnp.zeros(
            (batch, HAZARD_CAPACITY),
            dtype=jnp.bool_,
        ),
        hazard_overflow=jnp.zeros((batch,), dtype=jnp.bool_),
    )


def empty_injected_world_features(
    batch_size: int,
) -> InjectedWorldFeatures:
    """Return masked mock world inputs suitable for tests and flat combat."""

    batch = _batch_size(batch_size)
    return InjectedWorldFeatures(
        terrain_f32=jnp.zeros(
            (batch, TERRAIN_TOKEN_CAPACITY, TERRAIN_FLOAT_SIZE),
            dtype=jnp.float32,
        ),
        terrain_semantic_id=jnp.zeros(
            (batch, TERRAIN_TOKEN_CAPACITY),
            dtype=jnp.int32,
        ),
        terrain_flags=jnp.zeros(
            (batch, TERRAIN_TOKEN_CAPACITY),
            dtype=jnp.uint32,
        ),
        terrain_mask=jnp.zeros(
            (batch, TERRAIN_TOKEN_CAPACITY),
            dtype=jnp.bool_,
        ),
        terrain_overflow=jnp.zeros((batch,), dtype=jnp.bool_),
        traversal_f32=jnp.zeros(
            (batch, TRAVERSAL_TOKEN_CAPACITY, TRAVERSAL_FLOAT_SIZE),
            dtype=jnp.float32,
        ),
        traversal_id=jnp.zeros(
            (batch, TRAVERSAL_TOKEN_CAPACITY),
            dtype=jnp.int32,
        ),
        traversal_flags=jnp.zeros(
            (batch, TRAVERSAL_TOKEN_CAPACITY),
            dtype=jnp.uint32,
        ),
        traversal_mask=jnp.zeros(
            (batch, TRAVERSAL_TOKEN_CAPACITY),
            dtype=jnp.bool_,
        ),
        traversal_overflow=jnp.zeros((batch,), dtype=jnp.bool_),
        interaction_f32=jnp.zeros(
            (batch, INTERACTION_CAPACITY, INTERACTION_FLOAT_SIZE),
            dtype=jnp.float32,
        ),
        interaction_object_id=jnp.zeros(
            (batch, INTERACTION_CAPACITY),
            dtype=jnp.int32,
        ),
        interaction_is_door=jnp.zeros(
            (batch, INTERACTION_CAPACITY),
            dtype=jnp.bool_,
        ),
        interaction_door_intent_mask=jnp.zeros(
            (batch, INTERACTION_CAPACITY, DOOR_INTENT_COUNT),
            dtype=jnp.bool_,
        ),
        interaction_mask=jnp.zeros(
            (batch, INTERACTION_CAPACITY),
            dtype=jnp.bool_,
        ),
        interaction_overflow=jnp.zeros((batch,), dtype=jnp.bool_),
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
