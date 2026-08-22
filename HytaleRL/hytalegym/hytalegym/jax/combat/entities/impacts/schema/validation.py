"""Layout validation for logical projectile and area integration."""

from __future__ import annotations

import jax.numpy as jnp

from hytalegym.jax.combat.arsenal import AREA_CAPACITY, PROJECTILE_CAPACITY
from hytalegym.jax.combat.entities import ENTITY_CAPACITY
from hytalegym.jax.combat.entities.effects import validate_effect_layout
from hytalegym.jax.combat.entities.impacts.schema.types import (
    EntityImpactBindings,
    EntityImpactQueries,
    EntityImpactState,
)


def validate_impact_layout(
    state: EntityImpactState,
    bindings: EntityImpactBindings,
    queries: EntityImpactQueries,
) -> None:
    validate_effect_layout(state.effects)
    batch = state.effects.combat.roster.active.shape[0]
    _field(state.capability_bits, (batch,), jnp.uint32, "capability_bits")
    _field(state.failure_bits, (batch,), jnp.uint32, "failure_bits")
    _field(
        state.world_failure_bits,
        (batch,),
        jnp.uint32,
        "world_failure_bits",
    )
    projectile = (batch, PROJECTILE_CAPACITY)
    area = (batch, AREA_CAPACITY)
    projectile_entity = projectile + (ENTITY_CAPACITY,)
    for name, shape, dtype in (
        ("projectile_source_generation", projectile, jnp.uint32),
        ("projectile_interaction_kind", projectile, jnp.int32),
        ("projectile_damage_multiplier", projectile, jnp.float32),
        (
            "projectile_knockback_yaw_degrees",
            projectile,
            jnp.float32,
        ),
        ("projectile_friendly_fire", projectile, jnp.bool_),
        ("area_source_generation", area, jnp.uint32),
        ("area_friendly_fire", area, jnp.bool_),
        ("overflow", (batch,), jnp.bool_),
    ):
        _field(getattr(bindings, name), shape, dtype, f"bindings.{name}")
    for name, shape, dtype in (
        (
            "projectile_entity_hit_mask",
            projectile_entity,
            jnp.bool_,
        ),
        (
            "projectile_entity_hit_fraction",
            projectile_entity,
            jnp.float32,
        ),
        (
            "projectile_candidate_generation",
            projectile_entity,
            jnp.uint32,
        ),
        ("projectile_world_hit", projectile, jnp.bool_),
        (
            "projectile_world_hit_fraction",
            projectile,
            jnp.float32,
        ),
        (
            "projectile_explosion_candidate_mask",
            projectile_entity,
            jnp.bool_,
        ),
        (
            "projectile_explosion_distance",
            projectile_entity,
            jnp.float32,
        ),
        ("projectile_query_valid", projectile, jnp.bool_),
        ("projectile_failure_bits", projectile, jnp.uint32),
        ("area_candidate_mask", area + (ENTITY_CAPACITY,), jnp.bool_),
        (
            "area_candidate_generation",
            area + (ENTITY_CAPACITY,),
            jnp.uint32,
        ),
        ("area_query_valid", area, jnp.bool_),
        ("area_failure_bits", area, jnp.uint32),
    ):
        _field(getattr(queries, name), shape, dtype, f"queries.{name}")


def _field(value, shape, dtype, name):
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}")
