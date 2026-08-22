"""Factories for Crossbow projectile-impact handoff arrays."""

from __future__ import annotations

import operator

import jax.numpy as jnp

from hytalegym.jax.combat.entities.interactions.schema.contract import (
    CROSSBOW_IMPACT_CAPACITY,
    CROSSBOW_IMPACT_STANDARD,
)
from hytalegym.jax.combat.arsenal.projectiles.crossbow.types import (
    CrossbowProjectileImpactCommands,
)


def empty_crossbow_projectile_impact_commands(
    batch_size: int,
    projectile_capacity: int,
) -> CrossbowProjectileImpactCommands:
    batch = _size(batch_size, "batch_size")
    width = _size(projectile_capacity, "projectile_capacity")
    if width > CROSSBOW_IMPACT_CAPACITY:
        raise ValueError("projectile capacity exceeds Crossbow impact capacity")
    shape = (batch, width)
    return CrossbowProjectileImpactCommands(
        requested=jnp.zeros(shape, dtype=jnp.bool_),
        kind=jnp.full(shape, CROSSBOW_IMPACT_STANDARD, dtype=jnp.int32),
        source_slot=jnp.full(shape, -1, dtype=jnp.int32),
        target_slot=jnp.full(shape, -1, dtype=jnp.int32),
        knockback_yaw_degrees=jnp.zeros(shape, dtype=jnp.float32),
        standard_damage=jnp.zeros(shape, dtype=jnp.float32),
        combo_damage=jnp.zeros(shape, dtype=jnp.float32),
        big_arrow_damage=jnp.zeros(shape, dtype=jnp.float32),
    )


def _size(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result
