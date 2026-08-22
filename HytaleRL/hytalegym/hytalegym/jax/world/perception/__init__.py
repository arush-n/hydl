"""Provider-neutral Hytale 0.5.7 perception math."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


NATIVE_VIEW_MAX_FULL_ANGLE_RADIANS = math.tau
_COLOCATED_SQUARED = jnp.float32(1.0e-6)
_PI = jnp.float32(math.pi)
_TWO_PI = jnp.float32(math.tau)
_EPSILON = jnp.float32(1.0e-6)


def native_view_sector(
    horizontal_delta: jax.Array,
    normalized_forward_xz: jax.Array,
    full_angle_radians: jax.Array,
) -> jax.Array:
    """Reproduce NPCPhysicsMath.inViewSector on broadcast X/Z arrays."""

    squared = jnp.sum(horizontal_delta * horizontal_delta, axis=-1)
    length = jnp.sqrt(jnp.maximum(squared, 0.0))
    cosine = jnp.sum(
        horizontal_delta * normalized_forward_xz,
        axis=-1,
    ) / jnp.maximum(length, _EPSILON)
    colocated = squared <= _COLOCATED_SQUARED
    narrow = colocated | (cosine >= jnp.cos(full_angle_radians * jnp.float32(0.5)))
    complement = _TWO_PI - full_angle_radians
    rear = colocated | (-cosine >= jnp.cos(complement * jnp.float32(0.5)))
    return jnp.where(full_angle_radians > _PI, ~rear, narrow)


__all__ = [
    "NATIVE_VIEW_MAX_FULL_ANGLE_RADIANS",
    "native_view_sector",
]
