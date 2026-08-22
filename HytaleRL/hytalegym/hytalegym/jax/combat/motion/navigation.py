"""Fail-closed transport for World-certified target navigation steps."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class TargetNavigationStep(NamedTuple):
    """One exact target motion result produced by a certified World path."""

    certified_available: jax.Array
    position: jax.Array
    velocity: jax.Array
    geometry_exhausted: jax.Array


def validate_target_navigation_step(
    step: TargetNavigationStep,
    batch_size: int,
) -> TargetNavigationStep:
    """Validate and sanitize one provider row without inventing availability."""

    if not isinstance(step, TargetNavigationStep):
        raise TypeError("navigation provider must return TargetNavigationStep")
    available = jnp.asarray(step.certified_available, dtype=jnp.bool_)
    position = jnp.asarray(step.position, dtype=jnp.float32)
    velocity = jnp.asarray(step.velocity, dtype=jnp.float32)
    exhausted = jnp.asarray(step.geometry_exhausted, dtype=jnp.bool_)
    if available.shape != (batch_size,):
        raise ValueError(
            "navigation certified_available must have shape (batch,)"
        )
    if position.shape != (batch_size, 3):
        raise ValueError("navigation position must have shape (batch, 3)")
    if velocity.shape != (batch_size, 3):
        raise ValueError("navigation velocity must have shape (batch, 3)")
    if exhausted.shape != (batch_size,):
        raise ValueError(
            "navigation geometry_exhausted must have shape (batch,)"
        )
    finite = jnp.all(
        jnp.isfinite(position) & jnp.isfinite(velocity),
        axis=1,
    )
    certified = available & finite & ~exhausted
    return TargetNavigationStep(
        certified_available=certified,
        position=jnp.where(certified[:, None], position, 0.0),
        velocity=jnp.where(certified[:, None], velocity, 0.0),
        geometry_exhausted=exhausted,
    )


__all__ = [
    "TargetNavigationStep",
    "validate_target_navigation_step",
]
