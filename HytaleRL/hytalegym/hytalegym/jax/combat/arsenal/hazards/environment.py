"""Convert exact World contacts into source-less Environment damage events."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.mechanics import (
    DAMAGE_ENVIRONMENT,
    DamageEvents,
    empty_damage_events,
)
from hytalegym.jax.combat.types import AGENT_ENTITY
from hytalegym.jax.world import GeometryProvider
from hytalegym.jax.world.hazards import environment_damage_contact


Array = jax.Array


class EnvironmentDamageBatch(NamedTuple):
    """Entity-major event packet plus query-completeness evidence."""

    events: DamageEvents
    amount: Array
    block_damage: Array
    fluid_damage: Array
    geometry_exhausted: Array


def environment_damage_events(
    geometry: GeometryProvider | None,
    position: Array,
    health: Array,
) -> EnvironmentDamageBatch:
    """Build one null-source Environment packet per active entity.

    Hytale evaluates this contact every 30 Hz server step. Magnitudes remain
    geometry-owned, while Combat supplies the existing cause/resistance/death
    pipeline. No weapon, role, or asset-name branch is introduced here.
    """

    points = jnp.asarray(position, dtype=jnp.float32)
    current_health = jnp.asarray(health, dtype=jnp.float32)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("position must have shape [batch, entity, 3]")
    if current_health.shape != points.shape[:2]:
        raise ValueError("health must match the position entity axis")
    batch, entity_count = current_health.shape
    events = empty_damage_events(batch, entity_count)
    if geometry is None:
        zeros = jnp.zeros((batch, entity_count), dtype=jnp.float32)
        return EnvironmentDamageBatch(
            events=events,
            amount=zeros,
            block_damage=zeros.astype(jnp.int32),
            fluid_damage=zeros.astype(jnp.int32),
            geometry_exhausted=jnp.zeros((batch,), dtype=jnp.bool_),
        )

    agent_bounds = _broadcast_bounds(geometry.agent_bounds, batch)
    target_bounds = _broadcast_bounds(geometry.target_bounds, batch)
    entity_bounds = jnp.broadcast_to(
        target_bounds[:, None, :],
        (batch, entity_count, 6),
    ).at[:, AGENT_ENTITY].set(agent_bounds)
    contact = jax.vmap(
        lambda entity_position, bounds: environment_damage_contact(
            geometry,
            entity_position,
            bounds=bounds,
        ),
        in_axes=(1, 1),
        out_axes=1,
    )(points, entity_bounds)
    requested = contact.requested & (current_health > 0.0)
    target = jnp.broadcast_to(
        jnp.arange(entity_count, dtype=jnp.int32)[None, :],
        (batch, entity_count),
    )
    events = events._replace(
        requested=requested,
        target_entity_id=target,
        amount=jnp.where(requested, contact.amount, jnp.float32(0.0)),
        cause=jnp.full(
            (batch, entity_count),
            DAMAGE_ENVIRONMENT,
            dtype=jnp.int32,
        ),
    )
    return EnvironmentDamageBatch(
        events=events,
        amount=contact.amount,
        block_damage=contact.block_damage,
        fluid_damage=contact.fluid_damage,
        geometry_exhausted=jnp.any(contact.geometry_exhausted, axis=1),
    )


def _broadcast_bounds(bounds: Array, batch: int) -> Array:
    values = jnp.asarray(bounds, dtype=jnp.float32)
    if values.ndim == 1 and values.shape == (6,):
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("geometry bounds must have shape [1|batch, 6]")
    if values.shape[0] not in (1, batch):
        raise ValueError("geometry bounds batch must be one or match combat")
    if values.shape[0] == 1 and batch != 1:
        values = jnp.broadcast_to(values, (batch, 6))
    return values


__all__ = ["EnvironmentDamageBatch", "environment_damage_events"]
