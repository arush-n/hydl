"""Reset factories for fixed-shape opponent memory."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.opponents.schema.contract import (
    OPPONENT_MODE_INACTIVE,
    SEARCH_TIMEOUT_MAX_SECONDS,
    SEARCH_TIMEOUT_MIN_SECONDS,
)
from hytalegym.jax.combat.opponents.schema.types import OpponentMemoryState


def reset_opponent_memory(
    keys: jax.Array,
    position: jax.Array,
    health: jax.Array,
    controlled: jax.Array,
) -> OpponentMemoryState:
    """Create per-entity home state and episode-keyed Search timeouts."""

    if position.ndim != 3 or position.shape[2] != 3:
        raise ValueError("position must have shape (batch, entity, 3)")
    batch, entities, _ = position.shape
    if keys.shape[0] != batch:
        raise ValueError("keys must match position's batch axis")
    if health.shape != (batch, entities):
        raise ValueError("health must match position's batch and entity axes")
    if controlled.shape != (batch, entities):
        raise ValueError("controlled must match position's batch and entity axes")
    if controlled.dtype != jnp.dtype(jnp.bool_):
        raise ValueError("controlled must have boolean dtype")

    entity_ids = jnp.arange(entities, dtype=jnp.int32)

    def sample_row(key: jax.Array) -> jax.Array:
        return jax.vmap(
            lambda entity_id: jax.random.uniform(
                jax.random.fold_in(key, entity_id),
                (),
                minval=jnp.float32(SEARCH_TIMEOUT_MIN_SECONDS),
                maxval=jnp.float32(SEARCH_TIMEOUT_MAX_SECONDS),
                dtype=jnp.float32,
            )
        )(entity_ids)

    timeout = jax.vmap(sample_row)(keys)
    active = controlled & (health > jnp.float32(0.0))
    return OpponentMemoryState(
        mode=jnp.full(
            (batch, entities),
            OPPONENT_MODE_INACTIVE,
            dtype=jnp.int32,
        ),
        home_position=jnp.where(
            active[..., None],
            position,
            jnp.float32(0.0),
        ),
        home_valid=active,
        last_seen_position=jnp.zeros_like(position, dtype=jnp.float32),
        last_seen_valid=jnp.zeros((batch, entities), dtype=jnp.bool_),
        pursuit_elapsed_ticks=jnp.zeros(
            (batch, entities),
            dtype=jnp.int32,
        ),
        search_elapsed_seconds=jnp.zeros(
            (batch, entities),
            dtype=jnp.float32,
        ),
        search_timeout_seconds=jnp.where(
            active,
            timeout,
            jnp.float32(0.0),
        ),
        navigation_unavailable=jnp.zeros(
            (batch, entities),
            dtype=jnp.bool_,
        ),
    )


__all__ = ["reset_opponent_memory"]
