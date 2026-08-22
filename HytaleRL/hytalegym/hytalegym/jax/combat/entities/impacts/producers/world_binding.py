"""Bind complete public World results into logical impact query tensors.

This module deliberately does not import ``jax.world``. The World lane owns
physical explosion admission; Combat accepts its fixed-shape result fields
and preserves every diagnostic in the separate world-failure domain.
"""

from __future__ import annotations

import jax.numpy as jnp

from hytalegym.jax.combat.entities.impacts.schema.contract import (
    EXPLOSION_QUERY_FAILURE_CAPACITY_EXCEEDED,
    EXPLOSION_QUERY_FAILURE_DAMAGE_BLOCKS_UNSUPPORTED,
    EXPLOSION_QUERY_FAILURE_GEOMETRY_EXHAUSTED,
    EXPLOSION_QUERY_FAILURE_INVALID,
)
from hytalegym.jax.combat.entities.impacts.schema.types import EntityImpactQueries


def bind_entity_only_explosion_candidates(
    queries: EntityImpactQueries,
    *,
    query_mask,
    available,
    candidate_mask,
    distance,
    candidate_generation,
    geometry_exhausted,
    capacity_exceeded,
    damage_blocks_unsupported,
    invalid,
) -> EntityImpactQueries:
    """Merge entity-only explosion admission without altering other queries.

    Inputs correspond one-for-one to the public
    ``region_entity_only_explosion_candidates`` result plus the
    Combat-owned logical-slot generation tensor. Requested rows are valid only
    when the pre-existing projectile query and the complete World result are
    both valid. Unknown or malformed evidence closes the row.
    """

    projectile_shape = queries.projectile_query_valid.shape
    projectile_entity_shape = queries.projectile_explosion_candidate_mask.shape
    _field(query_mask, projectile_shape, jnp.bool_, "query_mask")
    _field(available, projectile_shape, jnp.bool_, "available")
    _field(
        candidate_mask,
        projectile_entity_shape,
        jnp.bool_,
        "candidate_mask",
    )
    _field(distance, projectile_entity_shape, jnp.float32, "distance")
    _field(
        candidate_generation,
        projectile_entity_shape,
        jnp.uint32,
        "candidate_generation",
    )
    _field(
        geometry_exhausted,
        projectile_shape,
        jnp.bool_,
        "geometry_exhausted",
    )
    _field(
        capacity_exceeded,
        projectile_shape,
        jnp.bool_,
        "capacity_exceeded",
    )
    _field(
        damage_blocks_unsupported,
        projectile_shape,
        jnp.bool_,
        "damage_blocks_unsupported",
    )
    _field(invalid, projectile_shape, jnp.bool_, "invalid")

    requested = jnp.asarray(query_mask, dtype=jnp.bool_)
    admitted = jnp.asarray(candidate_mask, dtype=jnp.bool_)
    measured_distance = jnp.asarray(distance, dtype=jnp.float32)
    malformed_payload = requested & (
        jnp.any(
            admitted
            & (
                ~jnp.isfinite(measured_distance)
                | (measured_distance < jnp.float32(0.0))
            ),
            axis=2,
        )
        | (~available & jnp.any(admitted, axis=2))
    )
    invalid_result = invalid | malformed_payload
    failure_bits = (
        jnp.where(
            requested & geometry_exhausted,
            jnp.uint32(EXPLOSION_QUERY_FAILURE_GEOMETRY_EXHAUSTED),
            jnp.uint32(0),
        )
        | jnp.where(
            requested & capacity_exceeded,
            jnp.uint32(EXPLOSION_QUERY_FAILURE_CAPACITY_EXCEEDED),
            jnp.uint32(0),
        )
        | jnp.where(
            requested & damage_blocks_unsupported,
            jnp.uint32(EXPLOSION_QUERY_FAILURE_DAMAGE_BLOCKS_UNSUPPORTED),
            jnp.uint32(0),
        )
        | jnp.where(
            requested & invalid_result,
            jnp.uint32(EXPLOSION_QUERY_FAILURE_INVALID),
            jnp.uint32(0),
        )
    )
    complete = (
        requested
        & available
        & ~geometry_exhausted
        & ~capacity_exceeded
        & ~damage_blocks_unsupported
        & ~invalid_result
    )
    bound_candidates = admitted & complete[..., None]
    return queries._replace(
        projectile_explosion_candidate_mask=jnp.where(
            requested[..., None],
            bound_candidates,
            queries.projectile_explosion_candidate_mask,
        ),
        projectile_explosion_distance=jnp.where(
            requested[..., None],
            jnp.where(
                bound_candidates,
                measured_distance,
                jnp.float32(0.0),
            ),
            queries.projectile_explosion_distance,
        ),
        projectile_candidate_generation=jnp.where(
            requested[..., None],
            candidate_generation,
            queries.projectile_candidate_generation,
        ),
        projectile_query_valid=jnp.where(
            requested,
            queries.projectile_query_valid & complete,
            queries.projectile_query_valid,
        ),
        projectile_failure_bits=(
            queries.projectile_failure_bits | failure_bits
        ),
    )


def _field(value, shape, dtype, name: str) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}")


__all__ = [
    "EXPLOSION_QUERY_FAILURE_CAPACITY_EXCEEDED",
    "EXPLOSION_QUERY_FAILURE_DAMAGE_BLOCKS_UNSUPPORTED",
    "EXPLOSION_QUERY_FAILURE_GEOMETRY_EXHAUSTED",
    "EXPLOSION_QUERY_FAILURE_INVALID",
    "bind_entity_only_explosion_candidates",
]
