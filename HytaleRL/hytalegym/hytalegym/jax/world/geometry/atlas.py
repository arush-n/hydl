"""JIT-safe selection of overlapping exact native geometry tiles."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.geometry.contract import CELL_RADIUS
from hytalegym.jax.world.types import GeometryAtlas, GeometryState


Array = jax.Array


class AtlasSelection(NamedTuple):
    geometry: GeometryState
    tile_index: Array
    available: Array


def select_geometry_tile(
    atlas: GeometryAtlas,
    entity_positions: Array,
    environment_world_id: Array,
    *,
    interior_margin: int = 1,
) -> AtlasSelection:
    """Select one complete tile per environment with pure array operations.

    ``entity_positions`` is ``[B,E,3]``. A tile is eligible only when every
    entity's floored cell lies inside its interior and its world ID matches.
    The nearest eligible origin wins deterministically; absence is explicit.
    """

    if entity_positions.ndim != 3 or entity_positions.shape[-1] != 3:
        raise ValueError("entity_positions must have shape (batch, entities, 3)")
    batch = entity_positions.shape[0]
    world_ids = jnp.asarray(environment_world_id, dtype=jnp.int32)
    if world_ids.shape != (batch,):
        raise ValueError("environment_world_id must have shape (batch,)")
    margin = int(interior_margin)
    if not 0 <= margin < CELL_RADIUS:
        raise ValueError("interior_margin must be in [0, CELL_RADIUS)")

    origin = atlas.geometry.origin
    relative = (
        jnp.floor(entity_positions).astype(jnp.int32)[:, None, :, :]
        - origin[None, :, None, :]
    )
    limit = jnp.int32(CELL_RADIUS - margin)
    inside = jnp.all(jnp.abs(relative) <= limit, axis=(2, 3))
    compatible = (
        atlas.tile_mask[None, :]
        & (atlas.world_id[None, :] == world_ids[:, None])
    )
    eligible = inside & compatible

    center = origin.astype(jnp.float32) + jnp.float32(0.5)
    scenario_center = jnp.mean(entity_positions, axis=1)
    distance_squared = jnp.sum(
        (scenario_center[:, None, :] - center[None, :, :]) ** 2,
        axis=2,
    )
    score = jnp.where(eligible, distance_squared, jnp.float32(jnp.inf))
    index = jnp.argmin(score, axis=1).astype(jnp.int32)
    available = jnp.any(eligible, axis=1)
    safe_index = jnp.where(available, index, jnp.int32(0))

    selected = GeometryState(
        *(
            None if leaf is None else jnp.take(leaf, safe_index, axis=0)
            for leaf in atlas.geometry
        )
    )
    return AtlasSelection(
        geometry=selected,
        tile_index=jnp.where(available, index, jnp.int32(-1)),
        available=available,
    )
