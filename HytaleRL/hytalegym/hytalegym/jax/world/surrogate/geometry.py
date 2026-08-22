"""JIT-safe multi-tile block and entity swept-AABB queries."""

from __future__ import annotations

import operator

import jax
import jax.numpy as jnp

from hytalegym.jax.world.surrogate.atlas import lookup_surrogate_blocks
from hytalegym.jax.world.surrogate.capabilities import (
    SURROGATE_CAPABILITY_ENTITY_GEOMETRY,
    SURROGATE_CAPABILITY_SWEPT_AABB,
    SURROGATE_QUERY_CAPACITY,
    SURROGATE_QUERY_COVERAGE,
    SURROGATE_QUERY_ENTITY_GEOMETRY,
    SURROGATE_QUERY_INVALID,
    SURROGATE_QUERY_RUNTIME,
)
from hytalegym.jax.world.surrogate.types import (
    SurrogateAtlas,
    SurrogateRuntimeState,
    SurrogateSweepResult,
)

SURROGATE_SWEEP_CELL_CAPACITY = 64
MAX_SURROGATE_SWEEP_CELL_CAPACITY = 4096
_EPSILON = jnp.float32(1.0e-6)


def query_surrogate_swept_aabb(
    atlas: SurrogateAtlas,
    runtime: SurrogateRuntimeState,
    start_position: jax.Array,
    end_position: jax.Array,
    local_bounds: jax.Array,
    *,
    include_entities: bool = True,
    ignore_entity_slot: jax.Array | None = None,
    max_cells: int = SURROGATE_SWEEP_CELL_CAPACITY,
) -> SurrogateSweepResult:
    """Test fixed query AABBs against exact current block/entity occupancy."""

    if not isinstance(atlas, SurrogateAtlas):
        raise TypeError("atlas must be a SurrogateAtlas")
    if not isinstance(runtime, SurrogateRuntimeState):
        raise TypeError("runtime must be a SurrogateRuntimeState")
    if not isinstance(include_entities, bool):
        raise TypeError("include_entities must be bool")
    capacity = _cell_capacity(max_cells)
    start = jnp.asarray(start_position, dtype=jnp.float32)
    end = jnp.asarray(end_position, dtype=jnp.float32)
    if start.ndim != 3 or start.shape[2] != 3 or end.shape != start.shape:
        raise ValueError("start_position and end_position must have shape [B, Q, 3]")
    batch, queries, _ = start.shape
    if runtime.environment_world_id.shape != (batch,):
        raise ValueError("runtime and swept queries must share a batch dimension")
    bounds = _query_bounds(local_bounds, batch, queries)
    ignored = _ignored_slots(ignore_entity_slot, batch, queries)

    finite = jnp.all(jnp.isfinite(start) & jnp.isfinite(end), axis=2) & jnp.all(
        jnp.isfinite(bounds), axis=2
    )
    ordered_bounds = jnp.all(bounds[..., :3] <= bounds[..., 3:], axis=2)
    valid = finite & ordered_bounds
    safe_start = jnp.where(valid[..., None], start, 0.0)
    safe_end = jnp.where(valid[..., None], end, 0.0)
    safe_bounds = jnp.where(valid[..., None], bounds, 0.0)

    start_minimum = safe_start + safe_bounds[..., :3]
    start_maximum = safe_start + safe_bounds[..., 3:]
    end_minimum = safe_end + safe_bounds[..., :3]
    end_maximum = safe_end + safe_bounds[..., 3:]
    swept_minimum = jnp.minimum(start_minimum, end_minimum)
    swept_maximum = jnp.maximum(start_maximum, end_maximum)
    broad_extent = jnp.floor(swept_maximum) - jnp.floor(swept_minimum) + 1.0
    broad_capacity = jnp.any(
        (broad_extent < 1.0) | (broad_extent > float(capacity)),
        axis=2,
    ) | (jnp.prod(broad_extent, axis=2) > float(capacity))
    safe_broad = valid & ~broad_capacity
    broad_minimum = jnp.where(safe_broad[..., None], swept_minimum, 0.0)
    upper = jnp.where(
        swept_maximum > swept_minimum + _EPSILON,
        swept_maximum - _EPSILON,
        swept_maximum,
    )
    broad_maximum = jnp.where(safe_broad[..., None], upper, 0.0)
    minimum_cell = jnp.floor(broad_minimum).astype(jnp.int32)
    maximum_cell = jnp.floor(broad_maximum).astype(jnp.int32)
    cell_extent = maximum_cell - minimum_cell + 1
    cell_count = jnp.prod(cell_extent, axis=2)
    capacity_exceeded = broad_capacity | (cell_count > capacity)
    safe_extent = jnp.where(
        capacity_exceeded[..., None],
        jnp.ones_like(cell_extent),
        cell_extent,
    )
    safe_count = jnp.where(capacity_exceeded, 0, cell_count)

    linear = jnp.arange(capacity, dtype=jnp.int32)
    size_x = safe_extent[..., 0, None]
    size_z = safe_extent[..., 2, None]
    offset_x = jnp.mod(linear, size_x)
    remainder = linear // size_x
    offset_z = jnp.mod(remainder, size_z)
    offset_y = remainder // size_z
    offsets = jnp.stack((offset_x, offset_y, offset_z), axis=-1)
    cells = minimum_cell[..., None, :] + offsets
    active_cells = linear < safe_count[..., None]

    selected = lookup_surrogate_blocks(
        atlas,
        runtime,
        cells.reshape(batch, queries * capacity, 3),
        require_core=False,
    )
    selected_available = selected.available.reshape(batch, queries, capacity)
    complete = jnp.all(~active_cells | selected_available, axis=2)
    box_shape = selected.collision_boxes.shape[2:]
    block_boxes = selected.collision_boxes.reshape(
        (batch, queries, capacity) + box_shape
    )
    block_box_mask = selected.collision_box_mask.reshape(
        (batch, queries, capacity) + selected.collision_box_mask.shape[2:]
    )
    world_boxes = block_boxes + jnp.concatenate((cells, cells), axis=-1)[
        ..., None, :
    ].astype(jnp.float32)
    block_hits = _swept_hits(
        safe_start[..., None, None, :],
        safe_end[..., None, None, :],
        safe_bounds[..., None, None, :],
        world_boxes,
    )
    hit_block = jnp.any(
        block_hits & block_box_mask & active_cells[..., None],
        axis=(2, 3),
    )

    has_entity_geometry = (
        runtime.capability_bits & jnp.uint32(SURROGATE_CAPABILITY_ENTITY_GEOMETRY)
    ) != 0
    entity_unsupported = include_entities & ~has_entity_geometry[:, None]
    entity_boxes = runtime.entity_position[..., None, :] + jnp.stack(
        (
            runtime.entity_local_bounds[..., :3],
            runtime.entity_local_bounds[..., 3:],
        ),
        axis=-2,
    )
    entity_boxes = jnp.concatenate(
        (entity_boxes[..., 0, :], entity_boxes[..., 1, :]),
        axis=-1,
    )
    entity_hits = _swept_hits(
        safe_start[..., None, :],
        safe_end[..., None, :],
        safe_bounds[..., None, :],
        entity_boxes[:, None, :, :],
    )
    entity_slots = jnp.arange(runtime.entity_active.shape[1], dtype=jnp.int32)
    entity_mask = (
        runtime.entity_active
        & runtime.entity_initialized
        & runtime.entity_geometry_supported
        & runtime.entity_collidable
    )
    entity_hits &= (
        include_entities
        & entity_mask[:, None, :]
        & (entity_slots[None, None, :] != ignored[..., None])
    )
    hit_entity = jnp.any(entity_hits, axis=2)
    first_entity = jnp.min(
        jnp.where(entity_hits, entity_slots[None, None, :], entity_slots.size),
        axis=2,
    )

    has_sweep = (
        runtime.capability_bits & jnp.uint32(SURROGATE_CAPABILITY_SWEPT_AABB)
    ) != 0
    runtime_invalid = runtime.unsupported_mechanics | ~has_sweep
    diagnostics = jnp.zeros((batch, queries), dtype=jnp.uint32)
    diagnostics |= jnp.where(
        ~valid,
        jnp.uint32(SURROGATE_QUERY_INVALID),
        jnp.uint32(0),
    )
    diagnostics |= jnp.where(
        capacity_exceeded,
        jnp.uint32(SURROGATE_QUERY_CAPACITY),
        jnp.uint32(0),
    )
    diagnostics |= jnp.where(
        valid & ~capacity_exceeded & ~complete,
        jnp.uint32(SURROGATE_QUERY_COVERAGE),
        jnp.uint32(0),
    )
    diagnostics |= jnp.where(
        entity_unsupported,
        jnp.uint32(SURROGATE_QUERY_ENTITY_GEOMETRY),
        jnp.uint32(0),
    )
    diagnostics |= jnp.where(
        runtime_invalid[:, None],
        jnp.uint32(SURROGATE_QUERY_RUNTIME),
        jnp.uint32(0),
    )
    available = diagnostics == 0
    return SurrogateSweepResult(
        available=available,
        clear=available & ~hit_block & ~hit_entity,
        diagnostics=diagnostics,
        visited_cell_count=jnp.where(available, safe_count, 0),
        hit_block=available & hit_block,
        hit_entity=available & hit_entity,
        hit_entity_slot=jnp.where(
            available & hit_entity,
            first_entity,
            -1,
        ),
    )


def query_surrogate_aabb_clearance(
    atlas: SurrogateAtlas,
    runtime: SurrogateRuntimeState,
    position: jax.Array,
    local_bounds: jax.Array,
    *,
    include_entities: bool = True,
    ignore_entity_slot: jax.Array | None = None,
    max_cells: int = SURROGATE_SWEEP_CELL_CAPACITY,
) -> SurrogateSweepResult:
    """Test stationary AABB occupancy through the same fail-closed kernel."""

    return query_surrogate_swept_aabb(
        atlas,
        runtime,
        position,
        position,
        local_bounds,
        include_entities=include_entities,
        ignore_entity_slot=ignore_entity_slot,
        max_cells=max_cells,
    )


def _swept_hits(
    start: jax.Array,
    end: jax.Array,
    bounds: jax.Array,
    boxes: jax.Array,
) -> jax.Array:
    delta = end - start
    lower_position = boxes[..., :3] - bounds[..., 3:] + _EPSILON
    upper_position = boxes[..., 3:] - bounds[..., :3] - _EPSILON
    moving = jnp.abs(delta) > _EPSILON
    safe_delta = jnp.where(moving, delta, 1.0)
    first = (lower_position - start) / safe_delta
    second = (upper_position - start) / safe_delta
    entry = jnp.where(moving, jnp.minimum(first, second), -jnp.inf)
    exit_time = jnp.where(moving, jnp.maximum(first, second), jnp.inf)
    stationary_inside = (~moving) & (start > lower_position) & (start < upper_position)
    axis_valid = moving | stationary_inside
    lower = jnp.max(entry, axis=-1)
    upper = jnp.min(exit_time, axis=-1)
    return (
        jnp.all(axis_valid, axis=-1)
        & (lower < upper - _EPSILON)
        & (lower < 1.0 - _EPSILON)
        & (upper > _EPSILON)
    )


def _query_bounds(value: jax.Array, batch: int, queries: int) -> jax.Array:
    bounds = jnp.asarray(value, dtype=jnp.float32)
    if bounds.shape == (6,):
        return jnp.broadcast_to(bounds, (batch, queries, 6))
    if bounds.shape == (batch, 6):
        return jnp.broadcast_to(bounds[:, None, :], (batch, queries, 6))
    if bounds.shape != (batch, queries, 6):
        raise ValueError("local_bounds must have shape [6], [B, 6], or [B, Q, 6]")
    return bounds


def _ignored_slots(
    value: jax.Array | None,
    batch: int,
    queries: int,
) -> jax.Array:
    if value is None:
        return jnp.full((batch, queries), -1, dtype=jnp.int32)
    slots = jnp.asarray(value, dtype=jnp.int32)
    if slots.shape == (batch,):
        return jnp.broadcast_to(slots[:, None], (batch, queries))
    if slots.shape != (batch, queries):
        raise ValueError("ignore_entity_slot must have shape [B] or [B, Q]")
    return slots


def _cell_capacity(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("max_cells must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError("max_cells must be an integer") from error
    if not 1 <= result <= MAX_SURROGATE_SWEEP_CELL_CAPACITY:
        raise ValueError(
            f"max_cells must be in [1, {MAX_SURROGATE_SWEEP_CELL_CAPACITY}]"
        )
    return result


__all__ = [
    "MAX_SURROGATE_SWEEP_CELL_CAPACITY",
    "SURROGATE_SWEEP_CELL_CAPACITY",
    "query_surrogate_aabb_clearance",
    "query_surrogate_swept_aabb",
]
