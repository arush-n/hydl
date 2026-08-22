"""Half-open voxel LOS across independently selected exact region cells."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.geometry.contract import FLAG_OPAQUE
from hytalegym.jax.world.region.atlas import lookup_region_blocks
from hytalegym.jax.world.region.mutable_geometry import (
    lookup_region_geometry_blocks,
)
from hytalegym.jax.world.region.types import (
    RegionAtlas,
    RegionGeometryState,
    RegionVisibilityResult,
)
from hytalegym.worldgen.region import (
    CAPTURE_BLOCKS_PER_AXIS,
    WORLD_HEIGHT,
)

MAX_REGION_LOS_CELLS = 2 * (CAPTURE_BLOCKS_PER_AXIS - 1) + (WORLD_HEIGHT - 1) + 1

#: Ray length a Region fixture budgets its *perception* LOS step buffer for.
#: `region_geometry_from_atlas` turns this into a cell count -- a segment of
#: length D crosses at most ceil(sqrt(3)*D)+2 cells -- so 72 blocks is a
#: 128-cell buffer. It must exceed the entity ranges callers actually trace:
#: `sensor_range` is 16 blocks and the pursuit lesson scores out to a 20-block
#: proximity radius. Budgeting this from a *block-interaction* reach instead
#: (4.0 -> 10 cells) makes every ray past ~9 blocks report `capacity_exceeded`,
#: which consumers AND out of `visible` and therefore cannot tell apart from
#: genuine occlusion.
REGION_PERCEPTION_LOS_DISTANCE_BLOCKS = 72.0

_BOUNDARY_EPSILON = jnp.float32(1e-6)


def region_line_of_sight_result(
    atlas: RegionAtlas,
    start: jax.Array,
    end: jax.Array,
    environment_world_id: jax.Array,
    *,
    max_cells: int = MAX_REGION_LOS_CELLS,
    geometry: RegionGeometryState | None = None,
) -> RegionVisibilityResult:
    """Convenience LOS for rebased/small float coordinates.

    Use ``region_block_line_of_sight_result`` with integer blocks and local
    fractions when native world coordinates may exceed float32 precision.
    """

    start = jnp.asarray(start, dtype=jnp.float32)
    end = jnp.asarray(end, dtype=jnp.float32)
    if start.ndim != 2 or start.shape[1] != 3 or end.shape != start.shape:
        raise ValueError("start and end must have shape (batch, 3)")
    finite = jnp.all(jnp.isfinite(start) & jnp.isfinite(end), axis=1)
    safe_start = jnp.where(finite[:, None], start, 0.0)
    safe_end = jnp.where(finite[:, None], end, 0.0)
    start_block = jnp.floor(safe_start).astype(jnp.int32)
    end_block = jnp.floor(safe_end).astype(jnp.int32)
    result = region_block_line_of_sight_result(
        atlas,
        start_block,
        safe_start - start_block.astype(jnp.float32),
        end_block,
        safe_end - end_block.astype(jnp.float32),
        environment_world_id,
        max_cells=max_cells,
        geometry=geometry,
    )
    return RegionVisibilityResult(
        visible=result.visible & finite,
        geometry_exhausted=result.geometry_exhausted | ~finite,
        capacity_exceeded=result.capacity_exceeded,
    )


def region_block_line_of_sight_result(
    atlas: RegionAtlas,
    start_block: jax.Array,
    start_fraction: jax.Array,
    end_block: jax.Array,
    end_fraction: jax.Array,
    environment_world_id: jax.Array,
    *,
    max_cells: int = MAX_REGION_LOS_CELLS,
    geometry: RegionGeometryState | None = None,
) -> RegionVisibilityResult:
    """Walk exact int32 world cells plus float32 cell-local endpoints."""

    if (
        isinstance(max_cells, bool)
        or not isinstance(max_cells, int)
        or not 1 <= max_cells <= MAX_REGION_LOS_CELLS
    ):
        raise ValueError(
            f"max_cells must be an integer in [1, {MAX_REGION_LOS_CELLS}]"
        )

    start_block = jnp.asarray(start_block, dtype=jnp.int32)
    end_block = jnp.asarray(end_block, dtype=jnp.int32)
    start_fraction = jnp.asarray(start_fraction, dtype=jnp.float32)
    end_fraction = jnp.asarray(end_fraction, dtype=jnp.float32)
    if (
        start_block.ndim != 2
        or start_block.shape[1] != 3
        or end_block.shape != start_block.shape
        or start_fraction.shape != start_block.shape
        or end_fraction.shape != start_block.shape
    ):
        raise ValueError("LOS block and fraction arrays must have shape (batch, 3)")
    batch = start_block.shape[0]
    world_ids = jnp.asarray(environment_world_id, dtype=jnp.int32)
    if world_ids.shape != (batch,):
        raise ValueError("environment_world_id must have shape (batch,)")
    valid_fraction = jnp.all(
        jnp.isfinite(start_fraction)
        & jnp.isfinite(end_fraction)
        & (start_fraction >= 0.0)
        & (start_fraction < 1.0)
        & (end_fraction >= 0.0)
        & (end_fraction < 1.0),
        axis=1,
    )
    safe_start_fraction = jnp.where(
        valid_fraction[:, None],
        start_fraction,
        0.0,
    )
    safe_end_fraction = jnp.where(
        valid_fraction[:, None],
        end_fraction,
        0.0,
    )
    block_delta = end_block - start_block
    delta_overflow = (
        ((start_block < 0) & (end_block >= 0) & (block_delta < 0))
        | ((start_block >= 0) & (end_block < 0) & (block_delta > 0))
        | (block_delta == jnp.iinfo(jnp.int32).min)
    )
    safe_block_delta = jnp.where(delta_overflow, 0, block_delta)
    direction = (
        safe_block_delta.astype(jnp.float32) + safe_end_fraction - safe_start_fraction
    )
    step = jnp.sign(direction).astype(jnp.int32)
    start_boundary = safe_start_fraction <= _BOUNDARY_EPSILON
    current = start_block - ((step < 0) & start_boundary).astype(jnp.int32)

    end_boundary = safe_end_fraction <= _BOUNDARY_EPSILON
    terminal = end_block - ((step > 0) & end_boundary).astype(jnp.int32)
    terminal_delta = terminal - current
    terminal_overflow = (
        ((current < 0) & (terminal >= 0) & (terminal_delta < 0))
        | ((current >= 0) & (terminal < 0) & (terminal_delta > 0))
        | (terminal_delta == jnp.iinfo(jnp.int32).min)
    )
    invalid_coordinate = jnp.any(delta_overflow | terminal_overflow, axis=1)
    safe_terminal_delta = jnp.where(terminal_overflow, 0, terminal_delta)
    required_upper_bound = jnp.sum(jnp.abs(safe_terminal_delta), axis=1) + 1
    capacity_exceeded = (
        required_upper_bound > max_cells
    ) | invalid_coordinate

    next_boundary = jnp.where(step > 0, current + 1, current)
    nonzero = step != 0
    safe_direction = jnp.where(nonzero, direction, 1.0)
    t_max = jnp.where(
        nonzero,
        ((next_boundary - start_block).astype(jnp.float32) - safe_start_fraction)
        / safe_direction,
        jnp.inf,
    )
    t_delta = jnp.where(nonzero, jnp.abs(1.0 / safe_direction), jnp.inf)
    initial_state = (
        current,
        t_max,
        jnp.zeros(batch, dtype=jnp.bool_),
        ~capacity_exceeded & valid_fraction,
        capacity_exceeded | ~valid_fraction,
    )

    def visit(
        _: int,
        state: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, ...]:
        cell, boundary_time, blocked, complete, done = state
        active = ~done
        selected = (
            lookup_region_blocks(
                atlas,
                cell[:, None, :],
                world_ids,
                require_core=False,
            )
            if geometry is None
            else lookup_region_geometry_blocks(
                geometry,
                cell[:, None, :],
                require_core=False,
            )
        )
        available = selected.available[:, 0]
        opaque = (selected.flags[:, 0].astype(jnp.int32) & FLAG_OPAQUE) != 0
        blocked_here = active & available & opaque
        reached = jnp.all(cell == terminal, axis=1)
        next_complete = complete & (~active | available)
        next_blocked = blocked | blocked_here
        next_done = done | (active & (blocked_here | ~available | reached))

        minimum_time = jnp.min(boundary_time, axis=1, keepdims=True)
        tied = jnp.abs(boundary_time - minimum_time) <= _BOUNDARY_EPSILON
        advance = active & ~next_done
        advance_axes = tied & advance[:, None]
        next_cell = cell + jnp.where(advance_axes, step, 0)
        next_time = boundary_time + jnp.where(
            advance_axes,
            t_delta,
            0.0,
        )
        return (
            next_cell,
            next_time,
            next_blocked,
            next_complete,
            next_done,
        )

    _, _, blocked, complete, done = jax.lax.fori_loop(
        0,
        max_cells,
        visit,
        initial_state,
    )
    complete = complete & done
    return RegionVisibilityResult(
        visible=complete & ~blocked,
        geometry_exhausted=~complete,
        capacity_exceeded=capacity_exceeded,
    )


def region_line_of_sight(
    atlas: RegionAtlas,
    start: jax.Array,
    end: jax.Array,
    environment_world_id: jax.Array,
) -> jax.Array:
    return region_line_of_sight_result(
        atlas,
        start,
        end,
        environment_world_id,
    ).visible


def region_block_line_of_sight(
    atlas: RegionAtlas,
    start_block: jax.Array,
    start_fraction: jax.Array,
    end_block: jax.Array,
    end_fraction: jax.Array,
    environment_world_id: jax.Array,
) -> jax.Array:
    return region_block_line_of_sight_result(
        atlas,
        start_block,
        start_fraction,
        end_block,
        end_fraction,
        environment_world_id,
    ).visible
