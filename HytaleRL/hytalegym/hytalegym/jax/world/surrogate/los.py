"""Distinct native perception and hit-confirmation LOS predicates."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.geometry.contract import FLAG_OPAQUE, FLAG_SOLID
from hytalegym.jax.world.surrogate.atlas import lookup_surrogate_blocks
from hytalegym.jax.world.surrogate.capabilities import (
    SURROGATE_CAPABILITY_ENTITY_GEOMETRY,
)
from hytalegym.jax.world.surrogate.types import (
    SurrogateAtlas,
    SurrogateHitboxLineOfSightResult,
    SurrogateRuntimeState,
    SurrogateVisibilityResult,
)
from hytalegym.worldgen.region import CAPTURE_BLOCKS_PER_AXIS, WORLD_HEIGHT

MAX_SURROGATE_LOS_CELLS = 2 * (CAPTURE_BLOCKS_PER_AXIS - 1) + (WORLD_HEIGHT - 1) + 1
_BOUNDARY_EPSILON = jnp.float32(1.0e-6)


def surrogate_perception_line_of_sight_result(
    atlas: SurrogateAtlas,
    runtime: SurrogateRuntimeState,
    start: jax.Array,
    end: jax.Array,
    *,
    role_opaque_cell_mask: jax.Array | None = None,
    max_cells: int = MAX_SURROGATE_LOS_CELLS,
) -> SurrogateVisibilityResult:
    """Trace NPC perception LOS through opacity cells, failing closed on gaps."""

    start = jnp.asarray(start, dtype=jnp.float32)
    end = jnp.asarray(end, dtype=jnp.float32)
    if start.ndim != 2 or start.shape[1] != 3 or end.shape != start.shape:
        raise ValueError("start and end must have shape [batch, 3]")
    finite = jnp.all(jnp.isfinite(start) & jnp.isfinite(end), axis=1)
    safe_start = jnp.where(finite[:, None], start, 0.0)
    safe_end = jnp.where(finite[:, None], end, 0.0)
    start_block = jnp.floor(safe_start).astype(jnp.int32)
    end_block = jnp.floor(safe_end).astype(jnp.int32)
    result = surrogate_perception_block_line_of_sight_result(
        atlas,
        runtime,
        start_block,
        safe_start - start_block.astype(jnp.float32),
        end_block,
        safe_end - end_block.astype(jnp.float32),
        role_opaque_cell_mask=role_opaque_cell_mask,
        max_cells=max_cells,
    )
    return SurrogateVisibilityResult(
        visible=result.visible & finite,
        geometry_exhausted=result.geometry_exhausted | ~finite,
        capacity_exceeded=result.capacity_exceeded,
    )


def surrogate_perception_block_line_of_sight_result(
    atlas: SurrogateAtlas,
    runtime: SurrogateRuntimeState,
    start_block: jax.Array,
    start_fraction: jax.Array,
    end_block: jax.Array,
    end_fraction: jax.Array,
    *,
    role_opaque_cell_mask: jax.Array | None = None,
    max_cells: int = MAX_SURROGATE_LOS_CELLS,
) -> SurrogateVisibilityResult:
    """Walk exact int32 cells with role opacity and blocked unloaded space."""

    if (
        isinstance(max_cells, bool)
        or not isinstance(max_cells, int)
        or not 1 <= max_cells <= MAX_SURROGATE_LOS_CELLS
    ):
        raise ValueError(f"max_cells must be in [1, {MAX_SURROGATE_LOS_CELLS}]")
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
        or runtime.environment_world_id.shape != (start_block.shape[0],)
    ):
        raise ValueError("LOS arrays and runtime must share shape [batch, 3]")
    role_opaque = _role_opaque_mask(
        role_opaque_cell_mask,
        start_block.shape[0],
        atlas.cell_flags.shape[0],
    )

    valid_fraction = jnp.all(
        jnp.isfinite(start_fraction)
        & jnp.isfinite(end_fraction)
        & (start_fraction >= 0.0)
        & (start_fraction < 1.0)
        & (end_fraction >= 0.0)
        & (end_fraction < 1.0),
        axis=1,
    )
    start_fraction = jnp.where(
        valid_fraction[:, None],
        start_fraction,
        0.0,
    )
    end_fraction = jnp.where(
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
    safe_delta = jnp.where(delta_overflow, 0, block_delta)
    direction = safe_delta.astype(jnp.float32) + end_fraction - start_fraction
    step = jnp.sign(direction).astype(jnp.int32)
    current = start_block - ((step < 0) & (start_fraction <= _BOUNDARY_EPSILON)).astype(
        jnp.int32
    )
    terminal = end_block - ((step > 0) & (end_fraction <= _BOUNDARY_EPSILON)).astype(
        jnp.int32
    )
    terminal_delta = terminal - current
    terminal_overflow = (
        ((current < 0) & (terminal >= 0) & (terminal_delta < 0))
        | ((current >= 0) & (terminal < 0) & (terminal_delta > 0))
        | (terminal_delta == jnp.iinfo(jnp.int32).min)
    )
    safe_terminal_delta = jnp.where(
        terminal_overflow,
        0,
        terminal_delta,
    )
    capacity_exceeded = (
        jnp.sum(jnp.abs(safe_terminal_delta), axis=1) + 1 > max_cells
    ) | jnp.any(delta_overflow | terminal_overflow, axis=1)

    next_boundary = jnp.where(step > 0, current + 1, current)
    nonzero = step != 0
    safe_direction = jnp.where(nonzero, direction, 1.0)
    boundary_time = jnp.where(
        nonzero,
        ((next_boundary - start_block).astype(jnp.float32) - start_fraction)
        / safe_direction,
        jnp.inf,
    )
    time_delta = jnp.where(
        nonzero,
        jnp.abs(1.0 / safe_direction),
        jnp.inf,
    )
    initial = (
        current,
        boundary_time,
        jnp.zeros(start_block.shape[0], dtype=jnp.bool_),
        ~capacity_exceeded & valid_fraction,
        capacity_exceeded | ~valid_fraction,
    )

    def visit(
        _: int,
        state: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, ...]:
        cell, times, blocked, complete, done = state
        active = ~done
        selected = lookup_surrogate_blocks(
            atlas,
            runtime,
            cell[:, None, :],
            require_core=False,
        )
        available = selected.available[:, 0]
        cell_code = selected.cell_code[:, 0].astype(jnp.int32)
        role_override = (
            False
            if role_opaque is None
            else jnp.take_along_axis(
                role_opaque,
                cell_code[:, None],
                axis=1,
            )[:, 0]
        )
        opaque = (
            (selected.flags[:, 0].astype(jnp.int32) & FLAG_OPAQUE) != 0
        ) | role_override
        blocked_here = active & available & opaque
        reached = jnp.all(cell == terminal, axis=1)
        next_complete = complete & (~active | available)
        next_blocked = blocked | blocked_here
        next_done = done | (active & (blocked_here | ~available | reached))

        minimum_time = jnp.min(times, axis=1, keepdims=True)
        tied = jnp.abs(times - minimum_time) <= _BOUNDARY_EPSILON
        advance = active & ~next_done
        axes = tied & advance[:, None]
        return (
            cell + jnp.where(axes, step, 0),
            times + jnp.where(axes, time_delta, 0.0),
            next_blocked,
            next_complete,
            next_done,
        )

    _, _, blocked, complete, done = jax.lax.fori_loop(
        0,
        max_cells,
        visit,
        initial,
    )
    complete &= done
    return SurrogateVisibilityResult(
        visible=complete & ~blocked,
        geometry_exhausted=~complete,
        capacity_exceeded=capacity_exceeded,
    )


def surrogate_perception_line_of_sight(
    atlas: SurrogateAtlas,
    runtime: SurrogateRuntimeState,
    start: jax.Array,
    end: jax.Array,
    *,
    role_opaque_cell_mask: jax.Array | None = None,
) -> jax.Array:
    return surrogate_perception_line_of_sight_result(
        atlas,
        runtime,
        start,
        end,
        role_opaque_cell_mask=role_opaque_cell_mask,
    ).visible


def surrogate_hitbox_line_of_sight_result(
    atlas: SurrogateAtlas,
    runtime: SurrogateRuntimeState,
    start: jax.Array,
    end: jax.Array,
    *,
    max_cells: int = MAX_SURROGATE_LOS_CELLS,
) -> SurrogateHitboxLineOfSightResult:
    """Trace selector LOS against solid detail boxes.

    Published solid cells use the exact rotated detail boxes stored in the
    atlas. Unpublished cells are non-blocking, matching ``HorizontalSelector``;
    ``geometry_exhausted`` remains true so coverage loss cannot be mistaken for
    complete evidence. Invalid inputs and fixed-capacity overflow fail closed.
    """

    start = jnp.asarray(start, dtype=jnp.float32)
    end = jnp.asarray(end, dtype=jnp.float32)
    if start.ndim != 2 or start.shape[1] != 3 or end.shape != start.shape:
        raise ValueError("start and end must have shape [batch, 3]")
    finite = jnp.all(jnp.isfinite(start) & jnp.isfinite(end), axis=1)
    safe_start = jnp.where(finite[:, None], start, 0.0)
    safe_end = jnp.where(finite[:, None], end, 0.0)
    start_block = jnp.floor(safe_start).astype(jnp.int32)
    end_block = jnp.floor(safe_end).astype(jnp.int32)
    result = surrogate_hitbox_block_line_of_sight_result(
        atlas,
        runtime,
        start_block,
        safe_start - start_block.astype(jnp.float32),
        end_block,
        safe_end - end_block.astype(jnp.float32),
        max_cells=max_cells,
    )
    return SurrogateHitboxLineOfSightResult(
        clear=result.clear & finite,
        geometry_exhausted=result.geometry_exhausted | ~finite,
        capacity_exceeded=result.capacity_exceeded,
        invalid=result.invalid | ~finite,
    )


def surrogate_hitbox_block_line_of_sight_result(
    atlas: SurrogateAtlas,
    runtime: SurrogateRuntimeState,
    start_block: jax.Array,
    start_fraction: jax.Array,
    end_block: jax.Array,
    end_fraction: jax.Array,
    *,
    max_cells: int = MAX_SURROGATE_LOS_CELLS,
) -> SurrogateHitboxLineOfSightResult:
    """Walk selector cells and test closed segment/detail-box intersections."""

    if (
        isinstance(max_cells, bool)
        or not isinstance(max_cells, int)
        or not 1 <= max_cells <= MAX_SURROGATE_LOS_CELLS
    ):
        raise ValueError(f"max_cells must be in [1, {MAX_SURROGATE_LOS_CELLS}]")
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
        or runtime.environment_world_id.shape != (start_block.shape[0],)
    ):
        raise ValueError("LOS arrays and runtime must share shape [batch, 3]")

    valid_fraction = jnp.all(
        jnp.isfinite(start_fraction)
        & jnp.isfinite(end_fraction)
        & (start_fraction >= 0.0)
        & (start_fraction < 1.0)
        & (end_fraction >= 0.0)
        & (end_fraction < 1.0),
        axis=1,
    )
    safe_start_fraction = jnp.where(valid_fraction[:, None], start_fraction, 0.0)
    safe_end_fraction = jnp.where(valid_fraction[:, None], end_fraction, 0.0)
    block_delta = end_block - start_block
    delta_overflow = (
        ((start_block < 0) & (end_block >= 0) & (block_delta < 0))
        | ((start_block >= 0) & (end_block < 0) & (block_delta > 0))
        | (block_delta == jnp.iinfo(jnp.int32).min)
    )
    safe_delta = jnp.where(delta_overflow, 0, block_delta)
    direction = (
        safe_delta.astype(jnp.float32)
        + safe_end_fraction
        - safe_start_fraction
    )
    step = jnp.sign(direction).astype(jnp.int32)
    current = start_block - (
        (step < 0) & (safe_start_fraction <= _BOUNDARY_EPSILON)
    ).astype(jnp.int32)
    terminal = end_block - (
        (step > 0) & (safe_end_fraction <= _BOUNDARY_EPSILON)
    ).astype(jnp.int32)
    terminal_delta = terminal - current
    terminal_overflow = (
        ((current < 0) & (terminal >= 0) & (terminal_delta < 0))
        | ((current >= 0) & (terminal < 0) & (terminal_delta > 0))
        | (terminal_delta == jnp.iinfo(jnp.int32).min)
    )
    safe_terminal_delta = jnp.where(terminal_overflow, 0, terminal_delta)
    capacity_exceeded = (
        jnp.sum(jnp.abs(safe_terminal_delta), axis=1) + 1 > max_cells
    ) | jnp.any(delta_overflow | terminal_overflow, axis=1)

    next_boundary = jnp.where(step > 0, current + 1, current)
    nonzero = step != 0
    safe_direction = jnp.where(nonzero, direction, 1.0)
    boundary_time = jnp.where(
        nonzero,
        (
            (next_boundary - start_block).astype(jnp.float32)
            - safe_start_fraction
        )
        / safe_direction,
        jnp.inf,
    )
    time_delta = jnp.where(nonzero, jnp.abs(1.0 / safe_direction), jnp.inf)
    invalid = ~valid_fraction
    initial = (
        current,
        boundary_time,
        jnp.zeros(start_block.shape[0], dtype=jnp.bool_),
        ~capacity_exceeded & ~invalid,
        capacity_exceeded | invalid,
    )
    relative_end = safe_delta.astype(jnp.float32) + safe_end_fraction

    def visit(
        _: int,
        state: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, ...]:
        cell, times, blocked, complete, done = state
        active = ~done
        selected = lookup_surrogate_blocks(
            atlas,
            runtime,
            cell[:, None, :],
            require_core=False,
        )
        available = selected.available[:, 0]
        solid = (
            selected.flags[:, 0].astype(jnp.int32) & FLAG_SOLID
        ) != 0
        relative_cell = (cell - start_block).astype(jnp.float32)
        boxes = selected.collision_boxes[:, 0] + jnp.concatenate(
            (relative_cell, relative_cell),
            axis=1,
        )[:, None, :]
        box_hits = _segment_intersects_closed_boxes(
            safe_start_fraction[:, None, :],
            relative_end[:, None, :],
            boxes,
        )
        blocked_here = (
            active
            & available
            & solid
            & jnp.any(box_hits & selected.collision_box_mask[:, 0], axis=1)
        )
        reached = jnp.all(cell == terminal, axis=1)
        next_complete = complete & (~active | available)
        next_blocked = blocked | blocked_here
        next_done = done | (active & (blocked_here | reached))

        minimum_time = jnp.min(times, axis=1, keepdims=True)
        tied = jnp.abs(times - minimum_time) <= _BOUNDARY_EPSILON
        advance = active & ~next_done
        axes = tied & advance[:, None]
        return (
            cell + jnp.where(axes, step, 0),
            times + jnp.where(axes, time_delta, 0.0),
            next_blocked,
            next_complete,
            next_done,
        )

    _, _, blocked, complete, done = jax.lax.fori_loop(
        0,
        max_cells,
        visit,
        initial,
    )
    walk_complete = done & ~capacity_exceeded & ~invalid
    return SurrogateHitboxLineOfSightResult(
        clear=walk_complete & ~blocked,
        geometry_exhausted=~(complete & walk_complete),
        capacity_exceeded=capacity_exceeded,
        invalid=invalid,
    )


def surrogate_hitbox_line_of_sight(
    atlas: SurrogateAtlas,
    runtime: SurrogateRuntimeState,
    start: jax.Array,
    end: jax.Array,
) -> jax.Array:
    """Return native selector hit-confirmation clearance only."""

    return surrogate_hitbox_line_of_sight_result(
        atlas,
        runtime,
        start,
        end,
    ).clear


def surrogate_entity_aware_line_of_sight_result(
    atlas: SurrogateAtlas,
    runtime: SurrogateRuntimeState,
    start: jax.Array,
    end: jax.Array,
    *,
    ignore_entity_slots: jax.Array | None = None,
    role_opaque_cell_mask: jax.Array | None = None,
    max_cells: int = MAX_SURROGATE_LOS_CELLS,
) -> SurrogateVisibilityResult:
    """Combine native-style block opacity with explicitly profiled LOS bodies."""

    start = jnp.asarray(start, dtype=jnp.float32)
    end = jnp.asarray(end, dtype=jnp.float32)
    if start.ndim != 2 or start.shape[1] != 3 or end.shape != start.shape:
        raise ValueError("start and end must have shape [batch, 3]")
    batch = start.shape[0]
    if runtime.environment_world_id.shape != (batch,):
        raise ValueError("runtime and LOS endpoints must share a batch dimension")
    if ignore_entity_slots is None:
        ignored = jnp.zeros(
            (batch, 0),
            dtype=jnp.int32,
        )
    else:
        ignored = jnp.asarray(ignore_entity_slots, dtype=jnp.int32)
        if ignored.ndim != 2 or ignored.shape[0] != batch:
            raise ValueError("ignore_entity_slots must have shape [batch, slots]")

    block = surrogate_perception_line_of_sight_result(
        atlas,
        runtime,
        start,
        end,
        role_opaque_cell_mask=role_opaque_cell_mask,
        max_cells=max_cells,
    )
    has_entity_geometry = (
        runtime.capability_bits & jnp.uint32(SURROGATE_CAPABILITY_ENTITY_GEOMETRY)
    ) != 0
    slots = jnp.arange(runtime.entity_active.shape[1], dtype=jnp.int32)
    ignored_mask = jnp.any(
        slots[None, :, None] == ignored[:, None, :],
        axis=2,
    )
    entity_mask = (
        runtime.entity_active
        & runtime.entity_initialized
        & runtime.entity_geometry_supported
        & runtime.entity_blocks_los
        & ~ignored_mask
    )
    boxes = jnp.concatenate(
        (
            runtime.entity_position + runtime.entity_local_bounds[..., :3],
            runtime.entity_position + runtime.entity_local_bounds[..., 3:],
        ),
        axis=2,
    )
    hits = _segment_hits_boxes(
        start[:, None, :],
        end[:, None, :],
        boxes,
    )
    entity_blocked = jnp.any(hits & entity_mask, axis=1)
    entity_complete = has_entity_geometry & ~runtime.unsupported_mechanics
    return SurrogateVisibilityResult(
        visible=block.visible & entity_complete & ~entity_blocked,
        geometry_exhausted=block.geometry_exhausted | ~entity_complete,
        capacity_exceeded=block.capacity_exceeded,
    )


def _segment_hits_boxes(
    start: jax.Array,
    end: jax.Array,
    boxes: jax.Array,
) -> jax.Array:
    delta = end - start
    moving = jnp.abs(delta) > _BOUNDARY_EPSILON
    safe_delta = jnp.where(moving, delta, 1.0)
    first = (boxes[..., :3] + _BOUNDARY_EPSILON - start) / safe_delta
    second = (boxes[..., 3:] - _BOUNDARY_EPSILON - start) / safe_delta
    entry = jnp.where(moving, jnp.minimum(first, second), -jnp.inf)
    exit_time = jnp.where(moving, jnp.maximum(first, second), jnp.inf)
    stationary_inside = (
        (~moving)
        & (start > boxes[..., :3] + _BOUNDARY_EPSILON)
        & (start < boxes[..., 3:] - _BOUNDARY_EPSILON)
    )
    lower = jnp.max(entry, axis=2)
    upper = jnp.min(exit_time, axis=2)
    return (
        jnp.all(moving | stationary_inside, axis=2)
        & (lower < upper - _BOUNDARY_EPSILON)
        & (lower < 1.0 - _BOUNDARY_EPSILON)
        & (upper > _BOUNDARY_EPSILON)
    )


def _segment_intersects_closed_boxes(
    start: jax.Array,
    end: jax.Array,
    boxes: jax.Array,
) -> jax.Array:
    """Match Hytale ``Box.intersectsLine`` closed slab semantics."""

    delta = end - start
    moving = jnp.abs(delta) >= jnp.float32(1.0e-10)
    safe_delta = jnp.where(moving, delta, 1.0)
    first = (boxes[..., :3] - start) / safe_delta
    second = (boxes[..., 3:] - start) / safe_delta
    entry = jnp.where(moving, jnp.minimum(first, second), 0.0)
    exit_time = jnp.where(moving, jnp.maximum(first, second), 1.0)
    stationary_inside = (
        (start >= boxes[..., :3]) & (start <= boxes[..., 3:])
    )
    return (
        jnp.all(moving | stationary_inside, axis=-1)
        & (jnp.maximum(jnp.max(entry, axis=-1), 0.0)
           <= jnp.minimum(jnp.min(exit_time, axis=-1), 1.0))
    )


# Compatibility aliases retain the old API while all in-tree callers use the
# predicate-specific names above.
surrogate_line_of_sight_result = surrogate_perception_line_of_sight_result
surrogate_block_line_of_sight_result = (
    surrogate_perception_block_line_of_sight_result
)
surrogate_line_of_sight = surrogate_perception_line_of_sight


def _role_opaque_mask(
    value: jax.Array | None,
    batch: int,
    palette_capacity: int,
) -> jax.Array | None:
    if value is None:
        return None
    result = jnp.asarray(value)
    if result.dtype != jnp.bool_:
        raise TypeError("role_opaque_cell_mask must have boolean dtype")
    if result.shape == (palette_capacity,):
        return jnp.broadcast_to(result, (batch, palette_capacity))
    if result.shape != (batch, palette_capacity):
        raise ValueError(
            "role_opaque_cell_mask must have shape [cell palette] "
            "or [batch, cell palette]"
        )
    return result


__all__ = [
    "MAX_SURROGATE_LOS_CELLS",
    "surrogate_hitbox_block_line_of_sight_result",
    "surrogate_hitbox_line_of_sight",
    "surrogate_hitbox_line_of_sight_result",
    "surrogate_perception_block_line_of_sight_result",
    "surrogate_perception_line_of_sight",
    "surrogate_perception_line_of_sight_result",
    "surrogate_block_line_of_sight_result",
    "surrogate_entity_aware_line_of_sight_result",
    "surrogate_line_of_sight",
    "surrogate_line_of_sight_result",
]
