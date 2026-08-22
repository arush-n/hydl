"""JIT-safe support, occlusion, and swept-AABB geometry kernels."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.geometry.contract import (
    CELL_RADIUS,
    FLAG_FLUID,
    FLAG_HAS_FLUID_MOVEMENT_SETTINGS,
    FLAG_OPAQUE,
)
from hytalegym.jax.world.types import (
    GeometryState,
)
from ._sweep import (  # noqa: F401  (re-exported: callers unchanged)
    AabbFirstContactResult,
    Array,
    GroundResult,
    SupportResult,
    SupportSweepResult,
    _CELL_OFFSETS,
    _FLOAT_EPSILON,
    _LOCAL_SWEEP_OFFSETS,
    _LOCAL_SWEEP_SIZE,
    _SKIN_DISTANCE,
    _SUPPORT_CHAIN_ITERATIONS,
    _aabb_first_contact_with_boxes,
    _broadcast_geometry,
    _first_contact_once,
    _flat_cell_index,
    _gather_geometry_cells,
    _local_motion_boxes,
    _support_from_boxes,
    _support_sweep_with_boxes,
    _sweep_once,
    _world_collision_boxes,
    aabb_first_contact_result,
    aabb_grounded,
    aabb_support_sweep,
    support_probe,
)


_MAX_SLIDE_ITERATIONS = 3
class MotionResult(NamedTuple):
    """Result of continuous collision plus sliding for one motion delta."""

    position: Array
    applied_displacement: Array
    first_normal: Array
    collided: Array
    grounded: Array
    ceiling_contact: Array
    geometry_exhausted: Array


class VisibilityResult(NamedTuple):
    """Opaque-block LOS plus whether the fixed local window was complete."""

    visible: Array
    geometry_exhausted: Array


class MovementMediumResult(NamedTuple):
    """FluidFX settings at each entity's feet position."""

    swim_up_speed: Array
    swim_down_speed: Array
    sink_speed: Array
    horizontal_speed_multiplier: Array
    field_of_view_multiplier: Array
    entry_velocity_multiplier: Array
    fluid_level: Array
    in_fluid: Array
    has_authored_settings: Array
    geometry_exhausted: Array


class CellEnvironmentResult(NamedTuple):
    """Captured 0.5.7 environment settings for one world-space cell."""

    present: Array
    flags: Array
    fluid_level: Array
    fluid_fill_height: Array
    support: Array
    block_damage: Array
    fluid_damage: Array
    movement: Array
    fluid_movement: Array
    geometry_exhausted: Array


def movement_medium_result(
    geometry: GeometryState,
    position: Array,
) -> MovementMediumResult:
    """Return native fluid-FX horizontal speed at ``[B,3]`` positions.

    Hytale 0.5.7's walk controller samples the fluid cell containing the NPC
    feet and multiplies its role maximum speed by the FluidFX movement
    setting. Dry cells use one. Leaving the complete frame fails closed.
    """

    position = jnp.asarray(position, dtype=jnp.float32)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position must have shape [batch, 3]")
    cell = cell_environment_result(geometry, position)
    in_fluid = (
        cell.present
        & ((cell.flags & jnp.int32(FLAG_FLUID)) != 0)
    )
    fluid = cell.fluid_movement
    return MovementMediumResult(
        swim_up_speed=jnp.where(
            in_fluid,
            fluid[:, 0],
            jnp.float32(0.0),
        ),
        swim_down_speed=jnp.where(
            in_fluid,
            fluid[:, 1],
            jnp.float32(0.0),
        ),
        sink_speed=jnp.where(
            in_fluid,
            fluid[:, 2],
            jnp.float32(0.0),
        ),
        horizontal_speed_multiplier=jnp.where(
            in_fluid,
            fluid[:, 3],
            jnp.float32(1.0),
        ),
        field_of_view_multiplier=jnp.where(
            in_fluid,
            fluid[:, 4],
            jnp.float32(1.0),
        ),
        entry_velocity_multiplier=jnp.where(
            in_fluid,
            fluid[:, 5],
            jnp.float32(1.0),
        ),
        fluid_level=jnp.where(
            in_fluid,
            cell.fluid_level,
            jnp.int32(0),
        ),
        in_fluid=in_fluid,
        has_authored_settings=(
            in_fluid
            & (
                (
                    cell.flags
                    & jnp.int32(FLAG_HAS_FLUID_MOVEMENT_SETTINGS)
                )
                != 0
            )
        ),
        geometry_exhausted=cell.geometry_exhausted,
    )


def cell_environment_result(
    geometry: GeometryState,
    position: Array,
) -> CellEnvironmentResult:
    """Gather one fixed-shape semantic cell per batch element.

    Values are captured directly from Hytale's versioned block and FluidFX
    assets by the native bridge. An absent in-window sparse cell is air;
    leaving the complete 9-cube is reported separately and must fail closed.
    """

    position = jnp.asarray(position, dtype=jnp.float32)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position must have shape [batch, 3]")
    batch = position.shape[0]
    world = _broadcast_geometry(geometry, batch)
    relative = jnp.floor(position).astype(jnp.int32) - world.origin
    inside = jnp.all(
        (relative >= -CELL_RADIUS) & (relative <= CELL_RADIUS),
        axis=1,
    )
    index = jnp.clip(
        _flat_cell_index(relative),
        0,
        _CELL_OFFSETS.shape[0] - 1,
    )

    def gather_scalar(values: Array) -> Array:
        return jnp.take_along_axis(values, index[:, None], axis=1)[:, 0]

    def gather_features(values: Array) -> Array:
        return jax.vmap(lambda cells, selected: cells[selected])(
            values,
            index,
        )

    present = inside & gather_scalar(world.cell_mask)
    movement = gather_features(world.movement)
    fluid_movement = gather_features(world.fluid_movement)
    fluid_fill_height = (
        jnp.full((batch,), -1.0, dtype=jnp.float32)
        if world.fluid_fill_height is None
        else gather_scalar(world.fluid_fill_height)
    )
    return CellEnvironmentResult(
        present=present,
        flags=jnp.where(
            present,
            gather_scalar(world.flags),
            jnp.int32(0),
        ),
        fluid_level=jnp.where(
            present,
            gather_scalar(world.fluid_level),
            jnp.int32(0),
        ),
        fluid_fill_height=jnp.where(
            present,
            fluid_fill_height,
            jnp.float32(-1.0),
        ),
        support=jnp.where(
            present,
            gather_scalar(world.support),
            jnp.int32(0),
        ),
        block_damage=jnp.where(
            present,
            gather_scalar(world.block_damage),
            jnp.int32(0),
        ),
        fluid_damage=jnp.where(
            present,
            gather_scalar(world.fluid_damage),
            jnp.int32(0),
        ),
        movement=jnp.where(
            present[:, None],
            movement,
            jnp.zeros_like(movement),
        ),
        fluid_movement=jnp.where(
            present[:, None],
            fluid_movement,
            jnp.zeros_like(fluid_movement),
        ),
        geometry_exhausted=~inside,
    )


def line_of_sight_result(
    geometry: GeometryState,
    start: Array,
    end: Array,
) -> VisibilityResult:
    """Test block-opaque voxel occlusion for ``[B,3]`` line segments.

    Hytale's NPC LOS classifies opaque blocks as cells, separately from their
    physical detail hitboxes. Using unit-cell slabs here preserves that
    distinction; native differential fixtures remain the final oracle.
    """

    start = jnp.asarray(start, dtype=jnp.float32)
    end = jnp.asarray(end, dtype=jnp.float32)
    if start.ndim != 2 or start.shape[1] != 3 or end.shape != start.shape:
        raise ValueError("start and end must have shape [batch, 3]")

    world = _broadcast_geometry(geometry, start.shape[0])
    cell_minimum = (
        world.origin[:, None, :].astype(jnp.float32)
        + _CELL_OFFSETS[None, :, :]
    )
    cell_maximum = cell_minimum + 1.0
    opaque = (
        world.cell_mask
        & ((world.flags & jnp.int32(FLAG_OPAQUE)) != 0)
    )
    direction = end - start
    expanded_start = start[:, None, :]
    expanded_direction = direction[:, None, :]
    nonzero = jnp.abs(expanded_direction) > _FLOAT_EPSILON
    inverse = jnp.where(nonzero, 1.0 / expanded_direction, 0.0)
    first = (cell_minimum - expanded_start) * inverse
    second = (cell_maximum - expanded_start) * inverse
    axis_entry = jnp.minimum(first, second)
    axis_exit = jnp.maximum(first, second)
    inside_slab = (
        (expanded_start >= cell_minimum)
        & (expanded_start < cell_maximum)
    )
    axis_entry = jnp.where(
        nonzero,
        axis_entry,
        jnp.where(inside_slab, -jnp.inf, jnp.inf),
    )
    axis_exit = jnp.where(
        nonzero,
        axis_exit,
        jnp.where(inside_slab, jnp.inf, -jnp.inf),
    )
    entry = jnp.max(axis_entry, axis=2)
    exit = jnp.min(axis_exit, axis=2)
    # Native PositionCache walks half-open voxel cells with BlockIterator.
    # Requiring a positive segment interval excludes cells touched only at an
    # edge/corner, while the half-open static-axis test selects the same side
    # as floor() for a ray that lies on a cell face.
    clipped_entry = jnp.maximum(entry, 0.0)
    clipped_exit = jnp.minimum(exit, 1.0)
    intersects = opaque & (
        clipped_entry < clipped_exit - _FLOAT_EPSILON
    )
    segment_minimum = (
        jnp.floor(jnp.minimum(start, end)).astype(jnp.int32)
        - world.origin
    )
    segment_maximum = (
        jnp.floor(jnp.maximum(start, end) - _FLOAT_EPSILON).astype(jnp.int32)
        - world.origin
    )
    complete = jnp.all(
        (segment_minimum >= -CELL_RADIUS)
        & (segment_maximum <= CELL_RADIUS),
        axis=1,
    )
    # Conservative failure: an incomplete window cannot assert visibility.
    exhausted = ~complete
    return VisibilityResult(
        visible=complete & ~jnp.any(intersects, axis=1),
        geometry_exhausted=exhausted,
    )


def line_of_sight(
    geometry: GeometryState,
    start: Array,
    end: Array,
) -> Array:
    """Return LOS only; use ``line_of_sight_result`` at fidelity boundaries."""

    return line_of_sight_result(geometry, start, end).visible


def resolve_aabb_motion(
    geometry: GeometryState,
    position: Array,
    displacement: Array,
    bounds: Array | None = None,
    stop_on_vertical_collision: Array | None = None,
    skin_distance: float = float(_SKIN_DISTANCE),
) -> MotionResult:
    """Continuously sweep one native entity AABB along solid detail boxes.

    ``bounds`` defaults to the captured agent AABB. Callers simulating another
    entity must pass that entity's native local bounds explicitly; silently
    borrowing the agent shape would make collision traces non-transferable.
    """

    position = jnp.asarray(position, dtype=jnp.float32)
    displacement = jnp.asarray(displacement, dtype=jnp.float32)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position must have shape [batch, 3]")
    if displacement.shape != position.shape:
        raise ValueError("displacement must match position")
    batch = position.shape[0]
    if (
        isinstance(skin_distance, bool)
        or not np.isfinite(skin_distance)
        or skin_distance < 0.0
    ):
        raise ValueError("skin_distance must be finite and non-negative")
    skin = jnp.float32(skin_distance)
    world = _broadcast_geometry(geometry, batch)
    if stop_on_vertical_collision is None:
        stop_vertical = jnp.zeros((batch,), dtype=jnp.bool_)
    else:
        stop_vertical = jnp.asarray(
            stop_on_vertical_collision,
            dtype=jnp.bool_,
        )
        if stop_vertical.ndim == 0:
            stop_vertical = jnp.broadcast_to(stop_vertical, (batch,))
        if stop_vertical.shape != (batch,):
            raise ValueError(
                "stop_on_vertical_collision must have shape [batch]"
            )
    if bounds is None:
        entity_bounds = world.agent_bounds
    else:
        entity_bounds = jnp.asarray(bounds, dtype=jnp.float32)
        if entity_bounds.ndim != 2 or entity_bounds.shape[1] != 6:
            raise ValueError("bounds must have shape [batch|1, 6]")
        if entity_bounds.shape[0] not in (1, batch):
            raise ValueError("bounds batch must be one or match position")
        if entity_bounds.shape[0] == 1 and batch != 1:
            entity_bounds = jnp.broadcast_to(entity_bounds, (batch, 6))
    (
        local_minimum,
        local_maximum,
        local_valid,
        local_capacity_exceeded,
        geometry_exhausted,
    ) = _local_motion_boxes(
        world,
        position,
        displacement,
        entity_bounds,
    )
    require_full_grid = jnp.any(local_capacity_exceeded)

    def full_resolution(_: None) -> MotionResult:
        minimum, maximum, valid = _world_collision_boxes(world)
        return _resolve_motion_with_boxes(
            position,
            displacement,
            entity_bounds,
            minimum,
            maximum,
            valid,
            geometry_exhausted,
            stop_vertical,
            skin,
        )

    def local_resolution(_: None) -> MotionResult:
        return _resolve_motion_with_boxes(
            position,
            displacement,
            entity_bounds,
            local_minimum,
            local_maximum,
            local_valid,
            geometry_exhausted,
            stop_vertical,
            skin,
        )

    return jax.lax.cond(
        require_full_grid,
        full_resolution,
        local_resolution,
        operand=None,
    )


def _resolve_motion_with_boxes(
    position: Array,
    displacement: Array,
    bounds: Array,
    minimum: Array,
    maximum: Array,
    valid: Array,
    geometry_exhausted: Array,
    stop_on_vertical_collision: Array,
    skin_distance: Array = _SKIN_DISTANCE,
) -> MotionResult:
    initial = position
    batch = position.shape[0]

    def iteration(carry, _):
        (
            current,
            remaining,
            collided,
            first_normal,
            grounded,
            ceiling,
        ) = carry
        next_position, next_remaining, normal, hit = _sweep_once(
            current,
            remaining,
            bounds,
            minimum,
            maximum,
            valid,
            skin_distance,
        )
        stop_vertical = (
            stop_on_vertical_collision
            & hit
            & (jnp.abs(normal[:, 1]) > 0.5)
        )
        next_remaining = jnp.where(
            stop_vertical[:, None],
            0.0,
            next_remaining,
        )
        first_normal = jnp.where(
            ((~collided) & hit)[:, None],
            normal,
            first_normal,
        )
        return (
            next_position,
            next_remaining,
            collided | hit,
            first_normal,
            grounded | (hit & (normal[:, 1] > 0.5)),
            ceiling | (hit & (normal[:, 1] < -0.5)),
        ), None

    carry, _ = jax.lax.scan(
        iteration,
        (
            position,
            displacement,
            jnp.zeros(batch, dtype=jnp.bool_),
            jnp.zeros((batch, 3), dtype=jnp.float32),
            jnp.zeros(batch, dtype=jnp.bool_),
            jnp.zeros(batch, dtype=jnp.bool_),
        ),
        xs=None,
        length=_MAX_SLIDE_ITERATIONS,
    )
    final, _, collided, first_normal, grounded, ceiling = carry
    return MotionResult(
        position=final,
        applied_displacement=final - initial,
        first_normal=first_normal,
        collided=collided,
        grounded=grounded,
        ceiling_contact=ceiling,
        geometry_exhausted=geometry_exhausted,
    )
