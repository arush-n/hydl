"""Leaf helpers extracted verbatim from geometry.py."""

from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from hytalegym.geometry.contract import CELL_COUNT, CELL_RADIUS, FLAG_SOLID, MAX_DETAIL_BOXES
from hytalegym.jax.world.types import (
    COLLISION_SHAPE_FULL_CUBE,
    GeometryState,
)


Array = jax.Array


_FLOAT_EPSILON = jnp.float32(1.0e-7)


_SKIN_DISTANCE = jnp.float32(1.0e-5)


_LOCAL_SWEEP_SIZE = 3


_SUPPORT_CHAIN_ITERATIONS = MAX_DETAIL_BOXES


_CELL_OFFSETS = jnp.asarray(
    [
        (dx, dy, dz)
        for dx in range(-CELL_RADIUS, CELL_RADIUS + 1)
        for dy in range(-CELL_RADIUS, CELL_RADIUS + 1)
        for dz in range(-CELL_RADIUS, CELL_RADIUS + 1)
    ],
    dtype=jnp.float32,
)


_LOCAL_SWEEP_OFFSETS = jnp.asarray(
    [
        (dx, dy, dz)
        for dx in range(_LOCAL_SWEEP_SIZE)
        for dy in range(_LOCAL_SWEEP_SIZE)
        for dz in range(_LOCAL_SWEEP_SIZE)
    ],
    dtype=jnp.int32,
)


class SupportResult(NamedTuple):
    """Nearest downward support for each supplied probe point."""

    present: Array
    distance: Array
    surface_y: Array
    geometry_exhausted: Array


class AabbFirstContactResult(NamedTuple):
    """First continuous AABB contact before any slide or bounce response."""

    hit: Array
    hit_fraction: Array
    contact_point: Array
    contact_normal: Array
    geometry_exhausted: Array


class GroundResult(NamedTuple):
    """Current AABB support state using a short continuous downward probe."""

    grounded: Array
    geometry_exhausted: Array


class SupportSweepResult(NamedTuple):
    """Continuous horizontal support interval containing the start position."""

    travel_fraction: Array
    leaves_support: Array
    geometry_exhausted: Array


def _world_collision_boxes(
    geometry: GeometryState,
) -> tuple[Array, Array, Array]:
    full_cell = geometry.collision_full_cube_cell.astype(jnp.int32)
    full_present = full_cell >= 0
    safe_full_cell = jnp.clip(full_cell, 0, CELL_COUNT - 1)
    full_base = (
        geometry.origin[:, None, :].astype(jnp.float32)
        + _CELL_OFFSETS[safe_full_cell]
    )
    solid_cell = (geometry.flags & jnp.int32(FLAG_SOLID)) != 0
    full_valid = (
        full_present
        & jnp.take_along_axis(
            geometry.cell_mask,
            safe_full_cell,
            axis=1,
        )
        & jnp.take_along_axis(
            solid_cell,
            safe_full_cell,
            axis=1,
        )
    )

    exception_cell = geometry.collision_exception_box_cell.astype(jnp.int32)
    exception_present = exception_cell >= 0
    safe_exception_cell = jnp.clip(exception_cell, 0, CELL_COUNT - 1)
    exception_base = (
        geometry.origin[:, None, :].astype(jnp.float32)
        + _CELL_OFFSETS[safe_exception_cell]
    )
    exception_minimum = (
        geometry.collision_exception_boxes[..., :3] + exception_base
    )
    exception_maximum = (
        geometry.collision_exception_boxes[..., 3:] + exception_base
    )
    exception_cell_valid = (
        exception_present
        & jnp.take_along_axis(
            geometry.cell_mask,
            safe_exception_cell,
            axis=1,
        )
        & jnp.take_along_axis(
            solid_cell,
            safe_exception_cell,
            axis=1,
        )
    )
    minimum = jnp.concatenate(
        (
            full_base,
            exception_minimum,
        ),
        axis=1,
    )
    maximum = jnp.concatenate(
        (
            full_base + jnp.float32(1.0),
            exception_maximum,
        ),
        axis=1,
    )
    valid = jnp.concatenate(
        (full_valid, exception_cell_valid),
        axis=1,
    )
    order = geometry.collision_world_order.astype(jnp.int32)
    return (
        jnp.take_along_axis(minimum, order[:, :, None], axis=1),
        jnp.take_along_axis(maximum, order[:, :, None], axis=1),
        jnp.take_along_axis(valid, order, axis=1),
    )


def _broadcast_geometry(
    geometry: GeometryState,
    batch: int,
) -> GeometryState:
    geometry_batch = geometry.origin.shape[0]
    if geometry_batch not in (1, batch):
        raise ValueError(
            "geometry batch must be one shared world or match entity batch"
        )
    if geometry_batch == batch:
        return geometry
    return GeometryState(
        *(
            None
            if leaf is None
            else jnp.broadcast_to(leaf, (batch,) + leaf.shape[1:])
            for leaf in geometry
        )
    )


def _flat_cell_index(relative: Array) -> Array:
    side = jnp.int32(CELL_RADIUS * 2 + 1)
    return (
        (relative[..., 0] + CELL_RADIUS) * side * side
        + (relative[..., 1] + CELL_RADIUS) * side
        + relative[..., 2]
        + CELL_RADIUS
    )


def _gather_geometry_cells(
    geometry: GeometryState,
    relative: Array,
    candidate_mask: Array,
) -> tuple[Array, Array, Array]:
    """Gather ``[B,G,N,3]`` cells as cubes plus sparse exceptions.

    Every exception remains eligible because native boxes may protrude several
    cells beyond their source. The flat table makes that exact scan bounded by
    active exception boxes rather than cells times the widest native shape.
    """

    batch, groups, _, _ = relative.shape
    world = _broadcast_geometry(geometry, batch)
    inside = candidate_mask & jnp.all(
        (relative >= -CELL_RADIUS) & (relative <= CELL_RADIUS),
        axis=3,
    )
    indices = jnp.clip(_flat_cell_index(relative), 0, _CELL_OFFSETS.shape[0] - 1)

    def gather_cells(values: Array) -> Array:
        return jax.vmap(lambda row, selected: row[selected])(
            values,
            indices,
        )

    cell_mask = gather_cells(world.cell_mask)
    flags = gather_cells(world.flags)
    shape_index = gather_cells(world.collision_shape_index)
    full_base = (
        world.origin[:, None, None, :].astype(jnp.float32)
        + relative.astype(jnp.float32)
    )
    solid = (flags & jnp.int32(FLAG_SOLID)) != 0
    full_valid = (
        inside
        & cell_mask
        & solid
        & (
            shape_index
            == jnp.int16(COLLISION_SHAPE_FULL_CUBE)
        )
    )

    exception_cell = world.collision_exception_box_cell.astype(jnp.int32)
    exception_present = exception_cell >= 0
    safe_exception_cell = jnp.clip(exception_cell, 0, CELL_COUNT - 1)
    exception_relative = _CELL_OFFSETS[
        safe_exception_cell
    ].astype(jnp.int32)
    exception_cell_valid = (
        exception_present
        & jnp.take_along_axis(
            world.cell_mask,
            safe_exception_cell,
            axis=1,
        )
        & jnp.take_along_axis(
            (world.flags & jnp.int32(FLAG_SOLID)) != 0,
            safe_exception_cell,
            axis=1,
        )
    )
    exception_base = (
        world.origin[:, None, :].astype(jnp.float32)
        + exception_relative.astype(jnp.float32)
    )[:, None, :, :]
    exception_minimum = (
        world.collision_exception_boxes[:, None, :, :3]
        + exception_base
    )
    exception_maximum = (
        world.collision_exception_boxes[:, None, :, 3:]
        + exception_base
    )
    exception_shape = (
        batch,
        groups,
    ) + exception_minimum.shape[2:]
    exception_minimum = jnp.broadcast_to(
        exception_minimum,
        exception_shape,
    )
    exception_maximum = jnp.broadcast_to(
        exception_maximum,
        exception_shape,
    )
    exception_valid = jnp.broadcast_to(
        exception_cell_valid[:, None, :],
        (batch, groups, exception_cell_valid.shape[1]),
    )
    minimum = jnp.concatenate(
        (
            full_base,
            exception_minimum,
        ),
        axis=2,
    )
    maximum = jnp.concatenate(
        (
            full_base + jnp.float32(1.0),
            exception_maximum,
        ),
        axis=2,
    )
    valid = jnp.concatenate(
        (
            full_valid,
            exception_valid,
        ),
        axis=2,
    )
    return (
        minimum,
        maximum,
        valid,
    )


def _local_motion_boxes(
    geometry: GeometryState,
    position: Array,
    displacement: Array,
    bounds: Array,
    *,
    include_closed_endpoint: bool = False,
) -> tuple[Array, Array, Array, Array, Array]:
    batch = position.shape[0]
    world = _broadcast_geometry(geometry, batch)
    start_minimum = position + bounds[:, :3]
    start_maximum = position + bounds[:, 3:]
    end_minimum = start_minimum + displacement
    end_maximum = start_maximum + displacement
    swept_minimum = jnp.minimum(start_minimum, end_minimum)
    swept_maximum = jnp.maximum(start_maximum, end_maximum)
    origin = world.origin.astype(jnp.int32)
    minimum_cell = jnp.floor(swept_minimum).astype(jnp.int32) - origin
    candidate_maximum = (
        swept_maximum
        if include_closed_endpoint
        else swept_maximum - _FLOAT_EPSILON
    )
    maximum_cell = jnp.floor(candidate_maximum).astype(jnp.int32) - origin
    extent = maximum_cell - minimum_cell + 1
    local_capacity_exceeded = jnp.any(
        extent > jnp.int32(_LOCAL_SWEEP_SIZE),
        axis=1,
    )
    geometry_exhausted = jnp.any(
        (minimum_cell < -CELL_RADIUS) | (maximum_cell > CELL_RADIUS),
        axis=1,
    )
    relative = (
        minimum_cell[:, None, :]
        + _LOCAL_SWEEP_OFFSETS[None, :, :]
    )
    candidate_mask = jnp.all(relative <= maximum_cell[:, None, :], axis=2)
    minimum, maximum, valid = _gather_geometry_cells(
        world,
        relative[:, None, :, :],
        candidate_mask[:, None, :],
    )
    return (
        minimum[:, 0],
        maximum[:, 0],
        valid[:, 0],
        local_capacity_exceeded,
        geometry_exhausted,
    )


def support_probe(
    geometry: GeometryState,
    points: Array,
    *,
    max_distance: float = 2.0,
) -> SupportResult:
    """Find exact AABB top surfaces directly below fixed probe points.

    ``points`` has shape ``[B, probes, 3]``. The returned distance is
    ``max_distance`` when no support is present and is never represented by a
    NaN or infinity inside a policy observation.
    """

    points = jnp.asarray(points, dtype=jnp.float32)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("points must have shape [batch, probes, 3]")
    batch, probe_count, _ = points.shape
    world = _broadcast_geometry(geometry, batch)

    vertical_count = int(np.ceil(max_distance)) + 3
    offsets = jnp.asarray(
        [
            (dx, dy, dz)
            for dx in (-1, 0)
            for dy in range(-vertical_count + 1, 1)
            for dz in (-1, 0)
        ],
        dtype=jnp.int32,
    )
    point_cell = (
        jnp.floor(points).astype(jnp.int32)
        - world.origin[:, None, :]
    )
    relative = point_cell[:, :, None, :] + offsets[None, None, :, :]
    exhausted = jnp.any(
        jnp.any(
            (relative < -CELL_RADIUS) | (relative > CELL_RADIUS),
            axis=3,
        ),
        axis=2,
    )
    candidate_count = relative.shape[2]
    minimum, maximum, valid = _gather_geometry_cells(
        world,
        relative,
        jnp.ones(
            (batch, probe_count, candidate_count),
            dtype=jnp.bool_,
        ),
    )
    return _support_from_boxes(
        points,
        minimum,
        maximum,
        valid,
        max_distance,
        exhausted,
    )


def _support_from_boxes(
    points: Array,
    minimum: Array,
    maximum: Array,
    valid: Array,
    max_distance: float,
    geometry_exhausted: Array,
) -> SupportResult:
    px = points[:, :, None, 0]
    py = points[:, :, None, 1]
    pz = points[:, :, None, 2]
    horizontal = (
        (px >= minimum[..., 0] - _FLOAT_EPSILON)
        & (px <= maximum[..., 0] + _FLOAT_EPSILON)
        & (pz >= minimum[..., 2] - _FLOAT_EPSILON)
        & (pz <= maximum[..., 2] + _FLOAT_EPSILON)
    )
    distance = py - maximum[..., 1]
    candidate = (
        valid
        & horizontal
        & (distance >= -_FLOAT_EPSILON)
        & (distance <= jnp.float32(max_distance))
    )
    finite_distance = jnp.where(candidate, distance, jnp.inf)
    nearest_index = jnp.argmin(finite_distance, axis=2)
    nearest_distance = jnp.min(finite_distance, axis=2)
    present = jnp.isfinite(nearest_distance)
    selected_y = jnp.take_along_axis(
        maximum[..., 1],
        nearest_index[:, :, None],
        axis=2,
    )[:, :, 0]
    return SupportResult(
        present=present,
        distance=jnp.where(
            present,
            jnp.maximum(0.0, nearest_distance),
            jnp.float32(max_distance),
        ),
        surface_y=jnp.where(present, selected_y, points[:, :, 1]),
        geometry_exhausted=geometry_exhausted,
    )


def _first_contact_once(
    position: Array,
    displacement: Array,
    bounds: Array,
    box_minimum: Array,
    box_maximum: Array,
    valid: Array,
) -> tuple[Array, Array, Array, Array]:
    moving_minimum = position[:, None, :] + bounds[:, None, :3]
    moving_maximum = position[:, None, :] + bounds[:, None, 3:]
    delta = displacement[:, None, :]
    positive = delta > _FLOAT_EPSILON
    negative = delta < -_FLOAT_EPSILON
    nonzero = positive | negative
    safe_delta = jnp.where(nonzero, delta, 1.0)

    positive_entry = (box_minimum - moving_maximum) / safe_delta
    positive_exit = (box_maximum - moving_minimum) / safe_delta
    negative_entry = (box_maximum - moving_minimum) / safe_delta
    negative_exit = (box_minimum - moving_maximum) / safe_delta
    static_overlap = (
        (moving_maximum > box_minimum + _FLOAT_EPSILON)
        & (moving_minimum < box_maximum - _FLOAT_EPSILON)
    )
    axis_entry = jnp.where(
        positive,
        positive_entry,
        jnp.where(
            negative,
            negative_entry,
            jnp.where(static_overlap, -jnp.inf, jnp.inf),
        ),
    )
    axis_exit = jnp.where(
        positive,
        positive_exit,
        jnp.where(
            negative,
            negative_exit,
            jnp.where(static_overlap, jnp.inf, -jnp.inf),
        ),
    )
    entry = jnp.max(axis_entry, axis=2)
    exit = jnp.min(axis_exit, axis=2)
    hit = (
        valid
        & (entry <= exit + _FLOAT_EPSILON)
        & (exit >= -_FLOAT_EPSILON)
        & (entry >= -_FLOAT_EPSILON)
        & (entry <= 1.0 + _FLOAT_EPSILON)
    )
    time = jnp.where(hit, jnp.maximum(0.0, entry), jnp.inf)
    box_index = jnp.argmin(time, axis=1)
    earliest = jnp.min(time, axis=1)
    has_hit = jnp.isfinite(earliest)
    selected_axis_entry = jnp.take_along_axis(
        axis_entry,
        box_index[:, None, None],
        axis=1,
    )[:, 0, :]
    normal_axis = jnp.argmax(selected_axis_entry, axis=1)
    axis_delta = jnp.take_along_axis(
        displacement,
        normal_axis[:, None],
        axis=1,
    )[:, 0]
    normal = (
        jax.nn.one_hot(normal_axis, 3, dtype=jnp.float32)
        * -jnp.sign(axis_delta)[:, None]
    )
    normal = jnp.where(has_hit[:, None], normal, 0.0)
    earliest = jnp.where(has_hit, jnp.clip(earliest, 0.0, 1.0), 1.0)
    contact_point = position + displacement * earliest[:, None]
    return earliest, contact_point, normal, has_hit


def _sweep_once(
    position: Array,
    displacement: Array,
    bounds: Array,
    box_minimum: Array,
    box_maximum: Array,
    valid: Array,
    skin_distance: Array = _SKIN_DISTANCE,
) -> tuple[Array, Array, Array, Array]:
    earliest, _, normal, has_hit = _first_contact_once(
        position,
        displacement,
        bounds,
        box_minimum,
        box_maximum,
        valid,
    )
    length = jnp.linalg.norm(displacement, axis=1)
    skin_fraction = jnp.minimum(
        earliest,
        skin_distance / jnp.maximum(length, _FLOAT_EPSILON),
    )
    travel = jnp.where(has_hit, earliest - skin_fraction, 1.0)
    next_position = position + displacement * travel[:, None]
    remaining = displacement * (1.0 - earliest)[:, None]
    normal_component = jnp.sum(remaining * normal, axis=1, keepdims=True)
    remaining = jnp.where(
        has_hit[:, None],
        remaining - normal * normal_component,
        0.0,
    )
    return next_position, remaining, normal, has_hit


def aabb_grounded(
    geometry: GeometryState,
    position: Array,
    *,
    probe_distance: float = 0.002,
) -> GroundResult:
    """Test whether the entity AABB has solid support immediately below it."""

    position = jnp.asarray(position, dtype=jnp.float32)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position must have shape [batch, 3]")
    batch = position.shape[0]
    world = _broadcast_geometry(geometry, batch)
    displacement = jnp.zeros_like(position).at[:, 1].set(
        -jnp.float32(probe_distance)
    )
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
        world.agent_bounds,
    )
    require_full_grid = jnp.any(local_capacity_exceeded)

    def test(
        minimum: Array,
        maximum: Array,
        valid: Array,
    ) -> GroundResult:
        _, _, normal, hit = _sweep_once(
            position,
            displacement,
            world.agent_bounds,
            minimum,
            maximum,
            valid,
        )
        return GroundResult(
            grounded=hit & (normal[:, 1] > 0.5),
            geometry_exhausted=geometry_exhausted,
        )

    def full_test(_: None) -> GroundResult:
        minimum, maximum, valid = _world_collision_boxes(world)
        return test(minimum, maximum, valid)

    def local_test(_: None) -> GroundResult:
        return test(local_minimum, local_maximum, local_valid)

    return jax.lax.cond(
        require_full_grid,
        full_test,
        local_test,
        operand=None,
    )


def aabb_support_sweep(
    geometry: GeometryState,
    position: Array,
    displacement: Array,
    bounds: Array | None = None,
    *,
    probe_distance: float = 0.002,
) -> SupportSweepResult:
    """Find where a grounded horizontal AABB first leaves connected support.

    Hytale's walk controller receives a floor slide interval from native
    collision detection. When the interval ends before the requested motion,
    the controller moves exactly to that edge, switches to falling, and
    applies gravity on the following tick. This kernel reconstructs the
    connected support interval analytically from exact captured detail boxes.

    Only X/Z displacement participates. Support intervals are unioned in a
    fixed loop so adjacent boxes and detail boxes do not create false ledges.
    The configured loop count covers every detail box in one native cell; a
    motion that leaves the captured cube still reports exhaustion separately.
    """

    position = jnp.asarray(position, dtype=jnp.float32)
    displacement = jnp.asarray(displacement, dtype=jnp.float32)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position must have shape [batch, 3]")
    if displacement.shape != position.shape:
        raise ValueError("displacement must match position")
    batch = position.shape[0]
    world = _broadcast_geometry(geometry, batch)
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
    horizontal = displacement.at[:, 1].set(0.0)
    candidate_displacement = horizontal.at[:, 1].set(
        -jnp.float32(probe_distance)
    )
    (
        local_minimum,
        local_maximum,
        local_valid,
        local_capacity_exceeded,
        geometry_exhausted,
    ) = _local_motion_boxes(
        world,
        position,
        candidate_displacement,
        entity_bounds,
    )
    require_full_grid = jnp.any(local_capacity_exceeded)

    def evaluate(
        minimum: Array,
        maximum: Array,
        valid: Array,
    ) -> SupportSweepResult:
        return _support_sweep_with_boxes(
            position,
            horizontal,
            entity_bounds,
            minimum,
            maximum,
            valid,
            probe_distance,
            geometry_exhausted,
        )

    def full_sweep(_: None) -> SupportSweepResult:
        minimum, maximum, valid = _world_collision_boxes(world)
        return evaluate(minimum, maximum, valid)

    def local_sweep(_: None) -> SupportSweepResult:
        return evaluate(local_minimum, local_maximum, local_valid)

    return jax.lax.cond(
        require_full_grid,
        full_sweep,
        local_sweep,
        operand=None,
    )


def _support_sweep_with_boxes(
    position: Array,
    displacement: Array,
    bounds: Array,
    minimum: Array,
    maximum: Array,
    valid: Array,
    probe_distance: float,
    geometry_exhausted: Array,
) -> SupportSweepResult:
    moving_minimum = position[:, None, :] + bounds[:, None, :3]
    moving_maximum = position[:, None, :] + bounds[:, None, 3:]
    delta = displacement[:, None, :]
    axes = jnp.asarray([0, 2], dtype=jnp.int32)
    moving_minimum = jnp.take(moving_minimum, axes, axis=2)
    moving_maximum = jnp.take(moving_maximum, axes, axis=2)
    box_minimum = jnp.take(minimum, axes, axis=2)
    box_maximum = jnp.take(maximum, axes, axis=2)
    delta = jnp.take(delta, axes, axis=2)

    positive = delta > _FLOAT_EPSILON
    negative = delta < -_FLOAT_EPSILON
    nonzero = positive | negative
    safe_delta = jnp.where(nonzero, delta, 1.0)
    positive_entry = (box_minimum - moving_maximum) / safe_delta
    positive_exit = (box_maximum - moving_minimum) / safe_delta
    negative_entry = (box_maximum - moving_minimum) / safe_delta
    negative_exit = (box_minimum - moving_maximum) / safe_delta
    static_overlap = (
        (moving_maximum > box_minimum + _FLOAT_EPSILON)
        & (moving_minimum < box_maximum - _FLOAT_EPSILON)
    )
    axis_entry = jnp.where(
        positive,
        positive_entry,
        jnp.where(
            negative,
            negative_entry,
            jnp.where(static_overlap, -jnp.inf, jnp.inf),
        ),
    )
    axis_exit = jnp.where(
        positive,
        positive_exit,
        jnp.where(
            negative,
            negative_exit,
            jnp.where(static_overlap, jnp.inf, -jnp.inf),
        ),
    )
    interval_start = jnp.max(axis_entry, axis=2)
    interval_end = jnp.min(axis_exit, axis=2)
    foot_y = position[:, None, 1] + bounds[:, None, 1]
    support_distance = foot_y - maximum[..., 1]
    interval_valid = (
        valid
        & (support_distance >= -_FLOAT_EPSILON)
        & (
            support_distance
            <= jnp.float32(probe_distance) + _FLOAT_EPSILON
        )
        & (interval_start <= interval_end + _FLOAT_EPSILON)
        & (interval_end >= -_FLOAT_EPSILON)
        & (interval_start <= jnp.float32(1.0) + _FLOAT_EPSILON)
    )
    contains_start = (
        interval_valid
        & (interval_start <= _FLOAT_EPSILON)
        & (interval_end >= -_FLOAT_EPSILON)
    )
    reach = jnp.max(
        jnp.where(contains_start, interval_end, -jnp.inf),
        axis=1,
    )
    has_start_support = jnp.isfinite(reach)
    reach = jnp.where(has_start_support, jnp.maximum(reach, 0.0), 0.0)

    def extend_once(current):
        connected = (
            interval_valid
            & (interval_start <= current[:, None] + _FLOAT_EPSILON)
            & (interval_end >= -_FLOAT_EPSILON)
        )
        extension = jnp.max(
            jnp.where(connected, interval_end, current[:, None]),
            axis=1,
        )
        return jnp.maximum(current, extension)

    # Most ordinary floors are covered by the starting support box for the
    # complete microtick. Avoid nine redundant full-box passes in that common
    # case while retaining the same bounded fixed-capacity transitive closure
    # for stairs and adjacent detail boxes.
    unresolved = has_start_support & (
        reach < jnp.float32(1.0) - _FLOAT_EPSILON
    )

    def continue_chain(carry):
        iteration, _, changed = carry
        return (
            (iteration < jnp.int32(_SUPPORT_CHAIN_ITERATIONS))
            & changed
        )

    def extend_chain(carry):
        iteration, current, _ = carry
        extended = extend_once(current)
        changed = jnp.any(
            unresolved
            & (extended > current + _FLOAT_EPSILON)
            & (extended < jnp.float32(1.0) - _FLOAT_EPSILON)
        )
        return iteration + jnp.int32(1), extended, changed

    iterations, reach, _ = jax.lax.while_loop(
        continue_chain,
        extend_chain,
        (
            jnp.int32(0),
            reach,
            jnp.any(unresolved),
        ),
    )
    next_reach = jax.lax.cond(
        iterations >= jnp.int32(_SUPPORT_CHAIN_ITERATIONS),
        extend_once,
        lambda current: current,
        reach,
    )
    chain_exhausted = (
        (iterations >= jnp.int32(_SUPPORT_CHAIN_ITERATIONS))
        & has_start_support
        & (reach < jnp.float32(1.0) - _FLOAT_EPSILON)
        & (next_reach > reach + _FLOAT_EPSILON)
    )
    leaves_support = (
        has_start_support
        & (reach < jnp.float32(1.0) - _FLOAT_EPSILON)
    )
    return SupportSweepResult(
        travel_fraction=jnp.where(
            leaves_support,
            jnp.clip(reach, 0.0, 1.0),
            jnp.float32(1.0),
        ),
        leaves_support=leaves_support,
        geometry_exhausted=geometry_exhausted | chain_exhausted,
    )


def aabb_first_contact_result(
    geometry: GeometryState,
    position: Array,
    displacement: Array,
    bounds: Array,
    *,
    include_closed_endpoint: bool = False,
) -> AabbFirstContactResult:
    """Return the first exact local-geometry contact without slide response.

    Motion sweeps preserve the native half-open destination candidate set by
    default. Zero-width camera rays opt into the closed endpoint because a
    block face exactly at the authored ray distance is a valid contact.
    """

    points = jnp.asarray(position, dtype=jnp.float32)
    delta = jnp.asarray(displacement, dtype=jnp.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("position must have shape [batch, 3]")
    if delta.shape != points.shape:
        raise ValueError("displacement must match position")
    batch = points.shape[0]
    world = _broadcast_geometry(geometry, batch)
    entity_bounds = jnp.asarray(bounds, dtype=jnp.float32)
    if entity_bounds.shape == (6,):
        entity_bounds = entity_bounds[None, :]
    if entity_bounds.ndim != 2 or entity_bounds.shape[1] != 6:
        raise ValueError("bounds must have shape [batch|1, 6]")
    if entity_bounds.shape[0] not in (1, batch):
        raise ValueError("bounds batch must be one or match position")
    entity_bounds = jnp.broadcast_to(entity_bounds, (batch, 6))
    (
        local_minimum,
        local_maximum,
        local_valid,
        local_capacity_exceeded,
        geometry_exhausted,
    ) = _local_motion_boxes(
        world,
        points,
        delta,
        entity_bounds,
        include_closed_endpoint=include_closed_endpoint,
    )
    require_full_grid = jnp.any(local_capacity_exceeded)

    def full_resolution(_: None) -> AabbFirstContactResult:
        minimum, maximum, valid = _world_collision_boxes(world)
        return _aabb_first_contact_with_boxes(
            points,
            delta,
            entity_bounds,
            minimum,
            maximum,
            valid,
            geometry_exhausted,
        )

    def local_resolution(_: None) -> AabbFirstContactResult:
        return _aabb_first_contact_with_boxes(
            points,
            delta,
            entity_bounds,
            local_minimum,
            local_maximum,
            local_valid,
            geometry_exhausted,
        )

    return jax.lax.cond(
        require_full_grid,
        full_resolution,
        local_resolution,
        operand=None,
    )


def _aabb_first_contact_with_boxes(
    position: Array,
    displacement: Array,
    bounds: Array,
    minimum: Array,
    maximum: Array,
    valid: Array,
    geometry_exhausted: Array,
) -> AabbFirstContactResult:
    fraction, point, normal, hit = _first_contact_once(
        position,
        displacement,
        bounds,
        minimum,
        maximum,
        valid,
    )
    return AabbFirstContactResult(
        hit=hit,
        hit_fraction=fraction,
        contact_point=jnp.where(hit[:, None], point, 0.0),
        contact_normal=jnp.where(hit[:, None], normal, 0.0),
        geometry_exhausted=geometry_exhausted,
    )
