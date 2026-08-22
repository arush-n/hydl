"""Candidate-selection helpers internal to the World-token package."""

from typing import NamedTuple
import jax
import jax.numpy as jnp
from hytalegym.geometry.contract import CELL_COUNT, CELL_RADIUS
from hytalegym.jax.world.perception.los import (
    GeometryProvider,
    geometry_perception_line_of_sight_result,
    geometry_perception_target_cell_visible_result,
)
from hytalegym.jax.world.perception import native_view_sector
from hytalegym.jax.world.region.mutable_geometry import (
    lookup_region_geometry_blocks,
)
from hytalegym.jax.world.region.types import RegionGeometryState
from hytalegym.jax.world.traversal import (
    GRAPH_DIAGNOSTIC_STATEFUL_TRANSITIONS,
    TraversalTokenObservation,
)
from hytalegym.jax.world.types import COLLISION_SHAPE_FULL_CUBE


Array = jax.Array


WORLD_TOKEN_KIND_TRAVERSAL_SURFACE = 1


WORLD_TOKEN_KIND_COLLISION_EXCEPTION = 2


_SIDE = CELL_RADIUS * 2 + 1


_LOCAL_CELL_OFFSETS = jnp.asarray(
    [
        (x, y, z)
        for x in range(-CELL_RADIUS, CELL_RADIUS + 1)
        for y in range(-CELL_RADIUS, CELL_RADIUS + 1)
        for z in range(-CELL_RADIUS, CELL_RADIUS + 1)
    ],
    dtype=jnp.int32,
)


_UNIT_CUBE = jnp.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), dtype=jnp.float32)

# A dense 5x5 neighbourhood plus four-block cardinal sight lines.  Ordinary
# terrain used to disappear from policy tokens because only unusual collision
# shapes were candidates.  This bounded stencil leaves eleven of the default
# 44 token slots for slabs, stairs, doors, and authored traversal nodes.
_SURFACE_XZ_OFFSETS = jnp.asarray(
    [
        (x, z)
        for x in range(-2, 3)
        for z in range(-2, 3)
    ]
    + [
        (distance * x, distance * z)
        for distance in (3, 4)
        for x, z in ((-1, 0), (0, -1), (0, 1), (1, 0))
    ],
    dtype=jnp.int32,
)


class _CandidatePayload(NamedTuple):
    kind: Array
    provenance: Array
    source_index: Array
    world_cell: Array
    relative_position: Array
    clearance: Array
    semantic_flags: Array
    dynamic_blocked: Array
    collision_box_mask: Array
    collision_boxes_relative: Array


class _CandidateSource(NamedTuple):
    mask: Array
    los_target: Array
    payload: _CandidatePayload


class _VisibleCandidates(NamedTuple):
    eligible: Array
    payload: _CandidatePayload


class _Visibility(NamedTuple):
    visible: Array
    failed: Array


class _SelectedEdges(NamedTuple):
    mask: Array
    destination: Array
    cost: Array
    kind: Array
    flags: Array


def _collision_exception_candidates(
    world: GeometryProvider,
    actor_position: Array,
    provenance: int,
) -> tuple[_CandidateSource, Array]:
    if isinstance(world, RegionGeometryState):
        return _region_collision_exception_candidates(
            world,
            actor_position,
            provenance,
        )

    batch, actors, _ = actor_position.shape
    cells = world.collision_exception_cell.astype(jnp.int32)
    cell_mask = cells >= 0
    safe_cells = jnp.clip(cells, 0, CELL_COUNT - 1)
    offsets = _cell_offsets(safe_cells)
    cell_base = world.origin[:, None, :].astype(jnp.float32) + offsets

    box_index = world.collision_exception_box_index.astype(jnp.int32)
    safe_box_index = jnp.clip(
        box_index,
        0,
        world.collision_exception_boxes.shape[1] - 1,
    )
    boxes = jax.vmap(lambda row, index: row[index])(
        world.collision_exception_boxes,
        safe_box_index,
    )
    box_mask = (box_index >= 0) & cell_mask[..., None]
    minimum = jnp.min(
        jnp.where(box_mask[..., None], boxes[..., :3], jnp.inf),
        axis=2,
    )
    maximum = jnp.max(
        jnp.where(box_mask[..., None], boxes[..., 3:], -jnp.inf),
        axis=2,
    )
    shape_center = cell_base + (minimum + maximum) * jnp.float32(0.5)
    world_position = jnp.broadcast_to(
        shape_center[:, None, :, :],
        (batch, actors, cells.shape[1], 3),
    )
    relative_position = world_position - actor_position[:, :, None, :]
    relative_base = cell_base[:, None, :, :] - actor_position[:, :, None, :]
    boxes_relative = (
        boxes[:, None, :, :, :]
        + jnp.concatenate(
            (relative_base, relative_base),
            axis=3,
        )[:, :, :, None, :]
    )
    flags = jnp.take_along_axis(world.flags, safe_cells, axis=1)
    flags = jnp.broadcast_to(
        flags[:, None, :],
        cell_mask.shape[:1]
        + (
            actors,
            cell_mask.shape[1],
        ),
    )
    exception = _CandidateSource(
            mask=jnp.broadcast_to(
                cell_mask[:, None, :],
                (batch, actors, cells.shape[1]),
            ),
            los_target=jnp.broadcast_to(
                (cell_base + jnp.float32(0.5))[:, None, :, :],
                world_position.shape,
            ),
            payload=_CandidatePayload(
                kind=jnp.full(
                    (batch, actors, cells.shape[1]),
                    WORLD_TOKEN_KIND_COLLISION_EXCEPTION,
                    dtype=jnp.uint8,
                ),
                provenance=jnp.full(
                    (batch, actors, cells.shape[1]),
                    provenance,
                    dtype=jnp.uint8,
                ),
                source_index=jnp.broadcast_to(
                    safe_cells[:, None, :],
                    (batch, actors, cells.shape[1]),
                ),
                world_cell=jnp.broadcast_to(
                    (
                        world.origin[:, None, :]
                        + offsets.astype(jnp.int32)
                    )[:, None, :, :],
                    (batch, actors, cells.shape[1], 3),
                ),
                relative_position=relative_position,
                clearance=jnp.zeros(
                    (batch, actors, cells.shape[1]),
                    dtype=jnp.float32,
                ),
                semantic_flags=jnp.broadcast_to(
                    flags.astype(jnp.uint16),
                    (batch, actors, cells.shape[1]),
                ),
                dynamic_blocked=jnp.zeros(
                    (batch, actors, cells.shape[1]),
                    dtype=jnp.bool_,
                ),
                collision_box_mask=jnp.broadcast_to(
                    box_mask[:, None, :, :],
                    (batch, actors) + box_mask.shape[1:],
                ),
                collision_boxes_relative=boxes_relative,
            ),
        )
    full_cube = (
        world.cell_mask
        & (world.collision_shape_index == jnp.int16(COLLISION_SHAPE_FULL_CUBE))
    )
    occupied = world.cell_mask & (world.collision_shape_index != jnp.int16(-1))
    surface = _surface_candidates(
        full_cube=jnp.broadcast_to(
            full_cube[:, None, :], (batch, actors, CELL_COUNT)
        ),
        occupied=jnp.broadcast_to(
            occupied[:, None, :], (batch, actors, CELL_COUNT)
        ),
        cells=jnp.broadcast_to(
            (
                world.origin[:, None, :]
                + _LOCAL_CELL_OFFSETS[None, :, :]
            )[:, None, :, :],
            (batch, actors, CELL_COUNT, 3),
        ),
        flags=jnp.broadcast_to(
            world.flags[:, None, :], (batch, actors, CELL_COUNT)
        ),
        actor_position=actor_position,
        provenance=provenance,
        detail_capacity=box_mask.shape[2],
    )
    return _concatenate_sources(surface, exception), jnp.ones(
        (batch, actors), dtype=jnp.bool_
    )


def _region_collision_exception_candidates(
    world: RegionGeometryState,
    actor_position: Array,
    provenance: int,
) -> tuple[_CandidateSource, Array]:
    """Project the exact local 9x9x9 exception window directly from a Region."""

    batch, actors, _ = actor_position.shape
    center = jnp.floor(actor_position).astype(jnp.int32)
    cells = center[:, :, None, :] + _LOCAL_CELL_OFFSETS
    selected = lookup_region_geometry_blocks(
        world,
        cells.reshape(batch, actors * CELL_COUNT, 3),
        require_core=False,
    )

    def actor_rows(values: Array) -> Array:
        return values.reshape((batch, actors, CELL_COUNT) + values.shape[2:])

    available = actor_rows(selected.available)
    filler_available = actor_rows(selected.filler_root_available)
    filler_offset = actor_rows(selected.filler_root_offset)
    boxes = actor_rows(selected.collision_boxes)
    translated_boxes = boxes + jnp.concatenate(
        (-filler_offset, -filler_offset),
        axis=3,
    )[..., None, :]
    box_mask = (
        actor_rows(selected.collision_box_mask)
        & available[..., None]
        & filler_available[..., None]
    )
    full_cube = (
        box_mask[..., 0]
        & ~jnp.any(box_mask[..., 1:], axis=3)
        & jnp.all(translated_boxes[..., 0, :] == _UNIT_CUBE, axis=3)
    )
    occupied = jnp.any(box_mask, axis=3)
    cell_mask = occupied & ~full_cube
    minimum = jnp.min(
        jnp.where(box_mask[..., None], translated_boxes[..., :3], jnp.inf),
        axis=3,
    )
    maximum = jnp.max(
        jnp.where(box_mask[..., None], translated_boxes[..., 3:], -jnp.inf),
        axis=3,
    )
    cell_base = cells.astype(jnp.float32)
    relative_base = cell_base - actor_position[:, :, None, :]
    source_index = jnp.broadcast_to(
        jnp.arange(CELL_COUNT, dtype=jnp.int32)[None, None, :],
        cell_mask.shape,
    )
    flags = actor_rows(selected.flags).astype(jnp.uint16)
    exception = _CandidateSource(
            mask=cell_mask,
            los_target=cell_base + jnp.float32(0.5),
            payload=_CandidatePayload(
                kind=jnp.full(
                    cell_mask.shape,
                    WORLD_TOKEN_KIND_COLLISION_EXCEPTION,
                    dtype=jnp.uint8,
                ),
                provenance=jnp.full(
                    cell_mask.shape,
                    provenance,
                    dtype=jnp.uint8,
                ),
                source_index=source_index,
                world_cell=cells,
                relative_position=(
                    cell_base
                    + (minimum + maximum) * jnp.float32(0.5)
                    - actor_position[:, :, None, :]
                ),
                clearance=jnp.zeros(cell_mask.shape, dtype=jnp.float32),
                semantic_flags=flags,
                dynamic_blocked=jnp.zeros(cell_mask.shape, dtype=jnp.bool_),
                collision_box_mask=box_mask,
                collision_boxes_relative=(
                    translated_boxes
                    + jnp.concatenate((relative_base, relative_base), axis=3)[
                        ..., None, :
                    ]
                ),
            ),
        )
    surface = _surface_candidates(
        full_cube=full_cube,
        occupied=occupied,
        cells=cells,
        flags=actor_rows(selected.flags).astype(jnp.uint16),
        actor_position=actor_position,
        provenance=provenance,
        detail_capacity=box_mask.shape[3],
    )
    return _concatenate_sources(surface, exception), jnp.all(
        available & filler_available, axis=2
    )


def _surface_candidates(
    *,
    full_cube: Array,
    occupied: Array,
    cells: Array,
    flags: Array,
    actor_position: Array,
    provenance: int,
    detail_capacity: int,
) -> _CandidateSource:
    """Select ordinary walk surfaces from the exact local 9x9x9 geometry."""

    batch, actors, count = full_cube.shape
    if count != CELL_COUNT:
        raise ValueError("local surface projection requires 9x9x9 geometry")
    columns = _SIDE * _SIDE

    def by_column(value: Array) -> Array:
        tail = value.shape[3:]
        return value.reshape((batch, actors, _SIDE, _SIDE, _SIDE) + tail).transpose(
            (0, 1, 2, 4, 3) + tuple(range(5, 5 + len(tail)))
        ).reshape((batch, actors, columns, _SIDE) + tail)

    full_columns = by_column(full_cube)
    occupied_columns = by_column(occupied)
    cell_columns = by_column(cells)
    flag_columns = by_column(flags)
    stencil = (
        (_SURFACE_XZ_OFFSETS[:, 0] + CELL_RADIUS) * _SIDE
        + _SURFACE_XZ_OFFSETS[:, 1]
        + CELL_RADIUS
    )
    full_columns = jnp.take(full_columns, stencil, axis=2)
    occupied_columns = jnp.take(occupied_columns, stencil, axis=2)
    cell_columns = jnp.take(cell_columns, stencil, axis=2)
    flag_columns = jnp.take(flag_columns, stencil, axis=2)

    # A surface is a full collision cube with certified empty geometry above.
    # The top boundary is excluded because its next cell was not captured.
    occupied_above = jnp.concatenate(
        (
            occupied_columns[..., 1:],
            jnp.ones_like(occupied_columns[..., :1]),
        ),
        axis=3,
    )
    surface = full_columns & ~occupied_above
    top = cell_columns[..., 1] + jnp.int32(1)
    vertical_distance = jnp.abs(
        top.astype(jnp.float32) - actor_position[:, :, None, None, 1]
    )
    choice = jnp.argmin(jnp.where(surface, vertical_distance, jnp.inf), axis=3)
    mask = jnp.any(surface, axis=3)

    def select(value: Array) -> Array:
        tail = value.shape[4:]
        index = choice.reshape(choice.shape + (1,) * (1 + len(tail)))
        index = jnp.broadcast_to(index, choice.shape + (1,) + tail)
        return jnp.take_along_axis(value, index, axis=3).squeeze(axis=3)

    selected_cells = select(cell_columns)
    selected_flags = select(flag_columns)
    y_axis = jnp.arange(_SIDE, dtype=jnp.int32)
    higher = y_axis[None, None, None, :] > choice[..., None]
    next_occupied = jnp.min(
        jnp.where(higher & occupied_columns, y_axis, _SIDE), axis=3
    )
    clearance = jnp.maximum(next_occupied - choice - 1, 0).astype(jnp.float32)
    surface_position = selected_cells.astype(jnp.float32) + jnp.asarray(
        (0.5, 1.0, 0.5), dtype=jnp.float32
    )
    relative = surface_position - actor_position[:, :, None, :]
    source_index = (
        (selected_cells[..., 0] - cells[..., 0].min(axis=2)[..., None])
        * (_SIDE * _SIDE)
        + (selected_cells[..., 1] - cells[..., 1].min(axis=2)[..., None])
        * _SIDE
        + (selected_cells[..., 2] - cells[..., 2].min(axis=2)[..., None])
    ).astype(jnp.int32)
    shape = mask.shape
    return _CandidateSource(
        mask=mask,
        los_target=selected_cells.astype(jnp.float32) + jnp.float32(0.5),
        payload=_CandidatePayload(
            kind=jnp.full(shape, WORLD_TOKEN_KIND_TRAVERSAL_SURFACE, dtype=jnp.uint8),
            provenance=jnp.full(shape, provenance, dtype=jnp.uint8),
            source_index=source_index,
            world_cell=selected_cells,
            relative_position=relative,
            clearance=clearance,
            semantic_flags=selected_flags.astype(jnp.uint16),
            dynamic_blocked=jnp.zeros(shape, dtype=jnp.bool_),
            collision_box_mask=jnp.zeros(
                shape + (detail_capacity,), dtype=jnp.bool_
            ),
            collision_boxes_relative=jnp.zeros(
                shape + (detail_capacity, 6), dtype=jnp.float32
            ),
        ),
    )


def _concatenate_sources(*sources: _CandidateSource) -> _CandidateSource:
    return jax.tree_util.tree_map(
        lambda *values: jnp.concatenate(values, axis=2), *sources
    )


def _traversal_candidates(
    traversal: TraversalTokenObservation,
    actor_position: Array,
    actor_eye_position: Array,
    dynamic_state_visible: Array | None,
    edge_state_visible: Array | None,
) -> tuple[_CandidateSource, Array]:
    shape = traversal.token_mask.shape
    dynamic_visible = _optional_boolean_evidence(
        dynamic_state_visible,
        shape,
        "traversal_dynamic_state_visible",
    )
    edge_visible = _optional_boolean_evidence(
        edge_state_visible,
        traversal.edge_mask.shape,
        "traversal_edge_state_visible",
    )
    static_graph = (
        traversal.diagnostics & jnp.uint32(GRAPH_DIAGNOSTIC_STATEFUL_TRANSITIONS)
    ) == 0
    edge_evidence = edge_visible | static_graph[..., None, None]
    world_position = actor_position[:, :, None, :] + traversal.relative_position
    eye_offset = actor_eye_position - actor_position
    return (
        _CandidateSource(
            mask=traversal.token_mask & traversal.available[..., None],
            los_target=world_position + eye_offset[:, :, None, :],
            payload=_CandidatePayload(
                kind=jnp.full(
                    shape,
                    WORLD_TOKEN_KIND_TRAVERSAL_SURFACE,
                    dtype=jnp.uint8,
                ),
                provenance=jnp.broadcast_to(
                    traversal.provenance[..., None],
                    shape,
                ).astype(jnp.uint8),
                source_index=traversal.node_index.astype(jnp.int32),
                world_cell=jnp.floor(world_position).astype(jnp.int32),
                relative_position=traversal.relative_position,
                clearance=traversal.clearance,
                semantic_flags=traversal.node_flags.astype(jnp.uint16),
                dynamic_blocked=(traversal.token_dynamic_blocked & dynamic_visible),
                collision_box_mask=jnp.zeros(
                    shape + (1,),
                    dtype=jnp.bool_,
                ),
                collision_boxes_relative=jnp.zeros(
                    shape + (1, 6),
                    dtype=jnp.float32,
                ),
            ),
        ),
        edge_evidence,
    )


def _pre_los_candidates(
    mask: Array,
    relative_position: Array,
    actor_valid: Array,
    maximum: Array,
    normalized_forward: Array,
    view_angle: Array,
) -> Array:
    distance_squared = jnp.sum(relative_position * relative_position, axis=3)
    in_sector = native_view_sector(
        relative_position[..., (0, 2)],
        normalized_forward[:, :, None, :],
        view_angle[:, :, None],
    )
    centered = jnp.sum(relative_position[..., (0, 2)] ** 2, axis=3) <= jnp.float32(
        1.0e-8
    )
    return (
        mask
        & actor_valid[..., None]
        & (distance_squared <= maximum[..., None] ** 2)
        & (in_sector | centered)
    )


def _candidate_visibility(
    geometry: GeometryProvider,
    eye: Array,
    target: Array,
    candidate_mask: Array,
    role_mask: Array,
    *,
    target_cell: bool,
    max_cells: int | None,
) -> _Visibility:
    starts = jnp.broadcast_to(eye[:, :, None, :], target.shape)
    ends = jnp.where(candidate_mask[..., None], target, starts)
    starts = jnp.transpose(starts, (1, 2, 0, 3))
    ends = jnp.transpose(ends, (1, 2, 0, 3))
    masks = jnp.swapaxes(role_mask, 0, 1)
    query = (
        geometry_perception_target_cell_visible_result
        if target_cell
        else geometry_perception_line_of_sight_result
    )

    def actor_queries(
        actor_starts: Array,
        actor_ends: Array,
        actor_mask: Array,
    ):
        return jax.vmap(
            lambda begin, end: query(
                geometry,
                begin,
                end,
                role_opaque_mask=actor_mask,
                max_cells=max_cells,
            )
        )(actor_starts, actor_ends)

    result = jax.vmap(actor_queries)(starts, ends, masks)

    def restore(values: Array) -> Array:
        return jnp.transpose(values, (2, 0, 1))

    visible = restore(result.visible)
    failed = restore(
        result.geometry_exhausted
        | result.capacity_exceeded
        | result.invalid
        | ~result.role_opacity_applied
    )
    return _Visibility(visible=visible, failed=failed)


def _combine_candidates(
    traversal: _VisibleCandidates | None,
    exception: _VisibleCandidates,
) -> _VisibleCandidates:
    if traversal is None:
        return exception

    detail_capacity = exception.payload.collision_box_mask.shape[3]
    traversal_mask = traversal.payload.collision_box_mask
    if traversal_mask.shape[3] != detail_capacity:
        traversal_boxes = traversal.payload.collision_boxes_relative
        traversal = traversal._replace(
            payload=traversal.payload._replace(
                collision_box_mask=jnp.broadcast_to(
                    traversal_mask,
                    traversal_mask.shape[:3] + (detail_capacity,),
                ),
                collision_boxes_relative=jnp.broadcast_to(
                    traversal_boxes,
                    traversal_boxes.shape[:3] + (detail_capacity, 6),
                ),
            )
        )
    return jax.tree_util.tree_map(
        lambda left, right: jnp.concatenate((left, right), axis=2),
        traversal,
        exception,
    )


def _pad_candidates(
    candidates: _VisibleCandidates,
    capacity: int,
) -> _VisibleCandidates:
    current = candidates.eligible.shape[2]
    if current >= capacity:
        return candidates

    def pad(values: Array) -> Array:
        padding = [(0, 0)] * values.ndim
        padding[2] = (0, capacity - current)
        return jnp.pad(values, padding)

    return jax.tree_util.tree_map(pad, candidates)


def _select_candidates(
    candidates: _VisibleCandidates,
    capacity: int,
) -> tuple[_VisibleCandidates, Array]:
    distance = jnp.sum(
        candidates.payload.relative_position**2,
        axis=3,
    )
    eligible = candidates.eligible
    maximum = jnp.iinfo(jnp.int32).max
    world_cell = candidates.payload.world_cell
    source_index = candidates.payload.source_index
    candidate_index = jnp.broadcast_to(
        jnp.arange(eligible.shape[2], dtype=jnp.int32),
        eligible.shape,
    )
    sorted_values = jax.lax.sort(
        (
            jnp.where(eligible, distance, jnp.inf),
            jnp.where(
                eligible,
                candidates.payload.kind,
                jnp.uint8(255),
            ),
            jnp.where(eligible, world_cell[..., 0], maximum),
            jnp.where(eligible, world_cell[..., 1], maximum),
            jnp.where(eligible, world_cell[..., 2], maximum),
            jnp.where(eligible, source_index, maximum),
            candidate_index,
        ),
        dimension=2,
        is_stable=True,
        num_keys=6,
    )
    selected_index = sorted_values[-1][:, :, :capacity]
    return (
        jax.tree_util.tree_map(
            lambda values: _gather_candidates(values, selected_index),
            candidates,
        ),
        selected_index,
    )


def _remap_selected_edges(
    traversal: TraversalTokenObservation | None,
    edge_state_visible: Array | None,
    candidate_index: Array,
    token_mask: Array,
    traversal_capacity: int,
    edge_capacity: int,
) -> _SelectedEdges:
    output_shape = token_mask.shape + (edge_capacity,)
    if traversal is None or edge_state_visible is None:
        return _SelectedEdges(
            mask=jnp.zeros(output_shape, dtype=jnp.bool_),
            destination=jnp.full(output_shape, -1, dtype=jnp.int32),
            cost=jnp.zeros(output_shape, dtype=jnp.float32),
            kind=jnp.zeros(output_shape, dtype=jnp.uint8),
            flags=jnp.zeros(output_shape, dtype=jnp.uint16),
        )

    selected_traversal = token_mask & (candidate_index < traversal_capacity)
    source_position = jnp.clip(
        candidate_index,
        0,
        traversal_capacity - 1,
    )
    source_axis = jnp.arange(traversal_capacity, dtype=jnp.int32)
    source_match = (
        source_position[..., None] == source_axis[None, None, None, :]
    ) & selected_traversal[..., None]
    source_to_output = jnp.where(
        jnp.any(source_match, axis=2),
        jnp.argmax(source_match, axis=2),
        -1,
    ).astype(jnp.int32)

    source_edge_mask = _gather_candidates(
        traversal.edge_mask,
        source_position,
    )
    source_edge_mask &= _gather_candidates(
        edge_state_visible,
        source_position,
    )
    destination_source = _gather_candidates(
        traversal.edge_destination,
        source_position,
    )
    safe_destination = jnp.clip(
        destination_source,
        0,
        traversal_capacity - 1,
    )
    destination = jnp.take_along_axis(
        jnp.broadcast_to(
            source_to_output[..., None],
            source_to_output.shape + (edge_capacity,),
        ),
        safe_destination,
        axis=2,
    )
    mask = (
        source_edge_mask
        & selected_traversal[..., None]
        & (destination_source >= 0)
        & (destination >= 0)
    )
    cost = _gather_candidates(traversal.edge_cost, source_position)
    kind = _gather_candidates(traversal.edge_kind, source_position)
    flags = _gather_candidates(traversal.edge_flags, source_position)
    return _SelectedEdges(
        mask=mask,
        destination=jnp.where(mask, destination, -1),
        cost=jnp.where(mask, cost, 0.0),
        kind=jnp.where(mask, kind, 0),
        flags=jnp.where(mask, flags, 0),
    )


def _gather_candidates(values: Array, index: Array) -> Array:
    tail = values.shape[3:]
    expanded = index.reshape(index.shape + (1,) * len(tail))
    expanded = jnp.broadcast_to(expanded, index.shape + tail)
    return jnp.take_along_axis(values, expanded, axis=2)


def _cell_offsets(index: Array) -> Array:
    x = index // (_SIDE * _SIDE)
    remainder = index % (_SIDE * _SIDE)
    y = remainder // _SIDE
    z = remainder % _SIDE
    return jnp.stack((x, y, z), axis=2).astype(jnp.float32) - CELL_RADIUS


def _optional_boolean_evidence(
    value: Array | None,
    shape: tuple[int, ...],
    name: str,
) -> Array:
    if value is None:
        return jnp.zeros(shape, dtype=jnp.bool_)
    result = jnp.asarray(value)
    if result.dtype != jnp.bool_:
        raise TypeError(f"{name} must have boolean dtype")
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return result


def _validate_traversal(
    traversal: TraversalTokenObservation,
    batch: int,
    actors: int,
) -> tuple[int, int]:
    if not isinstance(traversal, TraversalTokenObservation):
        raise TypeError("traversal must be TraversalTokenObservation")
    if traversal.available.shape != (batch, actors):
        raise ValueError("traversal row shape must match actors")
    if traversal.provenance.shape != (batch, actors):
        raise ValueError("traversal provenance shape must match actors")
    if traversal.token_mask.ndim != 3 or traversal.token_mask.shape[:2] != (
        batch,
        actors,
    ):
        raise ValueError("traversal token shape must be [B, A, T]")
    capacity = traversal.token_mask.shape[2]
    if capacity <= 0:
        raise ValueError("traversal token capacity must be positive")
    if (
        traversal.relative_position.shape != (batch, actors, capacity, 3)
        or traversal.edge_mask.ndim != 4
        or traversal.edge_mask.shape[:3] != (batch, actors, capacity)
    ):
        raise ValueError("traversal payload shape is inconsistent")
    edge_capacity = traversal.edge_mask.shape[3]
    if edge_capacity <= 0:
        raise ValueError("traversal edge capacity must be positive")
    return capacity, edge_capacity
