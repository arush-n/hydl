"""Fixed-shape native Region traversal graph publication and actor lookup."""

from __future__ import annotations

import math
import operator
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.perception.los import GeometryProvider
from hytalegym.jax.world.region.mutable_geometry import (
    region_geometry_has_mutations,
)
from hytalegym.jax.world.region.types import (
    RegionAtlas,
    RegionGeometryState,
    RegionTraversalAtlas,
)
from hytalegym.jax.world.tokens import (
    WorldGeometryTokenObservation,
    produce_actor_world_geometry_tokens,
)
from hytalegym.jax.world.traversal import (
    GRAPH_DIAGNOSTIC_QUERY_DISTANCE_EXCEEDED,
    GRAPH_DIAGNOSTIC_STATEFUL_TRANSITIONS,
    TRAVERSAL_PROVENANCE_NATIVE_EXACT_GEOMETRY,
    TraversalTokenObservation,
)
from hytalegym.worldgen.region import (
    CAPTURE_BLOCKS_PER_AXIS,
    CHUNK_SIZE,
    CORE_BLOCKS_PER_AXIS,
    CORE_CHUNKS_PER_AXIS,
)
from hytalegym.worldgen.region.traversal import (
    REGION_TRAVERSAL_EDGES_PER_NODE,
    NativeRegionTraversalGraph,
)

REGION_TRAVERSAL_POLICY_STAGING_CAPACITY = 56


def region_traversal_atlas_from_graphs(
    region_atlas: RegionAtlas,
    graphs: Sequence[NativeRegionTraversalGraph | None],
    *,
    node_capacity: int | None = None,
    column_node_capacity: int | None = None,
) -> RegionTraversalAtlas:
    """Pad native graph artifacts and reject every shape/provenance overflow."""

    if not isinstance(region_atlas, RegionAtlas):
        raise TypeError("region_atlas must be a RegionAtlas")
    source = tuple(graphs)
    regions = region_atlas.region_mask.shape[0]
    if len(source) != regions or any(
        value is not None
        and not isinstance(value, NativeRegionTraversalGraph)
        for value in source
    ):
        raise TypeError("graphs must align with the Region atlas capacity")
    present = tuple(value for value in source if value is not None)
    if not present:
        raise ValueError("at least one native Region traversal graph is required")
    query_distance = present[0].metadata.get("query_distance_capacity")
    if any(
        value.metadata.get("query_distance_capacity") != query_distance
        for value in present
    ):
        raise ValueError("Region traversal query distance capacities differ")
    query_distance_value = _positive_finite(
        query_distance,
        "query distance capacity",
    )
    radius = math.ceil(query_distance_value)
    offsets = np.asarray(
        [
            (x, z)
            for x in range(-radius, radius + 1)
            for z in range(-radius, radius + 1)
        ],
        dtype=np.int16,
    )
    required_nodes = max(value.node_count for value in present)
    nodes = _capacity(node_capacity, required_nodes, "node")
    required_columns = _required_column_capacity(present)
    columns = _capacity(
        column_node_capacity,
        required_columns,
        "column node",
    )

    graph_mask = np.zeros(regions, dtype=np.bool_)
    actor_bounds = np.zeros((regions, 6), dtype=np.float32)
    maximum_climb_height = np.zeros(regions, dtype=np.float32)
    maximum_drop_height = np.zeros(regions, dtype=np.float32)
    column_shape = (
        regions,
        CAPTURE_BLOCKS_PER_AXIS,
        CAPTURE_BLOCKS_PER_AXIS,
    )
    column_start = np.full(column_shape, -1, dtype=np.int32)
    column_count = np.zeros(column_shape, dtype=np.uint16)
    node_mask = np.zeros((regions, nodes), dtype=np.bool_)
    node_position = np.zeros((regions, nodes, 3), dtype=np.float32)
    node_clearance = np.zeros((regions, nodes), dtype=np.float32)
    edge_shape = (regions, nodes, REGION_TRAVERSAL_EDGES_PER_NODE)
    edge_mask = np.zeros(edge_shape, dtype=np.bool_)
    edge_destination = np.full(edge_shape, -1, dtype=np.int32)
    edge_cost = np.zeros(edge_shape, dtype=np.float32)
    edge_kind = np.zeros(edge_shape, dtype=np.uint8)
    edge_flags = np.zeros(edge_shape, dtype=np.uint8)
    atlas_core = np.asarray(
        jax.device_get(region_atlas.core_min_chunk_xz),
        dtype=np.int32,
    )
    atlas_mask = np.asarray(
        jax.device_get(region_atlas.region_mask),
        dtype=np.bool_,
    )
    for index, graph in enumerate(source):
        if graph is None:
            continue
        if not atlas_mask[index]:
            raise ValueError("graph cannot occupy an inactive Region slot")
        if not np.array_equal(
            graph.core_min_chunk_xz,
            atlas_core[index],
        ):
            raise ValueError("graph and Region core coordinates disagree")
        count = graph.node_count
        graph_mask[index] = True
        actor_bounds[index] = graph.actor_bounds
        maximum_climb_height[index] = float(
            graph.metadata["maximum_climb_height"]
        )
        maximum_drop_height[index] = float(
            graph.metadata["maximum_drop_height"]
        )
        node_mask[index, :count] = True
        node_position[index, :count] = graph.node_position
        node_clearance[index, :count] = graph.node_clearance
        edge_mask[index, :count] = graph.edge_mask
        edge_destination[index, :count] = graph.edge_destination
        edge_cost[index, :count] = graph.edge_cost
        edge_kind[index, :count] = graph.edge_kind
        edge_flags[index, :count] = graph.edge_flags
        _index_columns(
            graph,
            atlas_core[index],
            column_start[index],
            column_count[index],
            columns,
        )
    return RegionTraversalAtlas(
        graph_mask=jnp.asarray(graph_mask),
        world_id=region_atlas.world_id,
        core_min_chunk_xz=region_atlas.core_min_chunk_xz,
        actor_bounds=jnp.asarray(actor_bounds),
        maximum_climb_height=jnp.asarray(maximum_climb_height),
        maximum_drop_height=jnp.asarray(maximum_drop_height),
        query_distance_capacity=jnp.asarray(
            query_distance_value,
            dtype=jnp.float32,
        ),
        column_offsets=jnp.asarray(offsets),
        column_slot=jnp.arange(columns, dtype=jnp.int32),
        column_node_start=jnp.asarray(column_start),
        column_node_count=jnp.asarray(column_count),
        node_mask=jnp.asarray(node_mask),
        node_position=jnp.asarray(node_position),
        node_clearance=jnp.asarray(node_clearance),
        edge_mask=jnp.asarray(edge_mask),
        edge_destination=jnp.asarray(edge_destination),
        edge_cost=jnp.asarray(edge_cost),
        edge_kind=jnp.asarray(edge_kind),
        edge_flags=jnp.asarray(edge_flags),
    )


def query_region_traversal_tokens(
    atlas: RegionTraversalAtlas,
    environment_world_id: jax.Array,
    actor_positions: jax.Array,
    *,
    token_capacity: int = 44,
    max_distance: float = 4.0,
) -> TraversalTokenObservation:
    """Select one complete native graph neighborhood per actor."""

    if not isinstance(atlas, RegionTraversalAtlas):
        raise TypeError("atlas must be a RegionTraversalAtlas")
    points = jnp.asarray(actor_positions, dtype=jnp.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("actor_positions must have shape [batch, actors, 3]")
    batch, actors, _ = points.shape
    world_ids = jnp.asarray(environment_world_id, dtype=jnp.int32)
    if world_ids.shape != (batch,):
        raise ValueError("environment_world_id must have shape [batch]")
    tokens = _positive_int(token_capacity, "token_capacity")
    if tokens > atlas.node_mask.shape[1]:
        raise ValueError("token_capacity exceeds the graph node capacity")
    distance_value = _positive_finite(max_distance, "max_distance")

    finite = jnp.all(jnp.isfinite(points), axis=2)
    precise = jnp.all(jnp.abs(points) < jnp.float32(1 << 24), axis=2)
    valid_point = finite & precise
    safe_points = jnp.where(valid_point[..., None], points, 0.0)
    blocks = jnp.floor(safe_points).astype(jnp.int32)
    chunks = jnp.floor_divide(blocks[..., (0, 2)], CHUNK_SIZE)
    core_relative = (
        chunks[:, :, None, :]
        - atlas.core_min_chunk_xz[None, None, :, :]
    )
    inside_core = jnp.all(
        (core_relative >= 0)
        & (core_relative < CORE_CHUNKS_PER_AXIS),
        axis=3,
    )
    compatible = atlas.graph_mask[None, None, :] & (
        atlas.world_id[None, None, :]
        == world_ids[:, None, None]
    )
    eligible = inside_core & compatible & valid_point[:, :, None]
    local_xz = blocks[..., (0, 2)] & (CHUNK_SIZE - 1)
    center_delta_twice = (
        jnp.clip(
            core_relative,
            0,
            CORE_CHUNKS_PER_AXIS - 1,
        )
        * (2 * CHUNK_SIZE)
        + local_xz[:, :, None, :] * 2
        + 1
        - CORE_BLOCKS_PER_AXIS
    )
    score = jnp.where(
        eligible,
        jnp.sum(center_delta_twice**2, axis=3),
        jnp.iinfo(jnp.int32).max,
    )
    region_index = jnp.argmin(score, axis=2).astype(jnp.int32)
    has_region = jnp.any(eligible, axis=2)
    safe_region = jnp.where(has_region, region_index, 0)
    distance_supported = (
        jnp.float32(distance_value) <= atlas.query_distance_capacity
    )
    graph_available = has_region & distance_supported

    core_origin = (
        atlas.core_min_chunk_xz[safe_region] * jnp.int32(CHUNK_SIZE)
    )
    local_core = blocks[..., (0, 2)] - core_origin
    query_radius = math.ceil(distance_value)
    query_offsets = jnp.asarray(
        [
            (x, z)
            for x in range(-query_radius, query_radius + 1)
            for z in range(-query_radius, query_radius + 1)
        ],
        dtype=jnp.int32,
    )
    local_capture = (
        local_core[..., None, :]
        + jnp.int32(CHUNK_SIZE)
        + query_offsets[None, None, :, :]
    )
    column_inside = jnp.all(
        (local_capture >= 0)
        & (local_capture < CAPTURE_BLOCKS_PER_AXIS),
        axis=3,
    )
    safe_column = jnp.clip(
        local_capture,
        0,
        CAPTURE_BLOCKS_PER_AXIS - 1,
    )
    column_start = atlas.column_node_start[
        safe_region[..., None],
        safe_column[..., 0],
        safe_column[..., 1],
    ]
    column_count = atlas.column_node_count[
        safe_region[..., None],
        safe_column[..., 0],
        safe_column[..., 1],
    ].astype(jnp.int32)
    # Every column range is contiguous and the slot marker carries the
    # corpus-derived fixed maximum as a static JAX shape.
    column_slot = atlas.column_slot
    candidate_node = (
        jnp.maximum(column_start, 0)[..., None]
        + column_slot[None, None, None, :]
    )
    candidate_mask = (
        column_inside[..., None]
        & (column_slot[None, None, None, :] < column_count[..., None])
        & (candidate_node < atlas.node_mask.shape[1])
    )
    safe_node = jnp.clip(
        candidate_node,
        0,
        atlas.node_mask.shape[1] - 1,
    )
    region_axis = jnp.broadcast_to(
        safe_region[..., None, None],
        safe_node.shape,
    )
    candidate_position = atlas.node_position[region_axis, safe_node]
    relative = candidate_position - safe_points[..., None, None, :]
    distance_squared = jnp.sum(relative * relative, axis=4)
    within = (
        candidate_mask
        & (distance_squared <= jnp.float32(distance_value**2))
        & graph_available[..., None, None]
    )
    flat_within = within.reshape(batch, actors, -1)
    flat_distance = distance_squared.reshape(batch, actors, -1)
    flat_node = safe_node.reshape(batch, actors, -1)
    if tokens > flat_node.shape[2]:
        raise ValueError("token_capacity exceeds gathered traversal candidates")
    count = jnp.sum(flat_within, axis=2)
    capacity_exceeded = count > tokens
    ranked_distance = jnp.where(flat_within, flat_distance, jnp.inf)
    _, ranked_slot = jax.lax.top_k(-ranked_distance, tokens)
    ranked_valid = jnp.take_along_axis(
        flat_within,
        ranked_slot,
        axis=2,
    )
    ranked_node = jnp.take_along_axis(
        flat_node,
        ranked_slot,
        axis=2,
    )
    node_capacity = atlas.node_mask.shape[1]
    canonical_node = jnp.sort(
        jnp.where(ranked_valid, ranked_node, node_capacity),
        axis=2,
    )
    token_mask = (
        (canonical_node < node_capacity)
        & ~capacity_exceeded[..., None]
    )
    node_index = jnp.minimum(canonical_node, node_capacity - 1)
    selected_region = jnp.broadcast_to(
        safe_region[..., None],
        node_index.shape,
    )
    selected_position = atlas.node_position[
        selected_region,
        node_index,
    ]
    clearance = atlas.node_clearance[selected_region, node_index]
    source_edge_mask = atlas.edge_mask[selected_region, node_index]
    destination = atlas.edge_destination[selected_region, node_index]
    edge_cost = atlas.edge_cost[selected_region, node_index]
    edge_kind = atlas.edge_kind[selected_region, node_index]
    edge_flags = atlas.edge_flags[selected_region, node_index]
    flat_node = canonical_node.reshape(-1, tokens)
    flat_destination = destination.reshape(-1, destination.shape[2] * destination.shape[3])
    local_destination = jax.vmap(
        lambda nodes, values: jnp.searchsorted(
            nodes,
            values,
            side="left",
            method="scan",
        )
    )(flat_node, flat_destination).reshape(destination.shape)
    safe_local_destination = jnp.minimum(
        local_destination,
        tokens - 1,
    )
    destination_present = (
        (local_destination < tokens)
        & (
            jnp.take_along_axis(
                canonical_node[..., None, :],
                safe_local_destination,
                axis=3,
            )
            == destination
        )
    )
    selected_destination_mask = jnp.take_along_axis(
        token_mask[..., None, :],
        safe_local_destination,
        axis=3,
    )
    selected_edge_mask = (
        source_edge_mask
        & token_mask[..., None]
        & destination_present
        & selected_destination_mask
    )
    available = graph_available & ~capacity_exceeded
    diagnostics = jnp.where(
        distance_supported,
        jnp.uint32(0),
        jnp.uint32(GRAPH_DIAGNOSTIC_QUERY_DISTANCE_EXCEEDED),
    )
    diagnostics = jnp.broadcast_to(diagnostics, available.shape)
    return TraversalTokenObservation(
        available=available,
        capacity_exceeded=capacity_exceeded,
        diagnostics=diagnostics,
        provenance=jnp.full(
            available.shape,
            TRAVERSAL_PROVENANCE_NATIVE_EXACT_GEOMETRY,
            dtype=jnp.uint8,
        ),
        tile_index=jnp.where(has_region, region_index, -1),
        token_mask=token_mask,
        node_index=jnp.where(token_mask, node_index, -1),
        relative_position=jnp.where(
            token_mask[..., None],
            selected_position - safe_points[..., None, :],
            0.0,
        ),
        clearance=jnp.where(token_mask, clearance, 0.0),
        node_flags=jnp.where(token_mask, jnp.uint16(1), jnp.uint16(0)),
        token_dynamic_blocked=jnp.zeros_like(token_mask),
        edge_mask=selected_edge_mask,
        edge_destination=jnp.where(
            selected_edge_mask,
            local_destination,
            -1,
        ),
        edge_cost=jnp.where(selected_edge_mask, edge_cost, 0.0),
        edge_kind=jnp.where(selected_edge_mask, edge_kind, 0),
        edge_flags=jnp.where(selected_edge_mask, edge_flags, 0),
    )


def produce_actor_region_world_geometry_tokens(
    atlas: RegionTraversalAtlas,
    environment_world_id: jax.Array,
    geometry: GeometryProvider,
    actor_position: jax.Array,
    actor_eye_position: jax.Array,
    actor_forward: jax.Array,
    *,
    role_opaque_mask: jax.Array | None,
    geometry_provenance: int,
    geometry_tile_index: jax.Array | None = None,
    traversal_dynamic_state_visible: jax.Array | None = None,
    traversal_edge_state_visible: jax.Array | None = None,
    traversal_staging_capacity: int = (
        REGION_TRAVERSAL_POLICY_STAGING_CAPACITY
    ),
    token_capacity: int = 44,
    maximum_distance: float = 4.0,
    view_sector_full_angle_radians: float = math.tau,
    max_los_cells: int | None = None,
) -> WorldGeometryTokenObservation:
    """Publish native Region traversal through the actor-legal token gate."""

    traversal = query_region_traversal_tokens(
        atlas,
        environment_world_id,
        actor_position,
        token_capacity=traversal_staging_capacity,
        max_distance=maximum_distance,
    )
    if isinstance(geometry, RegionGeometryState):
        mutated = region_geometry_has_mutations(geometry)[:, None]
        traversal = traversal._replace(
            available=traversal.available & ~mutated,
            diagnostics=traversal.diagnostics
            | jnp.where(
                mutated,
                jnp.uint32(GRAPH_DIAGNOSTIC_STATEFUL_TRANSITIONS),
                jnp.uint32(0),
            ),
        )
    return produce_actor_world_geometry_tokens(
        geometry,
        actor_position,
        actor_eye_position,
        actor_forward,
        role_opaque_mask=role_opaque_mask,
        geometry_provenance=geometry_provenance,
        traversal=traversal,
        traversal_dynamic_state_visible=traversal_dynamic_state_visible,
        traversal_edge_state_visible=traversal_edge_state_visible,
        geometry_tile_index=geometry_tile_index,
        token_capacity=token_capacity,
        maximum_distance=maximum_distance,
        view_sector_full_angle_radians=view_sector_full_angle_radians,
        max_los_cells=max_los_cells,
    )


def estimated_region_traversal_atlas_bytes(
    atlas: RegionTraversalAtlas,
) -> int:
    """Return logical device bytes, excluding allocator/XLA overhead."""

    if not isinstance(atlas, RegionTraversalAtlas):
        raise TypeError("atlas must be a RegionTraversalAtlas")
    return sum(
        int(np.asarray(jax.device_get(value)).nbytes)
        for value in atlas
    )


def _required_column_capacity(
    graphs: Sequence[NativeRegionTraversalGraph],
) -> int:
    result = 0
    for graph in graphs:
        columns = np.floor(
            graph.node_position[:, (0, 2)]
        ).astype(np.int64)
        _, counts = np.unique(columns, axis=0, return_counts=True)
        result = max(result, int(np.max(counts)))
    return result


def _index_columns(
    graph: NativeRegionTraversalGraph,
    core_min_chunk_xz: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    capacity: int,
) -> None:
    capture_origin = (
        core_min_chunk_xz.astype(np.int64) - 1
    ) * CHUNK_SIZE
    for node, position in enumerate(graph.node_position):
        block = np.floor(position[[0, 2]]).astype(np.int64)
        local = block - capture_origin
        if np.any(
            (local < 0) | (local >= CAPTURE_BLOCKS_PER_AXIS)
        ):
            raise ValueError("Region traversal node exceeds capture coverage")
        x, z = (int(value) for value in local)
        count = int(counts[x, z])
        if count >= capacity:
            raise ValueError("Region traversal column exceeds fixed capacity")
        if count == 0:
            starts[x, z] = node
        elif node != int(starts[x, z]) + count:
            raise ValueError("Region traversal column nodes are not contiguous")
        counts[x, z] = count + 1


def _capacity(value: int | None, required: int, label: str) -> int:
    result = required if value is None else _positive_int(value, label)
    if result < required:
        raise ValueError(f"{label} capacity does not cover supplied graphs")
    return result


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _positive_finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return result


__all__ = [
    "REGION_TRAVERSAL_POLICY_STAGING_CAPACITY",
    "estimated_region_traversal_atlas_bytes",
    "produce_actor_region_world_geometry_tokens",
    "query_region_traversal_tokens",
    "region_traversal_atlas_from_graphs",
]
