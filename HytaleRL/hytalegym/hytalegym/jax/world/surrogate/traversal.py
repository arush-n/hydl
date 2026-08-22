"""Exact standable nodes and bounded walk/climb/drop edges from atlas boxes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.surrogate.capabilities import (
    SURROGATE_CAPABILITY_ENTITY_GEOMETRY,
)
from hytalegym.jax.world.surrogate.types import (
    SurrogateAtlas,
    SurrogateRuntimeState,
    SurrogateTraversalAtlas,
    TraversalTokenObservation,
)
from hytalegym.jax.world.traversal import (
    GRAPH_DIAGNOSTIC_OFF_GRID_SURFACE,
    GRAPH_DIAGNOSTIC_STATEFUL_TRANSITIONS,
    GRAPH_DIAGNOSTIC_VERTICAL_TRANSITIONS,
    TRAVERSAL_PROVENANCE_SURROGATE,
)
from hytalegym.worldgen.region import (
    CHUNK_SIZE,
    CORE_BLOCKS_PER_AXIS,
)
from hytalegym.worldgen.surrogate import (
    SurrogateWorldCapacity,
)
from ._geometry import (  # noqa: F401  (re-exported: callers unchanged)
    _EPSILON,
    _SUPPORT_EPSILON,
    _block_key,
    _decode_key,
    _face_intersects_core,
    _gate_allows,
    _host_index,
    _intervals_cover_unit,
    _leaf,
    _movement_bounds,
    _segment_hits_box,
    _segment_interval,
    _strict_overlap,
    _strict_overlap_2d,
    _support_interval,
    _swept_aabb_hits_box,
)
from ._host_tiles import (  # noqa: F401  (re-exported: callers unchanged)
    GRAPH_STATE_MASK_BITS,
    _HostTileGeometry,
)

NODE_STANDABLE = 1
EDGE_WALK = 1
EDGE_CLIMB = 2
EDGE_DROP = 3
EDGE_FLAG_SAFE = 1
EDGE_FLAG_FALL_RISK = 1 << 1
@dataclass(frozen=True)
class CompiledTraversalGraph:
    """One padded core graph; unavailable means mechanics were omitted."""

    available: bool
    diagnostics: int
    actor_bounds: np.ndarray
    maximum_climb_height: float
    maximum_safe_drop_height: float
    drop_safety_supported: bool
    node_mask: np.ndarray
    node_position: np.ndarray
    node_clearance: np.ndarray
    node_support_key: np.ndarray
    node_flags: np.ndarray
    node_gate_runtime_slot: np.ndarray
    node_gate_state_mask: np.ndarray
    edge_mask: np.ndarray
    edge_destination: np.ndarray
    edge_cost: np.ndarray
    edge_kind: np.ndarray
    edge_flags: np.ndarray
    edge_gate_runtime_slot: np.ndarray
    edge_gate_state_mask: np.ndarray

    @property
    def node_count(self) -> int:
        return int(np.count_nonzero(self.node_mask))

    @property
    def edge_count(self) -> int:
        return int(np.count_nonzero(self.edge_mask))

    @property
    def logical_bytes(self) -> int:
        return sum(
            value.nbytes
            for value in (
                self.node_mask,
                self.actor_bounds,
                self.node_position,
                self.node_clearance,
                self.node_support_key,
                self.node_flags,
                self.node_gate_runtime_slot,
                self.node_gate_state_mask,
                self.edge_mask,
                self.edge_destination,
                self.edge_cost,
                self.edge_kind,
                self.edge_flags,
                self.edge_gate_runtime_slot,
                self.edge_gate_state_mask,
            )
        )


def traversal_atlas_from_graphs(
    atlas: SurrogateAtlas,
    graphs: list[CompiledTraversalGraph | None],
) -> SurrogateTraversalAtlas:
    """Align compiled host graphs with immutable world tiles."""

    if not isinstance(atlas, SurrogateAtlas):
        raise TypeError("atlas must be a SurrogateAtlas")
    tile_capacity = atlas.tile_mask.shape[0]
    if len(graphs) != tile_capacity or any(
        graph is not None and not isinstance(graph, CompiledTraversalGraph)
        for graph in graphs
    ):
        raise TypeError("graphs must match the surrogate tile capacity")
    present = [graph for graph in graphs if graph is not None]
    if not present:
        raise ValueError("at least one compiled traversal graph is required")
    node_capacity = present[0].node_mask.shape[0]
    edge_capacity = present[0].edge_mask.shape[1]
    graph_mask = np.zeros(tile_capacity, dtype=np.bool_)
    graph_available = np.zeros(tile_capacity, dtype=np.bool_)
    diagnostics = np.zeros(tile_capacity, dtype=np.uint32)
    actor_bounds = np.zeros((tile_capacity, 6), dtype=np.float32)
    maximum_climb_height = np.zeros(tile_capacity, dtype=np.float32)
    maximum_safe_drop_height = np.zeros(tile_capacity, dtype=np.float32)
    drop_safety_supported = np.zeros(tile_capacity, dtype=np.bool_)
    column_shape = (
        tile_capacity,
        CORE_BLOCKS_PER_AXIS,
        CORE_BLOCKS_PER_AXIS,
    )
    column_node_start = np.full(column_shape, -1, dtype=np.int32)
    column_node_count = np.zeros(column_shape, dtype=np.uint16)
    node_mask = np.zeros((tile_capacity, node_capacity), dtype=np.bool_)
    node_position = np.zeros(
        (tile_capacity, node_capacity, 3),
        dtype=np.float32,
    )
    node_clearance = np.zeros(
        (tile_capacity, node_capacity),
        dtype=np.float32,
    )
    node_support_key = np.full(
        (tile_capacity, node_capacity),
        -1,
        dtype=np.int32,
    )
    node_flags = np.zeros(
        (tile_capacity, node_capacity),
        dtype=np.uint8,
    )
    node_gate_runtime_slot = np.full(
        (tile_capacity, node_capacity),
        -1,
        dtype=np.int16,
    )
    node_gate_state_mask = np.zeros(
        (tile_capacity, node_capacity),
        dtype=np.uint8,
    )
    edge_shape = (tile_capacity, node_capacity, edge_capacity)
    edge_mask = np.zeros(edge_shape, dtype=np.bool_)
    edge_destination = np.full(edge_shape, -1, dtype=np.int32)
    edge_cost = np.zeros(edge_shape, dtype=np.float32)
    edge_kind = np.zeros(edge_shape, dtype=np.uint8)
    edge_flags = np.zeros(edge_shape, dtype=np.uint8)
    edge_gate_runtime_slot = np.full(edge_shape, -1, dtype=np.int16)
    edge_gate_state_mask = np.zeros(edge_shape, dtype=np.uint8)
    for index, graph in enumerate(graphs):
        if graph is None:
            continue
        if (
            graph.node_mask.shape != (node_capacity,)
            or graph.actor_bounds.shape != (6,)
            or graph.node_position.shape != (node_capacity, 3)
            or graph.node_clearance.shape != (node_capacity,)
            or graph.node_support_key.shape != (node_capacity,)
            or graph.node_flags.shape != (node_capacity,)
            or graph.node_gate_runtime_slot.shape != (node_capacity,)
            or graph.node_gate_state_mask.shape != (node_capacity,)
            or graph.edge_mask.shape != (node_capacity, edge_capacity)
            or graph.edge_destination.shape != graph.edge_mask.shape
            or graph.edge_cost.shape != graph.edge_mask.shape
            or graph.edge_kind.shape != graph.edge_mask.shape
            or graph.edge_flags.shape != graph.edge_mask.shape
            or graph.edge_gate_runtime_slot.shape != graph.edge_mask.shape
            or graph.edge_gate_state_mask.shape != graph.edge_mask.shape
        ):
            raise ValueError("compiled traversal graph shapes disagree")
        graph_mask[index] = True
        graph_available[index] = graph.available
        diagnostics[index] = graph.diagnostics
        actor_bounds[index] = graph.actor_bounds
        maximum_climb_height[index] = graph.maximum_climb_height
        maximum_safe_drop_height[index] = graph.maximum_safe_drop_height
        drop_safety_supported[index] = graph.drop_safety_supported
        node_mask[index] = graph.node_mask
        node_position[index] = graph.node_position
        node_clearance[index] = graph.node_clearance
        node_support_key[index] = graph.node_support_key
        node_flags[index] = graph.node_flags
        node_gate_runtime_slot[index] = graph.node_gate_runtime_slot
        node_gate_state_mask[index] = graph.node_gate_state_mask
        edge_mask[index] = graph.edge_mask
        edge_destination[index] = graph.edge_destination
        edge_cost[index] = graph.edge_cost
        edge_kind[index] = graph.edge_kind
        edge_flags[index] = graph.edge_flags
        edge_gate_runtime_slot[index] = graph.edge_gate_runtime_slot
        edge_gate_state_mask[index] = graph.edge_gate_state_mask
        _index_graph_columns(
            graph,
            np.asarray(jax.device_get(atlas.core_min_chunk_xz[index])),
            column_node_start[index],
            column_node_count[index],
        )
    return SurrogateTraversalAtlas(
        *(
            jnp.asarray(value)
            for value in (
                graph_mask,
                graph_available,
                diagnostics,
                atlas.world_id,
                atlas.core_min_chunk_xz,
                actor_bounds,
                maximum_climb_height,
                maximum_safe_drop_height,
                drop_safety_supported,
                column_node_start,
                column_node_count,
                node_mask,
                node_position,
                node_clearance,
                node_support_key,
                node_flags,
                node_gate_runtime_slot,
                node_gate_state_mask,
                edge_mask,
                edge_destination,
                edge_cost,
                edge_kind,
                edge_flags,
                edge_gate_runtime_slot,
                edge_gate_state_mask,
            )
        )
    )


def _index_graph_columns(
    graph: CompiledTraversalGraph,
    core_min_chunk_xz: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
) -> None:
    """Build contiguous core-column ranges for bounded compiled lookup."""

    core_origin = np.asarray(core_min_chunk_xz, dtype=np.int64) * CHUNK_SIZE
    active = np.flatnonzero(graph.node_mask)
    positions = np.asarray(graph.node_position, dtype=np.float64)
    for node_index in active:
        block_xz = np.floor(positions[node_index, (0, 2)]).astype(np.int64)
        local = block_xz - core_origin
        if np.any((local < 0) | (local >= CORE_BLOCKS_PER_AXIS)):
            raise ValueError("traversal node lies outside its core tile")
        x, z = (int(value) for value in local)
        count = int(counts[x, z])
        if count == np.iinfo(np.uint16).max:
            raise ValueError("traversal column exceeds uint16 node capacity")
        if count == 0:
            starts[x, z] = int(node_index)
        elif int(node_index) != int(starts[x, z]) + count:
            raise ValueError("traversal nodes for one core column must be contiguous")
        counts[x, z] = count + 1


def query_traversal_tokens(
    atlas: SurrogateTraversalAtlas,
    runtime: SurrogateRuntimeState,
    actor_positions: jax.Array,
    *,
    token_capacity: int = 32,
    max_distance: float = 12.0,
    include_unsafe_drops: bool = False,
    actor_entity_slots: jax.Array | None = None,
) -> TraversalTokenObservation:
    """Select a bounded graph neighborhood independently for every actor."""

    if not isinstance(atlas, SurrogateTraversalAtlas):
        raise TypeError("atlas must be a SurrogateTraversalAtlas")
    points = jnp.asarray(actor_positions, dtype=jnp.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("actor_positions must have shape [batch, actors, 3]")
    batch, actors, _ = points.shape
    if runtime.environment_world_id.shape != (
        batch,
    ) or runtime.unsupported_mechanics.shape != (batch,):
        raise ValueError("runtime batch does not match actor positions")
    if (
        runtime.stateful_state.ndim != 2
        or runtime.stateful_state.shape[0] != batch
        or runtime.stateful_initialized.shape != runtime.stateful_state.shape
    ):
        raise ValueError("runtime stateful arrays do not match actor positions")
    if (
        isinstance(token_capacity, bool)
        or not isinstance(token_capacity, int)
        or not 1 <= token_capacity <= atlas.node_mask.shape[1]
    ):
        raise ValueError("token_capacity must fit the graph node capacity")
    if (
        isinstance(max_distance, bool)
        or not isinstance(max_distance, (int, float))
        or not math.isfinite(max_distance)
        or max_distance <= 0.0
    ):
        raise ValueError("max_distance must be positive and finite")
    if not isinstance(include_unsafe_drops, bool):
        raise TypeError("include_unsafe_drops must be bool")
    if actor_entity_slots is None:
        actor_slots = jnp.full((batch, actors), -1, dtype=jnp.int32)
    else:
        actor_slots = jnp.asarray(actor_entity_slots, dtype=jnp.int32)
        if actor_slots.shape != (batch, actors):
            raise ValueError("actor_entity_slots must have shape [batch, actors]")

    finite = jnp.all(jnp.isfinite(points), axis=2)
    precise = jnp.all(jnp.abs(points) < jnp.float32(1 << 24), axis=2)
    valid_point = finite & precise
    safe_points = jnp.where(valid_point[..., None], points, 0.0)
    blocks = jnp.floor(safe_points).astype(jnp.int32)
    chunks = jnp.floor_divide(blocks[..., (0, 2)], CHUNK_SIZE)
    relative = chunks[:, :, None, :] - atlas.core_min_chunk_xz[None, None, :, :]
    inside = jnp.all(
        (relative >= 0) & (relative < CORE_BLOCKS_PER_AXIS // CHUNK_SIZE),
        axis=3,
    )
    compatible = atlas.graph_mask[None, None, :] & (
        atlas.world_id[None, None, :] == runtime.environment_world_id[:, None, None]
    )
    eligible = (
        inside
        & compatible
        & valid_point[:, :, None]
        & ~runtime.unsupported_mechanics[:, None, None]
    )
    local_xz = blocks[..., (0, 2)] & (CHUNK_SIZE - 1)
    center_delta_twice = (
        jnp.clip(relative, 0, 2) * (2 * CHUNK_SIZE)
        + local_xz[:, :, None, :] * 2
        + 1
        - CORE_BLOCKS_PER_AXIS
    )
    distance = jnp.sum(center_delta_twice**2, axis=3)
    score = jnp.where(eligible, distance, jnp.iinfo(jnp.int32).max)
    tile_index = jnp.argmin(score, axis=2).astype(jnp.int32)
    has_tile = jnp.any(eligible, axis=2)
    safe_tile = jnp.where(has_tile, tile_index, 0)
    entity_geometry_available = (
        runtime.capability_bits & jnp.uint32(SURROGATE_CAPABILITY_ENTITY_GEOMETRY)
    ) != 0
    graph_available = (
        has_tile & atlas.graph_available[safe_tile] & entity_geometry_available[:, None]
    )
    diagnostics = jnp.where(
        has_tile,
        atlas.diagnostics[safe_tile],
        jnp.uint32(0),
    )

    positions = atlas.node_position[safe_tile]
    node_gate_slot = atlas.node_gate_runtime_slot[safe_tile]
    node_gate_mask = atlas.node_gate_state_mask[safe_tile]
    nodes = atlas.node_mask[safe_tile] & _runtime_gate_available(
        runtime,
        node_gate_slot,
        node_gate_mask,
    )
    relative_position = positions - safe_points[:, :, None, :]
    distance_squared = jnp.sum(relative_position**2, axis=3)
    within = (
        nodes
        & (distance_squared <= jnp.float32(max_distance * max_distance))
        & graph_available[:, :, None]
    )
    count = jnp.sum(within, axis=2)
    capacity_exceeded = count > token_capacity
    ranked_distance = jnp.where(within, distance_squared, jnp.inf)
    _, ranked_index = jax.lax.top_k(
        -ranked_distance,
        token_capacity,
    )
    ranked_valid = jnp.take_along_axis(
        within,
        ranked_index,
        axis=2,
    )
    node_capacity = atlas.node_mask.shape[1]
    canonical_index = jnp.sort(
        jnp.where(ranked_valid, ranked_index, node_capacity),
        axis=2,
    )
    token_mask = (canonical_index < node_capacity) & ~capacity_exceeded[:, :, None]
    node_index = jnp.minimum(canonical_index, node_capacity - 1)
    selected_relative = jnp.take_along_axis(
        relative_position,
        node_index[..., None],
        axis=2,
    )
    clearance = jnp.take_along_axis(
        atlas.node_clearance[safe_tile],
        node_index,
        axis=2,
    )
    node_flags = jnp.take_along_axis(
        atlas.node_flags[safe_tile],
        node_index,
        axis=2,
    )
    selected_world_position = jnp.take_along_axis(
        positions,
        node_index[..., None],
        axis=2,
    )
    actor_bounds = atlas.actor_bounds[safe_tile]
    node_minimum = (
        selected_world_position[..., None, :] + actor_bounds[..., None, None, :3]
    )
    node_maximum = (
        selected_world_position[..., None, :] + actor_bounds[..., None, None, 3:]
    )
    entity_minimum = (runtime.entity_position + runtime.entity_local_bounds[..., :3])[
        :, None, None, :, :
    ]
    entity_maximum = (runtime.entity_position + runtime.entity_local_bounds[..., 3:])[
        :, None, None, :, :
    ]
    entity_slots = jnp.arange(runtime.entity_active.shape[1], dtype=jnp.int32)
    entity_mask = (
        runtime.entity_active
        & runtime.entity_initialized
        & runtime.entity_geometry_supported
        & runtime.entity_collidable
    )[:, None, None, :]
    entity_mask &= entity_slots[None, None, None, :] != actor_slots[..., None, None]
    dynamic_blocked = jnp.any(
        jnp.all(
            (node_maximum > entity_minimum + _EPSILON)
            & (node_minimum < entity_maximum - _EPSILON),
            axis=4,
        )
        & entity_mask,
        axis=3,
    )
    source_edge_mask = atlas.edge_mask[
        safe_tile[..., None],
        node_index,
    ]
    destination = atlas.edge_destination[
        safe_tile[..., None],
        node_index,
    ]
    edge_cost = atlas.edge_cost[
        safe_tile[..., None],
        node_index,
    ]
    edge_kind = atlas.edge_kind[
        safe_tile[..., None],
        node_index,
    ]
    edge_flags = atlas.edge_flags[
        safe_tile[..., None],
        node_index,
    ]
    edge_gate_slot = atlas.edge_gate_runtime_slot[
        safe_tile[..., None],
        node_index,
    ]
    edge_gate_mask = atlas.edge_gate_state_mask[
        safe_tile[..., None],
        node_index,
    ]
    source_edge_mask &= _runtime_gate_available(
        runtime,
        edge_gate_slot,
        edge_gate_mask,
    )
    if not include_unsafe_drops:
        source_edge_mask &= (edge_flags & EDGE_FLAG_SAFE) != 0
    destination_match = destination[..., None] == node_index[:, :, None, None, :]
    destination_present = jnp.any(destination_match, axis=4)
    local_destination = jnp.argmax(
        destination_match,
        axis=4,
    ).astype(jnp.int32)
    selected_destination_mask = token_mask[
        jnp.arange(batch)[:, None, None, None],
        jnp.arange(actors)[None, :, None, None],
        local_destination,
    ]
    selected_edge_mask = (
        source_edge_mask
        & token_mask[..., None]
        & destination_present
        & selected_destination_mask
    )
    available = graph_available & ~capacity_exceeded & jnp.any(token_mask, axis=2)
    return TraversalTokenObservation(
        available=available,
        capacity_exceeded=capacity_exceeded,
        diagnostics=diagnostics,
        provenance=jnp.full(
            available.shape,
            TRAVERSAL_PROVENANCE_SURROGATE,
            dtype=jnp.uint8,
        ),
        tile_index=jnp.where(has_tile, tile_index, -1),
        token_mask=token_mask,
        node_index=jnp.where(token_mask, node_index, -1),
        relative_position=jnp.where(
            token_mask[..., None],
            selected_relative,
            0.0,
        ),
        clearance=jnp.where(token_mask, clearance, 0.0),
        node_flags=jnp.where(token_mask, node_flags, 0),
        token_dynamic_blocked=token_mask & dynamic_blocked,
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


def _runtime_gate_available(
    runtime: SurrogateRuntimeState,
    runtime_slot: jax.Array,
    state_mask: jax.Array,
) -> jax.Array:
    slots = jnp.asarray(runtime_slot, dtype=jnp.int32)
    masks = jnp.asarray(state_mask, dtype=jnp.uint8)
    capacity = runtime.stateful_state.shape[1]
    safe_slot = jnp.clip(slots, 0, capacity - 1)
    batch_axis = jnp.arange(slots.shape[0]).reshape(
        (slots.shape[0],) + (1,) * (slots.ndim - 1)
    )
    state = runtime.stateful_state[batch_axis, safe_slot]
    initialized = runtime.stateful_initialized[batch_axis, safe_slot]
    representable = state < GRAPH_STATE_MASK_BITS
    state_bit = jnp.left_shift(
        jnp.uint8(1),
        jnp.minimum(state, GRAPH_STATE_MASK_BITS - 1),
    )
    dynamic = (
        (slots < capacity) & initialized & representable & ((masks & state_bit) != 0)
    )
    return (slots < 0) | dynamic


def compile_standable_traversal_graph(
    atlas: SurrogateAtlas,
    tile_index: int,
    *,
    actor_bounds: np.ndarray,
    maximum_climb_height: float = 1.3,
    maximum_safe_drop_height: float | None = None,
    capacity: SurrogateWorldCapacity | None = None,
) -> CompiledTraversalGraph:
    """Compile exact ground-walk, climb, and first-landing drop connectivity."""

    if not isinstance(atlas, SurrogateAtlas):
        raise TypeError("atlas must be a SurrogateAtlas")
    layout = SurrogateWorldCapacity() if capacity is None else capacity
    if not isinstance(layout, SurrogateWorldCapacity):
        raise TypeError("capacity must be a SurrogateWorldCapacity")
    tile = _host_index(tile_index, atlas.tile_mask.shape[0])
    bounds = np.asarray(actor_bounds, dtype=np.float32)
    if (
        bounds.shape != (6,)
        or not np.all(np.isfinite(bounds))
        or np.any(bounds[:3] > bounds[3:])
        or np.any(bounds[3:] - bounds[:3] <= 0.0)
    ):
        raise ValueError("actor_bounds must contain one positive finite AABB")
    climb_height = float(maximum_climb_height)
    if (
        isinstance(maximum_climb_height, bool)
        or not math.isfinite(climb_height)
        or climb_height < 0.0
    ):
        raise ValueError("maximum_climb_height must be finite and non-negative")
    drop_safety_supported = maximum_safe_drop_height is not None
    safe_drop_height = (
        0.0 if maximum_safe_drop_height is None else float(maximum_safe_drop_height)
    )
    if (
        isinstance(maximum_safe_drop_height, bool)
        or not math.isfinite(safe_drop_height)
        or safe_drop_height < 0.0
    ):
        raise ValueError("maximum_safe_drop_height must be finite and non-negative")

    geometry = _HostTileGeometry(atlas, tile)
    candidates, diagnostics = geometry.candidate_surfaces()
    nodes: list[tuple[int, int, np.float32, np.float32, int, int, int]] = []
    for x, z in sorted(candidates):
        for foot in sorted(candidates[(x, z)]):
            standable, unsupported = geometry.gated_standable(
                x + 0.5,
                z + 0.5,
                float(foot),
                bounds,
            )
            if unsupported:
                diagnostics |= GRAPH_DIAGNOSTIC_STATEFUL_TRANSITIONS
            if standable is not None:
                clearance, support_key, gate_slot, gate_mask = standable
                nodes.append(
                    (
                        x,
                        z,
                        foot,
                        clearance,
                        support_key,
                        gate_slot,
                        gate_mask,
                    )
                )
                if len(nodes) > layout.graph_node_capacity:
                    raise ValueError(
                        "standable nodes exceed graph node capacity "
                        f"{layout.graph_node_capacity}"
                    )

    edges: list[list[tuple[int, float, int, int, int, int]]] = [[] for _ in nodes]
    nodes_by_column: dict[tuple[int, int], list[int]] = {}
    for index, node in enumerate(nodes):
        nodes_by_column.setdefault((node[0], node[1]), []).append(index)
    for values in nodes_by_column.values():
        values.sort(key=lambda value: float(nodes[value][2]))

    for index, (
        x,
        z,
        foot,
        _clearance,
        _support,
        source_gate_slot,
        source_gate_mask,
    ) in enumerate(nodes):
        for dx, dz in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            target_nodes = nodes_by_column.get((x + dx, z + dz), ())
            selected: list[tuple[int, int]] = []
            drops: list[int] = []
            for target in target_nodes:
                delta = float(nodes[target][2]) - float(foot)
                if abs(delta) <= _SUPPORT_EPSILON:
                    selected.append((target, EDGE_WALK))
                elif 0.0 < delta <= climb_height + _SUPPORT_EPSILON:
                    selected.append((target, EDGE_CLIMB))
                elif delta < -_SUPPORT_EPSILON:
                    drops.append(target)
            if drops:
                selected.append(
                    (
                        max(drops, key=lambda value: float(nodes[value][2])),
                        EDGE_DROP,
                    )
                )

            for target, kind in selected:
                target_gate_slot = nodes[target][5]
                target_gate_mask = nodes[target][6]
                start = (x + 0.5, float(foot), z + 0.5)
                end = (
                    x + dx + 0.5,
                    float(nodes[target][2]),
                    z + dz + 0.5,
                )
                gate, unsupported = geometry.gated_movement(
                    start,
                    end,
                    bounds,
                    kind == EDGE_WALK,
                    (source_gate_slot, source_gate_mask),
                    (target_gate_slot, target_gate_mask),
                )
                if unsupported:
                    diagnostics |= GRAPH_DIAGNOSTIC_STATEFUL_TRANSITIONS
                if gate is not None:
                    cost = 1.0 + abs(end[1] - start[1])
                    safe = kind != EDGE_DROP or (
                        drop_safety_supported
                        and start[1] - end[1] <= safe_drop_height + _SUPPORT_EPSILON
                    )
                    flags = EDGE_FLAG_SAFE if safe else EDGE_FLAG_FALL_RISK
                    edges[index].append((target, cost, kind, flags, gate[0], gate[1]))
        if len(edges[index]) > layout.graph_edges_per_node:
            raise ValueError(
                f"walk edges exceed graph edge capacity {layout.graph_edges_per_node}"
            )
    node_capacity = layout.graph_node_capacity
    edge_capacity = layout.graph_edges_per_node
    node_mask = np.zeros(node_capacity, dtype=np.bool_)
    node_position = np.zeros((node_capacity, 3), dtype=np.float32)
    node_clearance = np.zeros(node_capacity, dtype=np.float32)
    node_support_key = np.full(node_capacity, -1, dtype=np.int32)
    node_flags = np.zeros(node_capacity, dtype=np.uint8)
    node_gate_runtime_slot = np.full(node_capacity, -1, dtype=np.int16)
    node_gate_state_mask = np.zeros(node_capacity, dtype=np.uint8)
    edge_mask = np.zeros((node_capacity, edge_capacity), dtype=np.bool_)
    edge_destination = np.full(
        (node_capacity, edge_capacity),
        -1,
        dtype=np.int32,
    )
    edge_cost = np.zeros((node_capacity, edge_capacity), dtype=np.float32)
    edge_kind = np.zeros((node_capacity, edge_capacity), dtype=np.uint8)
    edge_flags = np.zeros((node_capacity, edge_capacity), dtype=np.uint8)
    edge_gate_runtime_slot = np.full(
        (node_capacity, edge_capacity),
        -1,
        dtype=np.int16,
    )
    edge_gate_state_mask = np.zeros(
        (node_capacity, edge_capacity),
        dtype=np.uint8,
    )
    for index, (
        x,
        z,
        foot,
        clearance,
        support_key,
        gate_slot,
        gate_mask,
    ) in enumerate(nodes):
        node_mask[index] = True
        node_position[index] = (x + 0.5, foot, z + 0.5)
        node_clearance[index] = clearance
        node_support_key[index] = support_key
        node_flags[index] = NODE_STANDABLE
        node_gate_runtime_slot[index] = gate_slot
        node_gate_state_mask[index] = gate_mask
        for edge_index, (
            destination,
            cost,
            kind,
            flags,
            edge_gate_slot,
            edge_gate_mask,
        ) in enumerate(edges[index]):
            edge_mask[index, edge_index] = True
            edge_destination[index, edge_index] = destination
            edge_cost[index, edge_index] = cost
            edge_kind[index, edge_index] = kind
            edge_flags[index, edge_index] = flags
            edge_gate_runtime_slot[index, edge_index] = edge_gate_slot
            edge_gate_state_mask[index, edge_index] = edge_gate_mask
    profile_bounds = bounds.copy()
    arrays = (
        profile_bounds,
        node_mask,
        node_position,
        node_clearance,
        node_support_key,
        node_flags,
        node_gate_runtime_slot,
        node_gate_state_mask,
        edge_mask,
        edge_destination,
        edge_cost,
        edge_kind,
        edge_flags,
        edge_gate_runtime_slot,
        edge_gate_state_mask,
    )
    for value in arrays:
        value.flags.writeable = False
    return CompiledTraversalGraph(
        available=diagnostics == 0,
        diagnostics=diagnostics,
        actor_bounds=profile_bounds,
        maximum_climb_height=climb_height,
        maximum_safe_drop_height=safe_drop_height,
        drop_safety_supported=drop_safety_supported,
        node_mask=node_mask,
        node_position=node_position,
        node_clearance=node_clearance,
        node_support_key=node_support_key,
        node_flags=node_flags,
        node_gate_runtime_slot=node_gate_runtime_slot,
        node_gate_state_mask=node_gate_state_mask,
        edge_mask=edge_mask,
        edge_destination=edge_destination,
        edge_cost=edge_cost,
        edge_kind=edge_kind,
        edge_flags=edge_flags,
        edge_gate_runtime_slot=edge_gate_runtime_slot,
        edge_gate_state_mask=edge_gate_state_mask,
    )


__all__ = [
    "EDGE_CLIMB",
    "EDGE_DROP",
    "EDGE_FLAG_FALL_RISK",
    "EDGE_FLAG_SAFE",
    "EDGE_WALK",
    "GRAPH_DIAGNOSTIC_OFF_GRID_SURFACE",
    "GRAPH_DIAGNOSTIC_STATEFUL_TRANSITIONS",
    "GRAPH_DIAGNOSTIC_VERTICAL_TRANSITIONS",
    "NODE_STANDABLE",
    "CompiledTraversalGraph",
    "compile_standable_traversal_graph",
    "query_traversal_tokens",
    "traversal_atlas_from_graphs",
]
