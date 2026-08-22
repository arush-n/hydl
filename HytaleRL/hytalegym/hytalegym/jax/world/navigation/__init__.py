"""Bounded JAX navigation over native-certified Region traversal edges."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.geometry.contract import FLAG_FLUID
from hytalegym.jax.world.region.atlas import lookup_region_blocks
from hytalegym.jax.world.region.geometry import (
    region_aabb_core_available,
    region_aabb_grounded,
    region_aabb_support_sweep,
    region_resolve_aabb_motion,
)
from hytalegym.jax.world.region.types import (
    RegionGeometryState,
    RegionTraversalAtlas,
)
from ._astar import (  # noqa: F401  (re-exported: callers unchanged)
    ASTAR_PROGRESS_ACCOMPLISHED,
    ASTAR_PROGRESS_COMPUTING,
    ASTAR_PROGRESS_TERMINATED,
    ASTAR_PROGRESS_TERMINATED_OPEN_NODE_LIMIT_EXCEEDED,
    ASTAR_PROGRESS_TERMINATED_TOTAL_NODE_LIMIT_EXCEEDED,
    Array,
    RegionGraphAStarResult,
    _EMPTY,
    _POSITION_OFFSET,
    _SearchState,
    _add_edge,
    _allocate_visited,
    _expand_once,
    _improve_visited,
    _insert_open,
    _native_float_distance,
    _native_position_key,
    _result_from_state,
)
from ._follower import (  # noqa: F401  (re-exported: callers unchanged)
    ASTAR_MAXIMUM_PATH_LENGTH,
    PATH_FOLLOWER_DEFAULT_REJECTION_WEIGHT,
    PATH_FOLLOWER_DIRECTION_REFRESH_DISTANCE,
    PATH_FOLLOWER_EXACT_DISTANCE_SQUARED,
    PATH_FOLLOWER_NEAR_DISTANCE_SQUARED,
    PATH_FOLLOWER_SCHEMA,
    PATH_FOLLOWER_VERSION,
    PathFollowerSteeringResult,
    PathFollowerWaypointResult,
    _batch_float,
    path_follower_contract,
    path_follower_contract_sha256,
    path_follower_steering_result,
    path_follower_waypoint_result,
)

REGION_GRAPH_ASTAR_SCHEMA = "hytalerl_region_graph_astar_v2"
REGION_GRAPH_ASTAR_VERSION = 2
REGION_WALK_PROBE_SUBSET_SCHEMA = "hytalerl_region_walk_probe_subset_v1"
REGION_WALK_PROBE_SUBSET_VERSION = 1

ASTAR_BASE_OPEN_NODES_LIMIT = 80
ASTAR_BASE_TOTAL_NODES_LIMIT = 400
ASTAR_DEFAULT_NODES_PER_TICK = 50
ASTAR_OPEN_NODES_LIMIT = 200
ASTAR_TOTAL_NODES_LIMIT = 900
ASTAR_PROGRESS_UNSTARTED = 0
ASTAR_PROGRESS_ABORTED = 1
_CELL_EPSILON = jnp.float32(1.0e-7)
_PROBE_SUPPORT_DISTANCE = 0.002
_PROBE_CELL_OFFSETS = jnp.asarray(
    [
        (x, y, z)
        for x in range(3)
        for y in range(3)
        for z in range(3)
    ],
    dtype=jnp.int32,
)


class RegionWalkProbeSubsetResult(NamedTuple):
    """Native-certifiable full-step subset of Walk ``probeMove``."""

    certified_available: Array
    travelled_distance: Array
    reached_half_step: Array
    reached_full_step: Array
    valid_position: Array
    half_step_position: Array
    successor_position: Array
    unsupported_mechanics: Array
    geometry_exhausted: Array
    invalid: Array


def region_walk_probe_subset_result(
    geometry: RegionGeometryState,
    start_position: Array,
    direction: Array,
) -> RegionWalkProbeSubsetResult:
    """Certify ordinary dry full steps and reject richer probe branches.

    This is deliberately not a general ``MotionControllerWalk.probeMove``
    replacement. It publishes only rows whose exact Region geometry proves a
    grounded, collision-free, continuously supported move through ordinary
    air. Climb, drop, edge, damage, fluid, slide, and partial-contact rows stay
    unavailable so they cannot open the certified navigation guard.
    """

    start = jnp.asarray(start_position, dtype=jnp.float32)
    delta = jnp.asarray(direction, dtype=jnp.float32)
    if start.ndim != 2 or start.shape[0] == 0 or start.shape[1] != 3:
        raise ValueError("start_position must have non-empty shape [batch,3]")
    if delta.shape != start.shape:
        raise ValueError("direction must match start_position")
    batch = start.shape[0]
    bounds = jnp.asarray(geometry.agent_bounds, dtype=jnp.float32)
    if bounds.ndim != 2 or bounds.shape[1] != 6:
        raise ValueError("geometry agent bounds must have shape [batch|1,6]")
    if bounds.shape[0] not in (1, batch):
        raise ValueError("geometry agent bounds batch must be one or match")
    bounds = jnp.broadcast_to(bounds, (batch, 6))

    finite = jnp.all(jnp.isfinite(start) & jnp.isfinite(delta), axis=1)
    bounds_valid = (
        jnp.all(jnp.isfinite(bounds), axis=1)
        & jnp.all(bounds[:, :3] < bounds[:, 3:], axis=1)
    )
    discrete = jnp.all(
        (delta >= -1.0)
        & (delta <= 1.0)
        & (delta == jnp.round(delta)),
        axis=1,
    )
    horizontal = (delta[:, 1] == 0.0) & (
        (delta[:, 0] != 0.0) | (delta[:, 2] != 0.0)
    )
    input_valid = finite & bounds_valid & discrete & horizontal
    end = start + delta

    grounded = region_aabb_grounded(
        geometry,
        start,
        probe_distance=_PROBE_SUPPORT_DISTANCE,
    )
    support = region_aabb_support_sweep(
        geometry,
        start,
        delta,
        bounds,
        probe_distance=_PROBE_SUPPORT_DISTANCE,
    )
    motion = region_resolve_aabb_motion(
        geometry,
        start,
        delta,
        bounds,
        skin_distance=0.0,
    )
    start_core = region_aabb_core_available(geometry, start, bounds)
    end_core = region_aabb_core_available(geometry, end, bounds)
    ordinary_air, air_exhausted = _region_probe_swept_ordinary_air(
        geometry,
        start,
        end,
        bounds,
    )
    geometry_exhausted = (
        grounded.geometry_exhausted
        | support.geometry_exhausted
        | motion.geometry_exhausted
        | air_exhausted
        | ~start_core
        | ~end_core
    )
    exact_displacement = jnp.all(
        motion.applied_displacement == delta,
        axis=1,
    )
    available = (
        input_valid
        & ~geometry_exhausted
        & grounded.grounded
        & ~support.leaves_support
        & (support.travel_fraction == jnp.float32(1.0))
        & ~motion.collided
        & exact_displacement
        & ordinary_air
    )
    distance = jnp.linalg.norm(delta[:, (0, 2)], axis=1)
    half = start + delta * jnp.float32(0.5)
    return RegionWalkProbeSubsetResult(
        certified_available=available,
        travelled_distance=jnp.where(available, distance, 0.0),
        reached_half_step=available,
        reached_full_step=available,
        valid_position=available,
        half_step_position=jnp.where(available[:, None], half, start),
        successor_position=jnp.where(available[:, None], end, start),
        unsupported_mechanics=input_valid & ~geometry_exhausted & ~available,
        geometry_exhausted=geometry_exhausted,
        invalid=~input_valid,
    )


def region_graph_astar_result(
    atlas: RegionTraversalAtlas,
    graph_index: Array,
    start_node: Array,
    goal_node: Array,
    *,
    nodes_to_process: int = ASTAR_DEFAULT_NODES_PER_TICK,
    maximum_path_length: int = ASTAR_MAXIMUM_PATH_LENGTH,
    open_nodes_limit: int = ASTAR_OPEN_NODES_LIMIT,
    total_nodes_limit: int = ASTAR_TOTAL_NODES_LIMIT,
) -> RegionGraphAStarResult:
    """Search one selected native-edge graph per batch row.

    Configuration integers are compile-time capacities. Invalid graph/node
    rows and every non-accomplished search return an empty path.
    """

    if not isinstance(atlas, RegionTraversalAtlas):
        raise TypeError("atlas must be a RegionTraversalAtlas")
    process_count = _bounded_capacity(
        nodes_to_process,
        ASTAR_TOTAL_NODES_LIMIT,
        "nodes_to_process",
    )
    path_capacity = _bounded_capacity(
        maximum_path_length,
        ASTAR_MAXIMUM_PATH_LENGTH,
        "maximum_path_length",
    )
    open_limit = _bounded_capacity(
        open_nodes_limit,
        ASTAR_OPEN_NODES_LIMIT,
        "open_nodes_limit",
    )
    total_limit = _bounded_capacity(
        total_nodes_limit,
        ASTAR_TOTAL_NODES_LIMIT,
        "total_nodes_limit",
    )
    # Native checks both limits only after all eight successors of one node.
    # The terminating state can therefore contain up to eight excess entries.
    open_capacity = open_limit + 8
    visited_capacity = total_limit + 8
    graph = jnp.asarray(graph_index, dtype=jnp.int32)
    start = jnp.asarray(start_node, dtype=jnp.int32)
    goal = jnp.asarray(goal_node, dtype=jnp.int32)
    if graph.ndim != 1 or graph.shape[0] == 0:
        raise ValueError("graph_index must have non-empty shape [batch]")
    if start.shape != graph.shape or goal.shape != graph.shape:
        raise ValueError("start_node and goal_node must match graph_index")

    batch = graph.shape[0]
    graph_capacity, node_capacity = atlas.node_mask.shape
    safe_graph = jnp.clip(graph, 0, graph_capacity - 1)
    safe_start = jnp.clip(start, 0, node_capacity - 1)
    safe_goal = jnp.clip(goal, 0, node_capacity - 1)
    graph_valid = (graph >= 0) & (graph < graph_capacity)
    start_valid = (start >= 0) & (start < node_capacity)
    goal_valid = (goal >= 0) & (goal < node_capacity)
    available = (
        graph_valid
        & start_valid
        & goal_valid
        & atlas.graph_mask[safe_graph]
        & atlas.node_mask[safe_graph, safe_start]
        & atlas.node_mask[safe_graph, safe_goal]
    )
    start_position = atlas.node_position[
        safe_graph,
        safe_start,
    ]
    goal_position = atlas.node_position[
        safe_graph,
        safe_goal,
    ]
    finite_endpoints = jnp.all(
        jnp.isfinite(start_position) & jnp.isfinite(goal_position),
        axis=1,
    )
    start_key, start_key_valid = _native_position_key(
        start_position,
        start_position,
    )
    _, goal_key_valid = _native_position_key(
        goal_position,
        start_position,
    )
    available &= finite_endpoints & start_key_valid & goal_key_valid

    visited_key = jnp.full(
        (batch, visited_capacity, 3),
        _EMPTY,
        dtype=jnp.int32,
    )
    visited_node = jnp.full(
        (batch, visited_capacity),
        _EMPTY,
        dtype=jnp.int32,
    )
    visited_cost = jnp.full(
        (batch, visited_capacity),
        jnp.float32(jnp.inf),
    )
    visited_total = jnp.full_like(visited_cost, jnp.float32(jnp.inf))
    visited_parent = jnp.full_like(visited_node, _EMPTY)
    visited_length = jnp.zeros_like(visited_node)
    visited_closed = jnp.zeros(
        (batch, visited_capacity),
        dtype=jnp.bool_,
    )
    start_estimate = _native_float_distance(
        goal_position,
        start_position,
    )
    visited_key = visited_key.at[:, 0].set(
        jnp.where(available[:, None], start_key, _EMPTY)
    )
    visited_node = visited_node.at[:, 0].set(
        jnp.where(available, safe_start, _EMPTY)
    )
    visited_cost = visited_cost.at[:, 0].set(
        jnp.where(available, jnp.float32(0.0), jnp.float32(jnp.inf))
    )
    visited_total = visited_total.at[:, 0].set(
        jnp.where(available, start_estimate, jnp.float32(jnp.inf))
    )
    visited_length = visited_length.at[:, 0].set(
        jnp.where(available, jnp.int32(1), jnp.int32(0))
    )
    open_slot = jnp.full(
        (batch, open_capacity),
        _EMPTY,
        dtype=jnp.int32,
    ).at[:, 0].set(jnp.where(available, jnp.int32(0), _EMPTY))
    open_total = jnp.full(
        (batch, open_capacity),
        jnp.float32(jnp.inf),
    ).at[:, 0].set(
        jnp.where(available, start_estimate, jnp.float32(jnp.inf))
    )
    state = _SearchState(
        progress=jnp.where(
            available,
            jnp.int8(ASTAR_PROGRESS_COMPUTING),
            jnp.int8(ASTAR_PROGRESS_ABORTED),
        ),
        iterations=jnp.zeros(batch, dtype=jnp.int32),
        visited_count=available.astype(jnp.int32),
        visited_key=visited_key,
        visited_node=visited_node,
        visited_cost=visited_cost,
        visited_total_cost=visited_total,
        visited_parent=visited_parent,
        visited_length=visited_length,
        visited_closed=visited_closed,
        open_count=available.astype(jnp.int32),
        open_slot=open_slot,
        open_total_cost=open_total,
        goal_slot=jnp.full(batch, _EMPTY, dtype=jnp.int32),
        open_overflow=jnp.zeros(batch, dtype=jnp.bool_),
        total_overflow=jnp.zeros(batch, dtype=jnp.bool_),
    )

    def expand_once(_: int, current: _SearchState) -> _SearchState:
        return _expand_once(
            current,
            atlas,
            safe_graph,
            safe_goal,
            goal_position,
            start_position,
            maximum_path_length=path_capacity,
            open_nodes_limit=open_limit,
            total_nodes_limit=total_limit,
        )

    state = jax.lax.fori_loop(0, process_count, expand_once, state)
    return _result_from_state(
        state,
        available,
        maximum_path_length=path_capacity,
    )


def region_graph_astar_contract() -> dict[str, object]:
    """Return the pinnable native-search and Region-adapter boundary."""

    return {
        "schema": REGION_GRAPH_ASTAR_SCHEMA,
        "version": REGION_GRAPH_ASTAR_VERSION,
        "function": "region_graph_astar_result",
        "source": {
            "search": "AStarBase.computePath->AStarNode",
            "goal_estimate": "BodyMotionFind.estimateToGoal_euclidean",
            "successor_adapter": "RegionTraversalAtlas.edge_*",
        },
        "bounds": {
            "maximum_path_length": ASTAR_MAXIMUM_PATH_LENGTH,
            "open_nodes": ASTAR_OPEN_NODES_LIMIT,
            "total_nodes": ASTAR_TOTAL_NODES_LIMIT,
            "limit_check": "after_all_eight_successors_of_one_node",
            "storage_headroom_per_limit": 8,
        },
        "configuration": {
            "AStarBase_field_defaults": {
                "open_nodes": ASTAR_BASE_OPEN_NODES_LIMIT,
                "total_nodes": ASTAR_BASE_TOTAL_NODES_LIMIT,
            },
            "BuilderBodyMotionFindBase_installed_defaults": {
                "nodes_per_tick": ASTAR_DEFAULT_NODES_PER_TICK,
                "maximum_path_length": ASTAR_MAXIMUM_PATH_LENGTH,
                "open_nodes": ASTAR_OPEN_NODES_LIMIT,
                "total_nodes": ASTAR_TOTAL_NODES_LIMIT,
                "build_optimised_path": True,
            },
            "installed_asset_overrides": {
                "StepsPerTick": 0,
                "MaxPathLength": 0,
                "MaxOpenNodes": 0,
                "MaxTotalNodes": 0,
            },
            "function_default": (
                "one_native_BodyMotionFind_nodes_per_tick_budget"
            ),
        },
        "grid": {
            "fractional_bits": 1,
            "position_bits": 11,
            "position_offset": 1024,
            "position_mask": 2047,
            "jax_key_layout": "three_int32_components_equivalent_to_native_int64",
        },
        "ordering": {
            "open": "descending_total_cost_pop_end",
            "insert_scan": "strict_existing_total_less_than_new_total",
            "equal_total_cost": "lifo",
            "cost": (
                "captured_controller_projected_travel_plus_"
                "euclidean_goal_estimate"
            ),
        },
        "successors": {
            "authority": (
                "NativeRegionTraversalGraph."
                "native_MotionControllerWalk_probeMove"
            ),
            "edge_cost": (
                "native_probeMove_return_distance_under_captured_"
                "component_selector"
            ),
            "shape": "fixed_padded_per_node",
            "unsupported_or_missing": "absent_and_fail_closed",
        },
        "progress": [
            "UNSTARTED",
            "ABORTED",
            "COMPUTING",
            "ACCOMPLISHED",
            "TERMINATED",
            "TERMINATED_OPEN_NODE_LIMIT_EXCEEDED",
            "TERMINATED_TOTAL_NODE_LIMIT_EXCEEDED",
        ],
        "path": {
            "goal": "exact_graph_node",
            "build": "full_predecessor_chain_not_corner_optimized",
            "non_accomplished": "empty",
        },
        "native_equivalence": {
            "status": (
                "T8_local_predecessor_divergent_"
                "T9_capacity_terminal_divergent"
            ),
            "certified": False,
            "measured_slice": {
                "fixture": "native_navigation_path_recording.json",
                "compared": (
                    "progress_path_length_predecessor_positions_"
                    "and_travel_cost"
                ),
                "result": "at_least_one_deterministic_short_path_diverges",
                "search_counts": (
                    "diagnostic_only_not_required_to_match"
                ),
                "scope": (
                    "short_paths_selected_from_current_fixed_and_"
                    "random_control_Region_graphs"
                ),
            },
            "missing_evidence": [
                (
                    "native_initComputePath_start_candidate_search_"
                    "is_not_represented"
                ),
                (
                    "rejected_probe_positions_are_not_stored_"
                    "in_the_region_graph"
                ),
                (
                    "AStarBase_diagonal_visited_neighbor_guard_"
                    "is_not_reconstructible"
                ),
                (
                    "AStarNode_recursive_successor_cost_propagation_"
                    "is_not_yet_ported"
                ),
                "incremental_search_state_is_not_yet_public",
                "native_default_corner_optimized_path_is_not_yet_published",
            ],
            "known_divergence": {
                "local_predecessor": (
                    "native_and_static_graph_path_length_or_positions_"
                    "do_not_match_for_every_deterministic_short_pair"
                ),
                "capacity_terminal": (
                    "base_80_open_400_total_native_progress_does_not_"
                    "match_the_static_graph_port_for_all_measured_pairs"
                ),
                "cause": (
                    "native_segment_interpolated_successor_positions_"
                    "are_path_dependent_and_not_representable_by_one_"
                    "static_position_per_half_grid_key"
                ),
                "consumer": "must_remain_fail_closed",
            },
            "claim": (
                "diagnostic_static_graph_search_not_native_AStarBase"
            ),
        },
        "fail_closed": [
            "invalid_graph_or_node",
            "non_finite_or_out_of_grid_endpoint",
            "non_positive_or_non_finite_edge_cost",
            "every_non_accomplished_progress_has_empty_path",
        ],
        "consumer": "combat_lane_owns_guard_and_action_binding",
        "provenance": "source_port_plus_native_edge_artifact",
    }


def region_graph_astar_contract_sha256() -> str:
    payload = json.dumps(
        region_graph_astar_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def region_walk_probe_subset_contract() -> dict[str, object]:
    """Return the exact-geometry boundary for the certified probe subset."""

    return {
        "schema": REGION_WALK_PROBE_SUBSET_SCHEMA,
        "version": REGION_WALK_PROBE_SUBSET_VERSION,
        "function": "region_walk_probe_subset_result",
        "source": (
            "MotionControllerWalk.probeMove_plus_"
            "AStarBase_full_step_successor_threshold"
        ),
        "controller": {
            "kind": "MotionControllerWalk",
            "component_selector": [1, 0, 1],
            "directions": "eight_horizontal_integer_neighbours",
        },
        "certified_subset": [
            "valid_finite_actor_bounds_and_input",
            "start_grounded",
            "complete_exact_Region_core_coverage",
            "ordinary_air_through_complete_swept_actor_volume",
            "zero_block_or_fluid_damage_in_swept_volume",
            "no_collision_or_slide",
            "connected_support_for_complete_step",
            "exact_full_displacement",
        ],
        "outputs": (
            "native_successor_full_step_fields_for_certified_rows_only"
        ),
        "unavailable_fail_closed": [
            "wall_or_partial_contact",
            "support_edge_or_drop",
            "climb",
            "fluid",
            "damage",
            "nonordinary_material",
            "incomplete_or_over_capacity_geometry",
            "unsupported_direction_or_invalid_input",
        ],
        "does_not_certify": [
            "general_probeMove",
            "AStarBase_start_search",
            "half_step_segment_interpolation_after_contact",
            "diagonal_visited_neighbour_guard",
            "recursive_successor_cost_repair",
            "PathFollower_or_replan_policy",
        ],
        "consumer": (
            "may_feed_a_future_exact_successor_port_but_does_not_"
            "independently_open_certified_navigation_step_available"
        ),
        "provenance": "native_source_port_plus_exact_Region_geometry",
    }


def region_walk_probe_subset_contract_sha256() -> str:
    payload = json.dumps(
        region_walk_probe_subset_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _region_probe_swept_ordinary_air(
    geometry: RegionGeometryState,
    start: Array,
    end: Array,
    bounds: Array,
) -> tuple[Array, Array]:
    minimum = jnp.floor(
        jnp.minimum(
            start + bounds[:, :3],
            end + bounds[:, :3],
        )
    ).astype(jnp.int32)
    maximum = jnp.floor(
        jnp.maximum(
            start + bounds[:, 3:],
            end + bounds[:, 3:],
        )
        - _CELL_EPSILON
    ).astype(jnp.int32)
    extent = maximum - minimum + 1
    within_capacity = jnp.all(
        (extent > 0) & (extent <= jnp.int32(3)),
        axis=1,
    )
    cells = minimum[:, None, :] + _PROBE_CELL_OFFSETS[None, :, :]
    mask = jnp.all(cells <= maximum[:, None, :], axis=2)
    selected = lookup_region_blocks(
        geometry.atlas,
        cells,
        geometry.environment_world_id,
        require_core=True,
    )
    complete = within_capacity & ~jnp.any(
        mask & ~selected.available,
        axis=1,
    )
    ordinary = (
        (selected.flags == 0)
        & (selected.block_damage <= 0.0)
        & (selected.fluid_damage <= 0.0)
        & ((selected.flags & jnp.int32(FLAG_FLUID)) == 0)
    )
    safe = complete & jnp.all(~mask | ordinary, axis=1)
    return safe, ~complete


def _bounded_capacity(value: int, maximum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be int")
    if value < 1 or value > maximum:
        raise ValueError(f"{label} must be in [1, {maximum}]")
    return value


__all__ = [
    "ASTAR_PROGRESS_ABORTED",
    "ASTAR_PROGRESS_ACCOMPLISHED",
    "ASTAR_PROGRESS_COMPUTING",
    "ASTAR_PROGRESS_TERMINATED",
    "ASTAR_PROGRESS_TERMINATED_OPEN_NODE_LIMIT_EXCEEDED",
    "ASTAR_PROGRESS_TERMINATED_TOTAL_NODE_LIMIT_EXCEEDED",
    "ASTAR_PROGRESS_UNSTARTED",
    "PATH_FOLLOWER_SCHEMA",
    "PATH_FOLLOWER_VERSION",
    "REGION_GRAPH_ASTAR_SCHEMA",
    "REGION_GRAPH_ASTAR_VERSION",
    "REGION_WALK_PROBE_SUBSET_SCHEMA",
    "REGION_WALK_PROBE_SUBSET_VERSION",
    "PathFollowerSteeringResult",
    "PathFollowerWaypointResult",
    "RegionGraphAStarResult",
    "RegionWalkProbeSubsetResult",
    "path_follower_contract",
    "path_follower_contract_sha256",
    "path_follower_steering_result",
    "path_follower_waypoint_result",
    "region_graph_astar_contract",
    "region_graph_astar_contract_sha256",
    "region_graph_astar_result",
    "region_walk_probe_subset_contract",
    "region_walk_probe_subset_contract_sha256",
    "region_walk_probe_subset_result",
]
