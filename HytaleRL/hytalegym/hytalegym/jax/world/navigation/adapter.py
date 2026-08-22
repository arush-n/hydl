"""Adapter from certified Region Walk edges to Combat navigation steps."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from hytalegym.jax.world.navigation import (
    RegionWalkProbeSubsetResult,
    region_walk_probe_subset_result,
)
from hytalegym.jax.world.region.types import (
    RegionGeometryState,
    RegionTraversalAtlas,
)
from hytalegym.worldgen.region import (
    CHUNK_SIZE,
    CORE_BLOCKS_PER_AXIS,
    CORE_CHUNKS_PER_AXIS,
    REGION_TRAVERSAL_EDGE_SAFE,
    REGION_TRAVERSAL_EDGE_WALK,
)

if TYPE_CHECKING:
    from hytalegym.jax.combat.motion.navigation import TargetNavigationStep
    from hytalegym.jax.combat.types import CombatParams, CombatState


Array = jax.Array

REGION_WALK_NAVIGATION_ADAPTER_SCHEMA = (
    "hytalerl_region_walk_target_navigation_adapter_v1"
)
REGION_WALK_NAVIGATION_ADAPTER_VERSION = 1
REGION_WALK_NAVIGATION_EDGE_EPSILON = 1.0e-5
SURROGATE_REGION_WALK_NAVIGATION_SCHEMA = (
    "hytalerl_surrogate_region_walk_target_navigation_provider_v1"
)
SURROGATE_REGION_WALK_NAVIGATION_VERSION = 1
NATIVE_REGION_GRAPH_NAVIGATION_SCHEMA = (
    "hytalerl_native_region_graph_target_navigation_provider_v1"
)
NATIVE_REGION_GRAPH_NAVIGATION_VERSION = 1
NATIVE_REGION_GRAPH_CANDIDATE_NODES = 8
_GREEDY_HORIZONTAL_DIRECTIONS = (
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 1.0),
    (1.0, 0.0, -1.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
    (-1.0, 0.0, 1.0),
    (-1.0, 0.0, -1.0),
    (-1.0, 0.0, 0.0),
)


def region_walk_target_navigation_step(
    probe: RegionWalkProbeSubsetResult,
    current_position: Array,
    requested_velocity: Array,
    motion_delta_seconds: Array,
    *,
    edge_epsilon: float = REGION_WALK_NAVIGATION_EDGE_EPSILON,
) -> TargetNavigationStep:
    """Adapt one incremental motion along a certified full horizontal edge.

    The full-edge probe supplies topology and swept-geometry evidence. This
    function admits only a finite, forward, on-segment substep. It does not
    choose a path and does not relabel captured imitation as certification.
    """

    # Imported lazily to keep ``jax.world`` importable by ``jax.combat``.
    from hytalegym.jax.combat.motion.navigation import TargetNavigationStep

    if not isinstance(probe, RegionWalkProbeSubsetResult):
        raise TypeError("probe must be RegionWalkProbeSubsetResult")
    available = jnp.asarray(probe.certified_available, dtype=jnp.bool_)
    if available.ndim != 1 or available.shape[0] == 0:
        raise ValueError("probe rows must have non-empty shape [batch]")
    batch = available.shape[0]
    current = _batch_vector(current_position, batch, "current_position")
    velocity = _batch_vector(requested_velocity, batch, "requested_velocity")
    delta = _batch_float(
        motion_delta_seconds,
        batch,
        "motion_delta_seconds",
    )
    half = _probe_vector(probe.half_step_position, batch, "half_step_position")
    successor = _probe_vector(
        probe.successor_position,
        batch,
        "successor_position",
    )
    for name in (
        "reached_half_step",
        "reached_full_step",
        "valid_position",
        "unsupported_mechanics",
        "geometry_exhausted",
        "invalid",
    ):
        if jnp.asarray(getattr(probe, name)).shape != (batch,):
            raise ValueError(f"probe.{name} must have shape [batch]")
    distance = jnp.asarray(probe.travelled_distance, dtype=jnp.float32)
    if distance.shape != (batch,):
        raise ValueError("probe.travelled_distance must have shape [batch]")
    if (
        isinstance(edge_epsilon, bool)
        or not isinstance(edge_epsilon, (int, float))
        or not 0.0 < float(edge_epsilon) <= 1.0e-3
    ):
        raise ValueError("edge_epsilon must be in (0, 1e-3]")

    edge_start = jnp.float32(2.0) * half - successor
    edge = successor - edge_start
    edge_squared = jnp.sum(edge * edge, axis=1)
    safe_edge_squared = jnp.where(edge_squared > 0.0, edge_squared, 1.0)
    proposed = current + velocity * delta[:, None]
    current_progress = jnp.sum(
        (current - edge_start) * edge,
        axis=1,
    ) / safe_edge_squared
    proposed_progress = jnp.sum(
        (proposed - edge_start) * edge,
        axis=1,
    ) / safe_edge_squared
    current_projection = edge_start + current_progress[:, None] * edge
    proposed_projection = edge_start + proposed_progress[:, None] * edge
    epsilon = jnp.float32(edge_epsilon)
    finite = jnp.all(
        jnp.isfinite(current)
        & jnp.isfinite(velocity)
        & jnp.isfinite(half)
        & jnp.isfinite(successor),
        axis=1,
    ) & jnp.isfinite(delta)
    on_edge = (
        jnp.linalg.norm(current - current_projection, axis=1) <= epsilon
    ) & (
        jnp.linalg.norm(proposed - proposed_projection, axis=1) <= epsilon
    )
    forward_substep = (
        (delta > 0.0)
        & (current_progress >= -epsilon)
        & (current_progress < jnp.float32(1.0) + epsilon)
        & (proposed_progress > current_progress)
        & (proposed_progress <= jnp.float32(1.0) + epsilon)
    )
    certified = (
        available
        & jnp.asarray(probe.reached_half_step, dtype=jnp.bool_)
        & jnp.asarray(probe.reached_full_step, dtype=jnp.bool_)
        & jnp.asarray(probe.valid_position, dtype=jnp.bool_)
        & ~jnp.asarray(probe.unsupported_mechanics, dtype=jnp.bool_)
        & ~jnp.asarray(probe.geometry_exhausted, dtype=jnp.bool_)
        & ~jnp.asarray(probe.invalid, dtype=jnp.bool_)
        & finite
        & (edge_squared > 0.0)
        & (distance > 0.0)
        & on_edge
        & forward_substep
    )
    return TargetNavigationStep(
        certified_available=certified,
        position=jnp.where(certified[:, None], proposed, 0.0),
        velocity=jnp.where(certified[:, None], velocity, 0.0),
        geometry_exhausted=jnp.asarray(
            probe.geometry_exhausted,
            dtype=jnp.bool_,
        ),
    )


def surrogate_region_walk_navigation_step(
    geometry: RegionGeometryState,
    current_position: Array,
    requested_displacement: Array,
    motion_delta_seconds: Array,
) -> TargetNavigationStep:
    """Choose a greedy exact local edge without claiming native route choice."""

    if not isinstance(geometry, RegionGeometryState):
        raise TypeError("geometry must be RegionGeometryState")
    current = jnp.asarray(current_position, dtype=jnp.float32)
    if current.ndim != 2 or current.shape[0] == 0 or current.shape[1] != 3:
        raise ValueError("current_position must have non-empty shape [batch,3]")
    batch = current.shape[0]
    displacement = _batch_vector(
        requested_displacement,
        batch,
        "requested_displacement",
    )
    delta = _batch_float(
        motion_delta_seconds,
        batch,
        "motion_delta_seconds",
    )
    directions = jnp.asarray(
        _GREEDY_HORIZONTAL_DIRECTIONS,
        dtype=jnp.float32,
    )
    probes = tuple(
        region_walk_probe_subset_result(
            geometry,
            current,
            jnp.broadcast_to(direction, current.shape),
        )
        for direction in directions
    )
    stacked = jax.tree.map(
        lambda *values: jnp.stack(values, axis=1),
        *probes,
    )

    horizontal = displacement[:, (0, 2)]
    requested_length = jnp.linalg.norm(horizontal, axis=1)
    safe_length = jnp.where(requested_length > 0.0, requested_length, 1.0)
    requested_direction = horizontal / safe_length[:, None]
    candidate_horizontal = directions[:, (0, 2)]
    candidate_direction = candidate_horizontal / jnp.linalg.norm(
        candidate_horizontal,
        axis=1,
    )[:, None]
    alignment = jnp.sum(
        requested_direction[:, None, :] * candidate_direction[None, :, :],
        axis=2,
    )
    valid = (
        jnp.all(jnp.isfinite(current) & jnp.isfinite(displacement), axis=1)
        & jnp.isfinite(delta)
        & (delta > 0.0)
        & (requested_length > 0.0)
        & (jnp.abs(displacement[:, 1]) <= REGION_WALK_NAVIGATION_EDGE_EPSILON)
    )
    candidate = (
        stacked.certified_available
        & valid[:, None]
        & (alignment >= 0.0)
    )
    selected_slot = jnp.argmax(
        jnp.where(candidate, alignment, -jnp.inf),
        axis=1,
    )
    selected = jax.tree.map(
        lambda value: value[jnp.arange(batch), selected_slot],
        stacked,
    )
    selected_available = valid & jnp.any(candidate, axis=1)
    selected = selected._replace(
        certified_available=(
            selected.certified_available & selected_available
        ),
        invalid=selected.invalid | ~valid,
    )
    selected_direction = directions[selected_slot]
    selected_unit = selected_direction / jnp.linalg.norm(
        selected_direction,
        axis=1,
    )[:, None]
    speed = requested_length / jnp.where(delta > 0.0, delta, 1.0)
    velocity = selected_unit * speed[:, None]
    return region_walk_target_navigation_step(
        selected,
        current,
        velocity,
        delta,
    )


def surrogate_region_walk_target_navigation_provider(
    state: CombatState,
    _params: CombatParams,
    geometry: object,
    requested_displacement: Array,
    motion_delta_seconds: Array,
) -> TargetNavigationStep:
    """Expose the greedy exact-edge selector as Combat's provider protocol."""

    from hytalegym.jax.combat.motion.navigation import TargetNavigationStep
    from hytalegym.jax.combat.types import TARGET_ENTITY

    position = jnp.asarray(state.position, dtype=jnp.float32)
    if position.ndim != 3 or position.shape[1] <= TARGET_ENTITY:
        raise ValueError("state.position must have shape [batch,entities,3]")
    if isinstance(geometry, RegionGeometryState):
        target_geometry = geometry._replace(
            agent_bounds=geometry.target_bounds,
        )
        return surrogate_region_walk_navigation_step(
            target_geometry,
            position[:, TARGET_ENTITY],
            requested_displacement,
            motion_delta_seconds,
        )
    batch = position.shape[0]
    return TargetNavigationStep(
        certified_available=jnp.zeros((batch,), dtype=jnp.bool_),
        position=jnp.zeros((batch, 3), dtype=jnp.float32),
        velocity=jnp.zeros((batch, 3), dtype=jnp.float32),
        geometry_exhausted=jnp.zeros((batch,), dtype=jnp.bool_),
    )


def native_region_graph_navigation_step(
    traversal: RegionTraversalAtlas,
    environment_world_id: Array,
    geometry: RegionGeometryState,
    current_position: Array,
    requested_displacement: Array,
    motion_delta_seconds: Array,
    *,
    candidate_node_capacity: int = NATIVE_REGION_GRAPH_CANDIDATE_NODES,
) -> TargetNavigationStep:
    """Follow a nearby native graph walk edge and certify its exact substep.

    The graph constrains route surface and edge direction. Selection remains a
    bounded local alignment heuristic rather than a claim of native A* route
    choice. Only dry horizontal walk edges are opened because the exact Region
    substep adapter does not yet certify climb/drop trajectories.
    """

    from hytalegym.jax.combat.motion.navigation import TargetNavigationStep

    if not isinstance(traversal, RegionTraversalAtlas):
        raise TypeError("traversal must be a RegionTraversalAtlas")
    if not isinstance(geometry, RegionGeometryState):
        raise TypeError("geometry must be a RegionGeometryState")
    current = jnp.asarray(current_position, dtype=jnp.float32)
    if current.ndim != 2 or current.shape[0] == 0 or current.shape[1] != 3:
        raise ValueError("current_position must have non-empty shape [batch,3]")
    batch = current.shape[0]
    requested = _batch_vector(
        requested_displacement,
        batch,
        "requested_displacement",
    )
    delta = _batch_float(
        motion_delta_seconds,
        batch,
        "motion_delta_seconds",
    )
    world_ids = jnp.asarray(environment_world_id, dtype=jnp.int32)
    if world_ids.shape != (batch,):
        raise ValueError("environment_world_id must have shape [batch]")
    candidates = _bounded_candidate_capacity(
        candidate_node_capacity,
        traversal.node_mask.shape[1],
    )

    graph_index, graph_available = _native_graph_for_position(
        traversal,
        world_ids,
        current,
    )
    safe_graph = jnp.clip(
        graph_index,
        0,
        traversal.graph_mask.shape[0] - 1,
    )
    node_position = traversal.node_position[safe_graph]
    node_mask = traversal.node_mask[safe_graph] & graph_available[:, None]
    node_distance_squared = jnp.sum(
        (node_position - current[:, None, :]) ** 2,
        axis=2,
    )
    _, nearest_node = jax.lax.top_k(
        -jnp.where(node_mask, node_distance_squared, jnp.inf),
        candidates,
    )
    batch_index = jnp.arange(batch, dtype=jnp.int32)
    source_position = node_position[
        batch_index[:, None],
        nearest_node,
    ]
    source_mask = node_mask[batch_index[:, None], nearest_node]
    destination = traversal.edge_destination[
        safe_graph[:, None],
        nearest_node,
    ]
    edge_mask = traversal.edge_mask[
        safe_graph[:, None],
        nearest_node,
    ]
    edge_kind = traversal.edge_kind[
        safe_graph[:, None],
        nearest_node,
    ]
    edge_flags = traversal.edge_flags[
        safe_graph[:, None],
        nearest_node,
    ]
    node_capacity = traversal.node_mask.shape[1]
    safe_destination = jnp.clip(destination, 0, node_capacity - 1)
    destination_position = traversal.node_position[
        safe_graph[:, None, None],
        safe_destination,
    ]
    destination_mask = traversal.node_mask[
        safe_graph[:, None, None],
        safe_destination,
    ]
    segment = destination_position - source_position[:, :, None, :]
    segment_squared = jnp.sum(segment * segment, axis=3)
    safe_segment_squared = jnp.where(segment_squared > 0.0, segment_squared, 1.0)
    offset = current[:, None, None, :] - source_position[:, :, None, :]
    progress = jnp.sum(offset * segment, axis=3) / safe_segment_squared
    projection = source_position[:, :, None, :] + progress[..., None] * segment
    epsilon = jnp.float32(REGION_WALK_NAVIGATION_EDGE_EPSILON)
    on_segment = jnp.linalg.norm(
        current[:, None, None, :] - projection,
        axis=3,
    ) <= epsilon

    horizontal = requested[:, (0, 2)]
    requested_length = jnp.linalg.norm(horizontal, axis=1)
    safe_requested_length = jnp.where(
        requested_length > 0.0,
        requested_length,
        1.0,
    )
    requested_direction = horizontal / safe_requested_length[:, None]
    segment_horizontal = segment[..., (0, 2)]
    segment_horizontal_length = jnp.linalg.norm(segment_horizontal, axis=3)
    safe_segment_horizontal_length = jnp.where(
        segment_horizontal_length > 0.0,
        segment_horizontal_length,
        1.0,
    )
    segment_direction = (
        segment_horizontal / safe_segment_horizontal_length[..., None]
    )
    alignment = jnp.sum(
        segment_direction * requested_direction[:, None, None, :],
        axis=3,
    )

    target_bounds = jnp.asarray(geometry.target_bounds, dtype=jnp.float32)
    if target_bounds.ndim != 2 or target_bounds.shape[1] != 6:
        raise ValueError("geometry target bounds must have shape [batch|1,6]")
    if target_bounds.shape[0] not in (1, batch):
        raise ValueError("geometry target bounds batch must be one or match")
    target_bounds = jnp.broadcast_to(target_bounds, (batch, 6))
    graph_bounds = traversal.actor_bounds[safe_graph]
    profile_match = jnp.all(
        jnp.abs(graph_bounds - target_bounds) <= jnp.float32(1.0e-6),
        axis=1,
    )
    finite = (
        jnp.all(jnp.isfinite(current) & jnp.isfinite(requested), axis=1)
        & jnp.isfinite(delta)
        & (delta > 0.0)
        & (requested_length > 0.0)
        & (jnp.abs(requested[:, 1]) <= epsilon)
    )
    discrete_walk = (
        (jnp.abs(segment[..., 1]) <= epsilon)
        & jnp.all(jnp.abs(segment) <= jnp.float32(1.0) + epsilon, axis=3)
        & jnp.all(
            jnp.abs(segment - jnp.round(segment)) <= epsilon,
            axis=3,
        )
        & (segment_horizontal_length > 0.0)
    )
    candidate = (
        graph_available[:, None, None]
        & profile_match[:, None, None]
        & source_mask[:, :, None]
        & edge_mask
        & destination_mask
        & (destination >= 0)
        & (destination < node_capacity)
        & (edge_kind == REGION_TRAVERSAL_EDGE_WALK)
        & ((edge_flags & REGION_TRAVERSAL_EDGE_SAFE) != 0)
        & discrete_walk
        & on_segment
        & (progress >= -epsilon)
        & (progress < jnp.float32(1.0) - epsilon)
        & (alignment >= 0.0)
        & finite[:, None, None]
    )
    flat_candidate = candidate.reshape(batch, -1)
    flat_alignment = alignment.reshape(batch, -1)
    selected_slot = jnp.argmax(
        jnp.where(flat_candidate, flat_alignment, -jnp.inf),
        axis=1,
    )
    selected_available = jnp.any(flat_candidate, axis=1)
    flat_segment = segment.reshape(batch, -1, 3)
    flat_source = jnp.broadcast_to(
        source_position[:, :, None, :],
        segment.shape,
    ).reshape(batch, -1, 3)
    flat_progress = progress.reshape(batch, -1)
    selected_segment = flat_segment[batch_index, selected_slot]
    selected_source = flat_source[batch_index, selected_slot]
    selected_progress = flat_progress[batch_index, selected_slot]
    selected_length = jnp.linalg.norm(selected_segment, axis=1)
    selected_unit = selected_segment / jnp.where(
        selected_length > 0.0,
        selected_length,
        1.0,
    )[:, None]
    remaining = jnp.maximum(
        (jnp.float32(1.0) - selected_progress) * selected_length,
        0.0,
    )
    step_distance = jnp.minimum(requested_length, remaining)
    velocity = selected_unit * (
        step_distance / jnp.where(delta > 0.0, delta, 1.0)
    )[:, None]

    target_geometry = geometry._replace(agent_bounds=target_bounds)
    probe = region_walk_probe_subset_result(
        target_geometry,
        selected_source,
        selected_segment,
    )
    step = region_walk_target_navigation_step(
        probe,
        current,
        velocity,
        delta,
    )
    certified = selected_available & step.certified_available
    return TargetNavigationStep(
        certified_available=certified,
        position=jnp.where(certified[:, None], step.position, 0.0),
        velocity=jnp.where(certified[:, None], step.velocity, 0.0),
        geometry_exhausted=(
            selected_available & step.geometry_exhausted
        ),
    )


def native_region_graph_target_navigation_provider(
    state: CombatState,
    _params: CombatParams,
    geometry: object,
    requested_displacement: Array,
    motion_delta_seconds: Array,
    *,
    traversal: RegionTraversalAtlas,
    environment_world_id: Array,
) -> TargetNavigationStep:
    """Expose native-graph-guided exact walk substeps to Combat."""

    from hytalegym.jax.combat.motion.navigation import TargetNavigationStep
    from hytalegym.jax.combat.types import TARGET_ENTITY

    position = jnp.asarray(state.position, dtype=jnp.float32)
    if position.ndim != 3 or position.shape[1] <= TARGET_ENTITY:
        raise ValueError("state.position must have shape [batch,entities,3]")
    if isinstance(geometry, RegionGeometryState):
        return native_region_graph_navigation_step(
            traversal,
            environment_world_id,
            geometry,
            position[:, TARGET_ENTITY],
            requested_displacement,
            motion_delta_seconds,
        )
    batch = position.shape[0]
    return TargetNavigationStep(
        certified_available=jnp.zeros((batch,), dtype=jnp.bool_),
        position=jnp.zeros((batch, 3), dtype=jnp.float32),
        velocity=jnp.zeros((batch, 3), dtype=jnp.float32),
        geometry_exhausted=jnp.zeros((batch,), dtype=jnp.bool_),
    )


def native_region_graph_navigation_contract() -> dict[str, object]:
    """Publish the native-edge/exact-substep/local-route boundary."""

    return {
        "schema": NATIVE_REGION_GRAPH_NAVIGATION_SCHEMA,
        "version": NATIVE_REGION_GRAPH_NAVIGATION_VERSION,
        "provider_protocol": "jax.combat.env.TargetNavigationProvider",
        "candidate_node_capacity": NATIVE_REGION_GRAPH_CANDIDATE_NODES,
        "selection": (
            "highest_nonnegative_requested_alignment_among_nearby_"
            "native_safe_walk_edges_containing_current_position"
        ),
        "provenance": {
            "route_surface": "native_MotionControllerWalk_accepted_graph_edge",
            "selected_substep": "exact_region_geometry_certified",
            "route_choice": "bounded_local_greedy_not_native_AStar",
            "actor_profile": "exact_match_to_captured_graph_bounds",
        },
        "not_claimed": [
            "native_AStar_route_choice",
            "persistent_path_memory_or_replanning",
            "climb_drop_or_fluid_edges",
            "PathFollower_smoothing_or_steering",
            "global_route_completion",
        ],
        "failure": (
            "missing_graph_profile_mismatch_off_graph_or_unavailable_exact_"
            "walk_substep_returns_false_certified_bit_and_zero_motion"
        ),
    }


def native_region_graph_navigation_contract_sha256() -> str:
    return hashlib.sha256(
        json.dumps(
            native_region_graph_navigation_contract(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def surrogate_region_walk_navigation_contract() -> dict[str, object]:
    """Publish the exact-transition/surrogate-route provenance split."""

    return {
        "schema": SURROGATE_REGION_WALK_NAVIGATION_SCHEMA,
        "version": SURROGATE_REGION_WALK_NAVIGATION_VERSION,
        "provider_protocol": "jax.combat.env.TargetNavigationProvider",
        "direction_capacity": len(_GREEDY_HORIZONTAL_DIRECTIONS),
        "selection": (
            "highest_nonnegative_alignment_among_exact_horizontal_edges"
        ),
        "cross_backend_velocity_tolerance_blocks_per_second": 3.0e-7,
        "provenance": {
            "selected_transition": "native_differential_exact_local_edge",
            "route_choice": "surrogate_greedy_not_native_certified",
            "certified_available_scope": "selected_transition_only",
        },
        "not_claimed": [
            "native_AStar_route_choice",
            "goal_memory",
            "replan_or_path_smoothing",
            "climb_drop_fluid_or_partial_contact_edges",
            "global_route_completion",
        ],
        "failure": (
            "non_region_invalid_or_no_nonnegative_exact_edge_returns_"
            "false_certified_bit_and_zero_motion"
        ),
    }


def surrogate_region_walk_navigation_contract_sha256() -> str:
    return hashlib.sha256(
        json.dumps(
            surrogate_region_walk_navigation_contract(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def region_walk_navigation_adapter_contract() -> dict[str, object]:
    """Return the certification and imitation-labelling decision."""

    return {
        "schema": REGION_WALK_NAVIGATION_ADAPTER_SCHEMA,
        "version": REGION_WALK_NAVIGATION_ADAPTER_VERSION,
        "input": "RegionWalkProbeSubsetResult",
        "output": "jax.combat.navigation.TargetNavigationStep",
        "dependency": "lazy_import_avoids_world_combat_import_cycle",
        "admission": [
            "certified_full_dry_horizontal_edge",
            "finite_positive_delta",
            "current_and_proposed_positions_on_same_edge",
            "strictly_forward_substep_not_past_successor",
            "no_unsupported_mechanics_or_geometry_exhaustion",
        ],
        "edge_epsilon": REGION_WALK_NAVIGATION_EDGE_EPSILON,
        "provenance": {
            "certified_available": "native_differential_exact_local_edge",
            "path_selection": "caller_owned",
            "native_path_imitation": (
                "separate_training_provenance_never_sets_certified_available"
            ),
        },
        "not_claimed": [
            "native_AStar_route_choice",
            "PathFollower_replan_or_smoothing",
            "climb_drop_fluid_or_partial_contact_edges",
        ],
        "failure": "zero_position_and_velocity_with_false_certified_bit",
    }


def region_walk_navigation_adapter_contract_sha256() -> str:
    return hashlib.sha256(
        json.dumps(
            region_walk_navigation_adapter_contract(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _batch_vector(value: Array, batch: int, label: str) -> Array:
    result = jnp.asarray(value, dtype=jnp.float32)
    if result.shape != (batch, 3):
        raise ValueError(f"{label} must have shape [batch,3]")
    return result


def _probe_vector(value: Array, batch: int, label: str) -> Array:
    result = jnp.asarray(value, dtype=jnp.float32)
    if result.shape != (batch, 3):
        raise ValueError(f"probe.{label} must have shape [batch,3]")
    return result


def _batch_float(value: Array, batch: int, label: str) -> Array:
    result = jnp.asarray(value, dtype=jnp.float32)
    if result.ndim == 0:
        result = jnp.broadcast_to(result, (batch,))
    if result.shape != (batch,):
        raise ValueError(f"{label} must be scalar or shape [batch]")
    return result


def _native_graph_for_position(
    traversal: RegionTraversalAtlas,
    environment_world_id: Array,
    position: Array,
) -> tuple[Array, Array]:
    finite = jnp.all(jnp.isfinite(position), axis=1)
    precise = jnp.all(jnp.abs(position) < jnp.float32(1 << 24), axis=1)
    valid = finite & precise
    safe_position = jnp.where(valid[:, None], position, 0.0)
    blocks = jnp.floor(safe_position).astype(jnp.int32)
    chunks = jnp.floor_divide(blocks[:, (0, 2)], CHUNK_SIZE)
    core_relative = chunks[:, None, :] - traversal.core_min_chunk_xz[None, :, :]
    inside = jnp.all(
        (core_relative >= 0) & (core_relative < CORE_CHUNKS_PER_AXIS),
        axis=2,
    )
    eligible = (
        traversal.graph_mask[None, :]
        & (traversal.world_id[None, :] == environment_world_id[:, None])
        & inside
        & valid[:, None]
    )
    local_xz = blocks[:, None, (0, 2)] - (
        traversal.core_min_chunk_xz[None, :, :] * jnp.int32(CHUNK_SIZE)
    )
    center_delta_twice = (
        local_xz * jnp.int32(2) + jnp.int32(1) - CORE_BLOCKS_PER_AXIS
    )
    score = jnp.where(
        eligible,
        jnp.sum(center_delta_twice**2, axis=2),
        jnp.iinfo(jnp.int32).max,
    )
    return (
        jnp.argmin(score, axis=1).astype(jnp.int32),
        jnp.any(eligible, axis=1),
    )


def _bounded_candidate_capacity(value: int, node_capacity: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("candidate_node_capacity must be a positive integer")
    if value > NATIVE_REGION_GRAPH_CANDIDATE_NODES:
        raise ValueError(
            "candidate_node_capacity exceeds the published native graph limit"
        )
    if node_capacity < 1:
        raise ValueError("native graph node capacity must be positive")
    return min(value, node_capacity)


__all__ = [
    "NATIVE_REGION_GRAPH_CANDIDATE_NODES",
    "NATIVE_REGION_GRAPH_NAVIGATION_SCHEMA",
    "NATIVE_REGION_GRAPH_NAVIGATION_VERSION",
    "REGION_WALK_NAVIGATION_ADAPTER_SCHEMA",
    "REGION_WALK_NAVIGATION_ADAPTER_VERSION",
    "REGION_WALK_NAVIGATION_EDGE_EPSILON",
    "SURROGATE_REGION_WALK_NAVIGATION_SCHEMA",
    "SURROGATE_REGION_WALK_NAVIGATION_VERSION",
    "region_walk_navigation_adapter_contract",
    "region_walk_navigation_adapter_contract_sha256",
    "region_walk_target_navigation_step",
    "native_region_graph_navigation_contract",
    "native_region_graph_navigation_contract_sha256",
    "native_region_graph_navigation_step",
    "native_region_graph_target_navigation_provider",
    "surrogate_region_walk_navigation_contract",
    "surrogate_region_walk_navigation_contract_sha256",
    "surrogate_region_walk_navigation_step",
    "surrogate_region_walk_target_navigation_provider",
]
