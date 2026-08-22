"""Path-following helpers internal to the navigation package."""

import hashlib
import json
from typing import NamedTuple
import jax.numpy as jnp
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


PATH_FOLLOWER_SCHEMA = "hytalerl_path_follower_v1"


PATH_FOLLOWER_VERSION = 1


ASTAR_MAXIMUM_PATH_LENGTH = 200


PATH_FOLLOWER_DEFAULT_REJECTION_WEIGHT = 3.0


PATH_FOLLOWER_EXACT_DISTANCE_SQUARED = 1.0000000000000002e-10


PATH_FOLLOWER_NEAR_DISTANCE_SQUARED = 0.01


PATH_FOLLOWER_DIRECTION_REFRESH_DISTANCE = 0.1


class PathFollowerWaypointResult(NamedTuple):
    """One ``PathFollower.updateCurrentTarget`` transition."""

    available: Array
    current_waypoint_index: Array
    last_waypoint_position: Array
    current_waypoint_distance_squared: Array
    current_target_position: Array
    active: Array
    reached_waypoint: Array
    path_finished: Array
    should_smooth_path: Array
    invalid: Array


class PathFollowerSteeringResult(NamedTuple):
    """One ``PathFollower.executePath`` translation update."""

    available: Array
    translation: Array
    rejection: Array
    max_distance: Array
    uses_waypoint_speed: Array
    invalid: Array


def path_follower_waypoint_result(
    path_position: Array,
    path_mask: Array,
    current_waypoint_index: Array,
    last_waypoint_position: Array,
    current_waypoint_distance_squared: Array,
    entity_position: Array,
    component_selector: Array,
) -> PathFollowerWaypointResult:
    """Advance at most one waypoint using native projected-dot semantics."""

    path = jnp.asarray(path_position, dtype=jnp.float32)
    mask = jnp.asarray(path_mask, dtype=jnp.bool_)
    current = jnp.asarray(current_waypoint_index, dtype=jnp.int32)
    last = jnp.asarray(last_waypoint_position, dtype=jnp.float32)
    previous_distance = jnp.asarray(
        current_waypoint_distance_squared,
        dtype=jnp.float32,
    )
    entity = jnp.asarray(entity_position, dtype=jnp.float32)
    selector = jnp.asarray(component_selector, dtype=jnp.float32)
    if path.ndim != 3 or path.shape[0] == 0 or path.shape[2] != 3:
        raise ValueError("path_position must have non-empty shape [batch,path,3]")
    batch, capacity, _ = path.shape
    if capacity > ASTAR_MAXIMUM_PATH_LENGTH:
        raise ValueError("path capacity exceeds the native maximum")
    if mask.shape != (batch, capacity):
        raise ValueError("path_mask must have shape [batch,path]")
    if current.shape != (batch,):
        raise ValueError("current_waypoint_index must have shape [batch]")
    for name, value in (
        ("last_waypoint_position", last),
        ("entity_position", entity),
        ("component_selector", selector),
    ):
        if value.shape != (batch, 3):
            raise ValueError(f"{name} must have shape [batch,3]")
    if previous_distance.shape != (batch,):
        raise ValueError(
            "current_waypoint_distance_squared must have shape [batch]"
        )

    count = jnp.sum(mask, axis=1, dtype=jnp.int32)
    contiguous = jnp.all(
        mask
        == (
            jnp.arange(capacity, dtype=jnp.int32)[None, :]
            < count[:, None]
        ),
        axis=1,
    )
    safe_current = jnp.clip(current, 0, capacity - 1)
    active_input = (
        (current >= 0)
        & (current < capacity)
        & mask[jnp.arange(batch), safe_current]
    )
    current_valid = (current == -1) | active_input
    finite_path = jnp.all(
        jnp.isfinite(jnp.where(mask[..., None], path, 0.0)),
        axis=(1, 2),
    )
    distance_valid = (
        (jnp.isfinite(previous_distance) & (previous_distance >= 0.0))
        | jnp.isposinf(previous_distance)
    )
    selector_valid = (
        jnp.all(jnp.isfinite(selector) & (selector >= 0.0), axis=1)
        & jnp.any(selector > 0.0, axis=1)
    )
    available = (
        contiguous
        & current_valid
        & finite_path
        & jnp.all(jnp.isfinite(last) & jnp.isfinite(entity), axis=1)
        & distance_valid
        & selector_valid
    )
    waypoint = path[jnp.arange(batch), safe_current]
    projected_delta = (waypoint - entity) * selector
    distance_squared = jnp.sum(projected_delta * projected_delta, axis=1)
    exact = (
        distance_squared
        <= jnp.float32(PATH_FOLLOWER_EXACT_DISTANCE_SQUARED)
    )
    near = (
        distance_squared
        < jnp.float32(PATH_FOLLOWER_NEAR_DISTANCE_SQUARED)
    )
    projection = jnp.sum(
        ((last - waypoint) * selector)
        * ((entity - waypoint) * selector),
        axis=1,
    )
    projection_branch = ~exact & (
        ~near | (distance_squared > previous_distance)
    )
    reached = available & active_input & (
        exact
        | (near & (distance_squared <= previous_distance))
        | (projection_branch & (projection < 0.0))
    )
    next_index = safe_current + 1
    safe_next = jnp.clip(next_index, 0, capacity - 1)
    next_active = (
        reached
        & (next_index < capacity)
        & mask[jnp.arange(batch), safe_next]
    )
    output_index = jnp.where(
        reached,
        jnp.where(next_active, next_index, _EMPTY),
        current,
    )
    output_active = available & (output_index >= 0)
    safe_output = jnp.clip(output_index, 0, capacity - 1)
    target = path[jnp.arange(batch), safe_output]
    output_distance = jnp.where(
        active_input,
        distance_squared,
        previous_distance,
    )
    output_distance = jnp.where(
        next_active,
        jnp.float32(jnp.inf),
        output_distance,
    )
    return PathFollowerWaypointResult(
        available=available,
        current_waypoint_index=jnp.where(available, output_index, _EMPTY),
        last_waypoint_position=jnp.where(
            (available & reached)[:, None],
            waypoint,
            jnp.where(available[:, None], last, 0.0),
        ),
        current_waypoint_distance_squared=jnp.where(
            available,
            output_distance,
            jnp.float32(0.0),
        ),
        current_target_position=jnp.where(
            output_active[:, None],
            target,
            jnp.float32(0.0),
        ),
        active=output_active,
        reached_waypoint=reached,
        path_finished=available & ~output_active,
        should_smooth_path=next_active,
        invalid=~available,
    )


def path_follower_steering_result(
    current_position: Array,
    target_position: Array,
    last_waypoint_position: Array,
    component_selector: Array,
    previous_direction: Array,
    *,
    waypoint_radius: Array | float,
    relative_speed: Array | float,
    relative_speed_waypoint: Array | float,
    rejection_weight: Array | float = (
        PATH_FOLLOWER_DEFAULT_REJECTION_WEIGHT
    ),
) -> PathFollowerSteeringResult:
    """Apply native path rejection and waypoint-radius speed selection."""

    current = jnp.asarray(current_position, dtype=jnp.float32)
    target = jnp.asarray(target_position, dtype=jnp.float32)
    last = jnp.asarray(last_waypoint_position, dtype=jnp.float32)
    selector = jnp.asarray(component_selector, dtype=jnp.float32)
    previous = jnp.asarray(previous_direction, dtype=jnp.float32)
    if current.ndim != 2 or current.shape[0] == 0 or current.shape[1] != 3:
        raise ValueError("current_position must have non-empty shape [batch,3]")
    batch = current.shape[0]
    for name, value in (
        ("target_position", target),
        ("last_waypoint_position", last),
        ("component_selector", selector),
        ("previous_direction", previous),
    ):
        if value.shape != (batch, 3):
            raise ValueError(f"{name} must have shape [batch,3]")
    radius = _batch_float(waypoint_radius, batch, "waypoint_radius")
    speed = _batch_float(relative_speed, batch, "relative_speed")
    waypoint_speed = _batch_float(
        relative_speed_waypoint,
        batch,
        "relative_speed_waypoint",
    )
    weight = _batch_float(
        rejection_weight,
        batch,
        "rejection_weight",
    )
    delta = target - current
    length = jnp.linalg.norm(delta, axis=1)
    outside = length > radius
    path = (target - last) * selector
    offset = (current - last) * selector
    path_squared = jnp.sum(path * path, axis=1)
    projection_scale = jnp.sum(path * offset, axis=1) / jnp.where(
        path_squared > 0.0,
        path_squared,
        jnp.float32(1.0),
    )
    rejection = (
        offset - path * projection_scale[:, None]
    ) * weight[:, None]
    inside_direction = jnp.where(
        (length > PATH_FOLLOWER_DIRECTION_REFRESH_DISTANCE)[:, None],
        delta,
        previous,
    )
    direction = jnp.where(
        outside[:, None],
        delta - rejection,
        inside_direction,
    )
    finite_vectors = jnp.all(
        jnp.isfinite(current)
        & jnp.isfinite(target)
        & jnp.isfinite(last)
        & jnp.isfinite(selector)
        & jnp.isfinite(previous),
        axis=1,
    )
    available = (
        finite_vectors
        & jnp.all(selector >= 0.0, axis=1)
        & jnp.any(selector > 0.0, axis=1)
        & jnp.isfinite(radius)
        & (radius > 0.0)
        & jnp.isfinite(speed)
        & (speed >= 0.0)
        & jnp.isfinite(waypoint_speed)
        & (waypoint_speed >= 0.0)
        & jnp.isfinite(weight)
        & (weight >= 0.0)
        & jnp.isfinite(length)
        & (length > 0.0)
        & (~outside | (path_squared > 0.0))
    )
    selected_speed = jnp.where(outside, speed, waypoint_speed)
    translation = direction * (
        selected_speed
        / jnp.where(length > 0.0, length, jnp.float32(1.0))
    )[:, None]
    return PathFollowerSteeringResult(
        available=available,
        translation=jnp.where(available[:, None], translation, 0.0),
        rejection=jnp.where(
            (available & outside)[:, None],
            rejection,
            0.0,
        ),
        max_distance=jnp.where(available, length, 0.0),
        uses_waypoint_speed=available & ~outside,
        invalid=~available,
    )


def path_follower_contract() -> dict[str, object]:
    """Return the fixed-shape PathFollower source/certification boundary."""

    return {
        "schema": PATH_FOLLOWER_SCHEMA,
        "version": PATH_FOLLOWER_VERSION,
        "functions": [
            "path_follower_waypoint_result",
            "path_follower_steering_result",
        ],
        "source": {
            "advance": (
                "PathFollower.updateCurrentTarget"
                "->NPCPhysicsMath.dotProduct"
                "->MotionControllerBase.waypointDistanceSquared"
            ),
            "steering": (
                "PathFollower.executePath->PathFollower.computeRejection"
            ),
        },
        "units": "blocks_and_projected_block_distance_squared",
        "configuration": {
            "component_selector": "active_motion_controller_live_value",
            "rejection_weight_default": (
                PATH_FOLLOWER_DEFAULT_REJECTION_WEIGHT
            ),
            "waypoint_radius": "role_authored_required_input",
            "relative_speeds": "role_authored_required_inputs",
        },
        "waypoint_advance": {
            "exact_distance_squared": (
                PATH_FOLLOWER_EXACT_DISTANCE_SQUARED
            ),
            "near_distance_squared": (
                PATH_FOLLOWER_NEAR_DISTANCE_SQUARED
            ),
            "near_rule": (
                "current_distance_not_greater_than_previous_distance"
            ),
            "otherwise": "projected_dot_less_than_zero",
            "maximum_advance_per_call": 1,
        },
        "steering": {
            "outside_radius": (
                "target_delta_minus_weighted_path_rejection"
            ),
            "inside_radius": (
                "refresh_direction_above_0.1_else_reuse_previous_direction"
            ),
            "speed_branch": (
                "relative_speed_outside_relative_speed_waypoint_inside"
            ),
        },
        "fail_closed": [
            "non_contiguous_or_invalid_path",
            "non_finite_or_negative_configuration",
            "zero_target_distance",
            "zero_projected_path_outside_waypoint_radius",
        ],
        "native_trace": {
            "schema": "hytalerl_native_path_follower_trace_v1",
            "reset_option": "native_navigation_trace",
            "fixture": "path_follower",
            "subject": "kill_trork_target",
            "stage": "after_avoidance_before_steering",
            "transition": (
                "consecutive_same_follower_current_is_same_or_previous_next"
            ),
            "discontinuities": [
                "missing_or_non_find_motion",
                "path_follower_changed",
                "previous_path_inactive",
                "path_smoothed_or_replanned",
            ],
            "steering_comparable_only_when": (
                "body_translation_present_and_unmodified_by_avoidance_or_"
                "separation_and_max_distance_matches_current_waypoint"
            ),
            "default_runtime_overhead": "disabled_and_wire_fields_omitted",
        },
        "certification": {
            "source": "installed_0.5.7_class_hash_pinned",
            "runtime": (
                "native_trace_published_shared_T1_comparison_pending"
            ),
            "duplicate_trace_harness": False,
        },
        "not_yet_ported": [
            "PathFollower.smoothPath_probeMove_shortcutting",
            "heading_blending_and_requirePreciseMovement_side_effects",
            "BodyMotionFindBase_and_WithTarget_replan_policy",
        ],
        "provenance": "native_source_port_jax_float32",
    }


def path_follower_contract_sha256() -> str:
    payload = json.dumps(
        path_follower_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _batch_float(
    value: Array | float,
    batch: int,
    label: str,
) -> Array:
    result = jnp.asarray(value, dtype=jnp.float32)
    if result.ndim == 0:
        result = jnp.broadcast_to(result, (batch,))
    if result.shape != (batch,):
        raise ValueError(f"{label} must be scalar or shape [batch]")
    return result
