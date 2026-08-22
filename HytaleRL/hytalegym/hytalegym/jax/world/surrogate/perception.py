"""Bounded per-observer entity visibility over surrogate world geometry."""

from __future__ import annotations

import hashlib
import json
import operator

import jax
import jax.numpy as jnp

from hytalegym.jax.world.surrogate.capabilities import (
    SURROGATE_QUERY_CAPACITY,
    SURROGATE_QUERY_COVERAGE,
    SURROGATE_QUERY_ENTITY_GEOMETRY,
    SURROGATE_QUERY_ENTITY_LOS_ENDPOINT,
    SURROGATE_QUERY_INVALID,
    SURROGATE_QUERY_RUNTIME,
)
from hytalegym.jax.world.perception import (
    NATIVE_VIEW_MAX_FULL_ANGLE_RADIANS,
    native_view_sector,
)
from hytalegym.jax.world.surrogate.channels import (
    surrogate_perception_channel_contract,
)
from hytalegym.jax.world.surrogate.los import (
    MAX_SURROGATE_LOS_CELLS,
    surrogate_perception_line_of_sight_result,
)
from hytalegym.jax.world.surrogate.types import (
    SurrogateAtlas,
    SurrogateEntityVisibilityResult,
    SurrogateRuntimeState,
)

SURROGATE_ENTITY_VISIBILITY_CAPACITY = 16
MAX_SURROGATE_ENTITY_VISIBILITY_CAPACITY = 64
SURROGATE_ENTITY_VISIBILITY_DEFAULT_MAX_CELLS = 128
SURROGATE_ENTITY_VISIBILITY_SCHEMA = "hytalerl_surrogate_entity_visibility_v5"
SURROGATE_ENTITY_VISIBILITY_VERSION = 5
_EPSILON = jnp.float32(1.0e-6)
_TWO_PI = jnp.float32(NATIVE_VIEW_MAX_FULL_ANGLE_RADIANS)

# Versioned learner-derived coverage points inside an exact target AABB.
# Native Hytale 0.5.7 visibility remains the separate single target-offset ray.
SURROGATE_VISIBILITY_SAMPLE_FRACTIONS = jnp.asarray(
    (
        (0.50, 0.50, 0.50),
        (0.50, 0.85, 0.50),
        (0.50, 0.20, 0.50),
        (0.15, 0.55, 0.50),
        (0.85, 0.55, 0.50),
        (0.50, 0.55, 0.15),
        (0.50, 0.55, 0.85),
        (0.20, 0.80, 0.50),
        (0.80, 0.80, 0.50),
    ),
    dtype=jnp.float32,
)


def surrogate_entity_visibility_contract() -> dict[str, object]:
    """Return the canonical host manifest for this query schema."""

    return {
        "schema": SURROGATE_ENTITY_VISIBILITY_SCHEMA,
        "version": SURROGATE_ENTITY_VISIBILITY_VERSION,
        "default_capacity": SURROGATE_ENTITY_VISIBILITY_CAPACITY,
        "maximum_capacity": MAX_SURROGATE_ENTITY_VISIBILITY_CAPACITY,
        "default_max_los_cells": SURROGATE_ENTITY_VISIBILITY_DEFAULT_MAX_CELLS,
        "maximum_los_cells": MAX_SURROGATE_LOS_CELLS,
        "sample_fractions": jax.device_get(
            SURROGATE_VISIBILITY_SAMPLE_FRACTIONS
        ).tolist(),
        "native_ray": "single_block_ray",
        "view_sector": {
            "plane": "horizontal_xz",
            "heading": "head_yaw_forward",
            "angle": "full_width_radians",
            "wide_cone": "negated_rear_complement",
            "colocated_squared_epsilon": 1.0e-6,
        },
        "role_opaque_blocks": "caller_selected_palette_mask_per_observer",
        "entity_obstruction": "caller_selected_per_observer",
        "partial_visibility": "learner_derived",
        "distance_metric": {
            "formula": "norm((target_position-observer_position)*component_selector)",
            "observer_origin": "transform_position_not_los_eye",
            "default_component_selector": [1.0, 1.0, 1.0],
            "use_projected_distance_false": [1.0, 1.0, 1.0],
            "use_projected_distance_true": (
                "active_motion_controller_component_selector"
            ),
            "component_selector_scope": "live_per_observer_per_query",
            "controller_class_defaults_are_cache_keys": False,
            "planar_component_selector": "distinct_steering_mask",
            "selection_order": "ascending_metric_distance_stable_slot_tiebreak",
        },
        "line_of_sight_predicates": {
            "perception": {
                "function": "surrogate_perception_line_of_sight_result",
                "test": "block_opacity_plus_role_opaque_palette",
                "unpublished_geometry": "blocked",
            },
            "hit_confirmation": {
                "function": "surrogate_hitbox_line_of_sight_result",
                "test": "solid_rotated_detail_box_closed_segment_intersection",
                "unpublished_geometry": (
                    "non_blocking_with_geometry_exhausted_diagnostic"
                ),
                "capacity_or_invalid": "blocked",
            },
        },
        "staleness": {
            "jax_policy": "fresh_per_query",
            "native_policy": "forward_inverse_independent_jittered_ttl_cache",
            "native_los_ttl_seconds": [0.09, 0.11],
            "native_position_ttl_seconds": 0.2,
            "declared_fidelity_divergence": True,
        },
        "privilege": {
            "query_boundary": "privileged_world_state",
            "actor_legal_predicates": [
                "native_block_visible",
                "target_point_visible",
                "sample_visible",
                "visible_fraction",
                "visible_any",
                "fully_visible",
                "partially_visible",
                "occluded",
            ],
            "must_be_visibility_gated": [
                "entity_slot",
                "distance",
                "target_identity",
                "target_kinematics",
            ],
        },
        "environment_channels": surrogate_perception_channel_contract(),
    }


def surrogate_entity_visibility_contract_sha256() -> str:
    """Return the checkpoint-pinnable hash of the canonical query contract."""

    encoded = json.dumps(
        surrogate_entity_visibility_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def query_surrogate_entity_visibility(
    atlas: SurrogateAtlas,
    runtime: SurrogateRuntimeState,
    observer_position: jax.Array,
    observer_eye_position: jax.Array,
    observer_forward: jax.Array,
    maximum_distance: jax.Array | float,
    view_sector_full_angle_radians: jax.Array | float,
    *,
    motion_distance_component_selector: (
        jax.Array | tuple[float, float, float]
    ) = (1.0, 1.0, 1.0),
    observer_entity_slots: jax.Array | None = None,
    entity_occluder_mask: jax.Array | None = None,
    role_opaque_cell_mask: jax.Array | None = None,
    capacity: int = SURROGATE_ENTITY_VISIBILITY_CAPACITY,
    max_cells: int = SURROGATE_ENTITY_VISIBILITY_DEFAULT_MAX_CELLS,
) -> SurrogateEntityVisibilityResult:
    """Select nearby entities and compute native plus derived visibility rays.

    ``observer_position`` is the transform position used for candidate range
    and view bearing; ``observer_eye_position`` is only the ray origin.
    ``observer_forward`` must be derived from head yaw; its vertical component
    is ignored. The view-sector angle is the authored full horizontal width,
    matching Hytale 0.5.7 including its inverted rear complement above pi.
    Candidate distance is the norm after multiplying each transform-position
    delta by ``motion_distance_component_selector``. Pass ``[1, 1, 1]`` when
    native ``UseProjectedDistance`` is false; otherwise pass the active motion
    controller's live component selector for each observer and query. The
    selector is runtime-mutable and must not be cached by controller class.
    The controller's separate planar selector is a steering mask, not this
    sensor-distance input.
    ``native_block_visible`` is the Hytale-style single target-offset ray
    against blocks. Entity occlusion is separate because native 0.5.7 checks
    friendly-fire bodies separately from its default block LOS predicate.
    """

    if not isinstance(atlas, SurrogateAtlas):
        raise TypeError("atlas must be a SurrogateAtlas")
    if not isinstance(runtime, SurrogateRuntimeState):
        raise TypeError("runtime must be a SurrogateRuntimeState")
    target_capacity = runtime.entity_active.shape[1]
    selected_capacity = _visibility_capacity(capacity, target_capacity)
    if (
        isinstance(max_cells, bool)
        or not isinstance(max_cells, int)
        or not 1 <= max_cells <= MAX_SURROGATE_LOS_CELLS
    ):
        raise ValueError(f"max_cells must be in [1, {MAX_SURROGATE_LOS_CELLS}]")

    positions = jnp.asarray(observer_position, dtype=jnp.float32)
    eyes = jnp.asarray(observer_eye_position, dtype=jnp.float32)
    forward = jnp.asarray(observer_forward, dtype=jnp.float32)
    if (
        positions.ndim != 3
        or positions.shape[2] != 3
        or eyes.shape != positions.shape
        or forward.shape != positions.shape
    ):
        raise ValueError(
            "observer position, eye position, and forward must have shape [B, A, 3]"
        )
    batch, observers, _ = positions.shape
    if runtime.environment_world_id.shape != (batch,):
        raise ValueError("runtime and observers must share a batch dimension")
    maximum = _observer_parameter(
        maximum_distance,
        batch,
        observers,
        "maximum_distance",
    )
    view_angle = _observer_parameter(
        view_sector_full_angle_radians,
        batch,
        observers,
        "view_sector_full_angle_radians",
    )
    distance_selector = _observer_vector_parameter(
        motion_distance_component_selector,
        batch,
        observers,
        "motion_distance_component_selector",
    )
    self_slots = _observer_slots(observer_entity_slots, batch, observers)
    role_opaque = _observer_role_opaque_mask(
        role_opaque_cell_mask,
        batch,
        observers,
        atlas.cell_flags.shape[0],
    )
    occluders = (
        None
        if entity_occluder_mask is None
        else _occluder_mask(
            entity_occluder_mask,
            batch,
            observers,
            target_capacity,
        )
    )

    forward_xz = forward[..., (0, 2)]
    forward_xz_norm = jnp.linalg.norm(forward_xz, axis=2)
    self_slot_valid = (self_slots >= -1) & (self_slots < target_capacity)
    observer_numeric = (
        jnp.all(
            jnp.isfinite(positions)
            & jnp.isfinite(eyes)
            & jnp.isfinite(forward),
            axis=2,
        )
        & jnp.isfinite(maximum)
        & (maximum > 0.0)
        & jnp.isfinite(view_angle)
        & (view_angle >= 0.0)
        & (view_angle <= _TWO_PI)
        & (forward_xz_norm > _EPSILON)
        & jnp.all(jnp.isfinite(distance_selector), axis=2)
        & jnp.any(jnp.abs(distance_selector) > _EPSILON, axis=2)
        & self_slot_valid
    )
    active = runtime.entity_active & runtime.entity_initialized
    entity_numeric = jnp.all(jnp.isfinite(runtime.entity_position), axis=2)
    runtime_numeric = jnp.all(~active | entity_numeric, axis=1)
    runtime_valid = ~runtime.unsupported_mechanics & runtime_numeric
    observer_ready = observer_numeric & runtime_valid[:, None]
    safe_positions = jnp.where(observer_ready[..., None], positions, 0.0)
    safe_eyes = jnp.where(observer_ready[..., None], eyes, 0.0)
    safe_forward_xz = jnp.where(
        observer_ready[..., None],
        forward_xz / jnp.maximum(forward_xz_norm[..., None], _EPSILON),
        jnp.asarray((0.0, 1.0), dtype=jnp.float32),
    )

    delta = (
        runtime.entity_position[:, None, :, :]
        - safe_positions[:, :, None, :]
    )
    metric_delta = delta * distance_selector[:, :, None, :]
    distance_squared = jnp.sum(metric_delta * metric_delta, axis=3)
    distance = jnp.sqrt(jnp.maximum(distance_squared, 0.0))
    in_view_sector = native_view_sector(
        delta[..., (0, 2)],
        safe_forward_xz[:, :, None, :],
        view_angle[:, :, None],
    )
    self_mask = (
        jnp.arange(target_capacity, dtype=jnp.int32)[None, None, :]
        == self_slots[:, :, None]
    )
    candidates = (
        active[:, None, :]
        & entity_numeric[:, None, :]
        & observer_ready[:, :, None]
        & ~self_mask
        & (distance <= maximum[:, :, None])
        & in_view_sector
    )
    candidate_count = jnp.sum(candidates, axis=2)
    capacity_exceeded = candidate_count > selected_capacity
    ordered = jnp.argsort(
        jnp.where(candidates, distance_squared, jnp.inf),
        axis=2,
        stable=True,
    )
    selected_slots = ordered[:, :, :selected_capacity].astype(jnp.int32)
    batch_index = jnp.arange(batch, dtype=jnp.int32)[:, None, None]
    observer_index = jnp.arange(observers, dtype=jnp.int32)[None, :, None]
    selected_candidates = candidates[
        batch_index,
        observer_index,
        selected_slots,
    ]
    published_mask = selected_candidates & ~capacity_exceeded[..., None]

    target_position = runtime.entity_position[batch_index, selected_slots]
    target_bounds = runtime.entity_local_bounds[batch_index, selected_slots]
    target_geometry = runtime.entity_geometry_supported[
        batch_index,
        selected_slots,
    ]
    target_endpoint_supported = runtime.entity_los_offset_supported[
        batch_index,
        selected_slots,
    ]
    target_offset = runtime.entity_los_offset[batch_index, selected_slots]
    finite_bounds = jnp.all(jnp.isfinite(target_bounds), axis=3)
    positive_bounds = jnp.all(target_bounds[..., :3] < target_bounds[..., 3:], axis=3)
    target_geometry_valid = target_geometry & finite_bounds & positive_bounds
    finite_target_offset = jnp.all(jnp.isfinite(target_offset), axis=3)
    target_endpoint_valid = target_endpoint_supported & finite_target_offset
    native_target_ready = published_mask & target_endpoint_valid
    derived_target_ready = native_target_ready & target_geometry_valid

    target_endpoint = target_position + target_offset
    target_minimum = target_position + target_bounds[..., :3]
    target_extent = target_bounds[..., 3:] - target_bounds[..., :3]
    sample_points = (
        target_minimum[..., None, :]
        + target_extent[..., None, :]
        * SURROGATE_VISIBILITY_SAMPLE_FRACTIONS[None, None, None, :, :]
    )
    endpoints = jnp.concatenate((target_endpoint[..., None, :], sample_points), axis=3)
    ray_required = jnp.concatenate(
        (
            native_target_ready[..., None],
            jnp.broadcast_to(
                derived_target_ready[..., None],
                sample_points.shape[:-1],
            ),
        ),
        axis=3,
    )
    starts = jnp.broadcast_to(safe_eyes[:, :, None, None, :], endpoints.shape)
    endpoints = jnp.where(ray_required[..., None], endpoints, starts)
    flat_shape = (batch, -1, 3)
    flat_starts = starts.reshape(flat_shape)
    flat_ends = endpoints.reshape(flat_shape)
    if role_opaque is None:
        block_result = jax.vmap(
            lambda start, end: surrogate_perception_line_of_sight_result(
                atlas,
                runtime,
                start,
                end,
                max_cells=max_cells,
            ),
            in_axes=(1, 1),
            out_axes=1,
        )(flat_starts, flat_ends)
    else:
        rays_per_observer = selected_capacity * endpoints.shape[3]
        ray_observer = (
            jnp.arange(flat_starts.shape[1], dtype=jnp.int32)
            // rays_per_observer
        )
        block_result = jax.vmap(
            lambda start, end, observer: surrogate_perception_line_of_sight_result(
                atlas,
                runtime,
                start,
                end,
                role_opaque_cell_mask=role_opaque[:, observer, :],
                max_cells=max_cells,
            ),
            in_axes=(1, 1, 0),
            out_axes=1,
        )(flat_starts, flat_ends, ray_observer)
    ray_shape = endpoints.shape[:-1]
    block_visible = block_result.visible.reshape(ray_shape)
    block_exhausted = block_result.geometry_exhausted.reshape(ray_shape)
    block_capacity = block_result.capacity_exceeded.reshape(ray_shape)

    if occluders is None:
        unknown_occluder = jnp.zeros((batch, observers), dtype=jnp.bool_)
        entity_blocked_rays = jnp.zeros(ray_shape, dtype=jnp.bool_)
    else:
        eligible_occluders = occluders & active[:, None, :]
        requested_occluders = eligible_occluders & runtime.entity_blocks_los[:, None, :]
        occluder_bounds = runtime.entity_local_bounds
        occluder_geometry_valid = (
            runtime.entity_geometry_supported
            & jnp.all(jnp.isfinite(occluder_bounds), axis=2)
            & jnp.all(occluder_bounds[..., :3] < occluder_bounds[..., 3:], axis=2)
        )
        unknown_occluder = jnp.any(
            eligible_occluders
            & (
                ~runtime.entity_geometry_supported[:, None, :]
                | (
                    runtime.entity_blocks_los[:, None, :]
                    & ~occluder_geometry_valid[:, None, :]
                )
            ),
            axis=2,
        )
        blocker_boxes = jnp.concatenate(
            (
                runtime.entity_position + runtime.entity_local_bounds[..., :3],
                runtime.entity_position + runtime.entity_local_bounds[..., 3:],
            ),
            axis=2,
        )
        entity_hits = _segment_hits_boxes(
            flat_starts[:, :, None, :],
            flat_ends[:, :, None, :],
            blocker_boxes[:, None, :, :],
        )
        ray_target_slots = jnp.broadcast_to(
            selected_slots[..., None],
            ray_shape,
        ).reshape(batch, -1)
        ray_self_slots = jnp.broadcast_to(
            self_slots[:, :, None, None],
            ray_shape,
        ).reshape(batch, -1)
        blocker_slots = jnp.arange(target_capacity, dtype=jnp.int32)
        ray_occluders = jnp.broadcast_to(
            requested_occluders[:, :, None, None, :],
            ray_shape + (target_capacity,),
        ).reshape(batch, -1, target_capacity)
        blocker_mask = (
            ray_occluders
            & occluder_geometry_valid[:, None, :]
            & (blocker_slots[None, None, :] != ray_target_slots[..., None])
            & (blocker_slots[None, None, :] != ray_self_slots[..., None])
        )
        entity_blocked_rays = jnp.any(entity_hits & blocker_mask, axis=2).reshape(
            ray_shape
        )

    native_observer_diagnostics = jnp.zeros(
        (batch, observers),
        dtype=jnp.uint32,
    )
    native_observer_diagnostics |= jnp.where(
        ~observer_numeric,
        jnp.uint32(SURROGATE_QUERY_INVALID),
        jnp.uint32(0),
    )
    native_observer_diagnostics |= jnp.where(
        ~runtime_valid[:, None],
        jnp.uint32(SURROGATE_QUERY_RUNTIME),
        jnp.uint32(0),
    )
    native_observer_diagnostics |= jnp.where(
        capacity_exceeded,
        jnp.uint32(SURROGATE_QUERY_CAPACITY),
        jnp.uint32(0),
    )
    observer_diagnostics = native_observer_diagnostics
    observer_diagnostics |= jnp.where(
        unknown_occluder,
        jnp.uint32(SURROGATE_QUERY_ENTITY_GEOMETRY),
        jnp.uint32(0),
    )

    diagnostics = jnp.zeros(published_mask.shape, dtype=jnp.uint32)
    diagnostics |= jnp.where(
        published_mask & ~target_geometry_valid,
        jnp.uint32(SURROGATE_QUERY_ENTITY_GEOMETRY),
        jnp.uint32(0),
    )
    diagnostics |= jnp.where(
        published_mask & ~target_endpoint_valid,
        jnp.uint32(SURROGATE_QUERY_ENTITY_LOS_ENDPOINT),
        jnp.uint32(0),
    )
    diagnostics |= jnp.where(
        published_mask
        & (
            (target_geometry & ~finite_bounds)
            | (target_endpoint_supported & ~finite_target_offset)
        ),
        jnp.uint32(SURROGATE_QUERY_INVALID),
        jnp.uint32(0),
    )
    diagnostics |= jnp.where(
        derived_target_ready & jnp.any(block_exhausted, axis=3),
        jnp.uint32(SURROGATE_QUERY_COVERAGE),
        jnp.uint32(0),
    )
    diagnostics |= jnp.where(
        derived_target_ready & jnp.any(block_capacity, axis=3),
        jnp.uint32(SURROGATE_QUERY_CAPACITY),
        jnp.uint32(0),
    )
    diagnostics |= jnp.where(
        published_mask & (observer_diagnostics[..., None] != 0),
        observer_diagnostics[..., None],
        jnp.uint32(0),
    )
    available = published_mask & (diagnostics == 0)
    native_available = (
        native_target_ready
        & (native_observer_diagnostics[..., None] == 0)
        & ~block_exhausted[..., 0]
        & ~block_capacity[..., 0]
    )
    native_block_visible = native_available & block_visible[..., 0]
    entity_blocked = available & entity_blocked_rays[..., 0]
    target_point_visible = native_block_visible & ~entity_blocked
    sample_visible = (
        available[..., None] & block_visible[..., 1:] & ~entity_blocked_rays[..., 1:]
    )
    visible_count = jnp.sum(sample_visible, axis=3)
    sample_count = SURROGATE_VISIBILITY_SAMPLE_FRACTIONS.shape[0]
    visible_fraction = jnp.where(
        available,
        visible_count.astype(jnp.float32) / jnp.float32(sample_count),
        0.0,
    )
    any_sample = jnp.any(sample_visible, axis=3)
    all_samples = jnp.all(sample_visible, axis=3)
    visible_any = available & (target_point_visible | any_sample)
    fully_visible = available & target_point_visible & all_samples
    partially_visible = visible_any & ~fully_visible
    occluded = available & ~visible_any

    return SurrogateEntityVisibilityResult(
        target_mask=published_mask,
        available=available,
        diagnostics=diagnostics,
        observer_diagnostics=observer_diagnostics,
        capacity_exceeded=capacity_exceeded,
        entity_slot=jnp.where(published_mask, selected_slots, -1),
        distance=jnp.where(
            published_mask,
            distance[batch_index, observer_index, selected_slots],
            0.0,
        ),
        native_available=native_available,
        native_block_visible=native_block_visible,
        entity_blocked=entity_blocked,
        target_point_visible=target_point_visible,
        sample_visible=sample_visible,
        visible_fraction=visible_fraction,
        visible_any=visible_any,
        fully_visible=fully_visible,
        partially_visible=partially_visible,
        occluded=occluded,
    )


def _segment_hits_boxes(
    start: jax.Array,
    end: jax.Array,
    boxes: jax.Array,
) -> jax.Array:
    delta = end - start
    moving = jnp.abs(delta) > _EPSILON
    safe_delta = jnp.where(moving, delta, 1.0)
    first = (boxes[..., :3] + _EPSILON - start) / safe_delta
    second = (boxes[..., 3:] - _EPSILON - start) / safe_delta
    entry = jnp.where(moving, jnp.minimum(first, second), -jnp.inf)
    exit_time = jnp.where(moving, jnp.maximum(first, second), jnp.inf)
    stationary_inside = (
        (~moving)
        & (start > boxes[..., :3] + _EPSILON)
        & (start < boxes[..., 3:] - _EPSILON)
    )
    lower = jnp.max(entry, axis=-1)
    upper = jnp.min(exit_time, axis=-1)
    return (
        jnp.all(moving | stationary_inside, axis=-1)
        & (lower < upper - _EPSILON)
        & (lower < 1.0 - _EPSILON)
        & (upper > _EPSILON)
    )


def _observer_parameter(
    value: jax.Array | float,
    batch: int,
    observers: int,
    label: str,
) -> jax.Array:
    result = jnp.asarray(value, dtype=jnp.float32)
    if result.ndim == 0:
        return jnp.broadcast_to(result, (batch, observers))
    if result.shape == (batch,):
        return jnp.broadcast_to(result[:, None], (batch, observers))
    if result.shape != (batch, observers):
        raise ValueError(f"{label} must be scalar or have shape [B] or [B, A]")
    return result


def _observer_vector_parameter(
    value: jax.Array,
    batch: int,
    observers: int,
    name: str,
) -> jax.Array:
    result = jnp.asarray(value, dtype=jnp.float32)
    if result.shape == (3,):
        return jnp.broadcast_to(result, (batch, observers, 3))
    if result.shape == (batch, 3):
        return jnp.broadcast_to(result[:, None, :], (batch, observers, 3))
    if result.shape != (batch, observers, 3):
        raise ValueError(f"{name} must have shape [3], [B, 3], or [B, A, 3]")
    return result


def _observer_slots(
    value: jax.Array | None,
    batch: int,
    observers: int,
) -> jax.Array:
    if value is None:
        return jnp.full((batch, observers), -1, dtype=jnp.int32)
    result = jnp.asarray(value, dtype=jnp.int32)
    if result.shape != (batch, observers):
        raise ValueError("observer_entity_slots must have shape [B, A]")
    return result


def _occluder_mask(
    value: jax.Array,
    batch: int,
    observers: int,
    entities: int,
) -> jax.Array:
    result = jnp.asarray(value, dtype=jnp.bool_)
    if result.shape == (batch, entities):
        return jnp.broadcast_to(result[:, None, :], (batch, observers, entities))
    if result.shape != (batch, observers, entities):
        raise ValueError("entity_occluder_mask must have shape [B, E] or [B, A, E]")
    return result


def _observer_role_opaque_mask(
    value: jax.Array | None,
    batch: int,
    observers: int,
    palette_capacity: int,
) -> jax.Array | None:
    if value is None:
        return None
    result = jnp.asarray(value)
    if result.dtype != jnp.bool_:
        raise TypeError("role_opaque_cell_mask must have boolean dtype")
    if result.shape == (palette_capacity,):
        return jnp.broadcast_to(
            result,
            (batch, observers, palette_capacity),
        )
    if result.shape == (batch, palette_capacity):
        return jnp.broadcast_to(
            result[:, None, :],
            (batch, observers, palette_capacity),
        )
    if result.shape != (batch, observers, palette_capacity):
        raise ValueError(
            "role_opaque_cell_mask must have shape [C], [B, C], or [B, A, C]"
        )
    return result


def _visibility_capacity(value: int, entity_capacity: int) -> int:
    if isinstance(value, bool):
        raise TypeError("capacity must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError("capacity must be an integer") from error
    maximum = min(MAX_SURROGATE_ENTITY_VISIBILITY_CAPACITY, entity_capacity)
    if not 1 <= result <= maximum:
        raise ValueError(f"capacity must be in [1, {maximum}]")
    return result


__all__ = [
    "MAX_SURROGATE_ENTITY_VISIBILITY_CAPACITY",
    "SURROGATE_ENTITY_VISIBILITY_CAPACITY",
    "SURROGATE_ENTITY_VISIBILITY_DEFAULT_MAX_CELLS",
    "SURROGATE_ENTITY_VISIBILITY_SCHEMA",
    "SURROGATE_ENTITY_VISIBILITY_VERSION",
    "SURROGATE_VISIBILITY_SAMPLE_FRACTIONS",
    "query_surrogate_entity_visibility",
    "surrogate_entity_visibility_contract",
    "surrogate_entity_visibility_contract_sha256",
]
