"""Fail-closed swept-volume clearance over compiled geometry providers."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.world.geometry import resolve_aabb_motion
from hytalegym.jax.world.region.geometry import region_resolve_aabb_motion
from hytalegym.jax.world.region.types import RegionGeometryState
from hytalegym.jax.world.types import GeometryState


Array = jax.Array
GeometryProvider = GeometryState | RegionGeometryState
SWEPT_VOLUME_CLEARANCE_SCHEMA = "hytalerl_swept_volume_clearance_v1"
SWEPT_VOLUME_CLEARANCE_VERSION = 1
DEFAULT_SWEEP_CLEARANCE_SEGMENTS = 16
MAX_SWEEP_CLEARANCE_SEGMENTS = 64
_EPSILON = jnp.float32(1.0e-6)
_MAX_SEGMENT_CELL_SPAN = jnp.float32(2.0)


class SweptVolumeClearanceResult(NamedTuple):
    """Straight-path AABB clearance and explicit failure diagnostics."""

    clear: Array
    available: Array
    collided: Array
    geometry_exhausted: Array
    capacity_exceeded: Array
    invalid: Array
    segment_count: Array


def geometry_swept_volume_clearance_result(
    geometry: GeometryProvider,
    position: Array,
    displacement: Array,
    bounds: Array | None = None,
    *,
    segment_capacity: int = DEFAULT_SWEEP_CLEARANCE_SEGMENTS,
) -> SweptVolumeClearanceResult:
    """Test a straight AABB corridor without treating a ray as a volume.

    Long paths are split into a fixed number of exact continuous sweeps. The
    split only bounds each provider query; any collision in any segment blocks
    the complete path. Missing coverage, invalid numeric state, degenerate
    motion, or insufficient segment capacity fail closed.
    """

    if not isinstance(geometry, (GeometryState, RegionGeometryState)):
        raise TypeError("geometry must be GeometryState or RegionGeometryState")
    if (
        isinstance(segment_capacity, bool)
        or not isinstance(segment_capacity, int)
        or not 1 <= segment_capacity <= MAX_SWEEP_CLEARANCE_SEGMENTS
    ):
        raise ValueError(
            "segment_capacity must be an integer in "
            f"[1, {MAX_SWEEP_CLEARANCE_SEGMENTS}]"
        )
    points = jnp.asarray(position, dtype=jnp.float32)
    delta = jnp.asarray(displacement, dtype=jnp.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("position must have shape [batch, 3]")
    if delta.shape != points.shape:
        raise ValueError("displacement must match position")
    batch = points.shape[0]
    provider_bounds = _provider_bounds(geometry, batch)
    entity_bounds = (
        provider_bounds if bounds is None else _broadcast_bounds(bounds, batch)
    )
    width = entity_bounds[:, 3:] - entity_bounds[:, :3]
    finite = jnp.all(jnp.isfinite(points) & jnp.isfinite(delta), axis=1) & jnp.all(
        jnp.isfinite(entity_bounds), axis=1
    )
    valid_bounds = jnp.all(width > _EPSILON, axis=1)
    moving = jnp.any(jnp.abs(delta) > _EPSILON, axis=1)
    valid = finite & valid_bounds & moving
    safe_points = jnp.where(valid[:, None], points, 0.0)
    safe_delta = jnp.where(valid[:, None], delta, 0.0)
    safe_bounds = jnp.where(
        valid[:, None],
        entity_bounds,
        provider_bounds,
    )

    required = jnp.ones(batch, dtype=jnp.int32)
    capacity_exceeded = jnp.zeros(batch, dtype=jnp.bool_)
    if isinstance(geometry, GeometryState):
        result = resolve_aabb_motion(
            geometry,
            safe_points,
            safe_delta,
            safe_bounds,
        )
        collided = valid & result.collided
        exhausted = valid & result.geometry_exhausted
    else:
        width = safe_bounds[:, 3:] - safe_bounds[:, :3]
        axis_moving = jnp.abs(safe_delta) > _EPSILON
        segment_limit = _MAX_SEGMENT_CELL_SPAN - width
        unsupported_axis = axis_moving & (segment_limit <= _EPSILON)
        ratio = jnp.where(
            axis_moving,
            jnp.abs(safe_delta) / jnp.maximum(segment_limit, _EPSILON),
            0.0,
        )
        required = jnp.maximum(
            jnp.ceil(jnp.max(ratio, axis=1)).astype(jnp.int32),
            1,
        )
        capacity_exceeded = (
            jnp.any(unsupported_axis, axis=1) | (required > jnp.int32(segment_capacity))
        ) & valid
        supported = valid & ~capacity_exceeded
        step = safe_delta / required[:, None].astype(jnp.float32)

        def sweep_segment(
            index: int,
            carry: tuple[Array, Array],
        ) -> tuple[Array, Array]:
            collided, exhausted = carry
            active = supported & (index < required)
            start = safe_points + step * jnp.float32(index)
            segment_delta = jnp.where(active[:, None], step, 0.0)
            result = region_resolve_aabb_motion(
                geometry,
                start,
                segment_delta,
                safe_bounds,
            )
            return (
                collided | (active & result.collided),
                exhausted | (active & result.geometry_exhausted),
            )

        active_segments = jnp.max(jnp.where(supported, required, jnp.int32(0)))
        collided, exhausted = jax.lax.fori_loop(
            0,
            active_segments,
            sweep_segment,
            (
                jnp.zeros(batch, dtype=jnp.bool_),
                jnp.zeros(batch, dtype=jnp.bool_),
            ),
        )
    invalid = ~valid
    geometry_exhausted = exhausted | capacity_exceeded | invalid
    available = valid & ~capacity_exceeded & ~exhausted
    return SweptVolumeClearanceResult(
        clear=available & ~collided,
        available=available,
        collided=collided,
        geometry_exhausted=geometry_exhausted,
        capacity_exceeded=capacity_exceeded,
        invalid=invalid,
        segment_count=jnp.where(valid, required, 0),
    )


def _provider_bounds(
    geometry: GeometryProvider,
    batch: int,
) -> Array:
    if isinstance(geometry, RegionGeometryState):
        if geometry.environment_world_id.shape != (batch,):
            raise ValueError("Region geometry world IDs must match the query batch")
    elif geometry.origin.shape[0] not in (1, batch):
        raise ValueError("GeometryState batch must be one or match the query")
    return _broadcast_bounds(geometry.agent_bounds, batch)


def _broadcast_bounds(value: Array, batch: int) -> Array:
    result = jnp.asarray(value, dtype=jnp.float32)
    if result.shape == (6,):
        result = result[None, :]
    if result.ndim != 2 or result.shape[1] != 6:
        raise ValueError("bounds must have shape [6] or [batch|1, 6]")
    if result.shape[0] not in (1, batch):
        raise ValueError("bounds batch must be one or match the query")
    return jnp.broadcast_to(result, (batch, 6))


def swept_volume_clearance_contract() -> dict[str, object]:
    """Return the checkpoint-pinnable provider clearance contract."""

    return {
        "schema": SWEPT_VOLUME_CLEARANCE_SCHEMA,
        "version": SWEPT_VOLUME_CLEARANCE_VERSION,
        "function": "geometry_swept_volume_clearance_result",
        "providers": ["GeometryState", "RegionGeometryState"],
        "shape": {
            "position": ["batch", 3],
            "displacement": ["batch", 3],
            "bounds": ["batch_or_one", 6],
        },
        "predicate": "continuous_straight_swept_AABB_against_exact_boxes",
        "not_equivalent_to": [
            "point_ray",
            "post_collision_slide_path",
            "standable_support_sweep",
        ],
        "segmentation": {
            "provider": "RegionGeometryState",
            "local_geometry": "single_exact_full_window_sweep",
            "purpose": "bound_each_region_provider_cell_gather",
            "default_capacity": DEFAULT_SWEEP_CLEARANCE_SEGMENTS,
            "maximum_capacity": MAX_SWEEP_CLEARANCE_SEGMENTS,
            "per_axis_swept_interval_limit_blocks": 2.0,
            "collision_reduction": "any_segment_blocks_complete_path",
        },
        "fail_closed": [
            "non_finite_input",
            "inverted_or_degenerate_bounds",
            "zero_displacement",
            "provider_coverage_exhaustion",
            "segment_capacity_exceeded",
            "entity_extent_unsupported_for_moving_axis",
        ],
        "consumers": {
            "clear_force_path": "supported",
            "dodge_corridor_clear": "supported",
            "clear_projectile_flight": (
                "use_hitbox_LOS_or_projectile_shape_not_actor_bounds"
            ),
        },
        "provenance": "exact_geometry_provider_not_surrogate_ray",
    }


def swept_volume_clearance_contract_sha256() -> str:
    payload = json.dumps(
        swept_volume_clearance_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "DEFAULT_SWEEP_CLEARANCE_SEGMENTS",
    "MAX_SWEEP_CLEARANCE_SEGMENTS",
    "SWEPT_VOLUME_CLEARANCE_SCHEMA",
    "SWEPT_VOLUME_CLEARANCE_VERSION",
    "SweptVolumeClearanceResult",
    "geometry_swept_volume_clearance_result",
    "swept_volume_clearance_contract",
    "swept_volume_clearance_contract_sha256",
]
