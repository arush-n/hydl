"""Exact point-ray first contact for deployable and interaction targeting.

This is deliberately distinct from projectile contact. Native
``SpawnDeployableFromRaycastInteraction`` consumes a zero-width camera ray,
not a swept projectile volume. Long rays are split into bounded one-block
segments so Region geometry does not inherit the three-cell broad-phase limit
used by actor AABB motion.
"""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.world.geometry._sweep import aabb_first_contact_result
from hytalegym.jax.world.region.geometry import (
    region_aabb_first_contact_result,
)
from hytalegym.jax.world.region.types import RegionGeometryState
from hytalegym.jax.world.types import GeometryState


Array = jax.Array
GeometryProvider = GeometryState | RegionGeometryState
POINT_RAY_FIRST_CONTACT_SCHEMA = "hytalerl_point_ray_first_contact_v1"
POINT_RAY_FIRST_CONTACT_VERSION = 1
DEFAULT_POINT_RAY_SEGMENT_CAPACITY = 64
_EPSILON = jnp.float32(1.0e-6)
_POINT_BOUNDS = jnp.zeros((6,), dtype=jnp.float32)


class PointRayFirstContactResult(NamedTuple):
    """First closed-box contact and explicit evidence validity."""

    available: Array
    hit: Array
    hit_point: Array
    hit_normal: Array
    distance: Array
    geometry_exhausted: Array
    capacity_exceeded: Array
    invalid: Array
    segment_count: Array


def geometry_point_ray_first_contact_result(
    geometry: GeometryProvider,
    origin: Array,
    direction: Array,
    maximum_distance: Array | float,
    *,
    segment_capacity: int = DEFAULT_POINT_RAY_SEGMENT_CAPACITY,
) -> PointRayFirstContactResult:
    """Trace a zero-width camera ray through local or Region geometry.

    ``direction`` is normalized internally. ``maximum_distance`` may be a
    scalar or one value per batch row. Invalid inputs, insufficient static
    segment capacity, or missing geometry before contact fail closed. A hit
    terminates traversal, so unavailable geometry beyond that contact cannot
    invalidate an already-determined result.
    """

    if not isinstance(geometry, (GeometryState, RegionGeometryState)):
        raise TypeError("geometry must be GeometryState or RegionGeometryState")
    if (
        isinstance(segment_capacity, bool)
        or not isinstance(segment_capacity, int)
        or segment_capacity < 1
    ):
        raise ValueError("segment_capacity must be a positive integer")
    points = jnp.asarray(origin, dtype=jnp.float32)
    vectors = jnp.asarray(direction, dtype=jnp.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("origin must have shape [batch, 3]")
    if vectors.shape != points.shape:
        raise ValueError("direction must match origin")
    batch = points.shape[0]
    distances = _broadcast_distance(maximum_distance, batch)
    direction_length = jnp.linalg.norm(vectors, axis=1)
    finite = (
        jnp.all(jnp.isfinite(points), axis=1)
        & jnp.all(jnp.isfinite(vectors), axis=1)
        & jnp.isfinite(distances)
    )
    invalid = ~finite | (direction_length <= _EPSILON) | (distances <= 0.0)
    safe_length = jnp.where(direction_length > _EPSILON, direction_length, 1.0)
    unit = jnp.where(finite[:, None], vectors / safe_length[:, None], 0.0)
    safe_distance = jnp.where(invalid, 0.0, distances)
    capacity_exceeded = safe_distance > jnp.float32(segment_capacity)

    def active_segments(index: Array, terminal: Array) -> Array:
        start = index.astype(jnp.float32)
        return (
            ~terminal
            & ~invalid
            & ~capacity_exceeded
            & (start < safe_distance - _EPSILON)
        )

    def continue_trace(carry) -> Array:
        index, terminal, *_ = carry
        return (index < jnp.int32(segment_capacity)) & jnp.any(
            active_segments(index, terminal)
        )

    def trace_segment(carry):
        (
            index,
            terminal,
            exhausted,
            hit,
            hit_point,
            hit_normal,
            travelled,
            count,
        ) = carry
        start = index.astype(jnp.float32)
        active = active_segments(index, terminal)
        length = jnp.where(
            active,
            jnp.minimum(jnp.float32(1.0), safe_distance - start),
            jnp.float32(0.0),
        )
        contact = _point_segment_contact(
            geometry,
            points + unit * start,
            unit * length[:, None],
        )
        segment_exhausted = active & contact.geometry_exhausted
        segment_hit = active & contact.hit & ~segment_exhausted
        first = ~terminal & (segment_exhausted | segment_hit)
        return (
            index + jnp.int32(1),
            terminal | first,
            exhausted | segment_exhausted,
            hit | segment_hit,
            jnp.where(first[:, None], contact.contact_point, hit_point),
            jnp.where(first[:, None], contact.contact_normal, hit_normal),
            jnp.where(first, start + contact.hit_fraction * length, travelled),
            count + active.astype(jnp.int32),
        )

    (
        _,
        terminal_found,
        exhausted,
        hit,
        hit_point,
        hit_normal,
        travelled,
        count,
    ) = jax.lax.while_loop(
        continue_trace,
        trace_segment,
        (
            jnp.int32(0),
            jnp.zeros((batch,), dtype=jnp.bool_),
            jnp.zeros((batch,), dtype=jnp.bool_),
            jnp.zeros((batch,), dtype=jnp.bool_),
            jnp.zeros_like(points),
            jnp.zeros_like(points),
            jnp.zeros((batch,), dtype=jnp.float32),
            jnp.zeros((batch,), dtype=jnp.int32),
        ),
    )
    complete = ~terminal_found & ~invalid & ~capacity_exceeded
    available = ~invalid & ~capacity_exceeded & ~exhausted & (hit | complete)
    hit &= available
    return PointRayFirstContactResult(
        available=available,
        hit=hit,
        hit_point=jnp.where(hit[:, None], hit_point, 0.0),
        hit_normal=jnp.where(hit[:, None], hit_normal, 0.0),
        distance=jnp.where(hit, travelled, safe_distance),
        geometry_exhausted=exhausted,
        capacity_exceeded=capacity_exceeded,
        invalid=invalid,
        segment_count=count,
    )


def native_deployable_surface_allowed(
    result: PointRayFirstContactResult,
    allow_place_on_walls: Array | bool,
) -> Array:
    """Apply the exact native deployable surface predicate to a ray hit."""

    if not isinstance(result, PointRayFirstContactResult):
        raise TypeError("result must be PointRayFirstContactResult")
    allow_walls = jnp.asarray(allow_place_on_walls, dtype=jnp.bool_)
    if allow_walls.ndim == 0:
        allow_walls = jnp.broadcast_to(allow_walls, result.hit.shape)
    if allow_walls.shape != result.hit.shape:
        raise ValueError("allow_place_on_walls must be scalar or shape [batch]")
    normal = result.hit_normal
    # Mirrors SpawnDeployableFromRaycastInteraction.isSurface exactly. The
    # native predicate checks a vertical-axis normal rather than assuming the
    # sign, so this intentionally admits both +/-Y axis faces.
    native_surface = (
        (normal[:, 0] == jnp.float32(0.0))
        & (normal[:, 1] - jnp.float32(1.0) < jnp.float32(0.01))
        & (normal[:, 2] == jnp.float32(0.0))
    )
    return result.available & result.hit & (allow_walls | native_surface)


def point_ray_first_contact_contract() -> dict[str, object]:
    """Return the versioned producer contract pinned by Combat."""

    return {
        "schema": POINT_RAY_FIRST_CONTACT_SCHEMA,
        "version": POINT_RAY_FIRST_CONTACT_VERSION,
        "providers": ["GeometryState", "RegionGeometryState"],
        "shape": "zero_width_point_ray",
        "direction": "normalized_internally",
        "segment_length_blocks": 1,
        "default_segment_capacity": DEFAULT_POINT_RAY_SEGMENT_CAPACITY,
        "first_contact": "closed_collision_box",
        "contact_normal": "entry_slab_axis_opposing_ray",
        "geometry_exhausted": "fail_closed_before_contact",
        "capacity_exceeded": "fail_closed",
        "native_surface_predicate": (
            "allow_place_on_walls || (normal.x == 0 && "
            "normal.y - 1 < 0.01 && normal.z == 0)"
        ),
        "source": (
            "SpawnDeployableFromRaycastInteraction.firstRun/isSurface"
        ),
        "certification": {
            "source": "local_0_5_7_decompile",
            "live_native": "not_yet_native_differential_certified",
        },
    }


def point_ray_first_contact_contract_sha256() -> str:
    payload = json.dumps(
        point_ray_first_contact_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _point_segment_contact(
    geometry: GeometryProvider,
    origin: Array,
    displacement: Array,
):
    bounds = jnp.broadcast_to(_POINT_BOUNDS, (origin.shape[0], 6))
    if isinstance(geometry, GeometryState):
        return aabb_first_contact_result(
            geometry,
            origin,
            displacement,
            bounds,
            include_closed_endpoint=True,
        )
    return region_aabb_first_contact_result(
        geometry,
        origin,
        displacement,
        bounds,
        include_closed_endpoint=True,
    )


def _broadcast_distance(value: Array | float, batch: int) -> Array:
    distance = jnp.asarray(value, dtype=jnp.float32)
    if distance.ndim == 0:
        return jnp.broadcast_to(distance, (batch,))
    if distance.shape == (batch,):
        return distance
    if distance.shape == (batch, 1):
        return distance[:, 0]
    raise ValueError("maximum_distance must be scalar, [batch], or [batch,1]")


__all__ = [
    "DEFAULT_POINT_RAY_SEGMENT_CAPACITY",
    "POINT_RAY_FIRST_CONTACT_SCHEMA",
    "POINT_RAY_FIRST_CONTACT_VERSION",
    "PointRayFirstContactResult",
    "geometry_point_ray_first_contact_result",
    "native_deployable_surface_allowed",
    "point_ray_first_contact_contract",
    "point_ray_first_contact_contract_sha256",
]
