"""First solid-world contact for fixed-shape projectile sweeps."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.world.geometry import aabb_first_contact_result
from hytalegym.jax.world.region.geometry import (
    region_aabb_first_contact_result,
)
from hytalegym.jax.world.region.types import RegionGeometryState
from hytalegym.jax.world.types import GeometryState


Array = jax.Array
GeometryProvider = GeometryState | RegionGeometryState
PROJECTILE_FIRST_CONTACT_SCHEMA = "hytalerl_projectile_first_contact_v1"
PROJECTILE_FIRST_CONTACT_VERSION = 1
DEFAULT_PROJECTILE_CONTACT_SEGMENTS = 32
MAX_PROJECTILE_CONTACT_SEGMENTS = 64
_EPSILON = jnp.float32(1.0e-6)
_MAX_SEGMENT_CELL_SPAN = jnp.float32(2.0)


class ProjectileFirstContactResult(NamedTuple):
    """First world contact plus explicit fail-closed diagnostics."""

    clear: Array
    available: Array
    world_hit: Array
    hit_fraction: Array
    contact_point: Array
    contact_normal: Array
    geometry_exhausted: Array
    capacity_exceeded: Array
    invalid: Array
    segment_count: Array


def geometry_projectile_first_contact_result(
    geometry: GeometryProvider,
    position: Array,
    displacement: Array,
    bounds: Array,
    entity_contact_fraction: Array,
    entity_contact_mask: Array,
    *,
    segment_capacity: int = DEFAULT_PROJECTILE_CONTACT_SEGMENTS,
) -> ProjectileFirstContactResult:
    """Return the first solid-world contact before the nearest entity contact.

    Native projectile physics computes the nearest entity fraction first and
    bounds the block cast by it. The explicit entity inputs preserve that
    ordering while leaving entity collision ownership outside World.
    """

    if not isinstance(geometry, (GeometryState, RegionGeometryState)):
        raise TypeError("geometry must be GeometryState or RegionGeometryState")
    if (
        isinstance(segment_capacity, bool)
        or not isinstance(segment_capacity, int)
        or not 1 <= segment_capacity <= MAX_PROJECTILE_CONTACT_SEGMENTS
    ):
        raise ValueError(
            "segment_capacity must be an integer in "
            f"[1, {MAX_PROJECTILE_CONTACT_SEGMENTS}]"
        )
    points = jnp.asarray(position, dtype=jnp.float32)
    delta = jnp.asarray(displacement, dtype=jnp.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("position must have shape [batch, 3]")
    if delta.shape != points.shape:
        raise ValueError("displacement must match position")
    batch = points.shape[0]
    provider_bounds = _provider_bounds(geometry, batch)
    projectile_bounds = _broadcast_bounds(bounds, batch)
    entity_fraction = _batch_vector(
        entity_contact_fraction,
        batch,
        jnp.float32,
        "entity_contact_fraction",
    )
    entity_mask = _batch_vector(
        entity_contact_mask,
        batch,
        jnp.bool_,
        "entity_contact_mask",
    )
    width = projectile_bounds[:, 3:] - projectile_bounds[:, :3]
    finite = (
        jnp.all(jnp.isfinite(points) & jnp.isfinite(delta), axis=1)
        & jnp.all(jnp.isfinite(projectile_bounds), axis=1)
        & jnp.isfinite(entity_fraction)
    )
    valid_bounds = jnp.all(width > _EPSILON, axis=1)
    valid_entity = (entity_fraction >= 0.0) & (entity_fraction <= 1.0)
    moving = jnp.any(jnp.abs(delta) > _EPSILON, axis=1)
    valid = finite & valid_bounds & valid_entity & moving
    safe_points = jnp.where(valid[:, None], points, 0.0)
    safe_delta = jnp.where(valid[:, None], delta, 0.0)
    safe_bounds = jnp.where(
        valid[:, None],
        projectile_bounds,
        provider_bounds,
    )
    cast_fraction = jnp.where(
        entity_mask,
        jnp.clip(entity_fraction, 0.0, 1.0),
        jnp.float32(1.0),
    )
    cast_delta = safe_delta * cast_fraction[:, None]

    required = jnp.ones(batch, dtype=jnp.int32)
    capacity_exceeded = jnp.zeros(batch, dtype=jnp.bool_)
    if isinstance(geometry, GeometryState):
        contact = aabb_first_contact_result(
            geometry,
            safe_points,
            cast_delta,
            safe_bounds,
        )
        raw_hit = valid & contact.hit
        raw_fraction = cast_fraction * contact.hit_fraction
        raw_normal = contact.contact_normal
        exhausted = valid & contact.geometry_exhausted
    else:
        width = safe_bounds[:, 3:] - safe_bounds[:, :3]
        axis_moving = jnp.abs(cast_delta) > _EPSILON
        segment_limit = _MAX_SEGMENT_CELL_SPAN - width
        unsupported_axis = axis_moving & (segment_limit <= _EPSILON)
        ratio = jnp.where(
            axis_moving,
            jnp.abs(cast_delta) / jnp.maximum(segment_limit, _EPSILON),
            0.0,
        )
        required = jnp.maximum(
            jnp.ceil(jnp.max(ratio, axis=1)).astype(jnp.int32),
            1,
        )
        capacity_exceeded = (
            jnp.any(unsupported_axis, axis=1)
            | (required > jnp.int32(segment_capacity))
        ) & valid
        supported = valid & ~capacity_exceeded
        step = cast_delta / required[:, None].astype(jnp.float32)

        def sweep_segment(
            index: int,
            carry: tuple[Array, Array, Array, Array],
        ) -> tuple[Array, Array, Array, Array]:
            hit, fraction, normal, exhausted = carry
            active = supported & (index < required) & ~hit
            start = safe_points + step * jnp.float32(index)
            segment_delta = jnp.where(active[:, None], step, 0.0)
            contact = region_aabb_first_contact_result(
                geometry,
                start,
                segment_delta,
                safe_bounds,
            )
            candidate_hit = active & contact.hit
            candidate_fraction = cast_fraction * (
                (
                    jnp.float32(index)
                    + contact.hit_fraction
                )
                / required.astype(jnp.float32)
            )
            return (
                hit | candidate_hit,
                jnp.where(candidate_hit, candidate_fraction, fraction),
                jnp.where(
                    candidate_hit[:, None],
                    contact.contact_normal,
                    normal,
                ),
                exhausted | (active & contact.geometry_exhausted),
            )

        active_segments = jnp.max(
            jnp.where(supported, required, jnp.int32(0))
        )
        raw_hit, raw_fraction, raw_normal, exhausted = jax.lax.fori_loop(
            0,
            active_segments,
            sweep_segment,
            (
                jnp.zeros(batch, dtype=jnp.bool_),
                jnp.ones(batch, dtype=jnp.float32),
                jnp.zeros((batch, 3), dtype=jnp.float32),
                jnp.zeros(batch, dtype=jnp.bool_),
            ),
        )

    invalid = ~valid
    geometry_exhausted = exhausted | capacity_exceeded | invalid
    available = valid & ~capacity_exceeded & ~exhausted
    block_precedes_entity = (
        ~entity_mask | (raw_fraction < entity_fraction)
    )
    world_hit = available & raw_hit & block_precedes_entity
    hit_fraction = jnp.where(world_hit, raw_fraction, jnp.float32(1.0))
    contact_point = safe_points + safe_delta * raw_fraction[:, None]
    return ProjectileFirstContactResult(
        clear=available & ~world_hit,
        available=available,
        world_hit=world_hit,
        hit_fraction=hit_fraction,
        contact_point=jnp.where(world_hit[:, None], contact_point, 0.0),
        contact_normal=jnp.where(world_hit[:, None], raw_normal, 0.0),
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


def _batch_vector(
    value: Array,
    batch: int,
    dtype,
    label: str,
) -> Array:
    result = jnp.asarray(value, dtype=dtype)
    if result.ndim == 0:
        result = jnp.broadcast_to(result, (batch,))
    if result.shape != (batch,):
        raise ValueError(f"{label} must have shape [batch] or be scalar")
    return result


def projectile_first_contact_contract() -> dict[str, object]:
    """Return the checkpoint-pinnable projectile contact contract."""

    return {
        "schema": PROJECTILE_FIRST_CONTACT_SCHEMA,
        "version": PROJECTILE_FIRST_CONTACT_VERSION,
        "function": "geometry_projectile_first_contact_result",
        "providers": ["GeometryState", "RegionGeometryState"],
        "shape": {
            "position": ["batch", 3],
            "displacement": ["batch", 3],
            "projectile_bounds": ["batch_or_one", 6],
            "entity_contact_fraction": ["batch"],
            "entity_contact_mask": ["batch"],
        },
        "ordering": {
            "entity_contact": "computed_by_consumer_before_world_cast",
            "world_cast": "bounded_at_nearest_entity_contact_fraction",
            "equal_fraction": "entity_contact_wins",
        },
        "contact": {
            "fraction": "projectile_reference_position_fraction_in_[0,1]",
            "point": "position_plus_displacement_times_fraction",
            "normal": "first_maximum_entry_axis_X_then_Y_then_Z",
        },
        "segmentation": {
            "provider": "RegionGeometryState",
            "default_capacity": DEFAULT_PROJECTILE_CONTACT_SEGMENTS,
            "maximum_capacity": MAX_PROJECTILE_CONTACT_SEGMENTS,
            "per_axis_swept_interval_limit_blocks": 2.0,
            "first_contact_stops_later_segment_queries": True,
        },
        "fail_closed": [
            "non_finite_input",
            "inverted_or_degenerate_projectile_bounds",
            "zero_displacement",
            "entity_contact_fraction_outside_[0,1]",
            "provider_coverage_exhaustion_before_first_contact",
            "segment_capacity_exceeded",
            "projectile_extent_unsupported_for_moving_axis",
        ],
        "source_semantics": [
            "StandardPhysicsTickSystem_nearest_entity_then_bounded_block_cast",
            "MovingBoxBoxCollisionEvaluator_first_contact",
            "StandardPhysicsProvider_entity_contact_precedes_block_on_tie",
        ],
        "provenance": "exact_static_geometry_provider",
        "certification": {
            "array_behavior": "randomized_eager_and_jit",
            "native_runtime": "not_yet_native_differential_certified",
        },
        "not_modelled": [
            "entity_collision",
            "fluid_entry_or_exit",
            "dynamic_block_mutation",
            "bounce",
            "rolling",
            "slide",
            "trigger_or_damage_materials",
            "impact_chain_dispatch",
        ],
    }


def projectile_first_contact_contract_sha256() -> str:
    payload = json.dumps(
        projectile_first_contact_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "DEFAULT_PROJECTILE_CONTACT_SEGMENTS",
    "MAX_PROJECTILE_CONTACT_SEGMENTS",
    "PROJECTILE_FIRST_CONTACT_SCHEMA",
    "PROJECTILE_FIRST_CONTACT_VERSION",
    "ProjectileFirstContactResult",
    "geometry_projectile_first_contact_result",
    "projectile_first_contact_contract",
    "projectile_first_contact_contract_sha256",
]
