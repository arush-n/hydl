"""Actor-legal fluid and ledge sensing over exact geometry providers."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.geometry.contract import FLAG_FLUID
from hytalegym.jax.world.geometry import (
    CellEnvironmentResult,
    cell_environment_result,
    resolve_aabb_motion,
)
from hytalegym.jax.world.region.geometry import (
    region_cell_environment_result,
    region_resolve_aabb_motion,
)
from hytalegym.jax.world.region.types import RegionGeometryState
from hytalegym.jax.world.types import GeometryState


Array = jax.Array
GeometryProvider = GeometryState | RegionGeometryState
ACTOR_WORLD_STATE_SCHEMA = "hytalerl_actor_world_state_v1"
ACTOR_WORLD_STATE_VERSION = 1
DEFAULT_DROP_SEGMENT_CAPACITY = 32
MAX_DROP_SEGMENT_CAPACITY = 64
_EPSILON = jnp.float32(1.0e-6)
_REGION_SEGMENT_CELL_SPAN = jnp.float32(2.0)


class ActorWorldStateResult(NamedTuple):
    """Observable body state and explicit per-component validity."""

    available: Array
    controller_medium_available: Array
    submersion_available: Array
    drop_available: Array
    controller_in_fluid: Array
    feet_submerged: Array
    eyes_submerged: Array
    drop_support_found: Array
    drop_height: Array
    geometry_exhausted: Array
    capacity_exceeded: Array
    invalid: Array
    drop_segment_count: Array


def geometry_actor_world_state_result(
    geometry: GeometryProvider,
    position: Array,
    eye_position: Array,
    bounds: Array | None = None,
    *,
    maximum_drop_distance: Array | float = 4.0,
    drop_segment_capacity: int = DEFAULT_DROP_SEGMENT_CAPACITY,
) -> ActorWorldStateResult:
    """Sense native fluid state and first downward AABB support.

    ``controller_in_fluid`` matches the walk controller's feet-cell sample.
    The two submersion fields additionally respect the native fluid fill
    surface. Drop height is the exact continuous collision-start distance,
    bounded by ``maximum_drop_distance``. Missing provider coverage, legacy
    fluid fill data, invalid input, and insufficient Region segmentation are
    never reinterpreted as clear or dry.
    """

    if not isinstance(geometry, (GeometryState, RegionGeometryState)):
        raise TypeError("geometry must be GeometryState or RegionGeometryState")
    if (
        isinstance(drop_segment_capacity, bool)
        or not isinstance(drop_segment_capacity, int)
        or not 1 <= drop_segment_capacity <= MAX_DROP_SEGMENT_CAPACITY
    ):
        raise ValueError(
            "drop_segment_capacity must be an integer in "
            f"[1, {MAX_DROP_SEGMENT_CAPACITY}]"
        )
    points = jnp.asarray(position, dtype=jnp.float32)
    eyes = jnp.asarray(eye_position, dtype=jnp.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("position must have shape [batch, 3]")
    if eyes.shape != points.shape:
        raise ValueError("eye_position must match position")
    batch = points.shape[0]
    entity_bounds = (
        _provider_bounds(geometry, batch)
        if bounds is None
        else _broadcast_bounds(bounds, batch)
    )
    maximum = _batch_scalar(
        maximum_drop_distance,
        batch,
        "maximum_drop_distance",
    )
    extent = entity_bounds[:, 3:] - entity_bounds[:, :3]
    valid = (
        jnp.all(jnp.isfinite(points) & jnp.isfinite(eyes), axis=1)
        & jnp.all(jnp.isfinite(entity_bounds), axis=1)
        & jnp.all(extent > _EPSILON, axis=1)
        & jnp.isfinite(maximum)
        & (maximum > 0.0)
    )
    safe_points = jnp.where(valid[:, None], points, 0.0)
    safe_eyes = jnp.where(valid[:, None], eyes, 0.0)
    safe_bounds = jnp.where(
        valid[:, None],
        entity_bounds,
        _provider_bounds(geometry, batch),
    )
    safe_maximum = jnp.where(valid, maximum, jnp.float32(1.0))
    feet = safe_points.at[:, 1].add(safe_bounds[:, 1])

    feet_cell = _cell_result(geometry, feet)
    eye_cell = _cell_result(geometry, safe_eyes)
    feet_fluid = (
        (feet_cell.flags & jnp.int32(FLAG_FLUID)) != 0
    )
    controller_available = valid & ~feet_cell.geometry_exhausted
    submersion_available = (
        controller_available
        & ~eye_cell.geometry_exhausted
        & _fill_available(feet_cell, feet_fluid)
        & _fill_available(
            eye_cell,
            (eye_cell.flags & jnp.int32(FLAG_FLUID)) != 0,
        )
    )
    controller_in_fluid = controller_available & feet_fluid
    feet_submerged = submersion_available & _point_submerged(
        feet_cell,
        feet,
    )
    eyes_submerged = submersion_available & _point_submerged(
        eye_cell,
        safe_eyes,
    )

    (
        drop_support,
        drop_distance,
        drop_exhausted,
        capacity_exceeded,
        segment_count,
    ) = _drop_result(
        geometry,
        safe_points,
        safe_bounds,
        safe_maximum,
        valid,
        drop_segment_capacity,
    )
    drop_available = valid & ~drop_exhausted & ~capacity_exceeded
    invalid = ~valid
    geometry_exhausted = (
        invalid
        | feet_cell.geometry_exhausted
        | eye_cell.geometry_exhausted
        | drop_exhausted
        | capacity_exceeded
    )
    available = (
        controller_available
        & submersion_available
        & drop_available
    )
    return ActorWorldStateResult(
        available=available,
        controller_medium_available=controller_available,
        submersion_available=submersion_available,
        drop_available=drop_available,
        controller_in_fluid=controller_in_fluid,
        feet_submerged=feet_submerged,
        eyes_submerged=eyes_submerged,
        drop_support_found=drop_available & drop_support,
        drop_height=jnp.where(
            drop_available,
            drop_distance,
            jnp.where(
                jnp.isfinite(maximum) & (maximum > 0.0),
                maximum,
                jnp.float32(0.0),
            ),
        ),
        geometry_exhausted=geometry_exhausted,
        capacity_exceeded=capacity_exceeded,
        invalid=invalid,
        drop_segment_count=jnp.where(valid, segment_count, 0),
    )


def _cell_result(
    geometry: GeometryProvider,
    point: Array,
) -> CellEnvironmentResult:
    if isinstance(geometry, GeometryState):
        return cell_environment_result(geometry, point)
    return region_cell_environment_result(geometry, point)


def _fill_available(
    cell: CellEnvironmentResult,
    is_fluid: Array,
) -> Array:
    return (~is_fluid) | (
        (cell.fluid_fill_height >= 0.0)
        & (cell.fluid_fill_height <= 1.0)
    )


def _point_submerged(
    cell: CellEnvironmentResult,
    point: Array,
) -> Array:
    is_fluid = (cell.flags & jnp.int32(FLAG_FLUID)) != 0
    relative_y = point[:, 1] - jnp.floor(point[:, 1])
    return (
        ~cell.geometry_exhausted
        & is_fluid
        & (relative_y <= cell.fluid_fill_height)
    )


def _drop_result(
    geometry: GeometryProvider,
    position: Array,
    bounds: Array,
    maximum: Array,
    valid: Array,
    segment_capacity: int,
) -> tuple[Array, Array, Array, Array, Array]:
    batch = position.shape[0]
    if isinstance(geometry, GeometryState):
        displacement = jnp.zeros_like(position).at[:, 1].set(-maximum)
        result = resolve_aabb_motion(
            geometry,
            position,
            displacement,
            bounds,
            jnp.ones(batch, dtype=jnp.bool_),
            skin_distance=0.0,
        )
        support = valid & result.grounded
        return (
            support,
            jnp.where(
                support,
                jnp.maximum(0.0, -result.applied_displacement[:, 1]),
                maximum,
            ),
            valid & result.geometry_exhausted,
            jnp.zeros(batch, dtype=jnp.bool_),
            jnp.ones(batch, dtype=jnp.int32),
        )

    height = bounds[:, 4] - bounds[:, 1]
    segment_limit = _REGION_SEGMENT_CELL_SPAN - height
    unsupported = segment_limit <= _EPSILON
    required = jnp.maximum(
        jnp.ceil(
            maximum / jnp.maximum(segment_limit, _EPSILON)
        ).astype(jnp.int32),
        1,
    )
    capacity_exceeded = valid & (
        unsupported | (required > jnp.int32(segment_capacity))
    )
    supported = valid & ~capacity_exceeded
    step_distance = maximum / required.astype(jnp.float32)

    def sweep_segment(
        index: int,
        carry: tuple[Array, Array, Array],
    ) -> tuple[Array, Array, Array]:
        found, distance, exhausted = carry
        active = supported & ~found & (index < required)
        start = position.at[:, 1].add(
            -step_distance * jnp.float32(index)
        )
        displacement = jnp.zeros_like(position).at[:, 1].set(
            jnp.where(active, -step_distance, 0.0)
        )
        result = region_resolve_aabb_motion(
            geometry,
            start,
            displacement,
            bounds,
            jnp.ones(batch, dtype=jnp.bool_),
            skin_distance=0.0,
        )
        hit = active & result.grounded
        hit_distance = (
            step_distance * jnp.float32(index)
            - result.applied_displacement[:, 1]
        )
        return (
            found | hit,
            jnp.where(hit & ~found, hit_distance, distance),
            exhausted | (active & result.geometry_exhausted),
        )

    active_segments = jnp.max(
        jnp.where(supported, required, jnp.int32(0))
    )
    found, distance, exhausted = jax.lax.fori_loop(
        0,
        active_segments,
        sweep_segment,
        (
            jnp.zeros(batch, dtype=jnp.bool_),
            maximum,
            jnp.zeros(batch, dtype=jnp.bool_),
        ),
    )
    return found, distance, exhausted, capacity_exceeded, required


def _provider_bounds(
    geometry: GeometryProvider,
    batch: int,
) -> Array:
    if isinstance(geometry, RegionGeometryState):
        if geometry.environment_world_id.shape != (batch,):
            raise ValueError(
                "Region geometry world IDs must match the query batch"
            )
    elif geometry.origin.shape[0] not in (1, batch):
        raise ValueError(
            "GeometryState batch must be one or match the query"
        )
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


def _batch_scalar(
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


def actor_world_state_contract() -> dict[str, object]:
    """Return the checkpoint-pinnable actor body-state contract."""

    return {
        "schema": ACTOR_WORLD_STATE_SCHEMA,
        "version": ACTOR_WORLD_STATE_VERSION,
        "function": "geometry_actor_world_state_result",
        "providers": ["GeometryState", "RegionGeometryState"],
        "inputs": {
            "position": ["batch", 3],
            "eye_position": ["batch", 3],
            "bounds": ["batch_or_one", 6],
            "maximum_drop_distance": "scalar_or_batch",
            "drop_segment_capacity": "static_integer",
        },
        "outputs": {
            "shape": ["batch"],
            "availability": [
                "available",
                "controller_medium_available",
                "submersion_available",
                "drop_available",
            ],
            "policy_values": [
                "controller_in_fluid",
                "feet_submerged",
                "eyes_submerged",
                "drop_support_found",
                "drop_height",
            ],
            "diagnostics": [
                "geometry_exhausted",
                "capacity_exceeded",
                "invalid",
                "drop_segment_count",
            ],
            "unavailable_values": (
                "booleans false; finite drop bound, or zero for invalid bound"
            ),
        },
        "fluid": {
            "controller_in_fluid": (
                "MotionControllerWalk feet-cell FluidId predicate"
            ),
            "feet_submerged": (
                "feet point at or below native level/MaxFluidLevel surface"
            ),
            "eyes_submerged": (
                "eye point at or below native level/MaxFluidLevel surface"
            ),
            "boundary": "surface equality is submerged",
            "legacy_region": (
                "raw level without MaxFluidLevel fails submersion closed"
            ),
        },
        "drop": {
            "predicate": (
                "first continuous downward entity-AABB collision start"
            ),
            "no_support_within_bound": (
                "support_found false and height equals query bound"
            ),
            "simulation_skin_distance": 0.0,
            "region_segmentation": {
                "default_capacity": DEFAULT_DROP_SEGMENT_CAPACITY,
                "maximum_capacity": MAX_DROP_SEGMENT_CAPACITY,
                "per_segment_swept_cell_span": 2.0,
            },
        },
        "native_sources": [
            "MotionControllerWalk.postReadPosition",
            "WorldUtil.getPackedMaterialAndFluidAtPosition",
            "MotionControllerWalk.findDropBlockCollision",
            "CollisionModule.findCollisions",
        ],
        "fail_closed": [
            "non_finite_input",
            "degenerate_or_inverted_bounds",
            "non_positive_drop_bound",
            "provider_coverage_exhaustion",
            "fluid_fill_height_unavailable",
            "region_segment_capacity_exceeded",
            "entity_height_unsupported_by_region_segment",
        ],
        "provenance": "native_exact_geometry_semantics",
    }


def actor_world_state_contract_sha256() -> str:
    payload = json.dumps(
        actor_world_state_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ACTOR_WORLD_STATE_SCHEMA",
    "ACTOR_WORLD_STATE_VERSION",
    "ActorWorldStateResult",
    "DEFAULT_DROP_SEGMENT_CAPACITY",
    "MAX_DROP_SEGMENT_CAPACITY",
    "actor_world_state_contract",
    "actor_world_state_contract_sha256",
    "geometry_actor_world_state_result",
]
