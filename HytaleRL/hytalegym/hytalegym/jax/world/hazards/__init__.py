"""Engine-grounded environmental hazard contact queries."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.geometry.contract import (
    CELL_COUNT,
    CELL_RADIUS,
    FLAG_DAMAGING,
    FLAG_FLUID,
)
from hytalegym.jax.world.types import GeometryState
from hytalegym.jax.world.region.types import RegionGeometryState
from hytalegym.jax.world.hazards.types import EnvironmentDamageContact
from hytalegym.jax.world.hazards.region import (
    region_environment_damage_contact,
)

Array = jax.Array

_CELL_OFFSETS = jnp.asarray(
    [
        (x, y, z)
        for x in range(-CELL_RADIUS, CELL_RADIUS + 1)
        for y in range(-CELL_RADIUS, CELL_RADIUS + 1)
        for z in range(-CELL_RADIUS, CELL_RADIUS + 1)
    ],
    dtype=jnp.float32,
)
_INTERSECTION_EPSILON = jnp.float32(1.0e-5)


def environment_damage_contact(
    geometry: GeometryState | RegionGeometryState,
    position: Array,
    *,
    bounds: Array | None = None,
) -> EnvironmentDamageContact:
    """Read per-step geometry damage for actor AABBs at ``position``.

    Native 0.5.7 tests exact block detail boxes and fluid fill boxes, then
    selects the maximum damage across all contacts. Missing cells inside the
    captured cube are air. Incomplete fluid shape data or a body reaching
    beyond the complete 9-cube fails closed instead of becoming zero damage.

    This World query admits contact only. Combat owns Environment-cause event
    creation, resistance bypass, health/death, and reward consequences.
    """

    if isinstance(geometry, RegionGeometryState):
        return region_environment_damage_contact(
            geometry,
            position,
            bounds=bounds,
        )
    if not isinstance(geometry, GeometryState):
        raise TypeError("geometry must be GeometryState or RegionGeometryState")

    position = jnp.asarray(position, dtype=jnp.float32)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position must have shape [batch, 3]")
    batch = position.shape[0]
    origin = _batch_rows(geometry.origin, batch, "geometry")
    actor_bounds = _batch_rows(
        geometry.agent_bounds if bounds is None else bounds,
        batch,
        "bounds",
    ).astype(jnp.float32)
    if actor_bounds.shape[1:] != (6,):
        raise ValueError("bounds must have shape [1|batch, 6]")

    cell_mask = _batch_rows(geometry.cell_mask, batch, "geometry")
    flags = _batch_rows(geometry.flags, batch, "geometry")
    block_damage = _batch_rows(geometry.block_damage, batch, "geometry")
    fluid_damage = _batch_rows(geometry.fluid_damage, batch, "geometry")
    if any(
        values.shape[1:] != (CELL_COUNT,)
        for values in (cell_mask, flags, block_damage, fluid_damage)
    ):
        raise ValueError("geometry cell arrays must use the fixed cell capacity")

    body_minimum = position + actor_bounds[:, :3]
    body_maximum = position + actor_bounds[:, 3:]
    cell_minimum = origin[:, None, :].astype(jnp.float32) + _CELL_OFFSETS
    cell_maximum = cell_minimum + jnp.float32(1.0)
    damaging = cell_mask.astype(jnp.bool_) & ((flags & jnp.int32(FLAG_DAMAGING)) != 0)

    full_cube_cell = _batch_rows(
        geometry.collision_full_cube_cell,
        batch,
        "geometry",
    )
    full_cube_damage = _block_box_damage(
        cell_indices=full_cube_cell,
        local_boxes=jnp.broadcast_to(
            jnp.asarray((0, 0, 0, 1, 1, 1), dtype=jnp.float32),
            full_cube_cell.shape + (6,),
        ),
        origin=origin,
        body_minimum=body_minimum,
        body_maximum=body_maximum,
        damaging=damaging,
        block_damage=block_damage,
        fluid_damage=fluid_damage,
    )
    exception_cell = _batch_rows(
        geometry.collision_exception_box_cell,
        batch,
        "geometry",
    )
    exception_damage = _block_box_damage(
        cell_indices=exception_cell,
        local_boxes=_batch_rows(
            geometry.collision_exception_boxes,
            batch,
            "geometry",
        ),
        origin=origin,
        body_minimum=body_minimum,
        body_maximum=body_maximum,
        damaging=damaging,
        block_damage=block_damage,
        fluid_damage=fluid_damage,
    )
    block_maximum = jnp.maximum(
        jnp.max(full_cube_damage, axis=1),
        jnp.max(exception_damage, axis=1),
    ).astype(jnp.int32)

    fluid_cell = damaging & ((flags & jnp.int32(FLAG_FLUID)) != 0)
    if geometry.fluid_fill_height is None:
        fill_height = jnp.full(cell_mask.shape, -1.0, dtype=jnp.float32)
    else:
        fill_height = _batch_rows(
            geometry.fluid_fill_height,
            batch,
            "geometry",
        ).astype(jnp.float32)
    broad_fluid_contact = (
        fluid_cell
        & (fluid_damage > 0)
        & _touches_boxes(
            body_minimum,
            body_maximum,
            cell_minimum,
            cell_maximum,
        )
    )
    fluid_shape_missing = broad_fluid_contact & (
        (fill_height < 0.0) | (fill_height > 1.0)
    )
    fluid_box_maximum = cell_maximum.at[..., 1].set(
        cell_minimum[..., 1] + jnp.clip(fill_height, 0.0, 1.0)
    )
    fluid_contact = (
        broad_fluid_contact
        & (fill_height >= 0.0)
        & _touches_boxes(
            body_minimum,
            body_maximum,
            cell_minimum,
            fluid_box_maximum,
        )
    )
    fluid_maximum = jnp.max(
        jnp.where(fluid_contact, jnp.maximum(fluid_damage, 0), 0),
        axis=1,
    ).astype(jnp.int32)

    frame_minimum = origin.astype(jnp.float32) - jnp.float32(CELL_RADIUS)
    frame_maximum = origin.astype(jnp.float32) + jnp.float32(CELL_RADIUS + 1)
    valid_body = (
        jnp.all(jnp.isfinite(position), axis=1)
        & jnp.all(jnp.isfinite(actor_bounds), axis=1)
        & jnp.all(actor_bounds[:, :3] <= actor_bounds[:, 3:], axis=1)
    )
    geometry_exhausted = (
        ~valid_body
        | jnp.any(
            (body_minimum - _INTERSECTION_EPSILON <= frame_minimum)
            | (body_maximum + _INTERSECTION_EPSILON >= frame_maximum),
            axis=1,
        )
        | jnp.any(fluid_shape_missing, axis=1)
    )
    block_maximum = jnp.where(geometry_exhausted, 0, block_maximum)
    fluid_maximum = jnp.where(geometry_exhausted, 0, fluid_maximum)
    amount = jnp.maximum(block_maximum, fluid_maximum).astype(jnp.float32)
    return EnvironmentDamageContact(
        requested=(amount > 0.0) & ~geometry_exhausted,
        amount=amount,
        block_damage=block_maximum,
        fluid_damage=fluid_maximum,
        geometry_exhausted=geometry_exhausted,
    )


def _block_box_damage(
    *,
    cell_indices: Array,
    local_boxes: Array,
    origin: Array,
    body_minimum: Array,
    body_maximum: Array,
    damaging: Array,
    block_damage: Array,
    fluid_damage: Array,
) -> Array:
    present = cell_indices >= 0
    safe_cell = jnp.clip(cell_indices, 0, CELL_COUNT - 1).astype(jnp.int32)
    cell_minimum = origin[:, None, :].astype(jnp.float32) + _CELL_OFFSETS[safe_cell]
    world_minimum = cell_minimum + local_boxes[..., :3]
    world_maximum = cell_minimum + local_boxes[..., 3:]
    valid = (
        present
        & _gather_cells(damaging, safe_cell)
        & _touches_boxes(
            body_minimum,
            body_maximum,
            world_minimum,
            world_maximum,
        )
    )
    # Native BlockData.getBlockDamage() is max(block field, fluid field).
    amount = jnp.maximum(
        _gather_cells(block_damage, safe_cell),
        _gather_cells(fluid_damage, safe_cell),
    )
    return jnp.where(valid, jnp.maximum(amount, 0), 0)


def _gather_cells(values: Array, indices: Array) -> Array:
    return jnp.take_along_axis(values, indices, axis=1)


def _touches_boxes(
    body_minimum: Array,
    body_maximum: Array,
    box_minimum: Array,
    box_maximum: Array,
) -> Array:
    return jnp.all(
        (body_minimum[:, None, :] <= box_maximum + _INTERSECTION_EPSILON)
        & (body_maximum[:, None, :] >= box_minimum - _INTERSECTION_EPSILON),
        axis=2,
    )


def _batch_rows(values: Array, batch: int, name: str) -> Array:
    values = jnp.asarray(values)
    if values.ndim < 1 or values.shape[0] not in (1, batch):
        raise ValueError(f"{name} batch must be one or match position batch")
    if values.shape[0] == batch:
        return values
    return jnp.broadcast_to(values, (batch,) + values.shape[1:])


__all__ = [
    "EnvironmentDamageContact",
    "environment_damage_contact",
]
