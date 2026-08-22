"""Exact environmental-damage contact over immutable or mutable Regions."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.geometry.contract import FLAG_DAMAGING, FLAG_FLUID
from hytalegym.jax.world.hazards.types import EnvironmentDamageContact
from hytalegym.jax.world.region.mutable_geometry import (
    lookup_region_geometry_blocks,
)
from hytalegym.jax.world.region.types import RegionGeometryState

Array = jax.Array
_CONTACT_AXIS_CAPACITY = 4
_CONTACT_OFFSETS = jnp.asarray(
    [
        (x, y, z)
        for x in range(_CONTACT_AXIS_CAPACITY)
        for y in range(_CONTACT_AXIS_CAPACITY)
        for z in range(_CONTACT_AXIS_CAPACITY)
    ],
    dtype=jnp.int32,
)
_INTERSECTION_EPSILON = jnp.float32(1.0e-5)
_FLOAT32_EXACT_INTEGER_LIMIT = jnp.int32(1 << 24)


def region_environment_damage_contact(
    geometry: RegionGeometryState,
    position: Array,
    *,
    bounds: Array | None = None,
) -> EnvironmentDamageContact:
    """Return maximum native hazard contact for one actor per Region row."""

    points = jnp.asarray(position, dtype=jnp.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("position must have shape [batch, 3]")
    batch = points.shape[0]
    if geometry.environment_world_id.shape != (batch,):
        raise ValueError("Region geometry batch must match position")
    actor_bounds = _broadcast_bounds(
        geometry.agent_bounds if bounds is None else bounds,
        batch,
    )

    body_minimum = points + actor_bounds[:, :3]
    body_maximum = points + actor_bounds[:, 3:]
    minimum_cell = jnp.floor(body_minimum - _INTERSECTION_EPSILON).astype(jnp.int32)
    maximum_cell = jnp.floor(body_maximum + _INTERSECTION_EPSILON).astype(jnp.int32)
    extent = maximum_cell - minimum_cell + jnp.int32(1)
    capacity_exceeded = jnp.any(
        extent > jnp.int32(_CONTACT_AXIS_CAPACITY),
        axis=1,
    )
    source_blocks = minimum_cell[:, None, :] + _CONTACT_OFFSETS[None, :, :]
    source_mask = jnp.all(
        source_blocks <= maximum_cell[:, None, :],
        axis=2,
    )
    selected = lookup_region_geometry_blocks(
        geometry,
        source_blocks,
        require_core=False,
    )

    damaging = (
        source_mask
        & selected.available
        & ((selected.flags & jnp.int32(FLAG_DAMAGING)) != 0)
    )
    root_blocks = source_blocks - selected.filler_root_offset
    block_minimum = (
        root_blocks[:, :, None, :].astype(jnp.float32)
        + selected.collision_boxes[..., :3]
    )
    block_maximum = (
        root_blocks[:, :, None, :].astype(jnp.float32)
        + selected.collision_boxes[..., 3:]
    )
    block_contact = (
        damaging[:, :, None]
        & selected.filler_root_available[:, :, None]
        & selected.collision_box_mask
        & _touches_boxes(
            body_minimum,
            body_maximum,
            block_minimum.reshape(batch, -1, 3),
            block_maximum.reshape(batch, -1, 3),
        ).reshape(selected.collision_box_mask.shape)
    )
    block_amount = jnp.maximum(
        selected.block_damage,
        selected.fluid_damage,
    )
    block_damage = jnp.max(
        jnp.where(
            block_contact,
            jnp.maximum(block_amount[:, :, None], 0),
            0,
        ),
        axis=(1, 2),
    ).astype(jnp.int32)

    fluid_candidate = (
        damaging
        & ((selected.flags & jnp.int32(FLAG_FLUID)) != 0)
        & (selected.fluid_damage > 0)
    )
    fill_height = selected.fluid_fill_height.astype(jnp.float32)
    broad_fluid_contact = fluid_candidate & _touches_boxes(
        body_minimum,
        body_maximum,
        source_blocks.astype(jnp.float32),
        source_blocks.astype(jnp.float32) + jnp.float32(1.0),
    )
    fluid_shape_missing = broad_fluid_contact & (
        (fill_height < 0.0) | (fill_height > 1.0)
    )
    fluid_minimum = source_blocks.astype(jnp.float32)
    fluid_maximum = fluid_minimum + jnp.float32(1.0)
    fluid_maximum = fluid_maximum.at[..., 1].set(
        fluid_minimum[..., 1] + jnp.clip(fill_height, 0.0, 1.0)
    )
    fluid_contact = (
        broad_fluid_contact
        & (fill_height >= 0.0)
        & _touches_boxes(
            body_minimum,
            body_maximum,
            fluid_minimum,
            fluid_maximum,
        )
    )
    fluid_damage = jnp.max(
        jnp.where(
            fluid_contact,
            jnp.maximum(selected.fluid_damage, 0),
            0,
        ),
        axis=1,
    ).astype(jnp.int32)

    valid_body = (
        jnp.all(jnp.isfinite(points), axis=1)
        & jnp.all(jnp.isfinite(actor_bounds), axis=1)
        & jnp.all(actor_bounds[:, :3] <= actor_bounds[:, 3:], axis=1)
    )
    precise = jnp.all(
        (source_blocks > -_FLOAT32_EXACT_INTEGER_LIMIT)
        & (source_blocks < _FLOAT32_EXACT_INTEGER_LIMIT),
        axis=(1, 2),
    )
    geometry_exhausted = (
        ~valid_body
        | ~precise
        | capacity_exceeded
        | jnp.any(source_mask & ~selected.available, axis=1)
        | jnp.any(fluid_shape_missing, axis=1)
    )
    block_damage = jnp.where(geometry_exhausted, 0, block_damage)
    fluid_damage = jnp.where(geometry_exhausted, 0, fluid_damage)
    amount = jnp.maximum(block_damage, fluid_damage).astype(jnp.float32)
    return EnvironmentDamageContact(
        requested=(amount > 0.0) & ~geometry_exhausted,
        amount=amount,
        block_damage=block_damage,
        fluid_damage=fluid_damage,
        geometry_exhausted=geometry_exhausted,
    )


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


def _broadcast_bounds(bounds: Array, batch: int) -> Array:
    values = jnp.asarray(bounds, dtype=jnp.float32)
    if values.ndim == 1 and values.shape == (6,):
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("bounds must have shape [1|batch, 6]")
    if values.shape[0] not in (1, batch):
        raise ValueError("bounds batch must be one or match position")
    if values.shape[0] == 1 and batch != 1:
        values = jnp.broadcast_to(values, (batch, 6))
    return values


__all__ = ["region_environment_damage_contact"]
