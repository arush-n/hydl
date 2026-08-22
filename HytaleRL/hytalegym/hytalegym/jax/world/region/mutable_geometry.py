"""Sparse per-environment mutations over immutable Region geometry."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.mutable_blocks import MutableBlockState
from hytalegym.jax.world.region.atlas import (
    lookup_region_blocks,
    lookup_region_cells,
)
from hytalegym.jax.world.region.types import (
    RegionCellSelection,
    RegionGeometryState,
)


Array = jax.Array


def bind_region_mutable_blocks(
    geometry: RegionGeometryState,
    state: MutableBlockState,
) -> RegionGeometryState:
    """Bind one synchronized sparse overlay row to each environment."""

    if not isinstance(geometry, RegionGeometryState):
        raise TypeError("geometry must be a RegionGeometryState")
    if not isinstance(state, MutableBlockState):
        raise TypeError("state must be a MutableBlockState")
    batch = geometry.environment_world_id.shape[0]
    if state.world_id.shape != (batch,):
        raise ValueError("mutable state batch must match Region geometry")
    synchronized = np.asarray(jax.device_get(state.synchronized))
    state_world = np.asarray(jax.device_get(state.world_id))
    geometry_world = np.asarray(
        jax.device_get(geometry.environment_world_id)
    )
    if not np.all(synchronized):
        raise ValueError("mutable state must be synchronized before binding")
    if not np.array_equal(state_world, geometry_world):
        raise ValueError("mutable state world IDs differ from Region geometry")
    return geometry._replace(mutable_blocks=state)


def lookup_region_geometry_cells(
    geometry: RegionGeometryState,
    positions: Array,
    *,
    require_core: bool = True,
) -> RegionCellSelection:
    """Look up float positions through the current sparse physical state."""

    points = jnp.asarray(positions, dtype=jnp.float32)
    base = lookup_region_cells(
        geometry.atlas,
        points,
        geometry.environment_world_id,
        require_core=require_core,
    )
    return _overlay_selection(
        geometry,
        jnp.floor(points).astype(jnp.int32),
        base,
    )


def lookup_region_geometry_blocks(
    geometry: RegionGeometryState,
    block_positions: Array,
    *,
    require_core: bool = True,
) -> RegionCellSelection:
    """Look up integer cells through the current sparse physical state."""

    positions = jnp.asarray(block_positions, dtype=jnp.int32)
    base = lookup_region_blocks(
        geometry.atlas,
        positions,
        geometry.environment_world_id,
        require_core=require_core,
    )
    return _overlay_selection(geometry, positions, base)


def region_geometry_has_mutations(geometry: RegionGeometryState) -> Array:
    """Return one flag per environment without treating health as geometry."""

    state = geometry.mutable_blocks
    if state is None:
        return jnp.zeros(
            geometry.environment_world_id.shape,
            dtype=jnp.bool_,
        )
    if not isinstance(state, MutableBlockState):
        raise TypeError("geometry.mutable_blocks must be MutableBlockState")
    return (
        state.synchronized
        & (state.world_id == geometry.environment_world_id)
        & jnp.any(state.cell_mask & state.geometry_override, axis=1)
    )


def _overlay_selection(
    geometry: RegionGeometryState,
    positions: Array,
    base: RegionCellSelection,
) -> RegionCellSelection:
    state = geometry.mutable_blocks
    if state is None:
        return base
    if not isinstance(state, MutableBlockState):
        raise TypeError("geometry.mutable_blocks must be MutableBlockState")
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [batch, query, 3]")
    batch, queries, _ = positions.shape
    if state.world_id.shape != (batch,):
        raise ValueError("mutable state batch must match Region query")

    active = state.cell_mask & state.geometry_override
    matches = active[:, None, :] & jnp.all(
        state.cell_position[:, None, :, :] == positions[:, :, None, :],
        axis=3,
    )
    match_count = jnp.sum(matches, axis=2)
    duplicate = match_count > 1
    overridden = match_count == 1
    slot = jnp.argmax(matches, axis=2)

    def gather(value: Array) -> Array:
        return jax.vmap(lambda row, index: row[index])(value, slot)

    current = jax.tree.map(gather, state.geometry)
    identity = state.synchronized & (
        state.world_id == geometry.environment_world_id
    )
    exact_override = overridden & current.exact
    available = (
        base.available
        & identity[:, None]
        & ~duplicate
        & (~overridden | exact_override)
    )

    def select(value: Array, fallback: Array) -> Array:
        gate = exact_override.reshape(
            exact_override.shape
            + (1,) * (value.ndim - exact_override.ndim)
        )
        selected = jnp.where(gate, value, fallback)
        valid = available.reshape(
            available.shape
            + (1,) * (selected.ndim - available.ndim)
        )
        return jnp.where(valid, selected, jnp.zeros_like(selected))

    return RegionCellSelection(
        available=available,
        region_index=jnp.where(available, base.region_index, -1),
        # Palette ordinals describe the immutable base only. Overrides carry
        # complete values directly and intentionally publish ordinal zero.
        cell_code=jnp.where(
            available,
            jnp.where(exact_override, -1, base.cell_code),
            0,
        ),
        flags=select(current.flags.astype(base.flags.dtype), base.flags),
        shape_index=jnp.where(
            available & ~exact_override,
            base.shape_index,
            0,
        ),
        # Mutable geometry currently certifies direct-cell outcomes. A
        # multi-cell/filler mutation must update every affected cell.
        filler_root_available=jnp.where(
            available,
            exact_override | base.filler_root_available,
            False,
        ),
        filler_root_offset=jnp.where(
            exact_override[..., None],
            0,
            jnp.where(
                available[..., None],
                base.filler_root_offset,
                0,
            ),
        ),
        fluid_level=select(
            current.fluid_level.astype(base.fluid_level.dtype),
            base.fluid_level,
        ),
        fluid_fill_height=select(
            current.fluid_fill_height,
            base.fluid_fill_height,
        ),
        support=select(current.support, base.support),
        block_damage=select(current.block_damage, base.block_damage),
        fluid_damage=select(current.fluid_damage, base.fluid_damage),
        movement=select(current.movement, base.movement),
        fluid_movement=select(
            current.fluid_movement,
            base.fluid_movement,
        ),
        collision_boxes=select(
            current.collision_boxes,
            base.collision_boxes,
        ),
        collision_box_mask=select(
            current.collision_box_mask,
            base.collision_box_mask,
        ),
    )


__all__ = [
    "bind_region_mutable_blocks",
    "lookup_region_geometry_blocks",
    "lookup_region_geometry_cells",
    "region_geometry_has_mutations",
]
