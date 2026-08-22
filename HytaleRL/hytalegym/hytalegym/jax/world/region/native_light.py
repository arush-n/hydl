"""Exact static native Region-light lookup with mutation invalidation."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.perception.lighting.tokens import (
    ActorBlockLightTokens,
    actor_block_light_query_cells,
    native_actor_block_light_tokens,
)
from hytalegym.jax.world.mutable_blocks import MutableBlockState
from hytalegym.jax.world.region.atlas import lookup_region_blocks
from hytalegym.jax.world.region.types import RegionGeometryState
from hytalegym.jax.world.surrogate.channels import (
    CHANNEL_BLOCK_LIGHT_RGB,
    CHANNEL_SKY_LIGHT,
    SURROGATE_PERCEPTION_CHANNEL_COUNT,
)
from hytalegym.jax.world.surrogate.types import SurrogatePerceptionChannelResult
from hytalegym.jax.world.tokens import WorldGeometryTokenObservation
from hytalegym.worldgen.region.contract import (
    CAPTURE_CHUNKS_PER_AXIS,
    CAPTURE_HALO_CHUNKS,
    CHUNK_SIZE,
    HEIGHT_SECTIONS,
    MIN_Y,
)
from hytalegym.worldgen.region.light_snapshot import NativeRegionLightSnapshot
from hytalegym.worldgen.region.native_light import (
    native_region_light_section_contract_sha256,
)


Array = jax.Array
REGION_NATIVE_LIGHT_ATLAS_SCHEMA = "hytalerl_region_native_light_atlas_v1"
REGION_NATIVE_LIGHT_ATLAS_VERSION = 1


class RegionNativeLightAtlas(NamedTuple):
    """Fixed native static-light sidecars in physical Region order."""

    region_mask: Array
    world_id: Array
    capture_min_chunk_xz: Array
    section_available: Array
    light_raw_yzx: Array


def region_native_light_atlas_from_snapshots(
    snapshots: Sequence[NativeRegionLightSnapshot],
    *,
    source_region_semantic_sha256: Sequence[str],
    world_ids: Sequence[int] | None = None,
) -> RegionNativeLightAtlas:
    """Load exact light sidecars without changing Region geometry identity."""

    values = tuple(snapshots)
    if not values:
        raise ValueError("at least one native Region light snapshot is required")
    if any(not isinstance(value, NativeRegionLightSnapshot) for value in values):
        raise TypeError("snapshots must contain NativeRegionLightSnapshot")
    return region_native_light_atlas_from_optional_snapshots(
        values,
        source_region_semantic_sha256=source_region_semantic_sha256,
        core_min_chunk_xz=tuple(
            tuple(int(axis) for axis in value.core_min_chunk_xz)
            for value in values
        ),
        world_ids=world_ids,
    )


def region_native_light_atlas_from_optional_snapshots(
    snapshots: Sequence[NativeRegionLightSnapshot | None],
    *,
    source_region_semantic_sha256: Sequence[str],
    core_min_chunk_xz: Sequence[tuple[int, int]],
    world_ids: Sequence[int] | None = None,
) -> RegionNativeLightAtlas:
    """Build a complete atlas whose missing sidecars remain unavailable."""

    values = tuple(snapshots)
    if not values:
        raise ValueError("at least one Region light slot is required")
    if any(
        value is not None and not isinstance(value, NativeRegionLightSnapshot)
        for value in values
    ):
        raise TypeError(
            "snapshots must contain NativeRegionLightSnapshot or None"
        )
    source = tuple(source_region_semantic_sha256)
    cores = tuple(core_min_chunk_xz)
    if len(source) != len(values) or len(cores) != len(values):
        raise ValueError("Region light slot metadata lengths differ")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
        for value in source
    ):
        raise ValueError("source Region identities must be SHA-256 strings")
    core = np.asarray(cores, dtype=np.int32)
    if core.shape != (len(values), 2):
        raise ValueError("core_min_chunk_xz must have shape [regions, 2]")
    for index, value in enumerate(values):
        if value is not None and (
            value.source_region_semantic_sha256 != source[index].lower()
            or not np.array_equal(value.core_min_chunk_xz, core[index])
        ):
            raise ValueError("native light sidecar differs from Region selection")
    ids = tuple(range(len(values))) if world_ids is None else tuple(world_ids)
    if len(ids) != len(values) or len(set(ids)) != len(ids):
        raise ValueError("world_ids must be unique and match snapshots")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in ids):
        raise TypeError("world_ids must contain integers")
    section_shape = (CAPTURE_CHUNKS_PER_AXIS**2, HEIGHT_SECTIONS)
    empty_sections = np.zeros(section_shape, dtype=np.bool_)
    empty_light = np.zeros(
        section_shape + (CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE),
        dtype=np.uint16,
    )
    return RegionNativeLightAtlas(
        region_mask=jnp.asarray(
            [value is not None for value in values],
            dtype=jnp.bool_,
        ),
        world_id=jnp.asarray(ids, dtype=jnp.int32),
        capture_min_chunk_xz=jnp.asarray(
            (core - np.int32(CAPTURE_HALO_CHUNKS)) * np.int32(CHUNK_SIZE),
            dtype=jnp.int32,
        ),
        section_available=jnp.asarray(
            np.stack(
                [
                    empty_sections
                    if value is None
                    else value.section_available
                    for value in values
                ]
            ),
            dtype=jnp.bool_,
        ),
        light_raw_yzx=jnp.asarray(
            np.stack(
                [
                    empty_light if value is None else value.light_raw_yzx
                    for value in values
                ]
            ),
            dtype=jnp.uint16,
        ),
    )


def query_region_native_light(
    light_atlas: RegionNativeLightAtlas,
    geometry: RegionGeometryState,
    block_positions: Array,
) -> SurrogatePerceptionChannelResult:
    """Read exact captured sky/RGB, invalidating all stale relighting."""

    if not isinstance(light_atlas, RegionNativeLightAtlas):
        raise TypeError("light_atlas has the wrong type")
    if not isinstance(geometry, RegionGeometryState):
        raise TypeError("geometry must be a RegionGeometryState")
    positions = jnp.asarray(block_positions)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("block_positions must have shape [batch, samples, 3]")
    if positions.dtype == jnp.bool_ or not jnp.issubdtype(
        positions.dtype,
        jnp.integer,
    ):
        raise TypeError("block_positions must have an integer dtype")
    positions = positions.astype(jnp.int32)
    batch, samples, _ = positions.shape
    selected = lookup_region_blocks(
        geometry.atlas,
        positions,
        geometry.environment_world_id,
        require_core=False,
    )
    safe_region = jnp.maximum(selected.region_index, 0)
    capture_min = light_atlas.capture_min_chunk_xz[safe_region]
    local_x = positions[..., 0] - capture_min[..., 0]
    local_z = positions[..., 2] - capture_min[..., 1]
    local_y = positions[..., 1] - jnp.int32(MIN_Y)
    extent = CAPTURE_CHUNKS_PER_AXIS * CHUNK_SIZE
    inside = (
        (local_x >= 0)
        & (local_x < extent)
        & (local_z >= 0)
        & (local_z < extent)
        & (local_y >= 0)
        & (local_y < HEIGHT_SECTIONS * CHUNK_SIZE)
    )
    safe_x = jnp.clip(local_x, 0, extent - 1)
    safe_z = jnp.clip(local_z, 0, extent - 1)
    safe_y = jnp.clip(local_y, 0, HEIGHT_SECTIONS * CHUNK_SIZE - 1)
    chunk_x = jnp.floor_divide(safe_x, CHUNK_SIZE)
    chunk_z = jnp.floor_divide(safe_z, CHUNK_SIZE)
    chunk_slot = chunk_x * CAPTURE_CHUNKS_PER_AXIS + chunk_z
    section_y = jnp.floor_divide(safe_y, CHUNK_SIZE)
    cell_y = safe_y & (CHUNK_SIZE - 1)
    cell_z = safe_z & (CHUNK_SIZE - 1)
    cell_x = safe_x & (CHUNK_SIZE - 1)
    identity = (
        light_atlas.region_mask[safe_region]
        & (
            light_atlas.world_id[safe_region]
            == geometry.environment_world_id[:, None]
        )
    )
    expected_capture_min = (
        geometry.atlas.core_min_chunk_xz[safe_region]
        - jnp.int32(CAPTURE_HALO_CHUNKS)
    ) * jnp.int32(CHUNK_SIZE)
    identity &= jnp.all(capture_min == expected_capture_min, axis=2)
    section_ready = light_atlas.section_available[
        safe_region,
        chunk_slot,
        section_y,
    ]
    mutation_clean = _native_light_mutation_clean(geometry)
    available = (
        selected.available
        & inside
        & identity
        & section_ready
        & mutation_clean[:, None]
    )
    raw = light_atlas.light_raw_yzx[
        safe_region,
        chunk_slot,
        section_y,
        cell_y,
        cell_z,
        cell_x,
    ]
    raw = jnp.where(available, raw, jnp.uint16(0))
    rgb = jnp.stack(
        tuple(
            ((raw >> jnp.uint16(4 * channel)) & jnp.uint16(0xF)).astype(
                jnp.uint8
            )
            for channel in range(3)
        ),
        axis=2,
    )
    sky = ((raw >> jnp.uint16(12)) & jnp.uint16(0xF)).astype(jnp.uint8)
    channel_valid = jnp.zeros(
        (batch, samples, SURROGATE_PERCEPTION_CHANNEL_COUNT),
        dtype=jnp.bool_,
    )
    channel_valid = channel_valid.at[..., CHANNEL_SKY_LIGHT].set(available)
    channel_valid = channel_valid.at[..., CHANNEL_BLOCK_LIGHT_RGB].set(
        available
    )
    return SurrogatePerceptionChannelResult(
        available=available,
        diagnostics=jnp.zeros((batch, samples), dtype=jnp.uint32),
        tile_index=jnp.where(available, selected.region_index, -1),
        channel_valid=channel_valid,
        heightmap_block_y=jnp.zeros((batch, samples), dtype=jnp.int16),
        height_above_surface=jnp.zeros((batch, samples), dtype=jnp.float32),
        sky_light=sky,
        block_light_rgb=rgb,
        environment_code=jnp.zeros((batch, samples), dtype=jnp.int32),
        tint_rgb=jnp.zeros((batch, samples, 3), dtype=jnp.uint8),
    )


def region_native_actor_block_light_tokens(
    light_atlas: RegionNativeLightAtlas,
    geometry: RegionGeometryState,
    tokens: WorldGeometryTokenObservation,
    actor_position: Array,
) -> ActorBlockLightTokens:
    """Publish exact captured native light aligned to actor-visible tokens."""

    cells = actor_block_light_query_cells(tokens, actor_position)
    batch, actors, capacity, _ = cells.shape
    channels = query_region_native_light(
        light_atlas,
        geometry,
        cells.reshape(batch, actors * capacity, 3),
    )
    return native_actor_block_light_tokens(
        tokens,
        actor_position,
        cells,
        channels,
    )


def region_native_light_atlas_contract() -> dict[str, object]:
    return {
        "schema": REGION_NATIVE_LIGHT_ATLAS_SCHEMA,
        "version": REGION_NATIVE_LIGHT_ATLAS_VERSION,
        "native_section_contract_sha256": (
            native_region_light_section_contract_sha256()
        ),
        "source": "frozen_native_region_light_snapshot",
        "channels": ["sky_light", "block_light_rgb"],
        "provenance": "native_static_capture",
        "geometry_identity": "paired_not_embedded",
        "mutation_policy": (
            "invalidate_entire_environment_on_any_geometry_override"
        ),
        "unavailable_policy": "invalid_not_darkness",
        "not_published": ["tint", "environment", "dynamic_relighting"],
    }


def region_native_light_atlas_contract_sha256() -> str:
    return hashlib.sha256(
        json.dumps(
            region_native_light_atlas_contract(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _native_light_mutation_clean(geometry: RegionGeometryState) -> Array:
    state = geometry.mutable_blocks
    if state is None:
        return jnp.ones(geometry.environment_world_id.shape, dtype=jnp.bool_)
    if not isinstance(state, MutableBlockState):
        raise TypeError("geometry.mutable_blocks must be MutableBlockState")
    synchronized = (
        state.synchronized
        & (state.world_id == geometry.environment_world_id)
    )
    changed = jnp.any(state.cell_mask & state.geometry_override, axis=1)
    return synchronized & ~changed


__all__ = [
    "REGION_NATIVE_LIGHT_ATLAS_SCHEMA",
    "REGION_NATIVE_LIGHT_ATLAS_VERSION",
    "RegionNativeLightAtlas",
    "query_region_native_light",
    "region_native_actor_block_light_tokens",
    "region_native_light_atlas_contract",
    "region_native_light_atlas_contract_sha256",
    "region_native_light_atlas_from_optional_snapshots",
    "region_native_light_atlas_from_snapshots",
]
