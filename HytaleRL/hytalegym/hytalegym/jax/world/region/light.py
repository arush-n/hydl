"""Region-backed surrogate sky light with mutable-column refresh."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.geometry.contract import FLAG_OPAQUE
from hytalegym.jax.world.perception.lighting.tokens import (
    ActorBlockLightTokens,
    actor_block_light_query_cells,
    actor_block_light_token_contract_sha256,
    surrogate_actor_block_light_tokens,
)
from hytalegym.jax.world.mutable_blocks import MutableBlockState
from hytalegym.jax.world.region.atlas import lookup_region_blocks
from hytalegym.jax.world.region.types import RegionAtlas, RegionGeometryState
from hytalegym.jax.world.surrogate.channels import (
    CHANNEL_HEIGHTMAP,
    CHANNEL_SKY_LIGHT,
    SURROGATE_PERCEPTION_CHANNEL_COUNT,
    surrogate_sky_light_from_height,
)
from hytalegym.jax.world.surrogate.types import (
    SurrogatePerceptionChannelResult,
)
from hytalegym.jax.world.tokens import WorldGeometryTokenObservation
from hytalegym.worldgen.region import (
    CAPTURE_CHUNK_COUNT,
    CAPTURE_CHUNKS_PER_AXIS,
    CAPTURE_HALO_CHUNKS,
    CHUNK_SIZE,
    HEIGHT_SECTIONS,
    MIN_Y,
    SECTION_VOLUME,
)


Array = jax.Array
REGION_SURROGATE_SKY_LIGHT_SCHEMA = (
    "hytalerl_region_surrogate_sky_light_v1"
)
REGION_SURROGATE_SKY_LIGHT_VERSION = 1
REGION_CAPTURE_BLOCKS_PER_AXIS = CAPTURE_CHUNKS_PER_AXIS * CHUNK_SIZE
_BLOCK_CHUNK_CLASS = (
    "com.hypixel.hytale.server.core.universe.world.chunk.BlockChunk"
)
_BLOCK_CHUNK_CLASS_SHA256 = (
    "ede8ac0297812ffa000a2cbf2bec081313bb364ee625b0dfed7bfc0953f339a6"
)


class RegionSurrogateSkyLightAtlas(NamedTuple):
    """Packed native-height predicate inputs for every captured column."""

    region_mask: Array
    world_id: Array
    capture_min_block_xz: Array
    column_known: Array
    opaque_y_words: Array


def region_surrogate_sky_light_atlas_from_region(
    atlas: RegionAtlas,
) -> RegionSurrogateSkyLightAtlas:
    """Compile exact Region opacity into ten uint32 words per X/Z column."""

    if not isinstance(atlas, RegionAtlas):
        raise TypeError("atlas must be a RegionAtlas")
    region_mask = np.asarray(jax.device_get(atlas.region_mask), dtype=np.bool_)
    world_id = np.asarray(jax.device_get(atlas.world_id), dtype=np.int32)
    core_min = np.asarray(
        jax.device_get(atlas.core_min_chunk_xz),
        dtype=np.int32,
    )
    known = np.asarray(jax.device_get(atlas.section_known), dtype=np.bool_)
    code = np.asarray(jax.device_get(atlas.cell_code), dtype=np.int32)
    flags = np.asarray(jax.device_get(atlas.cell_flags), dtype=np.int32)
    regions = region_mask.shape[0]
    if world_id.shape != (regions,) or core_min.shape != (regions, 2):
        raise ValueError("Region atlas identity leaves have invalid shapes")
    if known.shape != (
        regions,
        CAPTURE_CHUNK_COUNT,
        HEIGHT_SECTIONS,
    ):
        raise ValueError("Region atlas section coverage has an invalid shape")
    if code.shape != (
        regions,
        CAPTURE_CHUNK_COUNT,
        HEIGHT_SECTIONS,
        SECTION_VOLUME,
    ):
        raise ValueError("Region atlas cell codes have an invalid shape")
    if flags.ndim != 2 or flags.shape[0] != regions:
        raise ValueError("Region atlas flag palette has an invalid shape")
    if np.any(code < 0) or np.any(code >= flags.shape[1]):
        raise ValueError("Region atlas contains an out-of-range cell code")

    # Native cell indexing is local Y, Z, X; chunk slots are capture X, Z.
    shaped = code.reshape(
        regions,
        CAPTURE_CHUNKS_PER_AXIS,
        CAPTURE_CHUNKS_PER_AXIS,
        HEIGHT_SECTIONS,
        CHUNK_SIZE,
        CHUNK_SIZE,
        CHUNK_SIZE,
    )
    palette_flags = np.take_along_axis(
        flags[:, None, None, None, None, None, :],
        shaped,
        axis=6,
    )
    opaque = (palette_flags & FLAG_OPAQUE) != 0
    opaque = opaque.transpose(0, 1, 6, 2, 5, 3, 4).reshape(
        regions,
        REGION_CAPTURE_BLOCKS_PER_AXIS,
        REGION_CAPTURE_BLOCKS_PER_AXIS,
        HEIGHT_SECTIONS,
        CHUNK_SIZE,
    )
    bit_weight = np.left_shift(
        np.uint32(1),
        np.arange(CHUNK_SIZE, dtype=np.uint32),
    )
    words = np.sum(
        opaque.astype(np.uint32) * bit_weight,
        axis=-1,
        dtype=np.uint32,
    )
    chunk_known = np.all(
        known.reshape(
            regions,
            CAPTURE_CHUNKS_PER_AXIS,
            CAPTURE_CHUNKS_PER_AXIS,
            HEIGHT_SECTIONS,
        ),
        axis=3,
    )
    column_known = np.repeat(
        np.repeat(chunk_known, CHUNK_SIZE, axis=1),
        CHUNK_SIZE,
        axis=2,
    )
    capture_min = (
        core_min - np.int32(CAPTURE_HALO_CHUNKS)
    ) * np.int32(CHUNK_SIZE)
    return RegionSurrogateSkyLightAtlas(
        region_mask=jnp.asarray(region_mask),
        world_id=jnp.asarray(world_id),
        capture_min_block_xz=jnp.asarray(capture_min),
        column_known=jnp.asarray(column_known),
        opaque_y_words=jnp.asarray(words),
    )


def region_surrogate_actor_block_light_tokens(
    light_atlas: RegionSurrogateSkyLightAtlas,
    geometry: RegionGeometryState,
    tokens: WorldGeometryTokenObservation,
    actor_position: Array,
) -> ActorBlockLightTokens:
    """Publish token-aligned open-column sky light for a current Region."""

    if not isinstance(light_atlas, RegionSurrogateSkyLightAtlas):
        raise TypeError("light_atlas has the wrong type")
    if not isinstance(geometry, RegionGeometryState):
        raise TypeError("geometry must be a RegionGeometryState")
    cells = actor_block_light_query_cells(tokens, actor_position)
    batch, actors, capacity, _ = cells.shape
    flat = cells.reshape(batch, actors * capacity, 3)
    channels = query_region_surrogate_sky_light(
        light_atlas,
        geometry,
        flat,
    )
    return surrogate_actor_block_light_tokens(
        tokens,
        actor_position,
        cells,
        channels,
    )


def query_region_surrogate_sky_light(
    light_atlas: RegionSurrogateSkyLightAtlas,
    geometry: RegionGeometryState,
    block_positions: Array,
) -> SurrogatePerceptionChannelResult:
    """Apply native ``updateHeight`` opacity semantics to Region columns.

    The resulting binary sky value remains a surrogate: it does not model
    attenuation, day/night, colored emission, or the native lighting queue.
    Exact mutable geometry overrides do update the packed column before the
    height is selected, preventing stale roof-removal and roof-placement
    observations.
    """

    if not isinstance(light_atlas, RegionSurrogateSkyLightAtlas):
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
    if geometry.environment_world_id.shape != (batch,):
        raise ValueError("geometry batch differs from block_positions")

    # X/Z coverage and height validity are independent of the sample Y.
    column_probe = positions.at[..., 1].set(jnp.int32(MIN_Y))
    selected = lookup_region_blocks(
        geometry.atlas,
        column_probe,
        geometry.environment_world_id,
        require_core=False,
    )
    safe_region = jnp.maximum(selected.region_index, 0)
    capture_min = light_atlas.capture_min_block_xz[safe_region]
    local_x = positions[..., 0] - capture_min[..., 0]
    local_z = positions[..., 2] - capture_min[..., 1]
    inside = (
        (local_x >= 0)
        & (local_x < REGION_CAPTURE_BLOCKS_PER_AXIS)
        & (local_z >= 0)
        & (local_z < REGION_CAPTURE_BLOCKS_PER_AXIS)
    )
    safe_x = jnp.clip(local_x, 0, REGION_CAPTURE_BLOCKS_PER_AXIS - 1)
    safe_z = jnp.clip(local_z, 0, REGION_CAPTURE_BLOCKS_PER_AXIS - 1)
    identity = (
        light_atlas.region_mask[safe_region]
        & (
            light_atlas.world_id[safe_region]
            == geometry.environment_world_id[:, None]
        )
    )
    column_known = light_atlas.column_known[
        safe_region,
        safe_x,
        safe_z,
    ]
    available = selected.available & inside & identity & column_known
    words = light_atlas.opaque_y_words[
        safe_region,
        safe_x,
        safe_z,
    ]
    words, mutation_valid = _apply_mutable_column_overrides(
        words,
        positions,
        geometry,
    )
    available &= mutation_valid
    height = _highest_opaque_y(words)
    inside_y = (
        (positions[..., 1] >= MIN_Y)
        & (positions[..., 1] < MIN_Y + HEIGHT_SECTIONS * CHUNK_SIZE)
    )
    sky_valid = available & inside_y
    above = positions[..., 1].astype(jnp.float32) - height.astype(jnp.float32)
    channel_valid = jnp.zeros(
        (batch, samples, SURROGATE_PERCEPTION_CHANNEL_COUNT),
        dtype=jnp.bool_,
    )
    channel_valid = channel_valid.at[..., CHANNEL_HEIGHTMAP].set(available)
    channel_valid = channel_valid.at[..., CHANNEL_SKY_LIGHT].set(sky_valid)
    return SurrogatePerceptionChannelResult(
        available=available,
        diagnostics=jnp.zeros((batch, samples), dtype=jnp.uint32),
        tile_index=jnp.where(available, selected.region_index, -1),
        channel_valid=channel_valid,
        heightmap_block_y=jnp.where(available, height, 0).astype(jnp.int16),
        height_above_surface=jnp.where(available, above, 0.0),
        sky_light=surrogate_sky_light_from_height(above, sky_valid),
        block_light_rgb=jnp.zeros((batch, samples, 3), dtype=jnp.uint8),
        environment_code=jnp.zeros((batch, samples), dtype=jnp.int32),
        tint_rgb=jnp.zeros((batch, samples, 3), dtype=jnp.uint8),
    )


def region_surrogate_sky_light_contract() -> dict[str, object]:
    """Return the explicit Region sky-light fallback contract."""

    return {
        "schema": REGION_SURROGATE_SKY_LIGHT_SCHEMA,
        "version": REGION_SURROGATE_SKY_LIGHT_VERSION,
        "dependencies": {
            "actor_block_light_token_contract_sha256": (
                actor_block_light_token_contract_sha256()
            ),
        },
        "source_class": _BLOCK_CHUNK_CLASS,
        "source_class_sha256": _BLOCK_CHUNK_CLASS_SHA256,
        "source": {
            "height_predicate": (
                "highest_nonzero_nontransparent_block_or_zero"
            ),
        },
        "storage": "ten_uint32_opaque_y_words_per_captured_xz_column",
        "mutable_refresh": (
            "exact_geometry_overrides_clear_or_set_their_column_bit"
        ),
        "published_channels": {
            "heightmap": "surrogate_from_exact_region_opacity",
            "sky_light": "binary_15_at_or_above_height_else_0",
            "block_light_rgb": None,
            "environment": None,
            "tint": None,
        },
        "published_provenance": "surrogate_not_native_light_calibrated",
        "not_modelled": [
            "attenuation",
            "day_night_state",
            "colored_block_emission",
            "native_lighting_queue_timing",
        ],
        "fail_closed": [
            "unknown_column",
            "world_identity_mismatch",
            "inexact_mutable_override",
        ],
    }


def region_surrogate_sky_light_contract_sha256() -> str:
    payload = json.dumps(
        region_surrogate_sky_light_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _apply_mutable_column_overrides(
    words: Array,
    positions: Array,
    geometry: RegionGeometryState,
) -> tuple[Array, Array]:
    state = geometry.mutable_blocks
    if state is None:
        return words, jnp.ones(words.shape[:2], dtype=jnp.bool_)
    if not isinstance(state, MutableBlockState):
        raise TypeError("geometry.mutable_blocks must be MutableBlockState")
    active = state.cell_mask & state.geometry_override
    same_column = (
        active[:, None, :]
        & (
            state.cell_position[:, None, :, 0]
            == positions[:, :, None, 0]
        )
        & (
            state.cell_position[:, None, :, 2]
            == positions[:, :, None, 2]
        )
    )
    y = state.cell_position[..., 1] - jnp.int32(MIN_Y)
    y_valid = (y >= 0) & (y < HEIGHT_SECTIONS * CHUNK_SIZE)
    exact = state.geometry.exact
    matching_invalid = jnp.any(
        same_column & (~y_valid[:, None, :] | ~exact[:, None, :]),
        axis=2,
    )
    valid_identity = (
        state.synchronized
        & (state.world_id == geometry.environment_world_id)
    )
    mutation_valid = valid_identity[:, None] & ~matching_invalid

    word_index = jnp.floor_divide(jnp.maximum(y, 0), CHUNK_SIZE)
    bit_index = jnp.maximum(y, 0) & (CHUNK_SIZE - 1)
    bit = jnp.left_shift(jnp.uint32(1), bit_index.astype(jnp.uint32))
    word_axis = jnp.arange(HEIGHT_SECTIONS, dtype=jnp.int32)
    at_word = word_index[..., None] == word_axis
    selected = same_column[..., None] & y_valid[:, None, :, None] & at_word[:, None]
    clear = jnp.bitwise_or.reduce(
        jnp.where(selected, bit[:, None, :, None], jnp.uint32(0)),
        axis=2,
    )
    opaque = (
        state.geometry.block_present
        & ((state.geometry.flags & jnp.int32(FLAG_OPAQUE)) != 0)
    )
    set_bits = jnp.bitwise_or.reduce(
        jnp.where(
            selected & opaque[:, None, :, None],
            bit[:, None, :, None],
            jnp.uint32(0),
        ),
        axis=2,
    )
    current = (words & ~clear) | set_bits
    return jnp.where(mutation_valid[..., None], current, 0), mutation_valid


def _highest_opaque_y(words: Array) -> Array:
    nonzero = words != 0
    word_axis = jnp.arange(HEIGHT_SECTIONS, dtype=jnp.int32)
    selected_word = jnp.max(
        jnp.where(nonzero, word_axis, jnp.int32(0)),
        axis=2,
    )
    word = jnp.take_along_axis(
        words,
        selected_word[..., None],
        axis=2,
    )[..., 0]
    bit = jnp.int32(31) - jax.lax.clz(word).astype(jnp.int32)
    return jnp.where(
        jnp.any(nonzero, axis=2),
        selected_word * CHUNK_SIZE + bit + MIN_Y,
        jnp.int32(MIN_Y),
    )


__all__ = [
    "REGION_SURROGATE_SKY_LIGHT_SCHEMA",
    "REGION_SURROGATE_SKY_LIGHT_VERSION",
    "RegionSurrogateSkyLightAtlas",
    "query_region_surrogate_sky_light",
    "region_surrogate_actor_block_light_tokens",
    "region_surrogate_sky_light_atlas_from_region",
    "region_surrogate_sky_light_contract",
    "region_surrogate_sky_light_contract_sha256",
]
