"""Fixed raw perception-channel handoff for surrogate and future native data."""

from __future__ import annotations

import hashlib
import json

import jax
import jax.numpy as jnp

from hytalegym.jax.world.surrogate.atlas import lookup_surrogate_columns
from hytalegym.jax.world.surrogate.capabilities import (
    SURROGATE_QUERY_COVERAGE,
    SURROGATE_QUERY_INVALID,
    SURROGATE_QUERY_RUNTIME,
)
from hytalegym.jax.world.surrogate.types import (
    SurrogateAtlas,
    SurrogatePerceptionChannelResult,
    SurrogateRuntimeState,
)
from hytalegym.worldgen.region import MIN_Y, WORLD_HEIGHT
from hytalegym.worldgen.native_channels import (
    NATIVE_PERCEPTION_CHANNEL_MAX_SAMPLES,
    NATIVE_PERCEPTION_CHANNEL_SCHEMA,
    NATIVE_PERCEPTION_CHANNEL_VERSION,
)

SURROGATE_PERCEPTION_CHANNEL_SCHEMA = "hytalerl_surrogate_perception_channels_v3"
SURROGATE_PERCEPTION_CHANNEL_VERSION = 3
SURROGATE_SKY_LIGHT_MAX = 15
CHANNEL_HEIGHTMAP = 0
CHANNEL_SKY_LIGHT = 1
CHANNEL_BLOCK_LIGHT_RGB = 2
CHANNEL_ENVIRONMENT = 3
CHANNEL_TINT_RGB = 4
SURROGATE_PERCEPTION_CHANNEL_COUNT = 5


def native_perception_channel_contract() -> dict[str, object]:
    """Return the native provider contract with its explicit sky fallback."""

    return {
        "schema": NATIVE_PERCEPTION_CHANNEL_SCHEMA,
        "version": NATIVE_PERCEPTION_CHANNEL_VERSION,
        "published_provenance": "native",
        "sample_capacity": NATIVE_PERCEPTION_CHANNEL_MAX_SAMPLES,
        "sample_coordinates": "absolute_block_i32_xyz",
        "channel_order": [
            "heightmap",
            "sky_light",
            "block_light_rgb",
            "environment",
            "tint_rgb",
        ],
        "channels": {
            "heightmap": {
                "native_source": "BlockChunk.getHeight(x,z)",
                "native_domain": "xz_column",
                "native_no_height_sentinel": MIN_Y + WORLD_HEIGHT,
                "published_provenance": "native",
                "privilege": "requires_visual_or_contact_gating_for_actor_use",
            },
            "sky_light": {
                "native_source": "BlockChunk.getSkyLight",
                "native_readiness": (
                    "BlockSection.hasGlobalLight_and_globalLight_not_ChunkLightData.EMPTY"
                ),
                "native_y_domain_half_open": [MIN_Y, MIN_Y + WORLD_HEIGHT],
                "native_out_of_y_value": 0,
                "native_value_range_inclusive": [0, SURROGATE_SKY_LIGHT_MAX],
                "published_provenance": "native",
                "privilege": "actor_legal_at_observer_sample",
                "surrogate_fallback": {
                    "schema": SURROGATE_PERCEPTION_CHANNEL_SCHEMA,
                    "version": SURROGATE_PERCEPTION_CHANNEL_VERSION,
                    "contract_sha256": (
                        surrogate_perception_channel_contract_sha256()
                    ),
                    "published_provenance": "surrogate",
                    "source": (
                        "binary_open_column_v1(height_above_surface>=0)"
                    ),
                    "status": "available_not_native_calibrated",
                },
            },
            "block_light_rgb": {
                "native_source": (
                    "BlockChunk.getRedBlockLight/getGreenBlockLight/"
                    "getBlueBlockLight"
                ),
                "native_readiness": (
                    "BlockSection.hasGlobalLight_and_globalLight_not_ChunkLightData.EMPTY"
                ),
                "native_y_domain_half_open": [MIN_Y, MIN_Y + WORLD_HEIGHT],
                "native_out_of_y_value": 0,
                "native_value_range_inclusive": [0, 15],
                "published_provenance": "native",
                "surrogate_fallback": None,
                "privilege": "actor_legal_at_observer_sample",
            },
            "environment": {
                "native_source": "BlockChunk.getEnvironment",
                "native_y_domain_half_open": [MIN_Y, MIN_Y + WORLD_HEIGHT],
                "native_out_of_y_value": 0,
                "native_code_scope": "runtime_environment_id",
                "published_provenance": "native",
                "privilege": "requires_visual_context_gating_for_actor_use",
            },
            "tint_rgb": {
                "native_source": "BlockChunk.getTint",
                "native_domain": "xz_column",
                "native_encoding": "argb8888_rgb_low_24_bits",
                "published_provenance": "native",
                "surrogate_fallback": None,
                "privilege": "actor_legal_at_observer_sample",
            },
        },
        "available_semantics": "loaded_chunk_and_block_chunk_present",
        "channel_validity": {
            "heightmap": "available_and_height_not_320_sentinel",
            "sky_light": (
                "available_and_sample_y_in_native_domain_and_global_light_ready"
            ),
            "block_light_rgb": (
                "available_and_sample_y_in_native_domain_and_global_light_ready"
            ),
            "environment": "available_and_sample_y_in_native_domain",
            "tint_rgb": "available_xz_column",
        },
        "light_capture_precondition": (
            "requested_chunk_plus_one_chunk_halo_loaded_and_nonempty_global_light_ready"
        ),
        "invalid_value": "zero_with_channel_valid_false",
        "semantic_digest_excludes": ["ephemeral_native_world_name"],
        "capture_thread": "native_world_thread",
        "query_boundary": "privileged_world_state",
    }


def native_perception_channel_contract_sha256() -> str:
    """Return the checkpoint-pinnable native provider contract hash."""

    encoded = json.dumps(
        native_perception_channel_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def surrogate_perception_channel_contract() -> dict[str, object]:
    """Return the canonical raw-channel schema and privilege boundary."""

    return {
        "schema": SURROGATE_PERCEPTION_CHANNEL_SCHEMA,
        "version": SURROGATE_PERCEPTION_CHANNEL_VERSION,
        "channel_order": [
            "heightmap",
            "sky_light",
            "block_light_rgb",
            "environment",
            "tint_rgb",
        ],
        "channels": {
            "heightmap": {
                "native_source": "BlockChunk.getHeight(x,z)",
                "native_domain": "xz_column",
                "surrogate_source": "terrain_height",
                "surrogate_status": "available",
                "published_provenance": "surrogate",
                "privilege": "requires_visual_or_contact_gating_for_actor_use",
            },
            "sky_light": {
                "native_source": "BlockChunk.getSkyLight",
                "native_y_domain_half_open": [MIN_Y, MIN_Y + WORLD_HEIGHT],
                "native_out_of_y_value": 0,
                "surrogate_source": (
                    "binary_open_column_v1(height_above_surface>=0)"
                ),
                "surrogate_provenance": "surrogate_not_native_calibrated",
                "surrogate_value_range_inclusive": [
                    0,
                    SURROGATE_SKY_LIGHT_MAX,
                ],
                "surrogate_status": "available_not_native_calibrated",
                "surrogate_limitations": (
                    "terrain_height_only_no_structure_cave_day_night_or_attenuation"
                ),
                "published_provenance": "surrogate",
                "privilege": "actor_legal_at_observer_sample",
            },
            "block_light_rgb": {
                "native_source": (
                    "BlockChunk.getRedBlockLight/getGreenBlockLight/"
                    "getBlueBlockLight"
                ),
                "native_y_domain_half_open": [MIN_Y, MIN_Y + WORLD_HEIGHT],
                "native_out_of_y_value": 0,
                "surrogate_status": "unavailable",
                "published_provenance": "none",
                "privilege": "actor_legal_at_observer_sample",
            },
            "environment": {
                "native_source": "BlockChunk.getEnvironment",
                "native_y_domain_half_open": [MIN_Y, MIN_Y + WORLD_HEIGHT],
                "native_out_of_y_value": 0,
                "surrogate_source": "biome_code",
                "surrogate_status": "available_non_native_code",
                "published_provenance": "surrogate",
                "privilege": "requires_visual_context_gating_for_actor_use",
            },
            "tint_rgb": {
                "native_source": "BlockChunk.getTint",
                "native_domain": "xz_column",
                "surrogate_status": "unavailable",
                "published_provenance": "none",
                "privilege": "actor_legal_at_observer_sample",
            },
        },
        "available_semantics": "query_coverage_runtime_and_input_validity",
        "channel_validity": {
            "heightmap": {
                "surrogate": "available",
                "native": "available_and_height_not_no_height_sentinel",
                "native_no_height_sentinel": MIN_Y + WORLD_HEIGHT,
            },
            "environment": "available_and_sample_y_in_native_domain",
            "sky_light": {
                "surrogate": "available_and_sample_y_in_native_domain",
                "native": "provider_available_and_sample_y_in_native_domain",
            },
            "block_light_rgb": (
                "provider_available_and_sample_y_in_native_domain"
            ),
            "tint_rgb": "provider_available_xz_column",
        },
        "partial_availability": "channel_valid_mask",
        "unknown_encoding": "zero_with_channel_valid_false",
        "tile_selection": "xz_column_independent_of_y_and_block_state",
        "query_boundary": "privileged_world_state",
    }


def surrogate_perception_channel_contract_sha256() -> str:
    """Return the checkpoint-pinnable raw-channel contract hash."""

    encoded = json.dumps(
        surrogate_perception_channel_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def surrogate_sky_light_from_height(
    height_above_surface: jax.Array,
    valid: jax.Array,
) -> jax.Array:
    """Return the declared binary surrogate fallback, never native light."""

    height = jnp.asarray(height_above_surface, dtype=jnp.float32)
    mask = jnp.asarray(valid, dtype=jnp.bool_)
    if height.shape != mask.shape:
        raise ValueError("height_above_surface and valid must share a shape")
    return jnp.where(
        mask & jnp.isfinite(height) & (height >= 0.0),
        jnp.uint8(SURROGATE_SKY_LIGHT_MAX),
        jnp.uint8(0),
    )


def query_surrogate_perception_channels(
    atlas: SurrogateAtlas,
    runtime: SurrogateRuntimeState,
    sample_position: jax.Array,
) -> SurrogatePerceptionChannelResult:
    """Gather fixed raw fields independently for each environment/sample."""

    if not isinstance(atlas, SurrogateAtlas):
        raise TypeError("atlas must be a SurrogateAtlas")
    if not isinstance(runtime, SurrogateRuntimeState):
        raise TypeError("runtime must be a SurrogateRuntimeState")
    positions = jnp.asarray(sample_position, dtype=jnp.float32)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("sample_position must have shape [B, Q, 3]")
    batch, queries, _ = positions.shape
    if runtime.environment_world_id.shape != (batch,):
        raise ValueError("runtime and samples must share a batch dimension")

    finite = jnp.all(jnp.isfinite(positions), axis=2)
    safe_positions = jnp.where(finite[..., None], positions, 0.0)
    blocks = jnp.floor(safe_positions).astype(jnp.int32)
    selected = lookup_surrogate_columns(
        atlas,
        runtime,
        blocks[..., (0, 2)],
        require_core=False,
    )
    covered = selected.available & finite
    safe_tile = jnp.maximum(selected.tile_index, 0)
    height = atlas.terrain_height[
        safe_tile,
        selected.local_xz[..., 0],
        selected.local_xz[..., 1],
    ]
    environment = atlas.biome_code[
        safe_tile,
        selected.local_xz[..., 0],
        selected.local_xz[..., 1],
    ]
    available = covered
    inside_y = (blocks[..., 1] >= MIN_Y) & (
        blocks[..., 1] < MIN_Y + WORLD_HEIGHT
    )

    diagnostics = jnp.zeros((batch, queries), dtype=jnp.uint32)
    diagnostics |= jnp.where(
        ~finite,
        jnp.uint32(SURROGATE_QUERY_INVALID),
        jnp.uint32(0),
    )
    diagnostics |= jnp.where(
        finite & ~selected.available & ~runtime.unsupported_mechanics[:, None],
        jnp.uint32(SURROGATE_QUERY_COVERAGE),
        jnp.uint32(0),
    )
    diagnostics |= jnp.where(
        runtime.unsupported_mechanics[:, None],
        jnp.uint32(SURROGATE_QUERY_RUNTIME),
        jnp.uint32(0),
    )
    available &= diagnostics == 0
    height_above_surface = (
        safe_positions[..., 1] - (height.astype(jnp.float32) + 1.0)
    )
    sky_light_valid = available & inside_y
    channel_valid = jnp.zeros(
        (batch, queries, SURROGATE_PERCEPTION_CHANNEL_COUNT),
        dtype=jnp.bool_,
    )
    channel_valid = channel_valid.at[..., CHANNEL_HEIGHTMAP].set(available)
    channel_valid = channel_valid.at[..., CHANNEL_ENVIRONMENT].set(
        available & inside_y
    )
    channel_valid = channel_valid.at[..., CHANNEL_SKY_LIGHT].set(
        sky_light_valid
    )
    return SurrogatePerceptionChannelResult(
        available=available,
        diagnostics=diagnostics,
        tile_index=jnp.where(available, selected.tile_index, -1),
        channel_valid=channel_valid,
        heightmap_block_y=jnp.where(available, height, jnp.int16(0)),
        height_above_surface=jnp.where(
            available,
            height_above_surface,
            0.0,
        ),
        sky_light=surrogate_sky_light_from_height(
            height_above_surface,
            sky_light_valid,
        ),
        block_light_rgb=jnp.zeros((batch, queries, 3), dtype=jnp.uint8),
        environment_code=jnp.where(
            available & inside_y,
            environment.astype(jnp.int32),
            0,
        ),
        tint_rgb=jnp.zeros((batch, queries, 3), dtype=jnp.uint8),
    )


__all__ = [
    "CHANNEL_BLOCK_LIGHT_RGB",
    "CHANNEL_ENVIRONMENT",
    "CHANNEL_HEIGHTMAP",
    "CHANNEL_SKY_LIGHT",
    "CHANNEL_TINT_RGB",
    "SURROGATE_PERCEPTION_CHANNEL_COUNT",
    "SURROGATE_PERCEPTION_CHANNEL_SCHEMA",
    "SURROGATE_PERCEPTION_CHANNEL_VERSION",
    "SURROGATE_SKY_LIGHT_MAX",
    "native_perception_channel_contract",
    "native_perception_channel_contract_sha256",
    "query_surrogate_perception_channels",
    "surrogate_sky_light_from_height",
    "surrogate_perception_channel_contract",
    "surrogate_perception_channel_contract_sha256",
]
