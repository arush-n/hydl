"""Shared fixed-shape selection helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.world.surrogate.types import (
    SurrogateAtlas,
    SurrogateRuntimeState,
    SurrogateTraversalAtlas,
)
from hytalegym.worldgen.region import (
    CAPTURE_BLOCKS_PER_AXIS,
    CHUNK_SIZE,
    CORE_BLOCKS_PER_AXIS,
    MIN_Y,
)
from hytalegym.worldgen.surrogate import TRAVERSAL_STATE_MASK_BITS


def select_core_tiles(
    tile_mask: jax.Array,
    world_id: jax.Array,
    core_min_chunk_xz: jax.Array,
    environment_world_id: jax.Array,
    block_xz: jax.Array,
    query_valid: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Select the nearest compatible core independently for every query."""

    core_chunks = CORE_BLOCKS_PER_AXIS // CHUNK_SIZE
    chunks = jnp.floor_divide(block_xz, CHUNK_SIZE)
    relative = chunks[:, :, None, :] - core_min_chunk_xz[None, None, :, :]
    inside = jnp.all((relative >= 0) & (relative < core_chunks), axis=3)
    eligible = (
        inside
        & tile_mask[None, None, :]
        & (world_id[None, None, :] == environment_world_id[:, None, None])
        & query_valid[:, None, None]
    )
    local_xz = block_xz & (CHUNK_SIZE - 1)
    center_delta_twice = (
        jnp.clip(relative, 0, core_chunks - 1) * (2 * CHUNK_SIZE)
        + local_xz[:, :, None, :] * 2
        + 1
        - CORE_BLOCKS_PER_AXIS
    )
    distance = jnp.sum(center_delta_twice**2, axis=3)
    score = jnp.where(eligible, distance, jnp.iinfo(jnp.int32).max)
    tile_index = jnp.argmin(score, axis=2).astype(jnp.int32)
    has_tile = jnp.any(eligible, axis=2)
    return jnp.where(has_tile, tile_index, -1), has_tile


def decode_capture_keys(
    key: jax.Array,
    core_min_chunk_xz: jax.Array,
) -> jax.Array:
    """Decode capture-local linear keys to world block coordinates."""

    safe_key = jnp.maximum(jnp.asarray(key, dtype=jnp.int32), 0)
    area = CAPTURE_BLOCKS_PER_AXIS**2
    local_y = jnp.floor_divide(safe_key, area)
    remainder = safe_key - local_y * area
    local_z = jnp.floor_divide(remainder, CAPTURE_BLOCKS_PER_AXIS)
    local_x = remainder - local_z * CAPTURE_BLOCKS_PER_AXIS
    capture_origin = core_min_chunk_xz * CHUNK_SIZE - CHUNK_SIZE
    return jnp.stack(
        (
            capture_origin[..., 0] + local_x,
            local_y + MIN_Y,
            capture_origin[..., 1] + local_z,
        ),
        axis=-1,
    ).astype(jnp.int32)


def runtime_gate_available(
    runtime: SurrogateRuntimeState,
    runtime_slot: jax.Array,
    state_mask: jax.Array,
) -> jax.Array:
    """Resolve static or runtime-state-gated graph entries."""

    slots = jnp.asarray(runtime_slot, dtype=jnp.int32)
    masks = jnp.asarray(state_mask, dtype=jnp.uint8)
    capacity = runtime.stateful_state.shape[1]
    safe_slot = jnp.clip(slots, 0, capacity - 1)
    batch_axis = jnp.arange(slots.shape[0]).reshape(
        (slots.shape[0],) + (1,) * (slots.ndim - 1)
    )
    state = runtime.stateful_state[batch_axis, safe_slot]
    initialized = runtime.stateful_initialized[batch_axis, safe_slot]
    representable = state < TRAVERSAL_STATE_MASK_BITS
    state_bit = jnp.left_shift(
        jnp.uint8(1),
        jnp.minimum(state, TRAVERSAL_STATE_MASK_BITS - 1),
    )
    dynamic = (
        (slots >= 0)
        & (slots < capacity)
        & initialized
        & representable
        & ((masks & state_bit) != 0)
    )
    return (slots < 0) | dynamic


def pad_tokens(
    value: jax.Array,
    capacity: int,
    pad_value: int | float | bool,
) -> jax.Array:
    """Pad a compact batch-token array to its public capacity."""

    padding = capacity - value.shape[1]
    if padding < 0:
        raise ValueError("token source exceeds its fixed capacity")
    widths = ((0, 0), (0, padding)) + ((0, 0),) * (value.ndim - 2)
    return jnp.pad(value, widths, constant_values=pad_value)


def validate_feature_shapes(
    atlas: SurrogateAtlas,
    traversal: SurrogateTraversalAtlas,
    runtime: SurrogateRuntimeState,
    batch: int,
) -> None:
    """Reject structural drift before tracing the producer."""

    tile_capacity = atlas.tile_mask.shape[0]
    if traversal.graph_mask.shape != (tile_capacity,):
        raise ValueError("traversal atlas does not align with the surrogate atlas")
    expected_columns = (
        tile_capacity,
        CORE_BLOCKS_PER_AXIS,
        CORE_BLOCKS_PER_AXIS,
    )
    if (
        traversal.column_node_start.shape != expected_columns
        or traversal.column_node_count.shape != expected_columns
    ):
        raise ValueError("traversal column index has an invalid shape")
    if runtime.environment_world_id.shape != (batch,):
        raise ValueError("runtime batch does not match actor_positions")
    if (
        runtime.unsupported_mechanics.shape != (batch,)
        or runtime.capability_bits.shape != (batch,)
        or runtime.failure_bits.shape != (batch,)
    ):
        raise ValueError("runtime unsupported mask does not match actor_positions")


__all__ = [
    "decode_capture_keys",
    "pad_tokens",
    "runtime_gate_available",
    "select_core_tiles",
    "validate_feature_shapes",
]
