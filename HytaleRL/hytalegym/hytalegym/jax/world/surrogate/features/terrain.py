"""Exact standable-surface learner tokens."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.observation.v1.schema.contract import (
    TERRAIN_RADIUS_BLOCKS,
    TERRAIN_TOKEN_CAPACITY,
)
from hytalegym.jax.world.surrogate.atlas import lookup_surrogate_blocks
from hytalegym.jax.world.surrogate.features.common import (
    decode_capture_keys,
    pad_tokens,
    runtime_gate_available,
    select_core_tiles,
)
from hytalegym.jax.world.surrogate.features.contract import (
    TERRAIN_OFFSETS_XZ,
    TERRAIN_SURFACE_STACK_CAPACITY,
)
from hytalegym.jax.world.surrogate.types import (
    SurrogateAtlas,
    SurrogateRuntimeState,
    SurrogateTraversalAtlas,
)
from hytalegym.worldgen.region import CHUNK_SIZE, CORE_BLOCKS_PER_AXIS


class TerrainFeatures(NamedTuple):
    """Internal fixed terrain result."""

    f32: jax.Array
    semantic_id: jax.Array
    flags: jax.Array
    mask: jax.Array
    overflow: jax.Array
    missing_tile: jax.Array
    graph_unavailable: jax.Array
    graph_diagnostics: jax.Array
    tile_index: jax.Array


def terrain_features(
    atlas: SurrogateAtlas,
    traversal: SurrogateTraversalAtlas,
    runtime: SurrogateRuntimeState,
    actor_positions: jax.Array,
    actor_valid: jax.Array,
) -> TerrainFeatures:
    """Sample exact runtime-legal surfaces for every actor."""

    offsets = jnp.asarray(TERRAIN_OFFSETS_XZ, dtype=jnp.int32)
    sample_xz = (
        jnp.floor(actor_positions[:, None, (0, 2)]).astype(jnp.int32)
        + offsets[None, :, :]
    )
    tile_index, has_tile = select_core_tiles(
        traversal.graph_mask,
        traversal.world_id,
        traversal.core_min_chunk_xz,
        runtime.environment_world_id,
        sample_xz,
        actor_valid,
    )
    safe_tile = jnp.maximum(tile_index, 0)
    graph_available = has_tile & traversal.graph_available[safe_tile]
    graph_diagnostics = jnp.bitwise_or.reduce(
        jnp.where(
            has_tile,
            traversal.diagnostics[safe_tile],
            jnp.uint32(0),
        ),
        axis=1,
    )
    core_origin = traversal.core_min_chunk_xz[safe_tile] * CHUNK_SIZE
    local_xz = jnp.clip(
        sample_xz - core_origin,
        0,
        CORE_BLOCKS_PER_AXIS - 1,
    )
    start = traversal.column_node_start[
        safe_tile,
        local_xz[..., 0],
        local_xz[..., 1],
    ]
    count = traversal.column_node_count[
        safe_tile,
        local_xz[..., 0],
        local_xz[..., 1],
    ].astype(jnp.int32)
    stack_offset = jnp.arange(TERRAIN_SURFACE_STACK_CAPACITY, dtype=jnp.int32)
    node_index = jnp.clip(
        start[..., None] + stack_offset,
        0,
        traversal.node_mask.shape[1] - 1,
    )
    stack_mask = (
        (stack_offset < count[..., None])
        & (count[..., None] <= TERRAIN_SURFACE_STACK_CAPACITY)
        & traversal.node_mask[safe_tile[..., None], node_index]
        & graph_available[..., None]
    )
    stack_mask &= runtime_gate_available(
        runtime,
        traversal.node_gate_runtime_slot[
            safe_tile[..., None],
            node_index,
        ],
        traversal.node_gate_state_mask[
            safe_tile[..., None],
            node_index,
        ],
    )
    node_position = traversal.node_position[
        safe_tile[..., None],
        node_index,
    ]
    vertical_distance = jnp.where(
        stack_mask,
        jnp.abs(node_position[..., 1] - actor_positions[:, None, None, 1]),
        jnp.inf,
    )
    selected_stack = jnp.argmin(vertical_distance, axis=2)
    selected_position = jnp.take_along_axis(
        node_position,
        selected_stack[..., None, None],
        axis=2,
    )[..., 0, :]
    selected_node = jnp.take_along_axis(
        node_index,
        selected_stack[..., None],
        axis=2,
    )[..., 0]
    surface_found = jnp.any(stack_mask, axis=2)
    support_key = traversal.node_support_key[safe_tile, selected_node]
    support = lookup_surrogate_blocks(
        atlas,
        runtime,
        decode_capture_keys(
            support_key,
            traversal.core_min_chunk_xz[safe_tile],
        ),
        require_core=False,
    )
    relative = selected_position - actor_positions[:, None, :]
    distance_squared = jnp.sum(relative**2, axis=2)
    within_radius = distance_squared <= TERRAIN_RADIUS_BLOCKS**2
    overflow = actor_valid & jnp.any(
        graph_available & (count > TERRAIN_SURFACE_STACK_CAPACITY),
        axis=1,
    )
    missing_tile = actor_valid & jnp.any(~has_tile, axis=1)
    graph_unavailable = actor_valid & jnp.any(
        has_tile & ~graph_available,
        axis=1,
    )
    fatal = (
        ~actor_valid
        | runtime.unsupported_mechanics
        | missing_tile
        | graph_unavailable
        | overflow
    )
    token_mask = surface_found & within_radius & support.available & ~fatal[:, None]
    horizontal_multiplier = support.movement[..., 2]
    movement_cost = jnp.where(
        horizontal_multiplier > 0.0,
        jnp.clip(1.0 - horizontal_multiplier, 0.0, 1.0),
        0.0,
    )
    damage_cost = (jnp.maximum(support.block_damage, support.fluid_damage) > 0).astype(
        jnp.float32
    )
    interaction_relevance = (
        traversal.node_gate_runtime_slot[safe_tile, selected_node] >= 0
    ).astype(jnp.float32)
    f32 = jnp.stack(
        (
            *(relative[..., axis] / TERRAIN_RADIUS_BLOCKS for axis in range(3)),
            jnp.sqrt(distance_squared) / TERRAIN_RADIUS_BLOCKS,
            jnp.ones_like(distance_squared),
            movement_cost,
            damage_cost,
            interaction_relevance,
        ),
        axis=2,
    )
    f32 = jnp.clip(f32, -1.0, 1.0)
    semantic_id = support.cell_code.astype(jnp.int32)
    flags = support.flags.astype(jnp.uint32)
    source = jnp.broadcast_to(
        jnp.arange(sample_xz.shape[1], dtype=jnp.int32),
        token_mask.shape,
    )
    _, _, order = jax.lax.sort(
        (
            jnp.where(token_mask, distance_squared, jnp.inf),
            jnp.where(
                token_mask,
                semantic_id,
                jnp.iinfo(jnp.int32).max,
            ),
            source,
        ),
        dimension=1,
        num_keys=2,
        is_stable=True,
    )
    sorted_mask = jnp.take_along_axis(token_mask, order, axis=1)
    sorted_f32 = jnp.where(
        sorted_mask[..., None],
        jnp.take_along_axis(f32, order[..., None], axis=1),
        0.0,
    )
    sorted_semantic = jnp.where(
        sorted_mask,
        jnp.take_along_axis(semantic_id, order, axis=1),
        0,
    )
    sorted_flags = jnp.where(
        sorted_mask,
        jnp.take_along_axis(flags, order, axis=1),
        0,
    )
    return TerrainFeatures(
        f32=pad_tokens(sorted_f32, TERRAIN_TOKEN_CAPACITY, 0.0),
        semantic_id=pad_tokens(sorted_semantic, TERRAIN_TOKEN_CAPACITY, 0),
        flags=pad_tokens(sorted_flags, TERRAIN_TOKEN_CAPACITY, 0),
        mask=pad_tokens(sorted_mask, TERRAIN_TOKEN_CAPACITY, False),
        overflow=overflow,
        missing_tile=missing_tile,
        graph_unavailable=graph_unavailable,
        graph_diagnostics=graph_diagnostics,
        tile_index=pad_tokens(
            jnp.take_along_axis(tile_index, order, axis=1),
            TERRAIN_TOKEN_CAPACITY,
            -1,
        ),
    )


__all__ = ["TerrainFeatures", "terrain_features"]
