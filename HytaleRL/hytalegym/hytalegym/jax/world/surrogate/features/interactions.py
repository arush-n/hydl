"""Door observation and object-ID transition resolution."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.observation.v1.schema.contract import (
    DOOR_INTENT_CLOSE,
    DOOR_INTENT_NONE,
    DOOR_INTENT_OPEN,
    DOOR_INTENT_USE,
    INTERACTION_CAPACITY,
    INTERACTION_RADIUS_BLOCKS,
)
from hytalegym.jax.world.surrogate.atlas import (
    GENERATION_READY,
    apply_surrogate_door_use,
)
from hytalegym.jax.world.surrogate.capabilities import (
    SURROGATE_FAILURE_DOOR_INTENT,
)
from hytalegym.jax.world.surrogate.features.common import (
    decode_capture_keys,
    pad_tokens,
    select_core_tiles,
)
from hytalegym.jax.world.surrogate.types import (
    SurrogateAtlas,
    SurrogateDoorTransitionResult,
    SurrogateRuntimeState,
)
from hytalegym.worldgen.region import MIN_Y, WORLD_HEIGHT


class InteractionFeatures(NamedTuple):
    """Internal fixed door-candidate result."""

    f32: jax.Array
    object_id: jax.Array
    mask: jax.Array
    intent_mask: jax.Array
    overflow: jax.Array
    missing_tile: jax.Array
    tile_index: jax.Array


def interaction_features(
    atlas: SurrogateAtlas,
    runtime: SurrogateRuntimeState,
    actor_positions: jax.Array,
    actor_yaw_degrees: jax.Array,
    actor_valid: jax.Array,
) -> InteractionFeatures:
    """Select legal root-door interactions for every actor."""

    actor_xz = jnp.floor(actor_positions[:, None, (0, 2)]).astype(jnp.int32)
    tile_index_2d, has_tile_2d = select_core_tiles(
        atlas.tile_mask & (atlas.generation_status == GENERATION_READY),
        atlas.world_id,
        atlas.core_min_chunk_xz,
        runtime.environment_world_id,
        actor_xz,
        actor_valid,
    )
    tile_index = tile_index_2d[:, 0]
    has_tile = has_tile_2d[:, 0]
    safe_tile = jnp.maximum(tile_index, 0)
    definition_count = jnp.sum(
        atlas.stateful_mask[safe_tile],
        axis=1,
        dtype=jnp.int32,
    )
    selected_distance, selected_object, selected_definition, candidate_count = (
        _nearest_root_doors(
            atlas,
            runtime,
            actor_positions,
            actor_valid,
            safe_tile,
            has_tile,
            definition_count,
        )
    )
    overflow = candidate_count > INTERACTION_CAPACITY
    selected_candidate = jnp.isfinite(selected_distance)
    block_key = atlas.stateful_block_key[
        safe_tile[:, None],
        selected_definition,
    ]
    door_position = (
        decode_capture_keys(
            block_key,
            atlas.core_min_chunk_xz[safe_tile, None, :],
        ).astype(jnp.float32)
        + 0.5
    )
    relative = door_position - actor_positions[:, None, :]
    runtime_slot = atlas.stateful_runtime_slot[
        safe_tile[:, None],
        selected_definition,
    ].astype(jnp.int32)
    state_capacity = runtime.stateful_state.shape[1]
    safe_runtime_slot = jnp.clip(runtime_slot, 0, state_capacity - 1)
    batch_axis = jnp.arange(actor_positions.shape[0])[:, None]
    current_state = runtime.stateful_state[batch_axis, safe_runtime_slot]
    transition_supported = jnp.any(
        atlas.stateful_transition_mask[
            safe_tile[:, None],
            selected_definition,
            jnp.minimum(
                current_state,
                atlas.stateful_transition_mask.shape[2] - 1,
            ),
        ],
        axis=2,
    )
    mask = selected_candidate & ~overflow[:, None]
    distance = jnp.sqrt(selected_distance)
    yaw = jnp.deg2rad(actor_yaw_degrees)
    horizontal_distance = jnp.sqrt(relative[..., 0] ** 2 + relative[..., 2] ** 2)
    facing_alignment = jnp.where(
        horizontal_distance > 1.0e-6,
        (
            -jnp.sin(yaw)[:, None] * relative[..., 0]
            - jnp.cos(yaw)[:, None] * relative[..., 2]
        )
        / horizontal_distance,
        1.0,
    )
    f32 = jnp.stack(
        (
            *(relative[..., axis] / INTERACTION_RADIUS_BLOCKS for axis in range(3)),
            distance / INTERACTION_RADIUS_BLOCKS,
            facing_alignment,
            1.0 - distance / INTERACTION_RADIUS_BLOCKS,
        ),
        axis=2,
    )
    f32 = jnp.where(mask[..., None], jnp.clip(f32, -1.0, 1.0), 0.0)
    intent_mask = jnp.stack(
        (
            (current_state == 0) & transition_supported,
            (current_state > 0) & transition_supported,
            transition_supported,
        ),
        axis=2,
    )
    intent_mask &= mask[..., None]
    return InteractionFeatures(
        f32=pad_tokens(f32, INTERACTION_CAPACITY, 0.0),
        object_id=pad_tokens(
            jnp.where(mask, selected_object, 0),
            INTERACTION_CAPACITY,
            0,
        ),
        mask=pad_tokens(mask, INTERACTION_CAPACITY, False),
        intent_mask=pad_tokens(
            intent_mask,
            INTERACTION_CAPACITY,
            False,
        ),
        overflow=overflow,
        missing_tile=actor_valid & ~has_tile,
        tile_index=jnp.where(has_tile, tile_index, -1),
    )


def _nearest_root_doors(
    atlas: SurrogateAtlas,
    runtime: SurrogateRuntimeState,
    actor_positions: jax.Array,
    actor_valid: jax.Array,
    safe_tile: jax.Array,
    has_tile: jax.Array,
    definition_count: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Keep nearest roots while scanning only each tile's active prefix."""

    batch = actor_positions.shape[0]
    state_capacity = runtime.stateful_state.shape[1]
    initial = (
        jnp.full((batch, INTERACTION_CAPACITY), jnp.inf, dtype=jnp.float32),
        jnp.full(
            (batch, INTERACTION_CAPACITY),
            jnp.iinfo(jnp.int32).max,
            dtype=jnp.int32,
        ),
        jnp.zeros((batch, INTERACTION_CAPACITY), dtype=jnp.int32),
        jnp.zeros(batch, dtype=jnp.int32),
    )

    def body(index, carry):
        distances, objects, definitions, count = carry
        block_key = atlas.stateful_block_key[safe_tile, index]
        identity_key = atlas.stateful_identity_key[safe_tile, index]
        runtime_slot = atlas.stateful_runtime_slot[safe_tile, index].astype(jnp.int32)
        safe_runtime_slot = jnp.clip(runtime_slot, 0, state_capacity - 1)
        current_state = runtime.stateful_state[
            jnp.arange(batch),
            safe_runtime_slot,
        ]
        initialized = runtime.stateful_initialized[
            jnp.arange(batch),
            safe_runtime_slot,
        ]
        state_count = atlas.stateful_state_count[safe_tile, index]
        transition_supported = jnp.any(
            atlas.stateful_transition_mask[
                safe_tile,
                index,
                jnp.minimum(
                    current_state,
                    atlas.stateful_transition_mask.shape[2] - 1,
                ),
            ],
            axis=1,
        )
        position = (
            decode_capture_keys(
                block_key,
                atlas.core_min_chunk_xz[safe_tile],
            ).astype(jnp.float32)
            + 0.5
        )
        distance = jnp.sum((position - actor_positions) ** 2, axis=1)
        candidate = (
            (index < definition_count)
            & _is_door_definition(atlas, safe_tile, index)
            & (block_key == identity_key)
            & has_tile
            & actor_valid
            & ~runtime.unsupported_mechanics
            & (runtime_slot >= 0)
            & (runtime_slot < state_capacity)
            & initialized
            & (current_state < state_count)
            & transition_supported
            & (distance <= INTERACTION_RADIUS_BLOCKS**2)
        )
        candidate_distance = jnp.where(candidate, distance, jnp.inf)
        candidate_object = jnp.where(
            candidate,
            runtime_slot,
            jnp.iinfo(jnp.int32).max,
        )
        sorted_distance, sorted_object, sorted_definition = jax.lax.sort(
            (
                jnp.concatenate(
                    (distances, candidate_distance[:, None]),
                    axis=1,
                ),
                jnp.concatenate(
                    (objects, candidate_object[:, None]),
                    axis=1,
                ),
                jnp.concatenate(
                    (
                        definitions,
                        jnp.full((batch, 1), index, dtype=jnp.int32),
                    ),
                    axis=1,
                ),
            ),
            dimension=1,
            num_keys=2,
            is_stable=True,
        )
        return (
            sorted_distance[:, :INTERACTION_CAPACITY],
            sorted_object[:, :INTERACTION_CAPACITY],
            sorted_definition[:, :INTERACTION_CAPACITY],
            count + candidate.astype(jnp.int32),
        )

    return jax.lax.fori_loop(
        0,
        jnp.max(definition_count),
        body,
        initial,
    )


def apply_surrogate_door_intent(
    atlas: SurrogateAtlas,
    runtime: SurrogateRuntimeState,
    object_id: jax.Array,
    intent: jax.Array,
    actor_positions: jax.Array,
    target_clear: jax.Array,
    *,
    opposite_target_clear: jax.Array | None = None,
    partner_target_clear: jax.Array | None = None,
    partner_opposite_target_clear: jax.Array | None = None,
) -> SurrogateDoorTransitionResult:
    """Resolve stable object IDs and apply one clearance-checked door intent."""

    if not isinstance(atlas, SurrogateAtlas):
        raise TypeError("atlas must be a SurrogateAtlas")
    if not isinstance(runtime, SurrogateRuntimeState):
        raise TypeError("runtime must be a SurrogateRuntimeState")
    objects = jnp.asarray(object_id, dtype=jnp.int32)
    intents = jnp.asarray(intent, dtype=jnp.int32)
    actors = jnp.asarray(actor_positions, dtype=jnp.float32)
    if actors.ndim != 2 or actors.shape[1] != 3:
        raise ValueError("actor_positions must have shape [batch, 3]")
    batch = actors.shape[0]
    if objects.shape != (batch,) or intents.shape != (batch,):
        raise ValueError("object_id and intent must have shape [batch]")
    if runtime.environment_world_id.shape != (batch,):
        raise ValueError("runtime batch does not match actor_positions")

    finite = jnp.all(jnp.isfinite(actors), axis=1)
    precise = jnp.all(jnp.abs(actors) < jnp.float32(1 << 24), axis=1)
    inside_y = (actors[:, 1] >= MIN_Y) & (actors[:, 1] < MIN_Y + WORLD_HEIGHT)
    actor_valid = finite & precise & inside_y
    safe_actors = jnp.where(actor_valid[:, None], actors, 0.0)
    tile_index, has_tile_2d = select_core_tiles(
        atlas.tile_mask & (atlas.generation_status == GENERATION_READY),
        atlas.world_id,
        atlas.core_min_chunk_xz,
        runtime.environment_world_id,
        jnp.floor(safe_actors[:, None, (0, 2)]).astype(jnp.int32),
        actor_valid,
    )
    has_tile = has_tile_2d[:, 0]
    safe_tile = jnp.maximum(tile_index[:, 0], 0)
    definition_count = jnp.sum(
        atlas.stateful_mask[safe_tile],
        axis=1,
        dtype=jnp.int32,
    )
    definition, match_count = _matching_root_door(
        atlas,
        safe_tile,
        objects,
        definition_count,
    )
    door_block = decode_capture_keys(
        atlas.stateful_block_key[safe_tile, definition],
        atlas.core_min_chunk_xz[safe_tile],
    )
    in_reach = (
        jnp.sum(
            (door_block.astype(jnp.float32) + 0.5 - safe_actors) ** 2,
            axis=1,
        )
        <= INTERACTION_RADIUS_BLOCKS**2
    )

    state_capacity = runtime.stateful_state.shape[1]
    safe_object = jnp.clip(objects, 0, state_capacity - 1)
    current = runtime.stateful_state[jnp.arange(batch), safe_object]
    initialized = runtime.stateful_initialized[jnp.arange(batch), safe_object]
    state_count = atlas.stateful_state_count[safe_tile, definition]
    transition_supported = jnp.any(
        atlas.stateful_transition_mask[
            safe_tile,
            definition,
            jnp.minimum(
                current,
                atlas.stateful_transition_mask.shape[2] - 1,
            ),
        ],
        axis=1,
    )
    requested = intents != DOOR_INTENT_NONE
    semantic_legal = (
        ((intents == DOOR_INTENT_OPEN) & (current == 0))
        | ((intents == DOOR_INTENT_CLOSE) & (current > 0))
        | (intents == DOOR_INTENT_USE)
    )
    request_valid = (
        requested
        & actor_valid
        & has_tile
        & ~runtime.unsupported_mechanics
        & (match_count == 1)
        & (objects >= 0)
        & (objects < state_capacity)
        & initialized
        & (current < state_count)
        & transition_supported
        & semantic_legal
        & in_reach
    )
    result = apply_surrogate_door_use(
        atlas,
        runtime,
        door_block,
        safe_actors,
        target_clear,
        intent_mask=request_valid,
        opposite_target_clear=opposite_target_clear,
        partner_target_clear=partner_target_clear,
        partner_opposite_target_clear=partner_opposite_target_clear,
    )
    unsupported = result.unsupported | (requested & ~request_valid)
    failure_bits = result.runtime.failure_bits | jnp.where(
        unsupported,
        jnp.uint32(SURROGATE_FAILURE_DOOR_INTENT),
        jnp.uint32(0),
    )
    return result._replace(
        runtime=result.runtime._replace(
            failure_bits=failure_bits,
            unsupported_mechanics=(failure_bits != 0),
        ),
        unsupported=unsupported,
    )


def _matching_root_door(
    atlas: SurrogateAtlas,
    safe_tile: jax.Array,
    object_id: jax.Array,
    definition_count: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Resolve one stable object ID over each tile's active prefix."""

    def body(index, carry):
        definition, count = carry
        match = (
            (index < definition_count)
            & _is_door_definition(atlas, safe_tile, index)
            & (
                atlas.stateful_block_key[safe_tile, index]
                == atlas.stateful_identity_key[safe_tile, index]
            )
            & (
                atlas.stateful_runtime_slot[safe_tile, index].astype(jnp.int32)
                == object_id
            )
        )
        return (
            jnp.where(match, index, definition),
            count + match.astype(jnp.int32),
        )

    return jax.lax.fori_loop(
        0,
        jnp.max(definition_count),
        body,
        (
            jnp.zeros(object_id.shape, dtype=jnp.int32),
            jnp.zeros(object_id.shape, dtype=jnp.int32),
        ),
    )


def _is_door_definition(
    atlas: SurrogateAtlas,
    safe_tile: jax.Array,
    definition: int | jax.Array,
) -> jax.Array:
    """Distinguish side-dependent doors from simple state-change blocks."""

    success = atlas.stateful_success_target[safe_tile, definition]
    blocked = atlas.stateful_blocked_target[safe_tile, definition]
    transition = atlas.stateful_transition_mask[safe_tile, definition]
    state_count = atlas.stateful_state_count[safe_tile, definition]
    declared = (
        jnp.arange(transition.shape[1], dtype=jnp.uint8)[None, :]
        < state_count[:, None]
    )
    side_dependent = (
        (success[..., 0] != success[..., 1])
        | (blocked[..., 0] != blocked[..., 1])
        | (transition[..., 0] != transition[..., 1])
    )
    return jnp.any(declared & side_dependent, axis=1)


__all__ = [
    "InteractionFeatures",
    "apply_surrogate_door_intent",
    "interaction_features",
]
