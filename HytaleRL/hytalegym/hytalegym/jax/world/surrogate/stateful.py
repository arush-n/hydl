"""Fixed-shape state transitions for exact simple block Use interactions."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.world.surrogate.atlas import lookup_surrogate_blocks
from hytalegym.jax.world.surrogate.capabilities import (
    SURROGATE_FAILURE_STATE_CHANGE_USE,
)
from hytalegym.jax.world.surrogate.types import (
    SurrogateAtlas,
    SurrogateRuntimeState,
    SurrogateStateChangeResult,
)


def apply_surrogate_state_change_use(
    atlas: SurrogateAtlas,
    runtime: SurrogateRuntimeState,
    block_positions: jax.Array,
    *,
    intent_mask: jax.Array | None = None,
) -> SurrogateStateChangeResult:
    """Apply an exact side-independent ChangeState mapping.

    Actor targeting, distance, LOS, and permissions are external gates. This
    primitive accepts only initialized, covered definitions whose complete
    transition table is side-independent and has no paired block.
    """

    if not isinstance(atlas, SurrogateAtlas):
        raise TypeError("atlas must be a SurrogateAtlas")
    if not isinstance(runtime, SurrogateRuntimeState):
        raise TypeError("runtime must be a SurrogateRuntimeState")
    positions = jnp.asarray(block_positions, dtype=jnp.int32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("block_positions must have shape [batch, 3]")
    batch = positions.shape[0]
    intents = (
        jnp.ones(batch, dtype=jnp.bool_)
        if intent_mask is None
        else jnp.asarray(intent_mask, dtype=jnp.bool_)
    )
    if intents.shape != (batch,):
        raise ValueError("intent_mask must have shape [batch]")

    selected = lookup_surrogate_blocks(
        atlas,
        runtime,
        positions[:, None, :],
        require_core=True,
    )
    tile = jnp.maximum(selected.tile_index[:, 0], 0)
    definition = jnp.maximum(selected.stateful_definition_slot[:, 0], 0)
    runtime_slot = jnp.maximum(selected.stateful_runtime_slot[:, 0], 0)
    batch_axis = jnp.arange(batch)
    current = runtime.stateful_state[batch_axis, runtime_slot]
    initialized = runtime.stateful_initialized[batch_axis, runtime_slot]
    state_count = atlas.stateful_state_count[tile, definition]
    table_state = jnp.minimum(
        current,
        atlas.stateful_transition_mask.shape[2] - 1,
    )

    success = atlas.stateful_success_target[tile, definition]
    blocked = atlas.stateful_blocked_target[tile, definition]
    transition = atlas.stateful_transition_mask[tile, definition]
    declared = (
        jnp.arange(transition.shape[1], dtype=jnp.uint8)[None, :]
        < state_count[:, None]
    )
    side_independent = jnp.all(
        ~declared
        | (
            (success[..., 0] == success[..., 1])
            & (blocked[..., 0] == blocked[..., 1])
            & (transition[..., 0] == transition[..., 1])
        ),
        axis=1,
    )
    unpaired = atlas.stateful_partner_runtime_slot[tile, definition] < 0
    supported = transition[batch_axis, table_state, 0]
    valid = (
        intents
        & selected.available[:, 0]
        & selected.stateful[:, 0]
        & ~runtime.unsupported_mechanics
        & initialized
        & (current < state_count)
        & side_independent
        & unpaired
        & supported
    )
    target = success[batch_axis, table_state, 0]
    next_state = jnp.where(valid, target, current)
    next_states = runtime.stateful_state.at[batch_axis, runtime_slot].set(
        next_state
    )
    unsupported = intents & ~valid
    failure_bits = runtime.failure_bits | jnp.where(
        unsupported,
        jnp.uint32(SURROGATE_FAILURE_STATE_CHANGE_USE),
        jnp.uint32(0),
    )
    next_runtime = runtime._replace(
        stateful_state=next_states,
        failure_bits=failure_bits,
        unsupported_mechanics=(failure_bits != 0),
    )
    return SurrogateStateChangeResult(
        runtime=next_runtime,
        previous_state=current,
        next_state=next_state,
        accepted=valid,
        state_changed=valid & (next_state != current),
        unsupported=unsupported,
    )


__all__ = ["apply_surrogate_state_change_use"]
