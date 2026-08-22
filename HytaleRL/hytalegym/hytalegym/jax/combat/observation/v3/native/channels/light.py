"""Host binding from native token cells to actor-safe light policy rows."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat.observation.v3.tokens.light import (
    ActorLightPolicyTokens,
    encode_actor_light_policy_tokens,
)
from hytalegym.jax.world import (
    WorldGeometryTokenObservation,
    actor_block_light_query_cells,
    native_actor_block_light_tokens,
    native_perception_channel_result_from_capture,
)


NativePerceptionCapture = Callable[[np.ndarray], Any]


def capture_native_actor_light_policy_tokens(
    tokens: WorldGeometryTokenObservation,
    actor_position,
    capture: NativePerceptionCapture,
    *,
    actor_index: int = 0,
) -> ActorLightPolicyTokens:
    """Query the existing bridge endpoint at the exact geometry-token cells.

    This function is host-only: it performs one bounded bridge round-trip,
    verifies the echoed positions, then delegates all value/validity semantics
    to World's public native producer before removing provenance and
    diagnostics for the policy.
    """

    if not callable(capture):
        raise TypeError("capture must be callable")
    positions = jnp.asarray(actor_position, dtype=jnp.float32)
    queried = actor_block_light_query_cells(tokens, positions)
    queried_host = np.asarray(queried, dtype=np.int32)
    flat = queried_host.reshape((-1, 3))
    captured = capture(flat)
    echoed = np.asarray(getattr(captured, "positions", None), dtype=np.int32)
    if echoed.shape != flat.shape or not np.array_equal(echoed, flat):
        raise ValueError("native perception-channel positions changed")
    batch, actors, capacity = tokens.token_mask.shape
    channels = native_perception_channel_result_from_capture(
        captured,
        sample_shape=(batch, actors * capacity),
    )
    source = native_actor_block_light_tokens(
        tokens,
        positions,
        queried,
        channels,
    )
    return encode_actor_light_policy_tokens(
        source,
        actor_index=actor_index,
    )


__all__ = [
    "NativePerceptionCapture",
    "capture_native_actor_light_policy_tokens",
]
