"""Actor-only policy features for World's token-aligned light evidence."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.world import ActorBlockLightTokens


Array = jax.Array

ACTOR_LIGHT_POLICY_FEATURES = (
    "sky_valid",
    "block_light_rgb_valid",
    "tint_rgb_valid",
    "sky_light",
    "block_light_red",
    "block_light_green",
    "block_light_blue",
    "tint_red",
    "tint_green",
    "tint_blue",
)
ACTOR_LIGHT_POLICY_FEATURE_SIZE = len(ACTOR_LIGHT_POLICY_FEATURES)

_LIGHT_NORMALIZED_F32 = jnp.asarray(
    tuple(value / 15.0 for value in range(16)),
    dtype=jnp.float32,
)
_TINT_NORMALIZED_F32 = jnp.asarray(
    tuple(value / 255.0 for value in range(256)),
    dtype=jnp.float32,
)


class ActorLightPolicyTokens(NamedTuple):
    """Policy-only light rows with diagnostics and source identity removed."""

    available: Array
    token_f32: Array
    token_mask: Array


def actor_light_policy_flat_size(token_capacity: int) -> int:
    """Return the append width while reusing geometry's existing mask."""

    if (
        isinstance(token_capacity, bool)
        or not isinstance(token_capacity, int)
        or token_capacity < 1
    ):
        raise ValueError("token_capacity must be a positive integer")
    return token_capacity * ACTOR_LIGHT_POLICY_FEATURE_SIZE + 1


def empty_actor_light_policy_tokens(
    batch: int,
    token_capacity: int,
) -> ActorLightPolicyTokens:
    """Return fixed-shape unavailable light rows."""

    for name, value in (
        ("batch", batch),
        ("token_capacity", token_capacity),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return ActorLightPolicyTokens(
        available=jnp.zeros((batch,), dtype=jnp.bool_),
        token_f32=jnp.zeros(
            (
                batch,
                token_capacity,
                ACTOR_LIGHT_POLICY_FEATURE_SIZE,
            ),
            dtype=jnp.float32,
        ),
        token_mask=jnp.zeros(
            (batch, token_capacity),
            dtype=jnp.bool_,
        ),
    )


def encode_actor_light_policy_tokens(
    source: ActorBlockLightTokens,
    *,
    actor_index: int = 0,
) -> ActorLightPolicyTokens:
    """Project one actor's legal light evidence into normalized policy rows.

    The projection deliberately reads only the public light values, their
    independent validity bits, and the geometry-aligned token mask. World
    diagnostics, provenance, environment codes, and every other actor remain
    outside the returned policy type.
    """

    _validate_source_shapes(source, actor_index)

    source_available = source.available[:, actor_index]
    source_mask = source.token_mask[:, actor_index]
    source_valid = source.light_valid[:, actor_index]
    sky_raw = source.sky_light[:, actor_index]
    block_raw = source.block_light_rgb[:, actor_index]
    tint_raw = source.tint_rgb[:, actor_index]

    active = source_mask & source_available[:, None]
    malformed = active & (
        (source_valid[..., 0] & (sky_raw > jnp.uint8(15)))
        | (
            source_valid[..., 1]
            & jnp.any(block_raw > jnp.uint8(15), axis=2)
        )
    )
    available = source_available & ~jnp.any(malformed, axis=1)
    token_mask = source_mask & available[:, None]
    valid = source_valid & token_mask[..., None]

    sky = jnp.where(
        valid[..., 0],
        jnp.take(
            _LIGHT_NORMALIZED_F32,
            sky_raw.astype(jnp.int32),
        ),
        jnp.float32(0.0),
    )
    block = jnp.where(
        valid[..., 1, None],
        jnp.take(
            _LIGHT_NORMALIZED_F32,
            block_raw.astype(jnp.int32),
        ),
        jnp.float32(0.0),
    )
    tint = jnp.where(
        valid[..., 2, None],
        jnp.take(
            _TINT_NORMALIZED_F32,
            tint_raw.astype(jnp.int32),
        ),
        jnp.float32(0.0),
    )
    token_f32 = jnp.concatenate(
        (
            valid.astype(jnp.float32),
            sky[..., None],
            block,
            tint,
        ),
        axis=2,
    )
    token_f32 = jnp.where(
        token_mask[..., None],
        token_f32,
        jnp.zeros_like(token_f32),
    )
    return ActorLightPolicyTokens(
        available=available,
        token_f32=token_f32,
        token_mask=token_mask,
    )


def align_actor_light_policy_tokens(
    source: ActorLightPolicyTokens,
    geometry_token_mask: Array,
    geometry_available: Array,
    actor_valid: Array,
) -> ActorLightPolicyTokens:
    """Fail a light row closed unless it exactly accompanies geometry tokens."""

    if not isinstance(source, ActorLightPolicyTokens):
        raise TypeError("source must be ActorLightPolicyTokens")
    token_mask = jnp.asarray(geometry_token_mask)
    geometry_row = jnp.asarray(geometry_available)
    valid_row = jnp.asarray(actor_valid)
    if token_mask.ndim != 2:
        raise ValueError("geometry_token_mask must have shape [B, token]")
    batch, capacity = token_mask.shape
    if source.available.shape != (batch,):
        raise ValueError("light availability must have shape [B]")
    if source.token_mask.shape != (batch, capacity):
        raise ValueError("light and geometry token capacities must match")
    if source.token_f32.shape != (
        batch,
        capacity,
        ACTOR_LIGHT_POLICY_FEATURE_SIZE,
    ):
        raise ValueError("light token_f32 shape drift")
    for name, value in (
        ("geometry_token_mask", token_mask),
        ("geometry_available", geometry_row),
        ("actor_valid", valid_row),
        ("light available", source.available),
        ("light token_mask", source.token_mask),
    ):
        if value.dtype != jnp.bool_:
            raise TypeError(f"{name} must be boolean")
    if geometry_row.shape != (batch,) or valid_row.shape != (batch,):
        raise ValueError("geometry availability and actor validity must have shape [B]")
    if source.token_f32.dtype != jnp.float32:
        raise TypeError("light token_f32 must be float32")

    aligned = jnp.all(source.token_mask == token_mask, axis=1)
    available = source.available & geometry_row & valid_row & aligned
    mask = source.token_mask & token_mask & available[:, None]
    return ActorLightPolicyTokens(
        available=available,
        token_f32=jnp.where(
            mask[..., None],
            source.token_f32,
            jnp.float32(0.0),
        ),
        token_mask=mask,
    )


def _validate_source_shapes(
    source: ActorBlockLightTokens,
    actor_index: int,
) -> None:
    if not isinstance(source, ActorBlockLightTokens):
        raise TypeError("source must be ActorBlockLightTokens")
    if isinstance(actor_index, bool) or not isinstance(actor_index, int):
        raise TypeError("actor_index must be an integer")
    if source.available.ndim != 2 or source.available.shape[1] < 1:
        raise ValueError("source available must have shape [B, actor]")
    batch, actors = source.available.shape
    if not 0 <= actor_index < actors:
        raise ValueError("actor_index is outside the source actor axis")
    if source.token_mask.ndim != 3 or source.token_mask.shape[:2] != (
        batch,
        actors,
    ):
        raise ValueError("source token_mask must have shape [B, actor, token]")
    capacity = source.token_mask.shape[2]
    if capacity < 1:
        raise ValueError("source token capacity must be positive")
    token_shape = (batch, actors, capacity)
    expected = {
        "light_valid": token_shape + (3,),
        "sky_light": token_shape,
        "block_light_rgb": token_shape + (3,),
        "tint_rgb": token_shape + (3,),
    }
    for name, shape in expected.items():
        if getattr(source, name).shape != shape:
            raise ValueError(f"source {name} must have shape {shape}")
    for name in ("available", "token_mask", "light_valid"):
        if getattr(source, name).dtype != jnp.bool_:
            raise TypeError(f"source {name} must be boolean")
    for name in ("sky_light", "block_light_rgb", "tint_rgb"):
        if getattr(source, name).dtype != jnp.uint8:
            raise TypeError(f"source {name} must be uint8")


__all__ = [
    "ACTOR_LIGHT_POLICY_FEATURES",
    "ACTOR_LIGHT_POLICY_FEATURE_SIZE",
    "ActorLightPolicyTokens",
    "actor_light_policy_flat_size",
    "align_actor_light_policy_tokens",
    "empty_actor_light_policy_tokens",
    "encode_actor_light_policy_tokens",
]
