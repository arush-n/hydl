"""Actor-legal light and environment values aligned to geometry tokens."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.world.perception.channels import (
    ActorPerceptionChannelEvidence,
    actor_perception_channel_evidence_contract_sha256,
    native_actor_perception_channel_evidence,
    surrogate_actor_perception_channel_evidence,
)
from hytalegym.jax.world.surrogate.channels import (
    CHANNEL_BLOCK_LIGHT_RGB,
    CHANNEL_ENVIRONMENT,
    CHANNEL_SKY_LIGHT,
    CHANNEL_TINT_RGB,
)
from hytalegym.jax.world.surrogate.types import (
    SurrogatePerceptionChannelResult,
)
from hytalegym.jax.world.tokens import (
    WorldGeometryTokenObservation,
    world_geometry_token_contract_sha256,
)


Array = jax.Array

ACTOR_BLOCK_LIGHT_TOKEN_SCHEMA = "hytalerl_actor_block_light_tokens_v1"
ACTOR_BLOCK_LIGHT_TOKEN_VERSION = 1

LIGHT_PROVENANCE_NONE = 0
LIGHT_PROVENANCE_NATIVE = 1
LIGHT_PROVENANCE_SURROGATE = 2

ACTOR_BLOCK_LIGHT_DIAGNOSTIC_POSITION = jnp.uint32(1)
ACTOR_BLOCK_LIGHT_DIAGNOSTIC_SOURCE = jnp.uint32(1 << 1)
ACTOR_BLOCK_LIGHT_DIAGNOSTIC_NATIVE_FALLBACK = jnp.uint32(1 << 2)
ACTOR_BLOCK_LIGHT_FALLBACK_SCHEMA = (
    "hytalerl_actor_block_light_native_fallback_v1"
)
ACTOR_BLOCK_LIGHT_FALLBACK_VERSION = 1


class ActorBlockLightTokens(NamedTuple):
    """Raw light values aligned one-to-one with actor geometry tokens."""

    available: Array
    diagnostics: Array
    token_mask: Array
    light_valid: Array
    sky_light: Array
    block_light_rgb: Array
    tint_rgb: Array
    environment_valid: Array
    environment_code: Array
    provenance: Array


def actor_block_light_query_cells(
    tokens: WorldGeometryTokenObservation,
    actor_position: Array,
) -> Array:
    """Return integer block cells for the token-aligned channel request."""

    _validate_token_prefix(tokens)
    positions = jnp.asarray(actor_position, dtype=jnp.float32)
    if positions.shape != tokens.token_mask.shape[:2] + (3,):
        raise ValueError("actor_position must have shape [B, A, 3]")
    finite = jnp.all(jnp.isfinite(positions), axis=2)
    safe = jnp.where(finite[..., None], positions, 0.0)
    world_position = safe[:, :, None, :] + tokens.relative_position
    return jnp.floor(world_position).astype(jnp.int32)


def native_actor_block_light_tokens(
    tokens: WorldGeometryTokenObservation,
    actor_position: Array,
    queried_cell: Array,
    channels: SurrogatePerceptionChannelResult,
) -> ActorBlockLightTokens:
    """Publish bridge-derived token light after exact alignment checks."""

    return _actor_block_light_tokens(
        tokens,
        actor_position,
        queried_cell,
        channels,
        provider="native",
    )


def surrogate_actor_block_light_tokens(
    tokens: WorldGeometryTokenObservation,
    actor_position: Array,
    queried_cell: Array,
    channels: SurrogatePerceptionChannelResult,
) -> ActorBlockLightTokens:
    """Publish explicitly surrogate token light without provenance drift."""

    return _actor_block_light_tokens(
        tokens,
        actor_position,
        queried_cell,
        channels,
        provider="surrogate",
    )


def native_or_surrogate_actor_block_light_tokens(
    native: ActorBlockLightTokens,
    surrogate: ActorBlockLightTokens,
) -> ActorBlockLightTokens:
    """Prefer a complete native row, otherwise retain the surrogate row."""

    if not isinstance(native, ActorBlockLightTokens) or not isinstance(
        surrogate,
        ActorBlockLightTokens,
    ):
        raise TypeError("native and surrogate must be ActorBlockLightTokens")
    for name in ActorBlockLightTokens._fields:
        native_value = getattr(native, name)
        surrogate_value = getattr(surrogate, name)
        if (
            native_value.shape != surrogate_value.shape
            or native_value.dtype != surrogate_value.dtype
        ):
            raise ValueError(f"native and surrogate {name} differ")
    gate = native.available

    def select(native_value: Array, surrogate_value: Array) -> Array:
        shaped = gate.reshape(
            gate.shape + (1,) * (native_value.ndim - gate.ndim)
        )
        return jnp.where(shaped, native_value, surrogate_value)

    selected = ActorBlockLightTokens(
        *(select(left, right) for left, right in zip(native, surrogate))
    )
    used_fallback = ~native.available & surrogate.available
    return selected._replace(
        diagnostics=(
            selected.diagnostics
            | jnp.where(
                used_fallback,
                ACTOR_BLOCK_LIGHT_DIAGNOSTIC_NATIVE_FALLBACK,
                jnp.uint32(0),
            )
        )
    )


def actor_block_light_fallback_contract() -> dict[str, object]:
    """Return the row-level native/fallback selection contract."""

    return {
        "schema": ACTOR_BLOCK_LIGHT_FALLBACK_SCHEMA,
        "version": ACTOR_BLOCK_LIGHT_FALLBACK_VERSION,
        "source_contract_sha256": actor_block_light_token_contract_sha256(),
        "selection": "complete_native_row_else_complete_surrogate_row",
        "native_unavailable": "diagnosed_fallback_not_darkness",
        "mixed_channel_provenance": False,
    }


def actor_block_light_fallback_contract_sha256() -> str:
    payload = json.dumps(
        actor_block_light_fallback_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def actor_block_light_token_contract() -> dict[str, object]:
    """Return the additive token-light contract."""

    return {
        "schema": ACTOR_BLOCK_LIGHT_TOKEN_SCHEMA,
        "version": ACTOR_BLOCK_LIGHT_TOKEN_VERSION,
        "upstream": {
            "world_geometry_token_contract_sha256": (
                world_geometry_token_contract_sha256()
            ),
            "actor_perception_channel_evidence_contract_sha256": (
                actor_perception_channel_evidence_contract_sha256()
            ),
        },
        "shape": {
            "row": ["batch", "actor"],
            "token": ["batch", "actor", "token_capacity"],
            "light_valid": [
                "batch",
                "actor",
                "token_capacity",
                "sky_block_rgb_tint",
            ],
        },
        "alignment": (
            "floor_actor_plus_token_relative_position_matches_queried_cell"
        ),
        "values": {
            "sky_light": "raw_uint8_0_to_15",
            "block_light_rgb": "raw_uint8_0_to_15",
            "tint_rgb": "raw_uint8_0_to_255",
            "environment_code": (
                "raw_provider_code_not_portable_semantic_identity"
            ),
        },
        "validity": (
            "sky_block_rgb_tint_and_environment_remain_independent"
        ),
        "provenance": {
            str(LIGHT_PROVENANCE_NONE): "padding_or_unavailable",
            str(LIGHT_PROVENANCE_NATIVE): "native_bridge_BlockChunk",
            str(LIGHT_PROVENANCE_SURROGATE): (
                "surrogate_not_native_calibrated"
            ),
        },
        "policy_binding": (
            "additive_companion_to_geometry_tokens_not_in_learner_v3"
        ),
        "fail_closed": [
            "token_row_unavailable",
            "queried_cell_mismatch",
            "required_sample_unavailable",
            "upstream_channel_invalid",
        ],
    }


def actor_block_light_token_contract_sha256() -> str:
    payload = json.dumps(
        actor_block_light_token_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _actor_block_light_tokens(
    tokens: WorldGeometryTokenObservation,
    actor_position: Array,
    queried_cell: Array,
    channels: SurrogatePerceptionChannelResult,
    *,
    provider: str,
) -> ActorBlockLightTokens:
    _validate_token_prefix(tokens)
    shape = tokens.token_mask.shape
    batch, actors, capacity = shape
    queried = jnp.asarray(queried_cell, dtype=jnp.int32)
    expected = actor_block_light_query_cells(tokens, actor_position)
    if queried.shape != shape + (3,):
        raise ValueError(
            "queried_cell must have shape [B, A, token_capacity, 3]"
        )
    flat_legal = (
        tokens.token_mask & tokens.available[..., None]
    ).reshape(batch, actors * capacity)
    if provider == "native":
        evidence = native_actor_perception_channel_evidence(
            channels,
            flat_legal,
        )
        provenance_value = LIGHT_PROVENANCE_NATIVE
    elif provider == "surrogate":
        evidence = surrogate_actor_perception_channel_evidence(
            channels,
            flat_legal,
        )
        provenance_value = LIGHT_PROVENANCE_SURROGATE
    else:
        raise ValueError("provider must be native or surrogate")
    _validate_flat_evidence(evidence, batch, actors, capacity)

    def shaped(value: Array) -> Array:
        return value.reshape(shape + value.shape[2:])

    required = tokens.token_mask & tokens.available[..., None]
    position_mismatch = required & jnp.any(queried != expected, axis=3)
    source_missing = required & ~shaped(evidence.available)
    row_available = (
        tokens.available
        & ~jnp.any(position_mismatch | source_missing, axis=2)
    )
    mask = required & row_available[..., None]
    valid = shaped(evidence.channel_valid)

    def masked(value: Array, valid_mask: Array) -> Array:
        value = shaped(value)
        gate = valid_mask.reshape(
            valid_mask.shape + (1,) * (value.ndim - valid_mask.ndim)
        )
        return jnp.where(gate, value, jnp.zeros_like(value))

    sky_valid = mask & valid[..., CHANNEL_SKY_LIGHT]
    block_valid = mask & valid[..., CHANNEL_BLOCK_LIGHT_RGB]
    tint_valid = mask & valid[..., CHANNEL_TINT_RGB]
    environment_valid = mask & valid[..., CHANNEL_ENVIRONMENT]
    return ActorBlockLightTokens(
        available=row_available,
        diagnostics=(
            tokens.diagnostics
            | jnp.where(
                jnp.any(position_mismatch, axis=2),
                ACTOR_BLOCK_LIGHT_DIAGNOSTIC_POSITION,
                jnp.uint32(0),
            )
            | jnp.where(
                jnp.any(source_missing, axis=2),
                ACTOR_BLOCK_LIGHT_DIAGNOSTIC_SOURCE,
                jnp.uint32(0),
            )
        ),
        token_mask=mask,
        light_valid=jnp.stack(
            (sky_valid, block_valid, tint_valid),
            axis=3,
        ),
        sky_light=masked(evidence.sky_light, sky_valid),
        block_light_rgb=masked(evidence.block_light_rgb, block_valid),
        tint_rgb=masked(evidence.tint_rgb, tint_valid),
        environment_valid=environment_valid,
        environment_code=masked(
            evidence.environment_code,
            environment_valid,
        ),
        provenance=jnp.where(
            mask,
            jnp.uint8(provenance_value),
            jnp.uint8(LIGHT_PROVENANCE_NONE),
        ),
    )


def _validate_token_prefix(tokens: WorldGeometryTokenObservation) -> None:
    if not isinstance(tokens, WorldGeometryTokenObservation):
        raise TypeError("tokens must be a WorldGeometryTokenObservation")
    shape = tokens.token_mask.shape
    if len(shape) != 3 or shape[2] <= 0:
        raise ValueError("tokens must have shape [B, A, token_capacity]")
    if tokens.available.shape != shape[:2]:
        raise ValueError("token row shape differs from token mask")
    if tokens.relative_position.shape != shape + (3,):
        raise ValueError("token relative positions have an invalid shape")


def _validate_flat_evidence(
    evidence: ActorPerceptionChannelEvidence,
    batch: int,
    actors: int,
    capacity: int,
) -> None:
    if evidence.available.shape != (batch, actors * capacity):
        raise ValueError(
            "channels must be flattened in actor-major token order"
        )


__all__ = [
    "ACTOR_BLOCK_LIGHT_DIAGNOSTIC_POSITION",
    "ACTOR_BLOCK_LIGHT_DIAGNOSTIC_SOURCE",
    "ACTOR_BLOCK_LIGHT_DIAGNOSTIC_NATIVE_FALLBACK",
    "ACTOR_BLOCK_LIGHT_FALLBACK_SCHEMA",
    "ACTOR_BLOCK_LIGHT_FALLBACK_VERSION",
    "ACTOR_BLOCK_LIGHT_TOKEN_SCHEMA",
    "ACTOR_BLOCK_LIGHT_TOKEN_VERSION",
    "ActorBlockLightTokens",
    "LIGHT_PROVENANCE_NATIVE",
    "LIGHT_PROVENANCE_NONE",
    "LIGHT_PROVENANCE_SURROGATE",
    "actor_block_light_query_cells",
    "actor_block_light_fallback_contract",
    "actor_block_light_fallback_contract_sha256",
    "actor_block_light_token_contract",
    "actor_block_light_token_contract_sha256",
    "native_actor_block_light_tokens",
    "native_or_surrogate_actor_block_light_tokens",
    "surrogate_actor_block_light_tokens",
]
