"""Actor-legal projection of native or surrogate world perception channels."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.world.surrogate.channels import (
    CHANNEL_BLOCK_LIGHT_RGB,
    CHANNEL_ENVIRONMENT,
    CHANNEL_HEIGHTMAP,
    CHANNEL_SKY_LIGHT,
    CHANNEL_TINT_RGB,
    SURROGATE_PERCEPTION_CHANNEL_COUNT,
)
from hytalegym.jax.world.surrogate.types import (
    SurrogatePerceptionChannelResult,
)

Array = jax.Array

ACTOR_PERCEPTION_CHANNEL_EVIDENCE_SCHEMA = (
    "hytalerl_actor_perception_channel_evidence_v1"
)
ACTOR_PERCEPTION_CHANNEL_EVIDENCE_VERSION = 1
NATIVE_LIGHT_MAXIMUM = 15
NATIVE_CHANNEL_CAPTURE_DIAGNOSTIC_UNAVAILABLE = jnp.uint32(1)


class ActorPerceptionChannelEvidence(NamedTuple):
    """Raw actor-visible channels with independent validity masks."""

    available: Array
    channel_valid: Array
    heightmap_block_y: Array
    height_above_surface: Array
    sky_light: Array
    block_light_rgb: Array
    environment_code: Array
    tint_rgb: Array


def native_perception_channel_result_from_capture(
    capture,
    *,
    sample_shape: tuple[int, int],
) -> SurrogatePerceptionChannelResult:
    """Convert one validated host capture to the shared fixed-array shape.

    This is a host boundary, not a JIT entry point. ``sample_shape`` preserves
    the caller's batch/query layout; no sample is reordered or synthesized.
    """

    from hytalegym.worldgen.native_channels import (  # cycle-safe
        NativePerceptionChannelCapture,
    )

    if not isinstance(capture, NativePerceptionChannelCapture):
        raise TypeError(
            "capture must be a NativePerceptionChannelCapture"
        )
    if (
        not isinstance(sample_shape, tuple)
        or len(sample_shape) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in sample_shape
        )
    ):
        raise ValueError("sample_shape must be two positive integers")
    if sample_shape[0] * sample_shape[1] != capture.sample_count:
        raise ValueError(
            "sample_shape does not match the native capture sample count"
        )

    def shaped(value):
        array = jnp.asarray(value)
        return array.reshape(sample_shape + array.shape[1:])

    available = shaped(capture.available)
    channel_valid = shaped(capture.channel_valid)
    height = shaped(capture.heightmap_block_y)
    positions = shaped(capture.positions)
    height_valid = channel_valid[..., CHANNEL_HEIGHTMAP]
    height_above_surface = jnp.where(
        available & height_valid,
        positions[..., 1].astype(jnp.float32)
        - (height.astype(jnp.float32) + 1.0),
        0.0,
    )
    return SurrogatePerceptionChannelResult(
        available=available,
        diagnostics=jnp.where(
            available,
            jnp.uint32(0),
            NATIVE_CHANNEL_CAPTURE_DIAGNOSTIC_UNAVAILABLE,
        ),
        tile_index=jnp.full(sample_shape, -1, dtype=jnp.int32),
        channel_valid=channel_valid,
        heightmap_block_y=height,
        height_above_surface=height_above_surface,
        sky_light=shaped(capture.sky_light),
        block_light_rgb=shaped(capture.block_light_rgb),
        environment_code=shaped(capture.environment_code),
        tint_rgb=shaped(capture.tint_rgb),
    )


def native_actor_perception_channel_evidence(
    channels: SurrogatePerceptionChannelResult,
    actor_legal: Array,
) -> ActorPerceptionChannelEvidence:
    """Project bridge-derived channels after actor-legality gating."""

    return _actor_perception_channel_evidence(
        channels,
        actor_legal,
        provider="native",
    )


def surrogate_actor_perception_channel_evidence(
    channels: SurrogatePerceptionChannelResult,
    actor_legal: Array,
) -> ActorPerceptionChannelEvidence:
    """Project surrogate channels without relabelling them as native."""

    return _actor_perception_channel_evidence(
        channels,
        actor_legal,
        provider="surrogate",
    )


def actor_perception_channel_evidence_contract() -> dict[str, object]:
    """Return the provider-neutral actor-evidence boundary."""

    return {
        "schema": ACTOR_PERCEPTION_CHANNEL_EVIDENCE_SCHEMA,
        "version": ACTOR_PERCEPTION_CHANNEL_EVIDENCE_VERSION,
        "functions": {
            "native": "native_actor_perception_channel_evidence",
            "surrogate": "surrogate_actor_perception_channel_evidence",
        },
        "input": {
            "raw_shape": ["batch", "actor", "channel"],
            "actor_legal_shape": ["batch", "actor"],
            "actor_legal_authority": (
                "caller_supplied_certified_perception_or_self_visibility"
            ),
            "raw_channel_order": [
                "heightmap",
                "sky_light",
                "block_light_rgb",
                "environment",
                "tint_rgb",
            ],
        },
        "output": {
            "shape": ["batch", "actor"],
            "values": [
                "heightmap_block_y",
                "height_above_surface",
                "sky_light",
                "block_light_rgb",
                "environment_code",
                "tint_rgb",
            ],
            "channel_valid": "independent_per_actor_per_channel",
            "encoding": "raw_native_units_no_policy_normalization",
            "invalid": "zero_with_channel_valid_false",
        },
        "providers": {
            "native": {
                "provenance": "native_bridge_BlockChunk_accessors",
                "allowed_channels": [
                    "heightmap",
                    "sky_light",
                    "block_light_rgb",
                    "environment",
                    "tint_rgb",
                ],
                "runtime_readiness": (
                    "transported_independently_in_raw_channel_valid"
                ),
            },
            "surrogate": {
                "provenance": "surrogate_not_native_calibrated",
                "allowed_channels": [
                    "heightmap",
                    "sky_light",
                    "environment",
                ],
                "forced_unavailable": [
                    "block_light_rgb",
                    "tint_rgb",
                ],
            },
        },
        "ranges": {
            "sky_light": [0, NATIVE_LIGHT_MAXIMUM],
            "block_light_rgb": [0, NATIVE_LIGHT_MAXIMUM],
            "tint_rgb": [0, 255],
            "environment_code": "runtime_native_or_surrogate_provider_code",
        },
        "privilege": {
            "query_boundary": "privileged_world_state",
            "publication_boundary": "actor_legal_mask_before_any_value",
            "hidden_actor_invariance": (
                "all_values_and_validity_are_zero_independent_of_hidden_world"
            ),
            "diagnostics": "excluded_from_policy_evidence",
        },
        "fail_closed": [
            "invalid_or_non_finite_height",
            "light_outside_native_range",
            "raw_diagnostics_nonzero",
            "surrogate_native_only_channel_claim",
            "actor_not_legal",
        ],
        "consumer": (
            "combat_lane_may_add_this_evidence_in_one_versioned_observation_"
            "contract_move"
        ),
    }


def actor_perception_channel_evidence_contract_sha256() -> str:
    payload = json.dumps(
        actor_perception_channel_evidence_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _actor_perception_channel_evidence(
    channels: SurrogatePerceptionChannelResult,
    actor_legal: Array,
    *,
    provider: str,
) -> ActorPerceptionChannelEvidence:
    _validate_channel_shapes(channels)
    legal = jnp.asarray(actor_legal)
    if legal.dtype != jnp.bool_:
        raise TypeError("actor_legal must have boolean dtype")
    if legal.shape != channels.available.shape:
        raise ValueError("actor_legal must match channel batch and actor axes")

    height_finite = jnp.isfinite(channels.height_above_surface)
    sky_in_range = channels.sky_light <= jnp.uint8(NATIVE_LIGHT_MAXIMUM)
    block_in_range = jnp.all(
        channels.block_light_rgb <= jnp.uint8(NATIVE_LIGHT_MAXIMUM),
        axis=2,
    )
    value_valid = jnp.stack(
        (
            height_finite,
            sky_in_range,
            block_in_range,
            jnp.ones_like(height_finite),
            jnp.ones_like(height_finite),
        ),
        axis=2,
    )
    raw_available = channels.available & (channels.diagnostics == 0)
    valid = channels.channel_valid & value_valid & raw_available[..., None]
    if provider == "surrogate":
        allowed = jnp.asarray(
            (True, True, False, True, False),
            dtype=jnp.bool_,
        )
        unsupported_claim = jnp.any(
            channels.channel_valid & ~allowed,
            axis=2,
        )
        raw_available &= ~unsupported_claim
        valid &= allowed
    elif provider != "native":
        raise ValueError("provider must be native or surrogate")

    published = raw_available & legal
    valid &= published[..., None]
    height_valid = valid[..., CHANNEL_HEIGHTMAP]
    sky_valid = valid[..., CHANNEL_SKY_LIGHT]
    block_valid = valid[..., CHANNEL_BLOCK_LIGHT_RGB]
    environment_valid = valid[..., CHANNEL_ENVIRONMENT]
    tint_valid = valid[..., CHANNEL_TINT_RGB]
    return ActorPerceptionChannelEvidence(
        available=published,
        channel_valid=valid,
        heightmap_block_y=jnp.where(
            height_valid,
            channels.heightmap_block_y,
            jnp.int16(0),
        ),
        height_above_surface=jnp.where(
            height_valid,
            channels.height_above_surface,
            jnp.float32(0.0),
        ),
        sky_light=jnp.where(
            sky_valid,
            channels.sky_light,
            jnp.uint8(0),
        ),
        block_light_rgb=jnp.where(
            block_valid[..., None],
            channels.block_light_rgb,
            jnp.uint8(0),
        ),
        environment_code=jnp.where(
            environment_valid,
            channels.environment_code,
            jnp.int32(0),
        ),
        tint_rgb=jnp.where(
            tint_valid[..., None],
            channels.tint_rgb,
            jnp.uint8(0),
        ),
    )


def _validate_channel_shapes(
    channels: SurrogatePerceptionChannelResult,
) -> None:
    if not isinstance(channels, SurrogatePerceptionChannelResult):
        raise TypeError("channels must be a SurrogatePerceptionChannelResult")
    available = channels.available
    if available.ndim != 2 or available.shape[0] == 0:
        raise ValueError("channel batch must have non-empty shape [B,A]")
    shape = available.shape
    if channels.diagnostics.shape != shape:
        raise ValueError("channel diagnostics shape differs")
    if channels.channel_valid.shape != (
        *shape,
        SURROGATE_PERCEPTION_CHANNEL_COUNT,
    ):
        raise ValueError("channel_valid has an invalid shape")
    for name in (
        "heightmap_block_y",
        "height_above_surface",
        "sky_light",
        "environment_code",
    ):
        if getattr(channels, name).shape != shape:
            raise ValueError(f"{name} has an invalid shape")
    for name in ("block_light_rgb", "tint_rgb"):
        if getattr(channels, name).shape != (*shape, 3):
            raise ValueError(f"{name} has an invalid shape")


__all__ = [
    "ACTOR_PERCEPTION_CHANNEL_EVIDENCE_SCHEMA",
    "ACTOR_PERCEPTION_CHANNEL_EVIDENCE_VERSION",
    "NATIVE_CHANNEL_CAPTURE_DIAGNOSTIC_UNAVAILABLE",
    "ActorPerceptionChannelEvidence",
    "actor_perception_channel_evidence_contract",
    "actor_perception_channel_evidence_contract_sha256",
    "native_perception_channel_result_from_capture",
    "native_actor_perception_channel_evidence",
    "surrogate_actor_perception_channel_evidence",
]
