"""Dense actor-legal light tensors for convolutional world encoders."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.perception.lighting.tokens import (
    LIGHT_PROVENANCE_NATIVE,
    LIGHT_PROVENANCE_NONE,
    LIGHT_PROVENANCE_SURROGATE,
)
from hytalegym.jax.world.perception.channels import (
    ActorPerceptionChannelEvidence,
    NATIVE_LIGHT_MAXIMUM,
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


Array = jax.Array

ACTOR_LIGHT_VOXEL_SCHEMA = "hytalerl_actor_light_voxels_v1"
ACTOR_LIGHT_VOXEL_VERSION = 1
ACTOR_LIGHT_FEATURE_COUNT = 7
ACTOR_LIGHT_VALIDITY_COUNT = 3

ACTOR_LIGHT_VOXEL_DIAGNOSTIC_SOURCE = jnp.uint32(1)

_LIGHT_NORMALIZATION = jnp.asarray(
    np.linspace(0.0, 1.0, NATIVE_LIGHT_MAXIMUM + 1, dtype=np.float32)
)
_TINT_NORMALIZATION = jnp.asarray(
    np.linspace(0.0, 1.0, 256, dtype=np.float32)
)


class ActorLightVoxelObservation(NamedTuple):
    """Dense Y/X/Z light tensor with independent channel validity."""

    available: Array
    diagnostics: Array
    voxel_mask: Array
    light_valid: Array
    light_f32: Array
    environment_valid: Array
    environment_code: Array
    provenance: Array


def dense_actor_light_query_cells(
    actor_position: Array,
    *,
    horizontal_radius: int,
    below: int,
    above: int,
) -> Array:
    """Return a static [B,A,Y,X,Z,3] block-cell stencil."""

    radius = _nonnegative_static_int(horizontal_radius, "horizontal_radius")
    lower = _nonnegative_static_int(below, "below")
    upper = _nonnegative_static_int(above, "above")
    positions = jnp.asarray(actor_position, dtype=jnp.float32)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("actor_position must have shape [B, A, 3]")
    finite = jnp.all(jnp.isfinite(positions), axis=2)
    origin = jnp.floor(
        jnp.where(finite[..., None], positions, 0.0)
    ).astype(jnp.int32)
    y = jnp.arange(-lower, upper + 1, dtype=jnp.int32)
    x = jnp.arange(-radius, radius + 1, dtype=jnp.int32)
    z = jnp.arange(-radius, radius + 1, dtype=jnp.int32)
    yy, xx, zz = jnp.meshgrid(y, x, z, indexing="ij")
    offsets = jnp.stack((xx, yy, zz), axis=3)
    return origin[:, :, None, None, None, :] + offsets


def native_actor_light_voxels(
    channels: SurrogatePerceptionChannelResult,
    actor_legal: Array,
) -> ActorLightVoxelObservation:
    """Project bridge-derived light into a normalized CNN tensor."""

    return _actor_light_voxels(channels, actor_legal, provider="native")


def surrogate_actor_light_voxels(
    channels: SurrogatePerceptionChannelResult,
    actor_legal: Array,
) -> ActorLightVoxelObservation:
    """Project surrogate sky while keeping native-only planes invalid."""

    return _actor_light_voxels(channels, actor_legal, provider="surrogate")


def actor_light_voxel_contract() -> dict[str, object]:
    """Return the dense actor-light observation contract."""

    return {
        "schema": ACTOR_LIGHT_VOXEL_SCHEMA,
        "version": ACTOR_LIGHT_VOXEL_VERSION,
        "upstream_actor_evidence_contract_sha256": (
            actor_perception_channel_evidence_contract_sha256()
        ),
        "query": {
            "helper": "dense_actor_light_query_cells",
            "layout": ["batch", "actor", "y", "x", "z", "xyz"],
            "extent": "caller_selected_static_nonnegative_integers",
        },
        "output": {
            "voxel_mask": ["batch", "actor", "y", "x", "z"],
            "light_valid_order": ["sky", "block_rgb", "tint"],
            "light_f32_order": [
                "sky",
                "block_red",
                "block_green",
                "block_blue",
                "tint_red",
                "tint_green",
                "tint_blue",
            ],
            "normalization": {
                "sky_and_block_rgb": "divide_by_15",
                "tint_rgb": "divide_by_255",
            },
            "environment_code": (
                "separate_raw_int32_with_independent_validity"
            ),
        },
        "privilege": {
            "publication": "caller_actor_legal_per_voxel_before_projection",
            "hidden_values": "zero_and_bit_exactly_invariant",
        },
        "provenance": {
            str(LIGHT_PROVENANCE_NATIVE): "native_bridge_BlockChunk",
            str(LIGHT_PROVENANCE_SURROGATE): (
                "surrogate_not_native_calibrated"
            ),
        },
        "consumer": (
            "optional_dense_CNN_input_not_in_current_learner_v3_contract"
        ),
        "fail_closed": [
            "required_legal_voxel_unavailable",
            "invalid_upstream_channel",
            "invalid_static_extent",
        ],
    }


def actor_light_voxel_contract_sha256() -> str:
    payload = json.dumps(
        actor_light_voxel_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _actor_light_voxels(
    channels: SurrogatePerceptionChannelResult,
    actor_legal: Array,
    *,
    provider: str,
) -> ActorLightVoxelObservation:
    legal = jnp.asarray(actor_legal)
    if legal.dtype != jnp.bool_:
        raise TypeError("actor_legal must have boolean dtype")
    if legal.ndim != 5 or any(size <= 0 for size in legal.shape):
        raise ValueError("actor_legal must have shape [B, A, Y, X, Z]")
    batch, actors, y_size, x_size, z_size = legal.shape
    sample_count = y_size * x_size * z_size
    flat_legal = legal.reshape(batch, actors * sample_count)
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
    _validate_flat_evidence(evidence, batch, actors, sample_count)
    shape = legal.shape

    def shaped(value: Array) -> Array:
        return value.reshape(shape + value.shape[2:])

    source_available = shaped(evidence.available)
    missing = legal & ~source_available
    row_available = ~jnp.any(missing, axis=(2, 3, 4))
    voxel_mask = legal & row_available[..., None, None, None]
    valid = shaped(evidence.channel_valid)
    sky_valid = voxel_mask & valid[..., CHANNEL_SKY_LIGHT]
    block_valid = voxel_mask & valid[..., CHANNEL_BLOCK_LIGHT_RGB]
    tint_valid = voxel_mask & valid[..., CHANNEL_TINT_RGB]
    environment_valid = voxel_mask & valid[..., CHANNEL_ENVIRONMENT]

    sky = jnp.take(
        _LIGHT_NORMALIZATION,
        shaped(evidence.sky_light).astype(jnp.int32),
    )[..., None]
    block = jnp.take(
        _LIGHT_NORMALIZATION,
        shaped(evidence.block_light_rgb).astype(jnp.int32),
    )
    tint = jnp.take(
        _TINT_NORMALIZATION,
        shaped(evidence.tint_rgb).astype(jnp.int32),
    )
    light = jnp.concatenate(
        (sky, block, tint),
        axis=5,
    )
    light = jnp.where(voxel_mask[..., None], light, 0.0)
    environment = shaped(evidence.environment_code)
    return ActorLightVoxelObservation(
        available=row_available,
        diagnostics=jnp.where(
            jnp.any(missing, axis=(2, 3, 4)),
            ACTOR_LIGHT_VOXEL_DIAGNOSTIC_SOURCE,
            jnp.uint32(0),
        ),
        voxel_mask=voxel_mask,
        light_valid=jnp.stack(
            (sky_valid, block_valid, tint_valid),
            axis=5,
        ),
        light_f32=light,
        environment_valid=environment_valid,
        environment_code=jnp.where(
            environment_valid,
            environment,
            jnp.int32(0),
        ),
        provenance=jnp.where(
            voxel_mask,
            jnp.uint8(provenance_value),
            jnp.uint8(LIGHT_PROVENANCE_NONE),
        ),
    )


def _validate_flat_evidence(
    evidence: ActorPerceptionChannelEvidence,
    batch: int,
    actors: int,
    sample_count: int,
) -> None:
    if evidence.available.shape != (batch, actors * sample_count):
        raise ValueError(
            "channels must be flattened in actor-major Y/X/Z order"
        )


def _nonnegative_static_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


__all__ = [
    "ACTOR_LIGHT_FEATURE_COUNT",
    "ACTOR_LIGHT_VALIDITY_COUNT",
    "ACTOR_LIGHT_VOXEL_DIAGNOSTIC_SOURCE",
    "ACTOR_LIGHT_VOXEL_SCHEMA",
    "ACTOR_LIGHT_VOXEL_VERSION",
    "ActorLightVoxelObservation",
    "actor_light_voxel_contract",
    "actor_light_voxel_contract_sha256",
    "dense_actor_light_query_cells",
    "native_actor_light_voxels",
    "surrogate_actor_light_voxels",
]
