"""Shared actor-safe encoders for structured action candidates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.observation.v3.policy.candidates.recipes import (
    RecipeCandidatePolicyView,
    recipe_candidate_policy_staging_contract_sha256,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.contract import (
    RECIPE_CANDIDATE_EMBEDDING_SIZE,
    RECIPE_CANDIDATE_INPUT_FEATURE_SIZE,
    RECIPE_CANDIDATE_INT32_BIT_COUNT,
    RECIPE_CANDIDATE_OUTPUT_FEATURE_SIZE,
    RECIPE_CANDIDATE_REQUIREMENT_FEATURE_SIZE,
    RECIPE_CANDIDATE_SCALAR_SIZE,
    recipe_candidate_encoding_contract_sha256,
)
from hytalegym.jax.crafting.contract import (
    MAX_BENCH_REQUIREMENTS,
    MAX_INGREDIENTS,
    MAX_OUTPUTS,
)
from hytalegym.jax.policy import DenseParams
from hytalegym.jax.world import ACTOR_RECIPE_CANDIDATE_CAPACITY


Array = jax.Array

_INT32_BIT_COUNT = RECIPE_CANDIDATE_INT32_BIT_COUNT
_INPUT_FEATURE_SIZE = RECIPE_CANDIDATE_INPUT_FEATURE_SIZE
_OUTPUT_FEATURE_SIZE = RECIPE_CANDIDATE_OUTPUT_FEATURE_SIZE
_REQUIREMENT_FEATURE_SIZE = RECIPE_CANDIDATE_REQUIREMENT_FEATURE_SIZE
_CANDIDATE_SCALAR_SIZE = RECIPE_CANDIDATE_SCALAR_SIZE

RECIPE_CANDIDATE_ENCODER_STAGING_SCHEMA = (
    "hytalerl_recipe_candidate_encoder_staging_v1"
)
RECIPE_CANDIDATE_ENCODER_STAGING_VERSION = 1


@dataclass(frozen=True)
class RecipeCandidateEncoderConfig:
    """Static learned width shared by every recipe and nested row."""

    embedding_size: int = RECIPE_CANDIDATE_EMBEDDING_SIZE

    def __post_init__(self) -> None:
        if (
            isinstance(self.embedding_size, bool)
            or not isinstance(self.embedding_size, int)
            or self.embedding_size < 1
        ):
            raise ValueError("embedding_size must be a positive integer")


class RecipeCandidateEncoderParams(NamedTuple):
    """Learned projections shared over candidate and nested-slot axes."""

    input_projection: DenseParams
    output_projection: DenseParams
    requirement_projection: DenseParams
    candidate_projection: DenseParams


class RecipeCandidateEncoding(NamedTuple):
    """Bounded per-candidate embeddings with actor legality retained."""

    available: Array
    candidate_mask: Array
    candidate_embedding: Array


def initialize_recipe_candidate_encoder(
    key: Array,
    config: RecipeCandidateEncoderConfig | None = None,
) -> RecipeCandidateEncoderParams:
    """Initialize one permutation-equivariant hierarchical encoder."""

    selected = _normalize_config(config)
    keys = jax.random.split(key, 4)
    size = selected.embedding_size
    return RecipeCandidateEncoderParams(
        input_projection=_dense(keys[0], _INPUT_FEATURE_SIZE, size),
        output_projection=_dense(keys[1], _OUTPUT_FEATURE_SIZE, size),
        requirement_projection=_dense(
            keys[2],
            _REQUIREMENT_FEATURE_SIZE,
            size,
        ),
        candidate_projection=_dense(
            keys[3],
            3 * size + _CANDIDATE_SCALAR_SIZE,
            size,
        ),
    )


def encode_recipe_candidates(
    params: RecipeCandidateEncoderParams,
    view: RecipeCandidatePolicyView,
) -> RecipeCandidateEncoding:
    """Encode structured recipe meaning without global recipe identities.

    Categorical int32/uint32 fields expand as little-endian bits rather than
    becoming ordered float scalars. Numeric quantities use a bounded monotonic
    transform. Shared projections and masked means make nested-slot and
    candidate-slot permutations equivariant.
    """

    if not isinstance(params, RecipeCandidateEncoderParams):
        raise TypeError("params must be RecipeCandidateEncoderParams")
    if not isinstance(view, RecipeCandidatePolicyView):
        raise TypeError("view must be RecipeCandidatePolicyView")
    _validate_view(view)

    candidate_mask = view.candidate_mask & view.available[:, None]
    policy = view.policy
    input_mask = policy.input_mask & candidate_mask[..., None]
    output_mask = policy.output_mask & candidate_mask[..., None]
    requirement_mask = policy.requirement_mask & candidate_mask[..., None]

    input_features = jnp.concatenate(
        (
            _int32_bits(policy.input_item_id),
            _int32_bits(policy.input_resource_type_id),
            _bounded_nonnegative(policy.input_quantity)[..., None],
            policy.input_metadata_required[..., None].astype(jnp.float32),
            _hash_bits(policy.input_metadata_hash),
        ),
        axis=-1,
    )
    output_features = jnp.concatenate(
        (
            _int32_bits(policy.output_item_id),
            _bounded_nonnegative(policy.output_quantity)[..., None],
            _hash_bits(policy.output_metadata_hash),
        ),
        axis=-1,
    )
    requirement_features = jnp.concatenate(
        (
            _int32_bits(policy.requirement_bench_type),
            _hash_bits(policy.requirement_bench_id_hash),
            _bounded_nonnegative(policy.requirement_tier_level)[..., None],
        ),
        axis=-1,
    )

    input_embedding = _masked_mean(
        jnp.tanh(_linear(params.input_projection, input_features)),
        input_mask,
    )
    output_embedding = _masked_mean(
        jnp.tanh(_linear(params.output_projection, output_features)),
        output_mask,
    )
    requirement_embedding = _masked_mean(
        jnp.tanh(
            _linear(
                params.requirement_projection,
                requirement_features,
            )
        ),
        requirement_mask,
    )
    candidate_scalars = jnp.stack(
        (
            policy.knowledge_required.astype(jnp.float32),
            _bounded_nonnegative(policy.required_memories_level),
            _bounded_nonnegative(policy.time_seconds),
            jnp.sum(input_mask, axis=-1, dtype=jnp.float32)
            / jnp.float32(MAX_INGREDIENTS),
            jnp.sum(output_mask, axis=-1, dtype=jnp.float32)
            / jnp.float32(MAX_OUTPUTS),
            jnp.sum(requirement_mask, axis=-1, dtype=jnp.float32)
            / jnp.float32(MAX_BENCH_REQUIREMENTS),
        ),
        axis=-1,
    )
    candidate_features = jnp.concatenate(
        (
            input_embedding,
            output_embedding,
            requirement_embedding,
            candidate_scalars,
        ),
        axis=-1,
    )
    candidate_embedding = jnp.tanh(
        _linear(params.candidate_projection, candidate_features)
    )
    return RecipeCandidateEncoding(
        available=view.available,
        candidate_mask=candidate_mask,
        candidate_embedding=jnp.where(
            candidate_mask[..., None],
            candidate_embedding,
            jnp.float32(0.0),
        ),
    )


def recipe_candidate_encoder_staging_contract_manifest(
    config: RecipeCandidateEncoderConfig | None = None,
) -> dict[str, object]:
    """Describe the published implementation of the stable encoding contract."""

    selected = _normalize_config(config)
    return {
        "schema": RECIPE_CANDIDATE_ENCODER_STAGING_SCHEMA,
        "version": RECIPE_CANDIDATE_ENCODER_STAGING_VERSION,
        "source_consumer_sha256": (
            recipe_candidate_policy_staging_contract_sha256()
        ),
        "encoding_contract_sha256": (
            recipe_candidate_encoding_contract_sha256()
        ),
        "candidate_capacity": ACTOR_RECIPE_CANDIDATE_CAPACITY,
        "embedding_size": selected.embedding_size,
        "categorical_encoding": (
            "little_endian_32_bit_expansion_never_ordered_float_ids"
        ),
        "numeric_encoding": "x_over_one_plus_x_nonnegative",
        "nested_reduction": "shared_projection_masked_mean",
        "candidate_axis": "shared_projection_permutation_equivariant",
        "masked_values": "bit_exact_zero",
        "excluded": [
            "native_recipe_string_id",
            "global_recipe_index",
            "recipe_identity_hash",
            "diagnostics",
            "source_counts",
            "capacity_overflow",
        ],
        "publication": "published_in_policy_v6_actor_candidate_append",
    }


def recipe_candidate_encoder_staging_contract_sha256(
    config: RecipeCandidateEncoderConfig | None = None,
) -> str:
    """Return the selected staged encoder identity."""

    payload = json.dumps(
        recipe_candidate_encoder_staging_contract_manifest(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _validate_view(view: RecipeCandidatePolicyView) -> None:
    if view.available.ndim != 1:
        raise ValueError("view.available must have shape [B]")
    batch = view.available.shape[0]
    candidate_shape = (batch, ACTOR_RECIPE_CANDIDATE_CAPACITY)
    if view.available.dtype != jnp.bool_:
        raise TypeError("view.available must have dtype bool")
    if (
        view.candidate_mask.shape != candidate_shape
        or view.candidate_mask.dtype != jnp.bool_
    ):
        raise ValueError("view.candidate_mask has the wrong shape or dtype")
    expected_prefixes = {
        "input_mask": candidate_shape + (MAX_INGREDIENTS,),
        "output_mask": candidate_shape + (MAX_OUTPUTS,),
        "requirement_mask": candidate_shape + (MAX_BENCH_REQUIREMENTS,),
        "knowledge_required": candidate_shape,
        "required_memories_level": candidate_shape,
        "time_seconds": candidate_shape,
    }
    for name, expected in expected_prefixes.items():
        if getattr(view.policy, name).shape != expected:
            raise ValueError(f"view.policy.{name} has the wrong shape")


def _int32_bits(value: Array) -> Array:
    raw = jnp.asarray(value).astype(jnp.uint32)
    shifts = jnp.arange(_INT32_BIT_COUNT, dtype=jnp.uint32)
    return (
        (raw[..., None] >> shifts) & jnp.uint32(1)
    ).astype(jnp.float32)


def _hash_bits(value: Array) -> Array:
    bits = _int32_bits(value)
    return bits.reshape(bits.shape[:-2] + (-1,))


def _bounded_nonnegative(value: Array) -> Array:
    numeric = jnp.maximum(jnp.asarray(value, dtype=jnp.float32), 0.0)
    return numeric / (jnp.float32(1.0) + numeric)


def _masked_mean(value: Array, mask: Array) -> Array:
    weight = mask[..., None].astype(jnp.float32)
    count = jnp.maximum(
        jnp.sum(weight, axis=-2),
        jnp.float32(1.0),
    )
    return jnp.sum(value * weight, axis=-2) / count


def _dense(
    key: Array,
    input_size: int,
    output_size: int,
) -> DenseParams:
    limit = math.sqrt(6.0 / (input_size + output_size))
    return DenseParams(
        kernel=jax.random.uniform(
            key,
            (input_size, output_size),
            minval=-limit,
            maxval=limit,
            dtype=jnp.float32,
        ),
        bias=jnp.zeros((output_size,), dtype=jnp.float32),
    )


def _linear(params: DenseParams, value: Array) -> Array:
    return value @ params.kernel + params.bias


def _normalize_config(
    config: RecipeCandidateEncoderConfig | None,
) -> RecipeCandidateEncoderConfig:
    if config is None:
        return RecipeCandidateEncoderConfig()
    if not isinstance(config, RecipeCandidateEncoderConfig):
        raise TypeError("config must be RecipeCandidateEncoderConfig")
    return config


__all__ = [
    "RECIPE_CANDIDATE_ENCODER_STAGING_SCHEMA",
    "RECIPE_CANDIDATE_ENCODER_STAGING_VERSION",
    "RecipeCandidateEncoderConfig",
    "RecipeCandidateEncoderParams",
    "RecipeCandidateEncoding",
    "encode_recipe_candidates",
    "initialize_recipe_candidate_encoder",
    "recipe_candidate_encoder_staging_contract_manifest",
    "recipe_candidate_encoder_staging_contract_sha256",
]
