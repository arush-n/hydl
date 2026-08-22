"""Stable actor-side recipe-candidate encoding contract."""

from __future__ import annotations

import hashlib
import json

from hytalegym.jax.crafting.contract import (
    IDENTITY_HASH_WORDS,
    MAX_BENCH_REQUIREMENTS,
    MAX_INGREDIENTS,
    MAX_OUTPUTS,
    METADATA_HASH_WORDS,
)
from hytalegym.jax.world import (
    ACTOR_RECIPE_CANDIDATE_CAPACITY,
    actor_recipe_candidate_contract_sha256,
)


RECIPE_CANDIDATE_ENCODING_SCHEMA = "hytalerl_recipe_candidate_encoding_v1"
RECIPE_CANDIDATE_ENCODING_VERSION = 1
RECIPE_CANDIDATE_EMBEDDING_SIZE = 32
RECIPE_CANDIDATE_INT32_BIT_COUNT = 32
RECIPE_CANDIDATE_INPUT_FEATURE_SIZE = (
    RECIPE_CANDIDATE_INT32_BIT_COUNT
    + RECIPE_CANDIDATE_INT32_BIT_COUNT
    + 1
    + 1
    + METADATA_HASH_WORDS * RECIPE_CANDIDATE_INT32_BIT_COUNT
)
RECIPE_CANDIDATE_OUTPUT_FEATURE_SIZE = (
    RECIPE_CANDIDATE_INT32_BIT_COUNT
    + 1
    + METADATA_HASH_WORDS * RECIPE_CANDIDATE_INT32_BIT_COUNT
)
RECIPE_CANDIDATE_REQUIREMENT_FEATURE_SIZE = (
    RECIPE_CANDIDATE_INT32_BIT_COUNT
    + IDENTITY_HASH_WORDS * RECIPE_CANDIDATE_INT32_BIT_COUNT
    + 1
)
RECIPE_CANDIDATE_SCALAR_SIZE = 6


def recipe_candidate_encoding_contract_manifest() -> dict[str, object]:
    """Describe every value that can change the deterministic embedding."""

    return {
        "schema": RECIPE_CANDIDATE_ENCODING_SCHEMA,
        "version": RECIPE_CANDIDATE_ENCODING_VERSION,
        "source_contract_sha256": actor_recipe_candidate_contract_sha256(),
        "candidate_capacity": ACTOR_RECIPE_CANDIDATE_CAPACITY,
        "nested_capacities": {
            "ingredients": MAX_INGREDIENTS,
            "outputs": MAX_OUTPUTS,
            "bench_requirements": MAX_BENCH_REQUIREMENTS,
        },
        "feature_sizes": {
            "input": RECIPE_CANDIDATE_INPUT_FEATURE_SIZE,
            "output": RECIPE_CANDIDATE_OUTPUT_FEATURE_SIZE,
            "requirement": RECIPE_CANDIDATE_REQUIREMENT_FEATURE_SIZE,
            "candidate_scalars": RECIPE_CANDIDATE_SCALAR_SIZE,
            "embedding": RECIPE_CANDIDATE_EMBEDDING_SIZE,
        },
        "categorical_encoding": (
            "little_endian_32_bit_expansion_never_ordered_float_ids"
        ),
        "numeric_encoding": "x_over_one_plus_x_nonnegative",
        "nested_reduction": "shared_tanh_projection_masked_mean",
        "candidate_projection": "shared_tanh_projection_permutation_equivariant",
        "masked_values": "bit_exact_zero",
        "parameter_initialization": (
            "jax_random_key_from_first_32_bits_of_this_contract_sha256"
        ),
    }


def recipe_candidate_encoding_contract_sha256() -> str:
    payload = json.dumps(
        recipe_candidate_encoding_contract_manifest(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def recipe_candidate_encoding_parameter_seed() -> int:
    """Return the deterministic initializer seed bound by the policy contract."""

    return int(recipe_candidate_encoding_contract_sha256()[:8], 16)


__all__ = [
    "RECIPE_CANDIDATE_EMBEDDING_SIZE",
    "RECIPE_CANDIDATE_ENCODING_SCHEMA",
    "RECIPE_CANDIDATE_ENCODING_VERSION",
    "RECIPE_CANDIDATE_INPUT_FEATURE_SIZE",
    "RECIPE_CANDIDATE_INT32_BIT_COUNT",
    "RECIPE_CANDIDATE_OUTPUT_FEATURE_SIZE",
    "RECIPE_CANDIDATE_REQUIREMENT_FEATURE_SIZE",
    "RECIPE_CANDIDATE_SCALAR_SIZE",
    "recipe_candidate_encoding_contract_manifest",
    "recipe_candidate_encoding_contract_sha256",
    "recipe_candidate_encoding_parameter_seed",
]
