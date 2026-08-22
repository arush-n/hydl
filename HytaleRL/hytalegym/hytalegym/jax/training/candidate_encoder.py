"""Compatibility facade for the framework-neutral candidate encoder."""

from hytalegym.jax.combat.observation.v3.policy.candidates.encoder import (
    RECIPE_CANDIDATE_ENCODER_STAGING_SCHEMA,
    RECIPE_CANDIDATE_ENCODER_STAGING_VERSION,
    RecipeCandidateEncoderConfig,
    RecipeCandidateEncoderParams,
    RecipeCandidateEncoding,
    encode_recipe_candidates,
    initialize_recipe_candidate_encoder,
    recipe_candidate_encoder_staging_contract_manifest,
    recipe_candidate_encoder_staging_contract_sha256,
)

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

