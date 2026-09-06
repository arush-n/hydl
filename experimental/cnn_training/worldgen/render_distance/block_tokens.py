"""Generic numeric token derivation compatibility path."""

from ..viewpoints.tokens import *

RenderedBlockTokenDetails = BlockTokenBatch
build_rendered_block_token_details = build_block_tokens
build_dense_block_token_features_jax = build_dense_features_jax

__all__ = ["BlockTokenBatch", "RenderedBlockTokenDetails", "build_block_tokens", "build_rendered_block_token_details", "build_dense_features_jax", "build_dense_block_token_features_jax"]
