"""Staged actor-safe encoding of World's block-affordance token rows."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.world import (
    ActorBlockActionCandidates,
    ActorBlockAffordanceTokens,
    actor_block_action_candidate_contract,
    actor_block_action_candidate_contract_sha256,
    actor_block_affordance_token_contract,
    actor_block_affordance_token_contract_sha256,
)


Array = jax.Array

BLOCK_AFFORDANCE_TAG_BIT_COUNT = jnp.dtype(jnp.uint16).itemsize * 8
BLOCK_GATHER_TYPE_BIT_COUNT = jnp.dtype(jnp.uint8).itemsize * 8
BLOCK_AFFORDANCE_POLICY_FEATURE_SIZE = (
    BLOCK_AFFORDANCE_TAG_BIT_COUNT + BLOCK_GATHER_TYPE_BIT_COUNT + 1
)
BLOCK_ACTION_CANDIDATE_POLICY_FEATURE_SIZE = (
    3 + BLOCK_AFFORDANCE_POLICY_FEATURE_SIZE
)
BLOCK_AFFORDANCE_SOURCE_POLICY_FIELDS = (
    "available",
    "token_mask",
    "affordance_tags",
    "gather_type_index",
    "required_tool_quality",
)
BLOCK_ACTION_CANDIDATE_SOURCE_POLICY_FIELDS = (
    "available",
    "candidate_mask",
    "visible_relative_position",
    "affordance_tags",
    "gather_type_index",
    "required_tool_quality",
)

if tuple(actor_block_affordance_token_contract()["policy_fields"]) != (
    BLOCK_AFFORDANCE_SOURCE_POLICY_FIELDS
):
    raise RuntimeError("World block-affordance policy field contract drift")
if tuple(actor_block_action_candidate_contract()["policy_fields"]) != (
    BLOCK_ACTION_CANDIDATE_SOURCE_POLICY_FIELDS
):
    raise RuntimeError("World block-action candidate policy field contract drift")


class BlockAffordancePolicyTokens(NamedTuple):
    """Policy-only block affordances; diagnostics and identity cannot enter."""

    available: Array
    token_f32: Array
    token_mask: Array


class BlockActionCandidatePolicyView(NamedTuple):
    """Policy-only actor-legal block targets on a runtime-static slot axis."""

    available: Array
    candidate_f32: Array
    candidate_mask: Array


class BlockActionCandidateSelection(NamedTuple):
    """Privileged execution target selected through one policy slot."""

    requested: Array
    accepted: Array
    action_legal: Array
    candidate_index: Array
    action_position: Array


def empty_block_affordance_policy_tokens(
    batch: int,
    token_capacity: int,
) -> BlockAffordancePolicyTokens:
    """Return one fixed-shape unavailable actor row."""

    for name, value in (
        ("batch", batch),
        ("token_capacity", token_capacity),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return BlockAffordancePolicyTokens(
        available=jnp.zeros((batch,), dtype=jnp.bool_),
        token_f32=jnp.zeros(
            (
                batch,
                token_capacity,
                BLOCK_AFFORDANCE_POLICY_FEATURE_SIZE,
            ),
            dtype=jnp.float32,
        ),
        token_mask=jnp.zeros((batch, token_capacity), dtype=jnp.bool_),
    )


def empty_block_action_candidate_policy_view(
    batch: int,
    candidate_capacity: int,
) -> BlockActionCandidatePolicyView:
    """Return one fixed-shape unavailable actor target row."""

    for name, value in (
        ("batch", batch),
        ("candidate_capacity", candidate_capacity),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return BlockActionCandidatePolicyView(
        available=jnp.zeros((batch,), dtype=jnp.bool_),
        candidate_f32=jnp.zeros(
            (
                batch,
                candidate_capacity,
                BLOCK_ACTION_CANDIDATE_POLICY_FEATURE_SIZE,
            ),
            dtype=jnp.float32,
        ),
        candidate_mask=jnp.zeros(
            (batch, candidate_capacity),
            dtype=jnp.bool_,
        ),
    )


def encode_block_affordance_policy_tokens(
    source: ActorBlockAffordanceTokens,
    *,
    actor_index: int = 0,
) -> BlockAffordancePolicyTokens:
    """Copy only actor-legal fields and erase every masked source value.

    Stable tag and gather dictionaries are expanded as bits. This avoids
    treating a categorical dictionary index as an ordered scalar and keeps
    the shape valid if the source dictionary grows within its declared
    ``uint16``/``uint8`` domains.
    """

    _validate_source(source, actor_index)
    available = source.available[:, actor_index]
    source_mask = source.token_mask[:, actor_index] & available[:, None]
    tags = source.affordance_tags[:, actor_index]
    gather = source.gather_type_index[:, actor_index]
    quality = source.required_tool_quality[:, actor_index]

    quality_valid = quality >= 0
    row_valid = jnp.all(~source_mask | quality_valid, axis=1)
    available &= row_valid
    token_mask = source_mask & available[:, None]

    features = _affordance_features(tags, gather, quality)
    return BlockAffordancePolicyTokens(
        available=available,
        token_f32=jnp.where(token_mask[..., None], features, jnp.float32(0.0)),
        token_mask=token_mask,
    )


def encode_block_action_candidate_policy_view(
    source: ActorBlockActionCandidates,
    maximum_distance: Array | float,
    *,
    actor_index: int = 0,
) -> BlockActionCandidatePolicyView:
    """Encode actor-legal candidate slots without exposing absolute targets."""

    _validate_candidate_source(source, actor_index)
    batch = source.available.shape[0]
    scale = jnp.asarray(maximum_distance, dtype=jnp.float32)
    if scale.ndim == 0:
        scale = jnp.broadcast_to(scale, (batch,))
    if scale.shape != (batch,):
        raise ValueError("maximum_distance must be scalar or have shape [B]")
    scale_valid = jnp.isfinite(scale) & (scale > 0.0)
    safe_scale = jnp.where(scale_valid, scale, jnp.float32(1.0))

    available = source.available[:, actor_index] & scale_valid
    source_mask = source.candidate_mask[:, actor_index] & available[:, None]
    relative = source.visible_relative_position[:, actor_index]
    tags = source.affordance_tags[:, actor_index]
    gather = source.gather_type_index[:, actor_index]
    quality = source.required_tool_quality[:, actor_index]
    candidate_valid = jnp.all(jnp.isfinite(relative), axis=2) & (quality >= 0)
    row_valid = jnp.all(~source_mask | candidate_valid, axis=1)
    available &= row_valid
    candidate_mask = source_mask & available[:, None]

    normalized_relative = jnp.clip(
        relative / safe_scale[:, None, None],
        -1.0,
        1.0,
    )
    features = jnp.concatenate(
        (
            normalized_relative,
            _affordance_features(tags, gather, quality),
        ),
        axis=2,
    )
    return BlockActionCandidatePolicyView(
        available=available,
        candidate_f32=jnp.where(
            candidate_mask[..., None],
            features,
            jnp.float32(0.0),
        ),
        candidate_mask=candidate_mask,
    )


def resolve_block_action_candidate(
    source: ActorBlockActionCandidates,
    candidate_index: Array,
    *,
    actor_index: int = 0,
) -> BlockActionCandidateSelection:
    """Resolve ``-1`` or one actor-legal slot to a privileged block position."""

    _validate_candidate_source(source, actor_index)
    batch = source.available.shape[0]
    index = jnp.asarray(candidate_index, dtype=jnp.int32)
    if index.shape != (batch,):
        raise ValueError("candidate_index must have shape [B]")
    capacity = source.candidate_mask.shape[2]
    none_requested = index == -1
    requested = index >= 0
    in_range = requested & (index < capacity)
    safe_index = jnp.clip(index, 0, capacity - 1)
    row_available = (
        source.available[:, actor_index]
        & ~source.capacity_exceeded[:, actor_index]
    )
    selected_mask = jnp.take_along_axis(
        source.candidate_mask[:, actor_index],
        safe_index[:, None],
        axis=1,
    )[:, 0]
    selected_position = jnp.take_along_axis(
        source.action_position[:, actor_index],
        safe_index[:, None, None],
        axis=1,
    )[:, 0]
    rounded_position = jnp.rint(selected_position)
    position_valid = (
        jnp.all(jnp.isfinite(selected_position), axis=1)
        & jnp.all(selected_position == rounded_position, axis=1)
        & jnp.all(
            (rounded_position >= jnp.iinfo(jnp.int32).min)
            & (rounded_position <= jnp.iinfo(jnp.int32).max),
            axis=1,
        )
    )
    accepted = (
        in_range
        & row_available
        & selected_mask
        & position_valid
    )
    action_legal = none_requested | accepted
    return BlockActionCandidateSelection(
        requested=requested,
        accepted=accepted,
        action_legal=action_legal,
        candidate_index=jnp.where(accepted, index, jnp.int32(-1)),
        action_position=jnp.where(
            accepted[:, None],
            rounded_position.astype(jnp.int32),
            jnp.int32(0),
        ),
    )


def mask_block_affordance_policy_tokens(
    tokens: BlockAffordancePolicyTokens,
    valid: Array,
) -> BlockAffordancePolicyTokens:
    """Apply the outer learner-observation validity gate."""

    if not isinstance(tokens, BlockAffordancePolicyTokens):
        raise TypeError("tokens must be BlockAffordancePolicyTokens")
    valid_mask = jnp.asarray(valid, dtype=jnp.bool_)
    if valid_mask.shape != tokens.available.shape:
        raise ValueError("valid must match tokens.available")
    available = tokens.available & valid_mask
    token_mask = tokens.token_mask & available[:, None]
    return BlockAffordancePolicyTokens(
        available=available,
        token_f32=jnp.where(
            token_mask[..., None],
            tokens.token_f32,
            jnp.float32(0.0),
        ),
        token_mask=token_mask,
    )


def mask_block_action_candidate_policy_view(
    view: BlockActionCandidatePolicyView,
    valid: Array,
) -> BlockActionCandidatePolicyView:
    """Apply the outer learner-observation validity gate."""

    if not isinstance(view, BlockActionCandidatePolicyView):
        raise TypeError("view must be BlockActionCandidatePolicyView")
    valid_mask = jnp.asarray(valid, dtype=jnp.bool_)
    if valid_mask.shape != view.available.shape:
        raise ValueError("valid must match view.available")
    available = view.available & valid_mask
    candidate_mask = view.candidate_mask & available[:, None]
    return BlockActionCandidatePolicyView(
        available=available,
        candidate_f32=jnp.where(
            candidate_mask[..., None],
            view.candidate_f32,
            jnp.float32(0.0),
        ),
        candidate_mask=candidate_mask,
    )


def block_affordance_policy_staging_contract_manifest() -> dict[str, object]:
    """Describe the unpublished Combat consumer without moving policy ABI."""

    from hytalegym.jax.combat.observation.v3.policy import (
        ARSENAL_POLICY_ACTION_HEAD_SIZES,
        ARSENAL_POLICY_COMBINATION_COUNT,
        ARSENAL_POLICY_TARGET_RADIX_BUDGET,
    )

    source = actor_block_affordance_token_contract()
    return {
        "schema": "hytalerl_combat_block_affordance_policy_staging_v2",
        "version": 2,
        "source_contract_sha256": (
            actor_block_affordance_token_contract_sha256()
        ),
        "action_candidate_contract_sha256": (
            actor_block_action_candidate_contract_sha256()
        ),
        "affordance_dictionary_sha256": (
            source["affordance_dictionary_sha256"]
        ),
        "source_policy_fields": list(BLOCK_AFFORDANCE_SOURCE_POLICY_FIELDS),
        "action_candidate_policy_fields": list(
            BLOCK_ACTION_CANDIDATE_SOURCE_POLICY_FIELDS
        ),
        "excluded_source_fields": [
            "diagnostics",
            "provenance",
        ],
        "excluded_action_candidate_fields": [
            "capacity_exceeded",
            "diagnostics",
            "visible_position",
            "action_position",
            "provenance",
        ],
        "feature_size": BLOCK_AFFORDANCE_POLICY_FEATURE_SIZE,
        "action_candidate_feature_size": (
            BLOCK_ACTION_CANDIDATE_POLICY_FEATURE_SIZE
        ),
        "encoding": {
            "affordance_tags": "uint16_lsb_first_bit_expansion",
            "gather_type_index": "uint8_lsb_first_bit_expansion",
            "required_tool_quality": "nonnegative_float32",
            "masked_values": "bit_exact_zero",
        },
        "target_selection": (
            "minus_one_or_actor_legal_candidate_slot_never_global_block_id"
        ),
        "target_dictionary_sha256": (
            actor_block_action_candidate_contract()[
                "affordance_dictionary_sha256"
            ]
        ),
        "rollout_transport": {
            "dtype": "int32",
            "current_head_sizes": list(ARSENAL_POLICY_ACTION_HEAD_SIZES),
            "published_combination_count": ARSENAL_POLICY_COMBINATION_COUNT,
            "combined_target_radix_budget": (
                ARSENAL_POLICY_TARGET_RADIX_BUDGET
            ),
            "transport": "explicit_per_head_int32",
            "head_addition_order_affects_acceptance": False,
        },
        "publication": "published_in_policy_v6_actor_candidate_append",
    }


def block_affordance_policy_staging_contract_sha256() -> str:
    """Return the canonical identity of the staged consumer."""

    payload = json.dumps(
        block_affordance_policy_staging_contract_manifest(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _validate_source(
    source: ActorBlockAffordanceTokens,
    actor_index: int,
) -> None:
    if not isinstance(source, ActorBlockAffordanceTokens):
        raise TypeError("source must be ActorBlockAffordanceTokens")
    if isinstance(actor_index, bool) or not isinstance(actor_index, int):
        raise TypeError("actor_index must be an integer")
    if source.available.ndim != 2:
        raise ValueError("source.available must have shape [B, A]")
    batch, actors = source.available.shape
    if not 0 <= actor_index < actors:
        raise ValueError("actor_index is outside the source actor axis")
    if source.token_mask.ndim != 3:
        raise ValueError("source.token_mask must have shape [B, A, T]")
    token_shape = source.token_mask.shape
    if token_shape[:2] != (batch, actors):
        raise ValueError("source.token_mask must have shape [B, A, T]")
    for name in (
        "affordance_tags",
        "gather_type_index",
        "required_tool_quality",
    ):
        if getattr(source, name).shape != token_shape:
            raise ValueError(f"source.{name} must match source.token_mask")
    for name in ("diagnostics", "provenance"):
        if getattr(source, name).shape != (batch, actors):
            raise ValueError(f"source.{name} must match source.available")
    expected_dtypes = (
        ("available", source.available, jnp.bool_),
        ("token_mask", source.token_mask, jnp.bool_),
        ("affordance_tags", source.affordance_tags, jnp.uint16),
        ("gather_type_index", source.gather_type_index, jnp.uint8),
        ("required_tool_quality", source.required_tool_quality, jnp.int16),
    )
    for name, value, dtype in expected_dtypes:
        if jnp.asarray(value).dtype != jnp.dtype(dtype):
            raise TypeError(f"source.{name} must have dtype {jnp.dtype(dtype)}")


def _validate_candidate_source(
    source: ActorBlockActionCandidates,
    actor_index: int,
) -> None:
    if not isinstance(source, ActorBlockActionCandidates):
        raise TypeError("source must be ActorBlockActionCandidates")
    if isinstance(actor_index, bool) or not isinstance(actor_index, int):
        raise TypeError("actor_index must be an integer")
    if source.available.ndim != 2:
        raise ValueError("source.available must have shape [B, A]")
    batch, actors = source.available.shape
    if not 0 <= actor_index < actors:
        raise ValueError("actor_index is outside the source actor axis")
    if source.candidate_mask.ndim != 3:
        raise ValueError("source.candidate_mask must have shape [B, A, C]")
    candidate_shape = source.candidate_mask.shape
    if candidate_shape[:2] != (batch, actors) or candidate_shape[2] < 1:
        raise ValueError("source.candidate_mask must have shape [B, A, C]")
    for name in (
        "affordance_tags",
        "gather_type_index",
        "required_tool_quality",
    ):
        if getattr(source, name).shape != candidate_shape:
            raise ValueError(f"source.{name} must match source.candidate_mask")
    vector_shape = candidate_shape + (3,)
    for name in (
        "visible_position",
        "action_position",
        "visible_relative_position",
    ):
        if getattr(source, name).shape != vector_shape:
            raise ValueError(f"source.{name} must have shape {vector_shape}")
    for name in (
        "capacity_exceeded",
        "diagnostics",
    ):
        if getattr(source, name).shape != (batch, actors):
            raise ValueError(f"source.{name} must match source.available")
    if source.provenance.shape != candidate_shape:
        raise ValueError(
            "source.provenance must match source.candidate_mask"
        )
    expected_dtypes = (
        ("available", source.available, jnp.bool_),
        ("capacity_exceeded", source.capacity_exceeded, jnp.bool_),
        ("candidate_mask", source.candidate_mask, jnp.bool_),
        ("affordance_tags", source.affordance_tags, jnp.uint16),
        ("gather_type_index", source.gather_type_index, jnp.uint8),
        ("required_tool_quality", source.required_tool_quality, jnp.int16),
    )
    for name, value, dtype in expected_dtypes:
        if jnp.asarray(value).dtype != jnp.dtype(dtype):
            raise TypeError(f"source.{name} must have dtype {jnp.dtype(dtype)}")


def _affordance_features(
    tags: Array,
    gather: Array,
    quality: Array,
) -> Array:
    tag_shifts = jnp.arange(
        BLOCK_AFFORDANCE_TAG_BIT_COUNT,
        dtype=jnp.uint16,
    )
    gather_shifts = jnp.arange(
        BLOCK_GATHER_TYPE_BIT_COUNT,
        dtype=jnp.uint8,
    )
    tag_bits = (
        (tags[..., None] >> tag_shifts) & jnp.uint16(1)
    ).astype(jnp.float32)
    gather_bits = (
        (gather[..., None] >> gather_shifts) & jnp.uint8(1)
    ).astype(jnp.float32)
    return jnp.concatenate(
        (
            tag_bits,
            gather_bits,
            quality[..., None].astype(jnp.float32),
        ),
        axis=2,
    )


__all__ = [
    "BLOCK_ACTION_CANDIDATE_POLICY_FEATURE_SIZE",
    "BLOCK_ACTION_CANDIDATE_SOURCE_POLICY_FIELDS",
    "BLOCK_AFFORDANCE_POLICY_FEATURE_SIZE",
    "BLOCK_AFFORDANCE_SOURCE_POLICY_FIELDS",
    "BLOCK_AFFORDANCE_TAG_BIT_COUNT",
    "BLOCK_GATHER_TYPE_BIT_COUNT",
    "BlockActionCandidatePolicyView",
    "BlockActionCandidateSelection",
    "BlockAffordancePolicyTokens",
    "block_affordance_policy_staging_contract_manifest",
    "block_affordance_policy_staging_contract_sha256",
    "empty_block_action_candidate_policy_view",
    "empty_block_affordance_policy_tokens",
    "encode_block_action_candidate_policy_view",
    "encode_block_affordance_policy_tokens",
    "mask_block_action_candidate_policy_view",
    "mask_block_affordance_policy_tokens",
    "resolve_block_action_candidate",
]
