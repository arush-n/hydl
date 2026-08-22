"""Staged actor-safe consumption of World's recipe-candidate rows."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.crafting.contract import (
    BENCH_TYPE_COUNT,
    IDENTITY_HASH_WORDS,
    MAX_BENCH_REQUIREMENTS,
    MAX_INGREDIENTS,
    MAX_OUTPUTS,
    METADATA_HASH_WORDS,
)
from hytalegym.jax.world import (
    ACTOR_RECIPE_CANDIDATE_CAPACITY,
    ACTOR_RECIPE_TARGET_RADIX,
    ActorRecipeCandidatePolicy,
    ActorRecipeCandidates,
    actor_recipe_candidate_contract,
    actor_recipe_candidate_contract_sha256,
)


Array = jax.Array

RECIPE_CANDIDATE_SOURCE_POLICY_FIELDS = (
    "input_mask",
    "input_item_id",
    "input_resource_type_id",
    "input_quantity",
    "input_metadata_required",
    "input_metadata_hash",
    "output_mask",
    "output_item_id",
    "output_quantity",
    "output_metadata_hash",
    "requirement_mask",
    "requirement_bench_type",
    "requirement_bench_id_hash",
    "requirement_tier_level",
    "knowledge_required",
    "required_memories_level",
    "time_seconds",
)

if tuple(actor_recipe_candidate_contract()["policy_fields"]) != (
    RECIPE_CANDIDATE_SOURCE_POLICY_FIELDS
):
    raise RuntimeError("World recipe-candidate policy field contract drift")


class RecipeCandidatePolicyView(NamedTuple):
    """Policy-only legal recipes; native identity and diagnostics are absent."""

    available: Array
    candidate_mask: Array
    policy: ActorRecipeCandidatePolicy


class RecipeCandidateSelection(NamedTuple):
    """Privileged recipe identity revealed after one legal slot is selected."""

    requested: Array
    accepted: Array
    action_legal: Array
    candidate_index: Array
    recipe_index: Array
    recipe_id_hash: Array


def empty_recipe_candidate_policy_view(
    batch: int,
) -> RecipeCandidatePolicyView:
    """Return a fixed-capacity unavailable actor row."""

    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("batch must be a positive integer")
    candidate_shape = (batch, ACTOR_RECIPE_CANDIDATE_CAPACITY)
    return RecipeCandidatePolicyView(
        available=jnp.zeros((batch,), dtype=jnp.bool_),
        candidate_mask=jnp.zeros(candidate_shape, dtype=jnp.bool_),
        policy=_empty_policy(candidate_shape),
    )


def encode_recipe_candidate_policy_view(
    source: ActorRecipeCandidates,
    *,
    actor_index: int = 0,
) -> RecipeCandidatePolicyView:
    """Copy exactly the selected actor's legal policy fields.

    Global recipe indexes, native identity hashes, source counts, diagnostics,
    overflow state, other actors, and every masked candidate are structurally
    excluded. Corrupt policy rows fail closed rather than being partly
    consumed.
    """

    _validate_source(source, actor_index)
    source_mask = source.candidate_mask[:, actor_index]
    source_policy = jax.tree.map(
        lambda value: value[:, actor_index],
        source.policy,
    )
    candidate_valid = _candidate_policy_valid(source_policy, source_mask)
    available = (
        source.available[:, actor_index]
        & ~source.capacity_exceeded[:, actor_index]
        & jnp.all(~source_mask | candidate_valid, axis=1)
    )
    candidate_mask = source_mask & available[:, None]
    return RecipeCandidatePolicyView(
        available=available,
        candidate_mask=candidate_mask,
        policy=_mask_policy(source_policy, candidate_mask),
    )


def mask_recipe_candidate_policy_view(
    view: RecipeCandidatePolicyView,
    valid: Array,
) -> RecipeCandidatePolicyView:
    """Apply the outer learner-observation validity gate."""

    if not isinstance(view, RecipeCandidatePolicyView):
        raise TypeError("view must be RecipeCandidatePolicyView")
    valid_mask = jnp.asarray(valid, dtype=jnp.bool_)
    if valid_mask.shape != view.available.shape:
        raise ValueError("valid must match view.available")
    available = view.available & valid_mask
    candidate_mask = view.candidate_mask & available[:, None]
    return RecipeCandidatePolicyView(
        available=available,
        candidate_mask=candidate_mask,
        policy=_mask_policy(view.policy, candidate_mask),
    )


def resolve_recipe_candidate(
    source: ActorRecipeCandidates,
    candidate_index: Array,
    *,
    actor_index: int = 0,
) -> RecipeCandidateSelection:
    """Resolve ``-1`` or one legal slot to a privileged native recipe key."""

    _validate_source(source, actor_index)
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
    selected_recipe_index = jnp.take_along_axis(
        source.execution.recipe_index[:, actor_index],
        safe_index[:, None],
        axis=1,
    )[:, 0]
    selected_recipe_hash = jnp.take_along_axis(
        source.execution.recipe_id_hash[:, actor_index],
        safe_index[:, None, None],
        axis=1,
    )[:, 0]
    identity_valid = (
        (selected_recipe_index >= 0)
        & jnp.any(selected_recipe_hash != jnp.uint32(0), axis=1)
    )
    accepted = (
        in_range
        & row_available
        & selected_mask
        & identity_valid
    )
    action_legal = none_requested | accepted
    return RecipeCandidateSelection(
        requested=requested,
        accepted=accepted,
        action_legal=action_legal,
        candidate_index=jnp.where(accepted, index, jnp.int32(-1)),
        recipe_index=jnp.where(
            accepted,
            selected_recipe_index,
            jnp.int32(-1),
        ),
        recipe_id_hash=jnp.where(
            accepted[:, None],
            selected_recipe_hash,
            jnp.uint32(0),
        ),
    )


def recipe_candidate_policy_staging_contract_manifest() -> dict[str, object]:
    """Describe the unpublished Combat consumer without moving policy ABI."""

    source = actor_recipe_candidate_contract()
    return {
        "schema": "hytalerl_combat_recipe_candidate_policy_staging_v1",
        "version": 1,
        "source_contract_sha256": (
            actor_recipe_candidate_contract_sha256()
        ),
        "source_policy_fields": list(RECIPE_CANDIDATE_SOURCE_POLICY_FIELDS),
        "excluded_source_fields": list(source["excluded_from_policy"]),
        "target_selection": (
            "minus_one_or_actor_legal_candidate_slot_never_global_recipe_id"
        ),
        "target_radix": ACTOR_RECIPE_TARGET_RADIX,
        "representation": (
            "structured_typed_policy_fields_encoder_selected_at_publication"
        ),
        "rollout_transport": "explicit_per_head_int32_selected_by_PACK_01",
        "masked_values": "bit_exact_zero",
        "candidate_overflow": "fail_entire_actor_row_closed",
        "publication": "published_in_policy_v6_actor_candidate_append",
    }


def recipe_candidate_policy_staging_contract_sha256() -> str:
    """Return the canonical identity of the staged consumer."""

    payload = json.dumps(
        recipe_candidate_policy_staging_contract_manifest(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _candidate_policy_valid(
    policy: ActorRecipeCandidatePolicy,
    candidate_mask: Array,
) -> Array:
    input_valid = (
        policy.input_mask
        & (policy.input_quantity > 0)
        & (
            (policy.input_item_id >= 0)
            | (policy.input_resource_type_id >= 0)
        )
    )
    output_valid = (
        policy.output_mask
        & (policy.output_item_id >= 0)
        & (policy.output_quantity > 0)
    )
    requirement_valid = (
        policy.requirement_mask
        & (policy.requirement_bench_type >= 0)
        & (policy.requirement_bench_type < BENCH_TYPE_COUNT)
        & (policy.requirement_tier_level >= 0)
    )
    nested_masks_valid = (
        jnp.all(~policy.input_mask | candidate_mask[..., None], axis=2)
        & jnp.all(~policy.output_mask | candidate_mask[..., None], axis=2)
        & jnp.all(
            ~policy.requirement_mask | candidate_mask[..., None],
            axis=2,
        )
    )
    return (
        nested_masks_valid
        & jnp.all(~policy.input_mask | input_valid, axis=2)
        & jnp.all(~policy.output_mask | output_valid, axis=2)
        & jnp.all(
            ~policy.requirement_mask | requirement_valid,
            axis=2,
        )
        & (policy.required_memories_level >= 0)
        & jnp.isfinite(policy.time_seconds)
        & (policy.time_seconds >= 0.0)
    )


def _mask_policy(
    policy: ActorRecipeCandidatePolicy,
    candidate_mask: Array,
) -> ActorRecipeCandidatePolicy:
    input_mask = policy.input_mask & candidate_mask[..., None]
    output_mask = policy.output_mask & candidate_mask[..., None]
    requirement_mask = (
        policy.requirement_mask & candidate_mask[..., None]
    )
    return ActorRecipeCandidatePolicy(
        input_mask=input_mask,
        input_item_id=_where(input_mask, policy.input_item_id),
        input_resource_type_id=_where(
            input_mask,
            policy.input_resource_type_id,
        ),
        input_quantity=_where(input_mask, policy.input_quantity),
        input_metadata_required=_where(
            input_mask,
            policy.input_metadata_required,
        ),
        input_metadata_hash=_where(
            input_mask,
            policy.input_metadata_hash,
        ),
        output_mask=output_mask,
        output_item_id=_where(output_mask, policy.output_item_id),
        output_quantity=_where(output_mask, policy.output_quantity),
        output_metadata_hash=_where(
            output_mask,
            policy.output_metadata_hash,
        ),
        requirement_mask=requirement_mask,
        requirement_bench_type=_where(
            requirement_mask,
            policy.requirement_bench_type,
        ),
        requirement_bench_id_hash=_where(
            requirement_mask,
            policy.requirement_bench_id_hash,
        ),
        requirement_tier_level=_where(
            requirement_mask,
            policy.requirement_tier_level,
        ),
        knowledge_required=_where(
            candidate_mask,
            policy.knowledge_required,
        ),
        required_memories_level=_where(
            candidate_mask,
            policy.required_memories_level,
        ),
        time_seconds=_where(candidate_mask, policy.time_seconds),
    )


def _empty_policy(
    candidate_shape: tuple[int, int],
) -> ActorRecipeCandidatePolicy:
    inputs = candidate_shape + (MAX_INGREDIENTS,)
    outputs = candidate_shape + (MAX_OUTPUTS,)
    requirements = candidate_shape + (MAX_BENCH_REQUIREMENTS,)
    return ActorRecipeCandidatePolicy(
        input_mask=jnp.zeros(inputs, dtype=jnp.bool_),
        input_item_id=jnp.zeros(inputs, dtype=jnp.int32),
        input_resource_type_id=jnp.zeros(inputs, dtype=jnp.int32),
        input_quantity=jnp.zeros(inputs, dtype=jnp.int32),
        input_metadata_required=jnp.zeros(inputs, dtype=jnp.bool_),
        input_metadata_hash=jnp.zeros(
            inputs + (METADATA_HASH_WORDS,),
            dtype=jnp.uint32,
        ),
        output_mask=jnp.zeros(outputs, dtype=jnp.bool_),
        output_item_id=jnp.zeros(outputs, dtype=jnp.int32),
        output_quantity=jnp.zeros(outputs, dtype=jnp.int32),
        output_metadata_hash=jnp.zeros(
            outputs + (METADATA_HASH_WORDS,),
            dtype=jnp.uint32,
        ),
        requirement_mask=jnp.zeros(requirements, dtype=jnp.bool_),
        requirement_bench_type=jnp.zeros(requirements, dtype=jnp.int32),
        requirement_bench_id_hash=jnp.zeros(
            requirements + (IDENTITY_HASH_WORDS,),
            dtype=jnp.uint32,
        ),
        requirement_tier_level=jnp.zeros(
            requirements,
            dtype=jnp.int32,
        ),
        knowledge_required=jnp.zeros(candidate_shape, dtype=jnp.bool_),
        required_memories_level=jnp.zeros(
            candidate_shape,
            dtype=jnp.int32,
        ),
        time_seconds=jnp.zeros(candidate_shape, dtype=jnp.float32),
    )


def _where(mask: Array, value: Array) -> Array:
    expanded = mask
    while expanded.ndim < value.ndim:
        expanded = expanded[..., None]
    return jnp.where(expanded, value, jnp.zeros((), dtype=value.dtype))


def _validate_source(
    source: ActorRecipeCandidates,
    actor_index: int,
) -> None:
    if not isinstance(source, ActorRecipeCandidates):
        raise TypeError("source must be ActorRecipeCandidates")
    if isinstance(actor_index, bool) or not isinstance(actor_index, int):
        raise TypeError("actor_index must be an integer")
    if source.available.ndim != 2:
        raise ValueError("source.available must have shape [B, A]")
    batch, actors = source.available.shape
    if not 0 <= actor_index < actors:
        raise ValueError("actor_index is outside the source actor axis")
    candidate_shape = (
        batch,
        actors,
        ACTOR_RECIPE_CANDIDATE_CAPACITY,
    )
    if source.candidate_mask.shape != candidate_shape:
        raise ValueError(
            "source.candidate_mask must match the public candidate capacity"
        )
    prefix_fields = (
        "capacity_exceeded",
        "diagnostics",
        "emitted_count",
        "source_count",
    )
    for name in prefix_fields:
        if getattr(source, name).shape != (batch, actors):
            raise ValueError(f"source.{name} must match source.available")
    policy_shapes = {
        "input_mask": candidate_shape + (MAX_INGREDIENTS,),
        "input_item_id": candidate_shape + (MAX_INGREDIENTS,),
        "input_resource_type_id": candidate_shape + (MAX_INGREDIENTS,),
        "input_quantity": candidate_shape + (MAX_INGREDIENTS,),
        "input_metadata_required": candidate_shape + (MAX_INGREDIENTS,),
        "input_metadata_hash": (
            candidate_shape + (MAX_INGREDIENTS, METADATA_HASH_WORDS)
        ),
        "output_mask": candidate_shape + (MAX_OUTPUTS,),
        "output_item_id": candidate_shape + (MAX_OUTPUTS,),
        "output_quantity": candidate_shape + (MAX_OUTPUTS,),
        "output_metadata_hash": (
            candidate_shape + (MAX_OUTPUTS, METADATA_HASH_WORDS)
        ),
        "requirement_mask": (
            candidate_shape + (MAX_BENCH_REQUIREMENTS,)
        ),
        "requirement_bench_type": (
            candidate_shape + (MAX_BENCH_REQUIREMENTS,)
        ),
        "requirement_bench_id_hash": (
            candidate_shape
            + (MAX_BENCH_REQUIREMENTS, IDENTITY_HASH_WORDS)
        ),
        "requirement_tier_level": (
            candidate_shape + (MAX_BENCH_REQUIREMENTS,)
        ),
        "knowledge_required": candidate_shape,
        "required_memories_level": candidate_shape,
        "time_seconds": candidate_shape,
    }
    for name, shape in policy_shapes.items():
        if getattr(source.policy, name).shape != shape:
            raise ValueError(f"source.policy.{name} has the wrong shape")
    if source.execution.recipe_index.shape != candidate_shape:
        raise ValueError("source.execution.recipe_index has the wrong shape")
    if source.execution.recipe_id_hash.shape != (
        candidate_shape + (IDENTITY_HASH_WORDS,)
    ):
        raise ValueError(
            "source.execution.recipe_id_hash has the wrong shape"
        )
    expected_dtypes = {
        "available": (source.available, jnp.bool_),
        "candidate_mask": (source.candidate_mask, jnp.bool_),
        "capacity_exceeded": (source.capacity_exceeded, jnp.bool_),
        "diagnostics": (source.diagnostics, jnp.uint32),
        "source_count": (source.source_count, jnp.int32),
        "emitted_count": (source.emitted_count, jnp.int32),
        "input_mask": (source.policy.input_mask, jnp.bool_),
        "input_item_id": (source.policy.input_item_id, jnp.int32),
        "input_resource_type_id": (
            source.policy.input_resource_type_id,
            jnp.int32,
        ),
        "input_quantity": (source.policy.input_quantity, jnp.int32),
        "input_metadata_required": (
            source.policy.input_metadata_required,
            jnp.bool_,
        ),
        "input_metadata_hash": (
            source.policy.input_metadata_hash,
            jnp.uint32,
        ),
        "output_mask": (source.policy.output_mask, jnp.bool_),
        "output_item_id": (source.policy.output_item_id, jnp.int32),
        "output_quantity": (source.policy.output_quantity, jnp.int32),
        "output_metadata_hash": (
            source.policy.output_metadata_hash,
            jnp.uint32,
        ),
        "requirement_mask": (source.policy.requirement_mask, jnp.bool_),
        "requirement_bench_type": (
            source.policy.requirement_bench_type,
            jnp.int32,
        ),
        "requirement_bench_id_hash": (
            source.policy.requirement_bench_id_hash,
            jnp.uint32,
        ),
        "requirement_tier_level": (
            source.policy.requirement_tier_level,
            jnp.int32,
        ),
        "knowledge_required": (
            source.policy.knowledge_required,
            jnp.bool_,
        ),
        "required_memories_level": (
            source.policy.required_memories_level,
            jnp.int32,
        ),
        "time_seconds": (source.policy.time_seconds, jnp.float32),
        "recipe_index": (source.execution.recipe_index, jnp.int32),
        "recipe_id_hash": (
            source.execution.recipe_id_hash,
            jnp.uint32,
        ),
    }
    for name, (value, dtype) in expected_dtypes.items():
        if jnp.asarray(value).dtype != jnp.dtype(dtype):
            raise TypeError(f"source.{name} must have dtype {jnp.dtype(dtype)}")


__all__ = [
    "RECIPE_CANDIDATE_SOURCE_POLICY_FIELDS",
    "RecipeCandidatePolicyView",
    "RecipeCandidateSelection",
    "empty_recipe_candidate_policy_view",
    "encode_recipe_candidate_policy_view",
    "mask_recipe_candidate_policy_view",
    "recipe_candidate_policy_staging_contract_manifest",
    "recipe_candidate_policy_staging_contract_sha256",
    "resolve_recipe_candidate",
]
