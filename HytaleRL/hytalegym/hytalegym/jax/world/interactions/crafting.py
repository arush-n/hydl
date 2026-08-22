"""Fixed-shape actor-legal recipe candidates for policy consumers."""

from __future__ import annotations

import hashlib
import json
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.crafting.contract import (
    MAX_BENCH_REQUIREMENTS,
    MAX_INGREDIENTS,
    MAX_OUTPUTS,
    RECIPE_TABLE_CAPACITY,
    crafting_contract_sha256,
)
from hytalegym.jax.crafting.runtime import satisfiable_recipes
from hytalegym.jax.crafting.types import (
    CraftingContext,
    CraftingInventory,
    RecipeTable,
)


Array = jax.Array

ACTOR_RECIPE_CANDIDATE_SCHEMA = "hytalerl_actor_recipe_candidates_v1"
ACTOR_RECIPE_CANDIDATE_VERSION = 1
ACTOR_RECIPE_CANDIDATE_CAPACITY = 16
ACTOR_RECIPE_TARGET_RADIX = ACTOR_RECIPE_CANDIDATE_CAPACITY + 1
ACTOR_RECIPE_COMBINED_TARGET_RADIX_BUDGET = 324

ACTOR_RECIPE_DIAGNOSTIC_CATALOG_UNAVAILABLE = jnp.uint32(1 << 0)
ACTOR_RECIPE_DIAGNOSTIC_EVIDENCE_UNAVAILABLE = jnp.uint32(1 << 1)
ACTOR_RECIPE_DIAGNOSTIC_INVALID_REFERENCE = jnp.uint32(1 << 2)
ACTOR_RECIPE_DIAGNOSTIC_CAPACITY = jnp.uint32(1 << 3)


class ActorRecipeCandidatePolicy(NamedTuple):
    """Actor-visible recipe meaning; native recipe identity is excluded."""

    input_mask: Array
    input_item_id: Array
    input_resource_type_id: Array
    input_quantity: Array
    input_metadata_required: Array
    input_metadata_hash: Array
    output_mask: Array
    output_item_id: Array
    output_quantity: Array
    output_metadata_hash: Array
    requirement_mask: Array
    requirement_bench_type: Array
    requirement_bench_id_hash: Array
    requirement_tier_level: Array
    knowledge_required: Array
    required_memories_level: Array
    time_seconds: Array


class ActorRecipeCandidateExecution(NamedTuple):
    """Privileged lookup retained until one legal slot is selected."""

    recipe_index: Array
    recipe_id_hash: Array


class ActorRecipeCandidates(NamedTuple):
    """One complete, bounded recipe row per batch actor."""

    available: Array
    candidate_mask: Array
    policy: ActorRecipeCandidatePolicy
    execution: ActorRecipeCandidateExecution
    source_count: Array
    emitted_count: Array
    capacity_exceeded: Array
    diagnostics: Array


def actor_recipe_candidate_contract() -> dict[str, Any]:
    """Return the versioned actor-recipe publication contract."""

    return {
        "schema": ACTOR_RECIPE_CANDIDATE_SCHEMA,
        "version": ACTOR_RECIPE_CANDIDATE_VERSION,
        "source_contract_sha256": crafting_contract_sha256(),
        "shape": {
            "prefix": ["batch", "actor"],
            "candidate_capacity": ACTOR_RECIPE_CANDIDATE_CAPACITY,
            "catalog_capacity": RECIPE_TABLE_CAPACITY,
            "maximum_inputs": MAX_INGREDIENTS,
            "maximum_outputs": MAX_OUTPUTS,
            "maximum_bench_requirements": MAX_BENCH_REQUIREMENTS,
        },
        "selection": {
            "source": "complete_host_resolved_recipe_table",
            "legal_predicate": (
                "current_actor_window_bench_knowledge_memories_and_materials"
            ),
            "ordering": "canonical_resolved_recipe_table_order",
            "hidden_recipe_invariance": True,
            "overflow": "fail_entire_actor_row_closed",
            "empty_legal_set": "available_with_empty_candidate_mask",
        },
        "policy_fields": list(ActorRecipeCandidatePolicy._fields),
        "privileged_execution_fields": [
            "recipe_index",
            "recipe_id_hash",
        ],
        "excluded_from_policy": [
            "native_recipe_string_id",
            "global_recipe_index",
            "catalog_provenance",
            "bridge_identity",
            "source_count",
            "emitted_count",
            "capacity_exceeded",
            "diagnostics",
        ],
        "rollout_interop": {
            "none_slot": True,
            "target_radix": ACTOR_RECIPE_TARGET_RADIX,
            "combined_target_radix_budget": (
                ACTOR_RECIPE_COMBINED_TARGET_RADIX_BUDGET
            ),
            "maximum_companion_target_radix": (
                ACTOR_RECIPE_COMBINED_TARGET_RADIX_BUDGET
                // ACTOR_RECIPE_TARGET_RADIX
            ),
            "symmetric_17_by_17_product": (
                ACTOR_RECIPE_TARGET_RADIX**2
            ),
            "consumer_guard": "validate_action_target_head_sizes",
        },
        "provenance": {
            "runtime_output": "excluded",
            "publication_requires": [
                "resolved_catalog_content_sha256",
                "candidate_contract_sha256",
                "bridge_sha256",
            ],
        },
        "fail_closed": True,
    }


def actor_recipe_candidate_contract_sha256() -> str:
    """Return the canonical contract identity."""

    payload = json.dumps(
        actor_recipe_candidate_contract(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def actor_recipe_candidate_provenance(
    resolved_catalog_content_sha256: str,
    bridge_sha256: str,
) -> dict[str, str]:
    """Bind host-only catalog and bridge identities to this candidate ABI."""

    return {
        "schema": "hytalerl_actor_recipe_candidate_provenance_v1",
        "resolved_catalog_content_sha256": _sha256(
            resolved_catalog_content_sha256,
            "resolved_catalog_content_sha256",
        ),
        "candidate_contract_sha256": (
            actor_recipe_candidate_contract_sha256()
        ),
        "bridge_sha256": _sha256(bridge_sha256, "bridge_sha256"),
    }


def produce_actor_recipe_candidates_from_legal_mask(
    table: RecipeTable,
    legal_recipe_mask: Array,
    *,
    catalog_available: Array | bool,
    legal_evidence_available: Array,
) -> ActorRecipeCandidates:
    """Compact complete legal evidence without exposing global recipe IDs."""

    if not isinstance(table, RecipeTable):
        raise TypeError("table must be a RecipeTable")
    legal = jnp.asarray(legal_recipe_mask)
    if legal.ndim != 3 or legal.shape[2] != RECIPE_TABLE_CAPACITY:
        raise ValueError(
            "legal_recipe_mask must have shape "
            f"[batch, actor, {RECIPE_TABLE_CAPACITY}]"
        )
    if legal.dtype != jnp.bool_:
        raise TypeError("legal_recipe_mask must be boolean")
    prefix = legal.shape[:2]
    evidence = _prefix_flag(
        legal_evidence_available,
        prefix,
        "legal_evidence_available",
    )
    catalog = _prefix_flag(
        catalog_available,
        prefix,
        "catalog_available",
    )
    table_mask = jnp.asarray(table.recipe_mask, dtype=jnp.bool_)
    if table_mask.shape != (RECIPE_TABLE_CAPACITY,):
        raise ValueError(
            f"table.recipe_mask must have shape {(RECIPE_TABLE_CAPACITY,)}"
        )

    complete = catalog & evidence
    invalid_reference = jnp.any(legal & ~table_mask, axis=2) & complete
    visible_legal = legal & table_mask[None, None, :] & complete[:, :, None]
    source_count = jnp.sum(visible_legal, axis=2, dtype=jnp.int32)
    capacity_exceeded = (
        source_count > jnp.int32(ACTOR_RECIPE_CANDIDATE_CAPACITY)
    ) & complete
    available = complete & ~invalid_reference & ~capacity_exceeded

    flat = visible_legal.reshape((-1, RECIPE_TABLE_CAPACITY))
    indexes = jax.vmap(_first_candidate_indexes)(flat).reshape(
        prefix + (ACTOR_RECIPE_CANDIDATE_CAPACITY,)
    )
    candidate_mask = (
        jnp.arange(ACTOR_RECIPE_CANDIDATE_CAPACITY)[None, None, :]
        < source_count[:, :, None]
    ) & available[:, :, None]

    policy = _gather_policy(table, indexes, candidate_mask)
    execution = ActorRecipeCandidateExecution(
        recipe_index=jnp.where(
            candidate_mask,
            indexes,
            jnp.int32(-1),
        ),
        recipe_id_hash=_candidate_where(
            candidate_mask,
            jnp.take(table.recipe_id_hash, indexes, axis=0),
        ),
    )
    diagnostics = (
        jnp.where(
            catalog,
            jnp.uint32(0),
            ACTOR_RECIPE_DIAGNOSTIC_CATALOG_UNAVAILABLE,
        )
        | jnp.where(
            evidence,
            jnp.uint32(0),
            ACTOR_RECIPE_DIAGNOSTIC_EVIDENCE_UNAVAILABLE,
        )
        | jnp.where(
            invalid_reference,
            ACTOR_RECIPE_DIAGNOSTIC_INVALID_REFERENCE,
            jnp.uint32(0),
        )
        | jnp.where(
            capacity_exceeded,
            ACTOR_RECIPE_DIAGNOSTIC_CAPACITY,
            jnp.uint32(0),
        )
    )
    return ActorRecipeCandidates(
        available=available,
        candidate_mask=candidate_mask,
        policy=policy,
        execution=execution,
        source_count=source_count,
        emitted_count=jnp.where(
            available,
            source_count,
            jnp.int32(0),
        ),
        capacity_exceeded=capacity_exceeded,
        diagnostics=diagnostics,
    )


def produce_actor_recipe_candidates(
    table: RecipeTable,
    inventory: CraftingInventory,
    context: CraftingContext,
    *,
    catalog_available: Array | bool,
    legal_evidence_available: Array,
    quantity: Array | int = 1,
) -> ActorRecipeCandidates:
    """Evaluate actor crafting state and publish its bounded legal row."""

    prefix = _actor_crafting_prefix(inventory, context)
    requested = jnp.asarray(quantity, dtype=jnp.int32)
    if requested.ndim == 0:
        requested = jnp.broadcast_to(requested, prefix)
    if requested.shape != prefix:
        raise ValueError("quantity must be scalar or have shape [batch, actor]")

    flat_inventory = jax.tree.map(
        lambda value: value.reshape((-1,) + value.shape[2:]),
        inventory,
    )
    flat_context = jax.tree.map(
        lambda value: value.reshape((-1,) + value.shape[2:]),
        context,
    )
    flat_quantity = requested.reshape((-1,))
    legal = jax.vmap(
        lambda one_inventory, one_context, one_quantity: satisfiable_recipes(
            table,
            one_inventory,
            one_context,
            one_quantity,
        )
    )(flat_inventory, flat_context, flat_quantity).reshape(
        prefix + (RECIPE_TABLE_CAPACITY,)
    )
    return produce_actor_recipe_candidates_from_legal_mask(
        table,
        legal,
        catalog_available=catalog_available,
        legal_evidence_available=legal_evidence_available,
    )


def _first_candidate_indexes(mask: Array) -> Array:
    return jnp.nonzero(
        mask,
        size=ACTOR_RECIPE_CANDIDATE_CAPACITY,
        fill_value=0,
    )[0].astype(jnp.int32)


def _gather_policy(
    table: RecipeTable,
    indexes: Array,
    candidate_mask: Array,
) -> ActorRecipeCandidatePolicy:
    def take(name: str) -> Array:
        return jnp.take(getattr(table, name), indexes, axis=0)

    input_mask = take("input_mask") & candidate_mask[..., None]
    output_mask = take("output_mask") & candidate_mask[..., None]
    requirement_mask = take("requirement_mask") & candidate_mask[..., None]
    return ActorRecipeCandidatePolicy(
        input_mask=input_mask,
        input_item_id=_candidate_where(input_mask, take("input_item_id")),
        input_resource_type_id=_candidate_where(
            input_mask,
            take("input_resource_type_id"),
        ),
        input_quantity=_candidate_where(
            input_mask,
            take("input_quantity"),
        ),
        input_metadata_required=_candidate_where(
            input_mask,
            take("input_metadata_required"),
        ),
        input_metadata_hash=_candidate_where(
            input_mask,
            take("input_metadata_hash"),
        ),
        output_mask=output_mask,
        output_item_id=_candidate_where(
            output_mask,
            take("output_item_id"),
        ),
        output_quantity=_candidate_where(
            output_mask,
            take("output_quantity"),
        ),
        output_metadata_hash=_candidate_where(
            output_mask,
            take("output_metadata_hash"),
        ),
        requirement_mask=requirement_mask,
        requirement_bench_type=_candidate_where(
            requirement_mask,
            take("requirement_bench_type"),
        ),
        requirement_bench_id_hash=_candidate_where(
            requirement_mask,
            take("requirement_bench_id_hash"),
        ),
        requirement_tier_level=_candidate_where(
            requirement_mask,
            take("requirement_tier_level"),
        ),
        knowledge_required=_candidate_where(
            candidate_mask,
            take("knowledge_required"),
        ),
        required_memories_level=_candidate_where(
            candidate_mask,
            take("required_memories_level"),
        ),
        time_seconds=_candidate_where(
            candidate_mask,
            take("time_seconds"),
        ),
    )


def _candidate_where(mask: Array, value: Array) -> Array:
    expanded = mask
    while expanded.ndim < value.ndim:
        expanded = expanded[..., None]
    return jnp.where(expanded, value, jnp.zeros((), dtype=value.dtype))


def _actor_crafting_prefix(
    inventory: CraftingInventory,
    context: CraftingContext,
) -> tuple[int, int]:
    if not isinstance(inventory, CraftingInventory):
        raise TypeError("inventory must be a CraftingInventory")
    if not isinstance(context, CraftingContext):
        raise TypeError("context must be a CraftingContext")
    if inventory.item_id.ndim != 3:
        raise ValueError("inventory fields must start with [batch, actor]")
    prefix = inventory.item_id.shape[:2]
    inventory_suffixes = {
        "item_id": 1,
        "quantity": 1,
        "resource_type_id": 2,
        "metadata_hash": 2,
    }
    context_suffixes = {
        "window_open": 0,
        "bench_type": 0,
        "bench_id_hash": 1,
        "bench_tier_level": 0,
        "memories_level": 0,
        "known_recipe": 1,
        "creative_mode": 0,
        "input_mode": 0,
    }
    for value, suffixes, label in (
        (inventory, inventory_suffixes, "inventory"),
        (context, context_suffixes, "context"),
    ):
        for name, suffix_count in suffixes.items():
            shape = getattr(value, name).shape
            if shape[:2] != prefix or len(shape) != 2 + suffix_count:
                raise ValueError(
                    f"{label}.{name} must start with [batch, actor]"
                )
    if context.known_recipe.shape[2:] != (RECIPE_TABLE_CAPACITY,):
        raise ValueError(
            "context.known_recipe has the wrong recipe capacity"
        )
    return prefix


def _prefix_flag(
    value: Array | bool,
    prefix: tuple[int, int],
    name: str,
) -> Array:
    flag = jnp.asarray(value)
    if flag.ndim == 0:
        flag = jnp.broadcast_to(flag, prefix)
    if flag.shape != prefix or flag.dtype != jnp.bool_:
        raise ValueError(f"{name} must be boolean with shape [batch, actor]")
    return flag


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    return value.upper()


__all__ = [
    "ACTOR_RECIPE_CANDIDATE_CAPACITY",
    "ACTOR_RECIPE_CANDIDATE_SCHEMA",
    "ACTOR_RECIPE_CANDIDATE_VERSION",
    "ACTOR_RECIPE_COMBINED_TARGET_RADIX_BUDGET",
    "ACTOR_RECIPE_DIAGNOSTIC_CAPACITY",
    "ACTOR_RECIPE_DIAGNOSTIC_CATALOG_UNAVAILABLE",
    "ACTOR_RECIPE_DIAGNOSTIC_EVIDENCE_UNAVAILABLE",
    "ACTOR_RECIPE_DIAGNOSTIC_INVALID_REFERENCE",
    "ACTOR_RECIPE_TARGET_RADIX",
    "ActorRecipeCandidateExecution",
    "ActorRecipeCandidatePolicy",
    "ActorRecipeCandidates",
    "actor_recipe_candidate_contract",
    "actor_recipe_candidate_contract_sha256",
    "actor_recipe_candidate_provenance",
    "produce_actor_recipe_candidates",
    "produce_actor_recipe_candidates_from_legal_mask",
]
