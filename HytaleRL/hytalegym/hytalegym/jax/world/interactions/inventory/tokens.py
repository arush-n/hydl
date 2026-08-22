"""Configurable actor-legal semantic inventory tokens."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp


Array = jax.Array

ACTOR_INVENTORY_TOKEN_SCHEMA = "hytalerl_actor_inventory_tokens_v1"
ACTOR_INVENTORY_TOKEN_VERSION = 1

INVENTORY_TOKEN_PROVENANCE_NONE = 0
INVENTORY_TOKEN_PROVENANCE_NATIVE = 1
INVENTORY_TOKEN_PROVENANCE_SIMULATED = 2

ACTOR_INVENTORY_DIAGNOSTIC_LAYOUT = jnp.uint32(1)
ACTOR_INVENTORY_DIAGNOSTIC_REQUIRED_CONTAINER = jnp.uint32(1 << 1)
ACTOR_INVENTORY_DIAGNOSTIC_INVALID_STACK = jnp.uint32(1 << 2)
ACTOR_INVENTORY_DIAGNOSTIC_CAPACITY = jnp.uint32(1 << 3)
ACTOR_INVENTORY_DIAGNOSTIC_CONTAINER = jnp.uint32(1 << 4)
ACTOR_INVENTORY_DIAGNOSTIC_SOURCE = jnp.uint32(1 << 5)


class ActorInventoryTokens(NamedTuple):
    """Bounded occupied-stack tokens retaining native container semantics."""

    available: Array
    capacity_exceeded: Array
    diagnostics: Array
    container_available: Array
    container_capacity: Array
    token_mask: Array
    source_slot: Array
    container_id: Array
    container_slot: Array
    item_id: Array
    quantity: Array
    durability: Array
    max_durability: Array
    durability_fraction: Array
    metadata_hash: Array
    metadata_present: Array
    metadata_hash_valid: Array
    active: Array
    provenance: Array


def produce_actor_inventory_tokens(
    item_id: Array,
    quantity: Array,
    durability: Array,
    max_durability: Array,
    metadata_hash: Array,
    metadata_present: Array,
    metadata_hash_valid: Array,
    *,
    slot_container_id: Array,
    slot_container_index: Array,
    source_available: Array,
    container_available: Array,
    container_capacity: Array,
    active_container_slot: Array,
    actor_legal: Array,
    token_capacity: int,
    provenance: int,
    required_container_mask: Array | None = None,
) -> ActorInventoryTokens:
    """Publish occupied stacks in stable container/local-slot order.

    Container availability is independent. The caller chooses which
    containers are required for a complete actor row; unavailable optional
    containers are omitted without being misreported as empty.
    """

    capacity = _positive_static_int(token_capacity, "token_capacity")
    items = jnp.asarray(item_id)
    quantities = jnp.asarray(quantity)
    durability_value = jnp.asarray(durability)
    maximum = jnp.asarray(max_durability)
    metadata = jnp.asarray(metadata_hash)
    metadata_presence = jnp.asarray(metadata_present)
    metadata_validity = jnp.asarray(metadata_hash_valid)
    _validate_stack_arrays(
        items,
        quantities,
        durability_value,
        maximum,
        metadata,
        metadata_presence,
        metadata_validity,
    )
    if capacity > items.shape[2]:
        raise ValueError("token_capacity cannot exceed the source slot count")
    batch, actors, slot_count = items.shape
    source = jnp.asarray(source_available)
    if source.dtype != jnp.bool_ or source.shape != (batch, actors):
        raise TypeError("source_available must be boolean [B, A]")
    containers = jnp.asarray(container_available)
    container_capacities = jnp.asarray(container_capacity)
    active_slots = jnp.asarray(active_container_slot)
    legal = jnp.asarray(actor_legal)
    if containers.dtype != jnp.bool_ or containers.ndim != 3:
        raise TypeError(
            "container_available must be boolean [B, A, container]"
        )
    container_count = containers.shape[2]
    if containers.shape[:2] != (batch, actors) or container_count <= 0:
        raise ValueError("container_available has an invalid shape")
    if (
        container_capacities.dtype != jnp.int32
        or container_capacities.shape != containers.shape
    ):
        raise TypeError(
            "container_capacity must be int32 [B, A, container]"
        )
    if active_slots.dtype != jnp.int32 or active_slots.shape != (
        batch,
        actors,
        container_count,
    ):
        raise TypeError(
            "active_container_slot must be int32 [B, A, container]"
        )
    if legal.dtype != jnp.bool_ or legal.shape != (batch, actors):
        raise TypeError("actor_legal must be boolean [B, A]")
    source_container = jnp.asarray(slot_container_id)
    source_local = jnp.asarray(slot_container_index)
    if (
        source_container.dtype != jnp.int32
        or source_local.dtype != jnp.int32
    ):
        raise TypeError("slot layout arrays must have int32 dtype")
    if source_container.shape != (slot_count,) or source_local.shape != (
        slot_count,
    ):
        raise ValueError("slot layout arrays must match the source slot axis")
    required = _required_container_mask(
        required_container_mask,
        container_count,
    )
    provenance_value = _provenance_value(provenance)

    layout_range_valid = (
        (source_container >= 0)
        & (source_container < container_count)
        & (source_local >= 0)
        & (source_local < slot_count)
    )
    safe_container = jnp.clip(
        source_container,
        0,
        container_count - 1,
    )
    pair_equal = (
        (source_container[:, None] == source_container[None, :])
        & (source_local[:, None] == source_local[None, :])
    )
    duplicate_layout = jnp.any(
        pair_equal & jnp.triu(jnp.ones_like(pair_equal), k=1)
    )
    layout_valid = jnp.all(layout_range_valid) & ~duplicate_layout

    container_state_valid = (
        (container_capacities >= 0)
        & jnp.where(
            containers,
            active_slots >= -1,
            (container_capacities == 0) & (active_slots == -1),
        )
        & jnp.where(
            containers & (active_slots >= 0),
            active_slots < container_capacities,
            True,
        )
    )
    slot_available = (
        containers[:, :, safe_container]
        & layout_range_valid
        & (
            source_local[None, None, :]
            < container_capacities[:, :, safe_container]
        )
    )
    present = (items >= 0) & (quantities > 0)
    metadata_zero = jnp.all(metadata == 0, axis=3)
    metadata_state_valid = jnp.where(
        metadata_presence,
        metadata_validity | metadata_zero,
        metadata_validity & metadata_zero,
    )
    empty = (
        (items == -1)
        & (quantities == 0)
        & (durability_value == 0.0)
        & (maximum == 0.0)
        & metadata_zero
        & ~metadata_presence
        & ~metadata_validity
    )
    durable = (
        jnp.isfinite(durability_value)
        & jnp.isfinite(maximum)
        & (durability_value >= 0.0)
        & (maximum >= 0.0)
        & jnp.where(
            maximum > 0.0,
            durability_value <= maximum,
            durability_value == 0.0,
        )
    )
    valid_stack = (present & durable & metadata_state_valid) | empty
    unexpected_stack = (
        present
        & layout_range_valid[None, None, :]
        & ~slot_available
    )
    invalid_stack = jnp.any(
        (slot_available & ~valid_stack) | unexpected_stack,
        axis=2,
    )
    invalid_container = ~jnp.all(container_state_valid, axis=2)
    required_missing = jnp.any(
        required[None, None, :] & ~containers,
        axis=2,
    ) & source
    eligible = slot_available & present & valid_stack
    occupied_count = jnp.sum(eligible.astype(jnp.int32), axis=2)
    overflow = occupied_count > capacity

    source_index = jnp.arange(slot_count, dtype=jnp.int32)
    layout_key = source_container * jnp.int32(slot_count) + source_local
    layout_order = jnp.argsort(layout_key, stable=True)
    ordered_eligible = eligible[..., layout_order]
    ordered_source = jnp.broadcast_to(
        layout_order,
        (batch, actors, slot_count),
    )
    eligible_rank = (
        jnp.cumsum(ordered_eligible.astype(jnp.int32), axis=2) - 1
    )
    empty_rank = occupied_count[..., None] + (
        jnp.cumsum((~ordered_eligible).astype(jnp.int32), axis=2) - 1
    )
    destination = jnp.where(ordered_eligible, eligible_rank, empty_rank)
    selected = jnp.zeros_like(ordered_source).at[
        jnp.arange(batch, dtype=jnp.int32)[:, None, None],
        jnp.arange(actors, dtype=jnp.int32)[None, :, None],
        destination,
    ].set(ordered_source, unique_indices=True)[..., :capacity]
    selected_eligible = _gather_slots(eligible, selected)
    row_available = (
        legal
        & source
        & layout_valid
        & ~required_missing
        & ~invalid_container
        & ~invalid_stack
        & ~overflow
    )
    token_mask = selected_eligible & row_available[..., None]

    def gathered(value: Array) -> Array:
        return _gather_slots(value, selected)

    def masked(value: Array, fill: int | float | bool = 0) -> Array:
        value = gathered(value)
        gate = token_mask.reshape(
            token_mask.shape + (1,) * (value.ndim - token_mask.ndim)
        )
        return jnp.where(gate, value, jnp.full_like(value, fill))

    source_container_rows = jnp.broadcast_to(
        source_container,
        (batch, actors, slot_count),
    )
    source_local_rows = jnp.broadcast_to(
        source_local,
        (batch, actors, slot_count),
    )
    selected_container = gathered(source_container_rows)
    selected_local = gathered(source_local_rows)
    selected_active_slot = gathered(
        active_slots[:, :, safe_container]
    )
    selected_durability = masked(durability_value, 0.0)
    selected_maximum = masked(maximum, 0.0)
    durability_fraction = jnp.where(
        token_mask & (selected_maximum > 0.0),
        selected_durability / jnp.maximum(selected_maximum, 1.0),
        0.0,
    )
    return ActorInventoryTokens(
        available=row_available,
        capacity_exceeded=overflow & legal,
        diagnostics=(
            jnp.where(
                legal & ~layout_valid,
                ACTOR_INVENTORY_DIAGNOSTIC_LAYOUT,
                jnp.uint32(0),
            )
            | jnp.where(
                legal & required_missing,
                ACTOR_INVENTORY_DIAGNOSTIC_REQUIRED_CONTAINER,
                jnp.uint32(0),
            )
            | jnp.where(
                legal & invalid_stack,
                ACTOR_INVENTORY_DIAGNOSTIC_INVALID_STACK,
                jnp.uint32(0),
            )
            | jnp.where(
                legal & overflow,
                ACTOR_INVENTORY_DIAGNOSTIC_CAPACITY,
                jnp.uint32(0),
            )
            | jnp.where(
                legal & invalid_container,
                ACTOR_INVENTORY_DIAGNOSTIC_CONTAINER,
                jnp.uint32(0),
            )
            | jnp.where(
                legal & ~source,
                ACTOR_INVENTORY_DIAGNOSTIC_SOURCE,
                jnp.uint32(0),
            )
        ),
        container_available=containers & row_available[..., None],
        container_capacity=jnp.where(
            row_available[..., None],
            container_capacities,
            jnp.int32(0),
        ),
        token_mask=token_mask,
        source_slot=masked(
            jnp.broadcast_to(
                source_index,
                (batch, actors, slot_count),
            ),
            -1,
        ),
        container_id=jnp.where(
            token_mask,
            selected_container,
            jnp.int32(-1),
        ),
        container_slot=jnp.where(
            token_mask,
            selected_local,
            jnp.int32(-1),
        ),
        item_id=masked(items, -1),
        quantity=masked(quantities, 0),
        durability=selected_durability,
        max_durability=selected_maximum,
        durability_fraction=durability_fraction,
        metadata_hash=masked(metadata, 0),
        metadata_present=masked(metadata_presence, False),
        metadata_hash_valid=masked(metadata_validity, False),
        active=(
            token_mask
            & (selected_active_slot >= 0)
            & (selected_active_slot == selected_local)
        ),
        provenance=jnp.where(
            token_mask,
            jnp.uint8(provenance_value),
            jnp.uint8(INVENTORY_TOKEN_PROVENANCE_NONE),
        ),
    )


def actor_inventory_token_contract() -> dict[str, object]:
    """Return the configurable semantic inventory-token contract."""

    return {
        "schema": ACTOR_INVENTORY_TOKEN_SCHEMA,
        "version": ACTOR_INVENTORY_TOKEN_VERSION,
        "input": {
            "stack": ["batch", "actor", "source_slot"],
            "metadata": [
                "batch",
                "actor",
                "source_slot",
                "metadata_words",
            ],
            "metadata_present": ["batch", "actor", "source_slot"],
            "metadata_hash_valid": ["batch", "actor", "source_slot"],
            "container_availability": (
                "independent_per_actor_per_container"
            ),
            "source_available": (
                "inventory_component_present_independent_of_container_state"
            ),
            "container_capacity": (
                "dynamic_native_int32_capacity_zero_when_unavailable"
            ),
            "required_containers": (
                "caller_selected_static_mask_default_all"
            ),
        },
        "output": {
            "shape": ["batch", "actor", "token_capacity"],
            "capacity": "caller_selected_static_positive_integer",
            "ordering": "source_container_then_local_slot",
            "policy_fields": [
                "token_mask",
                "container_available",
                "container_capacity",
                "container_id",
                "container_slot",
                "item_id",
                "quantity",
                "durability_fraction",
                "active",
                "provenance",
            ],
            "verification_fields": [
                "source_slot",
                "durability",
                "max_durability",
                "metadata_hash",
                "metadata_present",
                "metadata_hash_valid",
                "container_available",
            ],
        },
        "identity": {
            "item_id": "caller_owned_int32_semantic_catalog_id",
            "metadata_hash": "caller_owned_canonical_digest_words",
            "native_runtime_index": (
                "sorted_asset_table_index_starting_at_one; zero_rejected_"
                "as_unmapped_by_native_host_adapter"
            ),
            "metadata_boundary": (
                "presence_is_exact; digest_validity_is_independent; "
                "present_without_digest_keeps_zero_hash_and_validity_false"
            ),
        },
        "quantity": "raw_positive_int32_not_presence_bit",
        "availability": (
            "unavailable_container_distinct_from_empty_container"
        ),
        "privilege": (
            "actor_legal_mask_applied_before_any_stack_or_container_value"
        ),
        "overflow": "count_all_eligible_stacks_then_clear_complete_actor_row",
        "fail_closed": [
            "invalid_or_duplicate_slot_layout",
            "invalid_container_capacity_or_active_slot",
            "required_container_unavailable",
            "inventory_source_unavailable",
            "item_quantity_or_durability_inconsistent",
            "metadata_presence_or_digest_state_inconsistent",
            "token_capacity_exceeded",
        ],
        "native_host_adapter": (
            "validated_sparse_inventory_v2_to_configured_dense_source_layout"
        ),
        "consumer": (
            "additive_semantic_token_surface_not_in_current_learner_v3"
        ),
    }


def actor_inventory_token_contract_sha256() -> str:
    payload = json.dumps(
        actor_inventory_token_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_stack_arrays(
    item_id: Array,
    quantity: Array,
    durability: Array,
    maximum: Array,
    metadata: Array,
    metadata_present: Array,
    metadata_hash_valid: Array,
) -> None:
    if item_id.dtype != jnp.int32 or item_id.ndim != 3:
        raise TypeError("item_id must be int32 [B, A, slot]")
    if quantity.dtype != jnp.int32 or quantity.shape != item_id.shape:
        raise TypeError("quantity must be int32 and match item_id")
    if durability.dtype != jnp.float32 or durability.shape != item_id.shape:
        raise TypeError("durability must be float32 and match item_id")
    if maximum.dtype != jnp.float32 or maximum.shape != item_id.shape:
        raise TypeError("max_durability must be float32 and match item_id")
    if (
        metadata.dtype != jnp.uint32
        or metadata.ndim != 4
        or metadata.shape[:3] != item_id.shape
        or metadata.shape[3] <= 0
    ):
        raise TypeError(
            "metadata_hash must be uint32 [B, A, slot, metadata_words]"
        )
    if (
        metadata_present.dtype != jnp.bool_
        or metadata_present.shape != item_id.shape
    ):
        raise TypeError(
            "metadata_present must be boolean and match item_id"
        )
    if (
        metadata_hash_valid.dtype != jnp.bool_
        or metadata_hash_valid.shape != item_id.shape
    ):
        raise TypeError(
            "metadata_hash_valid must be boolean and match item_id"
        )


def _required_container_mask(value: Array | None, count: int) -> Array:
    if value is None:
        return jnp.ones((count,), dtype=jnp.bool_)
    result = jnp.asarray(value)
    if result.dtype != jnp.bool_ or result.shape != (count,):
        raise TypeError(
            "required_container_mask must be boolean [container]"
        )
    return result


def _gather_slots(value: Array, index: Array) -> Array:
    expanded = index.reshape(
        index.shape + (1,) * (value.ndim - index.ndim)
    )
    expanded = jnp.broadcast_to(
        expanded,
        index.shape + value.shape[3:],
    )
    return jnp.take_along_axis(value, expanded, axis=2)


def _provenance_value(value: int) -> int:
    if isinstance(value, bool) or value not in (
        INVENTORY_TOKEN_PROVENANCE_NATIVE,
        INVENTORY_TOKEN_PROVENANCE_SIMULATED,
    ):
        raise ValueError("provenance must identify native or simulated data")
    return value


def _positive_static_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


__all__ = [
    "ACTOR_INVENTORY_DIAGNOSTIC_CAPACITY",
    "ACTOR_INVENTORY_DIAGNOSTIC_CONTAINER",
    "ACTOR_INVENTORY_DIAGNOSTIC_INVALID_STACK",
    "ACTOR_INVENTORY_DIAGNOSTIC_LAYOUT",
    "ACTOR_INVENTORY_DIAGNOSTIC_REQUIRED_CONTAINER",
    "ACTOR_INVENTORY_TOKEN_SCHEMA",
    "ACTOR_INVENTORY_TOKEN_VERSION",
    "ActorInventoryTokens",
    "INVENTORY_TOKEN_PROVENANCE_NATIVE",
    "INVENTORY_TOKEN_PROVENANCE_NONE",
    "INVENTORY_TOKEN_PROVENANCE_SIMULATED",
    "actor_inventory_token_contract",
    "actor_inventory_token_contract_sha256",
    "produce_actor_inventory_tokens",
]
