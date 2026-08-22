"""Array-only inventory values used by JAX combat."""

from __future__ import annotations

from typing import NamedTuple

import jax


Array = jax.Array


class InventoryLayout(NamedTuple):
    """Static flattening of six native containers into one slot axis."""

    capacities: Array
    offsets: Array
    container_id: Array
    container_slot: Array


class ItemStack(NamedTuple):
    """Native ItemStack fields represented on batch/entity axes."""

    item_id: Array
    quantity: Array
    durability: Array
    max_durability: Array
    metadata_hash: Array


class MaterialQuantity(NamedTuple):
    """One item, resource type, and/or tag requirement."""

    item_id: Array
    resource_type_id: Array
    tag_id: Array
    quantity: Array
    metadata_hash: Array


class ResourceQuantity(NamedTuple):
    """One resource-type quantity requirement."""

    resource_id: Array
    quantity: Array


class InventoryState(NamedTuple):
    """Fixed-shape inventory for every batch entity."""

    item_id: Array
    quantity: Array
    durability: Array
    max_durability: Array
    metadata_hash: Array
    active_hotbar_slot: Array
    active_utility_slot: Array
    active_tools_slot: Array
    failure_bits: Array


class InventoryTransaction(NamedTuple):
    """Result of one vectorized insertion request."""

    state: InventoryState
    remainder_quantity: Array
    accepted: Array
    changed: Array


__all__ = [
    "InventoryLayout",
    "InventoryState",
    "InventoryTransaction",
    "ItemStack",
    "MaterialQuantity",
    "ResourceQuantity",
]

