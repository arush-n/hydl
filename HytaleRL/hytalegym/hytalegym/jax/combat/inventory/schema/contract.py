"""Versioned fixed-shape inventory contract for combat runtimes."""

from __future__ import annotations

import hashlib
import json
from typing import Any


INVENTORY_SCHEMA = "hytalerl_combat_inventory_v1"
INVENTORY_VERSION = 1

CONTAINER_STORAGE = 0
CONTAINER_ARMOR = 1
CONTAINER_HOTBAR = 2
CONTAINER_UTILITY = 3
CONTAINER_TOOLS = 4
CONTAINER_BACKPACK = 5
CONTAINER_COUNT = 6

STORAGE_CAPACITY = 36
ARMOR_CAPACITY = 4
HOTBAR_CAPACITY = 9
UTILITY_CAPACITY = 4
TOOLS_CAPACITY = 23
DEFAULT_BACKPACK_CAPACITY = 0
DEFAULT_CONTAINER_CAPACITIES = (
    STORAGE_CAPACITY,
    ARMOR_CAPACITY,
    HOTBAR_CAPACITY,
    UTILITY_CAPACITY,
    TOOLS_CAPACITY,
    DEFAULT_BACKPACK_CAPACITY,
)

EMPTY_ITEM_ID = -1
ABSENT_MATERIAL_ID = -1
METADATA_HASH_WORDS = 2

INVENTORY_FAILURE_INVALID_STATE = 1 << 0
INVENTORY_FAILURE_INVALID_REQUEST = 1 << 1


def inventory_contract_manifest() -> dict[str, Any]:
    """Return the canonical device inventory contract."""

    return {
        "schema": INVENTORY_SCHEMA,
        "version": INVENTORY_VERSION,
        "containers": {
            "order": [
                "storage",
                "armor",
                "hotbar",
                "utility",
                "tools",
                "backpack",
            ],
            "capacities": {
                "storage": STORAGE_CAPACITY,
                "armor": ARMOR_CAPACITY,
                "hotbar": HOTBAR_CAPACITY,
                "utility": UTILITY_CAPACITY,
                "tools": TOOLS_CAPACITY,
                "backpack": "runtime_static",
            },
            "representation": (
                "flat_slots_with_explicit_container_and_local_slot_axes"
            ),
            "armor_slot_order": ["head", "chest", "hands", "legs"],
            "active_slots": {
                "hotbar": "zero_to_8",
                "utility": "minus_one_or_zero_to_3",
                "tools": "minus_one_or_zero_to_22",
            },
        },
        "item_stack": {
            "fields": {
                "item_id": "int32_semantic_id",
                "quantity": "int32_positive_when_present",
                "durability": "float32_nonnegative",
                "max_durability": "float32_nonnegative",
                "metadata_hash": ["uint32", METADATA_HASH_WORDS],
            },
            "empty": {
                "item_id": EMPTY_ITEM_ID,
                "quantity": 0,
                "durability": 0.0,
                "max_durability": 0.0,
                "metadata_hash": [0, 0],
            },
        },
        "material_quantity": {
            "selectors": ["item_id", "resource_type_id", "tag_id"],
            "selector_rule": "at_least_one_present",
            "quantity_rule": "positive",
        },
        "resource_quantity": {
            "resource_id_rule": "present",
            "quantity_rule": "positive",
        },
        "stacking": {
            "identity": [
                "item_id",
                "durability",
                "max_durability",
                "metadata_hash",
            ],
            "max_stack_source": "item_asset_catalog",
            "metadata_representation": (
                "caller_supplied_canonical_bson_sha256_prefix_64"
            ),
            "order": "existing_stackable_slots_then_empty_slots",
            "slot_order": "ascending_local_slot",
            "all_or_nothing": "preflight_equivalent_atomic_rollback",
            "full_stacks": "skip_existing_stackable_slots",
            "filters": "caller_supplied_slot_acceptance_mask_fail_closed",
            "remove_order": "compatible_slots_ascending_local_slot",
        },
        "boundaries": {
            "metadata": (
                "device_compares_canonical_digest_only_host_owns_bson_codec"
            ),
            "item_catalog": (
                "caller_resolves_max_stack_resource_types_tags_and_item_filters"
            ),
            "not_claimed": [
                "crafting",
                "block_harvest",
                "container_priority_routing",
                "native_inventory_differential",
            ],
        },
    }


def inventory_contract_json() -> str:
    return json.dumps(
        inventory_contract_manifest(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def inventory_contract_sha256() -> str:
    return hashlib.sha256(inventory_contract_json().encode("utf-8")).hexdigest().upper()


__all__ = [
    "ABSENT_MATERIAL_ID",
    "ARMOR_CAPACITY",
    "CONTAINER_ARMOR",
    "CONTAINER_BACKPACK",
    "CONTAINER_COUNT",
    "CONTAINER_HOTBAR",
    "CONTAINER_STORAGE",
    "CONTAINER_TOOLS",
    "CONTAINER_UTILITY",
    "DEFAULT_BACKPACK_CAPACITY",
    "DEFAULT_CONTAINER_CAPACITIES",
    "EMPTY_ITEM_ID",
    "HOTBAR_CAPACITY",
    "INVENTORY_FAILURE_INVALID_REQUEST",
    "INVENTORY_FAILURE_INVALID_STATE",
    "INVENTORY_SCHEMA",
    "INVENTORY_VERSION",
    "METADATA_HASH_WORDS",
    "STORAGE_CAPACITY",
    "TOOLS_CAPACITY",
    "UTILITY_CAPACITY",
    "inventory_contract_json",
    "inventory_contract_manifest",
    "inventory_contract_sha256",
]
