"""Versioned fixed-shape contract for the standalone crafting model."""

from __future__ import annotations

import hashlib
import json
from typing import Any


CRAFTING_SCHEMA = "hytalerl_crafting_v1"
CRAFTING_VERSION = 1

# Source: tests/fidelity/surrogate_crafting_source_audit.py:26.
# Installed 0.5.7 has 403 explicit plus 1,544 item-generated recipes.
RECIPE_TABLE_CAPACITY = 1947
# Source: Assets.zip!Server/Item/Items/Deco/Deco_Trophy_Harvest.json:11-124.
MAX_INGREDIENTS = 28
# Source: Assets.zip!Server/Item/Recipes/Salvage/Salvage_Armor_Thorium_Chest.json:12-29.
MAX_OUTPUTS = 4
# Source: Assets.zip!Server/Item/Items/Weapon/Shortbow/Weapon_Shortbow_Crude.json:23-45.
MAX_BENCH_REQUIREMENTS = 3
# Source: tests/fidelity/surrogate_crafting_source_audit.py:33.
# The installed item corpus has at most seven ResourceTypes on one item.
MAX_RESOURCE_TYPES_PER_ITEM = 7

# Contract representation: a full SHA-256 digest encoded as big-endian uint32.
IDENTITY_HASH_WORDS = hashlib.sha256().digest_size // 4
# Source: hytalegym/jax/combat/inventory/contract.py:38.
# Duplicated intentionally: this is a structural boundary, not a hard import.
METADATA_HASH_WORDS = 2
# Source: hytalegym/jax/combat/inventory/contract.py:37.
ABSENT_ID = -1

# Source: com/hypixel/hytale/protocol/BenchType.java:5-9.
BENCH_TYPE_CRAFTING = 0
BENCH_TYPE_PROCESSING = 1
BENCH_TYPE_DIAGRAM_CRAFTING = 2
BENCH_TYPE_STRUCTURAL_CRAFTING = 3
BENCH_TYPE_COUNT = 4

# Source: com/hypixel/hytale/builtin/crafting/component/CraftingManager.java:985-987.
INPUT_MODE_NORMAL = 0
INPUT_MODE_ORDERED = 1
INPUT_MODE_COUNT = 2

# Contract-owned failure bits; these are not native Hytale enum values.
CRAFTING_FAILURE_INVALID_REQUEST = 1 << 0
CRAFTING_FAILURE_INVALID_STATE = 1 << 1
CRAFTING_FAILURE_PRECONDITION = 1 << 2
CRAFTING_FAILURE_INGREDIENTS = 1 << 3


def identity_sha256_words(value: str) -> tuple[int, ...]:
    """Return a stable, device-safe identity for one native string asset key."""

    if not isinstance(value, str) or not value:
        raise ValueError("identity value must be a non-empty string")
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return tuple(
        int.from_bytes(digest[index : index + 4], "big")
        for index in range(0, len(digest), 4)
    )


def crafting_contract_manifest() -> dict[str, Any]:
    """Return the canonical standalone crafting contract."""

    return {
        "schema": CRAFTING_SCHEMA,
        "version": CRAFTING_VERSION,
        "source_baseline": {
            "release": "0.5.7",
            "explicit_recipe_assets": 403,
            "item_generated_recipe_assets": 1544,
            "recipe_assets_total": RECIPE_TABLE_CAPACITY,
            "maximum_inputs": MAX_INGREDIENTS,
            "maximum_outputs": MAX_OUTPUTS,
            "maximum_bench_requirements": MAX_BENCH_REQUIREMENTS,
            "maximum_item_resource_types": MAX_RESOURCE_TYPES_PER_ITEM,
        },
        "identity": {
            "native_recipe_key": "utf8_string_asset_id",
            "native_window_action": "CraftRecipeAction.recipeId_string",
            "device_representation": {
                "algorithm": "sha256",
                "dtype": "uint32",
                "word_order": "big_endian",
                "words": IDENTITY_HASH_WORDS,
            },
            "recipe_index": ("table_local_only_never_a_native_or_wire_recipe_identity"),
        },
        "recipe_table": {
            "capacity": RECIPE_TABLE_CAPACITY,
            "input_shape": [
                RECIPE_TABLE_CAPACITY,
                MAX_INGREDIENTS,
            ],
            "output_shape": [
                RECIPE_TABLE_CAPACITY,
                MAX_OUTPUTS,
            ],
            "bench_requirement_shape": [
                RECIPE_TABLE_CAPACITY,
                MAX_BENCH_REQUIREMENTS,
            ],
            "padding": {
                "mask": False,
                "semantic_id": ABSENT_ID,
                "quantity": 0,
                "hash_word": 0,
            },
            "material_selector": {
                "fields": ["item_id", "resource_type_id"],
                "precedence": "item_id_then_resource_type_id",
                "quantity": "positive_when_masked",
                "metadata": (
                    "optional_caller_supplied_canonical_bson_sha256_prefix_64"
                ),
                "native_tag_selector": (
                    "supported_by_installed_MaterialQuantity_clone_but_absent_"
                    "from_installed_recipe_assets_and_not_encoded_in_v1"
                ),
            },
            "output": {
                "item_id_required": True,
                "quantity": "positive_when_masked",
                "insertion": "owned_by_downstream_inventory_adapter",
                "native_stack_defaults": (
                    "durability_and_max_durability_from_item_asset"
                ),
                "native_overflow": "insert_what_fits_then_drop_remainder",
            },
        },
        "inventory_interface": {
            "fields": {
                "item_id": ["slots", "int32"],
                "quantity": ["slots", "int32"],
                "resource_type_id": [
                    "slots",
                    MAX_RESOURCE_TYPES_PER_ITEM,
                    "int32",
                ],
                "metadata_hash": [
                    "slots",
                    METADATA_HASH_WORDS,
                    "uint32",
                ],
            },
            "empty": {
                "item_id": ABSENT_ID,
                "quantity": 0,
                "resource_type_id": ABSENT_ID,
                "metadata_hash": 0,
            },
            "combat_dependency": (
                "inventory_adapter_imports_combat_transaction_primitives"
            ),
        },
        "inventory_boundary": {
            "simple_and_field_input_priority": [
                "backpack",
                "storage",
                "hotbar",
            ],
            "simple_extra_resources": (
                "adjacent_bench_containers_follow_player_inventory"
            ),
            "diagram_and_structural_inputs": (
                "ordered_window_local_containers_not_player_inventory"
            ),
            "default_output_priority": [
                "hotbar",
                "storage",
                "backpack",
            ],
            "output_priority_source": ("player_settings_and_item_asset_classification"),
            "output_catalog_fields": [
                "slot_resource_type_id",
                "max_stack",
                "max_durability",
                "container_priority",
                "slot_acceptance",
            ],
            "overflow": "world_drop_remainder_without_reverting_craft",
            "production_combat_adapter": True,
            "environment_consumer_connected": False,
            "adapter": {
                "state_axes": ["batch", "entity", "slot"],
                "input_projection": (
                    "static_unique_flat_slots_in_native_removal_order"
                ),
                "input_commit": (
                    "preserve_surviving_durability_and_canonicalize_empty_slots"
                ),
                "output_routing": (
                    "host_resolved_three_container_priority_using_combat_add_item_stack"
                ),
                "world_drop_effect": ("fixed_output_remainder_returned_to_world_owner"),
                "world_drop_spawner": False,
                "jit": True,
            },
        },
        "context": {
            "window_open": ("bench_window_or_pocket_crafting_window_is_active"),
            "bench": {
                "type_values": {
                    "crafting": BENCH_TYPE_CRAFTING,
                    "processing": BENCH_TYPE_PROCESSING,
                    "diagram_crafting": BENCH_TYPE_DIAGRAM_CRAFTING,
                    "structural_crafting": BENCH_TYPE_STRUCTURAL_CRAFTING,
                },
                "id": "sha256_words_of_native_string_id",
                "tier": "nonnegative_int32",
                "fieldcraft": (
                    "type_crafting_id_Fieldcraft_tier_zero_no_physical_bench"
                ),
            },
            "known_recipe": [
                RECIPE_TABLE_CAPACITY,
                "bool_host_resolved_from_PlayerConfigData.KnownRecipes",
            ],
            "memories_level": "positive_int32",
            "creative_mode": "bypasses_input_consumption",
            "input_modes": {
                "normal": INPUT_MODE_NORMAL,
                "ordered": INPUT_MODE_ORDERED,
            },
        },
        "runtime": {
            "satisfiable_recipes": (
                "window_bench_tier_learning_memories_and_sequential_material_allocation"
            ),
            "step_crafting": (
                "atomic_completed_craft_macro_consumes_inputs_and_emits_"
                "fixed_width_outputs"
            ),
            "craft_into_combat_inventory": (
                "batched_projection_step_commit_route_and_drop_remainder"
            ),
            "route_crafting_outputs": (
                "shared_output_route_for_player_or_window_local_crafting"
            ),
            "quantity": "positive_int32_with_overflow_rejected",
            "normal_removal_order": "input_order_then_slot_order",
            "ordered_removal": "input_i_from_slot_i",
            "timing": ("time_seconds_reported_but_queue_progress_is_not_simulated"),
            "fail_closed": True,
        },
        "bridge_findings": {
            "existing_action": (
                "craftRecipeId_int_is_simulator_registry_only_and_does_not_"
                "address_native_string_recipe_assets"
            ),
            "existing_observation": (
                "craftable_count_is_inventory_satisfiability_for_"
                "BaseBuilderTask_but_native_backend_emits_zero"
            ),
        },
        "not_claimed": [
            "native_execution",
            "native_differential_certification",
            "bench_navigation_or_interaction",
            "processing_bench_fuel_and_tick_state",
            "crafting_queue_progress_or_cancellation",
            "diagram_discovery_key_consistency",
            "adjacent_bench_resource_container_projection",
            "item_catalog_or_player_settings_resolution",
            "world_item_drop_spawning",
        ],
    }


def crafting_contract_json() -> str:
    return json.dumps(
        crafting_contract_manifest(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def crafting_contract_sha256() -> str:
    return hashlib.sha256(crafting_contract_json().encode("utf-8")).hexdigest().upper()


__all__ = [
    "ABSENT_ID",
    "BENCH_TYPE_COUNT",
    "BENCH_TYPE_CRAFTING",
    "BENCH_TYPE_DIAGRAM_CRAFTING",
    "BENCH_TYPE_PROCESSING",
    "BENCH_TYPE_STRUCTURAL_CRAFTING",
    "CRAFTING_FAILURE_INGREDIENTS",
    "CRAFTING_FAILURE_INVALID_REQUEST",
    "CRAFTING_FAILURE_INVALID_STATE",
    "CRAFTING_FAILURE_PRECONDITION",
    "CRAFTING_SCHEMA",
    "CRAFTING_VERSION",
    "IDENTITY_HASH_WORDS",
    "INPUT_MODE_COUNT",
    "INPUT_MODE_NORMAL",
    "INPUT_MODE_ORDERED",
    "MAX_BENCH_REQUIREMENTS",
    "MAX_INGREDIENTS",
    "MAX_OUTPUTS",
    "MAX_RESOURCE_TYPES_PER_ITEM",
    "METADATA_HASH_WORDS",
    "RECIPE_TABLE_CAPACITY",
    "crafting_contract_json",
    "crafting_contract_manifest",
    "crafting_contract_sha256",
    "identity_sha256_words",
]
