"""Array-only values for the standalone JAX crafting model."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax

if TYPE_CHECKING:
    from hytalegym.jax.combat.inventory import InventoryState


Array = jax.Array


class RecipeTable(NamedTuple):
    """One fixed-capacity, host-resolved native recipe table."""

    recipe_mask: Array
    recipe_id_hash: Array
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


class CraftingInventory(NamedTuple):
    """Minimal inventory interface consumed by crafting.

    Item and resource identifiers are caller-owned semantic ``int32`` IDs.
    This type deliberately does not import Combat's inventory package.
    """

    item_id: Array
    quantity: Array
    resource_type_id: Array
    metadata_hash: Array


class CraftingContext(NamedTuple):
    """World and per-player predicates needed to address one recipe."""

    window_open: Array
    bench_type: Array
    bench_id_hash: Array
    bench_tier_level: Array
    memories_level: Array
    known_recipe: Array
    creative_mode: Array
    input_mode: Array


class CraftingStepResult(NamedTuple):
    """One atomic crafting-completion macro transition."""

    inventory: CraftingInventory
    crafted: Array
    failure_bits: Array
    recipe_id_hash: Array
    output_item_id: Array
    output_quantity: Array
    output_metadata_hash: Array
    time_seconds: Array


class CombatCraftingCatalog(NamedTuple):
    """Host-resolved inventory data needed by the Combat adapter.

    Fields may omit leading batch/entity axes when normal broadcasting can
    supply them.
    """

    slot_resource_type_id: Array
    output_max_stack: Array
    output_max_durability: Array
    output_container_priority: Array
    output_slot_allowed: Array


class CraftedOutputRouting(NamedTuple):
    """Inventory insertion and world-drop effects for crafted outputs."""

    state: InventoryState
    inserted_quantity: Array
    world_drop_quantity: Array
    world_drop_mask: Array
    valid: Array


class CombatCraftingTransition(NamedTuple):
    """One batched crafting step committed to Combat inventory."""

    state: InventoryState
    crafting: CraftingStepResult
    inserted_quantity: Array
    world_drop_quantity: Array
    world_drop_mask: Array


__all__ = [
    "CombatCraftingCatalog",
    "CombatCraftingTransition",
    "CraftingContext",
    "CraftingInventory",
    "CraftingStepResult",
    "CraftedOutputRouting",
    "RecipeTable",
]
