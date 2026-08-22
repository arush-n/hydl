"""Quantity-aware fixed-shape inventory for JAX combat."""

from hytalegym.jax.combat.inventory.schema.contract import *
from hytalegym.jax.combat.inventory.runtime.engine import (
    add_item_stack,
    default_inventory_layout,
    empty_inventory_state,
    inventory_from_loadout,
    inventory_state_failure_bits,
    item_stack_at,
    item_stack_valid,
    material_quantity_valid,
    remove_item_stack,
    resource_quantity_valid,
    set_active_hotbar_slot,
)
from hytalegym.jax.combat.inventory.adapters.native import (
    NATIVE_INVENTORY_CONTAINER_ORDER,
    NATIVE_INVENTORY_SCHEMA,
    NATIVE_INVENTORY_SECTION_IDS,
    NATIVE_INVENTORY_VERSION,
    NATIVE_INVENTORY_V1_SCHEMA,
    NATIVE_INVENTORY_V1_VERSION,
    native_inventory_contract_manifest,
    native_inventory_contract_sha256,
    native_inventory_v1_contract_sha256,
    parse_native_inventory_frame,
)
from hytalegym.jax.combat.inventory.schema.types import (
    InventoryLayout,
    InventoryState,
    InventoryTransaction,
    ItemStack,
    MaterialQuantity,
    ResourceQuantity,
)


__all__ = [
    "InventoryLayout",
    "InventoryState",
    "InventoryTransaction",
    "ItemStack",
    "MaterialQuantity",
    "NATIVE_INVENTORY_CONTAINER_ORDER",
    "NATIVE_INVENTORY_SCHEMA",
    "NATIVE_INVENTORY_SECTION_IDS",
    "NATIVE_INVENTORY_VERSION",
    "NATIVE_INVENTORY_V1_SCHEMA",
    "NATIVE_INVENTORY_V1_VERSION",
    "ResourceQuantity",
    "add_item_stack",
    "default_inventory_layout",
    "empty_inventory_state",
    "inventory_from_loadout",
    "inventory_state_failure_bits",
    "item_stack_at",
    "item_stack_valid",
    "material_quantity_valid",
    "native_inventory_contract_manifest",
    "native_inventory_contract_sha256",
    "native_inventory_v1_contract_sha256",
    "parse_native_inventory_frame",
    "remove_item_stack",
    "resource_quantity_valid",
    "set_active_hotbar_slot",
]
__all__ += [
    name
    for name in globals()
    if name.isupper() or name.startswith("inventory_contract_")
]
