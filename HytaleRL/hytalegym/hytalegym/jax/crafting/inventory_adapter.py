"""Batched Combat-inventory adapter for completed crafting transitions."""

from __future__ import annotations

import operator
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat.inventory import (
    CONTAINER_BACKPACK,
    CONTAINER_COUNT,
    CONTAINER_HOTBAR,
    CONTAINER_STORAGE,
    EMPTY_ITEM_ID,
    InventoryLayout,
    InventoryState,
    ItemStack,
    add_item_stack,
    inventory_state_failure_bits,
)
from hytalegym.jax.crafting.contract import (
    ABSENT_ID,
    CRAFTING_FAILURE_INVALID_STATE,
    IDENTITY_HASH_WORDS,
    MAX_OUTPUTS,
    MAX_RESOURCE_TYPES_PER_ITEM,
    METADATA_HASH_WORDS,
    RECIPE_TABLE_CAPACITY,
)
from hytalegym.jax.crafting.runtime import step_crafting
from hytalegym.jax.crafting.types import (
    CombatCraftingCatalog,
    CombatCraftingTransition,
    CraftedOutputRouting,
    CraftingContext,
    CraftingInventory,
    CraftingStepResult,
    RecipeTable,
)


PICKUP_PRIORITY_HOTBAR = (
    CONTAINER_HOTBAR,
    CONTAINER_STORAGE,
    CONTAINER_BACKPACK,
)
PICKUP_PRIORITY_STORAGE = (
    CONTAINER_STORAGE,
    CONTAINER_HOTBAR,
    CONTAINER_BACKPACK,
)
PICKUP_PRIORITY_BACKPACK = (
    CONTAINER_BACKPACK,
    CONTAINER_STORAGE,
    CONTAINER_HOTBAR,
)
PICKUP_CONTAINER_PRIORITY_COUNT = 3
_PICKUP_CONTAINERS = frozenset(PICKUP_PRIORITY_HOTBAR)


def player_crafting_input_slot_order(
    layout: InventoryLayout,
) -> tuple[int, ...]:
    """Return native simple/field player-input order for one static layout."""

    if not isinstance(layout, InventoryLayout):
        raise TypeError("layout must be an InventoryLayout")
    container_id = np.asarray(layout.container_id)
    container_slot = np.asarray(layout.container_slot)
    if container_id.ndim != 1 or container_slot.shape != container_id.shape:
        raise ValueError("layout slot arrays must be matching vectors")
    order = np.concatenate(
        tuple(
            np.flatnonzero(container_id == container)
            for container in PICKUP_PRIORITY_BACKPACK
        )
    )
    return tuple(int(slot) for slot in order)


def validate_combat_crafting_catalog(
    state: InventoryState,
    layout: InventoryLayout,
    catalog: CombatCraftingCatalog,
    *,
    input_slot_order: Sequence[int],
) -> None:
    """Raise when host-resolved adapter data is malformed."""

    batch, entities, slots = _state_shape(state, layout)
    order = _normalize_input_slot_order(input_slot_order, slots)
    normalized = _normalize_catalog(catalog, batch, entities, slots)
    resources = np.asarray(normalized.slot_resource_type_id)
    if np.any(resources < ABSENT_ID):
        raise ValueError("slot_resource_type_id contains an invalid negative ID")
    empty = np.asarray(state.item_id) == EMPTY_ITEM_ID
    if np.any(empty[..., None] & (resources != ABSENT_ID)):
        raise ValueError("empty inventory slots cannot expose resource IDs")
    if np.any(np.asarray(normalized.output_max_stack) < 0):
        raise ValueError("output_max_stack must be nonnegative")
    durability = np.asarray(normalized.output_max_durability)
    if np.any(~np.isfinite(durability)) or np.any(durability < 0):
        raise ValueError("output_max_durability must be finite and nonnegative")
    priority = np.asarray(normalized.output_container_priority)
    expected = np.sort(np.asarray(tuple(_PICKUP_CONTAINERS), dtype=np.int32))
    if np.any(np.sort(priority, axis=-1) != expected):
        raise ValueError(
            "each output_container_priority must contain hotbar, storage, "
            "and backpack exactly once"
        )
    if np.asarray(normalized.output_slot_allowed).dtype != np.bool_:
        raise ValueError("output_slot_allowed must have bool dtype")
    if not order:
        raise ValueError("input_slot_order cannot be empty")


def craft_into_combat_inventory(
    table: RecipeTable,
    state: InventoryState,
    layout: InventoryLayout,
    context: CraftingContext,
    recipe_index,
    catalog: CombatCraftingCatalog,
    *,
    quantity=1,
    input_slot_order: Sequence[int],
) -> CombatCraftingTransition:
    """Craft every requested batch/entity lane and commit inventory effects."""

    batch, entities, slots = _state_shape(state, layout)
    order = _normalize_input_slot_order(input_slot_order, slots)
    order_array = jnp.asarray(order, dtype=jnp.int32)
    normalized_catalog = _normalize_catalog(catalog, batch, entities, slots)
    batched_context = _normalize_context(context, batch, entities)
    indexes = _broadcast_int32(recipe_index, (batch, entities), "recipe_index")
    quantities = _broadcast_int32(quantity, (batch, entities), "quantity")

    projected = CraftingInventory(
        item_id=jnp.take(state.item_id, order_array, axis=2),
        quantity=jnp.take(state.quantity, order_array, axis=2),
        resource_type_id=jnp.take(
            normalized_catalog.slot_resource_type_id,
            order_array,
            axis=2,
        ),
        metadata_hash=jnp.take(state.metadata_hash, order_array, axis=2),
    )
    raw = _batched_step(
        table,
        projected,
        batched_context,
        indexes,
        quantities,
        batch,
        entities,
    )
    combat_valid = (inventory_state_failure_bits(state, layout) == jnp.uint32(0))[
        :, None
    ]
    adapter_valid = combat_valid & _active_output_catalog_valid(
        raw,
        normalized_catalog,
    )
    crafting = _gate_crafting_result(raw, projected, adapter_valid)
    committed = _commit_crafting_inventory(
        state,
        crafting.inventory,
        order_array,
    )
    routing = route_crafting_outputs(
        committed,
        layout,
        crafting,
        normalized_catalog,
    )
    return CombatCraftingTransition(
        state=routing.state,
        crafting=crafting,
        inserted_quantity=routing.inserted_quantity,
        world_drop_quantity=routing.world_drop_quantity,
        world_drop_mask=routing.world_drop_mask,
    )


def route_crafting_outputs(
    state: InventoryState,
    layout: InventoryLayout,
    crafting: CraftingStepResult,
    catalog: CombatCraftingCatalog,
) -> CraftedOutputRouting:
    """Route batched crafted outputs and expose remainders for world spawning."""

    batch, entities, slots = _state_shape(state, layout)
    normalized_crafting = _normalize_crafting_result(
        crafting,
        batch,
        entities,
    )
    normalized_catalog = _normalize_catalog(catalog, batch, entities, slots)
    valid = (inventory_state_failure_bits(state, layout) == jnp.uint32(0))[
        :, None
    ] & _active_output_catalog_valid(
        normalized_crafting,
        normalized_catalog,
    )
    next_state = state
    remaining_outputs = []
    for output in range(MAX_OUTPUTS):
        remaining = normalized_crafting.output_quantity[:, :, output]
        output_active = (
            normalized_crafting.crafted
            & valid
            & (normalized_crafting.output_item_id[:, :, output] >= 0)
            & (remaining > 0)
        )
        for priority in range(PICKUP_CONTAINER_PRIORITY_COUNT):
            container = normalized_catalog.output_container_priority[
                :, :, output, priority
            ]
            valid_container = (container >= 0) & (container < CONTAINER_COUNT)
            safe_container = jnp.clip(container, 0, CONTAINER_COUNT - 1)
            capacity = jnp.take(layout.capacities, safe_container)
            requested = (
                output_active & (remaining > 0) & valid_container & (capacity > 0)
            )
            transaction = add_item_stack(
                next_state,
                layout,
                ItemStack(
                    item_id=normalized_crafting.output_item_id[:, :, output],
                    quantity=remaining,
                    durability=normalized_catalog.output_max_durability[:, :, output],
                    max_durability=normalized_catalog.output_max_durability[
                        :, :, output
                    ],
                    metadata_hash=normalized_crafting.output_metadata_hash[
                        :, :, output, :
                    ],
                ),
                item_max_stack=normalized_catalog.output_max_stack[:, :, output],
                container_id=container,
                slot_allowed=normalized_catalog.output_slot_allowed[:, :, output, :],
                request_mask=requested,
            )
            next_state = transaction.state
            remaining = transaction.remainder_quantity
        remaining_outputs.append(remaining)

    remaining = jnp.stack(remaining_outputs, axis=2)
    completed = normalized_crafting.crafted & valid
    world_drop_quantity = jnp.where(
        completed[:, :, None],
        remaining,
        jnp.int32(0),
    )
    inserted_quantity = jnp.where(
        completed[:, :, None],
        normalized_crafting.output_quantity - remaining,
        jnp.int32(0),
    )
    return CraftedOutputRouting(
        state=next_state,
        inserted_quantity=inserted_quantity,
        world_drop_quantity=world_drop_quantity,
        world_drop_mask=completed[:, :, None] & (remaining > 0),
        valid=valid,
    )


def _batched_step(
    table: RecipeTable,
    inventory: CraftingInventory,
    context: CraftingContext,
    recipe_index: jax.Array,
    quantity: jax.Array,
    batch: int,
    entities: int,
) -> CraftingStepResult:
    lanes = batch * entities
    flat_inventory = jax.tree.map(
        lambda value: value.reshape((lanes,) + value.shape[2:]),
        inventory,
    )
    flat_context = jax.tree.map(
        lambda value: value.reshape((lanes,) + value.shape[2:]),
        context,
    )
    flat = jax.vmap(
        lambda lane_inventory, lane_context, lane_index, lane_quantity: (
            step_crafting(
                table,
                lane_inventory,
                lane_context,
                lane_index,
                lane_quantity,
            )
        )
    )(
        flat_inventory,
        flat_context,
        recipe_index.reshape((lanes,)),
        quantity.reshape((lanes,)),
    )
    return jax.tree.map(
        lambda value: value.reshape((batch, entities) + value.shape[1:]),
        flat,
    )


def _gate_crafting_result(
    result: CraftingStepResult,
    original: CraftingInventory,
    adapter_valid: jax.Array,
) -> CraftingStepResult:
    accepted = result.crafted & adapter_valid
    invalid_adapter = result.crafted & ~adapter_valid
    return CraftingStepResult(
        inventory=CraftingInventory(
            item_id=jnp.where(
                accepted[:, :, None],
                result.inventory.item_id,
                original.item_id,
            ),
            quantity=jnp.where(
                accepted[:, :, None],
                result.inventory.quantity,
                original.quantity,
            ),
            resource_type_id=jnp.where(
                accepted[:, :, None, None],
                result.inventory.resource_type_id,
                original.resource_type_id,
            ),
            metadata_hash=jnp.where(
                accepted[:, :, None, None],
                result.inventory.metadata_hash,
                original.metadata_hash,
            ),
        ),
        crafted=accepted,
        failure_bits=result.failure_bits
        | jnp.where(
            invalid_adapter,
            jnp.uint32(CRAFTING_FAILURE_INVALID_STATE),
            jnp.uint32(0),
        ),
        recipe_id_hash=jnp.where(
            accepted[:, :, None],
            result.recipe_id_hash,
            jnp.uint32(0),
        ),
        output_item_id=jnp.where(
            accepted[:, :, None],
            result.output_item_id,
            jnp.int32(ABSENT_ID),
        ),
        output_quantity=jnp.where(
            accepted[:, :, None],
            result.output_quantity,
            jnp.int32(0),
        ),
        output_metadata_hash=jnp.where(
            accepted[:, :, None, None],
            result.output_metadata_hash,
            jnp.uint32(0),
        ),
        time_seconds=jnp.where(
            accepted,
            result.time_seconds,
            jnp.float32(0.0),
        ),
    )


def _commit_crafting_inventory(
    state: InventoryState,
    inventory: CraftingInventory,
    input_slot_order: jax.Array,
) -> InventoryState:
    emptied = inventory.item_id == ABSENT_ID
    durability = jnp.take(state.durability, input_slot_order, axis=2)
    max_durability = jnp.take(
        state.max_durability,
        input_slot_order,
        axis=2,
    )
    return state._replace(
        item_id=state.item_id.at[:, :, input_slot_order].set(inventory.item_id),
        quantity=state.quantity.at[:, :, input_slot_order].set(inventory.quantity),
        durability=state.durability.at[:, :, input_slot_order].set(
            jnp.where(emptied, jnp.float32(0.0), durability)
        ),
        max_durability=state.max_durability.at[:, :, input_slot_order].set(
            jnp.where(emptied, jnp.float32(0.0), max_durability)
        ),
        metadata_hash=state.metadata_hash.at[:, :, input_slot_order, :].set(
            inventory.metadata_hash
        ),
    )


def _active_output_catalog_valid(
    crafting: CraftingStepResult,
    catalog: CombatCraftingCatalog,
) -> jax.Array:
    active = (crafting.output_item_id >= 0) & (crafting.output_quantity > 0)
    priority = catalog.output_container_priority
    priority_id_valid = (
        (priority == CONTAINER_HOTBAR)
        | (priority == CONTAINER_STORAGE)
        | (priority == CONTAINER_BACKPACK)
    )
    priority_distinct = (
        (priority[..., 0] != priority[..., 1])
        & (priority[..., 0] != priority[..., 2])
        & (priority[..., 1] != priority[..., 2])
    )
    output_valid = (
        (catalog.output_max_stack > 0)
        & jnp.isfinite(catalog.output_max_durability)
        & (catalog.output_max_durability >= 0.0)
        & jnp.all(priority_id_valid, axis=3)
        & priority_distinct
    )
    return jnp.all(~active | output_valid, axis=2)


def _normalize_crafting_result(
    result: CraftingStepResult,
    batch: int,
    entities: int,
) -> CraftingStepResult:
    if not isinstance(result, CraftingStepResult):
        raise TypeError("crafting must be a CraftingStepResult")
    if result.crafted.shape == ():
        if (batch, entities) != (1, 1):
            raise ValueError("scalar crafting result requires one batch and one entity")
        return jax.tree.map(lambda value: value[None, None, ...], result)
    expected = {
        "crafted": (batch, entities),
        "failure_bits": (batch, entities),
        "recipe_id_hash": (batch, entities, IDENTITY_HASH_WORDS),
        "output_item_id": (batch, entities, MAX_OUTPUTS),
        "output_quantity": (batch, entities, MAX_OUTPUTS),
        "output_metadata_hash": (
            batch,
            entities,
            MAX_OUTPUTS,
            METADATA_HASH_WORDS,
        ),
        "time_seconds": (batch, entities),
    }
    for field, shape in expected.items():
        actual = tuple(getattr(result, field).shape)
        if actual != shape:
            raise ValueError(f"crafting.{field} must have shape {shape}, got {actual}")
    return result


def _normalize_context(
    context: CraftingContext,
    batch: int,
    entities: int,
) -> CraftingContext:
    if not isinstance(context, CraftingContext):
        raise TypeError("context must be a CraftingContext")
    leading = (batch, entities)
    return CraftingContext(
        window_open=_broadcast_field(
            context.window_open,
            leading,
            "context.window_open",
        ),
        bench_type=_broadcast_field(
            context.bench_type,
            leading,
            "context.bench_type",
        ),
        bench_id_hash=_broadcast_field(
            context.bench_id_hash,
            leading + (IDENTITY_HASH_WORDS,),
            "context.bench_id_hash",
        ),
        bench_tier_level=_broadcast_field(
            context.bench_tier_level,
            leading,
            "context.bench_tier_level",
        ),
        memories_level=_broadcast_field(
            context.memories_level,
            leading,
            "context.memories_level",
        ),
        known_recipe=_broadcast_field(
            context.known_recipe,
            leading + (RECIPE_TABLE_CAPACITY,),
            "context.known_recipe",
        ),
        creative_mode=_broadcast_field(
            context.creative_mode,
            leading,
            "context.creative_mode",
        ),
        input_mode=_broadcast_field(
            context.input_mode,
            leading,
            "context.input_mode",
        ),
    )


def _normalize_catalog(
    catalog: CombatCraftingCatalog,
    batch: int,
    entities: int,
    slots: int,
) -> CombatCraftingCatalog:
    if not isinstance(catalog, CombatCraftingCatalog):
        raise TypeError("catalog must be a CombatCraftingCatalog")
    leading = (batch, entities)
    return CombatCraftingCatalog(
        slot_resource_type_id=_broadcast_field(
            catalog.slot_resource_type_id,
            leading + (slots, MAX_RESOURCE_TYPES_PER_ITEM),
            "catalog.slot_resource_type_id",
        ).astype(jnp.int32),
        output_max_stack=_broadcast_field(
            catalog.output_max_stack,
            leading + (MAX_OUTPUTS,),
            "catalog.output_max_stack",
        ).astype(jnp.int32),
        output_max_durability=_broadcast_field(
            catalog.output_max_durability,
            leading + (MAX_OUTPUTS,),
            "catalog.output_max_durability",
        ).astype(jnp.float32),
        output_container_priority=_broadcast_field(
            catalog.output_container_priority,
            leading + (MAX_OUTPUTS, PICKUP_CONTAINER_PRIORITY_COUNT),
            "catalog.output_container_priority",
        ).astype(jnp.int32),
        output_slot_allowed=_broadcast_field(
            catalog.output_slot_allowed,
            leading + (MAX_OUTPUTS, slots),
            "catalog.output_slot_allowed",
        ).astype(jnp.bool_),
    )


def _state_shape(
    state: InventoryState,
    layout: InventoryLayout,
) -> tuple[int, int, int]:
    if not isinstance(state, InventoryState):
        raise TypeError("state must be an InventoryState")
    if not isinstance(layout, InventoryLayout):
        raise TypeError("layout must be an InventoryLayout")
    if state.item_id.ndim != 3:
        raise ValueError("state.item_id must have shape [batch, entity, slot]")
    batch, entities, slots = state.item_id.shape
    if layout.container_id.shape != (slots,):
        raise ValueError("layout slot axis does not match state")
    inventory_state_failure_bits(state, layout)
    return batch, entities, slots


def _normalize_input_slot_order(
    input_slot_order: Sequence[int],
    slots: int,
) -> tuple[int, ...]:
    try:
        order = tuple(operator.index(slot) for slot in input_slot_order)
    except TypeError as error:
        raise TypeError("input_slot_order must be a sequence of integers") from error
    if not order:
        raise ValueError("input_slot_order cannot be empty")
    if len(set(order)) != len(order):
        raise ValueError("input_slot_order cannot contain duplicate slots")
    if min(order) < 0 or max(order) >= slots:
        raise ValueError("input_slot_order contains an out-of-range slot")
    return order


def _broadcast_field(value, shape: tuple[int, ...], name: str) -> jax.Array:
    try:
        return jnp.broadcast_to(jnp.asarray(value), shape)
    except ValueError as error:
        raise ValueError(f"{name} must broadcast to {shape}") from error


def _broadcast_int32(
    value,
    shape: tuple[int, int],
    name: str,
) -> jax.Array:
    try:
        return jnp.broadcast_to(jnp.asarray(value, dtype=jnp.int32), shape)
    except ValueError as error:
        raise ValueError(f"{name} must broadcast to {shape}") from error


__all__ = [
    "PICKUP_CONTAINER_PRIORITY_COUNT",
    "PICKUP_PRIORITY_BACKPACK",
    "PICKUP_PRIORITY_HOTBAR",
    "PICKUP_PRIORITY_STORAGE",
    "craft_into_combat_inventory",
    "player_crafting_input_slot_order",
    "route_crafting_outputs",
    "validate_combat_crafting_catalog",
]
