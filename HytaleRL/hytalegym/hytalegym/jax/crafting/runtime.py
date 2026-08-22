"""Pure-JAX satisfiability and atomic crafting-completion transitions."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.crafting.contract import (
    ABSENT_ID,
    BENCH_TYPE_COUNT,
    CRAFTING_FAILURE_INGREDIENTS,
    CRAFTING_FAILURE_INVALID_REQUEST,
    CRAFTING_FAILURE_INVALID_STATE,
    CRAFTING_FAILURE_PRECONDITION,
    IDENTITY_HASH_WORDS,
    INPUT_MODE_COUNT,
    INPUT_MODE_ORDERED,
    MAX_BENCH_REQUIREMENTS,
    MAX_INGREDIENTS,
    MAX_OUTPUTS,
    MAX_RESOURCE_TYPES_PER_ITEM,
    METADATA_HASH_WORDS,
    RECIPE_TABLE_CAPACITY,
)
from hytalegym.jax.crafting.types import (
    CraftingContext,
    CraftingInventory,
    CraftingStepResult,
    RecipeTable,
)


_TABLE_SHAPES: Mapping[str, tuple[int, ...]] = {
    "recipe_mask": (RECIPE_TABLE_CAPACITY,),
    "recipe_id_hash": (RECIPE_TABLE_CAPACITY, IDENTITY_HASH_WORDS),
    "input_mask": (RECIPE_TABLE_CAPACITY, MAX_INGREDIENTS),
    "input_item_id": (RECIPE_TABLE_CAPACITY, MAX_INGREDIENTS),
    "input_resource_type_id": (
        RECIPE_TABLE_CAPACITY,
        MAX_INGREDIENTS,
    ),
    "input_quantity": (RECIPE_TABLE_CAPACITY, MAX_INGREDIENTS),
    "input_metadata_required": (
        RECIPE_TABLE_CAPACITY,
        MAX_INGREDIENTS,
    ),
    "input_metadata_hash": (
        RECIPE_TABLE_CAPACITY,
        MAX_INGREDIENTS,
        METADATA_HASH_WORDS,
    ),
    "output_mask": (RECIPE_TABLE_CAPACITY, MAX_OUTPUTS),
    "output_item_id": (RECIPE_TABLE_CAPACITY, MAX_OUTPUTS),
    "output_quantity": (RECIPE_TABLE_CAPACITY, MAX_OUTPUTS),
    "output_metadata_hash": (
        RECIPE_TABLE_CAPACITY,
        MAX_OUTPUTS,
        METADATA_HASH_WORDS,
    ),
    "requirement_mask": (
        RECIPE_TABLE_CAPACITY,
        MAX_BENCH_REQUIREMENTS,
    ),
    "requirement_bench_type": (
        RECIPE_TABLE_CAPACITY,
        MAX_BENCH_REQUIREMENTS,
    ),
    "requirement_bench_id_hash": (
        RECIPE_TABLE_CAPACITY,
        MAX_BENCH_REQUIREMENTS,
        IDENTITY_HASH_WORDS,
    ),
    "requirement_tier_level": (
        RECIPE_TABLE_CAPACITY,
        MAX_BENCH_REQUIREMENTS,
    ),
    "knowledge_required": (RECIPE_TABLE_CAPACITY,),
    "required_memories_level": (RECIPE_TABLE_CAPACITY,),
    "time_seconds": (RECIPE_TABLE_CAPACITY,),
}


def validate_recipe_table(table: RecipeTable) -> None:
    """Raise when a host recipe table cannot be consumed safely."""

    if not isinstance(table, RecipeTable):
        raise TypeError("table must be a RecipeTable")
    for field, expected in _TABLE_SHAPES.items():
        actual = tuple(getattr(table, field).shape)
        if actual != expected:
            raise ValueError(f"table.{field} must have shape {expected}, got {actual}")

    arrays = {field: np.asarray(getattr(table, field)) for field in table._fields}
    active = arrays["recipe_mask"].astype(np.bool_)
    if np.any(active & ~np.any(arrays["recipe_id_hash"] != 0, axis=1)):
        raise ValueError("every active recipe needs a nonzero recipe_id_hash")

    _validate_left_packed(
        arrays["input_mask"],
        active,
        "input_mask",
    )
    _validate_left_packed(
        arrays["output_mask"],
        active,
        "output_mask",
    )
    _validate_left_packed(
        arrays["requirement_mask"],
        active,
        "requirement_mask",
    )

    inputs = active[:, None] & arrays["input_mask"].astype(np.bool_)
    selectors = (arrays["input_item_id"] >= 0) | (arrays["input_resource_type_id"] >= 0)
    if np.any(inputs & ~selectors):
        raise ValueError("every masked input needs an item or resource selector")
    if np.any(inputs & (arrays["input_quantity"] <= 0)):
        raise ValueError("every masked input quantity must be positive")

    outputs = active[:, None] & arrays["output_mask"].astype(np.bool_)
    if np.any(active & ~np.any(outputs, axis=1)):
        raise ValueError("every active recipe needs at least one output")
    if np.any(outputs & (arrays["output_item_id"] < 0)):
        raise ValueError("every masked output needs an item_id")
    if np.any(outputs & (arrays["output_quantity"] <= 0)):
        raise ValueError("every masked output quantity must be positive")

    requirements = active[:, None] & arrays["requirement_mask"].astype(np.bool_)
    if np.any(active & ~np.any(requirements, axis=1)):
        raise ValueError("every active recipe needs a bench requirement")
    bench_type = arrays["requirement_bench_type"]
    if np.any(requirements & ((bench_type < 0) | (bench_type >= BENCH_TYPE_COUNT))):
        raise ValueError("masked bench requirement type is invalid")
    if np.any(requirements & ~np.any(arrays["requirement_bench_id_hash"] != 0, axis=2)):
        raise ValueError("masked bench requirement needs a nonzero id hash")
    if np.any(requirements & (arrays["requirement_tier_level"] < 0)):
        raise ValueError("masked bench tier must be nonnegative")
    if np.any(active & (arrays["required_memories_level"] < 1)):
        raise ValueError("active recipe memories level must be at least one")
    if np.any(active & (arrays["time_seconds"] < 0)):
        raise ValueError("active recipe time_seconds must be nonnegative")


def validate_inventory(inventory: CraftingInventory) -> None:
    """Raise when the minimal inventory interface is malformed."""

    if not isinstance(inventory, CraftingInventory):
        raise TypeError("inventory must be a CraftingInventory")
    _require_inventory_shapes(inventory)
    item_id = np.asarray(inventory.item_id)
    quantity = np.asarray(inventory.quantity)
    resources = np.asarray(inventory.resource_type_id)
    present = item_id >= 0
    if np.any(item_id < ABSENT_ID):
        raise ValueError("inventory item_id contains an invalid negative ID")
    if np.any(resources < ABSENT_ID):
        raise ValueError("inventory resource_type_id has an invalid negative ID")
    if np.any(present & (quantity <= 0)):
        raise ValueError("present inventory items need positive quantity")
    if np.any(~present & (quantity != 0)):
        raise ValueError("empty inventory slots need zero quantity")
    if np.any(~present[:, None] & (resources != ABSENT_ID)):
        raise ValueError("empty inventory slots cannot expose resource IDs")


def validate_context(context: CraftingContext) -> None:
    """Raise when host-resolved crafting predicates are malformed."""

    if not isinstance(context, CraftingContext):
        raise TypeError("context must be a CraftingContext")
    _require_context_shapes(context)
    bench_type = int(np.asarray(context.bench_type))
    input_mode = int(np.asarray(context.input_mode))
    if bench_type < 0 or bench_type >= BENCH_TYPE_COUNT:
        raise ValueError("context bench_type is invalid")
    if input_mode < 0 or input_mode >= INPUT_MODE_COUNT:
        raise ValueError("context input_mode is invalid")
    if int(np.asarray(context.bench_tier_level)) < 0:
        raise ValueError("context bench_tier_level must be nonnegative")
    if int(np.asarray(context.memories_level)) < 1:
        raise ValueError("context memories_level must be at least one")
    if not np.any(np.asarray(context.bench_id_hash) != 0):
        raise ValueError("context bench_id_hash must be nonzero")


def satisfiable_recipes(
    table: RecipeTable,
    inventory: CraftingInventory,
    context: CraftingContext,
    quantity=1,
) -> jax.Array:
    """Return one fixed recipe-axis mask for the current crafting state."""

    _require_runtime_shapes(table, inventory, context)
    requested_quantity = jnp.asarray(quantity, dtype=jnp.int32)
    if requested_quantity.shape != ():
        raise ValueError("quantity must be a scalar")
    indexes = jnp.arange(RECIPE_TABLE_CAPACITY, dtype=jnp.int32)

    def one(index):
        evaluation = _evaluate_recipe(
            table,
            inventory,
            context,
            index,
            requested_quantity,
        )
        return evaluation["craftable"]

    return jax.vmap(one)(indexes)


def craftable_recipe_count(
    table: RecipeTable,
    inventory: CraftingInventory,
    context: CraftingContext,
    quantity=1,
) -> jax.Array:
    """Count fully eligible and satisfiable recipes."""

    return jnp.sum(
        satisfiable_recipes(table, inventory, context, quantity),
        dtype=jnp.int32,
    )


def step_crafting(
    table: RecipeTable,
    inventory: CraftingInventory,
    context: CraftingContext,
    recipe_index,
    quantity=1,
) -> CraftingStepResult:
    """Complete one craft atomically and emit outputs for inventory routing."""

    _require_runtime_shapes(table, inventory, context)
    index = jnp.asarray(recipe_index, dtype=jnp.int32)
    requested_quantity = jnp.asarray(quantity, dtype=jnp.int32)
    if index.shape != ():
        raise ValueError("recipe_index must be a scalar")
    if requested_quantity.shape != ():
        raise ValueError("quantity must be a scalar")

    evaluation = _evaluate_recipe(
        table,
        inventory,
        context,
        index,
        requested_quantity,
    )
    accepted = evaluation["craftable"]
    remaining = evaluation["remaining"]
    emptied = remaining == 0
    candidate = CraftingInventory(
        item_id=jnp.where(emptied, jnp.int32(ABSENT_ID), inventory.item_id),
        quantity=remaining,
        resource_type_id=jnp.where(
            emptied[:, None],
            jnp.int32(ABSENT_ID),
            inventory.resource_type_id,
        ),
        metadata_hash=jnp.where(
            emptied[:, None],
            jnp.uint32(0),
            inventory.metadata_hash,
        ),
    )
    output_quantity = evaluation["output_quantity"] * requested_quantity
    return CraftingStepResult(
        inventory=jax.tree.map(
            lambda changed, original: jnp.where(
                accepted,
                changed,
                original,
            ),
            candidate,
            inventory,
        ),
        crafted=accepted,
        failure_bits=_failure_bits(evaluation),
        recipe_id_hash=jnp.where(
            accepted,
            evaluation["recipe_id_hash"],
            jnp.uint32(0),
        ),
        output_item_id=jnp.where(
            accepted & evaluation["output_mask"],
            evaluation["output_item_id"],
            jnp.int32(ABSENT_ID),
        ),
        output_quantity=jnp.where(
            accepted & evaluation["output_mask"],
            output_quantity,
            jnp.int32(0),
        ),
        output_metadata_hash=jnp.where(
            (accepted & evaluation["output_mask"])[:, None],
            evaluation["output_metadata_hash"],
            jnp.uint32(0),
        ),
        time_seconds=jnp.where(
            accepted,
            evaluation["time_seconds"] * requested_quantity.astype(jnp.float32),
            jnp.float32(0.0),
        ),
    )


def _evaluate_recipe(
    table: RecipeTable,
    inventory: CraftingInventory,
    context: CraftingContext,
    recipe_index: jax.Array,
    quantity: jax.Array,
) -> dict[str, jax.Array]:
    index_valid = (recipe_index >= 0) & (recipe_index < RECIPE_TABLE_CAPACITY)
    safe_index = jnp.clip(recipe_index, 0, RECIPE_TABLE_CAPACITY - 1)
    recipe = {
        name: jnp.take(getattr(table, name), safe_index, axis=0)
        for name in table._fields
    }
    recipe["_known"] = jnp.take(context.known_recipe, safe_index, axis=0)
    scale_valid = _quantity_scale_valid(recipe, quantity)
    request_valid = index_valid & (quantity > 0) & scale_valid
    inventory_valid = _inventory_runtime_valid(inventory)
    context_valid = _context_runtime_valid(context)
    recipe_valid = _recipe_runtime_valid(recipe)
    state_valid = inventory_valid & context_valid & recipe_valid
    precondition = _recipe_precondition(recipe, context)
    remaining, sufficient = _consume_inputs(
        recipe,
        inventory,
        quantity,
        context.input_mode,
    )
    sufficient = sufficient | context.creative_mode
    remaining = jnp.where(
        context.creative_mode,
        inventory.quantity,
        remaining,
    )
    craftable = request_valid & state_valid & precondition & sufficient
    return {
        **recipe,
        "request_valid": request_valid,
        "state_valid": state_valid,
        "precondition": precondition,
        "sufficient": sufficient,
        "craftable": craftable,
        "remaining": remaining,
    }


def _quantity_scale_valid(
    recipe: dict[str, jax.Array],
    quantity: jax.Array,
) -> jax.Array:
    positive = quantity > 0
    divisor = jnp.maximum(quantity, jnp.int32(1))
    maximum = jnp.iinfo(jnp.int32).max // divisor
    inputs = jnp.all(~recipe["input_mask"] | (recipe["input_quantity"] <= maximum))
    outputs = jnp.all(~recipe["output_mask"] | (recipe["output_quantity"] <= maximum))
    return positive & inputs & outputs


def _recipe_precondition(
    recipe: dict[str, jax.Array],
    context: CraftingContext,
) -> jax.Array:
    requirement_match = (
        recipe["requirement_mask"]
        & (recipe["requirement_bench_type"] == context.bench_type)
        & jnp.all(
            recipe["requirement_bench_id_hash"] == context.bench_id_hash[None, :],
            axis=1,
        )
        & (recipe["requirement_tier_level"] <= context.bench_tier_level)
    )
    # ``known_recipe`` is recipe-axis state, so the selected value is supplied
    # by _evaluate_recipe below rather than inferred from an identity mapping.
    knowledge = ~recipe["knowledge_required"] | recipe["_known"]
    memories = (recipe["required_memories_level"] <= 1) | (
        context.memories_level >= recipe["required_memories_level"]
    )
    return context.window_open & jnp.any(requirement_match) & knowledge & memories


def _consume_inputs(
    recipe: dict[str, jax.Array],
    inventory: CraftingInventory,
    quantity: jax.Array,
    input_mode: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    normal = _consume_inputs_normal(recipe, inventory, quantity)
    ordered = _consume_inputs_ordered(recipe, inventory, quantity)
    return jax.tree.map(
        lambda normal_value, ordered_value: jnp.where(
            input_mode == INPUT_MODE_ORDERED,
            ordered_value,
            normal_value,
        ),
        normal,
        ordered,
    )


def _consume_inputs_normal(
    recipe: dict[str, jax.Array],
    inventory: CraftingInventory,
    quantity: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    slot_count = inventory.item_id.shape[0]

    def ingredient_body(input_index, carry):
        remaining, all_sufficient = carry
        masked = recipe["input_mask"][input_index]
        needed = recipe["input_quantity"][input_index] * quantity

        def slot_body(slot, slot_carry):
            slot_remaining, still_needed = slot_carry
            matches = _slot_matches(
                recipe,
                inventory,
                input_index,
                slot,
            )
            take = jnp.where(
                masked & matches,
                jnp.minimum(slot_remaining[slot], still_needed),
                jnp.int32(0),
            )
            return slot_remaining.at[slot].add(-take), still_needed - take

        remaining, needed = jax.lax.fori_loop(
            0,
            slot_count,
            slot_body,
            (remaining, needed),
        )
        sufficient = ~masked | (needed == 0)
        return remaining, all_sufficient & sufficient

    return jax.lax.fori_loop(
        0,
        MAX_INGREDIENTS,
        ingredient_body,
        (inventory.quantity, jnp.bool_(True)),
    )


def _consume_inputs_ordered(
    recipe: dict[str, jax.Array],
    inventory: CraftingInventory,
    quantity: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    slot_count = inventory.item_id.shape[0]

    def ingredient_body(input_index, carry):
        remaining, all_sufficient = carry
        masked = recipe["input_mask"][input_index]
        slot_valid = input_index < slot_count
        safe_slot = jnp.minimum(input_index, slot_count - 1)
        needed = recipe["input_quantity"][input_index] * quantity
        matches = _slot_matches(
            recipe,
            inventory,
            input_index,
            safe_slot,
        )
        sufficient = ~masked | (slot_valid & matches & (remaining[safe_slot] >= needed))
        take = jnp.where(masked & sufficient, needed, jnp.int32(0))
        return (
            remaining.at[safe_slot].add(-take),
            all_sufficient & sufficient,
        )

    return jax.lax.fori_loop(
        0,
        MAX_INGREDIENTS,
        ingredient_body,
        (inventory.quantity, jnp.bool_(True)),
    )


def _slot_matches(
    recipe: dict[str, jax.Array],
    inventory: CraftingInventory,
    input_index: jax.Array,
    slot: jax.Array,
) -> jax.Array:
    item_selector = recipe["input_item_id"][input_index]
    resource_selector = recipe["input_resource_type_id"][input_index]
    item_match = inventory.item_id[slot] == item_selector
    resource_match = jnp.any(inventory.resource_type_id[slot] == resource_selector)
    selector_match = jnp.where(
        item_selector >= 0,
        item_match,
        (resource_selector >= 0) & resource_match,
    )
    metadata_match = ~recipe["input_metadata_required"][input_index] | jnp.all(
        inventory.metadata_hash[slot] == recipe["input_metadata_hash"][input_index]
    )
    return (
        (inventory.item_id[slot] >= 0)
        & (inventory.quantity[slot] > 0)
        & selector_match
        & metadata_match
    )


def _recipe_runtime_valid(recipe: dict[str, jax.Array]) -> jax.Array:
    input_valid = ~recipe["input_mask"] | (
        ((recipe["input_item_id"] >= 0) | (recipe["input_resource_type_id"] >= 0))
        & (recipe["input_quantity"] > 0)
    )
    output_valid = ~recipe["output_mask"] | (
        (recipe["output_item_id"] >= 0) & (recipe["output_quantity"] > 0)
    )
    requirement_valid = ~recipe["requirement_mask"] | (
        (recipe["requirement_bench_type"] >= 0)
        & (recipe["requirement_bench_type"] < BENCH_TYPE_COUNT)
        & jnp.any(
            recipe["requirement_bench_id_hash"] != 0,
            axis=1,
        )
        & (recipe["requirement_tier_level"] >= 0)
    )
    return (
        recipe["recipe_mask"]
        & jnp.any(recipe["recipe_id_hash"] != 0)
        & jnp.all(input_valid)
        & jnp.any(recipe["output_mask"])
        & jnp.all(output_valid)
        & jnp.any(recipe["requirement_mask"])
        & jnp.all(requirement_valid)
        & (recipe["required_memories_level"] >= 1)
        & jnp.isfinite(recipe["time_seconds"])
        & (recipe["time_seconds"] >= 0.0)
    )


def _inventory_runtime_valid(inventory: CraftingInventory) -> jax.Array:
    present = inventory.item_id >= 0
    empty_resources = jnp.all(
        inventory.resource_type_id == ABSENT_ID,
        axis=1,
    )
    return (
        jnp.all(inventory.item_id >= ABSENT_ID)
        & jnp.all(inventory.resource_type_id >= ABSENT_ID)
        & jnp.all(
            (present & (inventory.quantity > 0))
            | (~present & (inventory.quantity == 0))
        )
        & jnp.all(present | empty_resources)
    )


def _context_runtime_valid(context: CraftingContext) -> jax.Array:
    return (
        (context.bench_type >= 0)
        & (context.bench_type < BENCH_TYPE_COUNT)
        & jnp.any(context.bench_id_hash != 0)
        & (context.bench_tier_level >= 0)
        & (context.memories_level >= 1)
        & (context.input_mode >= 0)
        & (context.input_mode < INPUT_MODE_COUNT)
    )


def _failure_bits(evaluation: dict[str, jax.Array]) -> jax.Array:
    request = evaluation["request_valid"]
    state = evaluation["state_valid"]
    precondition = evaluation["precondition"]
    sufficient = evaluation["sufficient"]
    bits = jnp.uint32(0)
    bits |= jnp.where(
        ~request,
        jnp.uint32(CRAFTING_FAILURE_INVALID_REQUEST),
        jnp.uint32(0),
    )
    bits |= jnp.where(
        request & ~state,
        jnp.uint32(CRAFTING_FAILURE_INVALID_STATE),
        jnp.uint32(0),
    )
    bits |= jnp.where(
        request & state & ~precondition,
        jnp.uint32(CRAFTING_FAILURE_PRECONDITION),
        jnp.uint32(0),
    )
    bits |= jnp.where(
        request & state & precondition & ~sufficient,
        jnp.uint32(CRAFTING_FAILURE_INGREDIENTS),
        jnp.uint32(0),
    )
    return bits


def _validate_left_packed(
    mask: np.ndarray,
    active: np.ndarray,
    name: str,
) -> None:
    boolean = mask.astype(np.bool_)
    counts = np.sum(boolean, axis=1)
    expected = np.arange(boolean.shape[1], dtype=np.int32)[None, :] < counts[:, None]
    if np.any(active[:, None] & (boolean != expected)):
        raise ValueError(f"{name} must be left packed for active recipes")


def _require_runtime_shapes(
    table: RecipeTable,
    inventory: CraftingInventory,
    context: CraftingContext,
) -> None:
    if not isinstance(table, RecipeTable):
        raise TypeError("table must be a RecipeTable")
    for field, expected in _TABLE_SHAPES.items():
        actual = tuple(getattr(table, field).shape)
        if actual != expected:
            raise ValueError(f"table.{field} must have shape {expected}, got {actual}")
    _require_inventory_shapes(inventory)
    _require_context_shapes(context)


def _require_inventory_shapes(inventory: CraftingInventory) -> None:
    if not isinstance(inventory, CraftingInventory):
        raise TypeError("inventory must be a CraftingInventory")
    if inventory.item_id.ndim != 1 or inventory.item_id.shape[0] < 1:
        raise ValueError("inventory.item_id must have shape [positive_slots]")
    slots = inventory.item_id.shape[0]
    expected = {
        "quantity": (slots,),
        "resource_type_id": (slots, MAX_RESOURCE_TYPES_PER_ITEM),
        "metadata_hash": (slots, METADATA_HASH_WORDS),
    }
    for field, shape in expected.items():
        actual = tuple(getattr(inventory, field).shape)
        if actual != shape:
            raise ValueError(f"inventory.{field} must have shape {shape}, got {actual}")


def _require_context_shapes(context: CraftingContext) -> None:
    if not isinstance(context, CraftingContext):
        raise TypeError("context must be a CraftingContext")
    scalar_fields = (
        "window_open",
        "bench_type",
        "bench_tier_level",
        "memories_level",
        "creative_mode",
        "input_mode",
    )
    for field in scalar_fields:
        actual = tuple(getattr(context, field).shape)
        if actual != ():
            raise ValueError(f"context.{field} must be scalar, got {actual}")
    if tuple(context.bench_id_hash.shape) != (IDENTITY_HASH_WORDS,):
        raise ValueError(
            f"context.bench_id_hash must have shape ({IDENTITY_HASH_WORDS},)"
        )
    if tuple(context.known_recipe.shape) != (RECIPE_TABLE_CAPACITY,):
        raise ValueError(
            f"context.known_recipe must have shape ({RECIPE_TABLE_CAPACITY},)"
        )


__all__ = [
    "craftable_recipe_count",
    "satisfiable_recipes",
    "step_crafting",
    "validate_context",
    "validate_inventory",
    "validate_recipe_table",
]
