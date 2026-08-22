"""Pure-JAX fixed-shape inventory construction, queries, and transactions."""

from __future__ import annotations

import operator

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat.inventory.schema.contract import (
    ABSENT_MATERIAL_ID,
    CONTAINER_COUNT,
    CONTAINER_HOTBAR,
    DEFAULT_CONTAINER_CAPACITIES,
    EMPTY_ITEM_ID,
    HOTBAR_CAPACITY,
    INVENTORY_FAILURE_INVALID_REQUEST,
    INVENTORY_FAILURE_INVALID_STATE,
    METADATA_HASH_WORDS,
    TOOLS_CAPACITY,
    UTILITY_CAPACITY,
)
from hytalegym.jax.combat.inventory.schema.types import (
    InventoryLayout,
    InventoryState,
    InventoryTransaction,
    ItemStack,
    MaterialQuantity,
    ResourceQuantity,
)


def default_inventory_layout(
    *,
    backpack_capacity: int = 0,
) -> InventoryLayout:
    """Flatten native containers while retaining container and local-slot IDs."""

    backpack = _nonnegative_capacity(backpack_capacity, "backpack_capacity")
    capacities = np.asarray(
        (*DEFAULT_CONTAINER_CAPACITIES[:-1], backpack),
        dtype=np.int32,
    )
    offsets = np.concatenate(
        (
            np.asarray([0], dtype=np.int32),
            np.cumsum(capacities[:-1], dtype=np.int32),
        )
    )
    container_id = np.repeat(
        np.arange(CONTAINER_COUNT, dtype=np.int32),
        capacities,
    )
    container_slot = np.concatenate(
        tuple(np.arange(capacity, dtype=np.int32) for capacity in capacities)
    )
    return InventoryLayout(
        capacities=jnp.asarray(capacities),
        offsets=jnp.asarray(offsets),
        container_id=jnp.asarray(container_id),
        container_slot=jnp.asarray(container_slot),
    )


def empty_inventory_state(
    batch_size: int,
    *,
    entity_count: int,
    layout: InventoryLayout | None = None,
) -> InventoryState:
    """Return the canonical empty inventory."""

    batch = _positive_capacity(batch_size, "batch_size")
    entities = _positive_capacity(entity_count, "entity_count")
    resolved = default_inventory_layout() if layout is None else layout
    slot_count = _validate_layout(resolved)
    slots = (batch, entities, slot_count)
    entity = (batch, entities)
    return InventoryState(
        item_id=jnp.full(slots, EMPTY_ITEM_ID, dtype=jnp.int32),
        quantity=jnp.zeros(slots, dtype=jnp.int32),
        durability=jnp.zeros(slots, dtype=jnp.float32),
        max_durability=jnp.zeros(slots, dtype=jnp.float32),
        metadata_hash=jnp.zeros(
            slots + (METADATA_HASH_WORDS,),
            dtype=jnp.uint32,
        ),
        active_hotbar_slot=jnp.zeros(entity, dtype=jnp.int32),
        active_utility_slot=jnp.full(entity, -1, dtype=jnp.int32),
        active_tools_slot=jnp.full(entity, -1, dtype=jnp.int32),
        failure_bits=jnp.zeros((batch,), dtype=jnp.uint32),
    )


def inventory_from_loadout(
    loadout,
    layout: InventoryLayout | None = None,
) -> InventoryState:
    """Seed equipped loadout items into active hotbar slot zero.

    Loadout programs do not yet carry item-catalog durability, so seeded
    equipment uses ItemStack's unbreakable ``0/0`` representation. No
    weapon-name branch is involved.
    """

    resolved = default_inventory_layout() if layout is None else layout
    batch, entities = loadout.weapon_id.shape
    state = empty_inventory_state(
        batch,
        entity_count=entities,
        layout=resolved,
    )
    hotbar_zero = (
        (resolved.container_id == jnp.int32(CONTAINER_HOTBAR))
        & (resolved.container_slot == jnp.int32(0))
    )
    index = jnp.argmax(hotbar_zero.astype(jnp.int32))
    equipped = jnp.asarray(loadout.equipped, dtype=jnp.bool_)
    valid = equipped & (loadout.weapon_id >= 0)
    item_id = state.item_id.at[:, :, index].set(
        jnp.where(valid, loadout.weapon_id, jnp.int32(EMPTY_ITEM_ID))
    )
    quantity = state.quantity.at[:, :, index].set(
        valid.astype(jnp.int32)
    )
    invalid = jnp.any(equipped & ~valid, axis=1)
    return state._replace(
        item_id=item_id,
        quantity=quantity,
        failure_bits=jnp.where(
            invalid,
            jnp.uint32(INVENTORY_FAILURE_INVALID_REQUEST),
            state.failure_bits,
        ),
    )


def item_stack_at(
    state: InventoryState,
    layout: InventoryLayout,
    container_id,
    container_slot,
) -> ItemStack:
    """Read one container-local slot per batch entity, failing to empty."""

    batch, entities, slot_count = _state_shape(state, layout)
    shape = (batch, entities)
    container = _broadcast_int32(container_id, shape)
    local_slot = _broadcast_int32(container_slot, shape)
    valid_container = (container >= 0) & (container < CONTAINER_COUNT)
    safe_container = jnp.clip(container, 0, CONTAINER_COUNT - 1)
    capacity = jnp.take(layout.capacities, safe_container)
    valid = (
        valid_container
        & (local_slot >= 0)
        & (local_slot < capacity)
    )
    match = (
        (layout.container_id[None, None, :] == container[..., None])
        & (layout.container_slot[None, None, :] == local_slot[..., None])
        & valid[..., None]
    )
    index = jnp.argmax(match.astype(jnp.int32), axis=2)

    def gather(values):
        return jnp.take_along_axis(values, index[..., None], axis=2)[..., 0]

    metadata = jnp.take_along_axis(
        state.metadata_hash,
        jnp.broadcast_to(
            index[..., None, None],
            shape + (1, METADATA_HASH_WORDS),
        ),
        axis=2,
    )[..., 0, :]
    return ItemStack(
        item_id=jnp.where(valid, gather(state.item_id), EMPTY_ITEM_ID),
        quantity=jnp.where(valid, gather(state.quantity), 0),
        durability=jnp.where(valid, gather(state.durability), 0.0),
        max_durability=jnp.where(
            valid,
            gather(state.max_durability),
            0.0,
        ),
        metadata_hash=jnp.where(valid[..., None], metadata, 0),
    )


def add_item_stack(
    state: InventoryState,
    layout: InventoryLayout,
    stack: ItemStack,
    item_max_stack,
    container_id,
    all_or_nothing=False,
    full_stacks=False,
    slot_allowed=None,
    request_mask=None,
) -> InventoryTransaction:
    """Insert active stacks in native order with per-entity atomic rollback."""

    batch, entities, slot_count = _state_shape(state, layout)
    shape = (batch, entities)
    _validate_stack_shape(stack, shape)
    maximum = _broadcast_int32(item_max_stack, shape)
    container = _broadcast_int32(container_id, shape)
    atomic = _broadcast_bool(all_or_nothing, shape)
    skip_existing = _broadcast_bool(full_stacks, shape)
    requested = (
        jnp.ones(shape, dtype=jnp.bool_)
        if request_mask is None
        else _broadcast_bool(request_mask, shape)
    )
    allowed = (
        jnp.ones((batch, entities, slot_count), dtype=jnp.bool_)
        if slot_allowed is None
        else jnp.asarray(slot_allowed, dtype=jnp.bool_)
    )
    if allowed.shape != (batch, entities, slot_count):
        raise ValueError(
            "slot_allowed must match inventory item_id shape"
        )

    state_failure = inventory_state_failure_bits(state, layout)
    valid_container = (container >= 0) & (container < CONTAINER_COUNT)
    safe_container = jnp.clip(container, 0, CONTAINER_COUNT - 1)
    container_capacity = jnp.take(layout.capacities, safe_container)
    active_request_valid = (
        item_stack_valid(stack)
        & (maximum > 0)
        & valid_container
        & (container_capacity > 0)
    )
    request_valid = ~requested | active_request_valid
    batch_valid = (
        (state_failure == jnp.uint32(0))
        & jnp.all(request_valid, axis=1)
    )
    valid = requested & active_request_valid & batch_valid[:, None]

    initial = (
        state.item_id,
        state.quantity,
        state.durability,
        state.max_durability,
        state.metadata_hash,
        stack.quantity,
    )

    def fill_existing(slot, carry):
        item_id, quantity, durability, max_durability, metadata, remaining = carry
        same_metadata = jnp.all(
            metadata[:, :, slot, :] == stack.metadata_hash,
            axis=2,
        )
        compatible = (
            (item_id[:, :, slot] == stack.item_id)
            & (durability[:, :, slot] == stack.durability)
            & (max_durability[:, :, slot] == stack.max_durability)
            & same_metadata
        )
        room = jnp.maximum(0, maximum - quantity[:, :, slot])
        use = (
            valid
            & ~skip_existing
            & (remaining > 0)
            & allowed[:, :, slot]
            & (layout.container_id[slot] == container)
            & compatible
            & (room > 0)
        )
        adjustment = jnp.where(
            use,
            jnp.minimum(room, remaining),
            jnp.int32(0),
        )
        quantity = quantity.at[:, :, slot].add(adjustment)
        return (
            item_id,
            quantity,
            durability,
            max_durability,
            metadata,
            remaining - adjustment,
        )

    candidate = jax.lax.fori_loop(0, slot_count, fill_existing, initial)

    def fill_empty(slot, carry):
        item_id, quantity, durability, max_durability, metadata, remaining = carry
        use = (
            valid
            & (remaining > 0)
            & allowed[:, :, slot]
            & (layout.container_id[slot] == container)
            & (item_id[:, :, slot] == EMPTY_ITEM_ID)
        )
        adjustment = jnp.where(
            use,
            jnp.minimum(maximum, remaining),
            jnp.int32(0),
        )
        item_id = item_id.at[:, :, slot].set(
            jnp.where(use, stack.item_id, item_id[:, :, slot])
        )
        quantity = quantity.at[:, :, slot].set(
            jnp.where(use, adjustment, quantity[:, :, slot])
        )
        durability = durability.at[:, :, slot].set(
            jnp.where(use, stack.durability, durability[:, :, slot])
        )
        max_durability = max_durability.at[:, :, slot].set(
            jnp.where(
                use,
                stack.max_durability,
                max_durability[:, :, slot],
            )
        )
        metadata = metadata.at[:, :, slot, :].set(
            jnp.where(
                use[..., None],
                stack.metadata_hash,
                metadata[:, :, slot, :],
            )
        )
        return (
            item_id,
            quantity,
            durability,
            max_durability,
            metadata,
            remaining - adjustment,
        )

    candidate = jax.lax.fori_loop(0, slot_count, fill_empty, candidate)
    (
        candidate_item_id,
        candidate_quantity,
        candidate_durability,
        candidate_max_durability,
        candidate_metadata,
        candidate_remainder,
    ) = candidate
    accepted = valid & (~atomic | (candidate_remainder == 0))
    changed = accepted & (candidate_remainder != stack.quantity)
    slot_commit = accepted[..., None]
    metadata_commit = slot_commit[..., None]
    request_invalid = jnp.any(requested & ~active_request_valid, axis=1)
    failure_bits = state_failure | jnp.where(
        request_invalid,
        jnp.uint32(INVENTORY_FAILURE_INVALID_REQUEST),
        jnp.uint32(0),
    )
    next_state = state._replace(
        item_id=jnp.where(slot_commit, candidate_item_id, state.item_id),
        quantity=jnp.where(slot_commit, candidate_quantity, state.quantity),
        durability=jnp.where(
            slot_commit,
            candidate_durability,
            state.durability,
        ),
        max_durability=jnp.where(
            slot_commit,
            candidate_max_durability,
            state.max_durability,
        ),
        metadata_hash=jnp.where(
            metadata_commit,
            candidate_metadata,
            state.metadata_hash,
        ),
        failure_bits=failure_bits,
    )
    return InventoryTransaction(
        state=next_state,
        remainder_quantity=jnp.where(
            accepted,
            candidate_remainder,
            stack.quantity,
        ),
        accepted=accepted,
        changed=changed,
    )


def remove_item_stack(
    state: InventoryState,
    layout: InventoryLayout,
    stack: ItemStack,
    container_id,
    all_or_nothing=False,
    slot_allowed=None,
    request_mask=None,
) -> InventoryTransaction:
    """Remove active requests in ascending container-local slot order."""

    batch, entities, slot_count = _state_shape(state, layout)
    shape = (batch, entities)
    _validate_stack_shape(stack, shape)
    container = _broadcast_int32(container_id, shape)
    atomic = _broadcast_bool(all_or_nothing, shape)
    requested = (
        jnp.ones(shape, dtype=jnp.bool_)
        if request_mask is None
        else _broadcast_bool(request_mask, shape)
    )
    allowed = (
        jnp.ones((batch, entities, slot_count), dtype=jnp.bool_)
        if slot_allowed is None
        else jnp.asarray(slot_allowed, dtype=jnp.bool_)
    )
    if allowed.shape != (batch, entities, slot_count):
        raise ValueError("slot_allowed must match inventory item_id shape")

    state_failure = inventory_state_failure_bits(state, layout)
    valid_container = (container >= 0) & (container < CONTAINER_COUNT)
    safe_container = jnp.clip(container, 0, CONTAINER_COUNT - 1)
    container_capacity = jnp.take(layout.capacities, safe_container)
    active_request_valid = (
        item_stack_valid(stack)
        & valid_container
        & (container_capacity > 0)
    )
    request_valid = ~requested | active_request_valid
    batch_valid = (
        (state_failure == jnp.uint32(0))
        & jnp.all(request_valid, axis=1)
    )
    valid = requested & active_request_valid & batch_valid[:, None]
    initial = (
        state.item_id,
        state.quantity,
        state.durability,
        state.max_durability,
        state.metadata_hash,
        stack.quantity,
    )

    def remove_from_slot(slot, carry):
        item_id, quantity, durability, max_durability, metadata, remaining = carry
        compatible = (
            (item_id[:, :, slot] == stack.item_id)
            & (durability[:, :, slot] == stack.durability)
            & (max_durability[:, :, slot] == stack.max_durability)
            & jnp.all(
                metadata[:, :, slot, :] == stack.metadata_hash,
                axis=2,
            )
        )
        use = (
            valid
            & (remaining > 0)
            & allowed[:, :, slot]
            & (layout.container_id[slot] == container)
            & compatible
        )
        adjustment = jnp.where(
            use,
            jnp.minimum(quantity[:, :, slot], remaining),
            jnp.int32(0),
        )
        next_quantity = quantity[:, :, slot] - adjustment
        emptied = use & (next_quantity == 0)
        item_id = item_id.at[:, :, slot].set(
            jnp.where(emptied, EMPTY_ITEM_ID, item_id[:, :, slot])
        )
        quantity = quantity.at[:, :, slot].set(next_quantity)
        durability = durability.at[:, :, slot].set(
            jnp.where(emptied, 0.0, durability[:, :, slot])
        )
        max_durability = max_durability.at[:, :, slot].set(
            jnp.where(emptied, 0.0, max_durability[:, :, slot])
        )
        metadata = metadata.at[:, :, slot, :].set(
            jnp.where(
                emptied[..., None],
                jnp.uint32(0),
                metadata[:, :, slot, :],
            )
        )
        return (
            item_id,
            quantity,
            durability,
            max_durability,
            metadata,
            remaining - adjustment,
        )

    candidate = jax.lax.fori_loop(
        0,
        slot_count,
        remove_from_slot,
        initial,
    )
    (
        candidate_item_id,
        candidate_quantity,
        candidate_durability,
        candidate_max_durability,
        candidate_metadata,
        candidate_remainder,
    ) = candidate
    accepted = valid & (~atomic | (candidate_remainder == 0))
    changed = accepted & (candidate_remainder != stack.quantity)
    slot_commit = accepted[..., None]
    request_invalid = jnp.any(requested & ~active_request_valid, axis=1)
    failure_bits = state_failure | jnp.where(
        request_invalid,
        jnp.uint32(INVENTORY_FAILURE_INVALID_REQUEST),
        jnp.uint32(0),
    )
    return InventoryTransaction(
        state=state._replace(
            item_id=jnp.where(
                slot_commit,
                candidate_item_id,
                state.item_id,
            ),
            quantity=jnp.where(
                slot_commit,
                candidate_quantity,
                state.quantity,
            ),
            durability=jnp.where(
                slot_commit,
                candidate_durability,
                state.durability,
            ),
            max_durability=jnp.where(
                slot_commit,
                candidate_max_durability,
                state.max_durability,
            ),
            metadata_hash=jnp.where(
                slot_commit[..., None],
                candidate_metadata,
                state.metadata_hash,
            ),
            failure_bits=failure_bits,
        ),
        remainder_quantity=jnp.where(
            accepted,
            candidate_remainder,
            stack.quantity,
        ),
        accepted=accepted,
        changed=changed,
    )


def set_active_hotbar_slot(
    state: InventoryState,
    slot,
) -> InventoryState:
    """Select one hotbar slot per entity; invalid rows fail atomically."""

    shape = state.active_hotbar_slot.shape
    requested = _broadcast_int32(slot, shape)
    valid = (requested >= 0) & (requested < HOTBAR_CAPACITY)
    valid_batch = jnp.all(valid, axis=1)
    failure_bits = state.failure_bits | jnp.where(
        valid_batch,
        jnp.uint32(0),
        jnp.uint32(INVENTORY_FAILURE_INVALID_REQUEST),
    )
    return state._replace(
        active_hotbar_slot=jnp.where(
            valid_batch[:, None],
            requested,
            state.active_hotbar_slot,
        ),
        failure_bits=failure_bits,
    )


def item_stack_valid(stack: ItemStack):
    """Return per-stack validity under the native ItemStack codec."""

    return (
        (stack.item_id >= 0)
        & (stack.quantity > 0)
        & jnp.isfinite(stack.durability)
        & (stack.durability >= 0.0)
        & jnp.isfinite(stack.max_durability)
        & (stack.max_durability >= 0.0)
    )


def material_quantity_valid(value: MaterialQuantity):
    """Return validity for MaterialQuantity's at-least-one selector rule."""

    selector = (
        (value.item_id != ABSENT_MATERIAL_ID)
        | (value.resource_type_id != ABSENT_MATERIAL_ID)
        | (value.tag_id != ABSENT_MATERIAL_ID)
    )
    return selector & (value.quantity > 0)


def resource_quantity_valid(value: ResourceQuantity):
    """Return validity for ResourceQuantity."""

    return (value.resource_id != ABSENT_MATERIAL_ID) & (value.quantity > 0)


def inventory_state_failure_bits(
    state: InventoryState,
    layout: InventoryLayout,
):
    """Return sticky plus structural/device-value failures per environment."""

    batch, entities, _ = _state_shape(state, layout)
    empty = state.item_id == EMPTY_ITEM_ID
    metadata_empty = jnp.all(
        state.metadata_hash == jnp.uint32(0),
        axis=3,
    )
    canonical_empty = (
        (state.quantity == 0)
        & (state.durability == 0.0)
        & (state.max_durability == 0.0)
        & metadata_empty
    )
    valid_present = (
        (state.item_id >= 0)
        & (state.quantity > 0)
        & jnp.isfinite(state.durability)
        & (state.durability >= 0.0)
        & jnp.isfinite(state.max_durability)
        & (state.max_durability >= 0.0)
    )
    active_valid = (
        (state.active_hotbar_slot >= 0)
        & (state.active_hotbar_slot < HOTBAR_CAPACITY)
        & (state.active_utility_slot >= -1)
        & (state.active_utility_slot < UTILITY_CAPACITY)
        & (state.active_tools_slot >= -1)
        & (state.active_tools_slot < TOOLS_CAPACITY)
    )
    invalid = (
        jnp.any(~jnp.where(empty, canonical_empty, valid_present), axis=(1, 2))
        | jnp.any(~active_valid, axis=1)
    )
    return state.failure_bits | jnp.where(
        invalid,
        jnp.uint32(INVENTORY_FAILURE_INVALID_STATE),
        jnp.uint32(0),
    )


def _state_shape(
    state: InventoryState,
    layout: InventoryLayout,
) -> tuple[int, int, int]:
    slot_count = _validate_layout(layout)
    if state.item_id.ndim != 3:
        raise ValueError("inventory item_id must have rank 3")
    batch, entities, state_slots = state.item_id.shape
    if state_slots != slot_count:
        raise ValueError("inventory slot axis does not match layout")
    shape = (batch, entities, slot_count)
    expected = {
        "quantity": shape,
        "durability": shape,
        "max_durability": shape,
        "metadata_hash": shape + (METADATA_HASH_WORDS,),
        "active_hotbar_slot": (batch, entities),
        "active_utility_slot": (batch, entities),
        "active_tools_slot": (batch, entities),
        "failure_bits": (batch,),
    }
    for name, wanted in expected.items():
        if getattr(state, name).shape != wanted:
            raise ValueError(f"inventory {name} must have shape {wanted}")
    return batch, entities, slot_count


def _validate_stack_shape(stack: ItemStack, shape: tuple[int, int]) -> None:
    for name in ("item_id", "quantity", "durability", "max_durability"):
        if getattr(stack, name).shape != shape:
            raise ValueError(f"stack {name} must have shape {shape}")
    expected_metadata = shape + (METADATA_HASH_WORDS,)
    if stack.metadata_hash.shape != expected_metadata:
        raise ValueError(
            f"stack metadata_hash must have shape {expected_metadata}"
        )


def _validate_layout(layout: InventoryLayout) -> int:
    if layout.capacities.shape != (CONTAINER_COUNT,):
        raise ValueError("inventory layout capacities must have six entries")
    if layout.offsets.shape != (CONTAINER_COUNT,):
        raise ValueError("inventory layout offsets must have six entries")
    if layout.container_id.ndim != 1:
        raise ValueError("inventory layout container_id must have rank 1")
    if layout.container_slot.shape != layout.container_id.shape:
        raise ValueError("inventory layout slot arrays must match")
    return layout.container_id.shape[0]


def _broadcast_int32(value, shape: tuple[int, int]):
    return jnp.broadcast_to(jnp.asarray(value, dtype=jnp.int32), shape)


def _broadcast_bool(value, shape: tuple[int, int]):
    return jnp.broadcast_to(jnp.asarray(value, dtype=jnp.bool_), shape)


def _positive_capacity(value: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if isinstance(value, bool) or result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_capacity(value: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if isinstance(value, bool) or result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


__all__ = [
    "add_item_stack",
    "default_inventory_layout",
    "empty_inventory_state",
    "inventory_from_loadout",
    "inventory_state_failure_bits",
    "item_stack_at",
    "item_stack_valid",
    "material_quantity_valid",
    "remove_item_stack",
    "resource_quantity_valid",
    "set_active_hotbar_slot",
]
