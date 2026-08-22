"""Generic server-ordered item-interaction effects over ``InventoryState``.

This module mirrors ``ModifyInventoryInteraction.firstRun``.  It deliberately
does not know about weapons or abilities: callers provide one fixed-shape
request and the held container/slot.  Each authored field commits before the
next one runs, so a later failure preserves earlier mutations just like the
server transaction sequence.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.inventory.schema.contract import (
    CONTAINER_BACKPACK,
    CONTAINER_HOTBAR,
    CONTAINER_STORAGE,
    EMPTY_ITEM_ID,
    METADATA_HASH_WORDS,
)
from hytalegym.jax.combat.inventory.schema.types import InventoryLayout, InventoryState


GAME_MODE_ANY = -1
DEFAULT_ITEM_MAX_STACK = 99


class ModifyInventoryRequest(NamedTuple):
    """One vectorized native ModifyInventory node."""

    required_game_mode: jax.Array
    remove_item_id: jax.Array
    remove_quantity: jax.Array
    remove_durability: jax.Array
    remove_max_durability: jax.Array
    remove_metadata_hash: jax.Array
    adjust_held_item_quantity: jax.Array
    add_item_id: jax.Array
    add_quantity: jax.Array
    add_durability: jax.Array
    add_max_durability: jax.Array
    add_metadata_hash: jax.Array
    add_item_max_stack: jax.Array
    broken_item_specified: jax.Array
    broken_item_id: jax.Array
    broken_item_max_durability: jax.Array
    adjust_held_item_durability: jax.Array


class ModifyInventoryResult(NamedTuple):
    state: InventoryState
    succeeded: jax.Array
    failed: jax.Array
    changed: jax.Array
    dropped_quantity: jax.Array


class _HeldItem(NamedTuple):
    present: jax.Array
    item_id: jax.Array
    quantity: jax.Array
    durability: jax.Array
    max_durability: jax.Array
    metadata_hash: jax.Array


def apply_modify_inventory(
    state: InventoryState,
    layout: InventoryLayout,
    request: ModifyInventoryRequest,
    *,
    request_mask: jax.Array,
    active_game_mode: jax.Array,
    held_container_id: jax.Array,
    held_container_slot: jax.Array,
) -> ModifyInventoryResult:
    """Apply one node in native field order without cross-field rollback."""

    shape = state.active_hotbar_slot.shape
    requested = _shape(request_mask, shape, jnp.bool_, "request_mask")
    game_mode = _shape(active_game_mode, shape, jnp.int32, "active_game_mode")
    held_container = _shape(
        held_container_id,
        shape,
        jnp.int32,
        "held_container_id",
    )
    held_slot = _shape(
        held_container_slot,
        shape,
        jnp.int32,
        "held_container_slot",
    )
    _validate_request_shapes(request, shape)
    held = _held_item_at(state, layout, held_container, held_slot)

    required_mode = jnp.asarray(request.required_game_mode, dtype=jnp.int32)
    mode_matches = (required_mode == GAME_MODE_ANY) | (required_mode == game_mode)
    # Native returns from firstRun without setting Failed when the required
    # Player/game-mode surface is absent or mismatched.  The node therefore
    # remains a successful no-op and follows its Next edge.
    active = requested & mode_matches

    remove_present = request.remove_quantity > 0
    add_present = request.add_quantity > 0
    # Native returns on a game-mode mismatch before inspecting any payload
    # field.  Malformed inactive payload therefore remains a no-op here too.
    invalid = active & (
        (required_mode < GAME_MODE_ANY)
        | (request.remove_quantity < 0)
        | (request.add_quantity < 0)
        | (remove_present & (request.remove_item_id == EMPTY_ITEM_ID))
        | (add_present & (request.add_item_id == EMPTY_ITEM_ID))
        | (add_present & (request.add_item_max_stack <= 0))
    )
    alive = active & ~invalid
    failed = invalid
    changed = jnp.zeros(shape, dtype=jnp.bool_)
    dropped = jnp.zeros(shape, dtype=jnp.int32)
    current = state

    current, remove_ok, remove_changed = _remove_combined_exact(
        current,
        layout,
        request,
        alive & remove_present,
    )
    remove_failed = alive & remove_present & ~remove_ok
    failed |= remove_failed
    alive &= ~remove_failed
    changed |= remove_changed

    (
        current,
        held_ok,
        held_changed,
        held_dropped,
        held_after_quantity,
    ) = _adjust_held_quantity(
        current,
        layout,
        request.adjust_held_item_quantity,
        alive & (request.adjust_held_item_quantity != 0),
        held_container,
        held_slot,
        held,
    )
    held = held_after_quantity
    held_failed = alive & (request.adjust_held_item_quantity != 0) & ~held_ok
    failed |= held_failed
    alive &= ~held_failed
    changed |= held_changed
    dropped += held_dropped

    current, add_changed, add_dropped = _add_combined(
        current,
        layout,
        request.add_item_id,
        request.add_quantity,
        request.add_durability,
        request.add_max_durability,
        request.add_metadata_hash,
        request.add_item_max_stack,
        alive & add_present,
    )
    changed |= add_changed
    dropped += add_dropped

    current, durability_changed = _adjust_held_durability(
        current,
        layout,
        request.adjust_held_item_durability,
        request.broken_item_specified,
        request.broken_item_id,
        request.broken_item_max_durability,
        alive & (request.adjust_held_item_durability != 0.0),
        held_container,
        held_slot,
        held,
    )
    changed |= durability_changed

    succeeded = requested & ~failed
    return ModifyInventoryResult(
        state=current,
        succeeded=succeeded,
        failed=failed,
        changed=changed,
        dropped_quantity=dropped,
    )


def initialize_held_item_durability(
    state: InventoryState,
    layout: InventoryLayout,
    max_durability: jax.Array,
) -> InventoryState:
    """Seed active-hotbar durability from an item-catalog value by item ID."""

    shape = state.active_hotbar_slot.shape
    maximum = _shape(max_durability, shape, jnp.float32, "max_durability")
    flat, valid_slot = _resolved_slot_index(
        layout,
        jnp.full(shape, CONTAINER_HOTBAR, dtype=jnp.int32),
        state.active_hotbar_slot,
    )
    present = valid_slot & (_gather_slot(state.item_id, flat) != EMPTY_ITEM_ID)
    active = present & (maximum > 0.0)
    slot_mask = jax.nn.one_hot(
        flat,
        state.item_id.shape[2],
        dtype=jnp.bool_,
    ) & active[..., None]
    return state._replace(
        durability=jnp.where(slot_mask, maximum[..., None], state.durability),
        max_durability=jnp.where(
            slot_mask,
            maximum[..., None],
            state.max_durability,
        ),
    )


def apply_held_weapon_hit_durability_loss(
    state: InventoryState,
    layout: InventoryLayout,
    weapon_item_id_table: jax.Array,
    durability_loss_on_hit_table: jax.Array,
    hit_count: jax.Array,
) -> InventoryState:
    """Apply native attacker-tool durability loss for resolved hit events.

    Hytale's ``DamageAttackerTool`` resolves the currently held stack at the
    damage boundary. The fixed item table therefore keys the loss by semantic
    item ID instead of by profile, family, or ability slot. This also keeps
    hotbar switches and material variants data-driven.
    """

    shape = state.active_hotbar_slot.shape
    item_ids = jnp.asarray(weapon_item_id_table, dtype=jnp.int32)
    losses = jnp.asarray(durability_loss_on_hit_table, dtype=jnp.float32)
    if item_ids.ndim != 1 or losses.ndim != 1:
        raise ValueError("weapon durability tables must be one-dimensional")
    if item_ids.shape != losses.shape:
        raise ValueError("weapon durability tables must have equal widths")
    count = _shape(hit_count, shape, jnp.int32, "hit_count")
    held_container = jnp.full(shape, CONTAINER_HOTBAR, dtype=jnp.int32)
    held_slot = state.active_hotbar_slot
    held = _held_item_at(
        state,
        layout,
        held_container,
        held_slot,
    )
    matches = held.item_id[..., None] == item_ids
    match_count = jnp.sum(matches.astype(jnp.int32), axis=-1)
    loss = jnp.sum(
        jnp.where(matches, losses, jnp.float32(0.0)),
        axis=-1,
        dtype=jnp.float32,
    )
    requested = (match_count == 1) & (loss > 0.0) & (count > 0)
    candidate, _ = _adjust_held_durability(
        state,
        layout,
        -loss * count.astype(jnp.float32),
        jnp.zeros(shape, dtype=jnp.bool_),
        jnp.full(shape, EMPTY_ITEM_ID, dtype=jnp.int32),
        jnp.zeros(shape, dtype=jnp.float32),
        requested,
        held_container,
        held_slot,
        held,
    )
    return candidate


def _remove_combined_exact(state, layout, request, active):
    remaining = jnp.where(active, request.remove_quantity, jnp.int32(0))
    initial = (
        state.item_id,
        state.quantity,
        state.durability,
        state.max_durability,
        state.metadata_hash,
        remaining,
    )

    def remove_container(container_id, carry):
        def remove_slot(slot, inner):
            item_id, quantity, durability, maximum, metadata, remainder = inner
            compatible = (
                (item_id[:, :, slot] == request.remove_item_id)
                & (durability[:, :, slot] == request.remove_durability)
                & (maximum[:, :, slot] == request.remove_max_durability)
                & jnp.all(
                    metadata[:, :, slot, :] == request.remove_metadata_hash,
                    axis=2,
                )
            )
            use = (
                active
                & (remainder > 0)
                & (layout.container_id[slot] == container_id)
                & compatible
            )
            amount = jnp.where(
                use,
                jnp.minimum(quantity[:, :, slot], remainder),
                jnp.int32(0),
            )
            next_quantity = quantity[:, :, slot] - amount
            emptied = use & (next_quantity == 0)
            item_id = item_id.at[:, :, slot].set(
                jnp.where(emptied, EMPTY_ITEM_ID, item_id[:, :, slot])
            )
            quantity = quantity.at[:, :, slot].set(next_quantity)
            durability = durability.at[:, :, slot].set(
                jnp.where(emptied, 0.0, durability[:, :, slot])
            )
            maximum = maximum.at[:, :, slot].set(
                jnp.where(emptied, 0.0, maximum[:, :, slot])
            )
            metadata = metadata.at[:, :, slot, :].set(
                jnp.where(
                    emptied[..., None],
                    jnp.uint32(0),
                    metadata[:, :, slot, :],
                )
            )
            return item_id, quantity, durability, maximum, metadata, remainder - amount

        return jax.lax.fori_loop(0, state.item_id.shape[2], remove_slot, carry)

    candidate = initial
    for container_id in (CONTAINER_HOTBAR, CONTAINER_STORAGE, CONTAINER_BACKPACK):
        candidate = remove_container(jnp.int32(container_id), candidate)
    item_id, quantity, durability, maximum, metadata, remainder = candidate
    accepted = ~active | (remainder == 0)
    commit = active & accepted
    changed = commit & (request.remove_quantity > 0)
    return (
        state._replace(
            item_id=jnp.where(commit[..., None], item_id, state.item_id),
            quantity=jnp.where(commit[..., None], quantity, state.quantity),
            durability=jnp.where(commit[..., None], durability, state.durability),
            max_durability=jnp.where(
                commit[..., None],
                maximum,
                state.max_durability,
            ),
            metadata_hash=jnp.where(
                commit[..., None, None],
                metadata,
                state.metadata_hash,
            ),
        ),
        accepted,
        changed,
    )


def _adjust_held_quantity(
    state,
    layout,
    delta,
    active,
    held_container,
    held_slot,
    held,
):
    flat, valid_slot = _resolved_slot_index(
        layout,
        held_container,
        held_slot,
    )
    current_item_id = _gather_slot(state.item_id, flat)
    current_quantity = _gather_slot(state.quantity, flat)
    current_durability = _gather_slot(state.durability, flat)
    current_maximum = _gather_slot(state.max_durability, flat)
    current_metadata = _gather_slot(state.metadata_hash, flat)
    compatible = (
        valid_slot
        & (current_item_id == held.item_id)
        & (current_durability == held.durability)
        & (current_maximum == held.max_durability)
        & jnp.all(current_metadata == held.metadata_hash, axis=2)
    )
    removing = active & (delta < 0) & held.present
    remove_amount = -delta
    remove_ok = ~removing | (compatible & (current_quantity >= remove_amount))
    commit_remove = removing & remove_ok
    next_quantity = current_quantity - jnp.where(commit_remove, remove_amount, 0)
    emptied = commit_remove & (next_quantity == 0)
    slot_mask = jax.nn.one_hot(
        flat,
        state.item_id.shape[2],
        dtype=jnp.bool_,
    )
    commit_slot = slot_mask & commit_remove[..., None]
    emptied_slot = slot_mask & emptied[..., None]
    current = state._replace(
        item_id=jnp.where(emptied_slot, EMPTY_ITEM_ID, state.item_id),
        quantity=jnp.where(commit_slot, next_quantity[..., None], state.quantity),
        durability=jnp.where(emptied_slot, 0.0, state.durability),
        max_durability=jnp.where(emptied_slot, 0.0, state.max_durability),
        metadata_hash=jnp.where(
            emptied_slot[..., None],
            jnp.uint32(0),
            state.metadata_hash,
        ),
    )

    held_after = _HeldItem(
        present=jnp.where(commit_remove, ~emptied, held.present),
        item_id=jnp.where(emptied, EMPTY_ITEM_ID, held.item_id),
        quantity=jnp.where(commit_remove, next_quantity, held.quantity),
        durability=jnp.where(emptied, 0.0, held.durability),
        max_durability=jnp.where(emptied, 0.0, held.max_durability),
        metadata_hash=jnp.where(
            emptied[..., None],
            jnp.uint32(0),
            held.metadata_hash,
        ),
    )

    adding = active & (delta > 0) & held.present
    current, add_changed, dropped = _add_combined(
        current,
        layout,
        held.item_id,
        jnp.where(adding, delta, 0),
        held.durability,
        held.max_durability,
        held.metadata_hash,
        jnp.full(delta.shape, DEFAULT_ITEM_MAX_STACK, dtype=jnp.int32),
        adding,
    )
    # The native node guards the whole quantity branch with ``heldItem !=
    # null``.  No held stack is therefore a successful no-op; only an actual
    # insufficient removal fails the interaction.
    succeeded = ~(removing & ~remove_ok)
    changed = commit_remove | add_changed
    return current, succeeded, changed, dropped, held_after


def _add_combined(
    state,
    layout,
    item_id,
    quantity,
    durability,
    maximum,
    metadata,
    item_max_stack,
    active,
):
    remaining = jnp.where(active, quantity, jnp.int32(0))
    current = state
    # CombinedItemContainer performs one global existing-stack pass before
    # one global empty-slot pass.  It must not fill an earlier hotbar empty
    # while a compatible storage stack still has room.
    for existing_only in (True, False):
        for container_id in (
            CONTAINER_HOTBAR,
            CONTAINER_STORAGE,
            CONTAINER_BACKPACK,
        ):
            current, remaining = _add_to_container_phase(
                current,
                layout,
                item_id,
                remaining,
                durability,
                maximum,
                metadata,
                item_max_stack,
                active,
                container_id,
                existing_only=existing_only,
            )
    changed = active & (remaining != quantity)
    return current, changed, jnp.where(active, remaining, 0)


def _add_to_container_phase(
    state,
    layout,
    stack_item_id,
    stack_quantity,
    stack_durability,
    stack_maximum,
    stack_metadata,
    item_max_stack,
    active,
    container_id,
    *,
    existing_only,
):
    initial = (
        state.item_id,
        state.quantity,
        state.durability,
        state.max_durability,
        state.metadata_hash,
        stack_quantity,
    )

    def fill_existing(slot, carry):
        item_id, quantity, durability, maximum, metadata, remaining = carry
        compatible = (
            (item_id[:, :, slot] == stack_item_id)
            & (durability[:, :, slot] == stack_durability)
            & (maximum[:, :, slot] == stack_maximum)
            & jnp.all(metadata[:, :, slot, :] == stack_metadata, axis=2)
        )
        room = jnp.maximum(0, item_max_stack - quantity[:, :, slot])
        use = (
            active
            & (remaining > 0)
            & (layout.container_id[slot] == container_id)
            & compatible
            & (room > 0)
        )
        amount = jnp.where(use, jnp.minimum(room, remaining), 0)
        quantity = quantity.at[:, :, slot].add(amount)
        return item_id, quantity, durability, maximum, metadata, remaining - amount

    if existing_only:
        item_id, quantity, durability, maximum, metadata, remaining = (
            jax.lax.fori_loop(
                0,
                state.item_id.shape[2],
                fill_existing,
                initial,
            )
        )
        return (
            state._replace(
                item_id=item_id,
                quantity=quantity,
                durability=durability,
                max_durability=maximum,
                metadata_hash=metadata,
            ),
            remaining,
        )

    def fill_empty(slot, carry):
        item_id, quantity, durability, maximum, metadata, remaining = carry
        use = (
            active
            & (remaining > 0)
            & (layout.container_id[slot] == container_id)
            & (item_id[:, :, slot] == EMPTY_ITEM_ID)
        )
        amount = jnp.where(use, jnp.minimum(item_max_stack, remaining), 0)
        item_id = item_id.at[:, :, slot].set(
            jnp.where(use, stack_item_id, item_id[:, :, slot])
        )
        quantity = quantity.at[:, :, slot].set(
            jnp.where(use, amount, quantity[:, :, slot])
        )
        durability = durability.at[:, :, slot].set(
            jnp.where(use, stack_durability, durability[:, :, slot])
        )
        maximum = maximum.at[:, :, slot].set(
            jnp.where(use, stack_maximum, maximum[:, :, slot])
        )
        metadata = metadata.at[:, :, slot, :].set(
            jnp.where(use[..., None], stack_metadata, metadata[:, :, slot, :])
        )
        return item_id, quantity, durability, maximum, metadata, remaining - amount

    item_id, quantity, durability, maximum, metadata, remaining = jax.lax.fori_loop(
        0,
        state.item_id.shape[2],
        fill_empty,
        initial,
    )
    return (
        state._replace(
            item_id=item_id,
            quantity=quantity,
            durability=durability,
            max_durability=maximum,
            metadata_hash=metadata,
        ),
        remaining,
    )


def _adjust_held_durability(
    state,
    layout,
    delta,
    broken_specified,
    broken_item_id,
    broken_item_maximum,
    active,
    held_container,
    held_slot,
    held,
):
    flat, valid_slot = _resolved_slot_index(
        layout,
        held_container,
        held_slot,
    )
    applies = active & valid_slot & held.present
    next_durability = jnp.clip(
        held.durability + delta,
        0.0,
        held.max_durability,
    )
    broken = (
        applies
        & (held.max_durability > 0.0)
        & (next_durability == 0.0)
    )
    # BrokenItem conversion is keyed by the resulting broken state.  Only
    # player notification uses the fresh just-broke edge.
    transform = broken & broken_specified
    clear = transform & (broken_item_id == EMPTY_ITEM_ID)
    replace = transform & ~clear & (broken_item_id != held.item_id)
    slot_mask = jax.nn.one_hot(
        flat,
        state.item_id.shape[2],
        dtype=jnp.bool_,
    )
    applies_slot = slot_mask & applies[..., None]
    resolved_item = jnp.where(replace, broken_item_id, held.item_id)
    resolved_item = jnp.where(clear, EMPTY_ITEM_ID, resolved_item)
    resolved_quantity = jnp.where(replace, 1, held.quantity)
    resolved_quantity = jnp.where(clear, 0, resolved_quantity)
    resolved_maximum = jnp.where(
        replace,
        broken_item_maximum,
        held.max_durability,
    )
    resolved_maximum = jnp.where(clear, 0.0, resolved_maximum)
    resolved_durability = jnp.where(
        replace,
        broken_item_maximum,
        next_durability,
    )
    resolved_durability = jnp.where(clear, 0.0, resolved_durability)
    resolved_metadata = jnp.where(
        (clear | replace)[..., None],
        jnp.uint32(0),
        held.metadata_hash,
    )
    return (
        state._replace(
            item_id=jnp.where(
                applies_slot,
                resolved_item[..., None],
                state.item_id,
            ),
            quantity=jnp.where(
                applies_slot,
                resolved_quantity[..., None],
                state.quantity,
            ),
            durability=jnp.where(
                applies_slot,
                resolved_durability[..., None],
                state.durability,
            ),
            max_durability=jnp.where(
                applies_slot,
                resolved_maximum[..., None],
                state.max_durability,
            ),
            metadata_hash=jnp.where(
                applies_slot[..., None],
                resolved_metadata[..., None, :],
                state.metadata_hash,
            ),
        ),
        applies,
    )


def _held_item_at(state, layout, container_id, container_slot):
    flat, valid_slot = _resolved_slot_index(
        layout,
        container_id,
        container_slot,
    )
    item_id = _gather_slot(state.item_id, flat)
    present = valid_slot & (item_id != EMPTY_ITEM_ID)
    return _HeldItem(
        present=present,
        item_id=jnp.where(present, item_id, EMPTY_ITEM_ID),
        quantity=jnp.where(present, _gather_slot(state.quantity, flat), 0),
        durability=jnp.where(
            present,
            _gather_slot(state.durability, flat),
            0.0,
        ),
        max_durability=jnp.where(
            present,
            _gather_slot(state.max_durability, flat),
            0.0,
        ),
        metadata_hash=jnp.where(
            present[..., None],
            _gather_slot(state.metadata_hash, flat),
            jnp.uint32(0),
        ),
    )


def _resolved_slot_index(layout, container_id, container_slot):
    matches = (
        layout.container_id[None, None, :] == container_id[..., None]
    ) & (layout.container_slot[None, None, :] == container_slot[..., None])
    return (
        jnp.argmax(matches.astype(jnp.int32), axis=2),
        jnp.any(matches, axis=2),
    )


def _gather_slot(value, flat):
    if value.ndim == 3:
        return jnp.take_along_axis(value, flat[..., None], axis=2)[..., 0]
    return jnp.take_along_axis(value, flat[..., None, None], axis=2)[..., 0, :]


def _shape(value, shape, dtype, name):
    array = jnp.asarray(value, dtype=dtype)
    if array.ndim == 0:
        array = jnp.broadcast_to(array, shape)
    if array.shape != shape:
        raise ValueError(f"{name} must be scalar or {shape}")
    return array


def _validate_request_shapes(request, shape):
    for name in (
        "required_game_mode",
        "remove_item_id",
        "remove_quantity",
        "remove_durability",
        "remove_max_durability",
        "adjust_held_item_quantity",
        "add_item_id",
        "add_quantity",
        "add_durability",
        "add_max_durability",
        "add_item_max_stack",
        "broken_item_specified",
        "broken_item_id",
        "broken_item_max_durability",
        "adjust_held_item_durability",
    ):
        if getattr(request, name).shape != shape:
            raise ValueError(f"request.{name} must have shape {shape}")
    metadata_shape = shape + (METADATA_HASH_WORDS,)
    for name in ("remove_metadata_hash", "add_metadata_hash"):
        if getattr(request, name).shape != metadata_shape:
            raise ValueError(f"request.{name} must have shape {metadata_shape}")


__all__ = [
    "DEFAULT_ITEM_MAX_STACK",
    "GAME_MODE_ANY",
    "ModifyInventoryRequest",
    "ModifyInventoryResult",
    "apply_held_weapon_hit_durability_loss",
    "apply_modify_inventory",
    "initialize_held_item_durability",
]
