"""Harvest pickup and placed-item inventory effects for block interactions."""
from __future__ import annotations


import jax.numpy as jnp

from hytalegym.jax.combat.inventory import (
    CONTAINER_HOTBAR,
    CONTAINER_TOOLS,
    CONTAINER_UTILITY,
    ItemStack,
    add_item_stack,
    remove_item_stack,
)


def _apply_harvest_pickups(inventory, layout, drops, pickup_mask):
    drop_capacity = drops.mask.shape[2]
    accepted = jnp.zeros(drops.mask.shape, dtype=jnp.bool_)
    remainder = drops.quantity
    next_inventory = inventory
    for drop in range(drop_capacity):
        active = pickup_mask & drops.mask[:, :, drop]
        transaction = add_item_stack(
            next_inventory,
            layout,
            ItemStack(
                item_id=drops.item_id[:, :, drop],
                quantity=drops.quantity[:, :, drop],
                durability=drops.durability[:, :, drop],
                max_durability=drops.max_durability[:, :, drop],
                metadata_hash=drops.metadata_hash[:, :, drop, :],
            ),
            item_max_stack=drops.item_max_stack[:, :, drop],
            container_id=drops.pickup_container_id[:, :, drop],
            all_or_nothing=False,
            slot_allowed=drops.pickup_slot_allowed[:, :, drop, :],
            request_mask=active,
        )
        next_inventory = transaction.state
        accepted = accepted.at[:, :, drop].set(
            active & transaction.accepted
        )
        remainder = remainder.at[:, :, drop].set(
            jnp.where(
                active,
                transaction.remainder_quantity,
                drops.quantity[:, :, drop],
            )
        )
    return next_inventory, accepted, remainder


def _consume_placed_item(
    inventory,
    layout,
    current_item,
    command,
    consume_mask,
):
    slot_match = (
        (layout.container_id[None, None, :] == command.held_container_id[..., None])
        & (
            layout.container_slot[None, None, :]
            == command.held_container_slot[..., None]
        )
    )
    transaction = remove_item_stack(
        inventory,
        layout,
        ItemStack(
            item_id=current_item.item_id,
            quantity=jnp.ones_like(current_item.quantity),
            durability=current_item.durability,
            max_durability=current_item.max_durability,
            metadata_hash=current_item.metadata_hash,
        ),
        container_id=command.held_container_id,
        all_or_nothing=True,
        slot_allowed=slot_match,
        request_mask=consume_mask,
    )
    return transaction.state


def _active_slot_matches(inventory, container_id, slot):
    return jnp.where(
        container_id == CONTAINER_HOTBAR,
        inventory.active_hotbar_slot == slot,
        jnp.where(
            container_id == CONTAINER_UTILITY,
            inventory.active_utility_slot == slot,
            jnp.where(
                container_id == CONTAINER_TOOLS,
                inventory.active_tools_slot == slot,
                False,
            ),
        ),
    )
