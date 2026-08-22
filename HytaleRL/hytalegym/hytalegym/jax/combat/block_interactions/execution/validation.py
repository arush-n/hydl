"""Scalar coercion and shape/state validation for block interaction runtime."""
from __future__ import annotations

import operator

import jax.numpy as jnp

from hytalegym.jax.combat.block_interactions.schema.contract import (
    BLOCK_INTERACTION_CANCELLED,
    BLOCK_INTERACTION_IDLE,
    INTERACTION_MOVEMENT_ALL_MASK,
)
from hytalegym.jax.combat.inventory import (
    METADATA_HASH_WORDS,
)


def _integer(value: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return result


def _positive(value: int, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative(value: int, name: str) -> int:
    result = _integer(value, name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _shape(batch_size: int, entity_count: int) -> tuple[int, int]:
    return (
        _positive(batch_size, "batch_size"),
        _positive(entity_count, "entity_count"),
    )


def _actor_index(value: int, entity_count: int) -> int:
    actor = _integer(value, "actor_index")
    if actor < 0 or actor >= entity_count:
        raise ValueError("actor_index is outside the interaction entity axis")
    return actor


def _state_valid(state):
    phase_valid = (
        (state.phase >= BLOCK_INTERACTION_IDLE)
        & (state.phase <= BLOCK_INTERACTION_CANCELLED)
    )
    numeric_valid = (
        jnp.isfinite(state.elapsed_seconds)
        & (state.elapsed_seconds >= 0.0)
        & jnp.isfinite(state.duration_seconds)
        & (state.duration_seconds >= 0.0)
        & (state.elapsed_seconds <= state.duration_seconds)
        & jnp.isfinite(state.start_delay_seconds)
        & (state.start_delay_seconds >= 0.0)
        & jnp.isfinite(state.horizontal_speed_multiplier)
        & (state.horizontal_speed_multiplier >= 0.0)
        & (state.movement_lock_mask >= 0)
        & (state.movement_lock_mask <= INTERACTION_MOVEMENT_ALL_MASK)
        & (
            ~state.movement_disable_all
            | (
                state.movement_effects_present
                & (
                    state.movement_lock_mask
                    == INTERACTION_MOVEMENT_ALL_MASK
                )
            )
        )
    )
    return jnp.all(phase_valid & numeric_valid, axis=1)


def _validate_shapes(
    state,
    inventory,
    command,
    program,
    evidence,
    drops,
    inventory_layout,
):
    shape = state.phase.shape
    if len(shape) != 2:
        raise ValueError("block interaction phase must have rank 2")
    batch, entities = shape
    expected = {
        "state.kind": state.kind,
        "state.elapsed_seconds": state.elapsed_seconds,
        "state.duration_seconds": state.duration_seconds,
        "state.start_delay_seconds": state.start_delay_seconds,
        "state.wait_for_animation_to_finish": (
            state.wait_for_animation_to_finish
        ),
        "state.horizontal_speed_multiplier": (
            state.horizontal_speed_multiplier
        ),
        "state.movement_effects_present": state.movement_effects_present,
        "state.movement_disable_all": state.movement_disable_all,
        "state.movement_lock_mask": state.movement_lock_mask,
        "state.cancel_on_item_change": state.cancel_on_item_change,
        "state.next_program_id": state.next_program_id,
        "state.failed_program_id": state.failed_program_id,
        "state.held_container_id": state.held_container_id,
        "state.held_container_slot": state.held_container_slot,
        "state.held_item_id": state.held_item_id,
        "command.start": command.start,
        "command.held_container_id": command.held_container_id,
        "command.held_container_slot": command.held_container_slot,
    }
    expected.update(
        {f"program.{name}": value for name, value in program._asdict().items()}
    )
    expected.update(
        {
            f"evidence.{name}": value
            for name, value in evidence._asdict().items()
        }
    )
    for name, value in expected.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if state.target_block.shape != shape + (3,):
        raise ValueError("state.target_block must end in XYZ")
    if command.target_block.shape != shape + (3,):
        raise ValueError("command.target_block must end in XYZ")
    metadata_shape = shape + (METADATA_HASH_WORDS,)
    if state.held_metadata_hash.shape != metadata_shape:
        raise ValueError("state.held_metadata_hash has wrong shape")
    if state.failure_bits.shape != (batch,):
        raise ValueError("state.failure_bits must have shape [batch]")
    if inventory.item_id.shape[:2] != (batch, entities):
        raise ValueError("inventory batch/entity axes do not match")
    drop_shape = drops.mask.shape
    if drop_shape[:2] != (batch, entities) or len(drop_shape) != 3:
        raise ValueError("drop arrays must have shape [batch, entity, drop]")
    for name in (
        "item_id",
        "quantity",
        "durability",
        "max_durability",
        "item_max_stack",
        "pickup_container_id",
    ):
        if getattr(drops, name).shape != drop_shape:
            raise ValueError(f"drops.{name} must match drops.mask")
    if drops.metadata_hash.shape != drop_shape + (METADATA_HASH_WORDS,):
        raise ValueError("drops.metadata_hash has wrong shape")
    slot_count = inventory_layout.container_id.shape[0]
    if drops.pickup_slot_allowed.shape != drop_shape + (slot_count,):
        if drop_shape[2] != 0:
            raise ValueError("drops.pickup_slot_allowed has wrong slot axis")
