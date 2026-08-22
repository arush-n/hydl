"""Fixed-shape values for Combat-owned block interaction chains."""

from __future__ import annotations

from typing import NamedTuple

import jax


Array = jax.Array


class BlockInteractionProgram(NamedTuple):
    """Authored fields inherited by BreakBlock and PlaceBlock."""

    kind: Array
    run_time_seconds: Array
    animation_duration_seconds: Array
    start_delay_seconds: Array
    wait_for_animation_to_finish: Array
    horizontal_speed_multiplier: Array
    movement_effects_present: Array
    movement_disable_all: Array
    movement_lock_mask: Array
    cancel_on_item_change: Array
    next_program_id: Array
    failed_program_id: Array
    harvest: Array
    remove_item_in_hand: Array


class BlockInteractionCommand(NamedTuple):
    """One optional client block-interaction start per entity."""

    start: Array
    target_block: Array
    held_container_id: Array
    held_container_slot: Array


class BlockInteractionEvidence(NamedTuple):
    """World-owned facts and mutation acknowledgements."""

    mutation_supported: Array
    target_loaded: Array
    target_present: Array
    creative_mode: Array
    block_breaking_allowed: Array
    block_gathering_allowed: Array
    target_harvestable: Array
    damage_applied: Array
    block_health_before: Array
    block_damage: Array
    block_remove_applied: Array
    place_allowed: Array
    place_applied: Array
    placed_block_id: Array


class ResolvedBlockDrops(NamedTuple):
    """Asset-resolved first-run drops on a runtime-static drop axis."""

    mask: Array
    item_id: Array
    quantity: Array
    durability: Array
    max_durability: Array
    metadata_hash: Array
    item_max_stack: Array
    pickup_container_id: Array
    pickup_slot_allowed: Array


class BlockInteractionState(NamedTuple):
    """One in-flight interaction chain per batch entity."""

    phase: Array
    kind: Array
    elapsed_seconds: Array
    duration_seconds: Array
    start_delay_seconds: Array
    wait_for_animation_to_finish: Array
    horizontal_speed_multiplier: Array
    movement_effects_present: Array
    movement_disable_all: Array
    movement_lock_mask: Array
    cancel_on_item_change: Array
    next_program_id: Array
    failed_program_id: Array
    target_block: Array
    held_container_id: Array
    held_container_slot: Array
    held_item_id: Array
    held_metadata_hash: Array
    failure_bits: Array


class InteractionMovementConstraints(NamedTuple):
    """Current actor-local movement restrictions from one interaction."""

    active: Array
    movement_effects_present: Array
    movement_disable_all: Array
    movement_lock_mask: Array
    horizontal_speed_multiplier: Array


class BlockInteractionInfo(NamedTuple):
    """Policy-external diagnostics and World mutation requests."""

    start_accepted: Array
    first_run_effect: Array
    damage_requested: Array
    block_health_after: Array
    block_remove_requested: Array
    block_remove_applied: Array
    place_requested: Array
    place_applied: Array
    placed_block_id: Array
    world_drop_mask: Array
    pickup_accepted: Array
    pickup_remainder_quantity: Array
    item_change_cancelled: Array
    dispatched_program_id: Array


__all__ = [
    "BlockInteractionCommand",
    "BlockInteractionEvidence",
    "BlockInteractionInfo",
    "BlockInteractionProgram",
    "BlockInteractionState",
    "InteractionMovementConstraints",
    "ResolvedBlockDrops",
]
