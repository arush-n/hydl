"""Pure-JAX first-run block effects and interaction-chain scheduling."""

from __future__ import annotations


import jax.numpy as jnp

from hytalegym.jax.combat.block_interactions.schema.contract import (
    BLOCK_HEALTH_EPSILON,
    BLOCK_INTERACTION_BREAK,
    BLOCK_INTERACTION_CANCELLED,
    BLOCK_INTERACTION_FAILED,
    BLOCK_INTERACTION_FAILURE_INVALID_REQUEST,
    BLOCK_INTERACTION_FAILURE_INVALID_STATE,
    BLOCK_INTERACTION_FAILURE_WORLD_UNSUPPORTED,
    BLOCK_INTERACTION_IDLE,
    BLOCK_INTERACTION_NONE,
    BLOCK_INTERACTION_PLACE,
    BLOCK_INTERACTION_RUNNING,
    BLOCK_INTERACTION_SUCCEEDED,
    INTERACTION_MOVEMENT_ALL_MASK,
    INTERACTION_MOVEMENT_DISABLE_BACKWARD,
    INTERACTION_MOVEMENT_DISABLE_FORWARD,
    INTERACTION_MOVEMENT_DISABLE_JUMP,
    INTERACTION_MOVEMENT_DISABLE_LEFT,
    INTERACTION_MOVEMENT_DISABLE_RIGHT,
    NO_BLOCK_ID,
    NO_INTERACTION_PROGRAM,
)
from hytalegym.jax.combat.block_interactions.schema.types import (
    BlockInteractionCommand,
    BlockInteractionEvidence,
    BlockInteractionInfo,
    BlockInteractionProgram,
    BlockInteractionState,
    InteractionMovementConstraints,
    ResolvedBlockDrops,
)
from hytalegym.jax.combat.inventory import (
    CONTAINER_HOTBAR,
    CONTAINER_TOOLS,
    CONTAINER_UTILITY,
    EMPTY_ITEM_ID,
    METADATA_HASH_WORDS,
    item_stack_at,
)
from hytalegym.jax.combat.types import (
    ACTION_BACK,
    ACTION_FORWARD,
    ACTION_JUMP,
    ACTION_LEFT,
    ACTION_RIGHT,
)

# Re-exported to preserve this module's attribute surface.
from hytalegym.jax.combat.block_interactions.execution.inventory import (  # noqa: F401
    _active_slot_matches,
    _apply_harvest_pickups,
    _consume_placed_item,
)
from hytalegym.jax.combat.block_interactions.execution.validation import (  # noqa: F401
    _actor_index,
    _integer,
    _nonnegative,
    _positive,
    _shape,
    _state_valid,
    _validate_shapes,
)


def empty_block_interaction_program(
    batch_size: int,
    *,
    entity_count: int,
) -> BlockInteractionProgram:
    """Return an inert authored program."""

    shape = _shape(batch_size, entity_count)
    return BlockInteractionProgram(
        kind=jnp.full(shape, BLOCK_INTERACTION_NONE, dtype=jnp.int32),
        run_time_seconds=jnp.zeros(shape, dtype=jnp.float32),
        animation_duration_seconds=jnp.zeros(shape, dtype=jnp.float32),
        start_delay_seconds=jnp.zeros(shape, dtype=jnp.float32),
        wait_for_animation_to_finish=jnp.zeros(shape, dtype=jnp.bool_),
        horizontal_speed_multiplier=jnp.ones(shape, dtype=jnp.float32),
        movement_effects_present=jnp.zeros(shape, dtype=jnp.bool_),
        movement_disable_all=jnp.zeros(shape, dtype=jnp.bool_),
        movement_lock_mask=jnp.zeros(shape, dtype=jnp.int32),
        cancel_on_item_change=jnp.ones(shape, dtype=jnp.bool_),
        next_program_id=jnp.full(
            shape,
            NO_INTERACTION_PROGRAM,
            dtype=jnp.int32,
        ),
        failed_program_id=jnp.full(
            shape,
            NO_INTERACTION_PROGRAM,
            dtype=jnp.int32,
        ),
        harvest=jnp.zeros(shape, dtype=jnp.bool_),
        remove_item_in_hand=jnp.ones(shape, dtype=jnp.bool_),
    )


def empty_block_interaction_command(
    batch_size: int,
    *,
    entity_count: int,
) -> BlockInteractionCommand:
    """Return a no-start command."""

    shape = _shape(batch_size, entity_count)
    return BlockInteractionCommand(
        start=jnp.zeros(shape, dtype=jnp.bool_),
        target_block=jnp.zeros(shape + (3,), dtype=jnp.int32),
        held_container_id=jnp.full(
            shape,
            CONTAINER_HOTBAR,
            dtype=jnp.int32,
        ),
        held_container_slot=jnp.zeros(shape, dtype=jnp.int32),
    )


def empty_block_interaction_evidence(
    batch_size: int,
    *,
    entity_count: int,
) -> BlockInteractionEvidence:
    """Return fail-closed World evidence."""

    shape = _shape(batch_size, entity_count)
    false = jnp.zeros(shape, dtype=jnp.bool_)
    return BlockInteractionEvidence(
        mutation_supported=false,
        target_loaded=false,
        target_present=false,
        creative_mode=false,
        block_breaking_allowed=false,
        block_gathering_allowed=false,
        target_harvestable=false,
        damage_applied=false,
        block_health_before=jnp.ones(shape, dtype=jnp.float32),
        block_damage=jnp.zeros(shape, dtype=jnp.float32),
        block_remove_applied=false,
        place_allowed=false,
        place_applied=false,
        placed_block_id=jnp.full(shape, NO_BLOCK_ID, dtype=jnp.int32),
    )


def empty_resolved_block_drops(
    batch_size: int,
    *,
    entity_count: int,
    drop_capacity: int,
    inventory_slot_count: int = 0,
) -> ResolvedBlockDrops:
    """Return an empty runtime-static resolved drop bank."""

    batch, entities = _shape(batch_size, entity_count)
    drops = _nonnegative(drop_capacity, "drop_capacity")
    slots = _nonnegative(inventory_slot_count, "inventory_slot_count")
    shape = (batch, entities, drops)
    return ResolvedBlockDrops(
        mask=jnp.zeros(shape, dtype=jnp.bool_),
        item_id=jnp.full(shape, EMPTY_ITEM_ID, dtype=jnp.int32),
        quantity=jnp.zeros(shape, dtype=jnp.int32),
        durability=jnp.zeros(shape, dtype=jnp.float32),
        max_durability=jnp.zeros(shape, dtype=jnp.float32),
        metadata_hash=jnp.zeros(
            shape + (METADATA_HASH_WORDS,),
            dtype=jnp.uint32,
        ),
        item_max_stack=jnp.zeros(shape, dtype=jnp.int32),
        pickup_container_id=jnp.zeros(shape, dtype=jnp.int32),
        pickup_slot_allowed=jnp.zeros(
            shape + (slots,),
            dtype=jnp.bool_,
        ),
    )


def empty_block_interaction_state(
    batch_size: int,
    *,
    entity_count: int,
) -> BlockInteractionState:
    """Return the canonical idle chain state."""

    shape = _shape(batch_size, entity_count)
    return BlockInteractionState(
        phase=jnp.full(shape, BLOCK_INTERACTION_IDLE, dtype=jnp.int32),
        kind=jnp.full(shape, BLOCK_INTERACTION_NONE, dtype=jnp.int32),
        elapsed_seconds=jnp.zeros(shape, dtype=jnp.float32),
        duration_seconds=jnp.zeros(shape, dtype=jnp.float32),
        start_delay_seconds=jnp.zeros(shape, dtype=jnp.float32),
        wait_for_animation_to_finish=jnp.zeros(shape, dtype=jnp.bool_),
        horizontal_speed_multiplier=jnp.ones(shape, dtype=jnp.float32),
        movement_effects_present=jnp.zeros(shape, dtype=jnp.bool_),
        movement_disable_all=jnp.zeros(shape, dtype=jnp.bool_),
        movement_lock_mask=jnp.zeros(shape, dtype=jnp.int32),
        cancel_on_item_change=jnp.ones(shape, dtype=jnp.bool_),
        next_program_id=jnp.full(
            shape,
            NO_INTERACTION_PROGRAM,
            dtype=jnp.int32,
        ),
        failed_program_id=jnp.full(
            shape,
            NO_INTERACTION_PROGRAM,
            dtype=jnp.int32,
        ),
        target_block=jnp.zeros(shape + (3,), dtype=jnp.int32),
        held_container_id=jnp.full(
            shape,
            CONTAINER_HOTBAR,
            dtype=jnp.int32,
        ),
        held_container_slot=jnp.zeros(shape, dtype=jnp.int32),
        held_item_id=jnp.full(shape, EMPTY_ITEM_ID, dtype=jnp.int32),
        held_metadata_hash=jnp.zeros(
            shape + (METADATA_HASH_WORDS,),
            dtype=jnp.uint32,
        ),
        failure_bits=jnp.zeros((shape[0],), dtype=jnp.uint32),
    )


def step_block_interactions(
    state,
    inventory,
    inventory_layout,
    command,
    program,
    evidence,
    drops,
    *,
    dt_seconds,
):
    """Advance one server tick of Combat-owned block interaction state.

    World facts and mutation acknowledgements are injected. A caller with no
    mutable World implementation must pass the empty evidence and therefore
    fails closed.
    """

    batch, entities = state.phase.shape
    shape = (batch, entities)
    _validate_shapes(
        state,
        inventory,
        command,
        program,
        evidence,
        drops,
        inventory_layout,
    )
    dt = jnp.asarray(dt_seconds, dtype=jnp.float32)
    dt_valid = jnp.isfinite(dt) & (dt > 0.0)

    current_item = item_stack_at(
        inventory,
        inventory_layout,
        command.held_container_id,
        command.held_container_slot,
    )
    active_slot = _active_slot_matches(
        inventory,
        command.held_container_id,
        command.held_container_slot,
    )
    kind_valid = (
        (program.kind == BLOCK_INTERACTION_BREAK)
        | (program.kind == BLOCK_INTERACTION_PLACE)
    )
    numeric_valid = (
        jnp.isfinite(program.run_time_seconds)
        & (program.run_time_seconds >= 0.0)
        & jnp.isfinite(program.animation_duration_seconds)
        & (program.animation_duration_seconds >= 0.0)
        & jnp.isfinite(program.start_delay_seconds)
        & (program.start_delay_seconds >= 0.0)
        & jnp.isfinite(program.horizontal_speed_multiplier)
        & (program.horizontal_speed_multiplier >= 0.0)
        & (program.movement_lock_mask >= 0)
        & (program.movement_lock_mask <= INTERACTION_MOVEMENT_ALL_MASK)
        & (
            ~program.movement_disable_all
            | (
                program.movement_effects_present
                & (
                    program.movement_lock_mask
                    == INTERACTION_MOVEMENT_ALL_MASK
                )
            )
        )
    )
    held_reference_valid = (
        (
            (command.held_container_id == CONTAINER_HOTBAR)
            | (command.held_container_id == CONTAINER_UTILITY)
            | (command.held_container_id == CONTAINER_TOOLS)
        )
        & active_slot
    )
    request_valid = kind_valid & numeric_valid & held_reference_valid
    state_valid = _state_valid(state)
    can_start = state.phase != BLOCK_INTERACTION_RUNNING
    start_requested = command.start & can_start
    break_kind = program.kind == BLOCK_INTERACTION_BREAK
    place_kind = program.kind == BLOCK_INTERACTION_PLACE
    harvest_kind = break_kind & program.harvest
    normal_break = break_kind & ~program.harvest
    place_height_valid = (
        (command.target_block[..., 1] >= 0)
        & (command.target_block[..., 1] < 320)
    )
    held_present = current_item.item_id != EMPTY_ITEM_ID
    drop_item_valid = (
        (drops.item_id >= 0)
        & (drops.quantity > 0)
        & jnp.isfinite(drops.durability)
        & (drops.durability >= 0.0)
        & jnp.isfinite(drops.max_durability)
        & (drops.max_durability >= 0.0)
    )
    world_drop_valid = jnp.all(~drops.mask | drop_item_valid, axis=2)
    pickup_drop_valid = jnp.all(
        ~drops.mask
        | (
            drop_item_valid
            & (drops.item_max_stack > 0)
            & jnp.any(drops.pickup_slot_allowed, axis=3)
        ),
        axis=2,
    )
    damage_evidence_valid = (
        ~evidence.damage_applied
        | (
            jnp.isfinite(evidence.block_health_before)
            & (evidence.block_health_before >= 0.0)
            & (evidence.block_health_before <= 1.0)
            & jnp.isfinite(evidence.block_damage)
            & (evidence.block_damage >= 0.0)
        )
    )
    normal_break_evidence_valid = jnp.where(
        evidence.creative_mode,
        world_drop_valid,
        damage_evidence_valid & world_drop_valid,
    )
    break_precondition = (
        evidence.mutation_supported
        & evidence.target_loaded
        & evidence.target_present
        & jnp.where(
            program.harvest,
            evidence.block_gathering_allowed
            & evidence.target_harvestable
            & pickup_drop_valid,
            evidence.block_breaking_allowed
            & normal_break_evidence_valid,
        )
    )
    place_precondition = (
        evidence.mutation_supported
        & evidence.target_loaded
        & evidence.place_allowed
        & place_height_valid
        & held_present
    )
    precondition = jnp.where(
        break_kind,
        break_precondition,
        place_precondition,
    )
    start_success = (
        start_requested
        & request_valid
        & state_valid[:, None]
        & dt_valid
        & precondition
    )
    start_failed = start_requested & ~start_success
    duration = jnp.maximum(
        program.run_time_seconds,
        jnp.where(
            program.wait_for_animation_to_finish,
            program.animation_duration_seconds,
            jnp.float32(0.0),
        ),
    )

    stored_item = item_stack_at(
        inventory,
        inventory_layout,
        state.held_container_id,
        state.held_container_slot,
    )
    stored_slot_active = _active_slot_matches(
        inventory,
        state.held_container_id,
        state.held_container_slot,
    )
    item_equivalent = (
        (stored_item.item_id == state.held_item_id)
        & jnp.all(
            stored_item.metadata_hash == state.held_metadata_hash,
            axis=2,
        )
    )
    was_running = state.phase == BLOCK_INTERACTION_RUNNING
    item_change_cancelled = (
        was_running
        & state.cancel_on_item_change
        & (~stored_slot_active | ~item_equivalent)
    )
    advanced_elapsed = jnp.minimum(
        state.duration_seconds,
        state.elapsed_seconds + dt,
    )
    completed = (
        was_running
        & ~item_change_cancelled
        & (advanced_elapsed >= state.duration_seconds)
    )

    next_phase = state.phase
    next_phase = jnp.where(
        was_running & ~item_change_cancelled & ~completed,
        BLOCK_INTERACTION_RUNNING,
        next_phase,
    )
    next_phase = jnp.where(
        completed,
        BLOCK_INTERACTION_SUCCEEDED,
        next_phase,
    )
    next_phase = jnp.where(
        item_change_cancelled,
        BLOCK_INTERACTION_CANCELLED,
        next_phase,
    )
    next_phase = jnp.where(
        start_success & (duration > 0.0),
        BLOCK_INTERACTION_RUNNING,
        next_phase,
    )
    next_phase = jnp.where(
        start_success & (duration <= 0.0),
        BLOCK_INTERACTION_SUCCEEDED,
        next_phase,
    )
    next_phase = jnp.where(
        start_failed,
        BLOCK_INTERACTION_FAILED,
        next_phase,
    )

    elapsed = jnp.where(was_running, advanced_elapsed, state.elapsed_seconds)
    elapsed = jnp.where(start_requested, 0.0, elapsed)
    elapsed = jnp.where(
        completed,
        state.duration_seconds,
        elapsed,
    )

    damage_requested = (
        start_success
        & normal_break
        & ~evidence.creative_mode
        & evidence.damage_applied
    )
    raw_health_after = (
        evidence.block_health_before - evidence.block_damage
    )
    block_health_after = jnp.where(
        damage_requested,
        raw_health_after,
        evidence.block_health_before,
    )
    destroyed = damage_requested & (
        (block_health_after < 0.0)
        | (jnp.abs(block_health_after) <= BLOCK_HEALTH_EPSILON)
    )
    block_remove_requested = start_success & (
        harvest_kind
        | (normal_break & evidence.creative_mode)
        | destroyed
    )
    block_remove_applied = (
        block_remove_requested & evidence.block_remove_applied
    )
    place_requested = start_success & place_kind
    place_applied = place_requested & evidence.place_applied

    next_inventory, pickup_accepted, pickup_remainder = (
        _apply_harvest_pickups(
            inventory,
            inventory_layout,
            drops,
            block_remove_applied & harvest_kind,
        )
    )
    next_inventory = _consume_placed_item(
        next_inventory,
        inventory_layout,
        current_item,
        command,
        place_applied & program.remove_item_in_hand,
    )

    drop_mask = drops.mask & (
        block_remove_applied & normal_break
    )[..., None]
    dispatched = jnp.full(
        shape,
        NO_INTERACTION_PROGRAM,
        dtype=jnp.int32,
    )
    dispatched = jnp.where(
        completed
        | (start_success & (duration <= 0.0)),
        jnp.where(
            start_requested,
            program.next_program_id,
            state.next_program_id,
        ),
        dispatched,
    )
    dispatched = jnp.where(
        start_failed,
        program.failed_program_id,
        dispatched,
    )

    unsupported = start_requested & ~evidence.mutation_supported
    invalid_request = start_requested & (~request_valid | ~dt_valid)
    invalid_state = ~state_valid
    failure_bits = state.failure_bits
    failure_bits |= jnp.where(
        jnp.any(unsupported, axis=1),
        jnp.uint32(BLOCK_INTERACTION_FAILURE_WORLD_UNSUPPORTED),
        jnp.uint32(0),
    )
    failure_bits |= jnp.where(
        jnp.any(invalid_request, axis=1),
        jnp.uint32(BLOCK_INTERACTION_FAILURE_INVALID_REQUEST),
        jnp.uint32(0),
    )
    failure_bits |= jnp.where(
        invalid_state,
        jnp.uint32(BLOCK_INTERACTION_FAILURE_INVALID_STATE),
        jnp.uint32(0),
    )
    next_state = state._replace(
        phase=next_phase.astype(jnp.int32),
        kind=jnp.where(start_requested, program.kind, state.kind),
        elapsed_seconds=elapsed,
        duration_seconds=jnp.where(
            start_requested,
            duration,
            state.duration_seconds,
        ),
        start_delay_seconds=jnp.where(
            start_requested,
            program.start_delay_seconds,
            state.start_delay_seconds,
        ),
        wait_for_animation_to_finish=jnp.where(
            start_requested,
            program.wait_for_animation_to_finish,
            state.wait_for_animation_to_finish,
        ),
        horizontal_speed_multiplier=jnp.where(
            start_requested,
            program.horizontal_speed_multiplier,
            state.horizontal_speed_multiplier,
        ),
        movement_effects_present=jnp.where(
            start_requested,
            program.movement_effects_present,
            state.movement_effects_present,
        ),
        movement_disable_all=jnp.where(
            start_requested,
            program.movement_disable_all,
            state.movement_disable_all,
        ),
        movement_lock_mask=jnp.where(
            start_requested,
            program.movement_lock_mask,
            state.movement_lock_mask,
        ),
        cancel_on_item_change=jnp.where(
            start_requested,
            program.cancel_on_item_change,
            state.cancel_on_item_change,
        ),
        next_program_id=jnp.where(
            start_requested,
            program.next_program_id,
            state.next_program_id,
        ),
        failed_program_id=jnp.where(
            start_requested,
            program.failed_program_id,
            state.failed_program_id,
        ),
        target_block=jnp.where(
            start_requested[..., None],
            command.target_block,
            state.target_block,
        ),
        held_container_id=jnp.where(
            start_requested,
            command.held_container_id,
            state.held_container_id,
        ),
        held_container_slot=jnp.where(
            start_requested,
            command.held_container_slot,
            state.held_container_slot,
        ),
        held_item_id=jnp.where(
            start_requested,
            current_item.item_id,
            state.held_item_id,
        ),
        held_metadata_hash=jnp.where(
            start_requested[..., None],
            current_item.metadata_hash,
            state.held_metadata_hash,
        ),
        failure_bits=failure_bits,
    )
    info = BlockInteractionInfo(
        start_accepted=start_success,
        first_run_effect=start_success,
        damage_requested=damage_requested,
        block_health_after=block_health_after,
        block_remove_requested=block_remove_requested,
        block_remove_applied=block_remove_applied,
        place_requested=place_requested,
        place_applied=place_applied,
        placed_block_id=jnp.where(
            place_applied,
            evidence.placed_block_id,
            NO_BLOCK_ID,
        ),
        world_drop_mask=drop_mask,
        pickup_accepted=pickup_accepted,
        pickup_remainder_quantity=pickup_remainder,
        item_change_cancelled=item_change_cancelled,
        dispatched_program_id=dispatched,
    )
    return next_state, next_inventory, info


def interaction_movement_constraints(
    state: BlockInteractionState,
    *,
    actor_index: int = 0,
) -> InteractionMovementConstraints:
    """Project one actor's active interaction movement fields.

    Missing movement effects are preserved as unknown. Consumers decide how
    to fail closed; an idle actor is never constrained by stale state fields.
    """

    if not isinstance(state, BlockInteractionState):
        raise TypeError("state must be BlockInteractionState")
    actor = _actor_index(actor_index, state.phase.shape[1])
    active = state.phase[:, actor] == BLOCK_INTERACTION_RUNNING
    return InteractionMovementConstraints(
        active=active,
        movement_effects_present=state.movement_effects_present[:, actor],
        movement_disable_all=state.movement_disable_all[:, actor],
        movement_lock_mask=state.movement_lock_mask[:, actor],
        horizontal_speed_multiplier=jnp.where(
            active,
            state.horizontal_speed_multiplier[:, actor],
            jnp.float32(1.0),
        ),
    )


def apply_interaction_movement_constraints(
    state: BlockInteractionState,
    low_level_action,
    *,
    actor_index: int = 0,
):
    """Apply active native movement effects to the executable action.

    This is the hard runtime gate behind the actor-facing mask. It preserves
    individual local-direction semantics after the world-frame move and look
    factors have been jointly decoded.
    """

    actions = jnp.asarray(low_level_action)
    if actions.dtype != jnp.float32 or actions.ndim != 2:
        raise ValueError("low_level_action must be float32 with shape [B, A]")
    constraints = interaction_movement_constraints(
        state,
        actor_index=actor_index,
    )
    if constraints.active.shape != (actions.shape[0],):
        raise ValueError("interaction movement batch does not match actions")
    movement_known = (
        constraints.movement_effects_present
        & (constraints.movement_lock_mask >= 0)
        & (
            constraints.movement_lock_mask
            <= jnp.int32(INTERACTION_MOVEMENT_ALL_MASK)
        )
    )
    effective_mask = jnp.where(
        constraints.active,
        jnp.where(
            movement_known,
            jnp.where(
                constraints.movement_disable_all,
                jnp.int32(INTERACTION_MOVEMENT_ALL_MASK),
                constraints.movement_lock_mask,
            ),
            jnp.int32(INTERACTION_MOVEMENT_ALL_MASK),
        ),
        jnp.int32(0),
    )
    speed = jnp.where(
        jnp.isfinite(constraints.horizontal_speed_multiplier)
        & (constraints.horizontal_speed_multiplier >= 0.0),
        constraints.horizontal_speed_multiplier,
        jnp.float32(0.0),
    )
    result = actions
    for action_index, lock_bit in (
        (ACTION_FORWARD, INTERACTION_MOVEMENT_DISABLE_FORWARD),
        (ACTION_BACK, INTERACTION_MOVEMENT_DISABLE_BACKWARD),
        (ACTION_LEFT, INTERACTION_MOVEMENT_DISABLE_LEFT),
        (ACTION_RIGHT, INTERACTION_MOVEMENT_DISABLE_RIGHT),
    ):
        value = (
            result[:, action_index]
            * speed
        )
        result = result.at[:, action_index].set(
            jnp.where(
                (effective_mask & jnp.int32(lock_bit)) != 0,
                jnp.float32(0.0),
                value,
            )
        )
    result = result.at[:, ACTION_JUMP].set(
        jnp.where(
            (
                effective_mask
                & jnp.int32(INTERACTION_MOVEMENT_DISABLE_JUMP)
            )
            != 0,
            jnp.float32(0.0),
            result[:, ACTION_JUMP],
        )
    )
    return result


__all__ = [
    "apply_interaction_movement_constraints",
    "empty_block_interaction_command",
    "empty_block_interaction_evidence",
    "empty_block_interaction_program",
    "empty_block_interaction_state",
    "empty_resolved_block_drops",
    "interaction_movement_constraints",
    "step_block_interactions",
]
