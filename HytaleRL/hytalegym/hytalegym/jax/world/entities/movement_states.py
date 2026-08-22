"""Fixed-shape Hytale 0.5.7 NPC movement-state production."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp


Array = jax.Array

MOVEMENT_STATE_SCHEMA = "hytalerl_npc_walk_movement_states_v1"
MOVEMENT_STATE_VERSION = 1
MOVEMENT_STATE_ORDER = (
    "idle",
    "horizontal_idle",
    "jumping",
    "flying",
    "walking",
    "running",
    "sprinting",
    "crouching",
    "forced_crouching",
    "falling",
    "falling_far",
    "climbing",
    "in_fluid",
    "swimming",
    "swim_jumping",
    "on_ground",
    "mantling",
    "sliding",
    "mounting",
    "rolling",
    "sitting",
    "gliding",
    "sleeping",
)
MOVEMENT_STATE_COUNT = len(MOVEMENT_STATE_ORDER)

MOTION_KIND_ASCENDING = 0
MOTION_KIND_DESCENDING = 1
MOTION_KIND_DROPPING = 2
MOTION_KIND_STANDING = 3
MOTION_KIND_MOVING = 4
MOTION_KIND_FLYING = 5
MOTION_KIND_SWIMMING = 6
MOTION_KIND_SWIMMING_TURNING = 7
MOTION_KIND_UNINITIALIZED = -1

ASCENT_ANIMATION_WALK = 0
ASCENT_ANIMATION_JUMP = 1
ASCENT_ANIMATION_CLIMB = 2
ASCENT_ANIMATION_FLY = 3
ASCENT_ANIMATION_IDLE = 4

DESCENT_ANIMATION_WALK = 0
DESCENT_ANIMATION_FALL = 1
DESCENT_ANIMATION_IDLE = 2

_IDLE = 0
_HORIZONTAL_IDLE = 1
_JUMPING = 2
_FLYING = 3
_WALKING = 4
_RUNNING = 5
_SPRINTING = 6
_FALLING = 9
_CLIMBING = 11
_IN_FLUID = 12
_SWIMMING = 13
_SWIM_JUMPING = 14
_ON_GROUND = 15
_ALWAYS_WRITTEN = (_CLIMBING, _IN_FLUID, _SWIM_JUMPING, _ON_GROUND)


class MovementStateEvidence(NamedTuple):
    """Availability-masked values in exact protocol field order."""

    values: Array
    available: Array
    invalid: Array


class NpcWalkMovementState(NamedTuple):
    """Recurrent state retained by ``MotionControllerBase`` and Walk."""

    values: Array
    available: Array
    previous_speed: Array
    fast_motion_kind: Array
    last_motion_kind: Array


class NpcWalkMovementStateResult(NamedTuple):
    """One source-ordered movement-state update."""

    state: NpcWalkMovementState
    invalid: Array


def _filtered_selected_velocity_speed(
    previous_speed: Array,
    movement: Array,
    selector: Array,
) -> Array:
    """Evaluate the native source order identically under eager and JIT."""

    selected_velocity = movement * selector
    squared_speed = jax.lax.fori_loop(
        0,
        3,
        lambda index, total: (
            total
            + selected_velocity[:, index] * selected_velocity[:, index]
        ),
        jnp.zeros_like(previous_speed),
    )
    weighted_terms = jnp.stack(
        (
            jnp.float32(0.7) * previous_speed,
            jnp.float32(0.3) * jnp.sqrt(squared_speed),
        ),
        axis=0,
    )
    return jax.lax.fori_loop(
        0,
        2,
        lambda index, total: total + weighted_terms[index],
        jnp.zeros_like(previous_speed),
    )


def empty_npc_walk_movement_state(batch_size: int) -> NpcWalkMovementState:
    """Return a canonical unavailable state for a new fixed-size batch."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    shape = (batch_size, MOVEMENT_STATE_COUNT)
    return NpcWalkMovementState(
        values=jnp.zeros(shape, dtype=jnp.bool_),
        available=jnp.zeros(shape, dtype=jnp.bool_),
        previous_speed=jnp.zeros((batch_size,), dtype=jnp.float32),
        fast_motion_kind=jnp.zeros((batch_size,), dtype=jnp.bool_),
        last_motion_kind=jnp.full(
            (batch_size,),
            MOTION_KIND_UNINITIALIZED,
            dtype=jnp.int8,
        ),
    )


def native_movement_state_evidence(
    bits: Array,
    row_available: Array,
) -> MovementStateEvidence:
    """Decode the bridge bitset without treating missing state as all-false."""

    packed = jnp.asarray(bits, dtype=jnp.int32)
    available = jnp.asarray(row_available, dtype=jnp.bool_)
    if packed.ndim != 1 or packed.shape[0] == 0:
        raise ValueError("bits must have non-empty shape [batch]")
    if available.shape != packed.shape:
        raise ValueError("row_available must match bits")
    bit_range = (packed >= 0) & (packed < (1 << MOVEMENT_STATE_COUNT))
    canonical_missing = available | (packed == 0)
    valid = bit_range & canonical_missing
    values = (
        packed[:, None]
        & (jnp.int32(1) << jnp.arange(MOVEMENT_STATE_COUNT))
    ) != 0
    field_available = valid[:, None] & available[:, None]
    return MovementStateEvidence(
        values=field_available & values,
        available=field_available,
        invalid=~valid,
    )


def npc_walk_movement_state_result(
    state: NpcWalkMovementState,
    velocity: Array,
    steering_translation: Array,
    motion_kind: Array,
    *,
    controller_available: Array,
    in_fluid: Array,
    on_ground: Array,
    hover_height: Array,
    controller_jumping: Array,
    component_selector: Array,
    maximum_horizontal_speed: Array,
    fast_motion_threshold: Array,
    fast_motion_threshold_range: Array,
    ascent_animation_type: Array,
    descent_animation_type: Array,
    predicted_fall_height: Array,
    minimum_descent_animation_height: Array,
) -> NpcWalkMovementStateResult:
    """Port one ``MotionControllerBase.updateMovementState`` Walk update.

    Role-configurable values are required inputs. Unsupported action-owned
    fields retain their prior value and availability. Invalid rows are fully
    cleared so stale state cannot masquerade as evidence.
    """

    if not isinstance(state, NpcWalkMovementState):
        raise TypeError("state must be NpcWalkMovementState")
    movement = jnp.asarray(velocity, dtype=jnp.float32)
    steering = jnp.asarray(steering_translation, dtype=jnp.float32)
    if (
        movement.ndim != 2
        or movement.shape[0] == 0
        or movement.shape[1] != 3
    ):
        raise ValueError("velocity must have non-empty shape [batch,3]")
    if steering.shape != movement.shape:
        raise ValueError("steering_translation must match velocity")
    batch = movement.shape[0]
    _validate_state_shape(state, batch)

    kind = _batch_int(motion_kind, batch, "motion_kind")
    enabled = _batch_bool(
        controller_available,
        batch,
        "controller_available",
    )
    fluid = _batch_bool(in_fluid, batch, "in_fluid")
    grounded = _batch_bool(on_ground, batch, "on_ground")
    hover = _batch_float(hover_height, batch, "hover_height")
    jumping = _batch_bool(
        controller_jumping,
        batch,
        "controller_jumping",
    )
    selector = _batch_vector(
        component_selector,
        batch,
        "component_selector",
    )
    maximum_speed = _batch_float(
        maximum_horizontal_speed,
        batch,
        "maximum_horizontal_speed",
    )
    threshold = _batch_float(
        fast_motion_threshold,
        batch,
        "fast_motion_threshold",
    )
    threshold_range = _batch_float(
        fast_motion_threshold_range,
        batch,
        "fast_motion_threshold_range",
    )
    ascent = _batch_int(
        ascent_animation_type,
        batch,
        "ascent_animation_type",
    )
    descent = _batch_int(
        descent_animation_type,
        batch,
        "descent_animation_type",
    )
    predicted_drop = _batch_float(
        predicted_fall_height,
        batch,
        "predicted_fall_height",
    )
    minimum_drop = _batch_float(
        minimum_descent_animation_height,
        batch,
        "minimum_descent_animation_height",
    )

    canonical_state = jnp.all(
        state.available | ~state.values,
        axis=1,
    )
    state_valid = (
        canonical_state
        & jnp.all(jnp.isfinite(state.previous_speed[:, None]), axis=1)
        & (state.previous_speed >= 0.0)
        & (state.last_motion_kind >= MOTION_KIND_UNINITIALIZED)
        & (state.last_motion_kind <= MOTION_KIND_SWIMMING_TURNING)
    )
    valid = (
        enabled
        & state_valid
        & jnp.all(jnp.isfinite(movement), axis=1)
        & jnp.all(jnp.isfinite(steering), axis=1)
        & (kind >= MOTION_KIND_ASCENDING)
        & (kind <= MOTION_KIND_SWIMMING_TURNING)
        & jnp.isfinite(hover)
        & (hover >= 0.0)
        & jnp.all(jnp.isfinite(selector), axis=1)
        & jnp.all(selector >= 0.0, axis=1)
        & jnp.any(selector > 0.0, axis=1)
        & jnp.isfinite(maximum_speed)
        & (maximum_speed > 0.0)
        & jnp.isfinite(threshold)
        & (threshold >= 0.0)
        & (threshold <= 1.0)
        & jnp.isfinite(threshold_range)
        & (threshold_range >= 0.0)
        & (threshold_range <= 1.0)
        & (ascent >= ASCENT_ANIMATION_WALK)
        & (ascent <= ASCENT_ANIMATION_IDLE)
        & (descent >= DESCENT_ANIMATION_WALK)
        & (descent <= DESCENT_ANIMATION_IDLE)
        & jnp.isfinite(predicted_drop)
        & jnp.isfinite(minimum_drop)
        & (minimum_drop >= 0.0)
    )

    speed = _filtered_selected_velocity_speed(
        state.previous_speed,
        movement,
        selector,
    )
    run_boundary = jnp.where(
        state.fast_motion_kind,
        threshold - threshold_range,
        threshold + threshold_range,
    ) * maximum_speed
    fast_motion = jnp.where(
        jumping & (kind == MOTION_KIND_ASCENDING),
        state.fast_motion_kind,
        speed > run_boundary,
    )
    idle = jnp.all(steering == 0.0, axis=1)
    horizontal_idle = speed == 0.0

    values = state.values
    available = state.available
    values = values.at[:, _CLIMBING].set(False)
    values = values.at[:, _SWIM_JUMPING].set(False)
    values = values.at[:, _IN_FLUID].set(fluid)
    values = values.at[:, _ON_GROUND].set(grounded)
    for index in _ALWAYS_WRITTEN:
        available = available.at[:, index].set(True)

    update_due = valid & (
        (kind != state.last_motion_kind.astype(jnp.int32))
        | (values[:, _RUNNING] != fast_motion)
        | (values[:, _IDLE] != idle)
        | (values[:, _HORIZONTAL_IDLE] != horizontal_idle)
    )
    branch_values = values
    branch_writes = jnp.zeros_like(values)

    def write(
        index: int,
        value: Array | bool,
        condition: Array,
    ) -> None:
        nonlocal branch_values, branch_writes
        update = jnp.asarray(value, dtype=jnp.bool_)
        if update.ndim == 0:
            update = jnp.broadcast_to(update, (batch,))
        branch_values = branch_values.at[:, index].set(
            jnp.where(condition, update, branch_values[:, index])
        )
        branch_writes = branch_writes.at[:, index].set(
            branch_writes[:, index] | condition
        )

    flying = kind == MOTION_KIND_FLYING
    for index, value in (
        (_FLYING, True),
        (_IDLE, idle),
        (_HORIZONTAL_IDLE, False),
        (_WALKING, ~fast_motion),
        (_RUNNING, fast_motion),
        (_FALLING, False),
        (_SWIMMING, False),
        (_JUMPING, False),
    ):
        write(index, value, flying)

    swimming = kind == MOTION_KIND_SWIMMING
    for index, value in (
        (_FLYING, False),
        (_IDLE, idle),
        (_HORIZONTAL_IDLE, horizontal_idle),
        (_WALKING, ~fast_motion),
        (_RUNNING, fast_motion),
        (_FALLING, False),
        (_SWIMMING, True),
        (_JUMPING, False),
    ):
        write(index, value, swimming)

    turning = kind == MOTION_KIND_SWIMMING_TURNING
    for index, value in (
        (_FLYING, False),
        (_IDLE, False),
        (_HORIZONTAL_IDLE, False),
        (_WALKING, False),
        (_RUNNING, True),
        (_FALLING, False),
        (_SWIMMING, True),
        (_JUMPING, False),
    ):
        write(index, value, turning)

    moving = kind == MOTION_KIND_MOVING
    for index, value in (
        (_FLYING, hover > 0.0),
        (_IDLE, False),
        (_HORIZONTAL_IDLE, False),
        (_FALLING, False),
        (_WALKING, ~fast_motion),
        (_RUNNING, fast_motion),
        (_SWIMMING, False),
        (_JUMPING, False),
    ):
        write(index, value, moving)

    standing = kind == MOTION_KIND_STANDING
    for index, value in (
        (_FLYING, hover > 0.0),
        (_IDLE, True),
        (_HORIZONTAL_IDLE, True),
        (_WALKING, False),
        (_RUNNING, False),
        (_FALLING, False),
        (_SWIMMING, False),
        (_JUMPING, False),
    ):
        write(index, value, standing)

    write(_FALLING, True, kind == MOTION_KIND_DROPPING)

    ascending = kind == MOTION_KIND_ASCENDING
    ascent_walk = ascending & ~jumping & (ascent == ASCENT_ANIMATION_WALK)
    ascent_jump = ascending & ~jumping & (ascent == ASCENT_ANIMATION_JUMP)
    ascent_climb = ascending & ~jumping & (ascent == ASCENT_ANIMATION_CLIMB)
    ascent_fly = ascending & ~jumping & (ascent == ASCENT_ANIMATION_FLY)
    ascent_idle = ascending & ~jumping & (ascent == ASCENT_ANIMATION_IDLE)
    write(_JUMPING, False, ascent_walk | ascent_climb | ascent_fly | ascent_idle)
    write(_JUMPING, True, ascent_jump | (ascending & jumping))
    write(_IDLE, False, ascent_walk | ascent_jump | ascent_climb | ascent_fly)
    write(_IDLE, True, ascent_idle)
    write(_IDLE, False, ascending & jumping)
    write(_FLYING, False, ascent_walk | ascent_jump | ascent_climb | ascent_idle)
    write(_FLYING, True, ascent_fly)
    write(_FLYING, hover > 0.0, ascending & jumping)
    write(_CLIMBING, True, ascent_climb)
    for index, value in (
        (_HORIZONTAL_IDLE, horizontal_idle),
        (_FALLING, False),
        (_RUNNING, fast_motion),
        (_WALKING, ~fast_motion),
        (_SPRINTING, False),
        (_SWIMMING, False),
    ):
        write(index, value, ascending)

    descending = kind == MOTION_KIND_DESCENDING
    uses_descent = predicted_drop >= (
        minimum_drop - jnp.float32(1.0e-5)
    )
    descent_walk = descending & (
        ~uses_descent | (descent == DESCENT_ANIMATION_WALK)
    )
    descent_fall = (
        descending & uses_descent & (descent == DESCENT_ANIMATION_FALL)
    )
    descent_idle = (
        descending & uses_descent & (descent == DESCENT_ANIMATION_IDLE)
    )
    for index, value in (
        (_FALLING, False),
        (_IDLE, False),
        (_RUNNING, fast_motion),
        (_WALKING, ~fast_motion),
        (_FLYING, hover > 0.0),
    ):
        write(index, value, descent_walk)
    for index, value in (
        (_FALLING, True),
        (_IDLE, False),
        (_RUNNING, False),
        (_WALKING, False),
        (_FLYING, False),
    ):
        write(index, value, descent_fall)
    for index, value in (
        (_FALLING, False),
        (_IDLE, True),
        (_RUNNING, False),
        (_WALKING, False),
        (_FLYING, hover > 0.0),
    ):
        write(index, value, descent_idle)
    for index, value in (
        (_HORIZONTAL_IDLE, False),
        (_SWIMMING, False),
        (_JUMPING, False),
        (_SPRINTING, False),
    ):
        write(index, value, descending)

    changed = update_due[:, None] & branch_writes
    values = jnp.where(changed, branch_values, values)
    available = available | changed
    values = jnp.where(valid[:, None], values, False)
    available = valid[:, None] & available
    next_state = NpcWalkMovementState(
        values=values,
        available=available,
        previous_speed=jnp.where(valid, speed, 0.0),
        fast_motion_kind=valid & fast_motion,
        last_motion_kind=jnp.where(
            valid,
            kind,
            MOTION_KIND_UNINITIALIZED,
        ).astype(jnp.int8),
    )
    return NpcWalkMovementStateResult(
        state=next_state,
        invalid=~valid,
    )


def movement_state_contract() -> dict[str, object]:
    """Return the source-backed producer and unsupported-field boundary."""

    return {
        "schema": MOVEMENT_STATE_SCHEMA,
        "version": MOVEMENT_STATE_VERSION,
        "protocol_order": list(MOVEMENT_STATE_ORDER),
        "native_transport": {
            "schema": "hytalerl_native_movement_states_v1",
            "encoding": "availability_plus_23_bit_mask",
            "provenance": "native",
        },
        "simulator_producer": {
            "sources": [
                "MotionControllerBase.updateMovementState",
                "MotionControllerWalk.updateAscendingStates",
                "MotionControllerWalk.updateDescendingStates",
                "MotionControllerWalk.isFastMotionKind",
                "BuilderMotionControllerBase.RunThreshold",
                "BuilderMotionControllerBase.RunThresholdRange",
            ],
            "speed_filter": "0.7_previous_plus_0.3_selected_velocity_norm",
            "speed_filter_execution": (
                "source_ordered_fori_loop_for_eager_jit_bit_exactness"
            ),
            "idle": "steering_translation_exactly_zero",
            "horizontal_idle": "filtered_speed_exactly_zero",
            "configuration": "role_values_are_required_inputs",
            "provenance": "native_source_port_jax_float32",
        },
        "field_ownership": {
            "npc_assigned_field_count": 13,
            "simulator_available_field_count": 12,
            "controller": [
                "idle",
                "horizontal_idle",
                "jumping",
                "flying",
                "walking",
                "running",
                "sprinting",
                "falling",
                "climbing",
                "in_fluid",
                "swimming",
                "swim_jumping",
                "on_ground",
            ],
            "preserved_external": [
                "crouching",
                "forced_crouching",
                "falling_far",
                "mantling",
                "sliding",
                "mounting",
                "rolling",
                "sitting",
                "gliding",
                "sleeping",
            ],
        },
        "sprint_boundary": {
            "npc_fast_motion_field": "running",
            "npc_walk_sprinting": "source_pinned_false_no_npc_writer_sets_true",
            "player_sprinting": "client_transition_not_simulated_here",
            "unconditional_multiplier": None,
        },
        "fly_boundary": (
            "state_classification_only_no_fly_physics_or_action_publication"
        ),
        "failure": (
            "invalid_or_unavailable_controller_row_clears_all_state_and_masks"
        ),
    }


def movement_state_contract_sha256() -> str:
    return hashlib.sha256(
        json.dumps(
            movement_state_contract(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _validate_state_shape(state: NpcWalkMovementState, batch: int) -> None:
    expected = (batch, MOVEMENT_STATE_COUNT)
    if state.values.shape != expected or state.available.shape != expected:
        raise ValueError(f"movement state values and masks must have shape {expected}")
    for name in ("previous_speed", "fast_motion_kind", "last_motion_kind"):
        if getattr(state, name).shape != (batch,):
            raise ValueError(f"state.{name} must have shape [{batch}]")


def _batch_bool(value: Array, batch: int, label: str) -> Array:
    result = jnp.asarray(value, dtype=jnp.bool_)
    if result.ndim == 0:
        result = jnp.broadcast_to(result, (batch,))
    if result.shape != (batch,):
        raise ValueError(f"{label} must be scalar or shape [batch]")
    return result


def _batch_float(value: Array, batch: int, label: str) -> Array:
    result = jnp.asarray(value, dtype=jnp.float32)
    if result.ndim == 0:
        result = jnp.broadcast_to(result, (batch,))
    if result.shape != (batch,):
        raise ValueError(f"{label} must be scalar or shape [batch]")
    return result


def _batch_int(value: Array, batch: int, label: str) -> Array:
    result = jnp.asarray(value, dtype=jnp.int32)
    if result.ndim == 0:
        result = jnp.broadcast_to(result, (batch,))
    if result.shape != (batch,):
        raise ValueError(f"{label} must be scalar or shape [batch]")
    return result


def _batch_vector(value: Array, batch: int, label: str) -> Array:
    result = jnp.asarray(value, dtype=jnp.float32)
    if result.shape == (3,):
        result = jnp.broadcast_to(result, (batch, 3))
    if result.shape != (batch, 3):
        raise ValueError(f"{label} must have shape [3] or [batch,3]")
    return result


__all__ = [
    "ASCENT_ANIMATION_CLIMB",
    "ASCENT_ANIMATION_FLY",
    "ASCENT_ANIMATION_IDLE",
    "ASCENT_ANIMATION_JUMP",
    "ASCENT_ANIMATION_WALK",
    "DESCENT_ANIMATION_FALL",
    "DESCENT_ANIMATION_IDLE",
    "DESCENT_ANIMATION_WALK",
    "MOTION_KIND_ASCENDING",
    "MOTION_KIND_DESCENDING",
    "MOTION_KIND_DROPPING",
    "MOTION_KIND_FLYING",
    "MOTION_KIND_MOVING",
    "MOTION_KIND_STANDING",
    "MOTION_KIND_SWIMMING",
    "MOTION_KIND_SWIMMING_TURNING",
    "MOVEMENT_STATE_COUNT",
    "MOVEMENT_STATE_ORDER",
    "MOVEMENT_STATE_SCHEMA",
    "MOVEMENT_STATE_VERSION",
    "MovementStateEvidence",
    "NpcWalkMovementState",
    "NpcWalkMovementStateResult",
    "empty_npc_walk_movement_state",
    "movement_state_contract",
    "movement_state_contract_sha256",
    "native_movement_state_evidence",
    "npc_walk_movement_state_result",
]
