"""Lossless packed storage for entity-indexed Walk-controller memory.

The arithmetic kernel uses the named :class:`EntityLocomotionState` view, but
the persistent arena stores three dtype-homogeneous banks.  This keeps the
public semantics and exact dtypes while avoiding 21 independent GPU outputs
at every environment step.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.world import MOVEMENT_STATE_COUNT, NpcWalkMovementState

from ..locomotion import EntityLocomotionState


_FLOAT_WIDTH = 17
_BOOL_WIDTH = 7 + 2 * MOVEMENT_STATE_COUNT
_INT_WIDTH = 2


class PackedEntityLocomotionState(NamedTuple):
    """Three lossless banks with leading shape ``[B,N]``."""

    float32: jax.Array
    boolean: jax.Array
    int32: jax.Array


def pack_entity_locomotion_state(
    state: EntityLocomotionState,
) -> PackedEntityLocomotionState:
    """Pack a named locomotion view without changing any value."""

    float32 = jnp.concatenate(
        (
            state.desired_yaw[..., None],
            state.pitch[..., None],
            state.desired_pitch[..., None],
            state.desired_velocity,
            state.move_speed[..., None],
            state.fall_speed[..., None],
            state.fall_start_y[..., None],
            state.applied_vertical_velocity[..., None],
            state.force_velocity,
            state.walk.previous_speed[..., None],
            state.stamina[..., None],
            state.stamina_regen_delay[..., None],
            # Appended, not inserted: every index above is read back by literal
            # position below, so growing the bank at the tail is the only edit
            # that cannot silently re-map an existing channel.
            state.desired_body_yaw[..., None],
            state.head_yaw[..., None],
        ),
        axis=-1,
    ).astype(jnp.float32)
    boolean = jnp.concatenate(
        (
            state.vertical_impulse_applied[..., None],
            state.grounded_with_residual_velocity[..., None],
            state.knockback_control_lock[..., None],
            state.grounded[..., None],
            state.force_pushed[..., None],
            state.geometry_exhausted[..., None],
            state.walk.values,
            state.walk.available,
            state.walk.fast_motion_kind[..., None],
        ),
        axis=-1,
    ).astype(jnp.bool_)
    int32 = jnp.stack(
        (
            state.ticks_since_damage.astype(jnp.int32),
            state.walk.last_motion_kind.astype(jnp.int32),
        ),
        axis=-1,
    )
    _validate_packed_shapes(float32, boolean, int32)
    return PackedEntityLocomotionState(
        float32=float32,
        boolean=boolean,
        int32=int32,
    )


def unpack_entity_locomotion_state(
    packed: PackedEntityLocomotionState,
) -> EntityLocomotionState:
    """Restore the exact named locomotion view used by the motion kernel."""

    if not isinstance(packed, PackedEntityLocomotionState):
        raise TypeError("packed must be PackedEntityLocomotionState")
    _validate_packed_shapes(packed.float32, packed.boolean, packed.int32)
    floats = packed.float32
    booleans = packed.boolean
    integers = packed.int32
    values_start = 6
    available_start = values_start + MOVEMENT_STATE_COUNT
    fast_motion_index = available_start + MOVEMENT_STATE_COUNT
    return EntityLocomotionState(
        desired_yaw=floats[..., 0],
        desired_body_yaw=floats[..., 15],
        head_yaw=floats[..., 16],
        pitch=floats[..., 1],
        desired_pitch=floats[..., 2],
        desired_velocity=floats[..., 3:5],
        move_speed=floats[..., 5],
        fall_speed=floats[..., 6],
        fall_start_y=floats[..., 7],
        vertical_impulse_applied=booleans[..., 0],
        applied_vertical_velocity=floats[..., 8],
        grounded_with_residual_velocity=booleans[..., 1],
        knockback_control_lock=booleans[..., 2],
        grounded=booleans[..., 3],
        force_velocity=floats[..., 9:12],
        force_pushed=booleans[..., 4],
        ticks_since_damage=integers[..., 0],
        geometry_exhausted=booleans[..., 5],
        stamina=floats[..., 13],
        stamina_regen_delay=floats[..., 14],
        walk=NpcWalkMovementState(
            values=booleans[
                ..., values_start : values_start + MOVEMENT_STATE_COUNT
            ],
            available=booleans[
                ..., available_start : available_start + MOVEMENT_STATE_COUNT
            ],
            previous_speed=floats[..., 12],
            fast_motion_kind=booleans[..., fast_motion_index],
            last_motion_kind=integers[..., 1].astype(jnp.int8),
        ),
    )


def _validate_packed_shapes(
    float32: jax.Array,
    boolean: jax.Array,
    int32: jax.Array,
) -> None:
    if float32.ndim != 3 or float32.shape[-1] != _FLOAT_WIDTH:
        raise ValueError(f"float32 must have shape [B,N,{_FLOAT_WIDTH}]")
    leading = float32.shape[:2]
    if boolean.shape != leading + (_BOOL_WIDTH,):
        raise ValueError(f"boolean must have shape [B,N,{_BOOL_WIDTH}]")
    if int32.shape != leading + (_INT_WIDTH,):
        raise ValueError(f"int32 must have shape [B,N,{_INT_WIDTH}]")
    if float32.dtype != jnp.dtype(jnp.float32):
        raise TypeError("float32 bank must use float32")
    if boolean.dtype != jnp.dtype(jnp.bool_):
        raise TypeError("boolean bank must use bool")
    if int32.dtype != jnp.dtype(jnp.int32):
        raise TypeError("int32 bank must use int32")


__all__ = [
    "PackedEntityLocomotionState",
    "pack_entity_locomotion_state",
    "unpack_entity_locomotion_state",
]
