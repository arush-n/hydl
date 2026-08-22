"""Transactional JAX updates for existing surrogate entity instances."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.surrogate.capabilities import (
    SURROGATE_FAILURE_ENTITY_UPDATE,
)
from hytalegym.jax.world.surrogate.types import (
    SurrogateEntityUpdateBatch,
    SurrogateEntityUpdateResult,
    SurrogateRuntimeState,
)

SURROGATE_ENTITY_UPDATE_VERSION = 1
SURROGATE_ENTITY_UPDATE_CAPACITY = 64
MAX_SURROGATE_ENTITY_UPDATE_CAPACITY = 256

SURROGATE_ENTITY_UPDATE_INVALID_SLOT = 1 << 0
SURROGATE_ENTITY_UPDATE_UNINITIALIZED = 1 << 1
SURROGATE_ENTITY_UPDATE_IDENTITY_MISMATCH = 1 << 2
SURROGATE_ENTITY_UPDATE_NONFINITE = 1 << 3
SURROGATE_ENTITY_UPDATE_DUPLICATE_SLOT = 1 << 4
SURROGATE_ENTITY_UPDATE_RUNTIME = 1 << 5


def surrogate_entity_update_contract() -> dict[str, object]:
    """Return the stable host manifest for entity-state publication."""

    return {
        "version": SURROGATE_ENTITY_UPDATE_VERSION,
        "default_capacity": SURROGATE_ENTITY_UPDATE_CAPACITY,
        "maximum_capacity": MAX_SURROGATE_ENTITY_UPDATE_CAPACITY,
        "transaction_scope": "per_environment",
        "target_scope": "initialized_instances_only",
        "identity_guard": "world_absolute_instance_identity",
        "mutable_fields": ("active", "position", "rotation", "velocity"),
        "allocation_supported": False,
        "type_or_geometry_updates_supported": False,
        "behavior_supported": False,
    }


def apply_surrogate_entity_updates(
    runtime: SurrogateRuntimeState,
    updates: SurrogateEntityUpdateBatch,
) -> SurrogateEntityUpdateResult:
    """Publish fixed-shape entity state atomically for each environment.

    A rejected row preserves every entity leaf, records a runtime failure, and
    disables downstream world queries for that row. Entity allocation, identity,
    type, geometry, and behavior changes are deliberately unavailable.
    """

    batch, entity_capacity = _validate_runtime(runtime)
    if not isinstance(updates, SurrogateEntityUpdateBatch):
        raise TypeError("updates must be a SurrogateEntityUpdateBatch")
    if (
        getattr(updates.mask, "ndim", None) != 2
        or updates.mask.shape[0] != batch
    ):
        raise ValueError("entity update mask must have shape [batch, updates]")
    update_capacity = updates.mask.shape[1]
    if not 1 <= update_capacity <= MAX_SURROGATE_ENTITY_UPDATE_CAPACITY:
        raise ValueError(
            "entity update capacity must be in "
            f"[1, {MAX_SURROGATE_ENTITY_UPDATE_CAPACITY}]"
        )

    shape = (batch, update_capacity)
    mask = _field(updates.mask, np.bool_, shape, "entity update mask")
    slots = _field(updates.runtime_slot, np.int32, shape, "entity runtime slot")
    expected_identity = _field(
        updates.expected_identity_words,
        np.uint32,
        shape + (2,),
        "expected entity identity",
    )
    active = _field(updates.active, np.bool_, shape, "entity active state")
    position = _field(
        updates.position,
        np.float32,
        shape + (3,),
        "entity position",
    )
    rotation = _field(
        updates.rotation,
        np.float32,
        shape + (3,),
        "entity rotation",
    )
    velocity = _field(
        updates.velocity,
        np.float32,
        shape + (3,),
        "entity velocity",
    )

    safe_slots = jnp.clip(slots, 0, entity_capacity - 1)
    batch_index = jnp.arange(batch, dtype=jnp.int32)[:, None]
    initialized = runtime.entity_initialized[batch_index, safe_slots]
    current_identity = runtime.entity_identity_words[batch_index, safe_slots]
    current_position = runtime.entity_position[batch_index, safe_slots]
    current_rotation = runtime.entity_rotation[batch_index, safe_slots]
    current_velocity = runtime.entity_velocity[batch_index, safe_slots]
    current_active = runtime.entity_active[batch_index, safe_slots]

    valid_slot = (slots >= 0) & (slots < entity_capacity)
    identity_match = (
        jnp.any(expected_identity != 0, axis=2)
        & jnp.all(expected_identity == current_identity, axis=2)
    )
    proposed_finite = (
        jnp.all(jnp.isfinite(position), axis=2)
        & jnp.all(jnp.isfinite(rotation), axis=2)
        & jnp.all(jnp.isfinite(velocity), axis=2)
    )
    current_finite = (
        jnp.all(jnp.isfinite(current_position), axis=2)
        & jnp.all(jnp.isfinite(current_rotation), axis=2)
        & jnp.all(jnp.isfinite(current_velocity), axis=2)
    )

    request_diagnostics = jnp.zeros(shape, dtype=jnp.uint32)
    request_diagnostics |= jnp.where(
        mask & ~valid_slot,
        SURROGATE_ENTITY_UPDATE_INVALID_SLOT,
        0,
    ).astype(jnp.uint32)
    request_diagnostics |= jnp.where(
        mask & valid_slot & ~initialized,
        SURROGATE_ENTITY_UPDATE_UNINITIALIZED,
        0,
    ).astype(jnp.uint32)
    request_diagnostics |= jnp.where(
        mask & valid_slot & initialized & ~identity_match,
        SURROGATE_ENTITY_UPDATE_IDENTITY_MISMATCH,
        0,
    ).astype(jnp.uint32)
    request_diagnostics |= jnp.where(
        mask & ~proposed_finite,
        SURROGATE_ENTITY_UPDATE_NONFINITE,
        0,
    ).astype(jnp.uint32)
    request_diagnostics |= jnp.where(
        mask & valid_slot & initialized & ~current_finite,
        SURROGATE_ENTITY_UPDATE_RUNTIME,
        0,
    ).astype(jnp.uint32)

    sentinel = jnp.iinfo(jnp.int32).max
    ordered_slots = jnp.sort(jnp.where(mask, slots, sentinel), axis=1)
    duplicate_slot = jnp.any(
        (ordered_slots[:, 1:] == ordered_slots[:, :-1])
        & (ordered_slots[:, 1:] != sentinel),
        axis=1,
    )
    requested = jnp.any(mask, axis=1)
    runtime_ready = (
        ~runtime.unsupported_mechanics
        & (runtime.failure_bits == 0)
    )
    diagnostics = jnp.bitwise_or.reduce(request_diagnostics, axis=1)
    diagnostics |= jnp.where(
        requested & duplicate_slot,
        SURROGATE_ENTITY_UPDATE_DUPLICATE_SLOT,
        0,
    ).astype(jnp.uint32)
    diagnostics |= jnp.where(
        requested & ~runtime_ready,
        SURROGATE_ENTITY_UPDATE_RUNTIME,
        0,
    ).astype(jnp.uint32)
    accepted = ~requested | (diagnostics == 0)
    applied = mask & accepted[:, None]

    changed = applied & (
        (active != current_active)
        | jnp.any(position != current_position, axis=2)
        | jnp.any(rotation != current_rotation, axis=2)
        | jnp.any(velocity != current_velocity, axis=2)
    )
    entity_active = _scatter_value(
        runtime.entity_active,
        batch_index,
        safe_slots,
        active,
        applied,
    )
    entity_position = _scatter_value(
        runtime.entity_position,
        batch_index,
        safe_slots,
        position,
        applied,
    )
    entity_rotation = _scatter_value(
        runtime.entity_rotation,
        batch_index,
        safe_slots,
        rotation,
        applied,
    )
    entity_velocity = _scatter_value(
        runtime.entity_velocity,
        batch_index,
        safe_slots,
        velocity,
        applied,
    )

    rejected = requested & ~accepted
    failure_bits = runtime.failure_bits | jnp.where(
        rejected,
        SURROGATE_FAILURE_ENTITY_UPDATE,
        0,
    ).astype(jnp.uint32)
    updated_runtime = runtime._replace(
        entity_active=entity_active,
        entity_position=entity_position,
        entity_rotation=entity_rotation,
        entity_velocity=entity_velocity,
        failure_bits=failure_bits,
        unsupported_mechanics=runtime.unsupported_mechanics | rejected,
    )
    return SurrogateEntityUpdateResult(
        runtime=updated_runtime,
        accepted=accepted,
        applied=applied,
        state_changed=changed,
        diagnostics=diagnostics,
        request_diagnostics=request_diagnostics,
    )


def _scatter_value(
    values: jax.Array,
    batch_index: jax.Array,
    slots: jax.Array,
    proposed: jax.Array,
    applied: jax.Array,
) -> jax.Array:
    inactive_slot = values.shape[1]
    padding = jnp.zeros(
        (values.shape[0], 1) + values.shape[2:],
        dtype=values.dtype,
    )
    padded = jnp.concatenate((values, padding), axis=1)
    target_slots = jnp.where(applied, slots, inactive_slot)
    value_mask = applied.reshape(
        applied.shape + (1,) * (proposed.ndim - applied.ndim)
    )
    safe_proposed = jnp.where(value_mask, proposed, jnp.zeros_like(proposed))
    return padded.at[batch_index, target_slots].set(safe_proposed)[:, :inactive_slot]


def _field(
    value: jax.Array,
    dtype: np.dtype,
    shape: tuple[int, ...],
    label: str,
) -> jax.Array:
    source_dtype = getattr(value, "dtype", None)
    if source_dtype is None or np.dtype(source_dtype) != np.dtype(dtype):
        raise TypeError(f"{label} must use dtype {np.dtype(dtype)}")
    result = jnp.asarray(value)
    if result.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    return result


def _validate_runtime(runtime: SurrogateRuntimeState) -> tuple[int, int]:
    if not isinstance(runtime, SurrogateRuntimeState):
        raise TypeError("runtime must be a SurrogateRuntimeState")
    if (
        getattr(runtime.environment_world_id, "ndim", None) != 1
        or getattr(runtime.stateful_state, "ndim", None) != 2
        or getattr(runtime.entity_active, "ndim", None) != 2
    ):
        raise ValueError("surrogate runtime entity layout is invalid")
    batch = runtime.environment_world_id.shape[0]
    entity_capacity = runtime.entity_active.shape[1]
    if batch == 0 or entity_capacity == 0:
        raise ValueError("surrogate runtime entity layout must be non-empty")
    state_capacity = runtime.stateful_state.shape[1]
    fields = (
        (runtime.environment_world_id, np.int32, (batch,)),
        (runtime.stateful_state, np.uint8, (batch, state_capacity)),
        (runtime.stateful_initialized, np.bool_, (batch, state_capacity)),
        (runtime.entity_active, np.bool_, (batch, entity_capacity)),
        (runtime.entity_initialized, np.bool_, (batch, entity_capacity)),
        (runtime.entity_identity_words, np.uint32, (batch, entity_capacity, 2)),
        (runtime.entity_position, np.float32, (batch, entity_capacity, 3)),
        (runtime.entity_rotation, np.float32, (batch, entity_capacity, 3)),
        (runtime.entity_velocity, np.float32, (batch, entity_capacity, 3)),
        (runtime.entity_kind, np.uint8, (batch, entity_capacity)),
        (runtime.entity_type_code, np.uint16, (batch, entity_capacity)),
        (
            runtime.entity_type_identity_words,
            np.uint32,
            (batch, entity_capacity, 2),
        ),
        (runtime.entity_behavior_supported, np.bool_, (batch, entity_capacity)),
        (runtime.entity_geometry_supported, np.bool_, (batch, entity_capacity)),
        (runtime.entity_collidable, np.bool_, (batch, entity_capacity)),
        (runtime.entity_blocks_los, np.bool_, (batch, entity_capacity)),
        (runtime.entity_local_bounds, np.float32, (batch, entity_capacity, 6)),
        (
            runtime.entity_los_offset_supported,
            np.bool_,
            (batch, entity_capacity),
        ),
        (runtime.entity_los_offset, np.float32, (batch, entity_capacity, 3)),
        (runtime.capability_bits, np.uint32, (batch,)),
        (runtime.failure_bits, np.uint32, (batch,)),
        (runtime.unsupported_mechanics, np.bool_, (batch,)),
    )
    if any(
        getattr(value, "shape", None) != shape
        or getattr(value, "dtype", None) is None
        or np.dtype(value.dtype) != np.dtype(dtype)
        for value, dtype, shape in fields
    ):
        raise ValueError("surrogate runtime shapes or dtypes are invalid")
    return batch, entity_capacity


__all__ = [
    "MAX_SURROGATE_ENTITY_UPDATE_CAPACITY",
    "SURROGATE_ENTITY_UPDATE_CAPACITY",
    "SURROGATE_ENTITY_UPDATE_DUPLICATE_SLOT",
    "SURROGATE_ENTITY_UPDATE_IDENTITY_MISMATCH",
    "SURROGATE_ENTITY_UPDATE_INVALID_SLOT",
    "SURROGATE_ENTITY_UPDATE_NONFINITE",
    "SURROGATE_ENTITY_UPDATE_RUNTIME",
    "SURROGATE_ENTITY_UPDATE_UNINITIALIZED",
    "SURROGATE_ENTITY_UPDATE_VERSION",
    "apply_surrogate_entity_updates",
    "surrogate_entity_update_contract",
]
