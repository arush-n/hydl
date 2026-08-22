"""Variable policy-to-entity ownership for shared combat arenas."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class PolicyActorAssignment(NamedTuple):
    """Fixed-shape policy slots mapped onto a variable entity axis."""

    actor_index: jax.Array
    policy_id: jax.Array
    active: jax.Array
    trainable: jax.Array


def policy_actor_assignment(
    actor_index,
    policy_id,
    *,
    active=None,
    trainable=None,
    entity_count: int,
) -> PolicyActorAssignment:
    """Build and host-validate one reset-pinned actor assignment."""

    actors = jnp.asarray(actor_index, dtype=jnp.int32)
    policies = jnp.asarray(policy_id, dtype=jnp.int32)
    if active is None:
        active = actors >= 0
    active_rows = jnp.asarray(active, dtype=jnp.bool_)
    if trainable is None:
        trainable = active_rows
    trainable_rows = jnp.asarray(trainable, dtype=jnp.bool_)
    assignment = PolicyActorAssignment(
        actor_index=jnp.where(active_rows, actors, jnp.int32(-1)),
        policy_id=jnp.where(active_rows, policies, jnp.int32(-1)),
        active=active_rows,
        trainable=trainable_rows,
    )
    validate_policy_actor_assignment(assignment, entity_count=entity_count)
    return assignment


def single_policy_actor_assignment(
    batch_size: int,
    *,
    actor_index: int = 0,
    policy_id: int = 0,
    trainable: bool = True,
    entity_count: int,
) -> PolicyActorAssignment:
    """Return the exact legacy ownership layout as the P=1 special case."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return policy_actor_assignment(
        jnp.full((batch_size, 1), actor_index, dtype=jnp.int32),
        jnp.full((batch_size, 1), policy_id, dtype=jnp.int32),
        trainable=jnp.full((batch_size, 1), trainable, dtype=jnp.bool_),
        entity_count=entity_count,
    )


def validate_policy_actor_assignment(
    assignment: PolicyActorAssignment,
    *,
    entity_count: int,
) -> None:
    """Reject ambiguous ownership before the assignment enters a JIT."""

    if not isinstance(assignment, PolicyActorAssignment):
        raise TypeError("assignment must be PolicyActorAssignment")
    if isinstance(entity_count, bool) or not isinstance(entity_count, int):
        raise TypeError("entity_count must be an integer")
    if entity_count < 1:
        raise ValueError("entity_count must be positive")
    expected_shape = assignment.actor_index.shape
    if len(expected_shape) != 2 or expected_shape[0] < 1 or expected_shape[1] < 1:
        raise ValueError("policy actor assignment must have shape [B,P]")
    expected = (
        ("actor_index", assignment.actor_index, jnp.dtype(jnp.int32)),
        ("policy_id", assignment.policy_id, jnp.dtype(jnp.int32)),
        ("active", assignment.active, jnp.dtype(jnp.bool_)),
        ("trainable", assignment.trainable, jnp.dtype(jnp.bool_)),
    )
    for name, value, dtype in expected:
        if value.shape != expected_shape or value.dtype != dtype:
            raise ValueError(f"{name} must be {dtype}[B,P]")

    actors = np.asarray(jax.device_get(assignment.actor_index))
    policies = np.asarray(jax.device_get(assignment.policy_id))
    active = np.asarray(jax.device_get(assignment.active))
    trainable = np.asarray(jax.device_get(assignment.trainable))
    if np.any(trainable & ~active):
        raise ValueError("trainable policy actor slots must be active")
    if np.any(active & ((actors < 0) | (actors >= entity_count))):
        raise ValueError("active actor_index is outside the entity axis")
    if np.any(~active & (actors != -1)):
        raise ValueError("inactive actor_index must be -1")
    if np.any(active & (policies < 0)):
        raise ValueError("active policy_id must be non-negative")
    if np.any(~active & (policies != -1)):
        raise ValueError("inactive policy_id must be -1")
    for row_actors, row_active in zip(actors, active, strict=True):
        owned = row_actors[row_active]
        if owned.size != np.unique(owned).size:
            raise ValueError("an entity cannot be owned by two policy slots")


def policy_controlled_entity_mask(
    assignment: PolicyActorAssignment,
    *,
    entity_count: int,
) -> jax.Array:
    """Return bool[B,N] without assuming two entities or one policy."""

    safe_actor = jnp.clip(assignment.actor_index, 0, entity_count - 1)
    entity_axis = jnp.arange(entity_count, dtype=jnp.int32)[None, None, :]
    return jnp.any(
        assignment.active[..., None] & (safe_actor[..., None] == entity_axis),
        axis=1,
    )


def gather_policy_actor_rows(
    entity_rows: jax.Array,
    assignment: PolicyActorAssignment,
    *,
    fill_value=0,
) -> jax.Array:
    """Gather entity-shaped evidence into actor slots, closing inactive rows."""

    values = jnp.asarray(entity_rows)
    if values.ndim < 2:
        raise ValueError("entity_rows must have shape [B,N,...]")
    if values.shape[0] != assignment.actor_index.shape[0]:
        raise ValueError("entity rows and assignment batch differ")
    safe_actor = jnp.clip(assignment.actor_index, 0, values.shape[1] - 1)
    batch = jnp.arange(values.shape[0], dtype=jnp.int32)[:, None]
    gathered = values[batch, safe_actor]
    active = assignment.active.reshape(
        assignment.active.shape + (1,) * (values.ndim - 2)
    )
    return jnp.where(active, gathered, jnp.asarray(fill_value, dtype=values.dtype))


def scatter_policy_actor_rows(
    actor_rows: jax.Array,
    assignment: PolicyActorAssignment,
    *,
    entity_count: int,
    fill_value=0,
) -> jax.Array:
    """Scatter uniquely owned actor rows back onto the entity axis."""

    values = jnp.asarray(actor_rows)
    if values.ndim < 2 or values.shape[:2] != assignment.actor_index.shape:
        raise ValueError("actor_rows must have shape [B,P,...]")
    output = jnp.full(
        (values.shape[0], entity_count) + values.shape[2:],
        fill_value,
        dtype=values.dtype,
    )
    batch = jnp.arange(values.shape[0], dtype=jnp.int32)
    for slot in range(values.shape[1]):
        actor = jnp.clip(assignment.actor_index[:, slot], 0, entity_count - 1)
        current = output[batch, actor]
        active = assignment.active[:, slot].reshape(
            (values.shape[0],) + (1,) * (values.ndim - 2)
        )
        update = jnp.where(active, values[:, slot], current)
        output = output.at[batch, actor].set(update)
    return output


__all__ = [
    "PolicyActorAssignment",
    "gather_policy_actor_rows",
    "policy_actor_assignment",
    "policy_controlled_entity_mask",
    "scatter_policy_actor_rows",
    "single_policy_actor_assignment",
    "validate_policy_actor_assignment",
]
