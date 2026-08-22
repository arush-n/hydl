"""Recurrent policy-bank inference for variable actor assignments."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
)
from hytalegym.jax.training.policy import apply_policy
from hytalegym.jax.training.types import RecurrentPolicyParams

from .assignment import PolicyActorAssignment


class PolicyActorInference(NamedTuple):
    """One recurrent inference result per actor slot ``[B,P]``."""

    recurrent_state: jax.Array
    logits: jax.Array
    value: jax.Array
    valid: jax.Array


def _deterministic_factored_mask(action_mask: jax.Array) -> jax.Array:
    offset = 0
    valid = jnp.ones(action_mask.shape[:2], dtype=jnp.bool_)
    for size in ARSENAL_POLICY_ACTION_HEAD_SIZES:
        valid &= jnp.sum(action_mask[..., offset : offset + size], axis=-1) == 1
        offset += size
    if action_mask.shape[-1] != offset:
        raise ValueError("action mask width differs from the Arsenal factor layout")
    return valid


def apply_assigned_policy_bank(
    policy_bank: RecurrentPolicyParams,
    observation: jax.Array,
    recurrent_state: jax.Array,
    action_mask: jax.Array,
    assignment: PolicyActorAssignment,
) -> PolicyActorInference:
    """Apply each actor slot's reset-pinned ``policy_id`` under one JIT.

    Every leaf in ``policy_bank`` has a leading policy-catalog axis ``K``.
    Actor slots select from that bank independently, so policies may be shared
    across actors or mixed within an arena. Inactive slots and out-of-bank IDs
    fail closed without reading a different policy's carry or logits.
    """

    actor_shape = assignment.actor_index.shape
    observations = jnp.asarray(observation, dtype=jnp.float32)
    carries = jnp.asarray(recurrent_state, dtype=jnp.float32)
    masks = jnp.asarray(action_mask, dtype=jnp.bool_)
    if observations.ndim != 3 or observations.shape[:2] != actor_shape:
        raise ValueError("observation must have shape [B,P,O]")
    if carries.ndim != 3 or carries.shape[:2] != actor_shape:
        raise ValueError("recurrent_state must have shape [B,P,R]")
    if masks.ndim != 3 or masks.shape[:2] != actor_shape:
        raise ValueError("action_mask must have shape [B,P,A]")

    leaves = jax.tree_util.tree_leaves(policy_bank)
    if not leaves:
        raise ValueError("policy_bank cannot be empty")
    policy_count = leaves[0].shape[0]
    if policy_count < 1 or any(
        leaf.ndim < 1 or leaf.shape[0] != policy_count for leaf in leaves
    ):
        raise ValueError("every policy-bank leaf must share a positive K axis")
    if observations.shape[2] != policy_bank.encoder_input.kernel.shape[1]:
        raise ValueError("observation width does not match the policy bank")
    if carries.shape[2] != policy_bank.gru.recurrent_kernel.shape[1]:
        raise ValueError("recurrent width does not match the policy bank")
    if masks.shape[2] != policy_bank.actor.bias.shape[1]:
        raise ValueError("action-mask width does not match the policy bank")

    policy_id = assignment.policy_id
    valid = assignment.active & (policy_id >= 0) & (policy_id < policy_count)
    if actor_shape[1] == 1:
        # A P=1 compatibility batch represents one shared shipped policy.
        # Mixed policy IDs must be grouped into separate calls or represented
        # with P>1; fail them closed instead of giving up exact batched math.
        valid &= policy_id == policy_id[0, 0]
    safe_policy_id = jnp.clip(policy_id, 0, policy_count - 1)
    selected = jax.tree_util.tree_map(
        lambda leaf: leaf[safe_policy_id],
        policy_bank,
    )
    flat_count = actor_shape[0] * actor_shape[1]
    flat_params = jax.tree_util.tree_map(
        lambda leaf: leaf.reshape((flat_count,) + leaf.shape[2:]),
        selected,
    )
    flat_observation = observations.reshape((flat_count, observations.shape[2]))
    flat_carry = carries.reshape((flat_count, carries.shape[2]))
    flat_mask = masks.reshape((flat_count, masks.shape[2]))

    def apply_one(params, actor_observation, actor_carry, actor_mask):
        next_carry, logits, value = apply_policy(
            params,
            actor_observation[None, :],
            actor_carry[None, :],
            actor_mask[None, :],
        )
        return next_carry[0], logits[0], value[0]

    def heterogeneous_dispatch(_):
        next_carry, logits, value = jax.vmap(apply_one)(
            flat_params,
            flat_observation,
            flat_carry,
            flat_mask,
        )
        return (
            next_carry.reshape(actor_shape + (carries.shape[2],)),
            logits.reshape(actor_shape + (masks.shape[2],)),
            value.reshape(actor_shape),
        )

    if actor_shape[1] == 1:
        # Preserve the shipped policy's exact batched matrix operations for
        # the legacy homogeneous P=1 adapter. A row-wise vmap is numerically
        # equivalent but can change float32 accumulation by one ULP.
        def homogeneous_single_policy(_):
            params = jax.tree_util.tree_map(
                lambda leaf: leaf[safe_policy_id[0, 0]],
                policy_bank,
            )
            next_carry, logits, value = apply_policy(
                params,
                observations[:, 0],
                carries[:, 0],
                masks[:, 0],
            )
            return (
                next_carry[:, None, :],
                logits[:, None, :],
                value[:, None],
            )

        next_carry, logits, value = homogeneous_single_policy(None)
    else:
        next_carry, logits, value = heterogeneous_dispatch(None)
    return PolicyActorInference(
        recurrent_state=jnp.where(valid[..., None], next_carry, carries),
        logits=jnp.where(valid[..., None], logits, jnp.float32(-1.0e9)),
        value=jnp.where(valid, value, jnp.float32(0.0)),
        valid=valid,
    )


def apply_assigned_actor_controllers(
    policy_bank: RecurrentPolicyParams,
    observation: jax.Array,
    recurrent_state: jax.Array,
    action_mask: jax.Array,
    assignment: PolicyActorAssignment,
    *,
    scripted_actor_slots: tuple[int, ...] = (),
) -> PolicyActorInference:
    """Run neural policies only for learned slots; scripts use forced masks.

    A scripted slot must expose exactly one legal choice for every factored
    action head. Its logits are policy-independent and its recurrent carry is
    preserved. The static slot tuple lets XLA omit those policy forward passes.
    """

    slots = tuple(scripted_actor_slots)
    policy_slots = assignment.actor_index.shape[1]
    if slots != tuple(sorted(set(slots))) or any(
        isinstance(slot, bool)
        or not isinstance(slot, int)
        or slot < 0
        or slot >= policy_slots
        for slot in slots
    ):
        raise ValueError("scripted actor slots must be sorted unique policy slots")
    if not slots:
        return apply_assigned_policy_bank(
            policy_bank, observation, recurrent_state, action_mask, assignment
        )

    mask = jnp.asarray(action_mask, dtype=jnp.bool_)
    carries = jnp.asarray(recurrent_state, dtype=jnp.float32)
    logits = jnp.where(mask, jnp.float32(0.0), jnp.float32(-1.0e9))
    values = jnp.zeros(assignment.active.shape, dtype=jnp.float32)
    valid = jnp.zeros(assignment.active.shape, dtype=jnp.bool_)
    next_carry = carries
    scripted_supported = _deterministic_factored_mask(mask)
    for slot in slots:
        valid = valid.at[:, slot].set(
            assignment.active[:, slot] & scripted_supported[:, slot]
        )

    learned_slots = tuple(slot for slot in range(policy_slots) if slot not in slots)
    if learned_slots:
        learned_assignment = PolicyActorAssignment(
            *(value[:, learned_slots] for value in assignment)
        )
        learned = apply_assigned_policy_bank(
            policy_bank,
            observation[:, learned_slots],
            carries[:, learned_slots],
            mask[:, learned_slots],
            learned_assignment,
        )
        for local, slot in enumerate(learned_slots):
            next_carry = next_carry.at[:, slot].set(learned.recurrent_state[:, local])
            logits = logits.at[:, slot].set(learned.logits[:, local])
            values = values.at[:, slot].set(learned.value[:, local])
            valid = valid.at[:, slot].set(learned.valid[:, local])
    return PolicyActorInference(next_carry, logits, values, valid)


__all__ = [
    "PolicyActorInference",
    "apply_assigned_actor_controllers",
    "apply_assigned_policy_bank",
]
