"""Exactly two independently trainable policies in one shared JAX arena."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from hytalegym.jax.training.multi_actor import (
    PolicyActorAssignment,
    make_multi_actor_population_trainer,
    policy_actor_assignment,
)
from hytalegym.jax.training.types import PPOConfig


def duel_assignment(
    batch_size: int,
    *,
    policy_ids: Sequence[int] = (0, 1),
    entity_count: int = 2,
) -> PolicyActorAssignment:
    """Seat two distinct live policies on entities 0 and 1 in every arena."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    ids = tuple(policy_ids)
    if len(ids) != 2:
        raise ValueError("a duel requires exactly two policy IDs")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in ids):
        raise TypeError("duel policy IDs must be integers")
    if min(ids) < 0 or ids[0] == ids[1]:
        raise ValueError("duel policy IDs must be distinct and non-negative")
    actors = jnp.broadcast_to(
        jnp.asarray((0, 1), dtype=jnp.int32), (batch_size, 2)
    )
    policies = jnp.broadcast_to(jnp.asarray(ids, dtype=jnp.int32), (batch_size, 2))
    return policy_actor_assignment(
        actors,
        policies,
        trainable=jnp.ones((batch_size, 2), dtype=jnp.bool_),
        entity_count=entity_count,
    )


def make_duel_ppo_trainer(
    collector: Callable,
    assignment: PolicyActorAssignment,
    config: PPOConfig,
    *,
    compile: bool = True,
) -> Any:
    """Give both duel policies independent PPO optimizers over one rollout."""

    active = np.asarray(jax.device_get(assignment.active), dtype=np.bool_)
    trainable = np.asarray(jax.device_get(assignment.trainable), dtype=np.bool_)
    actors = np.asarray(jax.device_get(assignment.actor_index), dtype=np.int32)
    policies = np.asarray(jax.device_get(assignment.policy_id), dtype=np.int32)
    if active.ndim != 2 or active.shape[1] != 2:
        raise ValueError("duel assignment must have shape [B,2]")
    if not np.all(active & trainable):
        raise ValueError("both duel agents must be active and trainable")
    if np.any(actors[:, 0] == actors[:, 1]) or np.any(
        policies[:, 0] == policies[:, 1]
    ):
        raise ValueError("duel agents must own distinct entities and policies")
    return make_multi_actor_population_trainer(
        collector, assignment, config, compile=compile
    )


__all__ = ["duel_assignment", "make_duel_ppo_trainer"]
