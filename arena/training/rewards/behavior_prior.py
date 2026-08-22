"""Online recurrent behavior-prior rewards for JAX PPO environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from arena.jax_contract import HEAD_SPANS
from arena.training.rewards.exact_imitation import (
    ExactImitationConfig,
    exact_imitation_reward,
)
from hytalegym.jax.combat.observation.v3.policy import (
    greedy_arsenal_action_factors,
)
from hytalegym.jax.training.policy import apply_policy
from hytalegym.jax.training.ppo import PPOEnvironment
from hytalegym.jax.training.types import PPOConfig


class BehaviorPriorState(NamedTuple):
    environment: Any
    recurrent_state: jax.Array
    action_mask: jax.Array


@dataclass(frozen=True, slots=True)
class BehaviorPriorConfig:
    """Declare exactly which online expert heads may shape PPO reward."""

    supervised_heads: tuple[str, ...]
    imitation: ExactImitationConfig = ExactImitationConfig()
    lock_unsupervised_heads: bool = False

    def __post_init__(self) -> None:
        if (
            not self.supervised_heads
            or len(set(self.supervised_heads)) != len(self.supervised_heads)
            or not set(self.supervised_heads) <= set(HEAD_SPANS)
        ):
            raise ValueError("supervised_heads must be a unique nonempty policy subset")
        if not isinstance(self.imitation, ExactImitationConfig):
            raise TypeError("imitation must be ExactImitationConfig")
        if not isinstance(self.lock_unsupervised_heads, bool):
            raise TypeError("lock_unsupervised_heads must be a bool")


def make_behavior_prior_environment(
    environment: PPOEnvironment,
    prior_params: Any,
    prior_policy: PPOConfig,
    observation_projector: Callable[[jax.Array], jax.Array],
    config: BehaviorPriorConfig,
) -> PPOEnvironment:
    """Add an online, current-state imitation penalty to an environment.

    The fixed prior is queried on every actual policy state, so alignment is
    valid even after the learner diverges from the offline demonstrations.
    Unknown heads abstain completely.
    """

    if not isinstance(environment, PPOEnvironment):
        raise TypeError("environment must be PPOEnvironment")
    if not isinstance(prior_policy, PPOConfig):
        raise TypeError("prior_policy must be PPOConfig")
    if not isinstance(config, BehaviorPriorConfig):
        raise TypeError("config must be BehaviorPriorConfig")
    if environment.spec.observation_size is None:
        raise ValueError("behavior priors require a dense observation surface")
    head_sizes = tuple(size for _, size in HEAD_SPANS.values())
    if tuple(prior_policy.action_head_sizes) != head_sizes:
        raise ValueError("behavior prior action ABI differs from Arena")
    probe = observation_projector(
        jnp.zeros((1, environment.spec.observation_size), dtype=jnp.float32)
    )
    if probe.shape != (1, prior_policy.observation_size):
        raise ValueError("behavior prior observation projector shape differs")

    supervised = jnp.asarray(
        [name in config.supervised_heads for name in HEAD_SPANS],
        dtype=jnp.bool_,
    )
    action_scope = jnp.concatenate(
        tuple(
            jnp.ones((size,), dtype=jnp.bool_)
            if name in config.supervised_heads
            else jnp.arange(size) == 0
            for name, (_, size) in HEAD_SPANS.items()
        )
    )

    def scoped(mask):
        return mask & action_scope if config.lock_unsupervised_heads else mask

    def reset(keys):
        state, observation, mask = environment.reset(keys)
        mask = scoped(mask)
        carry = jnp.zeros(
            (observation.shape[0], prior_policy.recurrent_size),
            dtype=jnp.float32,
        )
        return BehaviorPriorState(state, carry, mask), observation, mask

    def advance(state, observation, action, keys, *, detailed):
        next_carry, logits, _ = apply_policy(
            prior_params,
            observation_projector(observation),
            state.recurrent_state,
            state.action_mask,
        )
        expert = greedy_arsenal_action_factors(logits)
        terms = exact_imitation_reward(
            action,
            expert,
            jnp.broadcast_to(supervised, action.shape),
            jnp.ones(action.shape[:-1], dtype=jnp.bool_),
            config.imitation,
        )
        step = environment.step_detailed if detailed else environment.step
        result = step(state.environment, observation, action, keys)
        next_state, next_observation, reward, done, mask, *tail = result
        mask = scoped(mask)
        return (
            BehaviorPriorState(next_state, next_carry, mask),
            next_observation,
            reward + terms.reward,
            done,
            mask,
            *tail,
        )

    def step(state, observation, action, keys):
        return advance(state, observation, action, keys, detailed=False)

    step_detailed = None
    if environment.step_detailed is not None:

        def step_detailed(state, observation, action, keys):
            return advance(state, observation, action, keys, detailed=True)

    episode_outcome = None
    if environment.episode_outcome is not None:

        def episode_outcome(state, done):
            return environment.episode_outcome(state.environment, done)

    return PPOEnvironment(
        reset=reset,
        step=step,
        spec=environment.spec,
        episode_outcome=episode_outcome,
        step_detailed=step_detailed,
    )


__all__ = [
    "BehaviorPriorConfig",
    "BehaviorPriorState",
    "make_behavior_prior_environment",
]
