"""Persistent control-arm rollouts for frozen Arsenal outcome probes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat import neutral_arsenal_policy_action_factors
from hytalegym.jax.training.policy import apply_policy, sample_actions
from hytalegym.jax.training.ppo import PPOEnvironment
from hytalegym.jax.training.types import (
    EpisodeOutcome,
    PPOConfig,
    PPOTrainState,
)


FrozenOutcomeArm = Literal[
    "stochastic_policy",
    "uniform_legal",
    "always_idle",
]


class FrozenOutcomeRollout(NamedTuple):
    """Time-major fields needed to audit terminal outcome controls."""

    action_mask: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    completed_episode_return: jax.Array
    completed_episode_length: jax.Array
    completed_episode_success: jax.Array
    completed_episode_death: jax.Array
    completed_episode_simultaneous: jax.Array
    completed_episode_other: jax.Array


FrozenOutcomeCollector = Callable[
    [PPOTrainState, jax.Array],
    tuple[PPOTrainState, FrozenOutcomeRollout],
]


def make_frozen_outcome_collector(
    config: PPOConfig,
    environment: PPOEnvironment,
    *,
    arm: FrozenOutcomeArm,
    compile: bool = True,
) -> FrozenOutcomeCollector:
    """Build one autoresetting rollout window with a replaceable action arm.

    The state, random-key schedule, recurrent reset, terminal classification,
    and autoreset order match the PPO rollout collector. Only action selection
    changes.
    """

    if arm not in ("stochastic_policy", "uniform_legal", "always_idle"):
        raise ValueError(f"unsupported frozen outcome arm: {arm!r}")
    if config.observation_size != environment.spec.observation_size:
        raise ValueError("config observation size does not match environment")
    if config.action_size != environment.spec.action_size:
        raise ValueError("config action size does not match environment")

    def collect(
        train_state: PPOTrainState,
        key: jax.Array,
    ) -> tuple[PPOTrainState, FrozenOutcomeRollout]:
        scan_keys = jax.random.split(key, config.rollout_steps)

        def rollout_step(carry, step_key):
            (
                environment_state,
                observation,
                action_mask,
                recurrent_state,
                episode_start,
                running_return,
                running_length,
            ) = carry
            action_key, environment_key, reset_key = jax.random.split(
                step_key,
                3,
            )
            recurrent_input = jnp.where(
                episode_start[:, None],
                jnp.float32(0.0),
                recurrent_state,
            )
            if arm == "stochastic_policy":
                next_recurrent_state, logits, _ = apply_policy(
                    train_state.policy_params,
                    observation,
                    recurrent_input,
                    action_mask,
                )
                action_ids, _ = sample_actions(
                    action_key,
                    logits,
                    config.action_head_sizes,
                    transport=config.action_transport,
                )
            elif arm == "uniform_legal":
                logits = jnp.where(
                    action_mask,
                    jnp.float32(0.0),
                    jnp.float32(-1.0e9),
                )
                action_ids, _ = sample_actions(
                    action_key,
                    logits,
                    config.action_head_sizes,
                    transport=config.action_transport,
                )
                next_recurrent_state = jnp.zeros_like(recurrent_state)
            else:
                action_ids = (
                    neutral_arsenal_policy_action_factors(config.num_envs)
                    if config.action_transport == "factors"
                    else jnp.zeros(
                        (config.num_envs,),
                        dtype=jnp.int32,
                    )
                )
                next_recurrent_state = jnp.zeros_like(recurrent_state)

            environment_keys = jax.random.split(
                environment_key,
                config.num_envs,
            )
            (
                stepped_environment_state,
                stepped_observation,
                reward,
                done,
                stepped_action_mask,
            ) = environment.step(
                environment_state,
                observation,
                action_ids,
                environment_keys,
            )
            episode_outcome = _episode_outcome(
                environment,
                stepped_environment_state,
                done,
            )

            reset_keys = jax.random.split(reset_key, config.num_envs)
            (
                reset_environment_state,
                reset_observation,
                reset_action_mask,
            ) = environment.reset(reset_keys)
            next_environment_state = _batch_select(
                done,
                reset_environment_state,
                stepped_environment_state,
            )
            next_observation = jnp.where(
                done[:, None],
                reset_observation,
                stepped_observation,
            )
            next_action_mask = jnp.where(
                done[:, None],
                reset_action_mask,
                stepped_action_mask,
            )
            next_recurrent_state = jnp.where(
                done[:, None],
                jnp.float32(0.0),
                next_recurrent_state,
            )

            accumulated_return = running_return + reward
            accumulated_length = running_length + jnp.int32(1)
            completed_return = jnp.where(
                done,
                accumulated_return,
                jnp.float32(0.0),
            )
            completed_length = jnp.where(
                done,
                accumulated_length,
                jnp.int32(0),
            )
            next_running_return = jnp.where(
                done,
                jnp.float32(0.0),
                accumulated_return,
            )
            next_running_length = jnp.where(
                done,
                jnp.int32(0),
                accumulated_length,
            )
            transition = FrozenOutcomeRollout(
                action_mask=action_mask,
                action=action_ids,
                reward=reward,
                done=done,
                completed_episode_return=completed_return,
                completed_episode_length=completed_length,
                completed_episode_success=episode_outcome.success,
                completed_episode_death=episode_outcome.death,
                completed_episode_simultaneous=episode_outcome.simultaneous,
                completed_episode_other=episode_outcome.other,
            )
            next_carry = (
                next_environment_state,
                next_observation,
                next_action_mask,
                next_recurrent_state,
                done,
                next_running_return,
                next_running_length,
            )
            return next_carry, transition

        initial_carry = (
            train_state.environment_state,
            train_state.observation,
            train_state.action_mask,
            train_state.recurrent_state,
            train_state.episode_start,
            train_state.running_episode_return,
            train_state.running_episode_length,
        )
        final_carry, rollout = jax.lax.scan(
            rollout_step,
            initial_carry,
            scan_keys,
        )
        (
            final_environment_state,
            final_observation,
            final_action_mask,
            final_recurrent_state,
            final_episode_start,
            final_running_return,
            final_running_length,
        ) = final_carry
        return (
            train_state._replace(
                environment_state=final_environment_state,
                observation=final_observation,
                action_mask=final_action_mask,
                recurrent_state=final_recurrent_state,
                episode_start=final_episode_start,
                running_episode_return=final_running_return,
                running_episode_length=final_running_length,
                total_environment_steps=(
                    train_state.total_environment_steps
                    + jnp.int32(config.num_envs * config.rollout_steps)
                ),
            ),
            rollout,
        )

    return jax.jit(collect) if compile else collect


def summarize_frozen_outcome_rollout(
    rollout: FrozenOutcomeRollout,
) -> dict[str, Any]:
    """Return an exhaustive, host-side outcome summary for one window."""

    counts = {
        "success": int(np.count_nonzero(rollout.completed_episode_success)),
        "death": int(np.count_nonzero(rollout.completed_episode_death)),
        "simultaneous": int(np.count_nonzero(rollout.completed_episode_simultaneous)),
        "other_terminal": int(np.count_nonzero(rollout.completed_episode_other)),
    }
    episodes_completed = int(np.count_nonzero(rollout.done))
    classified = sum(counts.values())
    if classified != episodes_completed:
        raise RuntimeError(
            "episode outcome taxonomy is not exhaustive: "
            f"{classified} != {episodes_completed}"
        )
    completed_return = float(np.asarray(rollout.completed_episode_return).sum())
    completed_length = int(np.asarray(rollout.completed_episode_length).sum())
    return {
        "mean_rollout_reward": float(np.asarray(rollout.reward).mean()),
        "mean_episode_return": (
            completed_return / episodes_completed if episodes_completed else 0.0
        ),
        "mean_episode_length": (
            completed_length / episodes_completed if episodes_completed else 0.0
        ),
        "episodes_completed": episodes_completed,
        "episode_outcomes": counts,
    }


def _episode_outcome(
    environment: PPOEnvironment,
    stepped_environment_state: Any,
    done: jax.Array,
) -> EpisodeOutcome:
    if environment.episode_outcome is None:
        false = jnp.zeros_like(done, dtype=jnp.bool_)
        return EpisodeOutcome(
            false, false, false, done, false, false, false, done, false
        )
    outcome = environment.episode_outcome(stepped_environment_state, done)
    return jax.tree.map(
        lambda value: done & jnp.asarray(value, dtype=jnp.bool_),
        outcome,
    )


def _batch_select(mask: jax.Array, when_true: Any, when_false: Any):
    def select_leaf(true_leaf, false_leaf):
        expanded_mask = mask.reshape((mask.shape[0],) + (1,) * (true_leaf.ndim - 1))
        return jnp.where(expanded_mask, true_leaf, false_leaf)

    return jax.tree.map(select_leaf, when_true, when_false)


__all__ = [
    "FrozenOutcomeArm",
    "FrozenOutcomeCollector",
    "FrozenOutcomeRollout",
    "make_frozen_outcome_collector",
    "summarize_frozen_outcome_rollout",
]
