"""Reference recurrent PPO wired directly to an :class:`ArenaScene`."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from hytalegym.jax.training.ppo import (
    calculate_gae,
    initialize_training,
    make_policy_update,
    make_rollout_collector,
    make_train_step,
    ppo_optimizer,
)
from hytalegym.jax.training.types import PPOConfig

from arena.jax_env import ArenaScene
from arena.training.runtime.execution import ReplicatedExecution, make_replicated_step


def _validate(scene: ArenaScene, config: PPOConfig) -> None:
    if config.num_envs != scene.expected_batch:
        raise ValueError("PPO config num_envs must equal the Arena scene batch")
    reward = getattr(scene.reward, "config", None)
    if (
        reward is not None
        and getattr(reward, "progress_scale", 0.0)
        and abs(float(config.gamma) - float(reward.discount)) > 1.0e-9
    ):
        raise ValueError("PPO gamma must equal the task reward discount")


def ppo_config(scene: ArenaScene, **overrides: Any) -> PPOConfig:
    """Derive every policy shape from the task environment's public spec."""

    requested = overrides.pop("num_envs", scene.expected_batch)
    if requested != scene.expected_batch:
        raise ValueError("PPO num_envs must equal the Arena scene batch")
    reward_config = getattr(scene.reward, "config", None)
    reward_discount = getattr(reward_config, "discount", None)
    if reward_discount is not None and getattr(reward_config, "progress_scale", 0.0):
        requested_discount = overrides.setdefault("gamma", reward_discount)
        if abs(float(requested_discount) - float(reward_discount)) > 1.0e-9:
            raise ValueError("PPO gamma must equal the task reward discount")
    return PPOConfig.from_environment_spec(
        scene.environment.spec, num_envs=requested, **overrides
    )


def initialize_ppo(key: Any, scene: ArenaScene, config: PPOConfig):
    _validate(scene, config)
    return initialize_training(key, config, environment=scene.environment)


def warm_start_ppo(train_state: Any, policy_params: Any, config: PPOConfig):
    """Install IL weights into a fresh PPO state and reset optimizer/carry state."""

    if int(jax.device_get(train_state.update_count)) != 0:
        raise ValueError("PPO warm starts require a fresh training state")
    expected = jax.tree.leaves(train_state.policy_params)
    supplied = jax.tree.leaves(policy_params)
    if jax.tree.structure(train_state.policy_params) != jax.tree.structure(
        policy_params
    ):
        raise ValueError("IL policy parameter tree does not match PPO")
    if any(
        left.shape != right.shape or left.dtype != right.dtype
        for left, right in zip(expected, supplied, strict=True)
    ):
        raise ValueError("IL policy parameter leaves do not match PPO")
    return train_state._replace(
        policy_params=policy_params,
        optimizer_state=ppo_optimizer(config).init(policy_params),
        recurrent_state=jnp.zeros_like(train_state.recurrent_state),
        episode_start=jnp.ones_like(train_state.episode_start),
    )


def make_ppo_step(scene: ArenaScene, config: PPOConfig, *, compile: bool = True):
    _validate(scene, config)
    return make_train_step(config, environment=scene.environment, compile=compile)


def make_replicated_ppo_step(
    scene: ArenaScene,
    config: PPOConfig,
    replicas: int,
    *,
    compile: bool = True,
) -> ReplicatedExecution:
    """Batch the existing PPO update across compatible policies on one device."""

    _validate(scene, config)
    return make_replicated_step(
        make_train_step(config, environment=scene.environment, compile=False),
        replicas,
        contract_sha256=scene.contract_sha256,
        compile=compile,
    )


def make_ppo_collector(scene: ArenaScene, config: PPOConfig, *, compile: bool = True):
    """Expose PPO's rollout half for learned reward relabeling."""

    _validate(scene, config)
    return make_rollout_collector(
        config, environment=scene.environment, compile=compile
    )


def make_ppo_update(config: PPOConfig, *, compile: bool = True):
    """Expose PPO's optimizer half for a relabeled rollout."""

    return make_policy_update(config, compile=compile)


def ppo_next_observation(batch: Any, final_observation: Any) -> jax.Array:
    """Shift a rollout without treating autoreset observations as successors."""

    observation = batch.rollout.observation
    final = jnp.asarray(final_observation, dtype=observation.dtype)
    if final.shape != observation.shape[1:]:
        raise ValueError("final_observation must match one rollout time slice")
    shifted = jnp.concatenate((observation[1:], final[None]), axis=0)
    mask = batch.rollout.done.reshape(
        batch.rollout.done.shape + (1,) * (observation.ndim - 2)
    )
    return jnp.where(mask, observation, shifted)


def relabel_ppo_rewards(batch: Any, reward: Any, config: PPOConfig):
    """Replace rollout reward and recompute GAE without recollecting experience."""

    rollout = batch.rollout
    reward = jnp.asarray(reward, dtype=jnp.float32)
    if reward.shape != rollout.reward.shape:
        raise ValueError("reward must match rollout.reward")
    if config.gamma:
        bootstrap = (
            batch.advantages[-1] + rollout.value[-1] - rollout.reward[-1]
        ) / jnp.float32(config.gamma)
        bootstrap = jnp.where(rollout.done[-1], 0.0, bootstrap)
    else:
        bootstrap = jnp.zeros_like(rollout.value[-1])
    advantages, returns = calculate_gae(
        reward,
        rollout.value,
        rollout.done,
        bootstrap,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        time_limit_bootstrap_values=rollout.time_limit_bootstrap_value,
    )
    return batch._replace(
        rollout=rollout._replace(reward=jax.lax.stop_gradient(reward)),
        advantages=jax.lax.stop_gradient(advantages),
        returns=jax.lax.stop_gradient(returns),
    )


__all__ = [
    "PPOConfig",
    "initialize_ppo",
    "make_ppo_collector",
    "make_replicated_ppo_step",
    "make_ppo_step",
    "make_ppo_update",
    "ppo_config",
    "ppo_next_observation",
    "relabel_ppo_rewards",
    "warm_start_ppo",
]
