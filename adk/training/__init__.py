"""Optional recurrent-PPO tools built on the general ADK environment port.

The canonical runtime lives in :mod:`adk.runtime.env_adapter`; this module
adapts it to one learner implementation.  Algorithms are free to consume the
structured reset/step/collection API directly.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

import jax

from hytalegym.jax.training.ppo import (
    initialize_training as gym_initialize_training,
)
from hytalegym.jax.training.types import PPOConfig, PPOTrainState

from adk.runtime.env_adapter import BuiltAgent
from adk.training._shared_state import make_shared_state_train_step
from adk.training.ppo_adapter import as_ppo_environment


def _require_jax(built: BuiltAgent) -> None:
    if not isinstance(built, BuiltAgent):
        raise TypeError("built must be a BuiltAgent")
    if built.backend.backend != "jax":
        raise RuntimeError("ADK training is JAX-only")


def _default_minibatches(batch: int) -> int:
    for candidate in range(min(8, batch), 0, -1):
        if batch % candidate == 0:
            return candidate
    return 1


def make_training_config(
    built: BuiltAgent,
    **overrides: Any,
) -> PPOConfig:
    """Derive every policy shape and factor-transport field from the environment."""

    _require_jax(built)
    protected = {
        "observation_size",
        "action_size",
        "action_head_sizes",
        "action_transport",
    }
    conflicts = protected.intersection(overrides)
    if conflicts:
        raise ValueError(
            "training shapes come from BuiltAgent and cannot be overridden: "
            + ", ".join(sorted(conflicts))
        )
    if "num_envs" in overrides and overrides["num_envs"] != built.batch:
        raise ValueError("training num_envs must equal BuiltAgent.batch")
    options = dict(overrides)
    options.setdefault("num_envs", built.batch)
    options.setdefault("num_minibatches", _default_minibatches(built.batch))
    config = PPOConfig.from_environment_spec(
        built.policy_surface.spec,
        **options,
    )
    _validate_config(built, config)
    return config


def initialize_training(
    built: BuiltAgent,
    key: jax.Array,
    config: PPOConfig | None = None,
    **config_overrides: Any,
) -> tuple[PPOConfig, PPOTrainState]:
    """Initialize policy, optimizer, recurrent carry, and the dense JAX env."""

    _require_jax(built)
    if config is not None and config_overrides:
        raise ValueError("pass either config or config overrides, not both")
    selected = (
        make_training_config(built, **config_overrides)
        if config is None
        else config
    )
    _validate_config(built, selected)
    return selected, gym_initialize_training(
        key,
        selected,
        environment=as_ppo_environment(built),
    )


def make_training_step(
    built: BuiltAgent,
    config: PPOConfig,
    *,
    compile: bool = True,
):
    """Build live Gym recurrent PPO with shared-state autoreset support."""

    _require_jax(built)
    _validate_config(built, config)
    return make_shared_state_train_step(
        config,
        environment=as_ppo_environment(built),
        compile=compile,
    )


def _validate_config(built: BuiltAgent, config: PPOConfig) -> None:
    if not isinstance(config, PPOConfig):
        raise TypeError("config must be a PPOConfig")
    for name in (
        "num_envs",
        "rollout_steps",
        "update_epochs",
        "num_minibatches",
        "encoder_size",
        "recurrent_size",
    ):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"PPO config {name} must be an integer")
    for name in (
        "learning_rate",
        "gamma",
        "gae_lambda",
        "clip_epsilon",
        "value_coefficient",
        "entropy_coefficient",
        "max_gradient_norm",
    ):
        value = getattr(config, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"PPO config {name} must be finite")
    expected = {
        "num_envs": built.batch,
        "observation_size": built.observation_size,
        "action_size": built.logit_size,
        "action_head_sizes": built.stamp.action_head_sizes,
        "action_transport": "factors",
    }
    errors = [
        f"{name}: config={getattr(config, name)!r}, built={value!r}"
        for name, value in expected.items()
        if getattr(config, name) != value
    ]
    if errors:
        raise ValueError("training config does not match BuiltAgent:\n- " + "\n- ".join(errors))


__all__ = [
    "PPOConfig",
    "PPOTrainState",
    "initialize_training",
    "make_training_config",
    "make_training_step",
    "as_ppo_environment",
]
