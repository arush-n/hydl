"""Reference recurrent PPO as one consumer of the general SDK primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import jax

from hytalegym.jax.training.types import (
    PPOConfig,
    PPOTrainState,
    RecurrentPolicyParams,
)

from adk.core.lifecycle import (
    LoadedCheckpoint,
    load_checkpoint,
    save_checkpoint,
)
from adk.evaluation import (
    EVALUATION_HORIZON_DEFAULT,
    EvaluationSuite,
    evaluate_controls,
    recurrent_policy,
)
from adk.runtime.env_adapter import BuiltAgent
from adk.training import (
    initialize_training,
    make_training_config,
    make_training_step,
)
from adk.training.ppo_adapter import as_ppo_environment


@dataclass(frozen=True, slots=True)
class PPOTools:
    """PPO-specific configuration, training, evaluation, and artifacts.

    This object is deliberately obtained from ``handle.ppo``.  Keeping it out
    of the base environment API makes the dependency direction visible: PPO
    adapts the SDK; the SDK does not adapt itself around PPO.
    """

    built: BuiltAgent

    @property
    def environment(self):
        return as_ppo_environment(self.built)

    def config(self, **overrides: Any) -> PPOConfig:
        return make_training_config(self.built, **overrides)

    def initialize(
        self,
        key: jax.Array,
        config: PPOConfig | None = None,
        **config_overrides: Any,
    ) -> tuple[PPOConfig, PPOTrainState]:
        return initialize_training(
            self.built,
            key,
            config,
            **config_overrides,
        )

    def train_step(self, config: PPOConfig, *, compile: bool = True):
        return make_training_step(self.built, config, compile=compile)

    def policy(
        self,
        params: RecurrentPolicyParams,
        *,
        decode_mode: str = "factored_argmax",
    ):
        return recurrent_policy(
            params,
            self.built.stamp.action_head_sizes,
            decode_mode=decode_mode,
        )

    def evaluate_controls(
        self,
        trained_params: RecurrentPolicyParams,
        config: PPOConfig,
        key: jax.Array,
        *,
        max_steps: int = EVALUATION_HORIZON_DEFAULT,
        trained_decode_mode: str = "factored_argmax",
        compile: bool = True,
        allow_short_horizon: bool = False,
    ) -> EvaluationSuite:
        return evaluate_controls(
            self.built,
            trained_params,
            config,
            key,
            max_steps=max_steps,
            trained_decode_mode=trained_decode_mode,
            compile=compile,
            allow_short_horizon=allow_short_horizon,
        )

    def save(
        self,
        path: str | Path,
        policy_params: RecurrentPolicyParams,
        config: PPOConfig,
        *,
        seed: int,
        updates: int,
        environment_steps: int,
        combat_target_active: bool = True,
        run_metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        return save_checkpoint(
            path,
            self.built,
            policy_params,
            config,
            seed=seed,
            updates=updates,
            environment_steps=environment_steps,
            combat_target_active=combat_target_active,
            run_metadata=run_metadata,
        )

    def load(self, path: str | Path) -> LoadedCheckpoint:
        return load_checkpoint(path, self.built)


__all__ = ["PPOTools"]
