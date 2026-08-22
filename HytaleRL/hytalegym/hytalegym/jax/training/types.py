"""Fixed PyTrees and static configuration for the reference JAX PPO tool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax

from hytalegym.jax.combat import (
    ACTIVE_OBSERVATION_SIZE,
    SKILL_COUNT,
)
from hytalegym.jax.combat.environment import EnvironmentSpec
from hytalegym.jax.policy import DenseParams


Array = jax.Array


@dataclass(frozen=True)
class PPOConfig:
    """Static shapes and hyperparameters captured by one compiled update."""

    num_envs: int = 256
    rollout_steps: int = 128
    update_epochs: int = 4
    num_minibatches: int = 8
    encoder_size: int = 64
    recurrent_size: int = 128
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 0.5
    observation_size: int = ACTIVE_OBSERVATION_SIZE
    action_size: int = SKILL_COUNT
    action_head_sizes: tuple[int, ...] = ()
    action_transport: str = "scalar"
    action_distribution: str = "independent"
    episode_horizon_ticks: int | None = None

    @classmethod
    def from_environment_spec(
        cls,
        environment_spec: EnvironmentSpec,
        **overrides: Any,
    ) -> PPOConfig:
        """Build PPO shapes from an explicit dense environment adapter."""

        if not environment_spec.is_dense:
            raise ValueError(
                "PPO needs a dense adapter with observation_size and action_size"
            )
        conflicting = {"observation_size", "action_size"}.intersection(overrides)
        if conflicting:
            names = ", ".join(sorted(conflicting))
            raise ValueError(
                f"{names} come from environment_spec and cannot be overridden"
            )
        derived: dict[str, Any] = {}
        component_sizes = tuple(
            component.size for component in environment_spec.action_components
        )
        if len(component_sizes) > 1 and all(
            size is not None for size in component_sizes
        ):
            derived["action_head_sizes"] = tuple(
                int(size) for size in component_sizes if size is not None
            )
            derived["action_transport"] = "factors"
        derived.update(overrides)
        return cls(
            observation_size=environment_spec.observation_size,
            action_size=environment_spec.action_size,
            **derived,
        )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action_head_sizes",
            tuple(self.action_head_sizes),
        )
        integer_fields = {
            "num_envs": self.num_envs,
            "rollout_steps": self.rollout_steps,
            "update_epochs": self.update_epochs,
            "num_minibatches": self.num_minibatches,
            "encoder_size": self.encoder_size,
            "recurrent_size": self.recurrent_size,
        }
        for name, value in integer_fields.items():
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.episode_horizon_ticks is not None and (
            isinstance(self.episode_horizon_ticks, bool)
            or not isinstance(self.episode_horizon_ticks, int)
            or self.episode_horizon_ticks < 1
        ):
            raise ValueError("episode_horizon_ticks must be a positive integer or None")
        if self.num_envs % self.num_minibatches:
            raise ValueError("num_envs must be divisible by num_minibatches")
        if self.observation_size < 1:
            raise ValueError("observation_size must be positive")
        if self.action_size < 2:
            raise ValueError("action_size must contain at least two actions")
        if self.action_head_sizes:
            if any(size < 2 for size in self.action_head_sizes):
                raise ValueError(
                    "every factored action head must contain at least two actions"
                )
            if sum(self.action_head_sizes) != self.action_size:
                raise ValueError("factored action heads must sum to action_size")
        if self.action_transport not in {"scalar", "factors"}:
            raise ValueError("action_transport must be 'scalar' or 'factors'")
        if self.action_transport == "factors" and not self.action_head_sizes:
            raise ValueError("factor action transport requires action_head_sizes")
        if self.action_distribution not in {
            "independent",
            "arsenal_standard_root_v1",
        }:
            raise ValueError(
                "action_distribution must be 'independent' or "
                "'arsenal_standard_root_v1'"
            )
        if self.action_distribution == "arsenal_standard_root_v1":
            from hytalegym.jax.combat.observation.v3.policy.distribution import (
                validate_arsenal_standard_root_distribution,
            )

            validate_arsenal_standard_root_distribution(
                self.action_head_sizes,
                self.action_transport,
            )
        bounded_fields = {
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_epsilon": self.clip_epsilon,
            "value_coefficient": self.value_coefficient,
            "entropy_coefficient": self.entropy_coefficient,
            "max_gradient_norm": self.max_gradient_norm,
            "learning_rate": self.learning_rate,
        }
        for name, value in bounded_fields.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")


class GRUParams(NamedTuple):
    input_kernel: Array
    recurrent_kernel: Array
    bias: Array


class RecurrentPolicyParams(NamedTuple):
    encoder_input: DenseParams
    encoder_hidden: DenseParams
    gru: GRUParams
    actor: DenseParams
    critic: DenseParams


class PPOTrainState(NamedTuple):
    """Persistent state carried from one compiled update to the next."""

    policy_params: RecurrentPolicyParams
    optimizer_state: Any
    environment_state: Any
    observation: Array
    action_mask: Array
    recurrent_state: Array
    episode_start: Array
    running_episode_return: Array
    running_episode_length: Array
    update_count: Array
    total_environment_steps: Array


class EpisodeOutcome(NamedTuple):
    """Natural/support outcomes plus a disjoint learner-side time limit."""

    success: Array
    death: Array
    simultaneous: Array
    other: Array
    geometry_exhausted: Array
    target_navigation_unsupported: Array
    invalid: Array
    unclassified: Array
    time_limit: Array


class RecurrentRollout(NamedTuple):
    """Time-major behavior-policy trajectory used by GAE and PPO."""

    observation: Array
    action_mask: Array
    action: Array
    log_probability: Array
    value: Array
    reward: Array
    done: Array
    episode_start: Array
    completed_episode_return: Array
    completed_episode_length: Array
    completed_episode_success: Array
    completed_episode_death: Array
    completed_episode_simultaneous: Array
    completed_episode_other: Array
    completed_episode_geometry_exhausted: Array
    completed_episode_target_navigation_unsupported: Array
    completed_episode_invalid: Array
    completed_episode_unclassified: Array
    completed_episode_time_limit: Array
    time_limit_bootstrap_value: Array


class PPOUpdateBatch(NamedTuple):
    """Rollout-derived tensors consumed by one PPO optimizer phase."""

    rollout: RecurrentRollout
    advantages: Array
    returns: Array
    initial_recurrent_state: Array


class LossMetrics(NamedTuple):
    total_loss: Array
    policy_loss: Array
    value_loss: Array
    entropy: Array
    approximate_kl: Array
    clip_fraction: Array
    #: Share of samples whose stored and recomputed log-probabilities were not
    #: comparable, so they took no part in the ratio. Above ~0 means the action
    #: masks and the legality predicate disagree.
    incomparable_fraction: Array
    #: Shape of |log ratio| over the comparable samples. `approximate_kl`
    #: exponentiates this quantity, so its mean says nothing about whether the
    #: whole distribution moved or a thin tail carried it. Read the three
    #: together: a small median beside a large maximum is a tail, not a policy
    #: that has walked away from its rollout.
    log_ratio_median: Array
    log_ratio_p999: Array
    maximum_log_ratio: Array


class PPOMetrics(NamedTuple):
    """Scalar device metrics returned once per compiled update."""

    total_loss: Array
    policy_loss: Array
    value_loss: Array
    entropy: Array
    approximate_kl: Array
    clip_fraction: Array
    incomparable_fraction: Array
    log_ratio_median: Array
    log_ratio_p999: Array
    maximum_log_ratio: Array
    explained_variance: Array
    mean_rollout_reward: Array
    episodes_completed: Array
    episode_successes: Array
    episode_deaths: Array
    episode_simultaneous: Array
    episode_other_terminal: Array
    episode_geometry_exhausted: Array
    episode_target_navigation_unsupported: Array
    episode_invalid: Array
    episode_unclassified: Array
    episode_time_limit: Array
    mean_episode_return: Array
    mean_episode_length: Array
    update_count: Array
    total_environment_steps: Array
