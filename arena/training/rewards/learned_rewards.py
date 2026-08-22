"""Small JAX kernels for learned exploration and imitation rewards."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True, slots=True)
class RewardMixConfig:
    """Static scaling for one extrinsic and one auxiliary reward stream."""

    extrinsic_scale: float = 1.0
    intrinsic_scale: float = 1.0
    clip: float | None = 10.0
    epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        for name in ("extrinsic_scale", "intrinsic_scale", "epsilon"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.epsilon == 0.0:
            raise ValueError("epsilon must be positive")
        if self.clip is not None and (
            isinstance(self.clip, bool)
            or not isinstance(self.clip, Real)
            or not math.isfinite(float(self.clip))
            or self.clip <= 0.0
        ):
            raise ValueError("clip must be positive or None")


def prediction_error(prediction: Any, target: Any) -> jax.Array:
    """Per-example feature MSE used by RND and forward-model curiosity."""

    prediction = jnp.asarray(prediction, dtype=jnp.float32)
    target = jnp.asarray(target, dtype=jnp.float32)
    if prediction.shape != target.shape or prediction.ndim < 1:
        raise ValueError("prediction and target need the same [..., features] shape")
    return jnp.mean(jnp.square(prediction - target), axis=-1)


def prediction_loss(prediction: Any, target: Any) -> jax.Array:
    """Train a predictor without updating its fixed or encoded target."""

    target = jax.lax.stop_gradient(jnp.asarray(target, dtype=jnp.float32))
    return jnp.mean(prediction_error(prediction, target))


def prediction_reward(prediction: Any, target: Any) -> jax.Array:
    """Novelty reward; gradients stay in the predictor's separate loss path."""

    return jax.lax.stop_gradient(prediction_error(prediction, target))


def impact_reward(before: Any, after: Any, visit_count: Any = 1.0) -> jax.Array:
    """RIDE-style latent impact divided by the next state's episodic count."""

    before = jnp.asarray(before, dtype=jnp.float32)
    after = jnp.asarray(after, dtype=jnp.float32)
    if before.shape != after.shape or before.ndim < 1:
        raise ValueError("before and after need the same [..., features] shape")
    count = jnp.asarray(visit_count, dtype=jnp.float32)
    valid = jnp.isfinite(count) & (count >= 1.0)
    safe_count = jnp.where(valid, count, 1.0)
    impact = jnp.sqrt(jnp.sum(jnp.square(after - before), axis=-1))
    return jax.lax.stop_gradient(jnp.where(valid, impact / jnp.sqrt(safe_count), 0.0))


def adversarial_imitation_reward(
    expert_logit: Any, *, objective: str = "gail"
) -> jax.Array:
    """Reward a policy from an expert-probability discriminator logit."""

    logit = jnp.asarray(expert_logit, dtype=jnp.float32)
    if objective == "gail":
        reward = jax.nn.softplus(logit)  # -log(1 - D_expert)
    elif objective == "airl":
        reward = logit  # log D_expert - log(1 - D_expert)
    else:
        raise ValueError("objective must be 'gail' or 'airl'")
    return jax.lax.stop_gradient(reward)


def discriminator_loss(expert_logit: Any, policy_logit: Any) -> jax.Array:
    """Stable binary loss for expert=1 and current-policy=0 samples."""

    expert = jnp.asarray(expert_logit, dtype=jnp.float32)
    policy = jnp.asarray(policy_logit, dtype=jnp.float32)
    return jnp.mean(jax.nn.softplus(-expert)) + jnp.mean(jax.nn.softplus(policy))


def mix_rewards(
    extrinsic: Any,
    intrinsic: Any,
    config: RewardMixConfig = RewardMixConfig(),
    *,
    intrinsic_std: Any | None = None,
) -> jax.Array:
    """Normalize, scale, and clip two same-shaped reward streams."""

    if not isinstance(config, RewardMixConfig):
        raise TypeError("config must be a RewardMixConfig")
    extrinsic = jnp.asarray(extrinsic, dtype=jnp.float32)
    intrinsic = jnp.asarray(intrinsic, dtype=jnp.float32)
    if extrinsic.shape != intrinsic.shape:
        raise ValueError("extrinsic and intrinsic rewards must have the same shape")
    if intrinsic_std is not None:
        std = jnp.asarray(intrinsic_std, dtype=jnp.float32)
        valid = jnp.isfinite(std) & (std > 0.0)
        intrinsic = jnp.where(valid, intrinsic / jnp.maximum(std, config.epsilon), 0.0)
    reward = config.extrinsic_scale * extrinsic + config.intrinsic_scale * intrinsic
    if config.clip is not None:
        reward = jnp.clip(reward, -config.clip, config.clip)
    return jax.lax.stop_gradient(reward.astype(jnp.float32))


__all__ = [
    "RewardMixConfig",
    "adversarial_imitation_reward",
    "discriminator_loss",
    "impact_reward",
    "mix_rewards",
    "prediction_error",
    "prediction_loss",
    "prediction_reward",
]
