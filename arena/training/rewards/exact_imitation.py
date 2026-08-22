"""Dense, alignment-safe rewards for exact Arena action imitation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from arena.jax_contract import HEAD_SPANS
from hytalegym.jax.combat.observation.v3.policy.look_deltas import (
    decode_look_delta,
)


@dataclass(frozen=True, slots=True)
class ExactImitationConfig:
    """Weights for expert action agreement and unnecessary rotation."""

    action_mismatch_scale: float = 1.0
    aim_error_scale: float = 1.0
    excess_spin_scale: float = 1.0
    spin_tolerance_degrees: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "action_mismatch_scale",
            "aim_error_scale",
            "excess_spin_scale",
            "spin_tolerance_degrees",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


class ExactImitationTerms(NamedTuple):
    reward: jax.Array
    action_mismatch: jax.Array
    aim_error: jax.Array
    excess_spin: jax.Array
    valid: jax.Array


def exact_imitation_reward(
    action: Any,
    expert_action: Any,
    supervision_mask: Any,
    alignment_valid: Any,
    config: ExactImitationConfig = ExactImitationConfig(),
) -> ExactImitationTerms:
    """Penalize deviation from an expert on provably aligned frames.

    ``alignment_valid`` is mandatory because equal tick numbers after a policy
    diverges do not identify the same state. Callers must use replay-aligned
    rows or an expert queried on the policy's current state (for example,
    DAgger). Unavailable expert heads contribute neither reward nor a negative
    label.
    """

    if not isinstance(config, ExactImitationConfig):
        raise TypeError("config must be ExactImitationConfig")
    action = jnp.asarray(action, dtype=jnp.int32)
    expert = jnp.asarray(expert_action, dtype=jnp.int32)
    supervised = jnp.asarray(supervision_mask, dtype=jnp.bool_)
    aligned = jnp.asarray(alignment_valid, dtype=jnp.bool_)
    head_sizes = tuple(size for _, size in HEAD_SPANS.values())
    if action.shape != expert.shape or action.shape != supervised.shape:
        raise ValueError("action, expert_action, and supervision_mask must match")
    if action.ndim < 1 or action.shape[-1] != len(head_sizes):
        raise ValueError("actions must use the current factored Arena action ABI")
    if aligned.shape != action.shape[:-1]:
        raise ValueError("alignment_valid must match the action leading axes")

    sizes = jnp.asarray(head_sizes, dtype=jnp.int32)
    legal = (action >= 0) & (action < sizes) & (expert >= 0) & (expert < sizes)
    known = supervised & legal
    support = jnp.sum(known, axis=-1)
    mismatch = jnp.sum(known & (action != expert), axis=-1) / jnp.maximum(support, 1)

    names = tuple(HEAD_SPANS)
    yaw = names.index("yaw_delta_bins")
    pitch = names.index("pitch_delta_bins")
    yaw_action = decode_look_delta(action[..., yaw], head_sizes[yaw])
    yaw_expert = decode_look_delta(expert[..., yaw], head_sizes[yaw])
    pitch_action = decode_look_delta(action[..., pitch], head_sizes[pitch])
    pitch_expert = decode_look_delta(expert[..., pitch], head_sizes[pitch])
    yaw_known = known[..., yaw]
    pitch_known = known[..., pitch]
    aim_support = yaw_known.astype(jnp.float32) + pitch_known.astype(jnp.float32)
    aim_error = (
        yaw_known * jnp.abs(yaw_action - yaw_expert)
        + pitch_known * jnp.abs(pitch_action - pitch_expert)
    ) / (jnp.maximum(aim_support, 1.0) * 45.0)
    excess_spin = (
        yaw_known
        * jnp.maximum(
            jnp.abs(yaw_action)
            - jnp.abs(yaw_expert)
            - jnp.float32(config.spin_tolerance_degrees),
            0.0,
        )
        / 45.0
    )
    valid = aligned & (support > 0)
    penalty = (
        config.action_mismatch_scale * mismatch
        + config.aim_error_scale * aim_error
        + config.excess_spin_scale * excess_spin
    )
    reward = jnp.where(valid, -penalty, 0.0).astype(jnp.float32)
    return ExactImitationTerms(
        jax.lax.stop_gradient(reward),
        jax.lax.stop_gradient(jnp.where(valid, mismatch, 0.0)),
        jax.lax.stop_gradient(jnp.where(valid, aim_error, 0.0)),
        jax.lax.stop_gradient(jnp.where(valid, excess_spin, 0.0)),
        valid,
    )


__all__ = [
    "ExactImitationConfig",
    "ExactImitationTerms",
    "exact_imitation_reward",
]
