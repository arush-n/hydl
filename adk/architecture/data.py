"""Backend-labelled temporal records for PLAN-style training pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Any, NamedTuple

import jax

from adk.architecture.inputs import ActorPolicyInput


class ExperienceSource(IntEnum):
    """Stable numeric source labels suitable for JAX arrays and replay."""

    HUMAN_DEMO = 0
    SCRIPTED_DEMO = 1
    NATIVE_NPC = 2
    REAL_HYTALE = 3
    JAX_SIM = 4
    LEARNED_DREAM = 5


class TickRecord(NamedTuple):
    """One rule-tick record; payload schemas stay owned by their producer."""

    header: Any
    privileged_scene: Any
    legal_observation_delta: Any
    exact_events: Any
    native_action_state: Any
    source: jax.Array


class DecisionTransition(NamedTuple):
    """One policy-decision record with no simulated/real source ambiguity."""

    actor_input: ActorPolicyInput
    privileged_scene_reference: Any
    recurrent_reset: jax.Array
    goal: Any
    legal_action_mask: jax.Array
    selected_action: jax.Array
    behavior_log_probability: jax.Array
    reward: jax.Array
    reward_components: Any
    next_actor_input: ActorPolicyInput
    terminal: jax.Array
    truncation: jax.Array
    source: jax.Array
    info: Any


class OptionTransition(NamedTuple):
    """One temporally extended goal/option record."""

    belief_at_start: Any
    goal: Any
    low_level_actions: Any
    duration: jax.Array
    accumulated_reward: jax.Array
    termination_reason: Any
    belief_at_end: Any
    source: jax.Array


@dataclass(frozen=True, slots=True)
class SequenceReplaySpec:
    """Static recurrent replay window semantics."""

    sequence_length: int = 64
    burn_in: int = 16
    overlap_fraction: float = 0.25

    def __post_init__(self) -> None:
        for name in ("sequence_length", "burn_in"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if self.burn_in < 0 or self.burn_in >= self.sequence_length:
            raise ValueError("burn_in must be in [0, sequence_length)")
        overlap = self.overlap_fraction
        if (
            isinstance(overlap, bool)
            or not isinstance(overlap, (int, float))
            or not math.isfinite(float(overlap))
            or not 0.0 <= float(overlap) < 1.0
        ):
            raise ValueError("overlap_fraction must be finite and in [0, 1)")
        object.__setattr__(self, "overlap_fraction", float(overlap))

    @property
    def stride(self) -> int:
        return max(1, round(self.sequence_length * (1.0 - self.overlap_fraction)))


__all__ = [
    "DecisionTransition",
    "ExperienceSource",
    "OptionTransition",
    "SequenceReplaySpec",
    "TickRecord",
]
