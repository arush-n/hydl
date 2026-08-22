"""Explicit, composable per-step rewards for Arena JAX tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any, Callable, Mapping, NamedTuple

import jax.numpy as jnp

from arena.jax_contract import GROUP_FEATURES, READABLE_GROUPS
from arena.tasks.framework.goals import Goal

TASK_REWARD_SCHEMA = "arena_task_reward_v3"


class RewardTerms(NamedTuple):
    potential_before: Any
    potential_after: Any
    state: Any
    event: Any
    completion: Any
    failure: Any


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Weights applied to explicit task signals and the engine reward."""

    native_scale: float = 1.0
    progress_scale: float = 1.0
    discount: float = 0.99
    state_scale: float = 0.0
    event_scale: float = 1.0
    completion_bonus: float = 1.0
    failure_penalty: float = 1.0
    step_penalty: float = 0.0
    clip: float | None = 10.0

    def __post_init__(self) -> None:
        values = {
            "native_scale": self.native_scale,
            "progress_scale": self.progress_scale,
            "state_scale": self.state_scale,
            "event_scale": self.event_scale,
            "completion_bonus": self.completion_bonus,
            "failure_penalty": self.failure_penalty,
            "step_penalty": self.step_penalty,
        }
        for name, value in values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"reward {name} must be finite")
            if name not in {"event_scale", "state_scale"} and value < 0:
                raise ValueError(f"reward {name} must be non-negative")
        if (
            isinstance(self.discount, bool)
            or not isinstance(self.discount, Real)
            or not math.isfinite(float(self.discount))
            or not 0.0 <= self.discount <= 1.0
        ):
            raise ValueError("reward discount must be finite and in [0, 1]")
        if self.clip is not None:
            if (
                isinstance(self.clip, bool)
                or not isinstance(self.clip, Real)
                or not math.isfinite(float(self.clip))
            ):
                raise ValueError("reward clip must be finite or None")
            if self.clip <= 0:
                raise ValueError("reward clip must be positive")

    def describe(self) -> dict[str, float | None]:
        return {
            "native_scale": float(self.native_scale),
            "progress_scale": float(self.progress_scale),
            "discount": float(self.discount),
            "state_scale": float(self.state_scale),
            "event_scale": float(self.event_scale),
            "completion_bonus": float(self.completion_bonus),
            "failure_penalty": float(self.failure_penalty),
            "step_penalty": float(self.step_penalty),
            "clip": None if self.clip is None else float(self.clip),
        }


@dataclass(frozen=True, slots=True)
class RewardSignal:
    """One named signal program; the callable stays inside the JAX trace."""

    label: str
    group: str
    feature: str
    evaluate: Callable[[Mapping[str, Any], Mapping[str, Any]], RewardTerms] = field(
        repr=False, compare=False
    )
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label or not callable(self.evaluate):
            raise ValueError("reward signal needs a label and callable evaluator")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("reward signal parameters must be a mapping")
        if self.group not in READABLE_GROUPS:
            raise ValueError(f"reward group {self.group!r} is not always readable")
        if self.feature not in GROUP_FEATURES[self.group]:
            raise ValueError(f"unknown reward feature {self.group}.{self.feature}")
        evidence = self.parameters.get("completion_evidence")
        if evidence is not None and (not isinstance(evidence, str) or not evidence):
            raise ValueError("completion evidence must be a non-empty name")

    def describe(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "group": self.group,
            "feature": self.feature,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class TaskReward:
    """A pure JAX reward contract attached explicitly to a :class:`Task`."""

    label: str
    signals: tuple[RewardSignal, ...]
    config: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        if not self.label or not isinstance(self.signals, tuple) or not self.signals:
            raise ValueError("task reward needs a label and at least one signal")
        if any(not isinstance(signal, RewardSignal) for signal in self.signals):
            raise TypeError("task reward signals must be RewardSignal instances")
        if not isinstance(self.config, RewardConfig):
            raise TypeError("task reward config must be a RewardConfig")

    @property
    def required_groups(self) -> frozenset[str]:
        return frozenset(signal.group for signal in self.signals)

    @property
    def required_completion_evidence(self) -> frozenset[str]:
        return frozenset(
            evidence
            for signal in self.signals
            if (evidence := signal.parameters.get("completion_evidence")) is not None
        )

    def terms(
        self,
        previous: Mapping[str, Any],
        current: Mapping[str, Any],
        *,
        completion_evidence: Mapping[str, Any] | None = None,
    ) -> RewardTerms:
        terms = []
        for signal in self.signals:
            term = signal.evaluate(previous, current)
            evidence = signal.parameters.get("completion_evidence")
            if evidence is not None and completion_evidence is not None:
                exact = jnp.asarray(completion_evidence[evidence], dtype=jnp.float32)
                term = term._replace(completion=jnp.maximum(term.completion, exact))
            terms.append(term)
        return RewardTerms(
            *(
                sum(term[index] for term in terms)
                for index in range(len(RewardTerms._fields))
            )
        )

    def apply(
        self,
        native_reward: Any,
        previous: Mapping[str, Any],
        current: Mapping[str, Any],
        *,
        terminal: Any | None = None,
        completion_evidence: Mapping[str, Any] | None = None,
    ) -> Any:
        terms = self.terms(previous, current, completion_evidence=completion_evidence)
        config = self.config
        next_potential = terms.potential_after
        if terminal is not None:
            next_potential = jnp.where(
                jnp.asarray(terminal, dtype=jnp.bool_), 0.0, next_potential
            )
        shaping = config.discount * next_potential - terms.potential_before
        reward = (
            jnp.asarray(native_reward, dtype=jnp.float32) * config.native_scale
            + shaping * config.progress_scale
            + terms.state * config.state_scale
            + terms.event * config.event_scale
            + terms.completion * config.completion_bonus
            - terms.failure * config.failure_penalty
            - config.step_penalty
        )
        if config.clip is not None:
            reward = jnp.clip(reward, -config.clip, config.clip)
        return reward.astype(jnp.float32)

    def describe(self) -> dict[str, Any]:
        return {
            "schema": TASK_REWARD_SCHEMA,
            "label": self.label,
            "signals": [signal.describe() for signal in self.signals],
            "required_groups": sorted(self.required_groups),
            "required_completion_evidence": sorted(self.required_completion_evidence),
            "config": self.config.describe(),
        }


def _feature_signal(
    label: str,
    group: str,
    feature: str,
    *,
    invert: bool = False,
    scale: float = 1.0,
    completion: float | None = None,
    failure: float | None = None,
    availability_feature: str | None = None,
    completion_evidence: str | None = None,
) -> RewardSignal:
    index = GROUP_FEATURES[group].index(feature)
    availability_index = (
        None
        if availability_feature is None
        else GROUP_FEATURES[group].index(availability_feature)
    )

    def score(record):
        value = jnp.asarray(record[group])[..., index] / jnp.float32(scale)
        value = jnp.clip(value, 0.0, 1.0)
        return 1.0 - value if invert else value

    def evaluate(previous, current):
        before, after = score(previous), score(current)
        available = (
            jnp.ones_like(after, dtype=jnp.bool_)
            if availability_index is None
            else (jnp.asarray(previous[group])[..., availability_index] > 0.5)
            & (jnp.asarray(current[group])[..., availability_index] > 0.5)
        )
        before = jnp.where(available, before, 0.0)
        after = jnp.where(available, after, 0.0)
        complete = (
            jnp.zeros_like(after)
            if completion is None
            else available & (before < completion) & (after >= completion)
        )
        failed = (
            jnp.zeros_like(after)
            if failure is None
            else available & (before >= failure) & (after < failure)
        )
        return RewardTerms(
            before,
            after,
            after,
            jnp.zeros_like(after),
            complete.astype(jnp.float32),
            failed.astype(jnp.float32),
        )

    return RewardSignal(
        label,
        group,
        feature,
        evaluate,
        {
            "invert": invert,
            "scale": scale,
            "completion": completion,
            "failure": failure,
            "availability_feature": availability_feature,
            "completion_evidence": completion_evidence,
        },
    )


def _change_signal(group: str, feature: str, minimum: float) -> RewardSignal:
    index = GROUP_FEATURES[group].index(feature)

    def evaluate(previous, current):
        before = jnp.asarray(previous[group])[..., index]
        after = jnp.asarray(current[group])[..., index]
        change = jnp.clip(jnp.abs(after - before) / minimum, 0.0, 1.0)
        zero = jnp.zeros_like(change)
        return RewardTerms(
            zero,
            zero,
            zero,
            change,
            (change >= 1.0).astype(jnp.float32),
            zero,
        )

    return RewardSignal(
        "terrain_change", group, feature, evaluate, {"minimum_change": minimum}
    )


def _signal_for_goal(goal: Goal) -> RewardSignal:
    kind, values = goal.kind, goal.parameters
    if kind in {"defeat_target", "defeat_within"}:
        return _feature_signal(
            kind,
            "target_f32",
            "health_fraction",
            invert=True,
            completion=1.0,
            availability_feature="visible",
            completion_evidence=(
                "episode_success" if kind == "defeat_target" else None
            ),
        )
    if kind == "reduce_target_health_below":
        return _feature_signal(
            kind,
            "target_f32",
            "health_fraction",
            invert=True,
            completion=1.0 - float(values["fraction"]),
            availability_feature="visible",
        )
    if kind in {"stay_alive", "survive_for"}:
        return _feature_signal(kind, "self_f32", "alive", failure=0.5)
    if kind == "keep_health_above":
        return _feature_signal(
            kind, "self_f32", "health_fraction", failure=float(values["fraction"])
        )
    if kind == "keep_target_visible":
        return _feature_signal(kind, "target_f32", "visible")
    if kind == "avoid_hazards":
        return _feature_signal(
            kind,
            "hazard_f32",
            "distance",
            scale=max(float(values["distance"]), 1.0e-6),
        )
    if kind == "reach_interaction":
        return _feature_signal(
            kind,
            "interaction_f32",
            "reach_fraction",
            completion=float(values["reach"]),
        )
    if kind == "changed_terrain_at_interaction":
        return _change_signal(
            "terrain_f32", "traversability", float(values["minimum_change"])
        )
    if kind == "hold_movement_state":
        return _feature_signal(kind, "movement_state_f32", str(values["state"]))
    if kind == "gather_resource":
        return _feature_signal(
            kind,
            "resource_f32",
            str(values["resource"]),
            completion=float(values["amount"]),
        )
    if kind == "spend_no_more_than":
        return _feature_signal(
            kind,
            "resource_f32",
            str(values["resource"]),
            failure=float(values["amount"]),
        )
    raise ValueError(
        f"goal {goal.label!r} has no sound per-step reward signal; "
        "attach a TaskReward explicitly"
    )


def reward_for_goal(goal: Goal, config: RewardConfig | None = None) -> TaskReward:
    """Create an explicit per-step contract; never changes goal semantics."""

    if not isinstance(goal, Goal):
        raise TypeError("reward_for_goal expects a Goal")
    return TaskReward(goal.label, (_signal_for_goal(goal),), config or RewardConfig())


def compose_rewards(
    *rewards: TaskReward,
    label: str = "composed_reward",
    config: RewardConfig | None = None,
) -> TaskReward:
    """Fuse signal programs; weights must be chosen once for the composition."""

    if not rewards:
        raise ValueError("compose_rewards needs at least one reward")
    selected = config or rewards[0].config
    if config is None and any(reward.config != selected for reward in rewards[1:]):
        raise ValueError("composed rewards need one explicit shared RewardConfig")
    return TaskReward(
        label, tuple(s for reward in rewards for s in reward.signals), selected
    )


__all__ = [
    "RewardConfig",
    "RewardSignal",
    "RewardTerms",
    "TaskReward",
    "TASK_REWARD_SCHEMA",
    "compose_rewards",
    "reward_for_goal",
]
