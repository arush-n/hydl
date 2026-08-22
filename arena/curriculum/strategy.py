"""Composable contracts for GPU-resident skill training.

Tasks describe what an evaluation means and :mod:`arena.curriculum.ladder`
orders those evaluations.  A training strategy fills the missing middle: it
binds causal learning signals, target motion, episode lifetime, and retained
skills without knowing anything about a particular policy parameter tree.

The contract is deliberately declarative.  Target motion must still pass
through the normal JAX action decoder and physics; a strategy never teleports
an actor or labels privileged state as an observation. Policy packages such as
Dawn adapt ``exercised_capabilities`` to their own executable action surface
and independently declare which parameters may change.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from numbers import Integral
from typing import Any, Iterable


ARENA_TRAINING_STRATEGY_SCHEMA = "arena-jax-training-strategy-v1"
MINIMUM_SKILL_EPISODE_TICKS = 1_000


def _hash(value: Any) -> str:
    return (
        hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        )
        .hexdigest()
        .upper()
    )


def _label(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical label")
    return value


def _labels(values: Iterable[str], name: str) -> tuple[str, ...]:
    selected = tuple(_label(value, name) for value in values)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError(f"{name} values must be non-empty and unique")
    return tuple(sorted(selected))


class SkillAxis(str, Enum):
    """Policy-independent abilities that a strategy can isolate."""

    PURSUIT = "pursuit"
    AIM = "aim"
    ATTACK_TIMING = "attack_timing"
    GUARD_TIMING = "guard_timing"


class TargetMotionPattern(str, Enum):
    """Target-motion families, all executed through actor-legal actions."""

    STATIONARY = "stationary"
    RADIAL = "radial_toward_or_away"
    STRAFE = "lateral_strafe"
    MIXED_POLICY = "sampled_frozen_policy"


@dataclass(frozen=True, slots=True)
class CausalObjective:
    """One reusable learning axis and the evidence that makes it sound."""

    axis: SkillAxis
    reward_law: str
    success_contract: str
    progress_evidence: str
    required_evidence: tuple[str, ...]
    exercised_capabilities: tuple[str, ...]
    speed_is_objective: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.axis, SkillAxis):
            raise TypeError("objective axis must be a SkillAxis")
        for name in ("reward_law", "success_contract", "progress_evidence"):
            object.__setattr__(self, name, _label(getattr(self, name), name))
        object.__setattr__(
            self,
            "required_evidence",
            _labels(self.required_evidence, "required evidence"),
        )
        object.__setattr__(
            self,
            "exercised_capabilities",
            _labels(self.exercised_capabilities, "exercised capability"),
        )
        if not isinstance(self.speed_is_objective, bool):
            raise TypeError("speed_is_objective must be boolean")

    def manifest(self) -> dict[str, Any]:
        return {
            "axis": self.axis.value,
            "reward_law": self.reward_law,
            "success_contract": self.success_contract,
            "progress_evidence": self.progress_evidence,
            "required_evidence": self.required_evidence,
            "exercised_capabilities": self.exercised_capabilities,
            "speed_is_objective": self.speed_is_objective,
        }


@dataclass(frozen=True, slots=True)
class TargetMotionContract:
    """How target motion is varied without bypassing the action surface."""

    patterns: tuple[TargetMotionPattern, ...]
    controller: str = "jax_policy_action_decoder_and_physics"
    reset_distribution: str = "lane_stratified_seeded_reset"
    learner_credit: str = "learner_owned_change_only"
    direct_state_mutation: bool = False

    def __post_init__(self) -> None:
        patterns = tuple(self.patterns)
        if (
            not patterns
            or any(not isinstance(value, TargetMotionPattern) for value in patterns)
            or len(patterns) != len(set(patterns))
        ):
            raise ValueError("target motion patterns must be non-empty and unique")
        object.__setattr__(
            self,
            "patterns",
            tuple(sorted(patterns, key=lambda value: value.value)),
        )
        for name in ("controller", "reset_distribution", "learner_credit"):
            object.__setattr__(self, name, _label(getattr(self, name), name))
        if self.direct_state_mutation is not False:
            raise ValueError("training target motion must pass through JAX actions")

    @property
    def moving(self) -> bool:
        return self.patterns != (TargetMotionPattern.STATIONARY,)

    def manifest(self) -> dict[str, Any]:
        return {
            "patterns": tuple(value.value for value in self.patterns),
            "controller": self.controller,
            "reset_distribution": self.reset_distribution,
            "learner_credit": self.learner_credit,
            "direct_state_mutation": self.direct_state_mutation,
        }


@dataclass(frozen=True, slots=True)
class EpisodeContract:
    """Long-horizon success/no-progress semantics shared by skill stages."""

    maximum_ticks: int
    no_progress_patience_ticks: int
    progress_evidence: str
    success_is_terminal: bool = True
    natural_combat_outcome_is_terminal: bool = True
    no_progress_is_truncated: bool = True
    safety_horizon_is_truncated: bool = True
    safety_horizon_bootstraps: bool = True
    optimizer_chunks_end_episode: bool = False

    def __post_init__(self) -> None:
        for name in ("maximum_ticks", "no_progress_patience_ticks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name.replace('_', ' ')} must be an integer")
            object.__setattr__(self, name, int(value))
        if self.maximum_ticks < MINIMUM_SKILL_EPISODE_TICKS:
            raise ValueError(
                "skill environments require at least 1000 maximum episode ticks"
            )
        if not 0 < self.no_progress_patience_ticks < self.maximum_ticks:
            raise ValueError("no-progress patience must be inside the episode horizon")
        object.__setattr__(
            self,
            "progress_evidence",
            _label(self.progress_evidence, "progress evidence"),
        )
        flags = (
            self.success_is_terminal,
            self.natural_combat_outcome_is_terminal,
            self.no_progress_is_truncated,
            self.safety_horizon_is_truncated,
            self.safety_horizon_bootstraps,
            self.optimizer_chunks_end_episode,
        )
        if any(not isinstance(value, bool) for value in flags):
            raise TypeError("episode semantics must be boolean")
        if flags != (True, True, True, True, True, False):
            raise ValueError(
                "skill episodes must end on success/natural outcome, truncate on "
                "no progress or safety horizon, bootstrap the horizon, and outlive "
                "optimizer chunks"
            )

    def manifest(self) -> dict[str, Any]:
        return {
            "maximum_ticks": self.maximum_ticks,
            "no_progress_patience_ticks": self.no_progress_patience_ticks,
            "progress_evidence": self.progress_evidence,
            "success_is_terminal": self.success_is_terminal,
            "natural_combat_outcome_is_terminal": (
                self.natural_combat_outcome_is_terminal
            ),
            "no_progress_is_truncated": self.no_progress_is_truncated,
            "safety_horizon_is_truncated": self.safety_horizon_is_truncated,
            "safety_horizon_bootstraps": self.safety_horizon_bootstraps,
            "optimizer_chunks_end_episode": self.optimizer_chunks_end_episode,
        }


@dataclass(frozen=True, slots=True)
class ArenaTrainingStrategy:
    """A policy-independent, content-addressed JAX training strategy."""

    name: str
    objectives: tuple[CausalObjective, ...]
    target_motion: TargetMotionContract
    episode: EpisodeContract
    retained_strategy_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _label(self.name, "strategy name"))
        objectives = tuple(self.objectives)
        if not objectives or any(
            not isinstance(value, CausalObjective) for value in objectives
        ):
            raise ValueError("training strategy needs causal objectives")
        axes = tuple(value.axis for value in objectives)
        if len(axes) != len(set(axes)):
            raise ValueError("training strategy repeats a skill axis")
        object.__setattr__(
            self,
            "objectives",
            tuple(sorted(objectives, key=lambda value: value.axis.value)),
        )
        if not isinstance(self.target_motion, TargetMotionContract):
            raise TypeError("training strategy target motion is invalid")
        if not isinstance(self.episode, EpisodeContract):
            raise TypeError("training strategy episode contract is invalid")
        retained = tuple(
            _label(value, "retained strategy ID")
            for value in self.retained_strategy_ids
        )
        if len(retained) != len(set(retained)) or self.name in retained:
            raise ValueError("retained strategy IDs must be unique predecessors")
        object.__setattr__(self, "retained_strategy_ids", tuple(sorted(retained)))

    @property
    def axes(self) -> tuple[SkillAxis, ...]:
        return tuple(value.axis for value in self.objectives)

    @property
    def exercised_capabilities(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    capability
                    for objective in self.objectives
                    for capability in objective.exercised_capabilities
                }
            )
        )

    @property
    def required_evidence(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    evidence
                    for objective in self.objectives
                    for evidence in objective.required_evidence
                }
            )
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": ARENA_TRAINING_STRATEGY_SCHEMA,
            "name": self.name,
            "objectives": tuple(value.manifest() for value in self.objectives),
            "target_motion": self.target_motion.manifest(),
            "episode": self.episode.manifest(),
            "retained_strategy_ids": self.retained_strategy_ids,
        }

    @property
    def contract_sha256(self) -> str:
        return _hash(self.manifest())


PURSUIT_OBJECTIVE = CausalObjective(
    SkillAxis.PURSUIT,
    "learner_displacement_toward_previous_target_plus_alignment_potential",
    "horizontal_target_distance_at_most_3_blocks_after_learner_progress",
    "learner_owned_radial_progress",
    (
        "actor_positions_before_and_after",
        "actor_yaws_before_and_after",
        "learner_actor_identity",
        "transition_valid",
    ),
    ("facing", "locomotion"),
)

AIM_OBJECTIVE = CausalObjective(
    SkillAxis.AIM,
    "learner_owned_yaw_error_reduction_to_current_target_minus_unnecessary_spin",
    "absolute_current_target_yaw_error_at_most_10_degrees",
    "learner_owned_alignment_gain",
    (
        "actor_positions_before_and_after",
        "actor_yaws_before_and_after",
        "learner_actor_identity",
        "transition_valid",
    ),
    ("facing",),
)

ATTACK_TIMING_OBJECTIVE = CausalObjective(
    SkillAxis.ATTACK_TIMING,
    "engine_attributed_damage_plus_causal_start_minus_false_activation",
    "required_engine_attributed_hits_from_causal_starts",
    "engine_attributed_damage_event",
    (
        "actor_visible_alignment_and_range",
        "authored_attack_readiness",
        "engine_ability_requested_and_accepted",
        "engine_attributed_damage",
        "previous_selected_action",
        "transition_valid",
    ),
    ("basic_attack",),
    speed_is_objective=False,
)

GUARD_TIMING_OBJECTIVE = CausalObjective(
    SkillAxis.GUARD_TIMING,
    "prevented_damage_plus_causal_guard_minus_guard_without_pressure",
    "engine_attributed_blocked_hit",
    "engine_attributed_blocked_hit",
    (
        "engine_attack_telegraph",
        "engine_blocked_hit",
        "engine_incoming_damage",
        "transition_valid",
    ),
    ("guard",),
)


def stationary_target_motion() -> TargetMotionContract:
    return TargetMotionContract((TargetMotionPattern.STATIONARY,))


def radial_target_motion() -> TargetMotionContract:
    """Declare actor-legal approach/retreat motion and nothing broader."""

    return TargetMotionContract((TargetMotionPattern.RADIAL,))


def strafe_target_motion() -> TargetMotionContract:
    """Declare actor-legal lateral motion and nothing broader."""

    return TargetMotionContract((TargetMotionPattern.STRAFE,))


def radial_strafe_target_motion() -> TargetMotionContract:
    """Declare the exact scripted moving-target families used together."""

    return TargetMotionContract(
        (TargetMotionPattern.RADIAL, TargetMotionPattern.STRAFE)
    )


def moving_target_motion() -> TargetMotionContract:
    """Declare every supported moving-target family, including frozen policy."""

    return TargetMotionContract(
        (
            TargetMotionPattern.RADIAL,
            TargetMotionPattern.STRAFE,
            TargetMotionPattern.MIXED_POLICY,
        )
    )


def _strategy(
    name: str,
    objective: CausalObjective,
    motion: TargetMotionContract,
    maximum_ticks: int,
    no_progress_patience_ticks: int,
) -> ArenaTrainingStrategy:
    return ArenaTrainingStrategy(
        name,
        (objective,),
        motion,
        EpisodeContract(
            maximum_ticks,
            no_progress_patience_ticks,
            objective.progress_evidence,
        ),
    )


def _resolved_motion(
    moving_target: bool,
    target_motion: TargetMotionContract | None,
) -> TargetMotionContract:
    if not isinstance(moving_target, bool):
        raise TypeError("moving_target must be boolean")
    resolved = (
        (moving_target_motion() if moving_target else stationary_target_motion())
        if target_motion is None
        else target_motion
    )
    if not isinstance(resolved, TargetMotionContract):
        raise TypeError("target_motion must be a TargetMotionContract")
    if resolved.moving != moving_target:
        raise ValueError("target_motion must agree with moving_target")
    return resolved


def pursuit_training_strategy(
    *,
    moving_target: bool = False,
    target_motion: TargetMotionContract | None = None,
    maximum_ticks: int = 1_000,
    no_progress_patience_ticks: int = 250,
) -> ArenaTrainingStrategy:
    return _strategy(
        "moving_target_pursuit" if moving_target else "stationary_target_pursuit",
        PURSUIT_OBJECTIVE,
        _resolved_motion(moving_target, target_motion),
        maximum_ticks,
        no_progress_patience_ticks,
    )


def aim_training_strategy(
    *,
    moving_target: bool = True,
    target_motion: TargetMotionContract | None = None,
    maximum_ticks: int = 1_000,
    no_progress_patience_ticks: int = 250,
) -> ArenaTrainingStrategy:
    return _strategy(
        "moving_target_aim" if moving_target else "stationary_target_aim",
        AIM_OBJECTIVE,
        _resolved_motion(moving_target, target_motion),
        maximum_ticks,
        no_progress_patience_ticks,
    )


def attack_timing_training_strategy(
    *,
    moving_target: bool = False,
    target_motion: TargetMotionContract | None = None,
    maximum_ticks: int = 2_048,
    no_progress_patience_ticks: int = 250,
) -> ArenaTrainingStrategy:
    return _strategy(
        "moving_target_attack_timing"
        if moving_target
        else "stationary_target_attack_timing",
        ATTACK_TIMING_OBJECTIVE,
        _resolved_motion(moving_target, target_motion),
        maximum_ticks,
        no_progress_patience_ticks,
    )


def guard_timing_training_strategy(
    *,
    maximum_ticks: int = 2_048,
    no_progress_patience_ticks: int = 250,
) -> ArenaTrainingStrategy:
    return _strategy(
        "telegraphed_guard_timing",
        GUARD_TIMING_OBJECTIVE,
        moving_target_motion(),
        maximum_ticks,
        no_progress_patience_ticks,
    )


def compose_training_strategies(
    name: str,
    *strategies: ArenaTrainingStrategy,
    target_motion: TargetMotionContract | None = None,
    episode: EpisodeContract | None = None,
) -> ArenaTrainingStrategy:
    """Compose axes while requiring explicit resolution of environment drift."""

    if not strategies or any(
        not isinstance(value, ArenaTrainingStrategy) for value in strategies
    ):
        raise TypeError("strategy composition needs ArenaTrainingStrategy values")
    objectives: dict[SkillAxis, CausalObjective] = {}
    for strategy in strategies:
        for objective in strategy.objectives:
            previous = objectives.get(objective.axis)
            if previous is not None and previous != objective:
                raise ValueError("composed strategies disagree on a causal objective")
            objectives[objective.axis] = objective
    motions = {value.target_motion for value in strategies}
    episodes = {value.episode for value in strategies}
    if target_motion is None:
        if len(motions) != 1:
            raise ValueError("mixed target-motion strategies need an explicit contract")
        target_motion = next(iter(motions))
    if episode is None:
        if len(episodes) != 1:
            raise ValueError("mixed episode strategies need an explicit contract")
        episode = next(iter(episodes))
    retained = {value.name for value in strategies} | {
        retained for value in strategies for retained in value.retained_strategy_ids
    }
    return ArenaTrainingStrategy(
        name,
        tuple(objectives.values()),
        target_motion,
        episode,
        tuple(retained),
    )


__all__ = [
    "AIM_OBJECTIVE",
    "ARENA_TRAINING_STRATEGY_SCHEMA",
    "ATTACK_TIMING_OBJECTIVE",
    "ArenaTrainingStrategy",
    "CausalObjective",
    "EpisodeContract",
    "GUARD_TIMING_OBJECTIVE",
    "MINIMUM_SKILL_EPISODE_TICKS",
    "PURSUIT_OBJECTIVE",
    "SkillAxis",
    "TargetMotionContract",
    "TargetMotionPattern",
    "aim_training_strategy",
    "attack_timing_training_strategy",
    "compose_training_strategies",
    "guard_timing_training_strategy",
    "moving_target_motion",
    "pursuit_training_strategy",
    "radial_strafe_target_motion",
    "radial_target_motion",
    "stationary_target_motion",
    "strafe_target_motion",
]
