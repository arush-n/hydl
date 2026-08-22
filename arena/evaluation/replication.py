"""Replicated held-out promotion gates for any Arena agent."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean, stdev
from types import MappingProxyType
from typing import Any, Mapping, Sequence


REPORT_SCHEMA = "hytalerl_arena_replication_assessment_v2"
_T95 = (
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


@dataclass(frozen=True, slots=True)
class ArmResult:
    """Exact episode counts for one policy arm."""

    successes: int
    episodes: int

    def __post_init__(self) -> None:
        for name in ("successes", "episodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.episodes < 1 or not 0 <= self.successes <= self.episodes:
            raise ValueError("successes must be in [0, episodes] and episodes positive")

    @property
    def rate(self) -> float:
        return self.successes / self.episodes

    def describe(self) -> dict[str, int | float]:
        return {
            "successes": self.successes,
            "episodes": self.episodes,
            "success_rate": self.rate,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReplicate:
    """One independently trained policy evaluated on a named world split."""

    task: str
    training_seed: int
    evaluation_seed: int
    split: str
    arms: Mapping[str, ArmResult]

    def __post_init__(self) -> None:
        if not self.task or self.task != self.task.strip():
            raise ValueError("task must be a non-empty label")
        for name in ("training_seed", "evaluation_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if not self.split or self.split != self.split.strip():
            raise ValueError("split must be a non-empty label")
        arms = dict(self.arms)
        if not arms or any(
            not name or name != name.strip() or not isinstance(result, ArmResult)
            for name, result in arms.items()
        ):
            raise ValueError("arms must map non-empty labels to ArmResult values")
        object.__setattr__(self, "arms", MappingProxyType(arms))

    def describe(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "training_seed": self.training_seed,
            "evaluation_seed": self.evaluation_seed,
            "split": self.split,
            "arms": {name: result.describe() for name, result in self.arms.items()},
        }


@dataclass(frozen=True, slots=True)
class PromotionGate:
    """A conservative, task-wise definition of a promotable policy."""

    candidate_arm: str = "after_ppo"
    baseline_arms: tuple[str, ...] = ("ppo_only", "before_il")
    non_regression_arms: tuple[str, ...] = ()
    required_split: str = "heldout"
    minimum_tasks: int = 2
    minimum_replicates_per_task: int = 3
    minimum_episodes_per_arm: int = 64
    minimum_success_rate: float = 0.70
    minimum_advantage: float = 0.05

    def __post_init__(self) -> None:
        names = (
            self.candidate_arm,
            *self.baseline_arms,
            *self.non_regression_arms,
        )
        if any(not name or name != name.strip() for name in names):
            raise ValueError("arm names must be non-empty labels")
        if len(set(names)) != len(names):
            raise ValueError("candidate and baseline arm names must be unique")
        if (
            not self.required_split
            or self.required_split != self.required_split.strip()
        ):
            raise ValueError("required_split must be a non-empty label")
        for name in (
            "minimum_tasks",
            "minimum_replicates_per_task",
            "minimum_episodes_per_arm",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.minimum_replicates_per_task < 2:
            raise ValueError("a promotion gate needs at least two training seeds")
        for name in ("minimum_success_rate", "minimum_advantage"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def describe(self) -> dict[str, Any]:
        return {
            "candidate_arm": self.candidate_arm,
            "baseline_arms": list(self.baseline_arms),
            "non_regression_arms": list(self.non_regression_arms),
            "required_split": self.required_split,
            "minimum_tasks": self.minimum_tasks,
            "minimum_replicates_per_task": self.minimum_replicates_per_task,
            "minimum_episodes_per_arm": self.minimum_episodes_per_arm,
            "minimum_success_rate": self.minimum_success_rate,
            "minimum_advantage": self.minimum_advantage,
            "non_regression_margin": 0.0,
            "confidence": 0.95,
            "replication_unit": "independently_trained_policy_seed",
            "pooling": "none; every named task must pass independently",
        }


def assess_promotion(
    replicates: Sequence[EvaluationReplicate],
    gate: PromotionGate = PromotionGate(),
) -> dict[str, Any]:
    """Assess a policy without treating lanes or tasks as seed replicates."""

    rows = tuple(replicates)
    if not rows:
        raise ValueError("replicates must not be empty")
    _validate_rows(rows, gate)
    by_task = {
        task: tuple(row for row in rows if row.task == task)
        for task in sorted({row.task for row in rows})
    }
    coverage_reasons = []
    if len(by_task) < gate.minimum_tasks:
        coverage_reasons.append(
            f"need {gate.minimum_tasks} tasks; observed {len(by_task)}"
        )
    for task, task_rows in by_task.items():
        if len(task_rows) < gate.minimum_replicates_per_task:
            coverage_reasons.append(
                f"{task}: need {gate.minimum_replicates_per_task} training seeds; "
                f"observed {len(task_rows)}"
            )
        for row in task_rows:
            for arm in (
                gate.candidate_arm,
                *gate.baseline_arms,
                *gate.non_regression_arms,
            ):
                episodes = row.arms[arm].episodes
                if episodes < gate.minimum_episodes_per_arm:
                    coverage_reasons.append(
                        f"{task}/seed={row.training_seed}/{arm}: need "
                        f"{gate.minimum_episodes_per_arm} episodes; observed {episodes}"
                    )

    task_reports = {
        task: _task_report(task_rows, gate) for task, task_rows in by_task.items()
    }
    failures = []
    if not coverage_reasons:
        for task, report in task_reports.items():
            candidate_lower = report["candidate"]["confidence_interval_95"][0]
            if candidate_lower < gate.minimum_success_rate:
                failures.append(
                    f"{task}: candidate lower confidence bound "
                    f"{candidate_lower:.3f} < {gate.minimum_success_rate:.3f}"
                )
            margins = {
                **dict.fromkeys(gate.baseline_arms, gate.minimum_advantage),
                **dict.fromkeys(gate.non_regression_arms, 0.0),
            }
            for baseline, summary in report["advantage_over_baseline"].items():
                lower = summary["confidence_interval_95"][0]
                margin = margins[baseline]
                if lower < margin:
                    failures.append(
                        f"{task}: advantage over {baseline} lower confidence bound "
                        f"{lower:.3f} < {margin:.3f}"
                    )

    verdict = "insufficient" if coverage_reasons else ("fail" if failures else "pass")
    candidate_task_means = [
        report["candidate"]["mean"] for report in task_reports.values()
    ]
    baseline_macro = {
        baseline: mean(
            report["advantage_over_baseline"][baseline]["mean"]
            for report in task_reports.values()
        )
        for baseline in (*gate.baseline_arms, *gate.non_regression_arms)
    }
    ceiling_baselines = {
        task: report["diagnostics"]["baseline_arms_at_ceiling"]
        for task, report in task_reports.items()
        if report["diagnostics"]["baseline_arms_at_ceiling"]
    }
    return {
        "schema": REPORT_SCHEMA,
        "verdict": verdict,
        "gate": gate.describe(),
        "coverage": {
            "tasks": list(by_task),
            "training_seeds_by_task": {
                task: [row.training_seed for row in task_rows]
                for task, task_rows in by_task.items()
            },
            "replicate_count": len(rows),
            "sufficient": not coverage_reasons,
        },
        "project_macro": {
            "candidate_success_rate": mean(candidate_task_means),
            "advantage_over_baseline": baseline_macro,
        },
        "tasks": task_reports,
        "diagnostics": {
            "tasks_with_ceiling_baselines": ceiling_baselines,
            "promotion_verdict_uses_diagnostics": False,
        },
        "reasons": [*coverage_reasons, *failures],
        "replicates": [row.describe() for row in rows],
    }


def _validate_rows(rows: tuple[EvaluationReplicate, ...], gate: PromotionGate) -> None:
    identities = [(row.task, row.training_seed) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("each task/training_seed pair must appear exactly once")
    for row in rows:
        if row.split != gate.required_split:
            raise ValueError(
                f"{row.task}/seed={row.training_seed} uses split {row.split!r}; "
                f"expected {gate.required_split!r}"
            )
        missing = set(
            (
                gate.candidate_arm,
                *gate.baseline_arms,
                *gate.non_regression_arms,
            )
        ) - set(row.arms)
        if missing:
            raise ValueError(
                f"{row.task}/seed={row.training_seed} lacks arms {sorted(missing)}"
            )


def _task_report(
    rows: tuple[EvaluationReplicate, ...], gate: PromotionGate
) -> dict[str, Any]:
    arm_names = (
        gate.candidate_arm,
        *gate.baseline_arms,
        *gate.non_regression_arms,
    )
    arm_values = {arm: [row.arms[arm].rate for row in rows] for arm in arm_names}
    candidate = arm_values[gate.candidate_arm]
    advantages = {
        baseline: _summary(
            [
                row.arms[gate.candidate_arm].rate - row.arms[baseline].rate
                for row in rows
            ]
        )
        for baseline in (*gate.baseline_arms, *gate.non_regression_arms)
    }
    return {
        "replication_unit": "training_seed",
        "training_seeds": [row.training_seed for row in rows],
        "evaluation_seeds": [row.evaluation_seed for row in rows],
        "candidate": _summary(candidate),
        "arms": {arm: _summary(values) for arm, values in arm_values.items()},
        "advantage_over_baseline": advantages,
        "diagnostics": {
            "arms_at_ceiling": [
                arm
                for arm, values in arm_values.items()
                if all(value == 1.0 for value in values)
            ],
            "arms_at_floor": [
                arm
                for arm, values in arm_values.items()
                if all(value == 0.0 for value in values)
            ],
            "baseline_arms_at_ceiling": [
                arm
                for arm in gate.baseline_arms
                if all(value == 1.0 for value in arm_values[arm])
            ],
        },
    }


def _summary(values: Sequence[float]) -> dict[str, Any]:
    values = tuple(float(value) for value in values)
    center = mean(values)
    if len(values) < 2:
        interval: list[float] | None = None
        spread = None
    else:
        spread = stdev(values)
        critical = _T95[min(len(values) - 2, len(_T95) - 1)]
        radius = critical * spread / sqrt(len(values))
        interval = [center - radius, center + radius]
    return {
        "n_training_seeds": len(values),
        "mean": center,
        "sample_standard_deviation": spread,
        "confidence_interval_95": interval,
        "values": list(values),
    }


__all__ = [
    "REPORT_SCHEMA",
    "ArmResult",
    "EvaluationReplicate",
    "PromotionGate",
    "assess_promotion",
]
