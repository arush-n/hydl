"""Generic fail-closed checks for reward exploitation and invalid runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from arena.training.runtime.contracts import MetricSpec, TrainingContract


REWARD_AUDIT_SCHEMA = "arena_reward_exploitation_audit_v2"


@dataclass(frozen=True, slots=True)
class TrialResult:
    parameters: Mapping[str, Any]
    metrics: Mapping[str, float]
    samples: Mapping[str, Sequence[float]] = field(default_factory=dict)
    artifact: Path | None = None
    checkpoint: Path | None = None
    peak_vram_bytes: int | None = None
    stop_loss_flags: Mapping[str, bool] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AuditFlag:
    code: str
    severity: str
    detail: str
    stop_training: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "detail": self.detail,
            "stop_training": self.stop_training,
        }


@dataclass(frozen=True, slots=True)
class RewardAudit:
    flags: tuple[AuditFlag, ...]
    sample_count: int
    reward_progress_correlation: float | None

    @property
    def promotable(self) -> bool:
        return not any(flag.severity == "error" for flag in self.flags)

    def describe(self) -> dict[str, Any]:
        return {
            "schema": REWARD_AUDIT_SCHEMA,
            "promotable": self.promotable,
            "sample_count": self.sample_count,
            "reward_progress_correlation": self.reward_progress_correlation,
            "flags": [flag.describe() for flag in self.flags],
        }


def _oriented(values: np.ndarray, metric: MetricSpec) -> np.ndarray:
    sign = 1.0 if metric.direction == "maximize" else -1.0
    return sign * values / metric.scale


def _sample_signal(
    result: TrialResult,
    metrics: Sequence[MetricSpec],
) -> np.ndarray | None:
    arrays = []
    for metric in metrics:
        if metric.name not in result.samples:
            continue
        arrays.append(_oriented(np.asarray(result.samples[metric.name], float), metric))
    if not arrays:
        return None
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("audit sample arrays must share shape")
    return np.mean(np.stack(arrays), axis=0)


def audit_reward(
    contract: TrainingContract,
    result: TrialResult,
    history: Sequence[TrialResult] = (),
) -> RewardAudit:
    """Audit reward against declared ground-truth progress and guardrails."""

    specs = {metric.name: metric for metric in contract.metrics}
    missing = sorted(set(specs) - set(result.metrics))
    flags = []
    if missing:
        flags.append(AuditFlag("missing_metrics", "error", ", ".join(missing)))
    for name, value in result.metrics.items():
        if name in specs and not math.isfinite(float(value)):
            flags.append(AuditFlag("nonfinite_metric", "error", name))
    for metric in contract.metrics:
        value = result.metrics.get(metric.name)
        if value is None or not math.isfinite(float(value)):
            continue
        if metric.minimum is not None and value < metric.minimum:
            flags.append(AuditFlag("guardrail_below_minimum", "error", metric.name))
        if metric.maximum is not None and value > metric.maximum:
            flags.append(AuditFlag("guardrail_above_maximum", "error", metric.name))
    declared_stop_losses = {item.name: item for item in contract.stop_losses}
    unknown_stop_losses = sorted(set(result.stop_loss_flags) - set(declared_stop_losses))
    if unknown_stop_losses:
        flags.append(
            AuditFlag(
                "unknown_stop_loss_flags",
                "error",
                ", ".join(unknown_stop_losses),
            )
        )
    triggered_stop_losses = {
        name for name, triggered in result.stop_loss_flags.items() if triggered
    }
    for stop_loss in contract.stop_losses:
        value = result.metrics.get(stop_loss.metric)
        metric_triggered = (
            value is not None
            and math.isfinite(float(value))
            and stop_loss.triggered(float(value))
        )
        if metric_triggered or stop_loss.name in triggered_stop_losses:
            evidence = (
                f"{stop_loss.metric}={float(value):.9g} "
                f"{stop_loss.comparison} {stop_loss.threshold:.9g}"
                if metric_triggered
                else "device feature emitted true"
            )
            flags.append(
                AuditFlag(
                    f"stop_loss.{stop_loss.name}",
                    "error",
                    evidence,
                    stop_loss.action == "stop_training",
                )
            )

    rewards = [metric for metric in contract.metrics if metric.role == "reward"]
    truth = [
        metric
        for metric in contract.metrics
        if metric.role in {"progress", "outcome", "objective"}
    ]
    reward_samples = _sample_signal(result, rewards)
    truth_samples = _sample_signal(result, truth)
    correlation = None
    sample_count = 0
    if reward_samples is not None and truth_samples is not None:
        if reward_samples.shape != truth_samples.shape:
            raise ValueError("reward and progress samples must share shape")
        sample_count = int(reward_samples.size)
        positive_without_progress = (reward_samples > 1.0e-6) & (
            np.abs(truth_samples) <= 1.0e-6
        )
        violation_fraction = float(np.mean(positive_without_progress))
        if sample_count >= 16 and violation_fraction > 0.01:
            flags.append(
                AuditFlag(
                    "positive_reward_without_progress",
                    "error",
                    f"{violation_fraction:.6f} of samples",
                )
            )
        if (
            sample_count >= 16
            and np.std(reward_samples) > 1.0e-8
            and np.std(truth_samples) > 1.0e-8
        ):
            correlation = float(np.corrcoef(reward_samples, truth_samples)[0, 1])
            if correlation < -0.20:
                flags.append(
                    AuditFlag(
                        "reward_opposes_progress",
                        "error",
                        f"correlation={correlation:.6f}",
                    )
                )
            elif correlation < 0.05:
                flags.append(
                    AuditFlag(
                        "reward_progress_weak",
                        "warning",
                        f"correlation={correlation:.6f}",
                    )
                )

    if history and rewards and truth:
        previous = history[-1]
        reward_gain = np.mean(
            [
                _oriented(np.asarray(result.metrics[item.name]), item)
                - _oriented(np.asarray(previous.metrics[item.name]), item)
                for item in rewards
                if item.name in result.metrics and item.name in previous.metrics
            ]
            or [0.0]
        )
        truth_gain = np.mean(
            [
                _oriented(np.asarray(result.metrics[item.name]), item)
                - _oriented(np.asarray(previous.metrics[item.name]), item)
                for item in truth
                if item.name in result.metrics and item.name in previous.metrics
            ]
            or [0.0]
        )
        if reward_gain > 0.05 and truth_gain < -0.05:
            flags.append(
                AuditFlag(
                    "reward_improves_while_outcome_regresses",
                    "error",
                    f"reward={reward_gain:.6f}, ground_truth={truth_gain:.6f}",
                )
            )

    return RewardAudit(tuple(flags), sample_count, correlation)


__all__ = [
    "AuditFlag",
    "REWARD_AUDIT_SCHEMA",
    "RewardAudit",
    "TrialResult",
    "audit_reward",
]
