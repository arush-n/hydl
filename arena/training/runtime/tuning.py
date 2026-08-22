"""Small mixture-of-experts tuner over completed training artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Mapping, Sequence

import numpy as np

from arena.params import Boolean, Choice, Integer, Real, sample_params
from arena.training.runtime.audit import RewardAudit, TrialResult
from arena.training.runtime.contracts import TrainingContract


TUNING_SCHEMA = "arena_artifact_tuning_ensemble_v1"


@dataclass(frozen=True, slots=True)
class TuningProposal:
    parameters: tuple[Mapping[str, Any], ...]
    expert_weights: Mapping[str, float]
    candidates_considered: int

    def describe(self) -> dict[str, Any]:
        return {
            "schema": TUNING_SCHEMA,
            "parameters": [dict(item) for item in self.parameters],
            "expert_weights": dict(self.expert_weights),
            "candidates_considered": self.candidates_considered,
        }


def trial_utility(contract: TrainingContract, result: TrialResult) -> float:
    """Declared, scale-normalized utility; reward can never be the only signal."""

    total = 0.0
    for metric in contract.metrics:
        if metric.role == "diagnostic" or metric.weight == 0.0:
            continue
        value = float(result.metrics[metric.name]) / metric.scale
        total += metric.weight * (value if metric.direction == "maximize" else -value)
    return total


def _vector(contract: TrainingContract, parameters: Mapping[str, Any]) -> np.ndarray:
    values = []
    for name, domain in sorted(contract.parameter_space.items()):
        value = domain.validate(parameters[name])
        if isinstance(domain, Real):
            width = domain.high - domain.low
            values.append(0.0 if width == 0.0 else (value - domain.low) / width)
        elif isinstance(domain, Integer):
            width = domain.high - domain.low
            values.append(0.0 if width == 0 else (value - domain.low) / width)
        elif isinstance(domain, Boolean):
            values.append(float(value))
        elif isinstance(domain, Choice):
            denominator = max(1, len(domain.options) - 1)
            values.append(domain.options.index(value) / denominator)
        else:  # pragma: no cover - custom domains need an explicit projection
            raise TypeError(f"automatic tuning cannot project {type(domain).__name__}")
    return np.asarray(values, dtype=np.float64)


def _local_sample(contract, best, rng, radius):
    candidate = {}
    for name, domain in contract.parameter_space.items():
        center = best[name]
        if isinstance(domain, Real):
            width = domain.high - domain.low
            candidate[name] = domain.validate(
                min(
                    domain.high,
                    max(domain.low, center + rng.gauss(0.0, radius * width)),
                )
            )
        elif isinstance(domain, Integer):
            width = domain.high - domain.low
            value = round(center + rng.gauss(0.0, radius * width))
            candidate[name] = domain.validate(min(domain.high, max(domain.low, value)))
        else:
            candidate[name] = center if rng.random() > radius else domain.sample(rng)
    return candidate


def _standardize(value: np.ndarray) -> np.ndarray:
    deviation = float(np.std(value))
    return (
        np.zeros_like(value)
        if deviation < 1.0e-12
        else (value - np.mean(value)) / deviation
    )


def propose_parameters(
    contract: TrainingContract,
    history: Sequence[tuple[TrialResult, RewardAudit]],
    *,
    count: int,
    seed: int,
    pool_size: int = 256,
) -> TuningProposal:
    """Combine surrogate, local, and exploration experts deterministically."""

    if count < 1 or pool_size < count:
        raise ValueError("count must be positive and no larger than pool_size")
    rng = random.Random(seed)
    admitted = [row for row, audit in history if audit.promotable]
    utility = np.asarray([trial_utility(contract, row) for row in admitted])
    tried = {_parameter_key(row.parameters) for row, _ in history}
    best = None if not admitted else admitted[int(np.argmax(utility))].parameters
    radius = max(0.05, 0.35 / math.sqrt(len(admitted) + 1))

    candidates = []
    candidate_keys = set()
    for _ in range(pool_size * 20):
        if len(candidates) >= pool_size:
            break
        candidate = (
            sample_params(contract.parameter_space, rng)
            if best is None or len(candidates) % 2 == 0
            else _local_sample(contract, best, rng, radius)
        )
        key = _parameter_key(candidate)
        if key not in tried and key not in candidate_keys:
            candidates.append(candidate)
            candidate_keys.add(key)
    if len(candidates) < count:
        raise ValueError("parameter space has no unseen candidate capacity")

    candidate_x = np.stack([_vector(contract, item) for item in candidates])
    if not admitted:
        weights = {"surrogate": 0.0, "local": 0.0, "explore": 1.0}
        scores = np.arange(len(candidates), dtype=float)[::-1]
    else:
        x = np.stack([_vector(contract, row.parameters) for row in admitted])
        design = np.concatenate((np.ones((len(x), 1)), x, x * x), axis=1)
        candidate_design = np.concatenate(
            (np.ones((len(candidate_x), 1)), candidate_x, candidate_x * candidate_x),
            axis=1,
        )
        ridge = np.linalg.solve(
            design.T @ design + 1.0e-3 * np.eye(design.shape[1]),
            design.T @ utility,
        )
        surrogate = candidate_design @ ridge
        distances = np.linalg.norm(candidate_x[:, None] - x[None], axis=2)
        kernel = np.exp(-np.square(distances / max(radius, 0.05)))
        local = (kernel @ utility) / np.maximum(np.sum(kernel, axis=1), 1.0e-9)
        explore = np.min(distances, axis=1)

        surrogate_rmse = float(np.sqrt(np.mean(np.square(design @ ridge - utility))))
        if len(x) > 1:
            historical_distance = np.linalg.norm(x[:, None] - x[None], axis=2)
            np.fill_diagonal(historical_distance, np.inf)
            local_prediction = utility[np.argmin(historical_distance, axis=1)]
            local_rmse = float(np.sqrt(np.mean(np.square(local_prediction - utility))))
        else:
            local_rmse = 1.0
        raw_weights = np.asarray(
            [
                1.0 / (surrogate_rmse + 0.1),
                1.0 / (local_rmse + 0.1),
                1.0 / math.sqrt(len(x) + 1),
            ]
        )
        raw_weights /= np.sum(raw_weights)
        weights = dict(zip(("surrogate", "local", "explore"), raw_weights, strict=True))
        scores = sum(
            weights[name] * _standardize(score)
            for name, score in (
                ("surrogate", surrogate),
                ("local", local),
                ("explore", explore),
            )
        )
    selected = np.argsort(scores)[-count:][::-1]
    return TuningProposal(
        tuple(candidates[int(index)] for index in selected),
        weights,
        len(candidates),
    )


def _parameter_key(parameters: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple((name, repr(value)) for name, value in sorted(parameters.items()))


__all__ = ["TUNING_SCHEMA", "TuningProposal", "propose_parameters", "trial_utility"]
