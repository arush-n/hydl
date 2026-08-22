"""Static composition of environment-owned feature kernels into one JAX program."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import jax

from arena.training.runtime.contracts import TrainingFeature


JAX_TRAINING_FEATURE_PROGRAM_SCHEMA = "arena_jax_training_feature_program_v1"


class FeatureStep(NamedTuple):
    """One feature's device state, metric values, and instantaneous stop flags."""

    state: Any
    metrics: tuple[Any, ...]
    stop_losses: tuple[Any, ...]


class ComposedFeatureStep(NamedTuple):
    """Flat device output with names pinned in :class:`JaxFeatureProgram`."""

    states: tuple[Any, ...]
    metrics: tuple[Any, ...]
    stop_losses: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class JaxTrainingFeature:
    """A pure-JAX implementation for one static :class:`TrainingFeature`."""

    contract: TrainingFeature
    initialize: Callable[[Any], Any]
    step: Callable[[Any, Any], FeatureStep]

    def __post_init__(self) -> None:
        if not callable(self.initialize) or not callable(self.step):
            raise TypeError("JAX training feature functions must be callable")


@dataclass(frozen=True, slots=True)
class JaxFeatureProgram:
    """Already-composed functions; Python feature dispatch occurs only at trace time."""

    feature_keys: tuple[str, ...]
    metric_names: tuple[str, ...]
    stop_loss_names: tuple[str, ...]
    initialize: Callable[[Any], tuple[Any, ...]]
    step: Callable[[tuple[Any, ...], Any], ComposedFeatureStep]

    def describe(self) -> dict[str, Any]:
        return {
            "schema": JAX_TRAINING_FEATURE_PROGRAM_SCHEMA,
            "feature_keys": list(self.feature_keys),
            "metric_names": list(self.metric_names),
            "stop_loss_names": list(self.stop_loss_names),
            "composition": "static_python_unroll_then_single_jax_trace",
        }


def compose_jax_features(
    features: tuple[JaxTrainingFeature, ...],
    *,
    compile: bool = True,
) -> JaxFeatureProgram:
    """Fuse environment features without callbacks or dispatch inside device loops."""

    if not isinstance(features, tuple):
        raise TypeError("features must be a tuple so composition is static")
    keys = tuple(feature.contract.key for feature in features)
    if len(set(keys)) != len(keys):
        raise ValueError("JAX training feature keys must be unique")
    metric_names = tuple(
        metric.name for feature in features for metric in feature.contract.metrics
    )
    stop_names = tuple(
        item.name for feature in features for item in feature.contract.stop_losses
    )
    if len(set(metric_names)) != len(metric_names):
        raise ValueError("composed JAX metric names must be unique")
    if len(set(stop_names)) != len(stop_names):
        raise ValueError("composed JAX stop-loss names must be unique")

    def initialize(context: Any) -> tuple[Any, ...]:
        return tuple(feature.initialize(context) for feature in features)

    def step(states: tuple[Any, ...], transition: Any) -> ComposedFeatureStep:
        if len(states) != len(features):
            raise ValueError("feature state count differs from the compiled program")
        results = tuple(
            feature.step(state, transition)
            for feature, state in zip(features, states, strict=True)
        )
        for feature, result in zip(features, results, strict=True):
            if not isinstance(result, FeatureStep):
                raise TypeError(f"{feature.contract.key} must return FeatureStep")
            if len(result.metrics) != len(feature.contract.metrics):
                raise ValueError(f"{feature.contract.key} metric count differs")
            if len(result.stop_losses) != len(feature.contract.stop_losses):
                raise ValueError(f"{feature.contract.key} stop-loss count differs")
        return ComposedFeatureStep(
            tuple(result.state for result in results),
            tuple(value for result in results for value in result.metrics),
            tuple(value for result in results for value in result.stop_losses),
        )

    return JaxFeatureProgram(
        keys,
        metric_names,
        stop_names,
        jax.jit(initialize) if compile else initialize,
        jax.jit(step) if compile else step,
    )


__all__ = [
    "ComposedFeatureStep",
    "FeatureStep",
    "JAX_TRAINING_FEATURE_PROGRAM_SCHEMA",
    "JaxFeatureProgram",
    "JaxTrainingFeature",
    "compose_jax_features",
]
