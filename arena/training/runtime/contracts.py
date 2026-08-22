"""Composable, versioned contracts for automatic JAX training."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

from arena.params import Domain, describe_space


TRAINING_CONTRACT_SCHEMA = "arena_recursive_training_contract"
TRAINING_CONTRACT_VERSION = 2
TRAINING_FEATURE_SCHEMA = "arena_training_feature_v1"
ENVIRONMENT_TRAINING_PROFILE_SCHEMA = "arena_environment_training_profile_v1"
_METRIC_ROLES = frozenset(
    {
        "objective",
        "reward",
        "progress",
        "outcome",
        "censoring",
        "support",
        "invalid",
        "loss",
        "stability",
        "diagnostic",
    }
)


def _freeze(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def _namespaced(value: str, label: str) -> None:
    if not value or "." not in value:
        raise ValueError(f"{label} must be namespaced, for example combat.progress")


@dataclass(frozen=True, slots=True)
class MetricSpec:
    name: str
    role: str
    direction: str = "maximize"
    weight: float = 1.0
    scale: float = 1.0
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if not self.name or self.role not in _METRIC_ROLES:
            raise ValueError("metric needs a name and a supported role")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("metric direction must be maximize or minimize")
        if not all(math.isfinite(value) for value in (self.weight, self.scale)):
            raise ValueError("metric weight and scale must be finite")
        if self.weight < 0.0 or self.scale <= 0.0:
            raise ValueError("metric weight must be nonnegative and scale positive")
        for bound in (self.minimum, self.maximum):
            if bound is not None and not math.isfinite(bound):
                raise ValueError("metric bounds must be finite")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("metric minimum exceeds maximum")

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "direction": self.direction,
            "weight": self.weight,
            "scale": self.scale,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass(frozen=True, slots=True)
class StopLossSpec:
    """An environment-owned metric boundary with an explicit run action."""

    name: str
    metric: str
    comparison: str
    threshold: float
    action: str = "reject_candidate"

    def __post_init__(self) -> None:
        _namespaced(self.name, "stop loss")
        if not self.metric:
            raise ValueError("stop loss metric must be nonempty")
        if self.comparison not in {"above", "below"}:
            raise ValueError("stop loss comparison must be above or below")
        if not math.isfinite(self.threshold):
            raise ValueError("stop loss threshold must be finite")
        if self.action not in {"reject_candidate", "stop_training"}:
            raise ValueError("stop loss action must reject_candidate or stop_training")

    def triggered(self, value: float) -> bool:
        return value > self.threshold if self.comparison == "above" else value < self.threshold

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "comparison": self.comparison,
            "threshold": self.threshold,
            "action": self.action,
        }


@dataclass(frozen=True, slots=True)
class TrainingFeature:
    """Static contract contributed by one environment or training strategy."""

    key: str
    version: int
    parameter_space: Mapping[str, Domain] = field(default_factory=dict)
    metrics: tuple[MetricSpec, ...] = ()
    stop_losses: tuple[StopLossSpec, ...] = ()
    capabilities: frozenset[str] = frozenset()
    manifest: Mapping[str, Any] = field(default_factory=dict)
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _namespaced(self.key, "training feature key")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("training feature version must be a positive integer")
        if any(not isinstance(value, Domain) for value in self.parameter_space.values()):
            raise TypeError("feature parameter_space values must be arena.params domains")
        if len({metric.name for metric in self.metrics}) != len(self.metrics):
            raise ValueError("feature metric names must be unique")
        if len({item.name for item in self.stop_losses}) != len(self.stop_losses):
            raise ValueError("feature stop-loss names must be unique")
        metric_names = {metric.name for metric in self.metrics}
        missing = sorted({item.metric for item in self.stop_losses} - metric_names)
        if missing:
            raise ValueError(f"feature stop losses reference missing metrics: {missing}")
        if any("." not in name for name in self.extensions):
            raise ValueError("feature extension keys must be namespaced")
        object.__setattr__(self, "parameter_space", _freeze(self.parameter_space))
        object.__setattr__(self, "manifest", _freeze(self.manifest))
        object.__setattr__(self, "extensions", _freeze(self.extensions))

    def describe(self) -> dict[str, Any]:
        return {
            "schema": TRAINING_FEATURE_SCHEMA,
            "key": self.key,
            "version": self.version,
            "parameter_space": describe_space(self.parameter_space),
            "metrics": [metric.describe() for metric in self.metrics],
            "stop_losses": [item.describe() for item in self.stop_losses],
            "capabilities": sorted(self.capabilities),
            "manifest": dict(self.manifest),
            "extensions": dict(self.extensions),
        }


@dataclass(frozen=True, slots=True)
class EnvironmentTrainingProfile:
    """Pre-init feature set owned by one environment/minigame contract."""

    environment_contract_sha256: str
    features: tuple[TrainingFeature, ...] = ()

    def __post_init__(self) -> None:
        if not self.environment_contract_sha256:
            raise ValueError("environment contract hash must be nonempty")
        keys = [feature.key for feature in self.features]
        if len(set(keys)) != len(keys):
            raise ValueError("environment training feature keys must be unique")

    def add(self, feature: TrainingFeature) -> "EnvironmentTrainingProfile":
        if any(item.key == feature.key for item in self.features):
            raise ValueError(f"training feature {feature.key!r} already exists")
        return replace(self, features=self.features + (feature,))

    def replace(self, feature: TrainingFeature) -> "EnvironmentTrainingProfile":
        if not any(item.key == feature.key for item in self.features):
            raise KeyError(feature.key)
        return replace(
            self,
            features=tuple(
                feature if item.key == feature.key else item for item in self.features
            ),
        )

    def remove(self, key: str) -> "EnvironmentTrainingProfile":
        if not any(item.key == key for item in self.features):
            raise KeyError(key)
        return replace(self, features=tuple(item for item in self.features if item.key != key))

    def describe(self) -> dict[str, Any]:
        return {
            "schema": ENVIRONMENT_TRAINING_PROFILE_SCHEMA,
            "environment_contract_sha256": self.environment_contract_sha256,
            "features": [feature.describe() for feature in self.features],
        }


@dataclass(frozen=True, slots=True)
class TrainingContractBuilder:
    """Immutable pre-init contract other agents can import and specialize."""

    strategy: str
    agent_contract_sha256: str
    environments: tuple[EnvironmentTrainingProfile, ...] = ()
    shared_features: tuple[TrainingFeature, ...] = ()

    def __post_init__(self) -> None:
        if not self.strategy or not self.agent_contract_sha256:
            raise ValueError("strategy and agent contract hash must be nonempty")

    def add_environment(self, profile: EnvironmentTrainingProfile) -> "TrainingContractBuilder":
        if any(
            item.environment_contract_sha256 == profile.environment_contract_sha256
            for item in self.environments
        ):
            raise ValueError("environment training profile already exists")
        return replace(self, environments=self.environments + (profile,))

    def add(self, feature: TrainingFeature) -> "TrainingContractBuilder":
        if any(item.key == feature.key for item in self.shared_features):
            raise ValueError(f"shared training feature {feature.key!r} already exists")
        return replace(self, shared_features=self.shared_features + (feature,))

    def replace(self, feature: TrainingFeature) -> "TrainingContractBuilder":
        if not any(item.key == feature.key for item in self.shared_features):
            raise KeyError(feature.key)
        return replace(
            self,
            shared_features=tuple(
                feature if item.key == feature.key else item
                for item in self.shared_features
            ),
        )

    def build(self) -> "TrainingContract":
        if not self.environments:
            raise ValueError("at least one environment training profile is required")
        empty = [
            item.environment_contract_sha256
            for item in self.environments
            if not item.features
        ]
        if empty:
            raise ValueError(
                "each environment must declare at least one training feature: "
                + ", ".join(empty)
            )
        features = self.shared_features + tuple(
            feature for environment in self.environments for feature in environment.features
        )
        parameters = _merge_mapping(
            "parameter", ((name, domain) for feature in features for name, domain in feature.parameter_space.items())
        )
        metrics = _merge_named("metric", (metric for feature in features for metric in feature.metrics))
        stop_losses = _merge_named(
            "stop loss", (item for feature in features for item in feature.stop_losses)
        )
        extensions = _merge_mapping(
            "extension", ((name, value) for feature in features for name, value in feature.extensions.items())
        )
        return TrainingContract(
            strategy=self.strategy,
            agent_contract_sha256=self.agent_contract_sha256,
            environment_contract_sha256=tuple(
                item.environment_contract_sha256 for item in self.environments
            ),
            parameter_space=parameters,
            metrics=metrics,
            stop_losses=stop_losses,
            capabilities=frozenset(
                capability for feature in features for capability in feature.capabilities
            ),
            extensions=extensions,
            environment_profiles=self.environments,
            shared_features=self.shared_features,
        )


def _merge_mapping(label: str, pairs) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for name, value in pairs:
        if name in merged and merged[name] != value:
            raise ValueError(f"conflicting {label} contribution {name!r}")
        merged[name] = value
    return merged


def _merge_named(label: str, values) -> tuple[Any, ...]:
    merged: dict[str, Any] = {}
    for value in values:
        if value.name in merged and merged[value.name] != value:
            raise ValueError(f"conflicting {label} contribution {value.name!r}")
        merged[value.name] = value
    return tuple(merged.values())


@dataclass(frozen=True, slots=True)
class TrainingContract:
    """Final immutable training contract consumed after environment init."""

    strategy: str
    agent_contract_sha256: str
    environment_contract_sha256: tuple[str, ...]
    parameter_space: Mapping[str, Domain]
    metrics: tuple[MetricSpec, ...]
    stop_losses: tuple[StopLossSpec, ...] = ()
    capabilities: frozenset[str] = frozenset()
    extensions: Mapping[str, Any] = field(default_factory=dict)
    environment_profiles: tuple[EnvironmentTrainingProfile, ...] = ()
    shared_features: tuple[TrainingFeature, ...] = ()

    @classmethod
    def preinitialize(
        cls, *, strategy: str, agent_contract_sha256: str
    ) -> TrainingContractBuilder:
        """Create the importable pre-init base; no environment defaults are implied."""

        if not strategy or not agent_contract_sha256:
            raise ValueError("strategy and agent contract hash must be nonempty")
        return TrainingContractBuilder(strategy, agent_contract_sha256)

    def __post_init__(self) -> None:
        if not self.strategy:
            raise ValueError("strategy must be nonempty")
        if not self.environment_contract_sha256:
            raise ValueError("at least one environment contract is required")
        if not self.environment_profiles:
            raise ValueError("final contracts must be built from environment profiles")
        if self.environment_contract_sha256 != tuple(
            item.environment_contract_sha256 for item in self.environment_profiles
        ):
            raise ValueError("environment identities differ from their profiles")
        if not self.metrics or len({item.name for item in self.metrics}) != len(self.metrics):
            raise ValueError("metrics must be nonempty with unique names")
        if not any(item.role in {"objective", "outcome", "progress"} for item in self.metrics):
            raise ValueError("at least one non-reward objective metric is required")
        if any(not isinstance(value, Domain) for value in self.parameter_space.values()):
            raise TypeError("parameter_space values must be arena.params domains")
        if any("." not in name for name in self.extensions):
            raise ValueError("extension keys must be namespaced")
        metric_names = {metric.name for metric in self.metrics}
        if len({item.name for item in self.stop_losses}) != len(self.stop_losses):
            raise ValueError("stop-loss names must be unique")
        missing = sorted({item.metric for item in self.stop_losses} - metric_names)
        if missing:
            raise ValueError(f"stop losses reference missing metrics: {missing}")
        object.__setattr__(self, "parameter_space", _freeze(self.parameter_space))
        object.__setattr__(self, "extensions", _freeze(self.extensions))

    def describe(self) -> dict[str, Any]:
        return {
            "schema": TRAINING_CONTRACT_SCHEMA,
            "version": TRAINING_CONTRACT_VERSION,
            "minimum_reader_version": 2,
            "strategy": self.strategy,
            "agent_contract_sha256": self.agent_contract_sha256,
            "environment_contract_sha256": list(self.environment_contract_sha256),
            "parameter_space": describe_space(self.parameter_space),
            "metrics": [metric.describe() for metric in self.metrics],
            "stop_losses": [item.describe() for item in self.stop_losses],
            "capabilities": sorted(self.capabilities),
            "extensions": dict(self.extensions),
            "environment_profiles": [item.describe() for item in self.environment_profiles],
            "shared_features": [item.describe() for item in self.shared_features],
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.describe(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest().upper()

    def require_compatible(self, candidate: "TrainingContract") -> None:
        if self.sha256 != candidate.sha256:
            raise ValueError("automatic-training contracts differ")


def require_supported_contract(value: Mapping[str, Any]) -> None:
    """Fail clearly on future core semantics; extensions remain forward-safe."""

    if value.get("schema") != TRAINING_CONTRACT_SCHEMA:
        raise ValueError("automatic-training contract schema differs")
    version = value.get("version")
    minimum = value.get("minimum_reader_version", version)
    if not isinstance(version, int) or not isinstance(minimum, int):
        raise ValueError("automatic-training contract versions must be integers")
    if minimum > TRAINING_CONTRACT_VERSION or version > TRAINING_CONTRACT_VERSION:
        raise ValueError(
            f"contract v{version} needs a newer reader than v{TRAINING_CONTRACT_VERSION}"
        )


__all__ = [
    "ENVIRONMENT_TRAINING_PROFILE_SCHEMA",
    "EnvironmentTrainingProfile",
    "MetricSpec",
    "StopLossSpec",
    "TRAINING_CONTRACT_SCHEMA",
    "TRAINING_CONTRACT_VERSION",
    "TRAINING_FEATURE_SCHEMA",
    "TrainingContract",
    "TrainingContractBuilder",
    "TrainingFeature",
    "require_supported_contract",
]
