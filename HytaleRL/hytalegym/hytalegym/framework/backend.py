"""Backend-neutral execution boundary; no backend is implemented here."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from hytalegym.framework.composition import ProviderContract
from hytalegym.framework.contracts import (
    ActorObservationEnvelope,
    AuthoritativeSnapshot,
    PrivilegedSceneEnvelope,
)


TaskT = TypeVar("TaskT")
ResetT = TypeVar("ResetT")
SnapshotT = TypeVar("SnapshotT")
ActionT = TypeVar("ActionT")
MetricsT = TypeVar("MetricsT")
HistoryT = TypeVar("HistoryT")
PerceptionConfigT = TypeVar("PerceptionConfigT")
ActorObservationT = TypeVar("ActorObservationT")
PrivilegedSceneT = TypeVar("PrivilegedSceneT")


@runtime_checkable
class HytaleBackend(
    Protocol[TaskT, ResetT, SnapshotT, ActionT, MetricsT]
):
    """Logical interface shared by real, JAX, replay, and learned backends."""

    @property
    def contract(self) -> ProviderContract: ...

    def reset(self, task: TaskT) -> ResetT: ...

    def poll_snapshot(self) -> AuthoritativeSnapshot[SnapshotT]: ...

    def submit_actions(self, batch: ActionT) -> None: ...

    def get_metrics(self) -> MetricsT: ...


@runtime_checkable
class ActorObservationCompiler(
    Protocol[SnapshotT, HistoryT, PerceptionConfigT, ActorObservationT]
):
    """Compile deployment-legal evidence without a privileged-scene input."""

    @property
    def contract(self) -> ProviderContract: ...

    def compile(
        self,
        snapshot: AuthoritativeSnapshot[SnapshotT],
        legal_history: HistoryT,
        perception_config: PerceptionConfigT,
    ) -> ActorObservationEnvelope[ActorObservationT]: ...


@runtime_checkable
class PrivilegedSceneCompiler(Protocol[SnapshotT, PrivilegedSceneT]):
    """Compile a training-only scene directly from an authoritative snapshot."""

    @property
    def contract(self) -> ProviderContract: ...

    def compile(
        self,
        snapshot: AuthoritativeSnapshot[SnapshotT],
    ) -> PrivilegedSceneEnvelope[PrivilegedSceneT]: ...


__all__ = [
    "ActorObservationCompiler",
    "HytaleBackend",
    "PrivilegedSceneCompiler",
]
