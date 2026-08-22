"""Composable contracts for future Hytale agent systems."""

from hytalegym.framework.backend import (
    ActorObservationCompiler,
    HytaleBackend,
    PrivilegedSceneCompiler,
)
from hytalegym.framework.composition import (
    CompositionState,
    ProviderContract,
    compose_provider_contracts,
)
from hytalegym.framework.contracts import (
    ActorObservationEnvelope,
    AuthoritativeSnapshot,
    CompatibilityError,
    ContentDigest,
    FrameworkContractError,
    PrivilegedSceneEnvelope,
    RuntimeIdentity,
    SchemaBundle,
    SchemaRef,
    SnapshotHeader,
)
from hytalegym.framework.registry import ComponentRegistry, RegistryError


__all__ = [
    "ActorObservationEnvelope",
    "ActorObservationCompiler",
    "AuthoritativeSnapshot",
    "CompatibilityError",
    "ComponentRegistry",
    "CompositionState",
    "ContentDigest",
    "FrameworkContractError",
    "HytaleBackend",
    "PrivilegedSceneEnvelope",
    "PrivilegedSceneCompiler",
    "ProviderContract",
    "RegistryError",
    "RuntimeIdentity",
    "SchemaBundle",
    "SchemaRef",
    "SnapshotHeader",
    "compose_provider_contracts",
]
