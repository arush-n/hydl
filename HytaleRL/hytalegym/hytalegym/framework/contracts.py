"""Versioned host-side identities and deliberately separate data channels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import operator
from typing import Any, Generic, TypeVar


PayloadT = TypeVar("PayloadT")


class FrameworkContractError(ValueError):
    """A framework value is malformed or internally inconsistent."""


class CompatibilityError(FrameworkContractError):
    """Two exact schema or capability contracts cannot be composed."""


@dataclass(frozen=True, slots=True)
class ContentDigest:
    """A normalized fixed-length content digest without a fixed algorithm."""

    algorithm: str
    value: str

    def __post_init__(self) -> None:
        algorithm = _text(self.algorithm, "digest algorithm").lower()
        try:
            digest_size = hashlib.new(algorithm).digest_size
        except ValueError as error:
            raise FrameworkContractError(
                f"unknown digest algorithm {algorithm!r}"
            ) from error
        value = _text(self.value, "digest value").lower()
        if digest_size <= 0 or len(value) != digest_size * 2:
            raise FrameworkContractError(
                f"{algorithm} digest must contain {digest_size * 2} hex characters"
            )
        if any(character not in "0123456789abcdef" for character in value):
            raise FrameworkContractError("digest value must be hexadecimal")
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "value", value)

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
        *,
        algorithm: str = "sha256",
    ) -> ContentDigest:
        digest = hashlib.new(algorithm)
        digest.update(payload)
        return cls(algorithm=algorithm, value=digest.hexdigest())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ContentDigest:
        mapping = _mapping(value, "content digest")
        return cls(
            algorithm=mapping.get("algorithm"),
            value=mapping.get("value"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"algorithm": self.algorithm, "value": self.value}


@dataclass(frozen=True, slots=True)
class SchemaRef:
    """Exact identity of one logical schema and manifest."""

    name: str
    version: int
    digest: ContentDigest

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "schema name"))
        object.__setattr__(
            self,
            "version",
            _positive_int(self.version, "schema version"),
        )
        if not isinstance(self.digest, ContentDigest):
            raise TypeError("schema digest must be a ContentDigest")

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, Any],
        *,
        algorithm: str = "sha256",
    ) -> SchemaRef:
        mapping = _mapping(manifest, "schema manifest")
        encoded = json.dumps(
            mapping,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return cls(
            name=mapping.get("schema"),
            version=mapping.get("version"),
            digest=ContentDigest.from_bytes(encoded, algorithm=algorithm),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SchemaRef:
        mapping = _mapping(value, "schema reference")
        return cls(
            name=mapping.get("name"),
            version=mapping.get("version"),
            digest=ContentDigest.from_mapping(mapping.get("digest")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "digest": self.digest.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SchemaBundle:
    """Role-to-schema mapping with deterministic order and exact matching."""

    entries: tuple[tuple[str, SchemaRef], ...] = ()

    def __post_init__(self) -> None:
        normalized: list[tuple[str, SchemaRef]] = []
        seen: set[str] = set()
        for entry in self.entries:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError("schema bundle entries must be (role, schema) pairs")
            role = _text(entry[0], "schema role")
            schema = entry[1]
            if not isinstance(schema, SchemaRef):
                raise TypeError("schema bundle values must be SchemaRef instances")
            if role in seen:
                raise FrameworkContractError(f"duplicate schema role {role!r}")
            seen.add(role)
            normalized.append((role, schema))
        object.__setattr__(self, "entries", tuple(sorted(normalized)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, SchemaRef]) -> SchemaBundle:
        mapping = _mapping(value, "schema bundle")
        return cls(tuple(mapping.items()))

    def get(self, role: str) -> SchemaRef | None:
        key = _text(role, "schema role")
        return dict(self.entries).get(key)

    def require(
        self,
        required: SchemaBundle,
        *,
        consumer: str,
    ) -> None:
        if not isinstance(required, SchemaBundle):
            raise TypeError("required schemas must be a SchemaBundle")
        errors: list[str] = []
        available = dict(self.entries)
        for role, expected in required.entries:
            actual = available.get(role)
            if actual is None:
                errors.append(f"missing role {role!r}")
            elif actual != expected:
                errors.append(
                    f"role {role!r} is {actual.name}@{actual.version} "
                    f"({actual.digest.value}), expected "
                    f"{expected.name}@{expected.version} "
                    f"({expected.digest.value})"
                )
        if errors:
            raise CompatibilityError(
                f"{_text(consumer, 'consumer')} schema requirements failed: "
                + "; ".join(errors)
            )

    def merge(self, produced: SchemaBundle, *, producer: str) -> SchemaBundle:
        if not isinstance(produced, SchemaBundle):
            raise TypeError("produced schemas must be a SchemaBundle")
        merged = dict(self.entries)
        for role, schema in produced.entries:
            existing = merged.get(role)
            if existing is not None and existing != schema:
                raise CompatibilityError(
                    f"{_text(producer, 'producer')} cannot replace role {role!r} "
                    "with an incompatible schema"
                )
            merged[role] = schema
        return SchemaBundle.from_mapping(merged)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {role: schema.to_dict() for role, schema in self.entries}


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    """Runtime provenance required before artifacts can be compared or reused."""

    game_version: str
    api_version: str
    asset_bundle: ContentDigest
    ruleset: SchemaRef

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "game_version",
            _text(self.game_version, "game version"),
        )
        object.__setattr__(
            self,
            "api_version",
            _text(self.api_version, "API version"),
        )
        if not isinstance(self.asset_bundle, ContentDigest):
            raise TypeError("asset_bundle must be a ContentDigest")
        if not isinstance(self.ruleset, SchemaRef):
            raise TypeError("ruleset must be a SchemaRef")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RuntimeIdentity:
        mapping = _mapping(value, "runtime identity")
        return cls(
            game_version=mapping.get("game_version"),
            api_version=mapping.get("api_version"),
            asset_bundle=ContentDigest.from_mapping(mapping.get("asset_bundle")),
            ruleset=SchemaRef.from_mapping(mapping.get("ruleset")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_version": self.game_version,
            "api_version": self.api_version,
            "asset_bundle": self.asset_bundle.to_dict(),
            "ruleset": self.ruleset.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SnapshotHeader:
    """Versioned immutable-tick metadata independent of snapshot payload shape."""

    schema: SchemaRef
    runtime: RuntimeIdentity
    world_id: int | str
    episode_id: int | str
    tick_id: int
    monotonic_time_ns: int
    dt_seconds: float
    controlled_agent_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.schema, SchemaRef):
            raise TypeError("snapshot schema must be a SchemaRef")
        if not isinstance(self.runtime, RuntimeIdentity):
            raise TypeError("snapshot runtime must be a RuntimeIdentity")
        object.__setattr__(
            self,
            "world_id",
            _stable_id(self.world_id, "world ID"),
        )
        object.__setattr__(
            self,
            "episode_id",
            _stable_id(self.episode_id, "episode ID"),
        )
        object.__setattr__(
            self,
            "tick_id",
            _nonnegative_int(self.tick_id, "tick ID"),
        )
        object.__setattr__(
            self,
            "monotonic_time_ns",
            _nonnegative_int(self.monotonic_time_ns, "monotonic time"),
        )
        delta = float(self.dt_seconds)
        if not math.isfinite(delta) or delta < 0.0:
            raise FrameworkContractError(
                "dt_seconds must be finite and non-negative"
            )
        object.__setattr__(self, "dt_seconds", delta)
        object.__setattr__(
            self,
            "controlled_agent_count",
            _nonnegative_int(
                self.controlled_agent_count,
                "controlled agent count",
            ),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SnapshotHeader:
        mapping = _mapping(value, "snapshot header")
        return cls(
            schema=SchemaRef.from_mapping(mapping.get("schema")),
            runtime=RuntimeIdentity.from_mapping(mapping.get("runtime")),
            world_id=mapping.get("world_id"),
            episode_id=mapping.get("episode_id"),
            tick_id=mapping.get("tick_id"),
            monotonic_time_ns=mapping.get("monotonic_time_ns"),
            dt_seconds=mapping.get("dt_seconds"),
            controlled_agent_count=mapping.get("controlled_agent_count"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema.to_dict(),
            "runtime": self.runtime.to_dict(),
            "world_id": self.world_id,
            "episode_id": self.episode_id,
            "tick_id": self.tick_id,
            "monotonic_time_ns": self.monotonic_time_ns,
            "dt_seconds": self.dt_seconds,
            "controlled_agent_count": self.controlled_agent_count,
        }


@dataclass(frozen=True, slots=True)
class AuthoritativeSnapshot(Generic[PayloadT]):
    """A copied tick snapshot; its payload is never an actor observation."""

    header: SnapshotHeader
    payload: PayloadT

    def __post_init__(self) -> None:
        if not isinstance(self.header, SnapshotHeader):
            raise TypeError("snapshot header must be a SnapshotHeader")


@dataclass(frozen=True, slots=True)
class PrivilegedSceneEnvelope(Generic[PayloadT]):
    """Training-only data product compiled from an authoritative snapshot."""

    source: SnapshotHeader
    schema: SchemaRef
    payload: PayloadT

    def __post_init__(self) -> None:
        _validate_envelope(self.source, self.schema)


@dataclass(frozen=True, slots=True)
class ActorObservationEnvelope(Generic[PayloadT]):
    """Deployment-legal data product with a physically distinct type."""

    source: SnapshotHeader
    schema: SchemaRef
    payload: PayloadT

    def __post_init__(self) -> None:
        _validate_envelope(self.source, self.schema)


def _validate_envelope(source: SnapshotHeader, schema: SchemaRef) -> None:
    if not isinstance(source, SnapshotHeader):
        raise TypeError("envelope source must be a SnapshotHeader")
    if not isinstance(schema, SchemaRef):
        raise TypeError("envelope schema must be a SchemaRef")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{label} must be a non-empty trimmed string")
    if any(ord(character) < 32 for character in value):
        raise FrameworkContractError(f"{label} cannot contain control characters")
    return value


def _stable_id(value: Any, label: str) -> int | str:
    if isinstance(value, str):
        return _text(value, label)
    return _integer(value, label)


def _positive_int(value: Any, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise FrameworkContractError(f"{label} must be positive")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    result = _integer(value, label)
    if result < 0:
        raise FrameworkContractError(f"{label} must be non-negative")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error


__all__ = [
    "ActorObservationEnvelope",
    "AuthoritativeSnapshot",
    "CompatibilityError",
    "ContentDigest",
    "FrameworkContractError",
    "PrivilegedSceneEnvelope",
    "RuntimeIdentity",
    "SchemaBundle",
    "SchemaRef",
    "SnapshotHeader",
]
