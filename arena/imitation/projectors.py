"""Fail-closed, versioned feature projection over composable traces."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from arena.imitation.contract import TraceComponent, TraceSequence


Check = Callable[[TraceSequence], bool]
Project = Callable[[TraceSequence], "TraceProjection"]
_MISSING = object()


@dataclass(frozen=True, slots=True)
class TraceRequirement:
    """One auditable precondition that must hold before projection."""

    name: str
    check: Check
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not callable(self.check):
            raise ValueError("trace requirements need a name and callable check")

    def require(self, trace: TraceSequence) -> None:
        try:
            satisfied = bool(self.check(trace))
        except Exception as error:
            raise ValueError(f"trace requirement failed: {self.name}") from error
        if not satisfied:
            raise ValueError(f"trace requirement failed: {self.name}")

    def manifest(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description}


@dataclass(frozen=True, slots=True)
class TraceProjection:
    """Components and provenance emitted by one projector."""

    components: tuple[TraceComponent, ...]
    parameters: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "parameters", dict(self.parameters or {}))


@dataclass(frozen=True, slots=True)
class TraceProjector:
    """A named transformation with explicit inputs and supervised outputs."""

    name: str
    version: int
    requires: tuple[TraceRequirement, ...]
    supervises: tuple[str, ...]
    project: Project

    def __post_init__(self) -> None:
        object.__setattr__(self, "requires", tuple(self.requires))
        object.__setattr__(self, "supervises", tuple(self.supervises))
        if "." not in self.name or isinstance(self.version, bool) or self.version < 1:
            raise ValueError("projectors need a namespaced name and positive version")
        if not self.supervises or len(self.supervises) != len(set(self.supervises)):
            raise ValueError("projector outputs must be nonempty and unique")
        if not callable(self.project):
            raise TypeError("project must be callable")

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "requires": tuple(item.manifest() for item in self.requires),
            "supervises": self.supervises,
        }


class TraceProjectorRegistry:
    """Small mutable registry; projection results remain immutable traces."""

    def __init__(self, projectors: Sequence[TraceProjector] = ()) -> None:
        self._projectors: dict[str, TraceProjector] = {}
        for projector in projectors:
            self.register(projector)

    def register(self, projector: TraceProjector) -> TraceProjector:
        if projector.name in self._projectors:
            raise ValueError(f"projector already registered: {projector.name}")
        self._projectors[projector.name] = projector
        return projector

    def compose(self, trace: TraceSequence, *names: str) -> TraceSequence:
        selected = tuple(self._get(name) for name in names)
        if not selected:
            raise ValueError("at least one projector is required")
        declared = [name for item in selected for name in item.supervises]
        if len(declared) != len(set(declared)):
            raise ValueError("selected projectors supervise overlapping outputs")

        components: list[TraceComponent] = []
        parameters: dict[str, Any] = {}
        for projector in selected:
            for requirement in projector.requires:
                requirement.require(trace)
            result = projector.project(trace)
            if not isinstance(result, TraceProjection):
                raise TypeError(f"{projector.name} did not return TraceProjection")
            emitted = tuple(item.spec.name for item in result.components)
            if emitted != projector.supervises:
                raise ValueError(
                    f"{projector.name} emitted {emitted}, expected {projector.supervises}"
                )
            if any(
                item.spec.name.startswith("reward.")
                and (item.spec.kind != "event" or item.spec.shape != ())
                for item in result.components
            ):
                raise ValueError("reward projector outputs must be scalar events")
            overlap = set(parameters) & set(result.parameters)
            if overlap:
                raise ValueError(f"projector parameters overlap: {sorted(overlap)}")
            components.extend(result.components)
            parameters.update(result.parameters)

        manifest = tuple(item.manifest() for item in selected)
        parameters["projection.manifest"] = manifest
        parameters["projection.manifest_sha256"] = _sha256(manifest)
        return trace.compose(*components, parameters=parameters)

    def manifest(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            self._projectors[name].manifest() for name in sorted(self._projectors)
        )

    def _get(self, name: str) -> TraceProjector:
        try:
            return self._projectors[name]
        except KeyError as error:
            raise KeyError(f"unknown trace projector: {name}") from error


def apply_episode_structure(
    trace: TraceSequence,
    *,
    name: str,
    version: int,
    episode_id: Any,
    episode_start: Any,
    terminated: Any,
    truncated: Any,
    episode_id_available: Any,
    episode_start_available: Any,
    terminated_available: Any,
    truncated_available: Any,
) -> TraceSequence:
    """Install capture-derived episode columns under a versioned projector identity."""

    if "." not in name or isinstance(version, bool) or version < 1:
        raise ValueError(
            "episode projectors need a namespaced name and positive version"
        )
    manifest = {
        "name": name,
        "version": version,
        "supervises": (
            "episode_id",
            "episode_start",
            "terminated",
            "truncated",
        ),
    }
    return trace.with_episode_structure(
        episode_id=episode_id,
        episode_start=episode_start,
        terminated=terminated,
        truncated=truncated,
        episode_id_available=episode_id_available,
        episode_start_available=episode_start_available,
        terminated_available=terminated_available,
        truncated_available=truncated_available,
        parameters={
            "episode.projection.manifest": manifest,
            "episode.projection.manifest_sha256": _sha256(manifest),
        },
    )


def install_episode_evidence(
    trace: TraceSequence,
    *,
    name: str,
    version: int,
    episode_start: Any,
    terminated: Any,
    truncated: Any,
    episode_start_available: Any,
    terminated_available: Any,
    truncated_available: Any,
    evidence: Sequence[str],
    derivation: str,
    initial_episode_id: int | None = None,
) -> TraceSequence:
    """Derive local episode IDs from authoritative capture boundary evidence."""

    if not evidence or not all(isinstance(item, str) and item for item in evidence):
        raise ValueError("episode evidence must name at least one capture field")
    if not isinstance(derivation, str) or not derivation:
        raise ValueError("episode derivation must be named")
    if initial_episode_id is not None and (
        isinstance(initial_episode_id, bool)
        or not isinstance(initial_episode_id, int)
        or not 0 <= initial_episode_id <= np.iinfo(np.int64).max
    ):
        raise ValueError("initial_episode_id must be a nonnegative int64")

    starts = _evidence_rows(
        episode_start,
        episode_start_available,
        trace.rows,
        "episode_start",
    )
    ends = _evidence_rows(
        terminated,
        terminated_available,
        trace.rows,
        "terminated",
    )
    cuts = _evidence_rows(
        truncated,
        truncated_available,
        trace.rows,
        "truncated",
    )
    if np.any(ends[0] & cuts[0]):
        raise ValueError("a transition cannot be both terminated and truncated")
    episode_id, episode_id_available = _derive_episode_ids(
        starts[0],
        starts[1],
        ends[0],
        ends[1],
        cuts[0],
        cuts[1],
        initial_episode_id,
    )
    installed = apply_episode_structure(
        trace,
        name=name,
        version=version,
        episode_id=episode_id,
        episode_start=starts[0],
        terminated=ends[0],
        truncated=cuts[0],
        episode_id_available=episode_id_available,
        episode_start_available=starts[1],
        terminated_available=ends[1],
        truncated_available=cuts[1],
    )
    contract = {
        "name": name,
        "version": version,
        "evidence": tuple(evidence),
        "derivation": derivation,
        "initial_episode_id": initial_episode_id,
        "unknown_boundary": "episode_id_unavailable_until_next_known_start",
    }
    return installed.compose(
        parameters={
            "episode.evidence.fields": tuple(evidence),
            "episode.evidence.derivation": derivation,
            "episode.evidence.initial_episode_id": initial_episode_id,
            "episode.evidence.contract_sha256": _sha256(contract),
        }
    )


def component_requirement(
    reference: str, *, require_all_valid: bool = False
) -> TraceRequirement:
    """Require a current/next component and optionally every validity bit."""

    following = reference.endswith(".next")
    name = reference[:-5] if following else reference

    def check(trace: TraceSequence) -> bool:
        trace.resolve(reference)
        return not require_all_valid or bool(
            np.all(trace.components[name].validity(following=following))
        )

    suffix = " with all rows valid" if require_all_valid else ""
    return TraceRequirement(
        f"component:{reference}{suffix}",
        check,
        f"Requires {reference}{suffix}.",
    )


def parameter_requirement(key: str, expected: Any = _MISSING) -> TraceRequirement:
    """Require provenance presence, or an exact value such as a contract hash."""

    def check(trace: TraceSequence) -> bool:
        return key in trace.parameters and (
            expected is _MISSING or trace.parameters[key] == expected
        )

    description = f"Requires provenance key {key}"
    if expected is not _MISSING:
        description += f" equal to {expected!r}"
    return TraceRequirement(f"parameter:{key}", check, description + ".")


def _evidence_rows(
    value: Any,
    available: Any,
    rows: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    result = np.asarray(value)
    known = np.asarray(available)
    if result.dtype != np.bool_ or result.shape != (rows,):
        raise ValueError(f"{name} must be bool with shape [{rows}]")
    if known.dtype != np.bool_ or known.shape != (rows,):
        raise ValueError(f"{name}_available must be bool with shape [{rows}]")
    if np.any(result & ~known):
        raise ValueError(f"unknown {name} rows must use the false placeholder")
    return result, known


def _derive_episode_ids(
    starts: np.ndarray,
    starts_known: np.ndarray,
    terminated: np.ndarray,
    terminated_known: np.ndarray,
    truncated: np.ndarray,
    truncated_known: np.ndarray,
    initial_episode_id: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    episode_id = np.zeros(starts.shape, dtype=np.int64)
    available = np.zeros(starts.shape, dtype=np.bool_)
    current = -1 if initial_episode_id is None else initial_episode_id
    known = initial_episode_id is not None
    previous_ended = False
    previous_end_known = True
    for row in range(starts.size):
        if previous_ended or not previous_end_known:
            known = False
        if not starts_known[row]:
            known = False
        elif starts[row]:
            if row != 0 or initial_episode_id is None:
                if current == np.iinfo(np.int64).max:
                    raise ValueError("derived episode_id exceeds int64")
                current += 1
            known = True
        if known:
            episode_id[row] = current
            available[row] = True
        previous_ended = bool(
            (terminated_known[row] and terminated[row])
            or (truncated_known[row] and truncated[row])
        )
        previous_end_known = bool(
            terminated_known[row] and truncated_known[row]
        )
    return episode_id, available


def _sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest().upper()


__all__ = [
    "TraceProjection",
    "TraceProjector",
    "TraceProjectorRegistry",
    "TraceRequirement",
    "apply_episode_structure",
    "component_requirement",
    "install_episode_evidence",
    "parameter_requirement",
]
