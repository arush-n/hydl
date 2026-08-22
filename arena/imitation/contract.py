"""Algorithm-neutral, composable transition traces for agents and world models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import numpy.typing as npt


TraceKind = Literal["state", "observation", "action", "mask", "event", "context"]
TraceLayer = Literal[
    "omniscient",
    "internal",
    "external",
    "decision",
    "transition",
    "provenance",
    "extension",
]
_KINDS = frozenset({"state", "observation", "action", "mask", "event", "context"})
_LAYERS = frozenset(
    {
        "omniscient",
        "internal",
        "external",
        "decision",
        "transition",
        "provenance",
        "extension",
    }
)
_NUMERIC_DTYPES = {
    "bool": np.bool_,
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
}
_NAME = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")


@dataclass(frozen=True, slots=True)
class TraceSpec:
    """Schema for one namespaced, row-aligned trace component."""

    name: str
    kind: TraceKind
    dtype: str
    shape: tuple[int, ...] | None = ()
    layout: tuple[str, ...] = ()
    stateful: bool = False
    units: str = ""
    description: str = ""
    layer: TraceLayer = "extension"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "shape", None if self.shape is None else tuple(self.shape)
        )
        object.__setattr__(self, "layout", tuple(self.layout))
        if _NAME.fullmatch(self.name) is None or self.name.endswith(".next"):
            raise ValueError("trace component names must be namespaced snake_case")
        if self.kind not in _KINDS:
            raise ValueError(f"unknown trace component kind: {self.kind}")
        if self.layer not in _LAYERS:
            raise ValueError(f"unknown trace component layer: {self.layer}")
        if not self.dtype:
            raise ValueError("trace component dtype is required")
        if self.shape is not None and any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in self.shape
        ):
            raise ValueError("trace component shape must be nonnegative integers")
        if self.layout and self.shape and self.shape[-1] != len(self.layout):
            raise ValueError("layout must name the component's final axis")

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "dtype": self.dtype,
            "shape": self.shape,
            "layout": self.layout,
            "stateful": self.stateful,
            "units": self.units,
            "description": self.description,
            "layer": self.layer,
        }


@dataclass(frozen=True, slots=True)
class TraceComponent:
    """Current data, optional next data, and an explicit per-row validity mask."""

    spec: TraceSpec
    current: Any
    next: Any = None
    valid: npt.NDArray[np.bool_] | None = None
    next_valid: npt.NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        current = _freeze(self.current, self.spec)
        rows = len(current)
        if self.spec.stateful != (self.next is not None):
            raise ValueError("stateful components require current and next values")
        following = None if self.next is None else _freeze(self.next, self.spec)
        if following is not None and len(following) != rows:
            raise ValueError("current and next component rows must match")
        valid = (
            np.ones(rows, dtype=np.bool_)
            if self.valid is None
            else np.asarray(self.valid, dtype=np.bool_).copy()
        )
        if valid.shape != (rows,):
            raise ValueError("component validity must have shape [rows]")
        valid.flags.writeable = False
        if following is None and self.next_valid is not None:
            raise ValueError("non-stateful components cannot have next validity")
        following_valid = None
        if following is not None:
            following_valid = (
                valid.copy()
                if self.next_valid is None
                else np.asarray(self.next_valid, dtype=np.bool_).copy()
            )
            if following_valid.shape != (rows,):
                raise ValueError("next component validity must have shape [rows]")
            following_valid.flags.writeable = False
        object.__setattr__(self, "current", current)
        object.__setattr__(self, "next", following)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "next_valid", following_valid)

    @property
    def rows(self) -> int:
        return len(self.current)

    def validity(self, *, following: bool = False) -> npt.NDArray[np.bool_]:
        if following:
            if self.next_valid is None:
                raise ValueError(f"component is not stateful: {self.spec.name}")
            return self.next_valid
        return self.valid


@dataclass(frozen=True, slots=True)
class TraceContract:
    """Stable schema assembled from independent trace components."""

    components: tuple[TraceSpec, ...]
    parameter_keys: tuple[str, ...] = ()
    schema: str = "arena_composable_transition_trace_v3"

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))
        names = tuple(spec.name for spec in self.components)
        if len(names) != len(set(names)):
            raise ValueError("trace component names must be unique")
        keys = tuple(self.parameter_keys)
        if len(keys) != len(set(keys)) or any(not key for key in keys):
            raise ValueError("trace parameter keys must be unique and non-empty")
        object.__setattr__(self, "parameter_keys", tuple(sorted(keys)))

    def compose(
        self, *specs: TraceSpec, parameter_keys: Sequence[str] = ()
    ) -> TraceContract:
        return TraceContract(
            self.components + tuple(specs),
            self.parameter_keys + tuple(parameter_keys),
            self.schema,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "components": [spec.manifest() for spec in self.components],
            "parameter_keys": self.parameter_keys,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.manifest())

    def layer(self, name: TraceLayer) -> tuple[TraceSpec, ...]:
        if name not in _LAYERS:
            raise ValueError(f"unknown trace component layer: {name}")
        return tuple(spec for spec in self.components if spec.layer == name)


@dataclass(frozen=True, slots=True)
class TraceSequence:
    """Validated transition rows plus immutable provenance parameters."""

    contract: TraceContract
    tick: npt.NDArray[np.int64]
    delta_seconds: npt.NDArray[np.float32]
    episode_id: npt.NDArray[np.int64]
    episode_start: npt.NDArray[np.bool_]
    terminated: npt.NDArray[np.bool_]
    truncated: npt.NDArray[np.bool_]
    episode_id_available: npt.NDArray[np.bool_]
    episode_start_available: npt.NDArray[np.bool_]
    terminated_available: npt.NDArray[np.bool_]
    truncated_available: npt.NDArray[np.bool_]
    components: Mapping[str, TraceComponent]
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        columns = {
            "tick": (self.tick, np.int64),
            "delta_seconds": (self.delta_seconds, np.float32),
            "episode_id": (self.episode_id, np.int64),
            "episode_start": (self.episode_start, np.bool_),
            "terminated": (self.terminated, np.bool_),
            "truncated": (self.truncated, np.bool_),
            "episode_id_available": (self.episode_id_available, np.bool_),
            "episode_start_available": (self.episode_start_available, np.bool_),
            "terminated_available": (self.terminated_available, np.bool_),
            "truncated_available": (self.truncated_available, np.bool_),
        }
        frozen = {
            name: _array(value, dtype) for name, (value, dtype) in columns.items()
        }
        rows = len(frozen["tick"])
        if any(value.shape != (rows,) for value in frozen.values()):
            raise ValueError("trace transition columns must have shape [rows]")
        if not np.all(np.isfinite(frozen["delta_seconds"])) or np.any(
            frozen["delta_seconds"] < 0.0
        ):
            raise ValueError("trace delta_seconds must be finite and nonnegative")
        for name in ("episode_start", "terminated", "truncated"):
            if np.any(frozen[name] & ~frozen[f"{name}_available"]):
                raise ValueError(f"unknown {name} rows must use the false placeholder")
        if np.any(~frozen["episode_id_available"] & (frozen["episode_id"] != 0)):
            raise ValueError("unknown episode_id rows must use the zero placeholder")
        if np.any(frozen["episode_id_available"] & (frozen["episode_id"] < 0)):
            raise ValueError("known episode IDs must be nonnegative")
        same_episode = frozen["episode_id"][1:] == frozen["episode_id"][:-1]
        if np.any(same_episode & (frozen["tick"][1:] <= frozen["tick"][:-1])):
            raise ValueError("trace ticks must increase within each episode")

        components = dict(self.components)
        expected = {spec.name: spec for spec in self.contract.components}
        if set(components) != set(expected):
            raise ValueError("trace components do not match the contract")
        if any(
            component.spec != expected[name] or component.rows != rows
            for name, component in components.items()
        ):
            raise ValueError("trace component schema or row count does not match")
        parameters = dict(self.parameters)
        if set(parameters) != set(self.contract.parameter_keys):
            raise ValueError("trace parameters do not match the contract")
        for name, value in frozen.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "components", MappingProxyType(components))
        object.__setattr__(self, "parameters", MappingProxyType(parameters))

    @property
    def rows(self) -> int:
        return int(self.tick.size)

    @property
    def identity_sha256(self) -> str:
        return _sha256(
            {"contract_sha256": self.contract.sha256, "parameters": self.parameters}
        )

    def compose(
        self,
        *components: TraceComponent,
        parameters: Mapping[str, Any] | None = None,
    ) -> TraceSequence:
        additions = dict(parameters or {})
        if set(additions) & set(self.parameters):
            raise ValueError("composed trace parameters must not overwrite provenance")
        merged = dict(self.components)
        for component in components:
            if component.spec.name in merged:
                raise ValueError(
                    f"trace component already exists: {component.spec.name}"
                )
            merged[component.spec.name] = component
        return TraceSequence(
            contract=self.contract.compose(
                *(component.spec for component in components),
                parameter_keys=additions,
            ),
            tick=self.tick,
            delta_seconds=self.delta_seconds,
            episode_id=self.episode_id,
            episode_start=self.episode_start,
            terminated=self.terminated,
            truncated=self.truncated,
            episode_id_available=self.episode_id_available,
            episode_start_available=self.episode_start_available,
            terminated_available=self.terminated_available,
            truncated_available=self.truncated_available,
            components=merged,
            parameters={**self.parameters, **additions},
        )

    def resolve(self, reference: str) -> Any:
        name, following = _reference(reference)
        component = self.components[name]
        if following and component.next is None:
            raise ValueError(f"component is not stateful: {name}")
        return component.next if following else component.current

    def with_episode_structure(
        self,
        *,
        episode_id: Any,
        episode_start: Any,
        terminated: Any,
        truncated: Any,
        episode_id_available: Any,
        episode_start_available: Any,
        terminated_available: Any,
        truncated_available: Any,
        parameters: Mapping[str, Any] | None = None,
    ) -> TraceSequence:
        """Replace episode columns with derived evidence and stamped provenance."""

        additions = dict(parameters or {})
        overlap = set(additions) & set(self.parameters)
        if overlap:
            raise ValueError(
                f"episode parameters overwrite trace provenance: {sorted(overlap)}"
            )
        return TraceSequence(
            contract=self.contract.compose(parameter_keys=additions),
            tick=self.tick,
            delta_seconds=self.delta_seconds,
            episode_id=episode_id,
            episode_start=episode_start,
            terminated=terminated,
            truncated=truncated,
            episode_id_available=episode_id_available,
            episode_start_available=episode_start_available,
            terminated_available=terminated_available,
            truncated_available=truncated_available,
            components=self.components,
            parameters={**self.parameters, **additions},
        )

    def layer(self, name: TraceLayer) -> Mapping[str, TraceComponent]:
        """Return one immutable modality layer without copying its rows."""

        specs = self.contract.layer(name)
        return MappingProxyType(
            {spec.name: self.components[spec.name] for spec in specs}
        )

    def require_episode_structure(self, *fields: str) -> None:
        """Fail unless every requested episode field is known for every row."""

        selected = fields or (
            "episode_id",
            "episode_start",
            "terminated",
            "truncated",
        )
        unknown = [
            name
            for name in selected
            if name not in {"episode_id", "episode_start", "terminated", "truncated"}
            or not np.all(getattr(self, f"{name}_available"))
        ]
        if unknown:
            raise ValueError(f"episode structure is unavailable: {', '.join(unknown)}")

    def bootstrap_mask(
        self, *, bootstrap_on_truncation: bool = True
    ) -> npt.NDArray[np.bool_]:
        """Return a TD bootstrap mask, refusing unknown terminal semantics."""

        fields = (
            ("terminated",)
            if bootstrap_on_truncation
            else (
                "terminated",
                "truncated",
            )
        )
        self.require_episode_structure(*fields)
        result = ~self.terminated.copy()
        if not bootstrap_on_truncation:
            result &= ~self.truncated
        result.flags.writeable = False
        return result

    def window_indices(
        self,
        length: int,
        *,
        stride: int = 1,
        require_episode_structure: bool = False,
    ) -> npt.NDArray[np.int64]:
        """Return contiguous windows that do not cross known episode boundaries."""

        if (
            isinstance(length, bool)
            or not isinstance(length, int)
            or length < 1
            or isinstance(stride, bool)
            or not isinstance(stride, int)
            or stride < 1
        ):
            raise ValueError("window length and stride must be positive integers")
        if require_episode_structure:
            self.require_episode_structure()
        windows = []
        for start in range(0, self.rows - length + 1, stride):
            indexes = np.arange(start, start + length, dtype=np.int64)
            if (
                np.any(
                    self.episode_id_available[indexes]
                    & self.episode_id_available[start]
                    & (self.episode_id[indexes] != self.episode_id[start])
                )
                or np.any(
                    self.episode_start_available[indexes[1:]]
                    & self.episode_start[indexes[1:]]
                )
                or np.any(
                    self.terminated_available[indexes[:-1]]
                    & self.terminated[indexes[:-1]]
                )
                or np.any(
                    self.truncated_available[indexes[:-1]]
                    & self.truncated[indexes[:-1]]
                )
            ):
                continue
            windows.append(indexes)
        result = np.stack(windows) if windows else np.empty((0, length), dtype=np.int64)
        result.flags.writeable = False
        return result


@dataclass(frozen=True, slots=True)
class TraceBatch:
    """A trainer-neutral bound view; mappings remain separate modalities."""

    inputs: Mapping[str, Any]
    targets: Mapping[str, Any]
    context: Mapping[str, Any]
    valid: npt.NDArray[np.bool_]
    source_index: npt.NDArray[np.int64]
    contract_sha256: str


@dataclass(frozen=True, slots=True)
class TraceView:
    """Select named current/next fields without choosing a learning algorithm."""

    inputs: tuple[str, ...]
    targets: tuple[str, ...]
    context: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        references = self.inputs + self.targets + self.context
        if not self.inputs or not self.targets:
            raise ValueError("trace views require at least one input and target")
        if len(references) != len(set(references)):
            raise ValueError("trace view references must be unique")

    def bind(self, trace: TraceSequence, rows: Any = None) -> TraceBatch:
        indexes = _indexes(rows, trace.rows)
        groups = tuple(self.inputs + self.targets + self.context)
        valid = np.ones(indexes.shape, dtype=np.bool_)
        for reference in groups:
            name, following = _reference(reference)
            if name not in trace.components:
                raise KeyError(f"trace has no component {name!r}")
            valid &= trace.components[name].validity(following=following)[indexes]
        valid.flags.writeable = False
        indexes.flags.writeable = False
        return TraceBatch(
            inputs=_values(trace, self.inputs, indexes),
            targets=_values(trace, self.targets, indexes),
            context=_values(trace, self.context, indexes),
            valid=valid,
            source_index=indexes,
            contract_sha256=trace.contract.sha256,
        )


def numeric_component(
    name: str,
    current: Any,
    *,
    next: Any = None,
    kind: TraceKind = "observation",
    layout: Sequence[str] = (),
    valid: Any = None,
    next_valid: Any = None,
    units: str = "",
    description: str = "",
    layer: TraceLayer = "extension",
) -> TraceComponent:
    """Create an immutable numeric feature; useful for spatial/terrain adapters."""

    value = np.asarray(current)
    if value.ndim < 1 or value.dtype.kind not in "bifu":
        raise TypeError("numeric trace components need a row-major numeric array")
    if value.dtype.kind == "b":
        dtype = "bool"
    elif value.dtype.kind == "f":
        dtype = "float32" if value.dtype.itemsize <= 4 else "float64"
    else:
        dtype = "int32" if value.dtype.itemsize <= 4 else "int64"
    spec = TraceSpec(
        name=name,
        kind=kind,
        dtype=dtype,
        shape=value.shape[1:],
        layout=tuple(layout),
        stateful=next is not None,
        units=units,
        description=description,
        layer=layer,
    )
    return TraceComponent(spec, current, next, valid, next_valid)


def reward_component(
    reward: Any,
    *,
    name: str = "reward.task",
    valid: Any = None,
    units: str = "",
) -> TraceComponent:
    """Declare one scalar reward; projector metadata supplies its version."""

    value = np.asarray(reward)
    if not name.startswith("reward."):
        raise ValueError("reward names must use the reward namespace")
    if value.ndim != 1:
        raise ValueError("each reward component must be scalar per transition")
    return numeric_component(
        name,
        value,
        kind="event",
        valid=valid,
        units=units,
        description="Task- or learner-defined transition reward.",
        layer="extension",
    )


def categorical_action_component(
    name: str,
    action: Any,
    supported: Any,
    *,
    valid: Any,
    layer: TraceLayer = "decision",
) -> TraceComponent:
    """Create exact categorical labels; unsupported or ambiguous rows abstain."""

    labels = np.asarray(action)
    legal = np.asarray(supported, dtype=np.bool_)
    mask = np.asarray(valid, dtype=np.bool_)
    if labels.ndim != 1 or labels.dtype.kind not in "iu":
        raise TypeError(
            "categorical action labels must be a one-dimensional integer array"
        )
    if legal.ndim != 2 or legal.shape[0] != labels.size or legal.shape[1] < 1:
        raise ValueError("supported actions must have shape [rows, action_count]")
    if mask.shape != labels.shape:
        raise ValueError("categorical action validity must have shape [rows]")
    action_count = legal.shape[1]
    if np.any(mask & ((labels < 0) | (labels >= action_count))):
        raise ValueError("valid categorical action labels are out of range")
    safe = np.zeros(labels.shape, dtype=np.int32)
    safe[mask] = labels[mask]
    if np.any(mask & ~legal[np.arange(labels.size), safe]):
        raise ValueError("valid categorical action labels must be supported")
    return numeric_component(
        name,
        safe,
        kind="action",
        valid=mask,
        description="Exact categorical action; invalid rows are explicit abstentions.",
        layer=layer,
    )


def structured_component(
    name: str,
    current: Sequence[Any],
    *,
    dtype: str,
    next: Sequence[Any] | None = None,
    kind: TraceKind = "context",
    valid: Any = None,
    next_valid: Any = None,
    description: str = "",
    layer: TraceLayer = "extension",
) -> TraceComponent:
    """Create a row-aligned immutable string/UUID/record component."""

    spec = TraceSpec(
        name=name,
        kind=kind,
        dtype=dtype,
        shape=None,
        stateful=next is not None,
        description=description,
        layer=layer,
    )
    return TraceComponent(spec, current, next, valid, next_valid)


def _freeze(value: Any, spec: TraceSpec) -> Any:
    if spec.dtype in _NUMERIC_DTYPES:
        result = np.asarray(value, dtype=_NUMERIC_DTYPES[spec.dtype]).copy()
        if result.ndim < 1 or result.shape[1:] != spec.shape:
            raise ValueError(f"{spec.name} rows must have trailing shape {spec.shape}")
        result.flags.writeable = False
        return result
    result = tuple(_immutable(item) for item in value)
    return result


def _immutable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_immutable(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_immutable(item) for item in value)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _immutable(item) for key, item in value.items()})
    return value


def _array(value: Any, dtype: Any) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.flags.writeable = False
    return result


def _reference(value: str) -> tuple[str, bool]:
    if not isinstance(value, str):
        raise TypeError("trace references must be strings")
    following = value.endswith(".next")
    return (value[:-5] if following else value), following


def _indexes(rows: Any, count: int) -> npt.NDArray[np.int64]:
    if rows is None:
        return np.arange(count, dtype=np.int64)
    value = np.asarray(rows)
    if value.dtype == np.bool_:
        if value.shape != (count,):
            raise ValueError("boolean row selector must have shape [trace.rows]")
        return np.flatnonzero(value).astype(np.int64)
    result = np.asarray(value, dtype=np.int64)
    if result.ndim < 1:
        raise ValueError("integer row selectors must have at least one axis")
    if np.any((result < 0) | (result >= count)):
        raise IndexError("trace row selector is out of range")
    return result.copy()


def _values(
    trace: TraceSequence, references: Sequence[str], indexes: npt.NDArray[np.int64]
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            reference: _take(trace.resolve(reference), indexes)
            for reference in references
        }
    )


def _take(value: Any, indexes: npt.NDArray[np.int64]) -> Any:
    if isinstance(value, np.ndarray):
        result = value[indexes].copy()
        result.flags.writeable = False
        return result
    if indexes.ndim == 1:
        return tuple(value[int(index)] for index in indexes)
    return tuple(_take(value, row) for row in indexes)


def _sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest().upper()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return (
        str(value)
        if not isinstance(value, (str, int, float, bool, type(None)))
        else value
    )


__all__ = [
    "TraceBatch",
    "TraceComponent",
    "TraceContract",
    "TraceKind",
    "TraceLayer",
    "TraceSequence",
    "TraceSpec",
    "TraceView",
    "categorical_action_component",
    "numeric_component",
    "reward_component",
    "structured_component",
]
