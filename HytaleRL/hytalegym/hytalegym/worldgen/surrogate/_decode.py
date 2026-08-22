"""Leaf helpers extracted verbatim from assets.py."""

from dataclasses import dataclass
import hashlib
import json
import math
import operator
import re
from typing import Any


_PREFAB_ENTITY_FIELDS = {"Components"}


_SPAWN_MARKER_CONFIGURATION_FIELDS = {
    "Name",
    "Weight",
    "RealtimeRespawnTime",
    "SpawnAfterGameTime",
    "Flock",
}


_JAVA_DURATION = re.compile(
    r"P(?:(?P<days>\d+)D)?"
    r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?"
)


class HytaleAssetError(ValueError):
    """A local asset is absent, malformed, unsupported, or over capacity."""


@dataclass(frozen=True)
class SpawnMarkerConfiguration:
    """One native weighted role choice; flock execution remains separate."""

    role_id: str | None
    weight: float
    realtime_respawn_seconds: float | None
    game_time_respawn: str | None
    game_time_respawn_seconds: float | None
    flock_json: str | None


@dataclass(frozen=True)
class PrefabEntity:
    """Exact prefab entity transform plus compact component provenance."""

    index: int
    position: tuple[float, float, float]
    rotation: tuple[float, float, float]
    velocity: tuple[float, float, float]
    kind: str
    type_id: str
    model_id: str | None
    component_types: tuple[str, ...]
    components_json: str
    components_sha256: str
    unsupported_wrapper_fields: tuple[str, ...]


def _decimal_constants(value: Any) -> tuple[tuple[str, float], ...]:
    result: dict[str, float] = {}
    for index, raw_group in enumerate(_array(value, "world structure Framework")):
        group = _object(raw_group, f"Framework {index}")
        if group.get("Type") != "DecimalConstants":
            continue
        for entry_index, raw_entry in enumerate(
            _array(group.get("Entries"), f"Framework {index}.Entries")
        ):
            entry = _object(
                raw_entry,
                f"Framework {index}.Entries.{entry_index}",
            )
            name = _nonempty_string(
                entry.get("Name"),
                f"Framework {index}.Entries.{entry_index}.Name",
            )
            if name in result:
                raise HytaleAssetError(f"duplicate decimal constant {name!r}")
            result[name] = _finite_number(
                entry.get("Value"),
                f"Framework {index}.Entries.{entry_index}.Value",
            )
    return tuple(sorted(result.items()))


def _generator_objects(
    root: dict[str, Any],
    capacity: int,
    label: str,
) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    stack: list[Any] = [root]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            result.append(value)
            if len(result) > capacity:
                raise HytaleAssetError(f"{label} exceeds node capacity {capacity}")
            for key, child in value.items():
                if key != "$NodeEditorMetadata":
                    stack.append(child)
        elif isinstance(value, list):
            stack.extend(value)
    return tuple(result)


def _component_types(value: Any, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    container = _object(value, label)
    if set(container) != {"Components"}:
        raise HytaleAssetError(f"{label} has unsupported wrapper fields")
    components = _object(container["Components"], f"{label}.Components")
    for name, component in components.items():
        _nonempty_string(name, f"{label} component name")
        _object(component, f"{label}.Components.{name}")
    return tuple(sorted(components))


def _spawn_marker_configuration(
    value: Any,
    index: int,
    realtime_respawn: bool,
) -> SpawnMarkerConfiguration:
    label = f"spawn marker NPCs[{index}]"
    data = _object(value, label)
    unknown = sorted(set(data).difference(_SPAWN_MARKER_CONFIGURATION_FIELDS))
    if unknown:
        raise HytaleAssetError(
            f"{label} has unsupported fields: " + ", ".join(unknown)
        )
    realtime_seconds = (
        None
        if data.get("RealtimeRespawnTime") is None
        else _positive_number(
            data["RealtimeRespawnTime"],
            f"{label}.RealtimeRespawnTime",
        )
    )
    game_time = (
        None
        if data.get("SpawnAfterGameTime") is None
        else _nonempty_string(
            data["SpawnAfterGameTime"],
            f"{label}.SpawnAfterGameTime",
        )
    )
    game_seconds = (
        None
        if game_time is None
        else _positive_java_duration_seconds(
            game_time,
            f"{label}.SpawnAfterGameTime",
        )
    )
    if realtime_respawn and realtime_seconds is None:
        raise HytaleAssetError(f"{label} requires RealtimeRespawnTime")
    if not realtime_respawn and game_seconds is None:
        raise HytaleAssetError(f"{label} requires SpawnAfterGameTime")
    flock = data.get("Flock")
    if flock is None:
        flock_json = None
    elif isinstance(flock, str):
        flock_json = json.dumps(
            _nonempty_string(flock, f"{label}.Flock"),
            separators=(",", ":"),
        )
    elif isinstance(flock, dict):
        _object(flock, f"{label}.Flock")
        flock_json = json.dumps(
            flock,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    else:
        raise HytaleAssetError(f"{label}.Flock must be an asset ID or object")
    return SpawnMarkerConfiguration(
        role_id=_optional_asset_id(data.get("Name"), f"{label}.Name"),
        weight=_positive_number(data.get("Weight"), f"{label}.Weight"),
        realtime_respawn_seconds=realtime_seconds,
        game_time_respawn=game_time,
        game_time_respawn_seconds=game_seconds,
        flock_json=flock_json,
    )


def _positive_java_duration_seconds(value: str, label: str) -> float:
    match = _JAVA_DURATION.fullmatch(value)
    if match is None or not any(match.groupdict().values()):
        raise HytaleAssetError(f"{label} is not a supported positive duration")
    seconds = (
        int(match.group("days") or 0) * 86_400
        + int(match.group("hours") or 0) * 3_600
        + int(match.group("minutes") or 0) * 60
        + float(match.group("seconds") or 0.0)
    )
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise HytaleAssetError(f"{label} must be a positive finite duration")
    return seconds


def _prefab_entity(value: Any, index: int) -> PrefabEntity:
    label = f"prefab entity {index}"
    wrapper = _object(value, label)
    components = _object(wrapper.get("Components"), f"{label}.Components")
    for name, component in components.items():
        _nonempty_string(name, f"{label} component name")
        _object(component, f"{label}.Components.{name}")
    transform = _object(
        components.get("Transform"),
        f"{label}.Components.Transform",
    )
    position = _vector3(
        transform.get("Position"),
        f"{label}.Components.Transform.Position",
    )
    rotation = _rotation3(
        transform.get("Rotation"),
        f"{label}.Components.Transform.Rotation",
    )
    velocity_component = components.get("Velocity")
    velocity = (
        (0.0, 0.0, 0.0)
        if velocity_component is None
        else _vector3(
            _object(
                velocity_component,
                f"{label}.Components.Velocity",
            ).get("Velocity"),
            f"{label}.Components.Velocity.Velocity",
        )
    )
    model_component = components.get("Model")
    model_id = None
    if model_component is not None:
        model_id = _nonempty_string(
            _object(
                _object(
                    model_component,
                    f"{label}.Components.Model",
                ).get("Model"),
                f"{label}.Components.Model.Model",
            ).get("Id"),
            f"{label}.Components.Model.Model.Id",
        )
    kind, type_id = _entity_kind_and_type(components, model_id, label)
    encoded = json.dumps(
        components,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return PrefabEntity(
        index=index,
        position=position,
        rotation=rotation,
        velocity=velocity,
        kind=kind,
        type_id=type_id,
        model_id=model_id,
        component_types=tuple(sorted(components)),
        components_json=encoded,
        components_sha256=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        unsupported_wrapper_fields=tuple(
            sorted(set(wrapper).difference(_PREFAB_ENTITY_FIELDS))
        ),
    )


def _entity_kind_and_type(
    components: dict[str, Any],
    model_id: str | None,
    label: str,
) -> tuple[str, str]:
    if "SpawnMarkerComponent" in components:
        marker = _object(
            components["SpawnMarkerComponent"],
            f"{label}.Components.SpawnMarkerComponent",
        )
        return "spawn_marker", _nonempty_string(
            marker.get("SpawnMarker"),
            f"{label}.Components.SpawnMarkerComponent.SpawnMarker",
        )
    if "NPC" in components:
        npc = _object(components["NPC"], f"{label}.Components.NPC")
        return "npc", _nonempty_string(
            npc.get("RoleName", model_id),
            f"{label}.Components.NPC.RoleName",
        )
    if "TriggerVolume" in components:
        return "trigger_volume", "TriggerVolume"
    if "PatrolPathMarker" in components:
        marker = _object(
            components["PatrolPathMarker"],
            f"{label}.Components.PatrolPathMarker",
        )
        return "patrol_marker", _nonempty_string(
            marker.get("PathName"),
            f"{label}.Components.PatrolPathMarker.PathName",
        )
    if "SpawnSuppression" in components:
        suppression = _object(
            components["SpawnSuppression"],
            f"{label}.Components.SpawnSuppression",
        )
        return "spawn_suppression", _nonempty_string(
            suppression.get("SpawnSuppression"),
            f"{label}.Components.SpawnSuppression.SpawnSuppression",
        )
    return "prefab_entity", model_id or "PrefabEntity"


def _entry_path(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("asset entry path must be a string")
    result = value.replace("\\", "/")
    parts = result.split("/")
    if (
        not result
        or result.startswith("/")
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise HytaleAssetError(f"unsafe asset entry path {value!r}")
    return result


def _relative_asset_reference(value: Any, label: str) -> str:
    raw = _nonempty_string(value, label).replace("\\", "/")
    result = raw.rstrip("/")
    if not result or raw.startswith("/") or any(
        part in {"", ".", ".."} for part in result.split("/")
    ):
        raise HytaleAssetError(f"{label} is not a safe relative asset path")
    return result


def _asset_id(value: Any, label: str) -> str:
    result = _nonempty_string(value, label)
    if any(character in result for character in "/\\") or result in {".", ".."}:
        raise HytaleAssetError(f"{label} is not a safe asset ID")
    return result


def _optional_asset_id(value: Any, label: str) -> str | None:
    if value in (None, ""):
        return None
    return _asset_id(value, label)


def _nonempty_selector(value: Any, label: str) -> bool:
    if value is None:
        return False
    source = _array(value, label)
    stack = list(source)
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        else:
            _nonempty_string(item, label)
    return bool(source)


def _native_glob_matches(pattern: str, candidate: str) -> bool:
    """Match Hytale ``StringUtil`` glob semantics: case-sensitive ``*``/``?``."""

    previous = [True] + [False] * len(candidate)
    for token in pattern:
        current = [False] * (len(candidate) + 1)
        if token == "*":
            current[0] = previous[0]
            for index in range(1, len(current)):
                current[index] = previous[index] or current[index - 1]
        else:
            for index, character in enumerate(candidate, start=1):
                current[index] = previous[index - 1] and (
                    token == "?" or token == character
                )
        previous = current
    return previous[-1]


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise HytaleAssetError(f"{label} must be a JSON object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise HytaleAssetError(f"{label} must be a JSON array")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise HytaleAssetError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise HytaleAssetError(f"{label} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise HytaleAssetError(f"{label} must be an integer") from error


def _optional_nonnegative_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    result = _integer(value, label)
    if result < 0:
        raise HytaleAssetError(f"{label} must be non-negative")
    return result


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HytaleAssetError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise HytaleAssetError(f"{label} must be finite")
    return result


def _positive_number(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if result <= 0.0:
        raise HytaleAssetError(f"{label} must be positive")
    return result


def _vector3(value: Any, label: str) -> tuple[float, float, float]:
    source = _object(value, label)
    return tuple(
        _finite_number(source.get(axis), f"{label}.{axis}") for axis in ("X", "Y", "Z")
    )


def _rotation3(value: Any, label: str) -> tuple[float, float, float]:
    source = _object(value, label)
    return tuple(
        _finite_number(source.get(axis), f"{label}.{axis}")
        for axis in ("Pitch", "Roll", "Yaw")
    )
