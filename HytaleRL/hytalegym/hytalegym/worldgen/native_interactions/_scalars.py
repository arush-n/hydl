"""Scalar and container validators internal to native interaction evidence."""
from __future__ import annotations

import math
from typing import Any, Mapping


NATIVE_ITEM_INTERACTION_SCHEMA_V2 = (
    "hytalerl_native_item_interaction_evidence_v2"
)
NATIVE_ITEM_INTERACTION_SCHEMA_V3 = (
    "hytalerl_native_item_interaction_evidence_v3"
)
NATIVE_ITEM_INTERACTION_SCHEMA = NATIVE_ITEM_INTERACTION_SCHEMA_V2
NATIVE_ITEM_INTERACTION_VERSION = 2
NATIVE_ITEM_INTERACTION_VERSION_V3 = 3
NATIVE_ITEM_TRIGGER_CAPACITY = 25
NATIVE_ITEM_INTERACTION_CAPACITY = 256
NATIVE_ITEM_EDGE_CAPACITY = 512
NATIVE_ITEM_CHARGE_TIME_CAPACITY = 16
NATIVE_ITEM_BLOCK_CHANGE_CAPACITY = 256
NATIVE_ITEM_METADATA_CAPACITY = 16_384

INTERACTION_TYPE_NAMES = (
    "Primary",
    "Secondary",
    "Ability1",
    "Ability2",
    "Ability3",
    "Use",
    "Pick",
    "Pickup",
    "CollisionEnter",
    "CollisionLeave",
    "Collision",
    "EntityStatEffect",
    "SwapTo",
    "SwapFrom",
    "Death",
    "Wielding",
    "ProjectileSpawn",
    "ProjectileHit",
    "ProjectileMiss",
    "ProjectileBounce",
    "Held",
    "HeldOffhand",
    "Equipped",
    "Dodge",
    "GameModeSwap",
)


def _array(value: Mapping[str, Any], key: str) -> list[Any] | tuple[Any, ...]:
    result = value.get(key)
    if not isinstance(result, (list, tuple)):
        raise ValueError(f"{key} must be an array")
    return result


def _string(
    value: Mapping[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
) -> str:
    result = value.get(key)
    if not isinstance(result, str) or (not allow_empty and not result):
        raise ValueError(f"{key} must be a string")
    return result


def _string_tuple(value: Mapping[str, Any], key: str) -> tuple[str, ...]:
    return tuple(
        _string({"value": item}, "value")
        for item in _array(value, key)
    )


def _boolean(value: Mapping[str, Any], key: str) -> bool:
    result = value.get(key)
    if not isinstance(result, bool):
        raise ValueError(f"{key} must be boolean")
    return result


def _integer(value: Mapping[str, Any], key: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int):
        raise ValueError(f"{key} must be an integer")
    return result


def _number(value: Mapping[str, Any], key: str) -> float:
    return _finite_number(value.get(key), key)


def _finite_number(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("native interaction row must be an object")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            "interaction payload fields changed: "
            f"missing={missing}, unknown={unknown}"
        )


def _pair(value: object) -> tuple[object, object]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("block-change entries must be pairs")
    return value[0], value[1]


def _jsonable(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: _jsonable(getattr(value, key))
            for key in value.__dataclass_fields__
        }
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value
