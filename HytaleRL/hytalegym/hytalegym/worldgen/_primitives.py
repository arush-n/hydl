"""Leaf helpers extracted verbatim from native_world_actions.py."""

import math
from typing import Mapping


def _boolean(info: Mapping[str, object], key: str) -> bool:
    value = info.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean")
    return value


def _optional_target(
    info: Mapping[str, object],
    availability_key: str,
    coordinate_prefix: str | None = None,
) -> tuple[int, int, int] | None:
    if not _boolean(info, availability_key):
        return None
    prefix = coordinate_prefix or availability_key
    return tuple(
        _integer(info, f"{prefix}_{axis}")
        for axis in ("x", "y", "z")
    )


def _nonempty_string_or_empty(
    info: Mapping[str, object],
    key: str,
) -> str:
    value = info.get(key)
    if not isinstance(value, str) or value != value.strip():
        raise ValueError(f"{key} must be a canonical string")
    return value


def _bounded_canonical_string(
    info: Mapping[str, object],
    key: str,
    *,
    maximum: int,
    allow_empty: bool = True,
) -> str:
    value = _nonempty_string_or_empty(info, key)
    if len(value) > maximum or (not allow_empty and not value):
        raise ValueError(f"{key} is outside its string contract")
    return value


def _optional_sha256(info: Mapping[str, object], key: str) -> str:
    value = info.get(key)
    if value == "":
        return ""
    return _sha256(value, key)


def _integer(
    info: Mapping[str, object],
    key: str,
    *,
    minimum: int | None = None,
) -> int:
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} is below its minimum")
    return value


def _real(
    info: Mapping[str, object],
    key: str,
    *,
    minimum: float | None = None,
) -> float:
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be real")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{key} is below its minimum")
    return result


def _sha256(value: object, key: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{key} must be a SHA-256")
    return value
