"""Structural extraction for run diagnostics already present in artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


HEAD_ACTIVITY = "active_head_active_fraction"


def diagnostic_groups(document: Any) -> dict[str, list[dict[str, Any]]]:
    """Return head tables and complete literal before/after pairs."""

    heads: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    default_teacher = _teacher_coefficient(document)

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if not isinstance(value, Mapping):
            return

        pair_keys = {key for key in ("before", "after") if key in value}
        if pair_keys:
            if pair_keys == {"before", "after"} and all(
                isinstance(value[key], Mapping) for key in pair_keys
            ):
                pairs.append({
                    "path": _path(path),
                    "arms": [
                        _outcome(key, value[key], default_teacher)
                        for key in ("before", "after")
                    ],
                })
                visit(value["before"], (*path, "before"))
                visit(value["after"], (*path, "after"))
            for key, child in value.items():
                if key not in pair_keys:
                    visit(child, (*path, str(key)))
            return

        activity = value.get(HEAD_ACTIVITY)
        if isinstance(activity, Mapping):
            heads.append({
                **_outcome(path[-1] if path else "[root]", value, default_teacher),
                "path": _path(path),
                "activity": dict(activity),
            })
        for key, child in value.items():
            visit(child, (*path, str(key)))

    visit(document, ())
    return {"head_tables": heads, "pairs": pairs}


def _outcome(name: str, value: Mapping[str, Any], default_teacher: Any) -> dict[str, Any]:
    return {
        "arm": name,
        "criterion_success_rate": value.get("criterion_success_rate"),
        "success_rate": value.get("success_rate"),
        "teacher_coefficient": _teacher_coefficient(value, default_teacher),
    }


def _teacher_coefficient(value: Any, default: Any = None) -> Any:
    if not isinstance(value, Mapping):
        return default
    for name, candidate in value.items():
        if name == "teacher_coefficient" or name.endswith("_teacher_coefficient"):
            return candidate
    config = value.get("config")
    if isinstance(config, Mapping):
        return _teacher_coefficient(config, default)
    return default


def _path(parts: tuple[str, ...]) -> str:
    return ".".join(parts) or "[root]"


__all__ = ["HEAD_ACTIVITY", "diagnostic_groups"]
