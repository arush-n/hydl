"""Strict asset-pinned projection for actor Walk movement-state telemetry."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib.resources import files
import json
import math
from typing import Any, Mapping

from hytalegym.combat.assets import HYTALE_0_5_7_ASSETS_SHA256


ACTOR_WALK_MOVEMENT_PROJECTION_RESOURCE = (
    "hytale_0_5_7/kweebec_razorleaf_movement_projection_v1.json"
)
ACTOR_WALK_MOVEMENT_PROJECTION_SCHEMA = (
    "hytalerl_actor_walk_movement_projection_v1"
)
ACTOR_WALK_MOVEMENT_PROJECTION_VERSION = 1

_ROOT_FIELDS = frozenset(
    (
        "schema",
        "version",
        "hytale_server_version",
        "assets_sha256",
        "role_id",
        "role_asset",
        "controller_type",
        "component_selector",
        "run_threshold",
        "run_threshold_range",
        "hover_height",
        "ascent_animation_type",
        "descent_animation_type",
        "minimum_descent_animation_height",
        "provenance",
        "not_claimed",
    )
)
_PROVENANCE_FIELDS = frozenset(
    (
        "authored_controller",
        "component_selector",
        "run_threshold",
        "run_threshold_range",
        "walk_animation_defaults",
        "enum_order",
        "hover_defaults",
    )
)
_ASCENT_ANIMATION = {
    "Walk": 0,
    "Jump": 1,
    "Climb": 2,
    "Fly": 3,
    "Idle": 4,
}
_DESCENT_ANIMATION = {
    "Walk": 0,
    "Fall": 1,
    "Idle": 2,
}


@dataclass(frozen=True)
class ActorWalkMovementProjection:
    """Role-authored and builder-default inputs to the Walk state producer."""

    role_id: str
    role_asset: str
    component_selector: tuple[float, float, float]
    run_threshold: float
    run_threshold_range: float
    hover_height: float
    ascent_animation_type: int
    descent_animation_type: int
    minimum_descent_animation_height: float
    not_claimed: tuple[str, ...]


def actor_walk_movement_projection_sha256() -> str:
    """Return the exact packaged projection identity."""

    return hashlib.sha256(_projection_bytes()).hexdigest()


def load_actor_walk_movement_projection() -> ActorWalkMovementProjection:
    """Return an isolated, validated Kweebec Walk projection."""

    return copy.deepcopy(_load_projection())


@lru_cache(maxsize=1)
def _projection_bytes() -> bytes:
    return (
        files("hytalegym.rulesets")
        .joinpath(ACTOR_WALK_MOVEMENT_PROJECTION_RESOURCE)
        .read_bytes()
    )


@lru_cache(maxsize=1)
def _load_projection() -> ActorWalkMovementProjection:
    document = json.loads(_projection_bytes())
    _require_fields(document, _ROOT_FIELDS, "movement projection")
    if (
        document["schema"] != ACTOR_WALK_MOVEMENT_PROJECTION_SCHEMA
        or _integer(document["version"], "version")
        != ACTOR_WALK_MOVEMENT_PROJECTION_VERSION
        or document["hytale_server_version"] != "0.5.7"
        or document["assets_sha256"] != HYTALE_0_5_7_ASSETS_SHA256
        or document["controller_type"] != "Walk"
    ):
        raise ValueError("unsupported or stale actor Walk movement projection")
    _require_fields(
        document["provenance"],
        _PROVENANCE_FIELDS,
        "movement projection provenance",
    )
    for name, value in document["provenance"].items():
        _nonempty_string(value, f"movement projection provenance {name}")

    selector = document["component_selector"]
    if not isinstance(selector, list) or len(selector) != 3:
        raise TypeError("component_selector must contain three values")
    component_selector = tuple(
        _nonnegative(value, "component_selector") for value in selector
    )
    if not any(value > 0.0 for value in component_selector):
        raise ValueError("component_selector must select at least one axis")

    run_threshold = _unit_interval(
        document["run_threshold"],
        "run_threshold",
    )
    run_threshold_range = _unit_interval(
        document["run_threshold_range"],
        "run_threshold_range",
    )
    if run_threshold_range > run_threshold:
        raise ValueError("run_threshold_range exceeds run_threshold")

    ascent = _nonempty_string(
        document["ascent_animation_type"],
        "ascent_animation_type",
    )
    descent = _nonempty_string(
        document["descent_animation_type"],
        "descent_animation_type",
    )
    if ascent not in _ASCENT_ANIMATION:
        raise ValueError(f"unknown ascent animation type: {ascent!r}")
    if descent not in _DESCENT_ANIMATION:
        raise ValueError(f"unknown descent animation type: {descent!r}")

    not_claimed = document["not_claimed"]
    if (
        not isinstance(not_claimed, list)
        or not not_claimed
        or len(set(not_claimed)) != len(not_claimed)
    ):
        raise ValueError("not_claimed must be a nonempty unique list")
    boundary = tuple(
        _nonempty_string(value, "not_claimed entry") for value in not_claimed
    )
    return ActorWalkMovementProjection(
        role_id=_nonempty_string(document["role_id"], "role_id"),
        role_asset=_nonempty_string(document["role_asset"], "role_asset"),
        component_selector=component_selector,
        run_threshold=run_threshold,
        run_threshold_range=run_threshold_range,
        hover_height=_nonnegative(document["hover_height"], "hover_height"),
        ascent_animation_type=_ASCENT_ANIMATION[ascent],
        descent_animation_type=_DESCENT_ANIMATION[descent],
        minimum_descent_animation_height=_nonnegative(
            document["minimum_descent_animation_height"],
            "minimum_descent_animation_height",
        ),
        not_claimed=boundary,
    )


def _require_fields(
    value: Any,
    expected: frozenset[str],
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a nonempty string")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _nonnegative(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _unit_interval(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0,1]")
    return result


__all__ = [
    "ACTOR_WALK_MOVEMENT_PROJECTION_RESOURCE",
    "ACTOR_WALK_MOVEMENT_PROJECTION_SCHEMA",
    "ACTOR_WALK_MOVEMENT_PROJECTION_VERSION",
    "ActorWalkMovementProjection",
    "actor_walk_movement_projection_sha256",
    "load_actor_walk_movement_projection",
]
