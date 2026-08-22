"""Asset-pinned projection rules for native-only entity effects."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib.resources import files
import json
import math
from typing import Any

from hytalegym.combat.assets import HYTALE_0_5_7_ASSETS_SHA256


NATIVE_STATUS_PROJECTION_RESOURCE = (
    "hytale_0_5_7/native_status_projection_v2.json"
)
NATIVE_STATUS_PROJECTION_SCHEMA = "hytalerl_native_status_projection_v2"
NATIVE_STATUS_PROJECTION_VERSION = 2
NATIVE_STATUS_PROJECTION_OMIT_VISUAL_ONLY = "omit_visual_only"
NATIVE_STATUS_PROJECTION_OMIT_DERIVED_MECHANIC = "omit_derived_mechanic"
NATIVE_STATUS_PROJECTION_STAMINA_BREAK_IMMUNITY = (
    "ability_resource_cost_prevents_stamina_break"
)
NATIVE_STATUS_PROJECTION_DODGE_INVULNERABILITY = (
    "dodge_invulnerability_defense_channel"
)

_ROOT_FIELDS = frozenset(
    (
        "schema",
        "version",
        "hytale_server_version",
        "assets_sha256",
        "provenance",
        "statuses",
    )
)
_PROVENANCE_FIELDS = frozenset(
    (
        "damage_application",
        "entity_effect_defaults",
        "application_effect_defaults",
        "stamina_break_immunity_effect",
        "stamina_break_immunity_sword",
        "stamina_break_immunity_mace",
        "stamina_break_immunity_shield",
        "dodge_left_effect",
        "dodge_right_effect",
        "dodge_invulnerability_effect",
        "dodge_invulnerability_projection",
    )
)
_STATUS_FIELDS = frozenset(
    (
        "effect_id",
        "effect_asset",
        "projection",
        "classification",
        "mechanism",
        "duration_seconds",
        "overlap_behavior",
        "mechanics",
    )
)
_MECHANICS_FIELDS = frozenset(
    (
        "damage_calculator",
        "stat_modifier_effects",
        "entity_stats",
        "stat_modifiers",
        "damage_resistance",
        "invulnerable",
        "debuff",
        "horizontal_speed_multiplier",
        "knockback_multiplier",
    )
)


@dataclass(frozen=True)
class NativeStatusProjection:
    """One native effect whose actor projection is asset-certified."""

    effect_id: str
    effect_asset: str
    projection: str
    classification: str
    mechanism: str | None
    duration_seconds: float
    overlap_behavior: str


def native_status_projection_sha256() -> str:
    """Return the exact packaged native-status projection identity."""

    return hashlib.sha256(_native_status_projection_bytes()).hexdigest()


def load_native_status_projections() -> dict[str, NativeStatusProjection]:
    """Return an isolated effect-id keyed projection table."""

    return copy.deepcopy(_load_native_status_projections())


@lru_cache(maxsize=1)
def _native_status_projection_bytes() -> bytes:
    return (
        files("hytalegym.rulesets")
        .joinpath(NATIVE_STATUS_PROJECTION_RESOURCE)
        .read_bytes()
    )


@lru_cache(maxsize=1)
def _load_native_status_projections() -> dict[str, NativeStatusProjection]:
    document = json.loads(_native_status_projection_bytes())
    _require_fields(document, _ROOT_FIELDS, "native-status projection root")
    if (
        document["schema"] != NATIVE_STATUS_PROJECTION_SCHEMA
        or _integer(document["version"], "native-status projection version")
        != NATIVE_STATUS_PROJECTION_VERSION
        or document["hytale_server_version"] != "0.5.7"
        or document["assets_sha256"] != HYTALE_0_5_7_ASSETS_SHA256
    ):
        raise ValueError("unsupported or stale native-status projection table")
    _require_fields(
        document["provenance"],
        _PROVENANCE_FIELDS,
        "native-status projection provenance",
    )
    for name, value in document["provenance"].items():
        _nonempty_string(value, f"native-status provenance {name}")

    raw_statuses = document["statuses"]
    if not isinstance(raw_statuses, list) or not raw_statuses:
        raise ValueError("native-status projection table requires status rows")
    result: dict[str, NativeStatusProjection] = {}
    for raw in raw_statuses:
        _require_fields(raw, _STATUS_FIELDS, "native-status projection")
        effect_id = _nonempty_string(raw["effect_id"], "effect_id")
        if effect_id in result:
            raise ValueError(f"duplicate native-status projection: {effect_id!r}")
        projection = _nonempty_string(raw["projection"], "projection")
        if projection not in {
            NATIVE_STATUS_PROJECTION_OMIT_VISUAL_ONLY,
            NATIVE_STATUS_PROJECTION_OMIT_DERIVED_MECHANIC,
        }:
            raise ValueError(f"unsupported native-status projection: {projection!r}")
        classification = _nonempty_string(
            raw["classification"], "classification"
        )
        mechanism = raw["mechanism"]
        if projection == NATIVE_STATUS_PROJECTION_OMIT_VISUAL_ONLY:
            expected_classification = "live_native_visual_only"
            if mechanism is not None:
                raise ValueError("visual-only native status cannot name a mechanism")
        else:
            expected_classification = "live_native_mechanical_derived"
            mechanism = _nonempty_string(
                mechanism,
                "native-status mechanism",
            )
            if mechanism not in {
                NATIVE_STATUS_PROJECTION_DODGE_INVULNERABILITY,
                NATIVE_STATUS_PROJECTION_STAMINA_BREAK_IMMUNITY,
            }:
                raise ValueError(
                    "unsupported derived native-status mechanism: "
                    f"{mechanism!r}"
                )
        if classification != expected_classification:
            raise ValueError(
                f"unsupported native-status classification: {classification!r}"
            )
        mechanics = raw["mechanics"]
        _require_fields(mechanics, _MECHANICS_FIELDS, "native-status mechanics")
        for name in (
            "damage_calculator",
            "stat_modifier_effects",
            "entity_stats",
            "stat_modifiers",
            "damage_resistance",
            "debuff",
        ):
            if mechanics[name] is not False:
                raise ValueError(
                    f"projected native status has active generic mechanic {name!r}"
                )
        expected_invulnerable = (
            mechanism == NATIVE_STATUS_PROJECTION_DODGE_INVULNERABILITY
        )
        if mechanics["invulnerable"] is not expected_invulnerable:
            raise ValueError(
                "projected native status has an inconsistent invulnerable "
                f"mechanic for {mechanism!r}"
            )
        for name in ("horizontal_speed_multiplier", "knockback_multiplier"):
            if _finite_number(mechanics[name], name) != 1.0:
                raise ValueError(
                    f"projected native status has non-neutral {name!r}"
                )
        duration = _finite_number(raw["duration_seconds"], "duration_seconds")
        if duration <= 0.0:
            raise ValueError("native-status duration must be positive")
        result[effect_id] = NativeStatusProjection(
            effect_id=effect_id,
            effect_asset=_nonempty_string(raw["effect_asset"], "effect_asset"),
            projection=projection,
            classification=classification,
            mechanism=mechanism,
            duration_seconds=duration,
            overlap_behavior=_nonempty_string(
                raw["overlap_behavior"], "overlap_behavior"
            ),
        )
    return result


def _require_fields(value: Any, expected: frozenset[str], label: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown or missing:
        raise ValueError(
            f"{label} fields differ: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


__all__ = [
    "NATIVE_STATUS_PROJECTION_DODGE_INVULNERABILITY",
    "NATIVE_STATUS_PROJECTION_OMIT_DERIVED_MECHANIC",
    "NATIVE_STATUS_PROJECTION_OMIT_VISUAL_ONLY",
    "NATIVE_STATUS_PROJECTION_RESOURCE",
    "NATIVE_STATUS_PROJECTION_SCHEMA",
    "NATIVE_STATUS_PROJECTION_VERSION",
    "NATIVE_STATUS_PROJECTION_STAMINA_BREAK_IMMUNITY",
    "NativeStatusProjection",
    "load_native_status_projections",
    "native_status_projection_sha256",
]
