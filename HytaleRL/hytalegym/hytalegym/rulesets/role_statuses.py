"""Strict packaged compiler for role-authored initial entity effects."""

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
from hytalegym.rulesets.damage_causes import load_damage_cause_rules
from hytalegym.rulesets.resistances import (
    DamageResistanceProfile,
    compile_damage_resistance_profile,
)


ROLE_INITIAL_STATUSES_RESOURCE = "hytale_0_5_7/role_initial_statuses_v1.json"
ROLE_INITIAL_STATUSES_SCHEMA = "hytalerl_role_initial_statuses_v1"
ROLE_INITIAL_STATUSES_VERSION = 1

_ROOT_FIELDS = frozenset(
    (
        "schema",
        "version",
        "hytale_server_version",
        "assets_sha256",
        "provenance",
        "roles",
    )
)
_PROVENANCE_FIELDS = frozenset(
    (
        "npc_spawn_application",
        "entity_effect_defaults",
        "application_effect_defaults",
    )
)
_ROLE_FIELDS = frozenset(("role_id", "role_asset", "statuses"))
_STATUS_FIELDS = frozenset(
    (
        "effect_id",
        "effect_asset",
        "infinite",
        "duration_seconds",
        "debuff",
        "invulnerable",
        "cycle_cooldown_seconds",
        "damage_per_cycle",
        "damage_cause",
        "healing_per_cycle",
        "resource_id",
        "resource_delta_per_cycle",
        "speed_multiplier",
        "overlap_mode",
        "flags",
        "damage_resistance",
    )
)
_OVERLAP_MODES = {"Ignore": 0, "Extend": 1, "Overwrite": 2}
_STATUS_FLAGS = {
    "Invulnerable": 1 << 0,
    "DisableMovement": 1 << 1,
    "DisableAbilities": 1 << 2,
    "IgnoreKnockback": 1 << 3,
    "Debuff": 1 << 4,
    "DisableSprint": 1 << 5,
    "ControlImmunityGated": 1 << 6,
}


@dataclass(frozen=True)
class RoleInitialStatus:
    """One complete fixed-shape status program attached by an NPC role."""

    role_id: str
    role_asset: str
    effect_id: str
    effect_asset: str
    infinite: bool
    duration_seconds: float | None
    debuff: bool
    invulnerable: bool
    cycle_cooldown_seconds: float
    damage_per_cycle: float
    damage_cause: int
    healing_per_cycle: float
    resource_id: int
    resource_delta_per_cycle: float
    speed_multiplier: float
    overlap_mode: int
    flags: int
    damage_resistance: DamageResistanceProfile


def role_initial_statuses_sha256() -> str:
    """Return the exact packaged role-status table identity."""

    return hashlib.sha256(_role_initial_statuses_bytes()).hexdigest()


def load_role_initial_statuses() -> dict[str, tuple[RoleInitialStatus, ...]]:
    """Return an isolated, asset-pinned role-to-status mapping."""

    return copy.deepcopy(_load_role_initial_statuses())


@lru_cache(maxsize=1)
def _role_initial_statuses_bytes() -> bytes:
    return (
        files("hytalegym.rulesets")
        .joinpath(ROLE_INITIAL_STATUSES_RESOURCE)
        .read_bytes()
    )


@lru_cache(maxsize=1)
def _load_role_initial_statuses() -> dict[str, tuple[RoleInitialStatus, ...]]:
    document = json.loads(_role_initial_statuses_bytes())
    _require_fields(document, _ROOT_FIELDS, "role-status root")
    if (
        document["schema"] != ROLE_INITIAL_STATUSES_SCHEMA
        or _integer(document["version"], "role-status version")
        != ROLE_INITIAL_STATUSES_VERSION
        or document["hytale_server_version"] != "0.5.7"
        or document["assets_sha256"] != HYTALE_0_5_7_ASSETS_SHA256
    ):
        raise ValueError("unsupported or stale role-initial-status table")
    _require_fields(
        document["provenance"],
        _PROVENANCE_FIELDS,
        "role-status provenance",
    )
    for name, value in document["provenance"].items():
        _nonempty_string(value, f"role-status provenance {name}")
    roles = document["roles"]
    if not isinstance(roles, list) or not roles:
        raise ValueError("role-initial-status table requires role rows")
    causes = {
        rule.asset_id: index for index, rule in enumerate(load_damage_cause_rules())
    }
    result: dict[str, tuple[RoleInitialStatus, ...]] = {}
    for raw_role in roles:
        _require_fields(raw_role, _ROLE_FIELDS, "role-status role")
        role_id = _nonempty_string(raw_role["role_id"], "role_id")
        role_asset = _nonempty_string(raw_role["role_asset"], "role_asset")
        if role_id in result:
            raise ValueError(f"duplicate role-initial-status row: {role_id!r}")
        raw_statuses = raw_role["statuses"]
        if not isinstance(raw_statuses, list):
            raise TypeError("role statuses must be a list")
        seen_effects: set[str] = set()
        statuses: list[RoleInitialStatus] = []
        for raw_status in raw_statuses:
            _require_fields(raw_status, _STATUS_FIELDS, "role status")
            effect_id = _nonempty_string(raw_status["effect_id"], "effect_id")
            if effect_id in seen_effects:
                raise ValueError(
                    f"duplicate initial effect {effect_id!r} for role {role_id!r}"
                )
            seen_effects.add(effect_id)
            infinite = _boolean(raw_status["infinite"], "infinite")
            duration = raw_status["duration_seconds"]
            if infinite:
                if duration is not None:
                    raise ValueError("infinite role status must have null duration")
                duration_seconds = None
            else:
                duration_seconds = _finite_number(duration, "duration_seconds")
                if duration_seconds <= 0.0:
                    raise ValueError("finite role status duration must be positive")
            damage_cause = _nonempty_string(
                raw_status["damage_cause"], "damage_cause"
            )
            if damage_cause not in causes:
                raise ValueError(
                    f"unknown role-status damage cause: {damage_cause!r}"
                )
            flags = raw_status["flags"]
            if not isinstance(flags, list) or any(
                not isinstance(value, str) for value in flags
            ):
                raise TypeError("role status flags must be a string list")
            unknown_flags = set(flags) - set(_STATUS_FLAGS)
            if unknown_flags:
                raise ValueError(
                    f"unknown role status flags: {sorted(unknown_flags)}"
                )
            if len(flags) != len(set(flags)):
                raise ValueError("role status flags must be unique")
            debuff = _boolean(raw_status["debuff"], "debuff")
            invulnerable = _boolean(raw_status["invulnerable"], "invulnerable")
            flag_bits = sum(_STATUS_FLAGS[value] for value in flags)
            if debuff:
                flag_bits |= _STATUS_FLAGS["Debuff"]
            if invulnerable:
                flag_bits |= _STATUS_FLAGS["Invulnerable"]
            overlap = _nonempty_string(raw_status["overlap_mode"], "overlap_mode")
            if overlap not in _OVERLAP_MODES:
                raise ValueError(f"unknown role status overlap mode: {overlap!r}")
            resource_id = _integer(raw_status["resource_id"], "resource_id")
            if resource_id < -1:
                raise ValueError("role status resource_id must be at least -1")
            resistance = compile_damage_resistance_profile(
                {"DamageResistance": raw_status["damage_resistance"]},
                source_kind="status",
            )
            statuses.append(
                RoleInitialStatus(
                    role_id=role_id,
                    role_asset=role_asset,
                    effect_id=effect_id,
                    effect_asset=_nonempty_string(
                        raw_status["effect_asset"], "effect_asset"
                    ),
                    infinite=infinite,
                    duration_seconds=duration_seconds,
                    debuff=debuff,
                    invulnerable=invulnerable,
                    cycle_cooldown_seconds=_nonnegative(
                        raw_status["cycle_cooldown_seconds"],
                        "cycle_cooldown_seconds",
                    ),
                    damage_per_cycle=_nonnegative(
                        raw_status["damage_per_cycle"], "damage_per_cycle"
                    ),
                    damage_cause=causes[damage_cause],
                    healing_per_cycle=_nonnegative(
                        raw_status["healing_per_cycle"], "healing_per_cycle"
                    ),
                    resource_id=resource_id,
                    resource_delta_per_cycle=_finite_number(
                        raw_status["resource_delta_per_cycle"],
                        "resource_delta_per_cycle",
                    ),
                    speed_multiplier=_positive(
                        raw_status["speed_multiplier"], "speed_multiplier"
                    ),
                    overlap_mode=_OVERLAP_MODES[overlap],
                    flags=flag_bits,
                    damage_resistance=resistance,
                )
            )
        result[role_id] = tuple(statuses)
    return result


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


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a bool")
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


def _positive(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


__all__ = [
    "ROLE_INITIAL_STATUSES_RESOURCE",
    "ROLE_INITIAL_STATUSES_SCHEMA",
    "ROLE_INITIAL_STATUSES_VERSION",
    "RoleInitialStatus",
    "load_role_initial_statuses",
    "role_initial_statuses_sha256",
]
