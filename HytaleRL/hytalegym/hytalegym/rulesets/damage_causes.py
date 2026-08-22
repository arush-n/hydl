"""Pinned Hytale 0.5.7 damage-cause assets and inheritance metadata."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib.resources import files
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from hytalegym.combat.assets import (
    HYTALE_0_5_7_ASSETS_SHA256,
)
from hytalegym.rulesets.asset_surface import require_hytale_0_5_7_assets_archive


DAMAGE_CAUSE_RULESET_SCHEMA = "hytalerl_damage_causes_v1"
DAMAGE_CAUSE_RULESET_RESOURCE = "hytale_0_5_7/damage_causes_v1.json"
_DAMAGE_CAUSE_PREFIX = "Server/Entity/Damage/"


@dataclass(frozen=True)
class DamageCauseRule:
    """One resolved server damage-cause asset."""

    index: int
    asset_id: str
    inherits: str | None
    durability_loss: bool
    stamina_loss: bool
    bypass_resistances: bool


def load_damage_cause_rules() -> tuple[DamageCauseRule, ...]:
    """Return the validated, asset-versioned damage-cause table."""

    return _load_damage_cause_rules()


def damage_cause_rules_sha256() -> str:
    """Return the exact packaged table identity."""

    return hashlib.sha256(_damage_cause_rules_bytes()).hexdigest().upper()


def audit_hytale_0_5_7_damage_causes(
    archive: str | Path | None = None,
) -> tuple[DamageCauseRule, ...]:
    """Resolve shipped assets and fail if they differ from the packaged table."""

    path = require_hytale_0_5_7_assets_archive(archive)
    expected = load_damage_cause_rules()
    expected_ids = tuple(rule.asset_id for rule in expected)
    with ZipFile(path) as source:
        paths = {
            Path(name).stem: name
            for name in source.namelist()
            if name.startswith(_DAMAGE_CAUSE_PREFIX)
            and name.endswith(".json")
            and "/" not in name[len(_DAMAGE_CAUSE_PREFIX) :]
        }
        if set(paths) != set(expected_ids):
            raise ValueError(
                "shipped damage-cause assets differ from the packaged table: "
                f"expected={sorted(expected_ids)}, actual={sorted(paths)}"
            )
        documents = {
            asset_id: _read_object(source, paths[asset_id]) for asset_id in expected_ids
        }

    resolved = tuple(
        _resolve_damage_cause(index, asset_id, documents)
        for index, asset_id in enumerate(expected_ids)
    )
    if resolved != expected:
        raise ValueError(
            "shipped damage-cause semantics differ from the packaged table"
        )
    return resolved


@lru_cache(maxsize=1)
def _damage_cause_rules_bytes() -> bytes:
    return (
        files("hytalegym.rulesets").joinpath(DAMAGE_CAUSE_RULESET_RESOURCE).read_bytes()
    )


@lru_cache(maxsize=1)
def _load_damage_cause_rules() -> tuple[DamageCauseRule, ...]:
    data = json.loads(_damage_cause_rules_bytes())
    if data.get("schema") != DAMAGE_CAUSE_RULESET_SCHEMA:
        raise ValueError("unsupported packaged damage-cause ruleset")
    if data.get("hytale_version") != "0.5.7":
        raise ValueError("damage-cause ruleset is not for Hytale 0.5.7")
    if data.get("assets_sha256") != HYTALE_0_5_7_ASSETS_SHA256:
        raise ValueError("damage-cause ruleset has the wrong asset identity")
    rows = data.get("causes")
    if not isinstance(rows, list) or not rows:
        raise ValueError("damage-cause ruleset must contain causes")

    result = tuple(_decode_rule(index, row) for index, row in enumerate(rows))
    ids = tuple(rule.asset_id for rule in result)
    if len(ids) != len(set(ids)):
        raise ValueError("damage-cause asset ids must be unique")
    if any(rule.inherits not in {*ids, None} for rule in result):
        raise ValueError("damage-cause inheritance references an unknown cause")
    return result


def _decode_rule(index: int, row: Any) -> DamageCauseRule:
    if not isinstance(row, dict):
        raise ValueError("each damage-cause row must be an object")
    expected_keys = {
        "id",
        "inherits",
        "durability_loss",
        "stamina_loss",
        "bypass_resistances",
    }
    if set(row) != expected_keys:
        raise ValueError(
            f"damage-cause row {index} fields differ from {sorted(expected_keys)}"
        )
    asset_id = row["id"]
    inherits = row["inherits"]
    flags = (
        row["durability_loss"],
        row["stamina_loss"],
        row["bypass_resistances"],
    )
    if not isinstance(asset_id, str) or not asset_id:
        raise ValueError("damage-cause id must be a non-empty string")
    if inherits is not None and not isinstance(inherits, str):
        raise ValueError("damage-cause inherits must be a string or null")
    if any(type(value) is not bool for value in flags):
        raise ValueError("damage-cause behavior fields must be booleans")
    return DamageCauseRule(index, asset_id, inherits, *flags)


def _resolve_damage_cause(
    index: int,
    asset_id: str,
    documents: dict[str, dict[str, Any]],
    active: tuple[str, ...] = (),
) -> DamageCauseRule:
    if asset_id in active:
        raise ValueError(
            f"damage-cause parent cycle: {' -> '.join((*active, asset_id))}"
        )
    document = documents[asset_id]
    parent_id = document.get("Parent")
    inherited = {}
    if isinstance(parent_id, str):
        if parent_id not in documents:
            raise ValueError(f"damage cause {asset_id} has unknown parent {parent_id}")
        parent = _resolve_damage_cause(
            index,
            parent_id,
            documents,
            (*active, asset_id),
        )
        inherited = {
            "DurabilityLoss": parent.durability_loss,
            "StaminaLoss": parent.stamina_loss,
            "BypassResistances": parent.bypass_resistances,
        }

    def flag(name: str) -> bool:
        value = document.get(name, inherited.get(name, False))
        if type(value) is not bool:
            raise ValueError(f"{asset_id}.{name} must be boolean")
        return value

    inherits = document.get("Inherits")
    if inherits is not None and not isinstance(inherits, str):
        raise ValueError(f"{asset_id}.Inherits must be a string or null")
    return DamageCauseRule(
        index,
        asset_id,
        inherits,
        flag("DurabilityLoss"),
        flag("StaminaLoss"),
        flag("BypassResistances"),
    )


def _read_object(source: ZipFile, path: str) -> dict[str, Any]:
    value = json.loads(source.read(path))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


__all__ = [
    "DAMAGE_CAUSE_RULESET_RESOURCE",
    "DAMAGE_CAUSE_RULESET_SCHEMA",
    "DamageCauseRule",
    "audit_hytale_0_5_7_damage_causes",
    "damage_cause_rules_sha256",
    "load_damage_cause_rules",
]
