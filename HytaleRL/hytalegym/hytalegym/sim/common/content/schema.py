"""Schema of the extracted content tables, the manifest and the overlay.

Tables are written by ``HytaleRL/hytalegym/tools/content_extract`` (the same
schema string is asserted equal by ``tests/sim``). Validation is structural
and recomputes each table's ``rows_sha256`` so a hand edit is caught.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

TABLE_SCHEMA = "hytalerl_content_table_v1"
MANIFEST_SCHEMA = "hytalerl_content_manifest_v1"
OVERLAY_SCHEMA = "hytalerl_content_calibration_v1"
FAMILIES = (
    "blocks",
    "consumables",
    "crafting",
    "durability",
    "movement",
    "npc_roles",
    "projectiles",
    "stats",
    "weapons",
)
_SHA256 = re.compile(r"[0-9A-F]{64}")
_HEADER = {
    "schema": str,
    "family": str,
    "extractor_version": int,
    "game_version": str,
    "assets_sha256": str,
    "source_count": int,
    "sources_sha256": str,
    "row_count": int,
    "rows_sha256": str,
    "rows": dict,
}
# family -> row-key prefix -> fields every such row carries.
ROW_KINDS: dict[str, dict[str, frozenset[str]]] = {
    "movement": {
        "config": frozenset({"kind", "source", "fields"}),
        "binding": frozenset({"kind", "source", "movement_config"}),
    },
    "stats": {
        "stat": frozenset({"kind", "source", "initial_value", "min", "max", "regenerating"}),
        "plugin": frozenset({"kind", "source", "sprint_regen_delay_value"}),
    },
    "durability": {
        "item": frozenset({"item_asset_id", "max_durability", "durability_loss_on_hit", "source"}),
    },
    "weapons": {
        "item": frozenset({"kind", "source", "parent_chain", "damage", "interaction_vars"}),
        "interaction": frozenset({"kind", "source", "type", "parent_chain", "document"}),
        "root": frozenset({"kind", "source", "document"}),
        "projectile_config": frozenset({"kind", "source", "parent_chain", "document"}),
        "effect": frozenset({"kind", "source", "duration", "document"}),
    },
    "projectiles": {
        "config": frozenset({"kind", "source", "launch_force", "gravity", "document"}),
        "projectile": frozenset({"kind", "source", "muzzle_velocity", "gravity", "document"}),
        "model": frozenset({"kind", "source", "hitbox"}),
    },
    "consumables": {
        "item": frozenset({"kind", "source", "effects_applied", "effects_cleared", "stat_changes"}),
        "effect": frozenset({"kind", "source"}),
    },
    "blocks": {
        "block": frozenset({"source"}),
        "hitbox": frozenset({"source", "boxes"}),
        "fluid": frozenset({"source", "document"}),
        "fluidfx": frozenset({"source", "document"}),
    },
    "crafting": {
        "recipe": frozenset(
            {
                "source",
                "inputs",
                "outputs",
                "primary_output",
                "bench_requirements",
                "time_seconds",
                "knowledge_required",
                "required_memories_level",
            }
        ),
    },
    "npc_roles": {
        "role": frozenset({"kind", "source", "type", "reference_chain", "parameters", "parameters_complete"}),
        "npc_root": frozenset({"kind", "source", "document"}),
        "npc_interaction": frozenset({"kind", "source", "type", "parent_chain", "document"}),
        "model": frozenset({"kind", "source", "hitbox", "eye_height"}),
    },
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest().upper()


def validate_table(table: Mapping[str, Any], *, family: str | None = None) -> list[str]:
    """Return every schema violation (empty when the table is valid)."""

    errors: list[str] = []
    for key, kind in _HEADER.items():
        if key not in table:
            errors.append(f"missing header field {key}")
        elif not isinstance(table[key], kind) or isinstance(table[key], bool):
            errors.append(f"{key} must be {kind.__name__}")
    if errors:
        return errors
    if table["schema"] != TABLE_SCHEMA:
        errors.append(f"schema {table['schema']!r} != {TABLE_SCHEMA!r}")
    name = table["family"]
    if name not in ROW_KINDS:
        return errors + [f"unknown family {name!r}"]
    if family is not None and name != family:
        errors.append(f"family {name!r} where {family!r} was expected")
    for key in ("assets_sha256", "sources_sha256", "rows_sha256"):
        if not _SHA256.fullmatch(table[key]):
            errors.append(f"{key} is not an upper-case SHA-256")
    rows = table["rows"]
    if table["row_count"] != len(rows):
        errors.append(f"row_count {table['row_count']} != {len(rows)} rows")
    if canonical_sha256(dict(sorted(rows.items()))) != table["rows_sha256"]:
        errors.append("rows_sha256 does not match the rows (edited by hand?)")
    kinds = ROW_KINDS[name]
    for key, row in rows.items():
        prefix = key.split(":", 1)[0]
        if prefix not in kinds or ":" not in key:
            errors.append(f"row {key!r}: unknown row kind for {name}")
            continue
        if not isinstance(row, dict):
            errors.append(f"row {key!r} is not an object")
            continue
        missing = kinds[prefix] - set(row)
        if missing:
            errors.append(f"row {key!r} misses {sorted(missing)}")
    sources = table.get("sources")
    if sources is not None:
        lines = "".join(f"{path}\t{sha}\n" for path, sha in sorted(sources.items()))
        if hashlib.sha256(lines.encode("utf-8")).hexdigest().upper() != table["sources_sha256"]:
            errors.append("sources do not match sources_sha256")
        if len(sources) != table["source_count"]:
            errors.append("source_count does not match sources")
    return errors


def validate_manifest(manifest: Mapping[str, Any], tables: Mapping[str, Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema") != MANIFEST_SCHEMA:
        errors.append(f"manifest schema {manifest.get('schema')!r} != {MANIFEST_SCHEMA!r}")
    entries = manifest.get("tables", {})
    if set(entries) != set(tables):
        errors.append(f"manifest tables {sorted(entries)} != files {sorted(tables)}")
    for family, entry in entries.items():
        table = tables.get(family)
        if table is None:
            continue
        for key in ("rows_sha256", "row_count", "extractor_version", "source_count", "sources_sha256"):
            if entry.get(key) != table.get(key):
                errors.append(f"manifest {family}.{key} disagrees with the table")
        if table.get("game_version") != manifest.get("game_version"):
            errors.append(f"{family} game_version disagrees with the manifest")
        if table.get("assets_sha256") != manifest.get("assets_sha256"):
            errors.append(f"{family} assets_sha256 disagrees with the manifest")
    expected = canonical_sha256({family: entry["rows_sha256"] for family, entry in sorted(entries.items())})
    if manifest.get("content_sha256") != expected:
        errors.append("manifest content_sha256 does not match its tables")
    return errors


def validate_overlay(overlay: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if overlay.get("schema") != OVERLAY_SCHEMA:
        errors.append(f"overlay schema {overlay.get('schema')!r} != {OVERLAY_SCHEMA!r}")
    for index, entry in enumerate(overlay.get("overrides", [])):
        for key in ("family", "row", "path", "value", "extracted", "constants", "classification", "evidence", "consumer"):
            if key not in entry:
                errors.append(f"override {index} misses {key}")
        if entry.get("classification") not in (
            "version_delta",
            "transcription_error",
            "native_measured",
            "unclassified",
        ):
            errors.append(f"override {index} has an unknown classification")
    return errors
