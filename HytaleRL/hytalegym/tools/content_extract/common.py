"""Shared pieces: asset inheritance, gameplay filtering, table assembly."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Callable, Mapping

TABLE_SCHEMA = "hytalerl_content_table_v1"
# Tables list every source file when they read at most this many; larger
# tables keep the digest and count only (each row still names its source).
INLINE_SOURCES_LIMIT = 400

# Codec fields whose child value replaces the parent's map instead of merging
# key-wise. ``DamageCalculator.BaseDamage`` is one inherited codec field: a
# child map replaces the parent's (rule copied from
# tests/combat/assets/weapon_variants/test_asset_scalar_variants.py::_merge).
REPLACE_KEYS = frozenset({"BaseDamage"})

# Presentation-only keys dropped when an extractor flattens gameplay data.
COSMETIC_KEYS = frozenset(
    {
        "$Comment",
        "$TODO",
        "Trails",
        "Particles",
        "WorldParticles",
        "LocalParticles",
        "CameraEffect",
        "CameraShake",
        "ItemAnimationId",
        "Animation",
        "AnimationId",
        "Icon",
        "IconProperties",
        "Texture",
        "Model",
        "TranslationProperties",
        "Light",
        "Sounds",
        "SoundEventId",
        "WorldSoundEventId",
        "LocalSoundEventId",
        "LaunchLocalSoundEventId",
        "LaunchWorldSoundEventId",
        "ItemSoundSetId",
        "BlockSoundSetId",
        "Description",
        "NameTranslationKey",
        "DisplayProgress",
    }
)


def merge_inherited(parent: Any, child: Any) -> Any:
    """Child-over-parent merge used by item, interaction and config assets."""

    if not isinstance(parent, dict) or not isinstance(child, dict):
        return copy.deepcopy(child)
    result = copy.deepcopy(parent)
    for key, value in child.items():
        if key in REPLACE_KEYS or key not in result:
            result[key] = copy.deepcopy(value)
        else:
            result[key] = merge_inherited(result[key], value)
    return result


def resolve_parented(
    stem: str,
    index: Mapping[str, str],
    read: Callable[[str], Any],
    *,
    parent_key: str = "Parent",
    capacity: int = 32,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Resolve ``stem`` through its ``Parent`` chain.

    Returns the merged document and the parent chain, nearest parent last
    (the same order ``combat/assets/catalog.py::_resolve_item`` reports).
    """

    chain: list[str] = []
    documents: list[dict[str, Any]] = []
    current: str | None = stem
    while current is not None:
        if current in chain or len(chain) > capacity:
            raise ValueError(f"parent cycle at {stem}: {chain}")
        if current not in index:
            if not documents:
                raise KeyError(current)
            raise KeyError(f"{stem} has unknown parent {current}")
        document = read(index[current])
        if not isinstance(document, dict):
            raise ValueError(f"{index[current]} is not a JSON object")
        documents.append(document)
        chain.append(current)
        parent = document.get(parent_key)
        current = parent if isinstance(parent, str) else None
    resolved: dict[str, Any] = {}
    for document in reversed(documents):
        resolved = merge_inherited(resolved, document) if resolved else copy.deepcopy(document)
    resolved.pop(parent_key, None)
    return resolved, tuple(reversed(chain[1:]))


def gameplay(value: Any) -> Any:
    """Drop cosmetic keys recursively; keep everything else verbatim."""

    if isinstance(value, dict):
        return {
            key: gameplay(child)
            for key, child in value.items()
            if key not in COSMETIC_KEYS and not key.endswith("SoundEventId")
        }
    if isinstance(value, list):
        return [gameplay(child) for child in value]
    return value


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sources_digest(sources: Mapping[str, str]) -> str:
    lines = "".join(f"{path}\t{sha}\n" for path, sha in sorted(sources.items()))
    return sha256_hex(lines.encode("utf-8"))


def build_table(
    *,
    family: str,
    extractor_version: int,
    game_version: str,
    assets_sha256: str,
    sources: Mapping[str, str],
    rows: Mapping[str, Any],
    notes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Assemble one deterministic table (no timestamps: re-runs are identical)."""

    ordered_rows = {key: rows[key] for key in sorted(rows)}
    table: dict[str, Any] = {
        "schema": TABLE_SCHEMA,
        "family": family,
        "extractor_version": extractor_version,
        "game_version": game_version,
        "assets_sha256": assets_sha256.upper(),
        "source_count": len(sources),
        "sources_sha256": sources_digest(sources),
        "row_count": len(ordered_rows),
        "rows_sha256": sha256_hex(canonical_bytes(ordered_rows)),
        "rows": ordered_rows,
    }
    if len(sources) <= INLINE_SOURCES_LIMIT:
        table["sources"] = dict(sorted(sources.items()))
    if notes:
        table["notes"] = list(notes)
    return table


def dump_table(table: Mapping[str, Any]) -> str:
    """Stable text form: header keys one per line, then one row per line.

    Valid JSON, sorted keys, deterministic, and diffable row by row.
    """

    def compact(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    lines = ["{"]
    header = [key for key in sorted(table) if key != "rows"]
    for key in header:
        lines.append(f" {compact(key)}:{compact(table[key])},")
    rows = table.get("rows", {})
    lines.append(' "rows":{')
    keys = sorted(rows)
    for position, key in enumerate(keys):
        comma = "," if position < len(keys) - 1 else ""
        lines.append(f"  {compact(key)}:{compact(rows[key])}{comma}")
    lines.append(" }")
    lines.append("}")
    return "\n".join(lines) + "\n"
