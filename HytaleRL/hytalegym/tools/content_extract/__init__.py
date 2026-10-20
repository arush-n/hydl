"""Content extractors: game files -> versioned JSON tables with provenance.

One module per content family. Each exposes ``FAMILY``, ``VERSION`` and
``extract(archive) -> rows``; :func:`extract_tables` wraps the rows with
provenance (asset paths and SHA-256, archive SHA-256, game version, extractor
version). Stdlib only, read-only on the archive. Run from this directory's
parent: ``python -m content_extract --help``.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from .archive import GameArchive
from .common import build_table, dump_table

FAMILY_MODULES = (
    "movement",
    "stats",
    "weapons",
    "projectiles",
    "consumables",
    "durability",
    "blocks",
    "crafting",
    "npc_roles",
)

#: Explicit families: extracted only when named (``--families``), never part of the content manifest. The gen-7
#: construct reads their checked-in table directly (see crafting_augment_tags.py).
EXTRA_FAMILY_MODULES = ("crafting_augment_tags", "hit_interrupt")


def family_module(name: str):
    if name not in FAMILY_MODULES + EXTRA_FAMILY_MODULES:
        raise KeyError(f"unknown content family {name!r}")
    return import_module(f"{__name__}.{name}")


def extract_table(archive: GameArchive, family: str) -> dict[str, Any]:
    module = family_module(family)
    sources = archive.begin_recording()
    try:
        rows = module.extract(archive)
    finally:
        sources = archive.end_recording(sources)
    return build_table(
        family=module.FAMILY,
        extractor_version=module.VERSION,
        game_version=archive.game_version,
        assets_sha256=archive.assets_sha256,
        sources=sources,
        rows=rows,
        notes=tuple(getattr(module, "NOTES", ())) + archive.last_duplicates,
    )


def extract_tables(
    archive: GameArchive, families: tuple[str, ...] = FAMILY_MODULES
) -> dict[str, dict[str, Any]]:
    return {family: extract_table(archive, family) for family in families}


def write_tables(tables: dict[str, dict[str, Any]], out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for family, table in sorted(tables.items()):
        path = out / f"{family}.json"
        path.write_text(dump_table(table), encoding="utf-8", newline="\n")
        written.append(path)
    return written


__all__ = [
    "FAMILY_MODULES",
    "GameArchive",
    "extract_table",
    "extract_tables",
    "family_module",
    "write_tables",
]
