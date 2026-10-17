"""Per-version data-directory manifest and the content SHA-256."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .archive import GameArchive
from .common import canonical_bytes, dump_table, sha256_hex

MANIFEST_SCHEMA = "hytalerl_content_manifest_v1"
MANIFEST_NAME = "MANIFEST.json"


def content_sha256(rows_sha256: Mapping[str, str]) -> str:
    """Identity of a version's content: the families' row hashes, canonical.

    Provenance (which files were read) is deliberately excluded: an asset file
    that changes without changing any extracted row leaves the content, and the
    simulator built from it, unchanged.
    """

    return sha256_hex(canonical_bytes(dict(sorted(rows_sha256.items()))))


def build_manifest(tables: Mapping[str, Mapping[str, Any]], archive: GameArchive) -> dict[str, Any]:
    entries = {}
    for family, table in sorted(tables.items()):
        text = dump_table(table)
        entries[family] = {
            "file": f"{family}.json",
            "file_sha256": sha256_hex(text.encode("utf-8")),
            "rows_sha256": table["rows_sha256"],
            "row_count": table["row_count"],
            "extractor_version": table["extractor_version"],
            "source_count": table["source_count"],
            "sources_sha256": table["sources_sha256"],
        }
    return {
        "schema": MANIFEST_SCHEMA,
        "game_version": archive.game_version,
        "assets_sha256": archive.assets_sha256,
        "lenient_json_sources": sorted(archive.lenient),
        "tables": entries,
        "content_sha256": content_sha256(
            {family: entry["rows_sha256"] for family, entry in entries.items()}
        ),
    }


def write_manifest(manifest: Mapping[str, Any], out_dir: str | Path) -> Path:
    path = Path(out_dir) / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return path
