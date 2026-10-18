"""Paths and helpers shared by the ``tests/sim`` suite."""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

HYTALEGYM = Path(__file__).resolve().parents[2]
TOOLS = HYTALEGYM / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
SIM = HYTALEGYM / "hytalegym" / "sim"
REPOSITORY = HYTALEGYM.parents[1]
INSTALL = Path.home() / "Library" / "Application Support" / "Hytale" / "install"
ARCHIVES = {
    "gen5": INSTALL / "v0.5" / "package" / "game" / "latest" / "Assets.zip",
    "gen7": INSTALL / "release" / "package" / "game" / "latest" / "Assets.zip",
}


def data_dir(generation: str) -> Path:
    return SIM / generation / "data"


def manifest(generation: str) -> dict:
    return json.loads((data_dir(generation) / "MANIFEST.json").read_text(encoding="utf-8"))


def table(generation: str, family: str) -> dict:
    return json.loads((data_dir(generation) / f"{family}.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def archive_sha256(path: str) -> str:
    from content_extract.archive import file_sha256

    return file_sha256(path)
