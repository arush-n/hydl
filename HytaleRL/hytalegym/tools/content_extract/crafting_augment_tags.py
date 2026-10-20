"""Crafting augment tags (gen 7, row 11x): the recipe tag requirements and the grant assets the construct reads.

An explicit family, not a manifest family: ``FAMILY_MODULES`` does not list it, so the gen-5 and gen-7 content identities are
unchanged. The construct reads ``sim/gen7/data/crafting_augment_tags.json`` directly and never the archive. Write it from a
scratch directory with an explicit family list, then copy the one file::

    python -m content_extract extract --assets <Assets.zip> --families crafting_augment_tags --out <scratch>
    cp <scratch>/crafting_augment_tags.json ../hytalegym/sim/gen7/data/

``verify --families crafting_augment_tags --assets <Assets.zip> --data ../hytalegym/sim/gen7/data`` checks the checked-in file
against the archive. A partial family list writes no manifest, so ``--out`` may also be the data directory.

0.7.2 facts (javap of ``Server/HytaleServer.jar``: ``CraftingManager.benchHasRequiredAugmentTags`` and
``getBenchGrantedAugmentTags``): an item's own ``Recipe.BenchRequirement[].RequiredAugmentTags`` is the generated recipe
``<item>_Recipe_Generated_0`` (the crafting extractor's id rule); explicit recipes under ``Server/Item/Recipes`` carry the same
field; any ``GrantsAugmentTags`` list on an item asset, at any depth, is a grant from an augment block near a bench.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

FAMILY = "crafting_augment_tags"
VERSION = 1
NOTES = ("extracted 2026-10-19 from the installed 0.7.2 Assets.zip (crafting_augment_tags v1); read by the gen-7 construct",)
ITEM_PREFIX = "Server/Item/Items/"
RECIPE_PREFIX = "Server/Item/Recipes/"
GENERATED_SUFFIX = "_Recipe_Generated_0"


def _granted(value: Any, found: list[str]) -> None:
    """Every ``GrantsAugmentTags`` list of a parsed asset, at any depth."""

    if isinstance(value, dict):
        for key, child in value.items():
            if key == "GrantsAugmentTags" and isinstance(child, list):
                found.extend(str(tag) for tag in child)
            else:
                _granted(child, found)
    elif isinstance(value, list):
        for child in value:
            _granted(child, found)


def _requirements(recipe: Any) -> list[tuple[str, list[str]]]:
    """(bench id, required tags) for each BenchRequirement that names augment tags."""

    if not isinstance(recipe, dict):
        return []
    return [
        (str(entry["Id"]), [str(tag) for tag in entry["RequiredAugmentTags"]])
        for entry in recipe.get("BenchRequirement") or []
        if entry.get("RequiredAugmentTags")
    ]


def _add(rows: dict[str, Any], key: str, row: dict[str, Any]) -> None:
    if key in rows:
        raise ValueError(f"duplicate augment-tag row {key!r}")
    rows[key] = row


def extract(archive: Any) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for path in archive.list(RECIPE_PREFIX):
        recipe_id = PurePosixPath(path).stem
        for bench_id, tags in _requirements(archive.read_json(path)):
            _add(rows, f"requirement:{recipe_id}|{bench_id}",
                 {"recipe_id": recipe_id, "bench_id": bench_id, "tags": tags, "source": path})
    for item_id, path in sorted(archive.index(ITEM_PREFIX).items()):
        raw = archive.read_json(path)
        recipe_id = f"{item_id}{GENERATED_SUFFIX}"
        for bench_id, tags in _requirements(raw.get("Recipe") if isinstance(raw, dict) else None):
            _add(rows, f"requirement:{recipe_id}|{bench_id}",
                 {"recipe_id": recipe_id, "bench_id": bench_id, "tags": tags, "source": path})
        found: list[str] = []
        _granted(raw, found)
        if found:
            _add(rows, f"grant:{item_id}", {"asset_id": item_id, "tags": sorted(set(found)), "source": path})
    return rows
