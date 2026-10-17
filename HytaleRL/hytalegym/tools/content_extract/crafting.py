"""Crafting recipes, offline: standalone recipe assets plus item recipes.

The native capture (``tests/fidelity/surrogate_crafting_native_compare.py``)
needs a live bridge, so this extractor reproduces the server's own recipe
assembly from the archive instead (0.5.x decompile):

* ``Server/Item/Recipes/**.json``: one ``CraftingRecipe`` per file, id = stem
  (``AssetRegistryLoader``: path ``Item/Recipes``).
* ``CraftingRecipe.processConfig`` (at decode, for both kinds): an empty
  ``Output`` takes ``PrimaryOutput``.
* Items with their OWN ``Recipe`` (the field is a plain ``append`` codec field,
  so a child item does not inherit its parent's recipe): id
  ``<item>_Recipe_Generated_0`` and the primary-output rule of
  ``Item.processConfig``.
* ``applyOutputExclusions``: inputs without an ``ItemId`` exclude every
  output item id.

Field defaults are the codec's: ``Quantity`` 1, ``OutputQuantity`` 1,
``TimeSeconds`` 0, ``KnowledgeRequired`` false, ``RequiredMemoriesLevel`` 1,
``RequiredTierLevel`` 0.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from .archive import GameArchive

FAMILY = "crafting"
VERSION = 1
ITEM_PREFIX = "Server/Item/Items/"
RECIPE_PREFIX = "Server/Item/Recipes/"


def _material(value: dict[str, Any]) -> dict[str, Any]:
    """Absent ``ItemId``/``ResourceTypeId``/``ItemTag``/``Metadata`` are omitted."""

    material: dict[str, Any] = {"quantity": int(value.get("Quantity", 1))}
    for key, name in (
        ("ItemId", "item_id"),
        ("ResourceTypeId", "resource_type_id"),
        ("ItemTag", "item_tag"),
        ("Metadata", "metadata"),
    ):
        if value.get(key) is not None:
            material[name] = value[key]
    return material


def _bench(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": value.get("Type"),
        "id": value.get("Id"),
        "categories": list(value.get("Categories") or []),
        "required_tier_level": int(value.get("RequiredTierLevel", 0)),
    }


def _recipe(raw: dict[str, Any], *, source: str, item_id: str | None) -> dict[str, Any]:
    inputs = [_material(entry) for entry in raw.get("Input") or []]
    outputs = [_material(entry) for entry in raw.get("Output") or []]
    primary = raw.get("PrimaryOutput")
    primary = _material(primary) if isinstance(primary, dict) else None
    output_quantity = int(raw.get("OutputQuantity", 1))
    # CraftingRecipe.processConfig (codec decode): no Output -> [PrimaryOutput].
    if not outputs and primary is not None:
        outputs = [primary]
    # Item.processConfig for recipes an item generates.
    if item_id is not None:
        if len(outputs) == 1 and outputs[0].get("item_id") == item_id:
            primary = outputs[0]
        else:
            primary = {"quantity": output_quantity, "item_id": item_id}
            if not outputs:
                outputs = [primary]
    output_ids = sorted(
        {entry["item_id"] for entry in outputs if entry.get("item_id") is not None}
        | ({primary["item_id"]} if primary and primary.get("item_id") is not None else set())
    )
    for entry in inputs:
        if entry.get("item_id") is None and output_ids:
            entry["excluded_item_ids"] = output_ids
    row = {
        "source": source,
        "inputs": inputs,
        "outputs": outputs,
        "primary_output": primary,
        "bench_requirements": [_bench(entry) for entry in raw.get("BenchRequirement") or []],
        "time_seconds": float(raw.get("TimeSeconds", 0.0)),
        "knowledge_required": bool(raw.get("KnowledgeRequired", False)),
        "required_memories_level": int(raw.get("RequiredMemoriesLevel", 1)),
    }
    if item_id is not None:
        row["generated_from_item"] = item_id
    return row


def extract(archive: GameArchive) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for path in archive.list(RECIPE_PREFIX):
        raw = archive.read_json(path)
        recipe_id = PurePosixPath(path).stem
        if "Parent" in raw:
            raise ValueError(f"standalone recipe with Parent is not modelled: {path}")
        rows[f"recipe:{recipe_id}"] = _recipe(raw, source=path, item_id=None)
    for item_id, path in sorted(archive.index(ITEM_PREFIX).items()):
        raw = archive.read_json(path)
        recipe = raw.get("Recipe") if isinstance(raw, dict) else None
        if not isinstance(recipe, dict):
            continue
        recipe_id = f"{item_id}_Recipe_Generated_0"
        if f"recipe:{recipe_id}" in rows:
            raise ValueError(f"duplicate recipe id {recipe_id}")
        rows[f"recipe:{recipe_id}"] = _recipe(recipe, source=path, item_id=item_id)
    return rows
