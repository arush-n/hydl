"""Region crafting with resource-type inputs. Switch ``Constructs.crafting_materials``; row 11, stage 1.

A 0.7.2 recipe input can name a resource type instead of an item: "any 4 Wood_Trunk". The server excludes every one of
the recipe's own outputs from such an input (``applyOutputExclusions``; 984 of the 988 resource inputs in the 0.7.2
catalog exclude exactly their recipe's outputs). The Region crafting action admits only concrete-item recipes
(``combat/arsenal/world_actions/crafting._lossless_region_crafting``). The crafting runtime already matches a
resource-type input against a slot's resource types (``crafting/runtime._slot_matches``), but the Region inventory
adapter publishes none: every slot's ``resource_type_id`` is ``ABSENT_ID``.

Gen 7 (``resource_types``) adds stage 1 with six hooks:

- ``item_resource_types`` wraps ``compile_region_fieldcraft_assets``: the archive's item ``ResourceTypes`` become a
  table (item semantic id -> up to ``MAX_RESOURCE_TYPES_PER_ITEM`` resource semantic ids). The resource types are a
  property of the archive, not of the catalog.
- ``admission`` wraps ``_lossless_region_crafting``: a recipe also passes when each non-concrete input names only a
  resource type and excludes exactly the recipe's outputs; outputs stay concrete items.
- ``resolution`` wraps ``_resolved_recipe``: such inputs keep their resource type.
- ``inventory_resource_types`` wraps ``_crafting_inventory``, and ``commit_resource_types`` wraps
  ``craft_into_combat_inventory`` (the commit catalog): both look up each slot's resource types in the table.
- ``output_exclusions`` wraps ``crafting/runtime._slot_matches``: a resource-type input never matches a slot that
  holds one of its recipe's outputs.

On the 0.7.2 catalog this admits 66 recipes beyond stage 0 (197 -> 263). They use 11 resource types, each carried by
at least two 0.7.2 items. The four recipes whose resource input excludes more than their outputs (raw fish) stay
out. Gen 5 (``concrete_only``) returns every original.
"""

from __future__ import annotations

from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import Any

SWITCH = "crafting_materials"
IMPLEMENTED = True
CONCRETE_ONLY = "concrete_only"
RESOURCE_TYPES = "resource_types"

#: (sorted item semantic ids [N], resource semantic ids [N, MAX_RESOURCE_TYPES_PER_ITEM]) of the last compiled archive.
_TABLE: dict[str, Any] = {}


def _enabled(profile: Any) -> bool:
    value = profile.constructs.crafting_materials
    if value == CONCRETE_ONLY:
        return False
    if value != RESOURCE_TYPES:
        raise ValueError(f"unknown crafting_materials switch {value!r}")
    return True


def _resource_input(material: Any, outputs: set[str]) -> bool:
    return bool(
        not material.item_asset_id
        and material.resource_type_id
        and not material.item_tag_present
        and not material.metadata_json_sha256
        and set(material.excluded_item_asset_ids) == outputs
    )


def _stand_in(recipe: Any) -> tuple[Any, bool]:
    """The recipe with each admissible resource input replaced by a concrete stand-in, and whether any was."""

    outputs = {material.item_asset_id for material in recipe.outputs}
    inputs = tuple(
        replace(material, item_asset_id="resource:" + material.resource_type_id, resource_type_id="",
                excluded_item_asset_ids=())
        if _resource_input(material, outputs) else material
        for material in recipe.inputs
    )
    return replace(recipe, inputs=inputs), inputs != recipe.inputs


def item_resource_types(original: Any, profile: Any) -> Any:
    """Hook for ``compile_region_fieldcraft_assets``: record the archive's item resource types."""

    if not _enabled(profile):
        return original
    env = original.__globals__
    np = env["np"]
    jnp = env["jnp"]
    semantic_id = env["semantic_id"]
    capacity = env["MAX_RESOURCE_TYPES_PER_ITEM"]
    absent = env["ABSENT_ID"]

    @wraps(original)
    def compile_region_fieldcraft_assets(recording_path, inventory_layout, *, expected_bridge_sha256, assets_path=None):
        archive_path = env["default_hytale_assets_path"]() if assets_path is None else Path(assets_path)
        rows: dict[int, list[int]] = {}
        with env["HytaleAssetArchive"](archive_path) as archive:
            resolver = env["LocalBlockSemanticsResolver"](archive)
            for asset_id in archive.item_asset_ids():
                item, _provenance = resolver.load_inherited_item_asset(asset_id)
                names = [entry.get("Id") for entry in item.get("ResourceTypes") or () if isinstance(entry, dict)]
                names = [name for name in names if isinstance(name, str) and name]
                if names:
                    if len(names) > capacity:
                        raise ValueError(f"item {asset_id!r} has {len(names)} resource types (capacity {capacity})")
                    rows[semantic_id(asset_id)] = [semantic_id(name) for name in names]
        keys = np.asarray(sorted(rows), dtype=np.int32)
        values = np.full((len(keys), capacity), absent, dtype=np.int32)
        for row, key in enumerate(keys):
            values[row, : len(rows[int(key)])] = rows[int(key)]
        _TABLE.update(keys=jnp.asarray(keys), values=jnp.asarray(values), jnp=jnp, absent=absent)
        return original(recording_path, inventory_layout, expected_bridge_sha256=expected_bridge_sha256,
                        assets_path=assets_path)

    return compile_region_fieldcraft_assets


def slot_resource_types(item_id: Any) -> Any:
    """Resource semantic ids of each item id (``[..., MAX_RESOURCE_TYPES_PER_ITEM]``); absent for unknown items."""

    jnp, keys, values = _TABLE["jnp"], _TABLE["keys"], _TABLE["values"]
    position = jnp.clip(jnp.searchsorted(keys, item_id), 0, keys.shape[0] - 1)
    found = (keys[position] == item_id) & (item_id >= 0)
    return jnp.where(found[..., None], values[position], jnp.int32(_TABLE["absent"]))


def admission(original: Any, profile: Any) -> Any:
    """Hook for ``_lossless_region_crafting``: resource inputs that exclude exactly the recipe's outputs."""

    if not _enabled(profile):
        return original

    @wraps(original)
    def _lossless_region_crafting(recipe, admitted_bench_ids):
        if original(recipe, admitted_bench_ids):
            return True
        stand_in, changed = _stand_in(recipe)
        return changed and original(stand_in, admitted_bench_ids)

    return _lossless_region_crafting


def resolution(original: Any, profile: Any) -> Any:
    """Hook for ``_resolved_recipe``: resource inputs keep their resource type."""

    if not _enabled(profile):
        return original
    material_type = original.__globals__["ResolvedMaterial"]

    @wraps(original)
    def _resolved_recipe(recipe):
        stand_in, changed = _stand_in(recipe)
        resolved = original(stand_in)
        if not changed:
            return resolved
        inputs = tuple(
            material_type(value.quantity, resource_type_id=source.resource_type_id)
            if stand.item_asset_id != source.item_asset_id else value
            for value, stand, source in zip(resolved.inputs, stand_in.inputs, recipe.inputs, strict=True)
        )
        return replace(resolved, inputs=inputs)

    return _resolved_recipe


def inventory_resource_types(original: Any, profile: Any) -> Any:
    """Hook for ``_crafting_inventory``: each input slot's resource types (provider legality)."""

    if not _enabled(profile):
        return original

    @wraps(original)
    def _crafting_inventory(state, layout):
        inventory = original(state, layout)
        if not _TABLE:
            return inventory
        return inventory._replace(resource_type_id=slot_resource_types(inventory.item_id))

    return _crafting_inventory


def commit_resource_types(original: Any, profile: Any) -> Any:
    """Hook for ``craft_into_combat_inventory``: the commit catalog's slot resource types."""

    if not _enabled(profile):
        return original

    @wraps(original)
    def craft_into_combat_inventory(table, state, layout, context, recipe_index, catalog, **kwargs):
        if _TABLE:
            catalog = catalog._replace(slot_resource_type_id=slot_resource_types(state.item_id))
        return original(table, state, layout, context, recipe_index, catalog, **kwargs)

    return craft_into_combat_inventory


def output_exclusions(original: Any, profile: Any) -> Any:
    """Hook for ``_slot_matches``: a resource-type input never consumes its recipe's own output item."""

    if not _enabled(profile):
        return original
    jnp = original.__globals__["jnp"]

    @wraps(original)
    def _slot_matches(recipe, inventory, input_index, slot):
        matches = original(recipe, inventory, input_index, slot)
        resource_input = (recipe["input_item_id"][input_index] < 0) & (recipe["input_resource_type_id"][input_index] >= 0)
        own_output = jnp.any(recipe["output_mask"] & (recipe["output_item_id"] == inventory.item_id[slot]))
        return matches & ~(resource_input & own_output)

    return _slot_matches
