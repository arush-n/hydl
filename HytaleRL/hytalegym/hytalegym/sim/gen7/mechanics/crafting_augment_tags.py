"""Crafting augment tags (gen 7, row 11x). Switch ``Constructs.crafting_augment_tags``.

0.7.2 ``CraftingManager.isValidBenchForRecipe`` checks ``benchHasRequiredAugmentTags`` in each per-requirement match (javap of
the 0.7.2 ``HytaleServer.jar``): every tag in the requirement's ``requiredAugmentTags`` must be in
``getBenchGrantedAugmentTags``, the bench's ``BenchBlock`` set, which the ``AugmentBlock``s near the bench refresh. A null or
empty list passes.

The tags are asset data. The table is ``sim/gen7/data/crafting_augment_tags.json``, extracted from the 0.7.2 archive by
``tools/content_extract/crafting_augment_tags.py`` (provenance inside the file). This module reads that table and never the
archive, at import or at call time, so importing the rendered engine needs no archive variables.

The engine's ``RecipeTable`` and ``CraftingContext`` carry no augment field, so the gen-7 render adds ``bench_augment_mask`` to
``CraftingContext`` and to ``RegionFieldcraftAssets`` by overlay patches (gen7/overlay.py, row 11x), carries each bench row's mask
into its bench context, and two hooks do the rest:

- ``compile_hook`` (``compile_region_fieldcraft_assets``): the scenario's grants, ``bench_augment_tags={bench id: tags}``, default
  none, become one uint32 mask per bench row.
- ``gate_hook`` (``crafting/runtime._recipe_precondition``): a requirement slot whose tags the context's bench does not grant is
  cleared from the recipe's ``requirement_mask`` before the engine's own precondition runs (bench match, knowledge and memories
  are unchanged).

Gen 5 (``absent``) returns the originals and reads nothing.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from hytalegym.jax.crafting import identity_sha256_words

SWITCH = "crafting_augment_tags"
IMPLEMENTED = True
ABSENT = "absent"
BENCH_GRANTS = "bench_grants"
#: The checked-in table, written by tools/content_extract/crafting_augment_tags.py.
DATA_FILE = "crafting_augment_tags.json"
FAMILY = "crafting_augment_tags"
TABLE_SCHEMA = "hytalerl_content_table_v1"
#: One bit per tag in the uint32 masks.
MAX_TAGS = 32


@dataclass(frozen=True)
class AugmentTable:
    """The vocabulary (bit i is ``tags[i]``), the tagged requirements and the grant assets, from the checked-in table."""

    tags: tuple[str, ...]
    requirements: tuple[tuple[str, str, tuple[str, ...]], ...]  # (recipe id, bench id, required tags)
    grants: tuple[tuple[str, tuple[str, ...]], ...]  # (asset id, granted tags)


def _enabled(profile: Any) -> bool:
    value = profile.constructs.crafting_augment_tags
    if value == ABSENT:
        return False
    if value != BENCH_GRANTS:
        raise ValueError(f"unknown crafting_augment_tags switch {value!r}")
    return True


@lru_cache(maxsize=None)
def _read(path: str) -> AugmentTable:
    table = json.loads(Path(path).read_text(encoding="utf-8"))
    if table.get("schema") != TABLE_SCHEMA or table.get("family") != FAMILY:
        raise ValueError(f"{path} is not a {FAMILY} table ({TABLE_SCHEMA})")
    requirements: list[tuple[str, str, tuple[str, ...]]] = []
    grants: list[tuple[str, tuple[str, ...]]] = []
    for key in sorted(table["rows"]):
        row = table["rows"][key]
        if key.startswith("requirement:"):
            requirements.append((row["recipe_id"], row["bench_id"], tuple(row["tags"])))
        elif key.startswith("grant:"):
            grants.append((row["asset_id"], tuple(row["tags"])))
        else:
            raise ValueError(f"unknown row {key!r} in {path}")
    names = sorted({tag for _recipe, _bench, tags in requirements for tag in tags}
                   | {tag for _asset, tags in grants for tag in tags})
    if len(names) > MAX_TAGS:
        raise ValueError(f"{len(names)} augment tags exceed the {MAX_TAGS}-bit masks")
    return AugmentTable(tuple(names), tuple(requirements), tuple(grants))


def load_table(data_dir: str | Path) -> AugmentTable:
    """The checked-in table in ``data_dir`` (gen 7: ``profile.data_dir``). It never reads the archive."""

    return _read(str(Path(data_dir) / DATA_FILE))


def tag_bits(table: AugmentTable, tags: Iterable[str]) -> int:
    if isinstance(tags, str):
        raise ValueError("augment tags are a collection of names, not one string")
    value = 0
    for tag in tags:
        if tag not in table.tags:
            raise ValueError(f"augment tag {tag!r} is not in the 0.7.2 vocabulary {table.tags}")
        value |= 1 << table.tags.index(tag)
    return value


def row_grant_masks(bench_rows: np.ndarray, grants: Mapping[str, Iterable[str]] | None, table: AugmentTable) -> np.ndarray:
    """One uint32 mask per bench row: the bits of the tags the scenario grants that row's bench id (default none)."""

    masks = np.zeros(bench_rows.shape[0], dtype=np.uint32)
    for bench_id, tags in (grants or {}).items():
        if not isinstance(bench_id, str) or not bench_id:
            raise ValueError("a grant names its bench by a nonempty string id")
        match = np.all(bench_rows == np.asarray(identity_sha256_words(bench_id), dtype=np.uint32), axis=1)
        if not match.any():
            raise ValueError(f"no compiled bench row has the id {bench_id!r}")
        masks[match] |= np.uint32(tag_bits(table, tags))
    return masks


def _requirement_arrays(table: AugmentTable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    recipes = np.asarray([identity_sha256_words(recipe) for recipe, _bench, _tags in table.requirements], dtype=np.uint32)
    benches = np.asarray([identity_sha256_words(bench) for _recipe, bench, _tags in table.requirements], dtype=np.uint32)
    bits = np.asarray([tag_bits(table, tags) for _recipe, _bench, tags in table.requirements], dtype=np.uint32)
    return recipes, benches, bits


def compile_hook(original: Any, profile: Any) -> Any:
    """Hook for ``compile_region_fieldcraft_assets``: the scenario's grants (``bench_augment_tags=``) as a mask per bench row."""

    if not _enabled(profile):
        return original
    table = load_table(profile.data_dir)

    @wraps(original)
    def compile_region_fieldcraft_assets(recording_path, inventory_layout, *args, bench_augment_tags=None, **kwargs):
        assets = original(recording_path, inventory_layout, *args, **kwargs)
        masks = row_grant_masks(np.asarray(assets.bench_id_hash), bench_augment_tags, table)
        return assets._replace(bench_augment_mask=jnp.asarray(masks, dtype=jnp.uint32))

    return compile_region_fieldcraft_assets


def gate_hook(original: Any, profile: Any) -> Any:
    """Hook for ``_recipe_precondition``: a requirement slot whose tags the context's bench does not grant is cleared first."""

    if not _enabled(profile):
        return original
    table = load_table(profile.data_dir)
    if not table.requirements:
        return original
    recipes, benches, bits = (jnp.asarray(value) for value in _requirement_arrays(table))

    @wraps(original)
    def _recipe_precondition(recipe: dict, context: Any) -> Any:
        recipe_match = jnp.all(recipe["recipe_id_hash"][None, :] == recipes, axis=-1)
        slot_match = jnp.all(recipe["requirement_bench_id_hash"][:, None, :] == benches[None, :, :], axis=-1)
        required = jnp.max(jnp.where(slot_match & recipe_match[None, :], bits[None, :], jnp.uint32(0)), axis=1)
        granted = jnp.uint32(0) if context.bench_augment_mask is None else context.bench_augment_mask
        narrowed = dict(recipe)
        narrowed["requirement_mask"] = recipe["requirement_mask"] & ((required & ~granted) == 0)
        return original(narrowed, context)

    return _recipe_precondition
