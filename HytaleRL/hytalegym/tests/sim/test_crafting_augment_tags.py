"""Crafting augment tags (gen 7, row 11x): a recipe that names augment tags is craftable only at a bench that grants them.

The tag table is the checked-in sim/gen7/data/crafting_augment_tags.json. The gate is checked on the rendered gen-7 engine (``_recipe_precondition``) over
the native catalog recording, which is the Region's recipe source. Gen 5 keeps the originals, and with no grants the gate changes
only the tagged recipes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.jax.crafting import identity_sha256_words
from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import crafting_augment_tags as aug
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7_OVERLAY
from hytalegym.sim.gen7.profile import PROFILE as GEN7

from .support import mount_fresh, render_for_test, unmount

RECORDING = Path(__file__).resolve().parents[1] / "fidelity" / "fixtures" / "world" / "v0_7" / "native_crafting_catalog_recording.json"
GOBLIN = "GoblinLab_Workbench"
FARMING = "Farmingbench"
POWER = "goblin_lab_power"
RECIPE_A = "goblin_lab_recipe_a"
GRANTS = {GOBLIN: (POWER, RECIPE_A)}
BREACH = "Goblin_Painting_Breach_Replica_Recipe_Generated_0"  # requires goblin_lab_power
BED = "Furniture_Goblin_Bed_Recipe_Generated_0"  # requires goblin_lab_power and goblin_lab_recipe_a


class _Context(NamedTuple):
    bench_augment_mask: object


def _bench(bench_id: str) -> np.ndarray:
    return np.asarray(identity_sha256_words(bench_id), dtype=np.uint32)


def test_gen5_keeps_the_original_functions() -> None:
    sentinel = object()
    assert aug.compile_hook(sentinel, GEN5) is sentinel
    assert aug.gate_hook(sentinel, GEN5) is sentinel


def test_the_fifty_tagged_recipes_are_enumerated_from_the_checked_in_table() -> None:
    table = aug.load_table(GEN7.data_dir)
    recipes = {recipe for recipe, _bench, _tags in table.requirements}
    assert len(table.requirements) == len(recipes) == 50
    assert {bench for _recipe, bench, _tags in table.requirements} == {GOBLIN}
    assert {tag for _recipe, _bench, tags in table.requirements for tag in tags} == {
        POWER, RECIPE_A, "goblin_lab_recipe_b", "goblin_lab_recipe_c"}
    assert [asset for asset, _tags in table.grants] == [
        "GoblinLab_Generator", "GoblinLab_Teleporter", "GoblinLab_Tinkering_Bench_Recipeset_A",
        "GoblinLab_Tinkering_Bench_Recipeset_B", "GoblinLab_Tinkering_Bench_Recipeset_C"]
    recording = json.loads(RECORDING.read_text(encoding="utf-8"))
    goblin = {row["recipe_id"] for row in recording["recipes"]
              if any(entry["id"] == GOBLIN for entry in row["bench_requirements"])}
    assert recipes == goblin
    assert {BREACH, BED} <= recipes


def test_grants_are_per_bench_row_and_unknown_names_are_refused() -> None:
    table = aug.load_table(GEN7.data_dir)
    rows = np.stack([_bench(GOBLIN), _bench(FARMING)])
    assert aug.row_grant_masks(rows, None, table).tolist() == [0, 0]
    assert aug.row_grant_masks(rows, {GOBLIN: [POWER]}, table).tolist() == [aug.tag_bits(table, [POWER]), 0]
    with pytest.raises(ValueError):
        aug.tag_bits(table, ["not_a_tag"])
    with pytest.raises(ValueError):
        aug.tag_bits(table, POWER)
    with pytest.raises(ValueError):
        aug.row_grant_masks(rows, {"NoSuchBench": [POWER]}, table)


def test_a_tagged_slot_is_cleared_unless_the_bench_grants_its_tags() -> None:
    table = aug.load_table(GEN7.data_dir)
    power = aug.tag_bits(table, [POWER])
    both = aug.tag_bits(table, [POWER, RECIPE_A])
    gate = aug.gate_hook(lambda recipe, context: jnp.any(recipe["requirement_mask"]), GEN7)

    def recipe(recipe_id: str) -> dict:
        return {
            "recipe_id_hash": jnp.asarray(identity_sha256_words(recipe_id), dtype=jnp.uint32),
            "requirement_bench_id_hash": jnp.asarray(np.stack([_bench(GOBLIN), np.zeros(8, np.uint32)]), dtype=jnp.uint32),
            "requirement_mask": jnp.asarray([True, False]),
        }

    def accepted(recipe_id: str, mask: int) -> bool:
        return bool(gate(recipe(recipe_id), _Context(jnp.uint32(mask))))

    assert not accepted(BREACH, 0)
    assert accepted(BREACH, power)
    assert not accepted(BED, 0)
    assert not accepted(BED, power)
    assert accepted(BED, both)


@pytest.fixture(scope="module")
def rendered(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("gen7_augment")
    out = render_for_test(GEN7_OVERLAY, out, source=HeadSource())
    return out


@pytest.fixture(scope="module")
def engine(rendered):
    name = mount_fresh(rendered).__name__
    yield name
    unmount(name)


def _assets(engine: str, grants, tmp_path: Path):
    """Compile from a private snapshot of the recording: other lanes regenerate the fixture directory, so the
    expected bridge pin and the compile input must come from the same bytes."""
    if not (os.environ.get("HYTALE_ASSETS_ZIP") or os.environ.get("APPDATA")):
        pytest.skip("the Region fieldcraft compile reads the recipe table from the 0.7.2 archive: set HYTALE_ASSETS_ZIP or APPDATA")
    crafting = import_module(f"{engine}.combat.arsenal.world_actions.crafting")
    inventory = import_module(f"{engine}.combat.inventory")
    snapshot = tmp_path / RECORDING.name
    shutil.copyfile(RECORDING, snapshot)
    recording = json.loads(snapshot.read_text(encoding="utf-8"))
    extra = {} if grants is None else {"bench_augment_tags": grants}
    return crafting.compile_region_fieldcraft_assets(
        snapshot, inventory.default_inventory_layout(), expected_bridge_sha256=recording["bridge_sha256"], **extra)


def _preconditions(engine: str, table, bench_id: str, mask: int, original: bool = False) -> np.ndarray:
    runtime = import_module(f"{engine}.crafting.runtime")
    types = import_module(f"{engine}.crafting.types")
    crafting = import_module(f"{engine}.crafting")
    function = runtime._recipe_precondition.__wrapped__ if original else runtime._recipe_precondition
    capacity = table.recipe_id_hash.shape[0]
    context = types.CraftingContext(
        window_open=jnp.bool_(True),
        bench_type=jnp.int32(0),
        bench_id_hash=jnp.asarray(identity_sha256_words(bench_id), dtype=jnp.uint32),
        bench_tier_level=jnp.int32(1),
        memories_level=jnp.int32(1),
        known_recipe=jnp.ones((capacity,), dtype=jnp.bool_),
        creative_mode=jnp.bool_(False),
        input_mode=jnp.int32(crafting.INPUT_MODE_NORMAL),
        bench_augment_mask=jnp.uint32(mask),
    )
    recipes = {field: getattr(table, field) for field in table._fields}
    recipes["_known"] = jnp.ones((capacity,), dtype=jnp.bool_)
    return np.asarray(jax.vmap(lambda recipe: function(recipe, context))(recipes))


def test_the_bench_rows_carry_the_grants_and_the_default_grants_nothing(engine: str, tmp_path: Path) -> None:
    table = aug.load_table(GEN7.data_dir)
    assert not np.asarray(_assets(engine, None, tmp_path).bench_augment_mask).any()
    granted = _assets(engine, GRANTS, tmp_path)
    at_goblin = np.all(np.asarray(granted.bench_id_hash) == _bench(GOBLIN), axis=1)
    masks = np.asarray(granted.bench_augment_mask)
    assert set(masks[at_goblin].tolist()) == {aug.tag_bits(table, [POWER, RECIPE_A])}
    assert not masks[~at_goblin].any()


def test_the_gate_on_the_rendered_engine_refuses_at_a_bare_bench_and_accepts_with_the_grants(engine: str, tmp_path: Path) -> None:
    table = aug.load_table(GEN7.data_dir)
    tagged = {recipe: tags for recipe, _bench, tags in table.requirements}
    assets = _assets(engine, GRANTS, tmp_path)
    ids = list(assets.packed.recipe_ids)
    row = {recipe: ids.index(recipe) for recipe in ids if recipe in tagged}
    admitted = sorted(row)
    # Furniture_Goblin_Bed is zero-time and not knowledge-required, so the Region admits it (the recording's row); the
    # knowledge-required rows (Goblin_Painting_Breach_Replica among them) are not admitted on either generation.
    assert BED in row
    assert np.array_equal(np.asarray(assets.packed.table.recipe_id_hash)[row[BED]], identity_sha256_words(BED))
    refused = _preconditions(engine, assets.packed.table, GOBLIN, 0)
    accepted = _preconditions(engine, assets.packed.table, GOBLIN, aug.tag_bits(table, [POWER, RECIPE_A]))
    assert not any(refused[row[recipe]] for recipe in admitted)
    expected = {recipe for recipe in admitted if set(tagged[recipe]) <= {POWER, RECIPE_A}}
    assert {recipe for recipe in admitted if accepted[row[recipe]]} == expected
    assert BED in expected


def test_defaults_leave_every_other_recipe_bit_identical(engine: str, tmp_path: Path) -> None:
    table = _assets(engine, None, tmp_path).packed
    tagged = {recipe for recipe, _bench, _tags in aug.load_table(GEN7.data_dir).requirements}
    ids = list(table.recipe_ids)
    farming = _preconditions(engine, table.table, FARMING, 0)
    assert np.array_equal(farming, _preconditions(engine, table.table, FARMING, 0, original=True))
    wrapped = _preconditions(engine, table.table, GOBLIN, 0)
    original = _preconditions(engine, table.table, GOBLIN, 0, original=True)
    changed = np.nonzero(wrapped != original)[0]
    assert {ids[index] for index in changed} <= tagged
    assert not wrapped[changed].any()


HYTALEGYM_ROOT = Path(__file__).resolve().parents[2]


def test_the_rendered_engine_imports_without_the_archive_environment(rendered: Path) -> None:
    """The gate reads the checked-in table, so the render imports in a process with no archive variables set."""

    code = (
        "from importlib import import_module\n"
        "from pathlib import Path\n"
        "from hytalegym.sim.common import render as R\n"
        f"name = R.mount(Path({str(rendered)!r})).__name__\n"
        "for module in ('crafting.runtime', 'crafting', 'combat.arsenal.world_actions.crafting'):\n"
        "    import_module(f'{name}.{module}')\n"
        "print('IMPORTED', name)\n"
    )
    env = {key: value for key, value in os.environ.items() if key not in ("HYTALE_ASSETS_ZIP", "APPDATA")}
    env.update(JAX_PLATFORMS="cpu", PYTHONPATH=str(HYTALEGYM_ROOT))
    completed = subprocess.run([sys.executable, "-c", code], env=env, cwd=HYTALEGYM_ROOT,
                               capture_output=True, text=True, timeout=900)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert "IMPORTED" in completed.stdout


def test_the_checked_in_table_matches_the_archive_when_the_variable_is_set() -> None:
    """Staleness is loud: with HYTALE_ASSETS_ZIP set, re-extract from that archive and compare the checked-in file."""

    archive = os.environ.get("HYTALE_ASSETS_ZIP")
    if not archive:
        pytest.skip("HYTALE_ASSETS_ZIP is unset: the archive check runs where the 0.7.2 archive is installed")
    from . import support  # puts tools/ on sys.path

    from content_extract import extract_table
    from content_extract.archive import GameArchive
    from content_extract.common import dump_table

    checked_in = (support.data_dir("gen7") / aug.DATA_FILE).read_text(encoding="utf-8")
    sha = support.archive_sha256(archive)
    recorded = json.loads(checked_in)["assets_sha256"]
    assert sha == recorded, (f"stale table: extracted from {recorded[:12]}, HYTALE_ASSETS_ZIP is {sha[:12]}; "
                             "re-extract (tools/content_extract/SKILL.md)")
    with GameArchive(archive, assets_sha256=sha) as found:
        assert dump_table(extract_table(found, aug.FAMILY)) == checked_in
