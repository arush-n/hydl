"""Bench crafting construct (gen 7, row 11 stage 0): gen 5 commits every Region craft under the Fieldcraft context and
fails an actor row closed above 16 legal recipes; gen 7 commits under a bench the recipe names and keeps the first 16
legal recipes in table order."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.jax.combat.types import ENTITY_COUNT
from hytalegym.jax.combat.arsenal.world_actions import crafting as region_crafting
from hytalegym.jax.combat.inventory import default_inventory_layout, empty_inventory_state
from hytalegym.jax.crafting.contract import BENCH_TYPE_CRAFTING, MAX_OUTPUTS, RECIPE_TABLE_CAPACITY
from hytalegym.jax.crafting.packing import (
    ResolvedBenchRequirement,
    ResolvedMaterial,
    ResolvedRecipe,
    pack_resolved_recipe_table,
)
from hytalegym.jax.world.interactions import crafting as candidates_module
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import bench_crafting
from hytalegym.sim.gen7.profile import PROFILE as GEN7

ITEMS = {"Ingredient_Fibre": 11, "Ingredient_Stick": 12, "Bench_Output": 13}
FIELDCRAFT = ResolvedBenchRequirement(BENCH_TYPE_CRAFTING, "Fieldcraft")
WORKBENCH = ResolvedBenchRequirement(BENCH_TYPE_CRAFTING, "Workbench")


def _recipe(recipe_id: str, requirement: ResolvedBenchRequirement, output: str = "Ingredient_Stick") -> ResolvedRecipe:
    return ResolvedRecipe(
        recipe_id=recipe_id,
        inputs=(ResolvedMaterial(1, item_asset_id="Ingredient_Fibre"),),
        outputs=(ResolvedMaterial(1, item_asset_id=output),),
        bench_requirements=(requirement,),
    )


# 20 Fieldcraft recipes and one Workbench-only recipe; the packer orders them by recipe id.
RECIPES = tuple(_recipe(f"Field_{n:02d}", FIELDCRAFT) for n in range(20)) + (
    _recipe("Workbench_Output", WORKBENCH, output="Bench_Output"),
)
PACKED = pack_resolved_recipe_table(
    RECIPES,
    source_sha256="0" * 64,
    expected_recipe_count=len(RECIPES),
    item_id_resolver=ITEMS.__getitem__,
    resource_type_id_resolver=lambda name: 0,
)
WORKBENCH_INDEX = PACKED.recipe_ids.index("Workbench_Output")
FIELD_INDEXES = [PACKED.recipe_ids.index(f"Field_{n:02d}") for n in range(20)]
ORIGINAL_PRODUCER = candidates_module.produce_actor_recipe_candidates_from_legal_mask
ORIGINAL_COMMIT = region_crafting.craft_into_combat_inventory


def _legal(indexes: list[int]) -> jnp.ndarray:
    return jnp.zeros((1, 1, RECIPE_TABLE_CAPACITY), dtype=jnp.bool_).at[0, 0, jnp.asarray(indexes, dtype=jnp.int32)].set(True)


def _candidates(producer, indexes: list[int]):
    return producer(PACKED.table, _legal(indexes), catalog_available=True,
                    legal_evidence_available=jnp.ones((1, 1), dtype=jnp.bool_))


def test_gen5_returns_both_originals() -> None:
    assert bench_crafting.first_candidates(ORIGINAL_PRODUCER, GEN5) is ORIGINAL_PRODUCER
    assert bench_crafting.bench_commit(ORIGINAL_COMMIT, GEN5) is ORIGINAL_COMMIT


def test_gen7_keeps_the_first_16_legal_recipes_in_table_order() -> None:
    gen5 = _candidates(ORIGINAL_PRODUCER, FIELD_INDEXES)
    assert not bool(gen5.available[0, 0]) and bool(gen5.capacity_exceeded[0, 0])
    gen7 = _candidates(bench_crafting.first_candidates(ORIGINAL_PRODUCER, GEN7), FIELD_INDEXES)
    assert bool(gen7.available[0, 0]) and not bool(gen7.capacity_exceeded[0, 0])
    assert np.asarray(gen7.candidate_mask[0, 0]).all()
    assert list(np.asarray(gen7.execution.recipe_index[0, 0])) == sorted(FIELD_INDEXES)[:16]
    assert (int(gen7.source_count[0, 0]), int(gen7.emitted_count[0, 0])) == (20, 16)
    assert int(gen7.diagnostics[0, 0]) & int(candidates_module.ACTOR_RECIPE_DIAGNOSTIC_CAPACITY)


def test_gen7_leaves_a_row_within_capacity_unchanged() -> None:
    hooked = bench_crafting.first_candidates(ORIGINAL_PRODUCER, GEN7)
    for indexes in (FIELD_INDEXES[:16], [WORKBENCH_INDEX], []):
        gen5, gen7 = _candidates(ORIGINAL_PRODUCER, indexes), _candidates(hooked, indexes)
        assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(gen5), jax.tree.leaves(gen7), strict=True))


def _assets() -> region_crafting.RegionFieldcraftAssets:
    return region_crafting.RegionFieldcraftAssets(
        packed=PACKED,
        output_max_stack=jnp.full((RECIPE_TABLE_CAPACITY, MAX_OUTPUTS), 64, dtype=jnp.int32),
        output_max_durability=jnp.zeros((RECIPE_TABLE_CAPACITY, MAX_OUTPUTS), dtype=jnp.float32),
        inventory_layout=default_inventory_layout(),
        source_bridge_sha256="0" * 64,
        bench_semantic_key=jnp.zeros((0,), dtype=jnp.int32),
        bench_id_hash=jnp.zeros((0, 8), dtype=jnp.uint32),
        bench_type=jnp.zeros((0,), dtype=jnp.int32),
        bench_tier_level=jnp.zeros((0,), dtype=jnp.int32),
        bench_valid=jnp.zeros((0,), dtype=jnp.bool_),
    )


def _craft(index: int, monkeypatch: pytest.MonkeyPatch | None = None):
    if monkeypatch is not None:
        monkeypatch.setattr(region_crafting, "craft_into_combat_inventory",
                            bench_crafting.bench_commit(ORIGINAL_COMMIT, GEN7))
    state = empty_inventory_state(1, entity_count=ENTITY_COUNT)
    state = state._replace(
        item_id=state.item_id.at[0, 0, 0].set(ITEMS["Ingredient_Fibre"]),
        quantity=state.quantity.at[0, 0, 0].set(3),
    )
    return region_crafting.execute_region_fieldcraft(
        _assets(), state, _candidates(ORIGINAL_PRODUCER, [index]), jnp.asarray([0], dtype=jnp.int32)
    )


def test_gen5_refuses_a_bench_recipe_at_the_commit() -> None:
    assert not bool(_craft(WORKBENCH_INDEX).accepted[0])
    assert bool(_craft(FIELD_INDEXES[0]).accepted[0])


def test_gen7_commits_a_bench_recipe_under_its_bench(monkeypatch: pytest.MonkeyPatch) -> None:
    gen5 = _craft(FIELD_INDEXES[0])
    crafted = _craft(WORKBENCH_INDEX, monkeypatch)
    assert bool(crafted.accepted[0])
    items = dict(zip(np.asarray(crafted.state.item_id[0, 0]).tolist(), np.asarray(crafted.state.quantity[0, 0]).tolist()))
    assert items[ITEMS["Ingredient_Fibre"]] == 2 and items[ITEMS["Bench_Output"]] == 1
    # A Fieldcraft recipe commits exactly as in gen 5.
    gen7 = _craft(FIELD_INDEXES[0])
    assert bool(gen7.accepted[0])
    assert np.array_equal(np.asarray(gen5.state.item_id), np.asarray(gen7.state.item_id))
    assert np.array_equal(np.asarray(gen5.state.quantity), np.asarray(gen7.state.quantity))
