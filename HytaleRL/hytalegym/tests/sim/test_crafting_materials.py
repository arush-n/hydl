"""Crafting materials construct (gen 7, row 11 stage 1): gen 5 admits only concrete-item Region recipes; gen 7 also
admits resource-type inputs that exclude exactly their recipe's outputs, publishes slot resource types from the
archive's items, and never lets such an input consume the recipe's own output."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.jax.combat.arsenal.world_actions import crafting as region_crafting
from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.combat.inventory import default_inventory_layout, empty_inventory_state
from hytalegym.jax.combat.types import ENTITY_COUNT
from hytalegym.jax.crafting import runtime as crafting_runtime
from hytalegym.jax.crafting.contract import (
    ABSENT_ID,
    BENCH_TYPE_CRAFTING,
    MAX_OUTPUTS,
    MAX_RESOURCE_TYPES_PER_ITEM,
    RECIPE_TABLE_CAPACITY,
)
from hytalegym.jax.crafting.packing import (
    ResolvedBenchRequirement,
    ResolvedMaterial,
    ResolvedRecipe,
    pack_resolved_recipe_table,
)
from hytalegym.jax.world.interactions import crafting as candidates_module
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import crafting_materials
from hytalegym.sim.gen7.profile import PROFILE as GEN7
from hytalegym.worldgen.native_crafting import (
    NativeCraftingBenchRequirement,
    NativeCraftingMaterial,
    NativeCraftingRecipe,
)

HOOKS = (
    (crafting_materials.item_resource_types, region_crafting.compile_region_fieldcraft_assets),
    (crafting_materials.admission, region_crafting._lossless_region_crafting),
    (crafting_materials.resolution, region_crafting._resolved_recipe),
    (crafting_materials.inventory_resource_types, region_crafting._crafting_inventory),
    (crafting_materials.commit_resource_types, region_crafting.craft_into_combat_inventory),
    (crafting_materials.output_exclusions, crafting_runtime._slot_matches),
)


def test_gen5_returns_every_original() -> None:
    for hook, original in HOOKS:
        assert hook(original, GEN5) is original


def _material(item: str = "", resource: str = "", quantity: int = 1, excluded: tuple[str, ...] = ()):
    return NativeCraftingMaterial(item, resource, quantity, "", False, excluded)


def _native(inputs, output: str = "Planks") -> NativeCraftingRecipe:
    return NativeCraftingRecipe(
        recipe_id="Planks_Recipe",
        primary_output_item_asset_id=output,
        inputs=tuple(inputs),
        outputs=(_material(output, quantity=4),),
        bench_requirements=(NativeCraftingBenchRequirement(BENCH_TYPE_CRAFTING, "Fieldcraft", 0),),
        knowledge_required=False,
        required_memories_level=1,
        time_seconds=0.0,
    )


def test_gen7_admits_resource_inputs_that_exclude_exactly_their_outputs() -> None:
    admit = crafting_materials.admission(region_crafting._lossless_region_crafting, GEN7)
    benches = {"Fieldcraft"}
    exact = _native([_material(resource="Wood_Trunk", quantity=2, excluded=("Planks",))])
    assert not region_crafting._lossless_region_crafting(exact, benches)
    assert admit(exact, benches)
    assert not admit(_native([_material(resource="Wood_Trunk", excluded=("Planks", "Stick"))]), benches)
    assert not admit(_native([_material(resource="Wood_Trunk")]), benches)
    concrete = _native([_material("Oak_Trunk")])
    assert admit(concrete, benches) is region_crafting._lossless_region_crafting(concrete, benches)
    resolved = crafting_materials.resolution(region_crafting._resolved_recipe, GEN7)(exact)
    assert resolved.inputs == (ResolvedMaterial(2, resource_type_id="Wood_Trunk"),)


ITEMS = ("Oak_Trunk", "Planks", "Ingredient_Fibre")


@pytest.fixture
def resource_table(monkeypatch: pytest.MonkeyPatch):
    """Oak_Trunk and Planks both carry Wood_Trunk (Planks so the exclusion has something to exclude)."""

    keys = sorted(semantic_id(item) for item in ("Oak_Trunk", "Planks"))
    values = np.full((2, MAX_RESOURCE_TYPES_PER_ITEM), ABSENT_ID, dtype=np.int32)
    values[:, 0] = semantic_id("Wood_Trunk")
    monkeypatch.setattr(crafting_materials, "_TABLE", {
        "keys": jnp.asarray(keys, dtype=jnp.int32), "values": jnp.asarray(values), "jnp": jnp, "absent": ABSENT_ID})
    monkeypatch.setattr(crafting_runtime, "_slot_matches",
                        crafting_materials.output_exclusions(crafting_runtime._slot_matches, GEN7))
    monkeypatch.setattr(region_crafting, "_crafting_inventory",
                        crafting_materials.inventory_resource_types(region_crafting._crafting_inventory, GEN7))
    monkeypatch.setattr(region_crafting, "craft_into_combat_inventory",
                        crafting_materials.commit_resource_types(region_crafting.craft_into_combat_inventory, GEN7))


def test_slot_resource_types_come_from_the_table(resource_table) -> None:
    types = crafting_materials.slot_resource_types(
        jnp.asarray([semantic_id("Oak_Trunk"), semantic_id("Ingredient_Fibre"), ABSENT_ID], dtype=jnp.int32))
    assert int(types[0, 0]) == semantic_id("Wood_Trunk")
    assert (np.asarray(types[1:]) == ABSENT_ID).all()


PACKED = pack_resolved_recipe_table(
    (ResolvedRecipe(
        recipe_id="Planks_Recipe",
        inputs=(ResolvedMaterial(2, resource_type_id="Wood_Trunk"),),
        outputs=(ResolvedMaterial(4, item_asset_id="Planks"),),
        bench_requirements=(ResolvedBenchRequirement(BENCH_TYPE_CRAFTING, "Fieldcraft"),),
    ),),
    source_sha256="0" * 64,
    expected_recipe_count=1,
    item_id_resolver=semantic_id,
    resource_type_id_resolver=semantic_id,
)


def _craft(stack: dict[str, int]):
    layout = default_inventory_layout()
    state = empty_inventory_state(1, entity_count=ENTITY_COUNT)
    order = region_crafting.player_crafting_input_slot_order(layout)
    for slot, (item, quantity) in zip(order, stack.items()):
        state = state._replace(item_id=state.item_id.at[0, 0, slot].set(semantic_id(item)),
                               quantity=state.quantity.at[0, 0, slot].set(quantity))
    assets = region_crafting.RegionFieldcraftAssets(
        packed=PACKED,
        output_max_stack=jnp.full((RECIPE_TABLE_CAPACITY, MAX_OUTPUTS), 64, dtype=jnp.int32),
        output_max_durability=jnp.zeros((RECIPE_TABLE_CAPACITY, MAX_OUTPUTS), dtype=jnp.float32),
        inventory_layout=layout,
        source_bridge_sha256="0" * 64,
        bench_semantic_key=jnp.zeros((0,), dtype=jnp.uint32),
        bench_id_hash=jnp.zeros((0, 8), dtype=jnp.uint32),
        bench_type=jnp.zeros((0,), dtype=jnp.int32),
        bench_tier_level=jnp.zeros((0,), dtype=jnp.int32),
        bench_valid=jnp.zeros((0,), dtype=jnp.bool_),
    )
    inventory = region_crafting._crafting_inventory(region_crafting._actor_inventory(state), layout)
    legal = region_crafting._legal_recipes(inventory, region_crafting._fieldcraft_context(1, inventory.item_id.shape[1]),
                                           PACKED.table)
    candidates = candidates_module.produce_actor_recipe_candidates_from_legal_mask(
        PACKED.table, legal, catalog_available=True, legal_evidence_available=jnp.ones(legal.shape[:2], dtype=jnp.bool_))
    crafted = region_crafting.execute_region_fieldcraft(assets, state, candidates, jnp.asarray([0], dtype=jnp.int32))
    items = dict(zip(np.asarray(crafted.state.item_id[0, 0]).tolist(), np.asarray(crafted.state.quantity[0, 0]).tolist()))
    return bool(legal[0, 0, 0]), bool(crafted.accepted[0]), items


def test_gen7_crafts_from_a_resource_type_and_never_from_its_own_output(resource_table) -> None:
    legal, accepted, items = _craft({"Oak_Trunk": 3})
    assert legal and accepted
    assert items[semantic_id("Oak_Trunk")] == 1 and items[semantic_id("Planks")] == 4
    # Planks carry Wood_Trunk too, but the recipe outputs Planks: the server's output exclusion.
    legal, accepted, _ = _craft({"Planks": 8})
    assert not legal and not accepted


def test_without_the_table_no_slot_has_a_resource_type() -> None:
    legal, accepted, _ = _craft({"Oak_Trunk": 3})
    assert not legal and not accepted
