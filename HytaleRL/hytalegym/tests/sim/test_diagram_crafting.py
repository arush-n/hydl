"""Diagram crafting (gen 7, row 11q, crafting oracle stage 5): the Armory receipts replayed on the gen-7 render.

The fixture (``stage5_oracle_072.json``) copies the two axe receipts (seeds 570057 and 31337; FIDELITY-0.7.2.md section (v)):
the request at R, the queue 0 -> 1 on R, the inputs out and the axe in the inventory on R + 2 (native offsets 2 and 2). The
simulator queues on the request tick and commits on the second tick after it.
"""

from __future__ import annotations

import json
from collections import namedtuple
from importlib import import_module
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.jax.crafting import identity_sha256_words
from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import crafting_materials, diagram_crafting, timed_crafting
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7_OVERLAY
from hytalegym.sim.gen7.profile import PROFILE as GEN7
from hytalegym.worldgen import block_semantic_key
from hytalegym.worldgen.native_crafting import NativeCraftingRecipe

from .support import mount_fresh, unmount

FIXTURE = Path(__file__).with_name("stage5_oracle_072.json")
FIXTURE_DATA = json.loads(FIXTURE.read_text(encoding="utf-8"))
RECIPES = {record["recipe_id"]: record for record in FIXTURE_DATA["recipes"]}
DIAGRAM = FIXTURE_DATA["diagram"]
AXE = "Weapon_Axe_Stone_Trork_Recipe_Generated_0"
ARMORY = {"id": "Armory", "block_id": "Bench_Armory", "type": 2, "tier": 1}
COMBAT_FIELDS = ("diagram_recipe", "diagram_ticks")
Combat = namedtuple("Combat", COMBAT_FIELDS)
World = namedtuple("World", "combat inventory")
Craft = namedtuple("Craft", "accepted recipe_index")
Params = namedtuple("Params", "ticks_per_second")


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """The gen-7 render, rendered once for this module and mounted under its manifest name."""

    out = tmp_path_factory.mktemp("render") / "gen7"
    R.render(GEN7_OVERLAY, out, source=HeadSource())
    package = mount_fresh(out).__name__
    yield {
        "crafting": import_module(f"{package}.combat.arsenal.world_actions.crafting"),
        "contract": import_module(f"{package}.crafting.contract"),
        "packing": import_module(f"{package}.crafting.packing"),
        "inventory": import_module(f"{package}.combat.inventory"),
        "types": import_module(f"{package}.combat.types"),
        "semantic": import_module(f"{package}.combat.contracts.semantic"),
        "executor": import_module(f"{package}.combat.arsenal.world_actions.executor"),
    }
    unmount(package)


def _admitted_rows(records, module):
    """The rendered admission and resolution, as the compile does (benches of types 0, 1, 2 and 3 are admitted)."""

    benches = {b["id"] for record in records for b in record["bench_requirements"] if b["type"] in (0, 1, 2, 3)}
    admitted = benches | {"Fieldcraft"}
    rows = []
    for record in records:
        row = NativeCraftingRecipe.from_response(record)
        if module._lossless_region_crafting(row, admitted):
            rows.append(module._resolved_recipe(row))
    return rows


def _install_item_resource_types(engine, table) -> None:
    """The compile hook's item table (crafting_materials._TABLE), for the stock: the archive's inherited ResourceTypes.

    The hook fills the table only when the assets are compiled; this test builds its assets by hand, so it installs the same
    table for the four stock items (fixture ``item_resource_types``, read with LocalBlockSemanticsResolver on 2026-10-18).
    """

    semantic = engine["semantic"]
    capacity = engine["crafting"].MAX_RESOURCE_TYPES_PER_ITEM
    rows = {semantic.semantic_id(item): [semantic.semantic_id(name) for name in names] for item, names in table.items() if names}
    keys = np.asarray(sorted(rows), dtype=np.int32)
    values = np.full((len(keys), capacity), -1, dtype=np.int32)
    for row, key in enumerate(keys):
        values[row, : len(rows[int(key)])] = rows[int(key)]
    crafting_materials._TABLE.clear()
    crafting_materials._TABLE.update(keys=jnp.asarray(keys), values=jnp.asarray(values), jnp=jnp, absent=-1)


class _Armory:
    """The axe's assets on the rendered engine, its stock in the hotbar, and the construct's jitted commit."""

    def __init__(self, engine, stock):
        crafting, contract = engine["crafting"], engine["contract"]
        resolved = _admitted_rows([RECIPES[AXE]], crafting)
        assert len(resolved) == 1
        packed = engine["packing"].pack_resolved_recipe_table(
            tuple(resolved), source_sha256="0" * 64, expected_recipe_count=1,
            item_id_resolver=engine["semantic"].semantic_id, resource_type_id_resolver=engine["semantic"].semantic_id,
        )
        self.engine = engine
        self.layout = engine["inventory"].default_inventory_layout()
        words = np.asarray(identity_sha256_words(ARMORY["id"]), dtype=np.uint32)[None, :]
        self.assets = crafting.RegionFieldcraftAssets(
            packed=packed,
            output_max_stack=jnp.full((contract.RECIPE_TABLE_CAPACITY, contract.MAX_OUTPUTS), 64, dtype=jnp.int32),
            output_max_durability=jnp.zeros((contract.RECIPE_TABLE_CAPACITY, contract.MAX_OUTPUTS), dtype=jnp.float32),
            inventory_layout=self.layout, source_bridge_sha256="0" * 64,
            bench_semantic_key=jnp.asarray([block_semantic_key(ARMORY["block_id"], 0)], dtype=jnp.uint32),
            bench_id_hash=jnp.asarray(words, dtype=jnp.uint32), bench_type=jnp.asarray([2], dtype=jnp.int32),
            bench_tier_level=jnp.asarray([1], dtype=jnp.int32), bench_valid=jnp.ones((1,), dtype=jnp.bool_), memories_level=1,
        )
        self.index = list(packed.recipe_ids).index(AXE)
        _install_item_resource_types(engine, FIXTURE_DATA["item_resource_types"])
        self.crafting_function = crafting.execute_region_fieldcraft
        self.params = Params(ticks_per_second=30.0)
        self.stock_inventory = self._inventory(stock)
        self.world = World(combat=Combat(diagram_recipe=jnp.asarray([-1], dtype=jnp.int32),
                                         diagram_ticks=jnp.zeros((1,), dtype=jnp.int32)),
                           inventory=self.stock_inventory)
        self.commit = jax.jit(lambda world: diagram_crafting.advance(self.crafting_function, world, self.assets, self.layout))

    def _inventory(self, stock):
        engine = self.engine
        state = engine["inventory"].empty_inventory_state(1, entity_count=engine["types"].ENTITY_COUNT)
        hotbar = np.flatnonzero(np.asarray(self.layout.container_id) == engine["inventory"].CONTAINER_HOTBAR)
        item_id = np.asarray(state.item_id).copy()
        quantity = np.asarray(state.quantity).copy()
        for slot, stack in zip(hotbar, stock, strict=False):
            item_id[0, 0, slot] = engine["semantic"].semantic_id(stack["item_asset_id"])
            quantity[0, 0, slot] = stack["quantity"]
        return state._replace(item_id=jnp.asarray(item_id), quantity=jnp.asarray(quantity))

    def totals(self, inventory):
        out = {}
        for item, qty in zip(np.asarray(inventory.item_id)[0, 0].tolist(), np.asarray(inventory.quantity)[0, 0].tolist(), strict=True):
            if qty:
                out[item] = out.get(item, 0) + qty
        return out

    def queue(self, *, found=True, world=None):
        """The request on the tick R: the zero-time commit is undone (its inventory is the pre-craft one) and the craft queued."""

        world = self.world if world is None else world
        craft = Craft(accepted=jnp.asarray([True]), recipe_index=jnp.asarray([self.index], dtype=jnp.int32))
        # The zero-time commit of execute_region_fieldcraft would have consumed the inputs; the queue restores the pre-craft inventory.
        committed = world._replace(inventory=self.stock_inventory)
        accepted, state = diagram_crafting.queue_at(self.crafting_function, committed, craft, self.stock_inventory, self.assets,
                                                    jnp.asarray([found]))
        self.world = state
        return bool(accepted[0])

    def advance(self, ticks):
        for _ in range(ticks):
            self.world = self.commit(self.world)


def _stock():
    return [{"item_asset_id": "Plant_Fern_Jungle_Trunk", "quantity": 2}, {"item_asset_id": "Deco_Coral_Shell_Purple", "quantity": 6},
            {"item_asset_id": "Ingredient_Leather_Light", "quantity": 1}, {"item_asset_id": "Weapon_Spear_Leaf", "quantity": 1}]


@pytest.mark.parametrize("record", DIAGRAM, ids=[d["receipt"] for d in DIAGRAM])
def test_the_armory_request_is_queued_on_its_tick_and_commits_two_ticks_later(engine, record) -> None:
    armory = _Armory(engine, _stock())
    semantic = engine["semantic"]
    assert armory.queue() is True
    assert int(armory.world.combat.diagram_recipe[0]) == armory.index
    assert armory.totals(armory.world.inventory).get(semantic.semantic_id("Plant_Fern_Jungle_Trunk")) == 2
    armory.advance(1)
    assert int(armory.world.combat.diagram_recipe[0]) == armory.index
    assert armory.totals(armory.world.inventory).get(semantic.semantic_id("Plant_Fern_Jungle_Trunk")) == 2
    armory.advance(1)
    totals = armory.totals(armory.world.inventory)
    assert int(armory.world.combat.diagram_recipe[0]) == -1
    assert totals.get(semantic.semantic_id("Plant_Fern_Jungle_Trunk"), 0) == 0
    assert totals.get(semantic.semantic_id("Deco_Coral_Shell_Purple"), 0) == 0
    assert totals.get(semantic.semantic_id("Ingredient_Leather_Light"), 0) == 0
    assert totals.get(semantic.semantic_id("Weapon_Axe_Stone_Trork"), 0) == 1
    # Native: the queue goes 0 -> 1 on R and the inputs and the axe change on R + 2 (the record's offset).
    assert record["native"]["inputs_out_offset"] == 2


def test_a_second_request_while_one_is_queued_is_refused_and_keeps_the_inventory(engine) -> None:
    armory = _Armory(engine, _stock())
    assert armory.queue() is True
    queued = armory.world
    assert armory.queue(world=queued) is False
    assert int(armory.world.combat.diagram_recipe[0]) == armory.index


def test_without_an_armory_in_reach_the_zero_time_craft_is_not_queued(engine) -> None:
    armory = _Armory(engine, _stock())
    before = armory.world.inventory
    assert armory.queue(found=False) is True
    assert int(armory.world.combat.diagram_recipe[0]) == -1
    assert np.array_equal(np.asarray(armory.world.inventory.quantity), np.asarray(before.quantity))


def test_a_timed_diagram_recipe_stays_with_timed_crafting(engine) -> None:
    """Weapon_Club_Copper_Recipe_Generated_0 (3.0 s, Armory and a crafting bench): not a zero-time craft, so not queued here."""

    armory = _Armory(engine, _stock())
    jnp_ = jnp
    club = FIXTURE_DATA["extra_recipe_roles"]["timed_diagram"]
    rows = _admitted_rows([RECIPES[club]], engine["crafting"])
    packed = engine["packing"].pack_resolved_recipe_table(tuple(rows), source_sha256="0" * 64, expected_recipe_count=1,
                                                          item_id_resolver=engine["semantic"].semantic_id,
                                                          resource_type_id_resolver=engine["semantic"].semantic_id)
    index = jnp_.asarray([0], dtype=jnp_.int32)
    assert not bool(diagram_crafting.zero_time_lane(jnp_, packed.table, index)[0])
    assert bool(timed_crafting.timed_recipe(jnp_, packed.table, index)[0])


def test_idle_steps_change_no_leaf(engine) -> None:
    armory = _Armory(engine, _stock())
    before = jax.tree.map(np.asarray, armory.world)
    armory.advance(3)
    after = jax.tree.map(np.asarray, armory.world)
    assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True))


def test_the_gen5_profile_and_the_gen5_overlay_keep_every_original() -> None:
    """Gen 5 (``absent``) returns the original functions; no diagram entry is in the gen-5 overlay."""

    from hytalegym.sim.gen5.overlay import OVERLAY as GEN5_OVERLAY

    sentinel = object()
    assert GEN5.constructs.diagram_crafting == "absent"
    assert GEN7.constructs.diagram_crafting == "click_craft"
    assert diagram_crafting.admission(sentinel, GEN5) is sentinel
    assert diagram_crafting.executor(sentinel, GEN5) is sentinel
    assert not any("diagram" in getattr(hook, "id", "") for hook in GEN5_OVERLAY.hooks)
    assert not any("diagram" in getattr(patch, "id", "") for patch in GEN5_OVERLAY.patches)


def test_the_armory_recipe_passes_the_zero_time_gate(engine) -> None:
    rows = _admitted_rows([RECIPES[AXE]], engine["crafting"])
    assert len(rows) == 1 and rows[0].time_seconds == 0.0


def test_the_craft_verb_calls_the_diagram_start_and_the_hooks_are_rendered(engine) -> None:
    """Rendered wiring (static): the diagram start follows the processing start; the factory and admission are hooked."""

    import inspect

    executor_source = inspect.getsource(engine["executor"])
    assert executor_source.count("diagram_crafting_start(") == 1
    assert executor_source.index("processing_benches_start(") < executor_source.index("diagram_crafting_start(")
    assert executor_source.count("_render_hook_diagram_executor(") == 1
    assert inspect.getsource(engine["crafting"]).count("_render_hook_diagram_admission(") == 1
