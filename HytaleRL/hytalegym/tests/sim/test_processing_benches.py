"""Processing benches (gen 7, row 11p, crafting oracle stage 5): the Campfire and Furnace receipts replayed on the gen-7 render.

The fixture (``stage5_oracle_072.json``) copies the four processing receipts (seeds 570057 and 31337; FIDELITY-0.7.2.md section (v))
and keeps each native tick beside the simulator's expected tick. Offsets count world ticks from the request tick R (offset 0).
The simulator counts 1/30 s per tick (the owner's 30 Hz ruling); the native steps advanced about 5 % more game time per tick, so
the native completion comes earlier (campfire 58/59 against 61; furnace 289/286 against 301). The first progress tick is R+2 on
both benches here; the furnace's native R+3 is one zero-advance world step, which the simulator does not have.
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
from hytalegym.sim.gen7.mechanics import processing_benches, timed_crafting
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7_OVERLAY
from hytalegym.sim.gen7.profile import PROFILE as GEN7
from hytalegym.worldgen import block_semantic_key
from hytalegym.worldgen.native_crafting import NativeCraftingRecipe

from .support import mount_fresh, unmount

FIXTURE = Path(__file__).with_name("stage5_oracle_072.json")
FIXTURE_DATA = json.loads(FIXTURE.read_text(encoding="utf-8"))
RECIPES = {record["recipe_id"]: record for record in FIXTURE_DATA["recipes"]}
PROCESSING = FIXTURE_DATA["processing"]
#: The simulator's first progress tick, the completion tick and the fuel and by-product ticks, from the 1/30 s rule (offsets).
SIM_FIRST_PROGRESS = 2
SIM_FUEL_OFFSETS = {"Campfire": [2], "Furnace": [2, 122, 242]}
SIM_COMPLETION = {"Campfire": 61, "Furnace": 301}
SIM_BYPRODUCT = {"Campfire": None, "Furnace": 122}

FIELDS = ("proc_bench_cell proc_recipe proc_active proc_has_fuel proc_wait_ticks proc_progress_seconds proc_fuel_seconds "
          "proc_fuel_item proc_fuel_quantity proc_fuel_quality proc_extra_countdown proc_input_item proc_input_quantity "
          "proc_output_item proc_output_quantity")
Combat = namedtuple("Combat", FIELDS)
World = namedtuple("World", "combat inventory")
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


def _assets(engine, recipe_id, bench):
    """The render's assets for one recipe and its bench row (a fresh bench, tier 1)."""

    crafting, contract = engine["crafting"], engine["contract"]
    resolved = _admitted_rows([RECIPES[recipe_id]], crafting)
    assert len(resolved) == 1, recipe_id
    packed = engine["packing"].pack_resolved_recipe_table(
        tuple(resolved),
        source_sha256="0" * 64,
        expected_recipe_count=1,
        item_id_resolver=engine["semantic"].semantic_id,
        resource_type_id_resolver=engine["semantic"].semantic_id,
    )
    words = np.asarray(identity_sha256_words(bench["id"]), dtype=np.uint32)[None, :]
    return crafting.RegionFieldcraftAssets(
        packed=packed,
        output_max_stack=jnp.full((contract.RECIPE_TABLE_CAPACITY, contract.MAX_OUTPUTS), 64, dtype=jnp.int32),
        output_max_durability=jnp.zeros((contract.RECIPE_TABLE_CAPACITY, contract.MAX_OUTPUTS), dtype=jnp.float32),
        inventory_layout=engine["inventory"].default_inventory_layout(),
        source_bridge_sha256="0" * 64,
        bench_semantic_key=jnp.asarray([block_semantic_key(bench["block_id"], 0)], dtype=jnp.uint32),
        bench_id_hash=jnp.asarray(words, dtype=jnp.uint32),
        bench_type=jnp.asarray([bench["type"]], dtype=jnp.int32),
        bench_tier_level=jnp.asarray([bench["tier"]], dtype=jnp.int32),
        bench_valid=jnp.ones((1,), dtype=jnp.bool_),
        memories_level=1,
    )


def _inventory(engine, layout, stock):
    """The stock in the hotbar (the first slots), in the order given."""

    state = engine["inventory"].empty_inventory_state(1, entity_count=engine["types"].ENTITY_COUNT)
    hotbar = np.flatnonzero(np.asarray(layout.container_id) == engine["inventory"].CONTAINER_HOTBAR)
    item_id = np.asarray(state.item_id).copy()
    quantity = np.asarray(state.quantity).copy()
    for slot, stack in zip(hotbar, stock, strict=False):
        item_id[0, 0, slot] = engine["semantic"].semantic_id(stack["item_asset_id"])
        quantity[0, 0, slot] = stack["quantity"]
    return state._replace(item_id=jnp.asarray(item_id), quantity=jnp.asarray(quantity))


def _idle_combat(batch=1):
    zeros_i = jnp.zeros((batch,), dtype=jnp.int32)
    return Combat(
        proc_bench_cell=jnp.zeros((batch, 3), dtype=jnp.int32),
        proc_recipe=jnp.full((batch,), -1, dtype=jnp.int32),
        proc_active=jnp.zeros((batch,), dtype=jnp.bool_),
        proc_has_fuel=jnp.zeros((batch,), dtype=jnp.bool_),
        proc_wait_ticks=zeros_i,
        proc_progress_seconds=jnp.zeros((batch,), dtype=jnp.float32),
        proc_fuel_seconds=jnp.zeros((batch,), dtype=jnp.float32),
        proc_fuel_item=jnp.full((batch,), -1, dtype=jnp.int32),
        proc_fuel_quantity=zeros_i,
        proc_fuel_quality=jnp.zeros((batch,), dtype=jnp.float32),
        proc_extra_countdown=jnp.full((batch,), 2, dtype=jnp.int32),
        proc_input_item=jnp.full((batch, 28), -1, dtype=jnp.int32),
        proc_input_quantity=jnp.zeros((batch, 28), dtype=jnp.int32),
        proc_output_item=jnp.full((batch, 4), -1, dtype=jnp.int32),
        proc_output_quantity=jnp.zeros((batch, 4), dtype=jnp.int32),
    )


class _Bench:
    """One scenario on the rendered engine: assets, the request and the jitted per-tick advance."""

    def __init__(self, engine, recipe_id, bench, stock):
        self.engine = engine
        self.recipe_id = recipe_id
        self.bench = bench
        self.assets = _assets(engine, recipe_id, bench)
        self.layout = self.assets.inventory_layout
        self.table = self.assets.packed.table
        self.index = list(self.assets.packed.recipe_ids).index(recipe_id)
        self.stock = stock
        self.params = Params(ticks_per_second=30.0)
        self.crafting_function = engine["crafting"].execute_region_fieldcraft
        self.advance = jax.jit(lambda world: processing_benches.advance(self.crafting_function, world, self.assets,
                                                                        self.params))
        self.world = World(combat=_idle_combat(), inventory=_inventory(engine, self.layout, stock))

    def request(self, *, has_fuel=True, found=True, inventory=None, combat=None):
        world = self.world if combat is None else self.world._replace(combat=combat)
        before = world.inventory if inventory is None else inventory
        started, inv, new_combat = processing_benches.request(
            self.crafting_function, world, jnp.asarray([self.index], dtype=jnp.int32), jnp.asarray([True]),
            jnp.asarray([[-2, 65, 0]], dtype=jnp.int32), jnp.asarray([found]), jnp.asarray([has_fuel]), before,
            self.assets, self.layout,
        )
        self.world = World(combat=new_combat, inventory=inv)
        return bool(started[0])

    def run(self, ticks):
        """Advance ``ticks`` world ticks; return the trace: offset -> (recipe, active, progress, fuel seconds, fuel quantity,
        by-product countdown, output stacks)."""

        trace = {}
        for offset in range(1, ticks + 1):
            self.world = self.advance(self.world)
            c = self.world.combat
            outputs = tuple(
                (int(item), int(qty))
                for item, qty in zip(np.asarray(c.proc_output_item)[0], np.asarray(c.proc_output_quantity)[0], strict=True)
                if qty
            )
            trace[offset] = (int(c.proc_recipe[0]), bool(c.proc_active[0]), float(c.proc_progress_seconds[0]),
                             float(c.proc_fuel_seconds[0]), int(c.proc_fuel_quantity[0]), int(c.proc_extra_countdown[0]),
                             outputs)
        return trace

    def totals(self, world=None):
        world = self.world if world is None else world
        out = {}
        for item, qty in zip(np.asarray(world.inventory.item_id)[0, 0].tolist(), np.asarray(world.inventory.quantity)[0, 0].tolist(),
                             strict=True):
            if qty:
                out[item] = out.get(item, 0) + qty
        return out


def _bench_of(record):
    return {"id": record["bench"]["bench_id"], "block_id": record["bench"]["block_id"], "type": 1, "tier": 1}


def _scenario(engine, name, stock=None):
    record = next(p for p in PROCESSING if p["receipt"] == name)
    stock = record["stock"] if stock is None else stock
    return _Bench(engine, record["recipe_id"], _bench_of(record), stock), record


SCENARIO_NAMES = [p["receipt"] for p in PROCESSING]


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_the_request_takes_the_inputs_and_the_fuel_out_of_the_inventory_and_activates(engine, name) -> None:
    scenario, record = _scenario(engine, name)
    assert scenario.request() is True
    totals = scenario.totals()
    semantic = engine["semantic"]
    for stack in record["stock"]:
        assert totals.get(semantic.semantic_id(stack["item_asset_id"]), 0) == 0
    c = scenario.world.combat
    assert int(c.proc_recipe[0]) == scenario.index
    assert bool(c.proc_active[0])
    assert int(c.proc_fuel_quantity[0]) == sum(s["quantity"] for s in record["stock"] if s["item_asset_id"] == "Ingredient_Stick")


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_the_first_progress_tick_is_two_ticks_after_the_request(engine, name) -> None:
    scenario, record = _scenario(engine, name)
    scenario.request()
    trace = scenario.run(3)
    assert trace[1][2] == 0.0 and trace[1][3] == 0.0 and trace[1][4] == int(record["stock"][1]["quantity"])
    assert trace[SIM_FIRST_PROGRESS][2] == pytest.approx(1.0 / 30.0, rel=1e-5)
    # Native: the first progress tick is A+2 on the Campfire and A+3 on the Furnace (one zero-advance world step).
    native = record["native"]["first_fuel_consumed_offset"]
    assert native in (2, 3)


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_fuel_burns_at_the_progress_rate_and_each_stick_lasts_four_seconds(engine, name) -> None:
    scenario, record = _scenario(engine, name)
    bench_id = record["bench"]["bench_id"]
    scenario.request()
    trace = scenario.run(SIM_COMPLETION[bench_id] + 1)
    consumed = [offset for offset in sorted(trace) if offset > 1 and trace[offset][4] < trace[offset - 1][4]]
    assert consumed == SIM_FUEL_OFFSETS[bench_id]
    # The lit stick loses one progress tick per tick: 4.0 s minus the progress made on the stick's ticks (a stick lasts 120 ticks).
    if bench_id == "Furnace":
        assert trace[122][3] == pytest.approx(4.0 - 1.0 / 30.0, rel=1e-4)


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_the_output_and_the_input_leave_on_the_completion_tick(engine, name) -> None:
    scenario, record = _scenario(engine, name)
    bench_id = record["bench"]["bench_id"]
    scenario.request()
    trace = scenario.run(SIM_COMPLETION[bench_id] + 1)
    done = SIM_COMPLETION[bench_id]
    assert trace[done - 1][0] == scenario.index and trace[done][0] == -1
    assert trace[done][1] is False
    assert trace[done][2] == 0.0
    # The native completion, beside the simulator's: earlier by about 5 % (the native world step is longer, see the module doc).
    native = record["native"]["output_elapsed_ticks"]
    assert 0 < done - native <= 0.06 * native + 1


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_fuel_left_at_completion_is_the_part_of_the_last_stick_that_the_recipe_did_not_burn(engine, name) -> None:
    scenario, record = _scenario(engine, name)
    bench_id = record["bench"]["bench_id"]
    scenario.request()
    trace = scenario.run(SIM_COMPLETION[bench_id] + 1)
    sticks = len(SIM_FUEL_OFFSETS[bench_id])
    time = float(RECIPES[record["recipe_id"]]["time_seconds"])
    assert trace[SIM_COMPLETION[bench_id]][3] == pytest.approx(4.0 * sticks - time, abs=1e-3)
    # One stick is left in the bench on both benches (2 inserted and 1 used; 4 inserted and 3 used).
    assert trace[SIM_COMPLETION[bench_id]][4] == 1
    # Native (receipts): 2.0 s left on the one remaining stick for both recipes.
    assert record["native"]["fuel_items_consumed"] == sticks


def test_the_furnace_second_fuel_consumption_yields_one_charcoal(engine) -> None:
    scenario, record = _scenario(engine, "v6_proc_furnace_copper_seed570057.json")
    scenario.request()
    trace = scenario.run(130)
    charcoal = {offset for offset, row in trace.items() if any(item and qty for item, qty in row[6])}
    assert min(charcoal) == SIM_BYPRODUCT["Furnace"]
    # The third consumption yields no second charcoal: the bench's output holds one charcoal before completion.
    assert trace[130][6] and trace[130][6][0][1] == 1
    native = record["native"]["first_byproduct_offset"]
    assert 0 < SIM_BYPRODUCT["Furnace"] - native <= 0.06 * native + 1


def test_an_idle_step_changes_no_leaf(engine) -> None:
    scenario = _scenario(engine, "v6_proc_campfire_fish_seed570057.json")[0]
    before = jax.tree.map(np.asarray, scenario.world.combat)
    scenario.run(2)
    after = jax.tree.map(np.asarray, scenario.world.combat)
    assert all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))


def test_a_request_without_fuel_on_a_fuel_bench_is_refused_and_changes_nothing(engine) -> None:
    """A fuel bench (the Campfire) refuses a request whose actor carries no stick (native setActive refuses an empty fuel slot)."""

    no_sticks = [stack for stack in _scenario(engine, "v6_proc_campfire_fish_seed570057.json")[1]["stock"]
                 if stack["item_asset_id"] != "Ingredient_Stick"]
    scenario = _scenario(engine, "v6_proc_campfire_fish_seed570057.json", stock=no_sticks)[0]
    inventory_before = scenario.world.inventory
    assert scenario.request(has_fuel=True) is False
    assert int(scenario.world.combat.proc_recipe[0]) == -1
    assert np.array_equal(np.asarray(scenario.world.inventory.quantity), np.asarray(inventory_before.quantity))


def test_a_request_is_refused_while_the_bench_runs(engine) -> None:
    scenario = _scenario(engine, "v6_proc_campfire_fish_seed570057.json")[0]
    assert scenario.request() is True
    combat = scenario.world.combat
    assert scenario.request(combat=combat) is False


def test_the_next_request_collects_the_previous_outputs_into_the_inventory(engine) -> None:
    """INFERRED (unmeasured collection): an output stack still in the bench moves to an empty inventory slot on a later request."""

    stock = [{"item_asset_id": "Food_Fish_Raw", "quantity": 2}, {"item_asset_id": "Ingredient_Stick", "quantity": 2}]
    scenario = _scenario(engine, "v6_proc_campfire_fish_seed570057.json", stock=stock)[0]
    scenario.request()
    scenario.run(62)
    assert int(np.asarray(scenario.world.combat.proc_output_quantity)[0].sum()) == 1
    assert scenario.request(has_fuel=True) is True
    assert int(np.asarray(scenario.world.combat.proc_output_quantity)[0].sum()) == 0
    food = engine["semantic"].semantic_id("Food_Fish_Grilled")
    assert scenario.totals().get(food, 0) == 1


def test_a_bench_without_a_fuel_slot_progresses_without_fuel(engine) -> None:
    """The Tannery (no Fuel slot, Bench_Tannery.json): 3.5 s of progress, no stick, completion on the 106th tick."""

    record = RECIPES[FIXTURE_DATA["extra_recipe_roles"]["tannery_no_fuel"]]
    bench = {"id": "Tannery", "block_id": "Bench_Tannery", "type": 1, "tier": 1}
    stock = [{"item_asset_id": record["inputs"][0]["item_asset_id"], "quantity": record["inputs"][0]["quantity"]}]
    scenario = _Bench(engine, record["recipe_id"], bench, stock)
    assert scenario.request(has_fuel=False) is True
    trace = scenario.run(2 + round(record["time_seconds"] * 30))
    completion = min(offset for offset, row in trace.items() if row[0] == -1)
    assert completion == 2 + round(record["time_seconds"] * 30) - 1
    assert not any(row[4] for row in trace.values())


def test_processing_recipes_pass_the_zero_time_gate_and_the_timed_construct_does_not_take_them(engine) -> None:
    """The timed admission keeps a processing recipe's time; the timed job predicate excludes processing recipes."""

    for recipe_id in ("Food_Fish_Grilled_Recipe_Generated_0", "Ingredient_Bar_Copper_Recipe_Generated_0"):
        rows = _admitted_rows([RECIPES[recipe_id]], engine["crafting"])
        assert len(rows) == 1 and rows[0].time_seconds > 0
    import jax.numpy as jnp_
    scenario, record = _scenario(engine, "v6_proc_campfire_fish_seed570057.json")
    index = jnp_.asarray([scenario.index], dtype=jnp_.int32)
    assert bool(timed_crafting.processing_recipe(jnp_, scenario.table, index)[0])
    assert not bool(timed_crafting.timed_recipe(jnp_, scenario.table, index)[0])


def test_the_gen5_profile_and_the_gen5_overlay_keep_every_original() -> None:
    """Gen 5 (``absent``) returns the original functions; no processing entry is in the gen-5 overlay."""

    from hytalegym.sim.gen5.overlay import OVERLAY as GEN5_OVERLAY

    sentinel = object()
    assert GEN5.constructs.processing_benches == "absent"
    assert processing_benches.admission(sentinel, GEN5) is sentinel
    assert processing_benches.executor(sentinel, GEN5) is sentinel
    assert processing_benches._enabled(GEN7) is True
    assert GEN7.constructs.processing_benches == "ticking_bench"
    assert not any("processing" in getattr(hook, "id", "") for hook in GEN5_OVERLAY.hooks)
    assert not any("processing" in getattr(patch, "id", "") for patch in GEN5_OVERLAY.patches)


def test_the_craft_verb_calls_the_processing_start_before_the_accepted_flag_and_the_hooks_are_rendered(engine) -> None:
    """Rendered wiring (static): the craft verb calls the processing and diagram starts, the factory and admission are hooked."""

    import inspect

    executor_source = inspect.getsource(engine["executor"])
    assert executor_source.count("processing_benches_start(") == 1
    assert executor_source.count("diagram_crafting_start(") == 1
    assert executor_source.index("processing_benches_start(") < executor_source.index("accepted = accepted.at[:, ACTION_SURFACE_VERB_CRAFT]")
    assert executor_source.count("_render_hook_processing_executor(") == 1
    crafting_source = inspect.getsource(engine["crafting"])
    assert crafting_source.count("_render_hook_processing_admission(") == 1
    assert crafting_source.count("_render_hook_diagram_admission(") == 1
