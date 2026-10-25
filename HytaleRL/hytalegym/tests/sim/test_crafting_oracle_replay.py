"""Replay of the live 0.7.2 crafting oracle (FIDELITY-0.7.2.md section (s); fixture ``crafting_oracle_072.json``) against the
gen-7 render: bench tiers (row 11b), the memories level (row 11m) and the structural bench (row 11).

Each scenario takes the receipt's recipe record through the rendered admission and resolution, stocks the receipt's inputs,
offers candidates the way the provider does (fieldcraft legality, plus the in-reach bench context with the bench's tier and
the scenario's memories level), and commits through the rendered ``execute_region_fieldcraft``. Not replayed: the block to
bench match (``_bench_contexts``) and the candidate producer's world geometry; the bench context is built by hand with the
same fields. The structural rows commit on the request tick in the render; the native tick is in the fixture.
"""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.jax.combat.arsenal.world_actions import crafting as gen5_crafting
from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import bench_tiers, timed_crafting
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7_OVERLAY
from hytalegym.sim.gen7.profile import PROFILE as GEN7
from hytalegym.worldgen import block_semantic_key
from hytalegym.worldgen.native_crafting import NativeCraftingRecipe
from .support import render_for_test

FIXTURE = Path(__file__).with_name("crafting_oracle_072.json")
FIXTURE_DATA = json.loads(FIXTURE.read_text(encoding="utf-8"))
SCENARIOS = FIXTURE_DATA["scenarios"]
RECIPES = {record["recipe_id"]: record for record in FIXTURE_DATA["recipes"]}


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """The gen-7 render, rendered once for this module and mounted under its manifest name."""

    out = tmp_path_factory.mktemp("render") / "gen7"
    out = render_for_test(GEN7_OVERLAY, out, source=HeadSource())
    package = R.mount(out).__name__
    return {
        "crafting": import_module(f"{package}.combat.arsenal.world_actions.crafting"),
        "interactions": import_module(f"{package}.world.interactions.crafting"),
        "contract": import_module(f"{package}.crafting.contract"),
        "packing": import_module(f"{package}.crafting.packing"),
        "inventory": import_module(f"{package}.combat.inventory"),
        "types": import_module(f"{package}.combat.types"),
        "semantic": import_module(f"{package}.combat.contracts.semantic"),
        "executor": import_module(f"{package}.combat.arsenal.world_actions.executor"),
    }


def _admitted_rows(records, module):
    """The rendered admission and resolution of the records, as the compile does (benches of type 0 and 3 are admitted)."""

    benches = {b["id"] for record in records for b in record["bench_requirements"] if b["type"] in (0, 3)}
    admitted = benches | {"Fieldcraft"}
    rows = []
    for record in records:
        row = NativeCraftingRecipe.from_response(record)
        if module._lossless_region_crafting(row, admitted):
            rows.append(module._resolved_recipe(row))
    return rows


def _assets(engine, scenario):
    """The render's assets for one scenario: its recipe, its bench row at the scenario's tier, its memories level."""

    crafting = engine["crafting"]
    contract = engine["contract"]
    resolved = _admitted_rows([RECIPES[scenario["recipe_id"]]], crafting)
    packed = engine["packing"].pack_resolved_recipe_table(
        tuple(resolved),
        source_sha256="0" * 64,
        expected_recipe_count=len(resolved),
        item_id_resolver=engine["semantic"].semantic_id,
        resource_type_id_resolver=engine["semantic"].semantic_id,
    )
    bench = scenario["bench"]
    words = np.asarray(crafting.identity_sha256_words(bench["bench_id"]), dtype=np.uint32)[None, :]
    return crafting.RegionFieldcraftAssets(
        packed=packed,
        output_max_stack=jnp.full((contract.RECIPE_TABLE_CAPACITY, contract.MAX_OUTPUTS), 64, dtype=jnp.int32),
        output_max_durability=jnp.zeros((contract.RECIPE_TABLE_CAPACITY, contract.MAX_OUTPUTS), dtype=jnp.float32),
        inventory_layout=engine["inventory"].default_inventory_layout(),
        source_bridge_sha256="0" * 64,
        bench_semantic_key=jnp.asarray([block_semantic_key(bench["block_id"], 0)], dtype=jnp.uint32),
        bench_id_hash=jnp.asarray(words, dtype=jnp.uint32),
        bench_type=jnp.asarray([bench["type"]], dtype=jnp.int32),
        bench_tier_level=jnp.asarray(bench_tiers.row_tiers(words, bench["tier"], crafting.identity_sha256_words),
                                     dtype=jnp.int32),
        bench_valid=jnp.ones((1,), dtype=jnp.bool_),
        memories_level=scenario["memories_level"],
    )


def _state(engine, scenario, layout):
    """The receipt's stock in the hotbar, in the order of the record's inputs."""

    state = engine["inventory"].empty_inventory_state(1, entity_count=engine["types"].ENTITY_COUNT)
    hotbar = np.flatnonzero(np.asarray(layout.container_id) == engine["inventory"].CONTAINER_HOTBAR)
    item_id = np.asarray(state.item_id).copy()
    quantity = np.asarray(state.quantity).copy()
    for slot, stock in zip(hotbar, scenario["stock"], strict=False):
        item_id[0, 0, slot] = engine["semantic"].semantic_id(stock["item_asset_id"])
        quantity[0, 0, slot] = stock["quantity"]
    return state._replace(item_id=jnp.asarray(item_id), quantity=jnp.asarray(quantity))


def _item_totals(state) -> dict[int, int]:
    totals: dict[int, int] = {}
    for item, count in zip(np.asarray(state.item_id)[0, 0].tolist(), np.asarray(state.quantity)[0, 0].tolist(), strict=True):
        if count:
            totals[item] = totals.get(item, 0) + count
    return totals


def _run(engine, scenario, *, level=None, tier=None):
    """Offer and commit the scenario's recipe as the provider does.

    Returns (offered, accepted, state_before, state_after). A recipe that is not offered is not committed.
    """

    crafting, interactions, contract = engine["crafting"], engine["interactions"], engine["contract"]
    scenario = {**scenario, "bench": {**scenario["bench"], "tier": scenario["bench"]["tier"] if tier is None else tier},
                "memories_level": scenario["memories_level"] if level is None else level}
    assets = _assets(engine, scenario)
    table, layout = assets.packed.table, assets.inventory_layout
    state = _state(engine, scenario, layout)
    inventory = crafting._crafting_inventory(crafting._actor_inventory(state), layout)
    level_now = scenario["memories_level"]
    legal = crafting._legal_recipes(inventory, crafting._fieldcraft_context(1, 1, level_now), table)
    bench = crafting.CraftingContext(
        window_open=jnp.ones((1, 1, 1), dtype=jnp.bool_),
        bench_type=jnp.full((1, 1, 1), scenario["bench"]["type"], dtype=jnp.int32),
        bench_id_hash=jnp.broadcast_to(assets.bench_id_hash[0], (1, 1, 1, 8)),
        bench_tier_level=jnp.full((1, 1, 1), int(assets.bench_tier_level[0]), dtype=jnp.int32),
        memories_level=jnp.full((1, 1, 1), level_now, dtype=jnp.int32),
        known_recipe=jnp.ones((1, 1, 1, contract.RECIPE_TABLE_CAPACITY), dtype=jnp.bool_),
        creative_mode=jnp.zeros((1, 1, 1), dtype=jnp.bool_),
        input_mode=jnp.full((1, 1, 1), contract.INPUT_MODE_NORMAL, dtype=jnp.int32),
    )
    repeated = crafting._repeat_inventory_for_candidates(inventory, 1)
    legal = legal | jnp.any(crafting._legal_recipes(repeated, bench, table) & bench.window_open[..., None], axis=2)
    candidates = interactions.produce_actor_recipe_candidates_from_legal_mask(
        table, legal, catalog_available=True, legal_evidence_available=jnp.ones((1, 1), dtype=jnp.bool_)
    )
    index = list(assets.packed.recipe_ids).index(scenario["recipe_id"])
    slots = np.flatnonzero(np.asarray(candidates.execution.recipe_index[0, 0]) == index)
    if slots.size == 0 or not bool(np.asarray(candidates.candidate_mask[0, 0])[slots[0]]):
        return False, False, state, state
    result = crafting.execute_region_fieldcraft(assets, state, candidates, jnp.asarray([int(slots[0])], dtype=jnp.int32))
    return True, bool(result.accepted[0]), state, result.state


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["name"] for s in SCENARIOS])
def test_render_matches_the_oracle_receipt(engine, scenario) -> None:
    offered, accepted, before, after = _run(engine, scenario)
    expect = scenario["expect"]
    assert accepted is expect["accepted"], scenario["name"]
    if not expect["accepted"]:
        # Every refusal in the oracle is decided before the commit: the recipe is not offered at all.
        assert offered is False, scenario["name"]
        assert np.array_equal(np.asarray(before.item_id), np.asarray(after.item_id)), scenario["name"]
        assert np.array_equal(np.asarray(before.quantity), np.asarray(after.quantity)), scenario["name"]
        return
    for delta in expect["deltas"]:
        item = engine["semantic"].semantic_id(delta["item"])
        assert _item_totals(before).get(item, 0) == delta["before"], (scenario["name"], delta["item"])
        assert _item_totals(after).get(item, 0) == delta["after"], (scenario["name"], delta["item"])


def _scenario(name):
    return next(s for s in SCENARIOS if s["name"] == name)


def test_the_tier_gate_and_the_memories_gate_each_flip_one_probe(engine) -> None:
    lettuce = _scenario("farm_t2_lettuce")
    assert _run(engine, lettuce, tier=1)[:2] == (False, False)
    assert _run(engine, lettuce, tier=2)[:2] == (True, True)
    chest = _scenario("mem_l2_ancient_chest")
    assert _run(engine, chest, level=1)[:2] == (False, False)
    assert _run(engine, chest, level=2)[:2] == (True, True)


def test_gen5_admits_neither_a_tier_nor_a_memories_recipe() -> None:
    """HEAD (gen 5): only tier-0 and level-1 recipes enter the table, so the two gates never see them."""

    lettuce = NativeCraftingRecipe.from_response(RECIPES["Ingredient_Life_Essence_Lettuce_Recipe_Generated_0"])
    chest = NativeCraftingRecipe.from_response(RECIPES["Furniture_Ancient_Chest_Small_Recipe_Generated_0"])
    admitted = {"Farmingbench", "Furniture_Bench", "Fieldcraft"}
    assert not gen5_crafting._lossless_region_crafting(lettuce, admitted)
    assert not gen5_crafting._lossless_region_crafting(chest, admitted)
    assert GEN5.constructs.crafting_bench_tier == "zero_fill" and GEN5.constructs.crafting_memories == "level_one"
    assert GEN7.constructs.crafting_bench_tier == "placed_bench" and GEN7.constructs.crafting_memories == "recorded_count"


def test_the_commit_trusts_the_candidate_and_does_not_recheck_the_bench_tier(engine) -> None:
    """A limit, pinned: the tier gate is the provider's (the candidate's legality under the bench context). The commit
    (execute_region_fieldcraft, with the gen-7 bench_commit rebind) does not read the bench's tier. Here the bench row is
    tier 1 and the requirement is 2, so the provider would not offer the recipe, but a candidate built anyway commits.
    Native refuses at the window (craft_recipe_not_available_at_bench, FIDELITY (s))."""

    crafting, interactions, contract = engine["crafting"], engine["interactions"], engine["contract"]
    scenario = _scenario("farm_t2_lettuce")
    assets = _assets(engine, {**scenario, "bench": {**scenario["bench"], "tier": 1}})
    table = assets.packed.table
    state = _state(engine, scenario, assets.inventory_layout)
    index = list(assets.packed.recipe_ids).index(scenario["recipe_id"])
    onehot = jnp.zeros((1, 1, contract.RECIPE_TABLE_CAPACITY), dtype=jnp.bool_).at[0, 0, index].set(True)
    candidates = interactions.produce_actor_recipe_candidates_from_legal_mask(
        table, onehot, catalog_available=True, legal_evidence_available=jnp.ones((1, 1), dtype=jnp.bool_)
    )
    assert bool(crafting.execute_region_fieldcraft(assets, state, candidates, jnp.asarray([0], dtype=jnp.int32)).accepted[0])


# --- Timed bench crafts (row 11j, construct timed_crafting; FIDELITY-0.7.2.md section (t), stage 3 receipts) ---------------
# The construct's ``advance`` ticks a job over world ticks; the request, the second request and the reach-out are driven as the
# executor drives them (``execute_region_fieldcraft`` for the candidate and the commit, ``start_at`` for the job). Ticks are
# world ticks: the request is tick R (its inputs stay), R+1 is the first progress tick, completion is R+90 for 3.0 s.

TIMED_RECIPE = "Armor_Copper_Chest_Recipe_Generated_0"
TIMED_BENCH_CELL = (-2, 65, 0)
TIMED_SCENARIOS = {scenario["name"]: scenario for scenario in FIXTURE_DATA["timed_scenarios"]}


class _Combat(NamedTuple):
    position: object
    health: object
    craft_job_recipe: object
    craft_job_elapsed_seconds: object
    craft_job_inputs_pending: object
    craft_queued_recipe: object
    craft_bench_position: object
    craft_refund_quantity: object
    craft_refund_item_id: object


class _World(NamedTuple):
    combat: _Combat
    inventory: object


class _Params(NamedTuple):
    ticks_per_second: float


@pytest.fixture(scope="module")
def timed(engine):
    """The armor recipe's assets on the rendered engine, and the construct's advance jitted once for the module."""

    crafting, contract = engine["crafting"], engine["contract"]
    resolved = _admitted_rows([RECIPES[TIMED_RECIPE]], crafting)
    assert len(resolved) == 1 and resolved[0].time_seconds == 3.0
    packed = engine["packing"].pack_resolved_recipe_table(
        tuple(resolved),
        source_sha256="0" * 64,
        expected_recipe_count=1,
        item_id_resolver=engine["semantic"].semantic_id,
        resource_type_id_resolver=engine["semantic"].semantic_id,
    )
    words = np.asarray(crafting.identity_sha256_words("Armor_Bench"), dtype=np.uint32)[None, :]
    stack = np.full((contract.RECIPE_TABLE_CAPACITY, contract.MAX_OUTPUTS), 64, dtype=np.int32)
    stack[0, 0] = 1  # the chest does not stack: the second chest of the stocked receipt took a new slot (section (t))
    layout = engine["inventory"].default_inventory_layout()
    assets = crafting.RegionFieldcraftAssets(
        packed=packed,
        output_max_stack=jnp.asarray(stack),
        output_max_durability=jnp.zeros((contract.RECIPE_TABLE_CAPACITY, contract.MAX_OUTPUTS), dtype=jnp.float32),
        inventory_layout=layout,
        source_bridge_sha256="0" * 64,
        bench_semantic_key=jnp.asarray([block_semantic_key("Bench_Armour", 0)], dtype=jnp.uint32),
        bench_id_hash=jnp.asarray(words, dtype=jnp.uint32),
        bench_type=jnp.asarray([0], dtype=jnp.int32),
        bench_tier_level=jnp.asarray(bench_tiers.row_tiers(words, 1, crafting.identity_sha256_words), dtype=jnp.int32),
        bench_valid=jnp.ones((1,), dtype=jnp.bool_),
        memories_level=1,
    )
    params = _Params(ticks_per_second=30.0)
    advance = jax.jit(lambda world: timed_crafting.advance(crafting.execute_region_fieldcraft, world, assets, layout, params, 0))
    index = list(packed.recipe_ids).index(TIMED_RECIPE)
    return {"assets": assets, "layout": layout, "advance": advance, "crafting": crafting, "engine": engine, "index": index}


def _world(timed, stock, *, position=None):
    """A world with the stock in the hotbar (the first slots), the agent at the bench's centre unless ``position`` is given."""

    engine = timed["engine"]
    inventory = _state(engine, {"stock": stock}, timed["layout"])
    entities = engine["types"].ENTITY_COUNT
    where = np.zeros((1, entities, 3), dtype=np.float32)
    where[0, 0] = (np.asarray(TIMED_BENCH_CELL, dtype=np.float32) + 0.5) if position is None else position
    combat = _Combat(
        position=jnp.asarray(where),
        health=jnp.full((1, entities), 105.0, dtype=jnp.float32),
        craft_job_recipe=jnp.asarray([-1], dtype=jnp.int32),
        craft_job_elapsed_seconds=jnp.zeros((1,), dtype=jnp.float32),
        craft_job_inputs_pending=jnp.zeros((1,), dtype=jnp.bool_),
        craft_queued_recipe=jnp.asarray([-1], dtype=jnp.int32),
        craft_bench_position=jnp.zeros((1, 3), dtype=jnp.int32),
        craft_refund_quantity=jnp.zeros((1, 76), dtype=jnp.int32),
        craft_refund_item_id=jnp.full((1, 76), -1, dtype=jnp.int32),
    )
    return _World(combat=combat, inventory=inventory)


def _request(timed, world):
    """The request as the provider offers it (the bench context at the window) and the executor starts it.

    Returns ``(accepted, world)``; a recipe the provider does not offer is ``(False, world)``."""

    crafting, assets, layout = timed["crafting"], timed["assets"], timed["layout"]
    engine = timed["engine"]
    table = assets.packed.table
    actor = crafting._crafting_inventory(crafting._actor_inventory(world.inventory), layout)
    legal = crafting._legal_recipes(actor, crafting._fieldcraft_context(1, 1), table)
    bench = crafting.CraftingContext(
        window_open=jnp.ones((1, 1, 1), dtype=jnp.bool_),
        bench_type=jnp.full((1, 1, 1), 0, dtype=jnp.int32),
        bench_id_hash=jnp.broadcast_to(assets.bench_id_hash[0], (1, 1, 1, 8)),
        bench_tier_level=jnp.full((1, 1, 1), 1, dtype=jnp.int32),
        memories_level=jnp.ones((1, 1, 1), dtype=jnp.int32),
        known_recipe=jnp.ones((1, 1, 1, engine["contract"].RECIPE_TABLE_CAPACITY), dtype=jnp.bool_),
        creative_mode=jnp.zeros((1, 1, 1), dtype=jnp.bool_),
        input_mode=jnp.full((1, 1, 1), engine["contract"].INPUT_MODE_NORMAL, dtype=jnp.int32),
    )
    repeated = crafting._repeat_inventory_for_candidates(actor, 1)
    legal = legal | jnp.any(crafting._legal_recipes(repeated, bench, table) & bench.window_open[..., None], axis=2)
    candidates = engine["interactions"].produce_actor_recipe_candidates_from_legal_mask(
        table, legal, catalog_available=True, legal_evidence_available=jnp.ones((1, 1), dtype=jnp.bool_)
    )
    slots = np.flatnonzero(np.asarray(candidates.execution.recipe_index[0, 0]) == timed["index"])
    if slots.size == 0 or not bool(np.asarray(candidates.candidate_mask[0, 0])[slots[0]]):
        return False, world
    craft = crafting.execute_region_fieldcraft(assets, world.inventory, candidates, jnp.asarray([int(slots[0])], dtype=jnp.int32))
    accepted, started = timed_crafting.start_at(
        crafting.execute_region_fieldcraft,
        world,
        craft,
        jnp.asarray([TIMED_BENCH_CELL], dtype=jnp.int32),
        jnp.asarray([True]),
        assets,
    )
    return bool(accepted[0]), started


def _hotbar_totals(timed, world):
    """(hotbar slot, item semantic id, quantity) of every occupied hotbar slot; the hotbar slot is its index in the hotbar."""

    engine = timed["engine"]
    layout = timed["layout"]
    hotbar = np.flatnonzero(np.asarray(layout.container_id) == engine["inventory"].CONTAINER_HOTBAR)
    item = np.asarray(world.inventory.item_id)[0, 0]
    quantity = np.asarray(world.inventory.quantity)[0, 0]
    return [(position, int(item[s]), int(quantity[s])) for position, s in enumerate(hotbar) if quantity[s] > 0]


def _drive(timed, scenario):
    """Request at R, then tick the construct; the events of the scenario happen before the tick they name.

    Returns the per-tick trace: tick -> (job recipe, queued recipe, elapsed seconds, [(slot, before, after)] of changed hotbar
    slots), and the final world."""

    engine = timed["engine"]
    world = _world(timed, scenario["stock"])
    accepted, world = _request(timed, world)
    assert accepted is scenario["expect"]["accepted"], scenario["name"]
    events = {event["tick"]: event["kind"] for event in scenario["events"]}
    trace = {}
    before_slots = dict((slot, (item, qty)) for slot, item, qty in _hotbar_totals(timed, world))
    for tick in range(scenario["request_tick"] + 1, scenario["last_tick"] + 1):
        kind = events.get(tick)
        if kind == "reach_out":
            position = np.asarray(world.combat.position).copy()
            position[0, 0, 0] += 9.0  # the native teleport of 9 blocks (moved_blocks 9.0)
            world = world._replace(combat=world.combat._replace(position=jnp.asarray(position)))
        elif kind == "damage":
            health = np.asarray(world.combat.health).copy()
            health[0, 0] -= 1.0  # one physical hit (health 105 -> 104); the construct does not read health
            world = world._replace(combat=world.combat._replace(health=jnp.asarray(health)))
        elif kind == "second_craft":
            accepted, world = _request(timed, world)
            trace.setdefault(tick, {})["second_accepted"] = accepted
        world = timed["advance"](world)
        after_slots = dict((slot, (item, qty)) for slot, item, qty in _hotbar_totals(timed, world))
        changed = [(slot, before_slots.get(slot, (0, 0)), after_slots.get(slot, (0, 0)))
                   for slot in sorted(set(before_slots) | set(after_slots)) if before_slots.get(slot) != after_slots.get(slot)]
        entry = trace.setdefault(tick, {})
        entry.update(job=int(world.combat.craft_job_recipe[0]), queued=int(world.combat.craft_queued_recipe[0]),
                     elapsed=float(world.combat.craft_job_elapsed_seconds[0]), changes=changed)
        before_slots = after_slots
    return trace, world


def _output_ticks(trace, item):
    """The ticks on which the item appears in a hotbar slot (a slot holds (item, quantity))."""

    return [tick for tick, entry in trace.items() if any(
        after[0] == item and before[0] != item for _slot, before, after in entry.get("changes", ())
    )]


def test_plain_timed_craft_takes_inputs_on_the_first_progress_tick_and_gives_the_output_on_the_ninetieth(timed) -> None:
    scenario = TIMED_SCENARIOS["timed_plain"]
    trace, world = _drive(timed, scenario)
    chest = timed["engine"]["semantic"].semantic_id("Armor_Copper_Chest")
    expect = scenario["expect"]
    # the inputs leave on R+1 (native first_removed 33 for request 32): slot 0 bars 11 -> 0, slot 1 fibre 4 -> 0
    removed = [slot for slot, before, after in trace[expect["inputs_out_tick"]]["changes"] if after[1] == 0 and before[1] > 0]
    assert removed == [0, 1]
    # the output on R+90 (122), in the first free hotbar slot (the freed slot 0); the native run landed on 120
    assert _output_ticks(trace, chest) == [expect["output_tick"]]
    assert trace[expect["output_tick"]]["changes"][0][0] == expect["output_slot"]
    assert trace[expect["output_tick"]]["job"] == -1 and world.combat.craft_job_recipe[0] == -1
    assert all(entry["job"] == 0 for tick, entry in trace.items() if expect["inputs_out_tick"] <= tick < expect["output_tick"])


def test_a_reach_out_cancels_on_its_tick_and_refunds_the_inputs_to_their_slots(timed) -> None:
    scenario = TIMED_SCENARIOS["timed_reach_out"]
    trace, world = _drive(timed, scenario)
    expect = scenario["expect"]
    assert expect["cancel_tick"] == scenario["events"][0]["tick"]
    cancel = trace[expect["cancel_tick"]]
    assert cancel["job"] == -1 and cancel["queued"] == -1
    refunded = [(slot, before[1], after[1]) for slot, before, after in cancel["changes"] if after[1] > before[1]]
    # the consumed bars (11) and fibre (4) return to slots 0 and 1, the slots they left
    assert [slot for slot, _b, _a in refunded] == expect["refund_slots"]
    hot = {slot: (item, qty) for slot, item, qty in _hotbar_totals(timed, world)}
    engine = timed["engine"]
    assert hot[0] == (engine["semantic"].semantic_id("Ingredient_Bar_Copper"), 11)
    assert hot[1] == (engine["semantic"].semantic_id("Ingredient_Fibre"), 4)
    assert all(entry["job"] == 0 for tick, entry in trace.items() if scenario["request_tick"] < tick < expect["cancel_tick"])
    assert _output_ticks(trace, engine["semantic"].semantic_id("Armor_Copper_Chest")) == []


def test_a_second_request_queues_and_its_inputs_leave_on_the_first_completion(timed) -> None:
    scenario = TIMED_SCENARIOS["timed_second_stocked"]
    trace, world = _drive(timed, scenario)
    expect = scenario["expect"]
    engine = timed["engine"]
    chest = engine["semantic"].semantic_id("Armor_Copper_Chest")
    assert trace[expect["second_queued_tick"]]["second_accepted"] is True
    assert trace[expect["second_queued_tick"]]["queued"] == 0  # the queued recipe is the armor (index 0), one deep
    # the first job's output and the queued job's inputs on the same tick (output first: it takes the first free slot)
    assert _output_ticks(trace, chest) == [expect["output_tick"], expect["second_output_tick"]]
    at_first = trace[expect["output_tick"]]["changes"]
    assert [slot for slot, _b, after in at_first if after[0] == chest] == [2]
    assert [slot for slot, before, after in at_first if before[1] > 0 and after[1] == 0] == [0, 1]
    assert trace[expect["output_tick"]]["job"] == 0 and trace[expect["output_tick"]]["queued"] == -1
    assert trace[expect["second_output_tick"]]["job"] == -1
    assert all(trace[t]["queued"] == 0 for t in range(scenario["events"][0]["tick"], expect["output_tick"]))


def test_a_second_request_without_its_inputs_is_refused_and_the_outcome_matches_native(timed) -> None:
    scenario = TIMED_SCENARIOS["timed_second_no_stock"]
    trace, world = _drive(timed, scenario)
    expect = scenario["expect"]
    chest = timed["engine"]["semantic"].semantic_id("Armor_Copper_Chest")
    # the sim does not offer the second craft (no bars left); native queues it and drops it on its start: one chest either way
    assert trace[43]["queued"] == -1 and trace[43].get("second_accepted") is False
    assert _output_ticks(trace, chest) == [expect["output_tick"]]
    assert all(entry["queued"] == -1 for entry in trace.values())


def test_damage_does_not_touch_the_job(timed) -> None:
    scenario = TIMED_SCENARIOS["timed_damage"]
    trace, world = _drive(timed, scenario)
    expect = scenario["expect"]
    chest = timed["engine"]["semantic"].semantic_id("Armor_Copper_Chest")
    assert all(entry["job"] == 0 for tick, entry in trace.items() if scenario["request_tick"] < tick < expect["output_tick"])
    assert _output_ticks(trace, chest) == [expect["output_tick"]]
    # the same tick with and without the hit leaves the same job leaves (the construct reads positions and the job only)
    plain = TIMED_SCENARIOS["timed_plain"]
    base_trace, _ = _drive(timed, {**plain, "request_tick": scenario["request_tick"], "last_tick": 60})
    assert base_trace[44]["elapsed"] == trace[44]["elapsed"] and base_trace[44]["job"] == trace[44]["job"]


def test_a_control_without_a_window_finishes_on_the_same_tick_as_the_plain_craft(timed) -> None:
    scenario = TIMED_SCENARIOS["timed_no_window_control"]
    trace, _ = _drive(timed, scenario)
    assert _output_ticks(trace, timed["engine"]["semantic"].semantic_id("Armor_Copper_Chest")) == [scenario["expect"]["output_tick"]]


def test_fieldcraft_does_not_offer_the_timed_recipe(timed) -> None:
    """Fieldcraft has no timed path: the bench-only recipe is not offered there, so it cannot start a job."""

    scenario = TIMED_SCENARIOS["timed_fieldcraft_refused"]
    world = _world(timed, scenario["stock"])
    crafting, assets, layout = timed["crafting"], timed["assets"], timed["layout"]
    actor = crafting._crafting_inventory(crafting._actor_inventory(world.inventory), layout)
    legal = crafting._legal_recipes(actor, crafting._fieldcraft_context(1, 1), assets.packed.table)
    assert not bool(np.asarray(legal)[0, 0, timed["index"]])
    assert scenario["expect"]["accepted"] is False


def test_a_timed_request_is_accepted_but_not_committed_on_its_tick(timed) -> None:
    world = _world(timed, TIMED_SCENARIOS["timed_plain"]["stock"])
    accepted, started = _request(timed, world)
    assert accepted is True
    assert np.array_equal(np.asarray(started.inventory.quantity), np.asarray(world.inventory.quantity))
    assert int(started.combat.craft_job_recipe[0]) == timed["index"]
    assert bool(started.combat.craft_job_inputs_pending[0]) is True


def test_a_queue_is_one_deep_and_refuses_a_third_request(timed) -> None:
    world = _world(timed, [{"item_asset_id": "Ingredient_Bar_Copper", "quantity": 33},
                           {"item_asset_id": "Ingredient_Fibre", "quantity": 12}])
    accepted, world = _request(timed, world)
    assert accepted is True
    accepted, world = _request(timed, world)
    assert accepted is True and int(world.combat.craft_queued_recipe[0]) == timed["index"]
    accepted, after = _request(timed, world)
    assert accepted is False
    assert np.array_equal(np.asarray(after.combat.craft_queued_recipe), np.asarray(world.combat.craft_queued_recipe))


def test_the_gen5_profile_and_the_gen5_overlay_keep_every_original(timed) -> None:
    """Gen 5 (``instant``) returns the original functions; no timed entry is in the gen-5 overlay."""

    from hytalegym.sim.gen5.overlay import OVERLAY as GEN5_OVERLAY

    sentinel = object()
    assert GEN5.constructs.crafting_time == "instant"
    assert timed_crafting.admission(sentinel, GEN5) is sentinel
    assert timed_crafting.execution(sentinel, GEN5) is sentinel
    assert timed_crafting.executor(sentinel, GEN5) is sentinel
    assert timed_crafting._enabled(GEN7) is True
    assert not any("timed" in getattr(hook, "id", "") for hook in GEN5_OVERLAY.hooks)
    assert not any("timed" in getattr(patch, "id", "") for patch in GEN5_OVERLAY.patches)


def test_the_executor_factory_is_wrapped_and_the_craft_verb_starts_the_job(engine) -> None:
    """Rendered wiring (not run here, see FIDELITY (section timed crafts)): the factory is hooked and the craft verb calls start."""

    import inspect

    executor = engine["executor"]
    assert executor.make_region_block_action_surface_executor.__wrapped__ is not None
    assert executor.timed_crafting_start.__name__ == "start"
    assert executor.timed_crafting_start.__module__ == "hytalegym.sim.gen7.mechanics.timed_crafting"
    assert inspect.getsource(executor).count("timed_crafting_start(") == 1


def test_the_bench_channel_takes_the_first_in_reach_bench_on_which_the_recipe_is_legal(timed, monkeypatch) -> None:
    """``bench_for_recipe``: three block candidates; candidate 0 is another bench (Farmingbench), candidate 1 is the armor's
    Armor_Bench in reach, candidate 2 is Armor_Bench out of the window. The job's bench is candidate 1's visible cell."""

    crafting, engine = timed["crafting"], timed["engine"]
    count = 3
    fake_current = {}
    monkeypatch.setitem(crafting.__dict__, "requery_region_block_action_candidates", lambda *_args: fake_current)
    farming = np.asarray(crafting.identity_sha256_words("Farmingbench"), dtype=np.uint32)
    armor = np.asarray(crafting.identity_sha256_words("Armor_Bench"), dtype=np.uint32)
    bench_hash = np.stack([farming, armor, armor])[None, None]
    window = np.asarray([[[True, True, False]]])

    def fake_contexts(assets, candidates, current):
        return crafting.CraftingContext(
            window_open=jnp.asarray(window),
            bench_type=jnp.zeros((1, 1, count), dtype=jnp.int32),
            bench_id_hash=jnp.asarray(bench_hash),
            bench_tier_level=jnp.ones((1, 1, count), dtype=jnp.int32),
            memories_level=jnp.ones((1, 1, count), dtype=jnp.int32),
            known_recipe=jnp.ones((1, 1, count, engine["contract"].RECIPE_TABLE_CAPACITY), dtype=jnp.bool_),
            creative_mode=jnp.zeros((1, 1, count), dtype=jnp.bool_),
            input_mode=jnp.full((1, 1, count), engine["contract"].INPUT_MODE_NORMAL, dtype=jnp.int32),
        )

    monkeypatch.setitem(crafting.__dict__, "_bench_contexts", fake_contexts)
    Candidates = NamedTuple("Candidates", [("available", object), ("candidate_mask", object), ("visible_position", object)])
    candidates = Candidates(
        available=jnp.ones((1, 1), dtype=jnp.bool_),
        candidate_mask=jnp.ones((1, 1, count), dtype=jnp.bool_),
        visible_position=jnp.asarray([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=jnp.int32),
    )
    World = NamedTuple("World", [("geometry", object), ("semantic_atlas", object)])
    world = World(geometry=None, semantic_atlas=None)
    world_state = _world(timed, [{"item_asset_id": "Ingredient_Bar_Copper", "quantity": 11},
                                 {"item_asset_id": "Ingredient_Fibre", "quantity": 4}])
    position, found = timed_crafting.bench_for_recipe(
        crafting.execute_region_fieldcraft, timed["assets"], timed["layout"], candidates, world, world_state.inventory,
        jnp.asarray([timed["index"]], dtype=jnp.int32),
    )
    assert bool(found[0]) is True
    assert np.asarray(position)[0].tolist() == [4, 5, 6]
    # with the armor bench out of the window the recipe has no bench: found is False and the position is zero
    window[:] = [[[True, False, False]]]
    position, found = timed_crafting.bench_for_recipe(
        crafting.execute_region_fieldcraft, timed["assets"], timed["layout"], candidates, world, world_state.inventory,
        jnp.asarray([timed["index"]], dtype=jnp.int32),
    )
    assert bool(found[0]) is False and np.asarray(position)[0].tolist() == [0, 0, 0]


def test_an_idle_step_runs_no_bench_scan_and_changes_no_leaf(timed, monkeypatch) -> None:
    """Cost (the advance and the bench scan are lax.cond on any lane): with no job and no timed request an executor step changes
    no job leaf or inventory slot and runs no bench scan (counted at run time by jax.debug.callback inside the scan's context
    builder). A timed request on the idle lane does run the scan (the positive control)."""

    crafting = timed["crafting"]
    scans: list[int] = []

    def counted_contexts(assets, candidates, current):
        jax.debug.callback(lambda: scans.append(1))
        shape = candidates.candidate_mask.shape
        return crafting.CraftingContext(
            window_open=jnp.zeros(shape, dtype=jnp.bool_),
            bench_type=jnp.zeros(shape, dtype=jnp.int32),
            bench_id_hash=jnp.zeros(shape + (8,), dtype=jnp.uint32),
            bench_tier_level=jnp.zeros(shape, dtype=jnp.int32),
            memories_level=jnp.ones(shape, dtype=jnp.int32),
            known_recipe=jnp.ones(shape + (timed["engine"]["contract"].RECIPE_TABLE_CAPACITY,), dtype=jnp.bool_),
            creative_mode=jnp.zeros(shape, dtype=jnp.bool_),
            input_mode=jnp.zeros(shape, dtype=jnp.int32),
        )

    monkeypatch.setitem(crafting.__dict__, "_bench_contexts", counted_contexts)
    monkeypatch.setitem(crafting.__dict__, "requery_region_block_action_candidates", lambda *_args: {})
    Candidates = NamedTuple("Candidates", [("available", object), ("candidate_mask", object), ("visible_position", object)])
    Geometry = NamedTuple("Geometry", [("geometry", object), ("semantic_atlas", object)])
    Craft = NamedTuple("Craft", [("accepted", object), ("recipe_index", object)])
    candidates = Candidates(
        available=jnp.ones((1, 1), dtype=jnp.bool_),
        candidate_mask=jnp.ones((1, 1, 3), dtype=jnp.bool_),
        visible_position=jnp.zeros((1, 1, 3, 3), dtype=jnp.int32),
    )
    world_geometry = Geometry(geometry=None, semantic_atlas=None)
    world = _world(timed, TIMED_SCENARIOS["timed_plain"]["stock"])
    before = [np.asarray(leaf) for leaf in jax.tree.leaves(world)]

    idle = Craft(accepted=jnp.zeros((1,), dtype=jnp.bool_), recipe_index=jnp.full((1,), -1, dtype=jnp.int32))
    accepted, started = timed_crafting.start(
        crafting.execute_region_fieldcraft, world, idle, candidates, world_geometry, timed["assets"], timed["layout"]
    )
    stepped = timed["advance"](started)
    jax.effects_barrier()
    assert not bool(np.any(np.asarray(accepted)))
    after = [np.asarray(leaf) for leaf in jax.tree.leaves(stepped)]
    assert len(before) == len(after)
    assert all(np.array_equal(left, right) for left, right in zip(before, after, strict=True))
    assert scans == []

    timed_request = Craft(
        accepted=jnp.ones((1,), dtype=jnp.bool_),
        recipe_index=jnp.full((1,), timed["index"], dtype=jnp.int32),
    )
    timed_crafting.start(crafting.execute_region_fieldcraft, world, timed_request, candidates, world_geometry,
                         timed["assets"], timed["layout"])
    jax.effects_barrier()
    assert scans
