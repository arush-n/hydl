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

import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.jax.combat.arsenal.world_actions import crafting as gen5_crafting
from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import bench_tiers
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7_OVERLAY
from hytalegym.sim.gen7.profile import PROFILE as GEN7
from hytalegym.worldgen import block_semantic_key
from hytalegym.worldgen.native_crafting import NativeCraftingRecipe

FIXTURE = Path(__file__).with_name("crafting_oracle_072.json")
FIXTURE_DATA = json.loads(FIXTURE.read_text(encoding="utf-8"))
SCENARIOS = FIXTURE_DATA["scenarios"]
RECIPES = {record["recipe_id"]: record for record in FIXTURE_DATA["recipes"]}


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """The gen-7 render, rendered once for this module and mounted under its manifest name."""

    out = tmp_path_factory.mktemp("render") / "gen7"
    R.render(GEN7_OVERLAY, out, source=HeadSource())
    package = R.mount(out).__name__
    return {
        "crafting": import_module(f"{package}.combat.arsenal.world_actions.crafting"),
        "interactions": import_module(f"{package}.world.interactions.crafting"),
        "contract": import_module(f"{package}.crafting.contract"),
        "packing": import_module(f"{package}.crafting.packing"),
        "inventory": import_module(f"{package}.combat.inventory"),
        "types": import_module(f"{package}.combat.types"),
        "semantic": import_module(f"{package}.combat.contracts.semantic"),
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
