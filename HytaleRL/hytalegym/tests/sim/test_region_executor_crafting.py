"""The gen-7 crafting constructs on the executor path, inside a live Region step (FIDELITY-0.7.2.md sections (ac) and (ae)).

timed_crafting, processing_benches and diagram_crafting were exercised by direct calls (the replays drive ``start_at`` and
``advance``) and by static wiring checks. Here the rendered engine runs the production arsenal environment: the environment is
built with the rendered ``make_region_block_action_surface_executor``, a recipe candidate provider and crafting assets, over a
small synthetic Region (one bench block, the actor on a three-block floor, the stock in the hotbar). Each scenario is a run of
``environment.step_factors`` with neutral policy actions; the craft is requested on the step the scenario names. A tick is an
env step, so a request on step R is tick R.

The request goes through the executor's recipe candidate head. The policy decode never sets that head (the test
``test_the_policy_decode_never_requests_a_recipe`` pins it), so the request is written into the decoded surface on its step, the
place the decode would write it. Each table holds one recipe, so the head's slot is 0, and ``execute_region_fieldcraft``
rechecks that slot's legality before any commit.

The Armory craft is the axe. Its inputs are resource types (Wood_Trunk, Rock), matched through the item table of the
crafting-materials construct, which the production compile fills from the archive; the scenario fills the same table for its stock
and restores the global after the run.

Expectations are the simulator's clock: a timed craft's inputs leave on R+1 and its output lands on R+90 (the second request
queues on R+11 and its inputs leave on the first completion); a processing craft's first progress is R+2 and its output lands on
R+61; an Armory craft is queued on R and commits on R+2.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.geometry.contract import FLAG_OPAQUE, FLAG_SOLID, FLUID_MOVEMENT_FEATURES, MOVEMENT_FEATURES
from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7_OVERLAY
from hytalegym.sim.gen7.mechanics import crafting_materials
from hytalegym.worldgen import block_semantic_key
from hytalegym.worldgen.block_actions import block_asset_key, resolve_local_block_action_semantics
from hytalegym.worldgen.block_affordances import GATHER_TYPES, block_affordance_tag_mask
from hytalegym.worldgen.native_crafting import NativeCraftingRecipe
from hytalegym.worldgen.region import (
    CAPTURE_CHUNK_COUNT,
    CHUNK_SIZE,
    HEIGHT_SECTIONS,
    MAX_SHAPE_BOXES,
    OFFLINE_COMPLETE,
    REGION_SNAPSHOT_SCHEMA,
    REGION_SNAPSHOT_VERSION,
    SECTION_VOLUME,
    CellPalette,
    NativeRegionBlockSemanticSnapshot,
    NativeRegionSnapshot,
    RegionBlockSemanticEntry,
    ShapePalette,
    capture_chunk_slot,
    certified_chunk_api_manifest,
)

from .support import ARCHIVES, mount_fresh, render_for_test, unmount

CRAFT = json.loads(Path(__file__).with_name("crafting_oracle_072.json").read_text(encoding="utf-8"))
STAGE5 = json.loads(Path(__file__).with_name("stage5_oracle_072.json").read_text(encoding="utf-8"))

WORLD_ID = 0
BRIDGE = "A" * 64  # the synthetic captures carry no native bytes; the value only has to be a well-formed digest
FLOOR_Y = 64
BENCH_CELL = (50, 65, 50)
ACTOR = (47.5, 65.0, 50.5)
TARGET = (57.5, 65.0, 57.5)
ACTOR_SEED = 570057
MAXIMUM_REACH = 4.0  # block candidates within this distance of the actor's eye (the example's block reach)
LOS_DISTANCE = 72.0  # the example's perception line-of-sight budget (REGION_PERCEPTION_LOS_DISTANCE_BLOCKS)
DT = 1.0 / 30.0  # one env step is one world tick at the 30 Hz clock


class Scenario(NamedTuple):
    recipe_id: str
    record: dict
    bench: dict
    stock: tuple
    output: str
    requests: tuple[int, ...]
    steps: int


def _record(recipe_id: str) -> dict:
    for source in (CRAFT["recipes"], STAGE5["recipes"]):
        for record in source:
            if record["recipe_id"] == recipe_id:
                return record
    raise KeyError(recipe_id)


TIMED = Scenario(
    recipe_id="Armor_Copper_Chest_Recipe_Generated_0",
    record=_record("Armor_Copper_Chest_Recipe_Generated_0"),
    bench={"bench_id": "Armor_Bench", "block_id": "Bench_Armour", "type": 0, "tier": 1},
    stock=(("Ingredient_Bar_Copper", 22), ("Ingredient_Fibre", 8)),
    output="Armor_Copper_Chest",
    requests=(0, 11),
    steps=91,
)
PROCESSING = Scenario(
    recipe_id="Food_Fish_Grilled_Recipe_Generated_0",
    record=_record("Food_Fish_Grilled_Recipe_Generated_0"),
    bench={"bench_id": "Campfire", "block_id": "Bench_Campfire", "type": 1, "tier": 1},
    stock=(("Food_Fish_Raw", 1), ("Ingredient_Stick", 2)),
    output="Food_Fish_Grilled",
    requests=(0,),
    steps=62,
)
AXE_RECORD = _record("Weapon_Axe_Stone_Trork_Recipe_Generated_0")
AXE_STOCK = (("Plant_Fern_Jungle_Trunk", 2), ("Deco_Coral_Shell_Purple", 6), ("Ingredient_Leather_Light", 1))
DIAGRAM = Scenario(
    recipe_id="Weapon_Axe_Stone_Trork_Recipe_Generated_0",
    record=AXE_RECORD,
    bench={"bench_id": "Armory", "block_id": "Bench_Armory", "type": 2, "tier": 1},
    stock=AXE_STOCK,
    output="Weapon_Axe_Stone_Trork",
    requests=(0,),
    steps=3,
)


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    """The gen-7 render, rendered once for this module and mounted under its manifest name."""

    out = tmp_path_factory.mktemp("render") / "gen7"
    out = render_for_test(GEN7_OVERLAY, out, source=HeadSource())
    name = mount_fresh(out).__name__
    yield name
    unmount(name)


def _modules(package: str) -> dict:
    module = lambda path: import_module(f"{package}.{path}")  # noqa: E731
    return {
        "environment": module("combat.arsenal.environment"),
        "executor": module("combat.arsenal.world_actions.executor"),
        "crafting": module("combat.arsenal.world_actions.crafting"),
        "placement": module("combat.arsenal.world_actions.placement"),
        "world": module("world"),
        "runtime": module("combat.arsenal.runtime"),
        "profiles": module("combat.arsenal.profiles.hytale_0_5_7"),
        "inventory": module("combat.inventory"),
        "combat": module("combat"),
        "types": module("combat.types"),
        "semantic": module("combat.contracts.semantic"),
        "packing": module("crafting.packing"),
        "contract": module("crafting.contract"),
        "policy_layout": module("combat.observation.v3.policy.layout"),
        "policy_surface": module("combat.observation.v3.policy.surface"),
        "world_tokens": module("combat.observation.v3.tokens.world"),
        "light_tokens": module("combat.observation.v3.tokens.light"),
        "features": module("combat.observation.v1.runtime.factory"),
        "block_interactions": module("combat.block_interactions.execution.runtime"),
        "catalog": module("world.block_actions._catalog"),
    }


# --- The synthetic Region: captures built the way test_region_block_semantics builds them ----------------------------------


def _cell_location(position):
    x, y, z = position
    slot = capture_chunk_slot(x // CHUNK_SIZE, z // CHUNK_SIZE, -1, -1)
    section = y // CHUNK_SIZE
    index = (y & (CHUNK_SIZE - 1)) * CHUNK_SIZE * CHUNK_SIZE + (z & (CHUNK_SIZE - 1)) * CHUNK_SIZE + (x & (CHUNK_SIZE - 1))
    return slot, section, index


def _floor() -> list[tuple[int, int, int]]:
    # Three by three blocks under the actor. A larger floor makes every floor block an actionable candidate, and the fixed
    # candidate capacity (16) then closes the row (OUTPUT_CAPACITY).
    return [(x, FLOOR_Y, z) for x in range(46, 49) for z in range(49, 52)]


def _physical_snapshot() -> NativeRegionSnapshot:
    known = np.ones((CAPTURE_CHUNK_COUNT, HEIGHT_SECTIONS), dtype=np.bool_)
    code = np.zeros((CAPTURE_CHUNK_COUNT, HEIGHT_SECTIONS, SECTION_VOLUME), dtype=np.uint16)
    for position in [*_floor(), BENCH_CELL]:
        slot, section, index = _cell_location(position)
        code[slot, section, index] = 1
    cell = CellPalette(
        flags=np.asarray([0, FLAG_SOLID], dtype=np.uint16),
        shape_index=np.asarray([0, 1], dtype=np.uint16),
        fluid_level=np.zeros(2, dtype=np.uint8),
        fluid_fill_height=np.zeros(2, dtype=np.float32),
        support=np.zeros(2, dtype=np.int32),
        block_damage=np.zeros(2, dtype=np.int32),
        fluid_damage=np.zeros(2, dtype=np.int32),
        movement=np.zeros((2, MOVEMENT_FEATURES), dtype=np.float32),
        fluid_movement=np.zeros((2, FLUID_MOVEMENT_FEATURES), dtype=np.float32),
    )
    boxes = np.zeros((2, MAX_SHAPE_BOXES, 6), dtype=np.float32)
    box_mask = np.zeros((2, MAX_SHAPE_BOXES), dtype=np.bool_)
    boxes[1, 0] = [0, 0, 0, 1, 1, 1]
    box_mask[1, 0] = True
    return NativeRegionSnapshot(
        metadata={
            "schema": REGION_SNAPSHOT_SCHEMA,
            "version": REGION_SNAPSHOT_VERSION,
            "server_version": "0.5.7",
            "chunk_api": certified_chunk_api_manifest(),
            "capture_mode": OFFLINE_COMPLETE,
            "exact_collision_shapes": True,
            "dynamic_state": "static",
            "section_protocol_version": 3,
            "exact_filler_root_offsets": True,
            "filler_root_offset_encoding": "uint16_le_signed_5bit_xyz_y_z_x",
            "native_evidence_jar_sha256": BRIDGE,
        },
        core_min_chunk_xz=np.asarray([0, 0], dtype=np.int32),
        section_known=known,
        cell_code=code,
        cell_palette=cell,
        shape_palette=ShapePalette(boxes=boxes, box_mask=box_mask),
        filler_root_offset_packed=np.zeros_like(code),
    )


def _entry(asset_id: str) -> RegionBlockSemanticEntry:
    return RegionBlockSemanticEntry(
        semantic_key=block_semantic_key(asset_id, 0),
        asset_key=block_asset_key(asset_id),
        valid=True,
        affordance_tags=block_affordance_tag_mask("breakable"),
        gather_type_index=1,
        required_tool_quality=2,
        rotation_index=0,
    )


def _semantic_snapshot(physical: NativeRegionSnapshot, bench_block_id: str) -> NativeRegionBlockSemanticSnapshot:
    code = np.zeros_like(physical.cell_code)
    for position in _floor():
        slot, section, index = _cell_location(position)
        code[slot, section, index] = 1
    slot, section, index = _cell_location(BENCH_CELL)
    code[slot, section, index] = 2
    return NativeRegionBlockSemanticSnapshot(
        core_min_chunk_xz=physical.core_min_chunk_xz,
        section_known=physical.section_known,
        cell_code=code,
        palette=(RegionBlockSemanticEntry.air(), _entry("Rock_Stone"), _entry(bench_block_id)),
        source_region_semantic_sha256=physical.semantic_digest(),
        evidence_bridge_sha256=BRIDGE,
    )


class Region(NamedTuple):
    base: object  # the region geometry: the physical floor the actor stands on
    runtime: object  # RegionActionRuntimeState: geometry with its mutable blocks, and the semantic atlas
    role_mask: object  # opaque palette flags, the shape the block producer expects


def _region(m: dict, bench_block_id: str) -> Region:
    world = m["world"]
    physical = _physical_snapshot()
    atlas = world.region_atlas_from_snapshots([physical], world_ids=[WORLD_ID], region_capacity=2)
    semantic_atlas = world.region_block_semantic_atlas_from_snapshots(
        [_semantic_snapshot(physical, bench_block_id)], [physical], atlas, world_ids=[WORLD_ID], palette_capacity=5,
    )
    base = world.region_geometry_from_atlas(
        atlas, [WORLD_ID],
        agent_bounds=[-0.3, 0.0, -0.3, 0.3, 1.8, 0.3], target_bounds=[-0.3, 0.0, -0.3, 0.3, 1.8, 0.3],
        agent_los_offset=[0.0, 1.6, 0.0], target_los_offset=[0.0, 1.6, 0.0], max_los_distance=LOS_DISTANCE,
    )
    mutable = world.mutable_block_state_from_region_snapshots([physical], base, world_ids=[WORLD_ID], mutation_capacity=4)
    runtime = world.RegionActionRuntimeState(world.bind_region_mutable_blocks(base, mutable), semantic_atlas)
    role_mask = (atlas.cell_flags.astype(jnp.int32) & jnp.int32(FLAG_OPAQUE)) != 0
    return Region(base=base, runtime=runtime, role_mask=role_mask)


# --- The craft: the recipe, its bench row and its stock --------------------------------------------------------------------


def _admitted_rows(m: dict, records: list[dict]) -> list:
    benches = {bench["id"] for record in records for bench in record["bench_requirements"] if bench["type"] in (0, 1, 2, 3)}
    admitted = benches | {"Fieldcraft"}
    rows = []
    for record in records:
        row = NativeCraftingRecipe.from_response(record)
        if m["crafting"]._lossless_region_crafting(row, admitted):
            rows.append(m["crafting"]._resolved_recipe(row))
    return rows


def _assets(m: dict, scenario: Scenario):
    crafting, contract = m["crafting"], m["contract"]
    resolved = _admitted_rows(m, [scenario.record])
    assert len(resolved) == 1, scenario.recipe_id
    packed = m["packing"].pack_resolved_recipe_table(
        tuple(resolved),
        source_sha256="0" * 64,
        expected_recipe_count=1,
        item_id_resolver=m["semantic"].semantic_id,
        resource_type_id_resolver=m["semantic"].semantic_id,
    )
    bench = scenario.bench
    return crafting.RegionFieldcraftAssets(
        packed=packed,
        output_max_stack=jnp.full((contract.RECIPE_TABLE_CAPACITY, contract.MAX_OUTPUTS), 64, dtype=jnp.int32),
        output_max_durability=jnp.zeros((contract.RECIPE_TABLE_CAPACITY, contract.MAX_OUTPUTS), dtype=jnp.float32),
        inventory_layout=m["inventory"].default_inventory_layout(),
        source_bridge_sha256="0" * 64,
        bench_semantic_key=jnp.asarray([block_semantic_key(bench["block_id"], 0)], dtype=jnp.uint32),
        bench_id_hash=jnp.asarray(
            np.asarray(m["crafting"].identity_sha256_words(bench["bench_id"]), dtype=np.uint32)[None, :], dtype=jnp.uint32,
        ),
        bench_type=jnp.asarray([bench["type"]], dtype=jnp.int32),
        bench_tier_level=jnp.asarray([bench["tier"]], dtype=jnp.int32),
        bench_valid=jnp.ones((1,), dtype=jnp.bool_),
        memories_level=1,
    )


def _inventory(m: dict, layout, stock) -> object:
    """The stock in the hotbar's first slots, as the replays place it."""

    inventory = m["inventory"]
    state = inventory.empty_inventory_state(1, entity_count=m["types"].ENTITY_COUNT)
    hotbar = np.flatnonzero(np.asarray(layout.container_id) == inventory.CONTAINER_HOTBAR)
    item_id = np.asarray(state.item_id).copy()
    quantity = np.asarray(state.quantity).copy()
    for slot, (asset_id, count) in zip(hotbar, stock, strict=False):
        item_id[0, 0, slot] = m["semantic"].semantic_id(asset_id)
        quantity[0, 0, slot] = count
    return state._replace(item_id=jnp.asarray(item_id), quantity=jnp.asarray(quantity))


def _inert_interactions(m: dict):
    """One item row that never matches (valid False): the craft scenarios request no block interaction."""

    executor, block_interactions, world = m["executor"], m["block_interactions"], m["world"]
    template = block_interactions.empty_block_interaction_program(1, entity_count=1)
    program = executor.BlockInteractionProgram(**{
        name: jnp.stack([jnp.stack([getattr(template, name)[0, 0]] * 2)]) for name in executor.BlockInteractionProgram._fields
    })
    return executor.RegionBlockInteractionTable(
        item_id=jnp.asarray([-1], dtype=jnp.int32),
        valid=jnp.zeros((1,), dtype=jnp.bool_),
        trigger_valid=jnp.zeros((1, 2), dtype=jnp.bool_),
        program=program,
        break_tool_type_id=jnp.zeros((1, 2), dtype=jnp.int32),
        break_match_tool=jnp.zeros((1, 2), dtype=jnp.bool_),
        place_semantic_key=jnp.zeros((1, 2, world.BLOCK_SEMANTIC_KEY_WORDS), dtype=jnp.uint32),
        place_semantic_key_valid=jnp.zeros((1, 2), dtype=jnp.bool_),
    )


def _environment(m: dict, scenario: Scenario, region: Region, assets):
    """The production arsenal environment over the Region, with the rendered executor and the scheduled craft requests."""

    environment, executor, world = m["environment"], m["executor"], m["world"]
    params = m["combat"].default_combat_params(microticks=1, target_active=False)
    config = m["runtime"].arsenal_runtime_config(
        m["profiles"].hytale_0_5_7_entity_loadouts((("iron_sword", "iron_mace"),)),
        opponent_controller_mask=jnp.asarray((False, False)),
    )
    block_provider = environment.make_region_runtime_block_candidate_provider(
        role_opaque_mask=region.role_mask, maximum_distance=MAXIMUM_REACH,
        view_sector_full_angle_radians=2.0 * np.pi, cell_radius=4,
    )
    recipe_provider = m["crafting"].make_region_fieldcraft_candidate_provider(assets, block_candidate_provider=block_provider)
    interactions = _inert_interactions(m)
    held = executor.RegionHeldToolTable(
        item_id=jnp.full((1,), -1, dtype=jnp.int32),
        valid=jnp.zeros((1,), dtype=jnp.bool_),
        tool=m["catalog"].empty_block_tool_state((1,)),
    )
    placed = m["placement"].compile_region_placed_geometry_table(
        region.runtime, interactions.place_semantic_key, interactions.place_semantic_key_valid,
    )
    catalog = world.block_action_catalog_from_local((resolve_local_block_action_semantics("Rock_Stone", {}, {}),))
    gather = world.block_gather_defaults_from_local([None] * len(GATHER_TYPES))
    craft = executor.make_region_block_action_surface_executor(
        block_candidate_provider=block_provider, interaction_table=interactions, held_tool_table=held,
        placed_geometry_table=placed, block_catalog=catalog, gather_defaults=gather,
        inventory_layout=assets.inventory_layout, drop_programs=None, recipe_candidate_provider=recipe_provider,
        crafting_assets=assets, state_change_use_table=None, modification_allowed=False,
    )
    # One recipe per table: the head's only slot is 0 (legal or not, execute_region_fieldcraft rechecks it).
    assert len(assets.packed.recipe_ids) == 1
    requested_ticks = jnp.asarray([tick + 1 for tick in scenario.requests], dtype=jnp.int32)  # executor ticks are env steps + 1

    def with_request(state, surface_runtime, surface, evidence, keys, params_, config_):
        due = jnp.any(state.combat.tick_count[:, None] == requested_ticks[None, :], axis=1)
        requested = jnp.where(due, jnp.int32(0), jnp.int32(-1))
        return craft(state, surface_runtime, surface._replace(recipe_candidate_index=requested), evidence, keys, params_, config_)

    action_provider = environment.make_arsenal_action_surface_provider(
        block_candidate_provider=block_provider, recipe_candidate_provider=recipe_provider,
        item_interaction_availability_provider=None,
    )

    def world_provider(state, current_world, combat_params):
        batch = state.combat.health.shape[0]
        current = current_world.geometry if isinstance(current_world, world.RegionActionRuntimeState) else current_world
        tokens = m["world_tokens"].empty_world_geometry_policy_tokens(batch)
        light = m["light_tokens"].empty_actor_light_policy_tokens(batch, tokens.token_mask.shape[1])
        return environment.ArsenalWorldRuntimeViews(
            physical_geometry=current,
            capabilities=environment.geometry_arsenal_world_capabilities(state, combat_params, geometry=current, config=config),
            features=m["features"].empty_injected_world_features(batch),
            tokens=tokens,
            light_tokens=light,
        )

    inventory = _inventory(m, assets.inventory_layout, scenario.stock)
    return environment.make_arsenal_environment(
        params, config, maximum_turn_degrees=45.0,
        geometry_provider=region.base,
        world_runtime_provider=world_provider,
        action_surface_provider=action_provider,
        action_surface_executor=with_request,
        action_surface_runtime_initializer=lambda *_args: region.runtime,
        reset_provider=lambda keys: environment.ArsenalResetBatch(
            agent_position=jnp.asarray([ACTOR], dtype=jnp.float32),
            target_position=jnp.asarray([TARGET], dtype=jnp.float32),
        ),
        inventory_reset_provider=lambda keys, runtime_config: inventory,
    )


# --- The run ---------------------------------------------------------------------------------------------------------------


def _snapshot(m: dict, layout, state) -> dict:
    combat, inventory = state.runtime.combat, state.runtime.inventory
    hotbar = np.flatnonzero(np.asarray(layout.container_id) == m["inventory"].CONTAINER_HOTBAR)
    slots = {
        position: (int(inventory.item_id[0, 0, slot]), int(inventory.quantity[0, 0, slot]))
        for position, slot in enumerate(hotbar)
        if int(inventory.quantity[0, 0, slot]) > 0
    }
    return {
        "tick": int(combat.tick_count[0]),
        "job": int(combat.craft_job_recipe[0]),
        "queued": int(combat.craft_queued_recipe[0]),
        "pending": bool(combat.craft_job_inputs_pending[0]),
        "elapsed": float(combat.craft_job_elapsed_seconds[0]),
        "bench": int(combat.proc_recipe[0]),
        "bench_active": bool(combat.proc_active[0]),
        "bench_wait": int(combat.proc_wait_ticks[0]),
        "bench_progress": float(combat.proc_progress_seconds[0]),
        "fuel_quantity": int(combat.proc_fuel_quantity[0]),
        "fuel_seconds": float(combat.proc_fuel_seconds[0]),
        "bench_inputs": [int(v) for v in np.asarray(combat.proc_input_quantity[0]) if v > 0],
        "bench_output": [(int(i), int(q)) for i, q in zip(np.asarray(combat.proc_output_item[0]),
                                                           np.asarray(combat.proc_output_quantity[0]), strict=True) if q > 0],
        "diagram": int(combat.diagram_recipe[0]),
        "diagram_ticks": int(combat.diagram_ticks[0]),
        "hotbar": slots,
    }


class Run(NamedTuple):
    start: dict  # the snapshot after reset
    steps: list  # steps[i] is the snapshot after env step i
    index: int  # the recipe's index in the table
    names: dict  # semantic item id -> asset id, for the items the scenario names


def _run(package: str, scenario: Scenario) -> Run:
    m = _modules(package)
    assets = _assets(m, scenario)
    layout = assets.inventory_layout
    region = _region(m, scenario.bench["block_id"])
    environment = _environment(m, scenario, region, assets)
    state, _structured = environment.reset(jax.random.split(jax.random.key(ACTOR_SEED), 1))
    start = _snapshot(m, layout, state)
    neutral = m["policy_layout"].neutral_arsenal_policy_action_factors(1)
    steps = []
    for step in range(scenario.steps):
        state, _observation, _reward, _done, _info = environment.step_factors(
            state, neutral, jax.random.split(jax.random.key(913771 + step), 1),
        )
        steps.append(_snapshot(m, layout, state))
    names = {m["semantic"].semantic_id(name): name for name in (*(asset for asset, _count in scenario.stock), scenario.output)}
    index = list(assets.packed.recipe_ids).index(scenario.recipe_id)
    return Run(start=start, steps=steps, index=index, names=names)


def _held(run: Run, snapshot: dict, asset_id: str) -> int:
    """The quantity of an asset in the hotbar of a snapshot."""

    item = next(key for key, name in run.names.items() if name == asset_id)
    return sum(quantity for stored, quantity in snapshot["hotbar"].values() if stored == item)


@pytest.fixture(scope="module")
def timed(package) -> Run:
    with _materials({}):
        return _run(package, TIMED)


@pytest.fixture(scope="module")
def processing(package) -> Run:
    with _materials({}):
        return _run(package, PROCESSING)


@pytest.fixture(scope="module")
def diagram(package) -> Run:
    with _materials(_item_resource_table(_modules(package), [item for item, _count in AXE_STOCK])):
        return _run(package, DIAGRAM)


# --- Assertions ------------------------------------------------------------------------------------------------------------


def test_the_policy_decode_never_requests_a_recipe(package) -> None:
    """The recipe head is not published to the combat policy (policy/surface.py: recipe_index is -1 for every action), so the
    craft verb of these scenarios is requested through the decoded surface, not through a policy action."""

    m = _modules(package)
    layout = m["policy_surface"].action_surface_layout(m["policy_layout"].ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY)
    rng = np.random.default_rng(20261018)
    sizes = np.asarray(layout.head_sizes)
    factors = np.stack([rng.integers(0, size, size=64) for size in sizes], axis=1).astype(np.int32)
    factors[0] = np.asarray(m["policy_layout"].neutral_arsenal_policy_action_factors(1))[0]
    mask = jnp.ones((factors.shape[0], int(sizes.sum())), dtype=jnp.bool_)
    decoded = m["policy_surface"].decode_staged_action_surface_factors(jnp.asarray(factors), mask, layout=layout)
    assert np.all(np.asarray(decoded.recipe_candidate_index) == -1)


def test_a_timed_craft_takes_its_inputs_on_the_first_progress_tick_and_outputs_on_the_ninetieth(timed: Run) -> None:
    """Replay (ac): inputs out on R+1 (the first progress tick), the output on R+90 (3.0 s at 1/30 s per tick)."""

    steps = timed.steps
    assert timed.start["job"] == -1
    # R: the request starts the job and leaves the inventory as it was.
    assert steps[0]["job"] == timed.index and steps[0]["pending"] is True
    assert _held(timed, steps[0], "Ingredient_Bar_Copper") == 22 and _held(timed, steps[0], "Ingredient_Fibre") == 8
    # R+1: the inputs of the first job leave (11 bars, 4 fibre) and the first progress tick is counted.
    assert steps[1]["pending"] is False
    assert _held(timed, steps[1], "Ingredient_Bar_Copper") == 11 and _held(timed, steps[1], "Ingredient_Fibre") == 4
    assert steps[1]["elapsed"] == pytest.approx(DT)
    # R+90: the output lands in the first free hotbar slot (slot 2: the queued job's inputs still hold slots 0 and 1 when it
    # is routed), and the queued job starts its own progress on the same tick.
    landed = [tick for tick, snap in enumerate(steps) if any(
        timed.names.get(item) == "Armor_Copper_Chest" for item, _quantity in snap["hotbar"].values())]
    assert landed == [90]
    assert steps[90]["job"] == timed.index and steps[90]["queued"] == -1
    assert steps[90]["elapsed"] == pytest.approx(DT)


def test_a_second_request_queues_behind_the_running_job_and_starts_on_its_completion(timed: Run) -> None:
    """Replay (ac): the second request is queued (one deep) on R+11 and its inputs leave on the first job's completion."""

    steps = timed.steps
    assert steps[10]["queued"] == -1
    assert steps[11]["queued"] == timed.index and steps[11]["job"] == timed.index
    assert all(steps[tick]["queued"] == timed.index for tick in range(11, 90))
    # The completion tick takes the queued job's inputs (the last 11 bars and 4 fibre).
    assert _held(timed, steps[89], "Ingredient_Bar_Copper") == 11
    assert _held(timed, steps[90], "Ingredient_Bar_Copper") == 0 and _held(timed, steps[90], "Ingredient_Fibre") == 0


def test_a_processing_craft_starts_on_r_plus_two_and_leaves_its_output_on_r_plus_sixty_one(processing: Run) -> None:
    """Replay (ae): the first progress tick is R+2 (one stick lit), the output and the inputs leave the bench on R+61."""

    steps = processing.steps
    # R: the request moves the fish and the stick stack into the bench; the bench is lit and waits one tick.
    assert steps[0]["bench"] == processing.index and steps[0]["bench_active"] is True
    assert steps[0]["bench_wait"] == 1 and steps[0]["fuel_quantity"] == 2
    assert _held(processing, steps[0], "Food_Fish_Raw") == 0 and _held(processing, steps[0], "Ingredient_Stick") == 0
    # R+1: no progress yet. R+2: the first progress tick, and one stick is lit (2 -> 1 in the bench).
    assert steps[1]["bench_progress"] == 0.0 and steps[1]["fuel_quantity"] == 2
    assert steps[2]["bench_wait"] == 0 and steps[2]["bench_progress"] == pytest.approx(DT)
    assert steps[2]["fuel_quantity"] == 1
    # R+60: still cooking. R+61: the recipe completes, the output is in the bench, the lit stick has 2.0 s left.
    assert steps[60]["bench"] == processing.index
    assert steps[61]["bench"] == -1 and steps[61]["bench_active"] is False
    food = next(item for item, name in processing.names.items() if name == PROCESSING.output)
    assert steps[61]["bench_output"] == [(food, 1)]
    assert steps[61]["bench_inputs"] == []
    assert steps[61]["fuel_seconds"] == pytest.approx(2.0, abs=1e-3)


def test_an_armory_craft_is_queued_on_r_and_commits_on_r_plus_two(diagram: Run) -> None:
    """Replay (ae): the Armory click-craft of the axe is queued on R (queue 0 -> 1), its inputs are still held on R+1, and it
    commits on R+2 (inputs out, the axe in). The inputs are resource types (Wood_Trunk x2, Rock x6) matched through the item
    table of the scenario (``diagram`` fixture)."""

    steps = diagram.steps
    assert steps[0]["diagram"] == diagram.index and steps[0]["diagram_ticks"] == 2
    assert _held(diagram, steps[0], "Plant_Fern_Jungle_Trunk") == 2 and _held(diagram, steps[0], "Deco_Coral_Shell_Purple") == 6
    assert steps[1]["diagram"] == diagram.index and steps[1]["diagram_ticks"] == 1
    assert _held(diagram, steps[1], "Plant_Fern_Jungle_Trunk") == 2 and _held(diagram, steps[1], "Deco_Coral_Shell_Purple") == 6
    assert steps[2]["diagram"] == -1
    assert _held(diagram, steps[2], "Plant_Fern_Jungle_Trunk") == 0 and _held(diagram, steps[2], "Deco_Coral_Shell_Purple") == 0
    assert _held(diagram, steps[2], "Ingredient_Leather_Light") == 0
    assert _held(diagram, steps[2], DIAGRAM.output) == 1


def test_the_axe_is_offered_at_the_armory_only_with_the_item_resource_table(package) -> None:
    """The axe's inputs are resource types. The candidate provider matches them through the crafting-materials table, which the
    production compile fills from the archive (``compile_region_fieldcraft_assets``); a hand-built asset set does not fill it.
    With an empty table the axe is not offered at the Armory; with the archive's ResourceTypes for its stock it is."""

    m = _modules(package)
    assets = _assets(m, Scenario(
        recipe_id=DIAGRAM.recipe_id, record=AXE_RECORD, bench=DIAGRAM.bench, stock=AXE_STOCK,
        output=DIAGRAM.output, requests=(0,), steps=0,
    ))
    with _materials({}):
        assert not _axe_offered(m, assets)
    with _materials(_item_resource_table(m, [item for item, _count in AXE_STOCK])):
        assert _axe_offered(m, assets)


def _axe_offered(m: dict, assets) -> bool:
    crafting = m["crafting"]
    inventory = crafting._crafting_inventory(
        crafting._actor_inventory(_inventory(m, assets.inventory_layout, AXE_STOCK)), assets.inventory_layout,
    )
    legal = crafting._legal_recipes(crafting._repeat_inventory_for_candidates(inventory, 1), _armory_context(m),
                                    assets.packed.table)
    index = list(assets.packed.recipe_ids).index(DIAGRAM.recipe_id)
    return bool(np.asarray(legal)[0, 0, 0, index])


def _item_resource_table(m: dict, items) -> dict:
    """The materials table of ``crafting_materials.item_resource_types``, for the given items: item semantic id -> the semantic ids
    of its ResourceTypes (read from the archive)."""

    from hytalegym.worldgen.surrogate._archive import HytaleAssetArchive
    from hytalegym.worldgen.surrogate.semantics import LocalBlockSemanticsResolver

    semantic_id, crafting = m["semantic"].semantic_id, m["crafting"]
    rows: dict[int, list[int]] = {}
    with HytaleAssetArchive(ARCHIVES["gen7"]) as archive:
        resolver = LocalBlockSemanticsResolver(archive)
        for asset_id in items:
            item, _provenance = resolver.load_inherited_item_asset(asset_id)
            names = [entry.get("Id") for entry in item.get("ResourceTypes") or () if isinstance(entry, dict)]
            names = [name for name in names if isinstance(name, str) and name]
            if names:
                rows[semantic_id(asset_id)] = [semantic_id(name) for name in names]
    keys = sorted(rows)
    values = np.full((len(keys), crafting.MAX_RESOURCE_TYPES_PER_ITEM), crafting.ABSENT_ID, dtype=np.int32)
    for row, key in enumerate(keys):
        values[row, : len(rows[key])] = rows[key]
    return {"keys": jnp.asarray(keys, dtype=jnp.int32), "values": jnp.asarray(values), "jnp": jnp,
            "absent": crafting.ABSENT_ID}


@contextmanager
def _materials(table):
    """Set the crafting-materials table for one scenario, and restore the previous one after it (the global is shared by tests)."""

    saved = crafting_materials._TABLE
    crafting_materials._TABLE = table
    try:
        yield
    finally:
        crafting_materials._TABLE = saved


def _armory_context(m: dict):
    """A bench context for one candidate at an Armory (type 2, tier 1), as the candidate producer builds it."""

    contract, crafting = m["contract"], m["crafting"]
    words = np.asarray(crafting.identity_sha256_words("Armory"), dtype=np.uint32)
    return crafting.CraftingContext(
        window_open=jnp.ones((1, 1, 1), dtype=jnp.bool_),
        bench_type=jnp.full((1, 1, 1), 2, dtype=jnp.int32),
        bench_id_hash=jnp.broadcast_to(jnp.asarray(words, dtype=jnp.uint32), (1, 1, 1, 8)),
        bench_tier_level=jnp.full((1, 1, 1), 1, dtype=jnp.int32),
        memories_level=jnp.ones((1, 1, 1), dtype=jnp.int32),
        known_recipe=jnp.ones((1, 1, 1, contract.RECIPE_TABLE_CAPACITY), dtype=jnp.bool_),
        creative_mode=jnp.zeros((1, 1, 1), dtype=jnp.bool_),
        input_mode=jnp.full((1, 1, 1), contract.INPUT_MODE_NORMAL, dtype=jnp.int32),
    )
