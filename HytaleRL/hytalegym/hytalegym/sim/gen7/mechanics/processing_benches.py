"""Processing benches (gen 7, row 11p, crafting oracle stage 5). Switch ``Constructs.processing_benches``.

Measured on the live 0.7.2 bridge (FIDELITY-0.7.2.md section (v), jar 4A5E5D6D; receipts
``~/.claude/jobs/31947bd7/tmp/phase4l/receipts/v6_proc_*.json``, seeds 570057 and 31337): Campfire with Food_Fish_Grilled (2.0 s,
two sticks loaded at activation) and Furnace with Ingredient_Bar_Copper (10 s, four sticks). Tick numbers below are the request
tick R and the native world steps after it. Labels: CONFIRMED (a receipt or a bytecode line) or INFERRED (named where it applies).

1. Progress (CONFIRMED): each world tick adds (game seconds elapsed) / getSecondsPerTick to the progress. getSecondsPerTick is 30
   (worldtime_072.txt 93-112: 86400 / (day + night seconds) = 86400 / 2880). The receipts give progress / game-seconds = 1/30 on
   both benches (0.033332 to 0.033334). The game clock is 30 game seconds per real second, so a recipe second is a real second.
2. Clock (CONFIRMED vs INFERRED): the native world step advanced at least 0.999 game seconds (mean 1.046 to 1.068 per run), so a
   native step is about 5 % longer than 1/30 s and the native counts are about 5 % lower. The simulator counts 1/30 s per tick (the
   owner's 30 Hz ruling). That fits the recipe seconds, not the native tick counts: 2.0 s completes on R+61 here (native R+58/R+59),
   10 s on R+301 (native R+289/R+286). The 1/30 s clock is the timed-crafts clock (row 11j); this module does not change it.
   The native deficit (campfire 2 to 3 ticks, furnace 12 to 15 ticks) is the server's tick jitter under load, not a rule to model;
   the simulator keeps its own 1/30 s tick.
3. Start (CONFIRMED natively, INFERRED for the simulator): the bench takes its time base on its first world tick after the request;
   progress and the first fuel use begin on the next world tick. Campfire: the first progress tick is R+2 (both seeds). Furnace:
   R+3 (both seeds) because one native step advanced no game time. The simulator has no zero-advance step and starts every bench at
   R+2 (not modelled: the furnace's extra tick).
4. Completion (CONFIRMED): the output appears and the inputs leave the bench on the tick progress reaches the recipe time
   (campfire fish on R+58 and R+59, copper bar on R+289 and R+286; inputs and output on the same tick). The recipe time at tier 1 is
   unreduced (Bench_Furnace.json TierLevels[0].CraftingTimeReductionModifier 0.0). Tiers 2 and up are not modelled.
5. Fuel (CONFIRMED): fuel burns at the progress rate. Fuel seconds fall by the same increment as progress (advanceProcessing,
   crafting_javap_072.txt 6756-7060; consumeFuelForDuration 7330). A stick is FuelQuality 4 (Ingredient_Stick.json); the next stick
   is taken when the lit one runs out, in the tick it is needed (receipt: 3.9655 = 4.0 - 1/30 on the tick of the first stick). The
   completion tick burns only the progress that remains (campfire 2.00717 -> 2.0). After completion the bench is inactive and the
   unburnt fuel stays in it. Why a 4 s stick lasts 56 or 57 ticks on the Campfire and about 114 on the Furnace: the Campfire recipe
   burns 2 s of it (its time), the Furnace recipe burns the full stick (its time is 10 s); the burn rate is the same on both benches.
6. By-products (CONFIRMED for the Furnace, INFERRED for the Campfire): ExtraOutput PerFuelItemsConsumed 2 and Outputs
   Ingredient_Charcoal 1 on both benches (Bench_Furnace.json, Bench_Campfire.json); the counter starts at 2 (initializeBenchConfig,
   crafting_javap_072.txt 5901-5915) and falls by one per fuel item that is not an IgnoredFuelSource (consumeOneFuel 7118-7140), so the
   second consumption yields one charcoal and the third none. Furnace receipts: charcoal on the second consumption (native R+114 and
   R+116). The Campfire's second consumption never happens in the receipts (untested).
7. Fuel exhaustion (INFERRED, unmeasured): progress is capped by the fuel the bench holds (advanceProcessing, fuelSets), and an
   active bench with no fuel left deactivates (ProcessingBenchTick, crafting_javap_072.txt 1226-1316). The simulator stops progress
   and deactivates the bench on that tick; a later request is refused while a recipe is pending.
8. Benches without a fuel slot (CONFIRMED in the assets, INFERRED in the rule): Salvagebench and Tannery have no Fuel entry
   (Bench_Salvage.json, Bench_Tannery.json); their progress takes no fuel. Unmeasured.
9. Collection (INFERRED, unmeasured): a request at a bench whose output stacks are still there first moves each stack into an empty
   inventory slot of the actor (no stacking); a stack that finds no slot stays. Native collection is a client action not measured.
10. Request (INFERRED): a processing request needs an in-reach bench of type 1 (Bench.Type Processing) on which the recipe is legal, an
    idle bench, and, for a fuel bench, fuel in the bench or a fuel stack in the actor's inventory (native setActive refuses an empty
    fuel slot, setActive at 6253-6306). The whole first fuel stack moves into the bench (the receipts loaded 2 and 4 sticks by scenario).
    The inputs leave the inventory at the request and sit in the bench until completion (receipt: bench_input until tick R+58).

State (``CombatState`` leaves, patches ``processing_fields`` and ``processing_init``; per lane, the controlled actor 0):
``proc_bench_cell`` (int32[3]), ``proc_recipe`` (int32, -1 idle), ``proc_active`` (bool), ``proc_has_fuel`` (bool),
``proc_wait_ticks`` (int32, ticks before the first progress tick), ``proc_progress_seconds`` (float32), ``proc_fuel_seconds``
(float32, the lit stick's remaining seconds), ``proc_fuel_item`` (int32), ``proc_fuel_quantity`` (int32), ``proc_fuel_quality``
(float32, FuelQuality of the loaded stack), ``proc_extra_countdown`` (int32, starts at 2), ``proc_input_item`` and
``proc_input_quantity`` (int32[28], the recipe's inputs held by the bench), ``proc_output_item`` and ``proc_output_quantity``
(int32[4], the bench's output slots).

Hooks and patches (``sim/gen7/overlay.py``, row 11p): patch ``processing_fields`` and ``processing_init`` (the leaves); patch
``processing_bench_types`` (``_BENCH_TYPE_NAMES`` maps Processing to 1, so the Campfire, Furnace, Salvagebench and Tannery rows exist)
and hook ``processing_admission`` (maps a type-1 requirement to type 0 for the zero-time gate only; the bench candidate matching
sees the real type; a timed recipe reaches it through the timed admission hook);
patch ``craft_pre_inventory`` (the inventory before the craft, shared with diagram_crafting); patches ``processing_executor_import``
and ``processing_executor_start`` (the craft verb calls ``start`` after the timed block); hook ``processing_executor`` (each executor
step first advances the benches, under ``lax.cond``). The timed construct's ``timed_recipe`` excludes processing recipes (its
predicate is ``processing_recipe``), so a processing request never starts a timed job. Gen 5 (``absent``) returns every original.

Not modelled (each named where it applies): the start tick is R+2 on every bench (see 3); tiers 2 and up (see 4); a stack of a
different fuel item than the one the bench holds is refused (inferred: a fuel slot holds one item type); the reach of a walk-away
(the bench keeps running, inferred from the tick, which reads no window); the furnace's extra start tick (see 3).
"""

from __future__ import annotations

from dataclasses import replace
from functools import wraps
from inspect import unwrap
from typing import Any

import numpy as np

from hytalegym.jax.crafting import identity_sha256_words
from hytalegym.sim.gen7.mechanics.timed_crafting import (
    _env,
    _remove,
    _select_tree,
    processing_recipe,
)

SWITCH = "processing_benches"
IMPLEMENTED = True
ABSENT = "absent"
TICKING_BENCH = "ticking_bench"
#: Bench.Type Processing (crafting/contract.py BENCH_TYPE_PROCESSING).
BENCH_TYPE_PROCESSING = 1
#: Ticks between the request and the first progress tick: the native first progress tick is R+2 on the Campfire (section (v)).
START_WAIT_TICKS = 1
#: Fuel consumptions between two by-products (ExtraOutput.PerFuelItemsConsumed, Bench_Furnace.json and Bench_Campfire.json).
PER_FUEL_ITEMS = 2
#: Bench output slots (OutputSlotsCount 4 on Campfire, Furnace and Salvagebench; Tannery has 2, not modelled).
OUTPUT_SLOTS = 4
#: Recipe inputs held by one bench (crafting/contract.py MAX_INGREDIENTS).
INPUT_SLOTS = 28
#: Item asset -> FuelQuality (the eleven 0.7.2 items that declare one, read from Assets.zip on 2026-10-18).
FUEL_QUALITY = {
    "Ingredient_Charcoal": 6.0,
    "Ingredient_Fibre": 2.0,
    "Ingredient_Stick": 4.0,
    "Ingredient_Tree_Bark": 3.0,
    "Ingredient_Tree_Sap": 6.0,
    "Wood_Oak_Trunk": 9.0,
    "Wood_Softwood_Beam": 3.0,
    "Wood_Softwood_Planks": 6.0,
    "Wood_Softwood_Planks_Half": 3.0,
    "Wood_Softwood_Stairs": 6.0,
    "Wood_Sticks": 4.0,
}
#: ExtraOutput.IgnoredFuelSources: these fuels do not count towards a by-product (Bench_Furnace.json, Bench_Campfire.json).
IGNORED_FUEL = ("Ingredient_Charcoal", "Ingredient_Fibre", "Ingredient_Tree_Sap")
#: The by-product (ExtraOutput.Outputs, one item of one).
BYPRODUCT = "Ingredient_Charcoal"
#: Benches with a Fuel slot (Bench_Campfire.json and Bench_Furnace.json; Bench_Salvage.json and Bench_Tannery.json have none).
FUEL_BENCHES = ("Campfire", "Furnace")
#: A completion within this many ticks of the recipe time counts (the float32 sum of 1/30 s).
COMPLETION_SLACK_TICKS = 0.5


def admission(original: Any, profile: Any) -> Any:
    """Hook for ``_lossless_region_crafting``: a processing (bench type 1) requirement passes the zero-time gate as an ordinary bench requirement.

    Exactly one gate sees the mapped type: ``_lossless_region_crafting`` (the compile's admission), which sees type 0 for a type-1
    requirement and still requires the bench id to be admitted (the Processing rows are admitted by patch
    ``processing_bench_types``). The packed table keeps the recipe's own type, because the compile packs the original recipes. The
    bench candidate matching is unchanged: ``_bench_contexts``, ``_legal_recipes`` and ``_recipe_precondition`` see the real bench
    type (a Campfire or Furnace context is type 1), so a processing recipe is legal only at a type-1 bench. The timed hook (listed
    before this one, so it runs first) keeps a timed recipe's time.
    """

    if not _enabled(profile):
        return original

    @wraps(original)
    def _lossless_region_crafting(recipe, admitted_bench_ids):
        requirements = tuple(
            replace(requirement, bench_type=0) if requirement.bench_type == BENCH_TYPE_PROCESSING else requirement
            for requirement in recipe.bench_requirements
        )
        return original(replace(recipe, bench_requirements=requirements), admitted_bench_ids)

    return _lossless_region_crafting


def _enabled(profile: Any) -> bool:
    value = profile.constructs.processing_benches
    if value == ABSENT:
        return False
    if value != TICKING_BENCH:
        raise ValueError(f"unknown processing_benches switch {value!r}")
    return True


def _ids(env: dict[str, Any], names) -> Any:
    return env["jnp"].asarray([env["semantic_id"](name) for name in names], dtype=env["jnp"].int32)


def _fuel_words() -> Any:
    return np.asarray([identity_sha256_words(name) for name in FUEL_BENCHES], dtype=np.uint32)


def bench_of_type(crafting_function: Any, assets: Any, layout: Any, candidates: Any, world: Any, inventory: Any, index: Any,
                  bench_type: int):
    """The first in-reach candidate of ``bench_type`` on which each lane's recipe is legal.

    Returns (cell int32[batch, 3], found bool[batch], bench id hash uint32[batch, 8]). Mirrors ``timed_crafting.bench_for_recipe``
    with the bench type filter (``_bench_contexts`` gives the type and the id hash of each candidate).
    """

    env = _env(crafting_function)
    jnp = env["jnp"]
    table = assets.packed.table
    count = candidates.candidate_mask.shape[2]
    current = env["requery_region_block_action_candidates"](world.geometry, world.semantic_atlas, candidates)
    bench = env["_bench_contexts"](assets, candidates, current)
    actor = env["_crafting_inventory"](env["_actor_inventory"](inventory), layout)
    legal = env["_legal_recipes"](env["_repeat_inventory_for_candidates"](actor, count), bench, table)
    batch = index.shape[0]
    safe = jnp.clip(index, 0, legal.shape[-1] - 1)
    per_candidate = jnp.take_along_axis(
        legal[:, 0], jnp.broadcast_to(safe[:, None, None], (batch, count, 1)), axis=2
    )[..., 0]
    usable = per_candidate & bench.window_open[:, 0] & (bench.bench_type[:, 0] == bench_type) & (index >= 0)[:, None]
    first = jnp.argmax(usable, axis=1).astype(jnp.int32)
    found = jnp.any(usable, axis=1)
    visible = candidates.visible_position[:, 0]
    position = jnp.take_along_axis(visible, jnp.broadcast_to(first[:, None, None], (batch, 1, 3)), axis=1)[:, 0]
    words = jnp.take_along_axis(
        bench.bench_id_hash[:, 0], jnp.broadcast_to(first[:, None, None], (batch, 1, 8)), axis=1
    )[:, 0]
    return (
        jnp.where(found[:, None], position, 0).astype(jnp.int32),
        found,
        jnp.where(found[:, None], words, 0).astype(jnp.uint32),
    )


def _place(jnp: Any, item: Any, quantity: Any, new_item: Any, new_quantity: Any, mask: Any):
    """Stack ``new_quantity`` of ``new_item`` into the bench's output slots: the same item's slot, else the first empty one."""

    slots = jnp.arange(OUTPUT_SLOTS)[None, :]
    same = (item == new_item[:, None]) & (quantity > 0)
    empty = quantity == 0
    target = jnp.where(jnp.any(same, axis=1), jnp.argmax(same, axis=1), jnp.argmax(empty, axis=1)).astype(jnp.int32)
    room = jnp.any(same | empty, axis=1)
    put = mask & room & (new_quantity > 0)
    hit = put[:, None] & (slots == target[:, None])
    quantity = jnp.where(hit, quantity + new_quantity[:, None], quantity)
    item = jnp.where(hit, new_item[:, None], item)
    return item.astype(jnp.int32), quantity.astype(jnp.int32)


def _collect(jnp: Any, actor: Any, out_item: Any, out_quantity: Any, allowed: Any):
    """Move each output stack into an empty allowed slot of the actor's inventory (INFERRED: first fit, no stacking)."""

    item, quantity = actor.item_id[:, 0], actor.quantity[:, 0]
    columns = jnp.arange(quantity.shape[1])[None, :]
    for k in range(OUTPUT_SLOTS):
        free = (quantity == 0) & allowed[None, :]
        slot = jnp.argmax(free, axis=1).astype(jnp.int32)
        move = (out_quantity[:, k] > 0) & jnp.any(free, axis=1)
        hit = move[:, None] & (columns == slot[:, None])
        item = jnp.where(hit, out_item[:, k][:, None], item)
        quantity = jnp.where(hit, out_quantity[:, k][:, None], quantity)
        out_quantity = out_quantity.at[:, k].set(jnp.where(move, 0, out_quantity[:, k]))
        out_item = out_item.at[:, k].set(jnp.where(move, -1, out_item[:, k]))
    return actor._replace(item_id=item[:, None], quantity=quantity[:, None]), out_item, out_quantity


def start(crafting_function: Any, state: Any, craft: Any, inventory_before: Any, candidates: Any, world: Any, assets: Any,
          layout: Any):
    """Executor patch after the timed block: a processing request starts the bench; every other lane passes through.

    The zero-time commit of ``execute_region_fieldcraft`` has already moved the inventory of a processing craft, so the request
    works on ``inventory_before`` (the craft's inventory before the commit). Returns ``(accepted, state)``.
    """

    env = _env(crafting_function)
    jnp, jax = env["jnp"], env["jax"]
    table = assets.packed.table
    index = craft.recipe_index
    processing = craft.accepted & processing_recipe(jnp, table, index)
    batch = index.shape[0]
    position, found, words = jax.lax.cond(
        jnp.any(processing),
        lambda: bench_of_type(crafting_function, assets, layout, candidates, world, inventory_before, index,
                              BENCH_TYPE_PROCESSING),
        lambda: (jnp.zeros((batch, 3), dtype=jnp.int32), jnp.zeros((batch,), dtype=jnp.bool_),
                 jnp.zeros((batch, 8), dtype=jnp.uint32)),
    )
    has_fuel = jnp.any(jnp.all(words[:, None, :] == jnp.asarray(_fuel_words())[None], axis=-1), axis=-1)
    started, inventory, combat = request(crafting_function, state, index, processing, position, found, has_fuel,
                                         inventory_before, assets, layout)
    accepted = jnp.where(processing, started, craft.accepted)
    inventory = _select_tree(env, processing, inventory, state.inventory)
    return accepted, state._replace(inventory=inventory, combat=combat)


def request(crafting_function: Any, state: Any, index: Any, processing: Any, position: Any, found: Any, has_fuel: Any,
            inventory_before: Any, assets: Any, layout: Any):
    """The request on the processing lanes: collect the old outputs, remove the inputs and the fuel stack, set the bench leaves.

    A refused lane keeps the pre-craft inventory and its bench leaves. Returns ``(started, inventory, combat)``.
    """

    env = _env(crafting_function)
    jnp = env["jnp"]
    combat = state.combat
    actor = env["_actor_inventory"](inventory_before)
    allowed = jnp.isin(
        layout.container_id,
        jnp.asarray((env["CONTAINER_HOTBAR"], env["CONTAINER_STORAGE"], env["CONTAINER_BACKPACK"]), dtype=jnp.int32),
    )
    idle = combat.proc_recipe < 0
    collected, out_item, out_quantity = _collect(jnp, actor, combat.proc_output_item, combat.proc_output_quantity, allowed)
    fuel_mask = jnp.isin(collected.item_id[:, 0], _ids(env, tuple(FUEL_QUALITY))) & (collected.quantity[:, 0] > 0)
    has_stack = jnp.any(fuel_mask, axis=1)
    slot = jnp.argmax(fuel_mask, axis=1).astype(jnp.int32)
    stack_item = jnp.take_along_axis(collected.item_id[:, 0], slot[:, None], axis=1)[:, 0]
    stack_quantity = jnp.take_along_axis(collected.quantity[:, 0], slot[:, None], axis=1)[:, 0]
    mismatch = (combat.proc_fuel_quantity > 0) & has_stack & (combat.proc_fuel_item != stack_item)
    # setActive refuses only an empty fuel slot: a bench that still holds fuel needs no stick from the actor.
    fuel_ok = (~has_fuel) | has_stack | (combat.proc_fuel_quantity > 0)
    accept = processing & found & idle & fuel_ok & ~mismatch
    removed, crafted, _, _ = _remove(env, assets, layout, collected, index)
    take = accept & crafted
    loads = take & has_fuel & has_stack
    columns = jnp.arange(collected.quantity.shape[-1])[None, :]
    drop = loads[:, None] & (columns == slot[:, None])
    actor_after = _select_tree(env, take, removed, collected)
    actor_after = actor_after._replace(
        item_id=jnp.where(drop[:, None, :], env["ABSENT_ID"], actor_after.item_id),
        quantity=jnp.where(drop[:, None, :], 0, actor_after.quantity),
    )
    inventory = env["_merge_actor_inventory"](inventory_before, _select_tree(env, take, actor_after, actor))
    table = assets.packed.table
    safe = jnp.clip(index, 0, table.recipe_mask.shape[0] - 1)
    inputs_item = jnp.where(table.input_mask[safe], table.input_item_id[safe], env["ABSENT_ID"])
    inputs_quantity = jnp.where(table.input_mask[safe], table.input_quantity[safe], 0)
    quality = jnp.take(
        jnp.asarray(list(FUEL_QUALITY.values()), dtype=jnp.float32),
        jnp.argmax(_ids(env, tuple(FUEL_QUALITY))[None, :] == stack_item[:, None], axis=1),
    )
    keep = take
    out_item = jnp.where(take[:, None], out_item, combat.proc_output_item)
    out_quantity = jnp.where(take[:, None], out_quantity, combat.proc_output_quantity)
    combat = combat._replace(
        proc_bench_cell=jnp.where(keep[:, None], position, combat.proc_bench_cell),
        proc_recipe=jnp.where(keep, index, combat.proc_recipe).astype(jnp.int32),
        proc_active=jnp.where(keep, True, combat.proc_active),
        proc_has_fuel=jnp.where(keep, has_fuel, combat.proc_has_fuel),
        proc_wait_ticks=jnp.where(keep, START_WAIT_TICKS, combat.proc_wait_ticks).astype(jnp.int32),
        proc_progress_seconds=jnp.where(keep, jnp.float32(0.0), combat.proc_progress_seconds),
        proc_fuel_item=jnp.where(loads, stack_item, combat.proc_fuel_item).astype(jnp.int32),
        proc_fuel_quantity=jnp.where(loads, combat.proc_fuel_quantity + stack_quantity,
                                     combat.proc_fuel_quantity).astype(jnp.int32),
        proc_fuel_quality=jnp.where(loads, quality, combat.proc_fuel_quality),
        proc_input_item=jnp.where(keep[:, None], inputs_item, combat.proc_input_item).astype(jnp.int32),
        proc_input_quantity=jnp.where(keep[:, None], inputs_quantity, combat.proc_input_quantity).astype(jnp.int32),
        proc_output_item=out_item,
        proc_output_quantity=out_quantity,
    )
    return take, inventory, combat


def advance(crafting_function: Any, state: Any, assets: Any, params: Any) -> Any:
    """One world tick of every bench (rules 3 to 7). Touches only the bench leaves; a step with no live bench runs nothing."""

    env = _env(crafting_function)
    jnp, jax = env["jnp"], env["jax"]
    busy = jnp.any((state.combat.proc_recipe >= 0) & state.combat.proc_active)
    return jax.lax.cond(
        busy,
        lambda current: _advance_benches(env, current, assets, params),
        lambda current: current,
        state,
    )


def _advance_benches(env: dict[str, Any], state: Any, assets: Any, params: Any) -> Any:
    """The body of ``advance``: start wait, fuel, progress, by-products, completion, fuel exhaustion (INFERRED)."""

    jnp = env["jnp"]
    combat = state.combat
    table = assets.packed.table
    dt = jnp.float32(1.0) / jnp.asarray(params.ticks_per_second, dtype=jnp.float32)
    recipe = combat.proc_recipe
    safe = jnp.clip(recipe, 0, table.time_seconds.shape[0] - 1)
    live = (recipe >= 0) & combat.proc_active
    waiting = live & (combat.proc_wait_ticks > 0)
    wait = jnp.where(waiting, combat.proc_wait_ticks - 1, combat.proc_wait_ticks)
    progressing = live & ~waiting
    remaining = jnp.maximum(jnp.take(table.time_seconds, safe) - combat.proc_progress_seconds, jnp.float32(0.0))
    burn = jnp.where(progressing, jnp.minimum(dt, remaining), jnp.float32(0.0))
    fuel_lane = progressing & combat.proc_has_fuel
    # A new stick is taken when the lit one cannot cover this tick (consumeFuelForDuration: one stick per tick here).
    consume = fuel_lane & (combat.proc_fuel_seconds < burn) & (combat.proc_fuel_quantity > 0)
    fuel_seconds = jnp.where(consume, combat.proc_fuel_seconds + combat.proc_fuel_quality, combat.proc_fuel_seconds)
    fuel_quantity = jnp.where(consume, combat.proc_fuel_quantity - 1, combat.proc_fuel_quantity)
    counted = consume & ~jnp.isin(combat.proc_fuel_item, _ids(env, IGNORED_FUEL))
    countdown = jnp.where(counted, combat.proc_extra_countdown - 1, combat.proc_extra_countdown)
    fire = counted & (countdown <= 0)
    countdown = jnp.where(fire, PER_FUEL_ITEMS, countdown)
    # The lit stick burns the tick's share (never more than it holds); progress advances by the same share.
    burn = jnp.where(fuel_lane, jnp.minimum(burn, jnp.maximum(fuel_seconds, jnp.float32(0.0))), burn)
    fuel_seconds = jnp.where(fuel_lane, fuel_seconds - burn, fuel_seconds)
    progress = jnp.where(progressing, combat.proc_progress_seconds + burn, combat.proc_progress_seconds)
    complete = progressing & (jnp.take(table.time_seconds, safe) - progress <= dt * jnp.float32(COMPLETION_SLACK_TICKS))
    out_item, out_quantity = combat.proc_output_item, combat.proc_output_quantity
    for k in range(table.output_item_id.shape[1]):
        mask = complete & table.output_mask[safe, k]
        out_item, out_quantity = _place(jnp, out_item, out_quantity, table.output_item_id[safe, k],
                                        table.output_quantity[safe, k], mask)
    byproduct = jnp.full(recipe.shape, env["semantic_id"](BYPRODUCT), dtype=jnp.int32)
    out_item, out_quantity = _place(jnp, out_item, out_quantity, byproduct, fire.astype(jnp.int32), fire)
    # Fuel exhaustion (INFERRED, unmeasured): no fuel left and the recipe still running: the bench deactivates on this tick.
    exhausted = fuel_lane & (fuel_seconds <= 0.0) & (fuel_quantity == 0) & ~complete
    return state._replace(combat=combat._replace(
        proc_recipe=jnp.where(complete, -1, recipe).astype(jnp.int32),
        proc_active=jnp.where(complete | exhausted, False, combat.proc_active),
        proc_wait_ticks=jnp.where(complete, 0, wait).astype(jnp.int32),
        proc_progress_seconds=jnp.where(complete, jnp.float32(0.0), progress),
        proc_fuel_seconds=fuel_seconds,
        proc_fuel_quantity=fuel_quantity.astype(jnp.int32),
        proc_extra_countdown=countdown.astype(jnp.int32),
        proc_input_item=jnp.where(complete[:, None], env["ABSENT_ID"], combat.proc_input_item).astype(jnp.int32),
        proc_input_quantity=jnp.where(complete[:, None], 0, combat.proc_input_quantity).astype(jnp.int32),
        proc_output_item=out_item,
        proc_output_quantity=out_quantity,
    ))


def executor(original: Any, profile: Any) -> Any:
    """Hook for ``make_region_block_action_surface_executor``: each executor step first advances the benches."""

    if not _enabled(profile):
        return original

    @wraps(original)
    def make_region_block_action_surface_executor(*args, **kwargs):
        built = original(*args, **kwargs)
        assets = kwargs.get("crafting_assets")
        if assets is None:
            return built
        crafting_function = unwrap(original).__globals__["execute_region_fieldcraft"]

        @wraps(built)
        def processing_executor(state, surface_runtime, surface, evidence, keys, params, config):
            # A lit bench changes state on every tick; the environment commits the executor's state only where state_commit is set.
            running = (state.combat.proc_recipe >= 0) & state.combat.proc_active
            advanced = advance(crafting_function, state, assets, params)
            execution = built(advanced, surface_runtime, surface, evidence, keys, params, config)
            return execution._replace(state_commit=execution.state_commit | running)

        return processing_executor

    return make_region_block_action_surface_executor
