"""Timed bench crafts (gen 7, row 11j, crafting oracle stage 3). Switch ``Constructs.crafting_time``.

Measured on the live 0.7.2 bridge (FIDELITY-0.7.2.md section (t), jar 5FE609E2; receipts
``~/.claude/jobs/31947bd7/tmp/phase4k/receipts/fin_*.json``): Armor_Copper_Chest (3.0 s, Armor_Bench tier 1), request at tick
R = 32, seed 570057. The inputs leave the inventory on tick R + 1 (the first progress tick F; progress 0.0350 s on F), and the
output lands at completion: native tick 120 (88 ticks after R, 87 after F); the same recipe completed on tick 122 in the
second run (89 after F). A second request while a job runs queues behind it (queue 1 to 2); with stock, the queued job's
inputs leave on the completion tick of the first, which gives the first output, and its own output follows (210 = 122 + 88).
Walking out of reach (teleport of 9 blocks, the bench window closed on tick R + 11) cancels the job on that tick and puts the
consumed inputs back in their original slots (native player slots 4 and 5). Taking damage does not cancel. An ability request has
no cancel path (inferred, not run). Zero-time crafts (type 0 and fieldcraft) commit on the request tick, as before.

Clock (the port's decision of 2026-10-18 under the owner's 30 Hz ruling of 2026-10-17, recorded here): progress is counted in
seconds at the simulator's exact tick, 1/30 s
(``params.ticks_per_second``). A job's progress starts on its first progress tick F = R + 1 and gains one tick on every world
tick from F on; it completes on the tick its progress reaches ``time_seconds`` (a half-tick tolerance). A 3.0 s recipe therefore
completes on the 90th progress tick: 90 progress ticks of 1/30 s reach 3.0 s, so the output lands on R + 90 (F + 89). The native
87 to 89 ticks after F are
the server's tick jitter (0.0333 to 0.0360 s per tick), not a rule.

State (``CombatState`` leaves, patches ``timed_job_fields`` and ``timed_job_init``; per lane, the controlled actor 0):
``craft_job_recipe`` (int32, -1 idle), ``craft_job_elapsed_seconds`` (float32), ``craft_job_inputs_pending`` (bool, set on the
request tick and cleared on F), ``craft_queued_recipe`` (int32, -1 none; one deep), ``craft_bench_position`` (int32[3], the bench
block cell the job was committed against), ``craft_refund_quantity`` (int32[slots]) and ``craft_refund_item_id`` (int32[slots]),
the inputs each slot gave up, -1 and 0 when idle. The slot count is the default inventory layout's 76 slots (checked against the
state at every call).

Rules, per tick and per lane:

1. Start (executor patch ``timed_executor_start`` at the craft verb, function ``start``): an accepted timed request (its
   recipe's time_seconds > 0) starts a job when none runs, or queues behind the running job when the queue is empty; a request
   with a running job and a queued one is refused. The bench position is the first in-reach bench block candidate on which the
   recipe is legal (``bench_for_recipe``); no such bench refuses the request. The inventory is untouched on the request tick
   (``execute_region_fieldcraft`` hook ``timed_crafting_execution``: a timed craft is accepted but not committed, not finished).
2. Advance (executor factory hook ``timed_crafting_executor``; runs first on every executor step, before the verbs): a
   running job whose actor stands more than ``REACH_BLOCKS`` (7.0) from the bench block centre is cancelled on that tick:
   the refund goes back into its slots, the queue is dropped (native cancelAllCrafting), no progress is made.
3. Otherwise a job whose inputs are pending removes them on F (the removal is ``craft_into_combat_inventory`` with the outputs
   kept out of the inventory) and gains its first tick of progress; every other live job gains one tick of progress.
4. A job whose progress reached its time gives its output (``route_crafting_outputs``) on that tick. A full inventory holds the
   job on the tick (inferred: no native measurement of a full inventory at completion). Then the queued job, if any, removes its
   inputs on the same tick (after the output, as the receipt's slots show) and starts its progress; without its inputs it is
   dropped (inferred: the removal-failure branch of native CraftingManager).
5. A removal that fails on F drops the job and its queue (inferred, not reached by any receipt).

Cost: the advance runs only on a step where some lane has a job, and the bench scan only on a step where a lane with no job
carries an accepted timed request (both are ``lax.cond`` on any lane; under ``vmap`` both branches run). An idle step pays the
predicates only. The zero-time craft transition of ``execute_region_fieldcraft`` is HEAD's per-step cost and is not gated here.

Hooks and patches (``sim/gen7/overlay.py``, row 11j): hook ``timed_crafting_admission`` admits a timed recipe with its time
removed for the zero-time gate of ``_lossless_region_crafting`` (the rest of the admission is the original's); patch
``timed_resolved_time`` keeps the recipe's ``time_seconds`` in the resolved recipe (a literal: a hook there broke the
``crafting_materials`` hook, which reads ``original.__globals__`` directly); hooks ``timed_crafting_execution`` and
``timed_crafting_executor`` as in rules 1 and 2; patch ``timed_executor_import`` imports ``start`` and patch
``timed_executor_start`` calls it at the craft verb. Gen 5 (``instant``) returns every original.

Not modelled (each named where it applies): the reach metric is the Euclidean distance from the actor's position to the centre of
the bench block (the native ``validateWindows`` metric is not measured; 9 blocks cancels, nothing nearer was measured); the queued
request is judged on the inventory at the request (native accepts a second request whose inputs the first job already took, then
drops it on its start: the outcome is the same, the queue event differs); the refund restores the quantity into the slot without
checking what the slot holds now; the bench is one row per bench type (as ``bench_tiers``).
"""

from __future__ import annotations

from dataclasses import replace
from functools import wraps
from inspect import unwrap
from typing import Any

SWITCH = "crafting_time"
IMPLEMENTED = True
INSTANT = "instant"
TIMED_JOB = "timed_job"
#: WindowManager.validateWindows closes a bench window beyond 7.0 blocks (section (t)).
REACH_BLOCKS = 7.0
#: A job completes on the first tick whose progress is within this many ticks of its time (the float32 sum of 1/30 s).
COMPLETION_SLACK_TICKS = 0.5
#: Slots of the default inventory layout (combat/inventory/schema/contract.py DEFAULT_CONTAINER_CAPACITIES, sum 76).
REFUND_SLOTS = 76


def _enabled(profile: Any) -> bool:
    value = profile.constructs.crafting_time
    if value == INSTANT:
        return False
    if value != TIMED_JOB:
        raise ValueError(f"unknown crafting_time switch {value!r}")
    return True


def _env(function: Any) -> dict[str, Any]:
    """The rendered crafting module namespace of a crafting function (the hooks wrap the function, so unwrap first)."""

    return unwrap(function).__globals__


def _adapter(env: dict[str, Any]) -> dict[str, Any]:
    """The inventory adapter's namespace (``craft_into_combat_inventory`` lives there; so do the routing and the result type)."""

    return _env(env["craft_into_combat_inventory"])


def _lanes(mask: Any, value: Any) -> Any:
    return mask.reshape(mask.shape + (1,) * (value.ndim - mask.ndim))


def _select(jnp: Any, mask: Any, new: Any, old: Any) -> Any:
    return jnp.where(_lanes(mask, new), new, old)


def _select_tree(env: dict[str, Any], mask: Any, new: Any, old: Any) -> Any:
    """``_select`` over every leaf of a state tuple (an InventoryState)."""

    return env["jax"].tree.map(lambda fresh, stale: _select(env["jnp"], mask, fresh, stale), new, old)


def _check_slots(combat: Any, inventory: Any) -> None:
    slots = inventory.item_id.shape[-1]
    if combat.craft_refund_quantity.shape[-1] != slots:
        raise ValueError(
            f"timed crafting refund leaves hold {combat.craft_refund_quantity.shape[-1]} slots, the inventory has {slots}"
        )


def processing_recipe(jnp: Any, table: Any, index: Any) -> Any:
    """Lanes whose recipe requires a processing bench (requirement bench type 1). processing_benches.py owns those recipes.

    The 0.7.2 catalog has no recipe that mixes a processing requirement with another bench type (manifest, 2026-10-18), so one
    type-1 requirement classifies the recipe.
    """

    safe = jnp.clip(index, 0, table.requirement_mask.shape[0] - 1)
    rows = jnp.take(table.requirement_mask & (table.requirement_bench_type == 1), safe, axis=0)
    return (index >= 0) & jnp.any(rows, axis=-1)


def timed_recipe(jnp: Any, table: Any, index: Any) -> Any:
    """Lanes whose requested recipe has a positive time and is not a processing recipe (``index`` -1 is not a recipe)."""

    safe = jnp.clip(index, 0, table.time_seconds.shape[0] - 1)
    return (index >= 0) & (jnp.take(table.time_seconds, safe) > 0.0) & ~processing_recipe(jnp, table, index)


def admission(original: Any, profile: Any) -> Any:
    """Hook for ``_lossless_region_crafting``: a timed recipe passes the zero-time gate with its time removed."""

    if not _enabled(profile):
        return original

    @wraps(original)
    def _lossless_region_crafting(recipe, admitted_bench_ids):
        if recipe.time_seconds > 0.0:
            return original(replace(recipe, time_seconds=0.0), admitted_bench_ids)
        return original(recipe, admitted_bench_ids)

    return _lossless_region_crafting


def execution(original: Any, profile: Any) -> Any:
    """Hook for ``execute_region_fieldcraft``: an accepted timed craft is not committed (the job starts in ``start``)."""

    if not _enabled(profile):
        return original

    @wraps(original)
    def execute_region_fieldcraft(assets, state, candidates, candidate_index):
        result = original(assets, state, candidates, candidate_index)
        env = _env(original)
        jnp, jax = env["jnp"], env["jax"]
        timed = timed_recipe(jnp, assets.packed.table, result.recipe_index)
        held = result.accepted & timed
        inventory = jax.tree.map(lambda new, old: _select(jnp, held, old, new), result.state, state)
        return result._replace(state=inventory, finished=result.accepted & ~timed)

    return execute_region_fieldcraft


def executor(original: Any, profile: Any) -> Any:
    """Hook for ``make_region_block_action_surface_executor``: each executor step first advances the running jobs."""

    if not _enabled(profile):
        return original

    @wraps(original)
    def make_region_block_action_surface_executor(*args, **kwargs):
        built = original(*args, **kwargs)
        assets = kwargs.get("crafting_assets")
        if assets is None:
            return built
        layout = kwargs["inventory_layout"]
        # Unwrap first: the processing and diagram hooks wrap this factory before it is rendered, so
        # ``original`` can be their wrapper, whose globals are the processing module's (no execute_region_fieldcraft).
        namespace = unwrap(original).__globals__
        crafting_function = namespace["execute_region_fieldcraft"]
        agent = namespace["AGENT_ENTITY"]

        @wraps(built)
        def timed_executor(state, surface_runtime, surface, evidence, keys, params, config):
            # The environment commits the executor's state only where state_commit is set (environment.py, the commit gate),
            # and a running job changes state on every tick: so a lane with a job sets it.
            running = state.combat.craft_job_recipe >= 0
            advanced = advance(crafting_function, state, assets, layout, params, agent)
            execution = built(advanced, surface_runtime, surface, evidence, keys, params, config)
            return execution._replace(state_commit=execution.state_commit | running)

        return timed_executor

    return make_region_block_action_surface_executor


def bench_for_recipe(crafting_function: Any, assets: Any, layout: Any, candidates: Any, world: Any, inventory: Any, index: Any):
    """The bench block cell (int32[batch, 3]) of the first in-reach bench candidate on which each lane's recipe is legal,
    and whether one exists. ``candidates``: the executor's block candidates; ``world``: its ``RegionActionRuntimeState``."""

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
    usable = per_candidate & bench.window_open[:, 0] & (index >= 0)[:, None]
    first = jnp.argmax(usable, axis=1).astype(jnp.int32)
    found = jnp.any(usable, axis=1)
    visible = candidates.visible_position[:, 0]
    position = jnp.take_along_axis(visible, jnp.broadcast_to(first[:, None, None], (batch, 1, 3)), axis=1)[:, 0]
    return jnp.where(found[:, None], position, 0).astype(jnp.int32), found


def start(crafting_function: Any, state: Any, craft: Any, candidates: Any, world: Any, assets: Any, layout: Any):
    """Executor patch at the craft verb: the bench is the first in-reach candidate that makes the recipe legal, then ``start_at``.

    The scan runs only when a lane with no job running carries an accepted timed request (``lax.cond`` on any lane); any other
    step skips it at runtime, and the position and flag it does not compute are zero and false.
    """

    env = _env(crafting_function)
    jnp, jax = env["jnp"], env["jax"]
    index = craft.recipe_index
    timed = craft.accepted & timed_recipe(jnp, assets.packed.table, index)
    scan = jnp.any(timed & (state.combat.craft_job_recipe < 0))
    batch = index.shape[0]
    position, found = jax.lax.cond(
        scan,
        lambda: bench_for_recipe(crafting_function, assets, layout, candidates, world, state.inventory, index),
        lambda: (jnp.zeros((batch, 3), dtype=jnp.int32), jnp.zeros((batch,), dtype=jnp.bool_)),
    )
    return start_at(crafting_function, state, craft, position, found, assets)


def start_at(crafting_function: Any, state: Any, craft: Any, position: Any, found: Any, assets: Any):
    """An accepted timed craft starts a job, or queues behind a running one; ``position``/``found`` name its bench.

    Returns ``(accepted, state)``: the craft's acceptance after the queue rule, and the state with the job leaves set. Zero-time
    lanes keep their acceptance and their state.
    """

    env = _env(crafting_function)
    jnp = env["jnp"]
    combat = state.combat
    _check_slots(combat, state.inventory)
    index = craft.recipe_index
    timed = craft.accepted & timed_recipe(jnp, assets.packed.table, index)
    running = combat.craft_job_recipe >= 0
    queued = combat.craft_queued_recipe >= 0
    start_job = timed & ~running & found
    queue_job = timed & running & ~queued
    accepted = jnp.where(timed, start_job | queue_job, craft.accepted)
    combat = combat._replace(
        craft_job_recipe=jnp.where(start_job, index, combat.craft_job_recipe),
        craft_job_elapsed_seconds=jnp.where(start_job, jnp.float32(0.0), combat.craft_job_elapsed_seconds),
        craft_job_inputs_pending=jnp.where(start_job, True, combat.craft_job_inputs_pending),
        craft_queued_recipe=jnp.where(queue_job, index, combat.craft_queued_recipe),
        craft_bench_position=_select(jnp, start_job, position, combat.craft_bench_position),
        craft_refund_quantity=jnp.where(start_job[:, None], jnp.int32(0), combat.craft_refund_quantity),
        craft_refund_item_id=jnp.where(start_job[:, None], env["ABSENT_ID"], combat.craft_refund_item_id),
    )
    return accepted, state._replace(combat=combat)


def _catalog(env: dict[str, Any], assets: Any, index: Any, layout: Any, *, allowed: bool) -> Any:
    """The commit catalog of one recipe; ``allowed`` False keeps every output out of the inventory (removal only)."""

    jnp = env["jnp"]
    slots = layout.container_id.shape[0]
    batch = index.shape[0]
    safe = jnp.clip(index, 0, assets.output_max_stack.shape[0] - 1)
    priority = jnp.asarray(
        (env["CONTAINER_HOTBAR"], env["CONTAINER_STORAGE"], env["CONTAINER_BACKPACK"]), dtype=jnp.int32
    )
    if allowed:
        open_slots = jnp.isin(layout.container_id, priority)
    else:
        open_slots = jnp.zeros((slots,), dtype=jnp.bool_)
    return env["CombatCraftingCatalog"](
        slot_resource_type_id=jnp.full((batch, 1, slots, env["MAX_RESOURCE_TYPES_PER_ITEM"]), env["ABSENT_ID"], dtype=jnp.int32),
        output_max_stack=assets.output_max_stack[safe][:, None, :],
        output_max_durability=assets.output_max_durability[safe][:, None, :],
        output_container_priority=jnp.broadcast_to(priority[None, None, None, :], (batch, 1, env["MAX_OUTPUTS"], 3)),
        output_slot_allowed=jnp.broadcast_to(open_slots[None, None, None, :], (batch, 1, env["MAX_OUTPUTS"], slots)),
    )


def _remove(env: dict[str, Any], assets: Any, layout: Any, actor: Any, index: Any):
    """One craft's inputs leave the actor's inventory; the outputs stay out. Returns (actor, crafted, removed, removed item)."""

    jnp = env["jnp"]
    batch = index.shape[0]
    transition = env["craft_into_combat_inventory"](
        assets.packed.table,
        actor,
        layout,
        env["_fieldcraft_context"](batch, 1, assets.memories_level),
        index[:, None],
        _catalog(env, assets, index, layout, allowed=False),
        quantity=jnp.ones((batch, 1), dtype=jnp.int32),
        input_slot_order=env["player_crafting_input_slot_order"](layout),
    )
    crafted = transition.crafting.crafted[:, 0] & (index >= 0)
    removed = jnp.maximum(actor.quantity[:, 0] - transition.state.quantity[:, 0], 0)
    removed = jnp.where(crafted[:, None], removed, 0)
    item = jnp.where(removed > 0, actor.item_id[:, 0], env["ABSENT_ID"])
    return transition.state, crafted, removed, item


def _outputs(env: dict[str, Any], assets: Any, index: Any, actor: Any, layout: Any) -> Any:
    """The completed recipe's outputs, read from the table (they do not depend on the inventory)."""

    jnp = env["jnp"]
    table = assets.packed.table
    batch = index.shape[0]
    safe = jnp.clip(index, 0, table.recipe_mask.shape[0] - 1)
    mask = table.output_mask[safe]
    return _adapter(env)["CraftingStepResult"](
        inventory=env["_crafting_inventory"](actor, layout),
        crafted=jnp.ones((batch, 1), dtype=jnp.bool_),
        failure_bits=jnp.zeros((batch, 1), dtype=jnp.uint32),
        recipe_id_hash=table.recipe_id_hash[safe][:, None],
        output_item_id=jnp.where(mask, table.output_item_id[safe], env["ABSENT_ID"])[:, None],
        output_quantity=jnp.where(mask, table.output_quantity[safe], 0)[:, None],
        output_metadata_hash=jnp.where(mask[..., None], table.output_metadata_hash[safe], 0)[:, None],
        time_seconds=table.time_seconds[safe][:, None],
    )


def _restore(jnp: Any, actor: Any, refund_quantity: Any, refund_item: Any, mask: Any) -> Any:
    """Put a cancelled job's consumed quantities back into the slots they came from."""

    give = (refund_quantity > 0) & mask[:, None]
    quantity = jnp.where(give, actor.quantity[:, 0] + refund_quantity, actor.quantity[:, 0])
    item = jnp.where(give, refund_item, actor.item_id[:, 0])
    return actor._replace(item_id=item[:, None], quantity=quantity[:, None])


def advance(crafting_function: Any, state: Any, assets: Any, layout: Any, params: Any, agent: int) -> Any:
    """One world tick of every job (rules 2 to 4). ``state`` carries ``combat`` and ``inventory``; idle lanes are unchanged.

    Skipped at runtime (``lax.cond``) on a step where no lane has a job, so an idle step runs neither the craft transition nor
    the completion routing.
    """

    env = _env(crafting_function)
    _check_slots(state.combat, state.inventory)
    busy = env["jnp"].any(state.combat.craft_job_recipe >= 0)
    return env["jax"].lax.cond(
        busy,
        lambda current: _advance_jobs(env, current, assets, layout, params, agent),
        lambda current: current,
        state,
    )


def _advance_jobs(env: dict[str, Any], state: Any, assets: Any, layout: Any, params: Any, agent: int) -> Any:
    """The body of ``advance``: rules 2 to 4 for every lane (run only on a step with a job)."""

    jnp = env["jnp"]
    combat, inventory = state.combat, state.inventory
    table = assets.packed.table
    dt = jnp.float32(1.0) / jnp.asarray(params.ticks_per_second, dtype=jnp.float32)
    job = combat.craft_job_recipe
    active = job >= 0
    center = combat.craft_bench_position.astype(jnp.float32) + jnp.float32(0.5)
    distance = jnp.linalg.norm(combat.position[:, agent] - center, axis=-1)
    out = active & (distance > jnp.float32(REACH_BLOCKS))
    actor = _restore(jnp, env["_actor_inventory"](inventory), combat.craft_refund_quantity,
                     combat.craft_refund_item_id, out)
    refund_q = jnp.where(out[:, None], jnp.int32(0), combat.craft_refund_quantity)
    refund_i = jnp.where(out[:, None], env["ABSENT_ID"], combat.craft_refund_item_id)
    job = jnp.where(out, -1, job)
    queued = jnp.where(out, -1, combat.craft_queued_recipe)
    pending = jnp.where(out, False, combat.craft_job_inputs_pending)
    elapsed = jnp.where(out, jnp.float32(0.0), combat.craft_job_elapsed_seconds)
    live = active & ~out

    # Rule 3: the first progress tick removes the inputs and counts one tick.
    removed_actor, crafted, took_q, took_i = _remove(env, assets, layout, actor, job)
    first = live & pending
    take = first & crafted
    drop = first & ~crafted
    actor = _select_tree(env, take, removed_actor, actor)
    refund_q = jnp.where(take[:, None], took_q, refund_q)
    refund_i = jnp.where(take[:, None], took_i, refund_i)
    pending = jnp.where(first, False, pending)
    job = jnp.where(drop, -1, job)
    queued = jnp.where(drop, -1, queued)
    progressing = live & ~drop
    elapsed = jnp.where(progressing, elapsed + dt, elapsed)

    # Rule 4: completion gives the output; a full inventory holds the job.
    safe = jnp.clip(job, 0, table.recipe_mask.shape[0] - 1)
    due = progressing & (job >= 0) & (elapsed >= jnp.take(table.time_seconds, safe) - dt * jnp.float32(COMPLETION_SLACK_TICKS))
    routing = _adapter(env)["route_crafting_outputs"](
        actor, layout, _outputs(env, assets, job, actor, layout), _catalog(env, assets, job, layout, allowed=True)
    )
    fits = ~jnp.any(routing.world_drop_mask[:, 0], axis=1)
    complete = due & fits
    actor = _select_tree(env, complete, routing.state, actor)
    starts = complete & (queued >= 0)
    queued_actor, queued_crafted, queued_q, queued_i = _remove(env, assets, layout, actor, queued)
    take_q = starts & queued_crafted
    actor = _select_tree(env, take_q, queued_actor, actor)
    refund_q = jnp.where(complete[:, None], jnp.where(take_q[:, None], queued_q, 0), refund_q)
    refund_i = jnp.where(complete[:, None], jnp.where(take_q[:, None], queued_i, env["ABSENT_ID"]), refund_i)
    job = jnp.where(complete, jnp.where(take_q, queued, -1), job)
    queued = jnp.where(complete, -1, queued)
    elapsed = jnp.where(complete, jnp.where(take_q, dt, jnp.float32(0.0)), elapsed)

    combat = combat._replace(
        craft_job_recipe=job.astype(jnp.int32),
        craft_job_elapsed_seconds=elapsed,
        craft_job_inputs_pending=pending,
        craft_queued_recipe=queued.astype(jnp.int32),
        craft_bench_position=jnp.where((job >= 0)[:, None], combat.craft_bench_position, jnp.int32(0)),
        craft_refund_quantity=refund_q,
        craft_refund_item_id=refund_i,
    )
    return state._replace(combat=combat, inventory=env["_merge_actor_inventory"](inventory, actor))

