"""Diagram crafting (gen 7, row 11q, crafting oracle stage 5). Switch ``Constructs.diagram_crafting``.

Measured on the live 0.7.2 bridge (FIDELITY-0.7.2.md section (v), jar 4A5E5D6D; receipts
``~/.claude/jobs/31947bd7/tmp/phase4l/receipts/v6_diag_axe_armory_seed{570057,31337}.json``): Armory (Bench.Type DiagramCrafting,
tier 1), Weapon_Axe_Stone_Trork (recipe time 0.0 s) through CraftItemAction in the Weapons / Axe category, its three inputs in
window slots 0 to 2 (capacity 4). The request is at R = 39 (seed 570057) and 37 (seed 31337): the queue goes from 0 to 1 on R,
the inputs leave the window and the axe reaches inventory slot 5 on R + 2 (41 and 39).

Labels: CONFIRMED (receipt or bytecode line) or INFERRED (named where it applies).

1. Click craft (CONFIRMED): a zero-time recipe that an in-reach Armory (bench type 2) makes legal is queued on its request tick R and
   commits on R + 2: its inputs leave the actor's inventory and its output enters it. The bytecode route is DiagramCraftingWindow
   CraftItemAction -> queueCraft(ORDERED) (crafting_javap_072.txt, DiagramCrafting window). The category is implicit in the request:
   the simulator has no window action and the receipts show no category cost.
2. Queue (CONFIRMED one entry per craft; INFERRED depth): the receipts show queue_before 0 and queue_after 1. The simulator queues
   one craft per lane; a request while one is queued is refused. Unmeasured: a second request.
3. Scope (CONFIRMED from the manifest, 2026-10-18): the 19 bench-type-2 recipes all also name a crafting bench (type 0, Weapon_Bench
   or Workbench). Two are zero-time (Weapon_Axe_Stone_Trork, Weapon_Shortbow_Crude) and take this construct; the other 17 carry a
   time (3.0 to 4.5 s) and stay with timed_crafting (a timed craft, not measured at the Armory).
4. Bench choice (INFERRED): when an in-reach Armory makes the recipe legal, the queued craft is the Armory's. Otherwise the zero-time
   craft commits on its request tick as before (the crafting bench path, unchanged).
5. Commit (INFERRED): the commit removes the recipe's inputs and routes its outputs (the timed construct's completion routing). If the
   output does not fit the inventory the commit waits on the tick (not measured); if an input is gone at R + 2 the queue is dropped.

State (``CombatState`` leaves, patches ``diagram_fields`` and ``diagram_init``): ``diagram_recipe`` (int32, -1 idle) and
``diagram_ticks`` (int32, ticks until the commit: 2 on R, one less each world tick; the commit runs on the tick it reaches 0).

Hooks and patches (``sim/gen7/overlay.py``, row 11q): the leaves; patch ``diagram_bench_types`` (``_BENCH_TYPE_NAMES`` maps
DiagramCrafting to 2, so the Armory row exists) and hook ``diagram_admission`` (maps a type-2 requirement to type 0 for the
zero-time gate only; the bench candidate matching sees the real type); the executor
patch ``diagram_executor_start`` (after the processing start: a diagram request is queued, and the zero-time commit that
``execute_region_fieldcraft`` made is undone); hook ``diagram_executor`` (each step first runs the pending commit). Gen 5
(``absent``) returns every original.

Not modelled (named): a negative control (category not selected, wrong input count, the Weapon_Bench second requirement, which the
receipts did not test); the Armory window's own output slot (the probe showed the axe there before the craft, and the commit does not
read it); the 17 timed diagram recipes (timed_crafting); a second request while one is queued (refused, inferred).
"""

from __future__ import annotations

from dataclasses import replace
from functools import wraps
from inspect import unwrap
from typing import Any

from hytalegym.sim.gen7.mechanics.processing_benches import bench_of_type
from hytalegym.sim.gen7.mechanics.timed_crafting import (
    _adapter,
    _catalog,
    _env,
    _outputs,
    _remove,
    _select_tree,
    processing_recipe,
)

SWITCH = "diagram_crafting"
IMPLEMENTED = True
ABSENT = "absent"
CLICK_CRAFT = "click_craft"
#: Bench.Type DiagramCrafting (crafting/contract.py BENCH_TYPE_DIAGRAM_CRAFTING).
BENCH_TYPE_DIAGRAM_CRAFTING = 2
#: Ticks from the request to the commit (receipts: queued on R, inputs out and output in on R + 2).
COMMIT_TICKS = 2


def admission(original: Any, profile: Any) -> Any:
    """Hook for ``_lossless_region_crafting``: a diagram (bench type 2) requirement passes the zero-time gate as an ordinary bench requirement.

    Exactly one gate sees the mapped type: the zero-time gate (the compile's admission), which sees type 0 for a type-2 requirement
    and still requires the bench id to be admitted (the Armory row is admitted by patch ``diagram_bench_types``). The packed table
    keeps the recipe's own type. The bench candidate matching is unchanged (``_bench_contexts``, ``_legal_recipes`` and
    ``_recipe_precondition`` see the real type 2 of an Armory context). The 19 type-2 rows also name a type-0 bench, so the mapping
    changes no admission on the 0.7.2 catalog. The timed hook keeps a timed recipe's time.
    """

    if not _enabled(profile):
        return original

    @wraps(original)
    def _lossless_region_crafting(recipe, admitted_bench_ids):
        requirements = tuple(
            replace(requirement, bench_type=0) if requirement.bench_type == BENCH_TYPE_DIAGRAM_CRAFTING else requirement
            for requirement in recipe.bench_requirements
        )
        return original(replace(recipe, bench_requirements=requirements), admitted_bench_ids)

    return _lossless_region_crafting


def _enabled(profile: Any) -> bool:
    value = profile.constructs.diagram_crafting
    if value == ABSENT:
        return False
    if value != CLICK_CRAFT:
        raise ValueError(f"unknown diagram_crafting switch {value!r}")
    return True


def zero_time_lane(jnp: Any, table: Any, index: Any) -> Any:
    """Lanes whose recipe takes no time and is not a processing recipe (the diagram craft's candidates)."""

    safe = jnp.clip(index, 0, table.time_seconds.shape[0] - 1)
    return (index >= 0) & (jnp.take(table.time_seconds, safe) <= 0.0) & ~processing_recipe(jnp, table, index)


def start(crafting_function: Any, state: Any, craft: Any, inventory_before: Any, candidates: Any, world: Any, assets: Any,
          layout: Any):
    """Executor patch after the processing start: a zero-time craft that an in-reach Armory makes legal is queued, not committed.

    ``state.inventory`` holds the zero-time commit that ``execute_region_fieldcraft`` made; a queued craft restores
    ``inventory_before``. Returns ``(accepted, state)``; lanes that are not diagram crafts pass through unchanged.
    """

    env = _env(crafting_function)
    jnp, jax = env["jnp"], env["jax"]
    index = craft.recipe_index
    wanted = craft.accepted & zero_time_lane(jnp, assets.packed.table, index)
    batch = index.shape[0]
    _position, found, _words = jax.lax.cond(
        jnp.any(wanted),
        lambda: bench_of_type(crafting_function, assets, layout, candidates, world, inventory_before, index,
                              BENCH_TYPE_DIAGRAM_CRAFTING),
        lambda: (jnp.zeros((batch, 3), dtype=jnp.int32), jnp.zeros((batch,), dtype=jnp.bool_),
                 jnp.zeros((batch, 8), dtype=jnp.uint32)),
    )
    return queue_at(crafting_function, state, craft, inventory_before, assets, found)


def queue_at(crafting_function: Any, state: Any, craft: Any, inventory_before: Any, assets: Any, found: Any):
    """The queue rule, given whether an in-reach Armory makes each lane's zero-time recipe legal (``found``).

    A queued lane takes ``inventory_before`` and two ticks of countdown; a lane whose queue is busy is refused with the same
    inventory; other lanes pass through. Returns ``(accepted, state)``.
    """

    env = _env(crafting_function)
    jnp = env["jnp"]
    index = craft.recipe_index
    diagram = craft.accepted & zero_time_lane(jnp, assets.packed.table, index) & found
    idle = state.combat.diagram_recipe < 0
    queued = diagram & idle
    combat = state.combat._replace(
        diagram_recipe=jnp.where(queued, index, state.combat.diagram_recipe).astype(jnp.int32),
        diagram_ticks=jnp.where(queued, COMMIT_TICKS, state.combat.diagram_ticks).astype(jnp.int32),
    )
    accepted = jnp.where(diagram, queued, craft.accepted)
    inventory = _select_tree(env, diagram, inventory_before, state.inventory)
    return accepted, state._replace(inventory=inventory, combat=combat)


def advance(crafting_function: Any, state: Any, assets: Any, layout: Any) -> Any:
    """One world tick of the pending diagram commit (rule 5). A step with no queued craft runs nothing."""

    env = _env(crafting_function)
    jnp, jax = env["jnp"], env["jax"]
    busy = jnp.any(state.combat.diagram_recipe >= 0)
    return jax.lax.cond(
        busy,
        lambda current: _commit_due(env, current, assets, layout),
        lambda current: current,
        state,
    )


def _commit_due(env: dict[str, Any], state: Any, assets: Any, layout: Any) -> Any:
    """Count the queued craft down; on the tick it reaches zero, remove the inputs and route the output into the inventory."""

    jnp = env["jnp"]
    combat = state.combat
    recipe = combat.diagram_recipe
    pending = recipe >= 0
    ticks = jnp.where(pending, combat.diagram_ticks - 1, combat.diagram_ticks)
    due = pending & (ticks <= 0)
    actor = env["_actor_inventory"](state.inventory)
    removed, crafted, _, _ = _remove(env, assets, layout, actor, recipe)
    safe_recipe = jnp.where(due, recipe, -1)
    routing = _adapter(env)["route_crafting_outputs"](
        removed, layout, _outputs(env, assets, safe_recipe, removed, layout),
        _catalog(env, assets, safe_recipe, layout, allowed=True),
    )
    fits = ~jnp.any(routing.world_drop_mask[:, 0], axis=1)
    commit = due & crafted & fits
    actor = _select_tree(env, commit, routing.state, actor)
    done = commit | (due & ~crafted)
    return state._replace(
        inventory=env["_merge_actor_inventory"](state.inventory, actor),
        combat=combat._replace(
            diagram_recipe=jnp.where(done, -1, recipe).astype(jnp.int32),
            diagram_ticks=jnp.where(done, 0, ticks).astype(jnp.int32),
        ),
    )


def executor(original: Any, profile: Any) -> Any:
    """Hook for ``make_region_block_action_surface_executor``: each executor step first runs the pending commit."""

    if not _enabled(profile):
        return original

    @wraps(original)
    def make_region_block_action_surface_executor(*args, **kwargs):
        built = original(*args, **kwargs)
        assets = kwargs.get("crafting_assets")
        if assets is None:
            return built
        layout = kwargs["inventory_layout"]
        crafting_function = unwrap(original).__globals__["execute_region_fieldcraft"]

        @wraps(built)
        def diagram_executor(state, surface_runtime, surface, evidence, keys, params, config):
            advanced = advance(crafting_function, state, assets, layout)
            return built(advanced, surface_runtime, surface, evidence, keys, params, config)

        return diagram_executor

    return make_region_block_action_surface_executor
