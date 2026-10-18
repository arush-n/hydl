"""Bench and identity workloads for a rendered simulator package.

    python -m hytalegym.sim.common.bench --rendered DIR [--original head|worktree]
        [--receipts DIR] [--skip-profile] [--skip-region]

Measures the original gen-5 engine and the render at DIR, each workload in a
fresh worker process (cold JIT, persistent compilation cache off, CPU):

* ``compile``: jit compile time of the combat step (``step_arsenal_batch``,
  iron sword against the Trork, batch 64, one step per 30 Hz game tick);
* ``throughput``: lane-steps per second at batch 8 and 64 (a jitted
  ``rollout_arsenal_batch`` of 64 steps, median of repeats);
* ``realtime``: the deployed per-tick path at batch 1 with 1 and 8 NPCs (engine
  step plus the learner observation-v3 encode; the action decode, the policy
  forward pass and bridge I/O are not included): p50/p95/max wall time of one
  step against the 33.3 ms tick budget (target p95 <= 15 ms) and the real-time
  factor, simulated seconds (the step clock the engine actually used) per wall
  second;
* ``region``: load ``region-library-pilot-v2`` and its traversal graphs and
  build a working-set atlas with its path table (read-only).

Then a profile of the rendered per-tick path (``jax.profiler`` trace; thunk time
attributed to engine source lines through the compiled HLO's stack frames). A
table is printed and receipts written under ``--receipts`` (default: a
``render_bench`` directory next to DIR).

By default the original is the render's own source commit (``git archive`` of
the manifest's commit into the receipts directory) and both variants run on
that tree, so the comparison measures rendering alone; ``--original worktree``
runs both on the working tree instead.

``identity_rollouts(package)`` are the fixed-seed rollouts the existing combat
tests run; ``python -P bench.py identity --rendered DIR`` (on a tree whose
``hytalegym.jax`` is the render's source commit) compares them bit for bit.
This module imports only the standard library at import time so it can run as
a worker in a HEAD-snapshot process.
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import importlib.util
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
SOURCE_PACKAGE = "hytalegym.jax"
TICK_BUDGET_MS = 1000.0 / 30.0
P95_TARGET_MS = 15.0
REGION_MANIFEST = REPO_ROOT / "HytaleRL/artifacts/worldgen/region-library-pilot-v2/manifest.json"
TRAVERSAL_MANIFEST = REGION_MANIFEST.parent / "traversal-v2" / "traversal-manifest.json"
_HLO_TABLES = ("FileNames", "FunctionNames", "FileLocations", "StackFrames")


def hlo_fingerprint(text: str) -> str:
    """sha256 of compiled HLO text without source metadata (file paths, lines,
    op names) or the stack-frame tables: equal for the same program."""

    import hashlib

    kept, section = [], None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in _HLO_TABLES:
            section = stripped
            continue
        if section and re.match(r"^\d+ ", stripped):
            continue
        section = None
        kept.append(re.sub(r",?\s*metadata=\{[^}]*\}", "", line))
    return hashlib.sha256("\n".join(kept).encode()).hexdigest()


def _load_render():
    """``render.py`` as a standalone module (``hytalegym.sim`` may not exist
    in the process, e.g. on a HEAD snapshot)."""

    name = "_hytalerl_sim_render"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "render.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def package_for(variant: str, rendered: str | Path | None) -> str:
    if variant == "original":
        return SOURCE_PACKAGE
    render = _load_render()
    render.mount(rendered)
    return render.read_manifest(rendered)["package"]


def _engine(package: str):
    combat = importlib.import_module(f"{package}.combat")
    arsenal = importlib.import_module(f"{package}.combat.arsenal")
    v3 = importlib.import_module(f"{package}.combat.observation.v3")
    return combat, arsenal, v3


def _quiet_jax():
    import jax

    jax.config.update("jax_enable_compilation_cache", False)
    return jax


# --------------------------------------------------------------------------
# Workloads shared by the identity gate and the bench


def _seeded_actions(jax, jnp, combat, steps: int, batch: int, seed: int = 314159):
    """The action stream of the null-arsenal certified-melee test."""

    random = jax.random.uniform(
        jax.random.key(seed), (steps, batch, combat.ACTION_SIZE), minval=-1.0, maxval=1.0
    )
    actions = jnp.zeros_like(random)
    actions = actions.at[..., :6].set((random[..., :6] > 0.35).astype(jnp.float32))
    actions = actions.at[..., 6:].set(random[..., 6:] * 20.0)
    return actions.at[::7, :, combat.ACTION_ATTACK].set(1.0)


def _broadcast(jax, jnp, tree, steps: int):
    return jax.tree_util.tree_map(lambda v: jnp.broadcast_to(v, (steps,) + v.shape), tree)


def _all_clear_capabilities(jnp, arsenal, batch: int, entities: int = 2):
    """``_capabilities(batch)`` of tests/jax/combat/arsenal/tests/environment/
    test_observation_and_runtime.py: every world capability available."""

    world = arsenal.empty_arsenal_commands(batch, entity_count=entities).world
    clear = jnp.full((batch, entities), True, dtype=jnp.bool_)
    candidate = jnp.full((batch, entities, entities), True, dtype=jnp.bool_)
    return world._replace(
        target_candidate_perceptible=candidate,
        target_candidate_perception_valid=candidate,
        line_of_sight=clear,
        line_of_sight_valid=clear,
        selector_line_of_sight=clear,
        selector_line_of_sight_valid=clear,
        direct_target_selected=clear,
        direct_target_selection_valid=clear,
        muzzle_position=jnp.zeros((batch, entities, 3), dtype=jnp.float32),
        muzzle_yaw_degrees=jnp.zeros((batch, entities), dtype=jnp.float32),
        muzzle_pitch_degrees=jnp.zeros((batch, entities), dtype=jnp.float32),
        muzzle_valid=clear,
        clear_projectile_flight=clear,
        projectile_world_collision_available=jnp.zeros_like(clear),
        clear_force_path=jnp.full_like(world.clear_force_path, True),
        dodge_corridor_clear=jnp.full((batch, entities, 4), True, dtype=jnp.bool_),
        entity_only_area=clear,
        static_area_placement=clear,
        area_center=jnp.zeros((batch, entities, 3), dtype=jnp.float32),
    )


def identity_rollouts(package: str) -> dict:
    """Fixed-seed rollouts the existing combat tests run, through ``package``.

    * ``melee_core`` and ``arsenal_null``: test_observation_and_runtime.py::
      test_null_arsenal_is_bit_exact_certified_melee_across_seeded_rollout
      (48 steps, batch 4, microticks 4, reset seeds 570057/913771/173205/271828,
      action key 314159, step key 161803);
    * ``arsenal_sword_bombs``: ::test_complete_arsenal_rollout_scan_matches_eager_jax
      (iron sword and bombs, 2 steps, batch 2, keys 70/71), jitted and eager.
    """

    jax = _quiet_jax()
    import jax.numpy as jnp

    combat, arsenal, _ = _engine(package)
    out: dict = {}
    steps, batch = 48, 4
    params = combat.default_combat_params(microticks=4, target_active=True)
    reset_keys = jnp.stack(tuple(jax.random.key(s) for s in (570057, 913771, 173205, 271828)))
    base_state, base_observation = combat.reset_batch(reset_keys, params)
    config = arsenal.arsenal_runtime_config(arsenal.empty_ability_loadout(batch))
    arsenal_state, arsenal_observation = arsenal.reset_arsenal_batch(reset_keys, params, config)
    actions = _seeded_actions(jax, jnp, combat, steps, batch)
    step_keys = jax.random.split(jax.random.key(161803), steps * batch).reshape((steps, batch))
    commands = _broadcast(jax, jnp, arsenal.empty_arsenal_commands(batch), steps)
    out["params.microticks4"] = params
    out["melee_core.reset"] = (base_state, base_observation)
    out["melee_core.rollout"] = jax.jit(combat.rollout_batch)(base_state, actions, step_keys, params)
    out["arsenal_null.config"] = config
    out["arsenal_null.reset"] = (arsenal_state, arsenal_observation)
    out["arsenal_null.rollout"] = jax.jit(arsenal.rollout_arsenal_batch)(
        arsenal_state, actions, step_keys, params, commands, config
    )

    steps, batch = 2, 2
    params = combat.default_combat_params(microticks=1, target_active=False)
    config = arsenal.arsenal_runtime_config(arsenal.hytale_0_5_7_loadouts(("iron_sword", "bombs")))
    state, observation = arsenal.reset_arsenal_batch(jax.random.split(jax.random.key(70), batch), params, config)
    actions = jnp.zeros((steps, batch, combat.ACTION_SIZE), dtype=jnp.float32)
    keys = jax.random.split(jax.random.key(71), steps * batch).reshape(steps, batch)
    command = arsenal.empty_arsenal_commands(batch)._replace(
        ability_slot=jnp.stack(
            (jnp.zeros((batch,), dtype=jnp.int32), jnp.full((batch,), -1, dtype=jnp.int32)), axis=1
        ),
        world=_all_clear_capabilities(jnp, arsenal, batch),
    )
    commands = _broadcast(jax, jnp, command, steps)
    out["params.microticks1"] = params
    out["arsenal_sword_bombs.config"] = config
    out["arsenal_sword_bombs.reset"] = (state, observation)
    out["arsenal_sword_bombs.rollout_jit"] = jax.jit(arsenal.rollout_arsenal_batch)(
        state, actions, keys, params, commands, config
    )
    out["arsenal_sword_bombs.rollout_eager"] = arsenal.rollout_arsenal_batch(
        state, actions, keys, params, commands, config
    )
    return out


def tree_arrays(tree) -> list[tuple[str, object]]:
    """``(path, numpy array)`` for every leaf; PRNG keys as their key data."""

    import jax
    import numpy as np

    leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
    arrays = []
    for path, leaf in leaves:
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            leaf = jax.random.key_data(leaf)
        arrays.append((jax.tree_util.keystr(path), np.asarray(leaf)))
    return arrays


def compare_outputs(left: dict, right: dict) -> dict:
    """Bit-for-bit comparison of two ``identity_rollouts`` results."""

    import hashlib

    import numpy as np

    summary = {"outputs": len(left), "arrays": 0, "elements": 0, "mismatched_arrays": 0,
               "max_abs_difference": 0.0, "first_mismatch": None, "per_output": {},
               "left_sha256": None, "right_sha256": None}
    digests = [hashlib.sha256(), hashlib.sha256()]
    if sorted(left) != sorted(right):
        summary["first_mismatch"] = f"output names differ: {sorted(set(left) ^ set(right))}"
        summary["mismatched_arrays"] += 1
        return summary
    for name in left:
        a_leaves, b_leaves = tree_arrays(left[name]), tree_arrays(right[name])
        summary["per_output"][name] = len(a_leaves)
        if [p for p, _ in a_leaves] != [p for p, _ in b_leaves]:
            summary["mismatched_arrays"] += 1
            summary["first_mismatch"] = summary["first_mismatch"] or f"{name}: tree structure differs"
            continue
        for (path, a), (_, b) in zip(a_leaves, b_leaves):
            summary["arrays"] += 1
            summary["elements"] += int(a.size)
            for digest, value in zip(digests, (a, b)):
                digest.update(f"{name}{path}{value.dtype.str}{value.shape}".encode())
                digest.update(np.ascontiguousarray(value).tobytes())
            same = a.dtype == b.dtype and a.shape == b.shape and (
                np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()
            )
            if same:
                continue
            summary["mismatched_arrays"] += 1
            difference = None
            if a.shape == b.shape and a.dtype.kind in "fiub" and b.dtype.kind in "fiub":
                with np.errstate(invalid="ignore"):
                    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
                difference = float(np.nanmax(delta)) if delta.size else 0.0
                summary["max_abs_difference"] = max(summary["max_abs_difference"], difference)
            if summary["first_mismatch"] is None:
                summary["first_mismatch"] = f"{name}{path} {a.dtype}{a.shape} vs {b.dtype}{b.shape}, max |diff| {difference}"
    summary["left_sha256"], summary["right_sha256"] = (d.hexdigest() for d in digests)
    return summary


# --------------------------------------------------------------------------
# Bench workloads (one variant per worker process)


def _scene(jax, jnp, combat, arsenal, batch: int, npcs: int, params):
    """Agent with an iron sword against ``npcs`` battleaxe NPCs on one team."""

    entities = npcs + 1
    loadout = arsenal.hytale_0_5_7_entity_loadouts(
        [("iron_sword",) + ("iron_battleaxe",) * npcs] * batch
    )
    if entities == 2:
        config = arsenal.arsenal_runtime_config(loadout)
        state, _ = arsenal.reset_arsenal_batch(jax.random.split(jax.random.key(570057), batch), params, config)
        return state, config, entities
    teams = jnp.asarray([[0] + [1] * npcs] * batch, dtype=jnp.int32)
    config = arsenal.arsenal_runtime_config(loadout, entity_team_id=teams)
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, npcs, endpoint=False)
    ring = jnp.stack((3.0 * jnp.cos(angles), jnp.zeros_like(angles), 3.0 * jnp.sin(angles)), axis=1)
    position = jnp.concatenate((jnp.zeros((1, 3)), ring), axis=0)[None].repeat(batch, axis=0)
    health = jnp.full((batch, entities), params.target_max_health, dtype=jnp.float32)
    health = health.at[:, combat.AGENT_ENTITY].set(params.agent_max_health)
    state, _ = arsenal.reset_arsenal_batch(
        jax.random.split(jax.random.key(570057), batch), params, config,
        entity_position=position.astype(jnp.float32), entity_health=health,
    )
    return state, config, entities


PROGRAMS = ("planned", "general")


def _planned_step(package: str, arsenal, params, config, program: str = "planned"):
    """The combat step as a caller compiles it. ``planned``: specialised by
    ``plan_arsenal_compilation`` (static flags), as ``training/multi_actor/
    runtime.py`` does. ``general``: no flags, as ``rollout_arsenal_batch`` and
    ``step_learner_arsenal_batch`` call it (every program branch compiled in)."""

    import functools

    if program == "general":
        return arsenal.step_arsenal_batch, None
    plan = importlib.import_module(f"{package}.combat.arsenal.compilation.plan").plan_arsenal_compilation(
        params, config)
    return functools.partial(
        arsenal.step_arsenal_batch,
        compiled_microticks=plan.microticks,
        compiled_null_loadout=plan.null_loadout,
        compiled_has_item_programs=plan.has_item_programs,
        compiled_has_projectiles=plan.has_projectiles,
        compiled_has_areas=plan.has_areas,
    ), plan._asdict()


def workload_compile(package: str, program: str = "planned") -> dict:
    jax = _quiet_jax()
    import jax.numpy as jnp

    combat, arsenal, _ = _engine(package)
    batch = 64
    params = combat.default_combat_params(microticks=1, target_active=True)
    state, config, entities = _scene(jax, jnp, combat, arsenal, batch, 1, params)
    actions = _seeded_actions(jax, jnp, combat, 1, batch)[0]
    keys = jax.random.split(jax.random.key(161803), batch)
    commands = arsenal.empty_arsenal_commands(batch)
    step, plan = _planned_step(package, arsenal, params, config, program)
    started = time.perf_counter()
    lowered = jax.jit(step).lower(state, actions, keys, params, commands, config)
    traced = time.perf_counter()
    compiled = lowered.compile()
    done = time.perf_counter()
    melee_state, _ = combat.reset_batch(jax.random.split(jax.random.key(570057), batch), params)
    started_melee = time.perf_counter()
    melee = jax.jit(combat.step_batch).lower(melee_state, actions, keys, params).compile()
    done_melee = time.perf_counter()
    return {"arsenal_step_trace_lower_s": traced - started, "arsenal_step_compile_s": done - traced,
            "arsenal_step_total_s": done - started, "melee_step_total_s": done_melee - started_melee,
            "batch": batch, "compile_plan": plan,
            "arsenal_step_hlo_fingerprint": hlo_fingerprint(compiled.as_text()),
            "melee_step_hlo_fingerprint": hlo_fingerprint(melee.as_text())}


def workload_throughput(package: str, program: str = "planned", batches=(8, 64), steps: int = 64,
                        repeats: int = 7) -> dict:
    jax = _quiet_jax()
    import jax.numpy as jnp

    combat, arsenal, _ = _engine(package)
    params = combat.default_combat_params(microticks=1, target_active=True)
    result = {"steps": steps, "repeats": repeats, "microticks": 1}
    for batch in batches:
        state, config, _ = _scene(jax, jnp, combat, arsenal, batch, 1, params)
        actions = _seeded_actions(jax, jnp, combat, steps, batch)
        keys = jax.random.split(jax.random.key(161803), steps * batch).reshape((steps, batch))
        commands = _broadcast(jax, jnp, arsenal.empty_arsenal_commands(batch), steps)
        step, plan = _planned_step(package, arsenal, params, config, program)

        def rollout(state, actions, keys, params, commands, config, step=step):
            def body(current, inputs):
                transition = step(current, *inputs[:2], params, inputs[2], config)
                return transition.state, (transition.observation, transition.reward)

            return jax.lax.scan(body, state, (actions, keys, commands))

        started = time.perf_counter()
        compiled = jax.jit(rollout).lower(state, actions, keys, params, commands, config).compile()
        compile_s = time.perf_counter() - started
        jax.block_until_ready(compiled(state, actions, keys, params, commands, config))
        times = []
        for _ in range(repeats):
            started = time.perf_counter()
            jax.block_until_ready(compiled(state, actions, keys, params, commands, config))
            times.append(time.perf_counter() - started)
        median = statistics.median(times)
        result[f"batch{batch}"] = {"rollout_compile_s": compile_s, "rollout_seconds": times,
                                   "lane_steps_per_s": batch * steps / median,
                                   "lane_steps_per_s_best": batch * steps / min(times), "compile_plan": plan}
    return result


def _per_tick(jax, package, combat, arsenal, v3, params, config, batch: int, entities: int,
              program: str = "planned"):
    import jax.numpy as jnp

    commands = arsenal.empty_arsenal_commands(batch, entity_count=entities)
    features = combat.empty_injected_world_features(batch)
    capabilities = _all_clear_capabilities(jnp, arsenal, batch, entities)
    step, _ = _planned_step(package, arsenal, params, config, program)

    def tick(state, action, keys):
        transition = step(state, action, keys, params, commands, config)
        observation = v3.encode_learner_observation_v3(transition.state, params, features, capabilities, config)
        return transition.state, observation, transition.reward

    return jax.jit(tick)


def workload_realtime(package: str, program: str = "planned", npc_counts=(1, 8), steps: int = 300) -> dict:
    jax = _quiet_jax()
    import jax.numpy as jnp
    import numpy as np

    combat, arsenal, v3 = _engine(package)
    params = combat.default_combat_params(microticks=1, target_active=True)
    result = {"steps": steps, "batch": 1, "microticks": 1, "tick_budget_ms": TICK_BUDGET_MS,
              "p95_target_ms": P95_TARGET_MS}
    for npcs in npc_counts:
        state, config, entities = _scene(jax, jnp, combat, arsenal, 1, npcs, params)
        tick = _per_tick(jax, package, combat, arsenal, v3, params, config, 1, entities, program)
        actions = _seeded_actions(jax, jnp, combat, steps, 1)
        keys = jax.random.split(jax.random.key(161803), steps).reshape((steps, 1))
        started = time.perf_counter()
        warm = tick(state, actions[0], keys[0])
        jax.block_until_ready(warm)
        first_call_s = time.perf_counter() - started
        walls, simulated = [], 0.0
        for step in range(steps):
            started = time.perf_counter()
            state, observation, reward = tick(state, actions[step], keys[step])
            jax.block_until_ready((state, observation, reward))
            walls.append(time.perf_counter() - started)
            simulated += float(np.asarray(state.combat.last_motion_delta_seconds)[0])
        ms = np.asarray(walls) * 1000.0
        result[f"npcs{npcs}"] = {
            "entities": entities, "first_call_s": first_call_s,
            "p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)),
            "max_ms": float(ms.max()), "mean_ms": float(ms.mean()),
            "simulated_s": simulated, "wall_s": float(np.sum(walls)),
            "realtime_factor": simulated / float(np.sum(walls)),
            "within_budget": bool(np.percentile(ms, 95) <= TICK_BUDGET_MS),
            "meets_p95_target": bool(np.percentile(ms, 95) <= P95_TARGET_MS),
        }
    return result


def workload_region(package: str, program: str = "planned", working_set: int = 8) -> dict:
    jax = _quiet_jax()

    from hytalegym.worldgen.region.library import RegionArtifactLibrary
    from hytalegym.worldgen.region.traversal_library import RegionTraversalGraphLibrary

    import inspect

    world = importlib.import_module(f"{package}.world")
    # The committed loader always verifies the artifacts; a later edit adds
    # verify_artifacts. Use the cheaper path when the tree has it.
    options = ({"verify_artifacts": False}
               if "verify_artifacts" in inspect.signature(RegionArtifactLibrary.load).parameters else {})
    started = time.perf_counter()
    library = RegionArtifactLibrary.load(REGION_MANIFEST, **options)
    loaded = time.perf_counter()
    traversal = RegionTraversalGraphLibrary.load(TRAVERSAL_MANIFEST, library)
    traversal_loaded = time.perf_counter()
    selection = world.region_atlas_from_artifact_library(
        library, split="train", working_set_capacity=working_set, selection_key=0,
        traversal_library=traversal,
    )
    jax.block_until_ready(selection.atlas)
    built = time.perf_counter()
    return {"library_load_s": loaded - started, "traversal_load_s": traversal_loaded - loaded,
            "atlas_and_path_table_build_s": built - traversal_loaded, "total_s": built - started,
            "working_set": working_set, "manifest": str(REGION_MANIFEST),
            "verify_artifacts": options.get("verify_artifacts", True)}


def _hlo_sources(text: str) -> dict[str, tuple[str, str, int, str]]:
    """Instruction name -> (file, function, line, op_name) from compiled HLO text.

    The module text carries ``FileNames``, ``FunctionNames``, ``FileLocations``
    and ``StackFrames`` tables (rows start with an integer id); an
    instruction's ``stack_frame_id`` names its innermost user frame.
    """

    tables: dict[str, dict[str, str]] = {name: {} for name in _HLO_TABLES}
    section = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in _HLO_TABLES:
            section = stripped
        elif section and re.match(r"^\d+ ", stripped):
            index, _, rest = stripped.partition(" ")
            tables[section][index] = rest
        elif stripped:
            section = None
    names = {k: v.strip('"') for k, v in tables["FileNames"].items()}
    functions = {k: v.strip('"') for k, v in tables["FunctionNames"].items()}
    locations = {}
    for index, rest in tables["FileLocations"].items():
        fields = dict(re.findall(r"(\w+)=(\d+)", rest))
        locations[index] = (names.get(fields.get("file_name_id"), "?"),
                            functions.get(fields.get("function_name_id"), "?"),
                            int(fields.get("line", 0)))
    frames = {}
    for index, rest in tables["StackFrames"].items():
        fields = dict(re.findall(r"(\w+)=(\d+)", rest))
        frames[index] = locations.get(fields.get("file_location_id"), ("?", "?", 0))
    sources = {}
    for match in re.finditer(r"^\s*(?:ROOT )?%([\w.\-]+) = .*?metadata=\{([^}]*)\}", text, re.M):
        name, metadata = match.groups()
        frame = re.search(r"stack_frame_id=(\d+)", metadata)
        op = re.search(r'op_name="([^"]*)"', metadata)
        location = frames.get(frame.group(1), ("?", "?", 0)) if frame else ("?", "?", 0)
        sources[name] = (*location, op.group(1) if op else "")
    return sources


def _engine_relative(path: str, root: str | None = None) -> str:
    if root and path.startswith(root):
        return path[len(root):]
    for marker in ("/hytalegym/jax/", "/engine/"):
        if marker in path:
            return path.split(marker, 1)[1]
    return path


def workload_profile(package: str, receipts: Path,
                     scenes=((64, 1, "planned"), (1, 8, "planned"), (1, 8, "general")), steps: int = 20) -> dict:
    jax = _quiet_jax()
    import jax.numpy as jnp

    combat, arsenal, v3 = _engine(package)
    root = str(Path(importlib.import_module(package).__file__).resolve().parent) + "/"
    params = combat.default_combat_params(microticks=1, target_active=True)
    result = {}
    for batch, npcs, program in scenes:
        state, config, entities = _scene(jax, jnp, combat, arsenal, batch, npcs, params)
        tick = _per_tick(jax, package, combat, arsenal, v3, params, config, batch, entities, program)
        actions = _seeded_actions(jax, jnp, combat, steps, batch)
        keys = jax.random.split(jax.random.key(161803), steps * batch).reshape((steps, batch))
        sources = _hlo_sources(tick.lower(state, actions[0], keys[0]).compile().as_text())
        jax.block_until_ready(tick(state, actions[0], keys[0]))
        unprofiled, current = [], state
        for step in range(steps):
            started = time.perf_counter()
            current, observation, reward = tick(current, actions[step], keys[step])
            jax.block_until_ready((current, observation, reward))
            unprofiled.append(time.perf_counter() - started)
        trace_dir = receipts / f"profile_{program}_b{batch}_npc{npcs}"
        shutil.rmtree(trace_dir, ignore_errors=True)
        started = time.perf_counter()
        with jax.profiler.trace(str(trace_dir), create_perfetto_trace=True):
            current = state
            for step in range(steps):
                current, observation, reward = tick(current, actions[step], keys[step])
                jax.block_until_ready((current, observation, reward))
        wall = time.perf_counter() - started
        trace = next(trace_dir.rglob("perfetto_trace.json.gz"))
        events = json.load(gzip.open(trace))
        events = events["traceEvents"] if isinstance(events, dict) else events
        by_function: Counter = Counter()
        by_line: Counter = Counter()
        by_op: Counter = Counter()
        events_by_function: Counter = Counter()
        total, count = 0.0, 0
        for event in events:
            if event.get("ph") != "X" or event.get("name") not in sources:
                continue
            file, function, line, op = sources[event["name"]]
            duration = float(event.get("dur", 0.0)) / 1000.0  # ms
            total += duration
            count += 1
            rel = _engine_relative(str(Path(file).resolve()) if file != "?" else file, root)
            by_function[f"{rel}::{function}"] += duration
            events_by_function[f"{rel}::{function}"] += 1
            by_line[f"{rel}:{line} ({function})"] += duration
            by_op[op.rsplit("/", 1)[-1] or event["name"]] += duration
        result[f"{program}_b{batch}_npc{npcs}"] = {
            "steps": steps, "wall_s": wall, "thunk_ms_total": total,
            "thunk_ms_per_step": total / steps, "thunk_events_per_step": count / steps,
            "unprofiled_ms_per_step_median": statistics.median(unprofiled) * 1000.0,
            "top_functions": [(k, v / steps, events_by_function[k] / steps)
                              for k, v in by_function.most_common(12)],
            "top_lines": [(k, v / steps) for k, v in by_line.most_common(15)],
            "top_ops": [(k, v / steps) for k, v in by_op.most_common(10)],
            "trace": str(trace),
        }
    return result


WORKLOADS = {
    "compile": workload_compile,
    "throughput": workload_throughput,
    "realtime": workload_realtime,
    "region": workload_region,
}


# --------------------------------------------------------------------------
# Orchestration


def _load_averages() -> list[float]:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        return []


def _head_tree(commit: str, receipts: Path) -> Path:
    """``git archive`` of the render's source commit (read-only on the repo)."""

    root = receipts / f"head-{commit[:12]}"
    stamp = root / "COMMIT"
    if stamp.is_file() and stamp.read_text().strip() == commit:
        return root / "HytaleRL/hytalegym"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "archive", commit, "HytaleRL/hytalegym/hytalegym"],
        check=True, capture_output=True,
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(root)], input=archive, check=True)
    stamp.write_text(commit + "\n")
    return root / "HytaleRL/hytalegym"


def _worker_env(pythonpath: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("HYTALERL_JAX_CACHE", "JAX_COMPILATION_CACHE_DIR")}
    env.update(PYTHONPATH=str(pythonpath), JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
    return env


def run_worker(workload: str, variant: str, rendered: Path, pythonpath: Path, receipts: Path,
               program: str = "planned") -> dict:
    out = receipts / f"{workload}_{program}_{variant}.json"
    command = [sys.executable, "-P", str(HERE / "bench.py"), "worker", workload,
               "--variant", variant, "--rendered", str(rendered), "--json", str(out),
               "--receipts", str(receipts), "--program", program]
    before = _load_averages()
    started = time.perf_counter()
    completed = subprocess.run(command, env=_worker_env(pythonpath), capture_output=True, text=True)
    record = {"workload": workload, "variant": variant, "program": program, "exit": completed.returncode,
              "wall_s": time.perf_counter() - started, "load_before": before, "load_after": _load_averages()}
    if completed.returncode != 0:
        record["stderr_tail"] = completed.stderr[-4000:]
        return record
    record["result"] = json.loads(out.read_text())
    return record


REALTIME_ROUNDS = 2


def _best_of(rounds: list[dict]) -> dict:
    """Realtime rounds -> the lower envelope (contention only adds time)."""

    best = dict(rounds[0])
    for key, value in rounds[0].items():
        if not (isinstance(value, dict) and "p95_ms" in value):
            continue
        scenes = [r[key] for r in rounds if key in r]
        merged = dict(value)
        for metric in ("p50_ms", "p95_ms", "max_ms", "mean_ms"):
            merged[metric] = min(s[metric] for s in scenes)
        merged["realtime_factor"] = max(s["realtime_factor"] for s in scenes)
        merged["within_budget"] = merged["p95_ms"] <= TICK_BUDGET_MS
        merged["meets_p95_target"] = merged["p95_ms"] <= P95_TARGET_MS
        best[key] = merged
    best["rounds"] = len(rounds)
    return best


def _rows(records: dict) -> list[tuple[str, str, float | None, float | None]]:
    def get(variant, *keys):
        node = records.get(variant)
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        return node

    rows = []
    for program in PROGRAMS:
        c, t, r = f"compile:{program}", f"throughput:{program}", f"realtime:{program}"
        rows += [
            ("compile", f"[{program}] combat step trace+lower+compile, batch 64 (s)", (c, "arsenal_step_total_s")),
            ("compile", f"[{program}]   of which XLA compile (s)", (c, "arsenal_step_compile_s")),
            ("throughput", f"[{program}] lane-steps/s, batch 8 (median)", (t, "batch8", "lane_steps_per_s")),
            ("throughput", f"[{program}] lane-steps/s, batch 64 (median)", (t, "batch64", "lane_steps_per_s")),
            ("throughput", f"[{program}] rollout compile, batch 64 (s)", (t, "batch64", "rollout_compile_s")),
            ("realtime", f"[{program}] per-tick p50, 1 NPC, best of 2 (ms)", (r, "npcs1", "p50_ms")),
            ("realtime", f"[{program}] per-tick p95, 1 NPC, best of 2 (ms)", (r, "npcs1", "p95_ms")),
            ("realtime", f"[{program}] real-time factor, 1 NPC (sim s / wall s)", (r, "npcs1", "realtime_factor")),
            ("realtime", f"[{program}] per-tick p50, 8 NPCs, best of 2 (ms)", (r, "npcs8", "p50_ms")),
            ("realtime", f"[{program}] per-tick p95, 8 NPCs, best of 2 (ms)", (r, "npcs8", "p95_ms")),
            ("realtime", f"[{program}] real-time factor, 8 NPCs (sim s / wall s)", (r, "npcs8", "realtime_factor")),
        ]
    rows += [
        ("compile", "melee core step trace+lower+compile, batch 64 (s)", ("compile:planned", "melee_step_total_s")),
        ("region", "Region library load (s)", ("region:planned", "library_load_s")),
        ("region", "traversal graphs load (s)", ("region:planned", "traversal_load_s")),
        ("region", "atlas + path table build, 8 worlds (s)", ("region:planned", "atlas_and_path_table_build_s")),
    ]
    return [(w, label, get("original", *keys), get("rendered", *keys)) for w, label, keys in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hytalegym.sim.common.bench")
    sub = parser.add_subparsers(dest="command")
    worker = sub.add_parser("worker")
    worker.add_argument("workload", choices=sorted(WORKLOADS) + ["profile"])
    worker.add_argument("--variant", choices=("original", "rendered"), required=True)
    worker.add_argument("--rendered", type=Path, required=True)
    worker.add_argument("--json", type=Path, required=True)
    worker.add_argument("--receipts", type=Path, required=True)
    worker.add_argument("--program", choices=PROGRAMS, default="planned")
    identity = sub.add_parser("identity")
    identity.add_argument("--rendered", type=Path, required=True)
    identity.add_argument("--json", type=Path)
    parser.add_argument("--rendered", type=Path)
    parser.add_argument("--original", choices=("head", "worktree"), default="head")
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--skip-region", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "worker":
        package = package_for(args.variant, args.rendered)
        if args.workload == "profile":
            result = workload_profile(package, args.receipts)
        else:
            result = WORKLOADS[args.workload](package, args.program)
        result["package"] = package
        args.json.write_text(json.dumps(result, indent=1, default=str))
        return 0
    if args.command == "identity":
        original = identity_rollouts(SOURCE_PACKAGE)
        package = package_for("rendered", args.rendered)
        rendered = identity_rollouts(package)
        summary = compare_outputs(original, rendered)
        summary["identical"] = summary["mismatched_arrays"] == 0 and summary["arrays"] > 0
        summary["original_module"] = importlib.import_module(f"{SOURCE_PACKAGE}.combat.arsenal.runtime").__file__
        summary["rendered_module"] = importlib.import_module(f"{package}.combat.arsenal.runtime").__file__
        text = json.dumps(summary, indent=1)
        if args.json:
            args.json.write_text(text)
        print(text)
        return 0 if summary["identical"] else 1
    if args.rendered is None:
        parser.error("--rendered is required")

    rendered = args.rendered.resolve()
    manifest = _load_render().read_manifest(rendered)
    receipts = (args.receipts or rendered.parent / "render_bench").resolve()
    receipts.mkdir(parents=True, exist_ok=True)
    if args.original == "head":
        pythonpath = _head_tree(manifest["source"]["commit"], receipts)
    else:
        pythonpath = REPO_ROOT / "HytaleRL/hytalegym"
    plan = [(w, p) for w in ("compile", "throughput", "realtime") for p in PROGRAMS]
    plan += [] if args.skip_region else [("region", "planned")]
    records: dict = {"original": {}, "rendered": {}}
    runs = []
    for workload, program in plan:
        rounds = REALTIME_ROUNDS if workload == "realtime" else 1
        for round_index in range(rounds):
            for variant in ("original", "rendered"):
                run = run_worker(workload, variant, rendered, pythonpath, receipts, program)
                run["round"] = round_index + 1
                runs.append({k: v for k, v in run.items() if k != "result"})
                if "result" not in run:
                    print(f"{workload}/{program}/{variant} FAILED (exit {run['exit']}):\n"
                          f"{run.get('stderr_tail', '')}", file=sys.stderr)
                    continue
                key = f"{workload}:{program}"
                if workload == "realtime":
                    records[variant].setdefault(f"{key}:rounds", []).append(run["result"])
                    records[variant][key] = _best_of(records[variant][f"{key}:rounds"])
                else:
                    records[variant][key] = run["result"]
    profile = None
    if not args.skip_profile:
        run = run_worker("profile", "rendered", rendered, pythonpath, receipts)
        run["round"] = 1
        runs.append({k: v for k, v in run.items() if k != "result"})
        profile = run.get("result")
        if profile is None:
            print(f"profile FAILED:\n{run.get('stderr_tail', '')}", file=sys.stderr)

    import jax  # noqa: F401  (version only)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    receipt = {
        "schema": "hytalerl_sim_render_bench_v1", "created_utc": stamp,
        "rendered": str(rendered), "package": manifest["package"], "generation": manifest["generation"],
        "source_commit": manifest["source"]["commit"], "overlay_sha256": manifest["overlay"]["sha256"],
        "original": args.original, "pythonpath": str(pythonpath),
        "machine": {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                    "python": platform.python_version(), "jax": sys.modules["jax"].__version__},
        "runs": runs, "results": records, "profile": profile,
        "table": [list(row) for row in _rows(records)],
    }
    path = receipts / f"bench-{stamp}.json"
    path.write_text(json.dumps(receipt, indent=1, default=str))

    print(f"bench: {manifest['generation']} render {rendered} vs original ({args.original}, "
          f"{manifest['source']['commit'][:12]}); CPU, persistent compile cache off")
    print(f"{'metric':52s} {'original':>12s} {'rendered':>12s} {'delta':>8s}")
    for _, label, a, b in _rows(records):
        fa = f"{a:12.4g}" if isinstance(a, (int, float)) else f"{'-':>12s}"
        fb = f"{b:12.4g}" if isinstance(b, (int, float)) else f"{'-':>12s}"
        delta = f"{(b - a) / a * 100:+7.1f}%" if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a else ""
        print(f"{label:52s} {fa} {fb} {delta:>8s}")
    print(f"tick budget {TICK_BUDGET_MS:.1f} ms, p95 target {P95_TARGET_MS:.0f} ms (engine step + observation encode only)")
    for program in PROGRAMS:
        for field in ("arsenal_step_hlo_fingerprint", "melee_step_hlo_fingerprint"):
            a = records["original"].get(f"compile:{program}", {}).get(field)
            b = records["rendered"].get(f"compile:{program}", {}).get(field)
            if a or b:
                print(f"compiled HLO, metadata stripped, [{program}] {field.split('_hlo')[0]}: "
                      f"{'IDENTICAL' if a == b else 'DIFFERENT'} ({(a or '-')[:12]} vs {(b or '-')[:12]})")
    for run in runs:
        print(f"  run {run['workload']:10s} {run['program']:8s} {run['variant']:8s} r{run['round']} exit {run['exit']} "
              f"{run['wall_s']:6.1f}s "
              f"load {run['load_before']} -> {run['load_after']}")
    if profile:
        for scene, data in profile.items():
            if not isinstance(data, dict) or "top_functions" not in data:
                continue
            print(f"profile {scene}: {data['thunk_ms_per_step']:.2f} ms thunk time per step "
                  f"({data['steps']} steps, wall {data['wall_s']:.2f}s); top functions (ms/step):")
            print(f"    unprofiled wall {data['unprofiled_ms_per_step_median']:.2f} ms/step; "
                  f"{data['thunk_events_per_step']:.0f} thunk events/step")
            for name, ms, events in data["top_functions"][:5]:
                print(f"    {ms:8.3f} ms  {events:7.0f} events  {name}")
    print(f"receipt: {path}")
    failed = [r for r in runs if r["exit"] != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
