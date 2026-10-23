"""What one tick costs when every agent has its own model (``docs/VIRTUAL_HUMAN.md`` 16.3; the owner, 2026-10-22:
"separate models for each agent"). Times the whole tick, ``runtime/state_tick.tick`` (the state layer, then the trunk
with its latent memory, background and object slots), on recorded Region ticks, with the setup of the tick-cost probe
(same scene, encoder, persona and warm-up; 30 recorded ticks, then 10 x 30 timed ticks), for N agents in five layouts:

- ``shared``: one trunk, one batch of N (every timing so far).
- ``stacked``: N trunks with their own weights, stacked on a leading axis and run in one program by ``jax.vmap``, each
  agent a batch of 1; the stacked params are an argument of the jitted program.
- ``sequential``: N trunks with their own weights; one batch-1 program, compiled once, called N times per tick.
- ``processes`` (N <= 8): N OS processes, each one agent with its own weights at batch 1, started together, each
  with XLA's default threads (so N of them oversubscribe the cores).
- ``processes_1thread``: the same, each process limited to one thread (``SINGLE_THREAD``): N agents on N cores.

Time it on the private gen-7 render with the ``perception_v2`` construct, as the probe does (``runtime/gen7.py``): the
channel gives the agent its position, so the stores write as they do in play. Without the render (the tests) the body
yaw comes from the world and the stores do not write. From the repo root, with DIR the render
(``--render`` once to make it): prints one JSON line per round, N and layout, and ``uptime`` before and after:
``python -m experimental.neural_agent.skill_agent.runtime.gen7 --engine DIR --construct perception_v2 -- \\
    experimental.neural_agent.skill_agent.bench.agent_models --engine DIR --n 1 2 4 8 16 --rounds 2``
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.bench import encoding_audit as audit
from experimental.neural_agent.skill_agent.contract import tokens_v3 as v3
from experimental.neural_agent.skill_agent.model.encoder import (
    EncoderConfig, init_encoder, initial_background, initial_memory, parameter_count, token_widths)
from experimental.neural_agent.skill_agent.model.skill_module import Trunk, make_trunk
from experimental.neural_agent.skill_agent.runtime import state_tick
from experimental.neural_agent.skill_agent.state import layer
from experimental.neural_agent.skill_agent.state import persona as pe
from experimental.neural_agent.skill_agent.state.senses import Light
from hytalegym.jax.combat.observation.v3.policy.layout import neutral_arsenal_policy_action_factors
from worlds.region import region_scene

CONFIG = EncoderConfig(token_types=token_widths(v3.TOKEN_SCHEMA_V3), condition_types=("persona", "affect"),
                       memory_slots=8, background_slots=16, object_types=("entity", "block_candidate"))
LAYOUTS = ("shared", "stacked", "sequential", "processes", "processes_1thread")
LANE_KEYS = ("light", "action_surface", "body_yaw_degrees")  # per agent; the others are the game's content, shared
PROCESS_LIMIT = 8
LAUNCHER = "experimental.neural_agent.skill_agent.runtime.gen7"
MODULE = "experimental.neural_agent.skill_agent.bench.agent_models"
ROOT = Path(__file__).resolve().parents[4]


def record(batch: int, ticks: int = 30) -> list[tuple[Any, jax.Array, dict]]:
    """The probe's recording: ``ticks`` ticks of ``batch`` agents in a Region world, each acting at random (legal,
    held). Per tick: the learner's observation, the last action, and the tick's keyword arguments."""

    scene = region_scene(weapons=("iron_sword",) * batch, target_active=True)
    reset, step, pick = jax.jit(scene.environment.reset), jax.jit(scene.environment.step), jax.jit(audit.random_legal)
    env, policy, _ = reset(jax.random.split(jax.random.key(7), batch))
    channel = getattr(env.learner_observation, "perception_v2", None) is not None  # the render's, under the launcher
    tables = audit.registry_tables()
    effects = getattr(audit, "effect_tables", lambda: None)()
    factors = neutral_arsenal_policy_action_factors(batch)
    recorded = []
    for t in range(ticks):
        light = env.light_tokens
        kw = {"tables": tables, "light": Light(light.token_f32, light.token_mask & light.available[:, None]),
              "action_surface": env.action_surface, "block_reach_blocks": float(scene.maximum_distance)}
        if effects is not None:
            kw["effects"] = effects
        if not channel:  # no channel: the body yaw from the world, as the encoding audit does (no position: the
            kw["body_yaw_degrees"] = env.runtime.combat.yaw[:, 0]  # stores do not write, so time with the render)
        recorded.append((env.learner_observation, audit.gen7_action(factors), kw))
        factors = pick(jax.random.key(3000 + t), env.action_mask, factors)
        env, policy = step(env, policy, factors, jax.random.split(jax.random.key(t), batch))[:2]
    return recorded


def start_state(params_per_lane: list, batch: int) -> layer.AgentState:
    """The batch's state at an episode's start. Lane i's latent memory and background are the initial slots of
    ``params_per_lane[i]``, so each agent starts from its own weights (one set of slots when they are all one trunk)."""

    latent = jnp.concatenate([initial_memory(params, CONFIG, 1) for params in params_per_lane])
    background = jnp.concatenate([initial_background(params, CONFIG, 1) for params in params_per_lane])
    return layer.initial_state(batch, latent=latent, persona=pe.sample_personas(jax.random.key(1), batch),
                               object_width=CONFIG.width, background=background)


def split_lanes(tree: Any, n: int) -> Any:
    """Each leaf's leading axis (the n lanes of a batch) split into [n, 1, ...]: a batch-1 slice per agent, on the
    leading axis. Raises when a leaf does not have n lanes in front."""

    def split(leaf):
        shape = np.shape(leaf)
        if not shape or shape[0] != n:
            raise ValueError(f"split_lanes: a leaf of shape {shape} has no leading lane axis of {n}")
        return leaf.reshape((n, 1) + shape[1:])

    return jax.tree_util.tree_map(split, tree)


def per_agent(split: Any, n: int) -> list:
    """A split tree's n agents, each a batch-1 tree."""

    return [jax.tree_util.tree_map(lambda leaf: leaf[i], split) for i in range(n)]


def split_inputs(recorded: list, n: int) -> list:
    """Each recorded tick split for n agents: (observation, last action, per-agent keywords, shared keywords)."""

    out = []
    for observation, action, kw in recorded:
        lane = {key: kw[key] for key in LANE_KEYS if key in kw}
        shared = {key: value for key, value in kw.items() if key not in LANE_KEYS}
        out.append((split_lanes(observation, n), split_lanes(action, n), split_lanes(lane, n), shared))
    return out


def sequential_inputs(split_tick: tuple, n: int) -> list:
    """One split tick as the list of its n agents' inputs, each (observation, last action, keywords, shared
    keywords)."""

    observation, action, lane, shared = split_tick
    return [(o, a, k, shared) for o, a, k in zip(per_agent(observation, n), per_agent(action, n), per_agent(lane, n))]


def tick_with(params: Any, state: layer.AgentState, observation: Any, action: jax.Array, lane: dict, shared: dict):
    """One agent's tick with its own trunk: the params are an argument of the program, not a closure."""

    return state_tick.tick(Trunk(CONFIG, params, ""), state, observation, last_action=action, **lane, **shared)


def shared_run(params: Any) -> Callable:
    """The probe's program: one trunk, closed over, for the whole batch. ``run(state, (observation, action, kw))``."""

    trunk = make_trunk(params, CONFIG)
    program = jax.jit(lambda state, observation, action, kw: state_tick.tick(trunk, state, observation,
                                                                             last_action=action, **kw))
    return lambda state, inp: program(state, *inp)


def stacked_run(params_list: list) -> Callable:
    """N trunks with their own weights in one program: ``jax.vmap`` over agents, each a batch of 1.
    ``run(state, inputs)`` takes the split state and the split inputs (``split_lanes``, ``split_inputs``)."""

    stacked = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *params_list)
    program = jax.jit(jax.vmap(tick_with, in_axes=(0, 0, 0, 0, 0, None)))
    return lambda state, inp: program(stacked, state, *inp)


def sequential_run(params_list: list) -> Callable:
    """N trunks with their own weights, one program compiled at batch 1 and called once per agent per tick.
    ``run(states, inputs)`` takes the N agents' states and their inputs (``sequential_inputs``)."""

    program = jax.jit(tick_with)

    def run(states: list, inputs: list):
        outs = [program(params, state, *inp) for params, state, inp in zip(params_list, states, inputs)]
        return [out[0] for out in outs], [out[1] for out in outs]

    return run


def warm(run: Callable, state: Any, inputs: list) -> tuple[float, Any]:
    """The probe's warm-up: the first call (its time is the compile plus one tick), then the recorded ticks threaded
    through the state. Returns (first call in ms, the state)."""

    start = time.perf_counter()
    view, state = run(state, inputs[0])
    jax.block_until_ready((view, state))
    first_ms = 1000 * (time.perf_counter() - start)
    for inp in inputs[1:]:
        view, state = run(state, inp)
    jax.block_until_ready((view, state))
    return first_ms, state


def timed(run: Callable, state: Any, inputs: list, rounds: int = 10) -> list[float]:
    """The probe's timing: each round times every recorded tick from the same state, one block per tick (ms)."""

    times = []
    for _ in range(rounds):
        for inp in inputs:
            start = time.perf_counter()
            jax.block_until_ready(run(state, inp))
            times.append(1000 * (time.perf_counter() - start))
    return times


def trunk_size(params: Any) -> tuple[int, int]:
    """(parameter count, bytes) of one trunk."""

    return parameter_count(params), sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(params))


def _row(layout: str, n: int, compile_ms: float, times: list, count: int, size: int) -> dict:
    p50, p95 = (float(x) for x in np.percentile(times, [50, 95]))
    return {"layout": layout, "n": n, "p50_ms": round(p50, 3), "p95_ms": round(p95, 3),
            "per_agent_p50_ms": round(p50 / n, 3), "per_agent_p95_ms": round(p95 / n, 3),
            "compile_ms": round(compile_ms, 1), "params_total": n * count, "bytes_total": n * size}


def layout_row(layout: str, n: int, recorded: list) -> dict:
    """``shared``, ``stacked`` or ``sequential`` at n agents on the recorded ticks, in this process."""

    if layout == "shared":
        params = init_encoder(jax.random.key(0), CONFIG)
        run, state, inputs = shared_run(params), start_state([params] * n, n), recorded
        count, size = trunk_size(params)
    else:
        params_list = [init_encoder(jax.random.key(seed), CONFIG) for seed in range(n)]
        start = start_state(params_list, n)
        if layout == "stacked":
            run, state, inputs = stacked_run(params_list), split_lanes(start, n), split_inputs(recorded, n)
        else:
            run, state = sequential_run(params_list), per_agent(split_lanes(start, n), n)
            inputs = [sequential_inputs(tick, n) for tick in split_inputs(recorded, n)]
        count, size = trunk_size(params_list[0])
    compile_ms, state = warm(run, state, inputs)
    return _row(layout, n, compile_ms, timed(run, state, inputs), count, size)


def child(key: int) -> None:
    """One agent of the ``processes`` layouts: its own weights (seed ``key``), its own batch-1 world, the shared program
    at batch 1. Prints READY after its warm-up, times its ticks once the parent sends GO, then prints its RESULT
    line."""

    recorded = record(1)
    params = init_encoder(jax.random.key(key), CONFIG)
    run = shared_run(params)
    compile_ms, state = warm(run, start_state([params], 1), recorded)
    print("READY", flush=True)
    if sys.stdin.readline().strip() != "GO":
        raise SystemExit(1)
    p50, p95 = (round(float(x), 3) for x in np.percentile(timed(run, state, recorded), [50, 95]))
    print("RESULT " + json.dumps({"key": key, "p50_ms": p50, "p95_ms": p95, "compile_ms": round(compile_ms, 1)}),
          flush=True)


def _await_ready(process: subprocess.Popen) -> bool:
    for line in process.stdout:
        if line.strip() == "READY":
            return True
    return False


SINGLE_THREAD = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
"""XLA flags for a child limited to one thread, so N agents use N cores instead of oversubscribing them."""


def processes_row(n: int, count: int, size: int, engine: str, single_thread: bool = False) -> dict:
    """N OS processes, each one agent at batch 1 with its own weights, started together (nice 10), each under the same
    gen-7 launcher as the parent. Each child warms up, reports READY, and waits for GO, so the timed windows overlap.
    The row is the slowest child's p50 and p95; ``single_thread`` limits each child to one thread."""

    command = ["nice", "-n", "10", sys.executable, "-m", LAUNCHER, "--engine", engine, "--construct", "perception_v2",
               "--", MODULE, "--engine", engine, "--child"]
    env = dict(os.environ)
    if single_thread:
        env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} {SINGLE_THREAD}".strip()
    children = [subprocess.Popen(command + [str(key)], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 text=True, env=env) for key in range(n)]
    ready = [_await_ready(child) for child in children]
    for child in children:
        if all(ready):
            child.stdin.write("GO\n")
        child.stdin.close()
    outputs = [child.stdout.read() for child in children]
    codes = [child.wait() for child in children]  # every child is waited for, even when one failed
    if not all(ready) or any(codes):
        raise RuntimeError(f"processes at {n}: ready {ready}, exit codes {codes}")
    results = [json.loads(next(line for line in out.splitlines() if line.startswith("RESULT "))[len("RESULT "):])
               for out in outputs]
    return {"layout": "processes_1thread" if single_thread else "processes", "n": n,
            "p50_ms": max(r["p50_ms"] for r in results),
            "p95_ms": max(r["p95_ms"] for r in results), "per_agent_p50_ms": max(r["p50_ms"] for r in results),
            "per_agent_p95_ms": max(r["p95_ms"] for r in results), "compile_ms": max(r["compile_ms"] for r in results),
            "params_total": n * count, "bytes_total": n * size, "per_process": results}


def uptime() -> str:
    return subprocess.run(["uptime"], capture_output=True, text=True, check=True).stdout.strip()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Per-tick cost with one model per agent (see the module docstring).")
    parser.add_argument("--engine", required=True, help="the gen-7 render the launcher mounts (the same DIR)")
    parser.add_argument("--n", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--layouts", nargs="+", choices=LAYOUTS, default=list(LAYOUTS))
    parser.add_argument("--rounds", type=int, default=2, help="the layouts are interleaved round by round")
    parser.add_argument("--child", type=int, help="one agent of the processes layout (set by the parent)")
    args = parser.parse_args(argv)
    if args.child is not None:
        child(args.child)
        return
    print(json.dumps({"uptime_start": uptime()}), flush=True)
    count, size = trunk_size(init_encoder(jax.random.key(0), CONFIG))
    print(json.dumps({"trunk_parameters": count, "trunk_bytes": size}), flush=True)
    for round_ in range(1, args.rounds + 1):
        for n in args.n:
            recorded = None
            for layout in args.layouts:
                if layout.startswith("processes"):
                    if n > PROCESS_LIMIT:
                        continue
                    result = processes_row(n, count, size, args.engine, layout == "processes_1thread")
                else:
                    recorded = record(n) if recorded is None else recorded
                    result = layout_row(layout, n, recorded)
                print(json.dumps({"round": round_, **result}), flush=True)
    print(json.dumps({"uptime_end": uptime()}), flush=True)


if __name__ == "__main__":
    main()
