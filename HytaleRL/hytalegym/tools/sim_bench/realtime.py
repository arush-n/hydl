"""Real-time factor and per-step wall time of the production training step.

Scene: ``arena.jax_env.make_scene`` (the JAX PPO/Arsenal environment the arena
tasks train on), flat world, ``microticks=1`` (one 30 Hz engine tick per
decision, the training setting), loadout ``iron_sword``, no opponent profile,
``inert_opponent_ability_slots`` as the opponent ability provider.
Actions are legal random factors from the step's own mask
(``sample_arsenal_action_factors``), drawn outside the timed region. Each step
is timed to ``block_until_ready``; the first ``--warmup`` steps (compile) are
reported separately. Simulated time per lane is its ``tick_count`` advance
times its ``last_motion_delta_seconds`` (gen 5 draws per-lane 1/30 s or 0.045 s
ticks). Real-time factor = simulated seconds / wall seconds, per lane and
summed over lanes.

Run from ``HytaleRL/hytalegym/tools`` with ``PYTHONPATH`` holding the engine and
``arena`` roots (``git archive HEAD HytaleRL/hytalegym/hytalegym arena worlds``
for the committed engine).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time

TICK_SECONDS = 1.0 / 30.0


def _leaf(tree, name: str):
    import jax

    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if any(getattr(entry, "name", None) == name for entry in path):
            return leaf
    raise KeyError(name)


def measure(batch: int, steps: int, warmup: int) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from arena.jax_env import make_scene
    from hytalegym.jax.combat.observation.v3.policy.distribution import sample_arsenal_action_factors
    from hytalegym.jax.combat.opponents.runtime.policy import inert_opponent_ability_slots

    start = time.perf_counter()
    scene = make_scene(
        "sim_bench/realtime",
        loadout="iron_sword",
        opponent=None,
        opponent_ability_provider=inert_opponent_ability_slots,
        batch=batch,
        microticks=1,
    )
    build_seconds = time.perf_counter() - start
    environment = scene.environment
    start = time.perf_counter()
    state, observation, mask = environment.reset(jax.random.split(jax.random.key(570057), batch))
    jax.block_until_ready((state, observation, mask))
    reset_seconds = time.perf_counter() - start
    step = jax.jit(environment.step)
    sample = jax.jit(lambda key, allowed: sample_arsenal_action_factors(key, jnp.where(allowed, 0.0, -jnp.inf))[0])
    wall: list[float] = []
    simulated: list[float] = []
    for t in range(warmup + steps):
        action_key, step_key = jax.random.split(jax.random.key(20261017 + t))
        actions = jax.block_until_ready(sample(action_key, mask))
        before = np.asarray(_leaf(state, "tick_count"))
        start = time.perf_counter()
        state, observation, reward, done, mask = step(state, observation, actions, jax.random.split(step_key, batch))
        jax.block_until_ready((state, observation, reward, done, mask))
        wall.append(time.perf_counter() - start)
        advanced = np.asarray(_leaf(state, "tick_count")) - before
        delta = np.asarray(_leaf(state, "last_motion_delta_seconds"))
        simulated.append(float(np.mean(np.clip(advanced, 0, None) * delta)))
    steady, sim = wall[warmup:], simulated[warmup:]
    ordered = sorted(steady)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    lane_rtf = sum(sim) / sum(steady)
    return {
        "batch": batch,
        "steps_timed": steps,
        "scene_build_seconds": round(build_seconds, 3),
        "reset_seconds": round(reset_seconds, 3),
        "first_step_seconds_including_compile": round(wall[0], 3),
        "step_p50_ms": round(1000 * statistics.median(steady), 3),
        "step_p95_ms": round(1000 * p95, 3),
        "step_max_ms": round(1000 * max(steady), 3),
        "tick_budget_ms": round(1000 * TICK_SECONDS, 3),
        "p95_within_tick": p95 <= TICK_SECONDS,
        "mean_simulated_seconds_per_lane_step": round(statistics.fmean(sim), 6),
        "real_time_factor_per_lane": round(lane_rtf, 2),
        "real_time_factor_all_lanes": round(lane_rtf * batch, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,8")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out")
    args = parser.parse_args()
    import jax

    report = {
        "schema": "hytalerl_sim_realtime_baseline_v1",
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "machine": platform.platform(),
        "load_average_start": os.getloadavg(),
        "runs": [measure(int(b), args.steps, args.warmup) for b in args.batches.split(",")],
    }
    report["load_average_end"] = os.getloadavg()
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
