"""Section 3 of R1: the straight-line walk_to teacher's baseline on the trainers' Region world.

The spec asked for one row per obstacle profile (climb, stairs, drop, detour). None of those profiles exists in the tree:
the Region benchmark's only placement knob is an actor-legal traversal pair inside a horizontal separation band
(``HytaleRL/examples/jax_arsenal_world_benchmark.py``, ``_region_case_with_geometry``). So the rows here are separation
bands, not obstacle types, and say nothing about which obstacle a lane holds.

Each lane starts on its placement (``tasks/region_tasks.py``), with the walk_to goal on the placement's goal and the walk_to
radius and gait sampled as ``WalkToTasks.sample`` samples them. The teacher is ``walk_to_choices`` with the trainers'
turn limit, and a lane succeeds if it is inside the radius within ``--ticks`` (20 s = 600 ticks at 1/30 s).

Run plain (the default engine, as the flat trainers run), from the repo root::

    JAX_PLATFORMS=cpu nice -n 10 .venv/bin/python -m experimental.neural_agent.skill_agent.bench.region_baseline \
        --batch 32 --ticks 600 --seed 0 [--bands 3:6,6:10,10:15] [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.neural_skills.walk_to.task import WalkToTasks
from experimental.neural_agent.skill_agent.neural_skills.walk_to.teacher import walk_to_choices
from experimental.neural_agent.skill_agent.tasks.region_tasks import WORKING_SET_CAPACITY, make_region_task_environment
from experimental.neural_agent.skill_agent.tasks.simple_tasks import TICKS_PER_SECOND, run_episode

MAXIMUM_TURN_DEGREES = 45.0
DEFAULT_BANDS = ((3.0, 6.0), (6.0, 10.0), (10.0, 15.0))


def parse_bands(text: str) -> tuple[tuple[float, float], ...]:
    bands = []
    for item in text.split(","):
        low, high = (float(value) for value in item.split(":"))
        bands.append((low, high))
    return tuple(bands)


def band_row(batch: int, ticks: int, band: tuple[float, float], seed: int) -> dict:
    started = time.perf_counter()
    env = make_region_task_environment(batch, MAXIMUM_TURN_DEGREES, separation_range=band, seed=seed)
    built = time.perf_counter() - started
    placed = env.task_placements
    agent_xz = np.asarray(placed.agent_xz, np.float64)
    goal_xyz = np.asarray(placed.goal_xyz, np.float64)
    floor = np.asarray(env.floor_y, np.float64)
    distance = np.hypot(goal_xyz[:, 0] - agent_xz[:, 0], goal_xyz[:, 2] - agent_xz[:, 1])
    goals = WalkToTasks.sample(jax.random.key(seed), batch)._replace(
        target_xz=jnp.asarray(goal_xyz[:, (0, 2)], jnp.float32),
        horizon_ticks=jnp.full((batch,), ticks, jnp.int32),
    )
    carry = env.reset(
        jax.random.split(jax.random.key(seed + 1), batch),
        placed.agent_xz,
        placed.agent_yaw,
        placed.other_xyz,
    )

    def policy(combat, lane_goals):
        return walk_to_choices(
            combat, lane_goals.target_xz, lane_goals.radius, lane_goals.gait,
            maximum_turn_degrees=MAXIMUM_TURN_DEGREES,
        ).factors

    started = time.perf_counter()
    result = run_episode(env, goals, WalkToTasks.success, policy, jax.random.key(seed + 2), ticks, carry=carry)
    success = np.asarray(result.success)
    first = np.asarray(result.first_success_tick)
    return {
        "band_blocks": list(band),
        "lanes": batch,
        "working_set_capacity": WORKING_SET_CAPACITY,
        "success_share": float(success.mean()),
        "successes": int(success.sum()),
        "median_first_success_seconds": (
            float(np.median(first[success]) / TICKS_PER_SECOND) if success.any() else None
        ),
        "distinct_spawns": int(np.unique(np.round(agent_xz, 3), axis=0).shape[0]),
        "goal_distance_blocks": [float(distance.min()), float(distance.max())],
        "goal_height_above_start_blocks": [float((goal_xyz[:, 1] - floor).min()), float((goal_xyz[:, 1] - floor).max())],
        "horizon_seconds": ticks / TICKS_PER_SECOND,
        "build_seconds": round(built, 1),
        "episode_seconds": round(time.perf_counter() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--ticks", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bands", type=parse_bands, default=DEFAULT_BANDS, help="low:high,... in blocks")
    parser.add_argument("--json", type=Path, default=None)
    options = parser.parse_args(argv)
    rows = []
    for band in options.bands:
        row = band_row(options.batch, options.ticks, band, options.seed)
        rows.append(row)
        print(json.dumps(row), flush=True)
    if options.json is not None:
        options.json.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
