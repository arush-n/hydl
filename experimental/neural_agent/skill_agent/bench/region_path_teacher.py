"""T1's certificate bench: the path teacher beside walk_to v3's straight teacher, on the Region trainers' world.

Each band (3 to 6, 6 to 10 and 10 to 15 blocks by default, batch 32, 600 ticks = 20 s) is one environment with its traversal
graphs (``make_region_task_environment(..., traversal=True)``). All teachers run on the same placements, goals (radius and
gait as ``WalkToTasks.sample`` draws them) and seed. The straight teacher is walk_to v3 as the trainers use it: the
registered rule (``state_for``) with the 0.25-block stop margin.

The path teachers are ``teaching/path_teacher.py``: ``path_hop1`` aims at the next hop (the default, one hop on);
``path_hop2`` aims two hops on (the first design). The 2026-10-25 runs named these ``path`` (two hops) and ``path_hop1``.

Per band and teacher, one row:

- ``reach_share``: the share of lanes inside the goal radius within the horizon;
- ``median_seconds``: the median first-success time of the lanes that reach;
- ``climb_or_drop_share``: the share of lanes whose shortest graph route has a climb or a drop (a property of the route,
  read down the path teacher's cost field, so the same in every row);
- ``path_ratio_mean`` and ``path_ratio_median``: over the reaching lanes, the agent's horizontal travel up to its first
  success over the shortest route's graph cost;
- ``failures`` and ``failure_categories``: one record per lane that does not reach, with its final planar distance to the
  goal, the node under its feet at the end (-1: none), the next-hop edge from that node (its kind and rise or depth), its
  height change from the start, whether it fell more than 0.5 blocks below its start floor, its planar speed over the last
  second, and whether it ever stood on the goal node. The category is ``never_routed``, ``off_graph`` (``_fell`` if it fell),
  ``no_next_edge``, ``at_goal_node_outside_radius``, ``stalled_<next edge>`` (speed under 0.25 blocks a second) or
  ``moving_at_horizon`` (still moving when the horizon ended).

Gen 7 (step-up on) through the launcher, rendering once into a directory of its own:

    JAX_PLATFORMS=cpu nice -n 10 .venv/bin/python -m experimental.neural_agent.skill_agent.runtime.gen7 \\
        --engine DIR --render -- experimental/neural_agent/skill_agent/bench/region_path_teacher.py --generation 7 --json PATH

Gen 5 (the default engine, no step-up) runs without the launcher, with ``--generation 5 --teachers straight``.
``--bands 6:10,10:15`` narrows the bands; ``--batch`` and ``--ticks`` are as the certificate.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.neural_skills.walk_to.task import WalkToTasks
from experimental.neural_agent.skill_agent.neural_skills.walk_to.teacher import (
    WALK_TO_V3_STOP_MARGIN_BLOCKS,
    walk_to_choices,
)
from experimental.neural_agent.skill_agent.tasks.region_tasks import WORKING_SET_CAPACITY, make_region_task_environment
from experimental.neural_agent.skill_agent.tasks.simple_tasks import TICKS_PER_SECOND, run_episode
from experimental.neural_agent.skill_agent.teaching.path_teacher import (
    PathTeacher,
    allowed_edges,
    next_edge,
    node_under_feet,
    route_edges,
)
from experimental.neural_agent.skill_agent.teaching.teachers import state_for
from hytalegym.jax.combat.types import AGENT_ENTITY
from hytalegym.worldgen.region.traversal import (
    REGION_TRAVERSAL_EDGE_CLIMB,
    REGION_TRAVERSAL_EDGE_DROP,
    REGION_TRAVERSAL_EDGE_WALK,
)

MAXIMUM_TURN_DEGREES = 45.0
DEFAULT_BANDS = ((3.0, 6.0), (6.0, 10.0), (10.0, 15.0))
# straight: walk_to v3 at the goal. path_hop1: the path teacher's one-hop waypoint (the default). path_hop2: two hops on.
TEACHERS = ("straight", "path_hop1", "path_hop2")
STALL_SPEED_BLOCKS_PER_SECOND = 0.25
FALL_BELOW_START_BLOCKS = 0.5
SPEED_WINDOW_TICKS = 30


def parse_bands(text: str) -> tuple[tuple[float, float], ...]:
    bands = []
    for item in text.split(","):
        low, high = (float(value) for value in item.split(":"))
        bands.append((low, high))
    return tuple(bands)


def _straight(combat, goals):
    """walk_to v3 as the trainers run it: the registered (observed-state) rule with the 0.25-block stop margin."""
    return walk_to_choices(
        state_for("walk_to", combat), goals.target_xz, goals.radius, goals.gait,
        maximum_turn_degrees=MAXIMUM_TURN_DEGREES, stop_margin_blocks=WALK_TO_V3_STOP_MARGIN_BLOCKS,
    ).factors


def _edge_label(kind: int, rise: float) -> str:
    if kind == REGION_TRAVERSAL_EDGE_WALK:
        return "flat_walk"
    if kind == REGION_TRAVERSAL_EDGE_CLIMB:
        return f"climb_rise_{rise:.2f}"
    if kind == REGION_TRAVERSAL_EDGE_DROP:
        return f"drop_depth_{-rise:.2f}"
    return f"kind_{kind}"


def _failure_records(traversal, ok, cost_to_goal, shortest, goal_xz, start_y, track, node_track, success, ticks) -> list[dict]:
    """One record per lane that did not reach: where it ended, what it was stuck at, and how it moved."""

    goal_node = np.asarray(traversal.goal_node)
    window = min(SPEED_WINDOW_TICKS, ticks)
    records = []
    for lane in np.flatnonzero(~success):
        world = int(traversal.environment_world_id[lane])
        end = int(node_track[-1, lane])
        final_xz = track[-1, lane, [0, 2]]
        distance = float(np.hypot(*(final_xz - goal_xz[lane])))
        fell = bool(track[:, lane, 1].min() < start_y[lane] - FALL_BELOW_START_BLOCKS)
        speed = float(np.hypot(*(final_xz - track[-1 - window, lane, [0, 2]])) / (window / TICKS_PER_SECOND))
        label = None
        if end >= 0 and end != goal_node[lane] and np.isfinite(cost_to_goal[lane, end]):
            step = next_edge(traversal, ok, cost_to_goal[lane], world, end)
            if step is not None:
                destination, kind = step
                rise = float(traversal.node_position[world, destination, 1] - traversal.node_position[world, end, 1])
                label = _edge_label(kind, rise)
        if not np.isfinite(shortest[lane]):
            category = "never_routed"
        elif end < 0:
            category = "off_graph_fell" if fell else "off_graph"
        elif end == goal_node[lane]:
            category = "at_goal_node_outside_radius"
        elif label is None:
            category = "no_next_edge"
        elif speed < STALL_SPEED_BLOCKS_PER_SECOND:
            category = f"stalled_{label}"
        else:
            category = "moving_at_horizon"
        records.append({
            "lane": int(lane),
            "category": category,
            "final_planar_distance_blocks": distance,
            "end_node": end,
            "next_edge": label,
            "height_change_blocks": float(track[-1, lane, 1] - start_y[lane]),
            "fell_below_start_floor": fell,
            "speed_last_second_blocks": speed,
            "ever_on_goal_node": bool((node_track[:, lane] == goal_node[lane]).any()),
        })
    return records


def band_rows(batch: int, ticks: int, band: tuple[float, float], seed: int, teachers: tuple[str, ...]) -> list[dict]:
    started = time.perf_counter()
    env = make_region_task_environment(
        batch, MAXIMUM_TURN_DEGREES, separation_range=band, seed=seed, traversal=True,
    )
    built = time.perf_counter() - started
    traversal = env.traversal
    placed = env.task_placements
    goal_xz = np.asarray(placed.goal_xyz, np.float64)[:, [0, 2]]
    agent_xz = np.asarray(placed.agent_xz, np.float64)
    start_y = np.asarray(env.floor_y, np.float64)
    distance = np.hypot(goal_xz[:, 0] - agent_xz[:, 0], goal_xz[:, 1] - agent_xz[:, 1])
    goals = WalkToTasks.sample(jax.random.key(seed), batch)._replace(
        target_xz=jnp.asarray(goal_xz, jnp.float32),
        horizon_ticks=jnp.full((batch,), ticks, jnp.int32),
    )
    carry = env.reset(jax.random.split(jax.random.key(seed + 1), batch), placed.agent_xz, placed.agent_yaw, placed.other_xyz)

    ok = allowed_edges(traversal)
    hop1 = PathTeacher(traversal, maximum_turn_degrees=MAXIMUM_TURN_DEGREES, waypoint_hops=1)
    hop2 = PathTeacher(traversal, maximum_turn_degrees=MAXIMUM_TURN_DEGREES, waypoint_hops=2)
    cost_to_goal = np.asarray(hop1.plan.dist, np.float64)
    shortest = cost_to_goal[np.arange(batch), traversal.start_node]
    routes = [route_edges(traversal, ok, cost_to_goal[lane], lane) for lane in range(batch)]
    needs_climb_or_drop = np.array([any(kind != REGION_TRAVERSAL_EDGE_WALK for _, _, kind in edges) for edges in routes])
    unreachable = ~np.isfinite(shortest)
    band_fields = {
        "band_blocks": list(band),
        "lanes": batch,
        "distinct_spawns": int(np.unique(np.round(agent_xz, 3), axis=0).shape[0]),
        "goal_distance_blocks": [float(distance.min()), float(distance.max())],
        "climb_or_drop_share": float(needs_climb_or_drop.mean()),
        "lanes_without_route": int(unreachable.sum()),
        "build_seconds": round(built, 1),
    }

    locate = jax.jit(node_under_feet)
    policies = {
        "straight": _straight,
        "path_hop1": lambda combat, lane_goals: hop1(combat, lane_goals.target_xz, lane_goals.radius, lane_goals.gait).factors,
        "path_hop2": lambda combat, lane_goals: hop2(combat, lane_goals.target_xz, lane_goals.radius, lane_goals.gait).factors,
    }
    rows = []
    for name in teachers:
        positions: list[np.ndarray] = []
        nodes: list[np.ndarray] = []

        def recording(combat, lane_goals, policy=policies[name]):
            position = combat.position[:, AGENT_ENTITY]
            positions.append(np.asarray(position, np.float64))
            nodes.append(np.asarray(locate(hop1.tables, hop1.plan.world, position)))
            return policy(combat, lane_goals)

        started = time.perf_counter()
        result = run_episode(env, goals, WalkToTasks.success, recording, jax.random.key(seed + 2), ticks, carry=carry)
        final = result.combat.position[:, AGENT_ENTITY]
        positions.append(np.asarray(final, np.float64))
        nodes.append(np.asarray(locate(hop1.tables, hop1.plan.world, final)))
        track = np.stack(positions)  # [ticks + 1, B, 3]: the position before each tick, then the last one
        node_track = np.stack(nodes)  # [ticks + 1, B]: the node under the feet at the same instants
        steps = np.hypot(np.diff(track[..., 0], axis=0), np.diff(track[..., 2], axis=0))  # [ticks, B]
        cumulative = np.vstack([np.zeros((1, batch)), np.cumsum(steps, axis=0)])
        success = np.asarray(result.success)
        first = np.asarray(result.first_success_tick)
        travelled = cumulative[np.maximum(first, 0), np.arange(batch)]
        scored = success & np.isfinite(shortest) & (shortest > 0)
        ratio = travelled[scored] / shortest[scored]
        failures = _failure_records(traversal, ok, cost_to_goal, shortest, goal_xz, start_y, track, node_track, success, ticks)
        row = {
            "teacher": name,
            **band_fields,
            "reach_share": float(success.mean()),
            "reached": int(success.sum()),
            "median_seconds": float(np.median(first[success] / TICKS_PER_SECOND)) if success.any() else None,
            "path_ratio_mean": float(ratio.mean()) if ratio.size else None,
            "path_ratio_median": float(np.median(ratio)) if ratio.size else None,
            "failure_categories": dict(Counter(record["category"] for record in failures)),
            "failures": failures,
            "episode_seconds": round(time.perf_counter() - started, 1),
        }
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generation", choices=("5", "7"), required=True, help="the engine the run is on (a label)")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--ticks", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bands", type=parse_bands, default=DEFAULT_BANDS, help="low:high,... in blocks")
    parser.add_argument("--teachers", default=",".join(TEACHERS), help="a comma list of straight, path_hop1, path_hop2")
    parser.add_argument("--json", type=Path, required=True)
    options = parser.parse_args(argv)
    teachers = tuple(item for item in options.teachers.split(",") if item)
    if not set(teachers) <= set(TEACHERS):
        raise SystemExit(f"--teachers takes {TEACHERS}")
    import hytalegym.jax.combat.types as engine_types

    engine = str(Path(engine_types.__file__).resolve())
    rows = []
    for band in options.bands:
        for row in band_rows(options.batch, options.ticks, band, options.seed, teachers):
            row.update(generation=options.generation, working_set_capacity=WORKING_SET_CAPACITY)
            rows.append(row)
            print(json.dumps(row), flush=True)
    options.json.write_text(json.dumps({"generation": options.generation, "engine_module": engine, "batch": options.batch,
                                        "ticks": options.ticks, "seed": options.seed, "rows": rows}, indent=1),
                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
