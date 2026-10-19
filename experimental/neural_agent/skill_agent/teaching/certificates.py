"""Teacher certificates: the scripted walk_to and look_at teachers measured against fixed checks.

docs/TRAINING_PLAN.md section 2c defines the certificate. This module measures T1 (success), T2 (efficiency), T3
(obedience) and a light T7 (failure sorting) on the flat npc_walk environment. T4 (reality), T5 (imitability) and
T6 (recovery) are not measured here.

Each seed is one batched closed-loop run of ``episodes_per_seed`` lanes. Every lane starts at the default spawn with a
goal sampled from the seed. Success is the environment's own test (``WalkToTasks.success``, ``LookAtTasks.success``)
at some tick within the lane's horizon. The teacher's done is recorded per tick and reported two ways: the first done
tick, and the first tick that starts 5 consecutive done ticks. Tick k is the state after k ticks; decision k is made at
that state. Ticks are 1/30 s (checked from the trace). Speeds are blocks per second.

From the repository root, for example:
python -m experimental.neural_agent.skill_agent.teaching.certificates --teacher walk_to --seeds 77001 --episodes 256 --out PATH --date 2026-10-18

77001 is the report seed; 66001 to 66004 are for tuning. ``--seeds`` takes a comma-separated list. The exit status is 0
when the overall check passes and 1 when it does not.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.tasks.simple_tasks import (
    AGENT_EYE_HEIGHT,
    DEFAULT_LOADOUT,
    DEFAULT_OTHER_XYZ,
    DEFAULT_SPAWN_XZ,
    NPC_WALK_BASE_SPEED,
    TICKS_PER_SECOND,
    LookAtTasks,
    WalkToTasks,
    make_flat_npc_environment,
)
from experimental.neural_agent.skill_agent.teaching.teachers import (
    TEACHERS,
    TeacherOutput,
    look_at_choices,
    state_for,
    walk_to_choices,
)
from hytalegym.jax.combat.observation.v3.policy.layout import ARSENAL_POLICY_ACTION_HEAD_SIZES
from hytalegym.jax.combat.types import (
    AGENT_ENTITY,
    GAIT_RUN,
    GAIT_SNEAK,
    GAIT_WALK,
    PLAYER_GAIT_FORWARD_SPEED_MULTIPLIERS,
)

MAXIMUM_TURN_DEGREES = 45.0
#: agent.acceleration of the 0.5.7 NPC ruleset, blocks per second squared
#: (HytaleRL/hytalegym/hytalegym/rulesets/hytale_0_5_7/kweebec_razorleaf_vs_trork_brawler_v9.json).
ACCELERATION = 10.0
#: Ticks excluded from the steady phase: a run gait reaches 5 blocks/s in 15 ticks at ACCELERATION.
RAMP_TICKS = 15
#: The steady phase is farther than this many radii from the goal.
STEADY_RADII = 2.0
#: Each lane's steady-phase mean planar speed must be within this fraction of v.
SPEED_BAND = 0.05
STUCK_SPEED = 0.1
STUCK_TICKS = 30
STREAK_TICKS = 5
T1_BAR = 0.95
T2_BAR = 1.2
Z_95 = 1.959963984540054
GAIT_NAMES = {GAIT_WALK: "walk", GAIT_RUN: "run", GAIT_SNEAK: "sneak"}
WALK_CLASSES = ("stuck", "timeout_far", "timeout_near")
LOOK_CLASSES = ("timeout_yaw", "timeout_pitch", "timeout_yaw_and_pitch")


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion, clipped to [0, 1]."""

    p = successes / trials
    z2 = z * z
    scale = 1.0 + z2 / trials
    centre = (p + z2 / (2.0 * trials)) / scale
    half = z * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials)) / scale
    return max(0.0, centre - half), min(1.0, centre + half)


def reference_seconds(distance: float, speed: float, accel: float = ACCELERATION) -> float:
    """Time to cover ``distance`` from rest: accelerate at ``accel`` to ``speed``, then cruise. d / v + v / (2 a)."""

    return distance / speed + speed / (2.0 * accel)


def gait_speed(gait):
    """Forward speed in blocks per second: the NPC base speed times the gait's forward multiplier."""

    return NPC_WALK_BASE_SPEED * np.asarray(PLAYER_GAIT_FORWARD_SPEED_MULTIPLIERS)[gait]


def _walk_gap(combat, goals):
    """f32[B, 1]: the learner's planar distance to the goal. The walk succeeds where this is <= radius."""

    planar = combat.position[:, AGENT_ENTITY][:, [0, 2]]
    offset = planar - goals.target_xz
    return jnp.hypot(offset[:, 0], offset[:, 1])[:, None]


def _look_gap(combat, goals):
    """f32[B, 2]: the head's yaw and pitch error against the true target, the same geometry as LookAtTasks.success."""

    eye = combat.position[:, AGENT_ENTITY] + jnp.asarray([0.0, AGENT_EYE_HEIGHT, 0.0], dtype=jnp.float32)
    offset = goals.target_xyz - eye
    dx, dy, dz = offset[:, 0], offset[:, 1], offset[:, 2]
    yaw_facing = jnp.rad2deg(jnp.arctan2(-dx, -dz))
    pitch_facing = jnp.rad2deg(jnp.arctan2(dy, jnp.hypot(dx, dz)))
    yaw_error = jnp.mod(combat.agent_head_yaw - yaw_facing + 180.0, 360.0) - 180.0
    return jnp.stack([yaw_error, combat.agent_head_pitch - pitch_facing], axis=1)


def _walk_teacher(combat, goals) -> TeacherOutput:
    return walk_to_choices(
        state_for("walk_to", combat), goals.target_xz, goals.radius, goals.gait,
        maximum_turn_degrees=MAXIMUM_TURN_DEGREES,
    )


def _look_teacher(combat, goals) -> TeacherOutput:
    return look_at_choices(
        state_for("look_at", combat), goals.target_xyz, goals.tolerance_degrees,
        maximum_turn_degrees=MAXIMUM_TURN_DEGREES,
    )


class _Task(NamedTuple):
    name: str
    sample: Callable[[jax.Array, int], Any]
    success: Callable[[Any, Any], jax.Array]
    gap: Callable[[Any, Any], jax.Array]
    teacher: Callable[[Any, Any], TeacherOutput]


_TASKS = {
    "walk_to": _Task("walk_to", WalkToTasks.sample, WalkToTasks.success, _walk_gap, _walk_teacher),
    "look_at": _Task("look_at", LookAtTasks.sample, LookAtTasks.success, _look_gap, _look_teacher),
}

_HEAD_SIZES = np.asarray(ARSENAL_POLICY_ACTION_HEAD_SIZES, dtype=np.int32)
_HEAD_STARTS = np.concatenate([[0], np.cumsum(_HEAD_SIZES)[:-1]]).astype(np.int32)


@jax.jit
def _legal(mask: jax.Array, factors: jax.Array) -> jax.Array:
    """bool[B]: every head's factor is in range and allowed by the flattened carry.action_mask."""

    in_range = (factors >= 0) & (factors < _HEAD_SIZES)
    index = _HEAD_STARTS + jnp.minimum(jnp.maximum(factors, 0), _HEAD_SIZES - 1)
    return jnp.all(in_range & jnp.take_along_axis(mask, index, axis=1), axis=1)


def _observe(task: _Task, combat, goals):
    """The environment's own success, the task's gap and the planar position, at one state."""

    return task.success(combat, goals), task.gap(combat, goals), combat.position[:, AGENT_ENTITY][:, [0, 2]]


@functools.lru_cache(maxsize=4)
def _environment(batch: int):
    return make_flat_npc_environment(
        batch,
        legal_observation=True,
        maximum_turn_degrees=MAXIMUM_TURN_DEGREES,
        loadout_profile=DEFAULT_LOADOUT,
    )


class _Trace(NamedTuple):
    """One batched run as host arrays. State k is the state after k ticks; decision k is made at state k."""

    held: np.ndarray  # bool[T+1, B]: the environment's success holds at state k
    gap: np.ndarray  # f32[T+1, B, K]: the gap the checks read at state k
    planar: np.ndarray  # f32[T+1, B, 2]: planar position (x, z) at state k
    done: np.ndarray  # bool[T, B]: the teacher's done at decision k
    legal: np.ndarray  # bool[T, B]: every factor of decision k is inside carry.action_mask
    seconds: np.ndarray  # f32[T, B]: the motion delta of the tick decision k executed
    terminated: np.ndarray  # bool[T, B]: the environment's terminated flag after decision k


def _run(env, task: _Task, observe, policy, seed: int):
    """Every lane of ``env`` closed loop for the longest horizon in the batch. The goals come from ``seed``."""

    batch = env.batch
    goal_key, reset_key, step_key = jax.random.split(jax.random.key(seed), 3)
    goals = task.sample(goal_key, batch)
    carry = env.reset(
        jax.random.split(reset_key, batch),
        jnp.broadcast_to(jnp.asarray(DEFAULT_SPAWN_XZ, dtype=jnp.float32), (batch, 2)),
        jnp.zeros((batch,), dtype=jnp.float32),
        jnp.broadcast_to(jnp.asarray(DEFAULT_OTHER_XYZ, dtype=jnp.float32), (batch, 3)),
    )
    ticks = int(np.max(np.asarray(goals.horizon_ticks)))
    step_keys = jax.random.split(step_key, ticks)
    rows: dict[str, list] = {field: [] for field in _Trace._fields}
    for tick in range(ticks + 1):
        combat = carry.runtime.combat
        held, gap, planar = observe(combat, goals)
        rows["held"].append(held)
        rows["gap"].append(gap)
        rows["planar"].append(planar)
        if tick == ticks:
            break
        out = policy(combat, goals)
        rows["done"].append(out.done)
        rows["legal"].append(_legal(carry.action_mask, out.factors))
        carry, _observation, info = env.step(carry, out.factors, jax.random.split(step_keys[tick], batch))
        rows["seconds"].append(carry.runtime.combat.last_motion_delta_seconds)
        rows["terminated"].append(info.terminated)
    trace = _Trace(**{field: np.stack([np.asarray(row) for row in rows[field]]) for field in _Trace._fields})
    return goals, trace


def _first_true(mask: np.ndarray) -> np.ndarray:
    """int[B]: the first index along axis 0 where the mask holds, or -1."""

    return np.where(mask.any(axis=0), mask.argmax(axis=0), -1)


def _run_end(mask: np.ndarray, length: int) -> np.ndarray:
    """int[B]: the index where the first run of ``length`` consecutive True values ends, or -1."""

    run = np.zeros(mask.shape[1], dtype=int)
    end = np.full(mask.shape[1], -1)
    for index, row in enumerate(mask):
        run = np.where(row, run + 1, 0)
        end = np.where((end < 0) & (run >= length), index, end)
    return end


def _longest_run(mask: np.ndarray) -> np.ndarray:
    """int[B]: the longest run of consecutive True values along axis 0."""

    run = np.zeros(mask.shape[1], dtype=int)
    best = np.zeros(mask.shape[1], dtype=int)
    for row in mask:
        run = np.where(row, run + 1, 0)
        best = np.maximum(best, run)
    return best


def _lane_stats(name: str, trace: _Trace, goals) -> dict[str, np.ndarray]:
    """Per-lane numbers for one seed. Decision k counts only while k is below the lane's horizon."""

    horizon = np.asarray(goals.horizon_ticks)
    lanes = np.arange(horizon.shape[0])
    decision_index = np.arange(trace.done.shape[0])[:, None]
    state_index = np.arange(trace.held.shape[0])[:, None]
    decision = decision_index < horizon
    inside = (state_index >= 1) & (state_index <= horizon)
    done = trace.done & decision
    first_done = _first_true(done)
    streak_end = _run_end(done, STREAK_TICKS)
    success_at = _first_true(trace.held & inside)
    stats: dict[str, np.ndarray] = {
        "horizon": horizon,
        "success": success_at >= 0,
        "first_success": success_at,
        "first_done": first_done,
        "streak_start": np.where(streak_end >= 0, streak_end - STREAK_TICKS + 1, -1),
        "leaves_done": (first_done >= 0) & np.any(decision & (decision_index > first_done) & ~trace.done, axis=0),
        "illegal": np.sum(~trace.legal & decision, axis=0),
        "decisions": np.sum(decision, axis=0),
        "terminated": np.sum(trace.terminated & decision, axis=0),
        "seconds_min": np.min(np.where(decision, trace.seconds, np.inf), axis=0),
        "seconds_max": np.max(np.where(decision, trace.seconds, -np.inf), axis=0),
    }
    if name == "walk_to":
        radius = np.asarray(goals.radius)
        gait = np.asarray(goals.gait)
        target = np.asarray(goals.target_xz)
        speed_ref = gait_speed(gait)
        start = np.hypot(target[:, 0] - DEFAULT_SPAWN_XZ[0], target[:, 1] - DEFAULT_SPAWN_XZ[1])
        distance = trace.gap[..., 0]
        within = distance <= radius
        step = np.linalg.norm(np.diff(trace.planar, axis=0), axis=-1)
        speed = np.divide(step, trace.seconds, out=np.zeros_like(step), where=trace.seconds > 0)
        steady = decision & (decision_index >= RAMP_TICKS) & (distance[:-1] > STEADY_RADII * radius)
        steady_count = np.sum(steady, axis=0)
        steady_speed = np.sum(np.where(steady, speed, 0.0), axis=0)
        stats.update(
            radius=radius,
            gait=gait,
            speed_ref=speed_ref,
            ref_ticks=reference_seconds(start - radius, speed_ref) * TICKS_PER_SECOND,
            steady_count=steady_count,
            steady_speed=steady_speed,
            steady_ratio=np.where(
                steady_count > 0, steady_speed / np.maximum(steady_count, 1) / speed_ref, np.nan
            ),
            stuck_ticks=_longest_run((speed < STUCK_SPEED) & ~trace.done & decision),
            final_distance=distance[horizon, lanes],
            distance_at_done=distance[np.maximum(first_done, 0), lanes],
        )
    else:
        tolerance = np.asarray(goals.tolerance_degrees)
        within = np.all(np.abs(trace.gap) <= tolerance[None, :, None], axis=-1)
        yaw, pitch = trace.gap[..., 0], trace.gap[..., 1]
        at_done = np.maximum(first_done, 0)
        stats.update(
            done_ok=(np.abs(yaw[at_done, lanes]) <= tolerance) & (np.abs(pitch[at_done, lanes]) <= tolerance),
            yaw_out=np.abs(yaw[horizon, lanes]) > tolerance,
            pitch_out=np.abs(pitch[horizon, lanes]) > tolerance,
        )
    # The gap formulas must agree with the environment's success on every state inside a lane's horizon.
    stats["gap_disagreements"] = np.sum((trace.held != within) & inside, axis=0)
    return stats


def _outcomes(name: str, stats: dict[str, np.ndarray]) -> np.ndarray:
    """One label per lane: success, or its failure class (T7-lite). A failure no class matches stays unclassified."""

    success = stats["success"]
    labels = np.full(success.shape, "unclassified", dtype=object)
    labels[success] = "success"
    failed = ~success
    if name == "walk_to":
        stuck = failed & (stats["stuck_ticks"] > STUCK_TICKS)
        far = failed & ~stuck & (stats["final_distance"] > STEADY_RADII * stats["radius"])
        near = failed & ~stuck & ~far
        labels[stuck] = "stuck"
        labels[far] = "timeout_far"
        labels[near] = "timeout_near"
    else:
        yaw, pitch = stats["yaw_out"], stats["pitch_out"]
        labels[failed & yaw & pitch] = "timeout_yaw_and_pitch"
        labels[failed & yaw & ~pitch] = "timeout_yaw"
        labels[failed & pitch & ~yaw] = "timeout_pitch"
    return labels


def _pooled(per_seed, key: str) -> np.ndarray:
    return np.concatenate([stats[key] for _, stats in per_seed])


def _pct(values: np.ndarray, q: float) -> float | None:
    return None if values.size == 0 else float(np.percentile(values, q))


def _t1(per_seed) -> dict:
    rows = []
    for seed, stats in per_seed:
        successes, episodes = int(stats["success"].sum()), int(stats["success"].size)
        rows.append(
            {
                "seed": seed,
                "successes": successes,
                "episodes": episodes,
                "rate": successes / episodes,
                "wilson_95": list(wilson_interval(successes, episodes)),
            }
        )
    successes = sum(row["successes"] for row in rows)
    episodes = sum(row["episodes"] for row in rows)
    rate = successes / episodes
    low, high = wilson_interval(successes, episodes)
    return {
        "name": "success",
        "per_seed": rows,
        "pooled": {"successes": successes, "episodes": episodes, "rate": rate, "wilson_95": [low, high]},
        "value": rate,
        "interval": [low, high],
        "bar": T1_BAR,
        "rule": "pooled rate >= bar; the Wilson lower bound is reported and not used for pass",
        "pass": bool(rate >= T1_BAR),
    }


def _t2_walk(per_seed) -> dict:
    rows, pooled = [], []
    for seed, stats in per_seed:
        ok = stats["success"]
        ratio = stats["first_success"][ok] / stats["ref_ticks"][ok]
        pooled.append(ratio)
        rows.append({"seed": seed, "successes": int(ok.sum()), "percentiles": {"p50": _pct(ratio, 50), "p90": _pct(ratio, 90)}})
    ratio = np.concatenate(pooled)
    p50 = _pct(ratio, 50)
    return {
        "name": "efficiency",
        "reference": "ticks to success / (30 x (d / v + v / (2 a))), d = start distance - radius, v = 5 x gait forward multiplier, a = 10",
        "per_seed": rows,
        "successful_episodes": int(ratio.size),
        "percentiles": {"p50": p50, "p90": _pct(ratio, 90)},
        "value": p50,
        "bar": T2_BAR,
        "rule": "pooled median ratio <= bar",
        "pass": bool(p50 is not None and p50 <= T2_BAR),
    }


def _t2_look(per_seed) -> dict:
    ticks = _pooled(per_seed, "first_done")
    ticks = ticks[ticks >= 0]
    return {
        "name": "efficiency",
        "ticks_to_done": {"lanes": int(ticks.size), "percentiles": {"median": _pct(ticks, 50), "p95": _pct(ticks, 95)}},
        "value": _pct(ticks, 50),
        "bar": "not defined",
        "rule": "not defined: the codebook limits the speed, and the teacher is that ideal controller",
        "pass": None,
    }


def _illegal_block(per_seed) -> dict:
    illegal = int(_pooled(per_seed, "illegal").sum())
    return {
        "value": illegal,
        "bar": 0,
        "decisions": int(_pooled(per_seed, "decisions").sum()),
        "rule": "every factor of every decision inside carry.action_mask",
        "pass": illegal == 0,
    }


def _t3_walk(per_seed) -> dict:
    steady_count = _pooled(per_seed, "steady_count")
    steady = steady_count > 0
    violating = steady & (np.abs(_pooled(per_seed, "steady_ratio") - 1.0) > SPEED_BAND)
    speed_sum = _pooled(per_seed, "steady_speed")
    speed_ref = _pooled(per_seed, "speed_ref")
    gait = _pooled(per_seed, "gait")
    v_ticks = steady_count * speed_ref
    per_gait = {}
    for code, label in GAIT_NAMES.items():
        mask = steady & (gait == code)
        if mask.any():
            per_gait[label] = {
                "v": float(gait_speed(code)),
                "steady_lanes": int(mask.sum()),
                "speed_ratio": float(speed_sum[mask].sum() / v_ticks[mask].sum()),
                "violations": int(violating[mask].sum()),
            }
    speed = {
        "value": int(violating.sum()),
        "bar": 0,
        "rule": (
            f"every lane's steady-phase mean planar speed within {SPEED_BAND:.0%} of v (steady: tick >= {RAMP_TICKS} "
            f"and planar distance > {STEADY_RADII:g} radii); value counts the lanes outside"
        ),
        "steady_lanes": int(steady.sum()),
        "pooled_speed_ratio": float(speed_sum[steady].sum() / v_ticks[steady].sum()) if steady.any() else None,
        "per_gait": per_gait,
        "pass": bool(steady.any() and not violating.any()),
    }
    first_done = _pooled(per_seed, "first_done")
    done = first_done >= 0
    done_outside = done & (_pooled(per_seed, "distance_at_done") > _pooled(per_seed, "radius"))
    done_distance = {
        "value": int(done_outside.sum()),
        "bar": 0,
        "done_lanes": int(done.sum()),
        "rule": "planar distance at the first done tick <= radius; value counts the lanes outside",
        "pass": bool(done.any() and not done_outside.any()),
    }
    illegal = _illegal_block(per_seed)
    return {
        "name": "obedience",
        "speed": speed,
        "done_distance": done_distance,
        "illegal_decisions": illegal,
        "pass": bool(speed["pass"] and done_distance["pass"] and illegal["pass"]),
    }


def _t3_look(per_seed) -> dict:
    done = _pooled(per_seed, "first_done") >= 0
    outside = done & ~_pooled(per_seed, "done_ok")
    tolerance = {
        "value": int(outside.sum()),
        "bar": 0,
        "done_lanes": int(done.sum()),
        "rule": "the true head error at the first done tick is within tolerance on yaw and pitch; value counts the lanes outside",
        "pass": bool(done.any() and not outside.any()),
    }
    illegal = _illegal_block(per_seed)
    return {
        "name": "obedience",
        "illegal_decisions": illegal,
        "done_tolerance": tolerance,
        "pass": bool(tolerance["pass"] and illegal["pass"]),
    }


def _summary(values: np.ndarray) -> dict:
    values = values[values >= 0]
    return {"lanes": int(values.size), "median": _pct(values, 50), "p90": _pct(values, 90)}


def _done_signal(per_seed) -> dict:
    first = _pooled(per_seed, "first_done")
    start = _pooled(per_seed, "streak_start")
    return {
        "rule": (
            f"first_done_tick: the first decision the teacher reports done. first_tick_done_{STREAK_TICKS}_consecutive: "
            f"the first tick k with done on ticks k to k+{STREAK_TICKS - 1}. Both over lanes that report done in the horizon"
        ),
        "first_done_tick": _summary(first),
        f"first_tick_done_{STREAK_TICKS}_consecutive": _summary(start),
        "lanes_done_without_streak": int(np.sum((first >= 0) & (start < 0))),
        "lanes_leaving_done": int(_pooled(per_seed, "leaves_done").sum()),
    }


def _failure_classes(name: str, per_seed) -> dict:
    classes = WALK_CLASSES if name == "walk_to" else LOOK_CLASSES
    labels_checked = (*classes, "unclassified")
    rows = []
    for seed, stats in per_seed:
        labels = _outcomes(name, stats)
        counts = {label: int(np.sum(labels == label)) for label in labels_checked}
        rows.append({"seed": seed, "failures": int(np.sum(~stats["success"])), "counts": counts})
    pooled = {label: sum(row["counts"][label] for row in rows) for label in labels_checked}
    return {
        "classes": list(classes),
        "pooled": {"failures": sum(row["failures"] for row in rows), "counts": pooled},
        "per_seed": rows,
    }


def _env_block(env, per_seed) -> dict:
    return {
        "environment": "make_flat_npc_environment",
        "motion_model": "npc_walk",
        "motion_timing_profile": [int(env.params.motion_timing_profile_min), int(env.params.motion_timing_profile_max)],
        "nominal_tick_seconds": float(env.params.nominal_dt),
        "observed_tick_seconds": [
            min(float(np.min(stats["seconds_min"])) for _, stats in per_seed),
            max(float(np.max(stats["seconds_max"])) for _, stats in per_seed),
        ],
        "maximum_turn_degrees": MAXIMUM_TURN_DEGREES,
        "loadout_profile": DEFAULT_LOADOUT,
        "legal_observation": True,
        "spawn_xz": [float(value) for value in DEFAULT_SPAWN_XZ],
        "other_xyz": [float(value) for value in DEFAULT_OTHER_XYZ],
        "terminated_ticks": sum(int(stats["terminated"].sum()) for _, stats in per_seed),
    }


def _certificate(name: str, seeds, episodes: int, *, date: str, policy_name: str, env, per_seed) -> dict:
    spec = TEACHERS[name]
    t1 = _t1(per_seed)
    if name == "walk_to":
        t2, t3 = _t2_walk(per_seed), _t3_walk(per_seed)
        overall = bool(t1["pass"] and t2["pass"] and t3["pass"])
        overall_rule = "T1 and T2 and T3"
    else:
        t2, t3 = _t2_look(per_seed), _t3_look(per_seed)
        overall = bool(t1["pass"] and t3["pass"])
        overall_rule = "T1 and T3 (T2 has no bar for look_at)"
    return {
        "scope": "T1, T2, T3 and T7 (light). T4, T5 and T6 are not measured here.",
        "teacher": {
            "name": name,
            "version": spec.version,
            "state": spec.state,
            "kind": spec.kind,
            "verbs": list(spec.verbs),
            "reads": list(spec.reads),
            "parameters": list(spec.parameters),
        },
        "policy": policy_name,
        "date": date,
        "seeds": seeds,
        "episodes_per_seed": episodes,
        "env": env,
        "checks": {"T1": t1, "T2": t2, "T3": t3},
        "failure_classes": _failure_classes(name, per_seed),
        "done_signal": _done_signal(per_seed),
        "sanity": {"success_vs_gap_disagreements": int(_pooled(per_seed, "gap_disagreements").sum())},
        "overall": {"pass": overall, "rule": overall_rule},
    }


def _write(cert: dict, out) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cert, indent=2, allow_nan=False) + "\n")


def _certify(name: str, seeds, episodes: int, *, date: str, out, policy) -> dict:
    task = _TASKS[name]
    seeds = [int(seed) for seed in seeds]
    env = _environment(episodes)
    observe = jax.jit(functools.partial(_observe, task))
    run_policy = jax.jit(task.teacher if policy is None else policy)
    per_seed = []
    for seed in seeds:
        goals, trace = _run(env, task, observe, run_policy, seed)
        per_seed.append((seed, _lane_stats(name, trace, goals)))
    cert = _certificate(
        name,
        seeds,
        episodes,
        date=date,
        policy_name="registered" if policy is None else "custom",
        env=_env_block(env, per_seed),
        per_seed=per_seed,
    )
    if out is not None:
        _write(cert, out)
    return cert


def certify_walk_to(
    seeds: Sequence[int],
    episodes_per_seed: int = 256,
    *,
    date: str,
    out: str | Path | None = None,
    policy: Callable[[Any, Any], TeacherOutput] | None = None,
) -> dict:
    """Certificate for walk_to on flat npc_walk, one batched run per seed; written to ``out`` when given.

    ``policy(combat, goals)`` replaces the registered teacher. The tests use it for a broken teacher, and the
    certificate then records ``policy: custom``.
    """

    return _certify("walk_to", seeds, episodes_per_seed, date=date, out=out, policy=policy)


def certify_look_at(
    seeds: Sequence[int],
    episodes_per_seed: int = 256,
    *,
    date: str,
    out: str | Path | None = None,
    policy: Callable[[Any, Any], TeacherOutput] | None = None,
) -> dict:
    """Certificate for look_at on flat npc_walk, one batched run per seed; written to ``out`` when given.

    ``policy`` has the same role as in ``certify_walk_to``.
    """

    return _certify("look_at", seeds, episodes_per_seed, date=date, out=out, policy=policy)


def _verdict(passed) -> str:
    return "not defined" if passed is None else ("PASS" if passed else "FAIL")


def _fmt(value) -> str:
    return "none" if value is None else f"{value:.4g}"


def summary_lines(cert: dict) -> list[str]:
    """A short human-readable summary of a certificate, one line per check."""

    checks = cert["checks"]
    t1, t2, t3 = checks["T1"], checks["T2"], checks["T3"]
    lines = [
        f"{cert['teacher']['name']} {cert['teacher']['version']} ({cert['policy']} policy), seeds {cert['seeds']}, "
        f"{cert['episodes_per_seed']} episodes each, date {cert['date']}",
        f"T1 success: pooled {t1['value']:.4f} ({t1['pooled']['successes']}/{t1['pooled']['episodes']}), "
        f"Wilson 95% [{t1['interval'][0]:.4f}, {t1['interval'][1]:.4f}], bar >= {t1['bar']}: {_verdict(t1['pass'])}",
    ]
    if cert["teacher"]["name"] == "walk_to":
        p = t2["percentiles"]
        lines.append(
            f"T2 efficiency: median ratio {_fmt(p['p50'])}, p90 {_fmt(p['p90'])}, bar <= {t2['bar']}: {_verdict(t2['pass'])}"
        )
        speed, done = t3["speed"], t3["done_distance"]
        lines.append(
            f"T3 obedience: lanes outside {SPEED_BAND:.0%} of v {speed['value']} of {speed['steady_lanes']} steady, "
            f"done outside radius {done['value']}, illegal decisions {t3['illegal_decisions']['value']}: "
            f"{_verdict(t3['pass'])}"
        )
    else:
        p = t2["ticks_to_done"]["percentiles"]
        lines.append(f"T2 efficiency: ticks to done median {_fmt(p['median'])}, p95 {_fmt(p['p95'])}, bar not defined")
        tolerance = t3["done_tolerance"]
        lines.append(
            f"T3 obedience: outside tolerance at done {tolerance['value']} of {tolerance['done_lanes']} done lanes, "
            f"illegal decisions {t3['illegal_decisions']['value']}: {_verdict(t3['pass'])}"
        )
    failures = cert["failure_classes"]["pooled"]
    counts = ", ".join(f"{label} {count}" for label, count in failures["counts"].items())
    lines.append(f"T7 failures {failures['failures']}: {counts}")
    done = cert["done_signal"]
    streak_key = next(key for key in done if key.startswith("first_tick_done_"))
    lines.append(
        f"done signal: first done median {_fmt(done['first_done_tick']['median'])} "
        f"p90 {_fmt(done['first_done_tick']['p90'])}; {streak_key} median {_fmt(done[streak_key]['median'])} "
        f"p90 {_fmt(done[streak_key]['p90'])}; lanes leaving done {done['lanes_leaving_done']}"
    )
    lines.append(f"overall: {_verdict(cert['overall']['pass'])}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Teacher certificate: T1 success, T2 efficiency, T3 obedience, T7 failures.")
    parser.add_argument("--teacher", required=True, choices=sorted(_TASKS))
    parser.add_argument("--seeds", required=True, help="comma-separated seeds; 77001 is the report seed")
    parser.add_argument("--episodes", type=int, default=256, help="lanes per seed")
    parser.add_argument("--out", type=Path, default=None, help="write the certificate JSON to this path")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD, written into the certificate")
    args = parser.parse_args(argv)
    seeds = [int(seed) for seed in args.seeds.split(",")]
    certify = certify_walk_to if args.teacher == "walk_to" else certify_look_at
    cert = certify(seeds, args.episodes, date=args.date, out=args.out)
    for line in summary_lines(cert):
        print(line)
    if args.out is not None:
        print(f"wrote {args.out}")
    return 0 if cert["overall"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
