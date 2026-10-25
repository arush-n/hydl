"""The teacher checks of the track and follow plans (their T1), with no training: the scripted teachers against a
scripted mover (``tasks/simple_tasks.py``, ``Mover``) on the flat world, per speed band. One JSON line per check.

track: look_at's teacher (``neural_skills/look_at/teacher.py``) aims the head at entity 1's position each tick, with
no lead and with a lead of ``LEAD_TICKS`` ticks (position plus velocity times the lead). The check is look_at's own
success test (``neural_skills/look_at/task.py``) after each step, at its 3-degree tolerance: the head within tolerance
of facing the mover. ``share_in_view_after_1s`` leaves out the first second (the head's acquisition).
``share_in_head_window`` is the ceiling the still body allows: the mover inside the head's +-45 degree window plus the
tolerance, the body not turning.

follow: walk_to v3's teacher (``neural_skills/walk_to/teacher.py``, stop margin 0.25) walks the learner to a point 3.5
blocks behind the mover (the middle of the 2-to-5-block band) at radius 1, at a gait, and idles inside its stop radius.
``share_in_band`` is the share of ticks, after each step, with the planar distance from learner to mover in 2 to 5
blocks. The learner starts 3.5 blocks behind the mover's first heading.

Every check runs ``--lanes`` independent episodes of ``--ticks`` decision ticks (1/30 s each). The mover is a random walk
of 8 waypoints, looped, within 8 blocks of the spawn, at one gait's speed (walk 1.5, sneak 2.75, run 5.0 blocks a
second); each band's paths serve all its checks.

Run from the repo root:
    python -m experimental.neural_agent.skill_agent.bench.mover_checks [--lanes 256] [--ticks 300] [--seed 0]
"""

from __future__ import annotations

import argparse
import functools
import json

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.neural_skills.look_at.task import (
    LOOK_HORIZON_TICKS,
    LOOK_TOLERANCE_DEGREES,
    LookAtGoals,
    LookAtTasks,
)
from experimental.neural_agent.skill_agent.neural_skills.look_at.teacher import HEAD_WINDOW_DEGREES, look_at_choices
from experimental.neural_agent.skill_agent.neural_skills.walk_to.teacher import (
    WALK_TO_V3_STOP_MARGIN_BLOCKS,
    walk_to_choices,
)
from experimental.neural_agent.skill_agent.tasks import simple_tasks as st
from hytalegym.jax.combat.types import (
    AGENT_ENTITY,
    GAIT_RUN,
    GAIT_SNEAK,
    GAIT_SPRINT,
    GAIT_WALK,
    TARGET_ENTITY,
)

MAXIMUM_TURN_DEGREES = 45.0
LEAD_TICKS = 3
BEHIND_BLOCKS = 3.5
WALK_RADIUS_BLOCKS = 1.0
BAND_LOW_BLOCKS, BAND_HIGH_BLOCKS = 2.0, 5.0
MOVER_WAYPOINTS, MOVER_RADIUS_BLOCKS = 8, 8.0
BANDS = {"walk": GAIT_WALK, "sneak": GAIT_SNEAK, "run": GAIT_RUN}
FOLLOW_GAITS = {"run": GAIT_RUN, "sprint": GAIT_SPRINT}


def _mover_height(env: st.FlatTaskEnvironment) -> jax.Array:
    """Entity 1's spawn [B, 3] for a reset: the floor's height; the mover sets its x and z."""

    return jnp.broadcast_to(jnp.asarray([0.0, float(env.params.floor_y), 0.0], dtype=jnp.float32), (env.batch, 3))


def _in_head_window(combat, target: jax.Array) -> jax.Array:
    """Whether ``target`` can be in view: its bearing lies within the head's +-45 degree window around the body, plus
    look_at's tolerance (the head sits at the window's edge and the success test allows the tolerance beyond it)."""

    reach = HEAD_WINDOW_DEGREES + LOOK_TOLERANCE_DEGREES
    eye = combat.position[:, AGENT_ENTITY] + jnp.asarray([0.0, st.AGENT_EYE_HEIGHT, 0.0], dtype=jnp.float32)
    offset = target - eye
    dx, dy, dz = offset[:, 0], offset[:, 1], offset[:, 2]
    yaw = jnp.rad2deg(jnp.arctan2(-dx, -dz))
    pitch = jnp.rad2deg(jnp.arctan2(dy, jnp.hypot(dx, dz)))
    yaw_offset = jnp.mod(yaw - combat.yaw[:, AGENT_ENTITY] + 180.0, 360.0) - 180.0
    return (jnp.abs(yaw_offset) <= reach) & (jnp.abs(pitch) <= reach)


def run_track(env: st.FlatTaskEnvironment, mover: st.Mover, lead_ticks: int, ticks: int, seed: int):
    """Per tick and lane: the look_at success after the step, and whether the mover is in the head's window."""

    moving = env.with_mover(mover)
    lanes = env.batch
    spawn = jnp.broadcast_to(jnp.asarray(st.DEFAULT_SPAWN_XZ, dtype=jnp.float32), (lanes, 2))
    carry = moving.reset(
        jax.random.split(jax.random.key(seed), lanes),
        spawn,
        jnp.zeros((lanes,), dtype=jnp.float32),
        _mover_height(env),
    )
    teacher = jax.jit(functools.partial(look_at_choices, maximum_turn_degrees=MAXIMUM_TURN_DEGREES))
    tolerance = jnp.full((lanes,), LOOK_TOLERANCE_DEGREES, dtype=jnp.float32)
    horizon = jnp.full((lanes,), LOOK_HORIZON_TICKS, dtype=jnp.int32)
    step_keys = jax.random.split(jax.random.key(seed + 1), ticks)
    in_view, in_window = [], []
    for tick in range(ticks):
        combat = carry.runtime.combat
        velocity = combat.velocity[:, TARGET_ENTITY]
        aim = combat.position[:, TARGET_ENTITY] + velocity * (lead_ticks / st.TICKS_PER_SECOND)
        out = teacher(combat, aim, LOOK_TOLERANCE_DEGREES)
        carry, _observation, _info = moving.step(carry, out.factors, jax.random.split(step_keys[tick], lanes))
        combat = carry.runtime.combat
        target = combat.position[:, TARGET_ENTITY]
        goals = LookAtGoals(target_xyz=target, tolerance_degrees=tolerance, horizon_ticks=horizon)
        in_view.append(np.asarray(LookAtTasks.success(combat, goals)))
        in_window.append(np.asarray(_in_head_window(combat, target)))
    return np.stack(in_view), np.stack(in_window)


def run_follow(env: st.FlatTaskEnvironment, mover: st.Mover, gait: int, ticks: int, seed: int) -> np.ndarray:
    """Per tick and lane: whether the learner is 2 to 5 planar blocks from the mover, after the step."""

    moving = env.with_mover(mover)
    lanes = env.batch
    zeros = jnp.zeros((lanes,), dtype=jnp.int32)
    start_xz, start_yaw, _ = st.mover_path(mover, zeros)
    start_radians = jnp.deg2rad(start_yaw)
    behind = jnp.stack((-jnp.sin(start_radians), -jnp.cos(start_radians)), axis=1) * BEHIND_BLOCKS
    carry = moving.reset(
        jax.random.split(jax.random.key(seed), lanes),
        start_xz - behind,
        start_yaw,
        _mover_height(env),
    )
    teacher = jax.jit(
        functools.partial(
            walk_to_choices,
            maximum_turn_degrees=MAXIMUM_TURN_DEGREES,
            stop_margin_blocks=WALK_TO_V3_STOP_MARGIN_BLOCKS,
        )
    )
    radius = jnp.full((lanes,), WALK_RADIUS_BLOCKS, dtype=jnp.float32)
    gaits = jnp.full((lanes,), gait, dtype=jnp.int32)
    step_keys = jax.random.split(jax.random.key(seed + 1), ticks)
    in_band = []
    for tick in range(ticks):
        combat = carry.runtime.combat
        person = combat.position[:, TARGET_ENTITY]
        person_radians = jnp.deg2rad(combat.yaw[:, TARGET_ENTITY])
        forward = jnp.stack((-jnp.sin(person_radians), -jnp.cos(person_radians)), axis=1)
        target_xz = person[:, [0, 2]] - BEHIND_BLOCKS * forward
        out = teacher(combat, target_xz, radius, gaits)
        carry, _observation, _info = moving.step(carry, out.factors, jax.random.split(step_keys[tick], lanes))
        combat = carry.runtime.combat
        gap = combat.position[:, AGENT_ENTITY][:, [0, 2]] - combat.position[:, TARGET_ENTITY][:, [0, 2]]
        distance = jnp.hypot(gap[:, 0], gap[:, 1])
        in_band.append(np.asarray((distance >= BAND_LOW_BLOCKS) & (distance <= BAND_HIGH_BLOCKS)))
    return np.stack(in_band)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--lanes", type=int, default=256)
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = st.make_flat_npc_environment(args.lanes)
    settle = int(st.TICKS_PER_SECOND)
    for index, (band, gait) in enumerate(BANDS.items()):
        mover = st.sample_random_walk_mover(
            jax.random.key(args.seed * 100 + index),
            args.lanes,
            speed_band=st.gait_band(gait),
            spawn_xz=st.DEFAULT_SPAWN_XZ,
            waypoints=MOVER_WAYPOINTS,
            radius=MOVER_RADIUS_BLOCKS,
        )
        speed = float(st.gait_speed(gait))
        for lead in (0, LEAD_TICKS):
            in_view, in_window = run_track(env, mover, lead, args.ticks, args.seed * 100 + index)
            print(json.dumps({
                "plan": "track", "band": band, "speed_blocks_per_second": speed, "lead_ticks": lead,
                "lanes": args.lanes, "ticks": args.ticks,
                "share_in_view": float(in_view.mean()),
                "share_in_view_after_1s": float(in_view[settle:].mean()),
                "share_in_head_window": float(in_window.mean()),
            }), flush=True)
        for follow_name, follow_gait in FOLLOW_GAITS.items():
            in_band = run_follow(env, mover, follow_gait, args.ticks, args.seed * 100 + index)
            print(json.dumps({
                "plan": "follow", "band": band, "speed_blocks_per_second": speed, "learner_gait": follow_name,
                "lanes": args.lanes, "ticks": args.ticks,
                "share_in_band": float(in_band.mean()),
                "share_in_band_after_1s": float(in_band[settle:].mean()),
            }), flush=True)


if __name__ == "__main__":
    main()
