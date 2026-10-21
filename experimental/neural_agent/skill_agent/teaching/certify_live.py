"""T4 for the walk_to and look_at teachers: the scripted teacher in closed loop on the LIVE 0.7.2 server, paired with
the simulator.

walk_to (the default). Per episode: reset the native env, read the agent's start pose from the native actor rows, draw a
goal (3 to 15 blocks, random direction, run gait, radius 1.0), then on every tick feed the teacher (the observed state)
the native position and body yaw and send its factors: v3, stopping ``MARGIN`` blocks inside the radius (default
``WALK_TO_V3_STOP_MARGIN_BLOCKS``, 0.25), or v2 with ``MARGIN`` 0. v1 read the commanded body yaw, tracked from the
deltas sent. Records success, ticks to success, legality under the native mask, and the path.

look_at (``--teacher look_at``, 2026-10-20 night). Per episode: the same reset, a point drawn by look_at's own sampler
around the native eye and body yaw (``LookAtTasks.sample``: 2 to 10 blocks, within 40 degrees of the body and 30 of
pitch), and on every tick the teacher (v2, the observed state) reads the native body yaw, head yaw
(``head_yaw_degrees``) and look pitch (``pitch_degrees``), the fields the native assembler maps to the learner's head.
Success is the head within ``LOOK_TOLERANCE_DEGREES`` (3) inside ``LOOK_HORIZON_TICKS`` (90).

Both teachers keep acting ``SETTLE_TICKS`` after their first success, and an episode is held when its last
``HELD_TICKS`` ticks stay successful (the students' held success). The same start and goal then run through the
simulator's teacher (simple_tasks, flat npc_walk; look_at from the native start's head angles) for the paired
comparison, on the engine the process has mounted: stock gen 5, or gen 7 when run through ``../runtime/gen7.py``
(``--engine DIR -- experimental.neural_agent.skill_agent.teaching.certify_live OUT.json 24 ...``). Writes one JSON.

Run it as the --then step of a private bridge, on a port of your own (the peer session uses 5556 and 5558)::

    .venv/bin/python HytaleRL/hytale-plugin-0.7.2/tools/launch_bridge_072.py --run-dir <new empty dir> \
        --bridge-port 5570 --game-port 25570 --ready-timeout 600 --then-timeout 5400 \
        --then "bash -c 'PYTHONPATH=$PWD:$PWD/HytaleRL/hytalegym JAX_PLATFORMS=cpu HYTALERL_SERVER_VERSION=0.7.2 \
        nice -n 10 .venv/bin/python -u -m experimental.neural_agent.skill_agent.teaching.certify_live OUT.json 24 \
        [MARGIN] [--teacher look_at]'"

First result, walk_to v1 (2026-10-18, jar 460054A8): 24 of 24 native, 24 of 24 simulated, 0 illegal decisions; native
finished a median 2 ticks sooner because some live ticks ran longer than 1/30 s under load (README 2.15). v2, v3 and
look_at have not run live.
"""

from __future__ import annotations

import argparse
import json
import time
import types
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.neural_skills.look_at.task import (
    LOOK_HORIZON_TICKS,
    LOOK_TOLERANCE_DEGREES,
    LookAtTasks,
)
from experimental.neural_agent.skill_agent.neural_skills.look_at.teacher import EYE_OFFSET, look_at_choices
from experimental.neural_agent.skill_agent.neural_skills.walk_to.teacher import WALK_TO_V3_STOP_MARGIN_BLOCKS, walk_to_choices
from experimental.neural_agent.skill_agent.tasks import simple_tasks as st
from experimental.neural_agent.skill_agent.teaching.teachers import TEACHERS, state_for
from hytalegym.envs.hytale_env import HytaleEnv
from hytalegym.jax.combat import AGENT_ENTITY, ARSENAL_POLICY_ACTION_HEAD_SIZES
from hytalegym.jax.combat.native.evidence import acquire_combat_native_evidence_leases

PORT = 5570
RADIUS, GAIT, SPEED = 1.0, 2, 5.0  # run
SETTLE_TICKS, HELD_TICKS = 30, 5
"""After its first success the teacher keeps acting ``SETTLE_TICKS`` ticks, where walk_to v3 walks on to its stop
radius and idles and look_at holds the head; the episode is held when its last ``HELD_TICKS`` ticks are successful
(the students' held success)."""


def native_pose(env):
    actor = env.native_actor_assembly.raw_wire["actors"][AGENT_ENTITY]
    return np.asarray(actor["position"], dtype=np.float64), float(actor["yaw_degrees"]), actor.get("head_yaw_degrees")


def native_head(env):
    """The native (position, body yaw, head yaw, look pitch); a bridge without ``head_yaw_degrees`` gives the body yaw,
    as the native assembler does."""

    actor = env.native_actor_assembly.raw_wire["actors"][AGENT_ENTITY]
    body = float(actor["yaw_degrees"])
    head = actor.get("head_yaw_degrees")
    return (np.asarray(actor["position"], dtype=np.float64), body, body if head is None else float(head),
            float(actor["pitch_degrees"]))


def _illegal(info, factors) -> int:
    mask = np.asarray(info.get("learner_v3_action_mask", []), dtype=bool)
    if not mask.size:
        return 0
    sizes = ARSENAL_POLICY_ACTION_HEAD_SIZES
    offsets = np.cumsum([0, *sizes[:-1]])
    return int(not all(mask[offsets[h] + factors[h]] for h in range(len(sizes))))


def _stop(tick: int, horizon: int, success_tick: int) -> bool:
    """The episode ends past its horizon without a success, or ``SETTLE_TICKS`` after its first success."""

    return (success_tick < 0 and tick > horizon) or (success_tick >= 0 and tick - 1 - success_tick >= SETTLE_TICKS)


def teacher_factors(position, body_yaw, target_xz, margin):
    assert TEACHERS["walk_to"].state == "observed", "this replay feeds the observed state (v2)"
    pos = np.zeros((1, 2, 3), dtype=np.float32)
    pos[0, AGENT_ENTITY] = position
    combat = types.SimpleNamespace(position=jnp.asarray(pos), desired_body_yaw=jnp.asarray([body_yaw], jnp.float32))
    out = walk_to_choices(combat, jnp.asarray([target_xz], jnp.float32), jnp.asarray([RADIUS], jnp.float32),
                          jnp.asarray([GAIT], jnp.int32), maximum_turn_degrees=45.0, stop_margin_blocks=margin)
    return np.asarray(out.factors[0], dtype=np.int64), bool(out.done[0])


def native_episode(env, seed, rng, margin):
    _, info = env.reset(seed=seed)
    start, body_yaw, head_yaw = native_pose(env)
    distance = rng.uniform(3.0, 15.0)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    target = start[[0, 2]] + distance * np.array([np.sin(angle), np.cos(angle)])
    horizon = int(np.ceil(distance / SPEED * 1.5 * 30)) + 30
    path, illegal, success_tick = [start.tolist()], 0, -1
    for tick in range(1, horizon + SETTLE_TICKS + 1):
        position, native_yaw, _ = native_pose(env)
        factors, done = teacher_factors(position, native_yaw, target, margin)
        if done and success_tick < 0:
            success_tick = tick - 1
        if _stop(tick, horizon, success_tick):
            break
        illegal += _illegal(info, factors)
        _, _, terminated, truncated, info = env.step(factors)
        path.append(native_pose(env)[0].tolist())
        if terminated or truncated:
            break
    final, _, _ = native_pose(env)
    held = success_tick >= 0 and all(np.hypot(p[0] - target[0], p[2] - target[1]) <= RADIUS for p in path[-HELD_TICKS:])
    return {"seed": seed, "start": start.tolist(), "start_body_yaw": body_yaw, "start_head_yaw": head_yaw,
            "target_xz": target.tolist(), "distance": distance, "horizon": horizon, "success_tick": success_tick,
            "final_distance": float(np.hypot(*(final[[0, 2]] - target))), "held": bool(held), "illegal": illegal,
            "path": path, "bridge_sha256": str(info.get("bridge_sha256", ""))[:8]}


def _sim_reset(env, start, body_yaw):
    keys = jnp.stack([jax.random.key(0)])
    return env.reset(keys, jnp.asarray([[start[0], start[2]]], jnp.float32), jnp.asarray([body_yaw], jnp.float32),
                     jnp.asarray([[start[0], start[1], start[2] - 40.0]], jnp.float32))  # the other entity far away


def sim_episode(env, start, body_yaw, target, horizon, margin):
    """The simulator's (success tick, final distance, held), by the native episode's rules."""

    carry = _sim_reset(env, start, body_yaw)
    success_tick, distances = -1, []
    for tick in range(1, horizon + SETTLE_TICKS + 1):
        combat = carry.runtime.combat
        position = np.asarray(combat.position)[0, AGENT_ENTITY]
        distances.append(float(np.hypot(position[0] - target[0], position[2] - target[1])))
        out = walk_to_choices(state_for("walk_to", combat), jnp.asarray([target], jnp.float32),
                              jnp.asarray([RADIUS], jnp.float32),
                              jnp.asarray([GAIT], jnp.int32), maximum_turn_degrees=45.0, stop_margin_blocks=margin)
        if bool(out.done[0]) and success_tick < 0:
            success_tick = tick - 1
        if _stop(tick, horizon, success_tick):
            break
        carry, _, _ = env.step(carry, out.factors, jnp.stack([jax.random.key(tick)]))
    position = np.asarray(carry.runtime.combat.position)[0, AGENT_ENTITY]
    final = float(np.hypot(position[0] - target[0], position[2] - target[1]))
    held = success_tick >= 0 and all(distance <= RADIUS for distance in distances[-HELD_TICKS:])
    return success_tick, final, bool(held)


def look_state(position, body_yaw, head_yaw, head_pitch):
    """The observed state look_at reads (``teachers.observed_state``): the actual head stands for the commanded one."""

    assert TEACHERS["look_at"].state == "observed", "this replay feeds the observed state (v2)"
    pos = np.zeros((1, 2, 3), dtype=np.float32)
    pos[0, AGENT_ENTITY] = position
    yaw = np.zeros((1, 2), dtype=np.float32)
    yaw[0, AGENT_ENTITY] = body_yaw
    head, pitch = jnp.asarray([head_yaw], jnp.float32), jnp.asarray([head_pitch], jnp.float32)
    return types.SimpleNamespace(position=jnp.asarray(pos), yaw=jnp.asarray(yaw), agent_head_yaw=head,
                                 agent_head_pitch=pitch, desired_yaw=head, desired_pitch=pitch)


def look_choices(state, target_xyz):
    out = look_at_choices(state, jnp.asarray([target_xyz], jnp.float32), LOOK_TOLERANCE_DEGREES,
                          maximum_turn_degrees=45.0)
    return np.asarray(out.factors[0], dtype=np.int64), bool(out.done[0])


def native_look_episode(env, seed, key):
    _, info = env.reset(seed=seed)
    start, body_yaw, head_yaw, head_pitch = native_head(env)
    eye = tuple(float(value) for value in start + np.asarray(EYE_OFFSET))
    target = np.asarray(LookAtTasks.sample(key, 1, eye_xyz=eye, body_yaw_degrees=body_yaw).target_xyz[0], np.float64)
    within, illegal, success_tick = [], 0, -1
    for tick in range(1, LOOK_HORIZON_TICKS + SETTLE_TICKS + 1):
        factors, done = look_choices(look_state(*native_head(env)), target)
        within.append(done)
        if done and success_tick < 0:
            success_tick = tick - 1
        if _stop(tick, LOOK_HORIZON_TICKS, success_tick):
            break
        illegal += _illegal(info, factors)
        _, _, terminated, truncated, info = env.step(factors)
        if terminated or truncated:
            break
    _, body, head, pitch = native_head(env)
    return {"seed": seed, "start": start.tolist(), "start_body_yaw": body_yaw, "start_head_yaw": head_yaw,
            "start_head_pitch": head_pitch, "target_xyz": target.tolist(), "success_tick": success_tick,
            "held": bool(success_tick >= 0 and all(within[-HELD_TICKS:])), "final_body_yaw": body,
            "final_head_yaw": head, "final_head_pitch": pitch, "illegal": illegal,
            "bridge_sha256": str(info.get("bridge_sha256", ""))[:8]}


def sim_look_episode(env, start, body_yaw, head_yaw, head_pitch, target):
    """The simulator's (success tick, held) from the native start's body and head angles."""

    carry = _sim_reset(env, start, body_yaw)
    head, pitch = jnp.asarray([head_yaw], jnp.float32), jnp.asarray([head_pitch], jnp.float32)
    combat = carry.runtime.combat._replace(agent_head_yaw=head, agent_head_pitch=pitch, desired_yaw=head,
                                           desired_pitch=pitch)
    carry = carry._replace(runtime=carry.runtime._replace(combat=combat))
    within, success_tick = [], -1
    for tick in range(1, LOOK_HORIZON_TICKS + SETTLE_TICKS + 1):
        out = look_at_choices(state_for("look_at", carry.runtime.combat), jnp.asarray([target], jnp.float32),
                              LOOK_TOLERANCE_DEGREES, maximum_turn_degrees=45.0)
        done = bool(out.done[0])
        within.append(done)
        if done and success_tick < 0:
            success_tick = tick - 1
        if _stop(tick, LOOK_HORIZON_TICKS, success_tick):
            break
        carry, _, _ = env.step(carry, out.factors, jnp.stack([jax.random.key(tick)]))
    return success_tick, bool(success_tick >= 0 and all(within[-HELD_TICKS:]))


def _paired(results) -> dict:
    paired = [(r["success_tick"], r["sim_success_tick"]) for r in results
              if r["success_tick"] >= 0 and r["sim_success_tick"] >= 0]
    return {"episodes": len(results), "native_success": sum(r["success_tick"] >= 0 for r in results),
            "sim_success": sum(r["sim_success_tick"] >= 0 for r in results),
            "native_held": sum(r["held"] for r in results), "sim_held": sum(r["sim_held"] for r in results),
            "illegal_decisions": sum(r["illegal"] for r in results),
            "median_native_minus_sim_ticks": float(np.median([a - b for a, b in paired])) if paired else None,
            "max_abs_native_minus_sim_ticks": int(max(abs(a - b) for a, b in paired)) if paired else None}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out")
    parser.add_argument("episodes", nargs="?", type=int, default=24)
    parser.add_argument("margin", nargs="?", type=float, default=WALK_TO_V3_STOP_MARGIN_BLOCKS,
                        help="walk_to's stop margin in blocks (0.25: v3; 0: v2)")
    parser.add_argument("--teacher", choices=("walk_to", "look_at"), default="walk_to")
    options = parser.parse_args(argv)
    rng = np.random.default_rng(77001)
    results = []
    with acquire_combat_native_evidence_leases((("127.0.0.1", PORT),), purpose=f"t4_{options.teacher}_live",
                                               evidence_kinds=("combat_fidelity", "native_actor_evidence")):
        env = HytaleEnv(task="kill_trork", port=PORT, max_episode_steps=4096, ticks_per_step=1, backend="native",
                        world="flat", npc_role="Kweebec_Razorleaf", combat_target_active=False,
                        learner_v3_profile="iron_sword")
        try:
            for episode in range(options.episodes):
                started = time.perf_counter()
                if options.teacher == "look_at":
                    result = native_look_episode(env, 570057 + episode, jax.random.key(77001 + episode))
                else:
                    result = native_episode(env, 570057 + episode, rng, options.margin)
                result["wall_seconds"] = time.perf_counter() - started
                results.append(result)
                print(json.dumps({k: v for k, v in result.items() if k != "path"}), flush=True)
        finally:
            env.close()
    sim = st.make_flat_npc_environment(1)
    for result in results:
        if options.teacher == "look_at":
            result["sim_success_tick"], result["sim_held"] = sim_look_episode(
                sim, result["start"], result["start_body_yaw"], result["start_head_yaw"], result["start_head_pitch"],
                result["target_xyz"])
        else:
            result["sim_success_tick"], result["sim_final_distance"], result["sim_held"] = sim_episode(
                sim, result["start"], result["start_body_yaw"], result["target_xz"], result["horizon"], options.margin)
    summary = {"teacher": options.teacher, **_paired(results)}
    if options.teacher == "walk_to":
        summary["stop_margin_blocks"] = options.margin
        summary["median_final_distance"] = {"native": float(np.median([r["final_distance"] for r in results])),
                                            "sim": float(np.median([r["sim_final_distance"] for r in results]))}
    print(json.dumps(summary), flush=True)
    with open(options.out, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "episodes": results}, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
