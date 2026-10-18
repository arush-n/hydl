"""T4 for the walk_to teacher: the scripted teacher in closed loop on the LIVE 0.7.2 server, paired with the simulator.

Per episode: reset the native env, read the agent's start pose from the native actor rows, draw a goal (3 to 15 blocks,
random direction, run gait, radius 1.0), then on every tick feed the teacher the native position and the commanded body
yaw (tracked from the body-yaw deltas sent) and send its factors. Records success, ticks to success, legality under the
native mask, and the path. The same start pose and goal then run through the simulator's teacher (simple_tasks, flat
npc_walk, profile 0) for the paired comparison. Writes one JSON.

Run it as the --then step of a private bridge, on a port of your own (the peer session uses 5556 and 5558)::

    .venv/bin/python HytaleRL/hytale-plugin-0.7.2/tools/launch_bridge_072.py --run-dir <new empty dir> \
        --bridge-port 5570 --game-port 25570 --ready-timeout 600 --then-timeout 5400 \
        --then "bash -c 'PYTHONPATH=$PWD:$PWD/HytaleRL/hytalegym JAX_PLATFORMS=cpu HYTALERL_SERVER_VERSION=0.7.2 \
        nice -n 10 .venv/bin/python -u -m experimental.neural_agent.skill_agent.teaching.certify_live OUT.json 24'"

First result (2026-10-18, jar 460054A8): 24 of 24 native, 24 of 24 simulated, 0 illegal decisions; native finished a
median 2 ticks sooner because some live ticks ran longer than 1/30 s under load (README 2.15).
"""

from __future__ import annotations

import json
import sys
import time
import types

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.tasks import simple_tasks as st
from experimental.neural_agent.skill_agent.teaching.teachers import walk_to_choices
from hytalegym.envs.hytale_env import HytaleEnv
from hytalegym.jax.combat import AGENT_ENTITY, ARSENAL_POLICY_ACTION_HEAD_SIZES
from hytalegym.jax.combat.native.evidence import acquire_combat_native_evidence_leases
from hytalegym.jax.combat.observation.v3.policy.look_deltas import look_delta_values

PORT = 5570
EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 24
OUT = sys.argv[1]
RADIUS, GAIT, SPEED = 1.0, 2, 5.0  # run
YAW_TABLE = np.asarray(look_delta_values(9), dtype=np.float64)


def native_pose(env):
    actor = env.native_actor_assembly.raw_wire["actors"][AGENT_ENTITY]
    return np.asarray(actor["position"], dtype=np.float64), float(actor["yaw_degrees"]), actor.get("head_yaw_degrees")


def teacher_factors(position, commanded_yaw, target_xz):
    pos = np.zeros((1, 2, 3), dtype=np.float32)
    pos[0, AGENT_ENTITY] = position
    combat = types.SimpleNamespace(position=jnp.asarray(pos), desired_body_yaw=jnp.asarray([commanded_yaw], jnp.float32))
    out = walk_to_choices(combat, jnp.asarray([target_xz], jnp.float32), jnp.asarray([RADIUS], jnp.float32),
                          jnp.asarray([GAIT], jnp.int32), maximum_turn_degrees=45.0)
    return np.asarray(out.factors[0], dtype=np.int64), bool(out.done[0])


def native_episode(env, seed, rng):
    _, info = env.reset(seed=seed)
    start, body_yaw, head_yaw = native_pose(env)
    distance = rng.uniform(3.0, 15.0)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    target = start[[0, 2]] + distance * np.array([np.sin(angle), np.cos(angle)])
    horizon = int(np.ceil(distance / SPEED * 1.5 * 30)) + 30
    commanded = body_yaw
    path, illegal, success_tick = [start.tolist()], 0, -1
    for tick in range(1, horizon + 1):
        position, _, _ = native_pose(env)
        factors, done = teacher_factors(position, commanded, target)
        if done:
            success_tick = tick - 1
            break
        mask = np.asarray(info.get("learner_v3_action_mask", []), dtype=bool)
        if mask.size:
            sizes = ARSENAL_POLICY_ACTION_HEAD_SIZES
            offsets = np.cumsum([0, *sizes[:-1]])
            illegal += int(not all(mask[offsets[h] + factors[h]] for h in range(len(sizes))))
        commanded += float(YAW_TABLE[factors[6]])
        _, _, terminated, truncated, info = env.step(factors)
        path.append(native_pose(env)[0].tolist())
        if terminated or truncated:
            break
    final, _, _ = native_pose(env)
    return {"seed": seed, "start": start.tolist(), "start_body_yaw": body_yaw, "start_head_yaw": head_yaw,
            "target_xz": target.tolist(), "distance": distance, "horizon": horizon, "success_tick": success_tick,
            "final_distance": float(np.hypot(*(final[[0, 2]] - target))), "illegal": illegal, "path": path,
            "bridge_sha256": str(info.get("bridge_sha256", ""))[:8]}


def sim_episode(env, start, body_yaw, target, horizon):
    keys = jnp.stack([jax.random.key(0)])
    carry = env.reset(keys, jnp.asarray([[start[0], start[2]]], jnp.float32), jnp.asarray([body_yaw], jnp.float32),
                      jnp.asarray([[start[0], start[1], start[2] - 40.0]], jnp.float32))  # the other entity far away
    for tick in range(1, horizon + 1):
        combat = carry.runtime.combat
        out = walk_to_choices(combat, jnp.asarray([target], jnp.float32), jnp.asarray([RADIUS], jnp.float32),
                              jnp.asarray([GAIT], jnp.int32), maximum_turn_degrees=45.0)
        if bool(out.done[0]):
            return tick - 1
        carry, _, _ = env.step(carry, out.factors, jnp.stack([jax.random.key(tick)]))
    return -1


def main() -> int:
    rng = np.random.default_rng(77001)
    results = []
    with acquire_combat_native_evidence_leases((("127.0.0.1", PORT),), purpose="t4_walk_to_live",
                                               evidence_kinds=("combat_fidelity", "native_actor_evidence")):
        env = HytaleEnv(task="kill_trork", port=PORT, max_episode_steps=4096, ticks_per_step=1, backend="native",
                        world="flat", npc_role="Kweebec_Razorleaf", combat_target_active=False,
                        learner_v3_profile="iron_sword")
        try:
            for episode in range(EPISODES):
                started = time.perf_counter()
                result = native_episode(env, 570057 + episode, rng)
                result["wall_seconds"] = time.perf_counter() - started
                results.append(result)
                print(json.dumps({k: v for k, v in result.items() if k != "path"}), flush=True)
        finally:
            env.close()
    sim = st.make_flat_npc_environment(1)
    for result in results:
        result["sim_success_tick"] = sim_episode(sim, result["start"], result["start_body_yaw"], result["target_xz"],
                                                 result["horizon"])
    native_ok = [r for r in results if r["success_tick"] >= 0]
    paired = [(r["success_tick"], r["sim_success_tick"]) for r in results if r["success_tick"] >= 0 and r["sim_success_tick"] >= 0]
    summary = {"episodes": len(results), "native_success": len(native_ok), "sim_success": sum(r["sim_success_tick"] >= 0 for r in results),
               "illegal_decisions": sum(r["illegal"] for r in results),
               "median_native_minus_sim_ticks": float(np.median([a - b for a, b in paired])) if paired else None,
               "max_abs_native_minus_sim_ticks": int(max(abs(a - b) for a, b in paired)) if paired else None}
    print(json.dumps(summary), flush=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "episodes": results}, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
