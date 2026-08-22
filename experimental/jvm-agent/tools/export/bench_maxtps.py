"""Max-throughput comparison, normalised to simulated game ticks per second.

`bench_throughput.py` compared *policy steps*, which flatters neither side
fairly: the native env can simulate many ticks per round-trip, so measuring
steps hides whether the cost is per-tick simulation or per-round-trip overhead.

Normalising to ticks/second answers both questions at once:

* if native ticks/s rises with `ticks_per_step`, the bridge is **overhead
  bound** -- the round-trip and the Python-side observation assembly dominate,
  and an in-server agent removes them;
* if it stays flat, the server simulation itself is the ceiling.

JAX with `microticks=1` runs one tick per env step, so its env-steps/s is
already ticks/s.
"""

from __future__ import annotations

import time

import jax
import numpy as np

PROFILE, SEED = "iron_sword", 570057
JAX_BATCHES = (256, 1024, 2048)
NATIVE_TICKS_PER_STEP = (1, 4, 16, 64)
JAX_STEPS = 32


def jax_ticks(batch: int, steps: int = JAX_STEPS):
    from hytalegym.jax.combat import (
        arsenal_runtime_config, default_combat_params, hytale_0_5_7_loadouts,
        neutral_arsenal_policy_action_factors,
    )
    from hytalegym.jax.training import (
        make_arsenal_ppo_environment, open_flat_arsenal_world_capabilities,
    )
    from adk.validation import reset_keys, stream

    env = make_arsenal_ppo_environment(
        default_combat_params(microticks=1, target_active=True),
        arsenal_runtime_config(hytale_0_5_7_loadouts((PROFILE,) * batch)),
        world_capability_provider=open_flat_arsenal_world_capabilities,
    )
    factors = neutral_arsenal_policy_action_factors(batch)
    keys = stream(SEED, steps, batch)

    def rollout(carry, step_keys):
        state, observation = carry
        result = env.step_detailed(state, observation, factors, step_keys)
        return (result[0], result[1]), None

    @jax.jit
    def run(reset):
        state, observation, _ = env.reset(reset)
        (state, observation), _ = jax.lax.scan(rollout, (state, observation), keys)
        return observation

    reset = reset_keys(SEED, batch)
    jax.block_until_ready(run(reset))
    best = None
    for _ in range(2):
        start = time.perf_counter()
        jax.block_until_ready(run(reset))
        elapsed = time.perf_counter() - start
        best = elapsed if best is None else min(best, elapsed)
    return batch * steps / best


def native_ticks(ticks_per_step: int, steps: int = 10):
    from hytalegym.envs import HytaleEnv
    from hytalegym.jax.combat.observation.v3.policy import (
        neutral_arsenal_policy_action_factors,
    )

    env = HytaleEnv(
        task="kill_trork", backend="native", host="127.0.0.1", port=5556,
        world="flat", ticks_per_step=ticks_per_step,
        max_episode_steps=steps + 4,
        npc_role="Kweebec_Razorleaf", combat_target_active=True,
        learner_v3_profile=PROFILE,
    )
    try:
        env.reset(seed=SEED)
        action = np.asarray(
            neutral_arsenal_policy_action_factors(1)[0], dtype=np.int32
        )
        env.step(action)                       # warm
        start = time.perf_counter()
        taken = 0
        for _ in range(steps):
            _, _, terminated, truncated, _ = env.step(action)
            taken += 1
            if terminated or truncated:
                break
        elapsed = time.perf_counter() - start
        return (taken * ticks_per_step) / elapsed, taken, elapsed / taken
    finally:
        try:
            env.close()
        except Exception:
            pass


def main() -> int:
    print("=== JAX (compiled, batched, microticks=1 so steps == ticks) ===")
    print(f"{'batch':>7} {'ticks/s':>14}")
    print("-" * 24)
    jax_best = 0.0
    for batch in JAX_BATCHES:
        try:
            rate = jax_ticks(batch)
            jax_best = max(jax_best, rate)
            print(f"{batch:7d} {rate:14,.0f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{batch:7d}   failed: {type(exc).__name__}: {str(exc)[:50]}",
                  flush=True)

    print("\n=== native Java server (one world, over the socket) ===")
    print(f"{'ticks/step':>11} {'ticks/s':>10} {'s per step':>12}")
    print("-" * 36)
    native_best = 0.0
    for ticks in NATIVE_TICKS_PER_STEP:
        try:
            rate, taken, per_step = native_ticks(ticks)
            native_best = max(native_best, rate)
            print(f"{ticks:11d} {rate:10,.2f} {per_step:11.2f}s"
                  f"   ({taken} steps)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{ticks:11d}   failed: {type(exc).__name__}: {str(exc)[:50]}",
                  flush=True)

    print(f"\nbest JAX    : {jax_best:12,.0f} ticks/s")
    print(f"best native : {native_best:12,.2f} ticks/s")
    if native_best:
        print(f"ratio       : {jax_best / native_best:12,.0f}x")
    print(f"server real-time budget at 30 TPS: 30 ticks/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
