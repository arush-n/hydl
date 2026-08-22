"""JAX env throughput vs the native Java server, in environment-steps/second.

The question this answers is why training happens in JAX at all. Both runtimes
implement the same combat rules; the difference is not that one is a faster
implementation of a step, it is that JAX runs thousands of *independent*
environments inside one compiled kernel with no I/O, while the server runs one
world behind a socket.

Measured warm, after compilation, with the result blocked on so the timing is
not measuring async dispatch.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

PROFILE, SEED = "iron_sword", 570057
BATCHES = (1, 8, 64, 256)
STEPS = 32


def jax_throughput(batch: int, steps: int = STEPS):
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
    jax.block_until_ready(run(reset))          # compile
    best = None
    for _ in range(3):
        start = time.perf_counter()
        jax.block_until_ready(run(reset))
        elapsed = time.perf_counter() - start
        best = elapsed if best is None else min(best, elapsed)
    return batch * steps / best, best


def native_throughput(steps: int = 64):
    from hytalegym.envs import HytaleEnv
    from hytalegym.jax.combat.observation.v3.policy import (
        neutral_arsenal_policy_action_factors,
    )

    env = HytaleEnv(
        task="kill_trork", backend="native", host="127.0.0.1", port=5556,
        world="flat", ticks_per_step=1, max_episode_steps=steps + 4,
        npc_role="Kweebec_Razorleaf", combat_target_active=True,
        learner_v3_profile=PROFILE,
    )
    try:
        env.reset(seed=SEED)
        action = np.asarray(
            neutral_arsenal_policy_action_factors(1)[0], dtype=np.int32
        )
        env.step(action)                        # warm the path
        start = time.perf_counter()
        taken = 0
        for _ in range(steps):
            _, _, terminated, truncated, _ = env.step(action)
            taken += 1
            if terminated or truncated:
                break
        elapsed = time.perf_counter() - start
        return taken / elapsed, taken, elapsed
    finally:
        try:
            env.close()
        except Exception:
            pass


def main() -> int:
    print("JAX (compiled, batched, no I/O)")
    print(f"{'batch':>6} {'env-steps/s':>14} {'wall for 32 steps':>20}")
    print("-" * 44)
    jax_rates = {}
    for batch in BATCHES:
        try:
            rate, wall = jax_throughput(batch)
            jax_rates[batch] = rate
            print(f"{batch:6d} {rate:14,.0f} {wall * 1000:17.1f} ms")
        except Exception as exc:  # noqa: BLE001
            print(f"{batch:6d}   failed: {type(exc).__name__}: {str(exc)[:60]}")

    print("\nNative Java server (one world, over a socket)")
    rate, taken, elapsed = native_throughput()
    print(f"{'1':>6} {rate:14,.1f}   ({taken} steps in {elapsed:.2f} s)")

    print("\nratio, JAX / native")
    for batch, jax_rate in jax_rates.items():
        print(f"  batch {batch:4d}: {jax_rate / rate:12,.0f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
