"""Decision latency of the middle layer: J0 and J1 inside a compiled call (ADR-001 D2).

Speed is the owner's first requirement for the decision layer (2026-10-17). This
probe times one JEV decision through ``JevRunner`` under ``jax.jit`` on CPU, for
the scripted J0 and the learned J1, at two batch sizes, and the raw J1 network
(``j1_inputs`` + ``apply_j1``). It writes ``probes/out/jev_latency.json``.

    python -m experimental.neural_agent.probes.probe_jev_latency

Host-mode instances are not timed here; the ledger records them: the fake J-LM
backend at 0.70 ms p50 through ``session.run_session`` and live Ollama qwen3:4b
at 471 ms p50 (budget 20 ms).
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from experimental.neural_agent import contracts as c
from experimental.neural_agent.jev.network import NetworkJev, apply_j1, init_j1, j1_inputs
from experimental.neural_agent.jev.runner import JevRunner
from experimental.neural_agent.jev.scripted import ScriptedJev
from experimental.neural_agent.tasks.composition import CompositionTask

SKILL_IDS = ("idle", "pursuit", "evader")
BATCHES = (8, 64)
REPEATS = 300


def _belief(batch: int) -> c.SharedBeliefState:
    return c.SharedBeliefState(
        features=jnp.zeros((batch, c.BELIEF_FEATURE_COUNT), jnp.float32),
        feature_known=jnp.ones((batch, c.BELIEF_FEATURE_COUNT), jnp.bool_),
        latent=jnp.zeros((batch, c.BELIEF_LATENT_WIDTH), jnp.float32),
    )


def _time(fn, *args) -> dict[str, float]:
    jax.block_until_ready(fn(*args))  # compile
    samples = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append((time.perf_counter() - started) * 1e6)
    samples.sort()
    return {"p50_us": statistics.median(samples), "p95_us": samples[int(0.95 * (len(samples) - 1))]}


def main() -> int:
    executive = CompositionTask(tick_budget=512).initial_executive_state(SKILL_IDS)
    params = init_j1(jax.random.key(0), num_skills=len(SKILL_IDS))
    results: dict[str, dict[str, float]] = {}
    for batch in BATCHES:
        belief = _belief(batch)
        goal = c.idle_goal(batch).payload
        key = jax.random.key(1)
        for name, instance in (("j0_scripted", ScriptedJev(SKILL_IDS)), ("j1_network", NetworkJev(params))):
            runner = JevRunner(instance, c.JevBudget(latency_ms=20.0), SKILL_IDS)
            state = runner.set_executive(runner.initial_state(batch), executive)

            @jax.jit
            def decide(state, belief, goal, key, runner=runner):
                decision = runner(state, belief, goal, None, key, decision_context=None, goal_runtime=None)
                return decision.goal.payload.skill_index

            results[f"{name}_runner_jit_batch{batch}"] = _time(decide, state, belief, goal, key)
        arrays = c.executive_arrays(executive, SKILL_IDS, batch)

        @jax.jit
        def network_only(belief, arrays, goal):
            return apply_j1(params, j1_inputs(belief, arrays, goal, num_skills=len(SKILL_IDS)))[0]

        results[f"j1_apply_only_jit_batch{batch}"] = _time(network_only, belief, arrays, goal)
    results["_notes"] = {
        "tick_seconds": 0.045,
        "host_budget_ms": 20.0,
        "repeats": REPEATS,
        "device": jax.devices()[0].platform,
        "host_mode_from_ledger": {"j_lm_fake_p50_ms": 0.70, "ollama_qwen3_4b_p50_ms": 471.0},
    }
    out = Path(__file__).resolve().parent / "out" / "jev_latency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
