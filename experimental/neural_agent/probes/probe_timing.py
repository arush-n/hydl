"""Time compile versus run for the pursuit-candidate hierarchy rollout on CPU.

Reuses the composition in ``probe_events`` (pass-through belief, constant
manager, pursuit_contact candidate actor) on ``combat/baseline``.

Run from the repository root::

    PYTHONPATH=.:HytaleRL/hytalegym JAX_PLATFORMS=cpu \
        .venv/bin/python -m experimental.neural_agent.probes.probe_timing

Writes ``probes/out/timing.json``.
"""

from __future__ import annotations

import json
import time

import jax

from experimental.neural_agent.probes.probe_events import (
    BASELINE,
    OUT,
    build_handle,
    compose,
    timed_rollout,
)

STEPS = 64
BATCHES = (8, 32)
REPEATS = 3
EPISODE_LIMIT = 128


def main() -> dict:
    result = {
        "scene": BASELINE,
        "actor": "pursuit",
        "steps": STEPS,
        "episode_limit_ticks": EPISODE_LIMIT,
        "jax_backend": jax.default_backend(),
        "repeats_after_first_run": REPEATS,
        "batches": {},
    }
    for batch in BATCHES:
        start = time.perf_counter()
        handle = build_handle(BASELINE, batch, EPISODE_LIMIT)
        built_at = time.perf_counter()
        core, initial_state = compose(batch, "pursuit")
        composed_at = time.perf_counter()
        _, lower_s, compile_s, first_run_s, compiled, key = timed_rollout(
            handle.built, core, initial_state, STEPS
        )
        repeats = []
        for _ in range(REPEATS):
            tick = time.perf_counter()
            jax.block_until_ready(compiled(key))
            repeats.append(time.perf_counter() - tick)
        steady = min(repeats)
        entry = {
            "build_seconds": built_at - start,
            "compose_seconds": composed_at - built_at,
            "lower_seconds": lower_s,
            "compile_seconds": compile_s,
            "first_run_seconds": first_run_s,
            "repeat_run_seconds": repeats,
            "best_run_seconds": steady,
            "lane_steps_per_second_best": batch * STEPS / steady,
        }
        result["batches"][str(batch)] = entry
        print(
            f"batch={batch} lower={lower_s:.2f}s compile={compile_s:.2f}s "
            f"first_run={first_run_s:.3f}s best_run={steady:.3f}s "
            f"lane_steps/s={entry['lane_steps_per_second_best']:.0f}",
            flush=True,
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "timing.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    main()
