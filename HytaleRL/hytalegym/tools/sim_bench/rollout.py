"""Fixed combat rollout: jit compile time, steady-state throughput, fingerprint.

Measures the engine that ``import hytalegym`` resolves to (point ``PYTHONPATH``
at a ``git archive HEAD`` extraction to measure the committed engine). Rollout:
``default_combat_params(microticks=4, target_active=True)``, ``reset_batch``
with ``jax.random.key(570057)``, then ``step_batch`` on a fixed action stream
(numpy seed 20261017) for ``--steps`` steps. Reported per batch size:

* ``lower_seconds`` / ``compile_seconds``: ``jax.jit(step_batch).lower(...)``
  and ``.compile()`` wall time in a fresh process (no persistent cache);
* ``lane_steps_per_second``: batch x steps / wall time of the compiled steps,
  median of ``--repeats`` passes, each ending in ``block_until_ready``;
* ``fingerprint``: SHA-256 over every output leaf of one pass (the identity
  gate's value: a render must reproduce it byte for byte).

Usage: ``python -m sim_bench.rollout --batches 8,64 --steps 64 --out f.json``
(run from ``HytaleRL/hytalegym/tools``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time


def _digest(tree, h) -> None:
    import jax
    import numpy as np

    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(leaf)
        h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode())
        h.update(np.ascontiguousarray(array).tobytes())


def measure(batch: int, steps: int, repeats: int) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np

    import hytalegym
    from hytalegym.jax.combat import ACTION_SIZE, default_combat_params, reset_batch, step_batch

    params = default_combat_params(microticks=4, target_active=True)
    state0, _ = reset_batch(jax.random.split(jax.random.key(570057), batch), params)
    rng = np.random.default_rng(20261017)
    actions = []
    for _ in range(steps):
        a = np.zeros((batch, ACTION_SIZE), dtype=np.float32)
        a[:, :4] = rng.integers(0, 2, size=(batch, 4))
        a[:, 4] = rng.integers(0, 2, size=batch) * (rng.random(batch) < 0.2)
        a[:, 5] = rng.integers(0, 2, size=batch) * (rng.random(batch) < 0.3)
        a[:, 6] = rng.uniform(-20, 20, size=batch)
        a[:, 7] = rng.uniform(-5, 5, size=batch)
        actions.append(jnp.asarray(a))
    keys = [jax.random.split(jax.random.key(913771 + t), batch) for t in range(steps)]

    jitted = jax.jit(step_batch)
    start = time.perf_counter()
    lowered = jitted.lower(state0, actions[0], keys[0], params)
    lower_seconds = time.perf_counter() - start
    start = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - start

    def run(record: bool):
        h = hashlib.sha256()
        state = state0
        out = None
        for t in range(steps):
            out = compiled(state, actions[t], keys[t], params)
            state = out[0]
            if record:
                _digest(out, h)
        jax.block_until_ready(out)
        return h.hexdigest()

    fingerprint = run(True)
    passes = []
    for _ in range(repeats):
        start = time.perf_counter()
        run(False)
        passes.append(time.perf_counter() - start)
    median = statistics.median(passes)
    return {
        "batch": batch,
        "steps": steps,
        "lower_seconds": round(lower_seconds, 4),
        "compile_seconds": round(compile_seconds, 4),
        "pass_seconds": [round(p, 4) for p in passes],
        "lane_steps_per_second": round(batch * steps / median, 1),
        "fingerprint": fingerprint,
        "engine": os.path.dirname(hytalegym.__file__),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="8,64")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out")
    args = parser.parse_args()
    import jax

    report = {
        "schema": "hytalerl_sim_rollout_baseline_v1",
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "machine": platform.platform(),
        "cpu_count": os.cpu_count(),
        "load_average_start": os.getloadavg(),
        "runs": [measure(int(b), args.steps, args.repeats) for b in args.batches.split(",")],
    }
    report["load_average_end"] = os.getloadavg()
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
