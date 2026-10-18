"""The per-tick latency bench of the generalist: the trunk once for all agents, then the generalist forward.

The gate is the one ``bench_tick.py`` states: the per-tick path has a p95 of at most 15 ms at batch 8 (the game's 30 Hz
tick is 33.3 ms). Per batch size B, B agents are built with ``bench_tick._build_agent`` from ``random.Random(seed)``, so
each carries the real per-agent token capacities (``bench_tick.CAPACITIES``). Each agent has six valid request slots
with random verbs, arguments, pointers and granted scopes. Each tick, two stages are timed with ``time.perf_counter``,
each ended by ``jax.block_until_ready``:

* trunk: ``encoder_apply_jit(encoder_config)(params, batch)``, once for all agents;
* generalist: ``generalist_apply_jit(config)(params, encoder_output, token_masks, requests)``.

The total is the sum of the two. Warm-up ticks run the same path and are discarded; the first call compiles. The
requests are built once, outside the timers. The gate is reported, not enforced, so the exit code is 0.

Run from the repository root, with JAX_PLATFORMS=cpu::

    .venv/bin/python -m experimental.neural_agent.skill_agent.bench_generalist --batch 1,8,64 --ticks 300 --warmup 30 --seed 0 [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.bench_tick import CAPACITIES, _build_agent
from experimental.neural_agent.skill_agent.encoder import (
    EncoderConfig,
    encoder_apply_jit,
    init_encoder,
    parameter_count,
    stack_batch,
)
from experimental.neural_agent.skill_agent.generalist import (
    ARG_GROUPS,
    GeneralistConfig,
    GeneralistRequests,
    generalist_apply_jit,
    init_generalist,
)
from experimental.neural_agent.skill_agent.surface import SCOPE_NAMES

__all__ = ["main"]

GATE_BATCH: int = 8
BAR_MS: float = 15.0
"""The gate: the p95 of the whole per-tick path at ``GATE_BATCH`` must not exceed this many milliseconds."""

_STAGES: tuple[str, ...] = ("trunk", "generalist", "total")


def _requests(seed: int, batch_size: int, config: GeneralistConfig) -> GeneralistRequests:
    """Six valid request slots per agent, with random verbs, arguments, pointers and granted scopes."""

    rng = np.random.default_rng(seed)
    shape = (batch_size, config.max_requests)
    return GeneralistRequests(
        verb=jnp.asarray(rng.integers(0, config.verb_count, size=shape, dtype=np.int32)),
        args=jnp.asarray(rng.normal(size=(*shape, config.arg_width)).astype(np.float32)),
        arg_present=jnp.asarray(rng.random((*shape, len(ARG_GROUPS))) < 0.8),
        entity_pointer=jnp.asarray(rng.integers(-1, CAPACITIES["entity"], size=shape, dtype=np.int32)),
        cell_pointer=jnp.asarray(rng.integers(-1, CAPACITIES["block_candidate"], size=shape, dtype=np.int32)),
        scope=jnp.asarray(rng.integers(0, len(SCOPE_NAMES), size=shape, dtype=np.int32)),
        valid=jnp.ones(shape, dtype=jnp.bool_),
    )


def _stats(samples: Sequence[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(values.max()),
    }


def _measure(batch_size: int, ticks: int, warmup: int, seed: int) -> dict[str, Any]:
    """Run ``warmup + ticks`` ticks for ``batch_size`` agents and return the timings of the measured ticks."""

    rng = random.Random(seed)
    agents = [_build_agent(rng) for _ in range(batch_size)]
    batch = stack_batch([agent.packed for agent in agents])
    token_masks = {name: mask for name, (_, mask) in batch.items()}
    encoder_config = EncoderConfig()
    config = GeneralistConfig()
    encoder_params = init_encoder(jax.random.PRNGKey(seed), encoder_config)
    params = init_generalist(jax.random.PRNGKey(seed + 1), config, encoder_config)
    requests = _requests(seed, batch_size, config)
    trunk = encoder_apply_jit(encoder_config)
    generalist = generalist_apply_jit(config)

    samples: dict[str, list[float]] = {stage: [] for stage in _STAGES}
    for tick in range(warmup + ticks):
        started = time.perf_counter()
        encoded = trunk(encoder_params, batch)
        jax.block_until_ready(encoded)
        trunk_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        output = generalist(params, encoded, token_masks, requests)
        jax.block_until_ready(output)
        generalist_ms = (time.perf_counter() - started) * 1000.0

        if tick >= warmup:
            for stage, value in zip(_STAGES, (trunk_ms, generalist_ms, trunk_ms + generalist_ms), strict=True):
                samples[stage].append(value)

    return {
        "batch": batch_size,
        "ticks": ticks,
        "warmup": warmup,
        "parameters": {"generalist": parameter_count(params), "trunk": parameter_count(encoder_params)},
        "stages": {stage: {**_stats(samples[stage]), "samples_ms": samples[stage]} for stage in _STAGES},
    }


def _gate(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    measured = next((result for result in results if result["batch"] == GATE_BATCH), None)
    if measured is None:
        return {"batch": GATE_BATCH, "bar_ms": BAR_MS, "verdict": "not run"}
    p95 = measured["stages"]["total"]["p95_ms"]
    return {"batch": GATE_BATCH, "p95_total_ms": p95, "bar_ms": BAR_MS,
            "verdict": "PASS" if p95 <= BAR_MS else "FAIL"}


def _print_report(results: Sequence[dict[str, Any]], gate: dict[str, Any],
                  load_before: tuple[float, ...], load_after: tuple[float, ...]) -> None:
    counts = results[0]["parameters"]
    print(f"parameters: generalist {counts['generalist']:,}, trunk {counts['trunk']:,}")
    print(f"loadavg before: {load_before[0]:.2f} {load_before[1]:.2f} {load_before[2]:.2f} (1, 5, 15 min)")
    print(f"{'batch':>5}  {'stage':<10}  {'p50 ms':>9}  {'p95 ms':>9}  {'max ms':>9}")
    for result in results:
        for stage in _STAGES:
            stats = result["stages"][stage]
            print(f"{result['batch']:>5}  {stage:<10}  {stats['p50_ms']:>9.3f}  {stats['p95_ms']:>9.3f}  "
                  f"{stats['max_ms']:>9.3f}")
    if gate["verdict"] == "not run":
        print(f"gate p95(total) at batch {GATE_BATCH} = not run (bar {BAR_MS:g} ms)")
    else:
        print(f"gate p95(total) at batch {GATE_BATCH} = {gate['p95_total_ms']:.3f} ms (bar {BAR_MS:g} ms): "
              f"{gate['verdict']}")
    print(f"loadavg after: {load_after[0]:.2f} {load_after[1]:.2f} {load_after[2]:.2f} (1, 5, 15 min)")


def _environment() -> dict[str, Any]:
    return {"python": sys.version.split()[0], "jax": jax.__version__, "backend": jax.default_backend(),
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS")}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-tick latency of the generalist: the trunk, then the forward.")
    parser.add_argument("--batch", default="1,8,64", help="agents per tick, comma-separated (default 1,8,64)")
    parser.add_argument("--ticks", type=int, default=300, help="measured ticks per batch size (default 300)")
    parser.add_argument("--warmup", type=int, default=30, help="discarded warm-up ticks (default 30)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the agents, requests and networks (default 0)")
    parser.add_argument("--json", metavar="PATH", help="also write every number and the raw per-tick samples")
    args = parser.parse_args(argv)
    try:
        batches = [int(item) for item in args.batch.split(",")]
    except ValueError:
        parser.error(f"--batch needs comma-separated integers, got {args.batch!r}")
    if not batches or min(batches) < 1 or args.ticks < 1 or args.warmup < 0:
        parser.error("batch sizes and --ticks must be at least 1, and --warmup at least 0")

    load_before = os.getloadavg()
    results = [_measure(batch, args.ticks, args.warmup, args.seed) for batch in batches]
    load_after = os.getloadavg()
    gate = _gate(results)
    _print_report(results, gate, load_before, load_after)
    if args.json:
        payload = {
            "environment": _environment(),
            "settings": {"batch": batches, "ticks": args.ticks, "warmup": args.warmup, "seed": args.seed,
                         "capacities": CAPACITIES, "max_requests": GeneralistConfig().max_requests,
                         "tick_ms": 1000.0 / 30.0},
            "loadavg": {"before": load_before, "after": load_after},
            "results": results,
            "gate": gate,
        }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
