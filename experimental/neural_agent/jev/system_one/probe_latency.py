"""Gate G2 probe on the two timescales of FORMULAS (M8), CPU, in process.

Slow path, off the tick, timed per call:

* ``render``: ``render_executive`` of the CompositionTask E_t (catalogue idle, pursuit,
  evader, each described by its id);
* ``trunk_executive``: ``trunk.encode`` of that one text (h_E, once per executive version);
* ``trunk_catalogue``: ``trunk.encode`` of the K skill descriptions (e_j, once per catalogue).

Fast path, every decision, for batch 1, 8 and 64 (belief rows from
``train_j1.synthetic_dataset``, matching heads from ``init_heads``):

* ``decide``: ``SystemOneJev.__call__``, what a host decider pays per decision, host checks included;
* ``compiled``: the compiled ``instance.fast_path`` alone, through ``jax.block_until_ready``.

The fast path makes no trunk call; the probe checks it with a counting wrapper and
reports ``trunk_calls_during_decisions``. Each timing is called once untimed (compilation,
caches), then ``--repeats`` times; the JSON gives the median and the 95th percentile in
milliseconds. ``--trunk torch:<model>`` loads a ``transformers`` model (the slow path only
changes); nothing is installed here.

Command::

    python -m experimental.neural_agent.jev.system_one.probe_latency --repeats 50
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from adk.architecture.inputs import ActorPolicyInput
from experimental.neural_agent.contracts import (
    ACTION_HEADS,
    PARAM_WIDTH,
    GoalPayload,
    SharedBeliefState,
    executive_arrays,
)
from experimental.neural_agent.jev.system_one.data import render_executive
from experimental.neural_agent.jev.system_one.heads import init_heads
from experimental.neural_agent.jev.system_one.instance import SystemOneJev, fast_path
from experimental.neural_agent.jev.system_one.train_supervised import DEFAULT_SKILL_IDS, load_executive, make_trunk
from experimental.neural_agent.jev.train_j1 import synthetic_dataset

BATCHES: tuple[int, ...] = (1, 8, 64)


class _CountingTrunk:
    """Delegate to a trunk and count its encode calls."""

    def __init__(self, inner: Any) -> None:
        self.dim = inner.dim
        self.calls = 0
        self._inner = inner

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        return self._inner.encode(texts)


def _time_ms(call: Callable[[], Any], repeats: int) -> dict[str, float]:
    """Call once untimed, then ``repeats`` times; return median and p95 in milliseconds."""

    call()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1000.0)
    return {"median": float(np.median(samples)), "p95": float(np.percentile(samples, 95))}


def probe(trunk_spec: str, repeats: int) -> dict[str, Any]:
    """Return the slow-path and fast-path timings for one trunk (module docstring)."""

    trunk = _CountingTrunk(make_trunk(trunk_spec))
    executive = load_executive(None, DEFAULT_SKILL_IDS)
    text = render_executive(DEFAULT_SKILL_IDS, None, executive)
    slow = {
        "render": _time_ms(lambda: render_executive(DEFAULT_SKILL_IDS, None, executive), repeats),
        "trunk_executive": _time_ms(lambda: trunk.encode([text]), repeats),
        "trunk_catalogue": _time_ms(lambda: trunk.encode(list(DEFAULT_SKILL_IDS)), repeats),
        "rendering_chars": len(text),
    }
    jev = SystemOneJev(trunk, init_heads(jax.random.key(0), trunk.dim), DEFAULT_SKILL_IDS)
    jev.set_executive_state(executive)
    fast: dict[str, Any] = {}
    calls_before = trunk.calls
    for batch in BATCHES:
        rows = synthetic_dataset(jax.random.key(batch), batch, num_skills=len(DEFAULT_SKILL_IDS))
        belief = SharedBeliefState(
            jnp.asarray(rows["features"]), jnp.asarray(rows["feature_known"]), jnp.asarray(rows["latent"])
        )
        current = jnp.asarray(rows["current_skill"])
        goal = GoalPayload(
            current, jnp.zeros((batch, PARAM_WIDTH)), jnp.zeros((batch,), jnp.int32), jnp.full((batch,), -1, jnp.int32)
        )
        actor = ActorPolicyInput(jnp.zeros((batch, 4)), jnp.ones((batch, ACTION_HEADS), bool), None, None)
        arrays = executive_arrays(executive, DEFAULT_SKILL_IDS, batch)
        state = jev.initial_state(batch)

        def decide() -> Any:
            return jax.block_until_ready(
                jev(state, belief, goal, actor, jax.random.key(0), decision_context=None, goal_runtime=None,
                    executive=arrays)
            )

        def compiled() -> Any:
            return jax.block_until_ready(
                fast_path(jev.params, jev._h_executive, belief, current, arrays, jev.skill_embeddings,
                          num_skills=len(DEFAULT_SKILL_IDS))
            )

        fast[f"batch_{batch}"] = {"decide": _time_ms(decide, repeats), "compiled": _time_ms(compiled, repeats)}
    return {
        "trunk": trunk_spec,
        "dim": trunk.dim,
        "skill_head": "matching",
        "slow_path_ms": slow,
        "fast_path_ms": fast,
        "trunk_calls_during_decisions": trunk.calls - calls_before,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print the probe's JSON."""

    parser = argparse.ArgumentParser(
        prog="python -m experimental.neural_agent.jev.system_one.probe_latency",
        description="Time J-S1's slow path (per executive version) and fast path (per decision) on CPU (gate G2).",
    )
    parser.add_argument("--trunk", default="hashed", help="hashed (default) or torch:<model>")
    parser.add_argument("--repeats", type=int, default=50, help="timed calls per measurement (default 50)")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    try:
        make_trunk(args.trunk) if args.trunk != "hashed" else None
    except (ImportError, ValueError) as error:
        parser.error(str(error))
    report = {"backend": jax.default_backend(), "machine": platform.platform(), "repeats": args.repeats}
    report.update(probe(args.trunk, args.repeats))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
