"""Gate G2 probe on the two timescales of FORMULAS (M8), CPU, in process.

Slow path, off the tick, timed per call:

* ``render``: ``render_executive`` of the CompositionTask E_t (catalogue idle, pursuit,
  evader, each described by its id);
* ``trunk_executive``: ``trunk.encode`` of that one text (h_E, once per executive version);
* ``trunk_catalogue``: ``trunk.encode`` of the K skill descriptions (e_j, once per catalogue).

Reading (gate G3 reading latency, SYSTEM_ONE.md 8.9), timed per call: ``set_executive_state``
on a new E_t version (the CompositionTask E_t with the version bumped, built before timing),
i.e. the time from a new E_t to a ready cache, for three untrained matching heads:

* ``structured``: the structured encoder alone (``slot_width=SLOT_WIDTH, text_path=False``): slot
  inputs and the compiled ``encode_executive``; a warm-up version before timing compiles it and caches the label;
* ``text``: the text path alone (the scaffold's head): render and one trunk call;
* ``both``: the structured encoder and the text path, J-S1's configuration;
* ``goal_text``: the structured encoder with goal text and no text path (``goal_text=True``, (M12), the
  G3-NL configuration), read through ``set_shared_state`` from a scratchpad whose executive section is that
  E_t with an intent and a condition per goal that are new in every version (three new texts per version,
  encoded in one batched trunk call).

``reading_trunk_calls_per_version`` gives the trunk calls per timed new version for each.

Fast path, every decision, for batch 1, 8 and 64 (belief rows from
``train_j1.synthetic_dataset``, matching heads with the structured encoder and the text path
from ``init_heads``):

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
import dataclasses
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
    ExecutiveState,
    GoalPayload,
    SharedBeliefState,
    executive_arrays,
)
from experimental.neural_agent.jev.system_one.data import render_executive
from experimental.neural_agent.jev.system_one.heads import SLOT_WIDTH, init_heads
from experimental.neural_agent.jev.system_one.instance import SystemOneJev, fast_path
from experimental.neural_agent.jev.system_one.train_supervised import DEFAULT_SKILL_IDS, load_executive, make_trunk
from experimental.neural_agent.jev.train_j1 import synthetic_dataset
from experimental.neural_agent.shared_state import ExecutiveSection, SharedState

BATCHES: tuple[int, ...] = (1, 8, 64)
READINGS: dict[str, dict[str, Any]] = {
    "structured": {"slot_width": SLOT_WIDTH, "text_path": False},
    "text": {},
    "both": {"slot_width": SLOT_WIDTH},
    "goal_text": {"slot_width": SLOT_WIDTH, "text_path": False, "goal_text": True},
}
# (intent, when) of the CompositionTask goals (pursuit, then evader); each version appends its own number.
WORDS: tuple[tuple[str, str], ...] = (("Go after the target and close in", ""), ("Get away from the enemy", "if you get hit"))


class _CountingTrunk:
    """Delegate to a trunk and count its encode calls."""

    def __init__(self, inner: Any) -> None:
        self.dim = inner.dim
        self.calls = 0
        self._inner = inner

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        return self._inner.encode(texts)


def _worded(executive: ExecutiveState, version: int) -> SharedState:
    """A scratchpad whose executive section is ``executive`` at ``version`` with WORDS new to that version."""

    section = ExecutiveSection.from_executive_state(dataclasses.replace(executive, version=version))
    goals = tuple(
        dataclasses.replace(goal, intent=f"{intent}, plan {version}.", when=when and f"{when} during plan {version}")
        for goal, (intent, when) in zip(section.goals, WORDS)
    )
    return SharedState(executive=dataclasses.replace(section, goals=goals))


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
    reading: dict[str, Any] = {}
    reading_calls: dict[str, float] = {}
    for name, options in READINGS.items():
        reader = SystemOneJev(trunk, init_heads(jax.random.key(0), trunk.dim, **options), DEFAULT_SKILL_IDS)
        if options.get("goal_text"):
            versions = iter([_worded(executive, version) for version in range(1, repeats + 3)])
            read = reader.set_shared_state
        else:
            versions = iter([dataclasses.replace(executive, version=version) for version in range(1, repeats + 3)])
            read = reader.set_executive_state
        read(next(versions))  # compiles encode_executive and caches the label embedding
        before = trunk.calls
        reading[name] = _time_ms(lambda: read(next(versions)), repeats)
        reading_calls[name] = (trunk.calls - before) / (repeats + 1)
    jev = SystemOneJev(trunk, init_heads(jax.random.key(0), trunk.dim, slot_width=SLOT_WIDTH), DEFAULT_SKILL_IDS)
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
                          jev._executive_code, jev._control, num_skills=len(DEFAULT_SKILL_IDS))
            )

        fast[f"batch_{batch}"] = {"decide": _time_ms(decide, repeats), "compiled": _time_ms(compiled, repeats)}
    return {
        "trunk": trunk_spec,
        "dim": trunk.dim,
        "skill_head": "matching",
        "slot_width": SLOT_WIDTH,
        "text_path": True,
        "slow_path_ms": slow,
        "reading_ms": reading,
        "reading_trunk_calls_per_version": reading_calls,
        "fast_path_ms": fast,
        "trunk_calls_during_decisions": trunk.calls - calls_before,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print the probe's JSON."""

    parser = argparse.ArgumentParser(
        prog="python -m experimental.neural_agent.jev.system_one.probe_latency",
        description=(
            "Time J-S1's slow path (per executive version), its reading of a new E_t (gate G3) and its "
            "fast path (per decision) on CPU (gate G2)."
        ),
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
