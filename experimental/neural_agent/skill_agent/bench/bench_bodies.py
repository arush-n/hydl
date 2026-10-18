"""The per-tick cost of trained skill bodies: the trunk per call, the trunk once per tick, and two batched layouts.

``bench_tick`` (B1) timed the resolver, the combiner with placeholder bodies and one encoder call over all agents:
p95 2.29 ms at batch 8, against the 15 ms bar of the 30 Hz ruling. A trained body is ``freezing.VerbBody``, a verb of
a ``freezing.SkillNetwork``: the shared trunk (``model/encoder.py``) and the verb's own adapter. The combiner calls each
granted body once per agent per tick (``Combiner.step``, step 6). On raw token sets a body runs its whole verb program
(``freezing.verb_logits``), so a tick of B agents that each hold k skills runs the trunk B * k times. With
``freezing.encode_contexts`` the trunk runs once per tick and each body runs only its adapter (owner ruling
2026-10-18). This bench times both through the combiner, and two batched layouts outside it, over the same agents
and weights. All four neural layouts choose the same factors (``test_sa_bench_bodies.py``):

* ``combiner_stub``: ``Combiner.step`` for every agent with ``StubBody`` bodies that never finish, the bookkeeping
  alone.
* ``as_built``: ``Combiner.step`` for every agent, the bodies on raw token sets: every body call runs the trunk.
* ``encoded``: ``encode_contexts`` once, then ``Combiner.step`` for every agent: every body call runs its adapter on
  the agent's context row.
* ``per_verb``: the bodies' work without the combiner, one ``verb_logits`` call per verb over the agents holding it.
  The trunk still runs once per verb.
* ``shared_trunk``: ``encode_agents`` once, then one ``adapter_logits`` call per verb over its agents' context rows,
  without the combiner.

Each agent holds one verb in each of its first k scopes (``SCOPE_ORDER``), and a verb owns every gen-7 head of its
scope. With ``--verbs shared`` every agent holds the same verb in a scope, so the batched layouts batch across agents;
with ``--verbs distinct`` each agent holds its own verbs, so nothing batches across agents. The weights are random
draws of freezing's shapes (``new_network``): the cost does not depend on their values. The agents and their tokens
are ``bench_tick``'s. Requests never expire (a one-hour horizon) and the bodies never finish, so every tick runs
every body. A tick's time is ``perf_counter`` around its whole work, the read back of every logit included (the
combiner needs Python ints). Warm-up ticks compile and are discarded.

Run from the repository root, with JAX_PLATFORMS=cpu::

    .venv/bin/python -m experimental.neural_agent.skill_agent.bench.bench_bodies --batch 1,8 --skills 1,2,3,6 \\
        --verbs shared,distinct --ticks 200 --warmup 20 --seed 0 [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import jax
import numpy as np

from experimental.neural_agent.skill_agent.bench import bench_tick
from experimental.neural_agent.skill_agent.contract.surface import scope_heads
from experimental.neural_agent.skill_agent.model.encoder import EncoderConfig, stack_batch
from experimental.neural_agent.skill_agent.model.freezing import (
    VerbBody,
    adapter_logits,
    encode_agents,
    encode_contexts,
    new_network,
    verb_logits,
)
from experimental.neural_agent.skill_agent.runtime.combiner import AgentContext, StubBody
from experimental.neural_agent.skill_agent.runtime.library import SkillLibrary

__all__ = ["LAYOUTS", "SCOPE_ORDER", "VERB_MODES", "layout_choices", "main"]

SCOPE_ORDER: tuple[str, ...] = ("locomotion", "gaze", "hands", "posture_defence", "abilities_magic", "interaction")
"""The scopes an agent's k skills occupy, first k in this order."""

LAYOUTS: tuple[str, ...] = ("combiner_stub", "as_built", "encoded", "per_verb", "shared_trunk")
HORIZON_SECONDS: float = 3600.0
VERB_MODES: tuple[str, ...] = ("shared", "distinct")


def _choices(logits: Mapping[str, jax.Array]) -> list[dict[str, int]]:
    """Per row, head -> the argmax choice, after one read back of every head's logits."""

    host = jax.device_get(dict(logits))
    rows = next(iter(host.values())).shape[0]
    return [{head: int(np.argmax(values[row])) for head, values in host.items()} for row in range(rows)]


@dataclass
class _Setup:
    """One configuration: the number of calls per tick and the callable of each layout."""

    calls: dict[str, int]
    run: dict[str, Callable[[int], list[dict[str, int]]]]


def _setup(batch: int, skills: int, verbs: str, seed: int) -> _Setup:
    if verbs not in VERB_MODES:
        raise ValueError(f"verbs must be one of {VERB_MODES}, got {verbs!r}")
    if not 1 <= skills <= len(SCOPE_ORDER):
        raise ValueError(f"skills must be in 1..{len(SCOPE_ORDER)}, got {skills}")
    rng = random.Random(seed)
    agents = [bench_tick._build_agent(rng) for _ in range(batch)]
    scopes = SCOPE_ORDER[:skills]
    held = [tuple(f"bench_{scope}" if verbs == "shared" else f"bench_{scope}_{index}" for scope in scopes)
            for index in range(batch)]
    verb_scope = {verb: scope for row in held for verb, scope in zip(row, scopes)}
    network = new_network(jax.random.PRNGKey(seed), EncoderConfig(),
                          {verb: scope_heads(scope) for verb, scope in verb_scope.items()})
    trained, stub = SkillLibrary(), SkillLibrary()
    for verb, scope in sorted(verb_scope.items()):
        trained.create(verb, description=f"a trained-shape {scope} skill", scopes=(scope,),
                       body=VerbBody(network, verb))
        stub.create(verb, description=f"a stub {scope} skill", scopes=(scope,),
                    body=StubBody(layout="gen7", heads=scope_heads(scope)))
    requests = [[trained.request(verb, agent=index, request_id=number + 1, tick=0, horizon_seconds=HORIZON_SECONDS)
                 for number, verb in enumerate(row)] for index, row in enumerate(held)]
    contexts = {index: AgentContext(world_bounds=bench_tick.WORLD_BOUNDS, known_entities=bench_tick.KNOWN_ENTITIES,
                                    observation_tokens=agent.packed) for index, agent in enumerate(agents)}
    rows = [agent.packed for agent in agents]
    groups = {verb: tuple(index for index, row in enumerate(held) if verb in row) for verb in sorted(verb_scope)}

    def combined(library: SkillLibrary, encode: bool) -> Callable[[int], list[dict[str, int]]]:
        combiner = library.combiner("gen7")
        names = combiner.head_names

        def step(tick: int) -> list[dict[str, int]]:
            current = encode_contexts(network, contexts) if encode else contexts
            out = []
            for index in range(batch):
                result = combiner.step(tick, index, requests[index], current[index])
                if any(status.kind not in ("accepted", "running") for status in result.statuses):
                    raise RuntimeError(f"agent {index} tick {tick}: {[s.kind for s in result.statuses]}")
                owned = {head for verb in held[index] for head in scope_heads(verb_scope[verb])}
                out.append({name: int(choice) for name, choice in zip(names, result.factors) if name in owned})
            return out

        return step

    def per_verb(tick: int) -> list[dict[str, int]]:
        out: list[dict[str, int]] = [{} for _ in range(batch)]
        everyone = stack_batch(rows) if verbs == "shared" else None
        for verb, members in groups.items():
            tokens = everyone if everyone is not None else stack_batch([rows[member] for member in members])
            for member, choice in zip(members, _choices(verb_logits(network, verb, tokens))):
                out[member].update(choice)
        return out

    def shared_trunk(tick: int) -> list[dict[str, int]]:
        out: list[dict[str, int]] = [{} for _ in range(batch)]
        context = np.stack([observation.context for observation in encode_agents(network, rows)])
        for verb, members in groups.items():
            selected = context if len(members) == batch else context[np.asarray(members)]
            for member, choice in zip(members, _choices(adapter_logits(network, verb, selected))):
                out[member].update(choice)
        return out

    neural = batch * skills
    return _Setup(
        calls={"combiner_stub": 0, "as_built": neural, "encoded": 1 + neural, "per_verb": len(groups),
               "shared_trunk": 1 + len(groups)},
        run={"combiner_stub": combined(stub, encode=False), "as_built": combined(trained, encode=False),
             "encoded": combined(trained, encode=True), "per_verb": per_verb, "shared_trunk": shared_trunk},
    )


def layout_choices(batch: int, skills: int, verbs: str, seed: int = 0) -> dict[str, list[dict[str, int]]]:
    """Each layout's choices on tick 0, per agent: head -> choice. The four neural layouts must agree."""

    setup = _setup(batch, skills, verbs, seed)
    return {layout: setup.run[layout](0) for layout in LAYOUTS}


def _stats(samples: Sequence[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    return {"p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
            "max_ms": float(values.max())}


def _measure(batch: int, skills: int, verbs: str, ticks: int, warmup: int, seed: int) -> dict[str, Any]:
    setup = _setup(batch, skills, verbs, seed)
    layouts: dict[str, Any] = {}
    for layout in LAYOUTS:
        samples = []
        for tick in range(warmup + ticks):
            started = time.perf_counter()
            setup.run[layout](tick)
            elapsed = (time.perf_counter() - started) * 1000.0
            if tick >= warmup:
                samples.append(elapsed)
        layouts[layout] = {"calls_per_tick": setup.calls[layout], **_stats(samples), "samples_ms": samples}
    stub_p95 = layouts["combiner_stub"]["p95_ms"]
    return {"batch": batch, "skills": skills, "verbs": verbs, "ticks": ticks, "warmup": warmup, "layouts": layouts,
            "batched_with_bookkeeping_p95_ms": {layout: stub_p95 + layouts[layout]["p95_ms"]
                                                for layout in ("per_verb", "shared_trunk")}}


def _print_report(results: Sequence[dict[str, Any]]) -> None:
    print(f"{'batch':>5} {'skills':>6} {'verbs':<8} {'layout':<13} {'calls':>5} {'p50 ms':>8} {'p95 ms':>8} "
          f"{'max ms':>8}")
    for result in results:
        for layout in LAYOUTS:
            row = result["layouts"][layout]
            print(f"{result['batch']:>5} {result['skills']:>6} {result['verbs']:<8} {layout:<13} "
                  f"{row['calls_per_tick']:>5} {row['p50_ms']:>8.3f} {row['p95_ms']:>8.3f} {row['max_ms']:>8.3f}")
        extra = ", ".join(f"{layout} {value:.3f}" for layout, value in result["batched_with_bookkeeping_p95_ms"].items())
        print(f"       p95 of a batched layout plus the bookkeeping's p95: {extra} (bar {bench_tick.BAR_MS:g} ms)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-tick cost of trained skill bodies: trunk per call, once, batched.")
    parser.add_argument("--batch", default="1,8", help="agents per tick, comma-separated (default 1,8)")
    parser.add_argument("--skills", default="1,2,3,6", help="skills each agent holds, comma-separated (default 1,2,3,6)")
    parser.add_argument("--verbs", default="shared,distinct", help="shared, distinct or both (default both)")
    parser.add_argument("--ticks", type=int, default=200, help="measured ticks per layout (default 200)")
    parser.add_argument("--warmup", type=int, default=20, help="discarded warm-up ticks (default 20)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the agents and the weights (default 0)")
    parser.add_argument("--json", metavar="PATH", help="also write every number and the raw per-tick samples")
    args = parser.parse_args(argv)
    try:
        batches = [int(item) for item in args.batch.split(",")]
        skills = [int(item) for item in args.skills.split(",")]
    except ValueError:
        parser.error("--batch and --skills need comma-separated integers")
    modes = [item.strip() for item in args.verbs.split(",")]
    if not batches or min(batches) < 1 or args.ticks < 1 or args.warmup < 0:
        parser.error("batch sizes and --ticks must be at least 1, and --warmup at least 0")
    if not skills or min(skills) < 1 or max(skills) > len(SCOPE_ORDER):
        parser.error(f"--skills takes values in 1..{len(SCOPE_ORDER)}")
    if not modes or set(modes) - set(VERB_MODES):
        parser.error(f"--verbs takes {VERB_MODES}")

    load_before = os.getloadavg()
    results = []
    for batch in batches:
        for count in skills:
            for mode in modes:
                if batch == 1 and mode == "distinct" and "shared" in modes:
                    continue  # one agent: distinct and shared verbs are the same configuration
                results.append(_measure(batch, count, mode, args.ticks, args.warmup, args.seed))
    load_after = os.getloadavg()
    _print_report(results)
    print(f"load averages (1, 5, 15 min): before {load_before[0]:.2f} {load_before[1]:.2f} {load_before[2]:.2f}, "
          f"after {load_after[0]:.2f} {load_after[1]:.2f} {load_after[2]:.2f}; {os.cpu_count()} CPUs")
    if args.json:
        payload = {
            "environment": {"python": sys.version.split()[0], "jax": jax.__version__, "backend": jax.default_backend(),
                            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"), "cpus": os.cpu_count(),
                            "load_before": load_before, "load_after": load_after},
            "settings": {"batch": batches, "skills": skills, "verbs": modes, "ticks": args.ticks,
                         "warmup": args.warmup, "seed": args.seed, "scope_order": SCOPE_ORDER,
                         "bar_ms": bench_tick.BAR_MS},
            "results": results,
        }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
