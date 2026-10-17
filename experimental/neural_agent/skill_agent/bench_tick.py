"""The per-tick latency bench of the lower layer: resolver, combiner and encoder at batch 1, 8 and 64.

The owner's 30 Hz ruling (commit 6b00064) paces the agent to the game clock: a tick is 33.3 ms, and the gate is that
the whole per-tick path, resolver + combiner + encoder, has a p95 of at most 15 ms at batch 8. The gate is reported,
not enforced, so the exit code is 0.

Per batch size B, B agents are built once from ``random.Random(seed)``: a pose, 16 entity tokens (handles 0 to 15,
with the look target marked by ``references.mark``), 16 block candidates, 36 inventory slots, 64 geometry tokens, one
self token and four kit tokens, packed with ``encoder.pack_tokens`` and stacked on the device. Two created verbs,
``walk`` and ``look``, carry placeholder bodies, so the combiner runs its whole path: check, grant, body and status.
Each tick, three stages are timed with ``time.perf_counter``:

* resolver: per agent, ``references.resolve_entity`` of the look target and ``references.to_agent_frame`` of the walk
  target;
* combiner: per agent, ``combiner.step(tick, agent, [walk, look], context)``;
* encoder: ``encoder_apply_jit(config)(params, batch)``, with ``jax.block_until_ready`` on the context.

The total is the sum of the three. A request whose status is done is re-issued with a fresh id on the next tick; the
re-issue and the request construction sit outside the timers, as the middle layer's work does. Warm-up ticks run the
same path and are discarded; the first encoder call compiles.

Run from the repository root, with JAX_PLATFORMS=cpu::

    .venv/bin/python -m experimental.neural_agent.skill_agent.bench_tick --batch 1,8,64 --ticks 300 --warmup 30 --seed 0 [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Sequence

import jax
import numpy as np

from experimental.neural_agent.requests import ActionRequest
from experimental.neural_agent.skill_agent import references
from experimental.neural_agent.skill_agent.combiner import AgentContext
from experimental.neural_agent.skill_agent.encoder import (
    TRUNK_TYPES,
    EncoderConfig,
    encoder_apply_jit,
    init_encoder,
    pack_tokens,
    stack_batch,
)
from experimental.neural_agent.skill_agent.library import SkillLibrary
from experimental.neural_agent.skill_agent.surface import (
    ENTITY_ROLES,
    FACES,
    FLUIDS,
    BlockCandidateToken,
    EntityToken,
    GeometryToken,
    InventorySlotToken,
    KitToken,
    SelfToken,
)

__all__ = ["main"]

GATE_BATCH: int = 8
BAR_MS: float = 15.0
"""The gate: the p95 of the whole per-tick path at ``GATE_BATCH`` must not exceed this many milliseconds."""

CAPACITIES: dict[str, int] = {"entity": 16, "block_candidate": 16, "inventory_slot": 36, "geometry": 64, "self": 1, "kit": 4}
"""Tokens per agent and type, which are also the packed slots: every slot is filled."""

WORLD_BOUNDS = ((-64.0, 0.0, -64.0), (64.0, 128.0, 64.0))
KNOWN_ENTITIES: frozenset[int] = frozenset(range(16))
_STAGES: tuple[str, ...] = ("resolver", "combiner", "encoder", "total")


def _entity_token(rng: random.Random, handle: int) -> EntityToken:
    return EntityToken(
        handle=handle,
        role=rng.choice(ENTITY_ROLES),
        position=(rng.uniform(-16.0, 16.0), rng.uniform(-4.0, 4.0), rng.uniform(-16.0, 16.0)),
        velocity=(rng.uniform(-1.0, 1.0), 0.0, rng.uniform(-1.0, 1.0)),
        visible=rng.random() < 0.8,
        last_seen_age_ticks=rng.randrange(0, 300),
        health_fraction=rng.random(),
    )


def _block_token(rng: random.Random) -> BlockCandidateToken:
    return BlockCandidateToken(
        cell=(rng.randrange(-32, 32), rng.randrange(0, 64), rng.randrange(-32, 32)),
        offset=(rng.uniform(-4.0, 4.0), rng.uniform(-4.0, 4.0), rng.uniform(-4.0, 4.0)),
        face=rng.choice(FACES),
        block_kind=rng.randrange(1, 200),
    )


def _slot_token(rng: random.Random, slot: int) -> InventorySlotToken:
    return InventorySlotToken(
        slot=slot, item_id=rng.randrange(0, 60), quantity=rng.randrange(0, 65), durability=rng.random()
    )


def _geometry_token(rng: random.Random) -> GeometryToken:
    return GeometryToken(
        offset=(rng.uniform(-8.0, 8.0), rng.uniform(-8.0, 8.0), rng.uniform(-8.0, 8.0)),
        solid=rng.random() < 0.5,
        climbable=rng.random() < 0.1,
        fluid=rng.choice(FLUIDS),
        hazard=rng.random() < 0.05,
        top_walkable=rng.random() < 0.5,
    )


def _self_token(rng: random.Random, yaw: float) -> SelfToken:
    return SelfToken(
        health=rng.random(),
        stamina=rng.random(),
        mana=rng.random(),
        signature_energy=rng.random(),
        oxygen=rng.random(),
        held_durability=rng.random(),
        grounded=rng.random() < 0.9,
        yaw=yaw,
        pitch=rng.uniform(-1.5, 1.5),
        speed=rng.uniform(0.0, 6.0),
    )


def _kit_token(rng: random.Random) -> KitToken:
    thresholds = tuple(sorted(rng.uniform(0.1, 1.5) for _ in range(rng.randrange(0, 6))))
    return KitToken(
        combo_window_seconds=rng.uniform(0.0, 0.6),
        hold_thresholds_seconds=thresholds,
        charged_tiers=rng.randrange(0, 6),
        stamina_cost=rng.uniform(0.0, 0.3),
        mana_cost=rng.uniform(0.0, 0.3),
        signature_cost=rng.uniform(0.0, 0.3),
        reach_blocks=rng.uniform(2.0, 6.0),
        projectile_launch=rng.uniform(0.0, 30.0),
        projectile_gravity=rng.uniform(0.0, 1.0),
    )


@dataclass(frozen=True)
class _Agent:
    """One agent's pose and targets, its entity tokens (look target marked) and its packed token sets."""

    position: tuple[float, float, float]
    yaw: float
    look_handle: int
    walk_target: tuple[float, float, float]
    entities: tuple[EntityToken, ...]
    packed: dict[str, tuple[np.ndarray, np.ndarray]]


def _build_agent(rng: random.Random) -> _Agent:
    position = (rng.uniform(-32.0, 32.0), rng.uniform(0.0, 64.0), rng.uniform(-32.0, 32.0))
    yaw = rng.uniform(-3.0, 3.0)
    look_handle = rng.randrange(CAPACITIES["entity"])
    walk_target = (rng.uniform(-32.0, 32.0), rng.uniform(0.0, 64.0), rng.uniform(-32.0, 32.0))
    entities = references.mark(tuple(_entity_token(rng, handle) for handle in range(CAPACITIES["entity"])), look_handle)
    tokens = {
        "entity": entities,
        "block_candidate": [_block_token(rng) for _ in range(CAPACITIES["block_candidate"])],
        "inventory_slot": [_slot_token(rng, slot) for slot in range(CAPACITIES["inventory_slot"])],
        "geometry": [_geometry_token(rng) for _ in range(CAPACITIES["geometry"])],
        "self": [_self_token(rng, yaw)],
        "kit": [_kit_token(rng) for _ in range(CAPACITIES["kit"])],
    }
    packed = {name: pack_tokens(tokens[name], CAPACITIES[name], cls) for name, cls in TRUNK_TYPES}
    return _Agent(position, yaw, look_handle, walk_target, entities, packed)


def _library() -> SkillLibrary:
    library = SkillLibrary()
    library.create("walk", description="walk to a point in the world", scopes=("locomotion",),
                   required=("position",), scalars=("radius",))
    library.create("look", description="look at an entity", scopes=("gaze",), required=("entity",))
    return library


def _request(library: SkillLibrary, verb: str, agent_index: int, agent: _Agent, request_id: int,
             tick: int) -> ActionRequest:
    if verb == "walk":
        return library.request("walk", agent=agent_index, request_id=request_id, tick=tick,
                               position=agent.walk_target, scalars={"radius": 1.0})
    return library.request("look", agent=agent_index, request_id=request_id, tick=tick, entity=agent.look_handle)


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
    library = _library()
    combiner = library.combiner("gen7")
    config = EncoderConfig()
    params = init_encoder(jax.random.PRNGKey(seed), config)
    batch = stack_batch([agent.packed for agent in agents])
    apply = encoder_apply_jit(config)
    contexts = [
        AgentContext(world_bounds=WORLD_BOUNDS, known_entities=KNOWN_ENTITIES, observation_tokens=agent.packed)
        for agent in agents
    ]
    current = [
        {"walk": _request(library, "walk", index, agent, 1, 0), "look": _request(library, "look", index, agent, 2, 0)}
        for index, agent in enumerate(agents)
    ]
    next_id = [3] * batch_size
    done: list[set[int]] = [set() for _ in agents]
    samples: dict[str, list[float]] = {stage: [] for stage in _STAGES}
    statuses: dict[str, int] = {}

    for tick in range(warmup + ticks):
        for index, agent in enumerate(agents):
            for verb in ("walk", "look"):
                if current[index][verb].request_id in done[index]:
                    current[index][verb] = _request(library, verb, index, agent, next_id[index], tick)
                    next_id[index] += 1

        started = time.perf_counter()
        for agent in agents:
            found, reason = references.resolve_entity(agent.look_handle, agent.entities)
            if found is None:
                raise RuntimeError(f"the look target is not among its entity tokens: {reason}")
            references.to_agent_frame(agent.walk_target, agent.position, agent.yaw)
        resolver_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        steps = [combiner.step(tick, index, [current[index]["walk"], current[index]["look"]], contexts[index])
                 for index in range(batch_size)]
        combiner_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        output = apply(params, batch)
        jax.block_until_ready(output.context)
        encoder_ms = (time.perf_counter() - started) * 1000.0

        for index, step in enumerate(steps):
            done[index] = {status.request_id for status in step.statuses if status.kind == "done"}
            if tick >= warmup:
                for status in step.statuses:
                    statuses[status.kind] = statuses.get(status.kind, 0) + 1
        if tick >= warmup:
            for stage, value in zip(_STAGES, (resolver_ms, combiner_ms, encoder_ms,
                                              resolver_ms + combiner_ms + encoder_ms), strict=True):
                samples[stage].append(value)

    return {
        "batch": batch_size,
        "ticks": ticks,
        "warmup": warmup,
        "stages": {stage: {**_stats(samples[stage]), "samples_ms": samples[stage]} for stage in _STAGES},
        "statuses": statuses,
    }


def _gate(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    measured = next((result for result in results if result["batch"] == GATE_BATCH), None)
    if measured is None:
        return {"batch": GATE_BATCH, "bar_ms": BAR_MS, "verdict": "not run"}
    p95 = measured["stages"]["total"]["p95_ms"]
    return {"batch": GATE_BATCH, "p95_total_ms": p95, "bar_ms": BAR_MS,
            "verdict": "PASS" if p95 <= BAR_MS else "FAIL"}


def _print_report(results: Sequence[dict[str, Any]], gate: dict[str, Any]) -> None:
    print(f"{'batch':>5}  {'stage':<9}  {'p50 ms':>9}  {'p95 ms':>9}  {'max ms':>9}")
    for result in results:
        for stage in _STAGES:
            stats = result["stages"][stage]
            print(f"{result['batch']:>5}  {stage:<9}  {stats['p50_ms']:>9.3f}  {stats['p95_ms']:>9.3f}  "
                  f"{stats['max_ms']:>9.3f}")
        counts = ", ".join(f"{kind} {count}" for kind, count in sorted(result["statuses"].items()))
        print(f"       statuses, all agents, {result['ticks']} measured ticks: {counts}")
    if gate["verdict"] == "not run":
        print(f"gate p95(total) at batch {GATE_BATCH} = not run (bar {BAR_MS:g} ms)")
    else:
        print(f"gate p95(total) at batch {GATE_BATCH} = {gate['p95_total_ms']:.3f} ms (bar {BAR_MS:g} ms): "
              f"{gate['verdict']}")


def _environment() -> dict[str, Any]:
    return {"python": sys.version.split()[0], "jax": jax.__version__, "backend": jax.default_backend(),
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS")}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-tick latency of the lower layer: resolver, combiner, encoder.")
    parser.add_argument("--batch", default="1,8,64", help="agents per tick, comma-separated (default 1,8,64)")
    parser.add_argument("--ticks", type=int, default=300, help="measured ticks per batch size (default 300)")
    parser.add_argument("--warmup", type=int, default=30, help="discarded warm-up ticks (default 30)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the agents and the encoder (default 0)")
    parser.add_argument("--json", metavar="PATH", help="also write every number and the raw per-tick samples")
    args = parser.parse_args(argv)
    try:
        batches = [int(item) for item in args.batch.split(",")]
    except ValueError:
        parser.error(f"--batch needs comma-separated integers, got {args.batch!r}")
    if not batches or min(batches) < 1 or args.ticks < 1 or args.warmup < 0:
        parser.error("batch sizes and --ticks must be at least 1, and --warmup at least 0")

    results = [_measure(batch, args.ticks, args.warmup, args.seed) for batch in batches]
    gate = _gate(results)
    _print_report(results, gate)
    if args.json:
        payload = {
            "environment": _environment(),
            "settings": {"batch": batches, "ticks": args.ticks, "warmup": args.warmup, "seed": args.seed,
                         "capacities": CAPACITIES, "world_bounds": WORLD_BOUNDS, "tick_ms": 1000.0 / 30.0},
            "results": results,
            "gate": gate,
        }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
