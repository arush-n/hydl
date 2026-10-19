"""The per-tick cost of the v2 tokens (``docs/PERCEPTION.md`` section 9, build step 5): the trunk at 16, 32 and 64
entities and at several description widths, beside v1, against the 30 Hz tick; and the registry gathers of a tick.

The trunk is the part of the per-tick path whose cost follows the token schema; the resolver and the combiner
(``bench_tick.py``) do not read token features. Per batch size B, entity capacity N and description width D, B agents
hold full token sets (``bench_tick.CAPACITIES`` with N entities, every slot valid: the worst case) of random features
at the v2 widths, where D numbers replace the contract's ``DESCRIPTION_WIDTH`` in the entity, block-candidate and
inventory-slot widths. They are stacked on the device once; each tick times
``encoder_apply_jit(config)(params, batch)`` with ``jax.block_until_ready``. v1 at the same capacities is the
baseline, timed in the same run, so a busy machine moves both.

The gathers: one jitted call per tick that looks up every entity, slot and block-candidate id of B agents in tables of
the 0.7.2 registry's sizes, gathers their property rows and descriptions, and runs ``tool_relations`` of the 36 slots
against the 16 block candidates.

Run from the repository root, with JAX_PLATFORMS=cpu, and record the load average with the numbers::

    .venv/bin/python -m experimental.neural_agent.skill_agent.bench.bench_perception --batch 1,8,64 \
        --entities 16,32,64 --description-widths 16,32,64 --ticks 200 --warmup 20 --seed 0 [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.bench.bench_tick import CAPACITIES
from experimental.neural_agent.skill_agent.contract.surface import (
    BLOCK_PROPERTIES,
    CREATURE_PROPERTIES,
    DESCRIPTION_WIDTH,
    ITEM_PROPERTIES,
    TOKEN_SCHEMA_V1,
    TOKEN_SCHEMA_V2,
)
from experimental.neural_agent.skill_agent.model.encoder import (
    EncoderConfig,
    encoder_apply_jit,
    init_encoder,
    parameter_count,
    token_widths,
)
from experimental.neural_agent.skill_agent.perception.adapter import RegistryTables, lookup, tool_relations

TICK_MS = 1000.0 / 30.0
GATE_MS = 15.0
"""The lower layer's share of a tick: p95 at most 15 ms at batch 8 (``README.md`` of this folder)."""
DESCRIBED = ("entity", "block_candidate", "inventory_slot")
REGISTRY_072 = {"items": 3960, "blocks": 3182, "creatures": 894, "gather_types": 19}


def widths(schema: str, description_width: int) -> tuple[tuple[str, int], ...]:
    """The trunk's token widths under ``schema``, with v2's descriptions ``description_width`` numbers wide."""

    change = description_width - DESCRIPTION_WIDTH if schema == TOKEN_SCHEMA_V2 else 0
    return tuple((name, width + (change if name in DESCRIBED else 0)) for name, width in token_widths(schema))


def _timed(function: Callable[[], Any], ticks: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        jax.block_until_ready(function())
    samples = []
    for _ in range(ticks):
        start = time.perf_counter()
        jax.block_until_ready(function())
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def _stats(samples: Sequence[float]) -> dict[str, float]:
    values = np.asarray(samples)
    return {"p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
            "mean_ms": float(values.mean())}


def trunk_row(schema: str, batch: int, entities: int, description_width: int, ticks: int, warmup: int,
              seed: int) -> dict[str, Any]:
    capacities = {**CAPACITIES, "entity": entities}
    config = EncoderConfig(token_types=widths(schema, description_width))
    params = init_encoder(jax.random.PRNGKey(seed), config)
    rng = np.random.default_rng(seed)
    tokens = {name: (jnp.asarray(rng.standard_normal((batch, capacities[name], width), dtype=np.float32)),
                     jnp.ones((batch, capacities[name]), dtype=jnp.bool_))
              for name, width in config.token_types}
    apply = encoder_apply_jit(config)
    samples = _timed(lambda: apply(params, tokens).context, ticks, warmup)
    return {"stage": "trunk", "schema": schema, "batch": batch, "entities": entities,
            "description_width": description_width if schema == TOKEN_SCHEMA_V2 else None,
            "tokens_per_agent": sum(capacities[name] for name, _ in config.token_types),
            "parameters": parameter_count(params), **_stats(samples)}


def _keys(rng: np.random.Generator, count: int) -> np.ndarray:
    keys = np.unique(rng.integers(1, 2**31 - 1, size=2 * count))[:count]
    return np.sort(keys).astype(np.int32)


def synthetic_tables(rng: np.random.Generator) -> RegistryTables:
    """Random tables of the 0.7.2 registry's sizes (``REGISTRY_072``)."""

    items, blocks, creatures, gathers = (REGISTRY_072[key] for key in ("items", "blocks", "creatures", "gather_types"))

    def rows(count: int, width: int) -> jax.Array:
        return jnp.asarray(rng.random((1 + count, width), dtype=np.float32))

    return RegistryTables(
        item_keys=jnp.asarray(_keys(rng, items)), item_rows=jnp.arange(1, items + 1, dtype=jnp.int32),
        item_features=rows(items, len(ITEM_PROPERTIES)), item_descriptions=rows(items, DESCRIPTION_WIDTH),
        tool_power=rows(items, gathers), tool_quality=rows(items, gathers),
        block_keys=jnp.asarray(_keys(rng, blocks)), block_rows=jnp.arange(1, blocks + 1, dtype=jnp.int32),
        block_features=rows(blocks, len(BLOCK_PROPERTIES)), block_descriptions=rows(blocks, DESCRIPTION_WIDTH),
        block_gather=jnp.asarray(rng.integers(-1, gathers, size=1 + blocks), dtype=jnp.int32),
        block_quality=jnp.asarray(rng.integers(0, 5, size=1 + blocks), dtype=jnp.float32),
        creature_keys=jnp.asarray(_keys(rng, creatures)), creature_rows=jnp.arange(1, creatures + 1, dtype=jnp.int32),
        creature_features=rows(creatures, len(CREATURE_PROPERTIES)),
        creature_descriptions=rows(creatures, DESCRIPTION_WIDTH),
        attitude_to_player=jnp.asarray(rng.integers(-1, 5, size=1 + creatures), dtype=jnp.int32),
    )


def gather_row(batch: int, entities: int, ticks: int, warmup: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    tables = synthetic_tables(rng)

    def ids(keys: jax.Array, count: int) -> jax.Array:
        return jnp.asarray(rng.choice(np.asarray(keys), size=(batch, count)), dtype=jnp.int32)

    entity_ids = ids(tables.creature_keys, entities)
    slot_ids = ids(tables.item_keys, CAPACITIES["inventory_slot"])
    block_ids = ids(tables.block_keys, CAPACITIES["block_candidate"])

    @jax.jit
    def gathers(entity_ids: jax.Array, slot_ids: jax.Array, block_ids: jax.Array) -> tuple[jax.Array, ...]:
        creature, _ = lookup(tables.creature_keys, tables.creature_rows, entity_ids)
        item, present = lookup(tables.item_keys, tables.item_rows, slot_ids)
        block, _ = lookup(tables.block_keys, tables.block_rows, block_ids)
        relations = tool_relations(tables, item, present, item[:, 0], block)
        return (tables.creature_features[creature], tables.creature_descriptions[creature],
                tables.attitude_to_player[creature], tables.item_features[item], tables.item_descriptions[item],
                tables.block_features[block], tables.block_descriptions[block], *relations)

    samples = _timed(lambda: gathers(entity_ids, slot_ids, block_ids), ticks, warmup)
    return {"stage": "gathers", "batch": batch, "entities": entities, **_stats(samples)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-tick cost of the v2 tokens: the trunk and the registry gathers.")
    parser.add_argument("--batch", default="1,8,64", help="agents per tick, comma-separated (default 1,8,64)")
    parser.add_argument("--entities", default="16,32,64", help="entity capacities (default 16,32,64)")
    parser.add_argument("--description-widths", default="16,32,64", help="v2 description widths (default 16,32,64)")
    parser.add_argument("--ticks", type=int, default=200, help="measured ticks per row (default 200)")
    parser.add_argument("--warmup", type=int, default=20, help="discarded warm-up ticks (default 20)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", metavar="PATH", help="also write every row")
    args = parser.parse_args(argv)
    batches = [int(value) for value in args.batch.split(",")]
    entity_counts = [int(value) for value in args.entities.split(",")]
    description_widths = [int(value) for value in args.description_widths.split(",")]

    rows = []
    for batch in batches:
        for entities in entity_counts:
            rows.append(trunk_row(TOKEN_SCHEMA_V1, batch, entities, DESCRIPTION_WIDTH, args.ticks, args.warmup,
                                  args.seed))
            for width in description_widths:
                rows.append(trunk_row(TOKEN_SCHEMA_V2, batch, entities, width, args.ticks, args.warmup, args.seed))
            rows.append(gather_row(batch, entities, args.ticks, args.warmup, args.seed))
    print(f"{'stage':8} {'schema':16} {'B':>3} {'N':>3} {'D':>3} {'tokens':>6} {'params':>8} {'p50 ms':>8} "
          f"{'p95 ms':>8} {'p95/tick':>8}")
    for row in rows:
        print(f"{row['stage']:8} {row.get('schema', '-'):16} {row['batch']:>3} {row['entities']:>3} "
              f"{row.get('description_width') or '-':>3} {row.get('tokens_per_agent', '-'):>6} "
              f"{row.get('parameters', '-'):>8} {row['p50_ms']:>8.3f} {row['p95_ms']:>8.3f} "
              f"{row['p95_ms'] / TICK_MS:>8.1%}")
    environment = {"load_average": os.getloadavg(), "jax": jax.__version__, "devices": [str(d) for d in jax.devices()]}
    print(json.dumps(environment))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"rows": rows, "environment": environment, "tick_ms": TICK_MS, "gate_ms": GATE_MS}, handle,
                      indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["GATE_MS", "TICK_MS", "gather_row", "main", "synthetic_tables", "trunk_row", "widths"]
