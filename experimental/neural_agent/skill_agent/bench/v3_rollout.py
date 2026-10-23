"""Rollout speed on v3 against v1, and the bytes of one agent's state (``docs/TRAINING_PLAN.md`` 8.3, stage V0).

One decision tick of a training rollout on the trainers' own flat world (``training/blocks.environment``; the reset and
the first neutral tick as ``training/skill_dagger.rollout`` does them, the spawn yaw 0). Two paths at each batch size:

- ``v1`` (today's trainers): ``blocks.token_kernel`` on the legal observation, then the trunk with no memory
  (``EncoderConfig()``), then the generalist head (``GeneralistConfig()``), built as ``bench/bench_generalist`` builds
  them.
- ``v3``: ``runtime/state_tick.tick`` (the state layer, then the trunk) with ``bench/agent_models.CONFIG``; the state
  starts from ``agent_models.start_state``, is carried tick to tick, and tick 0 is a reset; then the same head on the
  trunk's output. The state layer is also timed alone (``state/layer.step`` on the same inputs), so the trunk's share of
  ``tick`` is ``tick`` minus that, derived and marked as such.

Every call is timed with ``time.perf_counter`` and ``jax.block_until_ready``. Actions are neutral each tick: the
student's choice costs the head, which is timed. Each run discards ``WARMUP`` ticks, then times ``WINDOWS`` windows of
``--ticks`` ticks; p50 and p95 are over the timed ticks. A program is built once per process, so a later round reuses
it: ``first_call_ms`` (tick 0 of a run) includes the compile only in the first run that uses that program.

Each round runs v1 then v3 at every batch size, so the two paths are interleaved on a shared machine. ``uptime`` is
printed at the start and the end; each run records the 1, 5 and 15 minute load averages at its start and end. Once per
invocation: the bytes of one lane's v3 ``AgentState`` by top-level field and by part, and the parameter counts and bytes
of the v1 and v3 trunks and heads.

The flat world has no perception channel, so the body yaw is passed to the state layer. Under the gen-7 render with
``--construct perception_v2`` the channel is used when the observation carries it; each record says which. Output: one
JSON object per line.

From the worktree root (``PYTHONPATH="$PWD:$PWD/HytaleRL/hytalegym" JAX_PLATFORMS=cpu``)::

    .venv/bin/python -m experimental.neural_agent.skill_agent.bench.v3_rollout --batches 64 128 256 --rounds 2 --ticks 30
    .venv/bin/python -m experimental.neural_agent.skill_agent.runtime.gen7 --engine DIR --construct perception_v2 -- \\
        experimental/neural_agent/skill_agent/bench/v3_rollout.py --batches 64 128 256 --rounds 2 --ticks 30
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.bench import agent_models as am
from experimental.neural_agent.skill_agent.bench import bench_generalist as bg
from experimental.neural_agent.skill_agent.bench import encoding_audit as audit
from experimental.neural_agent.skill_agent.model.encoder import EncoderConfig, encoder_apply_jit, init_encoder
from experimental.neural_agent.skill_agent.model.generalist import (
    GeneralistConfig,
    generalist_apply_jit,
    init_generalist,
)
from experimental.neural_agent.skill_agent.state import layer
from experimental.neural_agent.skill_agent.training import blocks
from hytalegym.jax.combat.types import AGENT_ENTITY

BATCHES = (64, 128, 256)
TICKS = 30
"""Timed ticks per window (``--ticks``)."""
WINDOWS = 3
WARMUP = 10
"""Discarded ticks at the start of each run."""
MAXIMUM_TURN_DEGREES = 45.0
"""``skill_dagger.SkillTrainConfig.maximum_turn_degrees``, the trainers' default."""
PATHS = ("v1", "v3")
STAGES = ("env", "tokens", "trunk", "head", "path", "total")


def timed(fn: Callable[..., Any], *args: Any) -> tuple[Any, float]:
    """``fn(*args)`` and its wall time in ms, with the result ready."""

    started = time.perf_counter()
    value = fn(*args)
    jax.block_until_ready(value)
    return value, 1000.0 * (time.perf_counter() - started)


def neutral_factors(batch: int) -> jax.Array:
    return jnp.broadcast_to(jnp.asarray(blocks.NEUTRAL, dtype=jnp.int32), (batch, blocks.HEAD_COUNT))


def env_keys(seed: int, index: int, batch: int) -> jax.Array:
    """Per-lane keys for the environment's ``index``-th call: 0 is the reset, 1 the neutral tick, then one per decision."""

    return jax.random.split(jax.random.fold_in(jax.random.key(seed), index), batch)


def start_env(batch: int, seed: int) -> tuple[Any, Any, Any]:
    """The flat world's environment, reset and given its first neutral tick, as ``skill_dagger.rollout`` does it."""

    env = blocks.environment(batch, MAXIMUM_TURN_DEGREES)
    spawn = jnp.broadcast_to(jnp.asarray(blocks.SPAWN_XZ, dtype=jnp.float32), (batch, 2))
    other = jnp.broadcast_to(jnp.asarray(blocks.OTHER_XYZ, dtype=jnp.float32), (batch, 3))
    carry = env.reset(env_keys(seed, 0, batch), spawn, jnp.zeros((batch,), dtype=jnp.float32), other)
    carry, observation, _ = env.step(carry, neutral_factors(batch), env_keys(seed, 1, batch))
    return env, carry, observation


class V1Path:
    """Today's trainers: the legal tokens, the trunk with no memory, the generalist head."""

    name = "v1"
    tokens_is = "token_kernel"
    trunk_is = "measured"

    def __init__(self, batch: int, seed: int) -> None:
        self.config = EncoderConfig()
        self.trunk_params = init_encoder(jax.random.key(seed), self.config)
        self.head_config = GeneralistConfig()
        self.head_params = init_generalist(jax.random.key(seed + 1), self.head_config, self.config)
        self.tokens = blocks.token_kernel()
        self.trunk = encoder_apply_jit(self.config)
        self.head = generalist_apply_jit(self.head_config)
        self.requests = bg._requests(seed, batch, self.head_config)

    def start(self) -> None:
        """Nothing carried between ticks on this path."""

    def step(self, observation: Any, yaw: jax.Array, first: bool) -> tuple[Any, dict[str, float]]:
        tokens, tokens_ms = timed(self.tokens, observation, yaw)
        encoded, trunk_ms = timed(self.trunk, self.trunk_params, tokens)
        masks = {name: mask for name, (_, mask) in tokens.items()}
        out, head_ms = timed(self.head, self.head_params, encoded, masks, self.requests)
        return out, {"tokens": tokens_ms, "trunk": trunk_ms, "path": tokens_ms + trunk_ms, "head": head_ms}


class V3Path:
    """The v3 path: ``state_tick.tick`` with one trunk for the batch (``agent_models.shared_run``), the state carried,
    then the generalist head on the trunk's output."""

    name = "v3"
    tokens_is = "state_layer"
    trunk_is = "derived: tick minus its state layer"

    def __init__(self, batch: int, seed: int) -> None:
        self.batch = batch
        self.params = init_encoder(jax.random.key(seed), am.CONFIG)
        self.head_config = GeneralistConfig()
        self.head_params = init_generalist(jax.random.key(seed + 1), self.head_config, am.CONFIG)
        self.head = generalist_apply_jit(self.head_config)
        self.requests = bg._requests(seed, batch, self.head_config)
        self.tick = am.shared_run(self.params)
        self.layer = jax.jit(lambda state, observation, action, kw: layer.step(
            observation, state, last_action=action, **kw))
        self.action = audit.gen7_action(neutral_factors(batch))
        self.start()

    def start(self) -> None:
        self.state = am.start_state([self.params] * self.batch, self.batch)

    def step(self, observation: Any, yaw: jax.Array, first: bool) -> tuple[Any, dict[str, float]]:
        kw = {"reset": jnp.full((self.batch,), first, dtype=jnp.bool_)}
        if getattr(observation, "perception_v2", None) is None:  # no channel: the body yaw comes from the world
            kw["body_yaw_degrees"] = yaw
        _, tokens_ms = timed(self.layer, self.state, observation, self.action, kw)
        (view, self.state), path_ms = timed(self.tick, self.state, (observation, self.action, kw))
        out, head_ms = timed(self.head, self.head_params, view.encoded, view.masks, self.requests)
        return out, {"tokens": tokens_ms, "trunk": path_ms - tokens_ms, "path": path_ms, "head": head_ms}


@functools.lru_cache(maxsize=None)
def make_path(name: str, batch: int, seed: int = 0) -> V1Path | V3Path:
    """One path at one batch size, built once per process so its programs stay compiled across rounds."""

    if name == "v1":
        return V1Path(batch, seed)
    if name == "v3":
        return V3Path(batch, seed)
    raise ValueError(f"unknown path {name!r}; expected one of {PATHS}")


def stats(values: Sequence[float]) -> dict[str, float]:
    p50, p95 = (float(x) for x in np.percentile(np.asarray(values, dtype=np.float64), [50, 95]))
    return {"p50_ms": round(p50, 3), "p95_ms": round(p95, 3)}


def run(path: V1Path | V3Path, batch: int, seed: int, ticks: int) -> dict[str, Any]:
    """One run: a fresh episode, ``WARMUP`` discarded ticks, then ``WINDOWS`` windows of ``ticks`` timed ticks."""

    path.start()
    env, carry, observation = start_env(batch, seed)
    neutral = neutral_factors(batch)
    channel = getattr(observation, "perception_v2", None) is not None
    load_start = [round(float(x), 2) for x in os.getloadavg()]
    samples: dict[str, list[float]] = {stage: [] for stage in STAGES}
    first: dict[str, float] = {}
    for index in range(WARMUP + WINDOWS * ticks):
        (carry, observation, _), env_ms = timed(env.step, carry, neutral, env_keys(seed, index + 2, batch))
        yaw = carry.runtime.combat.yaw[:, AGENT_ENTITY]
        _, parts = path.step(observation, yaw, index == 0)
        row = {"env": env_ms, **parts, "total": env_ms + parts["path"] + parts["head"]}
        if index == 0:
            first = row
        if index >= WARMUP:
            for stage in STAGES:
                samples[stage].append(row[stage])
    totals = samples["total"]
    window_p50 = [round(float(np.percentile(totals[w * ticks:(w + 1) * ticks], 50)), 3) for w in range(WINDOWS)]
    total_p50 = stats(totals)["p50_ms"]
    return {
        "path": path.name, "batch": batch, "channel": channel, "tokens_is": path.tokens_is,
        "trunk_is": path.trunk_is, "samples": len(totals), "stages_ms": {s: stats(samples[s]) for s in STAGES},
        "decisions_per_s": round(batch / (total_p50 / 1000.0), 1), "window_total_p50_ms": window_p50,
        "first_call_ms": {s: round(first[s], 1) for s in STAGES},
        "load": {"start": load_start, "end": [round(float(x), 2) for x in os.getloadavg()]},
    }


def state_bytes(state: layer.AgentState) -> tuple[int, dict[str, int], dict[str, int]]:
    """The bytes of one state: the total, by top-level field, and by part of each field that is a named tuple (the
    spatial stores, the written tables, the event log, the places, the social records). Each is the sum of its leaves'
    ``nbytes``."""

    def size(tree: Any) -> int:
        return sum(int(leaf.nbytes) for leaf in jax.tree_util.tree_leaves(tree))

    by_field = {name: size(value) for name, value in zip(state._fields, state)}
    by_part: dict[str, int] = {}
    for name, value in zip(state._fields, state):
        if hasattr(value, "_fields"):
            by_part.update({f"{name}.{part}": size(piece) for part, piece in zip(value._fields, value)})
    return sum(by_field.values()), by_field, by_part


def param_size(params: Any) -> dict[str, int]:
    count, nbytes = am.trunk_size(params)
    return {"count": count, "bytes": nbytes}


def sizes(seed: int = 0) -> dict[str, Any]:
    """Once per invocation: one lane's v3 state, and the parameters of each trunk and head."""

    key = jax.random.key(seed)
    total, by_field, by_part = state_bytes(am.start_state([init_encoder(key, am.CONFIG)], 1))
    return {
        "state_bytes_v3_one_lane": total, "state_by_field": by_field, "state_by_part": by_part,
        "parameters": {
            "v1_trunk": param_size(init_encoder(key, EncoderConfig())),
            "v1_head": param_size(init_generalist(key, GeneralistConfig(), EncoderConfig())),
            "v3_trunk": param_size(init_encoder(key, am.CONFIG)),
            "v3_head": param_size(init_generalist(key, GeneralistConfig(), am.CONFIG)),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rollout tick cost on v3 against v1, and one lane's state bytes.")
    parser.add_argument("--batches", type=int, nargs="+", default=list(BATCHES))
    parser.add_argument("--rounds", type=int, default=2, help="the two paths are interleaved round by round")
    parser.add_argument("--ticks", type=int, default=TICKS, help="timed ticks per window (3 windows per run)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if min(args.batches) < 1 or args.ticks < 1 or args.rounds < 1:
        parser.error("--batches, --ticks and --rounds must be at least 1")
    print(json.dumps({"uptime_start": am.uptime(), "argv": sys.argv, "backend": jax.default_backend(),
                      "jax": jax.__version__, "batches": args.batches, "rounds": args.rounds, "ticks": args.ticks,
                      "windows": WINDOWS, "warmup": WARMUP, "seed": args.seed}), flush=True)
    print(json.dumps({"sizes": sizes(args.seed)}), flush=True)
    for round_ in range(1, args.rounds + 1):
        for batch in args.batches:
            for name in PATHS:
                record = run(make_path(name, batch, args.seed), batch, args.seed, args.ticks)
                print(json.dumps({"round": round_, **record}), flush=True)
    print(json.dumps({"uptime_end": am.uptime()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
