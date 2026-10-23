"""What one v3 decision costs to store, and the skills' episode horizons (TRAINING_PLAN.md section 8.3, stage V0:
measurements only, no training).

A v3 decision's tokens are 42,171 floats at the default capacities (168,684 bytes as float32), and the trainers store
every decision. This runs the state layer's real tick (``state/layer.step``, as ``encoding_audit.py`` does) on three
worlds and reports, per decision (one agent, one tick): the bytes if each token type keeps only its valid rows in
float16 plus one mask byte per slot; the share by type; the ratio to the dense float32 size; and how many stored
float16 values move by more than 1e-3 of their magnitude. It also summarises the horizons of the two skills with
trainers: walk_to's sampled ones and look_at's constant.

Worlds: ``flat`` (the trainers' flat world, ``training/blocks.environment``), ``region`` (the Region scenario on
``region-library-pilot-v2-072``) and ``duel``. Plain, and under the gen-7 render with the perception channel::

    PYTHONPATH=$W:$W/HytaleRL/hytalegym JAX_PLATFORMS=cpu $PY -m experimental.neural_agent.skill_agent.bench.v3_data_size \\
        --world flat region duel --ticks 300 --batch 4
    PYTHONPATH=$W:$W/HytaleRL/hytalegym JAX_PLATFORMS=cpu $PY -m experimental.neural_agent.skill_agent.runtime.gen7 \\
        --engine DIR --construct perception_v2 -- $W/experimental/neural_agent/skill_agent/bench/v3_data_size.py \\
        --world flat region duel --ticks 300 --batch 4

Output: one JSON line per world, then one line of horizons. The human-facing prints of the scenarios go to stderr.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import jax
import jax.numpy as jnp
import numpy as np

import hytalegym.jax.combat.types as combat_types
from experimental.neural_agent.skill_agent.bench import encoding_audit as audit
from experimental.neural_agent.skill_agent.contract import tokens_v3 as v3
from experimental.neural_agent.skill_agent.neural_skills.look_at.task import LOOK_HORIZON_TICKS
from experimental.neural_agent.skill_agent.neural_skills.walk_to.task import WalkToTasks
from experimental.neural_agent.skill_agent.state import layer
from experimental.neural_agent.skill_agent.state import persona as pe
from experimental.neural_agent.skill_agent.state.senses import Light
from experimental.neural_agent.skill_agent.tasks.simple_tasks import TICKS_PER_SECOND
from experimental.neural_agent.skill_agent.training import blocks
from experimental.neural_agent.skill_agent.training.skill_dagger import SkillTrainConfig

REPO = Path(__file__).resolve().parents[4]
REGION_LIBRARY = "region-library-pilot-v2-072"
FLOAT16_BYTES = 2
FLOAT32_BYTES = 4
MASK_BYTES = 1
FLOAT16_TOLERANCE = 1e-3
FLOAT16_SMALLEST_NORMAL = 2.0 ** -14
HORIZON_SAMPLES = 4096
HORIZON_SEED = 0
DENSE_FLOATS = sum(v3.CAPACITIES_V3[name] * len(cls.FEATURES) for name, cls in v3.TOKEN_TYPES_V3.items())
DENSE_FLOAT32_BYTES = FLOAT32_BYTES * DENSE_FLOATS

Step = audit.Step


def stored_bytes(tokens: Mapping[str, tuple[Any, Any]]) -> dict[str, np.ndarray]:
    """Per token type, the bytes of each decision (each row of the batch) if only its valid rows are kept: valid rows x
    width x 2 (float16), plus one mask byte per slot. ``tokens``: type -> (features [B, slots, width], valid [B, slots])."""

    out = {}
    for name, (features, valid) in tokens.items():
        valid = np.asarray(valid, bool)
        width, slots = np.shape(features)[-1], valid.shape[-1]
        out[name] = valid.sum(axis=-1) * width * FLOAT16_BYTES + slots * MASK_BYTES
    return out


def float16_differences(values: Any) -> tuple[int, int, int]:
    """Of float32 ``values``: how many their float16 round trip moves by more than 1e-3 of their magnitude, how many of
    those are below float16's smallest normal (2**-14), and how many overflow to infinity."""

    values = np.asarray(values, np.float32).ravel()
    half = values.astype(np.float16).astype(np.float32)
    moved = np.abs(half - values) > FLOAT16_TOLERANCE * np.abs(values)
    tiny = moved & (np.abs(values) < FLOAT16_SMALLEST_NORMAL)
    overflow = np.isinf(half) & np.isfinite(values)
    return int(moved.sum()), int(tiny.sum()), int(overflow.sum())


def horizon_summary(ticks: Iterable[float]) -> dict[str, float]:
    """p50, p95 and max of decision horizons, in ticks and in seconds (``TICKS_PER_SECOND``)."""

    ticks = np.asarray(list(ticks), np.float64)
    summary = {}
    for label, value in (("p50", np.percentile(ticks, 50)), ("p95", np.percentile(ticks, 95)), ("max", ticks.max())):
        summary[f"{label}_ticks"] = float(value)
        summary[f"{label}_seconds"] = float(value) / TICKS_PER_SECOND
    return summary


def horizons() -> dict[str, dict]:
    """walk_to's horizons from its sampler, and look_at's from its constant."""

    walk = np.asarray(WalkToTasks.sample(jax.random.key(HORIZON_SEED), HORIZON_SAMPLES).horizon_ticks)
    return {
        "walk_to": {"source": f"WalkToTasks.sample(key {HORIZON_SEED}, {HORIZON_SAMPLES}).horizon_ticks",
                    **horizon_summary(walk)},
        "look_at": {"source": "LOOK_HORIZON_TICKS", **horizon_summary([LOOK_HORIZON_TICKS])},
    }


def flat_steps(ticks: int, batch: int, seed: int = 0) -> Iterator[Step]:
    """The trainers' flat world, reset as ``skill_dagger.rollout`` resets it: each lane spawns at the default spawn with a
    yaw drawn uniformly in [-180, 180] (as walk_to's ``train.py`` draws it), the other entity at its default place, one
    neutral tick, then random legal actions held as ``encoding_audit.random_legal`` holds them."""

    env = blocks.environment(batch, SkillTrainConfig().maximum_turn_degrees)
    tables, effects = audit.registry_tables(), audit.effect_tables()
    keys = jax.random.split(jax.random.key(seed), ticks + 2)
    pick_keys = jax.random.split(jax.random.key(seed + 1), ticks)
    spawn_yaw = jax.random.uniform(jax.random.key(seed + 2), (batch,), dtype=jnp.float32, minval=-180.0, maxval=180.0)
    spawn = jnp.broadcast_to(jnp.asarray(blocks.SPAWN_XZ, dtype=jnp.float32), (batch, 2))
    other = jnp.broadcast_to(jnp.asarray(blocks.OTHER_XYZ, dtype=jnp.float32), (batch, 3))
    carry = env.reset(jax.random.split(keys[0], batch), spawn, spawn_yaw, other)
    neutral = jnp.broadcast_to(jnp.asarray(blocks.NEUTRAL, dtype=jnp.int32), (batch, blocks.HEAD_COUNT))
    carry, observation, _ = env.step(carry, neutral, jax.random.split(keys[1], batch))
    factors = neutral
    for tick in range(ticks):
        light = carry.light_tokens
        adapter = {"tables": tables, "effects": effects, "reset": jnp.full((batch,), tick == 0),
                   "light": Light(light.token_f32, light.token_mask & light.available[:, None])}
        if getattr(observation, "perception_v2", None) is None:  # no channel: the body yaw comes from the world
            adapter["body_yaw_degrees"] = carry.runtime.combat.yaw[:, 0]
        yield observation, audit.gen7_action(factors), adapter
        factors = audit.random_legal(pick_keys[tick], carry.action_mask, factors)
        carry, observation, _ = env.step(carry, factors, jax.random.split(keys[tick + 2], batch))


def duel_or_region_steps(scenario: str, ticks: int, batch: int) -> Iterator[Step]:
    if scenario == "duel":
        return audit.duel_steps(ticks, batch)
    return audit.region_steps(ticks, batch, library=REGION_LIBRARY)


WORLDS = {
    "flat": flat_steps,
    "region": lambda ticks, batch: duel_or_region_steps("region", ticks, batch),
    "duel": lambda ticks, batch: duel_or_region_steps("duel", ticks, batch),
}


class Recorder:
    """Every decision's stored bytes per token type, and the float16 check over every stored value."""

    def __init__(self) -> None:
        self.per_type: dict[str, list[np.ndarray]] = {name: [] for name in v3.TOKEN_TYPES_V3}
        self.values = self.moved = self.tiny = self.overflow = 0

    def add(self, tokens: Mapping[str, tuple[Any, Any]]) -> None:
        for name, sizes in stored_bytes(tokens).items():
            self.per_type[name].append(sizes)
        for features, valid in tokens.values():
            rows = np.asarray(features, np.float32)[np.asarray(valid, bool)]
            moved, tiny, overflow = float16_differences(rows)
            self.values += rows.size
            self.moved += moved
            self.tiny += tiny
            self.overflow += overflow


def measure(steps: Iterable[Step], batch: int) -> tuple[Recorder, bool]:
    """The state layer's tick over the steps, as ``encoding_audit.run`` runs it (seed 0), keeping every decision. Also
    whether the perception channel was on the observation."""

    state = layer.initial_state(batch, persona=pe.sample_personas(jax.random.key(0), batch))
    tick = jax.jit(lambda observation, carry, action, adapter: layer.step(observation, carry, last_action=action,
                                                                         **adapter))
    recorder, channel = Recorder(), None
    for observation, action, adapter in steps:
        if channel is None:
            channel = getattr(observation, "perception_v2", None) is not None
        tokens, state = tick(observation, state, action, adapter)
        recorder.add(tokens)
    return recorder, bool(channel)


def summarise(recorder: Recorder, world: str, channel: bool, batch: int, ticks: int) -> dict:
    per_type = {name: np.concatenate(parts) for name, parts in recorder.per_type.items()}
    total = np.sum(np.stack(list(per_type.values())), axis=0).astype(np.float64)
    mean = float(total.mean())
    combat_file = Path(combat_types.__file__).resolve()
    return {
        "world": world,
        "run": "plain" if combat_file.is_relative_to(REPO) else "render",
        "combat_types_file": str(combat_file),
        "channel": channel,
        "batch": batch,
        "ticks": ticks,
        "decisions": int(total.size),
        "bytes": {"mean": round(mean, 4), "p50": float(np.percentile(total, 50)),
                  "p95": float(np.percentile(total, 95)), "max": float(total.max())},
        "mean_bytes_by_type": {name: round(float(v.mean()), 4) for name, v in per_type.items()},
        "share_by_type": {name: round(float(v.mean()) / mean, 6) for name, v in per_type.items()},
        "dense_floats": DENSE_FLOATS,
        "dense_float32_bytes": DENSE_FLOAT32_BYTES,
        "ratio_to_dense_float32": round(mean / DENSE_FLOAT32_BYTES, 6),
        "float16": {"values": recorder.values, "moved_over_1e-3": recorder.moved,
                    "share_moved": round(recorder.moved / recorder.values, 6),
                    "moved_below_2^-14": recorder.tiny, "overflow_to_inf": recorder.overflow},
    }


WORLD_NAMES = tuple(WORLDS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", nargs="+", required=True, choices=WORLD_NAMES)
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4)
    options = parser.parse_args(argv)
    for world in options.world:
        with contextlib.redirect_stdout(sys.stderr):
            recorder, channel = measure(WORLDS[world](options.ticks, options.batch), options.batch)
        print(json.dumps(summarise(recorder, world, channel, options.batch, options.ticks)), flush=True)
    print(json.dumps({"horizons": horizons()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
