"""Does the v3 encoding carry what the agent needs? (``../docs/PERCEPTION.md`` section 10.5.) The state layer's real
tick (``state/layer.step``, with the content registry's tables) runs on rollouts in which the agent takes random legal
actions, each held for several ticks, against an opponent that attacks; for every column of every token type it counts
the valid rows, how often the column is non-zero, its range and its distinct values.

Scenarios:
- ``duel``: the open-flat duel (``tasks/duel_arena.py``), sword, shortbow and battleaxe; a chaser plays the opponent.
- ``region``: a Region world (``worlds/region.py``): real terrain, its geometry and light tokens, the built-in target.
  Run it through the gen-7 launcher with ``--construct perception_v2`` for the channel (position, inventory, entity
  kind): the stock engine is gen 5, and the channel reaches only this single-actor path.

CLI (from the repo root, ``PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym JAX_PLATFORMS=cpu``)::

    .venv/bin/python -m experimental.neural_agent.skill_agent.bench.encoding_audit run duel --out duel.json
    .venv/bin/python -m experimental.neural_agent.skill_agent.runtime.gen7 --engine DIR --render \\
        --construct perception_v2 -- experimental.neural_agent.skill_agent.bench.encoding_audit run region --out r.json
    .venv/bin/python -m experimental.neural_agent.skill_agent.bench.encoding_audit compare duel=duel.json region=r.json

The live server's run is ``live_encoding.py``; its report has the same shape.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.contract import tokens_v3 as v3
from experimental.neural_agent.skill_agent.contract.surface import head_layout
from experimental.neural_agent.skill_agent.state import layer
from experimental.neural_agent.skill_agent.state import persona as pe
from experimental.neural_agent.skill_agent.state.action_history import HEAD_NEUTRAL
from experimental.neural_agent.skill_agent.training.blocks import HEAD_NAMES

REPO = Path(__file__).resolve().parents[4]
REGISTRY_072 = REPO / "agent_skills/training_runs/perception/registry_072"
"""The saved 0.7.2 content registry (the private folder: read in place, never copied)."""
EFFECTS_072 = REPO / "agent_skills/training_runs/perception/effects_072"
"""The saved 0.7.2 status-effect registry, built from the game's archive (the private folder: read in place, never
copied; ``perception/effects.py`` builds it)."""
GYM_HEADS = len(HEAD_NAMES)
assert tuple(head.name for head in head_layout("gen7"))[:GYM_HEADS] == tuple(HEAD_NAMES)

Step = tuple[Any, jax.Array, dict]
"""One tick: the observation, the gen-7 action sent before it, and ``layer.step``'s keyword inputs."""


def registry_tables(prefix: Path = REGISTRY_072):
    from experimental.neural_agent.skill_agent.perception import adapter as v2
    from experimental.neural_agent.skill_agent.perception.registry import load_registry

    return v2.registry_tables(load_registry(prefix))


def effect_tables(prefix: Path = EFFECTS_072):
    """The status-effect registry's tables for the status token, or None when the saved folder is absent."""

    from experimental.neural_agent.skill_agent.perception import effects

    if not (prefix.with_suffix(".npz").exists() and prefix.with_suffix(".json").exists()):
        return None
    return effects.effect_tables(effects.load_effects(prefix))


def gen7_action(factors: jax.Array) -> jax.Array:
    """int32[B, 21]: the gym's 12 heads, then the planned heads at neutral."""

    rest = jnp.broadcast_to(jnp.asarray(HEAD_NEUTRAL[GYM_HEADS:]), (factors.shape[0], len(HEAD_NEUTRAL) - GYM_HEADS))
    return jnp.concatenate([jnp.asarray(factors, jnp.int32), rest], axis=1)


def random_legal(key: jax.Array, mask: jax.Array, previous: jax.Array, hold: float = 0.875) -> jax.Array:
    """int32[B, 12]: each head keeps its last choice with probability ``hold`` while it stays legal, else draws
    uniformly among its legal choices (``mask``, the flat action mask)."""

    from experimental.neural_agent.skill_agent.runtime.scripted_bodies import STARTS
    from hytalegym.jax.combat.observation.v3.policy.layout import ARSENAL_POLICY_ACTION_HEAD_SIZES

    keys = jax.random.split(key, len(STARTS) + 1)
    keep = jax.random.uniform(keys[-1], previous.shape) < hold
    columns = []
    for head, start in enumerate(STARTS):
        allowed = mask[:, start:start + ARSENAL_POLICY_ACTION_HEAD_SIZES[head]].astype(jnp.bool_)
        drawn = jax.random.categorical(keys[head], jnp.where(allowed, 0.0, -jnp.inf))
        still = jnp.take_along_axis(allowed, previous[:, head:head + 1], axis=1)[:, 0]
        columns.append(jnp.where(keep[:, head] & still, previous[:, head], drawn))
    return jnp.stack(columns, axis=1).astype(jnp.int32)


def chaser(observation: Any, mask: jax.Array) -> jax.Array:
    """The duel's opponent: closes to 1.5 to 2.5 blocks, turns to face, swings in reach (scripted bodies)."""

    from experimental.neural_agent.skill_agent.runtime import scripted_bodies as sb

    batch = mask.shape[0]
    ticks = jnp.zeros(batch, jnp.int32)
    scalars = jnp.zeros((batch, 4), jnp.float32).at[:, 0].set(1.5).at[:, 1].set(2.5)
    slot = sb.HEAD["ability_none_plus_slots"]
    swing = sb.sword(observation, mask, ticks, scalars)
    return sb.keep_range(observation, mask, ticks, scalars).at[:, slot].set(swing[:, slot])


class Coverage:
    """Per token type and column: valid rows, non-zero rows, min, max, and up to 64 distinct values."""

    def __init__(self) -> None:
        widths = {name: len(cls.FEATURES) for name, cls in v3.TOKEN_TYPES_V3.items()}
        self.rows = dict.fromkeys(widths, 0)
        self.nonzero = {name: np.zeros(width) for name, width in widths.items()}
        self.low = {name: np.full(width, np.inf) for name, width in widths.items()}
        self.high = {name: np.full(width, -np.inf) for name, width in widths.items()}
        self.values: dict[str, list[set]] = {name: [set() for _ in range(width)] for name, width in widths.items()}

    def add(self, tokens: dict[str, tuple[Any, Any]]) -> None:
        for name, (features, valid) in tokens.items():
            rows = np.asarray(features)[np.asarray(valid)]
            if not len(rows):
                continue
            self.rows[name] += len(rows)
            self.nonzero[name] += (rows != 0).sum(axis=0)
            self.low[name] = np.minimum(self.low[name], rows.min(axis=0))
            self.high[name] = np.maximum(self.high[name], rows.max(axis=0))
            for column, seen in enumerate(self.values[name]):
                if len(seen) < 64:
                    seen.update(np.round(rows[:256, column], 4).tolist())

    def report(self) -> dict[str, Any]:
        """Per type: rows, columns, the counts of live (varying) columns, and the zero and constant columns."""

        out = {}
        for name, cls in v3.TOKEN_TYPES_V3.items():
            rows = self.rows[name]
            detail = {feature: {"nonzero": round(float(self.nonzero[name][index] / rows), 4),
                                "min": round(float(self.low[name][index]), 4),
                                "max": round(float(self.high[name][index]), 4),
                                "distinct": len(self.values[name][index])}
                      for index, feature in enumerate(cls.FEATURES)} if rows else {}
            zero = [feature for feature, entry in detail.items() if entry["nonzero"] == 0]
            constant = [feature for feature, entry in detail.items() if entry["nonzero"] > 0 and entry["distinct"] == 1]
            out[name] = {"rows": rows, "columns": len(cls.FEATURES), "live": len(detail) - len(zero) - len(constant),
                         "zero": zero, "constant": constant, "detail": detail}
        return out


def run(steps: Iterable[Step], batch: int, seed: int = 0) -> Coverage:
    coverage = Coverage()
    state = layer.initial_state(batch, persona=pe.sample_personas(jax.random.key(seed), batch))
    tick = jax.jit(lambda observation, carry, action, adapter: layer.step(observation, carry, last_action=action,
                                                                         **adapter))
    for observation, action, adapter in steps:
        tokens, state = tick(observation, state, action, adapter)
        coverage.add(tokens)
    return coverage


def duel_steps(ticks: int, batch: int) -> Iterator[Step]:
    """Sword, shortbow and battleaxe in turn; the two face each other 3 to 12 blocks apart at the start."""

    from experimental.neural_agent.skill_agent.tasks.duel import BODY_Y
    from experimental.neural_agent.skill_agent.teaching.lookahead import VIEW_SECTOR_RADIANS
    from experimental.neural_agent.tasks.duel_arena import make_duel_arena
    from hytalegym.jax.combat.observation.v3.policy.layout import neutral_arsenal_policy_action_factors

    tables, effects = registry_tables(), effect_tables()
    for run_index, loadouts in enumerate((("iron_sword", "iron_sword"), ("iron_shortbow", "iron_sword"),
                                          ("iron_battleaxe", "iron_sword"))):
        arena = make_duel_arena(batch, loadouts, view_sector_full_angle_radians=VIEW_SECTOR_RADIANS,
                                legal_observation_sensor_range=True, max_health=(200.0, 200.0),
                                arrows=(200 if "bow" in loadouts[0] else 0, 0))
        rng = np.random.default_rng(run_index)
        distance, angle = rng.uniform(3.0, 12.0, batch), rng.uniform(-np.pi, np.pi, batch)
        position = np.zeros((batch, 2, 3), np.float32)
        position[:, :, 1] = BODY_Y
        position[:, 1, 0], position[:, 1, 2] = distance * np.sin(angle), distance * np.cos(angle)
        facing = np.degrees(np.arctan2(-np.sin(angle), -np.cos(angle)))  # the gym's forward is (-sin, -cos) of yaw
        yaw = (np.stack([facing, facing + 180.0], axis=1) + 180.0) % 360.0 - 180.0
        state, observation, opponent, mask, opponent_mask = arena.reset(
            jax.random.split(jax.random.key(run_index), batch), entity_position=jnp.asarray(position),
            entity_health=jnp.full((batch, 2), 200.0, jnp.float32), entity_yaw=jnp.asarray(yaw, jnp.float32))
        step, pick, chase = jax.jit(arena.step), jax.jit(random_legal), jax.jit(chaser)
        factors = neutral_arsenal_policy_action_factors(batch)
        for tick in range(ticks):
            yield observation, gen7_action(factors), {"body_yaw_degrees": state.arsenal.combat.yaw[:, 0],
                                                      "tables": tables, "effects": effects,
                                                      "reset": jnp.full((batch,), tick == 0)}
            factors = pick(jax.random.key(1000 * run_index + tick), mask, factors)
            state, observation, opponent, mask, opponent_mask, _, _ = step(
                state, factors, chase(opponent, opponent_mask), jax.random.split(jax.random.key(tick), batch))


def region_steps(ticks: int, batch: int, library: str | None = None) -> Iterator[Step]:
    """The Region loader's default world and spawn pair, the target active; with the channel when it is on. ``library``
    names a Region library (``worlds.region.load_region``): ``region-library-pilot-v2-072`` carries the 0.7.2 block
    identity (6f180b6), so block candidates exist; the default library has none."""

    from experimental.neural_agent.skill_agent.state.senses import Light
    from hytalegym.jax.combat.observation.v3.policy.layout import neutral_arsenal_policy_action_factors
    from worlds.region import region_scene

    chosen = {"library": library} if library else {}
    scene = region_scene(weapons=("iron_sword",) * batch, target_active=True, **chosen)
    tables, effects = registry_tables(), effect_tables()
    reset, step, pick = jax.jit(scene.environment.reset), jax.jit(scene.environment.step), jax.jit(random_legal)
    state, policy, _ = reset(jax.random.split(jax.random.key(7), scene.batch))
    channel = getattr(state.learner_observation, "perception_v2", None) is not None
    print(f"region: batch {scene.batch}, reach {scene.maximum_distance}, channel {channel}", flush=True)
    factors = neutral_arsenal_policy_action_factors(scene.batch)
    for tick in range(ticks):
        light = state.light_tokens
        adapter = {"tables": tables, "effects": effects, "reset": jnp.full((scene.batch,), tick == 0),
                   "light": Light(light.token_f32, light.token_mask & light.available[:, None]),
                   "action_surface": state.action_surface, "block_reach_blocks": float(scene.maximum_distance)}
        if not channel:
            adapter["body_yaw_degrees"] = state.runtime.combat.yaw[:, 0]
        yield state.learner_observation, gen7_action(factors), adapter
        factors = pick(jax.random.key(3000 + tick), state.action_mask, factors)
        state, policy = step(state, policy, factors, jax.random.split(jax.random.key(tick), scene.batch))[:2]


SCENARIOS = {"duel": (duel_steps, 8), "region": (region_steps, 4)}
"""Name -> (its steps, its default batch)."""


def _group(names: Sequence[str]) -> str:
    counts: dict[str, int] = {}
    for name in names:
        key = re.sub(r"_\d+$", "_N", name)
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{key}x{count}" if count > 1 else key for key, count in counts.items()) or "-"


def compare(runs: dict[str, dict]) -> str:
    """A table of the columns carrying information per type and run, then the columns whose state (live, constant,
    zero) differs between the runs that have rows."""

    def state(entry: dict, column: str) -> str:
        if not entry["rows"]:
            return ""
        return "zero" if column in entry["zero"] else "const" if column in entry["constant"] else "live"

    labels = list(runs)
    lines = ["type".ljust(16) + "".join(f"{label:>24s}" for label in labels)]
    for name in v3.TOKEN_TYPES_V3:
        cells = [f"{e['live']}+{len(e['constant'])}c/{e['columns']} r{e['rows']}" if e["rows"] else "no rows"
                 for e in (runs[label][name] for label in labels)]
        lines.append(name.ljust(16) + "".join(f"{cell:>24s}" for cell in cells))
    for name, cls in v3.TOKEN_TYPES_V3.items():
        differing: dict[tuple, list[str]] = {}
        for column in cls.FEATURES:
            states = tuple(state(runs[label][name], column) for label in labels)
            if len({s for s in states if s}) > 1:
                differing.setdefault(states, []).append(column)
        if differing:
            lines.append(f"== {name}")
        for states, columns in differing.items():
            pairs = " / ".join(f"{label}:{value or '-'}" for label, value in zip(labels, states))
            lines.append(f"   {pairs}: {_group(columns)}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("run", help="run a scenario and write its coverage report")
    one.add_argument("scenario", choices=sorted(SCENARIOS))
    one.add_argument("--out", type=Path, required=True)
    one.add_argument("--ticks", type=int, default=300)
    one.add_argument("--batch", type=int)
    one.add_argument("--library", help="region only: the Region library to load (worlds.region.load_region)")
    both = commands.add_parser("compare", help="line up reports: LABEL=PATH ...")
    both.add_argument("reports", nargs="+")
    options = parser.parse_args(argv)
    if options.command == "run":
        steps, batch = SCENARIOS[options.scenario]
        batch = options.batch or batch
        if options.library and options.scenario != "region":
            parser.error("--library applies to the region scenario only")
        extra = {"library": options.library} if options.library else {}
        report = run(steps(options.ticks, batch, **extra), batch).report()
        options.out.write_text(json.dumps(report, indent=1))
        runs = {options.scenario: report}
    else:
        runs = {label: json.loads(Path(path).read_text()) for label, path in (r.split("=", 1) for r in options.reports)}
    print(compare(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
