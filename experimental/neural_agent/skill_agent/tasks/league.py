"""The league (``../docs/TRAINING_PLAN.md`` section 7, step L2): the decision layer's combat styles as pure-JAX
controllers over the skill layer's bodies, played round robin in the duel world (``duel.py``), and their ratings.

The controllers mirror the style programs (``experimental/neural_agent/jev/combat_styles.py``, 0b22cda) and take
their ``NA_PROGRAM`` names, as agreed with the coordinator on 2026-10-20. The programs are the deployed side; these
controllers run where the decision layer cannot, inside the duel and the lookahead teacher's rollouts. Each follows
its program's priority list, and the first rule that holds acts alone:

- ``COUNTER``: guard the target's wind-up; punish its recovery within reach; else hold 3.1 to 3.6 blocks.
- ``PRESSURE``: evade when hurt in the last 40 ticks and below 0.3 health; guard the wind-up below 0.5 health; strike
  within reach in any phase; else close in.
- ``HIT_AND_RUN``: evade when hurt in the last 40 ticks and below 0.5 health; back out inside 3.1 blocks; strike from
  3.1 to 3.5 blocks; else close in.
- ``PRIORITY``: ``duel.priority``, the scripted priority (P_COMBAT's rules without its evade gate).

Two differences from the programs. Closing in is keep_range to 1.5 to 2.5 blocks while the target is seen, and a
search turn while it is not (the pursuit body's search, ``runtime/neural_pursuit.py``), where the programs run the
trained pursuit network. And the sword's hold_seconds is not read: the bodies build no charged holds yet, so the
styles strike alike.

The round robin plays every ordered pair of distinct styles from the same placements and start delays, so each pair
meets in both seats with each style keeping its own delay. Both seats are needed because the duel's seats are not
equal on gen 7 (LEDGER 2026-10-20). A bout is won by a kill, else on points (more damage dealt), else drawn. The
ratings are a Bradley-Terry fit of the bout scores, draws counting half, on the Elo scale (400 log10 odds, mean
1500). L2's gate: the styles beat each other in a cycle, not in a line, and no style beats every other; "beats" means
a score above one half by more than 2 standard errors.

CLI (one JSON line; gen 7 through ``../runtime/gen7.py``)::

    PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym JAX_PLATFORMS=cpu .venv/bin/python -m \\
        experimental.neural_agent.skill_agent.tasks.league --lanes 32 --ticks 600 --seed 0 \\
        --styles COUNTER,PRESSURE,HIT_AND_RUN,PRIORITY --loadout iron_sword
"""

from __future__ import annotations

import argparse
import itertools
import json
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.runtime import scripted_bodies as sb
from experimental.neural_agent.skill_agent.tasks import duel
from experimental.neural_agent.skill_agent.teaching.teachers import BODY_YAW_HEAD, YAW_BINS
from hytalegym.jax.combat.observation.v1.schema.contract import SELF_FLOAT_FEATURES
from hytalegym.jax.combat.observation.v3.policy.look_deltas import encode_look_delta

ON_DAMAGE_TICKS = 40
"""Hurt within this many ticks: the decision layer's ``on_damage`` window (``executive/goal_language.py``)."""
EVADE_HEALTH_FRACTION = 0.5
"""``EVADE_WHEN_HURT``'s health gate (``jev/programs.py``)."""
PRESSURE_EVADE_FRACTION, PRESSURE_GUARD_FRACTION = 0.3, 0.5
KEEP_RANGE_BAND = (3.1, 3.6)
CLOSE_BAND = (1.5, 2.5)
SEARCH_TURN_DEGREES = 45.0
"""The pursuit body's search turn, per tick (``runtime/neural_pursuit.py``)."""
SELF_HEALTH = SELF_FLOAT_FEATURES.index("health_fraction")


def _band(batch: int, band: tuple[float, float]) -> jax.Array:
    return jnp.broadcast_to(jnp.asarray([*band, 0.0, 0.0], jnp.float32), (batch, 4))


def _first(rules: Sequence[tuple[jax.Array, jax.Array]], otherwise: jax.Array) -> jax.Array:
    """The factors of the first rule that holds, else ``otherwise``."""

    chosen = otherwise
    for holds, factors in reversed(rules):
        chosen = jnp.where(holds[:, None], factors, chosen)
    return chosen


def _close_in(observation: Any, mask: jax.Array, seen: jax.Array) -> jax.Array:
    batch = mask.shape[0]
    factors = sb.keep_range(observation, mask, jnp.zeros(batch, jnp.int32), _band(batch, CLOSE_BAND))
    search = encode_look_delta(jnp.full((batch,), -SEARCH_TURN_DEGREES, jnp.float32), YAW_BINS,
                               maximum_degrees=sb.MAXIMUM_TURN_DEGREES)
    return sb.legal(factors.at[:, BODY_YAW_HEAD].set(jnp.where(seen, factors[:, BODY_YAW_HEAD], search)), mask)


def counter(observation: Any, mask: jax.Array, context: duel.Context) -> jax.Array:
    batch = mask.shape[0]
    ticks, none = jnp.zeros(batch, jnp.int32), jnp.zeros((batch, 4), jnp.float32)
    target = sb.target(observation)
    return _first([
        (target.seen & (target.phase == sb.WINDUP), sb.guard(observation, mask, ticks, none)),
        (target.seen & (target.phase == sb.RECOVERY) & (target.distance < sb.REACH_BLOCKS),
         sb.punish(observation, mask, ticks, none)),
    ], sb.keep_range(observation, mask, ticks, _band(batch, KEEP_RANGE_BAND)))


def pressure(observation: Any, mask: jax.Array, context: duel.Context) -> jax.Array:
    batch = mask.shape[0]
    ticks, none = jnp.zeros(batch, jnp.int32), jnp.zeros((batch, 4), jnp.float32)
    target = sb.target(observation)
    health = observation.base.self_f32[:, SELF_HEALTH]
    hurt = context.ticks_since_damage < ON_DAMAGE_TICKS
    return _first([
        (hurt & (health < PRESSURE_EVADE_FRACTION), sb.evader(observation, mask, ticks, none)),
        (target.seen & (target.phase == sb.WINDUP) & (health < PRESSURE_GUARD_FRACTION),
         sb.guard(observation, mask, ticks, none)),
        (target.seen & (target.distance < sb.REACH_BLOCKS), sb.sword(observation, mask, ticks, none)),
    ], _close_in(observation, mask, target.seen))


def hit_and_run(observation: Any, mask: jax.Array, context: duel.Context) -> jax.Array:
    batch = mask.shape[0]
    ticks, none = jnp.zeros(batch, jnp.int32), jnp.zeros((batch, 4), jnp.float32)
    target = sb.target(observation)
    health = observation.base.self_f32[:, SELF_HEALTH]
    hurt = context.ticks_since_damage < ON_DAMAGE_TICKS
    near, _ = KEEP_RANGE_BAND
    return _first([
        (hurt & (health < EVADE_HEALTH_FRACTION), sb.evader(observation, mask, ticks, none)),
        (target.seen & (target.distance < near), sb.keep_range(observation, mask, ticks, _band(batch, KEEP_RANGE_BAND))),
        (target.seen & (target.distance >= near) & (target.distance < sb.REACH_BLOCKS),
         sb.sword(observation, mask, ticks, none)),
    ], _close_in(observation, mask, target.seen))


STYLES: dict[str, duel.Controller] = {"COUNTER": counter, "PRESSURE": pressure, "HIT_AND_RUN": hit_and_run,
                                      "PRIORITY": duel.priority}


def select(styles: Sequence[duel.Controller], ids: jax.Array) -> duel.Controller:
    """One controller over the batch: lane ``i`` runs ``styles[ids[i]]``."""

    def controller(observation: Any, mask: jax.Array, context: duel.Context) -> jax.Array:
        chosen = styles[0](observation, mask, context)
        for index, style in enumerate(styles[1:], start=1):
            chosen = jnp.where((ids == index)[:, None], style(observation, mask, context), chosen)
        return chosen

    return controller


def ratings(scores: np.ndarray, games: np.ndarray, iterations: int = 500) -> np.ndarray:
    """Bradley-Terry strengths of ``scores[i, j]`` (i's points against j, out of ``games[i, j]``) on the Elo scale,
    mean 1500. Each pair carries half a point each way as a prior, so a style that never scores stays finite."""

    scores, games = scores + 0.5 * (games > 0), games + 1.0 * (games > 0)
    strength = np.ones(len(scores))
    for _ in range(iterations):
        expected = (games / (strength[:, None] + strength[None, :])).sum(axis=1)
        strength = scores.sum(axis=1) / expected
        strength /= np.exp(np.log(strength).mean())
    return 1500.0 + 400.0 * np.log10(strength)


def _cycle(beats: np.ndarray) -> list[int] | None:
    """A directed cycle in ``beats`` (i beats j), as style indices, or None."""

    for size in range(3, len(beats) + 1):
        for order in itertools.permutations(range(len(beats)), size):
            if order[0] == min(order) and all(beats[order[k], order[(k + 1) % size]] for k in range(size)):
                return list(order)
    return None


def round_robin(names: Sequence[str], *, lanes: int, ticks: int, seed: int, loadout: str = "iron_sword",
                delay_ticks: int = 20) -> dict[str, Any]:
    """Every ordered pair of distinct styles over ``lanes`` placements, as one batch (the module docstring)."""

    styles = [STYLES[name] for name in names]
    pairs = list(itertools.permutations(range(len(names)), 2))
    first = np.repeat([a for a, _ in pairs], lanes)
    second = np.repeat([b for _, b in pairs], lanes)
    base = duel.mirrored(lanes, seed)
    placement = {key: jnp.tile(value, (len(pairs),) + (1,) * (value.ndim - 1)) for key, value in base.items()}
    waits = np.random.default_rng(seed + 1).integers(0, delay_ticks + 1, (lanes, 2))
    delays = np.concatenate([waits if a < b else waits[:, ::-1] for a, b in pairs])
    bout = duel.fight(duel.arena(len(first), (loadout, loadout)), select(styles, jnp.asarray(first)),
                      select(styles, jnp.asarray(second)), placement, ticks=ticks, seed=seed, delays=delays)
    points = np.where(bout.dealt[:, 0] > bout.dealt[:, 1], 1.0, np.where(bout.dealt[:, 0] < bout.dealt[:, 1], 0.0, 0.5))
    seat0 = np.where(bout.winner == 0, 1.0, np.where(bout.winner == 1, 0.0, points))  # seat 0's score per bout
    size = len(names)
    scores, games, squares = np.zeros((size, size)), np.zeros((size, size)), np.zeros((size, size))
    kills = np.zeros((size, size), int)
    for (a, b), chunk in zip(pairs, np.split(np.arange(len(first)), len(pairs))):
        for i, j, score, won in ((a, b, seat0[chunk], bout.winner[chunk] == 0),
                                 (b, a, 1.0 - seat0[chunk], bout.winner[chunk] == 1)):
            scores[i, j] += score.sum()
            squares[i, j] += (score ** 2).sum()
            games[i, j] += len(chunk)
            kills[i, j] += int(won.sum())
    rate = np.divide(scores, games, out=np.full_like(scores, 0.5), where=games > 0)
    spread = np.sqrt(np.maximum(np.divide(squares, games, out=np.zeros_like(scores), where=games > 0) - rate ** 2, 0.0))
    error = np.divide(spread, np.sqrt(np.maximum(games - 1, 1)))
    beats = (rate - 0.5) > 2.0 * np.maximum(error, 1e-9)
    np.fill_diagonal(beats, False)
    cycle = _cycle(beats)
    dominant = [names[i] for i in range(size) if beats[i, [j for j in range(size) if j != i]].all()]
    return {"loadout": loadout, "styles": list(names), "lanes_per_pair": 2 * lanes, "ticks": ticks,
            "score": np.round(rate, 3).tolist(), "score_se": np.round(error, 3).tolist(), "kills": kills.tolist(),
            "decided_by_kill": float(np.mean(bout.winner >= 0)), "seat0_score": round(float(seat0.mean()), 3),
            "ratings": {name: round(float(value), 1) for name, value in zip(names, ratings(scores, games))},
            "beats": [[names[i], names[j]] for i in range(size) for j in range(size) if beats[i, j]],
            "cycle": [names[i] for i in cycle] if cycle else None, "dominant": dominant,
            "gate": bool(cycle) and not dominant}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lanes", type=int, default=32)
    parser.add_argument("--ticks", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--styles", default=",".join(STYLES))
    parser.add_argument("--loadout", default="iron_sword")
    parser.add_argument("--delay-ticks", type=int, default=20)
    args = parser.parse_args(argv)
    print(json.dumps(round_robin(args.styles.split(","), lanes=args.lanes, ticks=args.ticks, seed=args.seed,
                                 loadout=args.loadout, delay_ticks=args.delay_ticks)), flush=True)


__all__ = ["CLOSE_BAND", "KEEP_RANGE_BAND", "ON_DAMAGE_TICKS", "SEARCH_TURN_DEGREES", "STYLES", "counter",
           "hit_and_run", "main", "pressure", "ratings", "round_robin", "select"]


if __name__ == "__main__":
    main()
