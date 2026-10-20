"""The archery range: the shortbow body against a still or moving body on the duel (2026-10-20), per draw tier and
distance: the arrows loosed, the hits, the damage and the damage per second, so the decision layer can choose its
hold from measurements (P_GUERRILLA, #105).

Entity 0 is ``scripted_bodies.shortbow`` with the lane's ``hold_seconds`` (the tier's threshold), facing the other body
``distance`` blocks away with its head level. With ``face`` (the default) its body also turns toward the target, as
``avoid`` or ``keep_range`` would on the locomotion scope in the composition: the bow owns the head, and the head
turns at most 45 degrees off the body, so a bow alone loses a target that runs past that. Entity 1 carries a sword
and is driven by the mover:

- ``still``: every head neutral;
- ``strafe``: runs across the line of fire, turning to face the shooter, and reverses after runs of ``STRAFE_TICKS``
  ticks drawn per lane;
- ``approach``: runs at the shooter until ``APPROACH_STOP_BLOCKS`` away, then stands.

The shooter stops tapping after ``ticks`` ticks; the run continues ``DRAIN_TICKS`` ticks so arrows in flight land.
The damage per second counts the damage of the arrows loosed in the window over the window's seconds (1/30 s a tick,
the duel's ``nominal_dt``). Both bodies have ``max_health`` 1000, so nothing dies.

CLI (one JSON line per tier and distance; gen 7 through ``../runtime/gen7.py``)::

    PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym JAX_PLATFORMS=cpu .venv/bin/python -m \\
        experimental.neural_agent.skill_agent.tasks.archery --distances 4.5,8 --ticks 300 --mover still
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.runtime import scripted_bodies as sb
from experimental.neural_agent.skill_agent.tasks.duel import BODY_Y
from experimental.neural_agent.skill_agent.teaching.lookahead import VIEW_SECTOR_RADIANS
from experimental.neural_agent.tasks.duel_arena import make_duel_arena
from hytalegym.jax.combat.observation.v3.policy.layout import neutral_arsenal_policy_action_factors
from hytalegym.jax.combat.types import GAIT_RUN

LOADOUTS = ("iron_shortbow", "iron_sword")
HEALTH = 1000.0
TICK_SECONDS = 1.0 / 30.0
DRAIN_TICKS = 60
STRAFE_TICKS = (20, 60)
"""A strafing body reverses after a run of 20 to 60 ticks (0.7 to 2 s), drawn per lane and run, so no tier's shot
cycle locks onto the reversals (measured 2026-10-20: a fixed 45 did with tier 4's 50-tick cycle, every arrow released
just after a reversal)."""
APPROACH_STOP_BLOCKS = 3.0
MOVERS = ("still", "strafe", "approach")


def _strafe_sides(batch: int, ticks: int, seed: int) -> jax.Array:
    """Per lane and tick, the compass side a strafing body runs to (3 right, 7 left), reversing after each run."""

    rng = np.random.default_rng(seed + 1)
    sides = np.empty((batch, ticks), np.int32)
    for lane in range(batch):
        start, side = 0, 3
        while start < ticks:
            run = int(rng.integers(STRAFE_TICKS[0], STRAFE_TICKS[1] + 1))
            sides[lane, start:start + run] = side
            start, side = start + run, 10 - side
    return jnp.asarray(sides)


def _mover(name: str, batch: int, sides: jax.Array):
    """Entity 1's controller ``(observation, mask, index) -> factors`` for ``name`` (module docstring)."""

    neutral = neutral_arsenal_policy_action_factors(batch)
    if name == "still":
        return lambda observation, mask, index: neutral

    def move(observation, mask, index):
        shooter = sb.target(observation)
        if name == "strafe":
            side = sides[:, index]
            going = shooter.seen
        else:
            side = jnp.int32(1)
            going = shooter.seen & (shooter.distance > APPROACH_STOP_BLOCKS)
        factors = sb._set(neutral, "locomotion_gait_compass", (GAIT_RUN - 1) * sb.COMPASS_DIRECTIONS + side, going)
        return sb.legal(sb._set(factors, "body_yaw_delta_bins", sb._turn(shooter.bearing), shooter.seen), mask)

    return move


def archery(tiers: Sequence[int], distances: Sequence[float], *, ticks: int, mover: str = "still", seed: int = 0,
            arrows: int = 200, roles: tuple[str, str] | None = None, face: bool = True) -> list[dict[str, Any]]:
    """Every (tier, distance) pair as one lane; one result row per lane."""

    if mover not in MOVERS:
        raise ValueError(f"mover must be one of {MOVERS}, got {mover!r}")
    lanes = [(int(tier), float(distance)) for distance in distances for tier in tiers]
    batch = len(lanes)
    arena = make_duel_arena(batch, LOADOUTS, view_sector_full_angle_radians=VIEW_SECTOR_RADIANS,
                            legal_observation_sensor_range=True, max_health=(HEALTH, HEALTH), roles=roles,
                            arrows=(arrows, 0))
    apart = np.asarray([distance for _, distance in lanes], np.float32)
    zeros, height = np.zeros(batch), np.full(batch, BODY_Y)
    position = np.stack([np.stack([zeros, height, zeros], 1), np.stack([apart, height, zeros], 1)], 1)
    yaw = np.stack([np.full(batch, -90.0), np.full(batch, 90.0)], 1)  # facing each other along x
    state, observation0, observation1, mask0, mask1 = arena.reset(
        jax.random.split(jax.random.key(seed), batch), entity_position=jnp.asarray(position, jnp.float32),
        entity_health=jnp.full((batch, 2), HEALTH, jnp.float32), entity_yaw=jnp.asarray(yaw, jnp.float32))
    holds = jnp.asarray([sb.SHORTBOW_THRESHOLD_SECONDS[tier] for tier, _ in lanes], jnp.float32)
    scalars = jnp.zeros((batch, 4), jnp.float32).at[:, 0].set(holds)
    neutral = neutral_arsenal_policy_action_factors(batch)
    move = _mover(mover, batch, _strafe_sides(batch, ticks + DRAIN_TICKS, seed))

    @jax.jit
    def tick(state, observation0, observation1, mask0, mask1, key, index):
        shooting = sb.shortbow(observation0, mask0, jnp.zeros(batch, jnp.int32), scalars)
        if face:
            seen = sb.target(observation0)
            shooting = sb.legal(sb._set(shooting, "body_yaw_delta_bins", sb._turn(seen.bearing), seen.seen), mask0)
        factors0 = jnp.where(index < ticks, shooting, neutral)
        return arena.step(state, factors0, move(observation1, mask1, index), jax.random.split(key, batch))

    loosed, hits, damage = np.zeros(batch, int), np.zeros(batch, int), np.zeros(batch)
    first_hit = np.full(batch, -1)
    health = np.asarray(state.arsenal.combat.health)[:, 1].astype(float)
    for index in range(ticks + DRAIN_TICKS):
        before = np.asarray(state.arsenal.arsenal.projectiles.active).sum(axis=1)
        state, observation0, observation1, mask0, mask1, info, _ = tick(
            state, observation0, observation1, mask0, mask1, jax.random.key(seed * 1_000_000 + index), index)
        loosed += np.asarray(state.arsenal.arsenal.projectiles.active).sum(axis=1) > before
        after = np.asarray(info["health"])[:, 1].astype(float)
        landed = after < health
        hits += landed
        damage += np.maximum(health - after, 0.0)
        first_hit = np.where(landed & (first_hit < 0), index, first_hit)
        health = after
    seconds = ticks * TICK_SECONDS
    return [{"tier": tier, "distance": distance, "mover": mover, "face": face, "ticks": ticks, "loosed": int(loosed[lane]),
             "hits": int(hits[lane]), "hit_rate": round(float(hits[lane] / loosed[lane]), 3) if loosed[lane] else None,
             "damage": round(float(damage[lane]), 2), "damage_per_second": round(float(damage[lane] / seconds), 2),
             "first_hit_tick": int(first_hit[lane])} for lane, (tier, distance) in enumerate(lanes)]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tiers", default="0,1,2,3,4")
    parser.add_argument("--distances", default="4.5,8")
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--mover", default="still", choices=MOVERS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arrows", type=int, default=200)
    parser.add_argument("--roles", default=None, help="one role for both bodies, e.g. Trork_Brawler")
    parser.add_argument("--no-face", action="store_true", help="the bow alone: the shooter's body does not turn")
    args = parser.parse_args(argv)
    roles = None if args.roles is None else (args.roles, args.roles)
    for row in archery([int(tier) for tier in args.tiers.split(",")],
                       [float(distance) for distance in args.distances.split(",")], ticks=args.ticks,
                       mover=args.mover, seed=args.seed, arrows=args.arrows, roles=roles, face=not args.no_face):
        print(json.dumps(row), flush=True)


__all__ = ["APPROACH_STOP_BLOCKS", "DRAIN_TICKS", "MOVERS", "STRAFE_TICKS", "archery", "main"]


if __name__ == "__main__":
    main()
