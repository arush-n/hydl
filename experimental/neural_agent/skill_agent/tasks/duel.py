"""The duel world (``../docs/TRAINING_PLAN.md`` section 7, step L1): two bodies, each driven by a controller, on the
coordinator's two-actor arena (``experimental/neural_agent/tasks/duel_arena.py``: both entities take factors and the
built-in AI is off), and L1's mirror check.

A controller is a pure-JAX function ``(observation, mask, context) -> int32[B, 12]`` of one entity's own legal
observation (``LearnerCombatObservationV3``), its action mask (bool[B, 124]) and a ``Context``: the ticks since that
entity last lost health, which the entity knows from its own health (the decision layer's belief keeps the same
count, ``ticks_since_damage``). ``priority`` is ``teaching/lookahead.scripted_priority`` as one. A controller's factors
are kept to what the mask allows (``scripted_bodies.legal``).

``mirrored`` places each lane's pair symmetrically under a half turn about the vertical axis through the pair's
centre: ``distance`` blocks apart on an axis at a random angle, both turned by the same random offset from facing each
other, at equal health; ``arena`` puts both bodies in the same role (``ROLES``) at the same maximum health. With the
same controller on both bodies, a fair arena gives each entity the same chance. The mirror check (L1's gate, restated
with the coordinator on 2026-10-20 because the duel has no built-in AI to reproduce) counts wins and damage per entity
and tests both for parity within 2 standard errors. Each entity waits its own random start delay, drawn from the same
distribution for both: in an exact mirror the scripted priority swings on the same tick as its twin, each guards the
other's wind-up, and no hit lands (measured: 0 damage over 300 ticks on 8 lanes).

CLI (the mirror check, one JSON line per weapon; gen 7 through ``../runtime/gen7.py``)::

    PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym JAX_PLATFORMS=cpu .venv/bin/python -m \\
        experimental.neural_agent.skill_agent.tasks.duel --batch 64 --ticks 600 --seed 0 \\
        --loadouts iron_sword,iron_battleaxe,iron_mace,iron_daggers
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.runtime.scripted_bodies import legal
from experimental.neural_agent.skill_agent.teaching.lookahead import VIEW_SECTOR_RADIANS, scripted_priority
from hytalegym.jax.combat.observation.v3.policy.layout import neutral_arsenal_policy_action_factors
from experimental.neural_agent.tasks.duel_arena import make_duel_arena

Controller = Callable[[Any, jax.Array, "Context"], jax.Array]
"""``controller(observation, mask bool[B, 124], context) -> factors int32[B, 12]``, for one entity."""
BODY_Y = 65.0
"""The default reset's body height on the open-flat floor."""
CENTRE_XZ = (0.5, 0.0)
"""Where each lane's pair is centred."""
HEALTH = 61.0
"""Both roles' maximum and starting health (the target role's default maximum)."""
ROLES = ("Trork_Brawler", "Trork_Brawler")
"""Both bodies play the matchup's target role (the coordinator's ``roles=``, f2f5957), so both slots take the same
paired fields: box, eye height, speed, turn rate, head limits. In the default matchup (Kweebec_Razorleaf against
Trork_Brawler) the eye heights and boxes differ and a mirrored pair's first hits land a tick apart (#103b). The
seat gaps the roles do not close are in the duel step and the engine (#110: slot 1's selector aims with a stale
target head yaw, #112; the control lock after damage to slot 0 and regeneration on slot 0 only, #108)."""


class Context(NamedTuple):
    """What a controller knows besides its observation: the ticks since its entity last lost health, int32[B]
    (``NEVER_DAMAGED`` until the first hit)."""

    ticks_since_damage: jax.Array


NEVER_DAMAGED = 1_000_000


def priority(observation: Any, mask: jax.Array, context: Context) -> jax.Array:
    """``teaching/lookahead.scripted_priority`` as a controller (it reads no context)."""

    return scripted_priority(observation, mask)


class Bout(NamedTuple):
    """Per lane: ``winner`` 0 or 1, or -1 when both fell or neither did; the ticks run; each entity's damage dealt
    (the health the other lost), float[B, 2]."""

    winner: np.ndarray
    ticks: np.ndarray
    dealt: np.ndarray


def arena(batch: int, loadouts: Sequence[str] = ("iron_sword", "iron_sword"),
          roles: tuple[str, str] | None = ROLES) -> Any:
    """The duel with the open-flat combat arena's legal view sector and the sensor range on, both bodies in
    ``roles`` (``None``: the matchup's Kweebec against Trork) at ``HEALTH`` maximum health."""

    return make_duel_arena(batch, tuple(loadouts), view_sector_full_angle_radians=VIEW_SECTOR_RADIANS,
                           legal_observation_sensor_range=True, max_health=(HEALTH, HEALTH), roles=roles)


def mirrored(batch: int, seed: int, *, distance: tuple[float, float] = (3.0, 9.0), offset_degrees: float = 60.0,
             health: float = HEALTH) -> dict[str, jax.Array]:
    """``reset``'s placement arrays, symmetric under a half turn about each pair's centre (the module docstring)."""

    rng = np.random.default_rng(seed)
    apart = rng.uniform(*distance, batch)
    axis = np.deg2rad(rng.uniform(-180.0, 180.0, batch))
    offset = rng.uniform(-offset_degrees, offset_degrees, batch)
    toward = np.stack([np.cos(axis), np.sin(axis)], axis=1)  # (x, z), from entity 0 toward entity 1
    ends = [np.asarray(CENTRE_XZ) + sign * toward * apart[:, None] / 2.0 for sign in (-1.0, 1.0)]
    position = np.stack([np.stack([end[:, 0], np.full(batch, BODY_Y), end[:, 1]], axis=1) for end in ends], axis=1)
    facing = np.rad2deg(np.arctan2(-toward[:, 0], -toward[:, 1]))  # the yaw whose forward (-sin, -cos) is ``toward``
    yaw = np.mod(np.stack([facing + offset, facing + 180.0 + offset], axis=1) + 180.0, 360.0) - 180.0
    return {"entity_position": jnp.asarray(position, jnp.float32),
            "entity_health": jnp.full((batch, 2), health, jnp.float32), "entity_yaw": jnp.asarray(yaw, jnp.float32)}


def fight(duel: Any, first: Controller, second: Controller, placement: dict[str, jax.Array], *, ticks: int,
          seed: int, delays: np.ndarray | None = None) -> Bout:
    """Up to ``ticks`` ticks from ``placement``: entity 0 by ``first``, entity 1 by ``second``. An entity stays
    neutral until its start delay (``delays`` int[B, 2], ticks; none by default)."""

    batch = placement["entity_health"].shape[0]
    waits = jnp.zeros((batch, 2), jnp.int32) if delays is None else jnp.asarray(delays, jnp.int32)
    neutral = neutral_arsenal_policy_action_factors(batch)

    @jax.jit
    def tick(state, observation0, observation1, mask0, mask1, key, index, since):
        chosen0 = legal(first(observation0, mask0, Context(since[:, 0])), mask0)
        chosen1 = legal(second(observation1, mask1, Context(since[:, 1])), mask1)
        factors0 = jnp.where((index >= waits[:, 0])[:, None], chosen0, neutral)
        factors1 = jnp.where((index >= waits[:, 1])[:, None], chosen1, neutral)
        return duel.step(state, factors0, factors1, jax.random.split(key, batch))

    state, observation0, observation1, mask0, mask1 = duel.reset(jax.random.split(jax.random.key(seed), batch),
                                                                  **placement)
    health = np.asarray(state.arsenal.combat.health, dtype=float)
    since = np.full((batch, 2), NEVER_DAMAGED, np.int32)
    over = np.zeros(batch, bool)
    winner, length, dealt = np.full(batch, -1), np.full(batch, ticks), np.zeros((batch, 2))
    for index in range(ticks):
        state, observation0, observation1, mask0, mask1, info, done = tick(
            state, observation0, observation1, mask0, mask1, jax.random.key(seed * 1_000_000 + index), index,
            jnp.asarray(since))
        after = np.asarray(info["health"], dtype=float)
        since = np.where(after < health, 0, np.minimum(since + 1, NEVER_DAMAGED)).astype(np.int32)
        live = ~over
        dealt[live, 0] += np.maximum(health[live, 1] - after[live, 1], 0.0)
        dealt[live, 1] += np.maximum(health[live, 0] - after[live, 0], 0.0)
        finished = live & np.asarray(done)
        winner[finished & (after[:, 1] <= 0) & (after[:, 0] > 0)] = 0
        winner[finished & (after[:, 0] <= 0) & (after[:, 1] > 0)] = 1
        length[finished] = index + 1
        over |= finished
        health = after
        if over.all():
            break
    return Bout(winner, length, dealt)


def mirror_check(batch: int, *, ticks: int, seed: int, loadout: str = "iron_sword", delay_ticks: int = 20,
                 controller: Controller = priority) -> dict[str, Any]:
    """L1's gate: ``controller`` on both bodies from ``mirrored`` placements, each entity after its own start delay
    drawn from 0 to ``delay_ticks``. Parity holds when the wins differ by at most 2 standard errors of a fair coin over
    the decided bouts, and the mean paired difference in damage dealt is within 2 of its standard errors."""

    delays = np.random.default_rng(seed + 1).integers(0, delay_ticks + 1, (batch, 2))
    bout = fight(arena(batch, (loadout, loadout)), controller, controller, mirrored(batch, seed), ticks=ticks,
                 seed=seed, delays=delays)
    wins = [int((bout.winner == entity).sum()) for entity in (0, 1)]
    decided = wins[0] + wins[1]
    difference = bout.dealt[:, 0] - bout.dealt[:, 1]
    error = float(difference.std(ddof=1) / np.sqrt(batch)) if batch > 1 else 0.0
    return {"loadout": loadout, "lanes": batch, "delay_ticks": delay_ticks, "wins": wins, "undecided": batch - decided,
            "dealt_mean": [round(float(value), 3) for value in bout.dealt.mean(axis=0)],
            "dealt_difference_mean": round(float(difference.mean()), 3), "dealt_difference_se": round(error, 3),
            "ticks_median": float(np.median(bout.ticks)),
            "wins_parity": bool(abs(wins[0] - wins[1]) <= 2.0 * np.sqrt(max(decided, 1))),
            "damage_parity": bool(abs(difference.mean()) <= 2.0 * error)}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--ticks", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loadouts", default="iron_sword")
    parser.add_argument("--delay-ticks", type=int, default=20)
    args = parser.parse_args(argv)
    for loadout in args.loadouts.split(","):
        print(json.dumps(mirror_check(args.batch, ticks=args.ticks, seed=args.seed, loadout=loadout,
                                      delay_ticks=args.delay_ticks)), flush=True)


__all__ = ["BODY_Y", "CENTRE_XZ", "HEALTH", "NEVER_DAMAGED", "ROLES", "Bout", "Context", "Controller", "arena", "fight",
           "main", "mirror_check", "mirrored", "priority"]


if __name__ == "__main__":
    main()
