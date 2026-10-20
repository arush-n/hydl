"""The lookahead teacher (``../docs/TRAINING_PLAN.md`` section 7, step L3): a privileged controller that plans by
simulation, and the L3 measurement that sets it against the scripted combat priority.

Every ``replan`` ticks the teacher forks the simulator's state and rolls each candidate move ``horizon`` ticks forward
against the opponent, all candidates at once (``jax.vmap`` over a ``jax.lax.scan`` of the environment's step). It
scores each by damage dealt minus ``weight_taken`` times damage taken, plus ``KILL_SCORE`` for a kill and minus it for
a death, and plays the best for the next ``replan`` ticks. The opponent inside a rollout is the environment's own: the
production opponent today, a league member's policy once the two-actor arena (step L1) exists.

The moves (``CANDIDATES``): wait; strike with ability slot 1, 2, 3 or 4, pressed on the move's first tick only; guard;
step in (run), out, left or right (walk). In every move the head and the body turn to the target. The compass is taken
against the actual body yaw, as the scripted bodies take it.

The teacher reads the simulator's state (teachers may; students never do): positions, yaws and health, and it steps
forked copies of the environment. It is a teacher and a baseline, never the shipped skill.

Measured (2026-10-20, ``LEDGER.md``): on gen 7, 16 lanes, the production opponent at 4 times its health, 600 ticks,
horizon 24, replan 4, it won 64 of 64 fights against four weapons and never died, where the scripted priority won 31
and died 32 times. A plan costs about 0.1 s for 16 lanes on CPU, after a compile of 8 to 21 s (more under load).

CLI (the L3 measurement: one JSON line per weapon and controller; gen 7 through ``../runtime/gen7.py``)::

    PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym JAX_PLATFORMS=cpu .venv/bin/python -m \\
        experimental.neural_agent.skill_agent.teaching.lookahead --batch 16 --ticks 600 --horizon 24 --replan 4 \\
        --targets iron_sword,iron_battleaxe,iron_mace,iron_daggers --health-scale 4 --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import time
from functools import partial
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.contract.surface import ATTACK_PHASES
from experimental.neural_agent.skill_agent.runtime import scripted_bodies as sb
from experimental.neural_agent.skill_agent.teaching.teachers import (
    BODY_YAW_HEAD,
    COMPASS_DIRECTIONS,
    LOCOMOTION_HEAD,
    YAW_BINS,
    YAW_HEAD,
)
from hytalegym.jax.combat import arsenal_runtime_config, default_combat_params, hytale_0_5_7_loadouts
from hytalegym.jax.combat.arsenal.environment import make_arsenal_environment, open_flat_arsenal_world_capabilities
from hytalegym.jax.combat.observation.v3.policy.layout import neutral_arsenal_policy_action_factors
from hytalegym.jax.combat.observation.v3.policy.look_deltas import encode_look_delta
from hytalegym.jax.combat.runtime.math import _canonical_bearing, _normalize_degrees
from hytalegym.jax.combat.types import AGENT_ENTITY, GAIT_RUN, GAIT_WALK, TARGET_ENTITY

CANDIDATES = ("wait", "slot1", "slot2", "slot3", "slot4", "guard", "in", "out", "left", "right")
"""The moves. A strike's index is its ability slot, the ability head's choice."""
GUARD, IN, OUT, LEFT, RIGHT = (CANDIDATES.index(name) for name in ("guard", "in", "out", "left", "right"))
KILL_SCORE = 100.0
AGENT_WEAPON = "iron_sword"
ABILITY_HEAD, GUARD_HEAD = sb.HEAD["ability_none_plus_slots"], sb.HEAD["guard_off_on"]
SWEEP = ATTACK_PHASES.index("sweep")
VIEW_SECTOR_RADIANS = 2.0 * math.pi / 3.0
"""The legal view sector, 120 degrees: the open-flat combat arena's (``tasks/arenas.py``,
``COMBAT_OPEN_FLAT_VIEW_SECTOR_RADIANS``)."""


def _look(delta: jax.Array) -> jax.Array:
    return encode_look_delta(_normalize_degrees(delta), YAW_BINS, maximum_degrees=sb.MAXIMUM_TURN_DEGREES)


def macro(combat: Any, candidate: Any, tick: Any) -> jax.Array:
    """Move ``candidate`` (int, a scalar or [B]) on tick ``tick`` of the move, as factor rows int32[B, 12]."""

    batch = combat.position.shape[0]
    candidate = jnp.broadcast_to(jnp.asarray(candidate, jnp.int32), (batch,))
    agent, target = combat.position[:, AGENT_ENTITY], combat.position[:, TARGET_ENTITY]
    world = _canonical_bearing(target[:, 0] - agent[:, 0], target[:, 2] - agent[:, 2])
    right = -_normalize_degrees(world - combat.yaw[:, AGENT_ENTITY])  # the target's bearing right of the body
    factors = neutral_arsenal_policy_action_factors(batch)
    factors = factors.at[:, YAW_HEAD].set(_look(world - combat.desired_yaw))
    factors = factors.at[:, BODY_YAW_HEAD].set(_look(world - combat.desired_body_yaw))
    strike = (candidate >= 1) & (candidate <= 4) & (tick == 0)
    factors = factors.at[:, ABILITY_HEAD].set(jnp.where(strike, candidate, 0))
    factors = factors.at[:, GUARD_HEAD].set(jnp.where(candidate == GUARD, 1, 0))
    heading = jnp.select([candidate == IN, candidate == OUT, candidate == LEFT, candidate == RIGHT],
                         [right, right + 180.0, right - 90.0, right + 90.0], 0.0)
    gait = jnp.where(candidate == IN, GAIT_RUN, GAIT_WALK)
    direction = jnp.mod(jnp.round(heading / 45.0).astype(jnp.int32), COMPASS_DIRECTIONS) + 1
    moving = candidate >= IN
    factors = factors.at[:, LOCOMOTION_HEAD].set(jnp.where(moving, (gait - 1) * COMPASS_DIRECTIONS + direction, 0))
    return factors.astype(jnp.int32)


def step(environment: Any, carry: Any, factors: jax.Array, key: jax.Array) -> tuple:
    """One environment step on the factors the action mask allows, one key per lane."""

    keys = jax.random.split(key, factors.shape[0])
    return environment.step_factors(carry, sb.legal(factors, carry.action_mask), keys)


class LookaheadTeacher:
    """The teacher over ``environment`` (an arsenal environment). ``plan(carry, key)`` returns the best move per lane,
    int32[B], and every move's score, float32[C, B]. It forks: ``carry`` is not advanced."""

    def __init__(self, environment: Any, *, horizon: int = 24, weight_taken: float = 1.0) -> None:
        self.environment, self.horizon, self.weight_taken = environment, horizon, weight_taken
        self.plan = jax.jit(self._plan)

    def _plan(self, carry: Any, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        health = carry.runtime.combat.health

        def score(candidate: jax.Array) -> jax.Array:
            def tick(state: Any, index: jax.Array) -> tuple[Any, None]:
                factors = macro(state.runtime.combat, candidate, index)
                return step(self.environment, state, factors, jax.random.fold_in(key, index))[0], None

            after = jax.lax.scan(tick, carry, jnp.arange(self.horizon))[0].runtime.combat.health
            dealt = health[:, TARGET_ENTITY] - after[:, TARGET_ENTITY]
            taken = health[:, AGENT_ENTITY] - after[:, AGENT_ENTITY]
            return (dealt - self.weight_taken * taken + KILL_SCORE * (after[:, TARGET_ENTITY] <= 0)
                    - KILL_SCORE * (after[:, AGENT_ENTITY] <= 0))

        scores = jax.vmap(score)(jnp.arange(len(CANDIDATES)))
        return jnp.argmax(scores, axis=0), scores


def scripted_priority(observation: Any, mask: jax.Array) -> jax.Array:
    """The baseline: the decision layer's ``P_COMBAT`` rules (``jev/programs.py``) as one controller over the skill
    layer's bodies, on the legal observation, without P_COMBAT's evade gate and with keep_range closing in place of
    pursuit. Guard the opponent's wind-up within ``sb.GUARD_RANGE_BLOCKS``, punish its recovery within reach (the held
    weapon's, ``sb.reach``), hold 3.1 to 3.6 blocks in its sweep within 3, strike within reach, else close to 1.5 to
    2.5 blocks, no farther than the reach, facing it."""

    batch = mask.shape[0]
    ticks, none = jnp.zeros(batch, jnp.int32), jnp.zeros((batch, 4), jnp.float32)
    band = jnp.broadcast_to(jnp.asarray([3.1, 3.6, 0.0, 0.0], jnp.float32), (batch, 4))
    seen = sb.target(observation)
    reach = sb.reach(observation)
    zeros = jnp.zeros(batch, jnp.float32)
    close = jnp.stack([jnp.full(batch, 1.5, jnp.float32), jnp.minimum(reach, 2.5), zeros, zeros], axis=1)
    options = [
        (seen.seen & (seen.phase == sb.WINDUP) & (seen.distance <= sb.GUARD_RANGE_BLOCKS),
         sb.guard(observation, mask, ticks, none)),
        (seen.seen & (seen.phase == sb.RECOVERY) & (seen.distance <= reach),
         sb.punish(observation, mask, ticks, none)),
        (seen.seen & (seen.phase == SWEEP) & (seen.distance <= 3.0), sb.keep_range(observation, mask, ticks, band)),
        (seen.seen & (seen.distance <= reach), sb.sword(observation, mask, ticks, none)),
    ]
    chosen = sb.keep_range(observation, mask, ticks, close)
    for when, factors in reversed(options):
        chosen = jnp.where(when[:, None], factors, chosen)
    return chosen


def arena(batch: int, target: str, health_scale: float = 1.0) -> Any:
    """The open-flat duel: the agent with ``AGENT_WEAPON`` against the production opponent with ``target``'s weapon at
    ``health_scale`` times its health, in legal observation with the arena's view sector and the sensor range."""

    params = default_combat_params(microticks=1)
    params = params._replace(target_max_health=params.target_max_health * health_scale)
    return make_arsenal_environment(
        params,
        arsenal_runtime_config(hytale_0_5_7_loadouts([AGENT_WEAPON] * batch, target_profiles=[target] * batch),
                               legal_observation=True),
        maximum_turn_degrees=sb.MAXIMUM_TURN_DEGREES,
        world_capability_provider=partial(open_flat_arsenal_world_capabilities,
                                          view_sector_full_angle_radians=VIEW_SECTOR_RADIANS,
                                          legal_observation_sensor_range=True),
    )


def measure(environment: Any, batch: int, *, ticks: int, seed: int, teacher: LookaheadTeacher | None = None,
            replan: int = 4) -> dict[str, Any]:
    """One episode per lane, by ``teacher`` replanning every ``replan`` ticks, or by ``scripted_priority`` when it is
    None: kills, deaths and time-outs, damage dealt and taken, the ticks run, and for the teacher its moves and a
    plan's cost."""

    carry, observation = environment.reset(jax.random.split(jax.random.key(seed), batch))
    start = np.asarray(carry.runtime.combat.health)
    outcome = np.zeros(batch, int)  # 1 a kill, -1 a death, 0 a time-out
    over = np.zeros(batch, bool)
    dealt, taken = np.zeros(batch), np.zeros(batch)
    moves = np.zeros(len(CANDIDATES), int)
    plan_seconds = []
    current = jnp.zeros(batch, jnp.int32)
    for tick in range(ticks):
        if teacher is None:
            factors = scripted_priority(observation, jnp.asarray(carry.action_mask, jnp.bool_))
        else:
            if tick % replan == 0:
                begun = time.perf_counter()
                current = teacher.plan(carry, jax.random.key(10_000 + tick))[0]
                current.block_until_ready()
                plan_seconds.append(time.perf_counter() - begun)
                np.add.at(moves, np.asarray(current)[~over], 1)
            factors = macro(carry.runtime.combat, current, tick % replan)
        before = np.asarray(carry.runtime.combat.health)
        carry, observation, _, done, _ = step(environment, carry, factors, jax.random.key(seed * 1_000_000 + tick))
        after = np.asarray(carry.runtime.combat.health)
        live = ~over
        dealt[live] += np.maximum(before[live, TARGET_ENTITY] - after[live, TARGET_ENTITY], 0)
        taken[live] += np.maximum(before[live, AGENT_ENTITY] - after[live, AGENT_ENTITY], 0)
        finished = live & np.asarray(done)
        outcome[finished & (after[:, TARGET_ENTITY] <= 0)] = 1
        outcome[finished & (after[:, AGENT_ENTITY] <= 0) & (after[:, TARGET_ENTITY] > 0)] = -1
        over |= finished
        if over.all():
            break
    return {"controller": "scripted" if teacher is None else "teacher", "kills": int((outcome == 1).sum()),
            "deaths": int((outcome == -1).sum()), "timeouts": int((outcome == 0).sum()), "dealt": float(dealt.sum()),
            "taken": float(taken.sum()), "ticks": tick + 1, "start_health": start[0].tolist(),
            "plan_seconds_median": float(np.median(plan_seconds[1:])) if len(plan_seconds) > 1 else None,
            "first_plan_seconds": plan_seconds[0] if plan_seconds else None,
            "moves": dict(zip(CANDIDATES, moves.tolist())) if teacher is not None else None}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--ticks", type=int, default=600)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--replan", type=int, default=4)
    parser.add_argument("--weight-taken", type=float, default=1.0)
    parser.add_argument("--targets", default="iron_sword,iron_battleaxe,iron_mace,iron_daggers")
    parser.add_argument("--health-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--controllers", default="scripted,teacher")
    args = parser.parse_args(argv)
    for target in args.targets.split(","):
        environment = arena(args.batch, target, args.health_scale)
        teacher = LookaheadTeacher(environment, horizon=args.horizon, weight_taken=args.weight_taken)
        for controller in args.controllers.split(","):
            result = measure(environment, args.batch, ticks=args.ticks, seed=args.seed, replan=args.replan,
                             teacher=teacher if controller == "teacher" else None)
            print(json.dumps({"target": target, "scale": args.health_scale, **result}), flush=True)


__all__ = ["CANDIDATES", "KILL_SCORE", "VIEW_SECTOR_RADIANS", "LookaheadTeacher", "arena", "macro", "measure",
           "scripted_priority", "step"]


if __name__ == "__main__":
    main()
