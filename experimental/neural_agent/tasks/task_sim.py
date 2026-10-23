"""Sim compiler for TaskSpec: the pilot reach task on the gen-7 render (2026-10-23).

Built on the duel arena (``duel_arena.make_duel_arena``): entity 0 is the controlled body, entity 1 is parked
far away and stays still under neutral factors (the duel test pins this, ``test_neutral_factors_keep_both_bodies_still``).
The reward and termination are ``task_spec.reach_step`` and ``reach_done``, the same functions the live side uses.

Import this module only after the gen-7 mount (``skill_agent/runtime/gen7.py``): it imports hytalegym.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.tasks.arenas import COMBAT_OPEN_FLAT_VIEW_SECTOR_RADIANS
from experimental.neural_agent.tasks.duel_arena import make_duel_arena
from experimental.neural_agent.tasks.task_spec import (
    TaskSpec,
    planar_distance,
    reach_done,
    reach_step,
    time_limit_ticks,
    validate_pilot,
)
from hytalegym.jax.combat.observation.v3.policy.layout import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES,
    neutral_arsenal_policy_action_factors,
)
from hytalegym.jax.combat.observation.v3.policy.look_deltas import look_delta_values

FORWARD_WALK = 1  # locomotion_gait_compass choice 1: walk, first compass direction (forward); duel test constant
TURN_HEAD = "body_yaw_delta_bins"  # the body's own steer (layout.py); the camera head is yaw_delta_bins
LOCOMOTION_HEAD = "locomotion_gait_compass"
TURN_LIMIT_DEGREES = 45.0  # the duel's maximum_turn_degrees default
TURN_BINS = 9  # the head's size (layout.py: 9-bin geometric grid)
DUEL_DEFAULT_HEALTH = (105.0, 61.0)  # the reset health the duel test measures at its default (test_na_duel_arena)


def _head(name: str) -> int:
    return ARSENAL_POLICY_ACTION_HEAD_NAMES.index(name)


def _nearest_turn_bin(delta_degrees: np.ndarray) -> np.ndarray:
    """Nearest representable body-steer bin for each signed delta (ties to the lower index, as the encoder does)."""

    table = np.asarray(look_delta_values(TURN_BINS, maximum_degrees=TURN_LIMIT_DEGREES), dtype=np.float64)
    return np.argmin(np.abs(delta_degrees[:, None] - table[None, :]), axis=1).astype(np.int32)


def scripted_reach_factors(position, yaw_degrees, center, radius: float) -> np.ndarray:
    """The scripted policy: face the target, steer by the body-yaw bins, walk forward until inside ``radius``.

    ``position`` [B, 3] and ``yaw_degrees`` [B] are the controlled body's; returns int32 [B, heads] factors with
    every other head neutral. Heading convention (the duel test's): yaw 0 faces -z, so the forward vector is
    ``(-sin yaw, -cos yaw)`` on (x, z); the desired yaw is ``atan2(-dx, -dz)``.
    """

    position = np.asarray(position, dtype=np.float64)
    dx = np.asarray(center, dtype=np.float64)[0] - position[:, 0]
    dz = np.asarray(center, dtype=np.float64)[2] - position[:, 2]
    desired = np.degrees(np.arctan2(-dx, -dz))
    error = (desired - np.asarray(yaw_degrees, dtype=np.float64) + 180.0) % 360.0 - 180.0
    clipped = np.clip(error, -TURN_LIMIT_DEGREES, TURN_LIMIT_DEGREES)
    distance = planar_distance(position, center)
    factors = np.array(neutral_arsenal_policy_action_factors(position.shape[0]), dtype=np.int32)
    factors[:, _head(TURN_HEAD)] = _nearest_turn_bin(clipped)
    factors[:, _head(LOCOMOTION_HEAD)] = np.where(distance > radius, FORWARD_WALK, 0)
    return factors


@dataclass(frozen=True, slots=True)
class SimReachRun:
    """One batch of episodes. ``arrival_tick`` is -1 for a row that never arrived."""

    arrived: np.ndarray  # bool [B]
    arrival_tick: np.ndarray  # int [B]
    reward: np.ndarray  # float [B], the per-tick reach reward summed over the episode
    ticks: int  # ticks stepped (stops at all-done or the time limit)
    time_limit_ticks: int


class SimTask:
    """A compiled TaskSpec on the gen-7 duel arena. ``run(seed, policy)`` plays one batch to termination."""

    def __init__(self, spec: TaskSpec, batch: int) -> None:
        validate_pilot(spec)
        self.spec = spec
        self.batch = int(batch)
        self.arena = make_duel_arena(
            self.batch,
            (spec.bodies[0].loadout, spec.bodies[0].loadout),
            view_sector_full_angle_radians=COMBAT_OPEN_FLAT_VIEW_SECTOR_RADIANS,
            legal_observation_sensor_range=True,
        )
        self.center = np.asarray(spec.objectives[0].center, dtype=np.float64)
        self.radius = float(spec.objectives[0].radius)
        self.limit = time_limit_ticks(spec)

    def _place(self) -> dict:
        body = self.spec.bodies[0]
        positions = np.tile(np.asarray([body.spawn, self.spec.park], dtype=np.float32), (self.batch, 1, 1))
        yaws = np.tile(np.asarray([body.yaw_degrees, 0.0], dtype=np.float32), (self.batch, 1))
        healths = np.tile(np.asarray(DUEL_DEFAULT_HEALTH, dtype=np.float32), (self.batch, 1))
        return dict(entity_position=positions, entity_health=healths, entity_yaw=yaws)

    def run(self, seed: int, policy: Callable = scripted_reach_factors) -> SimReachRun:
        """Play one batch from reset to all-arrived or the time limit. Entity 1 gets neutral factors throughout."""

        base = jax.random.key(seed)
        state, *_ = self.arena.reset(jax.random.split(base, self.batch), **self._place())
        neutral = jnp.asarray(neutral_arsenal_policy_action_factors(self.batch), dtype=jnp.int32)
        position = np.asarray(state.arsenal.combat.position)[:, 0]
        yaw = np.asarray(state.arsenal.combat.yaw)[:, 0]
        prev = planar_distance(position, self.center)
        arrived = np.zeros(self.batch, dtype=bool)
        arrival_tick = np.full(self.batch, -1, dtype=np.int64)
        reward = np.zeros(self.batch, dtype=np.float64)
        tick = 0
        while not bool(np.all(reach_done(arrived, tick, self.limit))):
            factors0 = jnp.asarray(policy(position, yaw, self.center, self.radius), dtype=jnp.int32)
            step_keys = jax.random.split(jax.random.fold_in(base, tick), self.batch)
            state = self.arena.step(state, factors0, neutral, step_keys)[0]
            tick += 1
            position = np.asarray(state.arsenal.combat.position)[:, 0]
            yaw = np.asarray(state.arsenal.combat.yaw)[:, 0]
            distance = planar_distance(position, self.center)
            step_reward, arrived_now = reach_step(prev, distance, arrived, self.spec)
            reward += step_reward
            arrival_tick = np.where(arrived_now & ~arrived, tick, arrival_tick)
            arrived = arrived_now
            prev = distance
        return SimReachRun(arrived, arrival_tick, reward, tick, self.limit)


def compile_sim(spec: TaskSpec, batch: int) -> SimTask:
    """Compile ``spec`` to the gen-7 duel arena. Raises ``NotImplementedError`` outside the pilot's scope."""

    return SimTask(spec, batch)


__all__ = ["FORWARD_WALK", "SimReachRun", "SimTask", "compile_sim", "scripted_reach_factors"]
