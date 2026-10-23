"""Sim compiler for TaskSpec on the gen-7 render (2026-10-23).

Compiled objectives and their episodes:

* ``ReachPlace``, ``HoldArea``: the duel arena with entity 1 parked far away (inert under neutral factors, the duel
  test pins this). The scripted reach policy walks to the centre and stops inside the radius.
* ``DefeatTarget``: the duel arena with entity 1 = the matchup target role (``roles=``), a scripted melee: walk to
  2 blocks, then tap the sword (ability slot 1) every 12 ticks while in reach.
* ``EvadeTurret``: the duel arena with entity 1 = ``skeleton_archer_turret_bow`` driven by ``turret_controller``. The
  body backs off to the range edge (minus a margin), then stands and faces the turret.
* ``SurviveSeconds``: the stock arsenal PPO environment (``make_arsenal_ppo_environment``), whose opponent is the
  built-in Trork_Brawler. The duel arena cannot host it: ``make_duel_arena`` forces ``target_active`` off.

The reward and termination are the ``task_spec`` functions the live side also calls.

Import this module only after the gen-7 mount (``skill_agent/runtime/gen7.py``): it imports hytalegym.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.tasks.arenas import COMBAT_OPEN_FLAT_VIEW_SECTOR_RADIANS as SECTOR
from experimental.neural_agent.tasks.duel_arena import make_duel_arena
from experimental.neural_agent.tasks.task_spec import (
    DefeatTarget,
    EvadeTurret,
    HoldArea,
    ReachPlace,
    ReachPlaceRegion,
    SurviveSeconds,
    TaskSpec,
    defeat_done,
    defeat_step,
    evade_done,
    evade_step,
    evade_success,
    hold_done,
    hold_step,
    objective_ticks,
    reach_done,
    reach_step,
    survive_done,
    survive_step,
    survive_success,
    time_limit_ticks,
    validate_spec,
)
from experimental.neural_agent.tasks.turret_controller import initial_state as turret_initial_state
from experimental.neural_agent.tasks.turret_controller import turret_factors
from hytalegym.jax.combat import arsenal_runtime_config, default_combat_params, hytale_0_5_7_loadouts
from hytalegym.jax.combat.arsenal.environment import open_flat_arsenal_world_capabilities
from hytalegym.jax.combat.observation.v3.policy.layout import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES,
    neutral_arsenal_policy_action_factors,
)
from hytalegym.jax.combat.observation.v3.policy.look_deltas import look_delta_values
from hytalegym.jax.training.arsenal import make_arsenal_ppo_environment
from hytalegym.rulesets import load_combat_ruleset

FORWARD_WALK = 1  # locomotion_gait_compass choice 1: walk, first compass direction (forward); duel test constant
BACKWARD_WALK = 5  # choice 5: walk, the backward direction (layout.py split_locomotion_choice)
TURN_HEAD = "body_yaw_delta_bins"  # the body's own steer (layout.py); the camera head is yaw_delta_bins
LOCOMOTION_HEAD = "locomotion_gait_compass"
ABILITY_HEAD = "ability_none_plus_slots"
TURN_LIMIT_DEGREES = 45.0  # the duel's maximum_turn_degrees default
TURN_BINS = 9  # the head's size (layout.py: 9-bin geometric grid)
DUEL_DEFAULT_HEALTH = (105.0, 61.0)  # the reset health the duel test measures at its default (test_na_duel_arena)
SWORD_SLOT_ONE = 2  # ability_none_plus_slots choice k+1 is ability slot k (layout.py:722): slot 1 is the sword tap
SWORD_STAND_BLOCKS = 2.0  # walk until this close, then stop and tap (the sword hits for 10 at 2.5 blocks, measured)
SWORD_REACH_BLOCKS = 2.5
TAP_EVERY_TICKS = 12
DEFEAT_START_BLOCKS = 2.5  # the target stands this far in front of the body (probe-measured, inside the sword reach)
TURRET_START_BLOCKS = 12.0  # the turret stands this far in front of the body, facing it
EVADE_MARGIN_BLOCKS = 3.0  # the body backs off until this far inside the turret's range, then stands
TURRET_PROFILE = "skeleton_archer_turret_bow"

_TURN_TABLE = np.asarray(look_delta_values(TURN_BINS, maximum_degrees=TURN_LIMIT_DEGREES), dtype=np.float64)


def _per_row(center, rows: int) -> np.ndarray:
    """A centre as float64 [rows, 3]: one point for every row, or a point per row (the duel's target or turret)."""

    center = np.asarray(center, dtype=np.float64)
    return np.broadcast_to(center, (rows, 3)) if center.ndim == 1 else center


def _planar(position, center) -> np.ndarray:
    """Planar (x-z) distance from ``position`` [B, 3] to a centre that is one point or one per row."""

    position = np.asarray(position, dtype=np.float64)
    center = _per_row(center, position.shape[0])
    return np.hypot(position[:, 0] - center[:, 0], position[:, 2] - center[:, 2])


def _head(name: str) -> int:
    return ARSENAL_POLICY_ACTION_HEAD_NAMES.index(name)


def _nearest_turn_bin(delta_degrees: np.ndarray) -> np.ndarray:
    """Nearest representable body-steer bin for each signed delta (ties to the lower index, as the encoder does)."""

    return np.argmin(np.abs(delta_degrees[:, None] - _TURN_TABLE[None, :]), axis=1).astype(np.int32)


def _steer(position, yaw_degrees, center) -> np.ndarray:
    """The body-steer bin that turns ``position`` toward ``center`` (clipped to the turn limit)."""

    centers = _per_row(center, position.shape[0])
    dx = centers[:, 0] - position[:, 0]
    dz = centers[:, 2] - position[:, 2]
    desired = np.degrees(np.arctan2(-dx, -dz))
    error = (desired - np.asarray(yaw_degrees, dtype=np.float64) + 180.0) % 360.0 - 180.0
    return _nearest_turn_bin(np.clip(error, -TURN_LIMIT_DEGREES, TURN_LIMIT_DEGREES))


def scripted_reach_factors(position, yaw_degrees, center, radius: float) -> np.ndarray:
    """The scripted policy: face the target, steer by the body-yaw bins, walk forward until inside ``radius``.

    ``position`` [B, 3] and ``yaw_degrees`` [B] are the controlled body's; returns int32 [B, heads] factors with
    every other head neutral. Heading convention (the duel test's): yaw 0 faces -z, so the forward vector is
    ``(-sin yaw, -cos yaw)`` on (x, z); the desired yaw is ``atan2(-dx, -dz)``.
    """

    position = np.asarray(position, dtype=np.float64)
    distance = _planar(position, center)
    factors = np.array(neutral_arsenal_policy_action_factors(position.shape[0]), dtype=np.int32)
    factors[:, _head(TURN_HEAD)] = _steer(position, yaw_degrees, center)
    factors[:, _head(LOCOMOTION_HEAD)] = np.where(distance > radius, FORWARD_WALK, 0)
    return factors


def scripted_melee_factors(position, yaw_degrees, target, tick: int) -> np.ndarray:
    """Walk to ``SWORD_STAND_BLOCKS`` of the target, stop there, and tap the sword every ``TAP_EVERY_TICKS`` ticks."""

    factors = scripted_reach_factors(position, yaw_degrees, target, SWORD_STAND_BLOCKS)
    close = _planar(position, target) <= SWORD_REACH_BLOCKS
    if tick % TAP_EVERY_TICKS == 0:
        factors[close, _head(ABILITY_HEAD)] = SWORD_SLOT_ONE
    return factors


def scripted_evade_factors(position, yaw_degrees, turret, turret_range: float) -> np.ndarray:
    """Face the turret; back off while closer than ``turret_range - EVADE_MARGIN_BLOCKS``; then stand."""

    factors = scripted_reach_factors(position, yaw_degrees, turret, 1e9)  # steer only: never walk forward
    away = _planar(position, turret) < turret_range - EVADE_MARGIN_BLOCKS
    factors[:, _head(LOCOMOTION_HEAD)] = np.where(away, BACKWARD_WALK, 0)
    return factors


def _combat_arrays(state):
    """Host arrays of a duel state: positions [B, 2, 3], yaw [B, 2], health [B, 2]."""

    return np.asarray(state.arsenal.combat.position), np.asarray(state.arsenal.combat.yaw), np.asarray(state.arsenal.combat.health)


@dataclass(frozen=True, slots=True)
class SimRun:
    """One batch of episodes. ``success_tick`` is -1 for a row that never succeeded."""

    success: np.ndarray  # bool [B]
    success_tick: np.ndarray  # int [B]
    reward: np.ndarray  # float [B], the per-tick reward summed over the episode
    ticks: int  # ticks stepped (stops at all-done or the time limit)
    time_limit_ticks: int
    traces: dict = field(default_factory=dict)  # per-objective measurements (health, distance), host arrays

    @property
    def arrived(self) -> np.ndarray:  # the ReachPlace names of the pilot
        return self.success

    @property
    def arrival_tick(self) -> np.ndarray:
        return self.success_tick

    @property
    def limit(self) -> int:
        return self.time_limit_ticks


SimReachRun = SimRun  # the pilot's name


class SimTask:
    """A compiled TaskSpec on the gen-7 render. ``run(seed)`` plays one batch to termination."""

    def __init__(self, spec: TaskSpec, batch: int) -> None:
        validate_spec(spec)
        self.spec = spec
        self.batch = int(batch)
        self.objective = spec.objectives[0]
        self.limit = time_limit_ticks(spec)
        body = spec.bodies[0]
        self.body = body
        if isinstance(self.objective, SurviveSeconds):
            self._build_ppo()
            return
        if isinstance(self.objective, ReachPlaceRegion):
            self._build_region()
            return
        loadouts, roles = self._duel_loadouts(body)
        self.arena = make_duel_arena(
            self.batch,
            loadouts,
            view_sector_full_angle_radians=SECTOR,
            legal_observation_sensor_range=True,
            roles=roles,
        )

    def _duel_loadouts(self, body):
        objective = self.objective
        if isinstance(objective, EvadeTurret):
            return (body.loadout, TURRET_PROFILE), None
        if isinstance(objective, DefeatTarget):
            matchup = load_combat_ruleset()["matchup"]
            return (body.loadout, body.loadout), (matchup["agent_role"], objective.role)
        return (body.loadout, body.loadout), None

    def _build_region(self) -> None:
        from worlds.region import region_scene  # the Region world: the arsenal's Region environment, no jax edits

        scene = region_scene(library=self.spec.world.library, weapons=(self.body.loadout,))
        if self.batch != scene.batch:
            raise ValueError(f"this Region library builds {scene.batch} episode(s); compile with batch={scene.batch}")
        self.scene = scene

    def _build_ppo(self) -> None:
        config = arsenal_runtime_config(
            hytale_0_5_7_loadouts([self.body.loadout] * self.batch, target_profiles=[self.body.loadout] * self.batch),
            legal_observation=True,
        )
        self.ppo_config = config
        self.ppo = make_arsenal_ppo_environment(
            default_combat_params(microticks=1),
            config,
            maximum_turn_degrees=TURN_LIMIT_DEGREES,
            world_capability_provider=lambda state, p: open_flat_arsenal_world_capabilities(
                state, p, config=config, legal_observation_sensor_range=True, view_sector_full_angle_radians=SECTOR
            ),
        )

    def _place(self) -> dict:
        body = self.body
        objective = self.objective
        if isinstance(objective, DefeatTarget):
            x, y, z = body.spawn
            other = (x, y, z - DEFEAT_START_BLOCKS)
            yaw = (body.yaw_degrees, 0.0)
        elif isinstance(objective, EvadeTurret):
            x, y, z = body.spawn
            other = (x, y, z - TURRET_START_BLOCKS)
            yaw = (body.yaw_degrees, 180.0)  # the turret faces back toward the body
        else:
            other = self.spec.park
            yaw = (body.yaw_degrees, 0.0)
        positions = np.tile(np.asarray([body.spawn, other], dtype=np.float32), (self.batch, 1, 1))
        yaws = np.tile(np.asarray(yaw, dtype=np.float32), (self.batch, 1))
        healths = np.tile(np.asarray(DUEL_DEFAULT_HEALTH, dtype=np.float32), (self.batch, 1))
        return dict(entity_position=positions, entity_health=healths, entity_yaw=yaws)

    def run(self, seed: int) -> SimRun:
        """Play one batch from reset to termination (success, death, the target's death, or the time limit)."""

        if isinstance(self.objective, SurviveSeconds):
            return self._run_ppo(seed)
        if isinstance(self.objective, ReachPlaceRegion):
            return self._run_region(seed)
        if isinstance(self.objective, EvadeTurret):
            return self._run_evade(seed)
        return self._run_duel(seed)

    # --- duel objectives: ReachPlace, HoldArea, DefeatTarget ----------------------------------------------------

    def _run_duel(self, seed: int) -> SimRun:
        objective = self.objective
        base = jax.random.key(seed)
        state, *_ = self.arena.reset(jax.random.split(base, self.batch), **self._place())
        neutral = jnp.asarray(neutral_arsenal_policy_action_factors(self.batch), dtype=jnp.int32)
        goal = objective_ticks(objective.seconds) if isinstance(objective, HoldArea) else 0
        center = np.asarray(getattr(objective, "center", (0.0, 0.0, 0.0)), dtype=np.float64)
        position, yaw, health = _combat_arrays(state)
        distance = _planar(position[:, 0], center) if not isinstance(objective, DefeatTarget) else None
        prev_distance = distance
        prev_target = health[:, 1].astype(np.float64)
        streak = np.zeros(self.batch, dtype=np.int64)
        success = np.zeros(self.batch, dtype=bool)
        success_tick = np.full(self.batch, -1, dtype=np.int64)
        reward = np.zeros(self.batch, dtype=np.float64)
        done = np.zeros(self.batch, dtype=bool)
        tick = 0
        while not bool(np.all(done)) and tick < self.limit:
            if isinstance(objective, DefeatTarget):
                f0 = scripted_melee_factors(position[:, 0], yaw[:, 0], position[:, 1], tick)
            else:
                f0 = scripted_reach_factors(position[:, 0], yaw[:, 0], center, objective.radius)
            step_keys = jax.random.split(jax.random.fold_in(base, tick), self.batch)
            out = self.arena.step(state, jnp.asarray(f0, dtype=jnp.int32), neutral, step_keys)
            state = out[0]
            tick += 1
            position, yaw, health = _combat_arrays(state)
            active = ~done
            if isinstance(objective, ReachPlace):
                distance = _planar(position[:, 0], center)
                step_reward, now = reach_step(prev_distance, distance, success, self.spec)
                reward += np.where(active, step_reward, 0.0)
                newly = now & ~success
                success_tick = np.where(newly, tick, success_tick)
                success = now
                prev_distance = distance
                done = reach_done(success, tick, self.limit) | done
            elif isinstance(objective, HoldArea):
                inside = _planar(position[:, 0], center) <= objective.radius
                step_reward, streak = hold_step(streak, inside, goal, self.spec)
                reward += np.where(active, step_reward, 0.0)
                now = hold_done(streak, tick, goal, self.limit)
                success = streak >= goal
                success_tick = np.where(success & (success_tick < 0), tick, success_tick)
                done = done | now
            else:
                target = health[:, 1].astype(np.float64)
                reward += np.where(active, defeat_step(prev_target, target, self.spec), 0.0)
                prev_target = target
                now = defeat_done(target, tick, self.limit)
                success = target <= 0.0
                success_tick = np.where(success & (success_tick < 0), tick, success_tick)
                done = done | now
        return SimRun(success, success_tick, reward, tick, self.limit, {"final_health": health.tolist()})

    # --- EvadeTurret --------------------------------------------------------------------------------------------

    def _run_evade(self, seed: int) -> SimRun:
        objective = self.objective
        base = jax.random.key(seed)
        state, _, obs1, *_ = self.arena.reset(jax.random.split(base, self.batch), **self._place())
        goal = objective_ticks(objective.seconds)
        turret = turret_initial_state(self.batch, jax.random.fold_in(base, 1))
        turret_position = np.asarray(state.arsenal.combat.position)[:, 1]
        position = np.asarray(state.arsenal.combat.position)[:, 0]
        yaw = np.asarray(state.arsenal.combat.yaw)[:, 0]
        health = np.asarray(state.arsenal.combat.health)[:, 0].astype(np.float64)
        prev_distance = _planar(position, turret_position)
        alive = np.ones(self.batch, dtype=bool)
        reward = np.zeros(self.batch, dtype=np.float64)
        done = np.zeros(self.batch, dtype=bool)
        death_tick = np.full(self.batch, -1, dtype=np.int64)
        tick = 0
        while not bool(np.all(done)) and tick < self.limit:
            f0 = scripted_evade_factors(position, yaw, turret_position, objective.turret_range)
            f1, turret = turret_factors(obs1, turret, tick)
            step_keys = jax.random.split(jax.random.fold_in(base, tick), self.batch)
            out = self.arena.step(state, jnp.asarray(f0, jnp.int32), jnp.asarray(f1, jnp.int32), step_keys)
            state, _, obs1 = out[0], out[1], out[2]
            tick += 1
            position = np.asarray(state.arsenal.combat.position)[:, 0]
            yaw = np.asarray(state.arsenal.combat.yaw)[:, 0]
            health = np.asarray(state.arsenal.combat.health)[:, 0].astype(np.float64)
            distance = _planar(position, turret_position)
            now_alive = health > 0.0
            active = ~done
            reward += np.where(active, evade_step(alive, now_alive, prev_distance, distance, self.spec), 0.0)
            death_tick = np.where(alive & ~now_alive, tick, death_tick)
            alive = now_alive
            in_range = distance <= objective.turret_range
            done = done | evade_done(alive, in_range, tick, goal, self.limit)
            prev_distance = distance
        in_range = _planar(position, turret_position) <= objective.turret_range
        success = evade_success(alive, in_range, tick, goal)
        success_tick = np.where(success, tick, -1)
        traces = {"death_tick": death_tick.tolist(), "final_health": health.tolist(), "final_distance": prev_distance.tolist()}
        return SimRun(success, success_tick, reward, tick, self.limit, traces)

    # --- ReachPlaceRegion (the arsenal's Region environment, from worlds.region.region_scene) -------------------

    def _run_region(self, seed: int) -> SimRun:
        objective = self.objective
        env = self.scene.environment
        state, observation, *_ = env.reset(jax.random.split(jax.random.key(seed), self.batch))
        position = np.asarray(state.runtime.combat.position)[:, 0]
        yaw = np.asarray(state.runtime.combat.yaw)[:, 0]
        spawn = position[0].tolist()
        center = position + np.asarray([objective.offset[0], 0.0, objective.offset[1]])  # per row, from the spawn
        prev = _planar(position, center)
        arrived = np.zeros(self.batch, dtype=bool)
        arrival_tick = np.full(self.batch, -1, dtype=np.int64)
        reward = np.zeros(self.batch, dtype=np.float64)
        done = np.zeros(self.batch, dtype=bool)
        tick = 0
        while not bool(np.all(done)) and tick < self.limit:
            factors = scripted_reach_factors(position, yaw, center, objective.radius)
            step_keys = jax.random.split(jax.random.fold_in(jax.random.key(seed), tick), self.batch)
            state, observation, *_ = env.step(state, observation, jnp.asarray(factors, jnp.int32), step_keys)
            tick += 1
            position = np.asarray(state.runtime.combat.position)[:, 0]
            yaw = np.asarray(state.runtime.combat.yaw)[:, 0]
            distance = _planar(position, center)
            step_reward, now = reach_step(prev, distance, arrived, self.spec)
            reward += np.where(~done, step_reward, 0.0)
            arrival_tick = np.where(now & ~arrived, tick, arrival_tick)
            arrived = now
            prev = distance
            done = done | reach_done(arrived, tick, self.limit)
        return SimRun(arrived, arrival_tick, reward, tick, self.limit,
                      {"spawn": spawn, "centre": center[0].tolist(), "final": position[0].tolist()})

    # --- SurviveSeconds (stock PPO environment, built-in Trork_Brawler) ------------------------------------------

    def _run_ppo(self, seed: int) -> SimRun:
        objective = self.objective
        base = jax.random.key(seed)
        keys = jax.random.split(base, self.batch)
        state, observation, *_ = self.ppo.reset(keys)
        goal = objective_ticks(objective.seconds)
        alive = np.ones(self.batch, dtype=bool)
        reward = np.zeros(self.batch, dtype=np.float64)
        done = np.zeros(self.batch, dtype=bool)
        death_tick = np.full(self.batch, -1, dtype=np.int64)
        tick = 0
        health = np.asarray(state.runtime.combat.health)[:, 0].astype(np.float64)
        while not bool(np.all(done)) and tick < self.limit:
            position = np.asarray(state.runtime.combat.position)[:, 0]
            yaw = np.asarray(state.runtime.combat.yaw)[:, 0]
            target = np.asarray(state.runtime.combat.position)[:, 1]
            factors = scripted_melee_factors(position, yaw, target, tick)
            step_keys = jax.random.split(jax.random.fold_in(base, tick), self.batch)
            state, observation, *_ = self.ppo.step(state, observation, jnp.asarray(factors, jnp.int32), step_keys)
            tick += 1
            health = np.asarray(state.runtime.combat.health)[:, 0].astype(np.float64)
            now_alive = health > 0.0
            active = ~done
            reward += np.where(active, survive_step(alive, now_alive, self.spec), 0.0)
            death_tick = np.where(alive & ~now_alive, tick, death_tick)
            alive = now_alive
            done = done | survive_done(alive, tick, goal, self.limit)
        success = survive_success(alive, tick, goal)
        success_tick = np.where(success, tick, -1)
        return SimRun(success, success_tick, reward, tick, self.limit, {"death_tick": death_tick.tolist(), "final_health": health.tolist()})


def compile_sim(spec: TaskSpec, batch: int) -> SimTask:
    """Compile ``spec`` to a gen-7 episode. Raises ``NotImplementedError`` outside the compiled scope."""

    return SimTask(spec, batch)


__all__ = [
    "BACKWARD_WALK",
    "FORWARD_WALK",
    "SimReachRun",
    "SimRun",
    "SimTask",
    "compile_sim",
    "scripted_evade_factors",
    "scripted_melee_factors",
    "scripted_reach_factors",
]
