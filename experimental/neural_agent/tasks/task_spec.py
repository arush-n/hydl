"""TaskSpec: one task definition, compiled to the gen-7 sim and to the live bridge (2026-10-23).

The spec is data. ``task_sim`` compiles it to a gen-7 episode (reset arguments, a scripted policy, a per-tick reward);
``task_live`` compiles it to bridge reset options and, where the bridge supports it, the same reward. The reward and
termination functions here are the single definition both sides call (numpy only, no hytalegym import), so the two
compilers cannot drift apart on scoring.

Objectives (each has a shared reward and termination here, a sim compiler branch and a live option mapping):

* ``ReachPlace(center, radius)``: stand within ``radius`` of ``center`` on the x-z plane (pilot).
* ``ReachPlaceRegion(offset, radius)``: the same on a Region world, the point offset from the fixture's spawn.
* ``HoldArea(center, radius, seconds)``: stay inside for a continuous span; leaving resets the timer.
* ``SurviveSeconds(seconds)``: stay alive ``seconds`` with the built-in Trork_Brawler fighting back.
* ``DefeatTarget(role)``: bring the target role's health to 0 (the matchup's target role only).
* ``EvadeTurret(seconds, turret_range)``: stay alive ``seconds`` inside a Skeleton_Archer_Turret's range.

Ticks: the gen-7 engine steps at 30 Hz (``TICK_RATE``); the live bridge's native tick rate defaults to 30. Time
limits and objective spans are in seconds and converted here. The episode clock counts ticks, so it is not the paused
world clock (the time of day).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np

TICK_RATE = 30  # ticks per second on the gen-7 engine and the live native backend (native_tick_rate default)
TURRET_ROLE = "Skeleton_Archer_Turret"  # the scripted turret of turret_controller.py
BUILT_IN_OPPONENT = "Trork_Brawler"  # the matchup's target role (kweebec_razorleaf_vs_trork_brawler_v9.json)


@dataclass(frozen=True, slots=True)
class Body:
    """A controlled body: an NPC role on the live side, the duel's entity 0 on the sim side."""

    role: str = "Trork_Unarmed"  # HytaleEnv's default npc_role for non-kill_trork tasks
    loadout: str = "iron_sword"  # sim only; the live body's loadout is the role's
    spawn: tuple[float, float, float] = (0.5, 65.0, 0.5)
    yaw_degrees: float = 0.0  # the body faces -z at yaw 0 (duel test convention)


@dataclass(frozen=True, slots=True)
class ReachPlace:
    """Objective: stand within ``radius`` blocks of ``center`` on the x-z plane."""

    center: tuple[float, float, float]
    radius: float = 1.5


@dataclass(frozen=True, slots=True)
class ReachPlaceRegion:
    """Objective on a Region world: stand within ``radius`` of the fixture's spawn moved by ``offset`` (x, z) blocks.

    The Region fixture places the body (``worlds.region.region_scene``), so the point is defined from its spawn.
    """

    offset: tuple[float, float] = (0.0, -6.0)
    radius: float = 1.5


@dataclass(frozen=True, slots=True)
class HoldArea:
    """Objective: stay within ``radius`` of ``center`` for ``seconds`` in a row (leaving resets the timer)."""

    center: tuple[float, float, float]
    radius: float
    seconds: float


@dataclass(frozen=True, slots=True)
class SurviveSeconds:
    """Objective: stay alive ``seconds`` with the built-in Trork_Brawler fighting back."""

    seconds: float


@dataclass(frozen=True, slots=True)
class DefeatTarget:
    """Objective: bring the target role's health to 0. ``role`` must be the matchup's target role."""

    role: str


@dataclass(frozen=True, slots=True)
class EvadeTurret:
    """Objective: stay alive ``seconds`` inside a Skeleton_Archer_Turret's ``turret_range`` (its attack distance)."""

    seconds: float
    turret_range: float = 18.0


Objective = Union[ReachPlace, ReachPlaceRegion, HoldArea, SurviveSeconds, DefeatTarget, EvadeTurret]


@dataclass(frozen=True, slots=True)
class RewardTerms:
    progress_weight: float = 1.0  # ReachPlace: per block of closing distance while not yet arrived
    step_cost: float = 0.0  # ReachPlace: per tick while not yet arrived
    arrival_bonus: float = 10.0  # ReachPlace arrival and HoldArea completion, once
    held_per_second: float = 1.0  # HoldArea: per second inside the area
    alive_per_second: float = 1.0  # SurviveSeconds and EvadeTurret: per second alive
    death_penalty: float = 10.0  # SurviveSeconds and EvadeTurret: once, on the tick of death
    damage_weight: float = 1.0  # DefeatTarget: per health point taken from the target
    kill_bonus: float = 10.0  # DefeatTarget: once, when the target's health reaches 0
    distance_weight: float = 1.0  # EvadeTurret: per block the body gains from the turret


@dataclass(frozen=True, slots=True)
class World:
    kind: str = "open_flat"  # "open_flat" (compiled); "region_library" is named but not compiled yet
    library: str | None = None  # a worlds.libraries id, for region_library


@dataclass(frozen=True, slots=True)
class TaskSpec:
    name: str
    world: World
    bodies: tuple[Body, ...]
    objectives: tuple[Objective, ...]
    time_limit_seconds: float
    reward: RewardTerms = field(default_factory=RewardTerms)
    opponents: tuple[str, ...] = ()  # must equal required_opponents(objective); checked by validate_spec
    park: tuple[float, float, float] = (0.5, 65.0, 30.5)  # where an inert second entity stands on the sim side


def required_opponents(objective: Objective) -> tuple[str, ...]:
    """The opponent roles an objective needs (empty for a solo objective)."""

    if isinstance(objective, SurviveSeconds):
        return (BUILT_IN_OPPONENT,)
    if isinstance(objective, DefeatTarget):
        return (objective.role,)
    if isinstance(objective, EvadeTurret):
        return (TURRET_ROLE,)
    return ()


def validate_spec(spec: TaskSpec) -> None:
    """Raise ``NotImplementedError`` for anything outside the compiled scope, ``ValueError`` for a malformed spec."""

    if len(spec.bodies) != 1:
        raise NotImplementedError("the compilers drive one controlled body")
    if len(spec.objectives) != 1:
        raise NotImplementedError("the compilers take exactly one objective")
    objective = spec.objectives[0]
    if not isinstance(objective, (ReachPlace, ReachPlaceRegion, HoldArea, SurviveSeconds, DefeatTarget, EvadeTurret)):
        raise NotImplementedError(f"objective {type(objective).__name__} is not defined")
    if spec.world.kind not in ("open_flat", "region_library"):
        raise NotImplementedError(f"world kind {spec.world.kind!r} is not defined")
    if (spec.world.kind == "region_library") != isinstance(objective, ReachPlaceRegion):
        raise NotImplementedError("ReachPlaceRegion runs on the region_library world; the other objectives on open_flat")
    if spec.world.kind == "region_library" and not spec.world.library:
        raise ValueError("a region_library world names its library")
    if isinstance(objective, ReachPlaceRegion) and not objective.radius > 0:
        raise ValueError("ReachPlaceRegion.radius must be positive")
    if tuple(spec.opponents) != required_opponents(objective):
        raise NotImplementedError(
            f"{type(objective).__name__} compiles opponents {required_opponents(objective)!r}, "
            f"got {tuple(spec.opponents)!r}"
        )
    if not spec.time_limit_seconds > 0:
        raise ValueError("time_limit_seconds must be positive")
    if isinstance(objective, ReachPlace) and not objective.radius > 0:
        raise ValueError("ReachPlace.radius must be positive")
    if isinstance(objective, HoldArea) and not (objective.radius > 0 and objective.seconds > 0):
        raise ValueError("HoldArea radius and seconds must be positive")
    if isinstance(objective, (HoldArea, SurviveSeconds, EvadeTurret)):
        if not 0 < objective.seconds <= spec.time_limit_seconds:
            raise ValueError("the objective's seconds must be positive and within time_limit_seconds")
    if isinstance(objective, EvadeTurret) and not objective.turret_range > 0:
        raise ValueError("EvadeTurret.turret_range must be positive")
    if isinstance(objective, DefeatTarget) and not objective.role.strip():
        raise ValueError("DefeatTarget.role must be a non-empty role id")


def planar_distance(position, center) -> np.ndarray:
    """Distance on the x-z plane from ``position`` (..., 3) to ``center`` (3,)."""

    position = np.asarray(position, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    return np.hypot(position[..., 0] - center[0], position[..., 2] - center[2])


def time_limit_ticks(spec: TaskSpec) -> int:
    return int(round(spec.time_limit_seconds * TICK_RATE))


def objective_ticks(seconds: float) -> int:
    """A span in seconds as ticks at the 30 Hz episode clock."""

    return int(round(seconds * TICK_RATE))


# --- ReachPlace (pilot) -------------------------------------------------------------------------------------------


def reach_step(prev_distance, distance, arrived_before, spec: TaskSpec):
    """One tick of the reach reward. Returns ``(reward, arrived_now)``; arrays broadcast, scalars work too.

    While not yet arrived: ``progress_weight * (prev - distance) - step_cost``. The first tick inside the radius adds
    ``arrival_bonus`` once; after arrival the reward is zero.
    """

    terms = spec.reward
    radius = spec.objectives[0].radius
    prev = np.asarray(prev_distance, dtype=np.float64)
    now = np.asarray(distance, dtype=np.float64)
    before = np.asarray(arrived_before, dtype=bool)
    arrived_now = before | (now <= radius)
    active = ~before
    reward = np.where(active, terms.progress_weight * (prev - now) - terms.step_cost, 0.0)
    reward = reward + np.where(arrived_now & active, terms.arrival_bonus, 0.0)
    return reward, arrived_now


def reach_done(arrived, tick: int, limit_ticks: int):
    """Termination: arrival, or the time limit (``tick`` and ``limit_ticks`` count the same unit)."""

    return np.asarray(arrived, dtype=bool) | (tick >= limit_ticks)


# --- HoldArea -----------------------------------------------------------------------------------------------------


def hold_step(streak_before, inside_now, goal_ticks: int, spec: TaskSpec):
    """One tick of the hold reward. Returns ``(reward, streak_now)``: the continuous ticks inside (0 after leaving).

    Each tick inside pays ``held_per_second / TICK_RATE``; the tick that completes the span pays ``arrival_bonus``
    once. Outside the area nothing is paid and the timer restarts.
    """

    terms = spec.reward
    inside = np.asarray(inside_now, dtype=bool)
    streak = np.where(inside, np.asarray(streak_before, dtype=np.int64) + 1, 0)
    reward = np.where(inside, terms.held_per_second / TICK_RATE, 0.0)
    reward = reward + np.where(streak == goal_ticks, terms.arrival_bonus, 0.0)
    return reward, streak


def hold_done(streak, tick: int, goal_ticks: int, limit_ticks: int):
    """Termination: the continuous span is complete (success), or the time limit."""

    return (np.asarray(streak) >= goal_ticks) | (tick >= limit_ticks)


# --- SurviveSeconds and EvadeTurret -------------------------------------------------------------------------------


def survive_step(alive_before, alive_now, spec: TaskSpec):
    """One tick of survival: ``alive_per_second / TICK_RATE`` while alive, ``-death_penalty`` on the tick of death."""

    terms = spec.reward
    before = np.asarray(alive_before, dtype=bool)
    now = np.asarray(alive_now, dtype=bool)
    reward = np.where(before & now, terms.alive_per_second / TICK_RATE, 0.0)
    return reward - np.where(before & ~now, terms.death_penalty, 0.0)


def survive_done(alive, tick: int, goal_ticks: int, limit_ticks: int):
    """Termination: death, the span complete, or the time limit. ``survive_success`` says which."""

    return (~np.asarray(alive, dtype=bool)) | (tick >= goal_ticks) | (tick >= limit_ticks)


def survive_success(alive, tick: int, goal_ticks: int):
    """Success: still alive once the span is complete."""

    return np.asarray(alive, dtype=bool) & (tick >= goal_ticks)


def evade_step(alive_before, alive_now, distance_before, distance_now, spec: TaskSpec):
    """Survival (as ``survive_step``) plus distance: ``distance_weight`` per block the body gains from the turret."""

    gained = np.asarray(distance_now, dtype=np.float64) - np.asarray(distance_before, dtype=np.float64)
    both = np.asarray(alive_before, dtype=bool) & np.asarray(alive_now, dtype=bool)
    return survive_step(alive_before, alive_now, spec) + np.where(both, spec.reward.distance_weight * gained, 0.0)


def evade_done(alive, in_range, tick: int, goal_ticks: int, limit_ticks: int):
    """Termination: death, leaving the turret's range, the span complete, or the time limit."""

    return survive_done(alive, tick, goal_ticks, limit_ticks) | (~np.asarray(in_range, dtype=bool))


def evade_success(alive, in_range, tick: int, goal_ticks: int):
    """Success: alive and inside the turret's range once the span is complete."""

    return survive_success(alive, tick, goal_ticks) & np.asarray(in_range, dtype=bool)


# --- DefeatTarget -------------------------------------------------------------------------------------------------


def defeat_step(health_before, health_now, spec: TaskSpec):
    """Damage dealt to the target this tick (``damage_weight`` per point) and ``kill_bonus`` on the killing tick."""

    before = np.asarray(health_before, dtype=np.float64)
    now = np.asarray(health_now, dtype=np.float64)
    damage = np.maximum(before - now, 0.0)
    killed = (before > 0.0) & (now <= 0.0)
    return spec.reward.damage_weight * damage + np.where(killed, spec.reward.kill_bonus, 0.0)


def defeat_done(target_health, tick: int, limit_ticks: int):
    """Termination: the target's health is 0 (success), or the time limit."""

    return (np.asarray(target_health, dtype=np.float64) <= 0.0) | (tick >= limit_ticks)


__all__ = [
    "TICK_RATE",
    "BUILT_IN_OPPONENT",
    "TURRET_ROLE",
    "Body",
    "DefeatTarget",
    "EvadeTurret",
    "HoldArea",
    "Objective",
    "ReachPlace",
    "ReachPlaceRegion",
    "ReachPlaceRegion",
    "RewardTerms",
    "SurviveSeconds",
    "TaskSpec",
    "World",
    "defeat_done",
    "defeat_step",
    "evade_done",
    "evade_step",
    "evade_success",
    "hold_done",
    "hold_step",
    "objective_ticks",
    "planar_distance",
    "reach_done",
    "reach_step",
    "required_opponents",
    "survive_done",
    "survive_step",
    "survive_success",
    "time_limit_ticks",
    "validate_spec",
]
