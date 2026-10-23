"""TaskSpec: one task definition, compiled to the gen-7 sim and to the live bridge (pilot, 2026-10-23).

The spec is data. ``task_sim`` compiles it to reset arguments and a per-tick reward on the gen-7 render;
``task_live`` compiles it to bridge reset options and the same reward, computed from the live observation.
The reward and termination functions here are the single definition both sides call (numpy only, no
hytalegym import), so the two compilers cannot drift apart on scoring.

Pilot scope: one objective, ``ReachPlace`` (stand within ``radius`` of a point, measured on the x-z plane),
on an open flat world with one controlled body and no opponent. Every other feature is listed in
``DESIGN.md`` and rejected by both compilers with a named error, so nothing is silently dropped.

Ticks: the gen-7 engine steps at 30 Hz (``TICK_RATE``); the live bridge's native tick rate defaults to 30.
Time limits are in seconds and converted here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TICK_RATE = 30  # ticks per second on the gen-7 engine and the live native backend (native_tick_rate default)


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
class RewardTerms:
    progress_weight: float = 1.0  # per block of closing distance while not yet arrived
    step_cost: float = 0.0  # per tick while not yet arrived
    arrival_bonus: float = 10.0  # once, on the first tick inside the radius


@dataclass(frozen=True, slots=True)
class World:
    kind: str = "open_flat"  # "open_flat" (pilot); "region_library" is named but not compiled yet
    library: str | None = None  # a worlds.libraries id, for region_library


@dataclass(frozen=True, slots=True)
class TaskSpec:
    name: str
    world: World
    bodies: tuple[Body, ...]
    objectives: tuple[ReachPlace, ...]
    time_limit_seconds: float
    reward: RewardTerms = field(default_factory=RewardTerms)
    opponents: tuple[str, ...] = ()  # pilot: () only; the sim parks the second entity far away
    park: tuple[float, float, float] = (0.5, 65.0, 30.5)  # where the inert second entity stands on the sim side


def validate_pilot(spec: TaskSpec) -> None:
    """Raise ``ValueError`` for anything outside the pilot's compiled scope, or a malformed spec."""

    if spec.world.kind != "open_flat":
        raise NotImplementedError(f"world kind {spec.world.kind!r} is not compiled yet (DESIGN.md gap)")
    if len(spec.bodies) != 1:
        raise NotImplementedError("the pilot compiles one controlled body")
    if spec.opponents:
        raise NotImplementedError("the pilot compiles no opponent (DESIGN.md gap)")
    if len(spec.objectives) != 1 or not isinstance(spec.objectives[0], ReachPlace):
        raise NotImplementedError("the pilot compiles exactly one ReachPlace objective")
    if not spec.time_limit_seconds > 0:
        raise ValueError("time_limit_seconds must be positive")
    if not spec.objectives[0].radius > 0:
        raise ValueError("ReachPlace.radius must be positive")


def planar_distance(position, center) -> np.ndarray:
    """Distance on the x-z plane from ``position`` (..., 3) to ``center`` (3,)."""

    position = np.asarray(position, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    return np.hypot(position[..., 0] - center[0], position[..., 2] - center[2])


def reach_step(prev_distance, distance, arrived_before, spec: TaskSpec):
    """One tick of the reach reward. Returns ``(reward, arrived_now)``; arrays broadcast, scalars work too.

    Shared by both sides. While not yet arrived: ``progress_weight * (prev - distance) - step_cost``. The
    first tick inside the radius adds ``arrival_bonus`` once; after arrival the reward is zero.
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


def time_limit_ticks(spec: TaskSpec) -> int:
    return int(round(spec.time_limit_seconds * TICK_RATE))


__all__ = [
    "TICK_RATE",
    "Body",
    "ReachPlace",
    "RewardTerms",
    "TaskSpec",
    "World",
    "planar_distance",
    "reach_done",
    "reach_step",
    "time_limit_ticks",
    "validate_pilot",
]
