"""Live compiler for TaskSpec: bridge reset options and the live reward, as a dry-run (2026-10-23).

Nothing here opens a socket or launches a server. ``compile_live`` returns the reset ``options`` dict that
``HytaleEnv.reset`` sends (``hytale_env.py`` reset_options), validated against the keys the bridge reads
(``ClientHandler.handleReset``, ``EnvironmentOptions``); the test checks each emitted key against the Java source.

Live mapping per objective (what the bridge accepts today, read from the Java):

* ``ReachPlace``, ``HoldArea``: backend=native, world=flat, npc_role, spawn_x/y/z, fidelity_fixture=default. A spawn
  with world=flat and the default fixture passes ``NativeEnvironmentSession.createFidelityFixture``.
* ``SurviveSeconds``: the same reset plus ``combat_target_active=true``, and env task ``kill_trork``, the only task
  that spawns a combat target (``NativeEnvironmentSession`` line 3284). Which role it spawns is the matchup target,
  Trork_Brawler. Inferred, not launched.
* ``DefeatTarget(role)``: env task ``kill_trork``, fidelity ``native_duel``, ``combat_target_role=role``,
  ``combat_target_active=true``, and NO spawn: a named fixture requires the default spawn (``NativeEnvironmentSession``
  lines 6161-6167), so the body starts where the fixture puts it.
* ``EvadeTurret``: no live fixture (gap L3). ``compile_live`` raises ``NotImplementedError`` and names the option.

Reward on the live side: ``live_reach_step`` and ``live_hold_step`` call the sim's ``task_spec`` functions on the
observation's absolute ``position`` (``hytale_env.py`` ``_parse_observation``). The combat objectives have no live
reward yet, because their observation fields (health, target health) are not mapped here.

Body yaw goes out as ``spawn_yaw`` (degrees) on the default-fixture spawn path when it is non-zero (gap L1, bridge
option; absent keeps the yaw-0 spawn). The named native_duel fixture gets no spawn_yaw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from experimental.neural_agent.tasks.task_spec import (
    DefeatTarget,
    EvadeTurret,
    ReachPlaceRegion,
    HoldArea,
    SurviveSeconds,
    TaskSpec,
    hold_step,
    planar_distance,
    reach_step,
    time_limit_ticks,
    validate_spec,
)

# Every reset key this compiler can emit, each read by ClientHandler.handleReset (or the codec it calls).
LIVE_RESET_KEYS = (
    "backend",
    "world",
    "npc_role",
    "spawn_x",
    "spawn_y",
    "spawn_z",
    "fidelity_fixture",
    "combat_target_role",
    "combat_target_active",
    "spawn_yaw",
)
LIVE_GAP_REGION = (
    "ReachPlaceRegion has no live reset (gap L2). The bridge now accepts a spawn with static_region (NativeEnvironmentSession "
    "createFidelityFixture), but the bridge has no Region library reset that loads a Region library world. Closes with a "
    "bridge reset that loads a Region library world. Sim only until then."
)
LIVE_GAP_EVADE_TURRET = (
    "EvadeTurret has no live fixture (gap L3). Closes with a turret-role combat target: a bridge fixture or option "
    "that spawns Skeleton_Archer_Turret as the non-duel combat target, its validator line in EnvironmentOptions, and "
    "a combat_target_role that accepts a turret. Sim only until then."
)


@dataclass(frozen=True, slots=True)
class LiveTask:
    """The dry-run result: the reset ``options`` and the env kwargs a launch would use. Nothing is launched."""

    options: dict[str, Any]
    env_kwargs: dict[str, Any]
    max_episode_steps: int
    notes: tuple[str, ...] = field(default_factory=tuple)


LiveReach = LiveTask  # the pilot's name


def compile_live(spec: TaskSpec, *, ticks_per_step: int = 4) -> LiveTask:
    """Build the bridge reset options for the spec. Raises ``NotImplementedError`` for a feature the bridge lacks."""

    validate_spec(spec)
    if isinstance(ticks_per_step, bool) or not isinstance(ticks_per_step, int) or ticks_per_step < 1:
        raise ValueError("ticks_per_step must be a positive integer")
    objective = spec.objectives[0]
    if isinstance(objective, EvadeTurret):
        raise NotImplementedError(LIVE_GAP_EVADE_TURRET)
    if isinstance(objective, ReachPlaceRegion):
        raise NotImplementedError(LIVE_GAP_REGION)
    body = spec.bodies[0]
    x, y, z = (float(v) for v in body.spawn)
    options: dict[str, Any] = {"backend": "native", "world": "flat", "npc_role": body.role}
    env_kwargs: dict[str, Any] = {"backend": "native", "world": "flat", "npc_role": body.role}
    notes: list[str] = []
    task = None
    if isinstance(objective, DefeatTarget):
        options.update(combat_target_role=objective.role, combat_target_active=True, fidelity_fixture="native_duel")
        env_kwargs.update(combat_target_role=objective.role, fidelity_fixture="native_duel")
        task = "kill_trork"
        notes.append("native_duel rejects a spawn: the body starts where the fixture places it (no spawn sent)")
    else:
        options.update(spawn_x=x, spawn_y=y, spawn_z=z, fidelity_fixture="default")
        env_kwargs.update(spawn=(x, y, z), fidelity_fixture="default")
        if isinstance(objective, SurviveSeconds):
            options["combat_target_active"] = True
            task = "kill_trork"
            notes.append("built-in Trork_Brawler: kill_trork spawns the matchup target (inferred, not launched)")
        if body.yaw_degrees != 0.0:
            options["spawn_yaw"] = float(body.yaw_degrees)
    if task is not None:
        env_kwargs["task"] = task
    env_kwargs["ticks_per_step"] = ticks_per_step
    limit_ticks = time_limit_ticks(spec)
    steps = math.ceil(limit_ticks / ticks_per_step)
    if "spawn_yaw" in options:
        notes.append(f"body yaw {body.yaw_degrees} deg sent as spawn_yaw (gap L1 bridge option, not yet live-checked)")
    elif isinstance(objective, DefeatTarget) and body.yaw_degrees != 0.0:
        notes.append("body yaw not sent: native_duel gets no spawn_yaw (gap L1 stays open for the duel fixture)")
    else:
        notes.append("body yaw 0 deg: spawn_yaw absent, the body starts at yaw 0 as before (gap L1)")
    notes.append(f"time limit {spec.time_limit_seconds} s = {limit_ticks} ticks = {steps} steps at ticks_per_step={ticks_per_step}")
    return LiveTask(options=options, env_kwargs=env_kwargs, max_episode_steps=steps, notes=tuple(notes))


def live_reach_step(prev_distance, observation: Mapping[str, Any], arrived_before, spec: TaskSpec):
    """Live reward for one env step, from the live observation's absolute ``position``.

    Returns ``(reward, arrived_now, distance)``. The reward is ``task_spec.reach_step``, the sim's function.
    """

    position = np.asarray(observation["position"], dtype=np.float64)
    distance = float(planar_distance(position, spec.objectives[0].center))
    reward, arrived_now = reach_step(prev_distance, distance, arrived_before, spec)
    return float(np.asarray(reward).item()), bool(np.asarray(arrived_now).item()), distance


def live_hold_step(streak_before: int, observation: Mapping[str, Any], spec: TaskSpec):
    """Live hold reward for one env step: ``task_spec.hold_step`` on the observation's position.

    Returns ``(reward, streak_now)``. The goal span is ``HoldArea.seconds`` at the 30 Hz episode clock.
    """

    objective = spec.objectives[0]
    if not isinstance(objective, HoldArea):
        raise ValueError("live_hold_step needs a HoldArea spec")
    position = np.asarray(observation["position"], dtype=np.float64)
    inside = bool(planar_distance(position, objective.center) <= objective.radius)
    goal = int(round(objective.seconds * 30))
    reward, streak = hold_step(np.asarray(streak_before), np.asarray(inside), goal, spec)
    return float(np.asarray(reward).item()), int(np.asarray(streak).item())


__all__ = ["LIVE_GAP_EVADE_TURRET", "LIVE_RESET_KEYS", "LiveReach", "LiveTask", "compile_live", "live_hold_step", "live_reach_step"]
