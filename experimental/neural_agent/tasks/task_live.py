"""Live compiler for TaskSpec: bridge reset options and the live reward, as a dry-run (2026-10-23).

Nothing here opens a socket or launches a server. ``compile_live`` returns the reset ``options`` dict that
``HytaleEnv.reset`` sends (``hytale_env.py`` reset_options), validated against the keys the bridge reads
(``ClientHandler.handleReset``, ``EnvironmentOptions``); the test checks each emitted key against the Java source.

Reward and termination on the live side call the same ``task_spec.reach_step`` and ``reach_done`` as the sim,
fed the live observation's absolute ``position`` (``hytale_env.py`` ``_parse_observation``).

Pilot mapping (what the bridge accepts today, read from the Java):
  backend=native, world=flat, npc_role, spawn_x/y/z, fidelity_fixture=default.
  A spawn with world=flat and the default fixture passes ``NativeEnvironmentSession.createFidelityFixture``.
  Not sent: a body yaw (no reset key exists, so ``ReachPlace``'s start yaw is not honoured live) and any opponent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from experimental.neural_agent.tasks.task_spec import (
    TaskSpec,
    planar_distance,
    reach_step,
    time_limit_ticks,
    validate_pilot,
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
)


@dataclass(frozen=True, slots=True)
class LiveReach:
    """The dry-run result: the reset ``options`` and the env kwargs a launch would use. Nothing is launched."""

    options: dict[str, Any]
    env_kwargs: dict[str, Any]
    max_episode_steps: int
    notes: tuple[str, ...] = field(default_factory=tuple)


def compile_live(spec: TaskSpec, *, ticks_per_step: int = 4) -> LiveReach:
    """Build the bridge reset options for the pilot task. Raises ``NotImplementedError`` outside its scope."""

    validate_pilot(spec)
    if isinstance(ticks_per_step, bool) or not isinstance(ticks_per_step, int) or ticks_per_step < 1:
        raise ValueError("ticks_per_step must be a positive integer")
    body = spec.bodies[0]
    x, y, z = (float(v) for v in body.spawn)
    options = {
        "backend": "native",
        "world": "flat",
        "npc_role": body.role,
        "spawn_x": x,
        "spawn_y": y,
        "spawn_z": z,
        "fidelity_fixture": "default",
    }
    env_kwargs = {
        "backend": "native",
        "world": "flat",
        "npc_role": body.role,
        "spawn": (x, y, z),
        "fidelity_fixture": "default",
        "ticks_per_step": ticks_per_step,
    }
    limit_ticks = time_limit_ticks(spec)
    steps = math.ceil(limit_ticks / ticks_per_step)
    notes = (
        "body yaw is not a reset key: the live body starts at the role's heading",
        "no opponent: npc_role only; combat_target_role is not sent",
        f"time limit {spec.time_limit_seconds} s = {limit_ticks} ticks = {steps} steps at ticks_per_step={ticks_per_step}",
    )
    return LiveReach(options=options, env_kwargs=env_kwargs, max_episode_steps=steps, notes=notes)


def live_reach_step(prev_distance, observation: Mapping[str, Any], arrived_before, spec: TaskSpec):
    """Live reward for one env step, from the live observation's absolute ``position``.

    Returns ``(reward, arrived_now, distance)``. The reward is ``task_spec.reach_step``, the sim's function.
    """

    position = np.asarray(observation["position"], dtype=np.float64)
    distance = float(planar_distance(position, spec.objectives[0].center))
    reward, arrived_now = reach_step(prev_distance, distance, arrived_before, spec)
    return float(np.asarray(reward).item()), bool(np.asarray(arrived_now).item()), distance


__all__ = ["LIVE_RESET_KEYS", "LiveReach", "compile_live", "live_reach_step"]
