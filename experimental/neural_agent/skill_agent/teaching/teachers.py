"""Scripted teachers: privileged-state controllers that label data for the skill policies.

This module holds what every teacher shares: the head layout and the codebook sizes, ``TeacherOutput``, the registry
(``TEACHERS``) and the observed-state rule. Each skill's teacher is in its own folder:
``neural_skills/walk_to/teacher.py`` and ``neural_skills/look_at/teacher.py``.

A teacher reads the simulator's full ``CombatState`` (teachers may read privileged state; SKILL.md, privileged-input
rule) and returns one action row per lane on the gym's 12-head factored surface, ``int32[B, 12]`` (layout in
``hytalegym.jax.combat.observation.v3.policy.layout``). Every function is pure JAX over the batch axis, with no
Python loop over lanes. ``maximum_turn_degrees`` must be a Python float: the look-delta codebook is built from it at
trace time, so it is static under ``jit``.

Conventions, each checked against the engine:

* Degrees throughout. Axes are (x, y up, z). At body yaw ``y`` the body's forward is (-sin y, -cos y) and its right is
  (cos y, -sin y), in (x, z) (``runtime/step.py``, desired velocity).
* Yaw, body-yaw and pitch deltas are quantised against the decoder's own codebook,
  ``look_delta_values(size, maximum_degrees=M)`` (``observation/v3/policy/look_deltas.py``), so a factor decodes back
  to exactly the value that was chosen.
* Locomotion choice 0 is idle. Choice ``(gait - 1) * 8 + d`` walks gait 1..4 (walk, run, sprint, sneak) in body
  direction ``d``: 1 forward, 2 forward-right, 3 right, and so on in 45-degree steps
  (``observation/v3/policy/actions.py``, ``apply_world_move_override``). The engine moves along the COMMANDED body
  yaw, the body yaw after this tick's body-yaw delta, so a walking teacher takes its direction in that frame.

Which state a teacher reads (v2, 2026-10-18). The teacher functions steer from the commanded head and body yaw
(``desired_yaw``, ``desired_pitch``, ``desired_body_yaw``). The engine's head and body follow a command over several
ticks, and a student sees only where they actually point. Fed the raw state (v1), 28 % of the look teacher's yaw
labels could not be predicted from what the student sees, and the student never completed a look (TRAINING_PLAN
T5). The registered teachers (v2) are the same functions applied to ``observed_state(combat)``, whose commanded
fields are the actual ones. In closed loop they succeed as v1 does. ``state_for(name, combat)`` applies the
registered choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax

from hytalegym.jax.combat.observation.v3.policy.layout import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES,
)
from hytalegym.jax.combat.types import AGENT_ENTITY, CombatState

_HEAD = {name: index for index, name in enumerate(ARSENAL_POLICY_ACTION_HEAD_NAMES)}
LOCOMOTION_HEAD = _HEAD["locomotion_gait_compass"]
YAW_HEAD = _HEAD["yaw_delta_bins"]
BODY_YAW_HEAD = _HEAD["body_yaw_delta_bins"]
PITCH_HEAD = _HEAD["pitch_delta_bins"]

YAW_BINS = 9
PITCH_BINS = 5
COMPASS_DIRECTIONS = 8


class TeacherOutput(NamedTuple):
    """One teacher decision per lane: the factor row to execute and whether the goal already holds."""

    factors: jax.Array  # int32[B, 12]
    done: jax.Array  # bool[B]


@dataclass(frozen=True)
class TeacherSpec:
    """Registry record for one scripted teacher: the verbs it teaches, what it reads and its parameters."""

    verbs: tuple[str, ...]
    reads: tuple[str, ...]
    parameters: tuple[str, ...]
    kind: str = "scripted"
    version: str = "v1"
    certificate: object | None = None
    state: str = "commanded"
    """``commanded``: the raw state, steering from the desired head and body yaw; ``observed``: ``observed_state``."""


TEACHERS: dict[str, TeacherSpec] = {
    "walk_to": TeacherSpec(
        verbs=("walk_to",),
        reads=("position", "yaw"),
        parameters=("target_xz", "radius", "gait", "maximum_turn_degrees"),
        version="v2",
        state="observed",
    ),
    "look_at": TeacherSpec(
        verbs=("look_at",),
        reads=("position", "yaw", "agent_head_yaw", "agent_head_pitch"),
        parameters=("target_point", "tolerance_degrees", "maximum_turn_degrees"),
        version="v2",
        state="observed",
    ),
}


def observed_state(combat: CombatState) -> CombatState:
    """The state as a student sees it: the commanded head and body yaw replaced by where they actually point."""

    return combat._replace(desired_yaw=combat.agent_head_yaw, desired_pitch=combat.agent_head_pitch,
                           desired_body_yaw=combat.yaw[:, AGENT_ENTITY])


def state_for(name: str, combat: CombatState) -> CombatState:
    """The state the registered teacher ``name`` reads: ``observed_state(combat)`` or the raw state."""

    return observed_state(combat) if TEACHERS[name].state == "observed" else combat


__all__ = [
    "BODY_YAW_HEAD",
    "COMPASS_DIRECTIONS",
    "LOCOMOTION_HEAD",
    "PITCH_BINS",
    "PITCH_HEAD",
    "TEACHERS",
    "TeacherOutput",
    "TeacherSpec",
    "YAW_BINS",
    "YAW_HEAD",
    "observed_state",
    "state_for",
]
