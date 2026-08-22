"""Reusable high-level melee skills with identical sim/native preprocessing."""

from __future__ import annotations

from enum import IntEnum

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CombatSkill(IntEnum):
    """Discrete policy commands translated into ordinary bridge actions."""

    IDLE = 0
    FACE_TARGET = 1
    APPROACH = 2
    RETREAT = 3
    STRAFE_LEFT = 4
    STRAFE_RIGHT = 5
    ATTACK = 6
    APPROACH_ATTACK = 7
    RETREAT_ATTACK = 8


class CombatSkillWrapper(gym.Wrapper):
    """Expose nine backend-neutral melee skills as a discrete action space.

    This wrapper performs only deterministic policy-side preprocessing. It
    does not predict physics or damage, so the exact same wrapper and learned
    weights can be used against the high-throughput simulator and native
    headless Hytale.
    """

    def __init__(self, env: gym.Env, maximum_turn_degrees: float = 45.0) -> None:
        super().__init__(env)
        if maximum_turn_degrees <= 0.0:
            raise ValueError("maximum_turn_degrees must be positive")
        self.maximum_turn_degrees = float(maximum_turn_degrees)
        self.action_space = spaces.Discrete(len(CombatSkill))
        self._observation: dict | None = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._observation = observation
        return observation, info

    def step(self, action):
        skill = CombatSkill(int(action))
        bridge_action = self._skill_action(skill)
        observation, reward, terminated, truncated, info = self.env.step(
            bridge_action
        )
        self._observation = observation
        return observation, reward, terminated, truncated, info

    def _skill_action(self, skill: CombatSkill) -> dict[str, int | float]:
        action: dict[str, int | float] = {
            "forward": 0,
            "back": 0,
            "left": 0,
            "right": 0,
            "jump": 0,
            "attack": 0,
            "camera_delta_yaw": self._target_turn_delta(),
        }
        if skill == CombatSkill.IDLE:
            action["camera_delta_yaw"] = 0.0
        elif skill == CombatSkill.APPROACH:
            action["forward"] = 1
        elif skill == CombatSkill.RETREAT:
            action["back"] = 1
        elif skill == CombatSkill.STRAFE_LEFT:
            action["left"] = 1
        elif skill == CombatSkill.STRAFE_RIGHT:
            action["right"] = 1
        elif skill == CombatSkill.ATTACK:
            action["attack"] = 1
        elif skill == CombatSkill.APPROACH_ATTACK:
            action["forward"] = 1
            action["attack"] = 1
        elif skill == CombatSkill.RETREAT_ATTACK:
            action["back"] = 1
            action["attack"] = 1
        return action

    def _target_turn_delta(self) -> float:
        observation = self._observation
        if not isinstance(observation, dict):
            return 0.0
        entities = np.asarray(
            observation.get("nearby_entities", ()),
            dtype=np.float32,
        )
        if entities.ndim != 2 or entities.shape[1] < 4:
            return 0.0
        visible = np.flatnonzero(
            (entities[:, 0] == 0.0) & (entities[:, 3] > 0.0)
        )
        if not visible.size:
            return 0.0
        target = entities[int(visible[0])]
        dx, dz = float(target[1]), float(target[2])
        bearing = np.degrees(np.arctan2(-dx, -dz))
        yaw = float(observation.get("yaw", 0.0))
        error = ((bearing - yaw + 180.0) % 360.0) - 180.0
        return float(np.clip(
            error,
            -self.maximum_turn_degrees,
            self.maximum_turn_degrees,
        ))
