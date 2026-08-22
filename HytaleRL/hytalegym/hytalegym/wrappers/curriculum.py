"""Curriculum learning wrapper for multi-phase training."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class CurriculumWrapper(gym.Wrapper):
    """
    Curriculum wrapper that adjusts difficulty based on agent performance.

    Phases:
        0 - Gather: Only resource gathering rewarded, no creatures
        1 - Craft:  Gathering + crafting rewarded, no creatures
        2 - Build:  Gathering + crafting + building rewarded, few creatures
        3 - Survive: Full task with night cycle and hostile creatures

    The phase advances when the agent achieves the success threshold
    for the current phase across a window of episodes.
    """

    def __init__(
        self,
        env: gym.Env,
        success_threshold: float = 0.6,
        window_size: int = 20,
        initial_phase: int = 0,
    ):
        super().__init__(env)
        self.success_threshold = success_threshold
        self.window_size = window_size
        self.phase = initial_phase
        self.max_phase = 3
        self._episode_rewards: list[float] = []
        self._current_episode_reward = 0.0
        self._phase_thresholds = [10.0, 25.0, 50.0, 80.0]

    @property
    def current_phase(self) -> int:
        return self.phase

    def reset(self, **kwargs):
        self._current_episode_reward = 0.0
        # Pass phase info via options
        options = kwargs.get("options", {}) or {}
        options["curriculum_phase"] = self.phase
        kwargs["options"] = options
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Scale reward based on phase
        reward = self._shape_reward(reward, info)
        self._current_episode_reward += reward

        if terminated or truncated:
            self._episode_rewards.append(self._current_episode_reward)
            if len(self._episode_rewards) > self.window_size:
                self._episode_rewards.pop(0)
            self._maybe_advance_phase()

        info["curriculum_phase"] = self.phase
        return obs, reward, terminated, truncated, info

    def _shape_reward(self, reward: float, info: dict) -> float:
        # In early phases, give bonus for phase-relevant achievements
        if self.phase == 0:
            # Focus on gathering - reduce building/combat penalties
            return reward * 0.5 if reward < 0 else reward
        elif self.phase == 1:
            return reward * 0.7 if reward < 0 else reward
        return reward

    def _maybe_advance_phase(self):
        if self.phase >= self.max_phase:
            return
        if len(self._episode_rewards) < self.window_size:
            return

        threshold = self._phase_thresholds[self.phase]
        success_rate = sum(
            1 for r in self._episode_rewards if r >= threshold
        ) / len(self._episode_rewards)

        if success_rate >= self.success_threshold:
            self.phase = min(self.phase + 1, self.max_phase)
            self._episode_rewards.clear()
