"""Wrapper that flattens the Dict observation space into a single Box."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hytalegym.envs.hytale_env import INVENTORY_SIZE, NEARBY_BLOCK_MAX, ENTITY_MAX, NEARBY_ENTITY_FEATURES


class SimpleObsWrapper(gym.ObservationWrapper):
    """
    Flattens the Dict observation into a 1D float32 vector.

    Layout:
        [x, y, z, vx, vy, vz, yaw, pitch, health, food_buff_timer, stamina, mana,
         time_of_day_norm, craftable_count,
         inv[0..35],
         nearby_blocks[0..728],
         nearby_entities_flat[0..79]]

    Total: 3 + 3 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 36 + 729 + 80 = 859 values
    """

    OBS_SIZE = (
        3 + 3 + 1 + 1 + 1 + 1 + 1 + 1  # pos, vel, yaw, pitch, health, food_buff_timer, stamina, mana
        + 1 + 1                            # time_of_day (normalized), craftable_count
        + INVENTORY_SIZE                   # inventory
        + NEARBY_BLOCK_MAX                 # nearby blocks
        + ENTITY_MAX * NEARBY_ENTITY_FEATURES  # nearby entities
    )

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6, shape=(self.OBS_SIZE,), dtype=np.float32
        )

    def observation(self, obs: dict) -> np.ndarray:
        return np.concatenate([
            obs["position"].astype(np.float32),
            obs["velocity"].astype(np.float32),
            [float(obs["yaw"])],
            [float(obs["pitch"])],
            [float(obs["health"])],
            [float(obs["food_buff_timer"])],
            [float(obs["stamina"])],
            [float(obs["mana"])],
            [float(obs["time_of_day"]) / 24000.0],  # normalize to 0-1
            [float(obs["craftable_count"])],
            obs["inventory"].astype(np.float32),
            obs["nearby_blocks"].astype(np.float32),
            obs["nearby_entities"].flatten().astype(np.float32),
        ]).astype(np.float32)
