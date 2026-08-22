"""Wrapper that flattens the Dict action space into a single MultiDiscrete space."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FlatActionWrapper(gym.ActionWrapper):
    """
    Converts the HytaleEnv Dict action space into a flat MultiDiscrete space
    suitable for algorithms that don't support Dict actions (e.g., DQN).

    Flat action layout:
        [forward, back, left, right, jump, attack, use,
         camera_yaw_bin, camera_pitch_bin, hotbar,
         place_x, place_y, place_z, place_type,
         break_x, break_y, break_z, break_flag,
         craft_recipe]
    Camera is discretized into bins.
    """

    def __init__(self, env: gym.Env, camera_bins: int = 11):
        super().__init__(env)
        self.camera_bins = camera_bins
        self.action_space = spaces.MultiDiscrete([
            2, 2, 2, 2, 2, 2, 2,       # movement (7)
            camera_bins, camera_bins,    # camera (2)
            9,                           # hotbar (1)
            7, 7, 7, 29,                # placement (4) — 29 block types
            7, 7, 7, 2,                 # breaking (4)
            10,                          # crafting (1) — 9 recipes + no-craft
        ])
        # Precompute camera bin edges
        self._yaw_values = np.linspace(-180, 180, camera_bins)
        self._pitch_values = np.linspace(-90, 90, camera_bins)

    def action(self, action: np.ndarray) -> dict:
        return {
            "forward": int(action[0]),
            "back": int(action[1]),
            "left": int(action[2]),
            "right": int(action[3]),
            "jump": int(action[4]),
            "attack": int(action[5]),
            "use": int(action[6]),
            "camera_delta_yaw": float(self._yaw_values[action[7]]),
            "camera_delta_pitch": float(self._pitch_values[action[8]]),
            "hotbar_slot": int(action[9]),
            "place_block_x": int(action[10]),
            "place_block_y": int(action[11]),
            "place_block_z": int(action[12]),
            "place_block_type": int(action[13]),
            "break_block_x": int(action[14]),
            "break_block_y": int(action[15]),
            "break_block_z": int(action[16]),
            "break_block": int(action[17]),
            "craft_recipe_id": int(action[18]),
        }
