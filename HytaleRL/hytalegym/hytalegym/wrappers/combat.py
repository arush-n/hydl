"""Compact, backend-identical spaces for efficient melee-combat training."""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hytalegym.combat.telemetry import CombatPhase, CombatStateIndex
from hytalegym.rulesets import load_combat_ruleset


class CombatActionWrapper(gym.ActionWrapper):
    """Replace the 19-branch general action with five combat branches.

    Layout: ``[longitudinal, lateral, jump, attack, yaw_delta]``.

    Longitudinal and lateral each use mutually exclusive idle/positive/negative
    choices, preventing PPO from wasting samples on forward+back or left+right.
    The resulting dictionary is accepted unchanged by both the calibrated
    simulator and the native Hytale NPC controller.
    """

    def __init__(
        self,
        env: gym.Env,
        yaw_deltas: Sequence[float] = (-45.0, -15.0, -5.0, 0.0, 5.0, 15.0, 45.0),
    ) -> None:
        super().__init__(env)
        if not yaw_deltas:
            raise ValueError("yaw_deltas must contain at least one value")
        self.yaw_deltas = np.asarray(yaw_deltas, dtype=np.float32)
        self.action_space = spaces.MultiDiscrete(
            np.asarray([3, 3, 2, 2, len(self.yaw_deltas)], dtype=np.int64)
        )

    def action(self, action: np.ndarray) -> dict[str, int | float]:
        values = np.asarray(action, dtype=np.int64)
        if values.shape != (5,):
            raise ValueError(f"expected combat action shape (5,), got {values.shape}")
        longitudinal, lateral, jump, attack, yaw_index = values
        return {
            "forward": int(longitudinal == 1),
            "back": int(longitudinal == 2),
            "left": int(lateral == 1),
            "right": int(lateral == 2),
            "jump": int(jump),
            "attack": int(attack),
            "camera_delta_yaw": float(self.yaw_deltas[yaw_index]),
        }


class CombatObsWrapper(gym.ObservationWrapper):
    """Project the general observation into an invariant 11-value combat state.

    Values are normalized and contain agent-local velocity, health, heading,
    and the closest visible Trork's agent-local offset/distance/health. The
    projection intentionally drops absolute position, inventory, opaque asset
    IDs, and empty crafting/building features that slow combat learning and
    encourage simulator-specific overfitting.
    """

    FEATURE_NAMES = (
        "forward_velocity",
        "right_velocity",
        "vertical_velocity",
        "agent_health",
        "sin_yaw",
        "cos_yaw",
        "target_forward",
        "target_right",
        "target_distance",
        "target_health",
        "target_visible",
    )
    _RULESET = load_combat_ruleset()
    AGENT_SPEED_SCALE = float(_RULESET["agent"]["max_speed"])
    VERTICAL_SPEED_SCALE = float(
        _RULESET["observation_normalization"]["vertical_speed_scale"]
    )
    AGENT_HEALTH_SCALE = float(_RULESET["agent"]["max_health"])
    SENSOR_RANGE = float(_RULESET["target"]["sensor_range"])
    TARGET_HEALTH_SCALE = float(_RULESET["target"]["max_health"])

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.FEATURE_NAMES),),
            dtype=np.float32,
        )

    def observation(self, observation: dict) -> np.ndarray:
        yaw = np.deg2rad(float(observation["yaw"]))
        forward_x, forward_z = -np.sin(yaw), -np.cos(yaw)
        right_x, right_z = np.cos(yaw), -np.sin(yaw)

        velocity = np.asarray(observation["velocity"], dtype=np.float64)
        forward_velocity = (
            velocity[0] * forward_x + velocity[2] * forward_z
        ) / self.AGENT_SPEED_SCALE
        right_velocity = (
            velocity[0] * right_x + velocity[2] * right_z
        ) / self.AGENT_SPEED_SCALE

        target_forward = 0.0
        target_right = 0.0
        target_distance = 0.0
        target_health = 0.0
        target_visible = 0.0
        entities = np.asarray(observation["nearby_entities"], dtype=np.float32)
        # Native entity family 0 is Trork. Nearby NPCs are already sorted by
        # distance, so this remains locked to the task target if unrelated
        # generated-world NPCs appear closer in the observation list.
        visible = np.flatnonzero((entities[:, 0] == 0.0) & (entities[:, 3] > 0.0))
        if visible.size:
            target = entities[int(visible[0])]
            dx, dz = float(target[1]), float(target[2])
            target_forward = (dx * forward_x + dz * forward_z) / self.SENSOR_RANGE
            target_right = (dx * right_x + dz * right_z) / self.SENSOR_RANGE
            target_distance = np.hypot(dx, dz) / self.SENSOR_RANGE
            target_health = float(target[3]) / self.TARGET_HEALTH_SCALE
            target_visible = 1.0

        result = np.asarray(
            [
                forward_velocity,
                right_velocity,
                velocity[1] / self.VERTICAL_SPEED_SCALE,
                float(observation["health"]) / self.AGENT_HEALTH_SCALE,
                np.sin(yaw),
                np.cos(yaw),
                target_forward,
                target_right,
                target_distance,
                target_health,
                target_visible,
            ],
            dtype=np.float32,
        )
        return np.clip(result, -1.0, 1.0)


class ActiveCombatObsWrapper(CombatObsWrapper):
    """Add authored opponent attack state to the compact combat observation.

    The first 11 values are intentionally identical to
    :class:`CombatObsWrapper`, preserving passive-combat checkpoints. The
    appended values contain the shared native/simulator interaction phase,
    target facing and velocity needed to learn dodges, spacing, and punishes.
    """

    FEATURE_NAMES = CombatObsWrapper.FEATURE_NAMES + (
        "agent_attack_executing",
        "target_phase_idle",
        "target_phase_windup",
        "target_phase_sweep",
        "target_phase_recovery",
        "target_phase_cooldown",
        "target_phase_progress",
        "target_facing_error",
        "target_forward_velocity",
        "target_right_velocity",
        "target_attack_index",
        "target_head_facing_error",
        "target_head_pitch",
    )
    TARGET_SPEED_SCALE = float(CombatObsWrapper._RULESET["target"]["chase_speed"])
    FACING_ERROR_SCALE = float(
        CombatObsWrapper._RULESET["observation_normalization"][
            "facing_error_degrees_scale"
        ]
    )
    TARGET_ATTACK_INDEX_SCALE = float(
        len(CombatObsWrapper._RULESET["target"]["attacks"]) - 1
    )
    HEAD_PITCH_SCALE = float(
        CombatObsWrapper._RULESET["observation_normalization"][
            "head_pitch_degrees_scale"
        ]
    )

    def observation(self, observation: dict) -> np.ndarray:
        base = super().observation(observation)
        raw = np.zeros(len(CombatStateIndex), dtype=np.float32)
        raw[CombatStateIndex.TARGET_ATTACK_INDEX] = -1.0
        provided = np.asarray(
            observation.get("combat_state", raw),
            dtype=np.float32,
        ).reshape(-1)
        raw[: min(raw.size, provided.size)] = provided[: raw.size]

        phase_code = int(np.clip(
            np.rint(raw[CombatStateIndex.TARGET_ATTACK_PHASE]),
            int(CombatPhase.IDLE),
            int(CombatPhase.COOLDOWN),
        ))
        phase = np.zeros(len(CombatPhase), dtype=np.float32)
        phase[phase_code] = 1.0

        yaw = np.deg2rad(float(observation["yaw"]))
        forward_x, forward_z = -np.sin(yaw), -np.cos(yaw)
        right_x, right_z = np.cos(yaw), -np.sin(yaw)
        target_vx = raw[CombatStateIndex.TARGET_VELOCITY_X]
        target_vz = raw[CombatStateIndex.TARGET_VELOCITY_Z]
        target_forward_velocity = (
            target_vx * forward_x + target_vz * forward_z
        ) / self.TARGET_SPEED_SCALE
        target_right_velocity = (
            target_vx * right_x + target_vz * right_z
        ) / self.TARGET_SPEED_SCALE

        attack_index = raw[CombatStateIndex.TARGET_ATTACK_INDEX]
        normalized_attack_index = (
            -1.0
            if attack_index < 0.0
            else attack_index / self.TARGET_ATTACK_INDEX_SCALE
        )
        target_head_facing_error = _normalize_degrees(
            raw[CombatStateIndex.TARGET_FACING_ERROR_DEGREES]
            + raw[CombatStateIndex.TARGET_YAW_DEGREES]
            - raw[CombatStateIndex.TARGET_HEAD_YAW_DEGREES]
        )
        extra = np.asarray(
            [
                raw[CombatStateIndex.AGENT_ATTACK_EXECUTING],
                *phase,
                raw[CombatStateIndex.TARGET_ATTACK_PROGRESS],
                raw[CombatStateIndex.TARGET_FACING_ERROR_DEGREES]
                / self.FACING_ERROR_SCALE,
                target_forward_velocity,
                target_right_velocity,
                normalized_attack_index,
                target_head_facing_error / self.FACING_ERROR_SCALE,
                raw[CombatStateIndex.TARGET_HEAD_PITCH_DEGREES]
                / self.HEAD_PITCH_SCALE,
            ],
            dtype=np.float32,
        )
        return np.clip(np.concatenate((base, extra)), -1.0, 1.0)


def _normalize_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0
