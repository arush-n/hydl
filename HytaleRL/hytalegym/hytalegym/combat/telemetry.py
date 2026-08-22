"""Stable numeric combat state shared by simulator and native Hytale 0.5.7."""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Mapping

import numpy as np


class CombatPhase(IntEnum):
    """Authored interaction-chain phase codes emitted by both backends."""

    IDLE = 0
    WINDUP = 1
    SWEEP = 2
    RECOVERY = 3
    COOLDOWN = 4


class CombatStateIndex(IntEnum):
    """Indices in the raw ``combat_state`` observation vector."""

    AGENT_ATTACK_EXECUTING = 0
    TARGET_VISIBLE = 1
    TARGET_ATTACK_PHASE = 2
    TARGET_ATTACK_PROGRESS = 3
    TARGET_ATTACK_INDEX = 4
    TARGET_ATTACK_ELAPSED_TICKS = 5
    TARGET_FACING_ERROR_DEGREES = 6
    TARGET_YAW_DEGREES = 7
    TARGET_VELOCITY_X = 8
    TARGET_VELOCITY_Z = 9
    TARGET_HEAD_YAW_DEGREES = 10
    TARGET_HEAD_PITCH_DEGREES = 11


COMBAT_STATE_SIZE = len(CombatStateIndex)
COMBAT_STATE_LOW = np.asarray(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        -180.0,
        -180.0,
        -100.0,
        -100.0,
        -180.0,
        -90.0,
    ],
    dtype=np.float32,
)
COMBAT_STATE_HIGH = np.asarray(
    [
        1.0,
        1.0,
        4.0,
        1.0,
        4.0,
        10_000.0,
        180.0,
        180.0,
        100.0,
        100.0,
        180.0,
        90.0,
    ],
    dtype=np.float32,
)

_INFO_KEYS = (
    "combat_agent_attack_executing",
    "combat_target_visible",
    "combat_target_attack_phase",
    "combat_target_attack_progress",
    "combat_target_attack_index",
    "combat_target_attack_elapsed_ticks",
    "combat_target_facing_error_degrees",
    "combat_target_yaw_degrees",
    "combat_target_velocity_x",
    "combat_target_velocity_z",
    "combat_target_head_yaw_degrees",
    "combat_target_head_pitch_degrees",
)


def parse_combat_state(info: Mapping[str, Any] | None) -> np.ndarray:
    """Convert bridge info into the fixed numeric policy observation.

    Missing telemetry intentionally becomes an idle state, so non-combat tasks
    and older bridge responses remain valid Gymnasium observations.
    """

    source = info if isinstance(info, Mapping) else {}
    defaults = (
        0.0,
        0.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    values = np.asarray(
        [_as_float(source.get(key, default), default) for key, default in zip(
            _INFO_KEYS, defaults, strict=True
        )],
        dtype=np.float32,
    )
    return np.clip(values, COMBAT_STATE_LOW, COMBAT_STATE_HIGH)


def _as_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default
