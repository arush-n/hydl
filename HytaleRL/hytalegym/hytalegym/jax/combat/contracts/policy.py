"""Semantic identity of the flat observation and skill policy interface."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from hytalegym.jax.combat.observation.v1.schema.contract import COMBAT_FLOAT_FEATURES
from hytalegym.jax.combat.skills import (
    SKILL_COUNT,
    TARGET_FORWARD_OBSERVATION,
    TARGET_RIGHT_OBSERVATION,
    TARGET_VISIBLE_OBSERVATION,
)
from hytalegym.jax.combat.types import ACTION_SIZE, ACTIVE_OBSERVATION_SIZE


ACTIVE_COMBAT_OBSERVATION_CONTRACT_VERSION = 1
COMBAT_SKILL_ACTION_CONTRACT_VERSION = 1

_LOW_LEVEL_ACTION_ORDER = (
    "forward",
    "back",
    "left",
    "right",
    "jump",
    "attack",
    "yaw_delta_degrees",
    "pitch_delta_degrees",
    "gait",
    "body_yaw_delta_degrees",
    "hotbar_slot",
)
_SKILL_ACTIONS = (
    ("idle", ()),
    ("face_target", ()),
    ("approach", ("forward",)),
    ("retreat", ("back",)),
    ("strafe_left", ("left",)),
    ("strafe_right", ("right",)),
    ("attack", ("attack",)),
    ("approach_attack", ("forward", "attack")),
    ("retreat_attack", ("back", "attack")),
)


def active_combat_observation_contract_manifest() -> dict[str, Any]:
    """Describe the exact flat actor input consumed by the PPO policy."""

    if len(COMBAT_FLOAT_FEATURES) != ACTIVE_OBSERVATION_SIZE:
        raise RuntimeError("active combat observation feature-order drift")
    return {
        "schema": "hytalerl_active_combat_observation",
        "version": ACTIVE_COMBAT_OBSERVATION_CONTRACT_VERSION,
        "dtype": "float32",
        "shape": ["B", ACTIVE_OBSERVATION_SIZE],
        "feature_order": list(COMBAT_FLOAT_FEATURES),
        "range": [-1.0, 1.0],
        "target_evidence": {
            "predicate_order": ["alive", "line_of_sight", "sensor_range"],
            "hidden_encoding": "all_target_fields_zero",
            "visible_field": "target_visible",
        },
    }


def combat_skill_action_contract_manifest() -> dict[str, Any]:
    """Describe the nine discrete skills and their low-level action mapping."""

    if len(_LOW_LEVEL_ACTION_ORDER) != ACTION_SIZE:
        raise RuntimeError("low-level combat action-order drift")
    if len(_SKILL_ACTIONS) != SKILL_COUNT:
        raise RuntimeError("combat skill-order drift")
    return {
        "schema": "hytalerl_combat_skills",
        "version": COMBAT_SKILL_ACTION_CONTRACT_VERSION,
        "skill_order": [name for name, _ in _SKILL_ACTIONS],
        "low_level_action_order": list(_LOW_LEVEL_ACTION_ORDER),
        "pressed_components": {
            name: list(components) for name, components in _SKILL_ACTIONS
        },
        "facing": {
            "skills": [name for name, _ in _SKILL_ACTIONS if name != "idle"],
            "target_forward_index": TARGET_FORWARD_OBSERVATION,
            "target_right_index": TARGET_RIGHT_OBSERVATION,
            "target_visible_index": TARGET_VISIBLE_OBSERVATION,
            "formula": "degrees(atan2(-right, forward))",
            "clip_degrees": [-45.0, 45.0],
            "hidden_target_yaw_delta": 0.0,
        },
        "legality": {
            "always_legal": [
                "idle",
                "face_target",
                "approach",
                "retreat",
                "strafe_left",
                "strafe_right",
            ],
            "requires_perceptible_ready_target": [
                "attack",
                "approach_attack",
                "retreat_attack",
            ],
        },
    }


def active_combat_observation_contract_sha256() -> str:
    """Return the canonical semantic hash for the policy observation."""

    return _manifest_sha256(active_combat_observation_contract_manifest())


def combat_skill_action_contract_sha256() -> str:
    """Return the canonical semantic hash for the policy action interface."""

    return _manifest_sha256(combat_skill_action_contract_manifest())


def _manifest_sha256(manifest: dict[str, Any]) -> str:
    encoded = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


__all__ = [
    "ACTIVE_COMBAT_OBSERVATION_CONTRACT_VERSION",
    "COMBAT_SKILL_ACTION_CONTRACT_VERSION",
    "active_combat_observation_contract_manifest",
    "active_combat_observation_contract_sha256",
    "combat_skill_action_contract_manifest",
    "combat_skill_action_contract_sha256",
]
