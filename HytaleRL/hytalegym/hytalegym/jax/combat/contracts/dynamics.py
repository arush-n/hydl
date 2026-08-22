"""Canonical identity for episode and rejected-navigation semantics."""

from __future__ import annotations

import hashlib
import json
from typing import Any


COMBAT_DYNAMICS_SCHEMA = "hytalerl_combat_dynamics_v1"
COMBAT_DYNAMICS_VERSION = 1


def combat_dynamics_contract_manifest() -> dict[str, Any]:
    """Describe dynamics that must match when a combat policy is transferred."""

    return {
        "schema": COMBAT_DYNAMICS_SCHEMA,
        "version": COMBAT_DYNAMICS_VERSION,
        "episode_boundary": {
            "valid": (
                "(arsenal_failure_bits | mechanics_failure_bits | "
                "inventory_failure_bits) == 0"
            ),
            "terminated": {
                "meaning": "natural_combat_end",
                "predicate": "valid & ((agent_health<=0) | no_living_opponent)",
            },
            "truncated": {
                "meaning": "uncertified_transition",
                "predicate": "(~valid | geometry_exhausted) & ~terminated",
            },
            "done": "terminated | truncated",
        },
        "target_navigation_rejection": {
            "commit_scope": "uncertified_horizontal_motion_only",
            "horizontal_position": "freeze_previous",
            "horizontal_velocity": "zero",
            "independent_vertical_rotation_and_attack_lifecycle": "continue",
        },
        "target_navigation_unsupported": {
            "update": "sticky_or",
            "episode_boundary": "nonterminal_telemetry",
            "opponent_attack_lifecycle_gate": False,
        },
    }


def combat_dynamics_contract_sha256() -> str:
    """Return the canonical hash of combat dynamics visible to learners."""

    payload = json.dumps(
        combat_dynamics_contract_manifest(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


__all__ = [
    "COMBAT_DYNAMICS_SCHEMA",
    "COMBAT_DYNAMICS_VERSION",
    "combat_dynamics_contract_manifest",
    "combat_dynamics_contract_sha256",
]
