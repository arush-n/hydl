"""Canonical manifest for learner observation v2."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from hytalegym.jax.combat.loadout import (
    MELEE_LOADOUT_OBSERVATION_SCHEMA,
    MELEE_LOADOUT_OBSERVATION_VERSION,
    melee_loadout_contract_sha256,
)
from hytalegym.jax.combat.observation.v1.schema.contract import (
    LEARNER_ACTION_COUNT,
    LEARNER_OBSERVATION_SCHEMA,
    LEARNER_OBSERVATION_VERSION,
)
from hytalegym.jax.combat.observation.v1.schema.spec import (
    learner_observation_contract_sha256,
)
from hytalegym.jax.combat.observation.v2 import (
    LEARNER_OBSERVATION_V2_SCHEMA,
    LEARNER_OBSERVATION_V2_VERSION,
    V2_FAILURE_LOADOUT_OVERFLOW,
    LearnerCombatObservationV2,
)
from hytalegym.jax.combat.skills import (
    SKILL_APPROACH_ATTACK,
    SKILL_ATTACK,
    SKILL_RETREAT_ATTACK,
)


def learner_observation_v2_contract_manifest() -> dict[str, Any]:
    """Return the frozen extension manifest for checkpoint gating."""

    return {
        "schema": LEARNER_OBSERVATION_V2_SCHEMA,
        "version": LEARNER_OBSERVATION_V2_VERSION,
        "batch_axis": "B",
        "fields": {
            "schema_version": _field(("B",), "int32"),
            "base": {
                "schema": LEARNER_OBSERVATION_SCHEMA,
                "version": LEARNER_OBSERVATION_VERSION,
                "sha256": learner_observation_contract_sha256(),
            },
            "loadout": {
                "schema": MELEE_LOADOUT_OBSERVATION_SCHEMA,
                "version": MELEE_LOADOUT_OBSERVATION_VERSION,
                "sha256": melee_loadout_contract_sha256(),
            },
            "action_mask": _field(("B", LEARNER_ACTION_COUNT), "bool"),
            "valid": _field(("B",), "bool"),
            "failure_bits": _field(("B",), "uint32"),
        },
        "field_order": list(LearnerCombatObservationV2._fields),
        "action_contract": {
            "count": LEARNER_ACTION_COUNT,
            "requires_equipped_melee": [
                SKILL_ATTACK,
                SKILL_APPROACH_ATTACK,
                SKILL_RETREAT_ATTACK,
            ],
            "semantics": "inherited_from_base_v1",
        },
        "failure_bits": {
            "base_v1_passthrough": [0, 1, 2, 3, 4, 5],
            "loadout_overflow": V2_FAILURE_LOADOUT_OVERFLOW,
        },
        "lifecycle": {
            "loadout": "episode_pinned",
            "profile_mismatch": "truncate_zero_reward_and_freeze",
        },
    }


def learner_observation_v2_contract_json() -> str:
    """Return the canonical serialization used for SHA-256 identity."""

    return json.dumps(
        learner_observation_v2_contract_manifest(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def learner_observation_v2_contract_sha256() -> str:
    """Return the semantic v2 learner-contract identity."""

    encoded = learner_observation_v2_contract_json().encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _field(shape: tuple[str | int, ...], dtype: str) -> dict[str, Any]:
    return {"shape": list(shape), "dtype": dtype}
