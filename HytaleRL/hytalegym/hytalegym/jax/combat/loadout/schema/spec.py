"""Canonical host manifest for the fixed melee-loadout contract."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from hytalegym.jax.combat.loadout.schema.contract import (
    LOADOUT_FAILURE_ATTACK_OVERFLOW,
    MAX_MELEE_ATTACKS,
    MELEE_ATTACK_FLOAT_FEATURES,
    MELEE_ATTACK_FLOAT_SIZE,
    MELEE_ATTACK_INTEGER_FEATURES,
    MELEE_ATTACK_INTEGER_SIZE,
    MELEE_COOLDOWN_NORMALIZATION_SECONDS,
    MELEE_LOADOUT_OBSERVATION_SCHEMA,
    MELEE_LOADOUT_OBSERVATION_VERSION,
    MELEE_LOADOUT_SCHEMA,
    MELEE_LOADOUT_VERSION,
    MELEE_RANGE_NORMALIZATION_BLOCKS,
    MELEE_WEAPON_INTEGER_FEATURES,
    MELEE_WEAPON_INTEGER_SIZE,
    WEAPON_KIND_MELEE,
    WEAPON_KIND_NONE,
)
from hytalegym.jax.combat.loadout.schema.types import (
    MeleeLoadoutBatch,
    MeleeLoadoutObservation,
)


def melee_loadout_contract_manifest() -> dict[str, Any]:
    """Return JSON-compatible metadata for loadout producers and adapters."""

    input_fields = {
        "weapon_id": _field(("B",), "int32"),
        "weapon_kind": _field(("B",), "int32"),
        "equipped_mask": _field(("B",), "bool"),
        "attack_mask": _field(("B", MAX_MELEE_ATTACKS), "bool"),
        "hit_delay_ticks": _field(("B", MAX_MELEE_ATTACKS), "int32"),
        "range_blocks": _field(("B", MAX_MELEE_ATTACKS), "float32"),
        "half_angle_degrees": _field(
            ("B", MAX_MELEE_ATTACKS),
            "float32",
        ),
        "damage": _field(("B", MAX_MELEE_ATTACKS), "float32"),
        "cooldown_min_seconds": _field(
            ("B", MAX_MELEE_ATTACKS),
            "float32",
        ),
        "cooldown_max_seconds": _field(
            ("B", MAX_MELEE_ATTACKS),
            "float32",
        ),
        "requires_line_of_sight": _field(
            ("B", MAX_MELEE_ATTACKS),
            "bool",
        ),
        "overflow": _field(("B",), "bool"),
    }
    observation_fields = {
        "schema_version": _field(("B",), "int32"),
        "weapon_i32": _field(
            ("B", MELEE_WEAPON_INTEGER_SIZE),
            "int32",
        ),
        "equipped_mask": _field(("B",), "bool"),
        "attack_f32": _field(
            ("B", MAX_MELEE_ATTACKS, MELEE_ATTACK_FLOAT_SIZE),
            "float32",
        ),
        "attack_i32": _field(
            ("B", MAX_MELEE_ATTACKS, MELEE_ATTACK_INTEGER_SIZE),
            "int32",
        ),
        "attack_mask": _field(("B", MAX_MELEE_ATTACKS), "bool"),
        "valid": _field(("B",), "bool"),
        "failure_bits": _field(("B",), "uint32"),
    }
    return {
        "schema": MELEE_LOADOUT_SCHEMA,
        "version": MELEE_LOADOUT_VERSION,
        "observation_schema": MELEE_LOADOUT_OBSERVATION_SCHEMA,
        "observation_version": MELEE_LOADOUT_OBSERVATION_VERSION,
        "batch_axis": "B",
        "input_fields": input_fields,
        "input_field_order": list(MeleeLoadoutBatch._fields),
        "observation_fields": observation_fields,
        "observation_field_order": list(MeleeLoadoutObservation._fields),
        "feature_order": {
            "weapon_i32": list(MELEE_WEAPON_INTEGER_FEATURES),
            "attack_f32": list(MELEE_ATTACK_FLOAT_FEATURES),
            "attack_i32": list(MELEE_ATTACK_INTEGER_FEATURES),
        },
        "capacity": {"attacks_per_weapon": MAX_MELEE_ATTACKS},
        "normalization": {
            "range_blocks": MELEE_RANGE_NORMALIZATION_BLOCKS,
            "half_angle_degrees": 180.0,
            "damage": "target_max_health",
            "cooldown_seconds": MELEE_COOLDOWN_NORMALIZATION_SECONDS,
        },
        "weapon_kinds": {
            "none": WEAPON_KIND_NONE,
            "melee": WEAPON_KIND_MELEE,
        },
        "failure_bits": {
            "attack_overflow": LOADOUT_FAILURE_ATTACK_OVERFLOW,
        },
        "mask_contract": {
            "active_attacks": "compact_prefix",
            "overflow": "invalid_and_fully_masked",
            "masked_values": "ignored_and_zeroed_in_observation",
        },
        "lifecycle": {
            "scope": "one_batch_row_for_one_episode",
            "profile_mutation": "truncate_and_freeze",
        },
    }


def melee_loadout_contract_json() -> str:
    """Return the canonical serialization used for SHA-256 identity."""

    return json.dumps(
        melee_loadout_contract_manifest(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def melee_loadout_contract_sha256() -> str:
    """Return the semantic loadout-contract identity."""

    encoded = melee_loadout_contract_json().encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _field(shape: tuple[str | int, ...], dtype: str) -> dict[str, Any]:
    return {"shape": list(shape), "dtype": dtype}
