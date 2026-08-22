"""Canonical Ranged Controllers v1 contract manifest."""

from __future__ import annotations

import hashlib
import json

from hytalegym.jax.combat.controllers.schema.contract import *
from hytalegym.jax.combat.entities import ENTITY_CAPACITY


def ranged_controller_contract_manifest():
    return {
        "schema": CONTROLLER_SCHEMA,
        "version": CONTROLLER_VERSION,
        "entities": ENTITY_CAPACITY,
        "families": ["iron_shortbow", "iron_crossbow"],
        "phases": PHASE_COUNT,
        "actions": ACTION_COUNT,
        "maximum_microtick_seconds": MAX_CONTROLLER_DT_SECONDS,
        "shortbow": {
            "nock_seconds": SHORTBOW_NOCK_SECONDS,
            "charge_thresholds": list(SHORTBOW_CHARGE_THRESHOLDS),
            "volley_thresholds": list(SHORTBOW_VOLLEY_THRESHOLDS),
            "charge_speed_multiplier": (
                SHORTBOW_CHARGE_SPEED_MULTIPLIER
            ),
            "max_ammo": 1,
            "max_durability": 120.0,
            "durability_loss_per_projectile": 0.58,
            "signature_energy_cost": 6.0,
        },
        "crossbow": {
            "primary_cooldown_seconds": (
                CROSSBOW_PRIMARY_COOLDOWN_SECONDS
            ),
            "reload_entry_seconds": CROSSBOW_RELOAD_ENTRY_SECONDS,
            "reload_arrow_seconds": CROSSBOW_RELOAD_ARROW_SECONDS,
            "reload_speed_multiplier": (
                CROSSBOW_RELOAD_SPEED_MULTIPLIER
            ),
            "max_ammo": 6,
            "max_durability": 120.0,
            "durability_loss_per_projectile": 0.28,
            "signature_energy_cost": 5.0,
        },
        "broken_projectile_damage_multiplier": (
            BROKEN_WEAPON_DAMAGE_MULTIPLIER
        ),
        "failure_bits": {
            "invalid_state": CONTROLLER_FAILURE_INVALID_STATE,
            "invalid_command": CONTROLLER_FAILURE_INVALID_COMMAND,
            "invalid_dt": CONTROLLER_FAILURE_INVALID_DT,
            "uncertified_interrupt": (
                CONTROLLER_FAILURE_UNCERTIFIED_INTERRUPT
            ),
        },
        "scope": {
            "included": [
                "shortbow_inventory_nock_charge_release",
                "shortbow_signature_volley_charge_release",
                "crossbow_hold_to_fire",
                "crossbow_reload",
                "swap_ammo_return",
                "launch_time_durability",
            ],
            "deferred": [
                "arsenal_runtime_composition",
                "crossbow_target_combo_effects",
                "full_inventory_slots_and_metadata",
                "melee_durability",
                "native_differential_fixtures",
            ],
        },
        "evidence": {
            "level": "jar_asset_resolved_not_native_differential",
            "hytale_version": "0.5.7",
            "server_jar_sha256": (
                "43D9BCFF1DD31574577DBFC82147718DBE2AC16C19000071F965B521BA808CDC"
            ),
            "assets_sha256": (
                "1B8802C284C228AE4549DAC037716C175BC6B94B0FC9AAC2B7039AC6BD2FFD5D"
            ),
        },
    }


def ranged_controller_contract_json():
    return json.dumps(
        ranged_controller_contract_manifest(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def ranged_controller_contract_sha256():
    return hashlib.sha256(
        ranged_controller_contract_json().encode("utf-8")
    ).hexdigest().upper()
