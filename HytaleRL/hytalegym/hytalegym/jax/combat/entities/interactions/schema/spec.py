"""Canonical manifest for ordered entity interactions v1."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from hytalegym.jax.combat.entities import (
    COMBAT_ENTITIES_SCHEMA,
    COMBAT_ENTITIES_VERSION,
    ENTITY_CAPACITY,
)
from hytalegym.jax.combat.entities.interactions.schema.contract import (
    CROSSBOW_BIG_ARROW_DAMAGE,
    CROSSBOW_BIG_ARROW_FORCE,
    CROSSBOW_COMBO_1_EFFECT_ID,
    CROSSBOW_COMBO_2_EFFECT_ID,
    CROSSBOW_COMBO_DAMAGE,
    CROSSBOW_COMBO_DURATION_SECONDS,
    CROSSBOW_COMBO_FORCE,
    CROSSBOW_FORCE_DIRECTION,
    CROSSBOW_IMPACT_CAPACITY,
    CROSSBOW_STANDARD_DAMAGE,
    CROSSBOW_STANDARD_FORCE,
    ENTITY_INTERACTIONS_SCHEMA,
    ENTITY_INTERACTIONS_VERSION,
    INTERACTION_CAPABILITY_CROSSBOW_COMBO,
    INTERACTION_CAPABILITY_CROSSBOW_FORCE,
    INTERACTION_CAPABILITY_GAMEPLAY_EFFECT_CLEAR,
    INTERACTION_CAPABILITY_ORDERED_IMPACTS,
    INTERACTION_FAILURE_IMPACT_OVERFLOW,
    INTERACTION_FAILURE_INVALID_COMMAND,
    INTERACTION_FAILURE_INVALID_STATE,
    INTERACTION_FAILURE_MECHANICS,
    INTERACTION_FAILURE_STALE_HANDLE,
    INTERACTION_FAILURE_UNSUPPORTED_TARGET,
    INTERACTION_FAILURE_UPSTREAM,
    POTION_REGEN_EFFECT_IDS,
)
from hytalegym.jax.combat.entities.schema.spec import (
    HYTALE_0_5_7_SERVER_JAR_SHA256,
)
from hytalegym.jax.combat.mechanics import (
    COMBAT_MECHANICS_SCHEMA,
    COMBAT_MECHANICS_VERSION,
    HYTALE_0_5_7_ASSETS_SHA256,
    HYTALE_0_5_7_NATIVE_EVIDENCE_JAR_SHA256,
)


def entity_interactions_contract_manifest() -> dict[str, Any]:
    return {
        "schema": ENTITY_INTERACTIONS_SCHEMA,
        "version": ENTITY_INTERACTIONS_VERSION,
        "dependencies": {
            "entities": {
                "schema": COMBAT_ENTITIES_SCHEMA,
                "version": COMBAT_ENTITIES_VERSION,
            },
            "mechanics": {
                "schema": COMBAT_MECHANICS_SCHEMA,
                "version": COMBAT_MECHANICS_VERSION,
            },
        },
        "capacities": {
            "entities": ENTITY_CAPACITY,
            "ordered_crossbow_impacts": CROSSBOW_IMPACT_CAPACITY,
        },
        "crossbow_impact_input": {
            "requested": _field(
                ("B", CROSSBOW_IMPACT_CAPACITY), "bool"
            ),
            "kind": _field(
                ("B", CROSSBOW_IMPACT_CAPACITY), "int32"
            ),
            "source_slot": _field(
                ("B", CROSSBOW_IMPACT_CAPACITY), "int32"
            ),
            "source_generation": _field(
                ("B", CROSSBOW_IMPACT_CAPACITY), "uint32"
            ),
            "target_slot": _field(
                ("B", CROSSBOW_IMPACT_CAPACITY), "int32"
            ),
            "target_generation": _field(
                ("B", CROSSBOW_IMPACT_CAPACITY), "uint32"
            ),
            "knockback_yaw_degrees": _field(
                ("B", CROSSBOW_IMPACT_CAPACITY), "float32"
            ),
            "damage_multiplier": _field(
                ("B", CROSSBOW_IMPACT_CAPACITY), "float32"
            ),
            "overflow": _field(("B",), "bool"),
            "ordering": "ascending_command_index",
            "producer": "certified_projectile_hit_adapter",
        },
        "crossbow": {
            "standard_damage": CROSSBOW_STANDARD_DAMAGE,
            "combo_damage": CROSSBOW_COMBO_DAMAGE,
            "big_arrow_damage": CROSSBOW_BIG_ARROW_DAMAGE,
            "damage_classes": {
                "standard": "Light",
                "third_hit_combo": "Charged",
                "big_arrow": "Signature",
            },
            "standard_force": CROSSBOW_STANDARD_FORCE,
            "combo_force": CROSSBOW_COMBO_FORCE,
            "big_arrow_force": CROSSBOW_BIG_ARROW_FORCE,
            "force_direction": list(CROSSBOW_FORCE_DIRECTION),
            "combo_effect_ids": [
                CROSSBOW_COMBO_1_EFFECT_ID,
                CROSSBOW_COMBO_2_EFFECT_ID,
            ],
            "combo_duration_seconds": (
                CROSSBOW_COMBO_DURATION_SECONDS
            ),
            "combo_order": [
                "apply_combo_1_then_standard_damage",
                "apply_combo_2_then_standard_damage",
                "combo_damage_then_clear_combo_1_and_combo_2",
            ],
            "third_hit_signature_energy": 1.0,
            "damage_health_rounding": (
                "nonnegative_java_math_round_after_guard"
            ),
            "standard_parent_success_clear_effect_ids": list(
                POTION_REGEN_EFFECT_IDS
            ),
            "combo_success_next_replaces_parent_next": True,
            "big_arrow_participates_in_combo": False,
        },
        "capability_bits": {
            "ordered_impacts": (
                INTERACTION_CAPABILITY_ORDERED_IMPACTS
            ),
            "crossbow_combo": INTERACTION_CAPABILITY_CROSSBOW_COMBO,
            "gameplay_effect_clear": (
                INTERACTION_CAPABILITY_GAMEPLAY_EFFECT_CLEAR
            ),
            "crossbow_force": INTERACTION_CAPABILITY_CROSSBOW_FORCE,
        },
        "failure_bits": {
            "invalid_state": INTERACTION_FAILURE_INVALID_STATE,
            "invalid_command": INTERACTION_FAILURE_INVALID_COMMAND,
            "stale_handle": INTERACTION_FAILURE_STALE_HANDLE,
            "unsupported_target": (
                INTERACTION_FAILURE_UNSUPPORTED_TARGET
            ),
            "upstream": INTERACTION_FAILURE_UPSTREAM,
            "mechanics": INTERACTION_FAILURE_MECHANICS,
            "impact_overflow": INTERACTION_FAILURE_IMPACT_OVERFLOW,
        },
        "failure_behavior": "row_local_atomic_sticky_freeze",
        "scope": {
            "included": [
                "target_local_crossbow_combo_effects",
                "standard_and_big_arrow_resolved_damage",
                "force_knockback",
                "successful_damage_gameplay_regen_clear",
                "slot_generation_validation",
            ],
            "injected_not_implemented": [
                "projectile_launch",
                "projectile_flight",
                "projectile_entity_hit_detection",
                "projectile_world_collision",
            ],
            "deferred": [
                "live_native_interaction_trace",
                "post_death_same_tick_impact_order",
                "full_arsenal_to_entity_composition",
                "cosmetic_effects_audio_particles_and_red_flash",
            ],
        },
        "evidence": {
            "level": "jar_asset_resolved_not_native_differential",
            "hytale_version": "0.5.7",
            "server_jar_sha256": HYTALE_0_5_7_SERVER_JAR_SHA256,
            "assets_sha256": HYTALE_0_5_7_ASSETS_SHA256,
            "native_evidence_bridge_sha256": (
                HYTALE_0_5_7_NATIVE_EVIDENCE_JAR_SHA256
            ),
        },
    }


def entity_interactions_contract_json() -> str:
    return json.dumps(
        entity_interactions_contract_manifest(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def entity_interactions_contract_sha256() -> str:
    return hashlib.sha256(
        entity_interactions_contract_json().encode("utf-8")
    ).hexdigest().upper()


def _field(shape: tuple[str | int, ...], dtype: str) -> dict[str, Any]:
    return {"shape": list(shape), "dtype": dtype}


__all__ = [
    "entity_interactions_contract_json",
    "entity_interactions_contract_manifest",
    "entity_interactions_contract_sha256",
]
