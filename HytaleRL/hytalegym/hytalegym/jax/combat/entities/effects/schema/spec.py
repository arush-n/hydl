"""Canonical manifest for logical-entity effect ticking v1."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from hytalegym.jax.combat.entities import (
    COMBAT_ENTITIES_SCHEMA,
    COMBAT_ENTITIES_VERSION,
    ENTITY_CAPACITY,
)
from hytalegym.jax.combat.entities.effects.schema.contract import (
    ENTITY_EFFECT_CAPABILITY_NATIVE_DAMAGE_ROUNDING,
    ENTITY_EFFECT_CAPABILITY_PERIODIC_DAMAGE,
    ENTITY_EFFECT_CAPABILITY_PERIODIC_HEALING,
    ENTITY_EFFECT_CAPABILITY_PERIODIC_RESOURCES,
    ENTITY_EFFECT_CAPABILITY_SOURCE_GENERATION,
    ENTITY_EFFECT_DAMAGE_CAPACITY,
    ENTITY_EFFECT_FAILURE_DAMAGE_OVERFLOW,
    ENTITY_EFFECT_FAILURE_INVALID_DT,
    ENTITY_EFFECT_FAILURE_INVALID_STATE,
    ENTITY_EFFECT_FAILURE_MECHANICS,
    ENTITY_EFFECT_FAILURE_STALE_SOURCE,
    ENTITY_EFFECT_FAILURE_UPSTREAM,
    ENTITY_EFFECT_MAX_DT_SECONDS,
    ENTITY_EFFECTS_SCHEMA,
    ENTITY_EFFECTS_VERSION,
)
from hytalegym.jax.combat.entities.schema.spec import (
    HYTALE_0_5_7_SERVER_JAR_SHA256,
)
from hytalegym.jax.combat.mechanics import (
    COMBAT_MECHANICS_SCHEMA,
    COMBAT_MECHANICS_VERSION,
    CONTROL_IMMUNITY_INCREMENT,
    CONTROL_IMMUNITY_MAXIMUM,
    CONTROL_IMMUNITY_REGEN_AMOUNT,
    CONTROL_IMMUNITY_REGEN_INTERVAL_SECONDS,
    HYTALE_0_5_7_ASSETS_SHA256,
    HYTALE_0_5_7_NATIVE_EVIDENCE_JAR_SHA256,
    STATUS_CAPACITY,
)


def entity_effects_contract_manifest() -> dict[str, Any]:
    return {
        "schema": ENTITY_EFFECTS_SCHEMA,
        "version": ENTITY_EFFECTS_VERSION,
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
            "statuses_per_entity": STATUS_CAPACITY,
            "periodic_damage_events_per_microtick": (ENTITY_EFFECT_DAMAGE_CAPACITY),
        },
        "state_extension": {
            "source_generation": _field(
                ("B", ENTITY_CAPACITY, STATUS_CAPACITY),
                "uint32",
            ),
            "capability_bits": _field(("B",), "uint32"),
            "failure_bits": _field(("B",), "uint32"),
            "control_immunity": _field(
                ("B", ENTITY_CAPACITY),
                "float32",
            ),
            "control_immunity_regen_clock": _field(
                ("B", ENTITY_CAPACITY),
                "float32",
            ),
        },
        "control_immunity": {
            "maximum": CONTROL_IMMUNITY_MAXIMUM,
            "accepted_effect_increment": CONTROL_IMMUNITY_INCREMENT,
            "regen_amount": -CONTROL_IMMUNITY_REGEN_AMOUNT,
            "regen_interval_seconds": (CONTROL_IMMUNITY_REGEN_INTERVAL_SECONDS),
            "full_gate": "reject_wrapped_control_interaction",
        },
        "microtick": {
            "dt_dtype": "float32",
            "dt_shape": ["B"],
            "dt_range_seconds": [
                "exclusive_0",
                ENTITY_EFFECT_MAX_DT_SECONDS,
            ],
            "ordering": [
                "advance_fixed_interval_resources_and_control_immunity",
                "advance_status_clocks_and_periodic_status_resources",
                "apply_periodic_healing",
                "apply_entity_major_status_slot_major_damage",
                "commit_death",
            ],
            "nonliving": ("advance_effect_lifetime_without_gameplay_outputs"),
            "health_damage_rounding": "java_math_round_after_guard",
        },
        "overflow": {
            "damage_event_capacity": ENTITY_EFFECT_DAMAGE_CAPACITY,
            "behavior": "row_local_atomic_sticky_freeze",
        },
        "capability_bits": {
            "periodic_damage": (ENTITY_EFFECT_CAPABILITY_PERIODIC_DAMAGE),
            "periodic_healing": (ENTITY_EFFECT_CAPABILITY_PERIODIC_HEALING),
            "periodic_resources": (ENTITY_EFFECT_CAPABILITY_PERIODIC_RESOURCES),
            "source_generation": (ENTITY_EFFECT_CAPABILITY_SOURCE_GENERATION),
            "native_damage_rounding": (ENTITY_EFFECT_CAPABILITY_NATIVE_DAMAGE_ROUNDING),
        },
        "failure_bits": {
            "invalid_state": ENTITY_EFFECT_FAILURE_INVALID_STATE,
            "invalid_dt": ENTITY_EFFECT_FAILURE_INVALID_DT,
            "upstream": ENTITY_EFFECT_FAILURE_UPSTREAM,
            "stale_source": ENTITY_EFFECT_FAILURE_STALE_SOURCE,
            "damage_overflow": (ENTITY_EFFECT_FAILURE_DAMAGE_OVERFLOW),
            "mechanics": ENTITY_EFFECT_FAILURE_MECHANICS,
        },
        "scope": {
            "included": [
                "periodic_status_damage",
                "periodic_status_healing",
                "periodic_status_resources",
                "status_flags_and_speed",
                "source_generation_provenance",
                "logical_entity_death",
                "asset_gated_control_immunity",
            ],
            "deferred": [
                "native_effect_system_order_differential",
                "native_effect_removal_on_death_or_despawn",
                "asset_specific_purify",
                "physical_area_overlap_and_world_hazard_production",
            ],
        },
        "evidence": {
            "level": "jar_asset_resolved_not_native_differential",
            "hytale_version": "0.5.7",
            "server_jar_sha256": HYTALE_0_5_7_SERVER_JAR_SHA256,
            "assets_sha256": HYTALE_0_5_7_ASSETS_SHA256,
            "native_evidence_bridge_sha256": (HYTALE_0_5_7_NATIVE_EVIDENCE_JAR_SHA256),
        },
    }


def entity_effects_contract_json() -> str:
    return json.dumps(
        entity_effects_contract_manifest(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def entity_effects_contract_sha256() -> str:
    return (
        hashlib.sha256(entity_effects_contract_json().encode("utf-8"))
        .hexdigest()
        .upper()
    )


def _field(shape: tuple[str | int, ...], dtype: str) -> dict[str, Any]:
    return {"shape": list(shape), "dtype": dtype}


__all__ = [
    "entity_effects_contract_json",
    "entity_effects_contract_manifest",
    "entity_effects_contract_sha256",
]
