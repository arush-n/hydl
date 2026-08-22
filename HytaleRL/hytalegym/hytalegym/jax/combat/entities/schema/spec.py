"""Canonical adapter and checkpoint manifest for combat entities v2."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from hytalegym.jax.combat.entities.schema.contract import *
from hytalegym.jax.combat.mechanics import (
    COMBAT_MECHANICS_SCHEMA,
    COMBAT_MECHANICS_VERSION,
    HYTALE_0_5_7_ASSETS_SHA256,
    HYTALE_0_5_7_NATIVE_EVIDENCE_JAR_SHA256,
    RESOURCE_COUNT,
    STATUS_CAPACITY,
)


HYTALE_0_5_7_SERVER_JAR_SHA256 = (
    "43D9BCFF1DD31574577DBFC82147718DBE2AC16C19000071F965B521BA808CDC"
)


def combat_entities_contract_manifest() -> dict[str, Any]:
    return {
        "schema": COMBAT_ENTITIES_SCHEMA,
        "version": COMBAT_ENTITIES_VERSION,
        "mechanics": {
            "schema": COMBAT_MECHANICS_SCHEMA,
            "version": COMBAT_MECHANICS_VERSION,
        },
        "batch_axis": "B",
        "capacities": {
            "entities": ENTITY_CAPACITY,
            "spawns_per_step": SPAWN_CAPACITY,
            "despawns_per_step": DESPAWN_CAPACITY,
            "selector_queries_per_step": SELECTOR_CAPACITY,
            "damage_events_per_step": ENTITY_DAMAGE_CAPACITY,
            "statuses_per_entity": STATUS_CAPACITY,
            "resources_per_entity": RESOURCE_COUNT,
        },
        "roster": {
            "semantic_id": _field(("B", ENTITY_CAPACITY), "int32"),
            "generation": _field(("B", ENTITY_CAPACITY), "uint32"),
            "team_id": _field(("B", ENTITY_CAPACITY), "int32"),
            "position": _field(("B", ENTITY_CAPACITY, 3), "float32"),
            "velocity": _field(("B", ENTITY_CAPACITY, 3), "float32"),
            "yaw_degrees": _field(("B", ENTITY_CAPACITY), "float32"),
            "health": _field(("B", ENTITY_CAPACITY), "float32"),
            "max_health": _field(("B", ENTITY_CAPACITY), "float32"),
            "damageable": _field(("B", ENTITY_CAPACITY), "bool"),
            "intangible": _field(("B", ENTITY_CAPACITY), "bool"),
            "invulnerable": _field(("B", ENTITY_CAPACITY), "bool"),
            "dead": _field(("B", ENTITY_CAPACITY), "bool"),
            "active": _field(("B", ENTITY_CAPACITY), "bool"),
            "capability_bits": _field(("B",), "uint32"),
            "failure_bits": _field(("B",), "uint32"),
        },
        "mechanics_extension": {
            "control_immunity": _field(
                ("B", ENTITY_CAPACITY),
                "float32",
            ),
            "control_immunity_regen_clock": _field(
                ("B", ENTITY_CAPACITY),
                "float32",
            ),
        },
        "injected_selector_input": {
            "candidate_mask": _field(
                ("B", SELECTOR_CAPACITY, ENTITY_CAPACITY),
                "bool",
            ),
            "meaning": ("already-selected geometry and asset-matcher candidates"),
            "owner": "world_or_native_adapter",
        },
        "selection": {
            "ordering": "query_major_then_entity_slot",
            "default_owner_policy": "exclude_owner",
            "friendly_fire_default": True,
            "team_none": TEAM_NONE,
            "max_targets_zero": "uncapped",
            "oversized_capped_selector": ("fail_native_random_subset_no_targets"),
        },
        "damage_payload": {
            "random_percentage": _field(
                ("B", SELECTOR_CAPACITY),
                "float32",
            ),
            "damage_class": _field(
                ("B", SELECTOR_CAPACITY),
                "int32",
            ),
            "source_equipment_profile": (
                "resolved_by_shared_mechanics_before_target_filtering"
            ),
        },
        "capability_bits": {
            "lifecycle": ENTITY_CAPABILITY_LIFECYCLE,
            "exhaustive_selector": ENTITY_CAPABILITY_EXHAUSTIVE_SELECTOR,
            "team_filter": ENTITY_CAPABILITY_TEAM_FILTER,
            "multi_target_damage": ENTITY_CAPABILITY_MULTI_TARGET_DAMAGE,
            "shared_combat_mechanics": (ENTITY_CAPABILITY_SHARED_COMBAT_MECHANICS),
            "multi_target_status": ENTITY_CAPABILITY_MULTI_TARGET_STATUS,
        },
        "failure_bits": {
            "invalid_state": ENTITY_FAILURE_INVALID_STATE,
            "invalid_command": ENTITY_FAILURE_INVALID_COMMAND,
            "capacity": ENTITY_FAILURE_CAPACITY,
            "generation_exhausted": ENTITY_FAILURE_GENERATION_EXHAUSTED,
            "native_random_subset": ENTITY_FAILURE_NATIVE_RANDOM_SUBSET,
            "damage_overflow": ENTITY_FAILURE_DAMAGE_OVERFLOW,
            "combat_mechanics": ENTITY_FAILURE_COMBAT_MECHANICS,
            "status_overflow": ENTITY_FAILURE_STATUS_OVERFLOW,
        },
        "failure_behavior": {
            "lifecycle": "atomic_sticky_failure",
            "selection": "invalid_query_returns_no_targets",
            "damage": "atomic_sticky_failure_no_partial_hits",
        },
        "scope": {
            "included": [
                "logical_lifecycle",
                "stable_slot_generation_handles",
                "owner_and_team_target_filtering",
                "bounded_multi_target_damage",
                "bounded_multi_target_status_application",
                "shared_resources_guard_dodge_status_and_force",
                "control_immunity_state",
            ],
            "injected_not_implemented": [
                "selector_geometry",
                "line_of_sight",
                "physical_entity_occupancy",
                "world_collision",
            ],
            "deferred": [
                "arsenal_program_integration",
                "native_entity_adapter",
                "npc_behavior",
                "inventory_and_weapon_controllers",
                "corpse_and_asset_specific_despawn_timing",
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


def combat_entities_contract_json() -> str:
    return json.dumps(
        combat_entities_contract_manifest(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def combat_entities_contract_sha256() -> str:
    return (
        hashlib.sha256(combat_entities_contract_json().encode("utf-8"))
        .hexdigest()
        .upper()
    )


def _field(shape: tuple[str | int, ...], dtype: str) -> dict[str, Any]:
    return {"shape": list(shape), "dtype": dtype}


__all__ = [
    "HYTALE_0_5_7_SERVER_JAR_SHA256",
    "combat_entities_contract_json",
    "combat_entities_contract_manifest",
    "combat_entities_contract_sha256",
]
