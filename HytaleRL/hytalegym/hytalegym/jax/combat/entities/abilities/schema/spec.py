"""Canonical semantic manifest for 32-entity authored abilities."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from hytalegym.jax.combat.arsenal import (
    ABILITY_CAPACITY,
    ARSENAL_SCHEMA,
    ARSENAL_VERSION,
    EVENT_CAPACITY,
    EVENT_FLOAT_FEATURES,
    EVENT_INTEGER_FEATURES,
    EVENT_SCHEDULER_CLOCK_COUNT,
    PROFILE_NAMES,
    WORLD_CONDITIONAL_ABILITY_ASSETS,
    combat_arsenal_contract_sha256,
    hytale_0_5_7_catalog_counts,
)
from hytalegym.jax.combat.entities import (
    COMBAT_ENTITIES_SCHEMA,
    COMBAT_ENTITIES_VERSION,
    ENTITY_CAPACITY,
    combat_entities_contract_sha256,
)
from hytalegym.jax.combat.entities.abilities.schema.contract import (
    ABILITY_CAPABILITIES,
    ABILITY_CAPABILITY_COOLDOWNS,
    ABILITY_CAPABILITY_DIRECT_EVENTS,
    ABILITY_CAPABILITY_IMPACT_ROUTING,
    ABILITY_CAPABILITY_INTERRUPTION,
    ABILITY_CAPABILITY_RESOURCES,
    ABILITY_CAPABILITY_SCHEDULER,
    ABILITY_CAPABILITY_SEPARATE_WORLD_FAILURE,
    ABILITY_CAPABILITY_SOURCE_GENERATION,
    ABILITY_DAMAGE_CAPACITY,
    ABILITY_EVENT_CAPACITY,
    ABILITY_FAILURE_AMBIGUOUS_TARGET,
    ABILITY_FAILURE_AVAILABILITY,
    ABILITY_FAILURE_DAMAGE_OVERFLOW,
    ABILITY_FAILURE_EVENT_OVERFLOW,
    ABILITY_FAILURE_INVALID_COMMAND,
    ABILITY_FAILURE_INVALID_DT,
    ABILITY_FAILURE_INVALID_PROGRAM,
    ABILITY_FAILURE_INVALID_STATE,
    ABILITY_FAILURE_LAUNCH,
    ABILITY_FAILURE_MECHANICS,
    ABILITY_FAILURE_QUERY,
    ABILITY_FAILURE_STALE_SOURCE,
    ABILITY_FAILURE_STALE_TARGET,
    ABILITY_FAILURE_STATUS_OVERFLOW,
    ABILITY_FAILURE_UPSTREAM,
    ABILITY_PROGRAM_FAMILY_CAPACITY,
    ENTITY_ABILITIES_SCHEMA,
    ENTITY_ABILITIES_VERSION,
)
from hytalegym.jax.combat.entities.abilities.schema.identity import (
    entity_ability_program_sha256,
)
from hytalegym.jax.combat.entities.effects import (
    ENTITY_EFFECTS_SCHEMA,
    ENTITY_EFFECTS_VERSION,
    entity_effects_contract_sha256,
)
from hytalegym.jax.combat.entities.impacts import (
    ENTITY_IMPACTS_SCHEMA,
    ENTITY_IMPACTS_VERSION,
    entity_impacts_contract_sha256,
)
from hytalegym.jax.combat.entities.schema.spec import (
    HYTALE_0_5_7_SERVER_JAR_SHA256,
)
from hytalegym.jax.combat.mechanics import (
    COMBAT_MECHANICS_SCHEMA,
    COMBAT_MECHANICS_VERSION,
    HYTALE_0_5_7_ASSETS_SHA256,
    HYTALE_0_5_7_NATIVE_EVIDENCE_JAR_SHA256,
    RESOURCE_COUNT,
)


def entity_abilities_contract_manifest() -> dict[str, Any]:
    """Return the canonical model- and world-implementation-neutral contract."""

    event = ("B", ABILITY_EVENT_CAPACITY)
    entity = ("B", ENTITY_CAPACITY)
    return {
        "schema": ENTITY_ABILITIES_SCHEMA,
        "version": ENTITY_ABILITIES_VERSION,
        "dependencies": {
            "arsenal": _dependency(
                ARSENAL_SCHEMA,
                ARSENAL_VERSION,
                combat_arsenal_contract_sha256(),
            ),
            "entities": _dependency(
                COMBAT_ENTITIES_SCHEMA,
                COMBAT_ENTITIES_VERSION,
                combat_entities_contract_sha256(),
            ),
            "effects": _dependency(
                ENTITY_EFFECTS_SCHEMA,
                ENTITY_EFFECTS_VERSION,
                entity_effects_contract_sha256(),
            ),
            "impacts": _dependency(
                ENTITY_IMPACTS_SCHEMA,
                ENTITY_IMPACTS_VERSION,
                entity_impacts_contract_sha256(),
            ),
            "mechanics": {
                "schema": COMBAT_MECHANICS_SCHEMA,
                "version": COMBAT_MECHANICS_VERSION,
            },
        },
        "evidence": {
            "hytale_version": "0.5.7",
            "server_jar_sha256": HYTALE_0_5_7_SERVER_JAR_SHA256,
            "assets_sha256": HYTALE_0_5_7_ASSETS_SHA256,
            "native_bridge_evidence_sha256": (HYTALE_0_5_7_NATIVE_EVIDENCE_JAR_SHA256),
            "level": "asset_resolved_not_native_differential",
        },
        "capacities": {
            "entities": ENTITY_CAPACITY,
            "families": ABILITY_PROGRAM_FAMILY_CAPACITY,
            "abilities_per_family": ABILITY_CAPACITY,
            "events_per_ability": EVENT_CAPACITY,
            "packed_events_per_microtick": ABILITY_EVENT_CAPACITY,
            "direct_damage_events": ABILITY_DAMAGE_CAPACITY,
            "status_applications_per_target": 4,
            "status_clear_ids_per_event": 1,
            "resources": RESOURCE_COUNT,
        },
        "catalog": {
            "profiles": list(PROFILE_NAMES),
            **hytale_0_5_7_catalog_counts(),
            "compiled_program_sha256": (entity_ability_program_sha256()),
            "world_conditional_abilities": list(WORLD_CONDITIONAL_ABILITY_ASSETS),
            "source": "combat_arsenal_v1_program_bank",
        },
        "program_semantics": {
            "scheduler_prelude": (
                "per-family ability_scheduler_prelude_ticks delays every "
                "event/resource clock before the selected child program"
            ),
            "player_selector_context": (
                "this logical crowd runtime is role-only; its shared six-lane "
                "Arsenal v42 scheduler shape reserves the maximum authored "
                "delay, but it never infers Player context from inventory"
            ),
            "damage_randomization": (
                "caller-supplied batch keys; missing keys fail randomized damage closed"
            ),
            "resource_cost": (
                "mechanism-resolved native-NPC amount committed once at its "
                "authored queue/runtime/selector/parallel graph clock"
            ),
            "resource_minimum": (
                "pre-spend threshold; permits authored stamina overdraw"
            ),
        },
        "state": {
            "weapon_family": _field(entity, "int32"),
            "source_generation": _field(entity, "uint32"),
            "active_ability_slot": _field(entity, "int32"),
            "ability_elapsed_seconds": _field(entity, "float32"),
            "ability_scheduler_tick": _field(entity, "int32"),
            "ability_scheduler_clock_seconds": _field(
                entity + (EVENT_SCHEDULER_CLOCK_COUNT,),
                "float32",
            ),
            "ability_cooldown_seconds": _field(entity + (ABILITY_CAPACITY,), "float32"),
            "capability_bits": _field(("B",), "uint32"),
            "failure_bits": _field(("B",), "uint32"),
            "world_failure_bits": _field(("B",), "uint32"),
        },
        "commands": {
            "requested_slot": _field(entity, "int32"),
            "interrupted": _field(entity, "bool"),
            "failure_bits": _field(("B",), "uint32"),
            "valid": _field(("B",), "bool"),
        },
        "availability": {
            "source_generation": _field(entity, "uint32"),
            "available_requirement_bits": _field(entity, "uint32"),
            "valid": _field(entity, "bool"),
            "failure_bits": _field(entity, "uint32"),
        },
        "packed_events": {
            "requested": _field(event, "bool"),
            "source_slot": _field(event, "int32"),
            "source_generation": _field(event, "uint32"),
            "weapon_family": _field(event, "int32"),
            "ability_slot": _field(event, "int32"),
            "event_slot": _field(event, "int32"),
            "kind": _field(event, "int32"),
            "f32": _field(event + (EVENT_FLOAT_FEATURES,), "float32"),
            "i32": _field(event + (EVENT_INTEGER_FEATURES,), "int32"),
            "flags": _field(event, "uint32"),
            "overflow": _field(("B",), "bool"),
            "failure_bits": _field(("B",), "uint32"),
            "world_failure_bits": _field(("B",), "uint32"),
            "valid": _field(("B",), "bool"),
        },
        "damage_event_semantics": {
            "damage_class_source": "arsenal_i32_damage_class_channel",
            "direct_event_binding": (
                "preserve_random_percentage_and_damage_class"
            ),
            "source_equipment_formula": (
                "(randomized_damage + additive) "
                "* max(0, 1 + multiplicative)"
            ),
            "ordering": "before_guard_stamina_and_target_resistance",
        },
        "direct_query": {
            "candidate_mask": _field(event + (ENTITY_CAPACITY,), "bool"),
            "candidate_generation": _field(event + (ENTITY_CAPACITY,), "uint32"),
            "friendly_fire": _field(event, "bool"),
            "query_valid": _field(event, "bool"),
            "failure_bits": _field(event, "uint32"),
            "ownership": (
                "external owner supplies already-selected physical "
                "candidates; combat filters owner/team/generation"
            ),
        },
        "capabilities": {
            "bits": ABILITY_CAPABILITIES,
            "scheduler": ABILITY_CAPABILITY_SCHEDULER,
            "resources": ABILITY_CAPABILITY_RESOURCES,
            "cooldowns": ABILITY_CAPABILITY_COOLDOWNS,
            "interruption": ABILITY_CAPABILITY_INTERRUPTION,
            "direct_events": ABILITY_CAPABILITY_DIRECT_EVENTS,
            "impact_routing": ABILITY_CAPABILITY_IMPACT_ROUTING,
            "source_generation": (ABILITY_CAPABILITY_SOURCE_GENERATION),
            "separate_world_failure": (ABILITY_CAPABILITY_SEPARATE_WORLD_FAILURE),
        },
        "failures": {
            "invalid_state": ABILITY_FAILURE_INVALID_STATE,
            "invalid_dt": ABILITY_FAILURE_INVALID_DT,
            "invalid_program": ABILITY_FAILURE_INVALID_PROGRAM,
            "invalid_command": ABILITY_FAILURE_INVALID_COMMAND,
            "stale_source": ABILITY_FAILURE_STALE_SOURCE,
            "availability": ABILITY_FAILURE_AVAILABILITY,
            "event_overflow": ABILITY_FAILURE_EVENT_OVERFLOW,
            "upstream": ABILITY_FAILURE_UPSTREAM,
            "query": ABILITY_FAILURE_QUERY,
            "stale_target": ABILITY_FAILURE_STALE_TARGET,
            "ambiguous_target": ABILITY_FAILURE_AMBIGUOUS_TARGET,
            "damage_overflow": ABILITY_FAILURE_DAMAGE_OVERFLOW,
            "status_overflow": ABILITY_FAILURE_STATUS_OVERFLOW,
            "mechanics": ABILITY_FAILURE_MECHANICS,
            "launch": ABILITY_FAILURE_LAUNCH,
        },
        "order": [
            "interrupt",
            "check_legality",
            "spend_resources_and_start_cooldown",
            "advance_half_open_timed_events",
            "heal",
            "resource_events",
            "direct_forces",
            "statuses",
            "direct_damage_and_damage_forces",
            "projectile_and_area_allocation",
        ],
        "atomicity": (
            "any combat, capacity, provenance, query, or launch failure "
            "reverts the full row and preserves sticky failure evidence"
        ),
    }


def entity_abilities_contract_json() -> str:
    return json.dumps(
        entity_abilities_contract_manifest(),
        sort_keys=True,
        separators=(",", ":"),
    )


def entity_abilities_contract_sha256() -> str:
    return (
        hashlib.sha256(entity_abilities_contract_json().encode("utf-8"))
        .hexdigest()
        .upper()
    )


def _field(shape, dtype):
    return {"shape": list(shape), "dtype": dtype}


def _dependency(schema, version, sha256):
    return {
        "schema": schema,
        "version": version,
        "sha256": sha256,
    }
