"""Canonical manifest for crowd projectile and area integration."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from hytalegym.jax.combat.arsenal import (
    ABILITY_CAPACITY,
    AREA_CAPACITY,
    ARSENAL_SCHEMA,
    ARSENAL_VERSION,
    EVENT_FLOAT_FEATURES,
    EVENT_INTEGER_FEATURES,
    FAMILY_BOMB,
    FAMILY_CROSSBOW,
    FAMILY_FLAME_STAFF,
    FAMILY_ICE_STAFF,
    FAMILY_KUNAI,
    FAMILY_SHORTBOW,
    FAMILY_SPELLBOOK,
    PROJECTILE_CAPACITY,
    combat_arsenal_contract_sha256,
)
from hytalegym.jax.combat.controllers import (
    CONTROLLER_SCHEMA,
    CONTROLLER_VERSION,
    ranged_controller_contract_sha256,
)
from hytalegym.jax.combat.entities import (
    COMBAT_ENTITIES_SCHEMA,
    COMBAT_ENTITIES_VERSION,
    ENTITY_CAPACITY,
    combat_entities_contract_sha256,
)
from hytalegym.jax.combat.entities.effects import (
    ENTITY_EFFECTS_SCHEMA,
    ENTITY_EFFECTS_VERSION,
    entity_effects_contract_sha256,
)
from hytalegym.jax.combat.entities.impacts.schema.contract import (
    AREAS_PER_LAUNCH,
    AREA_PROGRAM_FAMILY_CAPACITY,
    ENTITY_IMPACTS_SCHEMA,
    ENTITY_IMPACTS_VERSION,
    IMPACT_CAPABILITY_AREA_PULSE,
    IMPACT_CAPABILITY_CROSSBOW_ROUTING,
    IMPACT_CAPABILITY_ENTITY_AREA_ALLOCATION,
    IMPACT_CAPABILITY_ENTITY_ONLY_EXPLOSION_BINDING,
    IMPACT_CAPABILITY_ENTITY_PROJECTILE_ALLOCATION,
    IMPACT_CAPABILITY_EXPLOSION,
    IMPACT_CAPABILITY_PROJECTILE_SWEEP,
    IMPACT_CAPABILITY_RANGED_LAUNCH_ALLOCATION,
    IMPACT_CAPABILITY_RANGED_LAUNCH_BINDING,
    IMPACT_CAPABILITY_SEPARATE_WORLD_FAILURE,
    IMPACT_CAPABILITY_SOURCE_GENERATION,
    IMPACT_CAPABILITY_WORLD_CONTACT,
    IMPACT_DAMAGE_CAPACITY,
    IMPACT_FAILURE_AMBIGUOUS_TRIGGER,
    IMPACT_FAILURE_CROSSBOW,
    IMPACT_FAILURE_DAMAGE_OVERFLOW,
    IMPACT_FAILURE_INVALID_BINDING,
    IMPACT_FAILURE_INVALID_DT,
    IMPACT_FAILURE_INVALID_STATE,
    IMPACT_FAILURE_MECHANICS,
    IMPACT_FAILURE_MIXED_INTERACTION_ORDER,
    IMPACT_FAILURE_QUERY,
    IMPACT_FAILURE_STALE_SOURCE,
    IMPACT_FAILURE_STALE_TARGET,
    IMPACT_FAILURE_UPSTREAM,
    EXPLOSION_QUERY_FAILURE_CAPACITY_EXCEEDED,
    EXPLOSION_QUERY_FAILURE_DAMAGE_BLOCKS_UNSUPPORTED,
    EXPLOSION_QUERY_FAILURE_GEOMETRY_EXHAUSTED,
    EXPLOSION_QUERY_FAILURE_INVALID,
    PROJECTILE_PROGRAM_FAMILY_CAPACITY,
    RANGED_PROJECTILES_PER_LAUNCH,
)
from hytalegym.jax.combat.entities.interactions import (
    ENTITY_INTERACTIONS_SCHEMA,
    ENTITY_INTERACTIONS_VERSION,
    entity_interactions_contract_sha256,
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


def entity_impacts_contract_manifest() -> dict[str, Any]:
    return {
        "schema": ENTITY_IMPACTS_SCHEMA,
        "version": ENTITY_IMPACTS_VERSION,
        "dependencies": {
            "arsenal": _dependency(
                ARSENAL_SCHEMA,
                ARSENAL_VERSION,
                combat_arsenal_contract_sha256(),
            ),
            "ranged_controllers": _dependency(
                CONTROLLER_SCHEMA,
                CONTROLLER_VERSION,
                ranged_controller_contract_sha256(),
            ),
            "entities": _dependency(
                COMBAT_ENTITIES_SCHEMA,
                COMBAT_ENTITIES_VERSION,
                combat_entities_contract_sha256(),
            ),
            "entity_effects": _dependency(
                ENTITY_EFFECTS_SCHEMA,
                ENTITY_EFFECTS_VERSION,
                entity_effects_contract_sha256(),
            ),
            "entity_interactions": _dependency(
                ENTITY_INTERACTIONS_SCHEMA,
                ENTITY_INTERACTIONS_VERSION,
                entity_interactions_contract_sha256(),
            ),
            "mechanics": {
                "schema": COMBAT_MECHANICS_SCHEMA,
                "version": COMBAT_MECHANICS_VERSION,
            },
        },
        "capacities": {
            "entities": ENTITY_CAPACITY,
            "projectiles": PROJECTILE_CAPACITY,
            "areas": AREA_CAPACITY,
            "damage_events_per_phase": IMPACT_DAMAGE_CAPACITY,
            "projectile_program_families": (PROJECTILE_PROGRAM_FAMILY_CAPACITY),
            "area_program_families": (AREA_PROGRAM_FAMILY_CAPACITY),
            "ranged_projectiles_per_launch": (RANGED_PROJECTILES_PER_LAUNCH),
            "areas_per_launch": AREAS_PER_LAUNCH,
        },
        "damage_randomization": {
            "keys": _field(("B",), "jax_prng_key"),
            "projectile_area_namespaces": "independent_fold_in",
            "missing_keys": "randomized_damage_fails_closed",
        },
        "projectile_program_bank": {
            "event_mask": _field(
                (
                    PROJECTILE_PROGRAM_FAMILY_CAPACITY,
                    ABILITY_CAPACITY,
                    RANGED_PROJECTILES_PER_LAUNCH,
                ),
                "bool",
            ),
            "event_f32": _field(
                (
                    PROJECTILE_PROGRAM_FAMILY_CAPACITY,
                    ABILITY_CAPACITY,
                    RANGED_PROJECTILES_PER_LAUNCH,
                    EVENT_FLOAT_FEATURES,
                ),
                "float32",
            ),
            "event_i32": _field(
                (
                    PROJECTILE_PROGRAM_FAMILY_CAPACITY,
                    ABILITY_CAPACITY,
                    RANGED_PROJECTILES_PER_LAUNCH,
                    EVENT_INTEGER_FEATURES,
                ),
                "int32",
            ),
            "event_flags": _field(
                (
                    PROJECTILE_PROGRAM_FAMILY_CAPACITY,
                    ABILITY_CAPACITY,
                    RANGED_PROJECTILES_PER_LAUNCH,
                ),
                "uint32",
            ),
            "overflow": _field((PROJECTILE_PROGRAM_FAMILY_CAPACITY,), "bool"),
            "families": {
                "iron_shortbow": FAMILY_SHORTBOW,
                "iron_crossbow": FAMILY_CROSSBOW,
                "flame_staff": FAMILY_FLAME_STAFF,
                "ice_staff": FAMILY_ICE_STAFF,
                "bombs": FAMILY_BOMB,
                "skeleton_mage_spellbook": FAMILY_SPELLBOOK,
                "kunai": FAMILY_KUNAI,
            },
        },
        "area_program_bank": {
            "event_mask": _field(
                (
                    AREA_PROGRAM_FAMILY_CAPACITY,
                    ABILITY_CAPACITY,
                    AREAS_PER_LAUNCH,
                ),
                "bool",
            ),
            "event_f32": _field(
                (
                    AREA_PROGRAM_FAMILY_CAPACITY,
                    ABILITY_CAPACITY,
                    AREAS_PER_LAUNCH,
                    EVENT_FLOAT_FEATURES,
                ),
                "float32",
            ),
            "event_i32": _field(
                (
                    AREA_PROGRAM_FAMILY_CAPACITY,
                    ABILITY_CAPACITY,
                    AREAS_PER_LAUNCH,
                    EVENT_INTEGER_FEATURES,
                ),
                "int32",
            ),
            "event_flags": _field(
                (
                    AREA_PROGRAM_FAMILY_CAPACITY,
                    ABILITY_CAPACITY,
                    AREAS_PER_LAUNCH,
                ),
                "uint32",
            ),
            "overflow": _field((AREA_PROGRAM_FAMILY_CAPACITY,), "bool"),
            "families": {"flame_staff": FAMILY_FLAME_STAFF},
        },
        "entity_projectile_launch_commands": {
            "weapon_family": _field(("B", ENTITY_CAPACITY), "int32"),
            "ability_slot": _field(("B", ENTITY_CAPACITY), "int32"),
            "requested": _field(("B", ENTITY_CAPACITY), "bool"),
            "damage_multiplier": _field(("B", ENTITY_CAPACITY), "float32"),
            "friendly_fire": _field(("B", ENTITY_CAPACITY), "bool"),
            "failure_bits": _field(("B",), "uint32"),
            "valid": _field(("B",), "bool"),
        },
        "projectile_launch_world": {
            "muzzle_position": _field(("B", ENTITY_CAPACITY, 3), "float32"),
            "muzzle_yaw_degrees": _field(("B", ENTITY_CAPACITY), "float32"),
            "muzzle_pitch_degrees": _field(("B", ENTITY_CAPACITY), "float32"),
            "source_generation": _field(("B", ENTITY_CAPACITY), "uint32"),
            "muzzle_valid": _field(("B", ENTITY_CAPACITY), "bool"),
            "failure_bits": _field(("B", ENTITY_CAPACITY), "uint32"),
        },
        "projectile_launch_info": {
            "projectile_requested": _field(("B",), "int32"),
            "projectile_spawned": _field(("B",), "int32"),
            "projectile_slots": _field(("B", PROJECTILE_CAPACITY), "bool"),
            "world_failure_bits": _field(("B",), "uint32"),
            "failure_bits": _field(("B",), "uint32"),
            "valid": _field(("B",), "bool"),
        },
        "entity_area_launch_commands": {
            "weapon_family": _field(("B", ENTITY_CAPACITY), "int32"),
            "ability_slot": _field(("B", ENTITY_CAPACITY), "int32"),
            "requested": _field(("B", ENTITY_CAPACITY), "bool"),
            "damage_multiplier": _field(("B", ENTITY_CAPACITY), "float32"),
            "friendly_fire": _field(("B", ENTITY_CAPACITY), "bool"),
            "failure_bits": _field(("B",), "uint32"),
            "valid": _field(("B",), "bool"),
        },
        "area_launch_world": {
            "area_center": _field(("B", ENTITY_CAPACITY, 3), "float32"),
            "source_yaw_degrees": _field(("B", ENTITY_CAPACITY), "float32"),
            "source_generation": _field(("B", ENTITY_CAPACITY), "uint32"),
            "area_valid": _field(("B", ENTITY_CAPACITY), "bool"),
            "failure_bits": _field(("B", ENTITY_CAPACITY), "uint32"),
        },
        "area_launch_info": {
            "area_requested": _field(("B",), "int32"),
            "area_spawned": _field(("B",), "int32"),
            "area_slots": _field(("B", AREA_CAPACITY), "bool"),
            "world_failure_bits": _field(("B",), "uint32"),
            "failure_bits": _field(("B",), "uint32"),
            "valid": _field(("B",), "bool"),
        },
        "bindings": {
            "projectile_source_generation": _field(
                ("B", PROJECTILE_CAPACITY), "uint32"
            ),
            "projectile_random_percentage": _field(
                ("B", PROJECTILE_CAPACITY), "float32"
            ),
            "projectile_damage_class": _field(
                ("B", PROJECTILE_CAPACITY), "int32"
            ),
            "projectile_interaction_kind": _field(("B", PROJECTILE_CAPACITY), "int32"),
            "projectile_damage_multiplier": _field(
                ("B", PROJECTILE_CAPACITY), "float32"
            ),
            "projectile_knockback_yaw_degrees": _field(
                ("B", PROJECTILE_CAPACITY), "float32"
            ),
            "projectile_friendly_fire": _field(("B", PROJECTILE_CAPACITY), "bool"),
            "area_source_generation": _field(("B", AREA_CAPACITY), "uint32"),
            "area_random_percentage": _field(
                ("B", AREA_CAPACITY), "float32"
            ),
            "area_damage_class": _field(("B", AREA_CAPACITY), "int32"),
            "area_friendly_fire": _field(("B", AREA_CAPACITY), "bool"),
            "overflow": _field(("B",), "bool"),
        },
        "damage_event_semantics": {
            "lifecycle_binding": (
                "projectile_and_area_state_preserve_random_percentage_"
                "and_damage_class_until_impact"
            ),
            "resolution": "shared_combat_mechanics_damage_pipeline",
            "source_equipment_formula": (
                "(randomized_damage + additive) "
                "* max(0, 1 + multiplicative)"
            ),
        },
        "world_query": {
            "projectile_entity_hit_mask": _field(
                ("B", PROJECTILE_CAPACITY, ENTITY_CAPACITY),
                "bool",
            ),
            "projectile_entity_hit_fraction": _field(
                ("B", PROJECTILE_CAPACITY, ENTITY_CAPACITY),
                "float32",
            ),
            "projectile_candidate_generation": _field(
                ("B", PROJECTILE_CAPACITY, ENTITY_CAPACITY),
                "uint32",
            ),
            "projectile_world_hit": _field(("B", PROJECTILE_CAPACITY), "bool"),
            "projectile_world_hit_fraction": _field(
                ("B", PROJECTILE_CAPACITY), "float32"
            ),
            "projectile_explosion_candidate_mask": _field(
                ("B", PROJECTILE_CAPACITY, ENTITY_CAPACITY),
                "bool",
            ),
            "projectile_explosion_distance": _field(
                ("B", PROJECTILE_CAPACITY, ENTITY_CAPACITY),
                "float32",
            ),
            "projectile_query_valid": _field(("B", PROJECTILE_CAPACITY), "bool"),
            "projectile_failure_bits": _field(("B", PROJECTILE_CAPACITY), "uint32"),
            "area_candidate_mask": _field(
                ("B", AREA_CAPACITY, ENTITY_CAPACITY), "bool"
            ),
            "area_candidate_generation": _field(
                ("B", AREA_CAPACITY, ENTITY_CAPACITY), "uint32"
            ),
            "area_query_valid": _field(("B", AREA_CAPACITY), "bool"),
            "area_failure_bits": _field(("B", AREA_CAPACITY), "uint32"),
        },
        "ordering": [
            "validate_controller_program_source_and_muzzle",
            "atomically_allocate_controller_requested_projectiles",
            "bind_new_slots_to_generation_pinned_sources",
            "validate_area_program_source_and_placement",
            "atomically_allocate_and_bind_requested_areas",
            "validate_state_bindings_sources_and_queries",
            "advance_projectile_physics_and_choose_entity_world_or_fuse",
            "resolve_generic_projectile_damage_then_status",
            "resolve_exclusive_ordered_crossbow_chain",
            "resolve_area_damage_then_status",
            "commit_projectile_area_lifecycle_and_death",
        ],
        "trigger_ties": "fail_closed_without_invented_native_order",
        "overflow": {
            "behavior": "row_local_atomic_sticky_freeze",
            "mixed_generic_crossbow_same_tick": "fail_closed",
        },
        "failure_domains": {
            "impact_failure_bits": "combat_owned_uint32",
            "world_failure_bits": "world_owned_uint32_preserved_separately",
            "entity_only_explosion_translation": {
                "geometry_exhausted": (
                    EXPLOSION_QUERY_FAILURE_GEOMETRY_EXHAUSTED
                ),
                "capacity_exceeded": (
                    EXPLOSION_QUERY_FAILURE_CAPACITY_EXCEEDED
                ),
                "damage_blocks_unsupported": (
                    EXPLOSION_QUERY_FAILURE_DAMAGE_BLOCKS_UNSUPPORTED
                ),
                "invalid": EXPLOSION_QUERY_FAILURE_INVALID,
            },
        },
        "capability_bits": {
            "projectile_sweep": IMPACT_CAPABILITY_PROJECTILE_SWEEP,
            "world_contact": IMPACT_CAPABILITY_WORLD_CONTACT,
            "explosion": IMPACT_CAPABILITY_EXPLOSION,
            "area_pulse": IMPACT_CAPABILITY_AREA_PULSE,
            "source_generation": (IMPACT_CAPABILITY_SOURCE_GENERATION),
            "crossbow_routing": (IMPACT_CAPABILITY_CROSSBOW_ROUTING),
            "separate_world_failure": (IMPACT_CAPABILITY_SEPARATE_WORLD_FAILURE),
            "ranged_launch_binding": (IMPACT_CAPABILITY_RANGED_LAUNCH_BINDING),
            "ranged_launch_allocation": (IMPACT_CAPABILITY_RANGED_LAUNCH_ALLOCATION),
            "entity_projectile_allocation": (
                IMPACT_CAPABILITY_ENTITY_PROJECTILE_ALLOCATION
            ),
            "entity_area_allocation": (IMPACT_CAPABILITY_ENTITY_AREA_ALLOCATION),
            "entity_only_explosion_binding": (
                IMPACT_CAPABILITY_ENTITY_ONLY_EXPLOSION_BINDING
            ),
        },
        "failure_bits": {
            "invalid_state": IMPACT_FAILURE_INVALID_STATE,
            "invalid_dt": IMPACT_FAILURE_INVALID_DT,
            "invalid_binding": IMPACT_FAILURE_INVALID_BINDING,
            "stale_source": IMPACT_FAILURE_STALE_SOURCE,
            "query": IMPACT_FAILURE_QUERY,
            "ambiguous_trigger": IMPACT_FAILURE_AMBIGUOUS_TRIGGER,
            "damage_overflow": IMPACT_FAILURE_DAMAGE_OVERFLOW,
            "mixed_interaction_order": (IMPACT_FAILURE_MIXED_INTERACTION_ORDER),
            "upstream": IMPACT_FAILURE_UPSTREAM,
            "mechanics": IMPACT_FAILURE_MECHANICS,
            "crossbow": IMPACT_FAILURE_CROSSBOW,
            "stale_target": IMPACT_FAILURE_STALE_TARGET,
        },
        "scope": {
            "included": [
                "standard_projectile_motion",
                "entity_world_fuse_trigger_order",
                "multi_target_explosion_falloff",
                "persistent_area_interval_pulses",
                "team_and_owner_filtering",
                "source_and_target_generation_checks",
                "controller_requested_ranged_projectile_allocation",
                "event_ready_flame_ice_and_bomb_projectile_allocation",
                "event_ready_flame_staff_area_allocation",
                "post_spawn_launch_count_and_source_binding",
                "stable_fragmented_projectile_slot_allocation",
                "stable_fragmented_area_slot_allocation",
                "launch_time_broken_weapon_damage_multiplier",
                "explicit_crossbow_impact_yaw_requirement",
                "generic_damage_status_force_and_death",
                "wrapped_control_immunity_gating",
                "dynamic_point_force_direction_for_explosions",
                "public_world_entity_only_explosion_result_binding",
                "crossbow_combo_and_big_arrow_routing",
            ],
            "injected": [
                "swept_entity_contact_fractions",
                "world_contact_fraction",
                "explosion_overlap_and_closest_distance",
                "persistent_area_overlap",
                "generation_correct_muzzle_transform",
                "generation_correct_static_area_placement",
                "world_failure_bits",
            ],
            "deferred": [
                "production_region_world_query_composition",
                "fused_controller_launch_impact_microtick",
                "crowd_magic_resource_cooldown_and_charge_controller",
                "authoritative_crossbow_impact_yaw_production",
                "block_mutation_from_explosions",
                "native_ecs_tie_and_mixed_interaction_order",
                "live_native_projectile_area_differentials",
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


def entity_impacts_contract_json() -> str:
    return json.dumps(
        entity_impacts_contract_manifest(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def entity_impacts_contract_sha256() -> str:
    return (
        hashlib.sha256(entity_impacts_contract_json().encode("utf-8"))
        .hexdigest()
        .upper()
    )


def _dependency(schema, version, sha256):
    return {"schema": schema, "version": version, "sha256": sha256}


def _field(shape, dtype):
    return {"shape": list(shape), "dtype": dtype}


__all__ = [
    "entity_impacts_contract_json",
    "entity_impacts_contract_manifest",
    "entity_impacts_contract_sha256",
]
