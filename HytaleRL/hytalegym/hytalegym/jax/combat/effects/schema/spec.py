"""Canonical semantic manifest for combat effects v1."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from hytalegym.jax.combat.effects.schema.contract import (
    COMBAT_EFFECTS_SCHEMA,
    COMBAT_EFFECTS_VERSION,
    EFFECT_FAILURE_HAZARD_OVERFLOW,
    EFFECT_FAILURE_INVALID_COMMAND,
    EFFECT_FAILURE_INVALID_STATE,
    EFFECT_FAILURE_PROJECTILE_OVERFLOW,
    EFFECT_FAILURE_UNSUPPORTED_COLLISION,
    HAZARD_CAPACITY,
    HYTALE_0_5_7_ASSETS_SHA256,
    HYTALE_0_5_7_NATIVE_EVIDENCE_JAR_SHA256,
    OUTLANDER_ARROW_ASSET_SHA256,
    PROJECTILE_CAPACITY,
)


def combat_effects_contract_manifest() -> dict[str, Any]:
    """Return the hashable mechanics and fixed-shape interface contract."""

    return {
        "schema": COMBAT_EFFECTS_SCHEMA,
        "version": COMBAT_EFFECTS_VERSION,
        "capacities": {
            "projectiles": PROJECTILE_CAPACITY,
            "hazards": HAZARD_CAPACITY,
            "spawn_commands_per_decision": {
                "projectile": 1,
                "hazard": 1,
            },
        },
        "state": {
            "projectile_vector_fields": [
                "position",
                "velocity",
                "half_extent",
            ],
            "projectile_float_fields": [
                "age_seconds",
                "despawn_seconds",
                "authored_lifetime_seconds",
                "damage",
                "gravity",
                "terminal_velocity",
                "dead_time_seconds",
                "dead_time_remaining",
                "velocity_scale",
            ],
            "projectile_integer_fields": [
                "kind",
                "owner_entity_id",
                "flags",
            ],
            "projectile_boolean_fields": [
                "hostile",
                "active",
                "impacted",
                "physics_initialized",
                "entity_collision_only",
            ],
            "hazard_vector_fields": ["center", "half_extent"],
            "hazard_float_fields": [
                "age_seconds",
                "duration_seconds",
                "damage_per_second",
                "intensity",
                "activation",
            ],
            "hazard_integer_fields": [
                "kind",
                "owner_entity_id",
                "flags",
            ],
            "hazard_boolean_fields": [
                "hostile",
                "active",
                "entity_overlap_only",
            ],
        },
        "state_field_specs": {
            "projectiles": {
                **{
                    name: _field(("B", PROJECTILE_CAPACITY, 3), "float32")
                    for name in ("position", "velocity", "half_extent")
                },
                **{
                    name: _field(("B", PROJECTILE_CAPACITY), "float32")
                    for name in (
                        "age_seconds",
                        "despawn_seconds",
                        "authored_lifetime_seconds",
                        "damage",
                        "gravity",
                        "terminal_velocity",
                        "dead_time_seconds",
                        "dead_time_remaining",
                        "velocity_scale",
                    )
                },
                "kind": _field(("B", PROJECTILE_CAPACITY), "int32"),
                "owner_entity_id": _field(("B", PROJECTILE_CAPACITY), "int32"),
                "flags": _field(("B", PROJECTILE_CAPACITY), "uint32"),
                **{
                    name: _field(("B", PROJECTILE_CAPACITY), "bool")
                    for name in (
                        "hostile",
                        "active",
                        "impacted",
                        "physics_initialized",
                        "entity_collision_only",
                    )
                },
            },
            "hazards": {
                **{
                    name: _field(("B", HAZARD_CAPACITY, 3), "float32")
                    for name in ("center", "half_extent")
                },
                **{
                    name: _field(("B", HAZARD_CAPACITY), "float32")
                    for name in (
                        "age_seconds",
                        "duration_seconds",
                        "damage_per_second",
                        "intensity",
                        "activation",
                    )
                },
                "kind": _field(("B", HAZARD_CAPACITY), "int32"),
                "owner_entity_id": _field(("B", HAZARD_CAPACITY), "int32"),
                "flags": _field(("B", HAZARD_CAPACITY), "uint32"),
                **{
                    name: _field(("B", HAZARD_CAPACITY), "bool")
                    for name in (
                        "hostile",
                        "active",
                        "entity_overlap_only",
                    )
                },
            },
            "failure_bits": _field(("B",), "uint32"),
        },
        "command_field_specs": {
            "projectile": {
                "requested": _field(("B",), "bool"),
                **{
                    name: _field(("B", 3), "float32")
                    for name in ("position", "velocity", "half_extent")
                },
                **{
                    name: _field(("B",), "float32")
                    for name in (
                        "despawn_seconds",
                        "authored_lifetime_seconds",
                        "damage",
                        "gravity",
                        "terminal_velocity",
                        "dead_time_seconds",
                        "velocity_scale",
                    )
                },
                "kind": _field(("B",), "int32"),
                "owner_entity_id": _field(("B",), "int32"),
                "flags": _field(("B",), "uint32"),
                "hostile": _field(("B",), "bool"),
                "entity_collision_only": _field(("B",), "bool"),
            },
            "hazard": {
                "requested": _field(("B",), "bool"),
                "center": _field(("B", 3), "float32"),
                "half_extent": _field(("B", 3), "float32"),
                **{
                    name: _field(("B",), "float32")
                    for name in (
                        "duration_seconds",
                        "damage_per_second",
                        "intensity",
                        "activation",
                    )
                },
                "kind": _field(("B",), "int32"),
                "owner_entity_id": _field(("B",), "int32"),
                "flags": _field(("B",), "uint32"),
                "hostile": _field(("B",), "bool"),
                "entity_overlap_only": _field(("B",), "bool"),
            },
        },
        "failure_bits": {
            "projectile_overflow": EFFECT_FAILURE_PROJECTILE_OVERFLOW,
            "hazard_overflow": EFFECT_FAILURE_HAZARD_OVERFLOW,
            "unsupported_collision": EFFECT_FAILURE_UNSUPPORTED_COLLISION,
            "invalid_command": EFFECT_FAILURE_INVALID_COMMAND,
            "invalid_state": EFFECT_FAILURE_INVALID_STATE,
        },
        "failure_behavior": {
            "sticky": True,
            "combat_state": "freeze_at_decision_input",
            "effect_state": "freeze_at_decision_input_except_failure_bits",
            "reward": 0,
            "done": True,
            "learner_semantics": "truncated_not_terminated",
        },
        "composition_order": {
            "microtick": "calibrated_v8_then_effects",
            "entity_bounds_sample": "post_calibrated_v8_microtick",
            "effect_damage": "projectiles_then_hazards",
            "completion_and_death": "after_effect_damage",
        },
        "projectile_order": [
            "decrement_existing_impact_dead_timer_and_remove",
            "apply_previous_tick_force_state",
            "symplectic_velocity",
            "symplectic_position",
            "swept_entity_aabb",
            "damage_and_mark_impact",
            "despawn_age_check",
        ],
        "projectile_force_model": {
            "first_physics_tick": "no_gravity_or_drag",
            "drag": ("legacy_projected_aabb_area_terminal_velocity_resistance"),
            "gravity_axis": "-y",
            "collision": "entity_only_swept_center_against_expanded_aabb",
        },
        "hazard_model": {
            "shape": "static_axis_aligned_box",
            "damage": ("damage_per_second*overlap_seconds*intensity*activation"),
            "native_certification": False,
        },
        "world_boundary": {
            "block_collision": "unsupported_fail_closed",
            "fluid_collision": "unsupported_fail_closed",
            "dynamic_world_collision": "unsupported_fail_closed",
            "required_capability_flag": ("entity_collision_only/entity_overlap_only"),
        },
        "native_evidence": {
            "named_projectile": "Outlander_Hunter_Arrow",
            "asset_sha256": OUTLANDER_ARROW_ASSET_SHA256,
            "assets_zip_sha256": HYTALE_0_5_7_ASSETS_SHA256,
            "jar_sha256": HYTALE_0_5_7_NATIVE_EVIDENCE_JAR_SHA256,
            "server_version": "0.5.7",
            "live_trace_certified": False,
        },
    }


def combat_effects_contract_json() -> str:
    """Return stable canonical JSON."""

    return json.dumps(
        combat_effects_contract_manifest(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def combat_effects_contract_sha256() -> str:
    """Return the uppercase semantic contract digest."""

    return (
        hashlib.sha256(combat_effects_contract_json().encode("ascii"))
        .hexdigest()
        .upper()
    )


def _field(shape: tuple[str | int, ...], dtype: str) -> dict[str, Any]:
    return {"shape": list(shape), "dtype": dtype}
