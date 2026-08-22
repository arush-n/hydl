"""Canonical checkpoint and adapter manifest for observation version 3."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from hytalegym.jax.combat.arsenal.schema.contract import OBSERVATION_CAPACITY
from hytalegym.jax.combat.mechanics import (
    RESOURCE_COUNT,
    STATUS_CAPACITY,
)
from hytalegym.jax.combat.observation.v1.schema.contract import (
    DOOR_INTENT_COUNT,
    INTERACTION_CAPACITY,
    LEARNER_OBSERVATION_SCHEMA,
    LEARNER_OBSERVATION_VERSION,
)
from hytalegym.jax.combat.observation.v1.schema.spec import (
    learner_observation_contract_sha256,
)
from hytalegym.jax.combat.observation.v3.schema.contract import (
    ABILITY_FLOAT_FEATURES,
    ABILITY_FLOAT_SIZE,
    ABILITY_INTEGER_FEATURES,
    ABILITY_INTEGER_SIZE,
    ACTOR_WORLD_FLOAT_FEATURES,
    ACTOR_WORLD_FLOAT_SIZE,
    ACTOR_WORLD_MASK_FEATURES,
    ACTOR_WORLD_MASK_SIZE,
    DEFENSE_FLOAT_FEATURES,
    DEFENSE_FLOAT_SIZE,
    DODGE_ACTION_COUNT,
    LEARNER_OBSERVATION_V3_SCHEMA,
    LEARNER_OBSERVATION_V3_VERSION,
    MOVEMENT_STATE_FEATURES,
    MOVEMENT_STATE_SIZE,
    OBSERVATION_FAILURE_ARSENAL,
    OBSERVATION_FAILURE_LOADOUT,
    OBSERVATION_FAILURE_MECHANICS,
    STATUS_FLOAT_FEATURES,
    STATUS_FLOAT_SIZE,
    STATUS_INTEGER_FEATURES,
    STATUS_INTEGER_SIZE,
    WEAPON_INTEGER_FEATURES,
    WEAPON_INTEGER_SIZE,
)
from hytalegym.jax.combat.observation.v3.schema.types import (
    LearnerCombatObservationV3,
    LearnerObservationV3ActorEvidence,
)
from hytalegym.jax.combat.observation.v3.tokens.world import (
    DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG,
    WORLD_GEOMETRY_SOURCE_POLICY_FIELDS,
    world_geometry_policy_config_manifest,
)
from hytalegym.jax.combat.skills import SKILL_COUNT
from hytalegym.jax.combat.types import ENTITY_COUNT
from hytalegym.jax.world import (
    MOVEMENT_STATE_SCHEMA,
    MOVEMENT_STATE_VERSION,
    WORLD_GEOMETRY_TOKEN_SCHEMA,
    WORLD_GEOMETRY_TOKEN_VERSION,
    movement_state_contract_sha256,
    world_geometry_token_contract_sha256,
)
from hytalegym.rulesets.movement_projection import (
    ACTOR_WALK_MOVEMENT_PROJECTION_SCHEMA,
    ACTOR_WALK_MOVEMENT_PROJECTION_VERSION,
    actor_walk_movement_projection_sha256,
)


def learner_observation_v3_contract_manifest(
    *,
    entity_count: int = ENTITY_COUNT,
) -> dict[str, Any]:
    _validate_entity_count(entity_count)
    fields = {
        "schema_version": _field(("B",), "int32"),
        "base": {
            "schema": LEARNER_OBSERVATION_SCHEMA,
            "version": LEARNER_OBSERVATION_VERSION,
            "sha256": learner_observation_contract_sha256(),
        },
        "weapon_i32": _field(
            ("B", entity_count, WEAPON_INTEGER_SIZE),
            "int32",
        ),
        "resource_f32": _field(
            ("B", entity_count, RESOURCE_COUNT),
            "float32",
        ),
        "resource_mask": _field(
            ("B", entity_count, RESOURCE_COUNT),
            "bool",
        ),
        "defense_f32": _field(
            ("B", entity_count, DEFENSE_FLOAT_SIZE),
            "float32",
        ),
        "status_f32": _field(
            ("B", entity_count, STATUS_CAPACITY, STATUS_FLOAT_SIZE),
            "float32",
        ),
        "status_i32": _field(
            ("B", entity_count, STATUS_CAPACITY, STATUS_INTEGER_SIZE),
            "int32",
        ),
        "status_flags": _field(
            ("B", entity_count, STATUS_CAPACITY),
            "uint32",
        ),
        "status_mask": _field(
            ("B", entity_count, STATUS_CAPACITY),
            "bool",
        ),
        "ability_f32": _field(
            ("B", entity_count, OBSERVATION_CAPACITY, ABILITY_FLOAT_SIZE),
            "float32",
        ),
        "ability_i32": _field(
            ("B", entity_count, OBSERVATION_CAPACITY, ABILITY_INTEGER_SIZE),
            "int32",
        ),
        "ability_mask": _field(
            ("B", entity_count, OBSERVATION_CAPACITY),
            "bool",
        ),
        "ability_legal": _field(
            ("B", entity_count, OBSERVATION_CAPACITY),
            "bool",
        ),
        "actor_world_f32": _field(
            ("B", ACTOR_WORLD_FLOAT_SIZE),
            "float32",
        ),
        "actor_world_mask": _field(
            ("B", ACTOR_WORLD_MASK_SIZE),
            "bool",
        ),
        "movement_state_f32": _field(
            ("B", MOVEMENT_STATE_SIZE),
            "float32",
        ),
        "movement_state_mask": _field(
            ("B", MOVEMENT_STATE_SIZE),
            "bool",
        ),
        "world_geometry": {
            "type": "WorldGeometryPolicyTokens",
            "available": _field(("B",), "bool"),
            "token_f32": _field(("B", "T", "F"), "float32"),
            "token_mask": _field(("B", "T"), "bool"),
        },
        "skill_action_mask": _field(("B", SKILL_COUNT), "bool"),
        "ability_action_mask": _field(
            ("B", OBSERVATION_CAPACITY),
            "bool",
        ),
        "jump_action_mask": _field(("B",), "bool"),
        "guard_action_mask": _field(("B",), "bool"),
        "dodge_action_mask": _field(
            ("B", DODGE_ACTION_COUNT),
            "bool",
        ),
        "door_action_mask": _field(
            ("B", INTERACTION_CAPACITY, DOOR_INTENT_COUNT),
            "bool",
        ),
        "valid": _field(("B",), "bool"),
        "failure_bits": _field(("B",), "uint32"),
        "mechanics_failure_bits": _field(("B",), "uint32"),
        "arsenal_failure_bits": _field(("B",), "uint32"),
    }
    return {
        "schema": LEARNER_OBSERVATION_V3_SCHEMA,
        "version": LEARNER_OBSERVATION_V3_VERSION,
        "batch_axis": "B",
        "fields": fields,
        "field_order": list(LearnerCombatObservationV3._fields),
        "feature_order": {
            "weapon_i32": list(WEAPON_INTEGER_FEATURES),
            "defense_f32": list(DEFENSE_FLOAT_FEATURES),
            "status_f32": list(STATUS_FLOAT_FEATURES),
            "status_i32": list(STATUS_INTEGER_FEATURES),
            "ability_f32": list(ABILITY_FLOAT_FEATURES),
            "ability_i32": list(ABILITY_INTEGER_FEATURES),
            "actor_world_f32": list(ACTOR_WORLD_FLOAT_FEATURES),
            "actor_world_mask": list(ACTOR_WORLD_MASK_FEATURES),
            "movement_state_f32": list(MOVEMENT_STATE_FEATURES),
            "movement_state_mask": list(MOVEMENT_STATE_FEATURES),
            "world_geometry.token_f32": [
                "token_kind_normalized",
                "token_provenance_normalized",
                "relative_position_xyz_normalized",
                "clearance_normalized",
                "semantic_flags_normalized",
                "dynamic_blocked",
                "edge_slots_mask_destination_cost_kind_flags",
                "collision_box_slots_mask_and_relative_bounds",
            ],
        },
        "status_projection": {
            "cycle_progress": (
                "periodic_elapsed_over_cooldown_else_zero_native_shared"
            ),
            "nonperiodic_has_cycled": "private_runtime_not_learner_observable",
        },
        "ability_projection": {
            "internal_queued_root": "not_actor_observable",
            "public_active_root": "interaction_queue_admitted_only",
            "public_elapsed_clock": "admitted_root_lifecycle_seconds",
            "public_legality": "computed_from_the_same_admitted_root_view",
            "native_active_row": "projected_as_already_queue_admitted",
        },
        "capacities": {
            "entities": entity_count,
            "resources": RESOURCE_COUNT,
            "statuses_per_entity": STATUS_CAPACITY,
            "abilities_per_entity": OBSERVATION_CAPACITY,
            "dodge_directions": DODGE_ACTION_COUNT,
            "door_candidates": INTERACTION_CAPACITY,
            "door_intents": DOOR_INTENT_COUNT,
            "movement_states": MOVEMENT_STATE_SIZE,
            "world_geometry_default": world_geometry_policy_config_manifest(
                DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG
            ),
            "world_geometry_capacity_basis": (
                "combined_collision_exception_and_traversal_tokens:"
                "world-token-capacity-v3"
            ),
        },
        "world_geometry_source": {
            "schema": WORLD_GEOMETRY_TOKEN_SCHEMA,
            "version": WORLD_GEOMETRY_TOKEN_VERSION,
            "sha256": world_geometry_token_contract_sha256(),
            "consumed_fields": list(WORLD_GEOMETRY_SOURCE_POLICY_FIELDS),
            "excluded_fields": (
                "all fields declared by the source as non_policy_diagnostics"
            ),
            "unavailable_or_invalid": "zero_row_fail_closed",
            "traversal_provenance": (
                "exact_local_support_clearance_projection_surrogate_provenance"
            ),
        },
        "movement_state_source": {
            "schema": MOVEMENT_STATE_SCHEMA,
            "version": MOVEMENT_STATE_VERSION,
            "sha256": movement_state_contract_sha256(),
            "role_projection": {
                "schema": ACTOR_WALK_MOVEMENT_PROJECTION_SCHEMA,
                "version": ACTOR_WALK_MOVEMENT_PROJECTION_VERSION,
                "sha256": actor_walk_movement_projection_sha256(),
                "role_id": "Kweebec_Razorleaf",
            },
            "consumed_fields": list(MOVEMENT_STATE_FEATURES),
            "unavailable_or_invalid": "zero_row_fail_closed",
            "update_boundary": "every_combat_microtick_after_actor_motion",
        },
        "world_capability_inputs": {
            "actor_world_state_available": _field(("B",), "bool"),
            "actor_controller_medium_available": _field(("B",), "bool"),
            "actor_submersion_available": _field(("B",), "bool"),
            "actor_drop_available": _field(("B",), "bool"),
            "actor_controller_in_fluid": _field(("B",), "bool"),
            "actor_feet_submerged": _field(("B",), "bool"),
            "actor_eyes_submerged": _field(("B",), "bool"),
            "actor_drop_support_found": _field(("B",), "bool"),
            "actor_drop_height": _field(("B",), "float32"),
            "line_of_sight": _field(("B", entity_count), "bool"),
            "line_of_sight_valid": _field(("B", entity_count), "bool"),
            "selector_line_of_sight": _field(
                ("B", entity_count),
                "bool",
            ),
            "selector_line_of_sight_valid": _field(
                ("B", entity_count),
                "bool",
            ),
            "muzzle_position": _field(("B", entity_count, 3), "float32"),
            "muzzle_yaw_degrees": _field(("B", entity_count), "float32"),
            "muzzle_pitch_degrees": _field(("B", entity_count), "float32"),
            "muzzle_valid": _field(("B", entity_count), "bool"),
            "clear_projectile_flight": _field(
                ("B", entity_count),
                "bool",
            ),
            "projectile_world_collision_available": _field(
                ("B", entity_count),
                "bool",
            ),
            "clear_force_path": _field(
                ("B", entity_count, OBSERVATION_CAPACITY),
                "bool",
            ),
            "dodge_corridor_clear": _field(
                ("B", entity_count, DODGE_ACTION_COUNT),
                "bool",
            ),
            "entity_only_area": _field(("B", entity_count), "bool"),
            "static_area_placement": _field(
                ("B", entity_count),
                "bool",
            ),
            "area_center": _field(("B", entity_count, 3), "float32"),
        },
        "action_contract": {
            "skill_id": _field(("B",), "int32"),
            "skill_id_domain": ("legacy_skills_0_8_or_intent_only_door_actions_9_11"),
            "ability_slot": _field(("B",), "int32"),
            "jump_held": _field(("B",), "bool"),
            "pitch_delta_degrees": _field(("B",), "float32"),
            "guard_held": _field(("B",), "bool"),
            "guard_support": ("masked_when_equipped_guard_stamina_value_is_zero"),
            "dodge_direction": _field(("B",), "int32"),
            "door_semantics": "intent_only_no_state_transition",
            "invalid_component": "masked_no_op_and_action_valid_false",
        },
        "failure_bits": {
            "base_v1_passthrough": [0, 1, 2, 3, 4, 5],
            "loadout": OBSERVATION_FAILURE_LOADOUT,
            "mechanics": OBSERVATION_FAILURE_MECHANICS,
            "arsenal": OBSERVATION_FAILURE_ARSENAL,
        },
        "lifecycle": {
            "loadout_and_rules": "episode_pinned",
            "world_features": "injected_already_selected",
            "world_geometry": (
                "actor_legal_current_frame_tokens_with_checkpoint_pinned_source"
            ),
            "overflow_or_runtime_failure": (
                "fully_mask_truncate_zero_reward_and_freeze"
            ),
        },
    }


def learner_observation_v3_contract_json(
    *,
    entity_count: int = ENTITY_COUNT,
) -> str:
    return json.dumps(
        learner_observation_v3_contract_manifest(
            entity_count=entity_count,
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def learner_observation_v3_contract_sha256(
    *,
    entity_count: int = ENTITY_COUNT,
) -> str:
    return (
        hashlib.sha256(
            learner_observation_v3_contract_json(
                entity_count=entity_count,
            ).encode("utf-8")
        )
        .hexdigest()
        .upper()
    )


def learner_observation_v3_actor_evidence_contract_manifest(
    *,
    entity_count: int = ENTITY_COUNT,
) -> dict[str, Any]:
    """Describe the actor-only seam feeding the shared v3 encoder body."""

    output = learner_observation_v3_contract_manifest(
        entity_count=entity_count,
    )
    output_fields = output["fields"]
    evidence_fields = {
        name: (
            _field(("B",), "bool") if name == "loadout_failure" else output_fields[name]
        )
        for name in LearnerObservationV3ActorEvidence._fields
    }
    return {
        "schema": "hytalerl_combat_observation_v3_actor_evidence",
        "version": 1,
        "batch_axis": "B",
        "fields": evidence_fields,
        "field_order": list(LearnerObservationV3ActorEvidence._fields),
        "output_contract_sha256": learner_observation_v3_contract_sha256(
            entity_count=entity_count,
        ),
        "projection": {
            "privileged_input": "ArsenalEnvironmentState",
            "privileged_state_scope": "projector_only",
            "encoder_input": "LearnerObservationV3ActorEvidence",
            "encoder_extra_inputs": [],
            "target_fields": "filtered_before_evidence",
            "world_fields": "actor_legal_selected_inputs_only",
            "invalid_rows": "fail_closed_in_shared_encoder",
        },
    }


def learner_observation_v3_actor_evidence_contract_json(
    *,
    entity_count: int = ENTITY_COUNT,
) -> str:
    return json.dumps(
        learner_observation_v3_actor_evidence_contract_manifest(
            entity_count=entity_count,
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def learner_observation_v3_actor_evidence_contract_sha256(
    *,
    entity_count: int = ENTITY_COUNT,
) -> str:
    return (
        hashlib.sha256(
            learner_observation_v3_actor_evidence_contract_json(
                entity_count=entity_count,
            ).encode("utf-8")
        )
        .hexdigest()
        .upper()
    )


def _field(shape: tuple[str | int, ...], dtype: str) -> dict[str, Any]:
    return {"shape": list(shape), "dtype": dtype}


def _validate_entity_count(entity_count: int) -> None:
    if (
        isinstance(entity_count, bool)
        or not isinstance(entity_count, int)
        or entity_count < 2
    ):
        raise ValueError("entity_count must be an integer of at least 2")
