"""Versioned Combat-owned half of native block interactions."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


BLOCK_INTERACTION_SCHEMA = "hytalerl_combat_block_interactions_v6"
BLOCK_INTERACTION_VERSION = 6

BLOCK_INTERACTION_NONE = 0
BLOCK_INTERACTION_BREAK = 1
BLOCK_INTERACTION_PLACE = 2

BLOCK_INTERACTION_IDLE = 0
BLOCK_INTERACTION_RUNNING = 1
BLOCK_INTERACTION_SUCCEEDED = 2
BLOCK_INTERACTION_FAILED = 3
BLOCK_INTERACTION_CANCELLED = 4

NO_INTERACTION_PROGRAM = -1
NO_BLOCK_ID = -1

BLOCK_HEALTH_EPSILON = float(np.spacing(np.float32(1.0)))

BLOCK_INTERACTION_FAILURE_INVALID_STATE = 1 << 0
BLOCK_INTERACTION_FAILURE_INVALID_REQUEST = 1 << 1
BLOCK_INTERACTION_FAILURE_WORLD_UNSUPPORTED = 1 << 2

INTERACTION_MOVEMENT_DISABLE_FORWARD = 1 << 0
INTERACTION_MOVEMENT_DISABLE_BACKWARD = 1 << 1
INTERACTION_MOVEMENT_DISABLE_LEFT = 1 << 2
INTERACTION_MOVEMENT_DISABLE_RIGHT = 1 << 3
INTERACTION_MOVEMENT_DISABLE_SPRINT = 1 << 4
INTERACTION_MOVEMENT_DISABLE_JUMP = 1 << 5
INTERACTION_MOVEMENT_DISABLE_CROUCH = 1 << 6
INTERACTION_MOVEMENT_DIRECTION_MASK = (
    INTERACTION_MOVEMENT_DISABLE_FORWARD
    | INTERACTION_MOVEMENT_DISABLE_BACKWARD
    | INTERACTION_MOVEMENT_DISABLE_LEFT
    | INTERACTION_MOVEMENT_DISABLE_RIGHT
)
INTERACTION_MOVEMENT_ALL_MASK = (1 << 7) - 1


def block_interaction_contract_manifest() -> dict[str, Any]:
    """Return the canonical break/place action-half contract."""

    return {
        "schema": BLOCK_INTERACTION_SCHEMA,
        "version": BLOCK_INTERACTION_VERSION,
        "kinds": {
            "none": BLOCK_INTERACTION_NONE,
            "break": BLOCK_INTERACTION_BREAK,
            "place": BLOCK_INTERACTION_PLACE,
        },
        "scheduler": {
            "first_run_side_effect": "before_runtime_wait",
            "duration": ("max_run_time_and_animation_duration_only_when_wait_enabled"),
            "start_delay": (
                "transported_client_effect_timing_not_server_completion_time"
            ),
            "completion": "elapsed_greater_than_or_equal_to_duration",
            "effect_count": "at_most_once_per_started_chain",
        },
        "movement": {
            "bit_order": [
                "disable_forward",
                "disable_backward",
                "disable_left",
                "disable_right",
                "disable_sprint",
                "disable_jump",
                "disable_crouch",
            ],
            "npc_disable_all": ("preserved_separately_and_projects_to_all_seven_locks"),
            "missing_effects_while_active": "fail_closed",
            "horizontal_speed": "multiply_active_local_translation",
            "current_policy_projection": {
                "world_compass": ("neutral_only_when_any_local_direction_is_disabled"),
                "jump": "disable_on_jump_bit",
                "sprint": "carried_pending_policy_head",
                "crouch": "carried_pending_policy_head",
            },
        },
        "native_item_compiler": {
            "supported_classes": [
                "BreakBlockInteraction",
                "PlaceBlockInteraction",
            ],
            "accepted_evidence_versions": [2, 3],
            "wait_for_data_from": "Client_only_for_supported_classes",
            "target_selection": ("authored_use_latest_target_preserved_host_side"),
            "break_payload": {
                "v2": "absent_backward_compatible",
                "v3": "typed_tool_id_and_match_tool_preserved_host_side",
            },
            "place_payload": (
                "optional_block_type_override_remove_item_and_drag_fields_preserved"
            ),
            "compiled_contract_identity": "matches_evidence_version",
            "unsupported_class_or_payload": "fail_closed",
        },
        "branches": {
            "finished": "dispatch_next_child",
            "failed": "dispatch_failed_child",
            "item_changed": "cancel_chain_without_failed_child",
        },
        "break": {
            "health": "normalized_one_to_zero",
            "destroyed": {
                "less_than_zero": True,
                "close_to_zero_epsilon": BLOCK_HEALTH_EPSILON,
            },
            "adventure": "one_resolved_damage_on_first_run",
            "creative": "immediate_remove_request",
            "normal_drop": "world_owned_drop_request",
        },
        "harvest": {
            "preconditions": [
                "block_gathering_allowed",
                "target_harvestable",
            ],
            "pickup": ("quantity_aware_inventory_then_remainder_world_drop"),
            "drop_axis": "runtime_static_derived_from_resolved_drop_arrays",
        },
        "place": {
            "block_identity": "authored_override_or_held_item_block_key",
            "applied_identity": (
                "exact_world_before_after_geometry_semantic_key"
            ),
            "runtime_block_id": "optional_process_local_diagnostic",
            "height": "zero_inclusive_to_320_exclusive",
            "reach": "world_resolves_native_noncreative_squared_49_gate",
            "consume": "exact_held_slot_only_after_world_applies_mutation",
        },
        "world_boundary": {
            "mutation": "world_owned_acknowledgement_required_fail_closed",
            "combat_outputs": [
                "damage_request",
                "remove_request",
                "place_request",
                "normal_world_drops",
                "harvest_pickup_remainders",
            ],
            "not_claimed": [
                "mutable_region_geometry",
                "block_asset_drop_resolution",
                "native_bridge_action_support",
                "policy_head_reachability",
                "root_interaction_rules_concurrency",
                "allow_skip_on_click",
                "horizontal_speed_application",
                "break_tool_match_executor_consumption",
            ],
        },
    }


def block_interaction_contract_json() -> str:
    return json.dumps(
        block_interaction_contract_manifest(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def block_interaction_contract_sha256() -> str:
    return (
        hashlib.sha256(block_interaction_contract_json().encode("utf-8"))
        .hexdigest()
        .upper()
    )


__all__ = [
    "BLOCK_HEALTH_EPSILON",
    "BLOCK_INTERACTION_BREAK",
    "BLOCK_INTERACTION_CANCELLED",
    "BLOCK_INTERACTION_FAILED",
    "BLOCK_INTERACTION_FAILURE_INVALID_REQUEST",
    "BLOCK_INTERACTION_FAILURE_INVALID_STATE",
    "BLOCK_INTERACTION_FAILURE_WORLD_UNSUPPORTED",
    "BLOCK_INTERACTION_IDLE",
    "BLOCK_INTERACTION_NONE",
    "BLOCK_INTERACTION_PLACE",
    "BLOCK_INTERACTION_RUNNING",
    "BLOCK_INTERACTION_SCHEMA",
    "BLOCK_INTERACTION_SUCCEEDED",
    "BLOCK_INTERACTION_VERSION",
    "INTERACTION_MOVEMENT_ALL_MASK",
    "INTERACTION_MOVEMENT_DIRECTION_MASK",
    "INTERACTION_MOVEMENT_DISABLE_BACKWARD",
    "INTERACTION_MOVEMENT_DISABLE_CROUCH",
    "INTERACTION_MOVEMENT_DISABLE_FORWARD",
    "INTERACTION_MOVEMENT_DISABLE_JUMP",
    "INTERACTION_MOVEMENT_DISABLE_LEFT",
    "INTERACTION_MOVEMENT_DISABLE_RIGHT",
    "INTERACTION_MOVEMENT_DISABLE_SPRINT",
    "NO_BLOCK_ID",
    "NO_INTERACTION_PROGRAM",
    "block_interaction_contract_json",
    "block_interaction_contract_manifest",
    "block_interaction_contract_sha256",
]
