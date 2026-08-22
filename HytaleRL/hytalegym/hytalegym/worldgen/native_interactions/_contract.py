"""Contract body and hashing internal to native interaction evidence."""
from __future__ import annotations

import hashlib
import json


NATIVE_ITEM_INTERACTION_SCHEMA_V2 = (
    "hytalerl_native_item_interaction_evidence_v2"
)
NATIVE_ITEM_INTERACTION_SCHEMA_V3 = (
    "hytalerl_native_item_interaction_evidence_v3"
)
NATIVE_ITEM_INTERACTION_SCHEMA = NATIVE_ITEM_INTERACTION_SCHEMA_V2
NATIVE_ITEM_INTERACTION_VERSION = 2
NATIVE_ITEM_INTERACTION_VERSION_V3 = 3
NATIVE_ITEM_TRIGGER_CAPACITY = 25
NATIVE_ITEM_INTERACTION_CAPACITY = 256
NATIVE_ITEM_EDGE_CAPACITY = 512
NATIVE_ITEM_CHARGE_TIME_CAPACITY = 16
NATIVE_ITEM_BLOCK_CHANGE_CAPACITY = 256
NATIVE_ITEM_METADATA_CAPACITY = 16_384

INTERACTION_TYPE_NAMES = (
    "Primary",
    "Secondary",
    "Ability1",
    "Ability2",
    "Ability3",
    "Use",
    "Pick",
    "Pickup",
    "CollisionEnter",
    "CollisionLeave",
    "Collision",
    "EntityStatEffect",
    "SwapTo",
    "SwapFrom",
    "Death",
    "Wielding",
    "ProjectileSpawn",
    "ProjectileHit",
    "ProjectileMiss",
    "ProjectileBounce",
    "Held",
    "HeldOffhand",
    "Equipped",
    "Dodge",
    "GameModeSwap",
)


def _sha256(value: str) -> str:
    if (
        len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError("bridge_sha256 must be a SHA-256")
    return value


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _native_item_interaction_contract(
    *,
    schema: str,
    version: int,
    resolved_subclass_payloads: str,
) -> dict[str, object]:
    return {
        "schema": schema,
        "version": version,
        "server_version": "0.5.7",
        "trigger_vocabulary": list(INTERACTION_TYPE_NAMES),
        "fixed_capacity": {
            "triggers": NATIVE_ITEM_TRIGGER_CAPACITY,
            "interactions_per_trigger": NATIVE_ITEM_INTERACTION_CAPACITY,
            "edges_per_trigger": NATIVE_ITEM_EDGE_CAPACITY,
            "charge_times": NATIVE_ITEM_CHARGE_TIME_CAPACITY,
            "block_changes_per_node": NATIVE_ITEM_BLOCK_CHANGE_CAPACITY,
            "item_metadata_characters": NATIVE_ITEM_METADATA_CAPACITY,
        },
        "resolved_evidence": [
            "active_item_after_main_offhand_tool_priority",
            "request_selected_Equipped_armor_slot",
            "native_armor_slot_capacity_and_availability",
            "root_presence",
            "effective_Adventure_and_Creative_cooldowns",
            "root_arbitration_masks_and_bypass_indices",
            "reachable_interaction_nodes_and_edges",
            (
                "runTime_next_failed_useLatestTarget_cancelOnItemChange_"
                "harvest"
            ),
            (
                "horizontalSpeedMultiplier_startDelay_"
                "waitForAnimationToFinish"
            ),
            (
                "raw_npc_disableAll_plus_seven_bit_player_input_"
                "movement_lock_mask_with_presence"
            ),
            resolved_subclass_payloads,
        ],
        "provenance": "native_runtime_resolved_asset_graph",
        "not_certified": [
            "authenticated_player_request_acceptance",
            "target_legality",
            "inventory_consumption",
            "chain_execution_outcome",
        ],
        "fail_closed": [
            "bridge_identity_mismatch",
            "missing_root_asset",
            "graph_or_charge_capacity_overflow",
            "partial_or_unordered_graph",
            "unknown_or_oversized_subclass_payload",
        ],
    }
