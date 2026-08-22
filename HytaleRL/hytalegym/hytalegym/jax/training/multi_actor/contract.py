"""Stable contract for policy-controlled actors sharing one arena."""

from __future__ import annotations

import hashlib
import json


MULTI_ACTOR_ASSIGNMENT_SCHEMA = "hytalerl_multi_actor_assignment_v1"
MULTI_ACTOR_ASSIGNMENT_VERSION = 1


def multi_actor_assignment_contract_manifest() -> dict[str, object]:
    """Describe the actor-slot contract without moving existing policy ABIs."""

    return {
        "schema": MULTI_ACTOR_ASSIGNMENT_SCHEMA,
        "version": MULTI_ACTOR_ASSIGNMENT_VERSION,
        "shape": {
            "actor_index": "int32[B,P]",
            "policy_id": "int32[B,P]",
            "active": "bool[B,P]",
            "trainable": "bool[B,P]",
        },
        "semantics": {
            "B": "independent shared arenas",
            "P": "bounded policy-actor slots; independent of entity capacity",
            "actor_index": "entity row controlled by this slot; -1 when inactive",
            "policy_id": "policy/checkpoint assignment; -1 when inactive",
            "active": "slot owns actions and an egocentric observation row",
            "trainable": "slot contributes gradients; implies active",
            "uniqueness": "an entity may be owned by at most one slot per arena",
            "control_overlay": (
                "valid active slots replace ability/defense commands only on "
                "their owned entity; unowned scripted rows are preserved"
            ),
            "locomotion_staging": (
                "low-level controls are scattered as float32[B,N,8] and use "
                "entity-indexed Walk state in the shared-arena runtime"
            ),
            "recurrent_carry": "float32[B,P,R], isolated per actor slot",
            "policy_dispatch": (
                "policy_id selects one row from a reset-pinned parameter bank; "
                "inactive and out-of-bank slots fail closed"
            ),
            "single_actor_dispatch": (
                "P=1 batches use one shared policy_id and preserve the shipped "
                "batched policy calculation bit-exactly"
            ),
            "observation_consumer": (
                "flat shared arenas project each owned actor through an "
                "actor-first legal view and the unchanged v3 encoder; Region "
                "world/action producers remain an upstream actor-major gate"
            ),
            "rollout_storage": (
                "behavior facts retain [T,B,P,...] axes, policy_id, trainable "
                "ownership, and independent recurrent carry"
            ),
            "shared_policy_update": (
                "any number of trainable actor slots may contribute to one "
                "reset-pinned policy_id; all other policy-bank rows remain frozen"
            ),
        },
        "compatibility": {
            "single_actor": "P=1, actor_index=0, policy_id=0",
            "existing_policy_contracts_moved": False,
            "runtime_integration": "flat_shared_policy_training_live",
        },
    }


def multi_actor_assignment_contract_sha256() -> str:
    """Return the canonical identity of the standalone assignment contract."""

    payload = json.dumps(
        multi_actor_assignment_contract_manifest(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


__all__ = [
    "MULTI_ACTOR_ASSIGNMENT_SCHEMA",
    "MULTI_ACTOR_ASSIGNMENT_VERSION",
    "multi_actor_assignment_contract_manifest",
    "multi_actor_assignment_contract_sha256",
]
