"""Canonical host manifest for the frozen learner observation contract."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from hytalegym.jax.combat.observation.v1.schema.contract import (
    ACTION_DOOR_CLOSE,
    ACTION_DOOR_OPEN,
    ACTION_DOOR_USE,
    COMBAT_FLOAT_FEATURES,
    COMBAT_FLOAT_SIZE,
    COMBAT_INTEGER_FEATURES,
    COMBAT_INTEGER_SIZE,
    DOOR_INTENT_NAMES,
    ENTITY_FLOAT_FEATURES,
    ENTITY_FLOAT_SIZE,
    ENTITY_INTEGER_FEATURES,
    ENTITY_INTEGER_SIZE,
    HAZARD_CAPACITY,
    HAZARD_FLOAT_FEATURES,
    HAZARD_FLOAT_SIZE,
    HAZARD_INTEGER_FEATURES,
    HAZARD_INTEGER_SIZE,
    HAZARD_RADIUS_BLOCKS,
    INTERACTION_CAPACITY,
    INTERACTION_FLOAT_FEATURES,
    INTERACTION_FLOAT_SIZE,
    INTERACTION_RADIUS_BLOCKS,
    LEARNER_ACTION_COUNT,
    LEARNER_ACTION_NAMES,
    LEARNER_OBSERVATION_SCHEMA,
    LEARNER_OBSERVATION_VERSION,
    NEARBY_ENTITY_CAPACITY,
    NEARBY_ENTITY_RADIUS_BLOCKS,
    OBSERVATION_NORMALIZATION_REVISION,
    OVERFLOW_ENTITY,
    OVERFLOW_HAZARD,
    OVERFLOW_INTERACTION,
    OVERFLOW_PROJECTILE,
    OVERFLOW_TERRAIN,
    OVERFLOW_TRAVERSAL,
    PROJECTILE_CAPACITY,
    PROJECTILE_FLOAT_FEATURES,
    PROJECTILE_FLOAT_SIZE,
    PROJECTILE_INTEGER_FEATURES,
    PROJECTILE_INTEGER_SIZE,
    PROJECTILE_RADIUS_BLOCKS,
    SELF_FLOAT_FEATURES,
    SELF_FLOAT_SIZE,
    SELF_INTEGER_FEATURES,
    SELF_INTEGER_SIZE,
    TARGET_FLOAT_FEATURES,
    TARGET_FLOAT_SIZE,
    TARGET_INTEGER_FEATURES,
    TARGET_INTEGER_SIZE,
    TARGET_EVIDENCE_POLICY_REVISION,
    TERRAIN_FLOAT_FEATURES,
    TERRAIN_FLOAT_SIZE,
    TERRAIN_RADIUS_BLOCKS,
    TERRAIN_TOKEN_CAPACITY,
    TRAVERSAL_FLOAT_FEATURES,
    TRAVERSAL_FLOAT_SIZE,
    TRAVERSAL_RADIUS_BLOCKS,
    TRAVERSAL_TOKEN_CAPACITY,
    WORLD_FEATURE_REQUIREMENTS_REVISION,
)
from hytalegym.jax.combat.observation.v1.schema.types import (
    CombatSceneFeatures,
    InjectedWorldFeatures,
    LearnerCombatObservation,
)


def learner_observation_contract_manifest() -> dict[str, Any]:
    """Return JSON-compatible metadata used to gate checkpoints/adapters."""

    fields = {
        "schema_version": _field(("B",), "int32"),
        "self_f32": _field(("B", SELF_FLOAT_SIZE), "float32"),
        "self_i32": _field(("B", SELF_INTEGER_SIZE), "int32"),
        "target_f32": _field(("B", TARGET_FLOAT_SIZE), "float32"),
        "target_i32": _field(("B", TARGET_INTEGER_SIZE), "int32"),
        "target_mask": _field(("B",), "bool"),
        "combat_f32": _field(("B", COMBAT_FLOAT_SIZE), "float32"),
        "combat_i32": _field(("B", COMBAT_INTEGER_SIZE), "int32"),
        "entity_f32": _field(
            ("B", NEARBY_ENTITY_CAPACITY, ENTITY_FLOAT_SIZE),
            "float32",
        ),
        "entity_i32": _field(
            ("B", NEARBY_ENTITY_CAPACITY, ENTITY_INTEGER_SIZE),
            "int32",
        ),
        "entity_mask": _field(("B", NEARBY_ENTITY_CAPACITY), "bool"),
        "projectile_f32": _field(
            ("B", PROJECTILE_CAPACITY, PROJECTILE_FLOAT_SIZE),
            "float32",
        ),
        "projectile_i32": _field(
            ("B", PROJECTILE_CAPACITY, PROJECTILE_INTEGER_SIZE),
            "int32",
        ),
        "projectile_mask": _field(("B", PROJECTILE_CAPACITY), "bool"),
        "hazard_f32": _field(
            ("B", HAZARD_CAPACITY, HAZARD_FLOAT_SIZE),
            "float32",
        ),
        "hazard_i32": _field(
            ("B", HAZARD_CAPACITY, HAZARD_INTEGER_SIZE),
            "int32",
        ),
        "hazard_mask": _field(("B", HAZARD_CAPACITY), "bool"),
        "terrain_f32": _field(
            ("B", TERRAIN_TOKEN_CAPACITY, TERRAIN_FLOAT_SIZE),
            "float32",
        ),
        "terrain_semantic_id": _field(
            ("B", TERRAIN_TOKEN_CAPACITY),
            "int32",
        ),
        "terrain_flags": _field(
            ("B", TERRAIN_TOKEN_CAPACITY),
            "uint32",
        ),
        "terrain_mask": _field(("B", TERRAIN_TOKEN_CAPACITY), "bool"),
        "traversal_f32": _field(
            ("B", TRAVERSAL_TOKEN_CAPACITY, TRAVERSAL_FLOAT_SIZE),
            "float32",
        ),
        "traversal_id": _field(
            ("B", TRAVERSAL_TOKEN_CAPACITY),
            "int32",
        ),
        "traversal_flags": _field(
            ("B", TRAVERSAL_TOKEN_CAPACITY),
            "uint32",
        ),
        "traversal_mask": _field(
            ("B", TRAVERSAL_TOKEN_CAPACITY),
            "bool",
        ),
        "interaction_f32": _field(
            ("B", INTERACTION_CAPACITY, INTERACTION_FLOAT_SIZE),
            "float32",
        ),
        "interaction_object_id": _field(
            ("B", INTERACTION_CAPACITY),
            "int32",
        ),
        "interaction_is_door": _field(
            ("B", INTERACTION_CAPACITY),
            "bool",
        ),
        "interaction_door_intent_mask": _field(
            ("B", INTERACTION_CAPACITY, len(DOOR_INTENT_NAMES)),
            "bool",
        ),
        "interaction_mask": _field(("B", INTERACTION_CAPACITY), "bool"),
        "action_mask": _field(("B", LEARNER_ACTION_COUNT), "bool"),
        "valid": _field(("B",), "bool"),
        "overflow_bits": _field(("B",), "uint32"),
    }
    input_fields = {
        "scene.entity_f32": fields["entity_f32"],
        "scene.entity_i32": fields["entity_i32"],
        "scene.entity_mask": fields["entity_mask"],
        "scene.entity_overflow": _field(("B",), "bool"),
        "scene.projectile_f32": fields["projectile_f32"],
        "scene.projectile_i32": fields["projectile_i32"],
        "scene.projectile_mask": fields["projectile_mask"],
        "scene.projectile_overflow": _field(("B",), "bool"),
        "scene.hazard_f32": fields["hazard_f32"],
        "scene.hazard_i32": fields["hazard_i32"],
        "scene.hazard_mask": fields["hazard_mask"],
        "scene.hazard_overflow": _field(("B",), "bool"),
        "world.terrain_f32": fields["terrain_f32"],
        "world.terrain_semantic_id": fields["terrain_semantic_id"],
        "world.terrain_flags": fields["terrain_flags"],
        "world.terrain_mask": fields["terrain_mask"],
        "world.terrain_overflow": _field(("B",), "bool"),
        "world.traversal_f32": fields["traversal_f32"],
        "world.traversal_id": fields["traversal_id"],
        "world.traversal_flags": fields["traversal_flags"],
        "world.traversal_mask": fields["traversal_mask"],
        "world.traversal_overflow": _field(("B",), "bool"),
        "world.interaction_f32": fields["interaction_f32"],
        "world.interaction_object_id": fields["interaction_object_id"],
        "world.interaction_is_door": fields["interaction_is_door"],
        "world.interaction_door_intent_mask": fields["interaction_door_intent_mask"],
        "world.interaction_mask": fields["interaction_mask"],
        "world.interaction_overflow": _field(("B",), "bool"),
    }
    return {
        "schema": LEARNER_OBSERVATION_SCHEMA,
        "version": LEARNER_OBSERVATION_VERSION,
        "normalization_revision": OBSERVATION_NORMALIZATION_REVISION,
        "target_evidence": {
            "policy_revision": TARGET_EVIDENCE_POLICY_REVISION,
            "predicate": "combat_f32.target_visible",
            "target_mask": "perceptible",
            "hidden_encoding": "all_target_fields_zero",
            "nearby_entity_slot_zero": "perceptible_and_within_radius",
        },
        "world_requirements_revision": WORLD_FEATURE_REQUIREMENTS_REVISION,
        "batch_axis": "B",
        "fields": fields,
        "field_order": list(LearnerCombatObservation._fields),
        "input_fields": input_fields,
        "input_field_order": {
            "scene": list(CombatSceneFeatures._fields),
            "world": list(InjectedWorldFeatures._fields),
        },
        "feature_order": {
            "self_f32": list(SELF_FLOAT_FEATURES),
            "self_i32": list(SELF_INTEGER_FEATURES),
            "target_f32": list(TARGET_FLOAT_FEATURES),
            "target_i32": list(TARGET_INTEGER_FEATURES),
            "combat_f32": list(COMBAT_FLOAT_FEATURES),
            "combat_i32": list(COMBAT_INTEGER_FEATURES),
            "entity_f32": list(ENTITY_FLOAT_FEATURES),
            "entity_i32": list(ENTITY_INTEGER_FEATURES),
            "projectile_f32": list(PROJECTILE_FLOAT_FEATURES),
            "projectile_i32": list(PROJECTILE_INTEGER_FEATURES),
            "hazard_f32": list(HAZARD_FLOAT_FEATURES),
            "hazard_i32": list(HAZARD_INTEGER_FEATURES),
            "terrain_f32": list(TERRAIN_FLOAT_FEATURES),
            "traversal_f32": list(TRAVERSAL_FLOAT_FEATURES),
            "interaction_f32": list(INTERACTION_FLOAT_FEATURES),
        },
        "capacities": {
            "entities": NEARBY_ENTITY_CAPACITY,
            "projectiles": PROJECTILE_CAPACITY,
            "hazards": HAZARD_CAPACITY,
            "terrain": TERRAIN_TOKEN_CAPACITY,
            "traversal": TRAVERSAL_TOKEN_CAPACITY,
            "interactions": INTERACTION_CAPACITY,
        },
        "radius_blocks": {
            "entities": NEARBY_ENTITY_RADIUS_BLOCKS,
            "projectiles": PROJECTILE_RADIUS_BLOCKS,
            "hazards": HAZARD_RADIUS_BLOCKS,
            "terrain": TERRAIN_RADIUS_BLOCKS,
            "traversal": TRAVERSAL_RADIUS_BLOCKS,
            "interactions": INTERACTION_RADIUS_BLOCKS,
        },
        "actions": {
            "count": LEARNER_ACTION_COUNT,
            "order": list(LEARNER_ACTION_NAMES),
            "door_open": ACTION_DOOR_OPEN,
            "door_close": ACTION_DOOR_CLOSE,
            "door_use": ACTION_DOOR_USE,
        },
        "overflow_bits": {
            "entities": OVERFLOW_ENTITY,
            "projectiles": OVERFLOW_PROJECTILE,
            "hazards": OVERFLOW_HAZARD,
            "terrain": OVERFLOW_TERRAIN,
            "traversal": OVERFLOW_TRAVERSAL,
            "interactions": OVERFLOW_INTERACTION,
        },
    }


def learner_observation_contract_json() -> str:
    """Return the canonical serialization used for the SHA-256 identity."""

    return json.dumps(
        learner_observation_contract_manifest(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def learner_observation_contract_sha256() -> str:
    """Return the semantic learner-contract identity."""

    encoded = learner_observation_contract_json().encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _field(shape: tuple[str | int, ...], dtype: str) -> dict[str, Any]:
    return {"shape": list(shape), "dtype": dtype}
