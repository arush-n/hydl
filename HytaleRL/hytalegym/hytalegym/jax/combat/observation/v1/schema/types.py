"""Array-only types for learner-facing combat observations."""

from __future__ import annotations

from typing import NamedTuple

import jax


Array = jax.Array


class CombatSceneFeatures(NamedTuple):
    """Combat-owned, already-selected entity/projectile/hazard features."""

    entity_f32: Array
    entity_i32: Array
    entity_mask: Array
    entity_overflow: Array

    projectile_f32: Array
    projectile_i32: Array
    projectile_mask: Array
    projectile_overflow: Array

    hazard_f32: Array
    hazard_i32: Array
    hazard_mask: Array
    hazard_overflow: Array


class InjectedWorldFeatures(NamedTuple):
    """World-lane outputs consumed without tile selection or world mutation."""

    terrain_f32: Array
    terrain_semantic_id: Array
    terrain_flags: Array
    terrain_mask: Array
    terrain_overflow: Array

    traversal_f32: Array
    traversal_id: Array
    traversal_flags: Array
    traversal_mask: Array
    traversal_overflow: Array

    interaction_f32: Array
    interaction_object_id: Array
    interaction_is_door: Array
    interaction_door_intent_mask: Array
    interaction_mask: Array
    interaction_overflow: Array


class LearnerCombatObservation(NamedTuple):
    """Frozen structured observation; every leaf has batch first."""

    schema_version: Array

    self_f32: Array
    self_i32: Array
    target_f32: Array
    target_i32: Array
    target_mask: Array
    combat_f32: Array
    combat_i32: Array

    entity_f32: Array
    entity_i32: Array
    entity_mask: Array
    projectile_f32: Array
    projectile_i32: Array
    projectile_mask: Array
    hazard_f32: Array
    hazard_i32: Array
    hazard_mask: Array

    terrain_f32: Array
    terrain_semantic_id: Array
    terrain_flags: Array
    terrain_mask: Array
    traversal_f32: Array
    traversal_id: Array
    traversal_flags: Array
    traversal_mask: Array
    interaction_f32: Array
    interaction_object_id: Array
    interaction_is_door: Array
    interaction_door_intent_mask: Array
    interaction_mask: Array

    action_mask: Array
    valid: Array
    overflow_bits: Array


class DoorIntentRequest(NamedTuple):
    """Intent-only request; this never changes a door or combat state."""

    requested: Array
    accepted: Array
    intent: Array
    interaction_slot: Array
    object_id: Array


class LearnerActionDecode(NamedTuple):
    """Low-level combat action plus an optional external door request."""

    low_level_action: Array
    action_legal: Array
    door: DoorIntentRequest
