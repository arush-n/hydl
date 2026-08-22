"""Array-only types for learner combat observation version 3."""

from __future__ import annotations

from typing import NamedTuple

import jax

from hytalegym.jax.combat.arsenal.schema.types import ArsenalCommands
from hytalegym.jax.combat.arsenal.schema.types import (
    ArsenalEnvironmentState,
    ArsenalInfo,
)
from hytalegym.jax.combat.observation.v1.schema.types import (
    DoorIntentRequest,
    LearnerCombatObservation,
)
from hytalegym.jax.combat.observation.v3.tokens.world import (
    WorldGeometryPolicyTokens,
)
from hytalegym.jax.combat.types import CombatInfo


Array = jax.Array


class LearnerCombatObservationV3(NamedTuple):
    schema_version: Array
    base: LearnerCombatObservation

    weapon_i32: Array
    resource_f32: Array
    resource_mask: Array
    defense_f32: Array

    status_f32: Array
    status_i32: Array
    status_flags: Array
    status_mask: Array

    ability_f32: Array
    ability_i32: Array
    ability_mask: Array
    ability_legal: Array

    actor_world_f32: Array
    actor_world_mask: Array
    movement_state_f32: Array
    movement_state_mask: Array
    world_geometry: WorldGeometryPolicyTokens

    skill_action_mask: Array
    ability_action_mask: Array
    jump_action_mask: Array
    guard_action_mask: Array
    dodge_action_mask: Array
    door_action_mask: Array

    valid: Array
    failure_bits: Array
    mechanics_failure_bits: Array
    arsenal_failure_bits: Array


class LearnerObservationV3ActorEvidence(NamedTuple):
    """Actor-facing inputs consumed by the shared v3 encoder body.

    Privileged simulator state is projected upstream. Target perception is
    already filtered in ``base`` and the dense actor adapter consumes only
    the agent entity slices of the structured resource/status/ability rows.
    """

    base: LearnerCombatObservation

    weapon_i32: Array
    resource_f32: Array
    resource_mask: Array
    defense_f32: Array

    status_f32: Array
    status_i32: Array
    status_flags: Array
    status_mask: Array

    ability_f32: Array
    ability_i32: Array
    ability_mask: Array
    ability_legal: Array

    actor_world_f32: Array
    actor_world_mask: Array
    movement_state_f32: Array
    movement_state_mask: Array
    world_geometry: WorldGeometryPolicyTokens

    skill_action_mask: Array
    jump_action_mask: Array
    guard_action_mask: Array
    dodge_action_mask: Array
    door_action_mask: Array

    loadout_failure: Array
    mechanics_failure_bits: Array
    arsenal_failure_bits: Array


class LearnerArsenalAction(NamedTuple):
    """Structured action with explicit target-slot factors.

    ``block_interaction_trigger`` always retains the independent authored
    Primary/Secondary choice. A block request exists only when
    ``block_candidate_index`` is nonnegative.

    ``gait`` and ``dodge_direction`` are decoded from one published locomotion
    head because they are physically exclusive, so at most one of them is ever
    non-idle. ``gait`` is idle/walk/run/sprint/sneak and scales the speed the
    walk controller is asked for; ``world_move_direction`` stays the compass
    choice it has always been. ``recipe_candidate_index`` is retained for the
    world-action executor but is no longer published, and decodes to -1.
    """

    skill_id: Array
    ability_slot: Array
    guard_held: Array
    dodge_direction: Array
    jump_held: Array
    gait: Array
    world_move_direction: Array
    #: Camera steer. The body's own steer is the field below; the two are
    #: separate levers, and travel rides the body.
    yaw_delta_degrees: Array
    body_yaw_delta_degrees: Array
    pitch_delta_degrees: Array
    #: Requested live hotbar slot, or -1 to leave the current one alone. A
    #: player presses a number key; this is the same edge.
    hotbar_slot: Array
    use_requested: Array
    block_interaction_trigger: Array
    recipe_candidate_index: Array
    block_candidate_index: Array


class LearnerArsenalDecode(NamedTuple):
    low_level_action: Array
    commands: ArsenalCommands
    door: DoorIntentRequest
    skill_legal: Array
    ability_legal: Array
    jump_legal: Array
    guard_legal: Array
    dodge_legal: Array
    valid: Array


class LearnerArsenalTransition(NamedTuple):
    state: ArsenalEnvironmentState
    observation: LearnerCombatObservationV3
    reward: Array
    terminated: Array
    truncated: Array
    action_valid: Array
    door: DoorIntentRequest
    combat_info: CombatInfo
    arsenal_info: ArsenalInfo
