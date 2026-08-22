"""Frozen learner-facing combat observation contract."""

from __future__ import annotations

from hytalegym.jax.combat.skills import SKILL_COUNT
from hytalegym.jax.combat.types import ACTIVE_OBSERVATION_SIZE


LEARNER_OBSERVATION_SCHEMA = "hytalerl_combat_observation_v1"
LEARNER_OBSERVATION_VERSION = 1
WORLD_FEATURE_REQUIREMENTS_REVISION = 1
OBSERVATION_NORMALIZATION_REVISION = 1
TARGET_EVIDENCE_POLICY_REVISION = 1
FROZEN_MELEE_SKILL_COUNT = 9

NEARBY_ENTITY_CAPACITY = 16
PROJECTILE_CAPACITY = 16
HAZARD_CAPACITY = 8
TERRAIN_TOKEN_CAPACITY = 48
TRAVERSAL_TOKEN_CAPACITY = 16
INTERACTION_CAPACITY = 8

NEARBY_ENTITY_RADIUS_BLOCKS = 24.0
PROJECTILE_RADIUS_BLOCKS = 32.0
HAZARD_RADIUS_BLOCKS = 24.0
TERRAIN_RADIUS_BLOCKS = 24.0
TRAVERSAL_RADIUS_BLOCKS = 48.0
INTERACTION_RADIUS_BLOCKS = 6.0

SELF_FLOAT_FEATURES = (
    "forward_velocity",
    "right_velocity",
    "vertical_velocity",
    "health_fraction",
    "yaw_sin",
    "yaw_cos",
    "pitch",
    "grounded",
    "attack_executing",
    "attack_cooldown",
    "knockback_control_lock",
    "damage_recovery",
    "applied_vertical_velocity",
    "fall_speed",
    "motion_delta",
    "alive",
)
SELF_INTEGER_FEATURES = (
    "tick_count",
    "attack_sequence_index",
    "pending_attack_index",
    "motion_timing_profile",
)

COMBAT_FLOAT_FEATURES = (
    "agent_forward_velocity",
    "agent_right_velocity",
    "agent_vertical_velocity",
    "agent_health_fraction",
    "agent_yaw_sin",
    "agent_yaw_cos",
    "visible_target_forward",
    "visible_target_right",
    "visible_target_planar_distance",
    "visible_target_health_fraction",
    "target_visible",
    "agent_attack_executing",
    "target_phase_idle",
    "target_phase_windup",
    "target_phase_sweep",
    "target_phase_recovery",
    "target_phase_cooldown",
    "target_attack_progress",
    "target_facing_error",
    "target_forward_velocity",
    "target_right_velocity",
    "target_attack_index",
    "target_head_facing_error",
    "target_head_pitch",
)

TARGET_FLOAT_FEATURES = (
    "relative_forward",
    "relative_right",
    "relative_up",
    "relative_velocity_forward",
    "relative_velocity_right",
    "relative_velocity_up",
    "planar_distance",
    "distance",
    "health_fraction",
    "bearing_sin",
    "bearing_cos",
    "facing_error",
    "head_facing_error",
    "head_pitch",
    "attack_progress",
    "visible",
)
TARGET_INTEGER_FEATURES = (
    "attack_phase",
    "attack_index",
    "attack_elapsed_ticks",
    "attack_queued",
)

COMBAT_INTEGER_FEATURES = (
    "target_attack_phase",
    "target_attack_index",
    "target_attack_elapsed_ticks",
    "agent_attack_sequence_index",
    "target_attack_sequence_index",
    "pending_agent_attack_index",
    "tick_count",
    "motion_timing_profile",
)

ENTITY_FLOAT_FEATURES = (
    "relative_forward",
    "relative_right",
    "relative_up",
    "relative_velocity_forward",
    "relative_velocity_right",
    "relative_velocity_up",
    "distance",
    "health_fraction",
    "yaw_sin",
    "yaw_cos",
    "visible",
    "attack_progress",
    "facing_error",
    "head_facing_error",
    "hostile",
    "alive",
)
ENTITY_INTEGER_FEATURES = (
    "entity_id",
    "entity_kind",
    "attack_phase",
    "attack_index",
)

PROJECTILE_FLOAT_FEATURES = (
    "relative_x",
    "relative_y",
    "relative_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "distance",
    "age_fraction",
    "lifetime_remaining",
    "damage_fraction",
    "radius_fraction",
    "hostile",
)
PROJECTILE_INTEGER_FEATURES = (
    "projectile_kind",
    "owner_entity_id",
    "flags",
)

HAZARD_FLOAT_FEATURES = (
    "relative_x",
    "relative_y",
    "relative_z",
    "half_extent_x",
    "half_extent_y",
    "half_extent_z",
    "distance",
    "intensity",
    "duration_remaining",
    "damage_rate",
    "activation",
    "hostile",
)
HAZARD_INTEGER_FEATURES = (
    "hazard_kind",
    "owner_entity_id",
    "flags",
)

TERRAIN_FLOAT_FEATURES = (
    "relative_x",
    "relative_y",
    "relative_z",
    "distance",
    "traversability",
    "movement_cost",
    "damage_cost",
    "interaction_relevance",
)
TRAVERSAL_FLOAT_FEATURES = (
    "relative_waypoint_x",
    "relative_waypoint_y",
    "relative_waypoint_z",
    "path_cost",
    "travel_time",
    "arrival_radius",
    "route_progress",
    "confidence",
)
INTERACTION_FLOAT_FEATURES = (
    "relative_x",
    "relative_y",
    "relative_z",
    "distance",
    "facing_alignment",
    "reach_fraction",
)

DOOR_INTENT_NONE = 0
DOOR_INTENT_OPEN = 1
DOOR_INTENT_CLOSE = 2
DOOR_INTENT_USE = 3
DOOR_INTENT_COUNT = 3
DOOR_INTENT_NAMES = ("open", "close", "use")

if SKILL_COUNT != FROZEN_MELEE_SKILL_COUNT:
    raise RuntimeError(
        "learner observation v1 requires exactly nine legacy melee skills"
    )

ACTION_DOOR_OPEN = 9
ACTION_DOOR_CLOSE = 10
ACTION_DOOR_USE = 11
LEARNER_ACTION_COUNT = 12
LEARNER_ACTION_NAMES = (
    "idle",
    "face_target",
    "approach",
    "retreat",
    "strafe_left",
    "strafe_right",
    "attack",
    "approach_attack",
    "retreat_attack",
    "door_open",
    "door_close",
    "door_use",
)

if len(LEARNER_ACTION_NAMES) != LEARNER_ACTION_COUNT:
    raise RuntimeError("learner observation v1 action-name contract drift")

ENTITY_KIND_NPC = 1

OVERFLOW_ENTITY = 1 << 0
OVERFLOW_PROJECTILE = 1 << 1
OVERFLOW_HAZARD = 1 << 2
OVERFLOW_TERRAIN = 1 << 3
OVERFLOW_TRAVERSAL = 1 << 4
OVERFLOW_INTERACTION = 1 << 5

SELF_FLOAT_SIZE = len(SELF_FLOAT_FEATURES)
SELF_INTEGER_SIZE = len(SELF_INTEGER_FEATURES)
COMBAT_FLOAT_SIZE = len(COMBAT_FLOAT_FEATURES)
TARGET_FLOAT_SIZE = len(TARGET_FLOAT_FEATURES)
TARGET_INTEGER_SIZE = len(TARGET_INTEGER_FEATURES)
COMBAT_INTEGER_SIZE = len(COMBAT_INTEGER_FEATURES)
ENTITY_FLOAT_SIZE = len(ENTITY_FLOAT_FEATURES)
ENTITY_INTEGER_SIZE = len(ENTITY_INTEGER_FEATURES)
PROJECTILE_FLOAT_SIZE = len(PROJECTILE_FLOAT_FEATURES)
PROJECTILE_INTEGER_SIZE = len(PROJECTILE_INTEGER_FEATURES)
HAZARD_FLOAT_SIZE = len(HAZARD_FLOAT_FEATURES)
HAZARD_INTEGER_SIZE = len(HAZARD_INTEGER_FEATURES)
TERRAIN_FLOAT_SIZE = len(TERRAIN_FLOAT_FEATURES)
TRAVERSAL_FLOAT_SIZE = len(TRAVERSAL_FLOAT_FEATURES)
INTERACTION_FLOAT_SIZE = len(INTERACTION_FLOAT_FEATURES)

if COMBAT_FLOAT_SIZE != ACTIVE_OBSERVATION_SIZE:
    raise RuntimeError(
        "learner observation v1 requires the frozen 24-value combat prefix"
    )
