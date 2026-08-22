"""Frozen learner observation contract for the data-driven combat arsenal."""

from __future__ import annotations

from hytalegym.jax.combat.arsenal.schema.contract import OBSERVATION_CAPACITY
from hytalegym.jax.combat.mechanics import (
    RESOURCE_COUNT,
    STATUS_CAPACITY,
)
from hytalegym.jax.combat.observation.v1.schema.contract import (
    DOOR_INTENT_COUNT,
    INTERACTION_CAPACITY,
)
from hytalegym.jax.combat.skills import SKILL_COUNT


LEARNER_OBSERVATION_V3_SCHEMA = "hytalerl_combat_observation_v3"
LEARNER_OBSERVATION_V3_VERSION = 10

WEAPON_INTEGER_FEATURES = (
    "weapon_id",
    "weapon_family",
    "active_ability_slot",
)
DEFENSE_FLOAT_FEATURES = (
    "guard_active",
    "stamina_broken",
    "dodge_invulnerability_fraction",
    "applied_force_speed_fraction",
    "stamina_regen_delay",
    "control_immunity_fraction",
    "alive",
    # The **locomotion** stamina -- the sprint budget -- as a fraction of
    # PLAYER_STAMINA_MAXIMUM. Every other stamina column above is the
    # guard/ability resource, which is a different bar that sprinting never
    # touches; the two share a word and nothing else.
    #
    # Added 2026-08-18, when the learner's gait began to affect its speed and
    # sprinting began to cost something. Without this the policy could spend a
    # bar it could not see and get silently dropped from 7.0 to 5.5 b/s, which
    # is a ~33% cut in its closing-speed reward with no observable cause.
    #
    # Only the self row carries evidence: it is sourced from actor zero's
    # CombatState, and an opponent's sprint budget is not observable in Hytale
    # anyway, so publishing a real value for other rows would be privileged
    # information rather than a fidelity gain.
    "locomotion_stamina_fraction",
)
STATUS_FLOAT_FEATURES = (
    "remaining_fraction",
    "cycle_progress",
    "damage_fraction",
    "healing_fraction",
    "resource_delta_fraction",
    "speed_multiplier",
)
STATUS_INTEGER_FEATURES = (
    "effect_id",
    "source_entity_id",
    "damage_cause",
    "resource_id",
    "overlap_mode",
)
ABILITY_FLOAT_FEATURES = (
    "duration_fraction",
    "cooldown_remaining_fraction",
    "active_progress",
    "affordable",
) + tuple(f"resource_cost_{index}" for index in range(RESOURCE_COUNT))
ABILITY_INTEGER_FEATURES = (
    "ability_id",
    "evidence_level",
    "requirement_bits",
)
ACTOR_WORLD_FLOAT_FEATURES = (
    "controller_in_fluid",
    "feet_submerged",
    "eyes_submerged",
    "drop_support_found",
    "drop_height_fraction",
)
ACTOR_WORLD_MASK_FEATURES = (
    "controller_medium_available",
    "submersion_available",
    "drop_available",
)
MOVEMENT_STATE_FEATURES = (
    "idle",
    "horizontal_idle",
    "jumping",
    "flying",
    "walking",
    "running",
    "sprinting",
    "crouching",
    "forced_crouching",
    "falling",
    "falling_far",
    "climbing",
    "in_fluid",
    "swimming",
    "swim_jumping",
    "on_ground",
    "mantling",
    "sliding",
    "mounting",
    "rolling",
    "sitting",
    "gliding",
    "sleeping",
)

WEAPON_INTEGER_SIZE = len(WEAPON_INTEGER_FEATURES)
DEFENSE_FLOAT_SIZE = len(DEFENSE_FLOAT_FEATURES)
STATUS_FLOAT_SIZE = len(STATUS_FLOAT_FEATURES)
STATUS_INTEGER_SIZE = len(STATUS_INTEGER_FEATURES)
ABILITY_FLOAT_SIZE = len(ABILITY_FLOAT_FEATURES)
ABILITY_INTEGER_SIZE = len(ABILITY_INTEGER_FEATURES)
ACTOR_WORLD_FLOAT_SIZE = len(ACTOR_WORLD_FLOAT_FEATURES)
ACTOR_WORLD_MASK_SIZE = len(ACTOR_WORLD_MASK_FEATURES)
MOVEMENT_STATE_SIZE = len(MOVEMENT_STATE_FEATURES)

DODGE_ACTION_COUNT = 4

OBSERVATION_FAILURE_LOADOUT = 1 << 6
OBSERVATION_FAILURE_MECHANICS = 1 << 7
OBSERVATION_FAILURE_ARSENAL = 1 << 8

if OBSERVATION_CAPACITY != 16 or STATUS_CAPACITY != 8:
    raise RuntimeError("learner observation v3 capacity drift")
if SKILL_COUNT != 9:
    raise RuntimeError("learner observation v3 skill contract drift")
if INTERACTION_CAPACITY != 8 or DOOR_INTENT_COUNT != 3:
    raise RuntimeError("learner observation v3 door contract drift")
