"""Fixed combat-resource, defense, damage, and status constants."""

from __future__ import annotations

from hytalegym.worldgen.region.stability import (
    current_native_evidence_jar_sha256,
)
from hytalegym.rulesets.damage_classes import DAMAGE_CLASS_IDS


COMBAT_MECHANICS_SCHEMA = "hytalerl_combat_mechanics_v16"
COMBAT_MECHANICS_VERSION = 16

RESOURCE_STAMINA = 0
RESOURCE_MANA = 1
RESOURCE_MAGIC_CHARGES = 2
RESOURCE_SIGNATURE_ENERGY = 3
RESOURCE_SIGNATURE_CHARGES = 4
# Channel four is family-local. Flame Staff uses the distinct native
# DeployablePreview stat; ranged families use SignatureCharges. They are
# never equipped by the same compiled family, so the alias costs no device
# memory and does not conflate their authored maxima or transitions.
RESOURCE_DEPLOYABLE_PREVIEW = RESOURCE_SIGNATURE_CHARGES
RESOURCE_AMMO = 5
RESOURCE_OXYGEN = 6
RESOURCE_COUNT = 7

DAMAGE_PHYSICAL = 0
DAMAGE_PROJECTILE = 1
DAMAGE_FIRE = 2
DAMAGE_ICE = 3
DAMAGE_POISON = 4
DAMAGE_ENVIRONMENTAL = 5
DAMAGE_ELEMENTAL = 6
DAMAGE_ENVIRONMENT = 7
DAMAGE_BLUDGEONING = 8
DAMAGE_SLASHING = 9
DAMAGE_COMMAND = 10
DAMAGE_DROWNING = 11
DAMAGE_FALL = 12
DAMAGE_OUT_OF_WORLD = 13
DAMAGE_SUFFOCATION = 14
DAMAGE_COUNT = 15
DAMAGE_CAUSE_IDS = (
    "Physical",
    "Projectile",
    "Fire",
    "Ice",
    "Poison",
    "Environmental",
    "Elemental",
    "Environment",
    "Bludgeoning",
    "Slashing",
    "Command",
    "Drowning",
    "Fall",
    "OutOfWorld",
    "Suffocation",
)

DAMAGE_CLASS_UNKNOWN = 0
DAMAGE_CLASS_LIGHT = 1
DAMAGE_CLASS_CHARGED = 2
DAMAGE_CLASS_SIGNATURE = 3
DAMAGE_CLASS_COUNT = 4
if len(DAMAGE_CLASS_IDS) != DAMAGE_CLASS_COUNT:
    raise RuntimeError("damage-class asset order differs from the mechanics contract")

DODGE_NONE = 0
DODGE_FORWARD = 1
DODGE_BACK = 2
DODGE_LEFT = 3
DODGE_RIGHT = 4
DODGE_DIRECTION_COUNT = 5
# Assets.zip:Server/Item/Interactions/Dodge.json routes Forward/Back and both
# forward/back diagonals to Simple (no dodge payload). Only the authored
# Dodge_Left and Dodge_Right branches apply the effect, force, and stamina
# transaction. This mask excludes DODGE_NONE and follows policy-mask order
# [forward, back, left, right].
DODGE_AUTHORED_ACTION_MASK = (False, False, True, True)

# A Dodge asset carries an authored VelocityConfig.  A pure server NPC reaches
# ApplyForceInteraction.simulateTick0 with a null config, but the bridge policy
# actor reconstructs the client config and routes it through the role motion
# controller.  Keep both contracts explicit; policy actors default to the
# measured configured path while ordinary server-NPC simulations can select
# the null-config profile without overwriting either model.
DODGE_EXECUTION_NATIVE_NPC_NULL_CONFIG = 0
DODGE_EXECUTION_AUTHORED_CLIENT_CONFIGURED = 1
DODGE_EXECUTION_PROFILE_COUNT = 2

STATUS_OVERLAP_IGNORE = 0
STATUS_OVERLAP_EXTEND = 1
STATUS_OVERLAP_OVERWRITE = 2

STATUS_FLAG_INVULNERABLE = 1 << 0
STATUS_FLAG_DISABLE_MOVEMENT = 1 << 1
STATUS_FLAG_DISABLE_ABILITIES = 1 << 2
STATUS_FLAG_IGNORE_KNOCKBACK = 1 << 3
STATUS_FLAG_DEBUFF = 1 << 4
STATUS_FLAG_DISABLE_SPRINT = 1 << 5
STATUS_FLAG_CONTROL_IMMUNITY_GATED = 1 << 6

STATUS_CAPACITY = 8
STATUS_APPLICATION_CAPACITY = 4
DAMAGE_EVENT_CAPACITY = 8

MECHANICS_FAILURE_STATUS_OVERFLOW = 1 << 0
MECHANICS_FAILURE_INVALID_COMMAND = 1 << 1
MECHANICS_FAILURE_INVALID_STATE = 1 << 2
MECHANICS_FAILURE_UNSUPPORTED_DODGE_COLLISION = 1 << 3

HYTALE_0_5_7_ASSETS_SHA256 = (
    "1B8802C284C228AE4549DAC037716C175BC6B94B0FC9AAC2B7039AC6BD2FFD5D"
)
HYTALE_0_5_7_NATIVE_EVIDENCE_JAR_SHA256 = current_native_evidence_jar_sha256()

# Assets.zip (HYTALE_0_5_7_ASSETS_SHA256):
# Server/Entity/Stats/Stamina.json and StaminaRegenDelay.json.  These values
# are direct asset magnitudes, not estimates from the curated asset-surface
# census (which intentionally excludes Server/Entity/Stats/).
STAMINA_MINIMUM = -4.0
STAMINA_MAXIMUM = 10.0
STAMINA_REGEN_AMOUNT = 0.3
STAMINA_REGEN_INTERVAL_SECONDS = 0.1
STAMINA_REGEN_DELAY_AMOUNT = 0.1
STAMINA_REGEN_DELAY_INTERVAL_SECONDS = 0.1
STAMINA_BROKEN_REGEN_DELAY_SECONDS = -0.5

# Server/Entity/Stats/Oxygen.json.  The two additive programs are mutually
# exclusive through the Suffocating condition, but both retain their own
# native RegeneratingValue cadence.  DamageSystems.CanBreathe is a separate
# one-second delayed system (10 drowning, 20 solid-block suffocation).
OXYGEN_MINIMUM = 0.0
OXYGEN_MAXIMUM = 100.0
OXYGEN_BREATHABLE_REGEN_AMOUNT = 25.0
OXYGEN_SUFFOCATING_REGEN_AMOUNT = -3.0
OXYGEN_REGEN_INTERVAL_SECONDS = 0.5
BREATHING_DAMAGE_INTERVAL_SECONDS = 1.0
DROWNING_DAMAGE_AMOUNT = 10.0
SUFFOCATION_DAMAGE_AMOUNT = 20.0

# Common_Guard_Wield and the common guard entry/exit chain.
GUARD_ENTRY_STAMINA_COST = 0.5
GUARD_HALF_ANGLE_DEGREES = 90.0
GUARD_EXIT_REGEN_DELAY_SECONDS = -1.0
GUARD_BASH_STAMINA_COST = 2.0
# The bridge queues the held root after InteractionManager has already run for
# the selected control tick. The installed StatsCondition/Serial/Wielding
# chain then commits its entry transaction on engine tick two and exposes
# Wielding on tick three. Release is likewise observed one tick after the
# falling input edge. These are lifecycle positions, not durations.
GUARD_ENTRY_COST_TICK = 2
GUARD_ACTIVATION_TICK = 3
GUARD_RELEASE_TICK = 2

# Assets.zip: Server/Item/Interactions/Dodge/Dodge_Left.json and
# Dodge_Right.json, plus the referenced Dodge_Invulnerability effect.
# Dodge_Left/Right use 2 as an affordability threshold, then execute one
# -2 Stamina ChangeStat operation.  StatsConditionWithModifier only checks
# Costs; it does not deduct them.  Keep admission and spend separate because
# armor may modify the threshold while the authored mutation remains explicit.
DODGE_STAMINA_ADMISSION_COST = 2.0
DODGE_STAMINA_SPEND_COST = 2.0
# Compatibility name retained for callers that mean admission. New code must
# use the explicit admission/spend constants above.
DODGE_STAMINA_COST = DODGE_STAMINA_ADMISSION_COST
DODGE_LAUNCH_TICK = 2
DODGE_COST_TICK = 3
DODGE_FORCE = 13.0
DODGE_INVULNERABILITY_SECONDS = 0.25
# Decompiled Hytale 0.5.7
# InteractionTypeUtils.DEFAULT_COOLDOWN. Dodge.json does not override it, so
# InteractionManager applies this engine default to InteractionType.Dodge.
# At 30 TPS its next admissible boundary is ceil(0.35 * 30) = 11 ticks.
DODGE_COOLDOWN_SECONDS = 0.35
DODGE_REGEN_DELAY_SECONDS = -0.7
DODGE_AIR_RESISTANCE = 0.97
DODGE_AIR_RESISTANCE_MAX = 0.96
DODGE_GROUND_RESISTANCE = 0.94
DODGE_GROUND_RESISTANCE_MAX = 0.82
DODGE_RESISTANCE_THRESHOLD = 5.0
DODGE_VELOCITY_REMOVAL_SQUARED = 0.001
# Internal discriminator for MotionControllerBase.forceVelocity damping.
# Asset VelocityConfig styles retain their server enum values (Linear=0, Exp=1).
APPLIED_RESISTANCE_STYLE_FORCE_VELOCITY = 2

# Server/Entity/Stats/Immunity.json. Control interactions add 25; reaching
# 100 applies native Immune and blocks the interaction's wrapped payload.
CONTROL_IMMUNITY_MAXIMUM = 100.0
CONTROL_IMMUNITY_INCREMENT = 25.0
CONTROL_IMMUNITY_REGEN_AMOUNT = 0.1
CONTROL_IMMUNITY_REGEN_INTERVAL_SECONDS = 0.1
