"""Fixed-shape data-program contract for Hytale 0.5.7 combat arsenals."""

from __future__ import annotations


ARSENAL_SCHEMA = "hytalerl_combat_arsenal_v45"
ARSENAL_VERSION = 45

# Protocol InteractionType ordinals used by the fixed-shape ability programs.
# The complete 25-value vocabulary is owned by ``worldgen.native_interactions``;
# Arsenal needs only a compact numeric identity at runtime so admission rules
# never branch on an item or weapon name.
INTERACTION_TYPE_UNRESOLVED = -1
INTERACTION_TYPE_PRIMARY = 0
INTERACTION_TYPE_SECONDARY = 1
INTERACTION_TYPE_ABILITY1 = 2
INTERACTION_TYPE_ABILITY3 = 4
INTERACTION_TYPE_COUNT = 25

EVIDENCE_UNRESOLVED = 0
EVIDENCE_ASSET_RESOLVED = 1
EVIDENCE_NATIVE_DIFFERENTIAL = 2

# The authored catalog remains lossless at 16x16. Execution may use any
# smaller shape that contains the selected programs, while the learner-facing
# ability axis remains stable for checkpoint compatibility.
CATALOG_ABILITY_CAPACITY = 16
CATALOG_EVENT_CAPACITY = 16
OBSERVATION_CAPACITY = 16

# Compatibility aliases for the authored storage contract. Runtime code must
# derive its bounds from the selected loadout's static array shape.
ABILITY_CAPACITY = CATALOG_ABILITY_CAPACITY
EVENT_CAPACITY = CATALOG_EVENT_CAPACITY
# RootInteraction.Cooldown.Charges is a fixed array. The installed 0.5.7
# combat catalog has at most four entries (Kunai and Combat Bow roots).
ABILITY_CHARGE_CAPACITY = 4
# Rows in a "Type": "Charging" root's hold-threshold table. This is unrelated to
# ABILITY_CHARGE_CAPACITY above, which sizes cooldown recharge slots. The widest
# authored table in 0.5.7 is the prototype Bow_*_Shoot_Charging draw ladder at
# ten rows (0.1 through 1.0 in 0.1 steps); the Shortbow Primary ladder uses
# five, the Volley three, and Sword/Mace/Daggers/Battleaxe two.
#
# These four fields live only on AbilityLoadout and are read back through
# ``.shape[-1]`` in programs/charge.py, so widening this does not move the
# observation. Confirmed at 10 by the 8271 pins in
# tests/jax/training/multi_actor/test_rollout.py, which still pass.
ABILITY_HOLD_CAPACITY = 10
PROJECTILE_CAPACITY = 16
AREA_CAPACITY = 8
# Projectile explosions and persistent areas expand across the complete
# logical entity axis, then pack in query-major/entity-slot order. Overflow
# rejects the complete environment row instead of truncating an effect.
IMPACT_DAMAGE_CAPACITY = 64
# Maximum authored events crossed by any shipped program in one supported
# 0.045-second engine tick. Overflow fails closed instead of truncating.
FIRED_EVENT_CAPACITY = 4

EVENT_NONE = 0
EVENT_MELEE_CONE = 1
EVENT_RADIAL_DAMAGE = 2
EVENT_PROJECTILE = 3
EVENT_STATUS = 4
EVENT_RESOURCE = 5
EVENT_HEAL = 6
EVENT_FORCE = 7
EVENT_AREA = 8
EVENT_CLEAR_STATUS = 9

TARGET_SELF = 0
TARGET_OTHER = 1
TARGET_BOTH = 2
TARGET_OTHER_OR_SELF = 3

FORCE_SET = 0
FORCE_ADD = 1

FORCE_DIRECTION_LOCAL = 0
FORCE_DIRECTION_POINT = 1
FORCE_DIRECTION_DIRECTIONAL = 2

REQUIRE_CLEAR_PROJECTILE_FLIGHT = 1 << 0
REQUIRE_CLEAR_FORCE_PATH = 1 << 1
REQUIRE_ENTITY_ONLY_AREA = 1 << 2
REQUIRE_STATIC_AREA_PLACEMENT = 1 << 3
REQUIRE_WORLD_PROJECTILE_COLLISION = 1 << 4
REQUIRE_LINE_OF_SIGHT = 1 << 5
REQUIRE_INJECTED_SELECTOR = 1 << 6
# Explicit opt-in evidence for the source contradiction where the native
# StandardPhysics impact callback appears to initialize ProjectileMiss with
# null hitLocation/hitNormal. Asset-intended terminal graphs remain unavailable
# until a live capture proves a contact-bound interaction chain.
REQUIRE_DEPLOYABLE_INTENDED_GRAPH_EVIDENCE = 1 << 7

# Event-control bits occupy the high half; low bits retain entity-effect flags.
EVENT_FLAG_VALUE_PERCENT = 1 << 16
EVENT_FLAG_STATUS_VALUE_PERCENT = 1 << 17
EVENT_FLAG_ANGLED_DAMAGE = 1 << 18
EVENT_FLAG_INJECTED_SELECTOR = 1 << 19
EVENT_FLAG_SERVER_SELECTOR = 1 << 20
EVENT_FLAG_SELECTOR_IGNORES_LINE_OF_SIGHT = 1 << 21
EVENT_FLAG_PROJECTILE_LEGACY_OFFSET = 1 << 22
EVENT_FLAG_PROJECTILE_PITCH_ADJUST_OFFSET = 1 << 23
# Independent asset evidence that an authored explosion/area affects entities
# only. This is distinct from runtime placement/contact availability: it says
# the program has no block-mutation branch, not that its center is known.
EVENT_FLAG_ENTITY_ONLY_AREA = 1 << 24
# The event is authored below a ParallelInteraction branch that the server
# starts through InteractionContext.fork rather than executing inline.
EVENT_FLAG_PARALLEL_FORK = 1 << 25
# DeployableAoeConfig.AttackTeam. The shipped Flame trap leaves it false;
# keeping it in the program avoids baking that asset choice into the kernel.
EVENT_FLAG_AREA_FRIENDLY_FIRE = 1 << 26
# The selected target is tested against an authored StabSelector slice each
# tick, but target selection itself is already resolved by the ordinary combat
# targeter. This is deliberately distinct from INJECTED_SELECTOR: the latter
# requires an externally supplied selector candidate, while this flag carries
# only the server's progressive selector geometry.
EVENT_FLAG_PROGRESSIVE_STAB_SELECTOR = 1 << 27
# The selected target is tested against the server's per-tick perspective
# HorizontalSelector frustum. Its authored direction is carried by the sign of
# EF_SELECTOR_YAW_LENGTH_DEGREES: positive ToLeft, negative ToRight.
EVENT_FLAG_PROGRESSIVE_HORIZONTAL_SELECTOR = 1 << 28
# StandardPhysics is an authored projectile mechanism, not a weapon family.
# Keeping its selectors in event data lets every current or future projectile
# opt in without adding weapon-name branches to the runtime.
EVENT_FLAG_PROJECTILE_STANDARD_PHYSICS = 1 << 29
EVENT_FLAG_PROJECTILE_ALLOW_ROLLING = 1 << 30
# Typed terminal graph: a world-contact ProjectileMiss atomically materializes
# one typed deployable area. Shape, raw native attack booleans, asymmetric
# collision bounds, and terminal-only physics aliases are owned exclusively by
# projectiles.terminal encode/decode helpers.
EVENT_FLAG_PROJECTILE_TERMINAL_DEPLOYABLE_AREA = 1 << 31
EVENT_STATUS_FLAG_MASK = (1 << 16) - 1
# The historical event ABI reserves low bits 0..15 for status controls, but
# the executable mechanics vocabulary currently defines only bits 0..6.
# Typed terminal deployables are a new v45 boundary and therefore reject the
# reserved/unknown bits rather than inheriting the broad legacy mask.
TERMINAL_STATUS_FLAG_MASK = (1 << 7) - 1
TERMINAL_UNKNOWN_STATUS_FLAG_MASK = 0xFFFFFFFF ^ TERMINAL_STATUS_FLAG_MASK

# Import-time uniqueness is a static proof over the complete authored event
# flag vocabulary. Bit 31 is deliberately the final available uint32 bit.
EVENT_CONTROL_FLAGS = (
    EVENT_FLAG_VALUE_PERCENT,
    EVENT_FLAG_STATUS_VALUE_PERCENT,
    EVENT_FLAG_ANGLED_DAMAGE,
    EVENT_FLAG_INJECTED_SELECTOR,
    EVENT_FLAG_SERVER_SELECTOR,
    EVENT_FLAG_SELECTOR_IGNORES_LINE_OF_SIGHT,
    EVENT_FLAG_PROJECTILE_LEGACY_OFFSET,
    EVENT_FLAG_PROJECTILE_PITCH_ADJUST_OFFSET,
    EVENT_FLAG_ENTITY_ONLY_AREA,
    EVENT_FLAG_PARALLEL_FORK,
    EVENT_FLAG_AREA_FRIENDLY_FIRE,
    EVENT_FLAG_PROGRESSIVE_STAB_SELECTOR,
    EVENT_FLAG_PROGRESSIVE_HORIZONTAL_SELECTOR,
    EVENT_FLAG_PROJECTILE_STANDARD_PHYSICS,
    EVENT_FLAG_PROJECTILE_ALLOW_ROLLING,
    EVENT_FLAG_PROJECTILE_TERMINAL_DEPLOYABLE_AREA,
)
if len(set(EVENT_CONTROL_FLAGS)) != len(EVENT_CONTROL_FLAGS):
    raise RuntimeError("arsenal event-control flag alias")
if any(value <= 0 or value & (value - 1) for value in EVENT_CONTROL_FLAGS):
    raise RuntimeError("arsenal event-control flags must be one-hot")
if EVENT_FLAG_PROJECTILE_TERMINAL_DEPLOYABLE_AREA != 0x80000000:
    raise RuntimeError("deployable terminal must reserve uint32 bit 31")

# Native NPC programs enter InteractionManager through a one-tick queue.
# Positive-runtime operations receive a time-zero first run, and non-player
# SelectInteraction receives one additional time-zero sample with no selector
# volume. ParallelInteraction branches after the first are forked into a later
# InteractionManager tick. Event clocks model those mechanical boundaries
# independently.
INTERACTION_QUEUE_DELAY_TICKS = 1
POSITIVE_RUNTIME_START_DELAY_TICKS = 1
NON_PLAYER_SELECTOR_START_DELAY_TICKS = 1
PARALLEL_FORK_START_DELAY_TICKS = 1
# A Player-backed SelectInteraction is client-sourced.  The server first
# exposes the client-wait operation to the synthetic peer, then consumes the
# returned InteractionSyncData on the following interaction boundary.  The
# deployed 23E7 bridge measured this as two additional ticks relative to the
# same role-only selector for three fixed seeds at 30 TPS.  This is a
# mechanism-wide actor-context value, never a weapon-specific adjustment.
PLAYER_SELECTOR_CLIENT_SYNC_DELAY_TICKS = 2
EVENT_SCHEDULER_CLOCK_COUNT = (
    INTERACTION_QUEUE_DELAY_TICKS
    + POSITIVE_RUNTIME_START_DELAY_TICKS
    + NON_PLAYER_SELECTOR_START_DELAY_TICKS
    + PARALLEL_FORK_START_DELAY_TICKS
    + PLAYER_SELECTOR_CLIENT_SYNC_DELAY_TICKS
)
# Native NPCs have no remote client. InteractionManager therefore executes
# SimpleInstantInteraction once against its simulated state and once against
# server state. ChangeStatInteraction reaches the same command-buffer stat map
# in both passes, so additive stat changes have two observable applications.
NATIVE_NPC_INSTANT_ADD_STAT_APPLICATIONS = 2

# Resource costs retain the authored amount and identify the interaction
# mechanism that commits it. This keeps execution-pass multiplicity out of
# individual weapon profiles and prevents inventory/set-stat transactions from
# being treated like additive ChangeStat interactions.
RESOURCE_COST_NONE = 0
RESOURCE_COST_SINGLE_APPLICATION = 1
RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT = 2
RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT_STAMINA_BREAK_IMMUNE = 3
RESOURCE_COST_SINGLE_APPLICATION_STAMINA_BREAK_IMMUNE = 4
RESOURCE_COST_KIND_COUNT = 5

ARSENAL_FAILURE_ABILITY_OVERFLOW = 1 << 0
ARSENAL_FAILURE_EVENT_OVERFLOW = 1 << 1
ARSENAL_FAILURE_PROJECTILE_OVERFLOW = 1 << 2
ARSENAL_FAILURE_AREA_OVERFLOW = 1 << 3
ARSENAL_FAILURE_INVALID_LOADOUT = 1 << 4
ARSENAL_FAILURE_INVALID_COMMAND = 1 << 5
ARSENAL_FAILURE_INVALID_STATE = 1 << 6
ARSENAL_FAILURE_UNSUPPORTED_WORLD = 1 << 7

PROJECTILE_NONE = 0
PROJECTILE_ARROW = 1
PROJECTILE_BIG_ARROW = 2
PROJECTILE_FIREBALL = 3
PROJECTILE_ICE_BALL = 4
PROJECTILE_ICE_BOLT = 5
PROJECTILE_BOMB = 6
PROJECTILE_CORRUPTION_ORB = 7
PROJECTILE_KUNAI = 8
PROJECTILE_SPEAR = 9
PROJECTILE_LEGACY_ARROW = 10
PROJECTILE_PROTOTYPE_ARROW = 11
PROJECTILE_GUN_BULLET = 12
PROJECTILE_BLUNDERBUSS_BULLET = 13
PROJECTILE_DEPLOYABLE = 14

# Legacy resolved relationship categories used by ordinary areas. These are
# deliberately not the DeployableAoeConfig attack booleans below: native
# deployable targeting applies those booleans in an ordered predicate.
AREA_TARGET_OWNER = 1 << 0
AREA_TARGET_TEAM = 1 << 1
AREA_TARGET_ENEMIES = 1 << 2
AREA_TARGET_MASK = AREA_TARGET_OWNER | AREA_TARGET_TEAM | AREA_TARGET_ENEMIES

# Raw authored DeployableAoeConfig booleans carried by EI_TARGET_MODE only
# under EVENT_FLAG_PROJECTILE_TERMINAL_DEPLOYABLE_AREA.
DEPLOYABLE_ATTACK_OWNER = 1 << 0
DEPLOYABLE_ATTACK_TEAM = 1 << 1
DEPLOYABLE_ATTACK_ENEMIES = 1 << 2
DEPLOYABLE_ATTACK_FLAG_MASK = (
    DEPLOYABLE_ATTACK_OWNER | DEPLOYABLE_ATTACK_TEAM | DEPLOYABLE_ATTACK_ENEMIES
)

# DeployableAoeConfig.Shape ordinals carried by the terminal-only EI_FORCE_MODE
# alias. They are not FORCE_SET/FORCE_ADD meanings under the typed bit31 row.
AREA_SHAPE_NONE = 0
AREA_SHAPE_SPHERE = 1
AREA_SHAPE_CYLINDER = 2

AREA_NONE = 0
AREA_FIRE_TRAP = 1
AREA_GENERIC = 2
AREA_DEPLOYABLE_AOE = 3

# Contiguous float-event channels. Keeping authored programs dense materially
# reduces device memory and gather cost compared with dozens of parallel leaves.
(
    EF_DAMAGE,
    EF_RANGE,
    EF_HALF_ANGLE_DEGREES,
    EF_RADIUS,
    EF_FALLOFF,
    EF_PROJECTILE_SPEED,
    EF_PROJECTILE_GRAVITY,
    EF_PROJECTILE_TERMINAL_VELOCITY,
    EF_PROJECTILE_HALF_EXTENT,
    EF_PROJECTILE_LIFETIME_SECONDS,
    EF_STATUS_DURATION_SECONDS,
    EF_STATUS_COOLDOWN_SECONDS,
    EF_STATUS_DAMAGE,
    EF_STATUS_HEALING,
    EF_STATUS_RESOURCE_DELTA,
    EF_STATUS_SPEED_MULTIPLIER,
    EF_AREA_DURATION_SECONDS,
    EF_AREA_INTERVAL_SECONDS,
    EF_AREA_END_RADIUS,
    EF_AREA_HEIGHT,
    EF_FORCE_X,
    EF_FORCE_Y,
    EF_FORCE_Z,
    EF_FORCE_MAGNITUDE,
    EF_YAW_OFFSET_DEGREES,
    EF_PITCH_OFFSET_DEGREES,
    EF_AIR_RESISTANCE,
    EF_AIR_RESISTANCE_MAX,
    EF_GROUND_RESISTANCE,
    EF_GROUND_RESISTANCE_MAX,
    EF_RESISTANCE_THRESHOLD,
    EF_PROJECTILE_FUSE_SECONDS,
    EF_AREA_RADIUS_CHANGE_SECONDS,
    EF_ON_HIT_RESOURCE_DELTA,
    EF_ANGLED_DAMAGE,
    EF_ANGLED_ANGLE_DEGREES,
    EF_ANGLED_DISTANCE_DEGREES,
    EF_ANGLED_FORCE_X,
    EF_ANGLED_FORCE_Y,
    EF_ANGLED_FORCE_Z,
    EF_ANGLED_FORCE_MAGNITUDE,
    EF_PROJECTILE_DEAD_TIME_SECONDS,
    EF_RANDOM_PERCENTAGE,
    EF_PROJECTILE_SPAWN_OFFSET_X,
    EF_PROJECTILE_SPAWN_OFFSET_Y,
    EF_PROJECTILE_SPAWN_OFFSET_Z,
    EF_SELECTOR_RUNTIME_SECONDS,
    EF_SELECTOR_START_DISTANCE,
    EF_SELECTOR_END_DISTANCE,
    EF_SELECTOR_EXTEND_LEFT,
    EF_SELECTOR_EXTEND_RIGHT,
    EF_SELECTOR_EXTEND_BOTTOM,
    EF_SELECTOR_EXTEND_TOP,
    EF_SELECTOR_YAW_OFFSET_DEGREES,
    EF_SELECTOR_PITCH_OFFSET_DEGREES,
    EF_SELECTOR_ROLL_OFFSET_DEGREES,
    EF_SELECTOR_YAW_LENGTH_DEGREES,
    EF_PROJECTILE_BOUNCINESS,
    EF_PROJECTILE_BOUNCE_LIMIT,
    EF_PROJECTILE_ROLLING_FRICTION_FACTOR,
    EF_PROJECTILE_DIRECT_DAMAGE,
    # C3 ForceProfile. `VerticalClamp` is authored as a two-element array on the
    # ApplyForce node (Downstrike [10, 30], Pounce [-20, 10]), so it needs two
    # channels rather than one.
    EF_VERTICAL_CLAMP_MIN,
    EF_VERTICAL_CLAMP_MAX,
) = range(63)
EVENT_FLOAT_FEATURES = 63
EVENT_FLOAT_CHANNEL_NAMES = (
    "damage",
    "range",
    "half_angle_degrees",
    "radius",
    "falloff",
    "projectile_speed",
    "projectile_gravity",
    "projectile_terminal_velocity",
    "projectile_half_extent",
    "projectile_lifetime_seconds",
    "status_duration_seconds",
    "status_cooldown_seconds",
    "status_damage",
    "status_healing",
    "status_resource_delta",
    "status_speed_multiplier",
    "area_duration_seconds",
    "area_interval_seconds",
    "area_end_radius",
    "area_height",
    "force_x",
    "force_y",
    "force_z",
    "force_magnitude",
    "yaw_offset_degrees",
    "pitch_offset_degrees",
    "air_resistance",
    "air_resistance_max",
    "ground_resistance",
    "ground_resistance_max",
    "resistance_threshold",
    "projectile_fuse_seconds",
    "area_radius_change_seconds",
    "on_hit_resource_delta",
    "angled_damage",
    "angled_angle_degrees",
    "angled_distance_degrees",
    "angled_force_x",
    "angled_force_y",
    "angled_force_z",
    "angled_force_magnitude",
    "projectile_dead_time_seconds",
    "random_percentage",
    "projectile_spawn_offset_x",
    "projectile_spawn_offset_y",
    "projectile_spawn_offset_z",
    "selector_runtime_seconds",
    "selector_start_distance",
    "selector_end_distance",
    "selector_extend_left",
    "selector_extend_right",
    "selector_extend_bottom",
    "selector_extend_top",
    "selector_yaw_offset_degrees",
    "selector_pitch_offset_degrees",
    "selector_roll_offset_degrees",
    "selector_yaw_length_degrees",
    "projectile_bounciness",
    "projectile_bounce_limit",
    "projectile_rolling_friction_factor",
    "projectile_direct_damage",
    "vertical_clamp_min",
    "vertical_clamp_max",
)

(
    EI_DAMAGE_CAUSE,
    EI_PROJECTILE_KIND,
    EI_STATUS_ID,
    EI_STATUS_DAMAGE_CAUSE,
    EI_STATUS_RESOURCE_ID,
    EI_STATUS_OVERLAP_MODE,
    EI_FORCE_MODE,
    EI_TARGET_MODE,
    EI_RESISTANCE_STYLE,
    EI_ON_HIT_RESOURCE_ID,
    EI_FORCE_DIRECTION_MODE,
    EI_DAMAGE_CLASS,
    EI_BLOCK_DAMAGE_RADIUS,
    EI_PROJECTILE_BOUNCE_COUNT,
    EI_PROJECTILE_DIRECT_DAMAGE_CAUSE,
    # C3 ForceProfile `AdjustVertical`. Authored as a bool, so it gates whether
    # the EF_VERTICAL_CLAMP_* pair is consulted at all; a cleared flag leaves
    # the authored direction untouched.
    EI_ADJUST_VERTICAL,
) = range(16)
EVENT_INTEGER_FEATURES = 16
EVENT_INTEGER_CHANNEL_NAMES = (
    "damage_cause",
    "projectile_or_area_kind",
    "status_id",
    "status_damage_cause",
    "status_resource_id",
    "status_overlap_mode",
    "force_mode",
    "target_mode",
    "resistance_style",
    "on_hit_resource_id",
    "force_direction_mode",
    "damage_class",
    "block_damage_radius",
    "projectile_bounce_count",
    "projectile_direct_damage_cause",
    "adjust_vertical",
)

if len(EVENT_FLOAT_CHANNEL_NAMES) != EVENT_FLOAT_FEATURES:
    raise RuntimeError("arsenal float-event channel drift")
if len(EVENT_INTEGER_CHANNEL_NAMES) != EVENT_INTEGER_FEATURES:
    raise RuntimeError("arsenal integer-event channel drift")

FAMILY_NONE = 0
FAMILY_SWORD = 1
FAMILY_MACE = 2
FAMILY_BATTLEAXE = 3
FAMILY_DAGGERS = 4
FAMILY_SHORTBOW = 5
FAMILY_CROSSBOW = 6
FAMILY_SHIELD = 7
FAMILY_FLAME_STAFF = 8
FAMILY_ICE_STAFF = 9
FAMILY_BOMB = 10
FAMILY_POTION = 11
FAMILY_WAND = 12
FAMILY_SPELLBOOK = 13
FAMILY_KUNAI = 14
FAMILY_ROOT_WAND = 15
FAMILY_SPEAR_ADAMANTITE = 16
FAMILY_SPEAR_ADAMANTITE_SAURIAN = 17
FAMILY_SPEAR_BONE = 18
FAMILY_SPEAR_BRONZE = 19
FAMILY_SPEAR_COBALT = 20
FAMILY_SPEAR_COPPER = 21
FAMILY_SPEAR_CRUDE = 22
FAMILY_SPEAR_FISHBONE = 23
FAMILY_SPEAR_IRON = 24
FAMILY_SPEAR_LEAF = 25
FAMILY_SPEAR_MITHRIL = 26
FAMILY_SPEAR_ONYXIUM = 27
FAMILY_SPEAR_SCRAP = 28
FAMILY_SPEAR_STONE_TRORK = 29
FAMILY_SPEAR_THORIUM = 30
FAMILY_SPEAR_TRIBAL = 31
