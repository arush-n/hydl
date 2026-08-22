"""Fixed-capacity combat-owned melee loadout contract."""

from __future__ import annotations


MELEE_LOADOUT_SCHEMA = "hytalerl_melee_loadout_v1"
MELEE_LOADOUT_VERSION = 1
MELEE_LOADOUT_OBSERVATION_SCHEMA = "hytalerl_melee_loadout_observation_v1"
MELEE_LOADOUT_OBSERVATION_VERSION = 1
MAX_MELEE_ATTACKS = 8

WEAPON_KIND_NONE = 0
WEAPON_KIND_MELEE = 1

LOADOUT_FAILURE_NONE = 0
LOADOUT_FAILURE_ATTACK_OVERFLOW = 1 << 0

MELEE_RANGE_NORMALIZATION_BLOCKS = 32.0
MELEE_COOLDOWN_NORMALIZATION_SECONDS = 10.0

MELEE_ATTACK_FLOAT_FEATURES = (
    "range_fraction",
    "half_angle_fraction",
    "damage_fraction",
    "cooldown_min_fraction",
    "cooldown_max_fraction",
    "requires_line_of_sight",
)
MELEE_ATTACK_INTEGER_FEATURES = (
    "attack_slot",
    "hit_delay_ticks",
)
MELEE_WEAPON_INTEGER_FEATURES = (
    "weapon_id",
    "weapon_kind",
    "attack_count",
)

MELEE_ATTACK_FLOAT_SIZE = len(MELEE_ATTACK_FLOAT_FEATURES)
MELEE_ATTACK_INTEGER_SIZE = len(MELEE_ATTACK_INTEGER_FEATURES)
MELEE_WEAPON_INTEGER_SIZE = len(MELEE_WEAPON_INTEGER_FEATURES)
