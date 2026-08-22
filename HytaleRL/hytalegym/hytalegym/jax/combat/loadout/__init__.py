"""Fixed-shape heterogeneous melee loadouts."""

from hytalegym.jax.combat.loadout.schema.contract import (
    LOADOUT_FAILURE_ATTACK_OVERFLOW,
    LOADOUT_FAILURE_NONE,
    MAX_MELEE_ATTACKS,
    MELEE_ATTACK_FLOAT_FEATURES,
    MELEE_ATTACK_FLOAT_SIZE,
    MELEE_ATTACK_INTEGER_FEATURES,
    MELEE_ATTACK_INTEGER_SIZE,
    MELEE_COOLDOWN_NORMALIZATION_SECONDS,
    MELEE_LOADOUT_SCHEMA,
    MELEE_LOADOUT_VERSION,
    MELEE_LOADOUT_OBSERVATION_SCHEMA,
    MELEE_LOADOUT_OBSERVATION_VERSION,
    MELEE_RANGE_NORMALIZATION_BLOCKS,
    MELEE_WEAPON_INTEGER_FEATURES,
    MELEE_WEAPON_INTEGER_SIZE,
    WEAPON_KIND_MELEE,
    WEAPON_KIND_NONE,
)
from hytalegym.jax.combat.loadout.runtime.encoder import encode_melee_loadout
from hytalegym.jax.combat.loadout.runtime.factory import (
    default_melee_loadout,
    empty_melee_loadout,
)
from hytalegym.jax.combat.loadout.schema.spec import (
    melee_loadout_contract_json,
    melee_loadout_contract_manifest,
    melee_loadout_contract_sha256,
)
from hytalegym.jax.combat.loadout.schema.types import (
    MeleeLoadoutBatch,
    MeleeLoadoutObservation,
)
from hytalegym.jax.combat.loadout.schema.validation import validate_melee_loadout

__all__ = [
    "LOADOUT_FAILURE_ATTACK_OVERFLOW",
    "LOADOUT_FAILURE_NONE",
    "MAX_MELEE_ATTACKS",
    "MELEE_ATTACK_FLOAT_FEATURES",
    "MELEE_ATTACK_FLOAT_SIZE",
    "MELEE_ATTACK_INTEGER_FEATURES",
    "MELEE_ATTACK_INTEGER_SIZE",
    "MELEE_COOLDOWN_NORMALIZATION_SECONDS",
    "MELEE_LOADOUT_SCHEMA",
    "MELEE_LOADOUT_VERSION",
    "MELEE_LOADOUT_OBSERVATION_SCHEMA",
    "MELEE_LOADOUT_OBSERVATION_VERSION",
    "MELEE_RANGE_NORMALIZATION_BLOCKS",
    "MELEE_WEAPON_INTEGER_FEATURES",
    "MELEE_WEAPON_INTEGER_SIZE",
    "WEAPON_KIND_MELEE",
    "WEAPON_KIND_NONE",
    "MeleeLoadoutBatch",
    "MeleeLoadoutObservation",
    "default_melee_loadout",
    "empty_melee_loadout",
    "encode_melee_loadout",
    "melee_loadout_contract_json",
    "melee_loadout_contract_manifest",
    "melee_loadout_contract_sha256",
    "validate_melee_loadout",
]
