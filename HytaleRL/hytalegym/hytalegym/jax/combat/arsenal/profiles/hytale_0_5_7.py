"""Pinned Hytale 0.5.7 combat programs resolved from local authored assets."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from functools import lru_cache, partial
from typing import NamedTuple

import jax
import numpy as np

from hytalegym.jax.combat.arsenal.schema.contract import *
from hytalegym.jax.combat.arsenal.factory import empty_ability_loadout
from hytalegym.jax.combat.arsenal.schema.types import (
    AbilityLoadout,
    ArsenalRuntimeCapacity,
)
from hytalegym.jax.combat.arsenal.profiles.archetypes import (
    DOUBLE_INCANDESCENT_SPEAR_ASSET_ID,
    DOUBLE_INCANDESCENT_SPEAR_FAMILY_ID,
    DOUBLE_INCANDESCENT_SPEAR_PROFILE_NAME,
    DEPLOYABLE_PROFILE_NAMES,
    DEPLOYABLE_SPECS,
    GUN_SPECS,
    PROTOTYPE_BOW_SPECS,
    KNIFE_WEAPON_VARIANTS,
    LEGACY_CASTER_PROFILE_NAMES,
    LEGACY_CASTER_VARIANTS,
    LEGACY_MELEE_PROFILE_NAMES,
    LEGACY_MELEE_VARIANTS,
    LegacyCasterVariant,
    LegacyMeleeVariant,
    KnifeWeaponVariant,
    GunSpec,
    PrototypeBowSpec,
    ProjectileTerminalDeployableSpec,
    WeaponScalarSet,
    SPECIAL_WEAPON_PROFILE_NAMES,
    TRIBAL_BLOWGUN_ASSET_ID,
    TRIBAL_BLOWGUN_FAMILY_ID,
    TRIBAL_BLOWGUN_PROFILE_NAME,
    TRIBAL_CLAWS_ASSET_ID,
    TRIBAL_CLAWS_FAMILY_ID,
    TRIBAL_CLAWS_PROFILE_NAME,
    VOID_SCYTHE_ASSET_ID,
    VOID_SCYTHE_FAMILY_ID,
    VOID_SCYTHE_PROFILE_NAME,
)
from hytalegym.jax.combat.arsenal.profiles.archetypes.deployables.adapter import (
    adapt_projectile_terminal_deployable,
)
from hytalegym.jax.combat.arsenal.profiles.variants import (
    SCALAR_VARIANTS,
    SCALAR_VARIANT_PROFILE_NAMES,
    ScalarVariant,
)
from hytalegym.jax.combat.arsenal.programs.timing import (
    CROSSBOW_RELOAD_DURATION_SECONDS,
    CROSSBOW_RELOAD_RESOURCE_TIMES_SECONDS,
)
from hytalegym.jax.combat.arsenal.programs.continuation import (
    CONTINUATION_NONE,
    CONTINUATION_WAIT_FOR_COLLISION,
    CONTINUATION_WAIT_FOR_GROUND,
)
from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.combat.mechanics import (
    DAMAGE_CLASS_CHARGED,
    DAMAGE_CLASS_LIGHT,
    DAMAGE_CLASS_SIGNATURE,
    DAMAGE_CLASS_UNKNOWN,
    DAMAGE_ENVIRONMENT,
    DAMAGE_FIRE,
    DAMAGE_ICE,
    DAMAGE_PHYSICAL,
    DAMAGE_PROJECTILE,
    RESOURCE_COUNT,
    RESOURCE_AMMO,
    RESOURCE_DEPLOYABLE_PREVIEW,
    RESOURCE_MAGIC_CHARGES,
    RESOURCE_MANA,
    RESOURCE_SIGNATURE_CHARGES,
    RESOURCE_SIGNATURE_ENERGY,
    RESOURCE_STAMINA,
    STATUS_FLAG_DEBUFF,
    STATUS_FLAG_CONTROL_IMMUNITY_GATED,
    STATUS_FLAG_DISABLE_ABILITIES,
    STATUS_FLAG_DISABLE_MOVEMENT,
    STATUS_FLAG_DISABLE_SPRINT,
    STATUS_FLAG_IGNORE_KNOCKBACK,
    STATUS_OVERLAP_IGNORE,
    STATUS_OVERLAP_OVERWRITE,
)

CORE_PROFILE_NAMES = (
    "iron_sword",
    "iron_mace",
    "iron_battleaxe",
    "iron_daggers",
    "iron_shortbow",
    "iron_crossbow",
    "iron_shield",
    "flame_staff",
    "ice_staff",
    "bombs",
    "potions",
    "stoneskin_wand",
    "skeleton_mage_spellbook",
    "kunai",
    "root_wand",
    "adamantite_spear",
    "adamantite_saurian_spear",
    "bone_spear",
    "bronze_spear",
    "cobalt_spear",
    "copper_spear",
    "crude_spear",
    "fishbone_spear",
    "iron_spear",
    "leaf_spear",
    "mithril_spear",
    "onyxium_spear",
    "scrap_spear",
    "stone_trork_spear",
    "thorium_spear",
    "tribal_spear",
)
PROFILE_NAMES = (
    CORE_PROFILE_NAMES
    + SCALAR_VARIANT_PROFILE_NAMES
    + (VOID_SCYTHE_PROFILE_NAME,)
    + LEGACY_MELEE_PROFILE_NAMES
    + LEGACY_CASTER_PROFILE_NAMES
    + SPECIAL_WEAPON_PROFILE_NAMES
    + DEPLOYABLE_PROFILE_NAMES
)
SPEAR_PROFILE_NAMES = CORE_PROFILE_NAMES[-16:]
SHORTBOW_FAMILY_IDS = (
    FAMILY_SHORTBOW,
    *(row.family_id for row in SCALAR_VARIANTS if row.template == "shortbow"),
)
CROSSBOW_FAMILY_IDS = (
    FAMILY_CROSSBOW,
    *(row.family_id for row in SCALAR_VARIANTS if row.template == "crossbow"),
)
NATIVE_RESOURCE_STAT_IDS = (
    "Stamina",
    "Mana",
    "MagicCharges",
    "SignatureEnergy",
    "SignatureCharges",
    "Ammo",
    "Oxygen",
)
WORLD_CONDITIONAL_ABILITY_ASSETS = (
    "Weapon_Mace_Signature_Groundslam",
    "Weapon_Daggers_Primary_Pounce",
)
_PROFILE_INDEX = {name: index + 1 for index, name in enumerate(PROFILE_NAMES)}


class _ResourceCost(NamedTuple):
    """One resolved cost plus its authored interaction-graph commit point."""

    amount: float
    kind: int
    commit_time_seconds: float = 0.0
    commit_flags: int = 0


class _ResourceScalarBinding(NamedTuple):
    """Bind one named asset scalar to one runtime resource operation.

    ``admission_required`` is deliberately independent of the authored cost.
    Some native roots expose a parent ``StatsCondition`` while the bridge's
    policy-selected child starts after it.  The later ``ChangeStat`` remains
    part of the graph, but an unavailable stat is a no-op in
    ``EntityStatMap.processStatChanges`` and must not close that child.
    """

    scalar_name: str
    resource_id: int
    cost_kind: int
    commit_time_seconds: float = 0.0
    commit_flags: int = 0
    admission_required: bool = True


class _ResourcePhase(NamedTuple):
    """One outer-root resource value visible over a bounded tick interval."""

    value: float
    inactive_value: float
    start_tick: int
    end_tick: int


class _ProjectileLaunchTransform(NamedTuple):
    """Asset-authored launch transform, before the first physics tick."""

    offset: tuple[float, float, float]
    yaw_degrees: float
    pitch_degrees: float
    legacy_ballistic: bool
    pitch_adjust_depth: bool


class _StabSelectorProgram(NamedTuple):
    """Resolved server ``StabSelector`` dimensions and active runtime."""

    runtime_seconds: float
    start_distance: float
    end_distance: float
    extend_left: float
    extend_right: float
    extend_bottom: float
    extend_top: float
    yaw_offset_degrees: float
    pitch_offset_degrees: float
    roll_offset_degrees: float


class _HorizontalSelectorProgram(NamedTuple):
    """Resolved server ``HorizontalSelector`` frustum and active runtime."""

    runtime_seconds: float
    start_distance: float
    end_distance: float
    extend_bottom: float
    extend_top: float
    yaw_length_degrees: float
    yaw_direction_multiplier: float
    yaw_start_offset_degrees: float
    pitch_offset_degrees: float
    roll_offset_degrees: float


_ZERO_PROJECTILE_LAUNCH = _ProjectileLaunchTransform(
    (0.0, 0.0, 0.0),
    0.0,
    0.0,
    False,
    False,
)

_STANDARD_SPEAR_STAB_SELECTOR = _StabSelectorProgram(
    runtime_seconds=0.111,
    start_distance=0.1,
    end_distance=3.0,
    extend_left=0.2,
    extend_right=0.2,
    extend_bottom=0.2,
    extend_top=0.2,
    yaw_offset_degrees=0.0,
    pitch_offset_degrees=0.0,
    roll_offset_degrees=0.0,
)

_DAGGER_PRIMARY_STAB_SELECTOR = _StabSelectorProgram(
    # Assets.zip:
    # Server/Item/Interactions/Weapons/Daggers/Primary/*/*_Selector.json
    runtime_seconds=0.069,
    start_distance=0.0,
    end_distance=2.0,
    extend_left=0.5,
    extend_right=0.5,
    extend_bottom=0.5,
    extend_top=0.5,
    yaw_offset_degrees=0.0,
    pitch_offset_degrees=0.0,
    roll_offset_degrees=0.0,
)

# Weapon_Daggers_Signature_Razorstrike runs Slash -> Sweep -> Lunge in series;
# each authors its own Stab selector. The Sweep is the wide one (ExtendLeft and
# ExtendRight 1.5 against 0.75 everywhere else).
_DAGGER_RAZORSTRIKE_SLASH_SELECTOR = _StabSelectorProgram(
    0.1, 0.1, 2.25, 0.75, 0.75, 0.75, 0.75, 0.0, 0.0, 0.0
)
_DAGGER_RAZORSTRIKE_SWEEP_SELECTOR = _StabSelectorProgram(
    0.1, 0.1, 2.25, 1.5, 1.5, 0.75, 0.75, 0.0, 0.0, 0.0
)
_DAGGER_RAZORSTRIKE_LUNGE_SELECTOR = _StabSelectorProgram(
    0.25, 0.1, 2.5, 0.75, 0.75, 0.75, 0.75, 0.0, 0.0, 0.0
)

# Weapon_Shield_Secondary_Guard_Bash_Selector.json -- note StartDistance 0.
_SHIELD_GUARD_BASH_SELECTOR = _StabSelectorProgram(
    0.083, 0.0, 2.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0
)

_VOID_SCYTHE_LEFT_SELECTOR = _HorizontalSelectorProgram(
    0.111, 0.1, 3.5, 0.5, 0.5, 90.0, 1.0, -60.0, 0.0, 0.0
)
_VOID_SCYTHE_RIGHT_SELECTOR = _HorizontalSelectorProgram(
    0.111, 0.1, 3.5, 0.5, 0.5, 90.0, -1.0, -60.0, 0.0, 0.0
)
_VOID_SCYTHE_CHARGED_SELECTOR = _HorizontalSelectorProgram(
    0.45, 0.1, 3.0, 0.5, 0.5, 120.0, 1.0, -60.0, 0.0, 0.0
)


class NativeInteractionBinding(NamedTuple):
    """One asset-derived bridge binding for a profile-local ability slot."""

    interaction_id: str
    interaction_type: str


class NativeProfileBindings(NamedTuple):
    """Sparse authored slots compacted onto the bridge's native slot list."""

    item_id: str
    abilities: tuple[NativeInteractionBinding, ...]
    authored_ability_slots: tuple[int, ...]
    guard: NativeInteractionBinding | None
    # One entry per authored profile ability. ``-1`` keeps private/internal
    # children unreachable; repeated values intentionally compact multiple
    # scalar choices onto one native selector root.
    authored_to_native_ability_slots: tuple[int, ...]
    # A fixed JAX channel can represent a family-local native EntityStat. The
    # bridge reads this profile data; it never switches on weapon names.
    resource_stat_ids: tuple[str, ...]


# Generated from the pinned Assets.zip equipped-interaction roots. An ability
# absent from these sets is deliberately not bound. The native adapter carries
# an explicit authored-slot map, so later resolved slots can be compacted onto
# the bridge list without changing policy slot semantics.
_NATIVE_ABILITY1_INTERACTIONS = frozenset(
    {
        "Weapon_Battleaxe_Signature_Whirlwind",
        "Weapon_Crossbow_Signature_BigArrow",
        "Weapon_Daggers_Signature_Razorstrike",
        "Weapon_Shortbow_Signature_Volley_Activate",
        "Weapon_Stick_Fire_Activate_Trap",
        "Weapon_Sword_Signature_Vortexstrike",
        "Bow_Bomb_Boomshot",
        "Bow_Combat_Trishot",
        "Bow_Pull_Pullshot",
        "Bow_Ricochet_Trishot",
        "Bow_Vamp_Supershot",
    }
)
_NATIVE_ABILITY3_INTERACTIONS = frozenset(
    {
        "Root_Common_StatAmmoReload_Entry",
    }
)
_NATIVE_SECONDARY_INTERACTIONS = frozenset(
    {
        "Ice_Staff_Primary_Entry",
        "Kunai_Throw",
        "Weapon_Shield_Secondary_Guard_Bash",
        "Gun_Attack",
    }
)
_NATIVE_PRIMARY_INTERACTIONS = frozenset(
    {
        "Bomb_Throw",
        "Bomb_Throw_Popberry",
        "Bomb_Throw_Stun",
        "Root_Cast",
        "Spear_Stab",
        "Spear_Throw_Charged",
        "Stoneskin_Cast",
        "Weapon_Battleaxe_Primary_Swing_Down",
        "Weapon_Battleaxe_Primary_Swing_Down_Left",
        "Weapon_Battleaxe_Primary_Swing_Down_Right",
        "Battleaxe_Swing_Left_Charged",
        "Block_Swing_Down",
        "Block_Swing_Left",
        "Block_Swing_Right",
        "Axe_Swing_Down_Left",
        "Axe_Swing_Left_Charged",
        "Axe_Swing_Up_Right",
        "Club_Flail_Spin_Swing_Left_Charged",
        "Club_Flail_Swing_Left",
        "Club_Flail_Swing_Right",
        "Club_Swing_Left",
        "Club_Swing_Right",
        "Longsword_Stab_Charged",
        "Longsword_Swing_Left",
        "Longsword_Swing_Right",
        "Longsword_Swing_Up_Left",
        "Knife_Lunge",
        "Knife_Stab",
        "Knife_Swing_Left",
        "Knife_Swing_Right",
        "Knife_Throw_Charged",
        "Daggers_Lunge_Double_Charged",
        "Daggers_Stab_Double_Charged",
        "Daggers_Swing_Left_Right",
        "Daggers_Swing_Right_Left",
        "Spear_Swing_Left",
        "Spear_Swing_Right",
        "Spellbook_Cast_Hurl_Charged",
        "Staff_Cast_Summon_Charged",
        "Sword_Swing_Left_Fast",
        "Sword_Swing_Right_Fast",
        "Wand_Cast_Left_Charged",
        "Bow_Combat_Shoot_Charging",
        "Bow_Ricochet_Shoot_Charging",
        "Bow_Shoot_Charging",
        "Gun_Shoot",
        "Gun_Shoot_Flintlock_Charging",
        "Weapon_Assault_Rifle_Primary",
        "Weapon_Handgun_Primary",
        "Weapon_Crossbow_Shot_Standard_Projectile",
        "Weapon_Crossbow_Signature_BigArrow_Shot",
        "Weapon_Daggers_Primary_Stab_Left",
        "Weapon_Daggers_Primary_Stab_Right",
        "Weapon_Daggers_Primary_Swing_Left",
        "Weapon_Daggers_Primary_Swing_Right",
        "Weapon_Mace_Primary_Swing_Left",
        "Weapon_Mace_Primary_Swing_Left_Charged",
        "Weapon_Mace_Primary_Swing_Right",
        "Weapon_Mace_Primary_Swing_Right_Charged",
        "Weapon_Mace_Primary_Swing_Up_Left",
        "Weapon_Mace_Primary_Swing_Up_Left_Charged",
        "Weapon_Shortbow_Primary_Shoot_Strength_0",
        "Weapon_Shortbow_Primary_Shoot_Strength_1",
        "Weapon_Shortbow_Primary_Shoot_Strength_2",
        "Weapon_Shortbow_Primary_Shoot_Strength_3",
        "Weapon_Shortbow_Primary_Shoot_Strength_4",
        "Weapon_Shortbow_Signature_Volley_Strength_0",
        "Weapon_Shortbow_Signature_Volley_Strength_1",
        "Weapon_Shortbow_Signature_Volley_Strength_2",
        "Weapon_Stick_Fire_Projectile_Charged_0",
        "Weapon_Stick_Fire_Projectile_Charged_1",
        "Weapon_Stick_Fire_Projectile_Charged_2",
        "Weapon_Stick_Fire_Projectile_Charged_3",
        "Weapon_Stick_Fire_Spawn_Trap",
        "Weapon_Sword_Primary_Swing_Down",
        "Weapon_Sword_Primary_Swing_Left",
        "Weapon_Sword_Primary_Swing_Right",
        "Weapon_Sword_Primary_Thrust",
        "Spear_Spin_Swing_Left",
        "Spear_Spin_Swing_Right",
    }
)
_NATIVE_GUARD_INTERACTIONS_BY_FAMILY = {
    FAMILY_SWORD: "Root_Weapon_Sword_Secondary_Guard",
    FAMILY_MACE: "Root_Weapon_Mace_Secondary_Guard",
    FAMILY_BATTLEAXE: "Root_Weapon_Battleaxe_Secondary_Guard",
    FAMILY_DAGGERS: "Root_Weapon_Daggers_Secondary_Guard",
    FAMILY_SHORTBOW: "Root_Weapon_Shortbow_Secondary_Guard",
    FAMILY_CROSSBOW: "Root_Weapon_Crossbow_Secondary_Guard",
    FAMILY_SHIELD: "Root_Weapon_Shield_Secondary_Guard",
    **{
        family: "Spear_Block"
        for family in range(FAMILY_SPEAR_ADAMANTITE, FAMILY_SPEAR_TRIBAL + 1)
    },
    **{
        row.family_id: {
            "sword": "Root_Weapon_Sword_Secondary_Guard",
            "mace": "Root_Weapon_Mace_Secondary_Guard",
            "battleaxe": "Root_Weapon_Battleaxe_Secondary_Guard",
            "daggers": "Root_Weapon_Daggers_Secondary_Guard",
            "shortbow": "Root_Weapon_Shortbow_Secondary_Guard",
            "crossbow": "Root_Weapon_Crossbow_Secondary_Guard",
            "shield": "Root_Weapon_Shield_Secondary_Guard",
        }[row.template]
        for row in SCALAR_VARIANTS
    },
    VOID_SCYTHE_FAMILY_ID: "Spear_Block",
    DOUBLE_INCANDESCENT_SPEAR_FAMILY_ID: "Spear_Block",
    TRIBAL_BLOWGUN_FAMILY_ID: "Root_Weapon_Sword_Secondary_Guard",
}
# Modern ProjectileConfig rows store a local SpawnOffset and rotation. Legacy
# BallisticData rows store horizontal/vertical/depth center-shot values; those
# are normalized here to local (x, y, z) before the runtime applies the
# source-matched rotation rule. Every entry is pinned by the asset audit.
_PROJECTILE_LAUNCH_TRANSFORMS = {
    PROJECTILE_NONE: _ZERO_PROJECTILE_LAUNCH,
    PROJECTILE_ARROW: _ProjectileLaunchTransform(
        (0.15, -0.25, 0.0),
        0.25,
        2.0,
        False,
        False,
    ),
    PROJECTILE_BIG_ARROW: _ProjectileLaunchTransform(
        (0.15, -0.25, 0.0),
        0.25,
        2.0,
        False,
        False,
    ),
    PROJECTILE_FIREBALL: _ProjectileLaunchTransform(
        (0.15, -0.25, -0.5),
        0.25,
        0.0,
        False,
        False,
    ),
    PROJECTILE_ICE_BALL: _ZERO_PROJECTILE_LAUNCH,
    PROJECTILE_ICE_BOLT: _ZERO_PROJECTILE_LAUNCH,
    PROJECTILE_BOMB: _ProjectileLaunchTransform(
        (0.3, 0.1, 0.0),
        1.0,
        10.0,
        False,
        False,
    ),
    PROJECTILE_CORRUPTION_ORB: _ProjectileLaunchTransform(
        (-0.25, -0.1, -1.0),
        0.0,
        0.0,
        True,
        True,
    ),
    PROJECTILE_KUNAI: _ProjectileLaunchTransform(
        (-0.15, -0.25, -1.0),
        -0.5,
        2.5,
        False,
        False,
    ),
    PROJECTILE_SPEAR: _ProjectileLaunchTransform(
        (0.1, 0.0, -1.0),
        0.0,
        0.0,
        True,
        False,
    ),
    PROJECTILE_LEGACY_ARROW: _ProjectileLaunchTransform(
        (0.1, -0.1, -1.0),
        0.0,
        0.0,
        True,
        True,
    ),
    PROJECTILE_PROTOTYPE_ARROW: _ProjectileLaunchTransform(
        (0.0, 0.0, -0.5),
        0.25,
        2.0,
        False,
        False,
    ),
    PROJECTILE_GUN_BULLET: _ProjectileLaunchTransform(
        (0.0, 0.0, -0.7),
        0.0,
        0.0,
        True,
        False,
    ),
    PROJECTILE_BLUNDERBUSS_BULLET: _ProjectileLaunchTransform(
        (0.0, 0.0, -1.0),
        0.0,
        0.0,
        True,
        False,
    ),
}


def _instant_add_stat_cost(
    amount: float,
    *,
    commit_time_seconds: float = 0.0,
    commit_flags: int = 0,
) -> _ResourceCost:
    return _ResourceCost(
        amount,
        RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT,
        commit_time_seconds,
        commit_flags,
    )


def _stamina_break_immune_instant_add_stat_cost(
    amount: float,
    *,
    commit_time_seconds: float = 0.0,
    commit_flags: int = 0,
) -> _ResourceCost:
    """Encode ApplyEffect(immune) before native NPC ChangeStat."""

    return _ResourceCost(
        amount,
        RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT_STAMINA_BREAK_IMMUNE,
        commit_time_seconds,
        commit_flags,
    )


def _single_application_cost(
    amount: float,
    *,
    commit_time_seconds: float = 0.0,
    commit_flags: int = 0,
) -> _ResourceCost:
    return _ResourceCost(
        amount,
        RESOURCE_COST_SINGLE_APPLICATION,
        commit_time_seconds,
        commit_flags,
    )


def _single_application_stamina_break_immune_cost(
    amount: float,
    *,
    commit_time_seconds: float = 0.0,
    commit_flags: int = 0,
) -> _ResourceCost:
    """Encode a remote-client ChangeStat after stamina-break immunity."""

    return _ResourceCost(
        amount,
        RESOURCE_COST_SINGLE_APPLICATION_STAMINA_BREAK_IMMUNE,
        commit_time_seconds,
        commit_flags,
    )


def _compile_resource_scalars(
    scalars: WeaponScalarSet,
    bindings: Sequence[_ResourceScalarBinding],
) -> tuple[dict[int, _ResourceCost], dict[int, float]]:
    """Compile named scalars without assuming a particular weapon family."""

    costs: dict[int, _ResourceCost] = {}
    minimums: dict[int, float] = {}
    for binding in bindings:
        if not 0 <= binding.resource_id < RESOURCE_COUNT:
            raise ValueError("resource scalar binding ID is outside capacity")
        if binding.resource_id in costs:
            raise ValueError("resource scalar bindings must target unique resources")
        if not np.isfinite(binding.commit_time_seconds) or (
            binding.commit_time_seconds < 0.0
        ):
            raise ValueError(
                "resource scalar commit time must be finite and non-negative"
            )
        amount = scalars.value(binding.scalar_name)
        if amount < 0.0:
            raise ValueError("authored resource scalar costs must be non-negative")
        if amount == 0.0:
            continue
        costs[binding.resource_id] = _ResourceCost(
            amount,
            binding.cost_kind,
            binding.commit_time_seconds,
            binding.commit_flags,
        )
        if binding.admission_required:
            minimums[binding.resource_id] = amount
    return costs, minimums


def hytale_0_5_7_loadouts(
    agent_profiles: str | Sequence[str],
    *,
    target_profiles: str | Sequence[str] | None = None,
):
    """Build heterogeneous batch-first programs from the pinned catalog."""

    agents = _names(agent_profiles)
    targets = (
        [""] * len(agents)
        if target_profiles is None
        else _names(target_profiles, expected=len(agents))
    )
    return hytale_0_5_7_entity_loadouts(
        tuple((agent, target) for agent, target in zip(agents, targets, strict=True))
    )


def hytale_0_5_7_entity_loadouts(
    entity_profiles: Sequence[Sequence[str]],
):
    """Build one authored profile per entity for every environment row."""

    rows = [list(row) for row in entity_profiles]
    if not rows:
        raise ValueError("at least one environment row is required")
    entity_count = len(rows[0])
    if entity_count < 2:
        raise ValueError("each environment row requires at least two entities")
    if any(len(row) != entity_count for row in rows):
        raise ValueError("every environment row must have the same entity count")
    if any(not isinstance(name, str) for row in rows for name in row):
        raise TypeError("profile names must be strings")
    indices = np.asarray(
        [[_profile_index(name) for name in row] for row in rows],
        dtype=np.int32,
    )
    bank = _profile_bank()
    selected = type(bank)(*(getattr(bank, field)[indices] for field in bank._fields))
    return jax.device_put(selected)


def native_ability_requested_charge_time_seconds(
    loadout: AbilityLoadout,
    ability_slot: int,
    *,
    batch_index: int = 0,
    source_entity_index: int = 0,
) -> float:
    """Return only an explicitly authored outer-Charging hold request.

    ``requestedChargeTime`` is protocol input to ``ChargingInteraction``.  It
    is not the selected child's run time, event time, or resource-commit time.
    A non-negative compiled value therefore wins verbatim, including zero;
    the negative-one sentinel means that the selected root has no outer charge
    request and transports zero.  No item or weapon family is special.
    """

    if not isinstance(loadout, AbilityLoadout):
        raise TypeError("loadout must be an AbilityLoadout")
    indices = {
        "ability_slot": ability_slot,
        "batch_index": batch_index,
        "source_entity_index": source_entity_index,
    }
    for name, value in indices.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")

    slot = int(ability_slot)
    batch = int(batch_index)
    entity = int(source_entity_index)
    event_mask = np.asarray(loadout.ability_event_mask, dtype=np.bool_)
    event_times = np.asarray(loadout.event_time_seconds, dtype=np.float32)
    if event_mask.ndim != 4 or event_times.shape != event_mask.shape:
        raise ValueError("loadout ability event fields must share shape (B, N, A, E)")
    if not 0 <= batch < event_mask.shape[0]:
        raise IndexError("batch_index is out of range")
    if not 0 <= entity < event_mask.shape[1]:
        raise IndexError("source_entity_index is out of range")
    if not 0 <= slot < event_mask.shape[2]:
        return 0.0

    mask = event_mask[batch, entity, slot]
    if not np.any(mask):
        return 0.0

    requested_charge_time = np.asarray(
        loadout.ability_requested_charge_time_seconds,
        dtype=np.float32,
    )[batch, entity, slot]
    if requested_charge_time >= np.float32(0.0):
        return float(requested_charge_time)

    return 0.0


def hytale_0_5_7_catalog_counts() -> dict[str, int]:
    """Return derived counts; no manifest value is maintained by hand."""

    profiles = [_CATALOG[name]() for name in PROFILE_NAMES]
    return {
        "profile_count": len(profiles),
        "ability_count": sum(len(profile["abilities"]) for profile in profiles),
        "event_count": sum(
            len(ability["events"])
            for profile in profiles
            for ability in profile["abilities"]
        ),
        "world_conditional_deferred_count": len(WORLD_CONDITIONAL_ABILITY_ASSETS),
    }


def ability_loadout_content_sha256(loadout: AbilityLoadout) -> str:
    """Hash every typed program field independently of Python source layout."""

    if not isinstance(loadout, AbilityLoadout):
        raise TypeError("loadout must be an AbilityLoadout")
    digest = hashlib.sha256()

    def take(value: bytes) -> None:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)

    take(b"hytalerl_ability_loadout_content_v1")
    for field, raw_value in zip(loadout._fields, loadout, strict=True):
        value = np.asarray(raw_value)
        if value.dtype.hasobject:
            raise TypeError(f"loadout field {field} has an object dtype")
        dtype = (
            value.dtype
            if value.dtype.byteorder == "|"
            else value.dtype.newbyteorder("<")
        )
        canonical = np.ascontiguousarray(value.astype(dtype, copy=False))
        take(field.encode("utf-8"))
        take(canonical.dtype.str.encode("ascii"))
        take(",".join(str(size) for size in canonical.shape).encode("ascii"))
        take(canonical.tobytes(order="C"))
    return digest.hexdigest().upper()


@lru_cache(maxsize=1)
def hytale_0_5_7_program_content_sha256() -> str:
    """Return the resolved 0.5.7 program-bank identity used by checkpoints."""

    return ability_loadout_content_sha256(_profile_bank())


@lru_cache(maxsize=None)
def hytale_0_5_7_native_profile_bindings(
    profile: str,
) -> NativeProfileBindings:
    """Return asset-certified native bindings with authored slot identities.

    Some compiled programs are internal interactions not reachable from a
    shipped item's equipped root. Those slots stay unbound; later independently
    resolved slots retain their authored indices through the explicit mapping.
    """

    if not isinstance(profile, str):
        raise TypeError("profile must be a string")
    try:
        authored = _CATALOG[profile]()
    except KeyError as error:
        raise ValueError(
            f"unknown 0.5.7 combat profile {profile!r}; choose from {PROFILE_NAMES}"
        ) from error
    bindings: list[NativeInteractionBinding] = []
    authored_slots: list[int] = []
    authored_to_native = [-1] * len(authored["abilities"])
    native_slot_by_root: dict[tuple[str, str], int] = {}
    for slot, ability in enumerate(authored["abilities"]):
        if not ability.get("policy_selectable", True) or not ability.get(
            "native_bindable", True
        ):
            continue
        interaction_id = ability.get("native_outer_root_id") or (
            _native_interaction_id(ability["asset"])
        )
        interaction_type = _interaction_type_name(ability["interaction_type"])
        if interaction_type is None:
            continue
        key = (interaction_id, interaction_type)
        native_slot = native_slot_by_root.get(key)
        if native_slot is None:
            native_slot = len(bindings)
            native_slot_by_root[key] = native_slot
            bindings.append(NativeInteractionBinding(interaction_id, interaction_type))
            authored_slots.append(slot)
        authored_to_native[slot] = native_slot
    guard_id = _NATIVE_GUARD_INTERACTIONS_BY_FAMILY.get(authored["family"])
    guard = (
        None if guard_id is None else NativeInteractionBinding(guard_id, "Secondary")
    )
    return NativeProfileBindings(
        item_id=authored["asset"] if bindings or guard is not None else "",
        abilities=tuple(bindings),
        authored_ability_slots=tuple(authored_slots),
        guard=guard,
        authored_to_native_ability_slots=tuple(authored_to_native),
        resource_stat_ids=tuple(authored["native_resource_stat_ids"]),
    )


def hytale_0_5_7_runtime_capacity(
    profiles: str | Sequence[str] = PROFILE_NAMES,
) -> ArsenalRuntimeCapacity:
    """Return the smallest lossless execution shape for selected profiles."""

    names = _names(profiles)
    authored = [_CATALOG[name]() for name in names if name]
    if not authored:
        return ArsenalRuntimeCapacity(1, 1, 1)
    return ArsenalRuntimeCapacity(
        abilities_per_entity=max(len(profile["abilities"]) for profile in authored),
        events_per_entity=max(
            sum(len(ability["events"]) for ability in profile["abilities"])
            for profile in authored
        ),
        events_per_ability=max(
            len(ability["events"])
            for profile in authored
            for ability in profile["abilities"]
        ),
    )


def _native_interaction_type(interaction_id: str) -> str | None:
    memberships = (
        interaction_id in _NATIVE_PRIMARY_INTERACTIONS,
        interaction_id in _NATIVE_SECONDARY_INTERACTIONS,
        interaction_id in _NATIVE_ABILITY1_INTERACTIONS,
        interaction_id in _NATIVE_ABILITY3_INTERACTIONS,
    )
    if sum(memberships) > 1:
        raise AssertionError(
            f"native interaction type is ambiguous for {interaction_id!r}"
        )
    if memberships[0]:
        return "Primary"
    if memberships[1]:
        return "Secondary"
    if memberships[2]:
        return "Ability1"
    if memberships[3]:
        return "Ability3"
    return None


def _native_interaction_id(ability_asset_id: str) -> str:
    """Map compiled ProjectileConfig leaves back to their equipped charge root."""

    roots = (
        ("Projectile_Config_Bow_Combat_Charge_", "Bow_Combat_Shoot_Charging"),
        (
            "Projectile_Config_Bow_Ricochet_Charge_",
            "Bow_Ricochet_Shoot_Charging",
        ),
        ("Projectile_Config_Bow_Vamp_Charge_", "Bow_Shoot_Charging"),
    )
    for prefix, root in roots:
        if ability_asset_id.startswith(prefix):
            return root
    if ability_asset_id in {
        "Weapon_Assault_Rifle_Primary",
        "Weapon_Handgun_Primary",
    }:
        return "Gun_Shoot"
    return ability_asset_id


def _profile_index(name: str) -> int:
    if not name:
        return 0
    try:
        return _PROFILE_INDEX[name]
    except KeyError as error:
        raise ValueError(
            f"unknown 0.5.7 combat profile {name!r}; choose from {PROFILE_NAMES}"
        ) from error


class _MutableProfileArray:
    """Writable NumPy adapter for ``array.at[index].set(value)`` calls."""

    def __init__(self, value: object) -> None:
        self.value = np.array(value, copy=True)
        self._index: object = ()

    @property
    def at(self) -> _MutableProfileArray:
        return self

    def __getitem__(self, index: object) -> _MutableProfileArray:
        self._index = index
        return self

    def set(self, value: object) -> _MutableProfileArray:
        self.value[self._index] = np.asarray(value)
        return self


@lru_cache(maxsize=1)
def _profile_bank():
    """Cache fixed profiles on the host; upload only selected episode rows."""

    template = empty_ability_loadout(1)
    template_arrays = {
        field: np.asarray(getattr(template, field)) for field in template._fields
    }
    rows: list[dict[str, np.ndarray]] = []
    for name in ("", *PROFILE_NAMES):
        arrays = {
            field: _MutableProfileArray(value)
            for field, value in template_arrays.items()
        }
        if name:
            _write_profile(arrays, 0, 0, _CATALOG[name]())
        rows.append({field: arrays[field].value[0, 0] for field in template._fields})
    return type(template)(
        *(np.stack([row[field] for row in rows], axis=0) for field in template._fields)
    )


def _write_profile(arrays, batch: int, entity: int, profile: dict) -> None:
    abilities = profile["abilities"]
    overflow = len(abilities) > ABILITY_CAPACITY or any(
        len(ability["events"]) > EVENT_CAPACITY
        or len(ability.get("charge_times", (0.0,))) > ABILITY_CHARGE_CAPACITY
        for ability in abilities
    )
    arrays["weapon_id"] = (
        arrays["weapon_id"].at[batch, entity].set(semantic_id(profile["asset"]))
    )
    arrays["weapon_family"] = (
        arrays["weapon_family"].at[batch, entity].set(profile["family"])
    )
    arrays["equipped"] = arrays["equipped"].at[batch, entity].set(True)
    guard = profile["guard"]
    for field, value in (
        ("guard_entry_cost", guard["entry_cost"]),
        ("guard_stamina_value", guard["stamina_value"]),
        ("guard_half_angle_degrees", guard["half_angle_degrees"]),
        ("guard_entry_delay_seconds", guard["entry_delay_seconds"]),
        (
            "guard_exit_regen_delay_seconds",
            guard["exit_regen_delay_seconds"],
        ),
        ("guard_required_resource_id", guard["required_resource_id"]),
        (
            "guard_required_resource_minimum",
            guard["required_resource_minimum"],
        ),
        ("guard_interrupting_type_mask", guard["interrupting_type_mask"]),
    ):
        arrays[field] = arrays[field].at[batch, entity].set(value)
    for resource_id, maximum in profile.get("resources", {}).items():
        arrays["resource_maximum"] = (
            arrays["resource_maximum"].at[batch, entity, resource_id].set(maximum)
        )
    for resource_id, initial in profile.get("initial_resources", {}).items():
        arrays["resource_initial"] = (
            arrays["resource_initial"].at[batch, entity, resource_id].set(initial)
        )
    arrays["overflow"] = arrays["overflow"].at[batch, entity].set(overflow)
    event_cursor = 0
    for ability_slot, ability in enumerate(abilities[:ABILITY_CAPACITY]):
        key = (batch, entity, ability_slot)
        arrays["ability_id"] = (
            arrays["ability_id"].at[key].set(semantic_id(ability["asset"]))
        )
        arrays["ability_interaction_type"] = (
            arrays["ability_interaction_type"].at[key].set(ability["interaction_type"])
        )
        arrays["ability_guard_fork_type"] = (
            arrays["ability_guard_fork_type"].at[key].set(ability["guard_fork_type"])
        )
        arrays["ability_mask"] = arrays["ability_mask"].at[key].set(not overflow)
        arrays["ability_evidence"] = (
            arrays["ability_evidence"]
            .at[key]
            .set(ability.get("evidence", EVIDENCE_ASSET_RESOLVED))
        )
        arrays["ability_duration_seconds"] = (
            arrays["ability_duration_seconds"].at[key].set(ability["duration"])
        )
        arrays["ability_cooldown_seconds"] = (
            arrays["ability_cooldown_seconds"].at[key].set(ability.get("cooldown", 0.0))
        )
        arrays["ability_requested_charge_time_seconds"] = (
            arrays["ability_requested_charge_time_seconds"]
            .at[key]
            .set(ability.get("requested_charge_time_seconds", -1.0))
        )
        charge_times = tuple(ability.get("charge_times", (0.0,)))
        arrays["ability_charge_capacity"] = (
            arrays["ability_charge_capacity"].at[key].set(len(charge_times))
        )
        arrays["ability_interrupt_recharge"] = (
            arrays["ability_interrupt_recharge"]
            .at[key]
            .set(ability.get("interrupt_recharge", False))
        )
        arrays["ability_interrupted_by_type_mask"] = (
            arrays["ability_interrupted_by_type_mask"]
            .at[key]
            .set(ability.get("interrupted_by_type_mask", 0))
        )
        arrays["ability_interruptible_after_seconds"] = (
            arrays["ability_interruptible_after_seconds"]
            .at[key]
            .set(ability.get("interruptible_after_seconds", 0.0))
        )
        for charge_index, charge_time in enumerate(
            charge_times[:ABILITY_CHARGE_CAPACITY]
        ):
            arrays["ability_charge_times_seconds"] = (
                arrays["ability_charge_times_seconds"]
                .at[key + (charge_index,)]
                .set(charge_time)
            )
        hold_table = tuple(ability.get("hold_table", ()))
        if len(hold_table) > ABILITY_HOLD_CAPACITY:
            raise ValueError(
                "charging hold table exceeds ABILITY_HOLD_CAPACITY"
            )
        if any(
            later[0] <= earlier[0]
            for earlier, later in zip(hold_table, hold_table[1:])
        ):
            raise ValueError(
                "charging hold thresholds must ascend; selection reads the "
                "last reached row and a misordered table silently picks the "
                "wrong child"
            )
        arrays["ability_hold_count"] = (
            arrays["ability_hold_count"].at[key].set(len(hold_table))
        )
        arrays["ability_hold_allow_indefinite"] = (
            arrays["ability_hold_allow_indefinite"]
            .at[key]
            .set(bool(ability.get("hold_allow_indefinite", False)))
        )
        arrays["ability_hold_speed_multiplier"] = (
            arrays["ability_hold_speed_multiplier"]
            .at[key]
            .set(float(ability.get("hold_speed_multiplier", 1.0)))
        )
        arrays["ability_hold_speed_multiplier_after_seconds"] = (
            arrays["ability_hold_speed_multiplier_after_seconds"]
            .at[key]
            .set(float(ability.get("hold_speed_multiplier_after_seconds", 0.0)))
        )
        arrays["ability_continuation_mode"] = (
            arrays["ability_continuation_mode"]
            .at[key]
            .set(int(ability.get("continuation_mode", 0)))
        )
        arrays["ability_continuation_ground_slot"] = (
            arrays["ability_continuation_ground_slot"]
            .at[key]
            .set(int(ability.get("continuation_ground_slot", -1)))
        )
        arrays["ability_continuation_collision_slot"] = (
            arrays["ability_continuation_collision_slot"]
            .at[key]
            .set(int(ability.get("continuation_collision_slot", -1)))
        )
        arrays["ability_continuation_ground_check_delay_seconds"] = (
            arrays["ability_continuation_ground_check_delay_seconds"]
            .at[key]
            .set(float(ability.get("continuation_ground_check_delay", 0.0)))
        )
        arrays["ability_continuation_run_time_seconds"] = (
            arrays["ability_continuation_run_time_seconds"]
            .at[key]
            .set(float(ability.get("continuation_run_time", -1.0)))
        )
        for hold_index, (threshold, child) in enumerate(hold_table):
            arrays["ability_hold_threshold_seconds"] = (
                arrays["ability_hold_threshold_seconds"]
                .at[key + (hold_index,)]
                .set(threshold)
            )
            arrays["ability_hold_child_slot"] = (
                arrays["ability_hold_child_slot"]
                .at[key + (hold_index,)]
                .set(child)
            )
        arrays["ability_stamina_regen_delay_seconds"] = (
            arrays["ability_stamina_regen_delay_seconds"]
            .at[key]
            .set(ability.get("stamina_delay", 0.0))
        )
        arrays["ability_stamina_regen_delay_start_tick"] = (
            arrays["ability_stamina_regen_delay_start_tick"]
            .at[key]
            .set(ability.get("stamina_delay_start_tick", 0))
        )
        arrays["ability_stamina_regen_delay_end_tick"] = (
            arrays["ability_stamina_regen_delay_end_tick"]
            .at[key]
            .set(ability.get("stamina_delay_end_tick", -1))
        )
        arrays["ability_scheduler_prelude_ticks"] = (
            arrays["ability_scheduler_prelude_ticks"]
            .at[key]
            .set(ability.get("scheduler_prelude_ticks", 0))
        )
        # A native binding may need to enter through a public item root so the
        # engine can run its own state selector.  That does not necessarily
        # mean the JAX child has an outer-root inventory/nock lifecycle of its
        # own.  Only an explicit dispatch tick enables that simulated
        # lifecycle; the binding root remains transport metadata.
        outer_root_item_dispatch_tick = ability.get("outer_root_item_dispatch_tick", -1)
        arrays["ability_outer_root_selector"] = (
            arrays["ability_outer_root_selector"]
            .at[key]
            .set(outer_root_item_dispatch_tick >= 0)
        )
        arrays["ability_outer_root_item_dispatch_tick"] = (
            arrays["ability_outer_root_item_dispatch_tick"]
            .at[key]
            .set(outer_root_item_dispatch_tick)
        )
        for resource_id, phase in ability.get("resource_phases", {}).items():
            phase_key = key + (resource_id,)
            arrays["ability_resource_phase_mask"] = (
                arrays["ability_resource_phase_mask"].at[phase_key].set(True)
            )
            arrays["ability_resource_phase_value"] = (
                arrays["ability_resource_phase_value"].at[phase_key].set(phase.value)
            )
            arrays["ability_resource_phase_inactive_value"] = (
                arrays["ability_resource_phase_inactive_value"]
                .at[phase_key]
                .set(phase.inactive_value)
            )
            arrays["ability_resource_phase_start_tick"] = (
                arrays["ability_resource_phase_start_tick"]
                .at[phase_key]
                .set(phase.start_tick)
            )
            arrays["ability_resource_phase_end_tick"] = (
                arrays["ability_resource_phase_end_tick"]
                .at[phase_key]
                .set(phase.end_tick)
            )
        for resource_id, cost in ability.get("cost", {}).items():
            arrays["ability_resource_cost"] = (
                arrays["ability_resource_cost"]
                .at[key + (resource_id,)]
                .set(cost.amount)
            )
            arrays["ability_resource_cost_kind"] = (
                arrays["ability_resource_cost_kind"]
                .at[key + (resource_id,)]
                .set(cost.kind)
            )
            arrays["ability_resource_commit_time_seconds"] = (
                arrays["ability_resource_commit_time_seconds"]
                .at[key + (resource_id,)]
                .set(cost.commit_time_seconds)
            )
            arrays["ability_resource_commit_flags"] = (
                arrays["ability_resource_commit_flags"]
                .at[key + (resource_id,)]
                .set(cost.commit_flags)
            )
        for resource_id, minimum in ability.get(
            "minimum", ability.get("cost", {})
        ).items():
            arrays["ability_resource_minimum"] = (
                arrays["ability_resource_minimum"].at[key + (resource_id,)].set(minimum)
            )
        requirements = ability.get("requirements", 0)
        arrays["ability_requirements"] = (
            arrays["ability_requirements"].at[key].set(requirements)
        )
        arrays["ability_static_placement_maximum_distance"] = (
            arrays["ability_static_placement_maximum_distance"]
            .at[key]
            .set(ability.get("static_placement_maximum_distance", 0.0))
        )
        arrays["ability_static_placement_allow_walls"] = (
            arrays["ability_static_placement_allow_walls"]
            .at[key]
            .set(ability.get("static_placement_allow_walls", False))
        )
        arrays["ability_event_count"] = (
            arrays["ability_event_count"]
            .at[key]
            .set(min(len(ability["events"]), EVENT_CAPACITY))
        )
        arrays["ability_event_start"] = (
            arrays["ability_event_start"].at[key].set(event_cursor)
        )
        event_cursor += min(len(ability["events"]), EVENT_CAPACITY)
        for event_slot, event in enumerate(ability["events"][:EVENT_CAPACITY]):
            event_key = key + (event_slot,)
            arrays["event_mask"] = arrays["event_mask"].at[event_key].set(True)
            arrays["ability_event_mask"] = (
                arrays["ability_event_mask"].at[event_key].set(True)
            )
            arrays["event_time_seconds"] = (
                arrays["event_time_seconds"].at[event_key].set(event["time"])
            )
            arrays["event_kind"] = arrays["event_kind"].at[event_key].set(event["kind"])
            for index, value in event.get("f", {}).items():
                arrays["event_f32"] = (
                    arrays["event_f32"].at[event_key + (index,)].set(value)
                )
            for index, value in event.get("i", {}).items():
                arrays["event_i32"] = (
                    arrays["event_i32"].at[event_key + (index,)].set(value)
                )
            arrays["event_flags"] = (
                arrays["event_flags"].at[event_key].set(event.get("flags", 0))
            )


def _continuation_mode(wait_for: tuple[str, ...]) -> int:
    """Fold the authored wait conditions into the C4 mode bitmask."""

    known = {
        "ground": CONTINUATION_WAIT_FOR_GROUND,
        "collision": CONTINUATION_WAIT_FOR_COLLISION,
    }
    unknown = sorted(set(wait_for) - set(known))
    if unknown:
        raise ValueError(f"unsupported continuation conditions: {unknown}")
    mode = CONTINUATION_NONE
    for name in wait_for:
        mode |= known[name]
    return mode


def _ability(
    asset: str,
    duration: float,
    *events: dict,
    cost: dict[int, _ResourceCost] | None = None,
    minimum: dict[int, float] | None = None,
    cooldown: float = 0.0,
    requested_charge_time_seconds: float = -1.0,
    charge_times: tuple[float, ...] = (0.0,),
    # C1 ChargeTable. ``hold_table`` is the authored "Type": "Charging" root as
    # ascending ``(hold_seconds, child_ability_index)`` pairs; the child with the
    # highest threshold at or below the held duration runs on release. Empty
    # means the ability is not a charging root. Author the *flattened*
    # thresholds -- nested Charging roots whose early branches share one target
    # collapse to a single table. See ``programs/charge.py``.
    hold_table: tuple[tuple[float, int], ...] = (),
    # AllowIndefiniteHold. False (the default, and what Sword/Daggers/
    # Battleaxe author) makes the root fire the instant its clock reaches the
    # top threshold; True holds until the control comes up, as the Spear does.
    hold_allow_indefinite: bool = False,
    # C2 HorizontalSpeedMultiplier: planar movement is scaled to this while the
    # root is held. 1.0 means the authored root sets none.
    hold_speed_multiplier: float = 1.0,
    # Nested pairs author the multiplier on the inner root only, so the outer
    # gate runs at full speed. See _NESTED_OUTER_GATE_SECONDS.
    hold_speed_multiplier_after_seconds: float = 0.0,
    # C4 Continuation. ``wait_for`` names the authored world conditions the
    # ApplyForce parks on -- any of ``("ground",)``, ``("collision",)`` or both
    # -- and the two ``*_next`` slots are the children each condition runs.
    # ``ground_check_delay`` and ``run_time`` carry the authored
    # ``GroundCheckDelay`` and ``RunTime``; ``run_time=None`` means the wait has
    # no authored ceiling. Empty ``wait_for`` is every ordinary ability.
    # See ``programs/continuation.py``.
    wait_for: tuple[str, ...] = (),
    ground_next: int = -1,
    collision_next: int = -1,
    ground_check_delay: float = 0.0,
    run_time: float | None = None,
    interrupt_recharge: bool = False,
    interrupted_by_type_mask: int = 0,
    interruptible_after_seconds: float = 0.0,
    stamina_delay: float = 0.0,
    stamina_delay_start_tick: int = 0,
    stamina_delay_end_tick: int = -1,
    scheduler_prelude_ticks: int = 0,
    native_outer_root_id: str = "",
    outer_root_item_dispatch_tick: int = -1,
    resource_phases: dict[int, _ResourcePhase] | None = None,
    requirements: int = 0,
    static_placement_maximum_distance: float = 0.0,
    static_placement_allow_walls: bool = False,
    interaction_type: int | None = None,
    guard_fork_type: int = INTERACTION_TYPE_UNRESOLVED,
    evidence: int = EVIDENCE_ASSET_RESOLVED,
    policy_selectable: bool = True,
    native_bindable: bool = True,
) -> dict:
    if isinstance(scheduler_prelude_ticks, bool) or not isinstance(
        scheduler_prelude_ticks,
        int,
    ):
        raise TypeError("scheduler_prelude_ticks must be an integer")
    if scheduler_prelude_ticks < 0:
        raise ValueError("scheduler_prelude_ticks must be non-negative")
    if not isinstance(native_outer_root_id, str):
        raise TypeError("native_outer_root_id must be a string")
    if isinstance(outer_root_item_dispatch_tick, bool) or not isinstance(
        outer_root_item_dispatch_tick,
        int,
    ):
        raise TypeError("outer_root_item_dispatch_tick must be an integer")
    if outer_root_item_dispatch_tick < -1:
        raise ValueError("outer_root_item_dispatch_tick must be -1 or non-negative")
    phases = dict(resource_phases or {})
    for resource_id, phase in phases.items():
        if (
            isinstance(resource_id, bool)
            or not isinstance(resource_id, int)
            or not 0 <= resource_id < RESOURCE_COUNT
        ):
            raise ValueError("resource phase ID is outside resource capacity")
        if not isinstance(phase, _ResourcePhase):
            raise TypeError("resource phases must contain _ResourcePhase values")
        if not np.isfinite(phase.value) or not np.isfinite(phase.inactive_value):
            raise ValueError("resource phase values must be finite")
        if (
            isinstance(phase.start_tick, bool)
            or isinstance(phase.end_tick, bool)
            or not isinstance(phase.start_tick, int)
            or not isinstance(phase.end_tick, int)
            or phase.start_tick < 0
            or phase.end_tick <= phase.start_tick
        ):
            raise ValueError(
                "resource phase ticks must form a non-empty non-negative interval"
            )
    if outer_root_item_dispatch_tick >= 0 and not native_outer_root_id:
        raise ValueError(
            "an outer-root item-dispatch tick requires a native binding root"
        )
    if phases and outer_root_item_dispatch_tick < 0:
        raise ValueError("resource phases require an outer-root dispatch tick")
    authored_cost = cost or {}
    cost_amount = {
        resource_id: value.amount for resource_id, value in authored_cost.items()
    }
    if any(
        event["kind"] == EVENT_MELEE_CONE
        and not (event.get("flags", 0) & EVENT_FLAG_SELECTOR_IGNORES_LINE_OF_SIGHT)
        for event in events
    ):
        requirements |= REQUIRE_LINE_OF_SIGHT
    if any(event.get("flags", 0) & EVENT_FLAG_INJECTED_SELECTOR for event in events):
        requirements |= REQUIRE_INJECTED_SELECTOR
    if any(
        abs(event.get("f", {}).get(EF_FORCE_MAGNITUDE, 0.0)) > 0.0 for event in events
    ):
        requirements |= REQUIRE_CLEAR_FORCE_PATH
    if (
        isinstance(guard_fork_type, bool)
        or not isinstance(guard_fork_type, int)
        or guard_fork_type < INTERACTION_TYPE_UNRESOLVED
        or guard_fork_type >= INTERACTION_TYPE_COUNT
    ):
        raise ValueError("guard_fork_type must be unresolved or a protocol ordinal")
    if interaction_type is None and guard_fork_type >= 0:
        interaction_type = guard_fork_type
    if interaction_type is None:
        interaction_type = _interaction_type_index(
            _native_interaction_type(_native_interaction_id(asset))
        )
    if (
        isinstance(interaction_type, bool)
        or not isinstance(interaction_type, int)
        or interaction_type < INTERACTION_TYPE_UNRESOLVED
        or interaction_type >= INTERACTION_TYPE_COUNT
    ):
        raise ValueError("interaction_type must be unresolved or a protocol ordinal")
    if not isinstance(policy_selectable, bool):
        raise TypeError("policy_selectable must be a bool")
    if not isinstance(native_bindable, bool):
        raise TypeError("native_bindable must be a bool")
    if not np.isfinite(requested_charge_time_seconds) or (
        requested_charge_time_seconds < -1.0
    ):
        raise ValueError(
            "requested_charge_time_seconds must be -1 or finite and non-negative"
        )
    if (
        isinstance(interrupted_by_type_mask, bool)
        or not isinstance(interrupted_by_type_mask, int)
        or interrupted_by_type_mask < 0
        or interrupted_by_type_mask >= (1 << INTERACTION_TYPE_COUNT)
    ):
        raise ValueError(
            "interrupted_by_type_mask must contain only protocol type bits"
        )
    if not np.isfinite(interruptible_after_seconds) or (
        interruptible_after_seconds < 0.0
    ):
        raise ValueError("interruptible_after_seconds must be finite and non-negative")
    if interrupted_by_type_mask == 0 and interruptible_after_seconds != 0.0:
        raise ValueError("interruptible_after_seconds requires an interrupted-by mask")
    if interrupted_by_type_mask != 0 and interruptible_after_seconds > duration:
        raise ValueError("interruptible_after_seconds cannot exceed ability duration")
    return {
        "asset": asset,
        "interaction_type": interaction_type,
        "guard_fork_type": guard_fork_type,
        "duration": duration,
        "cooldown": cooldown,
        "requested_charge_time_seconds": requested_charge_time_seconds,
        "charge_times": charge_times,
        "hold_table": hold_table,
        "hold_allow_indefinite": hold_allow_indefinite,
        "hold_speed_multiplier": hold_speed_multiplier,
        "hold_speed_multiplier_after_seconds": hold_speed_multiplier_after_seconds,
        "continuation_mode": _continuation_mode(wait_for),
        "continuation_ground_slot": ground_next,
        "continuation_collision_slot": collision_next,
        "continuation_ground_check_delay": ground_check_delay,
        "continuation_run_time": -1.0 if run_time is None else run_time,
        "interrupt_recharge": interrupt_recharge,
        "interrupted_by_type_mask": interrupted_by_type_mask,
        "interruptible_after_seconds": interruptible_after_seconds,
        "stamina_delay": stamina_delay,
        "stamina_delay_start_tick": stamina_delay_start_tick,
        "stamina_delay_end_tick": stamina_delay_end_tick,
        "scheduler_prelude_ticks": scheduler_prelude_ticks,
        "native_outer_root_id": native_outer_root_id,
        "outer_root_item_dispatch_tick": outer_root_item_dispatch_tick,
        "resource_phases": phases,
        "cost": authored_cost,
        "minimum": cost_amount if minimum is None else minimum,
        "requirements": requirements,
        "static_placement_maximum_distance": (static_placement_maximum_distance),
        "static_placement_allow_walls": static_placement_allow_walls,
        "evidence": evidence,
        "policy_selectable": policy_selectable,
        "native_bindable": native_bindable,
        "events": events,
    }


def _event(
    kind: int,
    time: float,
    *,
    f: dict[int, float] | None = None,
    i: dict[int, int] | None = None,
    flags: int = 0,
) -> dict:
    return {"kind": kind, "time": time, "f": f or {}, "i": i or {}, "flags": flags}


def _melee(
    time: float,
    damage: float,
    reach: float,
    half_angle: float,
    *,
    damage_class: int,
    force: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    force_mode: int = FORCE_SET,
    force_direction_mode: int = FORCE_DIRECTION_LOCAL,
    angled: tuple[
        float,
        float,
        float,
        tuple[float, float, float, float],
    ]
    | None = None,
    on_hit: tuple[int, float] | None = None,
    stab_selector: _StabSelectorProgram | None = None,
    horizontal_selector: _HorizontalSelectorProgram | None = None,
    selector_requires_injection: bool = True,
    selector_tests_line_of_sight: bool = True,
    random_percentage: float = 0.0,
    cause: int = DAMAGE_PHYSICAL,
) -> dict:
    if stab_selector is not None and horizontal_selector is not None:
        raise ValueError("one melee event cannot use two selector geometries")
    x, y, z, magnitude = force
    resource_id, resource_delta = on_hit or (-1, 0.0)
    f = {
        EF_DAMAGE: damage,
        EF_RANGE: reach,
        EF_HALF_ANGLE_DEGREES: half_angle,
        EF_FORCE_X: x,
        EF_FORCE_Y: y,
        EF_FORCE_Z: z,
        EF_FORCE_MAGNITUDE: magnitude,
        EF_AIR_RESISTANCE: 0.99,
        EF_AIR_RESISTANCE_MAX: 0.98,
        EF_GROUND_RESISTANCE: 0.94,
        EF_GROUND_RESISTANCE_MAX: 0.3,
        EF_RESISTANCE_THRESHOLD: 3.0,
        EF_ON_HIT_RESOURCE_DELTA: resource_delta,
        EF_RANDOM_PERCENTAGE: random_percentage,
    }
    flags = EVENT_FLAG_SERVER_SELECTOR
    if stab_selector is not None:
        flags |= (
            EVENT_FLAG_INJECTED_SELECTOR
            if selector_requires_injection
            else EVENT_FLAG_PROGRESSIVE_STAB_SELECTOR
        )
        f.update(
            {
                EF_SELECTOR_RUNTIME_SECONDS: stab_selector.runtime_seconds,
                EF_SELECTOR_START_DISTANCE: stab_selector.start_distance,
                EF_SELECTOR_END_DISTANCE: stab_selector.end_distance,
                EF_SELECTOR_EXTEND_LEFT: stab_selector.extend_left,
                EF_SELECTOR_EXTEND_RIGHT: stab_selector.extend_right,
                EF_SELECTOR_EXTEND_BOTTOM: stab_selector.extend_bottom,
                EF_SELECTOR_EXTEND_TOP: stab_selector.extend_top,
                EF_SELECTOR_YAW_OFFSET_DEGREES: (stab_selector.yaw_offset_degrees),
                EF_SELECTOR_PITCH_OFFSET_DEGREES: (stab_selector.pitch_offset_degrees),
                EF_SELECTOR_ROLL_OFFSET_DEGREES: (stab_selector.roll_offset_degrees),
            }
        )
    if horizontal_selector is not None:
        flags |= EVENT_FLAG_PROGRESSIVE_HORIZONTAL_SELECTOR
        f.update(
            {
                EF_SELECTOR_RUNTIME_SECONDS: (horizontal_selector.runtime_seconds),
                EF_SELECTOR_START_DISTANCE: (horizontal_selector.start_distance),
                EF_SELECTOR_END_DISTANCE: horizontal_selector.end_distance,
                EF_SELECTOR_EXTEND_BOTTOM: horizontal_selector.extend_bottom,
                EF_SELECTOR_EXTEND_TOP: horizontal_selector.extend_top,
                EF_SELECTOR_YAW_OFFSET_DEGREES: (
                    horizontal_selector.yaw_start_offset_degrees
                ),
                EF_SELECTOR_PITCH_OFFSET_DEGREES: (
                    horizontal_selector.pitch_offset_degrees
                ),
                EF_SELECTOR_ROLL_OFFSET_DEGREES: (
                    horizontal_selector.roll_offset_degrees
                ),
                EF_SELECTOR_YAW_LENGTH_DEGREES: (
                    horizontal_selector.yaw_length_degrees
                    * horizontal_selector.yaw_direction_multiplier
                ),
            }
        )
    if not selector_tests_line_of_sight:
        flags |= EVENT_FLAG_SELECTOR_IGNORES_LINE_OF_SIGHT
    if angled is not None:
        (
            angled_damage,
            angle,
            distance,
            angled_force,
        ) = angled
        f.update(
            {
                EF_ANGLED_DAMAGE: angled_damage,
                EF_ANGLED_ANGLE_DEGREES: angle,
                EF_ANGLED_DISTANCE_DEGREES: distance,
                EF_ANGLED_FORCE_X: angled_force[0],
                EF_ANGLED_FORCE_Y: angled_force[1],
                EF_ANGLED_FORCE_Z: angled_force[2],
                EF_ANGLED_FORCE_MAGNITUDE: angled_force[3],
            }
        )
        flags |= EVENT_FLAG_ANGLED_DAMAGE
    return _event(
        EVENT_MELEE_CONE,
        time,
        f=f,
        i={
            EI_DAMAGE_CAUSE: cause,
            EI_DAMAGE_CLASS: damage_class,
            EI_FORCE_MODE: force_mode,
            EI_FORCE_DIRECTION_MODE: force_direction_mode,
            EI_TARGET_MODE: TARGET_OTHER,
            EI_RESISTANCE_STYLE: 0,
            EI_ON_HIT_RESOURCE_ID: resource_id,
        },
        flags=flags,
    )


def _projectile(
    time: float,
    *,
    kind: int,
    damage_class: int,
    damage: float,
    cause: int,
    speed: float,
    gravity: float,
    terminal: float,
    half_extent: float = 0.1,
    lifetime: float = 20.0,
    radius: float = 0.0,
    falloff: float = 0.0,
    block_damage_radius: int = 0,
    yaw: float = 0.0,
    pitch: float = 0.0,
    fuse: float = 0.0,
    dead_time: float = -1.0,
    standard_physics: bool = False,
    bounciness: float = 0.0,
    bounce_limit: float = 0.4,
    bounce_count: int = 0,
    allow_rolling: bool = False,
    rolling_friction_factor: float = 1.0,
    force: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    force_mode: int = FORCE_SET,
    force_direction_mode: int = FORCE_DIRECTION_LOCAL,
    resistance: tuple[float, float, float, float, float] = (
        0.97,
        0.96,
        0.94,
        0.3,
        3.0,
    ),
    resistance_style: int = 1,
    status: tuple[int, float, float, float, int, int, float, float, int] | None = None,
    status_overlap: int = STATUS_OVERLAP_OVERWRITE,
    on_hit: tuple[int, float] | None = None,
    on_hit_healing: float = 0.0,
    direct_damage: float = 0.0,
    direct_damage_cause: int = DAMAGE_PROJECTILE,
    random_percentage: float = 0.0,
    entity_only_area: bool = False,
    area_friendly_fire: bool = False,
    parallel_fork: bool = False,
) -> dict:
    launch = _PROJECTILE_LAUNCH_TRANSFORMS[kind]
    force_x, force_y, force_z, force_magnitude = force
    air, air_max, ground, ground_max, threshold = resistance
    resource_id, resource_delta = on_hit or (-1, 0.0)
    f = {
        EF_DAMAGE: damage,
        EF_RADIUS: radius,
        EF_FALLOFF: falloff,
        EF_PROJECTILE_SPEED: speed,
        EF_PROJECTILE_GRAVITY: gravity,
        EF_PROJECTILE_TERMINAL_VELOCITY: terminal,
        EF_PROJECTILE_HALF_EXTENT: half_extent,
        EF_PROJECTILE_LIFETIME_SECONDS: lifetime,
        EF_YAW_OFFSET_DEGREES: yaw + launch.yaw_degrees,
        EF_PITCH_OFFSET_DEGREES: pitch + launch.pitch_degrees,
        EF_PROJECTILE_FUSE_SECONDS: fuse,
        EF_PROJECTILE_DEAD_TIME_SECONDS: dead_time,
        EF_FORCE_X: force_x,
        EF_FORCE_Y: force_y,
        EF_FORCE_Z: force_z,
        EF_FORCE_MAGNITUDE: force_magnitude,
        EF_AIR_RESISTANCE: air,
        EF_AIR_RESISTANCE_MAX: air_max,
        EF_GROUND_RESISTANCE: ground,
        EF_GROUND_RESISTANCE_MAX: ground_max,
        EF_RESISTANCE_THRESHOLD: threshold,
        EF_STATUS_SPEED_MULTIPLIER: 1.0,
        EF_ON_HIT_RESOURCE_DELTA: resource_delta,
        EF_STATUS_HEALING: on_hit_healing,
        EF_RANDOM_PERCENTAGE: random_percentage,
        EF_PROJECTILE_SPAWN_OFFSET_X: launch.offset[0],
        EF_PROJECTILE_SPAWN_OFFSET_Y: launch.offset[1],
        EF_PROJECTILE_SPAWN_OFFSET_Z: launch.offset[2],
        EF_PROJECTILE_BOUNCINESS: bounciness,
        EF_PROJECTILE_BOUNCE_LIMIT: bounce_limit,
        EF_PROJECTILE_ROLLING_FRICTION_FACTOR: rolling_friction_factor,
        EF_PROJECTILE_DIRECT_DAMAGE: direct_damage,
    }
    i = {
        EI_DAMAGE_CAUSE: cause,
        EI_DAMAGE_CLASS: damage_class,
        EI_BLOCK_DAMAGE_RADIUS: block_damage_radius,
        EI_PROJECTILE_KIND: kind,
        EI_STATUS_RESOURCE_ID: -1,
        EI_TARGET_MODE: TARGET_OTHER,
        EI_FORCE_MODE: force_mode,
        EI_FORCE_DIRECTION_MODE: force_direction_mode,
        EI_RESISTANCE_STYLE: resistance_style,
        EI_ON_HIT_RESOURCE_ID: resource_id,
        EI_PROJECTILE_BOUNCE_COUNT: bounce_count,
        EI_PROJECTILE_DIRECT_DAMAGE_CAUSE: direct_damage_cause,
    }
    flags = EVENT_FLAG_PROJECTILE_LEGACY_OFFSET if launch.legacy_ballistic else 0
    if launch.pitch_adjust_depth:
        flags |= EVENT_FLAG_PROJECTILE_PITCH_ADJUST_OFFSET
    if entity_only_area:
        flags |= EVENT_FLAG_ENTITY_ONLY_AREA
    if area_friendly_fire:
        flags |= EVENT_FLAG_AREA_FRIENDLY_FIRE
    if parallel_fork:
        flags |= EVENT_FLAG_PARALLEL_FORK
    if standard_physics:
        flags |= EVENT_FLAG_PROJECTILE_STANDARD_PHYSICS
    if allow_rolling:
        flags |= EVENT_FLAG_PROJECTILE_ALLOW_ROLLING
    if status is not None:
        (
            effect_id,
            duration,
            cooldown,
            status_damage,
            status_cause,
            resource_id,
            resource_delta,
            speed_multiplier,
            status_flags,
        ) = status
        f.update(
            {
                EF_STATUS_DURATION_SECONDS: duration,
                EF_STATUS_COOLDOWN_SECONDS: cooldown,
                EF_STATUS_DAMAGE: status_damage,
                EF_STATUS_RESOURCE_DELTA: resource_delta,
                EF_STATUS_SPEED_MULTIPLIER: speed_multiplier,
            }
        )
        i.update(
            {
                EI_STATUS_ID: effect_id,
                EI_STATUS_DAMAGE_CAUSE: status_cause,
                EI_STATUS_RESOURCE_ID: resource_id,
                EI_STATUS_OVERLAP_MODE: status_overlap,
            }
        )
        flags |= status_flags
    return _event(EVENT_PROJECTILE, time, f=f, i=i, flags=flags)


def _force(
    time: float,
    direction: tuple[float, float, float],
    magnitude: float,
    *,
    mode: int = FORCE_ADD,
    target: int = TARGET_SELF,
    style: int = 1,
    resistance: tuple[float, float, float, float, float] = (
        0.97,
        0.96,
        0.94,
        0.82,
        3.0,
    ),
    vertical_clamp: tuple[float, float] | None = None,
) -> dict:
    air, air_max, ground, ground_max, threshold = resistance
    # C3: `AdjustVertical` and `VerticalClamp` are authored together on the
    # ApplyForce node, so one argument carries both -- passing the clamp is what
    # sets the flag, and omitting it leaves the authored direction alone.
    if vertical_clamp is None:
        clamp_minimum, clamp_maximum = 0.0, 0.0
    else:
        clamp_minimum, clamp_maximum = vertical_clamp
        if not clamp_minimum <= clamp_maximum:
            raise ValueError("vertical_clamp must be ordered (minimum, maximum)")
    return _event(
        EVENT_FORCE,
        time,
        f={
            EF_FORCE_X: direction[0],
            EF_FORCE_Y: direction[1],
            EF_FORCE_Z: direction[2],
            EF_FORCE_MAGNITUDE: magnitude,
            EF_AIR_RESISTANCE: air,
            EF_AIR_RESISTANCE_MAX: air_max,
            EF_GROUND_RESISTANCE: ground,
            EF_GROUND_RESISTANCE_MAX: ground_max,
            EF_RESISTANCE_THRESHOLD: threshold,
            EF_VERTICAL_CLAMP_MIN: clamp_minimum,
            EF_VERTICAL_CLAMP_MAX: clamp_maximum,
        },
        i={
            EI_FORCE_MODE: mode,
            EI_TARGET_MODE: target,
            EI_RESISTANCE_STYLE: style,
            EI_ADJUST_VERTICAL: int(vertical_clamp is not None),
        },
    )


def _status(
    time: float,
    effect_id: str,
    duration: float,
    *,
    cooldown: float = 0.0,
    damage: float = 0.0,
    damage_cause: int = DAMAGE_PHYSICAL,
    healing: float = 0.0,
    resource_id: int = -1,
    resource_delta: float = 0.0,
    speed: float = 1.0,
    status_flags: int = 0,
    event_flags: int = 0,
    overlap: int = STATUS_OVERLAP_OVERWRITE,
    target: int = TARGET_SELF,
    value_percent: bool = False,
) -> dict:
    flags = (
        status_flags
        | event_flags
        | (EVENT_FLAG_STATUS_VALUE_PERCENT if value_percent else 0)
    )
    return _event(
        EVENT_STATUS,
        time,
        f={
            EF_STATUS_DURATION_SECONDS: duration,
            EF_STATUS_COOLDOWN_SECONDS: cooldown,
            EF_STATUS_DAMAGE: damage,
            EF_STATUS_HEALING: healing,
            EF_STATUS_RESOURCE_DELTA: resource_delta,
            EF_STATUS_SPEED_MULTIPLIER: speed,
        },
        i={
            EI_STATUS_ID: semantic_id(effect_id),
            EI_STATUS_DAMAGE_CAUSE: damage_cause,
            EI_STATUS_RESOURCE_ID: resource_id,
            EI_STATUS_OVERLAP_MODE: overlap,
            EI_TARGET_MODE: target,
        },
        flags=flags,
    )


def _clear_status(
    time: float,
    effect_id: str,
    *,
    target: int = TARGET_SELF,
) -> dict:
    return _event(
        EVENT_CLEAR_STATUS,
        time,
        i={
            EI_STATUS_ID: semantic_id(effect_id),
            EI_TARGET_MODE: target,
        },
    )


def _resource(
    time: float,
    resource_id: int,
    amount: float,
    *,
    percent: bool = False,
) -> dict:
    return _event(
        EVENT_RESOURCE,
        time,
        f={EF_STATUS_RESOURCE_DELTA: amount},
        i={EI_STATUS_RESOURCE_ID: resource_id, EI_TARGET_MODE: TARGET_SELF},
        flags=EVENT_FLAG_VALUE_PERCENT if percent else 0,
    )


def _heal(time: float, amount: float, *, percent: bool = False) -> dict:
    return _event(
        EVENT_HEAL,
        time,
        f={EF_STATUS_HEALING: amount},
        i={EI_TARGET_MODE: TARGET_SELF},
        flags=EVENT_FLAG_VALUE_PERCENT if percent else 0,
    )


def _area(
    time: float,
    *,
    kind: int,
    damage_class: int,
    duration: float,
    interval: float,
    start_radius: float,
    end_radius: float,
    radius_change: float | None = None,
    height: float,
    damage: float,
    cause: int,
    status: tuple[int, float, float, float, int, int, float, float, int] | None = None,
    random_percentage: float = 0.0,
    entity_only_area: bool = False,
    friendly_fire: bool = False,
) -> dict:
    event = _projectile(
        time,
        kind=PROJECTILE_NONE,
        damage_class=damage_class,
        damage=damage,
        cause=cause,
        speed=0.0,
        gravity=0.0,
        terminal=1.0,
        radius=start_radius,
        status=status,
        random_percentage=random_percentage,
        entity_only_area=entity_only_area,
        area_friendly_fire=friendly_fire,
    )
    event["kind"] = EVENT_AREA
    event["f"].update(
        {
            EF_AREA_DURATION_SECONDS: duration,
            EF_AREA_INTERVAL_SECONDS: interval,
            EF_AREA_END_RADIUS: end_radius,
            EF_AREA_HEIGHT: height,
            EF_AREA_RADIUS_CHANGE_SECONDS: (
                duration if radius_change is None else radius_change
            ),
        }
    )
    event["i"][EI_PROJECTILE_KIND] = kind
    return event


# Ability indices within _iron_sword(): 0 Swing_Left, 1 Swing_Right,
# 2 Swing_Down, 3 Thrust, 4 Signature_Vortexstrike.
_SWORD_SWING_LEFT_SLOT = 0
_SWORD_THRUST_SLOT = 3

# Every sword attack authors a server selector; none of them is a cone. The
# tuples below transcribe Assets.zip:
# Server/Item/Interactions/Weapons/Sword/Attacks/**/*_Selector.json verbatim.
# Field order is (runtime, start, end, extend_bottom, extend_top, yaw_length,
# yaw_direction, yaw_start_offset, pitch, roll); positive yaw_direction is
# ToLeft, matching the EF_SELECTOR_YAW_LENGTH_DEGREES sign convention.
_SWORD_SWING_LEFT_SELECTOR = _HorizontalSelectorProgram(
    0.05, 0.1, 2.75, 0.5, 0.5, 90.0, 1.0, -60.0, 0.0, 30.0
)
_SWORD_SWING_RIGHT_SELECTOR = _HorizontalSelectorProgram(
    0.05, 0.1, 2.75, 0.5, 0.5, 90.0, -1.0, -60.0, 0.0, 30.0
)
_SWORD_SWING_DOWN_SELECTOR = _HorizontalSelectorProgram(
    0.067, 0.1, 2.75, 0.4, 0.4, 90.0, 1.0, -60.0, 0.0, 90.0
)
# Vortexstrike_Spin_Selector is a 360-degree sweep whose Next Parallel nests a
# second 180-degree sweep; the profile's two spin events map one-to-one onto
# them.
_SWORD_VORTEX_SPIN_SELECTOR = _HorizontalSelectorProgram(
    0.15, 0.1, 3.0, 1.0, 0.5, 360.0, -1.0, -90.0, 0.0, 0.0
)
_SWORD_VORTEX_SPIN_FOLLOW_SELECTOR = _HorizontalSelectorProgram(
    0.1, 0.1, 3.0, 1.0, 0.5, 180.0, -1.0, -90.0, 0.0, 0.0
)
_SWORD_THRUST_SELECTOR = _StabSelectorProgram(
    runtime_seconds=0.3,
    start_distance=0.1,
    end_distance=2.25,
    extend_left=0.5,
    extend_right=0.5,
    extend_bottom=0.5,
    extend_top=0.5,
    yaw_offset_degrees=0.0,
    pitch_offset_degrees=0.0,
    roll_offset_degrees=0.0,
)
_SWORD_VORTEX_STAB_SELECTOR = _StabSelectorProgram(
    runtime_seconds=0.3,
    start_distance=0.1,
    end_distance=2.5,
    extend_left=0.5,
    extend_right=0.5,
    extend_bottom=0.5,
    extend_top=0.5,
    yaw_offset_degrees=0.0,
    pitch_offset_degrees=0.0,
    roll_offset_degrees=0.0,
)

# Weapon_Sword_Primary is a "Type": "Charging" root that gates twice -- an outer
# 0.2s branch into Weapon_Sword_Primary_Thrust_StaminaCondition, then an inner
# 0.65s branch inside Weapon_Sword_Primary_Thrust itself. Both early branches,
# and the StatsCondition's Failed edge, target Weapon_Sword_Primary_Chain, so
# the nesting collapses to the two rows below.
#
# RESOLVED: the inner clock restarts. ChargingInteraction.simulateTick0 tests
# the interaction-local `time` against its own highestChargeValue
# (ChargingInteraction.java:281) and the player handler's getChargeValue returns
# that same local `time` (InteractionSimulationHandler.java:33-36), so the two
# roots are two independent clocks that add: 0.2 to leave the outer root plus
# 0.65 inside Weapon_Sword_Primary_Thrust.
# Every nested Charging pair in 0.5.7 is `0.2 + inner`. The authored
# HorizontalSpeedMultiplier sits on the inner root only, so the first 0.2 s
# of one of these holds runs at full speed.
_NESTED_OUTER_GATE_SECONDS = 0.2
_SWORD_THRUST_HOLD_SECONDS = 0.85
# Weapon_Battleaxe_Primary reaches the Downstrike StatsCondition at 0.2, and
# Weapon_Battleaxe_Primary_Downstrike is itself a Charging root branching at
# 0.65, so the flattened release is their sum.
_DOWNSTRIKE_HOLD_SECONDS = _NESTED_OUTER_GATE_SECONDS + 0.65

# The chain fallback is approximated by its first swing until C5 ChainCycle
# lands; the authored Weapon_Sword_Primary_Chain cycles Left -> Right -> Down
# with a 2s ChainingAllowance, which no chain state exists to model yet.
_SWORD_PRIMARY_HOLD_TABLE = (
    (0.0, _SWORD_SWING_LEFT_SLOT),
    (_SWORD_THRUST_HOLD_SECONDS, _SWORD_THRUST_SLOT),
)


def _iron_sword(
    asset: str = "Weapon_Sword_Iron",
    family: int = FAMILY_SWORD,
    damage: tuple[float, ...] = (10, 10, 18, 26, 19, 56),
    guard_stamina_value: float = 10,
) -> dict:
    if len(damage) != 6:
        raise ValueError("sword variants require six damage values")
    left, right, down, thrust, vortex_spin, vortex_stab = damage
    push = (0.0, 1.0, -1.0, 6.0)
    abilities = [
        _ability(
            "Weapon_Sword_Primary_Swing_Left",
            0.334,
            _melee(
                0.117,
                left,
                2.75,
                45,
                damage_class=DAMAGE_CLASS_LIGHT,
                force=push,
                on_hit=(RESOURCE_SIGNATURE_ENERGY, 1),
                horizontal_selector=_SWORD_SWING_LEFT_SELECTOR,
            ),
            evidence=(
                EVIDENCE_NATIVE_DIFFERENTIAL
                if asset == "Weapon_Sword_Iron"
                else EVIDENCE_ASSET_RESOLVED
            ),
        ),
        _ability(
            "Weapon_Sword_Primary_Swing_Right",
            0.334,
            _melee(
                0.117,
                right,
                2.75,
                45,
                damage_class=DAMAGE_CLASS_LIGHT,
                force=(0, 1, -1, 7),
                on_hit=(RESOURCE_SIGNATURE_ENERGY, 1),
                horizontal_selector=_SWORD_SWING_RIGHT_SELECTOR,
            ),
        ),
        _ability(
            "Weapon_Sword_Primary_Swing_Down",
            0.75,
            _melee(
                0.433,
                down,
                2.75,
                45,
                damage_class=DAMAGE_CLASS_LIGHT,
                force=(0, 1, -2, 12),
                on_hit=(RESOURCE_SIGNATURE_ENERGY, 1),
                horizontal_selector=_SWORD_SWING_DOWN_SELECTOR,
            ),
        ),
        _ability(
            # Timed from the moment the 0.65 branch is taken, not from the
            # button press: hold_table already spends _SWORD_THRUST_HOLD_SECONDS
            # before this program starts. The authored child is
            # Serial[StaminaCost, StaminaRegenDelay, Simple 0.05 ->
            # Serial[Force, Selector 0.3]] with a 0.251 pad, so 0.601 total with
            # both the dash and the stab landing at 0.05.
            "Weapon_Sword_Primary_Thrust",
            0.601,
            _force(0.05, (0, 0, -10), 20),
            _melee(
                0.05,
                thrust,
                2.25,
                20,
                damage_class=DAMAGE_CLASS_CHARGED,
                force=(0, 1, -2, 15),
                on_hit=(RESOURCE_SIGNATURE_ENERGY, 3),
                stab_selector=_SWORD_THRUST_SELECTOR,
            ),
            cost={
                RESOURCE_STAMINA: (
                    _stamina_break_immune_instant_add_stat_cost(
                        4,
                        commit_time_seconds=0.0,
                    )
                )
            },
            minimum={RESOURCE_STAMINA: 0.1},
            stamina_delay=-2,
            requirements=REQUIRE_CLEAR_FORCE_PATH,
            hold_table=_SWORD_PRIMARY_HOLD_TABLE,
            # Weapon_Sword_Primary_Thrust: HorizontalSpeedMultiplier 0.8,
            # on the inner root, so it starts after the 0.2 s outer gate.
            hold_speed_multiplier=0.8,
            hold_speed_multiplier_after_seconds=_NESTED_OUTER_GATE_SECONDS,
        ),
        _ability(
            "Weapon_Sword_Signature_Vortexstrike",
            2.103,
            _force(0.50, (0, 0, -1), 4),
            _melee(
                0.50,
                vortex_spin,
                3.0,
                180,
                damage_class=DAMAGE_CLASS_SIGNATURE,
                force=(0, 1, -2, 4),
                horizontal_selector=_SWORD_VORTEX_SPIN_SELECTOR,
            ),
            _melee(
                0.65,
                vortex_spin,
                3.0,
                90,
                damage_class=DAMAGE_CLASS_SIGNATURE,
                force=(0, 1, -2, 4),
                horizontal_selector=_SWORD_VORTEX_SPIN_FOLLOW_SELECTOR,
            ),
            _force(1.65, (0, 0, -1), 18),
            _melee(
                1.65,
                vortex_stab,
                2.5,
                25,
                damage_class=DAMAGE_CLASS_SIGNATURE,
                force=(0, 2, -3, 18),
                stab_selector=_SWORD_VORTEX_STAB_SELECTOR,
            ),
            cost={
                RESOURCE_SIGNATURE_ENERGY: _instant_add_stat_cost(
                    20,
                    commit_time_seconds=2.103,
                    commit_flags=EVENT_FLAG_SERVER_SELECTOR,
                )
            },
            requirements=REQUIRE_CLEAR_FORCE_PATH,
        ),
    ]
    return _weapon(
        asset,
        family,
        abilities,
        guard_stamina_value,
        {RESOURCE_SIGNATURE_ENERGY: 20},
    )


# Weapon_Mace_Primary_Swing_<dir>_Charged gates at 2.0s on its own Charging
# clock, and the StatsCondition that reaches it wraps an inline Charging root
# gating at 0.2s. Those are two interactions, so their clocks are independent
# and add the same way the sword's do -- see _SWORD_THRUST_HOLD_SECONDS.
# AllowIndefiniteHold is false on both, so the pair fires at the threshold
# rather than waiting for release.
_MACE_CHARGED_HOLD_SECONDS = 2.2


def _iron_mace(
    asset: str = "Weapon_Mace_Iron",
    family: int = FAMILY_MACE,
    damage: tuple[float, ...] = (29, 29, 29, 41, 41, 41),
    guard_stamina_value: float = 10,
) -> dict:
    if len(damage) != 6:
        raise ValueError("mace variants require six damage values")
    normal = (
        (
            "Left",
            (-2, 1, -1, 12),
            _HorizontalSelectorProgram(
                0.1, 0.1, 3.0, 0.75, 0.75, 120.0, 1.0, -60.0, 0.0, 0.0
            ),
        ),
        (
            "Right",
            (2, 1, -1, 12),
            _HorizontalSelectorProgram(
                0.1, 0.1, 3.0, 0.75, 0.75, 120.0, -1.0, -60.0, 0.0, 0.0
            ),
        ),
        (
            "Up_Left",
            (0, 2, -2, 12),
            _HorizontalSelectorProgram(
                0.1, 0.1, 3.0, 0.75, 0.75, 120.0, 1.0, -60.0, 0.0, -45.0
            ),
        ),
    )
    charged = (
        (
            "Left",
            (-1, 1, -1, 18),
            _HorizontalSelectorProgram(
                0.25, 0.1, 3.0, 0.75, 0.75, 180.0, 1.0, -90.0, 0.0, 0.0
            ),
        ),
        (
            "Right",
            (1, 1, -1, 18),
            _HorizontalSelectorProgram(
                0.25, 0.1, 3.0, 0.75, 0.75, 180.0, -1.0, -90.0, 0.0, 0.0
            ),
        ),
        (
            "Up_Left",
            (0, 5, -3, 18),
            _HorizontalSelectorProgram(
                0.25, 0.1, 3.0, 0.75, 0.75, 180.0, 1.0, -90.0, 0.0, -45.0
            ),
        ),
    )
    abilities = [
        _ability(
            f"Weapon_Mace_Primary_Swing_{name}",
            1.117,
            _melee(
                0.6,
                hit_damage,
                3,
                60,
                damage_class=DAMAGE_CLASS_LIGHT,
                force=force,
                on_hit=(RESOURCE_SIGNATURE_ENERGY, 1),
                horizontal_selector=selector,
            ),
        )
        for (name, force, selector), hit_damage in zip(
            normal,
            damage[:3],
            strict=True,
        )
    ] + [
        _ability(
            # Child-relative like the sword thrust: hold_table spends
            # _MACE_CHARGED_HOLD_SECONDS first, then the authored Serial runs
            # Simple 0.05 -> Simple 0.083 -> Selector 0.25 -> pad 0.25.
            f"Weapon_Mace_Primary_Swing_{name}_Charged",
            0.633,
            _melee(
                0.133,
                hit_damage,
                3,
                90,
                damage_class=DAMAGE_CLASS_CHARGED,
                force=force,
                on_hit=(RESOURCE_SIGNATURE_ENERGY, 2),
                horizontal_selector=selector,
            ),
            cost={
                RESOURCE_STAMINA: (
                    _stamina_break_immune_instant_add_stat_cost(
                        2,
                        commit_time_seconds=0.0,
                    )
                )
            },
            minimum={RESOURCE_STAMINA: 0.1},
            stamina_delay=-1.5,
            # Weapon_Mace_Primary_Swing_<dir>_Charged is a Charging root at
            # AllowIndefiniteHold: false, whose 0 branch is that direction's
            # ordinary swing and whose 2.0 branch is the charged Serial. The
            # StatsCondition wrapping it (Costs Stamina 0.1) sends its Failed
            # edge to the same ordinary swing, and the outer 0.2 gate does too,
            # so all three early paths collapse into the single row below.
            hold_table=(
                (0.0, index),
                (_MACE_CHARGED_HOLD_SECONDS, index + len(normal)),
            ),
            # Weapon_Mace_Primary_Swing_*_Charged: 0.8 on all three, again
            # on the inner root behind the 0.2 s outer gate.
            hold_speed_multiplier=0.8,
            hold_speed_multiplier_after_seconds=_NESTED_OUTER_GATE_SECONDS,
        )
        for index, ((name, force, selector), hit_damage) in enumerate(
            zip(charged, damage[3:], strict=True)
        )
    ]
    return _weapon(
        asset,
        family,
        abilities,
        guard_stamina_value,
        {RESOURCE_SIGNATURE_ENERGY: 8},
    )


def _iron_battleaxe(
    asset: str = "Weapon_Battleaxe_Iron",
    family: int = FAMILY_BATTLEAXE,
    damage: tuple[float, ...] = (18, 23, 36, 14),
    guard_stamina_value: float = 10,
) -> dict:
    if len(damage) != 4:
        raise ValueError("battleaxe variants require four damage values")
    # Weapon_Battleaxe_Primary_Swing_Down{,_Left,_Right}_Selector.json: one
    # Horizontal sweep each, differing only in Direction and RollOffset.
    primary = (
        (
            "Down_Left",
            damage[0],
            (-0.25, 1, -1.5, 8),
            FORCE_ADD,
            _HorizontalSelectorProgram(
                0.15, 0.1, 3.0, 0.5, 0.5, 90.0, 1.0, -45.0, 0.0, 50.0
            ),
        ),
        (
            "Down_Right",
            damage[1],
            (0.25, 1, -1.5, 9),
            FORCE_ADD,
            _HorizontalSelectorProgram(
                0.15, 0.1, 3.0, 0.5, 0.5, 90.0, -1.0, -45.0, 0.0, -50.0
            ),
        ),
        (
            "Down",
            damage[2],
            (0, 1, -1.5, 13),
            FORCE_SET,
            _HorizontalSelectorProgram(
                0.15, 0.1, 3.0, 0.5, 0.5, 90.0, 1.0, -45.0, 0.0, 90.0
            ),
        ),
    )
    # Weapon_Battleaxe_Primary_Downstrike_Selector.json. `Direction: ToLeft`
    # is the +1.0 multiplier, as on Swing_Down which is also ToLeft.
    _DOWNSTRIKE_SELECTOR = _HorizontalSelectorProgram(
        0.083, 0.1, 4.0, 0.5, 0.5, 120.0, 1.0, -60.0, 0.0, 90.0
    )
    pull = dict(
        force=(0, 0, 0, -2),
        force_direction_mode=FORCE_DIRECTION_POINT,
    )
    # Whirlwind_Spin_Selector is a 360-degree sweep inside a Repeat; the ten
    # spin events are its ten iterations. Whirlwind_End_Selector closes with a
    # single 210-degree sweep.
    _WHIRLWIND_SPIN_SELECTOR = _HorizontalSelectorProgram(
        0.1, 0.1, 3.0, 1.5, 0.5, 360.0, 1.0, -180.0, 0.0, 0.0
    )
    _WHIRLWIND_END_SELECTOR = _HorizontalSelectorProgram(
        0.167, 0.1, 3.0, 1.5, 0.5, 210.0, 1.0, -180.0, 0.0, 0.0
    )
    whirlwind = tuple(
        _melee(
            0.167 + i * 0.1,
            damage[3],
            3,
            180,
            damage_class=DAMAGE_CLASS_SIGNATURE,
            horizontal_selector=_WHIRLWIND_SPIN_SELECTOR,
            **pull,
        )
        for i in range(10)
    )
    abilities = [
        _ability(
            f"Weapon_Battleaxe_Primary_Swing_{name}",
            1.0,
            _melee(
                0.4,
                damage,
                3,
                45,
                damage_class=DAMAGE_CLASS_LIGHT,
                force=force,
                force_mode=mode,
                on_hit=(RESOURCE_SIGNATURE_ENERGY, 1),
                horizontal_selector=selector,
            ),
            # Weapon_Battleaxe_Primary is itself a Charging root: its 0 branch
            # is the ordinary chain and its 0.2 branch reaches the Downstrike
            # StatsCondition, whose inner Charging root adds a further 0.65. One
            # hold clock is compared against this table, so the rows are the
            # *flattened* thresholds; asserting the inner 0.65 alone would fire
            # the leap 0.2 s early. The 0.6 HorizontalSpeedMultiplier is
            # authored on the inner root, hence the 0.2 s offset.
            hold_table=((0.0, index), (_DOWNSTRIKE_HOLD_SECONDS, 4)),
            hold_speed_multiplier=0.6,
            hold_speed_multiplier_after_seconds=_NESTED_OUTER_GATE_SECONDS,
        )
        for index, (name, damage, force, mode, selector) in enumerate(primary)
    ] + [
        _ability(
            "Weapon_Battleaxe_Signature_Whirlwind",
            1.334,
            *whirlwind,
            _melee(
                1.167,
                damage[3],
                3,
                105,
                damage_class=DAMAGE_CLASS_SIGNATURE,
                horizontal_selector=_WHIRLWIND_END_SELECTOR,
                **pull,
            ),
            cost={
                RESOURCE_SIGNATURE_ENERGY: _instant_add_stat_cost(
                    9,
                    commit_time_seconds=1.334,
                    commit_flags=EVENT_FLAG_SERVER_SELECTOR,
                )
            },
        ),
        # Index 4, the leap. Weapon_Battleaxe_Primary_Downstrike is a Serial of
        # the stamina spend, the regen-delay Set, a 0.05 s `Simple` pad and the
        # ApplyForce. That force authors WaitForGround, so the root parks in the
        # air and its landing child (index 5) runs on contact. Duration only has
        # to outlive the force by one 30 TPS tick -- events fire strictly inside
        # it, and the continuation owns everything after.
        _ability(
            "Weapon_Battleaxe_Primary_Downstrike",
            0.05 + 1.0 / 30.0,
            _force(
                0.05,
                (0, 1, -2),
                12,
                mode=FORCE_SET,
                # Authored on the force node; only the two air terms differ from
                # _force()'s default (0.99/0.98 against 0.97/0.96).
                resistance=(0.99, 0.98, 0.94, 0.82, 3.0),
                vertical_clamp=(10.0, 30.0),
            ),
            cost={RESOURCE_STAMINA: _single_application_stamina_break_immune_cost(3)},
            stamina_delay=-3.0,
            wait_for=("ground",),
            ground_next=5,
            # Reached by holding the Primary root, never requested directly. A
            # weapon that authors a guard rejects UNRESOLVED on any ability, so
            # this carries the owning root's type for guard arbitration.
            interaction_type=INTERACTION_TYPE_PRIMARY,
            policy_selectable=False,
        ),
        # Index 5, the landing. Weapon_Battleaxe_Primary_Downstrike_Selector
        # plus its Downstrike_Damage var (BaseDamage Physical 29, SignatureEnergy
        # +2) and the authored 0.199 s pad. _melee() already emits the knockback
        # VelocityConfig every battleaxe damage node authors.
        _ability(
            "Weapon_Battleaxe_Primary_Downstrike_Landing",
            0.083 + 0.199,
            _melee(
                0.0,
                29,
                4.0,
                60.0,
                damage_class=DAMAGE_CLASS_CHARGED,
                force=(0, 2, 1, 7),
                force_mode=FORCE_SET,
                on_hit=(RESOURCE_SIGNATURE_ENERGY, 2),
                horizontal_selector=_DOWNSTRIKE_SELECTOR,
            ),
            stamina_delay=-1.5,
            interaction_type=INTERACTION_TYPE_PRIMARY,
            policy_selectable=False,
        ),
    ]
    return _weapon(
        asset,
        family,
        abilities,
        guard_stamina_value,
        {RESOURCE_SIGNATURE_ENERGY: 9},
    )


def _iron_daggers(
    asset: str = "Weapon_Daggers_Iron",
    family: int = FAMILY_DAGGERS,
    damage: tuple[float, ...] = (4, 4, 8, 12, 25, 31, 37),
    angled_damage: tuple[float, ...] = (0, 0, 0, 0, 38, 47, 56),
    guard_stamina_value: float = 10,
) -> dict:
    if len(damage) != 7 or len(angled_damage) != 7:
        raise ValueError("dagger variants require seven direct and angled values")
    primary = (
        ("Swing_Left", damage[0], 1, (0, 1, -1, 2), FORCE_SET),
        ("Swing_Right", damage[1], 1, (0, 1, -1, 2), FORCE_ADD),
        ("Stab_Left", damage[2], 2, (0, 1, -1, 3), FORCE_SET),
        ("Stab_Right", damage[3], 2, (0, 1, -1, 3), FORCE_SET),
    )
    abilities = [
        _ability(
            f"Weapon_Daggers_Primary_{name}",
            0.207,
            _melee(
                0.069,
                damage,
                2,
                45,
                damage_class=DAMAGE_CLASS_LIGHT,
                force=force,
                force_mode=mode,
                on_hit=(RESOURCE_SIGNATURE_ENERGY, energy),
                stab_selector=_DAGGER_PRIMARY_STAB_SELECTOR,
                selector_requires_injection=False,
                selector_tests_line_of_sight=False,
            ),
        )
        for name, damage, energy, force, mode in primary
    ] + [
        _ability(
            "Weapon_Daggers_Signature_Razorstrike",
            1.334,
            _force(0.334, (0, 0, -5), 5),
            _melee(
                0.334,
                damage[4],
                2.25,
                45,
                damage_class=DAMAGE_CLASS_SIGNATURE,
                force=(0, 1, -2, 7),
                force_mode=FORCE_ADD,
                angled=(angled_damage[4], 180, 80, (0, 1, -2, 7)),
                stab_selector=_DAGGER_RAZORSTRIKE_SLASH_SELECTOR,
            ),
            _force(0.634, (0, 0, -5), 5),
            _melee(
                0.634,
                damage[5],
                2.25,
                45,
                damage_class=DAMAGE_CLASS_SIGNATURE,
                force=(0, 1, -1, 7),
                force_mode=FORCE_ADD,
                angled=(angled_damage[5], 180, 80, (0, 1, -1, 7)),
                stab_selector=_DAGGER_RAZORSTRIKE_SWEEP_SELECTOR,
            ),
            _force(
                0.984,
                (0, 1, -5),
                15,
                resistance=(0.99, 0.98, 0.94, 0.82, 3),
            ),
            _melee(
                0.984,
                damage[6],
                2.5,
                45,
                damage_class=DAMAGE_CLASS_SIGNATURE,
                force=(0, 2, -3, 12),
                angled=(angled_damage[6], 180, 80, (0, 2, -3, 12)),
                stab_selector=_DAGGER_RAZORSTRIKE_LUNGE_SELECTOR,
            ),
            cost={
                RESOURCE_SIGNATURE_ENERGY: _instant_add_stat_cost(
                    27,
                    commit_time_seconds=1.334,
                    commit_flags=EVENT_FLAG_SERVER_SELECTOR,
                )
            },
        )
    ]
    return _weapon(
        asset,
        family,
        abilities,
        guard_stamina_value,
        {RESOURCE_SIGNATURE_ENERGY: 27},
    )


def _iron_shortbow(
    asset: str = "Weapon_Shortbow_Iron",
    family: int = FAMILY_SHORTBOW,
    damage: tuple[float, ...] = (6, 9, 12, 16, 19, 14),
    guard_stamina_value: float = 10,
) -> dict:
    if len(damage) != 6:
        raise ValueError("shortbow variants require five draw tiers and volley damage")
    abilities = []
    charge_thresholds = (0.1, 0.3, 0.6, 0.9, 1.2)
    charge_clock_origin = charge_thresholds[0]
    for level, (requested_charge, hit_damage, speed, gravity) in enumerate(
        zip(
            charge_thresholds,
            damage[:5],
            (15, 30, 50, 70, 85),
            (45, 40, 32, 28, 25),
            strict=True,
        )
    ):
        # The fixed outer-root prelude already includes the minimum authored
        # Charging hold. Selected tiers therefore advance by their distance
        # from the first legal threshold, not by replaying the absolute hold
        # a second time. Live F623 release ticks are 9/15/24/33/42; using the
        # absolute threshold delays every non-minimum tier by exactly three
        # ticks at the native 30 TPS boundary.
        #
        # DELIBERATE C1 DIVERGENCE. Weapon_Shortbow_Primary_Shoot_Charge is a
        # five-row Charging root (0.1/0.3/0.6/0.9/1.2, AllowIndefiniteHold
        # true) and every other such root in this profile carries a
        # ``hold_table``. This one does not: the draw is instead modelled as
        # per-tier scheduler timing calibrated against live release ticks, and
        # ``release_tick`` below feeds stamina_delay_end_tick and the ammo
        # resource_phases window. Adding a hold table without also refitting
        # those windows would count the draw twice. Converting it needs a live
        # re-measurement, not an asset read -- see
        # docs/designs/SWORD-CHARGE-SPEC.md.
        selected_delay = requested_charge - charge_clock_origin
        release_tick = int(np.ceil(requested_charge * 30.0)) + 6
        abilities.append(
            _ability(
                f"Weapon_Shortbow_Primary_Shoot_Strength_{level}",
                selected_delay + 0.2,
                _projectile(
                    selected_delay,
                    kind=PROJECTILE_ARROW,
                    damage_class=DAMAGE_CLASS_CHARGED,
                    damage=hit_damage,
                    cause=DAMAGE_PROJECTILE,
                    speed=speed,
                    gravity=gravity,
                    terminal=50,
                    on_hit=((RESOURCE_SIGNATURE_ENERGY, 1) if level == 4 else None),
                ),
                requested_charge_time_seconds=requested_charge,
                stamina_delay=-10,
                # The outer root sets StaminaRegenDelay before its inventory
                # node, then the projectile node's Next sets it back to zero
                # after RunTime=0.2. Live fixed-tick evidence places those
                # authored Set nodes at lifecycle tick 2 and seven boundaries
                # after the selected release tick respectively.
                stamina_delay_start_tick=2,
                stamina_delay_end_tick=release_tick + 7,
                # The authored outer root traverses durability/stat gates,
                # Adventure inventory removal, the 0.1-second nock stat,
                # Replace, Charging, and the selected ChangeStat/Replace
                # branch before entering this projectile child. Eight fixed
                # scheduler boundaries reproduce the learner-visible damage
                # ticks; the physical held-item durability change is resolved
                # independently from native inventory evidence.
                scheduler_prelude_ticks=8,
                native_outer_root_id="Root_Weapon_Shortbow_Primary_Shoot",
                outer_root_item_dispatch_tick=2,
                resource_phases={
                    RESOURCE_AMMO: _ResourcePhase(
                        value=1.0,
                        inactive_value=0.0,
                        start_tick=2,
                        end_tick=release_tick,
                    )
                },
                requirements=REQUIRE_CLEAR_PROJECTILE_FLIGHT,
                evidence=(
                    EVIDENCE_NATIVE_DIFFERENTIAL
                    if level == 0
                    else EVIDENCE_ASSET_RESOLVED
                ),
            )
        )
    abilities.append(
        _ability(
            "Weapon_Shortbow_Signature_Volley_Activate",
            0.05,
            _resource(
                0,
                RESOURCE_SIGNATURE_CHARGES,
                100,
                percent=True,
            ),
            cost={RESOURCE_SIGNATURE_ENERGY: _instant_add_stat_cost(6)},
        )
    )
    for level, (charge, speed, spread) in enumerate(
        ((0, 30, 10), (0.75, 60, 5), (1.5, 90, 1))
    ):
        arrows = tuple(
            _projectile(
                0.0,
                kind=PROJECTILE_ARROW,
                damage_class=DAMAGE_CLASS_SIGNATURE,
                damage=damage[5],
                cause=DAMAGE_PROJECTILE,
                speed=speed,
                gravity=30,
                terminal=50,
                yaw=yaw,
            )
            for yaw in (-spread, 0, spread)
        )
        abilities.append(
            _ability(
                f"Weapon_Shortbow_Signature_Volley_Strength_{level}",
                0.2,
                *arrows,
                cost={
                    RESOURCE_SIGNATURE_CHARGES: _instant_add_stat_cost(
                        101,
                        commit_flags=EVENT_FLAG_PARALLEL_FORK,
                    )
                },
                requirements=REQUIRE_CLEAR_PROJECTILE_FLIGHT,
                requested_charge_time_seconds=charge,
                # Weapon_Shortbow_Signature_Volley_Charge:
                # Charging{0 -> Strength_0, 0.75 -> _1, 1.5 -> _2},
                # AllowIndefiniteHold true. Unlike the primary draw ladder
                # above, these three children carry no charge prefix -- each is
                # a flat 0.2 s Parallel that launches at 0.0 -- so C1 can meter
                # the hold without double-counting.
                hold_table=((0.0, 6), (0.75, 7), (1.5, 8)),
                # Weapon_Shortbow_Signature_Volley_Charge: 0.667.
                hold_speed_multiplier=0.667,
                hold_allow_indefinite=True,
            )
        )
    profile = _weapon(
        asset,
        family,
        abilities,
        guard_stamina_value,
        {
            RESOURCE_AMMO: 1,
            RESOURCE_SIGNATURE_ENERGY: 6,
            # SignatureCharges has authored Max=100 and the equipped
            # Shortbow adds one to MAX. Both activation and volley spend use
            # ValueType=Percent, so retain the engine's 0..101 unit scale.
            RESOURCE_SIGNATURE_CHARGES: 101,
        },
    )
    # The equipped Shortbow changes Ammo's maximum, but the public Primary root
    # nocks a physical arrow and sets the current value only while its Charging
    # selector is held.  The device phase rows above reproduce that transient.
    #
    # The nock has no physical arrow to draw here -- see the note in
    # _iron_crossbow(): there is no inventory->ammo bridge in this sim, so a
    # zero start leaves the bow permanently unarmed rather than pending a
    # reload. This is the shared root cause behind the long-standing "the
    # Shortbow accepts abilities but never lands or spends ammo" pin.
    #
    # Use the episode-loadout convention from _weapon() until the nock can
    # actually consume an inventory arrow.
    return profile


def _iron_crossbow(
    asset: str = "Weapon_Crossbow_Iron",
    family: int = FAMILY_CROSSBOW,
    damage: tuple[float, ...] = (10, 27, 78),
    guard_stamina_value: float = 10,
) -> dict:
    if len(damage) != 3:
        raise ValueError("crossbow variants require three damage values")

    def projectile(damage, damage_class, on_hit=None):
        return _projectile(
            0,
            kind=PROJECTILE_ARROW,
            damage_class=damage_class,
            damage=damage,
            cause=DAMAGE_PROJECTILE,
            speed=40,
            gravity=10,
            terminal=50,
            on_hit=on_hit,
        )

    abilities = [
        _ability(
            "Root_Weapon_Crossbow_Primary_Signature",
            0.2,
            projectile(damage[0], DAMAGE_CLASS_LIGHT),
            cost={
                RESOURCE_AMMO: _single_application_cost(
                    1,
                    commit_time_seconds=0.2,
                )
            },
            cooldown=0.3,
            requirements=REQUIRE_CLEAR_PROJECTILE_FLIGHT,
            interaction_type=INTERACTION_TYPE_PRIMARY,
        ),
        _ability(
            "Weapon_Crossbow_Damage_Combo_Projectile",
            0.2,
            projectile(
                damage[1],
                DAMAGE_CLASS_CHARGED,
                (RESOURCE_SIGNATURE_ENERGY, 1),
            ),
            cost={
                RESOURCE_AMMO: _single_application_cost(
                    1,
                    commit_time_seconds=0.2,
                )
            },
            requirements=REQUIRE_CLEAR_PROJECTILE_FLIGHT,
            # This resolved damage child is selected by the equipped
            # Crossbow's Primary root. It is not itself an item trigger, so
            # the native binding table correctly omits it while the runtime
            # still needs the owning root type for guard arbitration.
            interaction_type=INTERACTION_TYPE_PRIMARY,
            policy_selectable=False,
        ),
        _ability(
            "Weapon_Crossbow_Signature_BigArrow",
            0.05,
            _resource(
                0,
                RESOURCE_SIGNATURE_CHARGES,
                100,
                percent=True,
            ),
            cost={RESOURCE_SIGNATURE_ENERGY: _instant_add_stat_cost(5)},
        ),
        _ability(
            "Weapon_Crossbow_Signature_BigArrow_Shot",
            0.2,
            _projectile(
                0,
                kind=PROJECTILE_BIG_ARROW,
                damage_class=DAMAGE_CLASS_SIGNATURE,
                damage=damage[2],
                cause=DAMAGE_PROJECTILE,
                speed=100,
                gravity=40,
                terminal=100,
                half_extent=0.15,
            ),
            cost={
                RESOURCE_SIGNATURE_CHARGES: _instant_add_stat_cost(
                    101,
                    commit_time_seconds=0.2,
                )
            },
            requirements=REQUIRE_CLEAR_PROJECTILE_FLIGHT,
            policy_selectable=False,
        ),
        _ability(
            "Root_Common_StatAmmoReload_Entry",
            # Runtime includes RepeatInteraction's one-tick fork boundary and
            # the ceil(0.25 s * 30 TPS)=8-tick Common_Bow_No_Ammo tail.
            CROSSBOW_RELOAD_DURATION_SECONDS,
            *(
                _resource(time_seconds, RESOURCE_AMMO, 1)
                for time_seconds in CROSSBOW_RELOAD_RESOURCE_TIMES_SECONDS
            ),
            interaction_type=INTERACTION_TYPE_ABILITY3,
            interrupted_by_type_mask=(
                (1 << INTERACTION_TYPE_PRIMARY) | (1 << INTERACTION_TYPE_SECONDARY)
            ),
            interruptible_after_seconds=0.8,
        ),
    ]
    profile = _weapon(
        asset,
        family,
        abilities,
        guard_stamina_value,
        {
            RESOURCE_AMMO: 6,
            RESOURCE_SIGNATURE_ENERGY: 5,
            # The installed stat has Max=100 and the equipped Crossbow adds
            # one to MAX. Its authored activation and shot both use 100%, so
            # retain the native 0..101 unit scale instead of a Boolean proxy.
            RESOURCE_SIGNATURE_CHARGES: 101,
        },
    )
    # Equipping the native Crossbow changes Ammo's maximum but leaves its
    # current value at zero, because the outer Primary or Ability3 root refills
    # it from physical arrows through the authored reload program.
    #
    # That refill does not exist here. RESOURCE_AMMO is touched only by the
    # profile authoring, the outer root that reads it, and mechanics/factory.py
    # which sets its maximum -- the inventory subsystem never mentions ammo, and
    # inventory_from_loadout() seeds the equipped weapon alone (no arrows). So a
    # zero start is not "faithful pending reload", it is permanently unarmed:
    # measured on the flat control scene the Crossbow accepted 11 abilities and
    # emitted 0 events, 0 projectiles and 0 damage, while the same scene seeded
    # with ammo produced 298 projectiles and the authored 10.0 damage.
    #
    # Until an inventory->ammo bridge exists, use the episode-loadout convention
    # every other ammo weapon already uses (see _weapon()). Restore the zero the
    # moment reload can actually draw from inventory.
    return profile


def _iron_shield(
    asset: str = "Weapon_Shield_Iron",
    family: int = FAMILY_SHIELD,
    guard_stamina_value: float = 14,
) -> dict:
    return _weapon(
        asset,
        family,
        [
            _ability(
                "Weapon_Shield_Secondary_Guard_Bash",
                0.583,
                _melee(
                    0.167,
                    0,
                    2.5,
                    45,
                    damage_class=DAMAGE_CLASS_UNKNOWN,
                    force=(0, 1, -2, 16),
                    stab_selector=_SHIELD_GUARD_BASH_SELECTOR,
                ),
                cost={
                    RESOURCE_STAMINA: (_single_application_stamina_break_immune_cost(2))
                },
                minimum={RESOURCE_STAMINA: 0.5},
                stamina_delay=-1.5,
                requirements=REQUIRE_CLEAR_FORCE_PATH,
                guard_fork_type=INTERACTION_TYPE_PRIMARY,
                evidence=EVIDENCE_NATIVE_DIFFERENTIAL,
            )
        ],
        guard_stamina_value,
        {},
    )


def _flame_staff() -> dict:
    burn = (
        semantic_id("Flame_Staff_Burn"),
        3.0,
        2.0,
        1.0,
        DAMAGE_FIRE,
        -1,
        0.0,
        1.0,
        STATUS_FLAG_DEBUFF,
    )
    # ``Weapon_Stick_Fire_Shoot_Base`` is the ``Charging`` root; it authors only
    # ``{0.0: Charged_0}``.  The full ladder lives on the four
    # ``Parent: Weapon_Stick_Fire_Shoot_Base`` branches inside
    # ``Weapon_Stick_Fire_Primary_Entry``, each of which overrides ``Next``.
    # The richest branch (``StatsCondition`` ``MagicCharges: 4``) authors
    # ``0.0/0.5/1.0/1.5/2.0 -> Charged_0/1/2/3/3``, which is where these four
    # thresholds come from.  The poorer branches substitute a weaker projectile
    # at the same threshold rather than closing the tier; that substitution is
    # the open C6 ConditionFallback gap.  Here the per-tier ``MagicCharges``
    # cost closes the unaffordable tiers instead, which matches *which* tiers
    # are reachable at a given charge count but not *what* a long hold does
    # when you cannot afford the top one.
    _FLAME_STAFF_HOLD_TABLE = ((0.0, 0), (0.5, 1), (1.0, 2), (1.5, 3))
    abilities = []
    for level, (charge, damage, radius) in enumerate(
        ((0, 10, 2), (0.5, 20, 3), (1.0, 30, 4), (1.5, 40, 5))
    ):
        abilities.append(
            _ability(
                f"Weapon_Stick_Fire_Projectile_Charged_{level}",
                0.4,
                _projectile(
                    0.2,
                    kind=PROJECTILE_FIREBALL,
                    damage_class=DAMAGE_CLASS_UNKNOWN,
                    damage=damage,
                    # ExplodeInteraction always emits Environment damage;
                    # the burn status keeps its independently authored Fire cause.
                    cause=DAMAGE_ENVIRONMENT,
                    speed=50,
                    gravity=0,
                    terminal=50,
                    radius=radius,
                    falloff=0,
                    block_damage_radius=3,
                    force=(0, 1, -1, 5),
                    force_direction_mode=FORCE_DIRECTION_POINT,
                    status=burn if level == 3 else None,
                    on_hit=((RESOURCE_SIGNATURE_ENERGY, 1) if level >= 2 else None),
                    entity_only_area=True,
                ),
                cost={
                    # The selected child requires an Adventure Player and is
                    # driven through the remote-client interaction pass.  It
                    # therefore executes ChangeStat once.  The two-pass
                    # native-NPC mechanism applies only when no remote client
                    # owns the chain (for example Shield Bash).
                    RESOURCE_MAGIC_CHARGES: _single_application_cost(
                        level + 1,
                        # C1 meters the hold outside the ability, so the child's
                        # own clock starts at the branch.  Its first Parallel
                        # branch is ChangeStat, so the spend lands at that
                        # child-relative zero rather than at the old
                        # charge-prefixed boundary.
                        commit_time_seconds=0.0,
                        commit_flags=EVENT_FLAG_PARALLEL_FORK,
                    )
                },
                # This selected child belongs to the Staff's outer authored
                # Charging bank. Keep that client hold explicit; its child
                # projectile and resource clocks are independent.
                requested_charge_time_seconds=charge,
                # Every tier is a child of the one root, so each carries the
                # same table: pressing any of them pays the draw and lands on
                # the tier the hold actually earned.  The base sets
                # ``"AllowIndefiniteHold": false``, so a full hold fires tier 3
                # at its 1.5 s threshold without waiting for release.
                hold_table=_FLAME_STAFF_HOLD_TABLE,
                hold_allow_indefinite=False,
                requirements=(
                    REQUIRE_CLEAR_PROJECTILE_FLIGHT | REQUIRE_ENTITY_ONLY_AREA
                ),
            )
        )
    abilities.extend(
        [
            _ability(
                "Weapon_Stick_Fire_Secondary_Flamethrower",
                0.4,
                _melee(
                    0,
                    12,
                    5.5,
                    20,
                    damage_class=DAMAGE_CLASS_UNKNOWN,
                    force=(0, 2, -3, 8),
                    on_hit=(RESOURCE_SIGNATURE_ENERGY, 1),
                    # Inline Selector inside Weapon_Stick_Fire_Secondary_Entry:
                    # a long, very thin stab jet (extends 0.175 on every side).
                    stab_selector=_StabSelectorProgram(
                        0.1, 0.1, 5.5, 0.175, 0.175, 0.175, 0.175, 0.0, 0.0, 0.0
                    ),
                ),
                cost={
                    RESOURCE_STAMINA: _instant_add_stat_cost(
                        0.5,
                        commit_flags=EVENT_FLAG_PARALLEL_FORK,
                    )
                },
                minimum={RESOURCE_STAMINA: 0.51},
                stamina_delay=-2,
                requirements=REQUIRE_CLEAR_FORCE_PATH,
            ),
            _ability(
                "Weapon_Stick_Fire_Activate_Trap",
                0.001,
                _resource(0, RESOURCE_DEPLOYABLE_PREVIEW, 1),
                # Activation is its own public Ability1 interaction. The
                # subsequent Primary request re-enters the item root, whose
                # StatsCondition selects placement from this state.
                minimum={RESOURCE_SIGNATURE_ENERGY: 10},
            ),
            _ability(
                "Weapon_Stick_Fire_Spawn_Trap",
                0.001,
                # The resolved Flame Staff item overrides the parent
                # interaction's 2-second/1-damage defaults. Its deployable
                # config explicitly sets AttackOwner=false and
                # AttackTeam=false while inheriting AttackEnemies=true.
                _area(
                    0,
                    kind=AREA_FIRE_TRAP,
                    damage_class=DAMAGE_CLASS_UNKNOWN,
                    duration=5,
                    interval=1,
                    start_radius=1,
                    end_radius=5,
                    radius_change=4,
                    height=2,
                    damage=0,
                    cause=DAMAGE_FIRE,
                    status=burn,
                    entity_only_area=True,
                    friendly_fire=False,
                ),
                cost={
                    RESOURCE_DEPLOYABLE_PREVIEW: _single_application_cost(1),
                    RESOURCE_SIGNATURE_ENERGY: _single_application_cost(10),
                },
                cooldown=20,
                # Bind the public Primary root so native owns the
                # DeployablePreview selector and applies the equipped item's
                # Fire_Trap_Config override.  No dispatch tick is declared:
                # unlike Shortbow nocking, this root does not populate a
                # private child resource row that JAX should bypass.
                native_outer_root_id="Root_Weapon_Stick_Fire_Primary_Entry",
                requirements=(REQUIRE_STATIC_AREA_PLACEMENT | REQUIRE_ENTITY_ONLY_AREA),
                static_placement_maximum_distance=20.0,
                static_placement_allow_walls=False,
            ),
        ]
    )
    return _weapon(
        "Weapon_Staff_Crystal_Flame",
        FAMILY_FLAME_STAFF,
        abilities,
        0,
        {
            RESOURCE_MAGIC_CHARGES: 4,
            RESOURCE_SIGNATURE_ENERGY: 10,
            RESOURCE_DEPLOYABLE_PREVIEW: 1,
        },
        native_resource_stat_overrides={
            RESOURCE_DEPLOYABLE_PREVIEW: "DeployablePreview",
        },
    )


def _ice_staff() -> dict:
    abilities = [
        _ability(
            "Ice_Staff_Primary_Entry",
            0.3,
            _projectile(
                0,
                kind=PROJECTILE_ICE_BOLT,
                damage_class=DAMAGE_CLASS_CHARGED,
                damage=5,
                cause=DAMAGE_ICE,
                speed=50,
                gravity=3,
                terminal=55,
                lifetime=20,
                force=(0, 1, -3, 8),
                force_mode=FORCE_ADD,
                parallel_fork=True,
            ),
            cost={
                RESOURCE_STAMINA: _instant_add_stat_cost(
                    1.5,
                    commit_flags=EVENT_FLAG_PARALLEL_FORK,
                )
            },
            minimum={RESOURCE_STAMINA: 1.51},
            stamina_delay=-2,
            requirements=REQUIRE_CLEAR_PROJECTILE_FLIGHT,
        ),
        _ability(
            "Staff_Cast_Summon_Launch_Ice_Ball",
            1.667,
            _projectile(
                1.167,
                kind=PROJECTILE_ICE_BALL,
                damage_class=DAMAGE_CLASS_CHARGED,
                damage=20,
                cause=DAMAGE_ICE,
                speed=30,
                gravity=4.4,
                terminal=42.5,
                half_extent=0.2,
                lifetime=3,
                force=(0, 3, -1, 20),
                parallel_fork=True,
            ),
            cost={
                RESOURCE_STAMINA: _instant_add_stat_cost(
                    5,
                    commit_time_seconds=1.167,
                    commit_flags=EVENT_FLAG_PARALLEL_FORK,
                )
            },
            minimum={RESOURCE_STAMINA: 0.1},
            stamina_delay=-1.5,
            requirements=REQUIRE_CLEAR_PROJECTILE_FLIGHT,
        ),
    ]
    return _weapon("Weapon_Staff_Crystal_Ice", FAMILY_ICE_STAFF, abilities, 0, {})


def _bombs() -> dict:
    stun = (
        semantic_id("Stun"),
        10.0,
        0.0,
        0.0,
        DAMAGE_PHYSICAL,
        -1,
        0.0,
        1.0,
        STATUS_FLAG_DISABLE_MOVEMENT
        | STATUS_FLAG_DISABLE_ABILITIES
        | STATUS_FLAG_CONTROL_IMMUNITY_GATED,
    )
    projectile = dict(
        time=0.25,
        kind=PROJECTILE_BOMB,
        # ExplosionUtils.processTargetEntity uses DamageCause.ENVIRONMENT
        # even when the projectile source belongs to the attacker.
        cause=DAMAGE_ENVIRONMENT,
        speed=25,
        gravity=15,
        terminal=50,
        half_extent=0.2,
        lifetime=5,
        fuse=0.5,
        standard_physics=True,
        bounciness=0.5,
        bounce_limit=0.4,
        bounce_count=3,
        allow_rolling=True,
        rolling_friction_factor=1.0,
        # Explode_Generic omits EntityDamageFalloff, so the codec default is 1.
        falloff=1,
    )
    abilities = [
        _ability(
            "Bomb_Throw",
            0.45,
            _projectile(
                damage_class=DAMAGE_CLASS_UNKNOWN,
                damage=20,
                radius=4,
                force=(0, 0, 0, 5),
                block_damage_radius=3,
                force_direction_mode=FORCE_DIRECTION_POINT,
                entity_only_area=True,
                **projectile,
            ),
            requirements=(REQUIRE_CLEAR_PROJECTILE_FLIGHT | REQUIRE_ENTITY_ONLY_AREA),
        ),
        _ability(
            "Bomb_Throw_Stun",
            0.45,
            _projectile(
                damage_class=DAMAGE_CLASS_UNKNOWN,
                damage=15,
                radius=3,
                force=(0, 0, 0, 15),
                block_damage_radius=3,
                force_direction_mode=FORCE_DIRECTION_POINT,
                status=stun,
                # Bomb_Explode_Stun applies the generic Stun effect. That
                # effect omits OverlapBehavior, so EntityEffect defaults to
                # IGNORE rather than the projectile helper's common
                # OVERWRITE case.
                status_overlap=STATUS_OVERLAP_IGNORE,
                entity_only_area=True,
                **projectile,
            ),
            requirements=(REQUIRE_CLEAR_PROJECTILE_FLIGHT | REQUIRE_ENTITY_ONLY_AREA),
        ),
        _ability(
            "Bomb_Throw_Popberry",
            0.45,
            _projectile(
                damage_class=DAMAGE_CLASS_UNKNOWN,
                damage=20,
                radius=4,
                force=(0, 0, 0, 5),
                block_damage_radius=3,
                force_direction_mode=FORCE_DIRECTION_POINT,
                entity_only_area=True,
                **(projectile | {"fuse": 1.0}),
            ),
            requirements=(REQUIRE_CLEAR_PROJECTILE_FLIGHT | REQUIRE_ENTITY_ONLY_AREA),
        ),
    ]
    return _weapon("Weapon_Bomb", FAMILY_BOMB, abilities, 0, {})


def _potions() -> dict:
    abilities = [
        _ability(
            "Potion_Health",
            1.7,
            _heal(1.5, 25, percent=True),
            _status(
                1.5,
                "Potion_Health_Regen",
                5.05,
                cooldown=5,
                healing=35,
                value_percent=True,
            ),
        ),
        _ability(
            "Potion_Stamina",
            0.7,
            _resource(0.5, RESOURCE_STAMINA, 45, percent=True),
            _status(
                0.5,
                "Potion_Stamina_Cooldown",
                15,
                status_flags=STATUS_FLAG_DEBUFF,
            ),
            cooldown=15,
        ),
        _ability(
            "Potion_Antidote",
            1.7,
            _clear_status(1.5, "Poison_T1"),
            _clear_status(1.5, "Poison_T2"),
            _clear_status(1.5, "Poison_T3"),
            _status(
                1.5,
                "Antidote",
                120,
            ),
        ),
    ]
    return _weapon("Combat_Potions", FAMILY_POTION, abilities, 0, {})


def _stoneskin_wand() -> dict:
    return _weapon(
        "Weapon_Wand_Stoneskin",
        FAMILY_WAND,
        [
            _ability(
                "Stoneskin_Cast",
                0.201,
                _status(
                    0.2,
                    "Stoneskin",
                    10,
                    status_flags=STATUS_FLAG_DISABLE_SPRINT,
                    # Stoneskin_Cast.Next enters a non-player Raycast
                    # SelectInteraction before ApplyEffectInteraction.
                    event_flags=EVENT_FLAG_SERVER_SELECTOR,
                    overlap=STATUS_OVERLAP_IGNORE,
                    target=TARGET_OTHER_OR_SELF,
                ),
                requirements=REQUIRE_LINE_OF_SIGHT,
            )
        ],
        0,
        {},
    )


def _skeleton_mage_spellbook() -> dict:
    return _weapon(
        "Skeleton_Sand_Mage_Spellbook_Corruption_Orb",
        FAMILY_SPELLBOOK,
        [
            _ability(
                "Skeleton_Sand_Mage_Spellbook_Corruption_Orb",
                1.2,
                _projectile(
                    1.0,
                    kind=PROJECTILE_CORRUPTION_ORB,
                    damage_class=DAMAGE_CLASS_UNKNOWN,
                    damage=25,
                    cause=DAMAGE_PROJECTILE,
                    speed=30,
                    gravity=0,
                    terminal=50,
                    half_extent=0.1,
                    lifetime=3.1,
                ),
                requirements=REQUIRE_CLEAR_PROJECTILE_FLIGHT,
            )
        ],
        0,
        {},
    )


def _kunai() -> dict:
    return _weapon(
        "Weapon_Kunai",
        FAMILY_KUNAI,
        [
            _ability(
                "Kunai_Throw",
                0.251,
                _projectile(
                    0.25,
                    kind=PROJECTILE_KUNAI,
                    damage_class=DAMAGE_CLASS_UNKNOWN,
                    damage=6,
                    cause=DAMAGE_PHYSICAL,
                    speed=45,
                    gravity=20,
                    terminal=50,
                    half_extent=0.05,
                    lifetime=5,
                    force=(0, 1, -1, 5.5),
                ),
                cooldown=0.25,
                charge_times=(1.0, 1.0, 1.0, 1.0),
                interrupt_recharge=True,
                requirements=REQUIRE_CLEAR_PROJECTILE_FLIGHT,
            )
        ],
        0,
        {},
    )


def _root_wand() -> dict:
    return _weapon(
        "Weapon_Wand_Root",
        FAMILY_ROOT_WAND,
        [
            _ability(
                "Root_Cast",
                0.201,
                _status(
                    0.2,
                    "Root",
                    10,
                    status_flags=(
                        STATUS_FLAG_DISABLE_MOVEMENT
                        | STATUS_FLAG_IGNORE_KNOCKBACK
                        | STATUS_FLAG_CONTROL_IMMUNITY_GATED
                    ),
                    # Root_Cast.Next enters a non-player Raycast
                    # SelectInteraction before ApplyEffectInteraction.
                    event_flags=EVENT_FLAG_SERVER_SELECTOR,
                    overlap=STATUS_OVERLAP_IGNORE,
                    target=TARGET_OTHER,
                ),
                requirements=REQUIRE_LINE_OF_SIGHT,
            )
        ],
        0,
        {},
    )


def _spear(
    asset: str,
    family: int,
    stab_damage: float,
    projectile_damage: float,
    stack_size: int,
) -> dict:
    """Compile the standard player-spear graph with injected Stab selection."""

    abilities = [
        _ability(
            "Spear_Stab",
            0.557,
            _melee(
                0.223,
                stab_damage,
                3.0,
                0.0,
                damage_class=DAMAGE_CLASS_UNKNOWN,
                force=(0.0, 5.0, -5.0, 1.0),
                force_mode=FORCE_ADD,
                force_direction_mode=FORCE_DIRECTION_DIRECTIONAL,
                stab_selector=_STANDARD_SPEAR_STAB_SELECTOR,
            ),
            minimum={RESOURCE_AMMO: 0.1},
        ),
        _ability(
            "Spear_Throw_Charged",
            0.25,
            _projectile(
                0.0,
                kind=PROJECTILE_SPEAR,
                damage_class=DAMAGE_CLASS_UNKNOWN,
                damage=projectile_damage,
                cause=DAMAGE_PROJECTILE,
                speed=50.0,
                gravity=20.0,
                terminal=100.0,
                half_extent=0.05,
                lifetime=10.0,
                dead_time=0.0,
            ),
            _resource(0.0, RESOURCE_AMMO, -1.0),
            minimum={RESOURCE_AMMO: 1.0},
            requirements=REQUIRE_CLEAR_PROJECTILE_FLIGHT,
            requested_charge_time_seconds=1.0,
            # Spear_Attack is a Charging root: the 0 branch is an inline
            # Chaining node onto Spear_Stab and 1.0 throws. It is the one live
            # weapon that sets AllowIndefiniteHold, so the wind-up waits for
            # release instead of firing itself at the threshold.
            hold_table=((0.0, 0), (1.0, 1)),
            hold_allow_indefinite=True,
        ),
    ]
    return _weapon(
        asset,
        family,
        abilities,
        _guard(
            2.0,
            entry_cost=0.0,
            half_angle_degrees=95.0,
            entry_delay_seconds=0.1,
            exit_regen_delay_seconds=0.0,
            required_resource_id=RESOURCE_AMMO,
            required_resource_minimum=0.1,
            # ``Spear_Block`` carries no authored Rules object. It therefore
            # retains Secondary's default standard-input mutual exclusion and
            # does not inherit the seven common guard roots' Primary
            # interruption rule.
            interrupting_type_mask=0,
        ),
        {RESOURCE_AMMO: float(stack_size)},
    )


def _weapon(
    asset: str,
    family: int,
    abilities: list[dict],
    guard: float | dict,
    resources: dict[int, float],
    *,
    native_resource_stat_overrides: dict[int, str] | None = None,
) -> dict:
    resource_stat_ids = list(NATIVE_RESOURCE_STAT_IDS)
    for resource_id, stat_id in dict(native_resource_stat_overrides or {}).items():
        if (
            isinstance(resource_id, bool)
            or not isinstance(resource_id, int)
            or not 0 <= resource_id < RESOURCE_COUNT
        ):
            raise ValueError("native resource stat override ID is outside capacity")
        if not isinstance(stat_id, str) or not stat_id.strip():
            raise ValueError("native resource stat override must name a stat")
        resource_stat_ids[resource_id] = stat_id.strip()
    if len(set(resource_stat_ids)) != len(resource_stat_ids):
        raise ValueError("native resource stat IDs must be unique within a profile")
    return {
        "asset": asset,
        "family": family,
        "abilities": abilities,
        "guard": (_guard(guard) if isinstance(guard, (int, float)) else dict(guard)),
        "resources": resources,
        "native_resource_stat_ids": tuple(resource_stat_ids),
        # Ammo is supplied by the episode loadout. Native entity stats with
        # InitialValue=0 (charges/energy) remain empty and regenerate or earn.
        "initial_resources": {RESOURCE_AMMO: resources[RESOURCE_AMMO]}
        if RESOURCE_AMMO in resources
        else {},
    }


def _guard(
    stamina_value: float,
    *,
    entry_cost: float = 0.5,
    half_angle_degrees: float = 90.0,
    entry_delay_seconds: float = 0.0,
    exit_regen_delay_seconds: float = -1.0,
    required_resource_id: int = -1,
    required_resource_minimum: float = 0.0,
    interrupting_type_mask: int = 1 << INTERACTION_TYPE_PRIMARY,
) -> dict:
    return {
        "entry_cost": entry_cost,
        "stamina_value": stamina_value,
        "half_angle_degrees": half_angle_degrees,
        "entry_delay_seconds": entry_delay_seconds,
        "exit_regen_delay_seconds": exit_regen_delay_seconds,
        "required_resource_id": required_resource_id,
        "required_resource_minimum": required_resource_minimum,
        "interrupting_type_mask": interrupting_type_mask,
    }


def _interaction_type_index(name: str | None) -> int:
    """Map the asset-derived native binding onto its protocol ordinal."""

    return {
        "Primary": INTERACTION_TYPE_PRIMARY,
        "Secondary": INTERACTION_TYPE_SECONDARY,
        "Ability1": INTERACTION_TYPE_ABILITY1,
        "Ability3": INTERACTION_TYPE_ABILITY3,
        None: INTERACTION_TYPE_UNRESOLVED,
    }[name]


def _interaction_type_name(index: int) -> str | None:
    """Map a compiled protocol ordinal back onto the bridge ABI spelling."""

    return {
        INTERACTION_TYPE_PRIMARY: "Primary",
        INTERACTION_TYPE_SECONDARY: "Secondary",
        INTERACTION_TYPE_ABILITY1: "Ability1",
        INTERACTION_TYPE_ABILITY3: "Ability3",
        INTERACTION_TYPE_UNRESOLVED: None,
    }.get(index)


def _names(
    value: str | Sequence[str],
    *,
    expected: int | None = None,
) -> list[str]:
    result = [value] if isinstance(value, str) else list(value)
    if not result:
        raise ValueError("at least one profile is required")
    if expected is not None and len(result) != expected:
        raise ValueError(f"expected {expected} target profiles, got {len(result)}")
    if any(not isinstance(name, str) for name in result):
        raise TypeError("profile names must be strings")
    return result


def _legacy_caster_profile(spec: LegacyCasterVariant) -> dict:
    """Compile a shared Staff or Wand root with its authored cast resource."""

    if spec.archetype == "staff":
        return _legacy_staff(spec)
    if spec.archetype == "spellbook":
        return _legacy_spellbook(spec)
    if spec.archetype == "wand":
        return _legacy_wand(spec)
    raise AssertionError(f"unknown legacy caster archetype {spec.archetype!r}")


def _legacy_staff(spec: LegacyCasterVariant) -> dict:
    selectors = (
        _HorizontalSelectorProgram(
            0.111, 0.1, 3.5, 0.5, 0.5, 90.0, 1.0, -60.0, 0.0, 0.0
        ),
        _HorizontalSelectorProgram(
            0.111, 0.1, 3.5, 0.5, 0.5, 90.0, -1.0, -60.0, 0.0, 0.0
        ),
    )
    abilities = [
        _ability(
            ability_asset,
            0.557,
            _melee(
                0.223,
                5.0,
                3.5,
                45.0,
                damage_class=DAMAGE_CLASS_UNKNOWN,
                force=force,
                horizontal_selector=selector,
            ),
        )
        for ability_asset, force, selector in (
            ("Spear_Swing_Left", (-5.0, 5.0, -5.0, 1.0), selectors[0]),
            ("Spear_Swing_Right", (5.0, 5.0, -5.0, 1.0), selectors[1]),
        )
    ]
    cast_cost, cast_minimum = _compile_resource_scalars(
        spec.scalars.merged(stamina_cost=5.0),
        (
            _ResourceScalarBinding(
                scalar_name="mana_cost",
                resource_id=RESOURCE_MANA,
                cost_kind=RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT,
                commit_time_seconds=0.167,
                commit_flags=EVENT_FLAG_PARALLEL_FORK,
                admission_required=False,
            ),
            _ResourceScalarBinding(
                scalar_name="stamina_cost",
                resource_id=RESOURCE_STAMINA,
                cost_kind=RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT,
                commit_time_seconds=0.167,
                commit_flags=EVENT_FLAG_PARALLEL_FORK,
                admission_required=False,
            ),
        ),
    )
    abilities.append(
        _ability(
            "Staff_Cast_Summon_Charged",
            0.667,
            _legacy_caster_projectile(spec.projectile, 0.167),
            cost=cast_cost,
            minimum=cast_minimum,
            stamina_delay=-1.5,
            requirements=(
                REQUIRE_CLEAR_PROJECTILE_FLIGHT
                | (REQUIRE_ENTITY_ONLY_AREA if spec.projectile == "fireball" else 0)
            ),
            requested_charge_time_seconds=1.0,
            # Staff_Primary: Charging{0 -> swing chain, 1.0 -> this cast},
            # AllowIndefiniteHold true, HorizontalSpeedMultiplier 0.5.
            hold_table=((0.0, 0), (1.0, len(abilities))),
            hold_allow_indefinite=True,
            hold_speed_multiplier=0.5,
        )
    )
    profile = _weapon(
        spec.asset_id,
        spec.family_id,
        abilities,
        0.0,
        {},
    )
    return profile


def _legacy_wand(spec: LegacyCasterVariant) -> dict:
    selectors = (
        _HorizontalSelectorProgram(
            0.05, 0.1, 2.25, 0.5, 0.5, 90.0, 1.0, -60.0, 0.0, 0.0
        ),
        _HorizontalSelectorProgram(
            0.05, 0.1, 2.25, 0.5, 0.5, 90.0, -1.0, -60.0, 0.0, 0.0
        ),
    )
    abilities = [
        _ability(
            ability_asset,
            0.337,
            _melee(
                0.18,
                damage,
                2.25,
                45.0,
                damage_class=DAMAGE_CLASS_UNKNOWN,
                force=(0.0, 1.0, -1.5, 6.5),
                force_mode=FORCE_SET,
                on_hit=(RESOURCE_SIGNATURE_ENERGY, 1.0),
                horizontal_selector=selector,
            ),
        )
        for ability_asset, damage, selector in (
            ("Sword_Swing_Left_Fast", 6.0, selectors[0]),
            ("Sword_Swing_Right_Fast", 8.0, selectors[1]),
        )
    ]
    cast_cost, cast_minimum = _compile_resource_scalars(
        spec.scalars,
        (
            _ResourceScalarBinding(
                scalar_name="mana_cost",
                resource_id=RESOURCE_MANA,
                cost_kind=RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT,
                commit_time_seconds=0.167,
                commit_flags=EVENT_FLAG_PARALLEL_FORK,
                admission_required=False,
            ),
        ),
    )
    abilities.append(
        _ability(
            "Wand_Cast_Left_Charged",
            0.667,
            _legacy_caster_projectile(spec.projectile, 0.167),
            cost=cast_cost,
            minimum=cast_minimum,
            requirements=REQUIRE_CLEAR_PROJECTILE_FLIGHT,
            requested_charge_time_seconds=0.35,
            # Wand_Primary: Charging{0 -> swing chain, 0.35 -> this cast},
            # AllowIndefiniteHold true.
            hold_table=((0.0, 0), (0.35, len(abilities))),
            # Wand_Primary: HorizontalSpeedMultiplier 0.75.
            hold_speed_multiplier=0.75,
            hold_allow_indefinite=True,
        )
    )
    profile = _weapon(
        spec.asset_id,
        spec.family_id,
        abilities,
        0.0,
        {},
    )
    return profile


def _legacy_spellbook(spec: LegacyCasterVariant) -> dict:
    selectors = (
        _HorizontalSelectorProgram(
            0.055, 0.1, 2.5, 0.5, 0.5, 30.0, 1.0, -15.0, 0.0, 45.0
        ),
        _HorizontalSelectorProgram(
            0.055, 0.1, 2.5, 0.5, 0.5, 30.0, -1.0, -15.0, 0.0, 45.0
        ),
        _HorizontalSelectorProgram(
            0.055, 0.1, 2.5, 0.5, 0.5, 30.0, 1.0, -15.0, 0.0, 90.0
        ),
    )
    rows = (
        ("Block_Swing_Left", 0.278),
        ("Block_Swing_Right", 0.361),
        ("Block_Swing_Down", 0.278),
    )
    abilities = [
        _ability(
            ability_asset,
            duration,
            _melee(
                0.111,
                1.0,
                2.5,
                15.0,
                damage_class=DAMAGE_CLASS_UNKNOWN,
                force=(-5.0, 5.0, -5.0, 0.1),
                horizontal_selector=selector,
            ),
        )
        for (ability_asset, duration), selector in zip(rows, selectors, strict=True)
    ]
    cast_cost, cast_minimum = _compile_resource_scalars(
        spec.scalars,
        (
            _ResourceScalarBinding(
                scalar_name="mana_cost",
                resource_id=RESOURCE_MANA,
                cost_kind=RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT,
                commit_time_seconds=0.167,
                commit_flags=EVENT_FLAG_PARALLEL_FORK,
                admission_required=False,
            ),
        ),
    )
    abilities.append(
        _ability(
            "Spellbook_Cast_Hurl_Charged",
            0.667,
            _legacy_caster_projectile(spec.projectile, 0.167),
            cost=cast_cost,
            minimum=cast_minimum,
            requirements=REQUIRE_CLEAR_PROJECTILE_FLIGHT,
            requested_charge_time_seconds=1.0,
            # Spellbook_Primary: Charging{0 -> swing chain, 1.0 -> this cast},
            # AllowIndefiniteHold true, HorizontalSpeedMultiplier 0.8.
            hold_table=((0.0, 0), (1.0, len(abilities))),
            hold_allow_indefinite=True,
            hold_speed_multiplier=0.8,
        )
    )
    profile = _weapon(
        spec.asset_id,
        spec.family_id,
        abilities,
        0.0,
        {},
    )
    return profile


def _legacy_caster_projectile(projectile: str, time: float) -> dict:
    if projectile == "corruption_orb":
        return _projectile(
            time,
            kind=PROJECTILE_CORRUPTION_ORB,
            damage_class=DAMAGE_CLASS_UNKNOWN,
            damage=25.0,
            cause=DAMAGE_PROJECTILE,
            speed=30.0,
            gravity=0.0,
            terminal=50.0,
            half_extent=0.1,
            lifetime=3.1,
            parallel_fork=True,
        )
    if projectile == "ice_ball":
        return _projectile(
            time,
            kind=PROJECTILE_ICE_BALL,
            damage_class=DAMAGE_CLASS_CHARGED,
            damage=20.0,
            cause=DAMAGE_ICE,
            speed=30.0,
            gravity=4.4,
            terminal=42.5,
            half_extent=0.2,
            lifetime=3.0,
            force=(0.0, 3.0, -1.0, 20.0),
            parallel_fork=True,
        )
    if projectile == "fireball":
        return _projectile(
            time,
            kind=PROJECTILE_FIREBALL,
            damage_class=DAMAGE_CLASS_CHARGED,
            # ProjectileComponent applies this payload on death through
            # ExplosionUtils, separately from its direct projectile hit.
            damage=50.0,
            cause=DAMAGE_ENVIRONMENT,
            direct_damage=60.0,
            direct_damage_cause=DAMAGE_PROJECTILE,
            speed=40.0,
            gravity=4.0,
            terminal=100.0,
            half_extent=0.1,
            lifetime=20.0,
            radius=5.0,
            falloff=1.0,
            block_damage_radius=20,
            dead_time=0.0,
            force=(0.0, 0.0, 0.0, 5.0),
            force_direction_mode=FORCE_DIRECTION_POINT,
            entity_only_area=True,
            parallel_fork=True,
        )
    raise AssertionError(f"unknown legacy caster projectile {projectile!r}")


def _legacy_melee_profile(spec: LegacyMeleeVariant) -> dict:
    """Compile one shared legacy root without aliasing its item scalars."""

    if spec.archetype == "axe":
        return _legacy_axe(spec)
    if spec.archetype == "club":
        return _legacy_club(spec)
    if spec.archetype == "flail":
        return _legacy_flail(spec)
    if spec.archetype == "longsword":
        return _legacy_longsword(spec)
    raise AssertionError(f"unknown legacy melee archetype {spec.archetype!r}")


def _legacy_axe(spec: LegacyMeleeVariant) -> dict:
    selectors = (
        _HorizontalSelectorProgram(
            0.133, 0.1, 2.5, 0.5, 0.5, 60.0, 1.0, -30.0, 0.0, 45.0
        ),
        _HorizontalSelectorProgram(
            0.133, 0.1, 2.5, 0.5, 0.5, 60.0, -1.0, -30.0, 0.0, 60.0
        ),
        _HorizontalSelectorProgram(
            0.156, 0.1, 2.5, 0.5, 0.5, 105.0, 1.0, -60.0, 0.0, 0.0
        ),
    )
    rows = (
        (
            "Axe_Swing_Down_Left",
            0.555,
            0.244,
            (-3.0, 5.0, -5.0, 0.5),
            -1.0,
            0,
        ),
        (
            "Axe_Swing_Up_Right",
            0.555,
            0.244,
            (3.0, 5.0, -5.0, 0.5),
            -1.0,
            0,
        ),
        (
            "Axe_Swing_Left_Charged",
            0.334,
            0.023,
            (-5.0, 5.0, -5.0, 1.0),
            1.39,
            0,
        ),
    )
    abilities = [
        _ability(
            asset,
            duration,
            _melee(
                event_time,
                damage,
                2.5,
                52.5 if index == 2 else 30.0,
                damage_class=(
                    DAMAGE_CLASS_CHARGED if index == 2 else DAMAGE_CLASS_UNKNOWN
                ),
                force=force,
                horizontal_selector=selectors[index],
                random_percentage=0.2,
            ),
            requested_charge_time_seconds=requested_charge_time,
            scheduler_prelude_ticks=scheduler_prelude_ticks,
            # Axe_Attack: Charging{0 -> swing chain, 1.39 -> Swing_Left_Charged},
            # AllowIndefiniteHold false.
            hold_table=((0.0, 0), (1.39, 2)) if index == 2 else (),
        )
        for index, (
            (
                asset,
                duration,
                event_time,
                force,
                requested_charge_time,
                scheduler_prelude_ticks,
            ),
            damage,
        ) in enumerate(zip(rows, spec.damage, strict=True))
    ]
    return _weapon(spec.asset_id, spec.family_id, abilities, 0.0, {})


def _legacy_club(spec: LegacyMeleeVariant) -> dict:
    selectors = (
        _HorizontalSelectorProgram(
            0.111, 0.1, 2.5, 0.5, 0.5, 90.0, 1.0, -45.0, 0.0, 30.0
        ),
        _HorizontalSelectorProgram(
            0.111, 0.1, 2.5, 0.5, 0.5, 90.0, -1.0, -45.0, 0.0, -30.0
        ),
    )
    rows = (
        ("Club_Swing_Left", (-5.0, 5.0, -5.0, 1.0)),
        ("Club_Swing_Right", (5.0, 5.0, -5.0, 1.0)),
    )
    abilities = [
        _ability(
            asset,
            0.601,
            _melee(
                0.267,
                damage,
                2.5,
                45.0,
                damage_class=DAMAGE_CLASS_UNKNOWN,
                force=force,
                horizontal_selector=selector,
                random_percentage=0.1,
            ),
        )
        for (asset, force), selector, damage in zip(
            rows,
            selectors,
            spec.damage,
            strict=True,
        )
    ]
    return _weapon(spec.asset_id, spec.family_id, abilities, 0.0, {})


def _legacy_flail(spec: LegacyMeleeVariant) -> dict:
    selectors = (
        _HorizontalSelectorProgram(
            0.2, 0.1, 2.5, 0.5, 0.5, 90.0, 1.0, -45.0, 0.0, 15.0
        ),
        _HorizontalSelectorProgram(
            0.2, 0.1, 2.5, 0.5, 0.5, 90.0, -1.0, -45.0, 0.0, 15.0
        ),
        _HorizontalSelectorProgram(
            0.177, 0.1, 2.5, 0.5, 0.5, 105.0, 1.0, -45.0, 0.0, 75.0
        ),
    )
    rows = (
        (
            "Club_Flail_Swing_Left",
            0.666,
            0.289,
            (-5.0, 5.0, -5.0, 1.0),
            -1.0,
            0,
        ),
        (
            "Club_Flail_Swing_Right",
            0.666,
            0.289,
            (5.0, 5.0, -5.0, 1.0),
            -1.0,
            0,
        ),
        (
            "Club_Flail_Spin_Swing_Left_Charged",
            0.445,
            0.089,
            (-5.0, 5.0, -5.0, 1.5),
            1.39,
            0,
        ),
    )
    abilities = [
        _ability(
            asset,
            duration,
            _melee(
                event_time,
                damage,
                2.5,
                52.5 if index == 2 else 45.0,
                damage_class=(
                    DAMAGE_CLASS_CHARGED if index == 2 else DAMAGE_CLASS_UNKNOWN
                ),
                force=force,
                horizontal_selector=selectors[index],
                random_percentage=0.0 if index == 2 else 0.1,
            ),
            requested_charge_time_seconds=requested_charge_time,
            scheduler_prelude_ticks=scheduler_prelude_ticks,
            # Club_Flail_Attack: Charging{0 -> swing chain,
            # 1.39 -> Spin_Swing_Left_Charged}, AllowIndefiniteHold false.
            hold_table=((0.0, 0), (1.39, 2)) if index == 2 else (),
        )
        for index, (
            (
                asset,
                duration,
                event_time,
                force,
                requested_charge_time,
                scheduler_prelude_ticks,
            ),
            damage,
        ) in enumerate(zip(rows, spec.damage, strict=True))
    ]
    return _weapon(spec.asset_id, spec.family_id, abilities, 0.0, {})


def _legacy_longsword(spec: LegacyMeleeVariant) -> dict:
    damage_cause = {
        "physical": DAMAGE_PHYSICAL,
        "fire": DAMAGE_FIRE,
    }[spec.damage_cause]
    selectors = (
        _HorizontalSelectorProgram(
            0.104, 0.1, 3.0, 0.5, 0.5, 60.0, 1.0, -30.0, 0.0, 40.0
        ),
        _HorizontalSelectorProgram(
            0.104, 0.1, 3.0, 0.5, 0.5, 60.0, -1.0, -30.0, 0.0, 0.0
        ),
        _HorizontalSelectorProgram(
            0.104, 0.1, 3.0, 0.5, 0.5, 60.0, 1.0, -30.0, 0.0, -40.0
        ),
    )
    rows = (
        ("Longsword_Swing_Left", (-5.0, 5.0, -5.0, 0.5)),
        ("Longsword_Swing_Right", (5.0, 5.0, -5.0, 0.5)),
        ("Longsword_Swing_Up_Left", (-3.0, 5.0, -5.0, 0.5)),
    )
    abilities = [
        _ability(
            asset,
            0.52,
            _melee(
                0.229,
                damage,
                3.0,
                30.0,
                damage_class=DAMAGE_CLASS_UNKNOWN,
                force=force,
                horizontal_selector=selector,
                random_percentage=0.15,
                cause=damage_cause,
            ),
        )
        for (asset, force), selector, damage in zip(
            rows,
            selectors,
            spec.damage[:3],
            strict=True,
        )
    ]
    abilities.append(
        _ability(
            "Longsword_Stab_Charged",
            0.334,
            _melee(
                0.0,
                spec.damage[3],
                4.0,
                0.0,
                damage_class=DAMAGE_CLASS_CHARGED,
                force=(0.0, 5.0, -5.0, 1.5),
                stab_selector=_StabSelectorProgram(
                    0.083,
                    0.1,
                    4.0,
                    0.075,
                    0.075,
                    0.075,
                    0.075,
                    0.0,
                    0.0,
                    0.0,
                ),
                random_percentage=0.15,
                cause=damage_cause,
            ),
            requested_charge_time_seconds=1.565,
            # The equipped Primary root executes Charging before jumping into
            # this selected child.  Native advances the child on the next
            # interaction tick; the held 1.565 s itself is not replayed.
            scheduler_prelude_ticks=1,
            # Longsword_Attack: Charging{0 -> swing chain, 1.565 -> this stab},
            # AllowIndefiniteHold false.
            hold_table=((0.0, 0), (1.565, len(abilities))),
        )
    )
    return _weapon(spec.asset_id, spec.family_id, abilities, 0.0, {})


def _void_scythe() -> dict:
    """Compile the deprecated Void Scythe's distinct charging graph."""

    normal = (
        (
            "Spear_Spin_Swing_Left",
            _VOID_SCYTHE_LEFT_SELECTOR,
            (-5.0, 5.0, -5.0, 1.0),
        ),
        (
            "Spear_Spin_Swing_Right",
            _VOID_SCYTHE_RIGHT_SELECTOR,
            (5.0, 5.0, -5.0, 1.0),
        ),
    )
    abilities = [
        _ability(
            ability_asset,
            1.0,
            _melee(
                0.444,
                112.0,
                3.5,
                45.0,
                damage_class=DAMAGE_CLASS_UNKNOWN,
                force=force,
                horizontal_selector=selector,
                random_percentage=0.2,
            ),
        )
        for ability_asset, selector, force in normal
    ]
    abilities.append(
        _ability(
            "Battleaxe_Swing_Left_Charged",
            0.649 + 1.0 / 30.0,
            # ApplyForceInteraction defaults ChangeVelocityType to Set. Keep
            # that authored operation distinct from additive hit knockback.
            _force(
                0.0,
                (0.0, 2.0, -10.0),
                20.0,
                mode=FORCE_SET,
            ),
            _melee(
                1.0 / 30.0,
                224.0,
                3.0,
                60.0,
                damage_class=DAMAGE_CLASS_CHARGED,
                force=(0.0, 1.0, -1.0, 1.11),
                on_hit=(RESOURCE_SIGNATURE_ENERGY, 2.0),
                horizontal_selector=_VOID_SCYTHE_CHARGED_SELECTOR,
                random_percentage=0.2,
            ),
            requested_charge_time_seconds=1.67,
            # Battleaxe_Scythe_Void_Attack: Charging{0 -> spin swing chain,
            # 1.670 -> this charged swing}, AllowIndefiniteHold false.
            hold_table=((0.0, 0), (1.670, len(abilities))),
        )
    )
    # The Scythe item does not override Signature_Whirlwind_Damage. The
    # resolved default damage interaction has no BaseDamage, so its shared
    # Battleaxe signature program is retained with zero damage rather than
    # borrowing Iron's value.
    # Selected by asset, not by position. This borrowed the battleaxe's *last*
    # ability until Downstrike landed and took that slot, which silently gave
    # the Scythe the Downstrike landing's authored 29 damage -- the exact leak
    # `test_void_scythe_uses_its_unique_graph_and_not_iron_battleaxe_damage`
    # exists to catch.
    abilities.append(
        next(
            ability
            for ability in _iron_battleaxe(damage=(0.0, 0.0, 0.0, 0.0))["abilities"]
            if ability["asset"] == "Weapon_Battleaxe_Signature_Whirlwind"
        )
    )
    return _weapon(
        VOID_SCYTHE_ASSET_ID,
        VOID_SCYTHE_FAMILY_ID,
        abilities,
        _guard(
            2.0,
            entry_cost=0.0,
            half_angle_degrees=95.0,
            entry_delay_seconds=0.1,
            exit_regen_delay_seconds=0.0,
            interrupting_type_mask=0,
        ),
        {RESOURCE_SIGNATURE_ENERGY: 9.0},
    )


def _knife_weapon(spec: KnifeWeaponVariant) -> dict:
    """Compile the shared Knife chain used by Arrow and Dart weapon items."""

    selectors = (
        _HorizontalSelectorProgram(
            0.069, 0.1, 2.5, 0.5, 0.5, 30.0, 1.0, -15.0, 0.0, 30.0
        ),
        _HorizontalSelectorProgram(
            0.069, 0.1, 2.5, 0.5, 0.5, 30.0, -1.0, -15.0, 0.0, 30.0
        ),
        _HorizontalSelectorProgram(
            0.069, 0.1, 2.5, 0.5, 0.5, 30.0, 1.0, -15.0, 0.0, 75.0
        ),
        _HorizontalSelectorProgram(
            0.069, 0.1, 2.5, 0.5, 0.5, 30.0, 1.0, -15.0, 0.0, -75.0
        ),
    )
    rows = (
        ("Knife_Swing_Left", (-5.0, 5.0, -5.0, 0.5)),
        ("Knife_Swing_Right", (5.0, 5.0, -5.0, 0.5)),
        ("Knife_Stab", (0.0, 5.0, -5.0, 0.5)),
        ("Knife_Lunge", (0.0, 5.0, -5.0, 0.5)),
    )
    abilities = [
        _ability(
            asset,
            0.487,
            _melee(
                0.139,
                spec.melee_damage,
                2.5,
                15.0,
                damage_class=DAMAGE_CLASS_UNKNOWN,
                force=force,
                horizontal_selector=selector,
                random_percentage=spec.random_percentage,
            ),
        )
        for (asset, force), selector in zip(rows, selectors, strict=True)
    ]
    abilities.append(
        _ability(
            "Knife_Throw_Charged",
            0.25,
            _projectile(
                0.0,
                kind=PROJECTILE_LEGACY_ARROW,
                damage_class=DAMAGE_CLASS_UNKNOWN,
                damage=2.0,
                cause=DAMAGE_PROJECTILE,
                speed=35.0,
                gravity=25.0,
                terminal=50.0,
                half_extent=0.075,
                lifetime=20.0,
                dead_time=0.1,
            ),
            requirements=REQUIRE_CLEAR_PROJECTILE_FLIGHT,
            requested_charge_time_seconds=0.5,
            # Knife_Attack: Charging{0 -> swing chain, 0.5 -> this throw},
            # AllowIndefiniteHold true.
            hold_table=((0.0, 0), (0.5, len(abilities))),
            hold_allow_indefinite=True,
        )
    )
    # Knife_Block has a fixed 0.7 Physical/Projectile modifier and no stamina
    # cost. The current stamina-valued guard representation cannot encode it
    # without inventing a drain, so the graph is deliberately fail-closed.
    return _weapon(spec.asset_id, spec.family_id, abilities, 0.0, {})


def _double_incandescent_spear() -> dict:
    """Compile the two-swing legacy spear instead of aliasing standard Stab."""

    normal = (
        (
            "Spear_Spin_Swing_Left",
            _VOID_SCYTHE_LEFT_SELECTOR,
            (-5.0, 5.0, -5.0, 1.0),
        ),
        (
            "Spear_Spin_Swing_Right",
            _VOID_SCYTHE_RIGHT_SELECTOR,
            (5.0, 5.0, -5.0, 1.0),
        ),
    )
    abilities = [
        _ability(
            asset,
            1.0,
            _melee(
                0.444,
                14.0,
                3.5,
                45.0,
                damage_class=DAMAGE_CLASS_UNKNOWN,
                force=force,
                horizontal_selector=selector,
                random_percentage=0.15,
            ),
            minimum={RESOURCE_AMMO: 0.1},
        )
        for asset, selector, force in normal
    ]
    abilities.append(
        _ability(
            "Spear_Throw_Charged",
            0.25,
            _projectile(
                0.0,
                kind=PROJECTILE_SPEAR,
                damage_class=DAMAGE_CLASS_UNKNOWN,
                damage=24.0,
                cause=DAMAGE_PROJECTILE,
                speed=50.0,
                gravity=20.0,
                terminal=100.0,
                half_extent=0.05,
                lifetime=10.0,
                dead_time=0.0,
            ),
            _resource(0.0, RESOURCE_AMMO, -1.0),
            minimum={RESOURCE_AMMO: 1.0},
            requirements=REQUIRE_CLEAR_PROJECTILE_FLIGHT,
            requested_charge_time_seconds=1.0,
            # Spear_Double_Incandescent_Attack is the same Charging shape as
            # Spear_Attack -- the 0 branch is an inline Chaining node (here onto
            # *both* spin swings, so it enters at slot 0) and 1.0 throws --
            # including AllowIndefiniteHold, so the wind-up waits for release.
            hold_table=((0.0, 0), (1.0, 2)),
            hold_allow_indefinite=True,
        )
    )
    return _weapon(
        DOUBLE_INCANDESCENT_SPEAR_ASSET_ID,
        DOUBLE_INCANDESCENT_SPEAR_FAMILY_ID,
        abilities,
        _guard(
            2.0,
            entry_cost=0.0,
            half_angle_degrees=95.0,
            entry_delay_seconds=0.1,
            exit_regen_delay_seconds=0.0,
            required_resource_id=RESOURCE_AMMO,
            required_resource_minimum=0.1,
            interrupting_type_mask=0,
        ),
        {RESOURCE_AMMO: 30.0},
    )


def _tribal_claws() -> dict:
    """Compile the deprecated two-swing/two-charge Claws graph."""

    selectors = (
        _HorizontalSelectorProgram(
            0.069, 0.1, 2.5, 0.5, 0.5, 30.0, 1.0, -15.0, 0.0, 30.0
        ),
        _HorizontalSelectorProgram(
            0.069, 0.1, 2.5, 0.5, 0.5, 30.0, -1.0, -15.0, 0.0, 30.0
        ),
        _HorizontalSelectorProgram(
            0.069, 0.1, 2.5, 0.5, 0.5, 90.0, 1.0, -45.0, 0.0, 90.0
        ),
        _HorizontalSelectorProgram(
            0.069, 0.1, 2.5, 0.5, 0.5, -90.0, 1.0, 45.0, 0.0, 90.0
        ),
    )
    # Daggers_Claw_Bone_Attack is a Condition on Crouching whose two arms are
    # Charging{0 -> swing chain, 2.43 -> Stab_Double} and
    # Charging{0 -> swing chain, 2.43 -> Lunge_Double}. Both charged children
    # are Simple 0 -> Parallel[Selector 0.069 -> pad 0.139], so each is 0.208 s
    # long with its melee at 0.0 once the 2.43 s hold is metered by C1.
    rows = (
        ("Daggers_Swing_Left_Right", 0.487, 0.139, (-5.0, 5.0, -5.0, 0.5)),
        ("Daggers_Swing_Right_Left", 0.487, 0.139, (5.0, 5.0, -5.0, 0.5)),
        ("Daggers_Stab_Double_Charged", 0.208, 0.0, (0.0, 5.0, -5.0, 1.0)),
        ("Daggers_Lunge_Double_Charged", 0.208, 0.0, (0.0, 5.0, -5.0, 1.0)),
    )
    _CLAWS_CHARGED_HOLD_SECONDS = 2.43
    abilities = [
        _ability(
            asset,
            duration,
            _melee(
                event_time,
                5.0,
                2.5,
                15.0,
                damage_class=(
                    DAMAGE_CLASS_CHARGED if index >= 2 else DAMAGE_CLASS_UNKNOWN
                ),
                force=force,
                horizontal_selector=selectors[index],
                random_percentage=0.2,
            ),
            # The 2.43 s used to be encoded as a recharge-pool refill time,
            # which is a rate limit rather than a hold. C1 now meters the hold
            # itself, so the pool returns to its instant default.
            hold_table=(
                ((0.0, 0), (_CLAWS_CHARGED_HOLD_SECONDS, index))
                if index >= 2
                else ()
            ),
        )
        for index, (asset, duration, event_time, force) in enumerate(rows)
    ]
    return _weapon(
        TRIBAL_CLAWS_ASSET_ID,
        TRIBAL_CLAWS_FAMILY_ID,
        abilities,
        0.0,
        {},
    )


def _tribal_blowgun() -> dict:
    """Compile the item's inherited Sword roots with their zero damage maps."""

    return _iron_sword(
        asset=TRIBAL_BLOWGUN_ASSET_ID,
        family=TRIBAL_BLOWGUN_FAMILY_ID,
        damage=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        guard_stamina_value=7.0,
    )


_PROTOTYPE_BOW_ENERGY = (0.54, 0.81, 1.08, 1.35, 1.62, 1.89, 2.16, 2.43, 2.7, 3.0)


def _prototype_bow_profile(spec: PrototypeBowSpec) -> dict:
    """Compile one prototype bow without erasing its signature mechanic."""

    abilities = _prototype_bow_primary(spec)
    abilities.append(_prototype_bow_signature(spec.signature))
    resources = {RESOURCE_SIGNATURE_ENERGY: 100.0}
    if spec.primary in {"combat", "vampire"} or spec.signature == "trishot":
        resources[RESOURCE_AMMO] = 3.0
    return _weapon(spec.asset_id, spec.family_id, abilities, 0.0, resources)


def _prototype_bow_primary(spec: PrototypeBowSpec) -> list[dict]:
    if spec.primary == "vampire":
        rows = (
            (0.2, 2.0, 6.0, 3.0, 0.54, 2.0),
            (0.6, 12.0, 36.0, 8.0, 1.89, 12.0),
            (1.0, 20.0, 60.0, 12.0, 3.0, 20.0),
        )
        projectile_kind = PROJECTILE_PROTOTYPE_ARROW
        cause = DAMAGE_PHYSICAL
        standard_physics = True
        bounciness = 0.0
        arrow_cost = True
    else:
        rows = tuple(
            (
                index / 10.0,
                float(index * 2),
                float(index * (6 if spec.primary == "ricochet" else 8)),
                float(index + 2),
                _PROTOTYPE_BOW_ENERGY[index - 1],
                0.0,
            )
            for index in range(1, 11)
        )
        projectile_kind = PROJECTILE_ARROW
        cause = DAMAGE_PHYSICAL if spec.primary == "ricochet" else DAMAGE_PROJECTILE
        standard_physics = spec.primary == "ricochet"
        bounciness = 0.7 if spec.primary == "ricochet" else 0.0
        arrow_cost = spec.primary == "combat"

    # Bow_{Combat,Ricochet,Vamp}_Shoot_Charging is one ten-row draw ladder at
    # 0.1 through 1.0, AllowIndefiniteHold true (Ricochet inherits the Combat
    # root via Parent). Every tier is the same 0.2 s launch child differing only
    # in its projectile Config, so the tiers are timed from the branch and the
    # ladder itself is the hold table.
    #
    # The vampire profile compiles three of the ten authored tiers (0.2/0.6/1.0)
    # rather than all ten, so its table is a faithful *subset*: a hold that
    # would select an uncompiled tier resolves down to the nearest compiled one
    # below it, which is what jumpToChargeValue does anyway.
    hold_table = tuple((row[0], index) for index, row in enumerate(rows))
    abilities = []
    for charge, damage, speed, force, energy, healing in rows:
        cost = (
            {
                RESOURCE_AMMO: _single_application_cost(
                    1.0,
                    commit_time_seconds=0.0,
                )
            }
            if arrow_cost
            else None
        )
        minimum = {RESOURCE_AMMO: 1.0} if arrow_cost else None
        requirements = (
            REQUIRE_WORLD_PROJECTILE_COLLISION
            if spec.primary == "ricochet"
            else REQUIRE_CLEAR_PROJECTILE_FLIGHT
        )
        charge_index = int(round(charge * 10.0))
        ability_asset = {
            "combat": f"Projectile_Config_Bow_Combat_Charge_{charge_index:02d}",
            "ricochet": f"Projectile_Config_Bow_Ricochet_Charge_{charge_index:02d}",
            "vampire": f"Projectile_Config_Bow_Vamp_Charge_{charge_index:02d}",
        }[spec.primary]
        abilities.append(
            _ability(
                ability_asset,
                0.2,
                _projectile(
                    0.0,
                    kind=projectile_kind,
                    damage_class=DAMAGE_CLASS_CHARGED,
                    damage=damage,
                    cause=cause,
                    speed=speed,
                    gravity=30.0 if spec.primary == "ricochet" else 15.0,
                    terminal=50.0,
                    standard_physics=standard_physics,
                    bounciness=bounciness,
                    bounce_count=-1 if spec.primary == "ricochet" else 0,
                    force=(0.0, 0.5, -1.0, force),
                    force_mode=FORCE_ADD,
                    on_hit=(RESOURCE_SIGNATURE_ENERGY, energy),
                    on_hit_healing=healing,
                ),
                cost=cost,
                minimum=minimum,
                # The draw used to be encoded as a recharge-pool refill time,
                # which is a rate limit rather than a hold.
                charge_times=(0.0,),
                hold_table=hold_table,
                hold_allow_indefinite=True,
                requirements=requirements,
            )
        )
    return abilities


def _prototype_bow_signature(signature: str) -> dict:
    cost = {RESOURCE_SIGNATURE_ENERGY: _instant_add_stat_cost(100.0)}
    minimum = {RESOURCE_SIGNATURE_ENERGY: 100.0}
    if signature in {"trishot", "ricochet"}:
        ricochet = signature == "ricochet"
        events = tuple(
            _projectile(
                1.0 + 0.01 * index,
                kind=PROJECTILE_PROTOTYPE_ARROW,
                damage_class=DAMAGE_CLASS_SIGNATURE,
                damage=10.0,
                cause=DAMAGE_PHYSICAL,
                speed=64.0,
                gravity=15.0,
                terminal=50.0,
                standard_physics=ricochet,
                bounciness=1.0 if ricochet else 0.0,
                bounce_count=-1 if ricochet else 0,
                force=(0.0, 0.5, -1.0, 10.0),
                force_mode=FORCE_ADD,
            )
            for index in range(3)
        )
        if not ricochet:
            cost[RESOURCE_AMMO] = _single_application_cost(
                3.0,
                commit_time_seconds=1.0,
            )
            minimum[RESOURCE_AMMO] = 3.0
        return _ability(
            "Bow_Ricochet_Trishot" if ricochet else "Bow_Combat_Trishot",
            1.22,
            *events,
            cost=cost,
            minimum=minimum,
            cooldown=0.5,
            requirements=(
                REQUIRE_WORLD_PROJECTILE_COLLISION
                if ricochet
                else REQUIRE_CLEAR_PROJECTILE_FLIGHT
            ),
        )
    if signature == "bomb":
        return _ability(
            "Bow_Bomb_Boomshot",
            1.2,
            _force(0.0, (0.0, 1.0, 0.0), 20.0, mode=FORCE_SET),
            _projectile(
                1.0,
                kind=PROJECTILE_PROTOTYPE_ARROW,
                damage_class=DAMAGE_CLASS_SIGNATURE,
                damage=20.0,
                cause=DAMAGE_PHYSICAL,
                speed=64.0,
                gravity=15.0,
                terminal=50.0,
                radius=5.0,
                block_damage_radius=1,
                force=(0.0, 0.0, 0.0, 30.0),
                force_direction_mode=FORCE_DIRECTION_POINT,
                entity_only_area=True,
            ),
            cost=cost,
            minimum=minimum,
            cooldown=0.5,
            requirements=(REQUIRE_CLEAR_PROJECTILE_FLIGHT | REQUIRE_ENTITY_ONLY_AREA),
        )
    if signature == "pull":
        return _ability(
            "Bow_Pull_Pullshot",
            1.2,
            _projectile(
                1.0,
                kind=PROJECTILE_PROTOTYPE_ARROW,
                damage_class=DAMAGE_CLASS_SIGNATURE,
                damage=20.0,
                cause=DAMAGE_PHYSICAL,
                speed=64.0,
                gravity=15.0,
                terminal=50.0,
                force=(0.0, 0.3, 1.0, 40.0),
                force_mode=FORCE_ADD,
            ),
            cost=cost,
            minimum=minimum,
            cooldown=0.5,
            requirements=REQUIRE_CLEAR_PROJECTILE_FLIGHT,
        )
    if signature == "vampire":
        return _ability(
            "Bow_Vamp_Supershot",
            1.2,
            _projectile(
                1.0,
                kind=PROJECTILE_PROTOTYPE_ARROW,
                damage_class=DAMAGE_CLASS_SIGNATURE,
                damage=20.0,
                cause=DAMAGE_PHYSICAL,
                speed=64.0,
                gravity=15.0,
                terminal=50.0,
                force=(0.0, 0.5, -1.0, 5.5),
                force_mode=FORCE_ADD,
                on_hit_healing=100.0,
            ),
            cost=cost,
            minimum=minimum,
            cooldown=0.5,
            requirements=REQUIRE_CLEAR_PROJECTILE_FLIGHT,
        )
    raise AssertionError(f"unknown prototype bow signature {signature!r}")


def _gun_profile(spec: GunSpec) -> dict:
    """Compile the legacy firearm projectile plus its DPS-scaled stock hit."""

    kind = (
        PROJECTILE_BLUNDERBUSS_BULLET
        if spec.projectile_id == "Gun_Blunderbuss_Bullet"
        else PROJECTILE_GUN_BULLET
    )
    primary_cost = (
        {RESOURCE_MANA: _instant_add_stat_cost(spec.mana_cost)}
        if spec.mana_cost > 0.0
        else None
    )
    primary_minimum = (
        {RESOURCE_MANA: spec.mana_minimum} if spec.mana_minimum > 0.0 else None
    )
    # Gun_Shoot_Flintlock_Charging is a Charging root with a single authored
    # row -- {0 -> nothing, 2.0 -> Gun_Shoot_Flintlock_Charged} at
    # AllowIndefiniteHold true -- so releasing early fires nothing at all. C1
    # meters that hold, which makes the child's own timeline start at the
    # launch. Guns with charge_seconds == 0 (Gun_Shoot) have no such root.
    charging_root = spec.charge_seconds > 0.0
    abilities = [
        _ability(
            spec.ability_asset_id,
            # Zero-runtime inline GunPvP launch nodes still occupy one server
            # tick.  The scheduler runs at 30 Hz, so retain one 1/30-second
            # transition instead of inventing a longer animation window.
            max(
                1.0 / 30.0,
                (0.0 if charging_root else spec.charge_seconds)
                + spec.launch_runtime_seconds,
            ),
            _projectile(
                0.0 if charging_root else spec.charge_seconds,
                kind=kind,
                damage_class=DAMAGE_CLASS_CHARGED,
                damage=spec.damage,
                cause=DAMAGE_PROJECTILE,
                speed=spec.speed,
                gravity=10.0,
                terminal=spec.speed,
                half_extent=0.1,
                lifetime=50.0,
                dead_time=0.1,
            ),
            cost=primary_cost,
            minimum=primary_minimum,
            cooldown=spec.cooldown_seconds,
            # The hold used to be encoded as a recharge-pool refill time, which
            # is a rate limit rather than a hold.
            charge_times=(0.0,) if charging_root else (spec.charge_seconds,),
            hold_table=((spec.charge_seconds, 0),) if charging_root else (),
            # Gun_Shoot_Flintlock_Charging: HorizontalSpeedMultiplier 0.5.
            hold_speed_multiplier=0.5 if charging_root else 1.0,
            hold_allow_indefinite=charging_root,
            requirements=REQUIRE_CLEAR_PROJECTILE_FLIGHT,
        ),
        _ability(
            "Gun_Attack",
            0.584,
            _melee(
                0.167,
                # Type=Dps scales BaseDamage=5 by Selector.RunTime=0.083.
                5.0 * 0.083,
                2.5,
                15.0,
                damage_class=DAMAGE_CLASS_UNKNOWN,
                force=(-5.0, 5.0, -5.0, 0.5),
                # DirectionalKnockback adds the authored relative X/Z vector
                # to the normalized target-to-source direction before
                # applying Force, while VelocityY remains absolute. Treating
                # this tuple as a normalized local vector under-applies the
                # stock hit by nearly an order of magnitude.
                force_direction_mode=FORCE_DIRECTION_DIRECTIONAL,
                horizontal_selector=_HorizontalSelectorProgram(
                    0.083,
                    0.1,
                    2.5,
                    0.5,
                    0.5,
                    30.0,
                    1.0,
                    -15.0,
                    0.0,
                    0.0,
                ),
            ),
        ),
    ]
    return _weapon(spec.asset_id, spec.family_id, abilities, 0.0, {})


def _deployable_terminal_profile(spec: ProjectileTerminalDeployableSpec) -> dict:
    """Compile one inline Primary without claiming a native named root."""

    adapted = adapt_projectile_terminal_deployable(spec)
    event = _event(
        EVENT_PROJECTILE,
        adapted.event_time_seconds,
        f=adapted.f32,
        i=adapted.i32,
        flags=adapted.flags,
    )
    ability = _ability(
        spec.projectile_config_id,
        adapted.ability_duration_seconds,
        event,
        cooldown=adapted.cooldown_seconds,
        requirements=(
            REQUIRE_WORLD_PROJECTILE_COLLISION
            | REQUIRE_ENTITY_ONLY_AREA
            | REQUIRE_DEPLOYABLE_INTENDED_GRAPH_EVIDENCE
        ),
        interaction_type=INTERACTION_TYPE_PRIMARY,
        native_bindable=False,
    )
    return _weapon(spec.asset_id, spec.family_id, [ability], 0.0, {})


def _scalar_variant_profile(spec: ScalarVariant) -> dict:
    """Apply item-resolved scalars to a graph-compatible family template."""

    common = {
        "asset": spec.asset_id,
        "family": spec.family_id,
        "guard_stamina_value": spec.guard_stamina_value,
    }
    if spec.template == "sword":
        return _iron_sword(damage=spec.damage, **common)
    if spec.template == "mace":
        return _iron_mace(damage=spec.damage, **common)
    if spec.template == "battleaxe":
        return _iron_battleaxe(damage=spec.damage, **common)
    if spec.template == "daggers":
        return _iron_daggers(
            damage=spec.damage,
            angled_damage=spec.angled_damage,
            **common,
        )
    if spec.template == "shortbow":
        return _iron_shortbow(damage=spec.damage, **common)
    if spec.template == "crossbow":
        return _iron_crossbow(damage=spec.damage, **common)
    if spec.template == "shield":
        return _iron_shield(**common)
    raise AssertionError(f"unknown scalar weapon template {spec.template!r}")


_CATALOG = {
    "iron_sword": _iron_sword,
    "iron_mace": _iron_mace,
    "iron_battleaxe": _iron_battleaxe,
    "iron_daggers": _iron_daggers,
    "iron_shortbow": _iron_shortbow,
    "iron_crossbow": _iron_crossbow,
    "iron_shield": _iron_shield,
    "flame_staff": _flame_staff,
    "ice_staff": _ice_staff,
    "bombs": _bombs,
    "potions": _potions,
    "stoneskin_wand": _stoneskin_wand,
    "skeleton_mage_spellbook": _skeleton_mage_spellbook,
    "kunai": _kunai,
    "root_wand": _root_wand,
    "adamantite_spear": partial(
        _spear,
        "Weapon_Spear_Adamantite",
        FAMILY_SPEAR_ADAMANTITE,
        10,
        24,
        30,
    ),
    "adamantite_saurian_spear": partial(
        _spear,
        "Weapon_Spear_Adamantite_Saurian",
        FAMILY_SPEAR_ADAMANTITE_SAURIAN,
        10,
        24,
        5,
    ),
    "bone_spear": partial(
        _spear,
        "Weapon_Spear_Bone",
        FAMILY_SPEAR_BONE,
        8,
        22,
        5,
    ),
    "bronze_spear": partial(
        _spear,
        "Weapon_Spear_Bronze",
        FAMILY_SPEAR_BRONZE,
        6,
        14,
        5,
    ),
    "cobalt_spear": partial(
        _spear,
        "Weapon_Spear_Cobalt",
        FAMILY_SPEAR_COBALT,
        8,
        19,
        5,
    ),
    "copper_spear": partial(
        _spear,
        "Weapon_Spear_Copper",
        FAMILY_SPEAR_COPPER,
        5,
        12,
        30,
    ),
    "crude_spear": partial(
        _spear,
        "Weapon_Spear_Crude",
        FAMILY_SPEAR_CRUDE,
        4,
        10,
        5,
    ),
    "fishbone_spear": partial(
        _spear,
        "Weapon_Spear_Fishbone",
        FAMILY_SPEAR_FISHBONE,
        8,
        22,
        5,
    ),
    "iron_spear": partial(
        _spear,
        "Weapon_Spear_Iron",
        FAMILY_SPEAR_IRON,
        6,
        15,
        30,
    ),
    "leaf_spear": partial(
        _spear,
        "Weapon_Spear_Leaf",
        FAMILY_SPEAR_LEAF,
        6,
        14,
        5,
    ),
    "mithril_spear": partial(
        _spear,
        "Weapon_Spear_Mithril",
        FAMILY_SPEAR_MITHRIL,
        12,
        30,
        30,
    ),
    "onyxium_spear": partial(
        _spear,
        "Weapon_Spear_Onyxium",
        FAMILY_SPEAR_ONYXIUM,
        6,
        15,
        5,
    ),
    "scrap_spear": partial(
        _spear,
        "Weapon_Spear_Scrap",
        FAMILY_SPEAR_SCRAP,
        6,
        14,
        5,
    ),
    "stone_trork_spear": partial(
        _spear,
        "Weapon_Spear_Stone_Trork",
        FAMILY_SPEAR_STONE_TRORK,
        7,
        17,
        5,
    ),
    "thorium_spear": partial(
        _spear,
        "Weapon_Spear_Thorium",
        FAMILY_SPEAR_THORIUM,
        8,
        19,
        5,
    ),
    "tribal_spear": partial(
        _spear,
        "Weapon_Spear_Tribal",
        FAMILY_SPEAR_TRIBAL,
        8,
        19,
        30,
    ),
}
_CATALOG.update(
    {spec.profile: partial(_scalar_variant_profile, spec) for spec in SCALAR_VARIANTS}
)
_CATALOG[VOID_SCYTHE_PROFILE_NAME] = _void_scythe
_CATALOG.update(
    {
        spec.profile: partial(_legacy_melee_profile, spec)
        for spec in LEGACY_MELEE_VARIANTS
    }
)
_CATALOG.update(
    {
        spec.profile: partial(_legacy_caster_profile, spec)
        for spec in LEGACY_CASTER_VARIANTS
    }
)
_CATALOG.update(
    {spec.profile: partial(_knife_weapon, spec) for spec in KNIFE_WEAPON_VARIANTS}
)
_CATALOG[DOUBLE_INCANDESCENT_SPEAR_PROFILE_NAME] = _double_incandescent_spear
_CATALOG[TRIBAL_CLAWS_PROFILE_NAME] = _tribal_claws
_CATALOG[TRIBAL_BLOWGUN_PROFILE_NAME] = _tribal_blowgun
_CATALOG.update(
    {
        spec.profile: partial(_prototype_bow_profile, spec)
        for spec in PROTOTYPE_BOW_SPECS
    }
)
_CATALOG.update({spec.profile: partial(_gun_profile, spec) for spec in GUN_SPECS})
_CATALOG.update(
    {
        spec.profile: partial(_deployable_terminal_profile, spec)
        for spec in DEPLOYABLE_SPECS
    }
)
