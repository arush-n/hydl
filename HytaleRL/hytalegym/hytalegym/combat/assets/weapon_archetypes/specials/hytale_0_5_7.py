"""Pinned Hytale 0.5.7 weapon rows with small, non-template graphs."""

from __future__ import annotations

from dataclasses import dataclass

from .guns import GUN_PROFILE_NAMES
from .prototype_bows import PROTOTYPE_BOW_PROFILE_NAMES


@dataclass(frozen=True)
class KnifeWeaponVariant:
    """An Arrow/Dart item using the shared Knife attack-and-throw graph."""

    profile: str
    asset_id: str
    family_id: int
    melee_damage: float
    random_percentage: float
    has_block: bool


KNIFE_WEAPON_VARIANTS = (
    KnifeWeaponVariant("clearshot_arrow", "Weapon_Arrow_Clearshot", 205, 10.0, 0.1, True),
    KnifeWeaponVariant("crude_arrow", "Weapon_Arrow_Crude", 206, 1.0, 0.0, True),
    KnifeWeaponVariant("deadeye_arrow", "Weapon_Arrow_Deadeye", 207, 10.0, 0.1, True),
    KnifeWeaponVariant("iron_arrow", "Weapon_Arrow_Iron", 208, 10.0, 0.1, True),
    KnifeWeaponVariant("trueshot_arrow", "Weapon_Arrow_Trueshot", 209, 10.0, 0.1, True),
    KnifeWeaponVariant("tribal_dart", "Weapon_Dart_Tribal", 210, 10.0, 0.1, False),
)
KNIFE_WEAPON_PROFILE_NAMES = tuple(row.profile for row in KNIFE_WEAPON_VARIANTS)

DOUBLE_INCANDESCENT_SPEAR_PROFILE_NAME = "double_incandescent_spear"
DOUBLE_INCANDESCENT_SPEAR_ASSET_ID = "Weapon_Spear_Double_Incandescent"
DOUBLE_INCANDESCENT_SPEAR_FAMILY_ID = 211

TRIBAL_CLAWS_PROFILE_NAME = "tribal_claws"
TRIBAL_CLAWS_ASSET_ID = "Weapon_Claws_Tribal"
TRIBAL_CLAWS_FAMILY_ID = 212

TRIBAL_BLOWGUN_PROFILE_NAME = "tribal_blowgun"
TRIBAL_BLOWGUN_ASSET_ID = "Weapon_Blowgun_Tribal"
TRIBAL_BLOWGUN_FAMILY_ID = 213

SPECIAL_WEAPON_PROFILE_NAMES = KNIFE_WEAPON_PROFILE_NAMES + (
    DOUBLE_INCANDESCENT_SPEAR_PROFILE_NAME,
    TRIBAL_CLAWS_PROFILE_NAME,
    TRIBAL_BLOWGUN_PROFILE_NAME,
) + PROTOTYPE_BOW_PROFILE_NAMES + GUN_PROFILE_NAMES

assert len(KNIFE_WEAPON_VARIANTS) == 6
assert {row.family_id for row in KNIFE_WEAPON_VARIANTS} == set(range(205, 211))
assert len(set(SPECIAL_WEAPON_PROFILE_NAMES)) == len(SPECIAL_WEAPON_PROFILE_NAMES)
