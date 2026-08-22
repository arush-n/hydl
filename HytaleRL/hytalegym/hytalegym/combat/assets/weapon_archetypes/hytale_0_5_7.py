"""Hytale 0.5.7 weapons whose interaction graphs are not template clones."""

from __future__ import annotations

from dataclasses import dataclass

from .scalars import WeaponScalarSet


VOID_SCYTHE_ASSET_ID = "Weapon_Battleaxe_Scythe_Void"
VOID_SCYTHE_FAMILY_ID = 120
VOID_SCYTHE_PROFILE_NAME = "void_scythe"


@dataclass(frozen=True)
class LegacyMeleeVariant:
    """One item using a shared legacy melee root with item-resolved damage."""

    profile: str
    asset_id: str
    family_id: int
    archetype: str
    max_durability: float
    durability_loss_on_hit: float
    damage: tuple[float, ...]
    damage_cause: str = "physical"


@dataclass(frozen=True)
class LegacyCasterVariant:
    """One shared caster root with resolved projectile and named scalars."""

    profile: str
    asset_id: str
    family_id: int
    archetype: str
    projectile: str
    scalars: WeaponScalarSet

    @property
    def mana_cost(self) -> float:
        """Compatibility accessor for the pinned 0.5.7 asset field."""

        return self.scalars.value("mana_cost")


_ROWS = (
    ("adamantite_axe", "Weapon_Axe_Adamantite", 121, "axe", 150, 0.56, (52, 52, 104)),
    ("bone_axe", "Weapon_Axe_Bone", 122, "axe", 75, 0.56, (14, 14, 28)),
    ("cobalt_axe", "Weapon_Axe_Cobalt", 123, "axe", 140, 0.56, (36, 36, 72)),
    ("copper_axe", "Weapon_Axe_Copper", 124, "axe", 90, 0.56, (12, 12, 24)),
    ("crude_axe", "Weapon_Axe_Crude", 125, "axe", 0, 0.56, (8, 8, 16)),
    ("doomed_axe", "Weapon_Axe_Doomed", 126, "axe", 135, 0.56, (43, 43, 86)),
    ("iron_axe", "Weapon_Axe_Iron", 127, "axe", 120, 0.56, (17, 17, 34)),
    ("rusty_iron_axe", "Weapon_Axe_Iron_Rusty", 128, "axe", 81, 0.56, (19, 19, 38)),
    ("mithril_axe", "Weapon_Axe_Mithril", 129, "axe", 160, 0.56, (75, 75, 150)),
    ("onyxium_axe", "Weapon_Axe_Onyxium", 130, "axe", 170, 0.56, (108, 108, 216)),
    ("stone_trork_axe", "Weapon_Axe_Stone_Trork", 131, "axe", 45, 0.56, (10, 10, 20)),
    ("thorium_axe", "Weapon_Axe_Thorium", 132, "axe", 130, 0.56, (24, 24, 48)),
    ("tribal_axe", "Weapon_Axe_Tribal", 133, "axe", 120, 0.56, (17, 17, 34)),
    ("adamantite_club", "Weapon_Club_Adamantite", 134, "club", 150, 0.56, (52, 52)),
    ("cobalt_club", "Weapon_Club_Cobalt", 135, "club", 140, 0.56, (36, 36)),
    ("copper_club", "Weapon_Club_Copper", 136, "club", 90, 0.56, (12, 12)),
    ("crude_club", "Weapon_Club_Crude", 137, "club", 60, 0.56, (4, 4)),
    ("doomed_club", "Weapon_Club_Doomed", 138, "club", 135, 0.56, (43, 43)),
    ("iron_club", "Weapon_Club_Iron", 139, "club", 120, 0.56, (17, 17)),
    ("rusty_iron_club", "Weapon_Club_Iron_Rusty", 140, "club", 81, 0.56, (19, 19)),
    ("mithril_club", "Weapon_Club_Mithril", 141, "club", 160, 0.56, (75, 75)),
    ("onyxium_club", "Weapon_Club_Onyxium", 142, "club", 170, 0.56, (108, 108)),
    ("scrap_club", "Weapon_Club_Scrap", 143, "club", 45, 0.56, (6, 10)),
    ("stone_trork_club", "Weapon_Club_Stone_Trork", 144, "club", 45, 0.56, (10, 10)),
    ("thorium_club", "Weapon_Club_Thorium", 145, "club", 130, 0.56, (24, 24)),
    ("tribal_club", "Weapon_Club_Tribal", 146, "club", 120, 0.56, (17, 17)),
    (
        "rusty_steel_flail",
        "Weapon_Club_Steel_Flail_Rusty",
        147,
        "flail",
        81,
        0.56,
        (19, 19, 16),
    ),
    ("zombie_arm_club", "Weapon_Club_Zombie_Arm", 148, "flail", 16, 0.56, (7, 7, 16)),
    (
        "burnt_zombie_arm_club",
        "Weapon_Club_Zombie_Burnt_Arm",
        149,
        "flail",
        16,
        0.56,
        (7, 7, 16),
    ),
    (
        "burnt_zombie_leg_club",
        "Weapon_Club_Zombie_Burnt_Leg",
        150,
        "flail",
        16,
        0.56,
        (7, 7, 16),
    ),
    (
        "frost_zombie_arm_club",
        "Weapon_Club_Zombie_Frost_Arm",
        151,
        "flail",
        16,
        0.56,
        (7, 7, 16),
    ),
    (
        "frost_zombie_leg_club",
        "Weapon_Club_Zombie_Frost_Leg",
        152,
        "flail",
        16,
        0.56,
        (7, 7, 16),
    ),
    ("zombie_leg_club", "Weapon_Club_Zombie_Leg", 153, "flail", 16, 0.56, (7, 7, 16)),
    (
        "sand_zombie_arm_club",
        "Weapon_Club_Zombie_Sand_Arm",
        154,
        "flail",
        16,
        0.56,
        (7, 7, 16),
    ),
    (
        "sand_zombie_leg_club",
        "Weapon_Club_Zombie_Sand_Leg",
        155,
        "flail",
        16,
        0.56,
        (7, 7, 16),
    ),
    (
        "adamantite_longsword",
        "Weapon_Longsword_Adamantite",
        156,
        "longsword",
        150,
        0.52,
        (48, 48, 48, 144),
    ),
    (
        "adamantite_saurian_longsword",
        "Weapon_Longsword_Adamantite_Saurian",
        157,
        "longsword",
        100,
        0.52,
        (31, 31, 31, 39),
    ),
    (
        "cobalt_longsword",
        "Weapon_Longsword_Cobalt",
        158,
        "longsword",
        140,
        0.52,
        (33, 33, 33, 99),
    ),
    (
        "copper_longsword",
        "Weapon_Longsword_Copper",
        159,
        "longsword",
        90,
        0.52,
        (11, 11, 11, 33),
    ),
    (
        "crude_longsword",
        "Weapon_Longsword_Crude",
        160,
        "longsword",
        60,
        0.52,
        (8, 8, 8, 24),
    ),
    (
        "iron_longsword",
        "Weapon_Longsword_Iron",
        161,
        "longsword",
        120,
        0.52,
        (16, 16, 16, 48),
    ),
    (
        "katana_longsword",
        "Weapon_Longsword_Katana",
        162,
        "longsword",
        125,
        0.52,
        (31, 31, 31, 39),
    ),
    (
        "mithril_longsword",
        "Weapon_Longsword_Mithril",
        163,
        "longsword",
        160,
        0.52,
        (70, 70, 70, 210),
    ),
    (
        "onyxium_longsword",
        "Weapon_Longsword_Onyxium",
        164,
        "longsword",
        170,
        0.52,
        (102, 102, 102, 306),
    ),
    (
        "praetorian_longsword",
        "Weapon_Longsword_Praetorian",
        165,
        "longsword",
        200,
        0.52,
        (195, 195, 195, 307),
    ),
    (
        "praetorian_npc_longsword",
        "Weapon_Longsword_Praetorian_NPC",
        166,
        "longsword",
        200,
        0.52,
        (195, 195, 195, 307),
    ),
    (
        "scarab_longsword",
        "Weapon_Longsword_Scarab",
        167,
        "longsword",
        100,
        0.52,
        (31, 31, 31, 39),
    ),
    (
        "spectral_longsword",
        "Weapon_Longsword_Spectral",
        168,
        "longsword",
        100,
        0.52,
        (31, 31, 31, 39),
    ),
    (
        "stone_trork_longsword",
        "Weapon_Longsword_Stone_Trork",
        169,
        "longsword",
        40,
        0.52,
        (9, 9, 9, 27),
    ),
    (
        "thorium_longsword",
        "Weapon_Longsword_Thorium",
        170,
        "longsword",
        130,
        0.52,
        (23, 23, 23, 69),
    ),
    (
        "tribal_longsword",
        "Weapon_Longsword_Tribal",
        171,
        "longsword",
        125,
        0.52,
        (31, 31, 31, 39),
    ),
    (
        "void_longsword",
        "Weapon_Longsword_Void",
        172,
        "longsword",
        158,
        0.42,
        (47, 47, 47, 141),
    ),
    (
        "flame_longsword",
        "Weapon_Longsword_Flame",
        173,
        "longsword",
        0,
        0.52,
        (31, 31, 31, 39),
        "fire",
    ),
)

LEGACY_MELEE_VARIANTS = tuple(LegacyMeleeVariant(*row) for row in _ROWS)
LEGACY_MELEE_PROFILE_NAMES = tuple(row.profile for row in LEGACY_MELEE_VARIANTS)

assert len(LEGACY_MELEE_VARIANTS) == 53
assert len(set(LEGACY_MELEE_PROFILE_NAMES)) == len(LEGACY_MELEE_VARIANTS)
assert len({row.asset_id for row in LEGACY_MELEE_VARIANTS}) == len(
    LEGACY_MELEE_VARIANTS
)
assert {row.family_id for row in LEGACY_MELEE_VARIANTS} == set(range(121, 174))

_CASTER_ROWS = (
    (
        "halloween_broomstick",
        "Halloween_Broomstick",
        174,
        "staff",
        "corruption_orb",
        50,
    ),
    ("adamantite_staff", "Weapon_Staff_Adamantite", 175, "staff", "corruption_orb", 50),
    ("bamboo_bo_staff", "Weapon_Staff_Bo_Bamboo", 176, "staff", "corruption_orb", 50),
    ("wood_bo_staff", "Weapon_Staff_Bo_Wood", 177, "staff", "corruption_orb", 50),
    ("bone_staff", "Weapon_Staff_Bone", 178, "staff", "corruption_orb", 50),
    ("bronze_staff", "Weapon_Staff_Bronze", 179, "staff", "corruption_orb", 50),
    ("cane_staff", "Weapon_Staff_Cane", 180, "staff", "corruption_orb", 50),
    ("cobalt_staff", "Weapon_Staff_Cobalt", 181, "staff", "corruption_orb", 50),
    ("copper_staff", "Weapon_Staff_Copper", 182, "staff", "corruption_orb", 50),
    (
        "crystal_fire_trork_staff",
        "Weapon_Staff_Crystal_Fire_Trork",
        183,
        "staff",
        "fireball",
        50,
    ),
    (
        "crystal_purple_staff",
        "Weapon_Staff_Crystal_Purple",
        184,
        "staff",
        "corruption_orb",
        50,
    ),
    (
        "crystal_red_staff",
        "Weapon_Staff_Crystal_Red",
        185,
        "staff",
        "fireball",
        0,
    ),
    ("doomed_staff", "Weapon_Staff_Doomed", 186, "staff", "corruption_orb", 50),
    ("frost_staff", "Weapon_Staff_Frost", 187, "staff", "ice_ball", 50),
    ("iron_staff", "Weapon_Staff_Iron", 188, "staff", "corruption_orb", 50),
    ("mithril_staff", "Weapon_Staff_Mithril", 189, "staff", "corruption_orb", 50),
    ("onion_staff", "Weapon_Staff_Onion", 190, "staff", "corruption_orb", 50),
    ("onyxium_staff", "Weapon_Staff_Onyxium", 191, "staff", "corruption_orb", 50),
    ("thorium_staff", "Weapon_Staff_Thorium", 192, "staff", "corruption_orb", 50),
    ("wizard_staff", "Weapon_Staff_Wizard", 193, "staff", "corruption_orb", 50),
    ("wood_staff", "Weapon_Staff_Wood", 194, "staff", "corruption_orb", 50),
    (
        "kweebec_wood_staff",
        "Weapon_Staff_Wood_Kweebec",
        195,
        "staff",
        "corruption_orb",
        50,
    ),
    (
        "rotten_wood_staff",
        "Weapon_Staff_Wood_Rotten",
        196,
        "staff",
        "corruption_orb",
        50,
    ),
    (
        "demon_spellbook",
        "Weapon_Spellbook_Demon",
        197,
        "spellbook",
        "corruption_orb",
        100,
    ),
    (
        "fire_spellbook",
        "Weapon_Spellbook_Fire",
        198,
        "spellbook",
        "corruption_orb",
        100,
    ),
    (
        "frost_spellbook",
        "Weapon_Spellbook_Frost",
        199,
        "spellbook",
        "corruption_orb",
        100,
    ),
    (
        "brown_grimoire",
        "Weapon_Spellbook_Grimoire_Brown",
        200,
        "spellbook",
        "corruption_orb",
        100,
    ),
    (
        "purple_grimoire",
        "Weapon_Spellbook_Grimoire_Purple",
        201,
        "spellbook",
        "corruption_orb",
        100,
    ),
    ("tribal_wand", "Weapon_Wand_Tribal", 202, "wand", "corruption_orb", 25),
    ("wood_wand", "Weapon_Wand_Wood", 203, "wand", "corruption_orb", 25),
    ("rotten_wood_wand", "Weapon_Wand_Wood_Rotten", 204, "wand", "corruption_orb", 25),
)

def _legacy_caster_variant(
    profile: str,
    asset_id: str,
    family_id: int,
    archetype: str,
    projectile: str,
    mana_cost: float,
) -> LegacyCasterVariant:
    return LegacyCasterVariant(
        profile=profile,
        asset_id=asset_id,
        family_id=family_id,
        archetype=archetype,
        projectile=projectile,
        scalars=WeaponScalarSet.from_mapping({"mana_cost": mana_cost}),
    )


LEGACY_CASTER_VARIANTS = tuple(_legacy_caster_variant(*row) for row in _CASTER_ROWS)
LEGACY_CASTER_PROFILE_NAMES = tuple(row.profile for row in LEGACY_CASTER_VARIANTS)

assert len(LEGACY_CASTER_VARIANTS) == 31
assert len(set(LEGACY_CASTER_PROFILE_NAMES)) == len(LEGACY_CASTER_VARIANTS)
assert len({row.asset_id for row in LEGACY_CASTER_VARIANTS}) == len(
    LEGACY_CASTER_VARIANTS
)
assert len({row.family_id for row in LEGACY_CASTER_VARIANTS}) == len(
    LEGACY_CASTER_VARIANTS
)
assert {row.family_id for row in LEGACY_CASTER_VARIANTS} == set(range(174, 205))
