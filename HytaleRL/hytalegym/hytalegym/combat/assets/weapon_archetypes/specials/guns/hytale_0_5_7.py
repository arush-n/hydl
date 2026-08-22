"""Pinned Hytale 0.5.7 legacy Gun and Blunderbuss rows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GunSpec:
    """A ballistic item with its resolved launch and gating behaviour."""

    profile: str
    asset_id: str
    family_id: int
    ability_asset_id: str
    projectile_id: str
    speed: float
    damage: float
    charge_seconds: float
    launch_runtime_seconds: float
    cooldown_seconds: float
    mana_minimum: float
    mana_cost: float
    consumes_crude_arrow: bool


GUN_SPECS = (
    GunSpec(
        "gun",
        "Weapon_Gun",
        219,
        "Gun_Shoot",
        "GunPvP_Assault_Rifle_Bullet",
        300.0,
        200.0,
        0.0,
        0.05,
        0.0,
        0.0,
        0.0,
        False,
    ),
    GunSpec(
        "blunderbuss",
        "Weapon_Gun_Blunderbuss",
        220,
        "Gun_Shoot_Flintlock_Charging",
        "Gun_Blunderbuss_Bullet",
        100.0,
        200.0,
        2.0,
        0.25,
        0.0,
        50.0,
        0.0,
        False,
    ),
    GunSpec(
        "rusty_blunderbuss",
        "Weapon_Gun_Blunderbuss_Rusty",
        221,
        "Gun_Shoot_Flintlock_Charging",
        "Gun_Blunderbuss_Bullet",
        100.0,
        200.0,
        2.0,
        0.25,
        0.0,
        50.0,
        50.0,
        False,
    ),
    GunSpec(
        "assault_rifle",
        "Weapon_Assault_Rifle",
        222,
        "Weapon_Assault_Rifle_Primary",
        "GunPvP_Assault_Rifle_Bullet",
        300.0,
        200.0,
        0.0,
        0.0,
        0.05,
        0.0,
        0.0,
        True,
    ),
    GunSpec(
        "handgun",
        "Weapon_Handgun",
        223,
        "Weapon_Handgun_Primary",
        "GunPvP_Handgun_Bullet",
        500.0,
        60.0,
        0.0,
        0.0,
        0.2,
        0.0,
        0.0,
        True,
    ),
)
GUN_PROFILE_NAMES = tuple(row.profile for row in GUN_SPECS)

assert {row.family_id for row in GUN_SPECS} == set(range(219, 224))
assert len(set(GUN_PROFILE_NAMES)) == len(GUN_SPECS)


__all__ = ["GUN_PROFILE_NAMES", "GUN_SPECS", "GunSpec"]
