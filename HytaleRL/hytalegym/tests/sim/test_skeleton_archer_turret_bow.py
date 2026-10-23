"""Skeleton_Archer_Turret's bow is a gen-7 arsenal profile (``skeleton_archer_turret_bow``), absent on gen 5.

The values are the 0.7.2 assets: the variant ``Skeleton_Archer_Turret`` (Modify.Attack Skeleton_Archer_Bow_Shoot, held
Weapon_Shortbow_Iron_Rusty, family FAMILY_SHORTBOW), the attack root ``Skeleton_Archer_Bow_Shoot`` (Simple RunTime 1, then
LaunchProjectile at RunTime 0.2, so the root runs 1.2 s and the arrow leaves at 1.0 s) and the arrow
``Skeleton_Archer_Arrow`` (Arrow_FullCharge with Damage 16; MuzzleVelocity 50, Gravity 10, TimeToLive 20, DeadTime 0.1;
hit box 0.075). The arrow kind is PROJECTILE_ARROW, reused. The duel behaviour is measured in the lane report.
"""

from __future__ import annotations

from importlib import import_module

import pytest

from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.gen5.overlay import OVERLAY as GEN5
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7

from .support import mount_fresh, unmount

PROFILE = "combat/arsenal/profiles/hytale_0_5_7.py"
PINS = (
    ("def _skeleton_archer_turret_bow() -> dict:", 1),
    ('    "skeleton_archer_turret_bow": _skeleton_archer_turret_bow,', 1),
    ('PROFILE_NAMES = (*PROFILE_NAMES, "skeleton_archer_turret_bow")', 1),
)


@pytest.fixture(scope="module")
def gen7_render(tmp_path_factory):
    out = tmp_path_factory.mktemp("skeleton_archer_turret_bow") / "gen7"
    R.render(GEN7, out, source=HeadSource())
    return out


@pytest.fixture(scope="module")
def gen5_render(tmp_path_factory):
    out = tmp_path_factory.mktemp("skeleton_archer_turret_bow") / "gen5"
    R.render(GEN5, out, source=HeadSource())
    return out


def test_gen7_source_pins_the_builder_and_the_registry(gen7_render) -> None:
    text = (gen7_render / PROFILE).read_text()
    for line, count in PINS:
        assert text.count(line) == count, line


def test_gen7_builder_carries_the_asset_values(gen7_render) -> None:
    package = mount_fresh(gen7_render).__name__
    try:
        profile = import_module(package + ".combat.arsenal.profiles.hytale_0_5_7")
        contract = import_module(package + ".combat.arsenal.schema.contract")
        built = profile._skeleton_archer_turret_bow()
        assert built["asset"] == "Skeleton_Archer_Bow_Shoot"
        assert built["family"] == contract.FAMILY_SHORTBOW
        assert len(built["abilities"]) == 1
        ability = built["abilities"][0]
        assert ability["asset"] == "Skeleton_Archer_Bow_Shoot"
        assert ability["duration"] == pytest.approx(1.2)  # 1.0 wind-up + 0.2 LaunchProjectile RunTime
        (event,) = ability["events"]
        assert event["kind"] == contract.EVENT_PROJECTILE
        assert event["time"] == pytest.approx(1.0)
        f, i = event["f"], event["i"]
        assert f[contract.EF_DAMAGE] == 16
        assert f[contract.EF_PROJECTILE_SPEED] == 50
        assert f[contract.EF_PROJECTILE_GRAVITY] == 10
        assert f[contract.EF_PROJECTILE_TERMINAL_VELOCITY] == 50
        assert f[contract.EF_PROJECTILE_LIFETIME_SECONDS] == 20
        assert f[contract.EF_PROJECTILE_HALF_EXTENT] == pytest.approx(0.075)
        assert f[contract.EF_PROJECTILE_DEAD_TIME_SECONDS] == pytest.approx(0.1)
        assert i[contract.EI_PROJECTILE_KIND] == contract.PROJECTILE_ARROW == 1
        assert i[contract.EI_DAMAGE_CAUSE] == profile.DAMAGE_PROJECTILE
    finally:
        unmount(package)


def test_gen7_registry_and_profile_name_resolve(gen7_render) -> None:
    package = mount_fresh(gen7_render).__name__
    try:
        profile = import_module(package + ".combat.arsenal.profiles.hytale_0_5_7")
        assert profile.PROFILE_NAMES[-1] == "skeleton_archer_turret_bow"
        assert profile.SPEAR_PROFILE_NAMES == profile.CORE_PROFILE_NAMES[-16:]
        assert profile._profile_index("skeleton_archer_turret_bow") == len(profile.PROFILE_NAMES)
        assert profile._CATALOG["skeleton_archer_turret_bow"]()["asset"] == "Skeleton_Archer_Bow_Shoot"
    finally:
        unmount(package)


def test_gen5_has_no_turret_bow_profile(gen5_render) -> None:
    text = (gen5_render / PROFILE).read_text()
    assert "skeleton_archer_turret_bow" not in text
    assert "_skeleton_archer_turret_bow" not in text
    package = mount_fresh(gen5_render).__name__
    try:
        profile = import_module(package + ".combat.arsenal.profiles.hytale_0_5_7")
        assert "skeleton_archer_turret_bow" not in profile.PROFILE_NAMES
        assert "skeleton_archer_turret_bow" not in profile._CATALOG
    finally:
        unmount(package)
