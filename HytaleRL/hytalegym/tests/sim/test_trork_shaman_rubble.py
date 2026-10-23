"""Trork_Shaman's rubble throw is a gen-7 arsenal profile (``trork_shaman_rubble``), absent on gen 5.

The values are the 0.7.2 assets: the NPC root ``NPC_Rubble_Throw`` (Simple wind-up RunTime 0.2 s, then LaunchProjectile
at 0.2 s with RunTime 0.217 s, so the root runs 0.417 s) and the stone ``NPC_Rubble_Stone`` (Damage 16, MuzzleVelocity 30,
TerminalVelocity 30, Gravity 20, TimeToLive 5, Bounciness 0.5, DeadTime 0; hitbox half-extent 0.15 from
``Server/Models/Projectiles/Items/Rubble/Rubble_Stone.json``). The kind ``PROJECTILE_ROCK`` (15) is a gen-7 overlay
constant. The duel behaviour is measured in the lane report, not here.
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
    ("def _trork_shaman_rubble() -> dict:", 1),
    ('    "trork_shaman_rubble": _trork_shaman_rubble,', 1),
    ('    + ("trork_shaman_rubble",)', 1),
    ("    PROJECTILE_ROCK: _ProjectileLaunchTransform((0.1, 0.0, -1.0), 0.0, 0.0, True, False),", 1),
)
KIND_PINS = (
    ("combat/arsenal/schema/contract.py", "PROJECTILE_ROCK = 15\n"),
    ("combat/arsenal/schema/validation.py", "(i32[..., EI_PROJECTILE_KIND] > PROJECTILE_ROCK)"),
    ("combat/arsenal/schema/spec.py", '"rock": PROJECTILE_ROCK,'),
)


@pytest.fixture(scope="module")
def gen7_render(tmp_path_factory):
    out = tmp_path_factory.mktemp("trork_shaman_rubble") / "gen7"
    R.render(GEN7, out, source=HeadSource())
    return out


@pytest.fixture(scope="module")
def gen5_render(tmp_path_factory):
    out = tmp_path_factory.mktemp("trork_shaman_rubble") / "gen5"
    R.render(GEN5, out, source=HeadSource())
    return out


def test_gen7_source_pins_the_builder_the_registry_the_name_and_the_kind(gen7_render) -> None:
    text = (gen7_render / PROFILE).read_text()
    for line, count in PINS:
        assert text.count(line) == count, line
    for path, line in KIND_PINS:
        assert line in (gen7_render / path).read_text(), (path, line)


def test_gen7_builder_carries_the_asset_values(gen7_render) -> None:
    package = mount_fresh(gen7_render).__name__
    try:
        profile = import_module(package + ".combat.arsenal.profiles.hytale_0_5_7")
        contract = import_module(package + ".combat.arsenal.schema.contract")
        built = profile._trork_shaman_rubble()
        assert built["asset"] == "NPC_Rubble_Throw"
        assert len(built["abilities"]) == 1
        ability = built["abilities"][0]
        assert ability["asset"] == "NPC_Rubble_Throw"
        assert ability["duration"] == pytest.approx(0.417)  # 0.2 wind-up + 0.217 LaunchProjectile RunTime
        (event,) = ability["events"]
        assert event["kind"] == contract.EVENT_PROJECTILE
        assert event["time"] == pytest.approx(0.2)
        f, i = event["f"], event["i"]
        assert f[contract.EF_DAMAGE] == 16
        assert f[contract.EF_PROJECTILE_SPEED] == 30
        assert f[contract.EF_PROJECTILE_GRAVITY] == 20
        assert f[contract.EF_PROJECTILE_TERMINAL_VELOCITY] == 30
        assert f[contract.EF_PROJECTILE_LIFETIME_SECONDS] == 5
        assert f[contract.EF_PROJECTILE_BOUNCINESS] == 0.5
        assert f[contract.EF_PROJECTILE_HALF_EXTENT] == pytest.approx(0.15)
        assert f[contract.EF_PROJECTILE_DEAD_TIME_SECONDS] == 0.0
        assert i[contract.EI_PROJECTILE_KIND] == contract.PROJECTILE_ROCK == 15
        assert i[contract.EI_DAMAGE_CAUSE] == profile.DAMAGE_PROJECTILE
    finally:
        unmount(package)


def test_gen7_registry_and_profile_name_resolve(gen7_render) -> None:
    package = mount_fresh(gen7_render).__name__
    try:
        profile = import_module(package + ".combat.arsenal.profiles.hytale_0_5_7")
        # The turret's bow (test_skeleton_archer_turret_bow.py) is appended after this one, so the Shaman is second to last.
        assert profile.PROFILE_NAMES[-2] == "trork_shaman_rubble"
        assert profile._profile_index("trork_shaman_rubble") == len(profile.PROFILE_NAMES) - 1
        assert profile._CATALOG["trork_shaman_rubble"]()["asset"] == "NPC_Rubble_Throw"
    finally:
        unmount(package)


def test_gen5_has_no_trork_shaman_profile(gen5_render) -> None:
    text = (gen5_render / PROFILE).read_text()
    assert "trork_shaman_rubble" not in text
    assert "_trork_shaman_rubble" not in text
    assert "PROJECTILE_ROCK" not in text
    assert "PROJECTILE_ROCK" not in (gen5_render / "combat/arsenal/schema/contract.py").read_text()
    package = mount_fresh(gen5_render).__name__
    try:
        profile = import_module(package + ".combat.arsenal.profiles.hytale_0_5_7")
        assert "trork_shaman_rubble" not in profile.PROFILE_NAMES
        assert "trork_shaman_rubble" not in profile._CATALOG
    finally:
        unmount(package)
