"""Void scythe primary is a charging root at 0.7.2 (Battleaxe_Scythe_Void_Attack: Charging, thresholds [0.0, 1.67],
AllowIndefiniteHold false). The two spin abilities carry hold tables ((0.0, i), (1.67, 2)) on the gen-7 render only.

Pins the rendered profile text and the built ability data on both renders. The behavioural runs (a slot 0 tap and a
90-tick hold against a 1000 HP target) are too slow for the unit suite; they are recorded in mechanics_072.md.
"""

from __future__ import annotations

from importlib import import_module

import pytest

from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.gen5.overlay import OVERLAY as GEN5
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7

from .support import mount_fresh, unmount

SPIN_TABLES = ("hold_table=((0.0, index), (1.67, 2)),", "for index, (ability_asset, selector, force) in enumerate(normal)")


@pytest.fixture(scope="module")
def gen7_render(tmp_path_factory):
    out = tmp_path_factory.mktemp("void_scythe_charge") / "gen7"
    R.render(GEN7, out, source=HeadSource())
    return out


@pytest.fixture(scope="module")
def gen5_render(tmp_path_factory):
    out = tmp_path_factory.mktemp("void_scythe_charge") / "gen5"
    R.render(GEN5, out, source=HeadSource())
    return out


def _built_abilities(out) -> tuple[str, list[dict]]:
    package = mount_fresh(out).__name__
    try:
        profile = import_module(package + ".combat.arsenal.profiles.hytale_0_5_7")
        abilities = profile._void_scythe()["abilities"]
    finally:
        unmount(package)
    return package, abilities


def test_gen7_profile_text_carries_the_spin_hold_tables(gen7_render) -> None:
    text = (gen7_render / "combat/arsenal/profiles/hytale_0_5_7.py").read_text()
    for line in SPIN_TABLES:
        assert text.count(line) == 1, line


def test_gen7_built_spin_abilities_carry_their_own_slot_in_the_table(gen7_render) -> None:
    _, abilities = _built_abilities(gen7_render)
    spins = [a for a in abilities if a["asset"].startswith("Spear_Spin_Swing")]
    assert [a["asset"] for a in spins] == ["Spear_Spin_Swing_Left", "Spear_Spin_Swing_Right"]
    assert spins[0]["hold_table"] == ((0.0, 0), (1.67, 2))
    assert spins[1]["hold_table"] == ((0.0, 1), (1.67, 2))
    charged = next(a for a in abilities if a["asset"] == "Battleaxe_Swing_Left_Charged")
    # The charged swing is the authored root: its own table points back at the two spins (unchanged HEAD data).
    assert charged["hold_table"] == ((0.0, 0), (1.67, 2))
    assert abilities[2]["asset"] == "Battleaxe_Swing_Left_Charged"


def test_gen5_profile_has_no_spin_hold_tables(gen5_render) -> None:
    text = (gen5_render / "combat/arsenal/profiles/hytale_0_5_7.py").read_text()
    for line in SPIN_TABLES:
        assert line not in text
    _, abilities = _built_abilities(gen5_render)
    spins = [a for a in abilities if a["asset"].startswith("Spear_Spin_Swing")]
    assert len(spins) == 2 and all(a["hold_table"] == () for a in spins)
