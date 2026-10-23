"""Magic casts mana (gen 7, row 7c): the 0.7.2 cost per magic weapon, one charge per cast, and refusal for wands and spellbooks.

Gen 5 keeps the repo source (two native NPC applications, no refusal). The table is checked against ``Assets.zip`` when
``HYTALE_ASSETS_ZIP`` is set (the item JSON and its interaction override), and the rendered gen-7 loadout is read per weapon.
"""

from __future__ import annotations

import json
import os
import zipfile
from importlib import import_module

import pytest

from hytalegym.combat.assets.weapon_archetypes import LEGACY_CASTER_PROFILE_NAMES
from hytalegym.jax.combat.arsenal.schema import contract
from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import magic_costs
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7_OVERLAY

from .support import mount_fresh, unmount

NAMED = ("flame_staff", "ice_staff", "stoneskin_wand", "root_wand", "skeleton_mage_spellbook")
ZIP = os.environ.get("HYTALE_ASSETS_ZIP")
MANA = 1  # RESOURCE_MANA (jax/combat/mechanics/schema/contract.py), as in test_mana_pool.py


def test_table_covers_the_36_magic_profiles() -> None:
    assert len(LEGACY_CASTER_PROFILE_NAMES) == 31
    assert set(magic_costs.ROWS) == set(LEGACY_CASTER_PROFILE_NAMES) | set(NAMED)
    assert len(magic_costs.ROWS) == 36


def test_gen5_returns_the_original_loadout_builder() -> None:
    def original(entity_profiles):
        return entity_profiles

    assert magic_costs.construct_entities(original, GEN5) is original


@pytest.mark.parametrize(
    ("profile", "mana", "gate"),
    [
        ("iron_staff", 50.0, False),
        ("crystal_red_staff", 0.0, False),
        ("wood_wand", 25.0, True),
        ("fire_spellbook", 100.0, True),
        ("purple_grimoire", 100.0, True),
        ("skeleton_mage_spellbook", 0.0, False),
        ("ice_staff", 0.0, False),
    ],
)
def test_table_values_per_named_weapon(profile: str, mana: float, gate: bool) -> None:
    row = magic_costs.ROWS[profile]
    assert (row.mana, row.gate) == (mana, gate)


@pytest.mark.skipif(not ZIP, reason="HYTALE_ASSETS_ZIP not set")
def test_table_matches_the_0_7_2_assets() -> None:
    with zipfile.ZipFile(ZIP) as archive:
        names = archive.namelist()

        def item(asset: str) -> dict:
            (path,) = [n for n in names if n.startswith("Server/Item/Items/") and n.endswith("/" + asset + ".json")]
            return json.loads(archive.read(path).decode("utf-8", "replace"))

        for profile, row in magic_costs.ROWS.items():
            if profile == "skeleton_mage_spellbook":
                # the NPC root the profile compiles: no mana anywhere in the interaction file
                npc = "Server/Item/Interactions/NPCs/Undead/Skeleton_Sand_Mage/Skeleton_Sand_Mage_Spellbook_Corruption_Orb.json"
                assert npc in names
                assert "Mana" not in archive.read(npc).decode("utf-8", "replace"), profile
                assert row.mana == 0.0 and row.gate is False
                continue
            data = item(row.item)
            interaction_vars = data.get("InteractionVars", {})
            if row.item.startswith("Weapon_Staff_Crystal_Flame") or row.item in ("Weapon_Wand_Stoneskin", "Weapon_Wand_Root"):
                assert '"Mana"' not in json.dumps(data), profile
                continue
            if row.mana == 0.0 and row.item == "Weapon_Staff_Crystal_Ice":
                assert interaction_vars["Staff_Cast_Summon_Cost"]["Interactions"][0]["StatModifiers"]["Mana"] == 0
                continue
            if "Weapon_Staff" in row.item or row.item == "Halloween_Broomstick":
                override = interaction_vars["Staff_Cast_Summon_Cost"]["Interactions"][0]["StatModifiers"]["Mana"]
                assert -override == row.mana, profile
            elif "Wand" in row.item:
                assert interaction_vars["Wand_Cast_Left_Charged"]["Interactions"][0]["Costs"]["Mana"] == row.mana, profile
            else:
                assert interaction_vars["Spellbook_Cast_Hurl_Charged"]["Interactions"][0]["Costs"]["Mana"] == row.mana, profile


@pytest.fixture(scope="module")
def render(tmp_path_factory):
    target = tmp_path_factory.mktemp("magic_costs_render") / "gen7"
    R.render(GEN7_OVERLAY, target, source=HeadSource())
    package = mount_fresh(target).__name__
    yield package
    unmount(package)


def _cast_row(loadout, weapon_column: int, ability: int):
    return (
        float(loadout.ability_resource_cost[0, weapon_column, ability, MANA]),
        int(loadout.ability_resource_cost_kind[0, weapon_column, ability, MANA]),
        float(loadout.ability_resource_minimum[0, weapon_column, ability, MANA]),
    )


@pytest.mark.parametrize(
    ("weapon", "mana", "gate"),
    [("iron_staff", 50.0, False), ("wood_wand", 25.0, True), ("fire_spellbook", 100.0, True),
     ("purple_grimoire", 100.0, True), ("ice_staff", 0.0, False)],
)
def test_gen7_loadout_charges_one_0_7_2_cost_per_cast(render: str, weapon: str, mana: float, gate: bool) -> None:
    prof = import_module(render + ".combat.arsenal.profiles.hytale_0_5_7")
    loadout = prof.hytale_0_5_7_entity_loadouts([[weapon, "iron_sword"]])
    cast = len(prof._CATALOG[weapon]()["abilities"]) - 1
    cost, kind, minimum = _cast_row(loadout, 0, cast)
    assert cost == pytest.approx(mana)
    # a zero-mana item keeps the gen-5 row (no cost, no kind); a charged one is one application
    assert kind == (contract.RESOURCE_COST_SINGLE_APPLICATION if mana else contract.RESOURCE_COST_NONE)
    assert minimum == pytest.approx(mana if gate else 0.0)


def test_gen7_skeleton_npc_root_has_no_cost_and_no_gate(render: str) -> None:
    prof = import_module(render + ".combat.arsenal.profiles.hytale_0_5_7")
    loadout = prof.hytale_0_5_7_entity_loadouts([["skeleton_mage_spellbook", "iron_sword"]])
    assert _cast_row(loadout, 0, 0) == (0.0, contract.RESOURCE_COST_NONE, 0.0)


def test_gen5_repo_loadout_keeps_the_native_npc_cost() -> None:
    prof = import_module("hytalegym.jax.combat.arsenal.profiles.hytale_0_5_7")
    loadout = prof.hytale_0_5_7_entity_loadouts([["iron_staff", "iron_sword"]])
    cast = len(prof._CATALOG["iron_staff"]()["abilities"]) - 1
    cost, kind, minimum = _cast_row(loadout, 0, cast)
    assert cost == pytest.approx(50.0)
    assert kind == contract.RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT
    assert minimum == 0.0
