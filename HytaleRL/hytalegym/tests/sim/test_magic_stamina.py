"""Magic casts stamina (gen 7, row 7c, stamina part): each authored 0.7.2 stamina cost is charged once.

The staff cast (``Staff_Cast_Summon_Charged``, default ``Staff_Cast_Summon_StaminaCost`` -5 at 0.167) is one application on
gen 7. Crystal_Ice is not applied (its authored chain needs Ingredient_Ice_Essence, which the sim does not model) and keeps
its gen-5 rows. Gen 5 keeps the repo source (native NPC x2). Wands, spellbooks, Crystal_Flame and the zero-stamina rows keep
their rows.
"""

from __future__ import annotations

import json
import os
import zipfile
from importlib import import_module

import jax.numpy as jnp
import pytest

from hytalegym.jax.combat.arsenal.programs.ability import effective_resource_cost
from hytalegym.jax.combat.arsenal.schema import contract
from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.gen7.mechanics import magic_costs
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7_OVERLAY

from .support import mount_fresh, unmount

STAMINA = 0  # RESOURCE_STAMINA index (checked against the rendered profile in test_stamina_index_is_resource_stamina)
ZIP = os.environ.get("HYTALE_ASSETS_ZIP")


@pytest.fixture(scope="module")
def render(tmp_path_factory):
    target = tmp_path_factory.mktemp("magic_stamina_render") / "gen7"
    R.render(GEN7_OVERLAY, target, source=HeadSource())
    package = mount_fresh(target).__name__
    yield package
    unmount(package)


def _row(loadout, ability: int, index: int = STAMINA):
    return (
        float(loadout.ability_resource_cost[0, 0, ability, index]),
        int(loadout.ability_resource_cost_kind[0, 0, ability, index]),
        float(loadout.ability_resource_commit_time_seconds[0, 0, ability, index]),
        float(loadout.ability_resource_minimum[0, 0, ability, index]),
    )


def _cast_index(prof, name: str) -> int:
    return len(prof._CATALOG[name]()["abilities"]) - 1


def _asset_index(prof, name: str, asset: str) -> int:
    return [a.get("asset") for a in prof._CATALOG[name]()["abilities"]].index(asset)


def test_stamina_index_is_resource_stamina(render: str) -> None:
    prof = import_module(render + ".combat.arsenal.profiles.hytale_0_5_7")
    assert int(prof.RESOURCE_STAMINA) == STAMINA


@pytest.mark.parametrize(
    ("weapon", "cost", "gate"),
    [("iron_staff", 5.0, 0.0), ("crystal_red_staff", 5.0, 0.0), ("wood_staff", 5.0, 0.0)],
)
def test_gen7_staff_cast_charges_its_stamina_once(render: str, weapon: str, cost: float, gate: float) -> None:
    prof = import_module(render + ".combat.arsenal.profiles.hytale_0_5_7")
    loadout = prof.hytale_0_5_7_entity_loadouts([[weapon, "iron_sword"]])
    amount, kind, commit, minimum = _row(loadout, _cast_index(prof, weapon))
    assert amount == pytest.approx(cost)
    assert kind == contract.RESOURCE_COST_SINGLE_APPLICATION
    assert commit == pytest.approx(magic_costs.COMMIT_SECONDS)
    assert minimum == pytest.approx(gate)
    assert effective_resource_cost(jnp.float32(amount), jnp.int32(kind)) == pytest.approx(cost)


def test_gen7_ice_staff_stamina_is_not_applied(render: str) -> None:
    # Not applied: the authored ice chain needs Ingredient_Ice_Essence (ModifyInventory), which the sim does not model;
    # decided separately. Ice keeps its current sim rows on gen 7 (the same native rows as gen 5).
    prof = import_module(render + ".combat.arsenal.profiles.hytale_0_5_7")
    loadout = prof.hytale_0_5_7_entity_loadouts([["ice_staff", "iron_sword"]])
    cast = _row(loadout, _cast_index(prof, "ice_staff"))
    bolt = _row(loadout, _asset_index(prof, "ice_staff", "Ice_Staff_Primary_Entry"))
    assert cast == pytest.approx((5.0, contract.RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT, 1.167, 0.1))
    assert bolt == pytest.approx((1.5, contract.RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT, 0.0, 1.51))


@pytest.mark.parametrize("weapon", ["wood_wand", "fire_spellbook", "flame_staff", "stoneskin_wand"])
def test_gen7_rows_without_authored_stamina_keep_none(render: str, weapon: str) -> None:
    prof = import_module(render + ".combat.arsenal.profiles.hytale_0_5_7")
    loadout = prof.hytale_0_5_7_entity_loadouts([[weapon, "iron_sword"]])
    amount, kind, _, _ = _row(loadout, _cast_index(prof, weapon))
    assert (amount, kind) == (0.0, contract.RESOURCE_COST_NONE)


def test_gen5_keeps_the_native_stamina_rows() -> None:
    prof = import_module("hytalegym.jax.combat.arsenal.profiles.hytale_0_5_7")
    loadout = prof.hytale_0_5_7_entity_loadouts([["iron_staff", "ice_staff"]])
    amount, kind, _, minimum = _row(loadout, _cast_index(prof, "iron_staff"))
    assert (amount, kind, minimum) == (5.0, contract.RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT, 0.0)
    assert effective_resource_cost(jnp.float32(amount), jnp.int32(kind)) == pytest.approx(10.0)
    ice = prof.hytale_0_5_7_entity_loadouts([["ice_staff", "iron_sword"]])
    cast = _row(ice, _cast_index(prof, "ice_staff"))
    bolt = _row(ice, _asset_index(prof, "ice_staff", "Ice_Staff_Primary_Entry"))
    assert cast == pytest.approx((5.0, contract.RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT, 1.167, 0.1))
    assert bolt[1] == contract.RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT


@pytest.mark.skipif(not ZIP, reason="HYTALE_ASSETS_ZIP not set")
def test_stamina_table_matches_the_0_7_2_assets() -> None:
    with zipfile.ZipFile(ZIP) as archive:
        names = archive.namelist()

        def text(path: str) -> str:
            return archive.read(path).decode("utf-8", "replace")

        charged = text("Server/Item/Interactions/Weapons/Staff/Attacks/Staff_Cast_Summon_Charged.json")
        assert '"Var": "Staff_Cast_Summon_StaminaCost"' in charged
        assert json.loads(charged)["Next"]["Interactions"][0]["Interactions"][0]["DefaultValue"]["Interactions"][0][
            "StatModifiers"
        ]["Stamina"] == -5
        ice_item = json.loads(text("Server/Item/Items/Weapon/Staff/Weapon_Staff_Crystal_Ice.json"))["InteractionVars"]
        assert ice_item["Staff_Cast_Summon_StaminaCost"]["Interactions"][0] == {"Type": "Simple", "RunTime": 0}
        launch = json.dumps(ice_item["Staff_Cast_Summon_Launch"])
        assert '"Stamina": -5' in launch
        bolt = json.loads(text("Server/Item/Interactions/Weapons/Staff/Ice/Ice_Staff_Primary_Entry.json"))
        assert bolt["Costs"] == {"Stamina": 1.51}
        wrapper = json.loads(text("Server/Item/Interactions/Weapons/Staff/Ice/Ice_Staff_Primary_Wrapper.json"))
        assert wrapper["Costs"] == {"Stamina": 0.1}
        assert wrapper["Next"] == "Staff_Primary"
        ice_stamina_stat = bolt["Next"]["Next"]["Interactions"][1]["Interactions"][0]["StatModifiers"]["Stamina"]
        assert ice_stamina_stat == -1.5
        for weapon in ("Weapon_Wand_Wood", "Weapon_Spellbook_Fire", "Weapon_Staff_Crystal_Flame"):
            (path,) = [n for n in names if n.startswith("Server/Item/Items/") and n.endswith("/" + weapon + ".json")]
            assert "Stamina" not in text(path), weapon
