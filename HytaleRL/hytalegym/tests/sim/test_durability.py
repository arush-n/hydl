"""Weapon durability construct: 0.7.2 table values through the catalog loader; gen 5 keeps the original loader (row 8d)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from hytalegym.jax.combat.arsenal import item_programs
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import durability
from hytalegym.sim.gen7.profile import PROFILE as GEN7

ORIGINAL = item_programs.hytale_0_5_7_item_program_catalog

#: (0.5.7 shipped catalog, 0.7.2 table) as (max_durability, durability_loss_on_hit): sim/gen7/data/durability.json.
NAMED = {
    "Weapon_Sword_Iron": ((120.0, 0.21), (250.0, 0.13)),
    "Weapon_Mace_Iron": ((120.0, 0.5), (250.0, 0.3)),
    "Weapon_Daggers_Iron": ((120.0, 0.1), (250.0, 0.06)),
}


def _by_id(catalog) -> dict:
    return {item["item_asset_id"]: item for item in catalog.document["weapon_items"]}


def _values(item) -> tuple[float, float]:
    return (item["max_durability"], item["durability_loss_on_hit"])


def test_gen7_construct_serves_the_072_maximum_and_loss_for_named_weapons() -> None:
    profile = replace(GEN7, constructs=SimpleNamespace(weapon_durability="gen7_table"))
    served = _by_id(durability.construct(ORIGINAL, profile)())
    for name, (_, table) in NAMED.items():
        assert _values(served[name]) == table, name
    shipped = _by_id(ORIGINAL())  # the cached shipped catalog is not rewritten by the construct
    for name, (old, _) in NAMED.items():
        assert _values(shipped[name]) == old, name


def test_gen5_construct_is_the_original_loader() -> None:
    profile = replace(GEN5, constructs=SimpleNamespace(weapon_durability="shipped_json"))
    assert durability.construct(ORIGINAL, profile) is ORIGINAL


def test_npc_wielder_serves_the_072_maximum_and_zero_loss() -> None:
    # Live 0.7.2: an NPC-held weapon is never worn (no Player component; receipt durability_loss.json, 250.0 every tick).
    profile = replace(GEN7, constructs=SimpleNamespace(weapon_durability="gen7_npc_wielder"))
    served = _by_id(durability.construct(ORIGINAL, profile)())
    assert _values(served["Weapon_Sword_Iron"]) == (250.0, 0.0)
    for item_id, item in served.items():
        assert item["durability_loss_on_hit"] == 0.0, item_id
    table = _by_id(durability.construct(ORIGINAL, replace(GEN7, constructs=SimpleNamespace(
        weapon_durability="gen7_table")))())
    for item_id, item in served.items():
        assert item["max_durability"] == table[item_id]["max_durability"], item_id


def test_the_shipped_gen7_profile_uses_npc_wielder() -> None:
    assert GEN7.constructs.weapon_durability == "gen7_npc_wielder"
