"""Shortbow scalar-variant damage: the adapter reads the 12 variant tuples from both content tables.

The 0.5.7 tuples equal the literals of the shared weapon_variants module (the render keeps them). The 0.7.2 tuples
equal the table's Primary_Shoot_Damage_Strength_0..3 (timing_audit differences.md, tier 2 damage) and the unchanged
Strength_4 and Signature_Volley damage. The render wiring is a separate hook (not yet registered).
"""

from __future__ import annotations

import dataclasses

import pytest

from hytalegym.jax.combat.arsenal.profiles.variants import SCALAR_VARIANTS
from hytalegym.sim.common.content import load
from hytalegym.sim.common.content.adapters import scalar_variant_values
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import shortbow_variant_damage
from hytalegym.sim.gen7.profile import PROFILE as GEN7

SHORTBOW_ROWS = tuple(row for row in SCALAR_VARIANTS if row.template == "shortbow")

#: 0.7.2 tuples (Strength_0..4, Signature_Volley), read from the gen-7 item rows.
EXPECTED_072 = {
    "Weapon_Shortbow_Adamantite": (1.0, 3.0, 6.0, 10.0, 29.0, 22.0),
    "Weapon_Shortbow_Bronze": (1.0, 2.0, 3.0, 6.0, 17.0, 12.0),
    "Weapon_Shortbow_Cobalt": (1.0, 2.0, 5.0, 8.0, 23.0, 17.0),
    "Weapon_Shortbow_Copper": (1.0, 2.0, 3.0, 5.0, 15.0, 11.0),
    "Weapon_Shortbow_Crude": (1.0, 1.0, 2.0, 4.0, 12.0, 9.0),
    "Weapon_Shortbow_Doomed": (1.0, 2.0, 5.0, 8.0, 23.0, 17.0),
    "Weapon_Shortbow_Flame": (1.0, 2.0, 3.0, 5.0, 15.0, 25.0),
    "Weapon_Shortbow_Frost": (1.0, 2.0, 3.0, 5.0, 15.0, 25.0),
    "Weapon_Shortbow_Iron_Rusty": (1.0, 2.0, 3.0, 6.0, 17.0, 12.0),
    "Weapon_Shortbow_Mithril": (2.0, 4.0, 7.0, 13.0, 37.0, 27.0),
    "Weapon_Shortbow_Onyxium": (2.0, 4.0, 7.0, 13.0, 37.0, 27.0),
    "Weapon_Shortbow_Thorium": (1.0, 2.0, 5.0, 8.0, 23.0, 17.0),
}


def test_there_are_twelve_non_iron_shortbow_variants() -> None:
    assert {row.asset_id for row in SHORTBOW_ROWS} == set(EXPECTED_072)


def test_gen5_adapter_reads_the_shared_0_5_7_tuples() -> None:
    tables = load(GEN5)
    for row in SHORTBOW_ROWS:
        assert scalar_variant_values(tables, row.asset_id, "shortbow")["damage"] == row.damage, row.asset_id


def test_gen7_adapter_reads_the_0_7_2_tuples() -> None:
    tables = load(GEN7)
    for row in SHORTBOW_ROWS:
        served = scalar_variant_values(tables, row.asset_id, "shortbow")["damage"]
        assert served == EXPECTED_072[row.asset_id], row.asset_id



def _recorder():
    seen = []

    def original(spec):
        seen.append(spec)
        return spec

    return seen, original


def test_gen5_returns_the_original_function() -> None:
    _, original = _recorder()
    assert shortbow_variant_damage.construct(original, GEN5) is original


def test_gen7_wrapper_serves_the_0_7_2_tuple_for_every_variant() -> None:
    for row in SHORTBOW_ROWS:
        seen, original = _recorder()
        shortbow_variant_damage.construct(original, GEN7)(row)
        served = seen[-1]
        assert served.damage == EXPECTED_072[row.asset_id], row.asset_id
        assert served.max_durability == row.max_durability, row.asset_id
        assert served.guard_stamina_value == row.guard_stamina_value, row.asset_id


def test_gen7_passes_non_shortbow_and_iron_rows_through() -> None:
    seen, original = _recorder()
    wrapped = shortbow_variant_damage.construct(original, GEN7)
    crossbow = next(row for row in SCALAR_VARIANTS if row.template == "crossbow")
    wrapped(crossbow)
    assert seen[-1] is crossbow
    iron = dataclasses.replace(SHORTBOW_ROWS[0], profile="iron_shortbow", asset_id="Weapon_Shortbow_Iron",
                               damage=(6.0, 9.0, 12.0, 16.0, 19.0, 14.0))
    wrapped(iron)
    assert seen[-1].damage == (1.0, 2.0, 4.0, 7.0, 19.0, 14.0)


def test_unknown_switch_value_raises() -> None:
    _, original = _recorder()
    bad = dataclasses.replace(GEN7, constructs=dataclasses.replace(GEN7.constructs, shortbow_scalar_variant="bogus"))
    with pytest.raises(ValueError):
        shortbow_variant_damage.construct(original, bad)
