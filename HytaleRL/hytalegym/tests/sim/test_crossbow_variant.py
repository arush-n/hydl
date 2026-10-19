"""Crossbow scalar-variant damage: the gen-7 construct serves the 0.7.2 triple; gen 5 keeps the rows."""

from __future__ import annotations

import dataclasses
import types

from hytalegym.jax.combat.arsenal.profiles.variants import SCALAR_VARIANTS
from hytalegym.sim.common.content import load
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import crossbow_variant
from hytalegym.sim.gen7.profile import PROFILE as GEN7

ANCIENT_STEEL = next(row for row in SCALAR_VARIANTS if row.profile == "ancient_steel_crossbow")


def _recorder():
    seen = []

    def original(spec):
        seen.append(spec)
        return spec

    return seen, original


def test_gen5_returns_the_original_function() -> None:
    _, original = _recorder()
    assert crossbow_variant.construct(original, GEN5) is original


def test_gen5_ancient_steel_row_is_todays_triple() -> None:
    assert ANCIENT_STEEL.damage == (16.0, 42.0, 123.0)
    assert ANCIENT_STEEL.max_durability == 180.0
    assert ANCIENT_STEEL.durability_loss_on_hit == 0.28


def test_gen7_serves_the_0_7_2_triple_for_ancient_steel() -> None:
    seen, original = _recorder()
    wrapped = crossbow_variant.construct(original, GEN7)
    wrapped(ANCIENT_STEEL)
    served = seen[-1]
    assert served.damage == (10.5, 14.0, 82.0)
    table = load(GEN7).row("weapons", "item:Weapon_Crossbow_Ancient_Steel")["damage"]
    assert (
        float(table["Standard_Projectile_Damage"]["base_first"]),
        float(table["Combo_Projectile_Damage"]["base_first"]),
        float(table["Signature_BigArrow_Damage"]["base_first"]),
    ) == served.damage


def test_gen7_leaves_durability_to_the_weapon_durability_hook() -> None:
    seen, original = _recorder()
    crossbow_variant.construct(original, GEN7)(ANCIENT_STEEL)
    served = seen[-1]
    assert served.max_durability == ANCIENT_STEEL.max_durability
    assert served.durability_loss_on_hit == ANCIENT_STEEL.durability_loss_on_hit
    assert served.guard_stamina_value == ANCIENT_STEEL.guard_stamina_value


def test_gen7_iron_row_keeps_the_gen7_iron_triple() -> None:
    seen, original = _recorder()
    iron = dataclasses.replace(
        ANCIENT_STEEL,
        profile="iron_crossbow",
        asset_id="Weapon_Crossbow_Iron",
        damage=(10.0, 27.0, 78.0),
    )
    crossbow_variant.construct(original, GEN7)(iron)
    assert seen[-1].damage == (10.0, 13.0, 78.0)


def test_non_crossbow_rows_pass_through_unchanged() -> None:
    seen, original = _recorder()
    dagger = dataclasses.replace(ANCIENT_STEEL, template="daggers")
    crossbow_variant.construct(original, GEN7)(dagger)
    assert seen[-1] is dagger


def test_unknown_switch_value_raises() -> None:
    _, original = _recorder()
    profile = types.SimpleNamespace(
        constructs=dataclasses.replace(GEN7.constructs, crossbow_scalar_variant="bogus")
    )
    try:
        crossbow_variant.construct(original, profile)
    except ValueError:
        return
    raise AssertionError("an unknown switch value must raise")
