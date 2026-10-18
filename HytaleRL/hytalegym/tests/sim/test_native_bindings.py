"""Native binding types construct (gen 7, row 6c): gen 5 keeps the 0.5.7 profile's type names; gen 7 names a bound
root by its item's 0.7.2 Interactions map, which moves the crossbow reload from Ability3 to Ability4."""

from __future__ import annotations

from hytalegym.jax.combat.arsenal.profiles import hytale_0_5_7 as profile_module
from hytalegym.sim.common.content import load
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import native_bindings
from hytalegym.sim.gen7.profile import PROFILE as GEN7

ORIGINAL = profile_module.hytale_0_5_7_native_profile_bindings
RELOAD = "Root_Common_StatAmmoReload_Entry"


def test_gen5_returns_the_original() -> None:
    assert native_bindings.construct(ORIGINAL, GEN5) is ORIGINAL


def test_the_0_7_2_crossbow_binds_its_reload_under_ability4() -> None:
    types = native_bindings.item_binding_types(load(GEN7), "Weapon_Crossbow_Iron")
    assert types[RELOAD] == ("Ability4",)
    assert types["Root_Weapon_Crossbow_Primary_Signature"] == ("Primary",)
    assert native_bindings.item_binding_types(load(GEN7), "No_Such_Item") == {}


def test_gen7_moves_only_the_crossbow_reloads_to_ability4() -> None:
    hooked = native_bindings.construct(ORIGINAL, GEN7)
    changed = {}
    for name in profile_module.PROFILE_NAMES:
        gen5, gen7 = ORIGINAL(name), hooked(name)
        # Item, slot mapping, guard and stats unchanged; the ids too.
        assert gen7._replace(abilities=gen5.abilities) == gen5
        assert [b.interaction_id for b in gen7.abilities] == [b.interaction_id for b in gen5.abilities]
        changed.update(
            {
                (name, slot): (a.interaction_type, b.interaction_type)
                for slot, (a, b) in enumerate(zip(gen5.abilities, gen7.abilities))
                if a != b
            }
        )
    assert changed == {
        ("iron_crossbow", 2): ("Ability3", "Ability4"),
        ("ancient_steel_crossbow", 2): ("Ability3", "Ability4"),
    }
    assert hooked("iron_crossbow").abilities[2].interaction_id == RELOAD
    assert hooked("iron_crossbow") is hooked("iron_crossbow")
