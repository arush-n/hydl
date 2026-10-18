"""Guard cancel window construct (gen 7, row 5d): gen 5 keeps the guard root's rule; gen 7 lets Guard cancel a
Sword, Mace, Battleaxe or Daggers Primary attack only while its 0.7.2 root (InterruptedBy Secondary) winds up."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from hytalegym.jax.combat import AGENT_ENTITY
from hytalegym.jax.combat.arsenal import arsenal_runtime_config, empty_arsenal_state, hytale_0_5_7_loadouts
from hytalegym.jax.combat.arsenal.profiles import hytale_0_5_7 as profile_module
from hytalegym.jax.combat.arsenal.programs import interaction_rules
from hytalegym.jax.combat.arsenal.schema.contract import INTERACTION_TYPE_PRIMARY, INTERACTION_TYPE_SECONDARY
from hytalegym.sim.common.content import load
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import guard_window
from hytalegym.sim.gen7.profile import PROFILE as GEN7


def test_gen5_returns_both_originals() -> None:
    assert guard_window.ability_root_window(profile_module._ability, GEN5) is profile_module._ability
    original = interaction_rules.active_operation_interrupted_by
    assert guard_window.operation_window(original, GEN5) is original


@pytest.mark.parametrize(
    ("asset", "expected"),
    (
        ("Weapon_Sword_Primary_Swing_Left", 0.111),
        ("Weapon_Sword_Primary_Swing_Down", 0.361),
        ("Weapon_Mace_Primary_Swing_Left", 0.536),
        ("Weapon_Battleaxe_Primary_Swing_Down_Left", 0.348),
        ("Weapon_Daggers_Primary_Stab_Left", 0.066),
        ("Weapon_Sword_Primary_Thrust", None),  # a Charging root: no RunTime, its own row
        ("Weapon_Shortbow_Primary_Shoot", None),  # not one of the four families
    ),
)
def test_root_windows_come_from_the_0_7_2_table(asset: str, expected: float | None) -> None:
    window = guard_window.root_window_seconds(load(GEN7), asset)
    assert window == pytest.approx(expected) if expected is not None else window is None


def test_gen7_ability_stamps_the_root_window_and_leaves_others() -> None:
    hooked = guard_window.ability_root_window(profile_module._ability, GEN7)
    stamped = hooked("Weapon_Sword_Primary_Swing_Left", 0.334)
    assert stamped["interrupted_by_type_mask"] == 1 << INTERACTION_TYPE_SECONDARY
    assert stamped["interruptible_after_seconds"] == pytest.approx(0.111)
    assert hooked("Weapon_Shortbow_Primary_Shoot", 0.5) == profile_module._ability("Weapon_Shortbow_Primary_Shoot", 0.5)
    # An ability that authors its own mask keeps it (the crossbow reload's "after" boundary is one such).
    own = hooked("Weapon_Sword_Primary_Swing_Left", 0.334, interrupted_by_type_mask=1, interruptible_after_seconds=0.2)
    assert (own["interrupted_by_type_mask"], own["interruptible_after_seconds"]) == (1, 0.2)


@pytest.fixture
def gen7_sword_loadout(monkeypatch: pytest.MonkeyPatch):
    """The iron sword loadout with the gen-7 root windows stamped (profile bank rebuilt, then restored)."""

    monkeypatch.setattr(profile_module, "_ability", guard_window.ability_root_window(profile_module._ability, GEN7))
    profile_module._profile_bank.cache_clear()
    yield arsenal_runtime_config(hytale_0_5_7_loadouts(("iron_sword",))).loadout
    monkeypatch.undo()
    profile_module._profile_bank.cache_clear()


def _arsenal_at(slot: int, elapsed: float):
    arsenal = empty_arsenal_state(1)
    return arsenal._replace(
        active_ability_slot=arsenal.active_ability_slot.at[0, AGENT_ENTITY].set(slot),
        active_ability_root_slot=arsenal.active_ability_root_slot.at[0, AGENT_ENTITY].set(slot),
        ability_elapsed_seconds=arsenal.ability_elapsed_seconds.at[0, AGENT_ENTITY].set(elapsed),
    )


def test_gen7_guard_interrupts_the_left_swing_only_while_its_root_runs(gen7_sword_loadout) -> None:
    hooked = guard_window.operation_window(interaction_rules.active_operation_interrupted_by, GEN7)
    secondary = jnp.full((1, 2), INTERACTION_TYPE_SECONDARY, dtype=jnp.int32)
    primary = jnp.full((1, 2), INTERACTION_TYPE_PRIMARY, dtype=jnp.int32)
    swing = 0  # Weapon_Sword_Primary_Swing_Left
    for elapsed, expected in ((0.0, True), (0.1, True), (0.11, True), (0.1334, False), (0.3, False)):
        interrupted = hooked(_arsenal_at(swing, elapsed), gen7_sword_loadout, secondary)
        assert bool(interrupted[0, AGENT_ENTITY]) is expected, elapsed
    # A Primary request is not in the root's mask: never an interrupt.
    assert not bool(hooked(_arsenal_at(swing, 0.05), gen7_sword_loadout, primary)[0, AGENT_ENTITY])
    # Without the operation hook the stamped boundary would open the window instead (the "after" reading).
    plain = interaction_rules.active_operation_interrupted_by
    assert not bool(plain(_arsenal_at(swing, 0.05), gen7_sword_loadout, secondary)[0, AGENT_ENTITY])
    assert bool(plain(_arsenal_at(swing, 0.3), gen7_sword_loadout, secondary)[0, AGENT_ENTITY])
