"""Construct switches: gen 5 keeps today's behaviour; gen-7 stubs exist and refuse."""

from __future__ import annotations

import dataclasses
import importlib

import pytest

from hytalegym.sim.common.content import load
from hytalegym.sim.common.profile import Constructs
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7 import mechanics
from hytalegym.sim.gen7.profile import PROFILE as GEN7

from . import support

CHANGED = [
    field.name
    for field in dataclasses.fields(Constructs)
    if getattr(GEN5.constructs, field.name) != getattr(GEN7.constructs, field.name)
]


def test_gen5_profile_is_todays_behaviour() -> None:
    assert GEN5.constructs == Constructs()


#: Constructs implemented and differential-gated so far: line of sight (phase 3, 2026-10-18) and the NPC
#: walk jump velocity (phase 3b, 2026-10-18).
IMPLEMENTED_SWITCHES = {"line_of_sight", "npc_walk_jump_velocity", "dagger_swing_damage", "mana_pool", "guard_cancel_window",
                       "native_binding_types", "npc_jump_arc",
                       "bench_crafting", "crafting_materials", "weapon_durability", "landing_walk_resume"}


#: Switches a gen-7 decision leaves at the gen-5 default: the module stays a stub and is never reached
#: (slope_limits, decided 2026-10-18 from the 0.7.2 jar scan).
DECIDED_UNCHANGED = {"slope_limits"}


def test_every_changed_switch_has_an_owning_module_and_refuses_until_implemented() -> None:
    assert not set(CHANGED) & DECIDED_UNCHANGED
    assert set(CHANGED) | DECIDED_UNCHANGED == set(mechanics.CONSTRUCT_MODULES)
    for switch in CHANGED:
        module = mechanics.construct_module(switch)
        owned = getattr(module, "SWITCHES", (getattr(module, "SWITCH", None),))
        assert switch in owned
        assert module.IMPLEMENTED is (switch in IMPLEMENTED_SWITCHES)
        if switch in IMPLEMENTED_SWITCHES:
            assert mechanics.require_construct(GEN7, switch) is module
        else:
            with pytest.raises(NotImplementedError):
                mechanics.require_construct(GEN7, switch)


@pytest.mark.parametrize(
    "module_name",
    sorted(set(mechanics.CONSTRUCT_MODULES.values()) - {"line_of_sight", "npc_jump", "dagger_damage", "mana_pool", "guard_window",
                                                         "native_bindings", "npc_jump_arc",
                                                         "bench_crafting", "crafting_materials", "durability", "landing_walk"}),
)
def test_stub_entry_points_raise(module_name: str) -> None:
    module = importlib.import_module(f"hytalegym.sim.gen7.mechanics.{module_name}")
    functions = [
        value
        for name, value in vars(module).items()
        if callable(value) and getattr(value, "__module__", "") == module.__name__ and not isinstance(value, type)
    ]
    assert functions
    for function in functions:
        arity = function.__code__.co_argcount
        with pytest.raises(NotImplementedError):
            function(*([None] * arity))


def test_design_inputs_exist_in_the_gen7_tables() -> None:
    t = load(GEN7)
    assert t.interaction("DamageEntityParent_Hit_Interrupt_On_Hit").type == "Serial"
    assert t.effect("Hit_Interrupt_Recovery_Pending").duration == 2.0
    assert t.effect("Hit_Interrupt_Eligible").duration == 1.0
    assert t.effect("Hit_Interrupt_Recent_Reaction_A").duration == t.effect("Hit_Interrupt_Recent_Reaction_B").duration == 4.0
    role = t.row("npc_roles", "role:Template_Intelligent")
    assert "component_references" in role
    assert t.row("weapons", "interaction:Ability_Fireball_Cast")["document"]["Next"]["Interactions"][0]["Interactions"][0]["RunTime"] == 0.87
    assert "projectile_config:Projectile_Config_Hookshot" in t.table("weapons").rows
    assert "interaction:Ability_ChainHook_Cast" in t.table("weapons").rows
    assert t.table("blocks").rows["hitbox:" + t.block("Furniture_Kweebec_Door").hitbox_type]["boxes"]
    assert t.movement_config("Mount").fields
    assert t.movement_config().number("ClimbSpeed") == 0.035
    assert "item:Glider_Scrap" in t.table("weapons").rows
    assert t.stat("GlidingActive").maximum is not None


def test_mechanics_document_names_every_switch() -> None:
    text = (support.SIM / "gen7" / "mechanics_072.md").read_text(encoding="utf-8")
    for switch in CHANGED:
        assert f"`{switch}`" in text, switch
