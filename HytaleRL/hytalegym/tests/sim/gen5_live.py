"""Today's gen-5 engine values, read from the engine itself (read-only).

Each reader returns the same keys as the matching adapter in
``hytalegym.sim.common.content.adapters``; values are what the engine holds now.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

HYTALEGYM = Path(__file__).resolve().parents[2]
PACKAGE = HYTALEGYM / "hytalegym"
ITEM_PROGRAMS = PACKAGE / "jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json"
CRAFTING_RECORDING = HYTALEGYM / "tests/fidelity/fixtures/world/native_crafting_catalog_recording.json"


def player_locomotion() -> dict[str, Any]:
    from hytalegym.jax.combat import types
    from hytalegym.sim.common.content import adapters

    names = [entry.target for entry in adapters.SWAP_PLAN if entry.adapter == "player_locomotion"]
    return {name: getattr(types, name) for name in names}


def player_stamina() -> dict[str, Any]:
    from hytalegym.jax.combat import types
    from hytalegym.sim.common.content import adapters

    names = [entry.target for entry in adapters.SWAP_PLAN if entry.adapter == "player_stamina"]
    return {name: getattr(types, name) for name in names}


def mechanics_resource_constants() -> dict[str, Any]:
    from hytalegym.jax.combat.mechanics.schema import contract
    from hytalegym.sim.common.content import adapters

    names = [entry.target for entry in adapters.SWAP_PLAN if entry.adapter == "mechanics_resource_constants"]
    return {name: getattr(contract, name) for name in names}


def mechanics_factory_resources() -> dict[str, Any]:
    """float32 values as the engine stores them (see the adapter test)."""

    import numpy as np

    from hytalegym.jax.combat.mechanics import factory
    from hytalegym.jax.combat.mechanics.schema.contract import (
        RESOURCE_MAGIC_CHARGES,
        RESOURCE_MANA,
    )

    rules = factory.default_mechanics_rules(1)
    amount = np.asarray(rules.resource_regen_amount)[0, 0]
    interval = np.asarray(rules.resource_regen_interval_seconds)[0, 0]
    return {
        "mana_maximum": inspect.signature(factory.default_mechanics_rules).parameters["mana_maximum"].default,
        "mana_regen_amount": amount[RESOURCE_MANA],
        "mana_regen_interval_seconds": interval[RESOURCE_MANA],
        "magic_charges_regen_amount": amount[RESOURCE_MAGIC_CHARGES],
        "magic_charges_regen_interval_seconds": interval[RESOURCE_MAGIC_CHARGES],
    }


def walk_steering_sprint_multiplier() -> float:
    from hytalegym.jax.world.entities.locomotion import walk_steering_contract

    return walk_steering_contract()["sprint_boundary"]["installed_default_forward_multiplier"]


def ruleset() -> dict[str, Any]:
    from hytalegym.rulesets import load_combat_ruleset

    return load_combat_ruleset()


def health_regeneration() -> dict[str, Any]:
    return dict(ruleset()["regeneration"])


def npc_matchup() -> dict[str, Any]:
    r = ruleset()
    agent, target = r["agent"], r["target"]
    return {
        "agent.max_health": agent["max_health"],
        "agent.asset_max_walk_speed": agent["asset_max_walk_speed"],
        "agent.acceleration": agent["acceleration"],
        "agent.knockback_scale": agent["knockback_scale"],
        "agent.asset_eye_height": agent["asset_eye_height"],
        "agent.model_scale": agent["model_scale"],
        "agent.bounding_box": list(agent["bounding_box"]),
        "agent.walk_gravity": agent["walk_controller"]["gravity"],
        "agent.jump_velocity_gravity_floor": agent["jump_velocity_gravity_floor"],
        "agent.max_fall_speed": agent["walk_controller"]["max_fall_speed"],
        "agent.combat_relative_turn_speed": agent["steering_relative_turn_speed"],
        "target.max_health": target["max_health"],
        "target.asset_max_walk_speed": target["asset_max_walk_speed"],
        "target.relative_chase_speed": target["relative_chase_speed"],
        "target.chase_speed": target["chase_speed"],
        "target.acceleration": target["acceleration"],
        "target.asset_eye_height": target["asset_eye_height"],
        "target.model_scale": target["model_scale"],
        "target.effective_eye_height": target["effective_eye_height"],
        "target.bounding_box": list(target["bounding_box"]),
        "target.combat_relative_turn_speed": target["head_aim_relative_turn_speed"],
        "target.view_range": target["chase_senses"]["view_range"],
        "target.hearing_range": target["chase_senses"]["hearing_range"],
        "target.turn_degrees_per_second": target["turn_degrees_per_second"],
        "target.melee_damage": sorted({attack["damage"] for attack in target["attacks"]}),
    }


def compiled_weapon_durability() -> list[dict[str, Any]]:
    return json.loads(ITEM_PROGRAMS.read_text(encoding="utf-8"))["weapon_items"]


def compiled_source_ids() -> frozenset[str]:
    from hytalegym.combat.assets.catalog import COMPILED_SOURCE_IDS

    return COMPILED_SOURCE_IDS


def scalar_variants() -> list[Any]:
    from hytalegym.combat.assets.weapon_variants import SCALAR_VARIANTS

    return list(SCALAR_VARIANTS)


def _profiles():
    from hytalegym.jax.combat.arsenal.profiles import hytale_0_5_7

    return hytale_0_5_7


def iron_template_damage() -> dict[str, dict[str, Any]]:
    P = _profiles()
    out = {}
    for template in ("sword", "mace", "battleaxe", "daggers", "shortbow", "crossbow"):
        parameters = inspect.signature(getattr(P, f"_iron_{template}")).parameters
        out[template] = {
            "damage": tuple(float(v) for v in parameters["damage"].default),
            "guard_stamina_value": float(parameters["guard_stamina_value"].default),
        }
        if "angled_damage" in parameters:
            out[template]["angled_damage"] = tuple(float(v) for v in parameters["angled_damage"].default)
    return out


def sword_knockback() -> dict[str, dict[str, Any]]:
    """Force tuple and force mode of each damaging sword melee event."""

    P = _profiles()
    modes = {P.FORCE_SET: "Set", P.FORCE_ADD: "Add"}
    profile = P._CATALOG["iron_sword"]()
    out = {}
    for ability in profile["abilities"]:
        melee = [event for event in ability["events"] if P.EF_DAMAGE in event["f"]]
        names = (
            [ability["asset"]]
            if len(melee) == 1
            else [f"{ability['asset']}/spin", None, f"{ability['asset']}/stab"]
        )
        for name, event in zip(names, melee):
            if name is None:
                continue
            f = event["f"]
            out[name] = {
                "force": (f[P.EF_FORCE_X], f[P.EF_FORCE_Y], f[P.EF_FORCE_Z], f[P.EF_FORCE_MAGNITUDE]),
                "velocity_type": modes[event["i"][P.EI_FORCE_MODE]],
            }
    return out


def sword_selectors() -> dict[str, tuple[float, ...]]:
    from hytalegym.sim.common.content.adapters import SWORD_SELECTORS

    P = _profiles()
    return {name: tuple(float(v) for v in getattr(P, name)) for name in SWORD_SELECTORS}


def melee_hold_constants() -> dict[str, float]:
    P = _profiles()
    profile = P._CATALOG["iron_sword"]()
    thrust = next(a for a in profile["abilities"] if a["asset"] == "Weapon_Sword_Primary_Thrust")
    return {
        "_NESTED_OUTER_GATE_SECONDS": P._NESTED_OUTER_GATE_SECONDS,
        "_SWORD_THRUST_HOLD_SECONDS": P._SWORD_THRUST_HOLD_SECONDS,
        "_DOWNSTRIKE_HOLD_SECONDS": P._DOWNSTRIKE_HOLD_SECONDS,
        "_MACE_CHARGED_HOLD_SECONDS": P._MACE_CHARGED_HOLD_SECONDS,
        "sword_thrust_hold_speed_multiplier": thrust["hold_speed_multiplier"],
    }


def shortbow_tiers() -> dict[str, tuple[float, ...]]:
    P = _profiles()
    profile = P._CATALOG["iron_shortbow"]()
    shots, volleys = [], []
    for ability in profile["abilities"]:
        events = [event for event in ability["events"] if P.EF_PROJECTILE_SPEED in event["f"]]
        if not events:
            continue
        if "Primary_Shoot" in ability["asset"]:
            shots.append((ability["requested_charge_time_seconds"], events[0]["f"]))
        else:
            volleys.append(ability["requested_charge_time_seconds"])
    return {
        "charge_thresholds": tuple(float(t) for t, _ in shots),
        "speeds": tuple(float(f[P.EF_PROJECTILE_SPEED]) for _, f in shots),
        "gravities": tuple(float(f[P.EF_PROJECTILE_GRAVITY]) for _, f in shots),
        "terminal_velocities": tuple(float(f[P.EF_PROJECTILE_TERMINAL_VELOCITY]) for _, f in shots),
        "volley_charge_thresholds": tuple(float(t) for t in volleys),
        "damage": tuple(float(v) for v in inspect.signature(P._iron_shortbow).parameters["damage"].default),
    }


def spear_rows() -> dict[str, tuple[float, float, int]]:
    P = _profiles()
    out = {}
    for name in P.SPEAR_PROFILE_NAMES:
        asset, _family, stab, projectile, stack = P._CATALOG[name].args
        out[asset] = (float(stab), float(projectile), int(stack))
    return out


def potions() -> dict[str, float]:
    P = _profiles()
    profile = P._CATALOG["potions"]()
    by_asset = {ability["asset"]: ability for ability in profile["abilities"]}

    def status(asset: str, effect: str):
        for event in by_asset[asset]["events"]:
            if event["kind"] == P.EVENT_STATUS and event["i"][P.EI_STATUS_ID] == P.semantic_id(effect):
                return event["f"]
        raise KeyError(effect)

    health = by_asset["Potion_Health"]["events"]
    heal = next(event for event in health if event["kind"] != P.EVENT_STATUS)
    stamina = by_asset["Potion_Stamina"]["events"]
    restore = next(event for event in stamina if event["kind"] != P.EVENT_STATUS)
    regen = status("Potion_Health", "Potion_Health_Regen")
    return {
        "health_instant_percent": float(next(v for v in heal["f"].values() if v)),
        "health_regen_percent": float(regen[P.EF_STATUS_HEALING]),
        "health_regen_duration": float(regen[P.EF_STATUS_DURATION_SECONDS]),
        "health_regen_cooldown": float(regen[P.EF_STATUS_COOLDOWN_SECONDS]),
        "stamina_restore_percent": float(next(v for v in restore["f"].values() if v)),
        "stamina_cooldown_duration": float(status("Potion_Stamina", "Potion_Stamina_Cooldown")[P.EF_STATUS_DURATION_SECONDS]),
        "antidote_duration": float(status("Potion_Antidote", "Antidote")[P.EF_STATUS_DURATION_SECONDS]),
    }


def outlander_arrow() -> dict[str, float]:
    from hytalegym.jax.combat.effects.schema import contract
    from hytalegym.sim.common.content import adapters

    names = [entry.target for entry in adapters.SWAP_PLAN if entry.adapter == "outlander_arrow"]
    return {name: getattr(contract, name) for name in names}


def native_crafting_recipes() -> dict[str, dict[str, Any]]:
    """The 0.5.7 native catalog recording in the adapter's comparison shape."""

    recording = json.loads(CRAFTING_RECORDING.read_text(encoding="utf-8"))
    out = {}
    for recipe in recording["recipes"]:
        out[recipe["recipe_id"]] = {
            "inputs": [
                (m["item_asset_id"], m["resource_type_id"], m["quantity"], tuple(sorted(m["excluded_item_asset_ids"])))
                for m in recipe["inputs"]
            ],
            "outputs": [(m["item_asset_id"], m["quantity"]) for m in recipe["outputs"]],
            "primary_output_item_asset_id": recipe["primary_output_item_asset_id"],
            "bench_requirements": [(b["id"], b["type"], b["required_tier_level"]) for b in recipe["bench_requirements"]],
            "knowledge_required": recipe["knowledge_required"],
            "required_memories_level": recipe["required_memories_level"],
            "time_seconds": recipe["time_seconds"],
        }
    return out


def control_immunity() -> dict[str, float]:
    from hytalegym.jax.combat.mechanics.schema import contract
    from hytalegym.sim.common.content import adapters

    names = [entry.target for entry in adapters.SWAP_PLAN if entry.adapter == "control_immunity"]
    return {name: getattr(contract, name) for name in names}


def dodge_constants() -> dict[str, object]:
    from hytalegym.jax.combat.mechanics.schema import contract
    from hytalegym.sim.common.content import adapters

    names = [entry.target for entry in adapters.SWAP_PLAN if entry.adapter == "dodge_constants"]
    return {name: getattr(contract, name) for name in names}


def crafting_capacities() -> dict[str, int]:
    from hytalegym.jax.crafting import contract

    return {
        name: getattr(contract, name)
        for name in ("RECIPE_TABLE_CAPACITY", "MAX_INGREDIENTS", "MAX_OUTPUTS", "MAX_BENCH_REQUIREMENTS")
    }
