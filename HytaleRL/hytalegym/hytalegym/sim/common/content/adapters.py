"""Rebuild the engine's hand-held constants and tables from content tables.

Each function reproduces, from a generation's tables, exactly the values a
gen-5 engine module holds today (asserted bit for bit in
``tests/sim/test_adapters.py``). The gen-5 source is never edited: a render
(phase 2) copies it from ``git show HEAD:<path>`` and redirects each symbol in
``SWAP_PLAN`` to its adapter's output for the active profile. ``SWAP_PLAN`` (and
its machine-readable form ``redirects.json``, written by
``tools/content_extract/redirects.py``) lists every constant and table covered,
its location, how it is held and its adapter. Adapters never import the engine.

Arithmetic note: where the engine wrote an exact decimal derived from assets
(``3.0`` stamina per second from 0.3 per 0.1 s) the adapter divides in decimal;
where the engine summed floats (``0.2 + 0.65``) the adapter sums floats. Each
case is named in a comment so the bits are reproduced, not approximated.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable, Mapping

from hytalegym.sim.common.content.loader import ContentTables

TICKS_PER_SECOND = 30  # engine.ticks_per_second (ruleset); not an asset value
_TIER = re.compile(r"Weapon_Shortbow_Primary_Shoot_Strength_(\d+)")


def _dec(value: float) -> Decimal:
    return Decimal(repr(float(value)))


def _ratio(numerator: float, denominator: float) -> float:
    return float(_dec(numerator) / _dec(denominator))


def _decimal_sum(*values: float) -> float:
    return float(sum((_dec(v) for v in values), Decimal(0)))


def _rule(stat, *, condition: str, inverse: bool | None = None):
    for rule in stat.regenerating:
        for identifier, inverted in rule.condition_ids():
            if identifier == condition and (inverse is None or inverted == inverse):
                return rule
    raise KeyError(f"{stat.stat_id} has no regeneration rule with {condition} inverse={inverse}")


# -- jax/combat/types.py: player locomotion and stamina ----------------------
def player_locomotion(t: ContentTables) -> dict[str, Any]:
    """``PLAYER_*`` movement constants (types.py, from MovementConfig)."""

    m = t.movement_config()  # GameplayConfigs Player.MovementConfig binding
    run = m.number("ForwardRunSpeedMultiplier")
    sprint = m.number("ForwardSprintSpeedMultiplier")
    # Idle carries the run multiplier and sprint has only a forward key: the
    # engine's documented decomposition (types.py comments).
    forward = (run, m.number("ForwardWalkSpeedMultiplier"), run, sprint, m.number("ForwardCrouchSpeedMultiplier"))
    backward = (
        run,
        m.number("BackwardWalkSpeedMultiplier"),
        m.number("BackwardRunSpeedMultiplier"),
        sprint,
        m.number("BackwardCrouchSpeedMultiplier"),
    )
    strafe = (
        run,
        m.number("StrafeWalkSpeedMultiplier"),
        m.number("StrafeRunSpeedMultiplier"),
        sprint,
        m.number("StrafeCrouchSpeedMultiplier"),
    )
    return {
        "PLAYER_JUMP_FORCE": m.number("JumpForce"),
        "PLAYER_VARIABLE_JUMP_FALL_FORCE": m.number("VariableJumpFallForce"),
        "PLAYER_BASE_SPEED": m.number("BaseSpeed"),
        "PLAYER_GAIT_SPEED_MULTIPLIERS": forward,
        "PLAYER_GAIT_FORWARD_SPEED_MULTIPLIERS": forward,
        "PLAYER_GAIT_BACKWARD_SPEED_MULTIPLIERS": backward,
        "PLAYER_GAIT_STRAFE_SPEED_MULTIPLIERS": strafe,
        "PLAYER_AIR_DRAG_MIN": m.number("AirDragMin"),
        "PLAYER_AIR_DRAG_MAX": m.number("AirDragMax"),
        "PLAYER_AIR_DRAG_MIN_SPEED": m.number("AirDragMinSpeed"),
        "PLAYER_AIR_DRAG_MAX_SPEED": m.number("AirDragMaxSpeed"),
        "PLAYER_AIR_CONTROL_MIN_SPEED": m.number("AirControlMinSpeed"),
        "PLAYER_AIR_CONTROL_MAX_SPEED": m.number("AirControlMaxSpeed"),
        "PLAYER_AIR_CONTROL_MIN_MULTIPLIER": m.number("AirControlMinMultiplier"),
        "PLAYER_AIR_CONTROL_MAX_MULTIPLIER": m.number("AirControlMaxMultiplier"),
    }


def _stamina_parts(t: ContentTables):
    stamina = t.stat("Stamina")
    delay = t.stat("StaminaRegenDelay")
    regen = stamina.regenerating[0]  # "Positive stamina regeneration values"
    sprint = _rule(stamina, condition="Sprinting", inverse=False)
    broken = None
    for interaction in stamina.row["min_value_effects"]["Interactions"]["Interactions"]:
        if isinstance(interaction, Mapping) and interaction.get("Type") == "ChangeStat":
            broken = float(interaction["StatModifiers"]["StaminaRegenDelay"])
    if broken is None:
        raise KeyError("Stamina MinValueEffects sets no StaminaRegenDelay")
    sprint_delay = float(t.stamina_plugin()["sprint_regen_delay_value"])
    return stamina, delay.regenerating[0], regen, sprint, broken, sprint_delay


def player_stamina(t: ContentTables) -> dict[str, float]:
    """types.py stamina constants, stated per second (decimal ratios)."""

    stamina, delay_rule, regen, sprint, broken, sprint_delay = _stamina_parts(t)
    return {
        "PLAYER_STAMINA_MAXIMUM": stamina.maximum,
        "PLAYER_STAMINA_MINIMUM": stamina.minimum,
        "PLAYER_SPRINT_STAMINA_DRAIN_PER_SECOND": -_ratio(sprint.amount, sprint.interval),
        "PLAYER_STAMINA_REGEN_PER_SECOND": _ratio(regen.amount, regen.interval),
        "PLAYER_STAMINA_REGEN_DELAY_RECOVERY_PER_SECOND": _ratio(delay_rule.amount, delay_rule.interval),
        "PLAYER_SPRINT_STAMINA_REGEN_DELAY_SECONDS": -sprint_delay,
        "PLAYER_BROKEN_STAMINA_REGEN_DELAY_SECONDS": -broken,
    }


# -- jax/combat/mechanics: resource constants and factory defaults -----------
def mechanics_resource_constants(t: ContentTables) -> dict[str, float]:
    """mechanics/schema/contract.py ``STAMINA_*`` and ``OXYGEN_*``."""

    stamina, delay_rule, regen, _, broken, _ = _stamina_parts(t)
    oxygen = t.stat("Oxygen")
    breathable = _rule(oxygen, condition="Suffocating", inverse=True)
    suffocating = _rule(oxygen, condition="Suffocating", inverse=False)
    if breathable.interval != suffocating.interval:
        raise ValueError("oxygen regeneration intervals differ; OXYGEN_REGEN_INTERVAL_SECONDS is one value")
    return {
        "STAMINA_MINIMUM": stamina.minimum,
        "STAMINA_MAXIMUM": stamina.maximum,
        "STAMINA_REGEN_AMOUNT": regen.amount,
        "STAMINA_REGEN_INTERVAL_SECONDS": regen.interval,
        "STAMINA_REGEN_DELAY_AMOUNT": delay_rule.amount,
        "STAMINA_REGEN_DELAY_INTERVAL_SECONDS": delay_rule.interval,
        "STAMINA_BROKEN_REGEN_DELAY_SECONDS": broken,
        "OXYGEN_MINIMUM": oxygen.minimum,
        "OXYGEN_MAXIMUM": oxygen.maximum,
        "OXYGEN_BREATHABLE_REGEN_AMOUNT": breathable.amount,
        "OXYGEN_SUFFOCATING_REGEN_AMOUNT": suffocating.amount,
        "OXYGEN_REGEN_INTERVAL_SECONDS": breathable.interval,
    }


def control_immunity(t: ContentTables) -> dict[str, float]:
    """mechanics/schema/contract.py ``CONTROL_IMMUNITY_*`` from Stats/Immunity.json."""

    immunity = t.stat("Immunity")
    return {
        "CONTROL_IMMUNITY_MAXIMUM": immunity.maximum,
        # The asset decays immunity (-0.1 per 0.1 s); the engine stores the
        # magnitude and subtracts it (mechanics/runtime/statuses.py).
        "CONTROL_IMMUNITY_REGEN_AMOUNT": -immunity.regenerating[0].amount,
        "CONTROL_IMMUNITY_REGEN_INTERVAL_SECONDS": immunity.regenerating[0].interval,
    }


def dodge_constants(t: ContentTables) -> dict[str, Any]:
    """mechanics/schema/contract.py ``DODGE_*`` from Dodge.json and Dodge_Left.

    ``DODGE_COOLDOWN_SECONDS`` (engine default), ``DODGE_LAUNCH_TICK``,
    ``DODGE_COST_TICK`` and ``DODGE_VELOCITY_REMOVAL_SQUARED`` are not asset
    values and stay hand-maintained.
    """

    branches = t.interaction("Dodge").document["Next"]
    left = t.interaction("Dodge_Left").document
    steps = left["Next"]["Interactions"]
    force = next(step for step in steps if step.get("Type") == "ApplyForce")
    nested = next(step for step in steps if step.get("Type") == "Condition")["Next"]["Interactions"]
    spend = next(step for step in nested if "Stamina" in step.get("StatModifiers", {}))
    delay = next(step for step in nested if "StaminaRegenDelay" in step.get("StatModifiers", {}))
    velocity = force["VelocityConfig"]
    return {
        # policy-mask order [forward, back, left, right]: a branch that names
        # an interaction carries the dodge payload; a Simple does not.
        "DODGE_AUTHORED_ACTION_MASK": tuple(
            isinstance(branches.get(key), str) for key in ("Forward", "Back", "Left", "Right")
        ),
        "DODGE_STAMINA_ADMISSION_COST": float(left["Costs"]["Stamina"]),
        "DODGE_STAMINA_SPEND_COST": -float(spend["StatModifiers"]["Stamina"]),
        "DODGE_FORCE": float(force["Force"]),
        "DODGE_INVULNERABILITY_SECONDS": t.effect("Dodge_Invulnerability").duration,
        "DODGE_REGEN_DELAY_SECONDS": float(delay["StatModifiers"]["StaminaRegenDelay"]),
        "DODGE_AIR_RESISTANCE": float(velocity["AirResistance"]),
        "DODGE_AIR_RESISTANCE_MAX": float(velocity["AirResistanceMax"]),
        "DODGE_GROUND_RESISTANCE": float(velocity["GroundResistance"]),
        "DODGE_GROUND_RESISTANCE_MAX": float(velocity["GroundResistanceMax"]),
        "DODGE_RESISTANCE_THRESHOLD": float(velocity["Threshold"]),
    }


def mechanics_factory_resources(t: ContentTables) -> dict[str, float]:
    """mechanics/factory.py ``default_mechanics_rules`` resource defaults."""

    mana = t.stat("Mana")
    charges = t.stat("MagicCharges")
    return {
        "mana_maximum": mana.maximum,
        "mana_regen_amount": mana.regenerating[0].amount,
        "mana_regen_interval_seconds": mana.regenerating[0].interval,
        "magic_charges_regen_amount": charges.regenerating[0].amount,
        "magic_charges_regen_interval_seconds": charges.regenerating[0].interval,
    }


def walk_steering_sprint_multiplier(t: ContentTables) -> float:
    """jax/world/entities/locomotion.py ``installed_default_forward_multiplier``."""

    return t.movement_config().number("ForwardSprintSpeedMultiplier")


# -- rulesets/hytale_0_5_7/kweebec_razorleaf_vs_trork_brawler_v9.json -------------
def health_regeneration(t: ContentTables) -> dict[str, float | int]:
    """Ruleset ``regeneration``: Health.json's NPC rule in ticks."""

    rule = _rule(t.stat("Health"), condition="IsPlayer", inverse=True)
    delay = next(c["Delay"] for c in rule.conditions if c.get("Id") == "NoDamageTaken")
    return {
        "delay_ticks": int(round(float(delay) * TICKS_PER_SECOND)),
        "interval_ticks": int(round(rule.interval * TICKS_PER_SECOND)),
        "fraction": rule.amount,
    }


def _decimal_product(*values: float) -> float:
    product = Decimal(1)
    for value in values:
        product *= _dec(value)
    return float(product)


def npc_matchup(t: ContentTables, agent_role: str = "Kweebec_Razorleaf", target_role: str = "Trork_Brawler") -> dict[str, Any]:
    """The ruleset fields that come from role and model files, by ruleset path.

    Scaled values (bounding box, effective eye height, chase speed) are decimal
    products, as the ruleset JSON writes them.
    """

    def side(role_id: str) -> tuple[Any, Mapping[str, Any], Any, float]:
        role = t.role(role_id)
        model = t.model(role.appearance)
        scale = model.max_scale if model.max_scale is not None else 1.0
        return role, role.walk_controller() or {}, model, scale

    def box(model, scale: float) -> list[float]:
        low = [_decimal_product(v, scale) for v in model.hitbox_min]
        high = [_decimal_product(v, scale) for v in model.hitbox_max]
        return low + high

    agent, agent_walk, agent_model, agent_scale = side(agent_role)
    target, target_walk, target_model, target_scale = side(target_role)
    target_vars = target.fields.get("InteractionVars") or {}
    melee = target_vars["Melee_Damage"]["Interactions"][0]["DamageCalculator"]["BaseDamage"]["Physical"]
    return {
        "agent.max_health": agent.max_health,
        "agent.asset_max_walk_speed": float(agent_walk["MaxWalkSpeed"]),
        "agent.acceleration": float(agent_walk["Acceleration"]),
        "agent.knockback_scale": agent.knockback_scale,
        "agent.asset_eye_height": agent_model.eye_height,
        "agent.model_scale": agent_scale,
        "agent.bounding_box": box(agent_model, agent_scale),
        "agent.walk_gravity": float(agent_walk["Gravity"]),
        "agent.jump_velocity_gravity_floor": float(agent_walk["Gravity"]),
        "agent.max_fall_speed": float(agent_walk["MaxFallSpeed"]),
        "agent.combat_relative_turn_speed": float(agent.parameters["CombatRelativeTurnSpeed"]),
        "target.max_health": target.max_health,
        "target.asset_max_walk_speed": float(target_walk["MaxWalkSpeed"]),
        "target.relative_chase_speed": float(target.parameters["RelativeChaseSpeed"]),
        "target.chase_speed": _decimal_product(target_walk["MaxWalkSpeed"], target.parameters["RelativeChaseSpeed"]),
        "target.acceleration": float(target_walk["Acceleration"]),
        "target.asset_eye_height": target_model.eye_height,
        "target.model_scale": target_scale,
        "target.effective_eye_height": _decimal_product(target_model.eye_height, target_scale),
        "target.bounding_box": box(target_model, target_scale),
        "target.combat_relative_turn_speed": float(target.parameters["CombatRelativeTurnSpeed"]),
        "target.view_range": float(target.parameters["ViewRange"]),
        "target.hearing_range": float(target.parameters["HearingRange"]),
        "target.turn_degrees_per_second": float(target_walk["MaxRotationSpeed"]),
        "target.melee_damage": [float(melee)],
    }


# -- jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json -------------
def compiled_weapon_durability(t: ContentTables, item_ids: Iterable[str]) -> list[dict[str, Any]]:
    """``weapon_items`` rows (same rule as generate_item_durability.py)."""

    rows = []
    for item_id in sorted(item_ids):
        entry = t.durability(item_id)
        if entry is None:
            continue
        rows.append(
            {
                "item_asset_id": entry.item_id,
                "max_durability": entry.max_durability,
                "durability_loss_on_hit": entry.durability_loss_on_hit,
                "source_path": entry.source,
            }
        )
    return rows


# -- combat/assets/weapon_variants/hytale_0_5_7.py (SCALAR_VARIANTS) -----------
# Per template: the InteractionVars the variant table reads, in order, and the
# keys whose template default applies when an item omits the variable (others
# read 0.0). Mirrors tests/combat/assets/weapon_variants/
# test_asset_scalar_variants.py (_DAMAGE_KEYS, _DEFAULT_DAMAGE_INTERACTIONS).
VARIANT_DAMAGE_KEYS: dict[str, tuple[str, ...]] = {
    "sword": ("Swing_Left_Damage", "Swing_Right_Damage", "Swing_Down_Damage", "Thrust_Damage", "Vortexstrike_Spin_Damage", "Vortexstrike_Stab_Damage"),
    "mace": ("Swing_Left_Damage", "Swing_Right_Damage", "Swing_Up_Left_Damage", "Swing_Left_Charged_Damage", "Swing_Right_Charged_Damage", "Swing_Up_Left_Charged_Damage"),
    "battleaxe": ("Swing_Down_Left_Damage", "Swing_Down_Right_Damage", "Swing_Down_Damage", "Signature_Whirlwind_Damage"),
    "daggers": ("Swing_Left_Damage", "Swing_Right_Damage", "Stab_Left_Damage", "Stab_Right_Damage", "Razorstrike_Slash_Damage", "Razorstrike_Sweep_Damage", "Razorstrike_Lunge_Damage"),
    "shortbow": ("Primary_Shoot_Damage_Strength_0", "Primary_Shoot_Damage_Strength_1", "Primary_Shoot_Damage_Strength_2", "Primary_Shoot_Damage_Strength_3", "Primary_Shoot_Damage_Strength_4", "Signature_Volley_Damage"),
    "crossbow": ("Standard_Projectile_Damage", "Combo_Projectile_Damage", "Signature_BigArrow_Damage"),
    "shield": (),
}
VARIANT_DEFAULTED_KEYS: dict[str, frozenset[str]] = {
    "sword": frozenset(VARIANT_DAMAGE_KEYS["sword"]),
    "mace": frozenset(VARIANT_DAMAGE_KEYS["mace"]),
    "battleaxe": frozenset(VARIANT_DAMAGE_KEYS["battleaxe"]),
    "daggers": frozenset(VARIANT_DAMAGE_KEYS["daggers"]),
    "crossbow": frozenset({"Standard_Projectile_Damage", "Combo_Projectile_Damage"}),
    "shortbow": frozenset(),
    "shield": frozenset(),
}
GUARD_STAMINA_DEFAULT = 7.0  # variant table default when Guard_Wield sets none


def _variant_damage(item, template: str, key: str, *, angled: bool) -> float:
    entry = item.damage.get(key)
    if entry is None:
        return 0.0
    if entry.origin == "default" and key not in VARIANT_DEFAULTED_KEYS[template]:
        return 0.0
    value = entry.angled_first if angled else entry.base_first
    return 0.0 if value is None else value


def scalar_variant_values(t: ContentTables, asset_id: str, template: str) -> dict[str, Any]:
    item = t.weapon_item(asset_id)
    keys = VARIANT_DAMAGE_KEYS[template]
    return {
        "guard_stamina_value": GUARD_STAMINA_DEFAULT if item.guard_stamina_value is None else item.guard_stamina_value,
        "max_durability": item.max_durability or 0.0,
        "durability_loss_on_hit": item.durability_loss_on_hit or 0.0,
        "damage": tuple(_variant_damage(item, template, key, angled=False) for key in keys),
        "angled_damage": tuple(_variant_damage(item, template, key, angled=True) for key in keys)
        if template == "daggers"
        else (),
    }


# -- jax/combat/arsenal/profiles/hytale_0_5_7.py -------------------------------
IRON_TEMPLATE_ASSETS = {
    "sword": "Weapon_Sword_Iron",
    "mace": "Weapon_Mace_Iron",
    "battleaxe": "Weapon_Battleaxe_Iron",
    "daggers": "Weapon_Daggers_Iron",
    "shortbow": "Weapon_Shortbow_Iron",
    "crossbow": "Weapon_Crossbow_Iron",
}


def iron_template_damage(t: ContentTables) -> dict[str, dict[str, tuple[float, ...]]]:
    """``_iron_<template>(damage=..., angled_damage=..., guard_stamina_value=...)`` defaults."""

    out = {}
    for template, asset_id in IRON_TEMPLATE_ASSETS.items():
        values = scalar_variant_values(t, asset_id, template)
        out[template] = {"damage": values["damage"], "guard_stamina_value": values["guard_stamina_value"]}
        if template == "daggers":
            out[template]["angled_damage"] = values["angled_damage"]
    return out


SWORD_ATTACKS = {
    # profile ability -> damage interaction (its DamageEffects.Knockback)
    "Weapon_Sword_Primary_Swing_Left": "Weapon_Sword_Primary_Swing_Left_Damage",
    "Weapon_Sword_Primary_Swing_Right": "Weapon_Sword_Primary_Swing_Right_Damage",
    "Weapon_Sword_Primary_Swing_Down": "Weapon_Sword_Primary_Swing_Down_Damage",
    "Weapon_Sword_Primary_Thrust": "Weapon_Sword_Primary_Thrust_Damage",
    "Weapon_Sword_Signature_Vortexstrike/spin": "Weapon_Sword_Signature_Vortexstrike_Spin_Damage",
    "Weapon_Sword_Signature_Vortexstrike/stab": "Weapon_Sword_Signature_Vortexstrike_Stab_Damage",
}
SWORD_SELECTORS = {
    "_SWORD_SWING_LEFT_SELECTOR": "Weapon_Sword_Primary_Swing_Left_Selector",
    "_SWORD_SWING_RIGHT_SELECTOR": "Weapon_Sword_Primary_Swing_Right_Selector",
    "_SWORD_SWING_DOWN_SELECTOR": "Weapon_Sword_Primary_Swing_Down_Selector",
    "_SWORD_VORTEX_SPIN_SELECTOR": "Weapon_Sword_Signature_Vortexstrike_Spin_Selector",
    "_SWORD_THRUST_SELECTOR": "Weapon_Sword_Primary_Thrust_Selector",
    "_SWORD_VORTEX_STAB_SELECTOR": "Weapon_Sword_Signature_Vortexstrike_Stab_Selector",
}


def sword_knockback(t: ContentTables) -> dict[str, dict[str, Any]]:
    """Per sword melee event: ``force=(x, y, z, magnitude)`` and velocity type."""

    out = {}
    for ability, interaction_id in SWORD_ATTACKS.items():
        knockback = t.interaction(interaction_id).knockback
        x, y, z = knockback.direction
        out[ability] = {"force": (x, y, z, knockback.force), "velocity_type": knockback.velocity_type}
    return out


def sword_selectors(t: ContentTables) -> dict[str, tuple[float, ...]]:
    """``_SWORD_*_SELECTOR`` programs (the tuple field order of the profile)."""

    out = {}
    for constant, interaction_id in SWORD_SELECTORS.items():
        node = t.interaction(interaction_id)
        s = node.selector
        runtime = float(node.document["RunTime"])
        if s["Id"] == "Horizontal":
            direction = 1.0 if s.get("Direction") == "ToLeft" else -1.0
            out[constant] = (
                runtime,
                float(s.get("StartDistance", 0.0)),
                float(s.get("EndDistance", 0.0)),
                float(s.get("ExtendBottom", 0.0)),
                float(s.get("ExtendTop", 0.0)),
                float(s.get("Length", 0.0)),
                direction,
                float(s.get("YawStartOffset", 0.0)),
                float(s.get("PitchOffset", 0.0)),
                float(s.get("RollOffset", 0.0)),
            )
        else:
            out[constant] = (
                runtime,
                float(s.get("StartDistance", 0.0)),
                float(s.get("EndDistance", 0.0)),
                float(s.get("ExtendLeft", 0.0)),
                float(s.get("ExtendRight", 0.0)),
                float(s.get("ExtendBottom", 0.0)),
                float(s.get("ExtendTop", 0.0)),
                float(s.get("YawOffset", 0.0)),
                float(s.get("PitchOffset", 0.0)),
                float(s.get("RollOffset", 0.0)),
            )
    return out


def melee_hold_constants(t: ContentTables) -> dict[str, float]:
    """Nested Charging gates (profile module constants)."""

    outer = t.interaction("Weapon_Sword_Primary").charge_thresholds[1]
    thrust = t.interaction("Weapon_Sword_Primary_Thrust").charge_thresholds[1]
    downstrike = t.interaction("Weapon_Battleaxe_Primary_Downstrike").charge_thresholds[1]
    mace = t.interaction("Weapon_Mace_Primary_Swing_Left_Charged").charge_thresholds[1]
    return {
        "_NESTED_OUTER_GATE_SECONDS": outer,
        # The engine wrote the literal 0.85 here but the float sum below for the
        # battleaxe; both are reproduced as written.
        "_SWORD_THRUST_HOLD_SECONDS": _decimal_sum(outer, thrust),
        "_DOWNSTRIKE_HOLD_SECONDS": outer + downstrike,
        "_MACE_CHARGED_HOLD_SECONDS": _decimal_sum(outer, mace),
        "sword_thrust_hold_speed_multiplier": float(
            t.interaction("Weapon_Sword_Primary_Thrust").document["HorizontalSpeedMultiplier"]
        ),
    }


def sword_thrust_timing(t: ContentTables) -> dict[str, float]:
    """Iron sword thrust: the charged branch's event time (its Simple RunTime), 0.05 on 0.5.7 and 0.043 on 0.7.2.

    The charged branch is the highest ``Next`` threshold: Serial[StaminaCost, StaminaRegenDelay, Simple(event) ->
    Serial[Force, Selector]]. The thrust's total is not read here: the asset sum on 0.7.2 (0.043 + 0.267 + 0.223 =
    0.533) ends the render's thrust at r+40 where live ends at r+44 (``mechanics_072.md``, row 5f).
    """

    nexts = t.interaction("Weapon_Sword_Primary_Thrust").document["Next"]
    branch = nexts[max(nexts, key=float)]
    (simple,) = (node for node in branch["Interactions"] if node.get("Type") == "Simple")
    return {"event_seconds": float(simple["RunTime"])}


def shortbow_tiers(t: ContentTables, asset_id: str = "Weapon_Shortbow_Iron") -> dict[str, tuple[float, ...]]:
    """``_iron_shortbow`` draw ladder: thresholds, launch speeds, gravities."""

    # Each Charging branch names the tier it fires (Weapon_Shortbow_Primary_
    # Shoot_Strength_<n>); 0.7.2 adds a 0.0 branch that fires nothing.
    branches = t.effective_row("weapons", "interaction:Weapon_Shortbow_Primary_Shoot_Charge")["document"]["Next"]
    tiers = {}
    for key, branch in branches.items():
        match = _TIER.search(json.dumps(branch, sort_keys=True))
        if match:
            tiers[int(match.group(1))] = float(key)
    thresholds = tuple(tiers[n] for n in sorted(tiers))
    configs = [t.projectile_config(f"Projectile_Config_Arrow_Shortbow_Strength_{n}") for n in sorted(tiers)]
    volley = t.interaction("Weapon_Shortbow_Signature_Volley_Charge").charge_thresholds
    return {
        "charge_thresholds": thresholds,
        "speeds": tuple(c.launch_force for c in configs),
        "gravities": tuple(c.gravity for c in configs),
        "terminal_velocities": tuple(c.terminal_velocity_air for c in configs),
        "volley_charge_thresholds": tuple(volley),
        "damage": scalar_variant_values(t, asset_id, "shortbow")["damage"],
    }


def spear_rows(t: ContentTables, asset_ids: Iterable[str]) -> dict[str, tuple[float, float, int]]:
    """``partial(_spear, asset, family, stab_damage, projectile_damage, stack)``."""

    out = {}
    for asset_id in asset_ids:
        item = t.weapon_item(asset_id)
        row = t.effective_row("weapons", f"item:{asset_id}")
        projectile = t.legacy_projectile(row["var_projectile_ids"]["Spear_Throw_Charged_Projectile"])
        stack = row["max_stack"]
        out[asset_id] = (item.damage["Spear_Stab_Damage"].base_first, projectile.damage, int(stack))
    return out


# -- jax/combat/effects/schema/contract.py ------------------------------------
def outlander_arrow(t: ContentTables) -> dict[str, float]:
    p = t.legacy_projectile("Outlander_Hunter_Arrow")
    model = t.model(p.appearance)
    return {
        "OUTLANDER_ARROW_MODEL_HALF_EXTENT": model.hitbox_max[0],
        "OUTLANDER_ARROW_MUZZLE_VELOCITY": p.muzzle_velocity,
        "OUTLANDER_ARROW_TERMINAL_VELOCITY": p.terminal_velocity,
        "OUTLANDER_ARROW_GRAVITY": p.gravity,
        "OUTLANDER_ARROW_DAMAGE": p.damage,
        "OUTLANDER_ARROW_DEAD_TIME_SECONDS": p.dead_time,
        "OUTLANDER_ARROW_AUTHORED_LIFETIME_SECONDS": p.time_to_live,
        "OUTLANDER_ARROW_DEPTH_OFFSET": p.depth_shot,
        "OUTLANDER_ARROW_VERTICAL_OFFSET": p.vertical_center_shot,
    }


# -- profiles/hytale_0_5_7.py::_potions ----------------------------------------
def potions(t: ContentTables) -> dict[str, float]:
    # Reads the effect list each table holds. 0.5.10 applies a Potion_Stamina_Cooldown status (15 s) with the stamina
    # drink. 0.7.2 applies none: its drink cooldown (Potion_Consume_Stamina, 30 s) is a TriggerCooldown interaction,
    # which the tables do not hold. So stamina_cooldown_duration exists only where the table applies that effect.
    instant = t.effect("Potion_Health_Instant")
    regen = t.effect("Potion_Health_Regen")
    stamina = t.consumable("Potion_Stamina")
    stamina_effects = [t.effect(effect_id) for effect_id in stamina.effects_applied]
    restore = next(e for e in stamina_effects if "Stamina" in (e.document.get("StatModifiers") or {}))
    cooldown = next((e for e in stamina_effects if "Cooldown" in e.effect_id), None)
    antidote = t.effect("Antidote")
    out = {
        "health_instant_percent": float(instant.document["StatModifiers"]["Health"]),
        "health_regen_percent": float(regen.document["StatModifiers"]["Health"]),
        "health_regen_duration": regen.duration,
        "health_regen_cooldown": float(regen.document["DamageCalculatorCooldown"]),
        "stamina_restore_percent": float(restore.document["StatModifiers"]["Stamina"]),
        "antidote_duration": antidote.duration,
    }
    if cooldown is not None:
        out["stamina_cooldown_duration"] = cooldown.duration
    return out


# -- tests/fidelity native crafting recording format (jax/crafting packing) ------
# protocol BenchType ordinals; a requirement with no Type serializes as 0 in
# the native catalog evidence (Wood_Sticks' Farmingbench requirement).
_BENCH_TYPES = {None: 0, "Crafting": 0, "Processing": 1, "DiagramCrafting": 2, "StructuralCrafting": 3}


def native_crafting_recipes(t: ContentTables) -> dict[str, dict[str, Any]]:
    """Recipes in the native catalog recording's manifest shape (ids, I/O, benches)."""

    out = {}
    for recipe in t.recipes():
        out[recipe.recipe_id] = {
            "inputs": [(m.item_id or "", m.resource_type_id or "", m.quantity, tuple(sorted(m.excluded_item_ids))) for m in recipe.inputs],
            "outputs": [(m.item_id or "", m.quantity) for m in recipe.outputs],
            "primary_output_item_asset_id": recipe.primary_output.item_id if recipe.primary_output else "",
            "bench_requirements": [(b.bench_id, _BENCH_TYPES.get(b.type, -1), b.required_tier_level) for b in recipe.bench_requirements],
            "knowledge_required": recipe.knowledge_required,
            "required_memories_level": recipe.required_memories_level,
            "time_seconds": recipe.time_seconds,
        }
    return out


def crafting_capacities(t: ContentTables) -> dict[str, int]:
    """jax/crafting/contract.py fixed-shape capacities, from the recipe corpus."""

    recipes = t.recipes()
    return {
        "RECIPE_TABLE_CAPACITY": len(recipes),
        "MAX_INGREDIENTS": max(len(recipe.inputs) for recipe in recipes),
        "MAX_OUTPUTS": max(len(recipe.outputs) for recipe in recipes),
        "MAX_BENCH_REQUIREMENTS": max(len(recipe.bench_requirements) for recipe in recipes),
    }


@dataclass(frozen=True)
class SwapEntry:
    """One engine constant/table, where it lives, and the adapter replacing it.

    ``kind`` tells the renderer how the value is held: ``module_constant``
    (a top-level assignment), ``inline_literal`` (a literal inside a function
    body or call), ``function_default`` (a parameter default), ``json_table``
    (a packaged JSON resource).
    """

    target: str
    module_path: str  # relative to HytaleRL/hytalegym/hytalegym/
    symbol: str  # the text searched for to locate the line
    adapter: str
    kind: str = "module_constant"


SWAP_PLAN: tuple[SwapEntry, ...] = (
    *(
        SwapEntry(name, "jax/combat/types.py", name, "player_locomotion")
        for name in (
            "PLAYER_JUMP_FORCE",
            "PLAYER_VARIABLE_JUMP_FALL_FORCE",
            "PLAYER_BASE_SPEED",
            "PLAYER_GAIT_SPEED_MULTIPLIERS",
            "PLAYER_GAIT_FORWARD_SPEED_MULTIPLIERS",
            "PLAYER_GAIT_BACKWARD_SPEED_MULTIPLIERS",
            "PLAYER_GAIT_STRAFE_SPEED_MULTIPLIERS",
            "PLAYER_AIR_DRAG_MIN",
            "PLAYER_AIR_DRAG_MAX",
            "PLAYER_AIR_DRAG_MIN_SPEED",
            "PLAYER_AIR_DRAG_MAX_SPEED",
            "PLAYER_AIR_CONTROL_MIN_SPEED",
            "PLAYER_AIR_CONTROL_MAX_SPEED",
            "PLAYER_AIR_CONTROL_MIN_MULTIPLIER",
            "PLAYER_AIR_CONTROL_MAX_MULTIPLIER",
        )
    ),
    *(
        SwapEntry(name, "jax/combat/types.py", name, "player_stamina")
        for name in (
            "PLAYER_STAMINA_MAXIMUM",
            "PLAYER_STAMINA_MINIMUM",
            "PLAYER_SPRINT_STAMINA_DRAIN_PER_SECOND",
            "PLAYER_STAMINA_REGEN_PER_SECOND",
            "PLAYER_STAMINA_REGEN_DELAY_RECOVERY_PER_SECOND",
            "PLAYER_SPRINT_STAMINA_REGEN_DELAY_SECONDS",
            "PLAYER_BROKEN_STAMINA_REGEN_DELAY_SECONDS",
        )
    ),
    *(
        SwapEntry(name, "jax/combat/mechanics/schema/contract.py", name, "mechanics_resource_constants")
        for name in (
            "STAMINA_MINIMUM",
            "STAMINA_MAXIMUM",
            "STAMINA_REGEN_AMOUNT",
            "STAMINA_REGEN_INTERVAL_SECONDS",
            "STAMINA_REGEN_DELAY_AMOUNT",
            "STAMINA_REGEN_DELAY_INTERVAL_SECONDS",
            "STAMINA_BROKEN_REGEN_DELAY_SECONDS",
            "OXYGEN_MINIMUM",
            "OXYGEN_MAXIMUM",
            "OXYGEN_BREATHABLE_REGEN_AMOUNT",
            "OXYGEN_SUFFOCATING_REGEN_AMOUNT",
            "OXYGEN_REGEN_INTERVAL_SECONDS",
        )
    ),
    *(
        SwapEntry(name, "jax/combat/mechanics/schema/contract.py", name, "control_immunity")
        for name in ("CONTROL_IMMUNITY_MAXIMUM", "CONTROL_IMMUNITY_REGEN_AMOUNT", "CONTROL_IMMUNITY_REGEN_INTERVAL_SECONDS")
    ),
    *(
        SwapEntry(name, "jax/combat/mechanics/schema/contract.py", name, "dodge_constants")
        for name in (
            "DODGE_AUTHORED_ACTION_MASK",
            "DODGE_STAMINA_ADMISSION_COST",
            "DODGE_STAMINA_SPEND_COST",
            "DODGE_FORCE",
            "DODGE_INVULNERABILITY_SECONDS",
            "DODGE_REGEN_DELAY_SECONDS",
            "DODGE_AIR_RESISTANCE",
            "DODGE_AIR_RESISTANCE_MAX",
            "DODGE_GROUND_RESISTANCE",
            "DODGE_GROUND_RESISTANCE_MAX",
            "DODGE_RESISTANCE_THRESHOLD",
        )
    ),
    SwapEntry("mana_maximum default", "jax/combat/mechanics/factory.py", "mana_maximum: float", "mechanics_factory_resources", "function_default"),
    SwapEntry("mana regen 1.0 / 0.2 s", "jax/combat/mechanics/factory.py", "regen_interval.at[..., RESOURCE_MANA]", "mechanics_factory_resources", "inline_literal"),
    SwapEntry("magic charges regen 1.0 / 2.0 s", "jax/combat/mechanics/factory.py", "regen_interval.at[..., RESOURCE_MAGIC_CHARGES]", "mechanics_factory_resources", "inline_literal"),
    SwapEntry("installed_default_forward_multiplier", "jax/world/entities/locomotion.py", "1.273", "walk_steering_sprint_multiplier", "inline_literal"),
    SwapEntry("regeneration block", "rulesets/hytale_0_5_7/kweebec_razorleaf_vs_trork_brawler_v9.json", '"regeneration"', "health_regeneration", "json_table"),
    SwapEntry("agent/target explicit asset fields", "rulesets/hytale_0_5_7/kweebec_razorleaf_vs_trork_brawler_v9.json", '"max_health"', "npc_matchup", "json_table"),
    SwapEntry("weapon_items (143 rows)", "jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json", '"weapon_items"', "compiled_weapon_durability", "json_table"),
    SwapEntry("SCALAR_VARIANTS _ROWS", "combat/assets/weapon_variants/hytale_0_5_7.py", "_ROWS = (", "scalar_variant_values", "module_constant"),
    *(
        SwapEntry(f"_iron_{template} defaults", "jax/combat/arsenal/profiles/hytale_0_5_7.py", f"def _iron_{template}(", "iron_template_damage", "function_default")
        for template in IRON_TEMPLATE_ASSETS
    ),
    SwapEntry("sword melee force tuples", "jax/combat/arsenal/profiles/hytale_0_5_7.py", "push = (0.0, 1.0, -1.0, 6.0)", "sword_knockback", "inline_literal"),
    *(
        SwapEntry(constant, "jax/combat/arsenal/profiles/hytale_0_5_7.py", f"{constant} = ", "sword_selectors")
        for constant in SWORD_SELECTORS
    ),
    *(
        SwapEntry(constant, "jax/combat/arsenal/profiles/hytale_0_5_7.py", f"{constant} = ", "melee_hold_constants")
        for constant in ("_NESTED_OUTER_GATE_SECONDS", "_SWORD_THRUST_HOLD_SECONDS", "_DOWNSTRIKE_HOLD_SECONDS", "_MACE_CHARGED_HOLD_SECONDS")
    ),
    SwapEntry("shortbow draw ladder", "jax/combat/arsenal/profiles/hytale_0_5_7.py", "charge_thresholds = (0.1, 0.3, 0.6, 0.9, 1.2)", "shortbow_tiers", "inline_literal"),
    SwapEntry("standard spear partials", "jax/combat/arsenal/profiles/hytale_0_5_7.py", '"adamantite_spear": partial(', "spear_rows", "inline_literal"),
    SwapEntry("_potions values", "jax/combat/arsenal/profiles/hytale_0_5_7.py", "def _potions(", "potions", "inline_literal"),
    *(
        SwapEntry(name, "jax/combat/effects/schema/contract.py", name, "outlander_arrow")
        for name in (
            "OUTLANDER_ARROW_MODEL_HALF_EXTENT",
            "OUTLANDER_ARROW_MUZZLE_VELOCITY",
            "OUTLANDER_ARROW_TERMINAL_VELOCITY",
            "OUTLANDER_ARROW_GRAVITY",
            "OUTLANDER_ARROW_DAMAGE",
            "OUTLANDER_ARROW_DEAD_TIME_SECONDS",
            "OUTLANDER_ARROW_AUTHORED_LIFETIME_SECONDS",
            "OUTLANDER_ARROW_DEPTH_OFFSET",
            "OUTLANDER_ARROW_VERTICAL_OFFSET",
        )
    ),
    *(
        SwapEntry(name, "jax/crafting/contract.py", f"{name} = ", "crafting_capacities")
        for name in ("RECIPE_TABLE_CAPACITY", "MAX_INGREDIENTS", "MAX_OUTPUTS", "MAX_BENCH_REQUIREMENTS")
    ),
    SwapEntry("native crafting catalog (1947 recipes)", "../tests/fidelity/fixtures/world/native_crafting_catalog_recording.json", '"recipes"', "native_crafting_recipes", "json_table"),
)


# Content-table fields each adapter reads (``family/row#field.path``; ``<..>``
# is a placeholder filled per row). The renderer's redirect list carries these.
ADAPTER_SOURCES: dict[str, tuple[str, ...]] = {
    "player_locomotion": (
        "movement/binding:player#movement_config",
        "movement/config:<movement_config>#fields.<MovementConfig key>",
    ),
    "player_stamina": (
        "stats/stat:Stamina#min,max,regenerating[0],regenerating[Sprinting],min_value_effects",
        "stats/stat:StaminaRegenDelay#regenerating[0]",
        "stats/plugin:stamina#sprint_regen_delay_value",
    ),
    "mechanics_resource_constants": (
        "stats/stat:Stamina#min,max,regenerating[0],min_value_effects",
        "stats/stat:StaminaRegenDelay#regenerating[0]",
        "stats/stat:Oxygen#min,max,regenerating[Suffocating],regenerating[!Suffocating]",
    ),
    "control_immunity": ("stats/stat:Immunity#max,regenerating[0]",),
    "dodge_constants": (
        "weapons/interaction:Dodge#document.Next",
        "weapons/interaction:Dodge_Left#document.Costs,document.Next",
        "weapons/effect:Dodge_Invulnerability#duration",
    ),
    "mechanics_factory_resources": (
        "stats/stat:Mana#max,regenerating[0]",
        "stats/stat:MagicCharges#regenerating[0]",
    ),
    "walk_steering_sprint_multiplier": ("movement/config:Default#fields.ForwardSprintSpeedMultiplier",),
    "health_regeneration": ("stats/stat:Health#regenerating[IsPlayer inverse]",),
    "npc_matchup": (
        "npc_roles/role:Kweebec_Razorleaf#max_health,knockback_scale,appearance,motion_controllers,parameters",
        "npc_roles/role:Trork_Brawler#max_health,motion_controllers,parameters,fields.InteractionVars",
        "npc_roles/model:<appearance>#eye_height,max_scale,hitbox",
    ),
    "compiled_weapon_durability": ("durability/item:<item>#max_durability,durability_loss_on_hit,source",),
    "scalar_variant_values": (
        "weapons/item:<asset>#guard_stamina_value,max_durability,durability_loss_on_hit,damage.<var>.base_first,damage.<var>.angled_first,damage.<var>.origin",
    ),
    "iron_template_damage": ("weapons/item:Weapon_<Template>_Iron#damage.<var>.base_first,guard_stamina_value",),
    "sword_knockback": ("weapons/interaction:<SWORD_ATTACKS value>#knockback.direction,knockback.force,knockback.velocity_type",),
    "sword_selectors": ("weapons/interaction:<SWORD_SELECTORS value>#selector,document.RunTime",),
    "melee_hold_constants": (
        "weapons/interaction:Weapon_Sword_Primary#charge_thresholds",
        "weapons/interaction:Weapon_Sword_Primary_Thrust#charge_thresholds,document.HorizontalSpeedMultiplier",
        "weapons/interaction:Weapon_Battleaxe_Primary_Downstrike#charge_thresholds",
        "weapons/interaction:Weapon_Mace_Primary_Swing_Left_Charged#charge_thresholds",
    ),
    "sword_thrust_timing": (
        "weapons/interaction:Weapon_Sword_Primary_Thrust#document.Next (the highest threshold's Simple RunTime)",
    ),
    "shortbow_tiers": (
        "weapons/interaction:Weapon_Shortbow_Primary_Shoot_Charge#charge_thresholds",
        "weapons/interaction:Weapon_Shortbow_Signature_Volley_Charge#charge_thresholds",
        "projectiles/config:Projectile_Config_Arrow_Shortbow_Strength_<n>#launch_force,gravity,terminal_velocity_air",
        "weapons/item:Weapon_Shortbow_Iron#damage.<var>.base_first",
    ),
    "spear_rows": (
        "weapons/item:<asset>#damage.Spear_Stab_Damage.base_first,var_projectile_ids.Spear_Throw_Charged_Projectile,max_stack",
        "projectiles/projectile:<projectile>#damage",
    ),
    "outlander_arrow": (
        "projectiles/projectile:Outlander_Hunter_Arrow#muzzle_velocity,terminal_velocity,gravity,damage,dead_time,time_to_live,depth_shot,vertical_center_shot,appearance",
        "projectiles/model:<appearance>#hitbox.max[0]",
    ),
    "potions": (
        "consumables/effect:Potion_Health_Instant#document.StatModifiers.Health",
        "consumables/effect:Potion_Health_Regen#duration,document.StatModifiers.Health,document.DamageCalculatorCooldown",
        "consumables/item:Potion_Stamina#effects_applied",
        "consumables/effect:Antidote#duration",
    ),
    "crafting_capacities": ("crafting/recipe:*#inputs,outputs,bench_requirements (count and maxima)",),
    "native_crafting_recipes": ("crafting/recipe:<id>#inputs,outputs,primary_output,bench_requirements,time_seconds,knowledge_required,required_memories_level",),
}
