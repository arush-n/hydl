"""0.7.2 opponent data read: the opponents the sim models carry the values the 0.7.2 role table states.

The sim reads its combat numbers from the 0.5.7 combat ruleset (``rulesets.load_combat_ruleset``, consumed by
``jax/combat/types.py``). These tests pin that each 0.7.2 ``npc_roles`` row the sim models (the Trork matchup
target, the Kweebec agent) agrees with the ruleset value the sim uses today. A failure means a gen-7 opponent
would run on 0.5.7 numbers for that field.
"""

from __future__ import annotations

from hytalegym.rulesets import load_combat_ruleset
from hytalegym.sim.common.content import load


def _role(name: str) -> dict:
    return load("0.7.2").table("npc_roles").rows[f"role:{name}"]


def _melee_damage(role: dict) -> dict:
    variable = role["fields"]["InteractionVars"]["Melee_Damage"]
    return variable["Interactions"][0]["DamageCalculator"]["BaseDamage"]


def test_trork_matchup_target_matches_the_072_brawler_role() -> None:
    ruleset = load_combat_ruleset()["target"]
    role = _role("Trork_Brawler")

    assert role["max_health"] == ruleset["max_health"] == 61
    assert role["knockback_scale"] == 0.5
    assert role["motion_controllers"][0]["MaxWalkSpeed"] == ruleset["asset_max_walk_speed"] == 8.0
    assert role["motion_controllers"][0]["Acceleration"] == ruleset["acceleration"] == 10
    # The 0.7.2 role supplies its own melee base damage (InteractionVars.Melee_Damage); the battleaxe item does not.
    assert _melee_damage(role) == {"Physical": 23}
    assert {attack["damage"] for attack in ruleset["attacks"]} == {23.0}


def test_kweebec_agent_matches_the_072_razorleaf_role() -> None:
    ruleset = load_combat_ruleset()["agent"]
    role = _role("Kweebec_Razorleaf")

    assert role["max_health"] == ruleset["max_health"] == 105
    assert role["knockback_scale"] == ruleset["knockback_scale"] == 0.5
    assert role["motion_controllers"][0]["MaxWalkSpeed"] == ruleset["asset_max_walk_speed"] == 5.0
    # The 0.7.2 Kweebec role has no Melee_Damage variable; its 5.0 damage is not read from the role table.
    assert "Melee_Damage" not in role["fields"].get("InteractionVars", {})
    assert ruleset["damage"] == 5.0
