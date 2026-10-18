"""Mana pool and rune casting (0.7.2). Switches ``mana_pool``, ``rune_casting``; P3.

Inputs (gen-7 tables): ``stats/stat:Mana`` (initial 100, max 100, +1 per
0.3 s while Alive; gen 5: 0/0, +1 per 0.2 s gated on NoDamageTaken 6 s and not
Charging); rune items ``weapons/item:Rune_*`` with an ``ability`` block
(``Cost``, ``CostType`` Mana, ``Cooldown``, ``Slot``, ``AbilityKeywords``,
``Cast`` root), e.g. Rune_Fireball cost 30, cooldown 2; cast graphs
``weapons/interaction:Ability_*_Cast`` (StatsCondition on the ability cost,
Simple RunTime 0.87 for the fireball, ``LaunchAbilityProjectile``),
``weapons/projectile_config:Projectile_Config_Ability_*`` and hit interactions.
Modifier runes (Aoe_Up, Damage_Up, Fork, ...) need ``Server/Abilities`` keyword
semantics, not extracted yet. Differential: mana refill rate (+1 per 0.3 s,
within 5 %), per rune cost, cooldown, cast time, flight and effect
(FIDELITY-0.7.2.md rows 7a, 7b).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SWITCHES = ("rune_casting",)
IMPLEMENTED = False


@dataclass(frozen=True)
class RuneAbility:
    item_id: str
    slot: str
    cost: float
    cost_type: str
    cooldown_seconds: float
    cast_root: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class ManaParams:
    initial: float
    maximum: float
    regen_amount: float
    regen_interval_seconds: float
    runes: tuple[RuneAbility, ...]


def params_from_tables(tables: Any) -> ManaParams:
    raise NotImplementedError("mana/rune construct: design only (mechanics_072.md)")


def cast(state: Any, rune_slot: Any, params: ManaParams) -> Any:
    """Engine hook: admit a cast (cost, cooldown), then run its cast graph."""

    raise NotImplementedError("mana/rune construct: design only (mechanics_072.md)")
