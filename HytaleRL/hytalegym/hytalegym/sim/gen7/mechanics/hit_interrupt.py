"""Hit interrupt, stun and flinch (new in 0.7.2). Switch ``Constructs.hit_interrupt``; P3.

Inputs (gen-7 tables): ``weapons/interaction:*`` rows with ``hit_interrupt``
true (31 files: Weapons/Daggers 9, Mace 7, Sword 6, Battleaxe 5,
Abilities/Fireball, GroundSlam x2, WindStrike); the gate chain in
``weapons/interaction:DamageEntityParent_Hit_Interrupt_On_Hit`` (target has
``Hit_Interrupt_Eligible`` and none of Flinch/Enraged/Recovery_Pending ->
cancel attack, stun, flinch, record reaction); ``weapons/effect:Hit_Interrupt_*``
(Flinch 0.5 s at HorizontalSpeedMultiplier 0, Stunned 0.7 s, Enraged 8 s,
Recovery_Pending 2 s, Recent_Reaction_A/B 4 s, Eligible 1 s); eligibility from
``npc_roles/role:*`` ``component_references`` reaching
``Component_Instruction_Hit_Interrupt`` through ``reference_chain``.
Differential: live 0.7.2 hits on an eligible role (Skeleton_Fighter,
Goblin_Scrapper): cancel tick, flinch window and displacement, stun, enrage
after the third flinch (FIDELITY-0.7.2.md row 5a). Design: mechanics_072.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SWITCH = "hit_interrupt"
IMPLEMENTED = False


@dataclass(frozen=True)
class HitInterruptParams:
    flagged_interactions: frozenset[str]
    eligible_roles: frozenset[str]
    flinch_seconds: float
    flinch_speed_multiplier: float
    stun_seconds: float
    enrage_seconds: float
    recovery_pending_seconds: float
    recent_reaction_seconds: float


def params_from_tables(tables: Any) -> HitInterruptParams:
    raise NotImplementedError("hit interrupt construct: design only (mechanics_072.md)")


def apply_on_hit(state: Any, hits: Any, params: HitInterruptParams) -> Any:
    """Engine hook after damage resolution: cancel, stun, flinch, record."""

    raise NotImplementedError("hit interrupt construct: design only (mechanics_072.md)")
