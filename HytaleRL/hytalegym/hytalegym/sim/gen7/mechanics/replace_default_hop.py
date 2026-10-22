"""Replace default hop (gen 7, row 5f): a melee selector of a Simple -> Replace -> Selector attack starts one tick later.

Switch ``Constructs.replace_default_hop``: ``False`` (gen 5: the selector runs at its authored time) or ``True`` (gen 7: the
selector event of each attack in ``REPLACE_DEFAULT_ATTACKS`` moves one tick, 1/30 s, later).

Measured live (0.7.2, receipt ``hit_interrupt_Skeleton_Fighter_seed570057_swings6_s570057_p20.json.gz``, the iron sword left
swing on env tick 1; FIDELITY-0.7.2.md section (ao), row 5f): the claim on tick 1, the chain start on tick 2, the Swing_Left Simple
(RunTime 0.111 s) finishes on tick 6 with carried time 0.02233 s (4/30 - 0.111), the Replace node's default-value Selector is
entered on tick 7 (``native_combat_interaction_id`` ``**Weapon_Sword_Primary_Swing_Left_Next_DefaultValue``, progress 0), the
selector hit count is 1 on tick 8 and 2 on tick 9, the target health drops 70 -> 60 on tick 9, and the chain finishes on tick 14.
The gen-7 render before this construct hits on tick 8 (``probe_sword`` in lane 119-F4) with the same chain start 2 and finish 14.

Asset shape (``sim/gen7/data/weapons.json``): each attack row is a Simple whose ``Next`` is a Replace (``Var`` Swing_*_Selector,
``DefaultValue`` the attack's ``_Selector`` row); the 13 rows in ``REPLACE_DEFAULT_ATTACKS`` are the iron sword swings, the dagger
swings and stabs, the mace swings (not the charged ones) and the battleaxe swings, and ``tests/sim/test_replace_default_hop.py``
checks each row's shape against the table.

Mechanism: the engine models the chain as one melee event at its authored time (``profiles/hytale_0_5_7.py`` ``_melee``, sword
0.117 s), whose progressive selector opens when the scheduler clock passes that time and closes at time + RunTime. Moving the
event by one tick moves the whole selector window and its sweep by one tick, the same as entering the Replace's default chain one
tick later. The ability duration (the chain finish) is unchanged; the charge spend and recharge are unchanged.

Hook: ``_write_profile`` (profiles/hytale_0_5_7.py), the writer of each attack's events into the loadout arrays (its
``event_time_seconds`` is the only time the engine reads). The construct copies the profile with the listed attacks' selector
events one tick later and calls the original. (``_ability`` is not used: the guard hook already wraps it.) Gen 5 returns the
original function.

Rendered measurements (gen 7 render, seed 570057, one still target, one request on step 1; 1-based steps): the table in
``tests/sim/test_replace_default_hop.py`` and ``mechanics_072.md`` row 5f. Only the iron sword swing is measured live (hit 9,
finish 14). The mace, battleaxe and dagger shifts follow from the same shape (by mechanism, not measured live). The ledger's
bronze-dagger 17 -> 18 (row 5f) is a 0.5.7-pinned run that this scenario does not reproduce (stock gives 7 for the iron daggers).

Routing: a request for the thrust or the charged mace runs the listed Swing_Left (slot 0, the Primary root releases into it on
a short press); the signature attacks do not start without energy. Those lanes are classified by the attack they run.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

from hytalegym.jax.combat.arsenal.schema.contract import EVENT_FLAG_SERVER_SELECTOR

SWITCH = "replace_default_hop"
IMPLEMENTED = True
#: One tick at the gen-7 nominal step (``SimProfile.step_duration``, 1/30 s).
TICK_SECONDS = 1.0 / 30.0
#: The Simple -> Replace -> Selector primary attacks of the iron melee weapons (weapons.json), by their JAX asset name.
#: Charged holds, thrusts, vortexes, whirlwinds, the dagger razorstrike and the dodge/guard rows keep their authored times.
REPLACE_DEFAULT_ATTACKS = frozenset({
    "Weapon_Sword_Primary_Swing_Left",
    "Weapon_Sword_Primary_Swing_Right",
    "Weapon_Sword_Primary_Swing_Down",
    "Weapon_Daggers_Primary_Swing_Left",
    "Weapon_Daggers_Primary_Swing_Right",
    "Weapon_Daggers_Primary_Stab_Left",
    "Weapon_Daggers_Primary_Stab_Right",
    "Weapon_Mace_Primary_Swing_Left",
    "Weapon_Mace_Primary_Swing_Right",
    "Weapon_Mace_Primary_Swing_Up_Left",
    "Weapon_Battleaxe_Primary_Swing_Down",
    "Weapon_Battleaxe_Primary_Swing_Down_Left",
    "Weapon_Battleaxe_Primary_Swing_Down_Right",
})


def _enabled(profile: Any) -> bool:
    return bool(profile.constructs.replace_default_hop)


def _hop(event: dict) -> dict:
    """The selector event one tick later; a non-selector or instant event keeps its time."""

    if not event["flags"] & EVENT_FLAG_SERVER_SELECTOR or event["time"] <= 0.0:
        return event
    return {**event, "time": event["time"] + TICK_SECONDS}


def _hop_attack(ability: dict) -> dict:
    if ability["asset"] not in REPLACE_DEFAULT_ATTACKS:
        return ability
    return {**ability, "events": tuple(_hop(event) for event in ability["events"])}


def construct(original: Any, profile: Any) -> Any:
    """Return ``original`` unless the switch is on; for gen 7 the writer that hops the listed attacks' selector events."""

    if not _enabled(profile):
        return original

    @wraps(original)
    def _write_profile(arrays: Any, batch: int, entity: int, authored: dict) -> Any:
        hopped = {**authored, "abilities": [_hop_attack(ability) for ability in authored["abilities"]]}
        return original(arrays, batch, entity, hopped)

    return _write_profile
