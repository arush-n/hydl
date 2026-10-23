"""Magic casts: the mana each 0.7.2 magic item charges, and whether a low pool refuses it (gen 7, row 7c).

Switch ``Constructs.magic_costs``: gen 5 keeps the native NPC rows (cost x2, no refusal); gen 7 charges each cast once.

0.7.2 authored values (``Assets.zip``, paths under ``Server/Item/``; read 2026-10-23, FIDELITY row 7c):
* Staff: ``Interactions/Weapons/Staff/Attacks/Staff_Cast_Summon_Charged.json`` runs ``Staff_Cast_Summon_Cost`` (default
  ``Staff_Cast_Cost``, ``ChangeStat`` Mana -25) once at RunTime 0.167. Each staff ``Items/Weapon/Staff/*.json`` overrides
  ``InteractionVars/Staff_Cast_Summon_Cost`` to -50 (Crystal_Red and Crystal_Ice to 0). No ``StatsCondition``: never refused.
* Wand: ``Interactions/Weapons/Wand/Attacks/Wand_Cast_Left_Charged.json`` is a ``StatsCondition`` with ``Costs`` Mana 25
  (the refusal gate) and then ``Wand_Cast_Cost`` (ChangeStat -25) at 0.167. ``StatsConditionInteraction.firstRun`` only checks
  (javap of HytaleServer.jar): the cost is taken once.
* Spellbook: ``Interactions/Weapons/Spellbook/Attacks/Spellbook_Cast_Hurl_Charged.json`` gates on ``Costs`` (25 by default,
  100 by ``InteractionVars/Spellbook_Cast_Hurl_Charged/Costs``) and charges ``Spellbook_Cast_Hurl_Cost`` (-100).
* Zero mana: Crystal_Flame (MagicCharges and DeployablePreview only), Crystal_Ice (Stamina 0.1 gate), Stoneskin and Root wands.

``skeleton_mage_spellbook`` compiles the NPC root ``Server/Item/Interactions/NPCs/Undead/Skeleton_Sand_Mage/
Skeleton_Sand_Mage_Spellbook_Corruption_Orb.json`` (profiles/hytale_0_5_7.py:3367-3373), which has no Mana, Cost,
StatModifiers or ChangeStat: cost 0 and no gate on gen 7, as on gen 5. The legacy ``purple_grimoire`` profile is the
player grimoire (Hurl 100 with the gate) and keeps its cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any

import jax.numpy as jnp

SWITCH = "magic_costs"
IMPLEMENTED = True

#: Commit time of every magic cost (0.167 s in the Staff, Wand and Spellbook casts' Parallel).
COMMIT_SECONDS = 0.167


@dataclass(frozen=True)
class MagicRow:
    item: str  # 0.7.2 item asset (Server/Item/Items/...)
    mana: float  # authored mana cost (0 = no mana cost)
    gate: bool  # True: a StatsCondition refuses the cast below the cost (wands, spellbooks)


_STAFF = (50.0, False)
_WAND = (25.0, True)
_SPELLBOOK = (100.0, True)

#: Profile name -> row. The 31 legacy caster rows and the five named profiles of the 36 magic loadouts.
ROWS: dict[str, MagicRow] = {
    "halloween_broomstick": MagicRow("Halloween_Broomstick", *_STAFF),
    "adamantite_staff": MagicRow("Weapon_Staff_Adamantite", *_STAFF),
    "bamboo_bo_staff": MagicRow("Weapon_Staff_Bo_Bamboo", *_STAFF),
    "wood_bo_staff": MagicRow("Weapon_Staff_Bo_Wood", *_STAFF),
    "bone_staff": MagicRow("Weapon_Staff_Bone", *_STAFF),
    "bronze_staff": MagicRow("Weapon_Staff_Bronze", *_STAFF),
    "cane_staff": MagicRow("Weapon_Staff_Cane", *_STAFF),
    "cobalt_staff": MagicRow("Weapon_Staff_Cobalt", *_STAFF),
    "copper_staff": MagicRow("Weapon_Staff_Copper", *_STAFF),
    "crystal_fire_trork_staff": MagicRow("Weapon_Staff_Crystal_Fire_Trork", *_STAFF),
    "crystal_purple_staff": MagicRow("Weapon_Staff_Crystal_Purple", *_STAFF),
    "crystal_red_staff": MagicRow("Weapon_Staff_Crystal_Red", 0.0, False),
    "doomed_staff": MagicRow("Weapon_Staff_Doomed", *_STAFF),
    "frost_staff": MagicRow("Weapon_Staff_Frost", *_STAFF),
    "iron_staff": MagicRow("Weapon_Staff_Iron", *_STAFF),
    "mithril_staff": MagicRow("Weapon_Staff_Mithril", *_STAFF),
    "onion_staff": MagicRow("Weapon_Staff_Onion", *_STAFF),
    "onyxium_staff": MagicRow("Weapon_Staff_Onyxium", *_STAFF),
    "thorium_staff": MagicRow("Weapon_Staff_Thorium", *_STAFF),
    "wizard_staff": MagicRow("Weapon_Staff_Wizard", *_STAFF),
    "wood_staff": MagicRow("Weapon_Staff_Wood", *_STAFF),
    "kweebec_wood_staff": MagicRow("Weapon_Staff_Wood_Kweebec", *_STAFF),
    "rotten_wood_staff": MagicRow("Weapon_Staff_Wood_Rotten", *_STAFF),
    "demon_spellbook": MagicRow("Weapon_Spellbook_Demon", *_SPELLBOOK),
    "fire_spellbook": MagicRow("Weapon_Spellbook_Fire", *_SPELLBOOK),
    "frost_spellbook": MagicRow("Weapon_Spellbook_Frost", *_SPELLBOOK),
    "brown_grimoire": MagicRow("Weapon_Spellbook_Grimoire_Brown", *_SPELLBOOK),
    "purple_grimoire": MagicRow("Weapon_Spellbook_Grimoire_Purple", *_SPELLBOOK),
    "tribal_wand": MagicRow("Weapon_Wand_Tribal", *_WAND),
    "wood_wand": MagicRow("Weapon_Wand_Wood", *_WAND),
    "rotten_wood_wand": MagicRow("Weapon_Wand_Wood_Rotten", *_WAND),
    # The NPC root Skeleton_Sand_Mage_Spellbook_Corruption_Orb has no Mana, Cost, StatModifiers or ChangeStat.
    "skeleton_mage_spellbook": MagicRow("Skeleton_Sand_Mage_Spellbook_Corruption_Orb", 0.0, False),
    "flame_staff": MagicRow("Weapon_Staff_Crystal_Flame", 0.0, False),
    "ice_staff": MagicRow("Weapon_Staff_Crystal_Ice", 0.0, False),
    "stoneskin_wand": MagicRow("Weapon_Wand_Stoneskin", 0.0, False),
    "root_wand": MagicRow("Weapon_Wand_Root", 0.0, False),
}


def construct_entities(original: Any, profile: Any) -> Any:
    """Wrap ``hytale_0_5_7_entity_loadouts``: for magic entities, one 0.7.2 mana charge on the cast ability.

    The cast ability is the last authored ability of each magic profile (the staff, wand and spellbook casts are appended
    after the melee or the single attack). The cost kind becomes ``SINGLE_APPLICATION`` (x1, not the gen-5 NPC x2); a
    wand or spellbook gets ``ability_resource_minimum`` = cost, which refuses the cast below it (``ability.py`` admission).
    """

    if not profile.constructs.magic_costs:
        return original

    env = original.__globals__
    mana = env["RESOURCE_MANA"]
    single = env["RESOURCE_COST_SINGLE_APPLICATION"]
    parallel = env["EVENT_FLAG_PARALLEL_FORK"]

    @wraps(original)
    def hytale_0_5_7_entity_loadouts(entity_profiles: Any) -> Any:
        loadout = original(entity_profiles)
        catalog = env["_CATALOG"]  # defined after this hook is applied, so read it per call
        edits = [
            (i, j, name)
            for i, row in enumerate(entity_profiles)
            for j, name in enumerate(row)
            if name in ROWS and ROWS[name].mana > 0.0
        ]
        if not edits:
            return loadout
        cost = loadout.ability_resource_cost
        kind = loadout.ability_resource_cost_kind
        commit = loadout.ability_resource_commit_time_seconds
        flags = loadout.ability_resource_commit_flags
        minimum = loadout.ability_resource_minimum
        for i, j, name in edits:
            row = ROWS[name]
            ability = len(catalog[name]()["abilities"]) - 1
            index = (i, j, ability, mana)
            cost = cost.at[index].set(jnp.float32(row.mana))
            kind = kind.at[index].set(single)
            commit = commit.at[index].set(jnp.float32(COMMIT_SECONDS))
            flags = flags.at[index].set(parallel)
            minimum = minimum.at[index].set(jnp.float32(row.mana if row.gate else 0.0))
        return loadout._replace(
            ability_resource_cost=cost,
            ability_resource_cost_kind=kind,
            ability_resource_commit_time_seconds=commit,
            ability_resource_commit_flags=flags,
            ability_resource_minimum=minimum,
        )

    return hytale_0_5_7_entity_loadouts
