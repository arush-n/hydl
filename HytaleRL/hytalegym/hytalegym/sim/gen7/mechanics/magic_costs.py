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
* Zero mana: Crystal_Flame (MagicCharges and DeployablePreview only), Crystal_Ice (its Summon cost is 0; the cast below
  carries the stamina), Stoneskin and Root wands.

0.7.2 authored stamina (same asset tree, read 2026-10-23; FIDELITY row 7c, stamina part):
* Staff cast: ``Staff_Cast_Summon_Charged.json`` ``Replace`` ``Staff_Cast_Summon_StaminaCost`` with the default ``ChangeStat``
  Stamina -5 at RunTime 0.167 (Parallel). No staff item in ``Items/Weapon/Staff/*.json`` overrides it except Crystal_Ice.
  No ``StatsCondition``: never refused. Applies to every staff row below except Crystal_Flame and Crystal_Ice.
* Crystal_Ice, NOT APPLIED (the sim keeps its current gen-5 stamina rows): its ``InteractionVars/Staff_Cast_Summon_StaminaCost``
  is zeroed, and its ``Staff_Cast_Summon_Launch`` override carries ``ChangeStat`` Stamina -5 at 0.167. The cast is reached
  through the Primary root ``Ice_Staff_Primary_Wrapper.json`` (``StatsCondition`` Stamina 0.1, then ``Staff_Primary`` ->
  Charging -> index 1 ``Staff_Cast_Summon_Charged``). The bolt ``Ice/Ice_Staff_Primary_Entry.json`` is a ``StatsCondition``
  Stamina 1.51, then ``ChangeStat`` Stamina -1.5 at 0. Not applied: the authored ice chain needs ``Ingredient_Ice_Essence``
  (ModifyInventory), which the sim does not model; decided separately.
  The legacy ``Staff_Primary`` has no Stamina gate.
* Wands, spellbooks, Crystal_Flame, Stoneskin and Root: no Stamina in their item or interaction JSON (scan of ``Assets.zip``).

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

#: Stamina per profile: ((ability asset or None for the cast ability, authored cost, authored gate, commit seconds), ...).
#: Authored in the module docstring (0.7.2 staff cast default -5 at 0.167). Crystal_Ice is not listed; see the comment below.
_STAFF_CAST = (None, 5.0, 0.0, COMMIT_SECONDS)
STAMINA: dict[str, tuple[tuple[str | None, float, float, float], ...]] = {
    name: (_STAFF_CAST,)
    for name, row in ROWS.items()
    if (row.item.startswith("Weapon_Staff_") or row.item == "Halloween_Broomstick")
    and row.item not in ("Weapon_Staff_Crystal_Flame", "Weapon_Staff_Crystal_Ice")
}
# ice_staff is deliberately absent: not applied. The authored ice chain needs Ingredient_Ice_Essence (ModifyInventory), which
# the sim does not model; decided separately. Its stamina stays at the current sim rows (gen-5 values).


def construct_entities(original: Any, profile: Any) -> Any:
    """Wrap ``hytale_0_5_7_entity_loadouts``: for magic entities, one 0.7.2 mana and stamina charge per cast.

    The cast ability is the last authored ability of each magic profile (the staff, wand and spellbook casts are appended
    after the melee or the single attack). The cost kind becomes ``SINGLE_APPLICATION``
    (x1, not the gen-5 NPC x2); a wand or spellbook gets ``ability_resource_minimum`` = mana cost, which refuses the cast
    below it (``ability.py`` admission). A stamina row sets its authored gate (0 where the cast has none).
    """

    if not profile.constructs.magic_costs:
        return original

    env = original.__globals__
    mana = env["RESOURCE_MANA"]
    stamina = env["RESOURCE_STAMINA"]
    single = env["RESOURCE_COST_SINGLE_APPLICATION"]
    parallel = env["EVENT_FLAG_PARALLEL_FORK"]

    @wraps(original)
    def hytale_0_5_7_entity_loadouts(entity_profiles: Any) -> Any:
        loadout = original(entity_profiles)
        catalog = env["_CATALOG"]  # defined after this hook is applied, so read it per call
        mana_edits = [
            (i, j, name)
            for i, row in enumerate(entity_profiles)
            for j, name in enumerate(row)
            if name in ROWS and ROWS[name].mana > 0.0
        ]
        stamina_edits = [
            (i, j, name)
            for i, row in enumerate(entity_profiles)
            for j, name in enumerate(row)
            if name in STAMINA
        ]
        if not mana_edits and not stamina_edits:
            return loadout
        cost = loadout.ability_resource_cost
        kind = loadout.ability_resource_cost_kind
        commit = loadout.ability_resource_commit_time_seconds
        flags = loadout.ability_resource_commit_flags
        minimum = loadout.ability_resource_minimum
        for i, j, name in mana_edits:
            row = ROWS[name]
            ability = len(catalog[name]()["abilities"]) - 1
            index = (i, j, ability, mana)
            cost = cost.at[index].set(jnp.float32(row.mana))
            kind = kind.at[index].set(single)
            commit = commit.at[index].set(jnp.float32(COMMIT_SECONDS))
            flags = flags.at[index].set(parallel)
            minimum = minimum.at[index].set(jnp.float32(row.mana if row.gate else 0.0))
        for i, j, name in stamina_edits:
            assets = [a.get("asset") for a in catalog[name]()["abilities"]]
            for asset, amount, gate, seconds in STAMINA[name]:
                ability = len(assets) - 1 if asset is None else assets.index(asset)
                index = (i, j, ability, stamina)
                cost = cost.at[index].set(jnp.float32(amount))
                kind = kind.at[index].set(single)
                commit = commit.at[index].set(jnp.float32(seconds))
                flags = flags.at[index].set(parallel)
                minimum = minimum.at[index].set(jnp.float32(gate))
        return loadout._replace(
            ability_resource_cost=cost,
            ability_resource_cost_kind=kind,
            ability_resource_commit_time_seconds=commit,
            ability_resource_commit_flags=flags,
            ability_resource_minimum=minimum,
        )

    return hytale_0_5_7_entity_loadouts
