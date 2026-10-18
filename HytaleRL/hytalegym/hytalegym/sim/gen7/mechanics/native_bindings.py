"""Bridge binding types from the 0.7.2 item's ``Interactions`` map. Switch ``Constructs.native_binding_types``; row 6c.

The bridge requests a policy ability slot by its binding's root id and interaction type name, and the native
evidence advertises a slot only while the bridge reports the same name (``observation/v3/native/evidence.py``,
the ``root_bound`` term). 0.7.2 binds the crossbows' reload root ``Root_Common_StatAmmoReload_Entry`` under
``Interactions.Ability4`` (``weapons/item:Weapon_Crossbow_Iron`` and ``item:Weapon_Crossbow_Ancient_Steel``),
where 0.5.7 bound it under ``Ability3``; the bridge accepts ``Ability4`` by name since 9f12f0d (jar 32EF5EDD,
FIDELITY-0.7.2.md (m)).

Gen 7 (``item_interactions``) wraps ``hytale_0_5_7_native_profile_bindings``: a binding whose root id its item's
0.7.2 ``Interactions`` map binds under exactly one other type takes that type's name. Audit over every arsenal
profile (2026-10-18): only the two crossbow reload bindings change (compact slot 2, authored slot 4); the other
90 bindings whose root the map names agree. The guard binding and the engine are unchanged: the reload keeps its
compiled ordinal ``INTERACTION_TYPE_ABILITY3``, which no ability's mask references, so the change is the name
the bridge and the evidence see. Gen 5 (``profile_sets``) returns the original.
"""

from __future__ import annotations

from functools import lru_cache, wraps
from typing import Any

SWITCH = "native_binding_types"
IMPLEMENTED = True
PROFILE_SETS = "profile_sets"
ITEM_INTERACTIONS = "item_interactions"


def _enabled(profile: Any) -> bool:
    value = profile.constructs.native_binding_types
    if value == PROFILE_SETS:
        return False
    if value != ITEM_INTERACTIONS:
        raise ValueError(f"unknown native_binding_types switch {value!r}")
    return True


def item_binding_types(tables: Any, item_id: str) -> dict[str, tuple[str, ...]]:
    """Root id -> the interaction types the item's 0.7.2 ``Interactions`` map binds it under."""

    try:
        interactions = tables.row("weapons", "item:" + item_id).get("interactions") or {}
    except KeyError:
        return {}
    by_root: dict[str, list[str]] = {}
    for interaction_type, root in sorted(interactions.items()):
        by_root.setdefault(root, []).append(interaction_type)
    return {root: tuple(types) for root, types in by_root.items()}


def retyped_bindings(original: Any, tables: Any) -> Any:
    """``original`` with each binding typed by its item's 0.7.2 map where that map names one other type."""

    @lru_cache(maxsize=None)
    @wraps(original)
    def hytale_0_5_7_native_profile_bindings(profile: str) -> Any:
        bindings = original(profile)
        if not bindings.item_id:
            return bindings
        types = item_binding_types(tables, bindings.item_id)
        abilities = tuple(
            binding._replace(interaction_type=named[0])
            if len(named := types.get(binding.interaction_id, ())) == 1 and named[0] != binding.interaction_type
            else binding
            for binding in bindings.abilities
        )
        return bindings if abilities == bindings.abilities else bindings._replace(abilities=abilities)

    return hytale_0_5_7_native_profile_bindings


def construct(original: Any, profile: Any) -> Any:
    """Hook for ``hytale_0_5_7_native_profile_bindings``."""

    if not _enabled(profile):
        return original
    from hytalegym.sim.common.content import load

    return retyped_bindings(original, load(profile))
