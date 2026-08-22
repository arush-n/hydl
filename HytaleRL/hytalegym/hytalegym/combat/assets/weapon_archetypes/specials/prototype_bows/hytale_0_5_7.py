"""Pinned Hytale 0.5.7 prototype-bow graph identities.

These five items are not material skins of the production Shortbow. They
share portions of an older charged-shot graph, then diverge into authored
trishot, blast, pull, ricochet, and life-steal programs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrototypeBowSpec:
    """One equipped item and the two graph families it selects."""

    profile: str
    asset_id: str
    family_id: int
    primary: str
    signature: str


PROTOTYPE_BOW_SPECS = (
    PrototypeBowSpec(
        "combat_shortbow",
        "Weapon_Shortbow_Combat",
        214,
        "combat",
        "trishot",
    ),
    PrototypeBowSpec(
        "bomb_shortbow",
        "Weapon_Shortbow_Bomb",
        215,
        "combat",
        "bomb",
    ),
    PrototypeBowSpec(
        "pull_shortbow",
        "Weapon_Shortbow_Pull",
        216,
        "combat",
        "pull",
    ),
    PrototypeBowSpec(
        "ricochet_shortbow",
        "Weapon_Shortbow_Ricochet",
        217,
        "ricochet",
        "ricochet",
    ),
    PrototypeBowSpec(
        "vampire_shortbow",
        "Weapon_Shortbow_Vampire",
        218,
        "vampire",
        "vampire",
    ),
)
PROTOTYPE_BOW_PROFILE_NAMES = tuple(row.profile for row in PROTOTYPE_BOW_SPECS)

assert len(PROTOTYPE_BOW_SPECS) == 5
assert {row.family_id for row in PROTOTYPE_BOW_SPECS} == set(range(214, 219))
assert len(set(PROTOTYPE_BOW_PROFILE_NAMES)) == len(PROTOTYPE_BOW_SPECS)


__all__ = [
    "PROTOTYPE_BOW_PROFILE_NAMES",
    "PROTOTYPE_BOW_SPECS",
    "PrototypeBowSpec",
]
