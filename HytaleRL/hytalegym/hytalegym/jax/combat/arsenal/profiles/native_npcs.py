"""Asset-pinned JAX ability programs for native hostile NPC roles."""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple, Sequence

import jax
import numpy as np

from hytalegym.combat.assets.catalog import HYTALE_0_5_7_ASSETS_SHA256
from hytalegym.jax.combat.arsenal.factory import empty_ability_loadout
from hytalegym.jax.combat.arsenal.schema.contract import (
    FAMILY_BATTLEAXE,
    FAMILY_SPEAR_LEAF,
    FAMILY_SWORD,
    INTERACTION_TYPE_PRIMARY,
)
from hytalegym.jax.combat.mechanics import DAMAGE_CLASS_UNKNOWN

from .hytale_0_5_7 import (
    NativeInteractionBinding,
    NativeProfileBindings,
    _HorizontalSelectorProgram,
    _MutableProfileArray,
    _StabSelectorProgram,
    _ability,
    _melee,
    _weapon,
    _write_profile,
    ability_loadout_content_sha256,
)


NATIVE_NPC_ROLE_PROFILE_SCHEMA = "hytalerl_native_npc_role_program_v1"
NATIVE_NPC_ROLE_PROFILE_NAMES = (
    "Feran_Burrower",
    "Feran_Sharptooth",
    "Kweebec_Razorleaf",
    "Trork_Brawler",
    "Trork_Hunter",
    "Trork_Warrior",
)


class _MeleeSpec(NamedTuple):
    root: str
    windup: float
    duration: float
    half_angle: float
    selector: _HorizontalSelectorProgram | _StabSelectorProgram


_KWEBEC = (
    _MeleeSpec(
        "Kweebec_Sapling_Razorleaf_Spear_Swing_Left",
        0.3,
        0.667,
        45.0,
        _HorizontalSelectorProgram(0.2, 0.5, 3.0, 0.5, 0.5, 90.0, 1.0, -60.0, 0.0, 0.0),
    ),
    _MeleeSpec(
        "Kweebec_Sapling_Razorleaf_Spear_Swing_Right",
        0.223,
        0.557,
        45.0,
        _HorizontalSelectorProgram(
            0.111, 0.5, 3.0, 0.5, 0.5, 90.0, -1.0, -60.0, 0.0, 0.0
        ),
    ),
    _MeleeSpec(
        "Kweebec_Sapling_Razorleaf_Spear_Stab",
        0.5,
        1.0,
        18.0,
        _StabSelectorProgram(0.117, 0.5, 3.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0),
    ),
)


def _trork(root: str, direction: float, roll: float) -> _MeleeSpec:
    return _MeleeSpec(
        root,
        0.4,
        1.0,
        45.0,
        _HorizontalSelectorProgram(
            0.15, 0.1, 3.0, 0.5, 0.5, 90.0, direction, -45.0, 0.0, roll
        ),
    )


_TRORK = (
    _trork("Trork_Warrior_Battleaxe_Swing_Left", 1.0, 0.0),
    _trork("Trork_Warrior_Battleaxe_Swing_Right", -1.0, 0.0),
    _trork("Trork_Warrior_Battleaxe_Swing_Up_Left", 1.0, -50.0),
    _trork("Trork_Warrior_Battleaxe_Swing_Down", 1.0, 90.0),
    _trork("Trork_Warrior_Battleaxe_Swing_Up_Right", -1.0, 50.0),
)


def _sword(root: str, direction: float, roll: float) -> _MeleeSpec:
    return _MeleeSpec(
        root,
        0.167,
        0.417,
        30.0,
        _HorizontalSelectorProgram(
            0.083, 0.5, 2.5, 0.5, 0.5, 60.0, direction, -30.0, 0.0, roll
        ),
    )


_FERAN_SWORD = (
    _sword("Feran_Sharptooth_Sword_Swing_Left", 1.0, 30.0),
    _sword("Feran_Sharptooth_Sword_Swing_Right", -1.0, 30.0),
    _sword("Feran_Sharptooth_Sword_Swing_Down", 1.0, 75.0),
)
_TRORK_HUNTER_SWORD = (
    _sword("Trork_Hunter_Sword_Swing_Left", 1.0, 30.0),
    _sword("Trork_Hunter_Sword_Swing_Right", -1.0, 30.0),
    _sword("Trork_Hunter_Sword_Swing_Down", 1.0, 75.0),
)
_ROLE_SPECS = {
    "Feran_Burrower": (FAMILY_SWORD, 5.0, _FERAN_SWORD),
    "Feran_Sharptooth": (FAMILY_SWORD, 5.0, _FERAN_SWORD),
    "Kweebec_Razorleaf": (FAMILY_SPEAR_LEAF, 5.0, _KWEBEC),
    "Trork_Brawler": (FAMILY_BATTLEAXE, 23.0, _TRORK),
    "Trork_Hunter": (FAMILY_SWORD, 23.0, _TRORK_HUNTER_SWORD),
    "Trork_Warrior": (FAMILY_BATTLEAXE, 23.0, _TRORK),
}


def hytale_0_5_7_native_npc_role_loadouts(entity_roles: Sequence[Sequence[str]]):
    """Build fixed-shape role programs; an empty role leaves that entity unarmed."""

    rows = tuple(tuple(row) for row in entity_roles)
    if not rows or len(rows[0]) < 1 or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("native NPC role rows must form a nonempty rectangle")
    indices = np.asarray(
        [[_role_index(role) for role in row] for row in rows], dtype=np.int32
    )
    bank = _role_bank()
    return jax.device_put(type(bank)(*(np.asarray(value)[indices] for value in bank)))


def hytale_0_5_7_native_npc_role_bindings(role: str) -> NativeProfileBindings:
    """Return exact ordered Java roots for one executable role profile."""

    _, _, specs = _role_spec(role)
    abilities = tuple(NativeInteractionBinding(spec.root, "Primary") for spec in specs)
    slots = tuple(range(len(abilities)))
    return NativeProfileBindings(role, abilities, slots, None, slots, ())


def hytale_0_5_7_native_npc_role_program_content_sha256(role: str) -> str:
    """Hash the complete typed JAX program for one role independently of a duel."""

    index = _role_index(role)
    bank = _role_bank()
    loadout = type(bank)(*(np.asarray(value[index])[None, None, ...] for value in bank))
    return ability_loadout_content_sha256(loadout)


def hytale_0_5_7_native_npc_role_manifest(role: str) -> dict[str, object]:
    """Describe the exact source/runtime identity consumed by Arena transfer."""

    binding = hytale_0_5_7_native_npc_role_bindings(role)
    return {
        "schema": NATIVE_NPC_ROLE_PROFILE_SCHEMA,
        "hytale_version": "0.5.7",
        "assets_sha256": HYTALE_0_5_7_ASSETS_SHA256,
        "role": role,
        "slots": [row._asdict() for row in binding.abilities],
        "program_content_sha256": (
            hytale_0_5_7_native_npc_role_program_content_sha256(role)
        ),
    }


@lru_cache(maxsize=1)
def _role_bank():
    template = empty_ability_loadout(1, entity_count=1)
    base = {
        field: np.asarray(value) for field, value in zip(template._fields, template)
    }
    rows = []
    for role in ("", *NATIVE_NPC_ROLE_PROFILE_NAMES):
        arrays = {field: _MutableProfileArray(value) for field, value in base.items()}
        if role:
            _write_profile(arrays, 0, 0, _profile(role))
        rows.append({field: arrays[field].value[0, 0] for field in template._fields})
    return type(template)(
        *(np.stack([row[field] for row in rows]) for field in template._fields)
    )


def _profile(role: str) -> dict:
    family, damage, specs = _role_spec(role)
    abilities = []
    for spec in specs:
        selector = spec.selector
        abilities.append(
            _ability(
                spec.root,
                spec.duration,
                _melee(
                    spec.windup,
                    damage,
                    selector.end_distance,
                    spec.half_angle,
                    damage_class=DAMAGE_CLASS_UNKNOWN,
                    force=(0.0, 3.0, -1.0, 1.0),
                    stab_selector=(
                        selector if isinstance(selector, _StabSelectorProgram) else None
                    ),
                    horizontal_selector=(
                        selector
                        if isinstance(selector, _HorizontalSelectorProgram)
                        else None
                    ),
                    selector_requires_injection=False,
                ),
                native_outer_root_id=spec.root,
                interaction_type=INTERACTION_TYPE_PRIMARY,
            )
        )
    return _weapon(role, family, abilities, 0.0, {})


def _role_spec(role: str):
    if not isinstance(role, str):
        raise TypeError("native NPC role must be a string")
    try:
        return _ROLE_SPECS[role]
    except KeyError as error:
        raise ValueError(
            f"unsupported native NPC role {role!r}; choose from {NATIVE_NPC_ROLE_PROFILE_NAMES}"
        ) from error


def _role_index(role: str) -> int:
    if role == "":
        return 0
    _role_spec(role)
    return NATIVE_NPC_ROLE_PROFILE_NAMES.index(role) + 1


__all__ = [
    "NATIVE_NPC_ROLE_PROFILE_NAMES",
    "NATIVE_NPC_ROLE_PROFILE_SCHEMA",
    "hytale_0_5_7_native_npc_role_bindings",
    "hytale_0_5_7_native_npc_role_loadouts",
    "hytale_0_5_7_native_npc_role_manifest",
    "hytale_0_5_7_native_npc_role_program_content_sha256",
]
