"""Strict host compiler for Hytale 0.5.7 outgoing damage-class modifiers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


DAMAGE_CLASS_IDS = (
    "Unknown",
    "Light",
    "Charged",
    "Signature",
)

_CLASS_INDEX = {
    asset_id: index for index, asset_id in enumerate(DAMAGE_CLASS_IDS)
}
_ADDITIVE_TYPE = "Additive"
_MULTIPLICATIVE_TYPE = "Multiplicative"
_MODIFIER_FIELDS = frozenset(("Amount", "CalculationType", "Target"))


@dataclass(frozen=True)
class DamageClassEnhancementProfile:
    """One fixed four-class source-equipment enhancement profile.

    ``multiplier`` stores the authored contribution above native's identity
    value of ``1.0``. Native applies one profile as
    ``(damage + flat[class]) * max(0, 1 + multiplier[class])``.
    """

    flat: tuple[float, ...]
    multiplier: tuple[float, ...]


def damage_class_index(value: str | None) -> int:
    """Return the native enum ordinal, treating an omitted class as Unknown."""

    asset_id = "Unknown" if value is None else value
    try:
        return _CLASS_INDEX[asset_id]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown damage class: {value!r}") from error


def empty_damage_class_enhancement_profile() -> DamageClassEnhancementProfile:
    """Return the identity source-equipment enhancement profile."""

    count = len(DAMAGE_CLASS_IDS)
    return DamageClassEnhancementProfile(
        flat=(0.0,) * count,
        multiplier=(0.0,) * count,
    )


def compile_damage_class_enhancement_profile(
    armor: Mapping[str, Any],
) -> DamageClassEnhancementProfile:
    """Compile one decoded ``ItemArmor`` class-enhancement map.

    Damage-class enhancement uses ``StaticModifier``
    (``Additive``/``Multiplicative``), unlike incoming resistance's
    ``ResistanceModifier`` (``Flat``/``Percent``). Unknown fields, enum
    values, and non-finite amounts fail closed at the host boundary.
    """

    if not isinstance(armor, Mapping):
        raise TypeError("damage-class armor source must be a mapping")
    enhancement = armor.get("DamageClassEnhancement")
    if enhancement is None:
        return empty_damage_class_enhancement_profile()
    if not isinstance(enhancement, Mapping):
        raise TypeError("DamageClassEnhancement must be a class-to-modifier mapping")

    flat = [np.float32(0.0)] * len(DAMAGE_CLASS_IDS)
    multiplier = [np.float32(0.0)] * len(DAMAGE_CLASS_IDS)
    for class_id, modifiers in enhancement.items():
        index = damage_class_index(class_id)
        if not isinstance(modifiers, Sequence) or isinstance(
            modifiers,
            (str, bytes, bytearray),
        ):
            raise TypeError(
                f"damage-class modifiers for {class_id} must be a sequence"
            )
        for modifier in modifiers:
            if not isinstance(modifier, Mapping):
                raise TypeError(
                    f"damage-class modifier for {class_id} must be a mapping"
                )
            unknown = set(modifier) - _MODIFIER_FIELDS
            if unknown:
                raise ValueError(
                    f"unknown damage-class modifier fields for {class_id}: "
                    f"{sorted(unknown)}"
                )
            if "Target" in modifier:
                raise ValueError(
                    "damage-class modifier Target is unsupported until its "
                    "equipment-path semantics are certified"
                )
            amount = np.float32(
                _finite_number(
                    modifier.get("Amount", 0.0),
                    f"{class_id} damage-class amount",
                )
            )
            calculation_type = modifier.get("CalculationType")
            if calculation_type == _ADDITIVE_TYPE:
                flat[index] = np.float32(flat[index] + amount)
            elif calculation_type == _MULTIPLICATIVE_TYPE:
                multiplier[index] = np.float32(multiplier[index] + amount)
            else:
                raise ValueError(
                    f"unknown damage-class calculation type for {class_id}: "
                    f"{calculation_type!r}"
                )
    return DamageClassEnhancementProfile(
        flat=tuple(float(value) for value in flat),
        multiplier=tuple(float(value) for value in multiplier),
    )


def combine_damage_class_enhancement_profiles(
    profiles: Sequence[DamageClassEnhancementProfile],
) -> DamageClassEnhancementProfile:
    """Combine fixed equipped pieces in native armor-container order."""

    count = len(DAMAGE_CLASS_IDS)
    flat = [np.float32(0.0)] * count
    multiplier = [np.float32(0.0)] * count
    for profile in profiles:
        _validate_profile(profile, count)
        for index in range(count):
            flat[index] = np.float32(
                flat[index] + np.float32(profile.flat[index])
            )
            multiplier[index] = np.float32(
                multiplier[index] + np.float32(profile.multiplier[index])
            )
    return DamageClassEnhancementProfile(
        flat=tuple(float(value) for value in flat),
        multiplier=tuple(float(value) for value in multiplier),
    )


def _validate_profile(
    profile: DamageClassEnhancementProfile,
    count: int,
) -> None:
    if not isinstance(profile, DamageClassEnhancementProfile):
        raise TypeError(
            "damage-class profiles must be DamageClassEnhancementProfile values"
        )
    if len(profile.flat) != count or len(profile.multiplier) != count:
        raise ValueError(f"damage-class profiles must have {count} entries")
    if any(
        not math.isfinite(value)
        for values in (profile.flat, profile.multiplier)
        for value in values
    ):
        raise ValueError("damage-class profile values must be finite")


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


__all__ = [
    "DAMAGE_CLASS_IDS",
    "DamageClassEnhancementProfile",
    "combine_damage_class_enhancement_profiles",
    "compile_damage_class_enhancement_profile",
    "damage_class_index",
    "empty_damage_class_enhancement_profile",
]
