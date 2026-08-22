"""Strict host compiler for Hytale 0.5.7 damage-resistance maps."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from hytalegym.rulesets.damage_causes import load_damage_cause_rules


_FLAT_TYPE = "Flat"
_MULTIPLIER_TYPE = "Percent"
_MODIFIER_FIELDS = frozenset(("Amount", "CalculationType", "Target"))
_JAVA_INT_MINIMUM = -(1 << 31)
_JAVA_INT_MAXIMUM = (1 << 31) - 1


@dataclass(frozen=True)
class DamageResistanceProfile:
    """One fixed 15-cause resistance map after native-style aggregation."""

    present: tuple[bool, ...]
    inherits: tuple[bool, ...]
    flat: tuple[int, ...]
    multiplier: tuple[float, ...]


def empty_damage_resistance_profile() -> DamageResistanceProfile:
    """Return the identity resistance profile for the shipped cause table."""

    count = len(load_damage_cause_rules())
    return DamageResistanceProfile(
        present=(False,) * count,
        inherits=(False,) * count,
        flat=(0,) * count,
        multiplier=(0.0,) * count,
    )


def compile_damage_resistance_profile(
    source: Mapping[str, Any],
    *,
    source_kind: str,
) -> DamageResistanceProfile:
    """Compile one decoded ItemArmor or EntityEffect resistance block.

    ``source_kind`` is explicit because only ItemArmor entries populate the
    server's ``inheritedParentId`` field. Damage resistance uses the dedicated
    ``ResistanceModifier`` codec (``Flat``/``Percent``), not the similarly
    shaped entity-stat ``StaticModifier`` codec. Unknown causes, modifier
    kinds, and modifier fields fail closed instead of becoming zero resistance.
    """

    if source_kind not in {"armor", "status"}:
        raise ValueError("source_kind must be 'armor' or 'status'")
    if not isinstance(source, Mapping):
        raise TypeError("damage resistance source must be a mapping")
    cause_rules = load_damage_cause_rules()
    cause_index = {rule.asset_id: index for index, rule in enumerate(cause_rules)}
    count = len(cause_rules)
    present = [False] * count
    inherits = [False] * count
    flat = [0] * count
    multiplier = [np.float32(0.0)] * count
    base = _finite_number(source.get("BaseDamageResistance", 0.0), "base resistance")
    if base < 0.0:
        raise ValueError("negative BaseDamageResistance is unsupported")
    if source_kind == "status" and base != 0.0:
        raise ValueError("status resistance cannot define BaseDamageResistance")
    resistance_map = source.get("DamageResistance")
    if resistance_map is None:
        return empty_damage_resistance_profile()
    if not isinstance(resistance_map, Mapping):
        raise TypeError("DamageResistance must be a cause-to-modifier mapping")

    for cause_id, modifiers in resistance_map.items():
        if cause_id not in cause_index:
            raise ValueError(f"unknown damage cause in resistance map: {cause_id!r}")
        index = cause_index[cause_id]
        if not isinstance(modifiers, Sequence) or isinstance(
            modifiers, (str, bytes, bytearray)
        ):
            raise TypeError(f"resistance modifiers for {cause_id} must be a sequence")
        present[index] = True
        inherits[index] = (
            source_kind == "armor" and cause_rules[index].inherits is not None
        )
        for modifier in modifiers:
            if not isinstance(modifier, Mapping):
                raise TypeError(f"resistance modifier for {cause_id} must be a mapping")
            unknown = set(modifier) - _MODIFIER_FIELDS
            if unknown:
                raise ValueError(
                    f"unknown resistance modifier fields for {cause_id}: "
                    f"{sorted(unknown)}"
                )
            calculation_type = modifier.get("CalculationType")
            amount = _finite_number(
                modifier.get("Amount", 0.0),
                f"{cause_id} resistance amount",
            )
            if amount < 0.0:
                raise ValueError(
                    f"negative resistance amount for {cause_id} is unsupported"
                )
            if calculation_type == _FLAT_TYPE:
                flat[index] = _java_float_to_int(
                    float(np.float32(np.float32(flat[index]) + np.float32(amount)))
                )
            elif calculation_type == _MULTIPLIER_TYPE:
                multiplier[index] = np.float32(multiplier[index] + np.float32(amount))
            else:
                raise ValueError(
                    f"unknown resistance calculation type for {cause_id}: "
                    f"{calculation_type!r}"
                )
        if source_kind == "armor":
            flat[index] = _java_float_to_int(flat[index] + base)

    return DamageResistanceProfile(
        present=tuple(present),
        inherits=tuple(inherits),
        flat=tuple(flat),
        multiplier=tuple(float(value) for value in multiplier),
    )


def combine_damage_resistance_profiles(
    profiles: Sequence[DamageResistanceProfile],
) -> DamageResistanceProfile:
    """Combine equipment first and active effects second in server order."""

    result = empty_damage_resistance_profile()
    count = len(result.present)
    present = list(result.present)
    inherits = list(result.inherits)
    flat = list(result.flat)
    multiplier = [np.float32(value) for value in result.multiplier]
    for profile in profiles:
        _validate_profile(profile, count)
        for index in range(count):
            if not profile.present[index]:
                continue
            present[index] = True
            inherits[index] |= profile.inherits[index]
            flat[index] = _java_float_to_int(flat[index] + profile.flat[index])
            multiplier[index] = np.float32(
                multiplier[index] + np.float32(profile.multiplier[index])
            )
    return DamageResistanceProfile(
        present=tuple(present),
        inherits=tuple(inherits),
        flat=tuple(flat),
        multiplier=tuple(float(value) for value in multiplier),
    )


def _validate_profile(profile: DamageResistanceProfile, count: int) -> None:
    if not isinstance(profile, DamageResistanceProfile):
        raise TypeError("resistance profiles must be DamageResistanceProfile values")
    for name in ("present", "inherits", "flat", "multiplier"):
        if len(getattr(profile, name)) != count:
            raise ValueError(f"resistance profile {name} must have {count} entries")
    if any(not math.isfinite(value) for value in profile.multiplier):
        raise ValueError("resistance profile multipliers must be finite")


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


def _java_float_to_int(value: float) -> int:
    if value >= _JAVA_INT_MAXIMUM:
        return _JAVA_INT_MAXIMUM
    if value <= _JAVA_INT_MINIMUM:
        return _JAVA_INT_MINIMUM
    return int(value)


__all__ = [
    "DamageResistanceProfile",
    "combine_damage_resistance_profiles",
    "compile_damage_resistance_profile",
    "empty_damage_resistance_profile",
]
