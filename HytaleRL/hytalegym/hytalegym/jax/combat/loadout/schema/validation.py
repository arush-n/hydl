"""Host validation for episode-pinned melee loadouts."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

from hytalegym.jax.combat.loadout.schema.contract import (
    MAX_MELEE_ATTACKS,
    MELEE_COOLDOWN_NORMALIZATION_SECONDS,
    MELEE_RANGE_NORMALIZATION_BLOCKS,
    WEAPON_KIND_NONE,
)
from hytalegym.jax.combat.loadout.schema.types import MeleeLoadoutBatch


def validate_melee_loadout(
    loadout: MeleeLoadoutBatch,
    *,
    batch_size: int,
) -> None:
    """Reject shape, dtype, mask, and active-value contract drift."""

    batch = _batch_size(batch_size)
    vector_specs = (
        ("weapon_id", loadout.weapon_id, np.int32),
        ("weapon_kind", loadout.weapon_kind, np.int32),
        ("equipped_mask", loadout.equipped_mask, np.bool_),
        ("overflow", loadout.overflow, np.bool_),
    )
    for name, value, dtype in vector_specs:
        _require(name, value, (batch,), dtype)
    matrix_specs = (
        ("attack_mask", loadout.attack_mask, np.bool_),
        ("hit_delay_ticks", loadout.hit_delay_ticks, np.int32),
        ("range_blocks", loadout.range_blocks, np.float32),
        ("half_angle_degrees", loadout.half_angle_degrees, np.float32),
        ("damage", loadout.damage, np.float32),
        (
            "cooldown_min_seconds",
            loadout.cooldown_min_seconds,
            np.float32,
        ),
        (
            "cooldown_max_seconds",
            loadout.cooldown_max_seconds,
            np.float32,
        ),
        (
            "requires_line_of_sight",
            loadout.requires_line_of_sight,
            np.bool_,
        ),
    )
    for name, value, dtype in matrix_specs:
        _require(name, value, (batch, MAX_MELEE_ATTACKS), dtype)

    mask = np.asarray(loadout.attack_mask)
    effective = (
        mask
        & np.asarray(loadout.equipped_mask)[:, None]
        & ~np.asarray(loadout.overflow)[:, None]
    )
    gap = ~effective[:, :-1] & effective[:, 1:]
    if np.any(gap):
        row, slot = _first_index(gap)
        raise ValueError(
            "attack_mask must be a compact prefix; "
            f"batch {row} has an active slot after slot {slot}"
        )
    active_rows = np.asarray(loadout.equipped_mask) & ~np.asarray(loadout.overflow)
    missing = active_rows & ~np.any(effective, axis=1)
    if np.any(missing):
        row = int(np.flatnonzero(missing)[0])
        raise ValueError(f"equipped loadout at batch {row} has no active attacks")
    weapon_ids = np.asarray(loadout.weapon_id)
    weapon_kinds = np.asarray(loadout.weapon_kind)
    invalid_identity = active_rows & (
        (weapon_ids < 0) | (weapon_kinds == WEAPON_KIND_NONE)
    )
    if np.any(invalid_identity):
        row = int(np.flatnonzero(invalid_identity)[0])
        raise ValueError(
            f"equipped loadout at batch {row} has no valid weapon identity"
        )

    _finite_bounded(
        "range_blocks",
        loadout.range_blocks,
        effective,
        minimum=0.0,
        maximum=MELEE_RANGE_NORMALIZATION_BLOCKS,
    )
    _finite_bounded(
        "half_angle_degrees",
        loadout.half_angle_degrees,
        effective,
        minimum=0.0,
        maximum=180.0,
    )
    _finite_nonnegative("damage", loadout.damage, effective)
    _finite_bounded(
        "cooldown_min_seconds",
        loadout.cooldown_min_seconds,
        effective,
        minimum=0.0,
        maximum=MELEE_COOLDOWN_NORMALIZATION_SECONDS,
    )
    _finite_bounded(
        "cooldown_max_seconds",
        loadout.cooldown_max_seconds,
        effective,
        minimum=0.0,
        maximum=MELEE_COOLDOWN_NORMALIZATION_SECONDS,
    )
    delays = np.asarray(loadout.hit_delay_ticks)
    negative_delay = effective & (delays < 0)
    if np.any(negative_delay):
        row, slot = _first_index(negative_delay)
        raise ValueError(f"hit_delay_ticks is negative at batch {row}, slot {slot}")
    invalid_cooldown = effective & (
        np.asarray(loadout.cooldown_min_seconds)
        > np.asarray(loadout.cooldown_max_seconds)
    )
    if np.any(invalid_cooldown):
        row, slot = _first_index(invalid_cooldown)
        raise ValueError(
            "cooldown_min_seconds exceeds cooldown_max_seconds at "
            f"batch {row}, slot {slot}"
        )


def _batch_size(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("batch_size must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError("batch_size must be a positive integer") from error
    if result <= 0:
        raise ValueError("batch_size must be a positive integer")
    return result


def _require(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: Any,
) -> None:
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} does not match {shape}")
    if np.dtype(value.dtype) != np.dtype(dtype):
        raise TypeError(
            f"{name} dtype {np.dtype(value.dtype)} does not match {np.dtype(dtype)}"
        )


def _finite_nonnegative(
    name: str,
    values: Any,
    mask: np.ndarray,
) -> None:
    _finite_bounded(name, values, mask, minimum=0.0, maximum=None)


def _finite_bounded(
    name: str,
    values: Any,
    mask: np.ndarray,
    *,
    minimum: float,
    maximum: float | None,
) -> None:
    array = np.asarray(values)
    invalid = mask & (~np.isfinite(array) | (array < minimum))
    if maximum is not None:
        invalid |= mask & (array > maximum)
    if np.any(invalid):
        row, slot = _first_index(invalid)
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">={minimum}"
        raise ValueError(
            f"{name} must be finite and {bound}; invalid value at "
            f"batch {row}, slot {slot}"
        )


def _first_index(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in np.argwhere(mask)[0])
