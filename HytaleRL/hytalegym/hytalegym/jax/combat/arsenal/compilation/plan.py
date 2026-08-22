"""Derive static Arsenal branches from episode-pinned scalar banks.

This module is deliberately profile-agnostic. It inspects compiled event,
ability, and item-program masks, never item names, so new weapon variants gain
the same specialization without another branch.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import numpy as np

from hytalegym.jax.combat.arsenal.schema.contract import (
    AREA_CAPACITY,
    EVENT_AREA,
    EVENT_FLAG_PROJECTILE_TERMINAL_DEPLOYABLE_AREA,
    EVENT_PROJECTILE,
    PROJECTILE_CAPACITY,
)
from hytalegym.jax.combat.arsenal.schema.types import ArsenalRuntimeConfig
from hytalegym.jax.combat.types import CombatParams


class ArsenalCompilePlan(NamedTuple):
    """Host-derived branches and minimum inactive-bank capacities."""

    microticks: int
    null_loadout: bool | None
    has_item_programs: bool | None
    has_projectiles: bool | None
    has_areas: bool | None
    projectile_capacity: int
    area_capacity: int


def plan_arsenal_compilation(
    params: CombatParams,
    config: ArsenalRuntimeConfig,
) -> ArsenalCompilePlan:
    """Inspect static scalar banks without assuming a weapon archetype."""

    microticks = int(jax.device_get(params.microticks))
    projectile_sources = _concrete_event_sources(
        config,
        EVENT_PROJECTILE,
    )
    area_sources = _concrete_area_sources(config)
    has_item_programs = _concrete_item_programs(config)
    has_projectiles = (
        None if projectile_sources is None else bool(projectile_sources.size)
    )
    has_areas = None if area_sources is None else bool(area_sources.size)
    null_loadout = _concrete_null_loadout(config)
    return ArsenalCompilePlan(
        microticks=microticks,
        null_loadout=null_loadout,
        has_item_programs=has_item_programs,
        has_projectiles=has_projectiles,
        has_areas=has_areas,
        projectile_capacity=(1 if has_projectiles is False else PROJECTILE_CAPACITY),
        area_capacity=1 if has_areas is False else AREA_CAPACITY,
    )


def _concrete_event_sources(
    config: ArsenalRuntimeConfig,
    kind: int,
) -> np.ndarray | None:
    try:
        event_mask = np.asarray(config.loadout.event_mask, dtype=np.bool_)
        event_kind = np.asarray(config.loadout.event_kind)
    except (
        TypeError,
        ValueError,
        jax.errors.TracerArrayConversionError,
    ):
        return None
    selected = event_mask & (event_kind == kind)
    authored = np.any(selected, axis=tuple(range(2, selected.ndim)))
    return np.flatnonzero(authored.reshape(-1)).astype(np.int32, copy=False)


def _concrete_area_sources(config: ArsenalRuntimeConfig) -> np.ndarray | None:
    try:
        event_mask = np.asarray(config.loadout.event_mask, dtype=np.bool_)
        event_kind = np.asarray(config.loadout.event_kind)
        event_flags = np.asarray(config.loadout.event_flags, dtype=np.uint32)
    except (
        TypeError,
        ValueError,
        jax.errors.TracerArrayConversionError,
    ):
        return None
    selected = event_mask & (
        (event_kind == EVENT_AREA)
        | (
            (event_kind == EVENT_PROJECTILE)
            & (event_flags & np.uint32(EVENT_FLAG_PROJECTILE_TERMINAL_DEPLOYABLE_AREA))
            != 0
        )
    )
    authored = np.any(selected, axis=tuple(range(2, selected.ndim)))
    return np.flatnonzero(authored.reshape(-1)).astype(np.int32, copy=False)


def _concrete_null_loadout(config: ArsenalRuntimeConfig) -> bool | None:
    try:
        equipped = np.asarray(config.loadout.equipped, dtype=np.bool_)
        ability_mask = np.asarray(config.loadout.ability_mask, dtype=np.bool_)
        failure_bits = np.asarray(
            config.validation_failure_bits,
            dtype=np.uint32,
        )
    except (
        TypeError,
        ValueError,
        jax.errors.TracerArrayConversionError,
    ):
        return None
    return bool(
        not np.any(equipped)
        and not np.any(ability_mask)
        and np.all(failure_bits == np.uint32(0))
    )


def _concrete_item_programs(config: ArsenalRuntimeConfig) -> bool | None:
    try:
        return bool(np.any(np.asarray(config.item_programs.program_mask)))
    except (
        TypeError,
        ValueError,
        jax.errors.TracerArrayConversionError,
    ):
        return None


__all__ = ["ArsenalCompilePlan", "plan_arsenal_compilation"]
