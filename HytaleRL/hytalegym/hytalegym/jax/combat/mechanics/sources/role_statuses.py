"""Compile packaged NPC role effects into fixed-shape mechanics rows."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat.mechanics.schema.contract import (
    RESOURCE_COUNT,
    STATUS_APPLICATION_CAPACITY,
)
from hytalegym.jax.combat.mechanics.factory import empty_status_applications
from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.rulesets import (
    RoleInitialStatus,
    load_combat_ruleset,
    load_role_initial_statuses,
)


_INFINITE_DURATION_SECONDS = np.finfo(np.float32).max


@lru_cache(maxsize=1)
def role_status_programs_by_semantic_id() -> dict[int, RoleInitialStatus]:
    """Return complete installed role-status semantics keyed by device ID."""

    result: dict[int, RoleInitialStatus] = {}
    for statuses in load_role_initial_statuses().values():
        for status in statuses:
            effect_id = semantic_id(status.effect_id)
            previous = result.get(effect_id)
            if previous is not None and (
                previous.effect_id != status.effect_id
                or _status_program_signature(previous)
                != _status_program_signature(status)
            ):
                raise ValueError(
                    "role status semantic-ID collision: "
                    f"{previous.effect_id!r} and {status.effect_id!r}"
                )
            if previous is None:
                result[effect_id] = status
    return result


def _status_program_signature(status: RoleInitialStatus) -> tuple[object, ...]:
    """Compare reusable effect programs independently of role provenance."""

    return (
        status.effect_asset,
        status.infinite,
        status.duration_seconds,
        status.debuff,
        status.invulnerable,
        status.cycle_cooldown_seconds,
        status.damage_per_cycle,
        status.damage_cause,
        status.healing_per_cycle,
        status.resource_id,
        status.resource_delta_per_cycle,
        status.speed_multiplier,
        status.overlap_mode,
        status.flags,
        status.damage_resistance,
    )


def default_combat_role_ids(entity_count: int) -> tuple[str, ...]:
    """Return the fixed matchup roles followed by explicit empty padding."""

    if isinstance(entity_count, bool) or not isinstance(entity_count, int):
        raise TypeError("entity_count must be an integer")
    if entity_count < 2:
        raise ValueError("combat role rows require at least two entities")
    matchup = load_combat_ruleset()["matchup"]
    return (
        matchup["agent_role"],
        matchup["target_role"],
        *(("",) * (entity_count - 2)),
    )


def initial_role_status_applications(
    batch_size: int,
    entity_count: int,
    role_ids: Sequence[str] | None = None,
):
    """Build complete reset-time status applications for each entity role.

    Empty role IDs are explicit inactive padding. Unknown nonempty roles fail
    closed; they never inherit the default pair or silently lose an authored
    effect.
    """

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    roles = (
        default_combat_role_ids(entity_count)
        if role_ids is None
        else tuple(role_ids)
    )
    if len(roles) != entity_count:
        raise ValueError("role_ids must match the entity axis")
    if any(not isinstance(role_id, str) for role_id in roles):
        raise TypeError("role_ids must contain strings")
    catalog = load_role_initial_statuses()
    applications = empty_status_applications(
        batch_size,
        entity_count=entity_count,
    )
    arrays = {
        field: getattr(applications, field)
        for field in applications._fields
    }
    for entity, role_id in enumerate(roles):
        if not role_id:
            continue
        if role_id not in catalog:
            raise ValueError(f"unknown role-initial-status role: {role_id!r}")
        statuses = catalog[role_id]
        if len(statuses) > STATUS_APPLICATION_CAPACITY:
            raise ValueError(
                f"role {role_id!r} exceeds initial status capacity "
                f"{STATUS_APPLICATION_CAPACITY}"
            )
        for slot, status in enumerate(statuses):
            index = (slice(None), entity, slot)
            profile = status.damage_resistance
            values = {
                "requested": True,
                "effect_id": semantic_id(status.effect_id),
                "source_entity_id": entity,
                "duration_seconds": (
                    _INFINITE_DURATION_SECONDS
                    if status.infinite
                    else status.duration_seconds
                ),
                "cycle_cooldown_seconds": status.cycle_cooldown_seconds,
                "damage_per_cycle": status.damage_per_cycle,
                "damage_cause": status.damage_cause,
                "healing_per_cycle": status.healing_per_cycle,
                "resource_id": status.resource_id,
                "resource_delta_per_cycle": status.resource_delta_per_cycle,
                "speed_multiplier": status.speed_multiplier,
                "flags": np.uint32(status.flags),
                "overlap_mode": status.overlap_mode,
            }
            for field, value in values.items():
                arrays[field] = arrays[field].at[index].set(value)
            cause_index = (slice(None), entity, slot, slice(None))
            arrays["damage_resistance_present"] = arrays[
                "damage_resistance_present"
            ].at[cause_index].set(jnp.asarray(profile.present, dtype=jnp.bool_))
            arrays["damage_resistance_flat"] = arrays[
                "damage_resistance_flat"
            ].at[cause_index].set(jnp.asarray(profile.flat, dtype=jnp.float32))
            arrays["damage_resistance_multiplier"] = arrays[
                "damage_resistance_multiplier"
            ].at[cause_index].set(
                jnp.asarray(profile.multiplier, dtype=jnp.float32)
            )
    if arrays["damage_resistance_present"].shape[-1] <= 0:
        raise AssertionError("damage-cause axis must be nonempty")
    if RESOURCE_COUNT <= 0:
        raise AssertionError("resource axis must be nonempty")
    return type(applications)(**arrays)


__all__ = [
    "default_combat_role_ids",
    "initial_role_status_applications",
    "role_status_programs_by_semantic_id",
]
