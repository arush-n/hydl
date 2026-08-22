"""Host-built execution plans for exact authored force-path queries."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat.arsenal.schema.contract import (
    EF_FORCE_MAGNITUDE,
    EI_TARGET_MODE,
    EVENT_AREA,
    EVENT_FORCE,
    EVENT_MELEE_CONE,
    EVENT_PROJECTILE,
    REQUIRE_CLEAR_FORCE_PATH,
    TARGET_OTHER,
    TARGET_SELF,
)
from hytalegym.jax.combat.arsenal.schema.types import AbilityLoadout, ForceSweepPlan


def _host_ability_event_bank(loadout: AbilityLoadout) -> tuple[np.ndarray, ...]:
    """Return the runtime event bank as concrete ability-major host arrays."""

    values = jax.tree_util.tree_map(
        lambda value: np.asarray(jax.device_get(value)),
        (
            loadout.event_mask,
            loadout.event_time_seconds,
            loadout.event_kind,
            loadout.event_f32,
            loadout.event_i32,
            loadout.event_flags,
        ),
    )
    event_mask = values[0]
    if event_mask.ndim == 4:
        return values
    if event_mask.ndim != 3:
        raise ValueError("loadout event storage must be packed or ability-major")

    ability_capacity = loadout.ability_mask.shape[2]
    event_capacity = event_mask.shape[2]
    event_index = np.arange(event_capacity)[None, None, None, :]
    start = np.asarray(jax.device_get(loadout.ability_event_start))[..., None]
    count = np.asarray(jax.device_get(loadout.ability_event_count))[..., None]
    mask = (
        event_mask[:, :, None, :]
        & (event_index >= start)
        & (event_index < start + count)
    )

    def expand(value: np.ndarray) -> np.ndarray:
        return np.broadcast_to(
            value[:, :, None, ...],
            value.shape[:2] + (ability_capacity,) + value.shape[2:],
        )

    return (
        mask,
        *(expand(value) for value in values[1:]),
    )


def _force_program_analysis(
    loadout: AbilityLoadout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return exact-plan support, force events, and target modes on the host."""

    (
        event_mask,
        _event_time,
        event_kind,
        event_f32,
        event_i32,
        _event_flags,
    ) = _host_ability_event_bank(loadout)
    requirements = np.asarray(jax.device_get(loadout.ability_requirements))
    force_event = event_mask & (
        np.abs(event_f32[..., EF_FORCE_MAGNITUDE]) > np.float32(0.0)
    )
    direct = (event_kind == EVENT_MELEE_CONE) | (event_kind == EVENT_FORCE)
    target_mode = event_i32[..., EI_TARGET_MODE]
    target_supported = (target_mode == TARGET_SELF) | (target_mode == TARGET_OTHER)
    requires_force = (
        requirements & np.uint32(REQUIRE_CLEAR_FORCE_PATH)
    ) != np.uint32(0)
    supported = (
        requires_force
        & np.any(force_event, axis=3)
        & np.all(~force_event | direct, axis=3)
        & np.all(~force_event | target_supported, axis=3)
    )
    return supported, force_event, target_mode, event_kind


def force_sweep_support_mask(loadout: AbilityLoadout) -> np.ndarray:
    """Return abilities admitted by the exact direct-force geometry producer.

    This is the host-side realizability predicate used by plan construction,
    not a requirement-bit family approximation.
    """

    supported, _force_event, _target_mode, _event_kind = (
        _force_program_analysis(loadout)
    )
    return supported.copy()


def applied_force_collision_support_mask(
    loadout: AbilityLoadout,
) -> np.ndarray:
    """Return force programs handled by impact-time exact motion collision.

    The predicate is derived from event semantics rather than profile names.
    Direct selectors, projectile impacts, and persistent areas all feed the
    same applied-velocity state. Unsupported future event kinds therefore
    remain closed automatically.
    """

    _direct, force_event, target_mode, event_kind = _force_program_analysis(
        loadout
    )
    requirements = np.asarray(jax.device_get(loadout.ability_requirements))
    requires_force = (
        requirements & np.uint32(REQUIRE_CLEAR_FORCE_PATH)
    ) != np.uint32(0)
    supported_kind = (
        (event_kind == EVENT_MELEE_CONE)
        | (event_kind == EVENT_FORCE)
        | (event_kind == EVENT_PROJECTILE)
        | (event_kind == EVENT_AREA)
    )
    supported_target = (
        (target_mode == TARGET_SELF)
        | (target_mode == TARGET_OTHER)
    )
    return (
        requires_force
        & np.any(force_event, axis=3)
        & np.all(~force_event | supported_kind, axis=3)
        & np.all(~force_event | supported_target, axis=3)
    )


def build_force_sweep_plan(
    loadout: AbilityLoadout,
    *,
    compact: bool = True,
) -> ForceSweepPlan:
    """Derive exact force lanes from any fixed-shape authored loadout.

    The compact plan retains only supported source/ability programs and their
    affected victims. ``compact=False`` retains the full rectangular axes as a
    test oracle while applying the same semantic validity predicates.
    """

    if not isinstance(compact, bool):
        raise TypeError("compact must be a bool")
    supported, force_event, target_mode, _event_kind = _force_program_analysis(
        loadout
    )

    batch, entity_count, ability_capacity = supported.shape
    entity_ids = np.arange(entity_count)
    source_victim = entity_ids[:, None] == entity_ids[None, :]
    other_victim = ~source_victim
    affected = np.any(
        force_event[..., None]
        & np.where(
            (target_mode == TARGET_SELF)[..., None],
            source_victim[None, :, None, None, :],
            other_victim[None, :, None, None, :],
        ),
        axis=3,
    )

    dense_ability_count = entity_count * ability_capacity
    flat_supported = supported.reshape(batch, dense_ability_count)
    packed_ability_count = (
        max(1, int(np.max(np.sum(flat_supported, axis=1), initial=0)))
        if compact
        else dense_ability_count
    )
    ability_source_index = np.zeros(
        (batch, packed_ability_count),
        dtype=np.int32,
    )
    ability_slot_index = np.zeros_like(ability_source_index)
    ability_valid = np.zeros(
        (batch, packed_ability_count),
        dtype=np.bool_,
    )
    ability_affected = np.zeros(
        (batch, packed_ability_count, entity_count),
        dtype=np.bool_,
    )
    for batch_index in range(batch):
        selected = (
            np.flatnonzero(flat_supported[batch_index])
            if compact
            else np.arange(dense_ability_count)
        )
        count = selected.size
        sources = selected // ability_capacity
        slots = selected % ability_capacity
        ability_source_index[batch_index, :count] = sources
        ability_slot_index[batch_index, :count] = slots
        ability_valid[batch_index, :count] = flat_supported[
            batch_index,
            selected,
        ]
        ability_affected[batch_index, :count] = affected[
            batch_index,
            sources,
            slots,
        ]

    sweep_dense = packed_ability_count * entity_count
    if compact:
        active_sweep_counts = np.sum(
            ability_valid[..., None] & ability_affected,
            axis=(1, 2),
        )
        packed_sweep_count = max(
            1,
            int(np.max(active_sweep_counts, initial=0)),
        )
    else:
        packed_sweep_count = sweep_dense
    packed_sweep_index = np.full(
        (batch, packed_sweep_count),
        sweep_dense,
        dtype=np.int32,
    )
    for batch_index in range(batch):
        selected = (
            np.flatnonzero(
                (
                    ability_valid[batch_index, :, None]
                    & ability_affected[batch_index]
                ).reshape(-1)
            )
            if compact
            else np.arange(sweep_dense)
        )
        packed_sweep_index[batch_index, : selected.size] = selected

    program_environment: list[int] = []
    program_source: list[int] = []
    program_slot: list[int] = []
    program_valid: list[bool] = []
    program_affected: list[np.ndarray] = []
    for batch_index in range(batch):
        selected = (
            np.flatnonzero(ability_valid[batch_index])
            if compact
            else np.arange(packed_ability_count)
        )
        for local_program in selected:
            program_environment.append(batch_index)
            program_source.append(
                int(ability_source_index[batch_index, local_program])
            )
            program_slot.append(
                int(ability_slot_index[batch_index, local_program])
            )
            program_valid.append(
                bool(ability_valid[batch_index, local_program])
            )
            program_affected.append(
                ability_affected[batch_index, local_program].copy()
            )
    if not program_environment:
        program_environment.append(0)
        program_source.append(0)
        program_slot.append(0)
        program_valid.append(False)
        program_affected.append(np.zeros((entity_count,), dtype=np.bool_))

    program_environment_array = np.asarray(
        program_environment,
        dtype=np.int32,
    )
    program_source_array = np.asarray(program_source, dtype=np.int32)
    program_slot_array = np.asarray(program_slot, dtype=np.int32)
    program_valid_array = np.asarray(program_valid, dtype=np.bool_)
    program_affected_array = np.asarray(
        program_affected,
        dtype=np.bool_,
    )
    sweep_program: list[int] = []
    sweep_victim: list[int] = []
    sweep_valid: list[bool] = []
    for program_index in range(program_environment_array.size):
        selected = (
            np.flatnonzero(
                program_valid_array[program_index]
                & program_affected_array[program_index]
            )
            if compact
            else np.arange(entity_count)
        )
        for victim_index in selected:
            sweep_program.append(program_index)
            sweep_victim.append(int(victim_index))
            sweep_valid.append(
                bool(
                    program_valid_array[program_index]
                    and program_affected_array[program_index, victim_index]
                )
            )
    if not sweep_program:
        sweep_program.append(0)
        sweep_victim.append(0)
        sweep_valid.append(False)

    sweep_program_array = np.asarray(sweep_program, dtype=np.int32)
    sweep_victim_array = np.asarray(sweep_victim, dtype=np.int32)

    return ForceSweepPlan(
        ability_source_index=jnp.asarray(ability_source_index),
        ability_slot_index=jnp.asarray(ability_slot_index),
        ability_valid=jnp.asarray(ability_valid),
        ability_affected=jnp.asarray(ability_affected),
        packed_sweep_index=jnp.asarray(packed_sweep_index),
        program_environment_index=jnp.asarray(program_environment_array),
        program_source_index=jnp.asarray(program_source_array),
        program_slot_index=jnp.asarray(program_slot_array),
        program_valid=jnp.asarray(program_valid_array),
        program_affected=jnp.asarray(program_affected_array),
        sweep_program_index=jnp.asarray(sweep_program_array),
        sweep_victim_index=jnp.asarray(sweep_victim_array),
        sweep_environment_index=jnp.asarray(
            program_environment_array[sweep_program_array]
        ),
        sweep_valid=jnp.asarray(np.asarray(sweep_valid, dtype=np.bool_)),
    )


__all__ = [
    "applied_force_collision_support_mask",
    "build_force_sweep_plan",
    "force_sweep_support_mask",
]
