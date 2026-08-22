"""Entity gather/set helpers, status and resistance modifiers."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.mechanics.schema.contract import (
    DAMAGE_CLASS_COUNT,
    DAMAGE_COUNT,
    RESOURCE_COUNT,
)
from hytalegym.jax.combat.mechanics.schema.types import (
    CombatMechanicsRules,
    CombatMechanicsState,
    StatusState,
)


def _dodge_local_direction(direction: jax.Array) -> jax.Array:
    x = jnp.where(
        direction == 3,
        -1.0,
        jnp.where(direction == 4, 1.0, 0.0),
    )
    z = jnp.where(
        direction == 1,
        -1.0,
        jnp.where(direction == 2, 1.0, 0.0),
    )
    return jnp.stack((x, jnp.zeros_like(x), z), axis=2).astype(jnp.float32)


def _entity_dt(
    dt_seconds: jax.Array,
    batch: int,
    entity_count: int,
) -> jax.Array:
    value = jnp.asarray(dt_seconds, dtype=jnp.float32)
    if value.ndim == 0:
        value = jnp.broadcast_to(value, (batch,))
    if value.shape != (batch,):
        raise ValueError(f"dt_seconds must be scalar or [{batch}]")
    return jnp.broadcast_to(value[:, None], (batch, entity_count))


def _gather_entity(array: jax.Array, entity_id: jax.Array) -> jax.Array:
    batch = array.shape[0]
    return array[jnp.arange(batch), entity_id]


def _gather_entity_cause(
    array: jax.Array,
    entity_id: jax.Array,
    cause: jax.Array,
) -> jax.Array:
    return array[
        jnp.arange(array.shape[0]),
        entity_id,
        jnp.clip(cause, 0, DAMAGE_COUNT - 1),
    ]


def _gather_entity_class(
    array: jax.Array,
    entity_id: jax.Array,
    damage_class: jax.Array,
) -> jax.Array:
    return array[
        jnp.arange(array.shape[0])[:, None],
        entity_id,
        jnp.clip(damage_class, 0, DAMAGE_CLASS_COUNT - 1),
    ]


def _select_tree(mask: jax.Array, selected, fallback):
    return jax.tree_util.tree_map(
        lambda yes, no: jnp.where(
            mask.reshape((mask.shape[0],) + (1,) * (yes.ndim - 1)),
            yes,
            no,
        ),
        selected,
        fallback,
    )


def _set_entity(
    array: jax.Array,
    entity_id: jax.Array,
    value: jax.Array,
) -> jax.Array:
    return array.at[jnp.arange(array.shape[0]), entity_id].set(value)


def _set_failure(bits: jax.Array, mask: jax.Array, code: int) -> jax.Array:
    return jnp.where(
        mask,
        jnp.bitwise_or(bits, jnp.uint32(code)),
        bits,
    )


def _write_new(array: jax.Array, value: jax.Array, selected: jax.Array):
    return jnp.where(selected, value[..., None], array)


def _write_new_cause(
    array: jax.Array,
    value: jax.Array,
    selected: jax.Array,
) -> jax.Array:
    return jnp.where(
        selected[..., None],
        value[..., None, :],
        array,
    )


def clear_statuses(
    state: CombatMechanicsState,
    effect_ids: jax.Array,
) -> tuple[CombatMechanicsState, jax.Array]:
    """Clear requested effect IDs from ``[B,E,Q]`` without reallocating slots."""

    expected = state.statuses.active.shape[:2]
    if effect_ids.ndim != 3 or effect_ids.shape[:2] != expected:
        raise ValueError(
            f"effect_ids must have shape {expected + ('Q',)}, got {effect_ids.shape}"
        )
    requested = effect_ids > 0
    matches = (
        state.statuses.active[..., None]
        & requested[:, :, None, :]
        & (state.statuses.effect_id[..., None] == effect_ids[:, :, None, :])
    )
    cleared = jnp.any(matches, axis=3)
    statuses = state.statuses._replace(active=state.statuses.active & ~cleared)
    return state._replace(statuses=statuses), cleared


def damage_resistance_modifiers(
    state: CombatMechanicsState,
    rules: CombatMechanicsRules,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Combine episode-pinned equipment and currently active effect entries."""

    active = state.statuses.active[..., None]
    status_present = jnp.any(
        active & state.statuses.damage_resistance_present,
        axis=2,
    )
    status_flat = jnp.sum(
        jnp.where(
            active & state.statuses.damage_resistance_present,
            state.statuses.damage_resistance_flat,
            jnp.float32(0.0),
        ),
        axis=2,
    )
    status_multiplier = jnp.sum(
        jnp.where(
            active & state.statuses.damage_resistance_present,
            state.statuses.damage_resistance_multiplier,
            jnp.float32(0.0),
        ),
        axis=2,
    )
    return (
        rules.equipment_damage_resistance_present | status_present,
        rules.equipment_damage_resistance_inherits,
        rules.equipment_damage_resistance_flat + status_flat,
        rules.equipment_damage_resistance_multiplier + status_multiplier,
    )


def guard_resource_available(
    resources: jax.Array,
    rules: CombatMechanicsRules,
) -> jax.Array:
    """Return the per-entity equipment-resource gate for guard."""

    resource_id = jnp.clip(
        rules.guard_required_resource_id,
        0,
        RESOURCE_COUNT - 1,
    )
    value = jnp.take_along_axis(
        resources,
        resource_id[..., None],
        axis=2,
    )[..., 0]
    return (rules.guard_required_resource_id < 0) | (
        value + jnp.float32(1.0e-6) >= rules.guard_required_resource_minimum
    )


def status_modifiers(status: StatusState) -> tuple[jax.Array, jax.Array]:
    """Return bitwise flags and multiplicative speed for active slots."""

    flags = jnp.bitwise_or.reduce(
        jnp.where(status.active, status.flags, jnp.uint32(0)),
        axis=2,
    )
    speed = jnp.prod(
        jnp.where(status.active, status.speed_multiplier, jnp.float32(1.0)),
        axis=2,
    )
    return flags, speed
