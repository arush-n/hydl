"""Asset-sourced Oxygen regeneration and breathing damage events."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.mechanics.schema.contract import (
    BREATHING_DAMAGE_INTERVAL_SECONDS,
    DAMAGE_DROWNING,
    DAMAGE_SUFFOCATION,
    DROWNING_DAMAGE_AMOUNT,
    OXYGEN_BREATHABLE_REGEN_AMOUNT,
    OXYGEN_REGEN_INTERVAL_SECONDS,
    OXYGEN_SUFFOCATING_REGEN_AMOUNT,
    RESOURCE_OXYGEN,
    SUFFOCATION_DAMAGE_AMOUNT,
)
from hytalegym.jax.combat.mechanics.factory import empty_damage_events
from hytalegym.jax.combat.mechanics.schema.types import (
    CombatMechanicsRules,
    CombatMechanicsState,
    DamageEvents,
)


def tick_breathing(
    state: CombatMechanicsState,
    rules: CombatMechanicsRules,
    dt_seconds: jax.Array,
    cannot_breathe: jax.Array,
    in_fluid: jax.Array,
    alive: jax.Array,
) -> tuple[CombatMechanicsState, DamageEvents]:
    """Advance Oxygen and emit per-entity drowning/suffocation damage.

    ``cannot_breathe`` and ``in_fluid`` are independent so the same primitive
    covers the native distinction between drowning (a non-zero fluid id) and
    suffocation (solid material).  Callers must fail closed on unavailable
    environment evidence by passing ``cannot_breathe=False``; this routine
    never guesses material or fluid state.

    The timers reproduce ``RegeneratingValue``/``DelayedEntitySystem``: a
    zero-initialized remaining duration is decremented before a strict
    below-zero test, at most one cycle executes per server tick, and excess
    time carries to the next tick.
    """

    batch, entity_count = state.resources.shape[:2]
    entity_shape = (batch, entity_count)
    for name, value in (
        ("cannot_breathe", cannot_breathe),
        ("in_fluid", in_fluid),
        ("alive", alive),
    ):
        if value.shape != entity_shape:
            raise ValueError(f"{name} must have shape {entity_shape}")
    if dt_seconds.shape != (batch,):
        raise ValueError(f"dt_seconds must have shape {(batch,)}")

    dt = jnp.asarray(dt_seconds, dtype=jnp.float32)[:, None]
    cannot_breathe = jnp.asarray(cannot_breathe, dtype=jnp.bool_)
    in_fluid = jnp.asarray(in_fluid, dtype=jnp.bool_)
    alive = jnp.asarray(alive, dtype=jnp.bool_)

    oxygen_remaining, oxygen_due = _advance_delayed_timer(
        state.oxygen_regen_remaining_seconds,
        dt,
        OXYGEN_REGEN_INTERVAL_SECONDS,
    )
    current_oxygen = state.resources[..., RESOURCE_OXYGEN]
    oxygen_delta = jnp.where(
        cannot_breathe,
        jnp.float32(OXYGEN_SUFFOCATING_REGEN_AMOUNT),
        jnp.float32(OXYGEN_BREATHABLE_REGEN_AMOUNT),
    )
    oxygen = jnp.clip(
        current_oxygen + jnp.where(oxygen_due & alive, oxygen_delta, 0.0),
        rules.resource_minimum[..., RESOURCE_OXYGEN],
        rules.resource_maximum[..., RESOURCE_OXYGEN],
    )

    damage_remaining, damage_due = _advance_delayed_timer(
        state.breathing_damage_remaining_seconds,
        dt,
        BREATHING_DAMAGE_INTERVAL_SECONDS,
    )
    requested = (
        damage_due
        & alive
        & cannot_breathe
        & (oxygen <= rules.resource_minimum[..., RESOURCE_OXYGEN])
    )
    state = state._replace(
        resources=state.resources.at[..., RESOURCE_OXYGEN].set(oxygen),
        oxygen_regen_remaining_seconds=oxygen_remaining,
        breathing_damage_remaining_seconds=damage_remaining,
    )

    entity_ids = jnp.broadcast_to(
        jnp.arange(entity_count, dtype=jnp.int32)[None, :],
        entity_shape,
    )
    events = empty_damage_events(batch, entity_count)._replace(
        requested=requested,
        target_entity_id=entity_ids,
        amount=jnp.where(
            in_fluid,
            jnp.float32(DROWNING_DAMAGE_AMOUNT),
            jnp.float32(SUFFOCATION_DAMAGE_AMOUNT),
        ),
        cause=jnp.where(
            in_fluid,
            jnp.int32(DAMAGE_DROWNING),
            jnp.int32(DAMAGE_SUFFOCATION),
        ),
    )
    return state, events


def _advance_delayed_timer(
    remaining_seconds: jax.Array,
    dt_seconds: jax.Array,
    interval_seconds: float,
) -> tuple[jax.Array, jax.Array]:
    advanced = remaining_seconds - dt_seconds
    due = advanced < jnp.float32(0.0)
    remaining = jnp.where(
        due,
        advanced + jnp.float32(interval_seconds),
        advanced,
    )
    return remaining, due
