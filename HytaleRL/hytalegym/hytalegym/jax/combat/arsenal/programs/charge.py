"""C1 ChargeTable -- hold-time selection of an authored ``Charging`` child.

A Hytale ``{"Type": "Charging"}`` root is an ordered ``(hold_threshold ->
child)`` table.  The holder accumulates time while the control stays pressed;
on release the engine runs the child carrying the **highest threshold at or
below** the held duration.  Every charging weapon in 0.5.7 is that one table at
a different size -- Sword and Daggers and Battleaxe have two rows, the Shortbow
draw ladder has five -- so this module is deliberately weapon-agnostic and
contains no weapon, family or item branch.

The component is a *pre-processing stage*: it rewrites the requested ability
slot before :func:`start_abilities` sees it, and never reaches inside ability
admission.  An ability with no authored table (``ability_hold_count == 0``) is
passed through unchanged, so every non-charging weapon is bit-identical to the
behaviour before this module existed.

Holding is expressed by *sustaining the same request across ticks*, which needs
no new action head: the policy already re-emits a slot every tick, and
``ability_legal_mask`` gates acceptance on ``active_ability_slot < 0``, so a
charge accumulates strictly before anything is admitted.

**Nested roots collapse.**  The Sword's authored graph gates twice (an outer
0.2 s root, then an inner 0.65 s one), but *every* early branch targets the same
``_Chain``, so the agent-visible table is two rows.  Compile the flattened
thresholds; do not model the nesting.

Not modelled here, by design: ``HorizontalSpeedMultiplier`` (component C2, it
belongs to locomotion) and the charged child's own physics (C4 -- the Daggers'
Pounce is an airborne branch, the Battleaxe's Downstrike waits for ground).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "resolve_charge_hold",
    "select_charge_child",
]


def select_charge_child(
    thresholds: jax.Array,
    children: jax.Array,
    count: jax.Array,
    held_seconds: jax.Array,
) -> jax.Array:
    """Return the child slot for ``held_seconds`` against one authored table.

    ``thresholds`` and ``children`` are ``[..., capacity]`` and must be authored
    in ascending threshold order; ``count`` is how many rows are live.  Rows at
    or below the held duration are eligible and the **last** such row wins,
    which is the highest threshold because the table is ordered.

    A table whose first threshold exceeds ``held_seconds`` selects nothing and
    returns ``-1``; the Shortbow's ladder starts at 0.1 rather than 0, so this
    is authored behaviour and not an error.
    """

    capacity = thresholds.shape[-1]
    index = jnp.arange(capacity, dtype=jnp.int32)
    live = index < count[..., None]
    reached = live & (thresholds <= held_seconds[..., None])
    any_reached = jnp.any(reached, axis=-1)
    # Ascending order makes the last reached row the highest threshold. argmax
    # on the reversed mask finds it without a sort or a scan.
    reversed_first = jnp.argmax(reached[..., ::-1].astype(jnp.int32), axis=-1)
    selected = jnp.int32(capacity - 1) - reversed_first
    child = jnp.take_along_axis(
        children,
        selected[..., None],
        axis=-1,
    )[..., 0]
    return jnp.where(any_reached, child, jnp.int32(-1))


def resolve_charge_hold(
    charge_slot: jax.Array,
    charge_seconds: jax.Array,
    requested_slot: jax.Array,
    loadout,
    dt_seconds: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Advance held charges and emit the slot that should start this tick.

    Returns ``(charge_slot, charge_seconds, effective_slot)``.

    * A request for a slot with an authored table starts or continues a hold and
      emits ``-1`` -- nothing starts while the control is still down.
    * Releasing (requesting a different slot, or none) resolves the table and
      emits the selected child.
    * A root **without** ``AllowIndefiniteHold`` finishes on its own the tick its
      clock reaches the top threshold, without waiting for release.  This is
      ``ChargingInteraction.simulateTick0``'s
      ``isCharging(...) && (allowIndefiniteHold || time < highestChargeValue)``:
      Sword, Daggers and Battleaxe omit the flag and fire at the threshold, the
      Spear sets it and waits for the control to come up.
    * A request for a slot with no authored table is forwarded unchanged, and
      also cancels any hold in flight, matching a control that moved on.

    ``dt_seconds`` is the fixed interaction tick; the hold clock is
    ``InteractionManager``'s, not the measured frame delta.
    """

    ability_capacity = loadout.ability_hold_count.shape[-1]
    hold_capacity = loadout.ability_hold_threshold_seconds.shape[-1]
    requested = requested_slot >= 0
    safe_request = jnp.clip(requested_slot, 0, ability_capacity - 1)

    def gather_scalar(field, index):
        return jnp.take_along_axis(field, index[..., None], axis=2)[..., 0]

    def gather_row(field, index):
        rows = jnp.broadcast_to(
            index[..., None, None],
            index.shape + (1, hold_capacity),
        )
        return jnp.take_along_axis(field, rows, axis=2)[..., 0, :]

    request_count = jnp.where(
        requested,
        gather_scalar(loadout.ability_hold_count, safe_request),
        jnp.int32(0),
    )
    request_charges = requested & (request_count > 0)

    holding = charge_slot >= 0
    continuing = holding & request_charges & (requested_slot == charge_slot)
    starting = request_charges & ~continuing
    # A hold ends when its own slot stops being requested. Switching straight
    # from one chargeable slot to another releases the first and begins the
    # second on the same tick; the released child is what starts, and the new
    # hold still accumulates from this tick.
    releasing = holding & ~continuing

    accumulated = jnp.where(
        continuing,
        charge_seconds + dt_seconds,
        jnp.where(starting, dt_seconds, jnp.float32(0.0)),
    )

    live = jnp.arange(hold_capacity, dtype=jnp.int32) < request_count[..., None]
    request_thresholds = gather_row(
        loadout.ability_hold_threshold_seconds,
        safe_request,
    )
    highest = jnp.max(
        jnp.where(live, request_thresholds, jnp.float32(-jnp.inf)),
        axis=-1,
    )
    allow_indefinite = gather_scalar(
        loadout.ability_hold_allow_indefinite,
        safe_request,
    )
    auto_fire = request_charges & ~allow_indefinite & (accumulated >= highest)

    fired_now = select_charge_child(
        request_thresholds,
        gather_row(loadout.ability_hold_child_slot, safe_request),
        request_count,
        accumulated,
    )

    safe_hold = jnp.clip(charge_slot, 0, ability_capacity - 1)
    released_child = select_charge_child(
        gather_row(loadout.ability_hold_threshold_seconds, safe_hold),
        gather_row(loadout.ability_hold_child_slot, safe_hold),
        gather_scalar(loadout.ability_hold_count, safe_hold),
        charge_seconds,
    )

    # A release resolves the *previous* hold and wins over a charge that only
    # just began this tick, matching the engine finishing the old interaction
    # before the new root starts.
    effective_slot = jnp.where(
        releasing,
        released_child,
        jnp.where(
            auto_fire,
            fired_now,
            jnp.where(request_charges, jnp.int32(-1), requested_slot),
        ),
    )
    still_holding = request_charges & ~auto_fire
    next_slot = jnp.where(still_holding, requested_slot, jnp.int32(-1))
    next_seconds = jnp.where(still_holding, accumulated, jnp.float32(0.0))
    return next_slot, next_seconds, effective_slot
