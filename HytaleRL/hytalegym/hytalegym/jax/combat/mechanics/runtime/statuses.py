"""Status application and resource ticking."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.mechanics.schema.contract import (
    DAMAGE_COUNT,
    CONTROL_IMMUNITY_INCREMENT,
    CONTROL_IMMUNITY_MAXIMUM,
    CONTROL_IMMUNITY_REGEN_AMOUNT,
    CONTROL_IMMUNITY_REGEN_INTERVAL_SECONDS,
    MECHANICS_FAILURE_INVALID_COMMAND,
    MECHANICS_FAILURE_INVALID_STATE,
    MECHANICS_FAILURE_STATUS_OVERFLOW,
    RESOURCE_COUNT,
    RESOURCE_STAMINA,
    STAMINA_BROKEN_REGEN_DELAY_SECONDS,
    STAMINA_REGEN_DELAY_AMOUNT,
    STAMINA_REGEN_DELAY_INTERVAL_SECONDS,
    STATUS_FLAG_CONTROL_IMMUNITY_GATED,
    STATUS_OVERLAP_EXTEND,
    STATUS_OVERLAP_IGNORE,
    STATUS_OVERLAP_OVERWRITE,
)
from hytalegym.jax.combat.mechanics.schema.types import (
    CombatMechanicsRules,
    CombatMechanicsState,
    StatusApplications,
    StatusState,
    StatusTick,
)
from hytalegym.jax.combat.mechanics.runtime.access import (
    _entity_dt,
    _gather_entity_cause,
    _select_tree,
    _set_failure,
    _write_new,
    _write_new_cause,
    status_modifiers,
)


def _resolve_damage_resistance_amount(
    amount: jax.Array,
    active: jax.Array,
    target: jax.Array,
    cause: jax.Array,
    rules: CombatMechanicsRules,
    present: jax.Array,
    inherits: jax.Array,
    flat: jax.Array,
    multiplier: jax.Array,
) -> jax.Array:
    """Reproduce ArmorDamageReduction's exact-entry and parent walk."""

    cause = jnp.clip(cause, 0, DAMAGE_COUNT - 1)
    bypass = _gather_entity_cause(
        rules.cause_bypass_resistances,
        target,
        cause,
    )
    exact_present = _gather_entity_cause(present, target, cause)
    applies = active & ~bypass & exact_present

    def apply_entry(entry_cause: jax.Array) -> jax.Array:
        entry_flat = _gather_entity_cause(flat, target, entry_cause)
        entry_multiplier = _gather_entity_cause(
            multiplier,
            target,
            entry_cause,
        )
        return jnp.maximum(jnp.float32(0.0), amount - entry_flat) * jnp.maximum(
            jnp.float32(0.0),
            jnp.float32(1.0) - entry_multiplier,
        )

    resolved = jnp.where(applies, apply_entry(cause), amount)
    walking = applies & _gather_entity_cause(inherits, target, cause)

    def visit_parent(_, carry):
        current_cause, current_amount, continue_walk = carry
        parent = _gather_entity_cause(
            rules.cause_inherits,
            target,
            current_cause,
        )
        safe_parent = jnp.clip(parent, 0, DAMAGE_COUNT - 1)
        visit = (
            continue_walk
            & (parent >= 0)
            & _gather_entity_cause(present, target, safe_parent)
        )
        current_amount = jnp.where(
            visit,
            apply_entry(safe_parent),
            current_amount,
        )
        current_cause = jnp.where(visit, safe_parent, current_cause)
        continue_walk = visit & _gather_entity_cause(
            inherits,
            target,
            safe_parent,
        )
        return current_cause, current_amount, continue_walk

    _, resolved, _ = jax.lax.fori_loop(
        0,
        DAMAGE_COUNT,
        visit_parent,
        (cause, resolved, walking),
    )
    return resolved


def apply_resource_delta(
    state: CombatMechanicsState,
    rules: CombatMechanicsRules,
    entity_id: jax.Array,
    resource_delta: jax.Array,
) -> CombatMechanicsState:
    """Apply one dense ``[B,R]`` resource delta to selected entities."""

    batch = state.resources.shape[0]
    entity_count = state.resources.shape[1]
    valid_entity = (entity_id >= 0) & (entity_id < entity_count)
    valid_delta = resource_delta.shape == (
        batch,
        RESOURCE_COUNT,
    ) and entity_id.shape == (batch,)
    if not valid_delta:
        raise ValueError("resource delta or entity_id has the wrong shape")
    invalid = ~valid_entity | ~jnp.all(jnp.isfinite(resource_delta), axis=1)
    bits = _set_failure(
        state.failure_bits,
        invalid,
        MECHANICS_FAILURE_INVALID_COMMAND,
    )
    row_valid = bits == jnp.uint32(0)
    selected = jax.nn.one_hot(
        jnp.clip(entity_id, 0, entity_count - 1),
        entity_count,
        dtype=jnp.float32,
    )[..., None]
    resources = jnp.clip(
        state.resources + selected * resource_delta[:, None, :],
        rules.resource_minimum,
        rules.resource_maximum,
    )
    stamina = resources[..., RESOURCE_STAMINA]
    spent = selected[..., 0] > 0.0
    broken = state.stamina_broken | (
        spent & (resource_delta[:, None, RESOURCE_STAMINA] < 0.0) & (stamina <= 0.0)
    )
    delay = jnp.where(
        broken & ~state.stamina_broken,
        jnp.minimum(
            state.stamina_regen_delay_seconds,
            jnp.float32(STAMINA_BROKEN_REGEN_DELAY_SECONDS),
        ),
        state.stamina_regen_delay_seconds,
    )
    candidate = state._replace(
        resources=resources,
        stamina_broken=broken,
        guard_held=state.guard_held & ~broken,
        guard_active=state.guard_active & ~broken,
        guard_windup_elapsed_seconds=jnp.where(
            broken,
            jnp.float32(0.0),
            state.guard_windup_elapsed_seconds,
        ),
        stamina_regen_delay_seconds=delay,
        failure_bits=bits,
    )
    return _select_tree(row_valid, candidate, state)._replace(failure_bits=bits)


def apply_statuses(
    state: CombatMechanicsState,
    applications: StatusApplications,
) -> CombatMechanicsState:
    """Apply status overlap rules and reject capacity overflow atomically."""

    requested = applications.requested
    resistance_valid = jnp.all(
        jnp.isfinite(applications.damage_resistance_flat), axis=3
    ) & jnp.all(
        jnp.isfinite(applications.damage_resistance_multiplier),
        axis=3,
    )
    valid = (
        (applications.effect_id > 0)
        & jnp.isfinite(applications.duration_seconds)
        & (applications.duration_seconds > 0.0)
        & jnp.isfinite(applications.cycle_cooldown_seconds)
        & (applications.cycle_cooldown_seconds >= 0.0)
        & jnp.isfinite(applications.damage_per_cycle)
        & (applications.damage_per_cycle >= 0.0)
        & (applications.damage_cause >= 0)
        & (applications.damage_cause < DAMAGE_COUNT)
        & jnp.isfinite(applications.healing_per_cycle)
        & (applications.healing_per_cycle >= 0.0)
        & (applications.resource_id >= -1)
        & (applications.resource_id < RESOURCE_COUNT)
        & jnp.isfinite(applications.resource_delta_per_cycle)
        & jnp.isfinite(applications.speed_multiplier)
        & (applications.speed_multiplier > 0.0)
        & (applications.overlap_mode >= STATUS_OVERLAP_IGNORE)
        & (applications.overlap_mode <= STATUS_OVERLAP_OVERWRITE)
        & (applications.source_entity_id >= -1)
        & (applications.source_entity_id < state.resources.shape[1])
        & resistance_valid
    )
    invalid = jnp.any(requested & ~valid, axis=(1, 2))
    bits = _set_failure(
        state.failure_bits,
        invalid,
        MECHANICS_FAILURE_INVALID_COMMAND,
    )
    original = state
    status = state.statuses
    inputs = jax.tree_util.tree_map(
        lambda value: jnp.moveaxis(value, 2, 0),
        applications,
    )

    def apply_one(
        carry: tuple[StatusState, jax.Array, jax.Array],
        value,
    ) -> tuple[tuple[StatusState, jax.Array, jax.Array], None]:
        current, failure_bits, control_immunity = carry
        allowed = failure_bits == jnp.uint32(0)
        immunity_gated = (
            value.flags & jnp.uint32(STATUS_FLAG_CONTROL_IMMUNITY_GATED)
        ) != 0
        immune = control_immunity >= CONTROL_IMMUNITY_MAXIMUM
        request = value.requested & allowed[:, None] & ~(immunity_gated & immune)
        matches = current.active & (current.effect_id == value.effect_id[..., None])
        exists = jnp.any(matches, axis=2)
        available = jnp.any(~current.active, axis=2)
        overflow = request & ~exists & ~available
        failure_bits = _set_failure(
            failure_bits,
            jnp.any(overflow, axis=1),
            MECHANICS_FAILURE_STATUS_OVERFLOW,
        )
        can_apply = request & (failure_bits == 0)[:, None]
        control_immunity = jnp.minimum(
            jnp.float32(CONTROL_IMMUNITY_MAXIMUM),
            control_immunity
            + (can_apply & immunity_gated).astype(jnp.float32)
            * jnp.float32(CONTROL_IMMUNITY_INCREMENT),
        )
        existing_slot = jnp.argmax(matches, axis=2)
        free_slot = jnp.argmax(~current.active, axis=2)
        slot = jnp.where(exists, existing_slot, free_slot)
        selected = (
            jax.nn.one_hot(
                slot,
                current.active.shape[2],
                dtype=jnp.bool_,
            )
            & can_apply[..., None]
        )
        create = selected & ~exists[..., None]
        extend = (
            selected
            & exists[..., None]
            & (value.overlap_mode[..., None] == STATUS_OVERLAP_EXTEND)
        )
        overwrite = (
            selected
            & exists[..., None]
            & (value.overlap_mode[..., None] == STATUS_OVERLAP_OVERWRITE)
        )
        duration = jnp.where(
            create,
            value.duration_seconds[..., None],
            jnp.where(
                extend,
                current.remaining_seconds + value.duration_seconds[..., None],
                jnp.where(
                    overwrite,
                    value.duration_seconds[..., None],
                    current.remaining_seconds,
                ),
            ),
        )
        current = current._replace(
            effect_id=_write_new(
                current.effect_id,
                value.effect_id,
                create,
            ),
            source_entity_id=_write_new(
                current.source_entity_id,
                value.source_entity_id,
                create,
            ),
            remaining_seconds=duration,
            cycle_elapsed_seconds=_write_new(
                current.cycle_elapsed_seconds,
                jnp.zeros_like(value.duration_seconds),
                create,
            ),
            cycle_cooldown_seconds=_write_new(
                current.cycle_cooldown_seconds,
                value.cycle_cooldown_seconds,
                create,
            ),
            damage_per_cycle=_write_new(
                current.damage_per_cycle,
                value.damage_per_cycle,
                create,
            ),
            damage_cause=_write_new(
                current.damage_cause,
                value.damage_cause,
                create,
            ),
            healing_per_cycle=_write_new(
                current.healing_per_cycle,
                value.healing_per_cycle,
                create,
            ),
            resource_id=_write_new(
                current.resource_id,
                value.resource_id,
                create,
            ),
            resource_delta_per_cycle=_write_new(
                current.resource_delta_per_cycle,
                value.resource_delta_per_cycle,
                create,
            ),
            speed_multiplier=_write_new(
                current.speed_multiplier,
                value.speed_multiplier,
                create,
            ),
            damage_resistance_present=_write_new_cause(
                current.damage_resistance_present,
                value.damage_resistance_present,
                create,
            ),
            damage_resistance_flat=_write_new_cause(
                current.damage_resistance_flat,
                value.damage_resistance_flat,
                create,
            ),
            damage_resistance_multiplier=_write_new_cause(
                current.damage_resistance_multiplier,
                value.damage_resistance_multiplier,
                create,
            ),
            flags=_write_new(current.flags, value.flags, create),
            overlap_mode=_write_new(
                current.overlap_mode,
                value.overlap_mode,
                create,
            ),
            active=current.active | create,
            has_cycled=jnp.where(create, False, current.has_cycled),
        )
        return (current, failure_bits, control_immunity), None

    (status, bits, control_immunity), _ = jax.lax.scan(
        apply_one,
        (status, bits, state.control_immunity),
        inputs,
    )
    row_valid = bits == jnp.uint32(0)
    candidate = state._replace(
        statuses=status,
        control_immunity=control_immunity,
        failure_bits=bits,
    )
    return _select_tree(row_valid, candidate, original)._replace(failure_bits=bits)


def tick_resources(
    state: CombatMechanicsState,
    rules: CombatMechanicsRules,
    dt_seconds: jax.Array,
    regen_eligible: jax.Array,
) -> CombatMechanicsState:
    """Tick fixed-interval resource modifiers without a Python loop."""

    batch = state.resources.shape[0]
    dt = _entity_dt(
        dt_seconds,
        batch,
        state.resources.shape[1],
    )[..., None]
    intervals = rules.resource_regen_interval_seconds
    invalid = (
        ~jnp.all(jnp.isfinite(state.resources), axis=(1, 2))
        | ~jnp.all(jnp.isfinite(state.control_immunity), axis=1)
        | jnp.any(
            (state.control_immunity < 0.0)
            | (state.control_immunity > CONTROL_IMMUNITY_MAXIMUM),
            axis=1,
        )
        | ~jnp.all(
            jnp.isfinite(state.control_immunity_regen_clock),
            axis=1,
        )
        | ~jnp.all(
            jnp.isfinite(state.guard_windup_elapsed_seconds),
            axis=1,
        )
        | ~jnp.all(
            jnp.isfinite(state.stamina_regen_delay_clock),
            axis=1,
        )
        | jnp.any(state.stamina_regen_delay_clock < 0.0, axis=1)
        | jnp.any(
            state.guard_windup_elapsed_seconds < 0.0,
            axis=1,
        )
        | jnp.any(
            (rules.resource_regen_amount > 0.0)
            & (~jnp.isfinite(intervals) | (intervals <= 0.0)),
            axis=(1, 2),
        )
        | jnp.any(~jnp.isfinite(dt_seconds))
        | jnp.any(jnp.asarray(dt_seconds) < 0.0)
    )
    bits = _set_failure(
        state.failure_bits,
        invalid,
        MECHANICS_FAILURE_INVALID_STATE,
    )
    row_valid = bits == jnp.uint32(0)
    positive_regen = rules.resource_regen_amount > jnp.float32(0.0)
    negative_regen = rules.resource_regen_amount < jnp.float32(0.0)
    timer_running = (
        positive_regen & (state.resources < rules.resource_maximum)
    ) | (negative_regen & (state.resources > rules.resource_minimum))
    advanced_clock = state.resource_regen_clock + jnp.where(
        timer_running,
        dt,
        jnp.float32(0.0),
    )
    # RegeneratingValue subtracts one dt, tests remaining time strictly below
    # zero, and adds exactly one interval.  It never catches up multiple
    # cycles in one server tick; any excess remains due on a later tick.
    due = timer_running & (advanced_clock > intervals)
    cycles = due.astype(jnp.int32)
    clock = jnp.where(
        timer_running,
        jnp.where(due, advanced_clock - intervals, advanced_clock),
        state.resource_regen_clock,
    )
    eligible = regen_eligible & (rules.resource_maximum > rules.resource_minimum)
    eligible = eligible.at[..., RESOURCE_STAMINA].set(
        eligible[..., RESOURCE_STAMINA]
        & ~state.guard_active
        & (state.stamina_regen_delay_seconds >= 0.0)
    )
    resources = jnp.clip(
        state.resources
        + jnp.where(
            eligible,
            cycles.astype(jnp.float32) * rules.resource_regen_amount,
            jnp.float32(0.0),
        ),
        rules.resource_minimum,
        rules.resource_maximum,
    )
    delay_running = state.stamina_regen_delay_seconds < jnp.float32(0.0)
    delay_clock_advanced = state.stamina_regen_delay_clock + jnp.where(
        delay_running,
        dt[..., 0],
        jnp.float32(0.0),
    )
    delay_due = delay_running & (
        delay_clock_advanced
        > jnp.float32(STAMINA_REGEN_DELAY_INTERVAL_SECONDS)
    )
    delay_clock = jnp.where(
        delay_running,
        jnp.where(
            delay_due,
            delay_clock_advanced
            - jnp.float32(STAMINA_REGEN_DELAY_INTERVAL_SECONDS),
            delay_clock_advanced,
        ),
        state.stamina_regen_delay_clock,
    )
    delay = jnp.minimum(
        jnp.float32(0.0),
        state.stamina_regen_delay_seconds
        + delay_due.astype(jnp.float32)
        * jnp.float32(STAMINA_REGEN_DELAY_AMOUNT),
    )
    stamina = resources[..., RESOURCE_STAMINA]
    stamina_full = stamina >= rules.resource_maximum[..., RESOURCE_STAMINA]
    # Guard entry, release, and their resource transactions are interaction
    # chain operations.  They advance in ``advance_defense_interactions``
    # after this native stat-regeneration stage; keeping a second activation
    # path here would spend the entry cost and expose Wielding too early.
    broken = state.stamina_broken & ~stamina_full
    immunity_clock = state.control_immunity_regen_clock + dt[..., 0]
    immunity_cycles = jnp.floor(
        immunity_clock / CONTROL_IMMUNITY_REGEN_INTERVAL_SECONDS
    ).astype(jnp.int32)
    immunity_clock = jnp.mod(
        immunity_clock,
        CONTROL_IMMUNITY_REGEN_INTERVAL_SECONDS,
    )
    control_immunity = jnp.maximum(
        jnp.float32(0.0),
        state.control_immunity
        - immunity_cycles.astype(jnp.float32) * CONTROL_IMMUNITY_REGEN_AMOUNT,
    )
    candidate = state._replace(
        resources=resources,
        resource_regen_clock=clock,
        stamina_regen_delay_seconds=delay,
        stamina_regen_delay_clock=delay_clock,
        stamina_broken=broken,
        control_immunity=control_immunity,
        control_immunity_regen_clock=immunity_clock,
        failure_bits=bits,
    )
    return _select_tree(row_valid, candidate, state)._replace(failure_bits=bits)


def tick_statuses(
    state: CombatMechanicsState,
    dt_seconds: jax.Array,
    rules: CombatMechanicsRules,
) -> tuple[CombatMechanicsState, StatusTick]:
    """Advance effects using native cooldown/overlap scheduling semantics."""

    status = state.statuses
    entity_count = state.resources.shape[1]
    dt = _entity_dt(
        dt_seconds,
        state.resources.shape[0],
        entity_count,
    )[..., None]
    active_dt = jnp.minimum(status.remaining_seconds, dt)
    elapsed = status.cycle_elapsed_seconds + active_dt
    periodic = status.cycle_cooldown_seconds > 0.0
    safe_cooldown = jnp.where(
        periodic,
        status.cycle_cooldown_seconds,
        jnp.float32(1.0),
    )
    cycles = jnp.floor(elapsed / safe_cooldown).astype(jnp.int32)
    fires = (
        status.active
        & (active_dt > 0.0)
        & jnp.where(
            periodic,
            cycles > 0,
            ~status.has_cycled,
        )
    )
    cycle_elapsed = jnp.where(
        periodic & (cycles > 0),
        jnp.mod(elapsed, safe_cooldown),
        elapsed,
    )
    remaining = jnp.maximum(
        jnp.float32(0.0),
        status.remaining_seconds - dt,
    )
    active = status.active & (remaining > 0.0)
    flags, speed = status_modifiers(status)
    resource_ids = jnp.clip(status.resource_id, 0, RESOURCE_COUNT - 1)
    resource_one_hot = jax.nn.one_hot(
        resource_ids,
        RESOURCE_COUNT,
        dtype=jnp.float32,
    )
    has_resource = fires & (status.resource_id >= 0)
    resource_delta = jnp.sum(
        resource_one_hot
        * jnp.where(
            has_resource,
            status.resource_delta_per_cycle,
            jnp.float32(0.0),
        )[..., None],
        axis=2,
    )
    resources = jnp.clip(
        state.resources + resource_delta,
        rules.resource_minimum,
        rules.resource_maximum,
    )
    stamina_spent = resource_delta[..., RESOURCE_STAMINA] < 0.0
    newly_broken = stamina_spent & (resources[..., RESOURCE_STAMINA] <= 0.0)
    broken = state.stamina_broken | newly_broken
    delay = jnp.where(
        newly_broken,
        jnp.minimum(
            state.stamina_regen_delay_seconds,
            jnp.float32(STAMINA_BROKEN_REGEN_DELAY_SECONDS),
        ),
        state.stamina_regen_delay_seconds,
    )
    next_status = status._replace(
        remaining_seconds=remaining,
        cycle_elapsed_seconds=cycle_elapsed,
        active=active,
        has_cycled=status.has_cycled | fires,
    )
    candidate = state._replace(
        resources=resources,
        stamina_broken=broken,
        guard_held=state.guard_held & ~broken,
        guard_active=state.guard_active & ~broken,
        guard_windup_elapsed_seconds=jnp.where(
            broken,
            jnp.float32(0.0),
            state.guard_windup_elapsed_seconds,
        ),
        stamina_regen_delay_seconds=delay,
        statuses=next_status,
    )
    row_valid = state.failure_bits == jnp.uint32(0)
    target_ids = jnp.broadcast_to(
        jnp.arange(entity_count, dtype=jnp.int32)[None, :, None],
        fires.shape,
    )
    tick = StatusTick(
        requested=fires & row_valid[:, None, None],
        source_entity_id=status.source_entity_id,
        target_entity_id=target_ids,
        damage=jnp.where(fires, status.damage_per_cycle, 0.0),
        damage_cause=status.damage_cause,
        healing=jnp.where(fires, status.healing_per_cycle, 0.0),
        aggregate_flags=flags,
        speed_multiplier=speed,
    )
    return _select_tree(row_valid, candidate, state), tick
