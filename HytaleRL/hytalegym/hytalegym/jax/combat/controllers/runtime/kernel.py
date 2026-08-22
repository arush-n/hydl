"""Compiled shortbow charge and crossbow fire/reload state machines."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal import (
    CROSSBOW_FAMILY_IDS,
    SHORTBOW_FAMILY_IDS,
)
from hytalegym.jax.combat.controllers.schema.contract import *
from hytalegym.jax.combat.controllers.schema.types import (
    RangedActionMask,
    RangedControllerCommands,
    RangedControllerInfo,
    RangedControllerRules,
    RangedControllerState,
)
from hytalegym.jax.combat.entities import ENTITY_CAPACITY


def ranged_action_mask(
    state: RangedControllerState,
    rules: RangedControllerRules,
) -> RangedActionMask:
    _validate_layout(state, None, rules)
    idle = state.phase == PHASE_IDLE
    durable = state.durability > 0.0
    short = _family_matches(rules.weapon_family, SHORTBOW_FAMILY_IDS)
    cross = _family_matches(rules.weapon_family, CROSSBOW_FAMILY_IDS)
    primary = idle & durable & (
        (short & ((state.signature_charges >= 1.0) | (state.arrow_inventory > 0)))
        | (
            cross
            & (
                (state.signature_charges >= 1.0)
                | (state.ammo > 0)
                | (state.arrow_inventory > 0)
            )
        )
    )
    signature = (
        idle
        & durable
        & (state.signature_charges < 1.0)
        & (state.signature_energy >= rules.signature_energy_cost)
    )
    reload = (
        idle
        & durable
        & cross
        & (state.ammo < rules.max_ammo)
        & (state.arrow_inventory > 0)
    )
    swap = ~(
        (state.phase == PHASE_CROSSBOW_RELOAD_ENTRY)
        | (state.phase == PHASE_CROSSBOW_RELOAD_LOOP)
        | state.pending_inventory_arrow
    )
    legal = jnp.stack((primary, signature, reload, swap), axis=2)
    legal &= rules.active[..., None]
    legal &= (state.failure_bits == 0)[:, None, None]
    return RangedActionMask(legal=legal)


def step_ranged_controllers(
    state: RangedControllerState,
    commands: RangedControllerCommands,
    dt_seconds: jax.Array,
    rules: RangedControllerRules,
) -> tuple[RangedControllerState, RangedControllerInfo]:
    """Advance one engine microtick; failures are sticky and row-atomic."""

    _validate_layout(state, commands, rules)
    original = state
    batch = state.phase.shape[0]
    dt = jnp.asarray(dt_seconds, dtype=jnp.float32)
    if dt.ndim == 0:
        dt = jnp.broadcast_to(dt, (batch,))
    if dt.shape != (batch,):
        raise ValueError(f"dt_seconds must be scalar or [{batch}]")
    dt_entity = dt[:, None]
    state_invalid = _invalid_state(state, rules)
    dt_invalid = ~jnp.isfinite(dt) | (dt <= 0.0) | (
        dt > MAX_CONTROLLER_DT_SECONDS
    )
    edge = commands.primary_held & ~state.primary_was_held
    command_count = (
        edge.astype(jnp.int32)
        + commands.signature_pressed.astype(jnp.int32)
        + commands.reload_pressed.astype(jnp.int32)
        + commands.secondary_pressed.astype(jnp.int32)
        + commands.swap_away.astype(jnp.int32)
        + commands.interrupted.astype(jnp.int32)
    )
    command_invalid = jnp.any(command_count > 1, axis=1)
    unsafe_interrupt = jnp.any(
        state.pending_inventory_arrow
        & (
            edge
            | commands.secondary_pressed
            | commands.swap_away
            | commands.interrupted
        ),
        axis=1,
    )
    bits = state.failure_bits
    bits = _set_failure(
        bits, state_invalid, CONTROLLER_FAILURE_INVALID_STATE
    )
    bits = _set_failure(
        bits, dt_invalid, CONTROLLER_FAILURE_INVALID_DT
    )
    bits = _set_failure(
        bits, command_invalid, CONTROLLER_FAILURE_INVALID_COMMAND
    )
    bits = _set_failure(
        bits,
        unsafe_interrupt,
        CONTROLLER_FAILURE_UNCERTIFIED_INTERRUPT,
    )
    valid = bits == jnp.uint32(0)
    enabled = valid[:, None] & rules.active

    phase = state.phase
    elapsed = jnp.where(
        enabled & (phase != PHASE_IDLE),
        state.phase_elapsed_seconds + dt_entity,
        state.phase_elapsed_seconds,
    )
    cooldown = jnp.maximum(
        0.0, state.primary_cooldown_seconds - dt_entity
    )
    pending = state.pending_inventory_arrow
    arrows = state.arrow_inventory
    add_capacity = state.arrow_add_capacity
    ammo = state.ammo
    durability = state.durability
    energy = state.signature_energy
    charges = state.signature_charges

    ability_slot = jnp.full_like(phase, -1)
    charge_level = jnp.full_like(phase, -1)
    projectile_count = jnp.zeros_like(phase)
    damage_multiplier = jnp.ones_like(elapsed)
    signature_activated = jnp.zeros_like(enabled)
    ammo_loaded = jnp.zeros_like(phase)
    arrows_removed = jnp.zeros_like(phase)
    arrows_returned = jnp.zeros_like(phase)
    arrows_dropped = jnp.zeros_like(phase)
    swap_accepted = jnp.zeros_like(enabled)
    swap_blocked = jnp.zeros_like(enabled)

    cross_reload = (phase == PHASE_CROSSBOW_RELOAD_ENTRY) | (
        phase == PHASE_CROSSBOW_RELOAD_LOOP
    )
    swap_blocked = commands.swap_away & cross_reload & enabled
    safe_cancel = (
        (edge | commands.secondary_pressed | commands.interrupted)
        & (phase != PHASE_IDLE)
        & ~pending
        & enabled
    )
    short_cancel = safe_cancel & (
        (phase == PHASE_SHORTBOW_NOCK)
        | (phase == PHASE_SHORTBOW_CHARGE)
        | (phase == PHASE_SHORTBOW_VOLLEY_CHARGE)
    )
    cross_cancel = safe_cancel & cross_reload
    return_count = jnp.where(short_cancel, ammo, 0)
    returned = jnp.minimum(return_count, add_capacity)
    arrows += returned
    add_capacity -= returned
    arrows_returned += returned
    arrows_dropped += return_count - returned
    ammo = jnp.where(short_cancel, 0, ammo)
    cancel = short_cancel | cross_cancel
    phase = jnp.where(cancel, PHASE_IDLE, phase)
    elapsed = jnp.where(cancel, 0.0, elapsed)

    safe_swap = (
        commands.swap_away
        & enabled
        & ~swap_blocked
        & ~pending
    )
    loaded_to_return = jnp.where(safe_swap, ammo, 0)
    returned = jnp.minimum(loaded_to_return, add_capacity)
    arrows += returned
    add_capacity -= returned
    arrows_returned += returned
    arrows_dropped += loaded_to_return - returned
    ammo = jnp.where(safe_swap, 0, ammo)
    phase = jnp.where(safe_swap, PHASE_IDLE, phase)
    elapsed = jnp.where(safe_swap, 0.0, elapsed)
    swap_accepted = safe_swap

    nock_done = (
        enabled
        & (phase == PHASE_SHORTBOW_NOCK)
        & _reached(elapsed, SHORTBOW_NOCK_SECONDS)
    )
    ammo = jnp.where(nock_done, 1, ammo)
    pending &= ~nock_done
    elapsed = jnp.where(
        nock_done, elapsed - SHORTBOW_NOCK_SECONDS, elapsed
    )
    phase = jnp.where(nock_done, PHASE_SHORTBOW_CHARGE, phase)

    entry_done = (
        enabled
        & (phase == PHASE_CROSSBOW_RELOAD_ENTRY)
        & _reached(elapsed, CROSSBOW_RELOAD_ENTRY_SECONDS)
    )
    can_reserve = entry_done & (ammo < rules.max_ammo) & (arrows > 0)
    arrows -= can_reserve.astype(jnp.int32)
    add_capacity += can_reserve.astype(jnp.int32)
    arrows_removed += can_reserve.astype(jnp.int32)
    pending |= can_reserve
    phase = jnp.where(
        can_reserve,
        PHASE_CROSSBOW_RELOAD_LOOP,
        jnp.where(entry_done, PHASE_IDLE, phase),
    )
    elapsed = jnp.where(
        can_reserve,
        elapsed - CROSSBOW_RELOAD_ENTRY_SECONDS,
        jnp.where(entry_done, 0.0, elapsed),
    )

    loop_done = (
        enabled
        & (phase == PHASE_CROSSBOW_RELOAD_LOOP)
        & pending
        & _reached(elapsed, CROSSBOW_RELOAD_ARROW_SECONDS)
    )
    ammo += loop_done.astype(jnp.int32)
    ammo_loaded += loop_done.astype(jnp.int32)
    pending &= ~loop_done
    loop_elapsed = elapsed - CROSSBOW_RELOAD_ARROW_SECONDS
    reserve_next = (
        loop_done & (ammo < rules.max_ammo) & (arrows > 0)
    )
    arrows -= reserve_next.astype(jnp.int32)
    add_capacity += reserve_next.astype(jnp.int32)
    arrows_removed += reserve_next.astype(jnp.int32)
    pending |= reserve_next
    phase = jnp.where(
        loop_done & ~reserve_next,
        PHASE_IDLE,
        phase,
    )
    elapsed = jnp.where(
        reserve_next,
        loop_elapsed,
        jnp.where(loop_done, 0.0, elapsed),
    )

    release = ~commands.primary_held & state.primary_was_held
    standard_release = (
        enabled & release & (phase == PHASE_SHORTBOW_CHARGE)
    )
    standard_level = (
        jnp.sum(
            elapsed[..., None]
            + jnp.float32(1.0e-6)
            >= jnp.asarray(
                SHORTBOW_CHARGE_THRESHOLDS, dtype=jnp.float32
            ),
            axis=2,
        )
        - 1
    )
    standard_fire = standard_release & (standard_level >= 0) & (ammo > 0)
    too_early = standard_release & ~standard_fire
    returned = jnp.minimum(jnp.where(too_early, ammo, 0), add_capacity)
    arrows += returned
    add_capacity -= returned
    arrows_returned += returned
    arrows_dropped += jnp.where(too_early, ammo, 0) - returned
    ammo = jnp.where(standard_release, 0, ammo)
    ability_slot = jnp.where(
        standard_fire,
        SHORTBOW_PRIMARY_SLOT_BASE + standard_level,
        ability_slot,
    )
    charge_level = jnp.where(
        standard_fire, standard_level, charge_level
    )
    projectile_count += standard_fire.astype(jnp.int32)
    phase = jnp.where(standard_release, PHASE_IDLE, phase)
    elapsed = jnp.where(standard_release, 0.0, elapsed)

    volley_release = (
        enabled & release & (phase == PHASE_SHORTBOW_VOLLEY_CHARGE)
    )
    volley_level = (
        jnp.sum(
            elapsed[..., None]
            + jnp.float32(1.0e-6)
            >= jnp.asarray(
                SHORTBOW_VOLLEY_THRESHOLDS, dtype=jnp.float32
            ),
            axis=2,
        )
        - 1
    )
    volley_fire = volley_release & (charges >= 1.0)
    charges = jnp.where(volley_fire, charges - 1.0, charges)
    ability_slot = jnp.where(
        volley_fire,
        SHORTBOW_VOLLEY_SLOT_BASE + volley_level,
        ability_slot,
    )
    charge_level = jnp.where(volley_fire, volley_level, charge_level)
    projectile_count += volley_fire.astype(jnp.int32) * 3
    phase = jnp.where(volley_release, PHASE_IDLE, phase)
    elapsed = jnp.where(volley_release, 0.0, elapsed)

    idle = phase == PHASE_IDLE
    signature = (
        enabled
        & idle
        & commands.signature_pressed
        & (durability > 0.0)
        & (charges < 1.0)
        & (energy >= rules.signature_energy_cost)
    )
    energy = jnp.where(
        signature, energy - rules.signature_energy_cost, energy
    )
    charges = jnp.where(signature, 1.0, charges)
    signature_slot = jnp.where(
        _family_matches(rules.weapon_family, SHORTBOW_FAMILY_IDS),
        SHORTBOW_SIGNATURE_ACTIVATE_SLOT,
        CROSSBOW_SIGNATURE_ACTIVATE_SLOT,
    )
    ability_slot = jnp.where(signature, signature_slot, ability_slot)
    signature_activated = signature

    idle = phase == PHASE_IDLE
    short_start = (
        enabled
        & idle
        & edge
        & _family_matches(rules.weapon_family, SHORTBOW_FAMILY_IDS)
        & (durability > 0.0)
        & ~signature
    )
    volley_start = short_start & (charges >= 1.0)
    nock_start = short_start & ~volley_start & (arrows > 0)
    arrows -= nock_start.astype(jnp.int32)
    add_capacity += nock_start.astype(jnp.int32)
    arrows_removed += nock_start.astype(jnp.int32)
    ammo = jnp.where(nock_start, 0, ammo)
    pending |= nock_start
    phase = jnp.where(
        volley_start,
        PHASE_SHORTBOW_VOLLEY_CHARGE,
        jnp.where(nock_start, PHASE_SHORTBOW_NOCK, phase),
    )
    elapsed = jnp.where(volley_start | nock_start, 0.0, elapsed)

    idle = phase == PHASE_IDLE
    cross = enabled & _family_matches(rules.weapon_family, CROSSBOW_FAMILY_IDS)
    cross_reload_start = (
        cross
        & idle
        & commands.reload_pressed
        & (durability > 0.0)
        & (ammo < rules.max_ammo)
        & (arrows > 0)
    )
    phase = jnp.where(
        cross_reload_start, PHASE_CROSSBOW_RELOAD_ENTRY, phase
    )
    elapsed = jnp.where(cross_reload_start, 0.0, elapsed)

    idle = phase == PHASE_IDLE
    auto_primary = (
        cross
        & idle
        & commands.primary_held
        & (cooldown <= jnp.float32(1.0e-6))
        & (durability > 0.0)
        & ~commands.reload_pressed
        & ~signature
        & ~safe_swap
    )
    big_arrow = auto_primary & (charges >= 1.0)
    standard = auto_primary & ~big_arrow & (ammo > 0)
    start_empty_reload = (
        auto_primary
        & ~big_arrow
        & ~standard
        & (arrows > 0)
    )
    charges = jnp.where(big_arrow, charges - 1.0, charges)
    ammo = jnp.where(standard, ammo - 1, ammo)
    ability_slot = jnp.where(
        big_arrow,
        CROSSBOW_BIG_ARROW_SLOT,
        jnp.where(standard, CROSSBOW_PRIMARY_SLOT, ability_slot),
    )
    projectile_count += (big_arrow | standard).astype(jnp.int32)
    cooldown = jnp.where(
        big_arrow | standard,
        CROSSBOW_PRIMARY_COOLDOWN_SECONDS,
        cooldown,
    )
    phase = jnp.where(
        start_empty_reload, PHASE_CROSSBOW_RELOAD_ENTRY, phase
    )
    elapsed = jnp.where(start_empty_reload, 0.0, elapsed)

    launch = projectile_count > 0
    durability_loss = (
        projectile_count.astype(jnp.float32)
        * rules.durability_loss_per_projectile
    )
    prior_durability = durability
    durability = jnp.where(
        launch,
        jnp.maximum(0.0, durability - durability_loss),
        durability,
    )
    broke_on_launch = launch & (prior_durability > 0.0) & (
        durability <= 0.0
    )
    damage_multiplier = jnp.where(
        broke_on_launch,
        BROKEN_WEAPON_DAMAGE_MULTIPLIER,
        damage_multiplier,
    )
    ability_requested = ability_slot >= 0
    candidate = RangedControllerState(
        phase=phase,
        phase_elapsed_seconds=elapsed,
        primary_was_held=commands.primary_held,
        primary_cooldown_seconds=cooldown,
        pending_inventory_arrow=pending,
        arrow_inventory=arrows,
        arrow_add_capacity=add_capacity,
        ammo=ammo,
        durability=durability,
        signature_energy=energy,
        signature_charges=charges,
        failure_bits=bits,
    )
    result = _select_tree(valid, candidate, original)._replace(
        failure_bits=bits,
        primary_was_held=jnp.where(
            valid[:, None],
            commands.primary_held,
            original.primary_was_held,
        ),
    )
    info_mask = valid[:, None]
    return result, RangedControllerInfo(
        ability_slot=jnp.where(info_mask, ability_slot, -1),
        ability_requested=ability_requested & info_mask,
        charge_level=jnp.where(info_mask, charge_level, -1),
        projectile_count=jnp.where(info_mask, projectile_count, 0),
        projectile_damage_multiplier=jnp.where(
            info_mask, damage_multiplier, 1.0
        ),
        signature_activated=signature_activated & info_mask,
        ammo_loaded=jnp.where(info_mask, ammo_loaded, 0),
        arrows_removed=jnp.where(info_mask, arrows_removed, 0),
        arrows_returned=jnp.where(info_mask, arrows_returned, 0),
        arrows_dropped=jnp.where(info_mask, arrows_dropped, 0),
        swap_accepted=swap_accepted & info_mask,
        swap_blocked=swap_blocked & info_mask,
        failure_bits=bits,
        valid=valid,
    )


def movement_speed_multiplier(
    state: RangedControllerState,
) -> jax.Array:
    charge = (
        (state.phase == PHASE_SHORTBOW_CHARGE)
        | (state.phase == PHASE_SHORTBOW_VOLLEY_CHARGE)
    )
    reload_loop = state.phase == PHASE_CROSSBOW_RELOAD_LOOP
    return jnp.where(
        charge,
        SHORTBOW_CHARGE_SPEED_MULTIPLIER,
        jnp.where(
            reload_loop,
            CROSSBOW_RELOAD_SPEED_MULTIPLIER,
            1.0,
        ),
    ).astype(jnp.float32)


def _validate_layout(state, commands, rules):
    batch = state.phase.shape[0]
    shape = (batch, ENTITY_CAPACITY)
    state_fields = {
        "phase": jnp.int32,
        "phase_elapsed_seconds": jnp.float32,
        "primary_was_held": jnp.bool_,
        "primary_cooldown_seconds": jnp.float32,
        "pending_inventory_arrow": jnp.bool_,
        "arrow_inventory": jnp.int32,
        "arrow_add_capacity": jnp.int32,
        "ammo": jnp.int32,
        "durability": jnp.float32,
        "signature_energy": jnp.float32,
        "signature_charges": jnp.float32,
    }
    for name, dtype in state_fields.items():
        _field(getattr(state, name), shape, dtype, f"state.{name}")
    _field(state.failure_bits, (batch,), jnp.uint32, "state.failure_bits")
    rule_fields = {
        "weapon_family": jnp.int32,
        "active": jnp.bool_,
        "max_ammo": jnp.int32,
        "max_durability": jnp.float32,
        "durability_loss_per_projectile": jnp.float32,
        "signature_energy_cost": jnp.float32,
    }
    for name, dtype in rule_fields.items():
        _field(getattr(rules, name), shape, dtype, f"rules.{name}")
    if commands is not None:
        for name in commands._fields:
            _field(
                getattr(commands, name),
                shape,
                jnp.bool_,
                f"commands.{name}",
            )


def _invalid_state(state, rules):
    phase_invalid = (state.phase < 0) | (state.phase >= PHASE_COUNT)
    inactive_dirty = ~rules.active & (
        (state.phase != PHASE_IDLE)
        | state.pending_inventory_arrow
        | (state.ammo != 0)
        | (state.durability != 0.0)
        | (state.signature_energy != 0.0)
        | (state.signature_charges != 0.0)
    )
    entity_invalid = (
        phase_invalid
        | ~jnp.isfinite(state.phase_elapsed_seconds)
        | (state.phase_elapsed_seconds < 0.0)
        | ~jnp.isfinite(state.primary_cooldown_seconds)
        | (state.primary_cooldown_seconds < 0.0)
        | (state.arrow_inventory < 0)
        | (state.arrow_add_capacity < 0)
        | (state.ammo < 0)
        | (state.ammo > rules.max_ammo)
        | ~jnp.isfinite(state.durability)
        | (state.durability < 0.0)
        | (state.durability > rules.max_durability)
        | ~jnp.isfinite(state.signature_energy)
        | (state.signature_energy < 0.0)
        | (state.signature_energy > rules.signature_energy_cost)
        | ~jnp.isfinite(state.signature_charges)
        | (state.signature_charges < 0.0)
        | (state.signature_charges > 1.0)
        | (state.pending_inventory_arrow & (state.arrow_add_capacity <= 0))
        | inactive_dirty
    )
    return jnp.any(entity_invalid, axis=1)


def _field(value, shape, dtype, name):
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}")


def _set_failure(bits, mask, code):
    return jnp.where(mask, bits | jnp.uint32(code), bits)


def _reached(value, threshold):
    return value + jnp.float32(1.0e-6) >= jnp.float32(threshold)


def _family_matches(family, candidates):
    result = jnp.zeros_like(family, dtype=jnp.bool_)
    for candidate in candidates:
        result |= family == jnp.int32(candidate)
    return result


def _select_tree(mask, selected, fallback):
    return jax.tree_util.tree_map(
        lambda yes, no: jnp.where(
            mask.reshape((mask.shape[0],) + (1,) * (yes.ndim - 1)),
            yes,
            no,
        ),
        selected,
        fallback,
    )
