"""One compiled, fixed-capacity ability transition for 32 logical entities."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal import (
    ABILITY_CAPACITY,
    EVENT_CAPACITY,
    INTERACTION_QUEUE_DELAY_TICKS,
)
from hytalegym.jax.combat.arsenal.programs.scheduling import (
    advance_scheduler_clocks,
    event_clock_crossed,
    event_dispatch_delay_ticks,
    scheduler_clock,
)
from hytalegym.jax.combat.entities import (
    ENTITY_CAPACITY,
    invalid_roster_rows,
)
from hytalegym.jax.combat.entities.abilities.schema.contract import (
    ABILITY_CAPABILITIES,
    ABILITY_EVENT_CAPACITY,
    ABILITY_FAILURE_AVAILABILITY,
    ABILITY_FAILURE_EVENT_OVERFLOW,
    ABILITY_FAILURE_INVALID_COMMAND,
    ABILITY_FAILURE_INVALID_DT,
    ABILITY_FAILURE_INVALID_PROGRAM,
    ABILITY_FAILURE_INVALID_STATE,
    ABILITY_FAILURE_STALE_SOURCE,
    ABILITY_FAILURE_UPSTREAM,
    ABILITY_PROGRAM_FAMILY_CAPACITY,
)
from hytalegym.jax.combat.entities.abilities.schema.types import (
    EntityAbilityEvents,
    EntityAbilityStepInfo,
)
from hytalegym.jax.combat.entities.abilities.schema.validation import (
    validate_entity_ability_step_layout,
)
from hytalegym.jax.combat.entities.effects import (
    invalid_effect_rows,
    stale_effect_source_rows,
)
from hytalegym.jax.combat.mechanics import (
    RESOURCE_STAMINA,
    STAMINA_BROKEN_REGEN_DELAY_SECONDS,
    STATUS_FLAG_DISABLE_ABILITIES,
    status_modifiers,
)


def step_entity_abilities(
    state,
    effects,
    commands,
    availability,
    programs,
    dt_seconds,
    rules,
):
    """Start, advance, and pack every ability event in one atomic JAX step."""

    validate_entity_ability_step_layout(
        state,
        effects,
        commands,
        availability,
        programs,
        rules,
    )
    batch = state.weapon_family.shape[0]
    dt = _batch_dt(dt_seconds, batch)
    roster = effects.combat.roster
    mechanics = effects.combat.mechanics
    family = state.weapon_family
    safe_family = jnp.clip(family, 0, ABILITY_PROGRAM_FAMILY_CAPACITY - 1)
    equipped = family > 0
    state_active = state.active_ability_slot >= 0
    safe_state_active = jnp.clip(
        state.active_ability_slot,
        0,
        ABILITY_CAPACITY - 1,
    )
    state_prelude_ticks = _gather_ability(
        programs.ability_scheduler_prelude_ticks[safe_family],
        safe_state_active,
    )
    living = roster.active & roster.damageable & ~roster.dead & (roster.health > 0.0)

    bits = state.failure_bits
    world_bits = state.world_failure_bits
    invalid_state = (
        invalid_roster_rows(roster)
        | invalid_effect_rows(effects)
        | stale_effect_source_rows(effects)
        | jnp.any(
            (family < 0)
            | (family >= ABILITY_PROGRAM_FAMILY_CAPACITY)
            | (
                ~equipped
                & (
                    (state.source_generation != jnp.uint32(0))
                    | (state.active_ability_slot >= 0)
                )
            )
            | (
                equipped
                & (
                    (state.source_generation == jnp.uint32(0))
                    | (state.active_ability_slot >= ABILITY_CAPACITY)
                )
            )
            | (state.active_ability_slot < -1)
            | ~jnp.isfinite(state.ability_elapsed_seconds)
            | (state.ability_elapsed_seconds < 0.0)
            | (
                ~state_active
                & (state.ability_scheduler_tick != jnp.int32(0))
            )
            | (
                state_active
                & (
                    state.ability_scheduler_tick
                    < -state_prelude_ticks
                )
            )
            | jnp.any(
                ~jnp.isfinite(state.ability_scheduler_clock_seconds)
                | (state.ability_scheduler_clock_seconds < 0.0),
                axis=2,
            )
            | jnp.any(
                ~jnp.isfinite(state.ability_cooldown_seconds)
                | (state.ability_cooldown_seconds < 0.0),
                axis=2,
            ),
            axis=1,
        )
        | jnp.any(
            ~jnp.isfinite(mechanics.resources),
            axis=(1, 2),
        )
        | (state.capability_bits != jnp.uint32(ABILITY_CAPABILITIES))
    )
    bits = _set_failure(bits, invalid_state, ABILITY_FAILURE_INVALID_STATE)
    bits = _set_failure(
        bits,
        ~jnp.isfinite(dt) | (dt < 0.0),
        ABILITY_FAILURE_INVALID_DT,
    )
    program_invalid = equipped & (
        ~programs.family_mask[safe_family] | programs.overflow[safe_family]
    )
    bits = _set_failure(
        bits,
        jnp.any(program_invalid, axis=1),
        ABILITY_FAILURE_INVALID_PROGRAM,
    )
    stale_source = equipped & (
        ~roster.active | (state.source_generation != roster.generation)
    )
    bits = _set_failure(
        bits,
        jnp.any(stale_source, axis=1),
        ABILITY_FAILURE_STALE_SOURCE,
    )
    upstream = (
        ~commands.valid
        | (commands.failure_bits != jnp.uint32(0))
        | (effects.failure_bits != jnp.uint32(0))
        | (roster.failure_bits != jnp.uint32(0))
        | (mechanics.failure_bits != jnp.uint32(0))
    )
    bits = _set_failure(bits, upstream, ABILITY_FAILURE_UPSTREAM)

    interrupted = commands.interrupted & living
    active_slot = jnp.where(
        interrupted | ~living,
        jnp.int32(-1),
        state.active_ability_slot,
    )
    scheduler_tick = jnp.where(
        interrupted | ~living,
        jnp.int32(0),
        state.ability_scheduler_tick,
    )
    scheduler_clocks = jnp.where(
        (interrupted | ~living)[..., None],
        jnp.float32(0.0),
        state.ability_scheduler_clock_seconds,
    )
    requested_slot = jnp.where(interrupted, jnp.int32(-1), commands.requested_slot)
    requested = requested_slot >= 0
    invalid_command = requested_slot >= ABILITY_CAPACITY
    bits = _set_failure(
        bits,
        jnp.any(invalid_command, axis=1),
        ABILITY_FAILURE_INVALID_COMMAND,
    )
    slot = jnp.clip(requested_slot, 0, ABILITY_CAPACITY - 1)

    ability_mask = programs.ability_mask[safe_family]
    minimum = programs.ability_resource_minimum[safe_family]
    requirements = programs.ability_requirements[safe_family]
    available = availability.available_requirement_bits[..., None]
    generation_matches = availability.source_generation == state.source_generation
    requirement_available = (requirements == jnp.uint32(0)) | (
        availability.valid[..., None]
        & (availability.failure_bits[..., None] == jnp.uint32(0))
        & generation_matches[..., None]
        & ((requirements & ~available) == jnp.uint32(0))
    )
    requested_profile = requested & _gather_ability(ability_mask, slot)
    requested_requirements = _gather_ability(requirements, slot)
    requested_availability = requested_profile & (
        requested_requirements != jnp.uint32(0)
    )
    unavailable = requested_availability & ~_gather_ability(requirement_available, slot)
    bits = _set_failure(
        bits,
        jnp.any(unavailable, axis=1),
        ABILITY_FAILURE_AVAILABILITY,
    )
    world_bits |= jnp.bitwise_or.reduce(
        jnp.where(
            requested_availability,
            availability.failure_bits,
            jnp.uint32(0),
        ),
        axis=1,
    )

    flags, _ = status_modifiers(mechanics.statuses)
    status_allows = (flags & jnp.uint32(STATUS_FLAG_DISABLE_ABILITIES)) == 0
    resource_enough = jnp.all(
        mechanics.resources[..., None, :] + jnp.float32(1.0e-6) >= minimum,
        axis=3,
    )
    legal_mask = (
        ability_mask
        & equipped[..., None]
        & ~programs.overflow[safe_family][..., None]
        & (active_slot[..., None] < 0)
        & (state.ability_cooldown_seconds <= 0.0)
        & resource_enough
        & requirement_available
        & living[..., None]
        & status_allows[..., None]
        & (bits == jnp.uint32(0))[:, None, None]
        & (world_bits == jnp.uint32(0))[:, None, None]
    )
    accepted = (
        requested
        & _gather_ability(legal_mask, slot)
        & (bits == jnp.uint32(0))[:, None]
        & (world_bits == jnp.uint32(0))[:, None]
    )
    guard_ended = mechanics.guard_active & accepted
    delay_value = _gather_ability(
        programs.ability_stamina_regen_delay_seconds[safe_family],
        slot,
    )
    delay = mechanics.stamina_regen_delay_seconds
    delay = jnp.where(
        guard_ended,
        jnp.minimum(delay, rules.guard_exit_regen_delay_seconds),
        delay,
    )
    delay = jnp.where(
        accepted & (delay_value < 0.0),
        jnp.minimum(delay, delay_value),
        delay,
    )
    cooldown_value = _gather_ability(
        programs.ability_cooldown_seconds[safe_family],
        slot,
    )
    cooldowns = _set_ability(
        state.ability_cooldown_seconds,
        slot,
        jnp.where(
            accepted,
            cooldown_value,
            _gather_ability(state.ability_cooldown_seconds, slot),
        ),
    )
    active_slot = jnp.where(accepted, slot, active_slot)
    scheduler_prelude_ticks = _gather_ability(
        programs.ability_scheduler_prelude_ticks[safe_family],
        slot,
    )
    scheduler_tick = jnp.where(
        accepted,
        -scheduler_prelude_ticks,
        scheduler_tick,
    )
    scheduler_clocks = jnp.where(
        accepted[..., None],
        jnp.float32(0.0),
        scheduler_clocks,
    )

    dt_entity = jnp.broadcast_to(dt[:, None], (batch, ENTITY_CAPACITY))
    active = active_slot >= 0
    safe_active = jnp.clip(active_slot, 0, ABILITY_CAPACITY - 1)
    prior_clocks = scheduler_clocks
    after_clocks = advance_scheduler_clocks(
        active,
        scheduler_tick,
        prior_clocks,
        dt_entity,
    )
    active_cost = _gather_ability(
        programs.ability_resource_cost[safe_family],
        safe_active,
    )
    resource_commit_time = _gather_ability(
        programs.ability_resource_commit_time_seconds[safe_family],
        safe_active,
    )
    resource_commit_delay = event_dispatch_delay_ticks(
        resource_commit_time,
        _gather_ability(
            programs.ability_resource_commit_flags[safe_family],
            safe_active,
        ),
    )
    resource_prior = scheduler_clock(
        prior_clocks,
        resource_commit_delay,
    )
    resource_after = scheduler_clock(
        after_clocks,
        resource_commit_delay,
    )
    resource_clock_advanced = resource_after > resource_prior
    resource_crossed_commit = (
        (
            (resource_commit_time == jnp.float32(0.0))
            & (resource_prior == jnp.float32(0.0))
        )
        | (
            (resource_commit_time > resource_prior)
            & (resource_commit_time <= resource_after)
        )
    )
    resource_due = (
        active[..., None]
        & (active_cost > jnp.float32(0.0))
        & resource_clock_advanced
        & resource_crossed_commit
    )
    resources = jnp.clip(
        mechanics.resources
        - jnp.where(
            resource_due,
            active_cost,
            jnp.float32(0.0),
        ),
        rules.resource_minimum,
        rules.resource_maximum,
    )
    stamina = resources[..., RESOURCE_STAMINA]
    spent_stamina = (
        resource_due[..., RESOURCE_STAMINA]
        & (active_cost[..., RESOURCE_STAMINA] > 0.0)
    )
    newly_broken = spent_stamina & (stamina <= 0.0)
    stamina_broken = mechanics.stamina_broken | newly_broken
    delay = jnp.where(
        newly_broken,
        jnp.minimum(
            delay,
            jnp.float32(STAMINA_BROKEN_REGEN_DELAY_SECONDS),
        ),
        delay,
    )
    mechanics_candidate = mechanics._replace(
        resources=resources,
        stamina_broken=stamina_broken,
        guard_held=mechanics.guard_held & ~accepted & ~stamina_broken,
        guard_active=mechanics.guard_active & ~accepted & ~stamina_broken,
        guard_windup_elapsed_seconds=jnp.where(
            accepted | stamina_broken,
            jnp.float32(0.0),
            mechanics.guard_windup_elapsed_seconds,
        ),
        stamina_regen_delay_seconds=delay,
    )
    event_mask = _gather_ability(programs.event_mask[safe_family], safe_active)
    event_time = _gather_ability(programs.event_time_seconds[safe_family], safe_active)
    event_flags = _gather_ability(
        programs.event_flags[safe_family],
        safe_active,
    )
    event_delay = event_dispatch_delay_ticks(event_time, event_flags)
    event_prior = scheduler_clock(prior_clocks, event_delay)
    event_after = scheduler_clock(after_clocks, event_delay)
    fires = (
        active[..., None]
        & event_mask
        & event_clock_crossed(event_prior, event_after, event_time)
        & (bits == jnp.uint32(0))[:, None, None]
        & (world_bits == jnp.uint32(0))[:, None, None]
    )
    duration = _gather_ability(
        programs.ability_duration_seconds[safe_family],
        safe_active,
    )
    event_program_delay = jnp.max(
        jnp.where(
            event_mask,
            event_delay,
            jnp.int32(INTERACTION_QUEUE_DELAY_TICKS),
        ),
        axis=2,
    )
    has_resource_cost = active_cost > jnp.float32(0.0)
    resource_program_delay = jnp.max(
        jnp.where(
            has_resource_cost,
            resource_commit_delay,
            jnp.int32(INTERACTION_QUEUE_DELAY_TICKS),
        ),
        axis=2,
    )
    program_delay = jnp.maximum(
        event_program_delay,
        resource_program_delay,
    )
    duration = jnp.maximum(
        duration,
        jnp.max(
            jnp.where(
                has_resource_cost,
                resource_commit_time,
                jnp.float32(0.0),
            ),
            axis=2,
        ),
    )
    program_after = scheduler_clock(after_clocks, program_delay)
    finished = active & (program_after >= duration)
    cooldowns = jnp.maximum(
        jnp.float32(0.0),
        cooldowns - dt[:, None, None],
    )
    candidate_state = state._replace(
        active_ability_slot=jnp.where(finished, jnp.int32(-1), active_slot),
        ability_elapsed_seconds=jnp.where(
            finished | ~active,
            jnp.float32(0.0),
            program_after,
        ),
        ability_scheduler_tick=jnp.where(
            finished | ~active,
            jnp.int32(0),
            scheduler_tick + jnp.int32(1),
        ),
        ability_scheduler_clock_seconds=jnp.where(
            (finished | ~active)[..., None],
            jnp.float32(0.0),
            after_clocks,
        ),
        ability_cooldown_seconds=cooldowns,
    )
    events, count, overflow = _pack_events(
        fires,
        state.source_generation,
        family,
        safe_active,
        _gather_ability(programs.event_kind[safe_family], safe_active),
        _gather_ability(programs.event_f32[safe_family], safe_active),
        _gather_ability(programs.event_i32[safe_family], safe_active),
        event_flags,
    )
    bits = _set_failure(bits, overflow, ABILITY_FAILURE_EVENT_OVERFLOW)
    valid = (bits == jnp.uint32(0)) & (world_bits == jnp.uint32(0))
    result_state = _select_tree(valid, candidate_state, state)._replace(
        failure_bits=bits,
        world_failure_bits=world_bits,
    )
    result_effects = effects._replace(
        combat=effects.combat._replace(
            mechanics=_select_tree(valid, mechanics_candidate, mechanics)
        )
    )
    events = events._replace(
        requested=events.requested & valid[:, None],
        overflow=overflow,
        failure_bits=bits,
        world_failure_bits=world_bits,
        valid=valid,
    )
    info = EntityAbilityStepInfo(
        legal_mask=legal_mask & valid[:, None, None],
        requested=requested & valid[:, None],
        accepted=accepted & valid[:, None],
        interrupted=interrupted & valid[:, None],
        event_count=jnp.where(valid, count, jnp.int32(0)),
        failure_bits=bits,
        world_failure_bits=world_bits,
        valid=valid,
    )
    return result_state, result_effects, events, info


def _pack_events(
    fires,
    generation,
    family,
    ability_slot,
    kind,
    f32,
    i32,
    flags,
):
    batch = fires.shape[0]
    flat_size = ENTITY_CAPACITY * EVENT_CAPACITY
    flat = fires.reshape((batch, flat_size))
    count = jnp.sum(flat.astype(jnp.int32), axis=1)
    overflow = count > ABILITY_EVENT_CAPACITY
    score, index = jax.lax.top_k(
        jnp.where(
            flat,
            flat_size - jnp.arange(flat_size, dtype=jnp.int32)[None, :],
            jnp.int32(-1),
        ),
        ABILITY_EVENT_CAPACITY,
    )
    requested = (score >= 0) & ~overflow[:, None]
    source = index // EVENT_CAPACITY
    event = index % EVENT_CAPACITY
    batch_index = jnp.arange(batch, dtype=jnp.int32)[:, None]
    source_generation = generation[batch_index, source]
    weapon_family = family[batch_index, source]
    ability = ability_slot[batch_index, source]
    return (
        EntityAbilityEvents(
            requested=requested,
            source_slot=source,
            source_generation=source_generation,
            weapon_family=weapon_family,
            ability_slot=ability,
            event_slot=event,
            kind=kind[batch_index, source, event],
            f32=f32[batch_index, source, event],
            i32=i32[batch_index, source, event],
            flags=flags[batch_index, source, event],
            overflow=overflow,
            failure_bits=jnp.zeros((batch,), dtype=jnp.uint32),
            world_failure_bits=jnp.zeros((batch,), dtype=jnp.uint32),
            valid=jnp.ones((batch,), dtype=jnp.bool_),
        ),
        count,
        overflow,
    )


def _gather_ability(values, slot):
    trailing = values.shape[3:]
    index = slot[..., None].reshape(slot.shape + (1,) * (values.ndim - 2))
    index = jnp.broadcast_to(index, slot.shape + (1,) + trailing)
    return jnp.take_along_axis(values, index, axis=2)[:, :, 0]


def _set_ability(values, slot, replacement):
    one_hot = jax.nn.one_hot(slot, values.shape[2], dtype=jnp.bool_)
    return jnp.where(one_hot, replacement[..., None], values)


def _batch_dt(value, batch):
    result = jnp.asarray(value, dtype=jnp.float32)
    if result.ndim == 0:
        result = jnp.broadcast_to(result, (batch,))
    if result.shape != (batch,):
        raise ValueError(f"dt_seconds must be scalar or [{batch}]")
    return result


def _set_failure(bits, mask, code):
    return jnp.where(mask, bits | jnp.uint32(code), bits)


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
