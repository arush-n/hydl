"""Pure-JAX entity-only projectile and AABB-hazard mechanics."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.effects.schema.contract import (
    EFFECT_FAILURE_HAZARD_OVERFLOW,
    EFFECT_FAILURE_INVALID_COMMAND,
    EFFECT_FAILURE_INVALID_STATE,
    EFFECT_FAILURE_PROJECTILE_OVERFLOW,
    EFFECT_FAILURE_UNSUPPORTED_COLLISION,
)
from hytalegym.jax.combat.effects.schema.types import (
    CombatEffectCommands,
    CombatEffectsState,
    HazardState,
    ProjectileState,
)
from hytalegym.jax.combat.types import (
    AGENT_ENTITY,
    TARGET_ENTITY,
    CombatParams,
    CombatState,
    RewardComponents,
)


def apply_effect_commands(
    state: CombatEffectsState,
    commands: CombatEffectCommands,
) -> tuple[CombatEffectsState, jax.Array, jax.Array]:
    """Allocate at most one projectile and hazard per environment row."""

    prior_valid = state.failure_bits == jnp.uint32(0)
    projectile_requested = commands.projectile.requested & prior_valid
    hazard_requested = commands.hazard.requested & prior_valid
    projectile_valid = _projectile_command_valid(commands)
    hazard_valid = _hazard_command_valid(commands)
    projectile_available = jnp.any(~state.projectiles.active, axis=1)
    hazard_available = jnp.any(~state.hazards.active, axis=1)

    invalid = (projectile_requested & ~projectile_valid) | (
        hazard_requested & ~hazard_valid
    )
    unsupported = (
        projectile_requested
        & projectile_valid
        & ~commands.projectile.entity_collision_only
    ) | (hazard_requested & hazard_valid & ~commands.hazard.entity_overlap_only)
    projectile_overflow = (
        projectile_requested
        & projectile_valid
        & commands.projectile.entity_collision_only
        & ~projectile_available
    )
    hazard_overflow = (
        hazard_requested
        & hazard_valid
        & commands.hazard.entity_overlap_only
        & ~hazard_available
    )
    failure_bits = state.failure_bits
    failure_bits = _set_failure(
        failure_bits,
        invalid,
        EFFECT_FAILURE_INVALID_COMMAND,
    )
    failure_bits = _set_failure(
        failure_bits,
        unsupported,
        EFFECT_FAILURE_UNSUPPORTED_COLLISION,
    )
    failure_bits = _set_failure(
        failure_bits,
        projectile_overflow,
        EFFECT_FAILURE_PROJECTILE_OVERFLOW,
    )
    failure_bits = _set_failure(
        failure_bits,
        hazard_overflow,
        EFFECT_FAILURE_HAZARD_OVERFLOW,
    )
    row_valid = failure_bits == jnp.uint32(0)
    projectile_accepted = (
        projectile_requested & projectile_valid & projectile_available & row_valid
    )
    hazard_accepted = hazard_requested & hazard_valid & hazard_available & row_valid
    projectiles = _spawn_projectile(
        state.projectiles,
        commands,
        projectile_accepted,
    )
    hazards = _spawn_hazard(
        state.hazards,
        commands,
        hazard_accepted,
    )
    return (
        state._replace(
            projectiles=projectiles,
            hazards=hazards,
            failure_bits=failure_bits,
        ),
        projectile_accepted,
        hazard_accepted,
    )


def tick_effects(
    combat: CombatState,
    effects: CombatEffectsState,
    motion_delta: jax.Array,
    params: CombatParams,
) -> tuple[
    CombatState,
    CombatEffectsState,
    RewardComponents,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Advance one Hytale engine microtick after the calibrated melee tick."""

    invalid, unsupported = _state_failures(effects)
    failure_bits = _set_failure(
        effects.failure_bits,
        invalid,
        EFFECT_FAILURE_INVALID_STATE,
    )
    failure_bits = _set_failure(
        failure_bits,
        unsupported,
        EFFECT_FAILURE_UNSUPPORTED_COLLISION,
    )
    valid = failure_bits == jnp.uint32(0)
    effects = effects._replace(failure_bits=failure_bits)

    projectiles, projectile_hit, projectile_damage = _tick_projectiles(
        combat,
        effects.projectiles,
        motion_delta,
        params,
    )
    hazards, hazard_hit, hazard_damage = _tick_hazards(
        combat,
        effects.hazards,
        motion_delta,
        params,
    )
    candidate_effects = effects._replace(
        projectiles=projectiles,
        hazards=hazards,
    )
    (
        candidate_combat,
        components,
        projectile_actual,
        hazard_actual,
    ) = _apply_effect_damage(
        combat,
        projectile_damage,
        hazard_damage,
    )
    combat_result = _select_tree(valid, candidate_combat, combat)
    effects_result = _select_tree(valid, candidate_effects, effects)
    zero_f = jnp.float32(0.0)
    zero_i = jnp.int32(0)
    return (
        combat_result,
        effects_result,
        jax.tree_util.tree_map(
            lambda value: jnp.where(valid, value, jnp.zeros_like(value)),
            components,
        ),
        jnp.where(valid, projectile_hit, zero_i),
        jnp.where(valid, hazard_hit, zero_i),
        jnp.where(valid, projectile_actual, zero_f),
        jnp.where(valid, hazard_actual, zero_f),
    )


def _spawn_projectile(
    state: ProjectileState,
    commands: CombatEffectCommands,
    accepted: jax.Array,
) -> ProjectileState:
    slot = jnp.argmax(~state.active, axis=1)
    selected = (
        jax.nn.one_hot(
            slot,
            state.active.shape[1],
            dtype=jnp.bool_,
        )
        & accepted[:, None]
    )
    command = commands.projectile
    zeros = jnp.zeros_like(command.damage)
    false = jnp.zeros_like(command.requested)
    return state._replace(
        position=_write_slot(state.position, command.position, selected),
        velocity=_write_slot(state.velocity, command.velocity, selected),
        half_extent=_write_slot(
            state.half_extent,
            command.half_extent,
            selected,
        ),
        age_seconds=_write_slot(state.age_seconds, zeros, selected),
        despawn_seconds=_write_slot(
            state.despawn_seconds,
            command.despawn_seconds,
            selected,
        ),
        authored_lifetime_seconds=_write_slot(
            state.authored_lifetime_seconds,
            command.authored_lifetime_seconds,
            selected,
        ),
        damage=_write_slot(state.damage, command.damage, selected),
        gravity=_write_slot(state.gravity, command.gravity, selected),
        terminal_velocity=_write_slot(
            state.terminal_velocity,
            command.terminal_velocity,
            selected,
        ),
        dead_time_seconds=_write_slot(
            state.dead_time_seconds,
            command.dead_time_seconds,
            selected,
        ),
        dead_time_remaining=_write_slot(
            state.dead_time_remaining,
            zeros,
            selected,
        ),
        velocity_scale=_write_slot(
            state.velocity_scale,
            command.velocity_scale,
            selected,
        ),
        kind=_write_slot(state.kind, command.kind, selected),
        owner_entity_id=_write_slot(
            state.owner_entity_id,
            command.owner_entity_id,
            selected,
        ),
        flags=_write_slot(state.flags, command.flags, selected),
        hostile=_write_slot(state.hostile, command.hostile, selected),
        active=state.active | selected,
        impacted=_write_slot(state.impacted, false, selected),
        physics_initialized=_write_slot(
            state.physics_initialized,
            false,
            selected,
        ),
        entity_collision_only=_write_slot(
            state.entity_collision_only,
            command.entity_collision_only,
            selected,
        ),
    )


def _spawn_hazard(
    state: HazardState,
    commands: CombatEffectCommands,
    accepted: jax.Array,
) -> HazardState:
    slot = jnp.argmax(~state.active, axis=1)
    selected = (
        jax.nn.one_hot(
            slot,
            state.active.shape[1],
            dtype=jnp.bool_,
        )
        & accepted[:, None]
    )
    command = commands.hazard
    zeros = jnp.zeros_like(command.duration_seconds)
    return state._replace(
        center=_write_slot(state.center, command.center, selected),
        half_extent=_write_slot(
            state.half_extent,
            command.half_extent,
            selected,
        ),
        age_seconds=_write_slot(state.age_seconds, zeros, selected),
        duration_seconds=_write_slot(
            state.duration_seconds,
            command.duration_seconds,
            selected,
        ),
        damage_per_second=_write_slot(
            state.damage_per_second,
            command.damage_per_second,
            selected,
        ),
        intensity=_write_slot(
            state.intensity,
            command.intensity,
            selected,
        ),
        activation=_write_slot(
            state.activation,
            command.activation,
            selected,
        ),
        kind=_write_slot(state.kind, command.kind, selected),
        owner_entity_id=_write_slot(
            state.owner_entity_id,
            command.owner_entity_id,
            selected,
        ),
        flags=_write_slot(state.flags, command.flags, selected),
        hostile=_write_slot(state.hostile, command.hostile, selected),
        active=state.active | selected,
        entity_overlap_only=_write_slot(
            state.entity_overlap_only,
            command.entity_overlap_only,
            selected,
        ),
    )


def _tick_projectiles(
    combat: CombatState,
    state: ProjectileState,
    dt: jax.Array,
    params: CombatParams,
) -> tuple[ProjectileState, jax.Array, tuple[jax.Array, jax.Array]]:
    delta = dt[:, None]
    ticking_dead = state.active & state.impacted
    dead_remaining = jnp.where(
        ticking_dead,
        state.dead_time_remaining - delta,
        state.dead_time_remaining,
    )
    removed_dead = ticking_dead & (dead_remaining <= jnp.float32(0.0))
    active = state.active & ~removed_dead
    free = active & ~state.impacted

    velocity = _projectile_velocity(state, delta)
    candidate_position = state.position + velocity * delta[:, :, None]
    hit, fraction = _projectile_entity_hit(
        combat,
        state,
        candidate_position,
        free,
        params,
    )
    impact_position = (
        state.position + (candidate_position - state.position) * fraction[:, :, None]
    )
    position = jnp.where(
        free[:, :, None],
        jnp.where(hit[:, :, None], impact_position, candidate_position),
        state.position,
    )
    velocity = jnp.where(
        hit[:, :, None],
        jnp.float32(0.0),
        jnp.where(free[:, :, None], velocity, state.velocity),
    )
    impacted = (state.impacted & active) | hit
    dead_remaining = jnp.where(
        hit,
        state.dead_time_seconds,
        dead_remaining,
    )
    age = jnp.where(active, state.age_seconds + delta, state.age_seconds)
    expired = active & (age >= state.despawn_seconds)
    active &= ~expired
    hit_damage = jnp.where(hit, state.damage, jnp.float32(0.0))
    target_damage = jnp.sum(
        jnp.where(~state.hostile, hit_damage, jnp.float32(0.0)),
        axis=1,
    )
    agent_damage = jnp.sum(
        jnp.where(state.hostile, hit_damage, jnp.float32(0.0)),
        axis=1,
    )
    return (
        state._replace(
            position=position,
            velocity=velocity,
            age_seconds=age,
            dead_time_remaining=dead_remaining,
            active=active,
            impacted=impacted,
            physics_initialized=state.physics_initialized | free,
        ),
        jnp.sum(hit.astype(jnp.int32), axis=1),
        (target_damage, agent_damage),
    )


def _projectile_velocity(
    state: ProjectileState,
    dt: jax.Array,
) -> jax.Array:
    velocity = state.velocity
    size = state.half_extent * jnp.float32(2.0)
    width, height, depth = size[:, :, 0], size[:, :, 1], size[:, :, 2]
    weighted_area_speed = (
        jnp.abs(velocity[:, :, 0]) * depth * height
        + jnp.abs(velocity[:, :, 1]) * depth * width
        + jnp.abs(velocity[:, :, 2]) * width * height
    )
    horizontal_area = jnp.maximum(
        width * depth,
        jnp.float32(1.0e-12),
    )
    terminal_squared = jnp.maximum(
        state.terminal_velocity * state.terminal_velocity,
        jnp.float32(1.0e-12),
    )
    drag_rate = (
        state.gravity * weighted_area_speed / (horizontal_area * terminal_squared)
    )
    drag_delta = -velocity * drag_rate[:, :, None] * dt[:, :, None]
    after_drag = velocity + drag_delta
    reversed_component = ((velocity > 0.0) & (after_drag < 0.0)) | (
        (velocity < 0.0) & (after_drag > 0.0)
    )
    after_drag = jnp.where(reversed_component, jnp.float32(0.0), after_drag)
    gravity_delta = jnp.zeros_like(velocity)
    gravity_delta = gravity_delta.at[:, :, 1].set(-state.gravity * dt)
    forced = after_drag + gravity_delta
    return jnp.where(
        state.physics_initialized[:, :, None],
        forced,
        velocity,
    )


def _projectile_entity_hit(
    combat: CombatState,
    projectile: ProjectileState,
    end: jax.Array,
    eligible: jax.Array,
    params: CombatParams,
) -> tuple[jax.Array, jax.Array]:
    hostile = projectile.hostile
    victim_id = jnp.where(hostile, AGENT_ENTITY, TARGET_ENTITY)
    victim_position = jnp.where(
        hostile[:, :, None],
        combat.position[:, None, AGENT_ENTITY],
        combat.position[:, None, TARGET_ENTITY],
    )
    victim_bounds = jnp.where(
        hostile[:, :, None],
        params.agent_bounds[None, None, :],
        params.target_bounds[None, None, :],
    )
    box_min = victim_position + victim_bounds[:, :, :3] - projectile.half_extent
    box_max = victim_position + victim_bounds[:, :, 3:] + projectile.half_extent
    alive = jnp.where(
        hostile,
        combat.health[:, None, AGENT_ENTITY] > 0.0,
        combat.health[:, None, TARGET_ENTITY] > 0.0,
    )
    hit, fraction = _segment_aabb(
        projectile.position,
        end,
        box_min,
        box_max,
    )
    hit &= eligible & alive & (projectile.owner_entity_id != victim_id)
    return hit, fraction


def _tick_hazards(
    combat: CombatState,
    state: HazardState,
    dt: jax.Array,
    params: CombatParams,
) -> tuple[HazardState, jax.Array, tuple[jax.Array, jax.Array]]:
    remaining = jnp.maximum(
        state.duration_seconds - state.age_seconds,
        jnp.float32(0.0),
    )
    active_dt = jnp.minimum(dt[:, None], remaining)
    hostile = state.hostile
    victim_id = jnp.where(hostile, AGENT_ENTITY, TARGET_ENTITY)
    victim_position = jnp.where(
        hostile[:, :, None],
        combat.position[:, None, AGENT_ENTITY],
        combat.position[:, None, TARGET_ENTITY],
    )
    victim_bounds = jnp.where(
        hostile[:, :, None],
        params.agent_bounds[None, None, :],
        params.target_bounds[None, None, :],
    )
    hazard_min = state.center - state.half_extent
    hazard_max = state.center + state.half_extent
    victim_min = victim_position + victim_bounds[:, :, :3]
    victim_max = victim_position + victim_bounds[:, :, 3:]
    overlaps = jnp.all(
        (hazard_max >= victim_min) & (hazard_min <= victim_max),
        axis=2,
    )
    alive = jnp.where(
        hostile,
        combat.health[:, None, AGENT_ENTITY] > 0.0,
        combat.health[:, None, TARGET_ENTITY] > 0.0,
    )
    hit = (
        state.active
        & overlaps
        & alive
        & (active_dt > 0.0)
        & (state.activation > 0.0)
        & (state.intensity > 0.0)
        & (state.owner_entity_id != victim_id)
    )
    damage = jnp.where(
        hit,
        state.damage_per_second * active_dt * state.intensity * state.activation,
        jnp.float32(0.0),
    )
    target_damage = jnp.sum(
        jnp.where(~hostile, damage, jnp.float32(0.0)),
        axis=1,
    )
    agent_damage = jnp.sum(
        jnp.where(hostile, damage, jnp.float32(0.0)),
        axis=1,
    )
    age = jnp.where(
        state.active,
        state.age_seconds + dt[:, None],
        state.age_seconds,
    )
    active = state.active & (age < state.duration_seconds)
    return (
        state._replace(age_seconds=age, active=active),
        jnp.sum(hit.astype(jnp.int32), axis=1),
        (target_damage, agent_damage),
    )


def _apply_effect_damage(
    state: CombatState,
    projectile_damage: tuple[jax.Array, jax.Array],
    hazard_damage: tuple[jax.Array, jax.Array],
) -> tuple[CombatState, RewardComponents, jax.Array, jax.Array]:
    target_before = state.health[:, TARGET_ENTITY]
    agent_before = state.health[:, AGENT_ENTITY]
    target_after_projectile = jnp.maximum(
        jnp.float32(0.0),
        target_before - projectile_damage[0],
    )
    agent_after_projectile = jnp.maximum(
        jnp.float32(0.0),
        agent_before - projectile_damage[1],
    )
    target_after = jnp.maximum(
        jnp.float32(0.0),
        target_after_projectile - hazard_damage[0],
    )
    agent_after = jnp.maximum(
        jnp.float32(0.0),
        agent_after_projectile - hazard_damage[1],
    )
    projectile_actual = (
        target_before - target_after_projectile + agent_before - agent_after_projectile
    )
    hazard_actual = (
        target_after_projectile - target_after + agent_after_projectile - agent_after
    )
    health = state.health.at[:, TARGET_ENTITY].set(target_after)
    health = health.at[:, AGENT_ENTITY].set(agent_after)
    target_actual = target_before - target_after
    agent_actual = agent_before - agent_after
    completion = (target_after <= 0.0) & ~state.completion_awarded
    death = (agent_after <= 0.0) & ~state.death_penalty_awarded
    components = RewardComponents(
        target_damage=target_actual,
        agent_damage=agent_actual,
        completion=completion,
        death=death,
    )
    result = state._replace(
        health=health,
        ticks_since_agent_damage=jnp.where(
            agent_actual > 0.0,
            jnp.int32(0),
            state.ticks_since_agent_damage,
        ),
        target_damage_applied_this_tick=(
            state.target_damage_applied_this_tick | (agent_actual > 0.0)
        ),
        completion_awarded=state.completion_awarded | completion,
        death_penalty_awarded=state.death_penalty_awarded | death,
    )
    return result, components, projectile_actual, hazard_actual


def _segment_aabb(
    start: jax.Array,
    end: jax.Array,
    box_min: jax.Array,
    box_max: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    delta = end - start
    parallel = jnp.abs(delta) <= jnp.float32(1.0e-8)
    safe_delta = jnp.where(parallel, jnp.float32(1.0), delta)
    t0 = (box_min - start) / safe_delta
    t1 = (box_max - start) / safe_delta
    entry_axis = jnp.where(
        parallel,
        -jnp.inf,
        jnp.minimum(t0, t1),
    )
    exit_axis = jnp.where(
        parallel,
        jnp.inf,
        jnp.maximum(t0, t1),
    )
    entry = jnp.max(entry_axis, axis=2)
    exit_ = jnp.min(exit_axis, axis=2)
    parallel_inside = jnp.all(
        ~parallel | ((start >= box_min) & (start <= box_max)),
        axis=2,
    )
    hit = parallel_inside & (entry <= exit_) & (exit_ >= 0.0) & (entry <= 1.0)
    return hit, jnp.clip(entry, 0.0, 1.0)


def _projectile_command_valid(commands: CombatEffectCommands) -> jax.Array:
    value = commands.projectile
    return (
        _finite_vector(value.position)
        & _finite_vector(value.velocity)
        & _finite_vector(value.half_extent)
        & jnp.all(value.half_extent > 0.0, axis=1)
        & _finite_positive(value.despawn_seconds)
        & _finite_positive(value.authored_lifetime_seconds)
        & _finite_nonnegative(value.damage)
        & _finite_nonnegative(value.gravity)
        & _finite_positive(value.terminal_velocity)
        & _finite_nonnegative(value.dead_time_seconds)
        & _finite_positive(value.velocity_scale)
        & (value.kind > 0)
    )


def _hazard_command_valid(commands: CombatEffectCommands) -> jax.Array:
    value = commands.hazard
    return (
        _finite_vector(value.center)
        & _finite_vector(value.half_extent)
        & jnp.all(value.half_extent >= 0.0, axis=1)
        & _finite_positive(value.duration_seconds)
        & _finite_nonnegative(value.damage_per_second)
        & _finite_unit(value.intensity)
        & _finite_unit(value.activation)
        & (value.kind > 0)
    )


def _state_failures(
    state: CombatEffectsState,
) -> tuple[jax.Array, jax.Array]:
    projectile = state.projectiles
    hazard = state.hazards
    projectile_invalid = projectile.active & ~(
        jnp.all(jnp.isfinite(projectile.position), axis=2)
        & jnp.all(jnp.isfinite(projectile.velocity), axis=2)
        & jnp.all(jnp.isfinite(projectile.half_extent), axis=2)
        & jnp.all(projectile.half_extent > 0.0, axis=2)
        & jnp.isfinite(projectile.age_seconds)
        & (projectile.age_seconds >= 0.0)
        & _state_positive(projectile.despawn_seconds)
        & _state_positive(projectile.authored_lifetime_seconds)
        & _state_nonnegative(projectile.damage)
        & _state_nonnegative(projectile.gravity)
        & _state_positive(projectile.terminal_velocity)
        & _state_nonnegative(projectile.dead_time_seconds)
        & _state_positive(projectile.velocity_scale)
        & (projectile.kind > 0)
    )
    hazard_invalid = hazard.active & ~(
        jnp.all(jnp.isfinite(hazard.center), axis=2)
        & jnp.all(jnp.isfinite(hazard.half_extent), axis=2)
        & jnp.all(hazard.half_extent >= 0.0, axis=2)
        & jnp.isfinite(hazard.age_seconds)
        & (hazard.age_seconds >= 0.0)
        & _state_positive(hazard.duration_seconds)
        & _state_nonnegative(hazard.damage_per_second)
        & _state_unit(hazard.intensity)
        & _state_unit(hazard.activation)
        & (hazard.kind > 0)
    )
    invalid = jnp.any(projectile_invalid, axis=1) | jnp.any(
        hazard_invalid,
        axis=1,
    )
    unsupported = jnp.any(
        projectile.active & ~projectile.entity_collision_only,
        axis=1,
    ) | jnp.any(
        hazard.active & ~hazard.entity_overlap_only,
        axis=1,
    )
    return invalid, unsupported


def _write_slot(
    array: jax.Array,
    value: jax.Array,
    selected: jax.Array,
) -> jax.Array:
    shape = selected.shape + (1,) * (array.ndim - 2)
    return jnp.where(selected.reshape(shape), value[:, None], array)


def _set_failure(
    bits: jax.Array,
    failed: jax.Array,
    value: int,
) -> jax.Array:
    return jnp.bitwise_or(
        bits,
        jnp.where(failed, jnp.uint32(value), jnp.uint32(0)),
    )


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


def _finite_vector(value: jax.Array) -> jax.Array:
    return jnp.all(jnp.isfinite(value), axis=1)


def _finite_positive(value: jax.Array) -> jax.Array:
    return jnp.isfinite(value) & (value > 0.0)


def _finite_nonnegative(value: jax.Array) -> jax.Array:
    return jnp.isfinite(value) & (value >= 0.0)


def _finite_unit(value: jax.Array) -> jax.Array:
    return jnp.isfinite(value) & (value >= 0.0) & (value <= 1.0)


def _state_positive(value: jax.Array) -> jax.Array:
    return jnp.isfinite(value) & (value > 0.0)


def _state_nonnegative(value: jax.Array) -> jax.Array:
    return jnp.isfinite(value) & (value >= 0.0)


def _state_unit(value: jax.Array) -> jax.Array:
    return jnp.isfinite(value) & (value >= 0.0) & (value <= 1.0)
