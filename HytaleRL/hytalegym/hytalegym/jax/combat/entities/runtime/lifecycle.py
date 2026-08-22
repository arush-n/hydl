"""Deterministic slot allocation with generation-checked entity handles."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.entities.schema.contract import (
    ENTITY_CAPACITY,
    ENTITY_FAILURE_CAPACITY,
    ENTITY_FAILURE_GENERATION_EXHAUSTED,
    ENTITY_FAILURE_INVALID_COMMAND,
    ENTITY_FAILURE_INVALID_STATE,
    TEAM_NONE,
)
from hytalegym.jax.combat.entities.schema.types import (
    EntityCombatState,
    EntityLifecycleCommands,
    EntityLifecycleInfo,
    EntityRoster,
)
from hytalegym.jax.combat.mechanics import (
    CombatMechanicsRules,
    empty_mechanics_state,
)
from hytalegym.jax.combat.entities.schema.validation import (
    invalid_roster_rows,
    validate_lifecycle_layout,
)

_MAX_GENERATION = jnp.asarray(0xFFFFFFFF, dtype=jnp.uint32)


def apply_entity_lifecycle(
    roster: EntityRoster,
    commands: EntityLifecycleCommands,
) -> tuple[EntityRoster, EntityLifecycleInfo]:
    """Despawn valid handles, then allocate spawns into the lowest free slots."""

    validate_lifecycle_layout(roster, commands)
    slots = commands.despawn.slot
    slot_valid = (slots >= 0) & (slots < ENTITY_CAPACITY)
    clipped = jnp.clip(slots, 0, ENTITY_CAPACITY - 1)
    selected_despawn = (
        jax.nn.one_hot(clipped, ENTITY_CAPACITY, dtype=jnp.bool_)
        & commands.despawn.requested[..., None]
    )
    handle_valid = (
        slot_valid
        & _gather(roster.active, clipped)
        & (_gather(roster.generation, clipped) == commands.despawn.generation)
    )
    duplicate_despawn = jnp.any(
        jnp.sum(selected_despawn.astype(jnp.int32), axis=1) > 1,
        axis=1,
    )
    invalid_despawn = duplicate_despawn | jnp.any(
        commands.despawn.requested & ~handle_valid,
        axis=1,
    )
    spawn = commands.spawn
    damageable_valid = (
        jnp.isfinite(spawn.health)
        & jnp.isfinite(spawn.max_health)
        & (spawn.max_health > 0.0)
        & (spawn.health > 0.0)
        & (spawn.health <= spawn.max_health)
    )
    nondamageable_valid = (spawn.health == 0.0) & (spawn.max_health == 0.0)
    valid_spawn = (
        (spawn.semantic_id > 0)
        & (spawn.team_id >= TEAM_NONE)
        & jnp.all(jnp.isfinite(spawn.position), axis=2)
        & jnp.all(jnp.isfinite(spawn.velocity), axis=2)
        & jnp.isfinite(spawn.yaw_degrees)
        & jnp.all(jnp.isfinite(spawn.resource_initial), axis=2)
        & jnp.where(
            spawn.damageable,
            damageable_valid,
            nondamageable_valid,
        )
    )
    invalid_spawn = jnp.any(spawn.requested & ~valid_spawn, axis=1)
    bits = _set_failure(
        roster.failure_bits,
        invalid_roster_rows(roster),
        ENTITY_FAILURE_INVALID_STATE,
    )
    bits = _set_failure(
        bits,
        invalid_despawn | invalid_spawn,
        ENTITY_FAILURE_INVALID_COMMAND,
    )
    pre_valid = bits == jnp.uint32(0)
    slot_despawned = jnp.any(selected_despawn, axis=1)
    free = ~roster.active | slot_despawned
    reusable = free & (roster.generation != _MAX_GENERATION)
    requested_count = jnp.sum(
        spawn.requested.astype(jnp.int32),
        axis=1,
    )
    free_count = jnp.sum(free.astype(jnp.int32), axis=1)
    reusable_count = jnp.sum(reusable.astype(jnp.int32), axis=1)
    bits = _set_failure(
        bits,
        pre_valid & (requested_count > free_count),
        ENTITY_FAILURE_CAPACITY,
    )
    bits = _set_failure(
        bits,
        pre_valid
        & (requested_count <= free_count)
        & (requested_count > reusable_count),
        ENTITY_FAILURE_GENERATION_EXHAUSTED,
    )
    row_valid = bits == jnp.uint32(0)
    slot_despawned &= row_valid[:, None]
    current = _clear_slots(roster, slot_despawned)
    available = ~current.active & (current.generation != _MAX_GENERATION)
    inputs = jax.tree_util.tree_map(
        lambda value: jnp.moveaxis(value, 1, 0),
        spawn,
    )

    def spawn_one(carry, command):
        state, free_slots = carry
        requested = command.requested & row_valid
        slot = jnp.argmax(free_slots, axis=1).astype(jnp.int32)
        selected = (
            jax.nn.one_hot(slot, ENTITY_CAPACITY, dtype=jnp.bool_) & requested[:, None]
        )
        generation = state.generation + selected.astype(jnp.uint32)
        state = state._replace(
            semantic_id=_write(state.semantic_id, command.semantic_id, selected),
            generation=generation,
            team_id=_write(state.team_id, command.team_id, selected),
            position=_write(state.position, command.position, selected),
            velocity=_write(state.velocity, command.velocity, selected),
            yaw_degrees=_write(
                state.yaw_degrees,
                command.yaw_degrees,
                selected,
            ),
            health=_write(state.health, command.health, selected),
            max_health=_write(
                state.max_health,
                command.max_health,
                selected,
            ),
            damageable=_write(
                state.damageable,
                command.damageable,
                selected,
            ),
            intangible=_write(
                state.intangible,
                command.intangible,
                selected,
            ),
            invulnerable=_write(
                state.invulnerable,
                command.invulnerable,
                selected,
            ),
            dead=jnp.where(selected, False, state.dead),
            active=state.active | selected,
        )
        allocated_generation = _gather(generation, slot)
        output = (
            requested,
            jnp.where(requested, slot, jnp.int32(-1)),
            jnp.where(
                requested,
                allocated_generation,
                jnp.uint32(0),
            ),
            selected,
        )
        return (state, free_slots & ~selected), output

    (current, _), outputs = jax.lax.scan(
        spawn_one,
        (current, available),
        inputs,
    )
    spawned, spawn_slot, spawn_generation, assignment = (
        jnp.moveaxis(value, 0, 1) for value in outputs
    )
    current = current._replace(failure_bits=bits)
    return current, EntityLifecycleInfo(
        spawned=spawned,
        spawn_slot=spawn_slot,
        spawn_generation=spawn_generation,
        despawned=commands.despawn.requested & row_valid[:, None],
        slot_spawned=jnp.any(assignment, axis=1),
        slot_despawned=slot_despawned,
        spawn_assignment=assignment,
        failure_bits=bits,
        valid=row_valid,
    )


def apply_entity_combat_lifecycle(
    state: EntityCombatState,
    commands: EntityLifecycleCommands,
    rules: CombatMechanicsRules,
) -> tuple[EntityCombatState, EntityLifecycleInfo]:
    """Apply lifecycle changes and clear every reused mechanics slot."""

    batch = state.roster.active.shape[0]
    expected = (batch, ENTITY_CAPACITY)
    if state.mechanics.resources.shape[:2] != expected:
        raise ValueError(f"mechanics state must have shape {expected}")
    if rules.resource_maximum.shape[:2] != expected:
        raise ValueError(f"mechanics rules must have shape {expected}")
    roster, info = apply_entity_lifecycle(state.roster, commands)
    initial = jnp.sum(
        jnp.where(
            info.spawn_assignment[..., None],
            commands.spawn.resource_initial[:, :, None, :],
            jnp.float32(0.0),
        ),
        axis=1,
    )
    spawned = info.slot_spawned[..., None]
    invalid_resource = jnp.any(
        spawned
        & (
            ~jnp.isfinite(initial)
            | (initial < rules.resource_minimum)
            | (initial > rules.resource_maximum)
        ),
        axis=(1, 2),
    )
    bits = _set_failure(
        info.failure_bits,
        invalid_resource,
        ENTITY_FAILURE_INVALID_COMMAND,
    )
    valid = bits == jnp.uint32(0)
    changed = (info.slot_spawned | info.slot_despawned) & valid[:, None]
    baseline = empty_mechanics_state(batch, rules)

    def reset_leaf(old, fresh):
        if old.ndim >= 2 and old.shape[:2] == expected:
            mask = changed.reshape(changed.shape + (1,) * (old.ndim - 2))
            return jnp.where(mask, fresh, old)
        return old

    mechanics = jax.tree_util.tree_map(
        reset_leaf,
        state.mechanics,
        baseline,
    )
    mechanics = mechanics._replace(
        resources=jnp.where(
            (info.slot_spawned & valid[:, None])[..., None],
            initial,
            mechanics.resources,
        )
    )
    roster = _select_tree(valid, roster, state.roster)._replace(failure_bits=bits)
    mechanics = _select_tree(valid, mechanics, state.mechanics)
    return EntityCombatState(roster, mechanics), info._replace(
        spawned=info.spawned & valid[:, None],
        spawn_slot=jnp.where(valid[:, None], info.spawn_slot, -1),
        spawn_generation=jnp.where(
            valid[:, None],
            info.spawn_generation,
            jnp.uint32(0),
        ),
        despawned=info.despawned & valid[:, None],
        slot_spawned=info.slot_spawned & valid[:, None],
        slot_despawned=info.slot_despawned & valid[:, None],
        spawn_assignment=info.spawn_assignment & valid[:, None, None],
        failure_bits=bits,
        valid=valid,
    )


def _clear_slots(roster: EntityRoster, selected: jax.Array) -> EntityRoster:
    vectors = selected[..., None]
    return roster._replace(
        semantic_id=jnp.where(selected, jnp.int32(0), roster.semantic_id),
        team_id=jnp.where(selected, jnp.int32(TEAM_NONE), roster.team_id),
        position=jnp.where(vectors, jnp.float32(0.0), roster.position),
        velocity=jnp.where(vectors, jnp.float32(0.0), roster.velocity),
        yaw_degrees=jnp.where(selected, jnp.float32(0.0), roster.yaw_degrees),
        health=jnp.where(selected, jnp.float32(0.0), roster.health),
        max_health=jnp.where(selected, jnp.float32(0.0), roster.max_health),
        damageable=roster.damageable & ~selected,
        intangible=roster.intangible & ~selected,
        invulnerable=roster.invulnerable & ~selected,
        dead=roster.dead & ~selected,
        active=roster.active & ~selected,
    )


def _gather(array: jax.Array, slot: jax.Array) -> jax.Array:
    batch = jnp.arange(array.shape[0]).reshape(
        (array.shape[0],) + (1,) * (slot.ndim - 1)
    )
    return array[batch, slot]


def _write(
    array: jax.Array,
    value: jax.Array,
    selected: jax.Array,
) -> jax.Array:
    trailing = (1,) * (array.ndim - 2)
    mask = selected.reshape(selected.shape + trailing)
    expanded = value.reshape((value.shape[0], 1) + value.shape[1:])
    return jnp.where(mask, expanded, array)


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
