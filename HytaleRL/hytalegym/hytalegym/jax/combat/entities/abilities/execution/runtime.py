"""Fused atomic scheduler, direct resolver, and impact-launch transition."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.entities.abilities.execution.direct import (
    resolve_entity_ability_events,
)
from hytalegym.jax.combat.entities.abilities.execution.impacts import (
    launch_entity_ability_impacts,
)
from hytalegym.jax.combat.entities.abilities.execution.scheduler import (
    step_entity_abilities,
)
from hytalegym.jax.combat.entities.abilities.schema.types import (
    EntityAbilityRuntimeInfo,
)


def step_entity_ability_runtime(
    state,
    effects,
    arsenal,
    bindings,
    commands,
    availability,
    direct_queries,
    projectile_world,
    area_world,
    friendly_fire,
    programs,
    projectile_programs,
    area_programs,
    dt_seconds,
    rules,
    *,
    random_keys=None,
):
    """Commit the whole authored-event transition, or only sticky failures."""

    original_state = state
    original_effects = effects
    original_arsenal = arsenal
    original_bindings = bindings
    state, effects, events, scheduler_info = step_entity_abilities(
        state,
        effects,
        commands,
        availability,
        programs,
        dt_seconds,
        rules,
    )
    state, effects, direct_info = resolve_entity_ability_events(
        state,
        effects,
        events,
        direct_queries,
        programs,
        rules,
        random_keys=random_keys,
    )
    state, arsenal, bindings, launch_info = launch_entity_ability_impacts(
        state,
        arsenal,
        bindings,
        effects.combat.roster,
        events,
        programs,
        projectile_programs,
        area_programs,
        projectile_world,
        area_world,
        friendly_fire,
    )
    valid = scheduler_info.valid & direct_info.valid & launch_info.valid
    failure_bits = state.failure_bits
    world_failure_bits = state.world_failure_bits
    state = _select_tree(valid, state, original_state)._replace(
        failure_bits=failure_bits,
        world_failure_bits=world_failure_bits,
    )
    effects = _select_tree(valid, effects, original_effects)
    arsenal = _select_tree(valid, arsenal, original_arsenal)._replace(
        failure_bits=arsenal.failure_bits
    )
    bindings = _select_tree(valid, bindings, original_bindings)
    events = events._replace(
        requested=events.requested & valid[:, None],
        failure_bits=failure_bits,
        world_failure_bits=world_failure_bits,
        valid=valid,
    )
    return (
        state,
        effects,
        arsenal,
        bindings,
        events,
        EntityAbilityRuntimeInfo(
            scheduler=scheduler_info,
            direct=direct_info,
            launch=launch_info,
            failure_bits=failure_bits,
            world_failure_bits=world_failure_bits,
            valid=valid,
        ),
    )


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
