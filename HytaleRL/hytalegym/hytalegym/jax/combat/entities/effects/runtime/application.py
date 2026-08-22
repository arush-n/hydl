"""Status application wrapper that records source-generation provenance."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.entities import apply_selected_entity_statuses
from hytalegym.jax.combat.entities.effects.schema.contract import (
    ENTITY_EFFECT_FAILURE_UPSTREAM,
)
from hytalegym.jax.combat.entities.effects.schema.types import EntityEffectState
from hytalegym.jax.combat.entities.effects.schema.validation import (
    validate_effect_layout,
)


def apply_selected_entity_effects(
    state: EntityEffectState,
    selection,
    payloads,
):
    """Apply statuses and atomically bind new slots to source generations."""

    validate_effect_layout(state)
    clean = state.failure_bits == jnp.uint32(0)
    masked = payloads._replace(requested=payloads.requested & clean[:, None])
    combat, info = apply_selected_entity_statuses(
        state.combat,
        selection,
        masked,
    )
    before = state.combat.mechanics.statuses
    after = combat.mechanics.statuses
    changed = after.active & (
        ~before.active
        | (after.effect_id != before.effect_id)
        | (after.source_entity_id != before.source_entity_id)
    )
    source = jnp.clip(
        after.source_entity_id,
        0,
        combat.roster.active.shape[1] - 1,
    )
    generation = combat.roster.generation[
        jnp.arange(source.shape[0])[:, None, None],
        source,
    ]
    tracked = jnp.where(
        ~after.active,
        jnp.uint32(0),
        jnp.where(
            changed & (after.source_entity_id >= 0),
            generation,
            state.source_generation,
        ),
    )
    upstream = (combat.roster.failure_bits != jnp.uint32(0)) | (
        combat.mechanics.failure_bits != jnp.uint32(0)
    )
    bits = _set_failure(
        state.failure_bits,
        upstream,
        ENTITY_EFFECT_FAILURE_UPSTREAM,
    )
    valid = bits == jnp.uint32(0)
    result = EntityEffectState(
        combat=_select_tree(valid, combat, state.combat),
        source_generation=jnp.where(
            valid[:, None, None],
            tracked,
            state.source_generation,
        ),
        capability_bits=state.capability_bits,
        failure_bits=bits,
    )
    return result, info._replace(
        application_count=jnp.where(
            valid,
            info.application_count,
            jnp.int32(0),
        ),
        valid=info.valid & valid,
    )


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
