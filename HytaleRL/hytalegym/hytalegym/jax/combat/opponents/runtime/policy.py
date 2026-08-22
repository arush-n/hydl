"""Fixed-shape opponent ability policies for training and diagnostics."""

from __future__ import annotations

import hashlib
import json

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.programs.ability import ability_legal_mask
from hytalegym.jax.combat.arsenal.schema.types import (
    ArsenalEnvironmentState,
    ArsenalRuntimeConfig,
    ArsenalWorldCapabilities,
)
from hytalegym.jax.combat.opponents.schema.contract import OPPONENT_MODE_CHASE


FIRST_LEGAL_OPPONENT_ABILITY_POLICY_SCHEMA = (
    "hytalerl_first_legal_opponent_ability_policy_v1"
)


def first_legal_opponent_ability_policy_contract() -> dict[str, object]:
    """Return the artifact-pinnable semantics of the production baseline."""

    return {
        "schema": FIRST_LEGAL_OPPONENT_ABILITY_POLICY_SCHEMA,
        "controlled_sources": "opponent_controller_mask_and_chase_mode",
        "selection": "lowest_shared_arsenal_legal_ability_slot",
        "no_legal_slot": -1,
        "actor_row": "never_owned",
        "certification": "deterministic_training_baseline_not_native_role_choice",
    }


def first_legal_opponent_ability_policy_sha256() -> str:
    """Return the canonical identity of the production baseline."""

    payload = json.dumps(
        first_legal_opponent_ability_policy_contract(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def first_legal_opponent_ability_slots(
    state: ArsenalEnvironmentState,
    world: ArsenalWorldCapabilities,
    config: ArsenalRuntimeConfig,
) -> jax.Array:
    """Choose the lowest legal slot independently for each chasing opponent.

    This is a deterministic training baseline, not a claim about native
    Trork utility or interaction choice. Legality remains the shared Arsenal
    predicate, so cooldown, resources, statuses, equipment, and injected
    world evidence cannot be bypassed.
    """

    legal = ability_legal_mask(
        state.arsenal,
        state.mechanics,
        config.loadout,
        world,
        state.combat.health > jnp.float32(0.0),
    )
    available = jnp.any(legal, axis=2)
    selected = jnp.argmax(legal, axis=2).astype(jnp.int32)
    controlled = config.opponent_controller_mask & (
        state.opponent_memory.mode == jnp.int32(OPPONENT_MODE_CHASE)
    )
    return jnp.where(
        controlled & available,
        selected,
        jnp.int32(-1),
    )


def inert_opponent_ability_slots(
    state: ArsenalEnvironmentState,
    world: ArsenalWorldCapabilities,
    config: ArsenalRuntimeConfig,
) -> jax.Array:
    """Explicitly disable autonomous abilities for every entity.

    This named provider is the intentional inert-opponent override for
    diagnostics. Production training defaults to
    :func:`first_legal_opponent_ability_slots`, so an omitted provider cannot
    silently turn the opponent into a harmless target.
    """

    del world, config
    return jnp.full(state.combat.health.shape, -1, dtype=jnp.int32)


__all__ = [
    "FIRST_LEGAL_OPPONENT_ABILITY_POLICY_SCHEMA",
    "first_legal_opponent_ability_policy_contract",
    "first_legal_opponent_ability_policy_sha256",
    "first_legal_opponent_ability_slots",
    "inert_opponent_ability_slots",
]
