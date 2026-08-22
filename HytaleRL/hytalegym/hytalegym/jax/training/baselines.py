"""Reference action sources for policy competence evaluation."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from hytalegym.jax.combat import (
    SKILL_IDLE,
    arsenal_policy_action_mask,
    legal_skill_mask,
    neutral_arsenal_policy_action_factors,
    skills_to_actions,
)
from hytalegym.jax.combat.observation.v3.policy.distribution import (
    sample_arsenal_action_factors,
)
from hytalegym.jax.training.arsenal_evaluation import (
    ArsenalPolicyActionInput,
    policy_aware_arsenal_action_source,
)
BASELINE_IDENTITIES = {
    "uniform_legal": {
        "schema": "hytalerl_uniform_legal_action_baseline_v2",
        "version": 2,
        "sampling": (
            "uniform over decoder-valid joint Arsenal factors after legal masking"
        ),
    },
    "always_idle": {
        "schema": "hytalerl_always_idle_action_baseline_v1",
        "version": 1,
        "action": "idle skill, no ability, guard, dodge, or jump",
    },
    "untrained_init": {
        "schema": "hytalerl_untrained_policy_baseline_v1",
        "version": 1,
        "selection": "greedy legal action from seeded policy initialization",
    },
}
LEGACY_BASELINE_IDENTITIES = {
    **BASELINE_IDENTITIES,
    "uniform_legal": {
        "schema": "hytalerl_uniform_legal_action_baseline_v1",
        "version": 1,
        "sampling": "uniform independently within each legal categorical head",
    },
}


def uniform_legal_melee_action_source(
    observation: jax.Array,
    carry: Any,
    key: jax.Array,
):
    """Sample each row uniformly from its legal melee skills."""

    skill_ids = _uniform_legal_categorical(
        key,
        legal_skill_mask(observation),
    )
    return skills_to_actions(observation, skill_ids), carry


def always_idle_melee_action_source(
    observation: jax.Array,
    carry: Any,
    _key: jax.Array,
):
    """Emit the transfer-safe idle skill for every melee environment."""

    skill_ids = jnp.full(
        (observation.shape[0],),
        SKILL_IDLE,
        dtype=jnp.int32,
    )
    return skills_to_actions(observation, skill_ids), carry


def uniform_legal_arsenal_action_source(
    observation,
    carry: Any,
    key: jax.Array,
):
    """Sample every published Arsenal head uniformly after legal masking."""

    mask = arsenal_policy_action_mask(observation)
    return (
        _uniform_legal_arsenal_factors(key, mask),
        carry,
    )


def always_idle_arsenal_action_source(
    observation,
    carry: Any,
    _key: jax.Array,
):
    """Emit the explicit all-neutral Arsenal factors per environment."""

    factors = neutral_arsenal_policy_action_factors(
        int(observation.valid.shape[0])
    )
    return factors, carry


def _uniform_legal_categorical(
    key: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    logits = jnp.where(
        mask,
        jnp.float32(0.0),
        jnp.float32(-1.0e9),
    )
    return jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)


def _uniform_legal_arsenal_factors(
    key: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    logits = jnp.where(
        mask,
        jnp.float32(0.0),
        jnp.float32(-1.0e9),
    )
    factors, _ = sample_arsenal_action_factors(key, logits)
    return factors


def _uniform_legal_arsenal_policy_source(
    policy_input: ArsenalPolicyActionInput,
    carry: Any,
    key: jax.Array,
):
    """Sample the exact mask paired with the actor's flattened input."""

    return (
        _uniform_legal_arsenal_factors(key, policy_input.action_mask),
        carry,
    )


uniform_legal_arsenal_action_source = policy_aware_arsenal_action_source(
    uniform_legal_arsenal_action_source,
    _uniform_legal_arsenal_policy_source,
)
