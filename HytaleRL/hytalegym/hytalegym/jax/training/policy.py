"""Small recurrent actor-critic used inside the combat rollout scan."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from hytalegym.jax.combat import (
    ACTIVE_OBSERVATION_SIZE,
    SKILL_COUNT,
    legal_skill_mask,
)
from hytalegym.jax.policy import (  # noqa: F401
    action_factor_log_probabilities,
    action_log_probabilities,
    categorical_entropy,
    pack_action_factors,
    sample_action_factors,
    sample_actions,
    unpack_action_factors,
)
from hytalegym.jax.combat.observation.v3.policy.distribution import (
    ARSENAL_STANDARD_ROOT_DISTRIBUTION,
    arsenal_action_entropy,
    arsenal_action_log_probabilities,
    sample_arsenal_action_factors,
)
from hytalegym.jax.training.types import (
    DenseParams,
    GRUParams,
    PPOConfig,
    RecurrentPolicyParams,
)


def initialize_policy(
    key: jax.Array,
    config: PPOConfig,
) -> RecurrentPolicyParams:
    """Initialize an all-float32 MLP-GRU actor-critic."""

    keys = jax.random.split(key, 7)
    return RecurrentPolicyParams(
        encoder_input=_dense(
            keys[0],
            config.observation_size,
            config.encoder_size,
        ),
        encoder_hidden=_dense(
            keys[1],
            config.encoder_size,
            config.encoder_size,
        ),
        gru=GRUParams(
            input_kernel=_glorot(
                keys[2],
                config.encoder_size,
                3 * config.recurrent_size,
            ),
            recurrent_kernel=_glorot(
                keys[3],
                config.recurrent_size,
                3 * config.recurrent_size,
            ),
            bias=jnp.zeros(
                (3 * config.recurrent_size,),
                dtype=jnp.float32,
            ),
        ),
        actor=DenseParams(
            kernel=_glorot(
                keys[4],
                config.recurrent_size,
                config.action_size,
                scale=0.01,
            ),
            bias=jnp.zeros((config.action_size,), dtype=jnp.float32),
        ),
        critic=DenseParams(
            kernel=_glorot(
                keys[5],
                config.recurrent_size,
                1,
                scale=1.0,
            ),
            bias=jnp.zeros((1,), dtype=jnp.float32),
        ),
    )


def sample_configured_actions(
    key: jax.Array,
    logits: jax.Array,
    config: PPOConfig,
) -> tuple[jax.Array, jax.Array]:
    """Sample through the distribution declared by the checkpoint config."""

    if config.action_distribution == ARSENAL_STANDARD_ROOT_DISTRIBUTION:
        return sample_arsenal_action_factors(key, logits)
    return sample_actions(
        key,
        logits,
        config.action_head_sizes,
        transport=config.action_transport,
    )


def configured_action_log_probabilities(
    logits: jax.Array,
    actions: jax.Array,
    config: PPOConfig,
) -> jax.Array:
    """Score actions under the same joint distribution used at rollout."""

    if config.action_distribution == ARSENAL_STANDARD_ROOT_DISTRIBUTION:
        return arsenal_action_log_probabilities(logits, actions)
    return action_log_probabilities(
        logits,
        actions,
        config.action_head_sizes,
        transport=config.action_transport,
    )


def configured_action_entropy(
    logits: jax.Array,
    config: PPOConfig,
) -> jax.Array:
    """Compute entropy under the configured scalar or factored distribution."""

    if config.action_distribution == ARSENAL_STANDARD_ROOT_DISTRIBUTION:
        return arsenal_action_entropy(logits)
    return categorical_entropy(logits, config.action_head_sizes)


def apply_policy(
    params: RecurrentPolicyParams,
    observation: jax.Array,
    recurrent_state: jax.Array,
    action_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply one batched recurrent policy step and legal action mask."""

    encoded = jnp.tanh(_linear(params.encoder_input, observation))
    encoded = jnp.tanh(_linear(params.encoder_hidden, encoded))
    recurrent_state = _gru_step(params.gru, encoded, recurrent_state)
    logits = _linear(params.actor, recurrent_state)
    mask = _resolve_action_mask(
        observation,
        logits,
        action_mask,
    )
    logits = jnp.where(mask, logits, jnp.float32(-1.0e9))
    value = _linear(params.critic, recurrent_state)[:, 0]
    return recurrent_state, logits, value


def apply_policy_sequence(
    params: RecurrentPolicyParams,
    observations: jax.Array,
    initial_recurrent_state: jax.Array,
    episode_starts: jax.Array,
    action_masks: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Recompute a time-major recurrent sequence with reset masks."""

    if action_masks is None:
        if (
            observations.shape[-1] != ACTIVE_OBSERVATION_SIZE
            or params.actor.bias.shape[0] != SKILL_COUNT
        ):
            raise ValueError(
                "action_masks are required outside the legacy 24x9 skill adapter"
            )
        action_masks = jax.vmap(legal_skill_mask)(observations)

    encoded = jnp.tanh(_linear(params.encoder_input, observations))
    encoded = jnp.tanh(_linear(params.encoder_hidden, encoded))

    def sequence_step(recurrent_state, inputs):
        current_encoded, episode_start = inputs
        recurrent_state = jnp.where(
            episode_start[:, None],
            jnp.float32(0.0),
            recurrent_state,
        )
        recurrent_state = _gru_step(
            params.gru,
            current_encoded,
            recurrent_state,
        )
        return recurrent_state, recurrent_state

    final_state, recurrent_states = jax.lax.scan(
        sequence_step,
        initial_recurrent_state,
        (encoded, episode_starts),
    )
    logits = _linear(params.actor, recurrent_states)
    mask = _resolve_action_mask(observations, logits, action_masks)
    logits = jnp.where(mask, logits, jnp.float32(-1.0e9))
    values = _linear(params.critic, recurrent_states)[..., 0]
    return final_state, logits, values


def _resolve_action_mask(
    observation: jax.Array,
    logits: jax.Array,
    action_mask: jax.Array | None,
) -> jax.Array:
    if action_mask is not None:
        mask = jnp.asarray(action_mask, dtype=jnp.bool_)
        if mask.shape != logits.shape:
            raise ValueError(
                "action_mask must match policy logits exactly: "
                f"{mask.shape} != {logits.shape}"
            )
        return mask
    if (
        observation.shape[-1] != ACTIVE_OBSERVATION_SIZE
        or logits.shape[-1] != SKILL_COUNT
    ):
        raise ValueError(
            "action_mask is required outside the legacy 24x9 skill adapter"
        )
    return legal_skill_mask(observation)


def _gru_step(
    params: GRUParams,
    encoded: jax.Array,
    recurrent_state: jax.Array,
) -> jax.Array:
    input_projection = encoded @ params.input_kernel + params.bias
    recurrent_projection = recurrent_state @ params.recurrent_kernel
    input_reset, input_update, input_candidate = jnp.split(
        input_projection,
        3,
        axis=-1,
    )
    recurrent_reset, recurrent_update, recurrent_candidate = jnp.split(
        recurrent_projection,
        3,
        axis=-1,
    )
    reset = jax.nn.sigmoid(input_reset + recurrent_reset)
    update = jax.nn.sigmoid(input_update + recurrent_update)
    candidate = jnp.tanh(
        input_candidate + reset * recurrent_candidate
    )
    return (
        update * recurrent_state
        + (jnp.float32(1.0) - update) * candidate
    )


def _linear(params: DenseParams, inputs: jax.Array) -> jax.Array:
    return inputs @ params.kernel + params.bias


def _dense(
    key: jax.Array,
    input_size: int,
    output_size: int,
) -> DenseParams:
    return DenseParams(
        kernel=_glorot(key, input_size, output_size),
        bias=jnp.zeros((output_size,), dtype=jnp.float32),
    )


def _glorot(
    key: jax.Array,
    input_size: int,
    output_size: int,
    *,
    scale: float = 1.0,
) -> jax.Array:
    limit = math.sqrt(6.0 / (input_size + output_size)) * scale
    return jax.random.uniform(
        key,
        (input_size, output_size),
        minval=-limit,
        maxval=limit,
        dtype=jnp.float32,
    )
