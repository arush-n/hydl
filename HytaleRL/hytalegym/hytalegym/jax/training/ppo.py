"""One-call recurrent PPO update fused with the JAX combat environment."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax

from hytalegym.jax.combat import (
    ACTIVE_OBSERVATION_SIZE,
    AGENT_ENTITY,
    CombatParams,
    SKILL_COUNT,
    TARGET_ENTITY,
    legal_skill_mask,
    reset_batch,
    skills_to_actions,
    step_batch,
)
from hytalegym.jax.combat.environment import (
    ActionComponentSpec,
    EnvironmentSpec,
)
from hytalegym.jax.training.policy import (
    apply_policy,
    apply_policy_sequence,
    configured_action_entropy,
    configured_action_log_probabilities,
    initialize_policy,
    sample_configured_actions,
)
from hytalegym.jax.training.types import (
    EpisodeOutcome,
    LossMetrics,
    PPOConfig,
    PPOMetrics,
    PPOTrainState,
    PPOUpdateBatch,
    RecurrentPolicyParams,
    RecurrentRollout,
)


#: Largest |log ratio| treated as a real policy change. The clipped surrogate
#: bounds the ratio to [1-eps, 1+eps], so a legitimate value is order 1; a gap
#: this wide means the two scores were not drawn from comparable distributions,
#: which is what an illegality sentinel produces.
_MAXIMUM_COMPARABLE_LOG_RATIO = 20.0


TrainStep = Callable[
    [PPOTrainState, jax.Array],
    tuple[PPOTrainState, PPOMetrics],
]


@dataclass(frozen=True)
class PPOEnvironment:
    """Compiled reset/step boundary for one fixed-shape policy surface.

    ``episode_outcome`` receives the post-step state before autoreset. It is
    privileged metrics telemetry and must not contribute to observation.
    """

    reset: Callable
    step: Callable
    spec: EnvironmentSpec
    episode_outcome: Callable[[Any, jax.Array], EpisodeOutcome] | None = None
    step_detailed: Callable | None = None


def combat_episode_outcome(
    combat_state: Any,
    done: jax.Array,
    *,
    valid: jax.Array | None = None,
) -> EpisodeOutcome:
    """Classify terminal combat rows without interpreting shaped reward.

    The four environment-terminal fields are exhaustive before a trainer adds
    a time limit. The remaining fields split ``other`` so infrastructure
    failures cannot be mistaken for combat outcomes.
    """

    valid = (
        jnp.ones_like(done, dtype=jnp.bool_)
        if valid is None
        else jnp.asarray(valid, dtype=jnp.bool_)
    )
    target_dead = combat_state.health[:, TARGET_ENTITY] <= 0.0
    agent_dead = combat_state.health[:, AGENT_ENTITY] <= 0.0
    success = done & valid & target_dead & ~agent_dead
    death = done & valid & agent_dead & ~target_dead
    simultaneous = done & valid & target_dead & agent_dead
    other = done & ~success & ~death & ~simultaneous
    invalid = other & ~valid
    geometry_exhausted = other & valid & combat_state.geometry_exhausted
    target_navigation_unsupported = (
        other & valid & ~geometry_exhausted & combat_state.target_navigation_unsupported
    )
    return EpisodeOutcome(
        success=success,
        death=death,
        simultaneous=simultaneous,
        other=other,
        geometry_exhausted=geometry_exhausted,
        target_navigation_unsupported=target_navigation_unsupported,
        invalid=invalid,
        unclassified=(
            other & ~geometry_exhausted & ~target_navigation_unsupported & ~invalid
        ),
        time_limit=jnp.zeros_like(done, dtype=jnp.bool_),
    )


def initialize_training(
    key: jax.Array,
    config: PPOConfig,
    environment_params: CombatParams | None = None,
    *,
    environment: PPOEnvironment | None = None,
) -> PPOTrainState:
    """Create policy, optimizer, batched environments, and recurrent state."""

    selected_environment = _select_environment(
        environment_params,
        environment,
    )
    _validate_environment_spec(config, selected_environment.spec)
    policy_key, environment_key = jax.random.split(key)
    policy_params = initialize_policy(policy_key, config)
    optimizer = _make_optimizer(config)
    reset_keys = jax.random.split(environment_key, config.num_envs)
    environment_state, observation, action_mask = selected_environment.reset(reset_keys)
    return PPOTrainState(
        policy_params=policy_params,
        optimizer_state=optimizer.init(policy_params),
        environment_state=environment_state,
        observation=observation,
        action_mask=action_mask,
        recurrent_state=jnp.zeros(
            (config.num_envs, config.recurrent_size),
            dtype=jnp.float32,
        ),
        episode_start=jnp.ones((config.num_envs,), dtype=jnp.bool_),
        running_episode_return=jnp.zeros(
            (config.num_envs,),
            dtype=jnp.float32,
        ),
        running_episode_length=jnp.zeros(
            (config.num_envs,),
            dtype=jnp.int32,
        ),
        update_count=jnp.asarray(0, dtype=jnp.int32),
        total_environment_steps=jnp.asarray(0, dtype=jnp.int32),
    )


def make_train_step(
    config: PPOConfig,
    environment_params: CombatParams | None = None,
    *,
    environment: PPOEnvironment | None = None,
    compile: bool = True,
) -> TrainStep:
    """Build one update containing rollout, GAE, PPO epochs, and optimizer.

    The returned callable has one host boundary. Its nested scans cover rollout
    time, engine microticks, recurrent sequence evaluation, minibatches, and
    PPO epochs.
    """

    optimizer = _make_optimizer(config)
    selected_environment = _select_environment(
        environment_params,
        environment,
    )

    def train_step(
        train_state: PPOTrainState,
        key: jax.Array,
    ) -> tuple[PPOTrainState, PPOMetrics]:
        rollout_key, update_key = jax.random.split(key)
        rollout_state, update_batch = _rollout_phase(
            train_state,
            rollout_key,
            config,
            selected_environment,
        )
        return _policy_update_phase(
            rollout_state,
            update_batch,
            update_key,
            config,
            optimizer,
        )

    if compile:
        return jax.jit(train_step)
    return train_step


def make_rollout_collector(
    config: PPOConfig,
    environment_params: CombatParams | None = None,
    *,
    environment: PPOEnvironment | None = None,
    compile: bool = True,
):
    """Build the environment/GAE half of one PPO update."""

    selected_environment = _select_environment(
        environment_params,
        environment,
    )

    def collect(train_state: PPOTrainState, key: jax.Array):
        return _rollout_phase(
            train_state,
            key,
            config,
            selected_environment,
        )

    return jax.jit(collect) if compile else collect


def make_policy_update(
    config: PPOConfig,
    *,
    compile: bool = True,
):
    """Build the recurrent PPO epoch/minibatch optimizer half."""

    optimizer = _make_optimizer(config)

    def update(
        train_state: PPOTrainState,
        batch: PPOUpdateBatch,
        key: jax.Array,
    ):
        return _policy_update_phase(
            train_state,
            batch,
            key,
            config,
            optimizer,
        )

    return jax.jit(update) if compile else update


def ppo_optimizer(config: PPOConfig) -> optax.GradientTransformation:
    """Return the canonical PPO optimizer for alternate rollout adapters."""

    return _make_optimizer(config)


def update_recurrent_policy(
    policy_params: RecurrentPolicyParams,
    optimizer_state,
    optimizer: optax.GradientTransformation,
    rollout: RecurrentRollout,
    advantages: jax.Array,
    returns: jax.Array,
    initial_recurrent_state: jax.Array,
    key: jax.Array,
    config: PPOConfig,
):
    """Apply the canonical recurrent PPO update to a validated rollout.

    ``advantages`` arrive raw from :func:`calculate_gae` and are standardized
    here, which is the same step :func:`_policy_update_phase` applies before it
    calls ``_ppo_update`` directly. Doing it inside the update rather than in
    each adapter keeps the two entry points on one definition of the canonical
    update, and it is what makes the surrogate objective scale-free: without
    it, the policy gradient is proportional to the reward magnitude, and a
    reward whose sign is near-constant over a batch pushes every sampled action
    down uniformly instead of ranking them against each other.
    """

    normalized_advantages = (advantages - jnp.mean(advantages)) / (
        jnp.std(advantages) + jnp.float32(1.0e-8)
    )
    return _ppo_update(
        policy_params,
        optimizer_state,
        optimizer,
        rollout,
        normalized_advantages,
        returns,
        initial_recurrent_state,
        key,
        config,
    )


def calculate_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    bootstrap_value: jax.Array,
    *,
    gamma: float,
    gae_lambda: float,
    time_limit_bootstrap_values: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Calculate GAE, bootstrapping exact time limits without crossing resets."""

    gamma_value = jnp.float32(gamma)
    lambda_value = jnp.float32(gae_lambda)

    timeout_values = (
        jnp.zeros_like(rewards)
        if time_limit_bootstrap_values is None
        else jnp.asarray(time_limit_bootstrap_values, dtype=values.dtype)
    )
    if timeout_values.shape != rewards.shape:
        raise ValueError("time_limit_bootstrap_values must match rewards")

    def reverse_step(carry, transition):
        next_advantage, next_value = carry
        reward, value, done, timeout_value = transition
        nonterminal = jnp.float32(1.0) - done.astype(jnp.float32)
        delta = (
            reward + gamma_value * (nonterminal * next_value + timeout_value) - value
        )
        advantage = delta + gamma_value * lambda_value * nonterminal * next_advantage
        return (advantage, value), advantage

    initial_advantage = jnp.zeros_like(bootstrap_value)
    (_, _), advantages = jax.lax.scan(
        reverse_step,
        (initial_advantage, bootstrap_value),
        (rewards, values, dones, timeout_values),
        reverse=True,
    )
    return advantages, advantages + values


def _rollout_phase(
    train_state: PPOTrainState,
    key: jax.Array,
    config: PPOConfig,
    environment: PPOEnvironment,
) -> tuple[PPOTrainState, PPOUpdateBatch]:
    initial_recurrent_state = train_state.recurrent_state
    (
        final_environment_state,
        final_observation,
        final_action_mask,
        final_recurrent_state,
        final_episode_start,
        final_running_return,
        final_running_length,
        rollout,
    ) = _collect_rollout(
        train_state,
        key,
        config,
        environment,
    )
    bootstrap_recurrent_state = jnp.where(
        final_episode_start[:, None],
        jnp.float32(0.0),
        final_recurrent_state,
    )
    _, _, bootstrap_value = apply_policy(
        train_state.policy_params,
        final_observation,
        bootstrap_recurrent_state,
        final_action_mask,
    )
    advantages, returns = calculate_gae(
        rollout.reward,
        rollout.value,
        rollout.done,
        bootstrap_value,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        time_limit_bootstrap_values=rollout.time_limit_bootstrap_value,
    )
    batch = PPOUpdateBatch(
        rollout=rollout,
        advantages=jax.lax.stop_gradient(advantages),
        returns=jax.lax.stop_gradient(returns),
        initial_recurrent_state=initial_recurrent_state,
    )
    advanced_state = train_state._replace(
        environment_state=final_environment_state,
        observation=final_observation,
        action_mask=final_action_mask,
        recurrent_state=final_recurrent_state,
        episode_start=final_episode_start,
        running_episode_return=final_running_return,
        running_episode_length=final_running_length,
        total_environment_steps=(
            train_state.total_environment_steps
            + jnp.int32(config.num_envs * config.rollout_steps)
        ),
    )
    return advanced_state, batch


def _policy_update_phase(
    train_state: PPOTrainState,
    batch: PPOUpdateBatch,
    key: jax.Array,
    config: PPOConfig,
    optimizer: optax.GradientTransformation,
) -> tuple[PPOTrainState, PPOMetrics]:
    normalized_advantages = (batch.advantages - jnp.mean(batch.advantages)) / (
        jnp.std(batch.advantages) + jnp.float32(1.0e-8)
    )
    (
        updated_policy_params,
        updated_optimizer_state,
        loss_metrics,
    ) = _ppo_update(
        train_state.policy_params,
        train_state.optimizer_state,
        optimizer,
        batch.rollout,
        normalized_advantages,
        batch.returns,
        batch.initial_recurrent_state,
        key,
        config,
    )
    rollout = batch.rollout
    completed_count = jnp.sum(rollout.done, dtype=jnp.int32)
    completed_denominator = jnp.maximum(
        completed_count.astype(jnp.float32),
        jnp.float32(1.0),
    )
    return_variance = jnp.var(batch.returns)
    explained_variance = jnp.where(
        return_variance > jnp.float32(1.0e-8),
        jnp.float32(1.0) - jnp.var(batch.returns - rollout.value) / return_variance,
        jnp.float32(0.0),
    )
    updated_count = train_state.update_count + jnp.int32(1)
    metrics = PPOMetrics(
        total_loss=jnp.mean(loss_metrics.total_loss),
        policy_loss=jnp.mean(loss_metrics.policy_loss),
        value_loss=jnp.mean(loss_metrics.value_loss),
        entropy=jnp.mean(loss_metrics.entropy),
        approximate_kl=jnp.mean(loss_metrics.approximate_kl),
        clip_fraction=jnp.mean(loss_metrics.clip_fraction),
        incomparable_fraction=jnp.mean(loss_metrics.incomparable_fraction),
        log_ratio_median=jnp.mean(loss_metrics.log_ratio_median),
        log_ratio_p999=jnp.mean(loss_metrics.log_ratio_p999),
        # Max rather than mean here so one runaway minibatch is not averaged
        # away. The multi-actor trainer folds every loss field with a single
        # mean, so on that path this field is the average of the per-minibatch
        # maxima instead; read it as "typical worst", not "worst".
        maximum_log_ratio=jnp.max(loss_metrics.maximum_log_ratio),
        explained_variance=explained_variance,
        mean_rollout_reward=jnp.mean(rollout.reward),
        episodes_completed=completed_count,
        episode_successes=jnp.sum(
            rollout.completed_episode_success,
            dtype=jnp.int32,
        ),
        episode_deaths=jnp.sum(
            rollout.completed_episode_death,
            dtype=jnp.int32,
        ),
        episode_simultaneous=jnp.sum(
            rollout.completed_episode_simultaneous,
            dtype=jnp.int32,
        ),
        episode_other_terminal=jnp.sum(
            rollout.completed_episode_other,
            dtype=jnp.int32,
        ),
        episode_geometry_exhausted=jnp.sum(
            rollout.completed_episode_geometry_exhausted,
            dtype=jnp.int32,
        ),
        episode_target_navigation_unsupported=jnp.sum(
            rollout.completed_episode_target_navigation_unsupported,
            dtype=jnp.int32,
        ),
        episode_invalid=jnp.sum(
            rollout.completed_episode_invalid,
            dtype=jnp.int32,
        ),
        episode_unclassified=jnp.sum(
            rollout.completed_episode_unclassified,
            dtype=jnp.int32,
        ),
        episode_time_limit=jnp.sum(
            rollout.completed_episode_time_limit,
            dtype=jnp.int32,
        ),
        mean_episode_return=(
            jnp.sum(rollout.completed_episode_return) / completed_denominator
        ),
        mean_episode_length=(
            jnp.sum(
                rollout.completed_episode_length,
                dtype=jnp.float32,
            )
            / completed_denominator
        ),
        update_count=updated_count,
        total_environment_steps=train_state.total_environment_steps,
    )
    return (
        train_state._replace(
            policy_params=updated_policy_params,
            optimizer_state=updated_optimizer_state,
            update_count=updated_count,
        ),
        metrics,
    )


def _collect_rollout(
    train_state: PPOTrainState,
    key: jax.Array,
    config: PPOConfig,
    environment: PPOEnvironment,
):
    scan_keys = jax.random.split(key, config.rollout_steps)

    def rollout_step(carry, step_key):
        (
            environment_state,
            observation,
            action_mask,
            recurrent_state,
            episode_start,
            running_return,
            running_length,
        ) = carry
        action_key, environment_key, reset_key = jax.random.split(
            step_key,
            3,
        )
        recurrent_input = jnp.where(
            episode_start[:, None],
            jnp.float32(0.0),
            recurrent_state,
        )
        next_recurrent_state, logits, value = apply_policy(
            train_state.policy_params,
            observation,
            recurrent_input,
            action_mask,
        )
        action_ids, log_probability = sample_configured_actions(
            action_key,
            logits,
            config,
        )
        environment_keys = jax.random.split(
            environment_key,
            config.num_envs,
        )
        (
            stepped_environment_state,
            stepped_observation,
            reward,
            environment_done,
            stepped_action_mask,
        ) = environment.step(
            environment_state,
            observation,
            action_ids,
            environment_keys,
        )
        accumulated_return = running_return + reward
        accumulated_length = running_length + jnp.int32(1)
        if config.episode_horizon_ticks is None:
            time_limit = jnp.zeros_like(environment_done, dtype=jnp.bool_)
            time_limit_bootstrap_value = jnp.zeros_like(value)
        else:
            time_limit = ~environment_done & (
                accumulated_length >= jnp.int32(config.episode_horizon_ticks)
            )

            def bootstrap_time_limit(_):
                _, _, final_value = apply_policy(
                    train_state.policy_params,
                    stepped_observation,
                    next_recurrent_state,
                    stepped_action_mask,
                )
                return jnp.where(time_limit, final_value, jnp.float32(0.0))

            time_limit_bootstrap_value = jax.lax.cond(
                jnp.any(time_limit),
                bootstrap_time_limit,
                lambda _: jnp.zeros_like(value),
                operand=None,
            )
        done = environment_done | time_limit
        episode_outcome = _episode_outcome(
            environment,
            stepped_environment_state,
            environment_done,
        )._replace(time_limit=time_limit)

        def reset_terminated(_):
            (
                reset_environment_state,
                reset_observation,
                reset_action_mask,
            ) = environment.reset(
                jax.random.split(reset_key, config.num_envs),
            )
            return (
                _batch_select(
                    done,
                    reset_environment_state,
                    stepped_environment_state,
                ),
                jnp.where(
                    done[:, None],
                    reset_observation,
                    stepped_observation,
                ),
                jnp.where(
                    done[:, None],
                    reset_action_mask,
                    stepped_action_mask,
                ),
            )

        next_environment_state, next_observation, next_action_mask = jax.lax.cond(
            jnp.any(done),
            reset_terminated,
            lambda _: (
                stepped_environment_state,
                stepped_observation,
                stepped_action_mask,
            ),
            operand=None,
        )
        next_recurrent_state = jnp.where(
            done[:, None],
            jnp.float32(0.0),
            next_recurrent_state,
        )

        completed_return = jnp.where(
            done,
            accumulated_return,
            jnp.float32(0.0),
        )
        completed_length = jnp.where(
            done,
            accumulated_length,
            jnp.int32(0),
        )
        next_running_return = jnp.where(
            done,
            jnp.float32(0.0),
            accumulated_return,
        )
        next_running_length = jnp.where(
            done,
            jnp.int32(0),
            accumulated_length,
        )
        transition = RecurrentRollout(
            observation=observation,
            action_mask=action_mask,
            action=action_ids,
            log_probability=log_probability,
            value=value,
            reward=reward,
            done=done,
            episode_start=episode_start,
            completed_episode_return=completed_return,
            completed_episode_length=completed_length,
            completed_episode_success=episode_outcome.success,
            completed_episode_death=episode_outcome.death,
            completed_episode_simultaneous=episode_outcome.simultaneous,
            completed_episode_other=episode_outcome.other,
            completed_episode_geometry_exhausted=(episode_outcome.geometry_exhausted),
            completed_episode_target_navigation_unsupported=(
                episode_outcome.target_navigation_unsupported
            ),
            completed_episode_invalid=episode_outcome.invalid,
            completed_episode_unclassified=episode_outcome.unclassified,
            completed_episode_time_limit=episode_outcome.time_limit,
            time_limit_bootstrap_value=time_limit_bootstrap_value,
        )
        next_carry = (
            next_environment_state,
            next_observation,
            next_action_mask,
            next_recurrent_state,
            done,
            next_running_return,
            next_running_length,
        )
        return next_carry, transition

    initial_carry = (
        train_state.environment_state,
        train_state.observation,
        train_state.action_mask,
        train_state.recurrent_state,
        train_state.episode_start,
        train_state.running_episode_return,
        train_state.running_episode_length,
    )
    final_carry, rollout = jax.lax.scan(
        rollout_step,
        initial_carry,
        scan_keys,
    )
    return (*final_carry, rollout)


def _ppo_update(
    policy_params: RecurrentPolicyParams,
    optimizer_state,
    optimizer: optax.GradientTransformation,
    rollout: RecurrentRollout,
    advantages: jax.Array,
    returns: jax.Array,
    initial_recurrent_state: jax.Array,
    key: jax.Array,
    config: PPOConfig,
):
    epoch_keys = jax.random.split(key, config.update_epochs)

    def loss_for_minibatch(params, minibatch):
        return _ppo_loss(params, minibatch, config)

    def epoch_step(carry, epoch_key):
        current_params, current_optimizer_state = carry
        permutation = jax.random.permutation(
            epoch_key,
            config.num_envs,
        )
        minibatch_indices = permutation.reshape(
            (
                config.num_minibatches,
                config.num_envs // config.num_minibatches,
            )
        )

        def minibatch_step(minibatch_carry, indices):
            minibatch_params, minibatch_optimizer_state = minibatch_carry
            minibatch = (
                rollout.observation[:, indices],
                rollout.action_mask[:, indices],
                rollout.action[:, indices],
                jax.lax.stop_gradient(rollout.log_probability[:, indices]),
                jax.lax.stop_gradient(rollout.value[:, indices]),
                rollout.episode_start[:, indices],
                jax.lax.stop_gradient(advantages[:, indices]),
                jax.lax.stop_gradient(returns[:, indices]),
                jax.lax.stop_gradient(initial_recurrent_state[indices]),
            )
            (
                (
                    _,
                    loss_metrics,
                ),
                gradients,
            ) = jax.value_and_grad(
                loss_for_minibatch,
                has_aux=True,
            )(
                minibatch_params,
                minibatch,
            )
            updates, minibatch_optimizer_state = optimizer.update(
                gradients,
                minibatch_optimizer_state,
                minibatch_params,
            )
            minibatch_params = optax.apply_updates(
                minibatch_params,
                updates,
            )
            return (
                minibatch_params,
                minibatch_optimizer_state,
            ), loss_metrics

        (
            (
                current_params,
                current_optimizer_state,
            ),
            epoch_metrics,
        ) = jax.lax.scan(
            minibatch_step,
            (current_params, current_optimizer_state),
            minibatch_indices,
        )
        return (
            current_params,
            current_optimizer_state,
        ), epoch_metrics

    (
        (
            final_params,
            final_optimizer_state,
        ),
        metrics,
    ) = jax.lax.scan(
        epoch_step,
        (policy_params, optimizer_state),
        epoch_keys,
    )
    return final_params, final_optimizer_state, metrics


def _ppo_loss(
    policy_params: RecurrentPolicyParams,
    minibatch,
    config: PPOConfig,
) -> tuple[jax.Array, LossMetrics]:
    (
        observations,
        action_masks,
        actions,
        old_log_probabilities,
        old_values,
        episode_starts,
        advantages,
        returns,
        initial_recurrent_state,
    ) = minibatch
    _, logits, values = apply_policy_sequence(
        policy_params,
        observations,
        initial_recurrent_state,
        episode_starts,
        action_masks,
    )
    log_probabilities = configured_action_log_probabilities(
        logits,
        actions,
        config,
    )
    entropy = configured_action_entropy(logits, config)
    # An illegal factored action scores as a sentinel log-probability, not a
    # real one, and masked logits carry the same sentinel magnitude. If an
    # action's legality differs between the stored rollout score and this one,
    # their difference is a sentinel gap rather than a policy change, and
    # exponentiating it yields inf. Exclude those samples: a real PPO log-ratio
    # is order 1, and the surrogate is clipped to [1-eps, 1+eps], so nothing
    # legitimate lies beyond the bound. The excluded fraction is reported so a
    # mask/legality disagreement surfaces instead of being absorbed here.
    raw_log_ratio = log_probabilities - old_log_probabilities
    comparable = jnp.abs(raw_log_ratio) <= jnp.float32(_MAXIMUM_COMPARABLE_LOG_RATIO)
    log_ratio = jnp.where(comparable, raw_log_ratio, jnp.float32(0.0))
    ratio = jnp.exp(log_ratio)
    clip_epsilon = jnp.float32(config.clip_epsilon)
    unclipped_policy_loss = -advantages * ratio
    clipped_policy_loss = -advantages * jnp.clip(
        ratio,
        jnp.float32(1.0) - clip_epsilon,
        jnp.float32(1.0) + clip_epsilon,
    )
    per_sample_policy_loss = jnp.maximum(
        unclipped_policy_loss,
        clipped_policy_loss,
    )
    comparable_weight = comparable.astype(jnp.float32)
    comparable_count = jnp.sum(comparable_weight)
    # Mean over comparable samples only. Dividing by the full count would shrink
    # the gradient in proportion to how many samples were dropped, which would
    # quietly turn a masking bug into a smaller learning rate.
    policy_loss = jnp.sum(per_sample_policy_loss * comparable_weight) / jnp.maximum(
        comparable_count, jnp.float32(1.0)
    )
    incomparable_fraction = jnp.float32(1.0) - jnp.mean(comparable_weight)

    clipped_values = old_values + jnp.clip(
        values - old_values,
        -clip_epsilon,
        clip_epsilon,
    )
    value_loss = jnp.float32(0.5) * jnp.mean(
        jnp.maximum(
            jnp.square(values - returns),
            jnp.square(clipped_values - returns),
        )
    )
    mean_entropy = jnp.mean(entropy)
    total_loss = (
        policy_loss
        + jnp.float32(config.value_coefficient) * value_loss
        - jnp.float32(config.entropy_coefficient) * mean_entropy
    )
    # Both statistics average over COMPARABLE samples. A dropped sample sits at
    # ratio 1, so counting it would report a policy that moved less than it did
    # and would hide the drop from the guardrail that reads this number.
    denominator = jnp.maximum(comparable_count, jnp.float32(1.0))
    approximate_kl = (
        jnp.sum(((ratio - jnp.float32(1.0)) - log_ratio) * comparable_weight)
        / denominator
    )
    clip_fraction = (
        jnp.sum(
            (jnp.abs(ratio - jnp.float32(1.0)) > clip_epsilon).astype(jnp.float32)
            * comparable_weight
        )
        / denominator
    )
    # Quantiles over the comparable samples only. An incomparable sample is
    # excluded by making it NaN rather than zero: zero is a valid log ratio and
    # would pull the quantiles toward "nothing moved".
    comparable_log_ratio = jnp.where(
        comparable, jnp.abs(raw_log_ratio), jnp.float32(jnp.nan)
    )
    log_ratio_median = jnp.nanquantile(comparable_log_ratio, jnp.float32(0.5))
    log_ratio_p999 = jnp.nanquantile(comparable_log_ratio, jnp.float32(0.999))
    maximum_log_ratio = jnp.nanmax(comparable_log_ratio)
    return total_loss, LossMetrics(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=mean_entropy,
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
        incomparable_fraction=incomparable_fraction,
        log_ratio_median=log_ratio_median,
        log_ratio_p999=log_ratio_p999,
        maximum_log_ratio=maximum_log_ratio,
    )


def _batch_select(
    mask: jax.Array,
    when_true,
    when_false,
):
    """Reset batched rows while retaining immutable shared environment data."""

    def select_leaf(true_leaf, false_leaf):
        if true_leaf.shape != false_leaf.shape:
            raise ValueError(
                "PPO environment reset changed a state-leaf shape: "
                f"{true_leaf.shape!r} != {false_leaf.shape!r}"
            )
        if true_leaf.dtype != false_leaf.dtype:
            raise ValueError(
                "PPO environment reset changed a state-leaf dtype: "
                f"{true_leaf.dtype!r} != {false_leaf.dtype!r}"
            )
        if true_leaf.ndim == 0 or true_leaf.shape[0] != mask.shape[0]:
            # Fixed environment data (for example an exact-Region atlas) is
            # shared by the batch and cannot be reset for only some rows.
            # Keep the live value; reset/step must agree on its shape/dtype.
            return false_leaf
        expanded_mask = mask.reshape((mask.shape[0],) + (1,) * (true_leaf.ndim - 1))
        return jnp.where(expanded_mask, true_leaf, false_leaf)

    return jax.tree.map(select_leaf, when_true, when_false)


def _make_optimizer(config: PPOConfig) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(config.max_gradient_norm),
        optax.adam(
            config.learning_rate,
            eps=1.0e-5,
        ),
    )


def _select_environment(
    environment_params: CombatParams | None,
    environment: PPOEnvironment | None,
) -> PPOEnvironment:
    if environment is not None:
        if environment_params is not None:
            raise ValueError("provide environment_params or environment, not both")
        return environment
    if environment_params is None:
        raise ValueError("environment_params or environment is required")

    def reset(keys):
        state, observation = reset_batch(keys, environment_params)
        return state, observation, legal_skill_mask(observation)

    def step(state, observation, action_ids, keys):
        actions = skills_to_actions(observation, action_ids)
        next_state, next_observation, reward, done, _ = step_batch(
            state,
            actions,
            keys,
            environment_params,
        )
        return (
            next_state,
            next_observation,
            reward,
            done,
            legal_skill_mask(next_observation),
        )

    return PPOEnvironment(
        reset=reset,
        step=step,
        spec=EnvironmentSpec(
            observation_schema="active_combat_v1",
            action_schema="combat_skills_v1",
            action_components=(ActionComponentSpec("skill", "discrete", SKILL_COUNT),),
            observation_size=ACTIVE_OBSERVATION_SIZE,
            action_size=SKILL_COUNT,
        ),
        episode_outcome=combat_episode_outcome,
    )


def _episode_outcome(
    environment: PPOEnvironment,
    stepped_environment_state: Any,
    done: jax.Array,
) -> EpisodeOutcome:
    """Return an exhaustive terminal taxonomy for one batched step."""

    if environment.episode_outcome is None:
        false = jnp.zeros_like(done, dtype=jnp.bool_)
        return EpisodeOutcome(
            false, false, false, done, false, false, false, done, false
        )
    outcome = environment.episode_outcome(stepped_environment_state, done)
    return jax.tree.map(
        lambda value: done & jnp.asarray(value, dtype=jnp.bool_),
        outcome,
    )


def _validate_environment_spec(
    config: PPOConfig,
    environment_spec: EnvironmentSpec,
) -> None:
    if not environment_spec.is_dense:
        raise ValueError("PPO requires an explicitly dense environment adapter")
    if config.observation_size != environment_spec.observation_size:
        raise ValueError(
            "PPO observation_size does not match environment spec: "
            f"{config.observation_size} != {environment_spec.observation_size}"
        )
    if config.action_size != environment_spec.action_size:
        raise ValueError(
            "PPO action_size does not match environment spec: "
            f"{config.action_size} != {environment_spec.action_size}"
        )


__all__ = [
    "PPOEnvironment",
    "calculate_gae",
    "combat_episode_outcome",
    "initialize_training",
    "make_train_step",
]
