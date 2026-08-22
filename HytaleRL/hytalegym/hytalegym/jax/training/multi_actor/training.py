"""Shared-policy PPO updates over actor-major shared-arena rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.training.policy import apply_policy
from hytalegym.jax.training.ppo import (
    calculate_gae,
    ppo_optimizer,
    update_recurrent_policy,
)
from hytalegym.jax.training.types import (
    LossMetrics,
    PPOConfig,
    RecurrentPolicyParams,
    RecurrentRollout,
)

from .assignment import PolicyActorAssignment
from .rollout import MultiActorRolloutResult
from .runtime import MultiActorArenaState


class MultiActorSharedTrainingState(NamedTuple):
    """One trainable shared policy plus frozen population and arena state."""

    policy_bank: RecurrentPolicyParams
    optimizer_state: Any
    arena: MultiActorArenaState
    recurrent_state: jax.Array
    episode_start: jax.Array
    running_episode_return: jax.Array
    running_episode_length: jax.Array
    update_count: jax.Array
    total_trainable_actor_steps: jax.Array


class MultiActorTrainingMetrics(NamedTuple):
    """PPO and liveness metrics for one actor-major update."""

    loss: LossMetrics
    update_applied: jax.Array
    invalid_trainable_steps: jax.Array
    episodes_completed: jax.Array
    episode_successes: jax.Array
    episode_deaths: jax.Array
    episode_simultaneous: jax.Array
    episode_other_terminal: jax.Array
    mean_episode_return: jax.Array
    mean_episode_length: jax.Array
    update_count: jax.Array
    total_trainable_actor_steps: jax.Array


@dataclass(frozen=True)
class MultiActorSharedPolicyTrainer:
    """Host-side initializer and one compiled shared-policy update."""

    initialize: Callable
    train_step: Callable
    trainable_policy_id: int
    trainable_actor_count: int
    donates_state: bool


def make_multi_actor_shared_policy_trainer(
    collector: Callable[..., MultiActorRolloutResult],
    assignment: PolicyActorAssignment,
    config: PPOConfig,
    *,
    compile: bool = True,
    donate_state: bool = False,
) -> MultiActorSharedPolicyTrainer:
    """Train one policy shared by any number of owned arena actors.

    All reset-pinned trainable actor slots must select the same policy ID.
    Other active slots may select arbitrary frozen policies from the bank.
    This is the minimal production-safe self-play update: frozen actors are
    used for inference but never enter the optimizer batch.
    """

    if not isinstance(compile, bool) or not isinstance(donate_state, bool):
        raise TypeError("compile and donate_state must be boolean")
    if donate_state and not compile:
        raise ValueError("state donation requires a compiled trainer")
    trainable = np.asarray(jax.device_get(assignment.trainable), dtype=np.bool_)
    policy_ids = np.asarray(jax.device_get(assignment.policy_id), dtype=np.int32)
    actor_rows = np.argwhere(trainable)
    if actor_rows.shape[0] < 1:
        raise ValueError("at least one actor slot must be trainable")
    selected_ids = np.unique(policy_ids[trainable])
    if selected_ids.shape != (1,):
        raise ValueError("shared-policy v1 requires one trainable policy_id")
    trainable_policy_id = int(selected_ids[0])
    trainable_actor_count = int(actor_rows.shape[0])
    if config.num_envs != trainable_actor_count:
        raise ValueError(
            "PPO num_envs must equal the number of trainable actor slots: "
            f"{config.num_envs} != {trainable_actor_count}"
        )
    batch_index = jnp.asarray(actor_rows[:, 0], dtype=jnp.int32)
    slot_index = jnp.asarray(actor_rows[:, 1], dtype=jnp.int32)
    policy_slots = assignment.actor_index.shape[1]
    recorded_slots = tuple(
        getattr(collector, "record_policy_slots", tuple(range(policy_slots)))
    )
    try:
        rollout_slot_index = jnp.asarray(
            [recorded_slots.index(int(slot)) for slot in actor_rows[:, 1]],
            dtype=jnp.int32,
        )
    except ValueError as error:
        raise ValueError("collector must record every trainable actor slot") from error
    optimizer = ppo_optimizer(config)

    def initialize(
        policy_bank: RecurrentPolicyParams,
        arena: MultiActorArenaState,
    ) -> MultiActorSharedTrainingState:
        leaves = jax.tree_util.tree_leaves(policy_bank)
        if not leaves or trainable_policy_id >= leaves[0].shape[0]:
            raise ValueError("trainable policy_id is outside the policy bank")
        if any(leaf.shape[0] != leaves[0].shape[0] for leaf in leaves):
            raise ValueError("policy-bank leaves must share their K axis")
        selected = jax.tree_util.tree_map(
            lambda value: value[trainable_policy_id], policy_bank
        )
        batch, policy_slots = assignment.actor_index.shape
        state = MultiActorSharedTrainingState(
            policy_bank=policy_bank,
            optimizer_state=optimizer.init(selected),
            arena=arena,
            recurrent_state=jnp.zeros(
                (batch, policy_slots, config.recurrent_size),
                dtype=jnp.float32,
            ),
            episode_start=assignment.active,
            running_episode_return=jnp.zeros((batch, policy_slots), dtype=jnp.float32),
            running_episode_length=jnp.zeros((batch, policy_slots), dtype=jnp.int32),
            update_count=jnp.int32(0),
            total_trainable_actor_steps=jnp.int32(0),
        )
        if not donate_state:
            return state
        # Reset constructors intentionally reuse immutable zero buffers across
        # many combat fields. XLA cannot donate one physical buffer through
        # multiple flattened arguments, so give each state leaf unique storage
        # once before the first donated update. Subsequent updates preserve the
        # non-aliasing layout while reusing their superseded buffers.
        return jax.tree.map(jnp.copy, state)

    def train_step(state: MultiActorSharedTrainingState, key: jax.Array):
        rollout_key, update_key = jax.random.split(key)
        result = collector(
            state.arena,
            state.policy_bank,
            state.recurrent_state,
            state.episode_start,
            state.running_episode_return,
            state.running_episode_length,
            rollout_key,
        )
        behavior = _gather_trainable_rollout(
            result,
            state.recurrent_state,
            batch_index,
            rollout_slot_index,
            slot_index,
            state.policy_bank,
            trainable_policy_id,
            config,
        )
        ready = jnp.all(result.rollout.trainable[:, batch_index, rollout_slot_index])
        selected = jax.tree_util.tree_map(
            lambda value: value[trainable_policy_id], state.policy_bank
        )

        def apply_update(_):
            return update_recurrent_policy(
                selected,
                state.optimizer_state,
                optimizer,
                behavior.rollout,
                behavior.advantages,
                behavior.returns,
                behavior.initial_recurrent_state,
                update_key,
                config,
            )

        def skip_update(_):
            zeros = jax.tree_util.tree_map(
                lambda value: jnp.zeros(
                    (config.update_epochs, config.num_minibatches) + value.shape,
                    dtype=value.dtype,
                ),
                _empty_loss_metrics(),
            )
            return selected, state.optimizer_state, zeros

        updated, optimizer_state, loss = jax.lax.cond(
            ready, apply_update, skip_update, operand=None
        )
        policy_bank = jax.tree_util.tree_map(
            lambda bank, value: bank.at[trainable_policy_id].set(value),
            state.policy_bank,
            updated,
        )
        applied = ready.astype(jnp.int32)
        actor_steps = jnp.int32(config.rollout_steps * trainable_actor_count) * applied
        next_state = MultiActorSharedTrainingState(
            policy_bank=policy_bank,
            optimizer_state=optimizer_state,
            arena=result.state,
            recurrent_state=result.recurrent_state,
            episode_start=result.episode_start,
            running_episode_return=result.running_episode_return,
            running_episode_length=result.running_episode_length,
            update_count=state.update_count + applied,
            total_trainable_actor_steps=(
                state.total_trainable_actor_steps + actor_steps
            ),
        )
        rollout = behavior.rollout
        completed = jnp.sum(rollout.done, dtype=jnp.int32)
        denominator = jnp.maximum(completed.astype(jnp.float32), 1.0)
        metrics = MultiActorTrainingMetrics(
            loss=jax.tree_util.tree_map(jnp.mean, loss),
            update_applied=ready,
            invalid_trainable_steps=jnp.sum(
                ~result.rollout.trainable[:, batch_index, rollout_slot_index],
                dtype=jnp.int32,
            ),
            episodes_completed=completed,
            episode_successes=jnp.sum(
                rollout.completed_episode_success, dtype=jnp.int32
            ),
            episode_deaths=jnp.sum(rollout.completed_episode_death, dtype=jnp.int32),
            episode_simultaneous=jnp.sum(
                rollout.completed_episode_simultaneous, dtype=jnp.int32
            ),
            episode_other_terminal=jnp.sum(
                rollout.completed_episode_other, dtype=jnp.int32
            ),
            mean_episode_return=(
                jnp.sum(rollout.completed_episode_return) / denominator
            ),
            mean_episode_length=(
                jnp.sum(rollout.completed_episode_length, dtype=jnp.float32)
                / denominator
            ),
            update_count=next_state.update_count,
            total_trainable_actor_steps=next_state.total_trainable_actor_steps,
        )
        return next_state, metrics

    compiled_step = (
        jax.jit(train_step, donate_argnums=(0,))
        if compile and donate_state
        else jax.jit(train_step)
        if compile
        else train_step
    )
    return MultiActorSharedPolicyTrainer(
        initialize=initialize,
        train_step=compiled_step,
        trainable_policy_id=trainable_policy_id,
        trainable_actor_count=trainable_actor_count,
        donates_state=donate_state,
    )


class _SharedPolicyUpdateBatch(NamedTuple):
    rollout: RecurrentRollout
    advantages: jax.Array
    returns: jax.Array
    initial_recurrent_state: jax.Array


def _gather_trainable_rollout(
    result,
    initial_recurrent_state,
    batch_index,
    rollout_slot_index,
    slot_index,
    policy_bank,
    policy_id,
    config,
):
    source = result.rollout

    def gather(value):
        return value[:, batch_index, rollout_slot_index]

    final_observation = result.observation.dense[batch_index, slot_index]
    final_mask = result.observation.action_mask[batch_index, slot_index]
    final_carry = result.recurrent_state[batch_index, slot_index]
    final_carry = jnp.where(
        result.episode_start[batch_index, slot_index, None],
        jnp.float32(0.0),
        final_carry,
    )
    selected = jax.tree_util.tree_map(lambda value: value[policy_id], policy_bank)
    _, _, bootstrap = apply_policy(selected, final_observation, final_carry, final_mask)
    reward = gather(source.reward)
    value = gather(source.value)
    done = gather(source.done)
    advantages, returns = calculate_gae(
        reward,
        value,
        done,
        bootstrap,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    rollout = RecurrentRollout(
        observation=gather(source.observation),
        action_mask=gather(source.action_mask),
        action=gather(source.action),
        log_probability=gather(source.log_probability),
        value=value,
        reward=reward,
        done=done,
        episode_start=gather(source.episode_start),
        completed_episode_return=gather(source.completed_episode_return),
        completed_episode_length=gather(source.completed_episode_length),
        completed_episode_success=gather(source.completed_episode_success),
        completed_episode_death=gather(source.completed_episode_death),
        completed_episode_simultaneous=gather(source.completed_episode_simultaneous),
        completed_episode_other=gather(source.completed_episode_other),
        completed_episode_geometry_exhausted=jnp.zeros_like(done),
        completed_episode_target_navigation_unsupported=jnp.zeros_like(done),
        completed_episode_invalid=jnp.zeros_like(done),
        completed_episode_unclassified=gather(source.completed_episode_other),
        completed_episode_time_limit=jnp.zeros_like(done),
        time_limit_bootstrap_value=jnp.zeros_like(value),
    )
    return _SharedPolicyUpdateBatch(
        rollout=rollout,
        advantages=jax.lax.stop_gradient(advantages),
        returns=jax.lax.stop_gradient(returns),
        initial_recurrent_state=initial_recurrent_state[batch_index, slot_index],
    )


def _empty_loss_metrics() -> LossMetrics:
    # Built from the field list rather than a fixed run of positionals, so
    # adding a metric cannot silently leave this call one argument short.
    zero = jnp.float32(0.0)
    return LossMetrics(**{name: zero for name in LossMetrics._fields})


__all__ = [
    "MultiActorSharedPolicyTrainer",
    "MultiActorSharedTrainingState",
    "MultiActorTrainingMetrics",
    "make_multi_actor_shared_policy_trainer",
]
