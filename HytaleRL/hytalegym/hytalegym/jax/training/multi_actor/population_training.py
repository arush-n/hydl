"""Independent optimizer ownership for several trainable policy-bank rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.training.ppo import ppo_optimizer, update_recurrent_policy
from hytalegym.jax.training.types import (
    LossMetrics,
    PPOConfig,
    RecurrentPolicyParams,
)

from .assignment import PolicyActorAssignment
from .rollout import MultiActorRolloutResult
from .runtime import MultiActorArenaState
from .training import _empty_loss_metrics, _gather_trainable_rollout


class MultiActorPopulationTrainingState(NamedTuple):
    """Policy bank with one optimizer state per trainable policy ID."""

    policy_bank: RecurrentPolicyParams
    optimizer_states: tuple[Any, ...]
    arena: MultiActorArenaState
    recurrent_state: jax.Array
    episode_start: jax.Array
    running_episode_return: jax.Array
    running_episode_length: jax.Array
    update_count: jax.Array
    total_trainable_actor_steps: jax.Array


class MultiActorPopulationEpisodeOutcomes(NamedTuple):
    """Per-episode facts on ``[K_train,T,B_policy]``.

    Population summaries are useful for reporting but cannot drive a rating
    system: several opponents can contribute to one policy's aggregate.  The
    static rows below retain the exact completed-episode classification for
    host-side league bookkeeping without moving matchmaking into the JIT.
    """

    done: jax.Array
    success: jax.Array
    death: jax.Array
    simultaneous: jax.Array
    other: jax.Array


class MultiActorPopulationMetrics(NamedTuple):
    """Metrics on the static trainable-policy axis ``K_train``."""

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
    episode_outcomes: MultiActorPopulationEpisodeOutcomes


@dataclass(frozen=True)
class MultiActorPopulationTrainer:
    """Host initializer and one compiled balanced population update."""

    initialize: Callable
    train_step: Callable
    trainable_policy_ids: tuple[int, ...]
    trainable_actor_rows: tuple[tuple[tuple[int, int], ...], ...]
    trainable_actor_count_per_policy: int


@dataclass(frozen=True)
class MultiActorPopulationPlan:
    """Static policy ownership used by optimizer and checkpoint routing."""

    trainable_policy_ids: tuple[int, ...]
    trainable_actor_rows: tuple[tuple[tuple[int, int], ...], ...]
    trainable_actor_count_per_policy: int


def plan_multi_actor_population(
    assignment: PolicyActorAssignment,
    config: PPOConfig,
    *,
    bank_size: int | None = None,
) -> MultiActorPopulationPlan:
    """Validate and expose the balanced static population ownership plan."""

    if bank_size is not None:
        if isinstance(bank_size, bool) or not isinstance(bank_size, int):
            raise TypeError("bank_size must be an integer")
        if bank_size < 1:
            raise ValueError("bank_size must be positive")
    active = np.asarray(jax.device_get(assignment.active), dtype=np.bool_)
    trainable = np.asarray(jax.device_get(assignment.trainable), dtype=np.bool_)
    policy_ids = np.asarray(jax.device_get(assignment.policy_id), dtype=np.int32)
    if bank_size is not None and np.any(
        active & ((policy_ids < 0) | (policy_ids >= bank_size))
    ):
        raise ValueError("active policy_id is outside the policy bank")
    selected_ids = tuple(
        int(value) for value in np.unique(policy_ids[trainable]).tolist()
    )
    if not selected_ids:
        raise ValueError("at least one policy ID must be trainable")
    actor_rows = []
    for policy_id in selected_ids:
        rows = np.argwhere(trainable & (policy_ids == policy_id))
        if rows.shape[0] != config.num_envs:
            raise ValueError(
                "each trainable policy must own exactly PPO num_envs actor "
                f"rows: policy {policy_id} owns {rows.shape[0]}, expected "
                f"{config.num_envs}"
            )
        actor_rows.append(
            tuple((int(batch), int(slot)) for batch, slot in rows.tolist())
        )
    return MultiActorPopulationPlan(
        trainable_policy_ids=selected_ids,
        trainable_actor_rows=tuple(actor_rows),
        trainable_actor_count_per_policy=config.num_envs,
    )


def make_multi_actor_population_trainer(
    collector: Callable[..., MultiActorRolloutResult],
    assignment: PolicyActorAssignment,
    config: PPOConfig,
    *,
    compile: bool = True,
) -> MultiActorPopulationTrainer:
    """Train several policy IDs with isolated optimizer states in one rollout.

    V1 deliberately requires a balanced number of actor rows per trainable
    policy. That preserves one static PPO batch shape and identical optimizer
    semantics for every policy without padding fake behavior into an update.
    Frozen policy-bank rows remain inference-only and byte-identical.
    """

    plan = plan_multi_actor_population(assignment, config)
    selected_ids = plan.trainable_policy_ids
    actor_rows = tuple(
        (
            jnp.asarray(tuple(row[0] for row in rows), dtype=jnp.int32),
            jnp.asarray(tuple(row[1] for row in rows), dtype=jnp.int32),
        )
        for rows in plan.trainable_actor_rows
    )
    optimizer = ppo_optimizer(config)
    trainable_policy_count = len(selected_ids)
    batch, policy_slots = assignment.actor_index.shape
    recorded_slots = tuple(
        getattr(collector, "record_policy_slots", tuple(range(policy_slots)))
    )
    try:
        rollout_slot_rows = tuple(
            jnp.asarray(
                tuple(recorded_slots.index(row[1]) for row in rows),
                dtype=jnp.int32,
            )
            for rows in plan.trainable_actor_rows
        )
    except ValueError as error:
        raise ValueError("collector must record every trainable actor slot") from error

    def initialize(
        policy_bank: RecurrentPolicyParams,
        arena: MultiActorArenaState,
    ) -> MultiActorPopulationTrainingState:
        leaves = jax.tree_util.tree_leaves(policy_bank)
        if not leaves:
            raise ValueError("policy bank cannot be empty")
        bank_size = leaves[0].shape[0]
        if any(leaf.shape[0] != bank_size for leaf in leaves):
            raise ValueError("policy-bank leaves must share their K axis")
        plan_multi_actor_population(
            assignment,
            config,
            bank_size=bank_size,
        )
        optimizer_states = tuple(
            optimizer.init(
                jax.tree_util.tree_map(
                    lambda value, row=policy_id: value[row],
                    policy_bank,
                )
            )
            for policy_id in selected_ids
        )
        return MultiActorPopulationTrainingState(
            policy_bank=policy_bank,
            optimizer_states=optimizer_states,
            arena=arena,
            recurrent_state=jnp.zeros(
                (batch, policy_slots, config.recurrent_size),
                dtype=jnp.float32,
            ),
            episode_start=assignment.active,
            running_episode_return=jnp.zeros(
                (batch, policy_slots), dtype=jnp.float32
            ),
            running_episode_length=jnp.zeros(
                (batch, policy_slots), dtype=jnp.int32
            ),
            update_count=jnp.zeros(
                (trainable_policy_count,), dtype=jnp.int32
            ),
            total_trainable_actor_steps=jnp.zeros(
                (trainable_policy_count,), dtype=jnp.int32
            ),
        )

    def train_step(state: MultiActorPopulationTrainingState, key: jax.Array):
        rollout_key, *update_keys = jax.random.split(
            key,
            trainable_policy_count + 1,
        )
        result = collector(
            state.arena,
            state.policy_bank,
            state.recurrent_state,
            state.episode_start,
            state.running_episode_return,
            state.running_episode_length,
            rollout_key,
        )
        policy_bank = state.policy_bank
        optimizer_states = []
        losses = []
        applied_rows = []
        invalid_rows = []
        completed_rows = []
        success_rows = []
        death_rows = []
        simultaneous_rows = []
        other_rows = []
        return_rows = []
        length_rows = []
        outcome_done_rows = []
        outcome_success_rows = []
        outcome_death_rows = []
        outcome_simultaneous_rows = []
        outcome_other_rows = []

        for index, policy_id in enumerate(selected_ids):
            batch_index, slot_index = actor_rows[index]
            rollout_slot_index = rollout_slot_rows[index]
            behavior = _gather_trainable_rollout(
                result,
                state.recurrent_state,
                batch_index,
                rollout_slot_index,
                slot_index,
                policy_bank,
                policy_id,
                config,
            )
            ready = jnp.all(
                result.rollout.trainable[:, batch_index, rollout_slot_index]
            )
            selected = jax.tree_util.tree_map(
                lambda value, row=policy_id: value[row],
                policy_bank,
            )

            def apply_update(_):
                return update_recurrent_policy(
                    selected,
                    state.optimizer_states[index],
                    optimizer,
                    behavior.rollout,
                    behavior.advantages,
                    behavior.returns,
                    behavior.initial_recurrent_state,
                    update_keys[index],
                    config,
                )

            def skip_update(_):
                zeros = jax.tree_util.tree_map(
                    lambda value: jnp.zeros(
                        (config.update_epochs, config.num_minibatches)
                        + value.shape,
                        dtype=value.dtype,
                    ),
                    _empty_loss_metrics(),
                )
                return selected, state.optimizer_states[index], zeros

            updated, optimizer_state, loss = jax.lax.cond(
                ready,
                apply_update,
                skip_update,
                operand=None,
            )
            policy_bank = jax.tree_util.tree_map(
                lambda bank, value, row=policy_id: bank.at[row].set(value),
                policy_bank,
                updated,
            )
            optimizer_states.append(optimizer_state)
            losses.append(jax.tree_util.tree_map(jnp.mean, loss))
            applied_rows.append(ready)
            invalid_rows.append(
                jnp.sum(
                    ~result.rollout.trainable[:, batch_index, slot_index],
                    dtype=jnp.int32,
                )
            )
            rollout = behavior.rollout
            completed = jnp.sum(rollout.done, dtype=jnp.int32)
            denominator = jnp.maximum(completed.astype(jnp.float32), 1.0)
            completed_rows.append(completed)
            success_rows.append(
                jnp.sum(rollout.completed_episode_success, dtype=jnp.int32)
            )
            death_rows.append(
                jnp.sum(rollout.completed_episode_death, dtype=jnp.int32)
            )
            simultaneous_rows.append(
                jnp.sum(
                    rollout.completed_episode_simultaneous,
                    dtype=jnp.int32,
                )
            )
            other_rows.append(
                jnp.sum(rollout.completed_episode_other, dtype=jnp.int32)
            )
            return_rows.append(
                jnp.sum(rollout.completed_episode_return) / denominator
            )
            length_rows.append(
                jnp.sum(
                    rollout.completed_episode_length,
                    dtype=jnp.float32,
                )
                / denominator
            )
            outcome_done_rows.append(rollout.done)
            outcome_success_rows.append(rollout.completed_episode_success)
            outcome_death_rows.append(rollout.completed_episode_death)
            outcome_simultaneous_rows.append(
                rollout.completed_episode_simultaneous
            )
            outcome_other_rows.append(rollout.completed_episode_other)

        applied = jnp.stack(applied_rows)
        actor_steps = (
            jnp.int32(config.rollout_steps * config.num_envs)
            * applied.astype(jnp.int32)
        )
        next_state = MultiActorPopulationTrainingState(
            policy_bank=policy_bank,
            optimizer_states=tuple(optimizer_states),
            arena=result.state,
            recurrent_state=result.recurrent_state,
            episode_start=result.episode_start,
            running_episode_return=result.running_episode_return,
            running_episode_length=result.running_episode_length,
            update_count=state.update_count + applied.astype(jnp.int32),
            total_trainable_actor_steps=(
                state.total_trainable_actor_steps + actor_steps
            ),
        )
        metrics = MultiActorPopulationMetrics(
            loss=jax.tree_util.tree_map(
                lambda *values: jnp.stack(values),
                *losses,
            ),
            update_applied=applied,
            invalid_trainable_steps=jnp.stack(invalid_rows),
            episodes_completed=jnp.stack(completed_rows),
            episode_successes=jnp.stack(success_rows),
            episode_deaths=jnp.stack(death_rows),
            episode_simultaneous=jnp.stack(simultaneous_rows),
            episode_other_terminal=jnp.stack(other_rows),
            mean_episode_return=jnp.stack(return_rows),
            mean_episode_length=jnp.stack(length_rows),
            update_count=next_state.update_count,
            total_trainable_actor_steps=(
                next_state.total_trainable_actor_steps
            ),
            episode_outcomes=MultiActorPopulationEpisodeOutcomes(
                done=jnp.stack(outcome_done_rows),
                success=jnp.stack(outcome_success_rows),
                death=jnp.stack(outcome_death_rows),
                simultaneous=jnp.stack(outcome_simultaneous_rows),
                other=jnp.stack(outcome_other_rows),
            ),
        )
        return next_state, metrics

    return MultiActorPopulationTrainer(
        initialize=initialize,
        train_step=jax.jit(train_step) if compile else train_step,
        trainable_policy_ids=selected_ids,
        trainable_actor_rows=plan.trainable_actor_rows,
        trainable_actor_count_per_policy=config.num_envs,
    )


__all__ = [
    "MultiActorPopulationEpisodeOutcomes",
    "MultiActorPopulationMetrics",
    "MultiActorPopulationPlan",
    "MultiActorPopulationTrainer",
    "MultiActorPopulationTrainingState",
    "make_multi_actor_population_trainer",
    "plan_multi_actor_population",
]
