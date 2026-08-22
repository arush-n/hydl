"""Pure-JAX objective progress, reward, and termination for V2 recipes."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.worldgen.block_actions import BLOCK_SEMANTIC_KEY_WORDS

from .environment_recipe import (
    OBJECTIVE_CHECKPOINT,
    OBJECTIVE_ELIMINATE,
    OBJECTIVE_REACH,
    OBJECTIVE_SURVIVE,
    V2EnvironmentRuntime,
)


class V2MinigameActors(NamedTuple):
    """Fixed-capacity actor projection consumed by the objective kernel."""

    position: jax.Array
    team_key: jax.Array
    active: jax.Array
    alive: jax.Array


class V2MinigameState(NamedTuple):
    """Batched objective state; every field is safe to carry through JIT/scan."""

    step_count: jax.Array
    checkpoint_index: jax.Array
    team_seen: jax.Array
    target_seen: jax.Array
    team_completed: jax.Array
    episode_done: jax.Array
    episode_success: jax.Array


class V2MinigameStepResult(NamedTuple):
    """Objective transition and framework-neutral episode signals."""

    state: V2MinigameState
    reward: jax.Array
    checkpoint_advanced: jax.Array
    team_completed: jax.Array
    winner_mask: jax.Array
    objective_completed: jax.Array
    failed: jax.Array
    terminated: jax.Array
    truncated: jax.Array


def create_v2_minigame_actors(
    position: jax.Array,
    team_key: jax.Array,
    active: jax.Array,
    alive: jax.Array,
) -> V2MinigameActors:
    """Normalize and shape-check the actor fields before entering a JIT loop."""

    positions = jnp.asarray(position, dtype=jnp.float32)
    keys = jnp.asarray(team_key, dtype=jnp.uint32)
    active_mask = jnp.asarray(active, dtype=jnp.bool_)
    alive_mask = jnp.asarray(alive, dtype=jnp.bool_)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("actor positions must have shape [batch, actors, 3]")
    expected_actor_shape = positions.shape[:2]
    if keys.shape != (*expected_actor_shape, BLOCK_SEMANTIC_KEY_WORDS):
        raise ValueError(
            "actor team keys must have shape [batch, actors, key_words]"
        )
    if active_mask.shape != expected_actor_shape:
        raise ValueError("actor active mask must have shape [batch, actors]")
    if alive_mask.shape != expected_actor_shape:
        raise ValueError("actor alive mask must have shape [batch, actors]")
    return V2MinigameActors(positions, keys, active_mask, alive_mask)


def reset_v2_minigame(runtime: V2EnvironmentRuntime) -> V2MinigameState:
    """Create a zero-progress objective state for a materialized recipe batch."""

    batch_shape = runtime.objective_kind.shape
    team_shape = runtime.team_mask.shape
    return V2MinigameState(
        step_count=jnp.zeros(batch_shape, dtype=jnp.int32),
        checkpoint_index=jnp.zeros(team_shape, dtype=jnp.int32),
        team_seen=jnp.zeros(team_shape, dtype=jnp.bool_),
        target_seen=jnp.zeros(batch_shape, dtype=jnp.bool_),
        team_completed=jnp.zeros(team_shape, dtype=jnp.bool_),
        episode_done=jnp.zeros(batch_shape, dtype=jnp.bool_),
        episode_success=jnp.zeros(batch_shape, dtype=jnp.bool_),
    )


def step_v2_minigame(
    runtime: V2EnvironmentRuntime,
    state: V2MinigameState,
    actors: V2MinigameActors,
    *,
    reach_radius: float | None = None,
    checkpoint_reward: float = 0.1,
    completion_reward: float = 1.0,
    failure_penalty: float = -1.0,
) -> V2MinigameStepResult:
    """Advance sandbox, reach, checkpoint, eliminate, or survive objectives.

    Rewards are per recipe team. Termination is sticky, and calls after an
    episode ends return zero reward without changing objective progress.
    ``truncated`` is always false: a survive duration is an authored success
    condition, not an external rollout time limit.
    """

    _require_compatible_shapes(runtime, state, actors)
    live_environment = ~state.episode_done
    next_step_count = jnp.where(
        live_environment,
        state.step_count + jnp.asarray(1, dtype=jnp.int32),
        state.step_count,
    )

    assigned_actor, live_actor = _team_actor_masks(runtime, actors)
    observed_team = jnp.any(assigned_actor, axis=1)
    next_team_seen = state.team_seen | (
        observed_team & live_environment[:, None]
    )
    team_alive = jnp.any(live_actor, axis=1)
    any_team_alive = jnp.any(team_alive & runtime.team_mask, axis=1)
    any_team_seen = jnp.any(next_team_seen & runtime.team_mask, axis=1)

    is_reach = runtime.objective_kind == OBJECTIVE_REACH
    is_checkpoint = runtime.objective_kind == OBJECTIVE_CHECKPOINT
    is_eliminate = runtime.objective_kind == OBJECTIVE_ELIMINATE
    is_survive = runtime.objective_kind == OBJECTIVE_SURVIVE

    radius = (
        runtime.objective_radius
        if reach_radius is None
        else jnp.broadcast_to(
            jnp.asarray(reach_radius, dtype=jnp.float32),
            runtime.objective_radius.shape,
        )
    )
    radius_squared = radius * radius
    reach_delta = (
        actors.position - runtime.objective_position[:, 0, :][:, None, :]
    )
    actor_reached = (
        jnp.sum(reach_delta * reach_delta, axis=-1)
        <= radius_squared[:, None]
    )
    reach_hit = (
        jnp.any(live_actor & actor_reached[:, :, None], axis=1)
        & is_reach[:, None]
        & live_environment[:, None]
        & ~state.team_completed
    )
    reach_success = jnp.any(reach_hit, axis=1)

    point_count = jnp.sum(
        runtime.objective_point_mask, axis=1, dtype=jnp.int32
    )
    selected_checkpoint_index = jnp.minimum(
        state.checkpoint_index,
        jnp.maximum(point_count[:, None] - 1, 0),
    )
    batch_index = jnp.arange(runtime.objective_kind.shape[0])[:, None]
    selected_checkpoint = runtime.objective_position[
        batch_index, selected_checkpoint_index
    ]
    checkpoint_delta = (
        actors.position[:, :, None, :] - selected_checkpoint[:, None, :, :]
    )
    actor_at_checkpoint = (
        jnp.sum(checkpoint_delta * checkpoint_delta, axis=-1)
        <= radius_squared[:, None, None]
    )
    checkpoint_valid = state.checkpoint_index < point_count[:, None]
    checkpoint_hit = (
        jnp.any(live_actor & actor_at_checkpoint, axis=1)
        & checkpoint_valid
        & is_checkpoint[:, None]
        & live_environment[:, None]
        & ~state.team_completed
    )
    advanced_checkpoint_index = state.checkpoint_index + checkpoint_hit.astype(
        jnp.int32
    )
    checkpoint_finished = checkpoint_hit & (
        advanced_checkpoint_index >= point_count[:, None]
    )
    checkpoint_success = jnp.any(checkpoint_finished, axis=1)

    target_team = runtime.team_mask & jnp.all(
        runtime.team_key == runtime.objective_target_team_key[:, None, :],
        axis=-1,
    )
    target_observed = jnp.any(observed_team & target_team, axis=1)
    next_target_seen = state.target_seen | (
        target_observed & live_environment
    )
    target_alive = jnp.any(team_alive & target_team, axis=1)
    challenger_team = runtime.team_mask & ~target_team
    challenger_seen = next_team_seen & challenger_team
    challenger_alive = team_alive & challenger_team
    any_challenger_seen = jnp.any(challenger_seen, axis=1)
    any_challenger_alive = jnp.any(challenger_alive, axis=1)
    eliminate_success = (
        is_eliminate
        & live_environment
        & next_target_seen
        & ~target_alive
        & any_challenger_alive
    )
    eliminate_failure = (
        is_eliminate
        & live_environment
        & next_target_seen
        & any_challenger_seen
        & ~any_challenger_alive
    )
    eliminate_winner = (
        eliminate_success[:, None] & challenger_alive
    ) | (
        eliminate_failure[:, None] & target_alive[:, None] & target_team
    )

    survive_due = next_step_count >= runtime.objective_duration_steps
    survive_success = (
        is_survive & live_environment & survive_due & any_team_alive
    )
    survive_failure = (
        is_survive
        & live_environment
        & ~any_team_alive
        & (any_team_seen | survive_due)
    )
    survive_winner = survive_success[:, None] & team_alive

    navigation_failure = (
        (is_reach | is_checkpoint)
        & live_environment
        & any_team_seen
        & ~any_team_alive
    )
    success_event = (
        reach_success
        | checkpoint_success
        | eliminate_success
        | survive_success
    )
    failure_event = (
        navigation_failure | eliminate_failure | survive_failure
    )
    done_event = success_event | failure_event

    next_checkpoint_index = jnp.where(
        is_checkpoint[:, None],
        advanced_checkpoint_index,
        state.checkpoint_index,
    )
    winners = (
        reach_hit
        | checkpoint_finished
        | eliminate_winner
        | survive_winner
    )
    next_team_completed = state.team_completed | winners
    next_episode_done = state.episode_done | done_event
    next_episode_success = state.episode_success | success_event
    next_state = V2MinigameState(
        step_count=next_step_count,
        checkpoint_index=next_checkpoint_index,
        team_seen=next_team_seen,
        target_seen=next_target_seen,
        team_completed=next_team_completed,
        episode_done=next_episode_done,
        episode_success=next_episode_success,
    )

    reward = jnp.zeros(runtime.team_mask.shape, dtype=jnp.float32)
    reward = reward + reach_hit.astype(jnp.float32) * completion_reward
    reward = reward + checkpoint_hit.astype(jnp.float32) * jnp.where(
        checkpoint_finished,
        completion_reward,
        checkpoint_reward,
    )
    reward = reward + eliminate_winner.astype(jnp.float32) * completion_reward
    reward = reward + (
        eliminate_failure[:, None]
        & challenger_seen
        & ~target_team
    ).astype(jnp.float32) * failure_penalty
    reward = reward + survive_winner.astype(jnp.float32) * completion_reward
    reward = reward + (
        survive_failure[:, None] & next_team_seen & runtime.team_mask
    ).astype(jnp.float32) * failure_penalty
    reward = reward + (
        navigation_failure[:, None] & next_team_seen & runtime.team_mask
    ).astype(jnp.float32) * failure_penalty
    reward = jnp.where(live_environment[:, None], reward, 0.0)

    return V2MinigameStepResult(
        state=next_state,
        reward=reward,
        checkpoint_advanced=checkpoint_hit,
        team_completed=next_team_completed,
        winner_mask=next_team_completed,
        objective_completed=next_episode_success,
        failed=next_episode_done & ~next_episode_success,
        terminated=next_episode_done,
        truncated=jnp.zeros_like(next_episode_done),
    )


def _team_actor_masks(
    runtime: V2EnvironmentRuntime,
    actors: V2MinigameActors,
) -> tuple[jax.Array, jax.Array]:
    matches = jnp.all(
        actors.team_key[:, :, None, :] == runtime.team_key[:, None, :, :],
        axis=-1,
    )
    assigned = (
        matches
        & runtime.team_mask[:, None, :]
        & actors.active[:, :, None]
    )
    return assigned, assigned & actors.alive[:, :, None]


def _require_compatible_shapes(
    runtime: V2EnvironmentRuntime,
    state: V2MinigameState,
    actors: V2MinigameActors,
) -> None:
    batch_size, team_count = runtime.team_mask.shape
    batch_shape = (batch_size,)
    team_shape = (batch_size, team_count)
    if runtime.objective_kind.shape != batch_shape:
        raise ValueError("runtime objective kind shape is inconsistent")
    if runtime.bounds_min.shape != (batch_size, 3) or (
        runtime.bounds_max.shape != (batch_size, 3)
    ):
        raise ValueError("runtime bounds shape is inconsistent")
    if runtime.team_key.shape != (
        batch_size,
        team_count,
        BLOCK_SEMANTIC_KEY_WORDS,
    ):
        raise ValueError("runtime team key shape is inconsistent")
    if runtime.team_spawn_position.shape != (batch_size, team_count, 3):
        raise ValueError("runtime team spawn shape is inconsistent")
    if runtime.objective_point_mask.ndim != 2 or (
        runtime.objective_point_mask.shape[0] != batch_size
    ):
        raise ValueError("runtime objective point mask shape is inconsistent")
    point_count = runtime.objective_point_mask.shape[1]
    if point_count < 1 or runtime.objective_position.shape != (
        batch_size,
        point_count,
        3,
    ):
        raise ValueError("runtime objective position shape is inconsistent")
    if runtime.objective_target_team_key.shape != (
        batch_size,
        BLOCK_SEMANTIC_KEY_WORDS,
    ):
        raise ValueError("runtime objective target shape is inconsistent")
    if runtime.objective_duration_steps.shape != batch_shape:
        raise ValueError("runtime objective duration shape is inconsistent")
    if actors.position.ndim != 3 or actors.position.shape[0] != batch_size:
        raise ValueError("actor batch does not match environment runtime")
    actor_count = actors.position.shape[1]
    if actors.position.shape != (batch_size, actor_count, 3):
        raise ValueError("actor positions must have XYZ coordinates")
    if actors.team_key.shape != (
        batch_size,
        actor_count,
        BLOCK_SEMANTIC_KEY_WORDS,
    ):
        raise ValueError("actor team key shape is inconsistent")
    if actors.active.shape != (batch_size, actor_count):
        raise ValueError("actor active mask shape is inconsistent")
    if actors.alive.shape != (batch_size, actor_count):
        raise ValueError("actor alive mask shape is inconsistent")
    for field, expected in (
        (state.step_count, batch_shape),
        (state.checkpoint_index, team_shape),
        (state.team_seen, team_shape),
        (state.target_seen, batch_shape),
        (state.team_completed, team_shape),
        (state.episode_done, batch_shape),
        (state.episode_success, batch_shape),
    ):
        if field.shape != expected:
            raise ValueError("minigame state does not match environment runtime")


__all__ = [
    "V2MinigameActors",
    "V2MinigameState",
    "V2MinigameStepResult",
    "create_v2_minigame_actors",
    "reset_v2_minigame",
    "step_v2_minigame",
]
