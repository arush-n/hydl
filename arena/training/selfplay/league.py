"""Rating-aware matchmaking over immutable policy snapshots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from hytalegym.jax.training.multi_actor import (
    PolicyActorAssignment,
    policy_actor_assignment,
)


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    """Tunable rating-aware matchmaking and Elo updates."""

    temperature: float = 200.0
    uniform_mix: float = 0.1
    k_factor: float = 24.0
    rating_scale: float = 400.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= self.uniform_mix <= 1.0:
            raise ValueError("uniform_mix must be in [0, 1]")
        if (
            not math.isfinite(self.k_factor)
            or not math.isfinite(self.rating_scale)
            or self.k_factor <= 0.0
            or self.rating_scale <= 0.0
        ):
            raise ValueError("k_factor and rating_scale must be positive")


class SelfPlayState(NamedTuple):
    ratings: jax.Array
    games: jax.Array
    active: jax.Array


class Matchups(NamedTuple):
    learner_id: jax.Array
    opponent_id: jax.Array
    valid: jax.Array


class SelfPlayMetrics(NamedTuple):
    matches_applied: jax.Array
    invalid_results: jax.Array
    mean_expected_score: jax.Array
    mean_absolute_rating_change: jax.Array


def initialize_self_play(
    policy_count: int,
    *,
    active_ids: Sequence[int] = (0,),
    initial_rating: float = 1200.0,
) -> SelfPlayState:
    """Create host-validated fixed-capacity policy-league state."""

    if isinstance(policy_count, bool) or not isinstance(policy_count, int):
        raise TypeError("policy_count must be an integer")
    if policy_count < 2:
        raise ValueError("self-play needs at least two policy-bank rows")
    if not np.isfinite(initial_rating):
        raise ValueError("initial_rating must be finite")
    ids = np.asarray(tuple(active_ids), dtype=np.int32)
    if ids.size and (np.any(ids < 0) or np.any(ids >= policy_count)):
        raise ValueError("active policy ID is outside the policy bank")
    active = jnp.zeros((policy_count,), dtype=jnp.bool_).at[ids].set(True)
    return SelfPlayState(
        ratings=jnp.full((policy_count,), initial_rating, dtype=jnp.float32),
        games=jnp.zeros((policy_count,), dtype=jnp.int32),
        active=active,
    )


def activate_snapshot(
    state: SelfPlayState, *, source_id: int, target_id: int
) -> SelfPlayState:
    """Activate a policy row at its source's current rating with zero games."""

    count = state.ratings.shape[0]
    if not 0 <= source_id < count or not 0 <= target_id < count:
        raise ValueError("snapshot policy ID is outside the policy bank")
    return SelfPlayState(
        ratings=state.ratings.at[target_id].set(state.ratings[source_id]),
        games=state.games.at[target_id].set(0),
        active=state.active.at[target_id].set(True),
    )


def snapshot_policy(policy_bank: Any, *, source_id: int, target_id: int) -> Any:
    """Copy one policy PyTree row without changing any other population row."""

    leaves = jax.tree.leaves(policy_bank)
    if not leaves:
        raise ValueError("policy bank cannot be empty")
    count = leaves[0].shape[0]
    if any(leaf.ndim < 1 or leaf.shape[0] != count for leaf in leaves):
        raise ValueError("every policy-bank leaf must share its leading K axis")
    if not 0 <= source_id < count or not 0 <= target_id < count:
        raise ValueError("snapshot policy ID is outside the policy bank")
    return jax.tree.map(
        lambda leaf: leaf.at[target_id].set(leaf[source_id]), policy_bank
    )


def sample_opponents(
    state: SelfPlayState,
    learner_id: int,
    key: jax.Array,
    *,
    batch_size: int,
    config: SelfPlayConfig = SelfPlayConfig(),
) -> Matchups:
    """Sample active nonlearner rows, favoring informative close ratings."""

    count = state.ratings.shape[0]
    if not 0 <= learner_id < count:
        raise ValueError("learner_id is outside the policy bank")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    candidates = state.active.at[learner_id].set(False)
    candidate_count = jnp.sum(candidates)
    has_match = state.active[learner_id] & (candidate_count > 0)
    close_logits = -jnp.abs(state.ratings - state.ratings[learner_id])
    close_logits /= config.temperature
    close = jax.nn.softmax(jnp.where(candidates, close_logits, -jnp.inf))
    uniform = candidates.astype(jnp.float32) / jnp.maximum(candidate_count, 1)
    probability = (1.0 - config.uniform_mix) * close + config.uniform_mix * uniform
    fallback = jax.nn.one_hot(learner_id, count, dtype=jnp.float32)
    probability = jnp.where(has_match, probability, fallback)
    opponents = jax.random.categorical(
        key, jnp.log(probability), shape=(batch_size,)
    ).astype(jnp.int32)
    return Matchups(
        learner_id=jnp.full((batch_size,), learner_id, dtype=jnp.int32),
        opponent_id=opponents,
        valid=jnp.full((batch_size,), has_match, dtype=jnp.bool_),
    )


def update_ratings(
    state: SelfPlayState,
    matchups: Matchups,
    learner_scores: jax.Array,
    config: SelfPlayConfig = SelfPlayConfig(),
) -> tuple[SelfPlayState, SelfPlayMetrics]:
    """Apply batched zero-sum Elo results; invalid rows are reported and skipped."""

    scores = jnp.asarray(learner_scores, dtype=jnp.float32)
    if scores.shape != matchups.valid.shape:
        raise ValueError("learner_scores must match the matchup batch")
    count = state.ratings.shape[0]
    learner = jnp.clip(matchups.learner_id, 0, count - 1)
    opponent = jnp.clip(matchups.opponent_id, 0, count - 1)
    valid = (
        matchups.valid
        & (matchups.learner_id >= 0)
        & (matchups.learner_id < count)
        & (matchups.opponent_id >= 0)
        & (matchups.opponent_id < count)
        & (learner != opponent)
        & jnp.isfinite(scores)
        & (scores >= 0.0)
        & (scores <= 1.0)
        & state.active[learner]
        & state.active[opponent]
    )
    expected = 1.0 / (
        1.0
        + jnp.power(
            10.0,
            (state.ratings[opponent] - state.ratings[learner]) / config.rating_scale,
        )
    )
    change = jnp.where(valid, config.k_factor * (scores - expected), 0.0)
    ratings = state.ratings.at[learner].add(change).at[opponent].add(-change)
    played = valid.astype(jnp.int32)
    games = state.games.at[learner].add(played).at[opponent].add(played)
    denominator = jnp.maximum(jnp.sum(played), 1)
    next_state = SelfPlayState(ratings, games, state.active)
    return next_state, SelfPlayMetrics(
        matches_applied=jnp.sum(played),
        invalid_results=jnp.sum(~valid, dtype=jnp.int32),
        mean_expected_score=jnp.sum(jnp.where(valid, expected, 0.0)) / denominator,
        mean_absolute_rating_change=jnp.sum(jnp.abs(change)) / denominator,
    )


def frozen_opponent_assignment(
    matchups: Matchups,
    *,
    actor_indices: tuple[int, int] = (0, 1),
    policy_slot_order: tuple[str, str] = ("learner", "opponent"),
    entity_count: int | None = None,
) -> PolicyActorAssignment:
    """Bridge one pairwise schedule to frozen-opponent multi-actor PPO.

    Elo is pairwise, but neither physical entity IDs nor policy-slot order are
    part of that math.  Callers therefore name both explicitly.  This permits
    a rated pair inside an arena with an arbitrary entity capacity without
    smuggling an entity-zero/entity-one convention into league scheduling.
    """

    valid = np.asarray(jax.device_get(matchups.valid), dtype=np.bool_)
    if not np.all(valid):
        raise ValueError("every arena needs a valid sampled opponent")
    if (
        matchups.learner_id.shape != matchups.valid.shape
        or matchups.opponent_id.shape != matchups.valid.shape
    ):
        raise ValueError("matchup rows must share one batch shape")
    learner = np.asarray(jax.device_get(matchups.learner_id), dtype=np.int32)
    if np.unique(learner).size != 1:
        raise ValueError("a frozen-opponent schedule must share one learner policy")
    if (
        not isinstance(actor_indices, tuple)
        or len(actor_indices) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in actor_indices)
    ):
        raise TypeError("actor_indices must be two integer entity IDs")
    if actor_indices[0] == actor_indices[1] or min(actor_indices) < 0:
        raise ValueError("rated learner and opponent need distinct entity IDs")
    if tuple(policy_slot_order) not in {
        ("learner", "opponent"),
        ("opponent", "learner"),
    }:
        raise ValueError(
            "policy_slot_order must contain learner and opponent exactly once"
        )
    minimum_entity_count = max(actor_indices) + 1
    if entity_count is None:
        entity_count = minimum_entity_count
    if isinstance(entity_count, bool) or not isinstance(entity_count, int):
        raise TypeError("entity_count must be an integer")
    if entity_count < minimum_entity_count:
        raise ValueError("a rated actor index is outside the entity axis")

    batch = matchups.valid.shape[0]
    actor_by_role = {
        "learner": jnp.full(
            (batch,), actor_indices[0], dtype=jnp.int32
        ),
        "opponent": jnp.full(
            (batch,), actor_indices[1], dtype=jnp.int32
        ),
    }
    policy_by_role = {
        "learner": matchups.learner_id,
        "opponent": matchups.opponent_id,
    }
    trainable_by_role = {
        "learner": jnp.ones((batch,), dtype=jnp.bool_),
        "opponent": jnp.zeros((batch,), dtype=jnp.bool_),
    }
    return policy_actor_assignment(
        jnp.stack(
            tuple(actor_by_role[role] for role in policy_slot_order),
            axis=-1,
        ),
        jnp.stack(
            tuple(policy_by_role[role] for role in policy_slot_order),
            axis=-1,
        ),
        trainable=jnp.stack(
            tuple(trainable_by_role[role] for role in policy_slot_order),
            axis=-1,
        ),
        entity_count=entity_count,
    )


__all__ = [
    "Matchups",
    "SelfPlayConfig",
    "SelfPlayMetrics",
    "SelfPlayState",
    "activate_snapshot",
    "frozen_opponent_assignment",
    "initialize_self_play",
    "sample_opponents",
    "snapshot_policy",
    "update_ratings",
]
