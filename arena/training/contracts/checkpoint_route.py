"""Locomotion: reach a goal block on generated terrain, as fast as possible.

A goal block is chosen somewhere in a freshly generated world and the agent has
to get to it. Jumping, sprinting and route choice are the *means*; none of them
is rewarded directly.

That is the whole design. Rewarding jumps produces a bunny-hopper that never
travels; rewarding speed produces an agent that sprints off a cliff. What this
contract rewards is **closing distance to the goal**, and what it charges is
**time**. Sprinting and jumping then emerge because they are the cheapest way
to spend fewer ticks, and route quality emerges because a detour costs ticks.

Two smaller hazards are handled explicitly:

* Progress is scored symmetrically and clipped, so an agent cannot farm reward
  by oscillating toward and away from the goal, and one long fall cannot pay
  out more than a whole traverse.
* A no-progress clock ends episodes where the agent is wedged against terrain,
  so a stuck lane stops contributing gradient instead of quietly draining the
  batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from arena.training.contracts.common import (
    HorizonConfig,
    contract_hash,
    head_scope,
    lane_shapes_match,
    normalized_manifest,
    require_finite_nonnegative,
    require_idle_is_costly,
    require_positive_int,
)
from arena.training.skills.program import FactoredActionScope


CHECKPOINT_ROUTE_SCHEMA = "arena-contract-checkpoint-route-v1"


@dataclass(frozen=True, slots=True)
class CheckpointRouteConfig:
    """Tunable weights for goal-directed traversal over generated terrain."""

    horizon: HorizonConfig = HorizonConfig()

    #: How close counts as arrived, in blocks.
    arrival_radius_blocks: float = 1.5
    #: Progress below this per tick is treated as standing still.
    progress_deadband_blocks: float = 0.01
    #: Per-tick progress is clipped to this magnitude before scoring.
    progress_clip_blocks: float = 0.5

    progress_scale: float = 2.0
    arrival_reward: float = 5.0
    tick_cost: float = 0.01
    fall_damage_scale: float = 0.5

    #: Ticks without progress before the lane is declared stuck.
    no_progress_patience_ticks: int = 96
    stuck_penalty: float = 1.0

    schema: str = CHECKPOINT_ROUTE_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.horizon, HorizonConfig):
            raise TypeError("checkpoint route horizon must be a HorizonConfig")
        require_positive_int(
            self.no_progress_patience_ticks, "no_progress_patience_ticks"
        )
        require_finite_nonnegative(
            {
                "arrival_radius_blocks": self.arrival_radius_blocks,
                "progress_deadband_blocks": self.progress_deadband_blocks,
                "progress_clip_blocks": self.progress_clip_blocks,
                "progress_scale": self.progress_scale,
                "arrival_reward": self.arrival_reward,
                "tick_cost": self.tick_cost,
                "fall_damage_scale": self.fall_damage_scale,
                "stuck_penalty": self.stuck_penalty,
            },
            "checkpoint route",
        )
        if self.arrival_radius_blocks <= 0.0:
            raise ValueError("arrival_radius_blocks must be positive")
        if self.progress_clip_blocks <= self.progress_deadband_blocks:
            raise ValueError("progress clip must exceed the deadband")
        if self.tick_cost <= 0.0:
            raise ValueError("tick_cost must be positive or there is no hurry")
        require_idle_is_costly(
            passive_rewards={},
            per_tick_costs={"tick_cost": self.tick_cost},
            label="checkpoint_route",
        )
        if self.schema != CHECKPOINT_ROUTE_SCHEMA:
            raise ValueError("checkpoint route schema is not current")

    def manifest(self) -> dict[str, object]:
        return normalized_manifest(
            schema=self.schema,
            objective="reach_a_generated_goal_block_in_as_few_ticks_as_possible",
            reward={
                "progress_scale": self.progress_scale,
                "arrival_reward": self.arrival_reward,
            },
            cost={
                "tick_cost": self.tick_cost,
                "fall_damage_scale": self.fall_damage_scale,
                "stuck_penalty": self.stuck_penalty,
            },
            anti_farm=(
                "jump_and_sprint_are_unrewarded_means_time_is_charged_so_they_"
                "emerge_only_where_they_save_ticks"
            ),
            horizon=self.horizon,
            terminal="arrival_or_stuck_or_horizon",
            shaping={
                "arrival_radius_blocks": self.arrival_radius_blocks,
                "progress_deadband_blocks": self.progress_deadband_blocks,
                "progress_clip_blocks": self.progress_clip_blocks,
                "no_progress_patience_ticks": self.no_progress_patience_ticks,
            },
            credit="symmetric_clipped_distance_reduction_toward_the_goal",
        )

    @property
    def contract_sha256(self) -> str:
        return contract_hash(self.manifest())


class CheckpointRouteSignals(NamedTuple):
    """Traversal credit plus the terminal evidence for one tick."""

    reward: jax.Array
    progress: jax.Array
    arrived: jax.Array
    stuck: jax.Array
    next_stall_ticks: jax.Array


def checkpoint_route_signals(
    config: CheckpointRouteConfig,
    *,
    distance_before: jax.Array,
    distance_after: jax.Array,
    fall_damage: jax.Array,
    stall_ticks: jax.Array,
    valid: jax.Array,
) -> CheckpointRouteSignals:
    """Score one traversal tick against the goal block.

    ``stall_ticks`` is the running count of consecutive ticks without
    meaningful progress; pass the returned ``next_stall_ticks`` back in on the
    following tick.
    """

    before = jnp.asarray(distance_before, dtype=jnp.float32)
    after = jnp.asarray(distance_after, dtype=jnp.float32)
    damage = jnp.asarray(fall_damage, dtype=jnp.float32)
    stalled = jnp.asarray(stall_ticks, dtype=jnp.int32)
    evidence = jnp.asarray(valid, dtype=jnp.bool_)
    lane_shapes_match(before, after, damage, stalled, evidence)

    raw_progress = before - after
    deadband = jnp.float32(config.progress_deadband_blocks)
    moved = jnp.abs(raw_progress) > deadband
    progress = jnp.where(moved, raw_progress, jnp.float32(0.0))
    progress = jnp.clip(
        progress,
        -jnp.float32(config.progress_clip_blocks),
        jnp.float32(config.progress_clip_blocks),
    )

    arrived = evidence & (after <= jnp.float32(config.arrival_radius_blocks))
    next_stall = jnp.where(evidence & ~moved, stalled + 1, jnp.int32(0))
    stuck = evidence & (
        next_stall >= jnp.int32(config.no_progress_patience_ticks)
    )

    reward = (
        jnp.float32(config.progress_scale) * progress
        + jnp.float32(config.arrival_reward) * arrived
        - jnp.float32(config.fall_damage_scale) * damage
        - jnp.float32(config.stuck_penalty) * stuck
        - jnp.float32(config.tick_cost)
    )
    return CheckpointRouteSignals(
        jnp.where(evidence, reward, jnp.float32(0.0)),
        jnp.where(evidence, progress, jnp.float32(0.0)),
        arrived,
        stuck,
        next_stall,
    )


def goal_distance(position: jax.Array, goal: jax.Array) -> jax.Array:
    """Euclidean distance from each lane's actor position to its goal block."""

    offset = jnp.asarray(position, dtype=jnp.float32) - jnp.asarray(
        goal, dtype=jnp.float32
    )
    return jnp.linalg.norm(offset, axis=-1)


def checkpoint_route_action_scope() -> FactoredActionScope:
    """Movement, look, jump and dodge-dash; no combat heads at all."""

    return head_scope(
        "locomotion_gait_compass",
        "yaw_delta_bins",
        "body_yaw_delta_bins",
        "pitch_delta_bins",
        "jump_off_on",
        "locomotion_gait_compass",
    )


__all__ = [
    "CHECKPOINT_ROUTE_SCHEMA",
    "CheckpointRouteConfig",
    "CheckpointRouteSignals",
    "checkpoint_route_signals",
    "goal_distance",
    "checkpoint_route_action_scope",
]
