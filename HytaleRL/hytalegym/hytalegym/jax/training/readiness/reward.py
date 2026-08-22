"""Deterministic reward-component liveness before Arsenal training."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY,
    neutral_arsenal_policy_action_factors,
)
from hytalegym.jax.combat.observation.v3.policy.surface import (
    action_surface_layout,
)
from hytalegym.jax.combat.observation.v1.schema.contract import (
    SELF_FLOAT_FEATURES,
    TARGET_FLOAT_FEATURES,
)
from hytalegym.jax.combat.types import AGENT_ENTITY, TARGET_ENTITY
from hytalegym.jax.training.ppo import PPOEnvironment


class ArsenalRewardLiveness(NamedTuple):
    """Event counts from mirrored kill-ready warmup arms.

    The two ``*_ability_accepted`` values count legal request-admission events
    only. They are liveness predicates (``> 0``), not executed-program counts;
    distinct execution windows require rising edges of
    ``active_ability_slot >= 0``.
    """

    target_damage: jax.Array
    agent_damage: jax.Array
    completion: jax.Array
    death: jax.Array
    actor_ability_accepted: jax.Array
    opponent_ability_accepted: jax.Array
    opponent_attack_observable: jax.Array


_POLICY_TARGET_ATTACK_PROGRESS = len(SELF_FLOAT_FEATURES) + TARGET_FLOAT_FEATURES.index(
    "attack_progress"
)


def _batch_time_keys(
    keys: jax.Array,
    *,
    stream: int,
    maximum_steps: int,
) -> jax.Array:
    """Derive ``[time, batch]`` keys without coupling lanes to batch width."""

    lane_roots = jax.vmap(lambda key: jax.random.fold_in(key, stream))(keys)
    ticks = jnp.arange(maximum_steps, dtype=jnp.uint32)
    return jax.vmap(
        lambda tick: jax.vmap(
            lambda lane_root: jax.random.fold_in(lane_root, tick)
        )(lane_roots)
    )(ticks)


def _kill_ready_state(state, entity: int):
    combat = state.runtime.combat
    health = combat.health.at[:, entity].set(jnp.float32(1.0))
    return state._replace(
        runtime=state.runtime._replace(
            combat=combat._replace(health=health),
        )
    )


def _warmup_arm(
    environment: PPOEnvironment,
    keys: jax.Array,
    *,
    actor_attacks: bool,
    maximum_steps: int,
) -> ArsenalRewardLiveness:
    state, observation, action_mask = environment.reset(keys)
    state = _kill_ready_state(
        state,
        TARGET_ENTITY if actor_attacks else AGENT_ENTITY,
    )
    batch = keys.shape[0]
    layout = action_surface_layout(
        ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY
    )
    ability_head = layout.head_index("ability_none_plus_slots")

    def decision(carry, step_keys):
        current_state, current_observation, current_mask = carry
        factors = neutral_arsenal_policy_action_factors(batch)
        if actor_attacks:
            ability_mask = layout.split_mask(current_mask)[
                "ability_none_plus_slots"
            ]
            available = jnp.any(ability_mask[:, 1:], axis=1)
            selected = (
                jnp.argmax(ability_mask[:, 1:], axis=1).astype(jnp.int32)
                + jnp.int32(1)
            )
            factors = factors.at[:, ability_head].set(
                jnp.where(available, selected, jnp.int32(0))
            )
        transition = environment.step_detailed(
            current_state,
            current_observation,
            factors,
            step_keys,
        )
        next_state, next_observation = transition[:2]
        next_mask = transition[4]
        info = transition[-1]
        components = info.combat_info.reward_components
        accepted = info.arsenal_info.ability_accepted
        events = ArsenalRewardLiveness(
            target_damage=jnp.count_nonzero(components.target_damage > 0.0),
            agent_damage=jnp.count_nonzero(components.agent_damage > 0.0),
            completion=jnp.count_nonzero(components.completion),
            death=jnp.count_nonzero(components.death),
            actor_ability_accepted=jnp.count_nonzero(
                accepted[:, AGENT_ENTITY]
            ),
            opponent_ability_accepted=jnp.count_nonzero(
                accepted[:, AGENT_ENTITY + 1 :]
            ),
            opponent_attack_observable=jnp.count_nonzero(
                next_observation[:, _POLICY_TARGET_ATTACK_PROGRESS] > 0.0
            ),
        )
        return (next_state, next_observation, next_mask), events

    step_keys = _batch_time_keys(
        keys,
        stream=int(actor_attacks),
        maximum_steps=maximum_steps,
    )
    _, events = jax.lax.scan(
        decision,
        (state, observation, action_mask),
        step_keys,
    )
    return jax.tree_util.tree_map(jnp.sum, events)


def probe_arsenal_reward_liveness(
    environment: PPOEnvironment,
    keys: jax.Array,
    *,
    maximum_steps: int = 128,
) -> ArsenalRewardLiveness:
    """Exercise both terminal directions without changing the task rules.

    The two arms start from ordinary environment resets. Only the victim's
    health is reduced to one so the probe measures whether the configured
    action, opponent, damage, and reward paths are reachable rather than
    waiting for full-length episodes.
    """

    if keys.ndim != 1 or keys.shape[0] < 1:
        raise ValueError("keys must be a non-empty batch of typed JAX keys")
    if maximum_steps < 1:
        raise ValueError("maximum_steps must be positive")
    offensive = _warmup_arm(
        environment,
        keys,
        actor_attacks=True,
        maximum_steps=maximum_steps,
    )
    defensive_keys = jax.vmap(
        lambda key: jax.random.fold_in(key, 0xD3F3)
    )(keys)
    defensive = _warmup_arm(
        environment,
        defensive_keys,
        actor_attacks=False,
        maximum_steps=maximum_steps,
    )
    return jax.tree_util.tree_map(jnp.add, offensive, defensive)


def require_arsenal_reward_liveness(
    environment: PPOEnvironment,
    keys: jax.Array,
    *,
    maximum_steps: int = 128,
    require_opponent_ability: bool = False,
) -> dict[str, int]:
    """Return host counts or fail before an optimizer update can start."""

    report = jax.jit(
        lambda probe_keys: probe_arsenal_reward_liveness(
            environment,
            probe_keys,
            maximum_steps=maximum_steps,
        )
    )(keys)
    counts = {
        name: int(jax.device_get(value))
        for name, value in report._asdict().items()
    }
    required = [
        "target_damage",
        "agent_damage",
        "completion",
        "death",
    ]
    if require_opponent_ability:
        required.extend(
            ("opponent_ability_accepted", "opponent_attack_observable")
        )
    missing = [name for name in required if counts[name] == 0]
    if missing:
        raise RuntimeError(
            "Arsenal reward-liveness warmup did not exercise: "
            + ", ".join(missing)
            + f"; counts={counts}"
        )
    return counts
