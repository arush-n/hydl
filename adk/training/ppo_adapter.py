"""Optional PPO compatibility adapter for the algorithm-neutral SDK runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hytalegym.jax.training.ppo import PPOEnvironment, combat_episode_outcome

if TYPE_CHECKING:
    from adk.runtime.env_adapter import BuiltAgent


def as_ppo_environment(built: "BuiltAgent") -> PPOEnvironment:
    """Adapt one existing structured build to Gym's dense PPO interface.

    No second environment is built.  Reset, transition dynamics, and dense
    projection delegate to the same objects used by the general SDK path.
    """

    def reset(keys):
        wrapped_state, policy_input = built.reset(keys)
        return (
            wrapped_state.environment,
            policy_input.observation,
            policy_input.action_mask,
        )

    def step_detailed(state, observation, factors, keys):
        wrapped = _wrap_state(state, observation)
        (
            next_wrapped,
            next_input,
            reward,
            done,
            info,
        ) = built.step_factors(wrapped, factors, keys)
        return (
            next_wrapped.environment,
            next_input.observation,
            reward,
            done,
            next_input.action_mask,
            info,
        )

    def step(state, observation, factors, keys):
        return step_detailed(state, observation, factors, keys)[:5]

    def episode_outcome(state, done):
        return combat_episode_outcome(state.runtime.combat, done)

    return PPOEnvironment(
        reset=reset,
        step=step,
        spec=built.policy_surface.spec,
        episode_outcome=episode_outcome,
        step_detailed=step_detailed,
    )


def _wrap_state(state, _dense_observation):
    # Local import prevents the optional learner adapter from participating in
    # core runtime import order.
    from adk.runtime.env_adapter import JaxEnvironmentState

    # PPO carries its dense observation separately and never constructs an
    # actor-facing structured input from this temporary wrapper.
    return JaxEnvironmentState(state, None)


__all__ = ["as_ppo_environment"]
