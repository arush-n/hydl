"""Actor-safe and privileged inputs for non-trivial agent architectures.

The dense PPO adapter is useful, but it is not the complete learner-facing
surface.  Gym also retains a structured ``LearnerCombatObservationV3`` in its
environment state.  This module exposes that exact object to policies without
copying or reconstructing it, and keeps privileged simulator state on an
explicitly different training-only path.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax

from adk.runtime.env_adapter import JaxEnvironmentState, PolicyInput
from hytalegym.jax.combat.arsenal.environment import (
    materialize_arsenal_action_surface_evidence,
)


class ActorPolicyInput(NamedTuple):
    """Complete deployment-legal policy input shared by JAX and native.

    ``observation`` remains the exact dense Gym/bridge row for compatibility
    with existing policies. ``legal_observation`` is the upstream structured
    learner observation, not an ADK projection. ``action_surface`` retains
    any additional actor-safe JAX candidate evidence; it is ``None`` when a
    backend does not publish a structured equivalent. In every case the dense
    row remains available, so the optional structured view cannot hide data.

    The type deliberately contains no authoritative simulator state,
    privileged scene, backend identifier, or transition diagnostics.
    """

    observation: jax.Array
    action_mask: jax.Array
    legal_observation: Any
    action_surface: Any

    @property
    def dense_observation(self) -> jax.Array:
        """Explicit alias for code that also consumes the structured view."""

        return self.observation


class JaxTrainingInput(NamedTuple):
    """Training-only view of one JAX state for teachers and critics.

    JAX is authoritative only for its simulator transition. ``simulator_state``
    is therefore named precisely rather than pretending it is native Hytale
    truth. ``raw_environment_state`` is retained verbatim so newly added Gym
    fields remain reachable before the ADK gives them a stable name.
    """

    actor: ActorPolicyInput
    simulator_state: Any
    action_surface_runtime: Any
    raw_environment_state: Any


def actor_policy_input(
    state: JaxEnvironmentState,
    policy_input: PolicyInput,
) -> ActorPolicyInput:
    """Project a JAX frame onto the exact actor-safe structured channel.

    The legal observation was already compiled by Gym; the ADK never derives
    it by masking privileged state. An internal zero-storage action surface is
    expanded here so generic collectors can safely stack the actor input.
    """

    if not isinstance(state, JaxEnvironmentState):
        raise TypeError("state must be the JaxEnvironmentState returned by reset")
    if not isinstance(policy_input, PolicyInput):
        raise TypeError("policy_input must be returned by the JAX environment")
    raw_state = state.environment
    legal_observation = state.learner_observation
    if legal_observation is None:
        raise RuntimeError(
            "JAX environment state does not publish learner_observation; "
            "a structured actor policy cannot run on this adapter"
        )
    action_surface = getattr(raw_state, "action_surface", None)
    return ActorPolicyInput(
        observation=policy_input.observation,
        action_mask=policy_input.action_mask,
        legal_observation=legal_observation,
        action_surface=(
            None
            if action_surface is None
            else materialize_arsenal_action_surface_evidence(action_surface)
        ),
    )


def jax_training_input(
    state: JaxEnvironmentState,
    policy_input: PolicyInput,
) -> JaxTrainingInput:
    """Return the explicit teacher/critic channel for one JAX frame."""

    actor = actor_policy_input(state, policy_input)
    raw_state = state.environment
    simulator_state = getattr(raw_state, "runtime", None)
    if simulator_state is None:
        raise RuntimeError(
            "JAX environment state does not publish its simulator runtime"
        )
    return JaxTrainingInput(
        actor=actor,
        simulator_state=simulator_state,
        action_surface_runtime=getattr(
            raw_state,
            "action_surface_runtime",
            None,
        ),
        raw_environment_state=raw_state,
    )


__all__ = [
    "ActorPolicyInput",
    "JaxTrainingInput",
    "actor_policy_input",
    "jax_training_input",
]
