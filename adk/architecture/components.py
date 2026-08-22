"""Replaceable component protocols for PLAN-style agent architectures.

These are interface contracts, not model implementations.  They standardize
where legal observations, privileged training state, goals, recurrent state,
world-model predictions, and value products cross module boundaries while
leaving every PyTree and network topology under the agent author's control.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol, runtime_checkable

import jax

from hytalegym.framework import ProviderContract

from adk.architecture.inputs import ActorPolicyInput, JaxTrainingInput
from adk.architecture.decision_context import DecisionContext


class BeliefUpdate(NamedTuple):
    """One legal-observation belief update and its optional auxiliaries."""

    state: Any
    belief: Any
    prior: Any
    posterior: Any
    uncertainty: Any
    diagnostics: Any


class GoalDecision(NamedTuple):
    """One tactical-manager result; ``goal`` is an agent-defined PyTree."""

    state: Any
    goal: Any
    terminate: jax.Array
    diagnostics: Any


class ActorDecision(NamedTuple):
    """Low-level actor result on the exact factored action surface."""

    state: Any
    action_factors: jax.Array
    logits: Any
    behavior_log_probability: Any
    value: Any
    terminate_goal: Any
    diagnostics: Any


class WorldModelPrediction(NamedTuple):
    """Short-horizon prediction products without prescribing a latent model."""

    state: Any
    belief: Any
    events: Any
    reward: Any
    continuation: Any
    action_completion: Any
    uncertainty: Any
    diagnostics: Any


class ValuePrediction(NamedTuple):
    """Distinct value/outcome products required by the parent design."""

    return_value: Any
    win_logits: Any
    option_values: Any
    ability_affordances: Any
    diagnostics: Any


class HierarchicalCarry(NamedTuple):
    """Convenience PyTree for a belief/manager/actor policy carry.

    Every field is deliberately opaque. Agent authors may use this type or
    supply any other shape-stable PyTree to the normal ADK policy seam.
    """

    belief_state: Any
    manager_state: Any
    actor_state: Any
    world_model_state: Any
    goal: Any
    previous_action: Any
    goal_age: Any
    action_age: Any


@runtime_checkable
class ContractedComponent(Protocol):
    """A replaceable module with an exact Gym framework contract."""

    @property
    def contract(self) -> ProviderContract: ...


@runtime_checkable
class BeliefCore(ContractedComponent, Protocol):
    """Update persistent belief from legal evidence only."""

    def __call__(
        self,
        state: Any,
        actor_input: ActorPolicyInput,
        previous_action: Any,
        previous_goal: Any,
        delta_time: jax.Array,
        key: jax.Array,
        /,
    ) -> BeliefUpdate: ...


@runtime_checkable
class TacticalManager(ContractedComponent, Protocol):
    """Choose or terminate a goal using legal state and decision events.

    ``goal_runtime`` is the orchestration-owned commitment/age/interrupt view
    published by ``HierarchicalAgentCore``. It is typed as ``Any`` here to
    avoid coupling replaceable component protocols to one concrete runner.
    """

    def __call__(
        self,
        state: Any,
        belief: Any,
        current_goal: Any,
        actor_input: ActorPolicyInput,
        key: jax.Array,
        /,
        *,
        decision_context: DecisionContext | None = None,
        goal_runtime: Any = None,
    ) -> GoalDecision: ...


@runtime_checkable
class LowLevelActor(ContractedComponent, Protocol):
    """Produce explicit masked action factors from belief and goal."""

    def __call__(
        self,
        state: Any,
        belief: Any,
        goal: Any,
        actor_input: ActorPolicyInput,
        key: jax.Array,
        /,
    ) -> ActorDecision: ...


@runtime_checkable
class ShortHorizonWorldModel(ContractedComponent, Protocol):
    """Predict consequences without entering the deployed actor channel."""

    def __call__(
        self,
        state: Any,
        belief: Any,
        goal: Any,
        action_sequence: Any,
        exact_rule_state: Any,
        key: jax.Array,
        /,
    ) -> WorldModelPrediction: ...


@runtime_checkable
class PrivilegedTeacher(ContractedComponent, Protocol):
    """Training-only action provider with explicit simulator truth access."""

    def __call__(
        self,
        state: Any,
        training_input: JaxTrainingInput,
        goal: Any,
        key: jax.Array,
        /,
    ) -> ActorDecision: ...


@runtime_checkable
class PrivilegedCritic(ContractedComponent, Protocol):
    """Training-only critic/outcome provider on the privileged channel."""

    def __call__(
        self,
        state: Any,
        belief: Any,
        training_input: JaxTrainingInput,
        goal: Any,
        key: jax.Array,
        /,
    ) -> ValuePrediction: ...


__all__ = [
    "ActorDecision",
    "BeliefCore",
    "BeliefUpdate",
    "ContractedComponent",
    "GoalDecision",
    "HierarchicalCarry",
    "LowLevelActor",
    "PrivilegedCritic",
    "PrivilegedTeacher",
    "ShortHorizonWorldModel",
    "TacticalManager",
    "ValuePrediction",
    "WorldModelPrediction",
]
