"""Executable collection for :class:`HierarchicalAgentCore`.

This module is the runtime seam between the algorithm-neutral hierarchy and a
canonical structured ``BuiltAgent``-like environment.  It owns the repeated
reset/decision/step/context/autoreset sequence while leaving beliefs, goals,
actor models, and training records entirely agent-defined.

The candidate hierarchy result and candidate decision context are recorded
before autoreset.  The carried environment, context, and hierarchy state are
then reset per lane from the exact :class:`EpisodeBoundary.done` signal.  A
known terminated/truncated split remains available on the transition; a
collapsed producer remains explicitly unknown through ``EpisodeBoundary``.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from adk.architecture.decision_context import (
    DecisionContext,
    jax_decision_context,
)
from adk.architecture.hierarchy import (
    HierarchicalAgentCore,
    HierarchicalCoreState,
    HierarchicalStep,
)
from adk.architecture.inputs import ActorPolicyInput, actor_policy_input
from adk.runtime.env_adapter import step_with_boundary
from adk.runtime.actions import EnvironmentActionReceipt
from adk.runtime.episode import EpisodeBoundary
from adk.runtime.loop import ContextualPolicyInput, unknown_decision_context
from adk.runtime.tree import select_environment_rows


class HierarchicalLoopState(NamedTuple):
    """Final carried environment, decision context, and hierarchy state."""

    environment_state: Any
    policy_input: Any
    decision_context: DecisionContext
    hierarchy_state: HierarchicalCoreState


class HierarchicalRuntimeDiagnostics(NamedTuple):
    """Exact component diagnostics plus the unfiltered environment value."""

    belief: Any
    manager_proposal: Any
    manager_active: Any
    actor: Any
    environment: Any


class HierarchicalLoopTransition(NamedTuple):
    """One lossless hierarchy/environment transition supplied to a recorder.

    ``hierarchy_step.state`` is the candidate recurrent state produced for the
    transition. ``carried_hierarchy_state`` is the state selected for the next
    policy call after per-lane autoreset. Both remain available so terminal
    diagnostics are not replaced by reset sentinels.
    """

    state: Any
    policy_input: ContextualPolicyInput
    hierarchy_state: HierarchicalCoreState
    hierarchy_step: HierarchicalStep
    action_factors: jax.Array
    reward: jax.Array
    boundary: EpisodeBoundary
    info: Any
    next_state: Any
    next_policy_input: ContextualPolicyInput
    carried_hierarchy_state: HierarchicalCoreState

    @property
    def done(self):
        return self.boundary.done

    @property
    def terminated(self):
        return self.boundary.terminated

    @property
    def truncated(self):
        return self.boundary.truncated

    @property
    def actor_input(self) -> ActorPolicyInput:
        return self.policy_input.actor_input

    @property
    def decision_context(self) -> DecisionContext:
        return self.policy_input.decision_context

    @property
    def next_actor_input(self) -> ActorPolicyInput:
        return self.next_policy_input.actor_input

    @property
    def next_decision_context(self) -> DecisionContext:
        return self.next_policy_input.decision_context

    @property
    def candidate_hierarchy_state(self) -> HierarchicalCoreState:
        return self.hierarchy_step.state

    @property
    def diagnostics(self) -> HierarchicalRuntimeDiagnostics:
        """Retain every component diagnostic without projecting ``info``."""

        return HierarchicalRuntimeDiagnostics(
            belief=self.hierarchy_step.belief_update.diagnostics,
            manager_proposal=(
                self.hierarchy_step.manager_proposal.diagnostics
            ),
            manager_active=self.hierarchy_step.state.manager_diagnostics,
            actor=self.hierarchy_step.actor_decision.diagnostics,
            environment=self.info,
        )

    @property
    def raw_info(self) -> Any:
        return self.info

    @property
    def action_receipt(self) -> EnvironmentActionReceipt:
        """Expose the environment boundary without inferring joint validity."""

        return EnvironmentActionReceipt.unavailable(
            self.action_factors,
            reason=(
                "hierarchical collection received no environment-owned "
                "whole-composite receipt"
            ),
            raw_receipt=self.info,
        )


HierarchicalContextFn = Callable[
    [Any, ActorPolicyInput, ActorPolicyInput],
    DecisionContext,
]
HierarchicalActorInputFn = Callable[[Any, Any], ActorPolicyInput]
HierarchicalRecordFn = Callable[[HierarchicalLoopTransition], Any]


def jax_hierarchical_context(
    info: Any,
    actor_input: ActorPolicyInput,
    previous_actor_input: ActorPolicyInput,
) -> DecisionContext:
    """Map canonical JAX diagnostics and consecutive legal actor inputs."""

    return jax_decision_context(
        info,
        actor_input=actor_input,
        previous_actor_input=previous_actor_input,
    ).context


def collect_hierarchical(
    built: Any,
    core: HierarchicalAgentCore,
    record_fn: HierarchicalRecordFn,
    key: jax.Array,
    steps: int,
    *,
    initial_state: HierarchicalCoreState,
    context_fn: HierarchicalContextFn = jax_hierarchical_context,
    actor_input_fn: HierarchicalActorInputFn = actor_policy_input,
) -> tuple[HierarchicalLoopState, Any]:
    """Collect fixed-length hierarchical transitions with ``jax.lax.scan``.

    The environment must expose the canonical ``batch``, ``reset(keys)``, and
    detailed-or-legacy factor-step interface understood by
    :func:`step_with_boundary`.  ``context_fn`` is injectable so another
    backend can publish the same normalized :class:`DecisionContext` without
    the hierarchy depending on its raw diagnostic type.
    """

    batch = _validate_inputs(
        built,
        core,
        record_fn,
        steps,
        initial_state,
        context_fn,
        actor_input_fn,
    )
    previous_action = jnp.asarray(initial_state.carry.previous_action)
    expected_factor_shape = (batch, previous_action.shape[1])
    reset_key, scan_key = jax.random.split(key)
    environment_state, policy_input = built.reset(
        jax.random.split(reset_key, batch)
    )
    initial_context = unknown_decision_context(batch)
    hierarchy_structure = jax.tree.structure(initial_state)

    def body(loop_state: HierarchicalLoopState, step_key: jax.Array):
        (
            environment_state,
            policy_input,
            decision_context,
            hierarchy_state,
        ) = loop_state
        hierarchy_key, environment_key, restart_key = jax.random.split(
            step_key,
            3,
        )
        current_actor_input = actor_input_fn(
            environment_state,
            policy_input,
        )
        if not isinstance(current_actor_input, ActorPolicyInput):
            raise TypeError("actor_input_fn must return ActorPolicyInput")
        current_policy_input = ContextualPolicyInput(
            actor_input=current_actor_input,
            decision_context=decision_context,
        )
        hierarchy_step = core(
            hierarchy_state,
            current_actor_input,
            decision_context,
            hierarchy_key,
        )
        if not isinstance(hierarchy_step, HierarchicalStep):
            raise TypeError("core must return HierarchicalStep")
        if jax.tree.structure(hierarchy_step.state) != hierarchy_structure:
            raise TypeError(
                "hierarchy state PyTree must match the initial state"
            )
        factors = _factor_rows(
            hierarchy_step.actor_decision.action_factors,
            expected_factor_shape,
        )
        environment_step = step_with_boundary(
            built,
            environment_state,
            factors,
            jax.random.split(environment_key, batch),
        )
        candidate_actor_input = actor_input_fn(
            environment_step.state,
            environment_step.policy_input,
        )
        if not isinstance(candidate_actor_input, ActorPolicyInput):
            raise TypeError("actor_input_fn must return ActorPolicyInput")
        candidate_context = context_fn(
            environment_step.info,
            candidate_actor_input,
            current_actor_input,
        )
        if not isinstance(candidate_context, DecisionContext):
            raise TypeError("context_fn must return DecisionContext")
        candidate_policy_input = ContextualPolicyInput(
            actor_input=candidate_actor_input,
            decision_context=candidate_context,
        )

        done = environment_step.boundary.done
        carried_hierarchy_state = select_environment_rows(
            done,
            initial_state,
            hierarchy_step.state,
        )
        record = record_fn(
            HierarchicalLoopTransition(
                state=environment_state,
                policy_input=current_policy_input,
                hierarchy_state=hierarchy_state,
                hierarchy_step=hierarchy_step,
                action_factors=factors,
                reward=environment_step.reward,
                boundary=environment_step.boundary,
                info=environment_step.info,
                next_state=environment_step.state,
                next_policy_input=candidate_policy_input,
                carried_hierarchy_state=carried_hierarchy_state,
            )
        )

        fresh_state, fresh_input = built.reset(
            jax.random.split(restart_key, batch)
        )
        next_environment_state = select_environment_rows(
            done,
            fresh_state,
            environment_step.state,
        )
        next_policy_input = select_environment_rows(
            done,
            fresh_input,
            environment_step.policy_input,
        )
        carried_context = select_environment_rows(
            done,
            initial_context,
            candidate_context,
        )
        return (
            HierarchicalLoopState(
                environment_state=next_environment_state,
                policy_input=next_policy_input,
                decision_context=carried_context,
                hierarchy_state=carried_hierarchy_state,
            ),
            record,
        )

    return jax.lax.scan(
        body,
        HierarchicalLoopState(
            environment_state=environment_state,
            policy_input=policy_input,
            decision_context=initial_context,
            hierarchy_state=initial_state,
        ),
        jax.random.split(scan_key, steps),
    )


def compile_hierarchical_collector(
    built: Any,
    core: HierarchicalAgentCore,
    record_fn: HierarchicalRecordFn,
    steps: int,
    *,
    initial_state: HierarchicalCoreState,
    context_fn: HierarchicalContextFn = jax_hierarchical_context,
    actor_input_fn: HierarchicalActorInputFn = actor_policy_input,
) -> Callable[[jax.Array], tuple[HierarchicalLoopState, Any]]:
    """Compile one reusable fixed-shape hierarchy collector."""

    def run(key: jax.Array):
        return collect_hierarchical(
            built,
            core,
            record_fn,
            key,
            steps,
            initial_state=initial_state,
            context_fn=context_fn,
            actor_input_fn=actor_input_fn,
        )

    return jax.jit(run)


def _validate_inputs(
    built: Any,
    core: HierarchicalAgentCore,
    record_fn: HierarchicalRecordFn,
    steps: int,
    initial_state: HierarchicalCoreState,
    context_fn: HierarchicalContextFn,
    actor_input_fn: HierarchicalActorInputFn,
) -> int:
    batch = getattr(built, "batch", None)
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("built.batch must be a positive integer")
    if not callable(getattr(built, "reset", None)):
        raise TypeError("built must expose reset(keys)")
    if not callable(getattr(built, "step_detailed", None)) and not callable(
        getattr(built, "step_factors", None)
    ):
        raise TypeError(
            "built must expose step_detailed or step_factors"
        )
    if not isinstance(core, HierarchicalAgentCore):
        raise TypeError("core must be a HierarchicalAgentCore")
    if not callable(record_fn):
        raise TypeError("record_fn must be callable")
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError("steps must be an integer")
    if steps < 1:
        raise ValueError("steps must be positive")
    if not isinstance(initial_state, HierarchicalCoreState):
        raise TypeError("initial_state must be a HierarchicalCoreState")
    if not callable(context_fn):
        raise TypeError("context_fn must be callable")
    if not callable(actor_input_fn):
        raise TypeError("actor_input_fn must be callable")
    previous_action = jnp.asarray(initial_state.carry.previous_action)
    if previous_action.dtype != jnp.dtype(jnp.int32):
        raise TypeError("initial previous_action must have dtype int32")
    if previous_action.ndim != 2 or previous_action.shape[0] != batch:
        raise ValueError(
            "initial previous_action must have shape [built.batch, heads]"
        )
    if previous_action.shape[1] < 1:
        raise ValueError("initial previous_action must contain an action head")
    return batch


def _factor_rows(value: Any, expected_shape: tuple[int, int]) -> jax.Array:
    factors = jnp.asarray(value)
    if factors.dtype != jnp.dtype(jnp.int32):
        raise TypeError("hierarchical actor action_factors must have dtype int32")
    if factors.shape != expected_shape:
        raise ValueError(
            "hierarchical actor action_factors must have shape "
            f"{expected_shape}, got {factors.shape}"
        )
    return factors


__all__ = [
    "HierarchicalActorInputFn",
    "HierarchicalContextFn",
    "HierarchicalLoopState",
    "HierarchicalLoopTransition",
    "HierarchicalRecordFn",
    "HierarchicalRuntimeDiagnostics",
    "collect_hierarchical",
    "compile_hierarchical_collector",
    "jax_hierarchical_context",
]
