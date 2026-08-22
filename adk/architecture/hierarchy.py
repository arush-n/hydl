"""Executable, algorithm-neutral hierarchy for PLAN-style agents.

The core in this module owns orchestration, not a learning algorithm or model
topology.  It updates one legal-observation belief, schedules a tactical
manager, enforces goal commitment, and invokes a factored low-level actor.
All three components receive the original :class:`ActorPolicyInput`; the core
never reconstructs or flattens its structured evidence.

The functions are written as shape-stable JAX programs.  Mutable per-lane
leaves must have a leading batch dimension.  Non-batched leaves are treated as
shared immutable metadata and retained during masked selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from adk.architecture.components import (
    ActorDecision,
    BeliefCore,
    BeliefUpdate,
    GoalDecision,
    HierarchicalCarry,
    LowLevelActor,
    TacticalManager,
)
from adk.architecture.composition import AgentArchitecture
from adk.architecture.decision_context import (
    DecisionContext,
    DecisionEvent,
    ExecutionState,
    manager_decision_due,
)
from adk.architecture.inputs import ActorPolicyInput


class GoalControl(NamedTuple):
    """Runtime controls for an otherwise agent-defined goal payload.

    Every field is batched. ``emergency_interrupt_mask`` has shape
    ``[batch, len(DecisionEvent)]`` and uses the stable ``DecisionEvent``
    ordering. Only an event whose value *and* availability are true can match
    this mask. ``maximum_duration_seconds`` may contain ``jnp.inf``.
    """

    minimum_commitment_seconds: jax.Array
    maximum_duration_seconds: jax.Array
    emergency_interrupt_mask: jax.Array


class ManagedGoal(NamedTuple):
    """Lossless agent goal plus the controls enforced by the ADK core."""

    payload: Any
    control: GoalControl


class GoalRuntimeContext(NamedTuple):
    """Exact pre-manager goal state supplied to ``TacticalManager``.

    This value is returned by :class:`HierarchicalStep` as well. Managers get
    it through the additive ``goal_runtime`` keyword when invoked by this
    core; it makes commitment and interrupt decisions observable without
    placing ADK bookkeeping inside the learned goal payload.
    """

    valid: jax.Array
    age_seconds: jax.Array
    seconds_since_manager_decision: jax.Array
    commitment_elapsed: jax.Array
    timed_out: jax.Array
    actor_termination_requested: jax.Array
    active_events: jax.Array
    emergency_events: jax.Array
    emergency_interrupt: jax.Array
    manager_due: jax.Array


class HierarchicalCoreState(NamedTuple):
    """Shape-stable orchestration state around ``HierarchicalCarry``.

    ``manager_diagnostics`` is retained so an entirely idle manager cadence
    can skip manager computation with ``jax.lax.cond`` while preserving the
    exact diagnostic PyTree structure. It must have the same PyTree structure
    and leaf shapes as ``GoalDecision.diagnostics``.
    """

    carry: HierarchicalCarry
    goal_valid: jax.Array
    action_valid: jax.Array
    manager_age_seconds: jax.Array
    goal_termination_requested: jax.Array
    manager_diagnostics: Any


class HierarchicalStep(NamedTuple):
    """One complete belief-manager-actor result.

    ``manager_proposal`` is the exact batched manager result when at least one
    lane was due. Proposals for rows where ``manager_due`` is false are not
    applied. When no row was due it is a held result built from current state.
    ``belief_update`` and ``actor_decision`` retain every model output and
    diagnostic rather than projecting them to a training-algorithm schema.
    """

    state: HierarchicalCoreState
    belief_update: BeliefUpdate
    manager_proposal: GoalDecision
    actor_decision: ActorDecision
    active_goal: ManagedGoal
    goal_runtime: GoalRuntimeContext
    manager_due: jax.Array
    goal_switched: jax.Array
    action_lifecycle_ended: jax.Array
    delta_seconds: jax.Array
    delta_seconds_known: jax.Array


@dataclass(frozen=True, slots=True)
class HierarchicalAgentCore:
    """Compose replaceable PLAN components into one executable policy core.

    Components operate on a complete batch. Manager output is applied only to
    due rows, and a manager call is skipped globally when no row is due. In a
    mixed batch the manager still evaluates the complete static batch; this
    avoids dynamic shapes and keeps the program JIT/vmap/scan friendly.

    ``GoalDecision.terminate`` requests replacement of the current goal.
    ``ActorDecision.terminate_goal`` is the low-level request observed by the
    *next* hierarchy step. Ordinary requests cannot replace a goal before its
    minimum commitment. A matching goal-specific emergency event bypasses the
    minimum, while maximum duration always forces replacement. A manager must
    return a :class:`ManagedGoal`; ``terminate=False`` means CONTINUE_GOAL.
    """

    belief: BeliefCore
    manager: TacticalManager
    actor: LowLevelActor
    manager_cadence_seconds: float

    def __post_init__(self) -> None:
        cadence = self.manager_cadence_seconds
        if (
            isinstance(cadence, bool)
            or not isinstance(cadence, (int, float))
            or not math.isfinite(float(cadence))
            or float(cadence) <= 0.0
        ):
            raise ValueError("manager_cadence_seconds must be finite and positive")
        for name in ("belief", "manager", "actor"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} component must be callable")
        object.__setattr__(self, "manager_cadence_seconds", float(cadence))

    @classmethod
    def from_architecture(
        cls,
        architecture: AgentArchitecture,
        *,
        manager_cadence_seconds: float,
    ) -> HierarchicalAgentCore:
        """Build from the already fail-closed architecture composition."""

        if not isinstance(architecture, AgentArchitecture):
            raise TypeError("architecture must be an AgentArchitecture")
        architecture.require_roles("belief", "manager", "actor")
        return cls(
            belief=architecture.component("belief"),
            manager=architecture.component("manager"),
            actor=architecture.component("actor"),
            manager_cadence_seconds=manager_cadence_seconds,
        )

    def __call__(
        self,
        state: HierarchicalCoreState,
        actor_input: ActorPolicyInput,
        decision_context: DecisionContext,
        key: jax.Array,
        /,
        *,
        reset: jax.Array | None = None,
        initial_state: HierarchicalCoreState | None = None,
    ) -> HierarchicalStep:
        """Run one hierarchy decision.

        ``reset`` marks lanes whose current input begins a new episode (normally
        previous ``terminated | truncated``). Those lanes are restored from
        ``initial_state`` and their decision context is made wholly unknown so
        terminal lifecycle evidence cannot leak across an episode boundary.
        The current ``ActorPolicyInput`` is never altered.
        """

        if not isinstance(state, HierarchicalCoreState):
            raise TypeError("state must be a HierarchicalCoreState")
        if not isinstance(actor_input, ActorPolicyInput):
            raise TypeError("actor_input must be an ActorPolicyInput")
        if not isinstance(decision_context, DecisionContext):
            raise TypeError("decision_context must be a DecisionContext")

        action_mask = jnp.asarray(actor_input.action_mask)
        if action_mask.ndim != 2 or action_mask.shape[0] < 1:
            raise ValueError("ActorPolicyInput.action_mask must have shape [batch, N]")
        batch = action_mask.shape[0]
        reset_rows = _optional_lane_bool(reset, batch, "reset")
        if reset is not None:
            if not isinstance(initial_state, HierarchicalCoreState):
                raise TypeError(
                    "initial_state must be a HierarchicalCoreState when reset is set"
                )
            state = _select_rows(reset_rows, initial_state, state)
            decision_context = _clear_context_rows(decision_context, reset_rows)
        elif initial_state is not None:
            raise TypeError("initial_state is only valid together with reset")

        _validate_state(state, batch)
        _validate_context(decision_context, batch)
        current_goal = _require_managed_goal(state.carry.goal, batch, "current goal")
        valid = _lane_bool(state.goal_valid, batch, "goal_valid")
        action_valid = _lane_bool(state.action_valid, batch, "action_valid")
        pending_termination = _lane_bool(
            state.goal_termination_requested,
            batch,
            "goal_termination_requested",
        )

        delta_known = _lane_bool(
            decision_context.timing.delta_seconds_known,
            batch,
            "decision_context.timing.delta_seconds_known",
        )
        raw_delta = _lane_float(
            decision_context.timing.delta_seconds,
            batch,
            "decision_context.timing.delta_seconds",
        )
        # Unknown time never advances hidden clocks. Components still receive
        # the exact context and can distinguish unknown from a real zero.
        delta = jnp.where(delta_known, jnp.maximum(raw_delta, 0.0), 0.0)

        goal_age = _lane_float(state.carry.goal_age, batch, "carry.goal_age")
        action_age = _lane_float(
            state.carry.action_age,
            batch,
            "carry.action_age",
        )
        manager_age = _lane_float(
            state.manager_age_seconds,
            batch,
            "manager_age_seconds",
        )
        elapsed_goal_age = goal_age + delta
        elapsed_manager_age = manager_age + delta

        belief_key, manager_key, actor_key = jax.random.split(key, 3)
        belief_update = self.belief(
            state.carry.belief_state,
            actor_input,
            state.carry.previous_action,
            current_goal.payload,
            delta,
            belief_key,
        )
        if not isinstance(belief_update, BeliefUpdate):
            raise TypeError("belief component must return BeliefUpdate")

        control = current_goal.control
        commitment_elapsed = elapsed_goal_age >= control.minimum_commitment_seconds
        timed_out = valid & (
            elapsed_goal_age >= control.maximum_duration_seconds
        )
        active_events = (
            jnp.asarray(decision_context.events.values, dtype=jnp.bool_)
            & jnp.asarray(decision_context.events.known, dtype=jnp.bool_)
        )
        emergency_events = active_events & control.emergency_interrupt_mask
        emergency_interrupt = valid & jnp.any(emergency_events, axis=-1)
        preliminary_due = manager_decision_due(
            decision_context,
            elapsed_manager_age,
            self.manager_cadence_seconds,
            pending_termination | timed_out | ~valid,
        )
        manager_due = _lane_bool(preliminary_due, batch, "manager_due")
        goal_runtime = GoalRuntimeContext(
            valid=valid,
            age_seconds=elapsed_goal_age,
            seconds_since_manager_decision=elapsed_manager_age,
            commitment_elapsed=commitment_elapsed,
            timed_out=timed_out,
            actor_termination_requested=pending_termination,
            active_events=active_events,
            emergency_events=emergency_events,
            emergency_interrupt=emergency_interrupt,
            manager_due=manager_due,
        )

        def call_manager(_: None) -> GoalDecision:
            proposal = self.manager(
                state.carry.manager_state,
                belief_update.belief,
                current_goal.payload,
                actor_input,
                manager_key,
                decision_context=decision_context,
                goal_runtime=goal_runtime,
            )
            if not isinstance(proposal, GoalDecision):
                raise TypeError("manager component must return GoalDecision")
            normalized_goal = _require_managed_goal(
                proposal.goal,
                batch,
                "manager goal",
            )
            _lane_bool(proposal.terminate, batch, "GoalDecision.terminate")
            return proposal._replace(goal=normalized_goal)

        def hold_manager(_: None) -> GoalDecision:
            return GoalDecision(
                state=state.carry.manager_state,
                goal=current_goal,
                terminate=jnp.zeros((batch,), dtype=jnp.bool_),
                diagnostics=state.manager_diagnostics,
            )

        proposal = jax.lax.cond(
            jnp.any(manager_due),
            call_manager,
            hold_manager,
            operand=None,
        )
        candidate_goal = _require_managed_goal(
            proposal.goal,
            batch,
            "manager goal",
        )
        manager_termination = _lane_bool(
            proposal.terminate,
            batch,
            "GoalDecision.terminate",
        )
        switch_allowed = (
            ~valid | commitment_elapsed | timed_out | emergency_interrupt
        )
        switch_requested = (
            ~valid
            | timed_out
            | pending_termination
            | (manager_due & manager_termination)
        )
        goal_switched = manager_due & switch_allowed & switch_requested
        active_goal = _select_rows(goal_switched, candidate_goal, current_goal)
        active_goal = _require_managed_goal(active_goal, batch, "active goal")

        manager_state = _select_rows(
            manager_due,
            proposal.state,
            state.carry.manager_state,
        )
        manager_diagnostics = _select_rows(
            manager_due,
            proposal.diagnostics,
            state.manager_diagnostics,
        )
        next_manager_age = jnp.where(manager_due, 0.0, elapsed_manager_age)
        next_goal_age = jnp.where(goal_switched, 0.0, elapsed_goal_age)
        next_goal_valid = valid | goal_switched

        actor_decision = self.actor(
            state.carry.actor_state,
            belief_update.belief,
            active_goal.payload,
            actor_input,
            actor_key,
        )
        if not isinstance(actor_decision, ActorDecision):
            raise TypeError("actor component must return ActorDecision")
        action_factors = jnp.asarray(actor_decision.action_factors)
        previous_action = jnp.asarray(state.carry.previous_action)
        if action_factors.ndim < 2 or action_factors.shape[0] != batch:
            raise ValueError(
                "ActorDecision.action_factors must have shape [batch, ...]"
            )
        if previous_action.shape != action_factors.shape:
            raise ValueError(
                "carry.previous_action must match ActorDecision.action_factors"
            )

        action_lifecycle_ended = _action_lifecycle_ended(decision_context)
        same_action = jnp.all(
            action_factors == previous_action,
            axis=tuple(range(1, action_factors.ndim)),
        )
        next_action_age = jnp.where(
            action_valid & same_action & ~action_lifecycle_ended,
            action_age + delta,
            0.0,
        )
        next_termination = _optional_lane_bool(
            actor_decision.terminate_goal,
            batch,
            "ActorDecision.terminate_goal",
        )

        next_carry = HierarchicalCarry(
            belief_state=belief_update.state,
            manager_state=manager_state,
            actor_state=actor_decision.state,
            world_model_state=state.carry.world_model_state,
            goal=active_goal,
            previous_action=action_factors,
            goal_age=next_goal_age,
            action_age=next_action_age,
        )
        next_state = HierarchicalCoreState(
            carry=next_carry,
            goal_valid=next_goal_valid,
            action_valid=jnp.ones((batch,), dtype=jnp.bool_),
            manager_age_seconds=next_manager_age,
            goal_termination_requested=next_termination,
            manager_diagnostics=manager_diagnostics,
        )
        return HierarchicalStep(
            state=next_state,
            belief_update=belief_update,
            manager_proposal=proposal,
            actor_decision=actor_decision,
            active_goal=active_goal,
            goal_runtime=goal_runtime,
            manager_due=manager_due,
            goal_switched=goal_switched,
            action_lifecycle_ended=action_lifecycle_ended,
            delta_seconds=delta,
            delta_seconds_known=delta_known,
        )


def initialize_hierarchical_state(
    carry: HierarchicalCarry,
    *,
    manager_diagnostics: Any,
    goal_valid: Any = False,
    action_valid: Any = False,
) -> HierarchicalCoreState:
    """Add zeroed orchestration metadata to an agent-authored initial carry."""

    if not isinstance(carry, HierarchicalCarry):
        raise TypeError("carry must be a HierarchicalCarry")
    previous_action = jnp.asarray(carry.previous_action)
    if previous_action.ndim < 2 or previous_action.shape[0] < 1:
        raise ValueError("carry.previous_action must have shape [batch, ...]")
    batch = previous_action.shape[0]
    goal = _require_managed_goal(carry.goal, batch, "carry.goal")
    normalized_carry = carry._replace(
        goal=goal,
        goal_age=_lane_float(carry.goal_age, batch, "carry.goal_age"),
        action_age=_lane_float(carry.action_age, batch, "carry.action_age"),
    )
    return HierarchicalCoreState(
        carry=normalized_carry,
        goal_valid=_optional_lane_bool(goal_valid, batch, "goal_valid"),
        action_valid=_optional_lane_bool(action_valid, batch, "action_valid"),
        manager_age_seconds=jnp.zeros((batch,), dtype=jnp.float32),
        goal_termination_requested=jnp.zeros((batch,), dtype=jnp.bool_),
        manager_diagnostics=manager_diagnostics,
    )


def _validate_state(state: HierarchicalCoreState, batch: int) -> None:
    _lane_bool(state.goal_valid, batch, "goal_valid")
    _lane_bool(state.action_valid, batch, "action_valid")
    _lane_float(state.manager_age_seconds, batch, "manager_age_seconds")
    _lane_bool(
        state.goal_termination_requested,
        batch,
        "goal_termination_requested",
    )
    _lane_float(state.carry.goal_age, batch, "carry.goal_age")
    _lane_float(state.carry.action_age, batch, "carry.action_age")
    previous_action = jnp.asarray(state.carry.previous_action)
    if previous_action.ndim < 2 or previous_action.shape[0] != batch:
        raise ValueError("carry.previous_action must have shape [batch, ...]")


def _validate_context(context: DecisionContext, batch: int) -> None:
    for name, value in (
        ("timing.tick", context.timing.tick),
        ("timing.tick_known", context.timing.tick_known),
        ("timing.delta_seconds", context.timing.delta_seconds),
        ("timing.delta_seconds_known", context.timing.delta_seconds_known),
    ):
        array = jnp.asarray(value)
        if array.shape != (batch,):
            raise ValueError(f"decision_context.{name} must have shape [batch]")
    event_shape = (batch, len(DecisionEvent))
    if jnp.asarray(context.events.values).shape != event_shape:
        raise ValueError(
            "decision_context.events.values must have shape "
            f"{event_shape}"
        )
    if jnp.asarray(context.events.known).shape != event_shape:
        raise ValueError(
            "decision_context.events.known must match events.values"
        )
    lifecycle_values = jnp.asarray(context.lifecycle.values)
    lifecycle_known = jnp.asarray(context.lifecycle.known)
    if lifecycle_values.ndim != 3 or lifecycle_values.shape[0] != batch:
        raise ValueError(
            "decision_context.lifecycle.values must have shape [batch, verb, state]"
        )
    if lifecycle_known.shape != lifecycle_values.shape:
        raise ValueError(
            "decision_context.lifecycle.known must match lifecycle.values"
        )


def _require_managed_goal(value: Any, batch: int, name: str) -> ManagedGoal:
    if not isinstance(value, ManagedGoal):
        raise TypeError(f"{name} must be a ManagedGoal")
    if not isinstance(value.control, GoalControl):
        raise TypeError(f"{name}.control must be a GoalControl")
    minimum = _lane_float(
        value.control.minimum_commitment_seconds,
        batch,
        f"{name}.control.minimum_commitment_seconds",
    )
    maximum = _lane_float(
        value.control.maximum_duration_seconds,
        batch,
        f"{name}.control.maximum_duration_seconds",
    )
    mask = jnp.asarray(value.control.emergency_interrupt_mask, dtype=jnp.bool_)
    expected = (batch, len(DecisionEvent))
    if mask.shape != expected:
        raise ValueError(
            f"{name}.control.emergency_interrupt_mask must have shape {expected}"
        )
    return ManagedGoal(value.payload, GoalControl(minimum, maximum, mask))


def _action_lifecycle_ended(context: DecisionContext) -> jax.Array:
    terminal_states = jnp.asarray(
        (
            int(ExecutionState.COMPLETED),
            int(ExecutionState.CANCELLED),
            int(ExecutionState.INTERRUPTED),
            int(ExecutionState.FAILED),
            int(ExecutionState.REJECTED),
        ),
        dtype=jnp.int32,
    )
    values = jnp.take(context.lifecycle.values, terminal_states, axis=-1)
    known = jnp.take(context.lifecycle.known, terminal_states, axis=-1)
    return jnp.any(values & known, axis=(-2, -1))


def _clear_context_rows(
    context: DecisionContext,
    reset: jax.Array,
) -> DecisionContext:
    unknown = jax.tree.map(lambda value: jnp.zeros_like(value), context)
    return _select_rows(reset, unknown, context)


def _select_rows(mask: jax.Array, when_true: Any, when_false: Any) -> Any:
    """Select batch-owned leaves and keep shared immutable leaves."""

    selected = jnp.asarray(mask, dtype=jnp.bool_)
    if selected.ndim != 1:
        raise ValueError("row mask must have shape [batch]")

    def choose(true_value: Any, false_value: Any) -> Any:
        true_array = jnp.asarray(true_value)
        false_array = jnp.asarray(false_value)
        if true_array.shape != false_array.shape:
            raise ValueError("batched PyTree leaf shape mismatch")
        if true_array.ndim == 0 or true_array.shape[0] != selected.shape[0]:
            return false_array
        expanded = selected.reshape(
            (selected.shape[0],) + (1,) * (true_array.ndim - 1)
        )
        return jnp.where(expanded, true_array, false_array)

    return jax.tree.map(choose, when_true, when_false)


def _lane_float(value: Any, batch: int, name: str) -> jax.Array:
    array = jnp.asarray(value, dtype=jnp.float32)
    if array.ndim == 0:
        array = jnp.broadcast_to(array, (batch,))
    if array.shape != (batch,):
        raise ValueError(f"{name} must be scalar or have shape [batch]")
    return array


def _lane_bool(value: Any, batch: int, name: str) -> jax.Array:
    array = jnp.asarray(value, dtype=jnp.bool_)
    if array.shape != (batch,):
        raise ValueError(f"{name} must have shape [batch]")
    return array


def _optional_lane_bool(value: Any, batch: int, name: str) -> jax.Array:
    if value is None:
        return jnp.zeros((batch,), dtype=jnp.bool_)
    array = jnp.asarray(value, dtype=jnp.bool_)
    if array.ndim == 0:
        array = jnp.broadcast_to(array, (batch,))
    if array.shape != (batch,):
        raise ValueError(f"{name} must be scalar or have shape [batch]")
    return array


__all__ = [
    "GoalControl",
    "GoalRuntimeContext",
    "HierarchicalAgentCore",
    "HierarchicalCoreState",
    "HierarchicalStep",
    "ManagedGoal",
    "initialize_hierarchical_state",
]
