"""Algorithm-neutral compiled environment loop.

Algorithms supply two small functions: a policy and a record projection.  The
SDK owns key splitting, structured reset/step, terminal autoreset, and recurrent
carry hygiene once, so PPO, imitation learning, model-based agents, and custom
online learners do not each reimplement those error-prone mechanics.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from adk.architecture.decision_context import (
    ActionLifecycle,
    ActionVerb,
    DecisionContext,
    DecisionEvents,
    DecisionEvent,
    DecisionTiming,
    ExecutionState,
    jax_decision_context,
)
from adk.architecture.inputs import ActorPolicyInput, actor_policy_input
from adk.policy import Policy
from adk.runtime.env_adapter import (
    BuiltAgent,
    JaxEnvironmentState,
    PolicyInput,
    step_with_boundary,
)
from adk.runtime.episode import EpisodeBoundary
from adk.runtime.tree import select_environment_rows


class LoopState(NamedTuple):
    """Final environment, actor projection, and algorithm carry."""

    environment_state: JaxEnvironmentState
    policy_input: PolicyInput
    algorithm_carry: Any


class LoopTransition(NamedTuple):
    """Lossless transition offered to a caller-supplied record projection.

    Both wrapped states retain the complete upstream structured state.  Actor
    inputs retain the exact legal structured observation, action surface,
    bridge-compatible dense row, and mask.  ``info`` is unfiltered.
    """

    state: JaxEnvironmentState
    actor_input: ActorPolicyInput
    action_factors: jax.Array
    reward: jax.Array
    done: jax.Array
    info: Any
    next_state: JaxEnvironmentState
    next_actor_input: ActorPolicyInput
    terminated: jax.Array | None = None
    truncated: jax.Array | None = None

    @property
    def boundary(self) -> EpisodeBoundary:
        """Boundary signal plus availability of its separate causes."""

        return EpisodeBoundary(self.done, self.terminated, self.truncated)


RecordFn = Callable[[LoopTransition], Any]


class ContextualPolicyInput(NamedTuple):
    """Exact actor product plus normalized PLAN-style decision evidence."""

    actor_input: ActorPolicyInput
    decision_context: DecisionContext


class ContextualLoopState(NamedTuple):
    """Final environment, context, and algorithm carry for contextual loops."""

    environment_state: JaxEnvironmentState
    policy_input: PolicyInput
    decision_context: DecisionContext
    algorithm_carry: Any


class ContextualLoopTransition(NamedTuple):
    """Lossless transition supplied to a contextual record projection.

    ``next_policy_input.decision_context`` is always the context derived from
    this transition, including on terminal rows.  Autoreset is applied only
    to the next loop carry, so terminal evidence is never replaced by reset
    sentinels. ``info`` is the exact unfiltered environment-info PyTree.
    """

    state: JaxEnvironmentState
    policy_input: ContextualPolicyInput
    action_factors: jax.Array
    reward: jax.Array
    done: jax.Array
    info: Any
    next_state: JaxEnvironmentState
    next_policy_input: ContextualPolicyInput
    terminated: jax.Array | None = None
    truncated: jax.Array | None = None

    @property
    def boundary(self) -> EpisodeBoundary:
        """Boundary signal plus availability of its separate causes."""

        return EpisodeBoundary(self.done, self.terminated, self.truncated)

    @property
    def actor_input(self) -> ActorPolicyInput:
        """Compatibility-shaped access to the pre-step actor product."""

        return self.policy_input.actor_input

    @property
    def decision_context(self) -> DecisionContext:
        """Pre-step normalized decision context."""

        return self.policy_input.decision_context

    @property
    def next_actor_input(self) -> ActorPolicyInput:
        """Compatibility-shaped access to the candidate next actor product."""

        return self.next_policy_input.actor_input

    @property
    def next_decision_context(self) -> DecisionContext:
        """Candidate context before any terminal-row autoreset."""

        return self.next_policy_input.decision_context

    @property
    def raw_info(self) -> Any:
        """Explicit alias for the exact unfiltered transition diagnostics."""

        return self.info


ContextualPolicy = Callable[
    [Any, ContextualPolicyInput, jax.Array],
    tuple[Any, jax.Array],
]
ContextualRecordFn = Callable[[ContextualLoopTransition], Any]


def unknown_decision_context(batch: int) -> DecisionContext:
    """Return a fully unknown, zero-valued context for reset policy rows."""

    if isinstance(batch, bool) or not isinstance(batch, int):
        raise TypeError("batch must be an integer")
    if batch < 1:
        raise ValueError("batch must be positive")
    lifecycle_shape = (batch, len(ActionVerb), len(ExecutionState))
    event_shape = (batch, len(DecisionEvent))
    return DecisionContext(
        timing=DecisionTiming(
            tick=jnp.zeros((batch,), dtype=jnp.int32),
            tick_known=jnp.zeros((batch,), dtype=jnp.bool_),
            delta_seconds=jnp.zeros((batch,), dtype=jnp.float32),
            delta_seconds_known=jnp.zeros((batch,), dtype=jnp.bool_),
        ),
        lifecycle=ActionLifecycle(
            values=jnp.zeros(lifecycle_shape, dtype=jnp.bool_),
            known=jnp.zeros(lifecycle_shape, dtype=jnp.bool_),
        ),
        events=DecisionEvents(
            values=jnp.zeros(event_shape, dtype=jnp.bool_),
            known=jnp.zeros(event_shape, dtype=jnp.bool_),
        ),
    )


def collect(
    built: BuiltAgent,
    policy_fn: Policy,
    record_fn: RecordFn,
    key: jax.Array,
    steps: int,
    *,
    initial_carry: Any = None,
) -> tuple[LoopState, Any]:
    """Run a JAX ``lax.scan`` and stack whatever ``record_fn`` returns.

    ``record_fn`` executes while tracing and must return a fixed-shape JAX
    PyTree.  It can retain the full structured transition or project only the
    tensors an algorithm needs.  No host synchronization occurs in this loop.
    """

    if not callable(policy_fn):
        raise TypeError("policy_fn must be callable")
    if not callable(record_fn):
        raise TypeError("record_fn must be callable")
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError("steps must be an integer")
    if steps < 1:
        raise ValueError("steps must be positive")

    reset_key, scan_key = jax.random.split(key)
    state, policy_input = built.reset(jax.random.split(reset_key, built.batch))
    carry_structure = jax.tree.structure(initial_carry)

    def body(loop_state: LoopState, step_key):
        state, policy_input, algorithm_carry = loop_state
        policy_key, environment_key, restart_key = jax.random.split(step_key, 3)
        current_actor_input = actor_policy_input(state, policy_input)
        next_carry, factors = policy_fn(
            algorithm_carry,
            current_actor_input,
            policy_key,
        )
        if jax.tree.structure(next_carry) != carry_structure:
            raise TypeError(
                "policy carry PyTree must match initial_carry; pass an "
                "explicit initial carry for every stateful policy"
            )
        environment_step = step_with_boundary(
            built,
            state,
            factors,
            jax.random.split(environment_key, built.batch),
        )
        candidate_state = environment_step.state
        candidate_input = environment_step.policy_input
        reward = environment_step.reward
        done = environment_step.done
        info = environment_step.info
        candidate_actor_input = actor_policy_input(
            candidate_state,
            candidate_input,
        )
        record = record_fn(
            LoopTransition(
                state=state,
                actor_input=current_actor_input,
                action_factors=factors,
                reward=reward,
                done=done,
                info=info,
                next_state=candidate_state,
                next_actor_input=candidate_actor_input,
                terminated=environment_step.terminated,
                truncated=environment_step.truncated,
            )
        )

        fresh_state, fresh_input = built.reset(
            jax.random.split(restart_key, built.batch)
        )
        state = select_environment_rows(done, fresh_state, candidate_state)
        policy_input = select_environment_rows(done, fresh_input, candidate_input)
        if next_carry is not None:
            next_carry = select_environment_rows(
                done,
                initial_carry,
                next_carry,
            )
        return LoopState(state, policy_input, next_carry), record

    return jax.lax.scan(
        body,
        LoopState(state, policy_input, initial_carry),
        jax.random.split(scan_key, steps),
    )


def collect_contextual(
    built: BuiltAgent,
    policy_fn: ContextualPolicy,
    record_fn: ContextualRecordFn,
    key: jax.Array,
    steps: int,
    *,
    initial_carry: Any = None,
) -> tuple[ContextualLoopState, Any]:
    """Collect with exact actor input and normalized context inside one scan.

    This is additive to :func:`collect`: existing policies keep their original
    input contract. Contextual policies receive an all-unknown context at
    initial reset, then the exact candidate context derived from each prior
    transition. Terminal lanes reset to unknown before the next policy call,
    while the terminal transition record retains its candidate context.
    """

    if not callable(policy_fn):
        raise TypeError("policy_fn must be callable")
    if not callable(record_fn):
        raise TypeError("record_fn must be callable")
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError("steps must be an integer")
    if steps < 1:
        raise ValueError("steps must be positive")

    reset_key, scan_key = jax.random.split(key)
    state, policy_input = built.reset(jax.random.split(reset_key, built.batch))
    unknown_context = unknown_decision_context(built.batch)
    carry_structure = jax.tree.structure(initial_carry)

    def body(loop_state: ContextualLoopState, step_key):
        state, policy_input, decision_context, algorithm_carry = loop_state
        policy_key, environment_key, restart_key = jax.random.split(step_key, 3)
        current_actor_input = actor_policy_input(state, policy_input)
        current_policy_input = ContextualPolicyInput(
            actor_input=current_actor_input,
            decision_context=decision_context,
        )
        next_carry, factors = policy_fn(
            algorithm_carry,
            current_policy_input,
            policy_key,
        )
        if jax.tree.structure(next_carry) != carry_structure:
            raise TypeError(
                "policy carry PyTree must match initial_carry; pass an "
                "explicit initial carry for every stateful policy"
            )
        environment_step = step_with_boundary(
            built,
            state,
            factors,
            jax.random.split(environment_key, built.batch),
        )
        candidate_state = environment_step.state
        candidate_input = environment_step.policy_input
        reward = environment_step.reward
        done = environment_step.done
        info = environment_step.info
        candidate_actor_input = actor_policy_input(
            candidate_state,
            candidate_input,
        )
        candidate_context = jax_decision_context(
            info,
            actor_input=candidate_actor_input,
            previous_actor_input=current_actor_input,
        ).context
        candidate_policy_input = ContextualPolicyInput(
            actor_input=candidate_actor_input,
            decision_context=candidate_context,
        )
        record = record_fn(
            ContextualLoopTransition(
                state=state,
                policy_input=current_policy_input,
                action_factors=factors,
                reward=reward,
                done=done,
                info=info,
                next_state=candidate_state,
                next_policy_input=candidate_policy_input,
                terminated=environment_step.terminated,
                truncated=environment_step.truncated,
            )
        )

        fresh_state, fresh_input = built.reset(
            jax.random.split(restart_key, built.batch)
        )
        state = select_environment_rows(done, fresh_state, candidate_state)
        policy_input = select_environment_rows(done, fresh_input, candidate_input)
        decision_context = select_environment_rows(
            done,
            unknown_context,
            candidate_context,
        )
        if next_carry is not None:
            next_carry = select_environment_rows(
                done,
                initial_carry,
                next_carry,
            )
        return (
            ContextualLoopState(
                state,
                policy_input,
                decision_context,
                next_carry,
            ),
            record,
        )

    return jax.lax.scan(
        body,
        ContextualLoopState(
            state,
            policy_input,
            unknown_context,
            initial_carry,
        ),
        jax.random.split(scan_key, steps),
    )


__all__ = [
    "ContextualLoopState",
    "ContextualLoopTransition",
    "ContextualPolicy",
    "ContextualPolicyInput",
    "ContextualRecordFn",
    "LoopState",
    "LoopTransition",
    "RecordFn",
    "collect",
    "collect_contextual",
    "unknown_decision_context",
]
