"""Python-driven hierarchical step loop for host-mode decision models.

``adk.runtime.hierarchical.collect_hierarchical`` runs reset, decide, step,
context and autoreset inside ``jax.lax.scan``. Every component is traced once,
so no Python, network call or wall-clock budget runs per step. ``run_eager``
runs the same sequence one step at a time in Python, with the same key splits,
calls, validation and per-lane autoreset. ``test_na_eager_loop.py`` checks
that, for the same key on CPU, its records match the compiled collector's bit
for bit.

The host seam is ``host_hook``. After each step's record and autoreset, it
receives the step index, the full transition, and the carried hierarchy state.
It may return a replacement :class:`HierarchicalCoreState`, for example one
with a new executive state written into ``carry.manager_state``. The next
``core`` call receives that replacement. Lanes that end an episode later are
still reset to ``initial_state``, as in the collector.

The manager is not a host seam. ``HierarchicalAgentCore`` calls it inside
``jax.lax.cond``, and JAX 0.11.2 traces both cond branches unless
``jax.disable_jit()`` is active. A manager therefore sees tracers even when
``run_eager`` calls the core outside ``jit``.

Nothing here is jitted. ``built.reset`` and the environment step run as
``built`` defines them. ``built`` may also be a host-side environment.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from adk.architecture.decision_context import DecisionContext
from adk.architecture.hierarchy import (
    HierarchicalAgentCore,
    HierarchicalCoreState,
    HierarchicalStep,
)
from adk.architecture.inputs import ActorPolicyInput, actor_policy_input
from adk.runtime.env_adapter import step_with_boundary
from adk.runtime.hierarchical import (
    HierarchicalActorInputFn,
    HierarchicalContextFn,
    HierarchicalLoopState,
    HierarchicalLoopTransition,
    HierarchicalRecordFn,
    _factor_rows,
    _validate_inputs,
    jax_hierarchical_context,
)
from adk.runtime.loop import ContextualPolicyInput, unknown_decision_context
from adk.runtime.tree import select_environment_rows

HostHook = Callable[
    [int, HierarchicalLoopTransition, HierarchicalCoreState],
    "HierarchicalCoreState | None",
]


def run_eager(
    built: Any,
    core: HierarchicalAgentCore,
    record_fn: HierarchicalRecordFn,
    key: jax.Array,
    steps: int,
    *,
    initial_state: HierarchicalCoreState,
    context_fn: HierarchicalContextFn = jax_hierarchical_context,
    actor_input_fn: HierarchicalActorInputFn = actor_policy_input,
    host_hook: HostHook | None = None,
    jit_core: bool = True,
) -> tuple[HierarchicalLoopState, Any]:
    """Run ``collect_hierarchical`` step by step in Python.

    ``jit_core`` compiles the hierarchy core once instead of re-tracing its
    ``lax.cond`` branches on every step (about 5 s per step eagerly, measured
    2026-10-17 on in-graph arms). The core only ever contains in-graph
    components, since a host-mode manager cannot run in the manager slot
    anyway; host work belongs in ``host_hook``, which stays in Python.

    Returns the final :class:`HierarchicalLoopState` and the ``record_fn``
    outputs stacked on a leading ``steps`` axis, as ``jax.lax.scan`` does.
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
    if host_hook is not None and not callable(host_hook):
        raise TypeError("host_hook must be callable")

    def _core_step(state, actor_input, context, key):
        return core(state, actor_input, context, key)

    step_core = jax.jit(_core_step) if jit_core else _core_step
    previous_action = jnp.asarray(initial_state.carry.previous_action)
    expected_factor_shape = (batch, previous_action.shape[1])
    reset_key, scan_key = jax.random.split(key)
    environment_state, policy_input = built.reset(
        jax.random.split(reset_key, batch)
    )
    initial_context = unknown_decision_context(batch)
    hierarchy_structure = jax.tree.structure(initial_state)
    decision_context = initial_context
    hierarchy_state = initial_state
    records = []

    for step_index, step_key in enumerate(jax.random.split(scan_key, steps)):
        hierarchy_key, environment_key, restart_key = jax.random.split(
            step_key,
            3,
        )
        current_actor_input = actor_input_fn(environment_state, policy_input)
        if not isinstance(current_actor_input, ActorPolicyInput):
            raise TypeError("actor_input_fn must return ActorPolicyInput")
        current_policy_input = ContextualPolicyInput(
            actor_input=current_actor_input,
            decision_context=decision_context,
        )
        hierarchy_step = step_core(
            hierarchy_state,
            current_actor_input,
            decision_context,
            hierarchy_key,
        )
        if not isinstance(hierarchy_step, HierarchicalStep):
            raise TypeError("core must return HierarchicalStep")
        if jax.tree.structure(hierarchy_step.state) != hierarchy_structure:
            raise TypeError("hierarchy state PyTree must match the initial state")
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
        transition = HierarchicalLoopTransition(
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
        records.append(record_fn(transition))

        fresh_state, fresh_input = built.reset(
            jax.random.split(restart_key, batch)
        )
        environment_state = select_environment_rows(
            done,
            fresh_state,
            environment_step.state,
        )
        policy_input = select_environment_rows(
            done,
            fresh_input,
            environment_step.policy_input,
        )
        decision_context = select_environment_rows(
            done,
            initial_context,
            candidate_context,
        )
        hierarchy_state = carried_hierarchy_state
        if host_hook is not None:
            replacement = host_hook(step_index, transition, hierarchy_state)
            if replacement is not None:
                if (
                    not isinstance(replacement, HierarchicalCoreState)
                    or jax.tree.structure(replacement) != hierarchy_structure
                ):
                    raise TypeError(
                        "host_hook must return None or a HierarchicalCoreState "
                        "with the initial state's PyTree structure"
                    )
                hierarchy_state = replacement

    final = HierarchicalLoopState(
        environment_state=environment_state,
        policy_input=policy_input,
        decision_context=decision_context,
        hierarchy_state=hierarchy_state,
    )
    return final, jax.tree.map(lambda *xs: jnp.stack(xs), *records)


__all__ = ["HostHook", "run_eager"]
