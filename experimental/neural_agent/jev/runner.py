"""Level 2 runner: binds a JEV instance to the ADK hierarchy core (ADR-001 D2).

``HierarchicalAgentCore`` calls its tactical manager with the goal payload and
no executive keyword (``adk/architecture/hierarchy.py:304-312``). A
``JevInstance`` also needs the executive state E_t. :class:`JevRunner` keeps E_t
in its own state, passes it to the instance, and enforces the latency budget of
a host instance.

Shapes, with B lanes: the runner state holds an ``ExecutiveArrays`` whose
leaves have leading axis B, and a ``GoalControl`` whose
``minimum_commitment_seconds`` and ``maximum_duration_seconds`` are float32[B]
and whose ``emergency_interrupt_mask`` is bool[B, 14].

The instance state, the goal payload and every diagnostic array must also have
leading axis B. The runner refuses any other shape, because ``hierarchy._select_rows``
keeps the stored value of such a leaf, and the core does not check payload shapes.

Two modes (``JevMode``):

* ``IN_GRAPH`` instances are called directly. They are not timed.
* ``HOST`` instances are called eagerly, never under a trace. Their output is
  blocked with ``jax.block_until_ready`` and timed with ``perf_counter``. A call
  slower than ``budget.latency_ms`` is a miss. The previous goal stays in force
  and the miss is counted (ADR-001 D2; README formula 7).

Statistics count eager calls only. A jitted call runs its Python body once, at
trace time, so the runner cannot count its executions.
"""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from adk.architecture.components import GoalDecision
from adk.architecture.decision_context import DecisionContext
from adk.architecture.hierarchy import GoalControl, GoalRuntimeContext, ManagedGoal
from adk.architecture.inputs import ActorPolicyInput
from experimental.neural_agent.contracts import (
    ComputeBudget,
    ExecutiveArrays,
    ExecutiveState,
    GoalPayload,
    JevBudget,
    JevInstance,
    JevMode,
    JevStats,
    SharedBeliefState,
    SkillControlTable,
    _require_control_table,
    control_for,
    default_control_table,
    empty_executive_arrays,
    executive_arrays,
    idle_goal,
)

# Diagnostics key that the runner adds to every decision: bool[B], True on a miss.
MISS_DIAGNOSTIC: str = "miss"
# Optional instance diagnostic: compute units spent, an integer array int32[B].
COMPUTE_DIAGNOSTIC: str = "compute_used"

_EMPTY_CATALOG_STATE = ExecutiveState(
    version=0,
    issued_tick=0,
    goals=(),
    budget=ComputeBudget(),
    facts=(),
)
_NAN = float("nan")


class JevRunnerState(NamedTuple):
    """State that the runner carries as the hierarchy core's ``manager_state``.

    Attributes:
        instance_state: The instance's batched state PyTree, leading axis B.
        executive: The lowered executive state E_t, an ``ExecutiveArrays`` with
            leading axis B.
        goal_control: The ``GoalControl`` of the last goal this runner
            proposed, batched [B]. A miss re-emits the current goal with this
            control, because the ADK core does not show the manager the control
            of the goal in force.
    """

    instance_state: Any
    executive: ExecutiveArrays
    goal_control: GoalControl


class JevRunner:
    """Adapt a :class:`JevInstance` to the ADK tactical-manager seam.

    The runner is the manager that ``HierarchicalAgentCore`` calls. It accepts
    the core's positional and keyword arguments and returns a ``GoalDecision``
    whose ``state`` is a :class:`JevRunnerState`.

    Args:
        instance: The JEV instance to run. Its ``mode`` selects how it is called.
        budget: The per-decision budget. ``latency_ms`` bounds a HOST call.
        skill_ids: The skill catalog in dispatcher order, with ``"idle"`` first
            by convention. Goals in E_t name skills by these ids.
        control: Keyword only. The goal control of each catalog skill, one row
            per id; the runner's idle goal takes row 0. Default
            ``default_control_table(skill_ids)``. Pass the instance's table.

    Raises:
        TypeError: If ``instance`` is not a ``JevInstance``, its ``mode`` is not
            a ``JevMode``, ``budget`` is not a ``JevBudget``, ``skill_ids``
            is a single string, or ``control`` is not a SkillControlTable.
        ValueError: If ``skill_ids`` repeats an id or holds an empty id, or
            ``control`` does not have one row per id.
    """

    def __init__(
        self,
        instance: JevInstance,
        budget: JevBudget,
        skill_ids: Sequence[str],
        *,
        control: SkillControlTable | None = None,
    ) -> None:
        if not isinstance(instance, JevInstance):
            raise TypeError(
                "instance must implement JevInstance: name, mode, initial_state and __call__"
            )
        if not isinstance(instance.mode, JevMode):
            raise TypeError("instance.mode must be a JevMode")
        if not isinstance(budget, JevBudget):
            raise TypeError("budget must be a JevBudget")
        if isinstance(skill_ids, (str, bytes)):
            raise TypeError("skill_ids must be a sequence of skill ids, not one string")
        catalog = tuple(skill_ids)
        # The contract's own check of the catalog: types, non-empty ids, no repeats.
        executive_arrays(_EMPTY_CATALOG_STATE, catalog, 1)
        self._instance = instance
        self._budget = budget
        self._skill_ids = catalog
        self._control = _require_control_table(
            default_control_table(catalog) if control is None else control, len(catalog)
        )
        self._decisions = 0
        self._misses = 0
        self._compute_units = 0
        self._latencies_ms: list[float] = []

    def initial_state(self, batch: int) -> JevRunnerState:
        """Return the reset runner state for ``batch`` lanes.

        The executive is empty (``empty_executive_arrays``), the instance state
        is the instance's own reset, and the goal control is idle's row of the
        control table, which equals ``idle_goal(batch).control`` for the idle
        contract. Start the carry with ``idle_goal(batch)`` so that the control
        in force matches this state.

        Args:
            batch: Number of lanes B, a positive int.

        Returns:
            A JevRunnerState whose ``executive`` follows ``empty_executive_arrays``.

        Raises:
            TypeError: If ``batch`` is not an integer.
            ValueError: If ``batch`` is not positive.
        """

        return JevRunnerState(
            instance_state=self._instance.initial_state(batch),
            executive=empty_executive_arrays(batch),
            goal_control=self._idle_goal(batch).control,
        )

    def set_executive(
        self,
        state: JevRunnerState,
        executive_state: ExecutiveState,
    ) -> JevRunnerState:
        """Return ``state`` with E_t replaced by ``executive_state``. Host side only.

        Args:
            state: A JevRunnerState with B lanes.
            executive_state: The ExecutiveState to lower. Each goal names a skill
                in this runner's catalog.

        Returns:
            A new JevRunnerState. ``instance_state`` and ``goal_control`` are unchanged.

        Raises:
            TypeError: If ``state`` is not a JevRunnerState, or ``executive_state``
                is not an ExecutiveState.
            KeyError: If a goal names a skill id that is not in the catalog.
        """

        _check_state(state)
        lowered = executive_arrays(executive_state, self._skill_ids, _lanes(state))
        return state._replace(executive=lowered)

    def stats(self) -> JevStats:
        """Return aggregate statistics over the eager calls since construction.

        Latency percentiles use the recorded host-call latencies (linear
        interpolation, ``numpy.percentile``). They are NaN when no host call
        has been timed, which is always the case for IN_GRAPH instances.

        Returns:
            A JevStats. ``decisions`` counts eager calls; each call decides every
            lane. ``misses`` counts host calls over budget. ``compute_units`` sums
            the ``compute_used`` diagnostics of eager calls.
        """

        if self._latencies_ms:
            samples = np.asarray(self._latencies_ms, dtype=np.float64)
            p50 = float(np.percentile(samples, 50))
            p95 = float(np.percentile(samples, 95))
        else:
            p50 = _NAN
            p95 = _NAN
        return JevStats(
            decisions=self._decisions,
            misses=self._misses,
            latency_ms_p50=p50,
            latency_ms_p95=p95,
            compute_units=self._compute_units,
        )

    def __call__(
        self,
        state: JevRunnerState,
        belief: SharedBeliefState,
        current_goal: GoalPayload,
        actor_input: ActorPolicyInput,
        key: jax.Array,
        /,
        *,
        decision_context: DecisionContext | None = None,
        goal_runtime: GoalRuntimeContext | None = None,
    ) -> GoalDecision:
        """Make one goal decision for every lane.

        Args:
            state: A JevRunnerState with B lanes.
            belief: The shared belief B_t, ``features`` float32[B, 10],
                ``feature_known`` bool[B, 10] and ``latent`` float32[B, 64].
            current_goal: The goal in force, a GoalPayload batched [B]. The ADK
                core passes ``ManagedGoal.payload`` (hierarchy.py:307).
            actor_input: The legal ActorPolicyInput. Passed through unchanged.
            key: A JAX PRNG key. Passed through unchanged.
            decision_context: Optional DecisionContext. Passed through unchanged.
            goal_runtime: Optional GoalRuntimeContext. Passed through unchanged.

        Returns:
            A GoalDecision. ``state`` is a JevRunnerState. ``goal`` is a
            ManagedGoal whose payload is a GoalPayload. ``terminate`` is bool[B],
            where False means continue. ``diagnostics`` holds the instance's
            mapping plus ``miss`` as bool[B]. On a miss, ``state`` is the input
            state and ``goal`` is ``current_goal`` with the stored control.

        Raises:
            TypeError: If an argument has the wrong Python type, or the instance
                returns something other than a GoalDecision.
            ValueError: If a HOST instance runs under a JAX trace, the instance's
                ``terminate`` has the wrong shape, a state, payload or diagnostic array
                has no leading lane axis, or its diagnostics use the reserved key ``miss``.
        """

        _check_state(state)
        if not isinstance(current_goal, GoalPayload):
            raise TypeError(
                "current_goal must be a GoalPayload: the ADK core passes "
                "ManagedGoal.payload to the manager (adk/architecture/hierarchy.py:307)"
            )
        lanes = _lanes(state)
        if self._instance.mode is JevMode.HOST:
            started = perf_counter()
            raw = self._invoke(
                state, belief, current_goal, actor_input, key, decision_context, goal_runtime
            )
            if _any_tracer(
                (state, belief, current_goal, actor_input, decision_context, goal_runtime, raw)
            ):
                raise ValueError(
                    "a HOST JEV instance ran under a JAX trace; it cannot run inside jit, "
                    "scan or lax.cond, such as the manager branch of HierarchicalAgentCore"
                )
            raw = jax.block_until_ready(raw)
            latency_ms = (perf_counter() - started) * 1000.0
            self._decisions += 1
            self._latencies_ms.append(latency_ms)
            missed = latency_ms > self._budget.latency_ms
            if missed:
                self._misses += 1
            return self._decide(
                state, current_goal, raw, lanes, missed=missed, eager=True, goal_runtime=goal_runtime
            )

        raw = self._invoke(
            state, belief, current_goal, actor_input, key, decision_context, goal_runtime
        )
        eager = not _any_tracer(
            (state, belief, current_goal, actor_input, decision_context, goal_runtime, raw)
        )
        if eager:
            self._decisions += 1
        return self._decide(
            state, current_goal, raw, lanes, missed=False, eager=eager, goal_runtime=goal_runtime
        )

    def _idle_goal(self, lanes: int) -> ManagedGoal:
        """Return ``idle_goal(lanes)`` with the control that the table gives idle (index 0)."""

        goal = idle_goal(lanes)
        return goal._replace(
            control=GoalControl(*control_for(self._control, goal.payload.skill_index))
        )

    def _invoke(
        self,
        state: JevRunnerState,
        belief: SharedBeliefState,
        current_goal: GoalPayload,
        actor_input: ActorPolicyInput,
        key: jax.Array,
        decision_context: DecisionContext | None,
        goal_runtime: GoalRuntimeContext | None,
    ) -> Any:
        """Call the instance with E_t bound from the runner state."""

        return self._instance(
            state.instance_state,
            belief,
            current_goal,
            actor_input,
            key,
            decision_context=decision_context,
            goal_runtime=goal_runtime,
            executive=state.executive,
        )

    def _decide(
        self,
        state: JevRunnerState,
        current_goal: GoalPayload,
        raw: Any,
        lanes: int,
        *,
        missed: bool,
        eager: bool,
        goal_runtime: GoalRuntimeContext | None = None,
    ) -> GoalDecision:
        """Turn the instance output into the runner's decision, honouring a miss.

        On a miss the previous goal stands. When the core is forcing a switch on
        a lane at that moment (its goal is invalid, timed out, or the actor asked
        to end it), "stands" is impossible: the core adopts whatever the manager
        returns, and re-emitting the ended goal would renew it with a fresh age.
        Those lanes receive the idle goal instead (ADR-001 D2, decision
        2026-10-16). Without ``goal_runtime`` the forced lanes are unknown and
        the previous goal is re-emitted for every lane.
        """

        decision = _validated_decision(raw, lanes)
        if eager:
            self._compute_units += _compute_used(decision.diagnostics)
        diagnostics = _with_miss(decision.diagnostics, missed, lanes)
        if missed:
            held = ManagedGoal(payload=current_goal, control=state.goal_control)
            if goal_runtime is not None:
                forced = (
                    ~jnp.asarray(goal_runtime.valid, dtype=jnp.bool_)
                    | jnp.asarray(goal_runtime.timed_out, dtype=jnp.bool_)
                    | jnp.asarray(goal_runtime.actor_termination_requested, dtype=jnp.bool_)
                )
                idle = self._idle_goal(lanes)

                def pick(idle_leaf: Any, held_leaf: Any) -> Any:
                    idle_arr = jnp.asarray(idle_leaf)
                    held_arr = jnp.asarray(held_leaf, dtype=idle_arr.dtype)
                    mask = forced.reshape((lanes,) + (1,) * (idle_arr.ndim - 1))
                    return jnp.where(mask, idle_arr, held_arr)

                held = jax.tree.map(pick, idle, held)
            return GoalDecision(
                state=state,
                goal=held,
                terminate=jnp.zeros((lanes,), dtype=jnp.bool_),
                diagnostics=diagnostics,
            )
        return GoalDecision(
            state=JevRunnerState(
                instance_state=decision.state,
                executive=state.executive,
                goal_control=decision.goal.control,
            ),
            goal=decision.goal,
            terminate=decision.terminate,
            diagnostics=diagnostics,
        )


def _check_state(state: Any) -> None:
    """Raise TypeError unless ``state`` is a JevRunnerState."""

    if not isinstance(state, JevRunnerState):
        raise TypeError("state must be a JevRunnerState")


def _lanes(state: JevRunnerState) -> int:
    """Return the lane count B, read from the executive version axis."""

    return int(state.executive.version.shape[0])


def _any_tracer(tree: Any) -> bool:
    """Return True when a leaf of ``tree`` is a JAX tracer, so the code runs under a trace."""

    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(tree))


def _validated_decision(raw: Any, lanes: int) -> GoalDecision:
    """Check an instance output, its ``terminate`` shape and its lane axes."""

    if not isinstance(raw, GoalDecision):
        raise TypeError("JEV instance must return a GoalDecision")
    if not isinstance(raw.goal, ManagedGoal) or not isinstance(raw.goal.control, GoalControl):
        raise TypeError("GoalDecision.goal must be a ManagedGoal with a GoalControl")
    if not isinstance(raw.goal.payload, GoalPayload):
        raise TypeError("ManagedGoal.payload must be a GoalPayload")
    if not isinstance(raw.diagnostics, Mapping):
        raise TypeError("GoalDecision.diagnostics must be a mapping")
    terminate = jnp.asarray(raw.terminate, dtype=jnp.bool_)
    if terminate.shape != (lanes,):
        raise ValueError(
            f"GoalDecision.terminate must have shape ({lanes},), got {terminate.shape}"
        )
    _require_lane_leaves(raw.state, lanes, "GoalDecision.state")
    _require_lane_leaves(raw.goal.payload, lanes, "ManagedGoal.payload")
    _require_lane_leaves(raw.diagnostics, lanes, "GoalDecision.diagnostics")
    return raw._replace(terminate=terminate)


def _require_lane_leaves(tree: Any, lanes: int, name: str) -> None:
    """Raise ValueError unless every array leaf of ``tree`` has leading axis ``lanes``."""

    for leaf in jax.tree.leaves(tree):
        shape = tuple(jnp.shape(leaf))
        if not shape or shape[0] != lanes:
            raise ValueError(
                f"{name} must have the leading lane axis {lanes} on every array leaf, "
                f"got shape {shape}"
            )


def _with_miss(diagnostics: Mapping[str, Any], missed: bool, lanes: int) -> dict[str, Any]:
    """Return the instance diagnostics plus the ``miss`` flag, bool[lanes]."""

    if MISS_DIAGNOSTIC in diagnostics:
        raise ValueError(
            f"JEV diagnostics must not use the reserved key {MISS_DIAGNOSTIC!r}"
        )
    merged: dict[str, Any] = dict(diagnostics)
    merged[MISS_DIAGNOSTIC] = jnp.full((lanes,), missed, dtype=jnp.bool_)
    return merged


def _compute_used(diagnostics: Mapping[str, Any]) -> int:
    """Return the compute units that an eager call reports, or 0 when it reports none."""

    value = diagnostics.get(COMPUTE_DIAGNOSTIC)
    if value is None:
        return 0
    return int(np.sum(np.asarray(value)))


__all__ = ["JevRunner", "JevRunnerState", "MISS_DIAGNOSTIC", "COMPUTE_DIAGNOSTIC"]
