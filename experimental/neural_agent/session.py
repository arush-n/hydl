"""One continuous run of the three-level stack: executive seam, JEV, skills.

This is the glue no single layer owns. It drives
:func:`experimental.neural_agent.eval.eager_loop.run_eager` with a host hook
that does, after every environment step:

1. advance a virtual clock by one tick (``tick_seconds``, 0.045 s by default,
   the gym's ``loaded_dt``);
2. turn goal switches into :class:`~experimental.neural_agent.contracts.GoalReport`
   records for the executive's report channel;
3. step the executive seam at the current tick and, when it proposes a new
   E_t version, write it into the manager state through ``executive_writer``;
4. for a host-mode decision model, decide in Python at the JEV cadence or when
   a known event is active, time the decision, and on success write the new
   ``ManagedGoal`` into the carried hierarchy state. A decision slower than the
   budget leaves the previous goal in force and counts a miss (ADR-001 D2).

The ADK core calls its manager inside ``jax.lax.cond``, so a Python or
network-backed decision model can never run in the manager slot. Host-mode
instances therefore run here, between steps, while the in-graph manager holds
the current goal. In-graph instances (J0, J1) run in the manager slot as usual
and need no host decider.

Everything written back into the hierarchy state keeps the PyTree structure of
``initial_state``; the eager loop rejects anything else.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

from adk.architecture.hierarchy import (
    HierarchicalAgentCore,
    HierarchicalCoreState,
    ManagedGoal,
)
from adk.runtime.hierarchical import HierarchicalLoopTransition

from experimental.neural_agent.contracts import (
    ExecutiveState,
    GoalPayload,
    GoalReport,
    SkillOutcome,
)
from experimental.neural_agent.eval.eager_loop import run_eager
from experimental.neural_agent.shared_state import SharedState
from experimental.neural_agent.shared_state_writers import SessionScribe

TICK_SECONDS = 0.045

_FAILED = object()
"""Sentinel for a host decider that raised; distinct from None, which means keep the goal."""
"""Seconds per environment tick: the gym's ``loaded_dt``. ADK comments that say
30 ticks per second are stale; the memory note horizon-ticks-to-seconds and the
pursuit contract agree on 0.045."""


@runtime_checkable
class ExecutiveSeam(Protocol):
    """What the session needs from an executive harness.

    ``step(tick)`` returns a new :class:`ExecutiveState` when the executive
    proposed one at this tick, else ``None``. It must never block the caller
    for longer than a step budget; asynchronous harnesses return the latest
    completed proposal.
    """

    def step(self, tick: int, /) -> ExecutiveState | None: ...


ExecutiveWriter = Callable[[Any, ExecutiveState], Any]
"""``(manager_state, executive_state) -> manager_state`` with identical PyTree
structure. For an in-graph JEV this is ``JevRunner.set_executive``."""


@runtime_checkable
class HostDecider(Protocol):
    """A host-mode decision model adapter.

    ``decide`` receives the transition just recorded, the carried hierarchy
    state after autoreset, and the latest executive state (``None`` before the
    first proposal). It returns a batched :class:`ManagedGoal` with the same
    PyTree structure as ``initial_state.carry.goal``, or ``None`` to keep the
    current goal. Exceptions are caught and counted as misses.

    The session binds four optional attributes when the decider has them:
    ``scratchpad``, ``host_cadence_ticks``, ``tick_seconds`` and
    ``current_tick`` (see :func:`run_session`).
    """

    def decide(
        self,
        transition: HierarchicalLoopTransition,
        hierarchy_state: HierarchicalCoreState,
        executive: ExecutiveState | None,
        /,
    ) -> ManagedGoal | None: ...


class ReportSink(Protocol):
    def push_goal_report(self, report: GoalReport, /) -> None: ...


@dataclasses.dataclass(frozen=True, slots=True)
class SessionConfig:
    """Timing knobs for one session."""

    steps: int
    tick_seconds: float = TICK_SECONDS
    host_cadence_ticks: int = 11
    host_budget_ms: float = 20.0
    host_on_events: bool = True
    # When True, a host decider's goal is applied per lane only where the core would allow a
    # switch (the goal is invalid, its minimum commitment elapsed, it timed out, or an emergency
    # event fired) and only where it differs from the goal in force; unchanged lanes keep their
    # goal age. This mirrors ``HierarchicalAgentCore`` (switch_allowed) so host instances obey the
    # skill's contract (owner ruling 2026-10-17: commitment belongs to the skill). Default True since
    # 2026-10-17 late: the learned instance passed G4 on three seeds under it and the program executor
    # reproduces the rule under it once its clock reads the age's increase (ledger). False restores the
    # earlier seam, which applied every host goal and zeroed every lane's goal age.
    host_respects_commitment: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("SessionConfig.steps must be a positive int")
        if not (math.isfinite(self.tick_seconds) and self.tick_seconds > 0):
            raise ValueError("SessionConfig.tick_seconds must be positive")
        if self.host_cadence_ticks < 1:
            raise ValueError("SessionConfig.host_cadence_ticks must be >= 1")
        if not (math.isfinite(self.host_budget_ms) and self.host_budget_ms > 0):
            raise ValueError("SessionConfig.host_budget_ms must be positive")


class HostStats(NamedTuple):
    """Host-mode decision accounting (ADR-001 §9 metrics).

    ``misses`` counts decisions over the latency budget plus decisions that
    raised. A decider returning ``None`` kept the goal on purpose and is not a
    miss.
    """

    decisions: int
    misses: int
    errors: int
    latency_ms_p50: float
    latency_ms_p95: float
    held: int = 0
    """Lane-decisions where a host goal change was requested but held by the skill's commitment
    (only counted with ``SessionConfig.host_respects_commitment``)."""


class SessionMetrics(NamedTuple):
    ticks: int
    virtual_seconds: float
    goal_switches_per_lane: tuple[int, ...]
    goal_reports: int
    executive_versions_applied: int
    host: HostStats
    wall_seconds: float


class SessionResult(NamedTuple):
    final_state: Any
    records: Any
    metrics: SessionMetrics
    reports: tuple[GoalReport, ...]


def default_record(transition: HierarchicalLoopTransition) -> dict[str, jax.Array]:
    """Arrays worth keeping per step. Only array leaves, as the loops require."""

    step = transition.hierarchy_step
    payload = step.active_goal.payload
    record: dict[str, jax.Array] = {
        "action_factors": transition.action_factors,
        "reward": transition.reward,
        "done": transition.done,
        "goal_switched": step.goal_switched,
        "manager_due": step.manager_due,
        "goal_age_seconds": step.goal_runtime.age_seconds,
        "emergency_interrupt": step.goal_runtime.emergency_interrupt,
        "timed_out": step.goal_runtime.timed_out,
        "events_active": step.goal_runtime.active_events,
    }
    if isinstance(payload, GoalPayload):
        record["skill_index"] = payload.skill_index
        record["executive_version"] = payload.executive_version
        record["goal_slot"] = payload.goal_slot
    return record


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _goal_differs(new_payload: Any, current_payload: Any, batch: int) -> jax.Array:
    """bool[B]: lanes where the proposed goal differs from the goal in force (skill, slot or params).

    Payloads that are not GoalPayloads are treated as different on every lane.
    """

    if not (isinstance(new_payload, GoalPayload) and isinstance(current_payload, GoalPayload)):
        return jnp.ones((batch,), dtype=jnp.bool_)
    return (
        (jnp.asarray(new_payload.skill_index) != jnp.asarray(current_payload.skill_index))
        | (jnp.asarray(new_payload.goal_slot) != jnp.asarray(current_payload.goal_slot))
        | jnp.any(jnp.asarray(new_payload.params) != jnp.asarray(current_payload.params), axis=-1)
    )


def _lane_mask(mask: jax.Array, leaf: jax.Array) -> jax.Array:
    """Reshape a bool[B] lane mask so it broadcasts over ``leaf``'s trailing axes."""

    return jnp.reshape(mask, (mask.shape[0],) + (1,) * (jnp.ndim(leaf) - 1))


def _outcome_for_switch(step: Any, lane: int) -> SkillOutcome:
    """Classify why a lane's goal ended, from the goal runtime facts."""

    runtime = step.goal_runtime
    if bool(runtime.timed_out[lane]):
        return SkillOutcome.TIMEOUT
    if bool(runtime.emergency_interrupt[lane]):
        return SkillOutcome.INTERRUPTED
    if bool(runtime.actor_termination_requested[lane]):
        return SkillOutcome.COMPLETED
    return SkillOutcome.INTERRUPTED


def run_session(
    built: Any,
    core: HierarchicalAgentCore,
    initial_state: HierarchicalCoreState,
    key: jax.Array,
    config: SessionConfig,
    *,
    skill_ids: Sequence[str],
    executive: ExecutiveSeam | None = None,
    executive_writer: ExecutiveWriter | None = None,
    host_decider: HostDecider | None = None,
    report_sink: ReportSink | None = None,
    shared_state: SharedState | None = None,
    record_fn: Callable[[HierarchicalLoopTransition], Any] = default_record,
    state_binding: Any = None,
) -> SessionResult:
    """Run the stack for ``config.steps`` ticks with the executive in the loop.

    ``skill_ids`` is the dispatcher's registry order; it labels reports. When
    ``executive`` is given, ``executive_writer`` is required: it is how a new
    E_t reaches the manager. ``host_decider`` is only for host-mode JEV
    instances; in-graph instances live inside ``core``. When ``shared_state`` is
    given, a :class:`SessionScribe` writes the belief, events, skills and
    decision sections at the host cadence, and each goal report into the
    skills section (``docs/SYSTEM_ONE.md`` §8.10).

    A host decider may expose optional attributes that the session binds
    when the decider has them (``jev/programs.py`` ``ProgramExecutor`` does):
    ``scratchpad`` gets ``shared_state`` (when given), ``host_cadence_ticks``
    and ``tick_seconds`` get ``config``'s values, and ``host_respects_commitment``
    gets the seam's regime, all once before the run; ``current_tick`` gets the
    clock tick of step 1 before each decide.
    With them a decider writes its requests into S_t in the session's tick
    space (``docs/SYSTEM_ONE.md`` §9.10). A decider with a callable ``on_tick``
    is called on every tick, after the scratchpad sections of that tick are
    written, as ``on_tick(clock_tick, transition, hierarchy_state)``; the lower
    layer's combiner wrapper uses it to run between decisions (2026-10-17 late).
    ``on_tick`` returns None (the state is kept) or a replacement for
    ``carry.actor_state`` with the same PyTree structure (2026-10-19): the session
    writes it into the carried state, so the next tick sees it. A different
    structure raises ValueError.

    ``state_binding`` (optional, default None: the session is unchanged) runs the skill layer's state tick every tick,
    last, after the decision and ``on_tick`` (``shared_state.skill_writes.StateTickBinding``, task #136): its
    ``on_tick(shared_state, clock_tick, decided, transition)`` takes the decision's writes on a tick the host decided
    and the requests that ended on every tick. It requires ``shared_state``.
    """

    if executive is not None and executive_writer is None:
        raise ValueError("executive_writer is required when an executive is given")
    if state_binding is not None and shared_state is None:
        raise ValueError("state_binding requires shared_state (it reads the writes and the statuses there)")
    if not isinstance(initial_state, HierarchicalCoreState):
        raise TypeError("initial_state must be a HierarchicalCoreState")
    ids = tuple(str(s) for s in skill_ids)
    if not ids:
        raise ValueError("skill_ids must not be empty")
    scribe = None if shared_state is None else SessionScribe(shared_state, ids)
    if host_decider is not None:
        if shared_state is not None and hasattr(host_decider, "scratchpad"):
            host_decider.scratchpad = shared_state
        if hasattr(host_decider, "host_cadence_ticks"):
            host_decider.host_cadence_ticks = config.host_cadence_ticks
        if hasattr(host_decider, "tick_seconds"):
            host_decider.tick_seconds = config.tick_seconds
        if hasattr(host_decider, "host_respects_commitment"):
            host_decider.host_respects_commitment = config.host_respects_commitment
    binds_tick = host_decider is not None and hasattr(host_decider, "current_tick")
    observes_ticks = host_decider is not None and callable(getattr(host_decider, "on_tick", None))

    batch = int(jnp.asarray(initial_state.carry.previous_action).shape[0])
    goal_structure = jax.tree.structure(initial_state.carry.goal)

    clock_tick = 0
    latest_executive: ExecutiveState | None = None
    information_consumed = 0  # host decider information requests already handed to the scribe
    executive_questions_consumed = 0  # the executive seam's questions (unknown skills served) already handed over
    executive_versions_applied = 0
    switches = np.zeros((batch,), dtype=np.int64)
    reports: list[GoalReport] = []
    host_latencies: list[float] = []
    host_misses = 0
    host_errors = 0
    host_decisions = 0
    host_held = 0
    goal_started_tick = np.zeros((batch,), dtype=np.int64)

    def hook(
        step_index: int,
        transition: HierarchicalLoopTransition,
        hierarchy_state: HierarchicalCoreState,
    ) -> HierarchicalCoreState | None:
        nonlocal clock_tick, latest_executive, executive_versions_applied, information_consumed, executive_questions_consumed
        nonlocal host_misses, host_errors, host_decisions, host_held

        clock_tick += 1
        step = transition.hierarchy_step
        new_state = hierarchy_state
        changed = False
        decided = False

        # 2. goal switches -> reports. A lane whose goal was not yet valid
        # adopts its first goal; that is not a switch and produces no report.
        valid_before = np.asarray(transition.hierarchy_state.goal_valid, dtype=bool)
        switched = np.asarray(step.goal_switched, dtype=bool) & valid_before
        if switched.any():
            payload_before = transition.hierarchy_state.carry.goal.payload
            previous_index = (
                np.asarray(payload_before.skill_index)
                if isinstance(payload_before, GoalPayload)
                else np.zeros((batch,), dtype=np.int64)
            )
            previous_version = (
                np.asarray(payload_before.executive_version)
                if isinstance(payload_before, GoalPayload)
                else np.zeros((batch,), dtype=np.int64)
            )
            for lane in np.flatnonzero(switched):
                switches[lane] += 1
                index = int(previous_index[lane])
                report = GoalReport(
                    tick=clock_tick,
                    lane=int(lane),
                    skill_id=ids[index] if 0 <= index < len(ids) else f"skill_{index}",
                    executive_version=int(previous_version[lane]),
                    outcome=_outcome_for_switch(step, int(lane)),
                    duration_ticks=int(clock_tick - goal_started_tick[lane]),
                    detail="",
                )
                goal_started_tick[lane] = clock_tick
                reports.append(report)
                if report_sink is not None:
                    report_sink.push_goal_report(report)
                if scribe is not None:
                    scribe.push_goal_report(report)

        # 3. executive seam
        if executive is not None:
            proposal = executive.step(clock_tick)
            if proposal is not None:
                if not isinstance(proposal, ExecutiveState):
                    raise TypeError("executive.step must return an ExecutiveState or None")
                manager_state = executive_writer(new_state.carry.manager_state, proposal)  # type: ignore[misc]
                if jax.tree.structure(manager_state) != jax.tree.structure(
                    new_state.carry.manager_state
                ):
                    raise ValueError("executive_writer changed the manager_state structure")
                new_state = new_state._replace(
                    carry=new_state.carry._replace(manager_state=manager_state)
                )
                latest_executive = proposal
                executive_versions_applied += 1
                changed = True

        # 4. host-mode decision model
        if host_decider is not None:
            due = (step_index % config.host_cadence_ticks) == 0
            if config.host_on_events and not due:
                due = bool(np.asarray(step.goal_runtime.active_events).any())
            if due:
                host_decisions += 1
                decided = True
                if binds_tick:
                    host_decider.current_tick = clock_tick
                started = time.perf_counter()
                try:
                    goal = host_decider.decide(transition, new_state, latest_executive)
                except Exception:  # noqa: BLE001 - a host failure keeps the goal, by contract
                    goal = _FAILED
                    host_errors += 1
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                host_latencies.append(elapsed_ms)
                if elapsed_ms > config.host_budget_ms or goal is _FAILED:
                    # Over budget, or the decider raised: the previous goal stands.
                    host_misses += 1
                elif goal is None:
                    # The decider chose to keep the current goal; not a miss.
                    pass
                else:
                    if not isinstance(goal, ManagedGoal) or jax.tree.structure(goal) != goal_structure:
                        raise ValueError("host_decider returned a goal with the wrong structure")
                    if config.host_respects_commitment:
                        current = new_state.carry.goal
                        runtime = step.goal_runtime
                        differs = _goal_differs(goal.payload, current.payload, batch)
                        allowed = (
                            ~jnp.asarray(new_state.goal_valid, dtype=jnp.bool_)
                            | jnp.asarray(runtime.commitment_elapsed, dtype=jnp.bool_)
                            | jnp.asarray(runtime.timed_out, dtype=jnp.bool_)
                            | jnp.asarray(runtime.emergency_interrupt, dtype=jnp.bool_)
                        )
                        apply = differs & allowed
                        host_held += int(np.asarray(differs & ~allowed).sum())
                        if bool(np.asarray(apply).any()):
                            merged = jax.tree.map(
                                lambda new, old: jnp.where(_lane_mask(apply, new), new, old), goal, current
                            )
                            new_state = new_state._replace(
                                carry=new_state.carry._replace(
                                    goal=merged,
                                    goal_age=jnp.where(apply, 0.0, new_state.carry.goal_age),
                                ),
                                goal_valid=jnp.asarray(new_state.goal_valid, dtype=jnp.bool_) | apply,
                            )
                            changed = True
                    else:
                        new_state = new_state._replace(
                            carry=new_state.carry._replace(
                                goal=goal,
                                goal_age=jnp.zeros((batch,), dtype=jnp.float32),
                            ),
                            goal_valid=jnp.ones((batch,), dtype=jnp.bool_),
                        )
                        changed = True

        # 5. scratchpad: the sections the session owns, once per host cadence tick
        if scribe is not None and (step_index % config.host_cadence_ticks) == 0:
            questions: tuple[Any, ...] = ()
            pending = None if host_decider is None else getattr(host_decider, "information_requests", None)
            if isinstance(pending, list) and len(pending) > information_consumed:
                questions += tuple(pending[information_consumed:])
                information_consumed = len(pending)
            asked = None if executive is None else getattr(executive, "information_requests", None)
            if isinstance(asked, list) and len(asked) > executive_questions_consumed:
                questions += tuple(asked[executive_questions_consumed:])
                executive_questions_consumed = len(asked)
            readings = None if host_decider is None else getattr(host_decider, "readings", None)
            scribe.record_tick(
                clock_tick, transition, new_state, information_requests=questions,
                readings=None if readings is None else tuple(readings),
            )

        # 6. a host decider's per-tick hook, after the sections of this tick are written: the lower layer's
        #    combiner runs here when its wrapper defines ``on_tick`` (skill_agent/runtime/scratchpad.py), so the
        #    status stream carries running statuses between decisions, not only at them. It may return a
        #    replacement carry.actor_state (same PyTree structure); None keeps the state, as that wrapper does
        if observes_ticks:
            actor_state = host_decider.on_tick(clock_tick, transition, new_state)
            if actor_state is not None:
                if jax.tree.structure(actor_state) != jax.tree.structure(new_state.carry.actor_state):
                    raise ValueError("on_tick changed the actor_state structure")
                new_state = new_state._replace(carry=new_state.carry._replace(actor_state=actor_state))
                changed = True

        # 7. the skill layer's state tick (opt-in), last: the decision's writes and this tick's statuses are in S_t
        if state_binding is not None:
            state_binding.on_tick(shared_state, clock_tick, decided, transition)

        return new_state if changed else None

    started = time.perf_counter()
    final_state, records = run_eager(
        built,
        core,
        record_fn,
        key,
        config.steps,
        initial_state=initial_state,
        host_hook=hook,
    )
    wall = time.perf_counter() - started

    metrics = SessionMetrics(
        ticks=clock_tick,
        virtual_seconds=clock_tick * config.tick_seconds,
        goal_switches_per_lane=tuple(int(x) for x in switches),
        goal_reports=len(reports),
        executive_versions_applied=executive_versions_applied,
        host=HostStats(
            decisions=host_decisions,
            misses=host_misses,
            errors=host_errors,
            latency_ms_p50=_percentile(host_latencies, 0.5),
            latency_ms_p95=_percentile(host_latencies, 0.95),
            held=host_held,
        ),
        wall_seconds=wall,
    )
    return SessionResult(final_state=final_state, records=records, metrics=metrics, reports=tuple(reports))


def summarize(metrics: SessionMetrics) -> dict[str, Any]:
    """Flat dict for CSV or JSON reporting."""

    switches = metrics.goal_switches_per_lane
    return {
        "ticks": metrics.ticks,
        "virtual_seconds": round(metrics.virtual_seconds, 3),
        "goal_switches_total": int(sum(switches)),
        "goal_switches_mean_per_lane": float(statistics.fmean(switches)) if switches else 0.0,
        "goal_reports": metrics.goal_reports,
        "executive_versions_applied": metrics.executive_versions_applied,
        "host_decisions": metrics.host.decisions,
        "host_misses": metrics.host.misses,
        "host_errors": metrics.host.errors,
        "host_latency_ms_p50": metrics.host.latency_ms_p50,
        "host_latency_ms_p95": metrics.host.latency_ms_p95,
        "wall_seconds": round(metrics.wall_seconds, 3),
    }


__all__ = [
    "TICK_SECONDS",
    "ExecutiveSeam",
    "ExecutiveWriter",
    "HostDecider",
    "HostStats",
    "ReportSink",
    "SessionConfig",
    "SessionMetrics",
    "SessionResult",
    "default_record",
    "run_session",
    "summarize",
]
