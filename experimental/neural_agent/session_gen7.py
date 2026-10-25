"""The gen-7 host session: the three-level stack's per-tick loop, with the executive seam, goal reports and timing.

:mod:`experimental.neural_agent.session` runs the gen-5-era ADK loop, whose host decider cannot drive the skill layer's
host bodies. This module is the loop the gen-7 architecture test (``eval/run_architecture.py``) proved on the trained
bodies, promoted into a reusable runner. It carries the parts of ``session.py`` that do not depend on the ADK core:
the executive seam (:class:`~experimental.neural_agent.session.ExecutiveSeam`), the report sink and the scribe
(:class:`~experimental.neural_agent.session.ReportSink`), the host statistics (:class:`HostStats`) and the tick length
(:data:`~experimental.neural_agent.session.TICK_SECONDS`).

One host tick ``t`` for B lanes, in this order (``run_gen7_session``):

1. state tick: ``binding.on_tick(document, t - 1, False, transition_of(facts))``. The binding and its observer are the
   caller's; the statuses of the combiner's last tick reach the skill layer here.
2. executive: ``executive.step(t)``; a new :class:`ExecutiveState` becomes the latest and is counted as applied.
3. decision layer, on its cadence (``t % decide_every_ticks == 0``): ``decider.decide(t, document, facts, latest)``
   writes or withdraws requests in S_t. It is timed; over the budget, or raising, counts a miss (raising also an
   error). The S_t requests stand either way: a raise never aborts the run.
4. bodies: ``contexts(t, view, facts)`` gives each agent its context. With ``config.batch_bodies`` and bodies that
   carry a ``batch_results`` dict, the dict of each verb is refilled from ``state_tick.run_module_batches`` over the
   live requests in S_t (two live requests of one verb for one agent raise). Then ``combine_from_scratchpad`` grants,
   runs the bodies and writes the statuses. The action rows are each agent's ``factors[:head_count]``.
5. goal reports: a status that ends a request becomes a :class:`GoalReport`, pushed to ``report_sink`` and to
   ``scribe.push_goal_report`` when given. Outcome mapping (no timeout kind exists in ``requests.STATUS_KINDS``):
   ``done`` -> COMPLETED; ``aborted`` (withdrawn, expired, interrupted) -> INTERRUPTED; ``rejected`` -> FAILED.
   ``duration_ticks`` counts from the status ``accepted`` (the grant), or from the request's issue tick if no grant was
   seen; ``executive_version`` is the latest applied version (0 before the first).
6. world: ``facts = world.step(actions)``; then ``record(t, facts, steps)`` when given, with the facts this step
   produced and the tick's combined steps. Its returns are collected in order.

Facts are opaque to the runner. The world gives them (``reset()`` for tick 0, ``step(actions)`` after each tick); the
caller's ``transition_of``, ``contexts``, ``decider.decide`` and ``record`` are the only readers.

Timing: ``metrics.host_ms_per_tick`` gives the milliseconds per tick of each phase (``state_tick``, ``executive``,
``decision``, ``bodies``, ``world``) and their ``total``. The world's record hook is not timed.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple, Protocol, runtime_checkable

import jax
import numpy as np

from experimental.neural_agent.contracts import ExecutiveState, GoalReport, SkillOutcome
from experimental.neural_agent.requests import ActionRequest
from experimental.neural_agent.session import (
    TICK_SECONDS,
    ExecutiveSeam,
    HostStats,
    ReportSink,
    _percentile,
)
from experimental.neural_agent.shared_state import SharedState
from experimental.neural_agent.skill_agent.runtime import state_tick
from experimental.neural_agent.skill_agent.runtime.combiner import AgentContext, CombinedStep, Combiner
from experimental.neural_agent.skill_agent.runtime.scratchpad import combine_from_scratchpad

PHASES: tuple[str, ...] = ("state_tick", "executive", "decision", "bodies", "world")
"""The timed phases of one tick, in order."""

_OUTCOME_OF_KIND: dict[str, SkillOutcome] = {
    "done": SkillOutcome.COMPLETED,
    "aborted": SkillOutcome.INTERRUPTED,
    "rejected": SkillOutcome.FAILED,
}
"""The status kinds that end a request, as goal outcomes (requests.STATUS_KINDS has no timeout kind)."""


@dataclasses.dataclass(frozen=True, slots=True)
class Gen7SessionConfig:
    """Timing knobs for one gen-7 session."""

    ticks: int
    decide_every_ticks: int = 1
    decide_budget_ms: float = 20.0
    batch_bodies: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.ticks, bool) or not isinstance(self.ticks, int) or self.ticks < 1:
            raise ValueError("Gen7SessionConfig.ticks must be a positive int")
        if isinstance(self.decide_every_ticks, bool) or not isinstance(self.decide_every_ticks, int) or (
            self.decide_every_ticks < 1
        ):
            raise ValueError("Gen7SessionConfig.decide_every_ticks must be a positive int")
        if not (math.isfinite(self.decide_budget_ms) and self.decide_budget_ms > 0):
            raise ValueError("Gen7SessionConfig.decide_budget_ms must be positive")


@runtime_checkable
class Gen7World(Protocol):
    """The environment. ``reset()`` gives tick 0's facts; ``step(actions)`` gives the facts after the tick's actions."""

    def reset(self) -> Any: ...

    def step(self, actions: np.ndarray, /) -> Any: ...


@runtime_checkable
class Gen7Decider(Protocol):
    """The decision layer. Writes or withdraws the requests of S_t for ``tick``; ``executive`` is the latest E_t."""

    def decide(
        self, tick: int, document: SharedState, facts: Any, executive: ExecutiveState | None, /
    ) -> None: ...


class Gen7SessionMetrics(NamedTuple):
    ticks: int
    virtual_seconds: float
    goal_reports: int
    executive_versions_applied: int
    host: HostStats
    status_counts: Mapping[str, int]
    host_ms_per_tick: Mapping[str, float]
    """Per phase in :data:`PHASES`, then ``total``: milliseconds per tick."""
    wall_seconds: float


class Gen7SessionResult(NamedTuple):
    final_facts: Any
    records: list[Any]
    metrics: Gen7SessionMetrics
    reports: tuple[GoalReport, ...]


def _request_verb(request: ActionRequest) -> str:
    return request.verb


class _Ledger:
    """The requests seen live, and the grant tick of each, so a status can be made into a goal report."""

    def __init__(self, verb_of: Callable[[ActionRequest], str]) -> None:
        self.verb_of = verb_of
        self.live: dict[tuple[int, int], ActionRequest] = {}
        self.granted: dict[tuple[int, int], int] = {}

    def see(self, document: SharedState) -> None:
        for request in document.requests.live:
            self.live[(int(request.agent), int(request.request_id))] = request

    def reports(self, tick: int, steps: Mapping[int, CombinedStep], executive_version: int) -> list[GoalReport]:
        out: list[GoalReport] = []
        for agent in sorted(steps):
            for status in steps[agent].statuses:
                key = (int(status.agent), int(status.request_id))
                if status.kind == "accepted":
                    self.granted[key] = int(status.tick)
                outcome = _OUTCOME_OF_KIND.get(status.kind)
                if outcome is None:
                    continue
                request = self.live.pop(key, None)
                started = self.granted.pop(key, None)
                if started is None:
                    started = request.issued_tick if request is not None else tick
                out.append(GoalReport(
                    tick=tick,
                    lane=int(status.agent),
                    skill_id=self.verb_of(request) if request is not None else "unknown",
                    executive_version=executive_version,
                    outcome=outcome,
                    duration_ticks=int(tick - started),
                    detail=str(status.reason or ""),
                ))
        return out


def run_gen7_session(
    world: Gen7World,
    decider: Gen7Decider,
    combiner: Combiner,
    bodies_by_verb: Mapping[str, Any],
    binding: Any,
    config: Gen7SessionConfig,
    *,
    contexts: Callable[[int, Any, Any], Mapping[int, AgentContext]],
    transition_of: Callable[[Any], Any],
    head_count: int,
    document: SharedState | None = None,
    executive: ExecutiveSeam | None = None,
    report_sink: ReportSink | None = None,
    scribe: Any = None,
    verb_of: Callable[[ActionRequest], str] = _request_verb,
    record: Callable[[int, Any, Mapping[int, CombinedStep]], Any] | None = None,
) -> Gen7SessionResult:
    """Run ``config.ticks`` ticks of the gen-7 stack (the module docstring gives the order and the rules).

    ``binding`` is a state-tick binding with ``on_tick(document, tick, decided, transition)``, ``view`` (None before the
    first tick) and ``tick_seconds`` (``shared_state.skill_writes.StateTickBinding``). ``transition_of(facts)`` gives
    the binding's transition for a tick's facts. ``bodies_by_verb`` maps each verb to its body; a body whose
    ``batch_results`` is a dict is batched (``ModuleBody``). ``verb_of`` names a request's verb (the body's key).
    ``head_count`` is how many leading factors of each agent's step are the action row.
    """

    if document is None:
        document = SharedState()
    batched = {verb: body for verb, body in bodies_by_verb.items() if getattr(body, "batch_results", None) is not None}
    clock = dict.fromkeys(PHASES, 0.0)
    ledger = _Ledger(verb_of)
    status_counts: Counter = Counter()
    reports: list[GoalReport] = []
    records: list[Any] = []
    latencies: list[float] = []
    latest: ExecutiveState | None = None
    versions = decisions = misses = errors = ticks_run = 0
    started_wall = time.perf_counter()

    facts = world.reset()
    for tick in range(config.ticks):
        started = time.perf_counter()
        binding.on_tick(document, tick - 1, False, transition_of(facts))
        if binding.view is not None:
            jax.block_until_ready(binding.view.encoded)
        clock["state_tick"] += time.perf_counter() - started

        started = time.perf_counter()
        if executive is not None:
            proposal = executive.step(tick)
            if proposal is not None:
                if not isinstance(proposal, ExecutiveState):
                    raise TypeError("executive.step must return an ExecutiveState or None")
                latest = proposal
                versions += 1
        clock["executive"] += time.perf_counter() - started

        if tick % config.decide_every_ticks == 0:
            started = time.perf_counter()
            failed = False
            decisions += 1
            try:
                decider.decide(tick, document, facts, latest)
            except Exception:  # noqa: BLE001 - a decision failure keeps the S_t requests, by contract
                failed = True
                errors += 1
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(elapsed_ms)
            if failed or elapsed_ms > config.decide_budget_ms:
                misses += 1
            ledger.see(document)
            clock["decision"] += time.perf_counter() - started

        started = time.perf_counter()
        view = binding.view
        ctx = contexts(tick, view, facts)
        agents = sorted(ctx)
        if config.batch_bodies and batched:
            live: dict[int, dict[str, Any]] = {}
            for agent in agents:
                for request in document.requests.for_agent(agent):
                    verb = verb_of(request)
                    if verb in live.setdefault(agent, {}):
                        raise RuntimeError(f"agent {agent} has two live {verb} requests; a batch holds one")
                    live[agent][verb] = request
            results = state_tick.run_module_batches(view, ctx, batched, live)
            for verb, body in batched.items():
                body.batch_results.clear()
                body.batch_results.update(results.get(verb, {}))
        steps = combine_from_scratchpad(document, combiner, tick, ctx)
        for step in steps.values():
            status_counts.update(status.kind for status in step.statuses)
        actions = np.stack([np.asarray(steps[agent].factors[:head_count], np.int32) for agent in agents])
        clock["bodies"] += time.perf_counter() - started

        reports_now = ledger.reports(tick, steps, 0 if latest is None else latest.version)
        for report in reports_now:
            reports.append(report)
            if report_sink is not None:
                report_sink.push_goal_report(report)
            if scribe is not None:
                scribe.push_goal_report(report)

        started = time.perf_counter()
        facts = world.step(actions)
        clock["world"] += time.perf_counter() - started
        ticks_run += 1
        if record is not None:
            records.append(record(tick, facts, steps))

    total_ms = 1000.0 * sum(clock.values()) / ticks_run
    metrics = Gen7SessionMetrics(
        ticks=ticks_run,
        virtual_seconds=ticks_run * binding.tick_seconds,
        goal_reports=len(reports),
        executive_versions_applied=versions,
        host=HostStats(
            decisions=decisions,
            misses=misses,
            errors=errors,
            latency_ms_p50=_percentile(latencies, 0.5),
            latency_ms_p95=_percentile(latencies, 0.95),
        ),
        status_counts=dict(status_counts),
        host_ms_per_tick={**{name: round(1000.0 * value / ticks_run, 2) for name, value in clock.items()},
                          "total": round(total_ms, 2)},
        wall_seconds=time.perf_counter() - started_wall,
    )
    return Gen7SessionResult(final_facts=facts, records=records, metrics=metrics, reports=tuple(reports))


__all__ = [
    "PHASES",
    "TICK_SECONDS",
    "Gen7Decider",
    "Gen7SessionConfig",
    "Gen7SessionMetrics",
    "Gen7SessionResult",
    "Gen7World",
    "run_gen7_session",
]
