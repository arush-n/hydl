"""ExecutiveHarness: call the executive on a trigger, time it, log it, and store the result (ADR-001 D3, seams 3 and 4).

:meth:`ExecutiveHarness.step` is synchronous and blocks on the backend. That suits
deterministic baselines and replays. :meth:`ExecutiveHarness.start_async` runs the same
logic on a worker thread at a wall-clock interval. While the worker runs, :meth:`step`
returns at once, and the caller reads :meth:`latest`. The store publishes a version only
after the backend has returned a whole state. A slow or failing backend therefore leaves
the state in force unchanged, and Level 1 and Level 2 never wait for it.

Formulas
--------
``latency_ms = 1000 * (t_return - t_call)``, from ``time.perf_counter()`` around the
backend call only. Every call is logged once. It is accepted if the store took the
proposal, and rejected otherwise. ``last_call_tick`` moves on every call, including a
failed one, so the cadence counts real calls.

Known limit: reports are drained before the backend call. If the backend raises, those
reports are not put back, so the executive does not see them on its next call.

Citations: ADR-001 section 2 ("Level 1 never blocks"; the JEV answers within its budget or
not at all); ADR-001 D3 (asynchronous, event-driven, persistent; the interface commitment
that Level 3 never runs inside the bridge tick); ADR-001 section 5, seams 3 and 4. The
latency source is Python's time.perf_counter
(https://docs.python.org/3/library/time.html#time.perf_counter).
"""

from __future__ import annotations

import dataclasses

import math
import threading
import time
from typing import Any, Callable

from experimental.neural_agent.contracts import ExecutiveState
from experimental.neural_agent.executive.baselines import (
    NO_USAGE,
    CallUsage,
    ExecutiveBackend,
    UsageReportingBackend,
)
from experimental.neural_agent.executive.reports import ReportChannel
from experimental.neural_agent.executive.scheduler import (
    CallLedger,
    CallRecord,
    ExecutiveTrigger,
    VirtualClock,
)
from experimental.neural_agent.executive.store import ExecutiveStateStore
from experimental.neural_agent.shared_state import SECTION_WRITERS, ExecutiveSection, SharedState


def _elapsed_ms(started: float) -> float:
    """Return the milliseconds since ``started``, a ``time.perf_counter()`` reading."""

    return (time.perf_counter() - started) * 1000.0


class ExecutiveHarness:
    """Connects a backend, the E_t store, the report channel, the ledger, a trigger and a clock.

    Args:
        backend: An ExecutiveBackend. If it also satisfies UsageReportingBackend, its
            usage is logged; otherwise the call is logged with no token counts.
        store: The ExecutiveStateStore that holds E_t.
        channel: The ReportChannel that the lanes push into.
        ledger: The CallLedger that records each call.
        trigger: An ExecutiveTrigger that decides when a call is due.
        clock: The VirtualClock whose tick the asynchronous worker reads.
        shared_state: The scratchpad S_t (``shared_state.py``), or None. When given, every
            accepted proposal is also written into its ``executive`` section as the LLM's writer.
    """

    def __init__(
        self,
        backend: ExecutiveBackend,
        store: ExecutiveStateStore,
        channel: ReportChannel,
        ledger: CallLedger,
        trigger: ExecutiveTrigger,
        clock: VirtualClock,
        shared_state: SharedState | None = None,
        resolver: Callable[[str, str], Any] | None = None,
    ) -> None:
        self._backend = backend
        self._resolver = resolver
        self.information_requests: list[dict[str, Any]] = []  # questions to the LLM; the session's scribe collects them
        self._store = store
        self._channel = channel
        self._ledger = ledger
        self._trigger = trigger
        self._clock = clock
        self._shared_state = shared_state
        if shared_state is not None and hasattr(backend, "shared_state"):
            backend.shared_state = shared_state  # the backend's user turn then carries the briefing
        self._call_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_call_tick: int | None = None
        self._last_error: str | None = None

    def _serve_unknown_skills(self, state: ExecutiveState, tick: int) -> ExecutiveState:
        """Serve each goal whose skill id is not in the catalogue with the nearest catalogue id, or leave it out of
        this version, and record a question to the LLM either way (SYSTEM_ONE 9.16).

        Without a resolver the state is returned as it came and the store refuses an unknown id as before. When
        the backend keeps the LLM's words (``last_section`` matching ``state``), the kept goals keep their words.
        """

        if self._resolver is None:
            return state
        catalogue = frozenset(self._store.skill_ids)
        section = getattr(self._backend, "last_section", None)
        worded = isinstance(section, ExecutiveSection) and section.to_executive_state() == state
        goals, entries = [], []
        for slot, goal in enumerate(state.goals):
            words = section.goals[slot].intent if worded else ""
            if goal.skill_id not in catalogue:
                resolution = self._resolver(goal.skill_id, words)
                self.information_requests.append({
                    "version": state.version, "slot": slot, "when": words, "requested": resolution.requested,
                    "skill_id": resolution.skill_id, "source": resolution.source, "confidence": resolution.confidence,
                    "candidates": list(resolution.candidates), "reason": resolution.reason,
                })
                if resolution.skill_id is None:
                    continue
                goal = dataclasses.replace(goal, skill_id=resolution.skill_id)
            goals.append(goal)
            if worded:
                entries.append(dataclasses.replace(section.goals[slot], skill_id=goal.skill_id))
        if tuple(goals) == state.goals:
            return state
        if worded:
            self._backend.last_section = dataclasses.replace(section, goals=tuple(entries))
        return dataclasses.replace(state, goals=tuple(goals))

    def _section_for(self, state: ExecutiveState) -> ExecutiveSection:
        """The executive section to write for ``state``: the backend's parsed section when it has one for
        this state (words included), else the typed fields alone."""

        section = getattr(self._backend, "last_section", None)
        if isinstance(section, ExecutiveSection) and section.to_executive_state() == state:
            return section
        return ExecutiveSection.from_executive_state(state)

    @property
    def last_error(self) -> str | None:
        """The repr of the last backend exception, or the message of the last rejected proposal.

        None after a clean call.
        """

        return self._last_error

    def latest(self) -> ExecutiveState:
        """Return the state in force. This never waits for the backend."""

        return self._store.current()

    def step(self, tick: int) -> ExecutiveState | None:
        """Make one synchronous step at ``tick``.

        If the trigger is due, the reports are drained, the backend is called and timed,
        the call is logged, and the proposal is offered to the store.

        Args:
            tick: The current simulation tick, a non-negative int.

        Returns:
            The accepted state, or None when no call was due, when the proposal was
            rejected, or while the asynchronous worker owns the backend (then step does
            nothing and returns at once).

        Raises:
            Exception: Whatever the backend raises. The failed call is logged with
                ``accepted`` False, and ``last_error`` is set, before the error is re-raised.
        """

        if self._worker is not None:
            return None
        return self._run(tick)

    def start_async(self, interval_seconds: float = 0.5) -> None:
        """Start the worker thread. It returns at once.

        The worker calls the same logic as :meth:`step` with ``clock.tick`` every
        ``interval_seconds`` of wall-clock time. Errors are recorded and do not stop the worker.

        Args:
            interval_seconds: Wall-clock seconds between checks, finite and positive.

        Raises:
            TypeError: If ``interval_seconds`` is not a real number.
            ValueError: If it is not finite and positive.
            RuntimeError: If the worker is already running.
        """

        if isinstance(interval_seconds, bool) or not isinstance(interval_seconds, (int, float)):
            raise TypeError("interval_seconds must be a real number")
        interval = float(interval_seconds)
        if not math.isfinite(interval) or interval <= 0.0:
            raise ValueError(f"interval_seconds must be finite and positive, got {interval}")
        if self._worker is not None:
            raise RuntimeError("the executive worker is already running")
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._loop, args=(interval,), name="executive-harness", daemon=True
        )
        self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the worker and wait up to ``timeout`` seconds. A second call does nothing.

        Raises:
            RuntimeError: If the worker is still inside a backend call after ``timeout``.
                The worker stays registered, so a later ``stop`` can retry.
        """

        worker = self._worker
        if worker is None:
            return
        self._stop_event.set()
        worker.join(timeout)
        if worker.is_alive():
            raise RuntimeError("the executive worker did not stop within the timeout")
        self._worker = None

    def _loop(self, interval: float) -> None:
        """Worker body: run the due check on ``clock.tick`` until stopped."""

        while not self._stop_event.is_set():
            try:
                self._run(self._clock.tick)
            except Exception:
                # The failure is in last_error and in the ledger. The state in force stays.
                pass
            self._stop_event.wait(interval)

    def _run(self, tick: int) -> ExecutiveState | None:
        """Run one call if due, under the call lock. Return the accepted state or None."""

        if not self._call_lock.acquire(blocking=False):
            return None
        try:
            if not self._trigger.due(tick, self._channel.peek(), self._last_call_tick):
                return None
            reports = self._channel.drain()
            current = self._store.current()
            self._last_call_tick = tick
            started = time.perf_counter()
            try:
                state, usage = self._call(current, reports, tick)
            except Exception as error:
                self._record(tick, self._failed_usage(), _elapsed_ms(started), accepted=False)
                self._last_error = repr(error)
                raise
            latency_ms = _elapsed_ms(started)
            state = self._serve_unknown_skills(state, tick)
            try:
                self._store.propose(state)
            except (TypeError, ValueError) as error:
                self._record(tick, usage, latency_ms, accepted=False)
                self._last_error = str(error)
                return None
            if self._shared_state is not None:
                self._shared_state.write(
                    "executive",
                    self._section_for(state),
                    writer=SECTION_WRITERS["executive"],
                    tick=tick,
                )
            self._record(tick, usage, latency_ms, accepted=True)
            self._last_error = None
            return state
        finally:
            self._call_lock.release()

    def _call(
        self, current: ExecutiveState, reports: list[Any], tick: int
    ) -> tuple[ExecutiveState, CallUsage]:
        """Call the backend and return ``(state, usage)``. Baselines report NO_USAGE."""

        if isinstance(self._backend, UsageReportingBackend):
            return self._backend.propose_with_usage(current, reports, tick)
        return self._backend.propose(current, reports, tick), NO_USAGE

    def _failed_usage(self) -> CallUsage:
        """Return the requested knobs for a call that raised, with zero token counts."""

        if isinstance(self._backend, UsageReportingBackend):
            return CallUsage(model=self._backend.model, effort=self._backend.effort)
        return NO_USAGE

    def _record(self, tick: int, usage: CallUsage, latency_ms: float, *, accepted: bool) -> None:
        """Append one row to the ledger for this call."""

        self._ledger.record(
            CallRecord(
                tick=tick,
                trigger=type(self._trigger).__name__,
                model=usage.model,
                effort=usage.effort,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                cache_creation_tokens=usage.cache_creation_tokens,
                latency_ms=latency_ms,
                accepted=accepted,
            )
        )
