"""Virtual clock, call ledger and call triggers: the scheduler hooks (ADR-001 D3, seam 4).

:class:`VirtualClock` maps simulation ticks to seconds. :class:`CallLedger` records each
executive call with the four allocation knobs of ADR-001 D3: whether to call (the
trigger), the model tier, the effort level, and how much of the prompt was served from
cache. The triggers decide whether a call is due.

Formulas
--------
::

    seconds(tick)        = tick / ticks_per_second
    rank(p, n)           = ceil(p * n / 100) = (p * n + 99) // 100        (integers)
    nearest_rank(p)      = x_(rank(p, n)), over the ascending latencies x_(1) <= ... <= x_(n)
    prompt_tokens        = sum(input) + sum(cache_creation) + sum(cache_read)
    cache_read_fraction  = sum(cache_read) / prompt_tokens    (None when prompt_tokens == 0)

    due(tick, reports, last):
        FixedCadence(N):  last is None or tick - last >= N
        PerTransition:    some GoalReport is in reports
        EventTrigger(E):  some EventRecord in reports has event_name in E
        Never:            False

Citations: ADR-001 D3 (the four knobs are logged per call, so outcome per token is
measurable; the stable system prompt goes first so cached reads can be counted); ADR-001 P5
(baselines: fixed cadence and per-transition); H8 and H9 (event-driven and
value-of-reasoning triggers against fixed cadence, on outcome per call). Russell and Wefald
[R1] (DOI 10.1016/0004-3702(91)90015-C; see the References in ``__init__.py``) and Horvitz [L11]
(https://arxiv.org/abs/1304.2759) give the value of a computation, which a
value-of-reasoning trigger would use. That trigger is future work and is not implemented
here.

Formula sources: the nearest-rank definition is the secondary source
https://en.wikipedia.org/wiki/Percentile ("The nearest-rank method"; the page flags that
section as possibly original synthesis). ``cache_read_fraction`` follows the usage fields
of the Messages API (https://platform.claude.com/docs/en/build-with-claude/prompt-caching.md).
The seconds conversion and the trigger rules are design definitions from ADR-001 P5.
"""

from __future__ import annotations

import math
import threading
from typing import Any, NamedTuple, Protocol, Sequence

from experimental.neural_agent.contracts import EVENT_NAMES, GoalReport
from experimental.neural_agent.executive.reports import EventRecord


TICK_SECONDS: float = 0.045
"""Seconds per simulation tick (``loaded_dt``)."""

TICKS_PER_SECOND: float = 1.0 / TICK_SECONDS
"""Default clock rate; about 22.2 ticks per second, not 30."""

class VirtualClock:
    """Simulation tick counter that converts ticks to seconds. Thread-safe.

    Args:
        ticks_per_second: Simulation ticks per second, finite and positive. Default
            ``TICKS_PER_SECOND`` = 1 / 0.045, the gym's ``loaded_dt``; the ADK comments
            that say 30 ticks per second are stale (ADR-001 D3, memory horizon-ticks-to-seconds).

    Raises:
        TypeError: If ``ticks_per_second`` is not a real number.
        ValueError: If it is not finite and positive.
    """

    def __init__(self, ticks_per_second: float = TICKS_PER_SECOND) -> None:
        if isinstance(ticks_per_second, bool) or not isinstance(ticks_per_second, (int, float)):
            raise TypeError("ticks_per_second must be a real number")
        rate = float(ticks_per_second)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError(f"ticks_per_second must be finite and positive, got {rate}")
        self._rate = rate
        self._tick = 0
        self._lock = threading.Lock()

    @property
    def ticks_per_second(self) -> float:
        """The simulation rate."""

        return self._rate

    @property
    def tick(self) -> int:
        """The current tick, a non-negative int."""

        with self._lock:
            return self._tick

    def advance(self, n: int) -> int:
        """Advance the clock by ``n`` ticks and return the new tick.

        Raises:
            TypeError: If ``n`` is not an integer.
            ValueError: If ``n`` is negative.
        """

        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(f"n must be an integer, got {type(n).__name__}")
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        with self._lock:
            self._tick += n
            return self._tick

    def seconds(self) -> float:
        """Return elapsed simulation seconds: ``tick / ticks_per_second``."""

        return self.tick / self._rate


class CallRecord(NamedTuple):
    """One executive call, as logged by the harness.

    Attributes:
        tick: Simulation tick of the call.
        trigger: Class name of the trigger that made the call due.
        model: Requested model id, or "none" for a non-LLM baseline.
        effort: Requested effort level, or "none".
        input_tokens: Uncached input tokens.
        output_tokens: Output tokens.
        cache_read_tokens: Input tokens served from the prompt cache.
        cache_creation_tokens: Input tokens written to the prompt cache.
        latency_ms: Wall-clock milliseconds of the backend call.
        accepted: True when the store accepted the proposal.
    """

    tick: int
    trigger: str
    model: str
    effort: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    latency_ms: float
    accepted: bool


def _count(value: Any, name: str) -> int:
    """Return ``value`` as an int that is zero or greater. Bools are rejected."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def nearest_rank(sorted_values: Sequence[float], percent: int) -> float | None:
    """Return the nearest-rank percentile of ascending ``sorted_values``.

    The rank is ``ceil(percent * n / 100)``, computed in integers, and the result is
    the observed value at that rank. No interpolation is done, so a reported latency
    is a latency that really happened.

    Args:
        sorted_values: Values sorted in ascending order.
        percent: An integer from 1 to 100.

    Returns:
        The percentile value, or None when ``sorted_values`` is empty.

    Raises:
        ValueError: If ``percent`` is outside 1 to 100.
    """

    if isinstance(percent, bool) or not isinstance(percent, int) or not 1 <= percent <= 100:
        raise ValueError(f"percent must be an integer from 1 to 100, got {percent!r}")
    count = len(sorted_values)
    if count == 0:
        return None
    rank = (percent * count + 99) // 100
    return sorted_values[rank - 1]


class CallLedger:
    """Append-only, thread-safe record of executive calls, with a summary.

    Use :meth:`summary` for the aggregate formulas in this module's docstring.
    """

    def __init__(self) -> None:
        self._rows: list[CallRecord] = []
        self._lock = threading.Lock()

    def record(self, row: CallRecord) -> None:
        """Append one call.

        Raises:
            TypeError: If ``row`` is not a CallRecord, or a field has the wrong type.
            ValueError: If a count is negative, a text field is empty, or the latency is
                negative or not finite.
        """

        if not isinstance(row, CallRecord):
            raise TypeError(f"row must be a CallRecord, got {type(row).__name__}")
        _count(row.tick, "CallRecord.tick")
        for name in ("trigger", "model", "effort"):
            value = getattr(row, name)
            if not isinstance(value, str):
                raise TypeError(f"CallRecord.{name} must be a string")
            if not value:
                raise ValueError(f"CallRecord.{name} must be a non-empty string")
        for name in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens"):
            _count(getattr(row, name), f"CallRecord.{name}")
        if isinstance(row.latency_ms, bool) or not isinstance(row.latency_ms, (int, float)):
            raise TypeError("CallRecord.latency_ms must be a real number")
        if not math.isfinite(row.latency_ms) or row.latency_ms < 0:
            raise ValueError(f"CallRecord.latency_ms must be finite and non-negative, got {row.latency_ms}")
        if not isinstance(row.accepted, bool):
            raise TypeError("CallRecord.accepted must be a bool")
        with self._lock:
            self._rows.append(row)

    def rows(self) -> tuple[CallRecord, ...]:
        """Return every recorded call, in order."""

        with self._lock:
            return tuple(self._rows)

    def summary(self) -> dict[str, Any]:
        """Return the aggregate of all recorded calls.

        Returns:
            A dict with ``calls`` (int), ``accepted`` (int), ``tokens`` (a dict with the
            sums ``input``, ``output``, ``cache_read``, ``cache_creation`` and ``prompt``,
            where ``prompt = input + cache_creation + cache_read``), ``cache_read_fraction``
            (``cache_read / prompt``, or None when prompt is 0), and ``latency_ms_p50`` and
            ``latency_ms_p95`` (nearest-rank percentiles, or None when there are no calls).
        """

        rows = self.rows()
        input_total = sum(row.input_tokens for row in rows)
        output_total = sum(row.output_tokens for row in rows)
        read_total = sum(row.cache_read_tokens for row in rows)
        creation_total = sum(row.cache_creation_tokens for row in rows)
        prompt_total = input_total + creation_total + read_total
        latencies = sorted(row.latency_ms for row in rows)
        return {
            "calls": len(rows),
            "accepted": sum(1 for row in rows if row.accepted),
            "tokens": {
                "input": input_total,
                "output": output_total,
                "cache_read": read_total,
                "cache_creation": creation_total,
                "prompt": prompt_total,
            },
            "cache_read_fraction": (read_total / prompt_total) if prompt_total else None,
            "latency_ms_p50": nearest_rank(latencies, 50),
            "latency_ms_p95": nearest_rank(latencies, 95),
        }


class ExecutiveTrigger(Protocol):
    """Decides whether the executive should be called at this tick."""

    def due(
        self,
        tick: int,
        reports: Sequence[GoalReport | EventRecord],
        last_call_tick: int | None,
    ) -> bool:
        """Return True when a call should be made now.

        Args:
            tick: The current simulation tick.
            reports: The queued reports, without removing them (ReportChannel.peek).
            last_call_tick: The tick of the previous call, or None before the first call.
        """


class FixedCadenceTrigger:
    """Call every ``every_ticks`` ticks, and at the first opportunity.

    Formula: ``due = last is None or tick - last >= every_ticks``. This is ADR-001 P5
    baseline 1. The reports are not consulted.

    Args:
        every_ticks: Positive int, the cadence in ticks.

    Raises:
        TypeError: If ``every_ticks`` is not an integer.
        ValueError: If ``every_ticks`` is not positive.
    """

    def __init__(self, every_ticks: int) -> None:
        if isinstance(every_ticks, bool) or not isinstance(every_ticks, int):
            raise TypeError("every_ticks must be an integer")
        if every_ticks < 1:
            raise ValueError(f"every_ticks must be positive, got {every_ticks}")
        self._every_ticks = every_ticks

    def due(
        self,
        tick: int,
        reports: Sequence[GoalReport | EventRecord],
        last_call_tick: int | None,
    ) -> bool:
        """Return True on the first call, then once ``every_ticks`` have passed."""

        if last_call_tick is None:
            return True
        return tick - last_call_tick >= self._every_ticks


class PerTransitionTrigger:
    """Call whenever a goal outcome report is queued. ADR-001 P5 baseline 2.

    Formula: ``due = any(isinstance(item, GoalReport) for item in reports)``. An
    interrupt event alone does not make the call due.
    """

    def due(
        self,
        tick: int,
        reports: Sequence[GoalReport | EventRecord],
        last_call_tick: int | None,
    ) -> bool:
        """Return True when at least one GoalReport is queued."""

        return any(isinstance(item, GoalReport) for item in reports)


class EventTrigger:
    """Call when a named interrupt event is queued.

    Formula: ``due = any(isinstance(item, EventRecord) and item.event_name in E
    for item in reports)``. A goal report alone does not make the call due.

    Args:
        event_names: Non-empty collection of names from ``EVENT_NAMES``.

    Raises:
        TypeError: If ``event_names`` is a bare string.
        ValueError: If it is empty or names an unknown event.
    """

    def __init__(self, event_names: Sequence[str]) -> None:
        if isinstance(event_names, str):
            raise TypeError("event_names must be a collection of names, not one string")
        names = tuple(event_names)
        if not names:
            raise ValueError("event_names must name at least one event")
        unknown = [name for name in names if name not in EVENT_NAMES]
        if unknown:
            raise ValueError(f"unknown event name(s) {unknown}; expected names from {EVENT_NAMES}")
        self._names = frozenset(names)

    def due(
        self,
        tick: int,
        reports: Sequence[GoalReport | EventRecord],
        last_call_tick: int | None,
    ) -> bool:
        """Return True when a queued EventRecord names one of the configured events."""

        return any(
            isinstance(item, EventRecord) and item.event_name in self._names for item in reports
        )


class NeverTrigger:
    """Never call. Used with a fixed offline plan, which is set before the run. ADR-001 P5 baseline 3."""

    def due(
        self,
        tick: int,
        reports: Sequence[GoalReport | EventRecord],
        last_call_tick: int | None,
    ) -> bool:
        """Return False."""

        return False
