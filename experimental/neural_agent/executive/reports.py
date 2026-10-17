"""Upward report channel: goal outcomes and interrupt events from Levels 1 and 2 (ADR-001 section 5, seam 3).

:class:`ReportChannel` is a bounded FIFO. Pushes never block the simulation. When
the queue is full, the oldest item is dropped and the drop is counted.

Formulas
--------
With capacity ``C``, a push that finds ``len(queue) == C`` removes the oldest item
and adds one to ``dropped``. Items are drained in arrival order, so the drained
list is the last ``min(n, C)`` of the ``n`` pushes since the previous drain.

Citations: ADR-001 section 5, seam 3 ("the upward event and report channel"); ADR-001
section 2 (a goal outcome is reported on exit, and Level 1 never waits for Level 3).
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, NamedTuple

from experimental.neural_agent.contracts import EVENT_NAMES, GoalReport


class EventRecord(NamedTuple):
    """An interrupt event reported upward, named by its DecisionEvent member.

    Attributes:
        tick: Non-negative simulation tick at which the event was seen.
        lane: Non-negative lane index in the batch.
        event_name: One of ``EVENT_NAMES``.
    """

    tick: int
    lane: int
    event_name: str


def _count(value: Any, name: str) -> int:
    """Return ``value`` as an int that is zero or greater. Bools are rejected."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


class ReportChannel:
    """Bounded, thread-safe FIFO of GoalReport and EventRecord items.

    Args:
        capacity: Maximum number of queued items, a positive int. Default 256.

    Raises:
        TypeError: If ``capacity`` is not an integer.
        ValueError: If ``capacity`` is not positive.
    """

    def __init__(self, capacity: int = 256) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError(f"capacity must be an integer, got {type(capacity).__name__}")
        if capacity < 1:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = capacity
        self._items: deque[GoalReport | EventRecord] = deque()
        self._dropped = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        """The maximum number of queued items."""

        return self._capacity

    @property
    def dropped(self) -> int:
        """Items dropped because the queue was full, counted since construction."""

        with self._lock:
            return self._dropped

    def push_goal_report(self, report: GoalReport) -> None:
        """Queue a goal outcome report.

        Raises:
            TypeError: If ``report`` is not a GoalReport.
        """

        if not isinstance(report, GoalReport):
            raise TypeError(f"report must be a GoalReport, got {type(report).__name__}")
        self._push(report)

    def push_event(self, tick: int, lane: int, event_name: str) -> None:
        """Queue an interrupt event.

        Args:
            tick: Non-negative simulation tick.
            lane: Non-negative lane index.
            event_name: One of ``EVENT_NAMES``.

        Raises:
            TypeError: If ``tick`` or ``lane`` is not an integer, or ``event_name`` is not a string.
            ValueError: If ``tick`` or ``lane`` is negative, or ``event_name`` is unknown.
        """

        tick = _count(tick, "tick")
        lane = _count(lane, "lane")
        if not isinstance(event_name, str):
            raise TypeError(f"event_name must be a string, got {type(event_name).__name__}")
        if event_name not in EVENT_NAMES:
            raise ValueError(f"unknown event {event_name!r}; expected one of {EVENT_NAMES}")
        self._push(EventRecord(tick=tick, lane=lane, event_name=event_name))

    def drain(self) -> list[GoalReport | EventRecord]:
        """Return every queued item, oldest first, and empty the queue."""

        with self._lock:
            items = list(self._items)
            self._items.clear()
        return items

    def peek(self) -> tuple[GoalReport | EventRecord, ...]:
        """Return a snapshot of the queued items without removing them."""

        with self._lock:
            return tuple(self._items)

    def _push(self, item: GoalReport | EventRecord) -> None:
        """Append ``item``, dropping and counting the oldest item when full."""

        with self._lock:
            if len(self._items) == self._capacity:
                self._items.popleft()
                self._dropped += 1
            self._items.append(item)
