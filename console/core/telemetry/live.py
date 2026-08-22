"""In-process push stream for console jobs and metrics."""

from __future__ import annotations

from collections import deque
import threading
import time
from typing import Any


_condition = threading.Condition()
_events: deque[dict[str, Any]] = deque(maxlen=2_048)
_sequence = 0


def publish(kind: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Append one timestamped event and wake every SSE subscriber."""

    if not kind:
        raise ValueError("live event kind must be nonempty")
    global _sequence
    with _condition:
        event = {
            "sequence": _sequence,
            "kind": kind,
            "unix_seconds": time.time(),
            "payload": payload or {},
        }
        _sequence += 1
        _events.append(event)
        _condition.notify_all()
        return event


def wait(after: int, timeout: float = 15.0) -> list[dict[str, Any]]:
    """Wait for events newer than ``after``; an empty result is a heartbeat."""

    with _condition:
        ready = [event for event in _events if event["sequence"] > after]
        if not ready:
            _condition.wait(timeout)
            ready = [event for event in _events if event["sequence"] > after]
        return ready


def snapshot(after: int = -1) -> dict[str, Any]:
    with _condition:
        events = [event for event in _events if event["sequence"] > after]
        return {
            "last_sequence": _sequence - 1,
            "events": events,
        }


__all__ = ["publish", "snapshot", "wait"]
