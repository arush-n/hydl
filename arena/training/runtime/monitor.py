"""Stable file/API surface for observing automatic training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from threading import Lock
from time import time
from typing import Any, Mapping


TRAINING_EVENT_SCHEMA = "arena_training_event_v1"
TRAINING_STATUS_SCHEMA = "arena_training_status_v1"


@dataclass(frozen=True, slots=True)
class TrainingEvent:
    sequence: int
    kind: str
    cycle: int | None
    payload: Mapping[str, Any]
    unix_seconds: float

    def describe(self) -> dict[str, Any]:
        return {
            "schema": TRAINING_EVENT_SCHEMA,
            "sequence": self.sequence,
            "kind": self.kind,
            "cycle": self.cycle,
            "payload": dict(self.payload),
            "unix_seconds": self.unix_seconds,
        }


class TrainingMonitor:
    """Append events and atomically publish the latest machine-readable state."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.jsonl"
        self.status_path = self.output_dir / "status.json"
        self._lock = Lock()
        existing = self.events()
        self._sequence = 0 if not existing else existing[-1]["sequence"] + 1

    def emit(
        self,
        kind: str,
        payload: Mapping[str, Any] | None = None,
        *,
        cycle: int | None = None,
        phase: str | None = None,
    ) -> TrainingEvent:
        if not kind:
            raise ValueError("event kind must be nonempty")
        with self._lock:
            event = TrainingEvent(self._sequence, kind, cycle, payload or {}, time())
            self._sequence += 1
            with self.events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event.describe(), sort_keys=True) + "\n")
            status = {
                "schema": TRAINING_STATUS_SCHEMA,
                "state": kind,
                "phase": phase,
                "cycle": cycle,
                "last_sequence": event.sequence,
                "updated_unix_seconds": event.unix_seconds,
                "payload": dict(event.payload),
            }
            temporary = self.status_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
            )
            temporary.replace(self.status_path)
        return event

    def status(self) -> dict[str, Any] | None:
        return (
            None
            if not self.status_path.is_file()
            else json.loads(self.status_path.read_text(encoding="utf-8"))
        )

    def events(self, *, after_sequence: int = -1) -> list[dict[str, Any]]:
        if not self.events_path.is_file():
            return []
        return [
            value
            for line in self.events_path.read_text(encoding="utf-8").splitlines()
            if (value := json.loads(line))["sequence"] > after_sequence
        ]

    def snapshot(self, *, after_sequence: int = -1) -> dict[str, Any]:
        """Return the complete console/ADK polling response in one call."""

        return {
            "status": self.status(),
            "events": self.events(after_sequence=after_sequence),
        }


def open_training_monitor(output_dir: str | Path) -> TrainingMonitor:
    """Open the polling surface used by CLIs, ADK, and a future console panel."""

    return TrainingMonitor(Path(output_dir))


__all__ = [
    "TRAINING_EVENT_SCHEMA",
    "TRAINING_STATUS_SCHEMA",
    "TrainingEvent",
    "TrainingMonitor",
    "open_training_monitor",
]
