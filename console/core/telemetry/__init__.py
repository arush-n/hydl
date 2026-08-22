"""What the console can see while a run is happening.

``live``      in-process push stream for console jobs and metrics
``logs``      log files the console can find and tail
``streams``   metric streams: anything that trains can report here
``surfaces``  what the console reports about the ADK and the live server
"""

from __future__ import annotations

from console.core.telemetry import live as live
from console.core.telemetry import logs as logs
from console.core.telemetry import streams as streams
from console.core.telemetry import surfaces as surfaces


__all__ = [
    "live",
    "streams",
    "surfaces",
]
