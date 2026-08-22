"""Run snapshots visible before their target artifact lands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from console.core.execution.inflight import scan_inflight
from console.core.panels import panel

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_ROOT = ROOT / ".codex-local" / "run-snapshots"


@panel(
    "in-flight-work",
    title="In-flight work",
    tab="evidence",
    order=60,
    description="Run entry points become visible before an artifact exists.",
    refresh=10,
    agent="built-in",
    size="wide",
    tags=("snapshots", "provenance"),
)
def in_flight_work() -> dict[str, Any]:
    records = scan_inflight(SNAPSHOT_ROOT, workspace=ROOT)
    counts = {state: sum(row["state"] == state for row in records)
              for state in ("staged", "running", "landed")}
    rows = []
    tones = {"staged": "warn", "running": "info", "landed": "good"}
    for record in records:
        pid_detail = f" · pid {', '.join(map(str, record['pids']))}" if record["pids"] else ""
        snapshot = record["snapshot"]
        rows.append({
            "tone": tones[record["state"]],
            "badge": record["state"],
            "cells": [
                {"value": record["name"], "wrap": True},
                {"value": record["target_display"], "wrap": True, "mono": False},
                {
                    "value": snapshot.name if snapshot else "undeclared",
                    "tone": "info" if snapshot else "absent",
                    "mono": False,
                },
            ],
            "detail": f"{record['script'].name}{pid_detail}",
        })
    return {
        "view": "rows",
        "columns": ["Work", "Target output", "Frozen source"],
        "rows": rows,
        "count": len(records),
        "status": {
            "label": f"{counts['running']} running",
            "tone": "info" if counts["running"] else "good",
        },
        "summary": (
            f"{counts['staged']} staged · {counts['running']} running · "
            f"{counts['landed']} landed. State comes from the declared target "
            "and the live process table; missing targets remain undeclared."
        ),
        "updated_at": max((row["updated_at"] for row in records), default=0),
    }
