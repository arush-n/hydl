"""What compute a run is actually getting, and whether it paid for a cold start.

Two invisible things that change how a result reads. Which device the
environment ran on — the same number from CPU and GPU has an order of magnitude
of throughput between it. And whether the scene cache was hit: a Region scene
costs ~40 s to build and ~60 s more to compile the collector, so a repeated
configuration that pays it again is a cold start nobody asked for.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from console.core.execution import devices
from console.core.panels import panel


@panel("compute", title="Compute", tab="status", order=20, agent="built-in",
       refresh=10.0, size="wide", tags=("runtime", "jax"),
       description="Device, precision, and whether the last run was a cold start.")
def compute() -> dict[str, Any]:
    state = devices.status()
    cache = state.get("scene_cache") or {}

    runtime_facts: list[dict[str, Any]] = [
        {"label": "backend", "value": state["backend"],
         "tone": "pos" if state["backend"] != "cpu" else ""},
        {"label": "devices", "value": state["device_count"]},
        # float64 silently doubles memory and diverges from the Java port,
        # which matches JAX in float32 -- measured ~800x further apart in
        # double where the expression cancels.
        {"label": "x64", "value": "on" if state["x64_enabled"] else "off",
         "tone": "neg" if state["x64_enabled"] else "pos"},
        {"label": "last run", "value": "cache hit" if cache.get("last_was_hit")
         else "cold build", "tone": "pos" if cache.get("last_was_hit") else ""},
        {"label": "jax", "value": state["jax"]},
    ]

    device_rows: list[dict[str, Any]] = []
    for device in state["devices"]:
        memory = []
        if "in_use_mb" in device:
            memory.append(f"{device['in_use_mb']} MB in use")
        if "peak_mb" in device:
            memory.append(f"{device['peak_mb']} MB peak")
        device_rows.append({
            "tone": "good" if state["backend"] != "cpu" else "info",
            "cells": [
                {"value": f"device {device['id']}", "detail": device["type"]},
                {"value": ", ".join(memory) if memory else "memory not reported",
                 "mono": False},
            ],
        })

    note = state.get("note") or (
        "Repeating a configuration reuses its built handle, so the scene build "
        "and the collector compile are both skipped. Changing any scene field "
        "— loadout, world, zone, opponent policy, parameters — is a different "
        "scene and rebuilds."
    )
    entries = cache.get("entries", [])
    limit = max(1, int(cache.get("limit", 0) or 0))
    precision = "float64" if state["x64_enabled"] else "float32"
    return {
        "view": "detail",
        "status": {
            "label": state["backend"],
            "tone": "good" if state["backend"] != "cpu" else "info",
            "detail": f"{state['device_count']} device(s) available",
        },
        "summary": (
            "The runtime identity and cache state that determine whether a "
            "timing or parity result is comparable."
        ),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "tags": [precision, {"label": "cache hit", "tone": "good"}]
        if cache.get("last_was_hit") else [precision, "cold path"],
        "sections": [
            {
                "title": "Runtime contract",
                "description": "Backend and numerical mode.",
                "view": "facts",
                "facts": runtime_facts,
            },
            {
                "title": "Scene cache",
                "description": "Resident compiled scene handles.",
                "view": "progress",
                "items": [{
                    "label": "resident scenes",
                    "value": len(entries),
                    "max": limit,
                    "display": f"{len(entries)} / {cache.get('limit', 0)}",
                    "tone": "good" if cache.get("last_was_hit") else "info",
                    "detail": "A matching scene skips construction and collector compile.",
                }],
            },
            {
                "title": "Devices",
                "description": "Every device visible to this console process.",
                "view": "rows",
                "size": "full",
                "columns": ["device", "memory"],
                "rows": device_rows,
            },
        ],
        "note": note,
        "count": f"{state['device_count']} device(s)",
    }
