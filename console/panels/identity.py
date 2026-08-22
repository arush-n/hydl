"""Status identity strip backed by declared opaque hashes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from console.core.catalog import identities
from console.core.panels import panel


@panel(
    "identity-strip",
    title="Identity",
    tab="status",
    order=5,
    agent="built-in",
    refresh=15.0,
    size="wide",
    tags=("provenance", "live-contract"),
    description="Declared identities compared with the counterpart that is actually live.",
)
def identity_strip() -> dict[str, Any]:
    state = identities.bridge_identity()
    comparisons = state["comparisons"]
    mismatch = [item for item in comparisons if item["match"] is False]
    undeployed = state.get("undeployed_sha256")
    if undeployed:
        label = f"undeployed {undeployed[:12]}…"
        tone = "warn"
    elif mismatch:
        label = f"{len(mismatch)} mismatch(es)"
        tone = "warn"
    elif all(item["match"] is True for item in comparisons):
        label = "identities match"
        tone = "good"
    else:
        label = "live identity undeclared"
        tone = "absent"

    rows = []
    for comparison in comparisons:
        match = comparison["match"]
        rows.append({
            "tone": "good" if match is True else ("warn" if match is False else "absent"),
            "cells": [
                {"value": comparison["label"], "mono": False},
                {"value": _short(comparison["declared"]),
                 "detail": comparison["declared_source"]},
                {"value": _short(comparison["counterpart"]),
                 "detail": comparison["counterpart_source"]},
                {"value": comparison["state"], "mono": False},
            ],
        })

    deployment = state["deployment"]
    observed_times = [
        value for value in (
            deployment["deploy_time"], deployment["server_restart_time"],
            deployment["jar_build_mtime"],
        ) if value is not None
    ]
    facts = [
        {"label": "deploy observed", "value": _time(deployment["deploy_time"]),
         "detail": deployment["deploy_time_source"] or "no predecessor backup"},
        {"label": "server restart", "value": _time(deployment["server_restart_time"]),
         "detail": deployment["server_log"] or "no server log"},
        {"label": "jar file mtime", "value": _time(deployment["jar_build_mtime"]),
         "detail": "build/copy provenance only — never used as deploy time"},
        {"label": "canonical install", "value": _short(
            state["canonical_install"]["sha256"]),
         "detail": state["canonical_install"]["path"] or "undeclared"},
    ]
    return {
        "view": "detail",
        "status": {"label": label, "tone": tone},
        "summary": (
            "The bridge is one instance of the generic rule: compare a "
            "declared identity only when a live counterpart exists."
        ),
        "sections": [
            {"title": "Bridge comparisons", "view": "rows", "size": "full",
             "columns": ["comparison", "declared/build", "live", "result"],
             "rows": rows},
            {"title": "Deployment provenance", "view": "facts", "size": "full",
             "facts": facts},
        ],
        "updated_at": max(observed_times) if observed_times else None,
        "note": (
            "Jar content mtimes are preserved by these copies and are not "
            "deployment timestamps. The predecessor backup's creation time "
            "and the server-log start are the deployment signals."
        ),
    }


def _short(value: str | None) -> str:
    return value[:12] + "…" if value else "undeclared"


def _time(value: float | None) -> str:
    if value is None:
        return "undeclared"
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat(
        timespec="seconds",
    )
