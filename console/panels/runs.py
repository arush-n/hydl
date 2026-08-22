"""Panels over the stored-run archive.

Worked examples of the `rows` and `facts` views. Between them they answer the
two questions the archive is actually for -- *what did I just run* and *which
of these are even the same experiment*.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from console.core.evidence import store
from console.core.panels import panel


@panel("recent-runs", title="Recent runs", tab="status", order=30, agent="built-in",
       description="The last few rollouts, newest first.", refresh=20.0,
       size="wide")
def recent_runs() -> dict[str, Any]:
    runs = store.index(limit=8)
    if not runs:
        return {"view": "note",
                "text": "No runs stored yet. Every run writes an artifact "
                        "recording the contract hashes it ran against."}

    rows = []
    for run in runs:
        summary = run.get("summary") or {}
        tone = "neg" if run.get("void") else ("warn" if run.get("warnings") else "pos")
        rows.append({
            "run_id": run["run_id"],
            "tone": tone,
            "cells": [
                f"{run.get('loadout')} vs {run.get('opponent') or 'none'}",
                run.get("policy") or "—",
                f"seed {run.get('seed')}",
                f"{summary.get('landed', 0)} landed",
                f"reward {summary.get('reward_total', 0)}",
            ],
        })
    return {
        "view": "rows",
        "columns": ["matchup", "policy", "seed", "landed", "reward"],
        "rows": rows,
        "count": f"{len(runs)} shown",
    }


@panel("run-categories", title="How the archive splits", tab="archive", order=40, agent="built-in",
       description="Runs grouped by the things that decide comparability.")
def run_categories() -> dict[str, Any]:
    runs = store.index(limit=500)
    if not runs:
        return {"view": "note", "text": "No stored runs to group yet."}

    widths = Counter(str(r.get("observation_size") or "unknown") for r in runs)
    loadouts = Counter(str(r.get("loadout") or "unknown") for r in runs)
    policies = Counter(str(r.get("policy") or "unknown") for r in runs)
    void = sum(1 for r in runs if r.get("void"))

    facts = [
        {"label": "stored runs", "value": len(runs)},
        {"label": "observation widths", "value": len(widths),
         "tone": "neg" if len(widths) > 1 else "pos"},
        {"label": "void", "value": void, "tone": "neg" if void else "pos"},
        {"label": "loadouts", "value": len(loadouts)},
        {"label": "policies", "value": len(policies)},
    ]
    facts += [{"label": f"width {w}", "value": n} for w, n in widths.most_common(4)]

    note = (
        "More than one observation width means this archive spans a contract "
        "change: those runs are not data points in one experiment, they are "
        "separate experiments. The width went 7818 → 8259 → 8271 without "
        "anyone noticing, which is why it is counted here."
        if len(widths) > 1 else
        "One observation width across the archive — these runs are at least "
        "shape-comparable. Contract hashes still decide the rest; use compare."
    )
    return {"view": "facts", "facts": facts, "note": note}
