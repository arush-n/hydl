"""Run diagnostics rendered directly from declared artifact fields."""

from __future__ import annotations

import json
import mmap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from console.core.evidence.diagnostics import diagnostic_groups
from console.core.panels import panel

ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_ROOTS = (ROOT / "agents", ROOT / "experimental")
_NEEDLES = (b'"active_head_active_fraction"', b'"criterion_success_rate"')


def discover_diagnostics(
    roots: tuple[Path, ...] = DIAGNOSTIC_ROOTS, *, limit: int = 12,
) -> list[dict[str, Any]]:
    candidates: list[tuple[int, Path]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            try:
                stat = path.stat()
                if stat.st_size and stat.st_size <= 32 * 1024 * 1024 and _matches(path):
                    candidates.append((stat.st_mtime_ns, path))
            except OSError:
                continue

    records = []
    for _mtime, path in sorted(candidates, reverse=True)[:max(1, limit)]:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        groups = diagnostic_groups(document)
        if groups["head_tables"] or groups["pairs"]:
            records.append({
                "path": _relative(path),
                "mtime": path.stat().st_mtime,
                **groups,
            })
    return records


def _matches(path: Path) -> bool:
    try:
        with path.open("rb") as source, mmap.mmap(
            source.fileno(), 0, access=mmap.ACCESS_READ
        ) as data:
            return any(data.find(needle) >= 0 for needle in _NEEDLES)
    except (OSError, ValueError):
        return False


def diagnostics_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    sections = []
    ordered = sorted(records, key=lambda item: item["mtime"], reverse=True)
    for index, record in enumerate(ordered):
        latest = index == 0
        stamp = datetime.fromtimestamp(record["mtime"], timezone.utc).isoformat()
        outcomes = list(record["head_tables"])
        for pair in record["pairs"]:
            outcomes.extend({**arm, "path": pair["path"]} for arm in pair["arms"])
        if outcomes:
            sections.append({
                "title": record["path"],
                "description": f"{stamp} · Criterion and gated success are distinct; paired arms stay together.",
                "badge": "latest" if latest else "older",
                "collapsible": True,
                "open": latest,
                "size": "full",
                "view": "rows",
                "columns": ["Arm", "Criterion success", "Gated success", "Teacher coefficient"],
                "rows": [
                    {
                        "cells": [
                            {"value": item["arm"], "detail": item["path"], "wrap": True},
                            _rate(item["criterion_success_rate"]),
                            _rate(item["success_rate"]),
                            _rate(item["teacher_coefficient"]),
                        ]
                    }
                    for item in outcomes
                ],
            })
        for table in record["head_tables"]:
            sections.append({
                "title": f"{record['path']} · {table['arm']} head activity",
                "description": f"{stamp} · {table['path']}",
                "badge": f"{'latest · ' if latest else ''}{len(table['activity'])} heads",
                "collapsible": True,
                "open": latest,
                "size": "full",
                "view": "rows",
                "columns": ["Action head", "Active fraction"],
                "rows": [
                    {"cells": [name, _rate(fraction)]}
                    for name, fraction in table["activity"].items()
                ],
            })
    return {
        "view": "detail",
        "status": {
            "label": f"{sum(len(item['head_tables']) for item in records)} head tables",
            "tone": "info" if records else "absent",
        },
        "summary": (
            "Artifact fields are shown without reinterpreting success or inferring missing arms."
        ),
        "sections": sections or [{
            "title": "No run diagnostics found",
            "view": "note",
            "text": "No recent artifact declares per-head activity or paired success metrics.",
        }],
        "updated_at": max((item["mtime"] for item in ordered), default=0) or None,
    }


def _rate(value: Any) -> str:
    if value is None:
        return "undeclared"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


@panel(
    "run-diagnostics",
    title="Training diagnostics",
    tab="evidence",
    order=45,
    description="Newest training artifact first; older head activity and outcomes expand on demand.",
    refresh=30,
    agent="built-in",
    size="wide",
    tags=("activity", "paired evaluation"),
)
def run_diagnostics() -> dict[str, Any]:
    return diagnostics_payload(discover_diagnostics())


__all__ = ["diagnostics_payload", "discover_diagnostics", "run_diagnostics"]
