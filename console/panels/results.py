"""Artifact-declared result ordering on the Evidence tab."""

from __future__ import annotations

import json
import mmap
from pathlib import Path
from typing import Any

from console.core.panels import panel
from console.core.evidence.results import UNDECLARED, declared_result_blocks

ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOTS = (ROOT / "agents", ROOT / "experimental")
_CACHE: dict[Path, tuple[int, int, list[dict[str, Any]]]] = {}


def discover_declared_results(
    roots: tuple[Path, ...] = RESULT_ROOTS, *, limit: int = 12,
) -> list[dict[str, Any]]:
    """Find newest JSON artifacts that opt into the result declaration."""

    candidates: list[tuple[int, Path]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            try:
                stat = path.stat()
                if stat.st_size > 32 * 1024 * 1024:
                    continue
                # Avoid decoding the thousands of unrelated artifacts.  This
                # is an exact contract-key prefilter, not semantic inference.
                if not _contains_declaration(path, stat.st_size):
                    continue
                candidates.append((stat.st_mtime_ns, path))
            except OSError:
                continue

    found: list[dict[str, Any]] = []
    for _mtime, path in sorted(candidates, reverse=True)[:max(1, limit)]:
        stat = path.stat()
        cached = _CACHE.get(path.resolve())
        if cached and cached[:2] == (stat.st_mtime_ns, stat.st_size):
            blocks = cached[2]
        else:
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            blocks = declared_result_blocks(document)
            _CACHE[path.resolve()] = (stat.st_mtime_ns, stat.st_size, blocks)
        for block in blocks:
            found.append({
                "path": path,
                "relative_path": _relative(path),
                "mtime": stat.st_mtime,
                "block": block,
            })
    return found


def _contains_declaration(path: Path, size: int) -> bool:
    if size <= 0:
        return False
    try:
        with path.open("rb") as source:
            with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as data:
                return data.find(b'\"diagnostic_only_metrics\"') >= 0
    except (OSError, ValueError):
        return False


def result_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    sections = []
    diagnostic_count = 0
    for record in records:
        block = record["block"]
        rows = []
        for metric in block["metrics"]:
            diagnostic = bool(metric["diagnostic_only"])
            diagnostic_count += diagnostic
            chips = []
            if metric["primary"]:
                chips.append({"label": "primary", "tone": "good"})
            if diagnostic:
                chips.append({"label": "diagnostic_only", "tone": "absent"})
            coverage = metric.get("coverage")
            coverage_text = "—"
            if coverage is not None:
                coverage_text = (
                    f"resolved {coverage['resolved']} · "
                    f"censored {coverage['censored']}"
                )
            rows.append({
                "tone": "absent" if diagnostic else "info",
                "cells": [
                    {"value": metric["name"], "chips": chips, "wrap": True},
                    {
                        "value": _display(metric["value"]),
                        "detail": f"direction: {metric['direction']}",
                        "mono": False,
                    },
                    {"value": coverage_text, "wrap": True, "mono": False},
                ],
                "detail": (
                    f"artifact key {metric['source_name']}"
                    if metric["source_name"] != metric["name"] else ""
                ),
            })
        if not rows:
            rows.append({
                "tone": "absent",
                "cells": [
                    {"value": "metrics", "chips": [{"label": UNDECLARED, "tone": "absent"}]},
                    UNDECLARED,
                    "—",
                ],
            })
        declaration_tone = "info" if block["declaration_valid"] else "crit"
        sections.append({
            "title": record["relative_path"],
            "description": (
                f"{block['path']} · primary {block['primary_metric']} · "
                f"source {block['metric_source']}"
            ),
            "badge": (
                f"{len(block['metrics'])} metrics"
                if block["declaration_valid"] else "malformed declaration"
            ),
            "tone": declaration_tone,
            "size": "full",
            "view": "rows",
            "columns": ["Metric", "Declared result", "Coverage"],
            "rows": rows,
        })
    updated = max((record["mtime"] for record in records), default=0)
    return {
        "view": "detail",
        "status": {
            "label": f"{diagnostic_count} diagnostic",
            "tone": "info" if records else "absent",
        },
        "count": len(records),
        "summary": (
            "Non-diagnostic metrics lead. Artifact-declared diagnostic metrics "
            "are greyed; every rate carries declared resolved/censored coverage."
        ),
        "tags": ["declared ordering", "no semantic inference"],
        "sections": sections or [{
            "title": "No declaration found",
            "description": "Artifacts remain readable; result ordering is undeclared.",
            "badge": UNDECLARED,
            "tone": "absent",
            "view": "note",
            "text": "No JSON artifact currently declares diagnostic_only_metrics.",
        }],
        "updated_at": updated,
    }


def _display(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


@panel(
    "declared-results",
    title="Declared result cards",
    tab="evidence",
    order=35,
    description="Result emphasis comes from the artifact, never a lane-specific metric list.",
    refresh=30,
    agent="built-in",
    size="wide",
    tags=("results", "coverage"),
)
def declared_results() -> dict[str, Any]:
    return result_payload(discover_declared_results())
