"""JUnit receipts across every test-producing root.

This is provenance, not a recursive test total.  A directory whose files come
from different vintages is explicitly refused: adding archived or mid-rebuild
states together creates a number that never existed in any run.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from console.core.execution import checks
from console.core.panels import panel

ROOT = Path(__file__).resolve().parents[2]
RECEIPT_ROOTS = (
    ROOT / ".codex-local" / "logs",
    ROOT / ".codex-local" / "test-results",
    ROOT / "HytaleRL" / "hytalegym" / ".codex-local" / "test-results",
    ROOT / "HytaleRL" / ".codex-local",
    ROOT / "HytaleRL" / "hytale-plugin" / "build" / "test-results",
    ROOT / ".codex-local" / "run-snapshots",
)
VINTAGE_WINDOW_SECONDS = 15 * 60
# Receipt producers use ``-vN`` for the contract/schema version and append
# ``-final`` / ``-rN`` / ``-runN`` when that exact contract is rerun.  Keep
# those concepts separate: a later run supersedes an earlier run, while a v2
# receipt must not silently replace v1 evidence.
_REVISION = re.compile(r"(?i)(?:[-_.](?:(?:r|run)\d+|final))$")
_XML_SUFFIXES = (".junit.xml", ".xml")


def group_receipts(
    roots: Iterable[Path] = RECEIPT_ROOTS,
    *,
    vintage_window: float = VINTAGE_WINDOW_SECONDS,
    max_recent_groups: int = 120,
) -> list[dict[str, Any]]:
    """Group Gradle suite files by run directory and keep standalone receipts.

    A group only gets aggregate counters when every file belongs to one mtime
    vintage.  XML timestamps and filesystem mtimes are both retained because
    they frequently disagree and answer different provenance questions.
    """

    entries = _discover_xml(tuple(roots))
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    root_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        root_buckets[entry["root"]].append(entry)
        relative = Path(entry["relative_path"])
        if relative.name.startswith("TEST-"):
            group_name = relative.parent.as_posix() or "."
        else:
            group_name = relative.as_posix()
        buckets[(entry["root"], group_name)].append(entry)

    ordered_buckets = sorted(
        buckets.items(),
        key=lambda item: max(float(file["file_mtime"]) for file in item[1]),
        reverse=True,
    )
    selected = ordered_buckets[:max(1, max_recent_groups)]
    selected_keys = {key for key, _ in selected}
    selected.extend(
        (key, files) for key, files in ordered_buckets
        if key not in selected_keys and _is_mixed(files, vintage_window)
    )

    grouped = [
        _receipt_group(
            root, name, files, vintage_window=vintage_window, kind="receipt",
        )
        for (root, name), files in selected
    ]
    # A root summary is the guard against destructive recursive aggregation.
    # It remains a row even when the individual recent receipts are capped.
    grouped.extend(
        _receipt_group(
            root, "[root summary]", files,
            vintage_window=vintage_window, kind="root_summary",
        )
        for root, files in root_buckets.items()
    )
    _mark_superseded(grouped)
    return sorted(
        grouped,
        key=lambda receipt: float(receipt.get("mtime_max") or 0),
        reverse=True,
    )


def _receipt_group(
    root: str,
    name: str,
    files: list[dict[str, Any]],
    *,
    vintage_window: float,
    kind: str = "receipt",
) -> dict[str, Any]:
    mtimes = sorted(float(record["file_mtime"]) for record in files)
    histogram = Counter(_hour_bucket(value) for value in mtimes)
    mixed = _is_mixed(files, vintage_window)

    # Mtime alone is enough to refuse an invalid aggregate.  Only parse XML
    # for groups that could legally be totaled; on the live workspace this
    # avoids opening ~1,900 archived files merely to rediscover that they span
    # several days.
    parsed: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    if not mixed:
        for file in files:
            record = checks._junit_record(Path(file["path"]))
            if record is None:
                parse_errors.append(f"{file['relative_path']}: not JUnit XML")
                continue
            parsed.append({**file, **record})
            if record.get("error"):
                parse_errors.append(str(record["error"]))
    can_total = not mixed and not parse_errors and len(parsed) == len(files)

    totals = None
    if can_total:
        totals = {
            key: sum(int(record[key]) for record in parsed)
            for key in ("tests", "failures", "errors", "skipped")
        }

    timestamps = sorted({
        timestamp
        for record in parsed
        for timestamp in (record.get("timestamps") or [])
        if timestamp
    })
    root_path = Path(root)
    return {
        "root": root,
        "root_label": _relative(root_path),
        "name": name,
        "id": f"{root}::{name}",
        "kind": kind,
        "family": _family(name) if kind == "receipt" else "[root summary]",
        "file_count": len(files),
        "files": [record["relative_path"] for record in files],
        "mtime_min": mtimes[0] if mtimes else None,
        "mtime_max": mtimes[-1] if mtimes else None,
        "mtime_histogram": dict(sorted(histogram.items())),
        "xml_timestamps": timestamps,
        "mixed_vintage": mixed,
        "totals": totals,
        "parse_errors": parse_errors,
        "superseded_by": None,
    }


def _mark_superseded(receipts: list[dict[str, Any]]) -> None:
    families: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for receipt in receipts:
        if receipt.get("kind") == "root_summary":
            continue
        families[(receipt["root"], receipt["family"])].append(receipt)
    for family in families.values():
        if len(family) < 2:
            continue
        ordered = sorted(family, key=lambda item: float(item.get("mtime_max") or 0))
        newest = ordered[-1]
        for older in ordered[:-1]:
            older["superseded_by"] = newest["name"]


def _family(name: str) -> str:
    path = Path(name)
    filename = path.name
    lowered = filename.lower()
    stem = filename
    for suffix in _XML_SUFFIXES:
        if lowered.endswith(suffix):
            stem = filename[:-len(suffix)]
            break
    previous = None
    while stem != previous:
        previous = stem
        stem = _REVISION.sub("", stem)
    parent = path.parent.as_posix()
    return f"{parent}/{stem}" if parent not in {"", "."} else stem


def _discover_xml(roots: tuple[Path, ...]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for requested in roots:
        root = Path(requested)
        if root.is_file():
            paths = [root] if root.suffix.lower() == ".xml" else []
            root_dir = root.parent
        elif root.is_dir():
            paths = root.rglob("*.xml")
            root_dir = root
        else:
            continue
        for path in paths:
            try:
                stat = path.stat()
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                relative = path.relative_to(root_dir).as_posix()
            except ValueError:
                relative = path.name
            entries.append({
                "path": str(path),
                "root": str(root_dir),
                "relative_path": relative,
                "file_mtime": stat.st_mtime,
                "file_size": stat.st_size,
            })
    return entries


def _is_mixed(files: list[dict[str, Any]], vintage_window: float) -> bool:
    if len(files) < 2:
        return False
    mtimes = [float(file["file_mtime"]) for file in files]
    return max(mtimes) - min(mtimes) > vintage_window


def _hour_bucket(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:00Z",
    )


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _receipt_rows(receipts: list[dict[str, Any]], limit: int = 100) -> list[dict[str, Any]]:
    # Mixed directories are always visible even if an active workspace has
    # produced more than ``limit`` newer standalone receipts.
    selected = receipts[:limit]
    seen = {receipt["id"] for receipt in selected}
    selected.extend(
        receipt for receipt in receipts
        if receipt["mixed_vintage"] and receipt["id"] not in seen
    )

    rows: list[dict[str, Any]] = []
    for receipt in selected:
        totals = receipt["totals"]
        if receipt["mixed_vintage"]:
            result = "total refused — mixed vintages"
            tone = "warn"
        elif receipt["parse_errors"]:
            result = "unreadable JUnit XML"
            tone = "crit"
        else:
            assert totals is not None
            if totals["tests"] == 0:
                result = "no tests ran"
                tone = "warn"
            else:
                result = (
                    f"tests {totals['tests']} · failures {totals['failures']} · "
                    f"errors {totals['errors']} · skipped {totals['skipped']}"
                )
                tone = "crit" if totals["failures"] or totals["errors"] else "good"

        histogram = _histogram_text(receipt["mtime_histogram"])
        xml_time = (
            " → ".join((receipt["xml_timestamps"][0], receipt["xml_timestamps"][-1]))
            if len(receipt["xml_timestamps"]) > 1
            else (receipt["xml_timestamps"][0] if receipt["xml_timestamps"] else "undeclared")
        )
        mtime = _iso(receipt["mtime_max"])
        superseded = (
            "directory overview" if receipt.get("kind") == "root_summary"
            else (receipt["superseded_by"] or "current in family")
        )
        row_tone = "absent" if receipt["superseded_by"] else tone
        rows.append({
            "tone": row_tone,
            "cells": [
                {"value": receipt["name"], "detail": receipt["root_label"],
                 "mono": False, "grow": True, "wrap": True},
                {"value": result, "mono": False, "grow": True, "wrap": True},
                {"value": f"{receipt['file_count']} file(s)", "mono": False,
                 "detail": f"XML {xml_time} · mtime {mtime} · {histogram}",
                 "grow": True, "wrap": True},
                {"value": superseded, "mono": False,
                 "tone": "absent" if receipt["superseded_by"] else "",
                 "grow": True, "wrap": True},
            ],
        })
    return rows


def _iso(timestamp: float | None) -> str:
    if not timestamp:
        return "undeclared"
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(
        timespec="seconds",
    )


def _histogram_text(histogram: dict[str, int], limit: int = 8) -> str:
    items = list(histogram.items())
    visible = items[-limit:]
    text = " · ".join(f"{bucket}: {count}" for bucket, count in visible)
    if len(items) > limit:
        text = f"{len(items) - limit} older bucket(s) · {text}"
    return text or "no mtimes"


@panel(
    "test-receipts",
    title="Test receipts",
    tab="evidence",
    order=40,
    agent="built-in",
    refresh=30.0,
    size="wide",
    tags=("junit", "provenance"),
    description="JUnit evidence with roots, vintages, counters, and supersession.",
)
def receipts() -> dict[str, Any]:
    grouped = group_receipts()
    mixed = sum(receipt["mixed_vintage"] for receipt in grouped)
    superseded = sum(bool(receipt["superseded_by"]) for receipt in grouped)
    files = sum(
        int(receipt["file_count"]) for receipt in grouped
        if receipt.get("kind") == "root_summary"
    )
    logical = sum(receipt.get("kind") == "receipt" for receipt in grouped)
    return {
        "view": "rows",
        "scroll": True,
        "columns": ["receipt", "counters", "provenance", "supersession"],
        "rows": _receipt_rows(grouped),
        "status": {
            "label": f"{mixed} mixed-vintage director{'y' if mixed == 1 else 'ies'}",
            "tone": "warn" if mixed else "info",
            "detail": "Mixed vintages are listed but never totaled.",
        },
        "summary": (
            "A receipt is current file state, not a timeless citation. XML "
            "timestamps and file mtimes are both shown; later same-family "
            "receipts supersede earlier ones."
        ),
        "count": f"{logical} receipts · {files} XML files",
        "tags": [f"{len(RECEIPT_ROOTS)} roots", f"{superseded} superseded"],
        "updated_at": max(
            (float(receipt.get("mtime_max") or 0) for receipt in grouped),
            default=0,
        ) or None,
        "note": (
            "Directory totals are emitted only when all files fall within one "
            f"{int(VINTAGE_WINDOW_SECONDS / 60)}-minute vintage window."
        ),
    }
