"""Which library worlds a run is using now, and which any run has ever used.

Deprecating a world is only safe to judge with this in front of you: a world no
run has ever loaded can go without touching a recorded result, while one that
is live belongs to the run happening right now.

Both answers are derived, never listed. `live` recomputes the working set from
the running job's own launch spec, so it matches what the run actually
resolved rather than a copy kept in step by hand. `used` reads the world
identity each finished run stamped into its report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

#: Newest run reports to read. A full scan grows without bound and this list is
#: drawn on every page load; the recent past is what a decision turns on.
REPORT_SCAN_LIMIT = 60


def _selection_from_spec(spec: dict[str, Any]) -> tuple[int, ...]:
    from worlds.libraries import select

    count = int(spec.get("world_count") or 1)
    if count < 1:
        return ()
    order = str(spec.get("world_order") or "digest")
    if spec.get("flattest_worlds"):
        order = "flattest"
    elif spec.get("mixed_terrain_worlds"):
        order = "mixed"
    try:
        return select(count, order=order, key=int(spec.get("selection_key") or 0))
    except (ValueError, OSError):
        # A spec the library can no longer satisfy is not a reason to fail the
        # page: it means nothing is live, which is what an empty tuple says.
        return ()


def live_seeds() -> tuple[int, ...]:
    """Seeds in the working set of the job running now, or none."""

    try:
        from console.core.execution import jobs

        running = jobs.active()
    except Exception:  # noqa: BLE001 - an inventory must render without a job
        return ()
    if running is None or not getattr(running, "spec", None):
        return ()
    return _selection_from_spec(dict(running.spec))


#: Where a stage report carries its world identity. A report holds the same
#: identity in both places; reading only one of them returned nothing for every
#: run, which is indistinguishable from "no run has used a world".
_IDENTITY_PATHS = (
    ("replay", "world_identity"),
    ("contracts", "region_reset"),
)


def _at(document: Any, path: tuple[str, ...]) -> dict[str, Any] | None:
    for name in path:
        if not isinstance(document, dict):
            return None
        document = document.get(name)
    return document if isinstance(document, dict) else None


def _report_seeds(report: Path) -> Iterable[int]:
    try:
        document = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    for path in _IDENTITY_PATHS:
        identity = _at(document, path)
        if identity is None:
            continue
        seeds = identity.get("resident_artifact_seed")
        if isinstance(seeds, list) and seeds:
            return [int(seed) for seed in seeds]
        single = identity.get("artifact_seed")
        if isinstance(single, int):
            return [single]
    return ()


def used_seeds(roots: Iterable[Path]) -> tuple[int, ...]:
    """Every seed a finished run recorded, newest runs first."""

    reports: list[Path] = []
    for root in roots:
        if root.is_dir():
            reports.extend(root.glob("*/stage/report.json"))
    reports.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    found: set[int] = set()
    for report in reports[:REPORT_SCAN_LIMIT]:
        found.update(_report_seeds(report))
    return tuple(sorted(found))


__all__ = ["REPORT_SCAN_LIMIT", "live_seeds", "used_seeds"]
