"""Metric streams: anything that trains can report here and be seen.

The Train tab charted four fixed quantities -- total loss, policy loss, value
loss, entropy -- because PPO was the only learner. That is a ceiling written
into the UI: a Dreamer reports world-model loss, imagination return and a KL
term; a transformer reports attention entropy and a token-accuracy curve; a CNN
encoder reports something else again. None of them fit four canvases named
after PPO, and adding a fifth architecture would mean editing the HTML.

So the console does not decide what a metric is. **A stream declares its own
keys by logging them**, and the dashboard charts whatever turns up::

    POST /api/streams/dreamer-run-3/log
    {"step": 120, "metrics": {"world_model_loss": 0.42, "imagined_return": 8.1}}

Nothing needs to exist first -- the stream is created by its first log line, so
an architecture that is not built yet, or lives in a different process
entirely, can appear on the dashboard the moment it reports. That is the point:
the console is a place to watch *any* learner in this repo, not a PPO viewer.

Storage is one JSONL file per stream, appended a line at a time. A crashed run
keeps every point it managed to write, and a reader can tail the file with no
lock -- the same reason `jobs.py` writes `metrics.jsonl`.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]

#: One directory, one file per stream. Beside the run artifacts rather than
#: inside them: a stream may outlive any single run, and often belongs to a
#: process the console never launched.
STREAM_ROOT = _ROOT / "artifacts" / "console-streams"

#: Stream ids become filenames, so they are constrained rather than sanitised.
#: Rejecting is honest; quietly rewriting an id would split one run's metrics
#: across two files that look unrelated.
VALID_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

#: Points returned per key by default. A long run writes tens of thousands;
#: a chart 120 px tall cannot show them and the JSON would dwarf the page.
DEFAULT_LIMIT = 2000


class StreamError(ValueError):
    """A malformed stream id or log line."""


def _path(stream_id: str) -> Path:
    if not VALID_ID.match(stream_id or ""):
        raise StreamError(
            f"invalid stream id {stream_id!r}: use letters, digits, dot, dash "
            "or underscore, 1-64 characters"
        )
    return STREAM_ROOT / f"{stream_id}.jsonl"


def log(stream_id: str, *, step: int | None = None,
        metrics: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Append one point. Creates the stream if this is its first line.

    Only finite numbers are stored as metrics. A NaN reaching a chart makes the
    whole line vanish with no error, and in this repo a NaN is usually the
    finding -- so it is counted and reported rather than plotted.
    """

    path = _path(stream_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    clean: dict[str, float] = {}
    rejected: list[str] = []
    for key, value in (metrics or {}).items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            rejected.append(key)
            continue
        if number != number or number in (float("inf"), float("-inf")):
            rejected.append(key)
            continue
        clean[str(key)] = number

    existing = _count(path)
    row: dict[str, Any] = {
        "step": int(step) if step is not None else existing,
        "at": time.time(),
        "metrics": clean,
    }
    if meta:
        row["meta"] = meta
    if rejected:
        # Not silently dropped: a non-finite loss is the single most useful
        # thing this console can tell you, and it must not look like a gap.
        row["rejected"] = sorted(rejected)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")

    return {"stream": stream_id, "step": row["step"], "logged": sorted(clean),
            "rejected": sorted(rejected)}


def _count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _read(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # a torn last line on a crashed writer
    return rows[-limit:] if limit and len(rows) > limit else rows


def index() -> list[dict[str, Any]]:
    """Every stream, newest activity first, with the keys each declared."""

    if not STREAM_ROOT.is_dir():
        return []
    out = []
    for path in STREAM_ROOT.glob("*.jsonl"):
        rows = _read(path, DEFAULT_LIMIT)
        keys: set[str] = set()
        rejected: set[str] = set()
        for row in rows:
            keys.update((row.get("metrics") or {}).keys())
            rejected.update(row.get("rejected") or ())
        last = rows[-1] if rows else {}
        out.append({
            "id": path.stem,
            "points": len(rows),
            "keys": sorted(keys),
            "non_finite_keys": sorted(rejected),
            "last_step": last.get("step"),
            "updated_at": last.get("at") or path.stat().st_mtime,
            "meta": last.get("meta") or {},
        })
    out.sort(key=lambda s: s["updated_at"], reverse=True)
    return out


def series(stream_id: str, limit: int = DEFAULT_LIMIT) -> dict[str, Any] | None:
    """One stream as chartable series, or ``None`` if it does not exist."""

    path = _path(stream_id)
    if not path.is_file():
        return None
    rows = _read(path, limit)

    steps = [row.get("step") for row in rows]
    keys: list[str] = []
    for row in rows:
        for key in (row.get("metrics") or {}):
            if key not in keys:
                keys.append(key)

    # A key absent from a point is a hole, not a zero -- a Dreamer logging its
    # KL every tenth step must not draw a sawtooth to the floor. `null` lets the
    # chart break the line instead.
    out = []
    for key in keys:
        out.append({
            "name": key,
            "points": [(row.get("metrics") or {}).get(key) for row in rows],
        })

    rejected: set[str] = set()
    for row in rows:
        rejected.update(row.get("rejected") or ())

    return {
        "id": stream_id,
        "steps": steps,
        "series": out,
        "points": len(rows),
        "non_finite_keys": sorted(rejected),
        "meta": (rows[-1].get("meta") if rows else {}) or {},
    }


def delete(stream_id: str) -> bool:
    """Remove one stream's file. Returns whether anything was there."""

    path = _path(stream_id)
    if not path.is_file():
        return False
    path.unlink()
    return True
