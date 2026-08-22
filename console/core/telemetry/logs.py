"""Log files the console can find and tail.

Sources are DISCOVERED, never listed. Two places hold them:

``logs/<category>/<name>.log``
    Anything long-lived: the console itself, the game server, tools. A category
    is a directory, so adding one is creating a directory.

the active training run's own ``stdout.log`` and ``stderr.log``
    Surfaced under the ``training`` category without copying, so a run's output
    is readable while it is still being written.

Reads are bounded and never fail the caller: a log that is locked, truncated or
mid-write returns what could be read rather than raising into a request.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]

#: Long-lived logs live here, one directory per category.
LOG_ROOT = _ROOT / "logs"
#: Categories the console creates itself, so a fresh checkout has somewhere to
#: write rather than scattering files at the root.
STANDARD_CATEGORIES = ("console", "server", "training", "tools")
#: Files older layouts left directly in the log root.
UNSORTED = "unsorted"
#: Most bytes read when tailing. A log can reach hundreds of megabytes; a tail
#: must not depend on its total size.
TAIL_BYTES = 256 * 1024
DEFAULT_LINES = 200
MAXIMUM_LINES = 5000
_SOURCE_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._/-]{0,200}")


@dataclass(frozen=True)
class LogSource:
    id: str
    category: str
    name: str
    path: Path
    bytes: int
    modified: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "name": self.name,
            "bytes": self.bytes,
            "modified": self.modified,
        }


def ensure_layout() -> list[str]:
    """Create the standard categories. Returns the ones newly made."""

    created = []
    for category in STANDARD_CATEGORIES:
        directory = LOG_ROOT / category
        if not directory.is_dir():
            directory.mkdir(parents=True, exist_ok=True)
            created.append(category)
    return created


def _describe(path: Path, category: str, name: str) -> LogSource | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return LogSource(
        id=f"{category}/{name}",
        category=category,
        name=name,
        path=path,
        bytes=stat.st_size,
        modified=stat.st_mtime,
    )


def _run_sources() -> tuple[list[LogSource], str | None]:
    """Training job logs, read where they are written, and which to open.

    Returns the sources plus the id of the CURRENT job's stdout: the running
    job if there is one, otherwise the most recently started. `jobs.index()` is
    newest-first, so its head is that job even after everything has finished.
    """

    try:
        from console.core.execution import jobs

        running = jobs.active()
        running_id = running.job_id if running is not None else None
        rows = [job for job in jobs.index() if job.get("directory")]
    except Exception:  # noqa: BLE001 - logs must list even if jobs cannot
        return [], None

    found: list[LogSource] = []
    preferred: str | None = None
    for position, job in enumerate(rows):
        directory = Path(str(job["directory"]))
        job_id = str(job.get("job_id") or directory.name)
        streams = (
            ("stdout", directory / "stdout.log"),
            ("events", directory / "stage" / "events.jsonl"),
            ("stderr", directory / "stderr.log"),
        )
        # The running job wins, else the newest, which is position 0.
        current = job_id == running_id or (running_id is None and position == 0)
        for name, path in streams:
            source = _describe(path, "training", f"{job_id}.{name}")
            if source is None:
                continue
            found.append(source)
            # `stdout` carries one readable line per event and `events` the same
            # events as raw JSON, so prefer stdout and fall back only when it
            # has nothing in it.
            if current and source.bytes and (preferred is None or name == "stdout"):
                preferred = source.id
    return found, preferred


def _sources() -> tuple[list[LogSource], str | None]:
    """Every log the console can see, and which one to open by default."""

    found: list[LogSource] = []
    if LOG_ROOT.is_dir():
        for entry in sorted(LOG_ROOT.iterdir()):
            if entry.is_dir():
                for path in sorted(entry.iterdir()):
                    if path.is_file():
                        source = _describe(path, entry.name, path.name)
                        if source is not None:
                            found.append(source)
            elif entry.is_file() and entry.suffix in {".log", ".txt", ".jsonl"}:
                source = _describe(entry, UNSORTED, entry.name)
                if source is not None:
                    found.append(source)
    runs, preferred = _run_sources()
    found.extend(runs)
    return found, preferred


def index() -> dict[str, Any]:
    """Every readable log, newest first, and which one to open by default.

    The default is the CURRENT job's output -- the running one, or the most
    recently started if none is running. Only when there is no job at all does
    it fall back to the newest file, so an idle console still opens something.
    """

    found, preferred = _sources()
    found.sort(key=lambda item: item.modified, reverse=True)
    if preferred is None and found:
        preferred = found[0].id
    return {
        "sources": [item.as_dict() for item in found],
        "categories": sorted({item.category for item in found}),
        "default": preferred,
        "directory": str(LOG_ROOT),
    }


def _resolve(source_id: str) -> LogSource | None:
    if not _SOURCE_ID.fullmatch(source_id or "") or ".." in source_id:
        raise ValueError("invalid log source id")
    found, _ = _sources()
    return next((item for item in found if item.id == source_id), None)


def tail(source_id: str, lines: int = DEFAULT_LINES) -> dict[str, Any]:
    """Last ``lines`` lines of one source, read from the end of the file."""

    source = _resolve(source_id)
    if source is None:
        raise FileNotFoundError(f"unknown log source {source_id!r}")
    count = max(1, min(int(lines), MAXIMUM_LINES))
    try:
        with source.path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - TAIL_BYTES))
            data = handle.read()
    except OSError as error:
        # A locked or vanished log is a state to report, not a request failure.
        return {
            **source.as_dict(),
            "lines": [],
            "truncated": False,
            "error": f"{type(error).__name__}: {error}",
        }
    text = data.decode("utf-8", errors="replace")
    # A partial first line is an artefact of seeking into the middle of the file.
    if size > TAIL_BYTES:
        _, _, text = text.partition("\n")
    rows = text.splitlines()
    return {
        **source.as_dict(),
        "lines": rows[-count:],
        "truncated": size > TAIL_BYTES or len(rows) > count,
        "error": None,
    }
