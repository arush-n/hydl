"""Run logging: one directory per run, sorted and queryable.

Every measurement in this project so far has landed in an ad-hoc scratch file
with a name like `ttk2.log`, and the result was predictable -- a completed run
whose output nobody read for hours, and numbers quoted without the scene that
produced them. Two rules follow from that, and this module enforces both:

  **A result is never stored without its provenance.** `config.json` holds the
  full `SceneConfig` provenance dict, so a number can always answer "which
  weapon, which world, which opponent distance". A metric without that is not
  interpretable -- the same policy scores 0 deaths or 4 deaths purely on the
  opponent standoff.

  **Metrics stream to disk as they are produced.** `metrics.jsonl` is appended
  and flushed per row, so a run that crashes at update 40 still has 39 rows.

    from adk.runs import Run
    with Run.create("duelist-daggers", scene.provenance) as run:
        run.log(update=1, mean_episode_return=12.5, episodes=8)
    Run.index()          # every run, newest first
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
__all__ = ["Run", "ROOT"]


@dataclass
class Run:
    directory: Path
    name: str
    started: float

    # --- creating ----------------------------------------------------------

    @classmethod
    def create(cls, name: str, provenance: dict[str, Any] | None = None) -> "Run":
        """Open a run directory. Timestamp-first so `sorted()` is chronological."""

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in name)
        directory = ROOT / f"{stamp}-{safe}"
        suffix = 1
        while directory.exists():          # two runs in the same second
            directory = ROOT / f"{stamp}-{safe}-{suffix}"
            suffix += 1
        directory.mkdir(parents=True)
        run = cls(directory=directory, name=name, started=time.time())
        (directory / "config.json").write_text(
            json.dumps({
                "name": name,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "provenance": _plain(provenance or {}),
            }, indent=1), encoding="utf-8")
        return run

    def log(self, **row: Any) -> None:
        """Append one metric row. Flushed immediately -- a crash keeps the rest."""

        row.setdefault("elapsed_seconds", round(time.time() - self.started, 3))
        with (self.directory / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_plain(row)) + "\n")
            handle.flush()

    def finish(self, **summary: Any) -> None:
        summary.setdefault("elapsed_seconds", round(time.time() - self.started, 3))
        (self.directory / "summary.json").write_text(
            json.dumps(_plain(summary), indent=1), encoding="utf-8")

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if not (self.directory / "summary.json").exists():
            # A run that died still gets a summary saying so, rather than
            # looking indistinguishable from one still in progress.
            self.finish(status="failed" if exc_type else "completed",
                        error=None if exc is None else f"{exc_type.__name__}: {exc}")

    # --- reading -----------------------------------------------------------

    @staticmethod
    def index(*, sort: str = "newest", **match: Any) -> list[dict[str, Any]]:
        """Every run as a flat record. `sort`: newest | oldest | name.

        Filter on any provenance key: `Run.index(weapon="iron_daggers")`.
        """

        records = []
        for directory in ROOT.iterdir():
            if not directory.is_dir() or directory.name.startswith("_"):
                continue
            config = directory / "config.json"
            if not config.is_file():
                continue
            payload = json.loads(config.read_text(encoding="utf-8"))
            provenance = payload.get("provenance", {})
            if any(provenance.get(k) != v for k, v in match.items()):
                continue
            summary = directory / "summary.json"
            records.append({
                "directory": directory.name,
                "name": payload.get("name"),
                "started_utc": payload.get("started_utc"),
                "rows": sum(1 for _ in _rows(directory)),
                **{f"cfg.{k}": v for k, v in provenance.items()},
                **(json.loads(summary.read_text(encoding="utf-8"))
                   if summary.is_file() else {"status": "in_progress"}),
            })
        keys = {
            "newest": lambda r: r["directory"],
            "oldest": lambda r: r["directory"],
            "name": lambda r: (r["name"] or "", r["directory"]),
        }
        if sort not in keys:
            raise ValueError(f"unknown sort {sort!r}; have {sorted(keys)}")
        return sorted(records, key=keys[sort], reverse=(sort == "newest"))

    @staticmethod
    def metrics(directory: str) -> list[dict[str, Any]]:
        return list(_rows(ROOT / directory))


def _rows(directory: Path) -> Iterator[dict[str, Any]]:
    path = directory / "metrics.jsonl"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _plain(value: Any) -> Any:
    """JSON-safe. numpy/jax scalars are the usual thing that breaks a dump."""

    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)
