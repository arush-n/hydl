"""Keep or deprecate individual worlds, and have the selectors honour it.

A world library is captured in bulk, so some of what it contains is unusable --
terrain an agent cannot traverse, spawns with no line to the target, geometry
that turned out degenerate. Deleting those entries would rewrite the library and
change its semantic hash, invalidating every result recorded against it.

So a world is *deprecated* instead: the library is untouched and a separate
record says which seeds not to select. `worlds.libraries.select` reads this, so
deprecating a world removes it from future working sets without editing the
artifact it names or the runs that already used it.

The record keys on the world's semantic hash as well as its seed. A seed is
only meaningful inside one library, and if a library is regenerated the same
seed is a different world -- one that has not been judged.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = _ROOT / "artifacts" / "worlds" / "world-status.json"
SCHEMA = "hytalerl_world_status_v1"

KEPT = "kept"
DEPRECATED = "deprecated"
STATES = (KEPT, DEPRECATED)


@dataclass(frozen=True)
class WorldStatus:
    library_id: str
    seed: int
    state: str
    reason: str
    semantic_sha256: str
    decided_unix_seconds: float

    @property
    def key(self) -> str:
        return f"{self.library_id}/{self.seed}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "library_id": self.library_id,
            "seed": self.seed,
            "state": self.state,
            "reason": self.reason,
            "semantic_sha256": self.semantic_sha256,
            "decided_unix_seconds": self.decided_unix_seconds,
        }


def _read() -> dict[str, Any]:
    try:
        document = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        return {}
    entries = document.get("entries")
    return entries if isinstance(entries, dict) else {}


def _write(entries: dict[str, Any]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = {"schema": SCHEMA, "entries": entries}
    # Written via a temporary file: a partial status file would read as an
    # empty one, silently un-deprecating every world it recorded.
    handle, temporary = tempfile.mkstemp(dir=STATUS_PATH.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
        os.replace(temporary, STATUS_PATH)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def all_statuses() -> dict[str, WorldStatus]:
    """Every recorded decision, keyed ``<library id>/<seed>``."""

    found: dict[str, WorldStatus] = {}
    for key, entry in _read().items():
        if not isinstance(entry, dict) or entry.get("state") not in STATES:
            continue
        library_id, _, seed = str(key).rpartition("/")
        if not library_id or not seed.lstrip("-").isdigit():
            continue
        found[key] = WorldStatus(
            library_id=library_id,
            seed=int(seed),
            state=str(entry["state"]),
            reason=str(entry.get("reason") or ""),
            semantic_sha256=str(entry.get("semantic_sha256") or ""),
            decided_unix_seconds=float(entry.get("decided_unix_seconds") or 0.0),
        )
    return found


def set_state(
    library_id: str,
    seed: int,
    state: str,
    *,
    reason: str = "",
    semantic_sha256: str = "",
) -> WorldStatus:
    """Record a decision about one world. ``kept`` is also worth recording:
    it is the difference between "judged and fine" and "never looked at"."""

    if state not in STATES:
        raise ValueError(f"unknown world state {state!r}; expected one of {STATES}")
    record = WorldStatus(
        library_id=str(library_id),
        seed=int(seed),
        state=state,
        reason=str(reason),
        semantic_sha256=str(semantic_sha256),
        decided_unix_seconds=time(),
    )
    entries = _read()
    entries[record.key] = record.as_dict()
    _write(entries)
    deprecated_seeds.cache_clear()
    return record


def clear(library_id: str, seed: int) -> bool:
    """Forget a decision, returning it to unjudged. True if one was removed."""

    entries = _read()
    if entries.pop(f"{library_id}/{int(seed)}", None) is None:
        return False
    _write(entries)
    deprecated_seeds.cache_clear()
    return True


def _deprecated_for(library_id: str) -> frozenset[int]:
    return frozenset(
        status.seed
        for status in all_statuses().values()
        if status.library_id == library_id and status.state == DEPRECATED
    )


class _DeprecatedSeeds:
    """A tiny memo so a selector can ask on every call without re-reading."""

    def __init__(self) -> None:
        self._cache: dict[str, frozenset[int]] = {}

    def __call__(self, library_id: str) -> frozenset[int]:
        if library_id not in self._cache:
            self._cache[library_id] = _deprecated_for(library_id)
        return self._cache[library_id]

    def cache_clear(self) -> None:
        self._cache.clear()


#: Seeds to exclude from selection, by library id.
deprecated_seeds = _DeprecatedSeeds()


__all__ = [
    "DEPRECATED",
    "KEPT",
    "SCHEMA",
    "STATES",
    "STATUS_PATH",
    "WorldStatus",
    "all_statuses",
    "clear",
    "deprecated_seeds",
    "set_state",
]
