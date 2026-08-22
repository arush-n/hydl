"""World libraries discovered from a folder, and one query that selects from them.

A world library is a directory holding a ``manifest.json`` that declares the
region-artifact schema. Adding a library is adding a directory: nothing here
names one, so a library appears because it exists and disappears when it does
not.

Selection is a *query*, not a flag. It used to be a family of booleans -- one
per heuristic, each threaded through the run config, the API schema, two
forwarding sites and the replay stamp -- so every new intent cost five edits and
the set of intents was closed. Here the ordering is data:

    select(count=16, order="flattest")

``order`` names a function in :data:`ORDERS`, so a new ordering is a new entry
in one mapping.

Nothing in this package may import ``adk`` or ``arena``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Sequence

from worlds import status
from worlds.terrain import LIBRARY

#: A directory is a world library when its manifest declares this schema.
LIBRARY_SCHEMA = "hytalerl_region_artifact_library_v1"
#: Where libraries live. The default library sits inside it, so derive the root
#: from that rather than restating a path that would then have two sources.
LIBRARY_ROOT = LIBRARY.parent
MANIFEST = "manifest.json"
#: How the loader ranks a split when no ordering is asked for. Mirrors
#: ``RegionArtifactLibrary.select``; keep in step with it.
SPLIT_METHOD = "sha256_rank_v1"
DEFAULT_SPLIT = "train"


@dataclass(frozen=True)
class WorldLibrary:
    """One discovered library and the counts a caller needs to choose it."""

    id: str
    path: Path
    semantic_sha256: str
    counts: dict[str, int]

    @property
    def is_default(self) -> bool:
        return self.path == LIBRARY

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "semantic_sha256": self.semantic_sha256,
            "counts": dict(self.counts),
            "default": self.is_default,
        }


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        manifest = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    if manifest.get("schema") != LIBRARY_SCHEMA:
        return None
    return manifest


@lru_cache(maxsize=1)
def discover(root: Path | None = None) -> tuple[WorldLibrary, ...]:
    """Every world library under ``root``, newest-looking name last.

    Cheap: reads one manifest per candidate directory and nothing else.
    """

    base = Path(root) if root is not None else LIBRARY_ROOT
    if not base.is_dir():
        return ()
    found: list[WorldLibrary] = []
    for directory in sorted(base.iterdir()):
        if not directory.is_dir():
            continue
        manifest = _read_manifest(directory)
        if manifest is None:
            continue
        counts: dict[str, int] = {}
        for entry in manifest.get("entries") or ():
            split = str(entry.get("split") or "unsplit")
            counts[split] = counts.get(split, 0) + 1
        found.append(
            WorldLibrary(
                id=directory.name,
                path=directory,
                semantic_sha256=str(manifest.get("library_semantic_sha256") or ""),
                counts=counts,
            )
        )
    return tuple(found)


def get(library_id: str | None) -> WorldLibrary | None:
    """Resolve a library by id. ``None`` asks for the default."""

    libraries = discover()
    if library_id is None:
        return next((item for item in libraries if item.is_default), None)
    return next((item for item in libraries if item.id == library_id), None)


@lru_cache(maxsize=8)
def _entries(library_id: str, split: str) -> tuple[tuple[int, str], ...]:
    """``(seed, semantic_sha256)`` for one split of one library."""

    library = get(library_id)
    if library is None:
        return ()
    manifest = _read_manifest(library.path) or {}
    return tuple(
        (int(entry["seed"]), str(entry["semantic_sha256"]))
        for entry in manifest.get("entries") or ()
        if entry.get("split") == split
    )


def _digest_order(entries: Sequence[tuple[int, str]], key: int) -> list[int]:
    """The loader's own ranking: a keyed digest over the split."""

    ranked = sorted(
        entries,
        key=lambda entry: (
            sha256(f"{SPLIT_METHOD}\0{key}\0{entry[1]}".encode()).digest(),
            entry[1],
        ),
    )
    return [seed for seed, _ in ranked]


def _flatness(seeds: Sequence[int]) -> dict[int, float]:
    """``open_flat`` share of each seed's largest zone, its likely arena."""

    from worlds import zones

    largest: dict[int, Any] = {}
    for zone in zones.library():
        if zone.seed not in largest or zone.size > largest[zone.seed].size:
            largest[zone.seed] = zone
    wanted = set(seeds)
    return {
        seed: zone.terrain.get("open_flat", 0) / max(zone.size, 1)
        for seed, zone in largest.items()
        if seed in wanted
    }


def _flattest_order(entries: Sequence[tuple[int, str]], key: int) -> list[int]:
    """Flattest first. The curriculum floor: ground an agent can walk on.

    The digest ordering is a hash rank, so it is indifferent to terrain and can
    return a set well below the library's median flatness.
    """

    seeds = [seed for seed, _ in entries]
    flatness = _flatness(seeds)
    # Ties broken by seed so a set is reproducible, not census-order dependent.
    return sorted(seeds, key=lambda seed: (-flatness.get(seed, 0.0), seed))


def _mixed_order(entries: Sequence[tuple[int, str]], key: int) -> list[int]:
    """Alternate flattest and digest-ranked: legible ground plus terrain to generalise to.

    INTERLEAVED rather than "first half flat, then the rest", so that ANY prefix
    is about half of each. A block layout only balances at the one length it was
    computed for: taking the flat half of the whole split and then slicing the
    caller's count off the front returns nothing but flat worlds, which is
    exactly the bug this replaced.
    """

    flat = _flattest_order(entries, key)
    ranked = _digest_order(entries, key)
    chosen: list[int] = []
    seen: set[int] = set()
    for flat_seed, ranked_seed in zip(flat, ranked):
        for seed in (flat_seed, ranked_seed):
            if seed not in seen:
                chosen.append(seed)
                seen.add(seed)
    for seed in (*flat, *ranked):
        if seed not in seen:
            chosen.append(seed)
            seen.add(seed)
    return chosen


#: Orderings a caller may ask for. A new ordering is one entry here, not a new
#: boolean threaded through five files.
ORDERS: dict[str, Callable[[Sequence[tuple[int, str]], int], list[int]]] = {
    "digest": _digest_order,
    "flattest": _flattest_order,
    "mixed": _mixed_order,
}


def select(
    count: int,
    *,
    order: str = "digest",
    key: int = 0,
    library: str | None = None,
    split: str = DEFAULT_SPLIT,
) -> tuple[int, ...]:
    """``count`` world seeds from one library, ordered as asked.

    Raises rather than silently returning a short set: a caller that asked for
    sixteen worlds and got four would spread its batch over the wrong number and
    report the result as if it had what it requested.
    """

    if order not in ORDERS:
        raise ValueError(
            f"unknown world order {order!r}; available: {', '.join(sorted(ORDERS))}"
        )
    resolved = get(library)
    if resolved is None:
        available = ", ".join(item.id for item in discover()) or "none discovered"
        raise ValueError(f"unknown world library {library!r}; available: {available}")
    entries = _entries(resolved.id, split)
    # Deprecated worlds are dropped before the count is checked, so a request
    # that can no longer be met raises rather than quietly returning a short
    # set -- the same rule the split-size check below already enforces.
    excluded = status.deprecated_seeds(resolved.id)
    if excluded:
        entries = tuple(entry for entry in entries if entry[0] not in excluded)
    if len(entries) < count:
        deprecated = f", {len(excluded)} deprecated" if excluded else ""
        raise ValueError(
            f"library {resolved.id!r} split {split!r} has {len(entries)} worlds"
            f"{deprecated}, fewer than the {count} requested"
        )
    return tuple(ORDERS[order](entries, key)[:count])


def inventory(
    library: str | None = None,
    split: str = DEFAULT_SPLIT,
    *,
    live: Sequence[int] = (),
    used: Sequence[int] = (),
) -> dict[str, Any]:
    """Every world in one split, with what is known about it and its status.

    This is the read side of keep/deprecate: one row per world, carrying the
    terrain measure a decision is usually made on and whether a decision has
    already been recorded. `flatness` is None when the terrain census has no
    zone for that seed, which is itself worth seeing.

    Each world is also placed in one of three groups, which is what makes a
    deprecation decision safe to make:

    ``live``
        in the working set of the run happening now
    ``used``
        loaded by some earlier run, so a result already depends on it
    ``unused``
        in the shared library and never selected -- the safe things to
        deprecate, and the ones worth looking at first

    Both sets are supplied by the caller: this package knows the library, not
    which runs exist.
    """

    resolved = get(library)
    if resolved is None:
        return {"library": None, "split": split, "worlds": []}
    live_seeds = {int(seed) for seed in live}
    used_seeds = {int(seed) for seed in used} | live_seeds
    entries = _entries(resolved.id, split)
    flatness = _flatness([seed for seed, _ in entries])
    recorded = status.all_statuses()
    worlds = []
    for seed, semantic in entries:
        decision = recorded.get(f"{resolved.id}/{seed}")
        worlds.append(
            {
                "seed": seed,
                "semantic_sha256": semantic,
                "flatness": flatness.get(seed),
                "usage": (
                    "live"
                    if seed in live_seeds
                    else "used"
                    if seed in used_seeds
                    else "unused"
                ),
                "state": decision.state if decision else None,
                "reason": decision.reason if decision else "",
                "decided_unix_seconds": (
                    decision.decided_unix_seconds if decision else None
                ),
                # A decision recorded against a different hash was made about a
                # world this seed no longer names.
                "stale_decision": bool(
                    decision
                    and decision.semantic_sha256
                    and decision.semantic_sha256 != semantic
                ),
            }
        )
    worlds.sort(key=lambda row: row["seed"])
    return {
        "library": resolved.as_dict(),
        # The shared library this list is read from, so a reader can see which
        # folder a decision applies to rather than assuming.
        "directory": str(resolved.path),
        "split": split,
        "splits": sorted(resolved.counts),
        "worlds": worlds,
        "sections": {
            group: [row["seed"] for row in worlds if row["usage"] == group]
            for group in ("live", "used", "unused")
        },
        "deprecated": sum(1 for row in worlds if row["state"] == status.DEPRECATED),
        "kept": sum(1 for row in worlds if row["state"] == status.KEPT),
    }


def describe() -> dict[str, Any]:
    """Everything a client needs to offer world selection, discovered live."""

    return {
        "libraries": [item.as_dict() for item in discover()],
        "orders": sorted(ORDERS),
        "default_split": DEFAULT_SPLIT,
    }
