"""One place that decides where the console keeps things.

Before this module every writer computed its own root, and they had drifted into
ten of them: `artifacts/console`, `artifacts/console-checks`,
`artifacts/console-hytale`, `artifacts/console-streams`,
`artifacts/console-dispatch`, `artifacts/worlds`, `logs`, `agents/profiles`, and
two directories under `.codex-local/console`. Nothing was wrong with any single
one; the problem was that "where does the console put X" had ten answers and no
way to enumerate them.

The layout is one root with named areas:

    storage/
      runs/       run artifacts -- report.json, trajectory.json, metrics.jsonl
      worlds/     designs, minigames, captured previews
      policies/   minted initial policies
      presets/    training profiles and run presets
      logs/       server logs and stream captures
      checks/     check results
      dispatch/   cross-boundary scratch for WSL dispatch
      hytale/     native server runtime state

**Adding a new kind of stored thing means adding an area here**, not a fresh
`_ROOT / "artifacts" / "console-something"` in the module that needed it.

## Reads fall back, writes do not

Every area records where it used to live. `read_roots` returns the new location
first and the legacy ones after, so an artifact written before the move is still
found; `write_root` returns exactly one directory, so nothing new is ever added
to a legacy location. That asymmetry is the whole migration strategy: the tree
converges as things are rewritten, and nothing has to be moved to keep working.

## `runs` has not switched yet, deliberately

`artifacts/console/` holds 2.9 GB across 4,212 entries and a training job writes
into it while the console is up. Repointing it is a data migration, not a
constant change, so `RUNS` still resolves to the legacy root and says so. When
it moves, `store.ARTIFACT_ROOT` becomes `write_root("runs")` and the old path
stays in `LEGACY` -- no other module should need editing, which is the point of
routing them all through here.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Derived from THIS FILE's location, never from the working directory or a
#: literal path, so a checkout works wherever it is cloned and whatever the
#: process was launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: The single root every new console artifact belongs under.
#:
#: `HYTALERL_STORAGE_ROOT` overrides it, which is what makes this deployable
#: rather than merely relocatable: data can live on another disk, on a shared
#: volume, or in a container mount without editing source. An override is
#: resolved so a relative value cannot silently follow the working directory.
STORAGE_ROOT = (
    Path(os.environ["HYTALERL_STORAGE_ROOT"]).expanduser().resolve()
    if os.environ.get("HYTALERL_STORAGE_ROOT")
    else PROJECT_ROOT / "storage"
)

#: Areas, and where each one used to be written. The first entry of a tuple is
#: the oldest; order within the legacy tuple does not matter because reads try
#: all of them, but keeping them chronological documents the moves.
_LEGACY: dict[str, tuple[Path, ...]] = {
    "runs": (PROJECT_ROOT / "artifacts" / "console",),
    "worlds": (
        PROJECT_ROOT / "artifacts" / "worlds",
        PROJECT_ROOT / ".codex-local" / "console" / "worldgen-designs",
        PROJECT_ROOT / ".codex-local" / "console" / "worldgen-builds",
        PROJECT_ROOT / ".codex-local" / "console" / "custom-minigames",
    ),
    "policies": (PROJECT_ROOT / "artifacts" / "console" / "init-policies",),
    # NO `presets` AREA. It used to map to `agents/profiles/`, which is not an
    # artifact directory at all -- those nine files are published source and
    # ship in the repository. Migrating that area would have deleted source out
    # of the tree into a gitignored store. An agent profile is authored, not
    # produced, so it belongs beside the agent that declares it.
    "logs": (
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "artifacts" / "console-streams",
    ),
    "checks": (PROJECT_ROOT / "artifacts" / "console-checks",),
    # New in this layout, so it has no legacy root and is not in
    # `_NOT_YET_MOVED`: it writes to `storage/ladder` from its first byte.
    # This is the case the module was built for -- a new kind of stored thing
    # goes to an area here rather than to an eleventh ad-hoc directory.
    "ladder": (),
    "dispatch": (PROJECT_ROOT / "artifacts" / "console-dispatch",),
    "hytale": (PROJECT_ROOT / "artifacts" / "console-hytale",),
}

#: Areas still served from their legacy root, because relocating one is a data
#: migration rather than a constant change. Remove an entry here in the SAME
#: change that moves its contents -- flipping it early splits the store, which
#: is the failure this module exists to end.
#:
#: Today that is all of them, and that is deliberate. `artifacts/worlds` is
#: written by `worlds/scene_cache.py` during a live Region run, `artifacts/
#: console` is 2.9 GB with a training job writing into it, and none of the rest
#: is worth a move on its own. What this module changes NOW is where the next
#: new kind of artifact goes: `write_root` for an area absent from this set
#: returns `storage/<area>`, so nothing new lands in an eleventh ad-hoc root.
#:
#: Migration order when the device is idle, cheapest and least-referenced first:
#: `checks` (1 KB), `dispatch` (14 KB), `logs` (89 KB), `presets` (24 KB),
#: `worlds` (73 KB), `hytale` (163 MB), `policies`, then `runs` (2.9 GB) last.
_NOT_YET_MOVED = frozenset(
    {"runs", "policies", "worlds", "logs", "checks", "dispatch", "hytale"}
)

AREAS = tuple(sorted(_LEGACY))


def _known(area: str) -> str:
    if area not in _LEGACY:
        raise ValueError(
            f"unknown storage area {area!r}; have: {', '.join(AREAS)}"
        )
    return area


def write_root(area: str, *, create: bool = False) -> Path:
    """The one directory new artifacts of this kind go in."""

    # Validated before the branch, not inside it: putting `_known` in the
    # legacy arm let an unknown area skip the check entirely and silently mint
    # `storage/<typo>`.
    _known(area)
    root = _LEGACY[area][0] if area in _NOT_YET_MOVED else STORAGE_ROOT / area
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def read_roots(area: str) -> tuple[Path, ...]:
    """Every directory to look in, newest layout first.

    A caller resolving a name should take the first hit. Reading in this order
    is what makes a record that exists in both places resolve to the current
    one rather than the copy left behind.
    """

    write = write_root(area)
    legacy = tuple(path for path in _LEGACY[area] if path != write)
    return (write, *legacy)


def locate(area: str, name: str) -> Path | None:
    """Find `name` in an area, preferring the current layout. None if absent.

    `name` is treated as a single path component. A caller that needs nesting
    should join under the returned root rather than passing a path here, so a
    name can never walk out of the area it was looked up in.
    """

    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError(f"{name!r} is not a single storage name")
    for root in read_roots(area):
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def describe() -> dict[str, object]:
    """What is where, for the Compute tab and for answering the question fast."""

    return {
        "schema": "console-storage-layout-v1",
        "root": str(STORAGE_ROOT),
        "areas": {
            area: {
                "writes_to": str(write_root(area)),
                "reads_from": [str(path) for path in read_roots(area)],
                "migrated": area not in _NOT_YET_MOVED,
            }
            for area in AREAS
        },
    }


__all__ = [
    "AREAS",
    "PROJECT_ROOT",
    "STORAGE_ROOT",
    "describe",
    "locate",
    "read_roots",
    "write_root",
]
