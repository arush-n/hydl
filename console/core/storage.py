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

## Every area has now moved

Completed 2026-08-23: all seven remaining areas were relocated under
`STORAGE_ROOT` and `_NOT_YET_MOVED` is empty. 41,159 files, renamed on the same
volume with the console stopped, file counts checked equal on both sides.

No module outside this one was edited to do it, which was the point of routing
them all through here -- with two exceptions that had been spelling a legacy
path themselves rather than asking:
`console/core/catalog/identities.py` (native-server logs) and
`adk/probes/crosslang.py` (bridge jar discovery). Both now span the move.

Legacy roots are still listed in `_LEGACY` and still searched by `read_roots`,
so a path recorded before the move -- a source checkpoint in an old launch
spec, a bookmarked artifact -- continues to resolve.
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

#: Areas still served from their legacy root. **Empty: the migration is done.**
#:
#: Completed 2026-08-23 with the console stopped, by renaming each directory on
#: the same volume -- 41,159 files across seven areas, file counts verified
#: equal on both sides of every move. `policies` went first because it lived
#: INSIDE `runs` (`artifacts/console/init-policies`) and would otherwise have
#: travelled with it.
#:
#: The entry for an area must be removed in the SAME change that moves its
#: contents. Removing it early points writes at an empty directory while the
#: history stays behind; removing it late writes new artifacts into the location
#: just vacated. Either way the store splits, which is the failure this module
#: exists to end.
#:
#: `read_roots` still lists every legacy path after the current one, so a
#: reference held from before the move -- a checkpoint path in an old run's
#: spec, a bookmarked artifact -- still resolves.
_NOT_YET_MOVED: frozenset[str] = frozenset()

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
