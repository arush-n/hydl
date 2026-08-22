"""Persist each console run as a self-describing artifact.

The console used to be fire-and-forget: a run executed, drew itself, and left
nothing behind. That is fine for a single look and useless for the question this
project actually keeps hitting -- *was this number produced against the same
world as that one?* Bridge identity moved six times in one day; the observation
width went 7818 -> 8259 -> 8271 and nothing noticed.

So this writes the record `artifacts/DEV.md` already specifies, rather than
inventing a schema. Two primitives already exist and neither is reimplemented
here:

* :mod:`adk.contracts.stamp` -- ``current_stamp()`` gives the nine contract
  hashes, the action head sizes and the observation width the run was produced
  against.
* :mod:`adk.validation.identity` -- ``capture()`` gives the Gym source digest
  (a hash over every ``.py`` the Gym will execute, because a git SHA reads clean
  over a dirty tree) and ``assert_stable()`` voids a run whose identity moved
  *while it was measuring*.

The stability check is the part worth having. A run whose contracts changed
underneath it is not a bad result, it is a void one, and it has to be labelled
class 4 (stale identity) rather than filed as a fidelity mismatch.
"""

from __future__ import annotations

from console.core import storage

import json
import platform
import re
import shutil
import time
from dataclasses import asdict
from datetime import datetime, timezone
from fnmatch import fnmatch
from hashlib import sha256
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
#: One directory per run. Resolved through `storage` so the `runs` area is
#: the single place that decides where runs live; unchanged until that area
#: is migrated, which is a 2.9 GB data move rather than a constant change.
ARTIFACT_ROOT = storage.write_root("runs")

#: New artifacts owned by a long-lived agent live in one stable namespace.
#: The old flat layout remains readable indefinitely; see :func:`directories`
#: and :func:`locate_directory`.
SHARED_DIRECTORY = "shared"
#: `basic` is named here because the console's own training path declares it as
#: a default owner (see `jobs.py`), not because the archive knows anything about
#: that agent. Every OTHER owner is discovered -- see `_agent_namespaces`.
BASIC_SHARED_AGENT = "basic"

#: `report.json` is the source of truth and stays small enough to read. The
#: trajectory is thousands of steps wide, so it lives beside it as payload.
REPORT = "report.json"
#: How many directories `index` reads per requested row. One run writes a
#: replay-lane directory per archived lane per evaluated update, each with its
#: own report and the run's timestamp prefix, so the newest names are mostly
#: lanes. This is the headroom that keeps a page of them from crowding out real
#: runs; raise it if the archive lane count per run grows.
_INDEX_SCAN_MARGIN = 4
#: A run directory carries its timestamp but not always at the front --
#: `20260821T085656Z-...`, `basic-20260821T085656Z-...` and
#: `basic-ppo-20260815T045733Z-...` are all in the store. Sorting on the raw
#: name puts every letter-prefixed directory above every digit-prefixed one
#: regardless of date, so recency has to come from the stamp itself.
_INDEX_STAMP = re.compile(r"\d{8}T\d{6}Z")
#: A milestone lane replay of a parent job: `<job>-u0064-l003`.
_LANE_SUFFIX = re.compile(r"-u\d{4}-l\d{3}$")


def _index_recency(name: str) -> str:
    """Sort key that orders a run directory by when it was written.

    Stamped names sort above every unstamped one. Some legacy directories carry
    no stamp at all (`dawn-vision-auto-v1-r4-b64-seed577001`), and falling back
    to the bare name puts them at the very top, because every letter sorts above
    every digit. They are the oldest runs in the store, so ranking them last is
    both cheap and right -- and the only alternative, reading each report for its
    `written_at`, is the cost this key exists to avoid.
    """

    found = _INDEX_STAMP.search(name)
    return f"1{found.group(0)}" if found else f"0{name}"
PAYLOAD = "trajectory.json"


def _component(value: Any, *, label: str) -> str:
    """Return a filesystem-safe archive component or fail closed."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    if not value.replace("-", "").replace("_", "").isalnum():
        raise ValueError(
            f"invalid {label} {value!r}: letters, digits, dash and underscore only"
        )
    return value


def _agent_namespaces() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """``(agent_id, stream globs)`` for every agent discovered under ``agents/``.

    Agents declare themselves; the archive does not keep a list of them. An
    agent's ``console/agent.json`` may name ``streams`` globs, and an agent that
    declares none still matches its own id and the ``<id>-`` prefix, so adding a
    directory is enough to get a namespace.
    """

    try:
        from console.core.catalog import agents as agent_catalog

        entries = agent_catalog.index()
    except Exception:  # noqa: BLE001 - a console without the catalog still archives
        return ()
    found: list[tuple[str, tuple[str, ...]]] = []
    for entry in entries:
        agent_id = str(entry.get("id") or "")
        if not agent_id:
            continue
        globs = tuple(str(item) for item in (entry.get("streams") or ()))
        found.append((agent_id, globs or (f"{agent_id}-*",)))
    return tuple(found)


def _claims(agent_id: str, globs: tuple[str, ...], value: str) -> bool:
    if not value:
        return False
    if value == agent_id or value.startswith((f"{agent_id}-", f"{agent_id}_")):
        return True
    return any(fnmatch(value, glob) for glob in globs)


def shared_agent_for(spec: dict[str, Any] | None) -> str | None:
    """Resolve the stable archive owner for a run.

    ``shared_agent`` is the explicit contract. Otherwise the owner is matched
    against agents DISCOVERED under ``agents/`` -- by profile id, or by the
    ``kind``/``schema`` an agent stamps on its own receipts. No agent is named
    here: an agent appears because its directory exists, and disappears when it
    does not. Unrelated Console runs remain in the legacy flat root.
    """

    value = dict(spec or {})
    explicit = value.get("shared_agent")
    if explicit is not None:
        return _component(explicit, label="shared agent")
    owners = [str(owner) for owner in (value.get("profile_ids") or [])]
    if value.get("profile_id"):
        owners.append(str(value["profile_id"]))
    kind = str(value.get("kind") or value.get("schema") or "")
    for agent_id, globs in _agent_namespaces():
        if any(_claims(agent_id, globs, owner) for owner in owners):
            return agent_id
        if _claims(agent_id, globs, kind):
            return agent_id
    if value.get("training_stage") == "pursuit_tracking":
        return BASIC_SHARED_AGENT
    return None


def _profile_ids(spec: dict[str, Any], shared_agent: str | None) -> list[str]:
    owners: list[str] = []
    for owner in [
        *(spec.get("profile_ids") or []),
        spec.get("profile_id"),
        shared_agent,
    ]:
        if owner is None:
            continue
        canonical = _component(owner, label="profile id")
        if canonical not in owners:
            owners.append(canonical)
    return owners


def directory_for(
    name: str,
    *,
    spec: dict[str, Any] | None = None,
    shared_agent: str | None = None,
    root: Path | None = None,
) -> Path:
    """Canonical directory for a new artifact without touching the filesystem."""

    run_name = _component(name, label="run id")
    base = Path(root or ARTIFACT_ROOT)
    agent = shared_agent or shared_agent_for(spec)
    if agent is None:
        return base / run_name
    return base / SHARED_DIRECTORY / _component(agent, label="shared agent") / run_name


def directories(root: Path | None = None) -> list[Path]:
    """All canonical and legacy run directories, canonical ones first."""

    base = Path(root or ARTIFACT_ROOT)
    if not base.is_dir():
        return []
    found: list[Path] = []
    shared = base / SHARED_DIRECTORY
    if shared.is_dir():
        for agent in sorted(shared.iterdir()):
            if not agent.is_dir():
                continue
            found.extend(
                directory
                for directory in sorted(agent.iterdir(), reverse=True)
                if directory.is_dir()
            )
    found.extend(
        directory
        for directory in sorted(base.iterdir(), reverse=True)
        if directory.is_dir() and directory.name != SHARED_DIRECTORY
    )
    return found


def locate_directory(name: str, root: Path | None = None) -> Path | None:
    """Resolve a run ID across the shared namespace and legacy flat layout."""

    run_name = _component(name, label="run id")
    base = Path(root or ARTIFACT_ROOT)
    shared = base / SHARED_DIRECTORY
    if shared.is_dir():
        for agent in sorted(shared.iterdir()):
            candidate = agent / run_name
            if candidate.is_dir():
                return candidate
    legacy = base / run_name
    return legacy if legacy.is_dir() else None


def remove(name: str, root: Path | None = None) -> dict[str, Any] | None:
    """Delete one stored run and everything under it. None if it is not there.

    Resolution goes through `locate_directory`, so the name is validated by
    `_component` and can only land inside the archive -- a run id is a path
    component here, and this is the one function where letting one escape would
    delete something outside it. The resolved path is re-checked against the
    root anyway, because a symlinked run directory would otherwise carry the
    delete out of the tree.

    Milestone lane replays (`<job>-u####-l###`) are separate archive entries and
    are NOT swept up by deleting their parent job; they are deleted the same way
    they are listed, one at a time.
    """

    base = Path(root or ARTIFACT_ROOT).resolve()
    directory = locate_directory(name, root)
    if directory is None:
        return None
    resolved = directory.resolve()
    if not resolved.is_relative_to(base):
        raise ValueError(f"refusing to delete {name!r}: it resolves outside the archive")
    files = sum(1 for item in resolved.rglob("*") if item.is_file())
    size = sum(item.stat().st_size for item in resolved.rglob("*") if item.is_file())
    shutil.rmtree(resolved)
    return {
        "run_id": name,
        "path": str(directory),
        "files_removed": files,
        "bytes_removed": size,
    }


def republish_owner(
    name: str,
    shared_agent: str,
    *,
    new_name: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Copy one stored artifact into the correct agent namespace.

    Ownership corrections are additive: the source remains resolvable, while
    the copy preserves its original contracts, identity, trajectory, and any
    checkpoint payloads. The receipt makes clear that this is not a new run.
    """

    source = locate_directory(name, root)
    if source is None:
        raise FileNotFoundError(f"unknown run {name!r}")
    owner = _component(shared_agent, label="shared agent")
    target_name = _component(new_name or f"{owner}-{name}", label="run id")
    target = directory_for(
        target_name,
        shared_agent=owner,
        root=root,
    )
    if target.exists():
        raise FileExistsError(f"ownership copy already exists at {target}")
    shutil.copytree(source, target)
    report_path = target / REPORT
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        spec = dict(report.get("spec") or {})
        previous_owner = spec.get("shared_agent")
        owners = [
            value
            for value in spec.get("profile_ids") or []
            if value != previous_owner
        ]
        if owner not in owners:
            owners.append(owner)
        spec.update(shared_agent=owner, profile_ids=owners)
        report.update(run_id=target_name, spec=spec)
        report["ownership_migration"] = {
            "schema": "hytalerl_console_ownership_migration_v1",
            "source_run_id": name,
            "source_path": str(source),
            "target_path": str(target),
            "copied_at": datetime.now(timezone.utc).isoformat(),
            "non_destructive": True,
        }
        report_path.write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(target)
        raise
    return _entry(report, target)


def run_id(spec: dict[str, Any], when: float | None = None) -> str:
    """A sortable, content-addressed name for one run.

    Timestamp first so the directory listing is chronological, then a digest of
    the spec so two runs of the *same* configuration are visibly the same
    configuration and a changed knob is visibly a different one.
    """

    moment = datetime.fromtimestamp(when or time.time(), tz=timezone.utc)
    digest = sha256(
        json.dumps(spec, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8]
    return f"{moment:%Y%m%dT%H%M%SZ}-{digest}"


def _identity_pair():
    """Capture identity, or explain why it is unavailable rather than omitting it.

    A missing identity must be visible in the record. Writing the report without
    it would produce exactly the artifact this module exists to prevent: numbers
    with no statement of what produced them.
    """

    try:
        from adk.validation.identity import assert_stable, capture

        return capture, assert_stable, None
    except Exception as error:  # pragma: no cover - import-environment specific
        return None, None, f"{type(error).__name__}: {error}"


def _contract_stamp(world_geometry_config=None):
    """Serialise the live contract stamp.

    Built field by field rather than with ``dataclasses.asdict``: ``contracts``
    is normalised to a ``mappingproxy``, and ``asdict`` deep-copies, which fails
    with *"cannot pickle 'mappingproxy' object"*. That surfaced here as a report
    with zero contract hashes and a null observation width -- the exact
    uninterpretable artifact this module exists to prevent -- so the failure is
    also no longer swallowed silently below.
    """

    try:
        from adk.contracts.stamp import current_stamp

        stamp = current_stamp(world_geometry_config)
        return {
            "contracts": dict(stamp.contracts),
            "action_head_names": list(stamp.action_head_names),
            "action_head_sizes": list(stamp.action_head_sizes),
            "observation_size": int(stamp.observation_size),
        }, None
    except Exception as error:  # pragma: no cover - import-environment specific
        return None, f"{type(error).__name__}: {error}"


def begin() -> dict[str, Any]:
    """Read identity *before* the run so drift can be detected after it."""

    capture, _assert, unavailable = _identity_pair()
    if capture is None:
        return {"identity_unavailable": unavailable}
    return {"identity": capture()}


def write(
    spec: dict[str, Any],
    result: dict[str, Any],
    opened: dict[str, Any],
    *,
    root: Path | None = None,
    name: str | None = None,
    world_geometry_config=None,
) -> dict[str, Any]:
    """Write one artifact directory and return its index entry.

    ``opened`` is whatever :func:`begin` returned. Identity drift is recorded on
    the run rather than raised: the run already happened, and a void run that
    silently disappears is worse than one labelled void.

    ``name`` pins the directory. A caller that already created one -- a training
    job streaming metrics into it from the moment it started -- must pass its
    id, or the default clock-based name lands the report in a *second*
    directory and the metrics end up with no provenance beside them.
    """

    spec = dict(spec)
    shared_agent = shared_agent_for(spec)
    if shared_agent is not None:
        spec["shared_agent"] = shared_agent
        spec["profile_ids"] = _profile_ids(spec, shared_agent)
    directory = directory_for(
        name or run_id(spec),
        spec=spec,
        shared_agent=shared_agent,
        root=root,
    )
    directory.mkdir(parents=True, exist_ok=True)

    capture, assert_stable, unavailable = _identity_pair()
    before = opened.get("identity")
    identity_drift = None
    after = None
    if capture is not None:
        after = capture()
        if before is not None:
            try:
                assert_stable(before, after)
            except Exception as error:
                identity_drift = str(error)

    contracts, contracts_unavailable = _contract_stamp(world_geometry_config)
    summary = result.get("summary", {})
    warnings_out = list(result.get("warnings", []))

    report = {
        "schema": "hytalerl_console_run_v1",
        "run_id": directory.name,
        "written_at": datetime.now(timezone.utc).isoformat(),
        # What the run was produced against. This is the whole point of the
        # record; the numbers below are only interpretable next to it.
        "contracts": contracts,
        "contracts_unavailable": contracts_unavailable,
        "identity": asdict(after) if after is not None else None,
        "identity_unavailable": unavailable,
        # A run whose identity moved mid-measurement is class 4 (stale
        # identity), not a fidelity mismatch. Say so on the artifact.
        "identity_drift": identity_drift,
        "void": identity_drift is not None,
        "backend": _backend(),
        "host": platform.node(),
        # `artifacts/DEV.md` requires seeds and episode counts, and requires
        # every control arm rather than only the headline. The console runs one
        # arm per request, so the honest field is which arm this was: an inert
        # target or an unperceiving agent IS the control.
        "spec": spec,
        "control_arm": {
            "target_active": spec.get("target_active"),
            "armed": spec.get("armed"),
            "perceive": spec.get("perceive"),
            "opponent": spec.get("opponent"),
        },
        "seed": spec.get("seed"),
        "ticks": spec.get("ticks"),
        "decision_period": spec.get("decision_period"),
        "summary": summary,
        "warnings": warnings_out,
        # A single run is noise on every metric this console reports; the
        # project has refuted its own findings twice by reading one. Recorded so
        # a reader does not have to remember it.
        "single_run_caveat": (
            "one seed and one cadence measure nothing on their own -- sweep "
            "seed and period before treating any number here as a difference"
        ),
    }

    (directory / REPORT).write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    # The terrain reference is seed + semantic hash; world_identity retains the
    # complete pool contract. Cell arrays stay in the immutable capture and are
    # decoded only when a replay is opened.
    payload = {
        key: result[key]
        for key in (
            "steps",
            "head_names",
            "target_phases",
            "events",
            "trajectory",
            "terrain",
            "region",
            "world_identity",
        )
        if key in result
    }
    (directory / PAYLOAD).write_text(
        json.dumps(payload, default=str) + "\n", encoding="utf-8"
    )

    return _entry(report, directory)


def _backend() -> str:
    try:
        import jax

        return jax.default_backend()
    except Exception:  # pragma: no cover - import-environment specific
        return "unknown"


def _entry(report: dict[str, Any], directory: Path) -> dict[str, Any]:
    """The listing row: enough to choose a run without opening it."""

    spec = report.get("spec") or {}
    profile_ids = _profile_ids(spec, spec.get("shared_agent"))
    payload_state = _payload_state(directory)
    return {
        "run_id": report["run_id"],
        "written_at": report["written_at"],
        "profile_id": spec.get("profile_id"),
        "profile_ids": profile_ids,
        "shared_agent": spec.get("shared_agent"),
        "path": str(directory),
        "loadout": spec.get("loadout"),
        "opponent": spec.get("opponent"),
        "policy": spec.get("policy"),
        "world": spec.get("world"),
        "seed": spec.get("seed"),
        "ticks": spec.get("ticks"),
        "void": bool(report.get("void")),
        "warnings": len(report.get("warnings") or []),
        "summary": report.get("summary", {}),
        **payload_state,
        "source_report": spec.get("source_report"),
        "observation_size": (report.get("contracts") or {}).get("observation_size"),
        # Surfaced in the listing, not just buried in the report: a run stored
        # without its contract hashes is not a run you can compare later, and
        # that has to be visible without opening the artifact.
        "contracts_unavailable": report.get("contracts_unavailable"),
        "identity_unavailable": report.get("identity_unavailable"),
    }


def _payload_state(directory: Path) -> dict[str, Any]:
    payload = directory / PAYLOAD
    if not payload.is_file():
        return {
            "replay_available": False, "terrain_available": False,
            "terrain_seed": None, "terrain_semantic_sha256": None,
            "terrain_kind": None, "terrain_nodes": 0,
        }
    try:
        stored = json.loads(payload.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "replay_available": False, "terrain_available": False,
            "terrain_seed": None, "terrain_semantic_sha256": None,
            "terrain_kind": None, "terrain_nodes": 0,
        }
    terrain = stored.get("terrain") or {}
    return {
        "replay_available": bool(stored.get("trajectory")),
        "terrain_available": bool(terrain),
        "terrain_seed": terrain.get("seed"),
        "terrain_semantic_sha256": (
            terrain.get("semantic_sha256")
            or terrain.get("artifact_semantic_sha256")
        ),
        "terrain_kind": terrain.get("kind"),
        "terrain_nodes": int(terrain.get("nodes") or len(terrain.get("x") or ())),
    }


def index(root: Path | None = None, limit: int = 200) -> list[dict[str, Any]]:
    """Every stored run, newest first."""

    found = directories(root)
    # Read only the newest slice, chosen from the LISTING alone. Every run
    # directory is timestamp-prefixed, so the newest are the largest names and
    # the cutoff costs no file I/O. Reading them all cost 13.5s across 8283
    # directories -- and the Archive calls this to draw its list, so the run
    # list and every replay opened from it timed out.
    #
    # The margin over `limit` is deliberate: a run's replay-lane directories
    # share its timestamp prefix and carry their own report, so they sort
    # adjacent to it and collapse into one row through `rows_by_id` below.
    # Without the margin a page of lanes would crowd out real runs.
    scan = sorted((_index_recency(d.name) for d in found), reverse=True)
    cutoff = scan[min(len(scan), max(limit, 1) * _INDEX_SCAN_MARGIN) - 1] if scan else ""

    rows_by_id: dict[str, dict[str, Any]] = {}
    for directory in found:
        if _index_recency(directory.name) < cutoff:
            continue
        report = directory / REPORT
        if not report.is_file():
            continue
        try:
            loaded = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        row = _entry(loaded, directory)
        # :func:`directories` visits canonical shared copies first.  Keep them
        # when a non-destructive migration left the legacy original in place.
        rows_by_id.setdefault(row["run_id"], row)
    return sorted(
        rows_by_id.values(), key=lambda row: row.get("written_at") or "", reverse=True
    )[:limit]


def policies(root: Path | None = None, limit: int = 60) -> list[dict[str, Any]]:
    """Trained checkpoints usable as a source or an opponent, newest run first.

    Bounded by the same listing-only cutoff `index` uses, for the same reason:
    a full scan of the archive costs seconds and this feeds a picker.

    Milestone lane directories (`<job>-u####-l###`) are skipped. They are
    fixed-seed diagnostic replays of a parent job and carry no weights of their
    own, so offering one as an opponent would be offering nothing.
    """

    found = [
        directory
        for directory in directories(root)
        if not _LANE_SUFFIX.search(directory.name)
    ]
    scan = sorted((_index_recency(d.name) for d in found), reverse=True)
    cutoff = scan[min(len(scan), max(limit, 1) * _INDEX_SCAN_MARGIN) - 1] if scan else ""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for directory in found:
        if _index_recency(directory.name) < cutoff or directory.name in seen:
            continue
        weights = sorted((directory / "stage" / "policies").glob("update-*.npz"))
        if not weights:
            continue
        seen.add(directory.name)
        rows.append(
            {
                "run_id": directory.name,
                "updates": [
                    int(path.stem.rsplit("-", 1)[-1])
                    for path in weights
                    if path.stem.rsplit("-", 1)[-1].isdigit()
                ],
                # Relative to the project root, which is the form the train spec
                # takes -- an absolute Windows path does not survive the WSL
                # boundary the worker runs behind.
                "directory": str(
                    (directory / "stage" / "policies").relative_to(_ROOT)
                ).replace("\\", "/"),
            }
        )
    rows.sort(key=lambda row: row["run_id"], reverse=True)
    return rows[:limit]


def load(name: str, root: Path | None = None) -> dict[str, Any] | None:
    """One run's full report, or ``None`` if it is not stored."""

    directory = locate_directory(name, root)
    if directory is None:
        return None
    report = directory / REPORT
    if not report.is_file():
        return None
    return json.loads(report.read_text(encoding="utf-8"))


def replay(
    name: str, root: Path | None = None, *, include_blocks: bool = False,
) -> dict[str, Any] | None:
    """A stored run rebuilt into the shape the map draws, or ``None``.

    The store deliberately splits a run in two -- `report.json` is what it was
    produced against, `trajectory.json` is the thousands of steps -- so a
    replay has to put them back together. It is one route rather than two
    fetches because a report without its trajectory renders an empty map, and a
    trajectory without its report renders numbers with no provenance.
    """

    base = locate_directory(name, root)
    if base is None:
        return None
    report = base / REPORT
    payload = base / PAYLOAD
    if not report.is_file() or not payload.is_file():
        return None

    stored = json.loads(report.read_text(encoding="utf-8"))
    series = json.loads(payload.read_text(encoding="utf-8"))
    if not series.get("trajectory"):
        return None
    spec = stored.get("spec") or {}
    terrain = series.get("terrain") or {}
    trajectory = series["trajectory"]
    if include_blocks and not terrain.get("blocks"):
        from console.core.worlds import replay_terrain

        identity = series.get("world_identity") or series.get("region") or {}
        reference = terrain
        if not reference and str(identity.get("generator", "")).startswith(
            "worldgen_v2"
        ):
            reference = replay_terrain.worldgen_reference(
                identity,
                seed=identity.get("replay_world_seed"),
            )
        if spec.get("world_design"):
            # A design world has no captured geometry: its terrain entry is a
            # `reference_only` stub with zero cells, and every `resolve` branch
            # is a Region lookup that finds nothing -- so the archive drew an
            # empty scene. The ground is a plane, so rebuild it from the
            # trajectory the actors walked on.
            rebuilt = replay_terrain.flat_from_trajectory(trajectory)
        elif reference:
            rebuilt = replay_terrain.resolve(
                reference,
                trajectory,
                cache_key=f"{payload.resolve()}:{payload.stat().st_mtime_ns}",
            )
        else:
            rebuilt = None
            # Ground nodes alone are terrain worth drawing. The block shell adds
            # cliffs, walls and overhangs, but it needs block-semantic sidecars
            # and degrades to None without them -- so requiring it discarded a
            # perfectly good rebuild (345 standable nodes, measured) and left the
            # reference-only stub, which draws an empty map.
        if rebuilt and (rebuilt.get("blocks") or rebuilt.get("nodes")):
            series["terrain"] = rebuilt
    return {
        **series,
        "spec": spec,
        "summary": stored.get("summary") or {},
        "warnings": stored.get("warnings") or [],
        "replay_of": name,
        "stored_at": stored.get("written_at"),
        # A void run replays fine and must still say so: its identity moved
        # while it was measuring, so the numbers on screen are not a result.
        "void": bool(stored.get("void")),
    }


def compare(left: str, right: str, root: Path | None = None) -> dict[str, Any]:
    """Compare settings, source identity, and declared contracts separately.

    A settings-only comparison is interpretable only after source and contract
    equality are established.  Missing declarations therefore remain
    ``undeclared``; they are never treated as equality merely because both
    reports omitted the field.
    """

    a, b = load(left, root), load(right, root)
    missing = [name for name, value in ((left, a), (right, b)) if value is None]
    if missing:
        return {"error": f"unknown run(s): {missing}"}

    left_contracts = _declared_contracts(a)
    right_contracts = _declared_contracts(b)
    contract_drift = _diff(left_contracts, right_contracts)
    identity_drift = _diff(a.get("identity") or {}, b.get("identity") or {})
    settings_drift = _diff(a.get("spec") or {}, b.get("spec") or {})
    left_source = _snapshot_manifest_sha(a)
    right_source = _snapshot_manifest_sha(b)

    contracts_state = _axis_state(left_contracts, right_contracts, contract_drift)
    source_state = _scalar_axis_state(left_source, right_source)
    settings_state = (
        "same"
        if not settings_drift
        else "single-variable"
        if len(settings_drift) == 1
        else "multi-variable"
    )
    void = {left: bool(a.get("void")), right: bool(b.get("void"))}
    verdict = _comparison_verdict(
        contracts_state=contracts_state,
        source_state=source_state,
        settings_state=settings_state,
        void=void,
    )

    blocking = dict(contract_drift)
    if contracts_state == "undeclared":
        blocking["contracts"] = {
            "left": "declared" if left_contracts else "undeclared",
            "right": "declared" if right_contracts else "undeclared",
        }
    if source_state == "different":
        blocking["snapshot_manifest_sha256"] = {
            "left": left_source,
            "right": right_source,
        }
    elif source_state == "undeclared":
        blocking["snapshot_manifest_sha256"] = {
            "left": left_source or "undeclared",
            "right": right_source or "undeclared",
        }

    comparable = (
        contracts_state == "same" and source_state == "same" and not any(void.values())
    )
    return {
        "left": left,
        "right": right,
        "comparable": comparable,
        "verdict": verdict,
        "axes": {
            "settings": {
                "state": settings_state,
                "difference_count": len(settings_drift),
                "diff": settings_drift,
            },
            "source": {
                "state": source_state,
                "left": left_source or "undeclared",
                "right": right_source or "undeclared",
            },
            "contracts": {
                "state": contracts_state,
                "difference_count": len(contract_drift),
                "diff": contract_drift,
                "left_declared": bool(left_contracts),
                "right_declared": bool(right_contracts),
            },
        },
        "blocking": blocking,
        "identity_drift": identity_drift,
        "spec_diff": settings_drift,
        "summary_diff": _diff(a.get("summary") or {}, b.get("summary") or {}),
        "void": void,
    }


def _declared_contracts(report: dict[str, Any]) -> dict[str, Any]:
    stamp = report.get("contracts")
    if not isinstance(stamp, dict):
        return {}
    declared = stamp.get("contracts")
    result = dict(declared) if isinstance(declared, dict) else {}
    # Shapes are part of the executable contract even though they are not
    # hashes.  Keeping them on this axis preserves the store's original guard.
    for key in ("observation_size", "action_head_sizes"):
        if key in stamp and stamp[key] is not None:
            result[key] = stamp[key]
    return result


def _snapshot_manifest_sha(report: dict[str, Any]) -> str | None:
    """Read a declared snapshot-manifest receipt from common containers."""

    candidates: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            direct = value.get("snapshot_manifest_sha256")
            if isinstance(direct, str) and direct:
                candidates.append(direct)
            for key, child in value.items():
                # Input-receipt maps commonly key the receipt by its path.
                if (
                    isinstance(key, str)
                    and "snapshot-manifest" in key.lower()
                    and isinstance(child, dict)
                    and isinstance(child.get("sha256"), str)
                ):
                    candidates.append(child["sha256"])
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(report)
    unique = {candidate.upper() for candidate in candidates}
    return next(iter(unique)) if len(unique) == 1 else None


def _axis_state(
    left: dict[str, Any],
    right: dict[str, Any],
    drift: dict[str, Any],
) -> str:
    if not left or not right:
        return "undeclared"
    return "different" if drift else "same"


def _scalar_axis_state(left: str | None, right: str | None) -> str:
    if left is None or right is None:
        return "undeclared"
    return "same" if left == right else "different"


def _comparison_verdict(
    *,
    contracts_state: str,
    source_state: str,
    settings_state: str,
    void: dict[str, bool],
) -> str:
    if any(void.values()):
        return "void"
    if contracts_state == "different":
        return "different contract"
    if source_state == "different":
        return "different source"
    if contracts_state == "undeclared" or source_state == "undeclared":
        return "undeclared"
    if settings_state == "single-variable":
        return "single-variable"
    if settings_state == "multi-variable":
        return "multi-variable"
    return "same configuration"


def _diff(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {"left": left.get(key), "right": right.get(key)}
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }
