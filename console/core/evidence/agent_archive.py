"""Synchronise an agent's durable receipts into the Console shared archive.

An agent can write two different artifact families:

* Console-native runs under ``artifacts/console/<run-id>``;
* campaign receipts under ``agents/<agent>/artifacts/<campaign>``.

The Archive should not depend on remembering either producer's private path.
This module mirrors both, non-destructively, into the canonical namespace
``artifacts/console/shared/<agent>``.  Native runs retain their replay payloads;
campaign reports become receipt-only Archive entries that link to an exact
copied source report.  Re-running the synchroniser is content-idempotent.

**No agent is named in this file.** It was previously ``dawn_archive`` with
``dawn`` written into a constant, a root path and eighteen call sites, so a
second agent that wrote campaign receipts got nothing. Agents are discovered
from ``agents/`` by :mod:`console.core.catalog.agents`, which means an agent
gets archive mirroring by existing as a directory, and stops when it does not.
"""

from __future__ import annotations

import argparse
import json
import shutil
import threading
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from console.core.evidence import store


SCHEMA_ID = "hytalerl-console-agent-archive-sync-v1"
EXTERNAL_RECEIPT_SCHEMA_ID = "hytalerl-console-external-receipt-v1"
SOURCE_REPORT = "source-report.json"
_ROOT = Path(__file__).resolve().parents[3]
#: Where an agent puts campaign artifacts, relative to its own directory.
AGENT_ARTIFACT_DIR = "artifacts"
_REPORT_NAMES = (
    "report.json",
    "visual_evaluation.json",
    "summary.json",
    "frozen_source_probe.json",
)
_lock = threading.Lock()
_last_scan = 0.0
_last_result: dict[str, Any] | None = None


def agent_artifact_root(agent: str) -> Path:
    return _ROOT / "agents" / agent / AGENT_ARTIFACT_DIR


def _agents() -> tuple[str, ...]:
    """Every discovered agent id. Empty when the catalog is unavailable."""

    try:
        from console.core.catalog import agents as agent_catalog

        return tuple(
            str(entry["id"]) for entry in agent_catalog.index() if entry.get("id")
        )
    except Exception:  # noqa: BLE001 - a console without the catalog still serves runs
        return ()


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def _timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _primary_report(directory: Path) -> Path | None:
    return next(
        (directory / name for name in _REPORT_NAMES if (directory / name).is_file()),
        None,
    )


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _scalar(value: Any) -> Any:
    return (
        value if isinstance(value, (str, int, float, bool)) or value is None else None
    )


def _external_summary(source: Path, report: dict[str, Any]) -> dict[str, Any]:
    updates = report.get("updates")
    summary = {
        "source_schema": _scalar(report.get("schema")),
        "status": _scalar(report.get("status") or report.get("promotion_status")),
        "checkpoint_published": _scalar(report.get("checkpoint_published")),
        "accepted_round_count": _scalar(report.get("accepted_round_count")),
        "updates_completed": len(updates)
        if isinstance(updates, list)
        else _scalar(updates),
        "optimizer_step": _scalar(report.get("optimizer_step")),
        "elapsed_seconds": _scalar(
            report.get("elapsed_seconds") or report.get("wall_seconds")
        ),
        "device": _scalar(report.get("device")),
        "checkpoint_sha256": _scalar(report.get("checkpoint_sha256")),
        "source_report_sha256": _digest(source),
        "replay_available": False,
    }
    return {key: value for key, value in summary.items() if value is not None}


def _external_spec(
    agent: str, artifact: Path, source: Path, report: dict[str, Any]
) -> dict[str, Any]:
    config = _mapping(report.get("config"))
    runtime = _mapping(report.get("runtime_identity"))
    return {
        "kind": f"{agent}_external_training_receipt_v1",
        "shared_agent": agent,
        "profile_ids": [agent],
        "policy": artifact.name,
        "scenario": report.get("visual_training_skill")
        or report.get("training_objective")
        or report.get("purpose")
        or report.get("schema"),
        "seed": report.get("seed") or config.get("seed"),
        "ticks": report.get("episode_ticks") or config.get("episode_ticks"),
        "loadout": config.get("loadout"),
        "opponent": config.get("opponent"),
        "source_report": str(source.relative_to(_ROOT)).replace("\\", "/"),
        "source_artifact": str(artifact.relative_to(_ROOT)).replace("\\", "/"),
        "source_schema": report.get("schema"),
        "runtime_contract_sha256": runtime.get("contract_sha256"),
    }


def _write_external_receipt(agent: str, artifact: Path, source: Path) -> tuple[str, bool]:
    loaded = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{agent} source report must be an object: {source}")
    name = f"{agent}-{artifact.name}"
    directory = store.directory_for(name, shared_agent=agent)
    source_sha = _digest(source)
    normalized_path = directory / store.REPORT
    if normalized_path.is_file():
        try:
            current = json.loads(normalized_path.read_text(encoding="utf-8"))
            if current.get("summary", {}).get("source_report_sha256") == source_sha:
                _write_legacy_compatibility_view(agent, directory)
                return name, False
        except (OSError, json.JSONDecodeError):
            pass

    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, directory / SOURCE_REPORT)
    spec = _external_spec(agent, artifact, source, loaded)
    summary = _external_summary(source, loaded)
    normalized = {
        "schema": EXTERNAL_RECEIPT_SCHEMA_ID,
        "run_id": name,
        "written_at": _timestamp(source),
        "contracts": loaded.get("contracts"),
        "contracts_unavailable": (
            None
            if loaded.get("contracts")
            else "source campaign did not publish a Console contract stamp"
        ),
        "identity": loaded.get("identity"),
        "identity_unavailable": (
            None
            if loaded.get("identity")
            else "source campaign identity is retained in source-report.json"
        ),
        "identity_drift": loaded.get("identity_drift"),
        "void": bool(loaded.get("void")),
        "backend": loaded.get("platform") or loaded.get("device") or "unknown",
        "host": loaded.get("host"),
        "spec": spec,
        "control_arm": {},
        "seed": spec.get("seed"),
        "ticks": spec.get("ticks"),
        "decision_period": None,
        "summary": summary,
        "warnings": [],
        "single_run_caveat": (
            "receipt-only archive entry; open source-report.json for the complete "
            "campaign evidence"
        ),
    }
    normalized_path.write_text(
        json.dumps(normalized, indent=2, default=str) + "\n", encoding="utf-8"
    )
    _write_legacy_compatibility_view(agent, directory)
    return name, True


def _legacy_view_is_current(canonical: Path) -> bool:
    """True when the flat compatibility copy is at least as new as the source.

    This exists purely for speed, and it is the difference between /api/runs
    answering and timing out. Without it every request re-read, re-serialised
    and re-compared one report per archived run -- about 3,800 JSON round trips
    across 1,983 flat and 1,863 shared directories, which took over 120s.
    An mtime comparison is O(1) per directory and still refreshes the view
    whenever the canonical report actually changes.
    """

    source = canonical / store.REPORT
    target = store.ARTIFACT_ROOT / canonical.name / store.REPORT
    try:
        return target.stat().st_mtime >= source.stat().st_mtime
    except OSError:
        return False


def _write_legacy_compatibility_view(agent: str, canonical: Path) -> None:
    """Expose shared entries to an already-running pre-migration Console."""

    if _legacy_view_is_current(canonical):
        return
    legacy = store.ARTIFACT_ROOT / canonical.name
    legacy.mkdir(parents=True, exist_ok=True)
    report = json.loads((canonical / store.REPORT).read_text(encoding="utf-8"))
    spec = _mapping(report.get("spec"))
    spec["profile_id"] = agent
    spec["shared_agent"] = agent
    owners = list(spec.get("profile_ids") or [])
    if agent not in owners:
        owners.append(agent)
    spec["profile_ids"] = owners
    spec["canonical_archive_path"] = str(canonical)
    policy = str(spec.get("policy") or canonical.name)
    if agent.lower() not in policy.lower():
        spec["policy"] = f"[{agent.upper()}] {policy}"
    report["spec"] = spec
    report_text = json.dumps(report, indent=2, default=str) + "\n"
    legacy_report = legacy / store.REPORT
    if (
        not legacy_report.is_file()
        or legacy_report.read_text(encoding="utf-8") != report_text
    ):
        legacy_report.write_text(report_text, encoding="utf-8")
    for name in (store.PAYLOAD, SOURCE_REPORT):
        source = canonical / name
        target = legacy / name
        if source.is_file() and not target.is_file():
            shutil.copy2(source, target)


def _copy_legacy_run(agent: str, source: Path) -> tuple[str, bool]:
    target = store.directory_for(source.name, shared_agent=agent)
    changed = not target.is_dir()
    if changed:
        shutil.copytree(source, target)
    target_report = target / store.REPORT
    canonical = json.loads(target_report.read_text(encoding="utf-8"))
    spec = _mapping(canonical.get("spec"))
    owners = list(spec.get("profile_ids") or [])
    if spec.get("profile_id") and spec["profile_id"] not in owners:
        owners.append(spec["profile_id"])
    if agent not in owners:
        owners.append(agent)
    if spec.get("shared_agent") != agent or spec.get("profile_ids") != owners:
        spec["shared_agent"] = agent
        spec["profile_ids"] = owners
        canonical["spec"] = spec
        target_report.write_text(
            json.dumps(canonical, indent=2, default=str) + "\n", encoding="utf-8"
        )
        changed = True
    _write_legacy_compatibility_view(agent, target)
    return source.name, changed


def _scan_flat_root() -> tuple[dict[str, list[Path]], list[str]]:
    """Route every flat-layout run to its owning agent in ONE pass.

    This scan used to run once per agent, so with four agents the same ~1,983
    directories were opened and parsed four times to answer a single request.
    The owner is a property of the run, not of the agent asking, so resolve it
    once and group.
    """

    owned: dict[str, list[Path]] = {}
    errors: list[str] = []
    base = store.ARTIFACT_ROOT
    if not base.is_dir():
        return owned, errors
    for directory in sorted(base.iterdir()):
        if not directory.is_dir() or directory.name == store.SHARED_DIRECTORY:
            continue
        report_path = directory / store.REPORT
        if not report_path.is_file():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                continue
            if report.get("schema") == EXTERNAL_RECEIPT_SCHEMA_ID:
                continue
            owner = store.shared_agent_for(_mapping(report.get("spec")))
            if owner:
                owned.setdefault(owner, []).append(directory)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{directory.name}: {type(error).__name__}: {error}")
    return owned, errors


def sync_agent(
    agent: str, legacy_directories: list[Path] | None = None
) -> dict[str, Any]:
    """Mirror one agent's artifacts. Returns that agent's sync receipt.

    ``legacy_directories`` lets a caller that already scanned the flat root hand
    over this agent's share instead of making every agent rescan it. Omit it and
    the scan happens here, so a standalone call still works.
    """

    imported: list[str] = []
    unchanged: list[str] = []
    errors: list[str] = []
    base = store.ARTIFACT_ROOT
    canonical_base = base / store.SHARED_DIRECTORY / agent
    canonical_count = 0
    if canonical_base.is_dir():
        for directory in sorted(canonical_base.iterdir()):
            if not directory.is_dir() or not (directory / store.REPORT).is_file():
                continue
            try:
                _write_legacy_compatibility_view(agent, directory)
                canonical_count += 1
            except (OSError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"{directory.name}: {type(error).__name__}: {error}")

    if legacy_directories is None:
        owned, scan_errors = _scan_flat_root()
        legacy_directories = owned.get(agent, [])
        errors.extend(scan_errors)
    for directory in legacy_directories:
        try:
            name, changed = _copy_legacy_run(agent, directory)
            (imported if changed else unchanged).append(name)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{directory.name}: {type(error).__name__}: {error}")

    artifact_root = agent_artifact_root(agent)
    if artifact_root.is_dir():
        for artifact in sorted(artifact_root.iterdir()):
            if not artifact.is_dir():
                continue
            source = _primary_report(artifact)
            if source is None:
                continue
            try:
                name, changed = _write_external_receipt(agent, artifact, source)
                (imported if changed else unchanged).append(name)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"{artifact.name}: {type(error).__name__}: {error}")

    return {
        "shared_agent": agent,
        "directory": str(canonical_base),
        "canonical_count": canonical_count,
        "imported": imported,
        "unchanged_count": len(unchanged),
        "errors": errors,
    }


def sync(
    *, force: bool = False, minimum_interval_seconds: float = 30.0
) -> dict[str, Any]:
    """Import completed artifacts for every discovered agent.

    The top-level keys are the union across agents, so a caller that only wants
    "did anything import" keeps working; ``agents`` carries the per-agent
    breakdown.
    """

    global _last_result, _last_scan
    with _lock:
        now = time.monotonic()
        if (
            not force
            and _last_result is not None
            and now - _last_scan < minimum_interval_seconds
        ):
            return dict(_last_result)

        owned, scan_errors = _scan_flat_root()
        per_agent = [
            sync_agent(agent, legacy_directories=owned.get(agent, []))
            for agent in _agents()
        ]
        result = {
            "schema": SCHEMA_ID,
            "agents": per_agent,
            "canonical_count": sum(item["canonical_count"] for item in per_agent),
            "imported": [name for item in per_agent for name in item["imported"]],
            "unchanged_count": sum(item["unchanged_count"] for item in per_agent),
            "errors": scan_errors
            + [error for item in per_agent for error in item["errors"]],
        }
        _last_scan = now
        _last_result = result
        return dict(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument(
        "--agent",
        default=None,
        help="only this agent; default is every agent discovered under agents/",
    )
    args = parser.parse_args()

    def once() -> dict[str, Any]:
        if args.agent:
            return sync_agent(args.agent)
        return sync(force=True)

    if not args.watch:
        print(json.dumps(once(), indent=2))
        return
    if args.interval < 0.5:
        raise ValueError("archive watch interval must be at least 0.5 seconds")
    while True:
        receipt = once()
        if receipt["imported"] or receipt["errors"]:
            print(json.dumps(receipt, indent=2), flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
