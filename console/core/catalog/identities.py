"""Declared identities compared with live or staged counterparts.

The comparison primitive is generic.  The bridge is merely the first concrete
instance because it has three independently meaningful identities today:
built bytes, the deployed/runtime bytes, and the declared Gym pin.
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
BRIDGE_NAME = "HytaleRLBridge-0.1.0.jar"
GYM_ROOT = ROOT / "HytaleRL" / "hytalegym"
if str(GYM_ROOT) not in sys.path:
    sys.path.insert(0, str(GYM_ROOT))


def compare_identity(
    *,
    label: str,
    declared: str | None,
    counterpart: str | None,
    declared_source: str,
    counterpart_source: str,
    live: bool = True,
) -> dict[str, Any]:
    """Compare two declared opaque identities without interpreting contents."""

    left = _known(declared)
    right = _known(counterpart)
    if not left or not right or not live:
        match: bool | None = None
        state = "undeclared" if not left or not right else "not live"
    else:
        match = left.upper() == right.upper()
        state = "match" if match else "mismatch"
    return {
        "label": label,
        "declared": left,
        "counterpart": right,
        "declared_source": declared_source,
        "counterpart_source": counterpart_source,
        "live": bool(live),
        "match": match,
        "state": state,
    }


def bridge_identity(
    *,
    workspace: str | Path = ROOT,
    pin: str | None = None,
    native_listening: bool | None = None,
    canonical_deployed_path: str | Path | None = None,
) -> dict[str, Any]:
    """Three-way bridge identity plus deploy-time provenance.

    The deployed jar's mtime is deliberately returned only as
    ``jar_build_mtime``.  File copies preserve that timestamp in this workflow;
    deployment time comes from the predecessor backup, corroborated by the
    server log created at restart.
    """

    root = Path(workspace)
    build_path = root / "HytaleRL" / "hytale-plugin" / "build" / "libs" / BRIDGE_NAME
    runtime = _newest_runtime_candidate(root)
    runtime_path = runtime.get("jar") if runtime else None

    if pin is None:
        try:
            from hytalegym.worldgen.region.stability import (
                CURRENT_NATIVE_EVIDENCE_JAR_SHA256,
            )
            pin = CURRENT_NATIVE_EVIDENCE_JAR_SHA256
        except Exception:  # noqa: BLE001 - unavailable is a normal state
            pin = None
    if native_listening is None:
        try:
            from console.core.telemetry.surfaces import bridge_status
            native_listening = any(
                listener.get("role") == "native" and listener.get("listening")
                for listener in bridge_status().get("listeners", [])
            )
        except Exception:  # noqa: BLE001 - viewer must degrade
            native_listening = False

    if canonical_deployed_path is None:
        try:
            from adk.probes.crosslang import deployed_jar_path
            canonical_deployed_path = deployed_jar_path()
        except Exception:  # noqa: BLE001 - explicit path semantics stay intact
            canonical_deployed_path = None

    build_hash = _sha256(build_path)
    runtime_hash = _sha256(Path(runtime_path)) if runtime_path else None
    canonical_hash = (
        _sha256(Path(canonical_deployed_path)) if canonical_deployed_path else None
    )
    live = bool(native_listening and runtime_hash)
    comparisons = [
        compare_identity(
            label="build ↔ runtime",
            declared=build_hash,
            counterpart=runtime_hash,
            declared_source=str(build_path),
            counterpart_source=str(runtime_path or "undeclared"),
            live=live,
        ),
        compare_identity(
            label="pin ↔ runtime",
            declared=pin,
            counterpart=runtime_hash,
            declared_source="CURRENT_NATIVE_EVIDENCE_JAR_SHA256",
            counterpart_source=str(runtime_path or "undeclared"),
            live=live,
        ),
    ]

    backup = runtime.get("backup") if runtime else None
    server_log = runtime.get("log") if runtime else None
    deploy_time = _created_at(backup)
    restart_time = _log_start_time(server_log) or _mtime(server_log)
    jar_build_mtime = _mtime(Path(runtime_path)) if runtime_path else None
    undeployed = build_hash if live and build_hash and build_hash != runtime_hash else None
    return {
        "build": {"path": str(build_path), "sha256": build_hash},
        "runtime": {
            "path": str(runtime_path) if runtime_path else None,
            "sha256": runtime_hash,
            "live": live,
        },
        "pin": pin,
        "canonical_install": {
            "path": str(canonical_deployed_path) if canonical_deployed_path else None,
            "sha256": canonical_hash,
        },
        "comparisons": comparisons,
        "undeployed_sha256": undeployed,
        "deployment": {
            "deploy_time": deploy_time,
            "deploy_time_source": (
                f"creation time of {backup}" if backup else None
            ),
            "server_restart_time": restart_time,
            "server_log": str(server_log) if server_log else None,
            "jar_build_mtime": jar_build_mtime,
            "jar_mtime_is_deploy_time": False,
        },
    }


def _newest_runtime_candidate(root: Path) -> dict[str, Path] | None:
    base = root / "artifacts" / "console-hytale"
    if not base.is_dir():
        return None
    logs = sorted(
        base.rglob("*_server.log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for log in logs:
        candidate = log.parent.parent
        jar = candidate / "mods" / BRIDGE_NAME
        if not jar.is_file():
            continue
        try:
            body = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "Bridge server listening on 127.0.0.1:5556" not in body:
            continue
        backups = sorted(
            (candidate / "bridge-backups").glob("*previous-*.jar"),
            # Copying a jar preserves its content mtime. On Windows the file's
            # creation time records when this predecessor backup was actually
            # written during deployment.
            key=lambda path: _created_at(path) or 0,
            reverse=True,
        )
        return {
            "root": candidate,
            "jar": jar,
            "log": log,
            "backup": backups[0] if backups else None,
        }
    return None


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest().upper()


def _mtime(path: Path | None) -> float | None:
    if path is None:
        return None
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _created_at(path: Path | None) -> float | None:
    if path is None:
        return None
    try:
        return path.stat().st_ctime
    except OSError:
        return None


def _log_start_time(path: Path | None) -> float | None:
    if path is None:
        return None
    # The server log filename is created at restart and is more precise than
    # its final mtime, which advances for as long as the server writes.
    try:
        value = path.name.removesuffix("_server.log")
        parsed = datetime.strptime(value, "%Y-%m-%d_%H-%M-%S")
    except (ValueError, AttributeError):
        return None
    return parsed.astimezone().timestamp()


def _known(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if not text or text.lower() == "unknown" else text
