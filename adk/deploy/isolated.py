"""Run a contract-checked policy bundle in a private Hytale server."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time
from typing import Any

from adk.deploy.bundle import verify


SCHEMA = "hytalerl_isolated_policy_execution_v1"
_MARKERS = (
    "Hytale Server Booted!",
    "Bridge server listening on",
    "PolicyAgent combat: bridge facade (full control)",
    "PolicyAgent combat fixture ready:",
)


def run(
    bundle: Path,
    runtime: Path,
    *,
    policy_jar: Path,
    bridge_jar: Path,
    server_jar: Path,
    assets: Path,
    java: Path,
    game_port: int,
    bridge_port: int,
    fixture: str = "Kweebec_Razorleaf Trork_Brawler 0.5 65.0 0.5 2.5",
    minimum_policy_ticks: int = 120,
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    """Launch, observe, and cleanly stop one isolated closed-loop fixture."""

    inputs = {
        "bundle": Path(bundle).resolve(),
        "policy_jar": Path(policy_jar).resolve(),
        "bridge_jar": Path(bridge_jar).resolve(),
        "server_jar": Path(server_jar).resolve(),
        "assets": Path(assets).resolve(),
        "java": Path(java).resolve(),
    }
    missing = [name for name, path in inputs.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"isolated deployment inputs missing: {missing}")
    problems = verify(inputs["bundle"])
    if problems:
        raise ValueError(f"policy bundle is not deployable: {problems}")
    runtime = Path(runtime).resolve()
    if runtime.exists():
        raise FileExistsError(f"isolated runtime already exists: {runtime}")
    if runtime == Path.cwd().resolve() or Path.cwd().resolve() not in runtime.parents:
        raise ValueError("isolated runtime must be a new directory under the workspace")
    _require_free_port(bridge_port, socket.SOCK_STREAM)
    _require_free_port(game_port, socket.SOCK_DGRAM)

    data = runtime / "mods/policy-agent"
    data.parent.mkdir(parents=True)
    shutil.copytree(inputs["bundle"], data)
    shutil.copy2(inputs["policy_jar"], data.parent / "HytalePolicyAgent-0.1.0.jar")
    shutil.copy2(inputs["bridge_jar"], data.parent / "HytaleRLBridge-0.1.0.jar")
    (runtime / "universe").mkdir()
    (data / "takeover").write_text("", encoding="utf-8")
    (data / "lifecycle-trace").write_text("", encoding="utf-8")
    (data / "combat-fixture.txt").write_text(fixture + "\n", encoding="utf-8")

    command = (
        str(inputs["java"]),
        "-Xshare:off",
        f"-Dhytalerl.bridge.port={bridge_port}",
        "-jar",
        str(inputs["server_jar"]),
        "--assets",
        str(inputs["assets"]),
        "--disable-sentry",
        "--auth-mode",
        "offline",
        "--bind",
        f"127.0.0.1:{game_port}",
        "--universe",
        str(runtime / "universe"),
    )
    report_path = runtime / "report.json"
    log_path = runtime / "server.log"
    started = time.monotonic()
    process = None
    evidence: dict[str, Any] = {}
    error = None
    exit_code = None
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            process = subprocess.Popen(
                command,
                cwd=runtime,
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            _write_status(
                report_path,
                _report(False, "starting", inputs, command, process.pid, started),
            )
            while time.monotonic() - started < timeout_seconds:
                exit_code = process.poll()
                text = log_path.read_text(encoding="utf-8", errors="replace")
                if "PolicyAgent inert:" in text:
                    raise RuntimeError(_last_matching(text, "PolicyAgent inert:"))
                metrics = _read_json(data / "metrics.json")
                trace_files = sorted(data.glob("lifecycle-trace-*.jsonl"))
                lifecycle_rows = sum(_line_count(path) for path in trace_files)
                markers = {marker: marker in text for marker in _MARKERS}
                evidence = {
                    "markers": markers,
                    "metrics": metrics,
                    "lifecycle_trace_files": [str(path) for path in trace_files],
                    "lifecycle_rows": lifecycle_rows,
                }
                ready = (
                    all(markers.values())
                    and metrics.get("schema") == "hytale_policy_agent_metrics"
                    and int(metrics.get("schema_version", 0)) >= 3
                    and int(metrics.get("claimed", 0)) >= 1
                    and int(metrics.get("policy_ticks", 0)) >= minimum_policy_ticks
                    and bool(metrics.get("world_ticking"))
                    and float(metrics.get("world_tps", 0.0)) > 0.0
                    and float(metrics.get("policy_ticks_per_second", 0.0)) > 0.0
                    and bool(metrics.get("last_action_legal"))
                    and int(metrics.get("rejected_illegal", 0)) == 0
                    and int(metrics.get("motion_samples", 0)) >= 2
                    and int(metrics.get("locomotion_requests", 0)) >= 1
                    and bool(metrics.get("moved"))
                    and lifecycle_rows >= 1
                )
                _write_status(
                    report_path,
                    _report(
                        False,
                        "observing" if exit_code is None else "exited_early",
                        inputs,
                        command,
                        process.pid,
                        started,
                        evidence=evidence,
                        exit_code=exit_code,
                    ),
                )
                if ready:
                    break
                if exit_code is not None:
                    raise RuntimeError(f"Hytale exited before evidence with {exit_code}")
                time.sleep(1.0)
            else:
                raise TimeoutError("isolated policy execution timed out")
    except Exception as failure:  # receipt survives every failed live attempt
        error = f"{type(failure).__name__}: {failure}"
    finally:
        if process is not None:
            exit_code = _stop(process)

    trace_files = sorted(data.glob("lifecycle-trace-*.jsonl"))
    lifecycle_rows = sum(_line_count(path) for path in trace_files)
    metrics = _read_json(data / "metrics.json")
    passed = error is None and exit_code == 0
    report = _report(
        passed,
        "passed" if passed else "failed",
        inputs,
        command,
        None if process is None else process.pid,
        started,
        evidence={
            **evidence,
            "metrics": metrics,
            "lifecycle_trace_files": [str(path) for path in trace_files],
            "lifecycle_rows": lifecycle_rows,
            "dynamic_behavior": {
                "policy_ticks": int(metrics.get("policy_ticks", 0)),
                "locomotion_requests": int(metrics.get("locomotion_requests", 0)),
                "motion_samples": int(metrics.get("motion_samples", 0)),
                "horizontal_path": float(metrics.get("motion_horizontal_path", 0.0)),
                "policy_tps": float(metrics.get("policy_ticks_per_second", 0.0)),
                "lifecycle_rows": lifecycle_rows,
                "combat_requested": bool(
                    metrics.get("last_attack")
                    or int(metrics.get("last_ability_slot", -1)) >= 0
                ),
            },
        },
        error=error,
        exit_code=exit_code,
    )
    _write_status(report_path, report)
    return report


def _report(passed, status, inputs, command, pid, started, **extra):
    return {
        "schema": SCHEMA,
        "passed": passed,
        "status": status,
        "pid": pid,
        "command": list(command),
        "isolated": True,
        "shared_server_modified": False,
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path) if path.is_file() else None}
            for name, path in inputs.items()
        },
        "wall_seconds": time.monotonic() - started,
        **extra,
    }


def _require_free_port(port: int, kind: int) -> None:
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("ports must be integers in 1..65535")
    with socket.socket(socket.AF_INET, kind) as probe:
        probe.bind(("127.0.0.1", port))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _last_matching(text: str, marker: str) -> str:
    return next((line for line in reversed(text.splitlines()) if marker in line), marker)


def _stop(process: subprocess.Popen[str]) -> int:
    if process.poll() is None and process.stdin is not None:
        try:
            process.stdin.write("stop\n")
            process.stdin.flush()
            return process.wait(timeout=45)
        except (OSError, subprocess.TimeoutExpired):
            process.terminate()
    try:
        return process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=10)


def _line_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as source:
            return sum(1 for line in source if line.strip())
    except OSError:
        return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _write_status(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--policy-jar", type=Path, required=True)
    parser.add_argument("--bridge-jar", type=Path, required=True)
    parser.add_argument("--server-jar", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--java", type=Path, required=True)
    parser.add_argument("--game-port", type=int, default=25572)
    parser.add_argument("--bridge-port", type=int, default=5558)
    parser.add_argument("--minimum-policy-ticks", type=int, default=120)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()
    report = run(
        args.bundle,
        args.runtime,
        policy_jar=args.policy_jar,
        bridge_jar=args.bridge_jar,
        server_jar=args.server_jar,
        assets=args.assets,
        java=args.java,
        game_port=args.game_port,
        bridge_port=args.bridge_port,
        minimum_policy_ticks=args.minimum_policy_ticks,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "run"]
