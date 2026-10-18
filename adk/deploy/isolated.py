"""Run a contract-checked policy bundle in a private Hytale server."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import shutil
import socket
import subprocess
import time
from typing import Any

from adk.deploy.bundle import verify
from adk.deploy.compatibility import profile_content_sha256
from adk.tools.replay_native_policy_trace import (
    TRACE_SCHEMA as _POLICY_TRACE_SCHEMA,
    _is_valid_terminal_row,
    _validate_header,
    _validate_transition_rows,
)


SCHEMA = "hytalerl_isolated_policy_execution_v1"
DEFAULT_POLICY_TRACE_MAXIMUM_TRANSITIONS = 512
_MARKERS = (
    "Hytale Server Booted!",
    "Bridge server listening on",
    "PolicyAgent combat: bridge facade (full control)",
    "PolicyAgent combat fixture ready:",
)
DEFAULT_FIXTURE = "Kweebec_Razorleaf Trork_Brawler spawn 2.5 16"
_FAILURE_MARKERS = (
    "PolicyAgent inert:",
    "PolicyAgent combat fixture failed:",
    "PolicyAgent could not configure combat-fixture.txt:",
)
_TASK_CONTRACT_FIELDS = frozenset({
    "objective",
    "target_controller",
    "starting_separation",
    "catch_distance",
    "maximum_ticks",
    "horizon",
    "terminal",
    "perception",
    "target_context",
    "bridge_sha256",
    "bridge_jar_sha256",
    "expected_bridge_sha256",
})
_TARGET_CONTEXT_FIELDS = frozenset({"schema", "supplement", "availability"})
_CONTRACT_STRING_FIELDS = (
    "objective",
    "target_controller",
    "horizon",
    "terminal",
    "perception",
)
_RUNTIME_SURFACE_SCHEMA = "hytalerl_isolated_runtime_surface_v1"
_POLICY_TRACE_COMPLETENESS_SCHEMA = (
    "hytalerl_policy_trace_completeness_v1"
)
_POLICY_TRACE_SEQUENCE_ERRORS = (
    "duplicate or out-of-order sequence",
    "does not continue policy slot",
    "does not match the preceding next observation",
    "does not match the preceding next mask",
    "does not match the preceding next delta_seconds",
    "non-consecutive next tick",
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
    bind_address: str = "127.0.0.1",
    world_or_universe: Path | None = None,
    fixture: str = DEFAULT_FIXTURE,
    minimum_policy_ticks: int = 120,
    timeout_seconds: float = 300.0,
    pursuit_context: bool = False,
    flat_world: bool = False,
    maximum_target_distance: float | None = None,
    policy_trace: bool = False,
    policy_trace_maximum_transitions: int | None = None,
    require_complete_policy_trace: bool = False,
    task_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Launch, observe, and cleanly stop one isolated closed-loop fixture."""

    require_complete_policy_trace = _validate_complete_policy_trace_requirement(
        policy_trace,
        require_complete_policy_trace,
    )
    policy_trace_maximum_transitions = _validate_policy_trace_maximum_transitions(
        policy_trace,
        policy_trace_maximum_transitions,
    )
    task_contract = _validate_task_contract(task_contract)
    inputs = {
        "bundle": Path(bundle).resolve(),
        "policy_jar": Path(policy_jar).resolve(),
        "bridge_jar": Path(bridge_jar).resolve(),
        "server_jar": Path(server_jar).resolve(),
        "assets": Path(assets).resolve(),
        "java": Path(java).resolve(),
    }
    if world_or_universe is not None:
        inputs["world_or_universe"] = Path(world_or_universe).resolve()
    missing = [name for name, path in inputs.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"isolated deployment inputs missing: {missing}")
    if "world_or_universe" in inputs and not inputs["world_or_universe"].is_dir():
        raise ValueError("world_or_universe must be a world or universe directory")
    if "world_or_universe" in inputs:
        source_world = inputs["world_or_universe"]
        if not (source_world / "worlds").is_dir() and not (
            source_world / "config.json"
        ).is_file():
            raise ValueError(
                "world_or_universe must contain a 'worlds' directory or a "
                "world config.json"
            )
    if flat_world and world_or_universe is not None:
        raise ValueError("--flat-world and --world-or-universe are mutually exclusive")
    problems = verify(inputs["bundle"])
    if problems:
        raise ValueError(f"policy bundle is not deployable: {problems}")
    runtime = Path(runtime).resolve()
    if runtime.exists():
        raise FileExistsError(f"isolated runtime already exists: {runtime}")
    if runtime == Path.cwd().resolve() or Path.cwd().resolve() not in runtime.parents:
        raise ValueError("isolated runtime must be a new directory under the workspace")
    bind_endpoint = _bind_endpoint(bind_address, game_port)
    _require_free_port(bridge_port, socket.SOCK_STREAM)
    _require_free_port(game_port, socket.SOCK_DGRAM, bind_address)
    artifact_hashes = {
        name: _artifact_sha256(path) for name, path in inputs.items()
    }
    launch_configuration = {
        "bind_address": bind_address,
        "game_port": game_port,
        "bridge_port": bridge_port,
        "world_or_universe": (
            str(inputs["world_or_universe"])
            if "world_or_universe" in inputs
            else "generated flat world" if flat_world else "fresh empty universe"
        ),
        "require_complete_policy_trace": require_complete_policy_trace,
    }

    data = runtime / "mods/policy-agent"
    data.parent.mkdir(parents=True)
    shutil.copytree(inputs["bundle"], data)
    runtime_policy_jar = data.parent / "HytalePolicyAgent-0.1.0.jar"
    runtime_bridge_jar = data.parent / "HytaleRLBridge-0.1.0.jar"
    shutil.copy2(inputs["policy_jar"], runtime_policy_jar)
    shutil.copy2(inputs["bridge_jar"], runtime_bridge_jar)
    runtime_surface = _runtime_surface(data, runtime_policy_jar, runtime_bridge_jar)
    universe = runtime / "universe"
    universe.mkdir()
    if world_or_universe is not None:
        _stage_world_or_universe(inputs["world_or_universe"], universe)
    if flat_world:
        world = universe / "worlds/default"
        world.mkdir(parents=True)
        _write_status(world / "config.json", {
            "Version": 4, "Seed": 1346720323,
            "WorldGen": {"Type": "Flat", "Layers": [
                {"From": 0, "To": 64, "Environment": "Default", "BlockType": "Rock_Stone"},
                {"From": 64, "To": 65, "Environment": "Default", "BlockType": "Soil_Grass"},
            ]},
            "IsTicking": True, "IsSpawningNPC": False,
            "IsSpawnMarkersEnabled": False, "IsGameTimePaused": True,
        })
    (data / "takeover").write_text("", encoding="utf-8")
    (data / "lifecycle-trace").write_text("", encoding="utf-8")
    if policy_trace:
        (data / "policy-trace").write_text(
            f"{policy_trace_maximum_transitions}\n", encoding="utf-8")
    if pursuit_context:
        (data / "pursuit-context").write_text("", encoding="utf-8")
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
        bind_endpoint,
        "--universe",
        str(universe),
    )
    report_path = runtime / "report.json"
    log_path = runtime / "server.log"
    started = time.monotonic()
    process = None
    evidence: dict[str, Any] = {}
    error = None
    exit_code = None
    runtime_gate_passed = False
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            process = subprocess.Popen(
                command,
                cwd=runtime,
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            _write_status(
                report_path,
                _report(
                    False,
                    "starting",
                    inputs,
                    command,
                    process.pid,
                    started,
                    artifact_hashes=artifact_hashes,
                    launch_configuration=launch_configuration,
                    evidence={"runtime_surface": runtime_surface},
                    task_contract=task_contract,
                ),
            )
            while time.monotonic() - started < timeout_seconds:
                exit_code = process.poll()
                text = log_path.read_text(encoding="utf-8", errors="replace")
                for marker in _FAILURE_MARKERS:
                    if marker in text:
                        raise RuntimeError(_last_matching(text, marker))
                metrics = _read_json(data / "metrics.json")
                trace_files = sorted(data.glob("lifecycle-trace-*.jsonl"))
                lifecycle_rows = sum(_line_count(path) for path in trace_files)
                policy_trace_files = sorted(data.glob("policy-trace-*.jsonl"))
                policy_trace_rows = sum(
                    _transition_line_count(path) for path in policy_trace_files)
                policy_trace_completeness = _policy_trace_completeness(
                    policy_trace_files,
                    requested=policy_trace,
                    maximum_transitions=policy_trace_maximum_transitions,
                    policy_ticks=_policy_trace_policy_ticks(metrics),
                )
                markers = {marker: marker in text for marker in _MARKERS}
                minimum_trace_rows = min(
                    max(1, minimum_policy_ticks - 1),
                    policy_trace_maximum_transitions
                    or DEFAULT_POLICY_TRACE_MAXIMUM_TRANSITIONS,
                )
                evidence = {
                    "markers": markers,
                    "metrics": metrics,
                    "lifecycle_trace_files": [str(path) for path in trace_files],
                    "lifecycle_rows": lifecycle_rows,
                    "policy_trace_files": [str(path) for path in policy_trace_files],
                    "policy_trace_rows": policy_trace_rows,
                    "policy_trace_completeness": policy_trace_completeness,
                    "policy_trace_maximum_transitions":
                        policy_trace_maximum_transitions,
                    "runtime_surface": runtime_surface,
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
                    and (not policy_trace
                         or policy_trace_rows >= minimum_trace_rows)
                    and (not pursuit_context or (
                        int(metrics.get("observation_changes", 0)) >= 2
                        and int(metrics.get("target_observation_samples", 0))
                        >= 0.9 * int(metrics.get("policy_ticks", 0))
                    ))
                    and (maximum_target_distance is None or (
                        "target_min_planar_distance_clipped" in metrics
                        and float(metrics["target_min_planar_distance_clipped"])
                        <= maximum_target_distance
                    ))
                )
                runtime_gate_passed = runtime_gate_passed or ready
                if require_complete_policy_trace:
                    evidence["runtime_gate_passed"] = runtime_gate_passed
                _write_status(
                    report_path,
                    _report(
                        False,
                        "observing" if exit_code is None else "exited_early",
                        inputs,
                        command,
                        process.pid,
                        started,
                        artifact_hashes=artifact_hashes,
                        launch_configuration=launch_configuration,
                        task_contract=task_contract,
                        evidence=evidence,
                        exit_code=exit_code,
                    ),
                )
                terminal_error = _terminal_actor_failure(metrics)
                if require_complete_policy_trace:
                    invalid_trace_reason = policy_trace_completeness.get("reason")
                    if invalid_trace_reason in {
                        "invalid_policy_trace_header",
                        "malformed_policy_trace_json",
                        "policy_trace_row_not_object",
                        "unreadable_policy_trace",
                        "multiple_policy_trace_files",
                        "policy_trace_maximum_transitions_mismatch",
                        "policy_trace_sequence_inconsistent",
                        "malformed_policy_trace_rows",
                        "policy_trace_nonterminal_budget_exceeded",
                        "policy_trace_terminal_allowance_exceeded",
                    }:
                        raise RuntimeError(
                            "native policy trace is invalid: "
                            f"{invalid_trace_reason}"
                        )
                    if not runtime_gate_passed:
                        if (policy_trace_completeness.get("complete") is True
                                or terminal_error is not None):
                            raise RuntimeError(
                                "episode reached terminal before the runtime "
                                "gate passed"
                            )
                        if exit_code is not None:
                            raise RuntimeError(
                                f"Hytale exited before evidence with {exit_code}"
                            )
                    else:
                        if policy_trace_completeness.get("complete") is True:
                            break
                        maximum = policy_trace_maximum_transitions
                        policy_ticks = _policy_trace_policy_ticks(metrics)
                        if (
                            maximum is not None
                            and policy_trace_completeness.get(
                                "nonterminal_transition_rows", 0
                            ) >= maximum
                            and policy_trace_completeness.get("terminal_rows", 0) == 0
                            and policy_ticks is not None
                            and policy_ticks > maximum + 1
                        ):
                            policy_trace_completeness["reason"] = (
                                "policy_trace_nonterminal_budget_exhausted"
                            )
                        if policy_trace_completeness.get("reason") == (
                            "policy_trace_nonterminal_budget_exhausted"
                        ):
                            raise RuntimeError(
                                "native policy trace cannot become complete: "
                                f"{policy_trace_completeness['reason']}"
                            )
                        if exit_code is not None:
                            raise RuntimeError(
                                "Hytale exited before a complete native "
                                "policy trace"
                            )
                        time.sleep(1.0)
                        continue
                else:
                    if ready:
                        break
                    if terminal_error is not None:
                        raise RuntimeError(terminal_error)
                    if exit_code is not None:
                        raise RuntimeError(
                            f"Hytale exited before evidence with {exit_code}"
                        )
                time.sleep(1.0)
            else:
                if require_complete_policy_trace and runtime_gate_passed:
                    raise TimeoutError(
                        "isolated policy execution timed out before a complete "
                        "native policy trace"
                    )
                raise TimeoutError("isolated policy execution timed out")
    except Exception as failure:  # receipt survives every failed live attempt
        error = f"{type(failure).__name__}: {failure}"
    finally:
        if process is not None:
            exit_code = _stop(process)

    trace_files = sorted(data.glob("lifecycle-trace-*.jsonl"))
    lifecycle_rows = sum(_line_count(path) for path in trace_files)
    policy_trace_files = sorted(data.glob("policy-trace-*.jsonl"))
    policy_trace_rows = sum(
        _transition_line_count(path) for path in policy_trace_files)
    policy_trace_completeness = _policy_trace_completeness(
        policy_trace_files,
        requested=policy_trace,
        maximum_transitions=policy_trace_maximum_transitions,
    )
    metrics, metrics_read_error = _read_metrics_receipt(data / "metrics.json")
    policy_trace_completeness = _policy_trace_completeness(
        policy_trace_files,
        requested=policy_trace,
        maximum_transitions=policy_trace_maximum_transitions,
        policy_ticks=_policy_trace_policy_ticks(metrics),
    )
    if require_complete_policy_trace:
        maximum = policy_trace_maximum_transitions
        policy_ticks = _policy_trace_policy_ticks(metrics)
        if (
            maximum is not None
            and policy_trace_completeness.get("nonterminal_transition_rows", 0)
            >= maximum
            and policy_trace_completeness.get("terminal_rows", 0) == 0
            and policy_ticks is not None
            and policy_ticks > maximum + 1
        ):
            policy_trace_completeness["reason"] = (
                "policy_trace_nonterminal_budget_exhausted"
            )
        evidence["runtime_gate_passed"] = runtime_gate_passed
        if error is not None:
            error = (
                f"{error}; policy_trace_completeness.reason="
                f"{policy_trace_completeness.get('reason', 'unknown')}"
            )
    if metrics_read_error is not None and error is None:
        error = (
            "metrics receipt "
            f"{metrics_read_error['kind']}: {metrics_read_error['message']}"
        )
    if error is None and (data / "nojump").is_file():
        error = (
            "runtime gate cannot certify the nojump diagnostic: it suppresses "
            "a selected policy action"
        )
    passed = error is None and exit_code == 0 and metrics_read_error is None
    combat_requested = _combat_requested(metrics)
    report = _report(
        passed,
        "passed" if passed else "failed",
        inputs,
        command,
        None if process is None else process.pid,
        started,
        artifact_hashes=artifact_hashes,
        launch_configuration=launch_configuration,
        evidence={
            **evidence,
            "metrics": metrics,
            "lifecycle_trace_files": [str(path) for path in trace_files],
            "lifecycle_rows": lifecycle_rows,
            "policy_trace_files": [str(path) for path in policy_trace_files],
            "policy_trace_rows": policy_trace_rows,
            "policy_trace_completeness": policy_trace_completeness,
            "policy_trace_maximum_transitions":
                policy_trace_maximum_transitions,
            "runtime_surface": runtime_surface,
            "dynamic_behavior": {
                "policy_ticks": int(metrics.get("policy_ticks", 0)),
                "locomotion_requests": int(metrics.get("locomotion_requests", 0)),
                "motion_samples": int(metrics.get("motion_samples", 0)),
                "horizontal_path": float(metrics.get("motion_horizontal_path", 0.0)),
                "policy_tps": float(metrics.get("policy_ticks_per_second", 0.0)),
                "lifecycle_rows": lifecycle_rows,
                "policy_trace_rows": policy_trace_rows,
                "combat_requested": combat_requested,
                "combat_action_requests": int(
                    metrics.get("combat_action_requests", 0)),
                "combat_attack_accepted": int(
                    metrics.get("combat_attack_accepted", 0)),
                "combat_ability_accepted": int(
                    metrics.get("combat_ability_accepted", 0)),
                "combat_ability_finished": int(
                    metrics.get("combat_ability_finished", 0)),
            },
        },
        error=error,
        exit_code=exit_code,
        **({"metrics_read_error": metrics_read_error}
           if metrics_read_error is not None else {}),
        task_contract=task_contract,
        fixture=fixture,
        maximum_target_distance=maximum_target_distance,
        accommodations={
            "pursuit_context": (data / "pursuit-context").is_file(),
            "nojump": (data / "nojump").is_file(),
            "takeover": True,
            "headless_fixture": True,
            "flat_world": flat_world,
            "policy_trace": policy_trace,
            "require_complete_policy_trace": require_complete_policy_trace,
        },
    )
    _write_status(report_path, report)
    return report


def _report(
    passed,
    status,
    inputs,
    command,
    pid,
    started,
    *,
    task_contract: dict[str, Any] | None = None,
    artifact_hashes: dict[str, str] | None = None,
    **extra,
):
    task_contract = _validate_task_contract(task_contract)
    artifact_hashes = artifact_hashes or {}
    report = {
        "schema": SCHEMA,
        "passed": passed,
        "status": status,
        "pid": pid,
        "command": list(command),
        "isolated": True,
        "shared_server_modified": False,
        "inputs": {
            name: {"path": str(path), "sha256": artifact_hashes.get(name)}
            for name, path in inputs.items()
        },
        "wall_seconds": time.monotonic() - started,
        **extra,
    }
    if task_contract is not None:
        report["task_contract"] = task_contract
    return report


def _validate_task_contract(
    task_contract: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate and copy optional fields consumed by the pursuit transfer gate."""
    if task_contract is None:
        return None
    if not isinstance(task_contract, dict):
        raise TypeError("task_contract must be a JSON object")

    unknown = set(task_contract) - _TASK_CONTRACT_FIELDS
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ValueError(f"task_contract has unsupported fields: {names}")

    normalized: dict[str, Any] = {}
    for name in _CONTRACT_STRING_FIELDS:
        if name in task_contract:
            normalized[name] = _contract_string(task_contract[name], name)

    if "starting_separation" in task_contract:
        normalized["starting_separation"] = _contract_separation(
            task_contract["starting_separation"]
        )
    if "catch_distance" in task_contract:
        normalized["catch_distance"] = _contract_nonnegative_number(
            task_contract["catch_distance"], "catch_distance"
        )
    if "maximum_ticks" in task_contract:
        value = task_contract["maximum_ticks"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("maximum_ticks must be an integer")
        if value <= 0:
            raise ValueError("maximum_ticks must be positive")
        normalized["maximum_ticks"] = value

    if "target_context" in task_contract:
        context = task_contract["target_context"]
        if not isinstance(context, dict):
            raise TypeError("target_context must be a JSON object")
        unknown_context = set(context) - _TARGET_CONTEXT_FIELDS
        if unknown_context:
            names = ", ".join(sorted(str(name) for name in unknown_context))
            raise ValueError(f"target_context has unsupported fields: {names}")
        normalized["target_context"] = {
            name: _contract_string(context[name], f"target_context.{name}")
            for name in _TARGET_CONTEXT_FIELDS
            if name in context
        }

    bridge_values = [
        task_contract[name]
        for name in ("bridge_sha256", "bridge_jar_sha256", "expected_bridge_sha256")
        if name in task_contract
    ]
    if bridge_values:
        bridge = _contract_sha256(bridge_values[0], "bridge_sha256")
        if any(
            _contract_sha256(value, "bridge_sha256") != bridge
            for value in bridge_values[1:]
        ):
            raise ValueError("task_contract bridge identities disagree")
        normalized["bridge_sha256"] = bridge

    return normalized


def _validate_policy_trace_maximum_transitions(
    requested: bool,
    maximum_transitions: int | None,
) -> int | None:
    if not requested:
        if maximum_transitions is not None:
            raise ValueError(
                "--policy-trace-max-transitions requires --policy-trace")
        return None
    if maximum_transitions is None:
        return DEFAULT_POLICY_TRACE_MAXIMUM_TRANSITIONS
    if isinstance(maximum_transitions, bool) or not isinstance(
        maximum_transitions, int
    ):
        raise TypeError("policy trace maximum transitions must be an integer")
    if maximum_transitions < 1:
        raise ValueError("policy trace maximum transitions must be positive")
    return maximum_transitions


def _validate_complete_policy_trace_requirement(
    policy_trace: bool,
    required: bool,
) -> bool:
    if not isinstance(required, bool):
        raise TypeError("require_complete_policy_trace must be a boolean")
    if required and not policy_trace:
        raise ValueError(
            "--require-complete-policy-trace requires --policy-trace")
    return required


def _contract_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value


def _contract_nonnegative_number(value: Any, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _contract_separation(value: Any) -> int | float | list[int | float]:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("starting_separation range must contain two values")
        low = _contract_nonnegative_number(value[0], "starting_separation[0]")
        high = _contract_nonnegative_number(value[1], "starting_separation[1]")
        if low > high:
            raise ValueError("starting_separation range must be ordered")
        return [low, high]
    return _contract_nonnegative_number(value, "starting_separation")


def _contract_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA-256 hex string")
    candidate = value.strip().upper()
    if len(candidate) != 64 or any(character not in "0123456789ABCDEF" for character in candidate):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex string")
    return candidate


def _load_task_contract(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as failure:
        raise ValueError(f"could not read task contract {path}: {failure}") from failure
    return _validate_task_contract(value)


def _require_free_port(
    port: int,
    kind: int,
    address: str = "127.0.0.1",
) -> None:
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("ports must be integers in 1..65535")
    parsed = ipaddress.ip_address(address)
    family = socket.AF_INET6 if parsed.version == 6 else socket.AF_INET
    endpoint = (parsed.compressed, port, 0, 0) if parsed.version == 6 else (
        parsed.compressed, port
    )
    with socket.socket(family, kind) as probe:
        probe.bind(endpoint)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _terminal_actor_failure(metrics: dict[str, Any]) -> str | None:
    claimed = int(metrics.get("claimed", 0))
    terminal = int(metrics.get("terminal_actors", 0))
    if claimed <= 0 or terminal < claimed:
        return None
    reason = metrics.get("last_terminal_reason") or "unknown"
    return (
        "all claimed policy actors reached a terminal state before the runtime "
        f"gate passed (last_terminal_reason={reason})"
    )


def _read_metrics_receipt(
    path: Path,
) -> tuple[dict[str, Any], dict[str, str] | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as failure:
        return {}, _metrics_read_error(path, "missing", failure)
    except (OSError, UnicodeError) as failure:
        return {}, _metrics_read_error(path, "unreadable", failure)
    except json.JSONDecodeError as failure:
        return {}, _metrics_read_error(path, "invalid", failure)
    if not isinstance(value, dict):
        return {}, _metrics_read_error(
            path,
            "invalid",
            TypeError("metrics.json must contain a JSON object"),
        )
    return value, None


def _runtime_surface(
    bundle_directory: Path,
    policy_jar: Path,
    bridge_jar: Path,
) -> dict[str, Any]:
    """Record identities of the files the native process actually receives."""
    return {
        "schema": _RUNTIME_SURFACE_SCHEMA,
        "artifacts": {
            "policy_jar": _runtime_file_identity(policy_jar),
            "bridge_jar": _runtime_file_identity(bridge_jar),
        },
        "bundle": _runtime_bundle_identity(bundle_directory),
    }


def _runtime_file_identity(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "sha256": None}
    try:
        if not path.is_file():
            result.update({
                "status": "missing",
                "reason": "runtime artifact is not a regular file",
            })
        else:
            result.update({"status": "present", "sha256": _sha256(path)})
    except (OSError, ValueError) as failure:
        result.update({
            "status": "unreadable",
            "error_type": type(failure).__name__,
            "reason": str(failure),
        })
    return result


def _runtime_bundle_identity(bundle_directory: Path) -> dict[str, Any]:
    """Expose only canonical identity already recorded by the copied bundle."""
    manifest_path = bundle_directory / "agent.json"
    result: dict[str, Any] = {
        "path": str(bundle_directory),
        "manifest_path": str(manifest_path),
        "status": "missing",
    }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as failure:
        result["reason"] = str(failure)
        return result
    except (OSError, UnicodeError, json.JSONDecodeError) as failure:
        result.update({
            "status": "unreadable",
            "error_type": type(failure).__name__,
            "reason": str(failure),
        })
        return result
    if not isinstance(manifest, dict):
        result.update({
            "status": "invalid",
            "reason": "agent.json must contain a JSON object",
        })
        return result

    tensor_identity = manifest.get("policy_tensor_identity")
    tensor_identity = tensor_identity if isinstance(tensor_identity, dict) else {}
    compatibility = manifest.get("deployment_compatibility")
    compatibility = compatibility if isinstance(compatibility, dict) else {}
    result.update({
        "status": "observed",
        "bundle_schema": manifest.get("schema"),
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "policy_tensor_content_sha256": tensor_identity.get("content_sha256"),
        "deployment_compatibility_content_sha256": compatibility.get(
            "content_sha256"
        ),
    })

    deployment = compatibility.get("deployment")
    deployment = deployment if isinstance(deployment, dict) else {}
    declared_profile_sha = deployment.get("profile_content_sha256")
    profile_directory = bundle_directory / "profile"
    if declared_profile_sha is not None or profile_directory.exists():
        profile: dict[str, Any] = {
            "path": str(profile_directory),
            "declared_content_sha256": declared_profile_sha,
            "observed_content_sha256": None,
            "matches_declared": False,
        }
        if profile_directory.is_dir():
            try:
                observed_profile_sha = profile_content_sha256(profile_directory)
            except (OSError, ValueError) as failure:
                profile.update({
                    "status": "unreadable",
                    "error_type": type(failure).__name__,
                    "reason": str(failure),
                })
            else:
                profile.update({
                    "status": "observed",
                    "observed_content_sha256": observed_profile_sha,
                    "matches_declared": (
                        declared_profile_sha == observed_profile_sha
                    ),
                })
        else:
            profile.update({
                "status": "missing",
                "reason": "profile directory is not present",
            })
        result["profile"] = profile
    return result


def _policy_trace_completeness(
    paths: list[Path],
    *,
    requested: bool,
    maximum_transitions: int | None = None,
    policy_ticks: int | None = None,
) -> dict[str, Any]:
    """Validate the actual native JSONL receipt, including its terminal tail."""
    expected_policy_ticks = (
        policy_ticks
        if type(policy_ticks) is int and policy_ticks >= 0
        else None
    )
    result: dict[str, Any] = {
        "schema": _POLICY_TRACE_COMPLETENESS_SCHEMA,
        "requested": requested,
        "trace_schema": _POLICY_TRACE_SCHEMA,
        "files": [str(path) for path in paths],
        "header_valid": None if not requested else False,
        "rows_structurally_parseable": None if not requested else False,
        "sequence_consistent": None if not requested else False,
        "final_row_terminal": None if not requested else False,
        "complete": None if not requested else False,
        "nonterminal_budget_satisfied": None if not requested else False,
        "terminal_allowance_satisfied": None if not requested else False,
        "terminal_row_after_budget": None if not requested else False,
        "configured_maximum_transitions": (
            maximum_transitions if requested else None),
        "recorded_maximum_transitions": None,
        "policy_ticks": expected_policy_ticks if requested else None,
        "nonterminal_transition_rows": None if not requested else 0,
        "terminal_rows": None if not requested else 0,
        "missing_policy_tick_rows": (
            expected_policy_ticks if requested else None),
        "reason": "policy_trace_not_requested" if not requested else None,
    }
    if not requested:
        return result
    if not paths:
        result["reason"] = "missing_policy_trace"
        return result
    if len(paths) != 1:
        result["reason"] = "multiple_policy_trace_files"
        return result

    path = paths[0]
    rows, read_error = _read_policy_trace_jsonl(path)
    result["header_valid"] = _valid_policy_trace_header(rows)
    if read_error is not None:
        result.update({"reason": read_error["kind"], "error": read_error})
        return result
    if not result["header_valid"]:
        result["reason"] = "invalid_policy_trace_header"
        return result
    expected_maximum = (
        DEFAULT_POLICY_TRACE_MAXIMUM_TRANSITIONS
        if maximum_transitions is None
        else maximum_transitions
    )
    recorded_maximum = rows[0].get("maximum_transitions")
    if type(recorded_maximum) is int:
        result["recorded_maximum_transitions"] = recorded_maximum
    if (type(recorded_maximum) is not int or recorded_maximum < 1
            or recorded_maximum != expected_maximum):
        result["reason"] = "policy_trace_maximum_transitions_mismatch"
        return result
    transitions = rows[1:]
    result["transition_rows"] = len(transitions)

    try:
        if transitions:
            _validate_transition_rows(transitions, rows[0])
    except Exception as failure:
        message = str(failure)
        if any(marker in message for marker in _POLICY_TRACE_SEQUENCE_ERRORS):
            result["rows_structurally_parseable"] = True
            result["reason"] = "policy_trace_sequence_inconsistent"
        else:
            result["reason"] = "malformed_policy_trace_rows"
        result["error"] = {
            "error_type": type(failure).__name__,
            "message": message,
        }
        return result

    result["rows_structurally_parseable"] = True
    result["sequence_consistent"] = True
    terminal_rows = sum(
        int(_is_valid_terminal_row(row)) for row in transitions
    )
    nonterminal_rows = len(transitions) - terminal_rows
    result["terminal_rows"] = terminal_rows
    result["nonterminal_transition_rows"] = nonterminal_rows
    result["nonterminal_budget_satisfied"] = (
        nonterminal_rows <= expected_maximum)
    result["terminal_allowance_satisfied"] = terminal_rows <= 1
    result["terminal_row_after_budget"] = (
        terminal_rows == 1 and nonterminal_rows >= expected_maximum)
    if expected_policy_ticks is not None:
        result["missing_policy_tick_rows"] = max(
            0, expected_policy_ticks - len(transitions))
    if nonterminal_rows > expected_maximum:
        result["reason"] = "policy_trace_nonterminal_budget_exceeded"
        return result
    if terminal_rows > 1:
        result["reason"] = "policy_trace_terminal_allowance_exceeded"
        return result
    if (expected_policy_ticks is not None
            and len(transitions) != expected_policy_ticks):
        result["reason"] = "policy_trace_policy_tick_rows_mismatch"
        return result
    if not transitions:
        result["reason"] = "no_policy_trace_transition_rows"
        return result
    result["final_row_terminal"] = _is_valid_terminal_row(transitions[-1])
    result["complete"] = result["final_row_terminal"]
    result["reason"] = (
        "complete" if result["complete"] else "trace_ends_without_terminal"
    )
    return result


def _policy_trace_policy_ticks(metrics: dict[str, Any]) -> int | None:
    value = metrics.get("policy_ticks")
    return value if type(value) is int and value >= 0 else None


def _read_policy_trace_jsonl(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(
                        line,
                        parse_constant=_reject_policy_trace_constant,
                    )
                except (UnicodeError, ValueError) as failure:
                    return rows, {
                        "kind": "malformed_policy_trace_json",
                        "line": line_number,
                        "error_type": type(failure).__name__,
                        "message": str(failure),
                    }
                if not isinstance(value, dict):
                    return rows, {
                        "kind": "policy_trace_row_not_object",
                        "line": line_number,
                        "error_type": "TypeError",
                        "message": "policy trace rows must be JSON objects",
                    }
                rows.append(value)
    except FileNotFoundError as failure:
        return rows, {
            "kind": "missing_policy_trace",
            "error_type": type(failure).__name__,
            "message": str(failure),
        }
    except (OSError, UnicodeError) as failure:
        return rows, {
            "kind": "unreadable_policy_trace",
            "error_type": type(failure).__name__,
            "message": str(failure),
        }
    return rows, None


def _reject_policy_trace_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number {value!r}")


def _valid_policy_trace_header(rows: list[dict[str, Any]]) -> bool:
    if not rows or rows[0].get("event") != "header":
        return False
    if rows[0].get("schema") != _POLICY_TRACE_SCHEMA:
        return False
    try:
        _validate_header(rows[0])
    except (TypeError, ValueError, KeyError):
        return False
    return True


def _metrics_read_error(
    path: Path,
    kind: str,
    failure: BaseException,
) -> dict[str, str]:
    return {
        "schema": "hytalerl_metrics_read_error_v1",
        "path": str(path),
        "kind": kind,
        "error_type": type(failure).__name__,
        "message": str(failure),
    }


def _combat_requested(metrics: dict[str, Any]) -> bool:
    """Report combat from cumulative receipts, with legacy-field fallback."""
    return bool(
        int(metrics.get("combat_action_requests", 0)) > 0
        or int(metrics.get("combat_attack_accepted", 0)) > 0
        or int(metrics.get("combat_ability_accepted", 0)) > 0
        or int(metrics.get("combat_dodge_accepted", 0)) > 0
        or int(metrics.get("combat_guard_started", 0)) > 0
        or bool(metrics.get("last_attack"))
        or int(metrics.get("last_ability_slot", -1)) >= 0
    )


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


def _transition_line_count(path: Path) -> int:
    """Count transition rows in a policy trace, excluding its header."""
    return max(0, _line_count(path) - 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _artifact_sha256(path: Path) -> str:
    """Hash one file or a directory tree with relative names included."""
    if path.is_symlink():
        raise ValueError(f"deployment input cannot be a symlink: {path}")
    if path.is_file():
        return _sha256(path)
    if not path.is_dir():
        raise ValueError(f"deployment input is not a file or directory: {path}")

    digest = hashlib.sha256(b"HYTALERL_DEPLOYMENT_DIRECTORY\0")
    children = sorted(
        path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()
    )
    for child in children:
        if child.is_symlink():
            raise ValueError(f"deployment directory contains a symlink: {child}")
        if child.is_dir():
            continue
        if not child.is_file():
            raise ValueError(f"deployment directory contains a non-file: {child}")
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(child)))
    return digest.hexdigest().upper()


def _bind_endpoint(address: str, port: int) -> str:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError as error:
        raise ValueError("bind_address must be an IPv4 or IPv6 address") from error
    if parsed.version == 6:
        return f"[{parsed.compressed}]:{port}"
    return f"{parsed.compressed}:{port}"


def _stage_world_or_universe(source: Path, destination: Path) -> None:
    """Copy a source world/universe into the run's private universe directory."""
    if (source / "worlds").is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    elif (source / "config.json").is_file():
        shutil.copytree(
            source,
            destination / "worlds/default",
        )
    else:
        raise ValueError(
            "world_or_universe must contain a 'worlds' directory or a world "
            "config.json"
        )


def _write_status(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
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
    parser.add_argument("--bind-address", default="127.0.0.1",
                        help="server bind IP; default stays local to this machine")
    parser.add_argument("--world-or-universe", type=Path,
                        help="source world or universe directory to copy into this run")
    parser.add_argument("--minimum-policy-ticks", type=int, default=120)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE,
                        help="combat fixture roles and bounded native placement")
    parser.add_argument("--pursuit-context", action="store_true",
                        help="expose the trained pursuit target pose through occlusion")
    parser.add_argument("--flat-world", action="store_true",
                        help="create an isolated native flat world with floor at y=65")
    parser.add_argument("--maximum-target-distance", type=float,
                        help="require approaching within this many planar blocks")
    parser.add_argument("--policy-trace", action="store_true",
                        help="record native policy observations, masks, and actions")
    parser.add_argument(
        "--require-complete-policy-trace",
        action="store_true",
        help=(
            "keep the server running after the runtime gate until the native "
            "trace has a valid terminal row; requires --policy-trace"
        ),
    )
    parser.add_argument(
        "--policy-trace-max-transitions",
        dest="policy_trace_maximum_transitions",
        type=int,
        help="positive nonterminal row budget (default: 512); requires --policy-trace",
    )
    parser.add_argument(
        "--task-contract",
        type=Path,
        help="JSON object declaring optional pursuit-transfer task fields",
    )
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args(argv)
    if args.require_complete_policy_trace and not args.policy_trace:
        parser.error("--require-complete-policy-trace requires --policy-trace")
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
        bind_address=args.bind_address,
        world_or_universe=args.world_or_universe,
        minimum_policy_ticks=args.minimum_policy_ticks,
        fixture=args.fixture,
        pursuit_context=args.pursuit_context,
        flat_world=args.flat_world,
        maximum_target_distance=args.maximum_target_distance,
        policy_trace=args.policy_trace,
        policy_trace_maximum_transitions=(
            args.policy_trace_maximum_transitions),
        require_complete_policy_trace=args.require_complete_policy_trace,
        timeout_seconds=args.timeout_seconds,
        task_contract=_load_task_contract(args.task_contract),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "run"]
