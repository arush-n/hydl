#!/usr/bin/env python3
"""Launch a bridge-only Hytale 0.7.2 server with the HytaleRLBridge port jar.

Stdlib only. Mirrors the launch and stop logic of adk/deploy/isolated.py
run() and _stop(), without the policy bundle, policy jar or policy markers,
which a bridge-only run cannot satisfy.

The run directory must sit outside every install tree: the 0.7.2
InstallDirGuard exits with code 11 when the working directory is under an
install folder that holds env.dat.

Example (server, then a smoke command, then a clean stop):
    python3 tools/launch_bridge_072.py --run-dir <new-dir> \\
        --then "/path/to/python tools/smoke_bridge_072.py"
Without --then the server is held until SIGINT or SIGTERM.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

DEFAULT_INSTALL = Path(
    "/Users/arush/Library/Application Support/Hytale/install/release/package/game/latest"
)
DEFAULT_JAR = Path(__file__).resolve().parents[1] / "build/libs/HytaleRLBridge072-0.1.0.jar"
# BridgeConfig.PORT_PROPERTY in src/main/java/com/hytalerlbridge/BridgeConfig.java.
BRIDGE_PORT_PROPERTY = "hytalerl.bridge.port"
# HytaleServer.java boot line and BridgeServer.java bind line.
READY_MARKERS = ("Hytale Server Booted!", "Bridge server listening on ")
FAILURE_MARKERS = (
    "HytaleRLBridge failed to bind",
    "Failed to start bridge server",
    "Failed to load config.yml",
    "Skipping mod HytaleRL:HytaleRLBridge",
    "will not boot inside the install path",
    "Failed to start server!",
)
FAILURE_WORDS = ("failed", "severe", "error", "exception", "disabled")


def _install_tree_of(path: Path) -> Path | None:
    """Mirror InstallDirGuard.findInstallDir: an ancestor-or-self whose parent is named install and holds env.dat."""
    for directory in (path, *path.parents):
        if directory.parent.name.lower() == "install" and (directory / "env.dat").is_file():
            return directory
    return None


def _check_run_dir(run_dir: Path, install: Path) -> None:
    if run_dir == install or install in run_dir.parents:
        raise SystemExit(f"run-dir {run_dir} is inside the install folder {install}")
    tree = _install_tree_of(run_dir)
    if tree is not None:
        raise SystemExit(f"run-dir {run_dir} is under install tree {tree}; InstallDirGuard exits 11")


def _require_free_port(port: int, kind: int) -> None:
    with socket.socket(socket.AF_INET, kind) as probe:
        probe.bind(("127.0.0.1", port))


def _wait_ready(process: subprocess.Popen[str], log_path: Path, timeout: float) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout
    while True:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            lowered = line.lower()
            if any(marker in line for marker in FAILURE_MARKERS) or (
                "hytalerl" in lowered and any(word in lowered for word in FAILURE_WORDS)
            ):
                return False, f"failure line: {line.strip()}"
        if all(marker in text for marker in READY_MARKERS):
            return True, "boot and bridge-listening markers seen"
        if process.poll() is not None:
            return False, f"server exited with code {process.returncode} before ready"
        if time.monotonic() >= deadline:
            return False, f"timed out after {timeout:.0f}s waiting for {READY_MARKERS}"
        time.sleep(0.5)


def _stop(process: subprocess.Popen[str]) -> tuple[int, str]:
    if process.poll() is not None:
        return process.returncode, "already-exited"
    if process.stdin is not None:
        try:
            process.stdin.write("stop\n")
            process.stdin.flush()
            return process.wait(timeout=45), "stop-command"
        except (OSError, subprocess.TimeoutExpired):
            pass
    process.terminate()
    try:
        return process.wait(timeout=15), "terminate"
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=10), "kill"


def _first_lines(text: str, markers: tuple[str, ...]) -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for marker in markers:
        found[marker] = next((line.strip() for line in text.splitlines() if marker in line), None)
    return found


def _interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def build_server_command(
    java: str,
    server_jar: Path,
    assets: Path,
    universe: Path,
    bridge_port: int,
    game_port: int,
    early_plugins: Path | None = None,
) -> list[str]:
    """The server command line. ``early_plugins`` adds the early-plugin flags (the sound probe); absent, nothing changes."""

    command = [
        java,
        "-Xshare:off",
        f"-D{BRIDGE_PORT_PROPERTY}={bridge_port}",
        "-jar",
        str(server_jar),
        "--assets",
        str(assets),
        "--disable-sentry",
        "--auth-mode",
        "offline",
        "--bind",
        f"127.0.0.1:{game_port}",
        "--universe",
        str(universe),
    ]
    if early_plugins is not None:
        command += ["--accept-early-plugins", f"--early-plugins={early_plugins}"]
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch a bridge-only Hytale 0.7.2 server.")
    parser.add_argument("--install", type=Path, default=DEFAULT_INSTALL)
    parser.add_argument("--jar", type=Path, default=DEFAULT_JAR)
    parser.add_argument("--run-dir", type=Path, required=True, help="new or empty directory outside any install tree")
    parser.add_argument("--bridge-port", type=int, default=5556)
    parser.add_argument("--game-port", type=int, default=25572)
    parser.add_argument("--java", default="java")
    parser.add_argument("--ready-timeout", type=float, default=180.0)
    parser.add_argument("--then", dest="then_command", default=None,
                        help="shell-style command run after ready; the server is stopped after it exits")
    parser.add_argument("--then-timeout", type=float, default=1800.0)
    parser.add_argument("--sound-probe-jar", type=Path, default=None,
                        help="optional early-plugin jar (HytaleRLSoundProbe); copied to <run-dir>/earlyplugins")
    args = parser.parse_args(argv)

    install = args.install.resolve()
    server_jar = install / "Server" / "HytaleServer.jar"
    assets = install / "Assets.zip"
    required_files = [server_jar, assets, args.jar]
    if args.sound_probe_jar is not None:
        required_files.append(args.sound_probe_jar)
    for required in required_files:
        if not required.is_file():
            parser.error(f"missing file: {required}")
    java = shutil.which(args.java)
    if java is None:
        parser.error(f"java not found: {args.java}")
    run_dir = args.run_dir.resolve()
    _check_run_dir(run_dir, install)
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error(f"run-dir must be new or empty: {run_dir}")
    _require_free_port(args.bridge_port, socket.SOCK_STREAM)
    _require_free_port(args.game_port, socket.SOCK_DGRAM)

    (run_dir / "mods").mkdir(parents=True, exist_ok=True)
    (run_dir / "universe").mkdir(exist_ok=True)
    shutil.copy2(args.jar, run_dir / "mods" / args.jar.name)
    early_plugins = None
    if args.sound_probe_jar is not None:
        early_plugins = run_dir / "earlyplugins"
        early_plugins.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.sound_probe_jar, early_plugins / args.sound_probe_jar.name)

    command = build_server_command(
        java, server_jar, assets, run_dir / "universe", args.bridge_port, args.game_port, early_plugins
    )
    log_path = run_dir / "server.log"
    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "bridge_port": args.bridge_port,
        "game_port": args.game_port,
        "command": command,
    }
    ready, detail, then_code, error = False, "", None, None
    signal.signal(signal.SIGTERM, _interrupt)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            command,
            cwd=run_dir,
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        report["server_pid"] = process.pid
        print(f"started server pid={process.pid} run_dir={run_dir}", flush=True)
        try:
            ready, detail = _wait_ready(process, log_path, args.ready_timeout)
            print(f"{'READY' if ready else 'NOT READY'}: {detail}", flush=True)
            if ready and args.then_command:
                env = dict(
                    os.environ,
                    HYTALERL_BRIDGE_HOST="127.0.0.1",
                    HYTALERL_BRIDGE_PORT=str(args.bridge_port),
                    HYTALERL_GAME_PORT=str(args.game_port),
                    HYTALERL_SERVER_PID=str(process.pid),
                )
                then_code = subprocess.run(
                    shlex.split(args.then_command), env=env, timeout=args.then_timeout
                ).returncode
                print(f"--then exited with code {then_code}", flush=True)
            elif ready:
                print("holding server; send SIGINT or SIGTERM to stop", flush=True)
                while process.poll() is None:
                    time.sleep(1.0)
        except (KeyboardInterrupt, subprocess.TimeoutExpired) as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            exit_code, stop_mode = _stop(process)
    print(f"server exit code={exit_code} stop={stop_mode}", flush=True)

    text = log_path.read_text(encoding="utf-8", errors="replace")
    report.update({
        "ready": ready,
        "ready_detail": detail,
        "server_exit_code": exit_code,
        "stop_mode": stop_mode,
        "then_exit_code": then_code,
        "error": error,
        "markers": _first_lines(text, READY_MARKERS + ("HytaleRLBridge",)),
    })
    (run_dir / "launch-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    ok = ready and error is None and exit_code == 0 and stop_mode in ("stop-command", "already-exited") and then_code in (None, 0)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
