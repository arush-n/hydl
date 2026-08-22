"""One reusable WSL process for serial CUDA jobs."""

from __future__ import annotations

import atexit
import json
import os
import queue
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any


PROTOCOL = "HYTALERL_BASIC_WORKER:"


class ResidentWslWorker:
    """Keep imports, CUDA context, and JAX executable caches warm between jobs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self._diagnostics: list[str] = []
        self._metadata: dict[str, Any] | None = None
        self._started_unix_seconds: float | None = None
        self._last_request_unix_seconds: float | None = None
        self._last_completed_unix_seconds: float | None = None
        self._last_response: dict[str, Any] | None = None
        atexit.register(self.close)

    def _start(self, command: str, metadata: dict[str, Any]) -> None:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self._responses = queue.Queue()
        self._diagnostics = []
        self._metadata = dict(metadata)
        self._started_unix_seconds = time.time()
        self._process = subprocess.Popen(
            ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=flags,
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            text = line.rstrip()
            if text.startswith(PROTOCOL):
                self._responses.put(json.loads(text[len(PROTOCOL) :]))
            elif text:
                self._diagnostics.append(text)

    def run(
        self,
        command: str,
        request: dict[str, Any],
        *,
        poll: Callable[[], None],
        launch_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Submit one request, polling Console state without blocking training."""

        with self._lock:
            for attempt in range(2):
                if self._process is None or self._process.poll() is not None:
                    self.close()
                    self._start(command, launch_metadata)
                process = self._process
                assert process is not None and process.stdin is not None
                self._last_request_unix_seconds = time.time()
                process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
                process.stdin.flush()
                while process.poll() is None:
                    try:
                        response = self._responses.get(timeout=0.5)
                    except queue.Empty:
                        poll()
                        continue
                    if response.get("id") != request["id"]:
                        continue
                    if response.get("status") == "restart_required" and attempt == 0:
                        self.close()
                        break
                    response["launch_metadata"] = self._metadata
                    self._last_completed_unix_seconds = time.time()
                    self._last_response = dict(response)
                    return response
                else:
                    details = self._diagnostics[-3:]
                    raise RuntimeError(
                        "resident WSL worker exited before responding"
                        + (f": {' | '.join(details)}" if details else "")
                    )
            raise RuntimeError("resident WSL worker could not refresh changed source")

    def status(self) -> dict[str, Any]:
        """Non-blocking observability; a live training call holds the run lock."""

        process = self._process
        running = process is not None and process.poll() is None
        last = self._last_response or {}
        return {
            "schema": "console-resident-wsl-worker-v1",
            "running": running,
            "pid": process.pid if running else None,
            "reusable": True,
            "idle_timeout_seconds": 300,
            "started_unix_seconds": self._started_unix_seconds,
            "last_request_unix_seconds": self._last_request_unix_seconds,
            "last_completed_unix_seconds": self._last_completed_unix_seconds,
            "process_run": last.get("process_run", 0),
            "process_reused": bool(last.get("process_reused", False)),
            "contract_reused": bool(last.get("contract_reused", False)),
            "source_sha256": last.get("source_sha256"),
            "allocator": dict(self._metadata or {}),
        }

    def close(self) -> None:
        process, self._process = self._process, None
        self._metadata = None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)


basic_worker = ResidentWslWorker()


__all__ = ["ResidentWslWorker", "basic_worker"]
