"""Resident WSL/JAX worker for sequential Basic-agent training requests."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import sys
import threading
import traceback
from contextlib import contextmanager
from pathlib import Path

from arena.training.runs.pursuit_run import PursuitRunConfig, run_pursuit_training


PROTOCOL = "HYTALERL_BASIC_WORKER:"
ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = (
    ROOT / "agents" / "basic",
    ROOT / "arena",
    ROOT / "HytaleRL" / "hytalegym" / "hytalegym",
    ROOT / "experimental" / "worldgen-v2",
)


def _config(path: Path) -> PursuitRunConfig:
    values = json.loads(path.read_text(encoding="utf-8-sig"))
    values["source_policy"] = Path(values["source_policy"])
    values["output"] = Path(values["output"])
    for field in ("evaluation_updates", "memory_reset_updates"):
        if field in values:
            values[field] = tuple(values[field])
    return PursuitRunConfig(**values)


def _training_sources(paths: tuple[Path, ...] | None = None) -> dict[str, str]:
    if paths is None:
        paths = tuple(
            sorted(
                path
                for source_root in SOURCE_ROOTS
                if source_root.is_dir()
                for path in source_root.rglob("*.py")
                if not {"tests", "artifacts", "__pycache__"} & set(path.parts)
            )
        )
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): (
            hashlib.sha256(path.read_bytes()).hexdigest().upper()
            if path.is_file()
            else "MISSING"
        )
        for path in paths
    }


def _source_sha256(snapshot: dict[str, str]) -> str:
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest().upper()


@contextmanager
def _job_streams(stdout_path: Path, stderr_path: Path):
    sys.stdout.flush()
    sys.stderr.flush()
    saved_descriptors = os.dup(1), os.dup(2)
    saved_streams = sys.stdout, sys.stderr
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            os.dup2(stdout.fileno(), 1)
            os.dup2(stderr.fileno(), 2)
            # `sys.stdout` was bound when descriptor 1 was a pipe, so it keeps a
            # block buffer and holds the run's output in memory until the
            # process exits. Rebinding to line-buffered writers over the
            # redirected descriptors puts each line on disk as it is written,
            # which is what makes the log readable while the run is going.
            sys.stdout = open(1, "w", encoding="utf-8", buffering=1, closefd=False)
            sys.stderr = open(2, "w", encoding="utf-8", buffering=1, closefd=False)
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
    finally:
        sys.stdout, sys.stderr = saved_streams
        os.dup2(saved_descriptors[0], 1)
        os.dup2(saved_descriptors[1], 2)
        os.close(saved_descriptors[0])
        os.close(saved_descriptors[1])


def _emit(value: dict[str, object]) -> None:
    print(PROTOCOL + json.dumps(value, separators=(",", ":")), flush=True)


def _requests(idle_seconds: int):
    lines: queue.Queue[str | None] = queue.Queue()

    def read() -> None:
        for line in sys.stdin:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=read, daemon=True).start()
    while True:
        try:
            line = lines.get(timeout=idle_seconds)
        except queue.Empty:
            return
        if line is None:
            return
        yield line


def serve(*, idle_seconds: int = 300) -> None:
    """Serve one request at a time and release GPU memory after an idle window."""

    source_snapshot = _training_sources()
    previous_contract: str | None = None
    runs = 0
    for line in _requests(idle_seconds):
        if not line.strip():
            continue
        request = json.loads(line)
        request_id = str(request["id"])
        paths = tuple(ROOT / path for path in source_snapshot)
        if _training_sources(paths) != source_snapshot:
            _emit({"id": request_id, "status": "restart_required"})
            return
        before_run = source_snapshot
        contract = str(request["prepared_cache_key"])
        response: dict[str, object]
        with _job_streams(Path(request["stdout_path"]), Path(request["stderr_path"])):
            try:
                report = run_pursuit_training(_config(Path(request["spec_path"])))
                print(json.dumps({"status": report["status"]}), flush=True)
                response = {"status": "ok", "report_status": report["status"]}
            except Exception as error:  # the response must survive the failed run
                traceback.print_exc()
                response = {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
        runs += 1
        after_run = _training_sources()
        if after_run != before_run:
            response = {
                "status": "error",
                "error_type": "SourceDrift",
                "message": "training source changed while the request was running",
            }
        response.update(
            id=request_id,
            process_run=runs,
            process_reused=runs > 1,
            contract_reused=previous_contract == contract,
            source_sha256=_source_sha256(after_run),
        )
        source_snapshot = after_run
        previous_contract = contract
        _emit(response)


if __name__ == "__main__":
    serve()


__all__ = ["PROTOCOL", "serve"]
