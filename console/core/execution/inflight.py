"""Read staged run scripts without executing them or learning their lane."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

UNDECLARED = "undeclared"
_ASSIGNMENT = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)=(?:\"([^\"]*)\"|'([^']*)'|(\S+))\s*$"
)
_OUTPUT_DIR = re.compile(r"--output-dir\s+(?:\"([^\"]+)\"|'([^']+)'|(\S+))")
_VARIABLE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*|[0-9]+))")


def scan_inflight(
    snapshot_root: Path,
    *,
    workspace: Path,
    running: dict[str, list[int]] | None = None,
) -> list[dict[str, Any]]:
    """List each staged shell entry point and its artifact state.

    ``running`` is injectable so classification is deterministic in tests.  In
    production it is a read-only process-table snapshot keyed by script name.
    """

    active = _running_scripts() if running is None else running
    rows = []
    if not snapshot_root.is_dir():
        return rows
    for script in sorted(snapshot_root.glob("*.sh")):
        try:
            text = script.read_text(encoding="utf-8")
            stat = script.stat()
        except OSError:
            continue
        target_text = declared_output(text)
        target = _host_path(target_text, workspace) if target_text else None
        concrete = target is not None and "$" not in target_text and "<arg" not in target_text
        pids = active.get(script.name, [])
        if pids:
            state = "running"
        elif concrete and target is not None and _has_output(target):
            state = "landed"
        else:
            state = "staged"

        snapshot = declared_snapshot(text, snapshot_root=snapshot_root, workspace=workspace)
        updated = stat.st_mtime
        if concrete and target is not None and target.exists():
            try:
                updated = max(updated, target.stat().st_mtime)
            except OSError:
                pass
        rows.append({
            "name": script.stem.removeprefix("run_"),
            "script": script,
            "state": state,
            "pids": pids,
            "target": target,
            "target_declared": bool(target_text),
            "target_display": _display_target(target, target_text, workspace),
            "snapshot": snapshot,
            "updated_at": updated,
        })
    return rows


def declared_output(script: str) -> str | None:
    """Resolve a literal/simple-variable ``--output-dir`` declaration."""

    variables: dict[str, str] = {}
    for line in script.splitlines():
        match = _ASSIGNMENT.match(line)
        if not match:
            continue
        name = match.group(1)
        value = next(group for group in match.groups()[1:] if group is not None)
        variables[name] = _expand(value, variables)

    flattened = " ".join(
        line.strip().removesuffix("\\").strip() for line in script.splitlines()
    )
    match = _OUTPUT_DIR.search(flattened)
    if not match:
        return None
    value = next(group for group in match.groups() if group is not None)
    return _expand(value, variables)


def declared_snapshot(
    script: str, *, snapshot_root: Path, workspace: Path,
) -> Path | None:
    """Return an existing frozen-source directory literally named by a script."""

    # A script may name the directory in an assignment, a ``cd`` command, or a
    # PYTHONPATH.  Literal path discovery covers all three without executing
    # shell expansions or assuming a relationship between run and lane names.
    literals = re.findall(
        r"(?:[A-Za-z]:[\\/]|/mnt/[A-Za-z]/)[^\"'\s:]+", script,
    )
    resolved_root = snapshot_root.resolve()
    for literal in literals:
        candidate = _host_path(literal.rstrip("\\"), workspace)
        if candidate is None or not candidate.is_dir():
            continue
        try:
            candidate.resolve().relative_to(resolved_root)
        except ValueError:
            continue
        return candidate
    return None


def _expand(value: str, variables: dict[str, str]) -> str:
    expanded = value
    for _ in range(8):
        changed = False

        def replace(match: re.Match[str]) -> str:
            nonlocal changed
            name = match.group(1) or match.group(2)
            replacement = variables.get(name, f"<arg{name}>" if name.isdigit() else f"${name}")
            changed = changed or replacement != match.group(0)
            return replacement

        updated = _VARIABLE.sub(replace, expanded)
        expanded = updated
        if not changed:
            break
    return expanded


def _host_path(value: str | None, workspace: Path) -> Path | None:
    if not value or "$" in value or "<arg" in value:
        return None
    match = re.match(r"^/mnt/([A-Za-z])/(.*)$", value)
    if match:
        drive, rest = match.groups()
        return Path(f"{drive.upper()}:/{rest}")
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def _display_target(target: Path | None, declared: str | None, workspace: Path) -> str:
    if target is not None:
        try:
            return target.relative_to(workspace).as_posix()
        except ValueError:
            return str(target)
    return declared or UNDECLARED


def _has_output(target: Path) -> bool:
    if target.is_file():
        return True
    if not target.is_dir():
        return False
    try:
        return any(path.is_file() for path in target.rglob("*"))
    except OSError:
        return False


def _running_scripts() -> dict[str, list[int]]:
    active: dict[str, list[int]] = {}
    try:
        import psutil
    except ImportError:  # pragma: no cover - optional deployment dependency
        return active
    try:
        # Reading every process command line is unexpectedly expensive on
        # Windows.  Names are cheap; only shell/WSL candidates need the second
        # query because these entry points are ``.sh`` files by contract.
        shell_names = {
            "wsl", "wsl.exe", "bash", "bash.exe", "sh", "sh.exe",
            "zsh", "zsh.exe", "dash", "dash.exe",
        }
        processes = psutil.process_iter(["pid", "name"])
        for process in processes:
            try:
                if (process.info.get("name") or "").lower() not in shell_names:
                    continue
                command = " ".join(process.cmdline())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            for match in re.finditer(r"(?:^|[\\/\s])([^\\/\s]+\.sh)(?=$|\s)", command):
                active.setdefault(match.group(1), []).append(int(process.info["pid"]))
    except (psutil.Error, OSError):  # pragma: no cover - platform permissions
        return {}
    return active
