"""Agents discovered from `agents/`, and what each one needs from the console.

An agent in this repo is a directory under `agents/` — `combat`, `dawn`,
`design` today, and whatever comes next. The console had no idea they existed,
so every agent got the same screens whether or not they made sense for it.

**An agent declares its own console needs in `agents/<name>/console/`.** That
directory is the agent's half of the contract:

    agents/dawn/console/
      agent.json      what this agent is and what it wants shown
      panels.py       optional — panels only meaningful for this agent

`agent.json` is small and every field optional::

    {
      "title": "Dawn",
      "architecture": "dreamer",
      "summary": "World-model agent with an imagination rollout.",
      "run": {"world": "region", "zone": "353404383:51", "ticks": 512},
      "metrics": ["world_model_loss", "imagined_return", "kl"],
      "streams": ["dawn-*"],
      "entrypoint": "agents.dawn.train"
    }

Nothing is required and nothing is validated against a list of known
architectures. An agent that this console cannot run yet still declares what it
would need, and that record is the useful part — the alternative is a hardcoded
set of architectures that has to be edited every time someone adds one.

Discovery is a directory scan on every call, so an agent added while the server
is running appears without a restart, exactly like panels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]

#: One directory per agent.
AGENT_ROOT = _ROOT / "agents"

#: The folder an agent puts its console integration in.
CONSOLE_DIR = "console"
MANIFEST = "agent.json"

#: Directories under `agents/` that are not agents.
IGNORED = {"__pycache__", "profiles"}


def index() -> list[dict[str, Any]]:
    """Every agent directory, with its console manifest if it declared one."""

    if not AGENT_ROOT.is_dir():
        return []

    found: list[dict[str, Any]] = []
    for path in sorted(AGENT_ROOT.iterdir()):
        if not path.is_dir() or path.name in IGNORED or path.name.startswith("."):
            continue

        console_dir = path / CONSOLE_DIR
        manifest_path = console_dir / MANIFEST
        entry: dict[str, Any] = {
            "id": path.name,
            "title": path.name,
            "architecture": "unspecified",
            "path": str(path),
            # An agent with no `console/` is not broken -- it simply has not
            # said what it wants, and the console shows it as-is rather than
            # inventing needs on its behalf.
            "declares_console": console_dir.is_dir(),
            "has_panels": (console_dir / "panels.py").is_file(),
            "run": {},
            "metrics": [],
            "streams": [],
        }

        if manifest_path.is_file():
            try:
                raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                entry["error"] = f"{manifest_path.name}: {exc}"
                found.append(entry)
                continue
            if isinstance(raw, dict):
                entry.update({
                    "title": raw.get("title") or path.name,
                    "architecture": str(raw.get("architecture") or "unspecified"),
                    "summary": raw.get("summary") or "",
                    "run": raw.get("run") or {},
                    "metrics": list(raw.get("metrics") or []),
                    "streams": list(raw.get("streams") or []),
                    "entrypoint": raw.get("entrypoint") or "",
                })
            else:
                entry["error"] = f"{MANIFEST} must be a JSON object"

        found.append(entry)
    return found


def get(agent_id: str) -> dict[str, Any] | None:
    return next((a for a in index() if a["id"] == agent_id), None)


def panel_modules() -> list[Path]:
    """`panels.py` files agents shipped, for the panel loader to pick up."""

    return [
        path / CONSOLE_DIR / "panels.py"
        for path in sorted(AGENT_ROOT.iterdir())
        if path.is_dir() and path.name not in IGNORED
        and (path / CONSOLE_DIR / "panels.py").is_file()
    ] if AGENT_ROOT.is_dir() else []
