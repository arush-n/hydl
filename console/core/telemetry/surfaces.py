"""What the console reports about the ADK and the live server.

The console is a **frontend for the ADK**, not a parallel implementation. Every
value here is read from the ADK's own published surface — scenes describe
themselves through :func:`adk.describe_scene`, loadouts come from
:func:`adk.list_loadouts`, the action space from ``HEAD_SPANS``. Nothing is
restated as a literal, so the page cannot drift from the library.

Native (server-side) status is read-only and **never takes the lease**. Another
lane running the native suite holds `127.0.0.1:5556`; probing it here must not
disturb that, so this module only asks whether a socket is listening and what
the ADK's own fidelity check says about the deployed jar.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
for _entry in (_ROOT, _ROOT / "HytaleRL" / "hytalegym"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import adk  # noqa: E402
from adk.policy.fields import GROUP_FEATURES  # noqa: E402
from adk.probes.policies import HEAD_SPANS  # noqa: E402

#: Ports the bridge listens on. 5556 is the native server (loads the deployed
#: jar from the server's mods/), 5557 the simulator launched from build/libs.
BRIDGE_PORTS = {5556: "native", 5557: "simulator"}

#: Where the in-server policy mod publishes its counters. It writes a JSON file
#: rather than opening a port, so the console stays a pure reader and the
#: numbers survive the server going down — which is exactly when the last known
#: state is most worth seeing. ``HYTALE_POLICY_AGENT_DIR`` points at an
#: alternate instance (an isolated test server, for example).
POLICY_AGENT_METRICS = "metrics.json"
POLICY_AGENT_DIRECTORIES = (
    Path.home()
    / "AppData/Roaming/Hytale/install/release/package/game/latest/Server/mods/policy-agent",
)

#: Older than this and the mod is treated as not running. It publishes every
#: 5 s, so this tolerates a few missed writes without flapping.
POLICY_AGENT_STALE_SECONDS = 30.0


def _listening(port: int, host: str = "127.0.0.1", timeout: float = 0.35) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def adk_surface() -> dict[str, Any]:
    """The ADK's own description of itself, for the console to display."""

    scenes = []
    for scene in (adk.OPEN_FLAT_CONTROL_SCENE, adk.FAIL_CLOSED_SCENE):
        described = adk.describe_scene(scene)
        scenes.append(
            {
                "name": described.name,
                "contract_sha256": described.contract_sha256[:16],
                "active_providers": list(described.active_providers),
                "expected_batch": described.expected_batch,
                "expected_loadout": described.expected_loadout,
            }
        )

    groups = {name: len(cols) for name, cols in GROUP_FEATURES.items()}
    heads = [{"name": n, "offset": o, "size": s} for n, (o, s) in HEAD_SPANS.items()]

    return {
        "version": getattr(adk, "__version__", "unknown"),
        "scenes": scenes,
        "provider_seams": list(adk.JAX_SCENE_PROVIDER_ARGUMENTS),
        "loadouts": list(adk.list_loadouts()),
        "observation": {
            "groups": groups,
            "group_count": len(groups),
            "named_columns": sum(groups.values()),
        },
        "action": {
            "heads": heads,
            "head_count": len(heads),
            "logits": sum(h["size"] for h in heads),
        },
        "combat_parameters": len(adk.COMBAT_PARAMETER_NAMES),
    }


def _policy_agent_candidates() -> list[Path]:
    override = os.environ.get("HYTALE_POLICY_AGENT_DIR")
    roots = [Path(override)] if override else []
    roots.extend(POLICY_AGENT_DIRECTORIES)
    return [root / POLICY_AGENT_METRICS for root in roots]


def policy_agent_status() -> dict[str, Any]:
    """Counters published by the in-server policy mod, if it is running.

    Reports three states rather than two, because they mean different things
    and the fix differs: *absent* (no mod deployed anywhere we look), *stale*
    (a file exists but the server is not writing it — it stopped, or it never
    armed), and *live*.

    A stale file is still returned in full. The last values before a server
    went down are usually what you want to look at, and hiding them behind the
    staleness flag would throw away the only evidence of what happened.
    """

    searched = _policy_agent_candidates()
    for path in searched:
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, ValueError):
            continue
        try:
            metrics = json.loads(raw)
        except json.JSONDecodeError as exc:
            # The mod writes atomically, so a parse error is a real corruption
            # rather than a torn read; say so instead of reporting "absent".
            return {
                "state": "unreadable",
                "path": str(path),
                "error": f"{type(exc).__name__}: {exc}",
                "searched": [str(entry) for entry in searched],
            }

        published = float(metrics.get("timestamp_millis", 0)) / 1000.0
        age = max(0.0, time.time() - published) if published else None
        stale = age is None or age > POLICY_AGENT_STALE_SECONDS
        return {
            "state": "stale" if stale else "live",
            "path": str(path),
            "age_seconds": None if age is None else round(age, 1),
            "stale_after_seconds": POLICY_AGENT_STALE_SECONDS,
            "metrics": metrics,
        }

    return {
        "state": "absent",
        "note": (
            "No policy-agent metrics found. The mod publishes "
            f"{POLICY_AGENT_METRICS} next to its weights every 5 s; set "
            "HYTALE_POLICY_AGENT_DIR to point at another instance."
        ),
        "searched": [str(entry) for entry in searched],
    }


def bridge_status() -> dict[str, Any]:
    """Read-only view of the native bridge. Never opens a session.

    Opening a ``NativeBridgeSession`` takes the native evidence lease, which
    would evict whatever lane currently holds it. The console is a viewer, so
    it reports reachability only and leaves acquisition to the test suite.
    """

    listeners = [
        {"port": port, "role": role, "listening": _listening(port)}
        for port, role in BRIDGE_PORTS.items()
    ]
    return {
        "listeners": listeners,
        "any_up": any(entry["listening"] for entry in listeners),
        "note": (
            "Read-only. The console does not open a NativeBridgeSession, "
            "because doing so takes the native evidence lease from whoever "
            "holds it. Run adk/tests/test_bridge_fidelity.py for a real "
            "differential."
        ),
    }
