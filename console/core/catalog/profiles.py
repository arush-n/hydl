"""Agent profiles — what an agent *is*, not what test it happens to run.

The first version of this file got the axis wrong: a profile held a loadout, a
world, a zone and a tick count. That is a **test** configuration, and it made
the picker mean "set the form to these values" rather than "work with this
agent". The two change independently — one agent gets run against a dozen
scenes, and one scene is used to compare a dozen agents — so binding them
together makes both harder to vary.

A profile is a named agent configuration::

    agents/profiles/combat-64x128.json
    {
      "title": "Combat PPO 64/128",
      "agent": "combat",
      "architecture": "ppo",
      "checkpoint": "artifacts/console/…/checkpoint",
      "network": {"encoder": 64, "recurrent": 128},
      "contract": {"observation": 8271, "actions": 99},
      "hyperparameters": {"learning_rate": 0.0003, "clip": 0.2, "gamma": 0.99},
      "notes": "The shape that ships to Java."
    }

`agent` points at a directory under `agents/`, so a profile is a *variant* of
an agent — a checkpoint, a network width, a hyperparameter set — and several
profiles can share one agent.

`suggested_run` is allowed but deliberately secondary: it is a starting point
for the Run form, not the profile's purpose. A profile with no `suggested_run`
is complete.

Nothing is validated against a list of known architectures. A profile for a
Dreamer or a transformer is worth recording before the console can run one; the
alternative is a hardcoded list that has to be edited whenever someone adds an
architecture.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]

#: Where profiles live. Under `agents/` rather than `console/` because a
#: profile describes an agent, and outlives whatever is looking at it.
PROFILE_DIR = _ROOT / "agents" / "profiles"

#: Fields describing the agent itself.  Selection role and disposition are
#: deliberately separate: a candidate can be rejected, while the incumbent
#: may merely be the best available diagnostic checkpoint and not promoted.
AGENT_KEYS = (
    "agent",
    "architecture",
    "checkpoint",
    "checkpoint_sha256",
    "checkpoint_role",
    "selection_status",
    "selection_evidence",
    "network",
    "contract",
    "hyperparameters",
    "notes",
    "replays",
)

CHECKPOINT_ROLES = frozenset(("incumbent", "candidate"))
SELECTION_STATUSES = frozenset(("not_assessed", "promoted", "rejected"))


def _sha256(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("checkpoint_sha256 must be a SHA-256 string")
    canonical = value.upper()
    if len(canonical) != 64 or any(
        character not in "0123456789ABCDEF" for character in canonical
    ):
        raise ValueError("checkpoint_sha256 must be a SHA-256 string")
    return canonical


def _replays(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("profile replays must be a list")
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for replay in value:
        if not isinstance(replay, dict):
            raise ValueError("each profile replay must be an object")
        scenario = replay.get("scenario")
        run_id = replay.get("run_id")
        if not isinstance(scenario, str) or not scenario.strip():
            raise ValueError("profile replay scenario must be nonempty")
        if (
            not isinstance(run_id, str)
            or not run_id
            or not run_id.replace("-", "").replace("_", "").isalnum()
        ):
            raise ValueError("profile replay run_id is invalid")
        if scenario in seen:
            raise ValueError(f"duplicate profile replay scenario {scenario!r}")
        seen.add(scenario)
        label = replay.get("label") or scenario
        if not isinstance(label, str) or not label.strip():
            raise ValueError("profile replay label must be nonempty")
        recorded_at = replay.get("recorded_at")
        if not isinstance(recorded_at, str) or not recorded_at.strip():
            raise ValueError("profile replay recorded_at must be an ISO timestamp")
        try:
            parsed = datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                "profile replay recorded_at must be an ISO timestamp"
            ) from exc
        if parsed.tzinfo is None:
            raise ValueError("profile replay recorded_at must include a timezone")
        out.append({
            "scenario": scenario,
            "label": label,
            "run_id": run_id,
            "recorded_at": recorded_at,
        })
    return out


def _selection(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate optional checkpoint selection metadata as one atomic claim."""

    role = raw.get("checkpoint_role")
    status = raw.get("selection_status")
    checkpoint_sha = raw.get("checkpoint_sha256")
    classified = any(value is not None for value in (role, status, checkpoint_sha))
    if not classified:
        if raw.get("replays"):
            raise ValueError("checkpoint replays require classified checkpoint metadata")
        return {"replays": []}
    if role not in CHECKPOINT_ROLES:
        raise ValueError(
            "checkpoint_role must be one of: " + ", ".join(sorted(CHECKPOINT_ROLES))
        )
    if status not in SELECTION_STATUSES:
        raise ValueError(
            "selection_status must be one of: "
            + ", ".join(sorted(SELECTION_STATUSES))
        )
    if role == "incumbent" and status == "rejected":
        raise ValueError("a rejected checkpoint cannot be presented as incumbent")
    title_tokens = str(raw.get("title") or "").lower()
    for marker in "[]_-|/":
        title_tokens = title_tokens.replace(marker, " ")
    if status == "rejected" and "promoted" in title_tokens.split():
        raise ValueError("a rejected checkpoint title cannot claim promoted status")
    if not raw.get("checkpoint"):
        raise ValueError("classified checkpoint metadata requires checkpoint path")
    evidence = raw.get("selection_evidence") or ""
    if not isinstance(evidence, str):
        raise ValueError("selection_evidence must be a path or receipt string")
    return {
        "checkpoint_role": role,
        "selection_status": status,
        "checkpoint_sha256": _sha256(checkpoint_sha),
        "selection_evidence": evidence,
        "replays": _replays(raw.get("replays")),
    }


def _display_title(title: str, role: str | None, status: str | None) -> str:
    if role is None:
        return title
    disposition = {
        "not_assessed": "NOT PROMOTED",
        "promoted": "PROMOTED",
        "rejected": "REJECTED",
    }[status or "not_assessed"]
    return f"[{role.upper()} | {disposition}] {title}"

#: Keys a `suggested_run` may set. Anything else is ignored rather than
#: rejected — a profile written for a knob this console does not have yet is
#: still a valid record.
RUN_KEYS = (
    "loadout", "opponent", "policy", "world", "zone", "ticks", "seed",
    "slot", "period", "decision_period", "armed", "perceive", "target_active",
    "agent_max_health", "target_max_health", "task", "opponent_policy",
)


def index() -> list[dict[str, Any]]:
    """Every profile on disk, newest first. Rescanned per call."""

    if not PROFILE_DIR.is_dir():
        return []

    out: list[dict[str, Any]] = []
    for path in sorted(PROFILE_DIR.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # Surfaced, not skipped: a file that silently never appears is the
            # one failure its author cannot debug.
            out.append({
                "id": path.stem, "title": path.stem, "error": str(exc),
                "path": str(path), "architecture": "unknown", "agent": "",
            })
            continue
        if not isinstance(raw, dict):
            out.append({"id": path.stem, "title": path.stem, "agent": "",
                        "error": "profile must be a JSON object",
                        "path": str(path), "architecture": "unknown"})
            continue

        try:
            selection = _selection(raw)
            suggested_raw = raw.get("suggested_run") or {}
            if not isinstance(suggested_raw, dict):
                raise ValueError("suggested_run must be an object")
            suggested = {k: v for k, v in suggested_raw.items() if k in RUN_KEYS}
        except ValueError as exc:
            out.append({
                "id": path.stem,
                "title": path.stem,
                "agent": str(raw.get("agent") or ""),
                "error": str(exc),
                "path": str(path),
                "architecture": str(raw.get("architecture") or "unknown"),
            })
            continue
        title = str(raw.get("title") or path.stem)
        out.append({
            "id": path.stem,
            "title": title,
            "display_title": _display_title(
                title,
                selection.get("checkpoint_role"),
                selection.get("selection_status"),
            ),
            "agent": str(raw.get("agent") or ""),
            "architecture": str(raw.get("architecture") or "unspecified"),
            "checkpoint": raw.get("checkpoint") or "",
            "checkpoint_sha256": selection.get("checkpoint_sha256", ""),
            "checkpoint_role": selection.get("checkpoint_role"),
            "selection_status": selection.get("selection_status"),
            "selection_evidence": selection.get("selection_evidence", ""),
            "replays": selection["replays"],
            "network": raw.get("network") or {},
            "contract": raw.get("contract") or {},
            "hyperparameters": raw.get("hyperparameters") or {},
            "notes": raw.get("notes") or "",
            "suggested_run": suggested,
            "path": str(path),
            "updated_at": path.stat().st_mtime,
        })
    out.sort(key=lambda p: p.get("updated_at", 0), reverse=True)
    return out


def get(profile_id: str) -> dict[str, Any] | None:
    return next((p for p in index() if p["id"] == profile_id), None)


def replay_owners() -> dict[str, list[str]]:
    """Stored run id to profiles that explicitly bind it as a replay."""

    owners: dict[str, list[str]] = {}
    for profile in index():
        if profile.get("error"):
            continue
        for replay in profile.get("replays", []):
            owners.setdefault(replay["run_id"], []).append(profile["id"])
    return owners


def save(profile_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Write a profile. Overwrites by id, which is the file name."""

    if not profile_id or not profile_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError(
            f"invalid profile id {profile_id!r}: letters, digits, dash and "
            "underscore only — it becomes a file name")
    if not isinstance(payload, dict):
        raise ValueError("profile payload must be an object")
    selection = _selection(payload)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = {"title": payload.get("title") or profile_id}
    for key in AGENT_KEYS:
        if key in selection:
            if selection[key] not in (None, "", []):
                body[key] = selection[key]
        elif payload.get(key):
            body[key] = payload[key]
    body.setdefault("architecture", "unspecified")
    suggested = {k: v for k, v in (payload.get("suggested_run") or {}).items()
                 if k in RUN_KEYS}
    if suggested:
        body["suggested_run"] = suggested
    (PROFILE_DIR / f"{profile_id}.json").write_text(
        json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return get(profile_id) or body


def bind_replay(
    profile_id: str,
    *,
    scenario: str,
    run_id: str,
    recorded_at: str,
    label: str | None = None,
) -> dict[str, Any]:
    """Attach an archived trajectory without changing selection status."""

    path = PROFILE_DIR / f"{profile_id}.json"
    if not path.is_file():
        raise ValueError(f"unknown profile {profile_id!r}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"profile {profile_id!r} is unreadable") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"profile {profile_id!r} must be an object")
    selection = _selection(raw)
    replacement = {
        "scenario": scenario,
        "label": label or scenario,
        "run_id": run_id,
        "recorded_at": recorded_at,
    }
    raw["replays"] = [
        replay for replay in selection["replays"] if replay["scenario"] != scenario
    ] + [replacement]
    return save(profile_id, raw)


def bind_run(
    profile_id: str,
    artifact: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, str]:
    """Bind a stored artifact as the profile's latest replay for its scenario."""

    run_id = artifact.get("run_id")
    recorded_at = artifact.get("written_at")
    if not isinstance(run_id, str) or not isinstance(recorded_at, str):
        raise ValueError("stored replay requires run_id and written_at")
    scenario = str(
        spec.get("minigame")
        or spec.get("training_stage")
        or spec.get("task")
        or "rollout"
    )
    world = str(spec.get("world") or "unknown world").replace("_", " ")
    label = f"{scenario.replace('_', ' ').title()} · {world} · seed {spec.get('seed', '?')}"
    saved = bind_replay(
        profile_id,
        scenario=scenario,
        label=label,
        run_id=run_id,
        recorded_at=recorded_at,
    )
    return next(replay for replay in saved["replays"] if replay["run_id"] == run_id)
