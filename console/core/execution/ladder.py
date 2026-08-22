"""The objective ladder, driven: one rung at a time, advanced by its own gate.

`arena.curriculum.pursuit` declares the rungs and the promotion rule, and
`arena.curriculum.ladder.Ladder` owns the promote/hold/demote decision. Nothing
called it. The console rendered the thresholds and queued all four rungs at
once, so a rung that missed its bar was followed by the next one anyway and
`promote_at` was decoration.

QUEUEING THE WHOLE LADDER UP FRONT IS WHAT MAKES GATING IMPOSSIBLE. A gate has
three outcomes and two of them are not "the next rung": `hold` repeats the rung
it is on, `demote` steps back down. Which run comes after this one therefore
cannot be known until this one has been graded. This module keeps exactly ONE
rung outstanding and decides the next when it finishes.

The success rate is the run's OWN number -- `selection.selected.success_rate`
from `report.json`, which `pursuit_run` already inverts for an evade objective
so it means "fraction of episodes that went the learner's way" on either side of
the swap. Recomputing it here from catches and episodes would be a second
definition of success that could drift from the one the run selected on.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from console.core import storage

STATE_PATH = storage.write_root("ladder") / "objective-ladder.json"

#: Terminal statuses that produced a graded policy. A cancelled or faulted run
#: is NOT evidence about the rung -- recording it as a failure would demote a
#: policy for a device error, and treating it as a pass is worse.
_GRADED_PREFIX = "training_complete"

_lock = threading.Lock()


def _curriculum() -> dict[str, Any]:
    from arena.curriculum import pursuit

    return pursuit.describe()


def _rungs() -> list[dict[str, Any]]:
    return list(_curriculum()["objectives"]["stages"])


def _blank() -> dict[str, Any]:
    return {
        "schema": "console-objective-ladder-v1",
        "running": False,
        "index": 0,
        "history": [],
        "base_spec": {},
        "updates": 64,
        "awaiting_job": None,
        "transitions": [],
    }


def _read() -> dict[str, Any]:
    if not STATE_PATH.is_file():
        return _blank()
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _blank()
    return {**_blank(), **state} if isinstance(state, dict) else _blank()


def _write(state: Mapping[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def spec_for(index: int, base: Mapping[str, Any], updates: int) -> dict[str, Any]:
    """A launchable spec for one rung.

    The rung contributes its reward settings and its horizon; everything else
    -- world, loadout, compute -- comes from `base`, which is whatever the Train
    form resolved. `objective` and `target_controller` come from the curriculum
    rather than from the rung name: every rung here trains the EVADER, and
    `pursuit_run` refuses `objective="evade"` unless the opponent pursues.
    """

    payload = _curriculum()["objectives"]
    rungs = list(payload["stages"])
    rung = rungs[index]
    horizon = (payload.get("horizon_by_stage") or {}).get(rung["name"])
    spec: dict[str, Any] = {
        **dict(base),
        **dict(rung["settings"]),
        "updates": int(updates),
        "objective": payload.get("objective", "evade"),
        "target_controller": payload.get(
            "target_controller", "frozen_source_policy"
        ),
    }
    if horizon:
        # Refused when they disagree, so the rung's horizon sets both or
        # neither.
        spec["episode_ticks"] = int(horizon)
        spec["evaluation_steps"] = int(horizon)
    return spec


def status() -> dict[str, Any]:
    """Where the ladder is, for the Train tab."""

    state = _read()
    rungs = _rungs()
    index = int(state["index"])
    gate = _curriculum()["objectives"]["gate"]
    window = state["history"][-int(gate["window"]) :]
    return {
        "schema": state["schema"],
        "running": bool(state["running"]),
        "finished": index >= len(rungs),
        "index": index,
        "current": rungs[index]["name"] if index < len(rungs) else None,
        "stages": [rung["name"] for rung in rungs],
        "evaluations_at_stage": len(state["history"]),
        "history": list(state["history"]),
        "recent_success_rate": (sum(window) / len(window)) if window else None,
        "gate": dict(gate),
        "awaiting_job": state["awaiting_job"],
        "transitions": list(state["transitions"])[-12:],
    }


def start(base_spec: Mapping[str, Any], updates: int, *, restart: bool = False):
    """Begin (or resume) the ladder and launch its current rung.

    Returns whatever `enqueue` returned, so the caller can report the run.
    """

    from console.core.execution import jobs

    with _lock:
        state = _read()
        if state["running"] and state["awaiting_job"]:
            raise RuntimeError(
                f"the ladder is already waiting on run {state['awaiting_job']}; "
                "stop it before starting another"
            )
        if restart:
            state = _blank()
        index = int(state["index"])
        if index >= len(_rungs()):
            raise RuntimeError("the ladder has finished every rung; restart it")
        state["running"] = True
        state["base_spec"] = dict(base_spec)
        state["updates"] = int(updates)
        _write(state)
        spec = spec_for(index, state["base_spec"], state["updates"])

    result = jobs.enqueue(spec, rung=_rungs()[index]["name"])
    started = (result or {}).get("started") or {}
    with _lock:
        state = _read()
        state["awaiting_job"] = started.get("job_id")
        state["queued_id"] = (result or {}).get("queued", {}).get("queued_id")
        _write(state)
    return result


def stop() -> dict[str, Any]:
    """Stop advancing. Does not touch a run already on the device."""

    with _lock:
        state = _read()
        state["running"] = False
        state["awaiting_job"] = None
        _write(state)
    return status()


def note_started(job_id: str, rung: str) -> None:
    """Bind a launched job to the rung it is running.

    Needed because a rung can sit in the queue behind another run, in which
    case `start` never saw a job id.
    """

    with _lock:
        state = _read()
        if state["running"] and not state["awaiting_job"]:
            current = _rungs()[int(state["index"])]["name"]
            if current == rung:
                state["awaiting_job"] = job_id
                _write(state)


def note_finished(job: Any) -> None:
    """Grade a finished rung and launch whatever the gate says comes next.

    Called from the job completion path. Silent and harmless for any run the
    ladder did not launch.
    """

    from arena.curriculum.ladder import Ladder, SettingsStage
    from console.core.execution import jobs

    job_id = getattr(job, "job_id", None)
    with _lock:
        state = _read()
        if not state["running"] or state["awaiting_job"] != job_id:
            return
        status_text = str(getattr(job, "status", "") or "")
        report_status = str((getattr(job, "latest", {}) or {}).get("status") or "")
        rate = _success_rate(job)
        if rate is None:
            # Cancelled, faulted, or finished without a graded selection. Stop
            # rather than guess: a missing number is not a zero, and recording
            # one would demote the policy for a device failure.
            state["running"] = False
            state["awaiting_job"] = None
            state["transitions"].append({
                "unix_seconds": time.time(),
                "job_id": job_id,
                "rung": _rungs()[int(state["index"])]["name"],
                "success_rate": None,
                "transition": "stopped",
                "reason": (
                    f"no graded selection (job status {status_text!r}"
                    f"{', report ' + report_status if report_status else ''})"
                ),
            })
            _write(state)
            return

        rungs = _rungs()
        gate = _curriculum()["objectives"]["gate"]
        ladder = Ladder([
            SettingsStage(rung["name"], rung["settings"], **gate)
            for rung in rungs
        ])
        ladder.state.index = int(state["index"])
        ladder.state.history = [float(value) for value in state["history"]]
        transition = ladder.record(float(rate))

        state["index"] = ladder.state.index
        state["history"] = list(ladder.state.history)
        state["awaiting_job"] = None
        state["transitions"].append({
            "unix_seconds": time.time(),
            "job_id": job_id,
            "rung": rungs[min(int(state["index"]), len(rungs) - 1)]["name"],
            "success_rate": float(rate),
            "transition": transition,
        })
        if ladder.finished:
            state["running"] = False
            _write(state)
            return
        _write(state)
        next_index = int(state["index"])
        base = dict(state["base_spec"])
        updates = int(state["updates"])

    # Chain from the checkpoint this rung just produced. `latest` resolves at
    # dequeue, which is the only moment the previous rung's output exists.
    spec = spec_for(next_index, base, updates)
    spec["source_policy"] = "latest"
    try:
        result = jobs.enqueue(spec, rung=_rungs()[next_index]["name"])
    except Exception as error:  # noqa: BLE001 - a refused rung stops the ladder
        with _lock:
            state = _read()
            state["running"] = False
            state["transitions"].append({
                "unix_seconds": time.time(),
                "rung": _rungs()[next_index]["name"],
                "transition": "stopped",
                "reason": f"{type(error).__name__}: {error}",
            })
            _write(state)
        return
    started = (result or {}).get("started") or {}
    with _lock:
        state = _read()
        state["awaiting_job"] = started.get("job_id")
        _write(state)


def _success_rate(job: Any) -> float | None:
    """The run's own selected success rate, or None if it produced none."""

    if not str(getattr(job, "status", "") or "").startswith("finished"):
        # `finished` is the console's word for a run that completed and wrote a
        # report; `cancelled`, `stopped` and `failed` all mean the rung was not
        # graded.
        return None
    directory = getattr(job, "directory", None)
    if not directory:
        return None
    report_path = Path(directory) / "stage" / "report.json"
    if not report_path.is_file():
        report_path = Path(directory) / "report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not str(report.get("status") or "").startswith(_GRADED_PREFIX):
        return None
    selected = ((report.get("selection") or {}).get("selected")) or {}
    rate = selected.get("success_rate")
    if rate is None or not isinstance(rate, (int, float)):
        return None
    return max(0.0, min(1.0, float(rate)))


__all__ = ["STATE_PATH", "note_finished", "note_started", "spec_for", "start",
           "status", "stop"]
