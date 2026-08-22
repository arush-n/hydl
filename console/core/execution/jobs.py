"""Long-running training jobs, launched from the console and survivable.

A rollout finishes inside an HTTP request; a training run does not. This adds
the smallest thing that lets one be started, watched and read back:

* **One worker, serialised.** JAX compiles per configuration and holds device
  memory for the duration. Two training runs on one machine thrash rather than
  overlap -- a neighbouring JAX suite eating RAM has already produced a bare
  exit 255 here once -- so a second launch is refused rather than queued behind
  a job whose memory it would be competing with.
* **Metrics appended to disk as they arrive**, one JSON object per update. A run
  that dies at update 400 must leave 399 updates of evidence behind, not an
  empty file and a stack trace.
* **Identity read either side of the whole run.** This matters far more for
  training than for a rollout: the Gym pin moved *mid-session* on 2026-08-05
  (BA530547 -> D954BD79). A multi-hour run spanning that is void, and nothing
  else in the system would say so.

The job registry is deliberately in-memory. It describes what this process is
doing now; the durable record is the artifact directory, which is what a reader
comes back to tomorrow.
"""

from __future__ import annotations

import functools
import json
from math import isfinite
import hashlib
import shlex
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from console.core.catalog import profiles
from console.core import storage
from console.core.evidence import store
from console.core.telemetry import live
from console.core.execution import ladder
from console.core.execution.gpu_worker import basic_worker

METRICS = "metrics.jsonl"
CHECKPOINT = "checkpoint"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PURSUIT_POLICY = PROJECT_ROOT / "agents/basic/checkpoints/basic_policy.npz"
BASIC_ARTIFACT_ROOT = PROJECT_ROOT / "agents/basic/artifacts"

#: `source_policy` value meaning "train from scratch".
#:
#: There was no way to express this before. `pursuit_run` calls
#: `load_policy_checkpoint` unconditionally and then builds the network with
#: `replace(source_config, ...)`, which overrides batch and optimiser settings but
#: *not* `observation_size`, `action_size` or `encoder_size` -- so the checkpoint's
#: own `config_json` silently defines the network, and its `recurrent_size` beats
#: the preset's. Omitting `source_policy` fell through to `DEFAULT_PURSUIT_POLICY`
#: and quietly trained against whatever shape that file happened to be.
#:
#: So "from scratch" is served by *minting* a randomly-initialised checkpoint at
#: the live contract's dimensions rather than by passing nothing. Same result, and
#: it keeps the one place that defines the network honest.
SCRATCH_SOURCE_POLICY = "scratch"

#: "Start from whatever is newest WHEN THIS RUN ACTUALLY STARTS."
#:
#: This is what makes a ladder queueable. A rung must begin from the rung below
#: it, but a queued spec is written before the rung below has produced anything
#: -- its artifact directory is named after a run id that does not exist yet, so
#: no literal path can be written down at enqueue time.
#:
#: `drain_queue` already resolves specs late ("A queued spec is only validated
#: when it is finally launched"), so a sentinel resolved in the same place picks
#: up the previous rung's output without the caller predicting its name. Queue
#: rung 1 from an explicit checkpoint and every rung above it from "latest".
LATEST_SOURCE_POLICY = "latest"

#: Minted scratch checkpoints, keyed by the dimensions they encode. Kept out of
#: `agents/basic/checkpoints/` so a mint can never shadow a hand-pinned policy.
#:
#: Resolved through `storage` so it follows the `policies` area rather than
#: pinning the legacy path independently. Unchanged today: `write_root` returns
#: the legacy root until that area is migrated.
INIT_POLICY_ROOT = storage.write_root("policies")

#: Environments a training job runs in parallel, when the launch does not say.
#:
#: The console's own `BATCH` is 2 because it *draws* one environment. Training
#: inherited that silently and it is the wrong number: 2 envs x 256 steps is 512
#: steps per update, and a measured 12-update run at that width moved the policy
#: by a KL of 1e-5 with reward indistinguishable from noise.
#:
#: **This deliberately disagrees with the canonical trainer.**
#: `HytaleRL/examples/jax_arsenal_train.py` defaults to `--num-envs 128`, and
#: `PPOConfig.num_envs` is 256. Both assume better hardware than this backend,
#: which is CPU -- measured here, 32 envs x 256 steps took ~15 s per update.
#: 32 is chosen so the console stays interactive, not because it is the right
#: width for a real run.
#:
#: So: a console run and a `jax_arsenal_train.py` run are **not** comparable by
#: default, because their batch widths differ. For a serious run, either pass
#: `num_envs` explicitly to match, or use the canonical trainer -- it also has
#: the pre-training reward-liveness gate this path does not.
#:
#: **Lowered 32 -> 16 on 2026-08-08, because 32 now crashes.** The reasoning
#: above is a CPU rationale and CPU is no longer used. Measured on WSL CUDA,
#: one process per width, no geometry bound:
#:
#:     envs=16  exit 0    survived
#:     envs=32  exit 134  double free or corruption (fasttop)
#:     envs=64  exit 134  double free or corruption (fasttop)
#:
#: So the old default sat exactly on a deterministic abort and every console
#: training run would have died at it. To revert: set this back to 32 (and
#: expect the crash) -- nothing else depends on the value.
#:
#: 16 is verified only for the **no-geometry** path (`open_flat`, and the
#: synthetic worlds, which bind `world_capability_provider` rather than
#: geometry). `world="region"` binds real geometry, and the geometry sweep
#: fails at far lower widths -- clean only at 1, 2, 5, 8 in a `frames.py`
#: measurement. A Region training run at 16 is **unverified**; measure before
#: trusting it.
TRAIN_ENVS = 16
#: Written beside the weights when a save was refused, so the reason outlives
#: the process that produced it.
REFUSED = "checkpoint-refused.txt"

#: Terminal states. Anything else means the worker still holds the device.
#: "stopped" is a run a training guardrail ended early. It has to be listed here
#: or the console keeps treating the device as busy and refuses the next launch.
DONE = ("finished", "failed", "cancelled", "stopped")


@dataclass
class Job:
    """One training run's live state. The artifact on disk is the record."""

    job_id: str
    spec: dict[str, Any]
    profile_id: str | None = None
    #: The objective-ladder rung this run is grading, when the ladder driver
    #: launched it. None for every hand-launched run, which is what keeps the
    #: gate from recording a manual run as evidence about a rung.
    rung: str | None = None
    status: str = "starting"
    update: int = 0
    updates: int = 0
    started: float = field(default_factory=time.time)
    finished: float | None = None
    error: str | None = None
    directory: str | None = None
    #: Only the most recent metrics are held here; the full series is on disk.
    latest: dict[str, Any] = field(default_factory=dict)
    #: Path to the saved weights, or the reason there are none.
    checkpoint: str | None = None
    #: Why a guardrail ended the run early. None on every other terminal state.
    #: Kept separate from `error`, which means the run itself faulted.
    stop_reason: str | None = None
    profile_error: str | None = None
    cache: dict[str, Any] = field(default_factory=dict)
    #: Newest fixed-seed evaluation replay written by the running stage.
    latest_replay: dict[str, Any] | None = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    #: Set by `terminate`. Distinct from `_cancel`, which asks the run to stop
    #: at the next update boundary and lets it write its checkpoint. This one
    #: means the process was killed underneath it, so the worker fault that
    #: follows is expected rather than a training failure to report.
    _terminated: threading.Event = field(
        default_factory=threading.Event, repr=False
    )

    def view(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "profile_id": self.profile_id,
            "status": self.status,
            "update": self.update,
            "updates": self.updates,
            "elapsed_seconds": round((self.finished or time.time()) - self.started, 1),
            "started_unix_seconds": self.started,
            "finished_unix_seconds": self.finished,
            "error": self.error,
            "directory": self.directory,
            "latest": self.latest,
            "checkpoint": self.checkpoint,
            "stop_reason": self.stop_reason,
            "profile_error": self.profile_error,
            "cache": self.cache,
            "latest_replay": (
                {
                    key: value
                    for key, value in self.latest_replay.items()
                    if key != "path"
                }
                if self.latest_replay
                else None
            ),
            "spec": self.spec,
        }


_lock = threading.Lock()
_jobs: dict[str, Job] = {}
_active: str | None = None


@dataclass
class QueuedRun:
    """A launch spec waiting for the device, not yet a Job.

    Deliberately NOT resolved at enqueue time. `_resolve_launch_spec` reads the
    live contracts, mints scratch policies and computes cache keys, so resolving
    an entry that runs in two hours against the contracts of two hours ago would
    stamp it with an identity it never trained under.
    """

    queued_id: str
    spec: dict[str, Any]
    profile_id: str | None
    queued_at: float
    label: str
    rung: str | None = None

    def view(self, position: int) -> dict[str, Any]:
        return {
            "queued_id": self.queued_id,
            "position": position,
            "queued_at": self.queued_at,
            "label": self.label,
            "profile_id": self.profile_id,
            "rung": self.rung,
            "spec": self.spec,
        }


_queue: list[QueuedRun] = []


def _queue_label(spec: Mapping[str, Any]) -> str:
    """Something a human can tell two queued runs apart by."""

    parts = [str(spec.get("objective") or spec.get("training_stage") or "run")]
    if spec.get("updates"):
        parts.append(f"{spec['updates']}u")
    if spec.get("world_design"):
        parts.append(str(spec["world_design"]))
    elif spec.get("world_count"):
        parts.append(f"{spec['world_count']}x{spec.get('world_order') or 'digest'}")
    return " · ".join(parts)


def enqueue(
    spec: dict[str, Any], *, profile_id: str | None = None,
    rung: str | None = None,
) -> dict[str, Any]:
    """Put a run behind whatever holds the device. Starts it if nothing does."""

    entry = QueuedRun(
        queued_id=uuid.uuid4().hex[:12],
        spec=dict(spec),
        profile_id=profile_id,
        queued_at=time.time(),
        label=_queue_label(spec),
        rung=rung,
    )
    with _lock:
        _queue.append(entry)
        position = len(_queue)
    live.publish("queue_changed", {"queued": queue_index()})
    started = drain_queue()
    return {
        "queued": entry.view(position),
        "started": started,
        "depth": len(_queue),
    }


def queue_index() -> list[dict[str, Any]]:
    with _lock:
        entries = list(_queue)
    return [entry.view(index + 1) for index, entry in enumerate(entries)]


def dequeue(queued_id: str) -> bool:
    """Drop one waiting entry. A run already started is `cancel`'s business."""

    with _lock:
        before = len(_queue)
        _queue[:] = [item for item in _queue if item.queued_id != queued_id]
        removed = len(_queue) != before
    if removed:
        live.publish("queue_changed", {"queued": queue_index()})
    return removed


def clear_queue() -> int:
    with _lock:
        dropped = len(_queue)
        _queue.clear()
    if dropped:
        live.publish("queue_changed", {"queued": queue_index()})
    return dropped


def drain_queue() -> dict[str, Any] | None:
    """Start the next waiting run if the device is free. Returns what started.

    A queued spec is only validated when it is finally launched, so an entry can
    still be rejected here -- a checkpoint deleted while it waited, a contract
    that moved. Such an entry is DROPPED and the failure published, rather than
    left at the head of the queue where it would block every run behind it.
    """

    while True:
        with _lock:
            current = _jobs.get(_active) if _active else None
            if current is not None and current.status not in DONE:
                return None
            if not _queue:
                return None
            entry = _queue.pop(0)
        try:
            view = launch(
                entry.spec, profile_id=entry.profile_id, rung=entry.rung
            )
        except Exception as error:  # noqa: BLE001 - report and keep draining
            live.publish(
                "queue_rejected",
                {
                    "queued_id": entry.queued_id,
                    "label": entry.label,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            live.publish("queue_changed", {"queued": queue_index()})
            continue
        live.publish("queue_changed", {"queued": queue_index()})
        return view


def _drain_after_finish(job: "Job | None" = None) -> None:
    """Grade the ladder, then start the next queued run -- off this thread.

    `drain_queue` calls `launch`, which starts a worker thread; doing that from
    inside the outgoing job's `finally` would tie the new run's lifetime to the
    old thread's stack. A short daemon thread keeps them independent. The ladder
    is graded on that same thread and for the same reason: a promotion enqueues
    the next rung, which is another launch.
    """

    def _finish() -> None:
        if job is not None:
            try:
                ladder.note_finished(job)
            except Exception:  # noqa: BLE001 - a ladder fault must not wedge the queue
                pass
        drain_queue()

    if job is None and not _queue:
        return
    threading.Thread(target=_finish, daemon=True, name="train-queue-drain").start()


def active() -> Job | None:
    with _lock:
        job = _jobs.get(_active) if _active else None
        return None if job is None or job.status in DONE else job


def index() -> list[dict[str, Any]]:
    with _lock:
        jobs = sorted(_jobs.values(), key=lambda job: job.started, reverse=True)
    return [job.view() for job in jobs]


def get(job_id: str) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        return None
    view = job.view()
    payload = replay(job_id)
    if payload is not None:
        view["latest_replay_payload"] = payload
    return view


def cache_index() -> dict[str, Any]:
    """Durable compile/result identities with measured launch timings."""

    groups: dict[str, dict[str, Any]] = {}
    for row in _cached_runs():
        group = groups.setdefault(
            row["cache_key"],
            {
                "key": row["cache_key"],
                "runs": 0,
                "successful_runs": 0,
                "executable_candidate_runs": 0,
                "last_used": None,
                "latest": None,
            },
        )
        group["runs"] += 1
        # `"finished"` is the JOB status. These rows carry the RUN REPORT
        # status, which is always one of `training_complete_*`, `cancelled_*`,
        # `stopped_by_guardrail_*` or `failed` -- so the old equality could
        # never be true and this counter read 0 across 226 runs, making a
        # perfectly healthy cache look dead. Reuse never consulted it (that is
        # `executable_candidate_runs`), so the damage was diagnostic only.
        group["successful_runs"] += str(
            row["status"] or ""
        ).startswith("training_complete")
        group["executable_candidate_runs"] += row["program_reached"]
        if (row.get("written_at") or "") > (group["last_used"] or ""):
            group["last_used"] = row.get("written_at")
            group["latest"] = row
    entries = sorted(
        groups.values(), key=lambda item: item.get("last_used") or "", reverse=True
    )
    return {
        "schema": "console-jax-launch-cache-v1",
        "entries": entries,
        "policy": "content-addressed candidate tracking; JAX validates executable reuse",
    }


def cancel(job_id: str) -> bool:
    """Ask a job to stop after its current update. Never kills mid-update."""

    with _lock:
        job = _jobs.get(job_id)
    if job is None or job.status in DONE:
        return False
    job._cancel.set()
    return True


def terminate(job_id: str) -> bool:
    """Kill the run's process now. Unlike `cancel`, nothing is saved.

    `cancel` is the polite stop: it sets a sentinel the stage polls, the run
    finishes its update and writes a checkpoint. That is the right default and
    it is useless when the stage is wedged before its first update -- a Region
    scene build takes minutes and polls nothing, so a bad launch could not be
    called off at all.

    This closes the resident WSL worker's stdin, then terminates, then kills.
    The training thread sees its worker die and unwinds through the normal
    failure path; `_terminated` is what tells it the death was requested.
    """

    with _lock:
        job = _jobs.get(job_id)
    if job is None or job.status in DONE:
        return False
    job._terminated.set()
    job._cancel.set()
    job.stop_reason = "terminated by request"
    job.status = "cancelling"
    live.publish("job_status", job.view())
    basic_worker.close()
    return True


def launch(
    spec: dict[str, Any], *, profile_id: str | None = None,
    rung: str | None = None,
) -> dict[str, Any]:
    """Start one training run, or explain why this is not the moment."""

    global _active
    resolved = _resolve_launch_spec(spec)
    if profile_id is not None:
        profile = profiles.get(profile_id)
        if profile is None or profile.get("error"):
            raise ValueError(f"training profile {profile_id!r} is unavailable")
        agent = profile.get("agent")
        if not isinstance(agent, str) or not agent:
            raise ValueError(f"training profile {profile_id!r} has no archive agent")
        resolved["shared_agent"] = agent
        resolved["profile_ids"] = list(dict.fromkeys((agent, profile_id)))
    elif resolved.get("training_stage") == "pursuit_tracking":
        # Basic owns generic pursuit training. A selected checkpoint profile may
        # additionally own the run, but an empty picker must not strand it in
        # an anonymous flat directory or misfile it under another agent.
        resolved.setdefault("shared_agent", store.BASIC_SHARED_AGENT)
    cache_key = _prepared_cache_key(resolved)
    result_key = _result_cache_key(resolved)
    resolved.update(
        prepared_cache_key=cache_key,
        result_cache_key=result_key,
    )
    history = _cache_history(cache_key, result_key)
    with _lock:
        current = _jobs.get(_active) if _active else None
        if current is not None and current.status not in DONE:
            raise RuntimeError(
                f"training job {current.job_id} is still {current.status}; "
                "one at a time, because concurrent JAX runs contend for device "
                "memory rather than sharing it"
            )
        prior = history["executable_candidate_runs"] > 0 or any(
            item.spec.get("prepared_cache_key") == cache_key
            for item in _jobs.values()
            if item.cache.get("program_build_seconds") is not None
        )
        job = Job(
            job_id=store.run_id(resolved),
            spec=resolved,
            profile_id=profile_id,
            rung=rung,
            updates=int(resolved.get("updates", 0)),
            cache={
                "key": cache_key,
                "result_key": result_key,
                "persistent_candidate_hit": prior,
                "history": history,
                "reuse_scope": (
                    "persistent JAX executable; Region scenes are process-local"
                ),
            },
        )
        _jobs[job.job_id] = job
        _active = job.job_id

    # Bound here rather than at enqueue: a rung that waited behind another
    # run had no job id when the ladder queued it, and the gate needs one to
    # know which completion is its own.
    if rung is not None:
        ladder.note_started(job.job_id, rung)

    threading.Thread(
        target=_work, args=(job,), daemon=True, name=f"train-{job.job_id}"
    ).start()
    live.publish("job_started", job.view())
    return job.view()


def _write_artifact(
    job: Job,
    result: dict[str, Any],
    opened: dict[str, Any],
) -> dict[str, Any]:
    """Store profile provenance and bind only artifacts that contain a replay."""

    spec = dict(job.spec)
    if job.profile_id:
        spec["profile_id"] = job.profile_id
    artifact = store.write(spec, result, opened, name=job.job_id)
    if job.profile_id and result.get("trajectory"):
        try:
            profiles.bind_run(job.profile_id, artifact, spec)
        except Exception as exc:  # noqa: BLE001 - training already completed
            job.profile_error = f"{type(exc).__name__}: {exc}"
    return artifact


def preflight(spec: dict[str, Any]) -> dict[str, Any]:
    """Resolve a launch without building JAX; every returned check is actionable."""

    resolved = _resolve_launch_spec(spec)
    key = _prepared_cache_key(resolved)
    result_key = _result_cache_key(resolved)
    batch = int(resolved["num_envs"])
    rollout = int(resolved["rollout_steps"])
    minibatches = int(resolved["num_minibatches"])
    memory = _gpu_memory_plan()
    checks = [
        {
            "name": "gpu_runner",
            "passed": resolved.get("runner") == "wsl",
            "detail": "WSL CUDA selected",
        },
        {
            "name": "minibatch_divisibility",
            "passed": (batch * rollout) % minibatches == 0,
            "detail": f"{batch}×{rollout} transitions / {minibatches} minibatches",
        },
        {
            "name": "bounded_episode_horizon",
            "passed": 1 <= int(resolved["evaluation_steps"]) <= 512,
            "detail": f"{resolved['evaluation_steps']} evaluation ticks",
        },
        {
            "name": "source_policy",
            "passed": Path(resolved["source_policy"]).is_file(),
            "detail": str(resolved["source_policy"]),
        },
    ]
    # Only meaningful once the file is known to exist, and it is the check that
    # was missing: the run builds its network from this checkpoint's config, so a
    # stale one is not a warning, it is a guaranteed shape error under load.
    if checks[-1]["passed"]:
        mismatch = _policy_contract_mismatch(Path(resolved["source_policy"]))
        checks.append(
            {
                "name": "policy_contract",
                "passed": not mismatch,
                "detail": mismatch or "checkpoint matches the live observation/action surface",
            }
        )
    failed = [item for item in checks if not item["passed"]]
    if failed:
        raise ValueError(
            "preflight failed: " + ", ".join(item["name"] for item in failed)
        )
    history = _cache_history(key, result_key)
    with _lock:
        candidate_hit = history["executable_candidate_runs"] > 0 or any(
            item.spec.get("prepared_cache_key") == key
            for item in _jobs.values()
            if item.cache.get("program_build_seconds") is not None
        )
    return {
        "schema": "console-training-preflight-v1",
        "passed": True,
        "resolved": resolved,
        "checks": checks,
        "cache": {
            "key": key,
            "result_key": result_key,
            "persistent_candidate_hit": candidate_hit,
            "history": history,
            "invalidation": "any contract, environment, model, or static-shape change",
        },
        "gpu_memory": memory,
    }


# XLA prints a multi-line constant-folding advisory on most JAX runs and it is
# almost always the LAST thing in stderr. Tailing stderr therefore reported an
# XLA problem for every failed run regardless of the real cause.
_STDERR_NOISE = (
    "if you'd like to file a bug",
    "this isn't necessarily a bug",
    "constant folding an instruction is taking",
    "the operation took",
    "run with --info or --debug",
    "run with --scan",
    "get more help at",
)


def _worker_failure_detail(
    response: Mapping[str, Any],
    stderr_tail: str,
) -> str:
    """Name why the worker stopped, preferring its own structured report.

    The worker sends ``error_type`` and ``message``; those say what actually
    happened. Only when both are absent do we fall back to stderr, and then to
    the last line that is not known XLA advisory noise -- picking the last line
    outright is what made a SourceDrift failure read as an XLA bug.
    """

    error_type = str(response.get("error_type") or "").strip()
    message = str(response.get("message") or "").strip()
    reported = ": ".join(part for part in (error_type, message) if part)
    if reported:
        return reported

    for line in reversed(stderr_tail.strip().splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        lowered = candidate.lower()
        if any(noise in lowered for noise in _STDERR_NOISE):
            continue
        return candidate
    return "no error output"


def _prepared_cache_key(spec: Mapping[str, Any]) -> str:
    """Content identity for fields that change compiled array/program shapes."""

    static = {
        name: spec.get(name)
        for name in (
            "launch_contract",
            "training_stage",
            "loadout",
            "opponent",
            "num_envs",
            "rollout_steps",
            "evaluation_steps",
            "update_epochs",
            "num_minibatches",
            "learning_rate",
            "gamma",
            "gae_lambda",
            "clip",
            "clip_epsilon",
            "value_coef",
            "value_coefficient",
            "entropy_coef",
            "entropy_coefficient",
            "max_gradient_norm",
            "stage_options",
            "target_options",
            "target_controller",
            "target_separation_range",
            "world_artifact_seed",
            # Both change compiled shapes: the Region atlas leads with a world
            # axis, so `world_count` resizes every traversal and geometry
            # array, and `environment_diversity` changes reset positions from
            # one broadcast row to a per-environment bank. Reusing a
            # single-world program for a multi-world request would be wrong.
            "world_count",
            "environment_diversity",
            # Not a shape, but it rescales the observed-distance channels and
            # gates target perception, so two runs at different ranges are not
            # comparable and must not share a prepared-program identity.
            "sensor_range",
            # DELIBERATELY ABSENT: `selection_key`, `node_seed`,
            # `mixed_terrain_worlds`, `flattest_worlds`, `world_order`.
            #
            # They choose WHICH worlds are resident, which is data, not program
            # structure -- the shapes are already pinned by `world_count` and
            # `environment_diversity` above. Including them made this key unique
            # per round by construction, because the self-play driver varies
            # `selection_key` every round to rotate the resident set, so the
            # persistent executable could never be reported as reusable and the
            # Train tab showed a cache miss on every single launch.
            #
            # Nothing is lost by dropping them: `_result_cache_key` hashes the
            # WHOLE spec, so two runs on different world sets remain distinct
            # deterministic requests. This key answers only "might the compiled
            # program be reusable", and JAX stays the authority -- it recompiles
            # if its own content identity differs.
        )
    }
    source = Path(str(spec.get("source_policy", "")))
    if source.is_file():
        static["source_policy_program"] = _checkpoint_program_identity(source)
    else:
        static.update(
            encoder_size=spec.get("encoder_size"),
            recurrent_size=spec.get("recurrent_size"),
        )
    payload = json.dumps(static, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest().upper()


@functools.lru_cache(maxsize=1)
def _live_policy_contract() -> dict[str, Any] | None:
    """The observation/action surface the Gym actually emits right now.

    Imported lazily: the preflight promise is "no JAX *build*", and reading two
    published constants does not compile anything. Returns ``None`` when the Gym
    cannot be imported at all, so a Console running without it degrades to the
    old behaviour instead of refusing every launch.
    """

    try:
        from hytalegym.jax.combat.contracts.publication import (
            ARSENAL_POLICY_ACTION_HEAD_SIZES,
            ARSENAL_POLICY_ACTION_SIZE,
            ARSENAL_POLICY_OBSERVATION_SIZE,
        )
    except Exception:  # noqa: BLE001 - a missing Gym must not block the console
        return None
    return {
        "observation_size": int(ARSENAL_POLICY_OBSERVATION_SIZE),
        "action_size": int(ARSENAL_POLICY_ACTION_SIZE),
        "action_head_sizes": [int(size) for size in ARSENAL_POLICY_ACTION_HEAD_SIZES],
    }


def _checkpoint_contract(path: Path) -> dict[str, Any] | None:
    """Read a checkpoint's declared surface without loading it onto a device."""

    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as values:
            config = json.loads(str(values["config_json"].item()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    return {
        "observation_size": int(config.get("observation_size", -1)),
        "action_size": int(config.get("action_size", -1)),
        "recurrent_size": int(config.get("recurrent_size", -1)),
        "encoder_size": int(config.get("encoder_size", -1)),
    }


def _policy_contract_mismatch(path: Path) -> str:
    """Empty when the checkpoint matches the live surface; else why it does not.

    `load_policy_checkpoint` validates a checkpoint against its *own* embedded
    config, so a stale file is perfectly self-consistent and passes. Nothing
    compared it to the contract the environment emits until this check: on
    2026-08-18 the pinned default was 8271/99 against a live 8272/105 and the
    only symptom would have been a shape error minutes into a GPU run.
    """

    live = _live_policy_contract()
    found = _checkpoint_contract(path)
    if live is None or found is None:
        return ""
    problems = [
        f"{name} {found[name]} != live {live[name]}"
        for name in ("observation_size", "action_size")
        if found[name] != live[name]
    ]
    if not problems:
        return ""
    return (
        ", ".join(problems)
        + f'; re-pin or launch with "source_policy": "{SCRATCH_SOURCE_POLICY}"'
    )


def _latest_trained_policy() -> Path:
    """Newest archived checkpoint that still speaks the live contract.

    Contract-filtered rather than simply newest, because the newest run is not
    always usable: a rung whose contract moved leaves behind weights that load
    cleanly and mean something else. Skipping those is the difference between a
    ladder that chains and one that silently restarts from a stale generation.
    """

    for row in store.policies(limit=60):
        updates = [int(update) for update in (row.get("updates") or [])]
        directory = PROJECT_ROOT / str(row.get("directory") or "")
        for update in sorted(updates, reverse=True):
            candidate = directory / f"update-{update:04d}.npz"
            if candidate.is_file() and not _policy_contract_mismatch(candidate):
                return candidate
    raise ValueError(
        'no archived checkpoint matches the live contract, so '
        f'"source_policy": "{LATEST_SOURCE_POLICY}" cannot resolve; launch with '
        f'"{SCRATCH_SOURCE_POLICY}" or name a checkpoint explicitly'
    )


def _mint_scratch_policy(encoder_size: int, recurrent_size: int) -> Path:
    """Write (or reuse) a random-init checkpoint at the live contract's shape."""

    live = _live_policy_contract()
    if live is None:
        raise ValueError(
            "cannot mint a scratch policy: the Gym contract is unavailable, so "
            "the observation and action sizes are unknown"
        )
    observation = live["observation_size"]
    action = live["action_size"]
    destination = (
        INIT_POLICY_ROOT
        / f"init-obs{observation}-act{action}-enc{encoder_size}-rec{recurrent_size}.npz"
    )
    if destination.is_file() and not _policy_contract_mismatch(destination):
        return destination

    import jax

    from hytalegym.jax.training.checkpoint import save_policy_checkpoint
    from hytalegym.jax.training.policy import initialize_policy
    from hytalegym.jax.training.types import PPOConfig

    config = PPOConfig(
        encoder_size=encoder_size,
        recurrent_size=recurrent_size,
        observation_size=observation,
        action_size=action,
        action_head_sizes=tuple(live["action_head_sizes"]),
        action_transport="factors",
        action_distribution="arsenal_standard_root_v1",
        episode_horizon_ticks=1024,
    )
    # Seeded by shape, so the same request mints a byte-identical file rather than
    # a new random one each launch -- the checkpoint hash is part of run identity.
    seed = int(hashlib.sha256(destination.name.encode()).hexdigest()[:8], 16)
    save_policy_checkpoint(
        destination,
        initialize_policy(jax.random.key(seed), config),
        config,
        metadata={
            "origin": "console scratch mint, not trained",
            "reason": "launch requested source_policy='scratch'",
            "observation_size": observation,
            "action_size": action,
            "encoder_size": encoder_size,
            "recurrent_size": recurrent_size,
        },
    )
    return destination


def _checkpoint_program_identity(path: Path) -> dict[str, Any]:
    """Hash checkpoint structure and contracts, never parameter values."""

    fallback = {
        "opaque_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper()
    }
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as values:
            config = json.loads(str(values["config_json"].item()))
            metadata = json.loads(str(values["metadata_json"].item()))
            parameters = {
                name: {
                    "shape": list(values[name].shape),
                    "dtype": str(values[name].dtype),
                }
                for name in sorted(values.files)
                if name not in {"format_version", "config_json", "metadata_json"}
            }
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return fallback
    transfer = metadata.get("transfer_contract") or {}
    return {
        "format_version": metadata.get("format_version", 1),
        "policy": {
            name: config.get(name)
            for name in (
                "encoder_size",
                "recurrent_size",
                "observation_size",
                "action_size",
                "action_head_sizes",
                "action_transport",
                "action_distribution",
            )
        },
        "contracts": {
            name: transfer.get(name)
            for name in (
                "combat_dynamics_contract_sha256",
                "arsenal_contract_sha256",
                "observation_contract_sha256",
                "action_contract_sha256",
                "world_geometry_token_contract_sha256",
            )
        },
        "parameters": parameters,
    }


def _result_cache_key(spec: Mapping[str, Any]) -> str:
    """Identity of one deterministic request, distinct from its executable."""

    value = {
        name: item
        for name, item in spec.items()
        if name not in {"_explicit_fields", "prepared_cache_key", "result_cache_key"}
    }
    source = Path(str(spec.get("source_policy", "")))
    if source.is_file() and "source_policy_sha256" not in value:
        value["source_policy_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest().upper()


def _json_safe(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Replace non-finite metrics with strings so a payload stays encodable.

    JSON has no inf or nan. A diverging run produces both, and the encoder then
    raises while serialising the job, so `/api/jobs/{id}` fails with a 500 at
    exactly the moment someone needs to see what went wrong. Keeping the value
    as text preserves the diagnosis -- "inf" is the finding -- while letting the
    response through.
    """

    safe: dict[str, Any] = {}
    for name, value in metrics.items():
        if isinstance(value, float) and not isfinite(value):
            safe[name] = "inf" if value > 0 else "-inf" if value < 0 else "nan"
        else:
            safe[name] = value
    return safe


def _world_selection_flags(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the generic world order into the flags the runtime accepts.

    The API takes `world_order` because a named order is extensible where a
    boolean per heuristic is not. The run config still speaks in booleans and is
    constructed by splatting this spec, so an unrecognised key is fatal rather
    than ignored. This is the one place that knows both vocabularies; delete it
    once the run config takes a selection query.

    `flattest_worlds` stays honoured as a deprecated alias.
    """

    order = str(spec.get("world_order") or "digest")
    if spec.get("flattest_worlds"):
        order = "flattest"
    return {
        "mixed_terrain_worlds": order == "mixed"
        or bool(spec.get("mixed_terrain_worlds", False)),
        "flattest_worlds": order == "flattest",
    }


def _cache_history(cache_key: str, result_key: str) -> dict[str, Any]:
    """Recover launch/cache evidence from durable console artifacts."""

    program_rows = [row for row in _cached_runs() if row["cache_key"] == cache_key]
    exact = None
    for row in program_rows:
        if (
            row["result_key"] == result_key
            and row["status"] == "finished"
            and (
                exact is None
                or (row.get("written_at") or "") > (exact.get("written_at") or "")
            )
        ):
            exact = row
    latest = max(
        program_rows, key=lambda row: row.get("written_at") or "", default=None
    )
    return {
        "matching_program_runs": len(program_rows),
        "executable_candidate_runs": sum(
            row["program_reached"] for row in program_rows
        ),
        "successful_program_runs": sum(
            row["status"] == "finished" for row in program_rows
        ),
        "latest": latest,
        "exact_result": exact,
    }


def _cached_runs() -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for directory in store.directories():
        report_path = directory / "report.json"
        if not report_path.is_file():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_id = str(report.get("run_id") or directory.name)
        if run_id in seen:
            continue
        seen.add(run_id)
        spec = report.get("spec") or {}
        cache_key = spec.get("prepared_cache_key")
        if not cache_key:
            continue
        summary = report.get("summary") or {}
        rows.append(
            {
                "cache_key": cache_key,
                "result_key": spec.get("result_cache_key"),
                "run_id": report.get("run_id", directory.name),
                "written_at": report.get("written_at"),
                "status": summary.get("status"),
                "timings": summary.get("launch_cache") or {},
                "program_reached": (
                    (summary.get("launch_cache") or {}).get("program_build_seconds")
                    is not None
                ),
            }
        )
    return rows


def _resolve_launch_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Resolve a named stage preset once; explicit request fields always win."""

    value = dict(spec)
    explicit = set(value.pop("_explicit_fields", ()))
    stage = value.get("training_stage", "pursuit_tracking")
    if stage == "combat":
        value["training_stage"] = stage
        return value
    if stage != "pursuit_tracking":
        from arena.training.contracts.catalog import training_contract

        contract = training_contract(stage)
        blockers = contract["collector"]["blockers"]
        detail = "; ".join(blockers) or "no executable collector is registered"
        raise ValueError(
            f"training contract {stage!r} is visible but not launchable: {detail}"
        )

    from arena.training.runs.pursuit_run import pursuit_launch_options

    options = pursuit_launch_options()
    preset_name = value.get("training_preset") or options["default_preset"]
    try:
        preset = options["presets"][preset_name]
    except KeyError as error:
        raise ValueError(f"unknown pursuit preset {preset_name!r}") from error
    aliases = {"batch": "num_envs"}
    for name, preset_value in preset.items():
        wire_name = aliases.get(name, name)
        if wire_name not in explicit:
            value[wire_name] = preset_value

    world_artifact_seed = value.get("world_artifact_seed")
    available_worlds = {
        int(item["seed"]) for item in options["world_backend"]["artifacts"]
    }
    if (
        world_artifact_seed is not None
        and int(world_artifact_seed) not in available_worlds
    ):
        raise ValueError(
            f"saved training world seed {world_artifact_seed} is unavailable"
        )

    requested_world = str(value.get("world") or "").strip()
    if "world" in explicit and requested_world not in ("", "region"):
        raise ValueError(
            f"training_stage 'pursuit_tracking' is region-only; world="
            f"{requested_world!r} cannot be honoured because launch-spec.json "
            "carries no world key and pursuit_run has no no-geometry branch. "
            "Accepting it would label the run as that world while building "
            "Region scenes."
        )

    updates = int(value["updates"])
    if "evaluation_updates" not in explicit:
        points = [int(point) for point in value.get("evaluation_updates", ())]
        value["evaluation_updates"] = sorted(
            {0, updates, *(point for point in points if 0 <= point <= updates)}
        )
    if value["evaluation_updates"][-1] != updates:
        raise ValueError("evaluation_updates must end at updates")
    if (
        "memory_reset_updates" not in explicit
        or value.get("memory_reset_updates") is None
    ):
        memory = value.get("memory_reset_updates") or ()
        value["memory_reset_updates"] = sorted(
            point
            for point in {int(point) for point in memory}
            if point in value["evaluation_updates"]
        )
    if not set(value["memory_reset_updates"]).issubset(value["evaluation_updates"]):
        raise ValueError("memory_reset_updates must be evaluation milestones")
    requested_policy = str(value.get("source_policy") or "").strip()
    if requested_policy.lower() == SCRATCH_SOURCE_POLICY:
        # Minted against the preset's widths, which are already merged into
        # `value` above -- the checkpoint is what actually sizes the network.
        resolved_policy = _mint_scratch_policy(
            int(value.get("encoder_size") or 64),
            int(value.get("recurrent_size") or 128),
        )
    elif requested_policy.lower() == LATEST_SOURCE_POLICY:
        resolved_policy = _latest_trained_policy()
    elif requested_policy:
        # Named explicitly: hand back exactly what was asked for. A stale or
        # missing file is the caller's to see, and preflight's `policy_contract`
        # check is what says so.
        resolved_policy = Path(requested_policy)
    else:
        # No policy named. `DEFAULT_PURSUIT_POLICY` is only a default while it is
        # actually usable, and it is neither published (`checkpoints/` is
        # excluded) nor kept current (it sits at 8271/99 against a live
        # 8272/124). Falling through to it unconditionally made the
        # out-of-the-box launch fail `preflight: policy_contract` on this
        # machine and fail with a missing file on a fresh clone -- so an
        # unusable default now mints a scratch policy at the live contract
        # instead of blocking every default launch.
        resolved_policy = Path(DEFAULT_PURSUIT_POLICY)
        if not resolved_policy.is_file() or _policy_contract_mismatch(resolved_policy):
            resolved_policy = _mint_scratch_policy(
                int(value.get("encoder_size") or 64),
                int(value.get("recurrent_size") or 128),
            )
    value.update(
        training_stage=stage,
        training_preset=preset_name,
        runner=value.get("runner") or "wsl",
        source_policy=str(resolved_policy),
        # Pursuit is region-only, and says so instead of pretending otherwise.
        # This was an unconditional `world="region"` with the request field
        # accepted by the schema and then silently assigned over, so a
        # `world="open_flat"` launch produced a run LABELLED open_flat that
        # built Region scenes anyway. Honouring the field here alone would not
        # have helped: `launch-spec.json` carries no `world` key at all, so the
        # worker never sees it and `pursuit_run` has no no-geometry branch --
        # the lie would just move one layer down. Refusing is the only honest
        # option until that branch exists.
        world="region",
        # Stage semantics, not caller options: pursuit is a fleeing target that
        # does not fight back.
        target_active=False,
        armed=False,
        shared_agent=(value.get("shared_agent") or store.BASIC_SHARED_AGENT),
        target_controller=(
            value.get("target_controller") or options["custom"]["fixed"]["target_mode"]
        ),
        target_separation_range=(
            value.get("target_separation_range")
            or options["custom"]["fixed"]["target_separation_range"]
        ),
        estimated_environment_steps=(
            int(value["num_envs"]) * updates * int(value["rollout_steps"])
        ),
        launch_contract=options["schema"],
    )
    # The episode horizon IS the evaluation window: an evaluation that stops
    # mid-episode measures a fragment, so the two cannot diverge. That is why
    # this is derived rather than configured.
    #
    # It used to be an unconditional assignment, which made `episode_ticks` a
    # field the API accepted, the Train form exposed as an editable control, and
    # the resolver then silently discarded -- asking for 900 produced 512 with
    # no error. Both controls default to 512, so an explicit disagreement is
    # always a real one and is now refused rather than quietly resolved.
    evaluation_steps = int(value["evaluation_steps"])
    requested_ticks = value.get("episode_ticks")
    if (
        "episode_ticks" in explicit
        and requested_ticks is not None
        and int(requested_ticks) != evaluation_steps
    ):
        raise ValueError(
            f"episode_ticks ({int(requested_ticks)}) must equal evaluation_steps "
            f"({evaluation_steps}): the episode horizon is the evaluation "
            "window, so set evaluation_steps and leave episode_ticks unset"
        )
    value["episode_ticks"] = evaluation_steps
    if value["runner"] != "wsl":
        raise ValueError("pursuit training requires runner='wsl' for CUDA JAX")
    source = Path(value["source_policy"])
    if not source.is_file():
        raise ValueError(f"source policy does not exist: {source}")
    value["source_policy_sha256"] = (
        hashlib.sha256(source.read_bytes()).hexdigest().upper()
    )
    return value


def _work(job: Job) -> None:
    """Run the PPO loop, streaming metrics, and finalise the artifact."""

    if job.spec.get("training_stage") == "pursuit_tracking":
        _work_pursuit(job)
        return

    directory = store.directory_for(job.job_id, spec=job.spec)
    directory.mkdir(parents=True, exist_ok=True)
    job.directory = str(directory)
    metrics_path = directory / METRICS
    opened = store.begin()

    # Bound before the try so the `finally` can record it even when the build
    # itself is what failed.
    config = None
    try:
        job.status = "compiling"
        live.publish("job_status", job.view())
        build_started = time.perf_counter()
        handle, updates, key, resolved = _build(job.spec)
        build_seconds = time.perf_counter() - build_started
        job.spec = resolved
        config, state = handle.ppo.initialize(key, **_ppo_overrides(job.spec))
        step = handle.ppo.train_step(config)
        job.cache.update(
            {
                "scene_hit": bool(
                    resolved.get("resource_cost", {}).get("scene_cache_hit", False)
                ),
                "prepare_seconds": build_seconds,
            }
        )

        import jax

        job.status = "training"
        live.publish("job_status", job.view())
        with metrics_path.open("w", encoding="utf-8") as sink:
            for update in range(updates):
                if job._cancel.is_set():
                    job.status = "cancelled"
                    break
                key, subkey = jax.random.split(key)
                state, metrics = step(state, subkey)
                row = {"update": update, **_host(metrics)}
                # Flushed per update: the point of streaming is that a run which
                # dies later still leaves everything it had already earned.
                sink.write(json.dumps(row, default=str) + "\n")
                sink.flush()
                job.update = update + 1
                job.latest = row
                live.publish(
                    "job_metric",
                    {"job_id": job.job_id, "row": row, "job": job.view()},
                )
        if job.status != "cancelled":
            job.status = "finished"

        # Persist the weights. Without this a training run produces loss curves
        # and no agent -- the trained params go out of scope when this function
        # returns, which makes the whole job unusable for the thing it exists to
        # do. Saved for a cancelled run too: 40 of 100 updates is a real
        # starting point, and discarding it punishes stopping early.
        if job.update:
            job.checkpoint = _checkpoint(directory, handle, state, config, job)
    except Exception as error:  # noqa: BLE001 - recorded on the job and artifact
        job.status = "failed"
        job.error = f"{type(error).__name__}: {error}"
        (directory / "traceback.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
    finally:
        job.finished = time.time()
        try:
            result = {
                "summary": {
                    "updates_completed": job.update,
                    "status": job.status,
                    "launch_cache": dict(job.cache),
                    **{f"final_{k}": v for k, v in job.latest.items() if k != "update"},
                },
                "hyperparameters": _hyperparameters(config),
                "warnings": ([job.error] if job.error else []),
                "checkpoint": job.checkpoint,
            }
            # Same directory the metrics streamed into -- see store.write's
            # `name`: the default clock-based id would split them apart.
            _write_artifact(job, result, opened)
        except Exception:  # pragma: no cover - never mask the training outcome
            pass
        live.publish("job_finished", job.view())
        _drain_after_finish(job)


def _work_pursuit(job: Job) -> None:
    """Dispatch one monitored CUDA stage while the Windows console stays live."""

    from console.core.execution.runners import to_wsl_path

    directory = store.directory_for(job.job_id, spec=job.spec)
    stage_dir = directory / "stage"
    stage_dir.mkdir(parents=True, exist_ok=True)
    job.directory = str(directory)
    opened = store.begin()
    spec_path = directory / "launch-spec.json"
    stdout_path = directory / "stdout.log"
    stderr_path = directory / "stderr.log"
    metrics_path = directory / METRICS
    report_path = stage_dir / "report.json"
    cancel_path = stage_dir / "CANCEL"
    root = to_wsl_path(PROJECT_ROOT)
    remote = {
        "source_policy": to_wsl_path(job.spec["source_policy"]),
        "output": to_wsl_path(stage_dir),
        "loadout": job.spec["loadout"],
        "opponent": job.spec["opponent"],
        "batch": int(job.spec["num_envs"]),
        "updates": int(job.spec["updates"]),
        "rollout_steps": int(job.spec["rollout_steps"]),
        "evaluation_steps": int(job.spec["evaluation_steps"]),
        "evaluation_updates": list(job.spec["evaluation_updates"]),
        "memory_reset_updates": list(job.spec["memory_reset_updates"]),
        "archive_lanes": int(job.spec["archive_lanes"]),
        "update_epochs": int(job.spec["update_epochs"]),
        "num_minibatches": int(job.spec["num_minibatches"]),
        "learning_rate": float(job.spec["learning_rate"]),
        "entropy_coefficient": float(job.spec["entropy_coefficient"]),
        "seed": int(job.spec["seed"]),
        "stage_options": dict(job.spec.get("stage_options") or {}),
        "target_options": dict(job.spec.get("target_options") or {}),
        "target_controller": str(job.spec["target_controller"]),
        "world_artifact_seed": job.spec.get("world_artifact_seed"),
        "world_count": int(job.spec.get("world_count", 1)),
        "environment_diversity": bool(job.spec.get("environment_diversity", False)),
        # `world_order` is the generic control, but the run config does not
        # accept it yet and this dict is splatted into that dataclass, so
        # translate the order into the flags it does accept.
        **_world_selection_flags(job.spec),
        "target_separation_range": list(job.spec["target_separation_range"]),
        "minimum_baseline_visible_fraction": float(
            job.spec["minimum_baseline_visible_fraction"]
        ),
    }
    # Same rule as `sensor_range` below: forwarding a default the caller never
    # asked for is not neutral here. `pursuit_run` derives this ceiling from the
    # live action-head count when it is unset, and unconditionally forwarding
    # the schema default overrode that derivation -- run
    # 20260820T005935Z-f7ba0fed stopped at update 63 against 0.03 while the
    # derived ceiling for its twelve heads was 0.06.
    if job.spec.get("maximum_approximate_kl") is not None:
        remote["maximum_approximate_kl"] = float(
            job.spec["maximum_approximate_kl"]
        )
    # An explicit null means "use the authored sensor range", which is not the
    # same as omitting the key and inheriting the run config default, so only
    # forward the field when the caller actually supplied one.
    if "sensor_range" in job.spec:
        remote["sensor_range"] = job.spec["sensor_range"]
    # Design world and the adaptive evader's knobs. Same rule again: forward
    # only what the caller supplied, so an omitted field keeps the run config
    # default instead of being pinned to a schema default the caller never
    # chose. `world_design` unset is what keeps every existing Region launch
    # on exactly the path it used before.
    for _design_field in (
        "world_design",
        "arena_radius",
        "evader_gait_mix",
        "evader_gait_period_ticks",
        "evader_standoff_band",
        "evader_standoff_strafe_ticks",
        "opponent_checkpoint",
        "objective",
        "learner_stamina_fraction",
        "learner_stamina_regen_scale",
        "evader_terminal_reward",
        "evader_contact_reward",
        "evader_travel_reward",
        "evader_separation_reward",
        "evader_forward_travel_reward",
        "evader_alive_reward",
        "evader_allow_jump",
        "evader_airborne_idle_cost",
        "evader_strafe_reward",
        "evader_sprint_reward",
        "evader_turn_reward",
        "evader_body_away_reward",
        "evader_band_leads",
        "evader_caught_penalty",
        # Accepted by the train schema since before this list existed, but read
        # only by `_ppo_overrides`, which serves the in-process trainer and not
        # this path -- so a pursuit run silently kept the 0.5 default however it
        # was launched.
        "value_coefficient",
    ):
        if job.spec.get(_design_field) is not None:
            remote[_design_field] = job.spec[_design_field]
    # Same rule, and the same bug these two shared with `world_count` before
    # it: both were already in the prepared-program cache key above, so two
    # requests differing only here were correctly treated as distinct
    # programs -- and then the worker was handed a spec that named neither, so
    # both silently fell back to the `PursuitRunConfig` defaults. With
    # `world_count > 1` the resolver returns `run.selection_key` directly and
    # never consults `run.seed`, so every multi-world run so far trained on
    # the *same* sixteen worlds and the same spawn pool no matter what seed
    # was requested. Omit to keep the historical default; supply to rotate.
    for name in ("selection_key", "node_seed"):
        if name in job.spec and job.spec[name] is not None:
            remote[name] = int(job.spec[name])
    # `microticks` had the same shape of bug and a worse symptom: TrainRequest
    # accepts 1..4, preflight echoes the value back under `resolved`, and it
    # reached nothing at all -- neither this spec nor `pursuit_run`. A launch
    # could ask for 4, be told it got 4, and train at 1.
    #
    # That matters because it is the one knob task #36 prescribes for the
    # locomotion cap, which measured 25.1-25.6% of the sprint ceiling across
    # four reward configurations that should have moved it and did not.
    #
    # NOT FORWARDED. This dict is splatted into `PursuitRunConfig(**spec)`, so
    # a key that dataclass does not declare raises TypeError before the run
    # starts rather than being ignored. Re-enable only together with the arena
    # half: a `microticks` field on PursuitRunConfig threaded through
    # `_materialize_region` -> `load_region` -> `default_combat_params`, which
    # currently fixes microticks at 1.
    spec_path.write_text(json.dumps(remote, indent=2), encoding="utf-8")
    quoted_root = shlex.quote(root)
    memory = _gpu_memory_plan()
    job.cache["allocator"] = memory
    command = (
        f"cd {quoted_root} && "
        # `cuda,cpu` rather than `cuda`: CUDA stays the default so training is
        # unaffected, but a CPU device exists for the stage to place evaluation
        # on. With `cuda` alone `jax.devices("cpu")` raises "Unknown backend
        # cpu", so there is nowhere to move the evaluation to.
        "export JAX_PLATFORMS=cuda,cpu CUDA_VISIBLE_DEVICES=0 "
        "HYTALERL_JAX_RUNTIME_PROFILE=throughput "
        f"XLA_PYTHON_CLIENT_PREALLOCATE={str(memory['preallocate']).lower()} "
        f"XLA_PYTHON_CLIENT_MEM_FRACTION={memory['fraction']:.4f} "
        # $HOME rather than a literal home directory: this string is executed by
        # a WSL shell, so it resolves per user instead of pinning one machine.
        "HYTALERL_JAX_CACHE=\"$HOME/.cache/hytalerl/jax\" "
        f"PYTHONPATH={quoted_root}/HytaleRL/hytalegym:{quoted_root} && "
        "if [ -x \"$HOME/.cache/hytalerl-jax-cuda13-0.10.2/bin/python\" ]; "
        "then exec \"$HOME/.cache/hytalerl-jax-cuda13-0.10.2/bin/python\" "
        "-u -m agents.basic.worker; "
        "else exec python3 -u -m agents.basic.worker; fi"
    )
    last_sequence = -1
    job.status = "starting_gpu"
    live.publish("job_status", job.view())

    try:
        with metrics_path.open("w", encoding="utf-8") as metric_sink:

            def poll_worker() -> None:
                nonlocal last_sequence
                if job._cancel.is_set():
                    cancel_path.touch(exist_ok=True)
                    job.status = "cancelling"
                last_sequence = _ingest_stage_events(
                    job, stage_dir, metric_sink, last_sequence, opened
                )

            response = basic_worker.run(
                command,
                {
                    "id": job.job_id,
                    "prepared_cache_key": job.spec["prepared_cache_key"],
                    "spec_path": to_wsl_path(spec_path),
                    "stdout_path": to_wsl_path(stdout_path),
                    "stderr_path": to_wsl_path(stderr_path),
                },
                poll=poll_worker,
                launch_metadata=memory,
            )
            poll_worker()
            last_sequence = _ingest_stage_events(
                job, stage_dir, metric_sink, last_sequence, opened
            )
            job.cache["allocator"] = response["launch_metadata"]
            job.cache["resident_worker"] = {
                key: response[key]
                for key in (
                    "process_run",
                    "process_reused",
                    "contract_reused",
                    "source_sha256",
                )
            }
            if response["status"] != "ok":
                # The worker already reports why it stopped. Prefer that over
                # the stderr tail: under JAX the last stderr line is almost
                # always an XLA constant-folding advisory ("If you'd like to
                # file a bug, run with envvar XLA_FLAGS=..."), so a run that
                # failed for an entirely different reason -- SourceDrift, most
                # often -- reported an XLA problem it did not have, and sent
                # debugging after a nonexistent bug.
                tail = stderr_path.read_text(encoding="utf-8", errors="replace")
                detail = _worker_failure_detail(response, tail)
                # A worker fault after the stage finished is a *transport*
                # failure, not a training failure. The run already wrote its
                # report, checkpoints and replays; raising here threw all of
                # that away and reported a completed run as failed, which is
                # how a finished run could vanish from the Archive because its
                # worker died on the way home.
                #
                # SourceDrift is deliberately excluded: it means the training
                # source moved while the run was measuring, so the numbers are
                # void (class 4) rather than merely unreported, and accepting
                # them would file an uninterpretable artifact.
                drifted = str(response.get("error_type")) == "SourceDrift"
                salvageable = report_path.is_file() and not drifted
                if job._terminated.is_set():
                    # The worker died because someone killed it. Reporting that
                    # as a training failure would file a user's stop button as
                    # a defect in the run.
                    job.status = "stopped"
                    job.stop_reason = "terminated by request"
                    return
                if not salvageable:
                    raise RuntimeError("resident WSL pursuit failed: " + detail)
                job.cache["worker_fault_after_completion"] = {
                    "error_type": str(response.get("error_type") or "unknown"),
                    "detail": detail,
                    "note": (
                        "the stage completed and its report was recovered; the "
                        "worker failed after the run, not during it"
                    ),
                }

        if not report_path.is_file():
            raise RuntimeError("pursuit stage exited without report.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        # The stage reports three terminal statuses and they are not
        # interchangeable. Collapsing the guardrail one into "finished" made a
        # run that died at update 3 of 256 read exactly like a clean 256-update
        # run, with error=None, so the only way to tell was to open stdout.log.
        report_status = str(report.get("status", ""))
        if report_status.startswith("cancelled"):
            job.status = "cancelled"
        elif report_status.startswith("stopped_by_guardrail"):
            job.status = "stopped"
            job.stop_reason = str(
                (report.get("training_execution") or {}).get("stop_reason")
                or "guardrail"
            )
        else:
            job.status = "finished"
        agent_artifact = BASIC_ARTIFACT_ROOT / job.job_id
        shutil.copytree(stage_dir, agent_artifact, dirs_exist_ok=True)
        job.checkpoint = str(agent_artifact / "pursuit_policy.npz")
        selection = report["selection"]
        selected = selection["metrics"]
        selected_update = int(selection["selected_update"])
        job.latest = {"update": selected_update, **selected}
        replay = report["replay"]
        latest_flags = selection["latest"].get("flags", [])
        result = {
            **replay,
            "head_names": report["head_names"],
            "target_phases": report["target_phases"],
            "summary": {
                "status": report["status"],
                "updates_completed": job.update,
                "selected_update": selected_update,
                "environment_steps": job.update
                * int(job.spec["num_envs"])
                * int(job.spec["rollout_steps"]),
                "launch_cache": dict(job.cache),
                **selected,
            },
            "warnings": [
                "curriculum checkpoint; Java fidelity and promotion are not yet assessed",
                *(
                    ["latest milestone rejected: " + ", ".join(latest_flags)]
                    if selection.get("latest_was_rejected")
                    else []
                ),
                *(
                    [
                        "worker faulted after the stage completed and the report "
                        "was recovered: "
                        + str(job.cache["worker_fault_after_completion"]["detail"])
                    ]
                    if "worker_fault_after_completion" in job.cache
                    else []
                ),
            ],
            "checkpoint": job.checkpoint,
            "agent_artifact": str(agent_artifact),
        }
        try:
            result["summary"]["archived_replays"] = _archive_pursuit_replays(
                job, stage_dir, report, opened
            )
        except Exception as error:  # noqa: BLE001 - preserve the trained checkpoint
            result["warnings"].append(
                f"evaluation replay archive incomplete: {type(error).__name__}: {error}"
            )
        _write_artifact(job, result, opened)
    except Exception as error:  # noqa: BLE001 - durable job failure surface
        job.status = "failed"
        job.error = f"{type(error).__name__}: {error}"
        (directory / "traceback.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        try:
            _write_artifact(
                job,
                {
                    "summary": {
                        "status": job.status,
                        "updates_completed": job.update,
                        "launch_cache": dict(job.cache),
                    },
                    "warnings": [job.error],
                },
                opened,
            )
        except Exception:  # pragma: no cover - never replace the launch error
            pass
    finally:
        job.finished = time.time()
        live.publish("job_finished", job.view())
        _drain_after_finish(job)


def _archive_pursuit_replays(
    job: Job,
    stage_dir: Path,
    report: Mapping[str, Any],
    opened: Mapping[str, Any],
) -> int:
    """Publish every fixed-milestone replay as an honest archive entry."""

    root = stage_dir.resolve()
    metrics = {int(row["update"]): row["recurrent"] for row in report["curve"]}
    spec = {
        key: value
        for key, value in job.spec.items()
        if key not in {"prepared_cache_key", "result_cache_key"}
    }
    count = 0
    for entry in report.get("replay_archive", ()):
        path = (stage_dir / entry["path"]).resolve()
        path.relative_to(root)
        replay = json.loads(path.read_text(encoding="utf-8"))
        update, lane = int(entry["update"]), int(entry["lane"])
        store.write(
            {
                **spec,
                "artifact_kind": "pursuit_evaluation_replay",
                "parent_run_id": job.job_id,
                "evaluation_update": update,
                "evaluation_lane": lane,
                "ticks": int(replay["steps"]),
            },
            {
                **replay,
                "head_names": report["head_names"],
                "target_phases": report["target_phases"],
                "summary": {
                    "status": "archived_evaluation_replay",
                    "parent_run_id": job.job_id,
                    "evaluation_update": update,
                    "evaluation_lane": lane,
                    **metrics[update],
                },
                "warnings": [
                    "fixed-seed evaluation replay; lane is diagnostic, not a replicate"
                ],
            },
            dict(opened),
            name=f"{job.job_id}-u{update:04d}-l{lane:03d}",
        )
        count += 1
    return count


def _gpu_memory_plan() -> dict[str, Any]:
    """Use a bounded reservation only on an idle GPU; share on demand otherwise."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        total, used, free, utilization = (
            int(value.strip()) for value in result.stdout.splitlines()[0].split(",")
        )
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return {"mode": "on_demand_fallback", "preallocate": False, "fraction": 0.50}
    if utilization >= 20 or used >= 2_048:
        return {
            "mode": "shared_gpu_on_demand",
            "preallocate": False,
            "total_mib": total,
            "used_mib": used,
            "free_mib": free,
            "utilization_percent": utilization,
            "fraction": 0.50,
        }
    headroom = 2_048
    fraction = max(0.55, min(0.82, (free - headroom) / max(total, 1)))
    return {
        "mode": "dedicated_gpu_preallocated",
        "preallocate": True,
        "total_mib": total,
        "used_mib": used,
        "free_mib": free,
        "utilization_percent": utilization,
        "headroom_mib": headroom,
        "fraction": fraction,
    }


def compute_status() -> dict[str, Any]:
    """Live launcher state for Console/ADK compute clients."""

    from arena.publication_worlds import publication_cache_info
    from console.core.worlds import replay_terrain, terrain, worldgen

    job = active()
    return {
        "schema": "console-training-compute-runtime-v1",
        "allocator_plan": _gpu_memory_plan(),
        "resident_worker": basic_worker.status(),
        "terrain_replay_cache": terrain.cache_state(),
        "replay_world_cache": replay_terrain.cache_state(),
        "generated_training_world_cache": publication_cache_info(),
        "generated_replay_cache": worldgen.capture_cache_state(),
        "active_job": (
            None
            if job is None
            else {
                "job_id": job.job_id,
                "status": job.status,
                "stage": job.spec.get("training_stage"),
                "preset": job.spec.get("training_preset"),
                "num_envs": job.spec.get("num_envs"),
                "rollout_steps": job.spec.get("rollout_steps"),
                "world_artifact_seed": job.spec.get("world_artifact_seed"),
            }
        ),
        "launch_policy": {
            "single_gpu_jobs": "serialized",
            "persistent_compilation_cache": "$HOME/.cache/hytalerl/jax",
            "static_change_rule": "new prepared cache identity",
            "world_rule": "saved artifacts materialize before JIT",
        },
    }


def _publish_milestone_replays(job, stage_dir, update, metrics, opened) -> None:
    """Archive one milestone's replay lanes while the run is still going.

    Best effort by construction: an archive problem must never take down a
    training run that is otherwise healthy, so every failure is recorded on the
    job and swallowed. Entries are provisional -- `head_names`/`target_phases`
    only exist in the final report -- and `_archive_pursuit_replays` rewrites
    the same directory names at the end with the complete record.
    """

    if opened is None:
        return
    published = getattr(job, "_published_replays", None)
    if published is None:
        published = set()
        job._published_replays = published
    # Event metrics are role-prefixed; the report-backed entries are not.
    summary_metrics = {
        key.split(".", 1)[1] if key.startswith("learner.") else key: value
        for key, value in (metrics or {}).items()
    }
    spec = {
        key: value
        for key, value in job.spec.items()
        if key not in {"prepared_cache_key", "result_cache_key"}
    }
    try:
        lanes = sorted((stage_dir / "replays").glob(f"update-{update:04d}-lane-*.json"))
    except OSError as error:
        job.cache["replay_archive_error"] = f"{type(error).__name__}: {error}"
        return
    for path in lanes:
        try:
            lane = int(path.stem.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if (update, lane) in published:
            continue
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
            store.write(
                {
                    **spec,
                    "artifact_kind": "pursuit_evaluation_replay",
                    "parent_run_id": job.job_id,
                    "evaluation_update": update,
                    "evaluation_lane": lane,
                    "ticks": int(replay["steps"]),
                },
                {
                    **replay,
                    "summary": {
                        "status": "archived_evaluation_replay_in_flight",
                        "parent_run_id": job.job_id,
                        "evaluation_update": update,
                        "evaluation_lane": lane,
                        **summary_metrics,
                    },
                    "warnings": [
                        "fixed-seed evaluation replay; lane is diagnostic, "
                        "not a replicate",
                        "published mid-run; rewritten when the run reports",
                    ],
                },
                dict(opened),
                name=f"{job.job_id}-u{update:04d}-l{lane:03d}",
            )
        except (OSError, ValueError, KeyError) as error:
            job.cache["replay_archive_error"] = f"{type(error).__name__}: {error}"
            continue
        published.add((update, lane))
    job.cache["archived_replays_in_flight"] = len(published)


def _ingest_stage_events(
    job, stage_dir, metric_sink, last_sequence: int, opened=None
) -> int:
    """Fold append-only Arena monitor events into the console job surface."""

    path = stage_dir / "events.jsonl"
    if not path.is_file():
        return last_sequence
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        sequence = int(event["sequence"])
        if sequence <= last_sequence:
            continue
        last_sequence = sequence
        kind = event["kind"]
        payload = event.get("payload") or {}
        if kind == "preparing":
            job.status = "starting"
            job.cache.update(
                persistent_path=payload.get("persistent_compilation_cache"),
                policy="jax_content_addressed_fail_closed",
            )
        # The stage reports its two long startup phases separately, and they are
        # not the same wait: building the scene is world I/O, tracing and
        # compiling the program is XLA. Reporting both as "compiling" put the
        # status on the shorter of the two for the whole of the longer one.
        elif kind == "scene_build":
            job.status = "building_scene"
        elif kind == "scene_ready":
            job.status = "compiling"
            job.cache["scene_build_seconds"] = payload.get("scene_build_seconds")
        elif kind == "program_ready":
            job.status = "evaluating_baseline"
            job.cache.update(
                scene_build_seconds=payload.get("scene_build_seconds"),
                saved_world_materialization_cache_hit=payload.get(
                    "saved_world_materialization_cache_hit"
                ),
                program_build_seconds=payload.get("program_build_seconds"),
            )
        elif kind == "training_update":
            job.status = "training"
            job.update = int(payload["update"])
            row = {
                "kind": "training",
                "update": job.update,
                **_json_safe(payload.get("metrics") or {}),
            }
            job.latest = row
            metric_sink.write(json.dumps(row) + "\n")
            metric_sink.flush()
            if job.update == 1:
                job.cache["first_update_seconds"] = payload.get("wall_seconds")
            job.cache["steps_per_second"] = payload.get("steps_per_second")
        elif kind == "evaluation":
            job.status = "evaluating"
            row = {
                "kind": "evaluation",
                "update": int(payload["update"]),
                **(payload.get("metrics") or {}),
            }
            metric_sink.write(json.dumps(row) + "\n")
            metric_sink.flush()
            if int(payload["update"]) == 0:
                job.cache["baseline_evaluation_seconds"] = payload.get("wall_seconds")
            replay = payload.get("replay")
            if isinstance(replay, Mapping) and replay.get("path"):
                job.latest_replay = {
                    **replay,
                    "recorded_unix_seconds": event.get("unix_seconds"),
                }
            # Publish this milestone's lanes now rather than at the end. The
            # end-of-run pass only runs once report.json exists, so a run that
            # dies mid-flight leaves its replays on disk and nothing in the
            # archive. Same directory names, so the final pass overwrites these
            # with the authoritative report-backed entries.
            _publish_milestone_replays(
                job, stage_dir, int(payload["update"]), payload.get("metrics"), opened
            )
        elif kind == "failed":
            job.status = "failed"
            job.error = (
                f"{payload.get('error_type', 'Error')}: "
                f"{payload.get('message', 'training preparation failed')}"
            )
        live.publish(
            "job_metric" if kind in {"training_update", "evaluation"} else "job_status",
            {"job_id": job.job_id, "event": event, "job": job.view()},
        )
    return last_sequence


def replay(job_id: str) -> dict[str, Any] | None:
    """Read the newest completed evaluation lane for one live job."""

    with _lock:
        job = _jobs.get(job_id)
    if job is None or not job.latest_replay or not job.directory:
        return None
    root = (Path(job.directory) / "stage").resolve()
    path = (root / str(job.latest_replay["path"])).resolve()
    try:
        path.relative_to(root)
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None

    trajectory = payload.get("trajectory") or {}
    terrain = payload.get("terrain") or None
    if trajectory.get("agent_x") and not (terrain or {}).get("blocks"):
        from console.core.worlds import replay_terrain

        identity = payload.get("world_identity") or payload.get("region") or {}
        reference = terrain
        if not reference and str(identity.get("generator", "")).startswith(
            "worldgen_v2"
        ):
            reference = replay_terrain.worldgen_reference(
                identity,
                seed=identity.get("replay_world_seed"),
            )
        if job.spec.get("world_design"):
            # A design world has no captured geometry to hydrate: its reference
            # is a `reference_only` stub with zero cells, and every `resolve`
            # branch is a Region lookup that finds nothing, so the archive drew
            # an empty scene. The ground is a plane, so rebuild it from the
            # trajectory the actors walked on.
            terrain = replay_terrain.flat_from_trajectory(trajectory)
        elif reference:
            terrain = replay_terrain.resolve(
                reference,
                trajectory,
                cache_key=f"{path}:{path.stat().st_mtime_ns}",
            )
    width = len((trajectory.get("action") or [[]])[0])
    return {
        **payload,
        "run_id": job_id,
        "spec": dict(job.spec),
        "terrain": terrain,
        "events": [],
        "warnings": ["fixed-seed evaluation lane; diagnostic, not a replicate"],
        "head_names": [f"head_{index}" for index in range(width)],
        "target_phases": ["training"],
        "evaluation": {
            key: value for key, value in job.latest_replay.items() if key != "path"
        },
    }


def _hyperparameters(config) -> dict[str, Any]:
    """Every value the run actually resolved to, defaults included.

    The spec records what was *asked for*. A knob the caller left alone is
    absent from it entirely, so an artifact cannot distinguish "gamma was 0.99"
    from "gamma was never mentioned" -- and two runs that differ only in a
    default read as identical. That is precisely the comparison a sweep exists
    to make, so the resolved config is recorded rather than reconstructed.

    It also carries the derived contract fields -- observation size, the twelve
    head widths, the action transport. Those are not tuning knobs, but they are
    what decides whether a checkpoint from this run will ever load again, and
    they are the first thing to check when it does not.
    """

    if config is None:
        return {}
    from dataclasses import asdict, is_dataclass

    if not is_dataclass(config):
        return {}
    # Tuples become lists so the artifact is plain JSON rather than something
    # that only round-trips through Python.
    return {
        name: (list(value) if isinstance(value, tuple) else value)
        for name, value in asdict(config).items()
    }


def _checkpoint(directory: Path, handle, state, config, job: Job) -> str:
    """Save the trained weights beside the metrics that describe them.

    Saved through the *handle* rather than `adk.core.lifecycle.save_checkpoint`
    directly. That function's second argument is a `BuiltAgent`, and a handle is
    not one -- it *holds* one, and `AgentHandle.save_checkpoint` passes it along.
    Calling the module function with the handle raised `built must be a
    BuiltAgent` and cost a run its weights, which is how this is known.

    The whole train state goes in, not `state.policy_params`. It carries
    `update_count` and `total_environment_steps`, so the progress counters come
    from the run itself; the arithmetic they replace -- updates x rollout_steps
    -- silently ignored the batch width and understated the real step count by a
    factor of `BATCH`.

    The write still re-validates the build, the PPO config and the *live*
    contracts, and refuses when they no longer agree. That refusal is the useful
    outcome, not an obstacle: a checkpoint written against contracts that have
    already moved is one nobody can interpret later, and this project has a
    checkpoint that went stale about a minute after training. So the reason is
    recorded rather than swallowed.
    """

    try:
        path = handle.save_checkpoint(
            directory / CHECKPOINT,
            state,
            config,
            seed=int(job.spec.get("seed", 0)),
            combat_target_active=bool(job.spec.get("target_active", True)),
            run_metadata={
                "source": "console",
                "job_id": job.job_id,
                "status": job.status,
                "metrics": METRICS,
                # Needed to reload these weights at all: `_validate_config`
                # compares `num_envs` against the rebuilt handle's batch, so a
                # reader who does not know the width cannot open the file.
                "num_envs": job.spec.get("num_envs"),
            },
        )
        return str(path)
    except Exception as error:  # noqa: BLE001 - the reason is the record
        (directory / REFUSED).write_text(
            f"{type(error).__name__}: {error}\n", encoding="utf-8"
        )
        return f"refused: {type(error).__name__}: {error}"


def _host(metrics: Any) -> dict[str, Any]:
    """Pull one update's metrics to the host as plain floats.

    This synchronises, which is exactly what we want once per update and never
    inside one: the alternative is a growing queue of device arrays nobody has
    read, held alive by the job registry.
    """

    if hasattr(metrics, "_asdict"):
        metrics = metrics._asdict()
    elif hasattr(metrics, "__dict__") and not isinstance(metrics, dict):
        metrics = vars(metrics)
    if not isinstance(metrics, dict):
        return {"value": float(metrics)}
    out: dict[str, Any] = {}
    for name, value in metrics.items():
        try:
            out[name] = float(value)
        except (TypeError, ValueError):
            out[name] = str(value)
    return out


def _build(spec: dict[str, Any]):
    """Build the agent handle for training from the console's own scene knobs."""

    import jax

    from console.core.execution.runner import (
        RunSpec,
        _bounded,
        _scene_for,
        _shaped,
        rotated,
    )

    run_spec = RunSpec(
        **{k: v for k, v in spec.items() if k in RunSpec.__dataclass_fields__}
    )
    # Resolve `zone="rotate"` to a concrete zone BEFORE `resolved` is built and
    # before the scene is keyed, so the artifact records the terrain the policy
    # actually trained in rather than the word "rotate".
    run_spec = rotated(run_spec)
    envs = max(1, int(spec.get("num_envs") or TRAIN_ENVS))
    # Return the *resolved* configuration alongside the handle. A launch that
    # omits `armed` still ran with RunSpec's default, and an artifact leaving
    # that field null claims the run had no value for it -- the same
    # under-description this store exists to prevent, and it surfaces as a false
    # difference the moment a training run is compared against a rollout.
    from dataclasses import asdict

    resolved = {**asdict(run_spec), **spec, "num_envs": envs}
    # `region_note` carries which captured region actually loaded. Training on a
    # Region without recording that would leave a checkpoint whose world cannot
    # be identified afterwards.
    handle, region_note = _scene_for(run_spec, envs)
    # Shape the handle for TRAINING too, not only for rollouts. Without this the
    # console honoured a minigame when inspecting a policy and ignored it when
    # training one -- and `resolved` below still reported the minigame, so the
    # artifact claimed a shaped run that had trained on the unshaped reward.
    # Nothing raises on that path; the loss curve looks entirely normal.
    handle = _bounded(_shaped(handle, run_spec), run_spec)
    resolved = {**resolved, **({"region": region_note} if region_note else {})}
    updates = max(1, int(spec.get("updates", 10)))
    return handle, updates, jax.random.key(int(spec.get("seed", 0))), resolved


#: Console knob -> `PPOConfig` field. The console's names are the ones people
#: say out loud; the config's are the ones that exist. Passing a name PPOConfig
#: does not have raises, and passing only the three that happened to match
#: meant gamma, clip, entropy and the value coefficient were silently ignored --
#: the controls were decorative and a sweep over them would have measured
#: nothing.
PPO_FIELDS = {
    "rollout_steps": "rollout_steps",
    "episode_ticks": "episode_horizon_ticks",
    "update_epochs": "update_epochs",
    "num_minibatches": "num_minibatches",
    # Network width. Absent until now, so the console could not vary the
    # architecture at all -- every run trained the same 64/128 policy. They are
    # independent: the equal-width constraint was retired, and a checkpoint
    # remembers the widths it trained at, so these are also what decides
    # whether a saved policy will load later.
    "encoder_size": "encoder_size",
    "recurrent_size": "recurrent_size",
    "learning_rate": "learning_rate",
    "gamma": "gamma",
    "gae_lambda": "gae_lambda",
    "clip": "clip_epsilon",
    "clip_epsilon": "clip_epsilon",
    "value_coef": "value_coefficient",
    "value_coefficient": "value_coefficient",
    "entropy_coef": "entropy_coefficient",
    "entropy_coefficient": "entropy_coefficient",
    "max_gradient_norm": "max_gradient_norm",
}


def _ppo_overrides(spec: dict) -> dict:
    """Translate the console's hyperparameter names to `PPOConfig` fields."""

    out: dict = {}
    for name, field_name in PPO_FIELDS.items():
        value = spec.get(name)
        if value is None:
            continue
        try:
            out[field_name] = float(value) if isinstance(value, float) else value
        except (TypeError, ValueError):
            continue
    return out


def checkpoint(job_id: str) -> tuple[Path | None, str | None]:
    """Resolve a job's weights to ``(path, refusal)``, memory first, then disk.

    The registry is in-memory by design, but the checkpoint is the one thing
    about a job that is meant to outlive the process. Restarting the server must
    not orphan weights that are still on disk -- so a job this process never ran
    is answered from the artifact directory, which is the durable record
    anyway. Exactly one side of the pair is ever populated.
    """

    view = get(job_id)
    if view is not None:
        saved = view.get("checkpoint")
        if saved and str(saved).startswith("refused:"):
            return None, str(saved)
        if saved:
            return Path(saved), None

    directory = store.locate_directory(job_id)
    if directory is None:
        return None, None
    path = directory / CHECKPOINT
    if path.is_file():
        return path, None
    staged = directory / "stage" / "pursuit_policy.npz"
    if staged.is_file():
        return staged, None
    refused = directory / REFUSED
    if refused.is_file():
        return None, f"refused: {refused.read_text(encoding='utf-8').strip()}"
    return None, None


def metrics(job_id: str, limit: int = 5000) -> list[dict[str, Any]]:
    """Read a job's metric series back off disk."""

    directory = store.locate_directory(job_id)
    if directory is None:
        return []
    path = directory / METRICS
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(rows) >= limit:
                break
    return rows
