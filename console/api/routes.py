"""HTTP endpoints.

Every route is a thin translation onto :mod:`console.core`. If a route ever
grows logic of its own, that logic belongs in ``core`` instead — the browser is
one consumer of the engine, not its owner.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from console.api.schemas import (
    ProfileReplayRequest,
    RunRequest,
    StreamLogRequest,
    TrainRequest,
    WorldStatusRequest,
)
from console.core import panels
from console.core.catalog import agents, profiles
from console.core.evidence import agent_archive, store
from console.core.execution import checks, devices, hytale, jobs
from console.core.telemetry import live, streams
from console.core.worlds import custom_games, worldgen
from console.core.execution.runner import RunSpec, options, run, training_workflows
from console.core.telemetry.surfaces import adk_surface, bridge_status, policy_agent_status

STATIC = Path(__file__).resolve().parents[1] / "static"

router = APIRouter()


def _require_profile(profile_id: str | None) -> None:
    if not profile_id:
        return
    profile = profiles.get(profile_id)
    if profile is None:
        raise HTTPException(status_code=400, detail=f"unknown profile {profile_id!r}")
    if profile.get("error"):
        raise HTTPException(status_code=400, detail=profile["error"])


@router.get("/api/options", tags=["run"])
def api_options():
    """Everything the UI needs to populate its controls, read from the ADK."""
    return options()


@router.get("/api/worldgen/options", tags=["worldgen"])
def api_worldgen_options():
    """WorldGen Studio's field contract, presets and native-exactness boundary."""

    return worldgen.options()


@router.post("/api/worldgen/preview", tags=["worldgen"])
def api_worldgen_preview(payload: dict):
    """Generate one deterministic semantic design preview."""

    try:
        return worldgen.preview(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/worldgen/batch", tags=["worldgen"])
def api_worldgen_batch(payload: dict):
    """Generate a compact deterministic preview corpus and native seed plan."""

    try:
        return worldgen.batch(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/worldgen/builds", tags=["worldgen"])
def api_worldgen_builds():
    """Locally compiled, undeployed WorldGen V2 asset packs."""

    return worldgen.builds()


@router.get("/api/worldgen/captures", tags=["worldgen"])
def api_worldgen_captures():
    """Published exact composite worlds available for voxel inspection."""

    try:
        return worldgen.captured_worlds()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/api/worldgen/captures/{capture_id}/volume", tags=["worldgen"])
def api_worldgen_captured_volume(capture_id: str, target_columns: int = 96):
    """Render one authenticated composite Region world without regenerating it."""

    try:
        return worldgen.captured_volume(
            capture_id,
            target_columns=target_columns,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/api/worldgen/compile", tags=["worldgen"])
def api_worldgen_compile(payload: dict):
    """Compile one Studio design offline; never deploy or move a corpus pin."""

    try:
        return worldgen.compile_native(payload)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/worldgen/designs", tags=["worldgen"])
def api_worldgen_designs():
    """Saved portable design manifests, newest first."""

    return worldgen.saved()


@router.post("/api/worldgen/designs", tags=["worldgen"])
def api_worldgen_save_design(payload: dict):
    """Validate and atomically save a custom-world design manifest."""

    try:
        return worldgen.save(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/worldgen/designs/{design_id}", tags=["worldgen"])
def api_worldgen_design(design_id: str):
    """One saved design, suitable for restoring the controls."""

    try:
        record = worldgen.load(design_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(
            status_code=404, detail=f"unknown worldgen design {design_id!r}"
        )
    return record


@router.get("/api/custom/minigames/options", tags=["custom"])
def api_custom_minigame_options():
    """Arena goal and loadout choices for the declarative creator."""

    return custom_games.options()


@router.get("/api/custom/minigames", tags=["custom"])
def api_custom_minigames():
    """Saved custom minigame manifests, newest first."""

    return custom_games.saved()


@router.post("/api/custom/minigames", tags=["custom"])
def api_create_custom_minigame(payload: dict):
    """Validate one Arena task against one unique WorldGen design."""

    try:
        return custom_games.create(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/custom/minigames/{game_id}", tags=["custom"])
def api_custom_minigame(game_id: str):
    """One manifest, suitable for restoring both creator and world controls."""

    try:
        record = custom_games.load(game_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(
            status_code=404, detail=f"unknown custom minigame {game_id!r}"
        )
    return record


@router.post("/api/run", tags=["run"])
def api_run(request: RunRequest, profile_id: str | None = None):
    """Compile and execute one rollout against the live JAX environment.

    Identity is read either side of the rollout so a run whose contracts moved
    while it was measuring is stored as void rather than reported as a result.
    Persistence never fails the request: the run genuinely happened, and losing
    it because the artifact could not be written would be the worse outcome.
    """
    _require_profile(profile_id)
    spec = None
    try:
        spec = RunSpec(**request.model_dump())
        opened = store.begin()
        result = run(spec)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - a debug console must say why
        # Anything that is not a validation error used to propagate as a bare
        # 500 whose body is the literal string "Internal Server Error" -- no
        # type, no message, no frame. A rollout that dies inside the scan is
        # precisely the failure this console exists to diagnose, and the one
        # failure it said nothing about; two separate policy bugs were only
        # findable by reading the server's stdout.
        import traceback

        frames = traceback.format_exception(type(exc), exc, exc.__traceback__)
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"{type(exc).__name__}: {exc}",
                "spec": request.model_dump(),
                "traceback": "".join(frames[-6:]),
            },
        ) from exc

    from dataclasses import asdict

    try:
        artifact_spec = asdict(spec)
        if profile_id:
            artifact_spec["profile_id"] = profile_id
        result["artifact"] = store.write(artifact_spec, result, opened)
    except Exception as exc:  # noqa: BLE001 - surfaced, never fatal
        result["artifact"] = {"error": f"{type(exc).__name__}: {exc}"}
    if profile_id and not result["artifact"].get("error"):
        try:
            result["artifact"]["profile_replay"] = profiles.bind_run(
                profile_id, result["artifact"], artifact_spec
            )
        except Exception as exc:  # noqa: BLE001 - the artifact still exists
            result["artifact"]["profile_error"] = f"{type(exc).__name__}: {exc}"
    return result


@router.get("/api/runs", tags=["run"])
def api_runs(limit: int = 200, profile_id: str | None = None):
    """Every stored run, newest first."""
    archive_sync = agent_archive.sync()
    rows = store.index(limit=limit)
    bound = profiles.replay_owners()
    for row in rows:
        owner_ids = set(row.get("profile_ids") or [])
        owner_ids.update(bound.get(row["run_id"], []))
        if row.get("profile_id"):
            owner_ids.add(row["profile_id"])
        row["profile_ids"] = sorted(owner_ids)
        if not row.get("profile_id") and len(owner_ids) == 1:
            row["profile_id"] = next(iter(owner_ids))
    if profile_id:
        rows = [row for row in rows if profile_id in row["profile_ids"]]
    return {"runs": rows, "archive_sync": archive_sync}


@router.get("/api/runs/{run_id}", tags=["run"])
def api_run_report(run_id: str):
    """One run's full report -- what it was produced against, not just its numbers."""
    report = store.load(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    return report


@router.get("/api/runs/{run_id}/replay", tags=["run"])
def api_run_replay(run_id: str, blocks: bool = False):
    """A stored run rebuilt into the shape the map draws.

    One route rather than two fetches: the store splits a run into its report
    and its trajectory, and either half alone renders something misleading --
    an empty map, or numbers with no provenance.
    """
    payload = store.replay(run_id, include_blocks=blocks)
    if payload is None:
        raise HTTPException(
            status_code=404, detail=f"no replayable trajectory stored for {run_id!r}"
        )
    owner_ids = set(payload.get("spec", {}).get("profile_ids") or [])
    profile_id = payload.get("spec", {}).get("profile_id")
    if profile_id:
        owner_ids.add(profile_id)
    owner_ids.update(profiles.replay_owners().get(run_id, []))
    payload["profile_ids"] = sorted(owner_ids)
    if len(owner_ids) == 1:
        payload.setdefault("spec", {}).setdefault("profile_id", next(iter(owner_ids)))
    return payload


@router.get("/api/compare", tags=["run"])
def api_compare(left: str, right: str):
    """Whether two runs are comparable at all, and which field says otherwise."""
    result = store.compare(left, right)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


def _training_spec(request: TrainRequest) -> dict:
    spec = request.model_dump(exclude_none=True)
    spec.update(
        {
            name: None
            for name in request.model_fields_set
            if getattr(request, name) is None
        }
    )
    spec["_explicit_fields"] = sorted(request.model_fields_set)
    return spec


@router.post("/api/train/preflight", tags=["train"])
def api_train_preflight(request: TrainRequest):
    """Resolve presets, shapes, source policy and cache identity without JIT."""

    try:
        return jobs.preflight(_training_spec(request))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/train/cache", tags=["train"])
def api_train_cache():
    """Persistent executable/result candidates and measured launch costs."""

    return jobs.cache_index()


@router.post("/api/train", tags=["train"])
def api_train(
    request: TrainRequest, profile_id: str | None = None, queue: bool = False
):
    """Start one training run in the background and return its job immediately.

    Training outlives an HTTP request, so this hands back a job id rather than a
    result. Metrics stream to `metrics.jsonl` inside the run's artifact
    directory as each update completes.
    """
    _require_profile(profile_id)
    try:
        spec = _training_spec(request)
        if queue:
            return jobs.enqueue(spec, profile_id=profile_id)
        return (
            jobs.launch(spec, profile_id=profile_id)
            if profile_id
            else jobs.launch(spec)
        )
    except RuntimeError as exc:
        # Two JAX runs contend for device memory rather than sharing it, so a
        # second concurrent run is still refused HERE. `queue=true` is the way
        # to stack one behind the current run instead of being turned away.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/train/ladder", tags=["train"])
def api_train_ladder():
    """Which rung the objective ladder is on, and how it got there."""
    from console.core.execution import ladder

    return ladder.status()


@router.post("/api/train/ladder/start", tags=["train"])
def api_train_ladder_start(
    request: TrainRequest, restart: bool = False, updates: int = 64
):
    """Run the ladder from its current rung, gated.

    ONE rung is launched, not all of them. The gate has three outcomes and two
    of them are not "the next rung", so the run after this one is not knowable
    until this one is graded -- which is why the console used to queue the whole
    ladder and never promote.

    The body supplies the run's world, loadout and compute; the rung supplies
    its reward settings, its horizon and the objective.
    """
    from console.core.execution import ladder

    try:
        result = ladder.start(_training_spec(request), updates, restart=restart)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"started": result, "ladder": ladder.status()}


@router.post("/api/train/ladder/stop", tags=["train"])
def api_train_ladder_stop():
    """Stop advancing. A run already on the device keeps going."""
    from console.core.execution import ladder

    return ladder.stop()


@router.get("/api/train/curriculum", tags=["train"])
def api_train_curriculum():
    """The pursuit curriculum's rungs and its promotion gate.

    Read from `arena.curriculum.pursuit`, never restated here: a second copy of
    "what gets harder, and when" is how the console and the training loop come
    to disagree about which rung a run was on.
    """
    from arena.curriculum import pursuit

    return pursuit.describe()


@router.get("/api/train/policies", tags=["train"])
def api_train_policies(limit: int = 60):
    """Checkpoints that can be a run's source or its frozen opponent.

    Bounded like the Archive listing: this feeds a picker, and a full scan of
    the archive costs seconds.
    """
    return {"policies": store.policies(limit=limit)}


@router.get("/api/train/queue", tags=["train"])
def api_train_queue():
    """Runs waiting for the device, in the order they will start."""
    active = jobs.active()
    return {
        "queued": jobs.queue_index(),
        "active": active.job_id if active else None,
    }


@router.delete("/api/train/queue", tags=["train"])
def api_train_queue_clear():
    """Drop every waiting run. Does not touch the run already training."""
    return {"removed": jobs.clear_queue(), "queued": jobs.queue_index()}


@router.delete("/api/train/queue/{queued_id}", tags=["train"])
def api_train_dequeue(queued_id: str):
    """Drop one waiting run before it starts."""
    if not jobs.dequeue(queued_id):
        raise HTTPException(
            status_code=404, detail=f"no queued run {queued_id!r} is waiting"
        )
    return {"removed": queued_id, "queued": jobs.queue_index()}


@router.get("/api/jobs", tags=["train"])
def api_jobs():
    """Every training job this process has run, newest first."""
    return {
        "jobs": jobs.index(),
        "active": (jobs.active() or None) and jobs.active().job_id,
    }


@router.get("/api/live", tags=["train"])
def api_live(after: int = -1):
    """Current push-stream backlog for clients that cannot hold SSE open."""

    return live.snapshot(after)


@router.get("/api/live/events", tags=["train"])
async def api_live_events(request: Request, after: int = -1):
    """Server-sent job/metric events; polling remains only a fallback."""

    header = request.headers.get("last-event-id")
    cursor = max(after, int(header)) if header and header.isdigit() else after

    async def stream():
        nonlocal cursor
        while not await request.is_disconnected():
            events = await asyncio.to_thread(live.wait, cursor, 15.0)
            if not events:
                yield ": heartbeat\n\n"
                continue
            for event in events:
                cursor = int(event["sequence"])
                data = json.dumps(event, separators=(",", ":"), default=str)
                yield f"id: {cursor}\nevent: {event['kind']}\ndata: {data}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/jobs/{job_id}", tags=["train"])
def api_job(job_id: str):
    """One job's live status."""
    view = jobs.get(job_id)
    if view is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
    return view


@router.get("/api/jobs/{job_id}/metrics", tags=["train"])
def api_job_metrics(job_id: str, limit: int = 5000):
    """The per-update metric series, read back off disk."""
    return {"job_id": job_id, "metrics": jobs.metrics(job_id, limit=limit)}


@router.get("/api/jobs/{job_id}/replay", tags=["train"])
def api_job_replay(job_id: str):
    """Newest fixed-seed evaluation replay, available while training runs."""

    payload = jobs.replay(job_id)
    if payload is None:
        raise HTTPException(
            status_code=404, detail=f"no evaluation replay for job {job_id!r}"
        )
    return payload


@router.get("/api/jobs/{job_id}/checkpoint", tags=["train"])
def api_job_checkpoint(job_id: str):
    """Download the weights a training job produced.

    A checkpoint is only meaningful next to what it was trained against, so the
    artifact's `report.json` stays the record; this just hands back the payload.
    """
    target, refused = jobs.checkpoint(job_id)
    if refused:
        # A refusal is an answer, not a 404: it means the contracts moved and
        # the weights would not have been interpretable.
        raise HTTPException(status_code=409, detail=refused)
    if target is None:
        raise HTTPException(
            status_code=404, detail=f"no weights stored for job {job_id!r}"
        )
    # One compressed `.npz`, never a directory: `save_policy_checkpoint` writes
    # the whole payload through a single `np.savez_compressed` call. The file on
    # disk has no extension, so the media type is stated rather than sniffed --
    # left to guess, it comes back `text/plain` and a browser renders two
    # megabytes of binary instead of saving it.
    return FileResponse(
        target, filename=f"{job_id}.npz", media_type="application/octet-stream"
    )


@router.post("/api/jobs/{job_id}/cancel", tags=["train"])
def api_job_cancel(job_id: str):
    """Ask a job to stop after its current update; never kills mid-update."""
    if not jobs.cancel(job_id):
        raise HTTPException(status_code=404, detail=f"no cancellable job {job_id!r}")
    return {"job_id": job_id, "cancelled": True}


@router.post("/api/jobs/{job_id}/terminate", tags=["train"])
def api_job_terminate(job_id: str):
    """Kill the run's process now. Nothing is saved; use cancel to stop politely.

    This exists because `cancel` cannot reach a stage that is not yet polling --
    a Region scene build runs for minutes before update 0, and during it a bad
    launch could not be called off at all.
    """
    if not jobs.terminate(job_id):
        raise HTTPException(
            status_code=404, detail=f"no running job {job_id!r} to terminate"
        )
    return {"job_id": job_id, "terminated": True}


@router.delete("/api/runs/{run_id}", tags=["run"])
def api_run_delete(run_id: str):
    """Delete one stored run and its artifacts. Not reversible.

    A run still training is refused: its directory is being written to, and
    removing it underneath the worker would strand a live job. Terminate it
    first, then delete.
    """
    active = jobs.active()
    if active is not None and active.job_id == run_id:
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id!r} is still {active.status}; terminate it first",
        )
    try:
        removed = store.remove(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if removed is None:
        raise HTTPException(status_code=404, detail=f"no stored run {run_id!r}")
    return removed


@router.get("/api/profiles", tags=["profiles"])
def api_profiles():
    """Agent and training profiles from `agents/profiles/`, rescanned per call."""
    return {"profiles": profiles.index(), "directory": str(profiles.PROFILE_DIR)}


@router.post("/api/profiles/{profile_id}", tags=["profiles"])
def api_profile_save(profile_id: str, payload: dict):
    """Write a profile. Overwrites by id, which becomes the file name."""
    try:
        return profiles.save(profile_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/profiles/{profile_id}/replays", tags=["profiles"])
def api_profile_replay(profile_id: str, request: ProfileReplayRequest):
    """Bind a stored replay; its artifact supplies the authoritative timestamp."""

    _require_profile(profile_id)
    report = store.load(request.run_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"unknown run {request.run_id!r}")
    if store.replay(request.run_id) is None:
        raise HTTPException(
            status_code=400,
            detail=f"run {request.run_id!r} has no replayable trajectory",
        )
    try:
        return profiles.bind_replay(
            profile_id,
            scenario=request.scenario,
            run_id=request.run_id,
            label=request.label,
            recorded_at=report["written_at"],
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/zones/{seed}/{component}/preview", tags=["run"])
def api_zone_preview(seed: int, component: int, node_seed: int | None = None):
    """Ground around a zone's spawn, so an arena can be seen before it is run.

    A Region rollout costs minutes. Previewing the terrain first means picking
    an arena is not a two-minute guess, and the same patch shape the map draws
    is reused so the preview cannot disagree with the run.
    """
    from worlds.zones import seed_for, zones as zone_index
    from console.core.worlds import terrain as terrain_module

    match = next((z for z in zone_index(seed) if z.component == component), None)
    if match is None:
        raise HTTPException(
            status_code=404, detail=f"region {seed} has no zone {component}"
        )

    resolved = node_seed if node_seed is not None else seed_for(match)
    spawn = match.position[0]
    if resolved is not None:
        # Centre on the node the loader would actually draw, not on an
        # arbitrary member -- a preview of somewhere else is worse than none.
        import numpy as np

        core = match.core_nodes
        if len(core):
            picked = int(np.random.default_rng(resolved).choice(core))
            index = (
                int(np.flatnonzero(match.nodes == picked)[0])
                if picked in match.nodes
                else 0
            )
            spawn = match.position[index]

    patch = terrain_module.patch(
        seed, [float(spawn[0])], [float(spawn[1])], [float(spawn[2])]
    )
    if patch is None:
        raise HTTPException(status_code=404, detail="no captured ground here")
    return {
        "zone": f"{seed}:{component}",
        "character": match.character(),
        "spawnable": match.spawnable,
        "materials": match.materials,
        "spawn": [float(v) for v in spawn],
        "node_seed": resolved,
        "terrain": patch,
    }


@router.get("/api/logs", tags=["surface"])
def api_logs():
    """Every log the console can find, newest first."""

    from console.core.telemetry import logs

    return logs.index()


@router.get("/api/logs/{category}/{name}", tags=["surface"])
def api_log_tail(category: str, name: str, lines: int = 200):
    """The end of one log, for reading while it is still being written."""

    from console.core.telemetry import logs

    try:
        return logs.tail(f"{category}/{name}", lines=lines)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/api/worlds", tags=["run"])
def api_worlds():
    """World libraries found on disk, and the orders a run may ask for.

    Discovered, not listed: a library appears because its directory exists.
    """

    from worlds.libraries import describe

    return describe()


def _world_inventory(library: str | None, split: str):
    """The shared library, grouped into live / used / unused."""

    from console.core.evidence import world_usage
    from worlds.libraries import inventory

    shared = store.ARTIFACT_ROOT / store.SHARED_DIRECTORY
    roots = (
        [path for path in shared.iterdir() if path.is_dir()]
        if shared.is_dir()
        else []
    )
    return inventory(
        library,
        split,
        live=world_usage.live_seeds(),
        used=world_usage.used_seeds(roots),
    )


@router.get("/api/worlds/inventory", tags=["run"])
def api_worlds_inventory(library: str | None = None, split: str = "train"):
    """Every world in one split, with its terrain measure and keep/deprecate state."""

    return _world_inventory(library, split)


@router.post("/api/worlds/{library_id}/{seed}/status", tags=["run"])
def api_world_status(library_id: str, seed: int, body: WorldStatusRequest):
    """Keep or deprecate one world, or clear the decision.

    Deprecating never edits the library: it records a decision that
    `worlds.libraries.select` reads, so the artifact and its semantic hash --
    which every past result is recorded against -- stay exactly as captured.
    """

    from worlds import status

    if body.state is None:
        status.clear(library_id, seed)
    else:
        listed = _world_inventory(library_id, body.split)
        semantic = next(
            (
                str(row["semantic_sha256"])
                for row in listed["worlds"]
                if int(row["seed"]) == seed
            ),
            "",
        )
        if not semantic:
            raise HTTPException(
                status_code=404,
                detail=f"no world {seed} in {library_id!r} split {body.split!r}",
            )
        try:
            status.set_state(
                library_id, seed, body.state, reason=body.reason,
                semantic_sha256=semantic,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
    return _world_inventory(library_id, body.split)


@router.get("/api/agents", tags=["agents"])
def api_agents():
    """Agents found under `agents/`, with whatever each declared in its
    `console/agent.json`. Rescanned per call."""
    return {"agents": agents.index(), "directory": str(agents.AGENT_ROOT)}


@router.get("/api/devices", tags=["surface"])
def api_devices():
    """Compute: JAX backend, devices, memory, and scene-cache state."""
    return devices.status()


@router.get("/api/compute", tags=["surface"])
def api_compute():
    """Composable training shapes plus the live GPU launcher state."""

    return {
        "schema": "console-compute-composition-v1",
        "training_workflows": training_workflows(),
        "runtime": jobs.compute_status(),
    }


@router.post("/api/devices/clear-cache", tags=["surface"])
def api_clear_scene_cache():
    """Drop cached scene handles. Frees device buffers; next run is cold."""
    from arena.publication_worlds import clear_publication_cache, publication_cache_info
    from console.core.execution.runner import clear_scene_cache
    from console.core.worlds import replay_terrain, terrain, worldgen

    scenes = clear_scene_cache()
    native_replays = terrain.clear_cache()
    replay_worlds = replay_terrain.clear_cache()
    publication = publication_cache_info()
    generated_training = (
        publication["source"]["currsize"] + publication["resident"]["currsize"]
    )
    clear_publication_cache()
    generated_replays = worldgen.clear_capture_cache()
    return {
        "dropped": (
            scenes + native_replays + replay_worlds
            + generated_training + generated_replays
        ),
        "scene_handles": scenes,
        "replay_terrain": native_replays,
        "replay_worlds": replay_worlds,
        "generated_training_worlds": generated_training,
        "generated_replays": generated_replays,
    }


@router.post("/api/reload", tags=["surface"])
def api_reload(payload: dict):
    """Re-import one console module in place, without restarting.

    Panels and profiles already rescan per request; this is for the engine —
    `console.core.execution.runner`, `store`, `terrain` — so an edit takes effect while
    keeping cached scenes and the job registry. Scoped to `console.*` on
    purpose: reloading `adk` or `hytalegym` mid-process would leave live JAX
    handles pointing at functions from the previous module object, which fails
    in ways that look like a model bug rather than a reload.
    """
    import importlib
    import sys
    import traceback

    name = str(payload.get("module") or "")
    if not name.startswith("console."):
        raise HTTPException(
            status_code=400,
            detail=(
                f"refusing to reload {name!r}: only console.* modules. "
                "Reloading adk/hytalegym would strand live JAX handles on "
                "the old module object."
            ),
        )
    module = sys.modules.get(name)
    if module is None:
        raise HTTPException(status_code=404, detail=f"{name!r} is not loaded")
    try:
        importlib.reload(module)
    except Exception:
        # The old module object stays live, so a bad edit degrades to "not
        # applied" rather than taking the server down mid-session.
        raise HTTPException(
            status_code=400, detail=traceback.format_exc(limit=6)
        ) from None
    return {"module": name, "reloaded": True}


@router.get("/api/checks", tags=["checks"])
def api_checks():
    """Every named check and its last result."""
    return checks.index()


@router.post("/api/checks/{check_id}/run", tags=["checks"])
def api_check_run(check_id: str):
    """Launch one check in the background. One at a time."""
    try:
        return checks.launch(check_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/api/panels", tags=["panels"])
def api_panels():
    """Which panels exist right now, rescanned from disk on every call.

    Dynamic by design: an agent that writes `console/panels/mine.py` while the
    server is running gets a panel on the next refresh, with no restart. Files
    that failed to import are reported here too — "my panel did not appear" is
    otherwise unanswerable.
    """
    return panels.specs()


@router.get("/api/panels/{panel_id}", tags=["panels"])
def api_panel(panel_id: str):
    """One panel's data. A panel that raises returns its traceback as a card."""
    payload = panels.render(panel_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"unknown panel {panel_id!r}")
    return payload


@router.get("/api/streams", tags=["streams"])
def api_streams():
    """Every metric stream, with the keys each one declared by logging them."""
    return {"streams": streams.index()}


@router.get("/api/streams/{stream_id}", tags=["streams"])
def api_stream(stream_id: str, limit: int = streams.DEFAULT_LIMIT):
    """One stream as chartable series."""
    try:
        payload = streams.series(stream_id, limit=limit)
    except streams.StreamError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if payload is None:
        raise HTTPException(status_code=404, detail=f"unknown stream {stream_id!r}")
    return payload


@router.post("/api/streams/{stream_id}/log", tags=["streams"])
def api_stream_log(stream_id: str, request: StreamLogRequest):
    """Append one point. The stream is created by its first log line.

    Any learner can report here — PPO, a Dreamer, a transformer, something not
    written yet — and the dashboard charts whatever keys turn up. The console
    deliberately does not decide what a metric is.
    """
    try:
        return streams.log(
            stream_id, step=request.step, metrics=request.metrics, meta=request.meta
        )
    except streams.StreamError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/adk", tags=["surface"])
def api_adk():
    """What the ADK says about itself — scenes, observation, action surface."""
    return adk_surface()


@router.get("/api/bridge", tags=["surface"])
def api_bridge():
    """Native bridge reachability. Read-only; never takes the evidence lease."""
    return bridge_status()


@router.get("/api/policy-agent", tags=["surface"])
def api_policy_agent():
    """Counters from the policy running inside the Java server, if deployed."""
    return policy_agent_status()


@router.get("/api/version", tags=["surface"])
def api_version():
    """Cheap change stamp for hot reload.

    Styles are hot-swapped in place so the page keeps its loaded run and
    camera; a JS or HTML change needs a real reload, so they are reported
    separately rather than reloading on any edit.
    """

    def stamp(suffix: str, *roots: Path) -> int:
        newest = 0.0
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob(f"*{suffix}"):
                newest = max(newest, path.stat().st_mtime)
        return int(newest * 1000)

    return {
        "css": stamp(".css", STATIC / "css", STATIC),
        "js": stamp(".js", STATIC / "js", STATIC),
        "html": stamp(".html", STATIC),
    }


@router.get("/api/hytale", tags=["hytale"])
def api_hytale():
    """The game server: where it is, whether it runs, and what it accepts.

    Serves the command reference alongside the status because the two are read
    together -- knowing a server is up is only useful next to knowing which
    commands it will actually execute.
    """

    return {
        **hytale.status(),
        "install": hytale.install(),
        "console_safe": sorted(hytale.CONSOLE_SAFE),
        "needs_player": sorted(hytale.NEEDS_PLAYER),
    }


@router.get("/api/hytale/log", tags=["hytale"])
def api_hytale_log(after: int = 0, limit: int = 2_000):
    """Incremental stdout and command transcript for the managed server."""

    return hytale.transcript(after=max(0, after), limit=limit)


@router.post("/api/hytale/launch", tags=["hytale"])
def api_hytale_launch(payload: dict):
    """Start a server. The arguments are returned so the run is reproducible."""

    try:
        return hytale.launch(
            payload.get("mode", "client"),
            port=int(payload.get("port", 25565)),
            auth=payload.get("auth") or None,
            boot_commands=tuple(payload.get("boot_commands") or ()),
            mods=tuple(payload.get("mods") or ()),
            owner=payload.get("owner") or None,
        )
    except hytale.ServerError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post("/api/hytale/command", tags=["hytale"])
def api_hytale_command(payload: dict):
    command = (payload.get("command") or "").strip()
    if not command:
        raise HTTPException(status_code=400, detail="empty command")
    try:
        return hytale.send(command)
    except hytale.ServerError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post("/api/hytale/stop", tags=["hytale"])
def api_hytale_stop():
    return hytale.stop()


@router.post("/api/hytale/preview", tags=["hytale"])
def api_hytale_preview(payload: dict):
    """The command line a launch *would* use, without launching it."""

    return {
        "command": hytale.argv(
            payload.get("mode", "client"),
            port=int(payload.get("port", 25565)),
            auth=payload.get("auth") or None,
            boot_commands=tuple(payload.get("boot_commands") or ()),
            mods=tuple(payload.get("mods") or ()),
            owner=payload.get("owner") or None,
            universe=str(hytale.RUNTIME / "universe"),
        )
    }


@router.get("/favicon.ico", include_in_schema=False)
def favicon():
    """A real icon, so the browser console stays empty.

    Chrome requests this unprompted; without it every page load logged a 404.
    One unexplained error in the console trains you to ignore the console,
    which is where a real failure would have shown up.
    """
    from fastapi.responses import Response

    # A 1x1 transparent GIF. Smaller than shipping a binary asset, and the tab
    # icon is not what this tool is for.
    pixel = (
        b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00!\xf9\x04"
        b"\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D"
        b"\x01\x00;"
    )
    return Response(content=pixel, media_type="image/gif")


@router.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
