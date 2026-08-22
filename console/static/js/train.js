/* Training jobs: launch, watch, cancel.
 *
 * A rollout finishes inside its request and draws itself. Training does not, so
 * the job is already running on the server whether or not a browser is
 * watching, and closing the tab must not stop it. SSE carries normal updates;
 * polling is only a reconnect fallback. Everything shown is read back from
 * the server, so a reload rejoins a run in progress instead of losing it.
 *
 * Polling stops the moment the job reaches a terminal state. A console that
 * keeps hitting the server every two seconds forever is a console people close.
 */

import { $, duration, invalidate, isJobLive, setData, state } from "./state.js";
import { plot } from "./charts.js";
import { css, frameCamera } from "./gfx.js";
import { readForm, syncMinigames, syncRewardTerms } from "./launch.js";
import { drawReplayScene } from "./scene.js";
import { showTab } from "./tabs.js";

const POLL_MS = 2000;
let timer = null;
let pushConnected = false;
let workflows = [];
let clock = null;

function syncClock() {
  const active = (state.jobs || []).some(
    (job) => isJobLive(job));
  if (active && !clock) clock = setInterval(() => invalidate("jobs"), 1000);
  if (!active && clock) { clearInterval(clock); clock = null; }
}

export async function loadJobs() {
  try {
    const response = await fetch("/api/jobs");
    if (!response.ok) return;
    const { jobs } = await response.json();
    state.jobs = jobs;
    const live = jobs.find((job) => isJobLive(job));
    const selected = jobs.find((job) => job.job_id === state.activeJobId) || live || jobs[0];
    state.activeJobId = selected?.job_id || "";
    if (selected && state.jobMetricsFor !== selected.job_id) {
      await loadMetrics(selected.job_id);
    }
    if (live) {
      schedule();
    } else {
      stop();
    }
    invalidate("jobs");
  } catch {
    /* Training is optional; a console that cannot reach it still runs. */
  }
}

async function loadMetrics(jobId) {
  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/metrics`);
    if (!response.ok) return;
    const { metrics } = await response.json();
    state.jobMetrics = metrics;
    state.jobMetricsFor = jobId;
  } catch {
    /* leave the previous series up rather than blanking the chart */
  }
}

function schedule() {
  if (timer || pushConnected) return;
  timer = setInterval(loadJobs, POLL_MS);
}

function stop() {
  if (!timer) return;
  clearInterval(timer);
  timer = null;
}

export function setPushConnected(connected) {
  pushConnected = connected;
  if (connected) stop();
  else if ((state.jobs || []).some((job) => isJobLive(job))) schedule();
}

function parseObject(id) {
  const value = JSON.parse($(id).value || "{}");
  if (!value || Array.isArray(value) || typeof value !== "object") {
    throw new Error(`${id} must contain one JSON object`);
  }
  return value;
}

function currentWorkflow() {
  return workflows.find((item) => item.stage === $("trainStage").value);
}

function evaluationUpdates(updates) {
  const workflow = currentWorkflow();
  const preset = workflow?.presets?.[$("trainPreset").value] || {};
  return [...new Set([0, ...(preset.evaluation_updates || []), updates])]
    .filter((value) => value >= 0 && value <= updates).sort((a, b) => a - b);
}

/** The exact request the Train tab would launch, as a plain object.
 *
 * Extracted so the curriculum ladder queues the SAME body the launch
 * button sends. A second reader would drift, and a ladder that trained on
 * different settings than the tab displays is undetectable downstream.
 */
export function trainRequestBody() {
  // Parsed here rather than passed in, so the ladder does not need its own
  // copy of this logic. Throws on malformed JSON; every caller already
  // reports the message.
  const stageOptions = {
    ...parseObject("trainStageOptions"),
    ...readStageOptionFields(),
  };
  const targetOptions = parseObject("trainTargetOptions");
  const world = $("trainWorld").value;
  const zone = world === "region" ? ($("trainZone").value || null) : null;
  const updates = Number($("trainUpdates").value) || 64;
  const body = {
    ...readForm(),
    training_stage: $("trainStage").value,
    training_preset: $("trainPreset").value,
    // Was hardcoded "wsl". The picker beside it existed as a static badge, so
    // there was no way to run on CPU from here even though the schema has
    // accepted `local` all along.
    runner: $("trainRunner").value || "wsl",
    loadout: $("trainLoadout").value,
    opponent: $("trainOpponent").value || null,
    opponent_policy: $("trainOpponentPolicy").value || null,
    world,
    zone,
    // Absent from this payload until now, so the Train tab could only ever
    // launch the single-scenario shape. Both are already in the schema.
    world_count: Number($("trainWorldCount").value) || 1,
    environment_diversity: $("trainDiversity").checked,
    task: $("trainTask").value,
    // This tab's own controls win over the Run drawer's, same as the rest.
    // Without these two the reward a run trains on was whatever the Run tab
    // happened to be set to, and nothing here showed which one that was.
    minigame: $("trainMinigame").value || null,
    // `value || 1` inside Number, not `Number(value) || 1` -- the latter
    // turns an explicit weight of 0 back into 1.
    minigame_weight: Number($("trainMinigameWeight").value || 1),
    seed: Number($("trainSeed").value) || 0,
    updates,
    rollout_steps: Number($("trainRollout").value) || 32,
    update_epochs: Number($("trainEpochs").value) || 2,
    num_envs: Number($("trainEnvs").value) || 32,
    num_minibatches: Number($("trainMinibatches").value) || 2,
    // 0 means "no cap", and the schema floor is 1, so it must go as null.
    episode_ticks: Number($("trainEpisodeTicks").value) || null,
    learning_rate: Number($("trainLr").value),
    gamma: Number($("trainGamma").value),
    gae_lambda: Number($("trainLambda").value),
    clip: Number($("trainClip").value),
    entropy_coef: Number($("trainEntropy").value),
    value_coef: Number($("trainVf").value),
    // `value || d` inside Number, so an explicit 0 survives where it is
    // meaningful. Widths are never 0, but grad norm 0 is a real setting.
    max_gradient_norm: Number($("trainGradNorm").value || 0.5),
    encoder_size: Number($("trainEncoder").value) || 64,
    recurrent_size: Number($("trainRecurrent").value) || 128,
    evaluation_steps: Number($("trainEvaluationSteps").value) || 512,
    evaluation_updates: evaluationUpdates(updates),
    archive_lanes: Number($("trainArchiveLanes").value) || 8,
    target_controller: $("trainTargetController").value,
    objective: $("trainObjective").value,
    /* Absent, not empty: the schema treats a missing opponent_checkpoint as
       "the opponent is a copy of source_policy", which is what a
       single-policy run wants. Sending "" would be a path to nowhere. */
    ...($("trainOpponentCheckpoint").value
      ? { opponent_checkpoint: $("trainOpponentCheckpoint").value } : {}),
    world_artifact_seed: $("trainWorldSeed")?.value
      ? Number($("trainWorldSeed").value) : null,
    target_separation_range: [
      Number($("trainSeparationMin").value),
      Number($("trainSeparationMax").value),
    ],
    minimum_baseline_visible_fraction: Number($("trainMinimumVisibility").value),
    maximum_approximate_kl: Number($("trainMaximumKl").value),
    stage_options: stageOptions,
    target_options: targetOptions,
    ...designFields(),
  };

  // Reward weights are CombatParams overrides, not PPO knobs, so they go in
  // `parameters`. Only the ones actually changed: an override equal to the
  // default still suppresses a named task, which applies its own with
  // `setdefault`. `readForm` already collected the Run tab's, so this tab's
  // own boxes replace them wholesale rather than merging -- same rule as
  // every other control here.
  const rewards = {};
  for (const el of $("trainRewardTerms").querySelectorAll("input[data-reward]")) {
    const value = Number(el.value);
    if (el.value !== "" && value !== Number(el.dataset.reward)) {
      rewards[el.name] = value;
    }
  }
  if (Object.keys(rewards).length) body.parameters = rewards;
  else delete body.parameters;
  return body;
}

export function wireTrain() {
  const go = $("trainGo");
  if (!go) return;

  // `evade` and a scripted target are an invalid pair the schema refuses, so
  // both controls re-check the combination rather than letting a launch 400.
  $("trainObjective")?.addEventListener("change", syncObjective);
  $("trainTargetController")?.addEventListener("change", syncObjective);
  $("trainRunner")?.addEventListener("change", syncObjective);

  go.addEventListener("click", async (event) => {
    event.preventDefault();
    // Reuse the rollout's own form reader rather than picking fields by id.
    // It already covers the drawer's combat parameters, and two readers would
    // drift -- training against a different world than the rollout you used to
    // inspect it is a mismatch nothing downstream can detect.
    // Start from the rollout form so the drawer's combat parameters come
    // along, then let this tab's own controls win. Before, Train had no
    // environment or goal controls at all and silently inherited whatever the
    // Run tab happened to be set to -- so "train on a different world" was not
    // expressible from here.
    let body;
    try {
      // The typed fields win over the JSON escape hatch, so the value you can
      // see beats one buried in a textarea. The textarea stays for keys the
      // contract does not publish as scalars.
      body = trainRequestBody();
    } catch (error) {
      $("trainStatus").textContent = error.message;
      return;
    }
    $("trainStatus").textContent = "validating contracts and resolving cache identity…";
    try {
      const checked = await fetch("/api/train/preflight", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      const preflight = await checked.json();
      if (!checked.ok) throw new Error(preflight.detail || checked.statusText);
      const exact = preflight.cache.history?.exact_result;
      const cacheState = exact
        ? `exact prior result ${exact.run_id}`
        : preflight.cache.persistent_candidate_hit ? "warm executable candidate" : "cold key";
      $("trainStatus").textContent = `preflight passed · ${cacheState} · launching…`;
      const profileId = $("profilePick")?.value || "";
      const profileQuery = profileId
        ? `?profile_id=${encodeURIComponent(profileId)}` : "";
      const response = await fetch(`/api/train${profileQuery}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || response.statusText);
      $("trainStatus").textContent = `launched ${payload.job_id} · ${payload.spec?.estimated_environment_steps?.toLocaleString() || "?"} JAX steps`;
      state.jobMetrics = [];
      state.activeJobId = payload.job_id;
      clearJobReplay();
      await loadJobs();
      schedule();
    } catch (error) {
      // A refused launch is the expected answer while another job holds the
      // device, not a failure -- say which, so it does not read as a crash.
      $("trainStatus").textContent = error.message;
    }
  });

  const cancel = $("trainCancel");
  cancel?.addEventListener("click", async (event) => {
    event.preventDefault();
    const live = (state.jobs || []).find(
      (job) => isJobLive(job));
    if (!live) { $("trainStatus").textContent = "nothing running"; return; }
    await fetch(`/api/jobs/${encodeURIComponent(live.job_id)}/cancel`, { method: "POST" });
    $("trainStatus").textContent = "cancelling after this update…";
    loadJobs();
  });

  $("trainReplayScrub")?.addEventListener("input", (event) => {
    state.jobReplayStep = Number(event.target.value) || 0;
    drawTrainingReplay(selectedJob(state.jobs || []));
  });
  $("trainReplayOpen")?.addEventListener("click", openTrainingReplay);
}

export function drawJobs() {
  const jobs = state.jobs || [];
  const rows = $("jobRows");
  const evidenceRows = $("evidenceJobRows");
  if (!rows && !evidenceRows) return;
  syncClock();
  const selected = selectedJob(jobs);

  const rowsHtml = jobs.length
    ? jobs.map((job) => {
        const pct = job.updates ? Math.round((job.update / job.updates) * 100) : 0;
        const tone = job.status === "failed" ? "crit"
          : job.status === "finished" ? "ok" : "warn";
        const stamp = new Date(job.started_unix_seconds * 1000);
        const elapsed = isJobLive(job)
          ? Date.now() / 1000 - job.started_unix_seconds
          : job.elapsed_seconds;
        const prior = job.cache?.history?.latest?.timings || {};
        const cache = job.cache?.first_update_seconds
          ? `first update ${Number(job.cache.first_update_seconds).toFixed(1)}s`
          : job.cache?.persistent_path ? "persistent cache bound"
          : prior.first_update_seconds
            ? `warm candidate · prior first update ${Number(prior.first_update_seconds).toFixed(1)}s`
            : "cold candidate";
        const chosen = job.job_id === selected?.job_id ? " selected-job" : "";
        // The launch spec names the batch `batch`; `num_envs` is only set on
        // the in-process path, so reading it alone rendered "? envs" for every
        // WSL run. Worlds matter just as much: a run across 16 varied worlds
        // and one that repeats a single scenario 256 times are different
        // experiments and used to look identical here.
        const agent = job.spec?.shared_agent || job.profile_id;
        const envs = job.spec?.num_envs ?? job.spec?.batch;
        const worlds = job.spec?.world_count ?? 1;
        const scene = [
          job.spec?.world,
          worlds > 1 ? `${worlds} worlds` : null,
          job.spec?.environment_diversity ? "diverse spawns" : null,
        ].filter(Boolean).map(esc).join(" · ");
        return `<div class="runrow${chosen}">
          <code class="job-id" title="${esc(job.job_id)}">${esc(job.job_id)}</code>
          <span class="grow"><b>${esc(job.spec?.training_stage?.replaceAll("_", " ") || "training")}</b> ·
            ${esc(job.spec?.loadout || "?")} vs ${esc(job.spec?.opponent || "?")} · seed ${job.spec?.seed ?? "?"} ·
            ${job.update}/${job.updates} updates · ${envs ?? "?"} envs${scene ? ` · ${scene}` : ""}</span>
          <em class="muted" title="${stamp.toISOString()}">${stamp.toLocaleString()} · ${duration(elapsed)} · ${cache}</em>
          ${agent ? `<i class="k a">${esc(agent)}</i>` : ""}
          <i class="k ${tone}">${job.status}${job.updates ? ` ${pct}%` : ""}</i>
          ${job.stop_reason ? `<i class="k warn" title="a training guardrail ended this run early">${esc(job.stop_reason)}</i>` : ""}
          ${job.latest_replay ? `<i class="k ok">replay u${job.latest_replay.update}</i>` : ""}
          ${ckpt(job)}
          ${["finished", "stopped"].includes(job.status) ? `<a class="k a" href="/?replay=${encodeURIComponent(job.job_id)}">replay</a>` : ""}
          <button type="button" class="ghost mini" data-job-detail="${esc(job.job_id)}">details</button>
          ${isJobLive(job)
            ? `<button type="button" class="ghost mini" data-job-stop="${esc(job.job_id)}"
                 title="Kill the process now. Nothing is saved — use cancel to stop after the current update.">stop</button>`
            : `<button type="button" class="ghost mini" data-job-delete="${esc(job.job_id)}"
                 title="Delete this run and its artifacts. Not reversible.">delete</button>`}
        </div>` + (job.error || job.profile_error
          ? `<div class="diffrow"><span class="crit">${esc(job.error || job.profile_error)}</span></div>` : "");
      }).join("")
    : `<p class="muted">No training jobs yet. A run starts on a background
       worker and streams one metrics row per update; closing this tab does not
       stop it.</p>`;
  if (rows) rows.innerHTML = rowsHtml;
  if (evidenceRows) evidenceRows.innerHTML = rowsHtml;
  for (const root of [rows, evidenceRows].filter(Boolean)) {
    root.querySelectorAll("[data-job-detail]").forEach((button) => {
      button.addEventListener("click", () => selectJob(button.dataset.jobDetail));
    });
    /* Both destructive, so both confirm. `terminate` kills the process without
       saving; `cancel` is the polite stop and stays on the run card, because
       the common case for this button is "that launch is wrong, kill it". */
    root.querySelectorAll("[data-job-stop]").forEach((button) => {
      button.addEventListener("click", async () => {
        const id = button.dataset.jobStop;
        if (!confirm(`Kill ${id} now? Nothing is saved.`)) return;
        button.disabled = true;
        await fetch(`/api/jobs/${encodeURIComponent(id)}/terminate`, { method: "POST" });
        loadJobs();
      });
    });
    root.querySelectorAll("[data-job-delete]").forEach((button) => {
      button.addEventListener("click", async () => {
        const id = button.dataset.jobDelete;
        if (!confirm(`Delete ${id} and its artifacts? This cannot be undone.`)) return;
        button.disabled = true;
        const response = await fetch(`/api/runs/${encodeURIComponent(id)}`,
          { method: "DELETE" });
        if (!response.ok) {
          button.disabled = false;
          alert((await response.json()).detail || "delete failed");
          return;
        }
        loadJobs();
        // The Archive is a separate surface reading a separate endpoint, so a
        // deleted run would linger there until a manual reload.
        (await import("./runs.js")).loadRuns();
      });
    });
  }
  const evidenceCount = $("evidenceJobCount");
  if (evidenceCount) {
    const active = jobs.filter(
      (job) => isJobLive(job),
    ).length;
    evidenceCount.textContent = `${jobs.length} job${jobs.length === 1 ? "" : "s"} · ${active} active`;
  }

  const metrics = state.jobMetrics || [];
  const note = $("jobNote");
  if (note) {
    note.textContent = metrics.length
      ? `${metrics.length} update${metrics.length > 1 ? "s" : ""} recorded`
      : jobs.length ? "waiting for the first update" : "";
  }
  if ($("evidenceMetricCount")) $("evidenceMetricCount").textContent = selected
    ? `${metrics.length} metric row${metrics.length === 1 ? "" : "s"}` : "no job selected";
  if (!selected && $("evidenceJobDetails")) $("evidenceJobDetails").open = false;

  drawMetricCharts("train", metrics);
  drawMetricCharts("evidence", metrics);

  const lastEval = [...metrics].reverse().find((row) => row["learner.aligned_fraction"] !== undefined) || {};
  const elapsed = selected ? (isJobLive(selected)
    ? Date.now() / 1000 - selected.started_unix_seconds : selected.elapsed_seconds) : 0;
  const facts = selected ? [
    ["job", selected.job_id], ["profile", selected.profile_id || "unassigned"],
    ["state", selected.status], ["stage", selected.spec?.training_stage?.replaceAll("_", " ") || "training"],
    ["progress", `${selected.update}/${selected.updates}`],
    ["started", new Date(selected.started_unix_seconds * 1000).toLocaleString()],
    ["elapsed", duration(elapsed)],
    // Only when the run actually has a matchup. It was unconditional, so a
    // stage without a loadout showed "? vs ?" as though something were missing.
    ...(selected.spec?.loadout
      ? [["matchup", `${selected.spec.loadout} vs ${selected.spec.opponent || "none"}`]]
      : []),
    ["throughput", selected.cache?.steps_per_second ? `${Number(selected.cache.steps_per_second).toFixed(0)} steps/s` : "pending"],
    // Reward/behaviour columns come from what the agent DECLARES and what the
    // run actually emitted. They used to be two hardcoded keys, aim error and
    // distance, which are pursuit's -- a world-model or PPO run showed those
    // two as "pending" forever while its own declared metrics went unshown.
    ...declaredFacts(selected, lastEval),
  ].map(([label, value]) => `<span><small>${esc(label)}</small><b title="${esc(value)}">${esc(value)}</b></span>`).join("") : "";
  if ($("trainLiveFacts")) $("trainLiveFacts").innerHTML = facts;
  if ($("evidenceJobFacts")) $("evidenceJobFacts").innerHTML = facts;
  syncJobReplay(selected);
  drawTrainingReplay(selected);
}

function drawMetricCharts(prefix, metrics) {
  const aligned = metricValues(metrics, "learner.aligned_fraction");
  const resetAligned = metricValues(metrics, "memory_reset.aligned_fraction");
  plot(`${prefix}BehaviorChart`, [
    { values: aligned, color: css("--agent"), dots: true },
    { values: resetAligned, color: css("--target"), dots: true },
  ].filter((series) => series.values.length), { min: 0, max: 1, empty: "waiting for evaluation" });

  const aim = metricValues(metrics, "learner.mean_visible_aim_error_degrees")
    .map((value) => 1 - Math.min(180, Math.max(0, value)) / 180);
  const proximity = metricValues(metrics, "learner.mean_distance")
    .map((value) => 1 - Math.min(24, Math.max(0, value)) / 24);
  plot(`${prefix}GeometryChart`, [
    { values: aim, color: css("--ok"), dots: true },
    { values: proximity, color: css("--info"), dots: true },
  ].filter((series) => series.values.length), { min: 0, max: 1, empty: "waiting for geometry metrics" });

  const loss = firstMetric(metrics, ["loss.total_loss", "total_loss"]);
  const throughput = firstMetric(metrics, ["steps_per_second"]);
  plot(`${prefix}OptimizerChart`, [
    { values: normalize(loss), color: css("--accent") },
    { values: normalize(throughput), color: css("--info") },
  ].filter((series) => series.values.length), { min: 0, max: 1, empty: "waiting for first GPU update" });

}

function selectedJob(jobs) {
  return jobs.find((job) => job.job_id === state.activeJobId)
    || jobs.find((job) => isJobLive(job))
    || jobs[0];
}

async function selectJob(jobId) {
  if (!jobId) return;
  if ($("evidenceJobDetails")) $("evidenceJobDetails").open = true;
  if (jobId === state.activeJobId) return;
  state.activeJobId = jobId;
  clearJobReplay();
  await loadMetrics(jobId);
  invalidate("jobs");
}

function clearJobReplay() {
  state.jobReplay = null;
  state.jobReplayFor = null;
  state.jobReplayLoading = null;
  state.jobReplayStep = 0;
}

async function syncJobReplay(job) {
  const replay = job?.latest_replay;
  if (!job || !replay) return;
  const token = `${job.job_id}:${replay.update}:${replay.lane}`;
  if (state.jobReplayFor === token || state.jobReplayLoading === token) return;
  state.jobReplayLoading = token;
  try {
    let response = await fetch(`/api/jobs/${encodeURIComponent(job.job_id)}/replay`);
    let payload = response.ok ? await response.json() : null;
    if (!payload && response.status === 404) {
      response = await fetch(`/api/jobs/${encodeURIComponent(job.job_id)}`);
      payload = response.ok ? (await response.json()).latest_replay_payload : null;
    }
    if (!payload) {
      state.jobReplayFor = token;
      state.jobReplay = null;
      return;
    }
    if (state.activeJobId !== job.job_id) return;
    state.jobReplay = payload;
    state.jobReplayFor = token;
    state.jobReplayStep = Math.max(0, payload.steps - 1);
  } finally {
    if (state.jobReplayLoading === token) state.jobReplayLoading = null;
    invalidate("jobs");
  }
}

function drawTrainingReplay(job) {
  const payload = state.jobReplay;
  const matches = payload && state.jobReplayFor?.startsWith(`${job?.job_id}:`);
  drawReplayScene(
    "trainReplayView", matches ? payload : null, state.jobReplayStep,
    { follow: true },
  );
  const scrub = $("trainReplayScrub");
  if (scrub) {
    scrub.max = matches ? Math.max(0, payload.steps - 1) : 0;
    scrub.value = matches ? state.jobReplayStep : 0;
    scrub.disabled = !matches;
  }
  if ($("trainReplayOpen")) $("trainReplayOpen").disabled = !matches;
  if ($("trainReplayStep")) $("trainReplayStep").textContent = matches
    ? `step ${state.jobReplayStep}/${payload.steps - 1}` : "—";
  if ($("trainReplayNote")) $("trainReplayNote").textContent = matches
    ? `update ${payload.evaluation?.update} · lane ${payload.evaluation?.lane} · ${new Date(payload.evaluation?.recorded_unix_seconds * 1000).toLocaleString()}`
    : job ? "waiting for its first evaluation milestone" : "no job selected";
}

function openTrainingReplay() {
  if (!state.jobReplay) return;
  frameCamera(state.jobReplay.trajectory, state.jobReplay.terrain);
  setData(state.jobReplay);
  showTab("run");
}

function metricValues(rows, name) {
  return rows.map((row) => row[name]).filter((value) => Number.isFinite(value));
}

function firstMetric(rows, names) {
  for (const name of names) {
    const values = metricValues(rows, name);
    if (values.length) return values;
  }
  return [];
}

function normalize(values) {
  if (!values.length) return [];
  const lo = Math.min(...values), hi = Math.max(...values);
  if (hi - lo < 1e-12) return values.map(() => 0.5);
  return values.map((value) => (value - lo) / (hi - lo));
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]
  ));
}

/* Weights are the point of a training run, so their absence is stated rather
   than left blank -- a refused checkpoint means the contracts moved while the
   run was training, and the run is not reusable. */
function ckpt(job) {
  if (!job.checkpoint) return job.update ? `<i class="k warn">no weights</i>` : "";
  if (String(job.checkpoint).startsWith("refused:")) {
    return `<i class="k crit" title="${job.checkpoint}">weights refused</i>`;
  }
  return `<a class="k ok" href="/api/jobs/${encodeURIComponent(job.job_id)}/checkpoint">weights</a>`;
}

function renderWorkflow() {
  const workflow = currentWorkflow();
  if (!workflow) return;
  const launchable = workflow.launchable !== false;
  const go = $("trainGo");
  if (go) go.disabled = !launchable;
  if (!launchable) {
    $("trainStatus").textContent = `registered contract · collector binding needed: ${(workflow.blockers || []).join("; ")}`;
  } else if ($("trainStatus").textContent.startsWith("registered contract")) {
    $("trainStatus").textContent = "ready";
  }
  const editor = $("trainStageOptions");
  const contractId = workflow.contract?.contract_id || "";
  if (editor && editor.dataset.contractId !== contractId) {
    editor.dataset.contractId = contractId;
    editor.value = contractId && !launchable
      ? JSON.stringify(workflow.contract.defaults || {}, null, 2) : "{}";
  }
  // Reward and shaping terms as real fields, built from the stage contract's
  // own dataclass defaults. Tuning these used to mean hand-writing JSON into
  // the textarea below with no list of valid keys and no defaults shown, so a
  // mistyped name was silently dropped instead of refused.
  renderStageOptionFields(workflow);
  const presetSelect = $("trainPreset");
  const current = presetSelect.value;
  presetSelect.innerHTML = Object.keys(workflow.presets || {})
    .map((name) => `<option value="${name}">${name}</option>`).join("");
  presetSelect.value = (workflow.presets || {})[current]
    ? current : workflow.default_preset;
  const worldSeed = $("trainWorldSeed");
  if (worldSeed) {
    const selected = worldSeed.value;
    const artifacts = workflow.world_backend?.artifacts || [];
    worldSeed.innerHTML = [`<option value="">workflow default</option>`]
      .concat(artifacts.map((item) =>
        `<option value="${item.seed}">${item.seed}</option>`)).join("");
    if ([...worldSeed.options].some((item) => item.value === selected)) {
      worldSeed.value = selected;
    }
    worldSeed.disabled = !artifacts.length;
  }
  applyPreset();
}

/** Apply a Compute composition to the canonical Train form without launching. */
export function applyTrainingComposition(composition) {
  const workflow = workflows.find((item) => item.stage === composition.stage);
  if (!workflow) throw new Error(`training stage ${composition.stage} is unavailable`);
  $("trainStage").value = workflow.stage;
  renderWorkflow();
  if ((workflow.presets || {})[composition.preset]) {
    $("trainPreset").value = composition.preset;
    applyPreset(true);
  }
  if (composition.num_envs) $("trainEnvs").value = composition.num_envs;
  if (composition.rollout_steps) $("trainRollout").value = composition.rollout_steps;
  if (composition.target_controller) {
    $("trainTargetController").value = composition.target_controller;
  }
  if ($("trainWorldSeed")) {
    $("trainWorldSeed").value = composition.world_artifact_seed ?? "";
  }
  applyPreset(false);
  showTab("train");
}

/* Agents declare the metrics that matter to them in their own console
   manifest, so the Evidence facts can follow the run instead of naming
   pursuit's two columns for every stage. Fetched once; an agent list that
   cannot be read simply leaves the fallback in charge. */
let agentMetrics = new Map();

export async function loadAgentMetrics() {
  try {
    const response = await fetch("/api/agents");
    if (!response.ok) return;
    const payload = await response.json();
    const list = payload.agents || payload;
    agentMetrics = new Map(
      (Array.isArray(list) ? list : []).map((a) => [a.id, a.metrics || []]));
  } catch {
    /* leave whatever was loaded before */
  }
}

/** Format one metric for display without knowing what it means. */
function metricFact(name, value) {
  const label = name.replace(/^learner\./, "").replace(/_/g, " ")
    .replace(/ degrees$/, "");
  if (value === undefined || value === null) return [label, "pending"];
  const number = Number(value);
  if (!Number.isFinite(number)) return [label, String(value)];
  const suffix = name.endsWith("_degrees") ? "°" : "";
  const text = Math.abs(number) >= 1000 || (number !== 0 && Math.abs(number) < 0.01)
    ? number.toExponential(2)
    : number.toFixed(2);
  return [label, `${text}${suffix}`];
}

/** Metric facts for a run: what its agent declares, else what it emitted. */
function declaredFacts(job, lastEval) {
  const owner = job.spec?.shared_agent || job.profile_id;
  let names = (agentMetrics.get(owner) || []).filter((n) => n in lastEval);
  if (!names.length) {
    // No declaration, or none of the declared names present: show what this run
    // actually produced rather than nothing.
    names = Object.keys(lastEval)
      .filter((n) => n.startsWith("learner.") && typeof lastEval[n] === "number")
      .slice(0, 3);
  }
  return names.slice(0, 4).map((name) => metricFact(name, lastEval[name]));
}

/** Stage reward/shaping terms, rendered from the contract's own defaults. */
function renderStageOptionFields(workflow) {
  const host = $("trainStageOptionFields");
  if (!host) return;
  const fields = workflow?.custom?.stage_option_fields || [];
  if (!fields.length) {
    host.innerHTML = `<p class="note">this stage publishes no editable terms;
      use the JSON field below</p>`;
    return;
  }
  host.innerHTML = fields.map((field) => field.type === "bool"
    ? `<label title="${field.name}"><span>${field.name.replace(/_/g, " ")}</span>
         <input type="checkbox" data-stage-option="${field.name}"
                data-default="${field.default}"${field.default ? " checked" : ""}></label>`
    : `<label title="${field.name}"><span>${field.name.replace(/_/g, " ")}</span>
         <input type="number" step="any" data-stage-option="${field.name}"
                data-default="${field.default}" value="${field.default}"></label>`
  ).join("");
}

/** Only terms the operator actually moved, so a default never suppresses one. */
function readStageOptionFields() {
  const host = $("trainStageOptionFields");
  const moved = {};
  if (!host) return moved;
  for (const el of host.querySelectorAll("[data-stage-option]")) {
    const name = el.dataset.stageOption;
    if (el.type === "checkbox") {
      const base = el.dataset.default === "true" || el.dataset.default === "True";
      if (el.checked !== base) moved[name] = el.checked;
      continue;
    }
    if (el.value === "") continue;
    if (Number(el.value) !== Number(el.dataset.default)) moved[name] = Number(el.value);
  }
  return moved;
}

/* Measured on `open_flat` at batch 1, 64 ticks, against a zero-motion control:
 * these are what the gait head actually produces, not a speed table read from
 * the ruleset. `SKILL_RETREAT` -- the weave evader's whole movement -- travels
 * at exactly the run figure, which is why it has no slower setting. */
const GAIT_SPEED = { walk: 0.063, run: 0.194, sprint: 0.238 };

/* Only send what the caller actually set. An omitted `world_design` is what
 * keeps a launch on the captured Region path, so defaulting it here would
 * silently move every existing run onto a plane. */
function designFields() {
  const design = $("trainWorldDesign")?.value || "";
  if (!design) return {};
  const mix = [
    Number($("trainEvaderWalk").value),
    Number($("trainEvaderRun").value),
    Number($("trainEvaderSprint").value),
  ];
  const fields = {
    world_design: design,
    arena_radius: Number($("trainArenaRadius").value),
    evader_gait_mix: mix,
    evader_gait_period_ticks: Number($("trainEvaderPeriod").value),
  };
  /* Sent only for the controllers that read them, so a fleeing-evader launch
   * spec does not carry a band that describes nothing. */
  if (($("trainTargetController")?.value || "").includes("standoff")) {
    fields.evader_standoff_band = [
      Number($("trainStandoffInner").value),
      Number($("trainStandoffOuter").value),
    ];
    fields.evader_standoff_strafe_ticks = Number($("trainStandoffTicks").value);
  }
  return fields;
}

/* The mix has to sum to one or the launch is refused, and the speed it implies
 * decides whether the chase is winnable at all -- so both are shown before the
 * request is sent rather than surfacing as a 400. */
function refreshEvaderNote() {
  const note = $("trainEvaderNote");
  if (!note) return;
  if (!$("trainWorldDesign")?.value) {
    note.textContent = "Region world — evader gait controls apply to design worlds only.";
    return;
  }
  const walk = Number($("trainEvaderWalk").value);
  const run = Number($("trainEvaderRun").value);
  const sprint = Number($("trainEvaderSprint").value);
  const total = walk + run + sprint;
  const speed = walk * GAIT_SPEED.walk + run * GAIT_SPEED.run + sprint * GAIT_SPEED.sprint;
  const share = (100 * speed) / GAIT_SPEED.sprint;
  note.textContent = Math.abs(total - 1) > 1e-6
    ? `gait mix sums to ${total.toFixed(2)} — must be 1.00, the launch will be refused`
    : `evader averages ${speed.toFixed(3)} blocks/tick `
      + `(${share.toFixed(0)}% of the agent's ${GAIT_SPEED.sprint} sprint; `
      + `the weave evader is a flat ${GAIT_SPEED.run})`;
}

async function loadDesigns() {
  const select = $("trainWorldDesign");
  if (!select) return;
  try {
    const rows = await (await fetch("/api/worldgen/designs")).json();
    const designs = rows.designs || rows || [];
    for (const design of designs) {
      const option = document.createElement("option");
      option.value = design.design_id;
      option.textContent = `${design.design_id} (seed ${design.seed})`;
      select.append(option);
    }
  } catch {
    // A console without the worldgen store still launches Region runs; the
    // empty list is the honest state, not an error worth blocking the tab.
  }
  for (const id of ["trainWorldDesign", "trainEvaderWalk", "trainEvaderRun",
                    "trainEvaderSprint"]) {
    $(id)?.addEventListener("input", refreshEvaderNote);
    $(id)?.addEventListener("change", refreshEvaderNote);
  }
  refreshEvaderNote();
}

function applyPreset(setValues = true) {
  const workflow = currentWorkflow();
  const preset = workflow?.presets?.[$("trainPreset").value] || {};
  const fields = {
    batch: "trainEnvs", updates: "trainUpdates", rollout_steps: "trainRollout",
    evaluation_steps: "trainEvaluationSteps", update_epochs: "trainEpochs",
    num_minibatches: "trainMinibatches", archive_lanes: "trainArchiveLanes",
  };
  for (const [name, id] of Object.entries(fields)) {
    if (setValues && preset[name] !== undefined) $(id).value = preset[name];
  }
  const pursuit = workflow?.stage === "pursuit_tracking";
  if (pursuit) {
    // A design world is not a Region. Pinning it here would contradict the
    // design the caller selected two fields above, and the backend would then
    // silently take the Region path.
    $("trainWorld").value = $("trainWorldDesign")?.value ? "open_flat" : "region";
    $("trainEpisodeTicks").value = $("trainEvaluationSteps").value;
    const fixed = workflow?.custom?.fixed || {};
    if (setValues && fixed.target_mode) {
      $("trainTargetController").value = fixed.target_mode;
    }
    if (setValues && fixed.target_separation_range?.length === 2) {
      $("trainSeparationMin").value = fixed.target_separation_range[0];
      $("trainSeparationMax").value = fixed.target_separation_range[1];
    }
    if (setValues && fixed.minimum_baseline_visible_fraction !== undefined) {
      $("trainMinimumVisibility").value = fixed.minimum_baseline_visible_fraction;
    }
    if (setValues && fixed.maximum_approximate_kl !== undefined) {
      $("trainMaximumKl").value = fixed.maximum_approximate_kl;
    }
  }
  const steps = Number($("trainEnvs").value)
    * Number($("trainUpdates").value) * Number($("trainRollout").value);
  const contract = workflow?.contract;
  const actions = contract?.action_heads?.join(" + ")
    || (pursuit ? "move + jump + yaw + pitch" : "environment contract");
  const readiness = workflow?.launchable === false ? "binding needed" : "ready";
  $("trainContractSummary").innerHTML = [
    ["strategy", workflow?.strategy || "—"],
    ["device", pursuit ? "WSL CUDA" : "selected runner"],
    ["JAX steps", Number.isFinite(steps) ? steps.toLocaleString() : "—"],
    ["horizon", `${$("trainEvaluationSteps").value} ticks`],
    ["actions", actions],
    ["contract", contract ? `${readiness} · ${contract.contract_sha256.slice(0, 10)}` : "custom"],
  ].map(([label, value]) => `<span><small>${label}</small><b title="${value}">${value}</b></span>`).join("");
}


/* Populate the Train tab's own environment controls.
 *
 * Separate from the Run form on purpose: you train on one scene and then
 * inspect a rollout on another, and having one set of controls serve both
 * meant changing the inspection changed the next training run.
 */
/* Agent vs agent. The API has always accepted `objective` and
   `opponent_checkpoint` -- the self-play driver posts both -- but the Train tab
   offered neither, so from the browser you could pick "frozen source policy"
   without saying WHICH policy, and could not train the evader side at all. */
async function loadOpponentPolicies() {
  const select = $("trainOpponentCheckpoint");
  if (!select) return;
  let rows = [];
  try {
    rows = (await (await fetch("/api/train/policies")).json()).policies || [];
  } catch { /* a picker that cannot load still leaves the default usable */ }
  const options = [`<option value="">copy of source policy</option>`];
  for (const row of rows) {
    for (const update of row.updates) {
      const path = `${row.directory}/update-${String(update).padStart(4, "0")}.npz`;
      options.push(
        `<option value="${esc(path)}">${esc(row.run_id)} · u${update}</option>`
      );
    }
  }
  select.innerHTML = options.join("");
  syncObjective();
}

/* `evade` requires a policy-driven opponent: the schema refuses the pair
   outright (`objective 'evade' needs target_controller frozen_source_policy`),
   and a scripted flee target cannot be the other half of a recursive pair. Fix
   the controller here rather than letting the launch 400. */
function syncObjective() {
  const objective = $("trainObjective")?.value;
  const controller = $("trainTargetController");
  const note = $("trainObjectiveNote");
  if (!objective || !controller) return;
  const frozen = controller.value === "frozen_source_policy";
  if (objective === "evade" && !frozen) controller.value = "frozen_source_policy";
  if (note) {
    note.textContent = objective === "evade"
      ? "The learner is the EVADER: paid for surviving and holding range, not "
        + "for catching. Needs a frozen policy opponent, which is set for you."
      : "The learner is the CHASER: paid for catching. A scripted target is a "
        + "fixed curriculum; a frozen policy is agent vs agent.";
  }
  // Hidden rather than disabled: a checkpoint picker is meaningless against a
  // scripted controller, and a greyed control still reads as "something I am
  // failing to set".
  $("trainOpponentField")?.toggleAttribute(
    "hidden", controller.value !== "frozen_source_policy"
  );
  const compute = $("trainComputeNote");
  if (compute) {
    // Tied to the run, not a standalone fact: what matters at launch is
    // whether THIS run gets CUDA, and what it costs when it does not.
    compute.textContent = $("trainRunner")?.value === "wsl"
      ? "this run gets CUDA"
      : "CPU only — measured ~2x slower, and a Region scene build dominates";
  }
}

export async function loadTrainOptions() {
  let o;
  try {
    o = await (await fetch("/api/options")).json();
  } catch { return; }

  workflows = o.training_workflows || [];
  $("trainStage").innerHTML = workflows
    .map((item) => {
      const label = item.title || item.stage.replaceAll("_", " ");
      const readiness = item.launchable === false ? " · binding needed" : "";
      return `<option value="${item.stage}">${label}${readiness}</option>`;
    })
    .join("");
  $("trainStage").value = workflows.some((item) => item.stage === "pursuit_tracking")
    ? "pursuit_tracking" : workflows[0]?.stage;

  const fill = (id, values, blank) => {
    const el = $(id);
    if (!el) return;
    el.innerHTML = (blank ? [`<option value="">none</option>`] : [])
      .concat((values || []).map((v) => `<option>${v}</option>`)).join("");
  };
  fill("trainLoadout", o.loadouts);
  fill("trainOpponent", o.loadouts, true);
  fill("trainOpponentPolicy", o.opponent_policies);
  fill("trainWorld", o.worlds);
  fill("trainTask", o.tasks);
  syncMinigames(o.minigame_details || [], "trainMinigame", null);
  syncRewardTerms(o.reward_terms || [], "trainRewardTerms");
  await loadOpponentPolicies();

  const zones = o.zones || [];
  const zoneSelect = $("trainZone");
  if (zoneSelect) {
    zoneSelect.innerHTML = [`<option value="">library default spawn</option>`,
      `<option value="rotate">rotate by seed — a different zone per seed</option>`]
      .concat(zones.map((z) => `<option value="${z.id}">${z.label}</option>`))
      .join("");
  }

  const d = o.defaults || {};
  if (d.loadout) $("trainLoadout").value = d.loadout;
  if (d.opponent) $("trainOpponent").value = d.opponent;
  $("trainOpponentPolicy").value = d.opponent_policy || "first_legal";
  if (d.world) $("trainWorld").value = d.world;
  if (d.task) $("trainTask").value = d.task;

  const syncZone = () => {
    $("trainZoneField").hidden = $("trainWorld").value !== "region";
  };
  syncZone();
  $("trainWorld").addEventListener("change", syncZone);
  $("trainStage").addEventListener("change", renderWorkflow);
  loadDesigns();
  $("trainWorldDesign")?.addEventListener("change", () => applyPreset(false));
  $("trainPreset").addEventListener("change", () => applyPreset(true));
  for (const id of ["trainEnvs", "trainUpdates", "trainRollout", "trainEvaluationSteps"]) {
    $(id).addEventListener("input", () => applyPreset(false));
  }
  renderWorkflow();
}
