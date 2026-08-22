/* Push job state and metrics; polling is only the reconnect fallback. */

import { invalidate, state } from "./state.js";
import { loadRuns } from "./runs.js";
import { loadJobs, setPushConnected } from "./train.js";


function replaceJob(job) {
  if (!job) return;
  const jobs = [...(state.jobs || [])];
  const index = jobs.findIndex((item) => item.job_id === job.job_id);
  if (index >= 0) jobs[index] = job;
  else jobs.unshift(job);
  jobs.sort((left, right) => right.started_unix_seconds - left.started_unix_seconds);
  state.jobs = jobs;
}


function receive(message) {
  const event = JSON.parse(message.data);
  const payload = event.payload || {};
  const job = payload.job || (payload.job_id ? payload : null);
  replaceJob(job);
  const row = payload.row || ((payload.event || {}).payload?.metrics
    ? { kind: payload.event.kind === "evaluation" ? "evaluation" : "training",
        update: payload.event.payload.update, ...payload.event.payload.metrics }
    : null);
  if (row && (!state.jobMetricsFor || state.jobMetricsFor === payload.job_id)) {
    state.jobMetricsFor = payload.job_id;
    const prior = state.jobMetrics || [];
    const duplicate = prior.some((item) => item.update === row.update && item.kind === row.kind);
    if (!duplicate) state.jobMetrics = [...prior, row];
  }
  invalidate("jobs");
  // A run's artifact is written when the job ends, so this is the first moment
  // the Archive has a row to show. `noteRun` only covers runs launched from
  // this tab's Train panel; anything else -- another tab, a ladder or a script
  // posting to /api/train -- was invisible until a manual reload.
  if (message.type === "job_finished") loadRuns();
}


export function wireLive() {
  if (!("EventSource" in window)) return;
  const source = new EventSource("/api/live/events");
  source.onopen = () => {
    state.liveConnected = true;
    setPushConnected(true);
  };
  source.onerror = () => {
    state.liveConnected = false;
    setPushConnected(false);
    loadJobs();
    // Runs that finished while the stream was down produced no event, so the
    // reconnect has to re-read the list rather than wait for the next one.
    loadRuns();
  };
  for (const kind of ["job_started", "job_status", "job_metric", "job_finished"]) {
    source.addEventListener(kind, receive);
  }
}
