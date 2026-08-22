/* The Status workspace: what is running, and whether the contracts still hold.
 *
 * This opens first because it answers the question you actually have when you
 * come back to the tab. "What did that rollout do" is a question you go looking
 * for; "is my training still alive" is one you want answered on arrival.
 *
 * Every number here is read from a published surface -- `/api/jobs`,
 * `/api/adk`, `/api/bridge`, `/api/policy-agent`. Nothing is restated as a
 * literal, so the page cannot drift from the library it reports on.
 */

import { $, duration, isJobLive, state } from "./state.js";

/* ── training, as seen from Status ───────────────────────────────────────── */

export function drawTrainingNow() {
  const body = $("trainingNowBody");
  if (!body) return;

  const jobs = state.jobs || [];
  if (!jobs.length) {
    body.className = "empty";
    body.textContent = "No training job has run in this process.";
    return;
  }

  // These names have to match what `Job.view()` actually publishes. They did
  // not: this panel read `state`, `updates_done`, `num_envs`, `world` and
  // `wall_seconds`, none of which exist on the payload, so every field rendered
  // as an em dash and a live run looked like no run at all. The API returns
  // `status`, `update`, `elapsed_seconds`, and puts the run's configuration
  // under `spec`.
  const job = jobs[0];
  const status = job.status ?? "—";
  const running = isJobLive(job);
  const done = job.update ?? 0;
  const total = job.updates ?? 0;
  const pct = total ? Math.min(100, (done / total) * 100) : 0;
  const spec = job.spec || {};
  const worlds = spec.world_count ?? 1;
  // `elapsed_seconds` is computed by the server when the payload is built, so
  // it only advances on a fetch. This panel redraws every second from cached
  // job state, which made the value sit still. Derive it from the start stamp
  // while the job is live, and trust the server's figure once it has ended.
  const elapsed = running && job.started_unix_seconds
    ? Date.now() / 1000 - job.started_unix_seconds
    : job.elapsed_seconds;

  const remaining = running ? remainingSeconds(job, elapsed) : null;

  body.className = "";
  body.innerHTML = `
    <div class="facts">
      ${fact("job", job.job_id ? job.job_id.slice(0, 12) : "—")}
      ${fact("state", status, running ? "pos" : status === "failed" ? "neg" : "")}
      ${fact("updates", total ? `${done} / ${total}` : String(done))}
      ${fact("envs", spec.num_envs ?? spec.batch ?? "—")}
      ${fact("world", worlds > 1 ? `${spec.world ?? "—"} x${worlds}` : (spec.world ?? "—"))}
      ${fact("elapsed", elapsed != null ? duration(elapsed) : "—")}
      ${fact("remaining", remaining != null ? `~${duration(remaining)}` : "—")}
    </div>
    <div class="bar" role="progressbar" aria-valuenow="${pct.toFixed(0)}"
         aria-valuemin="0" aria-valuemax="100">
      <span style="width:${pct.toFixed(1)}%"></span>
    </div>
    ${job.stop_reason ? `<div class="flag sev-warn">stopped early: ${escape(job.stop_reason)}</div>` : ""}
    ${job.error ? `<div class="flag sev-crit">${escape(job.error)}</div>` : ""}`;
}

/* Time left, from the rate this run is actually achieving.

   The scene build, the program build and the first update are one-off costs --
   the first update alone measured 267s against a 13.5s steady state -- so they
   are subtracted before dividing. Including them would inflate the estimate for
   the whole run. Evaluations are deliberately left in: they happen on the way,
   so a figure that ignored them would always run short. */
function remainingSeconds(job, elapsed) {
  const done = job.update ?? 0;
  const total = job.updates ?? 0;
  // Two completed updates is the minimum that gives a rate to project from.
  if (!total || done < 2 || elapsed == null) return null;
  const cache = job.cache || {};
  const startup = (cache.scene_build_seconds || 0)
    + (cache.program_build_seconds || 0)
    + (cache.first_update_seconds || 0);
  const training = elapsed - startup;
  if (training <= 0) return null;
  return (training / (done - 1)) * (total - done);
}

/* ── contract health ─────────────────────────────────────────────────────── */

/* The things that silently move and void a result. A checkpoint only loads into
   a scene whose contract matches the one it trained against, so an observation
   width or action count that has shifted is not a detail -- it is the reason a
   stored number stops meaning anything. */
export function drawContractFacts(adk, bridge) {
  const box = $("contractFacts");
  if (!box) return;

  const obs = adk?.observation ?? {};
  const act = adk?.action ?? {};
  const listeners = bridge?.listeners ?? [];
  const native = listeners.find((l) => l.role === "native");

  box.innerHTML = [
    // `named_columns` is the *named* subset, not the full observation width --
    // 241 of 8271 on the current contract. Labelling it "observation columns"
    // invites exactly the misreading this project keeps making, so the label
    // says which number it is.
    fact("named obs columns", obs.named_columns ?? "—"),
    fact("observation groups", obs.group_count ?? "—"),
    fact("action logits", act.logits ?? "—"),
    fact("action heads", act.head_count ?? "—"),
    fact("combat parameters", adk?.combat_parameters ?? "—"),
    fact("loadouts", (adk?.loadouts ?? []).length || "—"),
    fact("native bridge", native ? (native.listening ? "listening" : "down") : "—",
      native?.listening ? "pos" : "neg"),
    fact("ADK version", adk?.version ?? "—"),
  ].join("");
}

/* ── helpers ─────────────────────────────────────────────────────────────── */

function fact(label, value, cls = "") {
  return `<div class="ro"><i>${escape(label)}</i><b class="${cls}">${escape(value)}</b></div>`;
}

function escape(value) {
  return String(value).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}
