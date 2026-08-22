/* Stored runs: history that survives a reload, and comparability.
 *
 * The run list used to live only in `state.history` -- a session array, gone on
 * refresh, and "clear" really did destroy it. That is fine for eyeballing two
 * runs in a row and useless for the question this project keeps hitting: *was
 * this number produced against the same world as that one?* Bridge identity has
 * moved six times in a day, and the observation width went 7818 -> 8259 -> 8271
 * without anyone noticing.
 *
 * So the list is now read from the artifact store, and each row carries the
 * observation width it was produced against. Two runs whose contracts differ
 * are not two data points -- they are two different experiments, and the
 * compare view says so rather than overlaying them.
 */

import { $, invalidate, setData, state } from "./state.js";
import { frameCamera } from "./gfx.js";
import { showTab } from "./tabs.js";

/** Rows currently selected for comparison, newest-first run ids. */
const picked = new Set();

/* How many stored runs the Archive asks for. Bounded because the reader used
   to scan every artifact on disk -- 8,283 of them, 13.5s cold and 99.9s while a
   job was writing, which is what made the tab look dead. `loadMoreRuns` raises
   it on demand rather than making everyone pay for the deep history. */
export const RUN_PAGE = 200;
let runLimit = RUN_PAGE;

export async function loadRuns({ autoloadRegion = false } = {}) {
  try {
    const response = await fetch(`/api/runs?limit=${runLimit}`);
    if (!response.ok) return;
    const { runs } = await response.json();
    state.runs = runs;
    // A short page means the archive is exhausted, so the button can retire.
    state.runsExhausted = runs.length < runLimit;
    invalidate("runs");
    if (autoloadRegion && !state.data) {
      const region = runs.find((run) =>
        run.world === "region" && run.terrain_available && run.replay_available);
      if (region) await replayRun(region.run_id, { reveal: false });
    }
  } catch {
    /* The store is additive: a console that cannot reach it still runs. */
  }
}

/** Ask for another page of stored runs. */
export async function loadMoreRuns() {
  runLimit += RUN_PAGE;
  await loadRuns();
}

/** Delete one stored run and drop it from the list without a refetch. */
export async function deleteRun(runId) {
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`,
    { method: "DELETE" });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `could not delete ${runId}`);
  }
  state.runs = (state.runs || []).filter((run) => run.run_id !== runId);
  picked.delete(runId);
  invalidate("runs");
  return response.json();
}

/** Fold a freshly-finished run in without a refetch. */
export function noteRun(artifact) {
  if (!artifact || artifact.error) return;
  state.runs = [artifact, ...(state.runs || [])];
  invalidate("runs");
}

function label(run) {
  const spec = run.loadout ? `${run.loadout} vs ${run.opponent || "none"}` : run.run_id;
  return `${spec} · ${run.policy || "?"} · seed ${run.seed ?? "?"}`;
}

export function drawRuns() {
  const allRuns = state.runs || [];
  const profileId = state.activeProfileId || "";
  const activeProfile = (state.profiles || []).find((item) => item.id === profileId);
  const sharedAgent = activeProfile?.agent || profileId;
  const runs = profileId
    ? allRuns.filter((run) => ownedBy(run, profileId) || ownedBy(run, sharedAgent))
    : allRuns;
  const target = $("storedRuns");
  if (!target) return;

  if (!runs.length) {
    target.innerHTML = `<p class="muted">${activeProfile
      ? `No archived runs belong to ${esc(activeProfile.agent || activeProfile.display_title || activeProfile.title)} yet.`
      : `No stored runs yet. Agent-owned runs write to
      <code>artifacts/console/shared/&lt;agent&gt;/</code>; legacy runs remain readable.
      Every artifact records the nine contract
      hashes, the observation width and the Gym source digest it ran against.`}</p>`;
    $("storedNote").textContent = "";
    return;
  }

  // A width that is not unanimous means the set spans a contract change, and
  // any cross-run reading in this panel is comparing different worlds.
  const widths = new Set(runs.map((r) => r.observation_size).filter(Boolean));
  const spansContracts = widths.size > 1;

  target.innerHTML = runs
    .map((run) => {
      const flags = [];
      if (run.void) flags.push(`<i class="k crit">void</i>`);
      if (run.contracts_unavailable) flags.push(`<i class="k crit">no contracts</i>`);
      if (run.warnings) flags.push(`<i class="k warn">${run.warnings} warning${run.warnings > 1 ? "s" : ""}</i>`);
      const owners = run.profile_ids?.length ? run.profile_ids : [run.profile_id].filter(Boolean);
      if (owners.length) flags.push(`<i class="k a">${esc(owners.join(", "))}</i>`);
      const checked = picked.has(run.run_id) ? " checked" : "";
      // Shared-layout APIs publish the explicit flag. The fallback keeps an
      // already-running pre-migration Console useful while it catches up. It
      // used to special-case the "dawn-" prefix; receipt-only entries declare
      // `summary.replay_available: false` themselves, so read that instead and
      // the rule holds for every agent.
      const replayable = run.replay_available
        ?? run.summary?.replay_available
        ?? run.summary?.status !== "failed";
      const action = replayable
        ? `<button type="button" class="ghost mini" data-replay="${run.run_id}">replay</button>`
        : `<button type="button" class="ghost mini" data-report="${run.run_id}">details</button>`;
      return `<div class="runrow">
        <input type="checkbox" data-run="${run.run_id}"${checked} aria-label="compare ${run.run_id}">
        <span class="grow">${label(run)}</span>
        <em class="num" title="${esc(run.written_at || "")}">${esc(formatTimestamp(run.written_at))}</em>
        <em class="num">obs ${run.observation_size ?? "—"}</em>
        ${flags.join(" ")}
        ${action}
        <button type="button" class="ghost mini" data-delete="${run.run_id}"
          title="Delete this run and its artifacts. Not reversible.">delete</button>
      </div>`;
    })
    .join("");

  target.querySelectorAll("input[data-run]").forEach((box) => {
    box.addEventListener("change", () => {
      const id = box.dataset.run;
      if (box.checked) picked.add(id); else picked.delete(id);
      // Only the verdict depends on the selection. Re-rendering the list here
      // would rebuild every row, discarding the checkbox the user is still
      // interacting with -- ticking a second box then acts on a detached node.
      drawVerdict();
    });
  });

  target.querySelectorAll("button[data-replay]").forEach((button) => {
    button.addEventListener("click", () => replayRun(button.dataset.replay));
  });
  target.querySelectorAll("button[data-report]").forEach((button) => {
    button.addEventListener("click", () => showRunDetails(button.dataset.report));
  });
  target.querySelectorAll("button[data-delete]").forEach((button) => {
    button.addEventListener("click", async () => {
      const id = button.dataset.delete;
      if (!confirm(`Delete ${id} and its artifacts? This cannot be undone.`)) return;
      button.disabled = true;
      try {
        const removed = await deleteRun(id);
        $("storedNote").textContent =
          `deleted ${id} · ${removed.files_removed} files freed`;
      } catch (error) {
        button.disabled = false;
        alert(error.message);
      }
    });
  });

  const note = spansContracts
    ? `${runs.length} stored · newest first · ${widths.size} observation widths — this set spans a contract change`
    : `${runs.length} stored · newest first`;
  $("storedNote").textContent = note;
  const more = $("loadMoreRuns");
  if (more) {
    // The archive is read newest-first in bounded pages, so "no more" is a
    // real answer here rather than a spinner that never resolves.
    more.hidden = Boolean(state.runsExhausted);
    more.textContent = `load ${RUN_PAGE} more`;
  }

  drawVerdict();
}

/* Replay: put a stored battle back on the map.
 *
 * A stored run keeps its whole trajectory and, for a Region run, the ground it
 * happened on -- so this is the real fight again, scrubbable, not a summary of
 * it. The run is loaded exactly as a fresh one would be, which is what lets the
 * map, the event list, the charts and the per-step readouts all work unchanged.
 */
export async function replayRun(runId, { reveal = true } = {}) {
  $("status").textContent = `loading ${runId}…`;
  try {
    const response = await fetch(
      `/api/runs/${encodeURIComponent(runId)}/replay?blocks=true`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || response.statusText);
    frameCamera(payload.trajectory, payload.terrain);
    setData(payload);
    const profileId = payload.spec?.profile_id;
    const picker = $("profilePick");
    if (profileId && picker && [...picker.options].some((option) => option.value === profileId)) {
      picker.value = profileId;
      picker.dispatchEvent(new Event("change"));
    }
    if (reveal) showTab("run");
    $("status").textContent = payload.void
      ? `replaying ${runId} — VOID, its identity moved mid-run`
      : `replaying ${runId}`;
  } catch (error) {
    $("status").textContent = `replay failed: ${error.message}`;
  }
}

function ownedBy(run, profileId) {
  const owners = run.profile_ids || [run.profile_id].filter(Boolean);
  if (owners.includes(profileId)) return true;
  // A Console process started before shared ownership fields existed still
  // reads the additive flat compatibility view, where the only owner marker is
  // the run id prefix and the policy label. That rule is the same for every
  // agent -- it was written as a `profileId === "dawn"` branch, so any other
  // agent's campaign entries were invisible to its own Archive filter.
  const id = String(run.run_id || "").toLowerCase();
  const policy = String(run.policy || "").toLowerCase();
  const owner = profileId.toLowerCase();
  return id.startsWith(`${owner}-`) || policy.includes(owner);
}

async function showRunDetails(runId) {
  const target = $("compareVerdict");
  target.innerHTML = `<p class="muted">loading ${esc(runId)}...</p>`;
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
    const report = await response.json();
    if (!response.ok) throw new Error(report.detail || response.statusText);
    const source = report.spec?.source_report
      ? `<p class="muted">source: <code>${esc(report.spec.source_report)}</code></p>`
      : "";
    target.innerHTML = `<h3 class="mini">${esc(runId)}</h3>${source}` +
      `<pre class="archive-report">${esc(JSON.stringify(report.summary || {}, null, 2))}</pre>`;
    target.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    target.innerHTML = `<p class="crit">details failed: ${esc(error.message)}</p>`;
  }
}

function formatTimestamp(value) {
  if (!value) return "time unknown";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
}

/** The comparison panel alone -- safe to call on every selection change. */
function drawVerdict() {
  const verdict = $("compareVerdict");
  if (!verdict) return;
  if (picked.size !== 2) {
    verdict.innerHTML = picked.size
      ? `<p class="muted">Select one more run to compare (${picked.size} of 2).</p>`
      : "";
    return;
  }
  const [left, right] = [...picked];
  compare(left, right, verdict);
}

async function compare(left, right, target) {
  target.innerHTML = `<p class="muted">comparing…</p>`;
  let result;
  try {
    const response = await fetch(
      `/api/compare?left=${encodeURIComponent(left)}&right=${encodeURIComponent(right)}`);
    result = await response.json();
    if (!response.ok) throw new Error(result.detail || response.statusText);
  } catch (error) {
    target.innerHTML = `<p class="crit">compare failed: ${error.message}</p>`;
    return;
  }

  // Comparability leads. Settings, source and contracts are independent axes;
  // an omitted declaration is not silently treated as equality.
  const blocking = Object.entries(result.blocking || {});
  const verdict = result.verdict || (result.comparable ? "comparable" : "not comparable");
  const verdictTone = ["single-variable", "same configuration"].includes(verdict)
    ? "ok" : ["multi-variable", "undeclared"].includes(verdict) ? "warn" : "crit";
  const head = `<p class="${verdictTone}"><b>${verdict}.</b> ${result.comparable
    ? "Source and contracts match; the settings difference is interpretable."
    : "Do not read the result delta until every blocking axis is resolved."}</p>`;

  const axes = result.axes || {};
  const axisRows = ["settings", "source", "contracts"].map((name) => {
    const axis = axes[name] || {};
    const detail = name === "settings"
      ? `${axis.difference_count ?? "?"} changed key${axis.difference_count === 1 ? "" : "s"}`
      : name === "source"
        ? `${short(axis.left ?? "undeclared")} → ${short(axis.right ?? "undeclared")}`
        : `${axis.difference_count ?? "?"} changed declaration${axis.difference_count === 1 ? "" : "s"}`;
    return `<div class="diffrow"><span>${name}</span>` +
      `<em class="muted">${axis.state || "undeclared"} · ${detail}</em></div>`;
  }).join("");
  const axisView = `<h3 class="mini">comparison axes</h3><div class="histrows">${axisRows}</div>`;

  const why = blocking.length
    ? `<div class="histrows">${blocking
        .map(([field, sides]) =>
          `<div class="diffrow"><span>${field}</span>` +
          `<em class="muted">${short(sides.left)} → ${short(sides.right)}</em></div>`)
        .join("")}</div>`
    : "";

  const spec = Object.entries(result.spec_diff || {});
  const knobs = spec.length
    ? `<h3 class="mini">configuration</h3><div class="histrows">${spec
        .map(([field, sides]) =>
          `<div class="diffrow"><span>${field}</span>` +
          `<em class="muted">${short(sides.left)} → ${short(sides.right)}</em></div>`)
        .join("")}</div>`
    : `<p class="muted">Identical configuration — any difference is seed noise.</p>`;

  target.innerHTML = head + axisView + why + knobs;
}

function short(value) {
  const text = String(value);
  return text.length > 22 ? `${text.slice(0, 19)}…` : text;
}
