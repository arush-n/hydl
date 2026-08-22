/* Wiring. Every surface declares what it depends on; nothing draws itself. */

import { $, invalidate, roles, setStep, state, surface } from "./state.js";
import { dropPalette, orbit, setMode, wasd } from "./gfx.js";
import { drawScene } from "./scene.js";
import { drawRunCharts, drawHistoryCharts } from "./charts.js";
import { drawEvents, drawFilters, drawTimeline, wireEvents } from "./events.js";
import { loadOptions, loadPolicyAgent, loadProfiles, loadSurfaces, verdict, wireLaunch } from "./launch.js";
import { watch } from "./hot.js";
import { drawRuns, loadMoreRuns, loadRuns, replayRun } from "./runs.js";
import { drawJobs, loadAgentMetrics, loadJobs, loadTrainOptions, wireTrain } from "./train.js";
import { wireLive } from "./live.js";
import { loadLadder, wireLadder } from "./ladder.js";
import { drawContractFacts, drawTrainingNow } from "./status.js";
import { activeTab, onTab, showTab, wireTabs, wireTheme } from "./tabs.js";
import { loadPanels, panelsTabChanged } from "./panels.js";
import { logsTabChanged, wireLogs } from "./logs.js";
import { loadWorlds } from "./worlds.js";
import { loadDevices, runCost, wireCompute } from "./compute.js";
import { javaTabChanged, wireJava } from "./java.js";
import { drawWorldgen, loadWorldgenSeed, wireWorldgen, worldgenTabChanged } from "./worldgen.js";

/* data  -> a new run arrived        frame -> the playhead moved
   view  -> the camera moved         history -> the run list changed */
surface("scene", ["data", "frame", "view"], drawScene);
surface("timeline", ["data", "frame"], drawTimeline);
surface("runCharts", ["data", "frame"], drawRunCharts);
surface("events", ["data", "frame"], drawEvents);
surface("verdict", ["data"], verdict);
surface("history", ["history"], drawHistoryCharts);
surface("runs", ["runs"], drawRuns);
surface("jobs", ["jobs"], drawJobs);
surface("detail", ["data", "frame"], drawDetail);
surface("scrub", ["data", "frame"], syncScrub);
surface("trainingNow", ["jobs"], drawTrainingNow);
surface("runCost", ["data"], runCost);

function syncScrub() {
  const DATA = state.data;
  const scrub = $("scrub");
  scrub.max = DATA ? DATA.steps - 1 : 0;
  scrub.value = state.step;
  $("stepn").textContent = DATA
    ? `step ${state.step} / ${DATA.steps - 1} · ${(state.step / 30).toFixed(2)}s`
    : "—";
}

function drawDetail() {
  const DATA = state.data;
  if (!DATA) { $("readouts").innerHTML = ""; $("heads").innerHTML = ""; return; }
  const t = DATA.trajectory, i = state.step;
  const seen = !!t.visible[i];
  let reward = 0;
  for (let k = 0; k <= i; k++) reward += t.reward[k];
  const staminaAvailable = t.stamina &&
    (!t.stamina_available || Boolean(t.stamina_available[i]));
  const staminaExhausted = staminaAvailable &&
    (t.stamina_exhausted ? Boolean(t.stamina_exhausted[i]) : t.stamina[i] <= 0);

  const role = roles();
  const rows = [
    ["visible", seen ? "YES" : "NO", seen ? "pos" : "neg"],
    ["observed dist", seen ? t.observed_distance[i].toFixed(3) : "masked", seen ? "" : "absent"],
    ["true dist", t.true_distance[i].toFixed(2), ""],
    [`${role.learner} hp`, t.agent_health[i].toFixed(0), ""],
    [`${role.opponent} hp`, t.target_health[i].toFixed(0), ""],
    ["ability slot", t.active_slot ? String(t.active_slot[i]) : "—",
      t.active_slot && t.active_slot[i] >= 0 ? "pos" : "absent"],
    ["attack requested", t.requested ? (t.requested[i] ? "YES" : "NO") : "\u2014",
      t.requested && t.requested[i] ? "pos" : ""],
    ["attack accepted", t.accepted ? (t.accepted[i] ? "YES" : "NO") : "\u2014",
      t.accepted && t.accepted[i] ? "pos" : ""],
    ["attack executing", t.attack_executing ? (t.attack_executing[i] ? "YES" : "NO") : "\u2014",
      t.attack_executing && t.attack_executing[i] ? "pos" : ""],
    ["cooldown", t.cooldown ? Number(t.cooldown[i]).toFixed(3) : "\u2014",
      t.cooldown && t.cooldown[i] > 0 ? "absent" : ""],
    ["busy retry", t.busy_reject ? (t.busy_reject[i] ? "YES" : "NO") : "\u2014",
      t.busy_reject && t.busy_reject[i] ? "neg" : ""],
    ["aim error", t.aim_error ? `${Number(t.aim_error[i]).toFixed(1)}\u00b0` : "\u2014",
      t.aim_error && t.aim_error[i] > 30 ? "neg" : t.aim_error && t.aim_error[i] <= 15 ? "pos" : ""],
    ["yaw error", t.yaw_error ? `${Number(t.yaw_error[i]).toFixed(1)}\u00b0` : "\u2014", ""],
    ["pitch error", t.pitch_error ? `${Number(t.pitch_error[i]).toFixed(1)}\u00b0` : "\u2014",
      t.pitch_error && t.pitch_error[i] > 15 ? "neg" : t.pitch_error ? "pos" : ""],
    ["pitch", t.agent_pitch ? `${Number(t.agent_pitch[i]).toFixed(1)}\u00b0 \u2192 ${Number(t.desired_pitch[i]).toFixed(1)}\u00b0` : "\u2014", ""],
    ["joint action", t.action_legal ? (t.action_legal[i] ? "legal" : "REJECTED") : "\u2014",
      t.action_legal && !t.action_legal[i] ? "neg" : ""],
    ["stamina", staminaAvailable ? Number(t.stamina[i]).toFixed(1) : "—",
      staminaExhausted ? "neg" : ""],
    ["projectiles", t.projectiles ? String(t.projectiles[i]) : "—", ""],
    // Guarded like every other optional column. Unguarded, a replay without
    // `target_phase` threw out of `drawDetail` and took the whole render with
    // it -- the readouts, the heads panel, and every surface flushed after it.
    // A missing field must read as "no reading", never abort the frame.
    ["phase", t.target_phase && DATA.target_phases
      ? (DATA.target_phases[t.target_phase[i]] ?? "—")
      : "—", ""],
    ["reward Σ", reward.toFixed(2), reward < 0 ? "neg" : reward > 0 ? "pos" : ""],
  ];
  $("readouts").innerHTML = rows
    .map(([k, v, c]) => `<div class="ro"><i>${k}</i><b class="${c}">${v}</b></div>`).join("");
  // Same rule as the readouts above: a replay that carries no head names or no
  // per-step action vector renders as absent, it does not abort the frame.
  // `drawDetail` is a registered surface, so throwing here stopped every
  // surface queued behind it -- which is how one missing optional field made
  // the Archive and Evidence panels look broken.
  const heads = DATA.head_names || [];
  const step = t.action?.[i];
  $("heads").innerHTML = heads.length && step
    ? heads.map((h, hi) => {
        const v = step[hi];
        return `<span class="${v ? "" : "z"}"><em>${h.split("_")[0]}</em>${v}</span>`;
      }).join("")
    : `<span class="z"><em>action heads</em>not recorded</span>`;
}

/* ── input ───────────────────────────────────────────────────────────────── */
$("scrub").addEventListener("input", (e) => setStep(+e.target.value));
$("trails").addEventListener("change", () => invalidate("view"));
$("showGrid").addEventListener("change", () => invalidate("view"));
$("terrainOn").addEventListener("change", () => invalidate("view"));
orbit($("view"), () => invalidate("view"));
/* WASD pans across the ground, QE raises and lowers. */
wasd($("view"), () => invalidate("view"));

/* 3D reads terrain shape; 2D reads position. Perspective makes the same
   separation look different depending where the actors stand, and a spacing
   band is exactly what this project keeps needing to eyeball. */
$("viewMode").addEventListener("click", (e) => {
  const button = e.target.closest("button[data-mode]");
  if (!button) return;
  setMode(button.dataset.mode);
  for (const b of $("viewMode").querySelectorAll("button")) {
    b.setAttribute("aria-pressed", String(b === button));
  }
  $("camhint").textContent = button.dataset.mode === "2d"
    ? "click map · WASD pan · shift+WASD precision · drag rotate · right drag pan · wheel zoom"
    : "click map · WASD pan · shift+WASD precision · drag orbit · right drag pan · wheel zoom";
  invalidate("view");
});
addEventListener("resize", () => invalidate("data", "frame", "view", "history"));
matchMedia("(prefers-color-scheme:dark)").addEventListener("change", () => {
  dropPalette();
  invalidate("data", "frame", "view", "history");
});

let timer = null;
$("play").addEventListener("click", () => {
  if (timer) { clearInterval(timer); timer = null; $("play").textContent = "▶"; return; }
  if (!state.data) return;
  $("play").textContent = "⏸";
  timer = setInterval(() => {
    if (!state.data || state.step >= state.data.steps - 1) {
      clearInterval(timer); timer = null; $("play").textContent = "▶"; return;
    }
    setStep(state.step + 1);
  }, 1000 / 30);
});

addEventListener("keydown", (e) => {
  if (e.target.matches("input,select,textarea")) return;
  if (e.key === "ArrowRight") setStep(state.step + (e.shiftKey ? 10 : 1));
  else if (e.key === "ArrowLeft") setStep(state.step - (e.shiftKey ? 10 : 1));
  else if (e.key === " ") { e.preventDefault(); $("play").click(); }
});

/* ── boot ────────────────────────────────────────────────────────────────── */
wireTabs();
wireTheme(() => {
  dropPalette();
  invalidate("data", "frame", "view", "history");
  drawWorldgen();
});
/* A finished run is worth seeing, so land on Run rather than leaving the
   result behind on Status -- but only when the user has not navigated away
   themselves in the meantime. */
/* The game server log is chatty and only worth polling while it is on screen,
   so the Java workspace starts and stops its own timer as the tab changes. */
onTab(javaTabChanged);
onTab(worldgenTabChanged);
onTab(panelsTabChanged);
onTab(logsTabChanged);
// The world library is only read when Evidence is open: it scans run
// reports, which is not work to do on every tab change.
onTab((tab) => { if (tab === "evidence") loadWorlds(); });

let devicesTimer = null;
let policyTimer = null;
function workspacePolling(tab) {
  if (devicesTimer) clearInterval(devicesTimer);
  if (policyTimer) clearInterval(policyTimer);
  devicesTimer = null;
  policyTimer = null;
  if (tab === "compute") {
    loadDevices();
    devicesTimer = setInterval(loadDevices, 10000);
  }
  if (tab === "status") {
    loadPolicyAgent();
    policyTimer = setInterval(loadPolicyAgent, 5000);
  }
}
onTab(workspacePolling);
wireEvents();
wireJava();
wireWorldgen();
$("terrainSeed").addEventListener("click", (event) => {
  showTab("custom");
  loadWorldgenSeed(Number(event.currentTarget.dataset.seed));
});
wireLaunch(() => showTab("run"));
drawFilters();
invalidate("data", "frame", "view", "history", "runs", "jobs");

/* Stored runs outlive the session, so the list is read back at boot rather
   than starting empty and pretending nothing came before. */
const directReplay = new URLSearchParams(location.search).get("replay");
loadRuns({ autoloadRegion: !directReplay });
$("loadMoreRuns")?.addEventListener("click", async (event) => {
  event.currentTarget.disabled = true;
  await loadMoreRuns();
  event.currentTarget.disabled = false;
});
if (directReplay) replayRun(directReplay);
wireTrain();
  wireLadder();
// Before loadJobs, so the first Evidence render already knows which metrics
// each agent declares instead of falling back for one frame.
loadAgentMetrics();
wireLogs();
loadJobs();
loadTrainOptions();
  loadLadder();
wireLive();

/* Panels are rescanned server-side on every request, so an agent that writes
   console/panels/mine.py mid-session appears here without a restart. */
loadProfiles();
wireCompute();
loadPanels();
setInterval(loadPanels, 30000);

loadOptions().catch(() => { $("status").textContent = "could not read /api/options"; });
loadSurfaces();
setInterval(loadSurfaces, 15000);
/* Device and live-policy reads are useful only in their owning workspaces. */
workspacePolling(activeTab());
watch();
