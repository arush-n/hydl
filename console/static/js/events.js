/* "What happened" -- the debugging core.
 *
 * The environment publishes arsenal failure bits saying *why* something did
 * not happen. Those were never surfaced; a run that silently did nothing
 * looked identical to a run that was refused. Here every discrete moment gets
 * a tick, a severity colour and a click that jumps the playhead to it.
 */

import { $, setStep, state } from "./state.js";
import { css, fit, g2d } from "./gfx.js";

const SEVERITY = { info: "sev-info", good: "sev-good", warn: "sev-warn", crit: "sev-crit" };
const KINDS = [
  ["ability_start", "info", "abilities"],
  ["hit", "good", "hits"],
  ["damage_taken", "warn", "damage taken"],
  ["failure", "crit", "failures"],
  ["episode_end", "crit", "episode end"],
];

function bucket(kind) {
  return kind.startsWith("failure:") ? "failure" : kind;
}

export function drawFilters() {
  $("eventFilters").innerHTML = KINDS
    .map(([kind, sev, label]) =>
      `<button type="button" class="chip ${SEVERITY[sev]}" data-kind="${kind}" ` +
      `aria-pressed="${!state.filters.has(kind)}">${label}</button>`)
    .join("");
}

export function drawEvents() {
  const DATA = state.data;
  const list = $("events");
  if (!DATA || !DATA.events) { list.innerHTML = `<li class="empty">no run yet</li>`; return; }

  const shown = DATA.events.filter((e) => !state.filters.has(bucket(e.kind)));
  $("eventCount").textContent =
    `${shown.length} of ${DATA.events.length} event${DATA.events.length === 1 ? "" : "s"}`;

  if (!shown.length) {
    list.innerHTML = `<li class="empty">nothing matched — this run was quiet, ` +
      `which is itself a result worth explaining</li>`;
    return;
  }

  const at = state.step;
  list.innerHTML = shown.map((e) => {
    const sev = SEVERITY[e.severity] || "sev-info";
    const near = Math.abs(e.tick - at) <= 1 ? " at" : "";
    const secs = (e.tick / 30).toFixed(2);
    return `<li class="ev ${sev}${near}" data-tick="${e.tick}">` +
      `<u>t${e.tick} · ${secs}s</u><b>${e.kind.replace("failure:", "⚠ ")}</b>` +
      `<span>${e.detail}</span></li>`;
  }).join("");
}

/** Timeline strip: visibility band, health lines, event ticks, playhead. */
export function drawTimeline() {
  const cv = $("timeline");
  const [W, H] = fit(cv);
  const g = g2d(cv);
  g.fillStyle = css("--panel");
  g.fillRect(0, 0, W, H);

  const DATA = state.data;
  if (!DATA) return;
  const t = DATA.trajectory, N = DATA.steps;

  g.fillStyle = css("--ok"); g.globalAlpha = 0.13;
  for (let i = 0; i < N; i++) if (t.visible[i]) g.fillRect((i / N) * W, 0, W / N + 0.7, H);
  g.globalAlpha = 1;

  const line = (arr, max, col) => {
    g.strokeStyle = col; g.lineWidth = 1.7; g.beginPath();
    for (let i = 0; i < N; i++) {
      const x = (i / N) * W, y = H - (arr[i] / max) * (H - 8) - 4;
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    }
    g.stroke();
  };
  line(t.agent_health, DATA.spec.agent_max_health, css("--agent"));
  line(t.target_health, DATA.spec.target_max_health, css("--target"));

  /* ability starts along the bottom, hits and failures full height */
  g.fillStyle = css("--info");
  (t.ability_start || []).forEach((v, i) => { if (v) g.fillRect((i / N) * W, H - 6, 1.6, 6); });
  for (const e of DATA.events || []) {
    if (e.severity === "good") g.fillStyle = css("--ok");
    else if (e.severity === "crit") g.fillStyle = css("--crit");
    else continue;
    g.fillRect((e.tick / N) * W - 1, 0, 2.4, H);
  }

  g.fillStyle = css("--ink");
  g.fillRect((state.step / N) * W - 0.8, 0, 1.6, H);
}

export function wireEvents() {
  $("events").addEventListener("click", (e) => {
    const row = e.target.closest(".ev");
    if (row) setStep(+row.dataset.tick);
  });
  $("eventFilters").addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (!chip) return;
    const kind = chip.dataset.kind;
    state.filters.has(kind) ? state.filters.delete(kind) : state.filters.add(kind);
    chip.setAttribute("aria-pressed", String(!state.filters.has(kind)));
    drawEvents();
  });
  $("timeline").addEventListener("click", (e) => {
    if (!state.data) return;
    const r = e.currentTarget.getBoundingClientRect();
    setStep(Math.round(((e.clientX - r.left) / r.width) * state.data.steps));
  });
}
