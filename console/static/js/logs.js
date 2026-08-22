/* Live logs in Status.
 *
 * Sources come from the server's own discovery, so the picker lists whatever
 * exists rather than a list kept in step by hand. Following polls the tail
 * endpoint; a log is a file being appended to, and there is nothing to push.
 */

import { $, state } from "./state.js";

const POLL_MS = 2000;
let timer = null;
let known = "";
let known_default = "";

/** Empty the pane so one run's output never sits under another's heading. */
function clearLog() {
  const body = $("logBody");
  if (body) body.textContent = "";
  const count = $("logCount");
  if (count) count.textContent = "0 lines";
  const meta = $("logMeta");
  if (meta) meta.textContent = "";
}

function meta(payload) {
  if (!payload) return "";
  if (payload.error) return payload.error;
  const kb = (payload.bytes / 1024).toFixed(0);
  const when = payload.modified
    ? new Date(payload.modified * 1000).toLocaleTimeString() : "";
  return `${kb} kB on disk · last written ${when}`
    + (payload.truncated ? " · showing the tail only" : "");
}

/** Mirror the Java console's chip: what this panel is doing right now. */
function setState(text, tone) {
  const chip = $("logState");
  if (!chip) return;
  chip.textContent = text;
  chip.dataset.tone = tone || "";
}

export async function loadLogSources() {
  const select = $("logSource");
  if (!select) return;
  try {
    const response = await fetch("/api/logs");
    if (!response.ok) return;
    const payload = await response.json();
    const sources = payload.sources || [];
    // The server nominates the current job's log. When that nomination moves, a
    // new run has started: follow it and empty the pane, so the previous run's
    // output is never left on screen under the new run's heading.
    const nominated = payload.default || "";
    const startedNewRun = Boolean(nominated) && nominated !== known_default;
    known_default = nominated;
    // Rebuild only when the set changes, so an open picker is not reset every
    // poll while someone is choosing from it.
    const signature = sources.map((s) => s.id).join("|");
    if (signature === known && !startedNewRun) return;
    known = signature;
    const chosen = select.value;
    select.innerHTML = sources.length
      ? sources.map((s) =>
          `<option value="${s.id}">${s.category} · ${s.name}</option>`).join("")
      : `<option value="">no logs found</option>`;
    if (startedNewRun) {
      select.value = nominated;
      clearLog();
    } else if (sources.some((s) => s.id === chosen)) {
      // Keep an explicit choice within one run.
      select.value = chosen;
    } else if (nominated) {
      select.value = nominated;
    }
  } catch {
    /* a console without the endpoint simply shows nothing */
  }
}

export async function drawLog() {
  const select = $("logSource");
  const body = $("logBody");
  if (!select || !body) return;
  const id = select.value;
  if (!id) {
    body.textContent = "Select a source.";
    $("logMeta").textContent = "";
    $("logPath").textContent = "No log selected.";
    $("logCount").textContent = "0 lines";
    setState("idle");
    return;
  }
  const lines = Number($("logLines")?.value) || 200;
  try {
    const response = await fetch(
      `/api/logs/${id.split("/").map(encodeURIComponent).join("/")}?lines=${lines}`);
    if (!response.ok) {
      body.textContent = `Could not read ${id} (${response.status}).`;
      setState("unreadable", "bad");
      return;
    }
    const payload = await response.json();
    // Keep the view pinned to the newest line unless the reader scrolled up.
    const pinned = body.scrollTop + body.clientHeight >= body.scrollHeight - 4;
    const rows = payload.lines || [];
    body.textContent = rows.join("\n") || "(empty)";
    $("logMeta").textContent = meta(payload);
    $("logPath").textContent = payload.id;
    $("logCount").textContent = `${rows.length} line${rows.length === 1 ? "" : "s"}`;
    setState(payload.error ? "unreadable" : (timer ? "following" : "paused"),
             payload.error ? "bad" : "ok");
    if (pinned) body.scrollTop = body.scrollHeight;
  } catch (error) {
    body.textContent = `Log read failed: ${error.message}`;
    setState("unreachable", "bad");
  }
}

function follow() {
  const on = $("logFollow")?.checked;
  if (on && !timer) {
    timer = setInterval(() => { loadLogSources(); drawLog(); }, POLL_MS);
  }
  if (!on && timer) { clearInterval(timer); timer = null; }
}

export function wireLogs() {
  const select = $("logSource");
  if (!select) return;
  select.addEventListener("change", drawLog);
  $("logLines")?.addEventListener("change", drawLog);
  $("logFollow")?.addEventListener("change", follow);
  loadLogSources().then(drawLog).then(follow);
}

/** Stop polling when the tab is not visible; a hidden log costs nothing. */
export function logsTabChanged(activeTab) {
  if (activeTab === "status") {
    follow();
    loadLogSources().then(drawLog);
  } else if (timer) {
    clearInterval(timer);
    timer = null;
  }
}

export { state };
