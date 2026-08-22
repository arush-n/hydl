/* The pursuit curriculum, and queueing it.
 *
 * Both ladders are READ from `/api/train/curriculum`, which reads them from
 * `arena.curriculum.pursuit`. Nothing here restates a rung name, a reward
 * weight or a gate threshold: a second copy of "what gets harder, and when" is
 * exactly how the console and the training loop come to disagree about which
 * rung a run was on.
 *
 * Queueing chains through the `latest` source-policy sentinel rather than by
 * writing checkpoint paths. A rung's output directory is named after a run id
 * that does not exist when the queue entry is written, so a literal path is
 * unwritable; `latest` resolves at dequeue, which is when the rung below has
 * finished and its checkpoint exists.
 */

import { $ } from "./state.js";
import { trainRequestBody } from "./train.js";

let curriculum = null;

function row(title, detail, tag) {
  const element = document.createElement("div");
  element.className = "row";
  element.innerHTML =
    `<div class="rowMain"><b>${title}</b>` +
    (tag ? ` <i class="k a">${tag}</i>` : "") +
    `</div><div class="note">${detail}</div>`;
  return element;
}

/** Non-zero reward terms, so a rung reads as what it PAYS, not as 13 fields. */
function paidTerms(settings) {
  const paid = Object.entries(settings || {})
    .filter(([, value]) => value !== 0 && value !== false)
    .map(([name, value]) => `${name.replace(/^evader_/, "")}=${value}`);
  return paid.length ? paid.join(", ") : "nothing paid";
}

export async function loadLadder() {
  const note = $("ladderNote");
  try {
    curriculum = await (await fetch("/api/train/curriculum")).json();
  } catch (error) {
    if (note) note.textContent = "curriculum unavailable";
    return;
  }
  const objectives = curriculum.objectives || {};
  const stages = objectives.stages || [];
  const recursive = objectives.recursive_stage;
  const terminal = objectives.terminal_multiple || {};

  const rewards = $("ladderObjectives");
  if (rewards) {
    rewards.innerHTML = "";
    stages.forEach((stage, index) => {
      const multiple = terminal[stage.name];
      rewards.append(row(
        `${index + 1}. ${stage.name}`,
        `terminal x${multiple ?? "?"} &middot; ${paidTerms(stage.settings)}`,
        stage.name === recursive ? "self-play" : "",
      ));
    });
    if (!stages.length) rewards.innerHTML = '<div class="empty">&mdash;</div>';
  }

  const worlds = $("ladderWorlds");
  if (worlds) {
    worlds.innerHTML = "";
    (curriculum.stages || []).forEach((stage, index) => {
      worlds.append(row(`${index + 1}. ${stage.name}`, "world rung", ""));
    });
    if (!(curriculum.stages || []).length) {
      worlds.innerHTML = '<div class="empty">&mdash;</div>';
    }
  }

  const gate = curriculum.gate || {};
  const band = curriculum.horizon_band || {};
  const flags = $("ladderGate");
  if (flags) {
    flags.innerHTML =
      `<span class="flag">promote &ge; ${gate.promote_at}</span>` +
      `<span class="flag">demote &lt; ${gate.demote_below}</span>` +
      `<span class="flag">window ${gate.window}</span>` +
      `<span class="flag">min evals ${gate.minimum_evaluations}</span>` +
      `<span class="flag">horizon ${band.minimum}&ndash;${band.maximum}</span>`;
  }
  if (note) {
    note.textContent =
      `${stages.length} reward rungs, ${(curriculum.stages || []).length} world rungs`;
  }
  await loadLadderState();
  await loadQueue();
}

export async function loadQueue() {
  const rows = $("queueRows");
  const note = $("queueNote");
  let payload;
  try {
    payload = await (await fetch("/api/train/queue")).json();
  } catch (error) {
    if (note) note.textContent = "unavailable";
    return;
  }
  const queued = payload.queued || [];
  if (note) {
    note.textContent = queued.length
      ? `${queued.length} waiting${payload.active ? ", 1 running" : ""}`
      : (payload.active ? "1 running, none waiting" : "empty");
  }
  if (!rows) return;
  rows.innerHTML = "";
  if (!queued.length) {
    rows.innerHTML = '<div class="empty">Nothing queued.</div>';
    return;
  }
  queued.forEach((entry) => {
    const element = row(
      `${entry.position}. ${entry.label || entry.queued_id}`,
      entry.queued_id,
      "",
    );
    const drop = document.createElement("button");
    drop.className = "ghost mini";
    drop.textContent = "Remove";
    drop.onclick = async () => {
      await fetch(`/api/train/queue/${entry.queued_id}`, { method: "DELETE" });
      await loadQueue();
    };
    element.querySelector(".rowMain").append(drop);
    rows.append(element);
  });
}

/** Queue `names` in order: rung 1 from the form, the rest from `latest`.
 *
 * A RUNG NAME IS NOT AN OBJECTIVE. `objective` is `Literal["pursue","evade"]`
 * and every rung of this ladder trains the EVADER, so sending "long_range"
 * there made the API reject all four with a 422 and the ladder never ran. The
 * role comes from the curriculum payload, which states it once beside the
 * rungs; the rung name selects the SETTINGS, which are what actually differ.
 */
async function queueRungs(names) {
  const note = $("ladderNote");
  const updates = Number($("ladderUpdates").value) || 64;
  const base = trainRequestBody();
  const objectives = curriculum?.objectives || {};
  const byName = new Map(
    (objectives.stages || []).map((stage) => [stage.name, stage.settings || {}]),
  );
  const horizons = objectives.horizon_by_stage || {};
  let queued = 0;
  for (const [index, name] of names.entries()) {
    // `episode_ticks` and `evaluation_steps` are refused when they disagree,
    // so the rung's horizon has to set both or neither.
    const horizon = horizons[name];
    const spec = {
      ...base,
      ...(byName.get(name) || {}),
      updates,
      objective: objectives.objective || "evade",
      target_controller: objectives.target_controller || "frozen_source_policy",
      ...(horizon
        ? { episode_ticks: horizon, evaluation_steps: horizon }
        : {}),
      // Rung 1 keeps whatever the form resolved. Every rung above it chains,
      // and cannot name a path that does not exist yet.
      ...(index === 0 ? {} : { source_policy: "latest" }),
    };
    const response = await fetch("/api/train?queue=true", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(spec),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      if (note) {
        note.textContent = `rung "${name}" refused: ${detail.detail || response.status}`;
      }
      break;
    }
    queued += 1;
  }
  if (note && queued === names.length) {
    note.textContent = `queued ${queued} rung${queued === 1 ? "" : "s"}`;
  }
  await loadQueue();
}

/** Where the gate has got to, so "not promoting" is visible rather than felt. */
export async function loadLadderState() {
  const note = $("ladderState");
  if (!note) return;
  let state;
  try {
    state = await (await fetch("/api/train/ladder")).json();
  } catch (error) {
    note.textContent = "ladder state unavailable";
    return;
  }
  const rate = state.recent_success_rate;
  const last = (state.transitions || []).slice(-1)[0];
  note.innerHTML =
    `<span class="flag">${state.running ? "running" : "idle"}</span>` +
    `<span class="flag">rung ${state.index + 1}/${state.stages.length}` +
    ` ${state.current || "done"}</span>` +
    `<span class="flag">evals ${state.evaluations_at_stage}` +
    `/${state.gate.minimum_evaluations}</span>` +
    `<span class="flag">recent ${rate === null || rate === undefined
      ? "&mdash;" : rate.toFixed(3)} vs ${state.gate.promote_at}</span>` +
    (last ? `<span class="flag">last ${last.transition}` +
      `${last.reason ? ": " + last.reason : ""}</span>` : "");
}

export function wireLadder() {
  const refresh = $("ladderRefresh");
  if (refresh) refresh.onclick = () => loadLadder();

  // Starts ONE rung. The gate picks the next when this one is graded, which is
  // the difference between a ladder and a playlist.
  const run = $("ladderQueue");
  if (run) {
    run.onclick = async () => {
      const note = $("ladderNote");
      const updates = Number($("ladderUpdates").value) || 64;
      let body;
      try {
        body = trainRequestBody();
      } catch (error) {
        if (note) note.textContent = String(error.message || error);
        return;
      }
      const response = await fetch(
        `/api/train/ladder/start?updates=${updates}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
      );
      const payload = await response.json().catch(() => ({}));
      if (note) {
        note.textContent = response.ok
          ? `ladder running: ${payload.ladder?.current}`
          : `refused: ${payload.detail || response.status}`;
      }
      await loadLadderState();
      await loadQueue();
    };
  }

  const stop = $("ladderStop");
  if (stop) {
    stop.onclick = async () => {
      await fetch("/api/train/ladder/stop", { method: "POST" });
      await loadLadderState();
    };
  }

  const recursive = $("ladderQueueRecursive");
  if (recursive) {
    recursive.onclick = () => {
      const name = curriculum?.objectives?.recursive_stage;
      if (name) queueRungs([name]);
    };
  }
}
