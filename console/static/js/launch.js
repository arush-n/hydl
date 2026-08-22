/* Launching runs, and saying plainly whether the result is usable.
 *
 * The verdict block is the point: a run that produced no usable signal must
 * not look like a run that did. Colour carries it -- green ran clean, amber
 * ran but you must not read it at face value, red failed.
 */

import { $, TPS, setData, state, invalidate } from "./state.js";
import { noteRun, replayRun } from "./runs.js";
import { frameCamera } from "./gfx.js";
import { drawContractFacts } from "./status.js";
import { showTab } from "./tabs.js";

const NUMERIC = new Set(["ticks", "seed", "slot", "period", "decision_period",
  "agent_max_health", "target_max_health", "minigame_weight", "episode_ticks"]);

export function readForm() {
  const form = $("form");
  const drawer = $("drawer");
  const spec = {};
  const parameters = {};
  for (const el of [...form.elements, ...drawer.querySelectorAll("input,select")]) {
    if (!el.name) continue;
    // Reward weights are CombatParams fields, so they travel as `parameters`
    // overrides rather than as top-level spec keys. Only send the ones that
    // were actually changed: an override equal to the default still SUPPRESSES
    // a named task, because the task applies its weights with `setdefault`.
    if (el.dataset.reward !== undefined) {
      const value = Number(el.value);
      if (el.value !== "" && value !== Number(el.dataset.reward)) {
        parameters[el.name] = value;
      }
      continue;
    }
    if (el.type === "checkbox") spec[el.name] = el.checked;
    else if (NUMERIC.has(el.name)) spec[el.name] = Number(el.value);
    else spec[el.name] = el.value;
  }
  if (Object.keys(parameters).length) spec.parameters = parameters;
  if (spec.opponent === "") spec.opponent = null;
  // 0 in the box means "no cap". The schema's floor is 1, so it has to become
  // null here or an uncapped run 400s instead of running uncapped.
  if (!spec.episode_ticks) spec.episode_ticks = null;
  // An empty zone means "library default spawn", not the string "". The server
  // rejects a zone on a flat world, so it must be dropped rather than sent.
  if (!spec.zone || spec.world !== "region") spec.zone = null;
  // "" is the no-minigame option; null is what the server means by unshaped.
  // Sending "" would 400 on an unknown name, which is a confusing way to say
  // "you left the picker alone".
  if (!spec.minigame) spec.minigame = null;
  return spec;
}

export function verdict() {
  const DATA = state.data;
  const badge = $("verdictBadge"), sub = $("verdictSub"), block = $("verdictBlock");
  const flags = $("warnings");

  if (!DATA) {
    block.className = "block verdict";
    badge.textContent = "no run yet";
    sub.textContent = "configure above and press Run";
    flags.innerHTML = "";
    return;
  }

  const s = DATA.summary;
  const failures = s.failures || [];
  const warn = DATA.warnings || [];

  let level = "good", label = "clean", why;
  if (failures.length) {
    level = "crit";
    label = "failed";
    why = `${failures.length} environment failure${failures.length > 1 ? "s" : ""} — see What happened`;
  } else if (warn.length) {
    level = "warn";
    label = "degraded";
    why = "ran, but do not read these numbers at face value";
  } else {
    why = `${s.ability_starts ?? 0} ability starts · ${s.landed} landed · ` +
      `reward ${s.reward_total} · ${s.wall_seconds}s compute`;
  }

  block.className = `block verdict sev-${level}`;
  badge.textContent = label;
  sub.textContent = why;
  flags.innerHTML = warn
    .map((w) => `<div class="flag sev-${level === "crit" ? "crit" : "warn"}">${w}</div>`)
    .join("");
}

export function wireLaunch(onLoaded) {
  $("more").addEventListener("click", (e) => {
    const open = $("drawer").hidden;
    $("drawer").hidden = !open;
    e.target.setAttribute("aria-expanded", String(open));
    e.target.textContent = open ? "Close settings" : "Settings";
  });

  const hints = () => {
    const p = Number($("form").period?.value || $("drawer").querySelector("[name=period]")?.value || 0);
    const dp = Number($("dp").value || 1);
    if ($("periodMs")) $("periodMs").textContent = p ? `${(p / TPS * 1000).toFixed(0)} ms` : "";
    if ($("dpMs")) $("dpMs").textContent = `${(dp / TPS * 1000).toFixed(0)} ms`;
    if ($("ticks")) {
      const secs = (Number($("ticks").value) / TPS).toFixed(1);
      $("ticks").title = `${secs} s of simulated time`;
    }
    $("dpHint").className = dp === 1 ? "hint warn" : "hint";
    $("dpHint").textContent = dp === 1
      ? "1 tick = 33 ms. Fine for training, but no human acts on a 33 ms loop — raise to 6–8 before comparing against a person."
      : `${dp} ticks between decisions (${(dp / TPS * 1000).toFixed(0)} ms).`;
  };
  $("form").addEventListener("input", hints);
  $("drawer").addEventListener("input", hints);

  $("form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const go = $("go");
    go.disabled = true;
    $("status").textContent = "compiling…";
    try {
      const launchProfileId = $("profilePick")?.value || "";
      const profileQuery = launchProfileId
        ? `?profile_id=${encodeURIComponent(launchProfileId)}` : "";
      const response = await fetch(`/api/run${profileQuery}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(readForm()),
      });
      if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
      const payload = await response.json();
      frameCamera(payload.trajectory, payload.terrain);
      setData(payload);
      const s = payload.summary;
      state.history.push({
        label: `${payload.spec.loadout} vs ${payload.spec.opponent || "none"} · ` +
          `${payload.spec.policy} p${payload.spec.period} · seed ${payload.spec.seed}`,
        reward_total: s.reward_total, landed: s.landed,
        damage_dealt: s.damage_dealt, damage_taken: s.damage_taken,
      });
      invalidate("history");
      // The run just wrote an artifact; show it without a refetch.
      noteRun(payload.artifact);
      const replay = payload.artifact?.profile_replay;
      const profile = (state.profiles || []).find((item) => item.id === launchProfileId);
      if (profile && replay) {
        profile.replays = (profile.replays || [])
          .filter((item) => item.scenario !== replay.scenario)
          .concat(replay);
        if ($("profilePick").value === launchProfileId) describeAgent(profile);
      }
      $("status").textContent = `done in ${s.wall_seconds}s`;
      onLoaded?.();
    } catch (err) {
      $("status").textContent = `failed: ${err.message}`;
    } finally {
      go.disabled = false;
    }
  });

  $("clearHist").addEventListener("click", (e) => {
    e.preventDefault();
    state.history.length = 0;
    invalidate("history");
  });

  hints();
}

export async function loadOptions() {
  const o = await (await fetch("/api/options")).json();
  const fill = (id, values, blank) => {
    const el = $(id);
    el.innerHTML = (blank ? [`<option value="">none</option>`] : [])
      .concat(values.map((v) => `<option>${v}</option>`)).join("");
  };
  fill("loadout", o.loadouts);
  fill("opponent", o.loadouts, true);
  fill("policy", o.policies);
  fill("world", o.worlds || ["open_flat"]);
  fill("task", o.tasks || ["baseline"]);
  /* A minigame is ADDITIVE, unlike a task: the four native reward terms stay
     intact underneath, so a shaped run is still comparable to an unshaped one
     on those alone. Blank leads, because unshaped is the baseline. */
  syncMinigames(o.minigame_details || []);
  syncRewardTerms(o.reward_terms || [], "rewardTerms");
  /* "first_legal" is the Gym's own default and the only one of four measured to
     land damage -- the others get abilities accepted and deal 0.0. */
  fill("opponentPolicy", o.opponent_policies || ["first_legal"]);

  /* The zone picker is deliberately a short curated list, not all 182 zones.
     Every entry is a strongly-connected component of a region's traversal
     graph, so an agent spawned in one can reach the whole arena -- and every
     entry is in the train split, because heldout regions cannot be selected at
     all and offering one would fail the moment it was chosen. */
  const zones = o.zones || [];
  $("zone").innerHTML = [`<option value="">library default spawn</option>`,
    // One run trains on ONE zone -- the loader takes a scalar region key and
    // spawn node, so every row of the batch is the same place. Rotating on the
    // seed is how a seed sweep becomes a terrain sweep.
    `<option value="rotate">rotate by seed — a different zone per seed</option>`]
    .concat(zones.map((z) => `<option value="${z.id}">${z.label}</option>`))
    .join("");

  const d = o.defaults || {};
  if (d.loadout) $("loadout").value = d.loadout;
  if (d.policy) $("policy").value = d.policy;
  if (d.world) $("world").value = d.world;
  // The opponent select leads with a blank "none", so without this the page
  // opened on a duel with nobody in it -- every defensive number would be zero
  // for the most boring possible reason. RunSpec's default is a real opponent.
  if (d.opponent) $("opponent").value = d.opponent;
  if (d.task) $("task").value = d.task;
  $("opponentPolicy").value = d.opponent_policy || "first_legal";
  syncWorld(zones);
  $("world").addEventListener("change", () => syncWorld(zones));
  return o;
}

/* A minigame pays for something the four native reward terms cannot express.
   Two of them need a world coordinate the picker has no way to choose -- an
   invented one can sit inside terrain, and the agent is then paid for walking
   into a wall -- so those are disabled here rather than offered and 400'd. */
/* The four numbers that ARE the environment's reward. Rendered from what the
 * server serves rather than written into the page, so the names and the
 * defaults cannot drift from `CombatParams`. `data-reward` carries the default
 * so `readForm` can tell an untouched box from a deliberate override. */
export function syncRewardTerms(terms, containerId) {
  const host = $(containerId);
  if (!host) return;
  if (!terms.length) {
    host.innerHTML = `<p class="note">reward terms unavailable</p>`;
    return;
  }
  host.innerHTML = terms.map((t) => `
    <label title="${t.meaning}"><span>${t.name.replace(/_reward_scale$/, "")
      .replace(/_/g, " ")}</span>
      <input type="number" step="0.5" name="${t.name}"
             data-reward="${t.default}" value="${t.default}"></label>`).join("");
}

// Parameterised by element because the Train tab needs the same picker. Two
// copies would drift, and the one that drifts is the one that keeps offering a
// game whose constructor argument the console cannot supply.
export function syncMinigames(details, selectId = "minigame",
                              hintId = "minigameHint") {
  const select = $(selectId);
  if (!select) return;
  select.innerHTML = [`<option value="">none (native reward only)</option>`]
    .concat(details.map((g) => {
      const blocked = (g.needs_options || []).length;
      return `<option value="${g.name}"${blocked ? " disabled" : ""}>`
        + `${g.name}${blocked ? ` — needs ${g.needs_options.join(", ")}` : ""}`
        + `</option>`;
    })).join("");

  // The Train tab reuses the picker without the explanatory line beneath it.
  const hint = hintId ? $(hintId) : null;
  if (!hint) return;
  const byName = new Map(details.map((g) => [g.name, g]));
  /* States what the reward for THIS run actually adds up to. The terms panel
     lists four native numbers and used to claim they were the whole reward,
     which is false the moment a minigame is picked -- its shaped term is added
     on top of them, not instead of them. */
  const scope = (game) => {
    const host = $("rewardScopeHint");
    if (!host) return;
    const weight = Number($("minigameWeight")?.value ?? 1);
    host.textContent = game
      ? `In force for this run: the four native terms below, PLUS the `
        + `${game.name} shaped term at weight ${weight}.`
      : "In force for this run: the four native terms below, and nothing else.";
  };
  const explain = () => {
    const game = byName.get(select.value);
    scope(game);
    if (!game) {
      hint.textContent = "Unshaped — the four native reward terms only. "
        + "This is the baseline every shaped run is read against.";
      return;
    }
    const needs = (game.requires || []).length
      ? ` Needs: ${game.requires.join(", ")}.` : "";
    hint.textContent = `Teaches ${game.teaches}. Tell: ${game.tell}.${needs}`;
  };
  $("minigameWeight")?.addEventListener("input", explain);
  select.addEventListener("change", explain);
  explain();
}

/* Only a Region run has terrain, and only a Region run can take a zone. Saying
   so next to the control beats letting someone pick a zone that is silently
   ignored -- which is what happens when `world` is left flat. */
function syncWorld(zones) {
  const region = $("world").value === "region";
  $("zoneField").hidden = !region;
  if (!region) $("zone").value = "";
  $("worldNote").textContent = region
    ? `Real captured geometry, and the only world the map can draw ground for. `
      + `Costs roughly 2.6x a flat run (measured 56s flat, 149s region on 48 `
      + `ticks). ${zones.length} navigable arenas offered.`
    : "open_flat pins every world query true — the permissive control the Gym's "
      + "own readiness gates use. Nothing to draw, and terrain cannot matter.";
}

export async function loadSurfaces() {
  try {
    const [adk, bridge] = await Promise.all([
      (await fetch("/api/adk")).json(),
      (await fetch("/api/bridge")).json(),
    ]);
    // bridge_status() publishes `any_up`; this read `bridge.reachable`, which
    // is never defined, so the health dot was permanently red.
    const up = !!bridge.any_up;
    const dot = $("health");
    dot.className = `dot ${up ? "ok" : "bad"}`;
    dot.title = up ? "native bridge reachable" : "native bridge not reachable";
    drawContractFacts(adk, bridge);
    $("adkPanel").innerHTML =
      `<div class="readouts">` +
      Object.entries(flatten(adk)).slice(0, 24)
        .map(([k, v]) => `<div class="ro"><i>${k}</i><b>${v}</b></div>`).join("") +
      `</div>`;
  } catch { /* the console must stay usable when a surface is down */ }
}

/* The policy running inside the Java server. It publishes a file every 5s
   rather than serving a port, so this is a plain read with an age on it —
   "stale" is a real state here, not an error, and its numbers still matter. */
export async function loadPolicyAgent() {
  const count = $("policyAgentCount");
  const panel = $("policyAgentPanel");
  if (!count || !panel) return;
  let s;
  try {
    s = await (await fetch("/api/policy-agent")).json();
  } catch {
    return;                       // keep the last good render on a blip
  }

  if (s.state === "absent" || s.state === "unreadable") {
    count.textContent = s.state;
    count.className = "count";
    panel.innerHTML =
      `<div class="readouts"><div class="ro"><i>state</i><b class="absent">${s.state}</b></div></div>` +
      `<p class="hint${s.state === "unreadable" ? " warn" : ""}">${s.note || s.error || ""}</p>`;
    return;
  }

  const m = s.metrics || {};
  const live = s.state === "live";
  count.textContent = live
    ? `live · ${m.claimed ?? 0} NPCs`
    : `stale ${s.age_seconds ?? "?"}s`;

  const n = (v, d = 0) => (v === undefined || v === null ? "—" : Number(v).toFixed(d));
  const rows = [
    ["state", s.state, live ? "pos" : "neg"],
    ["age", `${s.age_seconds ?? "?"}s`, live ? "" : "neg"],
    ["role", m.role ?? "—", ""],
    ["claimed NPCs", n(m.claimed), m.claimed ? "pos" : "absent"],
    ["policy ticks", n(m.policy_ticks), ""],
    ["policy ticks/s", n(m.policy_ticks_per_second, 1), ""],
    ["world tps", n(m.world_tps, 1), m.world_tps >= 29 ? "pos" : "neg"],
    ["world tick", n(m.world_tick), ""],
    ["skipped (no obs)", n(m.skipped_no_observation), m.skipped_no_observation ? "neg" : ""],
    ["illegal", n(m.rejected_illegal), m.rejected_illegal ? "neg" : ""],
    ["world verbs dropped", n(m.dropped_world_verb), m.dropped_world_verb ? "neg" : ""],
    ["interactions skipped", n(m.skipped_interactions), m.skipped_interactions ? "neg" : ""],
    // Motion is attributed only when a decoded locomotion request and a
    // meaningful horizontal path are both present.
    ["locomotion requests", n(m.locomotion_requests), m.locomotion_requests ? "pos" : "absent"],
    ["moved", m.moved ? "YES" : "NO", m.moved ? "pos" : "neg"],
    ["motion samples", n(m.motion_samples), ""],
    ["horiz path", n(m.motion_horizontal_path, 2), ""],
    ["horiz displacement", n(m.motion_horizontal_displacement, 2), ""],
    ["Δ x / y / z", `${n(m.motion_dx, 2)} / ${n(m.motion_dy, 2)} / ${n(m.motion_dz, 2)}`, ""],
    ["observation", n(m.observation_size), ""],
    ["actions", n(m.action_size), ""],
    ["capture non-zero", n(m.capture_non_zero_columns), m.capture_non_zero_columns ? "" : "neg"],
  ];
  if (m.last_skill !== undefined) {
    rows.push(
      ["last skill", n(m.last_skill), ""],
      ["last move", n(m.last_world_move), ""],
      ["last yaw Δ", `${n(m.last_yaw_delta_degrees, 2)}°`, ""],
      ["last jump", m.last_jump ? "YES" : "no", ""],
      ["last attack", m.last_attack ? "YES" : "no", ""],
      ["last dodge", n(m.last_dodge), m.last_dodge ? "pos" : "absent"],
      ["last guard", m.last_guard ? "YES" : "no", ""],
      ["last legal", m.last_action_legal ? "YES" : "NO", m.last_action_legal ? "pos" : "neg"],
      ["last value", n(m.last_value, 4), ""],
    );
  }
  panel.innerHTML =
    `<div class="readouts">` +
    rows.map(([k, v, c]) => `<div class="ro"><i>${k}</i><b class="${c}">${v}</b></div>`).join("") +
    `</div>`;
}

function flatten(obj, prefix = "", out = {}) {
  for (const [k, v] of Object.entries(obj || {})) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === "object" && !Array.isArray(v)) flatten(v, key, out);
    else out[key] = Array.isArray(v) ? `${v.length} items` : String(v);
  }
  return out;
}


/* Agent profiles.
 *
 * A profile says what an *agent* is — architecture, network widths, the
 * contract it was built against, its hyperparameters. It is not a test
 * configuration: one agent gets run against a dozen scenes, and one scene
 * compares a dozen agents, so binding them together makes both harder to vary.
 *
 * `suggested_run` is a courtesy. Selecting a profile shows the agent and, if
 * it offered a starting point, seeds the form with it — but the profile is
 * still about the agent either way.
 */
export async function loadProfiles() {
  const pick = $("profilePick");
  if (!pick) return;
  let profiles = [];
  try {
    profiles = (await (await fetch("/api/profiles")).json()).profiles || [];
  } catch { return; }
  state.profiles = profiles;

  pick.innerHTML = [`<option value="">— no agent —</option>`].concat(
    profiles.filter((p) => !p.error).map(
      (p) => `<option value="${esc(p.id)}">${esc(p.display_title || p.title)} · ${esc(p.architecture)}</option>`)
  ).join("");

  pick.addEventListener("change", () => {
    const profile = profiles.find((p) => p.id === pick.value);
    state.activeProfileId = pick.value;
    invalidate("runs", "jobs");
    describeAgent(profile);
    if (!profile) return;
    const run = profile.suggested_run || {};
    for (const [key, value] of Object.entries(run)) {
      const el = document.querySelector(`[name="${key}"]`);
      if (!el) continue;
      if (el.type === "checkbox") el.checked = !!value;
      else el.value = value;
    }
    // `world` decides whether the zone picker exists at all, so its handler
    // must run before a zone from the profile can be applied.
    $("world").dispatchEvent(new Event("change"));
    if (run.zone) $("zone").value = run.zone;
    $("status").textContent = `agent: ${profile.display_title || profile.title}`;
  });
}

/* What the selected agent is, shown next to the launch form -- a run is only
   interpretable next to the agent that produced it. */
function describeAgent(profile) {
  const box = $("agentCard");
  if (!box) return;
  if (!profile) { box.innerHTML = ""; box.hidden = true; return; }
  const keepOpen = box.dataset.profileId === profile.id && box.open;
  box.dataset.profileId = profile.id;
  box.hidden = false;

  const rows = [["architecture", profile.architecture, ""]];
  if (profile.checkpoint_role) rows.push(["checkpoint role", profile.checkpoint_role, ""]);
  if (profile.selection_status) {
    const tone = profile.selection_status === "promoted" ? "pos" :
      (profile.selection_status === "rejected" ? "neg" : "");
    const status = profile.selection_status === "not_assessed"
      ? "not promoted / not assessed" : profile.selection_status;
    rows.push(["selection", status, tone]);
  }
  if (profile.agent) rows.push(["agent dir", `agents/${profile.agent}`, ""]);
  for (const [k, v] of Object.entries(profile.network || {})) {
    rows.push([`net ${k}`, v, ""]);
  }
  for (const [k, v] of Object.entries(profile.contract || {})) {
    rows.push([`contract ${k}`, v, ""]);
  }
  for (const [k, v] of Object.entries(profile.hyperparameters || {})) {
    rows.push([k.replace(/_/g, " "), v, ""]);
  }
  if (profile.checkpoint) rows.push(["checkpoint", profile.checkpoint, "pos"]);
  if (profile.checkpoint_sha256) rows.push(["checkpoint sha256", profile.checkpoint_sha256, ""]);
  if (profile.selection_evidence) rows.push(["selection evidence", profile.selection_evidence, ""]);

  const replays = profile.replays || [];
  const replayButtons = replays.map((replay) =>
    `<button type="button" class="ghost mini" data-profile-replay="${esc(replay.run_id)}">` +
    `Watch Replay · ${esc(replay.label || replay.scenario)} · ` +
    `${esc(formatTimestamp(replay.recorded_at))}</button>`).join("");

  box.innerHTML =
    `<summary><h2>${esc(profile.display_title || profile.title)}</h2>
       <span class="count">${esc(profile.architecture)} · profile details</span></summary>` +
    `<div class="profileDetailBody"><div class="facts">` + rows.map(([k, v, c]) =>
      `<div class="ro"><i>${esc(k)}</i><b class="${c}">${esc(v)}</b></div>`).join("") +
    `</div>` +
    (replayButtons ? `<div class="profile-replays">${replayButtons}</div>` : "") +
    (profile.notes ? `<p class="note">${esc(profile.notes)}</p>` : "") + `</div>`;
  box.open = keepOpen;
  box.querySelectorAll("button[data-profile-replay]").forEach((button) => {
    button.addEventListener("click", () => replayRun(button.dataset.profileReplay));
  });
}

function formatTimestamp(value) {
  if (!value) return "time unknown";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

function esc(v) {
  return String(v ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
