"use strict";
/* HytaleRL debug console.
 *
 * The 3D view is a hand-rolled orbit camera projecting onto Canvas 2D: no
 * external library, so the console works offline and has nothing to vendor.
 * Depth is handled by painter's algorithm -- the scene is a ground plane plus
 * two bodies, so a full z-buffer would be effort spent on nothing.
 */

const $ = (id) => document.getElementById(id);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const TPS = 30;

let DATA = null;      // last /api/run payload
let cur = 0;          // current timestep
let timer = null;

/* ---------------------------------------------------------------- camera */
const cam = { yaw: 0.62, pitch: 0.46, dist: 15, cx: 0, cy: 65, cz: 0 };

function project(p, W, H) {
  const dx = p[0] - cam.cx, dy = p[1] - cam.cy, dz = p[2] - cam.cz;
  const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
  const X = dx * cy - dz * sy;
  const Zr = dx * sy + dz * cy;
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  const Y = dy * cp - Zr * sp;
  let Zc = dy * sp + Zr * cp + cam.dist;
  if (Zc < 0.15) Zc = 0.15;
  const f = H * 0.92;
  return { x: W / 2 + (X * f) / Zc, y: H / 2 - (Y * f) / Zc, z: Zc };
}


/** Aim the camera at the fight. Called once per loaded run.
 *  The scene sits near y=65 in world space and the duel occupies only a few
 *  units of it, so a fixed target leaves the bodies clipped off-frame. */
function frameCamera() {
  if (!DATA) return;
  const t = DATA.trajectory;
  const xs = t.agent_x.concat(t.target_x);
  const zs = t.agent_z.concat(t.target_z);
  const ys = t.agent_y.concat(t.target_y);
  const xlo = Math.min(...xs), xhi = Math.max(...xs);
  const zlo = Math.min(...zs), zhi = Math.max(...zs);
  cam.cx = (xlo + xhi) / 2;
  cam.cz = (zlo + zhi) / 2;
  cam.cy = Math.min(...ys) + 0.9;
  const spread = Math.max(xhi - xlo, zhi - zlo, 3);
  cam.dist = spread * 2.6 + 5;
}

/* ------------------------------------------------------------- rendering */
function fitCanvas(cv) {
  const r = cv.getBoundingClientRect();
  const dpr = Math.min(devicePixelRatio || 1, 2);
  if (cv.width !== Math.round(r.width * dpr) || cv.height !== Math.round(r.height * dpr)) {
    cv.width = Math.round(r.width * dpr);
    cv.height = Math.round(r.height * dpr);
  }
  return [cv.width, cv.height];
}

function drawScene() {
  const cv = $("view");
  const [W, H] = fitCanvas(cv);
  const g = cv.getContext("2d");
  g.clearRect(0, 0, W, H);
  g.fillStyle = css("--sunk");
  g.fillRect(0, 0, W, H);
  if (!DATA) {
    g.fillStyle = css("--ink-3");
    g.font = "14px ui-sans-serif, system-ui, sans-serif";
    g.textAlign = "center";
    g.fillText("Run a rollout to see the agents move.", W / 2, H / 2);
    return;
  }

  const t = DATA.trajectory, i = cur;
  const ay = t.agent_y[i], ty = t.target_y[i];
  const ground = Math.min(...t.agent_y, ...t.target_y);

  /* ground grid, drawn far-to-near */
  const span = Math.max(6, Math.ceil(cam.dist * 0.55));
  g.lineWidth = 1;
  for (let k = -span; k <= span; k++) {
    for (const [a, b] of [
      [[cam.cx + k, ground, cam.cz - span], [cam.cx + k, ground, cam.cz + span]],
      [[cam.cx - span, ground, cam.cz + k], [cam.cx + span, ground, cam.cz + k]],
    ]) {
      const p = project(a, W, H), q = project(b, W, H);
      const major = k % 5 === 0;
      g.strokeStyle = major ? css("--rule") : css("--rule-2");
      g.beginPath(); g.moveTo(p.x, p.y); g.lineTo(q.x, q.y); g.stroke();
    }
  }

  /* trails */
  if ($("trails").checked) {
    const TRAIL = 56, s = Math.max(0, i - TRAIL);
    const trail = (xs, ys, zs, col) => {
      for (let k = s; k < i; k++) {
        const p = project([xs[k], ys[k], zs[k]], W, H);
        const q = project([xs[k + 1], ys[k + 1], zs[k + 1]], W, H);
        g.globalAlpha = ((k - s) / TRAIL) * 0.6;
        g.strokeStyle = col; g.lineWidth = 2.4;
        g.beginPath(); g.moveTo(p.x, p.y); g.lineTo(q.x, q.y); g.stroke();
      }
      g.globalAlpha = 1;
    };
    trail(t.agent_x, t.agent_y, t.agent_z, css("--agent"));
    trail(t.target_x, t.target_y, t.target_z, css("--target"));
  }

  const A = [t.agent_x[i], ay, t.agent_z[i]];
  const T = [t.target_x[i], ty, t.target_z[i]];

  /* sightline -- present only when the policy can actually see */
  if (t.visible[i]) {
    const p = project([A[0], A[1] + 0.9, A[2]], W, H);
    const q = project([T[0], T[1] + 0.9, T[2]], W, H);
    g.strokeStyle = css("--good"); g.globalAlpha = 0.7; g.lineWidth = 1.5;
    g.setLineDash([6, 5]);
    g.beginPath(); g.moveTo(p.x, p.y); g.lineTo(q.x, q.y); g.stroke();
    g.setLineDash([]); g.globalAlpha = 1;
  }

  /* facing wedge on the ground */
  const wedge = (base, yaw, col) => {
    g.fillStyle = col; g.globalAlpha = 0.16;
    g.beginPath();
    const o = project([base[0], ground + 0.02, base[2]], W, H);
    g.moveTo(o.x, o.y);
    for (let d = -0.4; d <= 0.4001; d += 0.05) {
      const p = project(
        [base[0] + Math.sin(yaw + d) * 3.2, ground + 0.02, base[2] + Math.cos(yaw + d) * 3.2],
        W, H);
      g.lineTo(p.x, p.y);
    }
    g.closePath(); g.fill(); g.globalAlpha = 1;
  };
  wedge(T, t.target_yaw[i], css("--target"));
  wedge(A, t.agent_yaw[i], css("--agent"));

  /* bodies, painter-sorted */
  const bodies = [
    { p: T, col: css("--target"), hp: t.target_health[i] / DATA.spec.target_max_health, r: 0.42 },
    { p: A, col: css("--agent"), hp: t.agent_health[i] / DATA.spec.agent_max_health, r: 0.46 },
  ].map((b) => ({ ...b, d: project(b.p, W, H).z })).sort((a, b) => b.d - a.d);

  for (const b of bodies) {
    const foot = project([b.p[0], ground, b.p[2]], W, H);
    const head = project([b.p[0], b.p[1] + 1.7, b.p[2]], W, H);
    const scale = (H * 0.92) / foot.z;

    /* ground shadow */
    g.fillStyle = "rgba(0,0,0,.30)";
    g.beginPath();
    g.ellipse(foot.x, foot.y, b.r * scale * 0.9, b.r * scale * 0.42, 0, 0, 7);
    g.fill();

    /* pillar */
    g.strokeStyle = b.col; g.lineWidth = Math.max(3, b.r * scale * 0.5);
    g.lineCap = "round";
    g.beginPath(); g.moveTo(foot.x, foot.y); g.lineTo(head.x, head.y); g.stroke();
    g.lineCap = "butt";

    /* head + health ring */
    const hr = Math.max(4, b.r * scale * 0.62);
    g.fillStyle = b.col;
    g.beginPath(); g.arc(head.x, head.y, hr, 0, 7); g.fill();
    g.strokeStyle = css("--panel"); g.lineWidth = 2;
    g.beginPath(); g.arc(head.x, head.y, hr, 0, 7); g.stroke();
    g.strokeStyle = b.hp > 0.4 ? css("--good") : css("--crit");
    g.lineWidth = 3;
    g.beginPath();
    g.arc(head.x, head.y, hr + 4, -Math.PI / 2, -Math.PI / 2 + 6.2832 * Math.max(0, b.hp));
    g.stroke();
  }

  /* hit flash */
  if (t.damage_dealt[i] > 0) {
    const q = project([T[0], T[1] + 1.0, T[2]], W, H);
    g.strokeStyle = css("--accent"); g.lineWidth = 3;
    g.beginPath(); g.arc(q.x, q.y, 26, 0, 7); g.stroke();
  }
}

function drawTimeline() {
  const cv = $("timeline");
  const [W, H] = fitCanvas(cv);
  const g = cv.getContext("2d");
  g.clearRect(0, 0, W, H);
  g.fillStyle = css("--panel"); g.fillRect(0, 0, W, H);
  if (!DATA) return;
  const t = DATA.trajectory, N = DATA.steps;

  g.fillStyle = css("--good"); g.globalAlpha = 0.15;
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

  g.fillStyle = css("--ink-3");
  for (let i = 0; i < N; i++) if (t.accepted[i]) g.fillRect((i / N) * W, H - 6, 1.6, 6);
  g.fillStyle = css("--accent");
  for (let i = 0; i < N; i++) if (t.damage_dealt[i] > 0) g.fillRect((i / N) * W - 1, 0, 2.6, H);

  g.fillStyle = css("--ink"); g.fillRect((cur / N) * W - 0.8, 0, 1.6, H);
}

/* -------------------------------------------------------------- readouts */
function paint() {
  if (!DATA) { drawScene(); drawTimeline(); return; }
  const t = DATA.trajectory, i = cur, N = DATA.steps;
  $("stepn").textContent = `step ${i} / ${N - 1} · ${(i / TPS).toFixed(2)}s`;

  const seen = !!t.visible[i];
  let rw = 0; for (let k = 0; k <= i; k++) rw += t.reward[k];
  const rows = [
    ["visible", seen ? "YES" : "NO", seen ? "pos" : "neg"],
    ["observed dist", seen ? t.observed_distance[i].toFixed(3) : "no reading",
      seen ? "" : "absent"],
    ["true dist", t.true_distance[i].toFixed(2), ""],
    ["agent hp", t.agent_health[i].toFixed(0), ""],
    ["target hp", t.target_health[i].toFixed(0), ""],
    ["phase", DATA.target_phases[t.target_phase[i]], ""],
    ["accepted", String(t.accepted[i]), ""],
    ["reward Σ", rw.toFixed(2), rw < 0 ? "neg" : rw > 0 ? "pos" : ""],
  ];
  $("readouts").innerHTML = rows
    .map(([k, v, c]) => `<div class="ro"><i>${k}</i><b class="${c}">${v}</b></div>`)
    .join("");

  $("heads").innerHTML = DATA.head_names
    .map((h, hi) => {
      const v = t.action[i][hi];
      return `<span class="${v ? "" : "z"}"><em>${h.split("_")[0]}</em> ${v}</span>`;
    })
    .join("");

  drawScene();
  drawTimeline();
  drawRunCharts();
}

function showSummary() {
  const s = DATA.summary;
  const cell = (label, value, cls = "") =>
    `<div class="stat"><b>${label}</b><span class="${cls}">${value}</span></div>`;
  $("summary").innerHTML =
    cell("visible", `${s.visible_steps}/${DATA.steps}`,
      s.visible_steps ? "pos" : "neg") +
    cell("accepted", s.accepted) +
    cell("landed", s.landed, s.landed ? "pos" : "neg") +
    cell("hit rate", (s.hit_rate * 100).toFixed(0) + "%") +
    cell("dealt", s.damage_dealt, s.damage_dealt > 0 ? "pos" : "") +
    cell("taken", s.damage_taken, s.damage_taken > 0 ? "neg" : "") +
    cell("reward", s.reward_total, s.reward_total < 0 ? "neg" : s.reward_total > 0 ? "pos" : "") +
    cell("decision", s.decision_interval_ms + " ms") +
    cell("sim time", s.simulated_seconds + " s") +
    cell("compute", s.wall_seconds + " s");

  $("warnings").innerHTML = DATA.warnings
    .map((w) => `<div class="flag${w.startsWith("decision_period") ? " soft" : ""}">${w}</div>`)
    .join("");
}

/* ---------------------------------------------------------------- charts */
const HISTORY = [];

/** Minimal line/area plot. `series` is [{values, color, max?, fill?}]. */
function plot(id, series, opts = {}) {
  const cv = $(id);
  const [W, H] = fitCanvas(cv);
  const g = cv.getContext("2d");
  g.clearRect(0, 0, W, H);
  g.fillStyle = css("--sunk"); g.fillRect(0, 0, W, H);
  const all = series.flatMap((s) => s.values);
  if (!all.length) {
    g.fillStyle = css("--ink-3");
    g.font = "12px ui-sans-serif, system-ui, sans-serif";
    g.textAlign = "center";
    g.fillText(opts.empty || "no data yet", W / 2, H / 2);
    return;
  }
  let lo = opts.min !== undefined ? opts.min : Math.min(0, ...all);
  let hi = opts.max !== undefined ? opts.max : Math.max(...all);
  if (hi - lo < 1e-9) hi = lo + 1;
  const pad = 8;
  const X = (i, n) => pad + (i / Math.max(1, n - 1)) * (W - pad * 2);
  const Y = (v) => H - pad - ((v - lo) / (hi - lo)) * (H - pad * 2);

  /* zero rule where the range crosses it */
  if (lo < 0 && hi > 0) {
    g.strokeStyle = css("--rule"); g.lineWidth = 1;
    g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(pad, Y(0)); g.lineTo(W - pad, Y(0)); g.stroke();
    g.setLineDash([]);
  }

  for (const s of series) {
    const n = s.values.length;
    if (s.fill) {
      g.fillStyle = s.color; g.globalAlpha = 0.14;
      g.beginPath(); g.moveTo(X(0, n), Y(Math.max(lo, 0)));
      s.values.forEach((v, i) => g.lineTo(X(i, n), Y(v)));
      g.lineTo(X(n - 1, n), Y(Math.max(lo, 0))); g.closePath(); g.fill();
      g.globalAlpha = 1;
    }
    g.strokeStyle = s.color; g.lineWidth = 1.8;
    g.beginPath();
    s.values.forEach((v, i) => (i ? g.lineTo(X(i, n), Y(v)) : g.moveTo(X(i, n), Y(v))));
    g.stroke();
    if (s.dots || n <= 24) {
      g.fillStyle = s.color;
      s.values.forEach((v, i) => {
        g.beginPath(); g.arc(X(i, n), Y(v), 2.6, 0, 7); g.fill();
      });
    }
  }
  /* playhead for per-run charts */
  if (opts.playhead !== undefined && all.length > 1) {
    const n = series[0].values.length;
    g.fillStyle = css("--ink");
    g.fillRect(X(Math.min(opts.playhead, n - 1), n) - 0.7, 0, 1.4, H);
  }
}

function drawRunCharts() {
  if (!DATA) {
    ["chReward", "chDist", "chHealth"].forEach((id) => plot(id, [], {}));
    return;
  }
  const t = DATA.trajectory;
  let acc = 0;
  const cumulative = t.reward.map((r) => (acc += r));
  plot("chReward", [{ values: cumulative, color: css("--accent"), fill: true }],
    { playhead: cur });
  /* observed distance is only meaningful where the target is visible */
  const observed = t.observed_distance.map((v, i) => (t.visible[i] ? v : null))
    .filter((v) => v !== null);
  plot("chDist", [
    { values: t.true_distance, color: css("--ink-3") },
    ...(observed.length ? [{ values: t.observed_distance, color: css("--good") }] : []),
  ], { min: 0, playhead: cur });
  plot("chHealth", [
    { values: t.agent_health, color: css("--agent") },
    { values: t.target_health, color: css("--target") },
  ], { min: 0, playhead: cur });
}

function drawHistoryCharts() {
  const h = HISTORY;
  plot("hReward", [{ values: h.map((r) => r.reward_total), color: css("--accent"), dots: true }],
    { empty: "run something" });
  plot("hLanded", [{ values: h.map((r) => r.landed), color: css("--good"), dots: true }],
    { min: 0, empty: "run something" });
  plot("hDamage", [
    { values: h.map((r) => r.damage_dealt), color: css("--good"), dots: true },
    { values: h.map((r) => r.damage_taken), color: css("--crit"), dots: true },
  ], { min: 0, empty: "run something" });

  $("histrows").innerHTML = h
    .map((r, i) =>
      `<div class="hrow"><u>${i + 1}</u><span>${r.label}</span>` +
      `<b class="${r.reward_total < 0 ? "neg" : "pos"}">${r.reward_total}</b></div>`)
    .reverse()
    .join("");
  $("histNote").textContent = h.length
    ? `— ${h.length} run${h.length > 1 ? "s" : ""} this session`
    : "— is tuning moving the needle?";
}

function recordHistory() {
  const s = DATA.summary, p = DATA.spec;
  HISTORY.push({
    label: `${p.loadout} vs ${p.opponent || "none"} · ${p.policy} p${p.period} · dp${p.decision_period} · seed ${p.seed}`,
    reward_total: s.reward_total,
    landed: s.landed,
    damage_dealt: s.damage_dealt,
    damage_taken: s.damage_taken,
  });
  drawHistoryCharts();
}

/* ------------------------------------------------------------- ADK panel */
async function loadSurfaces() {
  try {
    const [a, b] = await Promise.all([
      (await fetch("/api/adk")).json(),
      (await fetch("/api/bridge")).json(),
    ]);
    $("statusbar").innerHTML =
      `<span class="chip up"><i></i><b>ADK</b> ${a.action.logits} logits · ` +
      `${a.observation.named_columns} columns · ${a.observation.group_count} groups</span>` +
      b.listeners.map((l) =>
        `<span class="chip ${l.listening ? "up" : "down"}"><i></i><b>${l.role}</b> ` +
        `:${l.port} ${l.listening ? "up" : "down"}</span>`).join("") +
      `<span class="chip"><b>loadouts</b> ${a.loadouts.length}</span>`;

    $("adkPanel").innerHTML =
      `<h2>ADK surface <em>read from the library, not restated here</em></h2>` +
      `<div class="adkgrid">` +
      a.scenes.map((s) =>
        `<div><b>${s.name}</b><code>${s.contract_sha256}…</code><br>` +
        `<code>${s.active_providers.length} provider(s) bound</code></div>`).join("") +
      `<div><b>provider seams</b><code>${a.provider_seams.length} total</code></div>` +
      `<div><b>action heads</b><code>${a.action.head_count} heads / ${a.action.logits} logits</code></div>` +
      `<div><b>observation</b><code>${a.observation.named_columns} named columns</code></div>` +
      `<div><b>combat params</b><code>${a.combat_parameters} fields</code></div>` +
      `</div>`;
  } catch (err) {
    $("statusbar").innerHTML = `<span class="chip down"><i></i>surface unavailable</span>`;
  }
}

/* ------------------------------------------------------------------ wire */
async function boot() {
  const o = await (await fetch("/api/options")).json();
  const ld = $("loadout"), op = $("opponent"), pol = $("policy");
  for (const n of o.loadouts) {
    ld.insertAdjacentHTML("beforeend", `<option${n === "iron_sword" ? " selected" : ""}>${n}</option>`);
    op.insertAdjacentHTML("beforeend", `<option${n === "iron_sword" ? " selected" : ""}>${n}</option>`);
  }
  op.insertAdjacentHTML("afterbegin", `<option value="">— none —</option>`);
  for (const p of o.policies) {
    pol.insertAdjacentHTML("beforeend",
      `<option${p === "pulse_ability" ? " selected" : ""}>${p}</option>`);
  }
  syncHints();
}

function syncHints() {
  const dp = +$("dp").value || 1;
  const ms = (dp / TPS) * 1000;
  $("dpMs").textContent = `${ms.toFixed(1)} ms between decisions at ${TPS} TPS`;
  const hint = $("dpHint");
  if (dp === 1) {
    hint.className = "hint warn";
    hint.textContent =
      "Acting every tick is a 33 ms loop. Symmetric in training, but no person " +
      "reacts that fast — raise to 6–8 ticks before comparing against a human.";
  } else if (dp <= 10) {
    hint.className = "hint";
    hint.textContent = "Roughly human reaction latency (200–266 ms at 6–8 ticks).";
  } else {
    hint.className = "hint";
    hint.textContent = "Slower than a person; useful for coarse strategy tests.";
  }
  const per = +document.querySelector("[name=period]").value || 1;
  $("periodMs").textContent = `${((per / TPS) * 1000).toFixed(0)} ms between requests`;
  const tk = +$("ticks").value || 1;
  $("ticksSec").textContent = `${(tk / TPS).toFixed(1)} s of simulated time`;
}

$("form").addEventListener("input", syncHints);

$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const body = {
    loadout: fd.get("loadout"),
    opponent: fd.get("opponent") || null,
    armed: fd.has("armed"),
    perceive: fd.has("perceive"),
    target_active: fd.has("target_active"),
    policy: fd.get("policy"),
    slot: +fd.get("slot"),
    period: +fd.get("period"),
    ticks: +fd.get("ticks"),
    seed: +fd.get("seed"),
    decision_period: +fd.get("decision_period"),
    agent_max_health: +fd.get("agent_max_health"),
    target_max_health: +fd.get("target_max_health"),
  };
  $("go").disabled = true;
  $("status").textContent = "compiling and rolling out…";
  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    DATA = await res.json();
    cur = 0;
    frameCamera();
    $("scrub").max = DATA.steps - 1;
    $("scrub").value = 0;
    showSummary();
    recordHistory();
    paint();
    $("status").textContent = `done in ${DATA.summary.wall_seconds}s`;
  } catch (err) {
    $("status").textContent = "failed: " + err.message;
  } finally {
    $("go").disabled = false;
  }
});

$("scrub").addEventListener("input", (e) => { cur = +e.target.value; paint(); });
$("trails").addEventListener("change", paint);
$("play").addEventListener("click", () => {
  if (timer) { clearInterval(timer); timer = null; $("play").textContent = "Play"; return; }
  if (!DATA) return;
  $("play").textContent = "Pause";
  timer = setInterval(() => {
    cur = (cur + 1) % DATA.steps;
    $("scrub").value = cur;
    paint();
  }, 1000 / TPS);
});

/* orbit */
let drag = null;
$("view").addEventListener("pointerdown", (e) => {
  drag = { x: e.clientX, y: e.clientY };
  $("view").setPointerCapture(e.pointerId);
});
$("view").addEventListener("pointermove", (e) => {
  if (!drag) return;
  cam.yaw += (e.clientX - drag.x) * 0.008;
  cam.pitch = Math.max(-0.2, Math.min(1.35, cam.pitch + (e.clientY - drag.y) * 0.006));
  drag = { x: e.clientX, y: e.clientY };
  drawScene();
});
addEventListener("pointerup", () => { drag = null; });
$("view").addEventListener("wheel", (e) => {
  e.preventDefault();
  cam.dist = Math.max(4, Math.min(60, cam.dist * (1 + Math.sign(e.deltaY) * 0.11)));
  drawScene();
}, { passive: false });

addEventListener("resize", paint);
matchMedia("(prefers-color-scheme:dark)").addEventListener("change", paint);

$("clearHist").addEventListener("click", () => {
  HISTORY.length = 0;
  drawHistoryCharts();
});

loadSurfaces();
setInterval(loadSurfaces, 15000);
boot().then(() => { drawHistoryCharts(); paint(); });
