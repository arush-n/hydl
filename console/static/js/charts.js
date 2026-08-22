/* Plots. One generic renderer; every chart is a call into it.
 *
 * Run charts redraw on `data` and on `frame` (the playhead moves). History
 * charts redraw only on `history`, so scrubbing never touches them.
 */

import { $, roles, state } from "./state.js";
import { css, fit, g2d } from "./gfx.js";

/** `series` is [{values, color, fill?, dots?}]. */
export function plot(id, series, opts = {}) {
  const cv = $(id);
  if (!cv) return;
  const [W, H] = fit(cv);
  const g = g2d(cv);
  g.fillStyle = css("--sunk");
  g.fillRect(0, 0, W, H);

  const all = series.flatMap((s) => s.values);
  if (!all.length) {
    g.fillStyle = css("--ink-3");
    g.font = "12px ui-sans-serif, system-ui, sans-serif";
    g.textAlign = "center";
    g.fillText(opts.empty || "no data yet", W / 2, H / 2);
    g.textAlign = "start";
    return;
  }

  const lo = opts.min !== undefined ? opts.min : Math.min(0, ...all);
  let hi = opts.max !== undefined ? opts.max : Math.max(...all);
  if (hi - lo < 1e-9) hi = lo + 1;
  const pad = 8;
  const X = (i, n) => pad + (i / Math.max(1, n - 1)) * (W - pad * 2);
  const Y = (v) => H - pad - ((v - lo) / (hi - lo)) * (H - pad * 2);

  if (lo < 0 && hi > 0) {
    g.strokeStyle = css("--rule"); g.lineWidth = 1;
    g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(pad, Y(0)); g.lineTo(W - pad, Y(0)); g.stroke();
    g.setLineDash([]);
  }

  for (const s of series) {
    const n = s.values.length;
    if (!n) continue;
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
      s.values.forEach((v, i) => { g.beginPath(); g.arc(X(i, n), Y(v), 2.6, 0, 7); g.fill(); });
    }
  }

  if (opts.playhead !== undefined && all.length > 1) {
    const n = series[0].values.length;
    g.fillStyle = css("--ink");
    g.fillRect(X(Math.min(opts.playhead, n - 1), n) - 0.7, 0, 1.4, H);
  }
}

export function drawRunCharts() {
  const DATA = state.data;
  if (!DATA) {
    ["chReward", "chHealth", "chDist", "chAim", "chStamina"].forEach((id) => plot(id, []));
    return;
  }
  const t = DATA.trajectory, head = state.step;
  const role = roles();
  $("legendLearner").textContent = role.learner;
  $("legendOpponent").textContent = role.opponent;
  let acc = 0;
  plot("chReward", [{ values: t.reward.map((r) => (acc += r)), color: css("--accent"), fill: true }],
    { playhead: head });
  plot("chHealth", [
    { values: t.agent_health, color: css("--agent") },
    { values: t.target_health, color: css("--target") },
  ], { min: 0, playhead: head });

  const anyVisible = t.visible.some(Boolean);
  plot("chDist", [
    { values: t.true_distance, color: css("--ink-3") },
    ...(anyVisible ? [{ values: t.observed_distance, color: css("--ok") }] : []),
  ], { min: 0, playhead: head, empty: "never visible" });
  plot("chAim", [
    ...(t.aim_error ? [{ values: t.aim_error, color: css("--accent") }] : []),
    ...(t.yaw_error ? [{ values: t.yaw_error, color: css("--ink-3") }] : []),
    ...(t.pitch_error ? [{ values: t.pitch_error, color: css("--ok") }] : []),
  ], { min: 0, playhead: head, empty: "3D aim not recorded" });

  const staminaRecorded = t.stamina &&
    (!t.stamina_available || t.stamina_available.some(Boolean));
  plot("chStamina", staminaRecorded ? [{ values: t.stamina, color: css("--info"), fill: true }] : [],
    { playhead: head, empty: "not recorded" });
}

export function drawHistoryCharts() {
  const h = state.history;
  plot("hReward", [{ values: h.map((r) => r.reward_total), color: css("--accent"), dots: true }],
    { empty: "run something" });
  plot("hLanded", [{ values: h.map((r) => r.landed), color: css("--ok"), dots: true }],
    { min: 0, empty: "run something" });
  plot("hDamage", [
    { values: h.map((r) => r.damage_dealt), color: css("--ok"), dots: true },
    { values: h.map((r) => r.damage_taken), color: css("--crit"), dots: true },
  ], { min: 0, empty: "run something" });

  $("histrows").innerHTML = h
    .map((r, i) => `<div class="hrow"><u>${i + 1}</u><span>${r.label}</span>` +
      `<b class="${r.reward_total < 0 ? "neg" : "pos"}">${r.reward_total}</b></div>`)
    .reverse().join("");
  $("histNote").textContent = h.length
    ? `${h.length} run${h.length > 1 ? "s" : ""} this session`
    : "is tuning moving the needle?";
}
