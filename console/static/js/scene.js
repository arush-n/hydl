/* The map: terrain, both actors, facing, sightline, and now the mechanics that
 * were previously invisible -- guard, ability execution and projectiles.
 *
 * Hand-rolled orbit camera onto Canvas 2D: no library, nothing to vendor, works
 * offline. Region runs include the exposed shell of the native block capture;
 * painter order is sufficient for the bounded local patch.
 */

import { $, roles, state } from "./state.js";
import { cam, css, fit, g2d, lambert, project } from "./gfx.js";

const TRAIL = 56;
const TOP = 1, X_NEG = 2, X_POS = 4, Z_NEG = 8, Z_POS = 16;

export function drawScene() {
  drawReplayScene("view", state.data, state.step, { interactive: true });
}

/** Draw the same 3D replay surface in Run and the compact Train preview. */
/* `follow` recentres on the two actors at the CURRENT step instead of leaving
   the camera where `frameCamera` parked it.
 *
 * `frameCamera` runs once, on load, and frames the whole episode's bounding
 * box. That is right for the Run tab, where you orbit a finished fight. It is
 * wrong for a small preview of a 256-tick episode: the actors travel away from
 * the episode centroid and end up hugging an edge, which reads as the view
 * being off-centre. Following costs the ability to orbit, which a preview card
 * does not offer anyway. */
export function drawReplayScene(canvasId, DATA, step = 0, options = {}) {
  if (!options.follow) return paintReplayScene(canvasId, DATA, step, options);
  /* `follow` assigns the shared `cam` singleton, which the Run tab's
     interactive map reads as well. The preview redraws on every live job
     event, so without this the camera is yanked back out from under anyone
     dragging that map -- orbit survived, pan and zoom did not. Restore only
     what follow writes; it never sets yaw or pitch. */
  const held = { cx: cam.cx, cy: cam.cy, cz: cam.cz, dist: cam.dist };
  try {
    return paintReplayScene(canvasId, DATA, step, options);
  } finally {
    Object.assign(cam, held);
  }
}

function paintReplayScene(
  canvasId, DATA, step = 0, { interactive = false, follow = false } = {},
) {
  const cv = $(canvasId);
  if (!cv) return;
  const [W, H] = fit(cv);
  const g = g2d(cv);
  g.fillStyle = css("--sunk");
  g.fillRect(0, 0, W, H);

  if (!DATA) {
    g.fillStyle = css("--ink-3");
    g.font = "14px ui-sans-serif, system-ui, sans-serif";
    g.textAlign = "center";
    g.fillText(
      interactive ? "Run a rollout to see the map." : "Waiting for an evaluation replay.",
      W / 2, H / 2,
    );
    g.textAlign = "start";
    return;
  }

  const t = DATA.trajectory;
  const i = Math.max(0, Math.min(Number(step) || 0, DATA.steps - 1));
  const ay = t.agent_y[i], ty = t.target_y[i];
  const ground = Math.min(...t.agent_y, ...t.target_y);

  if (follow) {
    cam.cx = (t.agent_x[i] + t.target_x[i]) / 2;
    cam.cz = (t.agent_z[i] + t.target_z[i]) / 2;
    cam.cy = Math.min(ay, ty) + 0.9;
    // Pull back with the gap so both stay framed, with a floor so a contact
    // range separation does not put the camera inside the ground.
    // Same bounds as `frameCamera`, so a card that follows and a view that
    // frames once agree on how close "watching this fight" is.
    const gap = Math.hypot(
      t.agent_x[i] - t.target_x[i], t.agent_z[i] - t.target_z[i],
    );
    cam.dist = Math.min(48, Math.max(14, gap * 2.2 + 6));
  }

  const terrain = DATA.terrain;
  const showTerrain = terrain && (!interactive || $("terrainOn")?.checked);
  // The grid is a stand-in for ground. With real ground it is just a mesh
  // drawn over the terrain, so it gives way rather than competing.
  if ((!interactive || $("showGrid")?.checked) && !showTerrain) drawGround(g, W, H, ground);
  if (showTerrain) drawTerrain(g, W, H, terrain);
  if (interactive) noteTerrain(DATA, showTerrain);

  if (!interactive || $("trails")?.checked) {
    trail(g, W, H, t.agent_x, t.agent_y, t.agent_z, css("--agent"), i);
    trail(g, W, H, t.target_x, t.target_y, t.target_z, css("--target"), i);
  }

  const A = [t.agent_x[i], ay, t.agent_z[i]];
  const T = [t.target_x[i], ty, t.target_z[i]];

  /* sightline -- present only when the policy can actually see */
  if (t.visible[i]) {
    const p = project([A[0], A[1] + 0.9, A[2]], W, H);
    const q = project([T[0], T[1] + 0.9, T[2]], W, H);
    g.strokeStyle = css("--ok"); g.globalAlpha = 0.7; g.lineWidth = 1.5;
    g.setLineDash([6, 5]);
    g.beginPath(); g.moveTo(p.x, p.y); g.lineTo(q.x, q.y); g.stroke();
    g.setLineDash([]); g.globalAlpha = 1;
  }

  wedge(g, W, H, T, t.target_yaw[i], css("--target"));
  wedge(g, W, H, A, t.agent_yaw[i], css("--agent"));

  if (t.agent_pitch) {
    const yaw = t.agent_yaw[i] * Math.PI / 180;
    const pitch = t.agent_pitch[i] * Math.PI / 180;
    const origin = [A[0], A[1] + 1.6, A[2]];
    const length = 4;
    const look = [
      origin[0] - Math.sin(yaw) * Math.cos(pitch) * length,
      origin[1] + Math.sin(pitch) * length,
      origin[2] - Math.cos(yaw) * Math.cos(pitch) * length,
    ];
    const p = project(origin, W, H), q = project(look, W, H);
    g.strokeStyle = css("--accent"); g.lineWidth = 2; g.globalAlpha = 0.9;
    g.beginPath(); g.moveTo(p.x, p.y); g.lineTo(q.x, q.y); g.stroke();
    g.globalAlpha = 1;
  }

  // Imported policy replays may not own the launch form's health knobs.  Their
  // trajectory is still authoritative, so derive a safe display maximum from
  // the observed episode instead of turning the health-bar geometry into NaN.
  const targetMaxHealth = Number(DATA.spec?.target_max_health) ||
    Math.max(1, ...t.target_health);
  const agentMaxHealth = Number(DATA.spec?.agent_max_health) ||
    Math.max(1, ...t.agent_health);
  const role = roles();
  const bodies = [
    { p: T, col: css("--target"), hp: t.target_health[i] / targetMaxHealth,
      r: 0.42, label: role.opponent.toUpperCase(), side: 1 },
    { p: A, col: css("--agent"), hp: t.agent_health[i] / agentMaxHealth,
      r: 0.46, label: role.learner.toUpperCase(), side: -1 },
  ].map((b) => ({ ...b, d: project(b.p, W, H).z })).sort((a, b) => b.d - a.d);

  for (const b of bodies) body(g, W, H, b);

  /* ── mechanics overlays: what the agent is DOING, not just where it is ── */
  const scaleAt = (p) => (H * 0.92) / project(p, W, H).z;

  if (t.guard_active && t.guard_active[i]) {
    const q = project([A[0], A[1] + 1.0, A[2]], W, H);
    const r = Math.max(10, 0.6 * scaleAt(A));
    g.strokeStyle = css("--info"); g.lineWidth = 3; g.globalAlpha = 0.9;
    g.beginPath(); g.arc(q.x, q.y, r, -2.4, 0.7); g.stroke();
    g.globalAlpha = 1;
  }

  if (t.active_slot && t.active_slot[i] >= 0) {
    const q = project([A[0], A[1] + 1.0, A[2]], W, H);
    const r = Math.max(12, 0.75 * scaleAt(A));
    const fresh = t.ability_start && t.ability_start[i];
    g.strokeStyle = css("--accent");
    g.lineWidth = fresh ? 3.5 : 1.6;
    g.globalAlpha = fresh ? 1 : 0.55;
    g.beginPath(); g.arc(q.x, q.y, r, 0, 7); g.stroke();
    g.globalAlpha = 1;
    g.fillStyle = css("--accent");
    g.font = "600 11px ui-monospace, Menlo, monospace";
    g.fillText(`slot ${t.active_slot[i]}`, q.x + r + 5, q.y - r * 0.4);
  }

  if (t.projectiles && t.projectiles[i] > 0) {
    /* No per-projectile position is published by the environment, so show the
       live count on the sightline rather than inventing coordinates. */
    const mid = project([(A[0] + T[0]) / 2, (A[1] + T[1]) / 2 + 1.1, (A[2] + T[2]) / 2], W, H);
    g.fillStyle = css("--warn");
    g.beginPath(); g.arc(mid.x, mid.y, 5, 0, 7); g.fill();
    g.font = "600 11px ui-monospace, Menlo, monospace";
    g.fillText(`×${t.projectiles[i]} in flight`, mid.x + 9, mid.y + 4);
  }

  if (t.damage_dealt[i] > 0) {
    const q = project([T[0], T[1] + 1.0, T[2]], W, H);
    g.strokeStyle = css("--ok"); g.lineWidth = 3;
    g.beginPath(); g.arc(q.x, q.y, 26, 0, 7); g.stroke();
  }

  if (interactive) hud(t, i);
}

/* Real captured ground, when the run had any.
 *
 * A flat scene has no geometry -- `open_flat` pins every world query true, so
 * there is genuinely nothing under the actors and the abstract grid is the
 * honest picture. A Region run ships the standable nodes near the fight, and
 * the difference matters: "walked into a wall" and "stopped for no reason" look
 * identical on a grid.
 *
 * New payloads carry exposed occupied cells and face masks, so cliffs,
 * overhangs and walls remain solid. Legacy artifacts fall back to their stored
 * standable tops rather than failing replay.
 */
function drawTerrain(g, W, H, terrain) {
  const cells = terrain.blocks?.visible ? terrain.blocks : terrain;
  const n = cells.x.length;
  const pts = new Array(n);
  const order = new Array(n);
  for (let k = 0; k < n; k++) {
    pts[k] = project([cells.x[k], cells.y[k], cells.z[k]], W, H);
    order[k] = k;
  }

  if (cam.mode === "2d") {
    const ys = cells.y;
    const lo = Math.min(...ys), hi = Math.max(...ys);
    const span = Math.max(1e-6, hi - lo);
    const size = Math.max(2, (pts[0]?.scale || 6) * 1.02);
    order.sort((a, b) => ys[a] - ys[b]);
    for (const k of order) {
      if (cells.faces && !(cells.faces[k] & TOP)) continue;
      const p = pts[k];
      if (p.x < -size || p.x > W + size || p.y < -size || p.y > H + size) continue;
      g.fillStyle = shade(terrain.colors[cells.material[k]] || "#5A6472",
                          0.62 + 0.55 * ((ys[k] - lo) / span));
      g.fillRect(p.x - size / 2, p.y - size / 2, size, size);
    }
    return;
  }
  // Painter's algorithm: far blocks first, so near ones occlude them. A cube
  // has real volume, so unlike the old flat markers the order is load-bearing.
  order.sort((a, b) => pts[b].z - pts[a].z);

  const coarseShape = !cells.faces;
  const voxel = Number(cells.voxel_size || terrain.voxel_size || 1);
  const half = coarseShape ? 0.72 : voxel / 2;
  for (const k of order) {
    const p = pts[k];
    // Cull what is behind the camera, and what is so close it fills the frame.
    // A block two units away projects larger than the viewport and hides the
    // fight behind it, which is what a near-clip plane exists to prevent.
    if (p.z <= 2.0) continue;
    const s = (H * 0.92) / p.z;
    if (s > H * 0.55) continue;
    if (p.x < -s || p.x > W + s || p.y < -s || p.y > H + s) continue;

    const [x, y, z] = [cells.x[k], cells.y[k], cells.z[k]];
    const t0 = project([x - half, y, z - half], W, H);
    const t1 = project([x + half, y, z - half], W, H);
    const t2 = project([x + half, y, z + half], W, H);
    const t3 = project([x - half, y, z + half], W, H);
    const bottom = y - (coarseShape ? 1.35 : voxel);
    const b0 = project([x - half, bottom, z - half], W, H);
    const b1 = project([x + half, bottom, z - half], W, H);
    const b2 = project([x + half, bottom, z + half], W, H);
    const b3 = project([x - half, bottom, z + half], W, H);

    const base = terrain.colors[cells.material[k]] || "#5A6472";
    const mask = cells.faces?.[k] ?? (TOP | X_NEG | X_POS | Z_NEG | Z_POS);
    const towardX = -Math.sin(cam.yaw), towardZ = -Math.cos(cam.yaw);
    if (towardX < 0 && (mask & X_NEG))
      face(g, [t0, t3, b3, b0], shade(base, lambert([-1, 0, 0])), true);
    if (towardX >= 0 && (mask & X_POS))
      face(g, [t2, t1, b1, b2], shade(base, lambert([1, 0, 0])), true);
    if (towardZ < 0 && (mask & Z_NEG))
      face(g, [t1, t0, b0, b1], shade(base, lambert([0, 0, -1])), true);
    if (towardZ >= 0 && (mask & Z_POS))
      face(g, [t3, t2, b2, b3], shade(base, lambert([0, 0, 1])), true);
    if (mask & TOP)
      face(g, [t0, t1, t2, t3], shade(base, lambert([0, 1, 0])), true);
  }
  g.globalAlpha = 1;
}

function face(g, points, fill, outline = false) {
  g.fillStyle = fill;
  g.beginPath();
  g.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) g.lineTo(points[i].x, points[i].y);
  g.closePath();
  g.fill();
  if (outline) {
    g.strokeStyle = "rgba(9,14,17,.14)";
    g.lineWidth = 0.35;
    g.stroke();
  }
}

/* The palette arrives as `hsl(H 34% 46%)`. Scaling lightness in HSL keeps each
   material's hue exactly, so a face reads as the same block in shadow -- which
   is the whole point of colouring by material identity. */
function shade(color, factor) {
  const m = /hsl\(\s*([\d.]+)\s+([\d.]+)%\s+([\d.]+)%\s*\)/.exec(color);
  if (!m) return color;
  const saturation = Math.max(12, parseFloat(m[2]) * 0.62);
  const light = Math.max(6, Math.min(92, parseFloat(m[3]) * factor));
  return `hsl(${m[1]} ${saturation.toFixed(1)}% ${light.toFixed(1)}%)`;
}

function noteTerrain(DATA, showing) {
  const el = $("terrainNote");
  if (!el) return;
  const terrain = DATA.terrain;
  const seed = terrain?.seed ?? DATA.spec?.seed ?? DATA.seed;
  const hasSeed = seed !== undefined && seed !== null && seed !== "";
  const seedButton = $("terrainSeed");
  if (seedButton) {
    seedButton.hidden = !hasSeed;
    seedButton.dataset.seed = hasSeed ? seed : "";
    seedButton.textContent = hasSeed ? `seed ${seed} → Custom` : "";
  }
  if (!terrain) {
    el.hidden = false;
    el.innerHTML = DATA.spec?.world === "region"
      ? "region run, but no captured ground near the fight"
      : "flat scene — no terrain to draw";
    return;
  }
  el.hidden = !showing;
  const thin = terrain.thinned_by > 1 ? ` · sampled 1:${terrain.thinned_by}` : "";
  const blocks = terrain.blocks;
  const blockCount = blocks
    ? `${blocks.visible.toLocaleString()} terrain blocks` +
      (blocks.available > blocks.visible ? ` of ${blocks.available.toLocaleString()}` : "")
    : `${terrain.nodes} shape blocks`;

  const identity = terrain.semantic_sha256
    ? ` · ${terrain.semantic_sha256.slice(0, 10)}` : "";
  el.textContent = `3D terrain ${hasSeed ? seed : "seed unknown"}${identity} · ${blockCount} · ${terrain.colors.length} materials${thin}`;
  el.title = `${terrain.nodes} standable cells. Colour is stable material identity, not a block name.`;
}

function drawGround(g, W, H, ground) {
  const span = Math.max(6, Math.ceil(cam.dist * 0.55));
  g.lineWidth = 1;
  for (let k = -span; k <= span; k++) {
    for (const [a, b] of [
      [[cam.cx + k, ground, cam.cz - span], [cam.cx + k, ground, cam.cz + span]],
      [[cam.cx - span, ground, cam.cz + k], [cam.cx + span, ground, cam.cz + k]],
    ]) {
      const p = project(a, W, H), q = project(b, W, H);
      g.strokeStyle = k % 5 === 0 ? css("--rule") : css("--rule-2");
      g.beginPath(); g.moveTo(p.x, p.y); g.lineTo(q.x, q.y); g.stroke();
    }
  }
}

function trail(g, W, H, xs, ys, zs, col, i) {
  const s = Math.max(0, i - TRAIL);
  for (let k = s; k < i; k++) {
    const p = project([xs[k], ys[k], zs[k]], W, H);
    const q = project([xs[k + 1], ys[k + 1], zs[k + 1]], W, H);
    const alpha = ((k - s) / TRAIL) * 0.82;
    g.globalAlpha = Math.max(0.18, alpha);
    g.strokeStyle = "rgba(3,8,11,.92)";
    g.lineWidth = 5.5;
    g.beginPath(); g.moveTo(p.x, p.y); g.lineTo(q.x, q.y); g.stroke();
    g.globalAlpha = alpha;
    g.strokeStyle = col;
    g.lineWidth = 2.6;
    g.beginPath(); g.moveTo(p.x, p.y); g.lineTo(q.x, q.y); g.stroke();
  }
  g.globalAlpha = 1;
}

function wedge(g, W, H, base, yawDegrees, col) {
  // Trajectory yaw is degrees, and forward is -sin/-cos as the engine's own
  // facing convention has it -- the same one the look ray below uses.
  const yaw = yawDegrees * Math.PI / 180;
  g.fillStyle = col; g.globalAlpha = 0.16;
  g.beginPath();
  const o = project([base[0], base[1] + 0.02, base[2]], W, H);
  g.moveTo(o.x, o.y);
  for (let d = -0.4; d <= 0.4001; d += 0.05) {
    const p = project(
      [base[0] - Math.sin(yaw + d) * 3.2, base[1] + 0.02, base[2] - Math.cos(yaw + d) * 3.2],
      W, H);
    g.lineTo(p.x, p.y);
  }
  g.closePath(); g.fill(); g.globalAlpha = 1;
}

function body(g, W, H, b) {
  const foot = project(b.p, W, H);
  const head = project([b.p[0], b.p[1] + 1.7, b.p[2]], W, H);
  const scale = (H * 0.92) / foot.z;
  const ui = Math.min(devicePixelRatio || 1, 2);

  g.fillStyle = "rgba(0,0,0,.58)";
  g.beginPath();
  g.ellipse(foot.x, foot.y, b.r * scale * 1.18, b.r * scale * 0.56, 0, 0, 7);
  g.fill();
  g.strokeStyle = "rgba(255,255,255,.92)";
  g.lineWidth = 1.5 * ui;
  g.beginPath();
  g.ellipse(foot.x, foot.y, b.r * scale, b.r * scale * 0.46, 0, 0, 7);
  g.stroke();

  g.lineCap = "round";
  g.strokeStyle = "rgba(3,8,11,.96)";
  g.lineWidth = Math.max(8 * ui, b.r * scale * 0.82);
  g.beginPath(); g.moveTo(foot.x, foot.y); g.lineTo(head.x, head.y); g.stroke();
  g.strokeStyle = b.col;
  g.lineWidth = Math.max(4 * ui, b.r * scale * 0.48);
  g.beginPath(); g.moveTo(foot.x, foot.y); g.lineTo(head.x, head.y); g.stroke();
  g.lineCap = "butt";

  const hr = Math.max(5 * ui, b.r * scale * 0.66);
  g.fillStyle = "rgba(3,8,11,.98)";
  g.beginPath(); g.arc(head.x, head.y, hr + 4 * ui, 0, 7); g.fill();
  g.fillStyle = b.col;
  g.beginPath(); g.arc(head.x, head.y, hr, 0, 7); g.fill();
  g.strokeStyle = "white"; g.lineWidth = 1.5 * ui;
  g.beginPath(); g.arc(head.x, head.y, hr, 0, 7); g.stroke();
  g.strokeStyle = "rgba(3,8,11,.96)";
  g.lineWidth = 6 * ui;
  g.beginPath();
  g.arc(head.x, head.y, hr + 5 * ui, -Math.PI / 2,
        -Math.PI / 2 + 6.2832 * Math.max(0, b.hp));
  g.stroke();
  g.strokeStyle = b.hp > 0.4 ? css("--ok") : css("--crit");
  g.lineWidth = 3 * ui;
  g.beginPath();
  g.arc(head.x, head.y, hr + 5 * ui, -Math.PI / 2,
        -Math.PI / 2 + 6.2832 * Math.max(0, b.hp));
  g.stroke();

  g.font = `700 ${10 * ui}px ui-monospace, Menlo, monospace`;
  const pad = 4 * ui, labelWidth = g.measureText(b.label).width + pad * 2;
  const labelX = b.side < 0
    ? head.x - hr - 7 * ui - labelWidth
    : head.x + hr + 7 * ui;
  const labelY = head.y - 7 * ui;
  g.fillStyle = "rgba(3,8,11,.92)";
  g.fillRect(labelX, labelY - 10 * ui, labelWidth, 15 * ui);
  g.strokeStyle = b.col;
  g.lineWidth = 1.5 * ui;
  g.strokeRect(labelX, labelY - 10 * ui, labelWidth, 15 * ui);
  g.fillStyle = "white";
  g.textBaseline = "middle";
  g.fillText(b.label, labelX + pad, labelY - 2.5 * ui);
  g.textBaseline = "alphabetic";
}

function hud(t, i) {
  const rows = [];
  if (!t.visible[i]) rows.push(["target not visible", "warn"]);
  if (t.pitch_error && t.pitch_error[i] > 15) rows.push(["vertical aim off target", "warn"]);
  if (t.active_slot && t.active_slot[i] >= 0) rows.push([`ability slot ${t.active_slot[i]}`, "info"]);
  if (t.guard_active && t.guard_active[i]) rows.push(["guarding", "info"]);
  const staminaAvailable = t.stamina &&
    (!t.stamina_available || Boolean(t.stamina_available[i]));
  const staminaExhausted = staminaAvailable &&
    (t.stamina_exhausted ? Boolean(t.stamina_exhausted[i]) : t.stamina[i] <= 0);
  if (staminaExhausted) rows.push(["stamina exhausted", "crit"]);
  $("hud").innerHTML = rows
    .map(([text, sev]) => `<span class="sev-${sev}" style="color:var(--sev)">${text}</span>`)
    .join("");
}
