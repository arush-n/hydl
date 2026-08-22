/* Canvas helpers and the orbit projection.
 *
 * Contexts and device-pixel sizing are cached: `getContext` and
 * `getBoundingClientRect` were previously called on every draw of every
 * canvas, and `getBoundingClientRect` forces layout. Now a resize observer
 * invalidates the cache instead.
 */

const ctx = new WeakMap();
const size = new WeakMap();

/** Cached 2D context. */
export function g2d(cv) {
  let g = ctx.get(cv);
  if (!g) { g = cv.getContext("2d", { alpha: false }); ctx.set(cv, g); }
  return g;
}

/** Cached backing-store size, refreshed only when the element resizes. */
export function fit(cv) {
  let s = size.get(cv);
  if (!s) {
    const ro = new ResizeObserver(() => { size.delete(cv); });
    ro.observe(cv);
    const r = cv.getBoundingClientRect();
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(r.width * dpr));
    const h = Math.max(1, Math.round(r.height * dpr));
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    s = [cv.width, cv.height];
    size.set(cv, s);
  }
  return s;
}

/* Palette lookups hit getComputedStyle, which is not free. Cache per theme and
   drop the cache when the theme changes. */
let palette = new Map();
export function css(name) {
  let v = palette.get(name);
  if (v === undefined) {
    v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    palette.set(name, v);
  }
  return v;
}
export function dropPalette() { palette = new Map(); }
matchMedia("(prefers-color-scheme:dark)").addEventListener("change", dropPalette);

/* ── camera ──────────────────────────────────────────────────────────────── */
export const cam = { yaw: 0.62, pitch: 0.46, dist: 15, cx: 0, cy: 65, cz: 0,
                     mode: "3d" };

/** Switch between the orbit view and a flat top-down plan. */
export function setMode(mode) { cam.mode = mode === "2d" ? "2d" : "3d"; }

export function project(p, W, H) {
  const dx = p[0] - cam.cx, dy = p[1] - cam.cy, dz = p[2] - cam.cz;

  if (cam.mode === "2d") {
    /* Top-down orthographic. Perspective is the wrong tool for reading
       positions: in the orbit view two actors at the same separation look
       different distances apart depending where they stand, and a spacing band
       is exactly what this project keeps needing to eyeball. Here one screen
       pixel is a fixed number of blocks everywhere.

       `z` still carries height so the painter's sort draws low ground first
       and high ground over it -- otherwise a pit would paint over the ridge
       beside it. */
    const cy2 = Math.cos(cam.yaw), sy2 = Math.sin(cam.yaw);
    const X = dx * cy2 - dz * sy2;
    const Zr = dx * sy2 + dz * cy2;
    const scale = (H * 0.82) / Math.max(4, cam.dist * 1.6);
    return { x: W / 2 + X * scale, y: H / 2 + Zr * scale, z: 1000 - dy,
             scale };
  }

  const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
  const X = dx * cy - dz * sy;
  const Zr = dx * sy + dz * cy;
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  // Positive pitch places the camera above the scene. The old signs put it
  // below the ground, so cube tops collapsed into a plate.
  const Y = dy * cp + Zr * sp;
  let Zc = -dy * sp + Zr * cp + cam.dist;
  if (Zc < 0.15) Zc = 0.15;
  const f = H * 0.92;
  return { x: W / 2 + (X * f) / Zc, y: H / 2 - (Y * f) / Zc, z: Zc };
}

/** Aim at the fight. The duel occupies a few units near y=65, so a fixed
 *  target leaves both bodies off-frame. */
/**
 * Point the camera at the two actors at `step`.
 *
 * <p>This used to frame the whole EPISODE's bounding box -- every position
 * either actor visited. That reads fine for a short duel and fails badly for a
 * chase: at a 256-tick horizon a sprinting actor covers ~60 blocks, so the box
 * was ~60 wide, `dist` came out near 160, and the replay opened zoomed so far
 * out that both actors were specks well away from the centre.
 *
 * <p>The old comment already had the right rule -- *distance is set by the
 * fight, not the terrain* -- but the code measured the whole path, which is the
 * map rather than the fight. Frame the pair instead, and bound it: the camera
 * stays orbitable, so a viewer who wants the whole path can zoom out.
 */
export function frameCamera(t, terrain, step = 0) {
  const last = Math.max(0, (t.agent_x?.length ?? 1) - 1);
  const i = Math.max(0, Math.min(Number(step) || 0, last));
  const ax = t.agent_x[i], az = t.agent_z[i];
  const tx = t.target_x[i], tz = t.target_z[i];
  const ys = t.agent_y.concat(t.target_y);
  cam.cx = (ax + tx) / 2;
  cam.cz = (az + tz) / 2;
  cam.cy = Math.min(...ys) + 0.9;

  // Separation, not path length. The floor keeps a contact-range pair from
  // pulling the camera inside the ground; the ceiling keeps a long opening gap
  // from reproducing the very problem this replaced.
  const separation = Math.hypot(ax - tx, az - tz);
  cam.dist = Math.min(48, Math.max(14, separation * 2.2 + 6));
  if (terrain && terrain.nodes) {
    cam.dist = Math.max(cam.dist, 16);
    if (terrain.blocks?.x?.length) {
      const bx = terrain.blocks.x, bz = terrain.blocks.z;
      const blockSpan = Math.max(
        Math.max(...bx) - Math.min(...bx),
        Math.max(...bz) - Math.min(...bz),
      );
      cam.dist = Math.max(cam.dist, Math.min(34, blockSpan * 0.75 + 7));
    }
    // Voxel ground needs to be looked *down* on. At the flat-scene pitch of
    // 0.46 rad (~26 deg) the view skims across a field of one-block cubes and
    // a descending slope reads as a ceiling -- you see their front faces, not
    // their tops. ~48 deg puts the tops toward the camera, which is what makes
    // the shape of the terrain legible.
    cam.pitch = Math.max(cam.pitch, 0.84);
    cam.cy = Math.min(...ys) + 1.6;
  }
}

/* A fixed light in world space, so a face's brightness means its orientation
   rather than its distance. The old shading dimmed by depth, which made far
   ground fade out and near ground glare -- terrain read as fog, not as slope.
   Normalised, pointing down-ish from the front-left. */
export const LIGHT = (() => {
  const v = [-0.45, 0.82, -0.35];
  const n = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / n, v[1] / n, v[2] / n];
})();

/** Lambert term for an axis-aligned cube face, in 0..1. */
export function lambert(normal) {
  const d = normal[0] * LIGHT[0] + normal[1] * LIGHT[1] + normal[2] * LIGHT[2];
  // Ambient floor: a face turned fully away is still lit by the sky, and
  // pure black hides the material colour the face exists to show.
  return 0.34 + 0.66 * Math.max(0, d);
}

/** WASD panning across the ground plane, in camera-relative directions. */
export function wasd(cv, onChange) {
  const held = new Set();
  let raf = null;
  let hovered = false;
  let last = 0;
  let precision = false;
  cv.tabIndex = 0;
  cv.addEventListener("mouseenter", () => { hovered = true; });
  cv.addEventListener("mouseleave", () => { hovered = false; held.clear(); });
  cv.addEventListener("pointerdown", () => cv.focus({ preventScroll: true }));

  const step = (now) => {
    if (!held.size) { raf = null; last = 0; return; }
    const elapsed = Math.min(0.04, (now - (last || now - 16)) / 1000);
    last = now;
    // Blocks per second, not blocks per frame. Shift gives precise inspection.
    const speed = Math.max(2.5, Math.min(18, cam.dist * 0.45))
      * (precision ? 0.28 : 1);
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
    let dx = 0, dz = 0, dy = 0;
    if (held.has("w")) { dx += sy; dz += cy; }
    if (held.has("s")) { dx -= sy; dz -= cy; }
    if (held.has("a")) { dx -= cy; dz += sy; }
    if (held.has("d")) { dx += cy; dz -= sy; }
    if (held.has("q")) dy -= 1;
    if (held.has("e")) dy += 1;
    const length = Math.hypot(dx, dz, dy) || 1;
    const rate = speed * elapsed / length;
    cam.cx += dx * rate;
    cam.cz += dz * rate;
    cam.cy += dy * rate;
    onChange();
    raf = requestAnimationFrame(step);
  };

  addEventListener("keydown", (event) => {
    if (event.target.matches("input,select,textarea")) return;
    if (!hovered && document.activeElement !== cv) return;
    precision = event.shiftKey;
    const key = event.code.replace("Key", "").toLowerCase();
    if (!"wasdqe".includes(key) || key.length !== 1) return;
    event.preventDefault();
    held.add(key);
    if (!raf) raf = requestAnimationFrame(step);
  });
  addEventListener("keyup", (event) => {
    precision = event.shiftKey;
    held.delete(event.code.replace("Key", "").toLowerCase());
  });
  addEventListener("blur", () => held.clear());
  document.addEventListener("visibilitychange", () => held.clear());
}

/** Attach orbit/zoom controls, calling `onChange` when the view moves. */
export function orbit(cv, onChange) {
  let drag = null;
  cv.style.touchAction = "none";
  cv.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX, y: e.clientY, button: e.button, shift: e.shiftKey };
    cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = Math.max(-48, Math.min(48, e.clientX - drag.x));
    const dy = Math.max(-48, Math.min(48, e.clientY - drag.y));
    if (drag.button === 2 || drag.shift) {
      const units = cam.dist / Math.max(420, cv.clientHeight) * 0.72;
      const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
      cam.cx += (-dx * cy - dy * sy) * units;
      cam.cz += (dx * sy - dy * cy) * units;
    } else {
      cam.yaw = (cam.yaw + dx * 0.006) % (Math.PI * 2);
      if (cam.mode === "3d") {
        cam.pitch = Math.max(-0.2, Math.min(1.35, cam.pitch + dy * 0.005));
      }
    }
    drag.x = e.clientX;
    drag.y = e.clientY;
    onChange();
  });
  const release = (e) => {
    if (e?.pointerId !== undefined && cv.hasPointerCapture(e.pointerId)) {
      cv.releasePointerCapture(e.pointerId);
    }
    drag = null;
  };
  cv.addEventListener("pointerup", release);
  cv.addEventListener("pointercancel", release);
  cv.addEventListener("lostpointercapture", () => { drag = null; });
  cv.addEventListener("contextmenu", (e) => e.preventDefault());
  cv.addEventListener("wheel", (e) => {
    e.preventDefault();
    cam.dist = Math.max(3, Math.min(120, cam.dist * (e.deltaY > 0 ? 1.09 : 0.92)));
    onChange();
  }, { passive: false });
}
