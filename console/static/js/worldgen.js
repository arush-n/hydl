/* WorldGen Studio.
 *
 * The server owns deterministic generation and validation.  This module owns
 * only authoring interaction and semantic browser renderers: it never labels a
 * preview as native Hytale output and it never changes a JAX pin.
 */

import { $ } from "./state.js";
import { css, fit, g2d } from "./gfx.js";
import { createVolumeRenderer } from "./worldgen_volume.js";

const GROUPS = [
  ["design", "Design"],
  ["randomization", "Procedural randomization"],
  ["terrain", "Terrain"],
  ["water", "Water"],
  ["climate", "Climate"],
  ["ecology", "Ecology"],
  ["caves", "Caves"],
  ["structures", "Structures"],
  ["gameplay", "Gameplay"],
];

const studio = {
  options: null,
  gameOptions: null,
  preview: null,
  request: null,
  sequence: 0,
  debounce: null,
  visible: false,
  mode: "3d",
  renderStyle: "voxels",
  batch: null,
  capture: null,
  voxelCache: null,
  volumeRenderer: null,
  volumeRendererError: null,
  drawFrame: 0,
  camera: { yaw: 0.72, pitch: 0.82, dist: 190, panX: 0, panZ: 0 },
};

const VOXEL_TARGET_COLUMNS = 64;
let pendingSeed = null;

const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]
));

function fieldId(name) { return `wgField-${name}`; }

/** Carry a replay's Region seed into the authoring preview without claiming parity. */
export function loadWorldgenSeed(seed) {
  pendingSeed = Number(seed);
  const input = $(fieldId("seed"));
  if (!studio.options || !input || !Number.isInteger(pendingSeed)) return;
  input.value = pendingSeed;
  const range = document.querySelector('[data-wg-range="seed"]');
  if (range) range.value = pendingSeed;
  $("wgBatchStart").value = pendingSeed;
  $("wgPreset").value = "custom";
  $("wgPresetNote").textContent =
    `Loaded Region seed ${pendingSeed}. This is a seeded Studio preview, not the native capture.`;
  pendingSeed = null;
  if (studio.preview) requestPreview();
}

function readError(response, body) {
  if (body && typeof body.detail === "string") return body.detail;
  if (body && body.detail) return JSON.stringify(body.detail);
  return `${response.status} ${response.statusText}`;
}

async function jsonFetch(url, init) {
  const response = await fetch(url, init);
  let body = null;
  try { body = await response.json(); } catch { /* the status still explains it */ }
  if (!response.ok) throw new Error(readError(response, body));
  return body;
}

function choiceControl(name, field, value) {
  const options = field.choices.map((choice) =>
    `<option value="${esc(choice)}"${String(choice) === String(value) ? " selected" : ""}>` +
    `${esc(String(choice).replaceAll("_", " "))}</option>`).join("");
  return `<label class="f"><span>${esc(field.label)}</span>` +
    `<select id="${fieldId(name)}" data-wg-field="${esc(name)}">${options}</select></label>`;
}

function plainControl(name, field, value) {
  const inputType = field.type === "integer" ? "number" : "text";
  const attrs = field.type === "integer"
    ? ` min="${field.min}" max="${field.max}" step="${field.step}"`
    : ` maxlength="64" spellcheck="false"`;
  return `<label class="f"><span>${esc(field.label)}</span>` +
    `<input id="${fieldId(name)}" data-wg-field="${esc(name)}" type="${inputType}"` +
    `${attrs} value="${esc(value)}"></label>`;
}

function sliderControl(name, field, value) {
  const unit = field.unit ? ` · ${field.unit}` : "";
  return `<label class="wg-slider"><span>${esc(field.label + unit)}</span>` +
    `<input type="range" data-wg-range="${esc(name)}" min="${field.min}" max="${field.max}" ` +
    `step="${field.step}" value="${esc(value)}" aria-label="${esc(field.label)}">` +
    `<input id="${fieldId(name)}" data-wg-field="${esc(name)}" type="number" ` +
    `min="${field.min}" max="${field.max}" step="${field.step}" value="${esc(value)}" ` +
    `aria-label="${esc(field.label)} value"></label>`;
}

function buildControls() {
  const fields = studio.options.fields;
  const defaults = studio.options.defaults;
  $("wgControls").innerHTML = GROUPS.map(([group, title]) => {
    const names = Object.keys(fields).filter((name) => fields[name].group === group);
    const body = names.map((name) => {
      const field = fields[name], value = defaults[name];
      if (field.type === "number") return sliderControl(name, field, value);
      if (field.type === "choice") return choiceControl(name, field, value);
      return plainControl(name, field, value);
    }).join("");
    const open = ["design", "randomization", "terrain", "water"].includes(group) ? " open" : "";
    const inline = group === "design" ? " wg-inline-fields" : "";
    return `<details class="wg-control-group"${open}><summary>${esc(title)}</summary>` +
      `<div class="wg-control-body${inline}">${body}</div></details>`;
  }).join("");

  $("wgPreset").innerHTML = studio.options.presets.map((preset) =>
    `<option value="${esc(preset.id)}">${esc(preset.name)}</option>`).join("") +
    `<option value="custom">Custom</option>`;

  $("wgControls").addEventListener("input", (event) => {
    const range = event.target.closest("[data-wg-range]");
    if (range) $(fieldId(range.dataset.wgRange)).value = range.value;
    const value = event.target.closest("[data-wg-field]");
    if (value) {
      const partner = document.querySelector(`[data-wg-range="${value.dataset.wgField}"]`);
      if (partner) partner.value = value.value;
    }
    syncGenerationMode();
    $("wgPreset").value = "custom";
    queuePreview(event.target.type === "text" ? 550 : 220);
  });
  $("wgControls").addEventListener("change", () => queuePreview(40));
}

function setControls(config) {
  for (const [name, field] of Object.entries(studio.options.fields)) {
    const input = $(fieldId(name));
    if (!input || config[name] === undefined) continue;
    input.value = config[name];
    if (field.type === "number") {
      const range = document.querySelector(`[data-wg-range="${name}"]`);
      if (range) range.value = config[name];
    }
  }
  syncGenerationMode();
}

function syncGenerationMode() {
  const mode = $(fieldId("generation_mode"))?.value;
  const family = $(fieldId("terrain_style"));
  if (family) family.disabled = mode === "procedural_genome";
}

function controls() {
  const payload = {};
  for (const [name, field] of Object.entries(studio.options.fields)) {
    const value = $(fieldId(name)).value;
    const numericChoice = field.type === "choice" && typeof field.choices?.[0] === "number";
    payload[name] = ["number", "integer"].includes(field.type) || numericChoice
      ? Number(value) : value;
  }
  return payload;
}

function applyPreset(id) {
  const preset = studio.options.presets.find((item) => item.id === id);
  if (!preset) return;
  setControls(preset.values);
  $("wgBatchStart").value = preset.values.seed;
  $("wgPresetNote").textContent = preset.description;
  resetCamera(Math.max(preset.values.world_width, preset.values.world_depth));
  requestPreview();
}

function queuePreview(delay = 220) {
  clearTimeout(studio.debounce);
  studio.debounce = setTimeout(requestPreview, delay);
}

async function requestPreview() {
  if (!studio.options) return;
  clearTimeout(studio.debounce);
  studio.request?.abort();
  const request = new AbortController();
  studio.request = request;
  const sequence = ++studio.sequence;
  $("wgLoading").hidden = false;
  $("wgRequestState").textContent = "generating…";
  $("wgGenerate").disabled = true;
  try {
    const payload = await jsonFetch("/api/worldgen/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(controls()),
      signal: request.signal,
    });
    if (sequence !== studio.sequence) return;
    const previousExtent = studio.preview
      ? [studio.preview.config.world_width, studio.preview.config.world_depth]
      : null;
    studio.preview = payload;
    studio.capture = null;
    $("wgTrees").disabled = false;
    $("wgStructures").disabled = false;
    setControls(payload.config);          // display normalized names/numbers
    showPreview(payload);
    if (!previousExtent || previousExtent[0] !== payload.config.world_width ||
        previousExtent[1] !== payload.config.world_depth) {
      resetCamera(Math.max(payload.config.world_width, payload.config.world_depth));
    }
    draw();
  } catch (error) {
    if (error.name === "AbortError") return;
    showFailure(error.message);
  } finally {
    if (sequence === studio.sequence) {
      $("wgLoading").hidden = true;
      $("wgGenerate").disabled = false;
    }
  }
}

function metric(label, value, tone = "") {
  return `<div class="ro"><i>${esc(label)}</i><b class="${esc(tone)}">${esc(value)}</b></div>`;
}

function percent(value) { return `${(value * 100).toFixed(1)}%`; }

function showPreview(payload) {
  const metrics = payload.metrics;
  const extentPlan = payload.design.extent.native_capture;
  $("wgRequestState").textContent = `${payload.grid.resolution}² samples`;
  $("wgDigest").textContent = `sha256 ${payload.digest.slice(0, 12)}`;
  $("wgQuality").textContent = payload.quality === "ready_for_native_authoring"
    ? "ready to author" : "review design";
  $("wgQuality").className = `badge ${payload.warnings.length ? "review" : "ready"}`;
  $("wgMetrics").innerHTML = [
    metric("realized relief", `${metrics.relief.toFixed(1)} blocks`),
    metric("water", percent(metrics.water_fraction)),
    metric("traversable", percent(metrics.traversable_fraction), metrics.traversable_fraction < .38 ? "neg" : "pos"),
    metric("trees", metrics.tree_count.toLocaleString()),
    metric("structures", metrics.structure_count.toLocaleString()),
    metric("cave passages", metrics.cave_segment_count.toLocaleString()),
    metric("macro regions", metrics.terrain_region_count),
    metric("route edges", metrics.route_edge_count),
  ].join("");
  const worldLabel = payload.config.generation_mode === "procedural_genome"
    ? "procedural hyperparameter world"
    : payload.config.name;
  $("wgHud").innerHTML = [
    `<span>${esc(worldLabel)}</span>`,
    `<span>seed ${payload.config.seed}</span>`,
    `<span>${payload.config.world_width} × ${payload.config.world_depth} blocks</span>`,
    `<span>Y ${metrics.minimum_height.toFixed(1)}…${metrics.maximum_height.toFixed(1)}</span>`,
    `<span>${payload.volume?.voxel_size || voxelUnitForExtent(
      payload.config.world_width, payload.config.world_depth)}-block volume voxels</span>`,
  ].join("");
  if (payload.volume?.recommended_view) {
    const recommended = payload.volume.recommended_view;
    $("wgCutAxis").value = recommended.cut_axis;
    $("wgCutDepth").value = recommended.cut_percent;
    $("wgCutDepthValue").textContent = `${recommended.cut_percent}%`;
  }
  $("cmWorld").className = `flag ${payload.warnings.length ? "sev-warn" : "sev-good"}`;
  $("cmWorld").textContent =
    `World ${worldLabel} · seed ${payload.config.seed} · ${payload.digest.slice(0, 12)} · ` +
    "unique ownership is checked when the minigame is saved.";

  const present = new Set(Object.keys(metrics.biome_counts));
  $("wgLegend").innerHTML = payload.biome_legend
    .filter((biome) => present.has(biome.name))
    .map((biome) => `<span><i style="background:${esc(biome.color)}"></i>` +
      `${esc(biome.name.replaceAll("_", " "))}</span>`).join("");

  const jaxWorld = extentPlan.jax_world;
  const nativeProfile = payload.design.native_handoff.template_compilation;
  const referenceValidation = nativeProfile.reference_validation;
  const nativeReferenceNote = referenceValidation
    ? `<div class="flag sev-good">Procedural compiler reference passed on Hytale 0.5.7 · ` +
      `${referenceValidation.seed_count} exact Regions · ` +
      `${percent(referenceValidation.minimum_pairwise_surface_difference)} minimum surface difference. ` +
      `This requested pack is not certified until it is generated and captured.</div>`
    : "";
  const extentNote = `<div class="flag sev-good">JAX target: 1 world · ` +
    `${jaxWorld.region_tiles} Region ${jaxWorld.region_tiles === 1 ? "tile" : "tiles"} · ` +
    `shared world ID ${jaxWorld.world_id}. Exact native capture is still required.</div>`;
  if (payload.warnings.length) {
    $("wgWarnings").innerHTML = payload.warnings.map((warning) =>
      `<div class="flag sev-warn">${esc(warning)}</div>`).join("") +
      nativeReferenceNote + extentNote;
    $("wgCheckCount").textContent = `${payload.warnings.length} to review`;
  } else {
    $("wgWarnings").innerHTML = `<div class="flag sev-good">Preview checks passed. Native generation and traversal validation are still required.</div>` + nativeReferenceNote + extentNote;
    $("wgCheckCount").textContent = "preview checks pass";
  }
  const spawn = payload.spawn;
  $("wgSpawnFacts").innerHTML = [
    metric("spawn", `${spawn.x.toFixed(1)}, ${spawn.y.toFixed(1)}, ${spawn.z.toFixed(1)}`, spawn.safe ? "pos" : "neg"),
    metric("spawn slope", spawn.slope.toFixed(3)),
    metric("clear radius", `${spawn.clearance} blocks`),
    metric("cave entrances", metrics.cave_entrance_count),
    metric("terrain components", metrics.terrain_component_count),
    metric("route coverage", payload.world_genome
      ? `${metrics.route_length.toFixed(1)} blocks · ${percent(metrics.route_traversable_fraction)} walkable`
      : "disabled in authored mode",
      payload.world_genome && metrics.route_connected && metrics.route_traversable_fraction >= .58 ? "pos" : ""),
    metric("intent", payload.config.objective.replaceAll("_", " ")),
    metric("native profile", payload.design.native_handoff.template_compilation.profile.replaceAll("_", " ")),
    metric("capture footprint", `${extentPlan.capture_tiles.x} x ${extentPlan.capture_tiles.z} core tiles`,
      extentPlan.jax_world.shared_world_id ? "pos" : "neg"),
    metric("JAX topology", `1 world / ${extentPlan.jax_world.region_tiles} Region tiles`, "pos"),
  ].join("");
}

function showFailure(message) {
  $("wgRequestState").textContent = "preview failed";
  $("wgQuality").textContent = "invalid design";
  $("wgQuality").className = "badge failed";
  $("wgWarnings").innerHTML = `<div class="flag sev-crit">${esc(message)}</div>`;
  $("wgCheckCount").textContent = "fix controls";
}

function showCapturedPreview(payload) {
  const volume = payload.volume;
  const metrics = payload.metrics;
  const capture = payload.capture;
  const exact = volume.voxel_size === volume.source_voxel_size;
  studio.preview = payload;
  studio.capture = capture;
  $("wgRequestState").textContent = exact
    ? "native-cell exact volume"
    : `${volume.voxel_size}-block declared display LOD`;
  $("wgDigest").textContent = `capture ${payload.digest.slice(0, 12)}`;
  $("wgQuality").textContent = "captured exact source";
  $("wgQuality").className = "badge ready";
  $("wgMetrics").innerHTML = [
    metric("source non-air", metrics.source_nonair_cells.toLocaleString()),
    metric("source solid", metrics.source_solid_cells.toLocaleString()),
    metric("source fluid", metrics.source_fluid_cells.toLocaleString()),
    metric("enclosed voids", metrics.enclosed_void_source_cell_equivalent.toLocaleString()),
    metric("stable materials", metrics.source_unique_materials.toLocaleString()),
    metric("display voxels", metrics.occupied_voxels.toLocaleString()),
    metric("display fidelity", exact ? "native cell" : `${volume.voxel_size}x LOD`, exact ? "pos" : ""),
  ].join("");
  $("wgHud").innerHTML = [
    `<span>${esc(capture.environment_id)}</span>`,
    `<span>native seed ${capture.seed}</span>`,
    `<span>${payload.config.world_width} x ${payload.config.world_depth} blocks</span>`,
    `<span>${capture.region_tiles} Region ${capture.region_tiles === 1 ? "tile" : "tiles"} / world 0</span>`,
    `<span>${volume.voxel_size}-block volume voxels</span>`,
  ].join("");
  const recommended = volume.recommended_view;
  $("wgCutAxis").value = recommended.cut_axis;
  $("wgCutDepth").value = recommended.cut_percent;
  $("wgCutDepthValue").textContent = `${recommended.cut_percent}%`;
  // Captured stable identities do not currently claim tree/structure taxonomy;
  // keep those preview-only filters visibly unavailable instead of guessing.
  $("wgTrees").disabled = true;
  $("wgStructures").disabled = true;
  const visiblePalette = volume.palette.filter((entry) => entry.id !== 0).slice(0, 14);
  $("wgLegend").innerHTML = visiblePalette.map((entry) =>
    `<span title="${esc(entry.name)}"><i style="background:${esc(entry.color)}"></i>` +
    `${esc(entry.name.replaceAll("_", " "))}</span>`).join("");
  $("wgWarnings").innerHTML =
    `<div class="flag sev-good">Authenticated ${capture.region_tiles}-tile composite; ` +
    `every tile maps to literal JAX world ID 0.</div>` +
    `<div class="flag ${exact ? "sev-good" : "sev-warn"}">` +
    `${exact ? "Displayed occupancy is one native cell per voxel." :
      `Display is reduced to ${volume.voxel_size}-block voxels; source Region artifacts remain exact.`}</div>`;
  $("wgCheckCount").textContent = exact ? "native-cell view" : "explicit visual LOD";
  $("wgSpawnFacts").innerHTML = [
    metric("native spawn", `${payload.spawn.x.toFixed(1)}, ${payload.spawn.y.toFixed(1)}, ${payload.spawn.z.toFixed(1)}`, "pos"),
    metric("structure", capture.structure),
    metric("capture grid", `${capture.tile_grid[0]} x ${capture.tile_grid[1]}`),
    metric("JAX topology", `1 world / ${capture.region_tiles} Region tiles`, "pos"),
    metric("material identity", "stable block + fluid", "pos"),
    metric("collision source", "exact Region palette", "pos"),
  ].join("");
  resetCamera(Math.max(payload.config.world_width, payload.config.world_depth));
  syncCutawayControls();
  draw();
}

function showCaptures(body) {
  const select = $("wgCaptureSelect");
  const selected = select.value;
  if (!body.captures.length) {
    select.innerHTML = `<option value="">No captured worlds found</option>`;
    $("wgCaptureState").textContent = body.errors.length ? "capture errors" : "none published";
    $("wgLoadCapture").disabled = true;
    $("wgCaptureFacts").innerHTML = body.errors.length
      ? body.errors.slice(0, 3).map((item) =>
        `<div class="flag sev-warn">${esc(item.error)}</div>`).join("")
      : `<div class="flag">Compile/generate/capture first; this viewer never invents an exact artifact.</div>`;
    return;
  }
  select.innerHTML = body.captures.map((item) =>
    `<option value="${esc(item.capture_id)}">${esc(item.environment_id)} - seed ${item.seed} - ` +
    `${item.region_tiles} Regions</option>`).join("");
  if (body.captures.some((item) => item.capture_id === selected)) select.value = selected;
  $("wgCaptureState").textContent = `${body.captures.length} published`;
  $("wgLoadCapture").disabled = false;
  $("wgCaptureFacts").innerHTML =
    `<div class="flag sev-good">Only locally discovered, manifest-bound artifact paths are loadable.</div>`;
}

async function loadCaptures() {
  try { showCaptures(await jsonFetch("/api/worldgen/captures")); }
  catch (error) {
    $("wgCaptureState").textContent = "scan failed";
    $("wgCaptureFacts").innerHTML = `<div class="flag sev-warn">${esc(error.message)}</div>`;
  }
}

async function loadCapturedVolume() {
  const captureId = $("wgCaptureSelect").value;
  if (!captureId) return;
  $("wgLoadCapture").disabled = true;
  $("wgCaptureState").textContent = "authenticating + voxelizing...";
  $("wgLoading").hidden = false;
  try {
    const detail = Number($("wgCaptureDetail").value);
    const payload = await jsonFetch(
      `/api/worldgen/captures/${encodeURIComponent(captureId)}/volume?target_columns=${detail}`,
    );
    showCapturedPreview(payload);
    $("wgCaptureState").textContent = payload.volume.voxel_size === 1
      ? "native-cell exact"
      : `${payload.volume.voxel_size}-block display LOD`;
    $("wgCaptureFacts").innerHTML =
      `<div class="flag sev-good">${payload.capture.region_tiles} Region tiles loaded as one JAX world - ` +
      `${payload.metrics.source_nonair_cells.toLocaleString()} source cells - ` +
      `${payload.volume.semantic_sha256.slice(0, 12)}</div>`;
  } catch (error) {
    $("wgCaptureState").textContent = "load failed";
    $("wgCaptureFacts").innerHTML = `<div class="flag sev-crit">${esc(error.message)}</div>`;
  } finally {
    $("wgLoading").hidden = true;
    $("wgLoadCapture").disabled = !$("wgCaptureSelect").value;
  }
}

function showPipeline() {
  $("wgPipeline").innerHTML = studio.options.pipeline.map((step, index) =>
    `<div class="wg-pipeline-step${step.exact ? " exact" : ""}">` +
    `<i>${index + 1} · ${esc(step.state)}</i><b>${esc(step.label)}</b></div>`).join("");
  $("wgBoundary").textContent = studio.options.boundary;
}

function syncCutawayControls() {
  const enabled = studio.renderStyle === "voxels" && $("wgCaves").checked;
  $("wgCutAxis").disabled = !enabled;
  $("wgCutDepth").disabled = !enabled;
}

function resetCamera(worldSize = Math.max(
  studio.preview?.config.world_width || 128,
  studio.preview?.config.world_depth || 128,
)) {
  studio.camera = {
    yaw: 0.72,
    pitch: studio.mode === "top" ? Math.PI / 2 : 0.82,
    dist: worldSize * (studio.mode === "top" ? 1.05 : 1.48),
    panX: 0,
    panZ: 0,
  };
  draw();
}

function scheduleDraw() {
  if (studio.drawFrame) return;
  studio.drawFrame = requestAnimationFrame(() => {
    studio.drawFrame = 0;
    draw();
  });
}

function projection(point, width, height, centerHeight) {
  const camera = studio.camera;
  const exaggeration = Number($("wgExaggeration").value);
  const dx = point[0] - camera.panX;
  const dy = (point[1] - centerHeight) * exaggeration;
  const dz = point[2] - camera.panZ;
  const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
  const x = dx * cy - dz * sy;
  const zr = dx * sy + dz * cy;

  if (studio.mode === "top") {
    const worldWidth = studio.preview.config.world_width;
    const worldDepth = studio.preview.config.world_depth;
    const diagonal = Math.hypot(worldWidth, worldDepth);
    const defaultDistance = Math.max(worldWidth, worldDepth) * 1.05;
    const scale = Math.min(width, height) * 0.84 / diagonal *
      (defaultDistance / camera.dist);
    return { x: width / 2 + x * scale, y: height / 2 + zr * scale, z: 1000 - dy, scale };
  }

  const cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
  const y = dy * cp - zr * sp;
  if (studio.renderStyle === "voxels") {
    // Orthographic projection keeps every block the same screen size while
    // orbiting. Perspective made the old terrain plates pulse and shimmer as
    // their painter order and apparent width changed during a drag.
    const worldWidth = studio.preview.config.world_width;
    const worldDepth = studio.preview.config.world_depth;
    const diagonal = Math.hypot(worldWidth, worldDepth);
    const vertical = studio.preview.metrics.relief * exaggeration * 1.65;
    const frameSpan = Math.max(24, diagonal, vertical);
    const defaultDistance = Math.max(worldWidth, worldDepth) * 1.48;
    const scale = Math.min(width, height) * 0.78 / frameSpan *
      (defaultDistance / camera.dist);
    return {
      x: width / 2 + x * scale,
      y: height / 2 - y * scale,
      z: dy * sp + zr * cp,
      scale,
    };
  }
  const zc = Math.max(0.2, dy * sp + zr * cp + camera.dist);
  const focal = height * 0.96;
  return {
    x: width / 2 + x * focal / zc,
    y: height / 2 - y * focal / zc,
    z: zc,
    scale: focal / zc,
  };
}

function hexChannels(hex) {
  const value = hex.replace("#", "");
  return [0, 2, 4].map((offset) => parseInt(value.slice(offset, offset + 2), 16));
}

function shade(hex, amount) {
  const [r, g, b] = hexChannels(hex);
  return `rgb(${Math.round(Math.max(0, Math.min(255, r * amount)))},` +
    `${Math.round(Math.max(0, Math.min(255, g * amount)))},` +
    `${Math.round(Math.max(0, Math.min(255, b * amount)))})`;
}

function voxelUnitForExtent(worldWidth, worldDepth) {
  const requested = Math.max(1, Math.ceil(
    Math.max(worldWidth, worldDepth) / VOXEL_TARGET_COLUMNS));
  return 2 ** Math.ceil(Math.log2(requested));
}

function sampleHeight(grid, worldX, worldZ) {
  const gx = Math.max(0, Math.min(grid.resolution - 1,
    (worldX + grid.world_width * .5) / grid.step_x));
  const gz = Math.max(0, Math.min(grid.resolution - 1,
    (worldZ + grid.world_depth * .5) / grid.step_z));
  const x0 = Math.floor(gx), z0 = Math.floor(gz);
  const x1 = Math.min(grid.resolution - 1, x0 + 1);
  const z1 = Math.min(grid.resolution - 1, z0 + 1);
  const tx = gx - x0, tz = gz - z0;
  const north = grid.heights[z0][x0] * (1 - tx) + grid.heights[z0][x1] * tx;
  const south = grid.heights[z1][x0] * (1 - tx) + grid.heights[z1][x1] * tx;
  return north * (1 - tz) + south * tz;
}

function sampleBiome(grid, worldX, worldZ) {
  const gx = Math.max(0, Math.min(grid.resolution - 1,
    Math.round((worldX + grid.world_width * .5) / grid.step_x)));
  const gz = Math.max(0, Math.min(grid.resolution - 1,
    Math.round((worldZ + grid.world_depth * .5) / grid.step_z)));
  return grid.biomes[gz][gx];
}

function voxelSurface(payload) {
  if (studio.voxelCache?.digest === payload.digest) return studio.voxelCache.model;
  const grid = payload.grid;
  const unit = voxelUnitForExtent(grid.world_width, grid.world_depth);
  const columnsX = Math.ceil(grid.world_width / unit);
  const columnsZ = Math.ceil(grid.world_depth / unit);
  const halfX = grid.world_width * .5;
  const halfZ = grid.world_depth * .5;
  const heights = Array.from({ length: columnsZ }, (_, z) =>
    Array.from({ length: columnsX }, (_, x) => {
      const wx = Math.min(halfX, -halfX + (x + .5) * unit);
      const wz = Math.min(halfZ, -halfZ + (z + .5) * unit);
      return Math.round(sampleHeight(grid, wx, wz) / unit) * unit;
    }));
  const biomes = Array.from({ length: columnsZ }, (_, z) =>
    Array.from({ length: columnsX }, (_, x) => {
      const wx = Math.min(halfX, -halfX + (x + .5) * unit);
      const wz = Math.min(halfZ, -halfZ + (z + .5) * unit);
      return sampleBiome(grid, wx, wz);
    }));
  const model = { unit, columnsX, columnsZ, halfX, halfZ, heights, biomes };
  studio.voxelCache = { digest: payload.digest, model };
  return model;
}

function polygon(g, points, fill, stroke = null, alpha = 1) {
  if (points.some((point) => !Number.isFinite(point.x + point.y))) return;
  g.beginPath();
  g.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) g.lineTo(points[i].x, points[i].y);
  g.closePath();
  g.globalAlpha = alpha;
  g.fillStyle = fill;
  g.fill();
  if (stroke) { g.strokeStyle = stroke; g.lineWidth = 0.65; g.stroke(); }
  g.globalAlpha = 1;
}

function treeDrawable(tree, width, height, centerHeight) {
  const ground = projection([tree.x, tree.y, tree.z], width, height, centerHeight);
  const top = projection([tree.x, tree.y + tree.height, tree.z], width, height, centerHeight);
  const crown = projection([tree.x, tree.y + tree.height * 0.52, tree.z], width, height, centerHeight);
  return {
    depth: ground.z,
    draw(g) {
      if (studio.mode === "top") {
        g.fillStyle = tree.kind === "pine" ? "#244f36" : (tree.kind === "birch" ? "#5d8849" : "#356c42");
        g.beginPath(); g.arc(ground.x, ground.y, Math.max(1.4, tree.height * ground.scale * .22), 0, Math.PI * 2); g.fill();
        return;
      }
      g.strokeStyle = tree.kind === "birch" ? "#c6c0a7" : "#5b4027";
      g.lineWidth = Math.max(1, ground.scale * .32);
      g.beginPath(); g.moveTo(ground.x, ground.y); g.lineTo(crown.x, crown.y); g.stroke();
      const halfWidth = Math.max(1.6, tree.height * crown.scale * .32);
      const color = tree.kind === "pine" ? "#244f36" : (tree.kind === "birch" ? "#608c4a" : "#356c42");
      polygon(g, [top, { x: crown.x - halfWidth, y: crown.y }, { x: crown.x + halfWidth, y: crown.y }], color);
    },
  };
}

function pushBoxFaces(drawables, bounds, color, width, height, centerHeight) {
  const [x0, y0, z0, x1, y1, z1] = bounds;
  const faces = [
    { tone: 1.08, world: [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]] },
    { tone: .68, world: [[x0, y1, z0], [x0, y1, z1], [x0, y0, z1], [x0, y0, z0]] },
    { tone: .82, world: [[x1, y1, z1], [x1, y1, z0], [x1, y0, z0], [x1, y0, z1]] },
    { tone: .74, world: [[x1, y1, z0], [x0, y1, z0], [x0, y0, z0], [x1, y0, z0]] },
    { tone: .9, world: [[x0, y1, z1], [x1, y1, z1], [x1, y0, z1], [x0, y0, z1]] },
  ];
  for (const face of faces) {
    const points = face.world.map((point) => projection(point, width, height, centerHeight));
    drawables.push({
      depth: points.reduce((sum, point) => sum + point.z, 0) / points.length,
      draw(g) { polygon(g, points, shade(color, face.tone), "rgba(12,18,20,.34)"); },
    });
  }
}

function pushVoxelTree(drawables, tree, unit, width, height, centerHeight) {
  const block = Math.max(1, Math.min(2, unit));
  const trunkHeight = Math.max(block * 2, Math.round(tree.height * .58 / block) * block);
  const trunkX = Math.round(tree.x / block) * block;
  const trunkZ = Math.round(tree.z / block) * block;
  pushBoxFaces(drawables, [
    trunkX - block * .5, tree.y, trunkZ - block * .5,
    trunkX + block * .5, tree.y + trunkHeight, trunkZ + block * .5,
  ], tree.kind === "birch" ? "#c6c0a7" : "#5b4027", width, height, centerHeight);
  const crownColor = tree.kind === "pine" ? "#244f36" :
    (tree.kind === "birch" ? "#608c4a" : "#356c42");
  const crownWidth = Math.max(block * 2,
    Math.round(tree.height * (tree.kind === "pine" ? .42 : .5) / block) * block);
  const crownBottom = tree.y + Math.max(block, trunkHeight - block * 1.5);
  const crownTop = tree.y + Math.max(trunkHeight + block,
    Math.round(tree.height / block) * block);
  pushBoxFaces(drawables, [
    trunkX - crownWidth * .5, crownBottom, trunkZ - crownWidth * .5,
    trunkX + crownWidth * .5, crownTop, trunkZ + crownWidth * .5,
  ], crownColor, width, height, centerHeight);
  if (tree.kind === "pine" && crownTop - crownBottom > block * 2) {
    pushBoxFaces(drawables, [
      trunkX - crownWidth * .32, crownTop, trunkZ - crownWidth * .32,
      trunkX + crownWidth * .32, crownTop + block, trunkZ + crownWidth * .32,
    ], shade(crownColor, 1.05), width, height, centerHeight);
  }
}

function structureDrawable(structure, width, height, centerHeight) {
  const size = structure.footprint * .55;
  const tall = structure.height || (structure.kind === "watchtower" ? 10 : structure.kind === "village_house" ? 5.5 : 4.2);
  const points = [
    [-size, 0, -size], [size, 0, -size], [size, 0, size], [-size, 0, size],
    [-size, tall, -size], [size, tall, -size], [size, tall, size], [-size, tall, size],
  ].map(([x, y, z]) => projection(
    [structure.x + x, structure.y + y, structure.z + z], width, height, centerHeight));
  const depth = points.reduce((sum, point) => sum + point.z, 0) / points.length;
  const baseColor = structure.kind === "village_house" ? "#9a6d43" :
    structure.kind === "fortified_camp" ? "#76664f" : "#8d8b83";
  return {
    depth,
    draw(g) {
      if (studio.mode === "top") {
        polygon(g, points.slice(0, 4), baseColor, "rgba(255,255,255,.55)");
        return;
      }
      polygon(g, [points[4], points[5], points[6], points[7]], shade(baseColor, 1.1));
      polygon(g, [points[0], points[1], points[5], points[4]], shade(baseColor, .76));
      polygon(g, [points[1], points[2], points[6], points[5]], shade(baseColor, .9));
      g.strokeStyle = "rgba(240,225,190,.75)";
      g.lineWidth = 1;
      g.beginPath(); g.moveTo(points[4].x, points[4].y); g.lineTo(points[6].x, points[6].y); g.stroke();
    },
  };
}

function spawnDrawable(spawn, width, height, centerHeight) {
  const ground = projection([spawn.x, spawn.y, spawn.z], width, height, centerHeight);
  const top = projection([spawn.x, spawn.y + 9, spawn.z], width, height, centerHeight);
  return {
    depth: ground.z - 0.01,
    draw(g) {
      const radius = Math.max(3, spawn.clearance * ground.scale);
      g.strokeStyle = spawn.safe ? css("--accent") : css("--crit");
      g.lineWidth = 1.5;
      g.globalAlpha = .8;
      g.beginPath();
      if (studio.mode === "top") g.arc(ground.x, ground.y, radius, 0, Math.PI * 2);
      else g.ellipse(ground.x, ground.y, radius, radius * .34, 0, 0, Math.PI * 2);
      g.stroke();
      g.globalAlpha = 1;
      if (studio.mode !== "top") {
        g.beginPath(); g.moveTo(ground.x, ground.y); g.lineTo(top.x, top.y); g.stroke();
      }
      g.fillStyle = css("--accent");
      g.beginPath(); g.arc(ground.x, ground.y, 2.5, 0, Math.PI * 2); g.fill();
    },
  };
}

function caveDrawable(caves, width, height, centerHeight) {
  return {
    // Cutaway diagnostics are deliberately overlaid after terrain. They show
    // authored topology and depth, not an impossible claim of browser-side
    // voxel excavation.
    depth: -1e9,
    draw(g) {
      g.save();
      g.globalAlpha = .78;
      g.lineCap = "round";
      g.setLineDash([5, 4]);
      for (const segment of caves.segments) {
        const start = projection(segment.start, width, height, centerHeight);
        const end = projection(segment.end, width, height, centerHeight);
        const depth = Math.max(0, centerHeight - (segment.start[1] + segment.end[1]) / 2);
        g.strokeStyle = depth > 28 ? "#9d79d6" : "#c08ce0";
        g.lineWidth = Math.max(1.2, Math.min(7, segment.radius * (start.scale + end.scale) * .35));
        g.beginPath(); g.moveTo(start.x, start.y); g.lineTo(end.x, end.y); g.stroke();
      }
      g.setLineDash([]);
      for (const chamber of caves.chambers) {
        const point = projection([chamber.x, chamber.y, chamber.z], width, height, centerHeight);
        const radius = Math.max(2, chamber.radius * point.scale * .42);
        g.fillStyle = "rgba(123,83,166,.34)";
        g.strokeStyle = "rgba(209,175,241,.82)";
        g.lineWidth = 1;
        g.beginPath(); g.arc(point.x, point.y, radius, 0, Math.PI * 2); g.fill(); g.stroke();
      }
      for (const entrance of caves.entrances) {
        const point = projection([entrance.x, entrance.y, entrance.z], width, height, centerHeight);
        const radius = Math.max(3, entrance.radius * point.scale);
        g.fillStyle = "rgba(18,13,24,.88)";
        g.strokeStyle = "#d1aff1";
        g.lineWidth = 1.5;
        g.beginPath(); g.arc(point.x, point.y, radius, 0, Math.PI * 2); g.fill(); g.stroke();
      }
      g.restore();
    },
  };
}

export function drawWorldgen() {
  draw();
}

function draw() {
  const payload = studio.preview;
  const canvas = $("wgView");
  const volumeCanvas = $("wgVoxelView");
  if (!payload || !canvas || !volumeCanvas || !studio.visible) return;

  if (studio.renderStyle === "voxels" && payload.volume) {
    if (studio.volumeRenderer === null) {
      try {
        studio.volumeRenderer = createVolumeRenderer(volumeCanvas) || false;
      } catch (error) {
        studio.volumeRenderer = false;
        studio.volumeRendererError = error.message;
        console.error("[worldgen-volume]", error);
      }
    }
    if (studio.volumeRenderer) {
      canvas.hidden = true;
      volumeCanvas.hidden = false;
      try {
        const stats = studio.volumeRenderer.render(payload, studio.camera, {
          water: $("wgWater").checked,
          trees: $("wgTrees").checked,
          structures: $("wgStructures").checked,
          cutaway: $("wgCaves").checked,
          cutAxis: $("wgCutAxis").value,
          cutPercent: Number($("wgCutDepth").value),
          grid: $("wgGrid").checked,
          exaggeration: Number($("wgExaggeration").value),
        });
        volumeCanvas.dataset.meshBuilds = String(stats.meshBuilds);
        volumeCanvas.dataset.volumeDigest = payload.volume.semantic_sha256;
        volumeCanvas.dataset.exposedFaces = String(stats.exposedFaces);
        const caveLabel = payload.volume.authority.startsWith("exact_captured")
          ? "enclosed void source cells"
          : "cave cells carved";
        $("wgVolumeStats").textContent =
          `${stats.dimensions.join("×")} cells · ${stats.occupiedVoxels.toLocaleString()} occupied · ` +
          `${stats.exposedFaces.toLocaleString()} exposed faces · ` +
          `${stats.carvedCaveVoxels.toLocaleString()} ${caveLabel}` +
          ( $("wgCaves").checked
            ? ` · ${$("wgCutAxis").value.toUpperCase()} cut ${$("wgCutDepth").value}%`
            : "" );
        return;
      } catch (error) {
        studio.volumeRendererError = error.message;
        console.error("[worldgen-volume]", error);
      }
    }
  }

  volumeCanvas.hidden = true;
  canvas.hidden = false;
  $("wgVolumeStats").textContent = studio.renderStyle === "voxels"
    ? `2D fallback · ${studio.volumeRendererError || "WebGL2 unavailable"}`
    : "surface compatibility view · volume hidden";
  const [width, height] = fit(canvas);
  const g = g2d(canvas);
  g.clearRect(0, 0, width, height);
  const backdrop = g.createLinearGradient(0, 0, 0, height);
  backdrop.addColorStop(0, css("--panel-2"));
  backdrop.addColorStop(1, css("--sunk"));
  g.fillStyle = backdrop;
  g.fillRect(0, 0, width, height);

  const grid = payload.grid;
  const n = grid.resolution;
  const halfX = grid.world_width * .5;
  const halfZ = grid.world_depth * .5;
  const stepX = grid.step_x;
  const stepZ = grid.step_z;
  const centerHeight = payload.metrics.mean_height;
  const colors = new Map(payload.biome_legend.map((entry) => [entry.id, entry.color]));
  const drawables = [];
  const showGrid = $("wgGrid").checked;

  function terrainFace(world, color, stroke = null, alpha = 1, bias = 0) {
    const points = world.map((point) => projection(point, width, height, centerHeight));
    drawables.push({
      depth: points.reduce((sum, point) => sum + point.z, 0) / points.length + bias,
      draw(context) { polygon(context, points, color, stroke, alpha); },
    });
  }

  let voxelModel = null;
  if (studio.renderStyle === "voxels") {
    voxelModel = voxelSurface(payload);
    const { unit, columnsX, columnsZ, heights, biomes } = voxelModel;
    const seam = showGrid ? "rgba(225,239,246,.42)" : "rgba(10,16,18,.3)";
    const isWater = (biome) => biome === 0 || biome === 1;
    // A real voxel terrain preview needs visible mass. The old boundary used
    // one shallow skirt cube, which made even rugged terrain read as a folded
    // map. Six quantized soil/stone layers make this a readable terrain slice
    // without the far-wall occlusion caused by a single deep global base.
    const boundaryDepth = unit * 6;
    for (let z = 0; z < columnsZ; z++) {
      const wz = -voxelModel.halfZ + z * unit;
      const z1 = Math.min(voxelModel.halfZ, wz + unit);
      for (let x = 0; x < columnsX; x++) {
        const wx = -voxelModel.halfX + x * unit;
        const x1 = Math.min(voxelModel.halfX, wx + unit);
        const surfaceY = heights[z][x];
        const heightTone = .9 + Math.max(-.08,
          Math.min(.12, (surfaceY - centerHeight) / 180));
        const baseColor = colors.get(biomes[z][x]) || "#777b78";
        const color = shade(baseColor, heightTone);
        terrainFace([
          [wx, surfaceY, wz], [x1, surfaceY, wz],
          [x1, surfaceY, z1], [wx, surfaceY, z1],
        ], shade(color, 1.08), seam);

        const neighbors = [
          { h: x > 0 ? heights[z][x - 1] : surfaceY - boundaryDepth, tone: .66,
            face: (lower, upper) => [[wx, upper, wz], [wx, upper, z1], [wx, lower, z1], [wx, lower, wz]] },
          { h: x + 1 < columnsX ? heights[z][x + 1] : surfaceY - boundaryDepth, tone: .82,
            face: (lower, upper) => [[x1, upper, z1], [x1, upper, wz], [x1, lower, wz], [x1, lower, z1]] },
          { h: z > 0 ? heights[z - 1][x] : surfaceY - boundaryDepth, tone: .72,
            face: (lower, upper) => [[x1, upper, wz], [wx, upper, wz], [wx, lower, wz], [x1, lower, wz]] },
          { h: z + 1 < columnsZ ? heights[z + 1][x] : surfaceY - boundaryDepth, tone: .9,
            face: (lower, upper) => [[wx, upper, z1], [x1, upper, z1], [x1, lower, z1], [wx, lower, z1]] },
        ];
        for (const side of neighbors) {
          for (let upper = surfaceY; upper > side.h; upper -= unit) {
            const lower = Math.max(side.h, upper - unit);
            const band = (Math.round(upper / unit) & 1) ? .97 : 1.0;
            const depth = surfaceY - upper;
            const layerColor = depth < unit * 2
              ? color
              : (depth < unit * 5 ? "#705c43" : "#596166");
            terrainFace(side.face(lower, upper),
              shade(layerColor, side.tone * band), seam);
          }
        }

        if ($("wgWater").checked && isWater(biomes[z][x])) {
          const waterY = Math.ceil((payload.config.water_level + .12) / unit) * unit;
          if (waterY > surfaceY) {
            terrainFace([
              [wx, waterY, wz], [x1, waterY, wz],
              [x1, waterY, z1], [wx, waterY, z1],
            ], "#3d91b4", "rgba(184,229,244,.48)", .72, -.02);
          }
        }
      }
    }
  } else {
    for (let z = 0; z < n - 1; z++) {
      const wz = -halfZ + z * stepZ;
      for (let x = 0; x < n - 1; x++) {
        const wx = -halfX + x * stepX;
        const average = (
          grid.heights[z][x] + grid.heights[z][x + 1] +
          grid.heights[z + 1][x + 1] + grid.heights[z + 1][x]
        ) / 4;
        const slope = grid.slopes[z][x];
        const heightTone = .87 + Math.max(-.08,
          Math.min(.12, (average - centerHeight) / 180));
        const orientation = .98 - Math.min(.18, slope * .08);
        const color = shade(colors.get(grid.biomes[z][x]) || "#777b78",
          heightTone * orientation);
        const world = [
          [wx, grid.heights[z][x], wz],
          [wx + stepX, grid.heights[z][x + 1], wz],
          [wx + stepX, grid.heights[z + 1][x + 1], wz + stepZ],
          [wx, grid.heights[z + 1][x], wz + stepZ],
        ];
        const points = world.map((point) => projection(point, width, height, centerHeight));
        // The optional smooth view keeps planar triangles; the default voxel
        // view above uses flat block tops plus exposed vertical cliff faces.
        for (const triangle of [[points[0], points[1], points[2]], [points[0], points[2], points[3]]]) {
          const depth = triangle.reduce((sum, point) => sum + point.z, 0) / 3;
          drawables.push({
            depth,
            draw(context) {
              polygon(context, triangle, color, showGrid ? "rgba(230,240,250,.14)" : null);
            },
          });
        }

        if ($("wgWater").checked && [0, 1].includes(grid.biomes[z][x])) {
          const waterY = payload.config.water_level + .12;
          const waterPoints = [
            [wx, waterY, wz], [wx + stepX, waterY, wz],
            [wx + stepX, waterY, wz + stepZ], [wx, waterY, wz + stepZ],
          ].map((point) => projection(point, width, height, centerHeight));
          const waterDepth = waterPoints.reduce((sum, point) => sum + point.z, 0) / 4 - .02;
          drawables.push({
            depth: waterDepth,
            draw(context) { polygon(context, waterPoints, "#3d91b4", "rgba(190,235,255,.18)", .72); },
          });
        }
      }
    }
  }

  if ($("wgTrees").checked) {
    for (const tree of payload.trees) {
      if (voxelModel) {
        pushVoxelTree(drawables, tree, voxelModel.unit, width, height, centerHeight);
      } else {
        drawables.push(treeDrawable(tree, width, height, centerHeight));
      }
    }
  }
  if ($("wgStructures").checked) {
    for (const structure of payload.structures) {
      drawables.push(structureDrawable(structure, width, height, centerHeight));
    }
  }
  if ($("wgCaves").checked && payload.caves.segments.length) {
    drawables.push(caveDrawable(payload.caves, width, height, centerHeight));
  }
  drawables.push(spawnDrawable(payload.spawn, width, height, centerHeight));
  for (let order = 0; order < drawables.length; order++) drawables[order].order = order;
  drawables.sort((a, b) => {
    const depth = b.depth - a.depth;
    return Math.abs(depth) > 1e-7 ? depth : a.order - b.order;
  });
  for (const drawable of drawables) drawable.draw(g);

}

function wireCamera() {
  for (const canvas of [$("wgView"), $("wgVoxelView")]) {
    let drag = null;
    canvas.addEventListener("pointerdown", (event) => {
      drag = { x: event.clientX, y: event.clientY, button: event.button, shift: event.shiftKey };
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!drag) return;
      const dx = Math.max(-48, Math.min(48, event.clientX - drag.x));
      const dy = Math.max(-48, Math.min(48, event.clientY - drag.y));
      if (drag.button === 2 || drag.shift) {
        const units = studio.camera.dist / Math.max(420, canvas.clientHeight) * .72;
        const cy = Math.cos(studio.camera.yaw), sy = Math.sin(studio.camera.yaw);
        studio.camera.panX += (-dx * cy - dy * sy) * units;
        studio.camera.panZ += (dx * sy - dy * cy) * units;
      } else {
        studio.camera.yaw = (studio.camera.yaw + dx * .0055) % (Math.PI * 2);
        if (studio.mode === "3d") {
          studio.camera.pitch = Math.max(.22,
            Math.min(1.28, studio.camera.pitch + dy * .0045));
        }
      }
      drag.x = event.clientX;
      drag.y = event.clientY;
      scheduleDraw();
    });
    canvas.addEventListener("pointerup", (event) => {
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
      drag = null;
    });
    canvas.addEventListener("pointercancel", () => { drag = null; });
    canvas.addEventListener("lostpointercapture", () => { drag = null; });
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("dblclick", () => resetCamera());
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const size = Math.max(
        studio.preview?.config.world_width || 128,
        studio.preview?.config.world_depth || 128,
      );
      const delta = Math.max(-160, Math.min(160, event.deltaY));
      studio.camera.dist = Math.max(size * .45, Math.min(size * 4.5,
        studio.camera.dist * Math.exp(delta * .0015)));
      scheduleDraw();
    }, { passive: false });
    new ResizeObserver(scheduleDraw).observe(canvas);
  }
}

function downloadDesign() {
  if (!studio.preview) return;
  const payload = {
    schema: "hytalerl_worldgen_v2_console_export_v1",
    version: 1,
    digest: studio.preview.digest,
    design: studio.preview.design,
    config: studio.preview.config,
    preview_receipt: {
      quality: studio.preview.quality,
      metrics: studio.preview.metrics,
      warnings: studio.preview.warnings,
      not_native_capture: true,
    },
  };
  const blob = new Blob([JSON.stringify(payload, null, 2) + "\n"], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${studio.preview.config.name}-${studio.preview.digest.slice(0, 10)}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

function downloadBatch() {
  if (!studio.batch) return;
  const blob = new Blob(
    [JSON.stringify(studio.batch, null, 2) + "\n"],
    { type: "application/json" },
  );
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${studio.preview?.config.name || "worldgen"}-seed-plan-${studio.batch.seed_range.count}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

function showBatch(payload) {
  const aggregate = payload.aggregate;
  $("wgBatchState").textContent = `${payload.seed_range.count} seeds · ${aggregate.seeds_per_second}/s`;
  $("wgBatchFacts").innerHTML = [
    metric("unique terrain", `${aggregate.unique_terrain_fields}/${payload.seed_range.count}`,
      aggregate.unique_terrain_fields === payload.seed_range.count ? "pos" : "neg"),
    metric("unique structures", `${aggregate.unique_structure_layouts}/${payload.seed_range.count}`),
    metric("unique routes", `${aggregate.unique_route_graphs}/${payload.seed_range.count}`),
    metric("preview plausible", `${aggregate.plausible}/${payload.seed_range.count}`,
      aggregate.plausible === payload.seed_range.count ? "pos" : ""),
    metric("native packs", payload.native_seed_plan.asset_packs_required),
    metric("capture tiles / seed", payload.native_seed_plan.extent_plan.capture_tiles.total,
      payload.native_seed_plan.extent_plan.jax_world.shared_world_id ? "pos" : "neg"),
    metric("elapsed", `${aggregate.generation_seconds.toFixed(2)}s`),
  ].join("");
  const visible = payload.rows.slice(0, 120);
  $("wgBatchRows").innerHTML = visible.map((row) =>
    `<button type="button" class="wg-batch-row ghost" data-batch-index="${row.index}">` +
      `<span class="seed mono">seed ${row.seed}</span>` +
      `<span class="grow"><b>${esc(row.world_genome ?
        `${row.world_genome.region_count} region hyperparam world` :
        row.resolved_config.terrain_style.replaceAll("_", " "))}</b><br>` +
      `<small>${esc(row.native_profile.replaceAll("_", " "))} · ` +
      `${row.metrics.structure_kind_count} structure kinds · ${row.metrics.route_edge_count} routes</small></span>` +
      `<span class="count">${percent(row.metrics.traversable_fraction)} walkable</span></button>`
  ).join("") + (payload.rows.length > visible.length
    ? `<div class="empty">${payload.rows.length - visible.length} more seeds are in the downloaded plan.</div>` : "");
}

async function generateBatch() {
  $("wgGenerateBatch").disabled = true;
  $("wgBatchState").textContent = "generating…";
  try {
    const payload = await jsonFetch("/api/worldgen/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        config: controls(),
        start_seed: Number($("wgBatchStart").value),
        count: Number($("wgBatchCount").value),
        stride: Number($("wgBatchStride").value),
        resolution: Number($("wgBatchResolution").value),
        strategy: $("wgBatchStrategy").value,
        variation_strength: Number($("wgBatchStrength").value),
      }),
    });
    studio.batch = payload;
    showBatch(payload);
    $("wgDownloadBatch").disabled = false;
  } catch (error) {
    $("wgBatchState").textContent = "batch failed";
    $("wgBatchFacts").innerHTML = `<div class="flag sev-crit">${esc(error.message)}</div>`;
  } finally {
    $("wgGenerateBatch").disabled = false;
  }
}

async function previewBatchRow(index) {
  const row = studio.batch?.rows[index];
  if (!row) return;
  setControls(row.resolved_config);
  $("wgPreset").value = "custom";
  $("wgPresetNote").textContent = `Loaded seed ${row.seed} from the current batch.`;
  resetCamera(Math.max(row.resolved_config.world_width, row.resolved_config.world_depth));
  await requestPreview();
}

function showBuilds(body) {
  if (!body.builds.length) {
    $("wgBuilds").innerHTML = `<div class="empty">No local Studio packs.</div>`;
    return;
  }
  $("wgBuilds").innerHTML = body.builds.map((item) =>
    `<div class="wg-saved-row">` +
      `<span class="grow"><b>${esc(item.environment_id)}</b><br>` +
      `<small>${esc(item.native_profile.replaceAll("_", " "))}</small></span>` +
      `<span class="count mono">${esc(item.pack_semantic_sha256.slice(0, 12))}</span></div>`
  ).join("");
}

async function loadBuilds() {
  try { showBuilds(await jsonFetch("/api/worldgen/builds")); }
  catch (error) {
    $("wgBuilds").innerHTML = `<div class="flag sev-warn">${esc(error.message)}</div>`;
  }
}

async function compileNative() {
  $("wgCompile").disabled = true;
  $("wgCompileState").textContent = "hashing + compiling…";
  $("wgCompileFacts").innerHTML = `<div class="flag sev-warn">Offline build in progress. No server is being started.</div>`;
  try {
    const result = await jsonFetch("/api/worldgen/compile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config: controls() }),
    });
    $("wgCompileState").textContent = result.reused ? "validated existing" : "compiled";
    const extent = result.requested_capture_extent;
    $("wgCompileFacts").innerHTML =
      `<div class="flag sev-good">${esc(result.asset_id)} · ${esc(result.native_profile.replaceAll("_", " "))} · ` +
      `${result.output_file_count} files · not deployed</div>` +
      (result.boundary.compiler_family_reference_validation
        ? `<div class="flag sev-good">Compiler family has a pinned live 0.5.7 reference. ` +
          `This exact pack remains capture-required.</div>`
        : "") +
      `<div class="flag ${extent.jax_world.shared_world_id ? "sev-good" : "sev-warn"}">` +
      `${extent.capture_tiles.x} x ${extent.capture_tiles.z} capture cores · ` +
      `1 composite JAX world · ${esc(extent.status.replaceAll("_", " "))}</div>`;
    await loadBuilds();
  } catch (error) {
    $("wgCompileState").textContent = "compile failed";
    $("wgCompileFacts").innerHTML = `<div class="flag sev-crit">${esc(error.message)}</div>`;
  } finally {
    $("wgCompile").disabled = false;
  }
}

async function saveDesign() {
  $("wgSave").disabled = true;
  $("wgSaveState").textContent = "validating and saving…";
  try {
    const saved = await jsonFetch("/api/worldgen/designs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(controls()),
    });
    $("wgSaveState").textContent = `Saved ${saved.design_id} · ${saved.path}`;
    await loadSaved();
  } catch (error) {
    $("wgSaveState").textContent = `Could not save: ${error.message}`;
  } finally {
    $("wgSave").disabled = false;
  }
}

async function loadSaved() {
  let body;
  try { body = await jsonFetch("/api/worldgen/designs"); }
  catch { return; }
  $("wgSavedCount").textContent = `${body.designs.length} local`;
  if (!body.designs.length) {
    $("wgSaved").innerHTML = `<div class="empty">No saved designs yet.</div>`;
    return;
  }
  $("wgSaved").innerHTML = body.designs.map((item) =>
    `<button type="button" class="wg-saved-row ghost" data-design-id="${esc(item.design_id)}">` +
    `<span class="grow"><b>${esc(item.name)}</b><br><small>${esc(item.digest.slice(0, 12))}</small></span>` +
    `<span class="seed mono">seed ${esc(item.seed)}</span>` +
    `<span class="count">${esc(new Date(item.saved_at).toLocaleString())}</span></button>`).join("");
}

async function restoreDesign(designId) {
  $("wgSaveState").textContent = `Loading ${designId}…`;
  try {
    const record = await jsonFetch(`/api/worldgen/designs/${encodeURIComponent(designId)}`);
    setControls(record.config);
    $("wgPreset").value = "custom";
    $("wgPresetNote").textContent = "Loaded from a saved local design receipt.";
    resetCamera(Math.max(record.config.world_width, record.config.world_depth));
    await requestPreview();
    $("wgSaveState").textContent = `Loaded ${designId}.`;
  } catch (error) {
    $("wgSaveState").textContent = `Could not load: ${error.message}`;
  }
}

function selectedGoal() {
  return studio.gameOptions.goals.find((goal) => goal.kind === $("cmGoal").value);
}

function renderGoalFields(values = {}) {
  const goal = selectedGoal();
  $("cmGoalNote").textContent = goal.description;
  $("cmGoalFields").innerHTML = goal.fields.map((field) => {
    const value = values[field.name] ?? field.default;
    if (field.type === "choice") {
      const choices = field.choices.map((choice) =>
        `<option value="${esc(choice)}"${choice === value ? " selected" : ""}>${esc(choice)}</option>`).join("");
      return `<label class="f"><span>${esc(field.label)}</span>` +
        `<select data-cm-goal-field="${esc(field.name)}" data-cm-goal-type="choice">${choices}</select></label>`;
    }
    const step = field.step ?? 1;
    return `<label class="f"><span>${esc(field.label)}</span>` +
      `<input type="number" data-cm-goal-field="${esc(field.name)}" data-cm-goal-type="number" ` +
      `min="${field.min}" max="${field.max}" step="${step}" value="${esc(value)}"></label>`;
  }).join("");
}

const rewardIds = {
  native_scale: "cmNativeScale",
  progress_scale: "cmProgressScale",
  discount: "cmRewardDiscount",
  state_scale: "cmStateScale",
  event_scale: "cmEventScale",
  completion_bonus: "cmCompletionBonus",
  failure_penalty: "cmFailurePenalty",
  step_penalty: "cmStepPenalty",
  clip: "cmRewardClip",
};

function setRewardControls(values) {
  for (const [name, id] of Object.entries(rewardIds)) {
    if (name !== "clip") $(id).value = values[name];
  }
  $("cmClipEnabled").checked = values.clip !== null;
  $("cmRewardClip").disabled = values.clip === null;
  $("cmRewardClip").value = values.clip ?? 10;
}

function rewardControls() {
  return {
    preset: $("cmRewardPreset").value,
    ...Object.fromEntries(Object.entries(rewardIds).map(([name, id]) => [
      name,
      name === "clip" && !$("cmClipEnabled").checked ? null : Number($(id).value),
    ])),
  };
}

function setSceneControls(values) {
  $("cmAgentHealth").value = values.agent_max_health;
  $("cmTargetHealth").value = values.target_max_health;
  $("cmSensorRange").value = values.sensor_range;
  $("cmMicroticks").value = values.microticks;
  $("cmTargetActive").checked = values.target_active;
}

function sceneControls() {
  return {
    agent_max_health: Number($("cmAgentHealth").value),
    target_max_health: Number($("cmTargetHealth").value),
    sensor_range: Number($("cmSensorRange").value),
    microticks: Number($("cmMicroticks").value),
    target_active: $("cmTargetActive").checked,
  };
}

function setPoolControls(values) {
  $("cmPoolCount").value = values.count;
  $("cmPoolStride").value = values.seed_stride;
  $("cmAssignmentKey").value = values.assignment_key;
}

function poolControls() {
  return {
    count: Number($("cmPoolCount").value),
    seed_stride: Number($("cmPoolStride").value),
    assignment_key: Number($("cmAssignmentKey").value),
  };
}

function buildGameControls() {
  const options = studio.gameOptions;
  $("cmGoal").innerHTML = options.goals.map((goal) =>
    `<option value="${esc(goal.kind)}">${esc(goal.label)}</option>`).join("");
  const loadouts = options.loadouts.map((name) =>
    `<option value="${esc(name)}">${esc(name)}</option>`).join("");
  $("cmLoadout").innerHTML = loadouts;
  $("cmOpponent").innerHTML = loadouts;
  $("cmRewardPreset").innerHTML = Object.keys(options.reward_presets).map((name) =>
    `<option value="${esc(name)}">${esc(name)}</option>`).join("");
  $("cmName").value = options.defaults.name;
  $("cmDescription").value = options.defaults.description;
  $("cmGoal").value = options.defaults.goal;
  $("cmLoadout").value = options.defaults.loadout;
  $("cmOpponent").value = options.defaults.opponent;
  $("cmArmed").checked = options.defaults.opponent_armed;
  $("cmRewardPreset").value = options.defaults.reward.preset;
  setRewardControls(options.defaults.reward);
  setSceneControls(options.defaults.scene);
  setPoolControls(options.defaults.world_pool);
  $("cmPoolCount").max = options.world_pool_limit;
  renderGoalFields();
}

function goalParameters() {
  return Object.fromEntries(Array.from(
    $("cmGoalFields").querySelectorAll("[data-cm-goal-field]"),
    (input) => [
      input.dataset.cmGoalField,
      input.dataset.cmGoalType === "choice" ? input.value : Number(input.value),
    ],
  ));
}

function showCustomGames(body) {
  $("cmSavedCount").textContent = `${body.minigames.length} drafts`;
  if (!body.minigames.length) {
    $("cmGames").innerHTML = `<div class="empty">No saved custom minigames.</div>`;
    return;
  }
  $("cmGames").innerHTML = body.minigames.map((game) => {
    const jaxState = game.jax_ready
      ? `JAX ${esc(game.jax_backend)} ready`
      : "JAX unchecked";
    return `<button type="button" class="wg-saved-row ghost" data-cm-game-id="${esc(game.game_id)}">` +
      `<span class="grow"><b>${esc(game.name)}</b><br>` +
      `<small>${esc(game.goal.replaceAll("_", " "))} · ${esc(game.world)} · ${esc(game.reward_preset)}</small><br>` +
      `<small class="mono">world design ${esc(game.design_id || "unsaved")}</small></span>` +
      `<span class="seed mono">seed ${esc(game.seed)} × ${esc(game.world_pool_size)}</span>` +
      `<span class="count">${jaxState} · needs capture</span></button>`;
  }).join("");
}

async function loadCustomGames() {
  try { showCustomGames(await jsonFetch("/api/custom/minigames")); }
  catch (error) {
    $("cmGames").innerHTML = `<div class="flag sev-warn">${esc(error.message)}</div>`;
  }
}

async function createCustomGame(event) {
  event.preventDefault();
  $("cmCreate").disabled = true;
  $("cmState").textContent = "Compiling narrow JAX goal + reward contracts and checking world uniqueness…";
  try {
    const created = await jsonFetch("/api/custom/minigames", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: $("cmName").value,
        description: $("cmDescription").value,
        goal: $("cmGoal").value,
        goal_parameters: goalParameters(),
        loadout: $("cmLoadout").value,
        opponent: $("cmOpponent").value,
        opponent_armed: $("cmArmed").checked,
        reward: rewardControls(),
        scene: sceneControls(),
        world: controls(),
        world_pool: poolControls(),
      }),
    });
    const compiled = created.jax_cache_hit
      ? `JAX ${created.jax_backend} executable reused from cache`
      : `JAX ${created.jax_backend} goal + reward compiled in ${created.jax_compile_ms} ms`;
    $("cmState").textContent = `${created.reused ? "Loaded" : "Saved"} ${created.game_id}. ` +
      `World saved as design ${created.design_id}, selectable on the Train tab. ` +
      `${compiled} from ${created.jax_groups.join(", ")}; exact Region capture remains.`;
    await loadCustomGames();
  } catch (error) {
    $("cmState").textContent = `Could not create minigame: ${error.message}`;
  } finally {
    $("cmCreate").disabled = false;
  }
}

async function restoreCustomGame(gameId) {
  $("cmState").textContent = `Loading ${gameId}…`;
  try {
    const record = await jsonFetch(`/api/custom/minigames/${encodeURIComponent(gameId)}`);
    const task = record.task;
    $("cmName").value = task.name;
    $("cmDescription").value = task.description;
    $("cmGoal").value = task.goal.kind;
    renderGoalFields(task.goal.parameters);
    $("cmLoadout").value = task.loadout;
    $("cmOpponent").value = task.difficulties[0].opponent_profile;
    $("cmArmed").checked = task.difficulties[0].opponent_armed;
    const defaults = studio.gameOptions.defaults;
    $("cmRewardPreset").value = task.reward_preset ?? defaults.reward.preset;
    setRewardControls({ ...defaults.reward, ...(task.reward?.config ?? {}) });
    setSceneControls({
      agent_max_health: task.difficulties[0].agent_max_health ?? defaults.scene.agent_max_health,
      target_max_health: task.difficulties[0].target_max_health ?? defaults.scene.target_max_health,
      sensor_range: task.difficulties[0].sensor_range ?? defaults.scene.sensor_range,
      microticks: task.scene_options?.microticks ?? defaults.scene.microticks,
      target_active: task.scene_options?.target_active ?? defaults.scene.target_active,
    });
    setPoolControls(record.world.pool ?? defaults.world_pool);
    setControls(record.world.config);
    $("wgPreset").value = "custom";
    resetCamera(Math.max(record.world.config.world_width, record.world.config.world_depth));
    await requestPreview();
    const contract = record.jax_contract;
    const jaxState = contract?.criterion_compiled && contract?.reward_compiled
      ? `JAX ${contract.backend} ready from ${contract.required_groups.join(", ")}`
      : "JAX contract unchecked";
    $("cmState").textContent = `Loaded ${gameId} · ${jaxState} · ${record.status.replaceAll("_", " ")}.`;
  } catch (error) {
    $("cmState").textContent = `Could not load minigame: ${error.message}`;
  }
}

export function worldgenTabChanged(name) {
  studio.visible = name === "custom";
  if (studio.visible) requestAnimationFrame(draw);
}

export async function wireWorldgen() {
  try {
    [studio.options, studio.gameOptions] = await Promise.all([
      jsonFetch("/api/worldgen/options"),
      jsonFetch("/api/custom/minigames/options"),
    ]);
  } catch (error) {
    showFailure(`Could not load Custom Studio options: ${error.message}`);
    return;
  }
  buildControls();
  buildGameControls();
  showPipeline();
  wireCamera();
  $("wgBatchStrategy").innerHTML = studio.options.batch.strategies.map((item) =>
    `<option value="${esc(item.id)}">${esc(item.label)}</option>`).join("");
  $("wgBatchStrategy").value = "parametric_worlds";
  $("wgBatchResolution").innerHTML = studio.options.batch.resolutions.map((value) =>
    `<option value="${value}">${value}² fast samples</option>`).join("");
  $("wgBatchCount").max = studio.options.batch.maximum_seeds;
  // `wireTabs()` runs before feature modules are wired.  A direct
  // `#custom` page load therefore happened before our tab listener existed;
  // recover the actual DOM state here so its first preview is not drawn into
  // a permanently "hidden" studio state.
  studio.visible = !document.querySelector('[data-panel="custom"]').hidden;

  $("wgPreset").addEventListener("change", (event) => {
    if (event.target.value !== "custom") applyPreset(event.target.value);
  });
  $("wgGenerate").addEventListener("click", requestPreview);
  $("wgRandomWorld").addEventListener("click", () => {
    $(fieldId("generation_mode")).value = "procedural_genome";
    if ($(fieldId("cave_style")).value !== "none") {
      $(fieldId("cave_style")).value = "procedural";
    }
    if ($(fieldId("structure_set")).value !== "none") {
      $(fieldId("structure_set")).value = "procedural_mix";
      $(fieldId("structure_layout")).value = "terrain_network";
    }
    const random = new Uint32Array(1);
    crypto.getRandomValues(random);
    $(fieldId("seed")).value = random[0] & 0x7fffffff;
    $("wgBatchStart").value = $(fieldId("seed")).value;
    $("wgPreset").value = "custom";
    $("wgPresetNote").textContent =
      "Seed sampled directly from the active hyperparameters; no named preset was selected.";
    syncGenerationMode();
    requestPreview();
  });
  $("wgNewSeed").addEventListener("click", () => {
    const random = new Uint32Array(1);
    crypto.getRandomValues(random);
    $(fieldId("seed")).value = random[0] & 0x7fffffff;
    $("wgBatchStart").value = $(fieldId("seed")).value;
    $("wgPreset").value = "custom";
    requestPreview();
  });
  $("wgSave").addEventListener("click", saveDesign);
  $("wgDownload").addEventListener("click", downloadDesign);
  $("wgGenerateBatch").addEventListener("click", generateBatch);
  $("wgDownloadBatch").addEventListener("click", downloadBatch);
  $("wgBatchRows").addEventListener("click", (event) => {
    const row = event.target.closest("[data-batch-index]");
    if (row) previewBatchRow(Number(row.dataset.batchIndex));
  });
  $("wgBatchStrategy").addEventListener("change", (event) => {
    $("wgBatchStrength").disabled = event.target.value === "native_seeds";
  });
  $("wgCompile").addEventListener("click", compileNative);
  $("wgRefreshBuilds").addEventListener("click", loadBuilds);
  $("wgRefreshCaptures").addEventListener("click", loadCaptures);
  $("wgLoadCapture").addEventListener("click", loadCapturedVolume);
  $("wgSaved").addEventListener("click", (event) => {
    const row = event.target.closest("[data-design-id]");
    if (row) restoreDesign(row.dataset.designId);
  });
  $("cmGoal").addEventListener("change", () => renderGoalFields());
  $("cmRewardPreset").addEventListener("change", (event) =>
    setRewardControls(studio.gameOptions.reward_presets[event.target.value]));
  $("cmClipEnabled").addEventListener("change", (event) => {
    $("cmRewardClip").disabled = !event.target.checked;
  });
  $("cmForm").addEventListener("submit", createCustomGame);
  $("cmGames").addEventListener("click", (event) => {
    const row = event.target.closest("[data-cm-game-id]");
    if (row) restoreCustomGame(row.dataset.cmGameId);
  });
  $("wgViewMode").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-mode]");
    if (!button) return;
    studio.mode = button.dataset.mode;
    for (const item of $("wgViewMode").querySelectorAll("button")) {
      item.setAttribute("aria-pressed", String(item === button));
    }
    $("wgCamHint").textContent = studio.mode === "top"
      ? "drag rotate · scroll zoom · orthographic"
      : "drag orbit · shift/right-drag pan · scroll zoom";
    resetCamera();
  });
  $("wgRenderStyle").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-render]");
    if (!button) return;
    studio.renderStyle = button.dataset.render;
    for (const item of $("wgRenderStyle").querySelectorAll("button")) {
      item.setAttribute("aria-pressed", String(item === button));
    }
    syncCutawayControls();
    scheduleDraw();
  });
  $("wgResetCamera").addEventListener("click", () => resetCamera());
  for (const id of ["wgWater", "wgTrees", "wgStructures", "wgGrid"]) {
    $(id).addEventListener("change", scheduleDraw);
  }
  $("wgCaves").addEventListener("change", () => {
    syncCutawayControls();
    scheduleDraw();
  });
  $("wgCutAxis").addEventListener("change", scheduleDraw);
  $("wgCutDepth").addEventListener("input", (event) => {
    $("wgCutDepthValue").textContent = `${Number(event.target.value).toFixed(0)}%`;
    scheduleDraw();
  });
  $("wgExaggeration").addEventListener("input", (event) => {
    $("wgExaggerationValue").textContent = `${Number(event.target.value).toFixed(2)}×`;
    scheduleDraw();
  });
  matchMedia("(prefers-color-scheme:dark)").addEventListener("change", scheduleDraw);

  const first = studio.options.presets[0];
  $("wgPreset").value = first.id;
  setControls(first.values);
  $("wgPresetNote").textContent = first.description;
  $("wgBatchStart").value = first.values.seed;
  if (pendingSeed !== null) loadWorldgenSeed(pendingSeed);
  $("wgBatchStrategy").dispatchEvent(new Event("change"));
  syncCutawayControls();
  await Promise.all([
    requestPreview(), loadSaved(), loadBuilds(), loadCaptures(), loadCustomGames(),
  ]);
}
