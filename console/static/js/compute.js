/* The Compute workspace: devices, what a run costs, and reloading code.
 *
 * Device memory is reported as a *delta* per run, because "what did this run
 * add" is the number that decides how many environments fit at once. On the
 * CPU backend there are no device stats at all and the fields are absent
 * rather than zero -- 0 MB of VRAM reads as "used no GPU memory" when the
 * truth is "there is no GPU".
 */

import { $, state } from "./state.js";
import { applyTrainingComposition } from "./train.js";

let workflows = [];

export async function loadDevices() {
  const facts = $("deviceFacts");
  if (!facts) return;

  let d, compute;
  try {
    const [deviceResponse, computeResponse] = await Promise.all([
      fetch("/api/devices"), fetch("/api/compute"),
    ]);
    d = await deviceResponse.json();
    compute = computeResponse.ok ? await computeResponse.json() : null;
  } catch {
    return;                        // keep the last good render on a blip
  }
  state.devices = d;
  if (compute) {
    state.compute = compute;
    renderComposition(compute);
  }

  const rows = [
    ["backend", d.backend, d.backend === "cpu" ? "" : "pos"],
    ["devices", d.device_count, ""],
    ["platform", d.platform, ""],
    ["jax", d.jax, ""],
    ["float64", d.x64_enabled ? "on" : "off", d.x64_enabled ? "neg" : "pos"],
    ["compilation cache", d.compilation_cache || "not set", ""],
  ];
  for (const dev of d.devices || []) {
    rows.push([`device ${dev.id}`, `${dev.type} (${dev.kind})`, ""]);
    if (dev.in_use_mb !== undefined) {
      rows.push([`device ${dev.id} in use`, `${dev.in_use_mb} MB`, ""]);
    }
    if (dev.peak_mb !== undefined) {
      rows.push([`device ${dev.id} peak`, `${dev.peak_mb} MB`, ""]);
    }
  }

  $("deviceBackend").textContent = d.backend;
  facts.innerHTML = rows.map(([k, v, c]) =>
    `<div class="ro"><i>${esc(k)}</i><b class="${c}">${esc(v)}</b></div>`).join("");
  $("deviceNote").textContent = d.note || "";

  const cache = d.scene_cache || {};
  const entries = cache.entries || [];
  $("cacheRows").innerHTML = entries.length
    ? entries.map((e) =>
        `<div><span class="grow">${esc(e.digest)}</span>` +
        `<span class="num">${esc(e.num_envs)} envs</span></div>`).join("")
    : `<div><span class="grow">nothing cached — the next run builds cold</span></div>`;
}

function currentWorkflow() {
  return workflows.find((item) => item.stage === $("computeStage").value);
}

function renderComposition(payload) {
  workflows = payload.training_workflows || [];
  const stage = $("computeStage");
  const previous = stage.value;
  stage.innerHTML = workflows.map((item) =>
    `<option value="${esc(item.stage)}">${esc(item.stage.replaceAll("_", " "))}</option>`
  ).join("");
  stage.value = workflows.some((item) => item.stage === previous)
    ? previous : workflows[0]?.stage || "";
  renderWorkflow(!$("computePreset").options.length);

  const runtime = payload.runtime || {};
  const allocator = runtime.allocator_plan || {};
  const worker = runtime.resident_worker || {};
  const terrainCache = runtime.terrain_replay_cache || {};
  const generatedTraining = runtime.generated_training_world_cache?.resident || {};
  const generatedReplay = runtime.generated_replay_cache?.volumes || {};
  const active = runtime.active_job;
  $("computeRuntimeState").textContent = active
    ? `${active.status} · ${active.num_envs} envs` : worker.running ? "worker warm" : "idle";
  $("computeRuntimeFacts").innerHTML = [
    ["allocator", allocator.mode || "unknown"],
    ["reservation", allocator.preallocate
      ? `${Math.round((allocator.fraction || 0) * 100)}% preallocated` : "on demand"],
    ["GPU free", allocator.free_mib === undefined ? "unavailable" : `${allocator.free_mib} MiB`],
    ["resident worker", worker.running ? `PID ${worker.pid}` : "starts on launch"],
    ["process runs", worker.process_run || 0],
    ["last contract", worker.contract_reused ? "reused" : "cold or changed"],
    ["compile cache", runtime.launch_policy?.persistent_compilation_cache || "unavailable"],
    ["terrain replay cache", `${terrainCache.entries || 0} / ${terrainCache.capacity || 0}`],
    ["generated training cache", `${generatedTraining.currsize || 0} / ` +
      `${runtime.generated_training_world_cache?.resident_limit || 0} windows`],
    ["generated replay cache", `${generatedReplay.currsize || 0} / ` +
      `${runtime.generated_replay_cache?.volume_limit || 0}`],
    ["active world", active?.world_artifact_seed ?? "workflow default"],
  ].map(([key, value]) =>
    `<div class="ro"><i>${esc(key)}</i><b>${esc(value)}</b></div>`).join("");
}

function renderWorkflow(setValues = true) {
  const workflow = currentWorkflow();
  if (!workflow) return;
  const presets = Object.keys(workflow.presets || {});
  const preset = $("computePreset");
  const previous = preset.value;
  preset.innerHTML = presets.map((name) =>
    `<option value="${esc(name)}">${esc(name)}</option>`).join("");
  preset.value = presets.includes(previous) ? previous : workflow.default_preset;
  const values = workflow.presets?.[preset.value] || {};
  if (setValues) {
    $("computeEnvs").value = values.batch ?? $("trainEnvs")?.value ?? 1;
    $("computeRollout").value = values.rollout_steps ?? $("trainRollout")?.value ?? 1;
  }

  const world = $("computeWorldSeed");
  const oldWorld = world.value;
  const artifacts = workflow.world_backend?.artifacts || [];
  world.innerHTML = [`<option value="">workflow default</option>`]
    .concat(artifacts.map((item) =>
      `<option value="${item.seed}">${item.seed}</option>`)).join("");
  if ([...world.options].some((item) => item.value === oldWorld)) world.value = oldWorld;
  world.disabled = !artifacts.length;

  const controller = $("computeController");
  const oldController = controller.value;
  const control = (workflow.compute?.controls || [])
    .find((item) => item.key === "target_controller");
  const choices = control?.choices || [];
  controller.innerHTML = choices.map((value) =>
    `<option value="${esc(value)}">${esc(value.replaceAll("_", " "))}</option>`
  ).join("");
  controller.value = choices.includes(oldController)
    ? oldController : workflow.custom?.fixed?.target_mode || choices[0] || "";
  controller.disabled = !choices.length;

  $("computeContract").textContent = workflow.compute?.schema || "no compute extension";
  const features = workflow.compute?.features || [];
  $("computeFeatures").innerHTML = features.length ? features.map((feature) =>
    `<div><code>${esc(feature.key)}</code><span class="grow">${esc(feature.summary)}</span>` +
    `<i class="k ${feature.mode === "enforced" ? "g" : "a"}">${esc(feature.mode)}</i></div>`
  ).join("") : `<div><span class="grow">This stage adds no compute-specific features.</span></div>`;
  renderSummary();
}

function renderSummary() {
  const workflow = currentWorkflow();
  const envs = Number($("computeEnvs").value) || 0;
  const ticks = Number($("computeRollout").value) || 0;
  const scripted = $("computeController").value === "scripted_flee_weave";
  $("computeSummary").innerHTML = [
    ["strategy", workflow?.strategy || "—"],
    ["samples / update", (envs * ticks).toLocaleString()],
    ["policy forwards / env", scripted ? "1 of 2" : "2 of 2"],
    ["world", $("computeWorldSeed").value || "saved workflow default"],
    ["storage", "learner policy only"],
    ["JIT", workflow?.compute?.composition || "stage owned"],
  ].map(([label, value]) =>
    `<span><small>${esc(label)}</small><b title="${esc(value)}">${esc(value)}</b></span>`
  ).join("");
}

/* Per-run cost, shown next to the run it belongs to rather than as a global
   gauge: the interesting question is "what did *this* configuration cost". */
export function runCost() {
  const box = $("runCost");
  if (!box) return;
  const cost = state.data && state.data.summary && state.data.summary.resources;
  if (!cost) { box.innerHTML = ""; return; }

  const rows = [["cpu seconds", cost.cpu_seconds, ""]];
  if (cost.scene_cache_hit !== undefined) {
    rows.push(["scene", cost.scene_cache_hit ? "cache hit" : "cold build",
               cost.scene_cache_hit ? "pos" : ""]);
  }
  if (cost.rss_mb !== undefined) rows.push(["process RSS", `${cost.rss_mb} MB`, ""]);
  if (cost.rss_delta_mb !== undefined) {
    rows.push(["RSS added", `${cost.rss_delta_mb} MB`, ""]);
  }
  for (const [key, value] of Object.entries(cost)) {
    if (key.startsWith("device") && key.endsWith("_delta")) {
      rows.push([key.replace("_delta", " VRAM added"), `${value} MB`, ""]);
    }
  }
  box.innerHTML = `<div class="facts">` + rows.map(([k, v, c]) =>
    `<div class="ro"><i>${esc(k)}</i><b class="${c}">${esc(v)}</b></div>`)
    .join("") + `</div>`;
}

export function wireCompute() {
  $("computeStage")?.addEventListener("change", () => renderWorkflow(true));
  $("computePreset")?.addEventListener("change", () => renderWorkflow(true));
  for (const id of ["computeEnvs", "computeRollout", "computeWorldSeed", "computeController"]) {
    $(id)?.addEventListener("input", renderSummary);
    $(id)?.addEventListener("change", renderSummary);
  }
  $("computeApply")?.addEventListener("click", () => {
    try {
      applyTrainingComposition({
        stage: $("computeStage").value,
        preset: $("computePreset").value,
        num_envs: Number($("computeEnvs").value),
        rollout_steps: Number($("computeRollout").value),
        world_artifact_seed: $("computeWorldSeed").value
          ? Number($("computeWorldSeed").value) : null,
        target_controller: $("computeController").value || null,
      });
      $("computeStatus").textContent = "Applied to Train. Review the contract, then validate and launch.";
    } catch (error) {
      $("computeStatus").textContent = error.message;
    }
  });

  $("clearCache")?.addEventListener("click", async () => {
    const out = await (await fetch("/api/devices/clear-cache", { method: "POST" })).json();
    $("status").textContent = `dropped ${out.scene_handles || 0} scene(s) and ${out.replay_terrain || 0} terrain patch(es)`;
    loadDevices();
  });

  $("hotGo")?.addEventListener("click", async () => {
    const name = $("hotModule").value.trim();
    const result = $("hotResult");
    result.innerHTML = `<p class="note">reloading ${esc(name)}…</p>`;
    try {
      const response = await fetch("/api/reload", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ module: name }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || response.statusText);
      result.innerHTML = `<p class="note">reloaded <b>${esc(name)}</b></p>`;
    } catch (error) {
      // Show the traceback: the old module is still live, so a failed reload
      // means "not applied", and the reason is the whole point of showing it.
      result.innerHTML = `<pre class="trace">${esc(error.message)}</pre>`;
    }
  });
}

function esc(v) {
  return String(v ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
