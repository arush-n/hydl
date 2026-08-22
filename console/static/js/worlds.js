/* The world library, in Evidence.
 *
 * Three sections, because the decision you are making depends entirely on
 * which one a world is in: `live` belongs to the run happening now, `used`
 * has a recorded result depending on it, and `unused` has never been selected
 * and is the safe thing to retire.
 *
 * Everything here is read from `/api/worlds/inventory`. The groups, the
 * terrain measure and the library folder all come from the server, so this
 * panel cannot drift from the library it reports on.
 */

import { $ } from "./state.js";

const SECTIONS = [
  ["live", "Live", "In the working set of the run happening now. Deprecating one takes effect on the next run, not this one."],
  ["used", "Used", "Loaded by an earlier run, so a recorded result depends on it."],
  ["unused", "Unused", "Never selected by any run. Retiring these costs nothing already measured."],
];

let inventory = null;
let loading = false;

export async function loadWorlds() {
  if (loading) return;
  loading = true;
  try {
    const library = $("worldLibrary")?.value || "";
    const split = $("worldSplit")?.value || "train";
    const query = new URLSearchParams({ split });
    if (library) query.set("library", library);
    const response = await fetch(`/api/worlds/inventory?${query}`);
    if (!response.ok) {
      $("worldSections").innerHTML =
        `<p class="empty">Could not read the world library (${response.status}).</p>`;
      return;
    }
    inventory = await response.json();
    drawWorlds();
  } catch (error) {
    $("worldSections").innerHTML =
      `<p class="empty">World library unavailable: ${escape(error.message)}</p>`;
  } finally {
    loading = false;
  }
}

function drawWorlds() {
  const host = $("worldSections");
  if (!host || !inventory) return;
  const worlds = inventory.worlds || [];
  if (!inventory.library) {
    host.innerHTML = `<p class="empty">No world library found on disk.</p>`;
    return;
  }
  fillPickers();
  $("worldCount").textContent =
    `${worlds.length} worlds · ${inventory.deprecated || 0} deprecated`;
  $("worldDirectory").textContent = inventory.directory || "";

  const groups = inventory.sections || {};
  const filter = ($("worldFilter")?.value || "").trim().toLowerCase();
  /* Sorted by flatness, descending, because that is the number a keep or
     deprecate call actually turns on -- an unordered list of 256 seeds makes
     the gentlest and the near-cliff worlds equally hard to find. A seed with no
     terrain census sorts last: unmeasured is not the same as flat. */
  const rank = (world) => (world.flatness == null ? -1 : world.flatness);
  const matches = (world) => !filter
    || String(world.seed).includes(filter)
    || (world.semantic_sha256 || "").toLowerCase().startsWith(filter)
    || (world.state || "").includes(filter);

  host.innerHTML = SECTIONS.map(([key, title, note]) => {
    const seeds = groups[key] || [];
    const rows = worlds
      .filter((world) => world.usage === key && matches(world))
      .sort((a, b) => rank(b) - rank(a));
    const hidden = seeds.length - rows.length;
    return `
      <details class="worldGroup" ${key === "unused" ? "" : "open"}>
        <summary><h3>${title}</h3><span class="count">${rows.length}${
          hidden > 0 ? ` of ${seeds.length}` : ""}</span></summary>
        <p class="note">${note}</p>
        ${rows.length
          ? table(rows)
          : `<p class="empty">${hidden > 0
              ? `None of the ${seeds.length} here match the filter.` : "None."}</p>`}
      </details>`;
  }).join("");
  drawSpread(worlds);

  host.querySelectorAll("button[data-seed]").forEach((button) => {
    button.addEventListener("click", () => decide(
      Number(button.dataset.seed), button.dataset.state || null));
  });
}

/* The library's shape in one line. `worlds/DEV.md` measured that a blindly
   picked working set sat BELOW the library median and included a world at
   0.066 -- effectively all cliff -- so "what am I choosing from" is the first
   question this panel should answer, before any individual row. */
function drawSpread(worlds) {
  const host = $("worldSpread");
  if (!host) return;
  const measured = worlds
    .map((world) => world.flatness)
    .filter((value) => value != null)
    .sort((a, b) => a - b);
  if (!measured.length) {
    host.textContent = "no terrain census for this split";
    return;
  }
  const at = (q) => measured[Math.min(measured.length - 1, Math.floor(q * measured.length))];
  const pct = (value) => `${(value * 100).toFixed(1)}%`;
  host.innerHTML =
    `<span class="count">flatness across ${measured.length} measured worlds</span> ` +
    `min ${pct(measured[0])} · p25 ${pct(at(0.25))} · <b>median ${pct(at(0.5))}</b> · ` +
    `p75 ${pct(at(0.75))} · max ${pct(measured[measured.length - 1])}`;
}

function table(rows) {
  return `<div class="rows worldRows">${rows.map((world) => {
    const state = world.state || "";
    // Flatness is the measure a keep/deprecate call usually turns on. It is
    // null when the terrain census has no zone for the seed, which is itself
    // a reason to look before keeping it.
    const flat = world.flatness == null
      ? "—" : `${(world.flatness * 100).toFixed(1)}%`;
    return `
      <div class="ro worldRow" data-state="${escape(state)}">
        <b>${world.seed}</b>
        <i class="flatBar" title="open_flat share of the largest zone: ${flat}"
           style="--flat:${world.flatness == null ? 0 : (world.flatness * 100).toFixed(1)}%"
           aria-label="flatness ${flat}"></i>
        <i title="open_flat share of the largest zone">flat ${flat}</i>
        <i class="mono" title="${escape(world.semantic_sha256)}">${escape(world.semantic_sha256.slice(0, 8))}</i>
        ${state ? `<i class="tag ${state === "deprecated" ? "neg" : "pos"}">${escape(state)}</i>` : ""}
        ${world.stale_decision ? `<i class="tag neg" title="Recorded against a different world hash">stale</i>` : ""}
        ${world.reason ? `<i class="reason">${escape(world.reason)}</i>` : ""}
        <span class="spacer"></span>
        <button type="button" data-seed="${world.seed}" data-state="kept">keep</button>
        <button type="button" data-seed="${world.seed}" data-state="deprecated">deprecate</button>
        ${state ? `<button type="button" data-seed="${world.seed}">clear</button>` : ""}
      </div>`;
  }).join("")}</div>`;
}

async function decide(seed, state) {
  const library = inventory?.library?.id;
  if (!library) return;
  const reason = state === "deprecated"
    ? (prompt(`Why deprecate world ${seed}?`, "") ?? null) : "";
  // A cancelled prompt means the decision was not made, so nothing is sent.
  if (reason === null) return;
  const split = inventory.split || "train";
  const response = await fetch(
    `/api/worlds/${encodeURIComponent(library)}/${seed}/status`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state, reason, split }),
    });
  if (!response.ok) {
    alert(`Could not record that: ${response.status}`);
    return;
  }
  inventory = await response.json();
  drawWorlds();
}

function fillPickers() {
  const library = $("worldLibrary");
  const split = $("worldSplit");
  if (library && !library.options.length) {
    fetch("/api/worlds").then((r) => r.json()).then((payload) => {
      library.innerHTML = (payload.libraries || []).map((item) =>
        `<option value="${item.id}"${item.default ? " selected" : ""}>${item.id}</option>`
      ).join("");
      library.addEventListener("change", loadWorlds);
    }).catch(() => { /* the picker simply stays empty */ });
  }
  if (split && !split.options.length) {
    split.innerHTML = (inventory.splits || ["train"]).map((name) =>
      `<option value="${name}"${name === inventory.split ? " selected" : ""}>${name}</option>`
    ).join("");
    split.addEventListener("change", loadWorlds);
  }
  const filter = $("worldFilter");
  if (filter && !filter.dataset.wired) {
    filter.dataset.wired = "1";
    // Redraw only -- the filter narrows the inventory already in hand, so it
    // must not refetch 288 worlds on every keystroke.
    filter.addEventListener("input", drawWorlds);
  }
}

function escape(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}
