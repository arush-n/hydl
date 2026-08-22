/* Agent frame rendering.
 *
 * Python owns discovery and data. This module owns one stable visual contract,
 * so an agent can add a detailed console frame without adding HTML, a route,
 * CSS, or JavaScript. The four original views remain compatible; `detail`
 * composes them with progress and code sections.
 */

import { $, state } from "./state.js";
import { plot } from "./charts.js";
import { css } from "./gfx.js";
import { replayRun } from "./runs.js";
import { activeTab } from "./tabs.js";

const timers = new Map();
const loaded = new Set();
const payloads = new Map();
const requests = new Map();
const filters = new Map();
const SAFE_TONES = new Set(["info", "good", "warn", "crit", "pos", "neg", "absent"]);
const SERIES_COLORS = ["--accent", "--ok", "--info", "--warn", "--crit", "--agent", "--target"];
let registrySignature = "";
let workspace = "status";
let mountGeneration = 0;

export async function loadPanels() {
  let registry;
  try {
    const response = await fetch("/api/panels");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    registry = await response.json();
  } catch {
    return;                       // the console stays usable without frames
  }
  state.panels = registry.panels || [];
  state.panelErrors = registry.errors || [];
  state.panelTabs = registry.tabs || [];
  const signature = JSON.stringify(registry);
  if (signature === registrySignature) return;
  registrySignature = signature;
  mount();
}

/** Load and poll only frames in the workspace the user can currently see. */
export function panelsTabChanged(tab) {
  workspace = tab;
  activate(tab);
}

/* Frames live in a host per workspace, after the built-in content. */
function host(tab) {
  const workspace = document.querySelector(`[data-panel="${cssEscape(tab)}"]`);
  if (!workspace) return null;
  let box = workspace.querySelector(".panelHost");
  if (!box) {
    box = document.createElement("div");
    box.className = "panelHost";
    box.setAttribute("aria-label", "Additional workspace panels");
    workspace.appendChild(box);
  }
  return box;
}

function mountTools(box, tab, count) {
  const placeholder = tab === "evidence"
    ? "Artifacts, receipts, diagnostics..." : "Panels and rows...";
  box.insertAdjacentHTML("beforeend", `
    <div class="panelTools" data-panel-tools="${esc(tab)}">
      <label class="panelFilter">
        <span>Search</span>
        <input type="search" data-panel-filter="${esc(tab)}"
          value="${esc(filters.get(tab) || "")}" autocomplete="off"
          placeholder="${esc(placeholder)}"
          aria-label="Search ${esc(tab)} workspace panels and rows">
      </label>
      <span class="panelToolCount" data-panel-tool-count>${esc(count)} panels</span>
      <div class="panelToolActions" data-section-actions hidden>
        <button type="button" class="ghost mini" data-sections="collapse">Collapse all</button>
        <button type="button" class="ghost mini" data-sections="expand">Expand all</button>
      </div>
    </div>`);
  if (box.dataset.toolsWired) return;
  box.dataset.toolsWired = "true";
  box.addEventListener("input", (event) => {
    const input = event.target.closest("[data-panel-filter]");
    if (!input) return;
    filters.set(input.dataset.panelFilter, input.value);
    applyWorkspaceFilter(input.dataset.panelFilter);
  });
  box.addEventListener("click", (event) => {
    const action = event.target.closest("[data-sections]")?.dataset.sections;
    if (action) {
      for (const section of box.querySelectorAll("details.panelSection")) {
        section.open = action === "expand";
        delete section.dataset.filterOpen;
      }
      refreshTool(box);
      return;
    }
    if (event.target.closest("summary.panelSectionHead")) {
      setTimeout(() => refreshTool(box), 0);
    }
  });
}

function applyWorkspaceFilter(tab) {
  const box = document.querySelector(
    `[data-panel="${cssEscape(tab)}"] .panelHost`,
  );
  if (!box) return;
  const query = String(filters.get(tab) || "").trim().toLowerCase();

  for (const card of box.querySelectorAll(".panelCard")) {
    const metaMatch = !query || (card.dataset.searchMeta || "").includes(query);
    const sections = [...card.querySelectorAll(".panelSections > .panelSection")];
    const rows = sections.length
      ? [] : [...card.querySelectorAll(".panelRows > .panelRow")];
    const items = sections.length ? sections : rows;

    if (!query || metaMatch) {
      for (const item of items) {
        item.hidden = false;
        restoreFilterOpen(item);
      }
      card.hidden = false;
      continue;
    }
    if (items.length) {
      let matches = 0;
      for (const item of items) {
        const match = item.textContent.toLowerCase().includes(query);
        item.hidden = !match;
        if (item.matches("details.panelSection")) {
          if (match) {
            if (item.dataset.filterOpen === undefined) {
              item.dataset.filterOpen = String(item.open);
            }
            item.open = true;
          } else {
            restoreFilterOpen(item);
          }
        }
        matches += Number(match);
      }
      card.hidden = matches === 0;
    } else {
      card.hidden = !card.textContent.toLowerCase().includes(query);
    }
  }
  refreshTool(box);
}

function restoreFilterOpen(item) {
  if (!item.matches("details.panelSection") || item.dataset.filterOpen === undefined) return;
  item.open = item.dataset.filterOpen === "true";
  delete item.dataset.filterOpen;
}

function refreshTool(box) {
  const tool = box.querySelector(".panelTools");
  if (!tool) return;
  const cards = [...box.querySelectorAll(".panelCard")];
  const visibleCards = cards.filter((card) => !card.hidden);
  let visibleItems = 0;
  for (const card of visibleCards) {
    const sections = [...card.querySelectorAll(".panelSections > .panelSection")];
    const items = sections.length
      ? sections : [...card.querySelectorAll(".panelRows > .panelRow")];
    visibleItems += items.filter((item) => !item.hidden).length;
  }
  const sections = [...box.querySelectorAll("details.panelSection")];
  const openSections = sections.filter((section) => section.open).length;
  const openCards = visibleCards.filter((card) => card.open).length;
  const query = String(filters.get(tool.dataset.panelTools) || "").trim();
  const summary = query
    ? `${visibleCards.length}/${cards.length} panels · ${visibleItems} matches`
    : `${openCards}/${cards.length} panels open${sections.length ? ` · ${openSections}/${sections.length} details open` : ""}`;
  tool.querySelector("[data-panel-tool-count]").textContent = summary;
  tool.querySelector("[data-section-actions]").hidden = sections.length === 0;
}

function mount() {
  mountGeneration += 1;
  for (const timer of timers.values()) clearInterval(timer);
  timers.clear();
  loaded.clear();
  payloads.clear();
  requests.clear();
  document.querySelectorAll(".panelHost").forEach((box) => { box.innerHTML = ""; });

  const tabCounts = new Map();
  const firstByTab = new Set();
  for (const spec of state.panels || []) {
    tabCounts.set(spec.tab, (tabCounts.get(spec.tab) || 0) + 1);
  }

  for (const spec of state.panels || []) {
    const box = host(spec.tab);
    if (!box) continue;
    if (tabCounts.get(spec.tab) > 1 && !box.querySelector(".panelTools")) {
      mountTools(box, spec.tab, tabCounts.get(spec.tab));
    }
    const owner = spec.owner || {};
    const ownerTitle = owner.title || spec.agent || "unknown";
    const architecture = owner.architecture || "unspecified";
    const architectureLabel = architecture.toLowerCase() === ownerTitle.toLowerCase()
      ? "" : `<span class="panelArch">${esc(architecture)}</span>`;
    const card = document.createElement("details");
    card.className = "card panelCard";
    card.open = !firstByTab.has(spec.tab);
    firstByTab.add(spec.tab);
    card.dataset.size = panelSize(spec.size);
    card.dataset.tab = spec.tab;
    card.dataset.searchMeta = [
      spec.title, spec.agent, owner.title, owner.architecture,
      spec.description, ...(spec.tags || []),
    ].filter(Boolean).join(" ").toLowerCase();
    card.id = `panel-${spec.id}`;
    card.innerHTML = `
      <summary class="panelFrameHead panelFrameToggle">
        <div class="panelFrameIdentity">
          <div class="panelOwnerLine">
            <span class="byline">${esc(ownerTitle)}</span>
            ${architectureLabel}
          </div>
          <h2>${esc(spec.title)}</h2>
        </div>
        <div class="panelFrameState">
          <span class="count" data-count></span>
          <span class="panelStatus" data-status hidden></span>
          <i class="panelExpand" aria-hidden="true"></i>
        </div>
      </summary>
      <div class="panelFrameContent">
      ${spec.description ? `<p class="note panelDescription">${esc(spec.description)}</p>` : ""}
      <p class="panelSummary" data-summary hidden></p>
      <div class="panelTags" data-tags hidden></div>
      <div data-body class="empty panelBody" aria-live="polite" aria-busy="false">Open this workspace to load.</div>
      <div class="panelLinks" data-links hidden></div>
      <details class="panelProvenance">
        <summary>Panel details</summary>
        <dl>
          <div><dt>owner</dt><dd>${esc(spec.agent || "unknown")}</dd></div>
          <div><dt>architecture</dt><dd>${esc(owner.architecture || "unspecified")}</dd></div>
          <div><dt>source</dt><dd>${esc(spec.source || "runtime")}</dd></div>
          <div><dt>refresh</dt><dd>${spec.refresh > 0 ? `${esc(spec.refresh)} s` : "on load"}</dd></div>
          <div><dt>registered</dt><dd>${esc(formatTime(spec.added_at))}</dd></div>
          <div><dt>updated</dt><dd data-updated>not reported</dd></div>
        </dl>
      </details>
      </div>`;
    card.addEventListener("toggle", () => {
      if (card.open && workspace === spec.tab) activateSpec(spec);
      if (!card.open && timers.has(spec.id)) {
        clearInterval(timers.get(spec.id));
        timers.delete(spec.id);
      }
      refreshTool(box);
    });
    box.appendChild(card);
  }

  /* A frame file that failed to import is surfaced. Silently missing is the
     one outcome an agent iterating on a frame cannot debug. */
  const box = host("status");
  for (const error of state.panelErrors || []) {
    const card = document.createElement("section");
    card.className = "card panelCard panelCardFailed";
    card.dataset.size = "wide";
    card.innerHTML =
      `<header class="panelFrameHead"><div class="panelFrameIdentity">
         <span class="byline">panel loader</span><h2>${esc(error.module)}</h2></div>
         <span class="panelStatus sev-crit">failed to import</span></header>
       <pre class="trace">${esc(error.traceback)}</pre>`;
    box?.appendChild(card);
  }

  workspace = activeTab() || workspace;
  activate(workspace);
}

function activate(tab) {
  for (const timer of timers.values()) clearInterval(timer);
  timers.clear();

  for (const spec of state.panels || []) {
    if (spec.tab !== tab) continue;
    activateSpec(spec);
  }
  applyWorkspaceFilter(tab);
}

function activateSpec(spec) {
  const card = $(`panel-${spec.id}`);
  if (!card?.open) return;
  const cached = payloads.get(spec.id);
  if (cached) requestAnimationFrame(() => paintAllSeries(cached));
  if (!loaded.has(spec.id) || spec.refresh > 0) draw(spec);
  if (spec.refresh > 0 && !timers.has(spec.id)) {
    timers.set(spec.id, setInterval(() => {
      if (workspace === spec.tab && card.open) draw(spec);
    }, spec.refresh * 1000));
  }
}

async function draw(spec) {
  if (requests.has(spec.id)) return requests.get(spec.id);
  const generation = mountGeneration;
  const request = drawPanel(spec, generation);
  requests.set(spec.id, request);
  try {
    return await request;
  } finally {
    if (requests.get(spec.id) === request) requests.delete(spec.id);
  }
}

async function drawPanel(spec, generation) {
  const card = $(`panel-${spec.id}`);
  if (!card) return;
  const body = card.querySelector("[data-body]");
  body.setAttribute("aria-busy", "true");
  if (!loaded.has(spec.id)) body.textContent = "loading...";
  card.classList.add("panelCardRefreshing");

  let payload;
  try {
    const response = await fetch(`/api/panels/${encodeURIComponent(spec.id)}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    payload = await response.json();
  } catch (error) {
    if (generation !== mountGeneration) return;
    body.className = "panelBody";
    body.innerHTML = `<p class="note warn">could not load: ${esc(error.message)}</p>`;
    body.setAttribute("aria-busy", "false");
    card.classList.remove("panelCardRefreshing");
    loaded.add(spec.id);
    applyWorkspaceFilter(spec.tab);
    return;
  }

  if (generation !== mountGeneration) return;

  updateFrameChrome(card, spec, payload);
  body.className = "panelBody";
  if (payload.failed) {
    card.classList.add("panelCardFailed");
    body.innerHTML = `<pre class="trace">${esc(payload.text || "frame failed")}</pre>`;
    body.setAttribute("aria-busy", "false");
    card.classList.remove("panelCardRefreshing");
    loaded.add(spec.id);
    payloads.set(spec.id, payload);
    applyWorkspaceFilter(spec.tab);
    return;
  }
  card.classList.remove("panelCardFailed");

  const openSectionKeys = new Set(
    [...body.querySelectorAll("details.panelSection[open]")]
      .map((section) => section.dataset.sectionKey).filter(Boolean),
  );
  body.innerHTML = renderView(payload, payload.id);
  for (const section of body.querySelectorAll("details.panelSection")) {
    if (openSectionKeys.has(section.dataset.sectionKey)) section.open = true;
  }
  if (payload.note) {
    body.insertAdjacentHTML(
      "beforeend",
      `<p class="note panelCaveat ${tone(payload.note_tone)}">${esc(payload.note)}</p>`,
    );
  }
  body.setAttribute("aria-busy", "false");
  card.classList.remove("panelCardRefreshing");
  loaded.add(spec.id);
  payloads.set(spec.id, payload);

  // Canvases need a laid-out box before their backing stores are fitted.
  if (workspace === spec.tab) requestAnimationFrame(() => paintAllSeries(payload));
  bindFrameActions(body);
  applyWorkspaceFilter(spec.tab);
}

function updateFrameChrome(card, spec, payload) {
  const count = card.querySelector("[data-count]");
  count.textContent = payload.count === undefined || payload.count === null
    ? "" : String(payload.count);

  const status = normalizeStatus(payload.status);
  const statusNode = card.querySelector("[data-status]");
  statusNode.hidden = !status.label;
  statusNode.textContent = status.label;
  statusNode.className = `panelStatus ${tone(status.tone)}`;
  statusNode.title = status.detail || "";

  const summary = card.querySelector("[data-summary]");
  summary.hidden = !payload.summary;
  summary.textContent = payload.summary || "";

  const tags = [...(spec.tags || []), ...(Array.isArray(payload.tags) ? payload.tags : [])];
  const uniqueTags = dedupeTags(tags);
  const tagBox = card.querySelector("[data-tags]");
  tagBox.hidden = uniqueTags.length === 0;
  tagBox.innerHTML = uniqueTags.map((tag) =>
    `<span class="panelTag ${tone(tag.tone)}">${esc(tag.label)}</span>`).join("");

  const links = Array.isArray(payload.links) ? payload.links : [];
  const linkBox = card.querySelector("[data-links]");
  const safeLinks = links.map(normalizeLink).filter(Boolean);
  linkBox.hidden = safeLinks.length === 0;
  linkBox.innerHTML = safeLinks.map((link) =>
    `<a href="${esc(link.href)}"${link.external ? ' target="_blank" rel="noopener noreferrer"' : ""}` +
    ` class="panelLink">${esc(link.label)}</a>`).join("");

  card.querySelector("[data-updated]").textContent = formatTime(payload.updated_at);
}

function renderView(payload, key) {
  if (payload.view === "facts") return facts(payload);
  if (payload.view === "rows") return rows(payload);
  if (payload.view === "series") return seriesBoxes(payload, key);
  if (payload.view === "progress") return progress(payload);
  if (payload.view === "code") return codeBlock(payload);
  if (payload.view === "detail") return detail(payload, key);
  return note(payload);
}

function facts(payload) {
  return `<div class="facts panelFacts">` + (payload.facts || []).map((fact) => {
    const href = safeHref(fact.href);
    const value = `<b class="${tone(fact.tone)}">${esc(fact.value)}` +
      `${fact.unit ? `<span class="factUnit">${esc(fact.unit)}</span>` : ""}</b>`;
    return `<div class="ro panelFact">` +
      `<i>${esc(fact.label)}</i>${href ? `<a href="${esc(href)}">${value}</a>` : value}` +
      `${fact.detail ? `<small>${esc(fact.detail)}</small>` : ""}` +
      `${Number.isFinite(Number(fact.progress)) ? miniProgress(fact.progress, fact.tone) : ""}` +
      `</div>`;
  }).join("") + `</div>`;
}

function rows(payload) {
  const head = (payload.columns || []).length
    ? `<div class="rowhead">${payload.columns.map((column) =>
        `<span>${esc(typeof column === "object" ? column.label : column)}</span>`).join("")}</div>`
    : "";
  const body = (payload.rows || []).map((row) => {
    const cells = (row.cells || []).map((cell, index) => renderCell(cell, index)).join("");
    const replay = row.run_id ?
      ` data-run-id="${esc(row.run_id)}" title="replay ${esc(row.run_id)}" role="button" tabindex="0"` : "";
    const badge = row.badge ? `<span class="rowBadge ${tone(row.tone)}">${esc(row.badge)}</span>` : "";
    const detailText = row.detail ? `<small class="rowDetail">${esc(row.detail)}</small>` : "";
    return `<div class="panelRow ${tone(row.tone) || "sev-info"}"${replay}>` +
      `<div class="panelRowMain">${cells}${badge}</div>${detailText}</div>`;
  }).join("");
  const scroll = payload.scroll === true ? " panelRowsScrollable" : "";
  return head + `<div class="rows panelRows${scroll}">${body || '<p class="empty">No rows.</p>'}</div>`;
}

function renderCell(cell, index) {
  const value = cell && typeof cell === "object" && !Array.isArray(cell) ? cell : { value: cell };
  const classes = ["panelCell"];
  const grows = value.grow === true || (value.grow !== false && index === 0);
  classes.push(grows ? "grow" : "num");
  if (value.mono === true) classes.push("num");
  if (value.mono === false) classes.push("plain");
  if (value.wrap === true) classes.push("panelCellWrap");
  const href = safeHref(value.href);
  const chips = Array.isArray(value.chips) ? `<span class="panelCellChips">` +
    value.chips.map((chip) => {
      const normalized = chip && typeof chip === "object" ? chip : { label: chip };
      return `<span class="panelTag ${tone(normalized.tone)}">${esc(normalized.label)}</span>`;
    }).join("") + `</span>` : "";
  const content = `<span class="cellValue ${tone(value.tone)}">${esc(value.value)}</span>` + chips +
    `${value.detail ? `<small>${esc(value.detail)}</small>` : ""}`;
  return `<span class="${classes.join(" ")}">${href ? `<a href="${esc(href)}">${content}</a>` : content}</span>`;
}

function seriesBoxes(payload, key) {
  return `<div class="charts panelCharts">` + (payload.series || []).map((series, index) => {
    const values = finitePoints(series.points);
    const last = values.length ? formatNumber(values.at(-1)) : "no data";
    const range = values.length ? `${formatNumber(Math.min(...values))} - ${formatNumber(Math.max(...values))}` : "";
    return `<figure><figcaption><span>${esc(series.name)}</span>` +
      `<small>${esc(series.summary || `${last}${range ? ` | ${range}` : ""}`)}</small></figcaption>` +
      `<canvas class="chart" id="${chartId(key, index)}" aria-label="${esc(series.name)} chart"></canvas></figure>`;
  }).join("") + `</div>`;
}

function progress(payload) {
  return `<div class="progressList">` + (payload.items || []).map((item) => {
    const maximum = positiveNumber(item.max, 100);
    const value = Math.max(0, Math.min(maximum, Number(item.value) || 0));
    const percent = maximum ? value / maximum * 100 : 0;
    const display = item.display ?? `${Math.round(percent)}%`;
    return `<div class="progressItem ${tone(item.tone)}">` +
      `<div class="progressHead"><span>${esc(item.label)}</span><b>${esc(display)}</b></div>` +
      `<div class="progressTrack" role="progressbar" aria-label="${esc(item.label)}" ` +
      `aria-valuemin="0" aria-valuemax="${esc(maximum)}" aria-valuenow="${esc(value)}">` +
      `<span style="width:${percent.toFixed(2)}%"></span></div>` +
      `${item.detail ? `<small>${esc(item.detail)}</small>` : ""}</div>`;
  }).join("") + `</div>`;
}

function miniProgress(value, requestedTone) {
  const numeric = Number(value);
  const percent = Math.max(0, Math.min(1, numeric > 1 ? numeric / 100 : numeric)) * 100;
  return `<span class="factProgress ${tone(requestedTone)}" aria-hidden="true"><i style="width:${percent.toFixed(2)}%"></i></span>`;
}

function codeBlock(payload) {
  const language = payload.language ? `<span>${esc(payload.language)}</span>` : "";
  return `<div class="panelCode"><div class="panelCodeHead">${language}<button type="button" ` +
    `class="ghost mini" data-copy-code>Copy</button></div>` +
    `<pre><code>${esc(payload.text || "")}</code></pre></div>`;
}

function note(payload) {
  const paragraphs = String(payload.text || "").split(/\n{2,}/).filter(Boolean);
  return `<div class="panelNote ${tone(payload.tone)}">` +
    (paragraphs.length ? paragraphs : [""]).map((text) => `<p>${esc(text)}</p>`).join("") + `</div>`;
}

function detail(payload, key) {
  const sections = payload.sections || [];
  if (!sections.length) return '<p class="empty">No detail sections.</p>';
  return `<div class="panelSections">` + sections.map((section, index) => {
    const sectionKey = `${key}-${index}`;
    const sectionSize = section.size === "full" ? "full" : "half";
    const title = section.title ? `<h3>${esc(section.title)}</h3>` : "";
    const description = section.description ? `<p>${esc(section.description)}</p>` : "";
    const badge = section.badge
      ? `<span class="panelTag ${tone(section.tone)}">${esc(section.badge)}</span>` : "";
    if (section.collapsible === true) {
      return `<details class="panelSection ${tone(section.tone)}" data-size="${sectionSize}" ` +
        `data-section-key="${esc(sectionKey)}"${section.open ? " open" : ""}>` +
        `<summary class="panelSectionHead"><div>${title || "<h3>Details</h3>"}${description}</div>` +
        `<span class="panelSectionMeta">${badge}<i aria-hidden="true"></i></span></summary>` +
        `<div class="panelSectionBody">${renderView(section, sectionKey)}</div></details>`;
    }
    const heading = title || description || badge
      ? `<header class="panelSectionHead"><div>${title}${description}</div>${badge}</header>` : "";
    return `<section class="panelSection ${tone(section.tone)}" data-size="${sectionSize}">${heading}` +
      `${renderView(section, sectionKey)}</section>`;
  }).join("") + `</div>`;
}

function paintAllSeries(payload, key = payload.id) {
  if (payload.view === "series") paintSeries(payload, key);
  if (payload.view === "detail") {
    (payload.sections || []).forEach((section, index) =>
      paintAllSeries(section, `${key}-${index}`));
  }
}

function paintSeries(payload, key) {
  (payload.series || []).forEach((series, index) => {
    const values = finitePoints(series.points);
    const token = seriesColor(series, index);
    plot(chartId(key, index), [{ values, color: css(token), fill: Boolean(series.fill) }], {
      min: Number.isFinite(series.min) ? series.min : undefined,
      max: Number.isFinite(series.max) ? series.max : undefined,
      empty: series.empty || "no data yet",
    });
  });
}

function bindFrameActions(body) {
  body.querySelectorAll("[data-run-id]").forEach((element) => {
    const replay = () => replayRun(element.dataset.runId);
    element.addEventListener("click", replay);
    element.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        replay();
      }
    });
  });
  body.querySelectorAll("[data-copy-code]").forEach((button) => {
    button.addEventListener("click", async () => {
      const text = button.closest(".panelCode")?.querySelector("code")?.textContent || "";
      try {
        await navigator.clipboard.writeText(text);
        button.textContent = "Copied";
      } catch {
        button.textContent = "Copy failed";
      }
      setTimeout(() => { button.textContent = "Copy"; }, 1400);
    });
  });
}

function normalizeStatus(value) {
  if (!value) return { label: "", tone: "", detail: "" };
  if (typeof value === "string") return { label: value, tone: "info", detail: "" };
  return {
    label: value.label || value.value || "",
    tone: value.tone || "info",
    detail: value.detail || "",
  };
}

function dedupeTags(values) {
  const seen = new Set();
  const result = [];
  for (const value of values) {
    const tag = typeof value === "object" && value !== null
      ? { label: value.label || value.value || "", tone: value.tone || "" }
      : { label: String(value ?? ""), tone: "" };
    const key = tag.label.trim().toLowerCase();
    if (!key || seen.has(key)) continue;
    seen.add(key);
    result.push(tag);
  }
  return result;
}

function normalizeLink(value) {
  if (!value || typeof value !== "object") return null;
  const href = safeHref(value.href);
  if (!href || !value.label) return null;
  let external = false;
  try { external = new URL(href, location.href).origin !== location.origin; } catch { /* local hash */ }
  return { href, label: value.label, external };
}

function safeHref(value) {
  if (!value) return "";
  try {
    const parsed = new URL(String(value), location.href);
    if (!["http:", "https:"].includes(parsed.protocol)) return "";
    return String(value);
  } catch {
    return "";
  }
}

function seriesColor(series, index) {
  const named = {
    accent: "--accent", good: "--ok", pos: "--ok", warn: "--warn",
    crit: "--crit", neg: "--crit", info: "--info", agent: "--agent", target: "--target",
  };
  return named[series.color] || named[series.tone] || SERIES_COLORS[index % SERIES_COLORS.length];
}

function finitePoints(points) {
  return (points || []).map(Number).filter(Number.isFinite);
}

function positiveNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : fallback;
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return "-";
  return Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < .01)
    ? value.toExponential(2) : Number(value.toFixed(3)).toLocaleString();
}

function formatTime(value) {
  if (!value) return "not reported";
  const raw = typeof value === "number" && value < 1e12 ? value * 1000 : value;
  const date = new Date(raw);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium", timeStyle: "short",
  }).format(date);
}

function chartId(key, index) {
  return `ps-${String(key).replace(/[^a-zA-Z0-9_-]/g, "-")}-${index}`;
}

function panelSize(value) {
  return ["compact", "standard", "wide"].includes(value) ? value : "standard";
}

function tone(value) {
  const candidate = String(value || "").toLowerCase();
  if (!SAFE_TONES.has(candidate)) return "";
  if (candidate === "pos") return "sev-good";
  if (candidate === "neg") return "sev-crit";
  if (candidate === "absent") return "sev-absent";
  return `sev-${candidate}`;
}

function cssEscape(value) {
  if (globalThis.CSS?.escape) return CSS.escape(String(value));
  return String(value).replace(/[^a-zA-Z0-9_-]/g, "");
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character]));
}
