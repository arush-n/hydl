/* Eight workspaces, one job each.
 *
 * Panels are hidden rather than unmounted, so a run stays loaded and the camera
 * keeps its orbit when you leave Run and come back. The cost is that hidden
 * canvases have zero size while hidden -- `fit()` would read 0x0 and draw
 * nothing -- so switching back invalidates every canvas surface.
 *
 * The tab lives in the URL hash, which makes a workspace linkable and survives
 * the hot-reload watcher's full page reloads.
 */

import { $, invalidate } from "./state.js";

const NAMES = [
  "status", "evidence", "run", "custom", "train", "archive", "compute", "java",
];
const ALIASES = { worldgen: "custom", runs: "archive" };
const DEFAULT = "status";

let current = null;
const listeners = [];

/** Run `fn(name)` whenever the visible workspace changes. */
export function onTab(fn) { listeners.push(fn); }

export function activeTab() { return current; }

export function showTab(name) {
  name = ALIASES[name] || name;
  if (!NAMES.includes(name)) name = DEFAULT;
  if (name === current) return;
  current = name;

  for (const panel of document.querySelectorAll("[data-panel]")) {
    panel.hidden = panel.dataset.panel !== name;
  }
  for (const button of $("tabs").querySelectorAll("button")) {
    button.setAttribute("aria-selected", String(button.dataset.tab === name));
  }
  if (location.hash.slice(1) !== name) history.replaceState(null, "", `#${name}`);

  // A canvas inside a hidden panel has no layout, so anything drawn while it
  // was away was drawn at zero size. Redraw everything the panel owns.
  invalidate("data", "frame", "view", "history", "runs", "jobs");
  for (const fn of listeners) {
    try { fn(name); } catch (err) { console.error("[tab]", err); }
  }
}

export function wireTabs() {
  $("tabs").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-tab]");
    if (button) showTab(button.dataset.tab);
  });
  addEventListener("hashchange", () => showTab(location.hash.slice(1)));
  showTab(location.hash.slice(1) || DEFAULT);
}

/* Theme: the toggle must beat the media query in both directions, so it stamps
   an explicit attribute rather than clearing back to "whatever the OS says". */
export function wireTheme(onChange) {
  const KEY = "hytalerl-console-theme";
  const stored = localStorage.getItem(KEY);
  if (stored) document.documentElement.dataset.theme = stored;

  $("theme").addEventListener("click", () => {
    const dark = matchMedia("(prefers-color-scheme: dark)").matches;
    const now = document.documentElement.dataset.theme
      || (dark ? "dark" : "light");
    const next = now === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem(KEY, next);
    onChange?.();
  });
}
