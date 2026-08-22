/* Hot reload.
 *
 * Polls a cheap server stamp (mtimes of the static tree). CSS is swapped in
 * place so the page keeps its loaded run and camera; JS changes need a real
 * reload, and say so rather than silently doing nothing.
 */

let stamp = null;
let timer = null;

export function watch(interval = 5000) {
  const tick = async () => {
    try {
      const next = await (await fetch("/api/version", { cache: "no-store" })).json();
      if (stamp === null) { stamp = next; return; }
      if (next.css !== stamp.css) { swapStyles(); stamp = { ...stamp, css: next.css }; }
      if (next.js !== stamp.js || next.html !== stamp.html) {
        toast("code changed — reloading");
        setTimeout(() => location.reload(), 250);
      }
    } catch { /* server restarting; try again next tick */ }
    finally {
      timer = setTimeout(tick, document.hidden ? 30000 : interval);
    }
  };
  clearTimeout(timer);
  tick();
}

function swapStyles() {
  for (const link of document.querySelectorAll('link[rel="stylesheet"]')) {
    const url = new URL(link.href, location.href);
    url.searchParams.set("v", Date.now());
    link.href = url.pathname + url.search;
  }
  toast("styles reloaded");
}

let node = null;
function toast(text) {
  clearTimeout(toast.timer);
  if (!node) { node = document.createElement("div"); node.className = "toast"; document.body.append(node); }
  node.textContent = text;
  node.hidden = false;
  toast.timer = setTimeout(() => { node.hidden = true; }, 1600);
}
