/* The Java workspace: the real game server, for manual verification only.
 *
 * This tab is deliberately not a training surface. Everything the console does
 * to *make* an agent happens in JAX; this is where you watch the ported policy
 * inside Hytale and find out whether it moves.
 *
 * Two decisions worth knowing about:
 *
 * The launch command is rendered before it runs. A launcher that hides its
 * arguments cannot be reproduced from a terminal when it misbehaves, and every
 * flag on that line was learned by something failing.
 *
 * Command refusals are predicted, not enforced. `/npc spawn` cannot work from
 * a server console -- see the panel copy for the source trail -- so typing it
 * gets a warning *before* the server replies with a translation key that does
 * not explain itself. The command is still sent: the server is the authority
 * on its own behaviour, and a client-side blocklist would drift from it.
 */

import { $ } from "./state.js";

let timer = null;
let logCursor = 0;
let logSession = null;
let loadingLog = false;

const esc = (s) => String(s).replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* `.ro` / <i> / <b class="pos|neg|absent"> is the house fact row -- same
   markup the Compute tab uses, so both inherit one set of rules. */
function facts(host, rows) {
  host.innerHTML = rows.map(([k, v, c]) =>
    `<div class="ro"><i>${esc(k)}</i><b class="${c || ""}">${esc(v)}</b></div>`
  ).join("");
}

/** The `/npc` subcommand in a command line, or null if it is not an npc one. */
function npcSubcommand(text) {
  const parts = text.trim().replace(/^\//, "").split(/\s+/);
  return parts[0] === "npc" && parts[1] ? parts[1].toLowerCase() : null;
}

export async function loadJava() {
  const host = $("javaInstall");
  if (!host) return;

  let d;
  try {
    d = await (await fetch("/api/hytale")).json();
  } catch {
    return;                        // keep the last good render on a blip
  }

  const rows = [];
  for (const [name, path] of Object.entries(d.install.paths)) {
    rows.push([name, d.install.present[name] ? path : `missing — ${path}`,
      d.install.present[name] ? "pos" : "neg"]);
  }
  facts(host, rows);
  $("javaReady").textContent = d.install.ready ? "ready" : "not launchable";

  $("javaState").textContent = d.running
    ? `running · pid ${d.pid} · ${d.mode}`
    : (d.exit_code === null || d.exit_code === undefined
       ? "stopped" : `exited ${d.exit_code}`);
  $("javaLaunch").disabled = d.running || !d.install.ready;
  $("javaStop").disabled = !d.running;
  $("javaSend").disabled = !d.running;
  $("javaConsoleState").textContent = d.running
    ? `live · pid ${d.pid}` : d.session_id ? "session ended" : "stopped";
  $("javaConsoleState").className =
    `serverConsoleState ${d.running ? "is-live" : ""}`;
  $("javaLogPath").textContent = d.log_path || "No server session has been started.";

  // Boot milestones. Ordered by when they happen so a stalled boot reads as a
  // prefix of ticks rather than a scatter, and absent-vs-false is meaningful:
  // "not yet" is not "not applicable".
  const LABELS = {
    booted: "server booted",
    game_port: "game port (QUIC/UDP)",
    bridge: "bridge on 5556",
    authenticated: "authenticated",
    lan_discovery: "LAN discovery",
  };
  facts($("javaReached"), Object.entries(LABELS).map(([key, label]) => {
    const hit = (d.reached || {})[key];
    // Only meaningful while something is running; a stopped server has not
    // failed to bind, it simply is not there.
    if (!d.running) return [label, "—", "absent"];
    return [label, hit ? "yes" : "not yet", hit ? "pos" : "absent"];
  }));

  if (d.command && d.command.length) {
    $("javaCommand").textContent = d.command.join(" \\\n  ");
  } else {
    await previewCommand();
  }

  // Which subcommands survive a senderless caller, straight from the server's
  // own class hierarchy rather than from a guess.
  const list = $("javaCommands");
  list.innerHTML = "";
  const seen = [
    ...d.console_safe.map((n) => [n, "console can run it", "pos"]),
    ...d.needs_player.map((n) => [n, "needs a connected player", "neg"]),
  ];
  list.innerHTML = seen.map(([name, why, tone]) =>
    `<div class="ro"><i>/npc ${esc(name)}</i>` +
    `<b class="${tone}">${esc(why)}</b></div>`).join("");
  $("javaNeedsPlayer").textContent = `${d.needs_player.length} need a client`;

  await loadLog();
}

async function loadLog() {
  if (loadingLog) return;
  loadingLog = true;
  try {
    const pre = $("javaLog");
    let more = true;
    while (more) {
      const response = await fetch(
        `/api/hytale/log?after=${logCursor}&limit=50000`,
      );
      if (!response.ok) break;
      const d = await response.json();
      // A server process that has not yet loaded the new route still returns
      // `{lines}`. Keep its live tail visible until the ordinary restart that
      // enables cursors and full-session catch-up.
      if (!Number.isFinite(d.cursor)) {
        pre.textContent = d.lines.join("\n") || "Waiting for a server session.";
        $("javaLogCount").textContent = `${d.lines.length.toLocaleString()} lines`;
        $("javaTranscriptState").textContent = d.lines.length
          ? "Live server tail." : "No server session yet.";
        more = false;
        break;
      }
      const sessionChanged = d.session_id !== logSession;
      const hadCursor = logCursor > 0;
      if (sessionChanged || d.reset) {
        logSession = d.session_id;
        logCursor = 0;
        pre.textContent = "";
        if (d.reset || (sessionChanged && hadCursor)) continue;
      }
      if (d.lines.length) {
        pre.append(document.createTextNode(
          `${pre.textContent ? "\n" : ""}${d.lines.join("\n")}`,
        ));
      }
      logCursor = d.cursor;
      more = d.has_more;
      $("javaLogCount").textContent = `${d.total.toLocaleString()} lines`;
      $("javaLogPath").textContent = d.log_path || "No server session has been started.";
      $("javaTranscriptState").textContent = d.truncated
        ? `Showing the newest ${d.total.toLocaleString()} retained lines; the session file has the complete transcript.`
        : d.session_id ? "Complete current-session transcript." : "No server session yet.";
    }
    if (!logSession && !pre.textContent) {
      pre.textContent = "Waiting for a server session.";
    }
    if ($("javaFollow").checked) pre.scrollTop = pre.scrollHeight;
  } catch { /* leave the last good log */ }
  finally { loadingLog = false; }
}

function payload() {
  const boot = $("javaBoot").value.split("\n")
    .map((s) => s.trim()).filter(Boolean);
  return {
    mode: $("javaMode").value,
    port: Number($("javaPort").value) || 25565,
    // Empty means "let the mode decide" -- the server's own default is
    // AUTHENTICATED regardless of mode, so omitting the flag is not neutral.
    auth: $("javaAuth").value,
    owner: $("javaOwner").value.trim(),
    boot_commands: boot,
  };
}

async function previewCommand() {
  try {
    const d = await (await fetch("/api/hytale/preview", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify(payload()),
    })).json();
    $("javaCommand").textContent = d.command.join(" \\\n  ");
  } catch { /* the preview is a convenience, not a gate */ }
}

async function post(url, body) {
  const response = await fetch(url, {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `${response.status}`);
  }
  return response.json();
}

export function wireJava() {
  if (!$("javaLaunch")) return;

  for (const id of ["javaMode", "javaAuth", "javaPort", "javaOwner", "javaBoot"]) {
    $(id).addEventListener("input", previewCommand);
  }

  $("javaLaunch").addEventListener("click", async () => {
    $("javaLaunch").disabled = true;
    try {
      await post("/api/hytale/launch", payload());
    } catch (error) {
      $("javaState").textContent = `launch failed: ${error.message}`;
    }
    await loadJava();
  });

  $("javaStop").addEventListener("click", async () => {
    $("javaStop").disabled = true;
    try { await post("/api/hytale/stop"); } catch { /* reported by status */ }
    await loadJava();
  });

  const send = async () => {
    const command = $("javaCmd").value.trim();
    if (!command) return;
    try {
      await post("/api/hytale/command", { command });
      $("javaCmd").value = "";
    } catch (error) {
      $("javaSpawnHint").textContent = error.message;
    }
    await loadLog();
  };
  $("javaSend").addEventListener("click", send);
  $("javaCmd").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      send();
    }
  });

  $("javaCmd").addEventListener("input", () => {
    const sub = npcSubcommand($("javaCmd").value);
    const hint = $("javaSpawnHint");
    if (sub === "spawn") {
      hint.textContent = "This will be refused. /npc spawn needs a player — "
        + "the console has no body to spawn next to. Connect a client and run "
        + "it in chat.";
      hint.className = "note neg";
    } else {
      hint.textContent = "";
      hint.className = "note";
    }
  });
}

/** Poll only while the tab is visible; a game server log is chatty. */
export function javaTabChanged(name) {
  clearInterval(timer);
  timer = null;
  if (name === "java") {
    loadJava();
    timer = setInterval(loadJava, 1000);
  }
}
