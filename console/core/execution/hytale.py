"""Launching the real Hytale server, and talking to its console.

This is the *manual verification* half of the project and nothing else. Training
and evaluation run in JAX on this machine; the game server exists to answer the
one question JAX cannot -- does the ported policy actually do anything sensible
inside Hytale. A training run never needs this module.

Three things about the server are non-obvious, all confirmed by reading the
decompiled source rather than by experiment, and all of them cost time when
guessed at:

**Assets must be passed explicitly.** `Options.ASSET_DIRECTORY` defaults to
``Paths.get("../HytaleAssets")`` (`Options.java:58-61`), but a real install
ships ``Assets.zip`` beside ``Server/``. So a bare `java -jar HytaleServer.jar`
finds no pack and exits 7 with
``client.disconnection.shutdownReason.missingAssets.failedToLoad``. That is a
default mismatch, not a broken install.

**Extra mods go in a separate directory.** `--mods` is documented as
"Additional mods directories" (`Options.java:62-65`), comma separated. Anything
we want to load for a test -- a policy agent, a fixture -- goes in a directory
of ours that is *added* to the install's own. Nothing here ever writes into
``<install>/Server/mods``.

**No console command can spawn an NPC.** This is the important one and it is
structural, not a permissions problem:

* the interactive console runs commands as `ConsoleSender.INSTANCE`
  (`ConsoleModule.java:113`), and so does `--boot-command`
  (`HytaleServer.java:447`) -- same sender, both paths;
* `ConsoleSender.hasPermission()` returns `true` unconditionally
  (`ConsoleSender.java:39-46`), so permission is *not* what blocks anything;
* but `NPCSpawnCommand extends AbstractPlayerCommand`, whose `executeAsync`
  fetches `context.senderAsPlayerRef()` and, when it is null, replies
  ``server.commands.errors.playerOrArg`` and returns **without executing**
  (`AbstractPlayerCommand.java:33-38`).

The console has no body, and `/npc spawn` places the NPC at the *sender's*
position. So "spawn an NPC from the server console" is not a thing that can be
configured into existence. A connected game client is required, or the spawn
has to go through the bridge mod, which creates entities programmatically and
never touches the command system. :data:`CONSOLE_SAFE` records which `/npc`
subcommands survive a senderless caller.

**The server's own error message sends you down a dead end here.** Measured
against a live server on 2026-08-08::

    > npc spawn Test_Kweebec_Playing
    Sender must be a player or provide the --player option!

    > npc spawn Test_Kweebec_Playing --player rush
    Could not find an optional argument with the name or unique abbreviation
    of 'player'.

Both base classes share one translation string, but only
`AbstractTargetPlayerCommand` actually declares the argument
(`withOptionalArg("player", ...)`, line 22). `AbstractPlayerCommand` -- what
`/npc spawn` extends -- declares only the message constant and no such option.
So the advice in the refusal is wrong for this command. Do not spend time on
it; the conclusion above is unchanged.

All of the above was confirmed by running it, not only by reading source:
`help` and `npc clean` both executed from a piped stdin, which also proves the
command channel this module opens actually reaches the server's JLine reader.
"""

from __future__ import annotations

from console.core import storage

import os
import subprocess
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

_ROOT = Path(__file__).resolve().parents[3]

#: The game install. `APPDATA` is where the launcher puts it on Windows.
INSTALL = Path(
    os.environ.get("HYTALE_INSTALL")
    or Path(os.environ.get("APPDATA", "")) /
    "Hytale/install/release/package/game/latest"
)

#: The JDK vendored for this project. The system `java` is not assumed to
#: exist, let alone to be 25.
JAVA = _ROOT / "HytaleRL/.tools/jdk25/jdk-25.0.3+9/bin/java.exe"

#: Scratch state for servers this console starts: universe saves and the mods
#: directory that gets *added* to the install's own. Kept out of the install so
#: a test server cannot corrupt the real one.
#: Through `storage` so the native runtime tree follows the `hytale` area
#: instead of pinning its own root. Identical today.
RUNTIME = storage.write_root("hytale")

#: Lines of server log kept in memory. The full log is on disk regardless; this
#: is only what the page can show.
LOG_LINES = 50_000

#: `/npc` subcommands that a senderless console can actually execute, by base
#: class in the decompiled source. `AbstractWorldCommand` acts on a world;
#: `AbstractPlayerCommand` needs a body and is refused. Everything absent from
#: this set needs a connected client.
CONSOLE_SAFE = frozenset({"clean", "freeze", "thaw", "step"})

#: `/npc` subcommands confirmed to require a player, with the reason the server
#: gives. Presented in the UI so a refusal is predicted rather than debugged.
NEEDS_PLAYER = frozenset({"spawn", "all", "message", "runtests", "sensorstats",
                          "descriptors"})


#: Boot milestones, keyed by what they prove, matched as substrings of the
#: server's own log. Each is the **server's*_ statement about itself, which is
#: the only reliable source for some of them -- see :meth:`_Server._latch`.
MILESTONES = {
    "booted": "Hytale Server Booted!",
    # QUIC/UDP. Invisible to a TCP `netstat` filter; this is the proof.
    "game_port": "Listening on /0.0.0.0:",
    "bridge": "Bridge server listening on",
    # Distinguishes a server a client can join from one that will refuse it.
    "authenticated": "Authentication successful!",
    "lan_discovery": "Enabled plugin Hytale:LANDiscovery",
}


class ServerError(RuntimeError):
    """The server could not be started, or is not running when asked."""


def install() -> dict[str, Any]:
    """Where everything is and whether it is actually there.

    Reported as facts rather than raised as errors: a missing JDK and a missing
    asset pack need different fixes, and a caller that sees only "cannot
    launch" cannot tell them apart.
    """

    server = INSTALL / "Server"
    parts = {
        "install": INSTALL,
        "jar": server / "HytaleServer.jar",
        "assets": INSTALL / "Assets.zip",
        "java": JAVA,
        "bridge": server / "mods" / "HytaleRLBridge-0.1.0.jar",
    }
    return {
        "paths": {name: str(path) for name, path in parts.items()},
        "present": {name: path.exists() for name, path in parts.items()},
        "ready": all(parts[n].exists() for n in ("jar", "assets", "java")),
    }


#: Auth mode per launch mode, and the reason for each. `Options.AuthMode` is
#: `authenticated|offline|insecure` (`Options.java:237-254`).
#:
#: **client -> authenticated.** This is the only mode a *release* client will
#: complete a login against. Measured 2026-08-08: with `insecure` the server
#: side works perfectly -- "Starting development flow for arush", "Connection
#: complete ... transitioning to setup", through `setup:world-settings` and
#: into `setup:assets-request` -- and then the client hangs up with "server
#: requires development mode which is not supported in build". The dev flow is
#: for internal builds. `authenticated` needs the server to hold its own
#: tokens: run `auth login device` and complete the code prompt.
#:
#: **headless -> offline.** No client connects, so the join gate never runs,
#: and `offline` is what every bridge measurement was taken against.
DEFAULT_AUTH = {"client": "authenticated", "headless": "offline"}


def argv(mode: str = "client", *, port: int = 25565,
         auth: str | None = None,
         boot_commands: tuple[str, ...] = (),
         mods: tuple[str, ...] = (), universe: str | None = None,
         allow_op: bool = True, owner: str | None = None) -> list[str]:
    """The exact command line, built without launching anything.

    Separated from :func:`launch` so the console can *show* what it would run.
    A launcher that hides its arguments is one the user cannot reproduce from a
    terminal when it goes wrong, and every hard-won flag here was learned by
    something failing.

    `mode` is `"client"` for a server a real game client connects to, or
    `"headless"` for one only the bridge talks to. The difference is not a
    server flag -- there is no such switch -- it is whether we bind a game port
    and permit an unauthenticated player to op themselves.
    """

    server = INSTALL / "Server"
    line = [
        str(JAVA),
        # Silences a noisy AOT-cache error on this install. Cosmetic.
        "-Xshare:off",
        "-jar", str(server / "HytaleServer.jar"),
        # Required. See the module docstring: the default is ../HytaleAssets.
        "--assets", str(INSTALL / "Assets.zip"),
        "--disable-sentry",
        "--auth-mode", auth or DEFAULT_AUTH.get(mode, "offline"),
    ]
    if mode == "client":
        line += ["--bind", f"0.0.0.0:{port}"]
        # LAN discovery is off by default and does **not** survive a restart --
        # it is per-process state toggled by a command, so a server relaunched
        # after being configured silently stops advertising itself and vanishes
        # from the client's "Local" list. Since that list is how a client is
        # actually meant to find a local server (typing into Direct Connect is
        # fiddly and rejects several plausible address formats), turn it on at
        # boot for every client-facing launch. `LANDiscoveryCommand` extends
        # `CommandBase`, so the senderless boot-command path can run it.
        boot_commands = ("landiscovery", *boot_commands)
        if allow_op:
            # Without this a connecting player cannot op themselves, and /npc
            # is gated on the `hytale:WorldEditor` permission group
            # (NPCCommand.java:74). The console bypasses that check; a player
            # does not.
            line += ["--allow-op"]
        if owner:
            line += ["--owner-name", owner]
    else:
        line += ["--bind", f"127.0.0.1:{port}"]

    if universe:
        line += ["--universe", universe]
    if mods:
        # Comma separated, and *additional* to the install's own mods dir.
        line += ["--mods", ",".join(mods)]
    for command in boot_commands:
        # Deliberately one flag per command rather than one comma-joined value:
        # the parser splits on commas (Options.java:144), so a command
        # containing a comma -- `npc spawn x --position 1,2,3` -- would be torn
        # into fragments that each fail on their own.
        line += ["--boot-command", command]
    return line


class _Server:
    """One launched server, its log, and its stdin.

    Only ever one: the bridge binds a fixed port and a second server would
    contend for it. Holding the `Popen` rather than looking up a PID by port is
    deliberate -- :func:`stop` must never be able to kill a server this console
    did not start, because that port is a shared lease other work depends on.
    """

    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.lines: deque[tuple[int, str]] = deque(maxlen=LOG_LINES)
        self.command: list[str] = []
        self.mode = ""
        self.log_path: Path | None = None
        self.reached: dict[str, bool] = {}
        self.session_id = ""
        self.sequence = 0
        self._sink: TextIO | None = None
        self._log_lock = threading.Lock()

    def _latch(self, line: str) -> None:
        """Record boot milestones as they stream past, once, permanently.

        Latched rather than searched-for later because the server can outlive
        the bounded browser transcript; early boot proof must survive even
        after its original line ages out of memory.

        The server's own log is the authority here, and it has to be: the game
        port is **QUIC over UDP**, so it never appears in `netstat` output
        filtered on LISTENING, which is TCP-only. Checking that way reports a
        perfectly healthy client-facing server as having no game port -- a
        mistake this project has now made once, and the reason the console
        answers this question itself instead of leaving it to a shell command.
        """

        for key, marker in MILESTONES.items():
            if not self.reached.get(key) and marker in line:
                self.reached[key] = True

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _record(self, line: str) -> int:
        """Append one ordered line to both the live buffer and session file."""

        with self._log_lock:
            self.sequence += 1
            self.lines.append((self.sequence, line))
            if self._sink is not None:
                self._sink.write(line + "\n")
                self._sink.flush()
            elif self.log_path is not None:
                with self.log_path.open(
                    "a", encoding="utf-8", errors="replace",
                ) as sink:
                    sink.write(line + "\n")
            return self.sequence

    def _pump(self, process: subprocess.Popen[str], path: Path) -> None:
        """Drain stdout into the ring buffer and a file, until EOF.

        Both, not either: the ring buffer is what the page shows, and the file
        is what survives a console restart. A server that dies at boot writes
        its reason once, and losing it means launching again to read it.
        """

        with path.open("a", encoding="utf-8", errors="replace") as sink:
            with self._log_lock:
                self._sink = sink
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    line = line.rstrip("\n")
                    self._latch(line)
                    self._record(line)
            finally:
                with self._log_lock:
                    self._sink = None

    def start(self, command: list[str], mode: str, cwd: Path) -> None:
        if self.running():
            raise ServerError("a server is already running; stop it first")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.log_path = RUNTIME / f"server-{self.session_id}.log"
        self.lines.clear()
        self.sequence = 0
        self.reached = {}
        self.command = list(command)
        self.mode = mode
        self.process = subprocess.Popen(
            command, cwd=str(cwd),
            # stdin stays open so commands can be typed at the running server:
            # its console thread reads lines through JLine
            # (ConsoleModule.java:102) and strips a leading '/'.
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        self._record(_console_line(
            f"session {self.session_id} started · pid {self.process.pid}",
        ))
        self._record(_console_line(
            "launch: " + subprocess.list2cmdline(command),
        ))
        for index, part in enumerate(command[:-1]):
            if part == "--boot-command":
                self._record(_console_line(f"> [boot] {command[index + 1]}"))
        threading.Thread(target=self._pump, daemon=True,
                         args=(self.process, self.log_path)).start()

    def send(self, command: str) -> None:
        if not self.running():
            raise ServerError("no server is running")
        assert self.process is not None and self.process.stdin is not None
        self._record(_console_line(f"> {command}"))
        self.process.stdin.write(command.rstrip("\n") + "\n")
        self.process.stdin.flush()

    def stop(self, timeout: float = 20.0) -> int | None:
        if self.process is None:
            return None
        if self.process.poll() is None:
            try:
                # `stop` is a real server command and shuts down cleanly,
                # saving the universe. Killing the process does not.
                self.send("stop")
                self.process.wait(timeout=timeout)
            except (ServerError, OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait(timeout=10.0)
        return self.process.returncode


_SERVER = _Server()


def launch(mode: str = "client", **kwargs: Any) -> dict[str, Any]:
    """Start a server and return its status. Raises if one already runs."""

    facts = install()
    if not facts["ready"]:
        missing = [n for n, ok in facts["present"].items() if not ok]
        raise ServerError(f"cannot launch, missing: {', '.join(missing)}")

    universe = kwargs.pop("universe", None)
    if universe is None:
        universe = str(RUNTIME / "universe")
        Path(universe).mkdir(parents=True, exist_ok=True)

    command = argv(mode, universe=universe, **kwargs)
    _SERVER.start(command, mode, cwd=INSTALL / "Server")
    return status()


def status() -> dict[str, Any]:
    """What is running, if anything, and how to reproduce it."""

    process = _SERVER.process
    return {
        "running": _SERVER.running(),
        "mode": _SERVER.mode,
        "pid": process.pid if process else None,
        "exit_code": process.poll() if process else None,
        "command": _SERVER.command,
        "log_path": str(_SERVER.log_path) if _SERVER.log_path else None,
        "lines": len(_SERVER.lines),
        "session_id": _SERVER.session_id or None,
        # False rather than absent for a milestone not yet seen: the page
        # renders these as a checklist, and a missing key would read as "not
        # applicable" when the truth is "has not happened yet".
        "reached": {key: _SERVER.reached.get(key, False) for key in MILESTONES},
    }


def log(limit: int = 400) -> list[str]:
    """The tail of the server's output."""

    lines = [line for _, line in _SERVER.lines]
    return lines[-limit:] if limit > 0 else lines


def transcript(after: int = 0, limit: int = 2_000) -> dict[str, Any]:
    """Incremental current-session stdout and commands for the browser."""

    entries = list(_SERVER.lines)
    first = entries[0][0] if entries else _SERVER.sequence + 1
    reset = bool(after and after < first - 1)
    if reset:
        after = 0
    available = [entry for entry in entries if entry[0] > after]
    selected = available[:max(1, min(int(limit), LOG_LINES))]
    cursor = selected[-1][0] if selected else max(after, _SERVER.sequence)
    return {
        "session_id": _SERVER.session_id or None,
        "lines": [line for _, line in selected],
        "cursor": cursor,
        "total": len(entries),
        "has_more": len(available) > len(selected),
        "reset": reset,
        "truncated": bool(entries and first > 1),
        "log_path": str(_SERVER.log_path) if _SERVER.log_path else None,
    }


def _console_line(message: str) -> str:
    stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    return f"[{stamp.replace('+00:00', 'Z')}] [CONSOLE] {message}"


def send(command: str) -> dict[str, Any]:
    """Type a command at the running server.

    Refuses nothing -- the server's own reply is the authority on whether a
    command worked, and predicting refusals here would drift from it. The UI
    warns about `AbstractPlayerCommand` cases up front instead, using
    :data:`NEEDS_PLAYER`.
    """

    _SERVER.send(command)
    return {"sent": command}


def stop() -> dict[str, Any]:
    """Shut the server down cleanly, saving the universe."""

    code = _SERVER.stop()
    return {"stopped": True, "exit_code": code}
