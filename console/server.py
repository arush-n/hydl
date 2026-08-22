"""Application assembly and entry point.

    python -m console.server        # http://127.0.0.1:8770

This module wires the pieces together and does nothing else. Routes live in
:mod:`console.api.routes`; the engine lives in :mod:`console.core`.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from console.api.routes import STATIC, router  # noqa: E402

#: Loopback by default, because a console that binds every interface the moment
#: it starts exposes an unauthenticated launch API to the local network.
#:
#: `HYTALERL_CONSOLE_HOST` overrides it, and a container MUST set it to
#: `0.0.0.0`. Inside a container `127.0.0.1` is the container's own loopback, so
#: `-p 8770:8770` publishes a port that nothing is listening on and the browser
#: gets a connection reset with a server that looks healthy from its own logs.
HOST = os.environ.get("HYTALERL_CONSOLE_HOST", "127.0.0.1")
PORT = int(os.environ.get("HYTALERL_CONSOLE_PORT", "8770"))

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Restore canonical worldgen designs before serving.

    The local design store is scratch and gets cleared, but the flat world every
    flat scenario is meant to use has to survive that. Startup is the right
    moment: doing it from `worldgen.saved()` made a read-only listing write to
    disk, which materialised an extra design inside a test's temporary store.
    Non-destructive -- a design already present is never overwritten.
    """

    try:
        from console.core.worlds import worldgen

        restored = worldgen.ensure_canonical_designs()
        if restored:
            print(f"restored seeded worldgen designs: {', '.join(restored)}")
    except Exception as error:  # noqa: BLE001 - never block startup on a seed
        print(f"could not restore seeded worldgen designs: {error}")
    yield


app = FastAPI(
    lifespan=lifespan,
    title="HytaleRL console",
    description=(
        "A frontend for the ADK. Configure a rollout, run it against the live "
        "environment, and watch it back in 3D. See console/docs/."
    ),
    docs_url="/api/docs",
)


@app.middleware("http")
async def revalidate_browser_code(request, call_next):
    """Prevent stale mixed-version ES module graphs after Console updates."""

    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


app.include_router(router)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _port_owner(port: int) -> str:
    """Who already has the port, so a conflict names the culprit."""

    import subprocess

    try:
        out = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:  # noqa: BLE001 - diagnostics must never be fatal
        return ""
    for line in out.splitlines():
        if f":{port} " in line and "LISTENING" in line.upper():
            return f" (held by PID {line.split()[-1]})"
    return ""


def _free(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
            return True
        except OSError:
            return False


def main() -> None:
    import uvicorn

    reload = "--no-reload" not in sys.argv
    port = PORT
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])

    # Check the port *before* announcing success. uvicorn prints its bind error
    # and then exits 0, and the banner used to print after it -- so a second
    # instance looked like it had started while serving nothing, and the
    # browser kept talking to the first (possibly stale) server. That is a
    # genuinely hard failure to diagnose from the page.
    if not _free(HOST, port):
        print(f"ERROR: {HOST}:{port} is already in use{_port_owner(port)}.")
        print("  Another console is running. Either use it, stop it, or start")
        print(f"  this one elsewhere:  python -m console.server --port {port + 1}")
        raise SystemExit(1)

    print(f"console -> http://{HOST}:{port}   (API docs at /api/docs)")
    print(
        f"  hot reload: {'on' if reload else 'off'}"
        "  ·  edit console/static/** and the page follows"
    )
    if reload:
        # Reload on Python changes; the browser polls /api/version for static
        # ones and swaps CSS without losing the loaded run.
        uvicorn.run(
            "console.server:app",
            host=HOST,
            port=port,
            log_level="warning",
            reload=True,
            reload_dirs=[str(Path(__file__).resolve().parent)],
        )
    else:
        uvicorn.run(app, host=HOST, port=port, log_level="warning")


if __name__ == "__main__":
    main()
