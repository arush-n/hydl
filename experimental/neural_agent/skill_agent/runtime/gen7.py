"""Run skill-layer code on a private gen-7 render (2026-10-19, the owner: "just do gen 7 even if it is not certified").

The gen-7 simulator is a render of the committed gen-5 engine with the 0.7.2 overlay
(``hytalegym/sim/common/render.py``, ``--generation gen7``). The shared render under ``sim/gen7/engine`` is
re-rendered in place by the port's coordinator, so this launcher renders its own copy and mounts it:
``render.redirect_imports`` makes every ``hytalegym.jax`` import load the render. It must run before ``hytalegym`` is
first imported, so this module imports no ``hytalegym`` code at its top. The launcher then runs the target module or
script as ``__main__``.

The redirect serves the v3 observation package's legacy module aliases (``world_tokens`` and the others that
``observation/v3/__init__.py`` registers) under the source package's name (``render.py``, 780b5de), so ``adk``'s
``from hytalegym.jax.combat.observation.v3.world_tokens import ...`` resolves to the rendered module.

CLI (from the repo root)::

    PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym JAX_PLATFORMS=cpu .venv/bin/python -m \\
        experimental.neural_agent.skill_agent.runtime.gen7 --engine DIR [--render] [--construct NAME ...] \\
        -- MODULE_OR_SCRIPT [ARGS...]

``--render`` (re)renders the gen-7 engine into DIR first (about 10 seconds); without it DIR must hold a render.
With ``HYTALEGYM_RENDER_CACHE`` set, ``--render`` mounts the shared cached render for the same source and overlay
(``render.render_cached``: rendered on a miss, read in place on a hit) instead, and DIR is not written.
``--construct NAME`` switches one gen-7 construct on in this process only (for example ``perception_v2``, the
perception channel of ``sim/gen7/PERCEPTION_V2.md``, off by default): the render's hooks read the gen-7 profile when
the rendered engine is first imported, so the launcher replaces the profile after mounting and before the target runs.
The shared profile is untouched; switching it is the port coordinator's call.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[4]
RENDER_PY = REPO / "HytaleRL/hytalegym/hytalegym/sim/common/render.py"
CACHE_ENV = "HYTALEGYM_RENDER_CACHE"
"""``render.RENDER_CACHE_ENV``; spelled out here because this module imports no hytalegym code."""
CACHED_RENDER = "from hytalegym.sim.common import render; print(render.render_cached(render.load_overlay('gen7')))"


def render(engine: Path) -> Path:
    """Render the gen-7 engine and return the directory to mount.

    Unset ``HYTALEGYM_RENDER_CACHE``: render into ``engine`` from the committed source, in a child process (it imports
    hytalegym). Set: the shared render for this source and overlay (``render.render_cached``, which renders only on a
    miss); ``engine`` is not written. The child prints the cached directory as its last line.
    """

    if os.environ.get(CACHE_ENV):
        done = subprocess.run([sys.executable, "-c", CACHED_RENDER], check=True, cwd=REPO / "HytaleRL/hytalegym",
                              stdout=subprocess.PIPE, text=True)
        return Path(done.stdout.strip().splitlines()[-1])
    subprocess.run([sys.executable, "-m", "hytalegym.sim.common.render", "--generation", "gen7", "--out", str(engine)],
                   check=True, cwd=REPO / "HytaleRL/hytalegym")
    return engine


def mount(engine: Path) -> None:
    """Make every ``hytalegym.jax`` import load the render at ``engine``."""

    if any(name == "hytalegym" or name.startswith("hytalegym.") for name in sys.modules):
        raise RuntimeError("mount must run before hytalegym is imported")
    spec = importlib.util.spec_from_file_location("_skill_agent_render", RENDER_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.redirect_imports(engine)


def switch_on(constructs: Sequence[str]) -> None:
    """Turn the named gen-7 constructs on in this process; run it after ``mount``, before the engine is imported."""

    if not constructs:
        return
    import dataclasses

    from hytalegym.sim.gen7 import profile

    profile.PROFILE = dataclasses.replace(
        profile.PROFILE, constructs=dataclasses.replace(profile.PROFILE.constructs, **dict.fromkeys(constructs, True)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", type=Path, required=True,
                        help="the private render's directory (not written by --render when HYTALEGYM_RENDER_CACHE is set)")
    parser.add_argument("--render", action="store_true", help="render the gen-7 engine into --engine first")
    parser.add_argument("--construct", action="append", default=[], metavar="NAME",
                        help="switch a gen-7 construct on in this process (repeatable), e.g. perception_v2")
    parser.add_argument("target", help="a module (a.b.c) or a script (path.py) to run as __main__")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    engine = options.engine.resolve()  # absolute: render() runs in hytalegym's folder
    if options.render:
        engine = render(engine)
    mount(engine)
    switch_on(options.construct)
    sys.argv = [options.target, *[arg for arg in options.args if arg != "--"]]
    if options.target.endswith(".py"):
        runpy.run_path(options.target, run_name="__main__")
    else:
        runpy.run_module(options.target, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "mount", "render", "switch_on"]
