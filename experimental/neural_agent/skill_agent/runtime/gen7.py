"""Run skill-layer code on a private gen-7 render (2026-10-19, the owner: "just do gen 7 even if it is not certified").

The gen-7 simulator is a render of the committed gen-5 engine with the 0.7.2 overlay
(``hytalegym/sim/common/render.py``, ``--generation gen7``). The shared render under ``sim/gen7/engine`` is
re-rendered in place by the port's coordinator, so this launcher renders its own copy and mounts it:
``render.redirect_imports`` makes every ``hytalegym.jax`` import load the render. It must run before ``hytalegym`` is
first imported, so this module imports no ``hytalegym`` code at its top. The launcher then runs the target module or
script as ``__main__``.

One workaround: the redirect does not carry the v3 observation package's legacy module aliases
(``observation/v3/__init__.py`` registers ``world_tokens`` and fourteen others under the rendered package's name, so
``adk``'s ``from hytalegym.jax.combat.observation.v3.world_tokens import ...`` would fail). After mounting, the
launcher registers the package's module attributes under the ``hytalegym`` names. Reported to the coordinator.

CLI (from the repo root)::

    PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym JAX_PLATFORMS=cpu .venv/bin/python -m \\
        experimental.neural_agent.skill_agent.runtime.gen7 --engine DIR [--render] -- MODULE_OR_SCRIPT [ARGS...]

``--render`` (re)renders the gen-7 engine into DIR first (about 10 seconds); without it DIR must hold a render.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import runpy
import subprocess
import sys
import types
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[4]
RENDER_PY = REPO / "HytaleRL/hytalegym/hytalegym/sim/common/render.py"
V3 = "hytalegym.jax.combat.observation.v3"


def render(engine: Path) -> None:
    """Render the gen-7 engine into ``engine`` from the committed source, in a child process (it imports hytalegym)."""

    subprocess.run([sys.executable, "-m", "hytalegym.sim.common.render", "--generation", "gen7", "--out", str(engine)],
                   check=True, cwd=REPO / "HytaleRL/hytalegym")


def mount(engine: Path) -> None:
    """Make every ``hytalegym.jax`` import load the render at ``engine``, with the v3 aliases restored."""

    if any(name == "hytalegym" or name.startswith("hytalegym.") for name in sys.modules):
        raise RuntimeError("mount must run before hytalegym is imported")
    spec = importlib.util.spec_from_file_location("_skill_agent_render", RENDER_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.redirect_imports(engine)
    package = importlib.import_module(V3)
    for attribute, value in vars(package).items():
        if isinstance(value, types.ModuleType):
            sys.modules.setdefault(f"{V3}.{attribute}", value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", type=Path, required=True, help="the private render's directory")
    parser.add_argument("--render", action="store_true", help="render the gen-7 engine into --engine first")
    parser.add_argument("target", help="a module (a.b.c) or a script (path.py) to run as __main__")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    if options.render:
        render(options.engine)
    mount(options.engine)
    sys.argv = [options.target, *[arg for arg in options.args if arg != "--"]]
    if options.target.endswith(".py"):
        runpy.run_path(options.target, run_name="__main__")
    else:
        runpy.run_module(options.target, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "mount", "render"]
