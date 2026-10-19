"""Paths and helpers shared by the ``tests/sim`` suite."""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

HYTALEGYM = Path(__file__).resolve().parents[2]
TOOLS = HYTALEGYM / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
SIM = HYTALEGYM / "hytalegym" / "sim"
REPOSITORY = HYTALEGYM.parents[1]
INSTALL = Path.home() / "Library" / "Application Support" / "Hytale" / "install"
ARCHIVES = {
    "gen5": INSTALL / "v0.5" / "package" / "game" / "latest" / "Assets.zip",
    "gen7": INSTALL / "release" / "package" / "game" / "latest" / "Assets.zip",
}


def data_dir(generation: str) -> Path:
    return SIM / generation / "data"


def manifest(generation: str) -> dict:
    return json.loads((data_dir(generation) / "MANIFEST.json").read_text(encoding="utf-8"))


def table(generation: str, family: str) -> dict:
    return json.loads((data_dir(generation) / f"{family}.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def archive_sha256(path: str) -> str:
    from content_extract.archive import file_sha256

    return file_sha256(path)


def mount_fresh(out_dir):
    """Mount a render under its manifest name, first dropping an earlier mount of that name made by another module.

    ``hytalegym.sim.common.render.mount`` refuses a second render under a name that is already imported (the gen-7 package is
    mounted by several modules in one pytest process), so a module that renders its own copy drops the earlier one first.
    """

    from hytalegym.sim.common import render as render_module

    name = render_module.read_manifest(Path(out_dir).resolve())["package"]
    unmount(name)
    return render_module.mount(out_dir)


def unmount(package: str) -> None:
    """Forget a mounted render: its package and every submodule, so the next mount under that name starts clean.

    The parent package's attribute is dropped too. ``render.mount`` sets it only after the package has run, so the package's
    own ``import a.b.c as x`` statements read the parent attribute first; a stale attribute from an earlier render would bind
    them to that render's modules (seen as a cross-render ``world`` module in the executor's line-of-sight call).
    """

    for key in [key for key in sys.modules if key == package or key.startswith(package + ".")]:
        del sys.modules[key]
    parent_name, _, leaf = package.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None and hasattr(parent, leaf):
        delattr(parent, leaf)
