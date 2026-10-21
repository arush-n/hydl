"""Server version / ``ServerProfile`` -> generation profile -> data directory.

Selection follows ``hytalegym.server_versions``: the release line of the given
version (or of ``HYTALERL_SERVER_VERSION``, default 0.7.2). A new generation is
one ``sim/genN/profile.py`` plus one ``GENERATION_MODULES`` entry, mirroring
the server profile registry.
"""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import Union

from hytalegym.server_versions import (
    DEFAULT_SERVER_VERSION,
    SERVER_VERSION_ENV,
    ServerProfile,
    release_line,
)
from hytalegym.sim.common.profile import SimProfile

GENERATION_MODULES = {
    (0, 5): "hytalegym.sim.gen5.profile",
    (0, 7): "hytalegym.sim.gen7.profile",
}

Selector = Union[None, str, ServerProfile, SimProfile]


def sim_profile(selector: Selector = None) -> SimProfile:
    """Return the generation profile for a version, server profile or profile."""

    if isinstance(selector, SimProfile):
        return selector
    if isinstance(selector, ServerProfile):
        line = selector.release_line
        label = selector.certified_version
    else:
        label = selector or os.environ.get(SERVER_VERSION_ENV) or DEFAULT_SERVER_VERSION
        line = release_line(label)
    if line not in GENERATION_MODULES:
        raise ValueError(
            f"no simulator generation for Hytale {label!r}; registered release lines: "
            + ", ".join(f"{a}.{b}" for a, b in sorted(GENERATION_MODULES))
        )
    return import_module(GENERATION_MODULES[line]).PROFILE


def generations() -> tuple[SimProfile, ...]:
    return tuple(import_module(GENERATION_MODULES[line]).PROFILE for line in sorted(GENERATION_MODULES))


def data_directory(selector: Selector = None) -> Path:
    return sim_profile(selector).data_dir
