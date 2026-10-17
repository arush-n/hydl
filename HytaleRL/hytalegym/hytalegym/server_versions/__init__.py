"""Per-release-line Hytale server profiles.

Each supported server release line has one module here (``v0_5``, ``v0_7``)
holding the values that differ between server versions: the bridge build the
gates expect, the native crafting catalog size, and which simulator rulesets
may be evaluated against that server. Gates ask ``server_profile`` for these
values instead of hard-coding them.

A field left as ``None`` means "keep the call site's own 0.5 constant", so the
0.5 line runs exactly the code paths it ran before profiles existed. Supporting
a new release line means adding one module and one ``_MODULES`` entry.

The active line comes from the version a live server reports (for example
``info["native_server_version"]``) or, for checks made before connecting, from
the ``HYTALERL_SERVER_VERSION`` environment variable (default 0.5.7).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import os

SERVER_VERSION_ENV = "HYTALERL_SERVER_VERSION"
DEFAULT_SERVER_VERSION = "0.5.7"

_MODULES = {
    (0, 5): "hytalegym.server_versions.v0_5",
    (0, 7): "hytalegym.server_versions.v0_7",
}


@dataclass(frozen=True)
class ServerProfile:
    """Version-specific values for one server release line."""

    release_line: tuple[int, int]
    certified_version: str
    # Path of the bridge JAR relative to the HytaleRL directory, and its pinned
    # SHA-256 (uppercase hex).
    bridge_jar: str | None = None
    bridge_jar_sha256: str | None = None
    # Recipes in the server's CraftingRecipe asset map.
    crafting_recipe_count: int | None = None
    # Checkpoint ``hytale_version`` labels (simulator rulesets) that native
    # evaluation accepts on this line. None keeps the exact-version check.
    checkpoint_rulesets: tuple[str, ...] | None = None


def release_line(version: object) -> tuple[int, int] | None:
    """Return ``(major, minor)`` for a dotted version string, else None."""

    if not isinstance(version, str):
        return None
    parts = version.split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        return None
    return int(parts[0]), int(parts[1])


def is_supported_server_version(version: object) -> bool:
    """Return whether a version belongs to a release line with a profile."""

    return release_line(version) in _MODULES


def server_profile(version: str | None = None) -> ServerProfile:
    """Return the profile for ``version``, or for the configured version."""

    if version is None:
        version = os.environ.get(SERVER_VERSION_ENV) or DEFAULT_SERVER_VERSION
    line = release_line(version)
    if line not in _MODULES:
        raise ValueError(
            f"unsupported Hytale server version {version!r}; supported release "
            f"lines: {', '.join(f'{a}.{b}' for a, b in sorted(_MODULES))}"
        )
    return import_module(_MODULES[line]).PROFILE


def checkpoint_runs_on_server(checkpoint_version: object, server_version: object) -> bool:
    """Return whether native evaluation may run a checkpoint on a server.

    ``checkpoint_version`` is the checkpoint's ``hytale_version`` label (the
    simulator ruleset it was trained on). An exact match always runs; a line's
    ``checkpoint_rulesets`` adds the rulesets accepted across versions.
    """

    if checkpoint_version == server_version:
        return True
    if not is_supported_server_version(server_version):
        return False
    accepted = server_profile(str(server_version)).checkpoint_rulesets
    return accepted is not None and checkpoint_version in accepted


__all__ = [
    "DEFAULT_SERVER_VERSION",
    "SERVER_VERSION_ENV",
    "ServerProfile",
    "checkpoint_runs_on_server",
    "is_supported_server_version",
    "release_line",
    "server_profile",
]
