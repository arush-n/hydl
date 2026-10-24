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
the ``HYTALERL_SERVER_VERSION`` environment variable (default 0.7.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import os
from pathlib import Path

SERVER_VERSION_ENV = "HYTALERL_SERVER_VERSION"
DEFAULT_SERVER_VERSION = "0.7.2"

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
    # Folder under artifacts/worldgen holding the Region library that world
    # evidence recertification reads. None keeps the 0.5 pilot folder.
    region_library: str | None = None
    # Subfolder of tests/fidelity/fixtures/world holding this line's recorded
    # World evidence. None keeps the 0.5 root.
    world_fixture_root: str | None = None
    # Capacity report of the line's Region library, relative to artifacts/worldgen.
    # None keeps the 0.5 report.
    world_capacity_report: str | None = None
    # Expected semantic SHA-256 of the line's Region library, and expected projection
    # digest of the recertification target seed. None means unchecked.
    region_library_sha256: str | None = None
    seed_projection_sha256: str | None = None
    # Rows of the World geometry token row (traversal and exception candidates
    # share them; overflow clears the actor's row). Read at import by
    # ``hytalegym.jax.combat.observation.v3.tokens.world``; None leaves the
    # policy config unbuildable, so every line must set it.
    world_geometry_token_capacity: int | None = None
    # Artifact that evidences ``world_geometry_token_capacity`` (named in the
    # policy config manifest, so it is checkpoint-visible).
    world_geometry_capacity_evidence: str | None = None
    # Projectile sweeps are checked inside the geometry window only, not refused
    # past its edge (opt-in, owner ruling 2026-10-23). False keeps the window gate.
    projectile_clear_past_window_edge: bool = False


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


_WORLD_FIXTURE_ROOT = Path("tests") / "fidelity" / "fixtures" / "world"


def world_fixture_root(
    hytalegym_root: str | Path | None = None,
    version: str | None = None,
) -> Path:
    """Return the recorded World evidence folder of a server line.

    ``hytalegym_root`` is the directory that holds ``tests/`` (default: the checkout
    this module lives in). The 0.5 line reads the root itself; a line that names
    ``world_fixture_root`` reads that subfolder.
    """

    base = (
        Path(__file__).resolve().parents[2]
        if hytalegym_root is None
        else Path(hytalegym_root)
    )
    root = base / _WORLD_FIXTURE_ROOT
    folder = server_profile(version).world_fixture_root
    return root if folder is None else root / folder


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
    "world_fixture_root",
]
