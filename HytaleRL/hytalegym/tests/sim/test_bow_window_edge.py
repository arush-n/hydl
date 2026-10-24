"""Bow and crossbow window edge (owner ruling 2026-10-23: check inside the window only).

The projectile cases (open floor 3, 4.5, 6, 8 blocks; walls inside and at the edge) are in
``tests/jax/world/geometry/test_projectile_window_edge.py``. This file pins the switches: the repo default stays off
(gen 5 unchanged), and only the 0.7.2 server line turns it on for the native host. The gen-7 render is NOT wired yet:
renders read HEAD, so the window-edge option needs a committed shared-source change before a Patch can anchor on it. Live receipt: ``artifacts/fidelity_evidence/gen7_probes/shortbow_tiers_b4eec032.json``.
"""

from __future__ import annotations

import inspect

from hytalegym.jax.combat.arsenal import environment as repo_environment
from hytalegym.server_versions import server_profile

PARAMETER = "projectile_clear_past_window_edge"


def test_repo_default_stays_off() -> None:
    parameter = inspect.signature(
        repo_environment.geometry_arsenal_world_capabilities
    ).parameters[PARAMETER]
    assert parameter.default is False


def test_only_the_0_7_2_line_turns_the_native_host_option_on() -> None:
    assert server_profile("0.7.2").projectile_clear_past_window_edge is True
    assert server_profile("0.5.7").projectile_clear_past_window_edge is False
