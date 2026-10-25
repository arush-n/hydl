"""Block breaking on gen 7 (owner yes 2026-10-23): the Region removal certificate's tier-1 rule.

The rule itself (a side neighbour resting on its own Down face no longer refuses; a neighbour above the target, leaves and
unknown neighbours still refuse) is tested on the repo engine with the options passed explicitly, in
``tests/jax/world/region/test_removal_cascade_provider_rule.py``. This file pins the switches: the repo defaults stay off
(gen 5 keeps the any-neighbour rule) and the gen-7 render turns all three on through the overlay Patches
``cascade_support_faces``, ``cascade_provider_cell_rule`` and ``cascade_catalog_support_faces``. The catalog builder's
default flips too, so a catalog built directly (not through the executor assets) still carries the faces the rule needs.
Design: tmp/lane_cascade_design/DESIGN.md.
"""

from __future__ import annotations

import inspect
from importlib import import_module

import pytest

from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7_OVERLAY

from .support import mount_fresh, render_for_test, unmount

SWITCHES = (
    ("combat.arsenal.world_actions.executor", "compile_region_block_executor_assets", "support_faces"),
    ("combat.arsenal.world_actions.executor", "make_region_block_action_surface_executor", "provider_cell_rule"),
    ("world.block_actions._resolution", "block_action_catalog_from_local", "support_faces"),
)


def _default(package: str, module: str, function: str, parameter: str):
    target = getattr(import_module(f"{package}.{module}"), function)
    return inspect.signature(target).parameters[parameter].default


@pytest.mark.parametrize(("module", "function", "parameter"), SWITCHES)
def test_repo_defaults_stay_off(module: str, function: str, parameter: str) -> None:
    assert _default("hytalegym.jax", module, function, parameter) is False


@pytest.fixture(scope="module")
def gen7_package(tmp_path_factory):
    path = tmp_path_factory.mktemp("break_cascade_rule") / "gen7"
    path = render_for_test(GEN7_OVERLAY, path, source=HeadSource())
    package = mount_fresh(path).__name__
    try:
        yield package
    finally:
        unmount(package)


@pytest.mark.parametrize(("module", "function", "parameter"), SWITCHES)
def test_the_gen7_render_turns_the_rule_on(gen7_package, module: str, function: str, parameter: str) -> None:
    assert _default(gen7_package, module, function, parameter) is True
