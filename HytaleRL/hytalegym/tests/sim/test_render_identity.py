"""Identity gate: the gen-5 render reproduces the original engine bit for bit.

The original and the render run in one subprocess on a ``git archive`` of the
render's source commit (never the working tree, where other sessions hold
uncommitted engine edits), through the fixed-seed rollouts the existing combat
tests run (``bench.identity_rollouts``: the null-arsenal certified-melee rollout
and the iron-sword/bombs arsenal rollout of test_observation_and_runtime.py),
and every array of params, configs, resets, final states and trajectories is
compared byte for byte. A negative control renders gen 5 with a one-constant
patch and must be caught.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from hytalegym.sim.common import bench
from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource, Patch
from hytalegym.sim.gen5.overlay import OVERLAY as GEN5


@pytest.fixture(scope="module")
def head(tmp_path_factory):
    source = HeadSource()
    root = tmp_path_factory.mktemp("identity")
    return source, root, bench._head_tree(source.commit, root)


def _identity(rendered: Path, tree: Path, out: Path) -> dict:
    completed = subprocess.run(
        [sys.executable, "-P", str(Path(bench.__file__)), "identity",
         "--rendered", str(rendered), "--json", str(out)],
        env=bench._worker_env(tree), capture_output=True, text=True,
    )
    assert out.is_file(), completed.stderr[-4000:]
    summary = json.loads(out.read_text())
    summary["exit"] = completed.returncode
    return summary


def test_gen5_render_reproduces_the_original_bit_for_bit(head):
    source, root, tree = head
    rendered = root / "gen5"
    R.render(GEN5, rendered, source=source)
    summary = _identity(rendered, tree, root / "identity.json")
    assert summary["identical"] and summary["exit"] == 0, summary["first_mismatch"]
    assert summary["mismatched_arrays"] == 0 and summary["max_abs_difference"] == 0.0
    assert summary["arrays"] > 1500 and summary["left_sha256"] == summary["right_sha256"]
    assert Path(summary["original_module"]).resolve().is_relative_to(tree.resolve())
    assert Path(summary["rendered_module"]).resolve().is_relative_to(rendered.resolve())


def test_step_clock_redirects_preserve_gen5_values(head):
    """The gen-7 step-clock rows, resolved against the gen-5 profile (today's
    1/30 s, 0.045 s and three timing profiles), must render a bit-identical
    engine: the redirect machinery changes only what the profile changes."""

    from hytalegym.sim.gen7.overlay import STEP_CLOCK

    source, root, tree = head
    rendered = root / "gen5_clock"
    manifest = R.render(replace(GEN5, redirects=STEP_CLOCK), rendered, source=source)
    assert len(manifest["applied"]) == len(STEP_CLOCK) == 13
    summary = _identity(rendered, tree, root / "clock.json")
    assert summary["identical"] and summary["exit"] == 0, summary["first_mismatch"]


def test_optimized_gen5_render_is_bit_identical(head):
    """Optimisation patches are offered only while the identity gate holds."""

    from hytalegym.sim.gen5.overlay import OPTIMIZED

    source, root, tree = head
    rendered = root / "gen5_optimized"
    manifest = R.render(OPTIMIZED, rendered, source=source)
    assert {a["id"] for a in manifest["applied"]} == {p.id for p in OPTIMIZED.patches}
    summary = _identity(rendered, tree, root / "optimized.json")
    assert summary["identical"] and summary["exit"] == 0, summary["first_mismatch"]


def test_identity_gate_detects_a_one_constant_change(head):
    source, root, tree = head
    control = replace(GEN5, patches=(Patch(
        "negative_control", "combat/types.py",
        "PLAYER_BASE_SPEED = 5.5\n", "PLAYER_BASE_SPEED = 5.75\n", "control", "negative control"),))
    rendered = root / "control"
    R.render(control, rendered, source=source)
    summary = _identity(rendered, tree, root / "control.json")
    assert not summary["identical"] and summary["exit"] == 1
    assert summary["mismatched_arrays"] > 0 and summary["first_mismatch"]
