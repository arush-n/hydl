"""The porting tools run on this repository. Fast: the identity gate is test_render_identity.py."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

HYTALEGYM = Path(__file__).resolve().parents[2]
TOOLS = HYTALEGYM / "tools"
WRITTEN = (
    HYTALEGYM / "hytalegym/sim/common/content/redirects.json",
    HYTALEGYM / "tests/sim/test_render.py",
)


def _digests() -> dict:
    return {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in WRITTEN}


def test_regen_swap_plan_dry_run_reports_the_step_clock_redirects_and_writes_nothing():
    before = _digests()
    completed = subprocess.run(
        [sys.executable, "-m", "sim_port.regen_swap_plan", "--dry-run"],
        cwd=TOOLS, capture_output=True, text=True, timeout=900,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert "dry run, nothing written" in completed.stdout, completed.stdout
    assert "step-clock redirects: 13 " in completed.stdout, completed.stdout
    assert _digests() == before
