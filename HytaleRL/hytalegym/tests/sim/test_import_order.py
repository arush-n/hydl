"""``import hytalegym`` loads no ``hytalegym.jax`` module.

A gen-7 render's hooks read the gen-7 profile when the rendered engine is first imported. A launcher that mounts a
render and then switches constructs on (``experimental/neural_agent/skill_agent/runtime/gen7.py`` ``switch_on``) imports
``hytalegym.sim.gen7.profile``, which runs ``hytalegym/__init__.py``. If that import reaches ``hytalegym.jax``, the
engine loads with the default profile and no construct reaches the observation. That happened from 6d9307f to this
test's commit: ``envs/block_type_table.py`` imported ``jax.combat.contracts.semantic`` at module level (found by the
skill layer, LEDGER 2026-10-24, bisected on the flat world's clock).
"""

from __future__ import annotations

import os
import subprocess
import sys

PROBE = "import sys, hytalegym; print(sorted(m for m in sys.modules if m.startswith('hytalegym.jax')))"


def test_importing_hytalegym_loads_no_jax_module():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() == "[]", out.stdout[:400]
