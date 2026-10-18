"""The rendered artifact-root anchor names the source tree through the repo-root shim and directly.

``world/tokens/__init__.py`` climbs out of its package for ``_WORLDGEN_ARTIFACT_ROOT``
(``Path(__file__).resolve().parents[5]`` in the source). The renderer re-anchors that climb on
``hytalegym.envs``. Anchoring on ``hytalegym`` itself resolved one directory too high when the package was
imported through the repo-root shim (``HytaleRL/hytalegym/__init__.py``): the staged Region library was then
looked up under ``/artifacts`` and the learner-v3 actor-evidence hash changed (2026-10-18). This test runs the
rendered statement in two processes, one importing ``hytalegym`` through the shim (cwd ``HytaleRL``, the
``cd HytaleRL && python -m ...`` case) and one importing the inner package directly, and asserts that both
resolve to the source tree's artifact root.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.gen5.overlay import OVERLAY as GEN5

TOKENS = "world/tokens/__init__.py"
HERE = Path(__file__).resolve()
SHIM_PARENT = HERE.parents[3]  # HytaleRL: holds the repo-root shim package hytalegym/
OUTER = HERE.parents[2]  # HytaleRL/hytalegym: holds the inner package hytalegym/
ARTIFACT_ROOT = SHIM_PARENT / "artifacts" / "worldgen"
MANIFEST = ARTIFACT_ROOT / "region-library-pilot-v2" / "manifest.json"

_PROBE = """
import json
from pathlib import Path
import hytalegym
ns = {"Path": Path}
exec(%(statement)r, ns)
root = ns["_WORLDGEN_ARTIFACT_ROOT"]
print(json.dumps({
    "package": hytalegym.__file__,
    "root": str(root),
    "manifest": str(root / "region-library-pilot-v2" / "manifest.json"),
}))
"""


@pytest.fixture(scope="module")
def rendered_tokens(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("anchors") / "engine"
    R.render(GEN5, out, source=HeadSource())
    return (out / TOKENS).read_text(encoding="utf-8")


def _artifact_root_statement(text: str) -> str:
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_WORLDGEN_ARTIFACT_ROOT" for t in node.targets):
            return ast.get_source_segment(text, node)
    raise AssertionError("the rendered tokens module does not assign _WORLDGEN_ARTIFACT_ROOT")


def _probe(cwd: Path, pythonpath: str | None, statement: str) -> dict:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if pythonpath is not None:
        env["PYTHONPATH"] = pythonpath
    done = subprocess.run(
        [sys.executable, "-c", _PROBE % {"statement": statement}],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_rendered_anchor_never_names_the_hytalegym_package(rendered_tokens: str) -> None:
    statement = _artifact_root_statement(rendered_tokens)
    assert '__import__("hytalegym.envs", fromlist=["envs"]).__file__' in statement
    assert '__import__("hytalegym").__file__' not in rendered_tokens
    assert str(SHIM_PARENT) not in rendered_tokens and str(OUTER) not in rendered_tokens


def test_artifact_root_resolves_through_the_shim_and_directly(rendered_tokens: str, tmp_path: Path) -> None:
    if not MANIFEST.is_file():
        pytest.skip(f"staged Region library absent ({MANIFEST}); the artifact is gitignored")
    statement = _artifact_root_statement(rendered_tokens)
    through_shim = _probe(SHIM_PARENT, None, statement)
    direct = _probe(tmp_path, str(OUTER), statement)
    assert Path(through_shim["package"]).resolve() == (SHIM_PARENT / "hytalegym" / "__init__.py").resolve()
    assert Path(direct["package"]).resolve() == (OUTER / "hytalegym" / "__init__.py").resolve()
    for mode in (through_shim, direct):
        assert Path(mode["root"]).resolve() == ARTIFACT_ROOT.resolve(), mode
        assert Path(mode["manifest"]).is_file(), mode
