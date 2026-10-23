"""Spider web movement settings reach the 0.7.2 line through the 0.7.2 Region library (FIDELITY row 3b).

0.7.2 source: Assets.zip ``Server/Item/Items/Deco/Deco_SpiderWeb.json``, ``BlockType.MovementSettings``
(HorizontalSpeedMultiplier 0.45, JumpForceMultiplier 0.45, Drag 0.82, TerminalVelocityModifier 0.45).
The engine reads these per cell: ``RegionGeometryCapture.java`` packs ``movementValues`` as
friction, drag, horizontal, jump, climb x3, terminal, bounce, so drag is column 1, horizontal 2, jump 3
and terminal 7. The 0.7.2 profile (``server_versions.v0_7``) selects ``region-library-pilot-v2-072``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hytalegym.server_versions import server_profile

ARTIFACTS = Path(__file__).resolve().parents[3] / "artifacts" / "worldgen"
COLUMNS = [1, 2, 3, 7]  # drag, horizontal speed, jump force, terminal velocity
WEB_072 = np.array([0.82, 0.45, 0.45, 0.45])
WEB_057 = np.array([0.45, 0.65, 0.5, 0.5])


def _cells(folder: str) -> np.ndarray:
    rows = []
    for path in sorted((ARTIFACTS / folder).rglob("*.npz")):
        with np.load(path) as archive:
            if "cell_movement" in archive.files:
                rows.append(archive["cell_movement"][:, COLUMNS])
    return np.concatenate(rows)


def _has(cells: np.ndarray, web: np.ndarray) -> bool:
    return bool(np.all(np.isclose(cells, web, atol=1e-6), axis=1).any())


def test_0_7_2_profile_selects_the_0_7_2_region_library() -> None:
    profile = server_profile("0.7.2")
    assert profile.region_library == "region-library-pilot-v2-072"
    manifest = json.loads((ARTIFACTS / profile.region_library / "manifest.json").read_text())
    assert manifest["capture_contract"]["server_version"] == "0.7.2"


def test_0_7_2_library_carries_the_0_7_2_web_values_and_not_the_0_5_7_values() -> None:
    cells = _cells("region-library-pilot-v2-072")
    assert _has(cells, WEB_072)
    assert not _has(cells, WEB_057)


def test_0_5_library_is_the_positive_control_for_the_0_5_7_web_values() -> None:
    cells = _cells("region-library-pilot-v2")
    assert _has(cells, WEB_057)
    assert not _has(cells, WEB_072)
