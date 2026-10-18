"""A1: the gen-5 tables reproduce the engine's hand-held values field by field.

Raw extraction (no overlay) is compared with today's engine. Every difference
must be listed exactly once in ``sim/gen5/calibration_057.json`` with its
classification; no entry may be unneeded. Data comes from the installed
0.5.10 archive; ``sim/gen5/evidence_057.json`` proves which inputs equal 0.5.7.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from hytalegym.sim.common.content import adapters as A
from hytalegym.sim.common.content import load

from . import gen5_live as live
from . import support
from .comparisons import compare


@pytest.fixture(scope="module")
def raw():
    return load("0.5.7", apply_overlay=False)


def test_a1_every_difference_is_an_overlay_entry(raw) -> None:
    compared, differences = compare(raw)
    listed = {constant for override in load("0.5.7").overrides for constant in override.constants}
    report = "\n".join(f"{key}: tables={a!r} engine={e!r}" for key, (a, e) in differences.items())
    print(f"A1 compared {compared} values; {len(differences)} differ:\n{report}")
    assert compared > 700
    assert set(differences) == listed, report


def test_a1_overlay_entries_name_the_raw_value(raw) -> None:
    for override in load("0.5.7").overrides:
        node = raw.row(override.family, override.row)
        for part in override.path:
            node = node[int(part)] if isinstance(node, list) else node[part]
        assert node == override.extracted, override
        assert override.value != override.extracted
        assert override.classification in ("version_delta", "transcription_error", "native_measured", "unclassified")


def test_a1_crafting_reproduces_the_native_0_5_7_recording(raw) -> None:
    adapted, recorded = A.native_crafting_recipes(raw), live.native_crafting_recipes()
    assert set(adapted) == set(recorded) and len(recorded) == 1947
    for recipe_id, expected in recorded.items():
        got = adapted[recipe_id]
        for field, value in expected.items():
            if field == "time_seconds":
                assert np.float32(got[field]) == np.float32(value), (recipe_id, field)
            else:
                assert got[field] == value, (recipe_id, field, got[field], value)


def test_a1_blocks_agree_with_the_surrogate_reader() -> None:
    archive_path = support.ARCHIVES["gen5"]
    if not archive_path.is_file():
        pytest.skip("0.5.x is not installed")
    from hytalegym.worldgen.surrogate.assets import HytaleAssetArchive, HytaleAssetError
    from hytalegym.worldgen.surrogate.semantics import LocalBlockSemanticsResolver

    tables = load("0.5.7")
    compared = 0
    with HytaleAssetArchive(archive_path) as archive:
        resolver = LocalBlockSemanticsResolver(archive)
        for key in tables.keys("blocks", "block"):
            block = tables.block(key.split(":", 1)[1])
            try:
                semantics = resolver.resolve(block.block_id)
            except HytaleAssetError:
                continue
            compared += 1
            assert (block.material, block.opacity, block.hitbox_type, block.is_door) == (
                semantics.material,
                semantics.opacity,
                semantics.hitbox_type,
                semantics.is_door,
            ), block.block_id
    print(f"blocks compared with LocalBlockSemanticsResolver: {compared}")
    assert compared > 2000


def test_a1_gen5_inputs_are_0_5_7_by_recorded_hashes() -> None:
    evidence = json.loads((support.SIM / "gen5" / "evidence_057.json").read_text(encoding="utf-8"))
    summary = evidence["summary"]
    assert summary["files_identical"] == summary["files_pinned"] == 28
    assert summary["weapon_graphs_identical"] == summary["weapon_graphs_pinned"] == 240
    assert summary["auxiliary_identical"] == summary["auxiliary_pinned"] == 41
    assert evidence["installed_assets_sha256"] == support.manifest("gen5")["assets_sha256"]
