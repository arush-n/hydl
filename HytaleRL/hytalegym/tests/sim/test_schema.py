"""Content schema: the checked-in tables validate; tampering is caught."""

from __future__ import annotations

import copy
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from . import support
from hytalegym.sim.common.content import (
    FAMILIES,
    MANIFEST_SCHEMA,
    TABLE_SCHEMA,
    ContentError,
    load,
    sim_profile,
    validate_manifest,
    validate_overlay,
    validate_table,
)
from hytalegym.sim.common.content.schema import canonical_sha256


def test_schema_strings_agree_with_the_extractors() -> None:
    from content_extract import FAMILY_MODULES
    from content_extract.common import TABLE_SCHEMA as EXTRACTOR_TABLE_SCHEMA
    from content_extract.manifest import MANIFEST_SCHEMA as EXTRACTOR_MANIFEST_SCHEMA

    assert TABLE_SCHEMA == EXTRACTOR_TABLE_SCHEMA
    assert MANIFEST_SCHEMA == EXTRACTOR_MANIFEST_SCHEMA
    assert sorted(FAMILIES) == sorted(FAMILY_MODULES)


@pytest.mark.parametrize("generation", ["gen5", "gen7"])
def test_checked_in_data_validates(generation: str) -> None:
    tables = {family: support.table(generation, family) for family in FAMILIES}
    for family, table in tables.items():
        assert validate_table(table, family=family) == []
    assert validate_manifest(support.manifest(generation), tables) == []


def test_gen5_overlay_validates() -> None:
    overlay = json.loads((support.SIM / "gen5" / "calibration_057.json").read_text(encoding="utf-8"))
    assert validate_overlay(overlay) == []


def test_tampering_is_reported() -> None:
    table = support.table("gen5", "movement")
    edited = copy.deepcopy(table)
    edited["rows"]["config:Default"]["fields"]["JumpForce"] = 99
    assert any("rows_sha256" in error for error in validate_table(edited))
    missing = copy.deepcopy(table)
    del missing["rows"]["config:Default"]["fields"]
    missing["rows_sha256"] = canonical_sha256(dict(sorted(missing["rows"].items())))
    assert any("misses ['fields']" in error for error in validate_table(missing))
    unknown = copy.deepcopy(table)
    unknown["family"] = "spells"
    assert validate_table(unknown) == ["unknown family 'spells'"]
    bad_kind = copy.deepcopy(table)
    bad_kind["rows"]["rune:X"] = {}
    bad_kind["row_count"] += 1
    bad_kind["rows_sha256"] = canonical_sha256(dict(sorted(bad_kind["rows"].items())))
    assert any("unknown row kind" in error for error in validate_table(bad_kind))


def test_loader_rejects_a_tampered_directory(tmp_path: Path) -> None:
    copy_dir = tmp_path / "data"
    shutil.copytree(support.data_dir("gen5"), copy_dir)
    stats = json.loads((copy_dir / "stats.json").read_text(encoding="utf-8"))
    stats["rows"]["stat:Mana"]["max"] = 100
    (copy_dir / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    profile = replace(sim_profile("0.5.7"), data_dir=copy_dir, calibration_overlay=None, generation="tampered")
    with pytest.raises(ContentError, match="rows_sha256"):
        load(profile)
