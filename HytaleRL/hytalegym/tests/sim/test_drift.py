"""Drift checker (sim/common/drift.py) and the gen-7 source declarations (sim/gen7/sources.py).

The declarations are read by the checker only; they are not render inputs. The overlay identity pinned
below is the manifest's ``overlay.sha256`` of the baseline gen-7 render (2026-10-23), so a declaration
that changed the render would fail here before the identity gates.
"""

from __future__ import annotations

import os
import re
import shutil
import zipfile
from pathlib import Path

import pytest

from hytalegym.sim.common import drift
from hytalegym.sim.common.drift import (
    AssetReader,
    AssetSource,
    ClassSource,
    Derived,
    JarReader,
    Missing,
    check,
    constant_value,
    method_block,
    overlay_rows,
    parse_jsonc,
    resolve_pointer,
    summary,
)
from hytalegym.sim.gen7.overlay import OVERLAY
from hytalegym.sim.gen7.sources import DECLARATIONS

_INSTALL = Path(
    "/Users/arush/Library/Application Support/Hytale/install/release/package/game/latest"
)
ZIP = Path(os.environ.get("HYTALE_ASSETS_ZIP", _INSTALL / "Assets.zip"))
JAR = _INSTALL / "Server" / "HytaleServer.jar"
HAS_RELEASE = ZIP.is_file() and JAR.is_file() and shutil.which("javap") is not None


def _zip(tmp_path: Path, name: str, members: dict[str, str]) -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as archive:
        for member, text in members.items():
            archive.writestr(member, text)
    return path


def test_declared_sources_parse_and_cover_every_overlay_row():
    rows = overlay_rows(OVERLAY)
    row_ids = [row_id for row_id, _ in rows]
    assert len(row_ids) == len(set(row_ids)), (
        "overlay row ids are unique across redirects, patches and hooks"
    )
    assert set(row_ids) == set(DECLARATIONS), (
        "every overlay row is declared, and no declaration is stale"
    )
    for row_id, value in DECLARATIONS.items():
        parts = value if isinstance(value, tuple) else (value,)
        assert parts, row_id
        for part in parts:
            if isinstance(part, Derived):
                assert part.reason.strip(), row_id
            elif isinstance(part, AssetSource):
                assert part.path.endswith(".json"), row_id
                assert part.pointer == "" or part.pointer.startswith("/"), row_id
            elif isinstance(part, ClassSource):
                assert part.mode in ("digest", "constant"), row_id
                if part.mode == "digest":
                    assert re.fullmatch(r"[0-9a-f]{64}", part.expected), row_id
            else:
                pytest.fail(f"{row_id}: not a source declaration: {part!r}")


def test_pointer_and_jsonc_helpers():
    doc = parse_jsonc(
        '{ // a comment\n "A/B": {"~x": [1, 2,],}, /* block */ "k": "v // not a comment", }'
    )
    assert resolve_pointer(doc, "/A~1B/~0x/1") == 2
    assert resolve_pointer(doc, "") is doc
    assert doc["k"] == "v // not a comment"
    with pytest.raises(Missing):
        resolve_pointer(doc, "/A~1B/nope")
    with pytest.raises(ValueError):
        resolve_pointer(doc, "no-leading-slash")


_JAVAP_A = """\
  protected double computeMove(int, double);
    Code:
       0: aload_0
       1: getfield      #12                 // Field speed:D
       4: dreturn
"""
_JAVAP_B = _JAVAP_A.replace(
    "#12", "#97"
)  # the same bytecode with another constant-pool index
_JAVAP_C = _JAVAP_A.replace("dreturn", "dstore_1")  # a changed instruction


def test_method_digest_ignores_constant_pool_indexes_only():
    a = method_block(_JAVAP_A, "computeMove/2")
    assert a == method_block(_JAVAP_B, "computeMove/2")
    assert a != method_block(_JAVAP_C, "computeMove/2")


def test_method_lookup_needs_arity_when_overloaded():
    text = _JAVAP_A + "\n  public void computeMove(int);\n    Code:\n       0: return\n"
    with pytest.raises(Missing):
        method_block(text, "computeMove")
    assert "dreturn" in method_block(text, "computeMove/2")
    with pytest.raises(Missing):
        method_block(text, "computeMove/3")


def test_constant_value_reads_javap_constants():
    text = "  public static final double JUMP_FORWARD_DISTANCE = 0.25;\n"
    assert constant_value(text, "JUMP_FORWARD_DISTANCE") == "0.25"
    with pytest.raises(Missing):
        constant_value(text, "ABSENT")


def test_altered_fixture_is_flagged(tmp_path):
    member = "Server/Item/Interactions/Left_Damage.json"
    original = '{"DamageEffects": {"Knockback": {"Force": 4.3, "VelocityType": "Add"}}}'
    altered = original.replace("4.3", "4.5")
    declarations = {
        "knockback": (AssetSource(member, "/DamageEffects/Knockback/Force", 4.3),),
        "velocity": AssetSource(member, "/DamageEffects/Knockback/VelocityType", "Add"),
        "absent": AssetSource("Server/Gone.json", "/Damage", 16),
    }
    rows = [
        ("knockback", "patch"),
        ("velocity", "patch"),
        ("absent", "patch"),
        ("undeclared", "patch"),
    ]

    results, stale = check(
        rows, declarations, AssetReader(_zip(tmp_path, "same.zip", {member: original}))
    )
    by_id = {r.row: r for r in results}
    assert by_id["knockback"].status == "unchanged"
    assert by_id["velocity"].status == "unchanged"
    assert by_id["absent"].status == "missing"
    assert by_id["undeclared"].status == "undeclared"
    assert stale == []

    results, _ = check(
        rows,
        declarations,
        AssetReader(_zip(tmp_path, "altered.zip", {member: altered})),
    )
    knockback = {r.row: r for r in results}["knockback"]
    assert knockback.status == "changed"
    assert knockback.parts[0].expected == 4.3 and knockback.parts[0].actual == 4.5
    assert summary(results, stale)["changed"] == 1


def test_declared_row_missing_from_the_overlay_is_stale(tmp_path):
    results, stale = check([], {"gone": Derived("reason")}, None, None)
    assert results == [] and stale == ["gone"]


def test_derived_rows_are_reported_not_read(tmp_path):
    results, _ = check(
        [("step", "literal redirect")], {"step": Derived("owner ruling")}, None, None
    )
    assert results[0].status == "derived"
    assert results[0].reason == "owner ruling"


@pytest.mark.skipif(
    not HAS_RELEASE,
    reason="needs the installed 0.7.2 Assets.zip, HytaleServer.jar and javap",
)
def test_0_7_2_release_reads_unchanged_for_every_declared_row():
    rows = overlay_rows(OVERLAY)
    results, stale = check(rows, DECLARATIONS, AssetReader(ZIP), JarReader(JAR))
    assert stale == []
    bad = [
        (r.row, r.status)
        for r in results
        if r.status in ("changed", "missing", "undeclared")
    ]
    assert bad == []
    re_read = [r for r in results if r.status != "derived"]
    assert re_read and all(r.status == "unchanged" for r in re_read)


@pytest.mark.skipif(
    not HAS_RELEASE,
    reason="needs the installed 0.7.2 Assets.zip, HytaleServer.jar and javap",
)
def test_cli_exit_status_on_the_0_7_2_release(capsys):
    assert (
        drift.main(["--assets", str(ZIP), "--jar", str(JAR), "--generation", "gen7"])
        == 0
    )
    assert "summary:" in capsys.readouterr().out
