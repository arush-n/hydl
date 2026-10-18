"""Extractors: byte-exact round trips on both installs, and synthetic-zip units."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from . import support  # noqa: F401  (puts tools/ on sys.path)
from content_extract import FAMILY_MODULES, extract_table, extract_tables
from content_extract.archive import GameArchive, parse_json_bytes
from content_extract.common import TABLE_SCHEMA, dump_table, resolve_parented
from content_extract.manifest import build_manifest, MANIFEST_NAME


@pytest.mark.parametrize("generation", ["gen5", "gen7"])
def test_checked_in_tables_round_trip_from_the_installed_archive(generation: str) -> None:
    archive_path = support.ARCHIVES[generation]
    if not archive_path.is_file():
        pytest.skip(f"{archive_path} is not installed")
    manifest = support.manifest(generation)
    sha = support.archive_sha256(str(archive_path))
    if sha != manifest["assets_sha256"]:
        pytest.skip(f"installed archive {sha[:8]} is not the extracted one {manifest['assets_sha256'][:8]}")
    with GameArchive(archive_path, assets_sha256=sha) as archive:
        assert archive.game_version == manifest["game_version"]
        tables = extract_tables(archive)
        for family, table in tables.items():
            path = support.data_dir(generation) / f"{family}.json"
            assert dump_table(table) == path.read_text(encoding="utf-8"), f"{generation}/{family} is stale"
        rebuilt = build_manifest(tables, archive)
    assert rebuilt == json.loads((support.data_dir(generation) / MANIFEST_NAME).read_text(encoding="utf-8"))


def test_extraction_is_deterministic() -> None:
    archive_path = support.ARCHIVES["gen7"]
    if not archive_path.is_file():
        pytest.skip("0.7.2 is not installed")
    with GameArchive(archive_path, assets_sha256="0" * 64) as archive:
        first = dump_table(extract_table(archive, "stats"))
        second = dump_table(extract_table(archive, "stats"))
    assert first == second


def _zip(tmp_path: Path, files: dict[str, object]) -> Path:
    path = tmp_path / "Assets.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value if isinstance(value, str) else json.dumps(value))
    return path


def _archive(tmp_path: Path, files: dict[str, object]) -> GameArchive:
    return GameArchive(_zip(tmp_path, files), game_version="9.9.9", assets_sha256="A" * 64)


def test_jsonc_fallback_and_provenance(tmp_path: Path) -> None:
    value, lenient = parse_json_bytes(b'{"a": 1, // comment\n "b": [1, 2,],}')
    assert value == {"a": 1, "b": [1, 2]} and lenient
    files = {
        "Server/Entity/MovementConfig/Default.json": '{"JumpForce": 11.8, /* c */ "BaseSpeed": 5.5,}',
        "Server/GameplayConfigs/Default.json": {"Player": {"MovementConfig": "Default"}},
    }
    with _archive(tmp_path, files) as archive:
        table = extract_table(archive, "movement")
        assert archive.lenient == {"Server/Entity/MovementConfig/Default.json"}
    assert table["schema"] == TABLE_SCHEMA and table["game_version"] == "9.9.9"
    assert table["rows"]["config:Default"]["fields"] == {"JumpForce": 11.8, "BaseSpeed": 5.5}
    assert set(table["sources"]) == set(files)


def test_parent_merge_replaces_base_damage_and_keeps_chain(tmp_path: Path) -> None:
    files = {
        "Server/Item/Interactions/A.json": {"Type": "DamageEntity", "DamageCalculator": {"BaseDamage": {"Physical": 5}, "Class": "Light"}},
        "Server/Item/Interactions/B.json": {"Parent": "A", "DamageCalculator": {"BaseDamage": {"Projectile": 3}}},
    }
    with _archive(tmp_path, files) as archive:
        index = archive.index("Server/Item/Interactions/")
        resolved, chain = resolve_parented("B", index, archive.read_json)
    assert resolved["DamageCalculator"] == {"BaseDamage": {"Projectile": 3}, "Class": "Light"}
    assert chain == ("A",)


def test_durability_follows_the_generator_rule(tmp_path: Path) -> None:
    files = {
        "Server/Item/Items/T.json": {"MaxDurability": 100, "DurabilityLossOnHit": 0.5},
        "Server/Item/Items/C.json": {"Parent": "T", "MaxDurability": 80},
        "Server/Item/Items/Z.json": {"MaxDurability": 0},
        "Server/Item/Items/N.json": {"MaxDurability": 10},
    }
    with _archive(tmp_path, files) as archive:
        rows = extract_table(archive, "durability")["rows"]
    assert rows["item:C"]["max_durability"] == 80.0 and rows["item:C"]["durability_loss_on_hit"] == 0.5
    assert rows["item:C"]["source"] == "Server/Item/Items/C.json"
    assert rows["item:N"]["durability_loss_on_hit"] == 0.0
    assert "item:Z" not in rows


def test_recipes_follow_the_server_assembly(tmp_path: Path) -> None:
    files = {
        "Server/Item/Items/Fish.json": {"Recipe": {"Input": [{"ResourceTypeId": "Fishes"}], "PrimaryOutput": {"ItemId": "Raw", "Quantity": 2}, "OutputQuantity": 2, "BenchRequirement": [{"Id": "Cook", "Type": "Crafting"}]}},
        "Server/Item/Items/Stick.json": {"Recipe": {"Input": [{"ItemId": "Wood", "Quantity": 5}], "BenchRequirement": [{"Id": "Farm"}]}},
        "Server/Item/Items/Child.json": {"Parent": "Stick"},
        "Server/Item/Recipes/Salvage_X.json": {"Input": [{"ItemId": "X"}], "PrimaryOutput": {"ItemId": "Scrap", "Quantity": 3}},
    }
    with _archive(tmp_path, files) as archive:
        rows = extract_table(archive, "crafting")["rows"]
    fish = rows["recipe:Fish_Recipe_Generated_0"]
    assert fish["outputs"] == [{"item_id": "Raw", "quantity": 2}]
    assert fish["primary_output"] == {"item_id": "Fish", "quantity": 2}
    assert fish["inputs"][0]["excluded_item_ids"] == ["Fish", "Raw"]
    stick = rows["recipe:Stick_Recipe_Generated_0"]
    assert stick["outputs"] == [{"item_id": "Stick", "quantity": 1}] and stick["bench_requirements"][0]["type"] is None
    assert "recipe:Child_Recipe_Generated_0" not in rows  # Recipe is not inherited
    assert rows["recipe:Salvage_X"]["outputs"] == [{"item_id": "Scrap", "quantity": 3}]


def test_every_family_module_declares_its_contract() -> None:
    from content_extract import family_module

    for name in FAMILY_MODULES:
        module = family_module(name)
        assert module.FAMILY == name and isinstance(module.VERSION, int) and callable(module.extract)
