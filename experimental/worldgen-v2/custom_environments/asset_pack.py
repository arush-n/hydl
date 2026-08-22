"""Reproducible authoring for deliberate WorldGen V2 environment packs.

The compiler does not synthesize an undocumented node format.  It follows the
same template-cloning boundary as Hytale 0.5.7's ``worldgen2 create`` command,
pins every source asset byte, and applies a small set of asserted graph
transformations.  The generated asset pack remains editable in Hytale's Node
Editor after publication.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping
import uuid
from zipfile import BadZipFile, ZipFile


ASSET_PACK_SPEC_SCHEMA = "hytalerl_worldgen_v2_custom_environment_spec_v1"
ASSET_PACK_SPEC_VERSION = 1
AUTHORING_RECEIPT_SCHEMA = "hytalerl_worldgen_v2_asset_pack_receipt_v1"
AUTHORING_RECEIPT_VERSION = 1
STRUCTURE_REGISTRY_SCHEMA = "hytalerl_worldgen_v2_structure_registry_v1"
STRUCTURE_REGISTRY_VERSION = 1

_SPEC_FIELDS = frozenset(
    {
        "schema",
        "version",
        "environment_id",
        "server_version",
        "assets_archive_sha256",
        "pack",
        "asset_ids",
        "terrain",
        "dense_trees",
        "structures",
        "source_assets",
    }
)
_PACK_FIELDS = frozenset({"group", "name", "version", "description"})
_ASSET_ID_FIELDS = frozenset(
    {
        "world_structure",
        "biome",
        "structure_assignment",
        "structure_marker",
        "structure_prefab_directory",
        "structure_prefab",
    }
)
_TERRAIN_FIELDS = frozenset(
    {"base_height", "water_height", "bedrock_height", "spawn_height"}
)
_TREE_FIELDS = frozenset({"source_assignment", "spacing"})
_STRUCTURE_FIELDS = frozenset(
    {"spacing", "jitter", "seed", "registry_asset_id"}
)
_SOURCE_NAMES = frozenset(
    {
        "mountain_biome",
        "water_biome",
        "world_structure",
        "structure_assignment",
        "structure_marker_prefab",
        "structure_prefab",
    }
)
_SOURCE_FIELDS = frozenset({"path", "sha256"})
_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "version",
        "environment_id",
        "server_version",
        "spec_file_sha256",
        "assets_archive_sha256",
        "source_assets",
        "pack_identity",
        "world_structure_asset_id",
        "biome_asset_id",
        "structure_assignment_asset_id",
        "structure_marker_asset_id",
        "structure_prefab_asset_id",
        "structure_registry_path",
        "structure_registry_semantic_sha256",
        "authoring_contract",
        "output_files",
        "pack_semantic_sha256",
        "receipt_semantic_sha256",
    }
)
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+")
_SEMVER = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class SourceAsset:
    path: str
    sha256: str


@dataclass(frozen=True)
class PackIdentity:
    group: str
    name: str
    version: str
    description: str


@dataclass(frozen=True)
class AssetIds:
    world_structure: str
    biome: str
    structure_assignment: str
    structure_marker: str
    structure_prefab_directory: str
    structure_prefab: str


@dataclass(frozen=True)
class TerrainDesign:
    base_height: int
    water_height: int
    bedrock_height: int
    spawn_height: int


@dataclass(frozen=True)
class DenseTreeDesign:
    source_assignment: str
    spacing: tuple[int, int, int]


@dataclass(frozen=True)
class StructureDesign:
    spacing: tuple[int, int, int]
    jitter: float
    seed: str
    registry_asset_id: str


@dataclass(frozen=True)
class CustomEnvironmentSpec:
    path: Path
    file_sha256: str
    environment_id: str
    server_version: str
    assets_archive_sha256: str
    pack: PackIdentity
    asset_ids: AssetIds
    terrain: TerrainDesign
    dense_trees: DenseTreeDesign
    structures: StructureDesign
    source_assets: Mapping[str, SourceAsset]


def default_hytale_assets_path() -> Path:
    """Return the explicit or standard installed Hytale ``Assets.zip``."""

    explicit = os.environ.get("HYTALE_ASSETS_ZIP")
    if explicit:
        return Path(explicit)
    app_data = os.environ.get("APPDATA")
    if not app_data:
        raise FileNotFoundError(
            "HYTALE_ASSETS_ZIP and APPDATA are unavailable; pass --assets"
        )
    return (
        Path(app_data)
        / "Hytale"
        / "install"
        / "release"
        / "package"
        / "game"
        / "latest"
        / "Assets.zip"
    )


def load_environment_spec(path: str | Path) -> CustomEnvironmentSpec:
    """Load the exact, strict custom-environment authoring contract."""

    source = Path(path).resolve()
    raw = source.read_bytes()
    value = _json_bytes(raw, "custom environment spec")
    _exact_fields(value, _SPEC_FIELDS, "custom environment spec")
    if value["schema"] != ASSET_PACK_SPEC_SCHEMA:
        raise ValueError("unsupported custom environment spec schema")
    if _integer(value["version"], "spec version") != ASSET_PACK_SPEC_VERSION:
        raise ValueError("unsupported custom environment spec version")

    pack_value = _object(value["pack"], "pack")
    _exact_fields(pack_value, _PACK_FIELDS, "pack")
    asset_value = _object(value["asset_ids"], "asset_ids")
    _exact_fields(asset_value, _ASSET_ID_FIELDS, "asset_ids")
    terrain_value = _object(value["terrain"], "terrain")
    _exact_fields(terrain_value, _TERRAIN_FIELDS, "terrain")
    tree_value = _object(value["dense_trees"], "dense_trees")
    _exact_fields(tree_value, _TREE_FIELDS, "dense_trees")
    structure_value = _object(value["structures"], "structures")
    _exact_fields(structure_value, _STRUCTURE_FIELDS, "structures")
    sources_value = _object(value["source_assets"], "source_assets")
    if set(sources_value) != _SOURCE_NAMES:
        raise ValueError(
            "source_assets must contain exactly " + ", ".join(sorted(_SOURCE_NAMES))
        )

    sources: dict[str, SourceAsset] = {}
    for name in sorted(sources_value):
        row = _object(sources_value[name], f"source_assets.{name}")
        _exact_fields(row, _SOURCE_FIELDS, f"source_assets.{name}")
        sources[name] = SourceAsset(
            path=_asset_path(row["path"], f"source_assets.{name}.path"),
            sha256=_sha256(row["sha256"], f"source_assets.{name}.sha256"),
        )

    pack = PackIdentity(
        group=_safe_id(pack_value["group"], "pack.group"),
        name=_safe_id(pack_value["name"], "pack.name"),
        version=_semver(pack_value["version"], "pack.version"),
        description=_text(pack_value["description"], "pack.description"),
    )
    asset_ids = AssetIds(
        world_structure=_safe_id(
            asset_value["world_structure"], "asset_ids.world_structure"
        ),
        biome=_safe_id(asset_value["biome"], "asset_ids.biome"),
        structure_assignment=_safe_id(
            asset_value["structure_assignment"],
            "asset_ids.structure_assignment",
        ),
        structure_marker=_safe_id(
            asset_value["structure_marker"], "asset_ids.structure_marker"
        ),
        structure_prefab_directory=_relative_directory(
            asset_value["structure_prefab_directory"],
            "asset_ids.structure_prefab_directory",
        ),
        structure_prefab=_safe_id(
            asset_value["structure_prefab"], "asset_ids.structure_prefab"
        ),
    )
    terrain = TerrainDesign(
        base_height=_bounded_int(terrain_value["base_height"], 1, 319, "base_height"),
        water_height=_bounded_int(
            terrain_value["water_height"], 1, 319, "water_height"
        ),
        bedrock_height=_bounded_int(
            terrain_value["bedrock_height"], 0, 318, "bedrock_height"
        ),
        spawn_height=_bounded_int(
            terrain_value["spawn_height"], 1, 319, "spawn_height"
        ),
    )
    if not (
        terrain.bedrock_height < terrain.water_height < terrain.spawn_height
        and terrain.bedrock_height < terrain.base_height < terrain.spawn_height
    ):
        raise ValueError("terrain heights must order bedrock below base/water below spawn")
    trees = DenseTreeDesign(
        source_assignment=_safe_id(
            tree_value["source_assignment"], "dense_trees.source_assignment"
        ),
        spacing=_positive_int3(tree_value["spacing"], "dense_trees.spacing"),
    )
    structures = StructureDesign(
        spacing=_positive_int3(structure_value["spacing"], "structures.spacing"),
        jitter=_bounded_float(structure_value["jitter"], 0.0, 1.0, "structures.jitter"),
        seed=_text(structure_value["seed"], "structures.seed"),
        registry_asset_id=_text(
            structure_value["registry_asset_id"], "structures.registry_asset_id"
        ),
    )
    return CustomEnvironmentSpec(
        path=source,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        environment_id=_safe_id(value["environment_id"], "environment_id"),
        server_version=_semver(value["server_version"], "server_version"),
        assets_archive_sha256=_sha256(
            value["assets_archive_sha256"], "assets_archive_sha256"
        ),
        pack=pack,
        asset_ids=asset_ids,
        terrain=terrain,
        dense_trees=trees,
        structures=structures,
        source_assets=sources,
    )


def build_asset_pack(
    spec: CustomEnvironmentSpec,
    assets_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Build a complete deterministic pack, refusing to replace differing output."""

    if not isinstance(spec, CustomEnvironmentSpec):
        raise TypeError("spec must be a CustomEnvironmentSpec")
    assets = Path(assets_path).resolve()
    if not assets.is_file():
        raise FileNotFoundError(f"Hytale Assets.zip not found: {assets}")
    archive_sha256 = _file_sha256(assets)
    if archive_sha256 != spec.assets_archive_sha256:
        raise ValueError(
            "installed Assets.zip differs from the pinned authoring source: "
            f"observed={archive_sha256} expected={spec.assets_archive_sha256}"
        )

    destination = _safe_destination(output)
    if destination.exists():
        report = validate_asset_pack(destination, spec)
        return {**report, "build_action": "reused_byte_identical_output"}
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with ZipFile(assets) as archive:
            source_values, source_rows = _load_source_assets(archive, spec)
    except BadZipFile as error:
        raise ValueError("Hytale Assets.zip is not a valid ZIP") from error

    world_structure = _author_world_structure(
        source_values["world_structure"], spec
    )
    biome = _author_biome(
        source_values["mountain_biome"],
        source_values["water_biome"],
        spec,
    )
    assignment = _author_structure_assignment(
        source_values["structure_assignment"], spec
    )
    prefab, registry_entry = _author_structure_prefab(
        source_values["structure_prefab"],
        source_values["structure_marker_prefab"],
        spec,
    )
    marker = _author_structure_marker(spec)
    manifest = _pack_manifest(spec)
    registry = _structure_registry(registry_entry)

    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    ).resolve()
    staging = staging_parent / destination.name
    try:
        files = _output_values(
            spec,
            manifest=manifest,
            world_structure=world_structure,
            biome=biome,
            assignment=assignment,
            prefab=prefab,
            marker=marker,
            registry=registry,
        )
        for relative, value in files.items():
            _write_json(staging / relative, value)
        output_hashes = {
            relative: _file_sha256(staging / relative)
            for relative in sorted(files)
        }
        pack_semantic = _path_hash_semantic(output_hashes)
        receipt = _authoring_receipt(
            spec,
            source_rows=source_rows,
            output_hashes=output_hashes,
            registry=registry,
            pack_semantic_sha256=pack_semantic,
        )
        _write_json(staging / "authoring-receipt.json", receipt)
        if destination.exists():
            raise FileExistsError(
                f"asset-pack output appeared during build: {destination}"
            )
        os.replace(staging, destination)
    finally:
        if staging_parent.exists():
            shutil.rmtree(staging_parent)

    report = validate_asset_pack(destination, spec)
    return {**report, "build_action": "created"}


def validate_asset_pack(
    root: str | Path,
    spec: CustomEnvironmentSpec,
) -> dict[str, Any]:
    """Validate authored intent, identities, hashes, and structure observability."""

    if not isinstance(spec, CustomEnvironmentSpec):
        raise TypeError("spec must be a CustomEnvironmentSpec")
    pack_root = Path(root).resolve()
    if not pack_root.is_dir():
        raise FileNotFoundError(f"custom environment pack not found: {pack_root}")
    expected = set(_output_paths(spec)) | {"authoring-receipt.json"}
    actual = {
        path.relative_to(pack_root).as_posix()
        for path in pack_root.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise ValueError(
            "asset-pack file set differs: "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )

    receipt = _read_json(pack_root / "authoring-receipt.json", "authoring receipt")
    _exact_fields(receipt, _RECEIPT_FIELDS, "authoring receipt")
    if receipt["schema"] != AUTHORING_RECEIPT_SCHEMA:
        raise ValueError("unsupported authoring receipt schema")
    if _integer(receipt["version"], "receipt version") != AUTHORING_RECEIPT_VERSION:
        raise ValueError("unsupported authoring receipt version")
    if receipt["environment_id"] != spec.environment_id:
        raise ValueError("authoring receipt names another environment")
    if receipt["server_version"] != spec.server_version:
        raise ValueError("authoring receipt names another server version")
    if receipt["spec_file_sha256"] != spec.file_sha256:
        raise ValueError("authoring receipt spec SHA-256 changed")
    if receipt["assets_archive_sha256"] != spec.assets_archive_sha256:
        raise ValueError("authoring receipt source archive changed")
    expected_receipt_semantic = _semantic_digest(
        receipt, "receipt_semantic_sha256"
    )
    if receipt["receipt_semantic_sha256"] != expected_receipt_semantic:
        raise ValueError("authoring receipt semantic SHA-256 mismatch")

    output_hashes = _object(receipt["output_files"], "receipt.output_files")
    if set(output_hashes) != expected - {"authoring-receipt.json"}:
        raise ValueError("authoring receipt output file set differs")
    for relative, expected_sha in output_hashes.items():
        digest = _sha256(expected_sha, f"output_files.{relative}")
        if _file_sha256(pack_root / relative) != digest:
            raise ValueError(f"authored output file SHA-256 changed: {relative}")
    if receipt["pack_semantic_sha256"] != _path_hash_semantic(output_hashes):
        raise ValueError("authored pack semantic SHA-256 mismatch")

    manifest = _read_json(pack_root / "manifest.json", "pack manifest")
    _validate_manifest(manifest, spec)
    structure = _read_json(
        pack_root / _world_structure_path(spec), "world structure"
    )
    _validate_world_structure(structure, spec)
    biome = _read_json(pack_root / _biome_path(spec), "custom biome")
    _validate_biome(biome, spec)
    assignment = _read_json(
        pack_root / _assignment_path(spec), "structure assignment"
    )
    _validate_structure_assignment(assignment, spec)
    prefab = _read_json(pack_root / _prefab_path(spec), "structure prefab")
    prefab_contract = _validate_structure_prefab(prefab, spec)
    marker = _read_json(pack_root / _marker_path(spec), "structure marker")
    _validate_structure_marker(marker, spec)
    registry = _read_json(
        pack_root / "structure-registry.registry", "structure registry"
    )
    _validate_structure_registry(registry, spec, prefab_contract)
    if receipt["structure_registry_semantic_sha256"] != registry[
        "registry_semantic_sha256"
    ]:
        raise ValueError("receipt and structure registry semantic differ")

    return {
        "status": "passed",
        "environment_id": spec.environment_id,
        "pack_root": str(pack_root),
        "pack_identity": f"{spec.pack.group}:{spec.pack.name}",
        "world_structure_asset_id": spec.asset_ids.world_structure,
        "biome_asset_id": spec.asset_ids.biome,
        "structure_assignment_asset_id": spec.asset_ids.structure_assignment,
        "structure_marker_asset_id": spec.asset_ids.structure_marker,
        "structure_prefab_asset_id": spec.structures.registry_asset_id,
        "custom_prefab_block_count": len(prefab["blocks"]),
        "custom_prefab_marker_count": 1,
        "tree_spacing": list(spec.dense_trees.spacing),
        "structure_spacing": list(spec.structures.spacing),
        "base_height": spec.terrain.base_height,
        "water_height": spec.terrain.water_height,
        "pack_semantic_sha256": receipt["pack_semantic_sha256"],
        "receipt_semantic_sha256": receipt["receipt_semantic_sha256"],
    }


def _load_source_assets(
    archive: ZipFile,
    spec: CustomEnvironmentSpec,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    values: dict[str, dict[str, Any]] = {}
    rows: dict[str, dict[str, Any]] = {}
    names = set(archive.namelist())
    for name in sorted(spec.source_assets):
        source = spec.source_assets[name]
        if source.path not in names:
            raise ValueError(f"pinned source asset is missing: {source.path}")
        raw = archive.read(source.path)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != source.sha256:
            raise ValueError(
                f"pinned source asset changed: {source.path}; "
                f"observed={digest} expected={source.sha256}"
            )
        values[name] = _json_bytes(raw, f"source asset {source.path}")
        rows[name] = {
            "path": source.path,
            "sha256": digest,
            "byte_count": len(raw),
        }
    return values, rows


def _author_world_structure(
    source: Mapping[str, Any], spec: CustomEnvironmentSpec
) -> dict[str, Any]:
    value = deepcopy(dict(source))
    if value.get("Type") != "NoiseRange":
        raise ValueError("world-structure template is not NoiseRange")
    value["Biomes"] = []
    value["DefaultBiome"] = spec.asset_ids.biome
    constants = _framework_entries(value, "DecimalConstants")
    expected = {"Base", "Water", "Bedrock"}
    by_name = {row.get("Name"): row for row in constants}
    if set(by_name) != expected:
        raise ValueError("world-structure template constants changed")
    by_name["Base"]["Value"] = spec.terrain.base_height
    by_name["Water"]["Value"] = spec.terrain.water_height
    by_name["Bedrock"]["Value"] = spec.terrain.bedrock_height
    positions = _framework_entries(value, "Positions")
    if len(positions) != 1:
        raise ValueError("world-structure template spawn group changed")
    raw_positions = positions[0].get("Positions", {}).get("Positions")
    if not isinstance(raw_positions, list) or not raw_positions:
        raise ValueError("world-structure template has no spawn positions")
    for position in raw_positions:
        if not isinstance(position, dict):
            raise ValueError("world-structure spawn position is not an object")
        position["Y"] = spec.terrain.spawn_height
    return value


def _author_biome(
    mountain_source: Mapping[str, Any],
    water_source: Mapping[str, Any],
    spec: CustomEnvironmentSpec,
) -> dict[str, Any]:
    value = deepcopy(dict(mountain_source))
    water = deepcopy(dict(water_source))
    if not isinstance(value.get("MaterialProvider"), dict):
        raise ValueError("mountain biome lacks MaterialProvider")
    water_empty = water.get("MaterialProvider", {}).get("Empty")
    if not isinstance(water_empty, dict):
        raise ValueError("water biome lacks an empty-cell material provider")
    if not _contains_pair(water_empty, "Fluid", "Water_Source"):
        raise ValueError("water biome no longer supplies Water_Source")
    value["Name"] = spec.asset_ids.biome
    value["MaterialProvider"]["Empty"] = water_empty

    tree_rows = [
        row
        for row in value.get("Props", [])
        if isinstance(row, dict)
        and row.get("Assignments", {}).get("Type") == "Imported"
        and row.get("Assignments", {}).get("Name")
        == spec.dense_trees.source_assignment
    ]
    if len(tree_rows) != 1:
        raise ValueError("mountain biome tree assignment changed")
    generator = tree_rows[0].get("Positions", {}).get("PointGenerator")
    if not isinstance(generator, dict) or generator.get("Type") != "Mesh":
        raise ValueError("mountain biome tree positions are no longer a mesh")
    original = tuple(generator.get(name) for name in ("ScaleX", "ScaleY", "ScaleZ"))
    if original != (15, 15, 15):
        raise ValueError(f"mountain biome source tree spacing changed: {original}")
    for name, spacing in zip(("ScaleX", "ScaleY", "ScaleZ"), spec.dense_trees.spacing):
        generator[name] = spacing

    runtime = _structure_runtime(spec)
    props = value.get("Props")
    if not isinstance(props, list):
        raise ValueError("mountain biome Props is not a list")
    props.append(runtime)
    metadata = value.setdefault("$NodeEditorMetadata", {})
    nodes = metadata.setdefault("$Nodes", {})
    if not isinstance(nodes, dict):
        raise ValueError("mountain biome Node Editor metadata changed")
    for index, node in enumerate(_walk_nodes(runtime)):
        node_id = node.get("$NodeId")
        if isinstance(node_id, str):
            nodes[node_id] = {"$Position": {"$x": -7000 + index * 420, "$y": 20000}}
    _require_unique_node_ids(value, "custom biome")
    return value


def _structure_runtime(spec: CustomEnvironmentSpec) -> dict[str, Any]:
    sx, sy, sz = spec.structures.spacing
    return {
        "$NodeId": _node_id(spec, "Runtime", "structure-runtime"),
        "Skip": False,
        "Runtime": 0,
        "Positions": {
            "$NodeId": _node_id(spec, "Mesh2DPositions", "structure-positions"),
            "Type": "Mesh2D",
            "Skip": False,
            "PointsY": 0,
            "PointGenerator": {
                "$NodeId": _node_id(
                    spec, "MeshPointGenerator", "structure-point-generator"
                ),
                "Type": "Mesh",
                "Jitter": spec.structures.jitter,
                "ScaleX": sx,
                "ScaleY": sy,
                "ScaleZ": sz,
                "Seed": spec.structures.seed,
            },
        },
        "Assignments": {
            "$NodeId": _node_id(
                spec, "Imported.Assignments", "structure-assignment"
            ),
            "Type": "Imported",
            "Name": spec.asset_ids.structure_assignment,
        },
    }


def _author_structure_assignment(
    source: Mapping[str, Any], spec: CustomEnvironmentSpec
) -> dict[str, Any]:
    source_prefabs = [
        node
        for node in _walk_nodes(source)
        if node.get("Type") == "Prefab" and "WeightedPrefabPaths" in node
    ]
    if len(source_prefabs) != 2:
        raise ValueError("structure assignment prefab source changed")
    matching = [
        node
        for node in source_prefabs
        if any(
            row.get("Path") == "Trees/Beech/Stage_1"
            for row in node.get("WeightedPrefabPaths", [])
            if isinstance(row, dict)
        )
    ]
    if len(matching) != 1:
        raise ValueError("structure placement template changed")
    prefab = deepcopy(matching[0])
    scanner = prefab.get("Scanner")
    if (
        not isinstance(scanner, dict)
        or scanner.get("Type") != "ColumnLinear"
        or scanner.get("BaseHeightName") != "Base"
        or scanner.get("TopDownOrder") is not True
        or scanner.get("ResultCap") != 1
        or not _contains_pair(prefab.get("Directionality", {}), "Solid", "Soil_Grass")
    ):
        raise ValueError("structure surface-placement template changed")
    prefab["WeightedPrefabPaths"] = [
        {
            "$NodeId": _node_id(spec, "WeightedPath.Prefab.Prop", "prefab-path"),
            "Path": spec.asset_ids.structure_prefab_directory + "/",
            "Weight": 100,
        }
    ]
    prefab["LegacyPath"] = False
    prefab["LoadEntities"] = True
    value = {
        "$NodeId": _node_id(spec, "Constant.Assignments", "structure-root"),
        "Type": "Constant",
        "ExportAs": spec.asset_ids.structure_assignment,
        "Prop": prefab,
        "$NodeEditorMetadata": {
            "$Nodes": {},
            "$FloatingNodes": [],
            "$Links": {},
            "$Groups": [],
            "$Comments": [],
            "$WorkspaceID": "HytaleGenerator - Assignments",
        },
    }
    nodes = value["$NodeEditorMetadata"]["$Nodes"]
    for index, node in enumerate(_walk_nodes(value)):
        node_id = node.get("$NodeId")
        if isinstance(node_id, str):
            nodes[node_id] = {
                "$Position": {"$x": 800 + index * 360, "$y": 2400}
            }
    _require_unique_node_ids(value, "custom structure assignment")
    return value


def _author_structure_prefab(
    source: Mapping[str, Any],
    marker_source: Mapping[str, Any],
    spec: CustomEnvironmentSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = deepcopy(dict(source))
    blocks = value.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("structure prefab source has no blocks")
    anchor = tuple(
        _integer(value.get(name), f"prefab {name}")
        for name in ("anchorX", "anchorY", "anchorZ")
    )
    occupied = {
        (
            _integer(row.get("x"), "prefab block x"),
            _integer(row.get("y"), "prefab block y"),
            _integer(row.get("z"), "prefab block z"),
        )
        for row in blocks
        if isinstance(row, dict)
    }
    if len(occupied) != len(blocks):
        raise ValueError("structure prefab source has malformed or duplicate blocks")
    maximum_y = max(position[1] for position in occupied)
    beacon_base = maximum_y + 1
    custom_positions = {
        (anchor[0] + dx, beacon_base, anchor[2] + dz)
        for dx in range(-1, 2)
        for dz in range(-1, 2)
    }
    custom_positions.update(
        (anchor[0], beacon_base + offset, anchor[2]) for offset in range(1, 5)
    )
    if occupied.intersection(custom_positions):
        raise ValueError("custom summit beacon overlaps the source prefab")
    blocks.extend(
        {"x": x, "y": y, "z": z, "name": "Rock_Quartzite"}
        for x, y, z in sorted(custom_positions)
    )
    entities = value.setdefault("entities", [])
    if not isinstance(entities, list):
        raise ValueError("structure prefab entities is not a list")
    marker_entities = [
        row
        for row in marker_source.get("entities", [])
        if isinstance(row, dict)
        and isinstance(row.get("Components"), dict)
        and "SpawnMarkerComponent" in row["Components"]
    ]
    if len(marker_entities) != 1:
        raise ValueError("pinned marker prefab entity template changed")
    marker_entity = deepcopy(marker_entities[0])
    components = marker_entity["Components"]
    required_components = {
        "UUID",
        "Transform",
        "HeadRotation",
        "Model",
        "SpawnMarkerComponent",
    }
    if not required_components.issubset(components):
        raise ValueError("pinned marker prefab lacks native entity components")
    components["SpawnMarkerComponent"] = {
        "SpawnMarker": spec.asset_ids.structure_marker,
        "RespawnTime": 0.0,
        "SpawnCount": 0,
        "NPCReferences": [],
    }
    components["Transform"] = {
        "Position": {
            "X": anchor[0] + 0.5,
            "Y": float(anchor[1]),
            "Z": anchor[2] + 0.5,
        },
        "Rotation": {"Pitch": 0.0, "Roll": 0.0, "Yaw": 0.0},
    }
    components["HeadRotation"]["Rotation"] = {
        "Pitch": 0.0,
        "Roll": 0.0,
        "Yaw": 0.0,
    }
    if isinstance(components.get("Nameplate"), dict):
        components["Nameplate"]["Text"] = spec.asset_ids.structure_marker
    entities.append(marker_entity)
    bounds_min, bounds_max = _prefab_local_bounds(value)
    registry_entry = {
        "marker_asset_id": spec.asset_ids.structure_marker,
        "structure_asset_id": spec.structures.registry_asset_id,
        "kind": "prefab",
        "marker_local_position": [0.5, 0.0, 0.5],
        "local_bounds_min": list(bounds_min),
        "local_bounds_max": list(bounds_max),
        "geometry_present": True,
    }
    return value, registry_entry


def _author_structure_marker(spec: CustomEnvironmentSpec) -> dict[str, Any]:
    return {
        "Model": "NPC_Spawn_Marker",
        "NPCs": [
            {
                "Name": "Test_Dummy_Damage",
                "Weight": 1,
                "SpawnAfterGameTime": "P1000000D",
            }
        ],
        "ManualTrigger": True,
        "ExclusionRadius": 0,
        "MaxDropHeight": 1,
    }


def _pack_manifest(spec: CustomEnvironmentSpec) -> dict[str, Any]:
    return {
        "Group": spec.pack.group,
        "Name": spec.pack.name,
        "Version": spec.pack.version,
        "Description": spec.pack.description,
        "Authors": [{"Name": "HytaleRL Contributors"}],
        "ServerVersion": f"={spec.server_version}",
        "Dependencies": {},
        "OptionalDependencies": {},
        "LoadBefore": {},
        "DisabledByDefault": False,
        # Standalone packs are registered directly by AssetModule.  This flag
        # is for a Java plugin JAR that embeds an additional asset pack.
        "IncludesAssetPack": False,
    }


def _structure_registry(entry: Mapping[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": STRUCTURE_REGISTRY_SCHEMA,
        "version": STRUCTURE_REGISTRY_VERSION,
        "entries": [deepcopy(dict(entry))],
        "registry_semantic_sha256": "",
    }
    value["registry_semantic_sha256"] = _semantic_digest(
        value, "registry_semantic_sha256"
    )
    return value


def _output_values(
    spec: CustomEnvironmentSpec,
    *,
    manifest: Mapping[str, Any],
    world_structure: Mapping[str, Any],
    biome: Mapping[str, Any],
    assignment: Mapping[str, Any],
    prefab: Mapping[str, Any],
    marker: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        "manifest.json": manifest,
        _world_structure_path(spec): world_structure,
        _biome_path(spec): biome,
        _assignment_path(spec): assignment,
        _prefab_path(spec): prefab,
        _marker_path(spec): marker,
        "structure-registry.registry": registry,
    }


def _authoring_receipt(
    spec: CustomEnvironmentSpec,
    *,
    source_rows: Mapping[str, Mapping[str, Any]],
    output_hashes: Mapping[str, str],
    registry: Mapping[str, Any],
    pack_semantic_sha256: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": AUTHORING_RECEIPT_SCHEMA,
        "version": AUTHORING_RECEIPT_VERSION,
        "environment_id": spec.environment_id,
        "server_version": spec.server_version,
        "spec_file_sha256": spec.file_sha256,
        "assets_archive_sha256": spec.assets_archive_sha256,
        "source_assets": deepcopy(dict(source_rows)),
        "pack_identity": {
            "group": spec.pack.group,
            "name": spec.pack.name,
            "version": spec.pack.version,
        },
        "world_structure_asset_id": spec.asset_ids.world_structure,
        "biome_asset_id": spec.asset_ids.biome,
        "structure_assignment_asset_id": spec.asset_ids.structure_assignment,
        "structure_marker_asset_id": spec.asset_ids.structure_marker,
        "structure_prefab_asset_id": spec.structures.registry_asset_id,
        "structure_registry_path": "structure-registry.registry",
        "structure_registry_semantic_sha256": registry[
            "registry_semantic_sha256"
        ],
        "authoring_contract": {
            "terrain": "pinned_mountain_graph_with_project_water_provider",
            "tree_density": "pinned_mesh_spacing_override",
            "structures": (
                "project_prefab_with_pinned_mountain_surface_scanner_and_"
                "manual_marker"
            ),
            "legacy_world_identity_changed": False,
            "jax_generation_required": False,
        },
        "output_files": dict(sorted(output_hashes.items())),
        "pack_semantic_sha256": pack_semantic_sha256,
        "receipt_semantic_sha256": "",
    }
    value["receipt_semantic_sha256"] = _semantic_digest(
        value, "receipt_semantic_sha256"
    )
    return value


def _validate_manifest(value: Mapping[str, Any], spec: CustomEnvironmentSpec) -> None:
    expected = _pack_manifest(spec)
    if value != expected:
        raise ValueError("custom asset-pack manifest differs from authored contract")


def _validate_world_structure(
    value: Mapping[str, Any], spec: CustomEnvironmentSpec
) -> None:
    if value.get("DefaultBiome") != spec.asset_ids.biome:
        raise ValueError("world structure does not select the custom biome")
    if value.get("Biomes") != []:
        raise ValueError("custom world structure must use one deliberate default biome")
    constants = {
        row.get("Name"): row.get("Value")
        for row in _framework_entries(value, "DecimalConstants")
    }
    expected = {
        "Base": spec.terrain.base_height,
        "Water": spec.terrain.water_height,
        "Bedrock": spec.terrain.bedrock_height,
    }
    if constants != expected:
        raise ValueError("custom world-structure terrain constants differ")
    positions = _framework_entries(value, "Positions")
    raw = positions[0].get("Positions", {}).get("Positions") if positions else None
    if not isinstance(raw, list) or not raw or any(
        row.get("Y") != spec.terrain.spawn_height for row in raw
    ):
        raise ValueError("custom world-structure spawn height differs")


def _validate_biome(value: Mapping[str, Any], spec: CustomEnvironmentSpec) -> None:
    if value.get("Name") != spec.asset_ids.biome:
        raise ValueError("custom biome name differs")
    empty = value.get("MaterialProvider", {}).get("Empty")
    if not isinstance(empty, dict) or not _contains_pair(
        empty, "Fluid", "Water_Source"
    ):
        raise ValueError("custom biome does not fill low empty cells with water")
    tree_rows = [
        row
        for row in value.get("Props", [])
        if isinstance(row, dict)
        and row.get("Assignments", {}).get("Name")
        == spec.dense_trees.source_assignment
    ]
    if len(tree_rows) != 1:
        raise ValueError("custom biome tree runtime differs")
    point = tree_rows[0].get("Positions", {}).get("PointGenerator", {})
    spacing = tuple(point.get(name) for name in ("ScaleX", "ScaleY", "ScaleZ"))
    if spacing != spec.dense_trees.spacing:
        raise ValueError("custom biome tree spacing differs")
    structure_rows = [
        row
        for row in value.get("Props", [])
        if isinstance(row, dict)
        and row.get("Assignments", {}).get("Name")
        == spec.asset_ids.structure_assignment
    ]
    if len(structure_rows) != 1:
        raise ValueError("custom biome structure runtime differs")
    point = structure_rows[0].get("Positions", {}).get("PointGenerator", {})
    spacing = tuple(point.get(name) for name in ("ScaleX", "ScaleY", "ScaleZ"))
    if spacing != spec.structures.spacing:
        raise ValueError("custom biome structure spacing differs")
    if point.get("Jitter") != spec.structures.jitter:
        raise ValueError("custom biome structure jitter differs")
    if point.get("Seed") != spec.structures.seed:
        raise ValueError("custom biome structure seed differs")
    _require_unique_node_ids(value, "custom biome")


def _validate_structure_assignment(
    value: Mapping[str, Any], spec: CustomEnvironmentSpec
) -> None:
    if value.get("ExportAs") != spec.asset_ids.structure_assignment:
        raise ValueError("custom structure assignment export differs")
    prefabs = [
        node
        for node in _walk_nodes(value)
        if node.get("Type") == "Prefab" and "WeightedPrefabPaths" in node
    ]
    if len(prefabs) != 1 or prefabs[0].get("LoadEntities") is not True:
        raise ValueError("custom structure assignment must load marker entities")
    paths = prefabs[0].get("WeightedPrefabPaths")
    expected = spec.asset_ids.structure_prefab_directory + "/"
    if not isinstance(paths, list) or len(paths) != 1 or paths[0].get("Path") != expected:
        raise ValueError("custom structure assignment prefab path differs")
    if paths[0].get("Weight") != 100:
        raise ValueError("custom structure assignment prefab weight differs")
    scanner = prefabs[0].get("Scanner")
    if (
        not isinstance(scanner, dict)
        or scanner.get("Type") != "ColumnLinear"
        or scanner.get("BaseHeightName") != "Base"
        or scanner.get("TopDownOrder") is not True
        or scanner.get("ResultCap") != 1
    ):
        raise ValueError("custom structure surface scanner differs")
    if not _contains_pair(
        prefabs[0].get("Directionality", {}), "Solid", "Soil_Grass"
    ):
        raise ValueError("custom structure floor pattern differs")
    _require_unique_node_ids(value, "custom structure assignment")


def _validate_structure_prefab(
    value: Mapping[str, Any], spec: CustomEnvironmentSpec
) -> dict[str, Any]:
    if value.get("version") != 8 or value.get("blockIdVersion") != 11:
        raise ValueError("custom structure prefab version differs")
    blocks = value.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("custom structure prefab has no geometry")
    positions = [
        (row.get("x"), row.get("y"), row.get("z"))
        for row in blocks
        if isinstance(row, dict)
    ]
    if len(positions) != len(blocks) or len(set(positions)) != len(positions):
        raise ValueError("custom structure prefab has malformed or duplicate cells")
    quartzite = sum(
        row.get("name") == "Rock_Quartzite" for row in blocks if isinstance(row, dict)
    )
    if quartzite < 13:
        raise ValueError("custom structure prefab lacks its authored summit beacon")
    entities = value.get("entities")
    if not isinstance(entities, list):
        raise ValueError("custom structure prefab has no entity list")
    markers = [
        row
        for row in entities
        if isinstance(row, dict)
        and row.get("Components", {})
        .get("SpawnMarkerComponent", {})
        .get("SpawnMarker")
        == spec.asset_ids.structure_marker
    ]
    if len(markers) != 1:
        raise ValueError("custom structure prefab requires exactly one authored marker")
    anchor = tuple(value[name] for name in ("anchorX", "anchorY", "anchorZ"))
    expected_position = {
        "X": anchor[0] + 0.5,
        "Y": float(anchor[1]),
        "Z": anchor[2] + 0.5,
    }
    transform = markers[0]["Components"].get("Transform", {})
    if transform.get("Position") != expected_position:
        raise ValueError("custom structure marker is not at the prefab anchor")
    if transform.get("Rotation") != {"Pitch": 0.0, "Roll": 0.0, "Yaw": 0.0}:
        raise ValueError("custom structure marker rotation differs")
    bounds_min, bounds_max = _prefab_local_bounds(value)
    return {
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "marker_local_position": (0.5, 0.0, 0.5),
    }


def _validate_structure_marker(
    value: Mapping[str, Any], spec: CustomEnvironmentSpec
) -> None:
    del spec
    if value.get("Model") != "NPC_Spawn_Marker":
        raise ValueError("custom structure marker model differs")
    if value.get("ManualTrigger") is not True:
        raise ValueError("custom structure marker must be inert/manual")
    choices = value.get("NPCs")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("custom structure marker NPC configuration differs")
    if choices[0].get("SpawnAfterGameTime") != "P1000000D":
        raise ValueError("custom structure marker delay differs")


def _validate_structure_registry(
    value: Mapping[str, Any],
    spec: CustomEnvironmentSpec,
    prefab_contract: Mapping[str, Any],
) -> None:
    expected_fields = {"schema", "version", "entries", "registry_semantic_sha256"}
    _exact_fields(value, expected_fields, "structure registry")
    if value["schema"] != STRUCTURE_REGISTRY_SCHEMA or value["version"] != 1:
        raise ValueError("custom structure registry schema differs")
    if value["registry_semantic_sha256"] != _semantic_digest(
        value, "registry_semantic_sha256"
    ):
        raise ValueError("custom structure registry semantic SHA-256 mismatch")
    rows = value["entries"]
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("custom structure registry must contain one entry")
    row = rows[0]
    expected = {
        "marker_asset_id": spec.asset_ids.structure_marker,
        "structure_asset_id": spec.structures.registry_asset_id,
        "kind": "prefab",
        "marker_local_position": list(prefab_contract["marker_local_position"]),
        "local_bounds_min": list(prefab_contract["bounds_min"]),
        "local_bounds_max": list(prefab_contract["bounds_max"]),
        "geometry_present": True,
    }
    if row != expected:
        raise ValueError("custom structure registry entry differs from prefab")


def _world_structure_path(spec: CustomEnvironmentSpec) -> str:
    return f"Server/HytaleGenerator/WorldStructures/{spec.asset_ids.world_structure}.json"


def _biome_path(spec: CustomEnvironmentSpec) -> str:
    return f"Server/HytaleGenerator/Biomes/Hydl/{spec.asset_ids.biome}.json"


def _assignment_path(spec: CustomEnvironmentSpec) -> str:
    return (
        "Server/HytaleGenerator/Assignments/Hydl/"
        f"{spec.asset_ids.structure_assignment}.json"
    )


def _prefab_path(spec: CustomEnvironmentSpec) -> str:
    return (
        f"Server/Prefabs/{spec.asset_ids.structure_prefab_directory}/"
        f"{spec.asset_ids.structure_prefab}.prefab.json"
    )


def _marker_path(spec: CustomEnvironmentSpec) -> str:
    return f"Server/NPC/Spawn/Markers/{spec.asset_ids.structure_marker}.json"


def _output_paths(spec: CustomEnvironmentSpec) -> tuple[str, ...]:
    return (
        "manifest.json",
        _world_structure_path(spec),
        _biome_path(spec),
        _assignment_path(spec),
        _prefab_path(spec),
        _marker_path(spec),
        "structure-registry.registry",
    )


def _prefab_local_bounds(value: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    anchor = tuple(
        _integer(value.get(name), f"prefab {name}")
        for name in ("anchorX", "anchorY", "anchorZ")
    )
    blocks = value.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("prefab has no blocks for bounds")
    positions = [
        tuple(_integer(row.get(axis), f"prefab block {axis}") for axis in "xyz")
        for row in blocks
    ]
    minimum = tuple(min(row[index] for row in positions) - anchor[index] for index in range(3))
    maximum = tuple(max(row[index] for row in positions) - anchor[index] + 1 for index in range(3))
    return minimum, maximum


def _framework_entries(value: Mapping[str, Any], kind: str) -> list[dict[str, Any]]:
    framework = value.get("Framework")
    if not isinstance(framework, list):
        raise ValueError("world structure Framework is not a list")
    groups = [row for row in framework if isinstance(row, dict) and row.get("Type") == kind]
    if len(groups) != 1 or not isinstance(groups[0].get("Entries"), list):
        raise ValueError(f"world structure requires one {kind} framework group")
    return groups[0]["Entries"]


def _walk_nodes(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_nodes(child)


def _require_unique_node_ids(value: Any, label: str) -> None:
    identifiers = [
        node["$NodeId"]
        for node in _walk_nodes(value)
        if isinstance(node.get("$NodeId"), str)
    ]
    if len(set(identifiers)) != len(identifiers):
        duplicates = sorted(
            identifier for identifier in set(identifiers) if identifiers.count(identifier) > 1
        )
        raise ValueError(f"{label} has duplicate node IDs: {duplicates[:5]}")


def _contains_pair(value: Any, key: str, expected: Any) -> bool:
    if isinstance(value, dict):
        return value.get(key) == expected or any(
            _contains_pair(child, key, expected) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_pair(child, key, expected) for child in value)
    return False


def _node_id(spec: CustomEnvironmentSpec, prefix: str, purpose: str) -> str:
    identifier = uuid.uuid5(
        uuid.NAMESPACE_URL, f"hytalerl:{spec.environment_id}:{purpose}"
    )
    return f"{prefix}-{identifier}"


def _path_hash_semantic(paths: Mapping[str, Any]) -> str:
    normalized = {str(key): _sha256(value, f"path hash {key}") for key, value in paths.items()}
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _semantic_digest(value: Mapping[str, Any], semantic_field: str) -> str:
    normalized = deepcopy(dict(value))
    normalized[semantic_field] = ""
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _safe_destination(path: str | Path) -> Path:
    destination = Path(path).resolve()
    if destination == Path(destination.anchor) or not destination.name:
        raise ValueError("asset-pack output must be a specific subdirectory")
    if destination.exists() and not destination.is_dir():
        raise ValueError("asset-pack output exists and is not a directory")
    return destination


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(encoded, encoding="utf-8", newline="\n")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    return _json_bytes(path.read_bytes(), label)


def _json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    return _object(value, label)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _exact_fields(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    fields = set(value)
    expected_set = set(expected)
    if fields != expected_set:
        raise ValueError(
            f"{label} fields differ: missing={sorted(expected_set - fields)} "
            f"extra={sorted(fields - expected_set)}"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value.lower()) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value.lower()


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} must contain only letters, numbers, dot, dash, underscore")
    return value


def _semver(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SEMVER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a three-part semantic version")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _asset_path(value: Any, label: str) -> str:
    text = _text(value, label).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or text != path.as_posix():
        raise ValueError(f"{label} must be a normalized relative asset path")
    return text


def _relative_directory(value: Any, label: str) -> str:
    text = _asset_path(value, label).rstrip("/")
    if not text or "." in PurePosixPath(text).parts:
        raise ValueError(f"{label} must be a non-empty asset directory")
    return text


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _bounded_int(value: Any, minimum: int, maximum: int, label: str) -> int:
    result = _integer(value, label)
    if not minimum <= result <= maximum:
        raise ValueError(f"{label} must be in [{minimum}, {maximum}]")
    return result


def _positive_int3(value: Any, label: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must contain three integers")
    result = tuple(_integer(item, f"{label}[{index}]") for index, item in enumerate(value))
    if any(item <= 0 for item in result):
        raise ValueError(f"{label} values must be positive")
    return result


def _bounded_float(value: Any, minimum: float, maximum: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{label} must be finite and in [{minimum}, {maximum}]")
    return result
