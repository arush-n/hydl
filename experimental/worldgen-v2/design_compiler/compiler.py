"""Compile WorldGen Studio recipes into pinned Hytale 0.5.7 asset packs.

Authored-family recipes deliberately use template cloning. Procedural-genome
recipes instead synthesize a bounded, documented density/material/prop AST
from supported 0.5.7 node types. In both modes the receipt lists controls that
still require capture or selection evidence rather than inventing support.

Compilation is offline.  It does not start a server, generate a world, capture
a Region, publish a corpus, or change legacy ``world="hytale"`` identity.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
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

from native_graph_compiler import (
    PINNED_BASIC_BIOME_PATH,
    PINNED_BASIC_BIOME_SHA256,
    compile_procedural_biome,
    finalize_procedural_biome_contract,
    validate_procedural_biome,
)


BUILD_RECEIPT_SCHEMA = "hytalerl_worldgen_v2_studio_pack_receipt_v1"
BUILD_RECEIPT_VERSION = 1
PINNED_SERVER_VERSION = "0.5.7"
PINNED_ASSETS_SHA256 = (
    "1b8802c284c228ae4549dac037716c175bc6b94b0fc9aac2b7039ac6bd2ffd5d"
)
WORLD_STRUCTURE_PATH = "Server/HytaleGenerator/WorldStructures/Basic.json"
WORLD_STRUCTURE_SHA256 = (
    "28a560d4d4ad4a273b1932070595fb41f5fb64f98b253cc43525459e0bd2654c"
)
ASSIGNMENT_TEMPLATE_PATH = (
    "Server/HytaleGenerator/Assignments/Plains1/Plains1_Mountains_Trees.json"
)
ASSIGNMENT_TEMPLATE_SHA256 = (
    "22b99cef58dcae874e3d3cced9f8cddf20c530ae5fdba69911ea71dcff261f0d"
)

PROFILE_CATALOG: dict[str, dict[str, Any]] = {
    "plains_mountains": {
        "path": "Server/HytaleGenerator/Biomes/Plains1/Plains1_Mountains.json",
        "sha256": "f9a3a186aebe81bb897cff8b485fb03703e6dd9baf4f42c76037aa24db794879",
        "native_character": "mountains_with_authored_plains_caves",
        "surface_materials": ["Soil_Grass", "Soil_Grass_Deep", "Rock_Shale"],
    },
    "taiga_mountains": {
        "path": "Server/HytaleGenerator/Biomes/Taiga1/Taiga1_Mountains.json",
        "sha256": "4df207ce00e27a1005e0138703f7bd23b9d77c8aec91fc255570edec33549531",
        "native_character": "alpine_taiga_mountains_with_authored_caves",
        "surface_materials": ["Soil_Snow", "Soil_Grass", "Rock_Quartzite"],
    },
    "plains_gorges": {
        "path": "Server/HytaleGenerator/Biomes/Plains1/Plains1_Gorges.json",
        "sha256": "808487d246b0ae9b80305dec8bba5acffb407662abc341b501416ef904d1ba12",
        "native_character": "gorges_plateaus_hills_and_authored_caves",
        "surface_materials": ["Soil_Grass", "Soil_Grass_Deep", "Rock_Shale"],
    },
    "deeproot_caverns": {
        "path": "Server/HytaleGenerator/Biomes/Plains1/Plains1_Deeproot.json",
        "sha256": "69caf250cd6d48ae704290af05401c61ebb34c1d489f1cd7271af0432c0e7fb2",
        "native_character": "deeproot_caverns_islands_and_authored_caves",
        "surface_materials": ["Soil_Grass", "Soil_Grass_Deep", "Rock_Shale"],
    },
    "desert_stacks": {
        "path": "Server/HytaleGenerator/Biomes/Desert1/Desert1_Stacks.json",
        "sha256": "20781f371685b12b67181aed395e3834e2d2ac5bc24daf46791b7e0144a7d6f4",
        "native_character": "desert_stacks_and_eroded_mesas",
        "surface_materials": ["Soil_Sand", "Rock_Sandstone"],
    },
    "default_flat": {
        "path": "Server/HytaleGenerator/Biomes/Default_Flat/Default_Flat.json",
        "sha256": "12af93a6289a15f78f8654b0e7b3dee81b17ae58e00e2fc5c9d95ea752e2e27e",
        "native_character": "flat_authored_reference_terrain",
        "surface_materials": ["Soil_Grass", "Soil_Grass_Deep", "Rock_Stone"],
    },
    "marble_pillars": {
        "path": "Server/HytaleGenerator/Biomes/Generative/Generative_Pillars_Marble_Large.json",
        "sha256": "9c842e94095c6cb389e78edb5981b06a0b667712899a10ca514470c049747492",
        "small_path": "Server/HytaleGenerator/Biomes/Generative/Generative_Pillars_Marble_Small.json",
        "small_sha256": "e639beaf5db667978c959f29c04178569b70ddb2866f12365ac956008d24beef",
        "native_character": "generative_marble_pillar_field",
        "surface_materials": ["Rock_Marble", "Rock_Stone"],
    },
    "twist_crater": {
        "path": "Server/HytaleGenerator/Biomes/Examples/Example_Twist_Crater.json",
        "sha256": "efd1dfcc5ef42db34a775dbb26be6feed5fdf414653a209ce557b1367d5e8801",
        "native_character": "example_twist_crater",
        "surface_materials": ["Rock_Stone", "Soil_Grass"],
    },
    "rooted_islands": {
        "path": "Server/HytaleGenerator/Biomes/Experimental/Islands_Roots.json",
        "sha256": "038aa16275132780539c65a7de96330cbe188373eb658a92885e8145a915a5dd",
        "native_character": "experimental_rooted_islands",
        "surface_materials": ["Soil_Grass", "Rock_Stone"],
    },
    "volcanic_caldera": {
        "path": "Server/HytaleGenerator/Biomes/Volcanic1/Volcanic1_Caldera.json",
        "sha256": "6b8ed460ab83f3a36421b92e18a4fb3e574a5093d1b7c2cfc426a8799d57d017",
        "native_character": "volcanic_caldera_and_authored_caves",
        "surface_materials": ["Soil_Ash", "Rock_Volcanic", "Rock_Basalt"],
    },
}

STRUCTURE_MATERIALS = {
    "shale": "Rock_Shale_Brick",
    "stone": "Rock_Stone",
    "marble": "Rock_Marble",
    "sandstone": "Rock_Sandstone_Brick",
}
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+")


def default_assets_path() -> Path:
    explicit = os.environ.get("HYTALE_ASSETS_ZIP")
    if explicit:
        return Path(explicit)
    app_data = os.environ.get("APPDATA")
    if not app_data:
        raise FileNotFoundError("HYTALE_ASSETS_ZIP and APPDATA are unavailable")
    return (
        Path(app_data) / "Hytale" / "install" / "release" / "package"
        / "game" / "latest" / "Assets.zip"
    )


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _semantic_sha256(value: Mapping[str, Any], field: str | None = None) -> str:
    copied = deepcopy(dict(value))
    if field is not None:
        copied[field] = ""
    return hashlib.sha256(_canonical_bytes(copied)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive_sha256(path: Path, cache_path: Path) -> str:
    # A strict cryptographic pin cannot trust path/size/mtime metadata: the
    # installed archive has been observed changing in place while preserving
    # all three. ``--entry-pins-only`` is the explicit fast mode. Strict mode
    # therefore rehashes the complete archive on every process invocation.
    del cache_path
    return _file_sha256(path)


def _entry(archive: ZipFile, path: str, expected: str) -> tuple[dict[str, Any], str]:
    try:
        raw = archive.read(path)
    except KeyError as exc:
        raise ValueError(f"pinned WorldGen source is missing: {path}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected:
        raise ValueError(
            f"pinned WorldGen source changed: {path}; observed={digest} expected={expected}"
        )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"pinned WorldGen source is not JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"pinned WorldGen source is not an object: {path}")
    return value, digest


def _profile_source(config: Mapping[str, Any], profile_id: str) -> dict[str, Any]:
    if profile_id not in PROFILE_CATALOG:
        raise ValueError(f"unknown native template profile {profile_id!r}")
    profile = dict(PROFILE_CATALOG[profile_id])
    if profile_id == "marble_pillars" and float(config["pillar_width"]) < 8.0:
        profile["path"] = profile.pop("small_path")
        profile["sha256"] = profile.pop("small_sha256")
        profile["native_character"] = "generative_small_marble_pillar_field"
    else:
        profile.pop("small_path", None)
        profile.pop("small_sha256", None)
    return profile


def _asset_id(name: str, digest: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", name)
    stem = "".join(word[:1].upper() + word[1:] for word in words) or "World"
    return f"Hydl_{stem[:42]}_{digest[:8]}"


def _node_id(environment_id: str, kind: str, label: str) -> str:
    value = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"hytalerl-worldgen-v2:{environment_id}:{kind}:{label}",
    )
    return f"{kind}-{value}"


def _walk(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _renew_node_ids(value: Any, environment_id: str) -> None:
    for index, node in enumerate(_walk(value)):
        original = node.get("$NodeId")
        if isinstance(original, str):
            kind = original.split("-", 1)[0] or "Node"
            node["$NodeId"] = _node_id(
                environment_id, kind, f"clone-{index}-{original}"
            )


def _framework_entries(value: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    framework = value.get("Framework")
    if not isinstance(framework, list):
        raise ValueError("world-structure template lacks Framework")
    groups = [
        row for row in framework
        if isinstance(row, dict) and row.get("Type") == name
    ]
    if len(groups) != 1 or not isinstance(groups[0].get("Entries"), list):
        raise ValueError(f"world-structure Framework.{name} changed")
    entries = groups[0]["Entries"]
    if not all(isinstance(row, dict) for row in entries):
        raise ValueError(f"world-structure Framework.{name} entries changed")
    return entries


def _author_world_structure(source: Mapping[str, Any], *, asset_id: str,
                            config: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(source))
    if value.get("Type") != "NoiseRange":
        raise ValueError("world-structure template is not NoiseRange")
    value["Biomes"] = []
    value["DefaultBiome"] = asset_id
    constants = {row.get("Name"): row for row in _framework_entries(value, "DecimalConstants")}
    if set(constants) != {"Base", "Water", "Bedrock"}:
        raise ValueError("world-structure constants changed")
    constants["Base"]["Value"] = round(float(config["base_height"]))
    constants["Water"]["Value"] = round(float(config["water_level"]))
    constants["Bedrock"]["Value"] = 0
    spawn_y = min(
        319,
        max(
            round(float(config["water_level"])) + 12,
            round(float(config["base_height"]) + float(config["relief"]) + 20),
        ),
    )
    positions = _framework_entries(value, "Positions")
    if len(positions) != 1:
        raise ValueError("world-structure spawn group changed")
    rows = positions[0].get("Positions", {}).get("Positions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("world-structure has no spawn positions")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("world-structure spawn position changed")
        row["Y"] = spawn_y
    return value


def _tree_rows(props: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for row in props:
        if not isinstance(row, dict):
            continue
        assignment = row.get("Assignments", {})
        name = assignment.get("Name") if isinstance(assignment, dict) else None
        if isinstance(name, str) and re.search(r"(?i)(tree|forest|cedar|redwood)", name):
            rows.append(row)
    return rows


def _runtime(environment_id: str, assignment_id: str, config: Mapping[str, Any]) -> dict[str, Any]:
    density = max(0.08, float(config["structure_density"]))
    spacing = max(8, round(float(config["structure_spacing"]) / math.sqrt(density)))
    return {
        "$NodeId": _node_id(environment_id, "Runtime", "custom-structures"),
        "Skip": False,
        "Runtime": 0,
        "Positions": {
            "$NodeId": _node_id(environment_id, "Mesh2DPositions", "custom-positions"),
            "Type": "Mesh2D",
            "Skip": False,
            "PointsY": 0,
            "PointGenerator": {
                "$NodeId": _node_id(environment_id, "MeshPointGenerator", "custom-generator"),
                "Type": "Mesh",
                "Jitter": float(config["structure_jitter"]),
                "ScaleX": spacing,
                "ScaleY": spacing,
                "ScaleZ": spacing,
                "Seed": f"{environment_id}-structures",
            },
        },
        "Assignments": {
            "$NodeId": _node_id(environment_id, "Imported.Assignments", "custom-assignment"),
            "Type": "Imported",
            "Name": assignment_id,
        },
    }


def _author_biome(source: Mapping[str, Any], *, asset_id: str,
                  assignment_id: str, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    value = deepcopy(dict(source))
    value["Name"] = asset_id
    raw_props = value.get("Props", [])
    props = list(raw_props) if isinstance(raw_props, list) else []
    tree_rows = _tree_rows(props)
    removed = 0
    scaled = 0
    if float(config["forest_density"]) <= 0:
        identities = {id(row) for row in tree_rows}
        before = len(props)
        props = [row for row in props if id(row) not in identities]
        removed = before - len(props)
    else:
        effective = max(
            3,
            round(float(config["tree_spacing"]) / math.sqrt(max(0.08, float(config["forest_density"])))),
        )
        for row in tree_rows:
            point = row.get("Positions", {}).get("PointGenerator", {})
            if isinstance(point, dict) and point.get("Type") == "Mesh":
                for field in ("ScaleX", "ScaleY", "ScaleZ"):
                    point[field] = effective
                scaled += 1
    if config["structure_set"] != "none" and float(config["structure_density"]) > 0:
        props.append(_runtime(str(config["name"]), assignment_id, config))
    value["Props"] = props
    metadata = value.get("$NodeEditorMetadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("$Nodes"), dict):
        for index, node in enumerate(_walk(props[-1] if props else {})):
            node_id = node.get("$NodeId")
            if isinstance(node_id, str) and node_id not in metadata["$Nodes"]:
                metadata["$Nodes"][node_id] = {
                    "$Position": {"$x": -7000 + index * 360, "$y": 20000}
                }
    return value, {
        "native_tree_rows_found": len(tree_rows),
        "native_tree_rows_scaled": scaled,
        "native_tree_rows_removed": removed,
        "custom_structure_runtime_added": config["structure_set"] != "none"
        and float(config["structure_density"]) > 0,
    }


def _author_assignment(source: Mapping[str, Any], *, environment_id: str,
                       assignment_id: str, prefab_directory: str,
                       surface_materials: list[str]) -> dict[str, Any]:
    candidates = [
        node for node in _walk(source)
        if node.get("Type") == "Prefab" and isinstance(node.get("WeightedPrefabPaths"), list)
    ]
    matching = [
        node for node in candidates
        if any(
            isinstance(row, dict) and row.get("Path") == "Trees/Beech/Stage_1"
            for row in node["WeightedPrefabPaths"]
        )
    ]
    if len(matching) != 1:
        raise ValueError("structure placement template changed")
    prefab = deepcopy(matching[0])
    _renew_node_ids(prefab, environment_id)
    prefab["WeightedPrefabPaths"] = [{
        "$NodeId": _node_id(environment_id, "WeightedPath.Prefab.Prop", "custom-path"),
        "Path": prefab_directory.rstrip("/") + "/",
        "Weight": 100,
    }]
    prefab["LegacyPath"] = False
    prefab["LoadEntities"] = False
    material_sets = [
        node for node in _walk(prefab)
        if node.get("Type") == "BlockSet"
        and isinstance(node.get("BlockSet"), dict)
        and isinstance(node["BlockSet"].get("Materials"), list)
    ]
    if len(material_sets) != 1:
        raise ValueError("structure surface material template changed")
    material_sets[0]["BlockSet"]["Materials"] = [
        {
            "$NodeId": _node_id(environment_id, "Material", f"surface-{index}"),
            "Solid": material,
        }
        for index, material in enumerate(surface_materials)
    ]
    value = {
        "$NodeId": _node_id(environment_id, "Constant.Assignments", "root"),
        "Type": "Constant",
        "ExportAs": assignment_id,
        "Prop": prefab,
        "$NodeEditorMetadata": {
            "$Nodes": {}, "$FloatingNodes": [], "$Links": {}, "$Groups": [],
            "$Comments": [], "$WorkspaceID": "HytaleGenerator - Assignments",
        },
    }
    for index, node in enumerate(_walk(value)):
        node_id = node.get("$NodeId")
        if isinstance(node_id, str):
            value["$NodeEditorMetadata"]["$Nodes"][node_id] = {
                "$Position": {"$x": 800 + index * 340, "$y": 2400}
            }
    return value


def _blocks(material: str, coordinates: Iterable[tuple[int, int, int]]) -> list[dict[str, Any]]:
    unique = sorted(set(coordinates), key=lambda row: (row[1], row[2], row[0]))
    return [{"name": material, "x": x, "y": y, "z": z} for x, y, z in unique]


def _primitive_prefabs(structure_set: str, material: str) -> dict[str, dict[str, Any]]:
    if structure_set == "procedural_mix":
        combined: dict[str, dict[str, Any]] = {}
        for family in (
            "mixed_ruins", "watchtowers", "fortified_camps", "villages",
            "arena_pillars", "shrines", "bridges",
        ):
            combined.update(_primitive_prefabs(family, material))
        return combined
    variants: dict[str, set[tuple[int, int, int]]] = {}
    if structure_set == "arena_pillars":
        variants["Pillar"] = {
            (x, y, z) for x in range(-1, 2) for z in range(-1, 2) for y in range(0, 13)
        }
    elif structure_set == "watchtowers":
        cells = set()
        for y in range(0, 11):
            for x in range(-2, 3):
                for z in range(-2, 3):
                    if abs(x) == 2 or abs(z) == 2:
                        if not (z == -2 and x == 0 and y < 3):
                            cells.add((x, y, z))
        cells.update((x, 11, z) for x in range(-3, 4) for z in range(-3, 4))
        variants["Watchtower"] = cells
    elif structure_set == "fortified_camps":
        cells = {(x, 0, z) for x in range(-5, 6) for z in range(-5, 6)}
        cells.update(
            (x, y, z) for y in range(1, 4) for x in range(-5, 6)
            for z in range(-5, 6) if abs(x) == 5 or abs(z) == 5
        )
        variants["FortifiedCamp"] = cells
    elif structure_set == "villages":
        cells = {(x, 0, z) for x in range(-3, 4) for z in range(-2, 3)}
        cells.update(
            (x, y, z) for y in range(1, 5) for x in range(-3, 4)
            for z in range(-2, 3) if abs(x) == 3 or abs(z) == 2
        )
        cells.update((x, 5, z) for x in range(-3, 4) for z in range(-2, 3))
        cells.difference_update({(0, 1, -2), (0, 2, -2)})
        variants["VillageHouse"] = cells
    elif structure_set == "bridges":
        cells = {(x, 0, z) for x in range(-7, 8) for z in range(-1, 2)}
        cells.update((x, 1, z) for x in range(-7, 8) for z in (-2, 2))
        variants["Bridge"] = cells
    elif structure_set == "shrines":
        cells = {(x, 0, z) for x in range(-3, 4) for z in range(-3, 4)}
        cells.update((0, y, 0) for y in range(1, 9))
        cells.update((x, 7, 0) for x in range(-2, 3))
        variants["Shrine"] = cells
    else:  # mixed_ruins
        arch = {(x, y, 0) for x in (-3, 3) for y in range(0, 7)}
        arch.update((x, 6, 0) for x in range(-3, 4))
        ruin = {(x, 0, z) for x in range(-4, 5) for z in range(-4, 5)}
        ruin.update(
            (x, y, z) for y in range(1, 5) for x in range(-4, 5)
            for z in range(-4, 5) if (abs(x) == 4 or abs(z) == 4)
            and ((x * 7 + z * 11 + y * 13) % 5 != 0)
        )
        shrine = {(x, 0, z) for x in range(-2, 3) for z in range(-2, 3)}
        shrine.update((0, y, 0) for y in range(1, 8))
        variants = {"RuinArch": arch, "BrokenRuin": ruin, "RuinShrine": shrine}
    return {
        name: {
            "anchorX": 0, "anchorY": 0, "anchorZ": 0,
            "blockIdVersion": 11,
            "blocks": _blocks(material, cells),
            "entities": [],
            "version": 8,
        }
        for name, cells in variants.items()
    }


def _manifest(pack_name: str) -> dict[str, Any]:
    return {
        "Group": "HytaleRL",
        "Name": pack_name,
        "Version": "1.0.0",
        "Description": "WorldGen Studio template-backed custom environment",
        "Authors": [{"Name": "HytaleRL Contributors"}],
        "ServerVersion": f"={PINNED_SERVER_VERSION}",
        "Dependencies": {}, "OptionalDependencies": {}, "LoadBefore": {},
        "DisabledByDefault": False, "IncludesAssetPack": False,
    }


def _safe_relative(path: str) -> str:
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe output path {path!r}")
    return pure.as_posix()


def _write_outputs(root: Path, outputs: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative, value in sorted(outputs.items()):
        safe = _safe_relative(relative)
        path = root / PurePosixPath(safe)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
        path.write_text(encoded, encoding="utf-8", newline="\n")
        hashes[safe] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return hashes


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def compile_design(payload: Mapping[str, Any], *, assets: str | Path | None = None,
                   output: str | Path, verify_archive: bool = True) -> dict[str, Any]:
    """Compile one normalized Studio config and atomically publish its pack."""

    from console.core.worlds import worldgen

    config = worldgen.normalize(dict(payload))
    procedural = config["generation_mode"] == "procedural_genome"
    design = worldgen._design(config)  # one canonical boundary shared with the console
    design_digest = worldgen.native_recipe_digest(config)
    native = design["native_handoff"]["template_compilation"]
    profile_id = str(native["profile"])
    if procedural:
        profile = {
            "path": PINNED_BASIC_BIOME_PATH,
            "sha256": PINNED_BASIC_BIOME_SHA256,
            "native_character": (
                "synthesized_hyperparameter_density_material_prop_graph"
            ),
            "surface_materials": [],
        }
    else:
        profile = _profile_source(config, profile_id)
    asset_id = _asset_id(config["name"], design_digest)
    assignment_id = f"{asset_id}_Structures"
    prefab_directory = f"Hydl/{asset_id}/{config['structure_set']}"
    pack_name = re.sub(r"[^A-Za-z0-9]", "", asset_id)[:60]
    target = Path(output).resolve()
    source_archive = Path(assets) if assets is not None else default_assets_path()
    source_archive = source_archive.resolve()
    if not source_archive.is_file():
        raise FileNotFoundError(source_archive)
    cache = target.parent / ".assets-sha256-cache.json"
    archive_sha = _archive_sha256(source_archive, cache) if verify_archive else None
    if verify_archive and archive_sha != PINNED_ASSETS_SHA256:
        raise ValueError(
            "installed Assets.zip differs from pinned Hytale 0.5.7 authoring source: "
            f"observed={archive_sha} expected={PINNED_ASSETS_SHA256}"
        )

    try:
        with ZipFile(source_archive) as archive:
            world_source, world_sha = _entry(
                archive, WORLD_STRUCTURE_PATH, WORLD_STRUCTURE_SHA256
            )
            biome_source, biome_sha = _entry(
                archive, str(profile["path"]), str(profile["sha256"])
            )
            assignment_source, assignment_sha = _entry(
                archive, ASSIGNMENT_TEMPLATE_PATH, ASSIGNMENT_TEMPLATE_SHA256
            )
    except BadZipFile as exc:
        raise ValueError(f"invalid Hytale asset archive: {source_archive}") from exc

    world_structure = _author_world_structure(
        world_source, asset_id=asset_id, config=config
    )
    native_graph_contract: dict[str, Any] | None = None
    if procedural:
        procedural_biome, native_graph_contract = compile_procedural_biome(
            biome_source,
            asset_id=asset_id,
            config=config,
        )
        profile["surface_materials"] = list(
            native_graph_contract["surface_materials"]
        )
        biome, vegetation_transform = _author_biome(
            procedural_biome,
            asset_id=asset_id,
            assignment_id=assignment_id,
            config=config,
        )
        native_graph_contract = finalize_procedural_biome_contract(
            biome,
            native_graph_contract,
        )
    else:
        biome, vegetation_transform = _author_biome(
            biome_source,
            asset_id=asset_id,
            assignment_id=assignment_id,
            config=config,
        )
    outputs: dict[str, Mapping[str, Any]] = {
        "manifest.json": _manifest(pack_name),
        f"Server/HytaleGenerator/WorldStructures/{asset_id}.json": world_structure,
        f"Server/HytaleGenerator/Biomes/Hydl/{asset_id}.json": biome,
    }
    prefab_names: list[str] = []
    if config["structure_set"] != "none" and float(config["structure_density"]) > 0:
        assignment = _author_assignment(
            assignment_source,
            environment_id=config["name"],
            assignment_id=assignment_id,
            prefab_directory=prefab_directory,
            surface_materials=list(profile["surface_materials"]),
        )
        outputs[f"Server/HytaleGenerator/Assignments/Hydl/{assignment_id}.json"] = assignment
        material = STRUCTURE_MATERIALS[str(config["structure_material"])]
        for name, prefab in _primitive_prefabs(str(config["structure_set"]), material).items():
            prefab_names.append(name)
            outputs[f"Server/Prefabs/{prefab_directory}/{name}.prefab.json"] = prefab

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        output_hashes = _write_outputs(temporary, outputs)
        pack_semantic = hashlib.sha256(
            _canonical_bytes({"files": output_hashes})
        ).hexdigest()
        receipt: dict[str, Any] = {
            "schema": BUILD_RECEIPT_SCHEMA,
            "version": BUILD_RECEIPT_VERSION,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "environment_id": config["name"],
            "design_digest": design_digest,
            "config": config,
            "server_version": PINNED_SERVER_VERSION,
            "assets_archive": {
                "path": str(source_archive),
                "sha256": archive_sha,
                "whole_archive_verified": verify_archive,
            },
            "source_assets": {
                "world_structure": {"path": WORLD_STRUCTURE_PATH, "sha256": world_sha},
                "biome_template": {"path": profile["path"], "sha256": biome_sha},
                "assignment_template": {"path": ASSIGNMENT_TEMPLATE_PATH, "sha256": assignment_sha},
            },
            "native_template": {
                "profile": profile_id,
                "native_character": profile["native_character"],
                "requested_cave_style": config["cave_style"],
                "intent_warnings": native["warnings"],
            },
            "pack": {
                "group": "HytaleRL", "name": pack_name,
                "asset_id": asset_id, "assignment_id": assignment_id,
                "prefab_directory": prefab_directory,
                "prefabs": sorted(prefab_names),
            },
            "transforms": {
                "world_constants": ["base_height", "water_level", "spawn_height"],
                "vegetation": vegetation_transform,
                "structure_density_mapping": (
                    "mesh_spacing_divided_by_sqrt_density"
                    if prefab_names else "disabled"
                ),
                "preview_only_controls": native[
                    "preview_only_until_density_graph_compiler"
                ],
                "compile_request_capture_extent": worldgen.extent_capture_plan(config),
                "capture_extent_authority": (
                    "The saved Studio design or seed plan is authoritative. "
                    "Extent is not part of this unbounded native pack identity."
                ),
            },
            "boundary": {
                "generator": "hytale_generator",
                "legacy_world_identity_unchanged": True,
                "asset_pack_compiled": True,
                "native_asset_codec_acceptance": (
                    "pack_specific_live_validation_required"
                    if procedural
                    else "inherited_from_pinned_template"
                ),
                "compiler_family_reference_validation": (
                    dict(worldgen.PROCEDURAL_GRAPH_REFERENCE_VALIDATION)
                    if procedural else None
                ),
                "native_generation_performed": False,
                "region_capture_performed": False,
                "jax_publication_performed": False,
            },
            "output_files": output_hashes,
            "pack_semantic_sha256": pack_semantic,
            "receipt_semantic_sha256": "",
        }
        if native_graph_contract is not None:
            receipt["native_graph"] = native_graph_contract
        receipt["receipt_semantic_sha256"] = _semantic_sha256(
            receipt, "receipt_semantic_sha256"
        )
        _atomic_json(temporary / "studio-authoring-receipt.json", receipt)

        if target.exists():
            existing = validate_pack(target)
            if existing["pack_semantic_sha256"] != pack_semantic:
                raise FileExistsError(
                    f"compiled pack already exists with different content: {target}"
                )
            existing["reused"] = True
            existing["requested_capture_extent"] = worldgen.extent_capture_plan(config)
            return existing
        os.replace(temporary, target)
        result = validate_pack(target)
        result["reused"] = False
        result["requested_capture_extent"] = worldgen.extent_capture_plan(config)
        return result
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def validate_pack(root: str | Path) -> dict[str, Any]:
    """Validate a compiled pack without consulting the installed archive."""

    pack_root = Path(root).resolve()
    receipt_path = pack_root / "studio-authoring-receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read Studio authoring receipt: {receipt_path}") from exc
    if not isinstance(receipt, dict) or receipt.get("schema") != BUILD_RECEIPT_SCHEMA:
        raise ValueError("unsupported Studio authoring receipt")
    if receipt.get("version") != BUILD_RECEIPT_VERSION:
        raise ValueError("unsupported Studio authoring receipt version")
    expected_receipt = _semantic_sha256(receipt, "receipt_semantic_sha256")
    if receipt.get("receipt_semantic_sha256") != expected_receipt:
        raise ValueError("Studio authoring receipt semantic hash differs")
    output_files = receipt.get("output_files")
    if not isinstance(output_files, dict) or not output_files:
        raise ValueError("Studio authoring receipt has no output files")
    observed: dict[str, str] = {}
    for relative, expected in sorted(output_files.items()):
        safe = _safe_relative(str(relative))
        path = pack_root / PurePosixPath(safe)
        if not path.is_file():
            raise ValueError(f"compiled pack output is missing: {safe}")
        digest = _file_sha256(path)
        if digest != expected:
            raise ValueError(f"compiled pack output changed: {safe}")
        observed[safe] = digest
    pack_semantic = hashlib.sha256(_canonical_bytes({"files": observed})).hexdigest()
    if receipt.get("pack_semantic_sha256") != pack_semantic:
        raise ValueError("compiled pack semantic hash differs")
    boundary = receipt.get("boundary", {})
    if boundary.get("generator") != "hytale_generator":
        raise ValueError("compiled pack lost explicit hytale_generator identity")
    if boundary.get("legacy_world_identity_unchanged") is not True:
        raise ValueError("compiled pack may redefine legacy world identity")

    config = receipt.get("config", {})
    pack = receipt.get("pack", {})
    asset_id = pack.get("asset_id")
    if not isinstance(asset_id, str) or _SAFE_ID.fullmatch(asset_id) is None:
        raise ValueError("compiled pack asset identity is unsafe")
    ws_path = pack_root / f"Server/HytaleGenerator/WorldStructures/{asset_id}.json"
    biome_path = pack_root / f"Server/HytaleGenerator/Biomes/Hydl/{asset_id}.json"
    world_structure = json.loads(ws_path.read_text(encoding="utf-8"))
    biome = json.loads(biome_path.read_text(encoding="utf-8"))
    if world_structure.get("DefaultBiome") != asset_id or world_structure.get("Biomes") != []:
        raise ValueError("compiled world structure does not select one custom biome")
    constants = {
        row.get("Name"): row.get("Value")
        for row in _framework_entries(world_structure, "DecimalConstants")
    }
    if constants.get("Base") != round(float(config["base_height"])):
        raise ValueError("compiled base height differs")
    if constants.get("Water") != round(float(config["water_level"])):
        raise ValueError("compiled water height differs")
    if biome.get("Name") != asset_id:
        raise ValueError("compiled biome identity differs")
    generation_mode = config.get("generation_mode", "authored_family")
    native_graph = receipt.get("native_graph")
    if generation_mode == "procedural_genome":
        if receipt.get("native_template", {}).get("profile") != "procedural_graph_v1":
            raise ValueError("procedural pack lost its graph compiler profile")
        if not isinstance(native_graph, dict):
            raise ValueError("procedural pack has no native graph contract")
        validate_procedural_biome(biome, native_graph)
    elif native_graph is not None:
        raise ValueError("template-backed pack unexpectedly carries a native graph")
    for relative in observed:
        if relative.endswith(".prefab.json"):
            prefab = json.loads((pack_root / PurePosixPath(relative)).read_text(encoding="utf-8"))
            cells = [
                (row.get("x"), row.get("y"), row.get("z"))
                for row in prefab.get("blocks", []) if isinstance(row, dict)
            ]
            if not cells or len(cells) != len(set(cells)):
                raise ValueError(f"compiled primitive prefab is empty or overlaps: {relative}")
    return {
        "status": "passed",
        "path": str(pack_root),
        "environment_id": receipt["environment_id"],
        "design_digest": receipt["design_digest"],
        "asset_id": asset_id,
        "native_profile": receipt["native_template"]["profile"],
        "output_file_count": len(observed),
        "prefab_count": len(receipt["pack"]["prefabs"]),
        "pack_semantic_sha256": pack_semantic,
        "receipt_semantic_sha256": receipt["receipt_semantic_sha256"],
        "boundary": boundary,
    }
