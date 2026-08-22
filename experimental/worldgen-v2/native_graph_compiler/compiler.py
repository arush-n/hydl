"""Compile one seed-independent procedural envelope to a V2 biome graph.

This compiler synthesizes the terrain density tree.  It uses Hytale's pinned
``Basic`` biome only as a codec/editor shell and replaces its terrain,
materials, props, environment, tint, and editor metadata.  Native generation
and Region capture remain the authority for what the graph actually produces.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Iterable, Mapping
import uuid


NATIVE_GRAPH_SCHEMA = "hytalerl_worldgen_v2_procedural_native_graph_v1"
NATIVE_GRAPH_VERSION = 1
PINNED_BASIC_BIOME_PATH = "Server/HytaleGenerator/Biomes/Basic.json"
PINNED_BASIC_BIOME_SHA256 = (
    "d4c9116a2caa2613849730b9bea0c00674422ea0253b071bb122b2037d5fab38"
)

COMPONENT_FIELDS = {
    "mountains": "terrain_mountain_weight",
    "hills": "terrain_hill_weight",
    "valleys": "terrain_valley_weight",
    "plateaus": "terrain_plateau_weight",
    "canyons": "terrain_canyon_weight",
    "pillars": "terrain_pillar_weight",
    "craters": "terrain_crater_weight",
    "flats": "terrain_flat_weight",
}

_ALLOWED_DENSITY_TYPES = {
    "BaseHeight",
    "CellNoise2D",
    "Constant",
    "CurveMapper",
    "Inverter",
    "Min",
    "Multiplier",
    "Normalizer",
    "SimplexNoise2D",
    "SimplexNoise3D",
    "Sum",
}


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _semantic_sha256(value: Mapping[str, Any], field: str | None = None) -> str:
    copied = deepcopy(dict(value))
    if field is not None:
        copied[field] = ""
    return hashlib.sha256(_canonical_bytes(copied)).hexdigest()


def _node_id(asset_id: str, kind: str, label: str) -> str:
    value = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"hytalerl-worldgen-v2:{asset_id}:{kind}:{label}",
    )
    return f"{kind}-{value}"


def _node(asset_id: str, kind: str, label: str, **values: Any) -> dict[str, Any]:
    return {
        "$NodeId": _node_id(asset_id, kind, label),
        **values,
    }


def _curve(
    asset_id: str,
    label: str,
    source: Mapping[str, Any],
    points: Iterable[tuple[float, float]],
) -> dict[str, Any]:
    rows = [
        _node(
            asset_id,
            "CurvePoint",
            f"{label}-{index}",
            **{"In": float(value), "Out": float(output)},
        )
        for index, (value, output) in enumerate(points)
    ]
    return _node(
        asset_id,
        "CurveMapper.Density",
        label,
        Type="CurveMapper",
        Skip=False,
        Curve=_node(
            asset_id,
            "ManualCurve",
            label,
            Type="Manual",
            Points=rows,
        ),
        Inputs=[dict(source)],
    )


def _normalizer(
    asset_id: str,
    label: str,
    source: Mapping[str, Any],
    output_minimum: float,
    output_maximum: float,
    *,
    source_minimum: float = -1.0,
    source_maximum: float = 1.0,
) -> dict[str, Any]:
    return _node(
        asset_id,
        "NormalizerDensityNode",
        label,
        Type="Normalizer",
        Skip=False,
        FromMin=float(source_minimum),
        FromMax=float(source_maximum),
        ToMin=float(output_minimum),
        ToMax=float(output_maximum),
        Inputs=[dict(source)],
    )


def _simplex2d(
    asset_id: str,
    label: str,
    *,
    scale: float,
    octaves: int,
    lacunarity: float,
    persistence: float,
) -> dict[str, Any]:
    return _node(
        asset_id,
        "SimplexNoise2DDensityNode",
        label,
        Type="SimplexNoise2D",
        Skip=False,
        Lacunarity=round(float(lacunarity), 6),
        Persistence=round(float(persistence), 6),
        Octaves=int(octaves),
        Scale=round(max(4.0, float(scale)), 6),
        Seed=f"{asset_id}:{label}",
    )


def _cell2d(
    asset_id: str,
    label: str,
    *,
    scale: float,
    jitter: float,
) -> dict[str, Any]:
    return _node(
        asset_id,
        "CellNoise2DDensityNode",
        label,
        Type="CellNoise2D",
        Skip=False,
        ScaleX=round(max(4.0, float(scale)), 6),
        ScaleZ=round(max(4.0, float(scale)), 6),
        Jitter=round(max(0.0, min(0.95, float(jitter))), 6),
        CellType="Distance2Div",
        Octaves=1,
        Seed=f"{asset_id}:{label}",
    )


def _weights(config: Mapping[str, Any]) -> dict[str, float]:
    raw = {
        component: max(0.0, float(config[field]))
        for component, field in COMPONENT_FIELDS.items()
    }
    total = sum(raw.values())
    if total <= 1.0e-12:
        raise ValueError("procedural graph requires a non-empty terrain distribution")
    return {name: value / total for name, value in raw.items()}


def _shape_source(
    asset_id: str,
    component: str,
    config: Mapping[str, Any],
    index: int,
) -> dict[str, Any]:
    scale = float(config["mountain_scale"])
    roughness = float(config["roughness"])
    erosion = float(config["erosion"])
    octaves = max(1, min(4, 1 + round(roughness * (2.8 - erosion))))
    lacunarity = 1.55 + roughness * 2.15
    persistence = 0.22 + roughness * 0.38 * (1.0 - erosion * 0.35)
    label = f"component-{component}"

    if component == "flats":
        return _node(
            asset_id,
            "ConstantDensityNode",
            label,
            Type="Constant",
            Skip=False,
            Value=0.0,
        )
    if component in {"canyons", "pillars", "craters"}:
        if component == "canyons":
            cell_scale = scale * (0.75 + float(config["canyon_strength"]) * 0.9)
            points = ((0.0, -1.0), (0.16, -0.85), (0.38, 0.1), (1.0, 0.35))
        elif component == "pillars":
            spacing = max(
                float(config["pillar_width"]) * 2.0,
                float(config["pillar_width"]) / max(0.08, float(config["pillar_density"])),
            )
            cell_scale = spacing
            points = ((0.0, 1.0), (0.14, 0.9), (0.32, -0.25), (1.0, -0.35))
        else:
            cell_scale = scale * (0.75 + (1.0 - float(config["crater_density"])) * 1.8)
            points = ((0.0, -0.7), (0.22, -1.0), (0.47, 0.65), (0.72, 0.1), (1.0, 0.0))
        return _curve(
            asset_id,
            label,
            _cell2d(
                asset_id,
                label,
                scale=cell_scale,
                jitter=0.12 + float(config["world_variation"]) * 0.45,
            ),
            points,
        )

    source = _simplex2d(
        asset_id,
        label,
        scale=scale * {
            "mountains": 1.65,
            "hills": 0.72,
            "valleys": 1.15,
            "plateaus": 1.32,
        }[component] * (1.0 + index * 0.025),
        octaves=octaves,
        lacunarity=lacunarity,
        persistence=persistence,
    )
    curves = {
        "mountains": ((-1.0, -0.2), (-0.3, -0.08), (0.0, 0.0), (0.45, 0.52), (1.0, 1.0)),
        "hills": ((-1.0, -0.65), (0.0, 0.0), (1.0, 0.65)),
        "valleys": ((-1.0, -1.0), (-0.35, -0.55), (0.15, 0.05), (1.0, 0.2)),
        "plateaus": ((-1.0, -0.25), (-0.2, -0.2), (0.0, 0.24), (0.18, 0.72), (1.0, 0.78)),
    }
    return _curve(asset_id, label, source, curves[component])


def _component_density(
    asset_id: str,
    component: str,
    weight: float,
    config: Mapping[str, Any],
    index: int,
) -> dict[str, Any]:
    total_amplitude = min(3.2, max(0.08, float(config["relief"]) / 28.0))
    character = {
        "mountains": 1.35 + float(config["ridge_strength"]) * 0.45,
        "hills": 0.72,
        "valleys": 0.92,
        "plateaus": 0.88,
        "canyons": 0.65 + float(config["canyon_strength"]) * 0.75,
        "pillars": 0.45 + min(1.4, float(config["pillar_height"]) / 52.0),
        "craters": 0.45 + min(1.5, float(config["crater_depth"]) / 48.0),
        "flats": 0.0,
    }[component]
    amplitude = total_amplitude * weight * character
    shape = _normalizer(
        asset_id,
        f"component-amplitude-{component}",
        _shape_source(asset_id, component, config, index),
        -amplitude,
        amplitude,
    )
    variation = float(config["world_variation"])
    blend = float(config["region_blend"])
    macro_count = max(1, int(config["macro_region_count"]))
    macro_scale = float(config["mountain_scale"]) * (
        1.2 + 3.4 / math.sqrt(float(macro_count))
    )
    mask = _normalizer(
        asset_id,
        f"macro-mask-{component}",
        _simplex2d(
            asset_id,
            f"macro-mask-{component}",
            scale=macro_scale * (0.82 + index * 0.13),
            octaves=1 if blend >= 0.45 else 2,
            lacunarity=1.8,
            persistence=0.32 + blend * 0.25,
        ),
        max(0.08, 1.0 - variation * (0.92 - blend * 0.38)),
        1.0 + variation * (0.32 + (1.0 - blend) * 0.42),
    )
    return _node(
        asset_id,
        "MultiplierDensityNode",
        f"spatial-component-{component}",
        Type="Multiplier",
        Skip=False,
        Inputs=[shape, mask],
    )


def _base_density(asset_id: str) -> dict[str, Any]:
    return _curve(
        asset_id,
        "base-height",
        _node(
            asset_id,
            "BaseHeight.Density",
            "base-height",
            Type="BaseHeight",
            Skip=False,
            BaseHeightName="Base",
            Distance=True,
        ),
        ((0.0, 1.0), (50.0, -1.0)),
    )


def _hydrology_density(asset_id: str, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lake_coverage = float(config["lake_coverage"])
    if lake_coverage > 0.0:
        basin = _curve(
            asset_id,
            "lake-basins",
            _cell2d(
                asset_id,
                "lake-basins",
                scale=float(config["mountain_scale"]) * (1.4 + (1.0 - lake_coverage) * 2.2),
                jitter=0.28 + float(config["world_variation"]) * 0.28,
            ),
            ((0.0, -1.0), (0.2, -0.75), (0.48, 0.0), (1.0, 0.15)),
        )
        rows.append(
            _normalizer(
                asset_id,
                "lake-basin-depth",
                basin,
                -min(1.35, lake_coverage * 1.35),
                0.0,
            )
        )
    river = float(config["river_strength"])
    if river > 0.0:
        channel = _curve(
            asset_id,
            "river-channels",
            _cell2d(
                asset_id,
                "river-channels",
                scale=max(20.0, float(config["mountain_scale"]) * 1.9),
                jitter=0.18 + float(config["world_variation"]) * 0.35,
            ),
            ((0.0, -1.0), (0.09, -0.8), (0.24, 0.0), (1.0, 0.05)),
        )
        rows.append(
            _normalizer(
                asset_id,
                "river-channel-depth",
                channel,
                -min(0.9, river * 0.9),
                0.0,
            )
        )
    return rows


def _cave_density(asset_id: str, config: Mapping[str, Any]) -> dict[str, Any] | None:
    density = float(config["cave_density"])
    if config["cave_style"] == "none" or density <= 0.0:
        return None
    scale = float(config["cave_scale"])
    verticality = float(config["cave_verticality"])
    noise = _node(
        asset_id,
        "SimplexNoise3DDensityNode",
        "cave-field",
        Type="SimplexNoise3D",
        Skip=False,
        Lacunarity=round(1.7 + density * 1.5, 6),
        Persistence=round(0.28 + density * 0.28, 6),
        Octaves=2 if density < 0.72 else 3,
        ScaleXZ=round(scale, 6),
        ScaleY=round(max(4.0, scale * (1.65 - verticality * 1.25)), 6),
        Seed=f"{asset_id}:caves:{config['cave_style']}",
    )
    half_width = 0.055 + density * 0.42
    cave = _curve(
        asset_id,
        "cave-threshold",
        noise,
        (
            (-1.0, 1.0),
            (-half_width, 1.0),
            (0.0, -1.0),
            (half_width, 1.0),
            (1.0, 1.0),
        ),
    )
    entrance_bias = float(config["cave_entrances"])
    if entrance_bias <= 0.0:
        return cave
    entrance = _normalizer(
        asset_id,
        "cave-entrance-field",
        _simplex2d(
            asset_id,
            "cave-entrance-field",
            scale=max(10.0, scale * 1.7),
            octaves=1,
            lacunarity=2.0,
            persistence=0.5,
        ),
        -entrance_bias * 0.35,
        entrance_bias * 0.35,
    )
    return _node(
        asset_id,
        "SumDensityNode",
        "cave-with-entrances",
        Type="Sum",
        Skip=False,
        Inputs=[cave, entrance],
    )


def _density_graph(asset_id: str, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    weights = _weights(config)
    terrain = [_base_density(asset_id)]
    for index, (component, weight) in enumerate(weights.items()):
        if weight > 1.0e-9 and component != "flats":
            terrain.append(
                _component_density(asset_id, component, weight, config, index)
            )
    terrain.extend(_hydrology_density(asset_id, config))
    surface = _node(
        asset_id,
        "SumDensityNode",
        "procedural-surface",
        Type="Sum",
        ExportAs=f"{asset_id}_Terrain",
        Skip=False,
        Inputs=terrain,
    )
    cave = _cave_density(asset_id, config)
    if cave is None:
        return surface, weights
    return (
        _node(
            asset_id,
            "MinDensityNode",
            "surface-with-caves",
            Type="Min",
            Skip=False,
            Inputs=[surface, cave],
        ),
        weights,
    )


def _material(asset_id: str, label: str, *, solid: str | None = None,
              fluid: str | None = None) -> dict[str, Any]:
    material: dict[str, Any] = {
        "$NodeId": _node_id(asset_id, "Material", label),
    }
    if solid is not None:
        material["Solid"] = solid
    if fluid is not None:
        material["Fluid"] = fluid
    return _node(
        asset_id,
        "ConstantMaterialProvider",
        label,
        Type="Constant",
        Material=material,
    )


def _surface_palette(config: Mapping[str, Any]) -> tuple[str, str, str]:
    temperature = float(config["temperature"])
    moisture = float(config["moisture"])
    if temperature <= 0.3 or float(config["snow_line"]) <= float(config["base_height"]):
        return "Soil_Snow", "Soil_Grass", "Rock_Quartzite"
    if temperature >= 0.68 and moisture <= 0.32:
        return "Soil_Sand", "Rock_Sandstone", "Rock_Sandstone"
    return "Soil_Grass", "Soil_Dirt", "Rock_Shale"


def _material_provider(asset_id: str, config: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[str, str, str]]:
    surface, subsurface, rock = _surface_palette(config)
    solid_queue = [
        _node(
            asset_id,
            "SimpleHorizontalMaterialProvider",
            "bedrock-band",
            Type="SimpleHorizontal",
            TopY=1,
            TopBaseHeight="Bedrock",
            BottomY=0,
            BottomBaseHeight="Bedrock",
            Material=_material(asset_id, "bedrock", solid="Rock_Bedrock"),
        ),
        _node(
            asset_id,
            "SpaceAndDepthMaterialProvider",
            "surface-layers",
            Type="SpaceAndDepth",
            LayerContext="DEPTH_INTO_FLOOR",
            MaxExpectedDepth=4,
            Layers=[
                _node(
                    asset_id,
                    "ConstantThicknessLayerSADMP",
                    "surface-layer",
                    Type="ConstantThickness",
                    Thickness=1,
                    Material=_material(asset_id, "surface", solid=surface),
                ),
                _node(
                    asset_id,
                    "ConstantThicknessLayerSADMP",
                    "subsurface-layer",
                    Type="ConstantThickness",
                    Thickness=3,
                    Material=_material(asset_id, "subsurface", solid=subsurface),
                ),
            ],
        ),
        _material(asset_id, "rock", solid=rock),
    ]
    empty_queue: list[dict[str, Any]] = []
    if float(config["lake_coverage"]) > 0.0 or float(config["river_strength"]) > 0.0:
        empty_queue.append(
            _node(
                asset_id,
                "SimpleHorizontalMaterialProvider",
                "water-band",
                Type="SimpleHorizontal",
                TopY=0,
                TopBaseHeight="Water",
                BottomY=0,
                BottomBaseHeight="Bedrock",
                Material=_material(asset_id, "water", fluid="Water_Source"),
            )
        )
    empty = _material(asset_id, "empty", solid="Empty")
    empty["$Comment"] = "REQUIRED"
    empty_queue.append(empty)
    return (
        _node(
            asset_id,
            "SolidityMaterialProvider",
            "solidity",
            Type="Solidity",
            Solid=_node(
                asset_id,
                "QueueMaterialProvider",
                "solid-queue",
                Type="Queue",
                Queue=solid_queue,
            ),
            Empty=_node(
                asset_id,
                "QueueMaterialProvider",
                "empty-queue",
                Type="Queue",
                Queue=empty_queue,
            ),
        ),
        (surface, subsurface, rock),
    )


def _prop_runtime(asset_id: str, label: str, assignment: str, *, spacing: float,
                  jitter: float, runtime: int = 0) -> dict[str, Any]:
    return _node(
        asset_id,
        "Runtime",
        label,
        Skip=False,
        Runtime=runtime,
        Positions=_node(
            asset_id,
            "Mesh2DPositions",
            f"{label}-positions",
            Type="Mesh2D",
            Skip=False,
            PointsY=0,
            PointGenerator=_node(
                asset_id,
                "MeshPointGenerator",
                f"{label}-mesh",
                Type="Mesh",
                Jitter=round(max(0.001, min(0.95, jitter)), 6),
                ScaleX=round(max(1.0, spacing), 6),
                ScaleY=round(max(1.0, spacing), 6),
                ScaleZ=round(max(1.0, spacing), 6),
                Seed=f"{asset_id}:{label}",
            ),
        ),
        Assignments=_node(
            asset_id,
            "Imported.Assignments",
            f"{label}-assignment",
            Type="Imported",
            Name=assignment,
        ),
    )


def _props(asset_id: str, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        _prop_runtime(
            asset_id,
            "ground-cover",
            "Plains1_Mountains_Grasses",
            spacing=1.0,
            jitter=0.001,
        )
    ]
    forest = float(config["forest_density"])
    if forest > 0.0:
        spacing = max(
            3.0,
            float(config["tree_spacing"]) / math.sqrt(max(0.08, forest)),
        )
        rows.append(
            _prop_runtime(
                asset_id,
                "forest-trees",
                "Plains1_Mountains_Trees",
                spacing=spacing,
                jitter=0.22 + float(config["world_variation"]) * 0.36,
            )
        )
    return rows


def _tint(config: Mapping[str, Any]) -> str:
    moisture = float(config["moisture"])
    temperature = float(config["temperature"])
    red = round(74 + temperature * 42)
    green = round(118 + moisture * 52 - temperature * 18)
    blue = round(38 + (1.0 - temperature) * 28)
    return f"#{red:02x}{max(0, min(255, green)):02x}{blue:02x}"


def _walk(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _editor_metadata(biome: Mapping[str, Any]) -> dict[str, Any]:
    nodes: dict[str, Any] = {}
    for index, node in enumerate(_walk(biome)):
        node_id = node.get("$NodeId")
        if isinstance(node_id, str):
            nodes[node_id] = {
                "$Position": {
                    "$x": -4200 + (index % 11) * 520,
                    "$y": -5200 + (index // 11) * 260,
                }
            }
    return {
        "$Nodes": nodes,
        "$FloatingNodes": [],
        "$Links": {},
        "$Groups": [],
        "$Comments": [],
        "$WorkspaceID": "HytaleGenerator - Biome",
    }


def compile_procedural_biome(
    shell: Mapping[str, Any],
    *,
    asset_id: str,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a seed-independent synthesized biome and its honest contract."""

    if config.get("generation_mode") != "procedural_genome":
        raise ValueError("native graph compiler requires generation_mode=procedural_genome")
    if not isinstance(shell, Mapping) or shell.get("Terrain", {}).get("Type") != "DAOTerrain":
        raise ValueError("pinned Basic biome shell changed")
    if not asset_id or any(character in asset_id for character in "/\\"):
        raise ValueError("asset_id is unsafe")

    density, weights = _density_graph(asset_id, config)
    material_provider, palette = _material_provider(asset_id, config)
    biome = deepcopy(dict(shell))
    biome["$NodeId"] = _node_id(asset_id, "Biome", "root")
    biome["Name"] = asset_id
    biome["Terrain"] = _node(
        asset_id,
        "Terrain",
        "terrain",
        Type="DAOTerrain",
        Density=density,
    )
    biome["MaterialProvider"] = material_provider
    biome["Props"] = _props(asset_id, config)
    biome["EnvironmentProvider"] = _node(
        asset_id,
        "Constant.EnvironmentProvider",
        "environment",
        Type="Constant",
        Environment="Env_Zone1_Plains",
    )
    biome["TintProvider"] = _node(
        asset_id,
        "Constant.TintProvider",
        "tint",
        Type="Constant",
        Color=_tint(config),
    )
    biome["Tags"] = {"Template": []}
    biome.pop("$Position", None)
    biome.pop("$Title", None)
    biome.pop("$NodeEditorMetadata", None)
    biome["$NodeEditorMetadata"] = _editor_metadata(biome)

    active = [name for name, value in weights.items() if value > 1.0e-9]
    contract: dict[str, Any] = {
        "schema": NATIVE_GRAPH_SCHEMA,
        "version": NATIVE_GRAPH_VERSION,
        "asset_id": asset_id,
        "source": "hyperparameter_distribution_not_named_terrain_preset",
        "biome_shell": {
            "path": PINNED_BASIC_BIOME_PATH,
            "sha256": PINNED_BASIC_BIOME_SHA256,
            "terrain_replaced": True,
            "materials_replaced": True,
            "props_replaced": True,
        },
        "component_weights": {
            name: round(value, 9) for name, value in weights.items()
        },
        "active_components": active,
        "surface_materials": list(palette),
        "density_graph_semantic_sha256": hashlib.sha256(
            _canonical_bytes(density)
        ).hexdigest(),
        "node_count": sum(
            isinstance(node.get("$NodeId"), str) for node in _walk(biome)
        ),
        "mapped_controls": [
            "world_variation",
            "macro_region_count",
            "region_blend",
            "terrain_component_weights",
            "base_height",
            "relief",
            "mountain_scale",
            "roughness",
            "ridge_strength",
            "erosion",
            "pillar_density",
            "pillar_width",
            "pillar_height",
            "canyon_strength",
            "crater_density",
            "crater_depth",
            "water_level",
            "lake_coverage",
            "river_strength",
            "moisture",
            "temperature",
            "snow_line",
            "forest_density",
            "tree_spacing",
            "cave_style",
            "cave_density",
            "cave_scale",
            "cave_verticality",
            "cave_entrances",
        ],
        "capture_or_selection_controls": [
            "world_width",
            "world_depth",
            "route_density",
            "route_width",
            "traversal_bias",
            "structure_variety",
            "spawn_safety_radius",
        ],
        "native_validation": "offline_structural_only",
        "not_claimed": [
            "native_codec_acceptance_before_live_load",
            "semantic_preview_to_native_seed_equivalence",
            "native_distribution_frequencies",
            "connected_route_or_spawn_safety_before_traversal_capture",
            "native_cave_topology_equivalence",
        ],
        "semantic_sha256": "",
    }
    contract["semantic_sha256"] = _semantic_sha256(contract, "semantic_sha256")
    validate_procedural_biome(biome, contract)
    return biome, contract


def validate_procedural_biome(
    biome: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    """Fail closed on graph/receipt drift without claiming native acceptance."""

    if contract.get("schema") != NATIVE_GRAPH_SCHEMA:
        raise ValueError("unsupported procedural native graph contract")
    if contract.get("version") != NATIVE_GRAPH_VERSION:
        raise ValueError("unsupported procedural native graph version")
    if contract.get("semantic_sha256") != _semantic_sha256(
        contract, "semantic_sha256"
    ):
        raise ValueError("procedural native graph contract hash differs")
    terrain = biome.get("Terrain")
    if not isinstance(terrain, Mapping) or terrain.get("Type") != "DAOTerrain":
        raise ValueError("procedural biome lost DAOTerrain root")
    density = terrain.get("Density")
    if not isinstance(density, Mapping):
        raise ValueError("procedural biome has no density graph")
    observed_graph = hashlib.sha256(_canonical_bytes(density)).hexdigest()
    if contract.get("density_graph_semantic_sha256") != observed_graph:
        raise ValueError("procedural biome density graph hash differs")

    node_ids: list[str] = []
    for node in _walk(biome):
        node_id = node.get("$NodeId")
        if isinstance(node_id, str):
            node_ids.append(node_id)
        node_type = node.get("Type")
        if node_type in _ALLOWED_DENSITY_TYPES:
            for value in node.values():
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("procedural graph contains a non-finite number")
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("procedural biome node IDs are not unique")
    if contract.get("node_count") != len(node_ids):
        raise ValueError("procedural biome node count differs")
    metadata = biome.get("$NodeEditorMetadata")
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("$Nodes"), Mapping):
        raise ValueError("procedural biome lacks node editor metadata")
    if set(metadata["$Nodes"]) != set(node_ids):
        raise ValueError("procedural biome editor metadata is incomplete")


def finalize_procedural_biome_contract(
    biome: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Refresh whole-biome counts after the common structure overlay."""

    updated = deepcopy(dict(contract))
    updated["node_count"] = sum(
        isinstance(node.get("$NodeId"), str) for node in _walk(biome)
    )
    updated["semantic_sha256"] = ""
    updated["semantic_sha256"] = _semantic_sha256(updated, "semantic_sha256")
    validate_procedural_biome(biome, updated)
    return updated
