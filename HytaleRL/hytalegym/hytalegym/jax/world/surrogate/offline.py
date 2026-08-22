"""Fail-closed offline assembly of complete surrogate world tiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import operator

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.surrogate.artifact import (
    SurrogateWorldArtifact,
    create_surrogate_world_artifact,
)
from hytalegym.jax.world.surrogate.atlas import surrogate_atlas_from_tiles
from hytalegym.jax.world.surrogate.capabilities import (
    SURROGATE_CAPABILITY_BLOCK_GEOMETRY,
    SURROGATE_CAPABILITY_ENTITY_GEOMETRY,
    SURROGATE_CAPABILITY_ENTITY_PERCEPTION,
    SURROGATE_CAPABILITY_STATEFUL_BLOCKS,
    SURROGATE_CAPABILITY_SWEPT_AABB,
)
from hytalegym.jax.world.surrogate.perception import (
    surrogate_entity_visibility_contract,
)
from hytalegym.jax.world.surrogate.spawn_markers import (
    spawn_marker_catalog_to_jax,
    spawn_marker_selection_contract,
)
from hytalegym.jax.world.surrogate.structures import (
    plan_biome_prefab_tile_from_terrain,
    plan_structure_tile_from_terrain,
)
from hytalegym.jax.world.surrogate.terrain import (
    PRIMARY_ZONE_V1_TERRAIN_SOURCE,
    SurrogateTerrainTile,
    TerrainGeneratorSpec,
    generate_surrogate_terrain,
)
from hytalegym.jax.world.surrogate.traversal import (
    CompiledTraversalGraph,
    compile_standable_traversal_graph,
    traversal_atlas_from_graphs,
)
from hytalegym.jax.world.surrogate.types import (
    SurrogateAtlas,
    SurrogateSpawnMarkerCatalog,
    SurrogateTraversalAtlas,
)
from hytalegym.worldgen.region import CHUNK_SIZE
from hytalegym.worldgen.surrogate import (
    CompiledSemanticPalette,
    CompiledEntitySpawns,
    CompiledSpawnMarkerCatalog,
    ENTITY_KIND_SPAWN_MARKER,
    EntityGeometryProfile,
    GeneratedStructurePlan,
    HytaleAssetArchive,
    LocalBlockSemanticsResolver,
    ResolvedStatefulPrefabOverlay,
    StructurePlacementConfig,
    SurrogateBiomePrefabLayer,
    SurrogateStructureCatalog,
    SurrogateWorldCapacity,
    biome_prefab_layer_contract,
    biome_prefab_source_bundle_sha256,
    compile_structure_plan,
    compile_entity_spawns,
    compile_spawn_marker_catalog,
    entity_type_identity_words,
    resolve_stateful_prefab_overlay,
)


@dataclass(frozen=True)
class OfflineSurrogateWorld:
    """One fully published immutable atlas and its host provenance."""

    atlas: SurrogateAtlas
    traversal: SurrogateTraversalAtlas
    spawn_markers: SurrogateSpawnMarkerCatalog
    artifact: SurrogateWorldArtifact
    terrain: SurrogateTerrainTile
    plans: tuple[GeneratedStructurePlan, ...]
    overlays: tuple[ResolvedStatefulPrefabOverlay | None, ...]
    entities: tuple[CompiledEntitySpawns, ...]
    graphs: tuple[CompiledTraversalGraph, ...]

    @property
    def active_tile_count(self) -> int:
        return len(self.plans)


def build_offline_surrogate_world(
    archive: HytaleAssetArchive,
    terrain_spec: TerrainGeneratorSpec,
    terrain_palette: CompiledSemanticPalette,
    structure_catalog: SurrogateStructureCatalog | None,
    *,
    seed_words: tuple[int, int],
    tile_min_xz: Sequence[tuple[int, int]],
    world_id: int,
    actor_bounds: np.ndarray,
    placement_config: StructurePlacementConfig | None = None,
    biome_prefab_layers: Sequence[SurrogateBiomePrefabLayer] = (),
    ignored_component_types: frozenset[str] = frozenset(),
    maximum_climb_height: float = 1.3,
    maximum_safe_drop_height: float | None = None,
    entity_geometry_profiles: Mapping[str, EntityGeometryProfile] | None = None,
    capacity: SurrogateWorldCapacity | None = None,
    tile_capacity: int | None = None,
) -> OfflineSurrogateWorld:
    """Generate terrain, structures, exact geometry, and traversal before use."""

    if not isinstance(archive, HytaleAssetArchive):
        raise TypeError("archive must be a HytaleAssetArchive")
    if not isinstance(terrain_spec, TerrainGeneratorSpec):
        raise TypeError("terrain_spec must be a TerrainGeneratorSpec")
    if not isinstance(terrain_palette, CompiledSemanticPalette):
        raise TypeError("terrain_palette must be a CompiledSemanticPalette")
    layers = tuple(biome_prefab_layers)
    if any(not isinstance(value, SurrogateBiomePrefabLayer) for value in layers):
        raise TypeError("biome_prefab_layers contains an invalid layer")
    if structure_catalog is None:
        if not layers:
            raise ValueError("a structure catalog or biome prefab layers are required")
        if placement_config is not None:
            raise ValueError("biome prefab layers own their placement policies")
    elif not isinstance(structure_catalog, SurrogateStructureCatalog):
        raise TypeError("structure_catalog must be a SurrogateStructureCatalog or None")
    elif layers:
        raise ValueError("legacy and biome prefab structure modes cannot be mixed")
    if layers:
        _validate_biome_prefab_layers(terrain_spec, layers)
    layout = SurrogateWorldCapacity() if capacity is None else capacity
    if not isinstance(layout, SurrogateWorldCapacity):
        raise TypeError("capacity must be a SurrogateWorldCapacity")
    seeds = tuple(_uint32(value, "seed word") for value in seed_words)
    if len(seeds) != 2:
        raise ValueError("seed_words must contain two uint32 values")
    origins = tuple(_origin(value) for value in tile_min_xz)
    if not origins:
        raise ValueError("at least one tile origin is required")
    if len(set(origins)) != len(origins):
        raise ValueError("tile origins must be unique")
    identifier = _int32(world_id, "world ID")
    if identifier < 0:
        raise ValueError("world ID must be non-negative")
    tiles = (
        len(origins)
        if tile_capacity is None
        else _positive_int(
            tile_capacity,
            "tile capacity",
        )
    )
    if tiles < len(origins):
        raise ValueError("tile capacity is smaller than the requested tile count")

    seed_array = jnp.asarray([seeds] * len(origins), dtype=jnp.uint32)
    origin_array = jnp.asarray(origins, dtype=jnp.int32)
    terrain = generate_surrogate_terrain(
        seed_array,
        origin_array,
        terrain_spec.config,
    )
    if not np.all(np.asarray(jax.device_get(terrain.valid), dtype=np.bool_)):
        raise ValueError("offline terrain generation did not complete")

    if structure_catalog is None:
        plans = tuple(
            plan_biome_prefab_tile_from_terrain(
                layers,
                seeds,
                tile_min_xz=origin,
                terrain_config=terrain_spec.config,
                capacity=layout,
            )
            for origin in origins
        )
    else:
        plans = tuple(
            plan_structure_tile_from_terrain(
                structure_catalog,
                seeds,
                tile_min_xz=origin,
                terrain_config=terrain_spec.config,
                placement_config=placement_config,
                capacity=layout,
            )
            for origin in origins
        )
    resolver = LocalBlockSemanticsResolver(archive)
    overlays: list[ResolvedStatefulPrefabOverlay | None] = []
    for plan in plans:
        raw = compile_structure_plan(
            plan,
            capacity=layout,
            ignored_component_types=ignored_component_types,
        )
        overlays.append(
            None
            if raw is None
            else resolve_stateful_prefab_overlay(
                raw,
                resolver,
                capacity=layout,
            )
        )

    entities = tuple(
        compile_entity_spawns(
            plan,
            geometry_profiles=entity_geometry_profiles,
            capacity=layout,
        )
        for plan in plans
    )
    compiled_spawn_markers = _compile_world_spawn_markers(
        archive,
        entities,
        layout,
    )
    spawn_markers = spawn_marker_catalog_to_jax(compiled_spawn_markers)
    atlas = surrogate_atlas_from_tiles(
        terrain,
        terrain_palette,
        overlays,
        entity_spawns=entities,
        world_ids=[identifier] * len(origins),
        capacity=layout,
        tile_capacity=tiles,
    )
    graphs = tuple(
        compile_standable_traversal_graph(
            atlas,
            index,
            actor_bounds=actor_bounds,
            maximum_climb_height=maximum_climb_height,
            maximum_safe_drop_height=maximum_safe_drop_height,
            capacity=layout,
        )
        for index in range(len(origins))
    )
    graph_slots: list[CompiledTraversalGraph | None] = list(graphs)
    graph_slots.extend([None] * (tiles - len(graphs)))
    traversal = traversal_atlas_from_graphs(atlas, graph_slots)
    artifact = create_surrogate_world_artifact(
        atlas,
        traversal,
        spawn_markers,
        contract=_artifact_contract(
            archive=archive,
            terrain_spec=terrain_spec,
            terrain_palette=terrain_palette,
            structure_catalog=structure_catalog,
            placement_config=placement_config,
            biome_prefab_layers=layers,
            ignored_component_types=ignored_component_types,
            capacity=layout,
            tile_capacity=tiles,
            origins=origins,
            world_id=identifier,
            seed_words=seeds,
            actor_bounds=np.asarray(actor_bounds, dtype=np.float32),
            maximum_climb_height=maximum_climb_height,
            maximum_safe_drop_height=maximum_safe_drop_height,
            entities=entities,
            spawn_markers=compiled_spawn_markers,
        ),
    )
    return OfflineSurrogateWorld(
        atlas=atlas,
        traversal=traversal,
        spawn_markers=spawn_markers,
        artifact=artifact,
        terrain=terrain,
        plans=plans,
        overlays=tuple(overlays),
        entities=entities,
        graphs=graphs,
    )


def _validate_biome_prefab_layers(
    terrain_spec: TerrainGeneratorSpec,
    layers: tuple[SurrogateBiomePrefabLayer, ...],
) -> None:
    names = terrain_spec.biome_names
    sources = terrain_spec.source.biomes
    if (
        terrain_spec.source.mode == PRIMARY_ZONE_V1_TERRAIN_SOURCE
        and len(sources) != len(names)
    ):
        raise ValueError("primary-zone terrain biome sources are incomplete")
    for layer in layers:
        if (
            layer.biome_code >= len(names)
            or names[layer.biome_code] != layer.biome_id
        ):
            raise ValueError("biome prefab layer disagrees with terrain biome order")
        if sources:
            source = sources[layer.biome_code]
            if (
                source.biome_id != layer.biome_id
                or layer.prop_graph_sha256 not in source.prop_graph_sha256
            ):
                raise ValueError("biome prefab source is absent from terrain provenance")


def _compile_world_spawn_markers(
    archive: HytaleAssetArchive,
    entities: tuple[CompiledEntitySpawns, ...],
    capacity: SurrogateWorldCapacity,
) -> CompiledSpawnMarkerCatalog:
    marker_ids: set[str] = set()
    for compiled in entities:
        active = np.asarray(compiled.entity_mask, dtype=np.bool_) & (
            np.asarray(compiled.kind, dtype=np.uint8) == ENTITY_KIND_SPAWN_MARKER
        )
        for code in np.asarray(compiled.type_code, dtype=np.uint16)[active]:
            marker_ids.add(compiled.type_ids[int(code)])
    assets = [archive.load_spawn_marker(marker_id) for marker_id in sorted(marker_ids)]
    return compile_spawn_marker_catalog(
        assets,
        known_role_ids=archive.npc_role_asset_ids() if assets else (),
        marker_capacity=capacity.spawn_marker_capacity,
        choice_capacity=capacity.spawn_marker_choice_capacity,
    )


def _artifact_contract(
    *,
    archive: HytaleAssetArchive,
    terrain_spec: TerrainGeneratorSpec,
    terrain_palette: CompiledSemanticPalette,
    structure_catalog: SurrogateStructureCatalog | None,
    placement_config: StructurePlacementConfig | None,
    biome_prefab_layers: tuple[SurrogateBiomePrefabLayer, ...],
    ignored_component_types: frozenset[str],
    capacity: SurrogateWorldCapacity,
    tile_capacity: int,
    origins: tuple[tuple[int, int], ...],
    world_id: int,
    seed_words: tuple[int, int],
    actor_bounds: np.ndarray,
    maximum_climb_height: float,
    maximum_safe_drop_height: float | None,
    entities: tuple[CompiledEntitySpawns, ...],
    spawn_markers: CompiledSpawnMarkerCatalog,
) -> dict[str, object]:
    type_ids = tuple(
        sorted({type_id for compiled in entities for type_id in compiled.type_ids})
    )
    profiles = {
        profile.type_id: profile
        for compiled in entities
        for profile in compiled.geometry_profiles
    }
    type_palette = []
    for type_id in type_ids:
        profile = profiles.get(type_id)
        type_palette.append(
            {
                "type_id": type_id,
                "identity_words": list(entity_type_identity_words(type_id)),
                "geometry_supported": profile is not None,
                "collidable": False if profile is None else profile.collidable,
                "blocks_los": False if profile is None else profile.blocks_los,
                "local_bounds": (
                    [0.0] * 6 if profile is None else list(profile.local_bounds)
                ),
                "line_of_sight_offset": (
                    None
                    if profile is None or profile.line_of_sight_offset is None
                    else list(profile.line_of_sight_offset)
                ),
                "provenance_sha256": (
                    None if profile is None else profile.provenance_sha256
                ),
                "native_certified": (
                    False if profile is None else profile.native_certified
                ),
            }
        )
    structure_catalogs = (
        (structure_catalog,)
        if structure_catalog is not None
        else tuple(layer.catalog for layer in biome_prefab_layers)
    )
    prefab_sources = sorted(
        {
            (
                template.provenance.entry_path,
                template.provenance.content_sha256,
            )
            for catalog in structure_catalogs
            for family in catalog.families
            for template in family.templates
        }
    )
    stat = archive.path.stat()
    return {
        "capacity": asdict(capacity),
        "tile_capacity": tile_capacity,
        "active_tile_count": len(origins),
        "world_id": world_id,
        "seed_words": list(seed_words),
        "tile_min_xz": [list(origin) for origin in origins],
        "actor_profile": {
            "local_bounds": actor_bounds.tolist(),
            "maximum_climb_height": float(maximum_climb_height),
            "maximum_safe_drop_height": (
                None
                if maximum_safe_drop_height is None
                else float(maximum_safe_drop_height)
            ),
        },
        "terrain": {
            "profile_sha256": terrain_spec.profile_sha256,
            "biome_names": list(terrain_spec.biome_names),
            "config_sha256": _tree_sha256(terrain_spec.config),
            "semantic_palette_sha256": _palette_sha256(terrain_palette),
            "native_seed_equivalent": False,
            "source": asdict(terrain_spec.source),
        },
        "structures": _structure_contract(
            structure_catalog,
            placement_config,
            biome_prefab_layers,
            prefab_sources,
            ignored_component_types,
        ),
        "entity_type_palette": type_palette,
        "spawn_markers": {
            "selection_contract": spawn_marker_selection_contract(),
            "marker_ids": list(spawn_markers.marker_ids),
            "asset_sha256": list(spawn_markers.asset_sha256),
            "role_ids": [list(choices) for choices in spawn_markers.role_ids],
        },
        "source_assets": {
            "archive_name": archive.path.name,
            "archive_bytes": stat.st_size,
        },
        "capabilities": {
            "block_geometry": SURROGATE_CAPABILITY_BLOCK_GEOMETRY,
            "stateful_blocks": SURROGATE_CAPABILITY_STATEFUL_BLOCKS,
            "entity_geometry": SURROGATE_CAPABILITY_ENTITY_GEOMETRY,
            "entity_perception": SURROGATE_CAPABILITY_ENTITY_PERCEPTION,
            "swept_aabb": SURROGATE_CAPABILITY_SWEPT_AABB,
        },
        "entity_visibility_contract": surrogate_entity_visibility_contract(),
        "unsupported_mechanics": [
            "authored_biome_prop_execution",
            "entity_behavior",
            "fluid_dynamics",
            "live_block_mutation_except_doors",
            "asynchronous_publication",
            "native_worldgen_equivalence",
        ],
        "certification": {
            "array_behavior": "requires_passing_checked_in_tests",
            "local_assets": "requires_hash_locked_installed_smoke",
            "native_mechanics": "uncertified",
            "native_distribution": "uncertified",
        },
    }


def _structure_contract(
    catalog: SurrogateStructureCatalog | None,
    placement: StructurePlacementConfig | None,
    layers: tuple[SurrogateBiomePrefabLayer, ...],
    prefab_sources: list[tuple[str, str]],
    ignored_component_types: frozenset[str],
) -> dict[str, object]:
    sources = [
        {"entry_path": path, "sha256": digest}
        for path, digest in prefab_sources
    ]
    if catalog is None:
        return {
            "mode": "surrogate_biome_prefab_layers_v1",
            "source_bundle_sha256": biome_prefab_source_bundle_sha256(layers),
            "layers": [
                biome_prefab_layer_contract(layer)
                for layer in sorted(
                    layers,
                    key=lambda value: (
                        value.biome_code,
                        value.prop_graph_sha256,
                        value.assignment_id,
                    ),
                )
            ],
            "prefab_sources": sources,
            "ignored_component_types": sorted(ignored_component_types),
        }
    settings = StructurePlacementConfig() if placement is None else placement
    allowed = settings.allowed_biome_codes
    return {
        "assignment_entry_path": catalog.assignment_provenance.entry_path,
        "assignment_sha256": catalog.assignment_provenance.content_sha256,
        "prefab_sources": sources,
        "placement": {
            "grid_spacing": settings.grid_spacing,
            "jitter": settings.jitter,
            "placement_probability": settings.placement_probability,
            "salt": settings.salt,
            "allowed_biome_codes": None if allowed is None else sorted(allowed),
        },
        "ignored_component_types": sorted(ignored_component_types),
    }


def _tree_sha256(tree: tuple[jax.Array, ...]) -> str:
    digest = hashlib.sha256()
    for name, value in zip(tree._fields, tree, strict=True):
        digest.update(name.encode())
        _update_digest(digest, value)
    return digest.hexdigest()


def _palette_sha256(palette: CompiledSemanticPalette) -> str:
    digest = hashlib.sha256(
        json.dumps(
            palette.reference_keys,
            separators=(",", ":"),
        ).encode()
    )
    for name in (
        "cell_flags",
        "cell_shape_index",
        "cell_fluid_level",
        "cell_support",
        "cell_block_damage",
        "cell_fluid_damage",
        "cell_movement",
        "cell_fluid_movement",
        "shape_boxes",
        "shape_box_mask",
        "reference_cell_code",
    ):
        digest.update(name.encode())
        _update_digest(digest, getattr(palette, name))
    return digest.hexdigest()


def _update_digest(digest: object, value: object) -> None:
    array = np.ascontiguousarray(jax.device_get(value))
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    digest.update(array.tobytes())


def _origin(value: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("tile origins must be 2-tuples")
    result = (_int32(value[0], "tile origin"), _int32(value[1], "tile origin"))
    if any(coordinate % CHUNK_SIZE for coordinate in result):
        raise ValueError("tile origins must be chunk aligned")
    return result


def _uint32(value: int, label: str) -> int:
    result = _integer(value, label)
    if not 0 <= result <= np.iinfo(np.uint32).max:
        raise ValueError(f"{label} exceeds uint32")
    return result


def _int32(value: int, label: str) -> int:
    result = _integer(value, label)
    if not np.iinfo(np.int32).min <= result <= np.iinfo(np.int32).max:
        raise ValueError(f"{label} exceeds int32")
    return result


def _positive_int(value: int, label: str) -> int:
    result = _integer(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _integer(value: int, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error


__all__ = ["OfflineSurrogateWorld", "build_offline_surrogate_world"]
