"""Leaf helpers extracted verbatim from assets.py."""

from collections.abc import Iterable
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile, ZipInfo
from ._decode import (  # noqa: F401  (re-exported: callers unchanged)
    HytaleAssetError,
    PrefabEntity,
    SpawnMarkerConfiguration,
    _JAVA_DURATION,
    _PREFAB_ENTITY_FIELDS,
    _SPAWN_MARKER_CONFIGURATION_FIELDS,
    _array,
    _asset_id,
    _component_types,
    _decimal_constants,
    _entity_kind_and_type,
    _entry_path,
    _finite_number,
    _generator_objects,
    _integer,
    _native_glob_matches,
    _nonempty_selector,
    _nonempty_string,
    _object,
    _optional_asset_id,
    _optional_nonnegative_int,
    _positive_java_duration_seconds,
    _positive_number,
    _prefab_entity,
    _relative_asset_reference,
    _rotation3,
    _spawn_marker_configuration,
    _vector3,
)
from ._types import (  # noqa: F401  (re-exported: callers unchanged)
    AssetProvenance,
    BiomeGeneratorAsset,
    BiomePropAssignment,
    BiomeRange,
    BlockSetAsset,
    PrefabBlock,
    PrefabTemplate,
    SpawnMarkerAsset,
    WorldStructureProfile,
    _filler_value,
    _support_value,
)
from ._fields import (  # noqa: F401  (re-exported: callers unchanged)
    AssignmentPrefabReference,
    CollisionBox,
    DoorGeometryAsset,
    DoorGeometryVariant,
    GeneratorAssignmentIndex,
    ResolvedBlockSet,
    _BIOME_FIELDS,
    _BLOCK_SET_FIELDS,
    _BLOCK_SET_RUNTIME_SELECTOR_FIELDS,
    _PREFAB_BLOCK_FIELDS,
    _PREFAB_TOP_LEVEL_FIELDS,
    _SPAWN_MARKER_FIELDS,
    _WORLD_STRUCTURE_FIELDS,
    _optional_bool,
)


WORLD_STRUCTURE_PREFIX = "Server/HytaleGenerator/WorldStructures/"


BIOME_PREFIX = "Server/HytaleGenerator/Biomes/"


ASSIGNMENT_PREFIX = "Server/HytaleGenerator/Assignments/"


PREFAB_PREFIX = "Server/Prefabs/"


WORLD_PREFAB_PREFIX = "Server/World/"


HITBOX_PREFIX = "Server/Item/Block/Hitboxes/"


ITEM_PREFIX = "Server/Item/Items/"


ITEM_DROP_LIST_PREFIX = "Server/Drops/"


UNARMED_GATHERING_PREFIX = "Server/Item/Unarmed/Gathering/"


BLOCK_SET_PREFIX = "Server/Item/Block/Sets/"


NPC_ROLE_PREFIX = "Server/NPC/Roles/"


SPAWN_MARKER_PREFIX = "Server/NPC/Spawn/Markers/"


DEFAULT_JSON_BYTE_CAPACITY = 64 * 1024 * 1024


DEFAULT_GENERATOR_NODE_CAPACITY = 65_536


DEFAULT_PREFAB_BLOCK_CAPACITY = 1_000_000


DEFAULT_PREFAB_ENTITY_CAPACITY = 65_536


DEFAULT_PREFAB_REFERENCE_CAPACITY = 65_536


DEFAULT_SPAWN_MARKER_CONFIGURATION_CAPACITY = 256


class HytaleAssetArchive:
    """Validated random access to one local ``Assets.zip`` without extraction."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Hytale asset archive not found: {self.path}")
        try:
            self._archive = ZipFile(self.path)
        except BadZipFile as error:
            raise HytaleAssetError("Hytale asset archive is not a valid ZIP") from error
        self._entries: dict[str, ZipInfo] = {}
        self._duplicates: set[str] = set()
        try:
            for info in self._archive.infolist():
                if info.is_dir():
                    continue
                name = _entry_path(info.filename)
                if name in self._entries:
                    self._duplicates.add(name)
                else:
                    self._entries[name] = info
        except Exception:
            self._archive.close()
            raise

    def __enter__(self) -> "HytaleAssetArchive":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self._archive.close()

    def entry_paths(
        self,
        *,
        prefix: str = "",
        suffix: str = "",
    ) -> tuple[str, ...]:
        normalized_prefix = _entry_prefix(prefix)
        return tuple(
            name
            for name in sorted(self._entries)
            if name.startswith(normalized_prefix) and name.endswith(suffix)
        )

    def load_item_asset(
        self,
        asset_id: str,
        *,
        json_byte_capacity: int = DEFAULT_JSON_BYTE_CAPACITY,
    ) -> tuple[dict[str, Any], AssetProvenance]:
        """Load one uniquely named item asset by its runtime asset ID."""

        identifier = _asset_id(asset_id, "item asset ID")
        suffix = f"/{identifier}.json"
        matches = self.entry_paths(prefix=ITEM_PREFIX, suffix=suffix)
        if len(matches) != 1:
            raise HytaleAssetError(
                f"item asset {identifier!r} resolved to {len(matches)} entries"
            )
        return self._read_json(matches[0], json_byte_capacity)

    def item_asset_ids(self) -> tuple[str, ...]:
        """Return unique local item IDs without loading every asset body."""

        result: dict[str, str] = {}
        for path in self.entry_paths(prefix=ITEM_PREFIX, suffix=".json"):
            asset_id = Path(path).stem
            previous = result.setdefault(asset_id, path)
            if previous != path:
                raise HytaleAssetError(
                    f"item asset {asset_id!r} resolves to multiple entries"
                )
        return tuple(sorted(result))

    def item_drop_list_asset_ids(self) -> tuple[str, ...]:
        """Return the unique filename-keyed native item-drop-list IDs."""

        result: dict[str, str] = {}
        for path in self.entry_paths(
            prefix=ITEM_DROP_LIST_PREFIX,
            suffix=".json",
        ):
            asset_id = Path(path).stem
            previous = result.setdefault(asset_id, path)
            if previous != path:
                raise HytaleAssetError(
                    f"item drop list {asset_id!r} resolves to multiple entries"
                )
        return tuple(sorted(result))

    def load_item_drop_list_asset(
        self,
        asset_id: str,
        *,
        json_byte_capacity: int = DEFAULT_JSON_BYTE_CAPACITY,
    ) -> tuple[dict[str, Any], AssetProvenance]:
        """Load one filename-keyed native item-drop-list program."""

        identifier = _asset_id(asset_id, "item drop list ID")
        suffix = f"/{identifier}.json"
        matches = self.entry_paths(
            prefix=ITEM_DROP_LIST_PREFIX,
            suffix=suffix,
        )
        if len(matches) != 1:
            raise HytaleAssetError(
                f"item drop list {identifier!r} resolved to "
                f"{len(matches)} entries"
            )
        return self._read_json(matches[0], json_byte_capacity)

    def load_unarmed_gathering_asset(
        self,
        gather_type: str,
        *,
        json_byte_capacity: int = DEFAULT_JSON_BYTE_CAPACITY,
    ) -> tuple[dict[str, Any], AssetProvenance]:
        """Load one exact unarmed gather-type default."""

        identifier = _asset_id(gather_type, "unarmed gather type")
        path = f"{UNARMED_GATHERING_PREFIX}{identifier}.json"
        return self._read_json(path, json_byte_capacity)

    def load_npc_role_asset(
        self,
        entry_path: str,
        *,
        json_byte_capacity: int = DEFAULT_JSON_BYTE_CAPACITY,
    ) -> tuple[dict[str, Any], AssetProvenance]:
        """Load one role document for provenance-backed schema inspection."""

        path = _required_asset_path(
            entry_path,
            prefix=NPC_ROLE_PREFIX,
            suffix=".json",
        )
        return self._read_json(path, json_byte_capacity)

    def npc_role_asset_ids(self) -> tuple[str, ...]:
        """Return installed role registry IDs without loading role bodies."""

        return tuple(
            sorted(
                {
                    Path(path).stem
                    for path in self.entry_paths(
                        prefix=NPC_ROLE_PREFIX,
                        suffix=".json",
                    )
                }
            )
        )

    def load_spawn_marker(
        self,
        asset_id: str,
        *,
        configuration_capacity: int = DEFAULT_SPAWN_MARKER_CONFIGURATION_CAPACITY,
        json_byte_capacity: int = DEFAULT_JSON_BYTE_CAPACITY,
    ) -> SpawnMarkerAsset:
        """Load one exact 0.5.7 spawn-marker asset or fail closed."""

        identifier = _asset_id(asset_id, "spawn marker ID")
        suffix = f"/{identifier}.json"
        matches = self.entry_paths(prefix=SPAWN_MARKER_PREFIX, suffix=suffix)
        if len(matches) != 1:
            raise HytaleAssetError(
                f"spawn marker {identifier!r} resolved to {len(matches)} entries"
            )
        data, provenance = self._read_json(matches[0], json_byte_capacity)
        unknown = sorted(set(data).difference(_SPAWN_MARKER_FIELDS))
        if unknown:
            raise HytaleAssetError(
                f"spawn marker {identifier!r} has unsupported fields: "
                + ", ".join(unknown)
            )
        realtime = _optional_bool(
            data.get("RealtimeRespawn"),
            "spawn marker RealtimeRespawn",
            default=False,
        )
        raw_configurations = _array(data.get("NPCs"), "spawn marker NPCs")
        capacity = _positive_int(
            configuration_capacity,
            "spawn marker configuration capacity",
        )
        if not raw_configurations:
            raise HytaleAssetError("spawn marker must contain at least one NPC choice")
        if len(raw_configurations) > capacity:
            raise HytaleAssetError(
                f"spawn marker has {len(raw_configurations)} choices, "
                f"capacity is {capacity}"
            )
        configurations = tuple(
            _spawn_marker_configuration(value, index, realtime)
            for index, value in enumerate(raw_configurations)
        )
        return SpawnMarkerAsset(
            asset_id=identifier,
            model_id=_optional_asset_id(data.get("Model"), "spawn marker Model"),
            configurations=configurations,
            exclusion_radius=_nonnegative_number(
                data.get("ExclusionRadius", 0.0),
                "spawn marker ExclusionRadius",
            ),
            maximum_drop_height=_positive_number(
                data.get("MaxDropHeight", 2.0),
                "spawn marker MaxDropHeight",
            ),
            realtime_respawn=realtime,
            manual_trigger=_optional_bool(
                data.get("ManualTrigger"),
                "spawn marker ManualTrigger",
                default=False,
            ),
            deactivation_distance=_positive_number(
                data.get("DeactivationDistance", 40.0),
                "spawn marker DeactivationDistance",
            ),
            deactivation_seconds=_positive_number(
                data.get("DeactivationTime", 5.0),
                "spawn marker DeactivationTime",
            ),
            provenance=provenance,
        )

    def load_block_set(
        self,
        asset_id: str,
        *,
        json_byte_capacity: int = DEFAULT_JSON_BYTE_CAPACITY,
    ) -> BlockSetAsset:
        """Load selectors from one exact local BlockSet asset."""

        identifier = _asset_id(asset_id, "block set ID")
        path = f"{BLOCK_SET_PREFIX}{identifier}.json"
        data, provenance = self._read_json(path, json_byte_capacity)
        runtime_selector_fields = tuple(
            field
            for field in _BLOCK_SET_RUNTIME_SELECTOR_FIELDS
            if _nonempty_selector(data.get(field), f"block set {field}")
        )
        return BlockSetAsset(
            asset_id=identifier,
            parent=_optional_asset_id(data.get("Parent"), "block set Parent"),
            include_all=_optional_bool(
                data.get("IncludeAll"),
                "block set IncludeAll",
                default=False,
            ),
            include_block_types=_string_array(
                data.get("IncludeBlockTypes", []),
                "block set IncludeBlockTypes",
            ),
            exclude_block_types=_string_array(
                data.get("ExcludeBlockTypes", []),
                "block set ExcludeBlockTypes",
            ),
            runtime_selector_fields=runtime_selector_fields,
            unsupported_fields=tuple(sorted(set(data).difference(_BLOCK_SET_FIELDS))),
            provenance=provenance,
        )

    def resolve_block_set_block_types(
        self,
        asset_id: str,
        block_type_ids: Iterable[str],
        *,
        inheritance_capacity: int = 32,
    ) -> ResolvedBlockSet:
        """Expand exact native ``*``/``?`` block-name selectors or fail closed."""

        identifier = _asset_id(asset_id, "block set ID")
        if isinstance(block_type_ids, (str, bytes)):
            raise TypeError("block_type_ids must be an iterable of asset IDs")
        candidates = tuple(
            sorted({_asset_id(value, "block type ID") for value in block_type_ids})
        )
        capacity = _positive_int(
            inheritance_capacity,
            "block set inheritance capacity",
        )

        def resolve(
            current_id: str,
            stack: tuple[str, ...],
        ) -> tuple[set[str], tuple[str, ...], tuple[AssetProvenance, ...]]:
            if current_id in stack:
                raise HytaleAssetError(
                    "block set inheritance cycle: "
                    + " -> ".join(stack + (current_id,))
                )
            if len(stack) >= capacity:
                raise HytaleAssetError("block set inheritance exceeds capacity")
            asset = self.load_block_set(current_id)
            if asset.runtime_selector_fields:
                fields = ", ".join(asset.runtime_selector_fields)
                raise HytaleAssetError(
                    f"block set {current_id!r} requires runtime metadata: {fields}"
                )
            if asset.unsupported_fields:
                fields = ", ".join(asset.unsupported_fields)
                raise HytaleAssetError(
                    f"block set {current_id!r} has unsupported fields: {fields}"
                )
            selected: set[str] = set()
            chain: tuple[str, ...] = ()
            provenance: tuple[AssetProvenance, ...] = ()
            if asset.parent is not None:
                selected, chain, provenance = resolve(
                    asset.parent,
                    stack + (current_id,),
                )
            if asset.include_all:
                selected.update(candidates)
            for pattern in asset.include_block_types:
                selected.update(
                    candidate
                    for candidate in candidates
                    if _native_glob_matches(pattern, candidate)
                )
            for pattern in asset.exclude_block_types:
                selected.difference_update(
                    candidate
                    for candidate in candidates
                    if _native_glob_matches(pattern, candidate)
                )
            return (
                selected,
                chain + (current_id,),
                provenance + (asset.provenance,),
            )

        selected, chain, provenance = resolve(identifier, ())
        return ResolvedBlockSet(
            asset_id=identifier,
            block_type_ids=tuple(sorted(selected)),
            inheritance_chain=chain,
            provenance=provenance,
        )

    def load_prefab(
        self,
        entry_path: str,
        *,
        block_capacity: int = DEFAULT_PREFAB_BLOCK_CAPACITY,
        entity_capacity: int = DEFAULT_PREFAB_ENTITY_CAPACITY,
        json_byte_capacity: int = DEFAULT_JSON_BYTE_CAPACITY,
    ) -> PrefabTemplate:
        path = _entry_path(entry_path)
        if not path.endswith(".prefab.json") or not path.startswith(
            (PREFAB_PREFIX, WORLD_PREFAB_PREFIX)
        ):
            raise HytaleAssetError(
                "prefab entry must be under Server/Prefabs or Server/World"
            )
        data, provenance = self._read_json(path, json_byte_capacity)
        capacity = _positive_int(block_capacity, "prefab block capacity")
        maximum_entities = _positive_int(
            entity_capacity,
            "prefab entity capacity",
        )
        version = _integer(data.get("version"), "prefab version")
        block_id_version = _integer(
            data.get("blockIdVersion"),
            "prefab block ID version",
        )
        anchor = tuple(
            _int32(data.get(name), f"prefab {name}")
            for name in ("anchorX", "anchorY", "anchorZ")
        )
        source_blocks = _array(data.get("blocks"), "prefab blocks")
        source_entities = _array(data.get("entities", []), "prefab entities")
        if not source_blocks and not source_entities:
            raise HytaleAssetError("prefab contains no blocks or entities")
        if len(source_blocks) > capacity:
            raise HytaleAssetError(
                f"prefab has {len(source_blocks)} blocks, capacity is {capacity}"
            )
        if len(source_entities) > maximum_entities:
            raise HytaleAssetError(
                f"prefab has {len(source_entities)} entities, "
                f"capacity is {maximum_entities}"
            )

        blocks: list[PrefabBlock] = []
        positions: set[tuple[int, int, int]] = set()
        component_types: set[str] = set()
        unsupported_block_fields: set[str] = set()
        for index, value in enumerate(source_blocks):
            block = _object(value, f"prefab block {index}")
            unsupported_block_fields.update(set(block).difference(_PREFAB_BLOCK_FIELDS))
            position = tuple(
                _int32(block.get(name), f"prefab block {index}.{name}")
                for name in ("x", "y", "z")
            )
            if position in positions:
                raise HytaleAssetError(
                    f"prefab contains duplicate block position {position}"
                )
            positions.add(position)
            asset_id = _nonempty_string(
                block.get("name"),
                f"prefab block {index}.name",
            )
            rotation = _optional_nonnegative_int(
                block.get("rotation"),
                f"prefab block {index}.rotation",
            )
            filler = _filler_value(
                block.get("filler"),
                f"prefab block {index}.filler",
            )
            support = _support_value(
                block.get("support"),
                f"prefab block {index}.support",
            )
            block_component_types = _component_types(
                block.get("components"),
                f"prefab block {index}.components",
            )
            component_types.update(block_component_types)
            blocks.append(
                PrefabBlock(
                    position=position,
                    asset_id=asset_id,
                    rotation=rotation,
                    filler=filler,
                    component_types=block_component_types,
                    support=support,
                )
            )

        entities = tuple(
            _prefab_entity(value, index) for index, value in enumerate(source_entities)
        )
        bound_positions = positions or {
            tuple(math.floor(value) for value in entity.position) for entity in entities
        }
        bounds_min = tuple(
            min(position[axis] for position in bound_positions) for axis in range(3)
        )
        bounds_max = tuple(
            max(position[axis] for position in bound_positions) for axis in range(3)
        )
        return PrefabTemplate(
            version=version,
            block_id_version=block_id_version,
            anchor=anchor,
            blocks=tuple(blocks),
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            component_types=tuple(sorted(component_types)),
            unsupported_top_level_fields=tuple(
                sorted(set(data).difference(_PREFAB_TOP_LEVEL_FIELDS))
            ),
            unsupported_block_fields=tuple(sorted(unsupported_block_fields)),
            provenance=provenance,
            entities=entities,
        )

    def load_world_structure_profile(
        self,
        entry_path: str,
        *,
        json_byte_capacity: int = DEFAULT_JSON_BYTE_CAPACITY,
    ) -> WorldStructureProfile:
        path = _required_asset_path(
            entry_path,
            prefix=WORLD_STRUCTURE_PREFIX,
            suffix=".json",
        )
        data, provenance = self._read_json(path, json_byte_capacity)
        profile_type = _nonempty_string(
            data.get("Type"),
            "world structure type",
        )
        if profile_type != "NoiseRange":
            raise HytaleAssetError(f"unsupported world structure type {profile_type!r}")
        biomes: list[BiomeRange] = []
        for index, value in enumerate(
            _array(data.get("Biomes"), "world structure biomes")
        ):
            biome = _object(value, f"biome range {index}")
            minimum = _finite_number(
                biome.get("Min"),
                f"biome range {index}.Min",
            )
            maximum = _finite_number(
                biome.get("Max"),
                f"biome range {index}.Max",
            )
            if minimum >= maximum:
                raise HytaleAssetError(f"biome range {index} must have Min < Max")
            biomes.append(
                BiomeRange(
                    biome=_nonempty_string(
                        biome.get("Biome"),
                        f"biome range {index}.Biome",
                    ),
                    minimum=minimum,
                    maximum=maximum,
                )
            )
        ordered = sorted(biomes, key=lambda item: item.minimum)
        for left, right in zip(ordered, ordered[1:], strict=False):
            if left.maximum > right.minimum:
                raise HytaleAssetError(
                    f"biome ranges {left.biome!r} and {right.biome!r} overlap"
                )
        density = data.get("Density")
        density_source = None
        if density is not None:
            density_object = _object(density, "world structure density")
            if density_object.get("Type") == "Imported":
                density_source = _nonempty_string(
                    density_object.get("Name"),
                    "world structure density name",
                )
        constants = _decimal_constants(data.get("Framework", []))
        return WorldStructureProfile(
            profile_type=profile_type,
            biomes=tuple(biomes),
            default_biome=_nonempty_string(
                data.get("DefaultBiome"),
                "default biome",
            ),
            default_transition_distance=_finite_number(
                data.get("DefaultTransitionDistance"),
                "default transition distance",
            ),
            maximum_biome_edge_distance=_finite_number(
                data.get("MaxBiomeEdgeDistance"),
                "maximum biome edge distance",
            ),
            density_source=density_source,
            decimal_constants=constants,
            unsupported_fields=tuple(
                sorted(set(data).difference(_WORLD_STRUCTURE_FIELDS))
            ),
            provenance=provenance,
        )

    def load_biome_generator(
        self,
        entry_path: str,
        *,
        node_capacity: int = DEFAULT_GENERATOR_NODE_CAPACITY,
        json_byte_capacity: int = DEFAULT_JSON_BYTE_CAPACITY,
    ) -> BiomeGeneratorAsset:
        """Summarize exact local biome inputs without executing its density AST."""

        path = _required_asset_path(
            entry_path,
            prefix=BIOME_PREFIX,
            suffix=".json",
        )
        data, provenance = self._read_json(path, json_byte_capacity)
        capacity = _positive_int(node_capacity, "generator node capacity")
        terrain = _object(data.get("Terrain"), "biome Terrain")
        material_provider = _object(
            data.get("MaterialProvider"),
            "biome MaterialProvider",
        )

        terrain_node_types: set[str] = set()
        terrain_imports: set[str] = set()
        terrain_noise_scales: set[float] = set()
        for node in _generator_objects(terrain, capacity, "biome Terrain"):
            node_type = node.get("Type")
            if node_type is not None:
                node_type = _nonempty_string(node_type, "generator node Type")
                terrain_node_types.add(node_type)
            if node_type == "Imported":
                terrain_imports.add(
                    _nonempty_string(
                        node.get("Name"),
                        "imported terrain node Name",
                    )
                )
            if (
                node_type in {"SimplexNoise", "SimplexNoise2D", "SimplexNoise3D"}
                and "Scale" in node
            ):
                terrain_noise_scales.add(
                    _positive_number(
                        node.get("Scale"),
                        f"{node_type} Scale",
                    )
                )

        solid_materials: set[str] = set()
        fluid_materials: set[str] = set()
        for node in _generator_objects(
            material_provider,
            capacity,
            "biome MaterialProvider",
        ):
            solid = node.get("Solid")
            if isinstance(solid, str) and solid not in {"", "Empty"}:
                solid_materials.add(_asset_id(solid, "solid material ID"))
            fluid = node.get("Fluid")
            if isinstance(fluid, str) and fluid not in {"", "Empty"}:
                fluid_materials.add(_asset_id(fluid, "fluid material ID"))
        if not solid_materials:
            raise HytaleAssetError("biome MaterialProvider contains no solids")

        prop_assignments: list[BiomePropAssignment] = []
        for index, value in enumerate(_array(data.get("Props", []), "biome Props")):
            prop = _object(value, f"biome prop {index}")
            source_fields = tuple(
                name for name in ("Assignments", "PropDistribution") if name in prop
            )
            if len(source_fields) != 1:
                raise HytaleAssetError(
                    f"biome prop {index} must contain exactly one prop graph"
                )
            source_field = source_fields[0]
            graph = prop[source_field]
            if not isinstance(graph, (dict, list)):
                raise HytaleAssetError(
                    f"biome prop {index}.{source_field} must be an object or array"
                )
            nodes = _generator_objects(
                graph,
                capacity,
                f"biome prop {index}.{source_field}",
            )
            imported = tuple(
                _nonempty_string(
                    node.get("Name"),
                    f"biome prop {index} imported assignment Name",
                )
                for node in nodes
                if node.get("Type") == "Imported"
            )
            root_type = (
                "List"
                if isinstance(graph, list)
                else _nonempty_string(
                    graph.get("Type"),
                    f"biome prop {index}.{source_field}.Type",
                )
            )
            encoded_graph = json.dumps(
                graph,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            prop_assignments.append(
                BiomePropAssignment(
                    assignment=imported[0] if len(imported) == 1 else None,
                    skipped=_optional_bool(
                        prop.get("Skip"),
                        f"biome prop {index}.Skip",
                        default=False,
                    ),
                    runtime=_optional_nonnegative_int(
                        prop.get("Runtime"),
                        f"biome prop {index}.Runtime",
                    ),
                    source_field=source_field,
                    root_type=root_type,
                    imported_assignments=imported,
                    graph_sha256=hashlib.sha256(encoded_graph).hexdigest(),
                )
            )

        return BiomeGeneratorAsset(
            biome_id=Path(path).stem,
            display_name=_optional_string(data.get("Name"), "biome Name"),
            terrain_type=_nonempty_string(
                terrain.get("Type"),
                "biome Terrain.Type",
            ),
            terrain_node_types=tuple(sorted(terrain_node_types)),
            terrain_imports=tuple(sorted(terrain_imports)),
            terrain_noise_scales=tuple(sorted(terrain_noise_scales)),
            solid_materials=tuple(sorted(solid_materials)),
            fluid_materials=tuple(sorted(fluid_materials)),
            prop_assignments=tuple(prop_assignments),
            unsupported_top_level_fields=tuple(
                sorted(set(data).difference(_BIOME_FIELDS))
            ),
            provenance=provenance,
        )

    def load_assignment_index(
        self,
        entry_path: str,
        *,
        reference_capacity: int = DEFAULT_PREFAB_REFERENCE_CAPACITY,
        json_byte_capacity: int = DEFAULT_JSON_BYTE_CAPACITY,
    ) -> GeneratorAssignmentIndex:
        path = _required_asset_path(
            entry_path,
            prefix=ASSIGNMENT_PREFIX,
            suffix=".json",
        )
        data, provenance = self._read_json(path, json_byte_capacity)
        capacity = _positive_int(
            reference_capacity,
            "prefab reference capacity",
        )
        references: list[AssignmentPrefabReference] = []

        def visit(value: Any, pointer: str) -> None:
            if isinstance(value, dict):
                weighted = value.get("WeightedPrefabPaths")
                if weighted is not None:
                    load_entities = _optional_bool(
                        value.get("LoadEntities"),
                        f"{pointer}/LoadEntities",
                        default=False,
                    )
                    for index, item in enumerate(
                        _array(weighted, f"{pointer}/WeightedPrefabPaths")
                    ):
                        reference = _object(
                            item,
                            f"{pointer}/WeightedPrefabPaths/{index}",
                        )
                        references.append(
                            AssignmentPrefabReference(
                                path=_relative_asset_reference(
                                    reference.get("Path"),
                                    f"{pointer}/WeightedPrefabPaths/{index}/Path",
                                ),
                                weight=_positive_number(
                                    reference.get("Weight"),
                                    f"{pointer}/WeightedPrefabPaths/{index}/Weight",
                                ),
                                load_entities=load_entities,
                                json_pointer=(f"{pointer}/WeightedPrefabPaths/{index}"),
                            )
                        )
                        if len(references) > capacity:
                            raise HytaleAssetError(
                                "generator assignment exceeds prefab "
                                f"reference capacity {capacity}"
                            )
                for key, item in value.items():
                    visit(item, f"{pointer}/{_json_pointer_token(str(key))}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    visit(item, f"{pointer}/{index}")

        visit(data, "")
        return GeneratorAssignmentIndex(
            root_type=_nonempty_string(
                data.get("Type"),
                "generator assignment root type",
            ),
            export_name=_optional_string(data.get("ExportAs"), "ExportAs"),
            prefab_references=tuple(references),
            provenance=provenance,
        )

    def load_door_geometry(
        self,
        entry_path: str,
        *,
        json_byte_capacity: int = DEFAULT_JSON_BYTE_CAPACITY,
    ) -> DoorGeometryAsset:
        path = _required_asset_path(
            entry_path,
            prefix=ITEM_PREFIX,
            suffix=".json",
        )
        data, item_provenance = self._read_json(path, json_byte_capacity)
        block = _object(data.get("BlockType"), "door BlockType")
        if block.get("IsDoor") is not True:
            raise HytaleAssetError("door asset must declare BlockType.IsDoor=true")
        default_hitbox = _nonempty_string(
            block.get("HitboxType"),
            "door default hitbox",
        )
        definitions = _object(
            _object(block.get("State"), "door State").get("Definitions"),
            "door State.Definitions",
        )
        state_hitboxes = [("default", default_hitbox)]
        for state_name, state_value in sorted(definitions.items()):
            state = _object(
                state_value,
                f"door state {state_name}",
            )
            hitbox = state.get("HitboxType", default_hitbox)
            state_hitboxes.append(
                (
                    _nonempty_string(state_name, "door state name"),
                    _nonempty_string(
                        hitbox,
                        f"door state {state_name}.HitboxType",
                    ),
                )
            )

        hitbox_cache: dict[str, tuple[tuple[CollisionBox, ...], AssetProvenance]] = {}
        variants: list[DoorGeometryVariant] = []
        for state_name, hitbox_type in state_hitboxes:
            if hitbox_type not in hitbox_cache:
                hitbox_cache[hitbox_type] = self.load_hitbox_type(
                    hitbox_type,
                    json_byte_capacity,
                )
            boxes, _provenance = hitbox_cache[hitbox_type]
            variants.append(
                DoorGeometryVariant(
                    state=state_name,
                    hitbox_type=hitbox_type,
                    boxes=boxes,
                )
            )

        interactions = _object(
            block.get("Interactions"),
            "door BlockType.Interactions",
        )
        return DoorGeometryAsset(
            asset_id=Path(path).stem,
            interaction_id=_nonempty_string(
                interactions.get("Use"),
                "door use interaction",
            ),
            opacity=_nonempty_string(
                block.get("Opacity"),
                "door opacity",
            ),
            scale=_finite_number(data.get("Scale", 1.0), "door scale"),
            variants=tuple(variants),
            item_provenance=item_provenance,
            hitbox_provenance=tuple(value[1] for value in hitbox_cache.values()),
        )

    def load_hitbox_type(
        self,
        hitbox_type: str,
        json_byte_capacity: int = DEFAULT_JSON_BYTE_CAPACITY,
    ) -> tuple[tuple[CollisionBox, ...], AssetProvenance]:
        """Load one uniquely named non-built-in hitbox asset."""

        hitbox_type = _asset_id(hitbox_type, "hitbox type")
        suffix = f"/{hitbox_type}.json"
        matches = self.entry_paths(prefix=HITBOX_PREFIX, suffix=suffix)
        if len(matches) != 1:
            raise HytaleAssetError(
                f"hitbox {hitbox_type!r} resolved to {len(matches)} entries"
            )
        data, provenance = self._read_json(matches[0], json_byte_capacity)
        boxes: list[CollisionBox] = []
        for index, value in enumerate(_array(data.get("Boxes"), "hitbox Boxes")):
            box = _object(value, f"hitbox box {index}")
            minimum = _vector3(box.get("Min"), f"hitbox box {index}.Min")
            maximum = _vector3(box.get("Max"), f"hitbox box {index}.Max")
            if any(left >= right for left, right in zip(minimum, maximum, strict=True)):
                raise HytaleAssetError(f"hitbox box {index} is inverted")
            boxes.append(CollisionBox(minimum=minimum, maximum=maximum))
        if not boxes:
            raise HytaleAssetError(f"hitbox {hitbox_type!r} contains no boxes")
        return tuple(boxes), provenance

    def _read_json(
        self,
        entry_path: str,
        byte_capacity: int,
    ) -> tuple[dict[str, Any], AssetProvenance]:
        path = _entry_path(entry_path)
        capacity = _positive_int(byte_capacity, "JSON byte capacity")
        if path in self._duplicates:
            raise HytaleAssetError(f"asset archive contains duplicate entry {path!r}")
        try:
            info = self._entries[path]
        except KeyError as error:
            raise HytaleAssetError(f"asset entry not found: {path}") from error
        if info.file_size > capacity:
            raise HytaleAssetError(
                f"asset entry has {info.file_size} bytes, capacity is {capacity}"
            )
        content = self._archive.read(info)
        if len(content) != info.file_size:
            raise HytaleAssetError("asset entry size changed while reading")
        try:
            decoded = json.loads(content.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HytaleAssetError(
                f"asset entry is not valid UTF-8 JSON: {path}"
            ) from error
        data = _object(decoded, f"asset entry {path}")
        stat = self.path.stat()
        return data, AssetProvenance(
            archive_name=self.path.name,
            archive_bytes=stat.st_size,
            archive_mtime_ns=stat.st_mtime_ns,
            entry_path=path,
            entry_bytes=info.file_size,
            entry_crc32=info.CRC,
            content_sha256=hashlib.sha256(content).hexdigest(),
        )


def _entry_prefix(value: str) -> str:
    if value == "":
        return ""
    return f"{_entry_path(value.rstrip('/'))}/"


def _required_asset_path(value: str, *, prefix: str, suffix: str) -> str:
    result = _entry_path(value)
    if not result.startswith(prefix) or not result.endswith(suffix):
        raise HytaleAssetError(f"asset entry must match {prefix!r}...{suffix!r}")
    return result


def _string_array(value: Any, label: str) -> tuple[str, ...]:
    return tuple(
        _nonempty_string(item, f"{label}[{index}]")
        for index, item in enumerate(_array(value, label))
    )


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, label)


def _int32(value: Any, label: str) -> int:
    result = _integer(value, label)
    if not -(1 << 31) <= result < (1 << 31):
        raise HytaleAssetError(f"{label} exceeds int32")
    return result


def _positive_int(value: Any, label: str) -> int:
    result = _integer(value, label)
    if result <= 0:
        raise HytaleAssetError(f"{label} must be positive")
    return result


def _nonnegative_number(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if result < 0.0:
        raise HytaleAssetError(f"{label} must be non-negative")
    return result


