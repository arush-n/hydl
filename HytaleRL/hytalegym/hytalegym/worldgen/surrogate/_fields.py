"""Leaf helpers extracted verbatim from _archive.py."""

from dataclasses import dataclass
from typing import Any
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


_PREFAB_TOP_LEVEL_FIELDS = {
    "version",
    "blockIdVersion",
    "anchorX",
    "anchorY",
    "anchorZ",
    "blocks",
    "entities",
}


_PREFAB_BLOCK_FIELDS = {
    "x",
    "y",
    "z",
    "name",
    "support",
    "rotation",
    "filler",
    "components",
}


_WORLD_STRUCTURE_FIELDS = {
    "Type",
    "Biomes",
    "DefaultBiome",
    "DefaultTransitionDistance",
    "MaxBiomeEdgeDistance",
    "Density",
    "SpawnPositions",
    "Framework",
    "Tags",
}


_BIOME_FIELDS = {
    "$NodeId",
    "Name",
    "Terrain",
    "MaterialProvider",
    "Props",
    "EnvironmentProvider",
    "TintProvider",
    "$NodeEditorMetadata",
}


_BLOCK_SET_FIELDS = {
    "Parent",
    "IncludeAll",
    "IncludeBlockTypes",
    "ExcludeBlockTypes",
    "IncludeBlockGroups",
    "ExcludeBlockGroups",
    "IncludeHitboxTypes",
    "ExcludeHitboxTypes",
    "IncludeCategories",
    "ExcludeCategories",
}


_BLOCK_SET_RUNTIME_SELECTOR_FIELDS = (
    "IncludeBlockGroups",
    "ExcludeBlockGroups",
    "IncludeHitboxTypes",
    "ExcludeHitboxTypes",
    "IncludeCategories",
    "ExcludeCategories",
)


_SPAWN_MARKER_FIELDS = {
    "Model",
    "NPCs",
    "ExclusionRadius",
    "MaxDropHeight",
    "RealtimeRespawn",
    "ManualTrigger",
    "DeactivationDistance",
    "DeactivationTime",
}


@dataclass(frozen=True)
class ResolvedBlockSet:
    """Block-name-only expansion over an explicit candidate universe."""

    asset_id: str
    block_type_ids: tuple[str, ...]
    inheritance_chain: tuple[str, ...]
    provenance: tuple[AssetProvenance, ...]


@dataclass(frozen=True)
class AssignmentPrefabReference:
    path: str
    weight: float
    load_entities: bool
    json_pointer: str


@dataclass(frozen=True)
class GeneratorAssignmentIndex:
    root_type: str
    export_name: str | None
    prefab_references: tuple[AssignmentPrefabReference, ...]
    provenance: AssetProvenance


@dataclass(frozen=True)
class CollisionBox:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]


@dataclass(frozen=True)
class DoorGeometryVariant:
    state: str
    hitbox_type: str
    boxes: tuple[CollisionBox, ...]


@dataclass(frozen=True)
class DoorGeometryAsset:
    asset_id: str
    interaction_id: str
    opacity: str
    scale: float
    variants: tuple[DoorGeometryVariant, ...]
    item_provenance: AssetProvenance
    hitbox_provenance: tuple[AssetProvenance, ...]


def _optional_bool(value: Any, label: str, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise HytaleAssetError(f"{label} must be boolean")
    return value
