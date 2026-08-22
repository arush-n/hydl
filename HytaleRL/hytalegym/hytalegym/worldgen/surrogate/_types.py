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


@dataclass(frozen=True)
class AssetProvenance:
    archive_name: str
    archive_bytes: int
    archive_mtime_ns: int
    entry_path: str
    entry_bytes: int
    entry_crc32: int
    content_sha256: str


@dataclass(frozen=True)
class BlockSetAsset:
    """Authored block-set selectors; runtime-only selectors remain explicit."""

    asset_id: str
    parent: str | None
    include_all: bool
    include_block_types: tuple[str, ...]
    exclude_block_types: tuple[str, ...]
    runtime_selector_fields: tuple[str, ...]
    unsupported_fields: tuple[str, ...]
    provenance: AssetProvenance


@dataclass(frozen=True)
class SpawnMarkerAsset:
    """Validated native marker metadata without simulating its lifecycle."""

    asset_id: str
    model_id: str | None
    configurations: tuple[SpawnMarkerConfiguration, ...]
    exclusion_radius: float
    maximum_drop_height: float
    realtime_respawn: bool
    manual_trigger: bool
    deactivation_distance: float
    deactivation_seconds: float
    provenance: AssetProvenance


@dataclass(frozen=True)
class PrefabBlock:
    position: tuple[int, int, int]
    asset_id: str
    rotation: int | None
    filler: int | None
    component_types: tuple[str, ...]
    support: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "filler",
            _filler_value(self.filler, "prefab block filler"),
        )
        object.__setattr__(
            self,
            "support",
            _support_value(self.support, "prefab block support"),
        )

    @property
    def is_state_variant(self) -> bool:
        return self.asset_id.startswith("*")


@dataclass(frozen=True)
class PrefabTemplate:
    version: int
    block_id_version: int
    anchor: tuple[int, int, int]
    blocks: tuple[PrefabBlock, ...]
    bounds_min: tuple[int, int, int]
    bounds_max: tuple[int, int, int]
    component_types: tuple[str, ...]
    unsupported_top_level_fields: tuple[str, ...]
    unsupported_block_fields: tuple[str, ...]
    provenance: AssetProvenance
    entities: tuple[PrefabEntity, ...] = ()

    @property
    def static_placement_supported(self) -> bool:
        return not (self.unsupported_top_level_fields or self.unsupported_block_fields)


@dataclass(frozen=True)
class BiomeRange:
    biome: str
    minimum: float
    maximum: float


@dataclass(frozen=True)
class WorldStructureProfile:
    profile_type: str
    biomes: tuple[BiomeRange, ...]
    default_biome: str
    default_transition_distance: float
    maximum_biome_edge_distance: float
    density_source: str | None
    decimal_constants: tuple[tuple[str, float], ...]
    unsupported_fields: tuple[str, ...]
    provenance: AssetProvenance


@dataclass(frozen=True)
class BiomePropAssignment:
    """One authored prop graph; execution is deliberately not implied."""

    assignment: str | None
    skipped: bool
    runtime: int | None
    source_field: str
    root_type: str
    imported_assignments: tuple[str, ...]
    graph_sha256: str


@dataclass(frozen=True)
class BiomeGeneratorAsset:
    biome_id: str
    display_name: str | None
    terrain_type: str
    terrain_node_types: tuple[str, ...]
    terrain_imports: tuple[str, ...]
    terrain_noise_scales: tuple[float, ...]
    solid_materials: tuple[str, ...]
    fluid_materials: tuple[str, ...]
    prop_assignments: tuple[BiomePropAssignment, ...]
    unsupported_top_level_fields: tuple[str, ...]
    provenance: AssetProvenance


def _filler_value(value: Any, label: str) -> int | None:
    if value is None:
        return None
    result = _integer(value, label)
    if not 0 <= result < (1 << 15):
        raise HytaleAssetError(f"{label} must fit the local 15-bit filler encoding")
    return None if result == 0 else result


def _support_value(value: Any, label: str) -> int:
    if value is None:
        return 0
    result = _integer(value, label)
    if not 0 <= result <= 15:
        raise HytaleAssetError(f"{label} must be in [0, 15]")
    return result
