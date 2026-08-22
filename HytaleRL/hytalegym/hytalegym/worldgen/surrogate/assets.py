"""Read-only Hytale asset ingestion for non-authoritative surrogate worlds."""

from __future__ import annotations

import os
from pathlib import Path
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
from ._archive import (  # noqa: F401  (re-exported: callers unchanged)
    ASSIGNMENT_PREFIX,
    AssetProvenance,
    AssignmentPrefabReference,
    BIOME_PREFIX,
    BLOCK_SET_PREFIX,
    BiomeGeneratorAsset,
    BiomePropAssignment,
    BiomeRange,
    BlockSetAsset,
    CollisionBox,
    DEFAULT_GENERATOR_NODE_CAPACITY,
    DEFAULT_JSON_BYTE_CAPACITY,
    DEFAULT_PREFAB_BLOCK_CAPACITY,
    DEFAULT_PREFAB_ENTITY_CAPACITY,
    DEFAULT_PREFAB_REFERENCE_CAPACITY,
    DEFAULT_SPAWN_MARKER_CONFIGURATION_CAPACITY,
    DoorGeometryAsset,
    DoorGeometryVariant,
    GeneratorAssignmentIndex,
    HITBOX_PREFIX,
    HytaleAssetArchive,
    ITEM_DROP_LIST_PREFIX,
    ITEM_PREFIX,
    NPC_ROLE_PREFIX,
    PREFAB_PREFIX,
    PrefabBlock,
    PrefabTemplate,
    ResolvedBlockSet,
    SPAWN_MARKER_PREFIX,
    SpawnMarkerAsset,
    UNARMED_GATHERING_PREFIX,
    WORLD_PREFAB_PREFIX,
    WORLD_STRUCTURE_PREFIX,
    WorldStructureProfile,
    _BIOME_FIELDS,
    _BLOCK_SET_FIELDS,
    _BLOCK_SET_RUNTIME_SELECTOR_FIELDS,
    _PREFAB_BLOCK_FIELDS,
    _PREFAB_TOP_LEVEL_FIELDS,
    _SPAWN_MARKER_FIELDS,
    _WORLD_STRUCTURE_FIELDS,
    _entry_prefix,
    _filler_value,
    _int32,
    _json_pointer_token,
    _nonnegative_number,
    _optional_bool,
    _optional_string,
    _positive_int,
    _required_asset_path,
    _string_array,
    _support_value,
)

# BuilderRole.readConfig() supplies this value when OpaqueBlockSet is omitted.
DEFAULT_NPC_OPAQUE_BLOCK_SET_ID = "Opaque"
HYTALE_ASSETS_ZIP_ENV = "HYTALE_ASSETS_ZIP"

def default_hytale_assets_path() -> Path:
    """Return the explicit or standard installed ``Assets.zip`` path.

    ``HYTALE_ASSETS_ZIP`` is the cross-platform override used by the ruleset
    loaders and WSL/CUDA gates.  ``APPDATA`` remains the Windows convenience
    fallback; neither path is assumed to exist here.
    """

    explicit = os.environ.get(HYTALE_ASSETS_ZIP_ENV)
    if explicit:
        return Path(explicit)
    app_data = os.environ.get("APPDATA")
    if not app_data:
        raise FileNotFoundError(
            f"{HYTALE_ASSETS_ZIP_ENV} and APPDATA are unavailable; "
            "supply Assets.zip explicitly"
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


def decode_prefab_filler(value: int | None) -> tuple[int, int, int]:
    """Decode the local API's signed five-bit X, Y, Z owner offset."""

    packed = _filler_value(value, "prefab filler")
    if packed is None:
        return 0, 0, 0
    return (
        _unpack_signed_five(packed, 0),
        _unpack_signed_five(packed, 10),
        _unpack_signed_five(packed, 5),
    )


def _unpack_signed_five(value: int, shift: int) -> int:
    result = (value >> shift) & 31
    return result | -32 if result & 16 else result
