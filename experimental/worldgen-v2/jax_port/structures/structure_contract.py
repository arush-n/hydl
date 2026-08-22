"""Shared authored-marker contract for WorldGen V2 structure identity."""

from __future__ import annotations

import hashlib
import json


STRUCTURE_MARKER_COMPONENT_FILTER = (
    "uuid_transform_worldgen_id_from_prefab_instance_spawn_marker_v1"
)
STRUCTURE_MARKER_IDENTITY = "SpawnMarkerEntity.getSpawnMarkerId"
STRUCTURE_MARKER_COMPONENTS = (
    "UUIDComponent",
    "TransformComponent",
    "WorldGenId",
    "FromPrefabInstance",
    STRUCTURE_MARKER_IDENTITY,
)


def structure_marker_contract() -> dict[str, object]:
    return {
        "schema": "hytalerl_worldgen_v2_structure_marker_contract_v1",
        "components": list(STRUCTURE_MARKER_COMPONENTS),
        "component_filter": STRUCTURE_MARKER_COMPONENT_FILTER,
        "marker_identity": STRUCTURE_MARKER_IDENTITY,
        "bounds": "registry_local_half_open_rotated_about_prefab_anchor",
        "identity": "marker_asset_plus_native_prefab_instance",
        "authoring_protocol": (
            "one_allowlisted_manual_spawn_marker_entity_per_prefab_instance"
        ),
        "block_only_prefabs": "unobservable_without_authored_marker",
    }


STRUCTURE_MARKER_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        structure_marker_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()
