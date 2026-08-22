"""Lossless host adapter for the native NPC role snapshot endpoint."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import uuid
from typing import Any

import numpy.typing as npt
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.entities.privileged import (
    PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_NATIVE,
    PrivilegedEntitySnapshot,
)
from hytalegym.worldgen.native_privileged_entities import (
    NATIVE_PRIVILEGED_ENTITY_MAX_CAPACITY,
    NativePrivilegedEntityCapture,
    native_privileged_entity_adapter_contract_sha256,
    native_privileged_entity_request,
    native_privileged_entity_snapshot,
)
from hytalegym.worldgen.surrogate.entities import EntityGeometryProfile


_LEGACY_NATIVE_PRIVILEGED_NPC_SCHEMA = "hytalerl_privileged_npc_snapshot_v1"
_LEGACY_NATIVE_PRIVILEGED_NPC_COMPONENT_FILTER = (
    "npc_uuid_transform_velocity_role_v1"
)
NATIVE_PRIVILEGED_NPC_SCHEMA = "hytalerl_privileged_npc_snapshot_v2"
NATIVE_PRIVILEGED_NPC_VERSION = 2
NATIVE_PRIVILEGED_NPC_COMPONENT_FILTER = (
    "npc_uuid_transform_velocity_role_geometry_v2"
)
_GEOMETRY_SEMANTICS = (
    "runtime_rotation_adjusted_local_aabb_position_cache_los_v1"
)


@dataclass(frozen=True, slots=True)
class NativePrivilegedNpcCapture:
    """Complete native NPC pose rows plus palette-independent role IDs."""

    spatial: NativePrivilegedEntityCapture
    type_asset_ids: tuple[str, ...]
    runtime_geometry_transport: bool
    geometry_supported: npt.NDArray[np.bool_]
    local_bounds: npt.NDArray[np.float64]
    collidable: npt.NDArray[np.bool_]
    blocks_los: npt.NDArray[np.bool_]
    los_offset_supported: npt.NDArray[np.bool_]
    los_offset: npt.NDArray[np.float64]
    model_present: npt.NDArray[np.bool_]
    model_asset_ids: tuple[str, ...]
    model_scale: npt.NDArray[np.float64]
    model_eye_height: npt.NDArray[np.float64]
    entity_scale_present: npt.NDArray[np.bool_]
    entity_scale: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if not isinstance(self.spatial, NativePrivilegedEntityCapture):
            raise TypeError("spatial must be a NativePrivilegedEntityCapture")
        values = tuple(self.type_asset_ids)
        if len(values) != self.spatial.emitted_count:
            raise ValueError("type_asset_ids must match emitted NPC rows")
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError("NPC type asset IDs must be non-empty strings")
        if not isinstance(self.runtime_geometry_transport, bool):
            raise TypeError("runtime_geometry_transport must be bool")
        count = self.spatial.emitted_count
        arrays = {
            "geometry_supported": _array(
                self.geometry_supported, np.bool_, (count,)
            ),
            "local_bounds": _array(self.local_bounds, np.float64, (count, 6)),
            "collidable": _array(self.collidable, np.bool_, (count,)),
            "blocks_los": _array(self.blocks_los, np.bool_, (count,)),
            "los_offset_supported": _array(
                self.los_offset_supported, np.bool_, (count,)
            ),
            "los_offset": _array(self.los_offset, np.float64, (count, 3)),
            "model_present": _array(self.model_present, np.bool_, (count,)),
            "model_scale": _array(self.model_scale, np.float64, (count,)),
            "model_eye_height": _array(
                self.model_eye_height, np.float64, (count,)
            ),
            "entity_scale_present": _array(
                self.entity_scale_present, np.bool_, (count,)
            ),
            "entity_scale": _array(self.entity_scale, np.float64, (count,)),
        }
        model_ids = tuple(self.model_asset_ids)
        if len(model_ids) != count or any(
            not isinstance(value, str) for value in model_ids
        ):
            raise ValueError("model_asset_ids must match emitted NPC rows")
        if not self.runtime_geometry_transport and (
            any(np.any(value) for value in arrays.values())
            or any(model_ids)
        ):
            raise ValueError("legacy NPC capture cannot carry runtime geometry")
        if not all(np.all(np.isfinite(value)) for name, value in arrays.items()
                   if value.dtype != np.bool_):
            raise ValueError("NPC runtime geometry must be finite")
        supported = arrays["geometry_supported"]
        bounds = arrays["local_bounds"]
        bounds_valid = np.all(bounds[:, :3] < bounds[:, 3:], axis=1)
        if np.any(supported & ~bounds_valid):
            raise ValueError("supported NPC geometry must have positive extent")
        if np.any(
            ~supported
            & (arrays["collidable"] | arrays["blocks_los"])
        ):
            raise ValueError("unsupported NPC geometry has physical flags")
        offset_supported = arrays["los_offset_supported"]
        if np.any(~offset_supported[:, None] & (arrays["los_offset"] != 0.0)):
            raise ValueError("unsupported NPC LOS offsets must be zero")
        present = arrays["model_present"]
        if any(present[index] and not model_ids[index] for index in range(count)):
            raise ValueError("present NPC models require an asset ID")
        if np.any(present & (arrays["model_scale"] <= 0.0)) or np.any(
            present & (arrays["model_eye_height"] < 0.0)
        ):
            raise ValueError("present NPC model evidence is invalid")
        if np.any(
            ~present
            & (
                (arrays["model_scale"] != 0.0)
                | (arrays["model_eye_height"] != 0.0)
            )
        ) or any(not present[index] and model_ids[index] for index in range(count)):
            raise ValueError("absent NPC model has contradictory evidence")
        expected_offset = np.where(
            present[:, None],
            np.stack(
                (
                    np.zeros(count),
                    arrays["model_eye_height"],
                    np.zeros(count),
                ),
                axis=1,
            ),
            (bounds[:, :3] + bounds[:, 3:]) * 0.5,
        )
        if np.any(offset_supported) and not np.array_equal(
            arrays["los_offset"][offset_supported],
            expected_offset[offset_supported],
        ):
            raise ValueError("NPC LOS offset disagrees with native endpoint rules")
        scale_present = arrays["entity_scale_present"]
        if np.any(scale_present & (arrays["entity_scale"] <= 0.0)) or np.any(
            ~scale_present & (arrays["entity_scale"] != 0.0)
        ):
            raise ValueError("NPC display-scale evidence is contradictory")
        object.__setattr__(self, "type_asset_ids", values)
        object.__setattr__(self, "model_asset_ids", model_ids)
        for name, value in arrays.items():
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    @classmethod
    def from_response(
        cls,
        response: Mapping[str, Any],
    ) -> NativePrivilegedNpcCapture:
        """Decode the additive v1 NPC endpoint without duplicating wire math."""

        source = dict(response)
        if source.get("type") != "privileged_npc_snapshot":
            raise ValueError("native response is not an NPC snapshot")
        schema_version = (source.get("schema"), source.get("version"))
        legacy = schema_version == (_LEGACY_NATIVE_PRIVILEGED_NPC_SCHEMA, 1)
        current = schema_version == (
            NATIVE_PRIVILEGED_NPC_SCHEMA,
            NATIVE_PRIVILEGED_NPC_VERSION,
        )
        if not (legacy or current):
            raise ValueError("unsupported privileged NPC schema")
        expected_filter = (
            _LEGACY_NATIVE_PRIVILEGED_NPC_COMPONENT_FILTER
            if legacy
            else NATIVE_PRIVILEGED_NPC_COMPONENT_FILTER
        )
        if source.get("component_filter") != expected_filter:
            raise ValueError("unsupported privileged NPC component filter")
        raw_type_ids = source.get("type_asset_ids")
        if not isinstance(raw_type_ids, list):
            raise TypeError("type_asset_ids must be an array")

        # Both endpoints deliberately share all spatial encoding fields. Use
        # the already-audited spatial decoder after replacing only the three
        # discriminators; do not maintain a second binary decoder.
        spatial_source = dict(source)
        spatial_source.update(
            {
                "type": "privileged_entity_snapshot",
                "schema": "hytalerl_privileged_entity_snapshot_v1",
                "version": 1,
                "component_filter": "uuid_transform_velocity_v1",
            }
        )
        spatial_source.pop("type_asset_ids", None)
        for name in (
            "geometry_semantics",
            "geometry_supported_u8",
            "local_bounds_f64_le_min_max_xyz",
            "collidable_u8",
            "blocks_los_u8",
            "los_offset_supported_u8",
            "los_offsets_f64_le_xyz",
            "model_present_u8",
            "model_asset_ids",
            "model_scales_f64_le",
            "model_eye_heights_f64_le",
            "entity_scale_present_u8",
            "entity_scales_f64_le",
        ):
            spatial_source.pop(name, None)
        spatial = NativePrivilegedEntityCapture.from_response(spatial_source)
        count = spatial.emitted_count
        if legacy:
            return cls(
                spatial=spatial,
                type_asset_ids=tuple(raw_type_ids),
                runtime_geometry_transport=False,
                geometry_supported=np.zeros(count, dtype=np.bool_),
                local_bounds=np.zeros((count, 6), dtype=np.float64),
                collidable=np.zeros(count, dtype=np.bool_),
                blocks_los=np.zeros(count, dtype=np.bool_),
                los_offset_supported=np.zeros(count, dtype=np.bool_),
                los_offset=np.zeros((count, 3), dtype=np.float64),
                model_present=np.zeros(count, dtype=np.bool_),
                model_asset_ids=("",) * count,
                model_scale=np.zeros(count, dtype=np.float64),
                model_eye_height=np.zeros(count, dtype=np.float64),
                entity_scale_present=np.zeros(count, dtype=np.bool_),
                entity_scale=np.zeros(count, dtype=np.float64),
            )
        if source.get("geometry_semantics") != _GEOMETRY_SEMANTICS:
            raise ValueError("unsupported privileged NPC geometry semantics")
        raw_model_ids = source.get("model_asset_ids")
        if not isinstance(raw_model_ids, list):
            raise TypeError("model_asset_ids must be an array")
        return cls(
            spatial=spatial,
            type_asset_ids=tuple(raw_type_ids),
            runtime_geometry_transport=True,
            geometry_supported=_decode_bool(source, "geometry_supported_u8", count),
            local_bounds=_decode(source, "local_bounds_f64_le_min_max_xyz", "<f8", count * 6).reshape(count, 6),
            collidable=_decode_bool(source, "collidable_u8", count),
            blocks_los=_decode_bool(source, "blocks_los_u8", count),
            los_offset_supported=_decode_bool(source, "los_offset_supported_u8", count),
            los_offset=_decode(source, "los_offsets_f64_le_xyz", "<f8", count * 3).reshape(count, 3),
            model_present=_decode_bool(source, "model_present_u8", count),
            model_asset_ids=tuple(raw_model_ids),
            model_scale=_decode(source, "model_scales_f64_le", "<f8", count),
            model_eye_height=_decode(source, "model_eye_heights_f64_le", "<f8", count),
            entity_scale_present=_decode_bool(source, "entity_scale_present_u8", count),
            entity_scale=_decode(source, "entity_scales_f64_le", "<f8", count),
        )

    @classmethod
    def from_uuid_response(
        cls,
        response: Mapping[str, Any],
        npc_uuid: uuid.UUID | str | bytes,
        bounds: npt.ArrayLike,
    ) -> NativePrivilegedNpcCapture:
        """Decode one exact-UUID response and reject selector ambiguity."""

        expected = _as_uuid(npc_uuid)
        capture = cls.from_response(response)
        if capture.spatial.capacity != 1 or capture.spatial.overflow:
            raise ValueError("exact NPC response must be complete at capacity one")
        if capture.emitted_count > 1:
            raise ValueError("exact NPC response emitted more than one row")
        if capture.uuids and capture.uuids != (expected,):
            raise ValueError("exact NPC response does not match requested UUID")
        request = native_privileged_npc_request(bounds, capacity=1)
        expected_bounds = np.frombuffer(
            request["bounds_f64_le_min_max_xyz"], dtype="<f8"
        )
        if not np.array_equal(capture.spatial.bounds, expected_bounds):
            raise ValueError("exact NPC response does not match requested bounds")
        return capture

    @property
    def bridge_sha256(self) -> str:
        return self.spatial.bridge_sha256

    @property
    def emitted_count(self) -> int:
        return self.spatial.emitted_count

    @property
    def uuids(self) -> tuple[uuid.UUID, ...]:
        """Return exact RFC-4122 identities accepted by NPC tracing."""

        return tuple(uuid.UUID(bytes=bytes(row)) for row in self.spatial.uuid_bytes)

    def instance_identity_words(self) -> npt.NDArray:
        return self.spatial.instance_identity_words()


def native_privileged_npc_request(
    bounds: npt.ArrayLike,
    *,
    capacity: int = NATIVE_PRIVILEGED_ENTITY_MAX_CAPACITY,
) -> dict[str, object]:
    """Build one bounded native NPC role/pose request."""

    request = native_privileged_entity_request(bounds, capacity=capacity)
    request["type"] = "privileged_npc_snapshot"
    request["component_filter"] = NATIVE_PRIVILEGED_NPC_COMPONENT_FILTER
    return request


def native_privileged_npc_by_uuid_request(
    npc_uuid: uuid.UUID | str | bytes,
    bounds: npt.ArrayLike,
) -> dict[str, object]:
    """Build one capacity-one native NPC lookup by exact RFC-4122 UUID."""

    request = native_privileged_npc_request(bounds, capacity=1)
    request.pop("capacity")
    request["type"] = "privileged_npc_snapshot_by_uuid"
    request["npc_uuid_bytes"] = _as_uuid(npc_uuid).bytes
    return request


def native_privileged_npc_snapshot(
    capture: NativePrivilegedNpcCapture,
    *,
    entity_capacity: int = NATIVE_PRIVILEGED_ENTITY_MAX_CAPACITY,
    geometry_profiles: Mapping[str, EntityGeometryProfile] | None = None,
) -> PrivilegedEntitySnapshot:
    """Adapt role rows and optionally bind exact native-certified profiles."""

    if not isinstance(capture, NativePrivilegedNpcCapture):
        raise TypeError("capture must be a NativePrivilegedNpcCapture")
    snapshot = native_privileged_entity_snapshot(
        capture.spatial,
        type_asset_ids=capture.type_asset_ids,
        entity_capacity=entity_capacity,
    )
    if geometry_profiles is None:
        if not capture.runtime_geometry_transport:
            return snapshot
        return _bind_runtime_geometry(snapshot, capture)
    profiles = dict(geometry_profiles)
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, EntityGeometryProfile)
        or key != value.type_id
        for key, value in profiles.items()
    ):
        raise TypeError(
            "geometry_profiles must map exact role IDs to EntityGeometryProfile"
        )
    if any(not value.native_certified for value in profiles.values()):
        raise ValueError("native NPC profiles must be native-certified")

    count = capture.emitted_count
    shape = tuple(int(value) for value in snapshot.initialized.shape)
    supported = np.zeros(shape, dtype=np.bool_)
    collidable = np.zeros(shape, dtype=np.bool_)
    blocks_los = np.zeros(shape, dtype=np.bool_)
    bounds = np.zeros(shape + (6,), dtype=np.float32)
    offset_supported = np.zeros(shape, dtype=np.bool_)
    offset = np.zeros(shape + (3,), dtype=np.float32)
    provenance = np.zeros(shape, dtype=np.uint8)
    for slot, role_id in enumerate(capture.type_asset_ids):
        profile = profiles.get(role_id)
        if profile is None:
            continue
        supported[0, slot] = True
        collidable[0, slot] = profile.collidable
        blocks_los[0, slot] = profile.blocks_los
        bounds[0, slot] = profile.local_bounds
        if profile.line_of_sight_offset is not None:
            offset_supported[0, slot] = True
            offset[0, slot] = profile.line_of_sight_offset
        provenance[0, slot] = PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_NATIVE
    if count < shape[1]:
        supported[0, count:] = False
    return snapshot._replace(
        geometry_supported=jnp.asarray(supported),
        collidable=jnp.asarray(collidable),
        blocks_los=jnp.asarray(blocks_los),
        local_bounds=jnp.asarray(bounds),
        los_offset_supported=jnp.asarray(offset_supported),
        los_offset=jnp.asarray(offset),
        geometry_provenance=jnp.asarray(provenance),
    )


def native_privileged_npc_adapter_contract() -> dict[str, object]:
    return {
        "schema": "hytalerl_native_privileged_npc_adapter_v3",
        "version": 3,
        "source": NATIVE_PRIVILEGED_NPC_SCHEMA,
        "component_filter": NATIVE_PRIVILEGED_NPC_COMPONENT_FILTER,
        "spatial_adapter_sha256": (native_privileged_entity_adapter_contract_sha256()),
        "selectors": {
            "bounded_enumeration": "complete_or_empty_capacity_at_most_256",
            "exact_uuid": (
                "server_UUID_index_capacity_one_requested_AABB_bound_"
                "zero_or_requested_UUID_only"
            ),
        },
        "type_identity": {
            "wire": "exact_NPCEntity_role_name_String",
            "palette_independent": True,
            "missing_blank_or_wrong_count": "reject_complete_capture",
        },
        "runtime_geometry": {
            "bounds": "exact_current_rotation_adjusted_BoundingBox_component",
            "collidable": "bounds_and_not_Intangible_and_not_DeathComponent",
            "blocks_los": "false_PositionCache_entity_LOS_tests_blocks_only",
            "los_offset": "model_eye_height_else_bounding_box_center",
            "entity_scale": "diagnostic_only_never_double_applied_to_bounds",
            "missing_or_invalid": "row_geometry_unsupported_environment_physical_fail_closed",
        },
        "physical_profile_override": (
            "optional_explicit_role_to_native_certified_profile_map_"
            "replaces_runtime_geometry_and_missing_role_remains_fail_closed"
        ),
        "behavior": "not_inferred_from_role_name",
        "overflow": "no_partial_rows",
    }


def native_privileged_npc_adapter_contract_sha256() -> str:
    payload = json.dumps(
        native_privileged_npc_adapter_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bind_runtime_geometry(
    snapshot: PrivilegedEntitySnapshot,
    capture: NativePrivilegedNpcCapture,
) -> PrivilegedEntitySnapshot:
    shape = tuple(int(value) for value in snapshot.initialized.shape)
    count = capture.emitted_count
    supported = np.zeros(shape, dtype=np.bool_)
    collidable = np.zeros(shape, dtype=np.bool_)
    blocks_los = np.zeros(shape, dtype=np.bool_)
    bounds = np.zeros(shape + (6,), dtype=np.float32)
    offset_supported = np.zeros(shape, dtype=np.bool_)
    offset = np.zeros(shape + (3,), dtype=np.float32)
    provenance = np.zeros(shape, dtype=np.uint8)
    supported[0, :count] = capture.geometry_supported
    collidable[0, :count] = capture.collidable
    blocks_los[0, :count] = capture.blocks_los
    bounds[0, :count] = capture.local_bounds
    offset_supported[0, :count] = capture.los_offset_supported
    offset[0, :count] = capture.los_offset
    provenance[0, :count] = np.where(
        capture.geometry_supported,
        PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_NATIVE,
        0,
    )
    return snapshot._replace(
        geometry_supported=jnp.asarray(supported),
        collidable=jnp.asarray(collidable),
        blocks_los=jnp.asarray(blocks_los),
        local_bounds=jnp.asarray(bounds),
        los_offset_supported=jnp.asarray(offset_supported),
        los_offset=jnp.asarray(offset),
        geometry_provenance=jnp.asarray(provenance),
    )


def _array(
    value: npt.ArrayLike,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> npt.NDArray:
    result = np.asarray(value)
    if result.dtype != np.dtype(dtype) or result.shape != shape:
        raise ValueError(f"array must have dtype {np.dtype(dtype)} and shape {shape}")
    return np.ascontiguousarray(result)


def _as_uuid(value: uuid.UUID | str | bytes) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    if isinstance(value, str):
        return uuid.UUID(value)
    if isinstance(value, bytes):
        return uuid.UUID(bytes=value)
    raise TypeError("UUID must be uuid.UUID, canonical string, or 16 bytes")


def _decode(
    source: Mapping[str, Any],
    name: str,
    dtype: str,
    count: int,
) -> npt.NDArray:
    payload = source.get(name)
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be binary")
    result = np.frombuffer(payload, dtype=dtype)
    if result.size != count:
        raise ValueError(f"{name} has the wrong element count")
    return np.ascontiguousarray(result)


def _decode_bool(
    source: Mapping[str, Any],
    name: str,
    count: int,
) -> npt.NDArray[np.bool_]:
    values = _decode(source, name, "u1", count)
    if np.any(values > 1):
        raise ValueError(f"{name} must contain only 0 or 1")
    return values.astype(np.bool_)


__all__ = [
    "NATIVE_PRIVILEGED_NPC_COMPONENT_FILTER",
    "NATIVE_PRIVILEGED_NPC_SCHEMA",
    "NATIVE_PRIVILEGED_NPC_VERSION",
    "NativePrivilegedNpcCapture",
    "native_privileged_npc_adapter_contract",
    "native_privileged_npc_adapter_contract_sha256",
    "native_privileged_npc_by_uuid_request",
    "native_privileged_npc_request",
    "native_privileged_npc_snapshot",
]
