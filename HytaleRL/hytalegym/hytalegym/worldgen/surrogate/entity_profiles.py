"""Provenance-backed promotion of native role captures into entity profiles."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from hytalegym.worldgen.snapshot import NativeWorldgenSnapshot
from hytalegym.worldgen.surrogate.entities import EntityGeometryProfile


_LEGACY_ENTITY_PROFILE_CAPTURE_SCHEMA = "hytalerl_native_entity_profile_capture_v1"
ENTITY_PROFILE_CAPTURE_SCHEMA = "hytalerl_native_entity_profile_capture_v2"
ENTITY_PROFILE_CAPTURE_VERSION = 2
_MODEL_EVIDENCE_VERSION = 1
_SUPPORTED_CAPTURE_SCHEMAS = {
    (_LEGACY_ENTITY_PROFILE_CAPTURE_SCHEMA, 1),
    (ENTITY_PROFILE_CAPTURE_SCHEMA, ENTITY_PROFILE_CAPTURE_VERSION),
}


@dataclass(frozen=True)
class _RuntimeModelEvidence:
    model_present: bool
    model_asset_id: str
    model_scale: float
    model_eye_height: float
    bounding_box_present: bool
    entity_scale_present: bool
    entity_scale: float

    @classmethod
    def from_snapshot(
        cls,
        snapshot: NativeWorldgenSnapshot,
        subject: str,
    ) -> "_RuntimeModelEvidence":
        metadata = snapshot.metadata
        if metadata.get("native_entity_model_evidence_version") != _MODEL_EVIDENCE_VERSION:
            raise ValueError("native entity model evidence version 1 is required")
        prefix = f"{subject}_"
        return cls(
            model_present=_required_bool(metadata, f"{prefix}model_present"),
            model_asset_id=_required_string(metadata, f"{prefix}model_asset_id"),
            model_scale=_required_float(metadata, f"{prefix}model_scale"),
            model_eye_height=_required_float(metadata, f"{prefix}model_eye_height"),
            bounding_box_present=_required_bool(
                metadata,
                f"{prefix}bounding_box_present",
            ),
            entity_scale_present=_required_bool(
                metadata,
                f"{prefix}entity_scale_present",
            ),
            entity_scale=_required_float(metadata, f"{prefix}entity_scale"),
        )

    def validate_geometry(
        self,
        bounds: np.ndarray,
        offset: np.ndarray,
        subject: str,
    ) -> None:
        if not self.bounding_box_present:
            raise ValueError("native entity bounding-box evidence is required")
        if self.model_present:
            if not self.model_asset_id:
                raise ValueError("native model asset ID is required")
            if self.model_scale <= 0.0 or self.model_eye_height < 0.0:
                raise ValueError("native model scale and eye height are invalid")
            expected_offset = np.asarray(
                [0.0, self.model_eye_height, 0.0],
                dtype=np.float32,
            )
        else:
            if (
                self.model_asset_id
                or self.model_scale != 0.0
                or self.model_eye_height != 0.0
            ):
                raise ValueError("absent native model has contradictory evidence")
            expected_offset = (
                np.zeros(3, dtype=np.float32)
                if subject == "agent"
                else np.asarray(
                    (bounds[:3] + bounds[3:]) * 0.5,
                    dtype=np.float32,
                )
            )
        if not np.array_equal(offset, expected_offset):
            raise ValueError("native LOS offset disagrees with runtime model evidence")
        if self.entity_scale_present:
            if self.entity_scale <= 0.0:
                raise ValueError("native entity display scale must be positive")
            if self.entity_scale != 1.0:
                raise ValueError(
                    "non-unit native entity display scale is unsupported"
                )
        elif self.entity_scale != 0.0:
            raise ValueError("absent native entity scale has contradictory evidence")

    def manifest(self) -> dict[str, Any]:
        return {
            "version": _MODEL_EVIDENCE_VERSION,
            "model": {
                "present": self.model_present,
                "asset_id": self.model_asset_id,
                "scale": self.model_scale,
                "eye_height": self.model_eye_height,
            },
            "bounding_box_present": self.bounding_box_present,
            "entity_scale_component": {
                "present": self.entity_scale_present,
                "scale": self.entity_scale,
            },
        }


@dataclass(frozen=True)
class NativeEntityProfileCapture:
    """Exact captured geometry plus its canonical, non-certifying manifest."""

    profile: EntityGeometryProfile
    manifest_json: str

    def __post_init__(self) -> None:
        manifest = json.loads(self.manifest_json)
        if not isinstance(manifest, dict):
            raise ValueError("entity profile capture manifest must be an object")
        schema = manifest.get("schema")
        version = manifest.get("version")
        if (
            not isinstance(schema, str)
            or not isinstance(version, int)
            or isinstance(version, bool)
        ):
            raise ValueError("unsupported entity profile capture schema/version")
        schema_version = (schema, version)
        if schema_version not in _SUPPORTED_CAPTURE_SCHEMAS:
            raise ValueError("unsupported entity profile capture schema/version")
        if _sha256(self.manifest_json.encode()) != self.profile.provenance_sha256:
            raise ValueError("entity profile capture provenance mismatch")
        if self.profile.native_certified:
            raise ValueError("capture promotion cannot certify entity mechanics")

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_json)


def capture_native_role_geometry_profile(
    snapshots: Sequence[NativeWorldgenSnapshot],
    *,
    type_id: str,
    subject: str,
    collidable: bool,
    blocks_los: bool,
    bridge_jar_sha256: str,
    server_jar_sha256: str,
    role_asset_sha256: str,
    model_asset_sha256: str | None,
) -> NativeEntityProfileCapture:
    """Promote stable native bounds/endpoints without certifying policy flags.

    Two or more captures must agree on geometry and versioned runtime-model
    evidence. ``collidable`` and ``blocks_los`` remain explicit caller policy
    because the geometry frame does not observe either mechanic.
    """

    values = tuple(snapshots)
    if len(values) < 2 or any(
        not isinstance(value, NativeWorldgenSnapshot) for value in values
    ):
        raise ValueError("at least two native worldgen snapshots are required")
    if not isinstance(type_id, str) or not type_id:
        raise TypeError("type_id must be a non-empty string")
    if subject not in {"agent", "target"}:
        raise ValueError("subject must be 'agent' or 'target'")
    if not isinstance(collidable, bool) or not isinstance(blocks_los, bool):
        raise TypeError("entity policy flags must be bool")

    role_key = "npc_role" if subject == "agent" else "target_role"
    bounds_key = f"{subject}_bounds"
    offset_key = f"{subject}_los_offset"
    reference_bounds = np.asarray(
        values[0].geometry[bounds_key],
        dtype=np.float32,
    )
    reference_offset = np.asarray(
        values[0].geometry[offset_key],
        dtype=np.float32,
    )
    reference_model = _RuntimeModelEvidence.from_snapshot(values[0], subject)
    reference_model.validate_geometry(
        reference_bounds,
        reference_offset,
        subject,
    )
    versions: set[tuple[str, str, str]] = set()
    snapshot_sha256: list[str] = []
    for snapshot in values:
        role = snapshot.metadata.get(role_key)
        if role != type_id:
            raise ValueError(
                f"snapshot {role_key} {role!r} does not match {type_id!r}"
            )
        if subject == "target" and (
            not snapshot.metadata.get("target_present")
            or not snapshot.geometry["target_los_valid"]
        ):
            raise ValueError("target capture requires a valid native LOS endpoint")
        version = tuple(
            snapshot.metadata.get(key)
            for key in (
                "native_server_version",
                "worldgen_provider",
                "worldgen_version",
            )
        )
        if any(
            not isinstance(value, str)
            or not value
            or value.lower() in {"unknown", "uninitialized"}
            for value in version
        ):
            raise ValueError("native capture versions must be known strings")
        versions.add(version)
        if not np.array_equal(snapshot.geometry[bounds_key], reference_bounds):
            raise ValueError("native entity bounds changed between captures")
        if not np.array_equal(snapshot.geometry[offset_key], reference_offset):
            raise ValueError("native entity LOS offset changed between captures")
        model = _RuntimeModelEvidence.from_snapshot(snapshot, subject)
        model.validate_geometry(
            np.asarray(snapshot.geometry[bounds_key], dtype=np.float32),
            np.asarray(snapshot.geometry[offset_key], dtype=np.float32),
            subject,
        )
        if model != reference_model:
            raise ValueError("native runtime model evidence changed between captures")
        snapshot_sha256.append(snapshot.semantic_digest())
    if len(set(snapshot_sha256)) != len(snapshot_sha256):
        raise ValueError("native captures must have distinct semantic provenance")
    if len(versions) != 1:
        raise ValueError("native capture versions must be identical")
    if np.any(reference_bounds[:3] >= reference_bounds[3:]):
        raise ValueError("captured native entity bounds must have positive extent")
    if reference_model.model_present:
        model_asset_hash = _validated_sha256(
            model_asset_sha256,
            "model asset",
        )
    else:
        if model_asset_sha256 is not None:
            raise ValueError("model asset SHA-256 requires a present native model")
        model_asset_hash = None

    server_version, worldgen_provider, worldgen_version = next(iter(versions))
    manifest = {
        "schema": ENTITY_PROFILE_CAPTURE_SCHEMA,
        "version": ENTITY_PROFILE_CAPTURE_VERSION,
        "type_id": type_id,
        "subject": subject,
        "native_server_version": server_version,
        "worldgen_provider": worldgen_provider,
        "worldgen_version": worldgen_version,
        "bridge_jar_sha256": _validated_sha256(bridge_jar_sha256, "bridge JAR"),
        "server_jar_sha256": _validated_sha256(server_jar_sha256, "server JAR"),
        "role_asset_sha256": _validated_sha256(role_asset_sha256, "role asset"),
        "model_asset_sha256": model_asset_hash,
        "runtime_model_evidence": reference_model.manifest(),
        "snapshot_semantic_sha256": snapshot_sha256,
        "local_bounds": [float(value) for value in reference_bounds],
        "line_of_sight_offset": [float(value) for value in reference_offset],
        "policy": {
            "collidable": collidable,
            "blocks_los": blocks_los,
            "source": "caller_supplied_uncertified",
        },
        "native_certified": False,
        "uncertified_mechanics": [
            "collision_policy",
            "entity_body_los_policy",
            "rendered_mesh_visibility",
        ],
    }
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    profile = EntityGeometryProfile(
        type_id=type_id,
        local_bounds=tuple(float(value) for value in reference_bounds),
        collidable=collidable,
        blocks_los=blocks_los,
        provenance_sha256=_sha256(manifest_json.encode()),
        line_of_sight_offset=tuple(float(value) for value in reference_offset),
        native_certified=False,
    )
    return NativeEntityProfileCapture(profile=profile, manifest_json=manifest_json)


def _required_bool(metadata: Any, key: str) -> bool:
    value = metadata.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be bool")
    return value


def _required_string(metadata: Any, key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _required_float(metadata: Any, key: str) -> float:
    value = metadata.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{key} must be a finite number")
    return result


def _validated_sha256(value: str | None, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} SHA-256 must be a string")
    result = value.lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{label} SHA-256 must contain 64 hexadecimal digits")
    return result


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = [
    "ENTITY_PROFILE_CAPTURE_SCHEMA",
    "ENTITY_PROFILE_CAPTURE_VERSION",
    "NativeEntityProfileCapture",
    "capture_native_role_geometry_profile",
]
