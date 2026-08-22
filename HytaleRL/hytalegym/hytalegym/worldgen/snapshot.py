"""Portable, lossless local snapshots from Hytale's native world generator.

Hytale remains the generator oracle. This module stores exact local collision,
support, movement, and LOS state in compressed fixed arrays so JAX can replay
native-generated situations without reimplementing or guessing the generator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hytalegym.geometry import (
    GEOMETRY_SCHEMA,
    GEOMETRY_VERSION,
    empty_geometry,
    geometry_space,
)
from hytalegym.geometry.contract import FLAG_FLUID


SNAPSHOT_SCHEMA = "hytalerl_native_worldgen_snapshot_v1"
SNAPSHOT_VERSION = 1
_META_KEY = "__metadata_json__"
_AGENT_KEY = "__agent_position__"
_TARGET_KEY = "__target_position__"
_GEOMETRY_PREFIX = "geometry__"

# Runtime asset indices are useful diagnostics but are deliberately excluded
# from the portable semantic checksum. Exact boxes and flags are the authority.
_SEMANTIC_GEOMETRY_KEYS = (
    "origin",
    "cell_mask",
    "flags",
    "fluid_level",
    "support",
    "block_damage",
    "fluid_damage",
    "movement",
    "fluid_movement",
    "collision_boxes",
    "collision_box_mask",
    "agent_bounds",
    "target_bounds",
    "agent_los_offset",
    "target_los_offset",
    "contacts",
    "contact_mask",
    "grounded",
    "ceiling_contact",
    "target_los",
    "target_los_valid",
)


@dataclass(frozen=True)
class NativeWorldgenSnapshot:
    """One exact native-generated local world frame and scenario placement."""

    metadata: Mapping[str, Any]
    geometry: Mapping[str, Any]
    agent_position: np.ndarray
    target_position: np.ndarray

    def __post_init__(self) -> None:
        metadata = dict(self.metadata)
        if metadata.get("schema") != SNAPSHOT_SCHEMA:
            raise ValueError("unsupported worldgen snapshot schema")
        if int(metadata.get("version", -1)) != SNAPSHOT_VERSION:
            raise ValueError("unsupported worldgen snapshot version")
        geometry_identity = (
            metadata.get("geometry_schema"),
            int(metadata.get("geometry_version", -1)),
        )
        legacy_geometry = geometry_identity == ("hytale_geometry_v4", 4)
        if geometry_identity != (GEOMETRY_SCHEMA, GEOMETRY_VERSION) and not (
            legacy_geometry
        ):
            raise ValueError("snapshot geometry schema does not match this build")

        geometry = _copy_geometry(
            self.geometry,
            legacy_geometry=legacy_geometry,
        )
        if not geometry_space().contains(geometry):
            raise ValueError("snapshot contains invalid fixed-shape geometry")
        if not int(geometry["available"]):
            raise ValueError("worldgen snapshot geometry is unavailable")
        if not int(geometry["exact_collision_shapes"]):
            raise ValueError("worldgen snapshot lacks native collision shapes")

        agent = _position(self.agent_position, "agent_position")
        target = _position(self.target_position, "target_position")
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "agent_position", agent)
        object.__setattr__(self, "target_position", target)

    @classmethod
    def from_native(
        cls,
        observation: Mapping[str, Any],
        info: Mapping[str, Any],
        *,
        seed: int,
    ) -> "NativeWorldgenSnapshot":
        """Capture a parsed Gym observation from the native Hytale backend."""

        if info.get("backend") != "native":
            raise ValueError("worldgen snapshots require the native backend")
        world_template = info.get("world_template")
        if world_template not in {"hytale", "hytale_generator"}:
            raise ValueError(
                "worldgen snapshots require a native generated-world template"
            )
        reported_seed = info.get("worldgen_seed", seed)
        if (
            not isinstance(reported_seed, int)
            or isinstance(reported_seed, bool)
            or reported_seed != seed
        ):
            raise ValueError("worldgen snapshot seed does not match native info")
        geometry = observation.get("geometry")
        if not isinstance(geometry, Mapping):
            raise ValueError("native observation is missing geometry")
        target_present = bool(info.get("target_present", False))
        target_position = np.asarray(
            [
                info.get("target_x", 0.0),
                info.get("target_y", 0.0),
                info.get("target_z", 0.0),
            ],
            dtype=np.float64,
        )
        model_evidence_version = info.get(
            "native_entity_model_evidence_version",
            0,
        )
        if (
            not isinstance(model_evidence_version, int)
            or isinstance(model_evidence_version, bool)
            or model_evidence_version not in {0, 1}
        ):
            raise ValueError("unsupported native entity model evidence version")
        metadata = {
            "schema": SNAPSHOT_SCHEMA,
            "version": SNAPSHOT_VERSION,
            "geometry_schema": GEOMETRY_SCHEMA,
            "geometry_version": GEOMETRY_VERSION,
            "seed": int(seed),
            "native_server_version": str(
                info.get("native_server_version", "unknown")
            ),
            "world_template": str(world_template),
            "worldgen_provider": str(info.get("worldgen_provider", "unknown")),
            "worldgen_version": str(info.get("worldgen_version", "unknown")),
            "worldgen_structure": str(info.get("worldgen_structure", "")),
            "npc_role": str(info.get("npc_role", "")),
            "task": str(info.get("task", "")),
            "target_present": target_present,
            "target_role": str(info.get("target_role", "")),
            "native_entity_model_evidence_version": model_evidence_version,
        }
        for subject in ("agent", "target"):
            metadata.update(_entity_model_metadata(info, subject))
        return cls(
            metadata=metadata,
            geometry=geometry,
            agent_position=np.asarray(observation["position"], dtype=np.float64),
            target_position=target_position,
        )

    def save(self, path: str | Path) -> Path:
        """Write a compressed NPZ with JSON metadata and no pickle payloads."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            _META_KEY: np.asarray(
                json.dumps(
                    dict(self.metadata),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            _AGENT_KEY: self.agent_position,
            _TARGET_KEY: self.target_position,
        }
        for name, value in self.geometry.items():
            payload[f"{_GEOMETRY_PREFIX}{name}"] = np.asarray(value)
        np.savez_compressed(destination, **payload)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "NativeWorldgenSnapshot":
        """Load and validate a snapshot without enabling object deserialization."""

        with np.load(Path(path), allow_pickle=False) as archive:
            required = {_META_KEY, _AGENT_KEY, _TARGET_KEY}
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"worldgen snapshot is missing {sorted(missing)}")
            metadata = json.loads(str(archive[_META_KEY].item()))
            geometry = empty_geometry()
            for name, template in geometry.items():
                key = f"{_GEOMETRY_PREFIX}{name}"
                if key not in archive.files:
                    if (
                        name == "fluid_fill_height"
                        and metadata.get("geometry_schema")
                        == "hytale_geometry_v4"
                        and int(metadata.get("geometry_version", -1)) == 4
                    ):
                        fluid = (
                            np.asarray(geometry["flags"], dtype=np.int32)
                            & np.int32(FLAG_FLUID)
                        ) != 0
                        geometry[name] = np.where(
                            fluid,
                            np.float32(-1.0),
                            np.float32(0.0),
                        )
                        continue
                    raise ValueError(f"worldgen snapshot is missing {key}")
                value = archive[key]
                if isinstance(template, np.ndarray):
                    if value.shape != template.shape:
                        raise ValueError(
                            f"{key} shape {value.shape} != {template.shape}"
                        )
                    geometry[name] = np.asarray(value, dtype=template.dtype).copy()
                else:
                    geometry[name] = int(np.asarray(value).item())
            return cls(
                metadata=metadata,
                geometry=geometry,
                agent_position=archive[_AGENT_KEY].copy(),
                target_position=archive[_TARGET_KEY].copy(),
            )

    def semantic_digest(self) -> str:
        """Hash portable behavior, excluding process-local runtime asset IDs."""

        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    key: self.metadata[key]
                    for key in (
                        "schema",
                        "version",
                        "geometry_schema",
                        "geometry_version",
                        "seed",
                        "native_server_version",
                        "worldgen_provider",
                        "worldgen_version",
                        "npc_role",
                        "task",
                        "target_present",
                        "target_role",
                    )
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        if "world_template" in self.metadata:
            digest.update(
                json.dumps(
                    {
                        "world_template": self.metadata["world_template"],
                        "worldgen_structure": self.metadata.get(
                            "worldgen_structure",
                            "",
                        ),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        for position in (self.agent_position, self.target_position):
            _update_array_digest(digest, position)
        for name in _SEMANTIC_GEOMETRY_KEYS:
            digest.update(name.encode("ascii"))
            _update_array_digest(digest, np.asarray(self.geometry[name]))
        if self.metadata.get("geometry_schema") == GEOMETRY_SCHEMA:
            digest.update(b"fluid_fill_height")
            _update_array_digest(
                digest,
                np.asarray(self.geometry["fluid_fill_height"]),
            )
        return digest.hexdigest()

    def jax_geometry(self, *, batch_size: int | None = None):
        """Convert outside JIT to the fixed JAX world state."""

        from hytalegym.jax.world import geometry_state_from_numpy

        return geometry_state_from_numpy(
            self.geometry,
            batch_size=batch_size,
            require_exact=True,
        )


def _position(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite values")
    return result.copy()


def _copy_geometry(
    source: Mapping[str, Any],
    *,
    legacy_geometry: bool = False,
) -> dict[str, Any]:
    result = empty_geometry()
    for name, template in result.items():
        if name not in source:
            if name == "fluid_fill_height" and legacy_geometry:
                fluid = (
                    np.asarray(result["flags"], dtype=np.int32)
                    & np.int32(FLAG_FLUID)
                ) != 0
                result[name] = np.where(
                    fluid,
                    np.float32(-1.0),
                    np.float32(0.0),
                )
                continue
            raise ValueError(f"geometry is missing {name}")
        if isinstance(template, np.ndarray):
            result[name] = np.asarray(source[name], dtype=template.dtype).copy()
        else:
            result[name] = int(source[name])
    return result


def _entity_model_metadata(
    info: Mapping[str, Any],
    subject: str,
) -> dict[str, Any]:
    prefix = f"{subject}_"
    result = {
        f"{prefix}model_present": _metadata_bool(
            info,
            f"{prefix}model_present",
        ),
        f"{prefix}model_asset_id": _metadata_string(
            info,
            f"{prefix}model_asset_id",
        ),
        f"{prefix}bounding_box_present": _metadata_bool(
            info,
            f"{prefix}bounding_box_present",
        ),
        f"{prefix}entity_scale_present": _metadata_bool(
            info,
            f"{prefix}entity_scale_present",
        ),
    }
    for name in ("model_scale", "model_eye_height", "entity_scale"):
        key = f"{prefix}{name}"
        raw = info.get(key, 0.0)
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise ValueError(f"{key} must be a finite number")
        value = float(raw)
        if not np.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        result[key] = value
    return result


def _metadata_bool(info: Mapping[str, Any], key: str) -> bool:
    value = info.get(key, False)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be bool")
    return value


def _metadata_string(info: Mapping[str, Any], key: str) -> str:
    value = info.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _update_array_digest(digest: Any, value: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(value)
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
