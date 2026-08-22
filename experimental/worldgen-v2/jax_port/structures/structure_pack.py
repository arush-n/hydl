"""Marker-backed placed structure sidecars for exact WorldGen V2 Regions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping, Sequence

from hytalegym.worldgen.region import (
    CHUNK_SIZE,
    CORE_CHUNKS_PER_AXIS,
    MIN_Y,
    WORLD_HEIGHT,
)

from ..bundles.bundle import V2JaxArtifact, V2JaxBundle
from .structure_contract import STRUCTURE_MARKER_CONTRACT_SHA256


STRUCTURE_REGISTRY_SCHEMA = "hytalerl_worldgen_v2_structure_registry_v1"
STRUCTURE_REGISTRY_VERSION = 1
STRUCTURE_SNAPSHOT_SCHEMA = "hytalerl_worldgen_v2_structure_snapshot_v1"
STRUCTURE_SNAPSHOT_VERSION = 1
STRUCTURE_PACK_SCHEMA = "hytalerl_worldgen_v2_structure_pack_v1"
STRUCTURE_PACK_VERSION = 1
STRUCTURE_PACK_AUTHORITY = (
    "native_FromPrefabInstance_marker_plus_pinned_authoring_registry"
)
_REGISTRY_FIELDS = frozenset(
    {"schema", "version", "entries", "registry_semantic_sha256"}
)
_REGISTRY_ENTRY_FIELDS = frozenset(
    {
        "marker_asset_id",
        "structure_asset_id",
        "kind",
        "marker_local_position",
        "local_bounds_min",
        "local_bounds_max",
        "geometry_present",
    }
)
_SNAPSHOT_FIELDS = frozenset(
    {
        "schema",
        "version",
        "authority",
        "source_region_semantic_sha256",
        "evidence_bridge_sha256",
        "marker_contract_sha256",
        "registry_semantic_sha256",
        "instances",
        "snapshot_semantic_sha256",
    }
)
_INSTANCE_FIELDS = frozenset(
    {
        "instance_id",
        "structure_asset_id",
        "marker_asset_id",
        "kind",
        "anchor_world",
        "quarter_turns",
        "bounds_min",
        "bounds_max",
        "capture_coverage",
        "geometry_present",
        "native_worldgen_id",
        "native_prefab_instance_id",
    }
)
_PACK_FIELDS = frozenset(
    {
        "schema",
        "version",
        "authority",
        "source_bundle_semantic_sha256",
        "evidence_bridge_sha256",
        "marker_contract_sha256",
        "registry_semantic_sha256",
        "registry_path",
        "registry_file_sha256",
        "coverage",
        "entries",
        "pack_semantic_sha256",
    }
)
_PACK_COVERAGE_FIELDS = frozenset(
    {
        "kind",
        "entry_count",
        "source_artifact_count",
        "complete",
        "structures",
    }
)
_PACK_ENTRY_FIELDS = frozenset(
    {
        "structure",
        "seed",
        "source_region_semantic_sha256",
        "path",
        "file_sha256",
        "snapshot_semantic_sha256",
        "instance_count",
    }
)
_KINDS = frozenset({"prefab", "static_prop"})
_COVERAGE = frozenset({"complete", "clipped_to_capture"})
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")


@dataclass(frozen=True)
class StructureRegistryEntry:
    marker_asset_id: str
    structure_asset_id: str
    kind: str
    marker_local_position: tuple[float, float, float]
    local_bounds_min: tuple[int, int, int]
    local_bounds_max: tuple[int, int, int]
    geometry_present: bool


@dataclass(frozen=True)
class V2StructureRegistry:
    path: Path
    semantic_sha256: str
    entries: tuple[StructureRegistryEntry, ...]

    @classmethod
    def load(cls, path: str | Path) -> "V2StructureRegistry":
        source = Path(path).resolve()
        value = _json_object(source)
        _exact_fields(value, _REGISTRY_FIELDS, "structure registry")
        if value["schema"] != STRUCTURE_REGISTRY_SCHEMA:
            raise ValueError("unsupported V2 structure registry schema")
        if _exact_int(value["version"], "registry version") != (
            STRUCTURE_REGISTRY_VERSION
        ):
            raise ValueError("unsupported V2 structure registry version")
        semantic = _sha256(
            value["registry_semantic_sha256"],
            "registry semantic",
        )
        if semantic != _semantic_digest(value, "registry_semantic_sha256"):
            raise ValueError("V2 structure registry semantic SHA-256 mismatch")
        raw_entries = value["entries"]
        if not isinstance(raw_entries, list) or not raw_entries:
            raise ValueError("V2 structure registry must contain entries")
        entries = tuple(_registry_entry(row) for row in raw_entries)
        marker_ids = [row.marker_asset_id for row in entries]
        if len(set(marker_ids)) != len(marker_ids):
            raise ValueError("V2 structure marker asset IDs must be unique")
        if list(marker_ids) != sorted(marker_ids):
            raise ValueError("V2 structure registry entries must be sorted")
        return cls(path=source, semantic_sha256=semantic, entries=entries)


@dataclass(frozen=True)
class V2NativeStructureMarker:
    """One generated marker entity preserving native prefab grouping."""

    marker_asset_id: str
    position: tuple[float, float, float]
    yaw_radians: float
    native_worldgen_id: int
    native_prefab_instance_id: int


@dataclass(frozen=True)
class V2StructureInstance:
    instance_id: str
    structure_asset_id: str
    marker_asset_id: str
    kind: str
    anchor_world: tuple[int, int, int]
    quarter_turns: int
    bounds_min: tuple[int, int, int]
    bounds_max: tuple[int, int, int]
    capture_coverage: str
    geometry_present: bool
    native_worldgen_id: int
    native_prefab_instance_id: int


@dataclass(frozen=True)
class V2StructureSnapshot:
    path: Path | None
    source_region_semantic_sha256: str
    evidence_bridge_sha256: str
    marker_contract_sha256: str
    registry_semantic_sha256: str
    instances: tuple[V2StructureInstance, ...]
    semantic_sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "V2StructureSnapshot":
        source = Path(path).resolve()
        value = _json_object(source)
        _exact_fields(value, _SNAPSHOT_FIELDS, "structure snapshot")
        if value["schema"] != STRUCTURE_SNAPSHOT_SCHEMA:
            raise ValueError("unsupported V2 structure snapshot schema")
        if _exact_int(value["version"], "snapshot version") != (
            STRUCTURE_SNAPSHOT_VERSION
        ):
            raise ValueError("unsupported V2 structure snapshot version")
        if value["authority"] != STRUCTURE_PACK_AUTHORITY:
            raise ValueError("V2 structure snapshot authority changed")
        if _sha256(value["marker_contract_sha256"], "marker contract") != (
            STRUCTURE_MARKER_CONTRACT_SHA256
        ):
            raise ValueError("V2 structure marker contract changed")
        semantic = _sha256(
            value["snapshot_semantic_sha256"],
            "snapshot semantic",
        )
        if semantic != _semantic_digest(value, "snapshot_semantic_sha256"):
            raise ValueError("V2 structure snapshot semantic SHA-256 mismatch")
        raw_instances = value["instances"]
        if not isinstance(raw_instances, list):
            raise TypeError("V2 structure instances must be a list")
        instances = tuple(_instance(row) for row in raw_instances)
        if len({row.instance_id for row in instances}) != len(instances):
            raise ValueError("V2 structure instance IDs must be unique")
        if tuple(row.instance_id for row in instances) != tuple(
            sorted(row.instance_id for row in instances)
        ):
            raise ValueError("V2 structure instances must be sorted")
        return cls(
            path=source,
            source_region_semantic_sha256=_sha256(
                value["source_region_semantic_sha256"],
                "source Region semantic",
            ),
            evidence_bridge_sha256=_sha256(
                value["evidence_bridge_sha256"],
                "evidence bridge",
            ),
            marker_contract_sha256=STRUCTURE_MARKER_CONTRACT_SHA256,
            registry_semantic_sha256=_sha256(
                value["registry_semantic_sha256"],
                "registry semantic",
            ),
            instances=instances,
            semantic_sha256=semantic,
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        value = _snapshot_value(self)
        _atomic_json(destination, value)
        restored = V2StructureSnapshot.load(destination)
        if restored.semantic_sha256 != self.semantic_sha256:
            raise ValueError("saved V2 structure snapshot differs")
        return destination


@dataclass(frozen=True)
class V2StructurePackEntry:
    structure: str
    seed: int
    source_region_semantic_sha256: str
    path: str
    file_sha256: str
    snapshot_semantic_sha256: str
    instance_count: int


@dataclass(frozen=True)
class V2StructurePack:
    manifest_path: Path
    manifest: Mapping[str, Any]
    bundle: V2JaxBundle
    entries: tuple[V2StructurePackEntry, ...]
    registry: V2StructureRegistry

    @classmethod
    def load(
        cls,
        manifest_path: str | Path,
        bundle: V2JaxBundle,
        *,
        verify_snapshots: bool = True,
    ) -> "V2StructurePack":
        if not isinstance(bundle, V2JaxBundle):
            raise TypeError("bundle must be a V2JaxBundle")
        path = Path(manifest_path).resolve()
        value = _json_object(path)
        _exact_fields(value, _PACK_FIELDS, "structure pack")
        if value["schema"] != STRUCTURE_PACK_SCHEMA:
            raise ValueError("unsupported V2 structure pack schema")
        if _exact_int(value["version"], "structure pack version") != (
            STRUCTURE_PACK_VERSION
        ):
            raise ValueError("unsupported V2 structure pack version")
        if value["authority"] != STRUCTURE_PACK_AUTHORITY:
            raise ValueError("V2 structure pack authority changed")
        if _sha256(
            value["source_bundle_semantic_sha256"],
            "source bundle semantic",
        ) != bundle.semantic_sha256:
            raise ValueError("V2 structure pack names another bundle")
        source_bridge = _sha256(
            bundle.manifest["source"]["bridge_jar_sha256"],
            "source bundle bridge",
        )
        bridge = _sha256(
            value["evidence_bridge_sha256"],
            "structure evidence bridge",
        )
        if bridge != source_bridge:
            raise ValueError("V2 structure and Region evidence bridges differ")
        if _sha256(value["marker_contract_sha256"], "marker contract") != (
            STRUCTURE_MARKER_CONTRACT_SHA256
        ):
            raise ValueError("V2 structure marker contract changed")
        registry_semantic = _sha256(
            value["registry_semantic_sha256"],
            "registry semantic",
        )
        registry_path = _contained_path(
            path.parent,
            _relative_path(value["registry_path"]),
        )
        if _file_sha256(registry_path) != _sha256(
            value["registry_file_sha256"],
            "registry file",
        ):
            raise ValueError("V2 structure registry file SHA-256 changed")
        registry = V2StructureRegistry.load(registry_path)
        if registry.semantic_sha256 != registry_semantic:
            raise ValueError("V2 structure pack registry semantic changed")
        if _sha256(value["pack_semantic_sha256"], "pack semantic") != (
            _semantic_digest(value, "pack_semantic_sha256")
        ):
            raise ValueError("V2 structure pack semantic SHA-256 mismatch")
        raw_entries = value["entries"]
        if not isinstance(raw_entries, list):
            raise TypeError("V2 structure pack entries must be a list")
        entries = tuple(_pack_entry(row) for row in raw_entries)
        identities = {
            (row.structure, row.seed, row.source_region_semantic_sha256)
            for row in entries
        }
        if len(identities) != len(entries):
            raise ValueError("V2 structure pack entries must be unique")
        bundle_ids = {
            (row.structure, row.seed, row.region_semantic_sha256)
            for row in bundle.artifacts
        }
        if not identities.issubset(bundle_ids):
            raise ValueError("V2 structure entry does not belong to the bundle")
        coverage = _mapping(value["coverage"], "structure coverage")
        _exact_fields(coverage, _PACK_COVERAGE_FIELDS, "structure coverage")
        if coverage["kind"] != "explicit_artifact_subset":
            raise ValueError("unsupported V2 structure coverage kind")
        if _nonnegative_int(coverage["entry_count"], "coverage entry count") != (
            len(entries)
        ):
            raise ValueError("V2 structure coverage entry count changed")
        if _positive_int(
            coverage["source_artifact_count"],
            "coverage source artifact count",
        ) != len(bundle.artifacts):
            raise ValueError("V2 structure source artifact count changed")
        complete = coverage["complete"]
        if not isinstance(complete, bool) or complete != (
            len(entries) == len(bundle.artifacts)
        ):
            raise ValueError("V2 structure complete flag changed")
        if coverage["structures"] != sorted({row.structure for row in entries}):
            raise ValueError("V2 structure coverage structures changed")
        result = cls(path, value, bundle, entries, registry)
        if verify_snapshots:
            for entry in entries:
                snapshot = result._load_entry(entry)
                if snapshot.evidence_bridge_sha256 != bridge:
                    raise ValueError("V2 structure snapshot bridge changed")
                if snapshot.registry_semantic_sha256 != registry_semantic:
                    raise ValueError("V2 structure snapshot registry changed")
        return result

    @property
    def semantic_sha256(self) -> str:
        return str(self.manifest["pack_semantic_sha256"])

    def covers(self, artifacts: Sequence[V2JaxArtifact]) -> bool:
        available = {
            (row.structure, row.seed, row.source_region_semantic_sha256)
            for row in self.entries
        }
        return all(
            (row.structure, row.seed, row.region_semantic_sha256) in available
            for row in artifacts
        )

    def load_for_artifacts(
        self,
        artifacts: Sequence[V2JaxArtifact],
    ) -> tuple[V2StructureSnapshot, ...]:
        by_source = {
            (row.structure, row.seed, row.source_region_semantic_sha256): row
            for row in self.entries
        }
        missing = [
            f"{row.structure}/{row.seed}"
            for row in artifacts
            if (row.structure, row.seed, row.region_semantic_sha256)
            not in by_source
        ]
        if missing:
            raise ValueError(
                "V2 structure pack does not cover selected worlds: "
                + ", ".join(missing)
            )
        return tuple(
            self._load_entry(
                by_source[
                    (row.structure, row.seed, row.region_semantic_sha256)
                ]
            )
            for row in artifacts
        )

    def _load_entry(self, entry: V2StructurePackEntry) -> V2StructureSnapshot:
        path = _contained_path(self.manifest_path.parent, entry.path)
        if _file_sha256(path) != entry.file_sha256:
            raise ValueError("V2 structure snapshot file SHA-256 changed")
        snapshot = V2StructureSnapshot.load(path)
        if snapshot.source_region_semantic_sha256 != (
            entry.source_region_semantic_sha256
        ):
            raise ValueError("V2 structure snapshot source Region changed")
        if snapshot.semantic_sha256 != entry.snapshot_semantic_sha256:
            raise ValueError("V2 structure snapshot semantic changed")
        if len(snapshot.instances) != entry.instance_count:
            raise ValueError("V2 structure snapshot instance count changed")
        return snapshot


def create_structure_registry(
    path: str | Path,
    entries: Sequence[StructureRegistryEntry],
) -> V2StructureRegistry:
    destination = Path(path).resolve()
    rows = tuple(entries)
    if not rows:
        raise ValueError("structure registry entries cannot be empty")
    encoded = [_registry_entry_value(row) for row in rows]
    encoded.sort(key=lambda row: str(row["marker_asset_id"]))
    value: dict[str, Any] = {
        "schema": STRUCTURE_REGISTRY_SCHEMA,
        "version": STRUCTURE_REGISTRY_VERSION,
        "entries": encoded,
        "registry_semantic_sha256": "",
    }
    value["registry_semantic_sha256"] = _semantic_digest(
        value,
        "registry_semantic_sha256",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(destination, value)
    return V2StructureRegistry.load(destination)


def compile_structure_snapshot(
    artifact: V2JaxArtifact,
    markers: Sequence[V2NativeStructureMarker],
    registry: V2StructureRegistry,
    *,
    evidence_bridge_sha256: str,
) -> V2StructureSnapshot:
    """Resolve native marker poses through a pinned authoring registry."""

    if not isinstance(artifact, V2JaxArtifact):
        raise TypeError("artifact must be a V2JaxArtifact")
    if not isinstance(registry, V2StructureRegistry):
        raise TypeError("registry must be a V2StructureRegistry")
    bridge = _sha256(evidence_bridge_sha256, "evidence bridge")
    by_marker = {row.marker_asset_id: row for row in registry.entries}
    instances: list[V2StructureInstance] = []
    native_groups: set[tuple[int, int]] = set()
    for marker in markers:
        if not isinstance(marker, V2NativeStructureMarker):
            raise TypeError("markers must contain V2NativeStructureMarker values")
        marker_asset_id = _asset_id(marker.marker_asset_id, "marker asset ID")
        registry_entry = by_marker.get(marker_asset_id)
        if registry_entry is None:
            raise ValueError(f"unregistered V2 structure marker: {marker_asset_id}")
        position = _float3(marker.position, "marker position")
        turns = _quarter_turns(marker.yaw_radians)
        worldgen_id = _nonnegative_int(
            marker.native_worldgen_id,
            "native_worldgen_id",
        )
        prefab_instance_id = _nonnegative_int(
            marker.native_prefab_instance_id,
            "native_prefab_instance_id",
        )
        group = (worldgen_id, prefab_instance_id)
        if group in native_groups:
            raise ValueError(
                "one authored marker is required per native prefab instance"
            )
        native_groups.add(group)
        rotated_marker = _rotate_float_xz(
            registry_entry.marker_local_position,
            turns,
        )
        anchor_float = tuple(
            position[index] - rotated_marker[index] for index in range(3)
        )
        anchor = tuple(_near_integer(value, "structure anchor") for value in anchor_float)
        bounds_min, bounds_max = _world_bounds(
            registry_entry.local_bounds_min,
            registry_entry.local_bounds_max,
            anchor,
            turns,
        )
        coverage = _capture_coverage(artifact, bounds_min, bounds_max)
        identity_value = {
            "source_region_semantic_sha256": artifact.region_semantic_sha256,
            "structure_asset_id": registry_entry.structure_asset_id,
            "marker_asset_id": marker_asset_id,
            "anchor_world": list(anchor),
            "quarter_turns": turns,
            "native_worldgen_id": worldgen_id,
            "native_prefab_instance_id": prefab_instance_id,
        }
        instance_id = hashlib.sha256(
            json.dumps(
                identity_value,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        instances.append(
            V2StructureInstance(
                instance_id=instance_id,
                structure_asset_id=registry_entry.structure_asset_id,
                marker_asset_id=marker_asset_id,
                kind=registry_entry.kind,
                anchor_world=anchor,
                quarter_turns=turns,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                capture_coverage=coverage,
                geometry_present=registry_entry.geometry_present,
                native_worldgen_id=worldgen_id,
                native_prefab_instance_id=prefab_instance_id,
            )
        )
    instances.sort(key=lambda row: row.instance_id)
    provisional = V2StructureSnapshot(
        path=None,
        source_region_semantic_sha256=artifact.region_semantic_sha256,
        evidence_bridge_sha256=bridge,
        marker_contract_sha256=STRUCTURE_MARKER_CONTRACT_SHA256,
        registry_semantic_sha256=registry.semantic_sha256,
        instances=tuple(instances),
        semantic_sha256="",
    )
    value = _snapshot_value(provisional)
    semantic = _semantic_digest(value, "snapshot_semantic_sha256")
    return V2StructureSnapshot(
        path=None,
        source_region_semantic_sha256=artifact.region_semantic_sha256,
        evidence_bridge_sha256=bridge,
        marker_contract_sha256=STRUCTURE_MARKER_CONTRACT_SHA256,
        registry_semantic_sha256=registry.semantic_sha256,
        instances=tuple(instances),
        semantic_sha256=semantic,
    )


def publish_v2_structure_pack(
    manifest_path: str | Path,
    bundle: V2JaxBundle,
    snapshot_paths: Sequence[str | Path],
    *,
    registry: V2StructureRegistry,
) -> V2StructurePack:
    if not isinstance(bundle, V2JaxBundle):
        raise TypeError("bundle must be a V2JaxBundle")
    if not isinstance(registry, V2StructureRegistry):
        raise TypeError("registry must be a V2StructureRegistry")
    destination = Path(manifest_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = tuple(Path(value).resolve() for value in snapshot_paths)
    if not paths:
        raise ValueError("at least one V2 structure snapshot is required")
    by_source = {
        row.region_semantic_sha256: row for row in bundle.artifacts
    }
    rows: list[dict[str, Any]] = []
    bridges: set[str] = set()
    registries: set[str] = set()
    for path in paths:
        snapshot = V2StructureSnapshot.load(path)
        artifact = by_source.get(snapshot.source_region_semantic_sha256)
        if artifact is None:
            raise ValueError("V2 structure snapshot names an unknown Region")
        rows.append(
            {
                "structure": artifact.structure,
                "seed": artifact.seed,
                "source_region_semantic_sha256": (
                    artifact.region_semantic_sha256
                ),
                "path": _relative_path(
                    path.relative_to(destination.parent).as_posix()
                ),
                "file_sha256": _file_sha256(path),
                "snapshot_semantic_sha256": snapshot.semantic_sha256,
                "instance_count": len(snapshot.instances),
            }
        )
        bridges.add(snapshot.evidence_bridge_sha256)
        registries.add(snapshot.registry_semantic_sha256)
    if len(bridges) != 1 or len(registries) != 1:
        raise ValueError("V2 structure pack provenance must be homogeneous")
    if next(iter(registries)) != registry.semantic_sha256:
        raise ValueError("V2 structure snapshots name another registry")
    source_bridge = _sha256(
        bundle.manifest["source"]["bridge_jar_sha256"],
        "source bundle bridge",
    )
    if next(iter(bridges)) != source_bridge:
        raise ValueError("V2 structure and Region evidence bridges differ")
    rows.sort(
        key=lambda row: (
            row["structure"],
            row["seed"],
            row["source_region_semantic_sha256"],
        )
    )
    if len(
        {
            (row["structure"], row["seed"], row["source_region_semantic_sha256"])
            for row in rows
        }
    ) != len(rows):
        raise ValueError("V2 structure snapshot sources must be unique")
    value: dict[str, Any] = {
        "schema": STRUCTURE_PACK_SCHEMA,
        "version": STRUCTURE_PACK_VERSION,
        "authority": STRUCTURE_PACK_AUTHORITY,
        "source_bundle_semantic_sha256": bundle.semantic_sha256,
        "evidence_bridge_sha256": next(iter(bridges)),
        "marker_contract_sha256": STRUCTURE_MARKER_CONTRACT_SHA256,
        "registry_semantic_sha256": next(iter(registries)),
        "registry_path": _relative_path(
            registry.path.relative_to(destination.parent).as_posix()
        ),
        "registry_file_sha256": _file_sha256(registry.path),
        "coverage": {
            "kind": "explicit_artifact_subset",
            "entry_count": len(rows),
            "source_artifact_count": len(bundle.artifacts),
            "complete": len(rows) == len(bundle.artifacts),
            "structures": sorted({str(row["structure"]) for row in rows}),
        },
        "entries": rows,
        "pack_semantic_sha256": "",
    }
    value["pack_semantic_sha256"] = _semantic_digest(
        value,
        "pack_semantic_sha256",
    )
    _atomic_json(destination, value)
    return V2StructurePack.load(destination, bundle, verify_snapshots=True)


def _registry_entry(value: object) -> StructureRegistryEntry:
    row = _mapping(value, "structure registry entry")
    _exact_fields(row, _REGISTRY_ENTRY_FIELDS, "structure registry entry")
    kind = _nonempty(row["kind"], "structure kind")
    if kind not in _KINDS:
        raise ValueError("unsupported V2 structure kind")
    geometry = row["geometry_present"]
    if not isinstance(geometry, bool):
        raise TypeError("geometry_present must be boolean")
    minimum = _int3(row["local_bounds_min"], "local_bounds_min")
    maximum = _int3(row["local_bounds_max"], "local_bounds_max")
    if any(a >= b for a, b in zip(minimum, maximum, strict=True)):
        raise ValueError("structure registry bounds must be positive")
    return StructureRegistryEntry(
        marker_asset_id=_asset_id(row["marker_asset_id"], "marker asset ID"),
        structure_asset_id=_asset_id(
            row["structure_asset_id"],
            "structure asset ID",
        ),
        kind=kind,
        marker_local_position=_float3(
            row["marker_local_position"],
            "marker_local_position",
        ),
        local_bounds_min=minimum,
        local_bounds_max=maximum,
        geometry_present=geometry,
    )


def _registry_entry_value(row: StructureRegistryEntry) -> dict[str, Any]:
    if not isinstance(row, StructureRegistryEntry):
        raise TypeError("entries must contain StructureRegistryEntry values")
    return {
        "marker_asset_id": row.marker_asset_id,
        "structure_asset_id": row.structure_asset_id,
        "kind": row.kind,
        "marker_local_position": list(row.marker_local_position),
        "local_bounds_min": list(row.local_bounds_min),
        "local_bounds_max": list(row.local_bounds_max),
        "geometry_present": row.geometry_present,
    }


def _instance(value: object) -> V2StructureInstance:
    row = _mapping(value, "structure instance")
    _exact_fields(row, _INSTANCE_FIELDS, "structure instance")
    kind = _nonempty(row["kind"], "structure kind")
    if kind not in _KINDS:
        raise ValueError("unsupported V2 structure kind")
    coverage = _nonempty(row["capture_coverage"], "capture coverage")
    if coverage not in _COVERAGE:
        raise ValueError("unsupported V2 structure capture coverage")
    geometry = row["geometry_present"]
    if not isinstance(geometry, bool):
        raise TypeError("geometry_present must be boolean")
    minimum = _int3(row["bounds_min"], "bounds_min")
    maximum = _int3(row["bounds_max"], "bounds_max")
    if any(a >= b for a, b in zip(minimum, maximum, strict=True)):
        raise ValueError("structure instance bounds must be positive")
    turns = _nonnegative_int(row["quarter_turns"], "quarter_turns")
    if turns > 3:
        raise ValueError("quarter_turns must be in [0, 3]")
    return V2StructureInstance(
        instance_id=_sha256(row["instance_id"], "instance ID"),
        structure_asset_id=_asset_id(
            row["structure_asset_id"],
            "structure asset ID",
        ),
        marker_asset_id=_asset_id(row["marker_asset_id"], "marker asset ID"),
        kind=kind,
        anchor_world=_int3(row["anchor_world"], "anchor_world"),
        quarter_turns=turns,
        bounds_min=minimum,
        bounds_max=maximum,
        capture_coverage=coverage,
        geometry_present=geometry,
        native_worldgen_id=_nonnegative_int(
            row["native_worldgen_id"],
            "native_worldgen_id",
        ),
        native_prefab_instance_id=_nonnegative_int(
            row["native_prefab_instance_id"],
            "native_prefab_instance_id",
        ),
    )


def _pack_entry(value: object) -> V2StructurePackEntry:
    row = _mapping(value, "structure pack entry")
    _exact_fields(row, _PACK_ENTRY_FIELDS, "structure pack entry")
    return V2StructurePackEntry(
        structure=_nonempty(row["structure"], "structure"),
        seed=_nonnegative_int(row["seed"], "seed"),
        source_region_semantic_sha256=_sha256(
            row["source_region_semantic_sha256"],
            "source Region semantic",
        ),
        path=_relative_path(row["path"]),
        file_sha256=_sha256(row["file_sha256"], "snapshot file"),
        snapshot_semantic_sha256=_sha256(
            row["snapshot_semantic_sha256"],
            "snapshot semantic",
        ),
        instance_count=_nonnegative_int(
            row["instance_count"],
            "instance_count",
        ),
    )


def _snapshot_value(snapshot: V2StructureSnapshot) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": STRUCTURE_SNAPSHOT_SCHEMA,
        "version": STRUCTURE_SNAPSHOT_VERSION,
        "authority": STRUCTURE_PACK_AUTHORITY,
        "source_region_semantic_sha256": (
            snapshot.source_region_semantic_sha256
        ),
        "evidence_bridge_sha256": snapshot.evidence_bridge_sha256,
        "marker_contract_sha256": snapshot.marker_contract_sha256,
        "registry_semantic_sha256": snapshot.registry_semantic_sha256,
        "instances": [
            {
                "instance_id": row.instance_id,
                "structure_asset_id": row.structure_asset_id,
                "marker_asset_id": row.marker_asset_id,
                "kind": row.kind,
                "anchor_world": list(row.anchor_world),
                "quarter_turns": row.quarter_turns,
                "bounds_min": list(row.bounds_min),
                "bounds_max": list(row.bounds_max),
                "capture_coverage": row.capture_coverage,
                "geometry_present": row.geometry_present,
                "native_worldgen_id": row.native_worldgen_id,
                "native_prefab_instance_id": row.native_prefab_instance_id,
            }
            for row in snapshot.instances
        ],
        "snapshot_semantic_sha256": snapshot.semantic_sha256,
    }
    return value


def _world_bounds(
    minimum: tuple[int, int, int],
    maximum: tuple[int, int, int],
    anchor: tuple[int, int, int],
    turns: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    corners = []
    for x in (minimum[0], maximum[0]):
        for z in (minimum[2], maximum[2]):
            rx, rz = _rotate_xz(x, z, turns)
            corners.append((anchor[0] + rx, anchor[2] + rz))
    xs = [row[0] for row in corners]
    zs = [row[1] for row in corners]
    return (
        (min(xs), anchor[1] + minimum[1], min(zs)),
        (max(xs), anchor[1] + maximum[1], max(zs)),
    )


def _capture_coverage(
    artifact: V2JaxArtifact,
    minimum: tuple[int, int, int],
    maximum: tuple[int, int, int],
) -> str:
    core_x = artifact.core_min_chunk_xz[0] * CHUNK_SIZE
    core_z = artifact.core_min_chunk_xz[1] * CHUNK_SIZE
    capture_min = (core_x - CHUNK_SIZE, MIN_Y, core_z - CHUNK_SIZE)
    capture_max = (
        core_x + (CORE_CHUNKS_PER_AXIS + 1) * CHUNK_SIZE,
        MIN_Y + WORLD_HEIGHT,
        core_z + (CORE_CHUNKS_PER_AXIS + 1) * CHUNK_SIZE,
    )
    if any(
        maximum[index] <= capture_min[index]
        or minimum[index] >= capture_max[index]
        for index in range(3)
    ):
        raise ValueError("structure marker bounds do not intersect the capture")
    if all(
        capture_min[index] <= minimum[index]
        and maximum[index] <= capture_max[index]
        for index in range(3)
    ):
        return "complete"
    return "clipped_to_capture"


def _quarter_turns(yaw: object) -> int:
    if isinstance(yaw, bool) or not isinstance(yaw, (int, float)):
        raise TypeError("marker yaw must be numeric")
    value = float(yaw)
    if not math.isfinite(value):
        raise ValueError("marker yaw must be finite")
    raw = value / (math.pi / 2.0)
    nearest = round(raw)
    if abs(raw - nearest) > 1.0e-4:
        raise ValueError("structure marker yaw is not a quarter turn")
    return int(nearest) % 4


def _rotate_xz(x: int, z: int, turns: int) -> tuple[int, int]:
    if turns == 0:
        return x, z
    if turns == 1:
        return -z, x
    if turns == 2:
        return -x, -z
    return z, -x


def _rotate_float_xz(
    position: tuple[float, float, float],
    turns: int,
) -> tuple[float, float, float]:
    x, y, z = position
    rx, rz = _rotate_xz_float(x, z, turns)
    return rx, y, rz


def _rotate_xz_float(x: float, z: float, turns: int) -> tuple[float, float]:
    if turns == 0:
        return x, z
    if turns == 1:
        return -z, x
    if turns == 2:
        return -x, -z
    return z, -x


def _near_integer(value: float, label: str) -> int:
    nearest = round(value)
    if abs(value - nearest) > 1.0e-4:
        raise ValueError(f"{label} does not resolve to an integer block anchor")
    if not -(1 << 31) <= nearest < (1 << 31):
        raise ValueError(f"{label} exceeds int32")
    return int(nearest)


def _semantic_digest(value: Mapping[str, Any], field: str) -> str:
    stable = dict(value)
    stable[field] = ""
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields changed")


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256")
    return value.lower()


def _asset_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty portable string")
    return value


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _exact_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    result = _exact_int(value, label)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _positive_int(value: object, label: str) -> int:
    result = _exact_int(value, label)
    if result < 1:
        raise ValueError(f"{label} must be positive")
    return result


def _int3(value: object, label: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must be one XYZ row")
    result = tuple(_exact_int(item, label) for item in value)
    if any(not -(1 << 31) <= item < (1 << 31) for item in result):
        raise ValueError(f"{label} must fit int32")
    return result


def _float3(value: object, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must be one XYZ row")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{label} must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{label} must be finite")
        result.append(number)
    return tuple(result)  # type: ignore[return-value]


def _relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("sidecar path must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("sidecar path must be safe and archive-relative")
    return path.as_posix()


def _contained_path(root: Path, relative: str) -> Path:
    path = (root / PurePosixPath(_relative_path(relative))).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("sidecar path escapes its pack root") from error
    if not path.is_file():
        raise ValueError(f"sidecar file does not exist: {path}")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()
