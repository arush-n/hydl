"""Exact stable-fluid sidecars for experimental WorldGen V2 Regions.

The physical Region artifact remains unchanged.  This module captures the
stable Hytale ``Fluid.getId()`` string for every physical fluid cell, requires
exact forward/reverse replay, and stores a pickle-free palette-compressed NPZ
aligned to the complete 5x5x10 Region.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import tempfile
from typing import Any

import numpy as np

from _workspace import use_workspace_hytalerl


use_workspace_hytalerl()

from hytalegym.geometry.contract import FLAG_FLUID  # noqa: E402
from hytalegym.worldgen.region import (  # noqa: E402
    CAPTURE_CHUNK_COUNT,
    CAPTURE_CHUNKS_PER_AXIS,
    HEIGHT_SECTIONS,
    SECTION_VOLUME,
    NativeRegionSnapshot,
    RegionCapturePlan,
    capture_chunk_slot,
)

from native_v2_region_fluid_probe import (  # noqa: E402
    aligned_section_metrics,
    parse_fluid_semantic_section,
)


SNAPSHOT_SCHEMA = "hytalerl_region_fluid_semantic_snapshot_v1"
SNAPSHOT_VERSION = 1
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_SNAPSHOT_KEYS = {
    "metadata",
    "core_min_chunk_xz",
    "section_known",
    "cell_code",
    "palette_asset_id",
}

ProgressCallback = Callable[[int, int, Any], None]


@dataclass(frozen=True)
class NativeRegionFluidSemanticSnapshot:
    """Complete stable-fluid identity aligned to one physical Region."""

    core_min_chunk_xz: np.ndarray
    section_known: np.ndarray
    cell_code: np.ndarray
    palette: tuple[str, ...]
    source_region_semantic_sha256: str
    evidence_bridge_sha256: str

    def __post_init__(self) -> None:
        core = np.asarray(self.core_min_chunk_xz)
        if core.dtype != np.int32 or core.shape != (2,):
            raise ValueError("fluid snapshot core must be int32[2]")
        section_shape = (CAPTURE_CHUNK_COUNT, HEIGHT_SECTIONS)
        known = np.asarray(self.section_known)
        if known.dtype != np.bool_ or known.shape != section_shape:
            raise ValueError("fluid snapshot section mask has wrong shape")
        if not np.all(known):
            raise ValueError("fluid snapshot must be offline-complete")
        code = np.asarray(self.cell_code)
        if code.dtype != np.uint16 or code.shape != section_shape + (
            SECTION_VOLUME,
        ):
            raise ValueError("fluid snapshot code has wrong shape")
        palette = tuple(self.palette)
        if not palette or len(palette) > (1 << 16):
            raise ValueError("fluid snapshot palette exceeds uint16 capacity")
        if not all(isinstance(asset_id, str) for asset_id in palette):
            raise ValueError("fluid snapshot asset IDs must be strings")
        if palette[0] != "":
            raise ValueError("fluid snapshot palette zero must be empty")
        if any(
            not asset_id or asset_id != asset_id.strip()
            for asset_id in palette[1:]
        ):
            raise ValueError("fluid snapshot nonzero asset ID is invalid")
        if tuple(sorted(set(palette[1:]))) != palette[1:]:
            raise ValueError(
                "fluid snapshot nonzero palette must be unique and sorted"
            )
        if np.any(code.astype(np.uint32) >= len(palette)):
            raise ValueError("fluid snapshot code exceeds its palette")
        object.__setattr__(self, "core_min_chunk_xz", core.copy())
        object.__setattr__(self, "section_known", known.copy())
        object.__setattr__(self, "cell_code", code.copy())
        object.__setattr__(self, "palette", palette)
        object.__setattr__(
            self,
            "source_region_semantic_sha256",
            _require_sha256(self.source_region_semantic_sha256, "Region"),
        )
        object.__setattr__(
            self,
            "evidence_bridge_sha256",
            _require_sha256(self.evidence_bridge_sha256, "bridge"),
        )

    def semantic_sha256(self) -> str:
        """Hash stable cell meaning; the palette ordering is canonical."""

        digest = hashlib.sha256()
        digest.update(SNAPSHOT_SCHEMA.encode("ascii"))
        digest.update(struct.pack("<I", SNAPSHOT_VERSION))
        digest.update(self.source_region_semantic_sha256.encode("ascii"))
        digest.update(self.core_min_chunk_xz.astype("<i4").tobytes())
        digest.update(self.section_known.tobytes(order="C"))
        digest.update(struct.pack("<I", len(self.palette)))
        for asset_id in self.palette:
            encoded = asset_id.encode("utf-8")
            digest.update(struct.pack("<I", len(encoded)))
            digest.update(encoded)
        digest.update(self.cell_code.astype("<u2", copy=False).tobytes(order="C"))
        return digest.hexdigest()

    def asset_cell_counts(self, *, core_only: bool = False) -> dict[str, int]:
        code = self.cell_code
        if core_only:
            slots = np.asarray(
                [
                    chunk_z * CAPTURE_CHUNKS_PER_AXIS + chunk_x
                    for chunk_z in range(1, CAPTURE_CHUNKS_PER_AXIS - 1)
                    for chunk_x in range(1, CAPTURE_CHUNKS_PER_AXIS - 1)
                ],
                dtype=np.intp,
            )
            code = code[slots]
        used, counts = np.unique(code, return_counts=True)
        return {
            self.palette[int(index)]: int(count)
            for index, count in zip(used, counts, strict=True)
            if int(index) != 0
        }

    def save(self, path: str | Path) -> Path:
        """Atomically write a pickle-free stable-fluid sidecar."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": SNAPSHOT_SCHEMA,
            "version": SNAPSHOT_VERSION,
            "source_region_semantic_sha256": (
                self.source_region_semantic_sha256
            ),
            "evidence_bridge_sha256": self.evidence_bridge_sha256,
            "semantic_sha256": self.semantic_sha256(),
            "identity": "stable_hytale_fluid_getId_string",
        }
        payload = {
            "metadata": np.asarray(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            ),
            "core_min_chunk_xz": self.core_min_chunk_xz,
            "section_known": self.section_known,
            "cell_code": self.cell_code,
            "palette_asset_id": np.asarray(self.palette, dtype=np.str_),
        }
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as output:
                temporary = Path(output.name)
                np.savez_compressed(output, **payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "NativeRegionFluidSemanticSnapshot":
        with np.load(Path(path), allow_pickle=False) as archive:
            actual = set(archive.files)
            if actual != _SNAPSHOT_KEYS:
                raise ValueError(
                    "Region fluid-semantic snapshot fields differ: "
                    f"missing={sorted(_SNAPSHOT_KEYS - actual)}, "
                    f"extra={sorted(actual - _SNAPSHOT_KEYS)}"
                )
            metadata = json.loads(str(archive["metadata"].item()))
            if not isinstance(metadata, dict) or (
                metadata.get("schema") != SNAPSHOT_SCHEMA
                or metadata.get("version") != SNAPSHOT_VERSION
            ):
                raise ValueError("unsupported Region fluid-semantic snapshot")
            palette_array = archive["palette_asset_id"]
            if palette_array.ndim != 1 or palette_array.dtype.kind != "U":
                raise ValueError("fluid snapshot palette archive differs")
            snapshot = cls(
                core_min_chunk_xz=archive["core_min_chunk_xz"],
                section_known=archive["section_known"],
                cell_code=archive["cell_code"],
                palette=tuple(str(value) for value in palette_array.tolist()),
                source_region_semantic_sha256=str(
                    metadata.get("source_region_semantic_sha256", "")
                ),
                evidence_bridge_sha256=str(
                    metadata.get("evidence_bridge_sha256", "")
                ),
            )
        if metadata.get("identity") != "stable_hytale_fluid_getId_string":
            raise ValueError("fluid snapshot identity contract differs")
        if metadata.get("semantic_sha256") != snapshot.semantic_sha256():
            raise ValueError("fluid snapshot semantic digest differs")
        return snapshot


def capture_native_region_fluid_semantics(
    transport: Any,
    physical_snapshot: NativeRegionSnapshot,
    *,
    evidence_bridge_sha256: str,
    progress: ProgressCallback | None = None,
) -> NativeRegionFluidSemanticSnapshot:
    """Capture and exactly replay all 250 stable-fluid sections."""

    core_x = int(physical_snapshot.core_min_chunk_xz[0])
    core_z = int(physical_snapshot.core_min_chunk_xz[1])
    plan = RegionCapturePlan.complete(
        core_x,
        core_z,
        maximum_source_reach_blocks=0.0,
    )
    forward = _capture_pass(
        transport,
        physical_snapshot,
        plan.tasks,
        evidence_bridge_sha256=evidence_bridge_sha256,
        progress=progress,
    )
    reverse = _capture_pass(
        transport,
        physical_snapshot,
        tuple(reversed(plan.tasks)),
        evidence_bridge_sha256=evidence_bridge_sha256,
        progress=progress,
    )
    if (
        forward.semantic_sha256() != reverse.semantic_sha256()
        or forward.palette != reverse.palette
        or not np.array_equal(forward.cell_code, reverse.cell_code)
    ):
        raise ValueError("forward/reverse Region fluid semantics differ")
    require_fluid_snapshot_identity(
        forward,
        physical_snapshot,
        evidence_bridge_sha256=evidence_bridge_sha256,
    )
    return forward


def require_fluid_snapshot_identity(
    sidecar: NativeRegionFluidSemanticSnapshot,
    physical_snapshot: NativeRegionSnapshot,
    *,
    evidence_bridge_sha256: str,
) -> None:
    """Require provenance and exact FLAG_FLUID occupancy equality."""

    mismatches: dict[str, tuple[Any, Any]] = {}
    expected_region = physical_snapshot.semantic_digest()
    if sidecar.source_region_semantic_sha256 != expected_region:
        mismatches["source_region_semantic_sha256"] = (
            expected_region,
            sidecar.source_region_semantic_sha256,
        )
    expected_bridge = _require_sha256(evidence_bridge_sha256, "bridge")
    if sidecar.evidence_bridge_sha256 != expected_bridge:
        mismatches["evidence_bridge_sha256"] = (
            expected_bridge,
            sidecar.evidence_bridge_sha256,
        )
    expected_core = np.asarray(physical_snapshot.core_min_chunk_xz, dtype=np.int32)
    if not np.array_equal(sidecar.core_min_chunk_xz, expected_core):
        mismatches["core_min_chunk_xz"] = (
            expected_core.tolist(),
            sidecar.core_min_chunk_xz.tolist(),
        )
    physical_flags = physical_snapshot.cell_palette.flags[
        physical_snapshot.cell_code
    ]
    physical_fluid = (physical_flags & np.uint16(FLAG_FLUID)) != 0
    semantic_fluid = sidecar.cell_code != 0
    if not np.array_equal(semantic_fluid, physical_fluid):
        mismatches["FLAG_FLUID_cells"] = (
            int(np.count_nonzero(physical_fluid)),
            int(np.count_nonzero(semantic_fluid)),
        )
    if mismatches:
        raise ValueError(f"Region fluid sidecar identity differs: {mismatches}")


def _capture_pass(
    transport: Any,
    physical_snapshot: NativeRegionSnapshot,
    tasks: Sequence[Any],
    *,
    evidence_bridge_sha256: str,
    progress: ProgressCallback | None,
) -> NativeRegionFluidSemanticSnapshot:
    connection = getattr(transport, "_connection", None)
    if connection is None:
        raise RuntimeError("native connection is unavailable after reset")
    core_x = int(physical_snapshot.core_min_chunk_xz[0])
    core_z = int(physical_snapshot.core_min_chunk_xz[1])
    section_shape = (CAPTURE_CHUNK_COUNT, HEIGHT_SECTIONS)
    known = np.zeros(section_shape, dtype=np.bool_)
    local_sections: dict[
        tuple[int, int], tuple[np.ndarray, tuple[str, ...]]
    ] = {}
    stable_ids: set[str] = set()
    total = len(tasks)
    for done, task in enumerate(tasks, start=1):
        chunk_x = int(task.chunk_x)
        chunk_z = int(task.chunk_z)
        section_y = int(task.section_y)
        response = connection.send_and_recv(
            {
                "type": "region_fluid_semantics",
                "chunk_x": chunk_x,
                "chunk_z": chunk_z,
                "section_y": section_y,
            }
        )
        codes, palette = parse_fluid_semantic_section(
            response,
            expected_chunk_x=chunk_x,
            expected_chunk_z=chunk_z,
            expected_section_y=section_y,
        )
        slot = capture_chunk_slot(
            chunk_x,
            chunk_z,
            core_x - 1,
            core_z - 1,
        )
        physical_codes = physical_snapshot.cell_code[slot, section_y]
        physical_fluid = (
            physical_snapshot.cell_palette.flags[physical_codes]
            & np.uint16(FLAG_FLUID)
        ) != 0
        aligned_section_metrics(codes, palette, physical_fluid)
        if known[slot, section_y]:
            raise ValueError("duplicate Region fluid-semantic section")
        known[slot, section_y] = True
        local_sections[(slot, section_y)] = (codes, palette)
        stable_ids.update(palette[1:])
        if progress is not None:
            progress(done, total, task)
    if not np.all(known):
        raise ValueError("Region fluid-semantic pass is incomplete")

    palette = ("", *sorted(stable_ids))
    global_index = {asset_id: index for index, asset_id in enumerate(palette)}
    cell_code = np.zeros(
        section_shape + (SECTION_VOLUME,),
        dtype=np.uint16,
    )
    for (slot, section_y), (codes, local_palette) in local_sections.items():
        mapping = np.asarray(
            [global_index[asset_id] for asset_id in local_palette],
            dtype=np.uint16,
        )
        cell_code[slot, section_y] = mapping[codes]
    return NativeRegionFluidSemanticSnapshot(
        core_min_chunk_xz=np.asarray(
            physical_snapshot.core_min_chunk_xz,
            dtype=np.int32,
        ),
        section_known=known,
        cell_code=cell_code,
        palette=palette,
        source_region_semantic_sha256=physical_snapshot.semantic_digest(),
        evidence_bridge_sha256=evidence_bridge_sha256,
    )


def _require_sha256(value: object, label: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{label} SHA-256 is invalid")
    return text.lower()


__all__ = [
    "NativeRegionFluidSemanticSnapshot",
    "SNAPSHOT_SCHEMA",
    "SNAPSHOT_VERSION",
    "capture_native_region_fluid_semantics",
    "require_fluid_snapshot_identity",
]
