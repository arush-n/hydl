"""Palette-compressed block identity/affordance sidecars for Region v1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
import operator
from pathlib import Path
import struct
import tempfile
from typing import Callable, Protocol

import numpy as np

from hytalegym.geometry.contract import (
    FLAG_FLUID,
    FLAG_HAS_FLUID_MOVEMENT_SETTINGS,
)
from hytalegym.worldgen.block_affordances import (
    BLOCK_AFFORDANCE_TAGS,
    GATHER_TYPES,
    block_affordance_dictionary_sha256,
)
from hytalegym.worldgen.block_actions import BLOCK_SEMANTIC_KEY_WORDS
from hytalegym.worldgen.region.contract import (
    CAPTURE_CHUNK_COUNT,
    CAPTURE_CHUNKS_PER_AXIS,
    HEIGHT_SECTIONS,
    SECTION_VOLUME,
    capture_chunk_slot,
)
from hytalegym.worldgen.region.capture import (
    RegionCapturePlan,
    RegionCaptureTask,
)
from hytalegym.worldgen.region.native import (
    NativeRegionTransport,
    capture_native_region_pass,
)
from hytalegym.worldgen.region.snapshot import NativeRegionSnapshot
from hytalegym.worldgen.region.stability import (
    RegionSemanticComparison,
    RegionSemanticMismatchError,
    compare_region_semantics,
)


REGION_BLOCK_SEMANTIC_SECTION_SCHEMA = (
    "hytalerl_native_region_block_semantics_v1"
)
REGION_BLOCK_SEMANTIC_SECTION_VERSION = 1
REGION_BLOCK_SEMANTIC_CODE_ENCODING = "uint16_le_y_z_x"
REGION_BLOCK_SEMANTIC_CODE_BYTES = SECTION_VOLUME * 2
REGION_BLOCK_SEMANTIC_PALETTE_CAPACITY = 1 << 16
REGION_BLOCK_SEMANTIC_ALIGNMENT_CELL_CAPACITY = 4
REGION_BLOCK_SEMANTIC_SNAPSHOT_SCHEMA = (
    "hytalerl_region_block_semantic_snapshot_v1"
)
REGION_BLOCK_SEMANTIC_SNAPSHOT_VERSION = 1
_SNAPSHOT_KEYS = {
    "metadata",
    "core_min_chunk_xz",
    "section_known",
    "cell_code",
    "palette_semantic_key",
    "palette_asset_key",
    "palette_valid",
    "palette_affordance_tags",
    "palette_gather_type_index",
    "palette_required_tool_quality",
    "palette_rotation_index",
}


class NativeRegionBlockSemanticTransport(NativeRegionTransport, Protocol):
    def capture_region_block_semantic_section(
        self,
        chunk_x: int,
        chunk_z: int,
        section_y: int,
    ) -> Mapping[str, object]: ...


ProgressCallback = Callable[[int, int, RegionCaptureTask], None]


@dataclass(frozen=True)
class ProjectedRegionBlockSemanticCapture:
    """Block semantics whose dry physical base matched a frozen Region."""

    snapshot: "NativeRegionBlockSemanticSnapshot"
    physical_comparison: RegionSemanticComparison


@dataclass(frozen=True)
class RegionBlockSemanticEntry:
    """One portable block state; entry zero is canonical non-block."""

    semantic_key: tuple[int, ...]
    asset_key: tuple[int, ...]
    valid: bool
    affordance_tags: int
    gather_type_index: int
    required_tool_quality: int
    rotation_index: int

    def __post_init__(self) -> None:
        key = _key_words(self.semantic_key, "block semantic key")
        asset_key = _key_words(self.asset_key, "block asset key")
        tags = _nonnegative_int(self.affordance_tags, "affordance_tags")
        gather = _nonnegative_int(
            self.gather_type_index,
            "gather_type_index",
        )
        quality = _nonnegative_int(
            self.required_tool_quality,
            "required_tool_quality",
        )
        if quality > 32_767:
            raise ValueError("required_tool_quality must fit int16")
        rotation = _nonnegative_int(self.rotation_index, "rotation_index")
        if tags & ~((1 << len(BLOCK_AFFORDANCE_TAGS)) - 1):
            raise ValueError("block semantic entry has unknown affordance bits")
        if gather >= len(GATHER_TYPES):
            raise ValueError("block semantic gather type is unknown")
        if self.valid:
            if not any(key) or not any(asset_key):
                raise ValueError("valid block identity cannot be zero")
        elif any(
            (
                any(key),
                any(asset_key),
                tags,
                gather,
                quality,
                rotation,
            )
        ):
            raise ValueError("invalid block semantic entry must be zero")
        object.__setattr__(self, "semantic_key", tuple(int(value) for value in key))
        object.__setattr__(self, "asset_key", asset_key)
        object.__setattr__(self, "affordance_tags", tags)
        object.__setattr__(self, "gather_type_index", gather)
        object.__setattr__(self, "required_tool_quality", quality)
        object.__setattr__(self, "rotation_index", rotation)

    @classmethod
    def air(cls) -> "RegionBlockSemanticEntry":
        key = (0,) * BLOCK_SEMANTIC_KEY_WORDS
        return cls(key, key, False, 0, 0, 0, 0)


@dataclass(frozen=True)
class NativeRegionBlockSemanticSection:
    chunk_x: int
    chunk_z: int
    section_y: int
    cell_code: np.ndarray
    palette: tuple[RegionBlockSemanticEntry, ...]

    def __post_init__(self) -> None:
        chunk_x = _int32(self.chunk_x, "chunk_x")
        chunk_z = _int32(self.chunk_z, "chunk_z")
        section_y = _nonnegative_int(self.section_y, "section_y")
        if section_y >= HEIGHT_SECTIONS:
            raise ValueError("semantic section Y is outside the world")
        code = np.asarray(self.cell_code)
        if code.dtype != np.uint16 or code.shape != (SECTION_VOLUME,):
            raise ValueError("semantic section code must be uint16[32768]")
        palette = tuple(self.palette)
        if not 1 <= len(palette) <= REGION_BLOCK_SEMANTIC_PALETTE_CAPACITY:
            raise ValueError("semantic section palette exceeds uint16 capacity")
        if palette[0] != RegionBlockSemanticEntry.air():
            raise ValueError("semantic section palette entry zero must be air")
        if np.any(code.astype(np.uint32) >= len(palette)):
            raise ValueError("semantic section code exceeds its palette")
        object.__setattr__(self, "chunk_x", chunk_x)
        object.__setattr__(self, "chunk_z", chunk_z)
        object.__setattr__(self, "section_y", section_y)
        object.__setattr__(self, "cell_code", code.copy())
        object.__setattr__(self, "palette", palette)


@dataclass(frozen=True)
class NativeRegionBlockSemanticSnapshot:
    """Complete 5x5x10 semantic sidecar aligned to one Region snapshot."""

    core_min_chunk_xz: np.ndarray
    section_known: np.ndarray
    cell_code: np.ndarray
    palette: tuple[RegionBlockSemanticEntry, ...]
    source_region_semantic_sha256: str
    evidence_bridge_sha256: str

    def __post_init__(self) -> None:
        core = np.asarray(self.core_min_chunk_xz)
        if core.dtype != np.int32 or core.shape != (2,):
            raise ValueError("semantic snapshot core must be int32[2]")
        known = np.asarray(self.section_known)
        code = np.asarray(self.cell_code)
        section_shape = (CAPTURE_CHUNK_COUNT, HEIGHT_SECTIONS)
        if known.dtype != np.bool_ or known.shape != section_shape:
            raise ValueError("semantic snapshot section mask has wrong shape")
        if code.dtype != np.uint16 or code.shape != section_shape + (
            SECTION_VOLUME,
        ):
            raise ValueError("semantic snapshot code has wrong shape")
        if not np.all(known):
            raise ValueError("semantic snapshot must be offline-complete")
        palette = tuple(self.palette)
        if not 1 <= len(palette) <= REGION_BLOCK_SEMANTIC_PALETTE_CAPACITY:
            raise ValueError("semantic snapshot palette exceeds capacity")
        if palette[0] != RegionBlockSemanticEntry.air():
            raise ValueError("semantic snapshot palette entry zero must be air")
        if np.any(code.astype(np.uint32) >= len(palette)):
            raise ValueError("semantic snapshot code exceeds its palette")
        region_sha = _sha256(self.source_region_semantic_sha256, "Region")
        bridge_sha = _sha256(self.evidence_bridge_sha256, "bridge")
        object.__setattr__(self, "core_min_chunk_xz", core.copy())
        object.__setattr__(self, "section_known", known.copy())
        object.__setattr__(self, "cell_code", code.copy())
        object.__setattr__(self, "palette", palette)
        object.__setattr__(self, "source_region_semantic_sha256", region_sha)
        object.__setattr__(self, "evidence_bridge_sha256", bridge_sha)

    def semantic_sha256(self) -> str:
        """Hash expanded meaning, independently of palette ordering."""

        entry_hash = np.stack(
            [_entry_sha256(entry) for entry in self.palette]
        )
        digest = hashlib.sha256()
        digest.update(REGION_BLOCK_SEMANTIC_SNAPSHOT_SCHEMA.encode("ascii"))
        digest.update(struct.pack("<I", REGION_BLOCK_SEMANTIC_SNAPSHOT_VERSION))
        digest.update(self.source_region_semantic_sha256.encode("ascii"))
        digest.update(self.core_min_chunk_xz.astype("<i4").tobytes())
        digest.update(self.section_known.tobytes())
        for slot in range(CAPTURE_CHUNK_COUNT):
            for section_y in range(HEIGHT_SECTIONS):
                digest.update(entry_hash[self.cell_code[slot, section_y]].tobytes())
        return digest.hexdigest()

    def save(self, path: str | Path) -> Path:
        """Atomically write a pickle-free frozen semantic sidecar."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": REGION_BLOCK_SEMANTIC_SNAPSHOT_SCHEMA,
            "version": REGION_BLOCK_SEMANTIC_SNAPSHOT_VERSION,
            "source_region_semantic_sha256": (
                self.source_region_semantic_sha256
            ),
            "evidence_bridge_sha256": self.evidence_bridge_sha256,
            "semantic_sha256": self.semantic_sha256(),
        }
        payload = {
            "metadata": np.asarray(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            ),
            "core_min_chunk_xz": self.core_min_chunk_xz,
            "section_known": self.section_known,
            "cell_code": self.cell_code,
            "palette_semantic_key": np.asarray(
                [entry.semantic_key for entry in self.palette],
                dtype=np.uint32,
            ),
            "palette_asset_key": np.asarray(
                [entry.asset_key for entry in self.palette],
                dtype=np.uint32,
            ),
            "palette_valid": np.asarray(
                [entry.valid for entry in self.palette],
                dtype=np.bool_,
            ),
            "palette_affordance_tags": np.asarray(
                [entry.affordance_tags for entry in self.palette],
                dtype=np.uint16,
            ),
            "palette_gather_type_index": np.asarray(
                [entry.gather_type_index for entry in self.palette],
                dtype=np.uint8,
            ),
            "palette_required_tool_quality": np.asarray(
                [entry.required_tool_quality for entry in self.palette],
                dtype=np.int16,
            ),
            "palette_rotation_index": np.asarray(
                [entry.rotation_index for entry in self.palette],
                dtype=np.int32,
            ),
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
    def load(cls, path: str | Path) -> "NativeRegionBlockSemanticSnapshot":
        with np.load(Path(path), allow_pickle=False) as archive:
            actual = set(archive.files)
            if actual != _SNAPSHOT_KEYS:
                raise ValueError(
                    "Region block-semantic snapshot fields differ: "
                    f"missing={sorted(_SNAPSHOT_KEYS - actual)}, "
                    f"extra={sorted(actual - _SNAPSHOT_KEYS)}"
                )
            metadata = json.loads(str(archive["metadata"].item()))
            if not isinstance(metadata, dict) or (
                metadata.get("schema")
                != REGION_BLOCK_SEMANTIC_SNAPSHOT_SCHEMA
                or metadata.get("version")
                != REGION_BLOCK_SEMANTIC_SNAPSHOT_VERSION
            ):
                raise ValueError(
                    "unsupported Region block-semantic snapshot"
                )
            semantic_key = archive["palette_semantic_key"]
            asset_key = archive["palette_asset_key"]
            valid = archive["palette_valid"]
            tags = archive["palette_affordance_tags"]
            gather = archive["palette_gather_type_index"]
            quality = archive["palette_required_tool_quality"]
            rotation = archive["palette_rotation_index"]
            palette_size = semantic_key.shape[0]
            expected = {
                "semantic_key": (palette_size, BLOCK_SEMANTIC_KEY_WORDS),
                "asset_key": (palette_size, BLOCK_SEMANTIC_KEY_WORDS),
                "valid": (palette_size,),
                "tags": (palette_size,),
                "gather": (palette_size,),
                "quality": (palette_size,),
                "rotation": (palette_size,),
            }
            actual_shapes = {
                "semantic_key": semantic_key.shape,
                "asset_key": asset_key.shape,
                "valid": valid.shape,
                "tags": tags.shape,
                "gather": gather.shape,
                "quality": quality.shape,
                "rotation": rotation.shape,
            }
            if actual_shapes != expected:
                raise ValueError(
                    "Region block-semantic palette arrays differ"
                )
            expected_dtypes = {
                "semantic_key": np.dtype(np.uint32),
                "asset_key": np.dtype(np.uint32),
                "valid": np.dtype(np.bool_),
                "tags": np.dtype(np.uint16),
                "gather": np.dtype(np.uint8),
                "quality": np.dtype(np.int16),
                "rotation": np.dtype(np.int32),
            }
            actual_dtypes = {
                "semantic_key": semantic_key.dtype,
                "asset_key": asset_key.dtype,
                "valid": valid.dtype,
                "tags": tags.dtype,
                "gather": gather.dtype,
                "quality": quality.dtype,
                "rotation": rotation.dtype,
            }
            if actual_dtypes != expected_dtypes:
                raise ValueError(
                    "Region block-semantic palette dtypes differ"
                )
            palette = tuple(
                RegionBlockSemanticEntry(
                    semantic_key=tuple(int(value) for value in semantic_key[i]),
                    asset_key=tuple(int(value) for value in asset_key[i]),
                    valid=bool(valid[i]),
                    affordance_tags=int(tags[i]),
                    gather_type_index=int(gather[i]),
                    required_tool_quality=int(quality[i]),
                    rotation_index=int(rotation[i]),
                )
                for i in range(palette_size)
            )
            snapshot = cls(
                core_min_chunk_xz=archive["core_min_chunk_xz"],
                section_known=archive["section_known"],
                cell_code=archive["cell_code"],
                palette=palette,
                source_region_semantic_sha256=metadata.get(
                    "source_region_semantic_sha256"
                ),
                evidence_bridge_sha256=metadata.get(
                    "evidence_bridge_sha256"
                ),
            )
        if snapshot.semantic_sha256() != metadata.get("semantic_sha256"):
            raise ValueError(
                "Region block-semantic snapshot SHA-256 mismatch"
            )
        return snapshot


class NativeRegionBlockSemanticAssembler:
    """Merge bounded native section palettes without trusting local ordinals."""

    def __init__(
        self,
        core_min_chunk_xz: Sequence[int],
        *,
        source_region_semantic_sha256: str,
        evidence_bridge_sha256: str,
    ):
        core = np.asarray(core_min_chunk_xz, dtype=np.int64)
        if core.shape != (2,) or np.any(
            (core < np.iinfo(np.int32).min)
            | (core > np.iinfo(np.int32).max)
        ):
            raise ValueError("semantic assembler core must be int32[2]")
        self._core = core.astype(np.int32)
        self._source_region_sha256 = _sha256(
            source_region_semantic_sha256,
            "Region",
        )
        self._bridge_sha256 = _sha256(evidence_bridge_sha256, "bridge")
        self._known = np.zeros(
            (CAPTURE_CHUNK_COUNT, HEIGHT_SECTIONS),
            dtype=np.bool_,
        )
        self._code = np.zeros(
            (CAPTURE_CHUNK_COUNT, HEIGHT_SECTIONS, SECTION_VOLUME),
            dtype=np.uint16,
        )
        air = RegionBlockSemanticEntry.air()
        self._palette = [air]
        self._indices = {air: 0}

    def add_wire_section(self, response: Mapping[str, object]) -> None:
        self.add_section(native_region_block_semantic_section_from_wire(response))

    def add_section(self, section: NativeRegionBlockSemanticSection) -> None:
        if not isinstance(section, NativeRegionBlockSemanticSection):
            raise TypeError("section must be NativeRegionBlockSemanticSection")
        slot = capture_chunk_slot(
            section.chunk_x,
            section.chunk_z,
            int(self._core[0]) - 1,
            int(self._core[1]) - 1,
        )
        if self._known[slot, section.section_y]:
            raise ValueError("duplicate Region block-semantic section")
        local_to_global = np.zeros(len(section.palette), dtype=np.uint16)
        for local, entry in enumerate(section.palette):
            global_index = self._indices.get(entry)
            if global_index is None:
                if (
                    len(self._palette)
                    >= REGION_BLOCK_SEMANTIC_PALETTE_CAPACITY
                ):
                    raise ValueError(
                        "Region block-semantic palette exceeds uint16 capacity"
                    )
                global_index = len(self._palette)
                self._indices[entry] = global_index
                self._palette.append(entry)
            local_to_global[local] = global_index
        self._code[slot, section.section_y] = local_to_global[
            section.cell_code
        ]
        self._known[slot, section.section_y] = True

    def build(self) -> NativeRegionBlockSemanticSnapshot:
        if not np.all(self._known):
            raise ValueError("Region block-semantic capture is incomplete")
        return NativeRegionBlockSemanticSnapshot(
            core_min_chunk_xz=self._core,
            section_known=self._known,
            cell_code=self._code,
            palette=tuple(self._palette),
            source_region_semantic_sha256=self._source_region_sha256,
            evidence_bridge_sha256=self._bridge_sha256,
        )


def capture_native_region_block_semantics(
    transport: NativeRegionBlockSemanticTransport,
    physical_snapshot: NativeRegionSnapshot,
    *,
    evidence_bridge_sha256: str,
    progress: ProgressCallback | None = None,
) -> NativeRegionBlockSemanticSnapshot:
    """Capture the complete semantic sidecar for one frozen physical Region."""

    _require_semantic_capture_base(physical_snapshot)
    core = physical_snapshot.core_min_chunk_xz
    live_physical = capture_native_region_pass(
        transport,
        requested_core=(int(core[0]), int(core[1])),
        order="forward",
    )
    if (
        live_physical.semantic_digest()
        != physical_snapshot.semantic_digest()
    ):
        raise ValueError(
            "live physical Region differs from semantic capture base"
        )
    return _capture_block_semantic_sections(
        transport,
        physical_snapshot,
        evidence_bridge_sha256=evidence_bridge_sha256,
        progress=progress,
    )


def capture_projected_native_region_block_semantics(
    transport: NativeRegionBlockSemanticTransport,
    physical_snapshot: NativeRegionSnapshot,
    *,
    evidence_bridge_sha256: str,
    progress: ProgressCallback | None = None,
) -> ProjectedRegionBlockSemanticCapture:
    """Capture block identity when only non-damaging fluid state drifted.

    This is not Region artifact or seed equality. It is valid only for the
    block-semantic sidecar because the dry block, filler, collision, support,
    damage, and movement meanings must remain exact.
    """

    _require_semantic_capture_base(physical_snapshot)
    core = physical_snapshot.core_min_chunk_xz
    live_physical = capture_native_region_pass(
        transport,
        requested_core=(int(core[0]), int(core[1])),
        order="forward",
    )
    comparison = compare_region_semantics(
        physical_snapshot,
        live_physical,
        require_same_world=False,
        difference_limit=REGION_BLOCK_SEMANTIC_ALIGNMENT_CELL_CAPACITY,
        native_evidence_jar_sha256=evidence_bridge_sha256,
    )
    if not _block_semantic_physical_alignment(comparison):
        raise RegionSemanticMismatchError(
            comparison,
            context="Region block-semantic dry physical alignment",
            first_snapshot=physical_snapshot,
            second_snapshot=live_physical,
        )
    snapshot = _capture_block_semantic_sections(
        transport,
        physical_snapshot,
        evidence_bridge_sha256=evidence_bridge_sha256,
        progress=progress,
    )
    return ProjectedRegionBlockSemanticCapture(snapshot, comparison)


def _require_semantic_capture_base(
    physical_snapshot: NativeRegionSnapshot,
) -> None:
    if not isinstance(physical_snapshot, NativeRegionSnapshot):
        raise TypeError("physical_snapshot must be NativeRegionSnapshot")
    if not np.all(physical_snapshot.section_known):
        raise ValueError("semantic capture requires an offline-complete Region")


def _capture_block_semantic_sections(
    transport: NativeRegionBlockSemanticTransport,
    physical_snapshot: NativeRegionSnapshot,
    *,
    evidence_bridge_sha256: str,
    progress: ProgressCallback | None,
) -> NativeRegionBlockSemanticSnapshot:
    core = physical_snapshot.core_min_chunk_xz
    assembler = NativeRegionBlockSemanticAssembler(
        core,
        source_region_semantic_sha256=physical_snapshot.semantic_digest(),
        evidence_bridge_sha256=evidence_bridge_sha256,
    )
    plan = RegionCapturePlan.complete(
        int(core[0]),
        int(core[1]),
        maximum_source_reach_blocks=0.0,
    )
    total = len(plan.tasks)
    for index, task in enumerate(plan.tasks, start=1):
        assembler.add_wire_section(
            transport.capture_region_block_semantic_section(
                task.chunk_x,
                task.chunk_z,
                task.section_y,
            )
        )
        if progress is not None:
            progress(index, total, task)
    return assembler.build()


def region_block_semantic_projection_capture_contract() -> dict[str, object]:
    return {
        "schema": "hytalerl_region_block_semantic_projection_capture_v1",
        "version": 1,
        "output_contract_sha256": region_block_semantic_contract_sha256(),
        "physical_base": "exact_frozen_region_artifact",
        "live_alignment": [
            "region_seed_projection_equal",
            (
                "or_each_changed_cell_has_non_damaging_fluid_and_exact_"
                "nonfluid_flags_movement_support_damage_shape_and_filler"
            ),
        ],
        "preserved": [
            "block_flags",
            "block_movement",
            "support",
            "block_and_damaging_fluid_damage",
            "collision_shape",
            "filler_root",
        ],
        "changed_cell_capacity": (
            REGION_BLOCK_SEMANTIC_ALIGNMENT_CELL_CAPACITY
        ),
        "capacity_overflow": "reject_without_partial_publication",
        "reason": (
            "block_identity_is_independent_of_non_damaging_fluid_occupancy"
        ),
        "artifact_or_seed_equality": False,
        "partial_publication": "unsupported",
        "fail_closed": True,
    }


def _block_semantic_physical_alignment(
    comparison: RegionSemanticComparison,
) -> bool:
    if comparison.equal or comparison.seed_projection_equal:
        return True
    if (
        comparison.layout_mismatches
        or comparison.differences_truncated
        or not comparison.differences
    ):
        return False
    fluid_flags = FLAG_FLUID | FLAG_HAS_FLUID_MOVEMENT_SETTINGS
    for difference in comparison.differences:
        first = difference.first
        second = difference.second
        if (
            not (first.flags | second.flags) & FLAG_FLUID
            or first.fluid_damage != 0
            or second.fluid_damage != 0
            or (first.flags & ~fluid_flags)
            != (second.flags & ~fluid_flags)
            or first.support != second.support
            or first.block_damage != second.block_damage
            or first.movement != second.movement
            or first.shape_box_count != second.shape_box_count
            or first.collision_boxes != second.collision_boxes
            or first.filler_root_offset != second.filler_root_offset
        ):
            return False
    return True


def region_block_semantic_projection_capture_contract_sha256() -> str:
    return hashlib.sha256(
        json.dumps(
            region_block_semantic_projection_capture_contract(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def native_region_block_semantic_section_from_wire(
    response: Mapping[str, object],
) -> NativeRegionBlockSemanticSection:
    raw = dict(response)
    if raw.get("type") != "region_block_semantics":
        raise ValueError("native response is not Region block semantics")
    if raw.get("schema") != REGION_BLOCK_SEMANTIC_SECTION_SCHEMA:
        raise ValueError("unsupported Region block-semantic schema")
    if (
        _nonnegative_int(raw.get("version"), "version")
        != REGION_BLOCK_SEMANTIC_SECTION_VERSION
    ):
        raise ValueError("unsupported Region block-semantic version")
    if raw.get("code_encoding") != REGION_BLOCK_SEMANTIC_CODE_ENCODING:
        raise ValueError("unsupported Region block-semantic code encoding")
    if (
        raw.get("affordance_dictionary_sha256")
        != block_affordance_dictionary_sha256()
    ):
        raise ValueError("Region block affordance dictionary differs")
    payload = raw.get("cell_codes")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError("Region block-semantic codes must be binary")
    encoded = bytes(payload)
    if len(encoded) != REGION_BLOCK_SEMANTIC_CODE_BYTES:
        raise ValueError("Region block-semantic code length differs")
    raw_palette = raw.get("palette")
    if not isinstance(raw_palette, Sequence) or isinstance(
        raw_palette,
        (str, bytes, bytearray),
    ):
        raise ValueError("Region block-semantic palette must be an array")
    palette = tuple(_wire_entry(value) for value in raw_palette)
    return NativeRegionBlockSemanticSection(
        chunk_x=_int32(raw.get("chunk_x"), "chunk_x"),
        chunk_z=_int32(raw.get("chunk_z"), "chunk_z"),
        section_y=_nonnegative_int(raw.get("section_y"), "section_y"),
        cell_code=np.frombuffer(encoded, dtype="<u2").astype(
            np.uint16,
            copy=True,
        ),
        palette=palette,
    )


def region_block_semantic_contract() -> dict[str, object]:
    return {
        "schema": REGION_BLOCK_SEMANTIC_SNAPSHOT_SCHEMA,
        "version": REGION_BLOCK_SEMANTIC_SNAPSHOT_VERSION,
        "server_version": "0.5.7",
        "section_transport": {
            "schema": REGION_BLOCK_SEMANTIC_SECTION_SCHEMA,
            "version": REGION_BLOCK_SEMANTIC_SECTION_VERSION,
            "code_encoding": REGION_BLOCK_SEMANTIC_CODE_ENCODING,
            "cell_count": SECTION_VOLUME,
            "palette_capacity": REGION_BLOCK_SEMANTIC_PALETTE_CAPACITY,
        },
        "capture_shape": [
            CAPTURE_CHUNKS_PER_AXIS,
            CAPTURE_CHUNKS_PER_AXIS,
            HEIGHT_SECTIONS,
            SECTION_VOLUME,
        ],
        "capture": {
            "coverage": "same_complete_5x5x10_plan_as_physical_Region",
            "live_physical_recheck": (
                "one_complete_exact_pass_before_semantic_capture"
            ),
            "partial_publication": "unsupported",
        },
        "identity": {
            "block_state": (
                "sha256_words_of_utf8_asset_id_nul_decimal_rotation"
            ),
            "asset_only": "sha256_words_of_utf8_asset_id",
            "rotation_index": "nonnegative_native_rotation_index",
            "filler_root_rule": (
                "same_asset_key_canonicalizes_to_root_"
                "different_asset_key_keeps_clicked_filler"
            ),
        },
        "affordance_dictionary_sha256": (
            block_affordance_dictionary_sha256()
        ),
        "alignment": "source_Region_semantic_sha256_required",
        "semantic_hash": (
            "expanded_cell_meaning_palette_order_independent_"
            "bridge_provenance_excluded"
        ),
        "artifact": {
            "format": "compressed_numeric_npz",
            "pickle": False,
            "declared_semantic_sha256_verified_on_load": True,
        },
        "runtime_ordinals": "never_published",
        "unknown_or_incomplete": "reject_complete_snapshot",
        "policy_binding": (
            "staged_for_single_announced_action_surface_contract_move"
        ),
    }


def region_block_semantic_contract_sha256() -> str:
    payload = json.dumps(
        region_block_semantic_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _wire_entry(value: object) -> RegionBlockSemanticEntry:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ) or len(value) != 7:
        raise ValueError("Region block-semantic entry must have seven fields")
    raw_key, raw_asset_key, valid, tags, gather, quality, rotation = value
    if not isinstance(valid, bool):
        raise ValueError("Region block-semantic valid flag must be boolean")
    return RegionBlockSemanticEntry(
        semantic_key=_wire_key(raw_key, valid, "semantic"),
        asset_key=_wire_key(raw_asset_key, valid, "asset"),
        valid=valid,
        affordance_tags=_nonnegative_int(tags, "affordance_tags"),
        gather_type_index=_nonnegative_int(gather, "gather_type_index"),
        required_tool_quality=_nonnegative_int(
            quality,
            "required_tool_quality",
        ),
        rotation_index=_nonnegative_int(rotation, "rotation_index"),
    )


def _wire_key(value: object, valid: bool, label: str) -> tuple[int, ...]:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError(f"Region block {label} key must be binary")
    key_bytes = bytes(value)
    if len(key_bytes) != (32 if valid else 0):
        raise ValueError(f"Region block {label} key has wrong length")
    if not valid:
        return (0,) * BLOCK_SEMANTIC_KEY_WORDS
    return struct.unpack(
        f">{BLOCK_SEMANTIC_KEY_WORDS}I",
        key_bytes,
    )


def _entry_sha256(entry: RegionBlockSemanticEntry) -> np.ndarray:
    digest = hashlib.sha256()
    digest.update(np.asarray(entry.semantic_key, dtype=">u4").tobytes())
    digest.update(np.asarray(entry.asset_key, dtype=">u4").tobytes())
    digest.update(
        struct.pack(
            "<?HBii",
            entry.valid,
            entry.affordance_tags,
            entry.gather_type_index,
            entry.required_tool_quality,
            entry.rotation_index,
        )
    )
    return np.frombuffer(digest.digest(), dtype=np.uint8)


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{label} SHA-256 must be 64 hexadecimal characters")
    return value.lower()


def _key_words(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ) or len(value) != BLOCK_SEMANTIC_KEY_WORDS:
        raise ValueError(
            f"{label} must contain {BLOCK_SEMANTIC_KEY_WORDS} uint32 words"
        )
    words: list[int] = []
    for raw in value:
        if isinstance(raw, bool):
            raise ValueError(f"{label} words must be integers")
        try:
            word = operator.index(raw)
        except TypeError as error:
            raise ValueError(f"{label} words must be integers") from error
        if not 0 <= word <= np.iinfo(np.uint32).max:
            raise ValueError(f"{label} words must fit uint32")
        words.append(word)
    return tuple(words)


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{label} must be an integer") from error
    if result < 0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _int32(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{label} must be an integer") from error
    if not np.iinfo(np.int32).min <= result <= np.iinfo(np.int32).max:
        raise ValueError(f"{label} must fit int32")
    return result


__all__ = [
    "BLOCK_SEMANTIC_KEY_WORDS",
    "NativeRegionBlockSemanticAssembler",
    "NativeRegionBlockSemanticSection",
    "NativeRegionBlockSemanticSnapshot",
    "NativeRegionBlockSemanticTransport",
    "ProjectedRegionBlockSemanticCapture",
    "REGION_BLOCK_SEMANTIC_CODE_BYTES",
    "REGION_BLOCK_SEMANTIC_CODE_ENCODING",
    "REGION_BLOCK_SEMANTIC_ALIGNMENT_CELL_CAPACITY",
    "REGION_BLOCK_SEMANTIC_PALETTE_CAPACITY",
    "REGION_BLOCK_SEMANTIC_SECTION_SCHEMA",
    "REGION_BLOCK_SEMANTIC_SECTION_VERSION",
    "REGION_BLOCK_SEMANTIC_SNAPSHOT_SCHEMA",
    "REGION_BLOCK_SEMANTIC_SNAPSHOT_VERSION",
    "RegionBlockSemanticEntry",
    "capture_native_region_block_semantics",
    "capture_projected_native_region_block_semantics",
    "native_region_block_semantic_section_from_wire",
    "region_block_semantic_contract",
    "region_block_semantic_contract_sha256",
    "region_block_semantic_projection_capture_contract",
    "region_block_semantic_projection_capture_contract_sha256",
]
