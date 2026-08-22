"""Assemble a complete Region v1 snapshot from bounded native bridge sections."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import operator
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from hytalegym.geometry.contract import (
    FLAG_FLUID,
    FLUID_MOVEMENT_FEATURES,
    MOVEMENT_FEATURES,
)
from hytalegym.worldgen.region.capture import (
    RegionCapturePlan,
    RegionCaptureTask,
)
from hytalegym.worldgen.region.contract import (
    CAPTURE_CHUNK_COUNT,
    CAPTURE_CHUNKS_PER_AXIS,
    CAPTURE_SECTION_COUNT,
    CERTIFIED_SOURCE_REACH_BOUND_BLOCKS,
    HEIGHT_SECTIONS,
    MAX_CELL_PALETTE,
    MAX_SHAPE_BOXES,
    MAX_SHAPE_PALETTE,
    SECTION_VOLUME,
    ChunkApiContract,
    capture_chunk_slot,
)
from hytalegym.worldgen.region.snapshot import (
    FILLER_ROOT_OFFSET_ENCODING,
    OFFLINE_COMPLETE,
    REGION_SNAPSHOT_SCHEMA,
    REGION_SNAPSHOT_VERSION,
    CellPalette,
    NativeRegionSnapshot,
    ShapePalette,
)
from hytalegym.worldgen.region.stability import (
    require_region_semantics_equal,
)

REGION_SECTION_SCHEMA = "hytalerl_native_region_section_v3"
REGION_SECTION_VERSION = 3
REGION_CODE_ENCODING = "uint16_le_y_z_x"
REGION_CODE_BYTES = SECTION_VOLUME * np.dtype("<u2").itemsize
REGION_FILLER_ROOT_ENCODING = FILLER_ROOT_OFFSET_ENCODING
REGION_FILLER_ROOT_BYTES = SECTION_VOLUME * np.dtype("<u2").itemsize
REGION_STABILITY_PASSES = 2


class NativeRegionTransport(Protocol):
    """The two bounded capture verbs exposed by a reset native Gym."""

    def capture_region_manifest(
        self,
        core_min_chunk_x: int | None = None,
        core_min_chunk_z: int | None = None,
    ) -> dict[str, Any]: ...

    def capture_region_section(
        self,
        chunk_x: int,
        chunk_z: int,
        section_y: int,
    ) -> dict[str, Any]: ...


ProgressCallback = Callable[[int, int, RegionCaptureTask], None]


class NativeRegionAssembler:
    """Validate section responses and merge their local semantic palettes."""

    def __init__(
        self,
        manifest: Mapping[str, Any],
        *,
        source_reach_bound_blocks: float = (CERTIFIED_SOURCE_REACH_BOUND_BLOCKS),
    ):
        source = dict(manifest)
        if source.get("type") != "region_manifest":
            raise ValueError("native response is not a region manifest")
        if source.get("schema") != REGION_SNAPSHOT_SCHEMA:
            raise ValueError("unsupported native region manifest schema")
        if _wire_int(source.get("version"), "version") != REGION_SNAPSHOT_VERSION:
            raise ValueError("unsupported native region manifest version")
        contract = ChunkApiContract.from_manifest(source.get("chunk_api", {}))
        if source.get("server_version") != contract.server_version:
            raise ValueError("region manifest and chunk API versions differ")
        if source.get("capture_mode") != OFFLINE_COMPLETE:
            raise ValueError("native region producer must be offline-complete")
        if source.get("exact_collision_shapes") is not True:
            raise ValueError("native region lacks exact collision shapes")
        if source.get("dynamic_state") != "static":
            raise ValueError("dynamic native region capture is unsupported")
        if (
            _wire_int(
                source.get("capture_chunks_per_axis"),
                "capture_chunks_per_axis",
            )
            != CAPTURE_CHUNKS_PER_AXIS
        ):
            raise ValueError("native region capture width is unsupported")
        if (
            _wire_int(
                source.get("capture_section_count"),
                "capture_section_count",
            )
            != CAPTURE_SECTION_COUNT
        ):
            raise ValueError("native region section count is unsupported")

        self._core = np.asarray(
            [
                _wire_int32(source.get("core_min_chunk_x"), "core_min_chunk_x"),
                _wire_int32(source.get("core_min_chunk_z"), "core_min_chunk_z"),
            ],
            dtype=np.int32,
        )
        expected_capture = self._core - 1
        actual_capture = np.asarray(
            [
                _wire_int32(
                    source.get("capture_min_chunk_x"),
                    "capture_min_chunk_x",
                ),
                _wire_int32(
                    source.get("capture_min_chunk_z"),
                    "capture_min_chunk_z",
                ),
            ],
            dtype=np.int32,
        )
        if not np.array_equal(actual_capture, expected_capture):
            raise ValueError("native region capture halo is inconsistent")
        RegionCapturePlan.complete(
            int(self._core[0]),
            int(self._core[1]),
            maximum_source_reach_blocks=source_reach_bound_blocks,
        )
        self._source_reach_bound = float(source_reach_bound_blocks)
        self._observed_source_reach = 0.0

        self._metadata = {
            "schema": REGION_SNAPSHOT_SCHEMA,
            "version": REGION_SNAPSHOT_VERSION,
            "server_version": contract.server_version,
            "chunk_api": contract.manifest(),
            "capture_mode": OFFLINE_COMPLETE,
            "exact_collision_shapes": True,
            "dynamic_state": "static",
            "seed": _wire_int(source.get("seed"), "seed"),
            "world": str(source.get("world", "")),
            "worldgen_provider": str(source.get("worldgen_provider", "")),
            "worldgen_version": str(source.get("worldgen_version", "")),
            "section_protocol_schema": REGION_SECTION_SCHEMA,
            "section_protocol_version": REGION_SECTION_VERSION,
            "exact_filler_root_offsets": True,
            "filler_root_offset_encoding": REGION_FILLER_ROOT_ENCODING,
            "source_reach_bound_blocks": self._source_reach_bound,
            "same_world_stability": "unverified",
        }
        self._known = np.zeros(
            (CAPTURE_CHUNK_COUNT, HEIGHT_SECTIONS),
            dtype=np.bool_,
        )
        self._code = np.zeros(
            (CAPTURE_CHUNK_COUNT, HEIGHT_SECTIONS, SECTION_VOLUME),
            dtype=np.uint16,
        )
        self._filler_root = np.zeros_like(self._code)

        empty_shape = np.zeros((0, 6), dtype=np.float32)
        self._shapes: list[np.ndarray] = [empty_shape]
        self._shape_index: dict[tuple[int, bytes], int] = {_shape_key(empty_shape): 0}
        air_movement = np.zeros(MOVEMENT_FEATURES, dtype=np.float32)
        air_fluid = np.zeros(FLUID_MOVEMENT_FEATURES, dtype=np.float32)
        air = (0, 0, 0, 0.0, 0, 0, 0, air_movement, air_fluid)
        self._cells: list[
            tuple[
                int,
                int,
                int,
                float,
                int,
                int,
                int,
                np.ndarray,
                np.ndarray,
            ]
        ] = [air]
        self._cell_index: dict[tuple[Any, ...], int] = {_cell_key(air): 0}

    @property
    def core_min_chunk_xz(self) -> np.ndarray:
        return self._core.copy()

    @property
    def section_count(self) -> int:
        return int(np.count_nonzero(self._known))

    def add_section(self, response: Mapping[str, Any]) -> None:
        raw = dict(response)
        if raw.get("type") != "region_section":
            raise ValueError("native response is not a region section")
        if raw.get("schema") != REGION_SECTION_SCHEMA:
            raise ValueError("unsupported native region section schema")
        if _wire_int(raw.get("version"), "version") != REGION_SECTION_VERSION:
            raise ValueError("unsupported native region section version")
        if raw.get("code_encoding") != REGION_CODE_ENCODING:
            raise ValueError("unsupported native region code encoding")
        if raw.get("filler_root_encoding") != REGION_FILLER_ROOT_ENCODING:
            raise ValueError("unsupported native filler-root encoding")

        chunk_x = _wire_int32(raw.get("chunk_x"), "chunk_x")
        chunk_z = _wire_int32(raw.get("chunk_z"), "chunk_z")
        section_y = _wire_int(raw.get("section_y"), "section_y")
        if not 0 <= section_y < HEIGHT_SECTIONS:
            raise ValueError("native region section Y is outside the world")
        slot = capture_chunk_slot(
            chunk_x,
            chunk_z,
            int(self._core[0]) - 1,
            int(self._core[1]) - 1,
        )
        if self._known[slot, section_y]:
            raise ValueError("duplicate native region section")

        raw_shapes = raw.get("shape_palette")
        if not isinstance(raw_shapes, Sequence) or isinstance(
            raw_shapes,
            (str, bytes, bytearray),
        ):
            raise ValueError("region shape palette must be an array")
        if not 1 <= len(raw_shapes) <= MAX_SHAPE_PALETTE:
            raise ValueError("region shape palette exceeds capacity")
        local_shape_to_global = np.empty(len(raw_shapes), dtype=np.int32)
        for local_index, payload in enumerate(raw_shapes):
            boxes = np.asarray(payload, dtype=np.float32).reshape(-1)
            if boxes.size % 6:
                raise ValueError("region shape payload is not six-value boxes")
            boxes = boxes.reshape((-1, 6))
            if boxes.shape[0] > MAX_SHAPE_BOXES:
                raise ValueError("region shape exceeds nine boxes")
            if not np.all(np.isfinite(boxes)):
                raise ValueError("region shape contains non-finite values")
            if np.any(boxes[:, :3] > boxes[:, 3:]):
                raise ValueError("region shape contains an inverted box")
            if local_index == 0 and boxes.size:
                raise ValueError("region shape zero must be empty")
            if boxes.size:
                horizontal_reach = np.max(
                    np.stack(
                        (
                            -boxes[:, 0],
                            boxes[:, 3] - 1.0,
                            -boxes[:, 2],
                            boxes[:, 5] - 1.0,
                        )
                    )
                )
                self._observed_source_reach = max(
                    self._observed_source_reach,
                    float(horizontal_reach),
                )
            key = _shape_key(boxes)
            global_index = self._shape_index.get(key)
            if global_index is None:
                if len(self._shapes) >= MAX_SHAPE_PALETTE:
                    raise ValueError("merged region shape palette overflow")
                global_index = len(self._shapes)
                self._shape_index[key] = global_index
                self._shapes.append(boxes.copy())
            local_shape_to_global[local_index] = global_index

        raw_cells = raw.get("cell_palette")
        if not isinstance(raw_cells, Sequence) or isinstance(
            raw_cells,
            (str, bytes, bytearray),
        ):
            raise ValueError("region cell palette must be an array")
        if not 1 <= len(raw_cells) <= MAX_CELL_PALETTE:
            raise ValueError("region cell palette exceeds capacity")
        local_cell_to_global = np.empty(len(raw_cells), dtype=np.uint32)
        for local_index, payload in enumerate(raw_cells):
            cell = self._parse_cell(payload, local_shape_to_global)
            if local_index == 0 and _cell_key(cell) != _cell_key(self._cells[0]):
                raise ValueError("region cell zero must be canonical air")
            key = _cell_key(cell)
            global_index = self._cell_index.get(key)
            if global_index is None:
                if len(self._cells) >= MAX_CELL_PALETTE:
                    raise ValueError("merged region cell palette overflow")
                global_index = len(self._cells)
                self._cell_index[key] = global_index
                self._cells.append(cell)
            local_cell_to_global[local_index] = global_index

        payload = raw.get("cell_codes")
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise ValueError("region cell codes must be MessagePack binary")
        encoded = memoryview(payload)
        if encoded.nbytes != REGION_CODE_BYTES:
            raise ValueError("region cell code payload has the wrong size")
        local_codes = np.frombuffer(encoded, dtype="<u2", count=SECTION_VOLUME)
        if local_codes.size != SECTION_VOLUME:
            raise ValueError("region cell code payload is incomplete")
        if np.any(local_codes.astype(np.uint32) >= len(raw_cells)):
            raise ValueError("region cell code exceeds its local palette")
        merged = local_cell_to_global[local_codes]
        if np.any(merged >= MAX_CELL_PALETTE):
            raise ValueError("merged region cell code exceeds uint16")
        filler_payload = raw.get("filler_root_offsets")
        if not isinstance(
            filler_payload,
            (bytes, bytearray, memoryview),
        ):
            raise ValueError(
                "region filler-root offsets must be MessagePack binary"
            )
        encoded_filler = memoryview(filler_payload)
        if encoded_filler.nbytes != REGION_FILLER_ROOT_BYTES:
            raise ValueError(
                "region filler-root offset payload has the wrong size"
            )
        filler_root = np.frombuffer(
            encoded_filler,
            dtype="<u2",
            count=SECTION_VOLUME,
        )
        if np.any(filler_root & np.uint16(0x8000)):
            raise ValueError(
                "region filler-root offset exceeds packed 15-bit capacity"
            )
        if np.any((local_codes == 0) & (filler_root != 0)):
            raise ValueError(
                "canonical air cannot carry a filler-root offset"
            )
        self._code[slot, section_y] = merged.astype(np.uint16)
        self._filler_root[slot, section_y] = filler_root
        self._known[slot, section_y] = True

    def finish(self) -> NativeRegionSnapshot:
        if not np.all(self._known):
            missing = self._known.size - int(np.count_nonzero(self._known))
            raise ValueError(f"native region capture is missing {missing} sections")
        if self._observed_source_reach > self._source_reach_bound:
            raise ValueError(
                "captured collision shape exceeds the certified source reach"
            )
        self._metadata["observed_source_reach_blocks"] = max(
            0.0,
            self._observed_source_reach,
        )

        flags = np.asarray([cell[0] for cell in self._cells], dtype=np.uint16)
        shape_index = np.asarray(
            [cell[1] for cell in self._cells],
            dtype=np.uint16,
        )
        fluid_level = np.asarray(
            [cell[2] for cell in self._cells],
            dtype=np.uint8,
        )
        fluid_fill_height = np.asarray(
            [cell[3] for cell in self._cells],
            dtype=np.float32,
        )
        support = np.asarray([cell[4] for cell in self._cells], dtype=np.int32)
        block_damage = np.asarray(
            [cell[5] for cell in self._cells],
            dtype=np.int32,
        )
        fluid_damage = np.asarray(
            [cell[6] for cell in self._cells],
            dtype=np.int32,
        )
        movement = np.stack([cell[7] for cell in self._cells]).astype(
            np.float32,
            copy=False,
        )
        fluid_movement = np.stack([cell[8] for cell in self._cells]).astype(
            np.float32, copy=False
        )

        boxes = np.zeros(
            (len(self._shapes), MAX_SHAPE_BOXES, 6),
            dtype=np.float32,
        )
        box_mask = np.zeros(
            (len(self._shapes), MAX_SHAPE_BOXES),
            dtype=np.bool_,
        )
        for index, shape in enumerate(self._shapes):
            count = shape.shape[0]
            boxes[index, :count] = shape
            box_mask[index, :count] = True

        return NativeRegionSnapshot(
            metadata=self._metadata,
            core_min_chunk_xz=self._core,
            section_known=self._known,
            cell_code=self._code,
            cell_palette=CellPalette(
                flags=flags,
                shape_index=shape_index,
                fluid_level=fluid_level,
                fluid_fill_height=fluid_fill_height,
                support=support,
                block_damage=block_damage,
                fluid_damage=fluid_damage,
                movement=movement,
                fluid_movement=fluid_movement,
            ),
            shape_palette=ShapePalette(
                boxes=boxes,
                box_mask=box_mask,
            ),
            filler_root_offset_packed=self._filler_root,
        )

    def _parse_cell(
        self,
        payload: Any,
        local_shape_to_global: np.ndarray,
    ) -> tuple[
        int,
        int,
        int,
        float,
        int,
        int,
        int,
        np.ndarray,
        np.ndarray,
    ]:
        if not isinstance(payload, Sequence) or isinstance(
            payload,
            (str, bytes, bytearray),
        ):
            raise ValueError("region cell palette entry must be an array")
        if len(payload) != 9:
            raise ValueError("region cell palette entry must have nine fields")
        flags = _wire_int(payload[0], "flags")
        local_shape = _wire_int(payload[1], "shape_index")
        fluid_level = _wire_int(payload[2], "fluid_level")
        fluid_fill_height = float(payload[3])
        support = _wire_int32(payload[4], "support")
        block_damage = _wire_int32(payload[5], "block_damage")
        fluid_damage = _wire_int32(payload[6], "fluid_damage")
        if not 0 <= flags <= np.iinfo(np.uint16).max:
            raise ValueError("region cell flags must fit uint16")
        if not 0 <= local_shape < local_shape_to_global.size:
            raise ValueError("region cell references an absent local shape")
        if not 0 <= fluid_level <= np.iinfo(np.uint8).max:
            raise ValueError("region fluid level must fit uint8")
        if (
            not np.isfinite(fluid_fill_height)
            or not 0.0 <= fluid_fill_height <= 1.0
        ):
            raise ValueError("region fluid fill height must be in [0, 1]")
        movement = np.asarray(payload[7], dtype=np.float32)
        fluid_movement = np.asarray(payload[8], dtype=np.float32)
        if movement.shape != (MOVEMENT_FEATURES,):
            raise ValueError("region block movement must contain nine values")
        if fluid_movement.shape != (FLUID_MOVEMENT_FEATURES,):
            raise ValueError("region FluidFX movement must contain six values")
        if not np.all(np.isfinite(movement)) or not np.all(np.isfinite(fluid_movement)):
            raise ValueError("region movement contains non-finite values")
        if flags & FLAG_FLUID and fluid_movement[3] <= 0.0:
            raise ValueError("region fluid requires a horizontal speed multiplier")
        return (
            flags,
            int(local_shape_to_global[local_shape]),
            fluid_level,
            fluid_fill_height,
            support,
            block_damage,
            fluid_damage,
            movement.copy(),
            fluid_movement.copy(),
        )


def capture_native_region(
    transport: NativeRegionTransport,
    *,
    core_min_chunk_x: int | None = None,
    core_min_chunk_z: int | None = None,
    progress: ProgressCallback | None = None,
    native_evidence_jar_sha256: str | None = None,
) -> NativeRegionSnapshot:
    """Capture one selected core twice and reject same-world semantic change."""

    requested_core = _requested_core(
        core_min_chunk_x,
        core_min_chunk_z,
    )

    first = capture_native_region_pass(
        transport,
        requested_core=requested_core,
        order="forward",
        progress=progress,
        progress_offset=0,
        progress_total=CAPTURE_SECTION_COUNT * REGION_STABILITY_PASSES,
    )
    second = capture_native_region_pass(
        transport,
        requested_core=requested_core,
        order="reverse",
        progress=progress,
        progress_offset=CAPTURE_SECTION_COUNT,
        progress_total=CAPTURE_SECTION_COUNT * REGION_STABILITY_PASSES,
    )
    comparison = require_region_semantics_equal(
        first,
        second,
        require_same_world=True,
        context="same-world Region confirmation",
        native_evidence_jar_sha256=native_evidence_jar_sha256,
    )
    metadata = dict(second.metadata)
    metadata.update(
        {
            "same_world_stability": "verified",
            "same_world_confirmation_passes": REGION_STABILITY_PASSES,
            "same_world_capture_order": "forward_then_reverse",
            "region_semantic_sha256": comparison.second_digest,
        }
    )
    if comparison.native_evidence_jar_sha256 is not None:
        metadata["native_evidence_jar_sha256"] = (
            comparison.native_evidence_jar_sha256
        )
    return NativeRegionSnapshot(
        metadata=metadata,
        core_min_chunk_xz=second.core_min_chunk_xz,
        section_known=second.section_known,
        cell_code=second.cell_code,
        cell_palette=second.cell_palette,
        shape_palette=second.shape_palette,
        filler_root_offset_packed=second.filler_root_offset_packed,
    )


def capture_native_region_pass(
    transport: NativeRegionTransport,
    *,
    requested_core: tuple[int, int] | None = None,
    order: str = "forward",
    progress: ProgressCallback | None = None,
    progress_offset: int = 0,
    progress_total: int = CAPTURE_SECTION_COUNT,
) -> NativeRegionSnapshot:
    """Capture one diagnostic pass in an explicit section request order."""

    if order not in {"forward", "reverse"}:
        raise ValueError("Region capture order must be 'forward' or 'reverse'")
    manifest = (
        transport.capture_region_manifest()
        if requested_core is None
        else transport.capture_region_manifest(*requested_core)
    )
    assembler = NativeRegionAssembler(manifest)
    core = assembler.core_min_chunk_xz
    if (
        requested_core is not None
        and (
            int(core[0]),
            int(core[1]),
        )
        != requested_core
    ):
        raise ValueError("native region manifest returned a different requested core")
    plan = RegionCapturePlan.complete(
        int(core[0]),
        int(core[1]),
        maximum_source_reach_blocks=CERTIFIED_SOURCE_REACH_BOUND_BLOCKS,
    )
    tasks = reversed(plan.tasks) if order == "reverse" else iter(plan.tasks)
    for index, task in enumerate(tasks, start=progress_offset + 1):
        assembler.add_section(
            transport.capture_region_section(
                task.chunk_x,
                task.chunk_z,
                task.section_y,
            )
        )
        if progress is not None:
            progress(index, progress_total, task)
    return assembler.finish()


def capture_native_region_to_file(
    transport: NativeRegionTransport,
    path: str | Path,
    *,
    core_min_chunk_x: int | None = None,
    core_min_chunk_z: int | None = None,
    progress: ProgressCallback | None = None,
    native_evidence_jar_sha256: str | None = None,
) -> Path:
    """Capture stable geometry and write the numeric NPZ snapshot."""

    return capture_native_region(
        transport,
        core_min_chunk_x=core_min_chunk_x,
        core_min_chunk_z=core_min_chunk_z,
        progress=progress,
        native_evidence_jar_sha256=native_evidence_jar_sha256,
    ).save(path)


def _requested_core(
    core_min_chunk_x: int | None,
    core_min_chunk_z: int | None,
) -> tuple[int, int] | None:
    if (core_min_chunk_x is None) != (core_min_chunk_z is None):
        raise ValueError(
            "core_min_chunk_x and core_min_chunk_z must be supplied together"
        )
    if core_min_chunk_x is None:
        return None
    plan = RegionCapturePlan.complete(
        core_min_chunk_x,
        core_min_chunk_z,
        maximum_source_reach_blocks=CERTIFIED_SOURCE_REACH_BOUND_BLOCKS,
    )
    return plan.core_min_chunk_x, plan.core_min_chunk_z


def _shape_key(boxes: np.ndarray) -> tuple[int, bytes]:
    contiguous = np.ascontiguousarray(boxes, dtype=np.float32)
    return int(contiguous.shape[0]), contiguous.tobytes()


def _cell_key(
    cell: tuple[
        int,
        int,
        int,
        float,
        int,
        int,
        int,
        np.ndarray,
        np.ndarray,
    ],
) -> tuple[Any, ...]:
    return (
        *cell[:7],
        np.ascontiguousarray(cell[7], dtype=np.float32).tobytes(),
        np.ascontiguousarray(cell[8], dtype=np.float32).tobytes(),
    )


def _wire_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


def _wire_int32(value: Any, name: str) -> int:
    result = _wire_int(value, name)
    if not np.iinfo(np.int32).min <= result <= np.iinfo(np.int32).max:
        raise ValueError(f"{name} exceeds int32")
    return result
