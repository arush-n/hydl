"""Portable exact geometry for a complete or explicitly partial region."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from hytalegym.geometry.contract import (
    FLAG_FLUID,
    FLUID_MOVEMENT_FEATURES,
    MOVEMENT_FEATURES,
)
from hytalegym.worldgen.region.contract import (
    CAPTURE_CHUNK_COUNT,
    HEIGHT_SECTIONS,
    MAX_CELL_PALETTE,
    MAX_CHUNK_COORDINATE,
    MAX_SHAPE_BOXES,
    MAX_SHAPE_PALETTE,
    MIN_CHUNK_COORDINATE,
    SECTION_VOLUME,
    ChunkApiContract,
)

REGION_SNAPSHOT_SCHEMA = "hytalerl_native_region_snapshot_v1"
REGION_SNAPSHOT_VERSION = 1
OFFLINE_COMPLETE = "offline_complete"
ASYNC_PARTIAL = "async_partial"
_CAPTURE_MODES = {OFFLINE_COMPLETE, ASYNC_PARTIAL}
_META_KEY = "__metadata_json__"
_CORE_KEY = "__core_min_chunk_xz__"
_KNOWN_KEY = "__section_known__"
_CODE_KEY = "__cell_code__"
_FILLER_ROOT_KEY = "__filler_root_offset_packed__"
_FLUID_FILL_KEY = "cell_fluid_fill_height"
_SNAPSHOT_KEYS = {
    _META_KEY,
    _CORE_KEY,
    _KNOWN_KEY,
    _CODE_KEY,
    "cell_flags",
    "cell_shape_index",
    "cell_fluid_level",
    "cell_support",
    "cell_block_damage",
    "cell_fluid_damage",
    "cell_movement",
    "cell_fluid_movement",
    "shape_boxes",
    "shape_box_mask",
}
_OPTIONAL_SNAPSHOT_KEYS = {_FLUID_FILL_KEY, _FILLER_ROOT_KEY}
FILLER_ROOT_OFFSET_ENCODING = "uint16_le_signed_5bit_xyz_y_z_x"


@dataclass(frozen=True)
class CellPalette:
    """Deduplicated immutable cell semantics; code zero is known air."""

    flags: np.ndarray
    shape_index: np.ndarray
    fluid_level: np.ndarray
    support: np.ndarray
    block_damage: np.ndarray
    fluid_damage: np.ndarray
    movement: np.ndarray
    fluid_movement: np.ndarray
    # Region section v2 publishes exact [0, 1] fill height. Legacy v1
    # artifacts load with -1 only on fluid entries and therefore fail closed.
    fluid_fill_height: np.ndarray | None = None

    def __post_init__(self) -> None:
        flags = _vector(self.flags, np.uint16, "flags")
        count = flags.shape[0]
        if not 1 <= count <= MAX_CELL_PALETTE:
            raise ValueError("cell palette exceeds its fixed uint16 capacity")
        values = {
            "shape_index": _vector(
                self.shape_index,
                np.uint16,
                "shape_index",
                count,
            ),
            "fluid_level": _vector(
                self.fluid_level,
                np.uint8,
                "fluid_level",
                count,
            ),
            "support": _vector(self.support, np.int32, "support", count),
            "block_damage": _vector(
                self.block_damage,
                np.int32,
                "block_damage",
                count,
            ),
            "fluid_damage": _vector(
                self.fluid_damage,
                np.int32,
                "fluid_damage",
                count,
            ),
        }
        movement = _matrix(
            self.movement,
            np.float32,
            (count, MOVEMENT_FEATURES),
            "movement",
        )
        fluid_movement = _matrix(
            self.fluid_movement,
            np.float32,
            (count, FLUID_MOVEMENT_FEATURES),
            "fluid_movement",
        )
        if not np.all(np.isfinite(movement)) or not np.all(np.isfinite(fluid_movement)):
            raise ValueError("cell palette contains non-finite movement values")
        if self.fluid_fill_height is None:
            fluid_fill_height = np.zeros(count, dtype=np.float32)
            fluid_fill_height[(flags & np.uint16(FLAG_FLUID)) != 0] = -1.0
        else:
            fluid_fill_height = np.asarray(
                self.fluid_fill_height,
                dtype=np.float32,
            )
            if fluid_fill_height.shape != (count,):
                raise ValueError(
                    f"fluid_fill_height must have shape ({count},)"
                )
            if (
                not np.all(np.isfinite(fluid_fill_height))
                or np.any(fluid_fill_height < -1.0)
                or np.any(fluid_fill_height > 1.0)
            ):
                raise ValueError(
                    "fluid fill height must be finite and in [-1, 1]"
                )
            if np.any(
                (fluid_fill_height < 0.0)
                & ((flags & np.uint16(FLAG_FLUID)) == 0)
            ):
                raise ValueError(
                    "unavailable fluid fill is valid only for fluid cells"
                )

        object.__setattr__(self, "flags", flags)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "movement", movement)
        object.__setattr__(self, "fluid_movement", fluid_movement)
        object.__setattr__(
            self,
            "fluid_fill_height",
            fluid_fill_height,
        )
        if (
            flags[0] != 0
            or any(value[0] != 0 for value in values.values())
            or fluid_fill_height[0] != 0.0
            or np.any(movement[0])
            or np.any(fluid_movement[0])
        ):
            raise ValueError("cell palette entry zero must be canonical air")

    @property
    def size(self) -> int:
        return int(self.flags.shape[0])


@dataclass(frozen=True)
class ShapePalette:
    """Exact cell-local collision boxes; shape zero is empty."""

    boxes: np.ndarray
    box_mask: np.ndarray

    def __post_init__(self) -> None:
        boxes = np.asarray(self.boxes, dtype=np.float32)
        mask = _bool_array(
            self.box_mask,
            "box_mask",
            expected_shape=boxes.shape[:2],
        )
        if boxes.ndim != 3 or boxes.shape[1:] != (MAX_SHAPE_BOXES, 6):
            raise ValueError("shape palette has an invalid fixed box shape")
        if not 1 <= boxes.shape[0] <= MAX_SHAPE_PALETTE:
            raise ValueError("shape palette exceeds its fixed uint14 capacity")
        if not np.all(np.isfinite(boxes)):
            raise ValueError("shape palette contains non-finite boxes")
        if np.any(boxes[..., :3][mask] > boxes[..., 3:][mask]):
            raise ValueError("shape palette contains an inverted box")
        if np.any(mask[0]) or np.any(boxes[0]):
            raise ValueError("shape palette entry zero must be empty")
        canonical = boxes.copy()
        canonical[~mask] = 0.0
        object.__setattr__(self, "boxes", canonical)
        object.__setattr__(self, "box_mask", mask.copy())

    @property
    def size(self) -> int:
        return int(self.boxes.shape[0])


@dataclass(frozen=True)
class NativeRegionSnapshot:
    """One palette-compressed 5x5x10 capture around a 3x3 chunk core."""

    metadata: Mapping[str, Any]
    core_min_chunk_xz: np.ndarray
    section_known: np.ndarray
    cell_code: np.ndarray
    cell_palette: CellPalette
    shape_palette: ShapePalette
    filler_root_offset_packed: np.ndarray | None = None

    def __post_init__(self) -> None:
        metadata = dict(self.metadata)
        if metadata.get("schema") != REGION_SNAPSHOT_SCHEMA:
            raise ValueError("unsupported native region snapshot schema")
        if (
            not _is_int(metadata.get("version"))
            or metadata["version"] != REGION_SNAPSHOT_VERSION
        ):
            raise ValueError("unsupported native region snapshot version")
        contract = ChunkApiContract.from_manifest(metadata.get("chunk_api", {}))
        if metadata.get("server_version") != contract.server_version:
            raise ValueError("region and chunk API server versions differ")
        mode = metadata.get("capture_mode")
        if mode not in _CAPTURE_MODES:
            raise ValueError("unsupported region capture mode")
        if metadata.get("exact_collision_shapes") is not True:
            raise ValueError("region lacks exact native collision shapes")
        if metadata.get("dynamic_state") != "static":
            raise ValueError("dynamic region mechanics are unsupported")
        stability = metadata.get("same_world_stability")
        if stability not in (None, "unverified", "verified"):
            raise ValueError("unsupported same-world stability status")
        if stability == "verified":
            passes = metadata.get("same_world_confirmation_passes")
            semantic_digest = metadata.get("region_semantic_sha256")
            if not _is_int(passes) or passes < 2:
                raise ValueError("verified region requires two capture passes")
            if (
                not isinstance(semantic_digest, str)
                or len(semantic_digest) != 64
                or any(
                    character not in "0123456789abcdef" for character in semantic_digest
                )
            ):
                raise ValueError("verified region requires a semantic SHA-256")

        core = _integer_array(
            self.core_min_chunk_xz,
            np.int32,
            "core_min_chunk_xz",
            expected_shape=(2,),
        )
        for value in core.astype(np.int64):
            if value - 1 < MIN_CHUNK_COORDINATE or value + 3 > MAX_CHUNK_COORDINATE:
                raise ValueError("region capture exceeds native chunk bounds")
        expected_known = (CAPTURE_CHUNK_COUNT, HEIGHT_SECTIONS)
        known = _bool_array(
            self.section_known,
            "section_known",
            expected_shape=expected_known,
        )
        if mode == OFFLINE_COMPLETE and not np.all(known):
            raise ValueError("offline region capture is incomplete")
        expected_code = expected_known + (SECTION_VOLUME,)
        code = _integer_array(
            self.cell_code,
            np.uint16,
            "cell_code",
            expected_shape=expected_code,
        )
        if np.any(code[~known]):
            raise ValueError("unknown sections must use zero-filled cell codes")
        if np.any(code[known].astype(np.uint32) >= self.cell_palette.size):
            raise ValueError("cell code exceeds the supplied palette")
        section_protocol = int(metadata.get("section_protocol_version", 1))
        if self.filler_root_offset_packed is None:
            filler_root = np.zeros(expected_code, dtype=np.uint16)
            filler_root_available = False
        else:
            filler_root = _integer_array(
                self.filler_root_offset_packed,
                np.uint16,
                "filler_root_offset_packed",
                expected_shape=expected_code,
            )
            filler_root_available = True
        if np.any(filler_root & np.uint16(0x8000)):
            raise ValueError("filler-root offset exceeds packed 15-bit capacity")
        if np.any(filler_root[~known]):
            raise ValueError(
                "unknown sections must use zero-filled filler-root offsets"
            )
        if np.any((code == 0) & (filler_root != 0)):
            raise ValueError("canonical air cannot carry a filler-root offset")
        if section_protocol >= 3:
            if not filler_root_available:
                raise ValueError(
                    "Region section v3 requires exact filler-root offsets"
                )
            if metadata.get("exact_filler_root_offsets") is not True:
                raise ValueError(
                    "Region section v3 must declare exact filler-root offsets"
                )
            if (
                metadata.get("filler_root_offset_encoding")
                != FILLER_ROOT_OFFSET_ENCODING
            ):
                raise ValueError(
                    "Region filler-root offset encoding is unsupported"
                )
        elif filler_root_available and np.any(filler_root):
            raise ValueError(
                "legacy Region sections cannot carry unlabeled filler-root offsets"
            )
        if np.any(
            self.cell_palette.shape_index.astype(np.uint32) >= self.shape_palette.size
        ):
            raise ValueError("cell palette references an absent shape")
        if (
            int(metadata.get("section_protocol_version", 1)) >= 2
            and np.any(self.cell_palette.fluid_fill_height < 0.0)
        ):
            raise ValueError(
                "Region section v2 requires exact fluid fill height"
            )

        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "core_min_chunk_xz", core.copy())
        object.__setattr__(self, "section_known", known.copy())
        object.__setattr__(self, "cell_code", code.copy())
        object.__setattr__(
            self,
            "filler_root_offset_packed",
            filler_root,
        )

    @property
    def capture_min_chunk_xz(self) -> np.ndarray:
        return self.core_min_chunk_xz - np.int32(1)

    @property
    def filler_root_offsets_available(self) -> bool:
        """Whether every known cell carries native filler placement."""

        return int(self.metadata.get("section_protocol_version", 1)) >= 3

    def save(self, path: str | Path) -> Path:
        """Atomically write a compressed numeric NPZ without pickle payloads."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            _META_KEY: np.asarray(
                json.dumps(
                    self.metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            _CORE_KEY: self.core_min_chunk_xz,
            _KNOWN_KEY: self.section_known,
            _CODE_KEY: self.cell_code,
            "cell_flags": self.cell_palette.flags,
            "cell_shape_index": self.cell_palette.shape_index,
            "cell_fluid_level": self.cell_palette.fluid_level,
            _FLUID_FILL_KEY: self.cell_palette.fluid_fill_height,
            "cell_support": self.cell_palette.support,
            "cell_block_damage": self.cell_palette.block_damage,
            "cell_fluid_damage": self.cell_palette.fluid_damage,
            "cell_movement": self.cell_palette.movement,
            "cell_fluid_movement": self.cell_palette.fluid_movement,
            "shape_boxes": self.shape_palette.boxes,
            "shape_box_mask": self.shape_palette.box_mask,
        }
        if self.filler_root_offsets_available:
            payload[_FILLER_ROOT_KEY] = self.filler_root_offset_packed
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
    def load(cls, path: str | Path) -> "NativeRegionSnapshot":
        with np.load(Path(path), allow_pickle=False) as archive:
            actual = set(archive.files)
            allowed = _SNAPSHOT_KEYS | _OPTIONAL_SNAPSHOT_KEYS
            if not _SNAPSHOT_KEYS.issubset(actual) or not actual.issubset(
                allowed
            ):
                raise ValueError(
                    "region snapshot fields differ: "
                    f"missing={sorted(_SNAPSHOT_KEYS - actual)}, "
                    f"extra={sorted(actual - allowed)}"
                )
            snapshot = cls(
                metadata=json.loads(str(archive[_META_KEY].item())),
                core_min_chunk_xz=archive[_CORE_KEY],
                section_known=archive[_KNOWN_KEY],
                cell_code=archive[_CODE_KEY],
                cell_palette=CellPalette(
                    flags=archive["cell_flags"],
                    shape_index=archive["cell_shape_index"],
                    fluid_level=archive["cell_fluid_level"],
                    fluid_fill_height=(
                        archive[_FLUID_FILL_KEY]
                        if _FLUID_FILL_KEY in actual
                        else None
                    ),
                    support=archive["cell_support"],
                    block_damage=archive["cell_block_damage"],
                    fluid_damage=archive["cell_fluid_damage"],
                    movement=archive["cell_movement"],
                    fluid_movement=archive["cell_fluid_movement"],
                ),
                shape_palette=ShapePalette(
                    boxes=archive["shape_boxes"],
                    box_mask=archive["shape_box_mask"],
                ),
                filler_root_offset_packed=(
                    archive[_FILLER_ROOT_KEY]
                    if _FILLER_ROOT_KEY in actual
                    else None
                ),
            )
        declared_digest = snapshot.metadata.get("region_semantic_sha256")
        if (
            declared_digest is not None
            and snapshot.semantic_digest() != declared_digest
        ):
            raise ValueError("region snapshot semantic SHA-256 mismatch")
        return snapshot

    def section_semantic_digest(
        self,
        chunk_slot: int,
        section_y: int,
    ) -> str:
        """Hash expanded semantics so palette ordering cannot hide conflicts."""

        slot = int(chunk_slot)
        section = int(section_y)
        if not (
            0 <= slot < CAPTURE_CHUNK_COUNT
            and 0 <= section < HEIGHT_SECTIONS
            and self.section_known[slot, section]
        ):
            raise ValueError("cannot hash an unknown region section")
        return self._section_semantic_digest(
            slot,
            section,
            self._cell_semantic_hashes(),
        )

    def _cell_semantic_hashes(self) -> np.ndarray:
        cell = self.cell_palette
        result = np.empty((cell.size, hashlib.sha256().digest_size), dtype=np.uint8)
        for code in range(cell.size):
            shape = int(cell.shape_index[code])
            mask = self.shape_palette.box_mask[shape]
            active_boxes = _canonical_collision_boxes(
                self.shape_palette.boxes[shape][mask]
            )
            digest = hashlib.sha256()
            semantic_values = [
                cell.flags[code : code + 1],
                cell.fluid_level[code : code + 1],
                cell.support[code : code + 1],
                cell.block_damage[code : code + 1],
                cell.fluid_damage[code : code + 1],
                _canonical_float_array(cell.movement[code]),
                _canonical_float_array(cell.fluid_movement[code]),
                np.asarray([active_boxes.shape[0]], dtype=np.uint8),
                active_boxes,
            ]
            if int(self.metadata.get("section_protocol_version", 1)) >= 2:
                semantic_values.insert(
                    2,
                    _canonical_float_array(
                        cell.fluid_fill_height[code : code + 1]
                    ),
                )
            for values in semantic_values:
                contiguous = np.ascontiguousarray(values)
                digest.update(str(contiguous.dtype).encode("ascii"))
                digest.update(contiguous.tobytes())
            result[code] = np.frombuffer(digest.digest(), dtype=np.uint8)
        return result

    def _section_semantic_digest(
        self,
        slot: int,
        section: int,
        cell_hashes: np.ndarray,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(
            np.ascontiguousarray(cell_hashes[self.cell_code[slot, section]]).tobytes()
        )
        if self.filler_root_offsets_available:
            digest.update(
                np.ascontiguousarray(
                    self.filler_root_offset_packed[slot, section],
                    dtype="<u2",
                ).tobytes()
            )
        return digest.hexdigest()

    def semantic_digest(self) -> str:
        """Hash complete expanded geometry independently of palette order."""

        digest = hashlib.sha256()
        digest.update(self.core_min_chunk_xz.tobytes())
        digest.update(self.section_known.tobytes())
        cell_hashes = self._cell_semantic_hashes()
        for slot in range(CAPTURE_CHUNK_COUNT):
            for section in range(HEIGHT_SECTIONS):
                if not self.section_known[slot, section]:
                    continue
                digest.update(slot.to_bytes(1, "little"))
                digest.update(section.to_bytes(1, "little"))
                digest.update(
                    bytes.fromhex(
                        self._section_semantic_digest(
                            slot,
                            section,
                            cell_hashes,
                        )
                    )
                )
        return digest.hexdigest()

    def semantic_artifact_digest(self) -> str:
        """Alias naming this digest as the immutable artifact content ID."""

        return self.semantic_digest()


def empty_cell_palette() -> CellPalette:
    return CellPalette(
        flags=np.zeros(1, dtype=np.uint16),
        shape_index=np.zeros(1, dtype=np.uint16),
        fluid_level=np.zeros(1, dtype=np.uint8),
        support=np.zeros(1, dtype=np.int32),
        block_damage=np.zeros(1, dtype=np.int32),
        fluid_damage=np.zeros(1, dtype=np.int32),
        movement=np.zeros((1, MOVEMENT_FEATURES), dtype=np.float32),
        fluid_movement=np.zeros(
            (1, FLUID_MOVEMENT_FEATURES),
            dtype=np.float32,
        ),
    )


def empty_shape_palette() -> ShapePalette:
    return ShapePalette(
        boxes=np.zeros((1, MAX_SHAPE_BOXES, 6), dtype=np.float32),
        box_mask=np.zeros((1, MAX_SHAPE_BOXES), dtype=np.bool_),
    )


def _vector(
    value: Any,
    dtype: np.dtype,
    name: str,
    size: int | None = None,
) -> np.ndarray:
    raw = np.asarray(value)
    expected = raw.shape[0] if size is None and raw.ndim == 1 else size
    return _integer_array(
        raw,
        dtype,
        name,
        expected_shape=(expected,),
    )


def _matrix(
    value: Any,
    dtype: np.dtype,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return result.copy()


def _canonical_float_array(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).copy()
    result[result == 0.0] = 0.0
    return result


def _canonical_collision_boxes(value: Any) -> np.ndarray:
    return _canonical_float_array(value)


def _integer_array(
    value: Any,
    dtype: np.dtype,
    name: str,
    *,
    expected_shape: tuple[int | None, ...],
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(
        raw.dtype,
        np.bool_,
    ):
        raise ValueError(f"{name} must contain integers")
    limits = np.iinfo(dtype)
    if np.any(raw < limits.min) or np.any(raw > limits.max):
        raise ValueError(f"{name} exceeds {np.dtype(dtype).name} capacity")
    return raw.astype(dtype, copy=True)


def _bool_array(
    value: Any,
    name: str,
    *,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if result.dtype != np.bool_:
        raise ValueError(f"{name} must contain booleans")
    return result.copy()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
