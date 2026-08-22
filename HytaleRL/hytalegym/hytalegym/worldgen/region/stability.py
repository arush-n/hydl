"""Palette-independent Region v1 stability diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import operator
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from hytalegym.geometry.contract import (
    FLAG_DAMAGING,
    FLAG_FLUID,
    FLAG_HAS_FLUID_MOVEMENT_SETTINGS,
    FLAG_HAS_MOVEMENT_SETTINGS,
)
from hytalegym.worldgen.region.contract import (
    CAPTURE_CHUNK_COUNT,
    CAPTURE_CHUNKS_PER_AXIS,
    CHUNK_SIZE,
    HEIGHT_SECTIONS,
    MIN_Y,
)
from hytalegym.worldgen.region.snapshot import NativeRegionSnapshot

# Canonical pin for the currently deployed native evidence bridge.
# Historical fixture manifests retain their own capture hashes.
CURRENT_NATIVE_EVIDENCE_JAR_SHA256 = (
    "21C4B1FAA9087F6AAED0151078BF985F1196B02767669ABF35DFE2A801F25D3D"
)
_SHA256_PATTERN = re.compile(r"^[0-9A-F]{64}$")
REGION_SEED_PROJECTION_SCHEMA = "hytalerl_region_seed_projection_contract_v2"
REGION_SEED_PROJECTION_VERSION = 2
_REGION_SEED_PROJECTION_HASH_SCHEMA = (
    "hytalerl_region_seed_projection_semantics_v2"
)
_IDENTITY_FIELDS = (
    "schema",
    "version",
    "server_version",
    "seed",
    "worldgen_provider",
    "worldgen_version",
    "chunk_api",
    "capture_mode",
    "exact_collision_shapes",
    "dynamic_state",
    "section_protocol_schema",
    "section_protocol_version",
    "exact_filler_root_offsets",
    "filler_root_offset_encoding",
)


def current_native_evidence_jar_sha256() -> str:
    """Return the single current bridge hash used by active native gates."""

    if _SHA256_PATTERN.fullmatch(CURRENT_NATIVE_EVIDENCE_JAR_SHA256) is None:
        raise RuntimeError("current native evidence JAR SHA-256 is malformed")
    return CURRENT_NATIVE_EVIDENCE_JAR_SHA256


def require_current_native_runtime_bridge(
    info: Mapping[str, Any],
    *,
    expected: str | None = None,
) -> str:
    """Fail if a live observation came from stale loaded bridge bytecode."""

    target = expected or current_native_evidence_jar_sha256()
    actual = info.get("bridge_sha256")
    if not isinstance(actual, str) or _SHA256_PATTERN.fullmatch(actual.upper()) is None:
        raise RuntimeError("native observation lacks a valid runtime bridge SHA-256")
    observed = actual.upper()
    if observed != target:
        raise RuntimeError(
            "running bridge differs from canonical world evidence: "
            f"observed={observed} expected={target}"
        )
    return observed


@dataclass(frozen=True)
class RegionCellSemantics:
    flags: int
    fluid_level: int
    fluid_fill_height: float
    support: int
    block_damage: int
    fluid_damage: int
    movement: tuple[float, ...]
    fluid_movement: tuple[float, ...]
    shape_box_count: int
    collision_boxes: tuple[tuple[float, ...], ...]
    filler_root_offset: tuple[int, int, int]


@dataclass(frozen=True)
class RegionCaptureIdentity:
    server_version: str
    seed: int
    world: str
    worldgen_provider: str
    worldgen_version: str
    core_min_chunk_xz: tuple[int, int]
    chunk_api: tuple[tuple[str, Any], ...]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class RegionSeedProjectionContract:
    """Bridge-bound scope for seed regeneration, never artifact equality."""

    server_version: str
    seed: int
    evidence_bridge_sha256: str
    reference_artifact_bridge_sha256: str
    reference_artifact_file_sha256: str
    reference_artifact_semantic_sha256: str
    reference_seed_projection_sha256: str
    max_non_damaging_fluid_only_cells: int

    def __post_init__(self) -> None:
        if not isinstance(self.server_version, str) or not self.server_version:
            raise ValueError("Region seed projection requires server_version")
        object.__setattr__(
            self,
            "seed",
            _exact_nonnegative_int(self.seed, "seed"),
        )
        for field in (
            "evidence_bridge_sha256",
            "reference_artifact_bridge_sha256",
            "reference_artifact_file_sha256",
            "reference_artifact_semantic_sha256",
            "reference_seed_projection_sha256",
        ):
            object.__setattr__(
                self,
                field,
                _required_sha256(getattr(self, field), field),
            )
        object.__setattr__(
            self,
            "max_non_damaging_fluid_only_cells",
            _exact_nonnegative_int(
                self.max_non_damaging_fluid_only_cells,
                "max_non_damaging_fluid_only_cells",
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "RegionSeedProjectionContract":
        if not isinstance(value, Mapping):
            raise TypeError("Region seed-projection contract must be a mapping")
        required_fields = {
            "schema",
            "version",
            "server_version",
            "seed",
            "evidence_bridge_sha256",
            "reference_artifact_bridge_sha256",
            "reference_artifact_file_sha256",
            "reference_artifact_semantic_sha256",
            "reference_seed_projection_sha256",
            "max_non_damaging_fluid_only_cells",
        }
        unknown_fields = set(value) - required_fields
        if unknown_fields:
            raise ValueError(
                "unknown Region seed-projection fields: "
                + ", ".join(sorted(unknown_fields))
            )
        if value.get("schema") != REGION_SEED_PROJECTION_SCHEMA:
            raise ValueError("unsupported Region seed-projection schema")
        version = value.get("version")
        if isinstance(version, bool) or version != REGION_SEED_PROJECTION_VERSION:
            raise ValueError("unsupported Region seed-projection version")
        server_version = value.get("server_version")
        if not isinstance(server_version, str) or not server_version:
            raise ValueError("Region seed projection requires server_version")
        seed = _exact_nonnegative_int(value.get("seed"), "seed")
        return cls(
            server_version=server_version,
            seed=seed,
            evidence_bridge_sha256=_required_sha256(
                value.get("evidence_bridge_sha256"),
                "evidence_bridge_sha256",
            ),
            reference_artifact_bridge_sha256=_required_sha256(
                value.get("reference_artifact_bridge_sha256"),
                "reference_artifact_bridge_sha256",
            ),
            reference_artifact_file_sha256=_required_sha256(
                value.get("reference_artifact_file_sha256"),
                "reference_artifact_file_sha256",
            ),
            reference_artifact_semantic_sha256=_required_sha256(
                value.get("reference_artifact_semantic_sha256"),
                "reference_artifact_semantic_sha256",
            ),
            reference_seed_projection_sha256=_required_sha256(
                value.get("reference_seed_projection_sha256"),
                "reference_seed_projection_sha256",
            ),
            max_non_damaging_fluid_only_cells=_exact_nonnegative_int(
                value.get("max_non_damaging_fluid_only_cells"),
                "max_non_damaging_fluid_only_cells",
            ),
        )

    def require_evidence_bridge(self, value: str | None) -> None:
        actual = _required_sha256(value, "native_evidence_jar_sha256")
        if actual != self.evidence_bridge_sha256:
            raise ValueError(
                "Region seed-projection evidence bridge changed: "
                f"contract={self.evidence_bridge_sha256}, active={actual}"
            )

    def require_snapshot_projection(
        self,
        snapshot: NativeRegionSnapshot,
        *,
        context: str = "Region snapshot",
    ) -> None:
        if snapshot.metadata.get("server_version") != self.server_version:
            raise ValueError(
                f"{context}: Region seed-projection server version changed"
            )
        if snapshot.metadata.get("seed") != self.seed:
            raise ValueError(f"{context}: Region seed-projection seed changed")
        actual = region_seed_projection_digest(snapshot).upper()
        if actual != self.reference_seed_projection_sha256:
            raise ValueError(
                f"{context}: Region seed-projection digest changed: "
                f"expected {self.reference_seed_projection_sha256}, got {actual}; "
                "inspect this capture before re-pinning evidence"
            )


def load_region_seed_projection_contract(
    path: str | Path,
) -> RegionSeedProjectionContract:
    """Load a strict JSON projection contract outside compiled execution."""

    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    return RegionSeedProjectionContract.from_mapping(value)


@dataclass(frozen=True)
class RegionSectionSemanticDigest:
    chunk_slot: int
    chunk_x: int
    chunk_z: int
    section_y: int
    first: str
    second: str
    equal: bool


@dataclass(frozen=True)
class RegionCellDifference:
    first_world: str
    second_world: str
    chunk_slot: int
    chunk_x: int
    chunk_z: int
    section_y: int
    local_x: int
    local_y: int
    local_z: int
    block_x: int
    block_y: int
    block_z: int
    fields: tuple[str, ...]
    first: RegionCellSemantics
    second: RegionCellSemantics


@dataclass(frozen=True)
class RegionSemanticComparison:
    gate: str
    first_identity: RegionCaptureIdentity
    second_identity: RegionCaptureIdentity
    first_digest: str
    second_digest: str
    first_seed_projection_digest: str
    second_seed_projection_digest: str
    differing_cell_count: int
    non_damaging_fluid_only_cell_count: int
    other_differing_cell_count: int
    differences: tuple[RegionCellDifference, ...]
    layout_mismatches: tuple[str, ...]
    section_digests: tuple[RegionSectionSemanticDigest, ...]
    native_evidence_jar_sha256: str | None
    disposition: str
    difference_limit: int

    @property
    def equal(self) -> bool:
        return not self.layout_mismatches and self.differing_cell_count == 0

    @property
    def differences_truncated(self) -> bool:
        return self.differing_cell_count > len(self.differences)

    @property
    def only_non_damaging_fluid_differences(self) -> bool:
        return (
            not self.layout_mismatches
            and self.differing_cell_count > 0
            and self.other_differing_cell_count == 0
        )

    @property
    def seed_projection_equal(self) -> bool:
        return (
            not self.layout_mismatches
            and self.other_differing_cell_count == 0
            and self.first_seed_projection_digest
            == self.second_seed_projection_digest
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["equal"] = self.equal
        result["differences_truncated"] = self.differences_truncated
        result["complete_artifact_semantic_sha256"] = {
            "first": self.first_digest,
            "second": self.second_digest,
        }
        result["seed_projection_semantic_sha256"] = {
            "first": self.first_seed_projection_digest,
            "second": self.second_seed_projection_digest,
        }
        result["per_section_semantic_digests"] = result.pop("section_digests")
        result["provenance"] = {
            "first": result["first_identity"],
            "second": result["second_identity"],
            "native_evidence_jar_sha256": self.native_evidence_jar_sha256,
        }
        for difference in result["differences"]:
            first = difference["first"]
            second = difference["second"]
            first["shape"] = {
                "box_count": first.pop("shape_box_count"),
            }
            second["shape"] = {
                "box_count": second.pop("shape_box_count"),
            }
            first["FluidFX"] = first.pop("fluid_movement")
            second["FluidFX"] = second.pop("fluid_movement")
        return result


class RegionSemanticMismatchError(ValueError):
    """A Region capture changed or cannot be aligned semantically."""

    def __init__(
        self,
        comparison: RegionSemanticComparison,
        *,
        context: str,
        first_snapshot: NativeRegionSnapshot | None = None,
        second_snapshot: NativeRegionSnapshot | None = None,
    ):
        self.comparison = comparison
        self.context = context
        self.first_snapshot = first_snapshot
        self.second_snapshot = second_snapshot
        details = list(comparison.layout_mismatches)
        if comparison.differences:
            first = comparison.differences[0]
            details.append(
                "first changed block "
                f"({first.block_x}, {first.block_y}, {first.block_z}) "
                f"fields={','.join(first.fields)}"
            )
        suffix = f": {'; '.join(details)}" if details else ""
        super().__init__(
            f"{context} changed {comparison.differing_cell_count} semantic cells"
            f"{suffix}"
        )


def compare_region_semantics(
    first: NativeRegionSnapshot,
    second: NativeRegionSnapshot,
    *,
    require_same_world: bool,
    difference_limit: int = 16,
    native_evidence_jar_sha256: str | None = None,
) -> RegionSemanticComparison:
    """Compare complete expanded semantics without depending on palette IDs."""

    limit = _exact_nonnegative_int(difference_limit, "difference_limit")
    jar_hash = _validated_sha256(native_evidence_jar_sha256)
    gate = (
        "same_world_stability" if require_same_world else "cross_reset_reproducibility"
    )
    _require_complete(first, "first Region artifact")
    _require_complete(second, "second Region artifact")
    layout_mismatches = _layout_mismatches(
        first,
        second,
        require_same_world=require_same_world,
    )
    first_digest = first.semantic_digest()
    second_digest = second.semantic_digest()
    first_seed_projection_digest = region_seed_projection_digest(first)
    second_seed_projection_digest = region_seed_projection_digest(second)
    section_digests = _section_digests(first, second)
    if layout_mismatches:
        return RegionSemanticComparison(
            gate=gate,
            first_identity=_capture_identity(first),
            second_identity=_capture_identity(second),
            first_digest=first_digest,
            second_digest=second_digest,
            first_seed_projection_digest=first_seed_projection_digest,
            second_seed_projection_digest=second_seed_projection_digest,
            differing_cell_count=0,
            non_damaging_fluid_only_cell_count=0,
            other_differing_cell_count=0,
            differences=(),
            layout_mismatches=layout_mismatches,
            section_digests=section_digests,
            native_evidence_jar_sha256=jar_hash,
            disposition=_failure_disposition(require_same_world),
            difference_limit=limit,
        )

    first_keys = _palette_keys(first)
    second_keys = _palette_keys(second)
    identifiers: dict[tuple[Any, ...], int] = {}
    first_ids = _palette_ids(first_keys, identifiers)
    second_ids = _palette_ids(second_keys, identifiers)
    projected_identifiers: dict[tuple[Any, ...], int] = {}
    first_projected_ids = _palette_ids(
        _non_damaging_fluid_projection_keys(first),
        projected_identifiers,
    )
    second_projected_ids = _palette_ids(
        _non_damaging_fluid_projection_keys(second),
        projected_identifiers,
    )
    differences: list[RegionCellDifference] = []
    differing_cell_count = 0
    non_damaging_fluid_only_cell_count = 0
    other_differing_cell_count = 0

    for slot, section in np.argwhere(first.section_known):
        first_codes = first.cell_code[slot, section]
        second_codes = second.cell_code[slot, section]
        changed = first_ids[first_codes] != second_ids[second_codes]
        projected_changed = (
            first_projected_ids[first_codes]
            != second_projected_ids[second_codes]
        )
        filler_changed = (
            first.filler_root_offset_packed[slot, section]
            != second.filler_root_offset_packed[slot, section]
        )
        changed |= filler_changed
        projected_changed |= filler_changed
        differing_cell_count += int(np.count_nonzero(changed))
        other_differing_cell_count += int(
            np.count_nonzero(changed & projected_changed)
        )
        non_damaging_fluid_only_cell_count += int(
            np.count_nonzero(changed & ~projected_changed)
        )
        remaining = limit - len(differences)
        if remaining <= 0:
            continue
        for index in np.flatnonzero(changed)[:remaining]:
            differences.append(
                _cell_difference(
                    first,
                    second,
                    int(slot),
                    int(section),
                    int(index),
                )
            )

    return RegionSemanticComparison(
        gate=gate,
        first_identity=_capture_identity(first),
        second_identity=_capture_identity(second),
        first_digest=first_digest,
        second_digest=second_digest,
        first_seed_projection_digest=first_seed_projection_digest,
        second_seed_projection_digest=second_seed_projection_digest,
        differing_cell_count=differing_cell_count,
        non_damaging_fluid_only_cell_count=(
            non_damaging_fluid_only_cell_count
        ),
        other_differing_cell_count=other_differing_cell_count,
        differences=tuple(differences),
        layout_mismatches=(),
        section_digests=section_digests,
        native_evidence_jar_sha256=jar_hash,
        disposition=(
            "pass"
            if differing_cell_count == 0
            else _failure_disposition(require_same_world)
        ),
        difference_limit=limit,
    )


def require_region_semantics_equal(
    first: NativeRegionSnapshot,
    second: NativeRegionSnapshot,
    *,
    require_same_world: bool,
    context: str,
    difference_limit: int = 16,
    native_evidence_jar_sha256: str | None = None,
) -> RegionSemanticComparison:
    comparison = compare_region_semantics(
        first,
        second,
        require_same_world=require_same_world,
        difference_limit=difference_limit,
        native_evidence_jar_sha256=native_evidence_jar_sha256,
    )
    if not comparison.equal:
        raise RegionSemanticMismatchError(
            comparison,
            context=context,
            first_snapshot=first,
            second_snapshot=second,
        )
    return comparison


def require_region_seed_projection_equal(
    first: NativeRegionSnapshot,
    second: NativeRegionSnapshot,
    *,
    contract: RegionSeedProjectionContract,
    context: str,
    difference_limit: int = 16,
    native_evidence_jar_sha256: str | None = None,
) -> RegionSemanticComparison:
    """Require bridge-bound non-fluid seed identity with an explicit bound.

    Exact artifact equality remains available through
    ``require_region_semantics_equal``. This narrower gate is valid only for
    cross-reset seed regeneration and never for same-world capture stability.
    """

    contract.require_evidence_bridge(native_evidence_jar_sha256)
    comparison = compare_region_semantics(
        first,
        second,
        require_same_world=False,
        difference_limit=difference_limit,
        native_evidence_jar_sha256=native_evidence_jar_sha256,
    )
    if (
        not comparison.seed_projection_equal
        or comparison.non_damaging_fluid_only_cell_count
        > contract.max_non_damaging_fluid_only_cells
    ):
        raise RegionSemanticMismatchError(
            comparison,
            context=context,
            first_snapshot=first,
            second_snapshot=second,
        )
    contract.require_snapshot_projection(
        first,
        context=f"{context}: first snapshot",
    )
    contract.require_snapshot_projection(
        second,
        context=f"{context}: second snapshot",
    )
    return comparison


def region_seed_projection_digest(snapshot: NativeRegionSnapshot) -> str:
    """Hash exact dry semantics after removing non-damaging fluid state."""

    _require_complete(snapshot, "Region seed-projection artifact")
    projected_keys = _non_damaging_fluid_projection_keys(snapshot)
    cell_hashes = np.stack(
        [
            np.frombuffer(
                hashlib.sha256(
                    json.dumps(
                        _projection_key_json(key),
                        separators=(",", ":"),
                    ).encode("ascii")
                ).digest(),
                dtype=np.uint8,
            )
            for key in projected_keys
        ]
    )

    digest = hashlib.sha256()
    digest.update(_REGION_SEED_PROJECTION_HASH_SCHEMA.encode("ascii"))
    digest.update(snapshot.core_min_chunk_xz.tobytes())
    digest.update(snapshot.section_known.tobytes())
    for slot in range(CAPTURE_CHUNK_COUNT):
        for section in range(HEIGHT_SECTIONS):
            if not snapshot.section_known[slot, section]:
                continue
            digest.update(slot.to_bytes(1, "little"))
            digest.update(section.to_bytes(1, "little"))
            digest.update(
                bytes.fromhex(
                    snapshot._section_semantic_digest(
                        slot,
                        section,
                        cell_hashes,
                    )
                )
            )
    return digest.hexdigest()


def _layout_mismatches(
    first: NativeRegionSnapshot,
    second: NativeRegionSnapshot,
    *,
    require_same_world: bool,
) -> tuple[str, ...]:
    mismatches: list[str] = []
    if not np.array_equal(first.core_min_chunk_xz, second.core_min_chunk_xz):
        mismatches.append("core_min_chunk_xz differs")
    if not np.array_equal(first.section_known, second.section_known):
        mismatches.append("section_known differs")
    fields = _IDENTITY_FIELDS + (("world",) if require_same_world else ())
    for field in fields:
        if first.metadata.get(field) != second.metadata.get(field):
            mismatches.append(f"metadata.{field} differs")
    return tuple(mismatches)


def _capture_identity(snapshot: NativeRegionSnapshot) -> RegionCaptureIdentity:
    metadata = snapshot.metadata
    chunk_api = metadata.get("chunk_api", {})
    return RegionCaptureIdentity(
        server_version=str(metadata.get("server_version", "")),
        seed=int(metadata.get("seed", 0)),
        world=str(metadata.get("world", "")),
        worldgen_provider=str(metadata.get("worldgen_provider", "")),
        worldgen_version=str(metadata.get("worldgen_version", "")),
        core_min_chunk_xz=(
            int(snapshot.core_min_chunk_xz[0]),
            int(snapshot.core_min_chunk_xz[1]),
        ),
        chunk_api=tuple(sorted(dict(chunk_api).items())),
        manifest=_json_safe(dict(metadata)),
    )


def _palette_keys(snapshot: NativeRegionSnapshot) -> list[tuple[Any, ...]]:
    palette = snapshot.cell_palette
    shape_keys = _shape_keys(snapshot)
    include_fill = int(
        snapshot.metadata.get("section_protocol_version", 1)
    ) >= 2
    return [
        (
            int(palette.flags[index]),
            int(palette.fluid_level[index]),
            *(
                (_float_bytes(palette.fluid_fill_height[index : index + 1]),)
                if include_fill
                else ()
            ),
            int(palette.support[index]),
            int(palette.block_damage[index]),
            int(palette.fluid_damage[index]),
            _float_bytes(palette.movement[index]),
            _float_bytes(palette.fluid_movement[index]),
            shape_keys[int(palette.shape_index[index])],
        )
        for index in range(palette.size)
    ]


def _non_damaging_fluid_projection_keys(
    snapshot: NativeRegionSnapshot,
) -> list[tuple[Any, ...]]:
    """Remove only non-damaging fluid state from otherwise exact cells.

    This projection is never artifact or same-world equality. It is accepted
    only by an explicit seed-projection contract. Pure fluid becomes air;
    fluid over a block preserves that block's flags, movement, support,
    damage, shape, and filler semantics. Damaging fluid remains exact.
    """

    palette = snapshot.cell_palette
    shape_keys = _shape_keys(snapshot)
    exact_keys = _palette_keys(snapshot)
    include_fill = int(
        snapshot.metadata.get("section_protocol_version", 1)
    ) >= 2
    pure_fluid_flags = (
        FLAG_FLUID
        | FLAG_HAS_MOVEMENT_SETTINGS
        | FLAG_HAS_FLUID_MOVEMENT_SETTINGS
    )
    fluid_overlay_flags = (
        FLAG_FLUID
        | FLAG_HAS_FLUID_MOVEMENT_SETTINGS
    )
    zero_fill = _float_bytes(np.zeros(1, dtype=np.float32))
    zero_fluid_movement = _float_bytes(
        np.zeros(palette.fluid_movement.shape[1], dtype=np.float32)
    )
    air = (
        0,
        0,
        *(
            (zero_fill,)
            if include_fill
            else ()
        ),
        0,
        0,
        0,
        _float_bytes(np.zeros(palette.movement.shape[1], dtype=np.float32)),
        zero_fluid_movement,
        (0, b""),
    )
    result: list[tuple[Any, ...]] = []
    for index in range(palette.size):
        flags = int(palette.flags[index])
        shape = shape_keys[int(palette.shape_index[index])]
        non_damaging_fluid = (
            bool(flags & FLAG_FLUID)
            and int(palette.fluid_damage[index]) == 0
            and not (
                flags & FLAG_DAMAGING
                and int(palette.block_damage[index]) == 0
            )
        )
        fluid_only = (
            non_damaging_fluid
            and flags & ~pure_fluid_flags == 0
            and int(palette.support[index]) == 0
            and int(palette.block_damage[index]) == 0
            and shape[0] == 0
        )
        if fluid_only:
            result.append(air)
        elif non_damaging_fluid:
            result.append(
                (
                    flags & ~fluid_overlay_flags,
                    0,
                    *((zero_fill,) if include_fill else ()),
                    int(palette.support[index]),
                    int(palette.block_damage[index]),
                    0,
                    _float_bytes(palette.movement[index]),
                    zero_fluid_movement,
                    shape,
                )
            )
        else:
            result.append(exact_keys[index])
    return result


def _shape_keys(snapshot: NativeRegionSnapshot) -> list[tuple[int, bytes]]:
    result: list[tuple[int, bytes]] = []
    for boxes, mask in zip(
        snapshot.shape_palette.boxes,
        snapshot.shape_palette.box_mask,
        strict=True,
    ):
        active = _canonical_collision_boxes(boxes[mask])
        result.append((int(active.shape[0]), _float_bytes(active)))
    return result


def _palette_ids(
    keys: list[tuple[Any, ...]],
    identifiers: dict[tuple[Any, ...], int],
) -> np.ndarray:
    result = np.empty(len(keys), dtype=np.int32)
    for index, key in enumerate(keys):
        result[index] = identifiers.setdefault(key, len(identifiers))
    return result


def _projection_key_json(value: object) -> object:
    if isinstance(value, tuple):
        return [_projection_key_json(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, int):
        return value
    raise TypeError(
        "Region seed-projection keys must contain tuples, bytes, and integers"
    )


def _cell_difference(
    first: NativeRegionSnapshot,
    second: NativeRegionSnapshot,
    slot: int,
    section: int,
    index: int,
) -> RegionCellDifference:
    local_y, remainder = divmod(index, CHUNK_SIZE * CHUNK_SIZE)
    local_z, local_x = divmod(remainder, CHUNK_SIZE)
    chunk_x = int(first.core_min_chunk_xz[0]) - 1 + slot // CAPTURE_CHUNKS_PER_AXIS
    chunk_z = int(first.core_min_chunk_xz[1]) - 1 + slot % CAPTURE_CHUNKS_PER_AXIS
    first_semantics = _cell_semantics(first, slot, section, index)
    second_semantics = _cell_semantics(second, slot, section, index)
    return RegionCellDifference(
        first_world=str(first.metadata.get("world", "")),
        second_world=str(second.metadata.get("world", "")),
        chunk_slot=slot,
        chunk_x=chunk_x,
        chunk_z=chunk_z,
        section_y=section,
        local_x=local_x,
        local_y=local_y,
        local_z=local_z,
        block_x=chunk_x * CHUNK_SIZE + local_x,
        block_y=MIN_Y + section * CHUNK_SIZE + local_y,
        block_z=chunk_z * CHUNK_SIZE + local_z,
        fields=_changed_fields(first_semantics, second_semantics),
        first=first_semantics,
        second=second_semantics,
    )


def _cell_semantics(
    snapshot: NativeRegionSnapshot,
    slot: int,
    section: int,
    index: int,
) -> RegionCellSemantics:
    palette = snapshot.cell_palette
    code = int(snapshot.cell_code[slot, section, index])
    shape = int(palette.shape_index[code])
    boxes = snapshot.shape_palette.boxes[shape]
    mask = snapshot.shape_palette.box_mask[shape]
    active_boxes = _canonical_collision_boxes(boxes[mask])
    return RegionCellSemantics(
        flags=int(palette.flags[code]),
        fluid_level=int(palette.fluid_level[code]),
        fluid_fill_height=float(palette.fluid_fill_height[code]),
        support=int(palette.support[code]),
        block_damage=int(palette.block_damage[code]),
        fluid_damage=int(palette.fluid_damage[code]),
        movement=tuple(float(value) for value in palette.movement[code]),
        fluid_movement=tuple(float(value) for value in palette.fluid_movement[code]),
        shape_box_count=int(active_boxes.shape[0]),
        collision_boxes=tuple(
            tuple(float(value) for value in box) for box in active_boxes
        ),
        filler_root_offset=_unpack_filler_root_offset(
            int(snapshot.filler_root_offset_packed[slot, section, index])
        ),
    )


def _changed_fields(
    first: RegionCellSemantics,
    second: RegionCellSemantics,
) -> tuple[str, ...]:
    result: list[str] = []
    for field in (
        "flags",
        "fluid_level",
        "fluid_fill_height",
        "support",
        "block_damage",
        "fluid_damage",
        "movement",
        "filler_root_offset",
    ):
        if getattr(first, field) != getattr(second, field):
            result.append(field)
    if first.fluid_movement != second.fluid_movement:
        result.append("FluidFX")
    if first.shape_box_count != second.shape_box_count:
        result.append("shape")
    if first.collision_boxes != second.collision_boxes:
        result.append("collision_boxes")
    return tuple(result)


def _float_bytes(values: np.ndarray) -> bytes:
    canonical = np.asarray(values, dtype=np.float32).copy()
    canonical[canonical == 0.0] = 0.0
    return np.ascontiguousarray(canonical).tobytes()


def _canonical_collision_boxes(values: np.ndarray) -> np.ndarray:
    boxes = np.asarray(values, dtype=np.float32).copy()
    boxes[boxes == 0.0] = 0.0
    return boxes


def _unpack_filler_root_offset(value: int) -> tuple[int, int, int]:
    def signed_axis(raw: int) -> int:
        return raw - 32 if raw & 16 else raw

    return (
        signed_axis(value & 31),
        signed_axis((value >> 10) & 31),
        signed_axis((value >> 5) & 31),
    )


def _section_digests(
    first: NativeRegionSnapshot,
    second: NativeRegionSnapshot,
) -> tuple[RegionSectionSemanticDigest, ...]:
    first_hashes = first._cell_semantic_hashes()
    second_hashes = second._cell_semantic_hashes()
    result: list[RegionSectionSemanticDigest] = []
    for slot in range(CAPTURE_CHUNK_COUNT):
        relative_x, relative_z = divmod(slot, CAPTURE_CHUNKS_PER_AXIS)
        chunk_x = int(first.core_min_chunk_xz[0]) - 1 + relative_x
        chunk_z = int(first.core_min_chunk_xz[1]) - 1 + relative_z
        for section in range(HEIGHT_SECTIONS):
            first_digest = first._section_semantic_digest(
                slot,
                section,
                first_hashes,
            )
            second_digest = second._section_semantic_digest(
                slot,
                section,
                second_hashes,
            )
            result.append(
                RegionSectionSemanticDigest(
                    chunk_slot=slot,
                    chunk_x=chunk_x,
                    chunk_z=chunk_z,
                    section_y=section,
                    first=first_digest,
                    second=second_digest,
                    equal=first_digest == second_digest,
                )
            )
    return tuple(result)


def _require_complete(
    snapshot: NativeRegionSnapshot,
    label: str,
) -> None:
    if not bool(np.all(snapshot.section_known)):
        missing = int(snapshot.section_known.size) - int(
            np.count_nonzero(snapshot.section_known)
        )
        raise ValueError(
            f"{label} is incomplete or contains unknown sections "
            f"({missing} missing)"
        )


def _validated_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    result = str(value).upper()
    if not _SHA256_PATTERN.fullmatch(result):
        raise ValueError("native evidence JAR SHA-256 must be 64 hexadecimal digits")
    return result


def _required_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a SHA-256 string")
    result = value.upper()
    if _SHA256_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{label} must be 64 hexadecimal digits")
    return result


def _failure_disposition(require_same_world: bool) -> str:
    if require_same_world:
        return "reject_dynamic_state_static_and_reject_snapshot_fail_closed"
    return (
        "retain_only_confirmed_immutable_artifact_by_semantic_hash;"
        "reject_seed_only_regeneration"
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _exact_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a non-negative integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be a non-negative integer") from error
    if result < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return result
