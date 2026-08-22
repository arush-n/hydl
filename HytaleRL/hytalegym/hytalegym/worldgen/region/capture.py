"""Deterministic coverage plan for one complete 3x3 traversal region."""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator

from hytalegym.worldgen.region.contract import (
    CAPTURE_CHUNKS_PER_AXIS,
    CAPTURE_HALO_CHUNKS,
    CHUNK_SIZE,
    HEIGHT_SECTIONS,
    MAX_CHUNK_COORDINATE,
    MIN_CHUNK_COORDINATE,
)


@dataclass(frozen=True)
class RegionCaptureTask:
    chunk_x: int
    chunk_z: int
    section_y: int


@dataclass(frozen=True)
class RegionCapturePlan:
    """The 5x5-column halo and all ten sections needed by a 3x3 core."""

    core_min_chunk_x: int
    core_min_chunk_z: int
    tasks: tuple[RegionCaptureTask, ...]

    @classmethod
    def complete(
        cls,
        core_min_chunk_x: int,
        core_min_chunk_z: int,
        *,
        maximum_source_reach_blocks: float,
    ) -> "RegionCapturePlan":
        reach = float(maximum_source_reach_blocks)
        if (
            not math.isfinite(reach)
            or reach < 0.0
            or reach >= CAPTURE_HALO_CHUNKS * CHUNK_SIZE
        ):
            raise ValueError("exact geometry source reach exceeds the one-chunk halo")
        core_x = _coordinate(core_min_chunk_x)
        core_z = _coordinate(core_min_chunk_z)
        for value in (core_x, core_z):
            if (
                value - CAPTURE_HALO_CHUNKS < MIN_CHUNK_COORDINATE
                or value + CAPTURE_CHUNKS_PER_AXIS - CAPTURE_HALO_CHUNKS - 1
                > MAX_CHUNK_COORDINATE
            ):
                raise ValueError("region capture exceeds native chunk bounds")
        minimum_x = core_x - CAPTURE_HALO_CHUNKS
        minimum_z = core_z - CAPTURE_HALO_CHUNKS
        tasks = tuple(
            RegionCaptureTask(minimum_x + dx, minimum_z + dz, section_y)
            for dx in range(CAPTURE_CHUNKS_PER_AXIS)
            for dz in range(CAPTURE_CHUNKS_PER_AXIS)
            for section_y in range(HEIGHT_SECTIONS)
        )
        return cls(
            core_min_chunk_x=core_x,
            core_min_chunk_z=core_z,
            tasks=tasks,
        )

    @property
    def capture_min_chunk_x(self) -> int:
        return self.core_min_chunk_x - CAPTURE_HALO_CHUNKS

    @property
    def capture_min_chunk_z(self) -> int:
        return self.core_min_chunk_z - CAPTURE_HALO_CHUNKS


def _coordinate(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("core chunk coordinates must be integers")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError("core chunk coordinates must be integers") from error
