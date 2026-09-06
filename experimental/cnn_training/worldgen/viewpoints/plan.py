"""Chunk-aligned capture planning for contiguous CNN scenes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

CHUNK_SIZE = 32
CORE_CHUNKS_PER_AXIS = 3
CAPTURE_HALO_CHUNKS = 1
DEFAULT_RENDER_DISTANCE_CHUNKS = 12
MAX_RENDER_DISTANCE_CHUNKS = 31


@dataclass(frozen=True, slots=True)
class AtlasCapturePlan:
    render_distance_chunks: int
    spawn_position_xyz: tuple[float, float, float]
    tile_grid_xz: tuple[int, int]
    core_origin_chunk_xz: tuple[int, int]
    core_coverage_chunks_xz: tuple[int, int]
    capture_union_min_chunk_xz: tuple[int, int]
    capture_union_max_chunk_xz_exclusive: tuple[int, int]
    camera_safe_min_chunk_xz: tuple[int, int]
    camera_safe_max_chunk_xz_exclusive: tuple[int, int]
    tiles: tuple[dict[str, tuple[int, int]], ...]

    @property
    def view_diameter_chunks(self) -> int:
        return self.render_distance_chunks * 2 + 1

    @property
    def view_diameter_blocks(self) -> int:
        return self.view_diameter_chunks * CHUNK_SIZE

    @property
    def safe_camera_path_chunk_xz(self) -> tuple[tuple[int, int], ...]:
        minimum_x, minimum_z = self.camera_safe_min_chunk_xz
        maximum_x, maximum_z = self.camera_safe_max_chunk_xz_exclusive
        path: list[tuple[int, int]] = []
        for row, z in enumerate(range(minimum_z, maximum_z)):
            xs = list(range(minimum_x, maximum_x))
            path.extend((x, z) for x in (reversed(xs) if row % 2 else xs))
        return tuple(path)

    def value(self) -> dict[str, Any]:
        return {"render_distance_chunks": self.render_distance_chunks, "view_diameter_chunks": self.view_diameter_chunks, "view_diameter_blocks": self.view_diameter_blocks, "spawn_position_xyz": list(self.spawn_position_xyz), "tile_grid_xz": list(self.tile_grid_xz), "tiles": [{"tile_xz": list(tile["tile_xz"]), "core_min_chunk_xz": list(tile["core_min_chunk_xz"])} for tile in self.tiles], "camera_safe_path_chunk_xz": [list(item) for item in self.safe_camera_path_chunk_xz]}


def _position(value: Sequence[float]) -> tuple[float, float, float]:
    result = tuple(float(item) for item in value)
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        raise ValueError("spawn must contain three finite coordinates")
    return result


def plan_render_distance_atlas(spawn: Sequence[float], render_distance_chunks: int = DEFAULT_RENDER_DISTANCE_CHUNKS, *, core_min_chunk_x: int | None = None, core_min_chunk_z: int | None = None) -> AtlasCapturePlan:
    """Plan a haloed 3x3-core tile grid covering a full render distance."""
    position = _position(spawn)
    radius = int(render_distance_chunks)
    if not 1 <= radius <= MAX_RENDER_DISTANCE_CHUNKS:
        raise ValueError("render_distance_chunks is out of range")
    if (core_min_chunk_x is None) != (core_min_chunk_z is None):
        raise ValueError("both explicit core coordinates are required")
    diameter = 2 * radius + 1
    axis_tiles = math.ceil(diameter / CORE_CHUNKS_PER_AXIS)
    coverage = axis_tiles * CORE_CHUNKS_PER_AXIS
    if core_min_chunk_x is None:
        origin_x = math.floor((position[0] - coverage * CHUNK_SIZE / 2) / CHUNK_SIZE)
        origin_z = math.floor((position[2] - coverage * CHUNK_SIZE / 2) / CHUNK_SIZE)
    else:
        origin_x, origin_z = int(core_min_chunk_x), int(core_min_chunk_z)
    union_min = (origin_x - CAPTURE_HALO_CHUNKS, origin_z - CAPTURE_HALO_CHUNKS)
    union_max = (origin_x + coverage + CAPTURE_HALO_CHUNKS, origin_z + coverage + CAPTURE_HALO_CHUNKS)
    safe_min = (origin_x + radius, origin_z + radius)
    safe_max = (origin_x + coverage - radius, origin_z + coverage - radius)
    spawn_chunk = (math.floor(position[0] / CHUNK_SIZE), math.floor(position[2] / CHUNK_SIZE))
    if not (origin_x <= spawn_chunk[0] < origin_x + coverage and origin_z <= spawn_chunk[1] < origin_z + coverage):
        raise ValueError("explicit core grid does not contain spawn")
    tiles = tuple({"tile_xz": (x, z), "core_min_chunk_xz": (origin_x + 3 * x, origin_z + 3 * z)} for z in range(axis_tiles) for x in range(axis_tiles))
    return AtlasCapturePlan(radius, position, (axis_tiles, axis_tiles), (origin_x, origin_z), (coverage, coverage), union_min, union_max, safe_min, safe_max, tiles)


def composite_capture_plan(spawn: Sequence[float], width: int, depth: int, *, core_min_chunk_x: int | None = None, core_min_chunk_z: int | None = None) -> dict[str, Any]:
    """Return a rectangular chunk plan for arbitrary numeric world extents."""
    position = _position(spawn)
    width, depth = int(width), int(depth)
    if width < 1 or depth < 1:
        raise ValueError("width and depth must be positive")
    if (core_min_chunk_x is None) != (core_min_chunk_z is None):
        raise ValueError("both explicit core coordinates are required")
    tiles_x = math.ceil(width / (CORE_CHUNKS_PER_AXIS * CHUNK_SIZE))
    tiles_z = math.ceil(depth / (CORE_CHUNKS_PER_AXIS * CHUNK_SIZE))
    origin_x = math.floor(position[0] / CHUNK_SIZE) if core_min_chunk_x is None else int(core_min_chunk_x)
    origin_z = math.floor(position[2] / CHUNK_SIZE) if core_min_chunk_z is None else int(core_min_chunk_z)
    return {"tile_grid": (tiles_x, tiles_z), "core_origin_chunk_xz": (origin_x, origin_z), "tiles": [{"tile_xz": (x, z), "core_min_chunk_xz": (origin_x + 3 * x, origin_z + 3 * z)} for z in range(tiles_z) for x in range(tiles_x)], "spawn_position_xyz": position, "bounds_min": (origin_x * CHUNK_SIZE, 0.0, origin_z * CHUNK_SIZE), "bounds_max": ((origin_x + tiles_x * 3) * CHUNK_SIZE, 320.0, (origin_z + tiles_z * 3) * CHUNK_SIZE)}


__all__ = ["AtlasCapturePlan", "CHUNK_SIZE", "composite_capture_plan", "plan_render_distance_atlas"]
