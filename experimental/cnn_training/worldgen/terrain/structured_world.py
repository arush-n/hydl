"""Topology-aware terrain variants used to diversify CNN training scenes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .random_world import RANDOM_WORLD_CHUNK_SIZE

STRUCTURED_WORLD_MODES = ("layered", "terraced", "rifted", "frontier")


def _classes(value: np.ndarray, name: str, minimum: int) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.int64)
    if result.ndim != 1 or result.size < minimum or np.unique(result).size != result.size or np.any(result < 0) or np.any(result > np.iinfo(np.int16).max):
        raise ValueError(f"{name} must contain {minimum}+ unique non-negative int16-safe labels")
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class StructuredTerrainPalette:
    """Opaque labels grouped by a terrain role, not by simulator semantics."""

    rock_class_indices: np.ndarray
    soil_class_indices: np.ndarray
    surface_class_indices: np.ndarray
    trunk_class_indices: np.ndarray
    foliage_class_indices: np.ndarray

    def __post_init__(self) -> None:
        names = ("rock", "soil", "surface", "trunk", "foliage")
        values = tuple(_classes(getattr(self, f"{name}_class_indices"), name, 1 if name in {"trunk", "foliage"} else 2) for name in names)
        merged = np.concatenate(values)
        if np.unique(merged).size != merged.size:
            raise ValueError("terrain role labels must be disjoint")
        for name, value in zip(names, values, strict=True):
            object.__setattr__(self, f"{name}_class_indices", value)

    @property
    def class_indices(self) -> np.ndarray:
        return np.concatenate((self.rock_class_indices, self.soil_class_indices, self.surface_class_indices, self.trunk_class_indices, self.foliage_class_indices)).astype(np.int64)

    @property
    def role_indices(self) -> np.ndarray:
        return np.concatenate(tuple(np.full(len(values), index, dtype=np.int8) for index, values in enumerate((self.rock_class_indices, self.soil_class_indices, self.surface_class_indices, self.trunk_class_indices, self.foliage_class_indices))))


@dataclass(frozen=True, slots=True)
class StructuredVoxelWorld:
    class_index_xyz: np.ndarray
    terrain_height_xz: np.ndarray
    palette: StructuredTerrainPalette
    camera_position_xyz: tuple[float, float, float]
    generation_mode: str
    render_distance_chunks: int
    seed: int

    def __post_init__(self) -> None:
        classes = np.asarray(self.class_index_xyz)
        heights = np.asarray(self.terrain_height_xz)
        side = (2 * self.render_distance_chunks + 1) * RANDOM_WORLD_CHUNK_SIZE
        if classes.dtype != np.int16 or classes.shape[0] != side or classes.shape[2] != side:
            raise ValueError("class_index_xyz has the wrong shape or dtype")
        if heights.dtype != np.int16 or heights.shape != (side, side) or np.any(heights < 1) or np.any(heights >= classes.shape[1]):
            raise ValueError("terrain_height_xz has the wrong shape or range")
        if not np.isin(classes[classes >= 0], self.palette.class_indices).all():
            raise ValueError("occupied labels must be in palette")
        camera = np.asarray(self.camera_position_xyz, dtype=np.float64)
        voxel = np.floor(camera).astype(np.int64)
        if camera.shape != (3,) or not np.isfinite(camera).all() or any(voxel[i] < 0 or voxel[i] >= classes.shape[i] for i in range(3)):
            raise ValueError("camera must be finite and inside the world")
        if classes[tuple(voxel)] >= 0:
            raise ValueError("camera starts inside a block")
        if self.generation_mode not in STRUCTURED_WORLD_MODES:
            raise ValueError("unsupported structured generation mode")
        classes.flags.writeable = False
        heights.flags.writeable = False


StructuredRandomVoxelWorld = StructuredVoxelWorld


def _smooth_noise(rng: np.random.Generator, side: int, scale: int) -> np.ndarray:
    coarse = rng.normal(0.0, 1.0, size=(math.ceil(side / scale), math.ceil(side / scale))).astype(np.float32)
    value = np.repeat(np.repeat(coarse, scale, 0), scale, 1)[:side, :side]
    for _ in range(4):
        value = (value + np.roll(value, 1, 0) + np.roll(value, -1, 0) + np.roll(value, 1, 1) + np.roll(value, -1, 1)) / 5.0
    return value


def _fill_layers(classes: np.ndarray, heights: np.ndarray, palette: StructuredTerrainPalette, rng: np.random.Generator) -> None:
    rock, soil, surface = (palette.rock_class_indices, palette.soil_class_indices, palette.surface_class_indices)
    for y in range(int(heights.max()) + 1):
        solid = heights >= y
        depth = heights - y
        choices = np.where(depth >= 7, rng.choice(rock, size=heights.shape), np.where(depth >= 3, rng.choice(soil, size=heights.shape), rng.choice(surface, size=heights.shape)))
        classes[:, y, :][solid] = choices[solid]


def _add_outcrops(classes: np.ndarray, heights: np.ndarray, palette: StructuredTerrainPalette, rng: np.random.Generator, rifted: bool) -> None:
    side, height, _ = classes.shape
    count = max(16, side * side // (5000 if rifted else 7000))
    for _ in range(count):
        x, z = (int(rng.integers(3, side - 3)) for _ in range(2))
        width, depth = (int(rng.integers(2, 7)) for _ in range(2))
        y0 = int(heights[x, z]) + int(rng.integers(1, 4))
        y1 = min(height, y0 + int(rng.integers(2, 10)))
        classes[x:x + width, y0:y1, z:z + depth] = rng.choice(palette.rock_class_indices)
        if rifted and _ % 3 == 0:
            classes[x:x + width, max(0, y0 - 2):y0, z:z + depth] = -1


def _add_groves(classes: np.ndarray, heights: np.ndarray, palette: StructuredTerrainPalette, rng: np.random.Generator) -> None:
    side = classes.shape[0]
    for _ in range(max(12, side * side // 12000)):
        x, z = (int(rng.integers(3, side - 3)) for _ in range(2))
        trunk = int(rng.choice(palette.trunk_class_indices))
        foliage = int(rng.choice(palette.foliage_class_indices))
        top = int(heights[x, z]) + int(rng.integers(4, 10))
        classes[x, int(heights[x, z]) + 1:min(classes.shape[1], top), z] = trunk
        radius = int(rng.integers(2, 4))
        y0, y1 = max(1, top - 2), min(classes.shape[1], top + 2)
        classes[max(0, x - radius):x + radius + 1, y0:y1, max(0, z - radius):z + radius + 1] = foliage


def generate_structured_voxel_world(
    palette: StructuredTerrainPalette,
    *,
    seed: int,
    generation_mode: str = "layered",
    render_distance_chunks: int = 12,
    height_blocks: int = 128,
) -> StructuredVoxelWorld:
    """Generate connected terrain with slopes, terraces, rifts, or a frontier."""
    if generation_mode not in STRUCTURED_WORLD_MODES:
        raise ValueError(f"generation_mode must be one of {STRUCTURED_WORLD_MODES}")
    radius, height = int(render_distance_chunks), int(height_blocks)
    if not 1 <= radius <= 12 or not 32 <= height <= 320:
        raise ValueError("render_distance_chunks or height_blocks is out of range")
    rng = np.random.default_rng(seed)
    side = (2 * radius + 1) * RANDOM_WORLD_CHUNK_SIZE
    axis = np.arange(side, dtype=np.float32)
    base = height * 0.30 + 5.0 * np.sin(axis[:, None] / 50.0) + 4.0 * np.cos(axis[None, :] / 43.0)
    noise = _smooth_noise(rng, side, 18) * 8.0
    if generation_mode == "terraced":
        terrain = np.floor((base + noise) / 5.0) * 5.0
    elif generation_mode == "rifted":
        rift = np.abs(np.sin(axis[:, None] / 19.0) + 0.6 * np.cos(axis[None, :] / 27.0))
        terrain = base + noise - np.where(rift < 0.12, 10.0, 0.0)
    elif generation_mode == "frontier":
        terrain = base + noise + np.where(axis[:, None] > side * 0.55, 10.0, -3.0)
    else:
        terrain = base + noise
    heights = np.clip(np.rint(terrain), 4, height - 8).astype(np.int16)
    classes = np.full((side, height, side), -1, dtype=np.int16)
    _fill_layers(classes, heights, palette, rng)
    if generation_mode in {"rifted", "frontier"}:
        _add_outcrops(classes, heights, palette, rng, generation_mode == "rifted")
    if generation_mode in {"layered", "frontier"}:
        _add_groves(classes, heights, palette, rng)
    center = side // 2
    ground = int(heights[center, center])
    classes[center - 3:center + 4, ground + 1:, center - 3:center + 4] = -1
    return StructuredVoxelWorld(classes, heights, palette, (center + 0.5, ground + 2.62, center + 0.5), generation_mode, radius, int(seed))


__all__ = ["STRUCTURED_WORLD_MODES", "StructuredRandomVoxelWorld", "StructuredTerrainPalette", "StructuredVoxelWorld", "generate_structured_voxel_world"]
