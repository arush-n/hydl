"""Deterministic numeric voxel terrain for CNN data generation.

The generator produces only tensors and a palette of integer labels.  Labels
are intentionally opaque: a caller may map them to colors or training classes
without this package knowing anything about a simulator or asset catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

RANDOM_WORLD_CHUNK_SIZE = 32
RANDOM_WORLD_MODES = ("patchwork", "clutter", "chaotic", "mixed")


@dataclass(frozen=True, slots=True)
class VoxelWorld:
    """A deterministic ``[x, y, z]`` field; ``-1`` denotes empty space."""

    class_index_xyz: np.ndarray
    palette_class_indices: np.ndarray
    camera_position_xyz: tuple[float, float, float]
    generation_mode: str
    render_distance_chunks: int
    seed: int

    def __post_init__(self) -> None:
        classes = np.asarray(self.class_index_xyz)
        palette = np.asarray(self.palette_class_indices)
        side = (2 * self.render_distance_chunks + 1) * RANDOM_WORLD_CHUNK_SIZE
        if classes.dtype != np.int16 or classes.ndim != 3 or classes.shape[0] != side or classes.shape[2] != side:
            raise ValueError("class_index_xyz must be int16 [side,height,side]")
        if palette.dtype != np.int64 or palette.ndim != 1 or palette.size < 4 or np.unique(palette).size != palette.size:
            raise ValueError("palette_class_indices must be unique int64 [P]")
        occupied = classes[classes >= 0]
        if occupied.size == 0 or not np.isin(occupied, palette).all():
            raise ValueError("occupied classes must be in palette_class_indices")
        camera = np.asarray(self.camera_position_xyz, dtype=np.float64)
        if camera.shape != (3,) or not np.isfinite(camera).all():
            raise ValueError("camera_position_xyz must contain three finite values")
        voxel = np.floor(camera).astype(np.int64)
        if any(voxel[i] < 0 or voxel[i] >= classes.shape[i] for i in range(3)):
            raise ValueError("camera lies outside the world")
        if classes[tuple(voxel)] >= 0:
            raise ValueError("camera starts inside a block")
        if self.generation_mode not in RANDOM_WORLD_MODES:
            raise ValueError("unsupported generation_mode")
        classes.flags.writeable = False
        palette.flags.writeable = False

    @property
    def present_class_indices(self) -> np.ndarray:
        return np.unique(self.class_index_xyz[self.class_index_xyz >= 0]).astype(np.int64)


RandomVoxelWorld = VoxelWorld


def _patch_map(rng: np.random.Generator, palette: np.ndarray, side: int, patch: int) -> np.ndarray:
    coarse = rng.choice(palette, size=(math.ceil(side / patch), math.ceil(side / patch)))
    return np.repeat(np.repeat(coarse, patch, axis=0), patch, axis=1)[:side, :side].astype(np.int16)


def _height_field(rng: np.random.Generator, side: int, height: int) -> np.ndarray:
    noise = _patch_map(rng, np.arange(-5, 6, dtype=np.int16), side, 16).astype(np.float32)
    for _ in range(3):
        noise = (noise + np.roll(noise, 1, 0) + np.roll(noise, -1, 0) + np.roll(noise, 1, 1) + np.roll(noise, -1, 1)) / 5.0
    axis = np.arange(side, dtype=np.float32)
    waves = 2.5 * np.sin(axis[:, None] / 27.0) + 2.0 * np.cos(axis[None, :] / 31.0)
    base = max(8, height // 4)
    return np.clip(np.rint(base + noise + waves), 5, max(6, height // 2)).astype(np.int16)


def _fill(classes: np.ndarray, heights: np.ndarray, surface: np.ndarray, foundation: np.ndarray) -> None:
    for y in range(int(heights.max()) + 1):
        solid = heights >= y
        classes[:, y, :][solid] = np.where(heights[solid] - y <= 2, surface[solid], foundation[solid])


def _structures(classes: np.ndarray, heights: np.ndarray, palette: np.ndarray, rng: np.random.Generator, chaotic: bool) -> None:
    side, height, _ = classes.shape
    count = max(32, side * side // (3072 if chaotic else 4096))
    center = side // 2
    for index in range(count):
        x, z = (int(rng.integers(2, side - 2)) for _ in range(2))
        if abs(x - center) < 8 and abs(z - center) < 8:
            continue
        width = int(rng.integers(1, 9 if chaotic else 7))
        depth = int(rng.integers(1, 9 if chaotic else 7))
        y0 = int(heights[x, z]) + int(rng.integers(1, 4 if chaotic else 2))
        y1 = min(height, y0 + int(rng.integers(2, 18 if chaotic else 10)))
        classes[x:min(side, x + width), y0:y1, z:min(side, z + depth)] = palette[int(rng.integers(len(palette)))]
        if chaotic and index % 7 == 0:
            classes[x, int(heights[x, z]) + 1:y0, z] = palette[int(rng.integers(len(palette)))]


def generate_random_voxel_world(
    eligible_class_indices: np.ndarray,
    *,
    seed: int,
    generation_mode: str = "mixed",
    render_distance_chunks: int = 12,
    height_blocks: int = 128,
    palette_size: int = 48,
) -> VoxelWorld:
    """Generate coherent ground plus optional atypical structures."""
    eligible = np.asarray(eligible_class_indices)
    if eligible.ndim != 1 or not np.issubdtype(eligible.dtype, np.integer) or eligible.size < 4 or np.unique(eligible).size != eligible.size or np.any(eligible < 0) or np.any(eligible > np.iinfo(np.int16).max):
        raise ValueError("eligible_class_indices must be unique non-negative int16-safe values")
    if generation_mode not in RANDOM_WORLD_MODES:
        raise ValueError(f"generation_mode must be one of {RANDOM_WORLD_MODES}")
    if not 1 <= int(render_distance_chunks) <= 12 or not 32 <= int(height_blocks) <= 320:
        raise ValueError("render_distance_chunks or height_blocks is out of range")
    if not 4 <= int(palette_size) <= min(256, len(eligible)):
        raise ValueError("palette_size is invalid")
    rng = np.random.default_rng(seed)
    palette = np.ascontiguousarray(rng.choice(eligible, size=palette_size, replace=False), dtype=np.int64)
    side = (2 * int(render_distance_chunks) + 1) * RANDOM_WORLD_CHUNK_SIZE
    classes = np.full((side, int(height_blocks), side), -1, dtype=np.int16)
    heights = _height_field(rng, side, int(height_blocks))
    _fill(classes, heights, _patch_map(rng, palette, side, 8), _patch_map(rng, palette, side, 32))
    if generation_mode in {"clutter", "mixed"}:
        _structures(classes, heights, palette, rng, False)
    if generation_mode in {"chaotic", "mixed"}:
        _structures(classes, heights, palette, rng, True)
    center = side // 2
    ground = int(heights[center, center])
    classes[center - 3:center + 4, ground + 1:, center - 3:center + 4] = -1
    return VoxelWorld(classes, palette, (center + 0.5, ground + 2.62, center + 0.5), generation_mode, int(render_distance_chunks), int(seed))


__all__ = ["RANDOM_WORLD_CHUNK_SIZE", "RANDOM_WORLD_MODES", "RandomVoxelWorld", "VoxelWorld", "generate_random_voxel_world"]
