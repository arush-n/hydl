"""Small composition layer for producing reusable CNN training samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .worldgen.terrain import VoxelWorld, generate_random_voxel_world
from .worldgen.viewpoints import BlockTokenBatch, CameraPose, build_block_tokens, render_voxel_frame


@dataclass(frozen=True, slots=True)
class CnnSample:
    """One image and its optional teacher labels/tokens."""

    frame: object
    tokens: BlockTokenBatch
    world: VoxelWorld
    pose: CameraPose


def generate_sample(
    eligible_class_indices: np.ndarray,
    face_colors_rgb: np.ndarray,
    *,
    seed: int,
    pose: CameraPose | None = None,
    render_distance_chunks: int = 1,
    height_blocks: int = 64,
    generation_mode: str = "mixed",
    side: int = 128,
    device: str | object | None = None,
) -> CnnSample:
    """Generate terrain, choose a first-person pose, render, and tokenize."""
    world = generate_random_voxel_world(eligible_class_indices, seed=seed, generation_mode=generation_mode, render_distance_chunks=render_distance_chunks, height_blocks=height_blocks, palette_size=min(48, len(eligible_class_indices)))
    selected = pose or CameraPose(world.camera_position_xyz, yaw_degrees=float(seed % 360))
    frame = render_voxel_frame(world.class_index_xyz, face_colors_rgb, camera_position_xyz=selected.position_xyz, yaw_degrees=selected.yaw_degrees, pitch_degrees=selected.pitch_degrees, horizontal_fov_degrees=selected.horizontal_fov_degrees, side=side, maximum_distance_blocks=float((2 * render_distance_chunks + 1) * 32), device=device)
    return CnnSample(frame, build_block_tokens(frame), world, selected)


def generate_batch(
    eligible_class_indices: np.ndarray,
    face_colors_rgb: np.ndarray,
    *,
    seeds: Sequence[int],
    **kwargs: object,
) -> tuple[CnnSample, ...]:
    """Deterministically generate independent samples (easy to shard by seed)."""
    return tuple(generate_sample(eligible_class_indices, face_colors_rgb, seed=int(seed), **kwargs) for seed in seeds)


__all__ = ["CnnSample", "generate_batch", "generate_sample"]
