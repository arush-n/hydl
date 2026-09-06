"""Shared, simulator-agnostic CNN data-generation primitives.

The package deliberately accepts numeric voxel fields and caller-provided
appearance tables.  It has no asset catalog, semantic lookup, native bridge,
or simulator runtime dependency.
"""

from .worldgen import (
    CameraPose,
    RaycastFrame,
    StructuredTerrainPalette,
    StructuredVoxelWorld,
    VoxelWorld,
    build_block_tokens,
    generate_random_voxel_world,
    generate_structured_voxel_world,
    generate_viewpoints,
    plan_render_distance_atlas,
    render_voxel_frame,
)
from .pipeline import CnnSample, generate_batch, generate_sample
from .dataset import load_sample_npz, write_sample_npz

__all__ = [
    "CameraPose",
    "CnnSample",
    "RaycastFrame",
    "StructuredTerrainPalette",
    "StructuredVoxelWorld",
    "VoxelWorld",
    "build_block_tokens",
    "generate_random_voxel_world",
    "generate_structured_voxel_world",
    "generate_viewpoints",
    "plan_render_distance_atlas",
    "render_voxel_frame",
    "generate_batch",
    "generate_sample",
    "load_sample_npz",
    "write_sample_npz",
]
