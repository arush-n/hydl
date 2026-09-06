"""Terrain, camera, raycast, and token tools for generic CNN corpora."""

from .terrain import (
    RANDOM_WORLD_MODES,
    StructuredTerrainPalette,
    StructuredVoxelWorld,
    VoxelWorld,
    generate_random_voxel_world,
    generate_structured_voxel_world,
)
from .viewpoints import (
    AtlasCapturePlan,
    CameraPose,
    RaycastFrame,
    build_block_tokens,
    generate_viewpoints,
    plan_render_distance_atlas,
    render_voxel_frame,
)

__all__ = [
    "AtlasCapturePlan",
    "CameraPose",
    "RANDOM_WORLD_MODES",
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
]
