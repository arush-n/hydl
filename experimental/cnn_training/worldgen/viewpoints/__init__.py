"""First-person viewpoints, raycast frames, and derived CNN targets."""

from .camera import CameraPose, generate_viewpoints
from .plan import AtlasCapturePlan, plan_render_distance_atlas
from .raycast import RaycastFrame, render_voxel_frame
from .tokens import BlockTokenBatch, build_block_tokens

__all__ = [
    "AtlasCapturePlan",
    "BlockTokenBatch",
    "CameraPose",
    "RaycastFrame",
    "build_block_tokens",
    "generate_viewpoints",
    "plan_render_distance_atlas",
    "render_voxel_frame",
]
