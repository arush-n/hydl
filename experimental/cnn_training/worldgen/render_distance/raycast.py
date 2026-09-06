"""Generic voxel raycast compatibility path."""

from ..viewpoints.raycast import RaycastFrame, render_voxel_frame

RaycastWorldgenView = RaycastFrame
render_voxel_atlas_first_person = render_voxel_frame

__all__ = ["RaycastFrame", "RaycastWorldgenView", "render_voxel_frame", "render_voxel_atlas_first_person"]
