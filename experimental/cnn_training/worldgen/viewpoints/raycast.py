"""Vectorized first-person voxel rendering with an optional CUDA device."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class RaycastFrame:
    """Rendered pixels plus numeric per-pixel supervision from the ray marcher."""

    pixels_rgba_hwc: np.ndarray
    class_index_hw: np.ndarray
    distance_hw: np.ndarray
    face_index_hw: np.ndarray
    voxel_xyz_hw3: np.ndarray
    surface_uv_hw2: np.ndarray
    surface_albedo_rgb_hwc: np.ndarray
    light_level_hw: np.ndarray
    fog_factor_hw: np.ndarray
    camera_position_xyz: tuple[float, float, float]
    yaw_degrees: float
    pitch_degrees: float
    step_blocks: float
    maximum_distance_blocks: float

    def __post_init__(self) -> None:
        pixels = np.asarray(self.pixels_rgba_hwc)
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 4:
            raise ValueError("pixels_rgba_hwc must be uint8 [h,w,4]")
        shape = pixels.shape[:2]
        contracts = ((self.class_index_hw, np.int16, shape), (self.distance_hw, np.float32, shape), (self.face_index_hw, np.int8, shape), (self.voxel_xyz_hw3, np.int32, (*shape, 3)), (self.surface_uv_hw2, np.float32, (*shape, 2)), (self.surface_albedo_rgb_hwc, np.uint8, (*shape, 3)), (self.light_level_hw, np.float32, shape), (self.fog_factor_hw, np.float32, shape))
        for value, dtype, expected in contracts:
            if np.asarray(value).dtype != dtype or np.asarray(value).shape != expected:
                raise ValueError("raycast supervision has the wrong dtype or shape")
        hit = np.asarray(self.class_index_hw) >= 0
        if np.any(np.isfinite(self.distance_hw[~hit])) or np.any(self.face_index_hw[~hit] != -1) or np.any(self.voxel_xyz_hw3[~hit] != -1) or np.any(np.isfinite(self.surface_uv_hw2[~hit])):
            raise ValueError("miss pixels do not follow the empty-surface contract")
        for value in (pixels, self.class_index_hw, self.distance_hw, self.face_index_hw, self.voxel_xyz_hw3, self.surface_uv_hw2, self.surface_albedo_rgb_hwc, self.light_level_hw, self.fog_factor_hw):
            np.asarray(value).flags.writeable = False

    def block_tokens(self):
        from .tokens import build_block_tokens
        return build_block_tokens(self)


def _rays(side: int, yaw: float, pitch: float, fov: float) -> np.ndarray:
    yaw, pitch = math.radians(yaw), math.radians(pitch)
    forward = np.asarray((math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)), dtype=np.float32)
    forward /= np.linalg.norm(forward)
    right = np.cross(np.asarray((0.0, 1.0, 0.0), dtype=np.float32), forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    grid = (np.arange(side, dtype=np.float32) + 0.5) / side * 2.0 - 1.0
    horizontal, vertical = np.meshgrid(grid, -grid)
    extent = math.tan(math.radians(fov) / 2.0)
    result = forward + horizontal[..., None] * extent * right + vertical[..., None] * extent * up
    result /= np.linalg.norm(result, axis=-1, keepdims=True)
    return result.reshape(-1, 3).astype(np.float32)


def _entry(camera: np.ndarray, rays: np.ndarray, voxels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive = rays > 0.0
    boundary = np.where(positive, voxels, voxels + 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        entry = (boundary - camera) / rays
    entry[~np.isfinite(entry)] = -np.inf
    axes = np.argmax(entry, axis=1)
    directions = rays[np.arange(len(rays)), axes]
    faces = (axes * 2 + (directions < 0.0)).astype(np.int8)
    distances = entry[np.arange(len(entry)), axes].astype(np.float32)
    points = camera + rays * distances[:, None]
    fraction = points - np.floor(points)
    uv = np.empty((len(rays), 2), dtype=np.float32)
    uv[axes == 0] = fraction[axes == 0][:, (2, 1)] * np.asarray((1.0, -1.0)) + np.asarray((0.0, 1.0))
    uv[axes == 1] = fraction[axes == 1][:, (0, 2)]
    uv[axes == 2] = fraction[axes == 2][:, (0, 1)] * np.asarray((1.0, -1.0)) + np.asarray((0.0, 1.0))
    return faces, np.mod(uv, 1.0), distances


def render_voxel_frame(
    class_index_xyz: np.ndarray,
    face_colors_rgb: np.ndarray,
    *,
    camera_position_xyz: Sequence[float],
    yaw_degrees: float,
    pitch_degrees: float = -4.0,
    side: int = 128,
    horizontal_fov_degrees: float = 78.0,
    maximum_distance_blocks: float = 256.0,
    step_blocks: float = 0.75,
    face_textures_rgba: np.ndarray | None = None,
    background_rgba: tuple[int, int, int, int] = (119, 169, 210, 255),
    device: str | object | None = None,
) -> RaycastFrame:
    """Ray-march a numeric field; use ``device='cuda'`` for GPU batches."""
    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("render_voxel_frame requires PyTorch") from error
    classes, colors = np.asarray(class_index_xyz), np.asarray(face_colors_rgb)
    if classes.dtype != np.int16 or classes.ndim != 3 or colors.dtype != np.uint8 or colors.ndim != 3 or colors.shape[1:] != (6, 3):
        raise ValueError("class_index_xyz must be int16 [x,y,z] and colors uint8 [classes,6,3]")
    textures = None if face_textures_rgba is None else np.asarray(face_textures_rgba)
    if textures is not None and (textures.dtype != np.uint8 or textures.ndim != 5 or textures.shape[:2] != (colors.shape[0], 6) or textures.shape[2] != textures.shape[3] or textures.shape[-1] != 4):
        raise ValueError("face_textures_rgba must be uint8 [classes,6,size,size,4]")
    if np.any(classes >= len(colors)) or np.any(classes < -1):
        raise ValueError("voxel labels must index face_colors_rgb")
    side = int(side)
    if not math.isfinite(float(yaw_degrees)) or not math.isfinite(float(pitch_degrees)) or not -89.0 <= float(pitch_degrees) <= 89.0:
        raise ValueError("yaw and pitch must be finite; pitch must lie in [-89,89]")
    if not 16 <= side <= 1024 or not 30.0 <= horizontal_fov_degrees <= 120.0 or not 1.0 <= maximum_distance_blocks <= 2048.0 or not 0.25 <= step_blocks <= 1.0:
        raise ValueError("invalid render parameters")
    if len(background_rgba) != 4 or any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= int(value) <= 255 for value in background_rgba):
        raise ValueError("background_rgba must contain four uint8 values")
    camera = np.asarray(camera_position_xyz, dtype=np.float32)
    if camera.shape != (3,) or not np.isfinite(camera).all() or any(not 0 <= camera[i] < classes.shape[i] for i in range(3)):
        raise ValueError("camera_position_xyz must be finite and inside the field")
    backend = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    rays_np = _rays(side, yaw_degrees, pitch_degrees, horizontal_fov_degrees)
    rays = torch.from_numpy(rays_np).to(backend)
    # Terrain objects are intentionally immutable; give PyTorch a writable
    # staging view so ``from_numpy`` cannot expose a read-only buffer.
    volume = torch.from_numpy(np.array(classes.reshape(-1), copy=True)).to(backend)
    cam = torch.from_numpy(camera).to(backend)
    count = side * side
    hit = torch.full((count,), -1, dtype=torch.int16, device=backend)
    hit_voxels = torch.full((count, 3), -1, dtype=torch.int32, device=backend)
    unresolved = torch.ones(count, dtype=torch.bool, device=backend)
    steps = int(math.ceil(maximum_distance_blocks / step_blocks)) + 1
    shape = classes.shape
    for step in range(steps):
        distance = float(step * step_blocks)
        voxels = torch.floor(cam + rays * distance).to(torch.int64)
        inside = (voxels >= 0).all(1) & (voxels[:, 0] < shape[0]) & (voxels[:, 1] < shape[1]) & (voxels[:, 2] < shape[2])
        safe = torch.stack(tuple(voxels[:, axis].clamp(0, shape[axis] - 1) for axis in range(3)), 1)
        flat = (safe[:, 0] * shape[1] + safe[:, 1]) * shape[2] + safe[:, 2]
        sampled = volume[flat]
        solid = unresolved & inside & (sampled >= 0)
        hit = torch.where(solid, sampled, hit)
        hit_voxels = torch.where(solid[:, None], safe.to(torch.int32), hit_voxels)
        unresolved &= ~solid
    classes_hw = hit.reshape(side, side).cpu().numpy()
    voxels_np = hit_voxels.cpu().numpy()
    hit_flat = classes_hw.reshape(-1) >= 0
    distances = np.full(count, np.nan, dtype=np.float32)
    faces = np.full(count, -1, dtype=np.int8)
    uv = np.full((count, 2), np.nan, dtype=np.float32)
    if hit_flat.any():
        faces[hit_flat], uv[hit_flat], distances[hit_flat] = _entry(camera, rays_np[hit_flat], voxels_np[hit_flat].astype(np.float32))
    canvas = np.empty((count, 4), dtype=np.uint8)
    canvas[:] = np.asarray(background_rgba, dtype=np.uint8)
    albedo = np.zeros((count, 3), dtype=np.uint8)
    light = np.zeros(count, dtype=np.float32)
    fog = np.zeros(count, dtype=np.float32)
    if hit_flat.any():
        labels, face_values = classes_hw.reshape(-1)[hit_flat].astype(np.int64), faces[hit_flat].astype(np.int64)
        rgb = colors[labels, face_values].astype(np.float32)
        if textures is not None:
            texture_side = textures.shape[2]
            tx = np.minimum((uv[hit_flat, 0] * texture_side).astype(np.int64), texture_side - 1)
            ty = np.minimum((uv[hit_flat, 1] * texture_side).astype(np.int64), texture_side - 1)
            texels = textures[labels, face_values, ty, tx].astype(np.float32)
            alpha = texels[:, 3:4] / 255.0
            rgb = texels[:, :3] * alpha + rgb * (1.0 - alpha)
        albedo[hit_flat] = np.clip(rgb, 0, 255).astype(np.uint8)
        shade_table = np.asarray((0.82, 0.96, 0.68, 1.0, 0.88, 0.92), dtype=np.float32)
        light[hit_flat] = shade_table[face_values]
        rgb *= light[hit_flat, None]
        fog[hit_flat] = np.clip(distances[hit_flat] / maximum_distance_blocks, 0.0, 1.0) ** 1.35
        rgb = rgb * (1.0 - 0.68 * fog[hit_flat, None]) + np.asarray(background_rgba[:3], dtype=np.float32) * (0.68 * fog[hit_flat, None])
        canvas[hit_flat, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        canvas[hit_flat, 3] = 255
    return RaycastFrame(canvas.reshape(side, side, 4), classes_hw, distances.reshape(side, side), faces.reshape(side, side), voxels_np.reshape(side, side, 3), uv.reshape(side, side, 2), albedo.reshape(side, side, 3), light.reshape(side, side), fog.reshape(side, side), tuple(float(value) for value in camera), float(yaw_degrees), float(pitch_degrees), float(step_blocks), float(maximum_distance_blocks))


__all__ = ["RaycastFrame", "render_voxel_frame"]
