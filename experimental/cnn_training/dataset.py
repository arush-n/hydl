"""Catalog-free NPZ persistence for generated CNN samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .pipeline import CnnSample


def write_sample_npz(sample: CnnSample, path: str | Path, *, compressed: bool = True) -> Path:
    """Write pixels, renderer targets, and token features without asset metadata."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "pixels_rgba_hwc": sample.frame.pixels_rgba_hwc,
        "class_index_hw": sample.frame.class_index_hw,
        "distance_hw": sample.frame.distance_hw,
        "face_index_hw": sample.frame.face_index_hw,
        "voxel_xyz_hw3": sample.frame.voxel_xyz_hw3,
        "surface_uv_hw2": sample.frame.surface_uv_hw2,
        "surface_albedo_rgb_hwc": sample.frame.surface_albedo_rgb_hwc,
        "light_level_hw": sample.frame.light_level_hw,
        "fog_factor_hw": sample.frame.fog_factor_hw,
        "dense_token_features_hwc": sample.tokens.dense_features_hwc,
        "visible_token_features_nf": sample.tokens.visible_features_nf,
        "visible_voxel_xyz_n3": sample.tokens.visible_voxel_xyz_n3,
        "visible_class_index_n": sample.tokens.visible_class_index_n,
        "visible_bbox_xyxy_n4": sample.tokens.visible_bbox_xyxy_n4,
        "camera_position_xyz": np.asarray(sample.pose.position_xyz, dtype=np.float32),
        "camera_yaw_pitch_fov": np.asarray((sample.pose.yaw_degrees, sample.pose.pitch_degrees, sample.pose.horizontal_fov_degrees), dtype=np.float32),
    }
    writer = np.savez_compressed if compressed else np.savez
    writer(destination, **arrays)
    return destination


def load_sample_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a generated sample as independent NumPy arrays."""
    with np.load(Path(path), allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


__all__ = ["load_sample_npz", "write_sample_npz"]
