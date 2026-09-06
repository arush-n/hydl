"""Derived numeric token channels for CNN supervision and routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

DENSE_FEATURE_NAMES = ("hit", "distance_linear_01", "distance_log_01", "visible_r_01", "visible_g_01", "visible_b_01", "albedo_r_01", "albedo_g_01", "albedo_b_01", "light_01", "fog_01", "normal_x", "normal_y", "normal_z", "surface_u", "surface_v")
VISIBLE_FEATURE_NAMES = ("nearest_distance_01", "nearest_distance_log_01", "mean_visible_r_01", "mean_visible_g_01", "mean_visible_b_01", "mean_albedo_r_01", "mean_albedo_g_01", "mean_albedo_b_01", "mean_light_01", "mean_fog_01", "center_x_01", "center_y_01", "width_01", "height_01", "pixel_fraction_01", "bbox_fill_01")
_NORMALS = np.asarray(((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)), dtype=np.float32)


@dataclass(frozen=True, slots=True)
class BlockTokenBatch:
    """Dense pixel tokens and sparse visible-voxel tokens from one frame."""

    dense_features_hwc: np.ndarray
    visible_features_nf: np.ndarray
    visible_voxel_xyz_n3: np.ndarray
    visible_class_index_n: np.ndarray
    visible_bbox_xyxy_n4: np.ndarray
    dense_feature_names: tuple[str, ...] = DENSE_FEATURE_NAMES
    visible_feature_names: tuple[str, ...] = VISIBLE_FEATURE_NAMES

    @property
    def visible_count(self) -> int:
        return int(self.visible_features_nf.shape[0])


def _means(values: np.ndarray, inverse: np.ndarray, count: int) -> np.ndarray:
    return np.stack([np.bincount(inverse, weights=values[:, index], minlength=count) for index in range(values.shape[1])], 1) / np.bincount(inverse, minlength=count)[:, None]


def build_block_tokens(view: Any) -> BlockTokenBatch:
    """Derive distances, colors, lighting, fog, and screen boxes without lookup."""
    hit = np.asarray(view.class_index_hw) >= 0
    height, width = hit.shape
    dense = np.zeros((height, width, len(DENSE_FEATURE_NAMES)), dtype=np.float32)
    if hit.any():
        distances = view.distance_hw[hit]
        maximum = float(view.maximum_distance_blocks)
        dense[hit, 0] = 1.0
        dense[hit, 1] = np.clip(distances / maximum, 0.0, 1.0)
        dense[hit, 2] = np.log1p(distances) / np.log1p(maximum)
        dense[hit, 3:6] = view.pixels_rgba_hwc[hit, :3] / 255.0
        dense[hit, 6:9] = view.surface_albedo_rgb_hwc[hit] / 255.0
        dense[hit, 9] = view.light_level_hw[hit]
        dense[hit, 10] = view.fog_factor_hw[hit]
        dense[hit, 11:14] = _NORMALS[view.face_index_hw[hit]]
        dense[hit, 14:16] = view.surface_uv_hw2[hit]
    rows, columns = np.nonzero(hit)
    if len(rows) == 0:
        empty = np.empty((0,), dtype=np.int32)
        return BlockTokenBatch(dense, np.empty((0, len(VISIBLE_FEATURE_NAMES)), dtype=np.float32), np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.int16), np.empty((0, 4), dtype=np.float32))
    voxels, inverse = np.unique(view.voxel_xyz_hw3[rows, columns], axis=0, return_inverse=True)
    count = len(voxels)
    counts = np.bincount(inverse, minlength=count).astype(np.float32)
    first = np.full(count, len(inverse), dtype=np.int64)
    np.minimum.at(first, inverse, np.arange(len(inverse)))
    labels = view.class_index_hw[rows, columns][first]
    distances = np.full(count, np.inf, dtype=np.float32)
    np.minimum.at(distances, inverse, view.distance_hw[rows, columns])
    rgb = _means(view.pixels_rgba_hwc[rows, columns, :3].astype(np.float32) / 255.0, inverse, count)
    albedo = _means(view.surface_albedo_rgb_hwc[rows, columns].astype(np.float32) / 255.0, inverse, count)
    light = np.bincount(inverse, weights=view.light_level_hw[rows, columns], minlength=count) / counts
    fog = np.bincount(inverse, weights=view.fog_factor_hw[rows, columns], minlength=count) / counts
    min_x = np.full(count, width, dtype=np.int32); min_y = np.full(count, height, dtype=np.int32); max_x = np.full(count, -1, dtype=np.int32); max_y = np.full(count, -1, dtype=np.int32)
    np.minimum.at(min_x, inverse, columns); np.minimum.at(min_y, inverse, rows); np.maximum.at(max_x, inverse, columns); np.maximum.at(max_y, inverse, rows)
    boxes = np.stack((min_x / width, min_y / height, (max_x + 1) / width, (max_y + 1) / height), 1).astype(np.float32)
    visible_fraction = counts / np.float32(height * width)
    fill_fraction = counts / ((max_x - min_x + 1) * (max_y - min_y + 1))
    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    features = np.column_stack((np.clip(distances / view.maximum_distance_blocks, 0.0, 1.0), np.log1p(distances) / np.log1p(view.maximum_distance_blocks), rgb, albedo, light, fog, centers, boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1], visible_fraction, fill_fraction)).astype(np.float32)
    return BlockTokenBatch(dense, features, voxels.astype(np.int32), labels.astype(np.int16), boxes)


def build_dense_features_jax(*, pixels_rgba_hwc: Any, distance_hw: Any, face_index_hw: Any, surface_uv_hw2: Any, surface_albedo_rgb_hwc: Any, light_level_hw: Any, fog_factor_hw: Any, maximum_distance_blocks: float) -> Any:
    """JAX equivalent of dense tokens for device-resident CNN pipelines."""
    try:
        import jax.numpy as jnp
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("JAX is required for build_dense_features_jax") from error
    pixels, distances, faces = jnp.asarray(pixels_rgba_hwc), jnp.asarray(distance_hw, dtype=jnp.float32), jnp.asarray(face_index_hw)
    maximum = jnp.asarray(maximum_distance_blocks, dtype=jnp.float32)
    hit = faces >= 0
    safe_faces = jnp.clip(faces, 0, 5)
    features = jnp.concatenate((hit[..., None].astype(jnp.float32), jnp.clip(distances / maximum, 0, 1)[..., None], (jnp.log1p(jnp.where(hit, distances, 0)) / jnp.log1p(maximum))[..., None], pixels[..., :3].astype(jnp.float32) / 255.0, jnp.asarray(surface_albedo_rgb_hwc).astype(jnp.float32) / 255.0, jnp.asarray(light_level_hw)[..., None], jnp.asarray(fog_factor_hw)[..., None], jnp.take(jnp.asarray(_NORMALS), safe_faces, axis=0), jnp.asarray(surface_uv_hw2, dtype=jnp.float32)), -1)
    return jnp.where(hit[..., None], features, 0.0)


__all__ = ["BlockTokenBatch", "DENSE_FEATURE_NAMES", "VISIBLE_FEATURE_NAMES", "build_block_tokens", "build_dense_features_jax"]
