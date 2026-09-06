"""Deterministic first-person camera schedules."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class CameraPose:
    """A perspective pose; the schedule never changes FOV to fake zoom."""

    position_xyz: tuple[float, float, float]
    yaw_degrees: float
    pitch_degrees: float = -6.0
    horizontal_fov_degrees: float = 78.0

    def __post_init__(self) -> None:
        if len(self.position_xyz) != 3 or not np.isfinite(self.position_xyz).all():
            raise ValueError("position_xyz must contain three finite values")
        if not math.isfinite(self.yaw_degrees) or not -89.0 <= self.pitch_degrees <= 89.0:
            raise ValueError("yaw/pitch are invalid")
        if not 30.0 <= self.horizontal_fov_degrees <= 120.0:
            raise ValueError("horizontal_fov_degrees must lie in [30,120]")


def generate_viewpoints(
    origin_xyz: Sequence[float],
    *,
    count: int = 16,
    seed: int = 0,
    radius_blocks: float = 0.0,
    yaw_degrees: float | None = None,
    pitch_degrees: float = -6.0,
    horizontal_fov_degrees: float = 78.0,
) -> tuple[CameraPose, ...]:
    """Create cardinal, oblique, and small-jitter first-person views.

    ``radius_blocks`` is an optional walk-around radius.  It is kept separate
    from FOV so the target is not permanently magnified.
    """
    if isinstance(count, bool) or not 1 <= int(count) <= 4096:
        raise ValueError("count must lie in [1,4096]")
    origin = np.asarray(origin_xyz, dtype=np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("origin_xyz must contain three finite values")
    if radius_blocks < 0.0 or not math.isfinite(radius_blocks):
        raise ValueError("radius_blocks must be finite and non-negative")
    rng = np.random.default_rng(seed)
    poses: list[CameraPose] = []
    base_yaws = np.asarray((0.0, 90.0, 180.0, 270.0), dtype=np.float64)
    for index in range(int(count)):
        angle = base_yaws[index % 4] if yaw_degrees is None else float(yaw_degrees)
        angle += float(rng.normal(0.0, 4.0 if index >= 4 else 0.0))
        distance = float(radius_blocks) if radius_blocks else 0.0
        radians = math.radians(angle)
        position = origin + np.asarray((math.sin(radians) * distance, 0.0, math.cos(radians) * distance))
        position += rng.normal(0.0, 0.03, size=3) if index >= 4 else 0.0
        pitch = float(pitch_degrees) + float(rng.normal(0.0, 2.0 if index >= 4 else 0.0))
        poses.append(CameraPose(tuple(float(value) for value in position), angle % 360.0, pitch, float(horizontal_fov_degrees)))
    return tuple(poses)


__all__ = ["CameraPose", "generate_viewpoints"]
