"""Host compiler for fixed-capacity native path imitation evidence."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.navigation_imitation import (
    NATIVE_PATH_IMITATION_MAX_ENTRIES,
    NATIVE_PATH_IMITATION_MAX_PATH_NODES,
    NativePathImitationAtlas,
)
from hytalegym.worldgen.native_navigation import (
    NATIVE_ASTAR_PROGRESS_ACCOMPLISHED,
    NativeNavigationPathProbeCapture,
)


def native_path_imitation_atlas_from_captures(
    captures: Sequence[NativeNavigationPathProbeCapture],
    world_ids: Sequence[int],
    *,
    entry_capacity: int | None = None,
    path_capacity: int | None = None,
) -> NativePathImitationAtlas:
    """Pack accomplished native paths canonically; never truncate."""

    values = tuple(captures)
    ids = tuple(world_ids)
    if not values or len(ids) != len(values):
        raise ValueError("captures and world_ids must be non-empty and agree")
    rows: list[tuple[object, ...]] = []
    for capture, world_id in zip(values, ids, strict=True):
        if isinstance(world_id, (bool, np.bool_)) or not isinstance(
            world_id,
            (int, np.integer),
        ):
            raise TypeError("world_ids must contain integers")
        if int(world_id) < 0:
            raise ValueError("world_ids must be non-negative")
        for sample in range(capture.sample_count):
            if (
                int(capture.progress[sample])
                != NATIVE_ASTAR_PROGRESS_ACCOMPLISHED
            ):
                continue
            count = int(capture.path_node_counts[sample])
            if count < 2:
                continue
            path = np.asarray(
                capture.path_positions[sample, :count],
                dtype=np.float64,
            )
            digest = hashlib.sha256(
                np.ascontiguousarray(path, dtype="<f8").tobytes()
            ).hexdigest()
            rows.append(
                (
                    int(world_id),
                    np.asarray(
                        capture.start_positions[sample],
                        dtype=np.float64,
                    ),
                    np.asarray(
                        capture.target_positions[sample],
                        dtype=np.float64,
                    ),
                    path,
                    np.asarray(capture.actor_bounds, dtype=np.float64),
                    np.asarray(
                        capture.direction_component_selector,
                        dtype=np.float64,
                    ),
                    digest,
                )
            )
    if not rows:
        raise ValueError("captures contain no accomplished paths")
    rows.sort(
        key=lambda row: (
            row[0],
            *np.asarray(row[1]).tolist(),
            *np.asarray(row[2]).tolist(),
            *np.asarray(row[4]).tolist(),
            *np.asarray(row[5]).tolist(),
            row[6],
        )
    )
    required_entries = len(rows)
    required_path = max(np.asarray(row[3]).shape[0] for row in rows)
    entries = _capacity(
        entry_capacity,
        required_entries,
        NATIVE_PATH_IMITATION_MAX_ENTRIES,
        "entry_capacity",
    )
    paths = _capacity(
        path_capacity,
        required_path,
        NATIVE_PATH_IMITATION_MAX_PATH_NODES,
        "path_capacity",
    )

    entry_mask = np.zeros(entries, dtype=np.bool_)
    world_id_array = np.full(entries, -1, dtype=np.int32)
    start = np.zeros((entries, 3), dtype=np.float32)
    target = np.zeros((entries, 3), dtype=np.float32)
    path_position = np.zeros((entries, paths, 3), dtype=np.float32)
    path_mask = np.zeros((entries, paths), dtype=np.bool_)
    bounds = np.zeros((entries, 6), dtype=np.float32)
    selector = np.zeros((entries, 3), dtype=np.float32)
    for slot, row in enumerate(rows):
        world, row_start, row_target, row_path, row_bounds, row_selector, _ = (
            row
        )
        count = np.asarray(row_path).shape[0]
        entry_mask[slot] = True
        world_id_array[slot] = int(world)
        start[slot] = row_start
        target[slot] = row_target
        path_position[slot, :count] = row_path
        path_mask[slot, :count] = True
        bounds[slot] = row_bounds
        selector[slot] = row_selector
    return NativePathImitationAtlas(
        entry_mask=jnp.asarray(entry_mask),
        world_id=jnp.asarray(world_id_array),
        start_position=jnp.asarray(start),
        target_position=jnp.asarray(target),
        path_position=jnp.asarray(path_position),
        path_mask=jnp.asarray(path_mask),
        actor_bounds=jnp.asarray(bounds),
        direction_component_selector=jnp.asarray(selector),
    )


def native_path_imitation_atlas_sha256(
    atlas: NativePathImitationAtlas,
) -> str:
    """Hash normalized fixed arrays for artifacts and checkpoints."""

    digest = hashlib.sha256()
    for name, value in atlas._asdict().items():
        array = np.ascontiguousarray(jax.device_get(value))
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _capacity(
    requested: int | None,
    required: int,
    maximum: int,
    name: str,
) -> int:
    if required < 1 or required > maximum:
        raise ValueError(f"{name} requires {required}, maximum is {maximum}")
    if requested is None:
        return required
    if isinstance(requested, (bool, np.bool_)) or not isinstance(
        requested,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be an integer")
    value = int(requested)
    if value < required or value > maximum:
        raise ValueError(f"{name} must be in [{required}, {maximum}]")
    return value


__all__ = [
    "NATIVE_PATH_IMITATION_MAX_ENTRIES",
    "NATIVE_PATH_IMITATION_MAX_PATH_NODES",
    "native_path_imitation_atlas_from_captures",
    "native_path_imitation_atlas_sha256",
]
