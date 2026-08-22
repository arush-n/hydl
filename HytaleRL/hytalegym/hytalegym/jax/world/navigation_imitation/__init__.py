"""Tolerance-bounded replay of captured native NPC navigation paths."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp


Array = jax.Array

NATIVE_PATH_IMITATION_SCHEMA = "hytalerl_native_path_imitation_v1"
NATIVE_PATH_IMITATION_VERSION = 1
NATIVE_PATH_IMITATION_MAX_ENTRIES = 4096
NATIVE_PATH_IMITATION_MAX_PATH_NODES = 200


class NativePathImitationAtlas(NamedTuple):
    """Fixed-capacity native-path examples prepared outside JIT."""

    entry_mask: Array
    world_id: Array
    start_position: Array
    target_position: Array
    path_position: Array
    path_mask: Array
    actor_bounds: Array
    direction_component_selector: Array


class NativePathImitationResult(NamedTuple):
    """Nearest captured path inside explicit start/target tolerances."""

    imitation_available: Array
    certified_navigation_step_available: Array
    path_position: Array
    path_mask: Array
    selected_slot: Array
    start_error: Array
    target_error: Array
    exact_query: Array
    ambiguous: Array
    configuration_mismatch: Array
    out_of_tolerance: Array
    invalid: Array


def native_path_imitation_result(
    atlas: NativePathImitationAtlas,
    world_id: Array,
    start_position: Array,
    target_position: Array,
    actor_bounds: Array,
    direction_component_selector: Array,
    *,
    start_tolerance: Array | float,
    target_tolerance: Array | float,
) -> NativePathImitationResult:
    """Select a captured native path without claiming live equivalence."""

    start = jnp.asarray(start_position, dtype=jnp.float32)
    target = jnp.asarray(target_position, dtype=jnp.float32)
    if start.ndim != 2 or start.shape[0] == 0 or start.shape[1] != 3:
        raise ValueError("start_position must have non-empty shape [batch,3]")
    if target.shape != start.shape:
        raise ValueError("target_position must match start_position")
    batch = start.shape[0]
    worlds = _batch_int(world_id, batch, "world_id")
    start_limit = _batch_float(
        start_tolerance,
        batch,
        "start_tolerance",
    )
    target_limit = _batch_float(
        target_tolerance,
        batch,
        "target_tolerance",
    )
    query_bounds = _batch_vector(actor_bounds, batch, 6, "actor_bounds")
    query_selector = _batch_vector(
        direction_component_selector,
        batch,
        3,
        "direction_component_selector",
    )
    capacity, path_capacity = _validate_atlas_shape(atlas)

    entry_mask = jnp.asarray(atlas.entry_mask, dtype=jnp.bool_)
    entry_world = jnp.asarray(atlas.world_id, dtype=jnp.int32)
    entry_start = jnp.asarray(atlas.start_position, dtype=jnp.float32)
    entry_target = jnp.asarray(atlas.target_position, dtype=jnp.float32)
    paths = jnp.asarray(atlas.path_position, dtype=jnp.float32)
    path_mask = jnp.asarray(atlas.path_mask, dtype=jnp.bool_)
    bounds = jnp.asarray(atlas.actor_bounds, dtype=jnp.float32)
    selector = jnp.asarray(
        atlas.direction_component_selector,
        dtype=jnp.float32,
    )

    counts = jnp.sum(path_mask, axis=1, dtype=jnp.int32)
    contiguous = jnp.all(
        path_mask
        == (
            jnp.arange(path_capacity, dtype=jnp.int32)[None, :]
            < counts[:, None]
        ),
        axis=1,
    )
    entry_valid = (
        entry_mask
        & (entry_world >= 0)
        & jnp.all(jnp.isfinite(entry_start), axis=1)
        & jnp.all(jnp.isfinite(entry_target), axis=1)
        & jnp.all(jnp.isfinite(bounds), axis=1)
        & jnp.all(bounds[:, :3] < bounds[:, 3:], axis=1)
        & jnp.all(jnp.isfinite(selector), axis=1)
        & jnp.all(selector >= 0.0, axis=1)
        & jnp.any(selector > 0.0, axis=1)
        & jnp.all(~path_mask | jnp.all(jnp.isfinite(paths), axis=2), axis=1)
        & contiguous
        & (counts >= 2)
    )
    input_valid = (
        jnp.all(jnp.isfinite(start), axis=1)
        & jnp.all(jnp.isfinite(target), axis=1)
        & jnp.isfinite(start_limit)
        & jnp.isfinite(target_limit)
        & (start_limit >= 0.0)
        & (target_limit >= 0.0)
        & (worlds >= 0)
        & jnp.all(jnp.isfinite(query_bounds), axis=1)
        & jnp.all(query_bounds[:, :3] < query_bounds[:, 3:], axis=1)
        & jnp.all(jnp.isfinite(query_selector), axis=1)
        & jnp.all(query_selector >= 0.0, axis=1)
        & jnp.any(query_selector > 0.0, axis=1)
    )
    start_error = jnp.linalg.norm(
        start[:, None, :] - entry_start[None, :, :],
        axis=2,
    )
    target_error = jnp.linalg.norm(
        target[:, None, :] - entry_target[None, :, :],
        axis=2,
    )
    position_candidate = (
        input_valid[:, None]
        & entry_valid[None, :]
        & (worlds[:, None] == entry_world[None, :])
        & (start_error <= start_limit[:, None])
        & (target_error <= target_limit[:, None])
    )
    configuration_match = (
        jnp.all(query_bounds[:, None, :] == bounds[None, :, :], axis=2)
        & jnp.all(
            query_selector[:, None, :] == selector[None, :, :],
            axis=2,
        )
    )
    candidate = position_candidate & configuration_match
    candidate_count = jnp.sum(candidate, axis=1, dtype=jnp.int32)
    score = start_error + target_error
    selected = jnp.argmin(jnp.where(candidate, score, jnp.inf), axis=1)
    available = input_valid & (candidate_count > 0)
    safe_selected = jnp.clip(selected, 0, capacity - 1)
    selected_paths = paths[safe_selected]
    selected_mask = path_mask[safe_selected]
    selected_start_error = jnp.take_along_axis(
        start_error,
        safe_selected[:, None],
        axis=1,
    )[:, 0]
    selected_target_error = jnp.take_along_axis(
        target_error,
        safe_selected[:, None],
        axis=1,
    )[:, 0]
    selected_paths = jnp.where(
        available[:, None, None],
        selected_paths,
        0.0,
    )
    selected_mask = available[:, None] & selected_mask
    zeros = jnp.zeros((batch,), dtype=jnp.float32)
    return NativePathImitationResult(
        imitation_available=available,
        certified_navigation_step_available=jnp.zeros(
            (batch,),
            dtype=jnp.bool_,
        ),
        path_position=selected_paths,
        path_mask=selected_mask,
        selected_slot=jnp.where(available, safe_selected, -1),
        start_error=jnp.where(available, selected_start_error, zeros),
        target_error=jnp.where(available, selected_target_error, zeros),
        exact_query=(
            available
            & (selected_start_error == 0.0)
            & (selected_target_error == 0.0)
        ),
        ambiguous=available & (candidate_count > 1),
        configuration_mismatch=(
            input_valid
            & jnp.any(position_candidate, axis=1)
            & (candidate_count == 0)
        ),
        out_of_tolerance=(
            input_valid & ~jnp.any(position_candidate, axis=1)
        ),
        invalid=~input_valid,
    )


def native_path_imitation_contract() -> dict[str, object]:
    """Return the checkpoint-pinnable imitation-only boundary."""

    return {
        "schema": NATIVE_PATH_IMITATION_SCHEMA,
        "version": NATIVE_PATH_IMITATION_VERSION,
        "source": "hytalerl_native_navigation_path_probe_v1",
        "storage": {
            "entries": "fixed_capacity_fail_closed_on_host",
            "paths": "fixed_capacity_prefix_mask",
            "positions": "float32_after_validated_native_float64_capture",
            "world_identity": "explicit_int32",
            "maximum_entries": NATIVE_PATH_IMITATION_MAX_ENTRIES,
            "maximum_path_nodes": NATIVE_PATH_IMITATION_MAX_PATH_NODES,
            "overflow": "reject_without_truncation",
        },
        "selection": {
            "metric": "full_xyz_euclidean",
            "admission": (
                "same_world_and_start_error_le_tolerance_and_"
                "target_error_le_tolerance_and_exact_float32_"
                "actor_bounds_and_direction_component_selector"
            ),
            "ranking": "minimum_start_plus_target_error_then_slot",
            "multiple_candidates": "available_with_ambiguous_diagnostic",
        },
        "outputs": [
            "imitation_available",
            "certified_navigation_step_available",
            "path_position",
            "path_mask",
            "selected_slot",
            "start_error",
            "target_error",
            "exact_query",
            "ambiguous",
            "configuration_mismatch",
            "out_of_tolerance",
            "invalid",
        ],
        "certification": {
            "provenance": "native_capture_imitation",
            "certified_navigation_step_available": "always_false",
            "not_equivalent_to": [
                "live_probeMove",
                "live_AStarWithTarget",
                "PathFollower_replan_or_smoothing",
            ],
        },
        "failure": (
            "invalid_query_or_no_in_tolerance_same_world_capture_"
            "returns_no_path"
        ),
    }


def native_path_imitation_contract_sha256() -> str:
    return hashlib.sha256(
        json.dumps(
            native_path_imitation_contract(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _validate_atlas_shape(
    atlas: NativePathImitationAtlas,
) -> tuple[int, int]:
    mask = jnp.asarray(atlas.entry_mask)
    if mask.ndim != 1 or mask.shape[0] == 0:
        raise ValueError("atlas entry_mask must have non-empty shape [capacity]")
    capacity = mask.shape[0]
    if capacity > NATIVE_PATH_IMITATION_MAX_ENTRIES:
        raise ValueError("atlas entry capacity exceeds contract maximum")
    paths = jnp.asarray(atlas.path_position)
    if paths.ndim != 3 or paths.shape[0] != capacity or paths.shape[2] != 3:
        raise ValueError("atlas path_position must have shape [capacity,path,3]")
    path_capacity = paths.shape[1]
    if not 2 <= path_capacity <= NATIVE_PATH_IMITATION_MAX_PATH_NODES:
        raise ValueError("atlas path capacity exceeds contract bounds")
    expected = {
        "world_id": (capacity,),
        "start_position": (capacity, 3),
        "target_position": (capacity, 3),
        "path_mask": (capacity, path_capacity),
        "actor_bounds": (capacity, 6),
        "direction_component_selector": (capacity, 3),
    }
    for name, shape in expected.items():
        if jnp.asarray(getattr(atlas, name)).shape != shape:
            raise ValueError(f"atlas {name} must have shape {shape}")
    return capacity, path_capacity


def _batch_float(value: Array | float, batch: int, name: str) -> Array:
    result = jnp.asarray(value, dtype=jnp.float32)
    if result.ndim == 0:
        result = jnp.broadcast_to(result, (batch,))
    if result.shape != (batch,):
        raise ValueError(f"{name} must be scalar or shape [batch]")
    return result


def _batch_int(value: Array, batch: int, name: str) -> Array:
    result = jnp.asarray(value, dtype=jnp.int32)
    if result.ndim == 0:
        result = jnp.broadcast_to(result, (batch,))
    if result.shape != (batch,):
        raise ValueError(f"{name} must be scalar or shape [batch]")
    return result


def _batch_vector(
    value: Array,
    batch: int,
    width: int,
    name: str,
) -> Array:
    result = jnp.asarray(value, dtype=jnp.float32)
    if result.shape == (width,):
        result = jnp.broadcast_to(result, (batch, width))
    if result.shape != (batch, width):
        raise ValueError(f"{name} must have shape [{width}] or [batch,{width}]")
    return result


__all__ = [
    "NATIVE_PATH_IMITATION_MAX_ENTRIES",
    "NATIVE_PATH_IMITATION_MAX_PATH_NODES",
    "NATIVE_PATH_IMITATION_SCHEMA",
    "NATIVE_PATH_IMITATION_VERSION",
    "NativePathImitationAtlas",
    "NativePathImitationResult",
    "native_path_imitation_contract",
    "native_path_imitation_contract_sha256",
    "native_path_imitation_result",
]
