"""Native camera-ray block candidates over a mutable exact Region."""

from __future__ import annotations

import hashlib
import json
import math

import jax
import jax.numpy as jnp

from hytalegym.jax.world.geometry.rays import (
    native_block_iterator_cells,
    native_block_iterator_contract_sha256,
)
from hytalegym.jax.world.interactions.blocks.tokens import (
    ACTOR_BLOCK_ACTION_DIAGNOSTIC_AFFORDANCE_UNAVAILABLE,
    ACTOR_BLOCK_ACTION_DIAGNOSTIC_INVALID_ACTOR,
    ACTOR_BLOCK_ACTION_DIAGNOSTIC_QUERY_UNAVAILABLE,
    ACTOR_BLOCK_ACTION_DIAGNOSTIC_SOURCE_INCOMPLETE,
    ActorBlockActionCandidates,
    actor_block_action_candidate_contract_sha256,
)
from hytalegym.jax.world.mutable_blocks import (
    mutable_block_contract_sha256,
    query_mutable_blocks,
)
from hytalegym.jax.world.region.action_runtime import (
    RegionActionRuntimeState,
    region_action_runtime_contract_sha256,
)
from hytalegym.jax.world.region.block_semantics import (
    query_region_block_action_target,
    region_block_semantic_atlas_contract_sha256,
)


Array = jax.Array
REGION_CAMERA_BLOCK_CANDIDATE_SCHEMA = (
    "hytalerl_region_camera_block_action_candidates_v1"
)
REGION_CAMERA_BLOCK_CANDIDATE_VERSION = 1
NATIVE_CAMERA_BLOCK_RAY_DISTANCE = 8.0
NATIVE_CAMERA_BLOCK_RAY_CELL_CAPACITY = (
    math.ceil(NATIVE_CAMERA_BLOCK_RAY_DISTANCE * math.sqrt(3.0)) + 2
)


def produce_region_runtime_camera_block_action_candidates(
    runtime: RegionActionRuntimeState,
    actor_position: Array,
    actor_eye_position: Array,
    actor_forward: Array,
    *,
    candidate_capacity: int,
    maximum_interaction_distance: Array | float,
    ray_distance: Array | float = NATIVE_CAMERA_BLOCK_RAY_DISTANCE,
    ray_cell_capacity: int = NATIVE_CAMERA_BLOCK_RAY_CELL_CAPACITY,
) -> ActorBlockActionCandidates:
    """Publish the first non-air camera target as an actor-safe candidate.

    Hytale's `SimpleBlockInteraction.simulateTick0` obtains an absent explicit
    target through `TargetUtil.getTargetBlock(ref, 8.0, ...)`. That API walks
    block IDs, not collision shapes. ``SimpleBlockInteraction`` canonicalizes
    filler cells and later measures that base-cell centre from the same eye
    point. Policy coordinates retain the raw clicked cell relative to the body.
    """

    if not isinstance(runtime, RegionActionRuntimeState):
        raise TypeError("runtime must be a RegionActionRuntimeState")
    if (
        isinstance(candidate_capacity, bool)
        or not isinstance(candidate_capacity, int)
        or candidate_capacity < 1
    ):
        raise ValueError("candidate_capacity must be a positive integer")
    body = jnp.asarray(actor_position, dtype=jnp.float32)
    eyes = jnp.asarray(actor_eye_position, dtype=jnp.float32)
    forward = jnp.asarray(actor_forward, dtype=jnp.float32)
    if (
        body.ndim != 3
        or body.shape[-1] != 3
        or eyes.shape != body.shape
        or forward.shape != body.shape
    ):
        raise ValueError("actor position, eye, and forward must be [B,A,3]")
    batch, actors, _ = body.shape
    rays = native_block_iterator_cells(
        eyes.reshape(batch * actors, 3),
        forward.reshape(batch * actors, 3),
        _ray_parameter(ray_distance, batch, actors).reshape(batch * actors),
        cell_capacity=ray_cell_capacity,
    )
    cells = rays.cells.reshape(batch, actors, ray_cell_capacity, 3)
    ray_mask = rays.mask.reshape(batch, actors, ray_cell_capacity)
    flat_cells = cells.reshape(batch, actors * ray_cell_capacity, 3)
    immutable = query_region_block_action_target(
        runtime.semantic_atlas,
        runtime.geometry.atlas,
        flat_cells,
        runtime.geometry.environment_world_id,
        require_core=True,
    )
    mutable_state = runtime.geometry.mutable_blocks
    if mutable_state is None:
        raise ValueError("Region camera candidates require mutable block state")
    current = query_mutable_blocks(
        mutable_state,
        world_id=runtime.geometry.environment_world_id,
        base_semantic_sha256=mutable_state.base_semantic_sha256,
        resync_epoch=mutable_state.resync_epoch,
        position=immutable.action_position,
        base_available=immutable.query.available,
        base_geometry=immutable.query.geometry,
        base_block_health=immutable.query.block_health,
    )

    def rows(value: Array) -> Array:
        return value.reshape(
            (batch, actors, ray_cell_capacity) + value.shape[2:]
        )

    query_available = rows(current.available) & ~rows(
        current.invalid | current.resync_required
    )
    ray_invalid = rays.invalid.reshape(batch, actors)
    ray_overflow = rays.capacity_exceeded.reshape(batch, actors)
    present = (
        ray_mask
        & query_available
        & rows(current.geometry.block_present)
    )
    first_index = jnp.argmax(present, axis=2)
    first_present = jnp.any(present, axis=2)
    source_index = jnp.arange(ray_cell_capacity, dtype=jnp.int32)
    required_prefix = ray_mask & (
        ~first_present[..., None]
        | (source_index[None, None, :] <= first_index[..., None])
    )
    query_complete = jnp.all(
        ~required_prefix | query_available,
        axis=2,
    )
    source_complete = ~ray_invalid & ~ray_overflow & query_complete
    first_mask = (
        source_index[None, None, :] == first_index[..., None]
    ) & first_present[..., None]
    def first(value: Array) -> Array:
        shaped = rows(value)
        index = first_index[..., None]
        index = index.reshape(
            index.shape + (1,) * (shaped.ndim - index.ndim)
        )
        index = jnp.broadcast_to(
            index,
            shaped.shape[:2] + (1,) + shaped.shape[3:],
        )
        return jnp.take_along_axis(shaped, index, axis=2)[:, :, 0]

    raw_position = jnp.take_along_axis(
        cells,
        jnp.broadcast_to(
            first_index[..., None, None],
            (batch, actors, 1, 3),
        ),
        axis=2,
    )[:, :, 0]
    action_position = first(immutable.action_position)
    exact = first(current.geometry.exact)
    semantic = first(current.geometry.semantic_key_valid)
    affordance = first(current.geometry.affordance_valid)
    first_query_available = first(current.available) & ~first(
        current.invalid | current.resync_required
    )
    semantics_available = exact & semantic & affordance
    maximum = _ray_parameter(
        maximum_interaction_distance,
        batch,
        actors,
    )
    actor_valid = (
        jnp.all(jnp.isfinite(body), axis=2)
        & jnp.all(jnp.isfinite(eyes), axis=2)
        & jnp.isfinite(maximum)
        & (maximum > 0.0)
    )
    action_center = action_position.astype(jnp.float32) + jnp.float32(0.5)
    within_reach = (
        jnp.sum((action_center - eyes) ** 2, axis=2)
        <= maximum**2
    )
    row_available = (
        actor_valid
        & source_complete
        & (
            ~first_present
            | (first_query_available & semantics_available)
        )
    )
    selected = first_present & within_reach & row_available
    candidate_mask = jnp.zeros(
        (batch, actors, candidate_capacity),
        dtype=jnp.bool_,
    ).at[:, :, 0].set(selected)
    diagnostics = (
        jnp.where(
            ~actor_valid,
            ACTOR_BLOCK_ACTION_DIAGNOSTIC_INVALID_ACTOR,
            jnp.uint32(0),
        )
        | jnp.where(
            ~source_complete,
            ACTOR_BLOCK_ACTION_DIAGNOSTIC_SOURCE_INCOMPLETE,
            jnp.uint32(0),
        )
        | jnp.where(
            first_present & ~first_query_available,
            ACTOR_BLOCK_ACTION_DIAGNOSTIC_QUERY_UNAVAILABLE,
            jnp.uint32(0),
        )
        | jnp.where(
            first_present & first_query_available & ~semantics_available,
            ACTOR_BLOCK_ACTION_DIAGNOSTIC_AFFORDANCE_UNAVAILABLE,
            jnp.uint32(0),
        )
    )

    def publish(value: Array) -> Array:
        output = jnp.zeros(
            (batch, actors, candidate_capacity) + value.shape[2:],
            dtype=value.dtype,
        ).at[:, :, 0].set(value)
        gate = candidate_mask.reshape(
            candidate_mask.shape
            + (1,) * (output.ndim - candidate_mask.ndim)
        )
        return jnp.where(gate, output, jnp.zeros_like(output))

    relative = (
        raw_position.astype(jnp.float32)
        + jnp.float32(0.5)
        - body
    )
    return ActorBlockActionCandidates(
        available=row_available,
        capacity_exceeded=jnp.zeros_like(row_available),
        diagnostics=diagnostics,
        candidate_mask=candidate_mask,
        visible_position=publish(raw_position),
        action_position=publish(action_position),
        visible_relative_position=publish(relative),
        affordance_tags=publish(first(current.geometry.affordance_tags)),
        gather_type_index=publish(first(current.geometry.gather_type_index)),
        required_tool_quality=publish(
            first(current.geometry.required_tool_quality)
        ),
        provenance=publish(first(current.provenance)),
    )


def _ray_parameter(value: Array | float, batch: int, actors: int) -> Array:
    result = jnp.asarray(value, dtype=jnp.float32)
    if result.ndim == 0:
        return jnp.broadcast_to(result, (batch, actors))
    if result.shape == (batch,):
        return jnp.broadcast_to(result[:, None], (batch, actors))
    if result.shape != (batch, actors):
        raise ValueError(
            "ray_distance must be scalar, [batch], or [batch,actor]"
        )
    return result


def region_camera_block_candidate_contract() -> dict[str, object]:
    """Return the separate native-camera producer identity."""

    return {
        "schema": REGION_CAMERA_BLOCK_CANDIDATE_SCHEMA,
        "version": REGION_CAMERA_BLOCK_CANDIDATE_VERSION,
        "native_version": "0.5.7",
        "native_sources": {
            "look": "server/core/util/TargetUtil.java#getLook",
            "target": "server/core/util/TargetUtil.java#getTargetBlock",
            "interaction": (
                "server/core/modules/interaction/interaction/config/client/"
                "SimpleBlockInteraction.java#simulateTick0"
            ),
        },
        "dependencies": {
            "voxel_ray": native_block_iterator_contract_sha256(),
            "region_runtime": region_action_runtime_contract_sha256(),
            "region_semantics": (
                region_block_semantic_atlas_contract_sha256()
            ),
            "mutable_blocks": mutable_block_contract_sha256(),
            "actor_candidates": (
                actor_block_action_candidate_contract_sha256()
            ),
        },
        "camera": {
            "origin": "model_eye_height",
            "direction": "head_rotation_3d",
            "ray_distance": NATIVE_CAMERA_BLOCK_RAY_DISTANCE,
            "cell_capacity": NATIVE_CAMERA_BLOCK_RAY_CELL_CAPACITY,
            "target": "first_current_non_air_block_id",
            "visibility": "camera_presence_ray_no_second_role_los",
        },
        "commit": {
            "distance_origin": "model_eye_height",
            "distance_target": "filler_canonical_base_cell_center",
            "maximum": "authored_adventure_use_distance_plus_2",
        },
        "policy_relative_origin": "actor_body_position",
        "fail_closed": [
            "invalid_or_exhausted_ray",
            "any_required_region_cell_unavailable",
            "mutable_overlay_unavailable",
            "selected_affordance_unavailable",
        ],
    }


def region_camera_block_candidate_contract_sha256() -> str:
    payload = json.dumps(
        region_camera_block_candidate_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "NATIVE_CAMERA_BLOCK_RAY_CELL_CAPACITY",
    "NATIVE_CAMERA_BLOCK_RAY_DISTANCE",
    "REGION_CAMERA_BLOCK_CANDIDATE_SCHEMA",
    "REGION_CAMERA_BLOCK_CANDIDATE_VERSION",
    "produce_region_runtime_camera_block_action_candidates",
    "region_camera_block_candidate_contract",
    "region_camera_block_candidate_contract_sha256",
]
