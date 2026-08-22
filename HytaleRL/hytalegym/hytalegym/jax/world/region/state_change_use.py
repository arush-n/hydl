"""Exact Region-backed generic block Use state transitions."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.block_actions import BlockMutationAcknowledgement
from hytalegym.jax.world.interactions.blocks.tokens import (
    ActorBlockActionCandidates,
    actor_block_action_candidate_contract_sha256,
)
from hytalegym.jax.world.mutable_blocks import (
    MUTABLE_BLOCK_PROVENANCE_NATIVE,
    MutableBlockGeometry,
    MutableBlockState,
    empty_mutable_block_update,
    mutable_block_contract_sha256,
)
from hytalegym.jax.world.region.action_runtime import (
    RegionActionRuntimeState,
    RegionSelectedBlockActionTarget,
    commit_region_block_mutation,
    requery_region_block_action_candidates,
    region_action_runtime_contract_sha256,
    select_region_block_action_target,
)
from hytalegym.jax.world.region.block_semantics import (
    query_region_block_action_target,
)
from hytalegym.worldgen.region import (
    CAPTURE_CHUNKS_PER_AXIS,
    CAPTURE_HALO_CHUNKS,
    CHUNK_SIZE,
    CORE_CHUNKS_PER_AXIS,
    MIN_Y,
)
from hytalegym.worldgen.region.state_change_use import (
    RegionStateChangeUseDefinition,
    region_state_change_use_contract_sha256,
)


REGION_STATE_CHANGE_USE_RUNTIME_SCHEMA = (
    "hytalerl_region_state_change_use_runtime_v1"
)
REGION_STATE_CHANGE_USE_RUNTIME_VERSION = 1

REGION_STATE_CHANGE_USE_DIAGNOSTIC_SELECTION = jnp.uint32(1 << 0)
REGION_STATE_CHANGE_USE_DIAGNOSTIC_TRANSITION = jnp.uint32(1 << 1)
REGION_STATE_CHANGE_USE_DIAGNOSTIC_GEOMETRY = jnp.uint32(1 << 2)
REGION_STATE_CHANGE_USE_DIAGNOSTIC_COMMIT = jnp.uint32(1 << 3)


class RegionStateChangeUseTable(NamedTuple):
    """Fixed transitions with exact target geometry per loaded Region."""

    source_semantic_key: jax.Array
    target_semantic_key: jax.Array
    transition_valid: jax.Array
    target_available: jax.Array
    target_geometry: MutableBlockGeometry


class RegionStateChangeUseResult(NamedTuple):
    runtime: RegionActionRuntimeState
    selection: RegionSelectedBlockActionTarget
    acknowledgement: BlockMutationAcknowledgement
    accepted: jax.Array
    state_changed: jax.Array
    unsupported: jax.Array
    diagnostics: jax.Array


class RegionStateChangeUseCandidates(NamedTuple):
    """Actor-legal candidates whose current state has an exact target row."""

    available: jax.Array
    candidate_mask: jax.Array


def region_state_change_use_candidates(
    runtime: RegionActionRuntimeState,
    candidates: ActorBlockActionCandidates,
    table: RegionStateChangeUseTable,
) -> RegionStateChangeUseCandidates:
    """Requery and mark candidates executable by the current Use table."""

    if not isinstance(runtime, RegionActionRuntimeState):
        raise TypeError("runtime must be a RegionActionRuntimeState")
    if not isinstance(candidates, ActorBlockActionCandidates):
        raise TypeError("candidates must be ActorBlockActionCandidates")
    if not isinstance(table, RegionStateChangeUseTable):
        raise TypeError("table must be a RegionStateChangeUseTable")
    current = requery_region_block_action_candidates(
        runtime.geometry,
        runtime.semantic_atlas,
        candidates,
    )
    batch, actors, capacity = candidates.candidate_mask.shape
    atlas = runtime.geometry.atlas
    world = runtime.geometry.environment_world_id
    region_matches = (
        atlas.region_mask[None, :]
        & (atlas.world_id[None, :] == world[:, None])
    )
    unique_region = jnp.sum(region_matches, axis=1) == 1
    region_row = jnp.argmax(region_matches, axis=1).astype(jnp.int32)
    geometry = current.geometry
    keys = geometry.semantic_key.reshape(
        batch,
        actors,
        capacity,
        geometry.semantic_key.shape[-1],
    )
    key_valid = geometry.semantic_key_valid.reshape(batch, actors, capacity)
    query_available = current.available.reshape(batch, actors, capacity)
    transition_match = (
        table.transition_valid[None, None, None, :]
        & key_valid[..., None]
        & jnp.all(
            keys[..., None, :]
            == table.source_semantic_key[None, None, None, :, :],
            axis=4,
        )
        & table.target_available[region_row][:, None, None, :]
    )
    mask = (
        candidates.candidate_mask
        & candidates.available[..., None]
        & query_available
        & unique_region[:, None, None]
        & (jnp.sum(transition_match, axis=3) == 1)
    )
    return RegionStateChangeUseCandidates(
        available=jnp.any(mask, axis=2),
        candidate_mask=mask,
    )


def compile_region_state_change_use_table(
    runtime: RegionActionRuntimeState,
    definitions: Sequence[RegionStateChangeUseDefinition],
) -> RegionStateChangeUseTable:
    """Bind source-derived transitions to native-exact Region geometry.

    A transition is available in one Region only when that Region contains a
    direct, non-filler instance of the exact target semantic key. No asset
    default cube or source-state geometry is reused as a substitute.
    """

    if not isinstance(runtime, RegionActionRuntimeState):
        raise TypeError("runtime must be a RegionActionRuntimeState")
    rows = tuple(definitions)
    if not rows or any(
        not isinstance(row, RegionStateChangeUseDefinition) for row in rows
    ):
        raise ValueError("definitions must contain at least one Use row")
    source_keys = np.asarray(
        [row.source_semantic_key for row in rows],
        dtype=np.uint32,
    )
    target_keys = np.asarray(
        [row.target_semantic_key for row in rows],
        dtype=np.uint32,
    )
    if len({tuple(int(value) for value in row) for row in source_keys}) != len(
        rows
    ):
        raise ValueError("Use source semantic keys must be unique")

    physical = runtime.geometry.atlas
    semantic = runtime.semantic_atlas
    region_mask = np.asarray(jax.device_get(physical.region_mask), dtype=np.bool_)
    world_id = np.asarray(jax.device_get(physical.world_id), dtype=np.int32)
    core_min = np.asarray(
        jax.device_get(physical.core_min_chunk_xz),
        dtype=np.int32,
    )
    semantic_code = np.asarray(
        jax.device_get(semantic.cell_code),
        dtype=np.uint16,
    )
    semantic_keys = np.asarray(
        jax.device_get(semantic.semantic_key),
        dtype=np.uint32,
    )
    entry_valid = np.asarray(
        jax.device_get(semantic.entry_valid),
        dtype=np.bool_,
    )
    semantic_known = np.asarray(
        jax.device_get(semantic.section_known),
        dtype=np.bool_,
    )
    physical_known = np.asarray(
        jax.device_get(physical.section_known),
        dtype=np.bool_,
    )
    filler_known = np.asarray(
        jax.device_get(physical.filler_root_known),
        dtype=np.bool_,
    )
    filler_offset = np.asarray(
        jax.device_get(physical.filler_root_offset_packed),
        dtype=np.uint16,
    )

    positions = np.zeros(
        (region_mask.shape[0], len(rows), 3),
        dtype=np.int32,
    )
    found = np.zeros((region_mask.shape[0], len(rows)), dtype=np.bool_)
    for region in range(region_mask.shape[0]):
        if not region_mask[region] or not filler_known[region]:
            continue
        for row, key in enumerate(target_keys):
            matches = np.flatnonzero(
                entry_valid[region]
                & np.all(semantic_keys[region] == key, axis=1)
            )
            if matches.size != 1:
                continue
            position = _first_direct_core_position(
                semantic_code[region],
                semantic_known[region],
                physical_known[region],
                filler_offset[region],
                core_min[region],
                int(matches[0]),
            )
            if position is None:
                continue
            positions[region, row] = position
            found[region, row] = True

    queried = query_region_block_action_target(
        semantic,
        physical,
        jnp.asarray(positions),
        jnp.asarray(world_id),
        require_core=True,
    )
    target_key_array = jnp.asarray(target_keys, dtype=jnp.uint32)
    key_match = jnp.all(
        queried.query.geometry.semantic_key
        == target_key_array[None, :, :],
        axis=2,
    )
    available = (
        jnp.asarray(found)
        & queried.available
        & jnp.all(queried.action_position == jnp.asarray(positions), axis=2)
        & ~queried.canonicalized_to_root
        & ~queried.direct_filler
        & queried.query.geometry.exact
        & queried.query.geometry.block_present
        & queried.query.geometry.semantic_key_valid
        & key_match
    )
    geometry = jax.tree.map(
        lambda value: jnp.where(
            available.reshape(
                available.shape + (1,) * (value.ndim - available.ndim)
            ),
            value,
            jnp.zeros_like(value),
        ),
        queried.query.geometry,
    )
    return RegionStateChangeUseTable(
        source_semantic_key=jnp.asarray(source_keys, dtype=jnp.uint32),
        target_semantic_key=target_key_array,
        transition_valid=jnp.ones((len(rows),), dtype=jnp.bool_),
        target_available=available,
        target_geometry=geometry,
    )


def execute_region_state_change_use(
    runtime: RegionActionRuntimeState,
    candidates: ActorBlockActionCandidates,
    selected_candidate_index: jax.Array,
    table: RegionStateChangeUseTable,
    *,
    intent_mask: jax.Array,
) -> RegionStateChangeUseResult:
    """Requery and atomically apply one exact current Region state change."""

    if not isinstance(runtime, RegionActionRuntimeState):
        raise TypeError("runtime must be a RegionActionRuntimeState")
    if not isinstance(candidates, ActorBlockActionCandidates):
        raise TypeError("candidates must be ActorBlockActionCandidates")
    if not isinstance(table, RegionStateChangeUseTable):
        raise TypeError("table must be a RegionStateChangeUseTable")
    state = runtime.geometry.mutable_blocks
    if not isinstance(state, MutableBlockState):
        raise ValueError("Region Use requires synchronized mutable block state")
    selection = select_region_block_action_target(
        runtime.geometry,
        runtime.semantic_atlas,
        candidates,
        selected_candidate_index,
    )
    intents = jnp.asarray(intent_mask)
    if intents.dtype != jnp.bool_ or intents.shape != selection.available.shape:
        raise ValueError("intent_mask must match the selected candidate shape")

    atlas = runtime.geometry.atlas
    world = runtime.geometry.environment_world_id
    region_matches = (
        atlas.region_mask[None, :]
        & (atlas.world_id[None, :] == world[:, None])
    )
    unique_region = jnp.sum(region_matches, axis=1) == 1
    region_row = jnp.argmax(region_matches, axis=1).astype(jnp.int32)
    source_match = (
        table.transition_valid[None, None, :]
        & selection.query.geometry.semantic_key_valid[..., None]
        & jnp.all(
            selection.query.geometry.semantic_key[..., None, :]
            == table.source_semantic_key[None, None, :, :],
            axis=3,
        )
        & table.target_available[region_row][:, None, :]
    )
    unique_transition = jnp.sum(source_match, axis=2) == 1
    transition_row = jnp.argmax(source_match, axis=2).astype(jnp.int32)
    target_geometry = jax.tree.map(
        lambda value: value[region_row[:, None], transition_row],
        table.target_geometry,
    )
    target_ready = (
        target_geometry.exact
        & target_geometry.block_present
        & target_geometry.semantic_key_valid
        & jnp.all(
            target_geometry.semantic_key
            == table.target_semantic_key[transition_row],
            axis=2,
        )
    )
    available = (
        intents
        & selection.available
        & unique_region[:, None]
        & unique_transition
        & target_ready
    )
    update = empty_mutable_block_update(
        state,
        query_capacity=available.shape[1],
    )._replace(
        mask=available,
        position=selection.position,
        geometry_changed=available,
        expected_health=selection.query.block_health,
        block_health_after=selection.query.block_health,
        provenance=jnp.where(
            available,
            jnp.uint8(MUTABLE_BLOCK_PROVENANCE_NATIVE),
            jnp.uint8(0),
        ),
        geometry=jax.tree.map(
            lambda value: jnp.where(
                available.reshape(
                    available.shape
                    + (1,) * (value.ndim - available.ndim)
                ),
                value,
                jnp.zeros_like(value),
            ),
            target_geometry,
        ),
    )
    committed = commit_region_block_mutation(
        runtime.geometry,
        runtime.semantic_atlas,
        update,
    )
    accepted = available & committed.acknowledgement.applied
    diagnostics = (
        jnp.where(
            intents & ~selection.available,
            REGION_STATE_CHANGE_USE_DIAGNOSTIC_SELECTION,
            jnp.uint32(0),
        )
        | jnp.where(
            intents & selection.available & ~unique_transition,
            REGION_STATE_CHANGE_USE_DIAGNOSTIC_TRANSITION,
            jnp.uint32(0),
        )
        | jnp.where(
            intents & unique_transition & ~target_ready,
            REGION_STATE_CHANGE_USE_DIAGNOSTIC_GEOMETRY,
            jnp.uint32(0),
        )
        | jnp.where(
            available & ~committed.acknowledgement.applied,
            REGION_STATE_CHANGE_USE_DIAGNOSTIC_COMMIT,
            jnp.uint32(0),
        )
    )
    return RegionStateChangeUseResult(
        runtime=runtime._replace(geometry=committed.geometry),
        selection=selection,
        acknowledgement=committed.acknowledgement,
        accepted=accepted,
        state_changed=accepted,
        unsupported=intents & ~accepted,
        diagnostics=diagnostics,
    )


def region_state_change_use_runtime_contract() -> dict[str, object]:
    return {
        "schema": REGION_STATE_CHANGE_USE_RUNTIME_SCHEMA,
        "version": REGION_STATE_CHANGE_USE_RUNTIME_VERSION,
        "dependencies": {
            "region_state_change_use_contract_sha256": (
                region_state_change_use_contract_sha256()
            ),
            "region_action_runtime_contract_sha256": (
                region_action_runtime_contract_sha256()
            ),
            "actor_block_action_candidate_contract_sha256": (
                actor_block_action_candidate_contract_sha256()
            ),
            "mutable_block_contract_sha256": mutable_block_contract_sha256(),
        },
        "targeting": "actor_legal_candidate_then_current_revision_requery",
        "availability": (
            "actor_legal_current_state_with_unique_transition_and_exact_"
            "same_region_target_geometry"
        ),
        "target_geometry": (
            "exact_direct_non_filler_matching_target_state_in_same_region"
        ),
        "commit": "atomic_mutable_region_overlay_and_world_reprojection",
        "unavailable_target_geometry": "reject_without_mutation",
        "published_provenance": (
            "source_exact_transition_plus_native_exact_region_geometry"
        ),
        "native_outcome_calibration": "separate_recording_required",
        "fail_closed": True,
    }


def region_state_change_use_runtime_contract_sha256() -> str:
    payload = json.dumps(
        region_state_change_use_runtime_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _first_direct_core_position(
    semantic_code: np.ndarray,
    semantic_known: np.ndarray,
    physical_known: np.ndarray,
    filler_offset: np.ndarray,
    core_min_chunk_xz: np.ndarray,
    palette_code: int,
) -> tuple[int, int, int] | None:
    for relative_x in range(
        CAPTURE_HALO_CHUNKS,
        CAPTURE_HALO_CHUNKS + CORE_CHUNKS_PER_AXIS,
    ):
        for relative_z in range(
            CAPTURE_HALO_CHUNKS,
            CAPTURE_HALO_CHUNKS + CORE_CHUNKS_PER_AXIS,
        ):
            chunk_slot = relative_x * CAPTURE_CHUNKS_PER_AXIS + relative_z
            for section in range(semantic_code.shape[1]):
                if not (
                    semantic_known[chunk_slot, section]
                    and physical_known[chunk_slot, section]
                ):
                    continue
                direct = (
                    (semantic_code[chunk_slot, section] == palette_code)
                    & (filler_offset[chunk_slot, section] == 0)
                )
                indexes = np.flatnonzero(direct)
                if indexes.size == 0:
                    continue
                cell = int(indexes[0])
                local_y, remainder = divmod(cell, CHUNK_SIZE * CHUNK_SIZE)
                local_z, local_x = divmod(remainder, CHUNK_SIZE)
                chunk_x = (
                    int(core_min_chunk_xz[0])
                    + relative_x
                    - CAPTURE_HALO_CHUNKS
                )
                chunk_z = (
                    int(core_min_chunk_xz[1])
                    + relative_z
                    - CAPTURE_HALO_CHUNKS
                )
                return (
                    chunk_x * CHUNK_SIZE + local_x,
                    MIN_Y + section * CHUNK_SIZE + local_y,
                    chunk_z * CHUNK_SIZE + local_z,
                )
    return None


__all__ = [
    "REGION_STATE_CHANGE_USE_DIAGNOSTIC_COMMIT",
    "REGION_STATE_CHANGE_USE_DIAGNOSTIC_GEOMETRY",
    "REGION_STATE_CHANGE_USE_DIAGNOSTIC_SELECTION",
    "REGION_STATE_CHANGE_USE_DIAGNOSTIC_TRANSITION",
    "REGION_STATE_CHANGE_USE_RUNTIME_SCHEMA",
    "REGION_STATE_CHANGE_USE_RUNTIME_VERSION",
    "RegionStateChangeUseResult",
    "RegionStateChangeUseCandidates",
    "RegionStateChangeUseTable",
    "compile_region_state_change_use_table",
    "execute_region_state_change_use",
    "region_state_change_use_candidates",
    "region_state_change_use_runtime_contract",
    "region_state_change_use_runtime_contract_sha256",
]
