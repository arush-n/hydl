"""Region-backed actor block evidence and exact mutation commits."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.block_actions import (
    BlockActionCatalog,
    BlockActionMutationPlan,
    BlockActionTargetResult,
    BlockDropProgramCatalog,
    BlockGatherDefaults,
    BlockMutationAcknowledgement,
    BlockToolState,
    exact_block_mutation_acknowledgement,
    plan_exact_block_action_mutation,
    resolve_block_action_targets,
)
from hytalegym.jax.world.interactions.blocks.tokens import (
    ActorBlockActionCandidates,
    dense_actor_block_candidate_cells,
    dense_actor_block_candidate_source_complete,
    produce_actor_block_action_candidates,
)
from hytalegym.jax.world.mutable_blocks import (
    MUTABLE_BLOCK_PROVENANCE_NATIVE,
    MutableBlockGeometry,
    MutableBlockQueryResult,
    MutableBlockState,
    MutableBlockUpdate,
    apply_mutable_block_updates,
    empty_mutable_block_resync,
    empty_mutable_block_state,
    empty_mutable_block_update,
    mutable_block_contract_sha256,
    mutable_block_geometry_from_native_cells,
    publish_mutable_block_resync,
    query_mutable_blocks,
)
from hytalegym.jax.world.region.block_semantics import (
    RegionBlockSemanticAtlas,
    query_region_block_action_target,
    region_block_semantic_atlas_contract_sha256,
)
from hytalegym.jax.world.region.types import RegionGeometryState
from hytalegym.worldgen.region import (
    CAPTURE_CHUNKS_PER_AXIS,
    HEIGHT_SECTIONS,
    NativeRegionSnapshot,
)


Array = jax.Array
REGION_ACTION_RUNTIME_SCHEMA = "hytalerl_region_action_runtime_v2"
REGION_ACTION_RUNTIME_VERSION = 2

REGION_ACTION_SELECTION_DIAGNOSTIC_INDEX = 1 << 0
REGION_ACTION_SELECTION_DIAGNOSTIC_CANDIDATE = 1 << 1
REGION_ACTION_SELECTION_DIAGNOSTIC_POSITION = 1 << 2



class RegionBlockMutationCommit(NamedTuple):
    """One atomic mutable-state commit and its physical Region projection."""

    geometry: RegionGeometryState
    state: MutableBlockState
    acknowledgement: BlockMutationAcknowledgement


class RegionActionRuntimeState(NamedTuple):
    """Persistent physical and semantic Region state carried by the env."""

    geometry: RegionGeometryState
    semantic_atlas: RegionBlockSemanticAtlas


class RegionSelectedBlockActionTarget(NamedTuple):
    """One policy-selected candidate, requeried from current exact state."""

    available: Array
    position: Array
    provenance: Array
    query: MutableBlockQueryResult
    diagnostics: Array


class RegionBlockActionPlan(NamedTuple):
    """Selected target, refreshed predicates, and revision-bound mutation."""

    selection: RegionSelectedBlockActionTarget
    actions: BlockActionTargetResult
    mutation: BlockActionMutationPlan


class RegionBlockActionExecution(NamedTuple):
    """One exact JAX World action and the resulting persistent runtime."""

    runtime: RegionActionRuntimeState
    plan: RegionBlockActionPlan
    acknowledgement: BlockMutationAcknowledgement


def mutable_block_state_from_region_snapshots(
    snapshots: Sequence[NativeRegionSnapshot],
    geometry: RegionGeometryState,
    *,
    world_ids: Sequence[int] | None = None,
    mutation_capacity: int,
    resync_epoch: int = 0,
) -> MutableBlockState:
    """Publish an empty exact overlay over selected frozen Region bases."""

    if not isinstance(geometry, RegionGeometryState):
        raise TypeError("geometry must be a RegionGeometryState")
    source = tuple(snapshots)
    if not source or any(
        not isinstance(value, NativeRegionSnapshot) for value in source
    ):
        raise TypeError(
            "snapshots must be a nonempty NativeRegionSnapshot sequence"
        )
    ids = (
        tuple(range(len(source)))
        if world_ids is None
        else tuple(_host_int(value, "world_id") for value in world_ids)
    )
    if len(ids) != len(source) or len(set(ids)) != len(ids):
        raise ValueError("world_ids must be unique and match snapshots")
    environments = np.asarray(
        jax.device_get(geometry.environment_world_id),
        dtype=np.int32,
    )
    if environments.ndim != 1 or environments.size == 0:
        raise ValueError("environment_world_id must have shape [batch]")
    by_world = dict(zip(ids, source, strict=True))
    try:
        selected = tuple(by_world[int(value)] for value in environments)
    except KeyError as error:
        raise ValueError(
            "environment_world_id references an absent Region snapshot"
        ) from error
    for snapshot, world_id in zip(selected, environments, strict=True):
        _require_snapshot_matches_atlas(
            geometry,
            snapshot,
            int(world_id),
        )
    cells = _positive_int(mutation_capacity, "mutation_capacity")
    epoch = _nonnegative_int(resync_epoch, "resync_epoch")
    if epoch > np.iinfo(np.uint32).max:
        raise ValueError("resync_epoch must fit uint32")

    section_capacity = (
        CAPTURE_CHUNKS_PER_AXIS
        * CAPTURE_CHUNKS_PER_AXIS
        * HEIGHT_SECTIONS
    )
    state = empty_mutable_block_state(
        environments.size,
        section_capacity=section_capacity,
        mutation_capacity=cells,
    )
    frame = empty_mutable_block_resync(state)
    section_coordinates = np.zeros(
        (environments.size, section_capacity, 3),
        dtype=np.int32,
    )
    digests = np.zeros((environments.size, 8), dtype=np.uint32)
    for batch_index, snapshot in enumerate(selected):
        core_x, core_z = (
            int(snapshot.core_min_chunk_xz[0]),
            int(snapshot.core_min_chunk_xz[1]),
        )
        section_coordinates[batch_index] = np.asarray(
            [
                (chunk_x, section_y, chunk_z)
                for chunk_x in range(core_x - 1, core_x + 4)
                for chunk_z in range(core_z - 1, core_z + 4)
                for section_y in range(HEIGHT_SECTIONS)
            ],
            dtype=np.int32,
        )
        digests[batch_index] = np.frombuffer(
            bytes.fromhex(snapshot.semantic_artifact_digest()),
            dtype="<u4",
        )
    frame = frame._replace(
        publish=jnp.ones((environments.size,), dtype=jnp.bool_),
        snapshot_complete=jnp.ones(
            (environments.size,),
            dtype=jnp.bool_,
        ),
        world_id=jnp.asarray(environments, dtype=jnp.int32),
        base_semantic_sha256=jnp.asarray(digests),
        resync_epoch=jnp.full(
            (environments.size,),
            epoch,
            dtype=jnp.uint32,
        ),
        base_provenance=jnp.full(
            (environments.size,),
            MUTABLE_BLOCK_PROVENANCE_NATIVE,
            dtype=jnp.uint8,
        ),
        section_mask=jnp.ones(
            (environments.size, section_capacity),
            dtype=jnp.bool_,
        ),
        section_coordinate=jnp.asarray(section_coordinates),
    )
    state, result = publish_mutable_block_resync(state, frame)
    if not np.all(np.asarray(jax.device_get(result.accepted))):
        raise RuntimeError("frozen Region mutable base publication failed")
    return state


def produce_region_actor_block_action_candidates(
    geometry: RegionGeometryState,
    semantic_atlas: RegionBlockSemanticAtlas,
    actor_position: Array,
    actor_eye_position: Array,
    actor_forward: Array,
    *,
    role_opaque_mask: Array,
    candidate_capacity: int,
    maximum_distance: Array | float,
    view_sector_full_angle_radians: Array | float,
    cell_radius: int,
    max_los_cells: int | None = None,
) -> ActorBlockActionCandidates:
    """Publish complete actor-legal candidates from current Region state."""

    state = _bound_state(geometry)
    positions = jnp.asarray(actor_position, dtype=jnp.float32)
    cells = dense_actor_block_candidate_cells(
        positions,
        cell_radius=cell_radius,
    )
    batch, actors, source_capacity, _ = cells.shape
    flat_cells = cells.reshape(batch, actors * source_capacity, 3)
    immutable = query_region_block_action_target(
        semantic_atlas,
        geometry.atlas,
        flat_cells,
        geometry.environment_world_id,
        require_core=True,
    )
    current = _query_current(
        geometry,
        state,
        immutable.action_position,
        immutable.query,
    )
    source_complete = dense_actor_block_candidate_source_complete(
        positions,
        maximum_distance,
        cell_radius=cell_radius,
    )
    return produce_actor_block_action_candidates(
        geometry,
        positions,
        actor_eye_position,
        actor_forward,
        cells,
        immutable.action_position.reshape(
            batch,
            actors,
            source_capacity,
            3,
        ),
        current,
        source_complete=source_complete,
        role_opaque_mask=role_opaque_mask,
        candidate_capacity=candidate_capacity,
        maximum_distance=maximum_distance,
        view_sector_full_angle_radians=view_sector_full_angle_radians,
        max_los_cells=max_los_cells,
    )


def produce_region_runtime_actor_block_action_candidates(
    runtime: RegionActionRuntimeState,
    actor_position: Array,
    actor_eye_position: Array,
    actor_forward: Array,
    *,
    role_opaque_mask: Array,
    candidate_capacity: int,
    maximum_distance: Array | float,
    view_sector_full_angle_radians: Array | float,
    cell_radius: int,
    max_los_cells: int | None = None,
) -> ActorBlockActionCandidates:
    """Produce candidates from the same mutable runtime used by physics."""

    if not isinstance(runtime, RegionActionRuntimeState):
        raise TypeError("runtime must be a RegionActionRuntimeState")
    return produce_region_actor_block_action_candidates(
        runtime.geometry,
        runtime.semantic_atlas,
        actor_position,
        actor_eye_position,
        actor_forward,
        role_opaque_mask=role_opaque_mask,
        candidate_capacity=candidate_capacity,
        maximum_distance=maximum_distance,
        view_sector_full_angle_radians=view_sector_full_angle_radians,
        cell_radius=cell_radius,
        max_los_cells=max_los_cells,
    )


def requery_region_block_action_candidates(
    geometry: RegionGeometryState,
    semantic_atlas: RegionBlockSemanticAtlas,
    candidates: ActorBlockActionCandidates,
) -> MutableBlockQueryResult:
    """Requery selected-slot coordinates without trusting stale policy rows."""

    if not isinstance(candidates, ActorBlockActionCandidates):
        raise TypeError("candidates must be ActorBlockActionCandidates")
    state = _bound_state(geometry)
    batch, actors, capacity = candidates.candidate_mask.shape
    positions = candidates.action_position.reshape(
        batch,
        actors * capacity,
        3,
    )
    immutable = query_region_block_action_target(
        semantic_atlas,
        geometry.atlas,
        positions,
        geometry.environment_world_id,
        require_core=True,
    )
    position_match = jnp.all(
        immutable.action_position == positions,
        axis=2,
    )
    base = immutable.query._replace(
        available=immutable.query.available & position_match,
    )
    return _query_current(
        geometry,
        state,
        positions,
        base,
    )


def select_region_block_action_target(
    geometry: RegionGeometryState,
    semantic_atlas: RegionBlockSemanticAtlas,
    candidates: ActorBlockActionCandidates,
    selected_candidate_index: Array,
) -> RegionSelectedBlockActionTarget:
    """Gather and requery one candidate per actor without trusting its row."""

    if not isinstance(candidates, ActorBlockActionCandidates):
        raise TypeError("candidates must be ActorBlockActionCandidates")
    state = _bound_state(geometry)
    batch, actors, capacity = candidates.candidate_mask.shape
    raw_index = jnp.asarray(selected_candidate_index)
    if raw_index.shape != (batch, actors):
        raise ValueError(
            "selected_candidate_index must have shape [batch, actor]"
        )
    if not jnp.issubdtype(raw_index.dtype, jnp.integer):
        raise TypeError("selected_candidate_index must be integer-valued")
    index = raw_index.astype(jnp.int32)
    index_valid = (index >= 0) & (index < capacity)
    safe_index = jnp.clip(index, 0, capacity - 1)
    position = _gather_actor_candidate(
        candidates.action_position,
        safe_index,
    )
    provenance = _gather_actor_candidate(
        candidates.provenance,
        safe_index,
    )
    candidate = (
        candidates.available
        & _gather_actor_candidate(candidates.candidate_mask, safe_index)
        & index_valid
    )
    flat_position = position.reshape(batch, actors, 3)
    immutable = query_region_block_action_target(
        semantic_atlas,
        geometry.atlas,
        flat_position,
        geometry.environment_world_id,
        require_core=True,
    )
    position_match = jnp.all(
        immutable.action_position == flat_position,
        axis=2,
    )
    base = immutable.query._replace(
        available=immutable.query.available & candidate & position_match,
    )
    current = _query_current(
        geometry,
        state,
        flat_position,
        base,
    )
    available = (
        candidate
        & position_match
        & current.available
        & ~current.invalid
        & ~current.resync_required
    )
    diagnostics = (
        jnp.where(
            ~index_valid,
            jnp.uint32(REGION_ACTION_SELECTION_DIAGNOSTIC_INDEX),
            jnp.uint32(0),
        )
        | jnp.where(
            index_valid & ~candidate,
            jnp.uint32(REGION_ACTION_SELECTION_DIAGNOSTIC_CANDIDATE),
            jnp.uint32(0),
        )
        | jnp.where(
            candidate & ~position_match,
            jnp.uint32(REGION_ACTION_SELECTION_DIAGNOSTIC_POSITION),
            jnp.uint32(0),
        )
    )
    return RegionSelectedBlockActionTarget(
        available=available,
        position=jnp.where(available[..., None], position, 0),
        provenance=jnp.where(
            available,
            provenance,
            jnp.uint8(0),
        ),
        query=current._replace(
            available=current.available & available,
        ),
        diagnostics=diagnostics,
    )


def plan_region_runtime_block_action(
    runtime: RegionActionRuntimeState,
    candidates: ActorBlockActionCandidates,
    selected_candidate_index: Array,
    tool: BlockToolState,
    defaults: BlockGatherDefaults,
    catalog: BlockActionCatalog,
    *,
    break_selected: Array,
    place_selected: Array,
    placed_geometry: MutableBlockGeometry,
    replacement_geometry: MutableBlockGeometry,
    modification_evidence_available: Array,
    modification_allowed: Array,
    place_evidence_available: Array,
    place_allowed: Array,
    break_interaction_tool_type_id: Array | None = None,
    break_interaction_match_tool: Array | None = None,
    drop_programs: BlockDropProgramCatalog | None = None,
    drop_random_samples: Array | None = None,
) -> RegionBlockActionPlan:
    """Requery, resolve, and plan one exact Break/Place action per actor."""

    if not isinstance(runtime, RegionActionRuntimeState):
        raise TypeError("runtime must be a RegionActionRuntimeState")
    selection = select_region_block_action_target(
        runtime.geometry,
        runtime.semantic_atlas,
        candidates,
        selected_candidate_index,
    )
    actions = resolve_block_action_targets(
        selection.query,
        tool,
        defaults,
        catalog,
        modification_evidence_available=modification_evidence_available,
        modification_allowed=modification_allowed,
        place_evidence_available=place_evidence_available,
        place_allowed=place_allowed,
        break_interaction_tool_type_id=break_interaction_tool_type_id,
        break_interaction_match_tool=break_interaction_match_tool,
        drop_programs=drop_programs,
        drop_random_samples=drop_random_samples,
    )
    mutation = plan_exact_block_action_mutation(
        _bound_state(runtime.geometry),
        selection.query,
        actions,
        position=selection.position,
        break_selected=break_selected,
        place_selected=place_selected,
        placed_geometry=placed_geometry,
        replacement_geometry=replacement_geometry,
        provenance=selection.provenance,
    )
    return RegionBlockActionPlan(selection, actions, mutation)


def execute_region_runtime_block_action(
    runtime: RegionActionRuntimeState,
    candidates: ActorBlockActionCandidates,
    selected_candidate_index: Array,
    tool: BlockToolState,
    defaults: BlockGatherDefaults,
    catalog: BlockActionCatalog,
    *,
    break_selected: Array,
    place_selected: Array,
    placed_geometry: MutableBlockGeometry,
    replacement_geometry: MutableBlockGeometry,
    modification_evidence_available: Array,
    modification_allowed: Array,
    place_evidence_available: Array,
    place_allowed: Array,
    break_interaction_tool_type_id: Array | None = None,
    break_interaction_match_tool: Array | None = None,
    drop_programs: BlockDropProgramCatalog | None = None,
    drop_random_samples: Array | None = None,
) -> RegionBlockActionExecution:
    """Execute the exact JAX counterpart of the typed native World action."""

    plan = plan_region_runtime_block_action(
        runtime,
        candidates,
        selected_candidate_index,
        tool,
        defaults,
        catalog,
        break_selected=break_selected,
        place_selected=place_selected,
        placed_geometry=placed_geometry,
        replacement_geometry=replacement_geometry,
        modification_evidence_available=modification_evidence_available,
        modification_allowed=modification_allowed,
        place_evidence_available=place_evidence_available,
        place_allowed=place_allowed,
        break_interaction_tool_type_id=break_interaction_tool_type_id,
        break_interaction_match_tool=break_interaction_match_tool,
        drop_programs=drop_programs,
        drop_random_samples=drop_random_samples,
    )
    committed = commit_region_block_mutation(
        runtime.geometry,
        runtime.semantic_atlas,
        plan.mutation.update,
    )
    return RegionBlockActionExecution(
        runtime=runtime._replace(geometry=committed.geometry),
        plan=plan,
        acknowledgement=committed.acknowledgement,
    )


def commit_region_block_mutation(
    geometry: RegionGeometryState,
    semantic_atlas: RegionBlockSemanticAtlas,
    update: MutableBlockUpdate,
) -> RegionBlockMutationCommit:
    """Commit exact values and bind them into every physical Region query."""

    if not isinstance(update, MutableBlockUpdate):
        raise TypeError("update must be a MutableBlockUpdate")
    state = _bound_state(geometry)
    immutable = query_region_block_action_target(
        semantic_atlas,
        geometry.atlas,
        update.position,
        geometry.environment_world_id,
        require_core=True,
    )
    canonical = jnp.all(
        immutable.action_position == update.position,
        axis=2,
    )
    base = immutable.query
    next_state, result = apply_mutable_block_updates(
        state,
        base_available=base.available & canonical,
        base_geometry=base.geometry,
        base_block_health=base.block_health,
        update=update,
    )
    acknowledgement = exact_block_mutation_acknowledgement(result)
    return RegionBlockMutationCommit(
        geometry=geometry._replace(mutable_blocks=next_state),
        state=next_state,
        acknowledgement=acknowledgement,
    )


def commit_native_mutable_block_cells(
    runtime: RegionActionRuntimeState,
    capture,
    *,
    expected_bridge_sha256: str,
) -> RegionBlockMutationCommit:
    """Atomically refresh one live environment from bridge-certified cells."""

    from hytalegym.worldgen.native_mutable_blocks import (  # cycle-safe
        NativeMutableBlockCells,
    )

    if not isinstance(runtime, RegionActionRuntimeState):
        raise TypeError("runtime must be a RegionActionRuntimeState")
    if not isinstance(capture, NativeMutableBlockCells):
        raise TypeError("capture must be NativeMutableBlockCells")
    if (
        not isinstance(expected_bridge_sha256, str)
        or len(expected_bridge_sha256) != 64
        or any(
            character not in "0123456789abcdefABCDEF"
            for character in expected_bridge_sha256
        )
    ):
        raise ValueError("expected_bridge_sha256 must be one SHA-256")
    if capture.bridge_sha256 != expected_bridge_sha256.upper():
        raise ValueError("mutable-cell capture came from a stale bridge")
    if not all(cell.available for cell in capture.cells):
        raise ValueError(
            "mutable-cell synchronization cannot publish unavailable cells"
        )
    state = _bound_state(runtime.geometry)
    if state.world_id.shape[0] != 1:
        raise ValueError(
            "one native mutable-cell capture requires one Region batch row"
        )
    cells = mutable_block_geometry_from_native_cells(capture)
    positions = cells.positions[None, ...]
    immutable = query_region_block_action_target(
        runtime.semantic_atlas,
        runtime.geometry.atlas,
        positions,
        runtime.geometry.environment_world_id,
        require_core=True,
    )
    canonical = jnp.all(
        immutable.action_position == positions,
        axis=2,
    )
    current = _query_current(
        runtime.geometry,
        state,
        positions,
        immutable.query._replace(
            available=immutable.query.available & canonical,
        ),
    )
    available = cells.available[None, ...] & current.available
    geometry = jax.tree_util.tree_map(
        lambda value: value[None, ...],
        cells.geometry,
    )
    update = empty_mutable_block_update(
        state,
        query_capacity=len(capture.cells),
    )._replace(
        mask=available,
        position=positions,
        geometry_changed=available,
        health_changed=available,
        expected_health=current.block_health,
        block_health_after=cells.block_health[None, ...],
        provenance=jnp.where(
            available,
            jnp.uint8(MUTABLE_BLOCK_PROVENANCE_NATIVE),
            jnp.uint8(0),
        ),
        geometry=geometry,
    )
    return commit_region_block_mutation(
        runtime.geometry,
        runtime.semantic_atlas,
        update,
    )


def region_action_runtime_contract() -> dict[str, object]:
    return {
        "schema": REGION_ACTION_RUNTIME_SCHEMA,
        "version": REGION_ACTION_RUNTIME_VERSION,
        "mutable_block_contract_sha256": mutable_block_contract_sha256(),
        "region_semantic_contract_sha256": (
            region_block_semantic_atlas_contract_sha256()
        ),
        "base": "frozen_offline_complete_region_snapshot",
        "candidate_source": (
            "dense_complete_reach_filler_canonicalized_actor_fov_and_los"
        ),
        "execution": (
            "policy_slot_gather_then_exact_position_requery_predicate_"
            "resolution_and_revision_guarded_atomic_mutable_update"
        ),
        "selection": (
            "one_bounded_candidate_per_actor_with_index_position_and_"
            "current_state_validation"
        ),
        "physical_projection": [
            "collision",
            "line_of_sight",
            "projectile_contact",
            "movement_medium",
            "collision_exception_tokens",
        ],
        "traversal_after_mutation": (
            "static_graph_requires_revalidation_or_must_be_withheld"
        ),
        "unsupported": [
            "partial_filler_footprint_mutation",
            "transparent_placed_block_role_opacity_without_explicit_evidence",
            "unacknowledged_fluid_or_support_cascade",
        ],
        "fail_closed": True,
    }


def region_action_runtime_contract_sha256() -> str:
    payload = json.dumps(
        region_action_runtime_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _query_current(
    geometry: RegionGeometryState,
    state: MutableBlockState,
    position: Array,
    base: MutableBlockQueryResult,
) -> MutableBlockQueryResult:
    return query_mutable_blocks(
        state,
        world_id=geometry.environment_world_id,
        base_semantic_sha256=state.base_semantic_sha256,
        resync_epoch=state.resync_epoch,
        position=position,
        base_available=base.available,
        base_geometry=base.geometry,
        base_block_health=base.block_health,
    )


def _gather_actor_candidate(value: Array, index: Array) -> Array:
    array = jnp.asarray(value)
    if array.ndim < 3 or array.shape[:2] != index.shape:
        raise ValueError("candidate value must have shape [batch, actor, ...]")
    gather = index[..., None]
    gather = gather.reshape(
        gather.shape + (1,) * (array.ndim - gather.ndim)
    )
    gather = jnp.broadcast_to(
        gather,
        array.shape[:2] + (1,) + array.shape[3:],
    )
    return jnp.take_along_axis(array, gather, axis=2)[:, :, 0]


def _bound_state(geometry: RegionGeometryState) -> MutableBlockState:
    if not isinstance(geometry, RegionGeometryState):
        raise TypeError("geometry must be a RegionGeometryState")
    state = geometry.mutable_blocks
    if not isinstance(state, MutableBlockState):
        raise ValueError("Region geometry has no mutable block state")
    return state


def _host_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    return int(value)


def _require_snapshot_matches_atlas(
    geometry: RegionGeometryState,
    snapshot: NativeRegionSnapshot,
    world_id: int,
) -> None:
    atlas = geometry.atlas
    active = np.asarray(jax.device_get(atlas.region_mask))
    atlas_world = np.asarray(jax.device_get(atlas.world_id))
    core = np.asarray(jax.device_get(atlas.core_min_chunk_xz))
    matches = (
        active
        & (atlas_world == world_id)
        & np.all(core == snapshot.core_min_chunk_xz, axis=1)
    )
    if np.count_nonzero(matches) != 1:
        raise ValueError(
            "snapshot does not identify exactly one active Region atlas row"
        )
    row = int(np.flatnonzero(matches)[0])
    cell_count = int(
        np.asarray(jax.device_get(atlas.cell_palette_size))[row]
    )
    shape_count = int(
        np.asarray(jax.device_get(atlas.shape_palette_size))[row]
    )
    comparisons = (
        (
            np.asarray(jax.device_get(atlas.section_known))[row],
            snapshot.section_known,
        ),
        (
            np.asarray(jax.device_get(atlas.cell_code))[row],
            snapshot.cell_code,
        ),
        (
            np.asarray(jax.device_get(atlas.filler_root_known))[row],
            snapshot.filler_root_offsets_available,
        ),
        (
            np.asarray(jax.device_get(atlas.filler_root_offset_packed))[row],
            snapshot.filler_root_offset_packed,
        ),
        (
            np.asarray(jax.device_get(atlas.cell_flags))[row, :cell_count],
            snapshot.cell_palette.flags,
        ),
        (
            np.asarray(jax.device_get(atlas.cell_shape_index))[
                row, :cell_count
            ],
            snapshot.cell_palette.shape_index,
        ),
        (
            np.asarray(jax.device_get(atlas.cell_fluid_level))[
                row, :cell_count
            ],
            snapshot.cell_palette.fluid_level,
        ),
        (
            np.asarray(jax.device_get(atlas.cell_fluid_fill_height))[
                row, :cell_count
            ],
            snapshot.cell_palette.fluid_fill_height,
        ),
        (
            np.asarray(jax.device_get(atlas.cell_support))[row, :cell_count],
            snapshot.cell_palette.support,
        ),
        (
            np.asarray(jax.device_get(atlas.cell_block_damage))[
                row, :cell_count
            ],
            snapshot.cell_palette.block_damage,
        ),
        (
            np.asarray(jax.device_get(atlas.cell_fluid_damage))[
                row, :cell_count
            ],
            snapshot.cell_palette.fluid_damage,
        ),
        (
            np.asarray(jax.device_get(atlas.cell_movement))[
                row, :cell_count
            ],
            snapshot.cell_palette.movement,
        ),
        (
            np.asarray(jax.device_get(atlas.cell_fluid_movement))[
                row, :cell_count
            ],
            snapshot.cell_palette.fluid_movement,
        ),
        (
            np.asarray(jax.device_get(atlas.collision_boxes))[
                row, :shape_count
            ],
            snapshot.shape_palette.boxes,
        ),
        (
            np.asarray(jax.device_get(atlas.collision_box_mask))[
                row, :shape_count
            ],
            snapshot.shape_palette.box_mask,
        ),
    )
    if (
        cell_count != snapshot.cell_palette.size
        or shape_count != snapshot.shape_palette.size
        or any(
            not np.array_equal(actual, expected)
            for actual, expected in comparisons
        )
    ):
        raise ValueError("frozen snapshot differs from the Region atlas base")


def _positive_int(value: object, label: str) -> int:
    result = _host_int(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_int(value: object, label: str) -> int:
    result = _host_int(value, label)
    if result < 0:
        raise ValueError(f"{label} must be nonnegative")
    return result


__all__ = [
    "REGION_ACTION_RUNTIME_SCHEMA",
    "REGION_ACTION_RUNTIME_VERSION",
    "REGION_ACTION_SELECTION_DIAGNOSTIC_CANDIDATE",
    "REGION_ACTION_SELECTION_DIAGNOSTIC_INDEX",
    "REGION_ACTION_SELECTION_DIAGNOSTIC_POSITION",
    "RegionActionRuntimeState",
    "RegionBlockActionExecution",
    "RegionBlockActionPlan",
    "RegionBlockMutationCommit",
    "RegionSelectedBlockActionTarget",
    "commit_region_block_mutation",
    "commit_native_mutable_block_cells",
    "execute_region_runtime_block_action",
    "mutable_block_state_from_region_snapshots",
    "plan_region_runtime_block_action",
    "produce_region_actor_block_action_candidates",
    "produce_region_runtime_actor_block_action_candidates",
    "region_action_runtime_contract",
    "region_action_runtime_contract_sha256",
    "requery_region_block_action_candidates",
    "select_region_block_action_target",
]
