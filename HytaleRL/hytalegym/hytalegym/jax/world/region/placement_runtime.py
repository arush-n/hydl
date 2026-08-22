"""Face-correct Region Place selection and exact mutation execution."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.world.block_actions import (
    BlockActionCatalog,
    BlockActionMutationPlan,
    BlockActionTargetResult,
    BlockDropProgramCatalog,
    BlockGatherDefaults,
    BlockMutationAcknowledgement,
    BlockToolState,
    plan_exact_block_action_mutation,
    resolve_block_action_targets,
)
from hytalegym.jax.world.interactions.blocks.candidates import (
    ActorBlockPlacementCandidates,
    actor_block_placement_candidate_contract_sha256,
)
from hytalegym.jax.world.interactions.blocks.tokens import ActorBlockActionCandidates
from hytalegym.jax.world.mutable_blocks import (
    MutableBlockGeometry,
    MutableBlockQueryResult,
    MutableBlockState,
    query_mutable_blocks,
)
from hytalegym.jax.world.region.action_runtime import (
    RegionActionRuntimeState,
    RegionSelectedBlockActionTarget,
    commit_region_block_mutation,
    region_action_runtime_contract_sha256,
    select_region_block_action_target,
)
from hytalegym.jax.world.region.block_semantics import (
    query_region_block_action_target,
)


Array = jax.Array

REGION_PLACEMENT_ACTION_RUNTIME_SCHEMA = "hytalerl_region_placement_action_runtime_v1"
REGION_PLACEMENT_ACTION_RUNTIME_VERSION = 1

REGION_PLACEMENT_SELECTION_DIAGNOSTIC_FACE = 1 << 0
REGION_PLACEMENT_SELECTION_DIAGNOSTIC_CANDIDATE = 1 << 1
REGION_PLACEMENT_SELECTION_DIAGNOSTIC_DESTINATION = 1 << 2


class RegionSelectedBlockPlacementTarget(NamedTuple):
    """One clicked face and its exact current empty destination."""

    available: Array
    clicked_position: Array
    destination_position: Array
    block_face: Array
    source_provenance: Array
    destination_provenance: Array
    query: MutableBlockQueryResult
    diagnostics: Array


class RegionBlockPlacementActionPlan(NamedTuple):
    """Base block predicates plus the exact destination used by Place."""

    base_selection: RegionSelectedBlockActionTarget
    placement: RegionSelectedBlockPlacementTarget
    actions: BlockActionTargetResult
    mutation: BlockActionMutationPlan


class RegionBlockPlacementActionExecution(NamedTuple):
    """One Break/Place action with face-correct Place destination semantics."""

    runtime: RegionActionRuntimeState
    plan: RegionBlockPlacementActionPlan
    acknowledgement: BlockMutationAcknowledgement


def select_region_block_placement_target(
    runtime: RegionActionRuntimeState,
    placements: ActorBlockPlacementCandidates,
    clicked_position: Array,
    block_face: Array,
) -> RegionSelectedBlockPlacementTarget:
    """Select one published face and requery its destination before mutation."""

    if not isinstance(runtime, RegionActionRuntimeState):
        raise TypeError("runtime must be a RegionActionRuntimeState")
    if not isinstance(placements, ActorBlockPlacementCandidates):
        raise TypeError("placements must be ActorBlockPlacementCandidates")
    state = _bound_state(runtime)
    batch, actors, _ = placements.candidate_mask.shape
    clicked = jnp.asarray(clicked_position)
    if clicked.shape != (batch, actors, 3):
        raise ValueError("clicked_position must have shape [batch, actor, 3]")
    if not jnp.issubdtype(clicked.dtype, jnp.integer):
        raise TypeError("clicked_position must be integer-valued")
    clicked = clicked.astype(jnp.int32)
    raw_face = jnp.asarray(block_face)
    if raw_face.shape != (batch, actors):
        raise ValueError("block_face must have shape [batch, actor]")
    if not jnp.issubdtype(raw_face.dtype, jnp.integer):
        raise TypeError("block_face must be integer-valued")
    face = raw_face.astype(jnp.int32)
    face_valid = (face >= 1) & (face <= 6)
    matches = (
        placements.candidate_mask
        & placements.available[..., None]
        & jnp.all(
            placements.clicked_position == clicked[..., None, :],
            axis=3,
        )
        & (placements.block_face.astype(jnp.int32) == face[..., None])
    )
    match_count = jnp.sum(matches, axis=2)
    candidate = face_valid & (match_count == 1)
    index = jnp.argmax(matches, axis=2).astype(jnp.int32)
    destination = _gather_candidate(placements.destination_position, index)
    source_provenance = _gather_candidate(
        placements.source_provenance,
        index,
    )
    destination_provenance = _gather_candidate(
        placements.destination_provenance,
        index,
    )
    immutable = query_region_block_action_target(
        runtime.semantic_atlas,
        runtime.geometry.atlas,
        destination,
        runtime.geometry.environment_world_id,
        require_core=True,
    )
    position_match = jnp.all(
        immutable.action_position == destination,
        axis=2,
    )
    base = immutable.query._replace(
        available=immutable.query.available & candidate & position_match,
    )
    current = query_mutable_blocks(
        state,
        world_id=runtime.geometry.environment_world_id,
        base_semantic_sha256=state.base_semantic_sha256,
        resync_epoch=state.resync_epoch,
        position=destination,
        base_available=base.available,
        base_geometry=base.geometry,
        base_block_health=base.block_health,
    )
    destination_ready = (
        current.available
        & current.geometry.exact
        & ~current.geometry.block_present
        & ~current.invalid
        & ~current.resync_required
    )
    available = candidate & position_match & destination_ready
    diagnostics = (
        jnp.where(
            ~face_valid,
            jnp.uint32(REGION_PLACEMENT_SELECTION_DIAGNOSTIC_FACE),
            jnp.uint32(0),
        )
        | jnp.where(
            face_valid & (match_count != 1),
            jnp.uint32(REGION_PLACEMENT_SELECTION_DIAGNOSTIC_CANDIDATE),
            jnp.uint32(0),
        )
        | jnp.where(
            candidate & (~position_match | ~destination_ready),
            jnp.uint32(REGION_PLACEMENT_SELECTION_DIAGNOSTIC_DESTINATION),
            jnp.uint32(0),
        )
    )

    def masked(value: Array) -> Array:
        gate = available.reshape(available.shape + (1,) * (value.ndim - available.ndim))
        return jnp.where(gate, value, jnp.zeros_like(value))

    return RegionSelectedBlockPlacementTarget(
        available=available,
        clicked_position=masked(clicked),
        destination_position=masked(destination),
        block_face=jnp.where(available, face.astype(jnp.uint8), jnp.uint8(0)),
        source_provenance=jnp.where(
            available,
            source_provenance,
            jnp.uint8(0),
        ),
        destination_provenance=jnp.where(
            available,
            destination_provenance,
            jnp.uint8(0),
        ),
        query=current._replace(available=current.available & available),
        diagnostics=diagnostics,
    )


def plan_region_runtime_block_action_with_placement(
    runtime: RegionActionRuntimeState,
    candidates: ActorBlockActionCandidates,
    placements: ActorBlockPlacementCandidates,
    selected_candidate_index: Array,
    selected_block_face: Array,
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
) -> RegionBlockPlacementActionPlan:
    """Plan Break at the selected block or Place at its selected face."""

    if not isinstance(runtime, RegionActionRuntimeState):
        raise TypeError("runtime must be a RegionActionRuntimeState")
    base_selection = select_region_block_action_target(
        runtime.geometry,
        runtime.semantic_atlas,
        candidates,
        selected_candidate_index,
    )
    # Native placement starts from the block cell intersected by the camera
    # ray, while Break/Use resolve the filler-canonical root.  The policy row
    # deliberately carries both identities.  Do not feed the canonical action
    # position back into the placement table: filler hits are keyed by their
    # raw clicked cell and would otherwise fail closed despite a valid face.
    raw_index = jnp.asarray(selected_candidate_index, dtype=jnp.int32)
    safe_index = jnp.clip(raw_index, 0, candidates.candidate_mask.shape[2] - 1)
    clicked_position = _gather_candidate(
        candidates.visible_position,
        safe_index,
    )
    clicked_position = jnp.where(
        base_selection.available[..., None],
        clicked_position,
        jnp.zeros_like(clicked_position),
    )
    actions = resolve_block_action_targets(
        base_selection.query,
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
    placement = select_region_block_placement_target(
        runtime,
        placements,
        clicked_position,
        selected_block_face,
    )
    select_place = jnp.asarray(place_selected)
    if select_place.shape != base_selection.available.shape:
        raise ValueError("place_selected must have shape [batch, actor]")
    if select_place.dtype != jnp.dtype(jnp.bool_):
        raise TypeError("place_selected must have dtype bool")
    mutation_target = _select_tree(
        select_place,
        placement.query,
        base_selection.query,
    )
    mutation_position = jnp.where(
        select_place[..., None],
        placement.destination_position,
        base_selection.position,
    )
    mutation_provenance = jnp.where(
        select_place,
        placement.destination_provenance,
        base_selection.provenance,
    )
    mutation = plan_exact_block_action_mutation(
        _bound_state(runtime),
        mutation_target,
        actions,
        position=mutation_position,
        break_selected=break_selected,
        place_selected=select_place,
        placed_geometry=placed_geometry,
        replacement_geometry=replacement_geometry,
        provenance=mutation_provenance,
    )
    return RegionBlockPlacementActionPlan(
        base_selection=base_selection,
        placement=placement,
        actions=actions,
        mutation=mutation,
    )


def execute_region_runtime_block_action_with_placement(
    runtime: RegionActionRuntimeState,
    candidates: ActorBlockActionCandidates,
    placements: ActorBlockPlacementCandidates,
    selected_candidate_index: Array,
    selected_block_face: Array,
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
) -> RegionBlockPlacementActionExecution:
    """Commit a face-correct Place or a canonical Break exactly once."""

    plan = plan_region_runtime_block_action_with_placement(
        runtime,
        candidates,
        placements,
        selected_candidate_index,
        selected_block_face,
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
    return RegionBlockPlacementActionExecution(
        runtime=runtime._replace(geometry=committed.geometry),
        plan=plan,
        acknowledgement=committed.acknowledgement,
    )


def region_placement_action_runtime_contract() -> dict[str, object]:
    """Return the additive face-correct Region Place execution contract."""

    return {
        "schema": REGION_PLACEMENT_ACTION_RUNTIME_SCHEMA,
        "version": REGION_PLACEMENT_ACTION_RUNTIME_VERSION,
        "dependencies": {
            "region_action_runtime_contract_sha256": (
                region_action_runtime_contract_sha256()
            ),
            "actor_block_placement_candidate_contract_sha256": (
                actor_block_placement_candidate_contract_sha256()
            ),
        },
        "selection": (
            "policy_block_slot_clicked_position_plus_actor_aim_protocol_face_"
            "must_identify_exactly_one_published_candidate"
        ),
        "place_destination": (
            "clicked_position_plus_protocol_face_offset_requeried_exact_empty_"
            "against_current_mutable_revision_before_commit"
        ),
        "break_destination": "selected_canonical_block_action_position",
        "native_alignment": (
            "PlaceBlockInteraction_clientState_blockPosition_is_destination_"
            "and_blockFace_is_carried_separately"
        ),
        "failure": (
            "invalid_face_missing_or_duplicate_candidate_stale_destination_"
            "occupied_destination_or_unknown_revision_fails_closed"
        ),
        "legacy_api": (
            "execute_region_runtime_block_action_retained_for_break_and_"
            "callers_without_place_destination_evidence"
        ),
    }


def region_placement_action_runtime_contract_sha256() -> str:
    payload = json.dumps(
        region_placement_action_runtime_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bound_state(runtime: RegionActionRuntimeState) -> MutableBlockState:
    state = runtime.geometry.mutable_blocks
    if not isinstance(state, MutableBlockState):
        raise ValueError("Region geometry has no mutable block state")
    return state


def _gather_candidate(value: Array, index: Array) -> Array:
    array = jnp.asarray(value)
    gather = index[..., None].reshape(index.shape + (1,) * (array.ndim - index.ndim))
    gather = jnp.broadcast_to(
        gather,
        array.shape[:2] + (1,) + array.shape[3:],
    )
    return jnp.take_along_axis(array, gather, axis=2)[:, :, 0]


def _select_tree(condition: Array, selected, fallback):
    def choose(selected_value: Array, fallback_value: Array) -> Array:
        gate = condition.reshape(
            condition.shape + (1,) * (selected_value.ndim - condition.ndim)
        )
        return jnp.where(gate, selected_value, fallback_value)

    return jax.tree.map(choose, selected, fallback)


__all__ = [
    "REGION_PLACEMENT_ACTION_RUNTIME_SCHEMA",
    "REGION_PLACEMENT_ACTION_RUNTIME_VERSION",
    "REGION_PLACEMENT_SELECTION_DIAGNOSTIC_CANDIDATE",
    "REGION_PLACEMENT_SELECTION_DIAGNOSTIC_DESTINATION",
    "REGION_PLACEMENT_SELECTION_DIAGNOSTIC_FACE",
    "RegionBlockPlacementActionExecution",
    "RegionBlockPlacementActionPlan",
    "RegionSelectedBlockPlacementTarget",
    "execute_region_runtime_block_action_with_placement",
    "plan_region_runtime_block_action_with_placement",
    "region_placement_action_runtime_contract",
    "region_placement_action_runtime_contract_sha256",
    "select_region_block_placement_target",
]
