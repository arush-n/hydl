"""Current-revision Region adapter for exact actor placement candidates."""

from __future__ import annotations

import hashlib
import json

import jax.numpy as jnp

from hytalegym.jax.world.interactions.blocks.candidates import (
    ActorBlockPlacementCandidates,
    actor_block_placement_candidate_contract_sha256,
    actor_block_placement_query,
    publish_actor_block_placement_candidates,
)
from hytalegym.jax.world.interactions.blocks.tokens import ActorBlockActionCandidates
from hytalegym.jax.world.mutable_blocks import (
    MutableBlockState,
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


REGION_ACTOR_BLOCK_PLACEMENT_SCHEMA = (
    "hytalerl_region_actor_block_placement_candidates_v1"
)
REGION_ACTOR_BLOCK_PLACEMENT_VERSION = 1


def produce_region_runtime_actor_block_placement_candidates(
    runtime: RegionActionRuntimeState,
    candidates: ActorBlockActionCandidates,
    *,
    candidate_capacity: int,
) -> ActorBlockPlacementCandidates:
    """Requery all adjacent destination cells from one persistent Region.

    The exact physical and semantic Region state used by collision supplies
    the immutable base and mutable overlay. Canonical-position drift remains
    visible to the generic publisher and therefore clears the actor row.
    """

    if not isinstance(runtime, RegionActionRuntimeState):
        raise TypeError("runtime must be a RegionActionRuntimeState")
    if not isinstance(candidates, ActorBlockActionCandidates):
        raise TypeError("candidates must be ActorBlockActionCandidates")
    state = runtime.geometry.mutable_blocks
    if not isinstance(state, MutableBlockState):
        raise ValueError("Region runtime has no mutable block state")

    query = actor_block_placement_query(candidates)
    batch, actors, expanded_capacity, _ = query.destination_position.shape
    flat_position = query.destination_position.reshape(
        batch,
        actors * expanded_capacity,
        3,
    )
    immutable = query_region_block_action_target(
        runtime.semantic_atlas,
        runtime.geometry.atlas,
        flat_position,
        runtime.geometry.environment_world_id,
        require_core=True,
    )
    position_match = jnp.all(
        immutable.action_position == flat_position,
        axis=2,
    )
    base = immutable.query._replace(
        available=immutable.query.available & position_match,
    )
    current = query_mutable_blocks(
        state,
        world_id=runtime.geometry.environment_world_id,
        base_semantic_sha256=state.base_semantic_sha256,
        resync_epoch=state.resync_epoch,
        position=flat_position,
        base_available=base.available,
        base_geometry=base.geometry,
        base_block_health=base.block_health,
    )
    return publish_actor_block_placement_candidates(
        candidates,
        immutable.action_position.reshape(
            batch,
            actors,
            expanded_capacity,
            3,
        ),
        current,
        candidate_capacity=candidate_capacity,
    )


def region_actor_block_placement_candidate_contract() -> dict[str, object]:
    """Return the exact Region-to-actor placement composition contract."""

    return {
        "schema": REGION_ACTOR_BLOCK_PLACEMENT_SCHEMA,
        "version": REGION_ACTOR_BLOCK_PLACEMENT_VERSION,
        "dependencies": {
            "actor_block_placement_candidate_contract_sha256": (
                actor_block_placement_candidate_contract_sha256()
            ),
            "region_action_runtime_contract_sha256": (
                region_action_runtime_contract_sha256()
            ),
            "region_block_semantic_atlas_contract_sha256": (
                region_block_semantic_atlas_contract_sha256()
            ),
        },
        "query": (
            "all_six_face_destinations_against_same_physical_semantic_and_"
            "mutable_region_runtime"
        ),
        "position_echo": (
            "immutable_filler_canonical_position_must_equal_requested_"
            "destination"
        ),
        "publication": (
            "generic_exact_empty_destination_and_protocol_face_contract"
        ),
        "failure": (
            "missing_mutable_state_unknown_cell_position_drift_or_capacity_"
            "overflow_fails_closed"
        ),
    }


def region_actor_block_placement_candidate_contract_sha256() -> str:
    payload = json.dumps(
        region_actor_block_placement_candidate_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "REGION_ACTOR_BLOCK_PLACEMENT_SCHEMA",
    "REGION_ACTOR_BLOCK_PLACEMENT_VERSION",
    "produce_region_runtime_actor_block_placement_candidates",
    "region_actor_block_placement_candidate_contract",
    "region_actor_block_placement_candidate_contract_sha256",
]
