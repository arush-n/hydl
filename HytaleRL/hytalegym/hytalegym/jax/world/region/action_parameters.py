"""Exact candidate-aligned Region block-action parameter publication."""

from __future__ import annotations

import hashlib
import json

from hytalegym.jax.world.block_actions import (
    BlockActionCatalog,
    BlockDropProgramCatalog,
    BlockGatherDefaults,
    BlockToolState,
    resolve_block_action_targets,
)
from hytalegym.jax.world.interactions.blocks.tokens import (
    ActorBlockActionCandidates,
    ActorBlockActionParameters,
    actor_block_action_parameter_contract_sha256,
    publish_actor_block_action_parameters,
)
from hytalegym.jax.world.region.action_runtime import (
    RegionActionRuntimeState,
    region_action_runtime_contract_sha256,
    requery_region_block_action_candidates,
)


REGION_ACTOR_BLOCK_PARAMETER_SCHEMA = (
    "hytalerl_region_actor_block_action_parameters_v1"
)
REGION_ACTOR_BLOCK_PARAMETER_VERSION = 1


def produce_region_runtime_actor_block_action_parameters(
    runtime: RegionActionRuntimeState,
    candidates: ActorBlockActionCandidates,
    tool: BlockToolState,
    defaults: BlockGatherDefaults,
    catalog: BlockActionCatalog,
    *,
    modification_evidence_available,
    modification_allowed,
    place_evidence_available,
    place_allowed,
    break_interaction_tool_type_id=None,
    break_interaction_match_tool=None,
    drop_programs: BlockDropProgramCatalog | None = None,
    drop_random_samples=None,
) -> ActorBlockActionParameters:
    """Requery and resolve every bounded candidate from current World state.

    Tool and permission inputs use the flattened actor-major candidate shape
    ``[B, A*C]``. The result keeps candidate alignment so a downstream
    selection can be rechecked without reconstructing replacement semantics.
    """

    if not isinstance(runtime, RegionActionRuntimeState):
        raise TypeError("runtime must be a RegionActionRuntimeState")
    if not isinstance(candidates, ActorBlockActionCandidates):
        raise TypeError("candidates must be ActorBlockActionCandidates")
    current = requery_region_block_action_candidates(
        runtime.geometry,
        runtime.semantic_atlas,
        candidates,
    )
    actions = resolve_block_action_targets(
        current,
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
    return publish_actor_block_action_parameters(
        candidates,
        candidates.action_position,
        current,
        actions,
    )


def region_actor_block_action_parameter_contract() -> dict[str, object]:
    """Return the Region composition contract for actor block parameters."""

    return {
        "schema": REGION_ACTOR_BLOCK_PARAMETER_SCHEMA,
        "version": REGION_ACTOR_BLOCK_PARAMETER_VERSION,
        "dependencies": {
            "region_action_runtime_contract_sha256": (
                region_action_runtime_contract_sha256()
            ),
            "actor_block_action_parameter_contract_sha256": (
                actor_block_action_parameter_contract_sha256()
            ),
        },
        "ordering": "actor_major_candidate_minor",
        "requery": (
            "current_mutable_revision_and_exact_filler_canonical_position"
        ),
        "tool_and_permission_shape": ["batch", "actor_times_candidate"],
        "published_exact_fields": [
            "block_health",
            "tool_legality_and_power",
            "interaction_tool_route_matched",
            "break_removes_block",
            "replacement_semantic_key",
            "place_legality",
            "resolved_drops_and_quantities",
        ],
        "consumer": (
            "downstream_selects_bounded_slot_then_rechecks_before_commit"
        ),
        "failure": (
            "stale_position_query_action_or_capacity_clears_complete_actor_row"
        ),
    }


def region_actor_block_action_parameter_contract_sha256() -> str:
    payload = json.dumps(
        region_actor_block_action_parameter_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "REGION_ACTOR_BLOCK_PARAMETER_SCHEMA",
    "REGION_ACTOR_BLOCK_PARAMETER_VERSION",
    "produce_region_runtime_actor_block_action_parameters",
    "region_actor_block_action_parameter_contract",
    "region_actor_block_action_parameter_contract_sha256",
]
