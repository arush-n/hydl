"""Fail-closed block-action evidence over exact mutable block values."""

from __future__ import annotations

import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.mutable_blocks import (
    MUTABLE_BLOCK_PROVENANCE_NATIVE,
    MUTABLE_BLOCK_PROVENANCE_SURROGATE,
    MutableBlockGeometry,
    MutableBlockQueryResult,
    MutableBlockState,
    empty_mutable_block_geometry,
    empty_mutable_block_update,
    mutable_block_contract_sha256,
)
from hytalegym.worldgen.block_actions import (
    BLOCK_DROP_METADATA_HASH_WORDS,
    BLOCK_DROP_OUTPUT_CAPACITY,
    BLOCK_TOOL_SPEC_CAPACITY,
    block_action_contract_sha256 as host_block_action_contract_sha256,
)
from hytalegym.worldgen.block_affordances import (
    GATHER_TYPES,
    block_affordance_dictionary_sha256,
    block_affordance_tag_mask,
)
from hytalegym.worldgen.drop_programs import (
    BLOCK_DROP_PROGRAM_CAPACITY,
    BLOCK_DROP_PROGRAM_OUTCOME_CAPACITY,
    block_drop_program_contract_sha256,
)

# Re-exported to preserve this module's attribute surface.
from hytalegym.jax.world.block_actions._catalog import (  # noqa: F401
    BlockActionCatalog,
    BlockActionTargetResult,
    _device_drop_table,
    _gather_drop_table,
    _publish_drops,
    _select_drop_table,
    _tool_value_valid,
    _validate_defaults,
    _validate_drop_program_catalog,
    _validate_tool,
    block_drop_program_catalog_from_local,
    block_gather_defaults_from_local,
    block_tool_state_from_local,
    empty_block_tool_state,
    exact_block_mutation_acknowledgement,
    native_block_interaction_outcome_matches,
)
from hytalegym.jax.world.block_actions._primitives import (  # noqa: F401
    BlockActionMutationPlan,
    BlockDropProgramCatalog,
    BlockDropTable,
    BlockGatherDefaults,
    BlockMutationAcknowledgement,
    BlockResolvedDrops,
    BlockToolState,
    NativeBlockInteractionOutcome,
    _array,
    _capacity,
    _catalog_value,
    _exact_present_geometry,
    _host_drop_table,
    _install_drop_route,
    _prefix,
    _select_geometry,
    _validate_mutation_geometry,
    _validate_target,
)
from hytalegym.jax.world.block_actions._resolution import (  # noqa: F401
    _resolve_drop_program,
    _validate_action_result,
    _validate_catalog,
    block_action_catalog_from_local,
)


Array = jax.Array
BLOCK_ACTION_TARGET_SCHEMA = "hytalerl_block_action_target_v4"
BLOCK_ACTION_TARGET_VERSION = 4

BLOCK_ACTION_DIAGNOSTIC_INVALID = 1 << 0
BLOCK_ACTION_DIAGNOSTIC_TARGET_UNAVAILABLE = 1 << 1
BLOCK_ACTION_DIAGNOSTIC_CATALOG_MISSING = 1 << 2
BLOCK_ACTION_DIAGNOSTIC_CATALOG_DUPLICATE = 1 << 3
BLOCK_ACTION_DIAGNOSTIC_TOOL_UNAVAILABLE = 1 << 4
BLOCK_ACTION_DIAGNOSTIC_DROP_UNAVAILABLE = 1 << 5
BLOCK_ACTION_DIAGNOSTIC_PERMISSION_UNAVAILABLE = 1 << 6
BLOCK_ACTION_DIAGNOSTIC_PLACE_UNAVAILABLE = 1 << 7
BLOCK_ACTION_DIAGNOSTIC_INTERACTION_TOOL_MISMATCH = 1 << 8

BLOCK_ACTION_MUTATION_SCHEMA = "hytalerl_block_action_mutation_v1"
BLOCK_ACTION_MUTATION_VERSION = 1
BLOCK_ACTION_MUTATION_DIAGNOSTIC_SELECTION = 1 << 0
BLOCK_ACTION_MUTATION_DIAGNOSTIC_PREDICATE = 1 << 1
BLOCK_ACTION_MUTATION_DIAGNOSTIC_GEOMETRY = 1 << 2
BLOCK_ACTION_MUTATION_DIAGNOSTIC_PROVENANCE = 1 << 3

_TAG_BREAKABLE = np.uint16(block_affordance_tag_mask("breakable"))
_TAG_HARVESTABLE = np.uint16(block_affordance_tag_mask("harvestable"))
_TAG_SOFT = np.uint16(block_affordance_tag_mask("soft"))


def resolve_block_action_targets(
    target: MutableBlockQueryResult,
    tool: BlockToolState,
    defaults: BlockGatherDefaults,
    catalog: BlockActionCatalog,
    *,
    modification_evidence_available: Array,
    modification_allowed: Array,
    place_evidence_available: Array,
    place_allowed: Array,
    break_interaction_tool_type_id: Array | None = None,
    break_interaction_match_tool: Array | None = None,
    drop_programs: BlockDropProgramCatalog | None = None,
    drop_random_samples: Array | None = None,
) -> BlockActionTargetResult:
    """Resolve native-shaped target predicates without permissive defaults."""

    shape = target.available.shape
    _validate_target(target, shape)
    _validate_tool(tool, shape)
    _validate_defaults(defaults)
    _validate_catalog(catalog)
    permission_available = _array(
        modification_evidence_available,
        shape,
        jnp.bool_,
        "modification_evidence_available",
    )
    permission_allowed = _array(
        modification_allowed,
        shape,
        jnp.bool_,
        "modification_allowed",
    )
    place_known = _array(
        place_evidence_available,
        shape,
        jnp.bool_,
        "place_evidence_available",
    )
    place_value = _array(place_allowed, shape, jnp.bool_, "place_allowed")
    interaction_tool_type_id = (
        jnp.zeros(shape, dtype=jnp.int32)
        if break_interaction_tool_type_id is None
        else _array(
            break_interaction_tool_type_id,
            shape,
            jnp.int32,
            "break_interaction_tool_type_id",
        )
    )
    interaction_match_tool = (
        jnp.zeros(shape, dtype=jnp.bool_)
        if break_interaction_match_tool is None
        else _array(
            break_interaction_match_tool,
            shape,
            jnp.bool_,
            "break_interaction_match_tool",
        )
    )

    key = target.geometry.semantic_key
    matches = (
        catalog.entry_mask[None, None, :]
        & jnp.all(key[..., None, :] == catalog.semantic_key, axis=-1)
    )
    match_count = jnp.sum(matches, axis=-1)
    duplicate = match_count > 1
    found = match_count == 1
    slot = jnp.argmax(matches, axis=-1).astype(jnp.int32)
    target_present = target.available & target.geometry.block_present
    catalog_known = target_present & found & ~duplicate

    tool_route = _catalog_value(catalog.tool_breakable, slot)
    soft_route = _catalog_value(catalog.soft_breakable, slot)
    soft_weapon = _catalog_value(catalog.soft_weapon_breakable, slot)
    harvest_route = _catalog_value(catalog.harvestable, slot)
    authored_interaction_route = _catalog_value(
        catalog.interaction_tool_route_present,
        slot,
    )
    authored_interaction_type = _catalog_value(
        catalog.interaction_tool_route_type_id,
        slot,
    )
    interaction_route_match = (
        authored_interaction_route
        & (interaction_tool_type_id > 0)
        & (interaction_tool_type_id == authored_interaction_type)
    )
    interaction_match_gate = (
        ~interaction_match_tool | interaction_route_match
    )
    tags = target.geometry.affordance_tags
    affordance_consistent = (
        (((tags & _TAG_BREAKABLE) != 0) == (tool_route | soft_route))
        & (((tags & _TAG_SOFT) != 0) == soft_route)
        & (((tags & _TAG_HARVESTABLE) != 0) == harvest_route)
    )
    gather_index = target.geometry.gather_type_index.astype(jnp.int32)
    safe_gather = jnp.clip(gather_index, 0, len(GATHER_TYPES) - 1)
    spec_matches = (
        tool.spec_mask
        & (
            tool.gather_type_index.astype(jnp.int32)
            == gather_index[..., None]
        )
    )
    spec_count = jnp.sum(spec_matches, axis=-1)
    spec_slot = jnp.argmax(spec_matches, axis=-1).astype(jnp.int32)
    explicit_spec = spec_count == 1
    tool_value_valid = _tool_value_valid(tool)
    explicit_quality = jnp.take_along_axis(
        tool.quality,
        spec_slot[..., None],
        axis=-1,
    )[..., 0]
    explicit_power = jnp.take_along_axis(
        tool.power,
        spec_slot[..., None],
        axis=-1,
    )[..., 0]
    default_available = defaults.available[safe_gather]
    default_quality = defaults.quality[safe_gather]
    default_power = defaults.power[safe_gather]
    selected_quality = jnp.where(
        tool.tool_present,
        explicit_quality,
        default_quality,
    )
    selected_power = jnp.where(
        tool.tool_present,
        explicit_power,
        default_power,
    )
    selected_spec = jnp.where(
        tool.tool_present,
        explicit_spec,
        default_available,
    )
    tool_compatible = (
        tool_route
        & ~tool.weapon
        & ~tool.builder_tool
        & selected_spec
        & (selected_quality >= target.geometry.required_tool_quality)
        & jnp.isfinite(selected_power)
        & (selected_power > 0.0)
    )
    soft_compatible = soft_route & (~tool.weapon | soft_weapon)
    use_tool = tool_compatible
    compatible = use_tool | (~use_tool & soft_compatible)
    ordinary_break_drop = _select_drop_table(
        use_tool,
        _gather_drop_table(catalog.tool_drop, slot),
        _gather_drop_table(catalog.soft_drop, slot),
    )
    interaction_break_drop = _gather_drop_table(
        catalog.interaction_tool_drop,
        slot,
    )
    selected_break_drop = _select_drop_table(
        interaction_route_match,
        interaction_break_drop,
        ordinary_break_drop,
    )
    selected_harvest_drop = _gather_drop_table(
        catalog.harvest_drop,
        slot,
    )
    samples_supplied = drop_random_samples is not None
    if samples_supplied:
        samples = _array(
            drop_random_samples,
            shape + (2,),
            jnp.float32,
            "drop_random_samples",
        )
    else:
        samples = jnp.zeros(shape + (2,), dtype=jnp.float32)
    selected_break_drop = _resolve_drop_program(
        selected_break_drop,
        drop_programs,
        samples[..., 0],
        samples_supplied=samples_supplied,
    )
    selected_harvest_drop = _resolve_drop_program(
        selected_harvest_drop,
        drop_programs,
        samples[..., 1],
        samples_supplied=samples_supplied,
    )

    base_known = (
        target_present
        & target.geometry.exact
        & target.geometry.semantic_key_valid
        & target.geometry.affordance_valid
        & catalog_known
        & affordance_consistent
        & tool_value_valid
    )
    break_available = (
        base_known
        & tool.available
        & permission_available
        & (
            ~(compatible & interaction_match_gate)
            | selected_break_drop.available
        )
    )
    break_allowed = (
        break_available
        & permission_allowed
        & compatible
        & interaction_match_gate
        & selected_break_drop.available
    )
    interaction_replacement_key = _catalog_value(
        catalog.interaction_tool_replacement_semantic_key,
        slot,
    )
    interaction_replacement_valid = _catalog_value(
        catalog.interaction_tool_replacement_valid,
        slot,
    )
    changes_state = (
        break_allowed
        & interaction_route_match
        & interaction_replacement_valid
        & jnp.any(interaction_replacement_key != key, axis=-1)
    )
    harvest_available = (
        base_known
        & permission_available
        & (~harvest_route | selected_harvest_drop.available)
    )
    harvest_allowed = (
        harvest_available
        & permission_allowed
        & harvest_route
        & selected_harvest_drop.available
    )
    place_available = target.available & permission_available & place_known
    place_result = (
        place_available & permission_allowed & place_value
    )
    invalid = (
        target.invalid
        | duplicate
        | (target_present & catalog_known & ~affordance_consistent)
        | ~tool_value_valid
    )
    drop_missing = base_known & (
        (
            compatible
            & interaction_match_gate
            & ~selected_break_drop.available
        )
        | (harvest_route & ~selected_harvest_drop.available)
    )
    diagnostics = (
        jnp.where(
            invalid,
            jnp.uint32(BLOCK_ACTION_DIAGNOSTIC_INVALID),
            jnp.uint32(0),
        )
        | jnp.where(
            ~target.available,
            jnp.uint32(BLOCK_ACTION_DIAGNOSTIC_TARGET_UNAVAILABLE),
            jnp.uint32(0),
        )
        | jnp.where(
            target_present & ~found,
            jnp.uint32(BLOCK_ACTION_DIAGNOSTIC_CATALOG_MISSING),
            jnp.uint32(0),
        )
        | jnp.where(
            duplicate,
            jnp.uint32(BLOCK_ACTION_DIAGNOSTIC_CATALOG_DUPLICATE),
            jnp.uint32(0),
        )
        | jnp.where(
            base_known & ~tool.available,
            jnp.uint32(BLOCK_ACTION_DIAGNOSTIC_TOOL_UNAVAILABLE),
            jnp.uint32(0),
        )
        | jnp.where(
            drop_missing,
            jnp.uint32(BLOCK_ACTION_DIAGNOSTIC_DROP_UNAVAILABLE),
            jnp.uint32(0),
        )
        | jnp.where(
            ~permission_available,
            jnp.uint32(BLOCK_ACTION_DIAGNOSTIC_PERMISSION_UNAVAILABLE),
            jnp.uint32(0),
        )
        | jnp.where(
            ~place_known,
            jnp.uint32(BLOCK_ACTION_DIAGNOSTIC_PLACE_UNAVAILABLE),
            jnp.uint32(0),
        )
        | jnp.where(
            base_known
            & tool.available
            & interaction_match_tool
            & ~interaction_route_match,
            jnp.uint32(
                BLOCK_ACTION_DIAGNOSTIC_INTERACTION_TOOL_MISMATCH
            ),
            jnp.uint32(0),
        )
    )
    return BlockActionTargetResult(
        target_available=target.available,
        diagnostics=diagnostics,
        catalog_index=jnp.where(catalog_known, slot, -1),
        break_available=break_available,
        break_allowed=break_allowed,
        tool_compatible=base_known & tool.available & compatible,
        tool_power=jnp.where(
            base_known & tool.available & compatible & use_tool,
            selected_power,
            jnp.where(
                base_known & tool.available & compatible,
                1.0,
                0.0,
            ),
        ),
        interaction_tool_route_matched=(
            base_known & interaction_route_match
        ),
        break_removes_block=break_allowed & ~changes_state,
        replacement_semantic_key_valid=changes_state,
        replacement_semantic_key=jnp.where(
            changes_state[..., None],
            interaction_replacement_key,
            jnp.zeros_like(interaction_replacement_key),
        ),
        harvest_available=harvest_available,
        harvest_allowed=harvest_allowed,
        target_harvestable=base_known & harvest_route,
        place_available=place_available,
        place_allowed=place_result,
        break_drops=_publish_drops(break_allowed, selected_break_drop),
        harvest_drops=_publish_drops(
            harvest_allowed,
            selected_harvest_drop,
        ),
    )


def plan_exact_block_action_mutation(
    state: MutableBlockState,
    target: MutableBlockQueryResult,
    actions: BlockActionTargetResult,
    *,
    position: Array,
    break_selected: Array,
    place_selected: Array,
    placed_geometry: MutableBlockGeometry,
    replacement_geometry: MutableBlockGeometry,
    provenance: Array,
) -> BlockActionMutationPlan:
    """Build the exact geometry-valued write for one selected World verb.

    Removal uses canonical exact air. A state-changing native tool route must
    supply geometry whose semantic key matches the route result. Placement
    likewise requires an exact, present geometry row; semantic resolution is
    never inferred from the target being replaced.
    """

    if not isinstance(state, MutableBlockState):
        raise TypeError("state must be MutableBlockState")
    if not isinstance(target, MutableBlockQueryResult):
        raise TypeError("target must be MutableBlockQueryResult")
    if not isinstance(actions, BlockActionTargetResult):
        raise TypeError("actions must be BlockActionTargetResult")
    if not isinstance(placed_geometry, MutableBlockGeometry):
        raise TypeError("placed_geometry must be MutableBlockGeometry")
    if not isinstance(replacement_geometry, MutableBlockGeometry):
        raise TypeError("replacement_geometry must be MutableBlockGeometry")
    shape = target.available.shape
    if len(shape) != 2 or state.world_id.shape != (shape[0],):
        raise ValueError("target must have shape [batch, query]")
    _validate_target(target, shape)
    _validate_action_result(actions, shape)
    _validate_mutation_geometry(placed_geometry, shape, "placed_geometry")
    _validate_mutation_geometry(
        replacement_geometry,
        shape,
        "replacement_geometry",
    )
    positions = _array(position, shape + (3,), jnp.int32, "position")
    select_break = _array(
        break_selected,
        shape,
        jnp.bool_,
        "break_selected",
    )
    select_place = _array(
        place_selected,
        shape,
        jnp.bool_,
        "place_selected",
    )
    source = _array(provenance, shape, jnp.uint8, "provenance")
    selected = select_break | select_place
    selection_valid = ~(select_break & select_place)
    provenance_valid = (
        (source == MUTABLE_BLOCK_PROVENANCE_NATIVE)
        | (source == MUTABLE_BLOCK_PROVENANCE_SURROGATE)
    )
    placed_ready = _exact_present_geometry(placed_geometry)
    replacement_key_matches = (
        replacement_geometry.semantic_key_valid
        & actions.replacement_semantic_key_valid
        & jnp.all(
            replacement_geometry.semantic_key
            == actions.replacement_semantic_key,
            axis=-1,
        )
    )
    replacement_ready = (
        _exact_present_geometry(replacement_geometry)
        & replacement_key_matches
    )
    break_geometry_ready = (
        actions.break_removes_block | replacement_ready
    )
    predicate_valid = (
        (~select_break | actions.break_allowed)
        & (~select_place | actions.place_allowed)
    )
    geometry_ready = (
        (~select_break | break_geometry_ready)
        & (~select_place | placed_ready)
    )
    available = (
        selected
        & selection_valid
        & predicate_valid
        & geometry_ready
        & provenance_valid
        & target.available
        & target.geometry.exact
    )
    air = empty_mutable_block_geometry(shape, exact=True)
    break_geometry = _select_geometry(
        actions.break_removes_block,
        air,
        replacement_geometry,
    )
    after_geometry = _select_geometry(
        select_place,
        placed_geometry,
        break_geometry,
    )
    present_after = after_geometry.block_present
    update = empty_mutable_block_update(
        state,
        query_capacity=shape[1],
    )._replace(
        mask=available,
        position=positions,
        geometry_changed=available,
        health_changed=available,
        expected_health=target.block_health,
        block_health_after=jnp.where(
            available & present_after,
            jnp.float32(1.0),
            jnp.float32(0.0),
        ),
        provenance=jnp.where(
            available,
            source,
            jnp.uint8(0),
        ),
        geometry=_select_geometry(
            available,
            after_geometry,
            empty_mutable_block_geometry(shape),
        ),
    )
    diagnostics = (
        jnp.where(
            selected & ~selection_valid,
            jnp.uint32(BLOCK_ACTION_MUTATION_DIAGNOSTIC_SELECTION),
            jnp.uint32(0),
        )
        | jnp.where(
            selected & ~predicate_valid,
            jnp.uint32(BLOCK_ACTION_MUTATION_DIAGNOSTIC_PREDICATE),
            jnp.uint32(0),
        )
        | jnp.where(
            selected & ~geometry_ready,
            jnp.uint32(BLOCK_ACTION_MUTATION_DIAGNOSTIC_GEOMETRY),
            jnp.uint32(0),
        )
        | jnp.where(
            selected & ~provenance_valid,
            jnp.uint32(BLOCK_ACTION_MUTATION_DIAGNOSTIC_PROVENANCE),
            jnp.uint32(0),
        )
    )
    return BlockActionMutationPlan(available, update, diagnostics)


def block_action_mutation_contract() -> dict[str, object]:
    return {
        "schema": BLOCK_ACTION_MUTATION_SCHEMA,
        "version": BLOCK_ACTION_MUTATION_VERSION,
        "target_contract_sha256": block_action_target_contract_sha256(),
        "mutable_block_contract_sha256": mutable_block_contract_sha256(),
        "break": {
            "ordinary": "canonical_exact_air",
            "tool_state_route": (
                "exact_geometry_with_matching_replacement_semantic_key"
            ),
        },
        "place": "caller_supplied_exact_present_geometry",
        "commit": "revision_bound_MutableBlockUpdate",
        "provenance": "explicit_native_or_surrogate_per_cell",
        "fail_closed": True,
    }


def block_action_mutation_contract_sha256() -> str:
    encoded = json.dumps(
        block_action_mutation_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def block_action_target_contract() -> dict[str, object]:
    """Return the checkpoint-pinnable neutral JAX producer contract."""

    return {
        "schema": BLOCK_ACTION_TARGET_SCHEMA,
        "version": BLOCK_ACTION_TARGET_VERSION,
        "host_contract_sha256": host_block_action_contract_sha256(),
        "mutable_block_contract_sha256": mutable_block_contract_sha256(),
        "affordance_dictionary_sha256": (
            block_affordance_dictionary_sha256()
        ),
        "shape": {
            "target": ["batch", "query"],
            "tool_specs": [
                "batch",
                "query",
                f"at_most_{BLOCK_TOOL_SPEC_CAPACITY}",
            ],
            "drop_outputs": [
                "batch",
                "query",
                f"at_most_{BLOCK_DROP_OUTPUT_CAPACITY}",
            ],
            "drop_metadata_hash": [
                "batch",
                "query",
                f"at_most_{BLOCK_DROP_OUTPUT_CAPACITY}",
                BLOCK_DROP_METADATA_HASH_WORDS,
            ],
            "drop_programs": [
                f"at_most_{BLOCK_DROP_PROGRAM_CAPACITY}",
                f"at_most_{BLOCK_DROP_PROGRAM_OUTCOME_CAPACITY}",
                f"at_most_{BLOCK_DROP_OUTPUT_CAPACITY}",
            ],
        },
        "break": {
            "damage": (
                "native_matching_tool_quality_or_Soft_fallback_with_"
                "explicit_runtime_permission"
            ),
            "interaction_tool_route": (
                "exact_positive_FNV1a_of_BreakBlockInteraction.Tool_"
                "matches_Gathering.Tools.Type"
            ),
            "match_tool": (
                "known_rejection_when_authored_exact_route_is_absent"
            ),
            "matched_state": (
                "publish_rotation_zero_replacement_semantic_key_only_when_"
                "different_from_target_otherwise_remove"
            ),
            "matched_drop": (
                "suppress_ordinary_break_drop_and_publish_route_drop"
            ),
        },
        "harvest": (
            "BlockGathering.Harvest_with_explicit_runtime_permission"
        ),
        "place": "explicit_acceptance_evidence_only_never_inferred",
        "drops": {
            "direct_or_default": "resolved_fixed_output",
            "program_contract_sha256": (
                block_drop_program_contract_sha256()
            ),
            "randomized_droplist": (
                "exact_bounded_distribution_with_explicit_half_open_sample"
            ),
            "native_same_seed_sample": "not_claimed_ThreadLocalRandom",
            "missing_named_asset": "known_native_empty_output",
            "partial_publication": False,
        },
        "mutation_acknowledgement": (
            "revision_guarded_exact_before_and_after_MutableBlockGeometry"
        ),
        "removal_cascade_safety": {
            "catalog_fields": [
                "support_dependent",
                "support_drop_type",
            ],
            "producer": (
                "region_removal_cascade_safety_positive_certificate"
            ),
            "unknown_or_support_dependent_neighbour": "fail_closed",
        },
        "fail_closed": [
            "target_or_base_unavailable",
            "semantic_catalog_missing_or_duplicate",
            "tool_evidence_unavailable",
            "tool_evidence_invalid_or_duplicate_gather_type",
            "BreakBlockInteraction_MatchTool_route_missing",
            "drop_program_unavailable",
            "runtime_permission_unavailable",
            "place_acceptance_unavailable",
            "mutation_not_exact_or_resync_required",
        ],
        "policy_binding": (
            "staged_for_single_announced_action_surface_contract_move"
        ),
    }


def block_action_target_contract_sha256() -> str:
    payload = json.dumps(
        block_action_target_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "BLOCK_ACTION_MUTATION_DIAGNOSTIC_GEOMETRY",
    "BLOCK_ACTION_MUTATION_DIAGNOSTIC_PREDICATE",
    "BLOCK_ACTION_MUTATION_DIAGNOSTIC_PROVENANCE",
    "BLOCK_ACTION_MUTATION_DIAGNOSTIC_SELECTION",
    "BLOCK_ACTION_MUTATION_SCHEMA",
    "BLOCK_ACTION_MUTATION_VERSION",
    "BLOCK_ACTION_DIAGNOSTIC_CATALOG_DUPLICATE",
    "BLOCK_ACTION_DIAGNOSTIC_CATALOG_MISSING",
    "BLOCK_ACTION_DIAGNOSTIC_DROP_UNAVAILABLE",
    "BLOCK_ACTION_DIAGNOSTIC_INTERACTION_TOOL_MISMATCH",
    "BLOCK_ACTION_DIAGNOSTIC_INVALID",
    "BLOCK_ACTION_DIAGNOSTIC_PERMISSION_UNAVAILABLE",
    "BLOCK_ACTION_DIAGNOSTIC_PLACE_UNAVAILABLE",
    "BLOCK_ACTION_DIAGNOSTIC_TARGET_UNAVAILABLE",
    "BLOCK_ACTION_DIAGNOSTIC_TOOL_UNAVAILABLE",
    "BLOCK_ACTION_TARGET_SCHEMA",
    "BLOCK_ACTION_TARGET_VERSION",
    "BlockActionCatalog",
    "BlockActionMutationPlan",
    "BlockActionTargetResult",
    "BlockDropProgramCatalog",
    "BlockGatherDefaults",
    "BlockMutationAcknowledgement",
    "NativeBlockInteractionOutcome",
    "BlockResolvedDrops",
    "BlockToolState",
    "block_action_catalog_from_local",
    "block_action_mutation_contract",
    "block_action_mutation_contract_sha256",
    "block_action_target_contract",
    "block_action_target_contract_sha256",
    "block_gather_defaults_from_local",
    "block_drop_program_catalog_from_local",
    "block_tool_state_from_local",
    "empty_block_tool_state",
    "exact_block_mutation_acknowledgement",
    "native_block_interaction_outcome_matches",
    "plan_exact_block_action_mutation",
    "resolve_block_action_targets",
]
