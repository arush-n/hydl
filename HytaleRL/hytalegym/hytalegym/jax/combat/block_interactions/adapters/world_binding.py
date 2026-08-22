"""Fail-closed binding from World's block producers into Combat."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.block_interactions.schema.contract import (
    BLOCK_INTERACTION_BREAK,
    BLOCK_INTERACTION_PLACE,
    NO_BLOCK_ID,
    block_interaction_contract_sha256,
)
from hytalegym.jax.combat.block_interactions.execution.runtime import (
    step_block_interactions,
)
from hytalegym.jax.combat.block_interactions.schema.types import (
    BlockInteractionEvidence,
    BlockInteractionProgram,
    ResolvedBlockDrops,
)
from hytalegym.jax.combat.inventory import (
    EMPTY_ITEM_ID,
    METADATA_HASH_WORDS,
    inventory_contract_sha256,
)
from hytalegym.jax.world import (
    BLOCK_SEMANTIC_KEY_WORDS,
    ActorBlockActionParameters,
    BlockActionTargetResult,
    BlockMutationAcknowledgement,
    RegionBlockPlacementSupport,
    RegionRemovalCascadeSafety,
    actor_block_action_parameter_contract_sha256,
    block_action_target_contract_sha256,
    mutable_block_contract_sha256,
    region_block_placement_support_contract_sha256,
    region_removal_cascade_contract_sha256,
)


Array = jax.Array
BLOCK_INTERACTION_WORLD_BINDING_SCHEMA = (
    "hytalerl_combat_block_interaction_world_binding_v4"
)
BLOCK_INTERACTION_WORLD_BINDING_VERSION = 4
BLOCK_INTERACTION_WORLD_BINDING_DIAGNOSTIC_SELECTION = jnp.uint32(1 << 31)


class BlockInteractionWorldInputs(NamedTuple):
    """Combat-ready values derived only from exact World producer output."""

    evidence: BlockInteractionEvidence
    drops: ResolvedBlockDrops


def select_actor_block_action_target(
    parameters: ActorBlockActionParameters,
    candidate_index: Array,
) -> BlockActionTargetResult:
    """Select one candidate-aligned World parameter row per actor.

    The policy selects bounded actor-local slots, never global block IDs.
    Missing rows, row diagnostics, masked slots, and out-of-range indices
    produce unavailable target evidence. Resolved quantities remain attached
    to the selected slot.
    """

    if not isinstance(parameters, ActorBlockActionParameters):
        raise TypeError("parameters must be ActorBlockActionParameters")
    shape = parameters.available.shape
    if len(shape) != 2:
        raise ValueError("parameters.available must have shape [batch, actor]")
    batch, actors = shape
    mask = parameters.candidate_mask
    if mask.ndim != 3 or mask.shape[:2] != shape:
        raise ValueError(
            "parameters.candidate_mask must have shape [batch, actor, candidate]"
        )
    capacity = mask.shape[2]
    if capacity <= 0:
        raise ValueError("parameters candidate capacity must be positive")
    index = jnp.asarray(candidate_index, dtype=jnp.int32)
    if index.shape != shape:
        raise ValueError("candidate_index must have shape [batch, actor]")
    safe_index = jnp.clip(index, 0, capacity - 1)

    def selected(value: Array) -> Array:
        array = jnp.asarray(value)
        if array.shape[:3] != mask.shape:
            raise ValueError(
                "candidate-aligned parameter fields must share "
                "[batch, actor, candidate]"
            )
        gather_index = safe_index.reshape(
            shape + (1,) * (array.ndim - 2)
        )
        gather_index = jnp.broadcast_to(
            gather_index,
            shape + (1,) + array.shape[3:],
        )
        return jnp.take_along_axis(array, gather_index, axis=2)[:, :, 0]

    requested = index >= 0
    in_range = requested & (index < capacity)
    selected_mask = selected(mask)
    accepted = (
        in_range
        & parameters.available
        & (parameters.diagnostics == 0)
        & selected_mask
    )

    def masked(value: Array, fill_value: int | float | bool = 0) -> Array:
        value = selected(value)
        gate = accepted.reshape(
            shape + (1,) * (value.ndim - len(shape))
        )
        return jnp.where(gate, value, jnp.full_like(value, fill_value))

    def drops(value):
        drop_mask = masked(value.mask, False)
        return type(value)(
            available=masked(value.available, False),
            mask=drop_mask,
            item_id=jnp.where(
                drop_mask,
                masked(value.item_id, -1),
                jnp.int32(-1),
            ),
            quantity=masked(value.quantity, 0),
            item_max_stack=masked(value.item_max_stack, 0),
            durability=masked(value.durability, 0.0),
            max_durability=masked(value.max_durability, 0.0),
            metadata_hash=masked(value.metadata_hash, 0),
        )

    diagnostics = parameters.diagnostics | jnp.where(
        requested & ~accepted,
        BLOCK_INTERACTION_WORLD_BINDING_DIAGNOSTIC_SELECTION,
        jnp.uint32(0),
    )
    return BlockActionTargetResult(
        target_available=accepted,
        diagnostics=diagnostics,
        # The actor parameter contract intentionally omits privileged catalog
        # ordinals; no Combat consumer reads this diagnostic-only field.
        catalog_index=jnp.full(shape, -1, dtype=jnp.int32),
        break_available=masked(parameters.break_available, False),
        break_allowed=masked(parameters.break_allowed, False),
        tool_compatible=masked(parameters.tool_compatible, False),
        tool_power=masked(parameters.tool_power, 0.0),
        interaction_tool_route_matched=masked(
            parameters.interaction_tool_route_matched,
            False,
        ),
        break_removes_block=masked(
            parameters.break_removes_block,
            False,
        ),
        replacement_semantic_key_valid=masked(
            parameters.replacement_semantic_key_valid,
            False,
        ),
        replacement_semantic_key=masked(
            parameters.replacement_semantic_key,
            0,
        ),
        harvest_available=masked(parameters.harvest_available, False),
        harvest_allowed=masked(parameters.harvest_allowed, False),
        target_harvestable=masked(parameters.target_harvestable, False),
        place_available=masked(parameters.place_available, False),
        place_allowed=masked(parameters.place_allowed, False),
        break_drops=drops(parameters.break_drops),
        harvest_drops=drops(parameters.harvest_drops),
    )


def bind_world_removal_cascade_safety(
    result: RegionRemovalCascadeSafety,
) -> Array:
    """Return only complete positive World cascade-safety certificates."""

    if not isinstance(result, RegionRemovalCascadeSafety):
        raise TypeError(
            "result must be RegionRemovalCascadeSafety"
        )
    shape = result.available.shape
    if len(shape) != 2:
        raise ValueError(
            "World removal-cascade values must have shape [batch, entity]"
        )
    for name in (
        "available",
        "safe",
        "target_single_cell",
        "support_dependent_neighbour",
        "falling_neighbour",
    ):
        value = getattr(result, name)
        if value.shape != shape:
            raise ValueError(
                f"result.{name} must match result.available"
            )
        if jnp.asarray(value).dtype != jnp.bool_:
            raise TypeError(f"result.{name} must have boolean dtype")
    if result.diagnostics.shape != shape:
        raise ValueError(
            "result.diagnostics must match result.available"
        )
    if jnp.asarray(result.diagnostics).dtype != jnp.uint32:
        raise TypeError("result.diagnostics must have uint32 dtype")
    return result.available & result.safe


def bind_world_placement_support(
    result: RegionBlockPlacementSupport,
) -> tuple[Array, Array]:
    """Return availability and the exact supporting-block verdict."""

    if not isinstance(result, RegionBlockPlacementSupport):
        raise TypeError("result must be RegionBlockPlacementSupport")
    shape = result.available.shape
    if len(shape) != 2:
        raise ValueError(
            "World placement-support values must have shape [batch, entity]"
        )
    for name in (
        "available",
        "supported",
        "target_available",
        "support_available",
        "solid_material",
        "direct_cell",
        "unit_bounding_box",
    ):
        value = jnp.asarray(getattr(result, name))
        if value.shape != shape:
            raise ValueError(f"result.{name} must match result.available")
        if value.dtype != jnp.bool_:
            raise TypeError(f"result.{name} must have boolean dtype")
    diagnostics = jnp.asarray(result.diagnostics)
    if diagnostics.shape != shape:
        raise ValueError("result.diagnostics must match result.available")
    if diagnostics.dtype != jnp.uint32:
        raise TypeError("result.diagnostics must have uint32 dtype")
    supported = (
        result.available
        & result.supported
        & result.target_available
        & result.support_available
        & result.solid_material
        & result.direct_cell
        & result.unit_bounding_box
    )
    return result.available, supported


def bind_world_block_interaction_inputs(
    program: BlockInteractionProgram,
    target: BlockActionTargetResult,
    mutation: BlockMutationAcknowledgement,
    *,
    creative_mode: Array,
    removal_cascade_safe: Array,
    pickup_container_id: Array,
    pickup_slot_allowed: Array,
    placement_support: Array | RegionBlockPlacementSupport | None = None,
) -> BlockInteractionWorldInputs:
    """Bind one World query/update row to each Combat entity.

    The query axis must already be entity ordered. A known denial needs no
    mutation, while an allowed action requires positive pre-mutation cascade
    safety and an exact applied acknowledgement. Missing or rejected evidence
    therefore cannot become a successful Combat first-run effect.
    """

    shape = _validate_binding_shapes(
        program,
        target,
        mutation,
        creative_mode,
        removal_cascade_safe,
        pickup_container_id,
        pickup_slot_allowed,
    )
    creative = jnp.asarray(creative_mode, dtype=jnp.bool_)
    cascade_safe = _cascade_safe_array(
        removal_cascade_safe,
        shape,
    )
    placement_evidence_available, placement_supported = (
        _placement_support_arrays(placement_support, shape)
    )
    pickup_container = jnp.asarray(
        pickup_container_id,
        dtype=jnp.int32,
    )
    pickup_allowed = jnp.asarray(
        pickup_slot_allowed,
        dtype=jnp.bool_,
    )

    break_kind = program.kind == BLOCK_INTERACTION_BREAK
    place_kind = program.kind == BLOCK_INTERACTION_PLACE
    harvest_kind = break_kind & program.harvest
    normal_break = break_kind & ~program.harvest
    valid_kind = break_kind | place_kind
    emits_resolved_drops = harvest_kind | (normal_break & ~creative)
    selected_world_drops = jax.tree.map(
        lambda normal, harvest: jnp.where(
            program.harvest.reshape(shape + (1,) * (normal.ndim - len(shape))),
            harvest,
            normal,
        ),
        target.break_drops,
        target.harvest_drops,
    )

    action_available = (
        (normal_break & target.break_available)
        | (harvest_kind & target.harvest_available)
        | (place_kind & target.place_available)
    )
    action_allowed_without_support = (
        (normal_break & target.break_allowed)
        | (harvest_kind & target.harvest_allowed)
        | (place_kind & target.place_allowed)
    )
    placement_evidence_required = place_kind & target.place_allowed
    placement_evidence_ready = (
        ~placement_evidence_required | placement_evidence_available
    )
    action_allowed = (
        action_allowed_without_support
        & (~place_kind | placement_supported)
    )
    cascade_supported = ~break_kind | ~action_allowed | cascade_safe
    effect_allowed = action_allowed & (~break_kind | cascade_safe)
    acknowledgement_available = (
        mutation.available[:, None]
        & mutation.accepted[:, None]
        & ~mutation.stale_base[:, None]
        & ~mutation.duplicate_position[:, None]
        & ~mutation.capacity_exceeded[:, None]
        & ~mutation.invalid[:, None]
        & ~mutation.resync_required[:, None]
    )
    acknowledgement_applied = acknowledgement_available & mutation.applied
    mutation_supported = (
        valid_kind
        & action_available
        & cascade_supported
        & placement_evidence_ready
        & (
            ~action_allowed
            | (
                acknowledgement_applied
                & (~emits_resolved_drops | selected_world_drops.available)
            )
        )
    )

    before = mutation.before
    after = mutation.after
    before_valid = (
        before.available
        & ~before.invalid
        & ~before.resync_required
        & before.geometry.exact
    )
    after_valid = (
        after.available & ~after.invalid & ~after.resync_required & after.geometry.exact
    )
    target_loaded = target.target_available & before_valid
    target_present = target_loaded & before.geometry.block_present
    before_health_valid = (
        target_present & before.block_health_valid & jnp.isfinite(before.block_health)
    )
    after_present = after_valid & after.geometry.block_present
    after_health_valid = (
        after_present & after.block_health_valid & jnp.isfinite(after.block_health)
    )
    same_remaining_block = (
        after_present
        & before.geometry.semantic_key_valid
        & after.geometry.semantic_key_valid
        & jnp.all(
            before.geometry.semantic_key == after.geometry.semantic_key,
            axis=-1,
        )
    )
    removal = (
        acknowledgement_applied
        & break_kind
        & effect_allowed
        & target_present
        & after_valid
        & ~after.geometry.block_present
    )
    health_after = jnp.where(
        same_remaining_block & after_health_valid,
        after.block_health,
        0.0,
    )
    health_delta = before.block_health - health_after
    damage_applied = (
        acknowledgement_applied
        & normal_break
        & ~creative
        & effect_allowed
        & before_health_valid
        & after_valid
        & (~after.geometry.block_present | same_remaining_block)
        & jnp.isfinite(health_delta)
        & (health_delta > 0.0)
    )
    place_applied = (
        acknowledgement_applied
        & place_kind
        & effect_allowed
        & before_valid
        & ~before.geometry.block_present
        & after_present
    )
    placed_id_valid = place_applied & after.geometry.runtime_block_id_valid

    drop_gate = (
        emits_resolved_drops
        & action_available
        & effect_allowed
        & selected_world_drops.available
    )
    drop_mask = selected_world_drops.mask & drop_gate[..., None]
    metadata_mask = drop_mask[..., None]
    drops = ResolvedBlockDrops(
        mask=drop_mask,
        item_id=jnp.where(
            drop_mask,
            selected_world_drops.item_id,
            jnp.int32(EMPTY_ITEM_ID),
        ),
        quantity=jnp.where(
            drop_mask,
            selected_world_drops.quantity,
            jnp.int32(0),
        ),
        durability=jnp.where(
            drop_mask,
            selected_world_drops.durability,
            jnp.float32(0.0),
        ),
        max_durability=jnp.where(
            drop_mask,
            selected_world_drops.max_durability,
            jnp.float32(0.0),
        ),
        metadata_hash=jnp.where(
            metadata_mask,
            selected_world_drops.metadata_hash,
            jnp.uint32(0),
        ),
        item_max_stack=jnp.where(
            drop_mask,
            selected_world_drops.item_max_stack,
            jnp.int32(0),
        ),
        pickup_container_id=jnp.where(
            drop_mask,
            pickup_container,
            jnp.int32(0),
        ),
        pickup_slot_allowed=(pickup_allowed & drop_mask[..., None]),
    )
    evidence = BlockInteractionEvidence(
        mutation_supported=mutation_supported,
        target_loaded=target_loaded,
        target_present=target_present,
        creative_mode=creative,
        block_breaking_allowed=(
            normal_break & target.break_allowed & cascade_safe
        ),
        block_gathering_allowed=(
            harvest_kind & target.harvest_allowed & cascade_safe
        ),
        target_harvestable=(target.target_harvestable & target.target_available),
        damage_applied=damage_applied,
        block_health_before=jnp.where(
            before_health_valid,
            before.block_health,
            1.0,
        ),
        block_damage=jnp.where(damage_applied, health_delta, 0.0),
        block_remove_applied=removal,
        place_allowed=(
            place_kind & target.place_allowed & placement_supported
        ),
        place_applied=place_applied,
        placed_block_id=jnp.where(
            placed_id_valid,
            after.geometry.runtime_block_id,
            NO_BLOCK_ID,
        ).astype(jnp.int32),
    )
    return BlockInteractionWorldInputs(evidence=evidence, drops=drops)


def step_block_interactions_from_world(
    state,
    inventory,
    inventory_layout,
    command,
    program,
    target,
    mutation,
    *,
    creative_mode,
    removal_cascade_safe,
    pickup_container_id,
    pickup_slot_allowed,
    dt_seconds,
    placement_support=None,
):
    """Commit a Combat interaction using exact World target/update evidence."""

    inputs = bind_world_block_interaction_inputs(
        program,
        target,
        mutation,
        creative_mode=creative_mode,
        removal_cascade_safe=removal_cascade_safe,
        pickup_container_id=pickup_container_id,
        pickup_slot_allowed=pickup_slot_allowed,
        placement_support=placement_support,
    )
    return step_block_interactions(
        state,
        inventory,
        inventory_layout,
        command,
        program,
        inputs.evidence,
        inputs.drops,
        dt_seconds=dt_seconds,
    )


def block_interaction_world_binding_contract() -> dict[str, object]:
    """Return the staged, checkpoint-bindable producer seam."""

    return {
        "schema": BLOCK_INTERACTION_WORLD_BINDING_SCHEMA,
        "version": BLOCK_INTERACTION_WORLD_BINDING_VERSION,
        "combat_block_interaction_contract_sha256": (
            block_interaction_contract_sha256()
        ),
        "combat_inventory_contract_sha256": inventory_contract_sha256(),
        "world_block_action_target_contract_sha256": (
            block_action_target_contract_sha256()
        ),
        "world_actor_block_action_parameter_contract_sha256": (
            actor_block_action_parameter_contract_sha256()
        ),
        "world_mutable_block_contract_sha256": (mutable_block_contract_sha256()),
        "world_removal_cascade_contract_sha256": (
            region_removal_cascade_contract_sha256()
        ),
        "world_placement_support_contract_sha256": (
            region_block_placement_support_contract_sha256()
        ),
        "query_mapping": "one_world_query_per_combat_entity_in_entity_order",
        "candidate_selection": (
            "bounded_actor_local_index_preserves_resolved_quantities"
        ),
        "catalog_index": "intentionally_omitted_by_actor_parameter_contract",
        "allowed_action_acknowledgement": ("exact_applied_ack_required_fail_closed"),
        "known_denial": "available_predicate_requires_no_mutation",
        "removal_cascade_safety": (
            "positive_pre_mutation_context_evidence_required_for_"
            "allowed_break_or_harvest"
        ),
        "removal_cascade_evidence_owner": (
            "world_region_removal_cascade_safety_positive_certificate"
        ),
        "placement_support": (
            "exact_BlockPlacementHelper_testSupportingBlock_certificate_"
            "required_for_allowed_place"
        ),
        "drop_selection": "break_or_harvest_from_authored_program",
        "break_drop_evidence": (
            "survival_normal_or_harvest_requires_selected_available_"
            "known_empty_distinct"
        ),
        "creative_normal_break_drops": (
            "suppressed_without_drop_program_lookup"
        ),
        "inactive_drop_slots": "canonical_empty_no_masked_payload",
        "inventory": (
            "caller_supplies_container_and_slot_legality_no_permissive_default"
        ),
        "policy_abi": "staged_not_published",
    }


def block_interaction_world_binding_contract_sha256() -> str:
    """Hash the complete staged World-to-Combat seam."""

    payload = json.dumps(
        block_interaction_world_binding_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _validate_binding_shapes(
    program,
    target,
    mutation,
    creative_mode,
    removal_cascade_safe,
    pickup_container_id,
    pickup_slot_allowed,
) -> tuple[int, int]:
    if not isinstance(program, BlockInteractionProgram):
        raise TypeError("program must be BlockInteractionProgram")
    if not isinstance(target, BlockActionTargetResult):
        raise TypeError("target must be BlockActionTargetResult")
    if not isinstance(mutation, BlockMutationAcknowledgement):
        raise TypeError("mutation must be BlockMutationAcknowledgement")
    shape = program.kind.shape
    if len(shape) != 2:
        raise ValueError("program kind must have shape [batch, entity]")
    if program.harvest.shape != shape:
        raise ValueError("program harvest must match program kind")
    for name in (
        "target_available",
        "break_available",
        "break_allowed",
        "interaction_tool_route_matched",
        "break_removes_block",
        "replacement_semantic_key_valid",
        "harvest_available",
        "harvest_allowed",
        "target_harvestable",
        "place_available",
        "place_allowed",
    ):
        if getattr(target, name).shape != shape:
            raise ValueError(f"target.{name} must match program kind")
    if target.replacement_semantic_key.shape != shape + (
        BLOCK_SEMANTIC_KEY_WORDS,
    ):
        raise ValueError(
            "target.replacement_semantic_key must match program kind "
            "plus the semantic-key axis"
        )
    if mutation.available.shape != (shape[0],):
        raise ValueError("mutation row values must have shape [batch]")
    for name in (
        "accepted",
        "stale_base",
        "duplicate_position",
        "capacity_exceeded",
        "invalid",
        "resync_required",
    ):
        if getattr(mutation, name).shape != (shape[0],):
            raise ValueError(f"mutation.{name} must have shape [batch]")
    if mutation.applied.shape != shape:
        raise ValueError("mutation.applied must have shape [batch, entity]")
    for label, query in (
        ("before", mutation.before),
        ("after", mutation.after),
    ):
        if query.available.shape != shape:
            raise ValueError(f"mutation.{label} must have shape [batch, entity]")
    if jnp.shape(creative_mode) != shape:
        raise ValueError("creative_mode must have shape [batch, entity]")
    if jnp.asarray(creative_mode).dtype != jnp.bool_:
        raise TypeError("creative_mode must have boolean dtype")
    _cascade_safe_array(removal_cascade_safe, shape)
    normal_shape = target.break_drops.mask.shape
    harvest_shape = target.harvest_drops.mask.shape
    if (
        len(normal_shape) != 3
        or normal_shape[:2] != shape
        or harvest_shape != normal_shape
    ):
        raise ValueError("World drops must share shape [batch, entity, drop]")
    for label, drops in (
        ("break_drops", target.break_drops),
        ("harvest_drops", target.harvest_drops),
    ):
        if drops.available.shape != shape:
            raise ValueError(f"target.{label}.available must match program kind")
        for name in (
            "item_id",
            "quantity",
            "item_max_stack",
            "durability",
            "max_durability",
        ):
            if getattr(drops, name).shape != normal_shape:
                raise ValueError(f"target.{label}.{name} must match its mask")
        if drops.metadata_hash.shape != normal_shape + (METADATA_HASH_WORDS,):
            raise ValueError(f"target.{label}.metadata_hash has wrong shape")
    if jnp.shape(pickup_container_id) != normal_shape:
        raise ValueError("pickup_container_id must match the World drop shape")
    if not jnp.issubdtype(
        jnp.asarray(pickup_container_id).dtype,
        jnp.integer,
    ):
        raise TypeError("pickup_container_id must have integer dtype")
    allowed_shape = jnp.shape(pickup_slot_allowed)
    if len(allowed_shape) != 4 or allowed_shape[:3] != normal_shape:
        raise ValueError(
            "pickup_slot_allowed must have shape [batch, entity, drop, inventory_slot]"
        )
    if jnp.asarray(pickup_slot_allowed).dtype != jnp.bool_:
        raise TypeError("pickup_slot_allowed must have boolean dtype")
    return shape


def _cascade_safe_array(
    value: Array | RegionRemovalCascadeSafety,
    shape: tuple[int, int],
) -> Array:
    if isinstance(value, RegionRemovalCascadeSafety):
        resolved = bind_world_removal_cascade_safety(value)
    else:
        resolved = jnp.asarray(value)
        if resolved.dtype != jnp.bool_:
            raise TypeError(
                "removal_cascade_safe must have boolean dtype"
            )
    if resolved.shape != shape:
        raise ValueError(
            "removal_cascade_safe must have shape [batch, entity]"
        )
    return resolved


def _placement_support_arrays(
    value: Array | RegionBlockPlacementSupport | None,
    shape: tuple[int, int],
) -> tuple[Array, Array]:
    if value is None:
        unavailable = jnp.zeros(shape, dtype=jnp.bool_)
        return unavailable, unavailable
    if isinstance(value, RegionBlockPlacementSupport):
        available, supported = bind_world_placement_support(value)
    else:
        supported = jnp.asarray(value)
        if supported.dtype != jnp.bool_:
            raise TypeError("placement_support must have boolean dtype")
        available = jnp.ones_like(supported, dtype=jnp.bool_)
    if available.shape != shape or supported.shape != shape:
        raise ValueError(
            "placement_support must have shape [batch, entity]"
        )
    return available, supported


__all__ = [
    "BLOCK_INTERACTION_WORLD_BINDING_SCHEMA",
    "BLOCK_INTERACTION_WORLD_BINDING_VERSION",
    "BLOCK_INTERACTION_WORLD_BINDING_DIAGNOSTIC_SELECTION",
    "BlockInteractionWorldInputs",
    "bind_world_placement_support",
    "bind_world_removal_cascade_safety",
    "bind_world_block_interaction_inputs",
    "block_interaction_world_binding_contract",
    "block_interaction_world_binding_contract_sha256",
    "select_actor_block_action_target",
    "step_block_interactions_from_world",
]
