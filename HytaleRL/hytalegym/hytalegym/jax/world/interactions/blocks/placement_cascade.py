"""Fail-closed positive certificates for block-placement cascade safety."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.geometry.contract import FLAG_SOLID
from hytalegym.jax.world.block_actions import (
    BlockActionCatalog,
    block_action_target_contract_sha256,
)
from hytalegym.jax.world.region.atlas import lookup_region_blocks
from hytalegym.jax.world.region.types import RegionAtlas
from hytalegym.worldgen.block_actions import (
    SUPPORT_DROP_FALL,
    SUPPORT_DROP_NONE,
)


Array = jax.Array
PLACEMENT_CASCADE_SCHEMA = "hytalerl_block_placement_cascade_safety_v1"
PLACEMENT_CASCADE_VERSION = 1

PLACEMENT_CASCADE_DIAGNOSTIC_CANDIDATE_UNAVAILABLE = 1 << 0
PLACEMENT_CASCADE_DIAGNOSTIC_CATALOG_INDEX_INVALID = 1 << 1
PLACEMENT_CASCADE_DIAGNOSTIC_SUPPORT_SEMANTICS_INVALID = 1 << 2
PLACEMENT_CASCADE_DIAGNOSTIC_SUPPORT_DEPENDENT = 1 << 3
PLACEMENT_CASCADE_DIAGNOSTIC_FALLING_CANDIDATE = 1 << 4

REGION_PLACEMENT_SUPPORT_SCHEMA = (
    "hytalerl_region_block_placement_support_v1"
)
REGION_PLACEMENT_SUPPORT_VERSION = 1

REGION_PLACEMENT_DIAGNOSTIC_TARGET_UNAVAILABLE = 1 << 0
REGION_PLACEMENT_DIAGNOSTIC_SUPPORT_UNAVAILABLE = 1 << 1
REGION_PLACEMENT_DIAGNOSTIC_REGION_MISMATCH = 1 << 2
REGION_PLACEMENT_DIAGNOSTIC_FILLER_UNKNOWN = 1 << 3
REGION_PLACEMENT_DIAGNOSTIC_FILLER_NONZERO = 1 << 4
REGION_PLACEMENT_DIAGNOSTIC_NON_SOLID = 1 << 5
REGION_PLACEMENT_DIAGNOSTIC_NON_UNIT_BOX = 1 << 6


class BlockPlacementCascadeSafety(NamedTuple):
    """A positive certificate; ``safe`` is meaningful only if available."""

    available: Array
    safe: Array
    support_dependent: Array
    support_drop_type: Array
    falling_candidate: Array
    diagnostics: Array


class RegionBlockPlacementSupport(NamedTuple):
    """Exact native unit-block support evidence for Region targets."""

    available: Array
    supported: Array
    target_available: Array
    support_available: Array
    solid_material: Array
    direct_cell: Array
    unit_bounding_box: Array
    diagnostics: Array


def block_placement_cascade_safety(
    catalog: BlockActionCatalog,
    catalog_index: Array,
    candidate_available: Array,
) -> BlockPlacementCascadeSafety:
    """Admit exact placed types that cannot enter native support handling."""

    if not isinstance(catalog, BlockActionCatalog):
        raise TypeError("catalog must be BlockActionCatalog")
    capacity = catalog.entry_mask.shape[0]
    if capacity < 1:
        raise ValueError("catalog must have positive capacity")
    for field in ("support_dependent", "support_drop_type"):
        if getattr(catalog, field).shape != (capacity,):
            raise ValueError(f"catalog.{field} has the wrong shape")

    index = jnp.asarray(catalog_index)
    candidates = jnp.asarray(candidate_available)
    if index.shape != candidates.shape:
        raise ValueError(
            "catalog_index and candidate_available must have equal shapes"
        )
    if index.dtype == jnp.bool_ or not jnp.issubdtype(
        index.dtype,
        jnp.integer,
    ):
        raise TypeError("catalog_index must have an integer dtype")
    if candidates.dtype != jnp.bool_:
        raise TypeError("candidate_available must have a boolean dtype")

    index = index.astype(jnp.int32)
    in_bounds = (index >= 0) & (index < capacity)
    slot = jnp.clip(index, 0, capacity - 1)
    entry_present = catalog.entry_mask[slot]
    dependent = catalog.support_dependent[slot]
    drop_type = catalog.support_drop_type[slot].astype(jnp.int32)
    semantics_valid = (
        (~dependent & (drop_type == SUPPORT_DROP_NONE))
        | (
            dependent
            & (drop_type > SUPPORT_DROP_NONE)
            & (drop_type <= SUPPORT_DROP_FALL)
        )
    )
    selected = candidates & in_bounds & entry_present
    available = selected & semantics_valid
    support_dependent = available & dependent
    falling = support_dependent & (drop_type == SUPPORT_DROP_FALL)
    safe = available & ~support_dependent
    diagnostics = (
        jnp.where(
            ~candidates,
            jnp.uint32(
                PLACEMENT_CASCADE_DIAGNOSTIC_CANDIDATE_UNAVAILABLE
            ),
            jnp.uint32(0),
        )
        | jnp.where(
            candidates & (~in_bounds | ~entry_present),
            jnp.uint32(
                PLACEMENT_CASCADE_DIAGNOSTIC_CATALOG_INDEX_INVALID
            ),
            jnp.uint32(0),
        )
        | jnp.where(
            selected & ~semantics_valid,
            jnp.uint32(
                PLACEMENT_CASCADE_DIAGNOSTIC_SUPPORT_SEMANTICS_INVALID
            ),
            jnp.uint32(0),
        )
        | jnp.where(
            support_dependent,
            jnp.uint32(
                PLACEMENT_CASCADE_DIAGNOSTIC_SUPPORT_DEPENDENT
            ),
            jnp.uint32(0),
        )
        | jnp.where(
            falling,
            jnp.uint32(
                PLACEMENT_CASCADE_DIAGNOSTIC_FALLING_CANDIDATE
            ),
            jnp.uint32(0),
        )
    )
    return BlockPlacementCascadeSafety(
        available=available,
        safe=safe,
        support_dependent=support_dependent,
        support_drop_type=jnp.where(
            available,
            drop_type,
            SUPPORT_DROP_NONE,
        ).astype(jnp.uint8),
        falling_candidate=falling,
        diagnostics=diagnostics,
    )


def region_block_placement_support_result(
    atlas: RegionAtlas,
    target_block_positions: Array,
    environment_world_id: Array,
) -> RegionBlockPlacementSupport:
    """Port ``BlockPlacementHelper.testSupportingBlock`` exactly.

    The installed 0.5.7 target-cell predicate is unconditional after its
    loaded-chunk lookup. The queried target therefore contributes coverage,
    while the cell directly below contributes the four native support
    conditions: solid material, zero filler offset, and a unit union box.
    """

    if not isinstance(atlas, RegionAtlas):
        raise TypeError("atlas must be RegionAtlas")
    if atlas.region_mask.ndim != 1 or atlas.region_mask.shape[0] < 1:
        raise ValueError("atlas must have positive region capacity")
    if atlas.filler_root_known.shape != atlas.region_mask.shape:
        raise ValueError("atlas filler-root availability has an invalid shape")

    targets = jnp.asarray(target_block_positions)
    if targets.ndim != 3 or targets.shape[-1] != 3:
        raise ValueError(
            "target_block_positions must have shape (batch, queries, 3)"
        )
    if targets.dtype == jnp.bool_ or not jnp.issubdtype(
        targets.dtype,
        jnp.integer,
    ):
        raise TypeError("target_block_positions must have an integer dtype")
    world_ids = jnp.asarray(environment_world_id)
    if world_ids.shape != (targets.shape[0],):
        raise ValueError("environment_world_id must have shape (batch,)")
    if world_ids.dtype == jnp.bool_ or not jnp.issubdtype(
        world_ids.dtype,
        jnp.integer,
    ):
        raise TypeError("environment_world_id must have an integer dtype")

    targets = targets.astype(jnp.int32)
    world_ids = world_ids.astype(jnp.int32)
    supports = targets + jnp.asarray((0, -1, 0), dtype=jnp.int32)
    target = lookup_region_blocks(
        atlas,
        targets,
        world_ids,
        require_core=True,
    )
    support = lookup_region_blocks(
        atlas,
        supports,
        world_ids,
        require_core=True,
    )

    both_cells = target.available & support.available
    same_region = both_cells & (
        target.region_index == support.region_index
    )
    safe_region = jnp.clip(
        support.region_index,
        0,
        atlas.region_mask.shape[0] - 1,
    )
    filler_known = same_region & atlas.filler_root_known[safe_region]
    available = same_region & filler_known

    solid_material = available & (
        (support.flags.astype(jnp.int32) & jnp.int32(FLAG_SOLID)) != 0
    )
    direct_cell = available & jnp.all(
        support.filler_root_offset == 0,
        axis=-1,
    )

    boxes = support.collision_boxes
    box_mask = support.collision_box_mask
    active_box = jnp.any(box_mask, axis=-1)
    union_minimum = jnp.min(
        jnp.where(box_mask[..., None], boxes[..., :3], jnp.inf),
        axis=-2,
    )
    union_maximum = jnp.max(
        jnp.where(box_mask[..., None], boxes[..., 3:], -jnp.inf),
        axis=-2,
    )
    unit_bounding_box = (
        available
        & active_box
        & jnp.all(union_minimum == 0.0, axis=-1)
        & jnp.all(union_maximum == 1.0, axis=-1)
    )
    supported = (
        solid_material
        & direct_cell
        & unit_bounding_box
    )

    diagnostics = (
        jnp.where(
            ~target.available,
            jnp.uint32(REGION_PLACEMENT_DIAGNOSTIC_TARGET_UNAVAILABLE),
            jnp.uint32(0),
        )
        | jnp.where(
            target.available & ~support.available,
            jnp.uint32(REGION_PLACEMENT_DIAGNOSTIC_SUPPORT_UNAVAILABLE),
            jnp.uint32(0),
        )
        | jnp.where(
            both_cells & ~same_region,
            jnp.uint32(REGION_PLACEMENT_DIAGNOSTIC_REGION_MISMATCH),
            jnp.uint32(0),
        )
        | jnp.where(
            same_region & ~filler_known,
            jnp.uint32(REGION_PLACEMENT_DIAGNOSTIC_FILLER_UNKNOWN),
            jnp.uint32(0),
        )
        | jnp.where(
            available & ~direct_cell,
            jnp.uint32(REGION_PLACEMENT_DIAGNOSTIC_FILLER_NONZERO),
            jnp.uint32(0),
        )
        | jnp.where(
            available & ~solid_material,
            jnp.uint32(REGION_PLACEMENT_DIAGNOSTIC_NON_SOLID),
            jnp.uint32(0),
        )
        | jnp.where(
            available & ~unit_bounding_box,
            jnp.uint32(REGION_PLACEMENT_DIAGNOSTIC_NON_UNIT_BOX),
            jnp.uint32(0),
        )
    )
    return RegionBlockPlacementSupport(
        available=available,
        supported=supported,
        target_available=target.available,
        support_available=support.available,
        solid_material=solid_material,
        direct_cell=direct_cell,
        unit_bounding_box=unit_bounding_box,
        diagnostics=diagnostics,
    )


def block_placement_cascade_contract() -> dict[str, object]:
    """Return the checkpoint-pinnable placement-containment boundary."""

    return {
        "schema": PLACEMENT_CASCADE_SCHEMA,
        "version": PLACEMENT_CASCADE_VERSION,
        "server_version": "0.5.7",
        "block_action_target_contract_sha256": (
            block_action_target_contract_sha256()
        ),
        "native_source": {
            "dependency": "BlockType.hasSupport",
            "dispatch": (
                "BlockPhysicsUtil_support_zero_dispatches_"
                "SupportDropType_BREAK_DESTROY_or_FALL"
            ),
            "fall": "FallingBlock_fallBlock",
            "impact": (
                "FallingBlockTickingSystem_dispatches_configured_impact_"
                "or_default_Place"
            ),
        },
        "certificate": {
            "available": (
                "exact_catalog_index_with_canonical_support_semantics"
            ),
            "safe": "BlockType_hasSupport_false",
            "support_dependent": "known_unsafe_not_unavailable",
            "unknown_or_invalid": "unavailable_and_safe_false",
        },
        "scope": (
            "positive_no_support_transition_certificate_not_native_"
            "support_or_falling_simulation"
        ),
        "unsupported": [
            "support_predicate_evaluation",
            "falling_entity_trajectory",
            "Break_Place_or_Explode_impact_effects",
        ],
    }


def region_block_placement_support_contract() -> dict[str, object]:
    """Return the native-derived unit-block support boundary."""

    return {
        "schema": REGION_PLACEMENT_SUPPORT_SCHEMA,
        "version": REGION_PLACEMENT_SUPPORT_VERSION,
        "server_version": "0.5.7",
        "published_provenance": "native_source_exact_region_projection",
        "native_source": {
            "predicate": (
                "BlockPlacementHelper.testSupportingBlock"
            ),
            "target_loaded_check": (
                "BlockPlacementHelper.canPlaceUnitBlock"
            ),
            "solid_material": (
                "BlockType.getMaterial_equals_BlockMaterial.Solid"
            ),
            "direct_cell": "filler_equals_zero",
            "unit_box": (
                "BlockBoundingBoxes.RotatedVariantBoxes.getBoundingBox"
                "_then_Box.isUnitBox"
            ),
            "installed_target_predicate": (
                "BlockPlacementHelper.testBlock_returns_true_after_lookup"
            ),
        },
        "region_projection": {
            "solid_material": (
                "FLAG_SOLID_captured_only_for_BlockMaterial.Solid"
            ),
            "direct_cell": "exact_native_filler_root_offset_equals_zero",
            "unit_box": (
                "union_minimum_equals_0_and_union_maximum_equals_1"
            ),
            "target_and_support": (
                "both_core_cells_available_in_same_region"
            ),
        },
        "validity": {
            "available": (
                "target_and_support_cells_available_same_region_and_"
                "exact_filler_offsets_known"
            ),
            "unknown": "available_false_and_supported_false",
            "known_rejection": (
                "available_true_supported_false_with_diagnostic"
            ),
        },
        "scope": (
            "unit_block_support_predicate_not_full_multicell_placement_"
            "or_player_interaction_certificate"
        ),
        "unsupported": [
            "placement_rotation_search",
            "multi_cell_placement",
            "item_consumption",
            "interaction_chain_timing_and_cancellation",
            "authenticated_player_acceptance",
        ],
    }


def block_placement_cascade_contract_sha256() -> str:
    payload = json.dumps(
        block_placement_cascade_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def region_block_placement_support_contract_sha256() -> str:
    payload = json.dumps(
        region_block_placement_support_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "PLACEMENT_CASCADE_DIAGNOSTIC_CANDIDATE_UNAVAILABLE",
    "PLACEMENT_CASCADE_DIAGNOSTIC_CATALOG_INDEX_INVALID",
    "PLACEMENT_CASCADE_DIAGNOSTIC_FALLING_CANDIDATE",
    "PLACEMENT_CASCADE_DIAGNOSTIC_SUPPORT_DEPENDENT",
    "PLACEMENT_CASCADE_DIAGNOSTIC_SUPPORT_SEMANTICS_INVALID",
    "PLACEMENT_CASCADE_SCHEMA",
    "PLACEMENT_CASCADE_VERSION",
    "REGION_PLACEMENT_DIAGNOSTIC_FILLER_NONZERO",
    "REGION_PLACEMENT_DIAGNOSTIC_FILLER_UNKNOWN",
    "REGION_PLACEMENT_DIAGNOSTIC_NON_SOLID",
    "REGION_PLACEMENT_DIAGNOSTIC_NON_UNIT_BOX",
    "REGION_PLACEMENT_DIAGNOSTIC_REGION_MISMATCH",
    "REGION_PLACEMENT_DIAGNOSTIC_SUPPORT_UNAVAILABLE",
    "REGION_PLACEMENT_DIAGNOSTIC_TARGET_UNAVAILABLE",
    "REGION_PLACEMENT_SUPPORT_SCHEMA",
    "REGION_PLACEMENT_SUPPORT_VERSION",
    "BlockPlacementCascadeSafety",
    "RegionBlockPlacementSupport",
    "block_placement_cascade_contract",
    "block_placement_cascade_contract_sha256",
    "block_placement_cascade_safety",
    "region_block_placement_support_contract",
    "region_block_placement_support_contract_sha256",
    "region_block_placement_support_result",
]
