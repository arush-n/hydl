"""Fail-closed positive certificates for block-removal cascade safety."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.world.block_actions import (
    BlockActionCatalog,
    block_action_target_contract_sha256,
)
from hytalegym.jax.world.region.atlas import lookup_region_blocks
from hytalegym.jax.world.region.block_semantics import (
    RegionBlockSemanticAtlas,
    lookup_region_block_semantics,
    query_region_block_action_target,
)
from hytalegym.jax.world.region.types import RegionAtlas
from hytalegym.worldgen.block_actions import (
    BLOCK_SEMANTIC_KEY_WORDS,
    SUPPORT_DROP_FALL,
    SUPPORT_DROP_NONE,
)


Array = jax.Array
REMOVAL_CASCADE_SCHEMA = "hytalerl_region_removal_cascade_safety_v1"
REMOVAL_CASCADE_VERSION = 1

REMOVAL_CASCADE_DIAGNOSTIC_TARGET_UNAVAILABLE = 1 << 0
REMOVAL_CASCADE_DIAGNOSTIC_TARGET_NOT_SINGLE_CELL = 1 << 1
REMOVAL_CASCADE_DIAGNOSTIC_NEIGHBOUR_UNAVAILABLE = 1 << 2
REMOVAL_CASCADE_DIAGNOSTIC_SUPPORT_DEPENDENT = 1 << 3

_NEIGHBOUR_OFFSETS = jnp.asarray(
    [
        (x, y, z)
        for y in (-1, 0, 1)
        for z in (-1, 0, 1)
        for x in (-1, 0, 1)
        if (x, y, z) != (0, 0, 0)
    ],
    dtype=jnp.int32,
)


class RegionRemovalCascadeSafety(NamedTuple):
    """A conservative certificate; ``safe`` is meaningful only if available."""

    available: Array
    safe: Array
    target_single_cell: Array
    support_dependent_neighbour: Array
    falling_neighbour: Array
    diagnostics: Array


def region_removal_cascade_safety(
    semantic_atlas: RegionBlockSemanticAtlas,
    physical_atlas: RegionAtlas,
    catalog: BlockActionCatalog,
    block_positions: Array,
    environment_world_id: Array,
    *,
    require_core: bool = True,
) -> RegionRemovalCascadeSafety:
    """Certify removals that cannot start native support propagation."""

    positions = jnp.asarray(block_positions, dtype=jnp.int32)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("block_positions must have shape [batch, query, 3]")
    if not isinstance(catalog, BlockActionCatalog):
        raise TypeError("catalog must be BlockActionCatalog")
    capacity = catalog.entry_mask.shape[0]
    if (
        catalog.semantic_key.shape
        != (capacity, BLOCK_SEMANTIC_KEY_WORDS)
        or catalog.support_dependent.shape != (capacity,)
        or catalog.support_drop_type.shape != (capacity,)
    ):
        raise ValueError("catalog support semantics have invalid shapes")

    target = query_region_block_action_target(
        semantic_atlas,
        physical_atlas,
        positions,
        environment_world_id,
        require_core=require_core,
    )
    boxes = target.query.geometry.collision_boxes
    box_mask = target.query.geometry.collision_box_mask
    box_inside_cell = jnp.all(
        ~box_mask
        | (
            jnp.all(boxes[..., :3] >= 0.0, axis=-1)
            & jnp.all(boxes[..., 3:] <= 1.0, axis=-1)
        ),
        axis=-1,
    )
    target_single_cell = (
        target.available
        & target.query.geometry.block_present
        & target.query.geometry.exact
        & ~target.canonicalized_to_root
        & ~target.direct_filler
        & box_inside_cell
    )

    batch, queries, _ = positions.shape
    neighbour_positions = (
        target.action_position[..., None, :] + _NEIGHBOUR_OFFSETS
    ).reshape(batch, queries * len(_NEIGHBOUR_OFFSETS), 3)
    physical = lookup_region_blocks(
        physical_atlas,
        neighbour_positions,
        environment_world_id,
        require_core=require_core,
    )
    semantic = lookup_region_block_semantics(
        semantic_atlas,
        physical_atlas,
        neighbour_positions,
        environment_world_id,
        require_core=require_core,
    )
    matches = (
        catalog.entry_mask[None, None, :]
        & jnp.all(
            semantic.semantic_key[..., None, :] == catalog.semantic_key,
            axis=-1,
        )
    )
    match_count = jnp.sum(matches, axis=-1)
    slot = jnp.argmax(matches, axis=-1).astype(jnp.int32)
    dependent = catalog.support_dependent[slot]
    drop_type = catalog.support_drop_type[slot].astype(jnp.int32)
    support_value_valid = (
        (~dependent & (drop_type == SUPPORT_DROP_NONE))
        | (
            dependent
            & (drop_type > SUPPORT_DROP_NONE)
            & (drop_type <= SUPPORT_DROP_FALL)
        )
    )
    neighbour_known = (
        physical.available
        & semantic.available
        & (
            ~semantic.block_present
            | ((match_count == 1) & support_value_valid)
        )
    )
    dependent = neighbour_known & semantic.block_present & dependent
    falling = dependent & (drop_type == SUPPORT_DROP_FALL)
    neighbourhood_shape = (batch, queries, len(_NEIGHBOUR_OFFSETS))
    neighbour_known = neighbour_known.reshape(neighbourhood_shape)
    dependent = dependent.reshape(neighbourhood_shape)
    falling = falling.reshape(neighbourhood_shape)
    neighbours_available = jnp.all(neighbour_known, axis=-1)
    support_dependent = jnp.any(dependent, axis=-1)
    falling_neighbour = jnp.any(falling, axis=-1)
    target_available = (
        target.available
        & target.query.geometry.block_present
        & target.query.geometry.exact
    )
    available = target_available & target_single_cell & neighbours_available
    safe = available & ~support_dependent
    diagnostics = (
        jnp.where(
            ~target_available,
            jnp.uint32(REMOVAL_CASCADE_DIAGNOSTIC_TARGET_UNAVAILABLE),
            jnp.uint32(0),
        )
        | jnp.where(
            target_available & ~target_single_cell,
            jnp.uint32(
                REMOVAL_CASCADE_DIAGNOSTIC_TARGET_NOT_SINGLE_CELL
            ),
            jnp.uint32(0),
        )
        | jnp.where(
            ~neighbours_available,
            jnp.uint32(
                REMOVAL_CASCADE_DIAGNOSTIC_NEIGHBOUR_UNAVAILABLE
            ),
            jnp.uint32(0),
        )
        | jnp.where(
            support_dependent,
            jnp.uint32(
                REMOVAL_CASCADE_DIAGNOSTIC_SUPPORT_DEPENDENT
            ),
            jnp.uint32(0),
        )
    )
    return RegionRemovalCascadeSafety(
        available=available,
        safe=safe,
        target_single_cell=target_single_cell,
        support_dependent_neighbour=support_dependent,
        falling_neighbour=falling_neighbour,
        diagnostics=diagnostics,
    )


def region_removal_cascade_contract() -> dict[str, object]:
    return {
        "schema": REMOVAL_CASCADE_SCHEMA,
        "version": REMOVAL_CASCADE_VERSION,
        "server_version": "0.5.7",
        "block_action_target_contract_sha256": (
            block_action_target_contract_sha256()
        ),
        "native_source": {
            "dependency": "BlockType.hasSupport",
            "neighbourhood": "BlockFace.VALUES_26_connecting_offsets",
            "result": (
                "BlockPhysicsUtil.applyBlockPhysics_support_zero_dispatches_"
                "SupportDropType_BREAK_DESTROY_or_FALL"
            ),
        },
        "certificate": {
            "target": (
                "exact_present_non_filler_collision_boxes_inside_unit_cell"
            ),
            "neighbours": (
                "all_26_covered_and_semantically_unique_in_action_catalog"
            ),
            "safe": "no_neighbour_hasSupport",
            "support_dependent": "known_unsafe_not_unavailable",
            "unknown_or_protruding": "unavailable_and_safe_false",
        },
        "scope": (
            "positive_no_cascade_certificate_not_native_cascade_simulation"
        ),
    }


def region_removal_cascade_contract_sha256() -> str:
    payload = json.dumps(
        region_removal_cascade_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "REMOVAL_CASCADE_DIAGNOSTIC_NEIGHBOUR_UNAVAILABLE",
    "REMOVAL_CASCADE_DIAGNOSTIC_SUPPORT_DEPENDENT",
    "REMOVAL_CASCADE_DIAGNOSTIC_TARGET_NOT_SINGLE_CELL",
    "REMOVAL_CASCADE_DIAGNOSTIC_TARGET_UNAVAILABLE",
    "REMOVAL_CASCADE_SCHEMA",
    "REMOVAL_CASCADE_VERSION",
    "RegionRemovalCascadeSafety",
    "region_removal_cascade_contract",
    "region_removal_cascade_contract_sha256",
    "region_removal_cascade_safety",
]
