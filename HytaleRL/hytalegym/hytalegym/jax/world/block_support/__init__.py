"""JIT-safe exact support predicates and bounded dry block cascades."""

from __future__ import annotations

import hashlib
import json

import jax
import jax.numpy as jnp

from hytalegym.worldgen.block_actions import (
    SUPPORT_DROP_BREAK,
    SUPPORT_DROP_DESTROY,
    SUPPORT_DROP_FALL,
    SUPPORT_DROP_NONE,
)
from hytalegym.worldgen.block_support import (
    SUPPORT_MATCH_DISALLOWED,
    SUPPORT_MATCH_IGNORED,
    SUPPORT_MATCH_REQUIRED,
    SUPPORT_NEIGHBOUR_OFFSETS,
    block_support_contract_sha256 as host_block_support_contract_sha256,
)

# Re-exported to preserve this module's attribute surface.
from hytalegym.jax.world.block_support._catalog import (  # noqa: F401
    _validate_catalog,
    block_support_catalog_from_local,
)
from hytalegym.jax.world.block_support._primitives import (  # noqa: F401
    BlockSupportCascadeResult,
    BlockSupportCatalog,
    BlockSupportResult,
    _array,
    _bits,
    _capacity,
    _face_types,
    _vocabulary,
    support_cascade_seed_mask,
)


Array = jax.Array
BLOCK_SUPPORT_CASCADE_SCHEMA = "hytalerl_jax_block_support_cascade_v1"
BLOCK_SUPPORT_CASCADE_VERSION = 1

BLOCK_SUPPORT_DIAGNOSTIC_CATALOG = 1 << 0
BLOCK_SUPPORT_DIAGNOSTIC_NEIGHBOUR = 1 << 1
BLOCK_SUPPORT_DIAGNOSTIC_FLUID = 1 << 2
BLOCK_SUPPORT_DIAGNOSTIC_SINGLE_CELL = 1 << 3
BLOCK_SUPPORT_DIAGNOSTIC_CAPACITY = 1 << 4
BLOCK_SUPPORT_DIAGNOSTIC_INPUT = 1 << 5


def block_support_result(
    catalog: BlockSupportCatalog,
    catalog_index: Array,
    current_support: Array,
    neighbour_catalog_index: Array,
    neighbour_support: Array,
    neighbour_available: Array,
    neighbour_fluid_id: Array,
    neighbour_fluid_valid: Array,
) -> BlockSupportResult:
    """Evaluate ``BlockPhysicsUtil.testBlockPhysics`` on root-cell evidence."""

    capacity = _validate_catalog(catalog)
    index = jnp.asarray(catalog_index)
    support = jnp.asarray(current_support)
    if index.shape != support.shape:
        raise ValueError("catalog_index and current_support shapes differ")
    if index.dtype == jnp.bool_ or not jnp.issubdtype(
        index.dtype,
        jnp.integer,
    ):
        raise TypeError("catalog_index must be integer")
    neighbours = index.shape + (len(SUPPORT_NEIGHBOUR_OFFSETS),)
    neighbour_index = _array(
        neighbour_catalog_index,
        neighbours,
        jnp.int32,
        "neighbour_catalog_index",
    )
    neighbour_support = _array(
        neighbour_support,
        neighbours,
        jnp.int32,
        "neighbour_support",
    )
    neighbour_available = _array(
        neighbour_available,
        neighbours,
        jnp.bool_,
        "neighbour_available",
    )
    neighbour_fluid_id = _array(
        neighbour_fluid_id,
        neighbours,
        jnp.uint8,
        "neighbour_fluid_id",
    )
    neighbour_fluid_valid = _array(
        neighbour_fluid_valid,
        neighbours,
        jnp.bool_,
        "neighbour_fluid_valid",
    )
    index = index.astype(jnp.int32)
    support = support.astype(jnp.int32)
    index_valid = (index >= 0) & (index < capacity)
    safe = jnp.clip(index, 0, capacity - 1)
    support_valid = (support >= 0) & (support <= 15)
    entry_valid = index_valid & catalog.entry_mask[safe] & support_valid
    deco = support == 15

    group_mask = catalog.group_mask[safe]
    slots = catalog.group_neighbour_slot[safe].astype(jnp.int32)
    group_neighbour_index = jnp.take_along_axis(
        neighbour_index,
        slots,
        axis=-1,
    )
    group_neighbour_support = jnp.take_along_axis(
        neighbour_support,
        slots,
        axis=-1,
    )
    group_neighbour_available = jnp.take_along_axis(
        neighbour_available,
        slots,
        axis=-1,
    )
    group_neighbour_fluid = jnp.take_along_axis(
        neighbour_fluid_id,
        slots,
        axis=-1,
    )
    group_neighbour_fluid_valid = jnp.take_along_axis(
        neighbour_fluid_valid,
        slots,
        axis=-1,
    )
    neighbour_index_valid = (
        (group_neighbour_index >= -1)
        & (group_neighbour_index < capacity)
    )
    safe_neighbour = jnp.clip(
        group_neighbour_index,
        0,
        capacity - 1,
    )
    neighbour_present = (
        group_neighbour_index >= 0
    ) & catalog.entry_mask[safe_neighbour]
    neighbour_rotation = jnp.where(
        neighbour_present,
        catalog.rotation_index[safe_neighbour],
        0,
    ).astype(jnp.int32)
    neighbour_asset_key = jnp.where(
        neighbour_present[..., None],
        catalog.asset_key[safe_neighbour],
        0,
    )
    neighbour_tag_bits = jnp.where(
        neighbour_present,
        catalog.tag_bits[safe_neighbour],
        0,
    )
    neighbour_set_bits = jnp.where(
        neighbour_present,
        catalog.block_set_bits[safe_neighbour],
        0,
    )
    neighbour_material_empty = jnp.where(
        neighbour_present,
        catalog.material_empty[safe_neighbour],
        True,
    )
    neighbour_face = catalog.group_neighbour_face[safe].astype(jnp.int32)
    neighbour_offers = catalog.supporting_face_types[
        safe_neighbour,
        neighbour_rotation,
        neighbour_face,
    ]
    neighbour_offers = jnp.where(
        neighbour_present,
        neighbour_offers,
        0,
    )
    block_face = catalog.group_block_face[safe].astype(jnp.int32)
    self_offers = catalog.supporting_face_types[
        safe[..., None],
        neighbour_rotation,
        block_face,
    ]

    clause_mask = catalog.clause_mask[safe] & group_mask[..., None]
    face_mask = catalog.clause_face_type_mask[safe]
    self_face_mask = catalog.clause_self_face_type_mask[safe]
    block_valid = catalog.clause_block_type_valid[safe]
    block_key = catalog.clause_block_type_key[safe]
    fluid_required = catalog.clause_fluid_id[safe]
    tag_mask = catalog.clause_tag_mask[safe]
    set_mask = catalog.clause_block_set_mask[safe]
    match_self = catalog.clause_match_self[safe]
    support_match = catalog.clause_support[safe]
    propagation = catalog.clause_allow_support_propagation[safe]

    face_matches = (face_mask == 0) | (
        (neighbour_offers[..., None] & face_mask) != 0
    )
    self_face_matches = (self_face_mask == 0) | (
        (self_offers[..., None] & self_face_mask) != 0
    )
    block_matches = ~block_valid | (
        neighbour_present[..., None]
        & jnp.all(
            neighbour_asset_key[..., None, :] == block_key,
            axis=-1,
        )
    )
    tag_matches = (tag_mask == 0) | (
        (neighbour_tag_bits[..., None] & tag_mask) != 0
    )
    set_matches = (set_mask == 0) | (
        (neighbour_set_bits[..., None] & set_mask) != 0
    )
    same_asset = neighbour_present & jnp.all(
        neighbour_asset_key == catalog.asset_key[safe][..., None, :],
        axis=-1,
    )
    self_matches = (
        (match_self == SUPPORT_MATCH_IGNORED)
        | (
            (match_self == SUPPORT_MATCH_REQUIRED)
            & same_asset[..., None]
        )
        | (
            (match_self == SUPPORT_MATCH_DISALLOWED)
            & ~same_asset[..., None]
        )
    )
    fluid_matches = (fluid_required == 0) | (
        group_neighbour_fluid_valid[..., None]
        & neighbour_material_empty[..., None]
        & (group_neighbour_fluid[..., None] == fluid_required)
    )
    matched = (
        clause_mask
        & face_matches
        & self_face_matches
        & block_matches
        & tag_matches
        & set_matches
        & self_matches
        & fluid_matches
    )
    tested = jnp.any(
        clause_mask & (support_match != SUPPORT_MATCH_IGNORED),
        axis=(-2, -1),
    )
    required = jnp.any(
        matched & (support_match == SUPPORT_MATCH_REQUIRED),
        axis=-1,
    )
    disallowed = jnp.any(
        matched & (support_match == SUPPORT_MATCH_DISALLOWED),
        axis=-1,
    )
    direct = jnp.any(group_mask & required & ~disallowed, axis=-1)
    maximum = catalog.max_support_distance[safe].astype(jnp.int32)
    propagation_match = (
        matched
        & propagation
        & (maximum[..., None, None] > 0)
    )
    propagated_source = jnp.where(
        group_neighbour_support == 15,
        1,
        group_neighbour_support,
    )
    lowest = jnp.min(
        jnp.where(
            propagation_match,
            propagated_source[..., None],
            jnp.iinfo(jnp.int32).max,
        ),
        axis=(-2, -1),
    )
    distance = lowest + 1
    propagated = (
        ~direct
        & (lowest < jnp.iinfo(jnp.int32).max)
        & (distance <= maximum)
    )
    dependent = catalog.support_dependent[safe]
    unsupported = dependent & tested & ~direct & ~propagated
    fluid_unknown = jnp.any(
        clause_mask
        & (fluid_required != 0)
        & ~group_neighbour_fluid_valid[..., None],
        axis=(-2, -1),
    )
    neighbour_unknown = jnp.any(
        group_mask
        & (
            ~group_neighbour_available
            | ~neighbour_index_valid
            | (
                (group_neighbour_index >= 0)
                & (
                    ~catalog.entry_mask[safe_neighbour]
                    | (group_neighbour_support < 0)
                    | (group_neighbour_support > 15)
                )
            )
        ),
        axis=-1,
    )
    available = entry_valid & (
        deco | (~neighbour_unknown & ~fluid_unknown)
    )
    direct &= available & ~deco
    propagated &= available & ~deco
    unsupported &= available & ~deco
    ignored = available & (
        deco | (~direct & ~propagated & ~unsupported)
    )
    stable = available & ~unsupported
    support_after = jnp.where(
        direct,
        0,
        jnp.where(propagated, distance, support),
    ).astype(jnp.int32)
    diagnostics = (
        jnp.where(
            ~entry_valid,
            jnp.uint32(BLOCK_SUPPORT_DIAGNOSTIC_CATALOG),
            jnp.uint32(0),
        )
        | jnp.where(
            neighbour_unknown & ~deco,
            jnp.uint32(BLOCK_SUPPORT_DIAGNOSTIC_NEIGHBOUR),
            jnp.uint32(0),
        )
        | jnp.where(
            fluid_unknown & ~deco,
            jnp.uint32(BLOCK_SUPPORT_DIAGNOSTIC_FLUID),
            jnp.uint32(0),
        )
    )
    return BlockSupportResult(
        available=available,
        stable=stable,
        ignored=ignored,
        direct=direct,
        propagated=propagated,
        unsupported=unsupported,
        support_after=jnp.where(available, support_after, support),
        support_drop_type=jnp.where(
            unsupported,
            catalog.support_drop_type[safe],
            SUPPORT_DROP_NONE,
        ).astype(jnp.uint8),
        diagnostics=diagnostics,
    )


def block_support_cascade(
    catalog: BlockSupportCatalog,
    *,
    cell_mask: Array,
    cell_position: Array,
    catalog_index: Array,
    support_value: Array,
    single_cell: Array,
    initial_remove: Array,
    initial_dirty: Array,
    coverage_min: Array,
    coverage_max_exclusive: Array,
    dry_world: Array,
) -> BlockSupportCascadeResult:
    """Resolve a bounded native-equivalent cascade or reject the whole row."""

    _validate_catalog(catalog)
    mask = jnp.asarray(cell_mask)
    position = jnp.asarray(cell_position, dtype=jnp.int32)
    if mask.ndim != 2 or mask.dtype != jnp.bool_:
        raise ValueError("cell_mask must be boolean [batch, cell]")
    batch, cells = mask.shape
    if position.shape != (batch, cells, 3):
        raise ValueError("cell_position must have shape [batch, cell, 3]")
    index = _array(catalog_index, mask.shape, jnp.int32, "catalog_index")
    support = _array(
        support_value,
        mask.shape,
        jnp.int32,
        "support_value",
    )
    single = _array(single_cell, mask.shape, jnp.bool_, "single_cell")
    removed_initial = _array(
        initial_remove,
        mask.shape,
        jnp.bool_,
        "initial_remove",
    )
    dirty_initial = _array(
        initial_dirty,
        mask.shape,
        jnp.bool_,
        "initial_dirty",
    )
    lower = _array(coverage_min, (batch, 3), jnp.int32, "coverage_min")
    upper = _array(
        coverage_max_exclusive,
        (batch, 3),
        jnp.int32,
        "coverage_max_exclusive",
    )
    dry = _array(dry_world, (batch,), jnp.bool_, "dry_world")

    catalog_capacity = catalog.entry_mask.shape[0]
    catalog_index_valid = (index >= 0) & (index < catalog_capacity)
    safe_index = jnp.clip(index, 0, catalog_capacity - 1)
    catalog_entry_valid = (
        catalog_index_valid & catalog.entry_mask[safe_index]
    )
    duplicate = jnp.any(
        mask[:, :, None]
        & mask[:, None, :]
        & ~jnp.eye(cells, dtype=jnp.bool_)[None, :, :]
        & jnp.all(
            position[:, :, None, :] == position[:, None, :, :],
            axis=-1,
        ),
        axis=(1, 2),
    )
    positions_inside = jnp.all(
        (position >= lower[:, None, :])
        & (position < upper[:, None, :]),
        axis=-1,
    )
    input_invalid = (
        duplicate
        | jnp.any(mask & ~catalog_entry_valid, axis=1)
        | jnp.any(mask & ~positions_inside, axis=1)
        | jnp.any(mask & ~single, axis=1)
        | jnp.any(mask & ((support < 0) | (support > 15)), axis=1)
        | jnp.any(removed_initial & ~mask, axis=1)
        | jnp.any(dirty_initial & ~mask, axis=1)
        | jnp.any(upper <= lower, axis=1)
        | ~dry
    )
    active_initial = mask & ~removed_initial
    valid_initial = ~input_invalid
    zeros = jnp.zeros_like(mask)
    state = (
        active_initial,
        removed_initial,
        zeros,
        zeros,
        zeros,
        support,
        dirty_initial & active_initial,
        valid_initial,
        jnp.zeros(batch, dtype=jnp.int32),
        jnp.zeros(batch, dtype=jnp.bool_),
        jnp.zeros(batch, dtype=jnp.uint32),
    )
    offsets = jnp.asarray(SUPPORT_NEIGHBOUR_OFFSETS, dtype=jnp.int32)

    def step(iteration: int, carry: tuple[Array, ...]) -> tuple[Array, ...]:
        (
            active,
            removed,
            broken,
            destroyed,
            falling,
            current_support,
            dirty,
            valid,
            iterations,
            overflow,
            diagnostics,
        ) = carry
        neighbour_position = position[:, :, None, :] + offsets
        matches = active[:, None, None, :] & jnp.all(
            neighbour_position[:, :, :, None, :]
            == position[:, None, None, :, :],
            axis=-1,
        )
        match_count = jnp.sum(matches, axis=-1)
        slot = jnp.argmax(matches, axis=-1)
        inside = jnp.all(
            (neighbour_position >= lower[:, None, None, :])
            & (neighbour_position < upper[:, None, None, :]),
            axis=-1,
        )
        neighbour_index = jnp.where(
            match_count == 1,
            jnp.take_along_axis(
                index[:, None, :],
                slot,
                axis=2,
            ),
            -1,
        )
        neighbour_support = jnp.where(
            match_count == 1,
            jnp.take_along_axis(
                current_support[:, None, :],
                slot,
                axis=2,
            ),
            0,
        )
        predicate = block_support_result(
            catalog,
            index,
            current_support,
            neighbour_index,
            neighbour_support,
            inside,
            jnp.zeros_like(neighbour_index, dtype=jnp.uint8),
            jnp.ones_like(neighbour_index, dtype=jnp.bool_),
        )
        dependent = catalog.support_dependent[
            jnp.clip(index, 0, catalog.entry_mask.shape[0] - 1)
        ]
        relevant = dirty & active & dependent
        unavailable = relevant & (~predicate.available | ~single)
        row_unavailable = jnp.any(unavailable, axis=1)
        dispatch = relevant & predicate.unsupported
        support_change = (
            relevant
            & predicate.stable
            & (predicate.support_after != current_support)
        )
        changed = dispatch | support_change
        apply = iteration < cells
        applied_dispatch = dispatch & apply
        applied_support = support_change & apply
        drop = predicate.support_drop_type
        next_active = active & ~applied_dispatch
        next_removed = removed | applied_dispatch
        next_broken = broken | (
            applied_dispatch & (drop == SUPPORT_DROP_BREAK)
        )
        next_destroyed = destroyed | (
            applied_dispatch & (drop == SUPPORT_DROP_DESTROY)
        )
        next_falling = falling | (
            applied_dispatch & (drop == SUPPORT_DROP_FALL)
        )
        next_support = jnp.where(
            applied_support,
            predicate.support_after,
            current_support,
        )
        delta = jnp.max(
            jnp.abs(position[:, :, None, :] - position[:, None, :, :]),
            axis=-1,
        )
        next_dirty = next_active & jnp.any(
            (changed & apply)[:, None, :] & (delta <= 1),
            axis=-1,
        )
        next_valid = valid & ~row_unavailable
        next_overflow = overflow | (
            (iteration == cells) & jnp.any(changed, axis=1)
        )
        next_diagnostics = diagnostics | jnp.where(
            jnp.any(relevant & ~predicate.available, axis=1),
            jnp.uint32(
                BLOCK_SUPPORT_DIAGNOSTIC_NEIGHBOUR
                | BLOCK_SUPPORT_DIAGNOSTIC_FLUID
                | BLOCK_SUPPORT_DIAGNOSTIC_CATALOG
            ),
            jnp.uint32(0),
        ) | jnp.where(
            jnp.any(relevant & ~single, axis=1),
            jnp.uint32(BLOCK_SUPPORT_DIAGNOSTIC_SINGLE_CELL),
            jnp.uint32(0),
        )
        return (
            next_active,
            next_removed,
            next_broken,
            next_destroyed,
            next_falling,
            next_support,
            next_dirty,
            next_valid,
            iterations
            + (jnp.any(relevant, axis=1) & apply).astype(jnp.int32),
            next_overflow,
            next_diagnostics,
        )

    (
        _active,
        removed,
        broken,
        destroyed,
        falling,
        support_after,
        _dirty,
        valid,
        iterations,
        overflow,
        diagnostics,
    ) = jax.lax.fori_loop(0, cells + 1, step, state)
    available = valid & ~overflow
    diagnostics |= jnp.where(
        input_invalid,
        jnp.uint32(BLOCK_SUPPORT_DIAGNOSTIC_INPUT),
        jnp.uint32(0),
    ) | jnp.where(
        overflow,
        jnp.uint32(BLOCK_SUPPORT_DIAGNOSTIC_CAPACITY),
        jnp.uint32(0),
    )
    effect_mask = available[:, None]
    return BlockSupportCascadeResult(
        available=available,
        removed=removed & effect_mask,
        broken=broken & effect_mask,
        destroyed=destroyed & effect_mask,
        falling=falling & effect_mask,
        support_after=jnp.where(effect_mask, support_after, support_value),
        iterations=iterations,
        capacity_exceeded=overflow,
        diagnostics=diagnostics,
    )


def block_support_cascade_contract() -> dict[str, object]:
    return {
        "schema": BLOCK_SUPPORT_CASCADE_SCHEMA,
        "version": BLOCK_SUPPORT_CASCADE_VERSION,
        "server_version": "0.5.7",
        "host_contract_sha256": host_block_support_contract_sha256(),
        "execution": {
            "shape": "fixed_batch_cell_capacity",
            "frontier": "native_26_neighbour_updates",
            "support": (
                "direct_propagated_ignored_or_unsupported_exact_predicate"
            ),
            "dispatch": "BREAK_DESTROY_and_FALL_are_distinct_masks",
            "termination": "one_final_nonmutating_overflow_probe",
        },
        "scope": "covered_dry_single_cell_root_blocks",
        "failure": {
            "catalog_shape_or_dtype": "host_error_before_jit",
            "unknown_catalog": "fail_closed",
            "uncovered_neighbour": "fail_closed",
            "fluid_or_filler": "fail_closed",
            "duplicate_position": "fail_closed",
            "iteration_capacity": "fail_closed_no_partial_effects",
        },
        "effects": {
            "drops": "resolved_by_existing_block_action_catalog",
            "falling_trajectory": "existing_falling_impact_dispatch_boundary",
            "mutation_write": "existing_atomic_mutable_block_update_boundary",
        },
    }


def block_support_cascade_contract_sha256() -> str:
    payload = json.dumps(
        block_support_cascade_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "BLOCK_SUPPORT_CASCADE_SCHEMA",
    "BLOCK_SUPPORT_CASCADE_VERSION",
    "BLOCK_SUPPORT_DIAGNOSTIC_CAPACITY",
    "BLOCK_SUPPORT_DIAGNOSTIC_CATALOG",
    "BLOCK_SUPPORT_DIAGNOSTIC_FLUID",
    "BLOCK_SUPPORT_DIAGNOSTIC_INPUT",
    "BLOCK_SUPPORT_DIAGNOSTIC_NEIGHBOUR",
    "BLOCK_SUPPORT_DIAGNOSTIC_SINGLE_CELL",
    "BlockSupportCascadeResult",
    "BlockSupportCatalog",
    "BlockSupportResult",
    "block_support_cascade",
    "block_support_cascade_contract",
    "block_support_cascade_contract_sha256",
    "block_support_catalog_from_local",
    "block_support_result",
    "support_cascade_seed_mask",
]
