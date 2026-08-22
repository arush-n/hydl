"""Atomic mutable-world publication of exact dry block-support cascades."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.world.block_actions import (
    BlockMutationAcknowledgement,
    exact_block_mutation_acknowledgement,
)
from hytalegym.jax.world.block_support import (
    BlockSupportCascadeResult,
    BlockSupportCatalog,
    block_support_cascade,
    block_support_cascade_contract_sha256,
    support_cascade_seed_mask,
)
from hytalegym.jax.world.mutable_blocks import (
    MUTABLE_BLOCK_PROVENANCE_NONE,
    MUTABLE_BLOCK_PROVENANCE_SURROGATE,
    MutableBlockGeometry,
    MutableBlockQueryResult,
    MutableBlockState,
    MutableBlockUpdate,
    MutableBlockUpdateResult,
    apply_mutable_block_updates,
    empty_mutable_block_geometry,
    empty_mutable_block_update,
    mutable_block_contract_sha256,
    query_mutable_blocks,
)


Array = jax.Array

BLOCK_SUPPORT_MUTATION_SCHEMA = "hytalerl_block_support_mutation_v1"
BLOCK_SUPPORT_MUTATION_VERSION = 1

BLOCK_SUPPORT_MUTATION_DIAGNOSTIC_QUERY = 1 << 0
BLOCK_SUPPORT_MUTATION_DIAGNOSTIC_CATALOG = 1 << 1
BLOCK_SUPPORT_MUTATION_DIAGNOSTIC_ROOT = 1 << 2
BLOCK_SUPPORT_MUTATION_DIAGNOSTIC_EFFECT = 1 << 3
BLOCK_SUPPORT_MUTATION_DIAGNOSTIC_COMMIT = 1 << 4
BLOCK_SUPPORT_MUTATION_DIAGNOSTIC_SOURCE = 1 << 5


class BlockSupportCascadePlan(NamedTuple):
    """Revision-bound cascade transaction before external effects commit."""

    available: Array
    before: MutableBlockQueryResult
    cascade: BlockSupportCascadeResult
    update: MutableBlockUpdate
    diagnostics: Array


class BlockSupportEffectReadiness(NamedTuple):
    """Typed proof that every non-geometry effect is ready to commit."""

    break_ready: Array
    destroy_ready: Array
    falling_state_ready: Array


class BlockSupportMutationResult(NamedTuple):
    """Atomic commit result; ``available`` is the only success signal."""

    available: Array
    effects_complete: Array
    mutation: MutableBlockUpdateResult
    acknowledgement: BlockMutationAcknowledgement
    diagnostics: Array


def plan_block_support_mutation(
    state: MutableBlockState,
    catalog: BlockSupportCatalog,
    *,
    cell_mask: Array,
    cell_position: Array,
    single_cell: Array,
    initial_remove: Array,
    coverage_min: Array,
    coverage_max_exclusive: Array,
    source_complete: Array,
    dry_world: Array,
    base_available: Array,
    base_geometry: MutableBlockGeometry,
    base_block_health: Array,
) -> BlockSupportCascadePlan:
    """Plan one exact final cascade against the current mutable revision."""

    if not isinstance(state, MutableBlockState):
        raise TypeError("state must be MutableBlockState")
    if not isinstance(catalog, BlockSupportCatalog):
        raise TypeError("catalog must be BlockSupportCatalog")
    mask = _array(cell_mask, jnp.bool_, 2, "cell_mask")
    batch, cells = mask.shape
    position = _array(
        cell_position,
        jnp.int32,
        3,
        "cell_position",
        shape=(batch, cells, 3),
    )
    one_cell = _array(
        single_cell,
        jnp.bool_,
        2,
        "single_cell",
        shape=mask.shape,
    )
    roots = _array(
        initial_remove,
        jnp.bool_,
        2,
        "initial_remove",
        shape=mask.shape,
    )
    lower = _array(
        coverage_min,
        jnp.int32,
        2,
        "coverage_min",
        shape=(batch, 3),
    )
    upper = _array(
        coverage_max_exclusive,
        jnp.int32,
        2,
        "coverage_max_exclusive",
        shape=(batch, 3),
    )
    complete = _array(
        source_complete,
        jnp.bool_,
        1,
        "source_complete",
        shape=(batch,),
    )
    dry = _array(
        dry_world,
        jnp.bool_,
        1,
        "dry_world",
        shape=(batch,),
    )
    available = _array(
        base_available,
        jnp.bool_,
        2,
        "base_available",
        shape=mask.shape,
    )
    health = _array(
        base_block_health,
        jnp.float32,
        2,
        "base_block_health",
        shape=mask.shape,
    )

    before = query_mutable_blocks(
        state,
        world_id=state.world_id,
        base_semantic_sha256=state.base_semantic_sha256,
        resync_epoch=state.resync_epoch,
        position=position,
        base_available=available,
        base_geometry=base_geometry,
        base_block_health=health,
    )
    matches = (
        catalog.entry_mask[None, None, :]
        & jnp.all(
            before.geometry.semantic_key[:, :, None, :]
            == catalog.semantic_key[None, None, :, :],
            axis=-1,
        )
    )
    match_count = jnp.sum(matches, axis=-1)
    unique_catalog = match_count == 1
    catalog_index = jnp.where(
        mask & unique_catalog,
        jnp.argmax(matches, axis=-1),
        0,
    ).astype(jnp.int32)
    query_valid = (
        before.available
        & before.geometry.exact
        & before.geometry.block_present
        & before.geometry.semantic_key_valid
    )
    root_valid = (
        jnp.any(roots, axis=1)
        & ~jnp.any(roots & ~(mask & query_valid & one_cell), axis=1)
    )
    row_query_valid = ~jnp.any(mask & ~query_valid, axis=1)
    row_catalog_valid = ~jnp.any(mask & ~unique_catalog, axis=1)
    observed_dry = ~jnp.any(
        mask
        & (
            (before.geometry.fluid_level != 0)
            | (before.geometry.fluid_fill_height != 0.0)
        ),
        axis=1,
    )
    input_valid = (
        root_valid
        & row_query_valid
        & row_catalog_valid
        & complete
        & observed_dry
    )

    dirty = support_cascade_seed_mask(
        position,
        mask,
        position,
        roots,
    )
    cascade = block_support_cascade(
        catalog,
        cell_mask=mask,
        cell_position=position,
        catalog_index=catalog_index,
        support_value=before.geometry.support,
        single_cell=one_cell,
        initial_remove=roots,
        initial_dirty=dirty,
        coverage_min=lower,
        coverage_max_exclusive=upper,
        dry_world=dry & complete & observed_dry,
    )
    plan_available = input_valid & cascade.available
    row_gate = plan_available[:, None]
    removed = cascade.removed & row_gate
    support_changed = (
        mask
        & ~removed
        & (cascade.support_after != before.geometry.support)
        & row_gate
    )
    changed = removed | support_changed
    support_geometry = before.geometry._replace(
        support=jnp.where(
            support_changed,
            cascade.support_after,
            before.geometry.support,
        )
    )
    final_geometry = _select_geometry(
        removed,
        empty_mutable_block_geometry((batch, cells), exact=True),
        support_geometry,
    )
    update = empty_mutable_block_update(
        state,
        query_capacity=cells,
    )._replace(
        mask=changed,
        position=position,
        geometry_changed=changed,
        health_changed=removed,
        expected_health=before.block_health,
        block_health_after=jnp.where(
            removed,
            jnp.float32(0.0),
            before.block_health,
        ),
        provenance=jnp.where(
            changed,
            jnp.uint8(MUTABLE_BLOCK_PROVENANCE_SURROGATE),
            jnp.uint8(MUTABLE_BLOCK_PROVENANCE_NONE),
        ),
        geometry=final_geometry,
    )
    diagnostics = (
        jnp.where(
            ~row_query_valid,
            jnp.uint32(BLOCK_SUPPORT_MUTATION_DIAGNOSTIC_QUERY),
            jnp.uint32(0),
        )
        | jnp.where(
            ~row_catalog_valid,
            jnp.uint32(BLOCK_SUPPORT_MUTATION_DIAGNOSTIC_CATALOG),
            jnp.uint32(0),
        )
        | jnp.where(
            ~root_valid,
            jnp.uint32(BLOCK_SUPPORT_MUTATION_DIAGNOSTIC_ROOT),
            jnp.uint32(0),
        )
        | jnp.where(
            ~complete | ~observed_dry,
            jnp.uint32(BLOCK_SUPPORT_MUTATION_DIAGNOSTIC_SOURCE),
            jnp.uint32(0),
        )
    )
    return BlockSupportCascadePlan(
        available=plan_available,
        before=before,
        cascade=cascade,
        update=update,
        diagnostics=diagnostics,
    )


def empty_block_support_effect_readiness(
    plan: BlockSupportCascadePlan,
) -> BlockSupportEffectReadiness:
    """Return an honest value with no prepared external effect."""

    if not isinstance(plan, BlockSupportCascadePlan):
        raise TypeError("plan must be BlockSupportCascadePlan")
    zeros = jnp.zeros_like(plan.cascade.removed)
    return BlockSupportEffectReadiness(zeros, zeros, zeros)


def commit_block_support_mutation(
    state: MutableBlockState,
    plan: BlockSupportCascadePlan,
    effects: BlockSupportEffectReadiness,
    *,
    base_available: Array,
    base_geometry: MutableBlockGeometry,
    base_block_health: Array,
) -> tuple[MutableBlockState, BlockSupportMutationResult]:
    """Commit geometry only after every typed cascade effect is complete."""

    if not isinstance(state, MutableBlockState):
        raise TypeError("state must be MutableBlockState")
    if not isinstance(plan, BlockSupportCascadePlan):
        raise TypeError("plan must be BlockSupportCascadePlan")
    if not isinstance(effects, BlockSupportEffectReadiness):
        raise TypeError(
            "effects must be BlockSupportEffectReadiness"
        )
    shape = plan.cascade.removed.shape
    break_ready = _array(
        effects.break_ready,
        jnp.bool_,
        2,
        "effects.break_ready",
        shape=shape,
    )
    destroy_ready = _array(
        effects.destroy_ready,
        jnp.bool_,
        2,
        "effects.destroy_ready",
        shape=shape,
    )
    falling_ready = _array(
        effects.falling_state_ready,
        jnp.bool_,
        2,
        "effects.falling_state_ready",
        shape=shape,
    )
    gate = plan.available[:, None]
    expected_break = plan.cascade.broken & gate
    expected_destroy = plan.cascade.destroyed & gate
    expected_falling = plan.cascade.falling & gate
    effects_complete = (
        jnp.all(break_ready == expected_break, axis=1)
        & jnp.all(destroy_ready == expected_destroy, axis=1)
        & jnp.all(falling_ready == expected_falling, axis=1)
    )
    safe_update = plan.update._replace(
        mask=plan.update.mask & effects_complete[:, None]
    )
    next_state, mutation = apply_mutable_block_updates(
        state,
        base_available=base_available,
        base_geometry=base_geometry,
        base_block_health=base_block_health,
        update=safe_update,
    )
    acknowledgement = exact_block_mutation_acknowledgement(mutation)
    result_available = (
        plan.available
        & effects_complete
        & acknowledgement.available
    )
    diagnostics = (
        plan.diagnostics
        | jnp.where(
            plan.available & ~effects_complete,
            jnp.uint32(BLOCK_SUPPORT_MUTATION_DIAGNOSTIC_EFFECT),
            jnp.uint32(0),
        )
        | jnp.where(
            plan.available
            & effects_complete
            & ~acknowledgement.available,
            jnp.uint32(BLOCK_SUPPORT_MUTATION_DIAGNOSTIC_COMMIT),
            jnp.uint32(0),
        )
    )
    return next_state, BlockSupportMutationResult(
        available=result_available,
        effects_complete=effects_complete,
        mutation=mutation,
        acknowledgement=acknowledgement,
        diagnostics=diagnostics,
    )


def block_support_mutation_contract() -> dict[str, object]:
    """Return the checkpoint-pinnable two-phase mutation contract."""

    return {
        "schema": BLOCK_SUPPORT_MUTATION_SCHEMA,
        "version": BLOCK_SUPPORT_MUTATION_VERSION,
        "server_version": "0.5.7",
        "dependencies": {
            "support_cascade": block_support_cascade_contract_sha256(),
            "mutable_blocks": mutable_block_contract_sha256(),
        },
        "catalog_lookup": (
            "exact_unique_portable_semantic_key_never_runtime_ordinal"
        ),
        "source": {
            "complete": "explicit_true_or_fail_closed",
            "dry": "declared_true_and_observed_zero_fluid",
        },
        "transaction": {
            "plan": "revision_bound_complete_final_cascade",
            "commit": "all_geometry_changes_atomically_or_none",
            "provenance": "surrogate_source_exact_not_native_runtime",
            "initial_remove": (
                "primary_action_effect_prepared_by_outer_consumer"
            ),
        },
        "required_effect_readiness": {
            "break": "drops_and_break_event_prepared_per_cell",
            "destroy": "destroy_event_prepared_per_cell",
            "fall": "falling_entity_state_prepared_per_cell",
            "extra_or_missing_readiness": "fail_closed",
        },
        "outer_commit": (
            "consumer_commits_prepared_effects_with_candidate_geometry"
        ),
        "scope": "covered_dry_single_cell_final_support_cascades",
        "not_claimed": [
            "native_per_tick_scheduling",
            "falling_entity_trajectory",
            "explosion_effects",
            "fluid_or_filler_support",
        ],
    }


def block_support_mutation_contract_sha256() -> str:
    payload = json.dumps(
        block_support_mutation_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _select_geometry(
    mask: Array,
    when_true: MutableBlockGeometry,
    when_false: MutableBlockGeometry,
) -> MutableBlockGeometry:
    def select(true_value: Array, false_value: Array) -> Array:
        expanded = mask.reshape(
            mask.shape + (1,) * (true_value.ndim - mask.ndim)
        )
        return jnp.where(expanded, true_value, false_value)

    return jax.tree.map(select, when_true, when_false)


def _array(
    value: Array,
    dtype,
    ndim: int,
    label: str,
    *,
    shape: tuple[int, ...] | None = None,
) -> Array:
    result = jnp.asarray(value)
    if result.dtype != jnp.dtype(dtype):
        raise ValueError(f"{label} must have dtype {jnp.dtype(dtype)}")
    if result.ndim != ndim or (shape is not None and result.shape != shape):
        expected = f"shape {shape}" if shape is not None else f"rank {ndim}"
        raise ValueError(f"{label} must have {expected}")
    return result


__all__ = [
    "BLOCK_SUPPORT_MUTATION_DIAGNOSTIC_CATALOG",
    "BLOCK_SUPPORT_MUTATION_DIAGNOSTIC_COMMIT",
    "BLOCK_SUPPORT_MUTATION_DIAGNOSTIC_EFFECT",
    "BLOCK_SUPPORT_MUTATION_DIAGNOSTIC_QUERY",
    "BLOCK_SUPPORT_MUTATION_DIAGNOSTIC_ROOT",
    "BLOCK_SUPPORT_MUTATION_DIAGNOSTIC_SOURCE",
    "BLOCK_SUPPORT_MUTATION_SCHEMA",
    "BLOCK_SUPPORT_MUTATION_VERSION",
    "BlockSupportCascadePlan",
    "BlockSupportEffectReadiness",
    "BlockSupportMutationResult",
    "block_support_mutation_contract",
    "block_support_mutation_contract_sha256",
    "commit_block_support_mutation",
    "empty_block_support_effect_readiness",
    "plan_block_support_mutation",
]
