"""Host orchestration for native block-damaging explosion evidence.

The bridge response is post-mutation evidence, never input to JAX's
pre-mutation planner. Complete evidence must be followed immediately on the
same connection by one targeted ``mutable_block_cells`` query. That query
supplies full geometry for ``MutableBlockUpdate``;
``commit_region_block_mutation`` remains the sole Region-state writer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

import numpy as np

from hytalegym.worldgen.native_explosion_mutation_contract import (
    NATIVE_EXPLOSION_MUTATION_CELL_CAPACITY,
    NATIVE_EXPLOSION_MUTATION_DROP_CAPACITY,
    NATIVE_EXPLOSION_MUTATION_SCHEMA,
    NATIVE_EXPLOSION_MUTATION_VERSION,
    NATIVE_METHOD,
    NativeExplosionFixtureKind,
    NativeExplosionMutationCapture,
    NativeExplosionMutationCellState,
    NativeExplosionMutationChangedCell,
    NativeExplosionMutationDrop,
    NativeExplosionMutationRequest,
    synthetic_config_semantic_sha256,
    synthetic_config_semantics,
)
from hytalegym.worldgen.native_mutable_blocks import (
    NativeMutableBlockCells,
    NativeMutableBlockRow,
    native_mutable_block_cells_contract_sha256,
    native_mutable_block_cells_request,
)


_EXPLOSION_UTILS_SHA256 = (
    "03FE187757565EA414053D795136044F3362246B26136B5B1A6387DDA2F6D4B5"
)


@dataclass(frozen=True, slots=True)
class NativeExplosionMutationRequeryPlan:
    """Exact targeted refresh that must immediately follow one receipt."""

    bridge_sha256: str
    world_epoch: str
    positions: tuple[tuple[int, int, int], ...]
    same_connection_required: bool = field(default=True, init=False)
    intervening_step_or_reset_allowed: bool = field(default=False, init=False)

    @property
    def no_op(self) -> bool:
        return not self.positions

    def to_message(self) -> dict[str, object] | None:
        return (
            None
            if self.no_op
            else native_mutable_block_cells_request(
                self.positions,
                expected_world_epoch=self.world_epoch,
            )
        )


def native_explosion_mutation_requery_plan(
    capture: NativeExplosionMutationCapture,
) -> NativeExplosionMutationRequeryPlan:
    """Plan the only supported post-capture geometry refresh."""

    if not isinstance(capture, NativeExplosionMutationCapture):
        raise TypeError("capture must be a native explosion mutation capture")
    if not capture.complete:
        if capture.resync_required:
            raise ValueError("executed incomplete evidence requires a full resync")
        raise ValueError("preflight rejection has no targeted requery")
    return NativeExplosionMutationRequeryPlan(
        capture.bridge_sha256,
        capture.request.world_epoch,
        tuple(cell.position for cell in capture.changed_cells),
    )


def validate_native_explosion_mutation_requery(
    capture: NativeExplosionMutationCapture,
    cells: NativeMutableBlockCells,
) -> tuple[NativeMutableBlockRow, ...]:
    """Join a receipt to full geometry before one existing update/commit.

    The caller is responsible for executing the plan on the same socket with
    no intervening environment command; bridge/world/seed/order/scalars are
    then checked here before any Region state may be committed.
    """

    plan = native_explosion_mutation_requery_plan(capture)
    if plan.no_op:
        raise ValueError("a complete no-op mutation has no targeted requery")
    if not isinstance(cells, NativeMutableBlockCells):
        raise TypeError("cells must be NativeMutableBlockCells")
    identity = (
        cells.bridge_sha256,
        cells.server_version,
        cells.world,
        cells.worldgen_provider,
        cells.worldgen_version,
        cells.seed,
    )
    expected = (
        capture.bridge_sha256,
        capture.server_version,
        capture.world,
        capture.worldgen_provider,
        capture.worldgen_version,
        capture.seed,
    )
    if identity != expected:
        raise ValueError("targeted requery came from another native world")
    positions = tuple(tuple(int(v) for v in cell.position) for cell in cells.cells)
    if positions != plan.positions:
        raise ValueError("targeted requery changed mutation row order")
    rows: list[NativeMutableBlockRow] = []
    for changed, cell in zip(capture.changed_cells, cells.cells, strict=True):
        if not cell.available or cell.row is None:
            raise ValueError("targeted mutation cell became unavailable")
        if not _after_matches(changed.after, cell.row):
            raise ValueError("targeted requery disagrees with mutation after state")
        rows.append(cell.row)
    return tuple(rows)


def native_explosion_mutation_host_contract() -> dict[str, object]:
    """Return the checkpoint-pinnable host validation boundary."""

    return {
        "schema": "hytalerl_native_explosion_mutation_host_v1",
        "version": 1,
        "wire_schema": NATIVE_EXPLOSION_MUTATION_SCHEMA,
        "native_method": NATIVE_METHOD,
        "source_class": "com.hypixel.hytale.server.core.entity.ExplosionUtils",
        "source_class_sha256": _EXPLOSION_UTILS_SHA256,
        "request": {
            "caller_fields": (
                "world_epoch",
                "fixture_kind",
                "cell_capacity",
                "drop_capacity",
            ),
            "server_owned": (
                "origin",
                "configuration_source_and_sha256",
                "damage_scalars",
            ),
            "fixture_semantic_sha256": {
                kind: synthetic_config_semantic_sha256(kind)
                for kind in ("direct_drop",)
            },
        },
        "scope": {
            "damage_entities": False,
            "dry_direct_cells": True,
            "cell_capacity": NATIVE_EXPLOSION_MUTATION_CELL_CAPACITY,
            "fixtures": ("direct_drop",),
        },
        "incomplete": {
            "preflight": "empty;execution_started=false;resync_required=false",
            "executed": "empty;execution_started=true;resync_required=true",
        },
        "geometry_refresh": {
            "request_contract_sha256": native_mutable_block_cells_contract_sha256(),
            "sequence": "same_connection_no_step_or_reset",
            "expected_world_epoch": "required_equals_mutation_request_epoch",
            "join": "bridge_world_seed_position_order_and_after_scalars",
        },
        "state_projection": {
            "update": "hytalegym.jax.world.mutable_blocks.MutableBlockUpdate",
            "sole_writer": (
                "hytalegym.jax.world.region.action_runtime."
                "commit_region_block_mutation"
            ),
            "second_overlay": False,
        },
        "planner_relation": "post_mutation_oracle_not_planner_input",
        "fail_closed": True,
    }


def native_explosion_mutation_host_contract_sha256() -> str:
    payload = json.dumps(
        native_explosion_mutation_host_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _after_matches(
    after: NativeExplosionMutationCellState,
    row: NativeMutableBlockRow,
) -> bool:
    semantic = (
        np.asarray(row.semantic_key, dtype=">u4").tobytes()
        if row.semantic_key_valid
        else b""
    )
    return after[:18] == (
        row.block_present,
        row.block_asset_id,
        row.runtime_block_id,
        semantic,
        row.semantic_key_valid,
        row.affordance_valid,
        row.affordance_tags,
        row.gather_type_index,
        row.required_tool_quality,
        row.rotation_index,
        row.flags,
        row.fluid_level,
        row.fluid_fill_height,
        row.support,
        row.block_damage,
        row.fluid_damage,
        row.block_health,
        row.block_health_valid,
    )


__all__ = [
    "NATIVE_EXPLOSION_MUTATION_CELL_CAPACITY",
    "NATIVE_EXPLOSION_MUTATION_DROP_CAPACITY",
    "NATIVE_EXPLOSION_MUTATION_SCHEMA",
    "NATIVE_EXPLOSION_MUTATION_VERSION",
    "NativeExplosionFixtureKind",
    "NativeExplosionMutationCapture",
    "NativeExplosionMutationCellState",
    "NativeExplosionMutationChangedCell",
    "NativeExplosionMutationDrop",
    "NativeExplosionMutationRequest",
    "NativeExplosionMutationRequeryPlan",
    "native_explosion_mutation_host_contract",
    "native_explosion_mutation_host_contract_sha256",
    "native_explosion_mutation_requery_plan",
    "synthetic_config_semantic_sha256",
    "synthetic_config_semantics",
    "validate_native_explosion_mutation_requery",
]
