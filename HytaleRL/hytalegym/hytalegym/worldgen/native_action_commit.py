"""Bind a typed native World-verb receipt to its exact live-cell capture."""

from __future__ import annotations

import hashlib
import json

from hytalegym.worldgen.native_mutable_blocks import (
    NativeMutableBlockCells,
    NativeMutableBlockRow,
    native_mutable_block_cells_contract_sha256,
)
from hytalegym.worldgen.native_world_actions import (
    NativeWorldVerbReceipt,
    native_world_verb_transport_contract_sha256,
)


NATIVE_ACTION_RECEIPT_CAPTURE_SCHEMA = "hytalerl_native_action_receipt_capture_v1"
NATIVE_ACTION_RECEIPT_CAPTURE_VERSION = 1
_BLOCK_VERBS = frozenset(("use", "place_block", "break_block"))


def validate_native_block_receipt_capture(
    receipt: NativeWorldVerbReceipt,
    capture: NativeMutableBlockCells,
    *,
    expected_bridge_sha256: str,
) -> NativeMutableBlockRow:
    """Return the selected current row only when receipt and capture agree.

    The caller must run this gate before passing ``capture`` to
    ``commit_native_mutable_block_cells``. Craft is intentionally excluded:
    it has inventory output but no selected block cell.
    """

    if not isinstance(receipt, NativeWorldVerbReceipt):
        raise TypeError("receipt must be a NativeWorldVerbReceipt")
    if not isinstance(capture, NativeMutableBlockCells):
        raise TypeError("capture must be NativeMutableBlockCells")
    expected_bridge = _sha256(expected_bridge_sha256)
    if (
        receipt.bridge_sha256.upper() != expected_bridge
        or capture.bridge_sha256.upper() != expected_bridge
    ):
        raise ValueError("receipt and mutable-cell capture must share the bridge")

    request = receipt.request
    if request is None or request.verb not in _BLOCK_VERBS:
        raise ValueError("receipt must echo one supported native block verb")
    if request.target_kind != "block":
        raise ValueError("native block receipt must carry a block target")
    if request.world_epoch != receipt.current_world_epoch:
        raise ValueError("native block receipt came from a stale world epoch")
    if not (
        receipt.accepted
        and receipt.started
        and receipt.finished
        and not receipt.active
        and not receipt.failed
        and receipt.acknowledgement_complete
        and receipt.headless_server_actor_certified
        and receipt.opens_action_mask
    ):
        raise ValueError("native block receipt is not complete and certified")
    if request.verb in {"place_block", "break_block"} and not receipt.mutation_applied:
        raise ValueError("native mutating block receipt applied no mutation")

    if len(capture.cells) != 1:
        raise ValueError("one native block receipt requires one selected cell")
    cell = capture.cells[0]
    position = tuple(int(value) for value in cell.position)
    if position != request.target:
        raise ValueError("mutable-cell capture does not match the request target")
    if not cell.available or cell.row is None:
        raise ValueError("selected native block cell is unavailable")

    row = cell.row
    semantic_after = row.block_asset_id if row.block_present else "Empty"
    if (
        semantic_after != receipt.semantic_block_id_after
        or row.runtime_block_id != receipt.runtime_block_id_after
    ):
        raise ValueError("mutable-cell capture disagrees with the native receipt")
    return row


def native_action_receipt_capture_contract() -> dict[str, object]:
    return {
        "schema": NATIVE_ACTION_RECEIPT_CAPTURE_SCHEMA,
        "version": NATIVE_ACTION_RECEIPT_CAPTURE_VERSION,
        "dependencies": {
            "native_world_verb_transport_contract_sha256": (
                native_world_verb_transport_contract_sha256()
            ),
            "native_mutable_block_cells_contract_sha256": (
                native_mutable_block_cells_contract_sha256()
            ),
        },
        "scope": sorted(_BLOCK_VERBS),
        "join": [
            "bridge_sha256",
            "selected_target_i32_xyz",
            "semantic_block_id_after",
            "runtime_block_id_after",
        ],
        "receipt_epoch_guard": (
            "request_world_epoch_equals_receipt_current_world_epoch"
        ),
        "capture_sequence": (
            "same_connection_no_reset_between_receipt_and_capture_required"
        ),
        "required_lifecycle": (
            "accepted_started_finished_acknowledgement_complete_"
            "headless_server_actor_certified_and_mask_open"
        ),
        "mutation": "place_and_break_require_mutation_applied;use_may_be_read_only",
        "next": "commit_native_mutable_block_cells_same_capture",
        "craft": "inventory_acknowledgement_has_no_selected_block_cell",
        "fail_closed": True,
    }


def native_action_receipt_capture_contract_sha256() -> str:
    payload = json.dumps(
        native_action_receipt_capture_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError("expected_bridge_sha256 must be one SHA-256")
    return value.upper()


__all__ = [
    "NATIVE_ACTION_RECEIPT_CAPTURE_SCHEMA",
    "NATIVE_ACTION_RECEIPT_CAPTURE_VERSION",
    "native_action_receipt_capture_contract",
    "native_action_receipt_capture_contract_sha256",
    "validate_native_block_receipt_capture",
]
