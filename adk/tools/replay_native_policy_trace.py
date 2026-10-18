"""Replay native policy inputs through the source JAX checkpoint.

This is the first comparison that consumes a real native policy-boundary
artifact. It does not recreate Hytale or compare world transitions: it asks
whether the JAX checkpoint selects the same 12 head factors when given the
exact observations and masks that Java recorded from Hytale.

    python -m adk.tools.replay_native_policy_trace \
        <trace.jsonl> <checkpoint.npz>

The trace's neighboring ``agent.json`` is used when present to verify the
checkpoint SHA-256 that was deployed. Traces produced with a constrained live
action diagnostic are reported but their constrained rows are excluded because
the current trace schema intentionally records the original evidence mask, not
the diagnostic's narrowed mask. Factor agreement remains a useful prefix
diagnostic, but ``passed`` is reserved for a fully processed trace with a
terminal tail row.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "hytale_native_policy_trace_replay_v1"
TRACE_SCHEMA = "hytale_policy_native_policy_trace_v1"
HEAD_SIZES = (12, 17, 2, 2, 37, 9, 9, 5, 10, 2, 2, 17)
SCHEMA_LIMITATIONS = (
    "policy-boundary factor replay does not compare world transitions, rewards, or outcomes",
    "policy_slot has no explicit actor/reset lifecycle ID; recurrent carry resets only on terminal rows",
    "control-constrained rows are excluded because the trace stores the original evidence mask",
    "action payload presence is counted only; the schema does not define applied semantic-action agreement",
    "explicit gap rows do not carry the omitted sample, so tick order across gaps is not recoverable",
)


def compare(trace: Path, checkpoint: Path, *, maximum_rows: int | None = None) -> dict[str, Any]:
    """Replay a native trace and return a machine-readable comparison."""

    trace = Path(trace).resolve()
    checkpoint = Path(checkpoint).resolve()
    rows = _read_jsonl(trace)
    header, transitions = _validated_trace(rows, maximum_rows=maximum_rows)

    _verify_deployed_checkpoint(trace, checkpoint)

    # Delayed imports keep malformed-trace diagnostics cheap and make the tool
    # usable on machines that have the native artifacts but no JAX install.
    import jax.numpy as jnp
    from hytalegym.jax.combat.observation.v3.policy.distribution import (
        greedy_arsenal_action_factors,
    )
    from hytalegym.jax.training.checkpoint import load_policy_checkpoint
    from hytalegym.jax.training.policy import apply_policy

    params, config, _ = load_policy_checkpoint(checkpoint)
    if config.observation_size != int(header["observation_size"]):
        raise ValueError("checkpoint and trace observation widths differ")
    if config.action_size != int(header["action_size"]):
        raise ValueError("checkpoint and trace action widths differ")
    if tuple(config.action_head_sizes) != HEAD_SIZES:
        raise ValueError("checkpoint action heads do not match the trace ABI")
    if config.action_distribution != "arsenal_standard_root_v1":
        raise ValueError(
            "checkpoint does not use the Arsenal standard-root distribution"
        )

    carries: dict[int, Any] = {}
    mismatches: list[dict[str, Any]] = []
    factor_mismatch_count = 0
    constrained_rows = 0
    paired_rows = 0
    terminal_rows = 0
    checked_rows = 0
    for row_index, row in enumerate(transitions, start=1):
        observation = _decode_floats(
            row, "observation_f32_le_b64", int(header["observation_size"])
        )
        action_mask = _decode_mask(
            row, "action_mask_u8_b64", int(header["action_size"])
        )
        slot = row["policy_slot"]
        carry = carries.setdefault(
            slot,
            jnp.zeros((1, config.recurrent_size), dtype=jnp.float32),
        )
        carry, logits, _ = apply_policy(
            params,
            jnp.asarray(observation)[None],
            carry,
            jnp.asarray(action_mask)[None],
        )
        carries[slot] = carry
        factors = np.asarray(
            greedy_arsenal_action_factors(logits)[0], dtype=np.int32
        )
        recorded = _decode_factors(row, row_index)

        next_available = row["next_observation_available"]
        if next_available:
            paired_rows += 1
            next_observation = _decode_floats(
                row,
                "next_observation_f32_le_b64",
                int(header["observation_size"]),
            )
            _decode_mask(
                row, "next_action_mask_u8_b64", int(header["action_size"])
            )
            if next_observation.shape != observation.shape:
                raise ValueError(f"row {row_index} has a malformed next observation")
        else:
            terminal = row["terminal"]
            terminal_rows += int(terminal)

        if not next_available and terminal:
            # A terminal skip ends the Java marker's lifecycle.  The trace
            # schema has no explicit reset event, so only reset a reused slot
            # after the terminal row itself; ordinary discontinuities retain
            # the recurrent carry because the Java actor remains active.
            carries.pop(slot, None)

        if row["control_constrained"]:
            constrained_rows += 1
            continue
        checked_rows += 1
        if not np.array_equal(factors, recorded):
            factor_mismatch_count += 1
            if len(mismatches) < 12:
                mismatches.append({
                    "row": row_index,
                    "policy_slot": slot,
                    "world_tick": int(row["world_tick"]),
                    "jax_factors": factors.tolist(),
                    "java_factors": recorded.tolist(),
                })

    factor_agreement = checked_rows > 0 and factor_mismatch_count == 0
    supplied_trace_rows = len(rows) - 1
    processed_all_supplied_rows = len(transitions) == supplied_trace_rows
    ends_with_valid_terminal = bool(transitions) and _is_valid_terminal_row(
        transitions[-1]
    )
    trace_complete = processed_all_supplied_rows and ends_with_valid_terminal
    fresh_decision_rows = sum(
        int(row["fresh_decision"]) for row in transitions
    )
    action_payload_present_rows = sum(
        int("action" in row) for row in transitions
    )

    result = {
        "schema": SCHEMA,
        "trace_schema": TRACE_SCHEMA,
        "trace": str(trace),
        "checkpoint": str(checkpoint),
        "trace_rows": len(transitions),
        "supplied_trace_rows": supplied_trace_rows,
        "processed_rows": len(transitions),
        "processed_all_supplied_rows": processed_all_supplied_rows,
        "checked_unconstrained_rows": checked_rows,
        "fresh_decision_rows": fresh_decision_rows,
        "constrained_rows": constrained_rows,
        "constrained_rows_excluded": constrained_rows,
        "action_payload_present_rows": action_payload_present_rows,
        "paired_rows": paired_rows,
        "terminal_rows": terminal_rows,
        "factor_mismatches": mismatches,
        "factor_mismatch_count": factor_mismatch_count,
        "factor_agreement": factor_agreement,
        "ends_with_valid_terminal": ends_with_valid_terminal,
        "trace_complete": trace_complete,
        "limitations": list(SCHEMA_LIMITATIONS),
        "passed": factor_agreement and trace_complete,
    }
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as source:
            rows = [
                json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                    parse_float=_parse_json_float,
                )
                for line in source
                if line.strip()
            ]
    except (OSError, UnicodeError, ValueError) as failure:
        raise ValueError(f"could not read native policy trace {path}: {failure}") from failure
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("native policy trace rows must be JSON objects")
    return rows


def _validated_trace(
    rows: list[dict[str, Any]],
    *,
    maximum_rows: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate the part of the writer contract needed for replay.

    The Java sink emits one header followed only by transition rows.  It may
    interleave policy slots, explicitly close a pending row as a gap, and
    reuse a slot after a terminal lifecycle.  Validation therefore follows
    sequence order and checks continuity only when the writer explicitly
    supplied a next sample.
    """

    _validate_maximum_rows(maximum_rows)
    if not rows or rows[0].get("event") != "header":
        raise ValueError("native policy trace has no header row")
    header = rows[0]
    if header.get("schema") != TRACE_SCHEMA:
        raise ValueError(
            f"unsupported trace schema {header.get('schema')!r}; "
            f"expected {TRACE_SCHEMA!r}"
        )
    _validate_header(header)

    all_transitions = rows[1:]
    if not all_transitions:
        raise ValueError("native policy trace contains no transition rows")
    for row_index, row in enumerate(all_transitions, start=1):
        if row.get("event") != "transition":
            raise ValueError(
                f"trace row {row_index} has unsupported event {row.get('event')!r}"
            )

    transitions = (
        all_transitions
        if maximum_rows is None
        else all_transitions[:maximum_rows]
    )
    _validate_transition_rows(transitions, header)
    return header, transitions


def _validate_header(header: dict[str, Any]) -> None:
    if not isinstance(header.get("head_sizes"), list) or tuple(
        header["head_sizes"]
    ) != HEAD_SIZES:
        raise ValueError("native trace head sizes do not match the Arsenal ABI")
    if any(type(size) is not int for size in header["head_sizes"]):
        raise ValueError("native trace head sizes must be integers")
    _positive_int(header, "observation_size", "header")
    action_size = _positive_int(header, "action_size", "header")
    if action_size != sum(HEAD_SIZES):
        raise ValueError(
            f"native trace action_size {action_size} does not match the Arsenal ABI"
        )
    if header.get("float_encoding") != "float32-le":
        raise ValueError("native trace does not use float32-le observations")
    if header.get("mask_encoding") != "uint8":
        raise ValueError("native trace does not use uint8 masks")


def _validate_transition_rows(
    transitions: list[dict[str, Any]],
    header: dict[str, Any],
) -> None:
    observation_width = header["observation_size"]
    action_width = header["action_size"]
    expected_next_by_slot: dict[int, dict[str, Any] | None] = {}

    for row_index, row in enumerate(transitions, start=1):
        if row.get("schema") != TRACE_SCHEMA:
            raise ValueError(f"row {row_index} has the wrong trace schema")
        sequence = _required_int(row, "sequence", row_index)
        if sequence != row_index:
            raise ValueError(
                f"row {row_index} has a duplicate or out-of-order sequence"
            )
        slot = _required_int(row, "policy_slot", row_index)
        if slot < 0:
            raise ValueError(f"row {row_index} has a negative policy_slot")
        world_tick = _required_int(row, "world_tick", row_index)
        delta_seconds = _required_finite_number(row, "delta_seconds", row_index)
        _required_bool(row, "fresh_decision", row_index)
        _required_bool(row, "control_constrained", row_index)
        if not isinstance(row.get("control_route"), str):
            raise ValueError(f"row {row_index} has an invalid control_route")

        observation = _decode_floats(
            row, "observation_f32_le_b64", observation_width
        )
        action_mask = _decode_mask(row, "action_mask_u8_b64", action_width)
        _decode_factors(row, row_index)

        expected = expected_next_by_slot.get(slot)
        if expected is not None:
            if world_tick != expected["world_tick"]:
                raise ValueError(
                    f"row {row_index} does not continue policy slot {slot} "
                    f"at world tick {expected['world_tick']}"
                )
            if not np.array_equal(observation, expected["observation"]):
                raise ValueError(
                    f"row {row_index} does not match the preceding next observation"
                )
            if not np.array_equal(action_mask, expected["action_mask"]):
                raise ValueError(
                    f"row {row_index} does not match the preceding next mask"
                )
            if delta_seconds != expected["delta_seconds"]:
                raise ValueError(
                    f"row {row_index} does not match the preceding next delta_seconds"
                )

        next_available = _required_bool(
            row, "next_observation_available", row_index
        )
        if next_available:
            if "terminal" in row or "gap_reason" in row:
                raise ValueError(
                    f"row {row_index} pairs a next observation with terminal/gap fields"
                )
            next_world_tick = _required_int(row, "next_world_tick", row_index)
            if next_world_tick != world_tick + 1:
                raise ValueError(f"row {row_index} has a non-consecutive next tick")
            next_delta_seconds = _required_finite_number(
                row, "next_delta_seconds", row_index
            )
            next_observation = _decode_floats(
                row,
                "next_observation_f32_le_b64",
                observation_width,
            )
            next_action_mask = _decode_mask(
                row, "next_action_mask_u8_b64", action_width
            )
            expected_next_by_slot[slot] = {
                "world_tick": next_world_tick,
                "delta_seconds": next_delta_seconds,
                "observation": next_observation,
                "action_mask": next_action_mask,
            }
        else:
            terminal = _required_bool(row, "terminal", row_index)
            gap_reason = row.get("gap_reason")
            if not isinstance(gap_reason, str) or not gap_reason:
                raise ValueError(f"row {row_index} has an invalid gap_reason")
            if terminal and not gap_reason.startswith("terminal:"):
                raise ValueError(
                    f"row {row_index} terminal gap_reason lacks terminal prefix"
                )
            expected_next_by_slot[slot] = None


def _is_valid_terminal_row(row: dict[str, Any]) -> bool:
    return (
        row.get("next_observation_available") is False
        and row.get("terminal") is True
        and isinstance(row.get("gap_reason"), str)
        and row["gap_reason"].startswith("terminal:")
    )


def _validate_maximum_rows(maximum_rows: int | None) -> None:
    if maximum_rows is not None and (
        type(maximum_rows) is not int or maximum_rows < 1
    ):
        raise ValueError("maximum_rows must be a positive integer")


def _positive_int(
    row: dict[str, Any], key: str, row_name: str
) -> int:
    value = row.get(key)
    if type(value) is not int or value < 1:
        raise ValueError(f"{row_name} {key} must be a positive integer")
    return value


def _required_int(row: dict[str, Any], key: str, row_index: int) -> int:
    value = row.get(key)
    if type(value) is not int:
        raise ValueError(f"row {row_index} has an invalid {key}")
    return value


def _required_bool(row: dict[str, Any], key: str, row_index: int) -> bool:
    value = row.get(key)
    if type(value) is not bool:
        raise ValueError(f"row {row_index} has an invalid {key}")
    return value


def _required_finite_number(
    row: dict[str, Any], key: str, row_index: int
) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"row {row_index} has an invalid {key}")
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, ValueError):
        finite = False
    if not finite:
        raise ValueError(f"row {row_index} has a nonfinite {key}")
    return float(value)


def _decode_factors(row: dict[str, Any], row_index: int) -> np.ndarray:
    values = row.get("inferred_factors")
    if not isinstance(values, list) or len(values) != len(HEAD_SIZES):
        raise ValueError(f"row {row_index} has malformed inferred_factors width")
    for index, (value, size) in enumerate(zip(values, HEAD_SIZES, strict=True)):
        if type(value) is not int or not 0 <= value < size:
            raise ValueError(
                f"row {row_index} has an invalid inferred_factors[{index}]"
            )
    return np.asarray(values, dtype=np.int32)


def _parse_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"nonfinite JSON number {value!r}")
    return parsed


def _reject_json_constant(value: str) -> float:
    raise ValueError(f"nonfinite JSON number {value!r}")


def _decode_floats(row: dict[str, Any], key: str, width: int) -> np.ndarray:
    try:
        values = np.frombuffer(
            base64.b64decode(row[key], validate=True), dtype="<f4"
        )
    except (KeyError, TypeError, ValueError, binascii.Error) as failure:
        raise ValueError(f"trace row has invalid {key}") from failure
    if values.size != width:
        raise ValueError(f"trace row {key} has {values.size} values, expected {width}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"trace row {key} contains nonfinite floats")
    return values


def _decode_mask(row: dict[str, Any], key: str, width: int) -> np.ndarray:
    try:
        values = np.frombuffer(
            base64.b64decode(row[key], validate=True), dtype=np.uint8
        )
    except (KeyError, TypeError, ValueError, binascii.Error) as failure:
        raise ValueError(f"trace row has invalid {key}") from failure
    if values.size != width or np.any((values != 0) & (values != 1)):
        raise ValueError(f"trace row {key} is not a {width}-bit uint8 mask")
    return values.astype(bool)


def _verify_deployed_checkpoint(trace: Path, checkpoint: Path) -> None:
    manifest_path = trace.parent / "agent.json"
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = str(manifest["checkpoint_sha256"]).upper()
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as failure:
        raise ValueError(f"deployed bundle manifest is unreadable: {failure}") from failure
    actual = _sha256(checkpoint)
    if actual != expected:
        raise ValueError(
            f"checkpoint SHA-256 {actual} does not match deployed bundle {expected}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--maximum-rows", type=int)
    arguments = parser.parse_args()
    try:
        result = compare(
            arguments.trace,
            arguments.checkpoint,
            maximum_rows=arguments.maximum_rows,
        )
    except (OSError, ValueError) as failure:
        print(json.dumps({
            "schema": SCHEMA,
            "passed": False,
            "factor_agreement": False,
            "trace_complete": False,
            "error": str(failure),
            "limitations": list(SCHEMA_LIMITATIONS),
        }))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["passed"]:
        print("PASS - JAX reproduces Java factors on native policy inputs")
        return 0
    failures = []
    if not result["factor_agreement"]:
        failures.append("JAX factor replay differs from Java")
    if not result["trace_complete"]:
        failures.append("trace is incomplete; factor agreement is diagnostic only")
    print("FAIL - " + "; ".join(failures))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
