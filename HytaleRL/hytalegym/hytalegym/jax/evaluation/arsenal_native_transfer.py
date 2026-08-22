"""Validation helpers for current-surface Arsenal JAX/native transfer runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any


ARSENAL_NATIVE_TRANSFER_SCHEMA = "hytalerl_arsenal_native_transfer_v1"
ARSENAL_NATIVE_TRANSFER_VERSION = 1


def canonical_sha256(value: Any) -> str:
    """Hash one JSON-compatible value canonically."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def paired_transfer_summary(
    jax_rows: Sequence[Mapping[str, Any]],
    native_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare outcome rows on their explicit ``(profile, seed)`` key."""

    jax_by_key = {
        (str(row["label"]), int(row["seed"])): row for row in jax_rows
    }
    native_by_key = {
        (str(row["profile"]), int(row["seed"])): row
        for row in native_rows
    }
    if len(jax_by_key) != len(jax_rows):
        raise ValueError("JAX transfer rows contain duplicate keys")
    if len(native_by_key) != len(native_rows):
        raise ValueError("native transfer rows contain duplicate keys")
    if set(jax_by_key) != set(native_by_key):
        raise ValueError("JAX/native transfer row keys differ")

    keys = sorted(jax_by_key)
    success_matches = 0
    termination_matches = 0
    length_deltas = []
    rows = []
    for profile, seed in keys:
        jax_row = jax_by_key[(profile, seed)]
        native_row = native_by_key[(profile, seed)]
        jax_success = bool(jax_row["success"])
        native_success = bool(native_row["success"])
        jax_terminated = bool(jax_row["terminated"])
        native_terminated = bool(native_row["terminated"])
        success_matches += int(jax_success == native_success)
        termination_matches += int(jax_terminated == native_terminated)
        length_delta = int(native_row["episode_length"]) - int(
            jax_row["episode_length"]
        )
        length_deltas.append(length_delta)
        rows.append(
            {
                "profile": profile,
                "seed": seed,
                "jax_success": jax_success,
                "native_success": native_success,
                "success_match": jax_success == native_success,
                "jax_terminated": jax_terminated,
                "native_terminated": native_terminated,
                "termination_match": jax_terminated == native_terminated,
                "episode_length_delta_native_minus_jax": length_delta,
            }
        )

    count = len(rows)
    return {
        "episode_count": count,
        "success_match_count": success_matches,
        "success_match_rate": success_matches / count if count else 0.0,
        "termination_match_count": termination_matches,
        "termination_match_rate": (
            termination_matches / count if count else 0.0
        ),
        "jax_success_count": sum(
            bool(row["success"]) for row in jax_rows
        ),
        "native_success_count": sum(
            bool(row["success"]) for row in native_rows
        ),
        "episode_length_delta_native_minus_jax": _number_summary(
            length_deltas
        ),
        "rows": rows,
    }


def arsenal_native_transfer_artifact_errors(
    artifact: Mapping[str, Any],
) -> list[str]:
    """Return all structural/provenance failures in one transfer artifact."""

    errors: list[str] = []
    if artifact.get("schema") != ARSENAL_NATIVE_TRANSFER_SCHEMA:
        errors.append("artifact schema differs")
    if artifact.get("version") != ARSENAL_NATIVE_TRANSFER_VERSION:
        errors.append("artifact version differs")

    suite = artifact.get("suite")
    contract = artifact.get("contract")
    results = artifact.get("results")
    if not isinstance(suite, Mapping):
        errors.append("suite must be a mapping")
    elif artifact.get("suite_sha256") != canonical_sha256(suite):
        errors.append("suite_sha256 does not match suite")
    if not isinstance(contract, Mapping):
        errors.append("contract must be a mapping")
    elif artifact.get("contract_sha256") != canonical_sha256(contract):
        errors.append("contract_sha256 does not match contract")
    if not isinstance(results, Mapping):
        errors.append("results must be a mapping")
    elif artifact.get("results_sha256") != canonical_sha256(results):
        errors.append("results_sha256 does not match results")
    if errors or not isinstance(suite, Mapping) or not isinstance(
        contract, Mapping
    ) or not isinstance(results, Mapping):
        return errors

    start = contract.get("publication_identity_at_start")
    end = contract.get("publication_identity_at_end")
    if not isinstance(start, Mapping) or not isinstance(end, Mapping):
        errors.append("publication identity stamps must be mappings")
    else:
        if start != end:
            errors.append("publication identity moved during execution")
        if contract.get("publication_identity_sha256") != canonical_sha256(
            start
        ):
            errors.append("publication identity digest differs")

    bridge = contract.get("connected_bridge_sha256")
    if not _is_sha256(bridge):
        errors.append("connected bridge identity must be 64-hex")
    checkpoint = contract.get("policy_checkpoint")
    if not isinstance(checkpoint, Mapping) or not _is_sha256(
        checkpoint.get("sha256")
    ):
        errors.append("policy checkpoint identity must be 64-hex")

    shape = contract.get("policy_shape")
    head_sizes = shape.get("action_head_sizes") if isinstance(
        shape, Mapping
    ) else None
    if (
        not isinstance(head_sizes, list)
        or not head_sizes
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 2
            for size in head_sizes
        )
    ):
        errors.append("policy action head sizes are invalid")
    elif isinstance(shape, Mapping) and shape.get("action_logit_size") != sum(
        head_sizes
    ):
        errors.append("policy action logit size differs from head sum")
    if isinstance(shape, Mapping) and (
        isinstance(shape.get("observation_size"), bool)
        or not isinstance(shape.get("observation_size"), int)
        or shape["observation_size"] < 1
    ):
        errors.append("policy observation size is invalid")

    process = contract.get("native_server_process")
    if not isinstance(process, Mapping):
        errors.append("native server process interval is unavailable")
    else:
        start_uptime = process.get("uptime_seconds_at_start")
        end_uptime = process.get("uptime_seconds_at_end")
        if not _nonnegative_finite(start_uptime):
            errors.append("native start uptime is invalid")
        if not _nonnegative_finite(end_uptime):
            errors.append("native end uptime is invalid")
        if (
            _nonnegative_finite(start_uptime)
            and _nonnegative_finite(end_uptime)
            and float(end_uptime) < float(start_uptime)
        ):
            errors.append("native process uptime decreased")

    profiles = suite.get("profiles")
    seeds = suite.get("episode_seeds")
    expected_keys = (
        {(str(profile), int(seed)) for profile in profiles for seed in seeds}
        if isinstance(profiles, list) and isinstance(seeds, list)
        else set()
    )
    jax_rows = _result_rows(results, "jax", errors)
    native_rows = _result_rows(results, "native", errors)
    for name, rows in (("JAX", jax_rows), ("native", native_rows)):
        actual = {
            (
                str(row.get("label", row.get("profile", ""))),
                int(row.get("seed", -1)),
            )
            for row in rows
        }
        if actual != expected_keys or len(rows) != len(expected_keys):
            errors.append(f"{name} episode keys differ from suite")

    native = results.get("native")
    if isinstance(native, Mapping):
        if native.get("connected_bridge_sha256") != bridge:
            errors.append("native result bridge differs from contract")
        if native.get("invalid_observation_count") != 0:
            errors.append("native run contains invalid actor observations")
        if native.get("host_rejected_step_count") != 0:
            errors.append("native run contains host-rejected policy steps")

    differential = results.get("differential")
    if not isinstance(differential, Mapping):
        errors.append("paired differential is unavailable")
    elif differential.get("episode_count") != len(expected_keys):
        errors.append("paired differential episode count differs")

    expected_valid = not errors
    if bool(artifact.get("execution_valid")) != expected_valid:
        errors.append("execution_valid does not match derived validity")
    return errors


def _result_rows(
    results: Mapping[str, Any],
    name: str,
    errors: list[str],
) -> list[Mapping[str, Any]]:
    value = results.get(name)
    rows = value.get("episodes") if isinstance(value, Mapping) else None
    if not isinstance(rows, list) or not all(
        isinstance(row, Mapping) for row in rows
    ):
        errors.append(f"{name} episode rows are unavailable")
        return []
    return rows


def _number_summary(values: Sequence[int | float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
        }
    numbers = tuple(float(value) for value in values)
    return {
        "mean": math.fsum(numbers) / len(numbers),
        "minimum": min(numbers),
        "maximum": max(numbers),
    }


def _nonnegative_finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


__all__ = [
    "ARSENAL_NATIVE_TRANSFER_SCHEMA",
    "ARSENAL_NATIVE_TRANSFER_VERSION",
    "arsenal_native_transfer_artifact_errors",
    "canonical_sha256",
    "paired_transfer_summary",
]
