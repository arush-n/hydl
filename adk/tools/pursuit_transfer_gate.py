"""Evaluate native pursuit runs against a checkpoint's recorded task.

The result deliberately separates observed native behavior from transfer
alignment.  A run may reach the checkpoint's catch radius while still being
ineligible for a transfer certificate when its task, deployment identity, or
runtime evidence is different or unknown.

Usage::

    python -m adk.tools.pursuit_transfer_gate checkpoint.npz run-a run-b
    python -m adk.tools.pursuit_transfer_gate certified.npz run \
        --source-checkpoint original.npz --checkpoint-report report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from adk.deploy.compatibility import profile_content_sha256
from adk.tools.replay_native_policy_trace import (
    _is_valid_terminal_row as _trace_is_valid_terminal_row,
    _read_jsonl as _read_trace_jsonl,
    _validate_header as _validate_trace_header,
    _validated_trace as _validate_trace,
)

from adk.tools import replay_native_policy_trace


SCHEMA = "hytalerl_pursuit_transfer_gate_v1"
_RUNNER_SCHEMA = "hytalerl_isolated_policy_execution_v1"
_METRICS_SCHEMA = "hytale_policy_agent_metrics"
_IGNORED_ACCOMMODATIONS = {"policy_trace"}
_POLICY_TENSOR_NAMES = (
    "encoder_input_kernel",
    "encoder_input_bias",
    "encoder_hidden_kernel",
    "encoder_hidden_bias",
    "gru_input_kernel",
    "gru_recurrent_kernel",
    "gru_bias",
    "actor_kernel",
    "actor_bias",
    "critic_kernel",
    "critic_bias",
)
_TENSOR_DOMAIN = b"HYTALERL_POLICY_TENSOR_CONTENT\0"
_CHECKPOINT_TENSOR_DOMAIN = b"HYTALERL_CHECKPOINT_TENSORS\0"
_CHECKPOINT_JSON_NAMES = frozenset({"config_json", "metadata_json"})
_CERTIFICATION_LINEAGE_SCHEMA = "hytalerl_checkpoint_certification_lineage_v1"
_CERTIFICATION_METADATA_NAMES = frozenset({
    "certification_lineage",
    "certified_from",
    "promotion_status",
    "transfer_contract",
})
_POLICY_TRACE_SCHEMA = "hytale_policy_native_policy_trace_v1"
_RUNTIME_ARTIFACT_CONTAINERS = (
    "runtime_artifacts",
    "deployed_runtime_artifacts",
    "runtime_surface",
    "deployed_artifacts",
)


def evaluate(
    checkpoint: str | Path,
    run_directories: Iterable[str | Path],
    *,
    checkpoint_report: str | Path | None = None,
    source_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate one or more native run directories.

    ``checkpoint`` is hashed locally.  Each run must contain a deployed
    ``agent.json`` whose checkpoint identity agrees with that hash.  When
    ``source_checkpoint`` is supplied, it is the only source path considered:
    the training report binds that source and the certified checkpoint is
    compared to it independently.  Missing or conflicting task-critical
    evidence is retained in ``gaps`` and blocks ``transfer_certificate.issued``.
    """

    checkpoint_path = Path(checkpoint).resolve()
    run_paths = [Path(value).resolve() for value in run_directories]
    if not run_paths:
        raise ValueError("at least one native run directory is required")

    source_path = (
        None if source_checkpoint is None else Path(source_checkpoint).resolve()
    )
    checkpoint_state = _load_checkpoint(
        checkpoint_path,
        checkpoint_report,
        source_checkpoint=source_path,
    )
    run_results = [
        _evaluate_run(path, checkpoint_state["checkpoint_sha256"], checkpoint_state["task"])
        for path in run_paths
    ]

    gaps = list(checkpoint_state["gaps"])
    for result in run_results:
        gaps.extend(result["gaps"])

    alignment = _aggregate_alignment(run_results)
    behavior = _aggregate_behavior(run_results, checkpoint_state["task"])
    behavior_blockers = []
    if behavior["task_success_count"] != len(run_results):
        outcome_status = (
            "mismatch"
            if behavior["task_failure_count"] > 0
            else "unknown"
        )
        behavior_blockers.append(_gap(
            "native_task_outcome",
            outcome_status,
            {
                "task_success_count": len(run_results),
                "task_failure_count": 0,
                "task_outcome_unknown_count": 0,
            },
            {
                "task_success_count": behavior["task_success_count"],
                "task_failure_count": behavior["task_failure_count"],
                "task_outcome_unknown_count": behavior["task_outcome_unknown_count"],
            },
            "all supplied native runs must report explicit successful task outcomes",
        ))
        gaps.extend(behavior_blockers)

    identity_ok = all(result["identity"]["verified"] for result in run_results)
    issued = not gaps and identity_ok
    blockers = list(gaps)
    result = {
        "schema": SCHEMA,
        "checkpoint": checkpoint_state["checkpoint"],
        "task": checkpoint_state["task"],
        "run_count": len(run_results),
        "runs": run_results,
        "task_alignment": alignment,
        "observed_behavior": behavior,
        "gaps": gaps,
        "certificate_blockers": blockers,
        "transfer_certificate": {
            "issued": issued,
            "scope": (
                "all supplied runs have verified deployment identity, matched "
                "recorded task fields, and explicit catch outcomes"
                if issued
                else "not issued; see gaps and certificate_blockers"
            ),
        },
        # Keep the common tool-level spelling available to shell callers.
        "passed": issued,
    }
    return result


def evaluate_transfer(
    checkpoint: str | Path,
    run_directories: Iterable[str | Path],
    *,
    checkpoint_report: str | Path | None = None,
    source_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Named alias for callers that prefer the gate's full purpose."""

    return evaluate(
        checkpoint,
        run_directories,
        checkpoint_report=checkpoint_report,
        source_checkpoint=source_checkpoint,
    )


def _load_checkpoint(
    checkpoint: Path,
    checkpoint_report: str | Path | None,
    *,
    source_checkpoint: Path | None = None,
) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is not a file: {checkpoint}")
    actual_sha = _sha256(checkpoint)
    metadata = _checkpoint_metadata(checkpoint)
    report_path = (
        Path(checkpoint_report).resolve()
        if checkpoint_report is not None
        else _discover_checkpoint_report(checkpoint)
    )
    report: dict[str, Any] = {}
    gaps: list[dict[str, Any]] = []
    if report_path is None:
        gaps.append(_gap(
            "checkpoint_report",
            "unknown",
            None,
            None,
            "no neighboring selected-checkpoint report was found",
        ))
    else:
        report = _read_json_or_gap(report_path, gaps, "checkpoint_report")
    training_report = _training_run_report(
        report_path,
        source_checkpoint if source_checkpoint is not None else checkpoint,
    )

    report_sha = _path_value(report, "checkpoint", "sha256")
    if source_checkpoint is None:
        if report_sha is None:
            gaps.append(_gap(
                "checkpoint_sha256_record",
                "unknown",
                actual_sha,
                None,
                "selected-checkpoint report has no checkpoint SHA-256",
            ))
        elif _canonical_sha(report_sha) != actual_sha:
            gaps.append(_gap(
                "checkpoint_sha256_record",
                "mismatch",
                actual_sha,
                report_sha,
                "checkpoint bytes do not match the selected-checkpoint report",
            ))
    elif report_sha is None:
        gaps.append(_gap(
            "source_checkpoint_sha256_record",
            "unknown",
            None,
            None,
            "training report has no SHA-256 binding for the explicit source",
        ))
    elif source_checkpoint.is_file():
        source_sha = _sha256(source_checkpoint)
        if _canonical_sha(report_sha) != source_sha:
            gaps.append(_gap(
                "source_checkpoint_sha256_record",
                "mismatch",
                source_sha,
                report_sha,
                "training report does not identify the explicit source checkpoint",
            ))

    stage = metadata.get("pursuit_stage")
    if not isinstance(stage, dict):
        stage = _path_value(report, "contracts", "stage")
    if not isinstance(stage, dict):
        gaps.append(_gap(
            "pursuit_stage",
            "unknown",
            None,
            None,
            "checkpoint has no recorded pursuit_stage object",
        ))
        stage = {}

    computed_stage_sha = _canonical_json_sha256(stage) if stage else None
    recorded_stage_sha = _first_present(
        metadata.get("pursuit_stage_sha256"),
        _path_value(report, "contracts", "stage_sha256"),
    )
    if computed_stage_sha is None or recorded_stage_sha is None:
        gaps.append(_gap(
            "pursuit_stage_identity",
            "unknown",
            computed_stage_sha,
            recorded_stage_sha,
            "pursuit stage identity is not fully recorded",
        ))
    elif _canonical_sha(recorded_stage_sha) != computed_stage_sha:
        gaps.append(_gap(
            "pursuit_stage_identity",
            "mismatch",
            computed_stage_sha,
            recorded_stage_sha,
            "recorded pursuit stage SHA-256 differs from its canonical content",
        ))
    report_stage_sha = _path_value(report, "contracts", "stage_sha256")
    if report_stage_sha is not None and _canonical_sha(report_stage_sha) != computed_stage_sha:
        gaps.append(_gap(
            "pursuit_stage_report_identity",
            "mismatch",
            computed_stage_sha,
            report_stage_sha,
            "neighboring report identifies a different pursuit stage",
        ))

    lineage = None
    if source_checkpoint is not None:
        lineage = _verify_checkpoint_lineage(
            checkpoint,
            source_checkpoint,
            report,
        )
        gaps.extend(lineage["gaps"])

    task = _task_from_checkpoint(metadata, report, stage, training_report)
    return {
        "checkpoint_sha256": actual_sha,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": actual_sha,
            "report_path": None if report_path is None else str(report_path),
            "report_sha256": report_sha,
            "pursuit_stage_sha256": computed_stage_sha,
            "recorded_pursuit_stage_sha256": recorded_stage_sha,
            "pursuit_stage": stage,
            "source_checkpoint": lineage,
        },
        "task": task,
        "gaps": gaps,
    }


def _checkpoint_payload(checkpoint: Path) -> dict[str, Any]:
    if checkpoint.suffix.lower() != ".npz":
        raise ValueError(f"checkpoint is not an NPZ file: {checkpoint}")
    try:
        with np.load(checkpoint, allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in archive.files
            }
    except (OSError, ValueError, TypeError) as failure:
        raise ValueError(f"could not read checkpoint arrays: {failure}") from failure

    for name in _CHECKPOINT_JSON_NAMES:
        if name not in arrays:
            raise ValueError(f"checkpoint has no {name} entry")
    config = _json_array_value(arrays["config_json"], "config_json")
    metadata = _json_array_value(arrays["metadata_json"], "metadata_json")
    if not isinstance(config, dict):
        raise ValueError("checkpoint config_json must be a JSON object")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata_json must be a JSON object")
    return {"arrays": arrays, "config": config, "metadata": metadata}


def _json_array_value(array: np.ndarray, name: str) -> Any:
    raw = array.item() if getattr(array, "shape", None) == () else array.tolist()
    if not isinstance(raw, str):
        raise ValueError(f"checkpoint {name} is not a JSON string")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as failure:
        raise ValueError(f"checkpoint {name} is invalid JSON: {failure}") from failure


def _verify_checkpoint_lineage(
    certified: Path,
    source: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    gaps: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "status": "blocked",
        "source_path": str(source),
        "source_sha256": None,
        "report_sha256": _path_value(report, "checkpoint", "sha256"),
        "config_sha256": None,
        "tensors_sha256": None,
        "gaps": gaps,
    }
    if not source.is_file():
        gaps.append(_gap(
            "source_checkpoint",
            "unknown",
            "existing explicit source checkpoint",
            str(source),
            "explicit source checkpoint is missing",
        ))
        return result
    same_file = source == certified
    if not same_file:
        try:
            same_file = source.samefile(certified)
        except OSError:
            same_file = False
    if same_file:
        gaps.append(_gap(
            "source_checkpoint",
            "mismatch",
            "different source and certified checkpoints",
            str(source),
            "source checkpoint must not be the certified checkpoint or a file alias",
        ))
        return result

    try:
        source_payload = _checkpoint_payload(source)
        certified_payload = _checkpoint_payload(certified)
        source_sha = _sha256(source)
    except (OSError, ValueError, TypeError) as failure:
        gaps.append(_gap(
            "checkpoint_lineage",
            "unknown",
            None,
            None,
            f"could not independently load source/certified checkpoints: {failure}",
        ))
        return result

    result["source_sha256"] = source_sha
    source_arrays = source_payload["arrays"]
    certified_arrays = certified_payload["arrays"]
    source_config = source_payload["config"]
    certified_config = certified_payload["config"]
    source_config_sha = _canonical_json_sha256(source_config)
    certified_config_sha = _canonical_json_sha256(certified_config)
    source_tensors_sha = _checkpoint_tensor_content_sha256(source_arrays)
    certified_tensors_sha = _checkpoint_tensor_content_sha256(certified_arrays)
    result["config_sha256"] = source_config_sha
    result["tensors_sha256"] = source_tensors_sha

    source_names = set(source_arrays) - _CHECKPOINT_JSON_NAMES
    certified_names = set(certified_arrays) - _CHECKPOINT_JSON_NAMES
    if source_names != certified_names:
        gaps.append(_gap(
            "certified_policy_tensors",
            "mismatch",
            sorted(source_names),
            sorted(certified_names),
            "source and certified checkpoints have different non-metadata entries",
        ))
    for name in sorted(source_names | certified_names):
        if name not in source_arrays or name not in certified_arrays:
            continue
        if not _array_equal_exact(source_arrays[name], certified_arrays[name]):
            gaps.append(_gap(
                "certified_policy_tensor",
                "mismatch",
                name,
                name,
                "source and certified policy tensor bytes differ",
            ))

    if source_config_sha != certified_config_sha or source_config != certified_config:
        gaps.append(_gap(
            "certified_config",
            "mismatch",
            source_config,
            certified_config,
            "source and certified checkpoint configs differ",
        ))

    source_metadata = source_payload["metadata"]
    certified_metadata = certified_payload["metadata"]
    metadata_keys = set(source_metadata) | set(certified_metadata)
    changed_metadata = sorted(
        name
        for name in metadata_keys
        if name not in _CERTIFICATION_METADATA_NAMES
        and source_metadata.get(name) != certified_metadata.get(name)
    )
    if changed_metadata:
        gaps.append(_gap(
            "certified_metadata",
            "mismatch",
            "all non-certification metadata unchanged",
            changed_metadata,
            "source and certified metadata differ outside the certification whitelist",
        ))

    lineage = certified_metadata.get("certification_lineage")
    if lineage is None:
        lineage_status = "legacy_verified"
    elif not isinstance(lineage, dict):
        lineage_status = "blocked"
        gaps.append(_gap(
            "certification_lineage",
            "unknown",
            "JSON object",
            lineage,
            "certified checkpoint lineage is not an object",
        ))
    else:
        lineage_status = "verified"
        if lineage.get("schema") != _CERTIFICATION_LINEAGE_SCHEMA:
            lineage_status = "blocked"
            gaps.append(_gap(
                "certification_lineage_schema",
                "mismatch",
                _CERTIFICATION_LINEAGE_SCHEMA,
                lineage.get("schema"),
                "certified checkpoint uses an unsupported lineage schema",
            ))
        recorded_source_path = lineage.get("source_checkpoint")
        if not isinstance(recorded_source_path, str) or not recorded_source_path:
            lineage_status = "blocked"
            gaps.append(_gap(
                "certification_lineage_source_checkpoint",
                "unknown",
                "non-empty source label",
                recorded_source_path,
                "certification lineage has no source checkpoint label",
            ))
        elif Path(recorded_source_path).name != source.name:
            lineage_status = "blocked"
            gaps.append(_gap(
                "certification_lineage_source_checkpoint",
                "mismatch",
                source.name,
                recorded_source_path,
                "lineage source label does not name the explicit source checkpoint",
            ))
        for field, actual in (
            ("source_sha256", source_sha),
            ("config_sha256", source_config_sha),
            ("tensors_sha256", source_tensors_sha),
        ):
            recorded = lineage.get(field)
            if _canonical_sha(recorded) != actual:
                lineage_status = "blocked"
                gaps.append(_gap(
                    f"certification_lineage_{field}",
                    "mismatch" if recorded is not None else "unknown",
                    actual,
                    recorded,
                    f"certification lineage {field} does not match independently computed content",
                ))

    if _canonical_sha(result["report_sha256"]) != source_sha:
        gaps.append(_gap(
            "source_checkpoint_sha256_record",
            "mismatch" if result["report_sha256"] is not None else "unknown",
            source_sha,
            result["report_sha256"],
            "training report does not bind the independently hashed source",
        ))

    result["status"] = (
        lineage_status
        if not gaps
        else "blocked"
    )
    return result


def _checkpoint_tensor_content_sha256(arrays: dict[str, np.ndarray]) -> str:
    names = sorted(set(arrays) - _CHECKPOINT_JSON_NAMES)
    digest = hashlib.sha256(_CHECKPOINT_TENSOR_DOMAIN)
    digest.update(struct.pack(">II", 1, len(names)))
    for name in names:
        array = np.asarray(arrays[name])
        if array.dtype.hasobject:
            raise ValueError(f"checkpoint tensor {name!r} has object dtype")
        encoded_name = name.encode("utf-8")
        encoded_dtype = array.dtype.str.encode("ascii")
        content = np.ascontiguousarray(array).tobytes(order="C")
        digest.update(struct.pack(">I", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack(">I", len(encoded_dtype)))
        digest.update(encoded_dtype)
        digest.update(struct.pack(">I", array.ndim))
        for dimension in array.shape:
            digest.update(struct.pack(">Q", int(dimension)))
        digest.update(struct.pack(">Q", len(content)))
        digest.update(content)
    return digest.hexdigest().upper()


def _array_equal_exact(left: np.ndarray, right: np.ndarray) -> bool:
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    return (
        np.ascontiguousarray(left).tobytes(order="C")
        == np.ascontiguousarray(right).tobytes(order="C")
    )


def _checkpoint_metadata(checkpoint: Path) -> dict[str, Any]:
    if checkpoint.suffix.lower() == ".npz":
        try:
            with np.load(checkpoint, allow_pickle=False) as archive:
                if "metadata_json" not in archive.files:
                    raise ValueError("checkpoint has no metadata_json entry")
                raw = archive["metadata_json"]
                raw = raw.item() if getattr(raw, "shape", None) == () else raw.tolist()
        except (OSError, ValueError, TypeError) as failure:
            raise ValueError(f"could not read checkpoint metadata: {failure}") from failure
        if not isinstance(raw, str):
            raise ValueError("checkpoint metadata_json is not a JSON string")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as failure:
            raise ValueError(f"checkpoint metadata_json is invalid: {failure}") from failure
    else:
        value = json.loads(checkpoint.read_text(encoding="utf-8"))
        raw = value.get("metadata_json", value.get("metadata", value))
        value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict):
        raise ValueError("checkpoint metadata must be a JSON object")
    return value


def _discover_checkpoint_report(checkpoint: Path) -> Path | None:
    candidates = [checkpoint.parent / "report.json"]
    if checkpoint.parent.name == "stage":
        candidates.append(checkpoint.parent.parent / "report.json")
    return next((path for path in candidates if path.is_file()), None)


def _training_run_report(
    stage_report_path: Path | None,
    stage_checkpoint: Path,
) -> dict[str, Any]:
    if (
        stage_report_path is None
        or stage_report_path.parent.name != "stage"
        or stage_checkpoint.resolve().parent != stage_report_path.resolve().parent
    ):
        return {}
    run_directory = stage_report_path.parent.parent
    report_path = run_directory / "report.json"
    if not report_path.is_file():
        return {}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if (
        not isinstance(report, dict)
        or report.get("schema") != "hytalerl_console_run_v1"
        or report.get("run_id") != run_directory.name
    ):
        return {}
    return report


def _training_run_bridge_sha256(report: dict[str, Any]) -> str | None:
    candidates = (
        _path_value(report, "contracts", "contracts", "bridge_jar"),
        _path_value(report, "identity", "bridge_jar_sha256"),
        _path_value(report, "identity", "deployed_jar_sha256"),
    )
    recorded = [value for value in candidates if value is not None]
    canonical = [_canonical_sha(value) for value in recorded]
    if not canonical or any(value is None for value in canonical):
        return None
    if len(set(canonical)) != 1:
        return None
    return canonical[0]


def _task_from_checkpoint(
    metadata: dict[str, Any],
    report: dict[str, Any],
    stage: dict[str, Any],
    training_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = report.get("settings") if isinstance(report.get("settings"), dict) else {}
    region = metadata.get("region_reset")
    if not isinstance(region, dict):
        region = _path_value(report, "contracts", "region_reset") or {}
    transfer = metadata.get("transfer_contract")
    if not isinstance(transfer, dict):
        transfer = _path_value(report, "contracts", "source_transfer") or {}
    context = stage.get("opponent_context")
    if not isinstance(context, dict):
        context = {}
    separation = _first_present(
        settings.get("target_separation_range"),
        region.get("target_separation_range"),
        region.get("target_initial_horizontal_separation"),
    )
    context_required = bool(context or region.get("privileged_target_context") is True)
    expected_accommodations: dict[str, bool] = {}
    world_design = _first_present(settings.get("world_design"), region.get("design_id"))
    if isinstance(world_design, str):
        expected_accommodations["flat_world"] = "flat" in world_design.lower()
    if context or region.get("privileged_target_context") is not None:
        expected_accommodations["pursuit_context"] = context_required
    owners = metadata.get("policy_training_context")
    if isinstance(owners, dict):
        owners = owners.get("owners")
    policy_owner = next(
        (
            owner
            for owner in owners or []
            if isinstance(owner, dict) and owner.get("controller") == "policy"
        ),
        None,
    )
    if isinstance(owners, list) and any(
        isinstance(owner, dict) and owner.get("controller") == "policy"
        for owner in owners
    ):
        expected_accommodations["takeover"] = True

    bridge_sha = _first_present(
        transfer.get("bridge_jar_sha256"),
        transfer.get("bridge_sha256"),
        metadata.get("bridge_jar_sha256"),
        report.get("bridge_jar_sha256"),
        _training_run_bridge_sha256(training_report or {}),
    )
    return {
        "objective": settings.get("objective"),
        "agent_profile": _first_present(
            settings.get("loadout"),
            _path_value(policy_owner, "profile"),
        ),
        "opponent_profile": _first_present(
            settings.get("opponent"),
            stage.get("opponent_profile"),
            stage.get("opponent"),
        ),
        "starting_separation": separation,
        "target_controller": settings.get("target_controller"),
        "catch_distance": _number(stage.get("catch_distance")),
        "target_context": {
            "required": context_required,
            "supplement": context.get("supplement"),
            "availability": context.get("availability"),
            "schema": context.get("schema"),
        },
        "horizon": {
            "maximum_ticks": stage.get("maximum_ticks"),
            "horizon": stage.get("horizon"),
            "terminal": stage.get("terminal"),
        },
        "perception": stage.get("perception"),
        "server_version": transfer.get("hytale_version"),
        "bridge_sha256": bridge_sha,
        "accommodations": expected_accommodations,
        "stage_identity": _first_present(
            metadata.get("pursuit_stage_sha256"),
            _path_value(report, "contracts", "stage_sha256"),
        ),
    }


def _evaluate_run(
    run: Path,
    checkpoint_sha: str,
    task: dict[str, Any],
) -> dict[str, Any]:
    gaps: list[dict[str, Any]] = []
    report_path = run / "report.json"
    report: dict[str, Any] = {}
    if report_path.is_file():
        report = _read_json_or_gap(report_path, gaps, "run_report", run.name)
    else:
        gaps.append(_gap(
            "run_report",
            "unknown",
            None,
            None,
            "native run has no report.json",
            run.name,
        ))

    manifest_path = _first_file(
        run / "mods" / "policy-agent" / "agent.json",
        run / "agent.json",
    )
    manifest: dict[str, Any] = {}
    if manifest_path is None:
        gaps.append(_gap(
            "deployed_manifest",
            "unknown",
            checkpoint_sha,
            None,
            "native run has no deployed agent.json",
            run.name,
        ))
    else:
        manifest = _read_json_or_gap(manifest_path, gaps, "deployed_manifest", run.name)

    metrics_path = _first_file(
        run / "mods" / "policy-agent" / "metrics.json",
        run / "metrics.json",
    )
    metrics: dict[str, Any] = {}
    if metrics_path is not None:
        metrics = _read_json_or_gap(metrics_path, gaps, "native_metrics", run.name)
    if not metrics:
        evidence_metrics = _path_value(report, "evidence", "metrics")
        if isinstance(evidence_metrics, dict):
            metrics = evidence_metrics
        elif isinstance(report.get("metrics"), dict):
            metrics = report["metrics"]

    _validate_runner_receipt(report, metrics, gaps, run.name)

    identity = _run_identity(
        run,
        manifest_path,
        manifest,
        checkpoint_sha,
        gaps,
    )
    observed = _observed_run(report, metrics, manifest, run, task)
    _validate_runtime_artifact(
        "policy_jar",
        observed["policy_jar"],
        gaps,
        run.name,
    )
    _validate_runtime_artifact(
        "bridge_jar",
        observed["bridge"],
        gaps,
        run.name,
    )
    _validate_policy_trace(observed["policy_trace"], gaps, run.name)
    alignment = _align_run(run.name, task, observed, identity, gaps)
    return {
        "id": run.name,
        "path": str(run),
        "report_path": str(report_path) if report_path.is_file() else None,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "metrics_path": str(metrics_path) if metrics_path else None,
        "identity": identity,
        "observed_behavior": observed,
        "task_alignment": alignment,
        "gaps": gaps,
    }


def _validate_runner_receipt(
    report: dict[str, Any],
    metrics: dict[str, Any],
    gaps: list[dict[str, Any]],
    run_id: str,
) -> None:
    """Require a successful, recognized receipt for isolated runner reports."""
    isolated = (
        report.get("isolated") is True
        or report.get("schema") == _RUNNER_SCHEMA
    )
    if not isolated:
        return

    checks = (
        (
            "native_runner_schema",
            _RUNNER_SCHEMA,
            report.get("schema"),
            report.get("schema") == _RUNNER_SCHEMA,
            "isolated report does not use the recognized runner schema",
        ),
        (
            "native_runner_passed",
            True,
            report.get("passed"),
            report.get("passed") is True,
            "isolated runner did not report passed=true",
        ),
        (
            "native_runner_exit_code",
            0,
            report.get("exit_code"),
            report.get("exit_code") == 0,
            "isolated runner did not report exit_code=0",
        ),
        (
            "native_runner_error",
            None,
            report.get("error"),
            report.get("error") in (None, ""),
            "isolated runner recorded an execution error",
        ),
        (
            "native_metrics_schema",
            _METRICS_SCHEMA,
            metrics.get("schema") if isinstance(metrics, dict) else None,
            isinstance(metrics, dict) and metrics.get("schema") == _METRICS_SCHEMA,
            "native metrics receipt does not use the recognized schema",
        ),
        (
            "native_metrics_schema_version",
            "integer >= 3",
            metrics.get("schema_version") if isinstance(metrics, dict) else None,
            (
                isinstance(metrics, dict)
                and type(metrics.get("schema_version")) is int
                and metrics["schema_version"] >= 3
            ),
            "native metrics receipt has no supported schema version",
        ),
    )
    for field, expected, observed, valid, reason in checks:
        if valid:
            continue
        status = "unknown" if observed is None else "mismatch"
        gaps.append(_gap(field, status, expected, observed, reason, run_id))
    if "metrics_read_error" in report:
        gaps.append(_gap(
            "native_metrics_receipt",
            "unknown",
            "readable metrics receipt",
            report["metrics_read_error"],
            "isolated runner recorded a metrics read error",
            run_id,
        ))


def _run_identity(
    run: Path,
    manifest_path: Path | None,
    manifest: dict[str, Any],
    checkpoint_sha: str,
    gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    deployed_sha = manifest.get("checkpoint_sha256")
    sha_status = "match" if _canonical_sha(deployed_sha) == checkpoint_sha else (
        "unknown" if deployed_sha is None else "mismatch"
    )
    if sha_status != "match":
        gaps.append(_gap(
            "deployed_checkpoint_sha256",
            sha_status,
            checkpoint_sha,
            deployed_sha,
            "deployed manifest does not identify the selected checkpoint",
            run.name,
        ))

    tensor_identity = manifest.get("policy_tensor_identity")
    tensor_expected = (
        tensor_identity.get("content_sha256")
        if isinstance(tensor_identity, dict)
        else None
    )
    tensor_actual = None
    tensor_status = "not_recorded"
    if tensor_expected is not None and manifest_path is not None:
        try:
            tensor_actual = _tensor_content_sha256(manifest_path.parent)
        except (OSError, ValueError) as failure:
            tensor_status = "unknown"
            gaps.append(_gap(
                "deployed_policy_tensor_identity",
                "unknown",
                tensor_expected,
                None,
                f"could not recompute deployed tensor identity: {failure}",
                run.name,
            ))
        else:
            tensor_status = "match" if _canonical_sha(tensor_expected) == tensor_actual else "mismatch"
            if tensor_status != "match":
                gaps.append(_gap(
                    "deployed_policy_tensor_identity",
                    tensor_status,
                    tensor_expected,
                    tensor_actual,
                    "deployed tensor files do not match agent.json identity",
                    run.name,
                ))
        ordered_names = tensor_identity.get("ordered_names")
        if ordered_names is not None and ordered_names != list(_POLICY_TENSOR_NAMES):
            tensor_status = "mismatch"
            gaps.append(_gap(
                "deployed_policy_tensor_identity_names",
                "mismatch",
                list(_POLICY_TENSOR_NAMES),
                ordered_names,
                "deployed tensor identity uses a different ordered tensor set",
                run.name,
            ))
    elif tensor_expected is not None:
        tensor_status = "unknown"
        gaps.append(_gap(
            "deployed_policy_tensor_identity",
            "unknown",
            tensor_expected,
            None,
            "manifest records tensor identity but its directory is unavailable",
            run.name,
        ))

    compatibility = manifest.get("deployment_compatibility")
    compatibility = compatibility if isinstance(compatibility, dict) else {}
    deployment = compatibility.get("deployment")
    deployment = deployment if isinstance(deployment, dict) else {}
    profile_expected = deployment.get("profile_content_sha256")
    profile_directory = (
        None if manifest_path is None else manifest_path.parent / "profile"
    )
    profile_status = "not_recorded"
    profile_actual = None
    profile_required = (
        profile_expected not in (None, "")
        or manifest.get("live_profile") is True
        or (profile_directory is not None and profile_directory.exists())
    )
    if profile_required:
        if profile_expected in (None, ""):
            profile_status = "unknown"
            profile_reason = (
                "deployed live profile has no recorded profile content SHA-256"
            )
        elif _canonical_sha(profile_expected) is None:
            profile_status = "mismatch"
            profile_reason = "deployed profile content SHA-256 is not canonical"
        elif profile_directory is None or not profile_directory.is_dir():
            profile_status = "unknown"
            profile_reason = "deployed live profile directory is missing"
        else:
            try:
                profile_actual = profile_content_sha256(profile_directory)
            except (OSError, ValueError) as failure:
                profile_status = "unknown"
                profile_reason = f"could not hash deployed live profile: {failure}"
            else:
                profile_status = (
                    "match"
                    if profile_actual == _canonical_sha(profile_expected)
                    else "mismatch"
                )
                profile_reason = (
                    "deployed live profile content matches its manifest"
                    if profile_status == "match"
                    else "deployed live profile content differs from its manifest"
                )
        if profile_status != "match":
            gaps.append(_gap(
                "deployed_profile_content_sha256",
                profile_status,
                profile_expected,
                profile_actual,
                profile_reason,
                run.name,
            ))

    return {
        "manifest": str(manifest_path) if manifest_path else None,
        "selected_checkpoint_sha256": checkpoint_sha,
        "deployed_checkpoint_sha256": deployed_sha,
        "checkpoint_sha256_status": sha_status,
        "tensor_identity_status": tensor_status,
        "tensor_identity_expected": tensor_expected,
        "tensor_identity_actual": tensor_actual,
        "profile_content_status": profile_status,
        "profile_content_expected": profile_expected,
        "profile_content_actual": profile_actual,
        "verified": (
            sha_status == "match"
            and tensor_status in {"match", "not_recorded"}
            and profile_status in {"match", "not_recorded"}
        ),
    }


def _observed_run(
    report: dict[str, Any],
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    run: Path,
    task: dict[str, Any],
) -> dict[str, Any]:
    sources = _native_observation_sources(report, metrics)
    accommodations = report.get("accommodations")
    if not isinstance(accommodations, dict):
        accommodations = _path_value(report, "evidence", "accommodations")
    if not isinstance(accommodations, dict):
        accommodations = {}

    ticks = _number(_source_value(
        sources,
        ("policy_ticks",),
    ), integer=True)
    minimum_distance = _number(_source_value(
        sources,
        ("target_min_planar_distance_clipped",),
        ("target_min_planar_distance",),
    ))
    first_distance = _number(_source_value(
        sources,
        ("target_first_planar_distance_clipped",),
        ("target_first_planar_distance",),
        ("starting_separation",),
    ))
    terminal_reason = _source_value(
        sources,
        ("last_terminal_reason",),
        ("terminal_reason",),
    )
    explicit_success = _explicit_success(metrics, report, terminal_reason)
    catch_distance = _number(task.get("catch_distance"))
    reached = (
        None
        if minimum_distance is None or catch_distance is None
        else minimum_distance <= catch_distance
    )
    admission_threshold = _number(report.get("maximum_target_distance"))
    profile_manifest = manifest.get("live_profile_manifest")
    if not isinstance(profile_manifest, dict):
        profile_manifest = _path_value(
            manifest, "deployment_compatibility", "deployment"
        )
    if not isinstance(profile_manifest, dict):
        profile_manifest = _path_value(report, "live_profile_manifest")
    if not isinstance(profile_manifest, dict):
        profile_manifest = {}
    return {
        "objective": _source_scalar(
            sources,
            ("objective",),
            ("task", "objective"),
        ),
        "agent_profile": profile_manifest.get("profile"),
        "policy_ticks": ticks,
        "min_target_distance": minimum_distance,
        "first_target_distance": first_distance,
        "terminal_reason": terminal_reason,
        "reached_within_catch_distance": reached,
        "catch_distance": catch_distance,
        "task_success_observed": explicit_success,
        "configured_catch_distance": _number(_source_value(
            sources,
            ("configured_catch_distance",),
            ("catch_distance",),
            ("pursuit_catch_distance",),
        )),
        "configured_horizon": _source_scalar(
            sources,
            ("configured_horizon",),
            ("horizon_mode",),
            ("horizon", "horizon"),
            ("horizon",),
        ),
        "configured_terminal": _source_scalar(
            sources,
            ("configured_terminal",),
            ("terminal",),
            ("horizon", "terminal"),
        ),
        "runner_admission": {
            "threshold": admission_threshold,
            "passed": report.get("passed") if isinstance(report.get("passed"), bool) else None,
            "meaning": (
                "native runner admission threshold; it is not substituted for "
                "the checkpoint catch distance"
            ),
        },
        "server_version": _server_version(report, run),
        "perception": _source_scalar(
            sources,
            ("perception",),
            ("perception_mode",),
            ("task", "perception"),
        ),
        "policy_jar": _artifact_identity(report, "policy_jar", run),
        "bridge": _artifact_identity(report, "bridge_jar", run),
        "server": _artifact_identity(report, "server_jar", run),
        "opponent_profile": _path_value(
            profile_manifest, "opponent"
        ),
        "target_controller": _source_scalar(
            sources,
            ("target_controller",),
            ("target", "controller"),
        ),
        "target_context": _context_observation(report, metrics),
        "starting_separation": first_distance,
        "configured_horizon_ticks": _number(_source_value(
            sources,
            ("configured_horizon_ticks",),
            ("maximum_ticks",),
            ("horizon_ticks",),
            ("horizon", "maximum_ticks"),
        ), integer=True),
        "accommodations": accommodations,
        "policy_trace": _policy_trace_observation(report, run),
    }


def _align_run(
    run_id: str,
    task: dict[str, Any],
    observed: dict[str, Any],
    identity: dict[str, Any],
    gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    fields: dict[str, dict[str, Any]] = {}

    _record_alignment(
        fields,
        gaps,
        run_id,
        "objective",
        task.get("objective"),
        observed.get("objective"),
        _compare_scalar(task.get("objective"), observed.get("objective")),
        "native run objective declaration",
    )
    _record_alignment(
        fields,
        gaps,
        run_id,
        "agent_profile",
        task.get("agent_profile"),
        observed.get("agent_profile"),
        _compare_scalar(task.get("agent_profile"), observed.get("agent_profile")),
        "native agent manifest live_profile_manifest.profile",
    )

    # Opponent identity is retained in observed behavior as context, but does
    # not gate whether the policy works on the native server. The native task
    # outcome below is the behavioral success check.
    _record_alignment(
        fields,
        gaps,
        run_id,
        "starting_separation",
        task.get("starting_separation"),
        observed.get("starting_separation"),
        _compare_separation(task.get("starting_separation"), observed.get("starting_separation")),
        "native metrics target_first_planar_distance(_clipped)",
    )
    expected_catch_distance = task.get("catch_distance")
    observed_catch_distance = observed.get("configured_catch_distance")
    _record_alignment(
        fields,
        gaps,
        run_id,
        "catch_distance",
        expected_catch_distance,
        observed_catch_distance,
        _compare_number(expected_catch_distance, observed_catch_distance),
        "declared native catch radius; runner admission is reported separately",
    )
    expected_context = task.get("target_context") or {}
    observed_context = observed.get("target_context") or {}
    context_status = _compare_context(expected_context, observed_context)
    _record_alignment(
        fields,
        gaps,
        run_id,
        "target_context",
        expected_context,
        observed_context,
        context_status,
        "native pursuit-context accommodation and target observation samples",
    )

    expected_horizon = task.get("horizon")
    expected_horizon_ticks = _number(
        _path_value(expected_horizon, "maximum_ticks"), integer=True
    )
    configured_horizon = observed.get("configured_horizon_ticks")
    _record_alignment(
        fields,
        gaps,
        run_id,
        "horizon",
        expected_horizon,
        {
            "maximum_ticks": configured_horizon,
            "horizon": observed.get("configured_horizon"),
            "terminal": observed.get("configured_terminal"),
        },
        _compare_horizon(
            expected_horizon,
            {
                "maximum_ticks": configured_horizon,
                "horizon": observed.get("configured_horizon"),
                "terminal": observed.get("configured_terminal"),
            },
        ),
        "declared native horizon and terminal contract; episode length is reported separately",
    )
    _record_alignment(
        fields,
        gaps,
        run_id,
        "perception",
        task.get("perception"),
        observed.get("perception"),
        _compare_scalar(task.get("perception"), observed.get("perception")),
        "native perception mode declaration",
    )
    ticks = observed.get("policy_ticks")
    tick_status = (
        "unknown"
        if expected_horizon_ticks is None or ticks is None
        else "mismatch" if ticks > expected_horizon_ticks else "match"
    )
    _record_alignment(
        fields,
        gaps,
        run_id,
        "ticks",
        {"maximum": expected_horizon_ticks},
        ticks,
        tick_status,
        "observed native policy ticks, checked only against the recorded maximum",
    )
    server_expected = task.get("server_version")
    server_observed = observed.get("server_version")
    _record_alignment(
        fields,
        gaps,
        run_id,
        "server",
        server_expected,
        server_observed,
        _compare_scalar(server_expected, server_observed),
        "native server version from report or server.log",
    )
    bridge_expected = task.get("bridge_sha256")
    bridge_observed = _path_value(observed, "bridge", "sha256")
    bridge_verified = _path_value(observed, "bridge", "verified")
    bridge_status = (
        "unknown"
        if bridge_expected is None or bridge_observed is None or bridge_verified is not True
        else "match" if _canonical_sha(bridge_expected) == _canonical_sha(bridge_observed) else "mismatch"
    )
    _record_alignment(
        fields,
        gaps,
        run_id,
        "bridge",
        bridge_expected,
        observed.get("bridge"),
        bridge_status,
        "independently hashed bridge artifact and checkpoint-declared bridge identity",
    )
    accommodation_status = _compare_accommodations(
        task.get("accommodations") or {}, observed.get("accommodations") or {}
    )
    _record_alignment(
        fields,
        gaps,
        run_id,
        "accommodations",
        task.get("accommodations") or {},
        observed.get("accommodations") or {},
        accommodation_status,
        "behavior-affecting native accommodations; policy_trace is instrumentation",
    )
    return {
        "status": "match" if all(value["status"] == "match" for value in fields.values()) else "blocked",
        "fields": fields,
        "identity_verified": identity["verified"],
    }


def _record_alignment(
    fields: dict[str, dict[str, Any]],
    gaps: list[dict[str, Any]],
    run_id: str,
    name: str,
    expected: Any,
    observed: Any,
    status: str,
    source: str,
) -> None:
    fields[name] = {
        "status": status,
        "expected": expected,
        "observed": observed,
        "source": source,
    }
    if status != "match":
        gaps.append(_gap(
            name,
            status,
            expected,
            observed,
            f"task-critical native alignment field is {status}",
            run_id,
        ))


def _aggregate_alignment(run_results: list[dict[str, Any]]) -> dict[str, Any]:
    names = tuple(
        name
        for result in run_results
        for name in result["task_alignment"]["fields"]
    )
    unique_names = list(dict.fromkeys(names))
    fields = {}
    for name in unique_names:
        entries = [
            {
                "run": result["id"],
                **result["task_alignment"]["fields"][name],
            }
            for result in run_results
            if name in result["task_alignment"]["fields"]
        ]
        fields[name] = {
            "status": "match" if all(entry["status"] == "match" for entry in entries) else "blocked",
            "runs": entries,
        }
    return {
        "status": "match" if all(value["status"] == "match" for value in fields.values()) else "blocked",
        "fields": fields,
    }


def _aggregate_behavior(
    run_results: list[dict[str, Any]],
    task: dict[str, Any],
) -> dict[str, Any]:
    outcomes = [result["observed_behavior"] for result in run_results]
    terminal_counts: dict[str, int] = {}
    admission_counts: dict[str, int] = {}
    distances = []
    for outcome in outcomes:
        reason = outcome.get("terminal_reason")
        terminal_counts[str(reason) if reason is not None else "<unknown>"] = (
            terminal_counts.get(str(reason) if reason is not None else "<unknown>", 0) + 1
        )
        threshold = outcome["runner_admission"].get("threshold")
        if threshold is not None:
            label = str(threshold)
            admission_counts[label] = admission_counts.get(label, 0) + 1
        if outcome.get("min_target_distance") is not None:
            distances.append(outcome["min_target_distance"])
    return {
        "run_count": len(outcomes),
        "catch_distance": task.get("catch_distance"),
        "reached_within_catch_distance_count": sum(
            outcome.get("reached_within_catch_distance") is True for outcome in outcomes
        ),
        "task_success_count": sum(
            outcome.get("task_success_observed") is True for outcome in outcomes
        ),
        "task_failure_count": sum(
            outcome.get("task_success_observed") is False for outcome in outcomes
        ),
        "task_outcome_unknown_count": sum(
            outcome.get("task_success_observed") is None for outcome in outcomes
        ),
        "terminal_reason_counts": terminal_counts,
        "min_target_distance_values": distances,
        "runner_admission_threshold_counts": admission_counts,
        "claim_scope": (
            "descriptive run counts and per-run observations only; no statistical "
            "claim is made from these runs"
        ),
    }


def _native_observation_sources(
    report: dict[str, Any],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return runtime evidence sources, excluding task declarations."""

    sources: list[dict[str, Any]] = []
    if isinstance(metrics, dict):
        sources.append(metrics)
    evidence = report.get("evidence")
    if not isinstance(evidence, dict):
        return sources
    for name in ("metrics", "native", "runtime", "server", "server_evidence", "observed", "dynamic_behavior"):
        source = evidence.get(name)
        if isinstance(source, dict):
            sources.append(source)
    for name in ("trace", "validated_trace", "trace_evidence"):
        source = evidence.get(name)
        if isinstance(source, dict) and _trace_evidence_validated(source):
            sources.append(source)
    return sources


def _trace_evidence_validated(source: dict[str, Any]) -> bool:
    if source.get("validated") is True:
        return True
    validation = source.get("validation")
    return isinstance(validation, dict) and validation.get("passed") is True


def _source_value(
    sources: Iterable[dict[str, Any]],
    *paths: tuple[Any, ...],
) -> Any:
    for source in sources:
        for path in paths:
            value = _path_value(source, *path)
            if value is not None:
                return value
    return None


def _source_scalar(
    sources: Iterable[dict[str, Any]],
    *paths: tuple[Any, ...],
) -> Any:
    value = _source_value(sources, *paths)
    return None if isinstance(value, (dict, list)) else value


def _source_mapping(
    sources: Iterable[dict[str, Any]],
    *paths: tuple[Any, ...],
) -> dict[str, Any]:
    for source in sources:
        for path in paths:
            value = _path_value(source, *path)
            if isinstance(value, dict):
                return value
    return {}


def _context_observation(report: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    sources = _native_observation_sources(report, metrics)
    observed_context = _source_mapping(
        sources,
        ("target_context",),
        ("pursuit_context",),
        ("target", "context"),
    )
    enabled = _source_value(
        sources,
        ("target_context_enabled",),
        ("pursuit_context_enabled",),
        ("target_context", "enabled"),
        ("pursuit_context", "enabled"),
        ("pursuit_context",),
    )
    samples = _number(_source_value(
        sources,
        ("target_observation_samples",),
    ), integer=True)
    return {
        "enabled": enabled if isinstance(enabled, bool) else None,
        "target_observation_samples": samples,
        "schema": _first_present(
            observed_context.get("schema"),
            _source_scalar(sources, ("target_context_schema",), ("pursuit_context_schema",)),
        ),
        "supplement": _first_present(
            observed_context.get("supplement"),
            _source_scalar(sources, ("target_context_supplement",), ("pursuit_context_supplement",)),
        ),
        "availability": _first_present(
            observed_context.get("availability"),
            _source_scalar(sources, ("target_context_availability",), ("pursuit_context_availability",)),
        ),
    }


def _policy_trace_observation(report: dict[str, Any], run: Path) -> dict[str, Any]:
    evidence = report.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}
    receipt = None
    for name in (
        "policy_trace_receipt",
        "policy_trace",
        "policy_trace_completeness",
        "trace_receipt",
    ):
        candidate = evidence.get(name)
        if isinstance(candidate, dict):
            receipt = candidate
            break

    paths = _trace_paths(report, evidence, receipt, run)
    file_checks = [_inspect_policy_trace(path) for path in paths]
    file_present = bool(file_checks)
    requested = _first_bool(
        _path_value(receipt, "requested"),
        _path_value(evidence, "policy_trace_requested"),
        _path_value(report, "policy_trace_requested"),
        _path_value(report, "accommodations", "policy_trace"),
    )
    explicit_present = _first_bool(
        _path_value(receipt, "present"),
        _path_value(receipt, "trace_present"),
    )
    present = explicit_present if explicit_present is not None else (
        file_present or receipt is not None
    )

    fields: dict[str, bool | None] = {}
    for name, aliases in {
        "header_valid": ("header_valid", "header_validity"),
        "sequence_consistent": ("sequence_consistent", "sequence_valid"),
        "structural_rows_valid": (
            "structural_rows_valid",
            "structural_row_valid",
            "rows_structurally_valid",
            "structural_valid",
        ),
        "final_terminal_row": (
            "final_terminal_row",
            "final_terminal",
            "ends_with_valid_terminal",
        ),
        "complete": ("complete", "trace_complete"),
    }.items():
        receipt_value = _receipt_bool(receipt, aliases)
        file_values = [check.get(name) for check in file_checks]
        if file_values:
            file_value = (
                False
                if any(value is False for value in file_values)
                else True
                if all(value is True for value in file_values)
                else None
            )
            # An identified file is authoritative for its own contents. A
            # receipt cannot turn absent or malformed file bytes into a
            # complete trace.
            fields[name] = file_value
        else:
            fields[name] = receipt_value

    reason = _receipt_string(receipt, ("reason", "trace_reason"))
    if reason is None:
        reasons = [check.get("reason") for check in file_checks if check.get("reason")]
        reason = reasons[0] if reasons else None
    if any(
        receipt is not None
        and isinstance(check.get(name), bool)
        and _receipt_bool(receipt, aliases) is not None
        and check.get(name) != _receipt_bool(receipt, aliases)
        for name, aliases in {
            "header_valid": ("header_valid", "header_validity"),
            "sequence_consistent": ("sequence_consistent", "sequence_valid"),
            "structural_rows_valid": (
                "structural_rows_valid",
                "structural_row_valid",
                "rows_structurally_valid",
                "structural_valid",
            ),
            "final_terminal_row": (
                "final_terminal_row",
                "final_terminal",
                "ends_with_valid_terminal",
            ),
            "complete": ("complete", "trace_complete"),
        }.items()
        for check in file_checks
    ):
        reason = "trace receipt disagrees with the inspected trace file"

    active = requested is True or present is True or receipt is not None or file_present
    return {
        "active": active,
        "requested": requested,
        "present": present,
        "paths": [str(path) for path in paths],
        **fields,
        "reason": reason,
    }


def _validate_policy_trace(
    trace: dict[str, Any],
    gaps: list[dict[str, Any]],
    run_id: str,
) -> None:
    if not trace.get("active"):
        return
    expected = {
        "header_valid": True,
        "sequence_consistent": True,
        "structural_rows_valid": True,
        "final_terminal_row": True,
        "complete": True,
    }
    observed = {
        name: trace.get(name)
        for name in expected
    }
    failures = [
        name for name, wanted in expected.items()
        if trace.get(name) is not wanted
    ]
    if trace.get("present") is False:
        failures.insert(0, "present")
        expected["present"] = True
        observed["present"] = False
    if not failures:
        return
    status = (
        "mismatch"
        if any(name != "present" and trace.get(name) is False for name in failures)
        else "unknown"
    )
    reason = trace.get("reason")
    if not isinstance(reason, str) or not reason:
        reason = "policy trace evidence is missing or incomplete: " + ", ".join(failures)
    gaps.append(_gap(
        "policy_trace",
        status,
        expected,
        observed,
        reason,
        run_id,
    ))


def _trace_paths(
    report: dict[str, Any],
    evidence: dict[str, Any],
    receipt: dict[str, Any] | None,
    run: Path,
) -> list[Path]:
    raw_values: list[Any] = []
    for source in (receipt, evidence, report):
        if not isinstance(source, dict):
            continue
        for name in (
            "path",
            "trace_path",
            "file",
            "trace_file",
            "policy_trace_path",
            "policy_trace_files",
            "files",
        ):
            value = source.get(name)
            if isinstance(value, list):
                raw_values.extend(value)
            elif value is not None:
                raw_values.append(value)
    paths = []
    for value in raw_values:
        path = _resolve_run_path(value, run)
        if path is not None and path not in paths:
            paths.append(path)
    if not paths:
        paths.extend(sorted((run / "mods" / "policy-agent").glob("policy-trace-*.jsonl")))
        paths.extend(sorted(run.glob("policy-trace-*.jsonl")))
    return paths


def _inspect_policy_trace(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "header_valid": None,
        "sequence_consistent": None,
        "structural_rows_valid": None,
        "final_terminal_row": None,
        "complete": None,
        "reason": None,
    }
    try:
        rows = _read_trace_jsonl(path)
    except (OSError, ValueError) as failure:
        result["reason"] = f"could not validate policy trace: {failure}"
        return result

    result["header_valid"] = False
    if rows and rows[0].get("event") == "header" and rows[0].get("schema") == _POLICY_TRACE_SCHEMA:
        try:
            _validate_trace_header(rows[0])
        except (TypeError, ValueError, KeyError) as failure:
            result["reason"] = f"invalid policy trace header: {failure}"
        else:
            result["header_valid"] = True
    if not result["header_valid"]:
        if result["reason"] is None:
            result["reason"] = "policy trace has no valid header"
        return result

    try:
        _, transitions = _validate_trace(rows)
    except (OSError, ValueError) as failure:
        message = str(failure)
        result["structural_rows_valid"] = False
        result["sequence_consistent"] = not any(
            marker in message
            for marker in (
                "duplicate or out-of-order sequence",
                "does not continue policy slot",
                "does not match the preceding next observation",
                "does not match the preceding next mask",
                "does not match the preceding next delta_seconds",
                "non-consecutive next tick",
            )
        )
        result["final_terminal_row"] = False
        result["reason"] = f"invalid policy trace: {message}"
        return result

    result["sequence_consistent"] = True
    result["structural_rows_valid"] = True
    result["transition_rows"] = len(transitions)
    result["final_terminal_row"] = _trace_is_valid_terminal_row(transitions[-1])
    result["complete"] = result["final_terminal_row"]
    result["reason"] = (
        "complete"
        if result["complete"]
        else "policy trace does not end with a valid terminal transition"
    )
    return result


def _receipt_bool(receipt: dict[str, Any] | None, names: Iterable[str]) -> bool | None:
    if not isinstance(receipt, dict):
        return None
    for source in (receipt, receipt.get("validation")):
        if not isinstance(source, dict):
            continue
        for name in names:
            value = source.get(name)
            if isinstance(value, bool):
                return value
    return None


def _receipt_string(receipt: dict[str, Any] | None, names: Iterable[str]) -> str | None:
    if not isinstance(receipt, dict):
        return None
    for source in (receipt, receipt.get("validation")):
        if not isinstance(source, dict):
            continue
        for name in names:
            value = source.get(name)
            if isinstance(value, str) and value:
                return value
    return None


def _first_bool(*values: Any) -> bool | None:
    return next((value for value in values if isinstance(value, bool)), None)


def _compare_context(expected: dict[str, Any], observed: dict[str, Any]) -> str:
    required = expected.get("required")
    enabled = observed.get("enabled")
    if not isinstance(required, bool) or not isinstance(enabled, bool):
        return "unknown"
    if required != enabled:
        return "mismatch"
    samples = observed.get("target_observation_samples")
    if required and samples is None:
        return "unknown"
    if required and (not isinstance(samples, int) or samples <= 0):
        return "mismatch"
    for name in ("schema", "supplement", "availability"):
        expected_value = expected.get(name)
        if expected_value is None:
            continue
        observed_value = observed.get(name)
        if observed_value is None:
            return "unknown"
        if observed_value != expected_value:
            return "mismatch"
    return "match"


def _compare_accommodations(expected: dict[str, Any], observed: dict[str, Any]) -> str:
    for name, value in expected.items():
        if name not in observed or not isinstance(observed[name], bool):
            return "unknown"
        if observed[name] != value:
            return "mismatch"
    for name, value in observed.items():
        if name in _IGNORED_ACCOMMODATIONS:
            continue
        if not isinstance(value, bool):
            return "unknown"
        if name not in expected and value:
            return "mismatch"
    return "match"


def _compare_separation(expected: Any, observed: Any) -> str:
    if expected is None or observed is None:
        return "unknown"
    observed_number = _number(observed)
    if observed_number is None:
        return "unknown"
    if isinstance(expected, list) and len(expected) == 2:
        low, high = (_number(expected[0]), _number(expected[1]))
        if low is None or high is None:
            return "unknown"
        return "match" if low <= observed_number <= high else "mismatch"
    expected_number = _number(expected)
    if expected_number is None:
        return "unknown"
    return "match" if math.isclose(expected_number, observed_number, rel_tol=0.0, abs_tol=1e-6) else "mismatch"


def _compare_scalar(expected: Any, observed: Any) -> str:
    if expected is None or observed is None:
        return "unknown"
    return "match" if expected == observed else "mismatch"


def _compare_number(expected: Any, observed: Any) -> str:
    expected_number = _number(expected)
    observed_number = _number(observed)
    if expected_number is None or observed_number is None:
        return "unknown"
    return (
        "match"
        if math.isclose(expected_number, observed_number, rel_tol=0.0, abs_tol=1e-6)
        else "mismatch"
    )


def _compare_horizon(expected: Any, observed: Any) -> str:
    if not isinstance(expected, dict) or not isinstance(observed, dict):
        return "unknown"
    for name in ("maximum_ticks", "horizon", "terminal"):
        expected_value = expected.get(name)
        if expected_value is None:
            continue
        observed_value = observed.get(name)
        if observed_value is None:
            return "unknown"
        if name == "maximum_ticks":
            status = _compare_number(expected_value, observed_value)
        else:
            status = _compare_scalar(expected_value, observed_value)
        if status != "match":
            return status
    return "match"


def _explicit_success(
    metrics: dict[str, Any],
    report: dict[str, Any],
    terminal_reason: Any,
) -> bool | None:
    for source in _native_observation_sources(report, metrics):
        for key in ("pursuit_success", "task_success", "caught", "distance_catch"):
            value = source.get(key)
            if isinstance(value, bool):
                return value
    if isinstance(terminal_reason, str):
        lowered = terminal_reason.strip().lower()
        if lowered in {"distance_catch", "target_caught", "caught"}:
            return True
        if lowered in {
            "agent_dead",
            "evidence_agent_dead",
            "timeout",
            "time_limit",
            "truncated",
            "uncaught",
            "failed",
            "failure",
        }:
            return False
    return None


def _server_version(report: dict[str, Any], run: Path) -> str | None:
    explicit = _first_present(
        report.get("server_version"),
        _path_value(report, "server", "version"),
        _path_value(report, "evidence", "server_version"),
    )
    if isinstance(explicit, str):
        return explicit
    log_path = run / "server.log"
    if not log_path.is_file():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(r"Version:\s*([0-9]+\.[0-9]+\.[0-9]+)", text)
    return match.group(1) if match else None


def _artifact_identity(
    report: dict[str, Any],
    name: str,
    run: Path,
) -> dict[str, Any]:
    inputs = report.get("inputs") if isinstance(report.get("inputs"), dict) else {}
    entry = inputs.get(name) if isinstance(inputs.get(name), dict) else {}
    source = _artifact_record(entry, run)
    runtime = _artifact_record(_runtime_artifact_entry(report, name), run)
    runtime_evidence = _runtime_artifact_evidence_present(report)
    source_sha = source["actual_sha256"]
    runtime_sha = runtime["actual_sha256"]
    source_recorded_sha = _canonical_sha(source["recorded_sha256"])
    runtime_recorded_sha = _canonical_sha(runtime["recorded_sha256"])
    same_as_source = (
        source_sha is not None
        and runtime_sha is not None
        and source_sha == runtime_sha
    )
    historical_source_verified = None
    source_current_status = None
    if name == "bridge_jar":
        if source["path"] is None or source_recorded_sha is None:
            source_current_status = "unknown"
        elif source_sha is None:
            source_current_status = "unavailable"
        elif source_sha == source_recorded_sha:
            source_current_status = "verified"
        else:
            source_current_status = "drifted"
        if runtime_evidence:
            historical_source_verified = (
                source["path"] is not None
                and source_recorded_sha is not None
                and runtime["verified"]
                and runtime_recorded_sha == source_recorded_sha
                and runtime_sha == source_recorded_sha
            )
    effective = runtime if runtime_evidence else source
    verified = (
        historical_source_verified
        if name == "bridge_jar" and runtime_evidence
        else (
            source["verified"] and runtime["verified"] and same_as_source
            if runtime_evidence
            else source["verified"]
        )
    )
    identity = {
        "source": source,
        "runtime": runtime,
        "runtime_evidence": runtime_evidence,
        "path": effective["path"],
        "recorded_sha256": effective["recorded_sha256"],
        "sha256": effective["actual_sha256"],
        "actual_sha256": effective["actual_sha256"],
        "source_verified": source["verified"],
        "runtime_verified": runtime["verified"],
        "same_as_source": same_as_source,
        "verified": verified,
    }
    if name == "bridge_jar":
        identity["source_current_status"] = source_current_status
        identity["historical_source_verified"] = historical_source_verified
    return identity


def _artifact_record(entry: Any, run: Path) -> dict[str, Any]:
    if not isinstance(entry, dict):
        entry = {}
    raw_path = _first_present(
        entry.get("path"),
        entry.get("runtime_path"),
        entry.get("source_path"),
    )
    path = _resolve_run_path(raw_path, run)
    recorded = _first_present(
        entry.get("sha256"),
        entry.get("runtime_sha256"),
        entry.get("source_sha256"),
    )
    actual = None
    if path is not None and path.is_file():
        try:
            actual = _sha256(path)
        except OSError:
            actual = None
    return {
        "path": str(path) if path is not None else None,
        "recorded_sha256": recorded,
        "sha256": actual,
        "actual_sha256": actual,
        "verified": (
            actual is not None
            and _canonical_sha(recorded) == actual
        ),
    }


def _runtime_artifact_entry(report: dict[str, Any], name: str) -> dict[str, Any]:
    evidence = report.get("evidence")
    if not isinstance(evidence, dict):
        return {}
    aliases = {
        "policy_jar": ("policy_jar", "policy", "agent_jar"),
        "bridge_jar": ("bridge_jar", "bridge"),
        "server_jar": ("server_jar", "server"),
    }.get(name, (name,))

    def select(container: Any) -> dict[str, Any]:
        if isinstance(container, dict):
            nested = container.get("artifacts")
            if isinstance(nested, (dict, list)):
                container = nested
            if isinstance(container, dict):
                for alias in aliases:
                    entry = container.get(alias)
                    if isinstance(entry, dict):
                        runtime = entry.get("runtime")
                        return runtime if isinstance(runtime, dict) else entry
            if isinstance(container, list):
                for entry in container:
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("name") in aliases or entry.get("artifact") in aliases:
                        runtime = entry.get("runtime")
                        return runtime if isinstance(runtime, dict) else entry
        return {}

    for container_name in _RUNTIME_ARTIFACT_CONTAINERS:
        entry = select(evidence.get(container_name))
        if entry:
            return entry
    for alias in aliases:
        entry = evidence.get(f"runtime_{alias}")
        if isinstance(entry, dict):
            return entry
    return {}


def _runtime_artifact_evidence_present(report: dict[str, Any]) -> bool:
    evidence = report.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}
    if any(name in evidence for name in _RUNTIME_ARTIFACT_CONTAINERS):
        return True
    if report.get("isolated") is True:
        return True
    for name in (
        "runtime_artifacts_requested",
        "runtime_surface_requested",
        "deployed_runtime_artifacts_requested",
    ):
        if evidence.get(name) is True or report.get(name) is True:
            return True
    return any(
        isinstance(evidence.get(f"runtime_{name}"), dict)
        for name in ("policy_jar", "bridge_jar", "server_jar", "policy", "bridge")
    )


def _validate_runtime_artifact(
    name: str,
    artifact: dict[str, Any],
    gaps: list[dict[str, Any]],
    run_id: str,
) -> None:
    source = artifact.get("source") if isinstance(artifact, dict) else None
    runtime = artifact.get("runtime") if isinstance(artifact, dict) else None
    if not isinstance(source, dict):
        source = {}
    if not isinstance(runtime, dict):
        runtime = {}

    bridge_runtime_evidence = (
        name == "bridge_jar" and artifact.get("runtime_evidence") is True
    )
    source_sha = _canonical_sha(source.get("recorded_sha256"))
    if bridge_runtime_evidence:
        if source.get("path") is None or source_sha is None:
            gaps.append(_gap(
                f"{name}_source",
                "unknown",
                {"path": "recorded source input", "sha256": "valid SHA-256"},
                source,
                f"{name} historical source identity is missing its path or SHA-256",
                run_id,
            ))
        records = []
    else:
        records = [("source", source)]
    if artifact.get("runtime_evidence") is True:
        records.append(("runtime", runtime))
    for role, record in records:
        if record.get("path") is None or record.get("recorded_sha256") is None:
            status = "unknown"
            reason = f"{name} {role} artifact record is missing its path or SHA-256"
        elif record.get("actual_sha256") is None:
            status = "unknown"
            reason = f"{name} {role} artifact could not be independently hashed"
        elif not record.get("verified"):
            status = "mismatch"
            reason = f"{name} {role} artifact SHA-256 does not match its file"
        else:
            continue
        gaps.append(_gap(
            f"{name}_{role}",
            status,
            {"path": "existing artifact", "sha256": "matches file bytes"},
            record,
            reason,
            run_id,
        ))

    if artifact.get("runtime_evidence") is True:
        if bridge_runtime_evidence:
            lineage_expected = source_sha
            lineage_observed = runtime.get("actual_sha256")
            lineage_comparable = lineage_expected is not None and lineage_observed is not None
        else:
            lineage_expected = source.get("actual_sha256")
            lineage_observed = runtime.get("actual_sha256")
            lineage_comparable = source.get("verified") and runtime.get("verified")
        if lineage_comparable and lineage_expected != lineage_observed:
            gaps.append(_gap(
                f"{name}_lineage",
                "mismatch",
                lineage_expected,
                lineage_observed,
                f"deployed {name} bytes differ from the recorded source input",
                run_id,
            ))


def _resolve_run_path(raw_path: Any, run: Path) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else run / path


def _read_json_or_gap(
    path: Path,
    gaps: list[dict[str, Any]],
    field: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as failure:
        gaps.append(_gap(field, "unknown", None, None, f"unreadable JSON: {failure}", run_id))
        return {}
    if not isinstance(value, dict):
        gaps.append(_gap(field, "unknown", None, value, "JSON evidence is not an object", run_id))
        return {}
    return value


def _first_file(*paths: Path) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def _path_value(value: Any, *keys: Any) -> Any:
    current = value
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        elif isinstance(current, list) and isinstance(key, int) and 0 <= key < len(current):
            current = current[key]
        else:
            return None
    return current


def _first_present(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _number(value: Any, *, integer: bool = False) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    if integer:
        return int(value) if value.is_integer() else None
    return value


def _canonical_sha(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) != 64:
        return None
    upper = value.upper()
    return upper if all(character in "0123456789ABCDEF" for character in upper) else None


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _tensor_content_sha256(directory: Path) -> str:
    digest = hashlib.sha256(_TENSOR_DOMAIN)
    digest.update(struct.pack(">II", 1, len(_POLICY_TENSOR_NAMES)))
    for name in _POLICY_TENSOR_NAMES:
        path = directory / f"{name}.bin"
        if not path.is_file():
            raise FileNotFoundError(path)
        encoded = name.encode("utf-8")
        digest.update(struct.pack(">I", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack(">Q", path.stat().st_size))
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest().upper()


def _gap(
    field: str,
    status: str,
    expected: Any,
    observed: Any,
    reason: str,
    run: str | None = None,
) -> dict[str, Any]:
    value = {
        "code": f"{field}_{status}",
        "field": field,
        "status": status,
        "expected": expected,
        "observed": observed,
        "reason": reason,
    }
    if run is not None:
        value["run"] = run
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("run_directories", nargs="+", type=Path)
    parser.add_argument("--checkpoint-report", type=Path)
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        help="explicit original checkpoint for one-hop certification lineage",
    )
    args = parser.parse_args(argv)
    try:
        result = evaluate(
            args.checkpoint,
            args.run_directories,
            checkpoint_report=args.checkpoint_report,
            source_checkpoint=args.source_checkpoint,
        )
    except (OSError, ValueError, TypeError) as failure:
        result = {
            "schema": SCHEMA,
            "passed": False,
            "transfer_certificate": {"issued": False},
            "error": f"{type(failure).__name__}: {failure}",
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("transfer_certificate", {}).get("issued") else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "evaluate", "evaluate_transfer", "main"]
