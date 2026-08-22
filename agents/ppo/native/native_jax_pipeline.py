"""Run the native-data to live-JAX-policy workflow as resumable stages."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np

from agents.ppo.native.native_behavior_il import REPORT_SCHEMA, Settings
from agents.ppo.native.pipeline import build_pipeline_receipt
from arena.jax_contract import HEAD_SPANS


RUN_CONFIG_SCHEMA = "hytalerl_native_jax_pipeline_config_v1"
RUN_SCHEMA = "hytalerl_native_jax_pipeline_run_v3"
_LIVE_DEFAULTS = {
    "task": "plains_duel",
    "actor_role": "Trork_Brawler",
    "opponent_role": "Kweebec_Razorleaf",
    "seed": 570401,
    "batch": 32,
    "ppo_updates": 8,
    "rollout_steps": 32,
    "update_epochs": 2,
    "num_minibatches": 2,
    "learning_rate": 3.0e-4,
    "entropy_coefficient": 1.0e-3,
    "encoder_size": 64,
    "recurrent_size": 64,
}


def run_pipeline(config_path: Path) -> dict[str, Any]:
    """Run or resume exact content-bound stages from one JSON config."""

    started = perf_counter()
    config_path = config_path.resolve()
    config = _json(config_path)
    if config.get("schema") != RUN_CONFIG_SCHEMA:
        raise ValueError("native JAX pipeline config schema differs")
    corpus = _path(config["corpus"], config_path.parent)
    run_dir = _path(config["run_dir"], config_path.parent)
    cache_root = _path(
        config.get("cache_root", "~/.cache/hytalerl/native-replay"),
        config_path.parent,
    )
    view = config.get("observation_view", "arsenal_visual_conditioned")
    training_settings = Settings(
        **config.get("training", {}),
        observation_view=view,
        replay_cache=str(cache_root),
    )
    live_settings = _settings(config.get("live", {}), _LIVE_DEFAULTS, "live")
    config_sha = _json_sha256(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    previous_result = (
        _json(run_dir / "run.json") if (run_dir / "run.json").is_file() else None
    )
    logs = run_dir / "logs"
    logs.mkdir(exist_ok=True)
    _bind_state(run_dir / "run-state.json", config_sha, config_path, config)
    stages = []

    training_dir = run_dir / "training"
    training_report = training_dir / "report.json"
    if training_report.is_file():
        training_row = {"stage": "jax_training", "skipped": True, "wall_seconds": 0.0}
    else:
        if training_dir.exists():
            raise ValueError("incomplete training directory blocks safe resume")
        command = [
            sys.executable,
            "-m",
            "agents.ppo.native.native_behavior_il",
            "--corpus",
            str(corpus),
            "--output",
            str(training_dir),
            *_flags(asdict(training_settings)),
        ]
        training_row = _run("jax_training", command, logs, _environment("cuda"))
        training_row.pop("stdout_last_line", None)
    training = _validate_training(training_report, training_settings, corpus)
    compilation = training["compilation"]
    stages.append(
        {
            "stage": "compile_replay",
            "skipped": training_row["skipped"] and compilation["cache_hit"],
            "wall_seconds": sum(
                compilation[name]
                for name in ("load_seconds", "sequence_seconds", "packing_seconds")
            ),
            "receipt": compilation,
        }
    )
    stages.append({**training_row, "receipt": training["receipt"]})

    live_policy = run_dir / "live_policy.npz"
    live_sidecar = live_policy.with_suffix(live_policy.suffix + ".json")
    if live_policy.is_file() and live_sidecar.is_file():
        live_row = {"stage": "publish_live_policy", "skipped": True, "wall_seconds": 0.0}
    else:
        if live_policy.exists() or live_sidecar.exists():
            raise ValueError("partial live policy blocks safe resume")
        command = [
            sys.executable,
            "-m",
            "agents.ppo.native.native_live_policy",
            "--native-checkpoint",
            str(training["after_bc"]),
            "--corpus",
            str(corpus),
            "--output",
            str(live_policy),
            "--replay-cache",
            str(cache_root),
            "--observation-view",
            view,
            *_flags(live_settings),
        ]
        live_row = _run("publish_live_policy", command, logs, _environment("cuda"))
        live_row.pop("stdout_last_line", None)
    live = _validate_live(live_policy, live_sidecar)
    stages.append({**live_row, "receipt": live})

    verification = config.get("verification")
    pipeline_receipt = None
    if verification is not None:
        bridge = _path(verification["bridge_jar"], config_path.parent)
        junit = tuple(
            _path(path, config_path.parent) for path in verification.get("junit", ())
        )
        native = verification.get("native_fidelity")
        native_path = _path(native, config_path.parent) if native else None
        started = perf_counter()
        pipeline_receipt = build_pipeline_receipt(
            corpus=corpus,
            training_report=training_report,
            live_policy=live_policy,
            bridge_jar=bridge,
            junit=junit,
            native_fidelity=native_path,
        )
        receipt_path = run_dir / "pipeline-receipt.json"
        receipt_path.write_text(
            json.dumps(pipeline_receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stages.append(
            {
                "stage": "native_fidelity_receipts",
                "skipped": False,
                "wall_seconds": perf_counter() - started,
                "receipt": {
                    "path": receipt_path.as_posix(),
                    "sha256": _file_sha256(receipt_path),
                    "status": pipeline_receipt["status"],
                },
            }
        )

    outputs = {
        "training_report": _receipt(training_report),
        "after_bc": _receipt(training["after_bc"]),
        "after_ppo": _receipt(training["after_ppo"]),
        "live_policy": _receipt(live_policy),
        "live_policy_sidecar": _receipt(live_sidecar),
    }
    if pipeline_receipt is not None:
        outputs["pipeline_receipt"] = _receipt(run_dir / "pipeline-receipt.json")
    wall_seconds = perf_counter() - started
    latest_execution = {"wall_seconds": wall_seconds, "stages": stages}
    first_execution = _first_execution(previous_result, config_sha, latest_execution)
    result = {
        "schema": RUN_SCHEMA,
        "status": (
            pipeline_receipt["status"]
            if pipeline_receipt is not None
            else "ready_for_native_fidelity_receipts"
        ),
        "config_sha256": config_sha,
        "execution": "JAX_CUDA_training_and_policy; Java_only_optional_fidelity_boundary",
        "policy_flow": {
            "publication_source": "after_bc",
            "publication_stage": "post_native_transplant_pre_worldgen_training",
            "after_ppo_scope": "recorded_context_diagnostic_not_live_publication",
        },
        "stages": stages,
        "outputs": outputs,
        "wall_seconds": wall_seconds,
        "first_execution": first_execution,
        "latest_execution": latest_execution,
    }
    result["content_sha256"] = _run_content_sha256(result)
    output = run_dir / "run.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _run(
    stage: str, command: list[str], logs: Path, environment: dict[str, str]
) -> dict[str, Any]:
    started = perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    log = logs / f"{stage}.log"
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"{stage} failed; see {log}")
    lines = completed.stdout.strip().splitlines()
    return {
        "stage": stage,
        "skipped": False,
        "wall_seconds": perf_counter() - started,
        "command": command,
        "log": log.as_posix(),
        "stdout_last_line": lines[-1] if lines else "{}",
    }


def _validate_training(report_path: Path, settings: Settings, corpus: Path) -> dict[str, Any]:
    report = _json(report_path)
    if report.get("schema") != REPORT_SCHEMA or report.get("settings") != asdict(settings):
        raise ValueError("training report does not match this pipeline configuration")
    if Path(report["corpus"]["path"]).resolve() != corpus.resolve():
        raise ValueError("training report used a different corpus")
    if report.get("backend") != "gpu":
        raise ValueError("native behavior training did not run on GPU")
    after_bc = _recorded(report["behavior_cloning"]["checkpoint"], report_path)
    after_ppo = _recorded(report["checkpoint"], report_path)
    with np.load(after_bc, allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint["metadata_json"]))
    coverage = report["corpus"]["action_head_coverage"]
    expected_heads = [
        name for name in HEAD_SPANS if coverage[name]["labelled_rows"] > 0
    ]
    if (
        metadata.get("native_replay_supervised_heads") != expected_heads
        or metadata.get("native_replay_tensor_sha256")
        != report["corpus"]["tensor_sha256"]
    ):
        raise ValueError("after-BC checkpoint metadata differs from the action ABI")
    return {
        "after_bc": after_bc,
        "after_ppo": after_ppo,
        "receipt": {
            "path": report_path.as_posix(),
            "sha256": _file_sha256(report_path),
            "schema": report["schema"],
            "cache_hit": report["data_compilation"]["cache_hit"],
            "wall_seconds": report["wall_seconds"],
        },
        "compilation": report["data_compilation"],
    }


def _validate_live(checkpoint: Path, sidecar: Path) -> dict[str, Any]:
    report = _json(sidecar)
    if report.get("schema") != "hytalerl_native_behavior_live_policy_v2":
        raise ValueError("live policy schema differs")
    if report.get("sha256") != _file_sha256(checkpoint):
        raise ValueError("live policy hash drift")
    runtime = report.get("metadata", {}).get("initialization", {}).get("runtime", {})
    if runtime.get("backend") != "gpu":
        raise ValueError("live policy was not initialized on GPU")
    return {"path": checkpoint.as_posix(), "sha256": report["sha256"], "runtime": runtime}


def _recorded(row: dict[str, Any], report: Path) -> Path:
    path = Path(row["path"])
    candidates = (path, report.parent / path.name)
    for candidate in candidates:
        if candidate.is_file() and _file_sha256(candidate) == row["sha256"]:
            return candidate.resolve()
    raise ValueError("recorded checkpoint is missing or changed")


def _settings(value: object, defaults: dict[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not set(value) <= set(defaults):
        raise ValueError(f"{label} settings contain unsupported keys")
    return {**defaults, **value}


def _flags(settings: dict[str, Any]) -> list[str]:
    result = []
    for key, value in settings.items():
        result.extend((f"--{key.replace('_', '-')}", str(value)))
    return result


def _environment(platform: str) -> dict[str, str]:
    value = dict(os.environ)
    value["JAX_PLATFORMS"] = platform
    if platform == "cuda":
        profile = value.setdefault("HYTALERL_JAX_RUNTIME_PROFILE", "throughput")
        if profile == "throughput":
            value["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
            value.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    return value


def _bind_state(path: Path, digest: str, source: Path, config: dict[str, Any]) -> None:
    state = {"config_sha256": digest, "source": source.as_posix(), "config": config}
    if path.is_file() and _json(path) != state:
        raise ValueError("run directory is already bound to a different config")
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _path(value: object, base: Path) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("pipeline paths must be strings")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest().upper()


def _run_content_sha256(result: dict[str, Any]) -> str:
    return _json_sha256(
        {
            key: result[key]
            for key in (
                "schema",
                "status",
                "config_sha256",
                "execution",
                "policy_flow",
                "outputs",
            )
        }
    )


def _first_execution(
    previous: dict[str, Any] | None,
    config_sha256: str,
    latest: dict[str, Any],
) -> dict[str, Any]:
    if (
        previous is not None
        and previous.get("schema") == RUN_SCHEMA
        and previous.get("config_sha256") == config_sha256
    ):
        first = previous.get("first_execution")
        if not isinstance(first, dict):
            raise ValueError("prior pipeline run lacks its first execution receipt")
        return first
    return latest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _receipt(path: Path) -> dict[str, str]:
    return {"path": path.as_posix(), "sha256": _file_sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_pipeline(arguments.config)
    print(json.dumps({"status": result["status"], "sha256": result["content_sha256"]}))


if __name__ == "__main__":
    main()
