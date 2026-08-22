"""Bind native data, JAX policy, and native-boundary evidence into one receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


PIPELINE_SCHEMA = "hytalerl_native_jax_pipeline_v2"
NATIVE_POLICY_FIDELITY_SCHEMA = "hytalerl_jax_policy_native_fidelity_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as checkpoint:
        value = json.loads(str(checkpoint["metadata_json"]))
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint metadata is not an object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _recorded_path(recorded: object, report: Path) -> Path:
    _require(isinstance(recorded, str) and bool(recorded), "recorded path is empty")
    path = Path(recorded)
    candidates = (path, report.parent / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f"recorded file is unavailable: {recorded}")


def _junit(path: Path) -> dict:
    files = (
        (path,)
        if path.is_file()
        else tuple(sorted(path.glob("TEST-*.xml")))
    )
    _require(bool(files), f"JUnit receipt has no XML files: {path}")
    roots = [ET.parse(file).getroot() for file in files]
    suites = [
        suite
        for root in roots
        for suite in (
            [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        )
    ]
    _require(bool(suites), f"JUnit receipt has no suites: {path}")
    counts = {
        name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    _require(counts["tests"] > 0, f"JUnit receipt has no tests: {path}")
    _require(
        counts["failures"] == counts["errors"] == 0,
        f"JUnit receipt is not green: {path}",
    )
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    return {
        "path": path.resolve().as_posix(),
        "files": len(files),
        "sha256": digest.hexdigest().upper(),
        **counts,
    }


def build_pipeline_receipt(
    *,
    corpus: Path,
    training_report: Path,
    live_policy: Path,
    bridge_jar: Path,
    junit: tuple[Path, ...],
    native_fidelity: Path | None = None,
) -> dict:
    """Validate every handoff and return a content-addressed pipeline receipt."""

    corpus = corpus.resolve()
    training_report = training_report.resolve()
    live_policy = live_policy.resolve()
    bridge_jar = bridge_jar.resolve()
    for path in (corpus, training_report, live_policy, bridge_jar):
        _require(path.is_file(), f"pipeline input is not a file: {path}")
    for path in junit:
        _require(path.exists(), f"JUnit input is unavailable: {path}")

    corpus_manifest_path = corpus.with_suffix(corpus.suffix + ".json")
    live_manifest_path = live_policy.with_suffix(live_policy.suffix + ".json")
    corpus_manifest = _json(corpus_manifest_path)
    training = _json(training_report)
    live = _json(live_manifest_path)
    live_metadata = _metadata(live_policy)
    corpus_sha = _sha256(corpus)
    live_sha = _sha256(live_policy)
    bridge_sha = _sha256(bridge_jar)

    _require(corpus_manifest.get("file_sha256") == corpus_sha, "corpus hash drift")
    trained_corpus = training.get("corpus", {})
    _require(
        trained_corpus.get("source_report_sha256")
        == corpus_manifest.get("source_report_sha256"),
        "training used a different native source report",
    )
    _require(
        trained_corpus.get("source_tensor_sha256")
        == corpus_manifest.get("tensor_sha256"),
        "training used a different native corpus tensor",
    )
    cloning = training.get("behavior_cloning", {})
    _require(int(cloning.get("steps", 0)) > 0, "behavior cloning did not run")
    _require(
        cloning.get("after_heldout", {}).get("negative_log_likelihood", np.inf)
        < cloning.get("before_heldout", {}).get("negative_log_likelihood", -np.inf),
        "held-out behavior-cloning NLL did not improve",
    )

    _require(live.get("schema") == "hytalerl_native_behavior_live_policy_v2", "live policy sidecar schema differs")
    _require(live.get("sha256") == live_sha, "live policy hash drift")
    _require(live.get("roundtrip_exact") is True, "live policy did not round-trip")
    _require(live_metadata.get("live_policy_compatible") is True, "checkpoint is not live compatible")
    promotion = live_metadata.get("promotion_status")
    _require(
        isinstance(promotion, str) and promotion not in {"failed", "rejected"},
        "live policy promotion status rejects publication",
    )
    runtime = live_metadata.get("initialization", {}).get("runtime", {})
    _require(runtime.get("backend") == "gpu", "live policy was not initialized on GPU")
    native_source = live_metadata.get("native_source", {})
    _require(
        native_source.get("corpus_source_report_sha256")
        == corpus_manifest.get("source_report_sha256"),
        "live policy source report differs from corpus",
    )
    _require(
        native_source.get("corpus_tensor_sha256")
        == corpus_manifest.get("tensor_sha256"),
        "live policy source tensor differs from corpus",
    )
    source_checkpoint = _recorded_path(
        native_source.get("checkpoint"), training_report
    )
    source_sha = _sha256(source_checkpoint)
    _require(
        native_source.get("checkpoint_sha256") == source_sha,
        "native source checkpoint hash drift",
    )
    source_metadata = _metadata(source_checkpoint)
    _require(source_metadata.get("stage") == "after_bc", "source is not after-BC")
    _require(
        source_metadata.get("native_replay_tensor_sha256")
        == trained_corpus.get("tensor_sha256"),
        "source checkpoint differs from the projected training corpus",
    )
    transfer = live_metadata.get("transfer_contract", {})
    observation_sha = transfer.get("observation_contract_sha256")
    _require(
        isinstance(observation_sha, str)
        and observation_sha == transfer.get("action_contract_sha256"),
        "live observation/action contract identity differs",
    )
    dynamics_sha = transfer.get("combat_dynamics_contract_sha256")
    _require(
        isinstance(dynamics_sha, str) and len(dynamics_sha) == 64,
        "live policy has no combat-dynamics identity",
    )

    junit_rows = [_junit(path.resolve()) for path in junit]
    _require(bool(junit_rows), "at least one Java test receipt is required")
    native_run = None
    if native_fidelity is not None:
        native_path = native_fidelity.resolve()
        native = _json(native_path)
        _require(
            native.get("schema") == NATIVE_POLICY_FIDELITY_SCHEMA,
            "native policy fidelity schema differs",
        )
        _require(native.get("passed") is True, "native policy fidelity failed")
        _require(native.get("policy_runtime") == "jax", "policy did not run in JAX")
        _require(
            native.get("environment_backend") == "native_java",
            "policy did not use the native environment",
        )
        _require(native.get("policy_sha256") == live_sha, "native run used a different policy")
        _require(native.get("bridge_jar_sha256") == bridge_sha, "native run used a different bridge")
        _require(native.get("observation_contract_sha256") == observation_sha, "native observation contract differs")
        _require(native.get("combat_dynamics_contract_sha256") == dynamics_sha, "native dynamics contract differs")
        _require(native.get("runtime", {}).get("backend") == "gpu", "native policy inference was not on GPU")
        episode = native.get("episode", {})
        _require(
            int(episode.get("steps", 0)) > 0
            and episode.get("terminal_required") is True
            and episode.get("terminated") is True
            and episode.get("truncated") is False
            and episode.get("invalid_observation_rows") == 0
            and episode.get("illegal_action_rows") == 0,
            "native policy episode did not end cleanly",
        )
        trace = native.get("trace", {})
        trace_path = _recorded_path(trace.get("path"), native_path)
        _require(_sha256(trace_path) == trace.get("sha256"), "native trace hash drift")
        _require(trace.get("rows") == episode["steps"], "native trace row count differs")
        native_run = {
            "path": native_path.as_posix(),
            "sha256": _sha256(native_path),
            "content_sha256": native.get("content_sha256"),
            "trace": {
                "path": trace_path.as_posix(),
                "sha256": trace["sha256"],
                "rows": trace["rows"],
            },
        }

    stages = {
        "native_corpus": {"passed": True, "rows": corpus_manifest.get("rows")},
        "jax_behavior_cloning": {
            "passed": True,
            "steps": cloning["steps"],
            "heldout_nll_before": cloning["before_heldout"]["negative_log_likelihood"],
            "heldout_nll_after": cloning["after_heldout"]["negative_log_likelihood"],
        },
        "canonical_live_policy": {"passed": True, "gpu": runtime},
        "bridge_build": {"passed": True},
        "offline_java_tests": {"passed": True, "receipts": junit_rows},
        "jax_policy_native_environment_fidelity": {
            "passed": native_run is not None,
            "status": "passed" if native_run is not None else "pending",
            "receipt": native_run,
        },
    }
    receipt = {
        "schema": PIPELINE_SCHEMA,
        "status": (
            "jax_policy_native_fidelity_passed"
            if native_run is not None
            else "ready_for_jax_policy_native_fidelity"
        ),
        "inputs": {
            "corpus": {"path": corpus.as_posix(), "sha256": corpus_sha},
            "training_report": {
                "path": training_report.as_posix(),
                "sha256": _sha256(training_report),
            },
            "source_checkpoint": {
                "path": source_checkpoint.as_posix(),
                "sha256": source_sha,
            },
            "live_policy": {"path": live_policy.as_posix(), "sha256": live_sha},
            "bridge_jar": {"path": bridge_jar.as_posix(), "sha256": bridge_sha},
        },
        "contracts": {
            "observation_action_sha256": observation_sha,
            "combat_dynamics_sha256": dynamics_sha,
        },
        "stages": stages,
    }
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    receipt["content_sha256"] = hashlib.sha256(payload).hexdigest().upper()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--live-policy", type=Path, required=True)
    parser.add_argument("--bridge-jar", type=Path, required=True)
    parser.add_argument("--junit", type=Path, action="append", required=True)
    parser.add_argument("--native-fidelity", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_pipeline_receipt(
        corpus=args.corpus,
        training_report=args.training_report,
        live_policy=args.live_policy,
        bridge_jar=args.bridge_jar,
        junit=tuple(args.junit),
        native_fidelity=args.native_fidelity,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
