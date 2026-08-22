"""Subject loading and seed execution for the frozen outcome report."""
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Callable
from typing import Any

import jax

from hytalegym.jax.evaluation.frozen_outcome_controls import (
    summarize_frozen_outcome_rollout,
)
from hytalegym.jax.training import (
    PPOConfig,
    initialize_training,
    load_policy_checkpoint,
)

from hytalegym.jax.evaluation._frozen_outcome_aggregate import (
    _aggregate_windows,
    _block,
)
from hytalegym.jax.evaluation._frozen_outcome_contracts import (
    _file_sha256,
    _is_sha256,
)


FROZEN_OUTCOME_REPORT_SCHEMA = "hytalerl_arsenal_frozen_outcome_controls_v1"
FROZEN_OUTCOME_REPORT_VERSION = 1
FROZEN_OUTCOME_SUITE_SCHEMA = (
    "hytalerl_arsenal_frozen_outcome_controls_suite_v1"
)
HISTORICAL_ARSENAL_CONTRACT_SHA256 = (
    "0B9F9ACDD44980F1A6C4292209FD6A4E930A2CA46F8B98AE8B2950097CFC21E8"
)
HISTORICAL_POLICY_CONTRACT_SHA256 = (
    "2F3A9A90C5BE4A98EA7F943A65833D7F9F24FC17249F495D3C3C436368BA5D25"
)
HISTORICAL_WORLD_TOKEN_CONTRACT_SHA256 = (
    "4a272c8341eae9c200769acb4aa5daf57ed5bee3660c1f81e753e3101a550761"
)
HISTORICAL_COMBAT_RULESET_SHA256 = (
    "905fc7630c2c7b1fe856a9fd5ef0e7c9005c8d213aa6f5683becf67d88d5e107"
)
HISTORICAL_BRIDGE_SHA256 = (
    "6C6D71B91ECF67D6AFB169D6DC4281AA7123D37A1198CF2205E240D628B0AB26"
)
MEASUREMENT_SEEDS = (1197923981, 291937848)
NUM_ENVIRONMENTS = 128
ROLLOUT_STEPS = 32
ROLLOUT_WINDOWS = 10
MICROTICKS = 1
_OUTCOME_FIELDS = ("success", "death", "simultaneous", "other_terminal")
_SUBJECTS = (
    {
        "training_seed": 1905554470,
        "checkpoint": (
            "artifacts/jax_arsenal_training_rev25_long_6c6d_seed1905554470/"
            "arsenal_seed_1905554470.npz"
        ),
        "checkpoint_sha256": (
            "D11E68D2DECB69713F2F9282195E76ABCF743E4912F3603EE29AFF51ADA20568"
        ),
        "report": (
            "artifacts/jax_arsenal_outcome_probe_rev28_seed1905554470/"
            "report.json"
        ),
        "report_sha256": (
            "D043C4E9A9D891AE16ABDA74A373DE64ED0F38F7C11B91DBB17DAF3E6D2FC570"
        ),
    },
    {
        "training_seed": 570057,
        "checkpoint": (
            "artifacts/jax_arsenal_training_rev25_long_6c6d_seed1905554470/"
            "arsenal_seed_570057.npz"
        ),
        "checkpoint_sha256": (
            "12065FA2149152CD14D2532B89D225572F5665E3F713ECD7B31DB6ED893F65F3"
        ),
        "report": (
            "artifacts/jax_arsenal_outcome_probe_rev28_seed570057/report.json"
        ),
        "report_sha256": (
            "DBC24A1A908A3F1D99554DE06F7C128FD33A7310A8D530E5ABB8AB1CE20F0E55"
        ),
    },
)
_SOURCE_PATHS = {
    "report": "hytalegym/hytalegym/jax/evaluation/frozen_outcome_report.py",
    "collector": (
        "hytalegym/hytalegym/jax/evaluation/frozen_outcome_controls.py"
    ),
    "ppo": "hytalegym/hytalegym/jax/training/ppo.py",
    "arsenal_environment": "hytalegym/hytalegym/jax/training/arsenal.py",
    "policy": "hytalegym/hytalegym/jax/training/policy.py",
}
_SEMANTIC_SOURCE_ROOTS = (
    "hytalegym/hytalegym/jax/combat",
    "hytalegym/hytalegym/jax/training",
    "hytalegym/hytalegym/jax/world",
    "hytalegym/hytalegym/rulesets",
)
_FROZEN_CONTROL_IDENTITIES = {
    "uniform_legal": {
        "schema": "hytalerl_uniform_legal_action_baseline_v1",
        "version": 1,
        "sampling": (
            "uniform independently within each legal categorical head"
        ),
    },
    "always_idle": {
        "schema": "hytalerl_always_idle_action_baseline_v1",
        "version": 1,
        "action": "idle skill, no ability, guard, dodge, or jump",
    },
}


def _load_subject(
    root: Path,
    specification: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_path = root / specification["checkpoint"]
    report_path = root / specification["report"]
    if _file_sha256(checkpoint_path) != specification["checkpoint_sha256"]:
        raise RuntimeError(
            f"frozen checkpoint hash changed: {checkpoint_path}"
        )
    if _file_sha256(report_path) != specification["report_sha256"]:
        raise RuntimeError(f"frozen probe report hash changed: {report_path}")
    policy_params, config, metadata = load_policy_checkpoint(checkpoint_path)
    expected_metadata = {
        "seed": specification["training_seed"],
        "arsenal_contract_sha256": HISTORICAL_ARSENAL_CONTRACT_SHA256,
        "observation_contract_sha256": (
            HISTORICAL_POLICY_CONTRACT_SHA256
        ),
        "action_contract_sha256": HISTORICAL_POLICY_CONTRACT_SHA256,
        "world_geometry_token_contract_sha256": (
            HISTORICAL_WORLD_TOKEN_CONTRACT_SHA256
        ),
        "combat_ruleset_sha256": HISTORICAL_COMBAT_RULESET_SHA256,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise RuntimeError(
                f"frozen checkpoint {field} changed: "
                f"{metadata.get(field)!r} != {expected!r}"
            )
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    if source_report.get("schema") != "hytalerl_arsenal_ppo_run_v2":
        raise RuntimeError("frozen subject report schema changed")
    source = source_report.get("outcome_probe_checkpoint", {})
    if source.get("sha256") != specification["checkpoint_sha256"]:
        raise RuntimeError("frozen report checkpoint hash changed")
    reference_runs = [
        _reference_run(
            seed_report,
            subject_training_seed=specification["training_seed"],
        )
        for seed_report in source_report.get("seeds", ())
    ]
    if tuple(run["measurement_seed"] for run in reference_runs) != (
        MEASUREMENT_SEEDS
    ):
        raise RuntimeError("frozen report measurement seeds changed")
    return {
        **specification,
        "policy_params": policy_params,
        "config": config,
        "metadata": metadata,
        "reference_runs": reference_runs,
    }


def _validate_subject_config(
    subject: dict[str, Any],
    current_config: PPOConfig,
) -> None:
    for field in (
        "observation_size",
        "action_size",
        "action_head_sizes",
        "encoder_size",
        "recurrent_size",
    ):
        expected = getattr(current_config, field)
        actual = getattr(subject["config"], field)
        if actual != expected:
            raise RuntimeError(
                f"frozen checkpoint config.{field}: {actual!r} != {expected!r}"
            )


def _reference_run(
    seed_report: dict[str, Any],
    *,
    subject_training_seed: int,
) -> dict[str, Any]:
    windows = []
    for update in seed_report.get("updates", ()):
        windows.append(
            {
                "index": update.get("index"),
                "episodes_completed": update.get("episodes_completed"),
                "episode_outcomes": dict(update.get("episode_outcomes", {})),
            }
        )
    run = {
        "arm": "trained_policy",
        "measurement_seed": seed_report.get("seed"),
        "subject_training_seed": subject_training_seed,
        "comparison_bridge_sha256": HISTORICAL_BRIDGE_SHA256,
        "execution_bridge_sha256": HISTORICAL_BRIDGE_SHA256,
        "decisions": NUM_ENVIRONMENTS * ROLLOUT_STEPS * ROLLOUT_WINDOWS,
        "windows": windows,
        "episode_outcomes": seed_report.get("episode_outcomes"),
    }
    errors = _run_errors(run)
    if errors:
        raise RuntimeError(
            "frozen subject outcome decomposition failed: "
            + "; ".join(errors)
        )
    return run


def _run_seed(
    config: PPOConfig,
    environment,
    collector,
    *,
    measurement_seed: int,
    arm: str,
    policy_params=None,
    subject_training_seed: int | None = None,
    execution_bridge_sha256: str,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    state = initialize_training(
        jax.random.key(measurement_seed),
        config,
        environment=environment,
    )
    if policy_params is not None:
        state = state._replace(policy_params=policy_params)
    windows = []
    for update_index in range(ROLLOUT_WINDOWS):
        rollout_key = jax.random.key(
            measurement_seed + 2 * update_index + 1
        )
        state, rollout = collector(state, rollout_key)
        _block((state, rollout))
        summary = summarize_frozen_outcome_rollout(rollout)
        windows.append(
            {
                "index": update_index + 1,
                **summary,
            }
        )
        if progress is not None:
            progress(
                {
                    "arm": arm,
                    "subject_training_seed": subject_training_seed,
                    "measurement_seed": measurement_seed,
                    "window": update_index + 1,
                    "episode_outcomes": summary["episode_outcomes"],
                    "episodes_completed": summary["episodes_completed"],
                }
            )
    run = {
        "arm": arm,
        "measurement_seed": measurement_seed,
        "subject_training_seed": subject_training_seed,
        "comparison_bridge_sha256": HISTORICAL_BRIDGE_SHA256,
        "execution_bridge_sha256": execution_bridge_sha256,
        "decisions": NUM_ENVIRONMENTS * ROLLOUT_STEPS * ROLLOUT_WINDOWS,
        "windows": windows,
        "episode_outcomes": _aggregate_windows(windows),
    }
    errors = _run_errors(run)
    if errors:
        raise RuntimeError(
            f"{arm} outcome decomposition failed: " + "; ".join(errors)
        )
    return run


def _find_run(
    runs: list[dict[str, Any]],
    *,
    measurement_seed: int,
) -> dict[str, Any]:
    matches = [
        run for run in runs if run["measurement_seed"] == measurement_seed
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one frozen run for measurement seed {measurement_seed}"
        )
    return matches[0]


def _run_errors(run: dict[str, Any]) -> list[str]:
    if not isinstance(run, dict):
        return ["run is not an object"]
    errors = []
    if run.get("comparison_bridge_sha256") != HISTORICAL_BRIDGE_SHA256:
        errors.append("comparison bridge is not the historical subject bridge")
    if not _is_sha256(run.get("execution_bridge_sha256")):
        errors.append("execution bridge is not a SHA-256")
    if run.get("decisions") != (
        NUM_ENVIRONMENTS * ROLLOUT_STEPS * ROLLOUT_WINDOWS
    ):
        errors.append("decision count changed")
    windows = run.get("windows")
    if not isinstance(windows, list) or len(windows) != ROLLOUT_WINDOWS:
        return ["run must contain ten windows"]
    for index, window in enumerate(windows, start=1):
        if not isinstance(window, dict):
            errors.append(f"window {index} is not an object")
            continue
        if window.get("index") != index:
            errors.append(f"window {index} index changed")
        counts = window.get("episode_outcomes")
        if not isinstance(counts, dict) or set(counts) != set(
            _OUTCOME_FIELDS
        ):
            errors.append(f"window {index} outcome fields are incomplete")
            continue
        if sum(counts.values()) != window.get("episodes_completed"):
            errors.append(f"window {index} outcome counts do not decompose")
    try:
        expected = _aggregate_windows(windows)
    except (KeyError, RuntimeError, TypeError) as error:
        errors.append(f"window aggregation failed: {error}")
    else:
        if run.get("episode_outcomes") != expected:
            errors.append("whole-seed outcome counts do not decompose")
    return errors
