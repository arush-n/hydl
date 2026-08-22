"""Historical-contract control report for the frozen Arsenal outcome probe."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from collections.abc import Callable
from typing import Any

import jax
import numpy as np

from hytalegym.jax.combat import (
    PROFILE_NAMES,
    arsenal_runtime_config,
    default_combat_params,
    hytale_0_5_7_loadouts,
)
from hytalegym.jax.evaluation.frozen_outcome_controls import (
    make_frozen_outcome_collector,
)
from hytalegym.jax.training import (
    PPOConfig,
    REFERENCE_ARSENAL_POLICY_ACTION_HEAD_SIZES,
    make_arsenal_ppo_environment,
    open_flat_arsenal_world_capabilities,
)
from hytalegym.jax.training.policy import initialize_policy

# Re-exported to preserve this module's attribute surface.
from hytalegym.jax.evaluation._frozen_outcome_aggregate import (  # noqa: F401
    _aggregate_runs,
    _aggregate_windows,
    _arm_result,
    _block,
    _episode_outcomes,
    _result_success_rate,
    _window_outcome_mismatches,
)
from hytalegym.jax.evaluation._frozen_outcome_contracts import (  # noqa: F401
    _canonical_sha256,
    _control_contract,
    _execution_identity,
    _file_sha256,
    _historical_binding_contract,
    _is_sha256,
    _protocol_contract,
    _source_identity,
    _source_identity_errors,
)
from hytalegym.jax.evaluation._frozen_outcome_runs import (  # noqa: F401
    _find_run,
    _load_subject,
    _reference_run,
    _run_errors,
    _run_seed,
    _validate_subject_config,
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


def run_frozen_outcome_controls(
    repository_root: Path,
    *,
    compile: bool = True,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Replay the frozen subjects, then run three outcome control arms.

    Current code is never relabelled with historical hashes. Instead, the
    report records both generations and requires exact per-window terminal
    count replay of each historical subject before any control is measured.
    """

    root = repository_root.resolve()
    source_identity_start = _source_identity(root)
    execution_identity_start = _execution_identity(source_identity_start)
    subjects = tuple(_load_subject(root, subject) for subject in _SUBJECTS)
    if execution_identity_start["combat_ruleset_sha256"] != (
        HISTORICAL_COMBAT_RULESET_SHA256
    ):
        raise RuntimeError("combat ruleset changed from the historical subject")
    execution_bridge_sha256 = execution_identity_start[
        "native_evidence_jar_sha256"
    ]
    params = default_combat_params(microticks=MICROTICKS)
    profile_batch = tuple(
        PROFILE_NAMES[index % len(PROFILE_NAMES)]
        for index in range(NUM_ENVIRONMENTS)
    )
    runtime = arsenal_runtime_config(hytale_0_5_7_loadouts(profile_batch))
    validation_bits = np.asarray(runtime.validation_failure_bits)
    if np.any(validation_bits):
        raise RuntimeError(
            f"Arsenal loadout validation failed: {validation_bits.tolist()}"
        )
    environment = make_arsenal_ppo_environment(
        params,
        runtime,
        world_capability_provider=open_flat_arsenal_world_capabilities,
    )
    config = PPOConfig.from_environment_spec(
        environment.spec,
        num_envs=NUM_ENVIRONMENTS,
        rollout_steps=ROLLOUT_STEPS,
        update_epochs=1,
        num_minibatches=8,
        encoder_size=64,
        recurrent_size=128,
        action_head_sizes=REFERENCE_ARSENAL_POLICY_ACTION_HEAD_SIZES,
    )
    for subject in subjects:
        _validate_subject_config(subject, config)

    policy_collector = make_frozen_outcome_collector(
        config,
        environment,
        arm="stochastic_policy",
        compile=compile,
    )
    uniform_collector = make_frozen_outcome_collector(
        config,
        environment,
        arm="uniform_legal",
        compile=compile,
    )
    idle_collector = make_frozen_outcome_collector(
        config,
        environment,
        arm="always_idle",
        compile=compile,
    )

    replay_runs = []
    trained_reference_runs = []
    for subject in subjects:
        trained_reference_runs.extend(subject["reference_runs"])
        for measurement_seed in MEASUREMENT_SEEDS:
            replay = _run_seed(
                config,
                environment,
                policy_collector,
                measurement_seed=measurement_seed,
                policy_params=subject["policy_params"],
                subject_training_seed=subject["training_seed"],
                arm="trained_replay",
                execution_bridge_sha256=execution_bridge_sha256,
                progress=progress,
            )
            reference = _find_run(
                subject["reference_runs"],
                measurement_seed=measurement_seed,
            )
            mismatches = _window_outcome_mismatches(reference, replay)
            replay["matches_reference"] = not mismatches
            replay["mismatches"] = mismatches
            if mismatches:
                raise RuntimeError(
                    "historical subject replay did not reproduce terminal "
                    f"counts for training seed {subject['training_seed']}, "
                    f"measurement seed {measurement_seed}: {mismatches}"
                )
            replay_runs.append(replay)

    uniform_runs = [
        _run_seed(
            config,
            environment,
            uniform_collector,
            measurement_seed=measurement_seed,
            arm="uniform_legal",
            execution_bridge_sha256=execution_bridge_sha256,
            progress=progress,
        )
        for measurement_seed in MEASUREMENT_SEEDS
    ]
    idle_runs = [
        _run_seed(
            config,
            environment,
            idle_collector,
            measurement_seed=measurement_seed,
            arm="always_idle",
            execution_bridge_sha256=execution_bridge_sha256,
            progress=progress,
        )
        for measurement_seed in MEASUREMENT_SEEDS
    ]
    untrained_runs = []
    for subject in subjects:
        policy_key, _ = jax.random.split(
            jax.random.key(subject["training_seed"])
        )
        untrained_params = initialize_policy(policy_key, config)
        for measurement_seed in MEASUREMENT_SEEDS:
            untrained_runs.append(
                _run_seed(
                    config,
                    environment,
                    policy_collector,
                    measurement_seed=measurement_seed,
                    policy_params=untrained_params,
                    subject_training_seed=subject["training_seed"],
                    arm="untrained_init",
                    execution_bridge_sha256=execution_bridge_sha256,
                    progress=progress,
                )
            )

    source_identity = _source_identity(root)
    execution_identity = _execution_identity(source_identity)
    if execution_identity != execution_identity_start:
        raise RuntimeError(
            "evaluation source or contract accessor changed during the "
            "control run; "
            "discarding the mixed-contract measurement"
        )
    suite_contract = {
        "schema": FROZEN_OUTCOME_SUITE_SCHEMA,
        "historical_target_contract": {
            "arsenal_contract_sha256": (
                HISTORICAL_ARSENAL_CONTRACT_SHA256
            ),
            "policy_contract_sha256": (
                HISTORICAL_POLICY_CONTRACT_SHA256
            ),
            "world_geometry_token_contract_sha256": (
                HISTORICAL_WORLD_TOKEN_CONTRACT_SHA256
            ),
            "combat_ruleset_sha256": HISTORICAL_COMBAT_RULESET_SHA256,
            "native_evidence_jar_sha256": HISTORICAL_BRIDGE_SHA256,
            "subjects": [
                {
                    key: subject[key]
                    for key in (
                        "training_seed",
                        "checkpoint",
                        "checkpoint_sha256",
                        "report",
                        "report_sha256",
                    )
                }
                for subject in _SUBJECTS
            ],
        },
        "execution_contract": {
            **execution_identity,
            "source_stable_during_run": True,
            "contract_accessors_stable_during_run": True,
        },
        "protocol": _protocol_contract(),
        "controls": _control_contract(),
        "historical_binding": _historical_binding_contract(),
    }
    results = {
        "trained_policy": _arm_result(
            trained_reference_runs,
            provenance="immutable_historical_probe_reports",
        ),
        "uniform_legal": _arm_result(
            uniform_runs,
            provenance="independent_control_execution",
        ),
        "always_idle": _arm_result(
            idle_runs,
            provenance="independent_control_execution",
        ),
        "untrained_init": _arm_result(
            untrained_runs,
            provenance="independent_control_execution",
        ),
    }
    trained_rate = results["trained_policy"]["aggregate"][
        "episode_outcomes"
    ]["rates"]["success"]
    comparisons = {}
    for arm in ("uniform_legal", "always_idle", "untrained_init"):
        control_rate = results[arm]["aggregate"]["episode_outcomes"][
            "rates"
        ]["success"]
        comparisons[arm] = {
            "trained_success_rate": trained_rate,
            "control_success_rate": control_rate,
            "trained_minus_control_success_rate": (
                trained_rate - control_rate
            ),
        }
    report = {
        "schema": FROZEN_OUTCOME_REPORT_SCHEMA,
        "version": FROZEN_OUTCOME_REPORT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "suite_contract": suite_contract,
        "suite_sha256": _canonical_sha256(suite_contract),
        "contract_interpretation": {
            "comparison_generation": "historical",
            "historical_target": (
                f"{HISTORICAL_ARSENAL_CONTRACT_SHA256}/"
                f"{HISTORICAL_POLICY_CONTRACT_SHA256}"
            ),
            "execution_generation": "recorded_separately",
            "relabelled_as_historical_runtime": False,
            "claim": (
                "controls target the frozen 95.8% outcome protocol after "
                "the current implementation exactly replayed every historical "
                "per-window terminal count"
            ),
        },
        "replay_verification": {
            "all_runs_match": all(
                run["matches_reference"] for run in replay_runs
            ),
            "runs": replay_runs,
        },
        "results": results,
        "comparisons": comparisons,
        "limitations": [
            (
                "The control executions use the separately recorded current "
                "runtime after an exact terminal-count replay gate; this is "
                "not full trajectory identity across contract generations."
            ),
            (
                "Rates are conditional on completed episodes; 128 active "
                "lanes remain censored at the end of each run."
            ),
            (
                "The result is pure JAX, open-flat, stochastic-policy "
                "evidence, not Region, greedy-decoder, or native transfer."
            ),
        ],
    }
    errors = frozen_outcome_report_errors(report)
    if errors:
        raise RuntimeError(
            "frozen outcome report failed validation:\n- "
            + "\n- ".join(errors)
        )
    return report


def write_frozen_outcome_report(path: Path, report: dict[str, Any]) -> None:
    """Write one already validated report."""

    errors = frozen_outcome_report_errors(report)
    if errors:
        raise ValueError("invalid frozen outcome report: " + "; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def frozen_outcome_report_errors(report: dict[str, Any]) -> list[str]:
    """Validate suite identity, run decomposition, and aggregate arithmetic."""

    errors = []
    if report.get("schema") != FROZEN_OUTCOME_REPORT_SCHEMA:
        errors.append("schema is not frozen outcome controls v1")
    if report.get("version") != FROZEN_OUTCOME_REPORT_VERSION:
        errors.append("version is not 1")
    suite = report.get("suite_contract")
    if not isinstance(suite, dict):
        return errors + ["suite_contract is missing"]
    if report.get("suite_sha256") != _canonical_sha256(suite):
        errors.append("suite_sha256 does not match suite_contract")
    if suite.get("schema") != FROZEN_OUTCOME_SUITE_SCHEMA:
        errors.append("suite schema is not frozen outcome controls suite v1")
    historical = suite.get("historical_target_contract", {})
    if historical.get("arsenal_contract_sha256") != (
        HISTORICAL_ARSENAL_CONTRACT_SHA256
    ):
        errors.append("historical Arsenal contract changed")
    if historical.get("policy_contract_sha256") != (
        HISTORICAL_POLICY_CONTRACT_SHA256
    ):
        errors.append("historical policy contract changed")
    if historical.get("world_geometry_token_contract_sha256") != (
        HISTORICAL_WORLD_TOKEN_CONTRACT_SHA256
    ):
        errors.append("historical World-token contract changed")
    if historical.get("combat_ruleset_sha256") != (
        HISTORICAL_COMBAT_RULESET_SHA256
    ):
        errors.append("historical combat ruleset changed")
    if historical.get("native_evidence_jar_sha256") != HISTORICAL_BRIDGE_SHA256:
        errors.append("historical bridge identity changed")
    expected_subjects = [
        {
            key: subject[key]
            for key in (
                "training_seed",
                "checkpoint",
                "checkpoint_sha256",
                "report",
                "report_sha256",
            )
        }
        for subject in _SUBJECTS
    ]
    if historical.get("subjects") != expected_subjects:
        errors.append("historical subject inventory changed")
    if suite.get("protocol") != _protocol_contract():
        errors.append("suite protocol changed")
    if suite.get("controls") != _control_contract():
        errors.append("control definitions changed")
    if suite.get("historical_binding") != _historical_binding_contract():
        errors.append("historical replay binding changed")
    execution = suite.get("execution_contract")
    execution_bridge = None
    if not isinstance(execution, dict):
        errors.append("execution_contract is missing")
    else:
        for field in (
            "arsenal_contract_sha256",
            "policy_contract_sha256",
            "world_geometry_token_contract_sha256",
            "combat_ruleset_sha256",
            "native_evidence_jar_sha256",
        ):
            if not _is_sha256(execution.get(field)):
                errors.append(f"execution {field} is not a SHA-256")
        if execution.get("combat_ruleset_sha256") != (
            HISTORICAL_COMBAT_RULESET_SHA256
        ):
            errors.append("execution combat ruleset changed")
        if execution.get("source_stable_during_run") is not True:
            errors.append("execution source was not stable during the run")
        if execution.get("contract_accessors_stable_during_run") is not True:
            errors.append(
                "execution contract accessors were not stable during the run"
            )
        errors.extend(_source_identity_errors(execution.get("source_identity")))
        execution_bridge = execution.get("native_evidence_jar_sha256")
    replay = report.get("replay_verification", {})
    replay_runs = replay.get("runs")
    if not isinstance(replay_runs, list) or len(replay_runs) != 4:
        errors.append("replay_verification must contain four runs")
    elif (
        not replay.get("all_runs_match")
        or not all(run.get("matches_reference") for run in replay_runs)
    ):
        errors.append("historical replay verification is not exact")
    results = report.get("results")
    if not isinstance(results, dict):
        return errors + ["results are missing"]
    expected_run_counts = {
        "trained_policy": 4,
        "uniform_legal": 2,
        "always_idle": 2,
        "untrained_init": 4,
    }
    expected_axes = {
        "trained_policy": [
            (subject["training_seed"], measurement_seed)
            for subject in _SUBJECTS
            for measurement_seed in MEASUREMENT_SEEDS
        ],
        "uniform_legal": [
            (None, measurement_seed) for measurement_seed in MEASUREMENT_SEEDS
        ],
        "always_idle": [
            (None, measurement_seed) for measurement_seed in MEASUREMENT_SEEDS
        ],
        "untrained_init": [
            (subject["training_seed"], measurement_seed)
            for subject in _SUBJECTS
            for measurement_seed in MEASUREMENT_SEEDS
        ],
    }
    for arm, expected_count in expected_run_counts.items():
        result = results.get(arm)
        if not isinstance(result, dict):
            errors.append(f"result {arm!r} is missing")
            continue
        runs = result.get("runs")
        if not isinstance(runs, list) or len(runs) != expected_count:
            errors.append(
                f"result {arm!r} must contain {expected_count} runs"
            )
            continue
        for index, run in enumerate(runs):
            errors.extend(
                f"{arm}.runs[{index}]: {error}"
                for error in _run_errors(run)
            )
            if run.get("arm") != arm:
                errors.append(f"{arm}.runs[{index}] has the wrong arm")
            expected_bridge = (
                HISTORICAL_BRIDGE_SHA256
                if arm == "trained_policy"
                else execution_bridge
            )
            if run.get("execution_bridge_sha256") != expected_bridge:
                errors.append(
                    f"{arm}.runs[{index}] has the wrong execution bridge"
                )
        axes = [
            (
                run.get("subject_training_seed"),
                run.get("measurement_seed"),
            )
            for run in runs
        ]
        if axes != expected_axes[arm]:
            errors.append(f"result {arm!r} run axes changed")
        try:
            expected_aggregate = _aggregate_runs(runs)
        except (KeyError, TypeError, RuntimeError) as error:
            errors.append(f"result {arm!r} aggregation failed: {error}")
        else:
            if result.get("aggregate") != expected_aggregate:
                errors.append(f"result {arm!r} aggregate does not decompose")
    trained_result = results.get("trained_policy")
    if (
        isinstance(replay_runs, list)
        and isinstance(trained_result, dict)
        and isinstance(trained_result.get("runs"), list)
    ):
        replay_axes = [
            (
                run.get("subject_training_seed"),
                run.get("measurement_seed"),
            )
            for run in replay_runs
        ]
        if replay_axes != expected_axes["trained_policy"]:
            errors.append("historical replay run axes changed")
        for replay_run in replay_runs:
            if replay_run.get("arm") != "trained_replay":
                errors.append("historical replay run has the wrong arm")
            if replay_run.get("execution_bridge_sha256") != execution_bridge:
                errors.append("historical replay execution bridge changed")
            matches = [
                run
                for run in trained_result["runs"]
                if run.get("measurement_seed")
                == replay_run.get("measurement_seed")
                and run.get("subject_training_seed")
                == replay_run.get("subject_training_seed")
            ]
            if len(matches) != 1:
                errors.append("replay run has no unique historical subject")
                continue
            if _window_outcome_mismatches(matches[0], replay_run):
                errors.append("replay run differs from historical subject")
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, dict):
        errors.append("comparisons are missing")
    elif isinstance(trained_result, dict):
        trained_rate = _result_success_rate(trained_result)
        for arm in ("uniform_legal", "always_idle", "untrained_init"):
            control_rate = _result_success_rate(results.get(arm))
            expected = {
                "trained_success_rate": trained_rate,
                "control_success_rate": control_rate,
                "trained_minus_control_success_rate": (
                    trained_rate - control_rate
                    if isinstance(trained_rate, (int, float))
                    and isinstance(control_rate, (int, float))
                    else None
                ),
            }
            if comparisons.get(arm) != expected:
                errors.append(f"comparison {arm!r} does not recompute")
    return errors


__all__ = [
    "FROZEN_OUTCOME_REPORT_SCHEMA",
    "FROZEN_OUTCOME_REPORT_VERSION",
    "FROZEN_OUTCOME_SUITE_SCHEMA",
    "HISTORICAL_ARSENAL_CONTRACT_SHA256",
    "HISTORICAL_BRIDGE_SHA256",
    "HISTORICAL_COMBAT_RULESET_SHA256",
    "HISTORICAL_POLICY_CONTRACT_SHA256",
    "HISTORICAL_WORLD_TOKEN_CONTRACT_SHA256",
    "MEASUREMENT_SEEDS",
    "frozen_outcome_report_errors",
    "run_frozen_outcome_controls",
    "write_frozen_outcome_report",
]
