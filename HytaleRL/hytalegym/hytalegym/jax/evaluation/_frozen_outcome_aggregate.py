"""Outcome aggregation and window statistics for the frozen outcome report."""
from __future__ import annotations

from typing import Any

import jax



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


def _episode_outcomes(
    counts: dict[str, int],
    episodes_completed: int,
) -> dict[str, Any]:
    if set(counts) != set(_OUTCOME_FIELDS):
        raise RuntimeError("outcome count fields are incomplete")
    if sum(counts.values()) != episodes_completed:
        raise RuntimeError("outcome counts do not sum to episodes_completed")
    return {
        "counts": counts,
        "rates": {
            field: (
                counts[field] / episodes_completed
                if episodes_completed
                else 0.0
            )
            for field in _OUTCOME_FIELDS
        },
        "episodes_completed": episodes_completed,
        "rate_denominator": "episodes_completed",
        "censoring": (
            "episodes still active after the final rollout are not classified"
        ),
        "taxonomy": {
            "success": "target_dead_and_agent_alive",
            "death": "agent_dead_and_target_alive",
            "simultaneous": "target_and_agent_dead",
            "other_terminal": (
                "terminal_without_either_health_reaching_zero"
            ),
        },
    }


def _aggregate_windows(windows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        field: sum(window["episode_outcomes"][field] for window in windows)
        for field in _OUTCOME_FIELDS
    }
    episodes_completed = sum(
        window["episodes_completed"] for window in windows
    )
    return _episode_outcomes(counts, episodes_completed)


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        field: sum(
            run["episode_outcomes"]["counts"][field] for run in runs
        )
        for field in _OUTCOME_FIELDS
    }
    episodes_completed = sum(
        run["episode_outcomes"]["episodes_completed"] for run in runs
    )
    return {
        "decisions": sum(run["decisions"] for run in runs),
        "episode_outcomes": _episode_outcomes(
            counts,
            episodes_completed,
        ),
    }


def _arm_result(
    runs: list[dict[str, Any]],
    *,
    provenance: str,
) -> dict[str, Any]:
    return {
        "provenance": provenance,
        "runs": runs,
        "aggregate": _aggregate_runs(runs),
    }


def _result_success_rate(result: Any) -> float | None:
    if not isinstance(result, dict):
        return None
    aggregate = result.get("aggregate")
    if not isinstance(aggregate, dict):
        return None
    outcomes = aggregate.get("episode_outcomes")
    if not isinstance(outcomes, dict):
        return None
    rates = outcomes.get("rates")
    if not isinstance(rates, dict):
        return None
    value = rates.get("success")
    return value if isinstance(value, (int, float)) else None


def _window_outcome_mismatches(
    reference: dict[str, Any],
    replay: dict[str, Any],
) -> list[str]:
    mismatches = []
    reference_windows = reference.get("windows")
    replay_windows = replay.get("windows")
    if not isinstance(reference_windows, list) or not isinstance(
        replay_windows,
        list,
    ):
        return ["window records are missing"]
    if len(reference_windows) != len(replay_windows):
        return [
            "window count differs: "
            f"{len(replay_windows)} != {len(reference_windows)}"
        ]
    for index, (expected, actual) in enumerate(
        zip(reference_windows, replay_windows, strict=True),
        start=1,
    ):
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            mismatches.append(f"window {index} is not an object")
            continue
        for field in ("episodes_completed", "episode_outcomes"):
            if expected.get(field) != actual.get(field):
                mismatches.append(
                    f"window {index} {field}: "
                    f"{actual.get(field)!r} != {expected.get(field)!r}"
                )
    if reference.get("episode_outcomes") != replay.get("episode_outcomes"):
        mismatches.append("whole-seed episode_outcomes differ")
    return mismatches


def _block(tree: Any) -> None:
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
