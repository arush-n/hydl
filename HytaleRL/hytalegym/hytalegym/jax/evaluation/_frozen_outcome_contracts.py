"""Contract digests and source/execution identity for the frozen outcome report."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


from hytalegym.jax.combat import (
    arsenal_policy_contract_sha256,
    combat_arsenal_contract_sha256,
)
from hytalegym.jax.world import world_geometry_token_contract_sha256
from hytalegym.rulesets import combat_ruleset_sha256
from hytalegym.worldgen.region import current_native_evidence_jar_sha256


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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _protocol_contract() -> dict[str, Any]:
    return {
        "backend": "pure_jax",
        "native_server_uptime_dependency": False,
        "world_capability_mode": "explicit_open_flat_control",
        "profile_assignment": "31_profiles_round_robin_across_128_fixed_lanes",
        "measurement_seeds": list(MEASUREMENT_SEEDS),
        "num_environments": NUM_ENVIRONMENTS,
        "rollout_steps": ROLLOUT_STEPS,
        "rollout_windows": ROLLOUT_WINDOWS,
        "microticks": MICROTICKS,
        "action_decoder": "stochastic_factored_categorical_sampling",
        "autoreset": "normal_after_terminal",
        "unfinished_episode_state": "carried_between_windows",
        "rate_denominator": "episodes_completed",
        "final_active_rows": "censored",
        "optimizer_steps": 0,
    }


def _control_contract() -> dict[str, Any]:
    return {
        "uniform_legal": {
            **_FROZEN_CONTROL_IDENTITIES["uniform_legal"],
            "subject_axis": (
                "checkpoint_independent_executed_once_per_measurement_seed"
            ),
        },
        "always_idle": {
            **_FROZEN_CONTROL_IDENTITIES["always_idle"],
            "subject_axis": (
                "checkpoint_independent_executed_once_per_measurement_seed"
            ),
        },
        "untrained_init": {
            "schema": "hytalerl_untrained_stochastic_policy_baseline_v1",
            "version": 1,
            "initialization": "same policy initialization seed as each subject",
            "selection": "stochastic legal factored categorical sampling",
        },
    }


def _historical_binding_contract() -> dict[str, Any]:
    return {
        "method": "exact_per_window_terminal_count_replay",
        "scope": (
            "terminal outcome counts and episode completion only; "
            "not a claim of full cross-contract trajectory identity"
        ),
        "required_before_controls": True,
    }


def _source_identity(root: Path) -> dict[str, Any]:
    identity = {
        name: {"path": path, "sha256": _file_sha256(root / path)}
        for name, path in _SOURCE_PATHS.items()
    }
    semantic_files = sorted(
        path
        for relative_root in _SEMANTIC_SOURCE_ROOTS
        for path in (root / relative_root).rglob("*")
        if path.is_file() and path.suffix in {".py", ".json"}
    )
    digest = hashlib.sha256()
    for path in semantic_files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    identity["semantic_source_tree"] = {
        "roots": list(_SEMANTIC_SOURCE_ROOTS),
        "file_count": len(semantic_files),
        "sha256": digest.hexdigest().upper(),
    }
    return identity


def _source_identity_errors(identity: Any) -> list[str]:
    if not isinstance(identity, dict):
        return ["execution source_identity is missing"]
    expected_keys = set(_SOURCE_PATHS) | {"semantic_source_tree"}
    if set(identity) != expected_keys:
        return ["execution source_identity fields changed"]
    errors = []
    for name, expected_path in _SOURCE_PATHS.items():
        item = identity.get(name)
        if (
            not isinstance(item, dict)
            or item.get("path") != expected_path
            or not _is_sha256(item.get("sha256"))
        ):
            errors.append(f"execution source identity {name!r} is invalid")
    semantic = identity.get("semantic_source_tree")
    if not isinstance(semantic, dict):
        errors.append("execution semantic source tree is missing")
    else:
        if semantic.get("roots") != list(_SEMANTIC_SOURCE_ROOTS):
            errors.append("execution semantic source roots changed")
        if not isinstance(semantic.get("file_count"), int) or (
            semantic["file_count"] <= 0
        ):
            errors.append("execution semantic source file count is invalid")
        if not _is_sha256(semantic.get("sha256")):
            errors.append("execution semantic source hash is invalid")
    return errors


def _execution_identity(source_identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "arsenal_contract_sha256": combat_arsenal_contract_sha256(),
        "policy_contract_sha256": arsenal_policy_contract_sha256(),
        "world_geometry_token_contract_sha256": (
            world_geometry_token_contract_sha256()
        ),
        "combat_ruleset_sha256": combat_ruleset_sha256(),
        "native_evidence_jar_sha256": current_native_evidence_jar_sha256(),
        "source_identity": source_identity,
    }
