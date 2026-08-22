"""Compiled checkpoint and action-source evaluation for melee combat."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
import hashlib
import json
import math
from typing import Any, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp

from hytalegym.jax.combat import (
    TARGET_ENTITY,
    CombatParams,
    reset_batch,
    skills_to_actions,
    step_batch,
)
from hytalegym.jax.training.policy import apply_policy
from hytalegym.jax.training.types import RecurrentPolicyParams


class EvaluationResult(NamedTuple):
    success: jax.Array
    terminated: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array


ActionSource = Callable[
    [jax.Array, Any, jax.Array],
    tuple[jax.Array, Any],
]
ActionCarryInitializer = Callable[[int], Any]
Evaluator = Callable[
    [RecurrentPolicyParams | None, jax.Array, jax.Array],
    EvaluationResult,
]

EVALUATION_ARTIFACT_SCHEMA = "hytalerl_evaluation_artifact_v1"
EVALUATION_ARTIFACT_VERSION = 1
NATIVE_SERVER_PROCESS_UPTIME_SOURCE = (
    "backend_info.native_server_process_uptime_seconds"
)
_EVALUATION_CONTRACT_REQUIRED_FIELDS = frozenset(
    (
        "schema",
        "version",
        "environment",
        "capability_provider",
        "evaluation_seeds",
        "episode_count",
        "policy_checkpoint",
        "baselines",
        "combat_ruleset_sha256",
        "arsenal_contract_sha256",
        "repository",
    )
)


def pure_jax_process_provenance() -> dict[str, Any]:
    """Describe why native process age cannot affect a pure-JAX result."""

    return {
        "mode": "pure_jax_uptime_independent",
        "in_timed_path": False,
        "uptime_independent": True,
        "statement": "pure-JAX evaluation does not contact a native server",
    }


def native_server_process_interval_errors(
    start: Any,
    end: Any,
    *,
    expected_in_timed_path: bool,
) -> list[str]:
    """Validate two connected-backend process-age captures."""

    errors = []
    captures = (
        ("native_server_process_at_start", start),
        ("native_server_process_at_end", end),
    )
    valid_uptimes: dict[str, float] = {}
    valid_times: dict[str, datetime] = {}
    for name, capture in captures:
        if not isinstance(capture, Mapping):
            errors.append(f"{name} must be a mapping")
            continue
        captured_utc = _utc_datetime(capture.get("captured_utc"))
        if captured_utc is None:
            errors.append(
                f"{name}.captured_utc must be an offset-aware UTC "
                "ISO-8601 timestamp"
            )
        else:
            valid_times[name] = captured_utc
        if capture.get("in_timed_path") is not expected_in_timed_path:
            errors.append(
                f"{name}.in_timed_path must be "
                f"{str(expected_in_timed_path).lower()}"
            )
        uptime = capture.get("native_server_process_uptime_seconds")
        if (
            isinstance(uptime, bool)
            or not isinstance(uptime, (int, float))
            or not math.isfinite(float(uptime))
            or uptime < 0
        ):
            errors.append(
                f"{name}.native_server_process_uptime_seconds must be "
                "nonnegative and finite"
            )
        else:
            valid_uptimes[name] = float(uptime)
        if capture.get("source") != NATIVE_SERVER_PROCESS_UPTIME_SOURCE:
            errors.append(
                f"{name}.source must identify connected-backend uptime"
            )
    if len(valid_times) == 2 and valid_times[
        "native_server_process_at_end"
    ] < valid_times["native_server_process_at_start"]:
        errors.append("native server process capture time decreased")
    if len(valid_uptimes) == 2 and valid_uptimes[
        "native_server_process_at_end"
    ] < valid_uptimes["native_server_process_at_start"]:
        errors.append(
            "native server process uptime decreased between start and end"
        )
    return errors


def make_evaluator(
    environment_params: CombatParams,
    *,
    max_policy_steps: int = 256,
    compile: bool = True,
    action_source: ActionSource | None = None,
    action_carry_initializer: ActionCarryInitializer | None = None,
) -> Evaluator:
    """Build a full-batch evaluation scan.

    With no ``action_source``, this preserves the original deterministic
    greedy-policy behavior. An explicit source receives
    ``(observation, carry, key)`` and may be evaluated with ``policy_params``
    set to ``None``. Stateless sources use an empty batched carry by default.
    """

    if max_policy_steps < 1:
        raise ValueError("max_policy_steps must be positive")
    if action_source is None and action_carry_initializer is not None:
        raise ValueError(
            "action_carry_initializer requires an explicit action_source"
        )

    def evaluate(
        policy_params: RecurrentPolicyParams | None,
        reset_keys: jax.Array,
        rollout_key: jax.Array,
    ) -> EvaluationResult:
        batch_size = reset_keys.shape[0]
        environment_state, observation = reset_batch(
            reset_keys,
            environment_params,
        )
        if action_source is None:
            if policy_params is None:
                raise ValueError(
                    "policy_params are required when action_source is omitted"
                )
            recurrent_size = policy_params.gru.recurrent_kernel.shape[0]
            initial_action_carry = jnp.zeros(
                (batch_size, recurrent_size),
                dtype=jnp.float32,
            )
        else:
            initializer = (
                _empty_action_carry
                if action_carry_initializer is None
                else action_carry_initializer
            )
            initial_action_carry = initializer(batch_size)
        initial_carry = (
            environment_state,
            observation,
            initial_action_carry,
            jnp.zeros((batch_size,), dtype=jnp.bool_),
            jnp.zeros((batch_size,), dtype=jnp.bool_),
            jnp.zeros((batch_size,), dtype=jnp.float32),
            jnp.zeros((batch_size,), dtype=jnp.int32),
        )
        step_keys = jax.random.split(rollout_key, max_policy_steps)

        def evaluate_step(carry, step_key):
            (
                state,
                current_observation,
                recurrent_state,
                finished,
                success,
                episode_return,
                episode_length,
            ) = carry
            if action_source is None:
                recurrent_candidate, logits, _ = apply_policy(
                    policy_params,
                    current_observation,
                    recurrent_state,
                )
                skill_ids = jnp.argmax(logits, axis=-1).astype(jnp.int32)
                actions = skills_to_actions(
                    current_observation,
                    skill_ids,
                )
            else:
                actions, recurrent_candidate = action_source(
                    current_observation,
                    recurrent_state,
                    jax.random.fold_in(step_key, 1),
                )
            environment_keys = jax.random.split(step_key, batch_size)
            (
                candidate_state,
                candidate_observation,
                reward,
                done,
                _,
            ) = step_batch(
                state,
                actions,
                environment_keys,
                environment_params,
            )
            active = ~finished
            newly_done = active & done
            state = _batch_select(active, candidate_state, state)
            current_observation = jnp.where(
                active[:, None],
                candidate_observation,
                current_observation,
            )
            recurrent_state = jnp.where(
                active[:, None],
                recurrent_candidate,
                recurrent_state,
            )
            episode_return = episode_return + jnp.where(
                active,
                reward,
                jnp.float32(0.0),
            )
            episode_length = episode_length + active.astype(jnp.int32)
            success = success | (
                newly_done
                & (candidate_state.health[:, TARGET_ENTITY] <= 0.0)
            )
            finished = finished | newly_done
            return (
                state,
                current_observation,
                recurrent_state,
                finished,
                success,
                episode_return,
                episode_length,
            ), None

        final_carry, _ = jax.lax.scan(
            evaluate_step,
            initial_carry,
            step_keys,
        )
        (
            _,
            _,
            _,
            finished,
            success,
            episode_return,
            episode_length,
        ) = final_carry
        return EvaluationResult(
            success=success,
            terminated=finished,
            episode_return=episode_return,
            episode_length=episode_length,
        )

    if compile:
        return jax.jit(evaluate)
    return evaluate


def evaluation_contract_sha256(contract: Mapping[str, Any]) -> str:
    """Hash one JSON-compatible evaluation contract canonically."""

    payload = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def evaluation_artifact_contract_errors(
    artifact: Mapping[str, Any],
) -> list[str]:
    """Return fail-closed schema and identity errors for one result artifact."""

    errors = []
    if artifact.get("schema") != EVALUATION_ARTIFACT_SCHEMA:
        errors.append(
            "schema: "
            f"artifact={artifact.get('schema')!r}, "
            f"current={EVALUATION_ARTIFACT_SCHEMA!r}"
        )
    if artifact.get("version") != EVALUATION_ARTIFACT_VERSION:
        errors.append(
            "version: "
            f"artifact={artifact.get('version')!r}, "
            f"current={EVALUATION_ARTIFACT_VERSION!r}"
        )
    contract = artifact.get("contract")
    if not isinstance(contract, Mapping):
        return errors + ["contract must be a mapping"]
    missing = sorted(_EVALUATION_CONTRACT_REQUIRED_FIELDS - contract.keys())
    if missing:
        errors.append(f"contract missing required fields: {missing}")
    actual_hash = artifact.get("contract_sha256")
    expected_hash = evaluation_contract_sha256(contract)
    if actual_hash != expected_hash:
        errors.append(
            "contract_sha256: "
            f"artifact={actual_hash!r}, current={expected_hash!r}"
        )
    environment = contract.get("environment")
    bridge_sha256 = contract.get("bridge_sha256")
    if environment in ("native", "headless"):
        if not _is_sha256(bridge_sha256):
            errors.append(
                "native/headless evaluation requires a 64-hex bridge_sha256"
            )
        errors.extend(
            native_server_process_interval_errors(
                contract.get("native_server_process_at_start"),
                contract.get("native_server_process_at_end"),
                expected_in_timed_path=False,
            )
        )
    elif environment == "jax_arsenal":
        if (
            contract.get("native_server_process")
            != pure_jax_process_provenance()
        ):
            errors.append(
                "jax_arsenal evaluation must declare pure-JAX "
                "uptime independence"
            )
    for name in ("combat_ruleset_sha256", "arsenal_contract_sha256"):
        if not _is_sha256(contract.get(name)):
            errors.append(f"{name} must be a 64-hex SHA-256")
    checkpoint = contract.get("policy_checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or not _is_sha256(checkpoint.get("sha256"))
    ):
        errors.append("policy_checkpoint.sha256 must be a 64-hex SHA-256")
    return errors


def evaluation_statistics(
    result: EvaluationResult,
    *,
    labels: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Convert a device result into aggregate and optional labeled statistics."""

    rows = _evaluation_rows(result, labels=labels, seeds=seeds)
    summary = _summarize_rows(rows)
    summary["episodes"] = rows
    if labels is not None:
        by_label: dict[str, Any] = {}
        for label in dict.fromkeys(labels):
            by_label[label] = _summarize_rows(
                [row for row in rows if row["label"] == label]
            )
        summary["by_label"] = by_label
    return summary


def summarize_evaluation_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive the canonical aggregate from stored evaluation episode rows."""

    return _summarize_rows(rows)


def _evaluation_rows(
    result: EvaluationResult,
    *,
    labels: Sequence[str] | None,
    seeds: Sequence[int] | None,
) -> list[dict[str, Any]]:
    fields = result._asdict()
    reward_components = fields.pop("reward_components", None)
    leaves = {
        name: jax.device_get(value).tolist()
        for name, value in fields.items()
    }
    component_leaves = (
        None
        if reward_components is None
        else {
            name: jax.device_get(value).tolist()
            for name, value in reward_components._asdict().items()
        }
    )
    episode_count = len(leaves["success"])
    if labels is not None and len(labels) != episode_count:
        raise ValueError("labels must match the evaluation batch")
    if seeds is not None and len(seeds) != episode_count:
        raise ValueError("seeds must match the evaluation batch")
    rows = []
    for index in range(episode_count):
        row = {
            "batch_index": index,
            "success": bool(leaves["success"][index]),
            "terminated": bool(leaves["terminated"][index]),
            "episode_return": float(leaves["episode_return"][index]),
            "episode_length": int(leaves["episode_length"][index]),
        }
        if labels is not None:
            row["label"] = labels[index]
        if seeds is not None:
            row["seed"] = int(seeds[index])
        if "profile_weapon_id" in leaves:
            row["profile_weapon_id"] = int(
                leaves["profile_weapon_id"][index]
            )
        if component_leaves is not None:
            row["reward_components"] = {
                "target_damage": float(
                    component_leaves["target_damage"][index]
                ),
                "agent_damage": float(
                    component_leaves["agent_damage"][index]
                ),
                "completion": bool(
                    component_leaves["completion"][index]
                ),
                "death": bool(component_leaves["death"][index]),
            }
        rows.append(row)
    return rows


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one evaluation episode is required")
    returns = jnp.asarray(
        [row["episode_return"] for row in rows],
        dtype=jnp.float32,
    )
    lengths = jnp.asarray(
        [row["episode_length"] for row in rows],
        dtype=jnp.float32,
    )
    episode_count = len(rows)
    success_count = sum(bool(row["success"]) for row in rows)
    terminated_count = sum(bool(row["terminated"]) for row in rows)
    summary = {
        "episode_count": episode_count,
        "success_count": success_count,
        "success_rate": success_count / episode_count,
        "terminated_count": terminated_count,
        "terminated_rate": terminated_count / episode_count,
        "episode_return": _spread(returns),
        "episode_length": _spread(lengths),
    }
    if "reward_components" in rows[0]:
        component_summary = {}
        for name in (
            "target_damage",
            "agent_damage",
            "completion",
            "death",
        ):
            values = jnp.asarray(
                [
                    row["reward_components"][name]
                    for row in rows
                ],
                dtype=jnp.float32,
            )
            component_summary[name] = _spread(values)
            if name in ("completion", "death"):
                component_summary[name]["count"] = int(jnp.sum(values))
                component_summary[name]["rate"] = float(jnp.mean(values))
        summary["reward_components"] = component_summary
    return summary


def _spread(values: jax.Array) -> dict[str, float]:
    return {
        "mean": float(jnp.mean(values)),
        "standard_deviation": float(jnp.std(values)),
        "minimum": float(jnp.min(values)),
        "maximum": float(jnp.max(values)),
    }


def _empty_action_carry(batch_size: int) -> jax.Array:
    return jnp.zeros((batch_size, 0), dtype=jnp.float32)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _utc_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.utcoffset() != timedelta(0):
        return None
    return parsed


def _batch_select(mask, when_true, when_false):
    def select_leaf(true_leaf, false_leaf):
        expanded_mask = mask.reshape(
            (mask.shape[0],)
            + (1,) * (true_leaf.ndim - 1)
        )
        return jnp.where(expanded_mask, true_leaf, false_leaf)

    return jax.tree.map(select_leaf, when_true, when_false)
