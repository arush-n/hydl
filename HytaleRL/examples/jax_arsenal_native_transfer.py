#!/usr/bin/env python3
"""Run one identity-stable current Arsenal policy on JAX and native Hytale."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.envs.hytale_env import HytaleEnv
from hytalegym.jax.combat import (
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
    PROFILE_NAMES,
    arsenal_runtime_config,
    default_combat_params,
    hytale_0_5_7_loadouts,
    summarize_native_action_accounting,
)
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES,
    greedy_arsenal_action_factors,
    neutral_arsenal_policy_action_factors,
)
from hytalegym.jax.combat.native.evidence import (
    acquire_combat_native_evidence_leases,
)
from hytalegym.jax.combat.contracts.publication import (
    combat_contract_stamp,
)
from hytalegym.jax.evaluation.arsenal_benchmark import (
    exact_empty_local_geometry,
)
from hytalegym.jax.evaluation.arsenal_native_transfer import (
    ARSENAL_NATIVE_TRANSFER_SCHEMA,
    ARSENAL_NATIVE_TRANSFER_VERSION,
    arsenal_native_transfer_artifact_errors,
    canonical_sha256,
    paired_transfer_summary,
)
from hytalegym.jax.training.arsenal import (
    geometry_arsenal_world_capabilities,
)
from hytalegym.jax.training.arsenal_evaluation import (
    make_arsenal_evaluator,
    recurrent_arsenal_policy_action_source,
)
from hytalegym.jax.training.checkpoint import (
    ARSENAL_POLICY_SURFACE,
    combat_checkpoint_contract_errors,
    load_policy_checkpoint,
)
from hytalegym.jax.training.evaluation import evaluation_statistics
from hytalegym.jax.training.policy import apply_policy


DEFAULT_EPISODE_SEEDS = (570057, 1553616197)
DEFAULT_ROLLOUT_SEED = 744903660


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--native-port", type=int, default=5556)
    parser.add_argument("--maximum-policy-steps", type=int, default=256)
    parser.add_argument(
        "--native-only-diagnostic",
        action="store_true",
        help=(
            "skip the JAX column and exercise native masks/actions only; "
            "no transfer certificate is published"
        ),
    )
    parser.add_argument(
        "--native-action-source",
        choices=("policy", "neutral"),
        default="policy",
        help="native diagnostic action source; certificates always use policy",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_EPISODE_SEEDS),
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=PROFILE_NAMES,
        default=list(PROFILE_NAMES),
    )
    args = parser.parse_args()
    if args.maximum_policy_steps < 1:
        parser.error("--maximum-policy-steps must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be distinct")
    if len(set(args.profiles)) != len(args.profiles):
        parser.error("--profiles must be distinct")
    if (
        args.native_action_source != "policy"
        and not args.native_only_diagnostic
    ):
        parser.error("--native-action-source=neutral is diagnostic-only")

    root = args.repository_root.resolve()
    checkpoint = args.checkpoint.resolve()
    git_sha_start = _git_sha(root)
    identity_start = combat_contract_stamp(
        root,
        repository_git_sha=git_sha_start,
    )
    policy_params, config, metadata = load_policy_checkpoint(checkpoint)
    _require_current_checkpoint(config, metadata, identity_start)
    checkpoint_sha = _file_sha256(checkpoint)

    profiles = tuple(args.profiles)
    seeds = tuple(args.seeds)
    suite = {
        "schema": "hytalerl_arsenal_native_transfer_suite_v1",
        "version": 1,
        "profiles": list(profiles),
        "episode_seeds": list(seeds),
        "episode_key_order": "profile_then_seed",
        "seed_provenance": {
            str(seeds[0]): "fixed_regression_seed",
            **(
                {str(seeds[1]): ("randomized_control_pcg64_seed_20260729_first_draw")}
                if len(seeds) > 1 and seeds[1] == 1553616197
                else {}
            ),
        },
        "rollout_seed": DEFAULT_ROLLOUT_SEED,
        "maximum_policy_steps": args.maximum_policy_steps,
        "microticks": 1,
        "world": "flat",
        "target_active": True,
        "policy_decoder": "deterministic_legality_conditioned_standard_root_argmax",
        "policy_checkpoint_sha256": checkpoint_sha,
        "limitations": {
            "opponent": "Trork_Brawler",
            "geometry": "controlled flat fixture",
            "simulator_column": (
                "unavailable: standalone 5557 rejects native actor evidence"
            ),
            "world_action_candidates": "unavailable_fail_closed",
            "outcome_equivalence": (
                "measured_not_assumed; execution validity does not require "
                "JAX/native success equality"
            ),
        },
    }
    if args.native_only_diagnostic:
        with acquire_combat_native_evidence_leases(
            ((args.host, args.native_port),),
            purpose="current_arsenal_native_transfer_diagnostic",
            repository_root=root,
            evidence_kinds=(
                "combat_fidelity",
                "native_actor_evidence",
                "learner_v3",
                "policy_transfer_diagnostic",
            ),
        ):
            native_result, process = _run_native(
                policy_params,
                config.recurrent_size,
                profiles,
                seeds,
                args.maximum_policy_steps,
                host=args.host,
                port=args.native_port,
                expected_bridge=str(identity_start["identities"]["bridge"]),
                action_source=args.native_action_source,
                include_step_trace=True,
            )
        diagnostic = {
            "schema": "hytalerl_arsenal_native_transfer_diagnostic_v1",
            "version": 1,
            "diagnostic_only": True,
            "suite": suite,
            "contract": {
                "publication_identity": identity_start,
                "publication_identity_sha256": canonical_sha256(identity_start),
                "policy_checkpoint": {
                    "filename": checkpoint.name,
                    "sha256": checkpoint_sha,
                    "metadata": metadata,
                    "config": asdict(config),
                },
            },
            "native_server_process": process,
            "native": native_result,
        }
        _write_json_atomic(args.output, diagnostic)
        print(f"bridge_sha256={native_result['connected_bridge_sha256']}")
        print(
            "native_rows="
            f"{native_result['episode_count']} episodes, "
            f"{native_result['invalid_observation_count']} invalid observations, "
            f"{native_result['host_rejected_step_count']} host-rejected steps"
        )
        print(f"artifact={args.output.resolve()}")
        return
    jax_result = _run_jax(
        policy_params,
        profiles,
        seeds,
        args.maximum_policy_steps,
    )

    with acquire_combat_native_evidence_leases(
        ((args.host, args.native_port),),
        purpose="current_arsenal_native_transfer",
        repository_root=root,
        evidence_kinds=(
            "combat_fidelity",
            "native_actor_evidence",
            "learner_v3",
            "policy_transfer",
        ),
    ):
        native_result, process = _run_native(
            policy_params,
            config.recurrent_size,
            profiles,
            seeds,
            args.maximum_policy_steps,
            host=args.host,
            port=args.native_port,
            expected_bridge=str(identity_start["identities"]["bridge"]),
            action_source="policy",
            include_step_trace=False,
        )

    git_sha_end = _git_sha(root)
    identity_end = combat_contract_stamp(
        root,
        repository_git_sha=git_sha_end,
    )
    if identity_start != identity_end:
        raise RuntimeError("Combat publication identity moved during the transfer run")

    differential = paired_transfer_summary(
        jax_result["episodes"],
        native_result["episodes"],
    )
    contract = {
        "repository_git_sha_at_start": git_sha_start,
        "repository_git_sha_at_end": git_sha_end,
        "publication_identity_at_start": identity_start,
        "publication_identity_at_end": identity_end,
        "publication_identity_sha256": canonical_sha256(identity_start),
        "connected_bridge_sha256": native_result["connected_bridge_sha256"],
        "policy_checkpoint": {
            "filename": checkpoint.name,
            "sha256": checkpoint_sha,
            "metadata": metadata,
            "config": asdict(config),
        },
        "policy_shape": dict(identity_start["shape"]),
        "native_server_process": process,
    }
    results = {
        "jax": jax_result,
        "native": native_result,
        "differential": differential,
    }
    artifact = {
        "schema": ARSENAL_NATIVE_TRANSFER_SCHEMA,
        "version": ARSENAL_NATIVE_TRANSFER_VERSION,
        "suite": suite,
        "suite_sha256": canonical_sha256(suite),
        "contract": contract,
        "contract_sha256": canonical_sha256(contract),
        "results": results,
        "results_sha256": canonical_sha256(results),
        "execution_valid": True,
        "interpretation": {
            "identity_stable": True,
            "current_surface_exercised": {
                "observation_size": identity_start["shape"]["observation_size"],
                "action_logit_size": identity_start["shape"]["action_logit_size"],
                "action_head_sizes": identity_start["shape"]["action_head_sizes"],
            },
            "fidelity_verdict": (
                "read results.differential; transport validity is separate "
                "from measured outcome agreement"
            ),
        },
    }
    errors = arsenal_native_transfer_artifact_errors(artifact)
    if errors:
        raise RuntimeError(
            "native transfer artifact failed validation:\n- " + "\n- ".join(errors)
        )
    _write_json_atomic(args.output, artifact)
    print(f"publication_identity_sha256={contract['publication_identity_sha256']}")
    print(f"bridge_sha256={native_result['connected_bridge_sha256']}")
    print(f"jax_success={jax_result['success_count']}/{jax_result['episode_count']}")
    print(
        "native_success="
        f"{native_result['success_count']}/{native_result['episode_count']}"
    )
    print(
        "success_match="
        f"{differential['success_match_count']}/"
        f"{differential['episode_count']}"
    )
    print(f"artifact={args.output.resolve()}")


def _run_jax(
    policy_params,
    profiles: Sequence[str],
    seeds: Sequence[int],
    maximum_policy_steps: int,
) -> dict[str, Any]:
    labels = tuple(profile for profile in profiles for _ in seeds)
    row_seeds = tuple(seed for _ in profiles for seed in seeds)
    runtime = arsenal_runtime_config(hytale_0_5_7_loadouts(labels))
    params = default_combat_params(microticks=1, target_active=True)
    geometry = exact_empty_local_geometry(params)
    provider = partial(
        geometry_arsenal_world_capabilities,
        geometry=geometry,
        config=runtime,
    )
    action_source, initializer = recurrent_arsenal_policy_action_source(
        policy_params,
        compile=True,
    )
    evaluator = make_arsenal_evaluator(
        params,
        runtime,
        world_capability_provider=provider,
        max_policy_steps=maximum_policy_steps,
        compile=True,
        action_source=action_source,
        action_carry_initializer=initializer,
    )
    reset_keys = jnp.stack(
        tuple(jax.random.key(seed) for seed in row_seeds),
        axis=0,
    )
    result = evaluator(
        None,
        reset_keys,
        jax.random.key(DEFAULT_ROLLOUT_SEED),
    )
    jax.block_until_ready(result)
    statistics = evaluation_statistics(
        result,
        labels=labels,
        seeds=row_seeds,
    )
    statistics["success_count"] = sum(
        bool(row["success"]) for row in statistics["episodes"]
    )
    return statistics


def _run_native(
    policy_params,
    recurrent_size: int,
    profiles: Sequence[str],
    seeds: Sequence[int],
    maximum_policy_steps: int,
    *,
    host: str,
    port: int,
    expected_bridge: str,
    action_source: str,
    include_step_trace: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    choose_action = (
        _native_greedy_action(policy_params)
        if action_source == "policy"
        else _native_neutral_action()
    )
    expected_bridge = expected_bridge.upper()
    episodes: list[dict[str, Any]] = []
    accounting: dict[str, Any] | None = None
    invalid_observations = 0
    host_rejected_steps = 0
    first_info: Mapping[str, Any] | None = None
    last_info: Mapping[str, Any] | None = None
    first_captured_utc: str | None = None
    last_captured_utc: str | None = None
    aggregate_head_counts = [
        np.zeros(size, dtype=np.int64) for size in ARSENAL_POLICY_ACTION_HEAD_SIZES
    ]

    for profile in profiles:
        env = HytaleEnv(
            task="kill_trork",
            backend="native",
            host=host,
            port=port,
            world="flat",
            ticks_per_step=1,
            max_episode_steps=maximum_policy_steps,
            npc_role="Kweebec_Razorleaf",
            combat_target_active=True,
            learner_v3_profile=profile,
        )
        try:
            for seed in seeds:
                observation, info = env.reset(seed=seed)
                captured_utc = _utc_now()
                _require_native_info(info, expected_bridge)
                if first_info is None:
                    first_info = dict(info)
                    first_captured_utc = captured_utc
                last_info = dict(info)
                last_captured_utc = captured_utc
                invalid_observations += int(
                    not bool(info["learner_v3_observation_valid"])
                )
                recurrent = jnp.zeros(
                    (1, recurrent_size),
                    dtype=jnp.float32,
                )
                reward = 0.0
                terminated = False
                truncated = False
                step_rows: list[Mapping[str, Any]] = []
                step_trace: list[dict[str, Any]] = []
                episode_head_counts = [
                    np.zeros(size, dtype=np.int64)
                    for size in ARSENAL_POLICY_ACTION_HEAD_SIZES
                ]
                for step in range(1, maximum_policy_steps + 1):
                    mask = np.asarray(
                        info["learner_v3_action_mask"],
                        dtype=np.bool_,
                    )
                    pre_step_info = info
                    _require_each_head_available(
                        mask,
                        context=(
                            f"profile={profile} seed={seed} step={step} "
                            "observation_valid="
                            f"{info.get('learner_v3_observation_valid')} "
                            "failure_bits="
                            f"{info.get('learner_v3_failure_bits')} "
                            "mechanics_failure_bits="
                            f"{info.get('learner_v3_mechanics_failure_bits')} "
                            "arsenal_failure_bits="
                            f"{info.get('learner_v3_arsenal_failure_bits')} "
                            f"target_health={info.get('target_health')} "
                            f"agent_health={info.get('health')} "
                            f"active_statuses={_native_status_context(env)}"
                        ),
                    )
                    recurrent, factors = choose_action(
                        jnp.asarray(observation, dtype=jnp.float32)[None],
                        recurrent,
                        jnp.asarray(mask, dtype=jnp.bool_)[None],
                    )
                    action = np.asarray(factors[0], dtype=np.int32)
                    for head, value in enumerate(action):
                        episode_head_counts[head][int(value)] += 1
                        aggregate_head_counts[head][int(value)] += 1
                    observation, value, terminated, truncated, info = env.step(action)
                    captured_utc = _utc_now()
                    _require_native_info(info, expected_bridge)
                    last_info = dict(info)
                    last_captured_utc = captured_utc
                    reward += float(value)
                    invalid_observations += int(
                        not bool(info["learner_v3_observation_valid"])
                    )
                    host_rejected_steps += int(
                        not bool(info["host_policy_action_legal"])
                    )
                    step_rows.append(dict(info))
                    if include_step_trace:
                        step_trace.append(
                            _native_step_trace(
                                step,
                                mask,
                                action,
                                pre_step_info,
                                info,
                            )
                        )
                    if terminated or truncated:
                        break
                episode_accounting = summarize_native_action_accounting(step_rows)
                accounting = _merge_accounting(
                    accounting,
                    episode_accounting,
                )
                target_health = float(info.get("target_health", math.inf))
                target_present = bool(info.get("target_present", True))
                success = (not target_present) or target_health <= 0.0
                episode = {
                        "profile": profile,
                        "seed": int(seed),
                        "success": success,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "episode_length": step,
                        "episode_return": reward,
                        "target_health": target_health,
                        "target_present": target_present,
                        "episode_engine_ticks": info.get("episode_engine_ticks"),
                        "learner_v3_failure_bits": int(info["learner_v3_failure_bits"]),
                        "learner_v3_mechanics_failure_bits": int(
                            info["learner_v3_mechanics_failure_bits"]
                        ),
                        "learner_v3_arsenal_failure_bits": int(
                            info["learner_v3_arsenal_failure_bits"]
                        ),
                        "action_head_counts": {
                            name: counts.tolist()
                            for name, counts in zip(
                                ARSENAL_POLICY_ACTION_HEAD_NAMES,
                                episode_head_counts,
                                strict=True,
                            )
                        },
                    }
                if include_step_trace:
                    episode["step_trace"] = step_trace
                episodes.append(episode)
                print(
                    "native_progress "
                    f"profile={profile} seed={seed} steps={step} "
                    f"success={success}",
                    flush=True,
                )
        finally:
            env.close()

    if first_info is None or last_info is None:
        raise RuntimeError("native transfer produced no connected evidence")
    result = {
        "backend": "native_hytale_0_5_7",
        "connected_bridge_sha256": expected_bridge,
        "episode_count": len(episodes),
        "success_count": sum(bool(row["success"]) for row in episodes),
        "terminated_count": sum(bool(row["terminated"]) for row in episodes),
        "invalid_observation_count": invalid_observations,
        "host_rejected_step_count": host_rejected_steps,
        "episode_return": _number_summary(
            [float(row["episode_return"]) for row in episodes]
        ),
        "episode_length": _number_summary(
            [int(row["episode_length"]) for row in episodes]
        ),
        "action_head_counts": {
            name: counts.tolist()
            for name, counts in zip(
                ARSENAL_POLICY_ACTION_HEAD_NAMES,
                aggregate_head_counts,
                strict=True,
            )
        },
        "action_accounting": accounting,
        "episodes": episodes,
    }
    process = {
        "capture_event_at_start": "first_episode_reset",
        "captured_utc_at_start": first_captured_utc,
        "uptime_seconds_at_start": float(
            first_info["native_server_process_uptime_seconds"]
        ),
        "capture_event_at_end": "last_episode_step",
        "captured_utc_at_end": last_captured_utc,
        "uptime_seconds_at_end": float(
            last_info["native_server_process_uptime_seconds"]
        ),
        "in_timed_path": False,
    }
    return result, process


def _native_greedy_action(policy_params):
    @jax.jit
    def choose(observation, recurrent, action_mask):
        recurrent, logits, _ = apply_policy(
            policy_params,
            observation,
            recurrent,
            action_mask,
        )
        factors = greedy_arsenal_action_factors(logits)
        return recurrent, factors

    return choose


def _native_neutral_action():
    factors = neutral_arsenal_policy_action_factors(1)

    def choose(_observation, recurrent, _action_mask):
        return recurrent, factors

    return choose


def _require_current_checkpoint(config, metadata, identity) -> None:
    # Runtime capacity is an execution specialization of the training loadout,
    # not a global-catalog ABI.  Training records the exact specialized value
    # in the checkpoint; validating against the default full-catalog capacity
    # incorrectly rejects checkpoints trained on a bounded profile batch.
    runtime_capacity = metadata.get("arsenal_runtime_capacity")
    errors = combat_checkpoint_contract_errors(
        metadata,
        arsenal_runtime_capacity=runtime_capacity,
        policy_surface=ARSENAL_POLICY_SURFACE,
    )
    shape = identity["shape"]
    if config.observation_size != shape["observation_size"]:
        errors.append("checkpoint observation size differs from publication")
    if config.action_size != shape["action_logit_size"]:
        errors.append("checkpoint action size differs from publication")
    if tuple(config.action_head_sizes) != tuple(shape["action_head_sizes"]):
        errors.append("checkpoint action heads differ from publication")
    if config.action_transport != "factors":
        errors.append("checkpoint action transport is not factored")
    if errors:
        raise RuntimeError("checkpoint is not exact-current:\n- " + "\n- ".join(errors))


def _require_native_info(info: Mapping[str, Any], expected_bridge: str) -> None:
    actual = str(info.get("bridge_sha256", "")).upper()
    if actual != expected_bridge:
        raise RuntimeError(f"connected bridge moved: {actual!r} != {expected_bridge!r}")
    uptime = info.get("native_server_process_uptime_seconds")
    if (
        isinstance(uptime, bool)
        or not isinstance(uptime, (int, float))
        or not math.isfinite(float(uptime))
        or float(uptime) < 0.0
    ):
        raise RuntimeError("native server process uptime is unavailable")


def _native_status_context(env: HytaleEnv) -> list[list[int]]:
    """Return active semantic effect IDs for fail-closed diagnostics."""

    assembly = env._native_actor_assembly
    if assembly is None:
        return []
    active = np.asarray(assembly.evidence.status_mask, dtype=np.bool_)
    effect_id = np.asarray(
        assembly.evidence.status_i32[..., 0],
        dtype=np.int32,
    )
    return [
        effect_id[0, entity][active[0, entity]].tolist()
        for entity in range(active.shape[1])
    ]


_NATIVE_TRACE_INFO_FIELDS = (
    "learner_v3_observation_valid",
    "learner_v3_failure_bits",
    "learner_v3_mechanics_failure_bits",
    "learner_v3_arsenal_failure_bits",
    "host_policy_action_legal",
    "host_policy_action_reject_reasons",
    "native_guard_requested",
    "native_guard_accepted",
    "native_guard_started",
    "native_guard_active",
    "native_guard_finished",
    "native_guard_reject_reason",
    "native_dodge_requested",
    "native_dodge_accepted",
    "native_dodge_started",
    "native_dodge_finished",
    "native_dodge_failed",
    "native_dodge_reject_reason",
    "native_ability_requested",
    "native_ability_accepted",
    "native_ability_started",
    "native_ability_finished",
    "native_ability_failed",
    "native_ability_reject_reason",
)


def _native_step_trace(
    step: int,
    mask: np.ndarray,
    action: np.ndarray,
    pre_step_info: Mapping[str, Any],
    post_step_info: Mapping[str, Any],
) -> dict[str, Any]:
    boundaries = np.cumsum(ARSENAL_POLICY_ACTION_HEAD_SIZES)[:-1]
    head_masks = np.split(np.asarray(mask, dtype=np.bool_), boundaries)
    return {
        "step": int(step),
        "action_factors": np.asarray(action, dtype=np.int32).tolist(),
        "pre_step": {
            "head_legal_counts": {
                name: int(np.count_nonzero(head_mask))
                for name, head_mask in zip(
                    ARSENAL_POLICY_ACTION_HEAD_NAMES,
                    head_masks,
                    strict=True,
                )
            },
            **{
                name: _json_safe_trace_value(pre_step_info.get(name))
                for name in _NATIVE_TRACE_INFO_FIELDS[:4]
            },
        },
        "post_step": {
            name: _json_safe_trace_value(post_step_info.get(name))
            for name in _NATIVE_TRACE_INFO_FIELDS
        },
    }


def _json_safe_trace_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_safe_trace_value(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_trace_value(item) for item in value]
    return value


def _require_each_head_available(
    mask: np.ndarray,
    *,
    context: str = "",
) -> None:
    if mask.shape != (sum(ARSENAL_POLICY_ACTION_HEAD_SIZES),):
        raise RuntimeError(
            "native policy mask has the wrong shape"
            + (f": {context}" if context else "")
        )
    offset = 0
    for name, size in zip(
        ARSENAL_POLICY_ACTION_HEAD_NAMES,
        ARSENAL_POLICY_ACTION_HEAD_SIZES,
        strict=True,
    ):
        if not bool(np.any(mask[offset : offset + size])):
            raise RuntimeError(
                f"native policy head {name!r} has no legal value"
                + (f": {context}" if context else "")
            )
        offset += size


def _merge_accounting(
    left: dict[str, Any] | None,
    right: Mapping[str, Any],
) -> dict[str, Any]:
    if left is None:
        return json.loads(json.dumps(right))
    left["steps"] += int(right["steps"])
    for field in ("host_reject_reasons", "bridge_unsupported_actions"):
        merged = Counter(left[field])
        merged.update(right[field])
        left[field] = dict(sorted(merged.items()))
    for verb, row in right["native_lifecycle"].items():
        target = left["native_lifecycle"][verb]
        for event, count in row["events"].items():
            target["events"][event] += int(count)
        reasons = Counter(target["reject_reasons"])
        reasons.update(row["reject_reasons"])
        target["reject_reasons"] = dict(sorted(reasons.items()))
    return left


def _number_summary(values: Sequence[int | float]) -> dict[str, float]:
    numbers = tuple(float(value) for value in values)
    return {
        "mean": math.fsum(numbers) / len(numbers) if numbers else 0.0,
        "minimum": min(numbers) if numbers else 0.0,
        "maximum": max(numbers) if numbers else 0.0,
    }


def _git_sha(root: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "--short", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
