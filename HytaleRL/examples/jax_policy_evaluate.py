#!/usr/bin/env python3
"""Evaluate one JAX policy checkpoint through simulator or native Hytale."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.combat.skills import CombatSkillWrapper
from hytalegym.envs.hytale_env import HytaleEnv
from hytalegym.jax.combat import (
    PROFILE_NAMES,
    arsenal_policy_contract_sha256,
    arsenal_runtime_capacity,
    arsenal_runtime_config,
    combat_arsenal_contract_sha256,
    default_combat_params,
    hytale_0_5_7_loadouts,
)
from hytalegym.jax.evaluation.arsenal_benchmark import (
    exact_empty_local_geometry,
)
from hytalegym.jax.training.arsenal import (
    geometry_arsenal_world_capabilities,
)
from hytalegym.jax.training.arsenal_evaluation import (
    make_arsenal_evaluator,
)
from hytalegym.jax.training.baselines import (
    BASELINE_IDENTITIES,
    always_idle_arsenal_action_source,
    uniform_legal_arsenal_action_source,
)
from hytalegym.jax.training.checkpoint import ARSENAL_POLICY_SURFACE
from hytalegym.jax.training.evaluation import (
    EVALUATION_ARTIFACT_SCHEMA,
    EVALUATION_ARTIFACT_VERSION,
    NATIVE_SERVER_PROCESS_UPTIME_SOURCE,
    evaluation_artifact_contract_errors,
    evaluation_contract_sha256,
    evaluation_statistics,
    native_server_process_interval_errors,
    pure_jax_process_provenance,
)
from hytalegym.jax.training import (
    apply_policy,
    combat_checkpoint_contract_errors,
    current_combat_checkpoint_contract,
    initialize_policy,
    load_policy_checkpoint,
    sample_actions,
)
from hytalegym.rulesets import combat_ruleset_sha256
from hytalegym.wrappers.combat import ActiveCombatObsWrapper


_NATIVE_BACKENDS = frozenset(("native", "headless"))
_NATIVE_EVIDENCE_INFO_KEYS = (
    "native_evidence_bridge_sha256",
    "native_evidence_jar_sha256",
    "bridge_sha256",
)
_NATIVE_PROCESS_UPTIME_INFO_KEY = "native_server_process_uptime_seconds"


def _resolve_evaluation_backend(
    requested_backend: str | None,
    *,
    jax_arsenal: bool,
) -> str:
    """Keep pure-JAX Arsenal runs distinct from bridge-backed evaluation."""

    if jax_arsenal:
        if requested_backend not in (None, "jax"):
            raise ValueError(
                "--jax-arsenal is pure JAX; omit --backend or use "
                "--backend jax"
            )
        return "jax"
    if requested_backend == "jax":
        raise ValueError("--backend jax requires --jax-arsenal")
    return "native" if requested_backend is None else requested_backend


def _native_evidence_identity(
    backend: str,
    info: Mapping[str, Any],
) -> dict[str, str] | None:
    """Identify the bridge used by a native result, preferring wire evidence."""

    if backend not in _NATIVE_BACKENDS:
        return None
    for key in _NATIVE_EVIDENCE_INFO_KEYS:
        value = info.get(key)
        if not isinstance(value, str):
            continue
        normalized = value.strip().upper()
        if len(normalized) == 64:
            try:
                int(normalized, 16)
            except ValueError:
                continue
            return {
                "bridge_sha256": normalized,
                "source": f"backend_info.{key}",
            }
    return None


def _native_process_provenance(
    backend: str,
    info: Mapping[str, Any],
    *,
    captured_utc: str | None = None,
    capture_event: str = "backend_response",
) -> dict[str, Any] | None:
    """Record process age at capture time without inferring it client-side."""

    if backend not in _NATIVE_BACKENDS:
        return None
    value = info.get(_NATIVE_PROCESS_UPTIME_INFO_KEY)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(float(value))
        or value < 0
    ):
        return None
    return {
        "captured_utc": (
            datetime.now(timezone.utc).isoformat()
            if captured_utc is None
            else captured_utc
        ),
        "in_timed_path": False,
        "native_server_process_uptime_seconds": float(value),
        "source": NATIVE_SERVER_PROCESS_UPTIME_SOURCE,
        "capture_event": capture_event,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument(
        "--backend",
        choices=("jax", "simulator", "native", "headless"),
        default=None,
        help=(
            "execution backend; defaults to native for the melee transfer "
            "probe and jax for --jax-arsenal"
        ),
    )
    parser.add_argument("--world", choices=("flat", "hytale"), default="flat")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=570057)
    parser.add_argument(
        "--evaluation-seeds",
        type=int,
        nargs="+",
        help=(
            "explicit held-out Arsenal episode seeds; when omitted, derive "
            "--episodes seeds away from the training seed"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--microticks", type=int, choices=(1, 2, 3, 4))
    parser.add_argument(
        "--combat-target-active",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--allow-unverified-contract",
        action="store_true",
        help=(
            "allow a legacy checkpoint without the exact current environment "
            "fingerprint; never use this for a release claim"
        ),
    )
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--jax-arsenal",
        action="store_true",
        help=(
            "evaluate the current Arsenal policy in pure JAX against three "
            "baselines instead of running the native melee transfer probe"
        ),
    )
    parser.add_argument(
        "--arsenal-capability-provider",
        choices=("fail-closed", "exact-empty-local"),
        default="fail-closed",
        help=(
            "Arsenal world-query source. Geometry requires explicit opt-in; "
            "the permissive open-flat fixture is intentionally unavailable."
        ),
    )
    parser.add_argument(
        "--untrained-seed",
        type=int,
        default=15485863,
        help="initialization seed for the untrained architecture baseline",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the full contract-hashed report to this JSON path",
    )
    parser.add_argument(
        "--no-jit",
        action="store_true",
        help="disable JIT for small evaluator diagnostics",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    try:
        args.backend = _resolve_evaluation_backend(
            args.backend,
            jax_arsenal=args.jax_arsenal,
        )
    except ValueError as error:
        parser.error(str(error))

    if args.jax_arsenal:
        _evaluate_jax_arsenal(args)
        return

    policy_params, config, metadata = load_policy_checkpoint(args.checkpoint)
    expected_metadata = current_combat_checkpoint_contract()
    contract_errors = combat_checkpoint_contract_errors(metadata)
    if contract_errors and not args.allow_unverified_contract:
        raise SystemExit(
            "checkpoint does not identify the exact current environment; "
            "refusing an unverified transfer:\n- "
            + "\n- ".join(contract_errors)
        )
    if contract_errors:
        print(
            "WARNING: evaluating an unverified legacy ruleset contract:\n- "
            + "\n- ".join(contract_errors)
        )
    trained_microticks = int(metadata.get("microticks", 4))
    microticks = (
        trained_microticks if args.microticks is None else args.microticks
    )
    if microticks != trained_microticks:
        raise SystemExit(
            f"checkpoint was trained with {trained_microticks} microticks; "
            f"requested {microticks}"
        )
    trained_target_active = bool(
        metadata.get("combat_target_active", True)
    )
    target_active = (
        trained_target_active
        if args.combat_target_active is None
        else args.combat_target_active
    )
    if target_active != trained_target_active:
        raise SystemExit(
            "combat-target-active differs from the checkpoint schema"
        )

    base_environment = HytaleEnv(
        task="kill_trork",
        backend=args.backend,
        host=args.host,
        port=args.port,
        world=args.world,
        ticks_per_step=microticks,
        max_episode_steps=args.max_steps,
        npc_role=str(expected_metadata["agent_role"]),
        combat_target_active=target_active,
    )
    environment = ActiveCombatObsWrapper(
        CombatSkillWrapper(base_environment)
    )

    @jax.jit
    def choose_action(observation, recurrent_state, action_key):
        recurrent_state, logits, value = apply_policy(
            policy_params,
            observation,
            recurrent_state,
        )
        if args.stochastic:
            action, _ = sample_actions(action_key, logits)
        else:
            action = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        return recurrent_state, action, value

    key = jax.random.key(args.seed)
    episodes = []
    native_evidence = None
    native_evidence_at_end = None
    native_server_process_at_start = None
    native_server_process_at_end = None
    try:
        for episode in range(args.episodes):
            observation, info = environment.reset(
                seed=args.seed + episode
            )
            if episode == 0:
                native_evidence = _native_evidence_identity(
                    args.backend,
                    info,
                )
                native_server_process_at_start = _native_process_provenance(
                    args.backend,
                    info,
                    capture_event="first_episode_reset",
                )
                backend_errors = []
                if args.backend in ("native", "headless"):
                    if native_evidence is None:
                        backend_errors.append(
                            "bridge_sha256: connected backend did not report "
                            "its loaded bridge identity"
                        )
                    if native_server_process_at_start is None:
                        backend_errors.append(
                            "native_server_process_uptime_seconds: connected "
                            "backend did not report a nonnegative finite "
                            "process uptime"
                        )
                    actual_version = info.get("native_server_version")
                    if actual_version != expected_metadata["hytale_version"]:
                        backend_errors.append(
                            "native_server_version: "
                            f"backend={actual_version!r}, "
                            f"checkpoint={expected_metadata['hytale_version']!r}"
                        )
                    actual_agent = info.get("npc_role")
                else:
                    actual_version = info.get("combat_model_version")
                    if actual_version != expected_metadata[
                        "combat_model_version"
                    ]:
                        backend_errors.append(
                            "combat_model_version: "
                            f"backend={actual_version!r}, "
                            "checkpoint="
                            f"{expected_metadata['combat_model_version']!r}"
                        )
                    actual_agent = info.get("agent_profile")
                if actual_agent != expected_metadata["agent_role"]:
                    backend_errors.append(
                        "agent_role: "
                        f"backend={actual_agent!r}, "
                        f"checkpoint={expected_metadata['agent_role']!r}"
                    )
                actual_target = info.get("target_role")
                if actual_target != expected_metadata["target_role"]:
                    backend_errors.append(
                        "target_role: "
                        f"backend={actual_target!r}, "
                        f"checkpoint={expected_metadata['target_role']!r}"
                    )
                if backend_errors:
                    raise RuntimeError(
                        "connected backend does not match the checkpoint:\n- "
                        + "\n- ".join(backend_errors)
                    )
            recurrent_state = jnp.zeros(
                (1, config.recurrent_size),
                dtype=jnp.float32,
            )
            total_reward = 0.0
            terminated = False
            truncated = False
            last_skill = 0
            for step in range(1, args.max_steps + 1):
                key, action_key = jax.random.split(key)
                recurrent_state, action, _ = choose_action(
                    jnp.asarray(observation, dtype=jnp.float32)[None],
                    recurrent_state,
                    action_key,
                )
                last_skill = int(action[0])
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = environment.step(last_skill)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            episodes.append({
                "episode": episode,
                "reward": total_reward,
                "steps": step,
                "terminated": terminated,
                "truncated": truncated,
                "agent_health": float(observation[3] * 105.0),
                "target_health": info.get("target_health"),
                "target_present": info.get("target_present"),
                "last_skill": last_skill,
                "engine_ticks": info.get(
                    "episode_engine_ticks",
                    info.get("tick"),
                ),
            })
        if args.backend in _NATIVE_BACKENDS:
            _, end_info = environment.reset(seed=args.seed + args.episodes)
            native_evidence_at_end = _native_evidence_identity(
                args.backend,
                end_info,
            )
            native_server_process_at_end = _native_process_provenance(
                args.backend,
                end_info,
                capture_event="post_evaluation_reset",
            )
            end_errors = []
            if native_evidence_at_end is None:
                end_errors.append(
                    "bridge_sha256: connected backend did not report its "
                    "loaded bridge identity after evaluation"
                )
            elif (
                native_evidence is not None
                and native_evidence_at_end["bridge_sha256"]
                != native_evidence["bridge_sha256"]
            ):
                end_errors.append(
                    "bridge_sha256 changed between evaluation start and end"
                )
            end_errors.extend(
                native_server_process_interval_errors(
                    native_server_process_at_start,
                    native_server_process_at_end,
                    expected_in_timed_path=False,
                )
            )
            if end_errors:
                raise RuntimeError(
                    "connected backend provenance changed or is incomplete:\n- "
                    + "\n- ".join(end_errors)
                )
    finally:
        environment.close()

    successes = sum(
        episode["terminated"]
        and episode["target_health"] == 0.0
        for episode in episodes
    )
    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_metadata": metadata,
        "backend": args.backend,
        "native_evidence": native_evidence,
        "native_evidence_at_end": native_evidence_at_end,
        "native_server_process_at_start": native_server_process_at_start,
        "native_server_process_at_end": native_server_process_at_end,
        "world": args.world,
        "microticks": microticks,
        "target_active": target_active,
        "stochastic": args.stochastic,
        "successes": successes,
        "episode_count": len(episodes),
        "mean_reward": float(np.mean([
            episode["reward"] for episode in episodes
        ])),
        "episodes": episodes,
    }
    serialized_report = json.dumps(report, indent=2)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized_report + "\n", encoding="utf-8")
        print(serialized_report)
        print(f"artifact={output}")
    else:
        print(serialized_report)


def _evaluate_jax_arsenal(args: argparse.Namespace) -> None:
    if args.episodes < 1:
        raise SystemExit("--episodes must be positive")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be positive")

    checkpoint_path = Path(args.checkpoint).resolve()
    policy_params, config, metadata = load_policy_checkpoint(checkpoint_path)
    profiles = tuple(PROFILE_NAMES)
    evaluation_seeds = _arsenal_evaluation_seeds(args, metadata)
    profile_batch = tuple(
        profile
        for _seed in evaluation_seeds
        for profile in profiles
    )
    row_seeds = tuple(
        seed
        for seed in evaluation_seeds
        for _profile in profiles
    )
    runtime = arsenal_runtime_config(hytale_0_5_7_loadouts(profile_batch))
    runtime_capacity = arsenal_runtime_capacity(runtime.loadout)
    contract_errors = combat_checkpoint_contract_errors(
        metadata,
        arsenal_runtime_capacity=runtime_capacity,
        policy_surface=ARSENAL_POLICY_SURFACE,
    )
    if contract_errors and not args.allow_unverified_contract:
        raise SystemExit(
            "checkpoint does not identify the exact current Arsenal "
            "environment; refusing an unverified result:\n- "
            + "\n- ".join(contract_errors)
        )

    microticks = int(metadata.get("microticks", 1))
    params = default_combat_params(
        microticks=microticks,
        target_active=bool(metadata.get("combat_target_active", True)),
    )
    provider = None
    geometry = None
    capability_contract: dict[str, Any] = {
        "id": "fail_closed_unavailable_queries",
        "selection": "default",
        "open_flat_control": False,
        "policy_world_geometry_tokens": "unavailable_default",
    }
    if args.arsenal_capability_provider == "exact-empty-local":
        geometry = exact_empty_local_geometry(params)
        provider = partial(
            geometry_arsenal_world_capabilities,
            geometry=geometry,
            config=runtime,
        )
        capability_contract = {
            "id": "exact_complete_empty_local_geometry_v1",
            "selection": "explicit",
            "open_flat_control": False,
            "policy_world_geometry_tokens": "unavailable_default",
            "frame": (
                "complete all-air local geometry; certified predicates run "
                "normally, projectile contact is bound dynamically, and "
                "unavailable placement paths remain fail-closed"
            ),
        }

    compile_evaluator = not args.no_jit
    evaluator_kwargs = {
        "geometry_provider": geometry,
        "world_capability_provider": provider,
        "max_policy_steps": args.max_steps,
        "compile": compile_evaluator,
    }
    policy_evaluator = make_arsenal_evaluator(
        params,
        runtime,
        **evaluator_kwargs,
    )
    uniform_evaluator = make_arsenal_evaluator(
        params,
        runtime,
        action_source=uniform_legal_arsenal_action_source,
        **evaluator_kwargs,
    )
    idle_evaluator = make_arsenal_evaluator(
        params,
        runtime,
        action_source=always_idle_arsenal_action_source,
        **evaluator_kwargs,
    )
    untrained_params = initialize_policy(
        jax.random.key(args.untrained_seed),
        config,
    )
    reset_keys = jnp.stack(
        tuple(jax.random.key(seed) for seed in row_seeds),
        axis=0,
    )
    rollout_seed = args.seed ^ 0x2C9277B5
    rollout_key = jax.random.key(rollout_seed)

    evaluations = (
        ("trained_policy", policy_evaluator, policy_params),
        ("uniform_legal", uniform_evaluator, None),
        ("always_idle", idle_evaluator, None),
        ("untrained_init", policy_evaluator, untrained_params),
    )
    raw_results = {}
    statistics = {}
    for name, evaluator, evaluated_params in evaluations:
        print(f"evaluating={name}", flush=True)
        result = evaluator(evaluated_params, reset_keys, rollout_key)
        _block_tree(result)
        raw_results[name] = result
        statistics[name] = evaluation_statistics(
            result,
            labels=profile_batch,
            seeds=row_seeds,
        )

    comparisons = {
        baseline: _paired_comparison(
            raw_results["trained_policy"],
            raw_results[baseline],
        )
        for baseline in ("uniform_legal", "always_idle", "untrained_init")
    }
    reward_objective = _reward_objective(params)
    objective_decomposition = {
        name: _objective_decomposition(summary, reward_objective)
        for name, summary in statistics.items()
    }
    repository_root = Path(__file__).resolve().parents[1]
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    training_seed = metadata.get("seed")
    episode_seeds_held_out = (
        isinstance(training_seed, int)
        and training_seed not in evaluation_seeds
    )
    contract = {
        "schema": "hytalerl_evaluation_contract_v1",
        "version": 1,
        "environment": "jax_arsenal",
        "native_server_process": pure_jax_process_provenance(),
        "capability_provider": capability_contract,
        "evaluation_seeds": list(evaluation_seeds),
        "rollout_seed": rollout_seed,
        "episode_count": len(profile_batch),
        "episodes_per_profile": len(evaluation_seeds),
        "profiles": list(profiles),
        "profile_identity": {
            profile: int(weapon_id)
            for profile, weapon_id in zip(
                profiles,
                np.asarray(
                    hytale_0_5_7_loadouts(profiles).weapon_id
                )[:, 0],
                strict=True,
            )
        },
        "held_out": {
            "episode_seeds_disjoint_from_training_seed": (
                episode_seeds_held_out
            ),
            "profile_content": False,
            "geometry_content": False,
            "statement": (
                "episode PRNG seeds are disjoint from the checkpoint training "
                "seed; weapon profiles and geometry content are not held out"
            ),
        },
        "policy_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "metadata": metadata,
            "config": asdict(config),
            "current_contract_errors": contract_errors,
        },
        "baselines": {
            name: BASELINE_IDENTITIES[name]
            for name in (
                "uniform_legal",
                "always_idle",
                "untrained_init",
            )
        },
        "untrained_initialization_seed": args.untrained_seed,
        "reward_objective": reward_objective,
        "combat_ruleset_sha256": combat_ruleset_sha256().upper(),
        "arsenal_contract_sha256": (
            combat_arsenal_contract_sha256().upper()
        ),
        "arsenal_policy_contract_sha256": (
            arsenal_policy_contract_sha256().upper()
        ),
        "bridge_sha256": None,
        "repository": _repository_identity(repository_root),
        "execution": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "argv": [sys.executable, *sys.argv],
            "cwd": str(Path.cwd().resolve()),
            "jax_device": str(jax.devices()[0]),
            "jit": compile_evaluator,
        },
    }
    trained = statistics["trained_policy"]
    uniform = statistics["uniform_legal"]
    delta = comparisons["uniform_legal"]["episode_return_delta"]["mean"]
    artifact = {
        "schema": EVALUATION_ARTIFACT_SCHEMA,
        "version": EVALUATION_ARTIFACT_VERSION,
        "contract": contract,
        "contract_sha256": evaluation_contract_sha256(contract),
        "results": statistics,
        "comparisons": comparisons,
        "objective_decomposition": objective_decomposition,
        "interpretation": {
            "headline": (
                "trained policy mean return "
                f"{trained['episode_return']['mean']:.6f} versus "
                f"uniform-legal {uniform['episode_return']['mean']:.6f} "
                f"(paired delta {delta:+.6f}); success "
                f"{trained['success_rate']:.3f} versus "
                f"{uniform['success_rate']:.3f}"
            ),
            "covers": (
                "current Arsenal evaluator, all shipped profiles, identical "
                "episode/environment seeds, three reference controls, and "
                "unweighted objective-component decomposition"
            ),
            "limitation": (
                "profiles are training profiles, not held-out content; "
                + (
                    "the exact all-air local frame is a geometry-query "
                    "control, not natural Hytale terrain, and the policy "
                    "world-token row remains unavailable"
                    if args.arsenal_capability_provider
                    == "exact-empty-local"
                    else (
                        "world-dependent observations and abilities are "
                        "unavailable by construction under fail-closed mode; "
                        "the policy world-token row is also unavailable"
                    )
                )
                + (
                    "; checkpoint contract mismatches make this diagnostic "
                    "rather than release evidence"
                    if contract_errors
                    else ""
                )
            ),
        },
    }
    artifact_errors = evaluation_artifact_contract_errors(artifact)
    if artifact_errors:
        raise RuntimeError(
            "evaluation artifact failed its own contract:\n- "
            + "\n- ".join(artifact_errors)
        )
    output = args.output
    if output is None:
        output = (
            Path("artifacts")
            / "evaluation"
            / f"{checkpoint_path.stem}-arsenal-evaluation-v1.json"
        )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    print(f"artifact={output}")


def _arsenal_evaluation_seeds(
    args: argparse.Namespace,
    metadata: Mapping[str, Any],
) -> tuple[int, ...]:
    if args.evaluation_seeds:
        return tuple(args.evaluation_seeds)
    training_seed = metadata.get("seed")
    base = args.seed + 1_000_003
    if isinstance(training_seed, int) and base == training_seed:
        base += 1
    return tuple(base + 104729 * index for index in range(args.episodes))


def _paired_comparison(policy_result, baseline_result) -> dict[str, Any]:
    policy_return = np.asarray(policy_result.episode_return, dtype=np.float64)
    baseline_return = np.asarray(
        baseline_result.episode_return,
        dtype=np.float64,
    )
    delta = policy_return - baseline_return
    policy_success = np.asarray(policy_result.success, dtype=np.float64)
    baseline_success = np.asarray(
        baseline_result.success,
        dtype=np.float64,
    )
    return {
        "episode_return_delta": _numpy_spread(delta),
        "success_rate_delta": float(
            np.mean(policy_success - baseline_success)
        ),
        "paired_episode_count": int(delta.size),
    }


def _reward_objective(params) -> dict[str, float]:
    return {
        "target_damage_reward_scale": float(
            params.target_damage_reward_scale
        ),
        "agent_damage_reward_scale": float(
            params.agent_damage_reward_scale
        ),
        "completion_reward": float(params.completion_reward),
        "death_reward": float(params.death_reward),
    }


def _objective_decomposition(
    summary: Mapping[str, Any],
    objective: Mapping[str, float],
) -> dict[str, Any]:
    components = summary["reward_components"]
    weights = {
        "target_damage": objective["target_damage_reward_scale"],
        "agent_damage": objective["agent_damage_reward_scale"],
        "completion": objective["completion_reward"],
        "death": objective["death_reward"],
    }
    contributions = {
        name: float(components[name]["mean"]) * weight
        for name, weight in weights.items()
    }
    recomposed_mean = float(sum(contributions.values()))
    observed_mean = float(summary["episode_return"]["mean"])
    absolute_error = abs(recomposed_mean - observed_mean)
    if not np.isclose(
        recomposed_mean,
        observed_mean,
        atol=1.0e-4,
        rtol=1.0e-6,
    ):
        raise RuntimeError(
            "unweighted reward components do not recompose episode return: "
            f"components={recomposed_mean}, return={observed_mean}"
        )
    return {
        "mean_contribution": contributions,
        "recomposed_episode_return_mean": recomposed_mean,
        "observed_episode_return_mean": observed_mean,
        "absolute_recomposition_error": absolute_error,
    }


def _numpy_spread(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _repository_identity(repository_root: Path) -> dict[str, Any]:
    source_paths = (
        Path("examples/jax_policy_evaluate.py"),
        Path("hytalegym/hytalegym/jax/evaluation/arsenal_benchmark.py"),
        Path("hytalegym/hytalegym/jax/training/evaluation.py"),
        Path("hytalegym/hytalegym/jax/training/arsenal_evaluation.py"),
        Path("hytalegym/hytalegym/jax/training/baselines.py"),
    )
    status = _git_output(repository_root, "status", "--short")
    return {
        "commit": _git_output(repository_root, "rev-parse", "HEAD").strip(),
        "worktree_dirty": bool(status.strip()),
        "evaluation_source_sha256": {
            path.as_posix(): _file_sha256(repository_root / path)
            for path in source_paths
        },
    }


def _git_output(repository_root: Path, *arguments: str) -> str:
    process = subprocess.run(
        ("git", *arguments),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout


def _block_tree(tree) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


if __name__ == "__main__":
    main()
