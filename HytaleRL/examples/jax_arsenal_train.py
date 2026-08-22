"""Train and checkpoint recurrent PPO on every shipped Arsenal profile."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.compilation_cache import (
    configure_persistent_compilation_cache,
)
from hytalegym.jax.combat import (
    PROFILE_NAMES,
    arsenal_runtime_capacity,
    arsenal_runtime_config,
    combat_arsenal_contract_manifest,
    default_combat_params,
    hytale_0_5_7_loadouts,
)
from hytalegym.jax.combat.contracts.publication import (
    verify_combat_bridge_artifact_identity,
)
from hytalegym.jax.combat.opponents.runtime.policy import (
    first_legal_opponent_ability_policy_sha256,
)
from hytalegym.jax.training import (
    arsenal_ppo_config,
    combat_checkpoint_contract_errors,
    combat_checkpoint_metadata,
    initialize_training,
    load_policy_checkpoint,
    make_arsenal_ppo_environment,
    make_policy_update,
    make_rollout_collector,
    open_flat_arsenal_world_capabilities,
    require_arsenal_reward_liveness,
    save_policy_checkpoint,
)
from hytalegym.jax.training.checkpoint import ARSENAL_POLICY_SURFACE
from jax_arsenal_world_benchmark import (
    RegionActionSurfaceConfig,
    RegionNativeLightConfig,
    load_arsenal_region_fixture,
    load_distributed_arsenal_region_fixture,
)

_FLAT_REPORT_SCHEMA = "hytalerl_arsenal_ppo_run_v2"
_REGION_REPORT_SCHEMA = "hytalerl_arsenal_ppo_run_v3"
_DISTRIBUTED_REGION_REPORT_SCHEMA = "hytalerl_arsenal_ppo_run_v4"
_TRAINING_IDENTITY_SCHEMA = "hytalerl_training_identity_v1"
_BRIDGE_IDENTITY_FIELDS = (
    "expected_sha256",
    "built_sha256",
    "host_actor_evidence_contract_sha256",
    "built_actor_evidence_contract_sha256",
    "deployed_sha256",
    "deployed_actor_evidence_contract_sha256",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--encoder-size", type=int, default=64)
    parser.add_argument("--recurrent-size", type=int, default=64)
    parser.add_argument(
        "--allow-nonportable-jvm-widths",
        action="store_true",
        help=(
            "deprecated compatibility flag; the Java policy now supports "
            "independent encoder and recurrent widths"
        ),
    )
    parser.add_argument("--microticks", type=int, default=1)
    parser.add_argument(
        "--opponent-profile",
        choices=PROFILE_NAMES,
        help=(
            "optional authored Arsenal profile for every opponent; omitted "
            "uses the legacy ruleset opponent, while a supplied profile must "
            "accept an Arsenal ability in the pre-training liveness gate"
        ),
    )
    parser.add_argument(
        "--reward-liveness-steps",
        type=int,
        default=128,
        help=(
            "maximum decisions in each mirrored pre-training reward probe; "
            "all four reward components must fire before optimization"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(570057, 913771),
        help="First seed is the control; later seeds are independent replications.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "jax_arsenal_training",
    )
    parser.add_argument(
        "--deployed-bridge",
        type=Path,
        help=(
            "optional deployed bridge JAR path for cross-platform identity "
            "verification; native-transfer eligibility fails closed when the "
            "deployed artifact cannot be verified"
        ),
    )
    parser.add_argument(
        "--open-flat-control",
        action="store_true",
        help=(
            "Explicitly use the permissive geometry-free control fixture. "
            "Without this flag, unavailable world queries fail closed."
        ),
    )
    parser.add_argument("--region-manifest", type=Path)
    parser.add_argument("--traversal-manifest", type=Path)
    parser.add_argument("--expected-region-sha256")
    parser.add_argument("--expected-traversal-sha256")
    parser.add_argument("--region-native-light-coverage-manifest", type=Path)
    parser.add_argument("--expected-region-native-light-coverage-sha256")
    parser.add_argument("--region-selection-key", type=int)
    parser.add_argument("--region-assignment-key", type=int)
    parser.add_argument("--region-node-seed", type=int)
    parser.add_argument("--region-block-semantic-manifest", type=Path)
    parser.add_argument("--region-mutation-capacity", type=int)
    parser.add_argument("--block-action-maximum-distance", type=float)
    parser.add_argument("--block-action-view-sector-degrees", type=float)
    parser.add_argument(
        "--block-candidate-selection",
        choices=("native_camera", "dense_complete"),
        default="native_camera",
        help=(
            "native_camera matches player targeting; dense_complete is an "
            "explicit exhaustive mechanics/debug surface"
        ),
    )
    parser.add_argument("--native-item-interaction-recording", type=Path)
    parser.add_argument(
        "--native-crafting-catalog-recording",
        type=Path,
        help=(
            "current bridge-attributed 1,947-row crafting catalog; enables "
            "the complete lossless zero-time Fieldcraft subset when exact "
            "Region block execution is also enabled"
        ),
    )
    parser.add_argument(
        "--hytale-assets",
        type=Path,
        help="optional Assets.zip path for host-side block-action compilation",
    )
    parser.add_argument(
        "--enable-region-block-execution",
        action="store_true",
        help=(
            "enable source-checked exact Region block mutation; requires "
            "native item-interaction evidence"
        ),
    )
    parser.add_argument(
        "--region-held-item-asset-id",
        help=(
            "equip one native-evidence-backed item in actor hotbar slot zero "
            "for an explicit World-action training scenario"
        ),
    )
    parser.add_argument("--region-reset-pool-report", type=Path)
    parser.add_argument("--expected-region-reset-pool-sha256")
    parser.add_argument("--expected-region-reset-pool-file-sha256")
    parser.add_argument("--region-working-set-capacity", type=int)
    parser.add_argument(
        "--outcome-probe-checkpoint",
        type=Path,
        help=(
            "freeze an exact-contract policy checkpoint and use --updates as "
            "the number of read-only rollout windows; no optimizer step or "
            "derived checkpoint is produced"
        ),
    )
    arguments = parser.parse_args()
    return arguments


def _block(tree) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _profiles(batch_size: int) -> tuple[str, ...]:
    return tuple(
        PROFILE_NAMES[index % len(PROFILE_NAMES)] for index in range(batch_size)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _declared_native_evidence_jar_sha256() -> str:
    manifest = combat_arsenal_contract_manifest()
    return str(manifest["evidence"]["native_evidence_jar_sha256"]).upper()


def _training_opponent_metadata(profile: str | None) -> dict[str, object]:
    if profile is None:
        return {
            "mode": "legacy_ruleset_target_controller",
            "arsenal_profile": None,
            "ability_policy_sha256": None,
        }
    return {
        "mode": "first_legal_arsenal_ability",
        "arsenal_profile": profile,
        "ability_policy_sha256": first_legal_opponent_ability_policy_sha256(),
    }


def _training_checkpoint_metadata(
    *,
    opponent_profile: str | None,
    **arguments,
) -> dict[str, object]:
    metadata = combat_checkpoint_metadata(**arguments)
    metadata["training_opponent"] = _training_opponent_metadata(opponent_profile)
    return metadata


def _capture_training_identity(
    repository_root: Path,
    *,
    checkpoint_metadata: dict[str, object],
    deployed_bridge: Path | None,
) -> dict[str, object]:
    declared_bridge_sha256 = _declared_native_evidence_jar_sha256()
    try:
        measured_bridge = verify_combat_bridge_artifact_identity(
            repository_root,
            deployed_bridge=deployed_bridge,
            require_deployed=True,
        )
        if measured_bridge["expected_sha256"] != declared_bridge_sha256:
            raise RuntimeError(
                "checkpoint and publication bridge identities disagree: "
                f"checkpoint={declared_bridge_sha256} "
                f"publication={measured_bridge['expected_sha256']}"
            )
        bridge_artifact: dict[str, object] = {
            "verified": True,
            "identity": measured_bridge,
            "error": None,
        }
    except Exception as error:
        # Pure-JAX training remains useful without a deployed bridge. Preserve
        # the exact reason it cannot be promoted to native-transfer evidence.
        bridge_artifact = {
            "verified": False,
            "identity": None,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
    return {
        "schema": _TRAINING_IDENTITY_SCHEMA,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_metadata": dict(checkpoint_metadata),
        "declared_native_evidence_jar_sha256": declared_bridge_sha256,
        "bridge_artifact": bridge_artifact,
    }


def _bridge_identity_digest(
    identity: dict[str, object],
) -> dict[str, object] | None:
    artifact = identity["bridge_artifact"]
    if not isinstance(artifact, dict) or not artifact.get("verified", False):
        return None
    measured = artifact.get("identity")
    if not isinstance(measured, dict):
        return None
    return {name: measured.get(name) for name in _BRIDGE_IDENTITY_FIELDS}


def _training_identity_stability(
    start: dict[str, object],
    end: dict[str, object],
) -> dict[str, object]:
    reasons = []
    contract_stable = (
        start["checkpoint_metadata"] == end["checkpoint_metadata"]
        and start["declared_native_evidence_jar_sha256"]
        == end["declared_native_evidence_jar_sha256"]
    )
    if not contract_stable:
        reasons.append("checkpoint_contract_changed_during_run")

    start_bridge = _bridge_identity_digest(start)
    end_bridge = _bridge_identity_digest(end)
    bridge_stable = (
        start_bridge is not None
        and end_bridge is not None
        and start_bridge == end_bridge
    )
    if start_bridge is None:
        reasons.append("start_bridge_artifact_not_verified")
    if end_bridge is None:
        reasons.append("end_bridge_artifact_not_verified")
    if (
        start_bridge is not None
        and end_bridge is not None
        and start_bridge != end_bridge
    ):
        reasons.append("bridge_artifact_changed_during_run")

    return {
        "contract_stable": contract_stable,
        "bridge_artifact_stable": bridge_stable,
        "native_transfer_eligible": contract_stable and bridge_stable,
        "reasons": reasons,
    }


def _rollout_outcome_metrics(batch) -> dict[str, object]:
    rollout = batch.rollout
    counts = {
        "success": int(np.count_nonzero(rollout.completed_episode_success)),
        "death": int(np.count_nonzero(rollout.completed_episode_death)),
        "simultaneous": int(np.count_nonzero(rollout.completed_episode_simultaneous)),
        "other_terminal": int(np.count_nonzero(rollout.completed_episode_other)),
    }
    episodes_completed = sum(counts.values())
    completed_return = float(np.asarray(rollout.completed_episode_return).sum())
    completed_length = int(np.asarray(rollout.completed_episode_length).sum())
    return {
        "mean_rollout_reward": float(np.asarray(rollout.reward).mean()),
        "mean_episode_return": (
            completed_return / episodes_completed if episodes_completed else 0.0
        ),
        "mean_episode_length": (
            completed_length / episodes_completed if episodes_completed else 0.0
        ),
        "episodes_completed": episodes_completed,
        "episode_outcomes": counts,
    }


def _exact_region_checkpoint_errors(
    checkpoint_metadata: dict[str, object],
    region_fixture,
) -> list[str]:
    if region_fixture is None:
        return []
    training_world = checkpoint_metadata.get("training_world")
    if not isinstance(training_world, dict):
        return ["training_world: exact Region checkpoint provenance is missing"]
    errors = []
    expected_fields = {
        "world_capability_mode": _region_world_capability_mode(region_fixture),
        "physical_geometry_bound": True,
        "physical_geometry_type": type(region_fixture.geometry_provider).__name__,
    }
    for field, expected in expected_fields.items():
        actual = training_world.get(field)
        if actual != expected:
            errors.append(
                f"training_world.{field}: checkpoint={actual!r}, probe={expected!r}"
            )
    if getattr(region_fixture, "target_navigation_provider", None) is not None:
        expected_navigation = _region_target_navigation_metadata(region_fixture)
        if training_world.get("target_navigation") != expected_navigation:
            errors.append(
                "training_world.target_navigation: "
                f"checkpoint={training_world.get('target_navigation')!r}, "
                f"probe={expected_navigation!r}"
            )
    checkpoint_world = training_world.get("world")
    if not isinstance(checkpoint_world, dict):
        errors.append("training_world.world: exact Region identity is missing")
        return errors
    for field, expected in region_fixture.metadata.items():
        actual = checkpoint_world.get(field)
        if actual != expected:
            errors.append(
                f"training_world.world.{field}: checkpoint={actual!r}, "
                f"probe={expected!r}"
            )
    return errors


def _environment_world_arguments(
    *,
    region_fixture,
    open_flat_control: bool,
) -> dict[str, object]:
    if region_fixture is not None:
        arguments = {
            "geometry_provider": region_fixture.geometry_provider,
        }
        world_runtime_provider = getattr(
            region_fixture,
            "world_runtime_provider",
            None,
        )
        if world_runtime_provider is None:
            arguments.update(
                {
                    "world_capability_provider": (
                        region_fixture.world_capability_provider
                    ),
                    "world_token_provider": (region_fixture.world_token_provider),
                }
            )
        else:
            arguments["world_runtime_provider"] = world_runtime_provider
        target_navigation_provider = getattr(
            region_fixture,
            "target_navigation_provider",
            None,
        )
        if target_navigation_provider is not None:
            arguments["target_navigation_provider"] = target_navigation_provider
        reset_provider = getattr(region_fixture, "reset_provider", None)
        if reset_provider is not None:
            arguments["reset_provider"] = reset_provider
        for name in (
            "inventory_reset_provider",
            "action_surface_provider",
            "action_surface_executor",
            "action_surface_runtime_initializer",
            "explosion_candidate_provider",
        ):
            binding = getattr(region_fixture, name, None)
            if binding is not None:
                arguments[name] = binding
        return arguments
    return {
        "geometry_provider": None,
        "world_capability_provider": (
            open_flat_arsenal_world_capabilities if open_flat_control else None
        ),
        "world_token_provider": None,
    }


def _region_world_capability_mode(region_fixture) -> str:
    if getattr(region_fixture, "reset_provider", None) is not None:
        return "native_region_exact_geometry_with_filtered_resets_and_actor_tokens"
    return "native_region_exact_geometry_with_actor_tokens"


def _region_target_navigation_metadata(
    region_fixture,
) -> dict[str, object]:
    provider = getattr(region_fixture, "target_navigation_provider", None)
    contract_sha256 = getattr(
        region_fixture,
        "target_navigation_contract_sha256",
        None,
    )
    if provider is None or contract_sha256 is None:
        raise ValueError("exact Region fixture lacks its target-navigation binding")
    return {
        "provider": f"{provider.__module__}.{provider.__name__}",
        "contract_sha256": contract_sha256,
        "selected_transition_provenance": ("native_differential_exact_local_edge"),
        "route_choice_provenance": "surrogate_greedy_not_native_certified",
    }


def main() -> None:
    configure_persistent_compilation_cache(
        Path(__file__).resolve().parents[1] / ".codex-local" / "jax-cache"
    )
    arguments = _arguments()
    if arguments.updates < 1:
        raise ValueError("--updates must be positive")
    region_arguments = (
        arguments.region_manifest,
        arguments.traversal_manifest,
        arguments.expected_region_sha256,
        arguments.expected_traversal_sha256,
        arguments.region_selection_key,
        arguments.region_assignment_key,
    )
    if any(value is not None for value in region_arguments) and not all(
        value is not None for value in region_arguments
    ):
        raise ValueError(
            "all Region manifests, hashes, and selection keys are required together"
        )
    reset_pool_arguments = (
        arguments.region_reset_pool_report,
        arguments.expected_region_reset_pool_sha256,
        arguments.expected_region_reset_pool_file_sha256,
        arguments.region_working_set_capacity,
    )
    if any(value is not None for value in reset_pool_arguments) and not all(
        value is not None for value in reset_pool_arguments
    ):
        raise ValueError("all Region reset-pool arguments are required together")
    distributed_region = all(value is not None for value in reset_pool_arguments)
    if distributed_region and not all(value is not None for value in region_arguments):
        raise ValueError("Region reset pools require complete Region inputs")
    if distributed_region and arguments.region_node_seed is not None:
        raise ValueError(
            "--region-node-seed belongs only to the fixed-spawn Region mode"
        )
    if (
        arguments.region_manifest is not None
        and not distributed_region
        and arguments.region_node_seed is None
    ):
        raise ValueError("--region-node-seed is required for fixed-spawn Region mode")
    if arguments.open_flat_control and arguments.region_manifest is not None:
        raise ValueError(
            "--open-flat-control and exact Region inputs are mutually exclusive"
        )
    native_light_arguments = (
        arguments.region_native_light_coverage_manifest,
        arguments.expected_region_native_light_coverage_sha256,
    )
    if any(value is not None for value in native_light_arguments) and not all(
        value is not None for value in native_light_arguments
    ):
        raise ValueError(
            "the Region native-light coverage manifest and SHA-256 are "
            "required together"
        )
    if all(value is not None for value in native_light_arguments) and (
        arguments.region_manifest is None
    ):
        raise ValueError("Region native light requires exact Region inputs")
    action_surface_arguments = (
        arguments.region_block_semantic_manifest,
        arguments.region_mutation_capacity,
        arguments.block_action_maximum_distance,
        arguments.block_action_view_sector_degrees,
    )
    if any(value is not None for value in action_surface_arguments) and not all(
        value is not None for value in action_surface_arguments
    ):
        raise ValueError("all Region action-surface arguments are required together")
    if arguments.native_item_interaction_recording is not None and not all(
        value is not None for value in action_surface_arguments
    ):
        raise ValueError("native item evidence requires the Region action runtime")
    if (
        arguments.native_crafting_catalog_recording is not None
        and not all(value is not None for value in action_surface_arguments)
    ):
        raise ValueError(
            "native crafting evidence requires the Region action runtime"
        )
    if (
        arguments.enable_region_block_execution
        and arguments.native_item_interaction_recording is None
    ):
        raise ValueError(
            "Region block execution requires native item evidence"
        )
    if (
        arguments.region_held_item_asset_id is not None
        and not arguments.enable_region_block_execution
    ):
        raise ValueError(
            "a Region held item requires Region block execution"
        )
    if (
        arguments.native_crafting_catalog_recording is not None
        and not arguments.enable_region_block_execution
    ):
        raise ValueError(
            "Region fieldcraft execution requires Region block execution"
        )
    if (
        all(value is not None for value in action_surface_arguments)
        and arguments.region_manifest is None
    ):
        raise ValueError("Region action evidence requires exact Region inputs")
    action_surface_config = (
        None
        if not all(value is not None for value in action_surface_arguments)
        else RegionActionSurfaceConfig(
            block_semantic_manifest=(arguments.region_block_semantic_manifest),
            mutation_capacity=arguments.region_mutation_capacity,
            maximum_distance=arguments.block_action_maximum_distance,
            view_sector_full_angle_degrees=(arguments.block_action_view_sector_degrees),
            candidate_selection=arguments.block_candidate_selection,
            native_item_interaction_recording=(
                arguments.native_item_interaction_recording
            ),
            native_crafting_catalog_recording=(
                arguments.native_crafting_catalog_recording
            ),
            assets_path=arguments.hytale_assets,
            modification_allowed=arguments.enable_region_block_execution,
            held_item_asset_id=arguments.region_held_item_asset_id,
        )
    )
    native_light_config = (
        None
        if not all(value is not None for value in native_light_arguments)
        else RegionNativeLightConfig(
            coverage_manifest=(
                arguments.region_native_light_coverage_manifest
            ),
            expected_coverage_semantic_sha256=(
                arguments.expected_region_native_light_coverage_sha256
            ),
        )
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    profile_batch = _profiles(arguments.num_envs)
    target_profiles = (
        None
        if arguments.opponent_profile is None
        else (arguments.opponent_profile,) * arguments.num_envs
    )
    runtime = arsenal_runtime_config(
        hytale_0_5_7_loadouts(
            profile_batch,
            target_profiles=target_profiles,
        )
    )
    runtime_capacity = arsenal_runtime_capacity(runtime.loadout)
    validation_bits = np.asarray(runtime.validation_failure_bits)
    if np.any(validation_bits):
        raise RuntimeError(
            f"Arsenal loadout validation failed: {validation_bits.tolist()}"
        )
    params = default_combat_params(microticks=arguments.microticks)
    region_fixture = None
    if arguments.region_manifest is not None:
        if distributed_region:
            region_fixture = load_distributed_arsenal_region_fixture(
                region_manifest=arguments.region_manifest,
                traversal_manifest=arguments.traversal_manifest,
                reset_pool_report=arguments.region_reset_pool_report,
                expected_region_sha256=arguments.expected_region_sha256,
                expected_traversal_sha256=(arguments.expected_traversal_sha256),
                expected_reset_pool_sha256=(
                    arguments.expected_region_reset_pool_sha256
                ),
                expected_reset_pool_file_sha256=(
                    arguments.expected_region_reset_pool_file_sha256
                ),
                working_set_capacity=arguments.region_working_set_capacity,
                batch_size=arguments.num_envs,
                selection_key=arguments.region_selection_key,
                assignment_key=arguments.region_assignment_key,
                params=params,
                runtime=runtime,
                action_surface_config=action_surface_config,
                native_light_config=native_light_config,
            )
        else:
            region_fixture = load_arsenal_region_fixture(
                region_manifest=arguments.region_manifest,
                traversal_manifest=arguments.traversal_manifest,
                expected_region_sha256=arguments.expected_region_sha256,
                expected_traversal_sha256=(arguments.expected_traversal_sha256),
                batch_size=arguments.num_envs,
                selection_key=arguments.region_selection_key,
                assignment_key=arguments.region_assignment_key,
                node_seed=arguments.region_node_seed,
                params=params,
                runtime=runtime,
                action_surface_config=action_surface_config,
                native_light_config=native_light_config,
            )
        params = region_fixture.params
    environment = make_arsenal_ppo_environment(
        params,
        runtime,
        **_environment_world_arguments(
            region_fixture=region_fixture,
            open_flat_control=arguments.open_flat_control,
        ),
    )
    if arguments.reward_liveness_steps < 1:
        raise ValueError("--reward-liveness-steps must be positive")
    liveness_keys = jnp.stack(
        tuple(
            jax.random.fold_in(
                jax.random.key(arguments.seeds[index % len(arguments.seeds)]),
                index // len(arguments.seeds),
            )
            for index in range(arguments.num_envs)
        )
    )
    reward_liveness = require_arsenal_reward_liveness(
        environment,
        liveness_keys,
        maximum_steps=arguments.reward_liveness_steps,
        require_opponent_ability=arguments.opponent_profile is not None,
    )
    config = arsenal_ppo_config(
        num_envs=arguments.num_envs,
        rollout_steps=arguments.rollout_steps,
        update_epochs=arguments.update_epochs,
        num_minibatches=arguments.num_minibatches,
        encoder_size=arguments.encoder_size,
        recurrent_size=arguments.recurrent_size,
    )
    repository_root = Path(__file__).resolve().parents[1]
    collect = make_rollout_collector(config, environment=environment)
    probe_policy = None
    probe_source = None
    if arguments.outcome_probe_checkpoint is not None:
        checkpoint_path = arguments.outcome_probe_checkpoint.resolve()
        probe_policy, checkpoint_config, checkpoint_metadata = load_policy_checkpoint(
            checkpoint_path
        )
        contract_errors = combat_checkpoint_contract_errors(
            checkpoint_metadata,
            arsenal_runtime_capacity=runtime_capacity,
            policy_surface=ARSENAL_POLICY_SURFACE,
        )
        contract_errors.extend(
            _exact_region_checkpoint_errors(
                checkpoint_metadata,
                region_fixture,
            )
        )
        expected_opponent = _training_opponent_metadata(
            arguments.opponent_profile
        )
        if checkpoint_metadata.get("training_opponent") != expected_opponent:
            contract_errors.append(
                "training_opponent: "
                f"checkpoint={checkpoint_metadata.get('training_opponent')!r}, "
                f"probe={expected_opponent!r}"
            )
        for field in (
            "observation_size",
            "action_size",
            "action_head_sizes",
            "encoder_size",
            "recurrent_size",
        ):
            checkpoint_value = getattr(checkpoint_config, field)
            probe_value = getattr(config, field)
            if checkpoint_value != probe_value:
                contract_errors.append(
                    f"config.{field}: checkpoint={checkpoint_value!r}, "
                    f"probe={probe_value!r}"
                )
        if contract_errors:
            raise RuntimeError(
                "outcome-probe checkpoint does not match the current "
                "policy/runtime contract:\n- " + "\n- ".join(contract_errors)
            )
        probe_source = {
            "path": str(checkpoint_path),
            "sha256": _file_sha256(checkpoint_path),
            "metadata": checkpoint_metadata,
            "training_config": asdict(checkpoint_config),
            "current_contract_errors": contract_errors,
            "policy_frozen": True,
            "optimizer_steps": 0,
        }
    update = None if probe_policy is not None else make_policy_update(config)
    transitions_per_update = config.num_envs * config.rollout_steps
    report_checkpoint_metadata = _training_checkpoint_metadata(
        opponent_profile=arguments.opponent_profile,
        microticks=arguments.microticks,
        combat_target_active=True,
        seed=arguments.seeds[0],
        updates=arguments.updates,
        environment_steps=transitions_per_update * arguments.updates,
        arsenal_runtime_capacity=runtime_capacity,
        policy_surface=ARSENAL_POLICY_SURFACE,
    )
    report_identity_start = _capture_training_identity(
        repository_root,
        checkpoint_metadata=report_checkpoint_metadata,
        deployed_bridge=arguments.deployed_bridge,
    )

    report = {
        "schema": (
            (
                _DISTRIBUTED_REGION_REPORT_SCHEMA
                if getattr(region_fixture, "reset_provider", None) is not None
                else _REGION_REPORT_SCHEMA
            )
            if region_fixture is not None
            else _FLAT_REPORT_SCHEMA
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": (
            "frozen_checkpoint_outcome_probe"
            if probe_policy is not None
            else "ppo_training"
        ),
        "execution_backend": "pure_jax",
        "action_decoder": "stochastic_factored_categorical_sampling",
        "native_server_uptime_dependency": False,
        "config": asdict(config),
        "microticks": arguments.microticks,
        "world_capability_mode": (
            _region_world_capability_mode(region_fixture)
            if region_fixture is not None
            else (
                "explicit_open_flat_control"
                if arguments.open_flat_control
                else "fail_closed_unavailable_queries"
            )
        ),
        "profile_count": len(PROFILE_NAMES),
        "profiles": list(PROFILE_NAMES),
        "opponent_profile": arguments.opponent_profile,
        "runtime_capacity": runtime_capacity._asdict(),
        "reward_liveness": {
            "maximum_steps_per_arm": arguments.reward_liveness_steps,
            "counts": reward_liveness,
        },
        "combat_ruleset_sha256": report_checkpoint_metadata["combat_ruleset_sha256"],
        "arsenal_contract_sha256": report_checkpoint_metadata[
            "arsenal_contract_sha256"
        ],
        "policy_contract_sha256": report_checkpoint_metadata[
            "observation_contract_sha256"
        ],
        "training_identity_start": report_identity_start,
        "outcome_probe_checkpoint": probe_source,
        "seeds": [],
    }
    if region_fixture is not None:
        trainer_path = Path(__file__).resolve()
        fixture_path = trainer_path.with_name("jax_arsenal_world_benchmark.py")
        report.update(
            {
                "physical_geometry_bound": True,
                "physical_geometry_type": type(
                    region_fixture.geometry_provider
                ).__name__,
                "target_navigation": (
                    _region_target_navigation_metadata(region_fixture)
                ),
                "world": region_fixture.metadata,
                "source_identity": {
                    "trainer": {
                        "path": str(trainer_path),
                        "sha256": _file_sha256(trainer_path),
                    },
                    "region_fixture_builder": {
                        "path": str(fixture_path),
                        "sha256": _file_sha256(fixture_path),
                    },
                },
            }
        )
        if getattr(region_fixture, "reset_provider", None) is not None:
            pool_builder_path = trainer_path.with_name(
                "jax_arsenal_region_reset_pool.py"
            )
            artifact_builder_sha256 = region_fixture.metadata["reset_distribution"][
                "builder_sha256"
            ]
            current_builder_sha256 = _file_sha256(pool_builder_path)
            report["source_identity"]["region_reset_pool_builder"] = {
                "path": str(pool_builder_path),
                "sha256": artifact_builder_sha256,
                "current_source_sha256": current_builder_sha256,
                "matches_current_source": (
                    artifact_builder_sha256 == current_builder_sha256
                ),
            }
            report["source_identity"]["region_reset_pool_report"] = {
                "path": str(arguments.region_reset_pool_report.resolve()),
                "sha256": _file_sha256(arguments.region_reset_pool_report.resolve()),
            }

    compile_times = None
    for seed_index, seed in enumerate(arguments.seeds):
        transitions = transitions_per_update * arguments.updates
        seed_checkpoint_metadata = _training_checkpoint_metadata(
            opponent_profile=arguments.opponent_profile,
            microticks=arguments.microticks,
            combat_target_active=True,
            seed=seed,
            updates=arguments.updates,
            environment_steps=transitions,
            arsenal_runtime_capacity=runtime_capacity,
            policy_surface=ARSENAL_POLICY_SURFACE,
        )
        seed_identity_start = _capture_training_identity(
            repository_root,
            checkpoint_metadata=seed_checkpoint_metadata,
            deployed_bridge=arguments.deployed_bridge,
        )
        state = initialize_training(
            jax.random.key(seed),
            config,
            environment=environment,
        )
        if probe_policy is not None:
            state = state._replace(policy_params=probe_policy)
        if compile_times is None:
            compile_start = time.perf_counter()
            warm_state, warm_batch = collect(
                state,
                jax.random.key(seed ^ 0x2C9277B5),
            )
            _block((warm_state, warm_batch))
            rollout_compile_seconds = time.perf_counter() - compile_start
            optimizer_compile_seconds = None
            if update is not None:
                compile_start = time.perf_counter()
                warm_state, warm_metrics = update(
                    warm_state,
                    warm_batch,
                    jax.random.key(seed ^ 0x5A17E3D1),
                )
                _block((warm_state, warm_metrics))
                optimizer_compile_seconds = time.perf_counter() - compile_start
            compile_times = {
                "rollout_seconds": rollout_compile_seconds,
                "optimizer_seconds": optimizer_compile_seconds,
            }

        seed_report = {
            "seed": seed,
            "role": "control" if seed_index == 0 else "replication",
            "updates": [],
        }
        outcome_totals = {
            "success": 0,
            "death": 0,
            "simultaneous": 0,
            "other_terminal": 0,
        }
        rollout_seconds = 0.0
        optimizer_seconds = 0.0
        for update_index in range(arguments.updates):
            rollout_key = jax.random.key(seed + 2 * update_index + 1)
            optimizer_key = jax.random.key(seed + 2 * update_index + 2)
            started = time.perf_counter()
            state, batch = collect(state, rollout_key)
            _block((state, batch))
            elapsed_rollout = time.perf_counter() - started
            if update is None:
                metrics_report = _rollout_outcome_metrics(batch)
                elapsed_optimizer = 0.0
            else:
                started = time.perf_counter()
                state, metrics = update(state, batch, optimizer_key)
                _block((state, metrics))
                elapsed_optimizer = time.perf_counter() - started
                metrics_report = {
                    "mean_rollout_reward": float(metrics.mean_rollout_reward),
                    "mean_episode_return": float(metrics.mean_episode_return),
                    "mean_episode_length": float(metrics.mean_episode_length),
                    "episodes_completed": int(metrics.episodes_completed),
                    "episode_outcomes": {
                        "success": int(metrics.episode_successes),
                        "death": int(metrics.episode_deaths),
                        "simultaneous": int(metrics.episode_simultaneous),
                        "other_terminal": int(metrics.episode_other_terminal),
                    },
                    "total_loss": float(metrics.total_loss),
                    "entropy": float(metrics.entropy),
                }
            rollout_seconds += elapsed_rollout
            optimizer_seconds += elapsed_optimizer
            episode_outcomes = metrics_report["episode_outcomes"]
            classified_episodes = sum(episode_outcomes.values())
            episodes_completed = int(metrics_report["episodes_completed"])
            if classified_episodes != episodes_completed:
                raise RuntimeError(
                    "episode outcome taxonomy is not exhaustive: "
                    f"{classified_episodes} != {episodes_completed}"
                )
            for name, count in episode_outcomes.items():
                outcome_totals[name] += count
            update_report = {
                "index": update_index + 1,
                "mean_rollout_reward": metrics_report["mean_rollout_reward"],
                "mean_episode_return": metrics_report["mean_episode_return"],
                "mean_episode_length": metrics_report["mean_episode_length"],
                "episodes_completed": episodes_completed,
                "episode_outcomes": episode_outcomes,
                "total_loss": metrics_report.get("total_loss"),
                "entropy": metrics_report.get("entropy"),
                "rollout_seconds": elapsed_rollout,
                "optimizer_seconds": elapsed_optimizer,
            }
            seed_report["updates"].append(update_report)
            print(
                "progress="
                + json.dumps(
                    {
                        "seed": seed,
                        "role": seed_report["role"],
                        "environment_steps": (
                            transitions_per_update * (update_index + 1)
                        ),
                        **update_report,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        total_completed = sum(outcome_totals.values())
        seed_report["episode_outcomes"] = {
            "counts": outcome_totals,
            "rates": {
                name: (count / total_completed if total_completed else 0.0)
                for name, count in outcome_totals.items()
            },
            "episodes_completed": total_completed,
            "rate_denominator": "episodes_completed",
            "censoring": (
                "episodes still active after the final rollout are not classified"
            ),
            "taxonomy": {
                "success": "target_dead_and_agent_alive",
                "death": "agent_dead_and_target_alive",
                "simultaneous": "target_and_agent_dead",
                "other_terminal": "terminal_without_either_health_reaching_zero",
            },
        }
        seed_report["throughput"] = {
            "transitions": transitions,
            "rollout_transitions_per_second": transitions / rollout_seconds,
            "optimizer_transitions_per_second": (
                transitions / optimizer_seconds if optimizer_seconds else None
            ),
            "end_to_end_transitions_per_second": (
                transitions / (rollout_seconds + optimizer_seconds)
            ),
        }
        seed_identity_end = _capture_training_identity(
            repository_root,
            checkpoint_metadata=_training_checkpoint_metadata(
                opponent_profile=arguments.opponent_profile,
                microticks=arguments.microticks,
                combat_target_active=True,
                seed=seed,
                updates=arguments.updates,
                environment_steps=transitions,
                arsenal_runtime_capacity=runtime_capacity,
                policy_surface=ARSENAL_POLICY_SURFACE,
            ),
            deployed_bridge=arguments.deployed_bridge,
        )
        identity_stability = _training_identity_stability(
            seed_identity_start,
            seed_identity_end,
        )
        seed_report["training_identity"] = {
            "start": seed_identity_start,
            "end": seed_identity_end,
            "stability": identity_stability,
        }
        if probe_policy is None:
            checkpoint_path = arguments.output_dir / f"arsenal_seed_{seed}.npz"
            metadata = dict(seed_checkpoint_metadata)
            metadata["training_identity"] = seed_report["training_identity"]
            metadata["native_transfer_eligible"] = identity_stability[
                "native_transfer_eligible"
            ]
            if region_fixture is not None:
                metadata["training_world"] = {
                    "world_capability_mode": report["world_capability_mode"],
                    "physical_geometry_bound": report["physical_geometry_bound"],
                    "physical_geometry_type": report["physical_geometry_type"],
                    "target_navigation": report["target_navigation"],
                    "world": report["world"],
                    "source_identity": report["source_identity"],
                }
            save_policy_checkpoint(
                checkpoint_path,
                state.policy_params,
                config,
                metadata=metadata,
            )
            _, loaded_config, loaded_metadata = load_policy_checkpoint(checkpoint_path)
            contract_errors = combat_checkpoint_contract_errors(
                loaded_metadata,
                arsenal_runtime_capacity=runtime_capacity,
                policy_surface=ARSENAL_POLICY_SURFACE,
            )
            if loaded_config != config or (
                identity_stability["contract_stable"] and contract_errors
            ):
                raise RuntimeError(
                    "saved Arsenal checkpoint did not pass its current "
                    f"contract: {contract_errors}"
                )
            stale_metadata = dict(loaded_metadata)
            stale_metadata["arsenal_contract_sha256"] = "0" * 64
            stale_errors = combat_checkpoint_contract_errors(
                stale_metadata,
                arsenal_runtime_capacity=runtime_capacity,
                policy_surface=ARSENAL_POLICY_SURFACE,
            )
            if not stale_errors:
                raise RuntimeError("stale Arsenal checkpoint was not rejected")
            seed_report["checkpoint"] = {
                "path": str(checkpoint_path),
                "sha256": _file_sha256(checkpoint_path),
                "native_transfer_eligible": identity_stability[
                    "native_transfer_eligible"
                ],
                "current_contract_errors": contract_errors,
                "stale_contract_errors": stale_errors,
            }
        else:
            seed_report["checkpoint"] = {
                "saved": False,
                "reason": "read_only_frozen_policy_outcome_probe",
            }
        report["seeds"].append(seed_report)

    report["compile"] = compile_times
    report_identity_end = _capture_training_identity(
        repository_root,
        checkpoint_metadata=_training_checkpoint_metadata(
            opponent_profile=arguments.opponent_profile,
            microticks=arguments.microticks,
            combat_target_active=True,
            seed=arguments.seeds[0],
            updates=arguments.updates,
            environment_steps=transitions_per_update * arguments.updates,
            arsenal_runtime_capacity=runtime_capacity,
            policy_surface=ARSENAL_POLICY_SURFACE,
        ),
        deployed_bridge=arguments.deployed_bridge,
    )
    report["training_identity_end"] = report_identity_end
    report["training_identity_stability"] = _training_identity_stability(
        report_identity_start,
        report_identity_end,
    )
    report["native_transfer_eligible"] = report["training_identity_stability"][
        "native_transfer_eligible"
    ] and all(
        seed["training_identity"]["stability"]["native_transfer_eligible"]
        for seed in report["seeds"]
    )
    report_path = arguments.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
