"""Run flat-Arsenal PPO with a static bank of independently trained policies."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat import default_combat_params
from hytalegym.jax.combat.arsenal.environment import (
    open_flat_arsenal_world_capabilities,
)
from hytalegym.jax.combat.arsenal.profiles import (
    PROFILE_NAMES,
    hytale_0_5_7_entity_loadouts,
)
from hytalegym.jax.combat.arsenal.runtime import (
    arsenal_runtime_capacity,
    arsenal_runtime_config,
    reset_arsenal_batch,
)
from hytalegym.jax.combat.opponents.runtime.policy import (
    first_legal_opponent_ability_policy_contract,
    first_legal_opponent_ability_policy_sha256,
    first_legal_opponent_ability_slots,
    inert_opponent_ability_slots,
)
from hytalegym.jax.training.arsenal import arsenal_ppo_config
from hytalegym.jax.training.checkpoint import (
    ARSENAL_POLICY_SURFACE,
    combat_checkpoint_contract_errors,
    load_policy_checkpoint,
)
from hytalegym.jax.training.policy import initialize_policy

from .assignment import policy_actor_assignment
from .checkpoint import RuntimeTrainingEntitySpec
from .production import make_flat_combat_population_entrypoint


FLAT_POPULATION_RUN_SCHEMA = "hytalerl_flat_population_ppo_run_v1"


@dataclass(frozen=True)
class FlatEntitySpec:
    """One entity row repeated across every vectorized arena."""

    profile: str
    team_id: int
    controller: str
    health: float


@dataclass(frozen=True)
class FlatActorSpec:
    """One policy slot repeated across every vectorized arena."""

    actor_index: int
    policy_id: int
    trainable: bool


@dataclass(frozen=True)
class FlatPopulationLaunch:
    """Validated executable inputs for one flat population run."""

    entrypoint: object
    policy_bank: object
    entity_specs: tuple[FlatEntitySpec, ...]
    actor_specs: tuple[FlatActorSpec, ...]
    policy_training_contexts: tuple[dict[str, object], ...]
    policy_training_context_attestations: tuple[dict[str, object], ...]
    runtime_config_content_sha256: str
    policy_sources: tuple[dict[str, object], ...]
    opponent_policy: dict[str, object]
    runtime_capacity: object
    bank_size: int


def build_parser() -> argparse.ArgumentParser:
    """Return the stable command-line surface without importing examples."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--encoder-size", type=int, default=64)
    parser.add_argument("--recurrent-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--max-gradient-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=570057)
    parser.add_argument(
        "--entity",
        action="append",
        metavar="PROFILE:TEAM:POLICY|SCRIPTED:HEALTH",
        help=(
            "repeat for each entity row; defaults to a trainable Iron Sword "
            "actor against one scripted Iron Mace opponent"
        ),
    )
    parser.add_argument(
        "--actor",
        action="append",
        metavar="ENTITY_INDEX:POLICY_ID:TRAINABLE|FROZEN",
        help=(
            "repeat for each policy-owned row; ownership is replicated across "
            "all arenas while recurrent carry remains actor-major; distinct "
            "trainable policy IDs use population PPO, while shared-policy "
            "mirror self-play belongs to make_multi_actor_shared_policy_trainer"
        ),
    )
    parser.add_argument(
        "--bank-size",
        type=int,
        help="static K; defaults to one greater than the largest active policy ID",
    )
    parser.add_argument(
        "--policy-checkpoint",
        action="append",
        metavar="POLICY_ID=PATH",
        help=(
            "initialize one bank row from a current-contract portable policy "
            "checkpoint; unspecified rows use deterministic random initialization"
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume bank, independent optimizers, and counters from population v1",
    )
    parser.add_argument(
        "--selected-policy-id",
        type=int,
        help="also export this bank row as a portable policy checkpoint",
    )
    parser.add_argument(
        "--inert-opponent-diagnostic",
        action="store_true",
        help=(
            "explicitly disable scripted opponent abilities for diagnostics; "
            "production defaults to the deterministic first-legal provider"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "jax_arsenal_population",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate CUDA, contracts, assignment, runtime, and policy sources only",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse arguments for tests and ``python -m`` dispatch."""

    return build_parser().parse_args(argv)


def build_flat_population_launch(
    arguments: argparse.Namespace,
    *,
    compile: bool = True,
) -> FlatPopulationLaunch:
    """Build the executable flat population without starting optimization."""

    _require_gpu_backend()
    if arguments.updates < 0:
        raise ValueError("--updates cannot be negative")
    if arguments.updates == 0 and not arguments.dry_run:
        raise ValueError("--updates must be positive outside --dry-run")
    if arguments.resume is not None and arguments.policy_checkpoint:
        raise ValueError("--resume cannot be combined with --policy-checkpoint")
    entity_specs = tuple(
        _parse_entity_spec(value)
        for value in (
            arguments.entity
            or ("iron_sword:0:policy:105", "iron_mace:1:scripted:61")
        )
    )
    actor_specs = tuple(
        _parse_actor_spec(value)
        for value in (arguments.actor or ("0:0:trainable",))
    )
    _validate_launch_specs(entity_specs, actor_specs)
    maximum_policy_id = max(spec.policy_id for spec in actor_specs)
    bank_size = (
        maximum_policy_id + 1
        if arguments.bank_size is None
        else arguments.bank_size
    )
    if bank_size < 1:
        raise ValueError("--bank-size must be positive")
    if maximum_policy_id >= bank_size:
        raise ValueError("an actor policy ID is outside --bank-size")

    profile_row = tuple(spec.profile for spec in entity_specs)
    profile_matrix = tuple(profile_row for _ in range(arguments.num_envs))
    team_row = tuple(spec.team_id for spec in entity_specs)
    team_matrix = jnp.asarray(
        tuple(team_row for _ in range(arguments.num_envs)),
        dtype=jnp.int32,
    )
    controller_row = tuple(
        spec.controller == "scripted" for spec in entity_specs
    )
    controller_matrix = jnp.asarray(
        tuple(controller_row for _ in range(arguments.num_envs)),
        dtype=jnp.bool_,
    )
    runtime_config = arsenal_runtime_config(
        hytale_0_5_7_entity_loadouts(profile_matrix),
        entity_team_id=team_matrix,
        opponent_controller_mask=controller_matrix,
    )
    failure_bits = np.asarray(runtime_config.validation_failure_bits)
    if np.any(failure_bits):
        raise RuntimeError(
            "Arsenal loadout validation failed: " + repr(failure_bits.tolist())
        )
    params = default_combat_params(microticks=1, target_active=True)
    positions, health, yaw = _flat_reset_arrays(
        entity_specs,
        arguments.num_envs,
        target_distance=float(params.target_initial_distance),
    )

    def reset_provider(keys):
        state, _ = reset_arsenal_batch(
            keys,
            params,
            runtime_config,
            entity_position=positions,
            entity_health=health,
            entity_yaw=yaw,
        )
        return state

    def capability_provider(state):
        return open_flat_arsenal_world_capabilities(
            state,
            params,
            config=runtime_config,
        )

    if arguments.inert_opponent_diagnostic:
        opponent_ability_provider = inert_opponent_ability_slots
        opponent_policy = {
            "mode": "explicit_inert_diagnostic",
            "provider": "inert_opponent_ability_slots",
            "certification": "diagnostic_only_not_a_training_opponent",
        }
    else:
        opponent_ability_provider = first_legal_opponent_ability_slots
        opponent_policy = {
            "mode": "deterministic_first_legal_default",
            "provider": "first_legal_opponent_ability_slots",
            "contract": first_legal_opponent_ability_policy_contract(),
            "contract_sha256": first_legal_opponent_ability_policy_sha256(),
        }

    actor_index_row = tuple(spec.actor_index for spec in actor_specs)
    policy_id_row = tuple(spec.policy_id for spec in actor_specs)
    trainable_row = tuple(spec.trainable for spec in actor_specs)
    assignment = policy_actor_assignment(
        tuple(actor_index_row for _ in range(arguments.num_envs)),
        tuple(policy_id_row for _ in range(arguments.num_envs)),
        trainable=tuple(trainable_row for _ in range(arguments.num_envs)),
        entity_count=len(entity_specs),
    )
    ppo_config = arsenal_ppo_config(
        num_envs=arguments.num_envs,
        rollout_steps=arguments.rollout_steps,
        update_epochs=arguments.update_epochs,
        num_minibatches=arguments.num_minibatches,
        encoder_size=arguments.encoder_size,
        recurrent_size=arguments.recurrent_size,
        learning_rate=arguments.learning_rate,
        gamma=arguments.gamma,
        gae_lambda=arguments.gae_lambda,
        clip_epsilon=arguments.clip_epsilon,
        value_coefficient=arguments.value_coefficient,
        entropy_coefficient=arguments.entropy_coefficient,
        max_gradient_norm=arguments.max_gradient_norm,
    )
    entrypoint = make_flat_combat_population_entrypoint(
        params,
        runtime_config,
        assignment,
        ppo_config,
        reset_provider=reset_provider,
        capability_provider=capability_provider,
        opponent_ability_provider=opponent_ability_provider,
        training_entity_specs=tuple(
            RuntimeTrainingEntitySpec(
                profile=spec.profile,
                team_id=spec.team_id,
                controller=spec.controller,
            )
            for spec in entity_specs
        ),
        compile=compile,
    )
    capacity = arsenal_runtime_capacity(runtime_config.loadout)
    policy_bank, policy_sources = _initialize_policy_bank(
        bank_size,
        arguments.seed,
        ppo_config,
        arguments.policy_checkpoint or (),
        entity_count=len(entity_specs),
        runtime_capacity=capacity,
    )
    return FlatPopulationLaunch(
        entrypoint=entrypoint,
        policy_bank=policy_bank,
        entity_specs=entity_specs,
        actor_specs=actor_specs,
        policy_training_contexts=entrypoint.policy_training_contexts,
        policy_training_context_attestations=(
            entrypoint.policy_training_context_attestations
        ),
        runtime_config_content_sha256=(
            entrypoint.runtime_config_content_sha256
        ),
        policy_sources=policy_sources,
        opponent_policy=opponent_policy,
        runtime_capacity=capacity,
        bank_size=bank_size,
    )


def run_flat_population(
    arguments: argparse.Namespace,
    launch: FlatPopulationLaunch,
) -> dict[str, object]:
    """Execute the requested updates and save one population checkpoint."""

    entrypoint = launch.entrypoint
    if arguments.resume is None:
        state = entrypoint.initialize(
            launch.policy_bank,
            jax.random.fold_in(jax.random.key(arguments.seed), 0),
        )
        resume_source = None
    else:
        resumed = entrypoint.load_checkpoint(
            arguments.resume,
            jax.random.fold_in(jax.random.key(arguments.seed), 0),
            arsenal_runtime_capacity=launch.runtime_capacity,
        )
        if resumed.manifest["bank_size"] != launch.bank_size:
            raise ValueError(
                "resumed population bank size differs from --bank-size: "
                f"{resumed.manifest['bank_size']} != {launch.bank_size}"
            )
        state = resumed.state
        resume_source = {
            "path": str(arguments.resume),
            "manifest_sha256": _file_sha256(
                arguments.resume / "manifest.json"
            ),
            "population_content_sha256": resumed.manifest["arrays"][
                "content_sha256"
            ],
        }

    updates = []
    for update_index in range(arguments.updates):
        started = time.perf_counter()
        state, metrics = entrypoint.train_step(
            state,
            jax.random.fold_in(
                jax.random.key(arguments.seed),
                update_index + 1,
            ),
        )
        _block_until_ready((state, metrics))
        elapsed = time.perf_counter() - started
        update_report = _metrics_report(
            metrics,
            entrypoint.trainer.trainable_policy_ids,
            update_index=update_index + 1,
            elapsed_seconds=elapsed,
        )
        updates.append(update_report)
        print("progress=" + json.dumps(update_report, sort_keys=True), flush=True)

    selected_policy_id = arguments.selected_policy_id
    if selected_policy_id is None:
        selected_policy_id = entrypoint.trainer.trainable_policy_ids[0]
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "population-checkpoint-v1"
    entrypoint.save_checkpoint(
        checkpoint_path,
        state,
        selected_policy_id=selected_policy_id,
        arsenal_runtime_capacity=launch.runtime_capacity,
        metadata={
            "run_schema": FLAT_POPULATION_RUN_SCHEMA,
            "seed": arguments.seed,
            "requested_updates": arguments.updates,
            "policy_sources": list(launch.policy_sources),
            "opponent_policy": launch.opponent_policy,
            "resume_source": resume_source,
        },
    )
    manifest = json.loads(
        (checkpoint_path / "manifest.json").read_text(encoding="utf-8")
    )
    report = _launch_report(arguments, launch)
    report.update(
        {
            "mode": "ppo_training",
            "resume_source": resume_source,
            "updates": updates,
            "checkpoint": {
                "path": str(checkpoint_path),
                "manifest_sha256": _file_sha256(
                    checkpoint_path / "manifest.json"
                ),
                "population_archive_sha256": manifest["arrays"]["sha256"],
                "population_content_sha256": manifest["arrays"][
                    "content_sha256"
                ],
                "selected_policy_export": manifest["selected_policy_export"],
            },
        }
    )
    report_path = output / "run.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Executable module entry point."""

    arguments = parse_args(argv)
    launch = build_flat_population_launch(arguments)
    if arguments.dry_run:
        report = _launch_report(arguments, launch)
        report["mode"] = "dry_run"
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 0
    run_flat_population(arguments, launch)
    return 0


def _initialize_policy_bank(
    bank_size: int,
    seed: int,
    config,
    checkpoint_specs: Sequence[str],
    *,
    entity_count: int,
    runtime_capacity,
):
    checkpoint_paths: dict[int, Path] = {}
    for value in checkpoint_specs:
        if "=" not in value:
            raise ValueError("--policy-checkpoint must be POLICY_ID=PATH")
        raw_policy_id, raw_path = value.split("=", 1)
        try:
            policy_id = int(raw_policy_id)
        except ValueError as error:
            raise ValueError("policy checkpoint ID must be an integer") from error
        if policy_id < 0 or policy_id >= bank_size:
            raise ValueError("policy checkpoint ID is outside --bank-size")
        if policy_id in checkpoint_paths:
            raise ValueError("duplicate --policy-checkpoint policy ID")
        checkpoint_paths[policy_id] = Path(raw_path)

    policies = []
    sources = []
    root_key = jax.random.key(seed)
    for policy_id in range(bank_size):
        path = checkpoint_paths.get(policy_id)
        if path is None:
            policy = initialize_policy(
                jax.random.fold_in(root_key, policy_id),
                config,
            )
            source = {
                "policy_id": policy_id,
                "kind": "random",
                "base_seed": seed,
                "jax_fold_in": policy_id,
            }
        else:
            policy, source_config, metadata = load_policy_checkpoint(path)
            errors = combat_checkpoint_contract_errors(
                metadata,
                entity_count=entity_count,
                arsenal_runtime_capacity=runtime_capacity,
                policy_surface=ARSENAL_POLICY_SURFACE,
            )
            errors.extend(_policy_architecture_errors(source_config, config))
            if errors:
                raise ValueError(
                    f"policy checkpoint {policy_id} is incompatible:\n- "
                    + "\n- ".join(errors)
                )
            source = {
                "policy_id": policy_id,
                "kind": "portable_checkpoint",
                "path": str(path),
                "sha256": _file_sha256(path),
                "source_metadata": metadata,
            }
        policies.append(policy)
        sources.append(source)
    bank = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *policies)
    return bank, tuple(sources)


def _policy_architecture_errors(source, target) -> list[str]:
    errors = []
    for name in (
        "observation_size",
        "action_size",
        "action_head_sizes",
        "action_transport",
        "action_distribution",
        "encoder_size",
        "recurrent_size",
    ):
        if getattr(source, name) != getattr(target, name):
            errors.append(
                f"config.{name}: checkpoint={getattr(source, name)!r}, "
                f"run={getattr(target, name)!r}"
            )
    return errors


def _parse_entity_spec(value: str) -> FlatEntitySpec:
    parts = value.split(":")
    if len(parts) != 4:
        raise ValueError(
            "--entity must be PROFILE:TEAM:POLICY|SCRIPTED:HEALTH"
        )
    profile, raw_team, controller, raw_health = parts
    if profile not in PROFILE_NAMES:
        raise ValueError(f"unknown Arsenal profile {profile!r}")
    if controller not in {"policy", "scripted"}:
        raise ValueError("entity controller must be policy or scripted")
    try:
        team_id = int(raw_team)
        health = float(raw_health)
    except ValueError as error:
        raise ValueError("entity team and health must be numeric") from error
    if not math.isfinite(health) or health <= 0.0:
        raise ValueError("entity health must be finite and positive")
    return FlatEntitySpec(profile, team_id, controller, health)


def _parse_actor_spec(value: str) -> FlatActorSpec:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(
            "--actor must be ENTITY_INDEX:POLICY_ID:TRAINABLE|FROZEN"
        )
    raw_actor, raw_policy, mode = parts
    if mode not in {"trainable", "frozen"}:
        raise ValueError("actor mode must be trainable or frozen")
    try:
        actor_index = int(raw_actor)
        policy_id = int(raw_policy)
    except ValueError as error:
        raise ValueError("actor entity and policy IDs must be integers") from error
    return FlatActorSpec(actor_index, policy_id, mode == "trainable")


def _validate_launch_specs(entity_specs, actor_specs) -> None:
    if len(entity_specs) < 2:
        raise ValueError("flat combat requires at least two --entity rows")
    if not actor_specs:
        raise ValueError("at least one --actor row is required")
    actor_indices = tuple(spec.actor_index for spec in actor_specs)
    if len(set(actor_indices)) != len(actor_indices):
        raise ValueError("an entity cannot be owned by two actor slots")
    if any(index < 0 or index >= len(entity_specs) for index in actor_indices):
        raise ValueError("actor entity index is outside the entity axis")
    policy_entities = {
        index
        for index, spec in enumerate(entity_specs)
        if spec.controller == "policy"
    }
    if set(actor_indices) != policy_entities:
        raise ValueError(
            "policy entity rows and --actor ownership must match exactly"
        )
    if entity_specs[0].controller == "scripted":
        raise ValueError("entity zero cannot use the scripted opponent controller")
    if not any(spec.trainable for spec in actor_specs):
        raise ValueError("at least one actor policy must be trainable")
    trainable_ids = [spec.policy_id for spec in actor_specs if spec.trainable]
    if len(set(trainable_ids)) != len(trainable_ids):
        raise ValueError(
            "flat population v1 permits one actor slot per trainable policy "
            "in each arena; shared-policy mirror self-play uses "
            "make_multi_actor_shared_policy_trainer"
        )
    frozen_ids = {spec.policy_id for spec in actor_specs if not spec.trainable}
    shared_trainability = frozen_ids.intersection(trainable_ids)
    if shared_trainability:
        raise ValueError(
            "a frozen actor cannot share a trainable policy ID; use a distinct "
            "bank row so frozen policy weights remain byte-identical"
        )
    if any(spec.policy_id < 0 for spec in actor_specs):
        raise ValueError("actor policy IDs must be non-negative")


def _flat_reset_arrays(entity_specs, num_envs, *, target_distance: float):
    if num_envs < 1:
        raise ValueError("--num-envs must be positive")
    entity_count = len(entity_specs)
    row = np.zeros((entity_count, 3), dtype=np.float32)
    yaw = np.zeros((entity_count,), dtype=np.float32)
    if entity_count >= 2:
        row[1] = (0.0, 0.0, -target_distance)
        yaw[1] = 180.0
    for entity_id in range(2, entity_count):
        angle = 2.0 * math.pi * (entity_id - 1) / max(entity_count - 1, 1)
        row[entity_id] = (
            target_distance * math.sin(angle),
            0.0,
            -target_distance * math.cos(angle),
        )
        yaw[entity_id] = math.degrees(angle) + 180.0
    health_row = np.asarray(
        tuple(spec.health for spec in entity_specs),
        dtype=np.float32,
    )
    return (
        jnp.asarray(np.broadcast_to(row, (num_envs,) + row.shape)),
        jnp.asarray(
            np.broadcast_to(health_row, (num_envs, entity_count))
        ),
        jnp.asarray(np.broadcast_to(yaw, (num_envs, entity_count))),
    )


def _metrics_report(metrics, policy_ids, *, update_index, elapsed_seconds):
    result = {
        "index": update_index,
        "elapsed_seconds": elapsed_seconds,
        "policies": [],
    }
    loss_fields = metrics.loss._fields
    for index, policy_id in enumerate(policy_ids):
        loss = {
            field: float(getattr(metrics.loss, field)[index])
            for field in loss_fields
        }
        result["policies"].append(
            {
                "policy_id": policy_id,
                "update_applied": bool(metrics.update_applied[index]),
                "invalid_trainable_steps": int(
                    metrics.invalid_trainable_steps[index]
                ),
                "episodes_completed": int(metrics.episodes_completed[index]),
                "episode_successes": int(metrics.episode_successes[index]),
                "episode_deaths": int(metrics.episode_deaths[index]),
                "episode_simultaneous": int(
                    metrics.episode_simultaneous[index]
                ),
                "episode_other_terminal": int(
                    metrics.episode_other_terminal[index]
                ),
                "mean_episode_return": float(
                    metrics.mean_episode_return[index]
                ),
                "mean_episode_length": float(
                    metrics.mean_episode_length[index]
                ),
                "update_count": int(metrics.update_count[index]),
                "total_trainable_actor_steps": int(
                    metrics.total_trainable_actor_steps[index]
                ),
                "loss": loss,
            }
        )
    return result


def _launch_report(arguments, launch) -> dict[str, object]:
    return {
        "schema": FLAT_POPULATION_RUN_SCHEMA,
        "backend": jax.default_backend(),
        "world_capability_mode": "explicit_open_flat_control",
        "seed": arguments.seed,
        "requested_updates": arguments.updates,
        "config": asdict(launch.entrypoint.ppo_config),
        "entity_specs": [asdict(spec) for spec in launch.entity_specs],
        "actor_specs": [asdict(spec) for spec in launch.actor_specs],
        "policy_training_contexts": list(launch.policy_training_contexts),
        "policy_training_context_attestations": list(
            launch.policy_training_context_attestations
        ),
        "runtime_config_content_sha256": (
            launch.runtime_config_content_sha256
        ),
        "bank_size": launch.bank_size,
        "trainable_policy_ids": list(
            launch.entrypoint.trainer.trainable_policy_ids
        ),
        "trainable_actor_rows": [
            [list(row) for row in rows]
            for rows in launch.entrypoint.trainer.trainable_actor_rows
        ],
        "policy_sources": list(launch.policy_sources),
        "opponent_policy": launch.opponent_policy,
        "runtime_capacity": launch.runtime_capacity._asdict(),
        "single_actor_compatibility": (
            launch.entrypoint.assignment.actor_index.shape[1] == 1
            and bool(
                np.all(
                    np.asarray(launch.entrypoint.assignment.actor_index) == 0
                )
            )
        ),
    }


def _require_gpu_backend() -> None:
    backend = jax.default_backend()
    if backend != "gpu":
        raise RuntimeError(
            f"flat population training requires JAX gpu backend, got {backend!r}"
        )


def _block_until_ready(tree) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


__all__ = [
    "FLAT_POPULATION_RUN_SCHEMA",
    "FlatActorSpec",
    "FlatEntitySpec",
    "FlatPopulationLaunch",
    "build_flat_population_launch",
    "build_parser",
    "main",
    "parse_args",
    "run_flat_population",
]
