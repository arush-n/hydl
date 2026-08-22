"""Persist a native behavior prior after exact transplant to the live JAX ABI."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

# Imported for its import-time side effect: configure_training_runtime()
# must run before JAX is imported. Not unused -- do not delete.
from agents import training_runtime as _training_runtime  # noqa: F401

import jax
import jaxlib
import numpy as np

from agents.ppo.contracts import task_contract
from arena.tasks.games.world.generated import GENERATED
from arena.training import (
    CombatFundamentalsConfig,
    load_native_behavior_policy,
    native_arsenal_profile_basic_attack_binding,
    transplant_native_behavior_policy,
)
from arena.training.imitation.native_replay_cache import compile_or_load_native_replay
from hytalegym.jax.training.checkpoint import (
    load_policy_checkpoint,
    save_policy_checkpoint,
)


LIVE_POLICY_SCHEMA = "hytalerl_native_behavior_live_policy_v2"
LIVE_POLICY_SET_SCHEMA = "hytalerl_native_behavior_live_policy_set_v1"


def _combat_fundamentals_config(
    enabled: bool, path: Path | None
) -> CombatFundamentalsConfig | None:
    if path is None:
        return CombatFundamentalsConfig() if enabled else None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("combat reward config must be a JSON object")
    weights = data.get("weights", data)
    if not isinstance(weights, dict):
        raise ValueError("combat reward weights must be a JSON object")
    return CombatFundamentalsConfig(**weights)


def persist_live_policy(
    path: Path,
    params: Any,
    config: Any,
    *,
    scene: Any,
    task_manifest: dict[str, Any],
    native_checkpoint: Path,
    native_checkpoint_sha256: str,
    corpus: Any,
    style: tuple[str, str],
    transfer: dict[str, Any],
    seed: int,
    ppo_updates: int,
) -> dict[str, Any]:
    """Write and round-trip the zero-update live-policy initialization."""

    metadata = {
        "schema": LIVE_POLICY_SCHEMA,
        "checkpoint_role": "post_native_transplant_pre_training",
        "live_policy_compatible": True,
        "promotion_status": "not_assessed",
        "training_state_resumable": False,
        "updates": 0,
        "environment_steps": 0,
        "transfer_contract": dict(scene.policy_contract),
        "scene_contract_sha256": scene.contract_sha256,
        "task_contract_sha256": task_manifest["sha256"],
        "world": dict(scene.world_identity),
        "native_source": {
            "checkpoint": str(native_checkpoint),
            "checkpoint_sha256": native_checkpoint_sha256,
            "corpus_tensor_sha256": corpus.tensor_sha256,
            "corpus_source_report_sha256": corpus.source_report_sha256,
            "style": list(style),
        },
        "transplant": transfer,
        "initialization": {
            "seed": seed,
            "ppo_updates": ppo_updates,
            "key_protocol": "jax_split_seed_2_plus_updates_index_1_v1",
            "runtime": _initialization_runtime(),
        },
    }
    save_policy_checkpoint(path, params, config, metadata=metadata)
    loaded, loaded_config, loaded_metadata = load_policy_checkpoint(path)
    expected_metadata = json.loads(json.dumps(metadata))
    if (
        loaded_config != config
        or loaded_metadata != expected_metadata
        or not _trees_equal(params, loaded)
    ):
        raise RuntimeError("native live-policy checkpoint failed round-trip")
    return {
        "schema": LIVE_POLICY_SCHEMA,
        "path": str(path),
        "sha256": _sha256(path),
        "roundtrip_exact": True,
        "metadata": expected_metadata,
    }


def load_live_policy(
    path: Path,
    expected_params: Any,
    expected_config: Any,
    *,
    scene: Any,
    task_manifest: dict[str, Any],
    native_checkpoint_sha256: str,
    corpus: Any,
    style: tuple[str, str],
    transfer: dict[str, Any],
    seed: int,
    ppo_updates: int,
) -> tuple[Any, dict[str, Any]]:
    """Fail closed unless a saved live checkpoint is the exact transplant."""

    params, config, metadata = load_policy_checkpoint(path)
    expected = {
        "schema": LIVE_POLICY_SCHEMA,
        "checkpoint_role": "post_native_transplant_pre_training",
        "live_policy_compatible": True,
        "promotion_status": "not_assessed",
        "training_state_resumable": False,
        "updates": 0,
        "environment_steps": 0,
        "transfer_contract": dict(scene.policy_contract),
        "scene_contract_sha256": scene.contract_sha256,
        "task_contract_sha256": task_manifest["sha256"],
        "native_checkpoint_sha256": native_checkpoint_sha256,
        "corpus_tensor_sha256": corpus.tensor_sha256,
        "style": list(style),
        "transplant": json.loads(json.dumps(transfer)),
        "initialization": {
            "seed": seed,
            "ppo_updates": ppo_updates,
            "key_protocol": "jax_split_seed_2_plus_updates_index_1_v1",
            "runtime": _initialization_runtime(),
        },
    }
    actual = {
        "schema": metadata.get("schema"),
        "checkpoint_role": metadata.get("checkpoint_role"),
        "live_policy_compatible": metadata.get("live_policy_compatible"),
        "promotion_status": metadata.get("promotion_status"),
        "training_state_resumable": metadata.get("training_state_resumable"),
        "updates": metadata.get("updates"),
        "environment_steps": metadata.get("environment_steps"),
        "transfer_contract": metadata.get("transfer_contract"),
        "scene_contract_sha256": metadata.get("scene_contract_sha256"),
        "task_contract_sha256": metadata.get("task_contract_sha256"),
        "native_checkpoint_sha256": metadata.get("native_source", {}).get(
            "checkpoint_sha256"
        ),
        "corpus_tensor_sha256": metadata.get("native_source", {}).get(
            "corpus_tensor_sha256"
        ),
        "style": metadata.get("native_source", {}).get("style"),
        "transplant": metadata.get("transplant"),
        "initialization": metadata.get("initialization"),
    }
    if config != expected_config:
        raise ValueError("native live-policy PPO config does not match")
    if actual != expected:
        raise ValueError("native live-policy contract does not match")
    if not _trees_equal(params, expected_params):
        raise ValueError("native live-policy tensors differ from exact transplant")
    return params, {
        "schema": LIVE_POLICY_SCHEMA,
        "path": str(path),
        "sha256": _sha256(path),
        "exact_transplant": True,
    }


def persist_live_policy_set(
    *,
    native_checkpoint: Path,
    corpus_path: Path,
    replay_cache: Path,
    observation_view: str,
    output_dir: Path | None,
    output: Path | None,
    tasks: tuple[Any, ...],
    seeds: tuple[int, ...],
    style: tuple[str, str],
    agent_settings: Any,
    difficulty: str = "armed",
) -> dict[str, Any]:
    """Compile native data once, then publish every exact task/seed transplant."""

    if not tasks or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("tasks and unique seeds are required")
    if (output_dir is None) == (output is None):
        raise ValueError("pass exactly one of output or output_dir")
    if output is not None and len(tasks) * len(seeds) != 1:
        raise ValueError("output supports exactly one task/seed policy")

    from agents.ppo.worldgen.worldgen_il import WorldgenILPPOAgent

    compiled = compile_or_load_native_replay(
        corpus_path, replay_cache, observation_view
    )
    projected = replace(
        compiled.training,
        tensor_sha256=compiled.manifest["projected"]["tensor_sha256"],
    )
    policy = load_native_behavior_policy(native_checkpoint, projected)
    source_identity = replace(
        projected,
        source_report_sha256=compiled.manifest["source"]["source_report_sha256"],
        tensor_sha256=compiled.manifest["source"]["tensor_sha256"],
    )
    policies = []
    for task in tasks:
        for seed in seeds:
            settings = replace(agent_settings, seed=seed)
            agent = WorldgenILPPOAgent(settings, task=task, difficulty=difficulty)
            config = agent.ppo_config()
            key = jax.random.split(jax.random.key(seed), 2 + settings.ppo_updates)[1]
            initial = agent.handle.arena.training.initialize_ppo(
                key, agent.scene, config
            ).policy_params
            binding = native_arsenal_profile_basic_attack_binding(
                projected, style[0], agent.scene.expected_loadout
            )
            params, transfer = transplant_native_behavior_policy(
                policy,
                initial,
                config,
                agent.scene.combat_params,
                style,
                ability_binding=binding,
                target_ability_slot_contract_sha256=(
                    binding.target_ability_slot_contract_sha256
                ),
            )
            path = (
                output
                if output is not None
                else output_dir / task.name / f"seed-{seed}" / "live_policy.npz"
            )
            assert path is not None
            path.parent.mkdir(parents=True, exist_ok=True)
            receipt = persist_live_policy(
                path,
                params,
                config,
                scene=agent.scene,
                task_manifest=task_contract(task, difficulty),
                native_checkpoint=native_checkpoint,
                native_checkpoint_sha256=policy.checkpoint_sha256,
                corpus=source_identity,
                style=style,
                transfer=transfer,
                seed=seed,
                ppo_updates=settings.ppo_updates,
            )
            path.with_suffix(path.suffix + ".json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            policies.append(
                {
                    "task": task.name,
                    "seed": seed,
                    "path": receipt["path"],
                    "sha256": receipt["sha256"],
                    "scene_contract_sha256": agent.scene.contract_sha256,
                    "task_contract_sha256": task_contract(task, difficulty)["sha256"],
                }
            )
    report = {
        "schema": LIVE_POLICY_SET_SCHEMA,
        "native_checkpoint": str(native_checkpoint),
        "native_checkpoint_sha256": policy.checkpoint_sha256,
        "corpus": compiled.report(),
        "style": list(style),
        "difficulty": difficulty,
        "policies": policies,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "manifest.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--observation-view",
        choices=(
            "omniscient",
            "arsenal_shared",
            "arsenal_conditioned",
            "arsenal_visual_shared",
            "arsenal_visual_conditioned",
        ),
        default="arsenal_visual_conditioned",
    )
    parser.add_argument(
        "--replay-cache", type=Path, default=Path("~/.cache/hytalerl/native-replay")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--task", choices=tuple(task.name for task in GENERATED), default="plains_duel"
    )
    parser.add_argument(
        "--tasks", nargs="+", choices=tuple(task.name for task in GENERATED)
    )
    parser.add_argument("--difficulty", default="armed")
    parser.add_argument("--actor-role", default="Trork_Brawler")
    parser.add_argument("--opponent-role", default="Kweebec_Razorleaf")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--ppo-updates", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--num-minibatches", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--entropy-coefficient", type=float, default=1.0e-3)
    parser.add_argument("--encoder-size", type=int, default=64)
    parser.add_argument("--recurrent-size", type=int, default=64)
    parser.add_argument("--combat-fundamentals-reward", action="store_true")
    parser.add_argument("--combat-fundamentals-config", type=Path)
    args = parser.parse_args()

    from agents.ppo.worldgen.worldgen_il import WorldgenILPPOSettings

    task_by_name = {task.name: task for task in GENERATED}
    tasks = tuple(
        task_by_name[name]
        for name in (args.tasks if args.tasks is not None else [args.task])
    )
    seeds = tuple(args.seeds if args.seeds is not None else [args.seed])
    if any(seed is None for seed in seeds):
        parser.error("--seed or --seeds is required")
    settings = WorldgenILPPOSettings(
        batch=args.batch,
        ppo_updates=args.ppo_updates,
        rollout_steps=args.rollout_steps,
        ppo_update_epochs=args.update_epochs,
        ppo_num_minibatches=args.num_minibatches,
        ppo_learning_rate=args.learning_rate,
        ppo_entropy_coefficient=args.entropy_coefficient,
        encoder_size=args.encoder_size,
        recurrent_size=args.recurrent_size,
        combat_fundamentals=_combat_fundamentals_config(
            args.combat_fundamentals_reward,
            args.combat_fundamentals_config,
        ),
        seed=seeds[0],
    )
    style = (args.actor_role, args.opponent_role)
    report = persist_live_policy_set(
        native_checkpoint=args.native_checkpoint,
        corpus_path=args.corpus,
        replay_cache=args.replay_cache,
        observation_view=args.observation_view,
        output_dir=args.output_dir,
        output=args.output,
        tasks=tasks,
        seeds=seeds,
        style=style,
        agent_settings=settings,
        difficulty=args.difficulty,
    )
    print(json.dumps({"schema": report["schema"], "policies": report["policies"]}))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _initialization_runtime() -> dict[str, str]:
    device = jax.devices()[0]
    return {
        "schema": "hytalerl_jax_initialization_runtime_v1",
        "backend": jax.default_backend(),
        "platform": device.platform,
        "device_kind": device.device_kind,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "prng_implementation": str(jax.config.jax_default_prng_impl),
    }


def _trees_equal(left: Any, right: Any) -> bool:
    return jax.tree.structure(left) == jax.tree.structure(right) and all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


if __name__ == "__main__":
    main()
