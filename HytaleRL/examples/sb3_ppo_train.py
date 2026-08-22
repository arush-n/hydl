#!/usr/bin/env python3
"""High-throughput PPO training against one shared HytaleRL bridge.

Examples:
    python examples/sb3_ppo_train.py --task kill_trork --port 5557
    python examples/sb3_ppo_train.py --task navigate --num-envs 8 --port 5557

The default eight subprocesses create eight independent TCP clients and eight
isolated simulator environments inside one Java process. For ``kill_trork``,
compact backend-identical combat spaces are enabled by default.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import gymnasium as gym
import hytalegym  # noqa: F401  # registers environments
from hytalegym.wrappers import (
    ActiveCombatObsWrapper,
    CombatActionWrapper,
    CombatObsWrapper,
    CombatSkillWrapper,
    FlatActionWrapper,
    SimpleObsWrapper,
)


ENV_IDS = {
    "survive": "HytaleSurvive-v0",
    "mine_adamantite": "HytaleMineAdamantite-v0",
    "kill_trork": "HytaleKillTrork-v0",
    "build_house": "HytaleBuildHouse-v0",
    "navigate": "HytaleNavigate-v0",
    "base_builder": "HytaleBaseBuilder-v0",
}


def make_env_factory(
    *,
    env_id: str,
    task: str,
    host: str,
    port: int,
    backend: str,
    world: str,
    ticks_per_step: int,
    episode_steps: int | None,
    combat_target_active: bool,
    compact_combat: bool,
    active_combat_observation: bool,
    combat_skills: bool,
    seed: int,
    rank: int,
) -> Callable[[], gym.Env]:
    """Return a cloudpickle-safe constructor for an SB3 vector worker."""

    def initialize() -> gym.Env:
        make_kwargs = {
            "host": host,
            "port": port,
            "backend": backend,
            "world": world,
            "ticks_per_step": ticks_per_step,
            "combat_target_active": combat_target_active,
        }
        if episode_steps is not None:
            make_kwargs["max_episode_steps"] = episode_steps
        env = gym.make(env_id, **make_kwargs)
        if task == "kill_trork" and compact_combat:
            env = (
                CombatSkillWrapper(env)
                if combat_skills
                else CombatActionWrapper(env)
            )
            env = (
                ActiveCombatObsWrapper(env)
                if active_combat_observation
                else CombatObsWrapper(env)
            )
        else:
            env = FlatActionWrapper(env)
            env = SimpleObsWrapper(env)
        env.action_space.seed(seed + rank)
        return env

    return initialize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on HytaleRL")
    parser.add_argument("--task", choices=sorted(ENV_IDS), default="kill_trork")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument(
        "--backend", choices=("simulator", "native", "headless"), default="simulator"
    )
    parser.add_argument("--world", choices=("flat", "hytale"), default="flat")
    parser.add_argument("--ticks-per-step", type=int, default=4)
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=None,
        help="TimeLimit per episode; kill_trork defaults to 1000 for faster failure recycling",
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument(
        "--vec-mode",
        choices=("auto", "subproc", "dummy"),
        default="auto",
        help="auto uses subprocesses when num-envs > 1",
    )
    parser.add_argument("--seed", type=int, default=570057)
    parser.add_argument("--n-steps", type=int, default=512,
                        help="rollout steps per environment per PPO update")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--tensorboard-log", default=None)
    parser.add_argument(
        "--device",
        default="cpu",
        help="SB3 MLP PPO is normally faster on CPU; override for large custom policies",
    )
    parser.add_argument(
        "--compact-combat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the 5-branch action and 11-value observation for kill_trork",
    )
    parser.add_argument(
        "--combat-target-active",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="opt into Brawler pursuit; passive is the offense curriculum default",
    )
    parser.add_argument(
        "--active-combat-observation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="append active opponent combat/head telemetry (24 values total)",
    )
    parser.add_argument(
        "--combat-skills",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="replace low-level combat actions with nine deterministic melee skills",
    )
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num-envs must be at least 1")
    if args.ticks_per_step < 1:
        parser.error("--ticks-per-step must be at least 1")
    if args.episode_steps is not None and args.episode_steps < 1:
        parser.error("--episode-steps must be at least 1")
    if args.episode_steps is None and args.task == "kill_trork":
        args.episode_steps = 1_000
    if args.n_steps < 1 or args.batch_size < 1 or args.n_epochs < 1:
        parser.error("PPO rollout and batch settings must be positive")
    if (args.active_combat_observation or args.combat_skills) and not args.compact_combat:
        parser.error("active combat observations/skills require --compact-combat")
    return args


def main() -> None:
    args = parse_args()
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            SubprocVecEnv,
            VecMonitor,
        )
    except ImportError as error:
        raise SystemExit(
            "stable-baselines3 is required; install hytalegym with: "
            "python -m pip install -e \".\\hytalegym[dev]\""
        ) from error

    if args.backend != "simulator" and args.num_envs > 1:
        print(
            "Warning: native worlds run at real Hytale tick speed and are intended "
            "for evaluation; use the simulator for bulk rollout collection."
        )

    env_id = ENV_IDS[args.task]
    factories = [
        make_env_factory(
            env_id=env_id,
            task=args.task,
            host=args.host,
            port=args.port,
            backend=args.backend,
            world=args.world,
            ticks_per_step=args.ticks_per_step,
            episode_steps=args.episode_steps,
            combat_target_active=args.combat_target_active,
            compact_combat=args.compact_combat,
            active_combat_observation=args.active_combat_observation,
            combat_skills=args.combat_skills,
            seed=args.seed,
            rank=rank,
        )
        for rank in range(args.num_envs)
    ]

    use_subprocesses = args.vec_mode == "subproc" or (
        args.vec_mode == "auto" and args.num_envs > 1
    )
    vector_env = (
        SubprocVecEnv(factories, start_method="spawn")
        if use_subprocesses
        else DummyVecEnv(factories)
    )
    vector_env = VecMonitor(vector_env)
    vector_env.seed(args.seed)

    rollout_size = args.n_steps * args.num_envs
    if rollout_size % args.batch_size:
        print(
            f"Warning: rollout size {rollout_size} is not divisible by batch size "
            f"{args.batch_size}; PPO will use a smaller final minibatch."
        )

    save_path = args.save_path or f"hytale_ppo_{args.task}"
    print(
        f"Training {args.task} on {args.backend} with {args.num_envs} environments; "
        f"{args.ticks_per_step} engine ticks per transition, "
        f"{rollout_size} transitions per PPO rollout."
    )
    if args.task == "kill_trork" and args.compact_combat:
        action_description = (
            "Discrete(9) skills"
            if args.combat_skills
            else "MultiDiscrete(3,3,2,2,7)"
        )
        observation_size = 22 if args.active_combat_observation else 11
        print(
            "Using transfer-safe compact combat spaces: "
            f"{action_description} / {observation_size} floats."
        )
        if args.combat_target_active and not args.active_combat_observation:
            print(
                "Warning: active target attack phases are omitted; pass "
                "--active-combat-observation for dodge/punish training."
            )

    try:
        model = PPO(
            "MlpPolicy",
            vector_env,
            verbose=1,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log=args.tensorboard_log,
            device=args.device,
            seed=args.seed,
            policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
        )
        model.learn(total_timesteps=args.total_timesteps)
        model.save(save_path)
        print(f"Model saved to {save_path}.zip")
    finally:
        vector_env.close()


if __name__ == "__main__":
    main()
