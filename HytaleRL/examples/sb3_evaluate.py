#!/usr/bin/env python3
"""Evaluate one SB3 policy through the same wrappers on simulator or Hytale."""

from __future__ import annotations

import argparse
import json

import gymnasium as gym
import hytalegym  # noqa: F401
from hytalegym.wrappers import (
    ActiveCombatObsWrapper,
    CombatActionWrapper,
    CombatObsWrapper,
    CombatSkillWrapper,
    FlatActionWrapper,
    SimpleObsWrapper,
)

from sb3_ppo_train import ENV_IDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--task", choices=sorted(ENV_IDS), default="kill_trork")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--backend", choices=("simulator", "native", "headless"),
                        default="native")
    parser.add_argument("--world", choices=("flat", "hytale"), default="flat")
    parser.add_argument("--ticks-per-step", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=570057)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--compact-combat", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--combat-target-active", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--active-combat-observation",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--combat-skills",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise SystemExit("stable-baselines3 is required for policy evaluation") from error

    env = gym.make(
        ENV_IDS[args.task],
        host=args.host,
        port=args.port,
        backend=args.backend,
        world=args.world,
        ticks_per_step=args.ticks_per_step,
        combat_target_active=args.combat_target_active,
    )
    if args.task == "kill_trork" and args.compact_combat:
        env = (
            CombatSkillWrapper(env)
            if args.combat_skills
            else CombatActionWrapper(env)
        )
        env = (
            ActiveCombatObsWrapper(env)
            if args.active_combat_observation
            else CombatObsWrapper(env)
        )
    else:
        env = FlatActionWrapper(env)
        env = SimpleObsWrapper(env)

    model = PPO.load(args.model, device="cpu")
    episodes: list[dict[str, object]] = []
    try:
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            terminated = False
            truncated = False
            for step in range(1, args.max_steps + 1):
                action, _ = model.predict(
                    observation, deterministic=not args.stochastic
                )
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            episodes.append(
                {
                    "episode": episode,
                    "reward": total_reward,
                    "steps": step,
                    "terminated": terminated,
                    "truncated": truncated,
                    "agent_health": float(
                        observation[3] * CombatObsWrapper.AGENT_HEALTH_SCALE
                        if args.task == "kill_trork" and args.compact_combat
                        else observation[8]
                    ),
                    "target_health": info.get("target_health"),
                    "target_present": info.get("target_present"),
                    "engine_ticks": info.get("episode_engine_ticks", info.get("tick")),
                }
            )
    finally:
        env.close()

    successes = sum(
        item["target_health"] == 0.0 and item["terminated"] for item in episodes
    )
    report = {
        "backend": args.backend,
        "world": args.world,
        "target_active": args.combat_target_active,
        "successes": successes,
        "episode_count": len(episodes),
        "mean_reward": sum(float(item["reward"]) for item in episodes) / len(episodes),
        "episodes": episodes,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
