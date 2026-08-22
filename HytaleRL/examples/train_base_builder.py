#!/usr/bin/env python3
"""
Train an NPC base-builder agent using PPO with curriculum learning.

The agent learns to:
  Phase 0: Gather resources (chop trees, mine stone)
  Phase 1: Craft items (planks, tools, torches)
  Phase 2: Build structures (walls, roof, door, furnishings)
  Phase 3: Survive nights while maintaining/defending the base

Usage:
    pip install stable-baselines3
    python examples/train_base_builder.py --total-timesteps 500000
    python examples/train_base_builder.py --phase 0 --total-timesteps 100000  # train gathering only
"""

import argparse

import gymnasium as gym
import hytalegym  # noqa: F401
from hytalegym.wrappers import FlatActionWrapper, SimpleObsWrapper, CurriculumWrapper


def main():
    parser = argparse.ArgumentParser(description="Train NPC Base Builder agent")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--phase", type=int, default=0,
                        help="Starting curriculum phase (0=gather, 1=craft, 2=build, 3=survive)")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum learning, train on full task")
    parser.add_argument("--save-path", default="hytale_base_builder",
                        help="Path to save the trained model")
    args = parser.parse_args()

    env = gym.make(
        "HytaleBaseBuilder-v0",
        host=args.host,
        port=args.port,
    )

    # Apply curriculum learning
    if not args.no_curriculum:
        env = CurriculumWrapper(env, initial_phase=args.phase)

    # Flatten for SB3 compatibility
    env = FlatActionWrapper(env)
    env = SimpleObsWrapper(env)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback

        class CurriculumLogger(BaseCallback):
            """Log curriculum phase transitions."""
            def _on_step(self) -> bool:
                # Check if curriculum wrapper is present
                infos = self.locals.get("infos", [])
                for info in infos:
                    if "curriculum_phase" in info:
                        self.logger.record("curriculum/phase", info["curriculum_phase"])
                return True

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=128,
            n_epochs=10,
            gamma=0.995,        # longer horizon for base building
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,      # higher entropy for exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
            ),
        )

        callbacks = [CurriculumLogger()] if not args.no_curriculum else []

        print(f"Training Base Builder NPC for {args.total_timesteps} timesteps...")
        print(f"Starting curriculum phase: {args.phase}")
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
        )

        model.save(args.save_path)
        print(f"Model saved to {args.save_path}.zip")

    except ImportError:
        print("stable-baselines3 not installed. Run: pip install stable-baselines3")

    env.close()


if __name__ == "__main__":
    main()
