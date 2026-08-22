#!/usr/bin/env python3
"""
Random agent example for HytaleRL.

Runs a random policy on the default survive task.
Requires a Hytale server with the HytaleRLBridge plugin running on localhost:5556.

Usage:
    python examples/random_agent.py
    python examples/random_agent.py --task navigate --episodes 5
"""

import argparse

import gymnasium as gym
import hytalegym  # noqa: F401 — registers environments


def main():
    parser = argparse.ArgumentParser(description="HytaleRL random agent")
    parser.add_argument("--task", default="survive", help="Task ID (survive, mine_adamantite, etc.)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--host", default="127.0.0.1", help="Bridge server host")
    parser.add_argument("--port", type=int, default=5556, help="Bridge server port")
    args = parser.parse_args()

    env_map = {
        "survive": "HytaleSurvive-v0",
        "mine_adamantite": "HytaleMineAdamantite-v0",
        "kill_trork": "HytaleKillTrork-v0",
        "build_house": "HytaleBuildHouse-v0",
        "navigate": "HytaleNavigate-v0",
    }

    env_id = env_map.get(args.task, "HytaleRL-v0")
    env = gym.make(env_id, host=args.host, port=args.port)

    for episode in range(args.episodes):
        obs, info = env.reset(seed=episode)
        total_reward = 0
        steps = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        print(f"Episode {episode + 1}: steps={steps}, reward={total_reward:.2f}")

    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
