"""Initialize a fresh policy checkpoint on the *current* Arsenal surface.

Sizes are never passed in. ``arsenal_ppo_config`` derives ``observation_size``,
``action_size`` and ``action_head_sizes`` from ``ARSENAL_ENVIRONMENT_SPEC`` and
raises if you try to override them, so a checkpoint written here cannot be born
stale the way a hand-pinned one can. When the action surface moves again, rerun
this instead of porting columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from hytalegym.jax.training.arsenal import arsenal_ppo_config
from hytalegym.jax.training.checkpoint import save_policy_checkpoint
from hytalegym.jax.training.policy import initialize_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--seed", type=int, default=570_057)
    parser.add_argument("--encoder-size", type=int, default=64)
    parser.add_argument("--recurrent-size", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=4.0e-5)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--entropy-coefficient", type=float, default=0.003)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.destination.exists() and not args.force:
        raise SystemExit(f"{args.destination} exists; pass --force to replace")

    config = arsenal_ppo_config(
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        encoder_size=args.encoder_size,
        recurrent_size=args.recurrent_size,
        learning_rate=args.learning_rate,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        entropy_coefficient=args.entropy_coefficient,
    )
    params = initialize_policy(jax.random.key(args.seed), config)
    save_policy_checkpoint(
        args.destination,
        params,
        config,
        metadata={
            "schema": "hytalerl_arsenal_fresh_policy_v1",
            "checkpoint_role": "fresh_initialization",
            "training_state_resumable": False,
            "updates": 0,
            "environment_steps": 0,
            "initialization": {
                "seed": args.seed,
                "note": (
                    "Random init on the live Arsenal surface. Sizes come from "
                    "ARSENAL_ENVIRONMENT_SPEC via arsenal_ppo_config, which "
                    "rejects overrides of observation/action widths."
                ),
            },
        },
    )
    print(
        json.dumps(
            {
                "destination": str(args.destination),
                "observation_size": config.observation_size,
                "action_size": config.action_size,
                "action_head_sizes": list(config.action_head_sizes),
                "encoder_size": config.encoder_size,
                "recurrent_size": config.recurrent_size,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
