#!/usr/bin/env python3
"""Train the recurrent combat policy with one compiled JAX call per update."""

from __future__ import annotations

import argparse
import time

import jax
import numpy as np

from hytalegym.jax.combat import default_combat_params
from hytalegym.jax.training import (
    PPOConfig,
    combat_checkpoint_metadata,
    initialize_training,
    make_evaluator,
    make_train_step,
    save_policy_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--encoder-size", type=int, default=64)
    parser.add_argument("--recurrent-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--microticks", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument(
        "--combat-target-active",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=570057)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--evaluation-envs", type=int, default=128)
    parser.add_argument("--evaluation-steps", type=int, default=256)
    parser.add_argument(
        "--checkpoint",
        default="jax_combat_policy.npz",
    )
    args = parser.parse_args()
    if args.updates < 1 or args.log_every < 1:
        parser.error("updates and log-every must be positive")
    if args.evaluation_envs < 1 or args.evaluation_steps < 1:
        parser.error("evaluation sizes must be positive")

    config = PPOConfig(
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        encoder_size=args.encoder_size,
        recurrent_size=args.recurrent_size,
        learning_rate=args.learning_rate,
    )
    environment_params = default_combat_params(
        microticks=args.microticks,
        target_active=args.combat_target_active,
    )
    master_key = jax.random.key(args.seed)
    master_key, initialization_key = jax.random.split(master_key)
    train_state = initialize_training(
        initialization_key,
        config,
        environment_params,
    )
    train_step = make_train_step(config, environment_params)

    compile_seconds = 0.0
    measured_transitions = 0
    measured_seconds = 0.0
    for update in range(1, args.updates + 1):
        master_key, update_key = jax.random.split(master_key)
        started = time.perf_counter()
        train_state, metrics = train_step(train_state, update_key)
        jax.block_until_ready(metrics)
        elapsed = time.perf_counter() - started
        if update == 1:
            compile_seconds = elapsed
        else:
            measured_seconds += elapsed
            measured_transitions += (
                config.num_envs * config.rollout_steps
            )
        if (
            update == 1
            or update % args.log_every == 0
            or update == args.updates
        ):
            rate = (
                measured_transitions / measured_seconds
                if measured_seconds > 0.0
                else 0.0
            )
            print(
                f"update={update:>5} "
                f"steps={int(metrics.total_environment_steps):>10} "
                f"reward={float(metrics.mean_rollout_reward):>8.4f} "
                f"return={float(metrics.mean_episode_return):>8.3f} "
                f"episodes={int(metrics.episodes_completed):>5} "
                f"loss={float(metrics.total_loss):>9.5f} "
                f"entropy={float(metrics.entropy):>7.4f} "
                f"kl={float(metrics.approximate_kl):>8.6f} "
                f"transitions/s={rate:>10,.0f}"
            )

    destination = save_policy_checkpoint(
        args.checkpoint,
        train_state.policy_params,
        config,
        metadata=combat_checkpoint_metadata(
            microticks=args.microticks,
            combat_target_active=args.combat_target_active,
            seed=args.seed,
            updates=args.updates,
            environment_steps=int(train_state.total_environment_steps),
        ),
    )

    evaluation_batch = args.evaluation_envs
    evaluator = make_evaluator(
        environment_params,
        max_policy_steps=args.evaluation_steps,
    )
    master_key, reset_key, rollout_key = jax.random.split(master_key, 3)
    evaluation = evaluator(
        train_state.policy_params,
        jax.random.split(reset_key, evaluation_batch),
        rollout_key,
    )
    jax.block_until_ready(evaluation)
    success_rate = float(np.mean(np.asarray(evaluation.success)))
    termination_rate = float(np.mean(np.asarray(evaluation.terminated)))
    steady_rate = (
        measured_transitions / measured_seconds
        if measured_seconds > 0.0
        else 0.0
    )
    print(
        f"checkpoint={destination} compile_seconds={compile_seconds:.3f} "
        f"steady_transitions/s={steady_rate:,.0f} "
        f"evaluation_success={success_rate:.3f} "
        f"evaluation_terminated={termination_rate:.3f}"
    )


if __name__ == "__main__":
    main()
