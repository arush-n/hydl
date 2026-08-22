"""Train the public WorldGen V2 IL -> PPO agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Imported for its import-time side effect: configure_training_runtime()
# must run before JAX is imported. Not unused -- do not delete.
from agents import training_runtime as _training_runtime  # noqa: F401

import jax

from agents.ppo import PPOAgent, PPOSettings
from agents.ppo.worldgen.minigames import CUSTOM_MINIGAMES, publication_minigame
from agents.ppo.runtime import default_compilation_cache
from agents.ppo.worldgen.worldgen_benchmark import make_worldgen_evaluator
from arena.jax_contract import HEAD_SPANS
from arena.tasks.games.world.generated import GENERATED


_TASKS = {task.name: task for task in (*CUSTOM_MINIGAMES, *GENERATED)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run recurrent IL -> PPO in an exact WorldGen V2 JAX task."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--task", choices=sorted(_TASKS))
    source.add_argument(
        "--publication",
        type=Path,
        help="verified custom WorldGen V2 .publication manifest",
    )
    parser.add_argument("--difficulty", choices=("inert", "armed"), default="inert")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--trace-steps", type=int, default=4)
    parser.add_argument("--cloning-steps", type=int, default=32)
    parser.add_argument("--cloning-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--cloning-label-smoothing", type=float, default=0.05)
    parser.add_argument("--cloning-entropy-weight", type=float, default=0.01)
    parser.add_argument("--cloning-target-accuracy", type=float, default=0.80)
    parser.add_argument("--cloning-min-steps", type=int, default=8)
    parser.add_argument(
        "--imitation-heads",
        nargs="+",
        choices=tuple(HEAD_SPANS),
        default=tuple(HEAD_SPANS),
    )
    parser.add_argument("--ppo-updates", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=2)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--num-minibatches", type=int, default=1)
    parser.add_argument("--encoder-size", type=int, default=8)
    parser.add_argument("--recurrent-size", type=int, default=8)
    parser.add_argument("--evaluation-steps", type=int, default=32)
    parser.add_argument("--evaluation-mode", choices=("sample", "greedy"), default="greedy")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--seed", type=int, default=570057)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("agents/ppo/artifacts/custom-minigame-il-smoke"),
    )
    parser.add_argument(
        "--compilation-cache",
        type=Path,
        default=default_compilation_cache(),
    )
    args = parser.parse_args()

    settings = PPOSettings(
        batch=args.batch,
        trace_steps=args.trace_steps,
        cloning_steps=args.cloning_steps,
        cloning_learning_rate=args.cloning_learning_rate,
        cloning_label_smoothing=args.cloning_label_smoothing,
        cloning_entropy_weight=args.cloning_entropy_weight,
        cloning_target_accuracy=args.cloning_target_accuracy,
        cloning_min_steps=args.cloning_min_steps,
        imitation_heads=tuple(args.imitation_heads),
        ppo_updates=args.ppo_updates,
        rollout_steps=args.rollout_steps,
        ppo_update_epochs=args.update_epochs,
        ppo_num_minibatches=args.num_minibatches,
        encoder_size=args.encoder_size,
        recurrent_size=args.recurrent_size,
        seed=args.seed,
        output_dir=args.output_dir,
        compilation_cache=args.compilation_cache,
    )
    task = (
        publication_minigame(args.publication)
        if args.publication is not None
        else _TASKS[args.task or "taiga_pressure"]
    )
    agent = PPOAgent(settings, task=task, difficulty=args.difficulty)
    evaluate = None
    if not args.skip_evaluation:
        evaluate = make_worldgen_evaluator(
            agent.scene,
            task,
            steps=args.evaluation_steps,
            recurrent_size=args.recurrent_size,
            config=agent.ppo_config(),
            mode=args.evaluation_mode,
            key=jax.random.key(args.seed + 1),
        )
    report = agent.run(evaluate=evaluate)
    print(
        json.dumps(
            {
                "report_path": report["report_path"],
                "device": report["runtime"],
                "scene": report["scene"],
                "imitation": report["imitation"],
                "evaluation": report.get("evaluation"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
