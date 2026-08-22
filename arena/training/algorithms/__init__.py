"""Learners and optimisers.

``ppo``
    Reference recurrent PPO wired directly to an ``ArenaScene``.
``evolution``
    Elitist genetic optimisation over arbitrary floating-point policy PyTrees.
``auto_improvement``
    Pure-JAX train/evaluate/select loops for persistent policy improvement.
"""

from __future__ import annotations

from arena.training.algorithms.auto_improvement import (
    AUTO_IMPROVEMENT_PROGRAM_SCHEMA,
    AUTO_IMPROVEMENT_SELECTION_LAW,
    AutoImprovementEvaluation,
    AutoImprovementProgram,
    AutoImprovementResult,
    AutoImprovementRound,
    run_auto_improvement,
)
from arena.training.algorithms.evolution import (
    EvolutionConfig,
    EvolutionMetrics,
    EvolutionState,
    evolve,
    initialize_evolution,
)
from arena.training.algorithms.ppo import (
    PPOConfig,
    initialize_ppo,
    make_ppo_collector,
    make_ppo_step,
    make_ppo_update,
    make_replicated_ppo_step,
    ppo_config,
    ppo_next_observation,
    relabel_ppo_rewards,
    warm_start_ppo,
)


__all__ = [
    "AUTO_IMPROVEMENT_PROGRAM_SCHEMA",
    "AUTO_IMPROVEMENT_SELECTION_LAW",
    "AutoImprovementEvaluation",
    "AutoImprovementProgram",
    "AutoImprovementResult",
    "AutoImprovementRound",
    "EvolutionConfig",
    "EvolutionMetrics",
    "EvolutionState",
    "PPOConfig",
    "evolve",
    "initialize_evolution",
    "initialize_ppo",
    "make_ppo_collector",
    "make_ppo_step",
    "make_ppo_update",
    "make_replicated_ppo_step",
    "ppo_config",
    "ppo_next_observation",
    "relabel_ppo_rewards",
    "run_auto_improvement",
    "warm_start_ppo",
]
