"""Pure-JAX train/evaluate/select loops for persistent policy improvement.

The callbacks supplied to :func:`run_auto_improvement` must themselves be
pure JAX programs.  The loop evaluates the incumbent and candidate on the
same held-out inputs each round, accepts only a gated improvement, and rolls
back the complete candidate state (parameters *and* optimizer moments) when
the candidate is rejected.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp


AUTO_IMPROVEMENT_PROGRAM_SCHEMA = "arena-auto-improvement-program-v1"
AUTO_IMPROVEMENT_SELECTION_LAW = (
    "paired_same_evaluation_input;candidate_eligible_and_finite;"
    "objective_delta_at_least_minimum;retention_drop_at_most_maximum;"
    "accept_entire_candidate_state_else_rollback_parameters_and_optimizer"
)


def _seed_rounds(
    value: tuple[tuple[int, ...], ...], name: str
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, tuple) or not value:
        raise TypeError(f"{name} must be a nonempty tuple of seed tuples")
    result: list[tuple[int, ...]] = []
    width: int | None = None
    for round_seeds in value:
        if not isinstance(round_seeds, tuple) or not round_seeds:
            raise TypeError(f"each {name} round must be a nonempty tuple")
        row: list[int] = []
        for seed in round_seeds:
            if isinstance(seed, bool) or not isinstance(seed, Integral):
                raise TypeError(f"{name} values must be integers")
            if int(seed) < 0:
                raise ValueError(f"{name} values must be nonnegative")
            row.append(int(seed))
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"every {name} round must have the same width")
        result.append(tuple(row))
    return tuple(result)


def _finite_nonnegative(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


@dataclass(frozen=True, slots=True)
class AutoImprovementProgram:
    """Content-addressed seed and acceptance contract for one device loop."""

    training_seed_rounds: tuple[tuple[int, ...], ...]
    evaluation_seed_rounds: tuple[tuple[int, ...], ...]
    minimum_objective_improvement: float = 0.0
    maximum_retention_drop: float = 0.0
    schema: str = AUTO_IMPROVEMENT_PROGRAM_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != AUTO_IMPROVEMENT_PROGRAM_SCHEMA:
            raise ValueError("auto-improvement schema is not current")
        training = _seed_rounds(self.training_seed_rounds, "training seeds")
        evaluation = _seed_rounds(self.evaluation_seed_rounds, "evaluation seeds")
        if len(training) != len(evaluation):
            raise ValueError("training and evaluation schedules must have equal rounds")
        flattened_training = tuple(seed for row in training for seed in row)
        flattened_evaluation = tuple(seed for row in evaluation for seed in row)
        if len(set(flattened_training)) != len(flattened_training):
            raise ValueError("training seeds must be globally unique")
        if len(set(flattened_evaluation)) != len(flattened_evaluation):
            raise ValueError("evaluation seeds must be globally unique")
        if set(flattened_training) & set(flattened_evaluation):
            raise ValueError("training and evaluation seeds must be disjoint")
        object.__setattr__(self, "training_seed_rounds", training)
        object.__setattr__(self, "evaluation_seed_rounds", evaluation)
        object.__setattr__(
            self,
            "minimum_objective_improvement",
            _finite_nonnegative(
                self.minimum_objective_improvement,
                "minimum objective improvement",
            ),
        )
        object.__setattr__(
            self,
            "maximum_retention_drop",
            _finite_nonnegative(self.maximum_retention_drop, "maximum retention drop"),
        )

    @property
    def rounds(self) -> int:
        return len(self.training_seed_rounds)

    @property
    def updates_per_round(self) -> int:
        return len(self.training_seed_rounds[0])

    @property
    def evaluations_per_round(self) -> int:
        return len(self.evaluation_seed_rounds[0])

    def training_seeds(self) -> jax.Array:
        return jnp.asarray(self.training_seed_rounds, dtype=jnp.uint32)

    def evaluation_seeds(self) -> jax.Array:
        return jnp.asarray(self.evaluation_seed_rounds, dtype=jnp.uint32)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "training_seed_rounds": [list(row) for row in self.training_seed_rounds],
            "evaluation_seed_rounds": [
                list(row) for row in self.evaluation_seed_rounds
            ],
            "minimum_objective_improvement": (self.minimum_objective_improvement.hex()),
            "maximum_retention_drop": self.maximum_retention_drop.hex(),
            "selection_law": AUTO_IMPROVEMENT_SELECTION_LAW,
        }

    @property
    def contract_sha256(self) -> str:
        payload = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest().upper()


class AutoImprovementEvaluation(NamedTuple):
    """Scalar candidate objective, prior-skill retention, and hard eligibility."""

    objective_score: jax.Array
    retention_score: jax.Array
    eligible: jax.Array


class AutoImprovementRound(NamedTuple):
    """Device-resident evidence for one train/evaluate/select round."""

    candidate: AutoImprovementEvaluation
    incumbent_before: AutoImprovementEvaluation
    incumbent_after: AutoImprovementEvaluation
    accepted: jax.Array
    training_metrics: Any


class AutoImprovementResult(NamedTuple):
    """Final accepted state plus every candidate and selection decision."""

    state: Any
    evaluation: AutoImprovementEvaluation
    rounds: AutoImprovementRound


def _as_evaluation(value: Any) -> AutoImprovementEvaluation:
    if not isinstance(value, AutoImprovementEvaluation):
        raise TypeError("evaluate must return AutoImprovementEvaluation")
    objective = jnp.asarray(value.objective_score, dtype=jnp.float32)
    retention = jnp.asarray(value.retention_score, dtype=jnp.float32)
    eligible = jnp.asarray(value.eligible, dtype=jnp.bool_)
    if objective.shape or retention.shape or eligible.shape:
        raise ValueError("auto-improvement evaluation fields must be scalars")
    return AutoImprovementEvaluation(objective, retention, eligible)


def _select_tree(predicate: jax.Array, candidate: Any, incumbent: Any) -> Any:
    if jax.tree_util.tree_structure(candidate) != jax.tree_util.tree_structure(
        incumbent
    ):
        raise TypeError("candidate and incumbent state trees differ")
    return jax.tree_util.tree_map(
        lambda new, old: jnp.where(predicate, new, old), candidate, incumbent
    )


def run_auto_improvement(
    program: AutoImprovementProgram,
    initial_state: Any,
    training_inputs: Any,
    evaluation_inputs: Any,
    *,
    train_round: Callable[[Any, Any], tuple[Any, Any]],
    evaluate: Callable[[Any, Any], AutoImprovementEvaluation],
) -> AutoImprovementResult:
    """Run all candidate updates, tests, and rollback decisions on device.

    ``training_inputs`` and ``evaluation_inputs`` are pytrees whose leading
    axis equals ``program.rounds``.  A caller should derive them from the
    program's exact seed arrays.  Both candidate and incumbent are evaluated
    on the same evaluation input for each round, eliminating seed drift from
    the selection decision.
    """

    if not isinstance(program, AutoImprovementProgram):
        raise TypeError("auto-improvement program is invalid")
    if not callable(train_round) or not callable(evaluate):
        raise TypeError("auto-improvement callbacks must be callable")
    for name, tree in (
        ("training inputs", training_inputs),
        ("evaluation inputs", evaluation_inputs),
    ):
        leaves = jax.tree_util.tree_leaves(tree)
        if not leaves:
            raise ValueError(f"{name} must contain at least one array")
        for leaf in leaves:
            array = jnp.asarray(leaf)
            if not array.shape or array.shape[0] != program.rounds:
                raise ValueError(f"{name} must have leading axis program.rounds")

    minimum = jnp.float32(program.minimum_objective_improvement)
    maximum_drop = jnp.float32(program.maximum_retention_drop)

    def step(incumbent: Any, inputs: tuple[Any, Any]):
        training_input, evaluation_input = inputs
        candidate, training_metrics = train_round(incumbent, training_input)
        incumbent_evaluation = _as_evaluation(evaluate(incumbent, evaluation_input))
        candidate_evaluation = _as_evaluation(evaluate(candidate, evaluation_input))
        finite = (
            jnp.isfinite(candidate_evaluation.objective_score)
            & jnp.isfinite(candidate_evaluation.retention_score)
            & jnp.isfinite(incumbent_evaluation.objective_score)
            & jnp.isfinite(incumbent_evaluation.retention_score)
        )
        accepted = (
            candidate_evaluation.eligible
            & finite
            & (
                candidate_evaluation.objective_score
                >= incumbent_evaluation.objective_score + minimum
            )
            & (
                candidate_evaluation.retention_score
                >= incumbent_evaluation.retention_score - maximum_drop
            )
        )
        next_state = _select_tree(accepted, candidate, incumbent)
        next_evaluation = _select_tree(
            accepted, candidate_evaluation, incumbent_evaluation
        )
        return next_state, AutoImprovementRound(
            candidate_evaluation,
            incumbent_evaluation,
            next_evaluation,
            accepted,
            training_metrics,
        )

    final_state, history = jax.lax.scan(
        step, initial_state, (training_inputs, evaluation_inputs)
    )
    final_evaluation = jax.tree_util.tree_map(
        lambda value: value[-1], history.incumbent_after
    )
    return AutoImprovementResult(final_state, final_evaluation, history)


__all__ = [
    "AUTO_IMPROVEMENT_PROGRAM_SCHEMA",
    "AUTO_IMPROVEMENT_SELECTION_LAW",
    "AutoImprovementEvaluation",
    "AutoImprovementProgram",
    "AutoImprovementResult",
    "AutoImprovementRound",
    "run_auto_improvement",
]
