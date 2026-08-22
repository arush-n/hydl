"""Elitist genetic optimization over arbitrary floating-point policy PyTrees."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    """All genetic choices are explicit and safe to tune per experiment."""

    population_size: int = 32
    elite_fraction: float = 0.125
    mutation_std: float = 0.02
    mutation_rate: float = 1.0
    crossover_rate: float = 0.5

    def __post_init__(self) -> None:
        if (
            isinstance(self.population_size, bool)
            or not isinstance(self.population_size, int)
            or self.population_size < 2
        ):
            raise ValueError("population_size must be an integer >= 2")
        if not 0.0 < self.elite_fraction < 1.0:
            raise ValueError("elite_fraction must be in (0, 1)")
        if not math.isfinite(self.mutation_std) or self.mutation_std <= 0.0:
            raise ValueError("mutation_std must be positive")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")


class EvolutionState(NamedTuple):
    population: Any
    generation: jax.Array


class EvolutionMetrics(NamedTuple):
    best_fitness: jax.Array
    mean_fitness: jax.Array
    diversity: jax.Array
    invalid_fitness: jax.Array
    generation: jax.Array
    update_applied: jax.Array


def _floating_leaves(tree: Any, population_size: int | None = None):
    leaves, structure = jax.tree.flatten(tree)
    if not leaves:
        raise ValueError("parameter PyTree cannot be empty")
    leaves = [jnp.asarray(leaf) for leaf in leaves]
    for leaf in leaves:
        if not jnp.issubdtype(leaf.dtype, jnp.floating):
            raise TypeError("evolution supports floating-point parameter leaves only")
        if population_size is not None and (
            leaf.ndim < 1 or leaf.shape[0] != population_size
        ):
            raise ValueError("every population leaf must start with population_size")
    return leaves, structure


def initialize_evolution(
    center: Any, key: jax.Array, config: EvolutionConfig = EvolutionConfig()
) -> EvolutionState:
    """Create a population around ``center`` while keeping row zero exact."""

    leaves, structure = _floating_leaves(center)
    keys = jax.random.split(key, len(leaves))
    population = []
    for leaf, leaf_key in zip(leaves, keys):
        noise = jax.random.normal(
            leaf_key, (config.population_size,) + leaf.shape, dtype=leaf.dtype
        )
        values = leaf[None, ...] + config.mutation_std * noise
        population.append(values.at[0].set(leaf))
    return EvolutionState(
        population=jax.tree.unflatten(structure, population),
        generation=jnp.int32(0),
    )


def _diversity(leaves) -> jax.Array:
    squared, count = jnp.float32(0.0), 0
    for leaf in leaves:
        squared += jnp.sum(jnp.square(leaf - jnp.mean(leaf, axis=0)))
        count += leaf.size
    return jnp.sqrt(squared / max(count, 1))


def evolve(
    state: EvolutionState,
    fitness: jax.Array,
    key: jax.Array,
    config: EvolutionConfig = EvolutionConfig(),
) -> tuple[EvolutionState, EvolutionMetrics]:
    """Select elites, uniformly cross them, mutate children, and retain elites."""

    leaves, structure = _floating_leaves(state.population, config.population_size)
    fitness = jnp.asarray(fitness, dtype=jnp.float32)
    if fitness.shape != (config.population_size,):
        raise ValueError("fitness must have shape [population_size]")
    finite = jnp.isfinite(fitness)
    safe_fitness = jnp.where(finite, fitness, -jnp.inf)
    order = jnp.argsort(safe_fitness)[::-1]
    elite_count = max(
        1,
        min(
            config.population_size - 1,
            int(config.elite_fraction * config.population_size),
        ),
    )
    child_count = config.population_size - elite_count
    finite_count = jnp.sum(finite)
    elite_rows = jnp.where(
        jnp.arange(elite_count) < finite_count,
        order[:elite_count],
        order[0],
    )
    parent_key, *leaf_keys = jax.random.split(key, 1 + 2 * len(leaves))
    parent_rows = jax.random.randint(parent_key, (2, child_count), 0, elite_count)

    def next_population(_):
        next_leaves = []
        for index, leaf in enumerate(leaves):
            elites = leaf[elite_rows]
            first, second = elites[parent_rows[0]], elites[parent_rows[1]]
            crossover = jax.random.bernoulli(
                leaf_keys[index * 2], config.crossover_rate, first.shape
            )
            children = jnp.where(crossover, second, first)
            mutation_key = leaf_keys[index * 2 + 1]
            noise_key, mask_key = jax.random.split(mutation_key)
            noise = jax.random.normal(noise_key, children.shape, dtype=leaf.dtype)
            mutate = jax.random.bernoulli(
                mask_key, config.mutation_rate, children.shape
            )
            children += config.mutation_std * noise * mutate
            next_leaves.append(jnp.concatenate((elites, children), axis=0))
        return EvolutionState(
            jax.tree.unflatten(structure, next_leaves), state.generation + 1
        )

    valid = jnp.any(finite)
    next_state = jax.lax.cond(valid, next_population, lambda _: state, operand=None)
    denominator = jnp.maximum(finite_count, 1)
    metrics = EvolutionMetrics(
        best_fitness=jnp.max(safe_fitness),
        mean_fitness=jnp.sum(jnp.where(finite, fitness, 0.0)) / denominator,
        diversity=_diversity(leaves),
        invalid_fitness=jnp.sum(~finite, dtype=jnp.int32),
        generation=next_state.generation,
        update_applied=valid,
    )
    return next_state, metrics


__all__ = [
    "EvolutionConfig",
    "EvolutionMetrics",
    "EvolutionState",
    "evolve",
    "initialize_evolution",
]
