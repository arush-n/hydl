"""Build a pursuit arena from an authored flat design, with no Region capture.

A Region fixture answers "where is the ground" from captured voxels, which is
why it costs a ~6 minute artifact hash and why its spawns are traversal nodes
that can sit in a cave. A design whose own preview receipt reports zero cave
volume and near-zero relief does not need any of that: the ground is a plane at
`base_height`, and every point on it is standable.

`agent_spawn` is a single `(3,)` and `floor_y` a scalar, so params cannot carry
one spawn per environment. Per-env placement therefore lives in the reset
provider, which is the same seam the Region path uses -- positions are drawn
from the reset keys, so they vary per environment and stay reproducible for a
run seed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import jax
import jax.numpy as jnp

from worlds.designs import Design

__all__ = ["FlatArena", "flat_arena", "FLAT_ARENA_SCHEMA"]

FLAT_ARENA_SCHEMA = "hytalerl-flat-design-arena-v1"

#: Half-width of the square the agent may spawn in, in blocks. The design is a
#: plane, so this bounds the arena rather than the terrain: it exists so two
#: actors cannot start a hundred blocks apart on an unbounded floor.
DEFAULT_ARENA_RADIUS = 48.0


@dataclass(frozen=True, slots=True)
class FlatArena:
    """The seams `bind_pursuit_collector` needs for a design-built world."""

    design: Design
    params: Any
    reset_provider: Any
    capability_provider: Any
    metadata: dict[str, Any]


def _spawn_positions(keys, *, floor: float, arena_radius: float,
                     separation_range: tuple[float, float]):
    """Agent anywhere on the plane, target at a random bearing and distance.

    Drawing the separation directly (rather than sampling two independent
    points and keeping the pairs that land in range) makes every environment
    usable and puts the requested band on the distance itself, which is what
    `target_separation_range` is supposed to control.
    """

    agent_key, bearing_key, distance_key = jax.random.split(keys, 3)
    low, high = float(separation_range[0]), float(separation_range[1])
    # Keep the pair inside the arena: the agent is drawn from a square shrunk
    # by the largest separation, so the target cannot be pushed outside it.
    margin = max(arena_radius - high, 0.0)
    agent_xz = jax.random.uniform(
        agent_key, (2,), minval=-margin, maxval=margin, dtype=jnp.float32
    )
    bearing = jax.random.uniform(
        bearing_key, (), minval=0.0, maxval=2.0 * math.pi, dtype=jnp.float32
    )
    # Uniform over the annulus by area, not over the radius: sampling the
    # radius uniformly would crowd the pair toward the inner edge, and the
    # cave runs already showed a separation distribution pinned at its floor.
    unit = jax.random.uniform(distance_key, (), dtype=jnp.float32)
    distance = jnp.sqrt(unit * (high**2 - low**2) + low**2)
    offset = jnp.stack((jnp.cos(bearing), jnp.sin(bearing))) * distance
    target_xz = agent_xz + offset
    height = jnp.float32(floor)
    return jnp.stack(
        (
            jnp.stack((agent_xz[0], height, agent_xz[1])),
            jnp.stack((target_xz[0], height, target_xz[1])),
        )
    )


def flat_arena(
    design: Design,
    *,
    params,
    config,
    target_separation_range: tuple[float, float],
    arena_radius: float = DEFAULT_ARENA_RADIUS,
    agent_health: float = 105.0,
    target_health: float = 105.0,
    agent_stamina_fraction: float = 1.0,
    stamina_regen_scale: float = 1.0,
) -> FlatArena:
    """Bind one authored flat design as a pursuit arena.

    Refuses a design its own preview receipt says is not flat, rather than
    quietly flattening it: a run that trained on terrain the design did not
    describe would be labelled with that design's digest and be wrong.
    """

    reasons = design.unsuitable()
    if reasons:
        raise ValueError(
            f"design {design.design_id!r} cannot back a flat arena: "
            f"{'; '.join(reasons)}"
        )
    low, high = float(target_separation_range[0]), float(target_separation_range[1])
    if not 0.0 < low <= high:
        raise ValueError("target separation range must be positive and ordered")
    if high >= arena_radius:
        raise ValueError(
            f"separation {high:g} does not fit an arena of radius {arena_radius:g}"
        )

    from hytalegym.jax.combat.arsenal.environment import (
        open_flat_arsenal_world_capabilities,
    )
    from hytalegym.jax.combat.arsenal.runtime import reset_arsenal_batch

    floor = float(design.base_height)
    bound = params._replace(
        floor_y=jnp.float32(floor),
        agent_spawn=jnp.asarray((0.0, floor, 0.0), dtype=jnp.float32),
        # Applies to the whole simulation, but only the agent actor carries a
        # locomotion-stamina bar, so in a two-actor duel this reaches the
        # learner alone.
        locomotion_stamina_regen_scale=jnp.float32(stamina_regen_scale),
    )

    def reset_provider(keys):
        positions = jax.vmap(
            lambda key: _spawn_positions(
                key,
                floor=floor,
                arena_radius=arena_radius,
                separation_range=(low, high),
            )
        )(keys)
        health = jnp.broadcast_to(
            jnp.asarray((agent_health, target_health), dtype=jnp.float32),
            positions.shape[:2],
        )
        state, _ = reset_arsenal_batch(
            keys, bound, config,
            entity_position=positions,
            entity_health=health,
        )
        if agent_stamina_fraction < 1.0:
            # The bar belongs to the agent actor, so this handicaps exactly the
            # side being trained. Starting it short is not the same as a lower
            # ceiling -- regeneration still clips to the engine's global
            # maximum, so the bar refills to full over an episode. This buys the
            # opening cost of a sprint, not a permanent cap.
            # `reset_arsenal_batch` returns an ArsenalEnvironmentState, whose
            # CombatState is directly on `.combat` -- there is no `.arsenal`
            # level here, unlike MultiActorArenaState.
            state = state._replace(
                combat=state.combat._replace(
                    agent_locomotion_stamina=(
                        state.combat.agent_locomotion_stamina
                        * jnp.float32(agent_stamina_fraction)
                    )
                )
            )
        return state

    def capability_provider(state):
        return open_flat_arsenal_world_capabilities(state, bound, config=config)

    metadata = {
        "schema": FLAT_ARENA_SCHEMA,
        "world": "design",
        "design_id": design.design_id,
        # The identity that matters: seed-independent, so re-previewing this
        # design at another seed does not relabel the run.
        "recipe_digest": design.recipe_digest,
        "design_name": design.name,
        "design_seed": design.seed,
        "floor_y": floor,
        "arena_radius": arena_radius,
        "target_separation_range": [low, high],
        "relief": design.relief,
        "maximum_slope": design.maximum_slope,
        "estimated_cave_volume": design.cave_volume,
        "structure_count": design.structure_count,
        # Stated, not implied: a design's authored structures are not built
        # here. The capability seam has no solid-obstacle field, so a design
        # with pillars would train as bare ground.
        "structures_realized": False,
    }
    return FlatArena(
        design=design,
        params=bound,
        reset_provider=reset_provider,
        capability_provider=capability_provider,
        metadata=metadata,
    )
