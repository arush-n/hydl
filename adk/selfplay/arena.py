"""Bind a :class:`~adk.selfplay.Population` to the Gym's multi-actor transport.

``make_arsenal_ppo_environment`` is single-agent by design -- it is the
compatibility path -- so self-play must go through
``hytalegym.jax.training.multi_actor``. That transport takes a
``PolicyActorAssignment(actor_index, policy_id, active, trainable)``, which is
almost the shape a population already has. The word that does not line up is
``trainable``:

* ``OpponentSpec.baseline`` distinguishes a **permanent anchor** from a member
  **produced by training**.
* ``PolicyActorAssignment.trainable`` distinguishes a slot the optimizer will
  **update right now** from a slot that must stay **byte-identical**.

A generation-3 self-play snapshot is not a baseline, and it is also not being
trained -- it is a frozen opponent. Reading ``trainable = not baseline`` marks
every historical snapshot as trainable and lets the optimizer rewrite it, which
destroys exactly the immutability :mod:`adk.selfplay.population` exists to
protect: once a recorded opponent moves, every historical comparison against it
becomes uninterpretable. So **exactly one member is the learner and every other
slot is frozen**, whether it is an anchor or an old snapshot.

Every assignment built here is checked with the Gym's own
``validate_policy_actor_assignment`` rather than against our assumptions about
its contract.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from adk.selfplay.population import Population

#: The learner is seated first so ``policy_id`` 0 is the row under training.
LEARNER_SLOT = 0


def assignment_for(
    population: Population,
    *,
    actors: int,
    learner: str | None = None,
    entities: int | None = None,
    batch: int = 1,
):
    """Seat ``actors`` arena slots from the population, learner first.

    ``learner`` names the member the optimizer may update. It defaults to the
    newest **learned** member (:attr:`Population.learned`, highest generation),
    matching ``Selection.LATEST``. Every other slot -- anchor or old snapshot
    alike -- arrives ``trainable=False``.

    A population with no learned members produces an all-frozen arena: that is
    a legitimate evaluation matchup (baseline versus baseline, nobody
    learning), and :func:`trainable_count` reports the zero rather than hiding
    it. The Gym's trainers refuse such an assignment loudly.

    ``entities`` is the arena's entity count, which is **not** the same as the
    actor count: an arena may hold scripted entities alongside the policy
    actors. It defaults to ``actors`` (every entity policy-controlled) and the
    Gym's validator rejects an actor index outside it.

    Every field is ``[batch, actors]`` -- the transport is batched, and the
    validator refuses a 1-D assignment.
    """

    from hytalegym.jax.training.multi_actor.assignment import (
        PolicyActorAssignment,
        validate_policy_actor_assignment,
    )

    entities = actors if entities is None else entities
    if entities < actors:
        raise ValueError(
            f"arena has {entities} entities but {actors} policy actors; an "
            "actor index outside the entity count is not addressable"
        )
    if actors < 1:
        raise ValueError(f"an arena needs at least one actor, got {actors}")
    if batch < 1:
        raise ValueError(f"batch must be positive, got {batch}")
    members = list(population.members)
    if not members:
        raise ValueError("cannot build an assignment from an empty population")
    if len(members) < actors:
        raise ValueError(
            f"population has {len(members)} members but the arena has {actors} "
            "actors; fill it explicitly rather than repeating a member, which "
            "would make a matchup look more diverse than it is"
        )

    chosen, flags = _seat(population, members, actors, learner)
    row = jnp.arange(actors, dtype=jnp.int32)

    def tile(value):
        return jnp.broadcast_to(value, (batch, actors))

    result = PolicyActorAssignment(
        actor_index=tile(row),
        policy_id=tile(row),
        active=tile(jnp.ones((actors,), dtype=jnp.bool_)),
        trainable=tile(jnp.asarray(flags, dtype=jnp.bool_)),
    )
    validate_policy_actor_assignment(result, entity_count=entities)
    return result


def seating(
    population: Population, *, actors: int, learner: str | None = None
) -> tuple:
    """Which member occupies which actor slot, learner first.

    The assignment itself carries only integers, so this is the only way to
    say *who* slot 2 was -- which is what a recorded matchup result has to
    name to stay interpretable.
    """

    members = list(population.members)
    if len(members) < actors:
        raise ValueError(
            f"population has {len(members)} members but the arena has {actors} actors"
        )
    return tuple(_seat(population, members, actors, learner)[0])


def _seat(population, members, actors, learner):
    """Order the arena's members and mark which single row may be updated."""

    if learner is None:
        pool = population.learned
        under_training = (
            max(pool, key=lambda member: member.generation) if pool else None
        )
    else:
        matches = [member for member in members if member.name == learner]
        if not matches:
            raise ValueError(
                f"no member named {learner!r} in the population; the learner "
                "must already be a recorded, versioned member"
            )
        under_training = matches[0]
        if under_training.baseline:
            raise ValueError(
                f"{learner!r} is a baseline and cannot be the learner; an "
                "anchor that moves is the one thing that can no longer reveal "
                "drift"
            )

    if under_training is None:
        return members[:actors], [False] * actors
    rest = [member for member in members if member.name != under_training.name]
    return (
        [under_training] + rest[: actors - 1],
        [True] + [False] * (actors - 1),
    )


def trainable_count(assignment) -> int:
    """How many actor rows the optimizer will update.

    This is also the value ``PPOConfig.num_envs`` must take: both Gym trainers
    reject a config whose ``num_envs`` disagrees with the assignment, so it is
    derived here rather than passed in. Zero means the run learns nothing.
    """

    return int(jnp.sum(assignment.trainable))


@dataclass(frozen=True, slots=True)
class Arena:
    """One built shared arena: assignment, environment, collector, bank.

    ``collect`` is the Gym's rollout collector, not ours. ``bank`` holds one
    policy row per actor slot, so a frozen opponent has its own parameters
    rather than sharing the learner's.
    """

    assignment: Any
    params: Any
    config: Any
    collect: Callable
    ppo: Any
    state: Any
    bank: Any
    entities: int
    actors: int
    batch: int
    world: Any = None

    @property
    def recurrent_state(self) -> jax.Array:
        """A zeroed ``[B,P,R]`` carry, the shape the collector demands."""

        return jnp.zeros(
            (self.batch, self.actors, self.ppo.recurrent_size),
            dtype=jnp.float32,
        )


def make_arena(
    population: Population,
    *,
    weapons: Sequence[str],
    actors: int | None = None,
    batch: int = 1,
    rollout_steps: int = 8,
    learner: str | None = None,
    seed: int = 0,
    health: float | Sequence[float] = 105.0,
    separation: float = 1.0,
    **ppo_overrides,
) -> Arena:
    """Compose the pieces ``make_multi_actor_rollout_collector`` requires.

    ``weapons`` names one authored profile per entity, so ``len(weapons)`` is
    the entity count. The first ``actors`` entities are policy-driven; any
    remainder keeps the scripted opponent controller
    (``opponent_controller_mask`` is True for *scripted* rows --
    ``arsenal/runtime.py:194-219``).

    **This is a control fixture, not production world capabilities.**
    ``open_flat_arsenal_world_capabilities`` documents itself as "explicit
    permissive control fixture; never the production default": the arena is
    open flat ground with everything permitted, which is what makes a
    self-play result about the policies rather than about terrain.
    """

    profiles = tuple(weapons)
    entities = len(profiles)
    actors = entities if actors is None else actors
    if entities < 2:
        raise ValueError(
            "an arena needs at least two entities; the authored loadout "
            "builder refuses a shorter row"
        )
    if batch < 1:
        raise ValueError(f"batch must be positive, got {batch}")
    assignment = assignment_for(
        population,
        actors=actors,
        entities=entities,
        batch=batch,
        learner=learner,
    )
    return _build_arena(
        profiles=profiles,
        actors=actors,
        assignment=assignment,
        batch=batch,
        rollout_steps=rollout_steps,
        seed=seed,
        health=health,
        separation=separation,
        ppo_overrides=ppo_overrides,
    )


def _build_arena(
    *,
    profiles: tuple[str, ...],
    actors: int,
    assignment: Any,
    batch: int,
    rollout_steps: int,
    seed: int,
    health: float | Sequence[float],
    separation: float,
    ppo_overrides: dict[str, Any],
) -> Arena:
    """Build the shared JAX runtime after a self-play mode owns assignment."""

    from hytalegym.jax.combat import default_combat_params
    from hytalegym.jax.combat.arsenal.environment import (
        open_flat_arsenal_world_capabilities,
    )
    from hytalegym.jax.combat.arsenal.profiles import (
        hytale_0_5_7_entity_loadouts,
    )
    from hytalegym.jax.combat.arsenal.runtime import (
        arsenal_runtime_config,
        reset_arsenal_batch,
    )
    from hytalegym.jax.training.arsenal import arsenal_ppo_config
    from hytalegym.jax.training.multi_actor import (
        initialize_multi_actor_arena_state,
        make_multi_actor_rollout_collector,
    )
    from hytalegym.jax.training.policy import initialize_policy

    entities = len(profiles)
    positions, yaws = _ring(entities, separation)
    healths = _healths(health, entities)

    def stack(row):
        return jnp.asarray([row] * batch, dtype=jnp.float32)

    entity_position, entity_yaw = stack(positions), stack(yaws)
    entity_health = stack(healths)

    loadouts = hytale_0_5_7_entity_loadouts(tuple(profiles for _ in range(batch)))
    scripted = jnp.asarray(
        [index >= actors for index in range(entities)], dtype=jnp.bool_
    )
    config = arsenal_runtime_config(loadouts, opponent_controller_mask=scripted)
    # Every entity is policy-driven unless we left one scripted, which is the
    # condition the packaged scripted-target ruleset exists for.
    params = default_combat_params(microticks=1, target_active=entities > actors)

    def reset_provider(keys):
        state, _ = reset_arsenal_batch(
            keys,
            params,
            config,
            entity_position=entity_position,
            entity_health=entity_health,
            entity_yaw=entity_yaw,
        )
        return state

    def capability_provider(state):
        return open_flat_arsenal_world_capabilities(state, params, config=config)

    ppo = arsenal_ppo_config(
        # PPO's batch is per trainable policy, and each builder seats a policy
        # exactly once in every arena.
        num_envs=batch,
        rollout_steps=rollout_steps,
        **{"update_epochs": 1, "num_minibatches": 1, **ppo_overrides},
    )

    policy_key, reset_key = jax.random.split(jax.random.key(seed))
    policies = tuple(
        initialize_policy(key, ppo) for key in jax.random.split(policy_key, actors)
    )
    bank = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *policies)
    state = initialize_multi_actor_arena_state(
        reset_provider(jax.random.split(reset_key, batch)), params
    )
    collect = make_multi_actor_rollout_collector(
        params,
        config,
        assignment,
        reset_provider=reset_provider,
        capability_provider=capability_provider,
        rollout_steps=rollout_steps,
    )
    return Arena(
        assignment=assignment,
        params=params,
        config=config,
        collect=collect,
        ppo=ppo,
        state=state,
        bank=bank,
        entities=entities,
        actors=actors,
        batch=batch,
    )


def _ring(entities: int, separation: float):
    """Place entities evenly on a circle, each facing the centre.

    Two entities degenerate to the reference fixture's head-on pair. Yaw 0
    faces -Z, which is the convention ``test_rollout.py:52-64`` encodes by
    pairing position ``(0,0,-1)`` with yaw ``180``.
    """

    if separation <= 0.0:
        raise ValueError(f"separation must be positive, got {separation}")
    radius = separation / 2.0
    angles = [2.0 * math.pi * index / entities for index in range(entities)]
    positions = [
        (radius * math.sin(angle), 0.0, -radius * math.cos(angle)) for angle in angles
    ]
    yaws = [(math.degrees(angle) + 180.0) % 360.0 for angle in angles]
    return positions, yaws


def _healths(health, entities: int):
    if isinstance(health, (int, float)):
        return [float(health)] * entities
    values = [float(value) for value in health]
    if len(values) != entities:
        raise ValueError(
            f"health has {len(values)} values but the arena has {entities} entities"
        )
    return values


def shared_policy_trainer(arena: Arena, *, compile: bool = True):
    """Bind the arena to the Gym's shared-policy PPO update.

    All trainable slots must select one policy ID, which :func:`assignment_for`
    guarantees by seating a single learner.
    """

    from hytalegym.jax.training.multi_actor import (
        make_multi_actor_shared_policy_trainer,
    )

    return make_multi_actor_shared_policy_trainer(
        arena.collect, arena.assignment, arena.ppo, compile=compile
    )


def make_duel(
    *,
    weapons: Sequence[str],
    batch: int = 1,
    rollout_steps: int = 8,
    seed: int = 0,
    health: float | Sequence[float] = 105.0,
    separation: float = 1.0,
    **ppo_overrides,
) -> Arena:
    """Build a head-to-head arena where two distinct policies both learn."""

    from arena.training.selfplay import duel_assignment

    profiles = tuple(weapons)
    if len(profiles) != 2:
        raise ValueError("a two-agent duel requires exactly two weapons")
    assignment = duel_assignment(batch, entity_count=2)
    return _build_arena(
        profiles=profiles,
        actors=2,
        assignment=assignment,
        batch=batch,
        rollout_steps=rollout_steps,
        seed=seed,
        health=health,
        separation=separation,
        ppo_overrides=ppo_overrides,
    )


def make_region_duel(
    *,
    weapons: Sequence[str],
    batch: int = 1,
    rollout_steps: int = 8,
    seed: int = 0,
    selection_key: int | None = None,
    node_seed: int | None = None,
    native_evidence: bool | None = None,
    **ppo_overrides,
) -> Arena:
    """Seat two live policies in one exact, mutable WorldGen Region.

    Terrain, actor-local observations, legal masks, and World actions come from
    the Gym's Region fixture and multi-actor collector. Both target-controller
    paths are disabled, so only the two policy rows can drive the duel.
    """

    from worlds import region as region_module
    from arena.training.selfplay import duel_assignment

    profiles = tuple(weapons)
    if len(profiles) != 2:
        raise ValueError("a two-agent Region duel requires exactly two weapons")
    if batch < 1:
        raise ValueError("batch must be positive")
    region_options = {"native_evidence": native_evidence}
    if selection_key is not None:
        region_options["selection_key"] = selection_key
    if node_seed is not None:
        region_options["node_seed"] = node_seed
    loaded = region_module.load_region(
        weapons=(profiles[0],) * batch,
        target_weapons=(profiles[1],) * batch,
        policy_controlled_targets=True,
        target_active=False,
        **region_options,
    )
    return _build_region_arena(
        loaded=loaded,
        assignment=duel_assignment(batch, entity_count=2),
        batch=batch,
        rollout_steps=rollout_steps,
        seed=seed,
        ppo_overrides=ppo_overrides,
    )


def _build_region_arena(
    *,
    loaded: Any,
    assignment: Any,
    batch: int,
    rollout_steps: int,
    seed: int,
    ppo_overrides: dict[str, Any],
) -> Arena:
    """Bind ADK policy rows to Gym's authoritative Region transport."""

    from hytalegym.jax.combat.arsenal.environment import ArsenalActionSurfaceRuntime
    from hytalegym.jax.combat.arsenal.runtime import reset_arsenal_batch
    from hytalegym.jax.combat.block_interactions import empty_block_interaction_state
    from hytalegym.jax.training.arsenal import arsenal_ppo_config
    from hytalegym.jax.training.multi_actor import (
        initialize_multi_actor_arena_state,
        make_multi_actor_region_rollout_collector,
    )
    from hytalegym.jax.training.policy import initialize_policy

    fixture, params, config = loaded.fixture, loaded.params, loaded.runtime

    def reset_provider(keys):
        reset = fixture.reset_provider(keys)
        inventory = (
            None
            if fixture.inventory_reset_provider is None
            else fixture.inventory_reset_provider(keys, config)
        )
        state, _ = reset_arsenal_batch(
            keys,
            params,
            config,
            fixture.geometry_provider,
            agent_position=reset.agent_position,
            target_position=reset.target_position,
            initial_inventory=inventory,
        )
        return state

    required = {
        name: getattr(fixture, name)
        for name in (
            "world_runtime_provider",
            "actor_world_runtime_provider",
            "action_surface_provider",
            "action_surface_runtime_initializer",
        )
    }
    missing = tuple(
        name for name, provider in required.items() if not callable(provider)
    )
    if missing:
        raise ValueError(
            f"Region duel lacks required provider(s): {', '.join(missing)}"
        )

    ppo = arsenal_ppo_config(
        num_envs=batch,
        rollout_steps=rollout_steps,
        **{"update_epochs": 1, "num_minibatches": 1, **ppo_overrides},
    )
    policy_key, reset_key = jax.random.split(jax.random.key(seed))
    bank = jax.tree_util.tree_map(
        lambda *rows: jnp.stack(rows),
        *(initialize_policy(key, ppo) for key in jax.random.split(policy_key, 2)),
    )
    reset_keys = jax.random.split(reset_key, batch)
    arsenal = reset_provider(reset_keys)
    action_runtime = ArsenalActionSurfaceRuntime(
        block_interactions=empty_block_interaction_state(batch, entity_count=2),
        world=fixture.action_surface_runtime_initializer(
            reset_keys, arsenal, params, config
        ),
    )
    state = initialize_multi_actor_arena_state(arsenal, params, action_runtime)
    collect = make_multi_actor_region_rollout_collector(
        params,
        config,
        assignment,
        reset_provider=reset_provider,
        rollout_steps=rollout_steps,
        target_navigation_provider=fixture.target_navigation_provider,
        explosion_candidate_provider=fixture.explosion_candidate_provider,
        action_surface_executor=fixture.action_surface_executor,
        **required,
    )
    return Arena(
        assignment=assignment,
        params=params,
        config=config,
        collect=collect,
        ppo=ppo,
        state=state,
        bank=bank,
        entities=2,
        actors=2,
        batch=batch,
        world={
            "kind": "worldgen-v2-region",
            "metadata": fixture.metadata,
            "native_evidence": loaded.evidence,
        },
    )


def duel_trainer(arena: Arena, *, compile: bool = True):
    """Bind independent PPO optimizer state to both agents in a duel."""

    from arena.training.selfplay import make_duel_ppo_trainer

    return make_duel_ppo_trainer(
        arena.collect, arena.assignment, arena.ppo, compile=compile
    )
