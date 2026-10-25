"""The trainers' terrain world: a 0.7.2 Region behind the flat world's interface.

``simple_tasks.make_flat_npc_environment`` is the flat world the skill trainers run on. This module builds the same
shape of environment on a Region library (real terrain), so a skill can learn to step up, hop and detour:
``.batch``, ``.reset(keys, agent_xz, agent_yaw, other_xyz) -> carry`` and ``.step(carry, factors, keys)``. ``run_episode``
takes it unchanged, with ``carry=`` from its ``reset``.

Units: blocks, degrees, decision ticks (1/30 s with the motion timing profile pinned to 0, as on the flat world).

The Region fixes its lanes when it loads: ``environment_diversity`` draws one actor-legal, line-of-sight pair per lane
inside the chosen separation band. ``reset`` therefore accepts only those positions, and the trainers must take them
from ``env.task_placements``. The floor of each lane is the placement's own height.

What differs from ``worlds.region.region_scene``, and why (the simulator and ``worlds/`` are not edited here):

- ``load_region`` builds the player motion model and leaves ``legal_observation`` off. This module calls the benchmark's
  fixture builder with the ``npc_walk`` parameters the flat world uses, so placements are legal for that body.
- The geometry tokens and the terrain line of sight are bound (``world_token_provider``, ``world_capability_provider``).
  ``REGION_PROVIDER_SEAMS`` leaves them out, so this world is not ``region_scene``'s.
- No action surface and no world runtime: the block heads stay closed, as on the flat world.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.tasks.simple_tasks import DEFAULT_LOADOUT
from hytalegym.jax.combat.arsenal.compilation import plan_arsenal_compilation
from hytalegym.jax.combat.arsenal.environment import (
    ArsenalActionSurfaceRuntime,
    ArsenalEnvironmentCarry,
    _EmptyArsenalActionSurfaceEvidence,
    make_arsenal_environment,
)
from hytalegym.jax.combat.arsenal.profiles.hytale_0_5_7 import hytale_0_5_7_loadouts
from hytalegym.jax.combat.arsenal.runtime import arsenal_runtime_config, reset_arsenal_batch
from hytalegym.jax.combat.block_interactions import (
    empty_block_interaction_state,
    interaction_movement_constraints,
)
from hytalegym.jax.combat.observation.v1.runtime.factory import empty_injected_world_features
from hytalegym.jax.combat.observation.v3.encoding.encoder import encode_learner_observation_v3
from hytalegym.jax.combat.observation.v3.policy import arsenal_policy_action_context_mask
from hytalegym.jax.combat.observation.v3.policy.actions import learner_arsenal_action_context
from hytalegym.jax.combat.observation.v3.tokens.inventory import inventory_policy_tokens_from_state
from hytalegym.jax.combat.observation.v3.tokens.light import empty_actor_light_policy_tokens
from hytalegym.jax.combat.observation.v3.tokens.world import (
    empty_world_geometry_policy_tokens,
    normalize_world_geometry_policy_config,
)
from hytalegym.jax.combat.types import AGENT_ENTITY, default_combat_params
from worlds import libraries
from worlds import region as world_region

LIBRARY = "region-library-pilot-v2-072"
"""The 0.7.2 Region library with block identity, the default server line's terrain."""
DEFAULT_SEPARATION_RANGE = (3.0, 15.0)
"""Horizontal distance band of a lane's goal from its start, in blocks: walk_to's distance range."""
PLACEMENT_TOLERANCE = 1.0e-3
"""How far a passed agent_xz or other_xyz may sit from its placement (blocks)."""
#: Worlds the batch is spread across by default (the fixture's working set). Measured on batch 32 with the
#: 0.7.2 pilot library, 2026-10-25: capacity 1 gave 2 distinct spawns and a 19 s build, 8 gave 17 and 57 s,
#: 16 gave 29 and 85 s. Each row's pair comes from its own world's pool of at most 8 legal sources, so the
#: capacity sets how many spawns a batch can hold.
WORKING_SET_CAPACITY = 8


class TaskPlacements(NamedTuple):
    """One placement per lane, read from the Region fixture when it loads."""

    agent_xz: jax.Array  # float32[B, 2]
    #: Facing the goal, as ``pursuit_run._facing_reset`` faces its target.
    agent_yaw: jax.Array  # float32[B], degrees
    #: The traversal destination node, the walk_to goal.
    goal_xyz: jax.Array  # float32[B, 3]
    #: Entity 1 stands on the goal; it is inert.
    other_xyz: jax.Array  # float32[B, 3]


@dataclass(frozen=True)
class RegionTaskEnvironment:
    """One compiled Region environment whose reset takes the lanes' placements."""

    batch: int
    params: Any
    config: Any
    environment: Any
    task_placements: TaskPlacements
    #: The floor height of each lane's start, from its placement.
    floor_y: jax.Array  # float32[B]
    _reset: Callable[..., ArsenalEnvironmentCarry]
    _step: Callable[..., Any]

    def reset(
        self,
        keys: jax.Array,
        agent_xz: jax.Array,
        agent_yaw: jax.Array,
        other_xyz: jax.Array,
    ) -> ArsenalEnvironmentCarry:
        """Spawn the learner (entity 0) and entity 1 on their placements; yaw is the caller's.

        ``keys`` is [B] PRNG keys, ``agent_xz`` [B, 2] and ``other_xyz`` [B, 3] must equal the lane's entries of
        ``task_placements`` (the Region fixes its lanes; a different spawn is refused when the arrays are concrete),
        ``agent_yaw`` [B] degrees of body heading.
        """

        agent_xz = jnp.asarray(agent_xz, dtype=jnp.float32)
        agent_yaw = jnp.asarray(agent_yaw, dtype=jnp.float32)
        other_xyz = jnp.asarray(other_xyz, dtype=jnp.float32)
        if agent_xz.shape != (self.batch, 2):
            raise ValueError(f"agent_xz must have shape ({self.batch}, 2)")
        if agent_yaw.shape != (self.batch,):
            raise ValueError(f"agent_yaw must have shape ({self.batch},)")
        if other_xyz.shape != (self.batch, 3):
            raise ValueError(f"other_xyz must have shape ({self.batch}, 3)")
        if not any(isinstance(value, jax.core.Tracer) for value in (agent_xz, other_xyz)):
            _check_placed(agent_xz, self.task_placements.agent_xz, "agent_xz")
            _check_placed(other_xyz, self.task_placements.other_xyz, "other_xyz")
        return self._reset(keys, agent_xz, agent_yaw, other_xyz)

    def step(self, carry: ArsenalEnvironmentCarry, factors: jax.Array, keys: jax.Array):
        """Advance one decision tick with int32[B, 12] factors. Returns (carry, observation, info)."""

        carry, observation, _reward, _done, info = self._step(carry, factors, keys)
        return carry, observation, info


def _check_placed(given: jax.Array, placed: jax.Array, name: str) -> None:
    if not np.allclose(np.asarray(given), np.asarray(placed), atol=PLACEMENT_TOLERANCE):
        raise ValueError(
            f"{name} must be the lane's placement from env.task_placements: the Region fixes its lanes when it loads"
        )


def _load_fixture(library: str, batch: int, params, config, separation_range, seed: int, working_set: int):
    """The benchmark's Region fixture for ``library``, with the npc_walk params and the legal runtime."""

    resolved = libraries.get(library)
    if resolved is None:
        known = ", ".join(item.id for item in libraries.discover()) or "none discovered"
        raise ValueError(f"unknown Region library {library!r}; known: {known}")
    region_manifest, traversal_manifest, _block_semantics = world_region._library_manifests(resolved.path)
    absent = world_region.missing((region_manifest, traversal_manifest, world_region.FIXTURE_BUILDER))
    if absent:
        raise FileNotFoundError(f"the Region corpus is incomplete: {absent}")
    return world_region.loader().load_arsenal_region_fixture(
        region_manifest=region_manifest,
        traversal_manifest=traversal_manifest,
        expected_region_sha256=world_region._identity(region_manifest),
        expected_traversal_sha256=world_region._identity(traversal_manifest),
        batch_size=batch,
        selection_key=world_region.SELECTION_KEY,
        assignment_key=world_region.ASSIGNMENT_KEY,
        node_seed=seed,
        params=params,
        runtime=config,
        target_separation_range=separation_range,
        require_initial_line_of_sight=True,
        working_set_capacity=working_set,
        environment_diversity=True,
        action_surface_config=None,
    )


def make_region_task_environment(
    batch: int,
    maximum_turn_degrees: float,
    *,
    separation_range: tuple[float, float] = DEFAULT_SEPARATION_RANGE,
    seed: int = 0,
    library: str = LIBRARY,
    working_set_capacity: int = WORKING_SET_CAPACITY,
) -> RegionTaskEnvironment:
    """Build the npc_walk Region environment with ``batch`` lanes, one placement each, from ``seed``.

    ``separation_range`` is the band of start-to-goal distances in blocks. ``seed`` picks the placements (the fixture's
    node seed) and ``library`` the terrain.
    """

    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError(f"batch must be a positive int, got {batch!r}")
    base = default_combat_params(microticks=1, target_active=False, motion_model="npc_walk")
    # The default band samples a motion timing profile at every reset; pin profile 0 for the 1/30 s tick (as the flat world).
    params = base._replace(
        motion_timing_profile_min=jnp.asarray(0, dtype=jnp.int32),
        motion_timing_profile_max=jnp.asarray(0, dtype=jnp.int32),
    )
    # Entity 1 has no loadout (as on the flat world) and target_active=False keeps it from acting.
    config = arsenal_runtime_config(
        hytale_0_5_7_loadouts([DEFAULT_LOADOUT] * batch), legal_observation=True
    )
    fixture = _load_fixture(library, batch, params, config, separation_range, seed, working_set_capacity)

    placed = fixture.reset_provider(jax.random.split(jax.random.key(seed), batch))
    agent_xyz = jnp.asarray(placed.agent_position, dtype=jnp.float32)
    goal_xyz = jnp.asarray(placed.target_position, dtype=jnp.float32)
    delta = goal_xyz[:, (0, 2)] - agent_xyz[:, (0, 2)]
    # Facing the goal, as pursuit_run's _facing_reset: atan2(-dx, -dz).
    agent_yaw = jnp.degrees(jnp.arctan2(-delta[:, 0], -delta[:, 1])).astype(jnp.float32)
    placements = TaskPlacements(
        agent_xz=agent_xyz[:, (0, 2)], agent_yaw=agent_yaw, goal_xyz=goal_xyz, other_xyz=goal_xyz
    )
    floor_y = agent_xyz[:, 1]

    geometry = fixture.geometry_provider
    world_capabilities = fixture.world_capability_provider
    world_tokens = fixture.world_token_provider
    environment = make_arsenal_environment(
        fixture.params,
        config,
        maximum_turn_degrees=maximum_turn_degrees,
        geometry_provider=geometry,
        world_capability_provider=world_capabilities,
        world_token_provider=world_tokens,
    )
    plan = plan_arsenal_compilation(fixture.params, config)
    token_config = normalize_world_geometry_policy_config(None)

    def carry_from_runtime(runtime):
        """The carry ``environment.reset`` composes, with this world's capability and geometry-token providers."""

        capabilities = world_capabilities(runtime, fixture.params)
        tokens = (
            empty_world_geometry_policy_tokens(batch, token_config)
            if world_tokens is None
            else world_tokens(runtime, fixture.params)
        )
        observation = encode_learner_observation_v3(
            runtime,
            fixture.params,
            empty_injected_world_features(batch),
            capabilities,
            config,
            world_geometry=tokens,
        )
        surface = _EmptyArsenalActionSurfaceEvidence(marker=jnp.empty((batch, 0), dtype=jnp.bool_))
        action_runtime = ArsenalActionSurfaceRuntime(
            block_interactions=empty_block_interaction_state(batch, entity_count=runtime.combat.health.shape[1]),
            world=(),
        )
        action_context = learner_arsenal_action_context(observation)
        action_mask = arsenal_policy_action_context_mask(
            action_context,
            surface.block_candidates,
            surface.recipe_candidates,
            use_available=surface.use_available,
            block_trigger_available=surface.block_trigger_available,
            interaction_movement=interaction_movement_constraints(
                action_runtime.block_interactions, actor_index=AGENT_ENTITY
            ),
        )
        return ArsenalEnvironmentCarry(
            runtime=runtime,
            action_context=action_context,
            action_surface=surface,
            action_surface_runtime=action_runtime,
            light_tokens=empty_actor_light_policy_tokens(batch, token_config.token_capacity),
            inventory_tokens=inventory_policy_tokens_from_state(
                runtime.inventory, config.inventory_layout, actor_valid=observation.valid
            ),
            world_capabilities=capabilities,
            action_mask=action_mask,
        )

    def reset_impl(keys, agent_xz, agent_yaw, other_xyz):
        entity_position = jnp.stack(
            (
                jnp.stack((agent_xz[:, 0], floor_y, agent_xz[:, 1]), axis=1),
                other_xyz,
            ),
            axis=1,
        )
        entity_health = jnp.stack(
            (
                jnp.full((batch,), fixture.params.agent_max_health, dtype=jnp.float32),
                jnp.full((batch,), fixture.params.target_max_health, dtype=jnp.float32),
            ),
            axis=1,
        )
        entity_yaw = jnp.stack((agent_yaw, jnp.zeros((batch,), dtype=jnp.float32)), axis=1)
        runtime, _ = reset_arsenal_batch(
            keys,
            fixture.params,
            config,
            geometry,
            entity_position=entity_position,
            entity_health=entity_health,
            entity_yaw=entity_yaw,
            projectile_capacity=plan.projectile_capacity,
            area_capacity=plan.area_capacity,
        )
        # The head is steered within a window of the body; a body turned by the spawn yaw would otherwise start with
        # its head at the window's edge (pursuit_run._facing_reset sets the head for the same reason).
        runtime = runtime._replace(combat=runtime.combat._replace(agent_head_yaw=agent_yaw))
        return carry_from_runtime(runtime)

    return RegionTaskEnvironment(
        batch=batch,
        params=fixture.params,
        config=config,
        environment=environment,
        task_placements=placements,
        floor_y=floor_y,
        _reset=jax.jit(reset_impl),
        _step=jax.jit(environment.step_factors),
    )


__all__ = [
    "DEFAULT_SEPARATION_RANGE",
    "LIBRARY",
    "RegionTaskEnvironment",
    "TaskPlacements",
    "make_region_task_environment",
]
