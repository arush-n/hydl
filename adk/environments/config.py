"""One place to declare a scene: weapon, opponent, terrain, reward, shaping.

Every probe in this project has been a hand-rolled `make_arsenal_ppo_environment`
call, and each one silently re-made the same mistakes -- `open_flat` treated as
production, batch treated as environment diversity, an opponent that cannot
land a hit treated as a control. This puts those defaults in one place with the
measurement behind each one, so a scene is declared rather than assembled.

    from adk.environments.config import SceneConfig
    scene = SceneConfig(weapon="iron_daggers", task="duelist").build()
    state, obs, mask = scene.reset(jax.random.key(0))

`SceneConfig` is frozen; use `replace(scene, weapon=...)` for sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

import jax
import jax.numpy as jnp

__all__ = ["SceneConfig", "BuiltScene", "replace", "OPPONENT_CLOSES"]

#: Opponent standoff that actually lets it fight. The ruleset default is 2.8
#: (`target.maintain_distance.desired_distance_min/_max`), which is tuned for
#: the legacy Trork battleaxe (authored reach 0.1->3.0, 23.0 damage). An arsenal
#: opponent attacks through the ability system, whose melee damage peaks near
#: 0.83 -- so at 2.8 it parks outside its own reach. Measured over 250 ticks x
#: batch 4, agent idle:
#:
#:     2.8 (default) ->  50.0 damage dealt to the agent, 0 deaths
#:     2.0           -> 420.0 damage, 4/4 deaths
#:     1.5           -> 420.0 damage, 4/4 deaths
#:
#: Anything measuring defence against the 2.8 default is measuring an opponent
#: that essentially cannot hurt you.
OPPONENT_CLOSES = 2.0


@dataclass(frozen=True)
class BuiltScene:
    environment: Any
    params: Any
    runtime: Any
    step: Callable
    reset: Callable
    config: "SceneConfig"
    #: What was actually bound, for the record a result should carry with it.
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneConfig:
    """A declared scene. Every default here is a measurement, not a guess."""

    # --- equipment ---------------------------------------------------------
    #: Any of the 223 authored profiles. All have >=1 ability (range 1-11);
    #: 114 can guard. Reach differs per weapon, so a spacing band tuned for one
    #: is wrong for another.
    weapon: str = "iron_sword"
    opponent_weapon: str = "iron_sword"

    # --- who the opponent is ----------------------------------------------
    #: `None` keeps the Gym default (`first_legal_opponent_ability_slots`).
    #: See `adk.scenarios.npcs` for alternatives.
    opponent_policy: Callable | None = None
    #: Close the standoff so the opponent can actually reach. Set `None` to keep
    #: the ruleset's 2.8 -- only do that when you *want* a near-harmless target,
    #: and say so in the result.
    opponent_distance: float | None = OPPONENT_CLOSES
    #: Disarm entirely. This is the real inert control; note that at the 2.8
    #: default an "armed" opponent is nearly indistinguishable from this.
    inert_opponent: bool = False

    # --- world -------------------------------------------------------------
    #: "open_flat" -- permissive control, and what the Gym's own readiness gates
    #:               use. Ability head {0,1,2,3,4}, dodge {0,3,4}.
    #: "fail_closed" -- the default when no provider is passed. Denies LOS,
    #:               target selection, projectile flight. Ability head is {0}:
    #:               **no attack is possible at all**. Rarely what you want.
    #: "region"   -- real captured terrain via geometry + world runtime
    #:               providers. Ability {0,1,2,3,4}, dodge {0,3,4} -- i.e. the
    #:               same combat surface as open_flat, on real geometry.
    world: str = "open_flat"
    #: Region only. Batch does NOT vary terrain -- all rows are byte-identical
    #: (0 of 8271 observation columns differ). Varying this does: three seeds
    #: gave 3/3 distinct spawns and 1493-1530 differing columns.
    #:
    #: This picks a node **inside whichever region is already selected**. It
    #: does not pick the region -- `selection_key` does. Setting only this
    #: applies the seed to the default region.
    node_seed: int | None = None
    #: Region only. Chooses *which* captured region, by ranking the train split
    #: on `sha256(f"sha256_rank_v1\0{key}\0{semantic_sha256}")`. All 16 train
    #: regions are reachable below 89; heldout regions are not candidates.
    selection_key: int | None = None
    #: Authored geometry from `adk.environments.frames` -- an arena name, an
    #: `Arena`, or a prebuilt `GeometryState`. Gives the scene real cells
    #: (collision, support, fluid) instead of `synthetic.py`'s fabricated query
    #: answers. Ignored when `world="region"`, which brings its own geometry.
    geometry: object | None = None

    #: Region only. `(region_seed, component)` from `worlds.zones`.
    #: Resolves both keys above, so the agent spawns in a mutually-reachable
    #: arena instead of a random node that may sit in a one-way pocket -- the
    #: largest such pocket is only 35.7% of a region. Overrides `node_seed` and
    #: `selection_key` when set.
    zone: tuple[int, int] | None = None

    # --- episode -----------------------------------------------------------
    num_envs: int = 8
    microticks: int = 1
    target_active: bool = True

    # --- movement tuning ---------------------------------------------------
    #: Any `CombatParams` float field, applied by `._replace()`. Defaults are
    #: the ruleset's; these are the ones worth knowing:
    #:
    #:   agent_jump_velocity            5.099
    #:   agent_max_speed                5.0
    #:   agent_acceleration            10.0
    #:   agent_turn_speed_degrees     540.0
    #:   world_gravity                 32.0
    #:   agent_walk_max_climb_height    1.3   auto step-up; ledges under this
    #:                                        need no jump and no mantle
    #:   agent_walk_max_drop_height     3.0
    #:   target_max_speed               8.0   <-- 60% FASTER than the agent
    #:   target_chase_speed             6.4
    #:
    #: That last pair matters: the agent cannot outrun the opponent, so any
    #: flee objective has to be won with terrain or timing, not speed.
    #:
    #: Mantling is deliberately absent. `MantlingPlugin` registers
    #: `ClientFeature.Mantling` and nothing else -- it is a client feature, and
    #: `MovementStates.mantling` is a flag the server receives rather than a
    #: behaviour it drives. A headless agent has no way to trigger it.
    movement: dict[str, float] = field(default_factory=dict)

    # --- objective ---------------------------------------------------------
    #: Name from `adk.scenarios.tasks.TASKS`; reweights the four native terms.
    task: str = "baseline"
    #: Name from `adk.scenarios.minigames.REGISTRY`; adds a shaped term.
    minigame: str | None = None
    minigame_weight: float = 1.0
    minigame_kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self) -> BuiltScene:
        from hytalegym.jax.combat.arsenal import (
            arsenal_runtime_config, hytale_0_5_7_loadouts,
            fail_closed_arsenal_world_capabilities,
            open_flat_arsenal_world_capabilities)
        from hytalegym.jax.combat.opponents.runtime.policy import (
            inert_opponent_ability_slots)
        from hytalegym.jax.combat.types import default_combat_params
        from hytalegym.jax.training.arsenal import make_arsenal_ppo_environment

        from ..scenarios import minigames as minigame_registry
        from ..scenarios import tasks as task_registry

        if self.task not in task_registry.TASKS:
            raise ValueError(
                f"unknown task {self.task!r}; have "
                f"{sorted(task_registry.TASKS)}")
        if self.minigame and self.minigame not in minigame_registry.REGISTRY:
            raise ValueError(
                f"unknown minigame {self.minigame!r}; have "
                f"{sorted(minigame_registry.REGISTRY)}")

        params = default_combat_params(
            microticks=self.microticks, target_active=self.target_active)
        params = task_registry.TASKS[self.task].params(params)
        if self.opponent_distance is not None:
            params = params._replace(
                target_maintain_desired_distance_min=jnp.float32(
                    self.opponent_distance),
                target_maintain_desired_distance_max=jnp.float32(
                    self.opponent_distance))
        if self.movement:
            unknown = set(self.movement) - set(params._fields)
            if unknown:
                raise ValueError(
                    f"unknown CombatParams movement field(s): {sorted(unknown)}")
            params = params._replace(**{
                key: jnp.asarray(value, dtype=jnp.float32)
                for key, value in self.movement.items()})

        provenance: dict[str, Any] = {
            "weapon": self.weapon, "opponent_weapon": self.opponent_weapon,
            "world": self.world, "task": self.task,
            "minigame": self.minigame, "num_envs": self.num_envs,
            "opponent_distance": self.opponent_distance,
            "inert_opponent": self.inert_opponent,
        }

        if self.world == "region":
            from ..validation import region as region_module

            node_seed, selection_key = self.node_seed, self.selection_key
            if self.zone is not None:
                from worlds.zones import seed_for, selection_for, zones as zone_index

                region_seed, component = self.zone
                match = next((z for z in zone_index(region_seed)
                              if z.component == component), None)
                if match is None:
                    raise ValueError(
                        f"region {region_seed} has no navigable zone "
                        f"{component}; run `python -m worlds.zones "
                        f"{region_seed}` to list them")
                selection_key = selection_for(region_seed)
                if selection_key is None:
                    raise ValueError(
                        f"region {region_seed} is not in the train split, so "
                        "no selection_key can reach it")
                node_seed = seed_for(match)
                if node_seed is None:
                    raise ValueError(
                        f"no node_seed found landing in zone {component} of "
                        f"region {region_seed}")

            kwargs: dict[str, Any] = {}
            if node_seed is not None:
                kwargs["node_seed"] = node_seed
            if selection_key is not None:
                kwargs["selection_key"] = selection_key
            scene = region_module.region_scene(
                weapons=(self.weapon,) * self.num_envs, native_evidence=False,
                **kwargs)
            environment, runtime = scene.environment, scene.runtime
            params = scene.params
            # `region_artifact` is the fixture's own identity. Confirm the zone
            # landed against this, never against the spawn position -- all 20
            # captures share a world origin, so a coordinate is a valid node in
            # several regions at once.
            provenance.update(
                node_seed=node_seed, selection_key=selection_key,
                zone=self.zone, reach=scene.maximum_distance,
                region_seams=scene.bound,
                region_artifact=scene.fixture.metadata.get("artifact_seed"))
        else:
            providers = {
                "open_flat": open_flat_arsenal_world_capabilities,
                "fail_closed": fail_closed_arsenal_world_capabilities,
            }
            # Synthetic arenas declare terrain as a function of position rather
            # than loading a capture. `world` may be a name or the world itself.
            from .synthetic import PRESETS, SyntheticWorld
            if isinstance(self.world, SyntheticWorld):
                world_key = self.world.name
                providers = {**providers, world_key: self.world.provider()}
            else:
                world_key = self.world
                providers = {**providers,
                             **{n: w.provider() for n, w in PRESETS.items()}}
            if world_key not in providers:
                raise ValueError(
                    f"unknown world {self.world!r}; "
                    f"have {sorted(providers)} or 'region'")
            runtime = arsenal_runtime_config(hytale_0_5_7_loadouts(
                [self.weapon] * self.num_envs,
                target_profiles=[self.opponent_weapon] * self.num_envs))
            extra = {}
            if self.inert_opponent:
                extra["opponent_ability_provider"] = inert_opponent_ability_slots
            elif self.opponent_policy is not None:
                extra["opponent_ability_provider"] = self.opponent_policy
            if self.geometry is not None:
                from .frames import Arena, build as build_geometry
                geometry = self.geometry
                if isinstance(geometry, (str, Arena)):
                    geometry = build_geometry(
                        geometry, batch_size=self.num_envs)
                extra["geometry_provider"] = geometry
            environment = make_arsenal_ppo_environment(
                params, runtime,
                world_capability_provider=providers[world_key], **extra)

        step = jax.jit(environment.step_detailed)
        if self.minigame:
            game = minigame_registry.REGISTRY[self.minigame]
            step = task_registry.shaped(
                step, game.build(**self.minigame_kwargs),
                weight=self.minigame_weight)
            provenance["minigame_requires"] = list(game.requires)

        def reset(key):
            return environment.reset(jax.random.split(key, self.num_envs))

        return BuiltScene(environment=environment, params=params,
                          runtime=runtime, step=step, reset=reset,
                          config=self, provenance=provenance)
