"""Batched flat-world episodes for the two simple skills, walk_to and look_at.

The world is a flat floor at ``params.floor_y`` with no geometry provider. The learner is
entity 0 and moves under the simulator's ``npc_walk`` motion model, the body the game
server's NPC uses. Entity 1 is inert (``target_active=False``, no loadout) and only holds
a point. Every lane is an independent episode.

Units: positions in blocks, angles in degrees, time in decision ticks. With the motion
timing profile pinned to 0 one tick is 1/30 s.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.compilation import plan_arsenal_compilation
from hytalegym.jax.combat.arsenal.environment import (
    ArsenalActionSurfaceRuntime,
    ArsenalEnvironment,
    ArsenalEnvironmentCarry,
    _EmptyArsenalActionSurfaceEvidence,
    fail_closed_arsenal_world_capabilities,
    make_arsenal_environment,
)
from hytalegym.jax.combat.arsenal.profiles.hytale_0_5_7 import hytale_0_5_7_loadouts
from hytalegym.jax.combat.arsenal.runtime import arsenal_runtime_config, reset_arsenal_batch
from hytalegym.jax.combat.arsenal.schema.types import ArsenalRuntimeConfig
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
from hytalegym.jax.combat.types import (
    AGENT_ENTITY,
    GAIT_RUN,
    GAIT_SNEAK,
    GAIT_WALK,
    PLAYER_GAIT_FORWARD_SPEED_MULTIPLIERS,
    CombatParams,
    CombatState,
    default_combat_params,
)

TICKS_PER_SECOND = 30.0
#: ``agent.effective_eye_height`` in the 0.5.7 NPC ruleset.
AGENT_EYE_HEIGHT = 1.0
#: ``agent.max_speed`` in the 0.5.7 NPC ruleset: the run gait's base speed under npc_walk.
NPC_WALK_BASE_SPEED = 5.0
#: The melee profile ``experimental/neural_agent/tasks/flat_arena.py`` uses for its learner.
DEFAULT_LOADOUT = "iron_sword"
#: The fixture spawn (``fixture.agent_spawn``) in XZ, and the other entity at its offset.
DEFAULT_SPAWN_XZ = (0.5, 0.5)
DEFAULT_OTHER_XYZ = (0.5, 65.0, -2.0)
DEFAULT_EYE_XYZ = (0.5, 66.0, 0.5)

WALK_DISTANCE_RANGE = (3.0, 15.0)
WALK_RADIUS_RANGE = (0.5, 1.5)
#: Walk, run and sneak. Sprint is left out for now.
WALK_GAITS = (GAIT_WALK, GAIT_RUN, GAIT_SNEAK)
WALK_HORIZON_SLACK = 1.5
WALK_HORIZON_PAD_TICKS = 30

LOOK_DISTANCE_RANGE = (2.0, 10.0)
LOOK_YAW_RANGE_DEGREES = 40.0
LOOK_PITCH_RANGE_DEGREES = 30.0
LOOK_TOLERANCE_DEGREES = 3.0
LOOK_HORIZON_TICKS = 90

#: The 12 policy factors (``ARSENAL_POLICY_ACTION_HEAD_SIZES``): base, ability, guard, jump,
#: locomotion, yaw, body yaw, pitch, hotbar, use, block trigger, block candidate. The
#: locomotion choice 9 is run, forward (``split_locomotion_choice(9) == (GAIT_RUN, 1, 0)``).
#: Yaw and body yaw bin 4 and pitch bin 2 are the zero deltas of their 9- and 5-bin heads.
NEUTRAL_FACTORS = (0, 0, 0, 0, 0, 4, 4, 2, 0, 0, 0, 0)
RUN_FORWARD_FACTORS = (0, 0, 0, 0, 9, 4, 4, 2, 0, 0, 0, 0)


class WalkToGoals(NamedTuple):
    target_xz: jax.Array  # f32[B, 2]
    radius: jax.Array  # f32[B]
    gait: jax.Array  # int32[B]
    horizon_ticks: jax.Array  # int32[B]


class LookAtGoals(NamedTuple):
    target_xyz: jax.Array  # f32[B, 3]
    tolerance_degrees: jax.Array  # f32[B]
    horizon_ticks: jax.Array  # int32[B]


class EpisodeResult(NamedTuple):
    #: True where the goal held at some tick within the lane's horizon.
    success: jax.Array  # bool[B]
    #: The 1-based decision tick at which the goal first held, or -1 if never.
    first_success_tick: jax.Array  # int32[B]
    #: The combat state after the last tick.
    combat: CombatState


def _wrap_degrees(value: jax.Array) -> jax.Array:
    return jnp.mod(value + 180.0, 360.0) - 180.0


def _carry_from_runtime(runtime, params: CombatParams, config: ArsenalRuntimeConfig):
    """Compose the carry the environment's own reset composes, for an explicit runtime.

    ``ArsenalEnvironment.reset`` takes only keys (``environment.py:1625``), so an explicit
    spawn cannot go through it. This mirrors that closure with no providers
    (``reset`` at ``environment.py:1625-1689``, ``compose_environment_state`` at
    ``environment.py:1584-1623``). ``FlatTaskEnvironment.reset`` is the only caller. The
    test that compares this carry with ``environment.reset`` at the default spawn keeps the
    two from drifting apart.
    """

    batch, entity_count = runtime.combat.health.shape
    token_config = normalize_world_geometry_policy_config(None)
    capabilities = fail_closed_arsenal_world_capabilities(runtime, params)
    features = empty_injected_world_features(batch)
    tokens = empty_world_geometry_policy_tokens(batch, token_config)
    light_tokens = empty_actor_light_policy_tokens(batch, token_config.token_capacity)
    action_runtime = ArsenalActionSurfaceRuntime(
        block_interactions=empty_block_interaction_state(batch, entity_count=entity_count),
        world=(),
    )
    observation = encode_learner_observation_v3(
        runtime, params, features, capabilities, config, world_geometry=tokens
    )
    surface = _EmptyArsenalActionSurfaceEvidence(marker=jnp.empty((batch, 0), dtype=jnp.bool_))
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
        light_tokens=light_tokens,
        inventory_tokens=inventory_policy_tokens_from_state(
            runtime.inventory, config.inventory_layout, actor_valid=observation.valid
        ),
        world_capabilities=capabilities,
        action_mask=action_mask,
    )


@dataclass(frozen=True)
class FlatTaskEnvironment:
    """One compiled flat-world environment whose reset takes explicit per-lane spawns."""

    batch: int
    params: CombatParams
    config: ArsenalRuntimeConfig
    environment: ArsenalEnvironment
    _reset: Callable[..., ArsenalEnvironmentCarry]

    def reset(
        self,
        keys: jax.Array,
        agent_xz: jax.Array,
        agent_yaw: jax.Array,
        other_xyz: jax.Array,
    ) -> ArsenalEnvironmentCarry:
        """Spawn the learner (entity 0) and the other entity (entity 1) per lane.

        ``keys`` is [B] PRNG keys, ``agent_xz`` is [B, 2] blocks, ``agent_yaw`` is [B]
        degrees of body heading, and ``other_xyz`` is [B, 3]. The learner stands on
        ``params.floor_y``. The other entity starts at full health with yaw 0.
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
        return self._reset(keys, agent_xz, agent_yaw, other_xyz)

    def step(self, carry: ArsenalEnvironmentCarry, factors: jax.Array, keys: jax.Array):
        """Advance one decision tick with int32[B, 12] factors. Returns (carry, observation, info)."""

        carry, observation, _reward, _done, info = self.environment.step_factors(
            carry, factors, keys
        )
        return carry, observation, info


def make_flat_npc_environment(
    batch: int,
    *,
    legal_observation: bool = True,
    maximum_turn_degrees: float = 45.0,
    loadout_profile: str = DEFAULT_LOADOUT,
) -> FlatTaskEnvironment:
    """Build the flat npc_walk environment with ``batch`` independent lanes."""

    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError(f"batch must be a positive int, got {batch!r}")
    base = default_combat_params(microticks=1, target_active=False, motion_model="npc_walk")
    # The default band samples a motion timing profile from 0..2 at every reset
    # (reset.py:123-127). Profiles 1 and 2 alternate in a 0.045 s loaded tick
    # (runtime/math.py:83-97). Pinning profile 0 gives the 1/30 s tick the horizons assume.
    params = base._replace(
        motion_timing_profile_min=jnp.asarray(0, dtype=jnp.int32),
        motion_timing_profile_max=jnp.asarray(0, dtype=jnp.int32),
    )
    # Entity 1 gets the empty profile (no loadout, as make_scene does with opponent=None),
    # and target_active=False keeps it from acting.
    loadouts = hytale_0_5_7_loadouts([loadout_profile] * batch)
    config = arsenal_runtime_config(loadouts, legal_observation=legal_observation)
    environment = make_arsenal_environment(
        params, config, maximum_turn_degrees=maximum_turn_degrees
    )
    plan = plan_arsenal_compilation(params, config)

    def reset_impl(keys, agent_xz, agent_yaw, other_xyz):
        agent_y = jnp.full((batch, 1), params.floor_y, dtype=jnp.float32)
        agent_xyz = jnp.concatenate((agent_xz[:, :1], agent_y, agent_xz[:, 1:]), axis=1)
        entity_position = jnp.stack((agent_xyz, other_xyz), axis=1)
        entity_health = jnp.stack(
            (
                jnp.full((batch,), params.agent_max_health, dtype=jnp.float32),
                jnp.full((batch,), params.target_max_health, dtype=jnp.float32),
            ),
            axis=1,
        )
        entity_yaw = jnp.stack((agent_yaw, jnp.zeros((batch,), dtype=jnp.float32)), axis=1)
        runtime, _ = reset_arsenal_batch(
            keys,
            params,
            config,
            None,
            entity_position=entity_position,
            entity_health=entity_health,
            entity_yaw=entity_yaw,
            projectile_capacity=plan.projectile_capacity,
            area_capacity=plan.area_capacity,
        )
        return _carry_from_runtime(runtime, params, config)

    return FlatTaskEnvironment(
        batch=batch,
        params=params,
        config=config,
        environment=environment,
        _reset=jax.jit(reset_impl),
    )


class WalkToTasks:
    """walk_to: walk to a world point at a pace and stop inside a radius."""

    @staticmethod
    def sample(
        key: jax.Array, batch: int, *, spawn_xz: tuple[float, float] = DEFAULT_SPAWN_XZ
    ) -> WalkToGoals:
        """Goals 3 to 15 blocks from ``spawn_xz`` in a random direction."""

        key_distance, key_angle, key_radius, key_gait = jax.random.split(key, 4)
        distance = jax.random.uniform(
            key_distance, (batch,), dtype=jnp.float32,
            minval=WALK_DISTANCE_RANGE[0], maxval=WALK_DISTANCE_RANGE[1],
        )
        angle = jax.random.uniform(
            key_angle, (batch,), dtype=jnp.float32, minval=0.0, maxval=2.0 * math.pi
        )
        direction = jnp.stack((jnp.sin(angle), jnp.cos(angle)), axis=1)
        spawn = jnp.asarray(spawn_xz, dtype=jnp.float32)
        target_xz = spawn[None, :] + distance[:, None] * direction
        radius = jax.random.uniform(
            key_radius, (batch,), dtype=jnp.float32,
            minval=WALK_RADIUS_RANGE[0], maxval=WALK_RADIUS_RANGE[1],
        )
        gait_table = jnp.asarray(WALK_GAITS, dtype=jnp.int32)
        gait = gait_table[jax.random.randint(key_gait, (batch,), 0, len(WALK_GAITS))]
        multiplier = jnp.asarray(PLAYER_GAIT_FORWARD_SPEED_MULTIPLIERS, dtype=jnp.float32)[gait]
        horizon_ticks = (
            jnp.ceil(
                distance / (NPC_WALK_BASE_SPEED * multiplier)
                * WALK_HORIZON_SLACK * TICKS_PER_SECOND
            ).astype(jnp.int32)
            + WALK_HORIZON_PAD_TICKS
        )
        return WalkToGoals(
            target_xz=target_xz, radius=radius, gait=gait, horizon_ticks=horizon_ticks
        )

    @staticmethod
    def success(combat: CombatState, goals: WalkToGoals) -> jax.Array:
        """True where the learner's planar distance to the target is within the radius."""

        planar = combat.position[:, AGENT_ENTITY][:, [0, 2]]
        offset = planar - goals.target_xz
        return jnp.hypot(offset[:, 0], offset[:, 1]) <= goals.radius


class LookAtTasks:
    """look_at: turn the head to face a point."""

    @staticmethod
    def sample(
        key: jax.Array,
        batch: int,
        *,
        eye_xyz: tuple[float, float, float] = DEFAULT_EYE_XYZ,
        body_yaw_degrees: float = 0.0,
    ) -> LookAtGoals:
        """Points 2 to 10 blocks from the eye, within +-40 degrees of the body yaw and +-30 of pitch."""

        key_distance, key_yaw, key_pitch = jax.random.split(key, 3)
        distance = jax.random.uniform(
            key_distance, (batch,), dtype=jnp.float32,
            minval=LOOK_DISTANCE_RANGE[0], maxval=LOOK_DISTANCE_RANGE[1],
        )
        body_yaw = jnp.broadcast_to(jnp.asarray(body_yaw_degrees, dtype=jnp.float32), (batch,))
        yaw_offset = jax.random.uniform(
            key_yaw, (batch,), dtype=jnp.float32,
            minval=-LOOK_YAW_RANGE_DEGREES, maxval=LOOK_YAW_RANGE_DEGREES,
        )
        pitch = jax.random.uniform(
            key_pitch, (batch,), dtype=jnp.float32,
            minval=-LOOK_PITCH_RANGE_DEGREES, maxval=LOOK_PITCH_RANGE_DEGREES,
        )
        yaw = jnp.deg2rad(body_yaw + yaw_offset)
        pitch = jnp.deg2rad(pitch)
        horizontal = distance * jnp.cos(pitch)
        # The inverse of the bearing and pitch in ``success``, and of
        # ``projectile_look_direction`` (arsenal/projectiles/runtime.py:2267-2275).
        direction = jnp.stack(
            (-jnp.sin(yaw) * horizontal, distance * jnp.sin(pitch), -jnp.cos(yaw) * horizontal),
            axis=1,
        )
        target_xyz = jnp.asarray(eye_xyz, dtype=jnp.float32)[None, :] + direction
        return LookAtGoals(
            target_xyz=target_xyz,
            tolerance_degrees=jnp.full((batch,), LOOK_TOLERANCE_DEGREES, dtype=jnp.float32),
            horizon_ticks=jnp.full((batch,), LOOK_HORIZON_TICKS, dtype=jnp.int32),
        )

    @staticmethod
    def success(combat: CombatState, goals: LookAtGoals) -> jax.Array:
        """True where the head's yaw and pitch are within tolerance of facing the target."""

        eye = combat.position[:, AGENT_ENTITY] + jnp.asarray(
            [0.0, AGENT_EYE_HEIGHT, 0.0], dtype=jnp.float32
        )
        offset = goals.target_xyz - eye
        dx, dy, dz = offset[:, 0], offset[:, 1], offset[:, 2]
        yaw_facing = jnp.rad2deg(jnp.arctan2(-dx, -dz))
        pitch_facing = jnp.rad2deg(jnp.arctan2(dy, jnp.hypot(dx, dz)))
        yaw_error = _wrap_degrees(combat.agent_head_yaw - yaw_facing)
        pitch_error = combat.agent_head_pitch - pitch_facing
        return (jnp.abs(yaw_error) <= goals.tolerance_degrees) & (
            jnp.abs(pitch_error) <= goals.tolerance_degrees
        )


def run_episode(
    env: FlatTaskEnvironment,
    tasks_goals: WalkToGoals | LookAtGoals,
    policy_fn: Callable[[CombatState, WalkToGoals | LookAtGoals], jax.Array],
    key: jax.Array,
    max_ticks: int,
    *,
    carry: ArsenalEnvironmentCarry | None = None,
) -> EpisodeResult:
    """Run ``max_ticks`` decision ticks on every lane and record the first tick the goal holds.

    ``policy_fn(combat, goals)`` returns int32[B, 12] factors. A lane counts as a success only
    at ticks no later than its ``horizon_ticks``. With ``carry=None`` every lane resets at the
    default spawn (``DEFAULT_SPAWN_XZ``, yaw 0, the other entity at ``DEFAULT_OTHER_XYZ``), so
    the goals must be sampled from that spawn. Pass a carry from ``env.reset`` otherwise.
    """

    if isinstance(max_ticks, bool) or not isinstance(max_ticks, int) or max_ticks < 1:
        raise ValueError(f"max_ticks must be a positive int, got {max_ticks!r}")
    if isinstance(tasks_goals, WalkToGoals):
        success_fn = WalkToTasks.success
    elif isinstance(tasks_goals, LookAtGoals):
        success_fn = LookAtTasks.success
    else:
        raise TypeError("tasks_goals must be WalkToGoals or LookAtGoals")

    reset_key, step_key = jax.random.split(key)
    if carry is None:
        batch = env.batch
        carry = env.reset(
            jax.random.split(reset_key, batch),
            jnp.broadcast_to(jnp.asarray(DEFAULT_SPAWN_XZ, dtype=jnp.float32), (batch, 2)),
            jnp.zeros((batch,), dtype=jnp.float32),
            jnp.broadcast_to(jnp.asarray(DEFAULT_OTHER_XYZ, dtype=jnp.float32), (batch, 3)),
        )
    step_keys = jax.random.split(step_key, max_ticks)
    first = jnp.full((env.batch,), -1, dtype=jnp.int32)
    for tick in range(max_ticks):
        factors = jnp.asarray(policy_fn(carry.runtime.combat, tasks_goals), dtype=jnp.int32)
        carry, _observation, _info = env.step(
            carry, factors, jax.random.split(step_keys[tick], env.batch)
        )
        held = success_fn(carry.runtime.combat, tasks_goals) & (
            tasks_goals.horizon_ticks >= tick + 1
        )
        first = jnp.where((first < 0) & held, jnp.int32(tick + 1), first)
    return EpisodeResult(
        success=first >= 0,
        first_success_tick=first,
        combat=carry.runtime.combat,
    )
