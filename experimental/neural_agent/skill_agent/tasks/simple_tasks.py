"""The flat task world the simple skills train and are measured on, and ``run_episode``.

Each skill's goals and success test are in its own folder: ``neural_skills/walk_to/task.py`` and
``neural_skills/look_at/task.py``.

The world is a flat floor at ``params.floor_y`` with no geometry provider. The learner is
entity 0 and moves under the simulator's ``npc_walk`` motion model, the body the game
server's NPC uses. Entity 1 is inert (``target_active=False``, no loadout) and only holds
a point, unless ``make_flat_npc_environment(..., mover=...)`` walks it along a scripted ``Mover`` path. Every lane is
an independent episode.

Units: positions in blocks, angles in degrees, time in decision ticks. With the motion
timing profile pinned to 0 one tick is 1/30 s.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
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
    open_flat_arsenal_world_capabilities,
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
    PLAYER_GAIT_FORWARD_SPEED_MULTIPLIERS,
    TARGET_ENTITY,
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
#: One tick of the world in seconds: a mover moves ``speed * MOVER_TICK_SECONDS`` blocks a tick.
MOVER_TICK_SECONDS = 1.0 / TICKS_PER_SECOND
_MOVER_EPSILON = 1.0e-6

#: The 12 policy factors (``ARSENAL_POLICY_ACTION_HEAD_SIZES``): base, ability, guard, jump,
#: locomotion, yaw, body yaw, pitch, hotbar, use, block trigger, block candidate. The
#: locomotion choice 9 is run, forward (``split_locomotion_choice(9) == (GAIT_RUN, 1, 0)``).
#: Yaw and body yaw bin 4 and pitch bin 2 are the zero deltas of their 9- and 5-bin heads.
NEUTRAL_FACTORS = (0, 0, 0, 0, 0, 4, 4, 2, 0, 0, 0, 0)
RUN_FORWARD_FACTORS = (0, 0, 0, 0, 9, 4, 4, 2, 0, 0, 0, 0)


class EpisodeResult(NamedTuple):
    #: True where the goal held at some tick within the lane's horizon.
    success: jax.Array  # bool[B]
    #: The 1-based decision tick at which the goal first held, or -1 if never.
    first_success_tick: jax.Array  # int32[B]
    #: The combat state after the last tick.
    combat: CombatState


def _carry_from_runtime(runtime, params: CombatParams, config: ArsenalRuntimeConfig, capability_provider=None):
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
    capabilities = (fail_closed_arsenal_world_capabilities(runtime, params) if capability_provider is None
                    else capability_provider(runtime, params))
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


class Mover(NamedTuple):
    """A scripted path for entity 1, one per lane: the opt-in mover of ``make_flat_npc_environment``.

    ``waypoints`` is [B, K, 2] in x-z with K >= 2, ``speed`` is [B] in blocks a second, and ``loop`` is [B] bool: True
    where the path closes from its last point back to its first, False where the person stops at its last point.
    Entity 1 starts at ``waypoints[:, 0]`` and moves ``speed * MOVER_TICK_SECONDS`` along the path each tick.
    """

    waypoints: jax.Array
    speed: jax.Array
    loop: jax.Array


def gait_speed(gait: int) -> float:
    """A gait's ground speed in blocks a second: the npc_walk base speed times the gait's forward multiplier."""

    return NPC_WALK_BASE_SPEED * PLAYER_GAIT_FORWARD_SPEED_MULTIPLIERS[gait]


def gait_band(low_gait: int, high_gait: int | None = None) -> tuple[float, float]:
    """A speed band in blocks a second: one gait's speed, or the range between two gaits."""

    high_gait = low_gait if high_gait is None else high_gait
    speeds = (gait_speed(low_gait), gait_speed(high_gait))
    return min(speeds), max(speeds)


def mover_path(mover: Mover, ticks: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Entity 1's place on each lane's path after ``ticks`` (int [B]): the x-z position [B, 2], the yaw in degrees
    (the engine's yaw of the direction of travel) and the x-z velocity [B, 2] in blocks a second. A stopped path rests
    at its last point with zero velocity and keeps its last heading.
    """

    waypoints = mover.waypoints
    count = waypoints.shape[1]
    rows = jnp.arange(waypoints.shape[0])
    # Each point to the next; the last one closes back to the first (walked only by a loop).
    delta = jnp.roll(waypoints, -1, axis=1) - waypoints
    norm = jnp.linalg.norm(delta, axis=2)
    walked = mover.loop[:, None] | (jnp.arange(count) < count - 1)[None, :]
    length = jnp.where(walked, norm, 0.0)
    starts = jnp.cumsum(length, axis=1) - length
    total = jnp.sum(length, axis=1)
    travelled = mover.speed * MOVER_TICK_SECONDS * ticks.astype(jnp.float32)
    along = jnp.where(
        mover.loop,
        jnp.mod(travelled, jnp.maximum(total, _MOVER_EPSILON)),
        jnp.minimum(travelled, total),
    )
    # The segment that has started last; a stopped path never walks its closing segment.
    last_segment = jnp.where(mover.loop, count - 1, count - 2)
    segment = jnp.clip(jnp.sum(starts <= along[:, None], axis=1) - 1, 0, last_segment)
    seg_length = length[rows, segment]
    fraction = jnp.where(
        seg_length > 0.0,
        jnp.clip((along - starts[rows, segment]) / jnp.maximum(seg_length, _MOVER_EPSILON), 0.0, 1.0),
        0.0,
    )
    xz = waypoints[rows, segment] + fraction[:, None] * delta[rows, segment]
    heading = delta[rows, segment] / jnp.maximum(norm[rows, segment], _MOVER_EPSILON)[:, None]
    yaw = jnp.rad2deg(jnp.arctan2(-heading[:, 0], -heading[:, 1]))
    stopped = ~mover.loop & (travelled >= total)
    velocity = jnp.where(stopped[:, None], 0.0, heading * mover.speed[:, None])
    return xz, yaw, velocity


def _place_entity(combat: CombatState, mover: Mover, ticks: jax.Array) -> CombatState:
    """Put entity 1 where ``mover`` has it after ``ticks``: its x-z position, yaw and velocity. Its height and the rest
    of the state stay as they are."""

    xz, yaw, velocity = mover_path(mover, ticks)
    height = combat.position[:, TARGET_ENTITY, 1]
    position = combat.position.at[:, TARGET_ENTITY].set(jnp.stack((xz[:, 0], height, xz[:, 1]), axis=1))
    walk = jnp.stack((velocity[:, 0], jnp.zeros_like(height), velocity[:, 1]), axis=1)
    return combat._replace(
        position=position,
        velocity=combat.velocity.at[:, TARGET_ENTITY].set(walk),
        yaw=combat.yaw.at[:, TARGET_ENTITY].set(yaw),
    )


@jax.jit
def _move_carry(carry: ArsenalEnvironmentCarry, mover: Mover) -> ArsenalEnvironmentCarry:
    """The carry with entity 1 moved to its next place, before the step's physics and observation are computed."""

    runtime = carry.runtime
    combat = _place_entity(runtime.combat, mover, runtime.combat.tick_count + 1)
    return carry._replace(runtime=runtime._replace(combat=combat))


def _check_mover(mover: Mover | None, batch: int) -> None:
    if mover is None:
        return
    waypoints_shape = tuple(jnp.shape(mover.waypoints))
    if len(waypoints_shape) != 3 or waypoints_shape[0] != batch or waypoints_shape[2] != 2 or waypoints_shape[1] < 2:
        raise ValueError(f"mover.waypoints must have shape ({batch}, K, 2) with K >= 2, got {waypoints_shape}")
    if tuple(jnp.shape(mover.speed)) != (batch,) or tuple(jnp.shape(mover.loop)) != (batch,):
        raise ValueError(f"mover.speed and mover.loop must have shape ({batch},)")
    if jnp.asarray(mover.loop).dtype != jnp.bool_:
        raise ValueError("mover.loop must be a bool array")


def _draw_speed(key: jax.Array, batch: int, speed_band: tuple[float, float]) -> jax.Array:
    low, high = float(speed_band[0]), float(speed_band[1])
    if not 0.0 <= low <= high:
        raise ValueError(f"speed_band must be (low, high) with 0 <= low <= high, got {speed_band}")
    return low + (high - low) * jax.random.uniform(key, (batch,), dtype=jnp.float32)


def sample_straight_mover(
    key: jax.Array,
    batch: int,
    *,
    start_xz: tuple[float, float],
    speed_band: tuple[float, float],
    length_blocks: float = 60.0,
) -> Mover:
    """A straight walk from ``start_xz`` in a random heading, at a speed drawn per lane from ``speed_band``, for
    ``length_blocks`` and then a stop."""

    key_heading, key_speed = jax.random.split(key)
    heading = jax.random.uniform(key_heading, (batch,), dtype=jnp.float32, minval=0.0, maxval=2.0 * jnp.pi)
    direction = jnp.stack((jnp.sin(heading), jnp.cos(heading)), axis=1)
    start = jnp.broadcast_to(jnp.asarray(start_xz, dtype=jnp.float32), (batch, 2))
    waypoints = jnp.stack((start, start + length_blocks * direction), axis=1)
    return Mover(waypoints, _draw_speed(key_speed, batch, speed_band), jnp.zeros((batch,), dtype=jnp.bool_))


def sample_random_walk_mover(
    key: jax.Array,
    batch: int,
    *,
    speed_band: tuple[float, float],
    spawn_xz: tuple[float, float] = DEFAULT_SPAWN_XZ,
    waypoints: int = 8,
    radius: float = 8.0,
) -> Mover:
    """``waypoints`` points drawn uniformly in a disc of ``radius`` blocks around ``spawn_xz``, walked in order and
    looped, at a speed drawn per lane from ``speed_band``."""

    key_radius, key_angle, key_speed = jax.random.split(key, 3)
    distance = radius * jnp.sqrt(jax.random.uniform(key_radius, (batch, waypoints), dtype=jnp.float32))
    angle = jax.random.uniform(
        key_angle, (batch, waypoints), dtype=jnp.float32, minval=0.0, maxval=2.0 * jnp.pi
    )
    spawn = jnp.asarray(spawn_xz, dtype=jnp.float32)
    points = spawn + jnp.stack((distance * jnp.sin(angle), distance * jnp.cos(angle)), axis=2)
    return Mover(points, _draw_speed(key_speed, batch, speed_band), jnp.ones((batch,), dtype=jnp.bool_))


@dataclass(frozen=True)
class FlatTaskEnvironment:
    """One compiled flat-world environment whose reset takes explicit per-lane spawns."""

    batch: int
    params: CombatParams
    config: ArsenalRuntimeConfig
    environment: ArsenalEnvironment
    _reset: Callable[..., ArsenalEnvironmentCarry]
    #: Entity 1's scripted path, or None for the inert entity of today's world.
    mover: Mover | None = None

    def with_mover(self, mover: Mover | None) -> FlatTaskEnvironment:
        """This environment with another path for entity 1. The compiled reset and step are shared, so a new path
        costs no compile."""

        _check_mover(mover, self.batch)
        return replace(self, mover=mover)

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
        ``params.floor_y``. The other entity starts at full health with yaw 0. With a mover, the other entity
        starts at the path's first point instead (its x-z from the mover; ``other_xyz``'s height is kept).
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
        return self._reset(keys, agent_xz, agent_yaw, other_xyz, self.mover)

    def step(self, carry: ArsenalEnvironmentCarry, factors: jax.Array, keys: jax.Array):
        """Advance one decision tick with int32[B, 12] factors. Returns (carry, observation, info).

        With a mover, entity 1 moves first, so the physics and the returned observation both see its new place.
        """

        if self.mover is not None:
            carry = _move_carry(carry, self.mover)
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
    world_capabilities: str = "open_flat",
    mover: Mover | None = None,
) -> FlatTaskEnvironment:
    """Build the flat npc_walk environment with ``batch`` independent lanes.

    ``world_capabilities`` is "open_flat" by default: line of sight is clear on a flat world with no blocks, so the
    other entity is perceptible within the sensor range, as in the neural_agent's legal open-flat arena. "fail_closed"
    keeps the environment's default, which denies line of sight, so the legal observation never shows the other
    entity. Field of view is not applied on the two-entity path, a simulator gap reported to its lane.

    ``mover`` (opt-in, None by default) walks entity 1 along a scripted path per lane; None is the inert entity.
    """

    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError(f"batch must be a positive int, got {batch!r}")
    _check_mover(mover, batch)
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
    if world_capabilities not in ("open_flat", "fail_closed"):
        raise ValueError(f"world_capabilities must be 'open_flat' or 'fail_closed', got {world_capabilities!r}")

    def open_flat(state, flat_params):
        return open_flat_arsenal_world_capabilities(state, flat_params, config=config)

    provider = open_flat if world_capabilities == "open_flat" else None
    environment = make_arsenal_environment(
        params, config, maximum_turn_degrees=maximum_turn_degrees, world_capability_provider=provider
    )
    plan = plan_arsenal_compilation(params, config)

    def reset_impl(keys, agent_xz, agent_yaw, other_xyz, mover):
        agent_y = jnp.full((batch, 1), params.floor_y, dtype=jnp.float32)
        agent_xyz = jnp.concatenate((agent_xz[:, :1], agent_y, agent_xz[:, 1:]), axis=1)
        other_yaw = jnp.zeros((batch,), dtype=jnp.float32)
        if mover is not None:
            # The state is built around the path's start, so the controller memory's home is the path's start too.
            start_xz, other_yaw, _ = mover_path(mover, jnp.zeros((batch,), dtype=jnp.int32))
            other_xyz = other_xyz.at[:, 0].set(start_xz[:, 0]).at[:, 2].set(start_xz[:, 1])
        entity_position = jnp.stack((agent_xyz, other_xyz), axis=1)
        entity_health = jnp.stack(
            (
                jnp.full((batch,), params.agent_max_health, dtype=jnp.float32),
                jnp.full((batch,), params.target_max_health, dtype=jnp.float32),
            ),
            axis=1,
        )
        entity_yaw = jnp.stack((agent_yaw, other_yaw), axis=1)
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
        if mover is not None:
            runtime = runtime._replace(
                combat=_place_entity(runtime.combat, mover, runtime.combat.tick_count)
            )
        return _carry_from_runtime(runtime, params, config, provider)

    return FlatTaskEnvironment(
        batch=batch,
        params=params,
        config=config,
        environment=environment,
        _reset=jax.jit(reset_impl),
        mover=mover,
    )


def run_episode(
    env: FlatTaskEnvironment,
    tasks_goals: NamedTuple,
    success_fn: Callable[[CombatState, NamedTuple], jax.Array],
    policy_fn: Callable[[CombatState, NamedTuple], jax.Array],
    key: jax.Array,
    max_ticks: int,
    *,
    carry: ArsenalEnvironmentCarry | None = None,
) -> EpisodeResult:
    """Run ``max_ticks`` decision ticks on every lane and record the first tick the goal holds.

    ``tasks_goals`` is one skill's goals (a NamedTuple with ``horizon_ticks``, e.g. ``WalkToGoals``) and
    ``success_fn(combat, goals)`` that skill's test (e.g. ``WalkToTasks.success``), so the runner knows no
    skill. ``policy_fn(combat, goals)`` returns int32[B, 12] factors. A lane counts as a success only
    at ticks no later than its ``horizon_ticks``. With ``carry=None`` every lane resets at the
    default spawn (``DEFAULT_SPAWN_XZ``, yaw 0, the other entity at ``DEFAULT_OTHER_XYZ``), so
    the goals must be sampled from that spawn. Pass a carry from ``env.reset`` otherwise.
    """

    if isinstance(max_ticks, bool) or not isinstance(max_ticks, int) or max_ticks < 1:
        raise ValueError(f"max_ticks must be a positive int, got {max_ticks!r}")

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
