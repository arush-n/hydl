"""DAgger distillation of the request-conditioned generalist: walk_to and look_at from two scripted teachers.

Two scripted teachers (``teaching/teachers.py``) label every state the student visits, so the student learns on its own
state distribution (DAgger, Ross, Gordon and Bagnell 2011). Each lane of a batched flat-world episode
(``tasks/simple_tasks.py``) runs one mode: ``walk`` (walk_to only), ``look`` (look_at only) or ``both``.

Requests (``model/generalist.py``) are rebuilt at every decision from the world goals and the agent's own pose (its
position and body yaw). The resolver's job is done here, which is legal: the agent knows its own pose. The teachers read
privileged state, but they only label; the student never sees the simulator state.

Observation at the first decision. ``FlatTaskEnvironment.reset`` returns no observation, so each episode takes one neutral
tick after reset and the first decision reads that tick's observation. The horizon counts decisions, not that tick.

Held-out argument combinations, never sampled in training (``sample_episodes``):
    (a) walk targets whose bearing from the spawn body, measured clockwise from forward (right positive), lies in
        [180, 270): the rear-left quadrant;
    (b) look points whose yaw offset from the body, in the engine's yaw convention (positive is left at yaw 0), lies in
        [25, 40].
A held-out 'both' lane holds out its walk target only: its look point sits at the walk bearing +/- 30, so a rear-left walk
cannot also be at a +25..+40 look offset.

CLI::

    PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym JAX_PLATFORMS=cpu \\
        .venv/bin/python -m experimental.neural_agent.skill_agent.training.distill --iterations 6 --lanes 128 \\
        --steps 2000 --seed 0 --eval-seed 77001 --eval-lanes 256 --out REPORT.json [--params-out PARAMS.npz]
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experimental.neural_agent.skill_agent.bench import bench_tick
from experimental.neural_agent.skill_agent.contract.surface import SCOPE_NAMES, scope_of
from experimental.neural_agent.skill_agent.model.encoder import (
    EncoderConfig,
    apply_encoder,
    init_encoder,
)
from experimental.neural_agent.skill_agent.model.generalist import (
    ARG_FEATURES,
    GeneralistConfig,
    GeneralistRequests,
    apply_generalist,
    init_generalist,
)
from experimental.neural_agent.skill_agent.model.observation_adapter import legal_tokens
from experimental.neural_agent.skill_agent.tasks import simple_tasks as st
from experimental.neural_agent.skill_agent.teaching.teachers import look_at_choices, walk_to_choices
from hytalegym.jax.combat.observation.v3.policy.layout import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES as HEAD_NAMES,
)
from hytalegym.jax.combat.observation.v3.policy.layout import (
    ARSENAL_POLICY_ACTION_HEAD_SIZES as HEAD_SIZES,
)
from hytalegym.jax.combat.types import AGENT_ENTITY, PLAYER_GAIT_FORWARD_SPEED_MULTIPLIERS

MODES: tuple[str, ...] = ("walk", "look", "both")
WALK, LOOK, BOTH = 0, 1, 2
"""Lane modes: the indices into ``MODES``."""

WALK_VERB, LOOK_VERB = 0, 1
REQUEST_SLOTS = 6
DEFAULT_GENERALIST = GeneralistConfig(max_requests=REQUEST_SLOTS, verb_count=2)
HEAD_COUNT = len(HEAD_NAMES)
HEAD_STARTS = tuple(int(start) for start in np.cumsum((0,) + tuple(HEAD_SIZES[:-1])))
"""Offset of each head's choices in the flattened action mask (``carry.action_mask``)."""
NEUTRAL = np.asarray(st.NEUTRAL_FACTORS, dtype=np.int32)
WALK_HEADS = np.asarray([scope_of(name) == "locomotion" for name in HEAD_NAMES])
"""The locomotion scope's heads in the gym: jump, locomotion gait compass and body yaw (3, 4, 6)."""
GAZE_HEADS = np.asarray([scope_of(name) == "gaze" for name in HEAD_NAMES])
"""The gaze scope's heads: yaw and pitch (5, 7)."""

SPAWN_XZ = st.DEFAULT_SPAWN_XZ
OTHER_XYZ = st.DEFAULT_OTHER_XYZ
EYE_XYZ = st.DEFAULT_EYE_XYZ
REQUEST_PRESENT = np.asarray([True, False, True, True, True, True])
"""Which argument groups a request sets: position, scalars, mode, urgency and age; direction is absent."""
MODE_PARALLEL = (1.0, 0.0, 0.0)
URGENCY = 0.5
POSITION_SCALE = 16.0
GAIT_SCALE = 4.0
TOLERANCE_SCALE = 45.0
AGE_SCALE = 256.0
REAR_LEFT_DEGREES = (180.0, 270.0)
LOOK_HELD_OUT_DEGREES = (25.0, 40.0)
BOTH_LOOK_OFFSET_DEGREES = 30.0
BOTH_LOOK_PITCH_DEGREES = 20.0
STEADY_FIRST_DECISION = 10
STEADY_CLEARANCE_BLOCKS = 2.0
STEADY_HEADING_DEGREES = 10.0
STEADY_MIN_TICKS = 10
PACE_TOLERANCE = 0.05


@dataclass(frozen=True)
class DistillConfig:
    """Everything that sizes a run. The defaults are the full run's; the CLI overrides the first four."""

    iterations: int = 6
    lanes: int = 128
    steps: int = 2000
    seed: int = 0
    batch_size: int = 1024
    learning_rate: float = 1e-3
    done_weight: float = 0.5
    maximum_turn_degrees: float = 45.0
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    generalist: GeneralistConfig = field(default_factory=lambda: DEFAULT_GENERALIST)


class Episodes(NamedTuple):
    """The goals of B lanes. ``horizon`` is the decisions each lane runs: the max over its active goals."""

    walk: st.WalkToGoals
    look: st.LookAtGoals
    mode: jax.Array  # int32[B]: WALK, LOOK or BOTH
    walk_active: jax.Array  # bool[B]
    look_active: jax.Array  # bool[B]
    spawn_yaw: jax.Array  # float32[B], degrees
    horizon: jax.Array  # int32[B]


class Labels(NamedTuple):
    factors: jax.Array  # int32[B, 12]: the teacher's action, neutral on the heads it does not own
    mask: jax.Array  # bool[B, 12]: the supervised heads, the active requests' scope heads
    done: jax.Array  # bool[B, R]: each request's teacher done (slot 0 walk_to, slot 1 look_at)


class Tick(NamedTuple):
    """What a policy sees at one decision, for every lane."""

    combat: Any
    action_mask: jax.Array  # bool[B, 124]
    episode: Episodes
    tokens: dict  # type -> (features, mask): the legal observation's token sets
    requests: GeneralistRequests
    teacher: Labels
    student: jax.Array  # int32[B, 12]: masked argmax on the active heads, neutral elsewhere


class Rollout(NamedTuple):
    walk_hit: np.ndarray  # int32[B]: the decision count at which walk_to first held, or -1
    look_hit: np.ndarray  # int32[B]: the same for look_at
    positions: np.ndarray  # float32[T+1, B, 2]: planar position before each decision, and after the last
    yaws: np.ndarray  # float32[T+1, B]: body yaw at the same instants
    seconds: np.ndarray  # float32[T, B]: the duration of each decision's tick
    decisions: int  # lane decisions inside each lane's horizon
    illegal: int  # executed factors outside the action mask, over those decisions
    teacher_illegal: int  # teacher labels outside the action mask, over those decisions
    samples: dict | None  # the dataset chunk for those decisions, when recording


class Kernels(NamedTuple):
    tokens: Callable[..., Any]
    requests: Callable[..., Any]
    teacher: Callable[..., Labels]
    student: Callable[..., jax.Array]
    success: Callable[..., Any]


class DaggerResult(NamedTuple):
    params: dict
    dataset: dict
    iterations: list[dict]


def _wrap(degrees: jax.Array) -> jax.Array:
    return jnp.mod(degrees + 180.0, 360.0) - 180.0


def _facing(dx: jax.Array, dz: jax.Array) -> jax.Array:
    """The engine's yaw of the bearing (dx, dz), in degrees: forward at yaw y is (-sin y, -cos y)."""

    return jnp.rad2deg(jnp.arctan2(-dx, -dz))


def _body_vector(vector: jax.Array, yaw_degrees: jax.Array) -> jax.Array:
    """A world vector [B, 3] as (forward, right, up) in the body frame: forward (-sin y, 0, -cos y), right (cos y, 0, -sin y)."""

    yaw = jnp.deg2rad(yaw_degrees)
    x, y, z = vector[:, 0], vector[:, 1], vector[:, 2]
    forward = -jnp.sin(yaw) * x - jnp.cos(yaw) * z
    right = jnp.cos(yaw) * x - jnp.sin(yaw) * z
    return jnp.stack([forward, right, y], axis=1)


def _planar(xz: jax.Array) -> jax.Array:
    return jnp.stack([xz[:, 0], jnp.zeros_like(xz[:, 0]), xz[:, 1]], axis=1)


def _rows(mask: jax.Array, leaf: jax.Array) -> jax.Array:
    return mask.reshape(mask.shape + (1,) * (leaf.ndim - 1))


def _resample(draw: Callable[[jax.Array], Any], accept: Callable[[Any], jax.Array], key: jax.Array, rounds: int = 500):
    """Redraw the lanes that fail ``accept`` until every lane passes. Rejection keeps the accepted distribution exact."""

    key, first = jax.random.split(key)
    value = draw(first)
    for _ in range(rounds):
        ok = accept(value)
        if bool(jnp.all(ok)):
            return value
        key, step_key = jax.random.split(key)
        fresh = draw(step_key)
        value = jax.tree_util.tree_map(lambda old, new: jnp.where(_rows(ok, old), old, new), value, fresh)
    raise RuntimeError(f"could not sample goals in the requested region within {rounds} rounds")


def _in_rear_left(target_xz: jax.Array, spawn_yaw: jax.Array) -> jax.Array:
    """Held-out (a): the walk target's bearing from the spawn body, clockwise from forward, is in [180, 270)."""

    offset = _planar(target_xz - jnp.asarray(SPAWN_XZ, dtype=jnp.float32))
    forward_right = _body_vector(offset, spawn_yaw)
    bearing = jnp.mod(jnp.rad2deg(jnp.arctan2(forward_right[:, 1], forward_right[:, 0])), 360.0)
    return (bearing >= REAR_LEFT_DEGREES[0]) & (bearing < REAR_LEFT_DEGREES[1])


def _look_offset(target_xyz: jax.Array, spawn_yaw: jax.Array) -> jax.Array:
    """The look point's yaw offset from the spawn body, in the engine's yaw convention (positive is left)."""

    eye = jnp.asarray(EYE_XYZ, dtype=jnp.float32)
    facing = _facing(target_xyz[:, 0] - eye[0], target_xyz[:, 2] - eye[2])
    return _wrap(facing - spawn_yaw)


def _in_look_held_out(target_xyz: jax.Array, spawn_yaw: jax.Array) -> jax.Array:
    """Held-out (b): the look point's yaw offset from the body is in [25, 40]."""

    offset = _look_offset(target_xyz, spawn_yaw)
    return (offset >= LOOK_HELD_OUT_DEGREES[0]) & (offset <= LOOK_HELD_OUT_DEGREES[1])


def _place_both_looks(key: jax.Array, walk_bearing: jax.Array) -> jax.Array:
    """Look points at the walk bearing +/- 30 degrees, 2 to 10 blocks from the eye, within +/- 20 degrees of pitch."""

    batch = walk_bearing.shape[0]
    key_offset, key_pitch, key_distance = jax.random.split(key, 3)
    yaw = jnp.deg2rad(
        walk_bearing
        + jax.random.uniform(key_offset, (batch,), dtype=jnp.float32,
                             minval=-BOTH_LOOK_OFFSET_DEGREES, maxval=BOTH_LOOK_OFFSET_DEGREES)
    )
    pitch = jnp.deg2rad(
        jax.random.uniform(key_pitch, (batch,), dtype=jnp.float32,
                           minval=-BOTH_LOOK_PITCH_DEGREES, maxval=BOTH_LOOK_PITCH_DEGREES)
    )
    distance = jax.random.uniform(key_distance, (batch,), dtype=jnp.float32,
                                  minval=st.LOOK_DISTANCE_RANGE[0], maxval=st.LOOK_DISTANCE_RANGE[1])
    horizontal = distance * jnp.cos(pitch)
    direction = jnp.stack(
        [-jnp.sin(yaw) * horizontal, distance * jnp.sin(pitch), -jnp.cos(yaw) * horizontal], axis=1
    )
    return jnp.asarray(EYE_XYZ, dtype=jnp.float32)[None, :] + direction


def _sample_looks(key: jax.Array, walk_xz: jax.Array, spawn_yaw: jax.Array, both: jax.Array, *,
                  held_out: bool) -> st.LookAtGoals:
    batch = both.shape[0]
    eye = jnp.asarray(EYE_XYZ, dtype=jnp.float32)
    walk_bearing = _facing(walk_xz[:, 0] - eye[0], walk_xz[:, 1] - eye[2])

    def draw(step_key: jax.Array) -> st.LookAtGoals:
        key_sampled, key_placed = jax.random.split(step_key)
        sampled = st.LookAtTasks.sample(key_sampled, batch, body_yaw_degrees=spawn_yaw)
        placed = _place_both_looks(key_placed, walk_bearing)
        target = jnp.where(both[:, None], placed, sampled.target_xyz)
        return st.LookAtGoals(target, sampled.tolerance_degrees, sampled.horizon_ticks)

    def accept(goals: st.LookAtGoals) -> jax.Array:
        held = _in_look_held_out(goals.target_xyz, spawn_yaw)
        # A 'both' lane's look is placed at the walk bearing, so only the walk target holds it out (module docstring).
        return (held | both) if held_out else ~held

    return _resample(draw, accept, key)


def make_episodes(walk: st.WalkToGoals, look: st.LookAtGoals, mode: jax.Array, spawn_yaw: jax.Array) -> Episodes:
    walk_active = (mode == WALK) | (mode == BOTH)
    look_active = (mode == LOOK) | (mode == BOTH)
    horizon = jnp.maximum(
        jnp.where(walk_active, walk.horizon_ticks, 0), jnp.where(look_active, look.horizon_ticks, 0)
    ).astype(jnp.int32)
    return Episodes(walk, look, mode.astype(jnp.int32), walk_active, look_active, spawn_yaw, horizon)


def sample_episodes(key: jax.Array, mode: jax.Array, *, held_out: bool = False) -> Episodes:
    """Goals and spawn yaw for one lane per entry of ``mode`` (int32[B], indices into ``MODES``).

    The spawn yaw is uniform in [-180, 180). With ``held_out=False`` (the training region) walk targets are never in the
    rear-left quadrant and look points never have a +25..+40 offset; with ``held_out=True`` they are only there (module
    docstring). The walk rule applies to every lane, which is harmless for a look-only lane, since its walk goal is inert.
    """

    batch = mode.shape[0]
    key_yaw, key_walk, key_look = jax.random.split(key, 3)
    spawn_yaw = jax.random.uniform(key_yaw, (batch,), dtype=jnp.float32, minval=-180.0, maxval=180.0)

    def walk_accept(goals: st.WalkToGoals) -> jax.Array:
        rear_left = _in_rear_left(goals.target_xz, spawn_yaw)
        return rear_left if held_out else ~rear_left

    walk = _resample(lambda step_key: st.WalkToTasks.sample(step_key, batch), walk_accept, key_walk)
    look = _sample_looks(key_look, walk.target_xz, spawn_yaw, mode == BOTH, held_out=held_out)
    return make_episodes(walk, look, mode, spawn_yaw)


def _argument_row(position: jax.Array, scalars: jax.Array, ticks: jax.Array) -> jax.Array:
    """One request's 15 argument features in ``ARG_GROUPS`` order: position, direction, scalars, mode, urgency, age."""

    batch = position.shape[0]
    mode = jnp.broadcast_to(jnp.asarray(MODE_PARALLEL, dtype=jnp.float32), (batch, 3))
    urgency = jnp.full((batch, 1), URGENCY, dtype=jnp.float32)
    age = jnp.full((batch, 1), ticks / AGE_SCALE, dtype=jnp.float32)
    return jnp.concatenate(
        [position, jnp.zeros((batch, 3), dtype=jnp.float32), scalars, mode, urgency, age], axis=1
    )


def _request_slots(episode: Episodes, walk_args: jax.Array, look_args: jax.Array) -> GeneralistRequests:
    batch = walk_args.shape[0]
    rest = REQUEST_SLOTS - 2
    valid = jnp.concatenate(
        [jnp.stack([episode.walk_active, episode.look_active], axis=1), jnp.zeros((batch, rest), dtype=jnp.bool_)],
        axis=1,
    )
    args = jnp.concatenate(
        [
            jnp.stack(
                [
                    jnp.where(episode.walk_active[:, None], walk_args, 0.0),
                    jnp.where(episode.look_active[:, None], look_args, 0.0),
                ],
                axis=1,
            ),
            jnp.zeros((batch, rest, ARG_FEATURES), dtype=jnp.float32),
        ],
        axis=1,
    )
    verbs = jnp.asarray([WALK_VERB, LOOK_VERB] + [0] * rest, dtype=jnp.int32)
    scopes = jnp.asarray([SCOPE_NAMES.index("locomotion"), SCOPE_NAMES.index("gaze")] + [0] * rest, dtype=jnp.int32)
    pointers = jnp.full((batch, REQUEST_SLOTS), -1, dtype=jnp.int32)
    return GeneralistRequests(
        verb=jnp.broadcast_to(verbs, (batch, REQUEST_SLOTS)),
        args=args,
        arg_present=valid[..., None] & jnp.asarray(REQUEST_PRESENT),
        entity_pointer=pointers,
        cell_pointer=pointers,
        scope=jnp.broadcast_to(scopes, (batch, REQUEST_SLOTS)),
        valid=valid,
    )


def build_requests(episode: Episodes, position: jax.Array, yaw: jax.Array, ticks: jax.Array) -> GeneralistRequests:
    """The request slots at one decision: walk_to in slot 0 and look_at in slot 1, from the agent's own pose.

    ``position`` is float32[B, 3] (the agent's position), ``yaw`` the body yaw in degrees, ``ticks`` the decisions taken
    since the episode started (the age feature). Positions are body-frame vectors over 16 blocks; the eye is the position
    plus one block up.
    """

    batch = position.shape[0]
    walk, look = episode.walk, episode.look
    to_walk = _planar(walk.target_xz - jnp.stack([position[:, 0], position[:, 2]], axis=1))
    to_look = look.target_xyz - (position + jnp.asarray([0.0, st.AGENT_EYE_HEIGHT, 0.0], dtype=jnp.float32))
    zeros = jnp.zeros((batch,), dtype=jnp.float32)
    walk_scalars = jnp.stack([walk.gait.astype(jnp.float32) / GAIT_SCALE, walk.radius, zeros, zeros], axis=1)
    look_scalars = jnp.stack([look.tolerance_degrees / TOLERANCE_SCALE, zeros, zeros, zeros], axis=1)
    walk_args = _argument_row(_body_vector(to_walk, yaw) / POSITION_SCALE, walk_scalars, ticks)
    look_args = _argument_row(_body_vector(to_look, yaw) / POSITION_SCALE, look_scalars, ticks)
    return _request_slots(episode, walk_args, look_args)


def _successes(combat: Any, episode: Episodes) -> tuple[jax.Array, jax.Array]:
    return st.WalkToTasks.success(combat, episode.walk), st.LookAtTasks.success(combat, episode.look)


def teacher_labels(combat: Any, episode: Episodes, *, maximum_turn_degrees: float) -> Labels:
    """The two teachers' labels: walk heads from walk_to_choices, gaze heads from look_at_choices, neutral elsewhere."""

    walk = walk_to_choices(
        combat, episode.walk.target_xz, episode.walk.radius, episode.walk.gait,
        maximum_turn_degrees=maximum_turn_degrees,
    )
    look = look_at_choices(
        combat, episode.look.target_xyz, episode.look.tolerance_degrees,
        maximum_turn_degrees=maximum_turn_degrees,
    )
    walk_on = episode.walk_active[:, None] & jnp.asarray(WALK_HEADS)[None, :]
    look_on = episode.look_active[:, None] & jnp.asarray(GAZE_HEADS)[None, :]
    factors = jnp.where(walk_on, walk.factors, jnp.asarray(NEUTRAL))
    factors = jnp.where(look_on, look.factors, factors)
    done = jnp.zeros((combat.position.shape[0], REQUEST_SLOTS), dtype=jnp.bool_)
    done = done.at[:, 0].set(walk.done & episode.walk_active).at[:, 1].set(look.done & episode.look_active)
    return Labels(factors=factors.astype(jnp.int32), mask=walk_on | look_on, done=done)


def student_factors(
    params: dict,
    action_mask: jax.Array,
    tokens: dict,
    requests: GeneralistRequests,
    walk_active: jax.Array,
    look_active: jax.Array,
    encoder_config: EncoderConfig,
    generalist_config: GeneralistConfig,
) -> jax.Array:
    """The student's action: the masked argmax on each head of an active scope, neutral on every other head."""

    masks = {name: mask for name, (_, mask) in tokens.items()}
    encoded = apply_encoder(params["trunk"], tokens, encoder_config)
    out = apply_generalist(params["generalist"], encoded, masks, requests, generalist_config)
    active = (walk_active[:, None] & jnp.asarray(WALK_HEADS)[None, :]) | (
        look_active[:, None] & jnp.asarray(GAZE_HEADS)[None, :]
    )
    columns = []
    for head, name in enumerate(HEAD_NAMES):
        start, size = HEAD_STARTS[head], HEAD_SIZES[head]
        legal = action_mask[:, start:start + size]
        choice = jnp.argmax(jnp.where(legal, out.head_logits[name], -jnp.inf), axis=-1).astype(jnp.int32)
        columns.append(jnp.where(active[:, head], choice, int(NEUTRAL[head])))
    return jnp.stack(columns, axis=1)


@functools.lru_cache(maxsize=None)
def _kernels(encoder: EncoderConfig, generalist: GeneralistConfig, maximum_turn_degrees: float) -> Kernels:
    def student(params, action_mask, tokens, requests, walk_active, look_active):
        return student_factors(params, action_mask, tokens, requests, walk_active, look_active, encoder, generalist)

    return Kernels(
        tokens=jax.jit(lambda observation, yaw: legal_tokens(
            observation, capacities=bench_tick.CAPACITIES, body_yaw_degrees=yaw)),
        requests=jax.jit(build_requests),
        teacher=jax.jit(functools.partial(teacher_labels, maximum_turn_degrees=maximum_turn_degrees)),
        student=jax.jit(student),
        success=jax.jit(_successes),
    )


@functools.lru_cache(maxsize=None)
def _environment(batch: int, maximum_turn_degrees: float) -> st.FlatTaskEnvironment:
    return st.make_flat_npc_environment(batch, maximum_turn_degrees=maximum_turn_degrees)


def _legal_rows(mask: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """Per lane: every head's factor is in range and allowed by the flattened action mask."""

    lanes = np.arange(factors.shape[0])
    legal = np.ones(factors.shape[0], dtype=bool)
    for head, (start, size) in enumerate(zip(HEAD_STARTS, HEAD_SIZES)):
        choice = factors[:, head]
        in_range = (choice >= 0) & (choice < size)
        legal &= in_range & mask[lanes, start + np.clip(choice, 0, size - 1)]
    return legal


def _host(value: Any, rows: np.ndarray) -> np.ndarray:
    return np.asarray(value)[rows]


def _chunk(tick: Tick, keep: np.ndarray) -> dict[str, np.ndarray]:
    """The dataset rows of one tick: the entity and self token sets, the requests, the teacher's labels and done flags."""

    rows = np.flatnonzero(keep)
    entity, entity_valid = tick.tokens["entity"]
    self_features, self_valid = tick.tokens["self"]
    requests = tick.requests
    return {
        "entity": _host(entity, rows),
        "entity_valid": _host(entity_valid, rows),
        "self": _host(self_features, rows),
        "self_valid": _host(self_valid, rows),
        "verb": _host(requests.verb, rows),
        "args": _host(requests.args, rows),
        "arg_present": _host(requests.arg_present, rows),
        "entity_pointer": _host(requests.entity_pointer, rows),
        "cell_pointer": _host(requests.cell_pointer, rows),
        "scope": _host(requests.scope, rows),
        "valid": _host(requests.valid, rows),
        "labels": _host(tick.teacher.factors, rows),
        "label_mask": _host(tick.teacher.mask, rows),
        "done": _host(tick.teacher.done, rows),
    }


def _concat(chunks: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}


def _rollout(env, params, episode: Episodes, decide: Callable[[Tick], Any], kernels: Kernels, key: jax.Array,
             *, record: bool) -> Rollout:
    """Run every lane for its horizon: one neutral tick, then one decision per tick, the policy's choice executed.

    ``decide(tick)`` returns the factors to execute. Lanes shorter than the batch's longest horizon keep stepping, but
    their decisions past their horizon are neither counted nor recorded.
    """

    batch = env.batch
    horizon = np.asarray(episode.horizon)
    ticks = int(horizon.max())
    keys = jax.random.split(key, ticks + 2)
    spawn = jnp.broadcast_to(jnp.asarray(SPAWN_XZ, dtype=jnp.float32), (batch, 2))
    other = jnp.broadcast_to(jnp.asarray(OTHER_XYZ, dtype=jnp.float32), (batch, 3))
    carry = env.reset(jax.random.split(keys[0], batch), spawn, episode.spawn_yaw, other)
    neutral = jnp.broadcast_to(jnp.asarray(NEUTRAL, dtype=jnp.int32), (batch, HEAD_COUNT))
    carry, observation, _ = env.step(carry, neutral, jax.random.split(keys[1], batch))

    walk_horizon = np.asarray(episode.walk.horizon_ticks)
    look_horizon = np.asarray(episode.look.horizon_ticks)
    walk_active = np.asarray(episode.walk_active)
    look_active = np.asarray(episode.look_active)
    walk_hit = np.full(batch, -1, dtype=np.int32)
    look_hit = np.full(batch, -1, dtype=np.int32)
    positions = np.zeros((ticks + 1, batch, 2), dtype=np.float32)
    yaws = np.zeros((ticks + 1, batch), dtype=np.float32)
    seconds = np.zeros((ticks, batch), dtype=np.float32)
    decisions = illegal = teacher_illegal = 0
    chunks: list[dict[str, np.ndarray]] = []

    def record_instant(index: int, combat: Any) -> None:
        positions[index] = np.asarray(combat.position[:, AGENT_ENTITY][:, [0, 2]])
        yaws[index] = np.asarray(combat.yaw[:, AGENT_ENTITY])

    record_instant(0, carry.runtime.combat)
    for index in range(ticks):
        combat = carry.runtime.combat
        position = combat.position[:, AGENT_ENTITY]
        yaw = combat.yaw[:, AGENT_ENTITY]
        requests = kernels.requests(episode, position, yaw, jnp.float32(index))
        tokens = kernels.tokens(observation, yaw)
        teacher = kernels.teacher(combat, episode)
        student = kernels.student(params, carry.action_mask, tokens, requests, episode.walk_active, episode.look_active)
        tick = Tick(combat, carry.action_mask, episode, tokens, requests, teacher, student)
        executed = jnp.asarray(decide(tick), dtype=jnp.int32)

        keep = index < horizon
        mask = np.asarray(carry.action_mask)
        decisions += int(keep.sum())
        illegal += int(np.sum(keep & ~_legal_rows(mask, np.asarray(executed))))
        teacher_illegal += int(np.sum(keep & ~_legal_rows(mask, np.asarray(teacher.factors))))
        if record:
            chunks.append(_chunk(tick, keep))

        carry, observation, _ = env.step(carry, executed, jax.random.split(keys[index + 2], batch))
        done_count = index + 1
        combat = carry.runtime.combat
        record_instant(done_count, combat)
        seconds[index] = np.asarray(combat.last_motion_delta_seconds)
        walk_ok, look_ok = (np.asarray(value) for value in kernels.success(combat, episode))
        walk_hit = np.where((walk_hit < 0) & walk_active & walk_ok & (done_count <= walk_horizon),
                            done_count, walk_hit).astype(np.int32)
        look_hit = np.where((look_hit < 0) & look_active & look_ok & (done_count <= look_horizon),
                            done_count, look_hit).astype(np.int32)

    samples = _concat(chunks) if record else None
    return Rollout(walk_hit, look_hit, positions, yaws, seconds, decisions, illegal, teacher_illegal, samples)


def init_params(key: jax.Array, config: DistillConfig) -> dict:
    trunk_key, generalist_key = jax.random.split(key)
    return {
        "trunk": init_encoder(trunk_key, config.encoder),
        "generalist": init_generalist(generalist_key, config.generalist, config.encoder),
    }


def _batch_tokens(batch: dict) -> dict:
    """The token sets of a stored batch: only entity and self, which are stored. The other trunk types are absent. The
    encoder and the generalist read an absent type as no valid token, and the flat world fills none of them (no blocks,
    inventory, geometry or kit). The loss equals the one with those four types zero-filled (test_sa_distill.py)."""

    return {name: (batch[name], batch[f"{name}_valid"]) for name in ("entity", "self")}


def loss_fn(params: dict, batch: dict, config: DistillConfig) -> jax.Array:
    """Mean cross-entropy over the supervised head labels, plus done_weight x the BCE of the done logit on valid requests."""

    tokens = _batch_tokens(batch)
    masks = {name: mask for name, (_, mask) in tokens.items()}
    encoded = apply_encoder(params["trunk"], tokens, config.encoder)
    requests = GeneralistRequests(
        verb=batch["verb"], args=batch["args"], arg_present=batch["arg_present"],
        entity_pointer=batch["entity_pointer"], cell_pointer=batch["cell_pointer"],
        scope=batch["scope"], valid=batch["valid"],
    )
    out = apply_generalist(params["generalist"], encoded, masks, requests, config.generalist)
    rows = jnp.arange(batch["labels"].shape[0])
    supervised = batch["label_mask"].astype(jnp.float32)
    total = 0.0
    for head, name in enumerate(HEAD_NAMES):
        log_probs = jax.nn.log_softmax(out.head_logits[name], axis=-1)
        total = total - jnp.sum(log_probs[rows, batch["labels"][:, head]] * supervised[:, head])
    cross_entropy = total / jnp.maximum(jnp.sum(supervised), 1.0)
    valid = batch["valid"].astype(jnp.float32)
    done_bce = optax.sigmoid_binary_cross_entropy(out.status[..., 0], batch["done"].astype(jnp.float32))
    done_loss = jnp.sum(done_bce * valid) / jnp.maximum(jnp.sum(valid), 1.0)
    return cross_entropy + config.done_weight * done_loss


def make_train_step(config: DistillConfig, optimizer: optax.GradientTransformation) -> Callable[..., Any]:
    def step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, batch, config))(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    return jax.jit(step)


def _collect(env, params, key: jax.Array, config: DistillConfig, beta: float, kernels: Kernels):
    """One DAgger iteration's episodes: each lane runs the teacher with probability ``beta``, else the student."""

    key_mode, key_goals, key_mix, key_rollout = jax.random.split(key, 4)
    mode = jax.random.randint(key_mode, (config.lanes,), 0, len(MODES))
    episode = sample_episodes(key_goals, mode, held_out=False)
    use_teacher = jax.random.uniform(key_mix, (config.lanes,)) < beta

    def decide(tick: Tick) -> jax.Array:
        return jnp.where(use_teacher[:, None], tick.teacher.factors, tick.student)

    rollout = _rollout(env, params, episode, decide, kernels, key_rollout, record=True)
    horizon = np.asarray(episode.horizon)
    stats = {
        "teacher_lanes": int(np.asarray(use_teacher).sum()),
        "lanes_by_mode": {name: int((np.asarray(mode) == code).sum()) for code, name in enumerate(MODES)},
        "mean_horizon_decisions": float(horizon.mean()),
        "new_samples": int(rollout.samples["labels"].shape[0]),
        "executed_illegal": rollout.illegal,
        "teacher_label_illegal": rollout.teacher_illegal,
    }
    return rollout.samples, stats


def run_dagger(config: DistillConfig = DistillConfig()) -> DaggerResult:
    """Iterate DAgger: collect, aggregate, then ``config.steps`` Adam steps on the whole aggregate, per iteration."""

    env = _environment(config.lanes, config.maximum_turn_degrees)
    kernels = _kernels(config.encoder, config.generalist, config.maximum_turn_degrees)
    key, init_key = jax.random.split(jax.random.key(config.seed))
    params = init_params(init_key, config)
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)
    train_step = make_train_step(config, optimizer)
    rng = np.random.default_rng(config.seed)

    chunks: list[dict[str, np.ndarray]] = []
    iterations: list[dict] = []
    dataset: dict[str, np.ndarray] = {}
    for index in range(config.iterations):
        beta = 1.0 if index == 0 else 0.5 ** index
        started = time.perf_counter()
        key, collect_key = jax.random.split(key)
        chunk, stats = _collect(env, params, collect_key, config, beta, kernels)
        chunks.append(chunk)
        dataset = _concat(chunks)
        collected = time.perf_counter()

        size = dataset["labels"].shape[0]
        losses = []
        for _ in range(config.steps):
            rows = rng.integers(0, size, size=config.batch_size)
            batch = {name: jnp.asarray(value[rows]) for name, value in dataset.items()}
            params, opt_state, loss = train_step(params, opt_state, batch)
            losses.append(loss)
        losses = [float(value) for value in losses]
        record = {
            "iteration": index,
            "beta": beta,
            **stats,
            "dataset_size": int(size),
            "steps": config.steps,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "collect_seconds": round(collected - started, 2),
            "train_seconds": round(time.perf_counter() - collected, 2),
            "wall_seconds": round(time.perf_counter() - started, 2),
        }
        iterations.append(record)
        print(f"iteration {index} beta {beta:.3f}: dataset {size}, loss {record['loss_first']} -> "
              f"{record['loss_last']}, wall {record['wall_seconds']} s", file=sys.stderr, flush=True)
    return DaggerResult(params=params, dataset=dataset, iterations=iterations)


def teacher_policy(tick: Tick) -> jax.Array:
    """The scripted teachers' merged action: the reference policy the evaluation compares the student against."""

    return tick.teacher.factors


def _wrap_np(degrees: np.ndarray) -> np.ndarray:
    return np.mod(degrees + 180.0, 360.0) - 180.0


def _pace(episode: Episodes, rollout: Rollout) -> dict:
    """Walk pace over the steady walk lanes: a lane is steady after ``STEADY_FIRST_DECISION`` decisions, while it is still
    well short of its target and its body points at the target to within ``STEADY_HEADING_DEGREES``. A steady lane's mean
    planar speed (over its steady decisions) must sit within 5% of the requested gait's forward speed."""

    target = np.asarray(episode.walk.target_xz, dtype=np.float64)
    radius = np.asarray(episode.walk.radius, dtype=np.float64)
    gait = np.asarray(episode.walk.gait)
    walk_active = np.asarray(episode.walk_active)
    positions = rollout.positions.astype(np.float64)
    yaws = rollout.yaws.astype(np.float64)
    seconds = np.maximum(rollout.seconds.astype(np.float64), 1e-9)
    before = positions[:-1]
    offset = target[None] - before
    distance = np.hypot(offset[..., 0], offset[..., 1])
    heading = _wrap_np(np.degrees(np.arctan2(-offset[..., 0], -offset[..., 1])) - yaws[:-1])
    decision = np.arange(seconds.shape[0])[:, None]
    steady = (
        (decision >= STEADY_FIRST_DECISION)
        & (distance > radius[None] + STEADY_CLEARANCE_BLOCKS)
        & (np.abs(heading) <= STEADY_HEADING_DEGREES)
        & walk_active[None]
    )
    delta = positions[1:] - before
    speed = np.hypot(delta[..., 0], delta[..., 1]) / seconds
    counts = steady.sum(axis=0)
    lanes = counts >= STEADY_MIN_TICKS
    mean_speed = (speed * steady).sum(axis=0) / np.maximum(counts, 1)
    requested = st.NPC_WALK_BASE_SPEED * np.asarray(PLAYER_GAIT_FORWARD_SPEED_MULTIPLIERS)[gait]
    within = np.abs(mean_speed - requested) <= PACE_TOLERANCE * requested
    count = int(lanes.sum())
    return {
        "steady_lanes": count,
        "within_5_percent": float(within[lanes].mean()) if count else None,
        "mean_speed_steady_lanes": float(mean_speed[lanes].mean()) if count else None,
    }


def _radius(episode: Episodes, rollout: Rollout) -> dict:
    """Radius respected at the first success: recomputed from the recorded position, not the success predicate."""

    lanes = np.flatnonzero((rollout.walk_hit >= 1) & np.asarray(episode.walk_active))
    if lanes.size == 0:
        return {"successes": 0, "within_radius": None}
    at = rollout.positions[rollout.walk_hit[lanes], lanes]
    target = np.asarray(episode.walk.target_xz)[lanes]
    distance = np.hypot(*(at - target).T)
    within = distance <= np.asarray(episode.walk.radius)[lanes] + 1e-4
    return {"successes": int(lanes.size), "within_radius": float(within.mean())}


def _summarise(episode: Episodes, rollout: Rollout) -> dict:
    mode = np.asarray(episode.mode)
    walk_ok = rollout.walk_hit >= 1
    look_ok = rollout.look_hit >= 1

    def rate(values: np.ndarray, lanes: np.ndarray) -> dict:
        count = int(lanes.sum())
        return {"lanes": count, "success": float(values[lanes].mean()) if count else None}

    both = mode == BOTH
    both_summary = rate(walk_ok & look_ok, both)
    both_summary["walk_success"] = rate(walk_ok, both)["success"]
    both_summary["look_success"] = rate(look_ok, both)["success"]
    return {
        "walk_only": rate(walk_ok, mode == WALK),
        "look_only": rate(look_ok, mode == LOOK),
        "both": both_summary,
        "walk_pace": _pace(episode, rollout),
        "walk_radius": _radius(episode, rollout),
        "decisions": rollout.decisions,
        "illegal_decisions": rollout.illegal,
        "teacher_label_illegal": rollout.teacher_illegal,
    }


def evaluate(
    params: dict,
    *,
    seed: int,
    lanes: int,
    held_out: bool,
    policy: str | Callable[[Tick], jax.Array] = "student",
    modes: Sequence[str] = MODES,
    config: DistillConfig = DistillConfig(),
) -> dict:
    """Run ``policy`` on fresh episodes and report per-mode success, walk pace, radius and legality.

    ``policy="student"`` runs the masked argmax; a callable takes a :class:`Tick` and returns int32[B, 12] factors (use
    ``teacher_policy`` for the teachers). Lanes take ``modes`` round-robin, so each mode gets an equal share. The same
    ``seed`` gives the same goals and spawns, so the student and the teacher see identical episodes.
    """

    if isinstance(policy, str) and policy != "student":
        raise ValueError(f"policy must be 'student' or a callable, got {policy!r}")
    if not modes or any(name not in MODES for name in modes):
        raise ValueError(f"modes must be a non-empty subset of {MODES}, got {modes!r}")
    chosen = np.asarray([MODES.index(name) for name in modes], dtype=np.int32)
    codes = chosen[np.arange(lanes) % len(chosen)]
    env = _environment(lanes, config.maximum_turn_degrees)
    kernels = _kernels(config.encoder, config.generalist, config.maximum_turn_degrees)
    key_goals, key_rollout = jax.random.split(jax.random.key(seed))
    episode = sample_episodes(key_goals, jnp.asarray(codes), held_out=held_out)
    decide = (lambda tick: tick.student) if isinstance(policy, str) else policy
    rollout = _rollout(env, params, episode, decide, kernels, key_rollout, record=False)
    name = policy if isinstance(policy, str) else getattr(policy, "__name__", "callable")
    return {"policy": name, "held_out": held_out, "seed": seed, "lanes": lanes, "modes": list(modes),
            **_summarise(episode, rollout)}


def _path_name(path: Sequence[Any]) -> str:
    names = []
    for entry in path:
        if hasattr(entry, "key"):
            names.append(str(entry.key))
        elif hasattr(entry, "name"):
            names.append(str(entry.name))
        elif hasattr(entry, "idx"):
            names.append(str(entry.idx))
        else:
            names.append(str(entry))
    return ".".join(names)


def save_params(path: Path, params: dict) -> None:
    flat = {}
    for entries, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        flat[_path_name(entries)] = np.asarray(leaf)
    np.savez(path, **flat)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _git(*args: str) -> str | None:
    """The stdout of ``git <args>`` run in this package's folder, or None when git cannot answer."""

    try:
        done = subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent, capture_output=True, text=True,
                              check=False)
    except OSError:
        return None
    return done.stdout if done.returncode == 0 else None


def _provenance() -> dict[str, Any]:
    """What tells one report's run from another's: the commit, whether this package had uncommitted changes (the
    shared tree always has some elsewhere), the command line, the load at the start (``main`` adds the end), and two
    settings this trainer does not set, so E12 runs can be told apart. ``world_capabilities`` is the flat
    environment's default (``tasks/simple_tasks.py``)."""

    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--", str(Path(__file__).resolve().parents[1]))
    return {
        "base_commit": None if head is None else head.strip(),
        "dirty": None if status is None else bool(status.strip()),
        "argv": list(sys.argv),
        "load_average": {"start": list(os.getloadavg()), "end": None},
        "world_capabilities": "open_flat",
        "view_sector_full_angle_radians": None,
        "legal_observation_sensor_range": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    defaults = DistillConfig()
    parser = argparse.ArgumentParser(description="DAgger distillation of walk_to and look_at into the generalist.")
    parser.add_argument("--iterations", type=int, default=defaults.iterations)
    parser.add_argument("--lanes", type=int, default=defaults.lanes)
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--eval-seed", type=int, default=77001)
    parser.add_argument("--eval-lanes", type=int, default=256)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--params-out", type=Path)
    args = parser.parse_args(argv)

    config = DistillConfig(iterations=args.iterations, lanes=args.lanes, steps=args.steps, seed=args.seed)
    provenance = _provenance()
    started = time.perf_counter()
    result = run_dagger(config)
    evaluation = {}
    for region, held_out in (("seen", False), ("held_out", True)):
        evaluation[region] = {
            "student": evaluate(result.params, seed=args.eval_seed, lanes=args.eval_lanes, held_out=held_out,
                                policy="student", config=config),
            "teacher": evaluate(result.params, seed=args.eval_seed, lanes=args.eval_lanes, held_out=held_out,
                                policy=teacher_policy, config=config),
        }
    provenance["load_average"]["end"] = list(os.getloadavg())
    samples = int(result.dataset["labels"].shape[0])
    bytes_per_sample = sum(value[0].nbytes for value in result.dataset.values()) if samples else 0
    report = {
        "schema": "skill_agent_distill_v1",
        "config": dataclasses.asdict(config),
        "heads": list(HEAD_NAMES),
        "walk_heads": [int(i) for i in np.flatnonzero(WALK_HEADS)],
        "gaze_heads": [int(i) for i in np.flatnonzero(GAZE_HEADS)],
        "held_out_rules": {
            "walk_rear_left_bearing_degrees": list(REAR_LEFT_DEGREES),
            "look_yaw_offset_degrees": list(LOOK_HELD_OUT_DEGREES),
        },
        "training": {
            "iterations": result.iterations,
            "dataset_size": samples,
            "bytes_per_sample": bytes_per_sample,
        },
        "evaluation": evaluation,
        "seconds": round(time.perf_counter() - started, 1),
        "provenance": provenance,
    }
    if args.params_out:
        save_params(args.params_out, result.params)
    text = json.dumps(report, indent=1, default=_json_default)
    if args.out:
        args.out.write_text(text + "\n")
    sys.stdout.write(text + "\n")
    return 0


__all__ = [
    "DistillConfig",
    "Episodes",
    "Labels",
    "Tick",
    "build_requests",
    "evaluate",
    "init_params",
    "loss_fn",
    "make_episodes",
    "run_dagger",
    "sample_episodes",
    "student_factors",
    "teacher_labels",
    "teacher_policy",
]


if __name__ == "__main__":
    raise SystemExit(main())
