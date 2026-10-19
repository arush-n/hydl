"""DAgger distillation of the request-conditioned generalist: walk_to and look_at from two scripted teachers.

Two scripted teachers (``neural_skills/walk_to/teacher.py``, ``neural_skills/look_at/teacher.py``) label every state
the student visits, so the student learns on its own state distribution (DAgger, Ross, Gordon and Bagnell 2011). Each
lane of a batched flat-world episode (``tasks/simple_tasks.py``, with the goals of each skill's ``task.py``) runs one
mode: ``walk`` (walk_to only), ``look`` (look_at only) or ``both``.

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
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experimental.neural_agent.skill_agent.contract.surface import SCOPE_NAMES, scope_of
from experimental.neural_agent.skill_agent.model.encoder import (
    EncoderConfig,
    apply_encoder,
    init_encoder,
)
from experimental.neural_agent.skill_agent.model.generalist import (
    ARG_FEATURES,
    ARG_GROUPS,
    GeneralistConfig,
    GeneralistRequests,
    apply_generalist,
    init_generalist,
)
from experimental.neural_agent.skill_agent.neural_skills.look_at.task import (
    LookAtGoals,
    LookAtTasks,
)
from experimental.neural_agent.skill_agent.neural_skills.look_at.skill import (
    LOOK_HELD_OUT_DEGREES,
    in_look_held_out,
    place_beside_walk,
)
from experimental.neural_agent.skill_agent.neural_skills.look_at.skill import request_args as look_request_args
from experimental.neural_agent.skill_agent.neural_skills.look_at.teacher import look_at_choices
from experimental.neural_agent.skill_agent.neural_skills.walk_to.task import WalkToGoals, WalkToTasks
from experimental.neural_agent.skill_agent.neural_skills.walk_to.skill import (
    REAR_LEFT_DEGREES,
    pace,
    radius_kept,
)
from experimental.neural_agent.skill_agent.neural_skills.walk_to.skill import request_args as walk_request_args
from experimental.neural_agent.skill_agent.neural_skills.walk_to.skill import sample as walk_sample
from experimental.neural_agent.skill_agent.neural_skills.walk_to.teacher import walk_to_choices, walk_to_version
from experimental.neural_agent.skill_agent.teaching.teachers import TEACHERS, state_for
from experimental.neural_agent.skill_agent.training.blocks import (
    EYE_XYZ,
    HEAD_COUNT,
    HEAD_STARTS,
    HOLD_TICKS,
    NEUTRAL,
    OTHER_XYZ,
    REQUEST_SLOTS,
    SPAWN_XZ,
    batch_tokens,
    chunk,
    concat_chunks,
    environment,
    facing_degrees,
    hold_update,
    json_default,
    legal_rows,
    provenance,
    resample,
    save_params,
    token_kernel,
)
from hytalegym.jax.combat.observation.v3.policy.layout import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES as HEAD_NAMES,
)
from hytalegym.jax.combat.observation.v3.policy.layout import (
    ARSENAL_POLICY_ACTION_HEAD_SIZES as HEAD_SIZES,
)
from hytalegym.jax.combat.types import AGENT_ENTITY

MODES: tuple[str, ...] = ("walk", "look", "both")
WALK, LOOK, BOTH = 0, 1, 2
"""Lane modes: the indices into ``MODES``."""

WALK_VERB, LOOK_VERB = 0, 1
DEFAULT_GENERALIST = GeneralistConfig(max_requests=REQUEST_SLOTS, verb_count=2)
WALK_HEADS = np.asarray([scope_of(name) == "locomotion" for name in HEAD_NAMES])
"""The locomotion scope's heads in the gym: jump, locomotion gait compass and body yaw (3, 4, 6)."""
GAZE_HEADS = np.asarray([scope_of(name) == "gaze" for name in HEAD_NAMES])
"""The gaze scope's heads: yaw and pitch (5, 7)."""

REQUEST_PRESENT = np.asarray([True, False, True, True, True, True])
"""Which argument groups a request sets: position, scalars, mode, urgency and age; direction is absent, unless
``look_direction`` sets it on the look request (``_request_slots``)."""
DIRECTION_GROUP = [name for name, _ in ARG_GROUPS].index("direction")


@dataclass(frozen=True)
class DistillConfig:
    """Everything that sizes a run. The defaults are the full run's; the CLI overrides the first four and last three."""

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
    walk_stop_margin: float = 0.0  # walk_to v3 when positive: it stops this many blocks inside the radius
    look_direction: bool = False  # look_at's request carries the unit direction to its target, in look_frame
    look_frame: str = "body"  # "head": that direction is measured from where the head points (needs look_direction)

    def __post_init__(self) -> None:
        if self.look_frame not in ("body", "head"):
            raise ValueError(f"look_frame must be 'body' or 'head', got {self.look_frame!r}")
        if self.look_frame == "head" and not self.look_direction:
            raise ValueError("look_frame 'head' needs look_direction: it measures the look request's direction")


class Episodes(NamedTuple):
    """The goals of B lanes. ``horizon`` is the decisions each lane runs: the max over its active goals."""

    walk: WalkToGoals
    look: LookAtGoals
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
    """One rollout's results. A hold is ``HOLD_TICKS`` consecutive decisions on target whose first tick is inside the
    lane's horizon (``hold_update``). The rollout stops at the batch's longest horizon, so a lane whose horizon is that
    longest cannot complete a hold that starts in its last ``HOLD_TICKS - 1`` decisions."""

    walk_hit: np.ndarray  # int32[B]: the decision count at which walk_to first held, or -1
    look_hit: np.ndarray  # int32[B]: the same for look_at
    positions: np.ndarray  # float32[T+1, B, 2]: planar position before each decision, and after the last
    yaws: np.ndarray  # float32[T+1, B]: body yaw at the same instants
    seconds: np.ndarray  # float32[T, B]: the duration of each decision's tick
    decisions: int  # lane decisions inside each lane's horizon
    illegal: int  # executed factors outside the action mask, over those decisions
    teacher_illegal: int  # teacher labels outside the action mask, over those decisions
    samples: dict | None  # the dataset chunk for those decisions, when recording
    walk_held: np.ndarray  # int32[B]: the decision count at which walk_to's hold completes, or -1
    look_held: np.ndarray  # int32[B]: the same for look_at


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


def _sample_looks(key: jax.Array, walk_xz: jax.Array, spawn_yaw: jax.Array, both: jax.Array, *,
                  held_out: bool) -> LookAtGoals:
    batch = both.shape[0]
    eye = jnp.asarray(EYE_XYZ, dtype=jnp.float32)
    walk_bearing = facing_degrees(walk_xz[:, 0] - eye[0], walk_xz[:, 1] - eye[2])

    def draw(step_key: jax.Array) -> LookAtGoals:
        key_sampled, key_placed = jax.random.split(step_key)
        sampled = LookAtTasks.sample(key_sampled, batch, body_yaw_degrees=spawn_yaw)
        placed = place_beside_walk(key_placed, walk_bearing)
        target = jnp.where(both[:, None], placed, sampled.target_xyz)
        return LookAtGoals(target, sampled.tolerance_degrees, sampled.horizon_ticks)

    def accept(goals: LookAtGoals) -> jax.Array:
        held = in_look_held_out(goals.target_xyz, spawn_yaw)
        # A 'both' lane's look is placed at the walk bearing, so only the walk target holds it out (module docstring).
        return (held | both) if held_out else ~held

    return resample(draw, accept, key)


def make_episodes(walk: WalkToGoals, look: LookAtGoals, mode: jax.Array, spawn_yaw: jax.Array) -> Episodes:
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

    walk = walk_sample(key_walk, batch, spawn_yaw, held_out=held_out)
    look = _sample_looks(key_look, walk.target_xz, spawn_yaw, mode == BOTH, held_out=held_out)
    return make_episodes(walk, look, mode, spawn_yaw)


def _request_slots(episode: Episodes, walk_args: jax.Array, look_args: jax.Array, *,
                   look_direction: bool = False) -> GeneralistRequests:
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
    present = valid[..., None] & jnp.asarray(REQUEST_PRESENT)
    if look_direction:
        present = present.at[:, 1, DIRECTION_GROUP].set(valid[:, 1])
    return GeneralistRequests(
        verb=jnp.broadcast_to(verbs, (batch, REQUEST_SLOTS)),
        args=args,
        arg_present=present,
        entity_pointer=pointers,
        cell_pointer=pointers,
        scope=jnp.broadcast_to(scopes, (batch, REQUEST_SLOTS)),
        valid=valid,
    )


def build_requests(episode: Episodes, position: jax.Array, yaw: jax.Array, ticks: jax.Array,
                   head_yaw: jax.Array | None = None, head_pitch: jax.Array | None = None, *,
                   look_direction: bool = False, look_frame: str = "body") -> GeneralistRequests:
    """The request slots at one decision: walk_to in slot 0 and look_at in slot 1, from the agent's own pose.

    ``position`` is float32[B, 3] (the agent's position), ``yaw`` the body yaw in degrees, ``ticks`` the decisions taken
    since the episode started (the age feature). Positions are body-frame vectors over 16 blocks; the eye is the position
    plus one block up. With ``look_direction`` the look request's direction is the unit body-frame vector from the eye.
    With ``look_frame="head"`` it is the unit vector in the head's frame instead, which needs ``head_yaw`` and
    ``head_pitch``, the head's absolute yaw and pitch in degrees.
    """

    walk_args = walk_request_args(episode.walk, position, yaw, ticks)
    look_args = look_request_args(episode.look, position, yaw, ticks, head_yaw, head_pitch, direction=look_direction,
                                  frame=look_frame)
    return _request_slots(episode, walk_args, look_args, look_direction=look_direction)


def _successes(combat: Any, episode: Episodes) -> tuple[jax.Array, jax.Array]:
    return WalkToTasks.success(combat, episode.walk), LookAtTasks.success(combat, episode.look)


def teacher_labels(combat: Any, episode: Episodes, *, maximum_turn_degrees: float,
                   walk_stop_margin: float = 0.0) -> Labels:
    """The two registered teachers' labels (``teachers.state_for``: v2 reads what the student sees): walk heads from
    walk_to_choices, gaze heads from look_at_choices, neutral elsewhere. ``walk_stop_margin`` > 0 is walk_to v3."""

    walk = walk_to_choices(
        state_for("walk_to", combat), episode.walk.target_xz, episode.walk.radius, episode.walk.gait,
        maximum_turn_degrees=maximum_turn_degrees, stop_margin_blocks=walk_stop_margin,
    )
    look = look_at_choices(
        state_for("look_at", combat), episode.look.target_xyz, episode.look.tolerance_degrees,
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
def _kernels(encoder: EncoderConfig, generalist: GeneralistConfig, maximum_turn_degrees: float, *,
             walk_stop_margin: float = 0.0, look_direction: bool = False, look_frame: str = "body") -> Kernels:
    def student(params, action_mask, tokens, requests, walk_active, look_active):
        return student_factors(params, action_mask, tokens, requests, walk_active, look_active, encoder, generalist)

    return Kernels(
        tokens=token_kernel(),
        requests=jax.jit(functools.partial(build_requests, look_direction=look_direction, look_frame=look_frame)),
        teacher=jax.jit(functools.partial(
            teacher_labels, maximum_turn_degrees=maximum_turn_degrees, walk_stop_margin=walk_stop_margin)),
        student=jax.jit(student),
        success=jax.jit(_successes),
    )


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
    walk_streak = np.zeros(batch, dtype=np.int32)
    look_streak = np.zeros(batch, dtype=np.int32)
    walk_held = np.full(batch, -1, dtype=np.int32)
    look_held = np.full(batch, -1, dtype=np.int32)
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
        # The head's angles feed the head frame only; the body frame ignores them (build_requests).
        requests = kernels.requests(episode, position, yaw, jnp.float32(index), combat.agent_head_yaw,
                                    combat.agent_head_pitch)
        tokens = kernels.tokens(observation, yaw)
        teacher = kernels.teacher(combat, episode)
        student = kernels.student(params, carry.action_mask, tokens, requests, episode.walk_active, episode.look_active)
        tick = Tick(combat, carry.action_mask, episode, tokens, requests, teacher, student)
        executed = jnp.asarray(decide(tick), dtype=jnp.int32)

        keep = index < horizon
        mask = np.asarray(carry.action_mask)
        decisions += int(keep.sum())
        illegal += int(np.sum(keep & ~legal_rows(mask, np.asarray(executed))))
        teacher_illegal += int(np.sum(keep & ~legal_rows(mask, np.asarray(teacher.factors))))
        if record:
            chunks.append(chunk(tick.tokens, tick.requests, tick.teacher.factors, tick.teacher.mask,
                                tick.teacher.done, keep))

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
        walk_streak, walk_held = hold_update(walk_streak, walk_held, walk_ok, done_count, walk_horizon, walk_active)
        look_streak, look_held = hold_update(look_streak, look_held, look_ok, done_count, look_horizon, look_active)

    samples = concat_chunks(chunks) if record else None
    return Rollout(walk_hit, look_hit, positions, yaws, seconds, decisions, illegal, teacher_illegal, samples,
                   walk_held, look_held)


def init_params(key: jax.Array, config: DistillConfig) -> dict:
    trunk_key, generalist_key = jax.random.split(key)
    return {
        "trunk": init_encoder(trunk_key, config.encoder),
        "generalist": init_generalist(generalist_key, config.generalist, config.encoder),
    }


def loss_fn(params: dict, batch: dict, config: DistillConfig) -> jax.Array:
    """Mean cross-entropy over the supervised head labels, plus done_weight x the BCE of the done logit on valid requests."""

    tokens = batch_tokens(batch)
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

    env = environment(config.lanes, config.maximum_turn_degrees)
    kernels = _kernels(config.encoder, config.generalist, config.maximum_turn_degrees,
                       walk_stop_margin=config.walk_stop_margin, look_direction=config.look_direction,
                       look_frame=config.look_frame)
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
        dataset = concat_chunks(chunks)
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


def _summarise(episode: Episodes, rollout: Rollout) -> dict:
    mode = np.asarray(episode.mode)
    walk_ok = rollout.walk_hit >= 1
    look_ok = rollout.look_hit >= 1
    walk_held = rollout.walk_held >= 1
    look_held = rollout.look_held >= 1

    def share(flags: np.ndarray, lanes: np.ndarray) -> float | None:
        return float(flags[lanes].mean()) if lanes.any() else None

    def rate(success: np.ndarray, held: np.ndarray, lanes: np.ndarray) -> dict:
        return {"lanes": int(lanes.sum()), "success": share(success, lanes), "held": share(held, lanes)}

    both = mode == BOTH
    both_summary = rate(walk_ok & look_ok, walk_held & look_held, both)
    both_summary["walk_success"] = share(walk_ok, both)
    both_summary["look_success"] = share(look_ok, both)
    both_summary["walk_held"] = share(walk_held, both)
    both_summary["look_held"] = share(look_held, both)
    return {
        "walk_only": rate(walk_ok, walk_held, mode == WALK),
        "look_only": rate(look_ok, look_held, mode == LOOK),
        "both": both_summary,
        "walk_pace": pace(episode.walk, episode.walk_active, rollout.positions, rollout.yaws, rollout.seconds),
        "walk_radius": radius_kept(episode.walk, episode.walk_active, rollout.positions, rollout.walk_hit),
        "decisions": rollout.decisions,
        "illegal_decisions": rollout.illegal,
        "teacher_label_illegal": rollout.teacher_illegal,
        "hold_ticks": HOLD_TICKS,
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
    """Run ``policy`` on fresh episodes and report per-mode success, held success, walk pace, radius and legality.

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
    env = environment(lanes, config.maximum_turn_degrees)
    kernels = _kernels(config.encoder, config.generalist, config.maximum_turn_degrees,
                       walk_stop_margin=config.walk_stop_margin, look_direction=config.look_direction,
                       look_frame=config.look_frame)
    key_goals, key_rollout = jax.random.split(jax.random.key(seed))
    episode = sample_episodes(key_goals, jnp.asarray(codes), held_out=held_out)
    decide = (lambda tick: tick.student) if isinstance(policy, str) else policy
    rollout = _rollout(env, params, episode, decide, kernels, key_rollout, record=False)
    name = policy if isinstance(policy, str) else getattr(policy, "__name__", "callable")
    return {"policy": name, "held_out": held_out, "seed": seed, "lanes": lanes, "modes": list(modes),
            **_summarise(episode, rollout)}


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
    parser.add_argument("--walk-stop-margin", type=float, default=defaults.walk_stop_margin,
                        help="walk_to v3 when positive: it stops this many blocks inside the radius "
                             "(walk_to.teacher.WALK_TO_V3_STOP_MARGIN_BLOCKS is 0.25); 0 is walk_to v2")
    parser.add_argument("--look-direction", action="store_true",
                        help="look_at's request also carries the unit direction to its target (frame: --look-frame)")
    parser.add_argument("--look-frame", choices=("body", "head"), default=defaults.look_frame,
                        help="the frame of --look-direction: body (default), or head: "
                             "measured from where the head points")
    args = parser.parse_args(argv)
    if args.walk_stop_margin < 0:
        parser.error("--walk-stop-margin must be at least 0: the stop is inside the radius")
    if args.look_frame == "head" and not args.look_direction:
        parser.error("--look-frame head needs --look-direction")

    config = DistillConfig(iterations=args.iterations, lanes=args.lanes, steps=args.steps, seed=args.seed,
                           walk_stop_margin=args.walk_stop_margin, look_direction=args.look_direction,
                           look_frame=args.look_frame)
    record = provenance()
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
    record["load_average"]["end"] = list(os.getloadavg())
    samples = int(result.dataset["labels"].shape[0])
    bytes_per_sample = sum(value[0].nbytes for value in result.dataset.values()) if samples else 0
    report = {
        "schema": "skill_agent_distill_v2",
        "config": dataclasses.asdict(config),
        "heads": list(HEAD_NAMES),
        "walk_heads": [int(i) for i in np.flatnonzero(WALK_HEADS)],
        "gaze_heads": [int(i) for i in np.flatnonzero(GAZE_HEADS)],
        "teachers": {
            "walk_to": {"version": walk_to_version(config.walk_stop_margin), "state": TEACHERS["walk_to"].state,
                        "stop_margin_blocks": config.walk_stop_margin},
            "look_at": {"version": TEACHERS["look_at"].version, "state": TEACHERS["look_at"].state},
        },
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
        "provenance": record,
    }
    if args.params_out:
        save_params(args.params_out, result.params)
    text = json.dumps(report, indent=1, default=json_default)
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
