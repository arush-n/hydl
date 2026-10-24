"""The v3 reference trainer (``docs/TRAINING_PLAN.md`` 8.3, stage V1): walk_to and look_at distilled into a fresh v3 trunk
and generalist, with memory, by DAgger over stored sequences.

The student reads the state layer's tokens (``runtime/state_tick.py``): ``layer.step`` each tick, then the trunk, which
carries a latent memory, a background summary and each remembered object's latent. Its trunk learns through time: each
rollout is cut into sequences of ``sequence_length`` decisions that overlap by ``burn_in``, and training unrolls the
trunk over each, the first ``burn_in`` decisions of a mid-episode sequence rebuilding the memory with no loss (R2D2,
Kapturowski et al. 2019; ``training/sequences.py``). The task, the teachers, the goals, the held-out rules and the
evaluation are ``distill.py``'s (A2), reused by import, with run C''s options by default (README 2.16, the run V1's gate
names): walk_to v3's stop margin, and the look request's direction measured in the head's frame. The control
(``--control``) is the same run with the trunk's memory, background and object latents removed.

CLI::

    PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym JAX_PLATFORMS=cpu nice -n 10 \\
        .venv/bin/python -m experimental.neural_agent.skill_agent.training.distill_v3 [--control] [--smoke] \\
        --out REPORT.json [--iterations 6 --lanes 128 --steps 2000 --seed 0 --eval-seed 77001 --eval-lanes 256]
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import math
import os
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experimental.neural_agent.skill_agent.bench import agent_models
from experimental.neural_agent.skill_agent.model.encoder import (
    EncoderConfig,
    initial_background,
    initial_memory,
    parameter_count,
)
from experimental.neural_agent.skill_agent.model.generalist import GeneralistConfig, GeneralistRequests, apply_generalist
from experimental.neural_agent.skill_agent.model.skill_module import Trunk
from experimental.neural_agent.skill_agent.neural_skills.walk_to.teacher import WALK_TO_V3_STOP_MARGIN_BLOCKS
from experimental.neural_agent.skill_agent.runtime import state_tick
from experimental.neural_agent.skill_agent.state import layer
from experimental.neural_agent.skill_agent.state import persona as pe
from experimental.neural_agent.skill_agent.training import blocks, distill, sequences
from experimental.neural_agent.skill_agent.training.sequences import PackedTokens, Sequences
from hytalegym.jax.combat.observation.v3.policy.layout import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES as HEAD_NAMES,
)
from hytalegym.jax.combat.observation.v3.policy.layout import (
    ARSENAL_POLICY_ACTION_HEAD_SIZES as HEAD_SIZES,
)
from hytalegym.jax.combat.types import AGENT_ENTITY


@dataclass(frozen=True)
class DistillV3Config:
    """What sizes a V1 run. The defaults are the full run's (``docs/TRAINING_PLAN.md`` 8.3); ``--smoke`` is one iteration of
    300 steps."""

    iterations: int = 6
    lanes: int = 128
    steps: int = 2000
    seed: int = 0
    learning_rate: float = 1e-3
    done_weight: float = 0.5
    maximum_turn_degrees: float = 45.0
    batch_sequences: int = 12  # with L = 90, about A2's 1,024 decisions a step
    sequence_length: int = 90  # L: a whole look_at episode (8.6)
    burn_in: int = 30  # B: the decisions of a mid-episode sequence that only rebuild its memory
    storage: str = "float16"  # the stored tokens' dtype, valid rows only (8.6: float16 costs nothing that matters)
    encoder: EncoderConfig = field(default_factory=lambda: agent_models.CONFIG)
    generalist: GeneralistConfig = field(default_factory=lambda: distill.DEFAULT_GENERALIST)
    walk_stop_margin: float = WALK_TO_V3_STOP_MARGIN_BLOCKS  # run C': walk_to v3 stops this far inside the radius
    look_direction: bool = True  # run C': the look request carries the unit direction to its target
    look_frame: str = "head"  # run C': that direction measured from where the head points

    def __post_init__(self) -> None:
        if not 0 <= self.burn_in < self.sequence_length:
            raise ValueError(f"burn_in must be in [0, sequence_length), got {self.burn_in} with {self.sequence_length}")
        if self.storage not in ("float16", "float32"):
            raise ValueError(f"storage must be float16 or float32, got {self.storage!r}")
        if self.look_frame not in ("body", "head"):
            raise ValueError(f"look_frame must be 'body' or 'head', got {self.look_frame!r}")
        if self.look_frame == "head" and not self.look_direction:
            raise ValueError("look_frame 'head' needs look_direction: it measures the look request's direction")

    def kernels(self) -> distill.Kernels:
        """``distill._kernels`` with this config's teachers and request options."""

        return distill._kernels(self.encoder, self.generalist, self.maximum_turn_degrees,
                                walk_stop_margin=self.walk_stop_margin, look_direction=self.look_direction,
                                look_frame=self.look_frame)


def memoryless(config: DistillV3Config) -> DistillV3Config:
    """The control: the same run with the trunk's memory, background and object latents removed. The state layer runs as
    before; the trunk cannot carry anything from tick to tick."""

    encoder = dataclasses.replace(config.encoder, memory_slots=0, background_slots=0, object_types=())
    return dataclasses.replace(config, encoder=encoder)


class Rollout3(NamedTuple):
    """One rollout on v3. ``summary`` is a :class:`distill.Rollout` (its ``samples`` None). The rest is kept only when
    recording: ``recording`` (``sequences.Recording``), ``decisions`` (per decision, stacked [T, B, ...]: the requests as
    ``GeneralistRequests`` fields, the teacher's labels, label mask and done, and ``keep``, inside the lane's horizon) and
    ``logits`` (the student head's logits per head, [T, B, size]). ``tick_seconds`` is the world's tick, measured on the
    neutral tick; ``tick_deviation`` the largest gap between any in-horizon decision's tick and it."""

    summary: distill.Rollout
    recording: sequences.Recording | None
    decisions: dict | None
    logits: dict | None
    tick_seconds: float
    tick_deviation: float


class Dataset(NamedTuple):
    """Sequences with what each decision needs: ``sequences`` (tokens packed, the stored carry), the requests, the
    teacher's labels, label mask and done flags, and ``loss`` (bool[n, L]: the decisions that carry the loss)."""

    sequences: Sequences
    requests: dict  # GeneralistRequests field -> [n, L, ...]
    labels: np.ndarray  # int32[n, L, 12]
    label_mask: np.ndarray  # bool[n, L, 12]
    done: np.ndarray  # bool[n, L, R]
    loss: np.ndarray  # bool[n, L]


class RunV3(NamedTuple):
    params: dict
    iterations: list[dict]


def _start_state(trunk_params: Any, lanes: int, encoder: EncoderConfig) -> layer.AgentState:
    """``agent_models.start_state`` for any encoder: each lane starts at the trunk's initial memory and background. That
    function reads ``agent_models.CONFIG``, which the control's parameters do not have."""

    return layer.initial_state(
        lanes, latent=initial_memory(trunk_params, encoder, lanes), persona=pe.sample_personas(jax.random.key(1), lanes),
        object_width=encoder.width, background=initial_background(trunk_params, encoder, lanes))


def _world_tick(combat: Any) -> float:
    """The world's tick in seconds, from the neutral tick's motion: every lane must share it (measured, not assumed)."""

    seconds = np.asarray(combat.last_motion_delta_seconds, np.float64)
    if not (np.all(seconds == seconds[0]) and np.isfinite(seconds[0]) and seconds[0] > 0):
        raise ValueError(f"the lanes do not share one positive tick length: {seconds}")
    return float(seconds[0])


@functools.lru_cache(maxsize=None)
def _layer_program(tick_seconds: float) -> Callable[..., Any]:
    """``layer.step`` with the agent's own pose as its perception channel, jitted per world tick. The channel is built
    inside the jit: a channel built per call is a new type each call, which would retrace every decision."""

    def run(state, observation, position, yaw, last_action, reset):
        feed = state_tick.own_pose_channel(observation, position, yaw)
        return layer.step(feed, state, last_action=last_action, reset=reset, tick_seconds=tick_seconds)

    return jax.jit(run)


def _student(head_logits: dict, action_mask: jax.Array, walk_active: jax.Array, look_active: jax.Array) -> jax.Array:
    """The student's factors (``distill.student_factors``' rule, on the head's logits): the masked argmax on each head of an
    active scope, neutral on every other head."""

    active = (walk_active[:, None] & jnp.asarray(distill.WALK_HEADS)[None, :]) | (
        look_active[:, None] & jnp.asarray(distill.GAZE_HEADS)[None, :])
    columns = []
    for head, name in enumerate(HEAD_NAMES):
        start, size = blocks.HEAD_STARTS[head], HEAD_SIZES[head]
        legal = action_mask[:, start:start + size]
        choice = jnp.argmax(jnp.where(legal, head_logits[name], -jnp.inf), axis=-1).astype(jnp.int32)
        columns.append(jnp.where(active[:, head], choice, int(blocks.NEUTRAL[head])))
    return jnp.stack(columns, axis=1)


@functools.lru_cache(maxsize=None)
def _decision_program(encoder: EncoderConfig, generalist: GeneralistConfig) -> Callable[..., Any]:
    """The trunk's step on the layer's tokens, then the head and the student's choice, jitted."""

    def run(trunk_params, head_params, tokens, state, reset, requests, action_mask, walk_active, look_active):
        view, next_state = state_tick.trunk_step(Trunk(encoder, trunk_params, ""), tokens, state, reset)
        out = apply_generalist(head_params, view.encoded, view.masks, requests, generalist)
        return next_state, out.head_logits, _student(out.head_logits, action_mask, walk_active, look_active)

    return jax.jit(run)


def _host(requests: GeneralistRequests) -> dict:
    return {name: np.asarray(getattr(requests, name)) for name in GeneralistRequests._fields}


def _rollout(env, params: dict, episode: distill.Episodes, decide: Callable[[distill.Tick], Any],
             config: DistillV3Config, kernels: distill.Kernels, key: jax.Array, *, record: bool) -> Rollout3:
    """``distill._rollout`` on v3: every lane runs its horizon, one decision per tick. A decision is the layer's step on
    the flat world with the agent's own pose, the recording (when ``record``), the trunk's step with the carried state,
    the head, and the student's choice; ``decide`` picks the executed factors. A lane's decisions past its horizon are
    neither counted nor recorded for its use. A recording keeps the carried state at each sequence start (every
    ``sequence_length - burn_in`` decisions, the cut's starts)."""

    batch = env.batch
    horizon = np.asarray(episode.horizon)
    ticks = int(horizon.max())
    keys = jax.random.split(key, ticks + 2)
    spawn = jnp.broadcast_to(jnp.asarray(blocks.SPAWN_XZ, dtype=jnp.float32), (batch, 2))
    other = jnp.broadcast_to(jnp.asarray(blocks.OTHER_XYZ, dtype=jnp.float32), (batch, 3))
    carry = env.reset(jax.random.split(keys[0], batch), spawn, episode.spawn_yaw, other)
    neutral = jnp.broadcast_to(jnp.asarray(blocks.NEUTRAL, dtype=jnp.int32), (batch, blocks.HEAD_COUNT))
    carry, observation, _ = env.step(carry, neutral, jax.random.split(keys[1], batch))
    tick_seconds = _world_tick(carry.runtime.combat)
    layer_step = _layer_program(tick_seconds)
    decide_step = _decision_program(config.encoder, config.generalist)
    state = _start_state(params["trunk"], batch, config.encoder)
    recording = sequences.Recording(config.encoder, batch) if record else None
    carry_at = set(range(0, ticks, config.sequence_length - config.burn_in))
    dtype = np.dtype(config.storage)

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
    per_decision: list[dict] = []
    per_logits: list[dict] = []

    def record_instant(index: int, combat: Any) -> None:
        positions[index] = np.asarray(combat.position[:, AGENT_ENTITY][:, [0, 2]])
        yaws[index] = np.asarray(combat.yaw[:, AGENT_ENTITY])

    record_instant(0, carry.runtime.combat)
    last = neutral
    for index in range(ticks):
        combat = carry.runtime.combat
        position = combat.position[:, AGENT_ENTITY]
        yaw = combat.yaw[:, AGENT_ENTITY]
        requests = kernels.requests(episode, position, yaw, jnp.float32(index), combat.agent_head_yaw,
                                    combat.agent_head_pitch)
        teacher = kernels.teacher(combat, episode)
        _, last_action, _, _ = state_tick.host_inputs(observation, last, jnp.zeros((batch,), jnp.bool_))
        reset = jnp.full((batch,), index == 0)
        before = state
        tokens, after = layer_step(before, observation, position, yaw, last_action, reset)
        if recording is not None:
            sequences.record(recording, before, tokens, after, reset, dtype=dtype, carry_at=carry_at)
        state, head_logits, student = decide_step(params["trunk"], params["generalist"], tokens, after, reset, requests,
                                                  carry.action_mask, episode.walk_active, episode.look_active)
        tick = distill.Tick(combat, carry.action_mask, episode, tokens, requests, teacher, student)
        executed = jnp.asarray(decide(tick), dtype=jnp.int32)

        keep = index < horizon
        mask = np.asarray(carry.action_mask)
        decisions += int(keep.sum())
        illegal += int(np.sum(keep & ~blocks.legal_rows(mask, np.asarray(executed))))
        teacher_illegal += int(np.sum(keep & ~blocks.legal_rows(mask, np.asarray(teacher.factors))))
        if recording is not None:
            per_decision.append({"requests": _host(requests), "labels": np.asarray(teacher.factors),
                                 "label_mask": np.asarray(teacher.mask), "done": np.asarray(teacher.done), "keep": keep})
            per_logits.append({name: np.asarray(value) for name, value in head_logits.items()})

        carry, observation, _ = env.step(carry, executed, jax.random.split(keys[index + 2], batch))
        last = executed
        done_count = index + 1
        combat = carry.runtime.combat
        record_instant(done_count, combat)
        seconds[index] = np.asarray(combat.last_motion_delta_seconds)
        walk_ok, look_ok = (np.asarray(value) for value in kernels.success(combat, episode))
        walk_hit = np.where((walk_hit < 0) & walk_active & walk_ok & (done_count <= walk_horizon),
                            done_count, walk_hit).astype(np.int32)
        look_hit = np.where((look_hit < 0) & look_active & look_ok & (done_count <= look_horizon),
                            done_count, look_hit).astype(np.int32)
        walk_streak, walk_held = blocks.hold_update(walk_streak, walk_held, walk_ok, done_count, walk_horizon,
                                                    walk_active)
        look_streak, look_held = blocks.hold_update(look_streak, look_held, look_ok, done_count, look_horizon,
                                                    look_active)

    inside = np.arange(ticks)[:, None] < horizon[None, :]
    summary = distill.Rollout(walk_hit=walk_hit, look_hit=look_hit, positions=positions, yaws=yaws, seconds=seconds,
                              decisions=decisions, illegal=illegal, teacher_illegal=teacher_illegal, samples=None,
                              walk_held=walk_held, look_held=look_held)
    deviation = float(np.abs(seconds - tick_seconds)[inside].max())
    if recording is None:
        return Rollout3(summary, None, None, None, tick_seconds, deviation)
    stacked = {name: np.stack([step[name] for step in per_decision]) for name in ("labels", "label_mask", "done", "keep")}
    stacked["requests"] = {name: np.stack([step["requests"][name] for step in per_decision])
                           for name in GeneralistRequests._fields}
    logits = {name: np.stack([step[name] for step in per_logits]) for name in HEAD_NAMES}
    return Rollout3(summary, recording, stacked, logits, tick_seconds, deviation)


def _gather(cut: Sequences, values: np.ndarray) -> np.ndarray:
    """``values`` [T, B, ...] at each sequence's decisions as [n, L, ...]: zeros past each sequence's ``steps``."""

    position = np.arange(cut.reset.shape[1])
    rows = np.minimum(cut.first[:, None] + position[None, :], values.shape[0] - 1)
    out = values[rows, cut.lane[:, None]]
    out[position[None, :] >= cut.steps[:, None]] = 0
    return out


def _dataset(rollout: Rollout3, horizon: np.ndarray, config: DistillV3Config) -> Dataset:
    """The rollout's sequences: each lane cut every ``sequence_length - burn_in`` decisions, each sequence ending at the lane's
    horizon. A sequence that starts at a reset carries the loss on all its decisions; any other sequence carries it past
    its first ``burn_in`` (those rebuild the memory). Every decision inside a horizon gets exactly one loss term."""

    cut = sequences.cut(rollout.recording, config.sequence_length, stride=config.sequence_length - config.burn_in,
                        ends=horizon)
    position = np.arange(cut.reset.shape[1])
    inside = position[None, :] < cut.steps[:, None]
    loss = inside & _gather(cut, rollout.decisions["keep"]) & (
        cut.reset[:, :1] | (position[None, :] >= config.burn_in))
    return Dataset(cut, {name: _gather(cut, value) for name, value in rollout.decisions["requests"].items()},
                   _gather(cut, rollout.decisions["labels"]), _gather(cut, rollout.decisions["label_mask"]),
                   _gather(cut, rollout.decisions["done"]), loss)


def _concat(parts: Sequence[Dataset]) -> Dataset:
    """Datasets as one: their sequences appended, each packed row's sequence index offset by the sequences before it."""

    cuts = [part.sequences for part in parts]
    offsets = np.cumsum([0] + [len(cut.lane) for cut in cuts])[:-1]
    tokens = {}
    for name, value in cuts[0].tokens.items():
        if isinstance(value, PackedTokens):
            index = np.concatenate([cut.tokens[name].index + np.array([offset, 0, 0], np.int32)
                                    for cut, offset in zip(cuts, offsets)])
            features = np.concatenate([cut.tokens[name].features for cut in cuts])
            total = sum(len(cut.lane) for cut in cuts)
            tokens[name] = PackedTokens(index, features, (total,) + tuple(value.shape[1:]))
        else:
            tokens[name] = (np.concatenate([cut.tokens[name][0] for cut in cuts]),
                            np.concatenate([cut.tokens[name][1] for cut in cuts]))
    cut = jax.tree_util.tree_map(lambda *leaves: np.concatenate(leaves), *[c._replace(tokens={}) for c in cuts])
    rest = jax.tree_util.tree_map(lambda *leaves: np.concatenate(leaves),
                                  *[(p.requests, p.labels, p.label_mask, p.done, p.loss) for p in parts])
    return Dataset(cut._replace(tokens=tokens), *rest)


def _filled(data: Dataset) -> dict[str, int]:
    """Each token type's slots that the dataset uses: the most rows any stored decision fills (at least 1), and at least
    the rows a request points at (``entity_pointer`` into the entity rows, ``cell_pointer`` into the block candidates).

    The trunk masks empty slots, so a batch cut to these computes what the full one does, to float32 rounding, at a
    fraction of the cost: on the trainers' flat world 39 of the 356 slots hold a row (``docs/TRAINING_PLAN.md`` 8.8).
    That needs every decision's valid rows to come first in each type, as ``layer.step`` packs them; checked here.

    Raises:
        ValueError: If a type's valid rows do not come first in some decision.
    """

    keep = {}
    for name, packed in data.sequences.tokens.items():
        index = packed.index
        if not len(index):
            keep[name] = 1
            continue
        cell = index[:, 0].astype(np.int64) * packed.shape[1] + index[:, 1]
        counts = np.bincount(cell, minlength=packed.shape[0] * packed.shape[1])
        top = np.zeros_like(counts)
        np.maximum.at(top, cell, index[:, 2].astype(np.int64) + 1)
        if not np.array_equal(top, counts):
            raise ValueError(f"{name}: the valid rows do not come first, so cutting the empty slots would drop rows")
        keep[name] = max(int(top.max()), 1)
    for name, pointer in (("entity", "entity_pointer"), ("block_candidate", "cell_pointer")):
        if name in keep:
            keep[name] = max(keep[name], int(data.requests[pointer].max()) + 1)
    return keep


def _batch(data: Dataset, rows: np.ndarray, keep: dict[str, int] | None = None) -> dict:
    """The chosen sequences as a dense batch: their tokens unpacked, the per-decision fields as they are. With ``keep``
    (:func:`_filled`), each token type and its object slots are cut to that many slots."""

    chosen = sequences.unpack(sequences.take(data.sequences, rows))
    if keep is not None:
        chosen = chosen._replace(
            tokens={name: (features[:, :, :keep[name]], valid[:, :, :keep[name]])
                    for name, (features, valid) in chosen.tokens.items()},
            slots={name: value[:, :, :keep[name]] for name, value in chosen.slots.items()})
    return {"sequences": chosen,
            "requests": {name: value[rows] for name, value in data.requests.items()},
            "labels": data.labels[rows], "label_mask": data.label_mask[rows], "done": data.done[rows],
            "loss": data.loss[rows]}


def _flat(leaf: Any) -> jax.Array:
    """[n, L, ...] as [n * L, ...]: the decisions of a batch."""

    return jnp.reshape(leaf, (-1,) + tuple(jnp.shape(leaf)[2:]))


def head_outputs(params: dict, batch: dict, config: DistillV3Config) -> Any:
    """The generalist's outputs at every decision of a batch, flattened to [n * L, ...]: the trunk unrolled over each
    sequence (``sequences.unroll``: a reset-start from the current initial memory, a mid-episode one from its stored
    carry), then the head on each decision."""

    data = batch["sequences"]
    encoded = jax.tree_util.tree_map(_flat, sequences.unroll(params["trunk"], config.encoder, data))
    masks = {name: _flat(valid) for name, (_, valid) in data.tokens.items()}
    requests = GeneralistRequests(**{name: _flat(batch["requests"][name]) for name in GeneralistRequests._fields})
    return apply_generalist(params["generalist"], encoded, masks, requests, config.generalist)


def loss_v3(params: dict, batch: dict, config: DistillV3Config) -> jax.Array:
    """``distill.loss_fn`` on the decisions that carry the loss: the cross-entropy over the supervised heads, plus
    ``done_weight`` times the done BCE over the valid requests. Each term is weighted by the decision's loss mask."""

    out = head_outputs(params, batch, config)
    weight = _flat(batch["loss"])
    rows = jnp.arange(weight.shape[0])
    labels = _flat(batch["labels"])
    supervised = _flat(batch["label_mask"]) & weight[:, None]
    total = 0.0
    for head, name in enumerate(HEAD_NAMES):
        log_probs = jax.nn.log_softmax(out.head_logits[name], axis=-1)
        total = total - jnp.sum(jnp.where(supervised[:, head], log_probs[rows, labels[:, head]], 0.0))
    cross_entropy = total / jnp.maximum(jnp.sum(supervised.astype(jnp.float32)), 1.0)
    valid = _flat(batch["requests"]["valid"]) & weight[:, None]
    done_bce = optax.sigmoid_binary_cross_entropy(out.status[..., 0], _flat(batch["done"]).astype(jnp.float32))
    done_loss = jnp.sum(jnp.where(valid, done_bce, 0.0)) / jnp.maximum(jnp.sum(valid.astype(jnp.float32)), 1.0)
    return cross_entropy + config.done_weight * done_loss


def make_train_step(config: DistillV3Config, optimizer: optax.GradientTransformation) -> Callable[..., Any]:
    """One Adam step on a batch of sequences; returns the loss and the gradient's global norm."""

    def step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(lambda p: loss_v3(p, batch, config))(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, optax.global_norm(grads)

    return jax.jit(step)


def _stored_bytes(data: Dataset) -> int:
    return sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(data) if isinstance(leaf, np.ndarray))


def _peak_rss() -> int:
    """This process's peak resident set size in bytes (``ru_maxrss`` is bytes on macOS, kilobytes elsewhere)."""

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _collect(env, params: dict, key: jax.Array, config: DistillV3Config, beta: float, kernels: distill.Kernels
             ) -> tuple[Dataset, dict]:
    """One DAgger iteration's episodes: each lane runs the teacher with probability ``beta``, else the student."""

    key_mode, key_goals, key_mix, key_rollout = jax.random.split(key, 4)
    mode = jax.random.randint(key_mode, (config.lanes,), 0, len(distill.MODES))
    episode = distill.sample_episodes(key_goals, mode, held_out=False)
    use_teacher = jax.random.uniform(key_mix, (config.lanes,)) < beta

    def decide(tick: distill.Tick) -> jax.Array:
        return jnp.where(use_teacher[:, None], tick.teacher.factors, tick.student)

    rollout = _rollout(env, params, episode, decide, config, kernels, key_rollout, record=True)
    horizon = np.asarray(episode.horizon)
    data = _dataset(rollout, horizon, config)
    stats = {
        "teacher_lanes": int(np.asarray(use_teacher).sum()),
        "lanes_by_mode": {name: int((np.asarray(mode) == code).sum()) for code, name in enumerate(distill.MODES)},
        "mean_horizon_decisions": float(horizon.mean()),
        "decisions": rollout.summary.decisions,
        "new_sequences": int(len(data.loss)),
        "executed_illegal": rollout.summary.illegal,
        "teacher_label_illegal": rollout.summary.teacher_illegal,
        "tick_seconds": rollout.tick_seconds,
        "tick_deviation": rollout.tick_deviation,
    }
    return data, stats


def run_dagger(config: DistillV3Config = DistillV3Config()) -> RunV3:
    """``distill.run_dagger``'s loop on v3: collect with ``beta`` (the teacher's share of lanes; 1, then halving), add the
    sequences to the aggregate, then ``config.steps`` Adam steps on the aggregate, per iteration."""

    env = blocks.environment(config.lanes, config.maximum_turn_degrees)
    kernels = config.kernels()
    key, init_key = jax.random.split(jax.random.key(config.seed))
    params = distill.init_params(init_key, distill.DistillConfig(encoder=config.encoder, generalist=config.generalist))
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)
    train_step = make_train_step(config, optimizer)
    rng = np.random.default_rng(config.seed)

    data: Dataset | None = None
    iterations: list[dict] = []
    for index in range(config.iterations):
        beta = 1.0 if index == 0 else 0.5 ** index
        started = time.perf_counter()
        key, collect_key = jax.random.split(key)
        part, stats = _collect(env, params, collect_key, config, beta, kernels)
        data = part if data is None else _concat([data, part])
        collected = time.perf_counter()

        count = len(data.loss)
        keep = _filled(data)
        losses, norms, step_seconds = [], [], []
        for step in range(config.steps):
            begun = time.perf_counter()
            rows = rng.integers(0, count, size=config.batch_sequences)
            params, opt_state, loss, norm = train_step(params, opt_state, _batch(data, rows, keep))
            loss = float(loss)
            step_seconds.append(time.perf_counter() - begun)
            if not math.isfinite(loss):
                raise FloatingPointError(f"iteration {index}: the loss is {loss} at step {step}")
            losses.append(loss)
            norms.append(float(norm))
            print(f"iteration {index} step {step}: loss {loss:.6f} grad {norms[-1]:.4g} {step_seconds[-1]:.3f} s",
                  file=sys.stderr, flush=True)
        points = sorted(set(range(0, config.steps, 50)) | {config.steps - 1}) if config.steps else []
        record = {
            "iteration": index,
            "beta": beta,
            **stats,
            "sequences_total": count,
            "stored_bytes_total": _stored_bytes(data),
            "stored_bytes_new": _stored_bytes(part),
            "steps": config.steps,
            "batch_sequences": config.batch_sequences,
            "token_slots": {"kept": keep, "total": sum(keep.values()),
                            "capacity": sum(packed.shape[2] for packed in data.sequences.tokens.values())},
            "collect_seconds": round(collected - started, 2),
            "compile_seconds": round(step_seconds[0], 2) if step_seconds else None,
            "step_seconds": None if len(step_seconds) < 2 else {
                "p50": round(float(np.percentile(step_seconds[1:], 50)), 3),
                "p95": round(float(np.percentile(step_seconds[1:], 95)), 3)},
            "loss": {str(point): losses[point] for point in points},
            "grad_norm": {"first": norms[0], "last": norms[-1], "max": max(norms)} if norms else None,
            "train_seconds": round(time.perf_counter() - collected, 2),
            "peak_rss_bytes": _peak_rss(),
            "parameters": {"trunk": parameter_count(params["trunk"]),
                           "generalist": parameter_count(params["generalist"])},
        }
        iterations.append(record)
        print(f"iteration {index} beta {beta:.3f}: sequences {count}, loss {losses[0]} -> {losses[-1]}, "
              f"collect {record['collect_seconds']} s, train {record['train_seconds']} s", file=sys.stderr, flush=True)
    return RunV3(params=params, iterations=iterations)


def evaluate(params: dict, config: DistillV3Config, *, seed: int, lanes: int, held_out: bool) -> dict:
    """The student on fresh episodes, as ``distill.evaluate``: the same seed gives the same goals and spawns, and the same
    report (``distill._summarise``)."""

    env = blocks.environment(lanes, config.maximum_turn_degrees)
    kernels = config.kernels()
    key_goals, key_rollout = jax.random.split(jax.random.key(seed))
    episode = distill.sample_episodes(key_goals, jnp.asarray(np.arange(lanes) % len(distill.MODES)), held_out=held_out)
    started = time.perf_counter()
    rollout = _rollout(env, params, episode, lambda tick: tick.student, config, kernels, key_rollout, record=False)
    report = {"policy": "student", "held_out": held_out, "seed": seed, "lanes": lanes, "modes": list(distill.MODES),
              **distill._summarise(episode, rollout.summary)}
    report["tick_seconds"] = rollout.tick_seconds
    report["eval_seconds"] = round(time.perf_counter() - started, 1)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    defaults = DistillV3Config()
    parser = argparse.ArgumentParser(description="V1: the v3 reference trainer (walk_to and look_at, with trunk memory).")
    parser.add_argument("--control", action="store_true",
                        help="the memory-less control: the trunk's memory, background and object latents removed")
    parser.add_argument("--smoke", action="store_true", help="the timed smoke: one iteration of 300 steps")
    parser.add_argument("--iterations", type=int, default=defaults.iterations)
    parser.add_argument("--lanes", type=int, default=defaults.lanes)
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--batch-sequences", type=int, default=defaults.batch_sequences)
    parser.add_argument("--sequence-length", type=int, default=defaults.sequence_length)
    parser.add_argument("--burn-in", type=int, default=defaults.burn_in)
    parser.add_argument("--walk-stop-margin", type=float, default=defaults.walk_stop_margin,
                        help="walk_to v3's stop margin in blocks (run C': 0.25); 0 is walk_to v2, as A2")
    parser.add_argument("--no-look-direction", action="store_true",
                        help="leave the look request's direction empty, as A2 (run C' carries it)")
    parser.add_argument("--look-frame", choices=("body", "head"), default=defaults.look_frame,
                        help="the frame of the look direction (run C': head)")
    parser.add_argument("--eval-seed", type=int, default=77001)
    parser.add_argument("--eval-lanes", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--params-out", type=Path, help="save the trained parameters here (.npz, blocks.save_params)")
    args = parser.parse_args(argv)
    config = DistillV3Config(iterations=args.iterations, lanes=args.lanes, steps=args.steps, seed=args.seed,
                             batch_sequences=args.batch_sequences, sequence_length=args.sequence_length,
                             burn_in=args.burn_in, walk_stop_margin=args.walk_stop_margin,
                             look_direction=not args.no_look_direction,
                             look_frame="body" if args.no_look_direction else args.look_frame)
    if args.control:
        config = memoryless(config)
    if args.smoke:
        config = dataclasses.replace(config, iterations=1, steps=300)
    record = blocks.provenance()
    started = time.perf_counter()
    result = run_dagger(config)
    evaluation = {}
    for region, held_out in (("seen", False), ("held_out", True)):
        evaluation[region] = evaluate(result.params, config, seed=args.eval_seed, lanes=args.eval_lanes,
                                      held_out=held_out)
    record["load_average"]["end"] = list(os.getloadavg())
    report = {
        "schema": "skill_agent_distill_v3",
        "control": args.control,
        "smoke": args.smoke,
        "config": dataclasses.asdict(config),
        "heads": list(HEAD_NAMES),
        "teachers": {"walk_to": distill.walk_to_version(config.walk_stop_margin),
                     "look_at": distill.TEACHERS["look_at"].version},
        "held_out_rules": {"walk_rear_left_bearing_degrees": list(distill.REAR_LEFT_DEGREES),
                           "look_yaw_offset_degrees": list(distill.LOOK_HELD_OUT_DEGREES)},
        "training": result.iterations,
        "evaluation": evaluation,
        "seconds": round(time.perf_counter() - started, 1),
        "provenance": record,
    }
    if args.params_out:
        blocks.save_params(args.params_out, result.params)
    text = json.dumps(report, indent=1, default=blocks.json_default)
    args.out.write_text(text + "\n")
    sys.stdout.write(text + "\n")
    return 0


__all__ = ["DistillV3Config", "Dataset", "Rollout3", "evaluate", "head_outputs", "loss_v3", "memoryless", "run_dagger"]


if __name__ == "__main__":
    raise SystemExit(main())
