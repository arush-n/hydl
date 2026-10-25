"""DAgger for one skill on the v3 trunk (``docs/TRAINING_PLAN.md`` 8.3, stage V2): ``skill_dagger``'s method on the frozen,
recurrent trunk that ``distill_v3`` runs.

The trunk is V1's: it is loaded from its export and never trained. Each decision:

1. the state layer steps on the flat world with the agent's own pose (``distill_v3._layer_program``);
2. the frozen trunk steps with its carried state (``state_tick.trunk_step``), so no gradient runs through time;
3. every active skill's request is built from its spec (``skill_dagger.build_requests``);
4. the trained skill acts by its teacher (probability beta) or by its module, read on the trunk's outputs;
5. each skill beside it acts by its teacher (``skill_dagger.teach`` and ``merge``).

Per decision the dataset stores what the module reads: the trunk's ``context`` and each token type's embeddings, as
their valid rows only (``sequences.PackedRows``, float16 by default), the masks, the requests, and the teacher's labels
on the trained skill's scope with its done. Batches are random decisions, cut to the slots the data fills (``_filled``,
as ``distill_v3`` does). The loss is ``skill_dagger``'s on the module alone: the trunk's outputs are data, so no trunk
parameter is an input and no gradient reaches the trunk. The evaluation is ``skill_dagger.summarise``'s report on fresh
v3 rollouts, with the schema ``skill_module_v1`` that ``training/bars.py --skill`` scores.

CLI::

    python -m experimental.neural_agent.skill_agent.training.skill_dagger_v3 --skill walk_to --trunk TRUNK.npz \\
        [--smoke] --out REPORT.json --module-out MODULE.npz

``--smoke`` is one iteration of 200 steps.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experimental.neural_agent.skill_agent.model.encoder import EncoderOutput
from experimental.neural_agent.skill_agent.model.freezing import params_sha256
from experimental.neural_agent.skill_agent.model.generalist import GeneralistConfig, GeneralistRequests
from experimental.neural_agent.skill_agent.model.skill_module import (
    SkillModule,
    Trunk,
    init_module,
    load_module,
    load_trunk,
    module_outputs,
    save_module,
    scope_heads,
)
from experimental.neural_agent.skill_agent.neural_skills.spec import VERBS, Pose, SkillSpec
from experimental.neural_agent.skill_agent.runtime import state_tick
from experimental.neural_agent.skill_agent.training import blocks
from experimental.neural_agent.skill_agent.training import distill_v3 as v3
from experimental.neural_agent.skill_agent.training.sequences import PackedRows
from experimental.neural_agent.skill_agent.training.skill_dagger import (
    Lanes,
    SkillTrainConfig,
    Strategy,
    build_requests,
    merge,
    summarise,
    teach,
)
from hytalegym.jax.combat.types import AGENT_ENTITY

STORAGE = "float16"
"""The stored trunk outputs' dtype (``docs/TRAINING_PLAN.md`` 8.6: float16 costs nothing that matters)."""


class Kernels(NamedTuple):
    requests: Callable[..., GeneralistRequests]
    teachers: Callable[..., Any]
    trunk: Callable[..., Any]  # (trunk params, tokens, state, reset) -> (StateView, state)
    student: Callable[..., Any]  # (module params, action mask, encoded, masks, requests, active) -> (factors, logits)
    success: Callable[..., jax.Array]
    points: Callable[..., GeneralistRequests]  # (view, requests) -> requests with each entity handle as its token row
    counts: Callable[..., jax.Array]  # (combat, lanes) -> bool[B, S]: the ticks each skill's share counts (holding skills)


class Decisions(NamedTuple):
    """The stored decisions, one row each, in rollout order: the trunk's outputs the module reads (``context``, and each
    token type's embeddings as valid rows, packed), the masks, the requests, and the teacher's labels on the trained
    skill's scope with its done."""

    context: np.ndarray  # [N, O], the storage dtype
    masks: dict  # token type -> bool[N, N_t]
    tokens: dict  # token type -> PackedRows: index (decision, row), features [M, width], shape (N, N_t, width)
    requests: dict  # GeneralistRequests field -> [N, ...]
    labels: np.ndarray  # int32[N, 12]
    label_mask: np.ndarray  # bool[N, 12]: the trained skill's scope heads
    done: np.ndarray  # bool[N]


class Trace(NamedTuple):
    """One rollout, with the fields ``skill_dagger.summarise`` and the skills' metrics read, the stored decisions
    (``samples``, when recorded), and the module's logits per head at those decisions (``logits``, when kept). ``counted``
    and ``counted_ok`` are a holding skill's share: the ticks inside its horizon that count, and those where it succeeded."""

    hit: np.ndarray  # int32[B, S]
    held: np.ndarray  # int32[B, S]
    positions: np.ndarray  # float32[T+1, B, 2]
    yaws: np.ndarray  # float32[T+1, B]
    seconds: np.ndarray  # float32[T, B]
    decisions: int
    illegal: int
    teacher_illegal: int
    samples: Decisions | None
    logits: dict | None
    counted: np.ndarray  # int32[B, S]
    counted_ok: np.ndarray  # int32[B, S]


class TrainResult(NamedTuple):
    module: SkillModule
    dataset: Decisions
    iterations: list[dict]


def make_kernels(strategy: Strategy, trunk: Trunk, known: tuple[int, ...]) -> Kernels:
    skills = strategy.skills
    spec, config = strategy.spec, strategy.config.module

    def student(module_params: Any, action_mask: jax.Array, encoded: EncoderOutput, masks: dict,
                requests: GeneralistRequests, active: jax.Array) -> tuple[jax.Array, dict]:
        """The module's action: the masked argmax on each head of its scope where its skill is active, neutral
        elsewhere (``skill_dagger.student_factors``' rule, on the trunk's outputs)."""
        out = module_outputs(module_params, config, known, spec.verb, encoded, masks, requests)
        own = active[:, None] & jnp.asarray(scope_heads(spec.scope))[None, :]
        columns = []
        for head, name in enumerate(blocks.HEAD_NAMES):
            start, size = blocks.HEAD_STARTS[head], blocks.HEAD_SIZES[head]
            legal = action_mask[:, start:start + size]
            choice = jnp.argmax(jnp.where(legal, out.head_logits[name], -jnp.inf), axis=-1).astype(jnp.int32)
            columns.append(jnp.where(own[:, head], choice, int(blocks.NEUTRAL[head])))
        return jnp.stack(columns, axis=1), out.head_logits

    def trunk_step(params: Any, tokens: dict, state: Any, reset: jax.Array) -> tuple[Any, Any]:
        return state_tick.trunk_step(Trunk(trunk.config, params, trunk.sha256), tokens, state, reset)

    return Kernels(
        requests=jax.jit(functools.partial(build_requests, skills)),
        teachers=jax.jit(functools.partial(teach, skills)),
        trunk=jax.jit(trunk_step),
        student=jax.jit(student),
        success=jax.jit(lambda combat, lanes: jnp.stack(
            [skill.success(combat, goals) for skill, goals in zip(skills, lanes.goals)], axis=1)),
        points=jax.jit(lambda view, requests: state_tick.point(view, requests, requests.entity_pointer)),
        counts=jax.jit(lambda combat, lanes: jnp.stack(
            [spec.counts(combat, goals) if spec.holding and spec.counts is not None
             else jnp.ones((combat.position.shape[0],), dtype=jnp.bool_)
             for spec, goals in zip(skills, lanes.goals)], axis=1)),
    )


def placed_other(lanes: Lanes, batch: int) -> jax.Array:
    """float32[B, 3]: entity 1 at reset, per lane. The trained skill's goals place it where they carry ``other_xyz``
    (come_here's do); otherwise it stands at ``blocks.OTHER_XYZ``, where every other skill spawns it."""

    placed = getattr(lanes.goals[0], "other_xyz", None)
    if placed is None:
        return jnp.broadcast_to(jnp.asarray(blocks.OTHER_XYZ, dtype=jnp.float32), (batch, 3))
    return jnp.asarray(placed, dtype=jnp.float32)


def _pack(features: np.ndarray, valid: np.ndarray, dtype: np.dtype) -> PackedRows:
    """One token type's valid rows of [n, N, F] features, in ``dtype``, indexed by (decision, row): as
    ``sequences.record`` packs a decision."""

    return PackedRows(np.argwhere(valid).astype(np.int32), features[valid].astype(dtype), features.shape)


def _store(view: state_tick.StateView, requests: GeneralistRequests, taught: Any, keep: np.ndarray,
           dtype: np.dtype) -> Decisions:
    """One tick's decisions for the lanes inside their horizons (``keep``)."""

    rows = np.flatnonzero(keep)
    encoded = view.encoded
    masks = {name: np.asarray(view.masks[name], bool)[rows] for name in encoded.tokens}
    tokens = {name: _pack(np.asarray(encoded.tokens[name], np.float32)[rows], masks[name], dtype)
              for name in encoded.tokens}
    return Decisions(
        context=np.asarray(encoded.context, np.float32)[rows].astype(dtype),
        masks=masks,
        tokens=tokens,
        requests={name: np.asarray(getattr(requests, name))[rows] for name in GeneralistRequests._fields},
        labels=np.asarray(taught.labels, np.int32)[rows],
        label_mask=np.asarray(taught.mask, bool)[rows],
        done=np.asarray(taught.done, bool)[rows],
    )


def _join(parts: Sequence[Decisions]) -> Decisions:
    """The stored decisions of several rollouts as one, in order: each packed row's decision index offset by the
    decisions before it."""

    counts = [len(part.labels) for part in parts]
    offsets = np.cumsum([0] + counts[:-1]).astype(np.int32)

    def cat(get: Callable[[Decisions], np.ndarray]) -> np.ndarray:
        return np.concatenate([get(part) for part in parts])

    tokens = {}
    for name, first in parts[0].tokens.items():
        index = np.concatenate([part.tokens[name].index + np.array([offset, 0], np.int32)
                                for part, offset in zip(parts, offsets)])
        tokens[name] = PackedRows(index, cat(lambda part: part.tokens[name].features), (sum(counts),) + first.shape[1:])
    return Decisions(
        context=cat(lambda part: part.context),
        masks={name: cat(lambda part: part.masks[name]) for name in parts[0].masks},
        tokens=tokens,
        requests={name: cat(lambda part: part.requests[name]) for name in parts[0].requests},
        labels=cat(lambda part: part.labels),
        label_mask=cat(lambda part: part.label_mask),
        done=cat(lambda part: part.done),
    )


SLOT_BUCKET = 8
"""The token-slot cut's bucket (``slot_cut``): each token type's kept slots are a multiple of this. They grow when a
decision fills more and are never fewer within a rollout, so the trunk compiles for a few shapes only."""

LANE_BUCKET = 16
"""Lane compaction's bucket (``compact_lanes``): the lanes run each tick are a multiple of this, padded with
indices past the batch, which are dropped on the way back."""


@jax.jit
def _take(tree: Any, pick: jax.Array) -> Any:
    """Each array's lanes at ``pick`` (int32[K], along the first axis). A pad index past the batch reads the last
    lane. Jitted: the state is hundreds of small arrays, and eager gathers would dispatch one by one each tick."""

    return jax.tree_util.tree_map(lambda leaf: jnp.take(leaf, pick, axis=0, mode="clip"), tree)


@jax.jit
def _put(tree: Any, pick: jax.Array, part: Any) -> Any:
    """``tree`` with its lanes at ``pick`` replaced by ``part`` (indices past the batch are dropped)."""

    return jax.tree_util.tree_map(lambda full, sub: full.at[pick].set(sub, mode="drop"), tree, part)


def _live_pick(index: int, horizon: np.ndarray, batch: int) -> tuple[np.ndarray, np.ndarray]:
    """The lanes run at decision ``index`` under ``compact_lanes``: those whose decision still counts, inside the
    horizon or in the hold after it (``blocks.HOLD_TICKS``), padded to LANE_BUCKET with indices past the batch.
    Returns the pick int32[K] and the real bool[K]."""

    live = np.flatnonzero(index < horizon + blocks.HOLD_TICKS)
    size = min(batch, max(LANE_BUCKET, -(-len(live) // LANE_BUCKET) * LANE_BUCKET))
    pick = np.full(size, batch, dtype=np.int32)
    pick[:len(live)] = live
    return pick, pick < batch


def _kept_slots(tokens: dict, cell_pointer: jax.Array, kept: dict[str, int]) -> dict[str, int]:
    """Each token type's kept slots for this decision: through the last valid row any lane holds (and past the block
    candidate a request points at), rounded up to SLOT_BUCKET, and never fewer than ``kept`` had."""

    last = {name: jnp.max(jnp.where(valid, jnp.arange(1, valid.shape[1] + 1), 0))
            for name, (_, valid) in tokens.items()}
    host = jax.device_get({**last, "pointer": jnp.max(cell_pointer) + 1})
    grown = {}
    for name, (_, valid) in tokens.items():
        need = int(host[name])
        if name == "block_candidate":
            need = max(need, int(host["pointer"]))
        rounded = -(-max(need, 1) // SLOT_BUCKET) * SLOT_BUCKET
        grown[name] = min(valid.shape[1], max(kept.get(name, 0), rounded))
    return grown


def _cut_trunk(kernels: Kernels, params: Any, tokens: dict, state: Any, reset: jax.Array,
               kept: dict[str, int]) -> tuple[Any, Any]:
    """The trunk on each token type's first ``kept`` slots and on those slots' object slots, as ``distill_v3._batch``
    cuts a batch. The state comes back with its object slots at full width."""

    cut = {name: (features[:, :kept[name]], valid[:, :kept[name]]) for name, (features, valid) in tokens.items()}
    slots = state.object_slots
    short = {name: value[:, :kept.get(name, value.shape[1])] for name, value in slots.items()}
    view, new = kernels.trunk(params, cut, state._replace(object_slots=short), reset)
    return view, new._replace(object_slots=slots)


def rollout(strategy: Strategy, trunk: Trunk, module_params: Any, lanes: Lanes, use_teacher: jax.Array,
            kernels: Kernels, key: jax.Array, *, record: bool, storage: str = STORAGE,
            keep_logits: bool = False, shortcut: bool = True, slot_cut: bool = True,
            compact_lanes: bool = False) -> Trace:
    """``skill_dagger.rollout`` on v3: every lane runs its horizon, one decision per tick. The trained skill acts by its
    teacher where ``use_teacher`` (bool[B]) and by its module elsewhere; the skills beside it by their teachers. Entity 1
    starts where the trained skill's goals place it (:func:`placed_other`), and each request's entity handle is resolved
    to its token row every decision. With ``record`` the decisions inside the horizons are stored (``Decisions``); with
    ``keep_logits`` so are the module's logits there. If the trained skill's goals carry a ``mover`` (track), the
    environment moves entity 1 along that path, and it starts at the path's first point.

    ``shortcut`` (default) skips work no output reads: with every lane on its teacher the module is not run, and with
    nothing recorded or logged the trunk, the layer and the requests are not run either. The executed actions, and so
    the trace, are those of ``shortcut=False``, which runs every kernel (the reference the tests compare with).

    ``slot_cut`` (on by default since 2026-10-25, the owner's ruling; ``slot_cut=False`` is the exact path of earlier
    runs) runs the trunk on each token type's filled slots only (:func:`_kept_slots`, as ``distill_v3`` cuts its
    batches), so the token rows the trunk sees are fewer, and its float32 rounding may differ. ``compact_lanes`` (off by
    default) runs the layer, trunk, requests and module only on the lanes whose
    decisions still count (inside their horizon or the hold after it, ``blocks.HOLD_TICKS``); the other lanes act by
    their teachers, and the environment still steps every lane. A lane's actions after its hold, and so the trace's
    ``positions`` after it, differ from ``compact_lanes=False``; the decisions stored and the counts do not."""

    if compact_lanes and keep_logits:
        raise ValueError("compact_lanes does not keep the module's logits")
    env = blocks.environment(int(lanes.kind.shape[0]), strategy.config.maximum_turn_degrees)
    mover = getattr(lanes.goals[0], "mover", None)
    if mover is not None:
        env = env.with_mover(mover)
    batch = env.batch
    horizon = np.asarray(lanes.horizon)
    ticks = int(horizon.max())
    keys = jax.random.split(key, ticks + 2)
    spawn = jnp.broadcast_to(jnp.asarray(blocks.SPAWN_XZ, dtype=jnp.float32), (batch, 2))
    carry = env.reset(jax.random.split(keys[0], batch), spawn, lanes.spawn_yaw, placed_other(lanes, batch))
    neutral = jnp.broadcast_to(jnp.asarray(blocks.NEUTRAL, dtype=jnp.int32), (batch, blocks.HEAD_COUNT))
    carry, observation, _ = env.step(carry, neutral, jax.random.split(keys[1], batch))
    layer_step = v3._layer_program(v3._world_tick(carry.runtime.combat))
    state = v3._start_state(trunk.params, batch, trunk.config)
    dtype = np.dtype(storage)

    active = np.asarray(lanes.active)
    skills = strategy.skills
    limits = np.stack([np.asarray(goal.horizon_ticks) for goal in lanes.goals], axis=1)
    hit = np.full(active.shape, -1, dtype=np.int32)
    held = np.full(active.shape, -1, dtype=np.int32)
    streak = np.zeros(active.shape, dtype=np.int32)
    counted = np.zeros(active.shape, dtype=np.int32)
    counted_ok = np.zeros(active.shape, dtype=np.int32)
    counting = any(spec.holding for spec in skills)
    positions = np.zeros((ticks + 1, batch, 2), dtype=np.float32)
    yaws = np.zeros((ticks + 1, batch), dtype=np.float32)
    seconds = np.zeros((ticks, batch), dtype=np.float32)
    decisions = illegal = teacher_illegal = 0
    chunks: list[Decisions] = []
    logit_parts: list[dict] = []

    def record_instant(index: int, combat: Any) -> None:
        positions[index] = np.asarray(combat.position[:, AGENT_ENTITY][:, [0, 2]])
        yaws[index] = np.asarray(combat.yaw[:, AGENT_ENTITY])

    all_teacher = bool(np.asarray(use_teacher).all())
    teacher_only = shortcut and all_teacher and not record and not keep_logits  # no trunk output is read
    student_idle = shortcut and all_teacher and not keep_logits  # the module's action is never executed
    kept: dict[str, int] = {}
    record_instant(0, carry.runtime.combat)
    last = neutral
    for index in range(ticks):
        combat = carry.runtime.combat
        taught = kernels.teachers(combat, lanes)
        teacher_executed = merge(taught.labels, taught.beside, strategy.spec, lanes.active[:, 0])
        keep = index < horizon
        stored = None
        if teacher_only:
            view = requests = logits = None
            executed = teacher_executed
        elif compact_lanes:
            pick_host, real = _live_pick(index, horizon, batch)
            pick = jnp.asarray(pick_host)
            position = combat.position[:, AGENT_ENTITY]
            yaw = combat.yaw[:, AGENT_ENTITY]
            pose = Pose(position, yaw, combat.agent_head_yaw, combat.agent_head_pitch)
            requests = kernels.requests(lanes, pose, jnp.float32(index))
            _, last_action, _, _ = state_tick.host_inputs(observation, last, jnp.zeros((batch,), jnp.bool_))
            sub_reset = jnp.full((len(pick_host),), index == 0)
            tokens, after = layer_step(_take(state, pick), _take(observation, pick), _take(position, pick),
                                       _take(yaw, pick), _take(last_action, pick), sub_reset)
            requests = _take(requests, pick)
            if slot_cut:
                kept = _kept_slots(tokens, requests.cell_pointer, kept)
                view, sub_state = _cut_trunk(kernels, trunk.params, tokens, after, sub_reset, kept)
            else:
                view, sub_state = kernels.trunk(trunk.params, tokens, after, sub_reset)
            state = _put(state, pick, sub_state)
            requests = kernels.points(view, requests)
            sub_taught = _take(taught, pick)
            sub_active = _take(lanes.active[:, 0], pick)
            if student_idle:
                own = sub_taught.labels
            else:
                student, _ = kernels.student(module_params, _take(carry.action_mask, pick), view.encoded,
                                             view.masks, requests, sub_active)
                own = jnp.where(_take(use_teacher, pick)[:, None], sub_taught.labels, student)
            executed = teacher_executed.at[pick].set(merge(own, sub_taught.beside, strategy.spec, sub_active),
                                                     mode="drop")
            if record:
                stored = _store(view, requests, sub_taught, real & keep[np.minimum(pick_host, batch - 1)], dtype)
        else:
            position = combat.position[:, AGENT_ENTITY]
            yaw = combat.yaw[:, AGENT_ENTITY]
            pose = Pose(position, yaw, combat.agent_head_yaw, combat.agent_head_pitch)
            requests = kernels.requests(lanes, pose, jnp.float32(index))
            _, last_action, _, _ = state_tick.host_inputs(observation, last, jnp.zeros((batch,), jnp.bool_))
            reset = jnp.full((batch,), index == 0)
            tokens, after = layer_step(state, observation, position, yaw, last_action, reset)
            if slot_cut:
                kept = _kept_slots(tokens, requests.cell_pointer, kept)
                view, state = _cut_trunk(kernels, trunk.params, tokens, after, reset, kept)
            else:
                view, state = kernels.trunk(trunk.params, tokens, after, reset)
            # each entity handle to the token row holding that entity (-1: none)
            requests = kernels.points(view, requests)
            if student_idle:
                own, logits = taught.labels, None  # every lane is on its teacher: where() would pick these labels
            else:
                student, logits = kernels.student(module_params, carry.action_mask, view.encoded, view.masks, requests,
                                                  lanes.active[:, 0])
                own = jnp.where(use_teacher[:, None], taught.labels, student)
            executed = merge(own, taught.beside, strategy.spec, lanes.active[:, 0])
            if record:
                stored = _store(view, requests, taught, keep, dtype)

        mask = np.asarray(carry.action_mask)
        decisions += int(keep.sum())
        illegal += int(np.sum(keep & ~blocks.legal_rows(mask, np.asarray(executed))))
        teacher_illegal += int(np.sum(keep & ~blocks.legal_rows(mask, np.asarray(teacher_executed))))
        if record:
            chunks.append(stored)
        if keep_logits:
            logit_parts.append({name: np.asarray(value)[keep] for name, value in logits.items()})

        carry, observation, _ = env.step(carry, executed, jax.random.split(keys[index + 2], batch))
        last = executed
        done_count = index + 1
        combat = carry.runtime.combat
        record_instant(done_count, combat)
        seconds[index] = np.asarray(combat.last_motion_delta_seconds)
        ok = np.asarray(kernels.success(combat, lanes))
        live = (done_count <= limits) & active  # each skill's ticks inside its horizon, on the lanes that run it
        if counting:
            live = live & np.asarray(kernels.counts(combat, lanes))
        counted += live
        counted_ok += live & ok
        hit = np.where((hit < 0) & active & ok & (done_count <= limits), done_count, hit).astype(np.int32)
        for s in range(len(skills)):
            streak[:, s], held[:, s] = blocks.hold_update(streak[:, s], held[:, s], ok[:, s], done_count,
                                                          limits[:, s], active[:, s])

    samples = _join(chunks) if record else None
    module_logits = None
    if keep_logits:
        module_logits = {name: np.concatenate([part[name] for part in logit_parts]) for name in blocks.HEAD_NAMES}
    return Trace(hit, held, positions, yaws, seconds, decisions, illegal, teacher_illegal, samples, module_logits,
                 counted, counted_ok)


def _filled(data: Decisions) -> dict[str, int]:
    """Each token type's slots the stored decisions fill: the most valid rows any decision holds (at least 1), and at
    least the rows the requests point at (``entity_pointer`` into the entity rows, ``cell_pointer`` into the block
    candidates). The trunk masks empty slots, so a batch cut to these computes what the full one does, as
    ``distill_v3._filled``. That needs each decision's valid rows to come first in each type, as ``layer.step`` packs
    them; checked here.

    Raises:
        ValueError: If a type's valid rows do not come first in some decision.
    """

    count = len(data.labels)
    keep = {}
    for name, packed in data.tokens.items():
        decision = packed.index[:, 0].astype(np.int64)
        counts = np.bincount(decision, minlength=count)
        top = np.zeros(count, np.int64)
        np.maximum.at(top, decision, packed.index[:, 1].astype(np.int64) + 1)
        if not np.array_equal(top, counts):
            raise ValueError(f"{name}: the valid rows do not come first, so cutting the empty slots would drop rows")
        keep[name] = max(int(top.max()), 1)
    for name, pointer in (("entity", "entity_pointer"), ("block_candidate", "cell_pointer")):
        if name in keep:
            keep[name] = max(keep[name], int(data.requests[pointer].max()) + 1)
    return keep


def _unpack(packed: PackedRows, rows: np.ndarray, slots: int) -> np.ndarray:
    """The chosen decisions' valid rows as dense float32 [len(rows), slots, F], zeros where a row is invalid."""

    decision = packed.index[:, 0]
    lo = np.searchsorted(decision, rows, side="left")
    hi = np.searchsorted(decision, rows, side="right")
    counts = hi - lo
    take = np.arange(int(counts.sum())) + np.repeat(lo - (np.cumsum(counts) - counts), counts)
    dense = np.zeros((len(rows), slots, packed.shape[2]), np.float32)
    dense[np.repeat(np.arange(len(rows)), counts), packed.index[take, 1]] = packed.features[take]
    return dense


def _batch(data: Decisions, rows: np.ndarray, keep: dict[str, int]) -> dict:
    """The chosen decisions as a dense batch, each token type cut to its first ``keep`` slots."""

    return {
        "context": data.context[rows].astype(np.float32),
        "masks": {name: data.masks[name][rows, :keep[name]] for name in data.tokens},
        "tokens": {name: _unpack(packed, rows, keep[name]) for name, packed in data.tokens.items()},
        "requests": {name: value[rows] for name, value in data.requests.items()},
        "labels": data.labels[rows],
        "label_mask": data.label_mask[rows],
        "done": data.done[rows],
    }


def loss_fn(module_params: Any, batch: dict, *, spec: SkillSpec, config: GeneralistConfig, known: tuple[int, ...],
            done_weight: float) -> jax.Array:
    """``skill_dagger.loss_fn`` on stored decisions: the mean cross-entropy over the trained skill's scope heads, plus
    ``done_weight`` x the BCE of the module's own done logit. The trunk's outputs are the batch's data, so no trunk
    parameter is an input."""

    encoded = EncoderOutput(context=batch["context"], tokens=batch["tokens"])
    out = module_outputs(module_params, config, known, spec.verb, encoded, batch["masks"],
                         GeneralistRequests(**batch["requests"]))
    rows = jnp.arange(batch["labels"].shape[0])
    supervised = batch["label_mask"].astype(jnp.float32)
    total = 0.0
    for head, name in enumerate(blocks.HEAD_NAMES):
        log_probs = jax.nn.log_softmax(out.head_logits[name], axis=-1)
        total = total - jnp.sum(log_probs[rows, batch["labels"][:, head]] * supervised[:, head])
    cross_entropy = total / jnp.maximum(jnp.sum(supervised), 1.0)
    own = out.own.astype(jnp.float32)
    done_bce = optax.sigmoid_binary_cross_entropy(out.done, batch["done"].astype(jnp.float32))
    done_loss = jnp.sum(done_bce * own) / jnp.maximum(jnp.sum(own), 1.0)
    return cross_entropy + done_weight * done_loss


def make_train_step(strategy: Strategy, known: tuple[int, ...], optimizer: optax.GradientTransformation) -> Callable:
    """One Adam step on the module alone: ``(params, opt_state, batch) -> (params, opt_state, loss)``."""

    loss = functools.partial(loss_fn, spec=strategy.spec, config=strategy.config.module, known=known,
                             done_weight=strategy.config.done_weight)

    @jax.jit
    def step(params: Any, opt_state: Any, batch: dict) -> tuple[Any, Any, jax.Array]:
        value, grads = jax.value_and_grad(lambda p: loss(p, batch))(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, value

    return step


def _stored_bytes(data: Decisions) -> int:
    arrays = [data.context, data.labels, data.label_mask, data.done, *data.masks.values(), *data.requests.values()]
    arrays += [array for packed in data.tokens.values() for array in (packed.index, packed.features)]
    return int(sum(array.nbytes for array in arrays))


def start_module(strategy: Strategy, trunk: Trunk, init_key: jax.Array, init_params: Any = None) -> SkillModule:
    """The module training starts from: a fresh one, or one whose parameters start from ``init_params`` (a trained
    module of the same configuration, such as a prerequisite skill's: come_here from walk_to's). Those parameters must
    have the fresh module's structure and shapes."""

    module = init_module(init_key, strategy.spec, strategy.beside, trunk, strategy.config.module)
    if init_params is None:
        return module
    fresh, given = jax.tree_util.tree_structure(module.params), jax.tree_util.tree_structure(init_params)
    shapes = [jnp.shape(leaf) for leaf in jax.tree_util.tree_leaves(module.params)]
    if fresh != given or shapes != [jnp.shape(leaf) for leaf in jax.tree_util.tree_leaves(init_params)]:
        raise ValueError("init_params must have the module's structure and shapes")
    return module._replace(params=init_params)


def train(strategy: Strategy, trunk: Trunk, *, storage: str = STORAGE, init_params: Any = None,
          shortcut: bool = True, slot_cut: bool = True, compact_lanes: bool = False) -> TrainResult:
    """Iterate DAgger for ``strategy.spec``: collect with ``beta`` (the teacher's share of lanes; 1, then halving), add
    the decisions to the dataset, then ``steps`` Adam steps on the module alone, per iteration. ``init_params`` starts
    the module from a trained one (:func:`start_module`); None starts it fresh, as before. ``shortcut``, ``slot_cut``
    and ``compact_lanes`` are :func:`rollout`'s."""

    config = strategy.config
    key, init_key = jax.random.split(jax.random.key(config.seed))
    module = start_module(strategy, trunk, init_key, init_params)
    kernels = make_kernels(strategy, trunk, module.known)
    optimizer = optax.adam(config.learning_rate)
    params = module.params
    opt_state = optimizer.init(params)
    train_step = make_train_step(strategy, module.known, optimizer)
    rng = np.random.default_rng(config.seed)
    parts: list[Decisions] = []
    iterations: list[dict] = []
    for index in range(config.iterations):
        beta = 1.0 if index == 0 else 0.5 ** index
        started = time.perf_counter()
        key, key_kind, key_lanes, key_mix, key_rollout = jax.random.split(key, 5)
        kind = jax.random.randint(key_kind, (config.lanes,), 0, len(strategy.kinds))
        lanes = strategy.episodes(key_lanes, kind, False)
        use_teacher = jax.random.uniform(key_mix, (config.lanes,)) < beta
        trace = rollout(strategy, trunk, params, lanes, use_teacher, kernels, key_rollout, record=True,
                        storage=storage, shortcut=shortcut, slot_cut=slot_cut, compact_lanes=compact_lanes)
        parts.append(trace.samples)
        data = _join(parts)
        collected = time.perf_counter()

        size = len(data.labels)
        keep = _filled(data)
        losses, step_seconds = [], []
        for _ in range(config.steps):
            begun = time.perf_counter()
            rows = rng.integers(0, size, size=config.batch_size)
            params, opt_state, value = train_step(params, opt_state, _batch(data, rows, keep))
            losses.append(float(value))
            step_seconds.append(time.perf_counter() - begun)
        record = {
            "iteration": index,
            "beta": beta,
            "teacher_lanes": int(np.asarray(use_teacher).sum()),
            "lanes_by_kind": {name: int((np.asarray(kind) == code).sum()) for code, name in enumerate(strategy.kinds)},
            "new_samples": int(len(trace.samples.labels)),
            "decisions": trace.decisions,
            "executed_illegal": trace.illegal,
            "teacher_label_illegal": trace.teacher_illegal,
            "dataset_size": size,
            "stored_bytes_total": _stored_bytes(data),
            "token_slots": {"kept": keep, "total": sum(keep.values()),
                            "capacity": sum(packed.shape[1] for packed in data.tokens.values())},
            "steps": config.steps,
            "batch_size": config.batch_size,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "collect_seconds": round(collected - started, 2),
            "train_seconds": round(time.perf_counter() - collected, 2),
            "step_seconds": None if len(step_seconds) < 2 else {
                "first": round(step_seconds[0], 3), "p50": round(float(np.percentile(step_seconds[1:], 50)), 3)},
        }
        iterations.append(record)
        print(f"{strategy.spec.name} iteration {index} beta {beta:.3f}: dataset {size}, loss {record['loss_first']} -> "
              f"{record['loss_last']}", file=sys.stderr, flush=True)
    return TrainResult(module=module._replace(params=params), dataset=data, iterations=iterations)


def evaluate(strategy: Strategy, trunk: Trunk, module_params: Any, *, seed: int, lanes: int, held_out: bool,
             policy: str = "module", kernels: Kernels | None = None, shortcut: bool = True, slot_cut: bool = True,
             compact_lanes: bool = False) -> dict:
    """Run the trained skill by its module (``policy="module"``) or its teacher (``"teacher"``) on fresh v3 lanes, the
    kinds taken round-robin (``skill_dagger.evaluate``'s report). The same ``seed`` gives the same lanes. ``kernels``
    (:func:`make_kernels` for this strategy and trunk) may be shared by several calls, which then compile once; None
    makes a set. ``shortcut``, ``slot_cut`` and ``compact_lanes`` are :func:`rollout`'s."""

    if policy not in ("module", "teacher"):
        raise ValueError(f"policy must be 'module' or 'teacher', got {policy!r}")
    if kernels is None:
        known = tuple(sorted({spec.verb for spec in strategy.skills}))
        kernels = make_kernels(strategy, trunk, known)
    kind = jnp.asarray(np.arange(lanes) % len(strategy.kinds), dtype=jnp.int32)
    key_lanes, key_rollout = jax.random.split(jax.random.key(seed))
    episode = strategy.episodes(key_lanes, kind, held_out)
    use_teacher = jnp.full((lanes,), policy == "teacher")
    trace = rollout(strategy, trunk, module_params, episode, use_teacher, kernels, key_rollout, record=False,
                    shortcut=shortcut, slot_cut=slot_cut, compact_lanes=compact_lanes)
    return {"policy": policy, "held_out": held_out, "seed": seed, "lanes": lanes,
            **summarise(strategy, episode, trace)}


def _strategy(skill: str, config: SkillTrainConfig) -> Strategy:
    """The skill's own strategy (``neural_skills/<skill>/train.py``) at these sizes."""

    module = importlib.import_module(f"experimental.neural_agent.skill_agent.neural_skills.{skill}.train")
    return module.strategy(config)


def main(argv: Sequence[str] | None = None) -> int:
    defaults = SkillTrainConfig()
    parser = argparse.ArgumentParser(description="One skill's module on the frozen v3 trunk (V2): DAgger, then the "
                                                 "evaluation the bars read.")
    parser.add_argument("--skill", required=True, choices=VERBS)
    parser.add_argument("--trunk", type=Path, required=True, help="the frozen V1 trunk (.npz with its .json)")
    parser.add_argument("--smoke", action="store_true", help="the timed smoke: one iteration of 200 steps")
    parser.add_argument("--out", type=Path, required=True, help="the report (.json); must not exist")
    parser.add_argument("--module-out", type=Path, required=True,
                        help="the module (.npz, with its .json); must not exist")
    parser.add_argument("--iterations", type=int, default=defaults.iterations)
    parser.add_argument("--lanes", type=int, default=defaults.lanes)
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--eval-seed", type=int, default=77001)
    parser.add_argument("--eval-lanes", type=int, default=256)
    parser.add_argument("--no-slot-cut", action="store_true",
                        help="run the trunk on every token slot, as earlier runs did (the slot cut is the default)")
    parser.add_argument("--compact-lanes", action="store_true",
                        help="opt-in (not bit-exact): the trunk, layer and module run only on lanes still inside their "
                             "horizon or hold")
    parser.add_argument("--init-module", type=Path, default=None,
                        help="start from this trained module (.npz with its .json) on the same trunk, scope and sizes")
    args = parser.parse_args(argv)
    for path in (args.out, args.module_out):
        if path.exists():
            parser.error(f"{path} exists: a run never overwrites another")
    if args.module_out.with_suffix(".json") == args.out:
        parser.error(f"{args.out} is the module's .json sidecar: name --out and --module-out apart")

    config = SkillTrainConfig(iterations=args.iterations, lanes=args.lanes, steps=args.steps, seed=args.seed)
    if args.smoke:
        config = dataclasses.replace(config, iterations=1, steps=200)
    strategy = _strategy(args.skill, config)
    trunk = load_trunk(args.trunk)
    source = None if args.init_module is None else load_module(args.init_module, trunk)  # checks the trunk
    if source is not None and (source.scope != strategy.spec.scope or source.config != config.module):
        parser.error(f"{args.init_module} is a {source.scope} module of another size; it cannot start {args.skill}")
    record = blocks.provenance()
    started = time.perf_counter()
    slot_cut = not args.no_slot_cut
    result = train(strategy, trunk, init_params=None if source is None else source.params, slot_cut=slot_cut,
                   compact_lanes=args.compact_lanes)
    args.module_out.parent.mkdir(parents=True, exist_ok=True)
    save_module(args.module_out, result.module)
    known = tuple(sorted({spec.verb for spec in strategy.skills}))
    kernels = make_kernels(strategy, trunk, known)  # the four evaluations share one set of compiled kernels
    evaluation = {
        region: {policy: evaluate(strategy, trunk, result.module.params, seed=args.eval_seed, lanes=args.eval_lanes,
                                  held_out=held_out, policy=policy, kernels=kernels, slot_cut=slot_cut,
                                  compact_lanes=args.compact_lanes)
                 for policy in ("module", "teacher")}
        for region, held_out in (("seen", False), ("held_out", True))
    }
    record["load_average"]["end"] = list(os.getloadavg())
    speed = {"slot_cut": slot_cut, "compact_lanes": args.compact_lanes}
    report = {
        "schema": "skill_module_v1",
        "skill": strategy.spec.name,
        "scope": strategy.spec.scope,
        "beside": [spec.name for spec in strategy.beside],
        "kinds": list(strategy.kinds),
        "config": dataclasses.asdict(config),
        "storage": STORAGE,
        "smoke": args.smoke,
        "trunk_sha256": trunk.sha256,
        "module_sha256": params_sha256(result.module.params),
        "init_module": None if source is None else {"path": str(args.init_module), "skill": source.skill,
                                                    "params_sha256": params_sha256(source.params)},
        "known_verbs": list(result.module.known),
        "training": {"iterations": result.iterations, "dataset_size": len(result.dataset.labels),
                     "stored_bytes": _stored_bytes(result.dataset)},
        "evaluation": evaluation,
        "seconds": round(time.perf_counter() - started, 1),
        "provenance": record,
    }
    if slot_cut or args.compact_lanes:  # not the exact path of earlier runs: the report says which path made it
        report["speed"] = speed
    text = json.dumps(report, indent=1, default=blocks.json_default)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n")
    sys.stdout.write(text + "\n")
    return 0


__all__ = [
    "Decisions",
    "Kernels",
    "Trace",
    "TrainResult",
    "evaluate",
    "loss_fn",
    "make_kernels",
    "make_train_step",
    "main",
    "rollout",
    "train",
]


if __name__ == "__main__":
    raise SystemExit(main())
