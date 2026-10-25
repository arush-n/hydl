"""The people trunk's trainer (``docs/TRAINING_PLAN.md`` 8.3, stage PT1): a fresh v3 trunk and generalist, trained through time
on stored sequences as ``distill_v3`` does (V1), but on lanes drawn from several skills' own strategies (V2's
``neural_skills/<skill>/train.py``): walk_to, look_at, come_here and track by default.

Why: V1's trunk was trained with entity 1 inert at ``blocks.OTHER_XYZ``, so a person's place is not encoded (the P2 probe,
``bench/entity_probe.py``). Here entity 1 is placed where come_here places it in walk_to's and look_at's lanes
(``--fixed-other`` keeps V1's spot), come_here and track put it where they put it, and track moves it along its path.

Each collection splits its lanes equally across the skills (the remainder to the first), and each skill's share is one
rollout with that skill's strategy (its lane kinds, placement and mover, round-robin over the kinds). Per decision the
layer steps, the trunk steps with its carried state, every active skill's request is built from the agent's pose and
its entity handle is resolved to its token row (``state_tick.point``), and the generalist's choice is the executed factors
on the active skills' scope heads, neutral elsewhere. The labels are every active skill's teacher on its own scope heads
(all supervised), and each request carries its own done label. The method is ``distill_v3``'s: its sequences, burn-in,
``loss_v3``, the slot cut, Adam, and DAgger with beta 1 then halving; those functions are imported, not copied.

Evaluation: A2's report (``distill_v3.evaluate``, V1's layout), and for come_here and track the generalist on those
skills' lanes, seen and held out, with the teacher beside it (``skill_dagger_v3.evaluate`` with the generalist as the
module: the generalist's config is the module config, and every verb is known to it).

CLI::

    PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym JAX_PLATFORMS=cpu nice -n 10 \\
        .venv/bin/python -m experimental.neural_agent.skill_agent.training.trunk_v3 [--skills walk_to,look_at,come_here,track] \\
        [--fixed-other] [--control] [--smoke] --out REPORT.json [--params-out PARAMS.npz] [--trunk-out TRUNK.npz]
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

from experimental.neural_agent.skill_agent.bench import agent_models
from experimental.neural_agent.skill_agent.contract import tokens_v3
from experimental.neural_agent.skill_agent.model.encoder import EncoderConfig, parameter_count, token_layout
from experimental.neural_agent.skill_agent.model.generalist import GeneralistConfig, GeneralistRequests, apply_generalist
from experimental.neural_agent.skill_agent.model.skill_module import MODULE_CONFIG, Trunk, make_trunk, save_trunk, scope_heads
from experimental.neural_agent.skill_agent.neural_skills.come_here.task import ComeHereTasks
from experimental.neural_agent.skill_agent.neural_skills.spec import VERBS, Pose
from experimental.neural_agent.skill_agent.runtime import state_tick
from experimental.neural_agent.skill_agent.training import blocks, distill, sequences
from experimental.neural_agent.skill_agent.training import distill_v3 as v3
from experimental.neural_agent.skill_agent.training import skill_dagger_v3 as d3
from experimental.neural_agent.skill_agent.training import skill_dagger
from experimental.neural_agent.skill_agent.training.skill_dagger import Lanes, SkillTrainConfig, Strategy, build_requests
from hytalegym.jax.combat.types import AGENT_ENTITY, TARGET_ENTITY

SKILLS: tuple[str, ...] = ("walk_to", "look_at", "come_here", "track")
"""The skills the default mixture draws lanes from (their V2 strategies)."""
EVAL_SKILLS: tuple[str, ...] = ("come_here", "track")
"""The skills evaluated per skill (the generalist on their lanes, seen and held out). walk_to and look_at are A2's."""


@dataclass(frozen=True)
class TrunkV3Config:
    """What sizes a people-trunk run. The defaults are V1's, with the short pass's steps (6 x 500, the schedule the owner
    approved on 2026-10-24) and V1's generalist sized for every verb (``MODULE_CONFIG``, 160 verb ids).

    ``horizon_cap`` (tests only) caps every lane's horizon, so a check runs in seconds; None runs the lanes' own horizons.
    """

    skills: tuple[str, ...] = SKILLS
    fixed_other: bool = False  # walk_to's and look_at's lanes keep entity 1 at blocks.OTHER_XYZ (V1's spot), as an ablation
    iterations: int = 6
    lanes: int = 128
    steps: int = 500
    seed: int = 0
    learning_rate: float = 1e-3
    done_weight: float = 0.5
    maximum_turn_degrees: float = 45.0
    batch_sequences: int = 12  # with L = 90, about A2's 1,024 decisions a step
    sequence_length: int = 90
    burn_in: int = 30
    storage: str = "float16"
    encoder: EncoderConfig = field(default_factory=lambda: agent_models.CONFIG)
    generalist: GeneralistConfig = field(default_factory=lambda: MODULE_CONFIG)
    horizon_cap: int | None = None

    def __post_init__(self) -> None:
        if not self.skills or len(set(self.skills)) != len(self.skills):
            raise ValueError(f"skills must be non-empty and distinct, got {self.skills!r}")
        unknown = [skill for skill in self.skills if skill not in SKILLS]
        if unknown:
            raise ValueError(f"skills {unknown} are not in {SKILLS}")
        if self.lanes < len(self.skills):
            raise ValueError(f"lanes ({self.lanes}) must give every skill a lane ({len(self.skills)} skills)")
        if not 0 <= self.burn_in < self.sequence_length:
            raise ValueError(f"burn_in must be in [0, sequence_length), got {self.burn_in} with {self.sequence_length}")
        if self.storage not in ("float16", "float32"):
            raise ValueError(f"storage must be float16 or float32, got {self.storage!r}")
        if self.horizon_cap is not None and self.horizon_cap < 1:
            raise ValueError(f"horizon_cap must be at least 1, got {self.horizon_cap}")


class Kernels(NamedTuple):
    """One skill strategy's jitted requests (``skill_dagger.build_requests``), teachers (:func:`teach_all`), and the
    strategy's own success and counted-tick rules (``skill_dagger_v3.make_kernels``), per slot."""

    requests: Callable[..., GeneralistRequests]
    teachers: Callable[..., Any]
    success: Callable[..., jax.Array]  # (combat, lanes) -> bool[B, S]
    counts: Callable[..., jax.Array]  # (combat, lanes) -> bool[B, S]: the ticks a holding skill's share counts


class Taught(NamedTuple):
    labels: jax.Array  # int32[B, 12]: every active skill's teacher on its scope heads, neutral elsewhere
    mask: jax.Array  # bool[B, 12]: the active skills' scope heads (the supervised heads, the student's)
    done: jax.Array  # bool[B, REQUEST_SLOTS]: each slot's teacher done where its skill is active (holding: never)


class Rollout(NamedTuple):
    """One skill's rollout on the trunk being trained. ``recording`` and ``decisions`` are what ``distill_v3._dataset``
    reads (``decisions``: the resolved requests, labels, label mask, done and ``keep``, each stacked [T, B, ...]).
    ``counts`` are the decisions inside the horizons and the executed and teacher factors outside the mask. ``target`` is
    entity 1's place before each tick ([T+1, B, 3]). ``logits`` (when kept) are the generalist's head logits per decision.
    ``hit`` is the trained skill's first success decision count inside its horizon (-1 if none); ``counted`` and
    ``counted_ok`` are a holding skill's counted ticks inside its horizon and those where it succeeded (int32[B])."""

    recording: sequences.Recording
    decisions: dict
    counts: dict
    target: np.ndarray
    tick_seconds: float
    logits: dict | None
    hit: np.ndarray
    counted: np.ndarray
    counted_ok: np.ndarray


def make_kernels(strategy: Strategy) -> Kernels:
    skills = strategy.skills
    return Kernels(
        requests=jax.jit(functools.partial(build_requests, skills)),
        teachers=jax.jit(functools.partial(teach_all, skills)),
        success=jax.jit(lambda combat, lanes: jnp.stack(
            [spec.success(combat, goals) for spec, goals in zip(skills, lanes.goals)], axis=1)),
        counts=jax.jit(lambda combat, lanes: jnp.stack(
            [spec.counts(combat, goals) if spec.holding and spec.counts is not None
             else jnp.ones((combat.position.shape[0],), dtype=jnp.bool_)
             for spec, goals in zip(skills, lanes.goals)], axis=1)),
    )


def teach_all(skills: tuple, combat: Any, lanes: Lanes) -> Taught:
    """Every active skill's teacher on its own scope heads, as one set of labels (the scopes of the skills in a lane are
    disjoint: walk_to, come_here on locomotion; look_at, track on gaze). Each slot's done is its teacher's, where active; a
    holding skill has no done (``skill_dagger.teach``)."""

    batch = lanes.kind.shape[0]
    labels = jnp.broadcast_to(jnp.asarray(blocks.NEUTRAL, dtype=jnp.int32), (batch, blocks.HEAD_COUNT))
    mask = jnp.zeros((batch, blocks.HEAD_COUNT), dtype=jnp.bool_)
    done = jnp.zeros((batch, blocks.REQUEST_SLOTS), dtype=jnp.bool_)
    for slot, spec in enumerate(skills):
        output = spec.teacher(combat, lanes.goals[slot])
        on = lanes.active[:, slot, None] & jnp.asarray(scope_heads(spec.scope))[None, :]
        labels = jnp.where(on, output.factors, labels)
        mask = mask | on
        finished = output.done & lanes.active[:, slot]
        if spec.holding:
            finished = jnp.zeros_like(finished)
        done = done.at[:, slot].set(finished)
    return Taught(labels=labels.astype(jnp.int32), mask=mask, done=done)


def _student_on(head_logits: dict, action_mask: jax.Array, on: jax.Array) -> jax.Array:
    """``distill_v3._student``'s rule on any heads: the masked argmax on each head where ``on`` (bool[B, 12]), neutral
    elsewhere. (``distill_v3._student`` takes the walk and look scopes; this takes the active scope heads of any skills.)"""

    columns = []
    for head, name in enumerate(blocks.HEAD_NAMES):
        start, size = blocks.HEAD_STARTS[head], blocks.HEAD_SIZES[head]
        legal = action_mask[:, start:start + size]
        choice = jnp.argmax(jnp.where(legal, head_logits[name], -jnp.inf), axis=-1).astype(jnp.int32)
        columns.append(jnp.where(on[:, head], choice, int(blocks.NEUTRAL[head])))
    return jnp.stack(columns, axis=1)


@functools.lru_cache(maxsize=None)
def _decision_program(encoder: EncoderConfig, generalist: GeneralistConfig) -> Callable[..., Any]:
    """The trunk's step on the layer's tokens, each request's entity handle resolved to its token row, the generalist's
    heads, and the student's choice on the active scope heads, jitted. Returns the next state, the head logits, the
    student's factors and the resolved requests."""

    def run(trunk_params, head_params, tokens, state, reset, requests, action_mask, on):
        view, next_state = state_tick.trunk_step(Trunk(encoder, trunk_params, ""), tokens, state, reset)
        resolved = state_tick.point(view, requests, requests.entity_pointer)
        out = apply_generalist(head_params, view.encoded, view.masks, resolved, generalist)
        return next_state, out.head_logits, _student_on(out.head_logits, action_mask, on), resolved

    return jax.jit(run)


def horizon_of(lanes: Lanes, config: TrunkV3Config) -> np.ndarray:
    """Each lane's decisions: its own horizon, capped at ``config.horizon_cap`` when that is set."""

    horizon = np.asarray(lanes.horizon, dtype=np.int32)
    return horizon if config.horizon_cap is None else np.minimum(horizon, config.horizon_cap)


def rollout(strategy: Strategy, params: dict, lanes: Lanes, use_teacher: jax.Array, other: jax.Array,
            config: TrunkV3Config, kernels: Kernels, key: jax.Array, *, keep_logits: bool = False) -> Rollout:
    """``distill_v3._rollout`` for one skill's lanes: every lane runs its horizon, one decision per tick. Entity 1 starts at
    ``other`` (float32[B, 3]; a mover moves it along its path), the environment is the mover's when the goals carry one,
    and the executed factors are the teachers' where ``use_teacher`` (bool[B]) and the student's elsewhere. Each stored
    request has its entity handle resolved to the row holding that entity at that decision."""

    batch = int(lanes.kind.shape[0])
    env = blocks.environment(batch, config.maximum_turn_degrees)
    mover = getattr(lanes.goals[0], "mover", None)
    if mover is not None:
        env = env.with_mover(mover)
    horizon = horizon_of(lanes, config)
    ticks = int(horizon.max())
    keys = jax.random.split(key, ticks + 2)
    spawn = jnp.broadcast_to(jnp.asarray(blocks.SPAWN_XZ, dtype=jnp.float32), (batch, 2))
    carry = env.reset(jax.random.split(keys[0], batch), spawn, lanes.spawn_yaw, other)
    neutral = jnp.broadcast_to(jnp.asarray(blocks.NEUTRAL, dtype=jnp.int32), (batch, blocks.HEAD_COUNT))
    carry, observation, _ = env.step(carry, neutral, jax.random.split(keys[1], batch))
    tick_seconds = v3._world_tick(carry.runtime.combat)
    layer_step = v3._layer_program(tick_seconds)
    decide_step = _decision_program(config.encoder, config.generalist)
    state = v3._start_state(params["trunk"], batch, config.encoder)
    recording = sequences.Recording(config.encoder, batch)
    carry_at = set(range(0, ticks, config.sequence_length - config.burn_in))
    dtype = np.dtype(config.storage)

    target = np.zeros((ticks + 1, batch, 3), dtype=np.float32)
    active = np.asarray(lanes.active[:, 0])
    limit = np.asarray(lanes.goals[0].horizon_ticks)
    hit = np.full(batch, -1, dtype=np.int32)
    counted = np.zeros(batch, dtype=np.int32)
    counted_ok = np.zeros(batch, dtype=np.int32)
    decisions = illegal = teacher_illegal = 0
    per_decision: list[dict] = []
    per_logits: list[dict] = []
    last = neutral
    for index in range(ticks):
        combat = carry.runtime.combat
        target[index] = np.asarray(combat.position[:, TARGET_ENTITY])
        position = combat.position[:, AGENT_ENTITY]
        yaw = combat.yaw[:, AGENT_ENTITY]
        pose = Pose(position, yaw, combat.agent_head_yaw, combat.agent_head_pitch)
        requests = kernels.requests(lanes, pose, jnp.float32(index))
        taught = kernels.teachers(combat, lanes)
        _, last_action, _, _ = state_tick.host_inputs(observation, last, jnp.zeros((batch,), jnp.bool_))
        reset = jnp.full((batch,), index == 0)
        before = state
        tokens, after = layer_step(before, observation, position, yaw, last_action, reset)
        sequences.record(recording, before, tokens, after, reset, dtype=dtype, carry_at=carry_at)
        state, head_logits, student, resolved = decide_step(params["trunk"], params["generalist"], tokens, after, reset,
                                                            requests, carry.action_mask, taught.mask)
        executed = jnp.where(use_teacher[:, None], taught.labels, student)

        keep = index < horizon
        mask = np.asarray(carry.action_mask)
        decisions += int(keep.sum())
        illegal += int(np.sum(keep & ~blocks.legal_rows(mask, np.asarray(executed))))
        teacher_illegal += int(np.sum(keep & ~blocks.legal_rows(mask, np.asarray(taught.labels))))
        per_decision.append({"requests": v3._host(resolved), "labels": np.asarray(taught.labels),
                             "label_mask": np.asarray(taught.mask), "done": np.asarray(taught.done), "keep": keep})
        if keep_logits:
            per_logits.append({name: np.asarray(value) for name, value in head_logits.items()})

        carry, observation, _ = env.step(carry, executed, jax.random.split(keys[index + 2], batch))
        last = executed
        # the trained skill's own rule (skill_dagger_v3.rollout): first success inside its horizon; a holding skill also
        # counts its ticks inside the horizon, and those in view
        done_count = index + 1
        combat = carry.runtime.combat
        ok = np.asarray(kernels.success(combat, lanes))[:, 0]
        in_horizon = done_count <= limit
        hit = np.where((hit < 0) & active & ok & in_horizon, done_count, hit).astype(np.int32)
        if strategy.spec.holding:
            live = active & in_horizon & np.asarray(kernels.counts(combat, lanes))[:, 0]
            counted += live
            counted_ok += live & ok
    target[ticks] = np.asarray(carry.runtime.combat.position[:, TARGET_ENTITY])

    stored = {name: np.stack([step[name] for step in per_decision]) for name in ("labels", "label_mask", "done", "keep")}
    stored["requests"] = {name: np.stack([step["requests"][name] for step in per_decision])
                          for name in GeneralistRequests._fields}
    logits = None
    if keep_logits:
        logits = {name: np.stack([step[name] for step in per_logits]) for name in blocks.HEAD_NAMES}
    counts = {"decisions": decisions, "illegal": illegal, "teacher_illegal": teacher_illegal}
    return Rollout(recording, stored, counts, target, tick_seconds, logits, hit, counted, counted_ok)


def placement(strategy: Strategy, lanes: Lanes, key: jax.Array, config: TrunkV3Config) -> jax.Array:
    """float32[B, 3]: entity 1 at reset. A strategy whose goals place it (come_here's ``other_xyz``) or move it (track's
    ``mover``) is taken as it is (``skill_dagger_v3.placed_other``). Otherwise (walk_to, look_at) it stands at
    ``blocks.OTHER_XYZ`` under ``fixed_other``, and else where come_here's placement draws it: 3 to 15 blocks from the
    spawn at any bearing (``neural_skills/come_here/task.py``)."""

    batch = lanes.kind.shape[0]
    goals = lanes.goals[0]
    placed = getattr(goals, "other_xyz", None) is not None or getattr(goals, "mover", None) is not None
    if placed or config.fixed_other:
        return d3.placed_other(lanes, batch)
    return ComeHereTasks.sample(key, batch).other_xyz


def teacher_success(strategy: Strategy, run: Rollout, use_teacher: np.ndarray) -> dict:
    """The trained skill's success on the lanes it ran by its teacher, under the strategy's own rule (as
    ``skill_dagger_v3.summarise`` reads it): the share whose first success came inside the horizon, or, for a holding skill
    (track), its share of the counted ticks in view (``skill_dagger._holding_share``)."""

    teacher = np.asarray(use_teacher, dtype=bool)
    if strategy.spec.holding:
        share = skill_dagger._holding_share(run.counted, run.counted_ok, teacher)
        return {"rule": "share of counted ticks in view", "success": share["share"],
                "counted_lanes": share["counted_lanes"], "teacher_lanes": int(teacher.sum())}
    return {"rule": "first success inside the horizon", "success": skill_dagger._share(run.hit >= 1, teacher),
            "teacher_lanes": int(teacher.sum())}


def _shares(lanes: int, skills: int) -> list[int]:
    """Lanes per skill: equal, the remainder to the first skills."""

    base, extra = divmod(lanes, skills)
    return [base + (1 if index < extra else 0) for index in range(skills)]


def _strategies(config: TrunkV3Config) -> tuple[Strategy, ...]:
    sizes = SkillTrainConfig(lanes=config.lanes, seed=config.seed, maximum_turn_degrees=config.maximum_turn_degrees,
                             module=config.generalist)
    return tuple(d3._strategy(skill, sizes) for skill in config.skills)


def _collect(strategies: tuple[Strategy, ...], kernels: tuple[Kernels, ...], params: dict, config: TrunkV3Config,
             key: jax.Array, beta: float) -> tuple[Any, dict]:
    """One DAgger collection: one rollout per skill (its share of the lanes), joined into one dataset."""

    parts, per_skill = [], {}
    for strategy, kernel, lanes_n, key_skill in zip(strategies, kernels, _shares(config.lanes, len(strategies)),
                                                    jax.random.split(key, len(strategies))):
        key_lanes, key_place, key_mix, key_rollout = jax.random.split(key_skill, 4)
        kind = jnp.asarray(np.arange(lanes_n) % len(strategy.kinds), dtype=jnp.int32)  # round-robin over the kinds
        lanes = strategy.episodes(key_lanes, kind, False)
        use_teacher = jax.random.uniform(key_mix, (lanes_n,)) < beta
        other = placement(strategy, lanes, key_place, config)
        run = rollout(strategy, params, lanes, use_teacher, other, config, kernel, key_rollout)
        horizon = horizon_of(lanes, config)
        data = v3._dataset(run, horizon, config)
        parts.append(data)
        steps = np.linalg.norm(np.diff(run.target[:, :, [0, 2]], axis=0), axis=-1)  # entity 1's planar step per tick
        inside = np.arange(steps.shape[0])[:, None] < horizon[None, :]
        per_skill[strategy.spec.name] = {
            "lanes": lanes_n, "lanes_by_kind": {name: int((np.asarray(kind) == code).sum())
                                                for code, name in enumerate(strategy.kinds)},
            "teacher_lanes": int(np.asarray(use_teacher).sum()), "teacher_success": teacher_success(strategy, run, use_teacher),
            "decisions": run.counts["decisions"],
            "executed_illegal": run.counts["illegal"], "teacher_label_illegal": run.counts["teacher_illegal"],
            "sequences": int(len(data.loss)), "tick_seconds": run.tick_seconds,
            "entity_metres_moved_mean": float((steps * inside).sum(axis=0).mean()),
        }
    return v3._concat(parts), per_skill


def run_dagger(config: TrunkV3Config = TrunkV3Config()) -> v3.RunV3:
    """``distill_v3.run_dagger`` on the mixture: collect with ``beta`` (1, then halving), add the sequences to the
    aggregate, then ``config.steps`` Adam steps on the aggregate, per iteration."""

    strategies = _strategies(config)
    kernels = tuple(make_kernels(strategy) for strategy in strategies)
    key, init_key = jax.random.split(jax.random.key(config.seed))
    params = distill.init_params(init_key, distill.DistillConfig(encoder=config.encoder, generalist=config.generalist))
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)
    train_step = v3.make_train_step(config, optimizer)
    rng = np.random.default_rng(config.seed)

    data = None
    iterations: list[dict] = []
    for index in range(config.iterations):
        beta = 1.0 if index == 0 else 0.5 ** index
        started = time.perf_counter()
        key, collect_key = jax.random.split(key)
        part, per_skill = _collect(strategies, kernels, params, config, collect_key, beta)
        data = part if data is None else v3._concat([data, part])
        collected = time.perf_counter()

        count = len(data.loss)
        keep = v3._filled(data)
        losses, norms, step_seconds = [], [], []
        for step in range(config.steps):
            begun = time.perf_counter()
            rows = rng.integers(0, count, size=config.batch_sequences)
            params, opt_state, loss, norm = train_step(params, opt_state, v3._batch(data, rows, keep))
            loss = float(loss)
            step_seconds.append(time.perf_counter() - begun)
            if not np.isfinite(loss):
                raise FloatingPointError(f"iteration {index}: the loss is {loss} at step {step}")
            losses.append(loss)
            norms.append(float(norm))
            print(f"iteration {index} step {step}: loss {loss:.6f} grad {norms[-1]:.4g} {step_seconds[-1]:.3f} s",
                  file=sys.stderr, flush=True)
        points = sorted(set(range(0, config.steps, 50)) | {config.steps - 1}) if config.steps else []
        record = {
            "iteration": index,
            "beta": beta,
            "skills": per_skill,
            "sequences_total": count,
            "stored_bytes_total": v3._stored_bytes(data),
            "stored_bytes_new": v3._stored_bytes(part),
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
            "peak_rss_bytes": v3._peak_rss(),
            "parameters": {"trunk": parameter_count(params["trunk"]),
                           "generalist": parameter_count(params["generalist"])},
        }
        iterations.append(record)
        print(f"iteration {index} beta {beta:.3f}: sequences {count}, loss {losses[0] if losses else None} -> "
              f"{losses[-1] if losses else None}, collect {record['collect_seconds']} s, "
              f"train {record['train_seconds']} s", file=sys.stderr, flush=True)
    return v3.RunV3(params=params, iterations=iterations)


def a2_evaluation(params: dict, config: TrunkV3Config, *, seed: int, lanes: int, held_out: bool) -> dict:
    """A2's evaluation (``distill_v3.evaluate``) on these parameters, with V1's teacher options (run C')."""

    a2 = v3.DistillV3Config(lanes=lanes, encoder=config.encoder, generalist=config.generalist,
                            maximum_turn_degrees=config.maximum_turn_degrees)
    return v3.evaluate(params, a2, seed=seed, lanes=lanes, held_out=held_out)


def skill_evaluation(params: dict, config: TrunkV3Config, *, seed: int, lanes: int) -> dict:
    """The generalist as the policy for each evaluated skill the run trained on, seen and held out, with the skills beside
    it by their teachers, and the teacher alone for reference (``skill_dagger_v3.evaluate``). The generalist is the module:
    its config is the module config and every verb is known to it."""

    trunk = make_trunk(params["trunk"], config.encoder)
    known = tuple(range(len(VERBS)))
    report = {}
    for skill in EVAL_SKILLS:
        if skill not in config.skills:
            continue
        strategy = d3._strategy(skill, SkillTrainConfig(lanes=lanes, seed=config.seed,
                                                        maximum_turn_degrees=config.maximum_turn_degrees,
                                                        module=config.generalist))
        kernels = d3.make_kernels(strategy, trunk, known)
        report[skill] = {
            region: {policy: d3.evaluate(strategy, trunk, params["generalist"], seed=seed, lanes=lanes,
                                         held_out=held_out, policy=policy, kernels=kernels)
                     for policy in ("module", "teacher")}
            for region, held_out in (("seen", False), ("held_out", True))
        }
    return report


def export_trunk(params: dict, encoder: EncoderConfig, path: Path) -> None:
    """The trunk as ``trunk.npz`` (and its ``.json``), loadable by ``skill_dagger_v3 --trunk`` (what ``export_v1`` writes
    for V1's trunk, without V1's modules): its own token types, by feature name."""

    layout = token_layout(tokens_v3.TOKEN_SCHEMA_V3)
    trained = {name: layout[name] for name, _ in encoder.token_types}
    save_trunk(path, make_trunk(params["trunk"], encoder), trained)


def memoryless(config: TrunkV3Config) -> TrunkV3Config:
    """The control: the same run with the trunk's memory, background and object latents removed."""

    return v3.memoryless(config)


def main(argv: Sequence[str] | None = None) -> int:
    defaults = TrunkV3Config()
    parser = argparse.ArgumentParser(description="PT1: the people trunk, trained on the skills' lanes (distill_v3's method).")
    parser.add_argument("--skills", default=",".join(SKILLS), help="the skills whose strategies draw the lanes")
    parser.add_argument("--fixed-other", action="store_true",
                        help="walk_to's and look_at's lanes keep entity 1 at blocks.OTHER_XYZ (V1's spot)")
    parser.add_argument("--control", action="store_true", help="the memory-less control (distill_v3.memoryless)")
    parser.add_argument("--smoke", action="store_true", help="the timed smoke: one iteration of 300 steps")
    parser.add_argument("--iterations", type=int, default=defaults.iterations)
    parser.add_argument("--lanes", type=int, default=defaults.lanes)
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--eval-seed", type=int, default=77001)
    parser.add_argument("--eval-lanes", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True, help="the report (.json)")
    parser.add_argument("--params-out", type=Path, help="save the trained parameters here (.npz, blocks.save_params)")
    parser.add_argument("--trunk-out", type=Path, help="export the trunk (.npz with .json), as skill_dagger_v3's --trunk")
    args = parser.parse_args(argv)
    config = TrunkV3Config(skills=tuple(s.strip() for s in args.skills.split(",") if s.strip()),
                           fixed_other=args.fixed_other, iterations=args.iterations, lanes=args.lanes,
                           steps=args.steps, seed=args.seed)
    if args.control:
        config = memoryless(config)
    if args.smoke:
        config = dataclasses.replace(config, iterations=1, steps=300)
    record = blocks.provenance()
    started = time.perf_counter()
    result = run_dagger(config)
    evaluation = {region: a2_evaluation(result.params, config, seed=args.eval_seed, lanes=args.eval_lanes,
                                        held_out=held_out)
                  for region, held_out in (("seen", False), ("held_out", True))}
    per_skill = skill_evaluation(result.params, config, seed=args.eval_seed, lanes=args.eval_lanes)
    record["load_average"]["end"] = list(os.getloadavg())
    report = {
        "schema": "skill_agent_trunk_v3",
        "control": args.control,
        "smoke": args.smoke,
        "fixed_other": config.fixed_other,
        "config": dataclasses.asdict(config),
        "heads": list(blocks.HEAD_NAMES),
        "training": result.iterations,
        "evaluation": evaluation,
        "skill_evaluation": per_skill,
        "seconds": round(time.perf_counter() - started, 1),
        "provenance": record,
    }
    if args.params_out:
        blocks.save_params(args.params_out, result.params)
    if args.trunk_out:
        export_trunk(result.params, config.encoder, args.trunk_out)
    text = json.dumps(report, indent=1, default=blocks.json_default)
    args.out.write_text(text + "\n")
    sys.stdout.write(text + "\n")
    return 0


__all__ = ["EVAL_SKILLS", "SKILLS", "Kernels", "Rollout", "Taught", "TrunkV3Config", "a2_evaluation", "export_trunk",
           "memoryless", "placement", "rollout", "run_dagger", "skill_evaluation", "teach_all"]


if __name__ == "__main__":
    raise SystemExit(main())
