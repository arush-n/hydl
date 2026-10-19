"""DAgger for one skill: train that skill's own module on the frozen trunk, with other skills active beside it.

A skill's strategy (``neural_skills/<skill>/train.py``) decides everything specific to it and passes a
:class:`Strategy`: its spec (with its teacher's settings), the skills that run beside it, the kinds of lanes and how
they are drawn, and the sizes. These blocks do the rest the same way for every skill (the method of the A2 reference
trainer, ``training/distill.py``, README 2.16):

- each lane is one episode in the flat world: the trained skill is active, and any skill beside it may be;
- each decision, the trunk encodes the legal observation once; every active skill's request is built from the agent's
  own pose (its spec's ``request``); the trained skill's teacher labels that skill's scope heads and its done; each
  skill beside it acts on its own scope by its own teacher; the trained skill acts by its teacher (probability beta)
  or by its module;
- iteration i: beta = 1 at i = 0, then 0.5^i (Ross, Gordon and Bagnell 2011); the labels join one dataset, and Adam
  fits the module alone on all of it. The trunk never changes, so no other skill's module can change;
- the loss: cross-entropy on the trained skill's scope heads, plus ``done_weight`` x the binary cross-entropy of its
  own request's done logit.

The evaluation runs the module or the teacher for the trained skill (the skills beside it always by their teachers) on
fresh lanes, seen and held out, and reports per lane kind the first-hit and held success of the trained skill and of
each skill beside it, legality, and the trained skill's own metrics.
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

from experimental.neural_agent.skill_agent.contract.surface import SCOPE_NAMES
from experimental.neural_agent.skill_agent.model.encoder import apply_encoder
from experimental.neural_agent.skill_agent.model.generalist import ARG_FEATURES, GeneralistConfig, GeneralistRequests
from experimental.neural_agent.skill_agent.model.freezing import params_sha256
from experimental.neural_agent.skill_agent.model.skill_module import (
    MODULE_CONFIG,
    SkillModule,
    Trunk,
    init_module,
    load_trunk,
    module_outputs,
    save_module,
    scope_heads,
)
from experimental.neural_agent.skill_agent.neural_skills.spec import Pose, SkillSpec
from experimental.neural_agent.skill_agent.training.blocks import (
    HEAD_COUNT,
    HEAD_NAMES,
    HEAD_SIZES,
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
    hold_update,
    json_default,
    legal_rows,
    provenance,
    token_kernel,
)
from hytalegym.jax.combat.types import AGENT_ENTITY


@dataclass(frozen=True)
class SkillTrainConfig:
    """The sizes of one skill's DAgger run. The defaults are A2's (README 2.16)."""

    iterations: int = 6
    lanes: int = 128
    steps: int = 2000
    seed: int = 0
    batch_size: int = 1024
    learning_rate: float = 1e-3
    done_weight: float = 0.5
    maximum_turn_degrees: float = 45.0
    module: GeneralistConfig = field(default_factory=lambda: MODULE_CONFIG)


class Lanes(NamedTuple):
    """B lanes for the trained skill and the skills beside it, in the order (trained, *beside): ``goals[s]`` and
    ``active[:, s]``. ``horizon`` is the decisions a lane runs, the longest of its active skills' horizons."""

    kind: jax.Array  # int32[B]: an index into the strategy's kinds
    goals: tuple
    active: jax.Array  # bool[B, S]
    spawn_yaw: jax.Array  # float32[B], degrees
    horizon: jax.Array  # int32[B]


def make_lanes(kind: jax.Array, goals: tuple, active: jax.Array, spawn_yaw: jax.Array) -> Lanes:
    horizons = jnp.stack([goal.horizon_ticks for goal in goals], axis=1)
    horizon = jnp.max(jnp.where(active, horizons, 0), axis=1).astype(jnp.int32)
    return Lanes(kind.astype(jnp.int32), tuple(goals), active, spawn_yaw, horizon)


class Strategy(NamedTuple):
    """What a skill's ``train.py`` passes: its spec, the skills beside it, its lane kinds and how they are drawn
    (``episodes(key, kind, held_out) -> Lanes``, ``kind`` int32[B]), and its sizes."""

    spec: SkillSpec
    beside: tuple[SkillSpec, ...]
    kinds: tuple[str, ...]
    episodes: Callable[[jax.Array, jax.Array, bool], Lanes]
    config: SkillTrainConfig = SkillTrainConfig()

    @property
    def skills(self) -> tuple[SkillSpec, ...]:
        return (self.spec, *self.beside)


class Kernels(NamedTuple):
    tokens: Callable[..., Any]
    requests: Callable[..., GeneralistRequests]
    teachers: Callable[..., Any]
    student: Callable[..., jax.Array]
    success: Callable[..., jax.Array]


class Teach(NamedTuple):
    labels: jax.Array  # int32[B, 12]: the trained skill's teacher on its scope heads, neutral elsewhere
    mask: jax.Array  # bool[B, 12]: the supervised heads, its scope's where it is active
    done: jax.Array  # bool[B]: its teacher's done, where active
    beside: jax.Array  # int32[B, 12]: the skills beside it, each on its scope where active, neutral elsewhere


class Trace(NamedTuple):
    """One rollout. ``hit[:, s]`` and ``held[:, s]`` are the decision counts at which skill s first succeeded and
    completed a hold (``blocks.hold_update``), or -1."""

    hit: np.ndarray  # int32[B, S]
    held: np.ndarray  # int32[B, S]
    positions: np.ndarray  # float32[T+1, B, 2]
    yaws: np.ndarray  # float32[T+1, B]
    seconds: np.ndarray  # float32[T, B]
    decisions: int
    illegal: int
    teacher_illegal: int
    samples: dict | None


def build_requests(skills: tuple[SkillSpec, ...], lanes: Lanes, pose: Pose, ticks: jax.Array) -> GeneralistRequests:
    """Slot s holds skill s's request where it is active; the remaining slots are empty."""

    batch = pose.position.shape[0]
    rows = [spec.request(goals, pose, ticks) for spec, goals in zip(skills, lanes.goals)]
    rest = REQUEST_SLOTS - len(skills)
    active = lanes.active
    args = jnp.stack([jnp.where(active[:, s, None], row.args, 0.0) for s, row in enumerate(rows)], axis=1)
    present = jnp.stack([active[:, s, None] & row.present for s, row in enumerate(rows)], axis=1)
    entity = jnp.stack([jnp.where(active[:, s], row.entity_pointer, -1) for s, row in enumerate(rows)], axis=1)
    cell = jnp.stack([jnp.where(active[:, s], row.cell_pointer, -1) for s, row in enumerate(rows)], axis=1)
    verbs = jnp.asarray([spec.verb for spec in skills] + [0] * rest, dtype=jnp.int32)
    scopes = jnp.asarray([SCOPE_NAMES.index(spec.scope) for spec in skills] + [0] * rest, dtype=jnp.int32)
    none = jnp.full((batch, rest), -1, dtype=jnp.int32)
    return GeneralistRequests(
        verb=jnp.broadcast_to(verbs, (batch, REQUEST_SLOTS)),
        args=jnp.concatenate([args, jnp.zeros((batch, rest, ARG_FEATURES), dtype=jnp.float32)], axis=1),
        arg_present=jnp.concatenate([present, jnp.zeros((batch, rest, present.shape[2]), dtype=jnp.bool_)], axis=1),
        entity_pointer=jnp.concatenate([entity, none], axis=1),
        cell_pointer=jnp.concatenate([cell, none], axis=1),
        scope=jnp.broadcast_to(scopes, (batch, REQUEST_SLOTS)),
        valid=jnp.concatenate([active, jnp.zeros((batch, rest), dtype=jnp.bool_)], axis=1),
    )


def teach(skills: tuple[SkillSpec, ...], combat: Any, lanes: Lanes) -> Teach:
    """Every active skill's teacher: the trained skill's labels on its scope, the skills beside it on theirs."""

    neutral = jnp.asarray(NEUTRAL)
    outputs = [spec.teacher(combat, goals) for spec, goals in zip(skills, lanes.goals)]
    own_on = lanes.active[:, 0, None] & jnp.asarray(scope_heads(skills[0].scope))[None, :]
    labels = jnp.where(own_on, outputs[0].factors, neutral)
    beside = jnp.broadcast_to(neutral, labels.shape)
    for s in range(1, len(skills)):
        on = lanes.active[:, s, None] & jnp.asarray(scope_heads(skills[s].scope))[None, :]
        beside = jnp.where(on, outputs[s].factors, beside)
    return Teach(labels=labels.astype(jnp.int32), mask=own_on, done=outputs[0].done & lanes.active[:, 0],
                 beside=beside.astype(jnp.int32))


def student_factors(module_params: Any, trunk_params: Any, action_mask: jax.Array, tokens: dict,
                    requests: GeneralistRequests, active: jax.Array, *, trunk: Trunk, spec: SkillSpec,
                    config: GeneralistConfig, known: tuple[int, ...]) -> jax.Array:
    """The module's action: the masked argmax on each of its scope's heads where its skill is active, neutral
    elsewhere."""

    masks = {name: mask for name, (_, mask) in tokens.items()}
    encoded = apply_encoder(trunk_params, tokens, trunk.config)
    out = module_outputs(module_params, config, known, spec.verb, encoded, masks, requests)
    own = active[:, None] & jnp.asarray(scope_heads(spec.scope))[None, :]
    columns = []
    for head, name in enumerate(HEAD_NAMES):
        start, size = HEAD_STARTS[head], HEAD_SIZES[head]
        legal = action_mask[:, start:start + size]
        choice = jnp.argmax(jnp.where(legal, out.head_logits[name], -jnp.inf), axis=-1).astype(jnp.int32)
        columns.append(jnp.where(own[:, head], choice, int(NEUTRAL[head])))
    return jnp.stack(columns, axis=1)


def make_kernels(strategy: Strategy, trunk: Trunk, known: tuple[int, ...]) -> Kernels:
    skills = strategy.skills
    student = functools.partial(student_factors, trunk=trunk, spec=strategy.spec, config=strategy.config.module,
                                known=known)
    return Kernels(
        tokens=token_kernel(),
        requests=jax.jit(functools.partial(build_requests, skills)),
        teachers=jax.jit(functools.partial(teach, skills)),
        student=jax.jit(student),
        success=jax.jit(lambda combat, lanes: jnp.stack(
            [spec.success(combat, goals) for spec, goals in zip(skills, lanes.goals)], axis=1)),
    )


def merge(own: jax.Array, beside: jax.Array, spec: SkillSpec, active: jax.Array) -> jax.Array:
    """The executed factors: the trained skill's on its scope where active, the skills beside it elsewhere."""

    on = active[:, None] & jnp.asarray(scope_heads(spec.scope))[None, :]
    return jnp.where(on, own, beside).astype(jnp.int32)


def rollout(strategy: Strategy, trunk: Trunk, module_params: Any, lanes: Lanes, use_teacher: jax.Array,
            kernels: Kernels, key: jax.Array, *, record: bool) -> Trace:
    """Run every lane for its horizon: one neutral tick, then one decision per tick. The trained skill acts by its
    teacher where ``use_teacher`` (bool[B]) and by its module elsewhere."""

    env = environment(int(lanes.kind.shape[0]), strategy.config.maximum_turn_degrees)
    batch = env.batch
    skills = strategy.skills
    horizon = np.asarray(lanes.horizon)
    ticks = int(horizon.max())
    keys = jax.random.split(key, ticks + 2)
    spawn = jnp.broadcast_to(jnp.asarray(SPAWN_XZ, dtype=jnp.float32), (batch, 2))
    other = jnp.broadcast_to(jnp.asarray(OTHER_XYZ, dtype=jnp.float32), (batch, 3))
    carry = env.reset(jax.random.split(keys[0], batch), spawn, lanes.spawn_yaw, other)
    neutral = jnp.broadcast_to(jnp.asarray(NEUTRAL, dtype=jnp.int32), (batch, HEAD_COUNT))
    carry, observation, _ = env.step(carry, neutral, jax.random.split(keys[1], batch))

    active = np.asarray(lanes.active)
    limits = np.stack([np.asarray(goal.horizon_ticks) for goal in lanes.goals], axis=1)
    hit = np.full(active.shape, -1, dtype=np.int32)
    held = np.full(active.shape, -1, dtype=np.int32)
    streak = np.zeros(active.shape, dtype=np.int32)
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
        yaw = combat.yaw[:, AGENT_ENTITY]
        pose = Pose(combat.position[:, AGENT_ENTITY], yaw, combat.agent_head_yaw, combat.agent_head_pitch)
        requests = kernels.requests(lanes, pose, jnp.float32(index))
        tokens = kernels.tokens(observation, yaw)
        taught = kernels.teachers(combat, lanes)
        student = kernels.student(module_params, trunk.params, carry.action_mask, tokens, requests, lanes.active[:, 0])
        own = jnp.where(use_teacher[:, None], taught.labels, student)
        executed = merge(own, taught.beside, strategy.spec, lanes.active[:, 0])
        teacher_executed = merge(taught.labels, taught.beside, strategy.spec, lanes.active[:, 0])

        keep = index < horizon
        mask = np.asarray(carry.action_mask)
        decisions += int(keep.sum())
        illegal += int(np.sum(keep & ~legal_rows(mask, np.asarray(executed))))
        teacher_illegal += int(np.sum(keep & ~legal_rows(mask, np.asarray(teacher_executed))))
        if record:
            chunks.append(chunk(tokens, requests, taught.labels, taught.mask, taught.done, keep))

        carry, observation, _ = env.step(carry, executed, jax.random.split(keys[index + 2], batch))
        done_count = index + 1
        combat = carry.runtime.combat
        record_instant(done_count, combat)
        seconds[index] = np.asarray(combat.last_motion_delta_seconds)
        ok = np.asarray(kernels.success(combat, lanes))
        hit = np.where((hit < 0) & active & ok & (done_count <= limits), done_count, hit).astype(np.int32)
        for s in range(len(skills)):
            streak[:, s], held[:, s] = hold_update(streak[:, s], held[:, s], ok[:, s], done_count, limits[:, s],
                                                   active[:, s])

    samples = concat_chunks(chunks) if record else None
    return Trace(hit, held, positions, yaws, seconds, decisions, illegal, teacher_illegal, samples)


def loss_fn(module_params: Any, trunk_params: Any, batch: dict, *, trunk: Trunk, spec: SkillSpec,
            config: GeneralistConfig, known: tuple[int, ...], done_weight: float) -> jax.Array:
    """Mean cross-entropy over the supervised heads, plus ``done_weight`` x the BCE of the module's own done logit."""

    tokens = batch_tokens(batch)
    masks = {name: mask for name, (_, mask) in tokens.items()}
    encoded = jax.lax.stop_gradient(apply_encoder(trunk_params, tokens, trunk.config))
    requests = GeneralistRequests(
        verb=batch["verb"], args=batch["args"], arg_present=batch["arg_present"],
        entity_pointer=batch["entity_pointer"], cell_pointer=batch["cell_pointer"],
        scope=batch["scope"], valid=batch["valid"],
    )
    out = module_outputs(module_params, config, known, spec.verb, encoded, masks, requests)
    rows = jnp.arange(batch["labels"].shape[0])
    supervised = batch["label_mask"].astype(jnp.float32)
    total = 0.0
    for head, name in enumerate(HEAD_NAMES):
        log_probs = jax.nn.log_softmax(out.head_logits[name], axis=-1)
        total = total - jnp.sum(log_probs[rows, batch["labels"][:, head]] * supervised[:, head])
    cross_entropy = total / jnp.maximum(jnp.sum(supervised), 1.0)
    own = out.own.astype(jnp.float32)
    done_bce = optax.sigmoid_binary_cross_entropy(out.done, batch["done"].astype(jnp.float32))
    done_loss = jnp.sum(done_bce * own) / jnp.maximum(jnp.sum(own), 1.0)
    return cross_entropy + done_weight * done_loss


class TrainResult(NamedTuple):
    module: SkillModule
    dataset: dict
    iterations: list[dict]


def train(strategy: Strategy, trunk: Trunk) -> TrainResult:
    """Iterate DAgger for ``strategy.spec``: collect, aggregate, then ``steps`` Adam steps on the module alone."""

    config = strategy.config
    key, init_key = jax.random.split(jax.random.key(config.seed))
    module = init_module(init_key, strategy.spec, strategy.beside, trunk, config.module)
    kernels = make_kernels(strategy, trunk, module.known)
    optimizer = optax.adam(config.learning_rate)
    params = module.params
    opt_state = optimizer.init(params)
    loss = functools.partial(loss_fn, trunk=trunk, spec=strategy.spec, config=config.module, known=module.known,
                             done_weight=config.done_weight)

    @jax.jit
    def train_step(params, opt_state, batch):
        value, grads = jax.value_and_grad(lambda p: loss(p, trunk.params, batch))(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, value

    rng = np.random.default_rng(config.seed)
    chunks: list[dict[str, np.ndarray]] = []
    iterations: list[dict] = []
    dataset: dict[str, np.ndarray] = {}
    for index in range(config.iterations):
        beta = 1.0 if index == 0 else 0.5 ** index
        started = time.perf_counter()
        key, key_kind, key_lanes, key_mix, key_rollout = jax.random.split(key, 5)
        kind = jax.random.randint(key_kind, (config.lanes,), 0, len(strategy.kinds))
        lanes = strategy.episodes(key_lanes, kind, False)
        use_teacher = jax.random.uniform(key_mix, (config.lanes,)) < beta
        trace = rollout(strategy, trunk, params, lanes, use_teacher, kernels, key_rollout, record=True)
        chunks.append(trace.samples)
        dataset = concat_chunks(chunks)
        collected = time.perf_counter()

        size = dataset["labels"].shape[0]
        losses = []
        for _ in range(config.steps):
            rows = rng.integers(0, size, size=config.batch_size)
            batch = {name: jnp.asarray(value[rows]) for name, value in dataset.items()}
            params, opt_state, value = train_step(params, opt_state, batch)
            losses.append(value)
        losses = [float(value) for value in losses]
        record = {
            "iteration": index,
            "beta": beta,
            "teacher_lanes": int(np.asarray(use_teacher).sum()),
            "lanes_by_kind": {name: int((np.asarray(kind) == code).sum()) for code, name in enumerate(strategy.kinds)},
            "new_samples": int(trace.samples["labels"].shape[0]),
            "executed_illegal": trace.illegal,
            "teacher_label_illegal": trace.teacher_illegal,
            "dataset_size": int(size),
            "steps": config.steps,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "collect_seconds": round(collected - started, 2),
            "train_seconds": round(time.perf_counter() - collected, 2),
        }
        iterations.append(record)
        print(f"{strategy.spec.name} iteration {index} beta {beta:.3f}: dataset {size}, loss {record['loss_first']} -> "
              f"{record['loss_last']}", file=sys.stderr, flush=True)
    return TrainResult(module=module._replace(params=params), dataset=dataset, iterations=iterations)


def _share(flags: np.ndarray, lanes: np.ndarray) -> float | None:
    return float(flags[lanes].mean()) if lanes.any() else None


def summarise(strategy: Strategy, lanes: Lanes, trace: Trace) -> dict:
    """Per lane kind: the trained skill's and each skill beside it's first-hit and held success, and the share of lanes
    where every active skill held. Then legality, and the trained skill's own metrics."""

    kind = np.asarray(lanes.kind)
    active = np.asarray(lanes.active)
    hit, held = trace.hit >= 1, trace.held >= 1
    kinds = {}
    for code, name in enumerate(strategy.kinds):
        in_kind = kind == code
        skills = {}
        for s, spec in enumerate(strategy.skills):
            lanes_s = in_kind & active[:, s]
            if lanes_s.any():
                skills[spec.name] = {"lanes": int(lanes_s.sum()), "success": _share(hit[:, s], lanes_s),
                                     "held": _share(held[:, s], lanes_s)}
        all_held = np.all(held | ~active, axis=1)
        kinds[name] = {"lanes": int(in_kind.sum()), "skills": skills, "all_held": _share(all_held, in_kind)}
    summary = {"kinds": kinds, "decisions": trace.decisions, "illegal_decisions": trace.illegal,
               "teacher_label_illegal": trace.teacher_illegal, "hold_ticks": HOLD_TICKS}
    if strategy.spec.metrics is not None:
        summary["metrics"] = strategy.spec.metrics(lanes.goals[0], active[:, 0], trace, trace.hit[:, 0])
    return summary


def evaluate(strategy: Strategy, trunk: Trunk, module_params: Any, *, seed: int, lanes: int, held_out: bool,
             policy: str = "module") -> dict:
    """Run the trained skill by its module (``policy="module"``) or its teacher (``"teacher"``) on fresh lanes, the
    kinds taken round-robin. The same ``seed`` gives the same lanes, so the module and the teacher see identical
    episodes."""

    if policy not in ("module", "teacher"):
        raise ValueError(f"policy must be 'module' or 'teacher', got {policy!r}")
    known = tuple(sorted({spec.verb for spec in strategy.skills}))
    kernels = make_kernels(strategy, trunk, known)
    kind = jnp.asarray(np.arange(lanes) % len(strategy.kinds), dtype=jnp.int32)
    key_lanes, key_rollout = jax.random.split(jax.random.key(seed))
    episode = strategy.episodes(key_lanes, kind, held_out)
    use_teacher = jnp.full((lanes,), policy == "teacher")
    trace = rollout(strategy, trunk, module_params, episode, use_teacher, kernels, key_rollout, record=False)
    return {"policy": policy, "held_out": held_out, "seed": seed, "lanes": lanes, **summarise(strategy, episode, trace)}


def run_main(build: Callable[[SkillTrainConfig], Strategy], description: str,
             argv: Sequence[str] | None = None) -> int:
    """The command line every skill's ``train.py`` shares: build the skill's strategy at the given sizes, train its
    module on the trunk, evaluate the module and the teacher on seen and held-out lanes, and write ``module.npz`` (with
    its .json) and ``report.json`` into ``--out``, a folder that must not exist yet."""

    defaults = SkillTrainConfig()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--trunk", type=Path, required=True, help="the shared frozen trunk (.npz with its .json)")
    parser.add_argument("--out", type=Path, required=True, help="the run folder (private, never staged); must be new")
    parser.add_argument("--iterations", type=int, default=defaults.iterations)
    parser.add_argument("--lanes", type=int, default=defaults.lanes)
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--eval-seed", type=int, default=77001)
    parser.add_argument("--eval-lanes", type=int, default=256)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"{args.out} exists: a run never overwrites another")

    config = SkillTrainConfig(iterations=args.iterations, lanes=args.lanes, steps=args.steps, seed=args.seed)
    strategy = build(config)
    trunk = load_trunk(args.trunk)
    record = provenance()
    started = time.perf_counter()
    result = train(strategy, trunk)
    evaluation = {
        region: {policy: evaluate(strategy, trunk, result.module.params, seed=args.eval_seed, lanes=args.eval_lanes,
                                  held_out=held_out, policy=policy) for policy in ("module", "teacher")}
        for region, held_out in (("seen", False), ("held_out", True))
    }
    record["load_average"]["end"] = list(os.getloadavg())
    args.out.mkdir(parents=True)
    save_module(args.out / "module.npz", result.module)
    report = {
        "schema": "skill_module_v1",
        "skill": strategy.spec.name,
        "scope": strategy.spec.scope,
        "beside": [spec.name for spec in strategy.beside],
        "kinds": list(strategy.kinds),
        "config": dataclasses.asdict(config),
        "trunk_sha256": trunk.sha256,
        "module_sha256": params_sha256(result.module.params),
        "known_verbs": list(result.module.known),
        "training": {"iterations": result.iterations, "dataset_size": int(result.dataset["labels"].shape[0])},
        "evaluation": evaluation,
        "seconds": round(time.perf_counter() - started, 1),
        "provenance": record,
    }
    text = json.dumps(report, indent=1, default=json_default)
    (args.out / "report.json").write_text(text + "\n")
    sys.stdout.write(text + "\n")
    return 0


__all__ = [
    "Lanes",
    "SkillTrainConfig",
    "Strategy",
    "TrainResult",
    "build_requests",
    "evaluate",
    "loss_fn",
    "make_kernels",
    "make_lanes",
    "merge",
    "rollout",
    "run_main",
    "student_factors",
    "summarise",
    "teach",
    "train",
]
