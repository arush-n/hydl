"""B6 (2026-10-25): each skill's trained module run as a runtime body, checked against the trainer's evaluation of the
same module on the same lanes and seed.

For one skill (``<folder>/<skill>.npz`` on ``<folder>/trunk.npz``, its strategy from ``neural_skills/<skill>/train.py``)
the flat world is driven one decision at a time as the coordinator's body path does it:

- the state through ``state_tick.step_lanes`` (the agent's own pose channel, as ``distill_v3._layer_program``), and each
  agent's view through ``state_tick.encode_contexts`` with the skills beside it as a ``JointView`` of live requests;
- the trained skill's module as ``state_tick.ModuleBody`` (layout ``state_tick.v2_layout``: the skill in slot 0, the
  skills beside it after, as ``skill_dagger.build_requests`` orders them), called once per lane per decision;
- each request a runtime ``middle.ActionRequest`` built from the lane's goals (the request's position, entity handle and
  scalars), lowered by each skill's own ``lowering``; the handle is entity 1, the person, for come_here, track and follow;
- the skills beside it by their teachers (``skill_dagger.teach``), merged as ``skill_dagger.merge`` does.

It is ``skill_dagger_v3.rollout`` with the module's action from ``ModuleBody`` in place of ``student_factors``, line for
line otherwise: the same environment, keys, counts and hold rules. The references are the trainer's own rollouts on the
same lanes and seed: ``skill_dagger_v3.rollout`` with ``slot_cut`` (what ``evaluate`` runs, the default) and with
``slot_cut=False`` (the runtime's path: every token slot). With no ``max_ticks`` cap, ``evaluate``'s own report is also
checked against the rollout's summary.

Checked per region (seen, held out): per lane, the first-hit and held decisions and the counted ticks of the trained skill
and of the skill beside it, against each reference; the first tick the trajectories part; the decisions' legality; each
lowered request's args and presence against ``skill_dagger.build_requests`` (atol 1e-6, every active slot and tick); the
entity handle's resolution to a token row; and ``ModuleBatch`` on a sample of lanes every tick, its choices and done equal
to the per-agent bodies'.

CLI::

    python -m experimental.neural_agent.skill_agent.bench.bodies_check --bodies DIR --lanes 32 --seed 77001 \\
        --json OUT [--skills come_here follow ...] [--max-ticks N] [--batch-lanes 8]

``--max-ticks`` caps every horizon: a short test config, recorded in the report (``max_ticks``). Without it the run is the
trainers' lanes. Each run writes one JSON (it never overwrites one).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent import requests as middle
from experimental.neural_agent.skill_agent.model.freezing import params_sha256
from experimental.neural_agent.skill_agent.model.skill_module import scope_heads
from experimental.neural_agent.skill_agent.model.skill_module import SkillModule, Trunk, load_module, load_trunk
from experimental.neural_agent.skill_agent.neural_skills.spec import Pose
from experimental.neural_agent.skill_agent.runtime import state_tick
from experimental.neural_agent.skill_agent.runtime.combiner import AgentContext
from experimental.neural_agent.skill_agent.training import blocks, distill_v3, skill_dagger, skill_dagger_v3 as trainer
from experimental.neural_agent.skill_agent.training.skill_dagger import SkillTrainConfig
from hytalegym.jax.combat.types import AGENT_ENTITY, TARGET_ENTITY

SKILLS = ("come_here", "follow", "look_at", "track", "walk_to")
"""The five skills of the people trunk's folder (``neural_skills/``)."""
ATOL = 1e-6
"""The lowered args' tolerance against the trainer's build_requests (as ``test_sa_export_v1``)."""
BOX = ((-100.0, -64.0, -100.0), (100.0, 320.0, 100.0))
"""The world box a context carries (the flat world is 16 blocks wide around the spawn; ``test_sa_export_v1``)."""
BATCH_LANES = 8
"""Lanes ModuleBatch is checked on, every tick, spread across the batch."""


def lowering_of(verb: str):
    """A skill's own lowering (``neural_skills/<verb>/skill.py``), the function ``ModuleBody`` lowers its requests with."""
    return importlib.import_module(f"experimental.neural_agent.skill_agent.neural_skills.{verb}.skill").lowering


def runtime_request(verb: str, goals: Any, lane: int, *, request_id: int, expires: int) -> middle.ActionRequest:
    """The middle layer's request for one lane's goal, as a host sends it: the goal's numbers as the request's position,
    entity handle and scalars (the scalars in each lowering's order). The handle is the person, entity 1."""

    def at(name: str) -> Any:
        return np.asarray(getattr(goals, name))[lane]

    if verb == "walk_to":
        x, z = (float(value) for value in at("target_xz"))
        args = middle.RequestArgs(position=(x, 0.0, z), scalars=(float(at("radius")), float(at("gait"))))
    elif verb == "look_at":
        args = middle.RequestArgs(position=tuple(float(value) for value in at("target_xyz")),
                                  scalars=(float(at("tolerance_degrees")),))
    elif verb == "come_here":
        args = middle.RequestArgs(entity=TARGET_ENTITY, scalars=(float(at("radius")), float(at("gait"))))
    elif verb == "follow":
        args = middle.RequestArgs(entity=TARGET_ENTITY, scalars=(float(at("near")), float(at("far"))))
    elif verb == "track":
        args = middle.RequestArgs(entity=TARGET_ENTITY, scalars=(float(at("tolerance_degrees")),))
    else:
        raise ValueError(f"no runtime request for verb {verb!r}")
    return middle.ActionRequest(request_id=request_id, agent=lane, issued_tick=0, expires_tick=expires, verb=verb,
                                args=args, mode="parallel", urgency=0.5, confidence=0.5, priority=1.0)


def capped(lanes: Any, max_ticks: int | None) -> Any:
    """The lanes with every horizon at most ``max_ticks`` (a short test config); the lanes themselves are unchanged."""
    if max_ticks is None:
        return lanes
    cap = jnp.int32(max_ticks)
    goals = tuple(goal._replace(horizon_ticks=jnp.minimum(goal.horizon_ticks, cap)) for goal in lanes.goals)
    return lanes._replace(goals=goals, horizon=jnp.minimum(lanes.horizon, cap))


def region_episode(strategy: Any, lanes: int, seed: int, held_out: bool, max_ticks: int | None) -> tuple[Any, Any]:
    """The lanes and the rollout key ``skill_dagger_v3.evaluate`` draws for ``seed`` (the kinds round-robin)."""
    kind = jnp.asarray(np.arange(lanes) % len(strategy.kinds), dtype=jnp.int32)
    key_lanes, key_rollout = jax.random.split(jax.random.key(seed))
    return capped(strategy.episodes(key_lanes, kind, held_out), max_ticks), key_rollout


def _step_program(trunk: Trunk, tick_seconds: float):
    """``state_tick.step_lanes`` on the agent's own pose channel, jitted; the channel is attached inside the jit so the
    compiled tick does not retrace (``state_tick.own_pose_channel``)."""

    def run(state, observation, position, yaw, last_action, reset):
        feed = state_tick.own_pose_channel(observation, position, yaw)
        view, state, _ = state_tick.step_lanes(trunk, state, feed, last_action=last_action, reset=reset,
                                               tick_seconds=tick_seconds)
        return view, state

    return jax.jit(run)


def drive(strategy: Any, trunk: Trunk, module: SkillModule, lanes: Any, key: Any, *, kernels: Any,
          sample: Sequence[int]) -> tuple[trainer.Trace, dict]:
    """The body path on ``lanes``: ``skill_dagger_v3.rollout``'s loop, the module's action from ``ModuleBody``, with the
    checks listed in the module docstring. Returns the trace (the trainer's :class:`Trace` fields) and the audit."""

    names = tuple(spec.name for spec in strategy.skills)
    layout = state_tick.v2_layout(strategy.spec.name, names[1:])
    body = state_tick.ModuleBody(module, trunk, lowering_of(names[0]), layout=layout)
    batch = state_tick.ModuleBatch(module, trunk, lowering_of(names[0]), layout=layout)
    count = int(lanes.kind.shape[0])
    env = blocks.environment(count, strategy.config.maximum_turn_degrees)
    mover = getattr(lanes.goals[0], "mover", None)
    if mover is not None:
        env = env.with_mover(mover)
    horizon = np.asarray(lanes.horizon)
    ticks = int(horizon.max())
    keys = jax.random.split(key, ticks + 2)
    spawn = jnp.broadcast_to(jnp.asarray(blocks.SPAWN_XZ, dtype=jnp.float32), (count, 2))
    carry = env.reset(jax.random.split(keys[0], count), spawn, lanes.spawn_yaw, trainer.placed_other(lanes, count))
    neutral = jnp.broadcast_to(jnp.asarray(blocks.NEUTRAL, dtype=jnp.int32), (count, blocks.HEAD_COUNT))
    carry, observation, _ = env.step(carry, neutral, jax.random.split(keys[1], count))
    step = _step_program(trunk, distill_v3._world_tick(carry.runtime.combat))
    state = distill_v3._start_state(trunk.params, count, trunk.config)

    active = np.asarray(lanes.active)
    limits = np.stack([np.asarray(goal.horizon_ticks) for goal in lanes.goals], axis=1)
    goals = dict(zip(names, lanes.goals))
    own = {lane: runtime_request(names[0], goals[names[0]], lane, request_id=1, expires=max(1, int(horizon[lane])))
           for lane in range(count)}
    beside = {lane: {name: runtime_request(name, goals[name], lane, request_id=slot + 2,
                                           expires=max(1, int(horizon[lane])))
                     for slot, name in enumerate(names) if slot and active[lane, slot]}
              for lane in range(count)}
    counting = any(spec.holding for spec in strategy.skills)
    hit = np.full(active.shape, -1, np.int32)
    held = np.full(active.shape, -1, np.int32)
    streak = np.zeros(active.shape, np.int32)
    counted = np.zeros(active.shape, np.int32)
    counted_ok = np.zeros(active.shape, np.int32)
    positions = np.zeros((ticks + 1, count, 2), np.float32)
    yaws = np.zeros((ticks + 1, count), np.float32)
    seconds = np.zeros((ticks, count), np.float32)
    decisions = illegal = teacher_illegal = 0
    audit = {"lowering_checked": 0, "lowering_mismatches": 0, "lowering_max_abs_diff": 0.0,
             "entity_checked": 0, "entity_resolved": 0, "batch_checked": 0, "batch_mismatches": 0,
             "batch_lane_ticks": 0, "sample_lanes": [int(lane) for lane in sample],
             "module_teacher_disagree": 0, "decisions": 0}

    def record_instant(index: int, combat: Any) -> None:
        positions[index] = np.asarray(combat.position[:, AGENT_ENTITY][:, [0, 2]])
        yaws[index] = np.asarray(combat.yaw[:, AGENT_ENTITY])

    record_instant(0, carry.runtime.combat)
    last = neutral
    for index in range(ticks):
        combat = carry.runtime.combat
        taught = kernels.teachers(combat, lanes)
        teacher_executed = skill_dagger.merge(taught.labels, taught.beside, strategy.spec, lanes.active[:, 0])
        keep = index < horizon
        position = combat.position[:, AGENT_ENTITY]
        yaw = combat.yaw[:, AGENT_ENTITY]
        pose = Pose(position, yaw, combat.agent_head_yaw, combat.agent_head_pitch)
        reference = kernels.requests(lanes, pose, jnp.float32(index))
        _, last_action, _, _ = state_tick.host_inputs(observation, last, jnp.zeros((count,), jnp.bool_))
        view, state = step(state, observation, position, yaw, last_action, jnp.full((count,), index == 0))
        mask = np.asarray(carry.action_mask)

        for lane in range(count):  # each lowering against the trainer's request for the same slot
            pose_lane = Pose(*(leaf[lane:lane + 1] for leaf in pose))
            for name, request in [(names[0], own[lane]), *beside[lane].items()]:
                slot = names.index(name)
                assert bool(np.asarray(reference.valid)[lane, slot]), f"{name} is not valid for lane {lane}"
                args, present = lowering_of(name)(request, pose_lane, index)
                diff = float(np.max(np.abs(np.asarray(args) - np.asarray(reference.args[lane, slot]))))
                audit["lowering_checked"] += 1
                audit["lowering_max_abs_diff"] = max(audit["lowering_max_abs_diff"], diff)
                if diff > ATOL or not np.array_equal(np.asarray(present), np.asarray(reference.arg_present[lane, slot])):
                    audit["lowering_mismatches"] += 1
        handles = jnp.full((count, 1), TARGET_ENTITY, jnp.int32)
        resolved = np.asarray(state_tick.entity_rows(handles, view.entity_ids))[:, 0] >= 0
        for lane in range(count):
            for request in [own[lane], *beside[lane].values()]:
                if request.args.entity is not None:
                    audit["entity_checked"] += 1
                    audit["entity_resolved"] += int(resolved[lane])

        contexts = state_tick.encode_contexts(
            view, {lane: AgentContext(world_bounds=BOX) for lane in range(count)}, list(range(count)), poses=pose,
            ticks=index, action_masks=mask, live_requests=beside)
        steps, factors = {}, np.broadcast_to(np.asarray(blocks.NEUTRAL, np.int32), (count, blocks.HEAD_COUNT)).copy()
        for lane in range(count):  # the module's action: each lane's own body call
            steps[lane] = body(own[lane], index, contexts[lane].observation_tokens, None)
            for head, name in enumerate(blocks.HEAD_NAMES):
                if name in steps[lane].factors:
                    factors[lane, head] = steps[lane].factors[name]
        if sample:  # ModuleBatch on the sample lanes: the same choices and done as their bodies
            got = batch.run(view, [contexts[lane].observation_tokens for lane in sample],
                            [own[lane] for lane in sample])
            for lane, choice in zip(sample, got):
                audit["batch_lane_ticks"] += 1
                audit["batch_checked"] += 1
                if dict(choice.factors) != dict(steps[lane].factors) or choice.done != steps[lane].done:
                    audit["batch_mismatches"] += 1
        executed = skill_dagger.merge(jnp.asarray(factors), taught.beside, strategy.spec, lanes.active[:, 0])
        on_scope = np.asarray(scope_heads(strategy.spec.scope))
        differs = (factors != np.asarray(taught.labels)) & on_scope[None, :] & keep[:, None]
        audit["module_teacher_disagree"] += int(np.sum(np.any(differs, axis=1)))  # the module is in control, not the teacher
        audit["decisions"] += int(keep.sum())

        decisions += int(keep.sum())
        illegal += int(np.sum(keep & ~blocks.legal_rows(mask, np.asarray(executed))))
        teacher_illegal += int(np.sum(keep & ~blocks.legal_rows(mask, np.asarray(teacher_executed))))
        carry, observation, _ = env.step(carry, executed, jax.random.split(keys[index + 2], count))
        last = executed
        done_count = index + 1
        combat = carry.runtime.combat
        record_instant(done_count, combat)
        seconds[index] = np.asarray(combat.last_motion_delta_seconds)
        ok = np.asarray(kernels.success(combat, lanes))
        live = (done_count <= limits) & active
        if counting:
            live = live & np.asarray(kernels.counts(combat, lanes))
        counted += live
        counted_ok += live & ok
        hit = np.where((hit < 0) & active & ok & (done_count <= limits), done_count, hit).astype(np.int32)
        for s in range(len(names)):
            streak[:, s], held[:, s] = blocks.hold_update(streak[:, s], held[:, s], ok[:, s], done_count,
                                                          limits[:, s], active[:, s])
    trace = trainer.Trace(hit, held, positions, yaws, seconds, decisions, illegal, teacher_illegal, None, None,
                          counted, counted_ok)
    return trace, audit


def _agreement(body: trainer.Trace, ref: trainer.Trace) -> dict:
    """Per lane: equal in every field (first-hit, held, counted, counted_ok, for every skill), and where not, the first
    tick the two trajectories part (positions, atol 1e-6)."""

    lanes = body.hit.shape[0]
    same = np.ones(lanes, bool)
    for field in ("hit", "held", "counted", "counted_ok"):
        same &= (getattr(body, field) == getattr(ref, field)).all(axis=1)
    gap = np.abs(body.positions - ref.positions).max(axis=2)  # [T+1, B]
    parted = {}
    for lane in np.flatnonzero(~same):
        moved = np.flatnonzero(gap[:, lane] > ATOL)
        parted[int(lane)] = int(moved[0]) if moved.size else None
    return {"lanes": lanes, "equal_lanes": int(same.sum()), "unequal_lanes": [int(lane) for lane in np.flatnonzero(~same)],
            "first_divergence_tick": parted, "trained_first_hit_equal": int((body.hit[:, 0] == ref.hit[:, 0]).sum()),
            "trained_held_equal": int((body.held[:, 0] == ref.held[:, 0]).sum()),
            "beside_first_hit_equal": int((body.hit[:, 1:] == ref.hit[:, 1:]).all(axis=1).sum()),
            "decisions_equal": bool(body.decisions == ref.decisions), "illegal_equal": bool(body.illegal == ref.illegal),
            "max_position_gap": float(gap.max())}


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _region(strategy: Any, trunk: Trunk, module: SkillModule, kernels: Any, *, lanes: int, seed: int, held_out: bool,
            max_ticks: int | None, batch_lanes: int) -> dict:
    episode, key_rollout = region_episode(strategy, lanes, seed, held_out, max_ticks)
    sample = [int(lane) for lane in np.unique(np.linspace(0, lanes - 1, min(batch_lanes, lanes)).round().astype(int))]
    started = time.time()
    body, audit = drive(strategy, trunk, module, episode, key_rollout, kernels=kernels, sample=sample)
    body_seconds = round(time.time() - started, 1)
    use = jnp.zeros((lanes,), jnp.bool_)
    cut = trainer.rollout(strategy, trunk, module.params, episode, use, kernels, key_rollout, record=False)
    nocut = trainer.rollout(strategy, trunk, module.params, episode, use, kernels, key_rollout, record=False,
                            slot_cut=False)
    report = {"held_out": held_out, "lanes": lanes, "seed": seed, "body_seconds": body_seconds,
              "lane_kinds": [strategy.kinds[code] for code in np.asarray(episode.kind)],
              "body": _plain(skill_dagger.summarise(strategy, episode, body)),
              "trainer": _plain(skill_dagger.summarise(strategy, episode, cut)),
              "trainer_no_slot_cut": _plain(skill_dagger.summarise(strategy, episode, nocut)),
              "agreement": _agreement(body, cut), "agreement_no_slot_cut": _agreement(body, nocut),
              "slot_cut_effect": _agreement(cut, nocut), "audit": _plain(audit)}
    if max_ticks is None:  # evaluate's own report, on the same lanes and seed, must be the rollout's summary
        evaluation = trainer.evaluate(strategy, trunk, module.params, seed=seed, lanes=lanes, held_out=held_out,
                                      policy="module", kernels=kernels)
        report["evaluate_kinds_equal_trainer_rollout"] = _plain(evaluation["kinds"]) == report["trainer"]["kinds"]
        report["evaluate"] = _plain(evaluation)
    return report


def check_skill(folder: Path, skill: str, *, lanes: int, seed: int, max_ticks: int | None = None,
                batch_lanes: int = BATCH_LANES, trunk: Trunk | None = None) -> dict:
    """The check for one skill's module in ``folder`` (``trunk.npz`` and ``<skill>.npz``), both regions."""
    trunk = load_trunk(folder / "trunk.npz") if trunk is None else trunk
    module = load_module(folder / f"{skill}.npz", trunk)
    if module.skill != skill:
        raise ValueError(f"{folder / f'{skill}.npz'} is the module of {module.skill!r}, not {skill!r}")
    strategy = trainer._strategy(skill, SkillTrainConfig())
    known = tuple(sorted({spec.verb for spec in strategy.skills}))
    if module.known != known or module.config != strategy.config.module:
        raise ValueError(f"{skill}: the module's verbs or sizes differ from its strategy's")
    kernels = trainer.make_kernels(strategy, trunk, module.known)
    result = {"skill": skill, "scope": module.scope, "verb": module.verb, "known": list(module.known),
              "layout": list(state_tick.v2_layout(skill, tuple(spec.name for spec in strategy.beside))),
              "kinds": list(strategy.kinds), "trunk_sha256": trunk.sha256, "module_sha256": params_sha256(module.params),
              "max_ticks": max_ticks, "regions": {}}
    started = time.time()
    for region, held_out in (("seen", False), ("held_out", True)):
        result["regions"][region] = _region(strategy, trunk, module, kernels, lanes=lanes, seed=seed,
                                            held_out=held_out, max_ticks=max_ticks, batch_lanes=batch_lanes)
        print(f"{skill} {region}: {result['regions'][region]['body_seconds']} s body", file=sys.stderr, flush=True)
    result["seconds"] = round(time.time() - started, 1)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B6: each skill's module run as a ModuleBody, checked against the "
                                                 "trainer's evaluation on the same lanes and seed.")
    parser.add_argument("--bodies", type=Path, required=True, help="a folder: trunk.npz and <skill>.npz (with .json)")
    parser.add_argument("--lanes", type=int, default=32, help="lanes per region (seen and held out)")
    parser.add_argument("--seed", type=int, default=77001)
    parser.add_argument("--json", type=Path, required=True, help="the report (.json); must not exist")
    parser.add_argument("--skills", nargs="+", choices=SKILLS, default=list(SKILLS))
    parser.add_argument("--max-ticks", type=int, default=None, help="cap every horizon (a short test config)")
    parser.add_argument("--batch-lanes", type=int, default=BATCH_LANES, help="lanes ModuleBatch is checked on")
    args = parser.parse_args(argv)
    if args.json.exists():
        parser.error(f"{args.json} exists: a run never overwrites another")
    trunk = load_trunk(args.bodies / "trunk.npz")
    report = {"schema": "bodies_check_v1", "folder": str(args.bodies), "lanes": args.lanes, "seed": args.seed,
              "max_ticks": args.max_ticks, "skills": {}, "absent": []}
    for skill in args.skills:
        if not (args.bodies / f"{skill}.npz").is_file():
            report["absent"].append(skill)
            continue
        report["skills"][skill] = check_skill(args.bodies, skill, lanes=args.lanes, seed=args.seed,
                                              max_ticks=args.max_ticks, batch_lanes=args.batch_lanes, trunk=trunk)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1) + "\n")  # after each skill: a slow later one loses nothing
    print(json.dumps(report, indent=1))
    return 0


__all__ = ["BATCH_LANES", "SKILLS", "check_skill", "drive", "main", "region_episode", "runtime_request"]


if __name__ == "__main__":
    raise SystemExit(main())
