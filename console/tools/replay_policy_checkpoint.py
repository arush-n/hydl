"""Replay compatible JAX policy checkpoints into the Console archive.

Run this with the immutable source snapshot that produced the checkpoints first
on ``PYTHONPATH``.  Several checkpoints in this workspace intentionally fail
against the live combat ABI; replaying them against their original code keeps
the visualization useful without pretending they are deployable.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from agents.ppo.worldgen.worldgen_benchmark import GENERATED
from arena.jax_contract import GROUP_FEATURES
from arena.jax_env import ArenaHandle
from arena.tasks.framework.base import Task
from arena.worlds import WorldSpec, world_spec_for_split
from console.core.evidence import store
from console.core.worlds import replay_terrain
from hytalegym.jax.combat import AGENT_ENTITY, TARGET_ENTITY
from hytalegym.jax.combat.observation.v3.policy import (
    greedy_arsenal_action_factors,
)
from hytalegym.jax.combat.observation.v3.tokens.world import (
    DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG,
)
from hytalegym.jax.training.checkpoint import (
    ARSENAL_POLICY_SURFACE,
    combat_checkpoint_contract_errors,
    load_policy_checkpoint,
)
from hytalegym.jax.training.policy import apply_policy, sample_configured_actions


PHASES = ("idle", "windup", "sweep", "recovery", "cooldown")
COMBAT = GROUP_FEATURES["combat_f32"]
VISIBLE = COMBAT.index("target_visible")
OBSERVED_DISTANCE = COMBAT.index("visible_target_planar_distance")
ATTACK_EXECUTING = COMBAT.index("agent_attack_executing")
PHASE_COLUMNS = jnp.asarray(
    [COMBAT.index(f"target_phase_{name}") for name in PHASES]
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--task", default="plains_duel")
    parser.add_argument("--difficulty", default="armed")
    parser.add_argument("--ticks", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=580057)
    parser.add_argument("--mode", choices=("sample", "greedy"), default="sample")
    parser.add_argument("--split", choices=("training", "heldout"), default="heldout")
    parser.add_argument("--console-root", type=Path, default=store.ARTIFACT_ROOT)
    args = parser.parse_args()
    if args.ticks < 1 or args.batch < 1 or args.worlds < 1:
        parser.error("ticks, batch, and worlds must be positive")
    if args.worlds > args.batch:
        parser.error("batch must cover every requested world")
    return args


def _task(name: str, split: str, worlds: int) -> Task:
    matches = [task for task in GENERATED if task.name == name]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate generated task {name!r}")
    task = matches[0]
    if split == "training":
        return task
    if not isinstance(task.world, WorldSpec):
        raise TypeError("held-out replay requires a WorldGen V2 WorldSpec")
    world = world_spec_for_split(
        f"{task.world.name}_heldout",
        task.world.structure,
        "heldout",
        count=worlds,
        selection_key=0,
    )
    overlap = set(task.world.seeds) & set(world.seeds)
    if overlap:
        raise RuntimeError(f"training and replay worlds overlap: {sorted(overlap)}")
    return replace(task, name=f"{task.name}_heldout", world=world)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def _seed_from_path(path: Path) -> int | None:
    for parent in path.parents:
        if parent.name.startswith("seed-"):
            try:
                return int(parent.name.removeprefix("seed-"))
            except ValueError:
                return None
    return None


def _load(paths: list[Path], scene: Any):
    loaded = []
    reference = None
    for path in paths:
        params, config, metadata = load_policy_checkpoint(path)
        contract = metadata.get("transfer_contract")
        if not isinstance(contract, dict):
            raise ValueError(f"{path}: missing transfer_contract")
        errors = combat_checkpoint_contract_errors(
            contract,
            arsenal_runtime_capacity=scene.runtime_capacity,
            policy_surface=ARSENAL_POLICY_SURFACE,
            world_geometry_config=DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG,
        )
        if errors:
            raise ValueError(f"{path}: incompatible checkpoint\n  " + "\n  ".join(errors))
        if config.observation_size != scene.environment.spec.observation_size:
            raise ValueError(f"{path}: observation width does not match scene")
        if tuple(config.action_head_sizes) != tuple(
            component.size for component in scene.environment.spec.action_components
        ):
            raise ValueError(f"{path}: action heads do not match scene")
        identity = json.dumps(asdict(config), sort_keys=True)
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError("all replay checkpoints must share one policy config")
        loaded.append((path.resolve(), params, config, metadata))
    return loaded


def _record(transition):
    runtime = transition.next_state.runtime
    previous = transition.state.runtime
    combat = runtime.combat
    arsenal = runtime.arsenal
    mechanics = runtime.mechanics
    combat_f32 = transition.next_actor_input.legal_observation.base.combat_f32
    return {
        "position": combat.position,
        "yaw": combat.yaw,
        "health": combat.health,
        "grounded": combat.agent_grounded,
        "visible": combat_f32[:, VISIBLE],
        "observed_distance": combat_f32[:, OBSERVED_DISTANCE],
        "attack_executing": combat_f32[:, ATTACK_EXECUTING],
        "target_phase": combat_f32[:, PHASE_COLUMNS],
        "requested": transition.info.arsenal_info.ability_requested,
        "accepted": transition.info.arsenal_info.ability_accepted,
        "damage_dealt": transition.info.arsenal_info.damage_dealt,
        "active_slot": arsenal.active_ability_slot,
        "cooldown": previous.arsenal.ability_cooldown_seconds[
            :, AGENT_ENTITY, 0
        ],
        "projectiles": transition.info.arsenal_info.projectile_count,
        "stamina": mechanics.resources[..., 0],
        "guard_active": mechanics.guard_active,
        "action_legal": transition.info.action_surface_legal,
        "arsenal_failures": arsenal.failure_bits,
        "geometry_exhausted": combat.geometry_exhausted,
        "navigation_unsupported": combat.target_navigation_unsupported,
        "reward": transition.reward,
        "terminated": transition.terminated,
        "truncated": transition.truncated,
        "done": transition.done,
        "action": transition.action_factors,
    }


def _policy(config, mode: str):
    def apply(parameters, carry, actor_input, key):
        carry, logits, _ = apply_policy(
            parameters,
            actor_input.observation,
            carry,
            actor_input.action_mask,
        )
        if mode == "sample":
            factors, _ = sample_configured_actions(key, logits, config)
        else:
            factors = greedy_arsenal_action_factors(logits)
        return carry, factors

    return apply


def _numbers(values: np.ndarray, digits: int = 4) -> list[float | None]:
    return [round(float(x), digits) if math.isfinite(float(x)) else None for x in values]


def _ints(values: np.ndarray) -> list[int]:
    return [int(x) for x in values]


def _aim_error(position: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    offset = position[:, TARGET_ENTITY] - position[:, AGENT_ENTITY]
    bearing = np.degrees(np.arctan2(-offset[:, 0], -offset[:, 2]))
    return np.abs((bearing - yaw[:, AGENT_ENTITY] + 180.0) % 360.0 - 180.0)


def _episode(trajectory: dict[str, np.ndarray], lane: int) -> dict[str, Any]:
    done = np.asarray(trajectory["done"][:, lane], dtype=np.bool_)
    end = int(np.flatnonzero(done)[0] + 1) if done.any() else len(done)
    take = lambda name: np.asarray(trajectory[name])[:end, lane]  # noqa: E731
    position = take("position")
    yaw = take("yaw")
    health = take("health")
    action = take("action")
    active_slot = take("active_slot")[:, AGENT_ENTITY]
    requested = take("requested")[:, AGENT_ENTITY].astype(np.bool_)
    accepted = take("accepted")[:, AGENT_ENTITY].astype(np.bool_)
    cooldown = take("cooldown")
    damage = take("damage_dealt")
    legal = take("action_legal").astype(np.bool_)
    visible = take("visible") > 0.5
    distance = np.linalg.norm(
        (position[:, TARGET_ENTITY] - position[:, AGENT_ENTITY])[:, (0, 2)], axis=1
    )
    aim = _aim_error(position, yaw)
    phase = np.argmax(take("target_phase"), axis=1)
    starts = (active_slot >= 0) & np.r_[True, active_slot[:-1] < 0]
    prior_slot = np.r_[-1, active_slot[:-1]]
    cooldown_reject = requested & ~accepted & (cooldown > 1.0e-3)
    busy_reject = requested & ~accepted & (prior_slot >= 0)
    yaw_delta = np.abs((np.diff(yaw[:, AGENT_ENTITY]) + 180.0) % 360.0 - 180.0)
    reward = take("reward")

    result = {
        "steps": end,
        "trajectory": {
            "agent_x": _numbers(position[:, AGENT_ENTITY, 0]),
            "agent_y": _numbers(position[:, AGENT_ENTITY, 1]),
            "agent_z": _numbers(position[:, AGENT_ENTITY, 2]),
            "target_x": _numbers(position[:, TARGET_ENTITY, 0]),
            "target_y": _numbers(position[:, TARGET_ENTITY, 1]),
            "target_z": _numbers(position[:, TARGET_ENTITY, 2]),
            "agent_yaw": _numbers(yaw[:, AGENT_ENTITY], 3),
            "target_yaw": _numbers(yaw[:, TARGET_ENTITY], 3),
            "agent_health": _numbers(health[:, AGENT_ENTITY], 2),
            "target_health": _numbers(health[:, TARGET_ENTITY], 2),
            "true_distance": _numbers(distance, 3),
            "observed_distance": _numbers(take("observed_distance"), 3),
            "visible": _ints(visible),
            "grounded": _ints(take("grounded")),
            "requested": _ints(requested),
            "accepted": _ints(accepted),
            "attack_executing": _ints(take("attack_executing") > 0.5),
            "damage_dealt": _numbers(damage, 3),
            "target_phase": _ints(phase),
            "reward": _numbers(reward, 6),
            "done": _ints(done[:end]),
            "terminated": _ints(take("terminated")),
            "truncated": _ints(take("truncated")),
            "action": [[int(value) for value in row] for row in action],
            "active_slot": _ints(active_slot),
            "ability_start": _ints(starts),
            "projectiles": _ints(take("projectiles")),
            "stamina": _numbers(take("stamina")[:, AGENT_ENTITY], 2),
            "guard_active": _ints(take("guard_active")[:, AGENT_ENTITY]),
            "cooldown": _numbers(cooldown, 4),
            "cooldown_reject": _ints(cooldown_reject),
            "busy_reject": _ints(busy_reject),
            "aim_error": _numbers(aim, 2),
            "action_legal": _ints(legal),
            "navigation_unsupported": _ints(take("navigation_unsupported")),
            "geometry_exhausted": _ints(take("geometry_exhausted")),
            "arsenal_failures": _ints(take("arsenal_failures")),
        },
    }

    events: list[dict[str, Any]] = []

    def note(tick: int, kind: str, detail: str, severity: str = "info") -> None:
        events.append({"tick": tick, "kind": kind, "detail": detail, "severity": severity})

    request_edge = requested & ~np.r_[False, requested[:-1]]
    for tick in np.flatnonzero(request_edge):
        note(int(tick), "attack_request", f"aim error {aim[tick]:.1f} degrees")
    for tick in np.flatnonzero(accepted):
        note(int(tick), "attack_accepted", "basic attack admitted", "good")
    for tick in np.flatnonzero(cooldown_reject):
        note(int(tick), "cooldown_reject", f"{cooldown[tick]:.3f}s remaining", "warn")
    for tick in np.flatnonzero(damage > 0):
        note(int(tick), "hit", f"{damage[tick]:.1f} damage dealt", "good")
    for tick in np.flatnonzero(~legal):
        note(int(tick), "illegal_action", "joint action rejected", "crit")
    for name, values in (
        ("navigation_unsupported", take("navigation_unsupported")),
        ("geometry_exhausted", take("geometry_exhausted")),
    ):
        found = np.flatnonzero(values)
        if found.size:
            note(int(found[0]), name, "support boundary first encountered", "crit")
    if done[:end].any():
        note(end - 1, "episode_end", "environment terminal", "crit")

    damage_total = max(0.0, float(health[0, TARGET_ENTITY] - health[-1, TARGET_ENTITY]))
    agent_damage = max(0.0, float(health[0, AGENT_ENTITY] - health[-1, AGENT_ENTITY]))
    landed = int(np.count_nonzero(damage > 0))
    accepted_count = int(np.count_nonzero(accepted))
    if health[-1, TARGET_ENTITY] <= 0 < health[-1, AGENT_ENTITY]:
        outcome = "victory"
    elif health[-1, AGENT_ENTITY] <= 0 < health[-1, TARGET_ENTITY]:
        outcome = "death"
    elif health[-1, AGENT_ENTITY] <= 0 and health[-1, TARGET_ENTITY] <= 0:
        outcome = "simultaneous"
    elif bool(take("truncated")[-1]):
        outcome = "support_truncation"
    else:
        outcome = "unresolved"
    result["events"] = sorted(events, key=lambda item: item["tick"])
    result["summary"] = {
        "outcome": outcome,
        "episode_ticks": end,
        "requested": int(np.count_nonzero(requested)),
        "accepted": accepted_count,
        "landed": landed,
        "hit_rate_per_accepted": round(landed / max(1, accepted_count), 4),
        "cooldown_rejects": int(np.count_nonzero(cooldown_reject)),
        "busy_rejects": int(np.count_nonzero(busy_reject)),
        "rejected_requests": int(np.count_nonzero(requested & ~accepted)),
        "illegal_actions": int(np.count_nonzero(~legal)),
        "damage_dealt": round(damage_total, 3),
        "damage_taken": round(agent_damage, 3),
        "reward_total": round(float(np.sum(reward)), 6),
        "visible_fraction": round(float(np.mean(visible)), 4),
        "aimed_within_15deg_fraction": round(float(np.mean(visible & (aim <= 15.0))), 4),
        "mean_visible_aim_error_degrees": round(
            float(np.mean(aim[visible])) if visible.any() else math.nan, 3
        ),
        "mean_abs_yaw_delta_degrees": round(float(np.mean(yaw_delta)) if yaw_delta.size else 0.0, 3),
        "navigation_unsupported": bool(np.any(take("navigation_unsupported"))),
        "geometry_exhausted": bool(np.any(take("geometry_exhausted"))),
        "resolved": bool(done[:end].any()),
    }
    result["warnings"] = []
    if outcome == "unresolved":
        result["warnings"].append(
            f"fight was still unresolved at the {end}-tick horizon"
        )
    if result["summary"]["navigation_unsupported"]:
        result["warnings"].append("target navigation crossed the surrogate support boundary")
    if result["summary"]["geometry_exhausted"]:
        result["warnings"].append("exact Region geometry was exhausted")
    if result["summary"]["busy_rejects"]:
        result["warnings"].append(
            f"{result['summary']['busy_rejects']} attack requests retried while "
            "the previous attack was still active"
        )
    if damage_total <= 1.0e-6 and float(np.sum(reward)) > 1.0e-6:
        result["warnings"].append("positive return without raw target damage")
    return result


def main() -> None:
    args = _arguments()
    started = perf_counter()
    task = _task(args.task, args.split, args.worlds)
    scene = task.build_scene(args.difficulty, batch=args.batch)
    loaded = _load(args.checkpoints, scene)
    config = loaded[0][2]
    collector = ArenaHandle(scene).compile_parameterized_collector(
        _policy(config, args.mode),
        _record,
        args.ticks,
        initial_carry=jnp.zeros((args.batch, config.recurrent_size), dtype=jnp.float32),
    )
    opened = store.begin()
    source_root = Path(__import__("agents.ppo.worldgen.worldgen_benchmark", fromlist=["x"]).__file__).resolve().parents[2]
    cycle = list(scene.world_identity.get("assignment_seed_cycle", ()))
    entries = []
    for path, params, _config, metadata in loaded:
        _, recorded = collector(params, jax.random.key(args.seed))
        recorded = jax.device_get(recorded)
        update = int(metadata.get("updates", -1))
        arm = str(metadata.get("training_arm", "policy")).replace("_", "-")
        for lane in range(args.batch):
            result = _episode(recorded, lane)
            world_seed = int(cycle[lane % len(cycle)]) if cycle else None
            label = f"{arm} u{update} / world {world_seed}"
            spec = {
                "kind": "jax_policy_checkpoint_replay_v1",
                "task": args.task,
                "difficulty": args.difficulty,
                "loadout": task.loadout,
                "opponent": task.difficulty(args.difficulty).opponent_profile,
                "policy": label,
                "checkpoint": str(path),
                "checkpoint_sha256": _sha(path),
                "checkpoint_update": update,
                "training_arm": metadata.get("training_arm"),
                "training_seed": _seed_from_path(path),
                "seed": args.seed,
                "lane": lane,
                "ticks": args.ticks,
                "decision_period": 1,
                "microticks": 1,
                "evaluation_mode": args.mode,
                "evaluation_split": args.split,
                "world": "worldgen_v2",
                "world_seed": world_seed,
                "world_pool": list(task.world.seeds),
                "target_active": True,
                "armed": True,
                "perceive": True,
                "immutable_source_root": str(source_root),
                "source_contract_note": "historical checkpoint replayed only against its exact frozen ABI",
            }
            result.update(
                head_names=[component.name for component in scene.environment.spec.action_components],
                target_phases=list(PHASES),
                region={**dict(scene.world_identity), "replay_lane": lane, "replay_world_seed": world_seed},
                world_identity={
                    **dict(scene.world_identity),
                    "replay_lane": lane,
                    "replay_world_seed": world_seed,
                },
                terrain=replay_terrain.worldgen_reference(
                    scene.world_identity,
                    seed=world_seed,
                ),
            )
            result["summary"].update(
                checkpoint_update=update,
                world_seed=world_seed,
                compile_and_replay_wall_seconds=round(perf_counter() - started, 2),
            )
            result["warnings"].insert(
                0,
                "historical frozen-contract replay; this checkpoint is not deployable on the live combat ABI",
            )
            entries.append(
                store.write(
                    spec,
                    result,
                    opened,
                    root=args.console_root.resolve(),
                    world_geometry_config=DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG,
                )
            )
    print(json.dumps({"backend": jax.default_backend(), "source_root": str(source_root), "runs": entries}, indent=2))


if __name__ == "__main__":
    main()
