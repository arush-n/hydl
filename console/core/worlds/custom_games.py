"""Validated Arena minigame drafts bound to unique WorldGen designs."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from arena.curriculum.strategy import (
    AIM_OBJECTIVE,
    ARENA_TRAINING_STRATEGY_SCHEMA,
    ATTACK_TIMING_OBJECTIVE,
    GUARD_TIMING_OBJECTIVE,
    PURSUIT_OBJECTIVE,
    CausalObjective,
)
from arena.jax_contract import list_loadouts
from arena.tasks.framework.base import Difficulty, Task
from arena.tasks.framework.goals import Goal
from arena.tasks.framework.jax_gate import (
    REWARD_SCHEMA,
    SCHEMA as GOAL_COMPILE_SCHEMA,
    compile_goal_contract,
    compile_reward_contract,
)
from arena.tasks.framework.rewards import (
    TASK_REWARD_SCHEMA,
    RewardConfig,
    reward_for_goal,
)
from arena.tasks.framework.validate import validate
from arena.worlds import MAX_WORLD_POOL
from hytalegym.jax.combat.types import default_combat_params
from console.core.worlds import worldgen
from console.core.worlds.custom_goals import GOALS as _GOALS
from console.core.worlds.custom_goals import build as _goal

SCHEMA = "hytalerl_arena_custom_minigame_v1"
VERSION = 7
#: Beside the designs, under the shared world artifacts. A minigame names a
#: world a training run is identified by, so it has to outlive a scratch wipe
#: the same way its design does.
GAME_DIR = worldgen.DESIGN_DIR.parent / "minigames"
_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}")


_REWARD_PRESETS = {
    "balanced": RewardConfig(native_scale=0.0, state_scale=0.01, step_penalty=0.001),
    "dense": RewardConfig(native_scale=0.0, state_scale=0.02, step_penalty=0.001),
    "sparse": RewardConfig(
        native_scale=0.0,
        progress_scale=0.0,
        event_scale=0.0,
        completion_bonus=1.0,
    ),
    "potential": RewardConfig(
        native_scale=0.0,
        event_scale=0.0,
    ),
    "native": RewardConfig(
        progress_scale=0.0,
        event_scale=0.0,
        completion_bonus=0.0,
        failure_penalty=0.0,
    ),
}
_OBJECTIVES = {
    objective.axis.value: objective
    for objective in (
        PURSUIT_OBJECTIVE,
        AIM_OBJECTIVE,
        ATTACK_TIMING_OBJECTIVE,
        GUARD_TIMING_OBJECTIVE,
    )
}
_NATIVE_SCENE = default_combat_params(microticks=1)
_SCENE_DEFAULTS = {
    "agent_max_health": float(_NATIVE_SCENE.agent_max_health),
    "target_max_health": float(_NATIVE_SCENE.target_max_health),
    "sensor_range": float(_NATIVE_SCENE.sensor_range),
    "microticks": 1,
    "target_active": True,
}
_POOL_DEFAULTS = {"count": 2, "seed_stride": 1, "assignment_key": 0}


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    if not text:
        raise ValueError("minigame name must contain a letter or number")
    if len(text) > 48:
        raise ValueError("minigame name must be at most 48 normalized characters")
    return text


def _mapping(raw: Any, label: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be an object")
    return raw


def _number(
    value: Any, label: str, *, low: float, high: float, integer: bool = False
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(float(value)) or not low <= value <= high:
        raise ValueError(f"{label} must be between {low} and {high}")
    if integer and int(value) != value:
        raise ValueError(f"{label} must be an integer")
    return int(value) if integer else float(value)


def _reward(raw: Any, goal: Goal):
    values = _mapping(raw, "reward")
    if "objective" in values:
        unknown = sorted(set(values) - {"objective"})
        if unknown:
            raise ValueError(
                "an objective cannot be combined with reward field(s): "
                + ", ".join(unknown)
            )
        name = values["objective"]
        if name not in _OBJECTIVES:
            raise ValueError(
                f"unknown reward objective: {name}; have: {', '.join(_OBJECTIVES)}"
            )
        return _OBJECTIVES[name], name

    allowed = {"preset", *RewardConfig().describe()}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError("unknown reward field(s): " + ", ".join(unknown))
    preset = str(values.get("preset", "balanced"))
    if preset not in _REWARD_PRESETS:
        raise ValueError("unknown reward preset: " + preset)
    config = _REWARD_PRESETS[preset].describe()
    config.update({key: value for key, value in values.items() if key != "preset"})
    reward = reward_for_goal(goal, RewardConfig(**config))
    return reward, preset


def _scene(raw: Any) -> dict[str, Any]:
    values = _mapping(raw, "scene")
    unknown = sorted(set(values) - set(_SCENE_DEFAULTS))
    if unknown:
        raise ValueError("unknown scene field(s): " + ", ".join(unknown))
    merged = {**_SCENE_DEFAULTS, **values}
    target_active = merged["target_active"]
    if not isinstance(target_active, bool):
        raise ValueError("target_active must be boolean")
    return {
        "agent_max_health": _number(
            merged["agent_max_health"], "agent_max_health", low=1.0, high=10000.0
        ),
        "target_max_health": _number(
            merged["target_max_health"], "target_max_health", low=1.0, high=10000.0
        ),
        "sensor_range": _number(
            merged["sensor_range"], "sensor_range", low=1.0, high=128.0
        ),
        "microticks": _number(
            merged["microticks"], "microticks", low=1, high=16, integer=True
        ),
        "target_active": target_active,
    }


def _world_pool(raw: Any, base_seed: int) -> dict[str, Any]:
    values = _mapping(raw, "world_pool")
    unknown = sorted(set(values) - set(_POOL_DEFAULTS))
    if unknown:
        raise ValueError("unknown world_pool field(s): " + ", ".join(unknown))
    merged = {**_POOL_DEFAULTS, **values}
    count = _number(
        merged["count"], "world_pool.count", low=1, high=MAX_WORLD_POOL, integer=True
    )
    stride = _number(
        merged["seed_stride"],
        "world_pool.seed_stride",
        low=1,
        high=1_000_000,
        integer=True,
    )
    assignment = _number(
        merged["assignment_key"],
        "world_pool.assignment_key",
        low=-2_147_483_648,
        high=2_147_483_647,
        integer=True,
    )
    return {
        "count": count,
        "seed_stride": stride,
        "assignment_key": assignment,
        "seeds": [base_seed + index * stride for index in range(count)],
    }


def options() -> dict[str, Any]:
    """The complete, finite creator surface consumed by the Custom tab."""

    loadouts = list(list_loadouts())
    return {
        "schema": SCHEMA,
        "goals": [
            {
                "kind": kind,
                "label": row[0],
                "description": row[1],
                "fields": list(row[2]),
            }
            for kind, row in _GOALS.items()
        ],
        "loadouts": loadouts,
        "defaults": {
            "name": "my-minigame",
            "description": "",
            "goal": "defeat_target",
            "loadout": "iron_sword",
            "opponent": "iron_sword",
            "opponent_armed": False,
            "reward": {
                "preset": "balanced",
                **_REWARD_PRESETS["balanced"].describe(),
            },
            "scene": dict(_SCENE_DEFAULTS),
            "world_pool": dict(_POOL_DEFAULTS),
        },
        "reward_presets": {
            name: config.describe() for name, config in _REWARD_PRESETS.items()
        },
        "reward_objectives": {
            name: objective.manifest() for name, objective in _OBJECTIVES.items()
        },
        "reward_fields": [
            "native_scale",
            "progress_scale",
            "discount",
            "state_scale",
            "event_scale",
            "completion_bonus",
            "failure_penalty",
            "step_penalty",
            "clip",
        ],
        "scene_limits": {
            "health": [1.0, 10000.0],
            "sensor_range": [1.0, 128.0],
            "microticks": [1, 16],
        },
        "world_pool_limit": MAX_WORLD_POOL,
        "jax_contract": {
            "maximum_observation_groups": 1,
            "criterion_output": "bool[batch]",
            "criterion_compile_schema": GOAL_COMPILE_SCHEMA,
            "reward_output": "float32[batch]",
            "reward_is_explicit": True,
            "reward_schema": TASK_REWARD_SCHEMA,
            "reward_compile_schema": REWARD_SCHEMA,
            "full_rollout_requires_region_capture": True,
        },
    }


def _stored() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    if GAME_DIR.is_dir():
        for path in GAME_DIR.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("schema") != SCHEMA or record.get("game_id") != path.stem:
                    raise ValueError("identity or schema mismatch")
                if record.get("version") != VERSION:
                    raise ValueError(
                        f"stale manifest version {record.get('version')!r}; "
                        "re-save it under the current reward contract"
                    )
                _ = record["saved_at"], record["task"], record["world"]
                reward_schema = record["task"].get(
                    "reward_schema", record["task"].get("reward", {}).get("schema")
                )
                if reward_schema not in {
                    TASK_REWARD_SCHEMA,
                    ARENA_TRAINING_STRATEGY_SCHEMA,
                }:
                    raise ValueError("stale task reward contract")
                if (
                    record.get("jax_contract", {}).get("criterion", {}).get("schema")
                    != GOAL_COMPILE_SCHEMA
                ):
                    raise ValueError("stale compiled goal contract")
                if reward_schema == TASK_REWARD_SCHEMA and (
                    record.get("jax_contract", {}).get("reward", {}).get("schema")
                    != REWARD_SCHEMA
                ):
                    raise ValueError("stale compiled reward contract")
                if reward_schema == ARENA_TRAINING_STRATEGY_SCHEMA and (
                    record.get("jax_contract", {}).get("reward", {}).get("schema")
                    != ARENA_TRAINING_STRATEGY_SCHEMA
                ):
                    raise ValueError("stale objective reward contract")
                records.append(record)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                errors.append({"path": str(path), "error": str(exc)})
    records.sort(key=lambda item: item["saved_at"], reverse=True)
    return records, errors


def _summary(record: Mapping[str, Any]) -> dict[str, Any]:
    task, world = record["task"], record["world"]
    contract = record.get("jax_contract", {})
    return {
        "game_id": record["game_id"],
        "name": task["name"],
        "goal": task["goal"]["kind"],
        "world": world["config"]["name"],
        "design_id": world.get("design_id"),
        "seed": world["config"]["seed"],
        "world_pool_size": world.get("pool", {}).get("count", 1),
        "world_digest": world["generation_digest"],
        "status": record["status"],
        "saved_at": record["saved_at"],
        "jax_ready": contract.get("criterion_compiled", False)
        and contract.get("reward_compiled", False),
        "jax_backend": contract.get("backend"),
        "jax_compile_ms": contract.get("compile_ms"),
        "jax_groups": contract.get("required_groups", []),
        "reward_preset": task.get("reward_preset"),
        "reward_objective": task.get("reward_objective"),
    }


def _persist(record: Mapping[str, Any]) -> Path:
    GAME_DIR.mkdir(parents=True, exist_ok=True)
    target = GAME_DIR / f"{record['game_id']}.json"
    fd, temporary = tempfile.mkstemp(
        prefix=f".{record['game_id']}-", suffix=".tmp", dir=GAME_DIR
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def create(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and atomically persist one task/world pair."""

    if not isinstance(payload, Mapping):
        raise ValueError("custom minigame must be an object")
    allowed = {
        "name",
        "description",
        "goal",
        "goal_parameters",
        "loadout",
        "opponent",
        "opponent_armed",
        "reward",
        "scene",
        "world",
        "world_pool",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError("unknown custom minigame field(s): " + ", ".join(unknown))

    defaults = options()["defaults"]
    name = _slug(payload.get("name", defaults["name"]))
    goal = _goal(
        payload.get("goal", defaults["goal"]), payload.get("goal_parameters", {})
    )
    reward, reward_name = _reward(payload.get("reward"), goal)
    causal_objective = reward if isinstance(reward, CausalObjective) else None
    task_reward = None if causal_objective is not None else reward
    scene = _scene(payload.get("scene"))
    loadout = str(payload.get("loadout", defaults["loadout"]))
    opponent = str(payload.get("opponent", defaults["opponent"]))
    armed = payload.get("opponent_armed", defaults["opponent_armed"])
    if not isinstance(armed, bool):
        raise ValueError("opponent_armed must be boolean")
    known_loadouts = set(list_loadouts())
    if loadout not in known_loadouts or opponent not in known_loadouts:
        raise ValueError("loadout and opponent must name public JAX combat profiles")
    raw_world = payload.get("world") or {}
    if not isinstance(raw_world, Mapping):
        raise ValueError("world must be an object")
    pool = _world_pool(payload.get("world_pool"), 0)
    preview = worldgen.preview(dict(raw_world))
    # Persist the authored world as a design as well, so it appears in the same
    # picker a training run chooses from. A minigame whose world only existed
    # inside its own manifest could never be trained on.
    design = worldgen.save(dict(raw_world))
    base_seed = int(preview["config"]["seed"])
    pool["seeds"] = [
        base_seed + index * pool["seed_stride"] for index in range(pool["count"])
    ]
    generation_digest = sha256(
        json.dumps(
            {
                "world": {
                    key: value
                    for key, value in preview["config"].items()
                    if key != "name"
                },
                "pool_seeds": pool["seeds"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    description = str(payload.get("description") or goal.description).strip()
    if not description or len(description) > 240:
        raise ValueError("description must contain 1 to 240 characters")

    difficulty = Difficulty(
        "armed" if armed else "inert",
        opponent_armed=armed,
        opponent_profile=opponent,
        agent_max_health=scene["agent_max_health"],
        target_max_health=scene["target_max_health"],
        sensor_range=scene["sensor_range"],
    )
    task = validate(
        Task(
            name=name,
            description=description,
            heads_exercised=(),
            loadout=loadout,
            difficulties=(difficulty,),
            success=goal,
            reward=task_reward,
            scene_options={
                "microticks": scene["microticks"],
                "target_active": scene["target_active"],
            },
        )
    )
    task_data = {
        "name": task.name,
        "description": task.description,
        "loadout": task.loadout,
        "heads_exercised": list(task.heads_exercised),
        "difficulties": [
            {
                "name": difficulty.name,
                "opponent_armed": difficulty.opponent_armed,
                "opponent_profile": difficulty.opponent_profile,
                "agent_max_health": difficulty.agent_max_health,
                "target_max_health": difficulty.target_max_health,
                "sensor_range": difficulty.sensor_range,
            }
        ],
        "goal": goal.describe(),
        "reward": (
            {"schema": ARENA_TRAINING_STRATEGY_SCHEMA, **causal_objective.manifest()}
            if causal_objective is not None
            else reward.describe()
        ),
        "reward_schema": (
            ARENA_TRAINING_STRATEGY_SCHEMA
            if causal_objective is not None
            else TASK_REWARD_SCHEMA
        ),
        **(
            {"reward_objective": reward_name}
            if causal_objective is not None
            else {"reward_preset": reward_name}
        ),
        "scene_options": dict(task.scene_options),
    }
    identity = sha256(
        json.dumps(
            {
                "task": task_data,
                "world_digest": generation_digest,
                "assignment_key": pool["assignment_key"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    game_id = f"{name}-{identity[:10]}"

    records, errors = _stored()
    if errors:
        raise ValueError(
            "cannot prove world uniqueness while a saved manifest is corrupt"
        )
    matching = None
    for record in records:
        if record["game_id"] == game_id:
            matching = record
            break
        if record["task"]["name"] == name:
            raise ValueError(f"custom minigame name {name!r} is already in use")
        existing_world = record["world"]
        same_recipe = {
            key: value
            for key, value in existing_world["config"].items()
            if key not in {"name", "seed"}
        } == {
            key: value
            for key, value in preview["config"].items()
            if key not in {"name", "seed"}
        }
        existing_seeds = set(
            existing_world.get("pool", {}).get(
                "seeds", [existing_world["config"]["seed"]]
            )
        )
        if same_recipe and existing_seeds.intersection(pool["seeds"]):
            raise ValueError(
                f"this world design already belongs to {record['task']['name']!r}; "
                "change its seed or terrain so every minigame has a unique world"
            )

    goal_contract, goal_cache_hit = compile_goal_contract(goal, max_groups=1)
    if causal_objective is None:
        reward_contract, reward_cache_hit = compile_reward_contract(
            reward, max_groups=1
        )
    else:
        reward_contract = {
            "schema": ARENA_TRAINING_STRATEGY_SCHEMA,
            "reward_compiled": False,
            "reason": "causal objective requires stage evidence binding",
            "required_groups": [],
            "output": None,
            "compile_ms": 0.0,
            "objective": causal_objective.manifest(),
        }
        reward_cache_hit = False
    jax_contract = {
        "schema": "arena_custom_jax_compile_v1",
        "criterion_compiled": goal_contract["criterion_compiled"],
        "reward_compiled": reward_contract["reward_compiled"],
        "backend": goal_contract["backend"],
        "device": goal_contract["device"],
        "required_groups": sorted(
            set(goal_contract["required_groups"])
            | set(reward_contract["required_groups"])
        ),
        "compile_ms": round(
            goal_contract["compile_ms"] + reward_contract["compile_ms"], 3
        ),
        "criterion": goal_contract,
        "reward": reward_contract,
        "full_rollout_compile": "pending_region_capture",
    }
    cache_hit = goal_cache_hit and reward_cache_hit
    if matching is not None:
        matching["version"] = VERSION
        matching["jax_contract"] = jax_contract
        target = _persist(matching)
        return {
            **_summary(matching),
            "reused": True,
            "jax_cache_hit": cache_hit,
            "jax_cache_hits": {
                "criterion": goal_cache_hit,
                "reward": reward_cache_hit,
            },
            "path": str(target),
        }

    record = {
        "schema": SCHEMA,
        "version": VERSION,
        "game_id": game_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "status": "needs_region_capture",
        "runnable": False,
        "arena_validated": True,
        "jax_contract": jax_contract,
        "task": task_data,
        "world": {
            "kind": "worldgen_v2_design",
            "design_id": design["design_id"],
            "digest": preview["digest"],
            "generation_digest": generation_digest,
            "config": preview["config"],
            "pool": pool,
            "quality": preview["quality"],
            "warnings": preview["warnings"],
            "native_handoff": preview["design"]["native_handoff"],
        },
        "next_step": (
            "Compile and capture every seed in the bounded V2 pool, then bind "
            "the verified publication as one arena.tasks.PublishedWorldSpec "
            "and "
            "compile the fused rewarded JAX rollout/evaluator before running."
        ),
    }
    target = _persist(record)
    return {
        **_summary(record),
        "reused": False,
        "jax_cache_hit": cache_hit,
        "jax_cache_hits": {
            "criterion": goal_cache_hit,
            "reward": reward_cache_hit,
        },
        "path": str(target),
    }


def saved() -> dict[str, Any]:
    records, errors = _stored()
    return {
        "minigames": [_summary(record) for record in records],
        "errors": errors,
        "directory": str(GAME_DIR),
    }


def load(game_id: str) -> dict[str, Any] | None:
    if not _ID.fullmatch(game_id or ""):
        raise ValueError("invalid custom minigame id")
    path = GAME_DIR / f"{game_id}.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != SCHEMA or record.get("game_id") != game_id:
        raise ValueError("custom minigame identity does not match its filename")
    if record.get("version") != VERSION:
        raise ValueError("custom minigame uses a stale contract version")
    reward_schema = record.get("task", {}).get(
        "reward_schema", record.get("task", {}).get("reward", {}).get("schema")
    )
    if reward_schema not in {TASK_REWARD_SCHEMA, ARENA_TRAINING_STRATEGY_SCHEMA}:
        raise ValueError("custom minigame uses a stale reward contract")
    if (
        record.get("jax_contract", {}).get("criterion", {}).get("schema")
        != GOAL_COMPILE_SCHEMA
    ):
        raise ValueError("custom minigame uses a stale compiled goal contract")
    expected_reward_schema = (
        REWARD_SCHEMA
        if reward_schema == TASK_REWARD_SCHEMA
        else ARENA_TRAINING_STRATEGY_SCHEMA
    )
    if (
        record.get("jax_contract", {}).get("reward", {}).get("schema")
        != expected_reward_schema
    ):
        raise ValueError("custom minigame uses a stale compiled reward contract")
    return record


__all__ = ["GAME_DIR", "SCHEMA", "create", "load", "options", "saved"]
