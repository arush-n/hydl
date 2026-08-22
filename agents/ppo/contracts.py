"""Stable identities shared by PPO training and evaluation receipts."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from arena.tasks.framework.base import Task


TASK_CONTRACT_SCHEMA = "hytalerl_arena_task_evaluation_contract_v3"


def task_contract(task: Task | None, difficulty: str) -> dict[str, Any] | None:
    """Pin the success, reward, opponent, and action intent of one task rung."""

    if task is None:
        return None
    rung = task.difficulty(difficulty)
    goal = task.goal
    success = (
        goal.describe()
        if goal is not None
        else {
            "kind": "callable",
            "identity": (
                f"{task.success.__module__}."
                f"{getattr(task.success, '__qualname__', type(task.success).__name__)}"
            ),
        }
    )
    body = {
        "schema": TASK_CONTRACT_SCHEMA,
        "name": task.name,
        "description": task.description,
        "loadout": task.loadout,
        "difficulty": {
            "name": rung.name,
            "opponent_armed": rung.opponent_armed,
            "opponent_profile": rung.opponent_profile,
            "parameters": rung.combat_parameters(),
        },
        "success": success,
        "reward": None if task.reward is None else task.reward.describe(),
        "heads_exercised": list(task.heads_exercised),
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return {
        **body,
        "sha256": hashlib.sha256(encoded.encode()).hexdigest().upper(),
    }


__all__ = ["TASK_CONTRACT_SCHEMA", "task_contract"]
