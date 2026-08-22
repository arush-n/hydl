"""Finite goal options for the Custom minigame editor."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

from arena.tasks.framework.goals import (
    Goal,
    MOVEMENT_STATES,
    RESOURCE_NAMES,
    avoid_hazards,
    changed_terrain_at_interaction,
    defeat_target,
    defeat_within,
    gather_resource,
    hold_movement_state,
    keep_health_above,
    keep_target_visible,
    reach_interaction,
    spend_no_more_than,
    stay_alive,
    survive_for,
)

GoalRow = tuple[str, str, tuple[dict[str, Any], ...], Callable]


def _number(name, label, default, low=0.0, high=1.0, step=0.05, *, integer=False):
    field = {
        "name": name,
        "label": label,
        "type": "integer" if integer else "number",
        "min": low,
        "max": high,
        "default": default,
    }
    if not integer:
        field["step"] = step
    return field


def _choice(name, label, choices, default):
    return {
        "name": name,
        "label": label,
        "type": "choice",
        "choices": list(choices),
        "default": default,
    }


def _resource(factory, values):
    return factory(resource=values["resource"], amount=values["amount"])


GOALS: dict[str, GoalRow] = {
    "defeat_target": (
        "Defeat target",
        "Observe the designated opponent at zero health.",
        (),
        lambda _v: defeat_target(),
    ),
    "defeat_within": (
        "Timed duel",
        "Observe the target at zero health inside a fixed tick budget.",
        (_number("ticks", "tick budget", 512, 1, 2048, integer=True),),
        lambda v: defeat_within(ticks=v["ticks"]),
    ),
    "stay_alive": (
        "Stay alive",
        "Be alive when the episode ends.",
        (),
        lambda _v: stay_alive(),
    ),
    "survive_for": (
        "Survive",
        "Stay alive continuously for a fixed number of ticks.",
        (_number("ticks", "survival ticks", 512, 1, 2048, integer=True),),
        lambda v: survive_for(ticks=v["ticks"]),
    ),
    "keep_health_above": (
        "Protect health",
        "Never fall below a health fraction.",
        (_number("fraction", "minimum health", 0.5),),
        lambda v: keep_health_above(v["fraction"]),
    ),
    "keep_target_visible": (
        "Track target",
        "Keep the opponent visible for a fraction of the rollout.",
        (_number("fraction", "visible fraction", 0.75),),
        lambda v: keep_target_visible(fraction=v["fraction"]),
    ),
    "avoid_hazards": (
        "Avoid hazards",
        "Never move inside the requested hazard distance.",
        (_number("distance", "minimum distance", 3.0, 0.0, 64.0, 0.5),),
        lambda v: avoid_hazards(distance=v["distance"]),
    ),
    "reach_interaction": (
        "Reach target",
        "Bring the interaction target within reach.",
        (_number("reach", "reach fraction", 0.999, 0.01, 1.0, 0.01),),
        lambda v: reach_interaction(reach=v["reach"]),
    ),
    "changed_terrain_at_interaction": (
        "Change terrain",
        "Change traversability at the interaction target.",
        (_number("minimum_change", "minimum change", 0.01, 0.0001, 1.0, 0.001),),
        lambda v: changed_terrain_at_interaction(minimum_change=v["minimum_change"]),
    ),
    "hold_movement_state": (
        "Movement state",
        "Spend a fraction of the rollout in one movement state.",
        (
            _choice("state", "movement state", MOVEMENT_STATES, MOVEMENT_STATES[0]),
            _number("fraction", "rollout fraction", 0.5),
        ),
        lambda v: hold_movement_state(state=v["state"], fraction=v["fraction"]),
    ),
    "gather_resource": (
        "Gather resource",
        "Reach a normalized resource amount.",
        (
            _choice("resource", "resource", RESOURCE_NAMES, "stamina"),
            _number("amount", "target amount", 0.75),
        ),
        lambda v: _resource(gather_resource, v),
    ),
    "spend_no_more_than": (
        "Resource economy",
        "Finish with at least a normalized resource amount.",
        (
            _choice("resource", "resource", RESOURCE_NAMES, "stamina"),
            _number("amount", "minimum remaining", 0.5),
        ),
        lambda v: _resource(spend_no_more_than, v),
    ),
}


def build(kind: Any, raw: Any) -> Goal:
    if kind not in GOALS:
        raise ValueError(f"unknown Arena goal {kind!r}; have: {', '.join(GOALS)}")
    if not isinstance(raw, Mapping):
        raise ValueError("goal_parameters must be an object")
    fields = GOALS[kind][2]
    unknown = sorted(set(raw) - {field["name"] for field in fields})
    if unknown:
        raise ValueError("unknown goal parameter(s): " + ", ".join(unknown))
    values: dict[str, Any] = {}
    for field in fields:
        value = raw.get(field["name"], field["default"])
        if field["type"] == "choice":
            if value not in field["choices"]:
                raise ValueError(
                    f"{field['name']} must be one of: {', '.join(field['choices'])}"
                )
            values[field["name"]] = value
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field['name']} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{field['name']} must be finite")
        if not field["min"] <= value <= field["max"]:
            raise ValueError(
                f"{field['name']} must be between {field['min']} and {field['max']}"
            )
        if field["type"] == "integer" and int(value) != value:
            raise ValueError(f"{field['name']} must be an integer")
        values[field["name"]] = (
            int(value) if field["type"] == "integer" else float(value)
        )
    return GOALS[kind][3](values)


__all__ = ["GOALS", "GoalRow", "build"]
