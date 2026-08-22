"""Run a forced action and read the result without the usual traps.

Three of them cost real time this session and each is one line here:

* **Shape.** ``combat.yaw`` is ``(T, B, ENTITY)``; ``combat.pitch`` is
  ``(T, B)`` -- agent only.  Indexing pitch with ``[..., AGENT]`` raises.
  Same family as ``ability_accepted`` ``(T, B, ENTITY)`` versus
  ``damage_dealt`` ``(T, B)``.
* **Wrap.** Yaw wraps, so a raw ``np.diff`` yields +-360 artifacts.  One probe
  reported a median of -18 deg/tick alongside a total of +108 -- opposite signs.
* **Frame.** Movement is relative to facing, so world-frame displacement is
  only interpretable when yaw is pinned or explicitly rotated out.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import jax
import numpy as np

from adk.probes import repeat

#: Entity index of the learning agent in per-entity arrays.
AGENT_ENTITY = 0


def run_forced(
    handle: Any,
    factors: np.ndarray,
    record: Callable[[Any], Mapping[str, Any]],
    *,
    ticks: int,
    seed: int = 11,
) -> dict[str, np.ndarray]:
    """Hold ``factors`` for ``ticks`` steps and return the recorded arrays."""

    _final, records = handle.compile_collector(repeat(factors), record, ticks)(
        jax.random.key(seed)
    )
    return {name: np.asarray(value) for name, value in records.items()}


def agent_series(records: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    """The agent's rows of ``name``, whether or not it carries an entity axis.

    Returns ``(T, B)`` for scalar fields and ``(T, B, C)`` for vector fields
    such as ``position``.
    """

    array = np.asarray(records[name])
    if array.ndim >= 3 and array.shape[2] > AGENT_ENTITY:
        # Per-entity: (T, B, ENTITY[, C]).  Position is (T, B, ENTITY, 3), and
        # a 3-dim array is ambiguous, so treat axis 2 as the entity axis only
        # when the field is known to carry one.
        return array[:, :, AGENT_ENTITY]
    return array


def shortest_arc(degrees: np.ndarray) -> np.ndarray:
    """Per-step angular delta reduced to ``(-180, 180]``.

    Use for anything that wraps.  ``np.diff`` alone is wrong on yaw.
    """

    delta = np.diff(np.asarray(degrees, dtype=np.float64), axis=0)
    return (delta + 180.0) % 360.0 - 180.0


def to_local_frame(delta_xz: np.ndarray, yaw_degrees: np.ndarray) -> np.ndarray:
    """Rotate world-frame XZ displacement into the agent's facing frame.

    ``delta_xz`` is ``(..., 2)`` as ``(dx, dz)``; ``yaw_degrees`` broadcasts
    against its leading axes.  Prefer *pinning* yaw to its neutral option over
    rotating after the fact -- rotation uses one yaw sample per step while the
    body may also be turning during that step.
    """

    angle = np.radians(np.asarray(yaw_degrees, dtype=np.float64))
    dx, dz = delta_xz[..., 0], delta_xz[..., 1]
    return np.stack(
        [dx * np.cos(angle) + dz * np.sin(angle),
         -dx * np.sin(angle) + dz * np.cos(angle)],
        axis=-1,
    )


def planar_travel(position: np.ndarray) -> float:
    """Total XZ distance between the first and last recorded position.

    ``position`` is the agent's ``(T, B, 3)`` series; environment row 0 is used
    because a forced action is identical across the batch.
    """

    delta = position[-1, 0] - position[0, 0]
    return float(np.linalg.norm(delta[[0, 2]]))


def planar_heading(position: np.ndarray) -> float | None:
    """Compass heading of net XZ travel in degrees, or None if stationary."""

    delta = position[-1, 0] - position[0, 0]
    if float(np.linalg.norm(delta[[0, 2]])) <= 1e-4:
        return None
    return float(np.degrees(np.arctan2(delta[2], delta[0])))


def per_tick(total: float, ticks: int) -> float:
    """Rate per tick.  Multiply by 30 for per second -- the server is 30 TPS."""

    return total / float(ticks)
