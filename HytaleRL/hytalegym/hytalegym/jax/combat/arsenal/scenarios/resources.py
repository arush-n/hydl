"""Host-side actor resource scenarios for authored Arsenal programs.

Weapon profiles describe interaction graphs and their resource requirements.
They do not grant an actor a resource capacity.  This module is the explicit
composition seam for fixtures that intentionally add a stat/effect-backed
resource budget.
"""

from __future__ import annotations

from collections.abc import Mapping
import operator

import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat.arsenal.schema.types import AbilityLoadout
from hytalegym.jax.combat.mechanics import RESOURCE_COUNT


def with_scenario_resources(
    loadout: AbilityLoadout,
    *,
    maximum_by_resource: Mapping[int, object] | None = None,
    initial_by_resource: Mapping[int, object] | None = None,
) -> AbilityLoadout:
    """Return ``loadout`` with explicit per-actor resource state.

    Mapping keys are resource IDs. Values may be scalars or arrays
    broadcastable to the loadout's ``[B, N]`` actor shape. Maximum and
    initial values are independent on purpose: raising a native stat maximum
    does not imply that its current value is also filled.

    This is a host construction helper, not a jitted transition. Callers that
    need Java parity must apply an equivalent native stat/effect modifier to
    the same actors; this function never infers one from a weapon name.
    """

    if not isinstance(loadout, AbilityLoadout):
        raise TypeError("loadout must be an AbilityLoadout")
    maximum_updates = _resource_updates(
        maximum_by_resource,
        actor_shape=loadout.resource_maximum.shape[:2],
        field="maximum_by_resource",
    )
    initial_updates = _resource_updates(
        initial_by_resource,
        actor_shape=loadout.resource_initial.shape[:2],
        field="initial_by_resource",
    )
    if not maximum_updates and not initial_updates:
        raise ValueError("at least one resource maximum or initial value is required")

    maximum = np.asarray(loadout.resource_maximum, dtype=np.float32).copy()
    initial = np.asarray(loadout.resource_initial, dtype=np.float32).copy()
    for resource_id, values in maximum_updates.items():
        maximum[..., resource_id] = values
    for resource_id, values in initial_updates.items():
        initial[..., resource_id] = values
    if np.any(initial > maximum):
        row = tuple(int(value) for value in np.argwhere(initial > maximum)[0])
        raise ValueError(
            "resource initial value exceeds its maximum at "
            f"batch={row[0]}, entity={row[1]}, resource={row[2]}"
        )
    return loadout._replace(
        resource_maximum=jnp.asarray(maximum, dtype=jnp.float32),
        resource_initial=jnp.asarray(initial, dtype=jnp.float32),
    )


def _resource_updates(
    values_by_resource: Mapping[int, object] | None,
    *,
    actor_shape: tuple[int, ...],
    field: str,
) -> dict[int, np.ndarray]:
    if values_by_resource is None:
        return {}
    if not isinstance(values_by_resource, Mapping):
        raise TypeError(f"{field} must be a mapping from resource ID to values")
    if len(actor_shape) != 2:
        raise ValueError("loadout resource arrays must expose a [B, N] actor axis")

    updates: dict[int, np.ndarray] = {}
    for raw_resource_id, raw_values in values_by_resource.items():
        if isinstance(raw_resource_id, bool):
            raise TypeError(f"{field} resource IDs must be integers")
        try:
            resource_id = operator.index(raw_resource_id)
        except TypeError as error:
            raise TypeError(f"{field} resource IDs must be integers") from error
        if not 0 <= resource_id < RESOURCE_COUNT:
            raise ValueError(
                f"{field} resource ID must be in [0, {RESOURCE_COUNT}), "
                f"got {resource_id}"
            )
        try:
            values = np.asarray(raw_values, dtype=np.float32)
            broadcast = np.broadcast_to(values, actor_shape)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{field}[{resource_id}] must be scalar or broadcastable to "
                f"actor shape {actor_shape}"
            ) from error
        if not np.all(np.isfinite(broadcast)):
            raise ValueError(f"{field}[{resource_id}] must be finite")
        if np.any(broadcast < 0.0):
            raise ValueError(f"{field}[{resource_id}] must be non-negative")
        updates[resource_id] = np.array(broadcast, dtype=np.float32, copy=True)
    return updates


__all__ = ["with_scenario_resources"]
