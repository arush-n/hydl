"""The agent's variables, read and written by name (``docs/VIRTUAL_HUMAN.md`` section 10.2), and migrated by name
(section 14, rule 4).

The variables of B agents are one array float32[B, V] in ``contract/variables.NAMES`` order (``AgentState.variables``,
``layer.py``). Every layer reads them through :func:`get`, :func:`take` or :func:`section`, never by position.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.contract import variables as registry

INDEX: Mapping[str, int] = {name: index for index, name in enumerate(registry.NAMES)}


def defaults(batch: int) -> jax.Array:
    """float32[B, V]: every variable at its declared default."""

    return jnp.broadcast_to(jnp.asarray(registry.DEFAULTS), (batch, len(registry.NAMES)))


def get(values: jax.Array, name: str) -> jax.Array:
    """float32[B]: the variable ``name``."""

    return values[..., INDEX[name]]


def take(values: jax.Array, names: Sequence[str]) -> jax.Array:
    """float32[B, len(names)]: those variables, in that order."""

    return values[..., np.asarray([INDEX[name] for name in names], np.int32)]


def section(values: jax.Array, name: str) -> dict[str, jax.Array]:
    """{variable: float32[B]} for every variable of section ``name`` (``variables.SECTIONS``)."""

    return {variable: get(values, variable) for variable in registry.names(name)}


def put(values: jax.Array, names: Sequence[str], new: jax.Array) -> jax.Array:
    """``values`` with those variables set to ``new`` float32[B, len(names)]."""

    return values.at[..., np.asarray([INDEX[name] for name in names], np.int32)].set(new)


def migrate(values: Any, names: Sequence[str], maps: Mapping[str, Mapping[str, float]] | None = None,
            dropped: Sequence[str] = ()) -> np.ndarray:
    """float32[B, V]: variables saved under the list ``names`` carried to today's list by name. A variable the save
    lacks takes its default; one that is gone must be merged by ``maps`` or listed in ``dropped``
    (``variables.transfer``)."""

    return registry.remap(values, names, registry.NAMES, maps, dropped, fill=registry.DEFAULTS)


__all__ = ["INDEX", "defaults", "get", "migrate", "put", "section", "take"]
