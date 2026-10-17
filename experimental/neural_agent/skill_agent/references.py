"""skill_agent/references.py: the resolver that turns a request's target into what a neural body sees.

A request names a world point, an entity handle, or a block cell and face. This module gives the
agent-frame vector of a world point, and the index of the matching token in a body's token set, or a
rejection reason when nothing matches (``BUILD.md`` 3.1, ``AGENT_BRIEF.md`` A4).

Frame: the gym's own (``hytalegym/jax/combat/runtime/reset.py`` lines 513-516). World axes are (x, y, z)
with y up, and yaw is in radians. For an agent with yaw ``yaw``::

    forward(yaw) = (-sin yaw, 0, -cos yaw)
    right(yaw)   = ( cos yaw, 0, -sin yaw)
    up           = (0, 1, 0)

The agent-frame vector of a world point ``p`` for an agent at position ``a`` is
``(d . forward, d . right, d_y)`` with ``d = p - a``, named ``(forward, right, up)``. The three axes are
orthonormal, so the inverse is ``p = a + F * forward + R * right + U * up``. Directions rotate without
translation.

A lookup returns the index of the token whose identifier matches, ``None`` when none does, and raises
``ValueError`` when two tokens share the identifier. ``mark`` returns a copy of a token set in which
exactly one token is marked.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from typing import Any, Sequence, TypeVar

import numpy as np

from experimental.neural_agent.skill_agent.surface import BlockCandidateToken, EntityToken, InventorySlotToken

__all__ = [
    "candidate_index",
    "direction_to_agent_frame",
    "direction_to_world_frame",
    "entity_index",
    "mark",
    "resolve_candidate",
    "resolve_entity",
    "resolve_slot",
    "slot_index",
    "to_agent_frame",
    "to_world_frame",
]

_Vector = tuple[float, float, float]
_Axes = tuple[_Vector, _Vector, _Vector]
_Token = TypeVar("_Token")


def _real(value: Any, name: str) -> float:
    """Return ``value`` as a float, rejecting bools (Python or NumPy), non-real types and non-finite values."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {number}")
    return number


def _triple(value: Any, name: str) -> _Vector:
    """Return ``value`` as three finite floats: a sequence of three real numbers."""

    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        raise TypeError(f"{name} must be a sequence of 3 real numbers, got {type(value).__name__}")
    items = tuple(value)
    if len(items) != 3:
        raise ValueError(f"{name} must have 3 values, got {len(items)}")
    x, y, z = items
    return _real(x, f"{name}[0]"), _real(y, f"{name}[1]"), _real(z, f"{name}[2]")


def _axes(agent_yaw: float) -> _Axes:
    """Return the agent's (forward, right, up) axes in world coordinates for yaw ``agent_yaw``."""

    sine, cosine = math.sin(agent_yaw), math.cos(agent_yaw)
    return (-sine, 0.0, -cosine), (cosine, 0.0, -sine), (0.0, 1.0, 0.0)


def _dot(first: _Vector, second: _Vector) -> float:
    """Return the dot product of two 3-vectors."""

    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _rotate_to_agent(offset: _Vector, agent_yaw: float) -> _Vector:
    """Return the components of a world offset along (forward, right, up)."""

    forward, right, up = _axes(agent_yaw)
    return _dot(forward, offset), _dot(right, offset), _dot(up, offset)


def _rotate_to_world(agent_vector: _Vector, agent_yaw: float) -> _Vector:
    """Return ``F * forward + R * right + U * up`` for an agent-frame vector ``(F, R, U)``."""

    forward, right, up = _axes(agent_yaw)
    along_forward, along_right, along_up = agent_vector
    return (
        along_forward * forward[0] + along_right * right[0] + along_up * up[0],
        along_forward * forward[1] + along_right * right[1] + along_up * up[1],
        along_forward * forward[2] + along_right * right[2] + along_up * up[2],
    )


def to_agent_frame(world_point: Sequence[float], agent_position: Sequence[float], agent_yaw: float) -> _Vector:
    """Return the agent-frame vector ``(forward, right, up)`` of a world point.

    Args:
        world_point: The point ``p`` in world axes (x, y, z): three finite real numbers.
        agent_position: The agent's position ``a``: three finite real numbers.
        agent_yaw: The agent's yaw in radians: any finite real number.

    Returns:
        ``(d . forward, d . right, d_y)`` with ``d = p - a``.

    Raises:
        TypeError: If an argument is not a sequence of real numbers, or a value is a bool.
        ValueError: If a sequence does not have three values, or a value is not finite.
    """

    point = _triple(world_point, "world_point")
    position = _triple(agent_position, "agent_position")
    yaw = _real(agent_yaw, "agent_yaw")
    offset = (point[0] - position[0], point[1] - position[1], point[2] - position[2])
    return _rotate_to_agent(offset, yaw)


def to_world_frame(agent_vector: Sequence[float], agent_position: Sequence[float], agent_yaw: float) -> _Vector:
    """Return the world point of an agent-frame vector: the exact inverse of :func:`to_agent_frame`.

    Args:
        agent_vector: ``(F, R, U)`` in the agent frame: three finite real numbers.
        agent_position: The agent's position ``a``: three finite real numbers.
        agent_yaw: The agent's yaw in radians: any finite real number.

    Returns:
        ``a + F * forward + R * right + U * up``.

    Raises:
        TypeError: If an argument is not a sequence of real numbers, or a value is a bool.
        ValueError: If a sequence does not have three values, or a value is not finite.
    """

    vector = _triple(agent_vector, "agent_vector")
    position = _triple(agent_position, "agent_position")
    yaw = _real(agent_yaw, "agent_yaw")
    offset = _rotate_to_world(vector, yaw)
    return offset[0] + position[0], offset[1] + position[1], offset[2] + position[2]


def direction_to_agent_frame(world_direction: Sequence[float], agent_yaw: float) -> _Vector:
    """Return the agent-frame components of a world direction. Rotation only: no translation.

    Args:
        world_direction: A direction in world axes: three finite real numbers.
        agent_yaw: The agent's yaw in radians: any finite real number.

    Returns:
        ``(d . forward, d . right, d_y)`` for the direction ``d``.

    Raises:
        TypeError: If an argument is not a sequence of real numbers, or a value is a bool.
        ValueError: If a sequence does not have three values, or a value is not finite.
    """

    direction = _triple(world_direction, "world_direction")
    return _rotate_to_agent(direction, _real(agent_yaw, "agent_yaw"))


def direction_to_world_frame(agent_direction: Sequence[float], agent_yaw: float) -> _Vector:
    """Return the world direction of an agent-frame direction. Rotation only: no translation.

    Args:
        agent_direction: ``(forward, right, up)`` components: three finite real numbers.
        agent_yaw: The agent's yaw in radians: any finite real number.

    Returns:
        The world direction ``F * forward + R * right + U * up``.

    Raises:
        TypeError: If an argument is not a sequence of real numbers, or a value is a bool.
        ValueError: If a sequence does not have three values, or a value is not finite.
    """

    direction = _triple(agent_direction, "agent_direction")
    return _rotate_to_world(direction, _real(agent_yaw, "agent_yaw"))


def _single_index(matches: list[int], description: str) -> int | None:
    """Return the only matching index, ``None`` when there is none, and raise when there are several."""

    if len(matches) > 1:
        raise ValueError(f"{description} is shared by {len(matches)} tokens; it must be unique")
    return matches[0] if matches else None


def entity_index(handle: int, entity_tokens: Sequence[EntityToken]) -> int | None:
    """Return the index of the entity token with ``handle``, or ``None`` when no token has it.

    Args:
        handle: The entity handle a request names.
        entity_tokens: The body's entity tokens.

    Returns:
        The token index, or ``None``.

    Raises:
        ValueError: If two tokens share the handle.
    """

    matches = [index for index, token in enumerate(entity_tokens) if token.handle == handle]
    return _single_index(matches, f"entity handle {handle}")


def candidate_index(cell: Sequence[int], face: str, block_tokens: Sequence[BlockCandidateToken]) -> int | None:
    """Return the index of the block candidate at ``cell`` on ``face``, or ``None`` when there is none.

    Args:
        cell: The block's world cell (x, y, z), matched exactly as ``tuple(cell)``.
        face: The face the request names, one of ``surface.FACES``.
        block_tokens: The body's block candidate tokens.

    Returns:
        The token index, or ``None``.

    Raises:
        ValueError: If two candidates share the cell and face.
    """

    key = tuple(cell)
    matches = [index for index, token in enumerate(block_tokens) if token.cell == key and token.face == face]
    return _single_index(matches, f"block candidate at cell {key} face {face}")


def slot_index(slot: int, slot_tokens: Sequence[InventorySlotToken]) -> int | None:
    """Return the index of the inventory slot token for ``slot``, or ``None`` when there is none.

    Args:
        slot: The inventory slot number a request names.
        slot_tokens: The body's inventory slot tokens.

    Returns:
        The token index, or ``None``.

    Raises:
        ValueError: If two tokens share the slot number.
    """

    matches = [index for index, token in enumerate(slot_tokens) if token.slot == slot]
    return _single_index(matches, f"inventory slot {slot}")


def _require_marked(token: Any) -> None:
    """Raise ``TypeError`` unless ``token`` is a dataclass with a ``marked`` field."""

    names = {field.name for field in dataclasses.fields(token)} if dataclasses.is_dataclass(token) else set()
    if "marked" not in names:
        raise TypeError(f"{type(token).__name__} has no 'marked' field")


def mark(tokens: Sequence[_Token], index: int) -> tuple[_Token, ...]:
    """Return a copy of ``tokens`` in which token ``index`` is marked and every other token is not.

    Args:
        tokens: A token set; every item is a dataclass with a ``marked`` field.
        index: The position of the instructed target, in ``[0, len(tokens))``.

    Returns:
        A new tuple built with ``dataclasses.replace``; ``tokens`` itself is not changed.

    Raises:
        TypeError: If ``index`` is not an integer, or an item has no ``marked`` field.
        IndexError: If ``index`` is outside ``[0, len(tokens))``.
    """

    if isinstance(index, bool) or not isinstance(index, numbers.Integral):
        raise TypeError(f"index must be an integer, got {type(index).__name__}")
    target = int(index)
    if not 0 <= target < len(tokens):
        raise IndexError(f"index {target} is outside [0, {len(tokens)})")
    for token in tokens:
        _require_marked(token)
    return tuple(dataclasses.replace(token, marked=position == target) for position, token in enumerate(tokens))


def resolve_entity(handle: int, entity_tokens: Sequence[EntityToken]) -> tuple[int | None, str | None]:
    """Return ``(index, None)`` for a known entity handle, else ``(None, reason)``.

    Args:
        handle: The entity handle a request names.
        entity_tokens: The body's entity tokens.

    Returns:
        ``(index, None)`` when found, or ``(None, "unknown entity handle <handle>")``.

    Raises:
        ValueError: If two tokens share the handle.
    """

    index = entity_index(handle, entity_tokens)
    if index is None:
        return None, f"unknown entity handle {handle}"
    return index, None


def resolve_candidate(
    cell: Sequence[int], face: str, block_tokens: Sequence[BlockCandidateToken]
) -> tuple[int | None, str | None]:
    """Return ``(index, None)`` for a known block candidate, else ``(None, reason)``.

    Args:
        cell: The block's world cell (x, y, z).
        face: The face the request names.
        block_tokens: The body's block candidate tokens.

    Returns:
        ``(index, None)`` when found, or ``(None, "no block candidate at cell <cell> face <face>")``.

    Raises:
        ValueError: If two candidates share the cell and face.
    """

    index = candidate_index(cell, face, block_tokens)
    if index is None:
        return None, f"no block candidate at cell {tuple(cell)} face {face}"
    return index, None


def resolve_slot(slot: int, slot_tokens: Sequence[InventorySlotToken]) -> tuple[int | None, str | None]:
    """Return ``(index, None)`` for a known inventory slot, else ``(None, reason)``.

    Args:
        slot: The inventory slot number a request names.
        slot_tokens: The body's inventory slot tokens.

    Returns:
        ``(index, None)`` when found, or ``(None, "no inventory slot <slot>")``.

    Raises:
        ValueError: If two tokens share the slot number.
    """

    index = slot_index(slot, slot_tokens)
    if index is None:
        return None, f"no inventory slot {slot}"
    return index, None
