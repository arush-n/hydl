"""Names for the action masks, and which actions are permanently unavailable.

The observation ships seven separate legality masks.  Their widths are the only
clue to what they mean, and two of them are actively misleading:

* ``base.action_mask`` is 12 wide -- nine skills followed by three door intents
  (``encoder.py:658`` concatenates ``skill_mask`` and ``door_mask``).  It is
  *not* the eight-wide low-level ``ACTION_*`` space used by the native
  transport, which is a different contract entirely.
* ``dodge_action_mask`` is 4 wide, but the Gym ANDs it with
  ``DODGE_AUTHORED_ACTION_MASK == (False, False, True, True)``
  (``encoder.py:405``), so *forward and back dodges can never be legal*.  Only
  the authored ``Dodge_Left``/``Dodge_Right`` branches carry the effect, force,
  and stamina transaction.  An agent that treats the width as the action count
  spends two of its four dodge logits on actions that never fire.

Every name here is read from the Gym's own constants, so a renumbering upstream
moves these names with it rather than silently mislabelling a column.

Nothing in this module synchronizes with the device except the functions marked
HOST-SIDE in their docstrings.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.observation import (
    DOOR_INTENT_NAMES,
    LEARNER_ACTION_COUNT,
    LEARNER_ACTION_NAMES,
)
from hytalegym.jax.combat.mechanics import DODGE_AUTHORED_ACTION_MASK
from hytalegym.jax.combat.skills import SKILL_COUNT


#: Dodge directions in policy-mask order, excluding ``DODGE_NONE``.
#: Stated verbatim by ``mechanics/contract.py:78-79``.
DODGE_ACTION_NAMES: tuple[str, ...] = ("forward", "back", "left", "right")

#: Which dodge directions the assets actually authored.  ``False`` means the
#: mask is hard-wired off, not merely unavailable in the current state.
DODGE_AUTHORED: tuple[bool, ...] = tuple(bool(x) for x in DODGE_AUTHORED_ACTION_MASK)

#: Mask leaf name -> the name of each column along its trailing axis.
#: Scalar masks (one bool per environment) map to a single name.
ACTION_MASK_FEATURES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "action_mask": tuple(LEARNER_ACTION_NAMES),
        "skill_action_mask": tuple(LEARNER_ACTION_NAMES[:SKILL_COUNT]),
        "dodge_action_mask": DODGE_ACTION_NAMES,
        "door_action_mask": tuple(DOOR_INTENT_NAMES),
        "guard_action_mask": ("guard",),
        "jump_action_mask": ("jump",),
    }
)

_INDEX: Mapping[str, Mapping[str, int]] = MappingProxyType(
    {
        mask: MappingProxyType({name: i for i, name in enumerate(names)})
        for mask, names in ACTION_MASK_FEATURES.items()
    }
)

if len(LEARNER_ACTION_NAMES) != LEARNER_ACTION_COUNT:
    raise RuntimeError("learner action names disagree with LEARNER_ACTION_COUNT")
if len(DODGE_AUTHORED) != len(DODGE_ACTION_NAMES):
    raise RuntimeError(
        "DODGE_AUTHORED_ACTION_MASK width moved away from the four policy-mask "
        f"directions: {DODGE_AUTHORED}"
    )


def action_names(mask: str) -> tuple[str, ...]:
    """Column names for one action mask, in array order."""

    try:
        return ACTION_MASK_FEATURES[mask]
    except KeyError as error:
        raise KeyError(
            f"unknown action mask {mask!r}; known: {sorted(ACTION_MASK_FEATURES)}"
        ) from error


def action_index(mask: str, name: str) -> int:
    """Column index of one named action.  O(1), no device work."""

    columns = _INDEX.get(mask)
    if columns is None:
        raise KeyError(
            f"unknown action mask {mask!r}; known: {sorted(ACTION_MASK_FEATURES)}"
        )
    try:
        return columns[name]
    except KeyError as error:
        raise KeyError(
            f"{mask!r} has no action {name!r}; available: {action_names(mask)}"
        ) from error


def action(values: Any, mask: str, name: str) -> jax.Array:
    """One named action's legality as an O(1) slice of the trailing axis."""

    array = getattr(values, "values", values)
    return array[..., action_index(mask, name)]


def unreachable_actions() -> Mapping[str, tuple[str, ...]]:
    """Actions whose mask is hard-wired false regardless of state.

    Derived from the Gym's authored-mask constants, not from a sample, so it
    stays true for every scene.  Exclude these from an action space rather than
    learning to avoid them.
    """

    return MappingProxyType(
        {
            "dodge_action_mask": tuple(
                name
                for name, authored in zip(DODGE_ACTION_NAMES, DODGE_AUTHORED)
                if not authored
            )
        }
    )


def legality(observation: Any) -> dict[str, jax.Array]:
    """Every action mask on an observation, keyed ``mask.action``.

    TRACED: returns arrays, safe inside ``jit``/``scan``.  Scalar masks keep
    their ``(batch,)`` shape; per-slot masks such as ``door_action_mask`` keep
    their slot axis so the caller can pick a target.
    """

    out: dict[str, jax.Array] = {}
    for mask, names in ACTION_MASK_FEATURES.items():
        holder = observation.base if mask == "action_mask" else observation
        array = getattr(holder, mask, None)
        if array is None:
            continue
        if len(names) == 1 and array.ndim == 1:
            out[f"{mask}.{names[0]}"] = array
            continue
        for index, name in enumerate(names):
            out[f"{mask}.{name}"] = array[..., index]
    return out


def legal_action_count(observation: Any) -> jax.Array:
    """How many of the 12 learner actions are legal, per environment.

    TRACED.  A cheap health signal for a rollout: if this collapses to one, the
    agent has been reduced to ``idle`` and nothing it learns will matter.
    """

    return jnp.sum(observation.base.action_mask, axis=-1)


def describe_legality(observation: Any, row: int = 0) -> dict[str, bool]:
    """HOST-SIDE: one environment's action legality as Python bools.

    Synchronizes with the device.  For printing and debugging, never a rollout.
    """

    return {
        key: bool(jnp.asarray(value)[row].any())
        for key, value in legality(observation).items()
    }


__all__ = [
    "ACTION_MASK_FEATURES",
    "DODGE_ACTION_NAMES",
    "DODGE_AUTHORED",
    "action",
    "action_index",
    "action_names",
    "describe_legality",
    "legal_action_count",
    "legality",
    "unreachable_actions",
]
