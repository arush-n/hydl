"""Find published observation columns that never move.

A named, correctly-shaped column that is always the same value is worse than a
missing one, because it looks like data. ``combat_f32.target_phase_*`` is
exactly that on the v3 arsenal path: the v1 fields are published and never
populated, so anything reading target attack phase from an arsenal scene reads
a default and cannot tell.

This is a **detector, not a verdict**. Constant is not the same as broken:

* a short run legitimately leaves columns flat -- an ability never used, a
  hazard never entered, a resource the loadout does not have;
* a column can be constant *because the agent never did the thing*, which is a
  fact about the policy, not the environment.

So the output is a list to explain, not a list of bugs. What it does give you
is the cheap half of the question: a column that never moves across a long,
active run is either dead or untested, and both are worth knowing before you
train on it.

Pair with :mod:`adk.probes.liveness`: liveness says whether the *run* was
vacuous, this says whether a *column* was.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp

from adk.policy.fields import GROUP_FEATURES
from adk.policy.observations import (
    AUXILIARY_GROUPS,
    SET_GROUPS,
    VECTOR_GROUPS,
    observation_auxiliaries,
    observation_sets,
    observation_vectors,
)


def _column_is_constant(column: jax.Array) -> bool:
    """True when every element equals the first, NaN-tolerantly."""

    flat = jnp.ravel(column)
    if flat.size == 0:
        return True
    first = flat[0]
    if jnp.issubdtype(flat.dtype, jnp.floating):
        both_nan = jnp.isnan(flat) & jnp.isnan(first)
        return bool(jnp.all((flat == first) | both_nan))
    return bool(flat.size == 1 or jnp.all(flat == first))


def column_activity(
    recorded: Mapping[str, Any],
) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Split each recorded group into (constant, moving) feature names.

    ``recorded`` maps a group name from :data:`GROUP_FEATURES` to an array whose
    **last axis is the feature axis**; every leading axis (steps, batch, tokens)
    is pooled, so a column counts as moving if it varies anywhere in the run.

    HOST-SIDE: reads concrete values and therefore syncs. Not traceable.
    """

    activity: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for group, array in recorded.items():
        names = GROUP_FEATURES.get(group)
        if names is None:
            raise KeyError(
                f"unknown observation group {group!r}; expected one of "
                f"{sorted(GROUP_FEATURES)}"
            )
        values = jnp.asarray(array)
        if values.ndim == 0 or values.shape[-1] != len(names):
            raise ValueError(
                f"group {group!r} has feature axis {values.shape[-1:]!r}, "
                f"expected {(len(names),)!r} to match its {len(names)} names"
            )
        constant = tuple(
            name
            for index, name in enumerate(names)
            if _column_is_constant(values[..., index])
        )
        moving = tuple(name for name in names if name not in set(constant))
        activity[group] = (constant, moving)
    return activity


def constant_columns(recorded: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Group -> feature names that never changed. Groups with none are omitted."""

    return {
        group: constant
        for group, (constant, _moving) in column_activity(recorded).items()
        if constant
    }


def observation_groups(legal_observation: Any) -> dict[str, jax.Array]:
    """Recordable vector groups, re-keyed to match :data:`GROUP_FEATURES`.

    ``observation_vectors`` keys groups by short name (``combat``) while the
    feature tables key them by array name (``combat_f32``). Re-key here so a
    caller never has to know both spellings, and fail loudly rather than
    silently dropping a group if that correspondence ever changes upstream.

    Returns the seven fixed-width **float** vector groups and nothing else --
    not the integer columns, not the availability masks, not the set-structured
    groups. Use :func:`all_observation_groups` for the full named schema; a
    column audit run through this function alone is blind to 64 of the 241
    named columns.

    TRACED: safe to call inside a compiled record function.
    """

    groups: dict[str, jax.Array] = {}
    for short_name, array in observation_vectors(legal_observation).items():
        name = f"{short_name}_f32"
        if name not in GROUP_FEATURES:
            raise KeyError(
                f"observation group {short_name!r} maps to {name!r}, which is "
                "not a GROUP_FEATURES key; the upstream naming changed"
            )
        groups[name] = array
    return groups


#: Group names :func:`all_observation_groups` can return, derived from the three
#: reader tables so it cannot drift from them. Every other ``GROUP_FEATURES``
#: name is scene-dependent and may simply not be on the observation tree, so a
#: caller must not promise it -- notably ``inventory_token_f32`` and
#: ``light_token_f32``, which depend on the providers the scene enabled.
READABLE_GROUPS: frozenset[str] = frozenset(
    {f"{name}_f32" for name in VECTOR_GROUPS}
    | {f"{name}_f32" for name in SET_GROUPS}
    | set(AUXILIARY_GROUPS)
) & frozenset(GROUP_FEATURES)


def all_observation_groups(legal_observation: Any) -> dict[str, jax.Array]:
    """Every named group a reader can reach: float vectors, sets, integers, masks.

    :func:`observation_groups` covers the float vector groups only. The set
    groups carry their own occupancy mask, and the integer and availability-mask
    groups have no reader of their own at all -- so an audit built on
    ``observation_groups`` cannot see 64 of the 241 named columns, including
    ``resource_mask``, the only column that separates a real zero from a stat
    the family does not have.

    ``inventory_token_f32`` and ``light_token_f32`` are named in
    ``GROUP_FEATURES`` but are not on the observation tree for every scene, so
    they are absent here rather than raising: which of them appear depends on
    the providers the scene enabled.

    TRACED: safe to call inside a compiled record function.
    """

    groups = observation_groups(legal_observation)
    for name, tokens in observation_sets(legal_observation).items():
        field = f"{name}_f32"
        if field in GROUP_FEATURES:
            groups[field] = tokens.values
    groups.update(observation_auxiliaries(legal_observation))
    return groups


def describe_columns(recorded: Mapping[str, Any]) -> str:
    """Readable constant/moving breakdown, most-suspicious group first."""

    activity = column_activity(recorded)
    if not activity:
        return "no observation groups recorded"
    lines = ["group                     constant  moving  never-moved"]
    ordered = sorted(
        activity.items(),
        key=lambda item: (-len(item[1][0]), item[0]),
    )
    for group, (constant, moving) in ordered:
        preview = ", ".join(constant[:4])
        if len(constant) > 4:
            preview += f", +{len(constant) - 4} more"
        lines.append(
            f"{group:24s} {len(constant):8d}  {len(moving):6d}  {preview}"
        )
    return "\n".join(lines)


__all__ = [
    "READABLE_GROUPS",
    "all_observation_groups",
    "column_activity",
    "constant_columns",
    "describe_columns",
    "observation_groups",
]
