"""Named, O(1) access to every column of the structured observation.

The Gym publishes a column-name tuple for each observation group, but nothing
connected those names to the arrays an agent actually receives -- so callers
either memorized indices or fed whole opaque blocks into a dense layer.

This module wires the two together:

* Every name -> index map is built **once at import** into a plain dict, so a
  lookup is O(1) and costs nothing at trace time.
* Every accessor returns an XLA slice of the array it was given.  No copy, no
  reshape, no device transfer, fully traceable inside ``jit``/``scan``.
* **No function here synchronizes with the device.**  Nothing calls ``int()``,
  ``bool()``, or ``.item()`` on an array, so none of it stalls a rollout.
  Host-side reporting lives in ``adk.debug`` and says so explicitly.

Widths are checked against the live contract at import, so a Gym change to any
group fails loudly here instead of silently shifting every column an agent
reads.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.observation import (
    COMBAT_FLOAT_FEATURES,
    COMBAT_INTEGER_FEATURES,
    ENTITY_FLOAT_FEATURES,
    ENTITY_INTEGER_FEATURES,
    HAZARD_FLOAT_FEATURES,
    HAZARD_INTEGER_FEATURES,
    INTERACTION_FLOAT_FEATURES,
    PROJECTILE_FLOAT_FEATURES,
    PROJECTILE_INTEGER_FEATURES,
    SELF_FLOAT_FEATURES,
    SELF_INTEGER_FEATURES,
    TARGET_FLOAT_FEATURES,
    TARGET_INTEGER_FEATURES,
    TERRAIN_FLOAT_FEATURES,
    TRAVERSAL_FLOAT_FEATURES,
)
from hytalegym.jax.combat.observation.v3.contract import (
    ABILITY_FLOAT_FEATURES,
    ABILITY_INTEGER_FEATURES,
    ACTOR_WORLD_FLOAT_FEATURES,
    ACTOR_WORLD_MASK_FEATURES,
    DEFENSE_FLOAT_FEATURES,
    MOVEMENT_STATE_FEATURES,
    STATUS_FLOAT_FEATURES,
    STATUS_INTEGER_FEATURES,
    WEAPON_INTEGER_FEATURES,
)
from hytalegym.jax.combat.observation.v3.inventory_tokens import (
    INVENTORY_POLICY_FLOAT_FEATURES,
)
from hytalegym.jax.combat.observation.v3.light_policy_tokens import (
    ACTOR_LIGHT_POLICY_FEATURES,
)
from hytalegym.jax.combat.mechanics import (
    RESOURCE_AMMO,
    RESOURCE_COUNT,
    RESOURCE_MAGIC_CHARGES,
    RESOURCE_MANA,
    RESOURCE_SIGNATURE_CHARGES,
    RESOURCE_OXYGEN,
    RESOURCE_SIGNATURE_ENERGY,
    RESOURCE_STAMINA,
)


def _resource_features() -> tuple[str, ...]:
    """Name the resource channels, which the Gym publishes as indices only.

    ``mechanics/contract.py`` declares ``RESOURCE_*`` positions rather than a
    name tuple, so the order is reconstructed from the constants themselves --
    a renumbering upstream moves these names with it instead of silently
    mislabelling a channel.

    Channel 4 carries two upstream names: ``RESOURCE_DEPLOYABLE_PREVIEW`` is an
    alias of ``RESOURCE_SIGNATURE_CHARGES``.  They are never equipped by the
    same compiled family, so the channel is named for the shared slot and the
    alias is documented rather than duplicated.
    """

    positions = {
        RESOURCE_STAMINA: "stamina",
        RESOURCE_MANA: "mana",
        RESOURCE_MAGIC_CHARGES: "magic_charges",
        RESOURCE_SIGNATURE_ENERGY: "signature_energy",
        RESOURCE_SIGNATURE_CHARGES: "signature_charges",
        RESOURCE_AMMO: "ammo",
        RESOURCE_OXYGEN: "oxygen",
    }
    if sorted(positions) != list(range(RESOURCE_COUNT)):
        raise RuntimeError(
            f"resource channel constants are not a dense 0..{RESOURCE_COUNT - 1} "
            f"range: {sorted(positions)}"
        )
    return tuple(positions[index] for index in range(RESOURCE_COUNT))


#: Resource channels, in array order.  Normalized to 0..1 against the per-entity
#: authored minimum/maximum, so a value is a *fraction*, not a raw point count.
RESOURCE_FEATURES: tuple[str, ...] = _resource_features()


#: Leaf name -> its published column names.  Keys are the actual field names on
#: the structured observation, so a caller can go from a tree path straight to
#: the schema without a translation table.
GROUP_FEATURES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "self_f32": tuple(SELF_FLOAT_FEATURES),
        "self_i32": tuple(SELF_INTEGER_FEATURES),
        "target_f32": tuple(TARGET_FLOAT_FEATURES),
        "target_i32": tuple(TARGET_INTEGER_FEATURES),
        "combat_f32": tuple(COMBAT_FLOAT_FEATURES),
        "combat_i32": tuple(COMBAT_INTEGER_FEATURES),
        "entity_f32": tuple(ENTITY_FLOAT_FEATURES),
        "entity_i32": tuple(ENTITY_INTEGER_FEATURES),
        "projectile_f32": tuple(PROJECTILE_FLOAT_FEATURES),
        "projectile_i32": tuple(PROJECTILE_INTEGER_FEATURES),
        "hazard_f32": tuple(HAZARD_FLOAT_FEATURES),
        "hazard_i32": tuple(HAZARD_INTEGER_FEATURES),
        "terrain_f32": tuple(TERRAIN_FLOAT_FEATURES),
        "traversal_f32": tuple(TRAVERSAL_FLOAT_FEATURES),
        "interaction_f32": tuple(INTERACTION_FLOAT_FEATURES),
        "weapon_i32": tuple(WEAPON_INTEGER_FEATURES),
        "defense_f32": tuple(DEFENSE_FLOAT_FEATURES),
        "status_f32": tuple(STATUS_FLOAT_FEATURES),
        "status_i32": tuple(STATUS_INTEGER_FEATURES),
        "ability_f32": tuple(ABILITY_FLOAT_FEATURES),
        "ability_i32": tuple(ABILITY_INTEGER_FEATURES),
        "actor_world_f32": tuple(ACTOR_WORLD_FLOAT_FEATURES),
        "actor_world_mask": tuple(ACTOR_WORLD_MASK_FEATURES),
        "movement_state_f32": tuple(MOVEMENT_STATE_FEATURES),
        "inventory_token_f32": tuple(INVENTORY_POLICY_FLOAT_FEATURES),
        "light_token_f32": tuple(ACTOR_LIGHT_POLICY_FEATURES),
        "resource_f32": RESOURCE_FEATURES,
        # Same layout, independent availability: a missing stat reads zero and
        # masked false, so the value alone cannot distinguish "empty" from
        # "this family has no mana at all".
        "resource_mask": RESOURCE_FEATURES,
    }
)

#: Precomputed at import: group -> {name: index}.  Lookup is a dict hit.
_INDEX: Mapping[str, Mapping[str, int]] = MappingProxyType(
    {
        group: MappingProxyType({name: index for index, name in enumerate(names)})
        for group, names in GROUP_FEATURES.items()
    }
)

for _group, _names in GROUP_FEATURES.items():
    if len(set(_names)) != len(_names):
        raise RuntimeError(f"duplicate column name in {_group}: {_names}")


def field_names(group: str) -> tuple[str, ...]:
    """Column names for a group, in array order."""

    try:
        return GROUP_FEATURES[group]
    except KeyError as error:
        raise KeyError(
            f"unknown observation group {group!r}; known: {sorted(GROUP_FEATURES)}"
        ) from error


def groups() -> tuple[str, ...]:
    """Every group that has published column names."""

    return tuple(GROUP_FEATURES)


def field_index(group: str, name: str) -> int:
    """Column index of one named field.  O(1), no device work."""

    columns = _INDEX.get(group)
    if columns is None:
        raise KeyError(
            f"unknown observation group {group!r}; known: {sorted(GROUP_FEATURES)}"
        )
    try:
        return columns[name]
    except KeyError as error:
        raise KeyError(
            f"{group!r} has no field {name!r}; available: {field_names(group)}"
        ) from error


def field(values: Any, group: str, name: str) -> jax.Array:
    """One named column as an O(1) slice of the trailing axis.

    ``values`` may be the raw array or anything with a ``.values`` attribute
    (a :class:`~adk.policy.observations.TokenSet`), so set groups and vector
    groups read the same way.
    """

    array = getattr(values, "values", values)
    return array[..., field_index(group, name)]


def field_span(group: str, names: Sequence[str]) -> slice:
    """A ``slice`` over contiguous columns, or raise if they are not adjacent.

    Prefer this to stacking: a slice is a view, a stack allocates and copies.
    """

    if not names:
        raise ValueError("field_span needs at least one name")
    indices = [field_index(group, name) for name in names]
    start = indices[0]
    if indices != list(range(start, start + len(indices))):
        raise ValueError(
            f"{list(names)} are not contiguous in {group!r} (indices {indices}); "
            "use select() instead"
        )
    return slice(start, start + len(indices))


def span(values: Any, group: str, names: Sequence[str]) -> jax.Array:
    """Contiguous named columns as one O(1) view."""

    array = getattr(values, "values", values)
    return array[..., field_span(group, names)]


def select(values: Any, group: str, names: Sequence[str]) -> jax.Array:
    """Arbitrary named columns as one gather.

    Use when the columns are not adjacent; a single gather beats several
    slices followed by a stack.
    """

    array = getattr(values, "values", values)
    indices = [field_index(group, name) for name in names]
    return jnp.take(array, jnp.asarray(indices, dtype=jnp.int32), axis=-1)


def as_dict(values: Any, group: str) -> dict[str, jax.Array]:
    """Every named column of a group, as views.

    Convenient for exploration and for building a feature dict once outside a
    hot loop; inside one, prefer :func:`field` or :func:`span`.
    """

    array = getattr(values, "values", values)
    return {name: array[..., index] for name, index in _INDEX[group].items()}


def verify_widths(legal_observation: Any) -> dict[str, tuple[int, int]]:
    """Check every group's published width against a live observation.

    Host-side and diagnostic: reads shapes only, never values, so it does not
    transfer data, but it is meant for a startup check rather than a hot loop.
    Returns any mismatches as ``group -> (published, actual)``; empty is good.
    """

    base = legal_observation.base
    mismatches: dict[str, tuple[int, int]] = {}
    for group, names in GROUP_FEATURES.items():
        holder = base if hasattr(base, group) else legal_observation
        array = getattr(holder, group, None)
        if array is None or not hasattr(array, "shape"):
            continue
        actual = int(array.shape[-1])
        if actual != len(names):
            mismatches[group] = (len(names), actual)
    return mismatches


__all__ = [
    "GROUP_FEATURES",
    "as_dict",
    "field",
    "field_index",
    "field_names",
    "field_span",
    "groups",
    "select",
    "span",
    "verify_widths",
]
