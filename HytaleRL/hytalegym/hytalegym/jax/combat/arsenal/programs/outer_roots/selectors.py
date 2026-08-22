"""Reusable fixed-shape routing for asset-authored outer selectors.

The engine often exposes one public interaction root whose graph selects a
private child from current state.  This module owns only that structural
operation.  Weapon adapters provide graph-derived family and slot metadata;
resource predicates and lifecycle timing remain in the common Arsenal
scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class PrioritySelectorRoute:
    """Static graph metadata for one public-root priority selector.

    ``priority_slots`` are evaluated from first to last.  ``fallback_slot``
    is selected when none is available.  Hidden slots cannot be requested by
    the policy, while projected slots are reported as ``root_slot`` at public
    observation boundaries.
    """

    family_ids: tuple[int, ...]
    root_slot: int
    priority_slots: tuple[int, ...]
    fallback_slot: int
    hidden_slots: tuple[int, ...] = ()
    projected_slots: tuple[int, ...] = ()
    availability_slots: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.family_ids:
            raise ValueError("selector route requires at least one family")
        if len(set(self.family_ids)) != len(self.family_ids):
            raise ValueError("selector route family ids must be unique")
        if self.root_slot < 0 or self.fallback_slot < 0:
            raise ValueError("selector route slots must be non-negative")
        if not self.priority_slots:
            raise ValueError("selector route requires a priority slot")
        for name, slots in (
            ("priority_slots", self.priority_slots),
            ("hidden_slots", self.hidden_slots),
            ("projected_slots", self.projected_slots),
            ("availability_slots", self.availability_slots),
        ):
            if any(slot < 0 for slot in slots):
                raise ValueError(f"{name} must be non-negative")
            if len(set(slots)) != len(slots):
                raise ValueError(f"{name} must not contain duplicates")

    @property
    def required_capacity(self) -> int:
        """Return the minimum ability-axis width needed by this route."""

        slots = (
            self.root_slot,
            self.fallback_slot,
            *self.priority_slots,
            *self.hidden_slots,
            *self.projected_slots,
            *self.availability_slots,
        )
        return max(slots) + 1


def route_matches(
    family: jax.Array,
    route: PrioritySelectorRoute,
) -> jax.Array:
    """Return rows whose family owns ``route`` without asset-name branches."""

    result = jnp.zeros_like(family, dtype=jnp.bool_)
    for family_id in route.family_ids:
        result |= family == jnp.int32(family_id)
    return result


def project_policy_mask(
    mask: jax.Array,
    family: jax.Array,
    route: PrioritySelectorRoute,
) -> jax.Array:
    """Hide private selector children only for rows owning ``route``."""

    _validate_ability_axis(mask, family, route)
    matched = route_matches(family, route)
    result = mask
    for slot in route.hidden_slots:
        result = result.at[..., slot].set(
            result[..., slot] & ~matched
        )
    return result


def select_priority_child(
    predicates: tuple[jax.Array, ...],
    slots: tuple[int, ...],
    fallback_slot: int,
) -> jax.Array:
    """Select the first true predicate, otherwise ``fallback_slot``."""

    if not predicates or len(predicates) != len(slots):
        raise ValueError("predicates and slots require equal non-zero length")
    if fallback_slot < 0 or any(slot < 0 for slot in slots):
        raise ValueError("priority selector slots must be non-negative")
    shape = predicates[0].shape
    if any(predicate.shape != shape for predicate in predicates):
        raise ValueError("priority predicates must share one shape")
    selected = jnp.full(shape, fallback_slot, dtype=jnp.int32)
    for predicate, slot in reversed(tuple(zip(predicates, slots, strict=True))):
        selected = jnp.where(predicate, jnp.int32(slot), selected)
    return selected


def resolve_priority_request(
    internal_legal: jax.Array,
    policy_mask: jax.Array,
    requested_slot: jax.Array,
    family: jax.Array,
    route: PrioritySelectorRoute,
) -> tuple[jax.Array, jax.Array]:
    """Resolve a public-root request to its first legal private program."""

    _validate_ability_axis(internal_legal, family, route)
    if policy_mask.shape != internal_legal.shape:
        raise ValueError("policy_mask must match internal_legal")
    if requested_slot.shape != family.shape:
        raise ValueError("requested_slot must match the family axes")
    capacity = internal_legal.shape[-1]
    safe = jnp.clip(requested_slot, 0, capacity - 1)
    selectable = _gather(policy_mask, safe)
    selected = select_priority_child(
        tuple(internal_legal[..., slot] for slot in route.priority_slots),
        route.priority_slots,
        route.fallback_slot,
    )
    owns_request = route_matches(family, route) & (
        requested_slot == route.root_slot
    )
    return jnp.where(owns_request, selected, safe), selectable


def project_observable_slot(
    internal_slot: jax.Array,
    family: jax.Array,
    route: PrioritySelectorRoute,
    authored_root_slot: jax.Array | None = None,
) -> jax.Array:
    """Map private execution children back to one public authored root."""

    if internal_slot.shape != family.shape:
        raise ValueError("internal_slot must match the family axes")
    if authored_root_slot is None:
        authored_root_slot = jnp.full_like(internal_slot, -1)
        explicit = jnp.zeros_like(internal_slot, dtype=jnp.bool_)
    else:
        if authored_root_slot.shape != internal_slot.shape:
            raise ValueError("authored_root_slot must match internal_slot")
        explicit = authored_root_slot >= 0
    private = jnp.zeros_like(internal_slot, dtype=jnp.bool_)
    for slot in route.projected_slots:
        private |= internal_slot == jnp.int32(slot)
    derived = jnp.where(
        route_matches(family, route) & private,
        jnp.int32(route.root_slot),
        internal_slot,
    )
    return jnp.where(explicit, authored_root_slot, derived)


def project_root_legality(
    result: jax.Array,
    internal_base: jax.Array,
    interaction_available: jax.Array,
    root_admission: jax.Array,
    family: jax.Array,
    route: PrioritySelectorRoute,
) -> jax.Array:
    """Publish a root when admission and at least one child path are live."""

    _validate_ability_axis(result, family, route)
    for value, name in (
        (internal_base, "internal_base"),
        (interaction_available, "interaction_available"),
        (root_admission, "root_admission"),
    ):
        if value.shape != result.shape:
            raise ValueError(f"{name} must match result")
    candidates = route.availability_slots or (
        *route.priority_slots,
        route.fallback_slot,
    )
    available = jnp.zeros_like(family, dtype=jnp.bool_)
    for slot in candidates:
        available |= internal_base[..., slot]
    root_legal = (
        root_admission[..., route.root_slot]
        & interaction_available[..., route.root_slot]
        & available
    )
    return result.at[..., route.root_slot].set(
        jnp.where(
            route_matches(family, route),
            root_legal,
            result[..., route.root_slot],
        )
    )


def condition_route_slot(
    internal_legal: jax.Array,
    family: jax.Array,
    route: PrioritySelectorRoute,
    slot: int,
    condition: jax.Array,
) -> jax.Array:
    """Apply a graph-derived child predicate only to route-owning rows."""

    _validate_ability_axis(internal_legal, family, route)
    if slot < 0 or slot >= internal_legal.shape[-1]:
        raise ValueError("conditioned slot is outside the ability axis")
    if condition.shape != family.shape:
        raise ValueError("condition must match the family axes")
    legal = internal_legal[..., slot] & (
        ~route_matches(family, route) | condition
    )
    return internal_legal.at[..., slot].set(legal)


def _validate_ability_axis(
    value: jax.Array,
    family: jax.Array,
    route: PrioritySelectorRoute,
) -> None:
    if value.ndim != family.ndim + 1 or value.shape[:-1] != family.shape:
        raise ValueError("ability value must extend the family axes by one")
    if value.shape[-1] < route.required_capacity:
        raise ValueError("ability axis is too small for selector route")


def _gather(value: jax.Array, slot: jax.Array) -> jax.Array:
    return jnp.take_along_axis(value, slot[..., None], axis=-1)[..., 0]


__all__ = [
    "PrioritySelectorRoute",
    "condition_route_slot",
    "project_observable_slot",
    "project_policy_mask",
    "project_root_legality",
    "resolve_priority_request",
    "route_matches",
    "select_priority_child",
]
