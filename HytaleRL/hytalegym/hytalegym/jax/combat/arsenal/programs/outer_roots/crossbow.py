"""Crossbow outer-Primary routing resolved from the 0.5.7 item graph."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.profiles.hytale_0_5_7 import (
    CROSSBOW_FAMILY_IDS,
)
from hytalegym.jax.combat.arsenal.programs.outer_roots.selectors import (
    PrioritySelectorRoute,
    condition_route_slot,
    project_observable_slot,
    project_policy_mask,
    project_root_legality,
    resolve_priority_request,
    route_matches,
    select_priority_child,
)
from hytalegym.jax.combat.arsenal.schema.types import AbilityLoadout
from hytalegym.jax.combat.mechanics import (
    CombatMechanicsState,
    RESOURCE_AMMO,
    RESOURCE_SIGNATURE_CHARGES,
)


# Stable authored slots in the Crossbow profile. Slots 1 and 3 are internal
# children: the target-local combo graph and Primary's Big Arrow selection.
CROSSBOW_PRIMARY_SLOT = 0
CROSSBOW_COMBO_CHILD_SLOT = 1
CROSSBOW_SIGNATURE_ACTIVATE_SLOT = 2
CROSSBOW_BIG_ARROW_CHILD_SLOT = 3
CROSSBOW_RELOAD_SLOT = 4

_CROSSBOW_PRIMARY_ROUTE = PrioritySelectorRoute(
    family_ids=CROSSBOW_FAMILY_IDS,
    root_slot=CROSSBOW_PRIMARY_SLOT,
    priority_slots=(
        CROSSBOW_BIG_ARROW_CHILD_SLOT,
        CROSSBOW_PRIMARY_SLOT,
    ),
    fallback_slot=CROSSBOW_RELOAD_SLOT,
    hidden_slots=(
        CROSSBOW_COMBO_CHILD_SLOT,
        CROSSBOW_BIG_ARROW_CHILD_SLOT,
    ),
    projected_slots=(
        CROSSBOW_COMBO_CHILD_SLOT,
        CROSSBOW_BIG_ARROW_CHILD_SLOT,
    ),
    availability_slots=(
        CROSSBOW_BIG_ARROW_CHILD_SLOT,
        CROSSBOW_PRIMARY_SLOT,
        CROSSBOW_RELOAD_SLOT,
    ),
)


def policy_ability_mask(loadout: AbilityLoadout) -> jax.Array:
    """Hide server-selected Crossbow children from the policy action surface."""

    if loadout.ability_mask.shape[2] < _CROSSBOW_PRIMARY_ROUTE.required_capacity:
        return loadout.ability_mask
    return project_policy_mask(
        loadout.ability_mask,
        loadout.weapon_family,
        _CROSSBOW_PRIMARY_ROUTE,
    )


def project_legal_mask(
    loadout: AbilityLoadout,
    internal_legal: jax.Array,
    internal_base: jax.Array | None = None,
    interaction_available: jax.Array | None = None,
    root_admission: jax.Array | None = None,
) -> jax.Array:
    """Project legal child programs onto the authored outer-root controls."""

    if internal_legal.shape != loadout.ability_mask.shape:
        raise ValueError("internal_legal and loadout ability axes must match")
    if internal_base is None:
        internal_base = internal_legal
    if interaction_available is None:
        interaction_available = jnp.ones_like(internal_legal)
    if root_admission is None:
        root_admission = jnp.ones_like(internal_legal)
    for value, name in (
        (internal_base, "internal_base"),
        (interaction_available, "interaction_available"),
        (root_admission, "root_admission"),
    ):
        if value.shape != internal_legal.shape:
            raise ValueError(f"{name} must match the loadout ability axes")
    result = internal_legal & policy_ability_mask(loadout)
    if internal_legal.shape[2] <= CROSSBOW_RELOAD_SLOT:
        return result
    return project_root_legality(
        result,
        internal_base,
        interaction_available,
        root_admission,
        loadout.weapon_family,
        _CROSSBOW_PRIMARY_ROUTE,
    )


def project_observable_active_slot(
    loadout: AbilityLoadout,
    internal_slot: jax.Array,
    authored_root_slot: jax.Array | None = None,
) -> jax.Array:
    """Map scheduler-only children back to the native authored root slot.

    The Crossbow Primary selector runs standard or Big Arrow children
    internally, while ``InteractionManager`` continues to report the equipped
    Primary root.  The same mapping is ready for the private combo child once
    its target-status progression is supplied.  Keep child identity private to
    execution so learner rows and public transition info use the same authored
    slot as native evidence.
    """

    if internal_slot.shape != loadout.weapon_family.shape:
        raise ValueError("internal_slot must match the loadout batch/entity axes")
    return project_observable_slot(
        internal_slot,
        loadout.weapon_family,
        _CROSSBOW_PRIMARY_ROUTE,
        authored_root_slot,
    )


def resolve_observed_execution_slot(
    loadout: AbilityLoadout,
    authored_root_slot: jax.Array,
    resources: jax.Array,
    resource_available: jax.Array | None = None,
) -> jax.Array:
    """Recover a selected child from an admitted native outer root.

    Native actor evidence publishes the authored root, not the child currently
    executing beneath it.  The Crossbow Primary selector is deterministic at
    admission: a full SignatureCharges stat selects Big Arrow, otherwise any
    loaded Ammo selects the standard shot, otherwise it selects reload.  A
    native adapter can therefore retain this result at the accepted edge
    without inventing an item-specific branch in its observation assembler.
    """

    if authored_root_slot.shape != loadout.weapon_family.shape:
        raise ValueError(
            "authored_root_slot must match the loadout batch/entity axes"
        )
    if resources.shape[:2] != authored_root_slot.shape:
        raise ValueError("resources must share the loadout batch/entity axes")
    if resources.shape[2] <= max(RESOURCE_AMMO, RESOURCE_SIGNATURE_CHARGES):
        raise ValueError("resources do not contain the Crossbow selector stats")
    if resource_available is None:
        resource_available = jnp.ones_like(resources, dtype=jnp.bool_)
    elif resource_available.shape != resources.shape:
        raise ValueError("resource_available must match resources")
    crossbow_primary = (
        route_matches(loadout.weapon_family, _CROSSBOW_PRIMARY_ROUTE)
        & (authored_root_slot == CROSSBOW_PRIMARY_SLOT)
    )
    signature_ready = (
        resources[..., RESOURCE_SIGNATURE_CHARGES] + jnp.float32(1.0e-6)
        >= loadout.resource_maximum[..., RESOURCE_SIGNATURE_CHARGES]
    )
    loaded = resources[..., RESOURCE_AMMO] >= jnp.float32(1.0)
    selector_available = (
        resource_available[..., RESOURCE_AMMO]
        & resource_available[..., RESOURCE_SIGNATURE_CHARGES]
    )
    selected = select_priority_child(
        (signature_ready, loaded),
        _CROSSBOW_PRIMARY_ROUTE.priority_slots,
        _CROSSBOW_PRIMARY_ROUTE.fallback_slot,
    )
    return jnp.where(
        crossbow_primary & ~selector_available,
        jnp.int32(-1),
        jnp.where(crossbow_primary, selected, authored_root_slot),
    )


def condition_internal_legality(
    loadout: AbilityLoadout,
    mechanics: CombatMechanicsState,
    internal_legal: jax.Array,
) -> jax.Array:
    """Apply child conditions encoded by the Crossbow's outer selectors."""

    if internal_legal.shape != loadout.ability_mask.shape:
        raise ValueError("internal_legal and loadout ability axes must match")
    if internal_legal.shape[2] <= CROSSBOW_RELOAD_SLOT:
        return internal_legal
    ammo_below_maximum = (
        mechanics.resources[..., RESOURCE_AMMO] + jnp.float32(1.0e-6)
        < loadout.resource_maximum[..., RESOURCE_AMMO]
    )
    return condition_route_slot(
        internal_legal,
        loadout.weapon_family,
        _CROSSBOW_PRIMARY_ROUTE,
        CROSSBOW_RELOAD_SLOT,
        ammo_below_maximum,
    )


def resolve_requested_slot(
    loadout: AbilityLoadout,
    internal_legal: jax.Array,
    requested_slot: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Resolve a policy request to the child selected by the native root.

    Crossbow Primary prioritizes a live Big Arrow charge, otherwise a loaded
    standard shot, otherwise the reload root. The returned ``selectable`` bit
    rejects attempts to address internal children directly.
    """

    batch, entity_count, ability_capacity = internal_legal.shape
    if loadout.ability_mask.shape != internal_legal.shape:
        raise ValueError("internal_legal and loadout ability axes must match")
    if requested_slot.shape != (batch, entity_count):
        raise ValueError(
            "requested_slot must match the loadout batch/entity axes"
        )
    selectable_mask = policy_ability_mask(loadout)
    if ability_capacity <= CROSSBOW_RELOAD_SLOT:
        safe = jnp.clip(requested_slot, 0, ability_capacity - 1)
        selectable = _gather(selectable_mask, safe)
        return safe, selectable
    return resolve_priority_request(
        internal_legal,
        selectable_mask,
        requested_slot,
        loadout.weapon_family,
        _CROSSBOW_PRIMARY_ROUTE,
    )


def _gather(value: jax.Array, slot: jax.Array) -> jax.Array:
    return jnp.take_along_axis(value, slot[..., None], axis=2)[..., 0]


__all__ = [
    "CROSSBOW_BIG_ARROW_CHILD_SLOT",
    "CROSSBOW_COMBO_CHILD_SLOT",
    "CROSSBOW_PRIMARY_SLOT",
    "CROSSBOW_RELOAD_SLOT",
    "CROSSBOW_SIGNATURE_ACTIVATE_SLOT",
    "condition_internal_legality",
    "policy_ability_mask",
    "project_observable_active_slot",
    "project_legal_mask",
    "resolve_observed_execution_slot",
    "resolve_requested_slot",
]
