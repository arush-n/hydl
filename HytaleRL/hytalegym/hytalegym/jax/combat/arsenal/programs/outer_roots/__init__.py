"""Asset-authored outer-root request routing for policy abilities."""

from hytalegym.jax.combat.arsenal.programs.outer_roots.crossbow import (
    CROSSBOW_BIG_ARROW_CHILD_SLOT,
    CROSSBOW_COMBO_CHILD_SLOT,
    CROSSBOW_PRIMARY_SLOT,
    CROSSBOW_RELOAD_SLOT,
    CROSSBOW_SIGNATURE_ACTIVATE_SLOT,
    condition_internal_legality,
    policy_ability_mask,
    project_legal_mask,
    project_observable_active_slot,
    resolve_observed_execution_slot,
    resolve_requested_slot,
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


__all__ = [
    "CROSSBOW_BIG_ARROW_CHILD_SLOT",
    "CROSSBOW_COMBO_CHILD_SLOT",
    "CROSSBOW_PRIMARY_SLOT",
    "CROSSBOW_RELOAD_SLOT",
    "CROSSBOW_SIGNATURE_ACTIVATE_SLOT",
    "PrioritySelectorRoute",
    "condition_internal_legality",
    "condition_route_slot",
    "policy_ability_mask",
    "project_legal_mask",
    "project_observable_active_slot",
    "project_observable_slot",
    "project_policy_mask",
    "project_root_legality",
    "resolve_observed_execution_slot",
    "resolve_priority_request",
    "resolve_requested_slot",
    "route_matches",
    "select_priority_child",
]
