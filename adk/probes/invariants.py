"""Properties the environment must never violate, checked per step.

Every function here returns ``bool[batch]`` that is ``True`` **where the
invariant was violated**, so a healthy run is all-``False`` and a violation
count is a sum. All of them are traced and sync-free, so a whole audit runs
inside one compiled scan.

The bar for adding one: it must be a statement about the *environment*, true
for every scene and every policy, and checkable from the published observation
alone. "The agent should approach the target" is a strategy, not an invariant.

Some checks are deliberately narrow because a broad version would be wrong.
Padded slots hold arbitrary data by design, so "all entity distances are
non-negative" is only meaningful at occupied slots -- the mask is part of the
invariant, not a detail.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.types import AGENT_ENTITY

from adk.diagnostics.failures import any_failure
from adk.policy.actions import DODGE_ACTION_NAMES, DODGE_AUTHORED, action
from adk.policy.fields import field
from adk.policy.observations import observation_sets, observation_vectors


def _rows(flag: jax.Array) -> jax.Array:
    """Reduce any trailing axes so every check returns one bool per row."""

    array = jnp.asarray(flag)
    if array.ndim <= 1:
        return array
    return jnp.any(array, axis=tuple(range(1, array.ndim)))


def observation_is_finite(observation: Any, _state: Any = None) -> jax.Array:
    """No NaN or infinity anywhere in the float observation.

    The cheapest real bug detector there is: a NaN entering the observation
    silently poisons every gradient downstream, and nothing else reports it.
    """

    violated = None
    for values in observation_vectors(observation).values():
        bad = _rows(~jnp.isfinite(values))
        violated = bad if violated is None else (violated | bad)
    for tokens in observation_sets(observation).values():
        # Only occupied slots: padding may legitimately hold anything.
        finite = jnp.isfinite(tokens.values).all(axis=-1)
        bad = _rows(tokens.mask & ~finite)
        violated = bad if violated is None else (violated | bad)
    return violated


def health_within_range(observation: Any, _state: Any = None) -> jax.Array:
    """Health is a fraction, so it must stay inside [0, 1]."""

    health = field(observation.base.self_f32, "self_f32", "health_fraction")
    return (health < 0.0) | (health > 1.0)


def alive_matches_health(observation: Any, _state: Any = None) -> jax.Array:
    """An entity with positive health must not be flagged dead, or vice versa.

    Two independent columns describe the same fact, so they can disagree --
    and a policy reading the one the environment does not honour will act on a
    corpse.
    """

    health = field(observation.base.self_f32, "self_f32", "health_fraction")
    alive = field(observation.base.self_f32, "self_f32", "alive") > 0.5
    return alive != (health > 0.0)


def resources_within_range(observation: Any, _state: Any = None) -> jax.Array:
    """Available resources are normalized fractions; unavailable ones are zero.

    Both halves matter. A value outside [0, 1] means the normalization broke;
    a non-zero value at a masked channel means a stat leaked from a family that
    does not have it.
    """

    values = observation.resource_f32
    mask = observation.resource_mask
    out_of_range = mask & ((values < 0.0) | (values > 1.0))
    leaked = (~mask) & (values != 0.0)
    return _rows(out_of_range | leaked)


def some_action_is_legal(observation: Any, _state: Any = None) -> jax.Array:
    """At least one action must always be legal.

    The environment guarantees a safe fallback to ``idle``.  If this fires the
    policy is sampling from an empty distribution, and everything it produces
    downstream is undefined rather than merely bad.
    """

    return ~jnp.any(observation.base.action_mask, axis=-1)


def authored_dodges_only(observation: Any, _state: Any = None) -> jax.Array:
    """Dodge directions the assets never authored must never be offered.

    ``DODGE_AUTHORED_ACTION_MASK`` hard-wires forward and back off.  If this
    fires, either the asset changed or the mask stopped being applied -- and an
    agent would start spending logits on an action with no effect.
    """

    violated = None
    for name, authored in zip(DODGE_ACTION_NAMES, DODGE_AUTHORED):
        if authored:
            continue
        offered = action(observation.dodge_action_mask, "dodge_action_mask", name)
        violated = offered if violated is None else (violated | offered)
    return _rows(violated)


def occupied_distances_are_sane(observation: Any, _state: Any = None) -> jax.Array:
    """Distances at occupied entity slots must be finite and non-negative.

    Checked only where the mask says something is really there; padded slots
    are allowed to hold anything at all.
    """

    entities = observation_sets(observation)["entity"]
    distance = field(entities, "entity_f32", "distance")
    bad = (distance < 0.0) | ~jnp.isfinite(distance)
    return _rows(entities.mask & bad)


def no_environment_failure(observation: Any, _state: Any = None) -> jax.Array:
    """No failure bit is set.

    Deliberately excludes overflow, which is truncation rather than
    malfunction and fires by design in a crowded scene.
    """

    return any_failure(observation)


def guard_implies_stamina(observation: Any, _state: Any = None) -> jax.Array:
    """A guarding entity must not also be stamina-broken.

    Guard is gated on stamina upstream, so holding both at once means the gate
    was bypassed -- exactly the kind of state a random prober reaches and a
    trained policy never does.
    """

    defense = observation.defense_f32[:, AGENT_ENTITY]
    guarding = field(defense, "defense_f32", "guard_active") > 0.5
    broken = field(defense, "defense_f32", "stamina_broken") > 0.5
    return guarding & broken


#: Name -> check.  Every entry is traced and returns ``bool[batch]``.
INVARIANTS: Mapping[str, Callable[..., jax.Array]] = MappingProxyType(
    {
        "observation_is_finite": observation_is_finite,
        "health_within_range": health_within_range,
        "alive_matches_health": alive_matches_health,
        "resources_within_range": resources_within_range,
        "some_action_is_legal": some_action_is_legal,
        "authored_dodges_only": authored_dodges_only,
        "occupied_distances_are_sane": occupied_distances_are_sane,
        "no_environment_failure": no_environment_failure,
        "guard_implies_stamina": guard_implies_stamina,
    }
)


def check(observation: Any, state: Any = None) -> dict[str, jax.Array]:
    """Run every invariant.  Traced; ``True`` marks a violation."""

    return {name: rule(observation, state) for name, rule in INVARIANTS.items()}


def any_violation(observation: Any, state: Any = None) -> jax.Array:
    """One boolean per row: did anything at all go wrong?  Traced."""

    total = None
    for flag in check(observation, state).values():
        total = flag if total is None else (total | flag)
    return total


def invariant_names() -> tuple[str, ...]:
    """Every checkable invariant, in report order.  Pure Python."""

    return tuple(INVARIANTS)


__all__ = [
    "INVARIANTS",
    "alive_matches_health",
    "any_violation",
    "authored_dodges_only",
    "check",
    "guard_implies_stamina",
    "health_within_range",
    "invariant_names",
    "no_environment_failure",
    "observation_is_finite",
    "occupied_distances_are_sane",
    "resources_within_range",
    "some_action_is_legal",
]
