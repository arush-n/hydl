"""DAMAGE_RECEIVED read helper and ablation injector for the composition task.

``DAMAGE_RECEIVED`` is ``DecisionEvent`` index 2. ``adk.architecture.decision_context``
derives it only from legal observation: the learner's own health fraction falls
between two consecutive legal observations. The event is known only when both
observations are supplied (``_legal_perceptual_events``). A context has one bool
per event for ``values`` (active) and one for ``known``. An event can trigger a
manager decision only when it is both active and known (``manager_decision_due``,
``decision_context.py:415``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from adk.architecture.decision_context import DecisionContext, DecisionEvent

#: Column of DAMAGE_RECEIVED in ``DecisionEvents.values`` and ``.known``.
DAMAGE_EVENT_INDEX: int = int(DecisionEvent.DAMAGE_RECEIVED)


def _require_context(decision_context: object) -> None:
    """Raise TypeError unless ``decision_context`` is a DecisionContext."""

    if not isinstance(decision_context, DecisionContext):
        raise TypeError(
            "decision_context must be a DecisionContext, "
            f"got {type(decision_context).__name__}"
        )


def damage_received_active(decision_context: DecisionContext) -> jax.Array:
    """Return True for each lane where DAMAGE_RECEIVED is both active and known.

    Args:
        decision_context: A DecisionContext. Its ``events.values`` and
            ``events.known`` have shape bool[B, 14].

    Returns:
        bool[B]. Entry b is True when lane b has a known, active damage event in
        this decision.

    Raises:
        TypeError: If ``decision_context`` is not a DecisionContext.
    """

    _require_context(decision_context)
    values = jnp.asarray(decision_context.events.values)
    known = jnp.asarray(decision_context.events.known)
    return values[..., DAMAGE_EVENT_INDEX] & known[..., DAMAGE_EVENT_INDEX]


def inject_damage_event(decision_context: DecisionContext, lanes: jax.Array) -> DecisionContext:
    """ABLATION ONLY (ADR-001 D5). Set DAMAGE_RECEIVED active and known in some lanes.

    The flat pursuit arena closes combat for both actors, so the scripted
    opponent cannot lower the learner's health, and the environment does not
    produce this event there. The composition's evade phase is therefore measured
    with this injected event. A metric computed on a context that this function
    touched is an ablation result. It must be labelled with the injection and
    must never be reported as an environment outcome.

    Args:
        decision_context: The DecisionContext of one decision step.
        lanes: bool[B], True for each lane that receives the event. B is the
            context's batch size.

    Returns:
        A new DecisionContext. Only the DAMAGE_RECEIVED column changes. In the
        True lanes both ``values`` and ``known`` become True. Every other event,
        the timing and the lifecycle are unchanged.

    Raises:
        TypeError: If ``decision_context`` is not a DecisionContext, or ``lanes``
            is not a bool array.
        ValueError: If the context's event arrays are not [B, 14], or ``lanes``
            does not have shape (B,).
    """

    _require_context(decision_context)
    mask = jnp.asarray(lanes)
    if mask.dtype != jnp.bool_:
        raise TypeError(f"lanes must be a bool array, got dtype {mask.dtype}")
    values = jnp.asarray(decision_context.events.values)
    known = jnp.asarray(decision_context.events.known)
    if values.ndim != 2 or values.shape[1] != len(DecisionEvent):
        raise ValueError(
            f"events.values must have shape (B, {len(DecisionEvent)}), got {values.shape}"
        )
    if mask.shape != (values.shape[0],):
        raise ValueError(f"lanes must have shape ({values.shape[0]},), got {mask.shape}")
    events = decision_context.events._replace(
        values=values.at[:, DAMAGE_EVENT_INDEX].set(
            values[:, DAMAGE_EVENT_INDEX] | mask
        ),
        known=known.at[:, DAMAGE_EVENT_INDEX].set(
            known[:, DAMAGE_EVENT_INDEX] | mask
        ),
    )
    return decision_context._replace(events=events)


__all__ = [
    "DAMAGE_EVENT_INDEX",
    "damage_received_active",
    "inject_damage_event",
]
