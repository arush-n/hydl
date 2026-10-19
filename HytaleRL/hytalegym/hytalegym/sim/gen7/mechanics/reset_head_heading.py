"""Head yaw at reset (gen 7, FIDELITY section (ak), row 5g): an Arsenal reset given ``entity_yaw`` seeds both heads on their
body headings.

Switch ``Constructs.reset_head_heading``: ``none`` (gen 5: the reset zeroes ``agent_head_yaw`` and ``target_head_yaw``
whatever the body yaw is, ``combat/runtime/reset.py`` 156-161) or ``body_heading`` (gen 7).

Measured in section (ak) on the live 0.7.2 bridge (pin 460054A8, seeds 570057 and 913771): a target spawned facing the agent
(180 degrees for the fixture's target offset 0, 0, -2.5) has HeadRotation equal to its body yaw at tick 0 and keeps it on ticks
1 to 5 with no input. Two seeds, one spawn yaw. Inferred, not measured: the agent at a non-zero spawn yaw (no native reset sets
one; the agent case rests on the target measurement and on the pursuit precedent ``arena/training/runs/pursuit_run.py``
``_facing_reset``), and the tick-1 effect of the zeroed head in the simulator (the head window clamp), which no render run.

Hook: ``combat/arsenal/runtime.py`` ``reset_arsenal_batch`` (line 509 at HEAD; ``entity_yaw`` at 519). The original reads
``entity_yaw`` only on the entity-position path, so the wrapper acts only there: it sets the agent's and the target's head yaw
to their stored body yaw (pitch unchanged), then rebuilds the returned observation from that state. The engine functions come
from the rendered package that owns the wrapped function, at call time. On gen 7 ``observe_batch`` reads the patched perception
line of sight (``perception_detail_hitbox_los`` in ``world/perception/los.py``) when a geometry is given, which a HEAD import
would not. The one read from the wrapped module is ``combat_target_selection``, which that module defines below the reset.
"""

from __future__ import annotations

from functools import wraps
from importlib import import_module
from typing import Any

import jax.numpy as jnp

from hytalegym.jax.combat.types import AGENT_ENTITY, DEFAULT_ENTITY_COUNT, TARGET_ENTITY

SWITCH = "reset_head_heading"
IMPLEMENTED = True
NONE = "none"
BODY_HEADING = "body_heading"
#: The wrapped function lives in ``<package>.combat.arsenal.runtime``; its rendered siblings are ``<package>.<relative>``.
_OWNER_SUFFIX = ".combat.arsenal.runtime"


def _enabled(profile: Any) -> bool:
    value = profile.constructs.reset_head_heading
    if value == NONE:
        return False
    if value != BODY_HEADING:
        raise ValueError(f"unknown reset_head_heading switch {value!r}")
    return True


def _rendered(original: Any, relative: str) -> Any:
    """Module ``relative`` (for example ``combat.runtime.reset``) of the rendered package that owns ``original``."""

    return import_module(original.__module__[: -len(_OWNER_SUFFIX)] + "." + relative)


def _observation(original: Any, combat: Any, params: Any, config: Any, geometry: Any) -> Any:
    """The observation the original returns for ``combat``; a dynamic entity axis passes the engagement target."""

    observe = _rendered(original, "combat.runtime.reset").observe_batch
    if combat.health.shape[1] == DEFAULT_ENTITY_COUNT:
        return observe(combat, params, geometry)
    selection = _rendered(original, "combat.arsenal.runtime").combat_target_selection(combat, params, config)
    return observe(combat, params, geometry, target_entity_id=selection.engagement_target_id[:, AGENT_ENTITY])


def construct(original: Any, profile: Any) -> Any:
    """Return ``original`` for gen 5; for ``body_heading`` the reset with both heads on the stored body headings."""

    if not _enabled(profile):
        return original

    @wraps(original)
    def _reset_arsenal_batch(keys, params, config, geometry=None, *, entity_position=None, entity_yaw=None, **kwargs):
        result = original(
            keys, params, config, geometry, entity_position=entity_position, entity_yaw=entity_yaw, **kwargs
        )
        if entity_yaw is None or entity_position is None:
            return result
        state, observation = result
        combat = state.combat._replace(
            agent_head_yaw=jnp.asarray(state.combat.yaw[:, AGENT_ENTITY], dtype=jnp.float32),
            target_head_yaw=jnp.asarray(state.combat.yaw[:, TARGET_ENTITY], dtype=jnp.float32),
        )
        return state._replace(combat=combat), _observation(original, combat, params, config, geometry)

    return _reset_arsenal_batch
