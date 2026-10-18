"""Shortbow reset ammo (gen 7, row 5f): the shortbow starts with no ammo loaded.

Switch ``Constructs.shortbow_reset_ammo``: ``loadout_one`` (gen 5: ``RESOURCE_AMMO`` starts at 1, the episode-loadout
convention of ``_iron_shortbow``) or ``native_zero`` (gen 7).

Measured in FIDELITY-0.7.2.md section (af) on the live 0.7.2 bridge, seeds 570057 and 913771: native ammo (the
``RESOURCE_AMMO`` feature) is 0 on the reset row and on tick 1, and 1 from tick 2, the tick the shot is accepted. The gen-5
charging phase already has ``value=1.0`` from ``start_tick=2`` and ``inactive_value=0.0`` before it, so it gives 0 on tick 1
and 1 from tick 2 without change. Only the reset value differs. The stock differential (two seeds) shows that single
mismatch, on the reset row.

Hook: ``hytale_0_5_7_loadouts`` (combat/arsenal/profiles/hytale_0_5_7.py): the agent's initial ``RESOURCE_AMMO`` is set to 0
for the rows whose agent profile is ``iron_shortbow``. No state leaf.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import jax.numpy as jnp

SWITCH = "shortbow_reset_ammo"
IMPLEMENTED = True
LOADOUT_ONE = "loadout_one"
NATIVE_ZERO = "native_zero"
SHORTBOW_PROFILE = "iron_shortbow"
#: RESOURCE_AMMO (combat/mechanics/schema/contract.py) and AGENT_ENTITY (combat/types.py). Pinned against the engine
#: source by tests/sim/test_timing_constructs.py, so the wrapper needs no lookup in the wrapped module's globals.
RESOURCE_AMMO = 5
AGENT_ENTITY = 0


def _enabled(profile: Any) -> bool:
    value = profile.constructs.shortbow_reset_ammo
    if value == LOADOUT_ONE:
        return False
    if value != NATIVE_ZERO:
        raise ValueError(f"unknown shortbow_reset_ammo switch {value!r}")
    return True


def construct(original: Any, profile: Any) -> Any:
    """Return ``original`` for gen 5; for ``native_zero`` the loadout builder with the shortbow's reset ammo at zero."""

    if not _enabled(profile):
        return original

    @wraps(original)
    def _loadouts(agent_profiles: Any, *, target_profiles: Any = None) -> Any:
        loadout = original(agent_profiles, target_profiles=target_profiles)
        names = [agent_profiles] if isinstance(agent_profiles, str) else list(agent_profiles)
        shortbow = jnp.asarray([name == SHORTBOW_PROFILE for name in names])
        initial = loadout.resource_initial
        ammo_initial = jnp.where(shortbow, jnp.float32(0.0), initial[:, AGENT_ENTITY, RESOURCE_AMMO])
        return loadout._replace(
            resource_initial=initial.at[:, AGENT_ENTITY, RESOURCE_AMMO].set(ammo_initial),
        )

    return _loadouts
