"""Crossbow reset ammo (gen 7): the iron crossbow starts with no bolt loaded.

Switch ``Constructs.crossbow_reset_ammo``: ``loadout_six`` (gen 5: ``RESOURCE_AMMO`` starts at 6, the episode-loadout
convention of ``_iron_crossbow``) or ``native_zero`` (gen 7: starts at 0).

Measured on the live 0.7.2 bridge (FIDELITY-0.7.2.md section (ax); receipts ``HytaleRL/artifacts/fidelity_evidence/
gen7_probes/crossbow_standard_slot0_jar14.json`` and ``..._jar14_press2.json``, one ``Weapon_Arrow_Crude`` in hotbar slot 1):
the first Primary press at tick 55 finds Ammo 0; the outer Primary routes to the reload (``Root_Common_StatAmmoReload_Entry``),
which takes one arrow and raises Ammo 0 -> 1 at tick 91 (+36 ticks). The second press at tick 105 fires with Ammo 1 -> 0.

The sim already routes a Primary press to the reload while Ammo is below its maximum (``programs/outer_roots/crossbow.py``),
and the reload's first ``RESOURCE_AMMO`` step lands 34 ticks after its start (``programs/timing.py``). Only the reset value
differs from the live server, so this construct changes the crossbow's initial ``RESOURCE_AMMO`` and nothing else. The sim
does not model arrows (``inventory_from_loadout`` seeds the equipped weapon alone), so one press loads one bolt by the
reload's own single step, and an empty hotbar is not represented.

Hook: ``hytale_0_5_7_loadouts`` (jax/combat/arsenal/profiles/hytale_0_5_7.py): the agent's initial ``RESOURCE_AMMO`` is set to 0
for the rows whose agent profile is ``iron_crossbow``. No state leaf.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import jax.numpy as jnp

SWITCH = "crossbow_reset_ammo"
IMPLEMENTED = True
LOADOUT_SIX = "loadout_six"
NATIVE_ZERO = "native_zero"
CROSSBOW_PROFILE = "iron_crossbow"
#: RESOURCE_AMMO (combat/mechanics/schema/contract.py) and AGENT_ENTITY (combat/types.py). Pinned against the engine
#: source by tests/sim/test_timing_constructs.py, so the wrapper needs no lookup in the wrapped module's globals.
RESOURCE_AMMO = 5
AGENT_ENTITY = 0


def _enabled(profile: Any) -> bool:
    value = profile.constructs.crossbow_reset_ammo
    if value == LOADOUT_SIX:
        return False
    if value != NATIVE_ZERO:
        raise ValueError(f"unknown crossbow_reset_ammo switch {value!r}")
    return True


def construct(original: Any, profile: Any) -> Any:
    """Return ``original`` for gen 5; for ``native_zero`` the loadout builder with the crossbow's reset ammo at zero."""

    if not _enabled(profile):
        return original

    @wraps(original)
    def _loadouts(agent_profiles: Any, *, target_profiles: Any = None) -> Any:
        loadout = original(agent_profiles, target_profiles=target_profiles)
        names = [agent_profiles] if isinstance(agent_profiles, str) else list(agent_profiles)
        crossbow = jnp.asarray([name == CROSSBOW_PROFILE for name in names])
        initial = loadout.resource_initial
        ammo_initial = jnp.where(crossbow, jnp.float32(0.0), initial[:, AGENT_ENTITY, RESOURCE_AMMO])
        return loadout._replace(
            resource_initial=initial.at[:, AGENT_ENTITY, RESOURCE_AMMO].set(ammo_initial),
        )

    return _loadouts
